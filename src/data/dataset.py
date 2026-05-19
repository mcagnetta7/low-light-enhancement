from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

# Estensioni immagine supportate
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


class PairedImageDataset(Dataset):
    """
    Dataset PyTorch per coppie di immagini (bassa luminosità / normale luminosità).

    Accoppia le immagini tramite una chiave derivata dallo stem del file.
    Per default la chiave coincide con lo stem (es. LOL-v2 Synthetic, LOL-v1).
    Per dataset dove low e normal hanno prefissi diversi (es. LOL-v2 Real:
    "low00001" ↔ "normal00001") passare key_fn=lambda s: s.lstrip("abcdefghijklmnopqrstuvwxyz")
    per estrarre la parte numerica comune.

    Pipeline per il training (con augmentation):
        dataset = PairedImageDataset(
            low_dir="data/LOL-v2/Real_captured/Train/Low",
            normal_dir="data/LOL-v2/Real_captured/Train/Normal",
            key_fn=lambda s: s.lstrip("abcdefghijklmnopqrstuvwxyz"),
            paired_transform=get_paired_augmentation(256),
            transform=get_preprocessing_transform(256),
        )

    Pipeline per validation e test (solo preprocessing deterministico):
        dataset = PairedImageDataset(
            low_dir="data/LOL-v2/Real_captured/Test/Low",
            normal_dir="data/LOL-v2/Real_captured/Test/Normal",
            key_fn=lambda s: s.lstrip("abcdefghijklmnopqrstuvwxyz"),
            transform=get_preprocessing_transform(256),
        )
    """

    def __init__(
        self,
        low_dir: str | Path,
        normal_dir: str | Path,
        key_fn: Optional[Callable[[str], str]] = None,
        paired_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        strict: bool = True,
    ) -> None:
        """
        Args:
            low_dir:          percorso alla cartella delle immagini a bassa luminosità.
            normal_dir:       percorso alla cartella delle immagini a normale luminosità.
            key_fn:           funzione stem → chiave di pairing. Default: identità (usa lo stem
                              intero). Usare per dataset dove low e normal hanno prefissi diversi,
                              es. LOL-v2 Real ("low00001" ↔ "normal00001"):
                              key_fn=lambda s: s.lstrip("abcdefghijklmnopqrstuvwxyz")
            paired_transform: trasformazione accoppiata applicata all'intero sample dict
                              prima di `transform`. Riceve e restituisce un dict con chiavi
                              "low", "normal", "stem", "filename" (valori PIL Image).
                              Usare per augmentations geometriche casuali (flip, crop) che
                              devono essere identiche per low e normal.
                              Solo per il training — non usare su validation e test.
            transform:        trasformazione deterministica applicata separatamente a
                              sample["low"] e sample["normal"] dopo paired_transform.
                              Tipicamente get_preprocessing_transform(size): Resize + ToTensor.
                              Applicata su tutti gli split (train, val, test).
            strict:           se True (default) solleva ValueError su qualsiasi mismatch.
                              Se False stampa un avviso e salta le coppie incomplete.
                              Usare strict=False per dataset con imperfezioni note (es. LOL-v1
                              ha 2 immagini low senza corrispondente high).
        """
        self.low_dir = Path(low_dir)
        self.normal_dir = Path(normal_dir)
        self.key_fn = key_fn if key_fn is not None else lambda s: s
        self.paired_transform = paired_transform
        self.transform = transform

        low_map = self._collect_images(self.low_dir)
        normal_map = self._collect_images(self.normal_dir)

        # Costruisce {chiave: path} applicando key_fn a ogni stem
        low_keyed    = {self.key_fn(stem): path for stem, path in low_map.items()}
        normal_keyed = {self.key_fn(stem): path for stem, path in normal_map.items()}

        only_in_low    = sorted(low_keyed.keys() - normal_keyed.keys())
        only_in_normal = sorted(normal_keyed.keys() - low_keyed.keys())

        if only_in_low or only_in_normal:
            msg = "Nomi file non corrispondenti tra le due cartelle.\n"
            if only_in_low:
                msg += f"  Solo in '{self.low_dir}': {only_in_low[:10]}"
                if len(only_in_low) > 10:
                    msg += f" ... (+{len(only_in_low) - 10} altri)"
                msg += "\n"
            if only_in_normal:
                msg += f"  Solo in '{self.normal_dir}': {only_in_normal[:10]}"
                if len(only_in_normal) > 10:
                    msg += f" ... (+{len(only_in_normal) - 10} altri)"
            if strict:
                raise ValueError(msg)
            print(f"AVVISO: {msg.strip()}\n  Le coppie incomplete vengono saltate.")
            # Mantieni solo le chiavi presenti in entrambe le cartelle
            for k in only_in_low:
                del low_keyed[k]
            for k in only_in_normal:
                del normal_keyed[k]

        # Coppie ordinate per chiave per garantire ordine deterministico
        keys = sorted(low_keyed.keys())
        self.stems        = keys
        self.low_paths    = [low_keyed[k]    for k in keys]
        self.normal_paths = [normal_keyed[k] for k in keys]

    def _collect_images(self, directory: Path) -> dict[str, Path]:
        # Restituisce un dizionario {stem: path} per tutte le immagini valide.
        # Usare lo stem come chiave permette di accoppiare per nome file
        # indipendentemente dall'estensione (.png vs .jpg).
        if not directory.exists():
            raise FileNotFoundError(f"Cartella non trovata: '{directory}'.")

        seen: dict[str, Path] = {}
        duplicates: list[str] = []
        for p in directory.iterdir():
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if p.stem in seen:
                duplicates.append(f"'{seen[p.stem].name}' e '{p.name}'")
            else:
                seen[p.stem] = p

        if duplicates:
            raise ValueError(
                f"Stem duplicati in '{directory}': {duplicates}. "
                "Rinomina o rimuovi i file con lo stesso nome base."
            )

        if not seen:
            raise FileNotFoundError(f"Nessuna immagine trovata in '{directory}'.")
        return seen

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        """
        Restituisce un dizionario con:
            "low"      : immagine a bassa luminosità (tensore o PIL Image)
            "normal"   : immagine a normale luminosità / ground truth
            "stem"     : nome base del file senza estensione
            "filename" : nome file originale con estensione

        Ordine di applicazione delle trasformazioni:
            1. paired_transform(sample)          — augmentation geometrica accoppiata (PIL → PIL)
            2. transform(sample["low"])           — preprocessing deterministico (PIL → Tensor)
               transform(sample["normal"])
        """
        low = Image.open(self.low_paths[idx]).convert("RGB")
        normal = Image.open(self.normal_paths[idx]).convert("RGB")

        sample = {
            "low": low,
            "normal": normal,
            "stem": self.stems[idx],
            "filename": self.low_paths[idx].name,
        }

        if self.paired_transform is not None:
            sample = self.paired_transform(sample)

        if self.transform is not None:
            sample["low"] = self.transform(sample["low"])
            sample["normal"] = self.transform(sample["normal"])

        return sample

    def __repr__(self) -> str:
        return (
            f"PairedImageDataset("
            f"low='{self.low_dir}', "
            f"normal='{self.normal_dir}', "
            f"n={len(self)})"
        )
