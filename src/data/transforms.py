import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# Risoluzioni previste dallo spec (§7)
SIZE_DEFAULT = 256
SIZE_FALLBACK = 192


def get_preprocessing_transform(size: int = SIZE_DEFAULT) -> transforms.Compose:
    """
    Pipeline di preprocessing deterministica per immagini paired.

    Operazioni applicate in ordine:
      1. Resize(size, size) — porta tutte le immagini a una risoluzione uniforme.
         Usa interpolazione bilineare (default di torchvision), che preserva
         meglio i dettagli cromatici rispetto a nearest-neighbor.
      2. ToTensor() — converte PIL Image [0, 255] in tensore float [0.0, 1.0]
         e riordina gli assi da HWC a CHW richiesto da PyTorch.

    Perché non si applica Normalize(mean, std)?
        Per image restoration/enhancement l'output del modello deve tornare
        nello stesso spazio dell'input. Se si normalizza con media/std
        (es. ImageNet), bisogna denormalizzare l'output prima di calcolare
        le metriche (PSNR, SSIM) e prima del confronto con il ground truth.
        Mantenere i valori in [0, 1] elimina questo overhead e semplifica
        il calcolo della loss L1/SSIM direttamente sui tensori.

    Questa trasformazione è deterministica: può essere passata direttamente
    a PairedImageDataset(transform=...) senza rischi di disallineamento
    tra low e normal, perché Resize e ToTensor producono lo stesso risultato
    indipendentemente dall'ordine di applicazione.

    Args:
        size: lato del quadrato di output in pixel.
              Usare SIZE_DEFAULT (256) oppure SIZE_FALLBACK (192) se la VRAM
              non è sufficiente per batch size >= 8 a 256×256.

    Returns:
        transforms.Compose pronto per essere passato a PairedImageDataset.

    Esempio:
        from src.data.transforms import get_preprocessing_transform, SIZE_FALLBACK
        transform = get_preprocessing_transform(SIZE_FALLBACK)
        dataset = PairedImageDataset(low_dir, normal_dir, transform=transform)
    """
    if size <= 0:
        raise ValueError(f"size deve essere positivo, ricevuto: {size}")

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


class PairedAugmentation:
    """
    Augmentation casuale per coppie di immagini (low / normal).

    Il problema fondamentale dell'augmentation paired è che flip e crop
    geometrici devono essere identici per le due immagini: applicarli
    separatamente (come farebbe transforms.Compose) produrrebbe un low e
    un normal geometricamente disallineati, invalidando la loss pixel-wise.

    Soluzione: API funzionale di torchvision (TF.*). I parametri casuali
    vengono campionati UNA volta e applicati esplicitamente a entrambe le
    immagini. ColorJitter viene invece applicato SOLO all'immagine low
    perché modificare il ground truth cambierebbe il target di supervisione.

    Input/output: opera sul dict restituito da PairedImageDataset.__getitem__
    (chiavi "low", "normal", "stem", "filename") con valori PIL Image.
    Va applicata PRIMA di get_preprocessing_transform (che converte in tensore).

    Pipeline consigliata per il training:
        augmentation  = get_paired_augmentation(size=256)
        preprocessing = get_preprocessing_transform(size=256)

        dataset = PairedImageDataset(
            low_dir=low_dir,
            normal_dir=normal_dir,
            paired_transform=get_paired_augmentation(size=256),
            transform=get_preprocessing_transform(size=256),
        )
    
    Per validation/test:
        dataset = PairedImageDataset(
            low_dir=low_dir,
            normal_dir=normal_dir,
            transform=get_preprocessing_transform(size=256),
        )

    Per validation e test NON applicare augmentation, solo preprocessing.
    """

    def __init__(
        self,
        size: int = SIZE_DEFAULT,
        pad: int = 16,
        hflip_p: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.05,
    ) -> None:
        """
        Args:
            size:       lato del crop di output in pixel. Deve corrispondere
                        alla dimensione usata in get_preprocessing_transform.
            pad:        pixel di padding (reflect) aggiunti prima del crop.
                        Permette random crop su immagini già ridimensionate a
                        `size` senza perdere bordi utili. Default 16.
            hflip_p:    probabilità di flip orizzontale. Default 0.5.
            brightness: intensità della variazione di luminosità (solo low).
            contrast:   intensità della variazione di contrasto (solo low).
            saturation: intensità della variazione di saturazione (solo low).
                        Mantenuto basso (0.05) per non alterare troppo il
                        bilanciamento del bianco rispetto al ground truth.
        """
        if size <= 0:
            raise ValueError(f"size deve essere positivo, ricevuto: {size}")
        if pad < 0:
            raise ValueError(f"pad deve essere >= 0, ricevuto: {pad}")

        self.size = size
        self.pad = pad
        self.hflip_p = hflip_p
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=0.0,  # hue=0: non cambia il colore dominante, solo l'intensità
        )

    def __call__(self, sample: dict) -> dict:
        """
        Args:
            sample: dict con chiavi "low", "normal" (PIL Image) e metadati.

        Returns:
            Stesso dict con "low" e "normal" augmentati. I metadati
            ("stem", "filename") vengono copiati inalterati.
        """
        low: Image.Image = sample["low"]
        normal: Image.Image = sample["normal"]

        # 1. Random horizontal flip — stessa decisione per entrambe le immagini.
        #    torch.rand è usato invece di random.random per rispettare il seed
        #    globale impostato da set_seed() in src/utils/reproducibility.py.
        if torch.rand(1).item() < self.hflip_p:
            low = TF.hflip(low)
            normal = TF.hflip(normal)

        # 2. Pad + random crop — i parametri (i, j, h, w) vengono campionati
        #    una sola volta e applicati identicamente a low e normal.
        if self.pad > 0:
            low = TF.pad(low, self.pad, padding_mode="reflect")
            normal = TF.pad(normal, self.pad, padding_mode="reflect")

        i, j, h, w = transforms.RandomCrop.get_params(low, (self.size, self.size))
        low = TF.crop(low, i, j, h, w)
        normal = TF.crop(normal, i, j, h, w)

        # 3. ColorJitter solo su low — simula variabilità della degradazione
        #    senza modificare il ground truth. Applicare jitter al normal
        #    cambierebbe il target di supervisione, corrompendo la loss L1/SSIM.
        low = self.color_jitter(low)

        return {**sample, "low": low, "normal": normal}

    def __repr__(self) -> str:
        return (
            f"PairedAugmentation(size={self.size}, pad={self.pad}, "
            f"hflip_p={self.hflip_p}, jitter={self.color_jitter})"
        )


def get_paired_augmentation(
    size: int = SIZE_DEFAULT,
    pad: int = 16,
    hflip_p: float = 0.5,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.05,
) -> PairedAugmentation:
    """
    Factory per PairedAugmentation con i parametri consigliati per il progetto.

    I valori di default sono calibrati per immagini LOL-v2 a 256×256:
    - pad=16: crop range di ±16px per lato, abbastanza per variabilità senza
      introdurre artefatti ai bordi
    - jitter leggero: il low è già degradato, non serve aggiungere molta variazione

    Args:
        size:       lato del crop di output. Usare SIZE_DEFAULT o SIZE_FALLBACK.
        pad:        padding prima del crop (default 16).
        hflip_p:    probabilità flip orizzontale (default 0.5).
        brightness: variazione di luminosità su low (default 0.1).
        contrast:   variazione di contrasto su low (default 0.1).
        saturation: variazione di saturazione su low (default 0.05).

    Returns:
        Istanza di PairedAugmentation pronta all'uso nel training loop.
    """
    return PairedAugmentation(
        size=size,
        pad=pad,
        hflip_p=hflip_p,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
    )
