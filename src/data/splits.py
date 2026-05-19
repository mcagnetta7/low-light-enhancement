import random
from pathlib import Path

# Valori di default coerenti con src/utils/reproducibility.py
SEED = 42
VAL_FRACTION = 0.1  # 10% degli stem di training usati per validazione


def make_split(
    stems: list[str],
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> tuple[list[str], list[str]]:
    """
    Divide una lista di stem in (train, val) in modo deterministico.

    Gli stem vengono prima ordinati (per neutralizzare qualsiasi ordine
    del filesystem), poi mescolati con seed fisso. Questo garantisce che
    lo split sia identico su qualsiasi macchina e in qualsiasi sessione.

    Args:
        stems:        lista di nomi file senza estensione (da PairedImageDataset.stems).
        val_fraction: frazione degli stem da riservare alla validazione.
                      Default 0.1 → ~10% val, ~90% train.
        seed:         seed per il mescolamento casuale. Default 42.

    Returns:
        (train_stems, val_stems): due liste disgiunte che coprono tutti gli stem.

    Raises:
        ValueError: se val_fraction non è in (0, 1) o se stems è vuoto.
    """
    if not stems:
        raise ValueError("La lista di stem è vuota.")
    if not (0 < val_fraction < 1):
        raise ValueError(f"val_fraction deve essere in (0, 1), ricevuto: {val_fraction}.")

    shuffled = sorted(stems)               # ordine deterministico prima del shuffle
    rng = random.Random(seed)              # generatore isolato — non altera lo stato globale
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_fraction))
    val_stems = sorted(shuffled[:n_val])   # riordina per leggibilità nei file salvati
    train_stems = sorted(shuffled[n_val:])

    return train_stems, val_stems


def save_split(stems: list[str], path: str | Path) -> None:
    """
    Salva una lista di stem in un file di testo (uno per riga, ordine alfabetico).

    Il file risultante è human-readable e versionabile con git
    (la cartella data/splits/ può essere committata, a differenza di data/).

    Args:
        stems: lista di stem da salvare.
        path:  percorso del file di output (la directory viene creata se necessario).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(stems)) + "\n", encoding="utf-8")


def load_split(path: str | Path) -> list[str]:
    """
    Carica una lista di stem da un file di testo salvato con save_split.

    Args:
        path: percorso del file di split.

    Returns:
        Lista di stem ordinata alfabeticamente.

    Raises:
        FileNotFoundError: se il file non esiste.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File di split non trovato: '{path}'.")
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(stems)


def save_test_split(stems: list[str], path: str | Path) -> None:
    """
    Salva gli stem della cartella Test ufficiale senza alcun mescolamento.

    LOL-v2 (e LOL-v1) forniscono una cartella Test separata definita dagli
    autori del dataset. Questi stem NON devono mai essere mescolati con quelli
    di training o validazione: farlo invaliderebbe la valutazione in-domain
    e renderebbe i risultati non confrontabili con la letteratura.

    Questa funzione è distinta da save_split per rendere esplicito nel codice
    che il test set è immutabile e di provenienza ufficiale.

    Args:
        stems: lista di stem della cartella Test ufficiale.
        path:  percorso del file di output.
    """
    # Gli stem vengono solo ordinati alfabeticamente, mai mescolati
    save_split(stems, path)


def create_and_save_train_val_split(
    train_stems: list[str],
    splits_dir: str | Path,
    split_name: str,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """
    Crea gli split train/val dalla cartella Train ufficiale e li salva su disco.

    Questa funzione opera SOLO sugli stem della cartella Train ufficiale del
    dataset. La cartella Test ufficiale rimane separata e va gestita con
    save_test_split — i due insiemi non devono mai sovrapporsi.

    File generati:
        {splits_dir}/{split_name}_train.txt
        {splits_dir}/{split_name}_val.txt

    Args:
        train_stems:  stem della cartella Train ufficiale (da PairedImageDataset.stems).
        splits_dir:   cartella in cui salvare i file di split.
        split_name:   prefisso del nome file, es. "lolv2_real" o "lolv2_synth".
        val_fraction: frazione da riservare alla validazione. Default 0.1.
        seed:         seed per la riproducibilità. Default 42.

    Returns:
        Dizionario {"train": [...], "val": [...]} con gli stem di ciascun split.

    Esempio:
        from src.data.dataset import PairedImageDataset
        from src.data.splits import create_and_save_train_val_split, save_test_split

        train_ds = PairedImageDataset("data/LOL-v2/Real_captured/Train/Low",
                                      "data/LOL-v2/Real_captured/Train/Normal")
        test_ds  = PairedImageDataset("data/LOL-v2/Real_captured/Test/Low",
                                      "data/LOL-v2/Real_captured/Test/Normal")

        result = create_and_save_train_val_split(
            train_ds.stems, "data/splits", "lolv2_real"
        )
        save_test_split(test_ds.stems, "data/splits/lolv2_real_test.txt")
    """
    split_stems = make_split(train_stems, val_fraction=val_fraction, seed=seed)
    train, val = split_stems

    splits_dir = Path(splits_dir)
    save_split(train, splits_dir / f"{split_name}_train.txt")
    save_split(val,   splits_dir / f"{split_name}_val.txt")

    print(f"Split '{split_name}' salvato in '{splits_dir}':")
    print(f"  train : {len(train)} stem")
    print(f"  val   : {len(val)} stem  ({val_fraction:.0%} del totale Train ufficiale)")

    return {"train": train, "val": val}
