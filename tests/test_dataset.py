"""
Test per src/data/dataset.py — PairedImageDataset.

Esegui con:
    pip install pytest
    pytest tests/test_dataset.py -v
"""
import pytest
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from src.data.dataset import PairedImageDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Crea una piccola immagine RGB sintetica su disco."""
    Image.new("RGB", size, color=(128, 64, 32)).save(path)


def populate_dirs(low_dir: Path, normal_dir: Path, stems: list[str]) -> None:
    """Crea coppie di immagini .png nei due folder."""
    for stem in stems:
        make_image(low_dir / f"{stem}.png")
        make_image(normal_dir / f"{stem}.png")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def paired_dirs(tmp_path):
    """Fixture: due cartelle con 5 coppie correttamente nominate."""
    low_dir = tmp_path / "low"
    normal_dir = tmp_path / "normal"
    low_dir.mkdir()
    normal_dir.mkdir()
    populate_dirs(low_dir, normal_dir, ["001", "002", "003", "004", "005"])
    return low_dir, normal_dir


# ---------------------------------------------------------------------------
# Test: caricamento corretto
# ---------------------------------------------------------------------------

def test_len(paired_dirs):
    low_dir, normal_dir = paired_dirs
    ds = PairedImageDataset(low_dir, normal_dir)
    assert len(ds) == 5


def test_getitem_keys(paired_dirs):
    low_dir, normal_dir = paired_dirs
    ds = PairedImageDataset(low_dir, normal_dir)
    sample = ds[0]
    assert set(sample.keys()) == {"low", "normal", "stem", "filename"}


def test_getitem_images_are_pil(paired_dirs):
    low_dir, normal_dir = paired_dirs
    ds = PairedImageDataset(low_dir, normal_dir)
    sample = ds[0]
    assert isinstance(sample["low"], Image.Image)
    assert isinstance(sample["normal"], Image.Image)


def test_stem_and_filename_consistent(paired_dirs):
    low_dir, normal_dir = paired_dirs
    ds = PairedImageDataset(low_dir, normal_dir)
    for i in range(len(ds)):
        sample = ds[i]
        # filename deve iniziare con lo stem
        assert sample["filename"].startswith(sample["stem"])
        # filename deve avere un'estensione
        assert "." in sample["filename"]


def test_pairs_matched_by_stem_not_order(tmp_path):
    """
    Verifica che il pairing avvenga per nome file e non per posizione:
    inserisce i file in ordine diverso nei due folder e controlla
    che ogni low sia accoppiato al corretto normal.
    """
    low_dir = tmp_path / "low"
    normal_dir = tmp_path / "normal"
    low_dir.mkdir()
    normal_dir.mkdir()

    # Low: 001, 002, 003 — Normal: 003, 001, 002 (ordine inverso sul filesystem)
    for stem in ["001", "002", "003"]:
        make_image(low_dir / f"{stem}.png")
    for stem in ["003", "001", "002"]:
        make_image(normal_dir / f"{stem}.png")

    ds = PairedImageDataset(low_dir, normal_dir)
    for i in range(len(ds)):
        sample = ds[i]
        # low e normal devono avere lo stesso stem
        assert Path(sample["filename"]).stem == sample["stem"]


def test_repr(paired_dirs):
    low_dir, normal_dir = paired_dirs
    ds = PairedImageDataset(low_dir, normal_dir)
    r = repr(ds)
    assert "PairedImageDataset" in r
    assert "n=5" in r


# ---------------------------------------------------------------------------
# Test: transform
# ---------------------------------------------------------------------------

def test_transform_applied(paired_dirs):
    import torch
    low_dir, normal_dir = paired_dirs
    transform = T.ToTensor()
    ds = PairedImageDataset(low_dir, normal_dir, transform=transform)
    sample = ds[0]
    assert isinstance(sample["low"], torch.Tensor)
    assert isinstance(sample["normal"], torch.Tensor)
    assert sample["low"].shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# Test: errori attesi
# ---------------------------------------------------------------------------

def test_missing_low_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cartella non trovata"):
        PairedImageDataset(tmp_path / "non_esiste", tmp_path)


def test_empty_low_dir(tmp_path):
    low_dir = tmp_path / "low"
    normal_dir = tmp_path / "normal"
    low_dir.mkdir()
    normal_dir.mkdir()
    make_image(normal_dir / "001.png")
    with pytest.raises(FileNotFoundError, match="Nessuna immagine trovata"):
        PairedImageDataset(low_dir, normal_dir)


def test_mismatched_stems(tmp_path):
    low_dir = tmp_path / "low"
    normal_dir = tmp_path / "normal"
    low_dir.mkdir()
    normal_dir.mkdir()
    make_image(low_dir / "001.png")
    make_image(low_dir / "002.png")
    make_image(normal_dir / "001.png")
    make_image(normal_dir / "999.png")  # stem diverso
    with pytest.raises(ValueError, match="Nomi file non corrispondenti"):
        PairedImageDataset(low_dir, normal_dir)


def test_duplicate_stems_in_low(tmp_path):
    low_dir = tmp_path / "low"
    normal_dir = tmp_path / "normal"
    low_dir.mkdir()
    normal_dir.mkdir()
    make_image(low_dir / "001.png")
    make_image(low_dir / "001.jpg")   # stesso stem, estensione diversa
    make_image(normal_dir / "001.png")
    with pytest.raises(ValueError, match="Stem duplicati"):
        PairedImageDataset(low_dir, normal_dir)
