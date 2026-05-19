"""
Test per src/data/splits.py.

Esegui con:
    pytest tests/test_split.py -v
"""
import tempfile
from pathlib import Path

import pytest

from src.data.splits import (
    make_split,
    save_split,
    load_split,
    create_and_save_train_val_split,
    save_test_split,
)

STEMS = [f"img_{i:03d}" for i in range(20)]


# ---------------------------------------------------------------------------
# make_split
# ---------------------------------------------------------------------------

def test_make_split_sizes():
    train, val = make_split(STEMS, val_fraction=0.1, seed=42)
    assert len(train) == 18
    assert len(val) == 2


def test_make_split_covers_all_stems():
    train, val = make_split(STEMS, val_fraction=0.1, seed=42)
    assert sorted(train + val) == sorted(STEMS)


def test_make_split_disjoint():
    train, val = make_split(STEMS, val_fraction=0.1, seed=42)
    assert set(train).isdisjoint(set(val))


def test_make_split_deterministic():
    train1, val1 = make_split(STEMS, seed=42)
    train2, val2 = make_split(STEMS, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_make_split_different_seeds_differ():
    train1, _ = make_split(STEMS, seed=42)
    train2, _ = make_split(STEMS, seed=99)
    assert train1 != train2


def test_make_split_empty_raises():
    with pytest.raises(ValueError, match="vuota"):
        make_split([])


def test_make_split_invalid_fraction_raises():
    with pytest.raises(ValueError, match="val_fraction"):
        make_split(STEMS, val_fraction=1.5)


# ---------------------------------------------------------------------------
# save_split / load_split
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.txt"
        save_split(STEMS, path)
        loaded = load_split(path)
        assert loaded == sorted(STEMS)


def test_save_creates_parent_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "dir" / "split.txt"
        save_split(STEMS, path)
        assert path.exists()


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="non trovato"):
        load_split(Path("non_esiste.txt"))


# ---------------------------------------------------------------------------
# save_test_split
# ---------------------------------------------------------------------------

def test_save_test_split_preserves_stems():
    test_stems = ["test_001", "test_002", "test_003"]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.txt"
        save_test_split(test_stems, path)
        loaded = load_split(path)
        assert loaded == sorted(test_stems)


def test_test_split_disjoint_from_train_val():
    # Simula il caso reale: train/val da Train/, test da Test/
    train_stems = [f"train_{i:03d}" for i in range(10)]
    test_stems  = [f"test_{i:03d}"  for i in range(5)]
    train, val = make_split(train_stems, seed=42)
    assert set(test_stems).isdisjoint(set(train))
    assert set(test_stems).isdisjoint(set(val))


# ---------------------------------------------------------------------------
# create_and_save_train_val_split
# ---------------------------------------------------------------------------

def test_create_and_save_produces_files():
    with tempfile.TemporaryDirectory() as tmp:
        create_and_save_train_val_split(STEMS, tmp, "lolv2_real")
        assert (Path(tmp) / "lolv2_real_train.txt").exists()
        assert (Path(tmp) / "lolv2_real_val.txt").exists()


def test_create_and_save_loaded_matches_returned():
    with tempfile.TemporaryDirectory() as tmp:
        result = create_and_save_train_val_split(STEMS, tmp, "lolv2_real", seed=42)
        loaded_train = load_split(Path(tmp) / "lolv2_real_train.txt")
        loaded_val   = load_split(Path(tmp) / "lolv2_real_val.txt")
        assert loaded_train == result["train"]
        assert loaded_val   == result["val"]
