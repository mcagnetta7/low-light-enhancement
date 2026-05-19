"""
Test per src/data/transforms.py — get_preprocessing_transform.

Esegui con:
    pytest tests/test_transforms.py -v
"""
import pytest
import torch
from PIL import Image

from src.data.transforms import get_preprocessing_transform, SIZE_DEFAULT, SIZE_FALLBACK


def make_image(size: tuple[int, int] = (640, 480)) -> Image.Image:
    return Image.new("RGB", size, color=(20, 30, 40))


# ---------------------------------------------------------------------------
# Test: output corretto
# ---------------------------------------------------------------------------

def test_output_is_tensor():
    out = get_preprocessing_transform(SIZE_DEFAULT)(make_image())
    assert isinstance(out, torch.Tensor)


def test_output_shape_default():
    out = get_preprocessing_transform(SIZE_DEFAULT)(make_image())
    assert out.shape == (3, SIZE_DEFAULT, SIZE_DEFAULT)


def test_output_shape_fallback():
    out = get_preprocessing_transform(SIZE_FALLBACK)(make_image())
    assert out.shape == (3, SIZE_FALLBACK, SIZE_FALLBACK)


def test_output_range():
    out = get_preprocessing_transform(SIZE_DEFAULT)(make_image())
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_output_shape_custom_size():
    out = get_preprocessing_transform(128)(make_image())
    assert out.shape == (3, 128, 128)


def test_works_on_non_square_input():
    # Immagine non quadrata deve essere ridimensionata correttamente
    out = get_preprocessing_transform(SIZE_DEFAULT)(make_image((1920, 1080)))
    assert out.shape == (3, SIZE_DEFAULT, SIZE_DEFAULT)


# ---------------------------------------------------------------------------
# Test: errori attesi
# ---------------------------------------------------------------------------

def test_invalid_size_zero():
    with pytest.raises(ValueError, match="size deve essere positivo"):
        get_preprocessing_transform(0)


def test_invalid_size_negative():
    with pytest.raises(ValueError, match="size deve essere positivo"):
        get_preprocessing_transform(-64)
