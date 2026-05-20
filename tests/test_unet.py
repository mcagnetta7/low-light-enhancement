"""
Test per src/models/unet.py e src/models/unet_parts.py.

Esegui con:
    pytest tests/test_unet.py -v
"""
import pytest
import torch

from src.models import UNetBaseline
from src.models.unet_parts import DoubleConv, Down, Up, OutConv


# ---------------------------------------------------------------------------
# Costanti condivise
# ---------------------------------------------------------------------------

BATCH  = 2
H, W   = 256, 256
BASE_C = 32


# ---------------------------------------------------------------------------
# UNetBaseline — shape e range output
# ---------------------------------------------------------------------------

def test_output_shape_default():
    model = UNetBaseline(base_channels=BASE_C)
    x = torch.rand(BATCH, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (BATCH, 3, H, W)


def test_output_range_in_01():
    model = UNetBaseline(base_channels=BASE_C)
    x = torch.rand(BATCH, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_output_dtype_float32():
    model = UNetBaseline(base_channels=BASE_C)
    x = torch.rand(BATCH, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.dtype == torch.float32


def test_output_shape_bilinear():
    model = UNetBaseline(base_channels=BASE_C, bilinear=True)
    x = torch.rand(BATCH, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (BATCH, 3, H, W)


def test_output_range_bilinear():
    model = UNetBaseline(base_channels=BASE_C, bilinear=True)
    x = torch.rand(BATCH, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_output_shape_base64():
    """base_channels=64 — equivalente alla UNet originale."""
    model = UNetBaseline(base_channels=64)
    x = torch.rand(1, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, H, W)


def test_output_shape_non_square():
    """Dimensioni non quadrate — il padding dinamico deve gestirle."""
    model = UNetBaseline(base_channels=BASE_C)
    x = torch.rand(1, 3, 192, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 192, 256)


def test_output_shape_not_power_of_two():
    """Dimensioni non potenza di 2 — il padding dinamico in Up deve reggere."""
    model = UNetBaseline(base_channels=BASE_C)
    x = torch.rand(1, 3, 240, 320)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 240, 320)


# ---------------------------------------------------------------------------
# UNetBaseline — parametri e attributi
# ---------------------------------------------------------------------------

def test_param_count_base32():
    """Con base_channels=32 il modello deve avere meno di 10 M parametri."""
    model = UNetBaseline(base_channels=BASE_C)
    n = sum(p.numel() for p in model.parameters())
    assert n < 10_000_000, f"Troppi parametri: {n:,}"


def test_param_count_base64_larger_than_base32():
    model32 = UNetBaseline(base_channels=32)
    model64 = UNetBaseline(base_channels=64)
    n32 = sum(p.numel() for p in model32.parameters())
    n64 = sum(p.numel() for p in model64.parameters())
    assert n64 > n32


def test_repr_contains_key_info():
    model = UNetBaseline(base_channels=BASE_C)
    r = repr(model)
    assert "base_channels=32" in r
    assert "params=" in r


# ---------------------------------------------------------------------------
# UNetBaseline — gradiente (backprop)
# ---------------------------------------------------------------------------

def test_backward_pass():
    """Il gradiente deve fluire fino all'input senza errori."""
    model = UNetBaseline(base_channels=BASE_C)
    x = torch.rand(1, 3, 256, 256, requires_grad=False)
    y = model(x)
    loss = y.mean()
    loss.backward()
    # Verifica che almeno un parametro abbia gradiente non-None
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# Blocchi (unet_parts)
# ---------------------------------------------------------------------------

def test_double_conv_shape():
    block = DoubleConv(3, 32)
    x = torch.rand(1, 3, 64, 64)
    assert block(x).shape == (1, 32, 64, 64)


def test_down_halves_spatial():
    block = Down(32, 64)
    x = torch.rand(1, 32, 64, 64)
    y = block(x)
    assert y.shape == (1, 64, 32, 32)


def test_up_restores_spatial():
    block = Up(64, 32, bilinear=False)
    x1 = torch.rand(1, 64, 16, 16)   # dal livello inferiore
    x2 = torch.rand(1, 32, 32, 32)   # skip connection
    y = block(x1, x2)
    assert y.shape == (1, 32, 32, 32)


def test_up_bilinear_restores_spatial():
    block = Up(64, 32, bilinear=True)
    x1 = torch.rand(1, 32, 16, 16)   # bottleneck dimezzato con bilinear
    x2 = torch.rand(1, 32, 32, 32)
    y = block(x1, x2)
    assert y.shape == (1, 32, 32, 32)


def test_outconv_shape():
    block = OutConv(32, 3)
    x = torch.rand(1, 32, 64, 64)
    assert block(x).shape == (1, 3, 64, 64)
