"""
Test per src/models/unet_attention.py e src/models/cbam.py.

Esegui con:
    pytest tests/test_unet_attention.py -v
"""
import pytest
import torch

from src.models import UNetAttention
from src.models.cbam import ChannelAttention, SpatialAttention, CBAM


BATCH  = 2
H, W   = 256, 256
BASE_C = 32


# ---------------------------------------------------------------------------
# CBAM — ChannelAttention
# ---------------------------------------------------------------------------

def test_channel_attention_output_shape():
    block = ChannelAttention(in_channels=64)
    x = torch.rand(2, 64, 32, 32)
    out = block(x)
    assert out.shape == (2, 64, 1, 1)


def test_channel_attention_range():
    block = ChannelAttention(in_channels=64)
    x = torch.rand(2, 64, 32, 32)
    out = block(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_channel_attention_small_channels():
    """Canali < ratio — non deve sollevare errori (hidden = max(1, C//ratio))."""
    block = ChannelAttention(in_channels=4, ratio=16)
    x = torch.rand(1, 4, 8, 8)
    out = block(x)
    assert out.shape == (1, 4, 1, 1)


# ---------------------------------------------------------------------------
# CBAM — SpatialAttention
# ---------------------------------------------------------------------------

def test_spatial_attention_output_shape():
    block = SpatialAttention(kernel_size=7)
    x = torch.rand(2, 64, 32, 32)
    out = block(x)
    assert out.shape == (2, 1, 32, 32)


def test_spatial_attention_range():
    block = SpatialAttention(kernel_size=7)
    x = torch.rand(2, 64, 32, 32)
    out = block(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_spatial_attention_preserves_spatial_dims():
    block = SpatialAttention(kernel_size=7)
    x = torch.rand(1, 128, 64, 48)
    out = block(x)
    assert out.shape == (1, 1, 64, 48)


# ---------------------------------------------------------------------------
# CBAM — modulo completo
# ---------------------------------------------------------------------------

def test_cbam_preserves_shape():
    block = CBAM(in_channels=64)
    x = torch.rand(2, 64, 32, 32)
    assert block(x).shape == x.shape


def test_cbam_output_differs_from_input():
    """CBAM deve modificare le feature map (non identità)."""
    torch.manual_seed(0)
    block = CBAM(in_channels=64)
    x = torch.rand(1, 64, 16, 16)
    with torch.no_grad():
        out = block(x)
    assert not torch.allclose(out, x)


def test_cbam_various_channels():
    for C in [32, 64, 128, 256]:
        block = CBAM(in_channels=C)
        x = torch.rand(1, C, 16, 16)
        assert block(x).shape == (1, C, 16, 16)


# ---------------------------------------------------------------------------
# UNetAttention — shape e range
# ---------------------------------------------------------------------------

def test_attention_output_shape():
    model = UNetAttention(base_channels=BASE_C)
    x = torch.rand(BATCH, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (BATCH, 3, H, W)


def test_attention_output_range():
    model = UNetAttention(base_channels=BASE_C)
    x = torch.rand(BATCH, 3, H, W)
    with torch.no_grad():
        y = model(x)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_attention_output_dtype():
    model = UNetAttention(base_channels=BASE_C)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.dtype == torch.float32


def test_attention_non_square_input():
    model = UNetAttention(base_channels=BASE_C)
    x = torch.rand(1, 3, 192, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 192, 256)


def test_attention_bilinear_variant():
    model = UNetAttention(base_channels=BASE_C, bilinear=True)
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 256, 256)
    assert y.min() >= 0.0 and y.max() <= 1.0


# ---------------------------------------------------------------------------
# UNetAttention — parametri
# ---------------------------------------------------------------------------

def test_attention_more_params_than_baseline():
    """UNetAttention deve avere più parametri della baseline."""
    from src.models import UNetBaseline
    baseline = UNetBaseline(base_channels=BASE_C)
    attention = UNetAttention(base_channels=BASE_C)
    n_base = sum(p.numel() for p in baseline.parameters())
    n_att  = sum(p.numel() for p in attention.parameters())
    assert n_att > n_base


def test_attention_overhead_small():
    """L'overhead CBAM deve essere < 5% dei parametri baseline."""
    from src.models import UNetBaseline
    baseline  = UNetBaseline(base_channels=BASE_C)
    attention = UNetAttention(base_channels=BASE_C)
    n_base = sum(p.numel() for p in baseline.parameters())
    n_att  = sum(p.numel() for p in attention.parameters())
    overhead_pct = (n_att - n_base) / n_base
    assert overhead_pct < 0.05, f"Overhead troppo alto: {overhead_pct:.1%}"


def test_attention_repr_contains_key_info():
    model = UNetAttention(base_channels=BASE_C)
    r = repr(model)
    assert "base_channels=32" in r
    assert "params=" in r


# ---------------------------------------------------------------------------
# UNetAttention — backprop
# ---------------------------------------------------------------------------

def test_attention_backward():
    model = UNetAttention(base_channels=BASE_C)
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
    y.mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


# ---------------------------------------------------------------------------
# Confronto baseline vs attention — output diverso
# ---------------------------------------------------------------------------

def test_attention_output_differs_from_baseline():
    """Con gli stessi pesi iniziali casuali i due modelli danno output diversi."""
    from src.models import UNetBaseline
    torch.manual_seed(42)
    baseline  = UNetBaseline(base_channels=BASE_C)
    torch.manual_seed(42)
    attention = UNetAttention(base_channels=BASE_C)
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y_base = baseline(x)
        y_att  = attention(x)
    assert not torch.allclose(y_base, y_att)
