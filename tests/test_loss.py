"""
Test per src/losses/loss.py.

Esegui con:
    pytest tests/test_loss.py -v
"""
import pytest
import torch

from src.losses import CombinedLoss


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def pred_target():
    """Coppia (pred, target) casuali in [0, 1], shape (2, 3, 64, 64)."""
    torch.manual_seed(0)
    pred   = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    return pred, target


# ---------------------------------------------------------------------------
# Costruttore
# ---------------------------------------------------------------------------

def test_default_alpha():
    criterion = CombinedLoss()
    assert criterion.alpha == 0.8


def test_custom_alpha():
    criterion = CombinedLoss(alpha=0.5)
    assert criterion.alpha == 0.5


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        CombinedLoss(alpha=1.5)

    with pytest.raises(ValueError, match="alpha"):
        CombinedLoss(alpha=-0.1)


# ---------------------------------------------------------------------------
# Forward — tipo e shape
# ---------------------------------------------------------------------------

def test_forward_returns_scalar(pred_target):
    pred, target = pred_target
    loss = CombinedLoss()(pred, target)
    assert loss.ndim == 0


def test_forward_dtype_float32(pred_target):
    pred, target = pred_target
    loss = CombinedLoss()(pred, target)
    assert loss.dtype == torch.float32


def test_forward_positive(pred_target):
    """La loss deve essere >= 0 per qualsiasi coppia di input."""
    pred, target = pred_target
    loss = CombinedLoss()(pred, target)
    assert loss.item() >= 0.0


def test_forward_zero_on_identical():
    """Loss = 0 quando pred == target (identità perfetta)."""
    x = torch.rand(2, 3, 64, 64)
    loss = CombinedLoss()(x, x)
    assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# Forward — comportamento rispetto ad alpha
# ---------------------------------------------------------------------------

def test_alpha_1_equals_pure_l1(pred_target):
    """alpha=1 → solo L1."""
    pred, target = pred_target
    loss_combined = CombinedLoss(alpha=1.0)(pred, target).item()
    loss_l1 = torch.nn.L1Loss()(pred, target).item()
    assert abs(loss_combined - loss_l1) < 1e-6


def test_alpha_0_equals_pure_ssim(pred_target):
    """alpha=0 → solo SSIMLoss (1 - SSIM)."""
    import piq
    pred, target = pred_target
    loss_combined = CombinedLoss(alpha=0.0)(pred, target).item()
    loss_ssim     = piq.SSIMLoss(data_range=1.0)(pred, target).item()
    assert abs(loss_combined - loss_ssim) < 1e-5


def test_higher_alpha_closer_to_l1(pred_target):
    """Con alpha più alto la loss deve avvicinarsi a L1."""
    pred, target = pred_target
    l1_val   = torch.nn.L1Loss()(pred, target).item()
    loss_08  = CombinedLoss(alpha=0.8)(pred, target).item()
    loss_02  = CombinedLoss(alpha=0.2)(pred, target).item()
    assert abs(loss_08 - l1_val) < abs(loss_02 - l1_val)


# ---------------------------------------------------------------------------
# components()
# ---------------------------------------------------------------------------

def test_components_keys(pred_target):
    pred, target = pred_target
    comps = CombinedLoss().components(pred, target)
    assert set(comps.keys()) == {"l1", "ssim_loss", "combined"}


def test_components_consistent_with_forward(pred_target):
    """components['combined'] deve coincidere con forward()."""
    pred, target = pred_target
    criterion = CombinedLoss(alpha=0.8)
    loss      = criterion(pred, target)
    comps     = criterion.components(pred, target)
    assert abs(comps["combined"].item() - loss.item()) < 1e-6


def test_components_no_grad(pred_target):
    """components() usa no_grad — nessun tensore deve richiedere grad."""
    pred, target = pred_target
    pred.requires_grad_(True)
    comps = CombinedLoss().components(pred, target)
    for v in comps.values():
        assert not v.requires_grad


# ---------------------------------------------------------------------------
# Backprop
# ---------------------------------------------------------------------------

def test_backward_computes_grad(pred_target):
    """Il gradiente deve fluire attraverso la loss fino a pred."""
    pred, target = pred_target
    pred = pred.detach().requires_grad_(True)
    loss = CombinedLoss()(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_contains_alpha():
    criterion = CombinedLoss(alpha=0.7)
    assert "alpha=0.7" in repr(criterion)
