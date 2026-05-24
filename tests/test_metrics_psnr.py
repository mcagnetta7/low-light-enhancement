"""
Test E01 — PSNR metric.
"""

import math
import pytest
import torch

from src.metrics import PSNR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metric():
    return PSNR(data_range=1.0)


@pytest.fixture
def identical_batch():
    """Output identico al target → PSNR infinito (nessun errore)."""
    t = torch.rand(4, 3, 64, 64)
    return t, t.clone()


@pytest.fixture
def noisy_batch():
    """Output leggermente rumoroso."""
    target = torch.rand(4, 3, 64, 64)
    output = (target + 0.05 * torch.randn_like(target)).clamp(0.0, 1.0)
    return output, target


@pytest.fixture
def fp16_batch():
    """Tensori fp16 come escono dall'autocast AMP."""
    t = torch.rand(2, 3, 64, 64).half()
    return t, t.clone()


# ---------------------------------------------------------------------------
# T01 — costruttore e repr
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_data_range(self):
        m = PSNR()
        assert m.data_range == 1.0

    def test_custom_data_range(self):
        m = PSNR(data_range=255.0)
        assert m.data_range == 255.0

    def test_repr(self):
        assert "PSNR" in repr(PSNR())
        assert "1.0" in repr(PSNR())


# ---------------------------------------------------------------------------
# T02 — __call__: tipo e range
# ---------------------------------------------------------------------------

class TestCall:
    def test_returns_float(self, metric, noisy_batch):
        out, tgt = noisy_batch
        result = metric(out, tgt)
        assert isinstance(result, float)

    def test_positive(self, metric, noisy_batch):
        out, tgt = noisy_batch
        assert metric(out, tgt) > 0.0

    def test_finite(self, metric, noisy_batch):
        out, tgt = noisy_batch
        assert math.isfinite(metric(out, tgt))

    def test_identical_images_high_psnr(self, metric, identical_batch):
        """Immagini identiche → PSNR molto alto (>60 dB in pratica)."""
        out, tgt = identical_batch
        score = metric(out, tgt)
        assert score > 60.0

    def test_lower_psnr_for_higher_noise(self, metric):
        """Più rumore → PSNR più basso."""
        target = torch.rand(4, 3, 64, 64)
        low_noise  = (target + 0.01 * torch.randn_like(target)).clamp(0, 1)
        high_noise = (target + 0.20 * torch.randn_like(target)).clamp(0, 1)
        assert metric(low_noise, target) > metric(high_noise, target)

    def test_accepts_fp16(self, metric, fp16_batch):
        """Tensori fp16 devono essere accettati senza errori (fix AMP)."""
        out, tgt = fp16_batch
        result = metric(out, tgt)
        assert isinstance(result, float)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# T03 — per_image
# ---------------------------------------------------------------------------

class TestPerImage:
    def test_returns_list(self, metric, noisy_batch):
        out, tgt = noisy_batch
        result = metric.per_image(out, tgt)
        assert isinstance(result, list)

    def test_length_equals_batch_size(self, metric, noisy_batch):
        out, tgt = noisy_batch
        result = metric.per_image(out, tgt)
        assert len(result) == out.shape[0]

    def test_all_finite(self, metric, noisy_batch):
        out, tgt = noisy_batch
        for v in metric.per_image(out, tgt):
            assert math.isfinite(v), f"PSNR non finito: {v}"

    def test_all_positive(self, metric, noisy_batch):
        out, tgt = noisy_batch
        for v in metric.per_image(out, tgt):
            assert v > 0.0

    def test_mean_consistent_with_call(self, metric, noisy_batch):
        """La media di per_image deve approssimare il valore di __call__."""
        out, tgt = noisy_batch
        per_img = metric.per_image(out, tgt)
        mean_per_img = sum(per_img) / len(per_img)
        mean_call    = metric(out, tgt)
        # La media di psnr per-immagine NON è identica alla psnr sul batch
        # (piq batch calcola mse medio poi converte), ma devono essere vicini.
        assert abs(mean_per_img - mean_call) < 3.0  # tolleranza 3 dB

    def test_per_image_fp16(self, metric, fp16_batch):
        out, tgt = fp16_batch
        result = metric.per_image(out, tgt)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_single_image_batch(self, metric):
        t = torch.rand(1, 3, 64, 64)
        result = metric.per_image(t, t.clone())
        assert len(result) == 1
        assert result[0] > 60.0
