"""
Test E02 — SSIM metric.
"""

import math
import pytest
import torch

from src.metrics import SSIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metric():
    return SSIM(data_range=1.0)


@pytest.fixture
def identical_batch():
    t = torch.rand(4, 3, 64, 64)
    return t, t.clone()


@pytest.fixture
def noisy_batch():
    target = torch.rand(4, 3, 64, 64)
    output = (target + 0.05 * torch.randn_like(target)).clamp(0.0, 1.0)
    return output, target


@pytest.fixture
def fp16_batch():
    t = torch.rand(2, 3, 64, 64).half()
    return t, t.clone()


# ---------------------------------------------------------------------------
# T01 — costruttore e repr
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_params(self):
        m = SSIM()
        assert m.data_range   == 1.0
        assert m.kernel_size  == 11
        assert m.kernel_sigma == 1.5

    def test_custom_data_range(self):
        m = SSIM(data_range=255.0)
        assert m.data_range == 255.0

    def test_repr(self):
        r = repr(SSIM())
        assert "SSIM"         in r
        assert "1.0"          in r
        assert "11"           in r
        assert "kernel_sigma" in r


# ---------------------------------------------------------------------------
# T02 — __call__: tipo e range
# ---------------------------------------------------------------------------

class TestCall:
    def test_returns_float(self, metric, noisy_batch):
        out, tgt = noisy_batch
        result = metric(out, tgt)
        assert isinstance(result, float)

    def test_in_range_zero_one(self, metric, noisy_batch):
        out, tgt = noisy_batch
        result = metric(out, tgt)
        assert 0.0 <= result <= 1.0

    def test_finite(self, metric, noisy_batch):
        out, tgt = noisy_batch
        assert math.isfinite(metric(out, tgt))

    def test_identical_images_near_one(self, metric, identical_batch):
        """Immagini identiche → SSIM ≈ 1."""
        out, tgt = identical_batch
        assert metric(out, tgt) > 0.99

    def test_lower_ssim_for_higher_noise(self, metric):
        """Più rumore → SSIM più basso."""
        target     = torch.rand(4, 3, 64, 64)
        low_noise  = (target + 0.01 * torch.randn_like(target)).clamp(0, 1)
        high_noise = (target + 0.30 * torch.randn_like(target)).clamp(0, 1)
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
        assert isinstance(metric.per_image(out, tgt), list)

    def test_length_equals_batch_size(self, metric, noisy_batch):
        out, tgt = noisy_batch
        assert len(metric.per_image(out, tgt)) == out.shape[0]

    def test_all_in_range(self, metric, noisy_batch):
        out, tgt = noisy_batch
        for v in metric.per_image(out, tgt):
            assert 0.0 <= v <= 1.0, f"SSIM fuori range: {v}"

    def test_all_finite(self, metric, noisy_batch):
        out, tgt = noisy_batch
        for v in metric.per_image(out, tgt):
            assert math.isfinite(v), f"SSIM non finito: {v}"

    def test_mean_consistent_with_call(self, metric, noisy_batch):
        """La media di per_image deve essere vicina al valore di __call__."""
        out, tgt = noisy_batch
        per_img   = metric.per_image(out, tgt)
        mean_per  = sum(per_img) / len(per_img)
        mean_call = metric(out, tgt)
        assert abs(mean_per - mean_call) < 0.05  # tolleranza 5%

    def test_per_image_fp16(self, metric, fp16_batch):
        out, tgt = fp16_batch
        result = metric.per_image(out, tgt)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_single_image_batch(self, metric):
        t = torch.rand(1, 3, 64, 64)
        result = metric.per_image(t, t.clone())
        assert len(result) == 1
        assert result[0] > 0.99
