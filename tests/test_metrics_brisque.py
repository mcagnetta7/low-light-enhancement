"""
Test E04 — BRISQUE metric.
"""

import math
import pytest
import torch

from src.metrics import BRISQUE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def metric():
    return BRISQUE()


@pytest.fixture
def noise_batch():
    return torch.rand(4, 3, 256, 256)


@pytest.fixture
def fp16_batch():
    return torch.rand(2, 3, 256, 256).half()


# ---------------------------------------------------------------------------
# T01 — costruttore e repr
# ---------------------------------------------------------------------------

class TestInit:
    def test_lower_better_is_true(self):
        assert BRISQUE.lower_better is True

    def test_default_params(self):
        m = BRISQUE()
        assert m.data_range  == 1.0
        assert m.kernel_size == 7

    def test_custom_data_range(self):
        m = BRISQUE(data_range=255.0)
        assert m.data_range == 255.0

    def test_repr(self):
        r = repr(BRISQUE())
        assert "BRISQUE"      in r
        assert "kernel_size"  in r
        assert "kernel_sigma" in r


# ---------------------------------------------------------------------------
# T02 — __call__: tipo e range
# ---------------------------------------------------------------------------

class TestCall:
    def test_returns_float(self, metric, noise_batch):
        result = metric(noise_batch)
        assert isinstance(result, float)

    def test_finite(self, metric, noise_batch):
        assert math.isfinite(metric(noise_batch))

    def test_non_negative(self, metric, noise_batch):
        assert metric(noise_batch) >= 0.0

    def test_accepts_fp16(self, metric, fp16_batch):
        """Tensori fp16 devono essere accettati senza errori."""
        result = metric(fp16_batch)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_single_image(self, metric):
        img = torch.rand(1, 3, 256, 256)
        result = metric(img)
        assert isinstance(result, float)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# T03 — per_image
# ---------------------------------------------------------------------------

class TestPerImage:
    def test_returns_list(self, metric, noise_batch):
        assert isinstance(metric.per_image(noise_batch), list)

    def test_length_equals_batch_size(self, metric, noise_batch):
        result = metric.per_image(noise_batch)
        assert len(result) == noise_batch.shape[0]

    def test_all_finite(self, metric, noise_batch):
        for v in metric.per_image(noise_batch):
            assert math.isfinite(v), f"BRISQUE non finito: {v}"

    def test_all_non_negative(self, metric, noise_batch):
        for v in metric.per_image(noise_batch):
            assert v >= 0.0

    def test_mean_consistent_with_call(self, metric, noise_batch):
        """La media di per_image deve coincidere con __call__."""
        per_img   = metric.per_image(noise_batch)
        mean_per  = sum(per_img) / len(per_img)
        mean_call = metric(noise_batch)
        assert abs(mean_per - mean_call) < 1e-3

    def test_per_image_fp16(self, metric, fp16_batch):
        result = metric.per_image(fp16_batch)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_single_image_per_image(self, metric):
        img = torch.rand(1, 3, 256, 256)
        result = metric.per_image(img)
        assert len(result) == 1
        assert math.isfinite(result[0])
