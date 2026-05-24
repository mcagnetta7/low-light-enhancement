"""
Test E03 — NIQE metric.

I test richiedono pyiqa e il file niqe_modelparameters.mat in cache.
Se pyiqa non è installato o la cache non è disponibile, i test vengono
saltati automaticamente con pytest.importorskip.
"""

import math
import pytest
import torch

pyiqa = pytest.importorskip("pyiqa", reason="pyiqa non installato")

from src.metrics import NIQE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def metric():
    """Singola istanza condivisa tra i test — crea il modello una volta sola."""
    return NIQE(device="cpu")


@pytest.fixture
def natural_batch():
    """Batch di immagini con texture naturale (gradiente smooth)."""
    B = 2
    imgs = []
    for _ in range(B):
        base = torch.linspace(0.2, 0.8, 256).unsqueeze(0).expand(256, 256)
        noise = 0.02 * torch.randn(256, 256)
        img = (base + noise).clamp(0, 1).unsqueeze(0).expand(3, -1, -1)
        imgs.append(img)
    return torch.stack(imgs)  # (2, 3, 256, 256)


@pytest.fixture
def noise_batch():
    """Batch di rumore puro — NIQE alto (qualità peggiore)."""
    return torch.rand(2, 3, 256, 256)


@pytest.fixture
def fp16_batch():
    """Tensori fp16 come escono dall'autocast AMP."""
    return torch.rand(2, 3, 256, 256).half()


# ---------------------------------------------------------------------------
# T01 — costruttore, repr e attributi
# ---------------------------------------------------------------------------

class TestInit:
    def test_lower_better_is_true(self):
        assert NIQE.lower_better is True

    def test_default_device_is_cpu(self):
        m = NIQE()
        assert m.device == torch.device("cpu")

    def test_custom_device_string(self):
        m = NIQE(device="cpu")
        assert m.device == torch.device("cpu")

    def test_lazy_init_metric_is_none(self):
        """Il modello NON viene caricato al costruttore."""
        m = NIQE()
        assert m._metric is None

    def test_repr(self):
        r = repr(NIQE())
        assert "NIQE"         in r
        assert "lower_better" in r


# ---------------------------------------------------------------------------
# T02 — __call__: tipo e proprietà
# ---------------------------------------------------------------------------

class TestCall:
    def test_returns_float(self, metric, noise_batch):
        result = metric(noise_batch)
        assert isinstance(result, float)

    def test_positive(self, metric, noise_batch):
        assert metric(noise_batch) > 0.0

    def test_finite(self, metric, noise_batch):
        assert math.isfinite(metric(noise_batch))

    def test_accepts_fp16(self, metric, fp16_batch):
        """Tensori fp16 devono essere accettati senza errori."""
        result = metric(fp16_batch)
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_metric_loaded_after_call(self, noise_batch):
        """Dopo il primo __call__ il modello deve essere caricato."""
        m = NIQE()
        assert m._metric is None
        m(noise_batch)
        assert m._metric is not None


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
            assert math.isfinite(v), f"NIQE non finito: {v}"

    def test_all_positive(self, metric, noise_batch):
        for v in metric.per_image(noise_batch):
            assert v > 0.0

    def test_mean_consistent_with_call(self, metric, noise_batch):
        """La media di per_image deve coincidere con __call__."""
        per_img   = metric.per_image(noise_batch)
        mean_per  = sum(per_img) / len(per_img)
        mean_call = metric(noise_batch)
        assert abs(mean_per - mean_call) < 1e-4

    def test_per_image_fp16(self, metric, fp16_batch):
        result = metric.per_image(fp16_batch)
        assert len(result) == 2
        assert all(math.isfinite(v) for v in result)

    def test_single_image(self, metric):
        img = torch.rand(1, 3, 256, 256)
        result = metric.per_image(img)
        assert len(result) == 1
        assert math.isfinite(result[0])
