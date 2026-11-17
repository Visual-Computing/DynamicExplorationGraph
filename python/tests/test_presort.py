import numpy as np
import pytest

from deglib.distances import Metric
from deglib.optimization import presort


@pytest.mark.parametrize("metric", [Metric.FP32_L2, Metric.FP32_InnerProduct, "FP32_L2"])
@pytest.mark.parametrize("threads", [0, 1, 2])
def test_presort_basic(metric, threads):
    rng = np.random.default_rng(42)
    n, d = 100, 16
    data = rng.standard_normal((n, d)).astype(np.float32)

    indices = presort(data, metric=metric, radius_decay=0.9, threads=threads)

    assert isinstance(indices, np.ndarray)
    assert indices.shape == (n,)
    assert indices.dtype == np.uint32
    assert sorted(indices.tolist()) == list(range(n))


def test_presort_with_bound_instruction():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 16)).astype(np.float32)
    indices = presort(data, metric=Metric.FP32_L2.with_instruction("Scalar"), radius_decay=0.9, threads=2)
    assert indices.shape == (100,)
    assert indices.dtype == np.uint32
    assert sorted(indices.tolist()) == list(range(100))


def test_presort_empty_array():
    data = np.empty((0, 16), dtype=np.float32)
    indices = presort(data)
    assert len(indices) == 0
    assert indices.dtype == np.uint32


def test_presort_callback():
    rng = np.random.default_rng(42)
    n, d = 100, 16
    data = rng.standard_normal((n, d)).astype(np.float32)

    progress_history = []

    def my_callback(p: float):
        progress_history.append(p)
        return False

    indices = presort(data, callback=my_callback)
    assert len(indices) == n
    assert len(progress_history) > 0
    assert progress_history[-1] == 1.0


@pytest.mark.parametrize(
    "invalid_metric",
    [
        Metric.Uint8_L2,
        Metric.Uint8_InnerProduct,
        Metric.FP16_L2,
        Metric.FP16_InnerProduct,
        Metric.EVP_InnerProduct,
    ],
)
def test_presort_invalid_metric_raises(invalid_metric):
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 16)).astype(np.float32)
    with pytest.raises(ValueError, match="FP32_L2 and FP32_InnerProduct"):
        presort(data, metric=invalid_metric)


def test_presort_invalid_type_raises():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 16)).astype(np.float32)
    with pytest.raises(ValueError):
        presort(data, metric="invalid_string")
