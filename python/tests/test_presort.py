import numpy as np
import pytest

from deglib.optimization import presort
from deglib.distances import Metric


@pytest.mark.parametrize("threads", [0, 1, 2])
@pytest.mark.parametrize("metric", [Metric.FP32_L2, Metric.FP32_InnerProduct, "FP32_L2"])
def test_presort_basic(threads, metric):
    np.random.seed(42)
    N, d = 50, 16
    data = np.random.randn(N, d).astype(np.float32)

    indices = presort(data, metric=metric, radius_decay=0.9, threads=threads)

    assert isinstance(indices, np.ndarray)
    assert indices.shape == (N,)
    assert indices.dtype == np.uint32

    # Verify that returned indices form a valid permutation of [0..N-1]
    assert sorted(indices.tolist()) == list(range(N))


def test_presort_empty():
    data = np.empty((0, 16), dtype=np.float32)
    indices = presort(data)
    assert len(indices) == 0


def test_presort_callback():
    np.random.seed(42)
    N, d = 50, 16
    data = np.random.randn(N, d).astype(np.float32)

    progress_history = []
    def my_callback(progress: float) -> bool:
        progress_history.append(progress)
        return False

    indices = presort(data, callback=my_callback)
    assert len(indices) == N
    assert len(progress_history) > 0
    assert progress_history[-1] == 1.0
