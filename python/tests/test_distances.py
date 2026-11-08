import numpy as np
import pytest

from deglib import FloatSpace, Metric
from deglib.transformation import make_scalar_quantizer_int8


def test_compute_distances_batch_shapes():
    np.random.seed(1)
    n, q, d = 200, 7, 32
    base = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    queries = np.ascontiguousarray(base[:q])
    space = FloatSpace.create(d, Metric.FP32_L2)

    # 2D batch -> [Q, N]
    dists = space.compute_distances(queries, base)
    assert dists.shape == (q, n)

    # 1D single query -> [1, N]
    single = space.compute_distances(queries[0], base)
    assert single.shape == (1, n)
    np.testing.assert_allclose(single[0], dists[0], rtol=1e-6)


def test_compute_distances_threads_match_serial():
    np.random.seed(2)
    n, q, d = 500, 10, 64
    base = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    queries = np.ascontiguousarray(base[:q])
    space = FloatSpace.create(d, Metric.FP32_L2)

    serial = space.compute_distances(queries, base, 1)
    parallel = space.compute_distances(queries, base, 4)
    np.testing.assert_allclose(serial, parallel, rtol=1e-6, atol=1e-6)


def test_compute_distances_int8_dtype_ok():
    np.random.seed(3)
    n, d = 100, 64
    base = np.random.uniform(-3, 3, (n, d)).astype(np.float32)
    q = make_scalar_quantizer_int8(base)
    bq, qq = np.ascontiguousarray(q.quantize(base)), np.ascontiguousarray(q.quantize(base[:5]))
    space = FloatSpace.create(d, Metric.Int8_InnerProduct)
    dists = space.compute_distances(qq, bq)
    assert dists.shape == (5, n)
