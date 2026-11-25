import numpy as np
import pytest

import deglib
from deglib import Metric
from deglib.cpu import InstructionSet
from deglib.transformation import apply_on_database, make_scalar_quantizer_int8


def test_compute_distances_batch_shapes():
    rng = np.random.default_rng(0)
    n, d, q = 200, 32, 12
    base = rng.standard_normal((n, d), dtype=np.float32)
    queries = np.ascontiguousarray(base[:q])

    # 2D batch -> [Q, N]
    dists = Metric.FP32_L2.compute_distances(queries, base)
    assert dists.shape == (q, n)
    assert dists.dtype == np.float32
    expected = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(dists, expected, rtol=1e-4, atol=1e-5)

    # 1D query against 2D base -> [1, N]
    d1 = Metric.FP32_L2.compute_distances(queries[0], base)
    assert d1.shape == (1, n)
    np.testing.assert_allclose(d1[0], expected[0], rtol=1e-4, atol=1e-5)

    # Single compute_distance
    scalar_d = Metric.FP32_L2.compute_distance(queries[0], base[1])
    assert isinstance(scalar_d, float)
    assert scalar_d == pytest.approx(float(expected[0, 1]), rel=1e-4, abs=1e-5)


def test_compute_distances_threads_match_serial():
    rng = np.random.default_rng(1)
    n, d, q = 300, 16, 16
    base = rng.standard_normal((n, d), dtype=np.float32)
    queries = np.ascontiguousarray(base[:q])

    serial = Metric.FP32_L2.compute_distances(queries, base, 1)
    parallel = Metric.FP32_L2.compute_distances(queries, base, 4)
    np.testing.assert_allclose(parallel, serial, rtol=1e-6, atol=1e-6)


def test_compute_distances_int8_dtype_ok():
    rng = np.random.default_rng(2)
    n, d = 64, 16
    base = rng.standard_normal((n, d), dtype=np.float32)
    quant = make_scalar_quantizer_int8(base)
    bq = np.ascontiguousarray(apply_on_database(quant, base))
    qq = np.ascontiguousarray(apply_on_database(quant, base[:5]))
    dists = Metric.Int8_InnerProduct.compute_distances(qq, bq)
    assert dists.shape == (5, n)


def test_metric_resolve_and_with_instruction():
    assert Metric.resolve("FP32_L2") == Metric.FP32_L2
    assert Metric.resolve("Uint8_L2") == Metric.Uint8_L2
    assert Metric.resolve("int8", "ip") == Metric.Int8_InnerProduct
    assert Metric.resolve("fp32", "cosine") == Metric.FP32_InnerProduct
    assert Metric.resolve("evp") == Metric.EVP_InnerProduct

    scalar_m = Metric.FP32_L2.with_instruction("Scalar")
    assert isinstance(scalar_m, Metric)
    assert scalar_m == Metric.FP32_L2
    assert scalar_m.get_instruction(16) == InstructionSet.Scalar
    assert scalar_m.get_data_size(16) == 64
    assert scalar_m.get_query_data_size(16) == 64

    resolved_scalar = Metric.resolve("fp32", "l2", instruction=InstructionSet.Scalar)
    assert resolved_scalar.get_instruction(16) == InstructionSet.Scalar
    assert not hasattr(deglib, "FloatSpace")

    with pytest.raises(ValueError):
        Metric.resolve("evp", "l2")
