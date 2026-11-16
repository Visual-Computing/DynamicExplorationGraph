import numpy as np
import pytest

from deglib.transformation import (
    ScalarQuantizerInt8,
    ScalarQuantizerUint8,
    ScalarQuantizerFP16,
    make_scalar_quantizer_fp16,
)


def test_int8_quantizer_fit_and_quantize():
    np.random.seed(42)
    # Database vectors in range [-5.0, 5.0]
    db_vectors = (np.random.uniform(-5.0, 5.0, size=(100, 32))).astype(np.float32)
    # Query vectors in range [-1.0, 1.0]
    query_vectors = (np.random.uniform(-1.0, 1.0, size=(10, 32))).astype(np.float32)

    quantizer = ScalarQuantizerInt8(32)
    assert not quantizer.is_fitted

    quantizer.fit(db_vectors, quantile=1.0)
    assert quantizer.is_fitted
    assert quantizer.abs_max > 4.5
    assert quantizer.scale == pytest.approx(127.0 / quantizer.abs_max, rel=1e-5)

    # Quantize database and query vectors against the SAME distribution
    quant_db = quantizer.quantize(db_vectors)
    quant_query = quantizer.quantize(query_vectors)

    assert quant_db.shape == (100, 32)
    assert quant_db.dtype == np.int8
    assert quant_query.shape == (10, 32)
    assert quant_query.dtype == np.int8

    # Query elements should have smaller quantized values because they are on the db scale
    assert np.max(quant_query) < np.max(quant_db)


def test_make_scalar_quantizer_factory_functions():
    np.random.seed(42)
    vectors = np.random.randn(50, 16).astype(np.float32)

    from deglib.transformation import (
        make_scalar_quantizer_int8,
        make_scalar_quantizer_uint8,
    )

    q_int8 = make_scalar_quantizer_int8(vectors)
    assert isinstance(q_int8, ScalarQuantizerInt8)
    assert q_int8.is_fitted
    assert q_int8.quantize(vectors).shape == (50, 16)

    q_uint8 = make_scalar_quantizer_uint8(vectors)
    assert isinstance(q_uint8, ScalarQuantizerUint8)
    assert q_uint8.is_fitted
    assert q_uint8.quantize(vectors).shape == (50, 16)


def test_int8_quantizer_fit_and_quantize_basic():
    np.random.seed(42)
    vectors = np.random.randn(50, 16).astype(np.float32)

    q1 = ScalarQuantizerInt8(16)
    q1.fit(vectors)
    res1 = q1.quantize(vectors)

    assert q1.is_fitted
    assert res1.shape == (50, 16)
    assert res1.dtype == np.int8


def test_int8_quantizer_fp16_input():
    np.random.seed(42)
    vectors_f32 = (np.random.uniform(-2.0, 2.0, size=(20, 16))).astype(np.float32)
    vectors_f16 = vectors_f32.astype(np.float16)

    q_f32 = ScalarQuantizerInt8(16)
    q_f32.fit(vectors_f32)
    res_f32 = q_f32.quantize(vectors_f32)

    q_f16 = ScalarQuantizerInt8(16)
    q_f16.fit(vectors_f16)
    res_f16 = q_f16.quantize(vectors_f16)

    assert q_f32.abs_max == pytest.approx(q_f16.abs_max, rel=1e-3)
    np.testing.assert_allclose(res_f32, res_f16, atol=1)


def test_uint8_quantizer_fit_and_quantize():
    np.random.seed(42)
    db_vectors = (np.random.uniform(10.0, 50.0, size=(100, 16))).astype(np.float32)
    query_vectors = (np.random.uniform(20.0, 40.0, size=(10, 16))).astype(np.float32)

    quantizer = ScalarQuantizerUint8(16)
    quantizer.fit(db_vectors)

    assert quantizer.is_fitted
    assert quantizer.min_val == pytest.approx(np.min(db_vectors))
    assert quantizer.max_val == pytest.approx(np.max(db_vectors))

    quant_db = quantizer.quantize(db_vectors)
    quant_query = quantizer.quantize(query_vectors)

    assert quant_db.shape == (100, 16)
    assert quant_db.dtype == np.uint8
    assert quant_query.shape == (10, 16)
    assert quant_query.dtype == np.uint8


def test_uint8_fp16_input():
    np.random.seed(42)
    vectors_f32 = (np.random.uniform(5.0, 50.0, size=(30, 8))).astype(np.float32)
    vectors_f16 = vectors_f32.astype(np.float16)

    from deglib.transformation import ScalarQuantizerUint8

    q_u8_f32 = ScalarQuantizerUint8(8)
    q_u8_f32.fit(vectors_f32)
    res_u8_f32 = q_u8_f32.quantize(vectors_f32)

    q_u8_f16 = ScalarQuantizerUint8(8)
    q_u8_f16.fit(vectors_f16)
    res_u8_f16 = q_u8_f16.quantize(vectors_f16)

    np.testing.assert_allclose(res_u8_f32, res_u8_f16, atol=1)


def test_fp16_quantizer_identity_in_range():
    np.random.seed(42)
    # Values well within FP16 range -> scale stays 1.0 (bit-exact conversion).
    vectors = np.random.uniform(-2.0, 2.0, size=(40, 16)).astype(np.float32)

    q = ScalarQuantizerFP16(16)
    assert not q.is_fitted
    q.fit(vectors)
    assert q.is_fitted
    assert q.scale == pytest.approx(1.0)
    assert q.inv_scale == pytest.approx(1.0)

    quant = q.quantize(vectors)
    assert quant.shape == (40, 16)
    assert quant.dtype == np.float16
    # In-range, scale==1 -> quantized equals direct FP16 cast of the float32 input.
    np.testing.assert_array_equal(quant, vectors.astype(np.float16))


def test_fp16_quantizer_scales_large_values():
    np.random.seed(42)
    # Audio-like dataset with values far above FP16 max (65504).
    vectors = (np.random.uniform(-100000.0, 100000.0, size=(64, 16))).astype(np.float32)

    q = ScalarQuantizerFP16(16)
    q.fit(vectors)
    assert q.abs_max > 65504.0
    assert q.scale < 1.0

    quant = q.quantize(vectors)
    assert quant.dtype == np.float16
    assert not np.any(np.isinf(quant))
    assert not np.any(np.isnan(quant))


def test_fp16_quantizer_dequantize_roundtrip():
    np.random.seed(42)
    vectors = (np.random.uniform(-100000.0, 100000.0, size=(32, 16))).astype(np.float32)

    q = ScalarQuantizerFP16(16)
    q.fit(vectors)

    quant = q.quantize(vectors)
    dequant = q.dequantize(quant)
    assert dequant.dtype == np.float32
    assert dequant.shape == vectors.shape
    # FP16 relative precision ~ 2^-11; power-of-two scaling is exact.
    np.testing.assert_allclose(dequant, vectors, rtol=2e-3, atol=1.0)


def test_fp16_quantizer_serialize_roundtrip():
    np.random.seed(42)
    vectors = (np.random.uniform(-90000.0, 90000.0, size=(20, 8))).astype(np.float32)

    q = ScalarQuantizerFP16(8)
    q.fit(vectors)

    data = q.serialize()
    q2 = ScalarQuantizerFP16.deserialize(data)
    assert q2.dim == q.dim
    assert q2.abs_max == pytest.approx(q.abs_max)
    assert q2.scale == pytest.approx(q.scale)
    assert q2.inv_scale == pytest.approx(q.inv_scale)
    assert q2.is_fitted

    # Quantization output identical after roundtrip.
    np.testing.assert_array_equal(q2.quantize(vectors), q.quantize(vectors))


def test_fp16_quantizer_factory_and_fp16_input():
    np.random.seed(42)
    vectors_f32 = np.random.uniform(-2.0, 2.0, size=(50, 16)).astype(np.float32)

    q = make_scalar_quantizer_fp16(vectors_f32)
    assert isinstance(q, ScalarQuantizerFP16)
    assert q.is_fitted
    assert q.quantize(vectors_f32).shape == (50, 16)

    # FP16 input path (scale==1) matches float32 path.
    vectors_f16 = vectors_f32.astype(np.float16)
    q2 = ScalarQuantizerFP16(16)
    q2.fit(vectors_f16)
    np.testing.assert_array_equal(q.quantize(vectors_f32), q.quantize(vectors_f16))


def test_int7_quantizer_fit_and_quantize():
    np.random.seed(42)
    dim = 32
    db_vectors = np.random.uniform(-5.0, 5.0, size=(100, dim)).astype(np.float32)

    from deglib.transformation import ScalarQuantizerUInt7, make_scalar_quantizer_uint7

    q = ScalarQuantizerUInt7(dim)
    assert not q.is_fitted
    q.fit(db_vectors, quantile=1.0)
    assert q.is_fitted
    assert q.dim == dim

    quant = q.quantize(db_vectors)
    assert quant.shape == (100, dim)
    assert quant.dtype == np.int8
    assert quant.min() >= 0
    assert quant.max() <= 127

    q2 = make_scalar_quantizer_uint7(db_vectors)
    np.testing.assert_array_equal(quant, q2.quantize(db_vectors))


def test_uint7_distance_computation():
    np.random.seed(7)
    dim = 64
    a = np.random.randint(0, 128, size=(1, dim)).astype(np.int8)
    b = np.random.randint(0, 128, size=(1, dim)).astype(np.int8)

    from deglib.distances import FloatSpace, Metric

    space = FloatSpace.create(dim, Metric.UInt7_L2)
    distance = space.compute_distance(a[0], b[0])
    expected = np.sum((a[0].astype(np.int64) - b[0].astype(np.int64)) ** 2)
    np.testing.assert_allclose(distance, expected, rtol=1e-5)


def test_uint4_quantizer_fit_and_quantize():
    np.random.seed(42)
    dim = 32
    db_vectors = np.random.uniform(-5.0, 5.0, size=(100, dim)).astype(np.float32)

    from deglib.transformation import ScalarQuantizerUint4, make_scalar_quantizer_uint4

    q = ScalarQuantizerUint4(dim)
    assert not q.is_fitted
    q.fit(db_vectors, quantile=1.0)
    assert q.is_fitted
    assert q.dim == dim

    quant = q.quantize(db_vectors)
    assert quant.shape == (100, (dim + 1) // 2)
    assert quant.dtype == np.uint8

    # Test factory function
    q2 = make_scalar_quantizer_uint4(db_vectors)
    assert q2.is_fitted
    assert q2.dim == dim
    np.testing.assert_array_equal(q.quantize(db_vectors), q2.quantize(db_vectors))


def test_uint4_distance_computation():
    from deglib.distances import FloatSpace, Metric

    dim = 32
    space_l2 = FloatSpace.create(dim, Metric.Uint4_L2)
    assert space_l2.get_data_size() == (dim + 1) // 2

    space_ip = FloatSpace.create(dim, Metric.Uint4_InnerProduct)
    assert space_ip.get_data_size() == (dim + 1) // 2
