import numpy as np
import pytest

import deglib
from deglib.distances import Metric
from deglib.transformation import (
    FeatureTransformation,
    apply_on_database,
    apply_on_query,
    make_scalar_quantizer_fp16,
    make_scalar_quantizer_int8,
    make_scalar_quantizer_uint4,
    make_scalar_quantizer_uint7,
    make_scalar_quantizer_uint8,
)


def test_no_raw_scalar_quantizer_classes_exported():
    for name in (
        "ScalarQuantizerInt8",
        "ScalarQuantizerUint8",
        "ScalarQuantizerUInt7",
        "ScalarQuantizerUint4",
        "ScalarQuantizerInt4",
        "ScalarQuantizerFP16",
    ):
        assert not hasattr(deglib, name)
        assert not hasattr(deglib.transformation, name)


def test_int8_quantizer_fit_and_quantize():
    rng = np.random.default_rng(42)
    db_vectors = rng.uniform(-5.0, 5.0, size=(100, 32)).astype(np.float32)
    query_vectors = rng.uniform(-1.0, 1.0, size=(10, 32)).astype(np.float32)

    quantizer = make_scalar_quantizer_int8(db_vectors, quantile=1.0)
    assert isinstance(quantizer, FeatureTransformation)
    assert quantizer.is_fitted
    assert quantizer.abs_max > 4.5
    assert quantizer.scale == pytest.approx(127.0 / quantizer.abs_max, rel=1e-5)

    quant_db = quantizer.transform_database(db_vectors)
    quant_query = quantizer.transform_query(query_vectors)

    assert quant_db.shape == (100, 32)
    assert quant_db.dtype == np.int8
    assert quant_query.shape == (10, 32)
    assert quant_query.dtype == np.int8
    assert np.max(quant_query) < np.max(quant_db)


def test_make_scalar_quantizer_factory_functions():
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((50, 16)).astype(np.float32)

    q_int8 = make_scalar_quantizer_int8(vectors)
    assert q_int8.name == "ScalarQuantizerInt8"
    assert q_int8.is_fitted
    assert q_int8.transform_database(vectors).shape == (50, 16)

    q_uint8 = make_scalar_quantizer_uint8(vectors)
    assert q_uint8.name == "ScalarQuantizerUint8"
    assert q_uint8.is_fitted
    assert q_uint8.transform_database(vectors).shape == (50, 16)


def test_int8_quantizer_fp16_input():
    rng = np.random.default_rng(42)
    vectors_f32 = rng.uniform(-2.0, 2.0, size=(20, 16)).astype(np.float32)
    vectors_f16 = vectors_f32.astype(np.float16)

    q_f32 = make_scalar_quantizer_int8(vectors_f32)
    res_f32 = q_f32.transform_database(vectors_f32)

    q_f16 = make_scalar_quantizer_int8(vectors_f16)
    res_f16 = q_f16.transform_database(vectors_f16)

    assert q_f32.abs_max == pytest.approx(q_f16.abs_max, rel=1e-3)
    np.testing.assert_allclose(res_f32, res_f16, atol=1)


def test_uint8_quantizer_fit_and_quantize():
    rng = np.random.default_rng(42)
    db_vectors = rng.uniform(0.0, 10.0, size=(100, 16)).astype(np.float32)
    quantizer = make_scalar_quantizer_uint8(db_vectors, quantile=1.0)
    assert quantizer.is_fitted

    quant_db = apply_on_database(quantizer, db_vectors)
    assert quant_db.shape == (100, 16)
    assert quant_db.dtype == np.uint8
    assert quant_db.min() >= 0
    assert quant_db.max() <= 255

    deq = quantizer.dequantize(quant_db)
    np.testing.assert_allclose(deq, db_vectors, atol=0.1)


def test_uint8_fp16_input():
    rng = np.random.default_rng(7)
    vectors_f32 = rng.uniform(-1.0, 1.0, size=(20, 8)).astype(np.float32)
    vectors_f16 = vectors_f32.astype(np.float16)

    q_f32 = make_scalar_quantizer_uint8(vectors_f32)
    res_f32 = q_f32.transform_database(vectors_f32)

    q_f16 = make_scalar_quantizer_uint8(vectors_f16)
    res_f16 = q_f16.transform_database(vectors_f16)
    np.testing.assert_allclose(res_f32, res_f16, atol=1)


def test_fp16_quantizer_identity_in_range():
    rng = np.random.default_rng(42)
    vectors = rng.uniform(-10.0, 10.0, size=(50, 16)).astype(np.float32)
    q = make_scalar_quantizer_fp16(vectors)
    assert q.is_fitted
    assert q.scale == 1.0

    quant = q.transform_database(vectors)
    assert quant.dtype == np.float16
    np.testing.assert_array_equal(quant, vectors.astype(np.float16))


def test_fp16_quantizer_scales_large_values():
    rng = np.random.default_rng(42)
    vectors = rng.uniform(-100000.0, 100000.0, size=(50, 16)).astype(np.float32)
    q = make_scalar_quantizer_fp16(vectors)
    assert q.abs_max > 65504.0
    assert q.scale < 1.0

    quant = q.transform_database(vectors)
    assert quant.dtype == np.float16
    assert np.all(np.isfinite(quant))


def test_fp16_quantizer_dequantize_roundtrip():
    rng = np.random.default_rng(12)
    vectors = rng.uniform(-100000.0, 100000.0, size=(20, 16)).astype(np.float32)
    q = make_scalar_quantizer_fp16(vectors)
    quant = q.transform_database(vectors)
    reconstructed = q.dequantize(quant)
    assert reconstructed.dtype == np.float32
    np.testing.assert_allclose(reconstructed, vectors, rtol=1e-3, atol=1e-3)


def test_fp16_quantizer_serialize_roundtrip():
    rng = np.random.default_rng(5)
    vectors = rng.uniform(-100000.0, 100000.0, size=(20, 8)).astype(np.float32)
    q = make_scalar_quantizer_fp16(vectors)
    data = q.serialize()
    q2 = type(q).deserialize(data)
    assert q2.input_dim == q.input_dim
    assert q2.abs_max == pytest.approx(q.abs_max)
    np.testing.assert_array_equal(q2.transform_database(vectors), q.transform_database(vectors))


def test_fp16_quantizer_factory_and_fp16_input():
    rng = np.random.default_rng(42)
    vectors_f32 = rng.uniform(-5.0, 5.0, size=(50, 16)).astype(np.float32)
    q = make_scalar_quantizer_fp16(vectors_f32)
    assert q.is_fitted
    assert q.transform_database(vectors_f32).shape == (50, 16)

    vectors_f16 = vectors_f32.astype(np.float16)
    np.testing.assert_array_equal(apply_on_query(q, vectors_f32), apply_on_query(q, vectors_f16))


def test_uint7_quantizer_fit_quantize_and_distance():
    rng = np.random.default_rng(42)
    dim = 32
    db_vectors = rng.uniform(-1.0, 1.0, size=(100, dim)).astype(np.float32)
    q = make_scalar_quantizer_uint7(db_vectors, quantile=1.0)
    assert q.is_fitted
    assert q.output_dtype == np.int8

    quant_db = q.transform_database(db_vectors)
    assert quant_db.dtype == np.int8
    assert quant_db.min() >= 0
    assert quant_db.max() <= 127

    dist = Metric.UInt7_L2.compute_distance(quant_db[0], quant_db[1])
    expected = float(np.sum((quant_db[0].astype(np.int64) - quant_db[1].astype(np.int64)) ** 2))
    assert dist == pytest.approx(expected)


def test_uint4_quantizer_fit_quantize_and_distance():
    rng = np.random.default_rng(42)
    dim = 32
    db_vectors = rng.uniform(-1.0, 1.0, size=(64, dim)).astype(np.float32)
    q = make_scalar_quantizer_uint4(db_vectors, quantile=1.0)
    assert q.is_fitted

    quant_db = q.transform_database(db_vectors)
    assert quant_db.shape == (64, (dim + 1) // 2)
    assert quant_db.dtype == np.uint8

    assert Metric.Uint4_L2.get_data_size(dim) == (dim + 1) // 2
    assert Metric.Uint4_InnerProduct.get_data_size(dim) == (dim + 1) // 2
    dist = Metric.Uint4_L2.compute_distance(quant_db[0], quant_db[1])
    assert dist >= 0.0
