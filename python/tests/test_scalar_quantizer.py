import numpy as np
import pytest

from deglib.optimization import (
    ScalarQuantizerInt8,
    ScalarQuantizerUint8,
    ScalarQuantizerUint8PerDim,
)


def test_int8_quantizer_fit_and_quantize():
    np.random.seed(42)
    # Database vectors in range [-5.0, 5.0]
    db_vectors = (np.random.uniform(-5.0, 5.0, size=(100, 32))).astype(np.float32)
    # Query vectors in range [-1.0, 1.0]
    query_vectors = (np.random.uniform(-1.0, 1.0, size=(10, 32))).astype(np.float32)

    quantizer = ScalarQuantizerInt8()
    assert not quantizer.is_fitted

    quantizer.fit(db_vectors, drop_ratio=0.0)
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

    # Query elements should have smaller absolute quantized values because they are on the db scale
    assert np.max(np.abs(quant_query)) < np.max(np.abs(quant_db))


def test_make_scalar_quantizer_factory_functions():
    np.random.seed(42)
    vectors = np.random.randn(50, 16).astype(np.float32)

    from deglib.optimization import (
        make_scalar_quantizer_int8,
        make_scalar_quantizer_uint8,
        make_scalar_quantizer_uint8_perdim,
    )

    q_int8 = make_scalar_quantizer_int8(vectors)
    assert isinstance(q_int8, ScalarQuantizerInt8)
    assert q_int8.is_fitted
    assert q_int8.quantize(vectors).shape == (50, 16)

    q_uint8 = make_scalar_quantizer_uint8(vectors)
    assert isinstance(q_uint8, ScalarQuantizerUint8)
    assert q_uint8.is_fitted
    assert q_uint8.quantize(vectors).shape == (50, 16)

    q_perdim = make_scalar_quantizer_uint8_perdim(vectors)
    assert isinstance(q_perdim, ScalarQuantizerUint8PerDim)
    assert q_perdim.is_fitted
    assert q_perdim.quantize(vectors).shape == (50, 16)


def test_int8_quantizer_fit_and_quantize():
    np.random.seed(42)
    vectors = np.random.randn(50, 16).astype(np.float32)

    q1 = ScalarQuantizerInt8()
    q1.fit(vectors)
    res1 = q1.quantize(vectors)

    assert q1.is_fitted
    assert res1.shape == (50, 16)
    assert res1.dtype == np.int8


def test_int8_quantizer_fp16_input():
    np.random.seed(42)
    vectors_f32 = (np.random.uniform(-2.0, 2.0, size=(20, 16))).astype(np.float32)
    vectors_f16 = vectors_f32.astype(np.float16)

    q_f32 = ScalarQuantizerInt8()
    q_f32.fit(vectors_f32)
    res_f32 = q_f32.quantize(vectors_f32)

    q_f16 = ScalarQuantizerInt8()
    q_f16.fit(vectors_f16)
    res_f16 = q_f16.quantize(vectors_f16)

    assert q_f32.abs_max == pytest.approx(q_f16.abs_max, rel=1e-3)
    np.testing.assert_allclose(res_f32, res_f16, atol=1)


def test_uint8_quantizer_fit_and_quantize():
    np.random.seed(42)
    db_vectors = (np.random.uniform(10.0, 50.0, size=(100, 16))).astype(np.float32)
    query_vectors = (np.random.uniform(20.0, 40.0, size=(10, 16))).astype(np.float32)

    quantizer = ScalarQuantizerUint8()
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


def test_uint8_per_dim_quantizer():
    # Dim 0: [0, 100], Dim 1: [-10, 10]
    db_vectors = np.array([
        [0.0, -10.0],
        [50.0, 0.0],
        [100.0, 10.0],
    ], dtype=np.float32)

    quantizer = ScalarQuantizerUint8PerDim()
    quantizer.fit(db_vectors)
    quant_db = quantizer.quantize(db_vectors)

    assert quantizer.is_fitted
    assert quantizer.dim == 2
    assert quantizer.mins[0] == pytest.approx(0.0)
    assert quantizer.maxs[0] == pytest.approx(100.0)
    assert quantizer.mins[1] == pytest.approx(-10.0)
    assert quantizer.maxs[1] == pytest.approx(10.0)

    # 0.0 -> 0, 50.0 -> 128, 100.0 -> 255
    assert quant_db[0, 0] == 0
    assert quant_db[1, 0] == 128
    assert quant_db[2, 0] == 255

    # -10.0 -> 0, 0.0 -> 128, 10.0 -> 255
    assert quant_db[0, 1] == 0
    assert quant_db[1, 1] == 128
    assert quant_db[2, 1] == 255

    # Quantize new query on this per-dim scale
    query = np.array([[25.0, 5.0]], dtype=np.float32)
    quant_query = quantizer.quantize(query)

    # 25.0 in [0, 100] -> round(25/100 * 255) = 64
    assert quant_query[0, 0] == 64
    # 5.0 in [-10, 10] -> round((5 - (-10))/20 * 255) = round(15/20 * 255) = 191
    assert quant_query[0, 1] == 191


def test_int8_per_dim_quantizer():
    from deglib.optimization import ScalarQuantizerInt8PerDim, make_scalar_quantizer_int8_perdim

    db_vectors = np.array([
        [-100.0, -1.0],
        [0.0, 0.0],
        [100.0, 1.0],
    ], dtype=np.float32)

    quantizer = make_scalar_quantizer_int8_perdim(db_vectors)
    assert quantizer.is_fitted
    assert quantizer.dim == 2
    assert quantizer.abs_maxs[0] == pytest.approx(100.0)
    assert quantizer.abs_maxs[1] == pytest.approx(1.0)

    quant_db = quantizer.quantize(db_vectors)
    assert quant_db[0, 0] == -127
    assert quant_db[2, 0] == 127
    assert quant_db[0, 1] == -127
    assert quant_db[2, 1] == 127

    # FP16 test
    db_f16 = db_vectors.astype(np.float16)
    q_f16 = ScalarQuantizerInt8PerDim()
    q_f16.fit(db_f16)
    quant_f16 = q_f16.quantize(db_f16)
    assert np.array_equal(quant_db, quant_f16)


def test_uint8_fp16_input():
    np.random.seed(42)
    vectors_f32 = (np.random.uniform(5.0, 50.0, size=(30, 8))).astype(np.float32)
    vectors_f16 = vectors_f32.astype(np.float16)

    from deglib.optimization import ScalarQuantizerUint8, ScalarQuantizerUint8PerDim

    q_u8_f32 = ScalarQuantizerUint8()
    q_u8_f32.fit(vectors_f32)
    res_u8_f32 = q_u8_f32.quantize(vectors_f32)

    q_u8_f16 = ScalarQuantizerUint8()
    q_u8_f16.fit(vectors_f16)
    res_u8_f16 = q_u8_f16.quantize(vectors_f16)

    np.testing.assert_allclose(res_u8_f32, res_u8_f16, atol=1)

    q_pdim_f32 = ScalarQuantizerUint8PerDim()
    q_pdim_f32.fit(vectors_f32)
    res_pdim_f32 = q_pdim_f32.quantize(vectors_f32)

    q_pdim_f16 = ScalarQuantizerUint8PerDim()
    q_pdim_f16.fit(vectors_f16)
    res_pdim_f16 = q_pdim_f16.quantize(vectors_f16)

    np.testing.assert_allclose(res_pdim_f32, res_pdim_f16, atol=1)
