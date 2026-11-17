import numpy as np
from deglib.transformation import apply_on_database, make_scalar_quantizer_int4


def test_int4_quantizer_fit_and_quantize():
    rng = np.random.default_rng(42)
    dim = 32
    data = rng.normal(size=(200, dim)).astype(np.float32)

    q = make_scalar_quantizer_int4(data)
    assert q.name == "ScalarQuantizerInt4"
    assert q.is_fitted
    assert q.input_dim == dim
    assert q.output_dim == dim
    assert q.output_byte_size == (dim + 1) // 2

    quantized = apply_on_database(q, data)
    assert quantized.shape == (200, dim // 2)
    assert quantized.dtype == np.uint8

    reconstructed = q.dequantize(quantized)
    assert reconstructed.shape == (200, dim)
    assert reconstructed.dtype == np.float32

    mae = np.mean(np.abs(data - reconstructed))
    assert mae < 0.5


def test_int4_quantizer_clamps_to_signed_int4_range():
    dim = 16
    data = np.linspace(-10.0, 10.0, num=64 * dim, dtype=np.float32).reshape(64, dim)
    q = make_scalar_quantizer_int4(data, quantile=1.0)
    reconstructed = q.dequantize(apply_on_database(q, data))
    np.testing.assert_allclose(reconstructed, data, atol=1.5)
