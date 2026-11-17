import numpy as np
from deglib.transformation import apply_on_database, make_scalar_quantizer_uint7


def test_uint7_quantizer_uses_affine_range():
    dim = 4
    data = np.array([[0.0, 2.5, 5.0, 10.0]], dtype=np.float32)
    quantizer = make_scalar_quantizer_uint7(data, quantile=1.0)
    assert quantizer.output_dtype == np.int8

    quantized = quantizer.transform_database(data)
    assert quantized.dtype == np.int8
    assert np.all((quantized >= 0) & (quantized <= 127))
    np.testing.assert_array_equal(quantized[0], [0, 32, 64, 127])

    via_pipe = apply_on_database(quantizer, data)
    np.testing.assert_array_equal(via_pipe, quantized)
