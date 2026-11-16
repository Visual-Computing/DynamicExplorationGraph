import numpy as np


def test_uint7_quantizer_uses_affine_range():
    from deglib.transformation import ScalarQuantizerUInt7

    data = np.array([[-1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    quantizer = ScalarQuantizerUInt7(4)
    quantizer.fit(data, quantile=1.0)

    np.testing.assert_allclose(quantizer.min, -1.0, rtol=1e-6)
    np.testing.assert_allclose(quantizer.max, 1.0, rtol=1e-6)
    np.testing.assert_allclose(quantizer.scale, 127.0 / 2.0, rtol=1e-6)
    np.testing.assert_array_equal(quantizer.quantize(data), np.array([[0, 64, 95, 127]], dtype=np.int8))
