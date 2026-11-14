import numpy as np


def test_int4_quantizer_fit_and_quantize():
    from deglib.transformation import ScalarQuantizerInt4

    dim = 4
    data = np.array([[1.0, -1.0, 0.5, -0.5]], dtype=np.float32)
    q = ScalarQuantizerInt4(dim=dim)
    q.fit(data)

    assert q.abs_max == 1.0
    assert q.scale == 8.0

    codes = q.quantize(data)
    assert codes.dtype == np.uint8
    assert codes.shape == (1, dim // 2)
    assert codes.tolist() == [[0x87, 0xC4]]


def test_int4_quantizer_clamps_to_signed_int4_range():
    from deglib.transformation import ScalarQuantizerInt4

    dim = 4
    data = np.array([[100.0, -100.0, 100.0, -100.0]], dtype=np.float32)
    q = ScalarQuantizerInt4(dim=dim)
    q.fit(data)
    codes = q.quantize(data)
    assert (codes[0, 0] & 0x0F) == 7
    assert (codes[0, 0] >> 4) == 8
