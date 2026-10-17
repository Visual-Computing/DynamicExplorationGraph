import pytest
import numpy as np
import deglib


class TestEvpQuantizerUnit:
    def test_evp_quantizer_init(self):
        q = deglib.optimization.EvpQuantizer(16)
        assert q.non_zeros == 16

        q.non_zeros = 32
        assert q.non_zeros == 32

    def test_evp_quantizer_quantize_float32(self):
        dim = 16
        count = 10
        non_zeros = 4
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((count, dim), dtype=np.float32)

        q = deglib.optimization.EvpQuantizer(non_zeros)
        bits = q.quantize(vecs, num_threads=2)

        assert bits.shape == (count, 2 * (dim // 8))
        assert bits.dtype == np.uint8

        # Verify positive and negative bitmasks do not overlap
        mask_bytes = dim // 8
        ones = bits[:, :mask_bytes]
        negs = bits[:, mask_bytes:]
        assert np.all((ones & negs) == 0)

    def test_evp_quantizer_float32_vs_float16_equivalence(self):
        dim = 32
        count = 10
        non_zeros = 8
        rng = np.random.default_rng(123)
        vecs_f32 = rng.standard_normal((count, dim), dtype=np.float32)
        vecs_f16 = vecs_f32.astype(np.float16)

        q1 = deglib.optimization.EvpQuantizer(non_zeros)
        bits_f32 = q1.quantize(vecs_f16.astype(np.float32))
        q2 = deglib.optimization.EvpQuantizer(non_zeros)
        bits_f16 = q2.quantize(vecs_f16)

        # FP16 and FP32 on the same float16 values produce bit-identical results
        np.testing.assert_array_equal(bits_f32, bits_f16)

    def test_evp_quantizer_invalid_arguments(self):
        q = deglib.optimization.EvpQuantizer(4)

        # Dim not divisible by 8
        invalid_dim_vecs = np.ones((5, 10), dtype=np.float32)
        with pytest.raises(Exception):
            q.quantize(invalid_dim_vecs)
