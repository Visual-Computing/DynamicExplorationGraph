import pytest
import numpy as np
import deglib


class TestEvpQuantizerUnit:
    def test_evp_quantizer_init(self):
        q = deglib.transformation.EvpQuantizer(16)
        assert q.non_zeros == 16
        assert q.mean is None

        q.non_zeros = 32
        assert q.non_zeros == 32

    def test_evp_quantizer_quantize_float32(self):
        dim = 16
        count = 10
        non_zeros = 4
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((count, dim), dtype=np.float32)

        q = deglib.transformation.EvpQuantizer(non_zeros)
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

        q1 = deglib.transformation.EvpQuantizer(non_zeros)
        bits_f32 = q1.quantize(vecs_f16.astype(np.float32))
        q2 = deglib.transformation.EvpQuantizer(non_zeros)
        bits_f16 = q2.quantize(vecs_f16)

        # FP16 and FP32 on the same float16 values produce bit-identical results
        np.testing.assert_array_equal(bits_f32, bits_f16)

    def test_evp_quantizer_fit_mean_float32(self):
        dim = 16
        count = 20
        rng = np.random.default_rng(99)
        vecs = rng.uniform(-5.0, 5.0, size=(count, dim)).astype(np.float32)

        q = deglib.transformation.EvpQuantizer(8)
        assert q.mean is None

        q.fit_mean(vecs, num_threads=2)
        assert q.mean is not None
        assert q.mean.shape == (dim,)
        assert q.mean.dtype == np.float32

        expected_mean = np.mean(vecs, axis=0, dtype=np.float64).astype(np.float32)
        np.testing.assert_allclose(q.mean, expected_mean, rtol=1e-5, atol=1e-5)

        q.clear_mean()
        assert q.mean is None

    def test_evp_quantizer_fit_mean_float16(self):
        dim = 16
        count = 20
        rng = np.random.default_rng(99)
        vecs = rng.uniform(-5.0, 5.0, size=(count, dim)).astype(np.float16)

        q = deglib.transformation.EvpQuantizer(8)
        q.fit_mean(vecs, num_threads=2)
        assert q.mean is not None

        expected_mean = np.mean(vecs.astype(np.float32), axis=0, dtype=np.float64).astype(np.float32)
        np.testing.assert_allclose(q.mean, expected_mean, rtol=1e-3, atol=1e-3)

    def test_evp_quantizer_quantize_always_centers_mean(self):
        dim = 16
        count = 50
        non_zeros = 6
        rng = np.random.default_rng(77)
        vecs = rng.uniform(-3.0, 7.0, size=(count, dim)).astype(np.float32)

        # 1. Quantize automatically centers by mean
        q_auto = deglib.transformation.EvpQuantizer(non_zeros)
        bits_auto = q_auto.quantize(vecs, num_threads=2)
        assert q_auto.mean is not None
        computed_mean = q_auto.mean

        # 2. Manual centering with zero mean
        vecs_centered = vecs - computed_mean
        q_manual = deglib.transformation.EvpQuantizer(non_zeros)
        q_manual.set_mean(np.zeros(dim, dtype=np.float32))
        bits_manual = q_manual.quantize(vecs_centered, num_threads=2)

        np.testing.assert_array_equal(bits_auto, bits_manual)

    def test_evp_quantizer_set_mean(self):
        dim = 8
        count = 5
        vecs = np.array([[5.0, 1.0, 6.0, 0.0, 7.0, -1.0, 8.0, -2.0]] * count, dtype=np.float32)
        mean = np.array([4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0], dtype=np.float32)

        q = deglib.transformation.EvpQuantizer(4)
        q.set_mean(mean)
        np.testing.assert_array_equal(q.mean, mean)

        bits = q.quantize(vecs)
        # Centered: [1, -1, 2, -2, 3, -3, 4, -4]
        # ones: 0x50, negs: 0xA0
        assert bits.shape == (count, 2)
        for i in range(count):
            assert bits[i, 0] == 0x50
            assert bits[i, 1] == 0xA0

    def test_evp_quantizer_invalid_arguments(self):
        q = deglib.transformation.EvpQuantizer(4)

        # Dim not divisible by 8
        invalid_dim_vecs = np.ones((5, 10), dtype=np.float32)
        with pytest.raises(Exception):
            q.quantize(invalid_dim_vecs)

        # Mean dimension mismatch
        valid_vecs = np.ones((5, 16), dtype=np.float32)
        wrong_mean = np.ones(8, dtype=np.float32)
        q.set_mean(wrong_mean)
        with pytest.raises(Exception):
            q.quantize(valid_vecs)
