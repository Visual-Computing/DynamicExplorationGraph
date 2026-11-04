import pytest
import numpy as np
import deglib


class TestEvpQuantizerUnit:
    def test_evp_quantizer_init(self):
        q = deglib.transformation.EvpQuantizer(16, 4)
        assert q.dim == 16
        assert q.non_zeros == 4
        assert q.mean is None
        assert not q.is_fitted

    def test_evp_quantizer_quantize_float32(self):
        dim = 16
        count = 10
        non_zeros = 4
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((count, dim), dtype=np.float32)

        q = deglib.transformation.EvpQuantizer(dim, non_zeros)
        q.fit(vecs)
        assert q.is_fitted
        bits = q.quantize(vecs, num_threads=2)

        assert bits.shape == (count, 2 * ((dim + 7) // 8))
        assert bits.dtype == np.uint8

        # Verify positive and negative bitmasks do not overlap
        mask_bytes = (dim + 7) // 8
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

        q1 = deglib.transformation.EvpQuantizer(dim, non_zeros)
        q1.fit(vecs_f16.astype(np.float32))
        bits_f32 = q1.quantize(vecs_f16.astype(np.float32))

        q2 = deglib.transformation.EvpQuantizer(dim, non_zeros)
        q2.fit(vecs_f16)
        bits_f16 = q2.quantize(vecs_f16)

        # FP16 and FP32 on the same float16 values produce bit-identical results
        np.testing.assert_array_equal(bits_f32, bits_f16)

    def test_evp_quantizer_fit_mean_float32(self):
        dim = 16
        count = 20
        rng = np.random.default_rng(99)
        vecs = rng.uniform(-5.0, 5.0, size=(count, dim)).astype(np.float32)

        q = deglib.transformation.EvpQuantizer(dim, 8)
        assert q.mean is None

        q.fit(vecs, num_threads=2)
        assert q.mean is not None
        assert q.mean.shape == (dim,)
        assert q.mean.dtype == np.float32

        expected_mean = np.mean(vecs, axis=0, dtype=np.float64).astype(np.float32)
        np.testing.assert_allclose(q.mean, expected_mean, rtol=1e-5, atol=1e-5)

    def test_evp_quantizer_fit_mean_float16(self):
        dim = 16
        count = 20
        rng = np.random.default_rng(99)
        vecs = rng.uniform(-5.0, 5.0, size=(count, dim)).astype(np.float16)

        q = deglib.transformation.EvpQuantizer(dim, 8)
        q.fit(vecs, num_threads=2)
        assert q.mean is not None

        expected_mean = np.mean(vecs.astype(np.float32), axis=0, dtype=np.float64).astype(np.float32)
        np.testing.assert_allclose(q.mean, expected_mean, rtol=1e-3, atol=1e-3)

    def test_evp_quantizer_arbitrary_dimension(self):
        # Test that dimensions NOT divisible by 8 work seamlessly
        for dim in [7, 10, 15, 23, 77]:
            count = 25
            rng = np.random.default_rng(42 + dim)
            vecs = rng.standard_normal((count, dim), dtype=np.float32)

            q = deglib.transformation.EvpQuantizer(dim, 0)
            q.fit(vecs)
            assert q.is_fitted
            assert q.non_zeros > 0
            assert q.non_zeros < dim

            bits = q.quantize(vecs)
            expected_mask_bytes = (dim + 7) // 8
            assert bits.shape == (count, 2 * expected_mask_bytes)

            ones = bits[:, :expected_mask_bytes]
            negs = bits[:, expected_mask_bytes:]
            assert np.all((ones & negs) == 0)

            # High unused bits in last byte must be 0
            remainder = dim % 8
            if remainder != 0:
                valid_mask = np.uint8((1 << remainder) - 1)
                assert np.all((ones[:, -1] & ~valid_mask) == 0)
                assert np.all((negs[:, -1] & ~valid_mask) == 0)

    def test_evp_quantizer_unfitted_raises(self):
        dim = 16
        q = deglib.transformation.EvpQuantizer(dim, 4)
        vecs = np.ones((5, dim), dtype=np.float32)
        with pytest.raises(Exception):
            q.quantize(vecs)

    def test_evp_quantizer_dequantize(self):
        dim = 8
        non_zeros = 4
        q = deglib.transformation.EvpQuantizer(dim, non_zeros)
        # Train on zeros so mean is 0
        q.fit(np.zeros((10, dim), dtype=np.float32))

        vec = np.array([[1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]], dtype=np.float32)
        quantized = q.quantize(vec)
        dequantized = q.dequantize(quantized)

        expected = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
        np.testing.assert_allclose(dequantized, expected)

