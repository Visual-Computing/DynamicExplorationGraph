import numpy as np
import pytest

import deglib
from deglib.transformation import apply_on_database, apply_on_query, make_evp_quantizer


class TestEvpQuantizerUnit:
    def test_no_raw_evp_class_exported(self):
        assert not hasattr(deglib, "EvpQuantizer")
        assert not hasattr(deglib.transformation, "EvpQuantizer")

    def test_evp_quantizer_fit_and_quantize_float32(self):
        rng = np.random.default_rng(42)
        dim = 64
        count = 100
        non_zeros = 16
        vecs = rng.uniform(-10.0, 10.0, size=(count, dim)).astype(np.float32)

        q = make_evp_quantizer(vecs, non_zeros=non_zeros, threads=2)
        assert q.name == "EvpQuantizer"
        assert q.input_dim == dim
        assert q.output_dim == dim
        assert q.non_zeros == non_zeros
        assert q.is_fitted
        assert q.output_dtype == np.uint8

        bits = q.transform_database(vecs, threads=2)
        assert bits.shape == (count, 2 * ((dim + 7) // 8))
        assert bits.dtype == np.uint8

        q_bits = apply_on_query(q, vecs[:5])
        np.testing.assert_array_equal(q_bits, bits[:5])

    def test_evp_quantizer_float32_vs_float16_equivalence(self):
        rng = np.random.default_rng(123)
        dim = 32
        count = 50
        non_zeros = 8
        vecs_f32 = rng.uniform(-10.0, 10.0, size=(count, dim)).astype(np.float32)
        vecs_f16 = vecs_f32.astype(np.float16)

        q1 = make_evp_quantizer(vecs_f16.astype(np.float32), non_zeros=non_zeros)
        bits_f32 = apply_on_database(q1, vecs_f16.astype(np.float32))

        q2 = make_evp_quantizer(vecs_f16, non_zeros=non_zeros)
        bits_f16 = apply_on_database(q2, vecs_f16)

        np.testing.assert_array_equal(bits_f32, bits_f16)

    def test_evp_quantizer_fit_mean_float32_and_float16(self):
        rng = np.random.default_rng(7)
        dim = 32
        vecs_f32 = rng.uniform(-5.0, 5.0, size=(100, dim)).astype(np.float32) + 10.0
        q_f32 = make_evp_quantizer(vecs_f32, non_zeros=8, threads=2)
        assert q_f32.mean is not None
        assert q_f32.mean.shape == (dim,)
        np.testing.assert_allclose(q_f32.mean, vecs_f32.mean(axis=0), atol=1e-4)

        vecs_f16 = vecs_f32.astype(np.float16)
        q_f16 = make_evp_quantizer(vecs_f16, non_zeros=8, threads=2)
        assert q_f16.mean is not None
        assert q_f16.mean.shape == (dim,)

    def test_evp_quantizer_arbitrary_dimension(self):
        rng = np.random.default_rng(99)
        for dim in (5, 8, 12, 16, 17, 32, 64, 65):
            vecs = rng.normal(size=(40, dim)).astype(np.float32)
            q = make_evp_quantizer(vecs, non_zeros=0)
            assert q.is_fitted
            assert q.non_zeros > 0
            out = q.transform_database(vecs)
            assert out.shape == (40, 2 * ((dim + 7) // 8))

    def test_evp_quantizer_validation_errors(self):
        empty = np.empty((0, 16), dtype=np.float32)
        with pytest.raises(Exception):
            make_evp_quantizer(empty, non_zeros=4)
        with pytest.raises(Exception):
            make_evp_quantizer(np.ones(16, dtype=np.float32), non_zeros=4)

    def test_evp_quantizer_dequantize(self):
        rng = np.random.default_rng(11)
        dim = 32
        vecs = rng.normal(size=(40, dim)).astype(np.float32)
        q = make_evp_quantizer(vecs, non_zeros=8)
        codes = q.transform_database(vecs)
        reconstructed = q.dequantize(codes)
        assert reconstructed.shape == (40, dim)
        assert reconstructed.dtype == np.float32
