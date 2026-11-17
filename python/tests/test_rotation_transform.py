import numpy as np
import pytest

import deglib


@pytest.mark.parametrize("dim", [64, 96, 128, 200])
def test_rotation_orthogonality_preserves_distances(dim):
    rng = np.random.default_rng(123)
    X = rng.standard_normal((50, dim), dtype=np.float32)
    rot = deglib.make_fast_hadamard_rotation(X, seed=42, auto_detect=False)
    assert rot.is_enabled is True
    Y = rot.transform(X)
    assert Y.shape == X.shape
    assert Y.dtype == np.float32

    # Norms and pairwise inner products / L2 distances must be preserved
    np.testing.assert_allclose(np.linalg.norm(Y, axis=1), np.linalg.norm(X, axis=1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(Y @ Y.T, X @ X.T, rtol=1e-4, atol=1e-4)


def test_rotation_auto_detect_isotropic_vs_outliers():
    rng = np.random.default_rng(7)
    X_iso = rng.standard_normal((2000, 128), dtype=np.float32)
    X_iso /= np.linalg.norm(X_iso, axis=1, keepdims=True)

    rot_iso = deglib.make_fast_hadamard_rotation(X_iso, auto_detect=True)
    assert rot_iso.is_enabled is False
    # Pass-through must be exact
    np.testing.assert_array_equal(rot_iso.transform(X_iso[:10]), X_iso[:10])

    # Inject a strong outlier channel
    X_out = X_iso.copy()
    X_out[:, 0] -= 0.65
    X_out /= np.linalg.norm(X_out, axis=1, keepdims=True)
    rot_out = deglib.make_fast_hadamard_rotation(X_out, auto_detect=True)
    assert rot_out.is_enabled is True
    assert rot_out.crest_factor > 9.5 or rot_out.excess_kurtosis > 2.0


def test_rotation_serialization_roundtrip():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((100, 96), dtype=np.float32)
    X[:, 0] += 5.0

    rot = deglib.make_fast_hadamard_rotation(X, seed=99, auto_detect=True)
    blob = deglib.transformation.serialize_transformations([rot])
    restored = deglib.transformation.deserialize_transformations(blob)
    assert len(restored) == 1
    r2 = restored[0]
    assert r2.is_enabled == rot.is_enabled
    np.testing.assert_allclose(r2.transform(X[:10]), rot.transform(X[:10]), rtol=1e-6, atol=1e-6)


def test_rotation_chained_with_uint4_in_searcher():
    rng = np.random.default_rng(21)
    dim = 128
    n = 300
    X = rng.standard_normal((n, dim), dtype=np.float32)
    X[:, 0] -= 15.0
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    space_fp32 = deglib.FloatSpace.create(dim=dim, metric=deglib.Metric.FP32_InnerProduct)
    graph = deglib.builder.build_from_data(
        X, edges_per_vertex=16, metric=deglib.Metric.FP32_InnerProduct, thread_count=2
    )

    rot = deglib.make_fast_hadamard_rotation(X, seed=42, auto_detect=True)
    assert rot.is_enabled is True
    X_rot = rot.transform(X, num_threads=2)

    q4 = deglib.make_scalar_quantizer_uint4(X_rot)
    q_feats = q4.quantize(X_rot, num_threads=2)
    target_space = deglib.FloatSpace.create(dim=dim, metric=deglib.Metric.Uint4_L2)
    ro_graph = graph.to_readonly(target_space, q_feats, transformations=[rot, q4])

    rerank_space = deglib.FloatSpace.create(dim=dim, metric=deglib.Metric.FP16_InnerProduct)
    rq = deglib.make_scalar_quantizer_fp16(X)
    rf = rq.quantize(X, num_threads=2)
    searcher = deglib.search.create_searcher(
        ro_graph, rerank_space=rerank_space, rerank_data=rf, rerank_transformations=[rq]
    )

    q = X[0]
    res = searcher.search(q, k=10, eps_or_ef=32.0, rerank_factor=1.5, threads=1, return_distances=False)
    assert 0 in res.tolist()


def test_quantizer_pipeline_thresholds_and_search():
    def make_with_outlier(spike: float) -> np.ndarray:
        rng = np.random.default_rng(123)
        X = rng.standard_normal((1000, 128), dtype=np.float32)
        X[:, 0] -= spike
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    # Isotropic data -> kurtosis ~ 0 -> all pipelines omit rotation (length 1)
    X_iso = make_with_outlier(0.0)
    for qname in ("uint4", "int4", "uint7", "uint8", "int8"):
        assert len(deglib.make_quantizer_pipeline(X_iso, qname, use_rerank=False)) == 1
        assert len(deglib.make_quantizer_pipeline(X_iso, qname, use_rerank=True)) == 1

    # Moderate outlier (0.8 < k < 1.85): uint4 & uint7 rotate (no rerank), uint8/int8 do not;
    # with use_rerank=True, uint7 also omits rotation.
    X_mid = make_with_outlier(4.0)
    p_u4 = deglib.make_quantizer_pipeline(X_mid, "uint4", use_rerank=False)
    assert len(p_u4) == 2
    assert len(deglib.make_quantizer_pipeline(X_mid, deglib.Metric.Uint4_L2, use_rerank=False)) == 2
    assert isinstance(p_u4[0], deglib.FastHadamardRotation)
    assert isinstance(p_u4[1], deglib.ScalarQuantizerUint4)
    assert 0.8 < p_u4[0].excess_kurtosis < 1.85
    assert len(deglib.make_quantizer_pipeline(X_mid, "uint7", use_rerank=False)) == 2
    assert len(deglib.make_quantizer_pipeline(X_mid, "uint7", use_rerank=True)) == 1
    assert len(deglib.make_quantizer_pipeline(X_mid, "uint8", use_rerank=False)) == 1
    assert len(deglib.make_quantizer_pipeline(X_mid, "int8", use_rerank=False)) == 1

    # Medium-high outlier (5.0 < k < 25.0): with use_rerank=True, uint7 & uint8 rotate, int8 does not.
    X_high = make_with_outlier(6.0)
    assert len(deglib.make_quantizer_pipeline(X_high, "int8", use_rerank=False)) == 2
    assert len(deglib.make_quantizer_pipeline(X_high, "uint7", use_rerank=True)) == 2
    assert len(deglib.make_quantizer_pipeline(X_high, "uint8", use_rerank=True)) == 2
    assert len(deglib.make_quantizer_pipeline(X_high, "int8", use_rerank=True)) == 1

    # Extreme outlier (k > 25.0): int8 also rotates with use_rerank=True.
    X_extreme = make_with_outlier(15.0)
    p_ext = deglib.make_quantizer_pipeline(X_extreme, "int8", use_rerank=True)
    assert len(p_ext) == 2
    assert p_ext[0].excess_kurtosis > 25.0

    # End-to-end searcher with pipeline [FastHadamardRotation, ScalarQuantizerUint4]
    sub = X_extreme[:200]
    graph = deglib.builder.build_from_data(
        sub, edges_per_vertex=16, metric=deglib.Metric.FP32_InnerProduct, thread_count=2
    )
    pipeline = deglib.make_quantizer_pipeline(sub, "uint4", use_rerank=False, num_threads=2)
    assert len(pipeline) == 2
    q_feats = deglib.transformation.apply_database(pipeline, sub)
    target_space = deglib.FloatSpace.create(dim=128, metric=deglib.Metric.Uint4_L2)
    ro_graph = graph.to_readonly(target_space, q_feats, transformations=pipeline)
    searcher = deglib.search.create_searcher(ro_graph)
    res = searcher.search(sub[0], k=5, eps_or_ef=32.0, threads=1, return_distances=False)
    assert 0 in res.tolist()

    # EVP via make_quantizer_pipeline + self-transforming to_readonly (no custom_features buffer)
    p_evp = deglib.transformation.make_quantizer_pipeline(sub, "evp", num_threads=2)
    assert len(p_evp) == 1
    assert isinstance(p_evp[0], deglib.transformation.EvpQuantizer)
    assert len(deglib.transformation.make_quantizer_pipeline(sub, deglib.Metric.EVP_InnerProduct, num_threads=2)) == 1
    ro_evp = graph.to_readonly(transformations=p_evp)
    searcher_evp = deglib.search.create_searcher(ro_evp)
    res_evp = searcher_evp.search(sub[0], k=5, eps_or_ef=32.0, threads=1, return_distances=False)
    assert 0 in res_evp.tolist()
