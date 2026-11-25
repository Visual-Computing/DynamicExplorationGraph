import numpy as np
import pytest

import deglib


@pytest.mark.parametrize("dim", [64, 96, 128, 200])
def test_rotation_orthogonality_preserves_distances(dim):
    rng = np.random.default_rng(123)
    X = rng.standard_normal((50, dim), dtype=np.float32)
    rot = deglib.make_fast_hadamard_rotation(X, seed=42, auto_detect=False)
    assert rot.is_enabled is True
    Y = rot.transform_database(X)
    assert Y.shape == X.shape
    assert Y.dtype == np.float32

    np.testing.assert_allclose(np.linalg.norm(Y, axis=1), np.linalg.norm(X, axis=1), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(Y @ Y.T, X @ X.T, rtol=1e-4, atol=1e-4)


def test_rotation_auto_detect_isotropic_vs_outliers():
    rng = np.random.default_rng(7)
    X_iso = rng.standard_normal((2000, 128), dtype=np.float32)
    X_iso /= np.linalg.norm(X_iso, axis=1, keepdims=True)

    rot_iso = deglib.make_fast_hadamard_rotation(X_iso, auto_detect=True)
    assert rot_iso.is_enabled is False
    np.testing.assert_array_equal(rot_iso.transform_database(X_iso[:10]), X_iso[:10])

    X_out = X_iso.copy()
    X_out[:, 0] -= 0.65
    X_out /= np.linalg.norm(X_out, axis=1, keepdims=True)
    rot_out = deglib.make_fast_hadamard_rotation(X_out, auto_detect=True)
    assert rot_out.is_enabled is True
    assert rot_out.crest_factor > 9.5 or rot_out.excess_kurtosis > 2.0


def test_rotation_chained_with_uint4_in_searcher():
    rng = np.random.default_rng(21)
    dim = 128
    n = 300
    X = rng.standard_normal((n, dim), dtype=np.float32)
    X[:, 0] -= 15.0
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    graph = deglib.build(X, edges_per_vertex=16, metric=deglib.Metric.FP32_InnerProduct, threads=2, optimize=False)

    rot = deglib.make_fast_hadamard_rotation(X, seed=42, auto_detect=True)
    assert rot.is_enabled is True
    X_rot = rot.transform_database(X, threads=2)

    q4 = deglib.make_scalar_quantizer_uint4(X_rot)
    q_feats = q4.transform_database(X_rot, threads=2)
    ro_graph = graph.with_features(
        q_feats,
        metric=deglib.Metric.Uint4_L2,
        transformations=[rot, q4],
        threads=2,
    )

    rq = deglib.make_scalar_quantizer_fp16(X)
    rf = rq.transform_database(X, threads=2)
    reranker = deglib.Reranker(deglib.Metric.FP16_InnerProduct, rf, transformations=[rq], threads=2)
    searcher = ro_graph.to_searcher(
        reranker=reranker,
        optimize=False,
    )
    q = X[0]
    res = searcher.search(q, k=10, eps_or_ef=32.0, rerank_factor=1.5, threads=1, return_distances=False)
    assert 0 in res.indices.tolist()


def test_quantizer_pipeline_thresholds_and_search():
    def make_with_outlier(spike: float) -> np.ndarray:
        rng = np.random.default_rng(123)
        X = rng.standard_normal((1000, 128), dtype=np.float32)
        X[:, 0] -= spike
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    X_iso = make_with_outlier(0.0)
    for qname in ("uint4", "int4", "uint7", "uint8", "int8"):
        assert len(deglib.make_quantizer_pipeline(X_iso, qname, use_rerank=False)) == 1
        assert len(deglib.make_quantizer_pipeline(X_iso, qname, use_rerank=True)) == 1

    X_mid = make_with_outlier(4.0)
    p_u4 = deglib.make_quantizer_pipeline(X_mid, "uint4", use_rerank=False)
    assert len(p_u4) == 2
    assert len(deglib.make_quantizer_pipeline(X_mid, deglib.Metric.Uint4_L2, use_rerank=False)) == 2
    assert p_u4[0].name == "FastHadamardRotation"
    assert p_u4[1].name == "ScalarQuantizerUint4"
    assert 0.8 < p_u4[0].excess_kurtosis < 1.85
    assert len(deglib.make_quantizer_pipeline(X_mid, "uint7", use_rerank=False)) == 2
    assert len(deglib.make_quantizer_pipeline(X_mid, "uint7", use_rerank=True)) == 1
    assert len(deglib.make_quantizer_pipeline(X_mid, "uint8", use_rerank=False)) == 1
    assert len(deglib.make_quantizer_pipeline(X_mid, "int8", use_rerank=False)) == 1

    X_high = make_with_outlier(6.0)
    assert len(deglib.make_quantizer_pipeline(X_high, "int8", use_rerank=False)) == 2
    assert len(deglib.make_quantizer_pipeline(X_high, "uint7", use_rerank=True)) == 2
    assert len(deglib.make_quantizer_pipeline(X_high, "uint8", use_rerank=True)) == 2
    assert len(deglib.make_quantizer_pipeline(X_high, "int8", use_rerank=True)) == 1

    X_extreme = make_with_outlier(15.0)
    p_ext = deglib.make_quantizer_pipeline(X_extreme, "int8", use_rerank=True)
    assert len(p_ext) == 2
    assert p_ext[0].excess_kurtosis > 25.0

    sub = X_extreme[:200]
    graph = deglib.build(sub, edges_per_vertex=16, metric=deglib.Metric.FP32_InnerProduct, threads=2, optimize=False)
    pipeline = deglib.make_quantizer_pipeline(sub, "uint4", use_rerank=False, threads=2)
    assert len(pipeline) == 2
    q_feats = deglib.apply_on_database(pipeline, sub)
    ro_graph = graph.with_features(q_feats, metric=deglib.Metric.Uint4_L2, transformations=pipeline)
    searcher = ro_graph.to_searcher(optimize=False)
    res = searcher.search(sub[0], k=5, eps_or_ef=32.0, threads=1, return_distances=False)
    assert 0 in res.indices.tolist()

    # EVP via make_quantizer_pipeline + self-transforming with_transformations()
    p_evp = deglib.make_quantizer_pipeline(sub, "evp", threads=2)
    assert len(p_evp) == 1
    assert p_evp[0].name == "EvpQuantizer"
    assert len(deglib.make_quantizer_pipeline(sub, deglib.Metric.EVP_InnerProduct, threads=2)) == 1
    ro_evp = graph.with_transformations(p_evp)
    searcher_evp = ro_evp.to_searcher(optimize=False)
    res_evp = searcher_evp.search(sub[0], k=5, eps_or_ef=32.0, threads=1, return_distances=False)
    assert 0 in res_evp.indices.tolist()
