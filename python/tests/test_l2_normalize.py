import numpy as np
import pytest

import deglib
from deglib.transformation import apply_on_query, make_l2_normalize_transform


def test_l2_normalize_transform_1d_and_2d():
    dim = 16
    norm_transform = make_l2_normalize_transform(dim)
    assert norm_transform.name == "L2NormalizeTransform"
    assert norm_transform.input_dim == dim
    assert norm_transform.output_dim == dim
    assert norm_transform.is_fitted

    rng = np.random.default_rng(42)
    q = rng.normal(size=(dim,)).astype(np.float32) * 5.0
    q_norm = apply_on_query([norm_transform], q)
    assert q_norm.shape == (dim,)
    assert q_norm.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(q_norm), 1.0, atol=1e-6)
    np.testing.assert_allclose(q_norm, q / np.linalg.norm(q), atol=1e-6)

    q_batch = rng.normal(size=(10, dim)).astype(np.float32) * 3.0
    q_batch_norm = apply_on_query([norm_transform], q_batch)
    assert q_batch_norm.shape == (10, dim)
    np.testing.assert_allclose(np.linalg.norm(q_batch_norm, axis=1), np.ones(10), atol=1e-6)

    # Also test passing sample array to make_l2_normalize_transform
    norm_from_arr = make_l2_normalize_transform(q_batch)
    assert norm_from_arr.input_dim == dim
    with pytest.raises(ValueError):
        make_l2_normalize_transform(0)


def test_l2_normalize_with_graph_search():
    dim = 16
    n = 100
    rng = np.random.default_rng(123)
    data = rng.normal(size=(n, dim)).astype(np.float32) * 4.0
    data_norm = data / np.linalg.norm(data, axis=1, keepdims=True)

    graph = deglib.build(
        data_norm, metric=deglib.Metric.FP32_InnerProduct, edges_per_vertex=8, presort=False, optimize=False
    )

    searcher = graph.to_searcher(normalize_query=True, optimize=True)
    raw_query = data[0] * 10.0
    res = searcher.search(raw_query, k=5, eps_or_ef=0.2)
    assert res.indices[0] == 0

    graph = graph.with_features(
        data_norm,
        metric=deglib.Metric.FP32_InnerProduct,
        transformations=[make_l2_normalize_transform(data)],
    )
    res_g = graph.search(raw_query, eps_or_ef=0.2, k=5)
    assert res_g.indices[0] == 0


def test_build_ip_auto_switches_to_normalized_on_unit_vectors():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((120, 16), dtype=np.float32)
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    unit = np.ascontiguousarray(unit, dtype=np.float32)

    m_unit, x_unit = deglib.transformation.resolve_spatial_metric(unit, "ip")
    assert m_unit == deglib.Metric.FP32_InnerProduct
    assert x_unit == []

    g_unit = deglib.build(unit, metric="ip", edges_per_vertex=8, optimization_target="LowLID")
    assert g_unit.get_metric() == deglib.Metric.FP32_InnerProduct
    assert g_unit.get_dim() == 16
    assert g_unit.get_transformations() == []

    # Unnormalized IP still uses MipsTransform during build
    unnorm = raw * rng.uniform(0.5, 3.0, size=(120, 1)).astype(np.float32)
    m_unnorm, x_unnorm = deglib.transformation.resolve_spatial_metric(unnorm, "ip")
    assert m_unnorm == deglib.Metric.FP32_L2
    assert len(x_unnorm) == 1 and x_unnorm[0].name == "MipsTransform"
