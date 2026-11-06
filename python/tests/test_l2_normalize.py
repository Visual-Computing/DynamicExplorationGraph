import numpy as np
import pytest
import deglib
from deglib.transformation import L2NormalizeTransform, make_l2_normalize_transform


def test_l2_normalize_basic():
    vecs = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    norm_transform = make_l2_normalize_transform(dim=2)

    assert norm_transform.dim == 2
    assert norm_transform.name == "L2NormalizeTransform"
    assert norm_transform.tag != 0

    res = norm_transform.transform(vecs)
    # [3, 4] -> norm 5 -> [0.6, 0.8]
    assert np.allclose(res[0], [0.6, 0.8], atol=1e-6)
    # [1, 1] -> norm sqrt(2) -> [0.70710677, 0.70710677]
    assert np.allclose(res[1], [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], atol=1e-6)
    # [0, 0] -> [0, 0]
    assert np.allclose(res[2], [0.0, 0.0], atol=1e-6)


def test_l2_normalize_serialization():
    t1 = make_l2_normalize_transform(dim=128)
    bytes_data = deglib.transformation.serialize_transformations([t1])
    deserialized = deglib.transformation.deserialize_transformations(bytes_data)

    assert len(deserialized) == 1
    assert isinstance(deserialized[0], L2NormalizeTransform)
    assert deserialized[0].dim == 128


def test_l2_normalize_in_searcher_pipeline():
    dim = 8
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, dim)).astype(np.float32)

    # Normalize data for cosine
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data_norm = data / norms

    graph = deglib.builder.build_from_data(data_norm, metric=deglib.Metric.FP32_InnerProduct, edges_per_vertex=8)

    # Searcher with L2NormalizeTransform in front of search
    norm_trans = make_l2_normalize_transform(dim=dim)
    ro_graph = graph.to_readonly(transformations=[norm_trans])
    searcher = deglib.search.create_searcher(ro_graph)

    raw_query = rng.standard_normal((dim,)).astype(np.float32) * 10.0
    # Search with raw unnormalized query directly
    res_indices = searcher.search(raw_query, k=5)
    assert len(res_indices) == 5

    # Should match searching with explicitly normalized query
    explicit_norm_query = raw_query / np.linalg.norm(raw_query)
    res_indices_exp = searcher.search(explicit_norm_query, k=5)
    np.testing.assert_array_equal(res_indices, res_indices_exp)
