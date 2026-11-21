import numpy as np
import pytest
import deglib
from deglib import Metric, create_searcher


def test_searcher_direct_f32():
    np.random.seed(42)
    dim = 16
    n = 100
    data = np.random.randn(n, dim).astype(np.float32)

    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)
    searcher = create_searcher(graph)

    query = data[0]
    res = searcher.search(query, k=5, eps_or_ef=0.1)
    assert isinstance(res, deglib.SearchResults)
    assert len(res) == 5
    assert res.distances is None
    assert res.graph_distance_count > 0
    assert res.graph_hop_count > 0
    assert res.rerank_distance_count == 0
    assert res.indices[0] == 0

    # With return_distances
    res_with_dist = searcher.search(query, k=5, eps_or_ef=0.1, return_distances=True)
    indices, distances = res_with_dist.indices, res_with_dist.distances
    assert len(indices) == 5
    assert len(distances) == 5
    assert indices[0] == 0
    assert distances[0] == pytest.approx(0.0, abs=1e-5)
    # Check ascending order
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1]

    # With unsorted
    res_unsorted = searcher.search(query, k=5, eps_or_ef=0.1, return_distances=True, unsorted=True)
    indices_unsorted, distances_unsorted = res_unsorted.indices, res_unsorted.distances
    assert len(indices_unsorted) == 5
    assert set(indices_unsorted) == set(indices)

    # Unified search method with 1D, (1, dim), and (dim, 1) queries
    res_1xDim = searcher.search(query.reshape(1, -1), k=5, eps_or_ef=0.1).indices
    assert res_1xDim.shape == (5,)
    assert res_1xDim[0] == 0

    res_Dimx1 = searcher.search(query.reshape(-1, 1), k=5, eps_or_ef=0.1).indices
    assert res_Dimx1.shape == (5,)
    assert res_Dimx1[0] == 0

    # Unified search method with 2D batch queries
    res_batch_unified = searcher.search(data[:10], k=5, eps_or_ef=0.1, threads=2).indices
    assert res_batch_unified.shape == (10, 5)
    assert res_batch_unified[0, 0] == 0

    batch_res = searcher.search(data[:10], k=5, eps_or_ef=0.1, threads=2, return_distances=True)
    indices_batch, dists_batch = batch_res.indices, batch_res.distances
    assert batch_res.graph_distance_count > 0
    assert batch_res.graph_hop_count > 0
    assert batch_res.rerank_distance_count == 0
    assert indices_batch.shape == (10, 5)
    assert dists_batch.shape == (10, 5)
    assert indices_batch[0, 0] == 0
    assert dists_batch[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_searcher_quantized_int8_with_rerank():
    np.random.seed(42)
    dim = 32
    n = 200
    data = np.random.randn(n, dim).astype(np.float32)

    # 1. Base graph in FP32
    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)

    # 2. INT8 Quantization
    quantizer = deglib.transformation.make_scalar_quantizer_int8(data)
    int8_data = quantizer.transform_database(data)
    ro_graph = graph.to_readonly(Metric.Int8_L2, int8_data, transformations=[quantizer])

    # 3. Refine features in FP16 with calibrated ScalarQuantizerFP16
    fp16_quantizer = deglib.transformation.make_scalar_quantizer_fp16(data)
    base_fp16 = fp16_quantizer.transform_database(data)

    # 4. Create Searcher
    searcher = create_searcher(
        graph=ro_graph,
        rerank_metric=Metric.FP16_L2,
        rerank_data=base_fp16,
        rerank_transformations=[fp16_quantizer],
    )

    query = data[5]
    res_obj = searcher.search(query, k=5, eps_or_ef=0.2, rerank_factor=1.5)
    assert res_obj.graph_distance_count > 0
    assert res_obj.rerank_distance_count > 0
    assert res_obj.graph_hop_count > 0
    res = res_obj.indices
    assert len(res) == 5
    assert res[0] == 5

    # With return_distances
    res_dist_obj = searcher.search(query, k=5, eps_or_ef=0.2, rerank_factor=1.5, return_distances=True)
    indices, distances = res_dist_obj.indices, res_dist_obj.distances
    assert indices[0] == 5
    assert distances[0] == pytest.approx(0.0, abs=1e-3)
    for i in range(1, len(distances)):
        assert distances[i] >= distances[i - 1]

    # Batch search
    res_batch_obj = searcher.search(data[:5], k=5, eps_or_ef=32, rerank_factor=1.5, threads=2)
    assert res_batch_obj.rerank_distance_count > 0
    res_batch = res_batch_obj.indices
    assert res_batch.shape == (5, 5)
    for i in range(5):
        assert res_batch[i, 0] == i


def test_searcher_quantized_int8_optimize_without_reranker():
    np.random.seed(42)
    dim = 32
    n = 200
    data = np.random.randn(n, dim).astype(np.float32)

    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)
    quantizer = deglib.transformation.make_scalar_quantizer_int8(data)
    int8_data = quantizer.transform_database(data)
    ro_graph = graph.to_readonly(Metric.Int8_L2, int8_data, transformations=[quantizer])

    searcher = create_searcher(graph=ro_graph)
    searcher.optimize(n_clusters=16, n_iter=3, sample_size=0)

    query = data[5]
    res = searcher.search(query, k=5, eps_or_ef=0.2, rerank_factor=1.0).indices
    assert len(res) == 5
    assert res[0] == 5


def test_searcher_quantized_uint8_with_rerank():
    np.random.seed(42)
    dim = 32
    n = 200
    data = np.random.uniform(0.0, 10.0, size=(n, dim)).astype(np.float32)

    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)

    quantizer = deglib.transformation.make_scalar_quantizer_uint8(data)
    uint8_data = quantizer.transform_database(data)
    ro_graph = graph.to_readonly(Metric.Uint8_L2, uint8_data, transformations=[quantizer])

    searcher = create_searcher(
        graph=ro_graph,
        rerank_metric=Metric.FP32_L2,
        rerank_data=data,
    )

    query = data[12]
    res = searcher.search(query, k=5, eps_or_ef=0.2, rerank_factor=1.5).indices
    assert len(res) == 5
    assert res[0] == 12

    res_dist = searcher.search(query, k=5, eps_or_ef=0.2, rerank_factor=1.5, return_distances=True)
    indices, distances = res_dist.indices, res_dist.distances
    assert indices[0] == 12
    assert distances[0] == pytest.approx(0.0, abs=1e-5)


def test_searcher_evp_quantizer():
    np.random.seed(42)
    dim = 64
    n = 200
    data = np.random.randn(n, dim).astype(np.float32)
    # L2 normalize for MIPS / InnerProduct
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_InnerProduct, optimize=False)

    # 16 non-zeros for EVP, shared by database and query quantization
    quantizer = deglib.transformation.make_evp_quantizer(data, non_zeros=16)
    evp_data = quantizer.transform_database(data)
    ro_graph = graph.to_readonly(Metric.EVP_InnerProduct, evp_data, transformations=[quantizer])

    searcher = create_searcher(
        graph=ro_graph,
        rerank_metric=Metric.FP32_InnerProduct,
        rerank_data=data,
    )

    query = data[3]
    res = searcher.search(query, k=3, eps_or_ef=0.3, rerank_factor=2.0).indices
    assert len(res) == 3
    assert res[0] == 3

    # FP16 query
    query_fp16 = deglib.distances.floats_to_fp16(query)
    r_fp16 = searcher.search(query_fp16, k=3, eps_or_ef=0.3, rerank_factor=2.0, return_distances=True)
    res_fp16, dists_fp16 = r_fp16.indices, r_fp16.distances
    assert len(res_fp16) == 3
    assert res_fp16[0] == 3
    assert dists_fp16[0] == pytest.approx(0.0, abs=1e-3)


def test_searcher_quantized_without_reranker():
    np.random.seed(42)
    dim = 8
    n = 100
    data = np.random.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)

    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)

    quantizer = deglib.transformation.make_scalar_quantizer_int8(data)
    int8_data = quantizer.transform_database(data)
    ro_graph = graph.to_readonly(Metric.Int8_L2, int8_data, transformations=[quantizer])

    searcher = create_searcher(graph=ro_graph)
    query = data[5]
    res = searcher.search(query, k=5, eps_or_ef=0.2).indices
    assert len(res) == 5
    assert res[0] == 5


def test_searcher_optimize_sets_kmeans_entries():
    np.random.seed(42)
    dim = 16
    n = 200
    data = np.random.randn(n, dim).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    labels = np.random.permutation(n).astype(np.uint32)

    graph = deglib.build(data, labels=labels, edges_per_vertex=16, metric=Metric.FP32_InnerProduct, optimize=False)
    searcher = create_searcher(graph)

    searcher.optimize(n_clusters=8, n_iter=5, sample_size=100)

    res = searcher.search(data[0], k=5, eps_or_ef=0.3).indices
    assert len(res) == 5
    assert res[0] == labels[0]


def test_searcher_eps_or_ef_dual_modes():
    np.random.seed(42)
    dim = 16
    n = 100
    data = np.random.randn(n, dim).astype(np.float32)

    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)
    searcher = create_searcher(graph)

    query = data[0]

    # Mode 1: eps < 1.0 (relative margin exploration)
    res_eps = searcher.search(query, k=5, eps_or_ef=0.1).indices
    assert len(res_eps) == 5
    assert res_eps[0] == 0

    # Mode 2: ef >= 1.0 (fixed pool exploration)
    res_ef = searcher.search(query, k=5, eps_or_ef=64).indices
    assert len(res_ef) == 5
    assert res_ef[0] == 0

    # Mode 2 batch with ef
    batch_ef = searcher.search(data[:5], k=5, eps_or_ef=64, threads=2).indices
    assert batch_ef.shape == (5, 5)
    for i in range(5):
        assert batch_ef[i, 0] == i


def test_searcher_rerank_insertion_order_with_permuted_labels():
    """rerank_data is in the graph's insertion order (internal id space), independent of labels.

    Builds a graph whose labels differ from insertion order (internal slot j -> label perm[j]) and
    attaches a Reranker with rerank_data in insertion order. Self-matches must resolve to the
    query's own external label, proving the reranker is scored on internal ids and the result is
    remapped internal->label.
    """
    np.random.seed(7)
    dim = 32
    n = 300
    data = np.random.randn(n, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)

    # insertion slot j -> feature data[j], external label perm[j]
    perm = np.random.permutation(n).astype(np.uint32)
    graph = deglib.build(
        data, labels=perm, edges_per_vertex=24, metric=Metric.FP32_InnerProduct, seed=7, optimize=False
    )

    # 1. Explicit rerank_order="internal" with insertion-ordered vectors
    fp16_quantizer = deglib.transformation.make_scalar_quantizer_fp16(data)
    rerank_data_internal = np.ascontiguousarray(fp16_quantizer.transform_database(data))

    searcher = create_searcher(
        graph=graph,
        rerank_metric=Metric.FP16_InnerProduct,
        rerank_data=rerank_data_internal,
        rerank_order="internal",
        rerank_transformations=[fp16_quantizer],
    )

    for j in (0, 5, 17, 99):
        res = searcher.search(data[j], k=5, eps_or_ef=128, rerank_factor=1.5).indices
        assert int(perm[j]) in set(int(x) for x in res)
        assert res[0] == perm[j]

    # 2. Default rerank_order="label" with vectors indexed by external label (rerank_data[perm[j]] = data[j])
    rerank_data_by_label = np.empty_like(data)
    rerank_data_by_label[perm] = data
    searcher_label = graph.to_searcher(
        metric=Metric.Uint8_L2,
        rerank_metric=Metric.FP16_InnerProduct,
        rerank_data=rerank_data_by_label,
    )
    for j in (0, 5, 17, 99):
        res = searcher_label.search(data[j], k=5, eps_or_ef=128, rerank_factor=1.5).indices
        assert res[0] == perm[j]


def test_searcher_normalize_query():
    np.random.seed(42)
    dim = 32
    n = 100
    # Create unnormalized vectors
    data = (np.random.randn(n, dim) * 5.0).astype(np.float32)
    # Base graph uses normalized vectors with InnerProduct metric
    norm_data = data / np.linalg.norm(data, axis=1, keepdims=True)
    graph = deglib.build(norm_data, edges_per_vertex=16, metric=Metric.FP32_InnerProduct, optimize=False)

    # Searcher with normalize_query=True
    searcher = create_searcher(graph, normalize_query=True)
    assert searcher.normalize_query is True

    # Search with unnormalized query
    raw_query = data[10] * 3.5
    r_single = searcher.search(raw_query, k=5, eps_or_ef=0.2, return_distances=True)
    res, dist = r_single.indices, r_single.distances
    assert res[0] == 10
    # For normalized inner product (1 - <q, v>), identical directions yield distance ~ 0
    assert dist[0] == pytest.approx(0.0, abs=1e-5)

    # Batch search with normalize_query=True
    raw_batch = data[:5] * 2.7
    r_batch = searcher.search(raw_batch, k=5, eps_or_ef=0.2, return_distances=True, threads=2)
    res_batch, dist_batch = r_batch.indices, r_batch.distances
    for i in range(5):
        assert res_batch[i, 0] == i
        assert dist_batch[i, 0] == pytest.approx(0.0, abs=1e-5)

    # Toggle normalize_query setter
    searcher.normalize_query = False
    assert searcher.normalize_query is False


def test_searcher_single_query_small_ef_exact_sizing():
    """When ef < k on a single query, the result array must be trimmed to found count without padding."""
    np.random.seed(42)
    dim = 8
    n = 100
    data = np.random.randn(n, dim).astype(np.float32)
    graph = deglib.build(data, edges_per_vertex=8, metric=Metric.FP32_L2, optimize=False)
    searcher = create_searcher(graph)

    # Ask for k=50 with small ef=4 (graph explore may expand fewer than 50 elements)
    res = searcher.search(data[0], k=50, eps_or_ef=4).indices
    # Result must not contain uint32 max padding and shape must equal actual found (found < 50)
    assert len(res) < 50
    assert np.all(res != np.iinfo(np.uint32).max)

    # With return_distances=True
    res_with_dist = searcher.search(data[0], k=50, eps_or_ef=4, return_distances=True)
    res_idx, res_dist = res_with_dist.indices, res_with_dist.distances
    assert len(res_idx) == len(res_dist)
    assert len(res_idx) < 50
    assert np.all(res_idx != np.iinfo(np.uint32).max)
    assert not np.any(np.isnan(res_dist))


def test_searcher_max_distance_computation_count():
    """Verify max_distance_computation_count bounds single and batch Searcher queries."""
    np.random.seed(42)
    dim = 8
    n = 200
    data = np.random.randn(n, dim).astype(np.float32)
    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)
    searcher = create_searcher(graph)

    # Unlimited (0) returns full k=10 and exact match 0
    res_full = searcher.search(data[0], k=10, eps_or_ef=64, max_distance_computation_count=0)
    assert len(res_full.indices) == 10
    assert res_full.indices[0] == 0

    # Tight budget (5) limits returned candidates and graph_distance_count in both eps and ef modes
    res_bounded_eps = searcher.search(data[0], k=10, eps_or_ef=0.2, max_distance_computation_count=5)
    assert 1 <= len(res_bounded_eps.indices) <= 5
    assert 1 <= res_bounded_eps.graph_distance_count <= 5

    res_bounded_ef = searcher.search(data[0], k=10, eps_or_ef=64, max_distance_computation_count=5)
    assert 1 <= len(res_bounded_ef.indices) <= 5
    assert 1 <= res_bounded_ef.graph_distance_count <= 5

    # Batch search with tight budget (max_distance_computation_count=5)
    batch_full = searcher.search(data[:4], k=10, eps_or_ef=64, threads=2, max_distance_computation_count=0)
    batch_bounded = searcher.search(data[:4], k=10, eps_or_ef=64, threads=2, max_distance_computation_count=5)
    assert batch_bounded.indices.shape == (4, 10)
    assert batch_bounded.graph_distance_count <= 4 * 5
    assert np.array_equal(batch_bounded.indices[0, : len(res_bounded_ef.indices)], res_bounded_ef.indices)
    assert not np.array_equal(batch_bounded.indices[0, :5], batch_full.indices[0, :5])


def test_to_searcher_self_extraction_rerank_and_validation():
    """graph.to_searcher extracts & quantizes rerank data from an FP32 graph when rerank_data=None."""
    rng = np.random.default_rng(12)
    data = rng.normal(size=(120, 32)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)

    fp32_graph = deglib.build(data, edges_per_vertex=16, extend_k=32, metric=Metric.FP32_L2, optimize=False)

    # Self-extraction prior to Uint8_L2 graph quantization
    searcher = fp32_graph.to_searcher(
        metric=Metric.Uint8_L2,
        rerank_metric=Metric.FP16_InnerProduct,
        rerank_data=None,
        optimize=False,
    )
    assert searcher.graph.get_metric() == Metric.Uint8_L2
    res = searcher.search(data[7], k=5, eps_or_ef=32, rerank_factor=2.0).indices
    assert int(res[0]) == 7

    # rerank_metric=None with rerank_data supplied must raise ValueError
    with pytest.raises(ValueError, match="rerank_metric"):
        fp32_graph.to_searcher(rerank_data=data, rerank_metric=None)


def test_searcher_unsorted_and_return_distances():
    """Verify return_distances=False/True and unsorted=True on single & batch Searcher calls."""
    rng = np.random.default_rng(99)
    data = rng.normal(size=(80, 16)).astype(np.float32)
    graph = deglib.build(data, edges_per_vertex=16, metric=Metric.FP32_L2, optimize=False)
    searcher = graph.to_searcher(optimize=False)

    r_sorted = searcher.search(data[0], k=5, eps_or_ef=32, return_distances=True, unsorted=False)
    sorted_idx, sorted_dist = r_sorted.indices, r_sorted.distances
    assert np.all(np.diff(sorted_dist) >= -1e-6)

    res_only = searcher.search(data[0], k=5, eps_or_ef=32, return_distances=False, unsorted=True)
    assert res_only.distances is None
    unsorted_only = res_only.indices
    assert isinstance(unsorted_only, np.ndarray)
    assert unsorted_only.shape == (5,)
    assert set(unsorted_only.tolist()) == set(sorted_idx.tolist())

    r_unsorted = searcher.search(data[0], k=5, eps_or_ef=32, return_distances=True, unsorted=True)
    unsorted_idx, unsorted_dist = r_unsorted.indices, r_unsorted.distances
    assert unsorted_idx.shape == (5,)
    assert unsorted_dist.shape == (5,)
    assert set(unsorted_idx.tolist()) == set(sorted_idx.tolist())

    batch_only_res = searcher.search(data[:4], k=5, eps_or_ef=32, threads=2, return_distances=False, unsorted=True)
    assert batch_only_res.distances is None
    batch_only = batch_only_res.indices
    assert isinstance(batch_only, np.ndarray)
    assert batch_only.shape == (4, 5)
    assert set(batch_only[0].tolist()) == set(sorted_idx.tolist())


def test_shared_reranker_across_multiple_quantized_searchers():
    """Build one graph and one Reranker once, then share that Reranker across multiple quantized Searchers."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(150, 32)).astype(np.float32)

    base_graph = deglib.build(
        data,
        edges_per_vertex=16,
        extend_k=32,
        metric=Metric.FP32_L2,
        presort=True,
        optimize=False,
    )

    shared_reranker = base_graph.to_reranker(Metric.FP16_L2, data=data, order="label", optimize=True)
    assert shared_reranker.get_num_base_vectors() == len(data)

    searchers = {
        "Uint4_L2": base_graph.to_searcher(metric="Uint4_L2", reranker=shared_reranker, optimize=True),
        "Int8_L2": base_graph.to_searcher(metric="Int8_L2", reranker=shared_reranker, optimize=True),
        "Uint8_L2": base_graph.to_searcher(metric="Uint8_L2", reranker=shared_reranker, optimize=True),
    }

    assert searchers["Uint4_L2"].graph.get_metric() == Metric.Uint4_L2
    assert searchers["Int8_L2"].graph.get_metric() == Metric.Int8_L2
    assert searchers["Uint8_L2"].graph.get_metric() == Metric.Uint8_L2

    for q_idx in (0, 13, 77):
        for s in searchers.values():
            res = s.search(data[q_idx], k=5, eps_or_ef=48, rerank_factor=2.0).indices
            assert int(res[0]) == q_idx

    # Self-extraction via to_reranker(metric) with data=None
    extracted_reranker = base_graph.to_reranker(Metric.FP16_L2)
    s_extracted = base_graph.to_searcher(metric="Int8_L2", reranker=extracted_reranker, optimize=False)
    assert int(s_extracted.search(data[5], k=5, eps_or_ef=48, rerank_factor=2.0).indices[0]) == 5

    # Cannot combine `reranker` with `rerank_metric` or `rerank_data`
    with pytest.raises(ValueError, match="Cannot pass 'reranker'"):
        base_graph.to_searcher(reranker=shared_reranker, rerank_metric=Metric.FP16_L2)
    with pytest.raises(ValueError, match="Cannot pass 'reranker'"):
        deglib.Searcher(base_graph, reranker=shared_reranker, rerank_data=data)


def test_search_results_zero_copy_lifetime_and_telemetry():
    import gc

    rng = np.random.default_rng(7)
    data = rng.standard_normal((80, 16)).astype(np.float32)
    graph = deglib.build(data, edges_per_vertex=8, metric=Metric.FP32_L2, optimize=False)
    reranker = graph.to_reranker(Metric.FP16_L2)
    searcher = graph.to_searcher(metric=Metric.Int8_L2, reranker=reranker, optimize=False)

    res = searcher.search(data[3], k=5, eps_or_ef=32.0, rerank_factor=2.0, return_distances=True)
    assert isinstance(res, deglib.SearchResults)
    assert res.graph_distance_count > 0
    assert res.rerank_distance_count > 0
    assert res.graph_hop_count > 0

    idx = res.indices
    dist = res.distances
    assert idx.shape == (5,)
    assert dist.shape == (5,)
    # Views point into SearchResults storage via capsule owner, so they do not own data directly
    assert not idx.flags["OWNDATA"]
    assert not dist.flags["OWNDATA"]
    idx_copy = idx.copy()
    dist_copy = dist.copy()

    # Dropping the SearchResults wrapper must keep arrays valid through the owner capsule
    del res
    gc.collect()
    np.testing.assert_array_equal(idx, idx_copy)
    np.testing.assert_allclose(dist, dist_copy)

    # Same lifetime contract for batch SearchResults with return_distances=False
    batch_res = searcher.search(data[:4], k=5, eps_or_ef=32.0, threads=2, return_distances=False)
    assert batch_res.distances is None
    assert batch_res.graph_distance_count > 0
    assert batch_res.graph_hop_count > 0
    b_idx = batch_res.indices
    assert not b_idx.flags["OWNDATA"]
    b_copy = b_idx.copy()
    del batch_res
    gc.collect()
    np.testing.assert_array_equal(b_idx, b_copy)
