import os
import pathlib
import platform
import random
import tempfile

import numpy as np
import pytest

import deglib
from deglib.distances import Metric
from deglib.search import Filter

IS_MACOS_M1 = platform.system() == "Darwin" and platform.machine() == "arm64"


def get_tmp_graph_file(samples: int, dims: int) -> pathlib.Path:
    tmpdir = os.path.join(tempfile.gettempdir(), "deglib_test")
    os.makedirs(tmpdir, exist_ok=True)

    return pathlib.Path(os.path.join(tmpdir, f"test_graph_S{samples}_D{dims}.deg"))


def get_ranking(features: np.ndarray, graph: deglib.DynamicExplorationGraph, query: np.ndarray) -> np.ndarray:
    """
    Returns the ranking for each feature vector in the graph
    """
    query = query.reshape(1, graph.get_dim())

    if features.dtype in (np.uint8, np.int8):
        features = features.astype(np.float32)
    if query.dtype in (np.uint8, np.int8):
        query = query.astype(np.float32)

    metric = graph.get_metric()
    if metric in (Metric.FP32_L2, Metric.Uint8_L2, Metric.Int8_L2):
        distances = np.sum(np.square(features - query), axis=1)
    elif metric == Metric.FP32_InnerProduct:
        distances = 1.0 - np.dot(features, query.T).flatten()
    elif metric in (Metric.Uint8_InnerProduct, Metric.Int8_InnerProduct):
        distances = -np.dot(features, query.T).flatten()
    else:
        raise ValueError(f"unknown metric: {metric}")
    return np.argsort(distances)


def _build_mutable(
    data: np.ndarray,
    edges_per_vertex: int = 16,
    metric: Metric = Metric.FP32_L2,
    optimization_target=deglib.builder.OptimizationTarget.LowLID,
    seed: int = 7,
):
    g = deglib.create_empty(
        capacity=data.shape[0],
        dim=deglib.distances._infer_dim_from_array(metric, data),
        edges_per_vertex=edges_per_vertex,
        metric=metric,
    )
    b = deglib.GraphBuilder(g, seed=seed, optimization_target=optimization_target)
    b.add_entry(np.arange(data.shape[0], dtype=np.uint32), data)
    b.build()
    return g


class Configuration:
    def __init__(
        self,
        edges_per_vertex: int,
        samples: int,
        dims: int,
        data: np.ndarray,
        graph: deglib.DynamicExplorationGraph,
        graph_path: pathlib.Path | None,
        query: np.ndarray,
        metric: Metric,
    ):
        self.edges_per_vertex = edges_per_vertex
        self.samples = samples
        self.dims = dims
        self.data = data
        self.graph = graph
        self.graph_path = graph_path
        self.query = query
        self.metric = metric

    @staticmethod
    def generate(samples, dims, metric, edges_per_vertex):
        rng = np.random.RandomState(42 + int(metric.value) * 17)
        if metric == Metric.FP32_InnerProduct:
            # normalize data
            data = rng.random_sample((samples, dims)).astype(np.float32)
            data /= np.linalg.norm(data, axis=1).reshape(-1, 1)

            query = rng.random_sample((dims,)).astype(np.float32)
            query /= np.linalg.norm(query)
        elif metric == Metric.FP32_L2:
            data = rng.normal(size=(samples, dims)).astype(np.float32)
            query = rng.normal(size=(dims,)).astype(np.float32)
        elif metric == Metric.Uint8_L2 or metric == Metric.Uint8_InnerProduct:
            data = rng.randint(0, 256, size=(samples, dims)).astype(np.uint8)
            query = rng.randint(0, 256, size=(dims,)).astype(np.uint8)
        elif metric == Metric.Int8_L2:
            data = rng.randint(0, 128, size=(samples, dims)).astype(np.int8)
            query = rng.randint(0, 128, size=(dims,)).astype(np.int8)
        elif metric == Metric.Int8_InnerProduct:
            data = rng.randint(-128, 128, size=(samples, dims)).astype(np.int8)
            query = rng.randint(-128, 128, size=(dims,)).astype(np.int8)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        size_bounded_graph = _build_mutable(
            data,
            edges_per_vertex=edges_per_vertex,
            metric=metric,
            optimization_target=deglib.builder.OptimizationTarget.LowLID,
            seed=42,
        )

        graph_path = get_tmp_graph_file(samples, dims)
        size_bounded_graph.save_graph(graph_path)
        read_only_graph = deglib.load_readonly_graph(graph_path)
        read_only_graph_converted = size_bounded_graph.to_readonly()

        return [
            Configuration(edges_per_vertex, samples, dims, data, size_bounded_graph, graph_path, query, metric),
            Configuration(edges_per_vertex, samples, dims, data, read_only_graph, None, query, metric),
            Configuration(edges_per_vertex, samples, dims, data, read_only_graph_converted, None, query, metric),
        ]

    def create_new_size_bounded_graph(self):
        return _build_mutable(self.data, edges_per_vertex=self.edges_per_vertex, metric=self.metric)

    def __repr__(self):
        return f"Conf({type(self.graph).__name__}, metric={self.metric.name})"


configurations = [
    *Configuration.generate(100, 128, Metric.FP32_L2, 16),
    *Configuration.generate(100, 128, Metric.Uint8_L2, 16),
    *Configuration.generate(100, 128, Metric.Int8_L2, 16),
    *Configuration.generate(100, 128, Metric.FP32_InnerProduct, 16),
    *Configuration.generate(100, 128, Metric.Uint8_InnerProduct, 16),
    *Configuration.generate(100, 128, Metric.Int8_InnerProduct, 16),
]

large_configurations = [
    *Configuration.generate(20_000, 2, Metric.Uint8_L2, 10),
]

mutable_configurations = [c for c in configurations if c.graph.is_mutable()]


@pytest.mark.parametrize("conf", configurations)
def test_search(conf: Configuration):
    if IS_MACOS_M1 and conf.metric == Metric.FP32_InnerProduct:
        pytest.skip("This test is skipped on macOS with M1 chip, as avx2 is not supported on m1 chip.")

    k = 10
    res = conf.graph.search(conf.query, eps_or_ef=0.2, k=k)
    assert isinstance(res, deglib.SearchResults)
    assert res.graph_distance_count > 0
    assert res.graph_hop_count > 0
    assert res.rerank_distance_count == 0
    graph_result, dists = res.indices, res.distances
    graph_result = graph_result.flatten()
    correct_result = get_ranking(conf.data, conf.graph, conf.query)[:k]

    assert graph_result.shape[-1] == k, f"expected {k} results, but got {graph_result.shape[-1]}"

    # test matches are good
    matches = set(graph_result).intersection(set(correct_result))
    assert len(matches) >= k - 2, f"expected at least {k - 2} matching results, but got only {len(matches)}"

    # test result is sorted
    last_distance = float("-inf")
    for index, distance in enumerate(dists):
        assert last_distance <= distance, (
            f"ResultSet is not sorted.\ndistance {last_distance} at index {index - 1} larger than\ndistance {distance} at index {index}"
        )


@pytest.mark.parametrize("conf", configurations)
def test_search_eps_or_ef(conf: Configuration):
    """Test both eps (< 1.0) and ef (>= 1.0) modes of eps_or_ef parameter."""
    if IS_MACOS_M1 and conf.metric == Metric.FP32_InnerProduct:
        pytest.skip("This test is skipped on macOS with M1 chip, as avx2 is not supported on m1 chip.")

    k = 10
    correct_result = get_ranking(conf.data, conf.graph, conf.query)[:k]

    # 1. eps mode (< 1.0)
    r_eps = conf.graph.search(conf.query, eps_or_ef=0.2, k=k)
    res_eps, dists_eps = r_eps.indices, r_eps.distances
    assert len(res_eps) == k
    assert len(dists_eps) == k
    assert np.all(np.diff(dists_eps) >= -1e-6)

    # 2. ef mode (>= 1.0)
    r_ef = conf.graph.search(conf.query, eps_or_ef=64.0, k=k)
    res_ef, dists_ef = r_ef.indices, r_ef.distances
    assert len(res_ef) == k
    assert len(dists_ef) == k
    assert np.all(np.diff(dists_ef) >= -1e-6)

    matches_ef = set(res_ef).intersection(set(correct_result))
    assert len(matches_ef) >= k - 2, f"Expected at least {k - 2} matches with ef=64, got {len(matches_ef)}"

    # 3. Batch search with ef mode and multiple threads
    queries = np.vstack([conf.query, conf.query])
    r_batch = conf.graph.search(queries, eps_or_ef=64.0, k=k, threads=2)
    batch_res, batch_dists = r_batch.indices, r_batch.distances
    assert r_batch.graph_distance_count >= r_ef.graph_distance_count
    assert batch_res.shape == (2, k)
    assert batch_dists.shape == (2, k)
    assert np.array_equal(batch_res[0], res_ef)


@pytest.mark.parametrize("conf", mutable_configurations)
def test_prune_non_mrng_edges(conf: Configuration):
    graph = conf.create_new_size_bounded_graph()
    deglib.optimization.prune_non_mrng_edges(graph)


@pytest.mark.parametrize("conf", configurations)
def test_threaded_search(conf: Configuration):
    k = 10
    r = conf.graph.search(conf.query, eps_or_ef=0.1, k=k)
    graph_result, dists = r.indices, r.distances
    for n_threads in range(2, 8):
        tr = conf.graph.search(conf.query, eps_or_ef=0.1, k=k, threads=n_threads)
        threaded_graph_result, threaded_dists = tr.indices, tr.distances
        assert np.all(np.equal(threaded_graph_result, graph_result)), (
            f"Threaded and non threaded results differ (n_threads={n_threads})"
        )
        assert np.allclose(threaded_dists, dists), f"Threaded and non threaded dists differ (n_threads={n_threads})"


@pytest.mark.parametrize("conf", configurations)
def test_explore(conf: Configuration):
    k = 10
    include_entry = True
    entry_vertex_index = random.randint(0, conf.samples - 1)
    res_explore = conf.graph.explore(
        entry_vertex_index, k, max_distance_computation_count=k * 10, include_entry=include_entry
    )
    assert res_explore.graph_distance_count > 0
    assert res_explore.graph_hop_count > 0
    indices, distances = res_explore.indices, res_explore.distances
    assert len(indices) == k
    assert len(distances) == k
    # check that valid (non-NaN) distances are sorted (ascending)
    valid_distances = distances[~np.isnan(distances)]
    for i in range(len(valid_distances) - 1):
        assert valid_distances[i] <= valid_distances[i + 1], "Distances should be sorted in ascending order"


@pytest.mark.parametrize("conf", configurations)
def test_get_edges_per_vertex(conf: Configuration):
    assert conf.graph.get_edges_per_vertex() == conf.edges_per_vertex


@pytest.mark.parametrize("conf", configurations)
def test_has_vertex(conf: Configuration):
    assert conf.graph.has_vertex(0)
    assert not conf.graph.has_vertex(conf.graph.size())


@pytest.mark.parametrize("conf", mutable_configurations)
def test_modify_graph(conf: Configuration):
    graph = conf.create_new_size_bounded_graph()
    # remove vertex via builder
    builder = deglib.GraphBuilder(graph)
    builder.remove_entry(graph.size() - 1)
    builder.build()
    assert graph.size() == conf.samples - 1
    # add vertex back via builder
    builder2 = deglib.GraphBuilder(graph)
    builder2.add_entry(conf.samples - 1, conf.data[-1])
    builder2.build()


@pytest.mark.parametrize("conf", mutable_configurations)
def test_load_graph(conf: Configuration):
    graph = deglib.load_readonly_graph(conf.graph_path)
    assert not graph.is_mutable()

    with pytest.raises(FileNotFoundError):
        _graph = deglib.load_readonly_graph(pathlib.Path("path") / "does" / "not" / "exist")


@pytest.mark.parametrize("conf", mutable_configurations)
def test_save_graph(tmp_path, conf):
    target_path = tmp_path / "save_path.deg"
    if target_path.is_file():
        os.remove(target_path)
    conf.graph.save_graph(target_path)
    assert target_path.is_file()
    os.remove(target_path)


@pytest.mark.parametrize("conf", mutable_configurations)
def test_convert_graph(conf: Configuration):
    rd_graph = conf.graph.to_readonly()
    assert not rd_graph.is_mutable()


@pytest.mark.parametrize("conf", large_configurations)
def test_filters(conf: Configuration):
    k = 400

    valid_labels = np.random.choice(conf.graph.size(), size=12_000, replace=False)
    results = conf.graph.search(conf.query, filter_labels=Filter(valid_labels), eps_or_ef=0.01, k=k).indices

    if not np.all(np.isin(results, valid_labels)):
        raise ValueError("Found results that should have been filtered out.")


@pytest.mark.parametrize("conf", large_configurations)
def test_small_filters(conf: Configuration):
    for n_valid in [200, 400, 600]:
        k = 400

        valid_labels = np.random.choice(conf.graph.size(), size=n_valid, replace=False)
        res = conf.graph.search(conf.query, filter_labels=Filter(valid_labels), eps_or_ef=0.01, k=k)
        results, _dists = res.indices, res.distances

        assert results.shape[-1] == k

        valid_results = results[results != np.iinfo(np.uint32).max]
        if not np.all(np.isin(valid_results, valid_labels)):
            raise ValueError("Found results that should have been filtered out.")

        if n_valid < k:
            num_unfilled = k - n_valid
            assert np.sum(results == np.iinfo(np.uint32).max) == num_unfilled
            assert np.sum(np.isnan(_dists)) == num_unfilled


@pytest.mark.parametrize("conf", configurations[:1])
def test_filter_edge_cases(conf: Configuration):
    k = 5
    # All valid labels filter
    all_labels = np.arange(conf.graph.size(), dtype=np.int32)
    results_all = conf.graph.search(conf.query, filter_labels=Filter(all_labels), eps_or_ef=0.1, k=k).indices
    assert results_all.shape[-1] == k

    # No valid labels filter with max_value specified
    no_labels = np.array([], dtype=np.int32)
    r_none = conf.graph.search(conf.query, filter_labels=Filter(no_labels, max_value=0), eps_or_ef=0.1, k=k)
    results_none, dists_none = r_none.indices, r_none.distances
    assert results_none.shape[-1] == k
    assert np.all(results_none == np.iinfo(np.uint32).max)
    assert np.all(np.isnan(dists_none))


def _build_test_graph_with_distinct_labels(samples=50, dims=16, edges_per_vertex=10):
    """Build a graph with external labels that differ from internal indices."""
    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)
    labels = np.arange(samples, dtype=np.uint32) * 10
    graph = deglib.build(
        data,
        labels=labels,
        edges_per_vertex=edges_per_vertex,
        optimization_target=deglib.builder.OptimizationTarget.LowLID,
        extend_k=10,
        extend_eps=0.001,
        seed=42,
    )
    return graph, data


def test_has_vertex_accepts_external_label():
    """has_vertex should accept external labels, not internal indices."""
    g, _ = _build_test_graph_with_distinct_labels()
    # External labels are 0, 10, 20, ...
    assert g.has_vertex(0) is True
    assert g.has_vertex(10) is True
    assert g.has_vertex(1) is False  # internal index, not external label
    assert g.has_vertex(99999) is False


def test_search_returns_external_labels():
    """DynamicExplorationGraph.search returns external labels."""
    g, data = _build_test_graph_with_distinct_labels()
    r = g.search(data[0:1], eps_or_ef=0.1, k=5)
    indices, distances = r.indices, r.distances
    assert indices.shape == (1, 5)
    # search_batch returns external labels (multiples of 10)
    for idx in indices.flatten():
        if idx != 0xFFFFFFFF:
            assert idx % 10 == 0, f"Search result {idx} should be an external label (multiple of 10)"


def test_facade_has_vertex_accepts_external_label():
    """DynamicExplorationGraph facade has_vertex should accept external labels."""
    g, _ = _build_test_graph_with_distinct_labels()
    # External labels are 0, 10, 20, ...
    assert g.has_vertex(0) is True
    assert g.has_vertex(10) is True
    assert g.has_vertex(1) is False  # internal index, not external label


def test_facade_explore_uses_external_label():
    """DynamicExplorationGraph facade explore should use external labels."""
    g, _ = _build_test_graph_with_distinct_labels()
    r = g.explore(0, k=5, max_distance_computation_count=50)
    indices, distances = r.indices, r.distances
    assert len(indices) == 5
    assert len(distances) == 5
    for idx in indices:
        if idx != 0xFFFFFFFF:
            assert idx % 10 == 0, f"Facade explore result should be external label (multiple of 10), got {idx}"


def test_facade_get_neighbors_uses_external_label():
    """DynamicExplorationGraph facade get_neighbors should return external labels."""
    g, _ = _build_test_graph_with_distinct_labels()
    neighbors = g.get_neighbors(0)
    assert len(neighbors) == g.get_edges_per_vertex()
    for n in neighbors:
        if n != 0xFFFFFFFF:
            assert n % 10 == 0, f"Facade get_neighbors should return external labels, got {n}"


def test_facade_search_returns_external_labels():
    """DynamicExplorationGraph facade search should return external labels."""
    g, data = _build_test_graph_with_distinct_labels()
    r = g.search(data[0:1], eps_or_ef=0.1, k=5)
    indices, distances = r.indices, r.distances
    assert indices.shape == (1, 5)
    for idx in indices.flatten():
        if idx != 0xFFFFFFFF:
            assert idx % 10 == 0, f"Facade search should return external labels, got {idx}"


def test_facade_to_readonly():
    """DynamicExplorationGraph.to_readonly() should return a read-only graph."""
    g, _ = _build_test_graph_with_distinct_labels()
    rd_graph = g.to_readonly()
    assert not rd_graph.is_mutable()
    assert rd_graph.size() == g.size()


def test_facade_explore_return_distances_false():
    """explore with return_distances=False should return only indices array."""
    g, _ = _build_test_graph_with_distinct_labels()
    res = g.explore(0, k=5, max_distance_computation_count=50, return_distances=False)
    assert res.distances is None
    indices = res.indices
    assert isinstance(indices, np.ndarray)
    assert len(indices) == 5


def test_facade_explore_unsorted():
    """explore with unsorted=True should return indices."""
    g, _ = _build_test_graph_with_distinct_labels()
    res = g.explore(0, k=5, max_distance_computation_count=50, unsorted=True)
    indices, distances = res.indices, res.distances
    assert len(indices) == 5
    assert len(distances) == 5


def test_facade_explore_batch():
    """explore_batch with return_distances and unsorted options."""
    g, _ = _build_test_graph_with_distinct_labels()
    labels = np.array([0, 10], dtype=np.uint32)

    # Default return_distances=True
    r_def = g.explore(labels, k=5, max_distance_computation_count=50)
    indices, distances = r_def.indices, r_def.distances
    assert indices.shape == (2, 5)
    assert distances.shape == (2, 5)

    # return_distances=False
    r_only = g.explore(labels, k=5, max_distance_computation_count=50, return_distances=False)
    assert r_only.distances is None
    indices_only = r_only.indices
    assert indices_only.shape == (2, 5)
    assert isinstance(indices_only, np.ndarray)

    # unsorted=True
    r_uns = g.explore(labels, k=5, max_distance_computation_count=50, unsorted=True)
    indices_unsorted, distances_unsorted = r_uns.indices, r_uns.distances
    assert indices_unsorted.shape == (2, 5)
    assert distances_unsorted.shape == (2, 5)


def test_facade_explore_ef():
    """explore and explore_batch with eps_or_ef >= 1.0 (ef mode)."""
    g, _ = _build_test_graph_with_distinct_labels()
    labels = np.array([0, 10], dtype=np.uint32)

    # Single explore with ef
    r_single = g.explore(0, k=5, max_distance_computation_count=50, eps_or_ef=10.0)
    ind_single, dist_single = r_single.indices, r_single.distances
    assert len(ind_single) == 5
    assert len(dist_single) == 5
    assert ind_single[0] == 0

    # Batch explore with ef
    r_batch = g.explore(labels, k=5, max_distance_computation_count=50, eps_or_ef=10.0)
    indices, distances = r_batch.indices, r_batch.distances
    assert indices.shape == (2, 5)
    assert distances.shape == (2, 5)
    assert indices[0, 0] == 0
    assert indices[1, 0] == 10


def test_facade_get_neighbors_batch():
    """get_neighbors handles a batch array via its unified overloaded interface."""
    g, _ = _build_test_graph_with_distinct_labels()
    labels = np.array([0, 10], dtype=np.uint32)
    edges = g.get_edges_per_vertex()

    batch_neighbors = g.get_neighbors(labels, threads=2)
    assert batch_neighbors.shape == (2, edges)
    assert set(batch_neighbors[0]) == set(g.get_neighbors(0))
    assert set(batch_neighbors[1]) == set(g.get_neighbors(10))


def test_facade_has_path_reaches_adjacent_target():
    """has_path should walk from the entry to an adjacent target and return external labels."""
    g, _ = _build_test_graph_with_distinct_labels()
    entry = 0
    neighbors = [int(n) for n in g.get_neighbors(entry)]
    target = next(n for n in neighbors if n != entry and n != 0xFFFFFFFF)

    path, distances = g.has_path([entry], target, eps=0.0, k=1)

    assert len(path) == 2
    assert int(path[0]) == target
    assert int(path[1]) == entry
    assert len(distances) == 2
    assert distances[0] == 0.0
    assert distances[1] > 0.0


def test_facade_has_path_traverses_multiple_hops():
    """has_path should backtrack the whole traversal chain, not only direct neighbors."""
    g, _ = _build_test_graph_with_distinct_labels()
    entry = 0
    neighbors = {int(n) for n in g.get_neighbors(entry)}
    far = next(label for label in range(0, 500, 10) if label != entry and label not in neighbors)

    path, distances = g.has_path([entry], far, eps=0.5, k=1)

    assert len(path) > 2, "a non-adjacent target requires a multi-hop path"
    assert int(path[0]) == far
    assert int(path[-1]) == entry
    assert len(distances) == len(path)
    for label in path:
        assert int(label) % 10 == 0, f"has_path should return external labels, got {label}"


def test_facade_has_path_accepts_scalar_entry_without_distances():
    """has_path should accept a single entry label and honor return_distances=False."""
    g, _ = _build_test_graph_with_distinct_labels()
    entry = 0
    target = next(int(n) for n in g.get_neighbors(entry) if int(n) != entry and int(n) != 0xFFFFFFFF)

    path = g.has_path(entry, target, return_distances=False)

    assert isinstance(path, np.ndarray)
    assert [int(label) for label in path] == [target, entry]


def test_facade_has_path_from_multiple_entries():
    """has_path should accept several entry vertices and end the path at the entry that reached the target."""
    g, data = _build_test_graph_with_distinct_labels()
    entries = [0, 10, 20, 30]
    target = 250
    # The traversal pops the entry closest to the target first, so that one must close the path.
    target_feature = data[target // 10]
    closest = min(entries, key=lambda label: float(np.sum((data[label // 10] - target_feature) ** 2)))

    path, distances = g.has_path(entries, target, eps=0.5, k=1)

    assert len(path) == len(distances)
    assert len(path) >= 2
    assert int(path[0]) == target
    assert int(path[-1]) in entries


def test_facade_search_return_distances_false():
    """search with return_distances=False should set distances to None."""
    g, data = _build_test_graph_with_distinct_labels()
    res = g.search(data[0:1], eps_or_ef=0.1, k=5, return_distances=False)
    assert res.distances is None
    indices = res.indices
    assert isinstance(indices, np.ndarray)
    assert indices.shape == (1, 5)


def test_facade_search_unsorted():
    """search with unsorted=True should return results."""
    g, data = _build_test_graph_with_distinct_labels()
    res = g.search(data[0:2], eps_or_ef=0.1, k=5, unsorted=True)
    indices, distances = res.indices, res.distances
    assert indices.shape == (2, 5)
    assert distances.shape == (2, 5)


def test_to_readonly_custom_features():
    """to_readonly with custom metric and FP16 custom feature buffer."""
    N, d = 20, 16
    data = np.random.randn(N, d).astype(np.float32)

    # Build FP32 L2 graph
    g = deglib.create_empty(N, d, edges_per_vertex=4, metric=Metric.FP32_L2)
    builder = deglib.GraphBuilder(g, extend_k=10, extend_eps=0.2)
    builder.add_entry(range(N), data)
    builder.build()

    # Convert features to FP16
    fp16_data = deglib.distances.floats_to_fp16(data)

    # Swap to FP16 ReadOnlyGraph
    ro_g = g.to_readonly().with_features(fp16_data, metric=Metric.FP16_InnerProduct)
    assert ro_g.size() == N
    assert not ro_g.is_mutable()
    assert ro_g.get_metric() == Metric.FP16_InnerProduct

    # Search on FP16 ReadOnlyGraph
    query_fp16 = deglib.distances.floats_to_fp16(data[0:2])
    r = ro_g.search(query_fp16, eps_or_ef=0.1, k=3)
    res_indices, res_dists = r.indices, r.distances
    assert res_indices.shape == (2, 3)
    assert res_dists.shape == (2, 3)


def test_to_readonly_swap_fp16_l2():
    """Test swapping a mutable graph's metric to FP16 L2 ReadOnlyGraph."""
    rng = np.random.default_rng(42)
    N, d = 20, 8
    data = rng.standard_normal((N, d)).astype(np.float32)

    # Build FP32 L2 graph
    g = deglib.create_empty(N, d, edges_per_vertex=4, metric=Metric.FP32_L2)
    builder = deglib.GraphBuilder(g, extend_k=10, extend_eps=0.2)
    builder.add_entry(range(N), data)
    builder.build()

    # Convert features to FP16
    fp16_data = deglib.distances.floats_to_fp16(data)

    # Swap to FP16 L2 ReadOnlyGraph
    ro_g = g.to_readonly().with_features(fp16_data, metric=Metric.FP16_L2)
    assert ro_g.size() == N
    assert not ro_g.is_mutable()
    assert ro_g.get_metric() == Metric.FP16_L2

    # Search on FP16 ReadOnlyGraph
    query_fp16 = deglib.distances.floats_to_fp16(data[0:2])
    r = ro_g.search(query_fp16, eps_or_ef=0.1, k=3)
    res_indices, res_dists = r.indices, r.distances
    assert res_indices.shape == (2, 3)
    assert res_dists.shape == (2, 3)
    assert res_indices[0, 0] == 0
    assert res_dists[0, 0] < 1e-4


def test_create_random_graph_fp32():
    """Test create_random_graph with FP32 L2 data."""
    samples = 100
    dims = 8
    edges_per_vertex = 10

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)

    graph = deglib.create_random_graph(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2, seed=7)

    assert graph.size() == samples
    assert graph.get_edges_per_vertex() == edges_per_vertex
    assert graph.is_mutable()

    # Search should work
    query = data[0:1]
    r = graph.search(query, eps_or_ef=0.1, k=5)
    indices, distances = r.indices, r.distances
    assert indices.shape == (1, 5)
    assert distances.shape == (1, 5)


def test_graph_optimize_preserves_search_results():
    """graph.optimize() tunes traversal prefetch parameters but must not change search results."""
    samples = 200
    dims = 8
    data = np.random.default_rng(1).standard_normal((samples, dims)).astype(np.float32)
    graph = deglib.create_random_graph(data, edges_per_vertex=16, metric=Metric.FP32_L2, seed=7)

    query = data[0:1]
    before = graph.search(query, eps_or_ef=0.1, k=5).indices

    graph.optimize(sample_count=32, median_subset=0)

    after = graph.search(query, eps_or_ef=0.1, k=5).indices
    np.testing.assert_array_equal(before, after)

    # Default median_subset computes a graph-median entry point and completes cleanly
    graph.optimize(sample_count=32)
    after_median = graph.search(query, eps_or_ef=0.1, k=5).indices
    assert after_median.shape == (1, 5)


def test_create_random_graph_uint8():
    """Test create_random_graph with Uint8 L2 data."""
    samples = 50
    dims = 16
    edges_per_vertex = 8

    data = np.random.default_rng(42).integers(0, 256, size=(samples, dims), dtype=np.uint8)

    graph = deglib.create_random_graph(data, edges_per_vertex=edges_per_vertex, metric=Metric.Uint8_L2, seed=42)

    assert graph.size() == samples
    assert graph.get_edges_per_vertex() == edges_per_vertex


def test_create_random_graph_deterministic():
    """Test that create_random_graph with the same seed produces the same graph."""
    samples = 30
    dims = 4
    edges_per_vertex = 6

    data = np.random.default_rng(99).standard_normal((samples, dims)).astype(np.float32)

    graph1 = deglib.create_random_graph(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2, seed=123)
    graph2 = deglib.create_random_graph(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2, seed=123)

    assert graph1.size() == graph2.size()
    for v in range(samples):
        assert graph1.get_neighbors(v) == graph2.get_neighbors(v)


def test_create_random_graph_inner_product():
    """Test create_random_graph with FP32 InnerProduct metric."""
    samples = 40
    dims = 8
    edges_per_vertex = 6

    data = np.random.default_rng(77).standard_normal((samples, dims)).astype(np.float32)
    # Normalize for inner product
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    graph = deglib.create_random_graph(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_InnerProduct, seed=7)

    assert graph.size() == samples
    assert graph.get_edges_per_vertex() == edges_per_vertex


def test_create_random_graph_explore():
    """Test that explore works on a graph created via create_random_graph."""
    samples = 50
    dims = 4
    edges_per_vertex = 10

    data = np.random.default_rng(55).standard_normal((samples, dims)).astype(np.float32)

    graph = deglib.create_random_graph(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2, seed=7)

    # Explore from vertex 0
    r = graph.explore(0, k=5, max_distance_computation_count=50)
    indices, distances = r.indices, r.distances
    assert len(indices) == 5
    assert len(distances) == 5

    # All indices should be valid external labels (0..samples-1)
    for idx in indices:
        assert idx in range(samples), f"Index {idx} out of range [0, {samples})"


def test_sizebounded_graph_from_graph():
    """Test DynamicExplorationGraph.to_mutable() creates a mutable graph from a readonly graph."""
    samples = 50
    dims = 16
    edges_per_vertex = 10

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)

    # Build a mutable graph, then convert to readonly
    graph = deglib.build(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)
    readonly_graph = graph.to_readonly()
    assert not readonly_graph.is_mutable()

    # Convert back to mutable
    mutable_graph = readonly_graph.to_mutable()
    assert mutable_graph.is_mutable()
    assert mutable_graph.size() == samples
    assert mutable_graph.get_edges_per_vertex() == edges_per_vertex

    # Search should work
    query = data[0:1]
    r = mutable_graph.search(query, eps_or_ef=0.1, k=5)
    indices, distances = r.indices, r.distances
    assert indices.shape == (1, 5)
    assert distances.shape == (1, 5)


def test_sizebounded_graph_from_graph_custom_features():
    """Test to_mutable() with custom features buffer."""
    samples = 30
    dims = 8
    edges_per_vertex = 6

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)

    graph = _build_mutable(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)

    # Create scaled custom features
    custom_features = data * 10.0

    # Replace features on mutable graph (preserves mutable backend)
    mutable_graph = graph.with_features(custom_features, metric=Metric.FP32_L2)
    assert mutable_graph.is_mutable()
    assert mutable_graph.size() == samples

    # Topology should match
    for i in range(samples):
        orig_neighbors = sorted(graph.get_neighbors(i))
        new_neighbors = sorted(mutable_graph.get_neighbors(i))
        assert orig_neighbors == new_neighbors, f"Neighbor mismatch at vertex {i}"


def test_graph_get_labels_and_custom_features_with_permuted_labels():
    """Verify get_labels() returns internal index order and custom_features respects custom_features_order."""
    samples = 30
    dims = 8
    rng = np.random.default_rng(123)
    data = rng.standard_normal((samples, dims)).astype(np.float32)
    # Non-trivial external labels (large values, permuted)
    external_labels = (rng.permutation(samples) * 100 + 7).astype(np.uint32)

    graph = deglib.build(data, labels=external_labels, edges_per_vertex=4, metric=Metric.FP32_L2, seed=42)

    labels_in_order = graph.get_labels()
    assert len(labels_in_order) == samples
    # For a newly built graph, internal order matches builder insertion order
    np.testing.assert_array_equal(labels_in_order, external_labels)

    # Prepare custom features aligned to internal order
    scaled_data = data * 2.0
    # Provide custom_features in internal order
    readonly_graph = graph.to_readonly().with_features(
        scaled_data,
        metric=Metric.FP32_L2,
        order="internal",
    )
    assert readonly_graph.size() == samples
    assert not readonly_graph.is_mutable()
    np.testing.assert_array_equal(readonly_graph.get_labels(), external_labels)

    # Provide custom_features in label order (dense 0..samples-1 permuted labels)
    dense_labels = rng.permutation(samples).astype(np.uint32)
    dense_graph = deglib.build(
        data[dense_labels],
        labels=dense_labels,
        edges_per_vertex=4,
        metric=Metric.FP32_L2,
        seed=42,
    )
    ro_label_order = dense_graph.to_readonly().with_features(
        data * 2.0,
        metric=Metric.FP32_L2,
        order="label",
    )
    q = data[dense_labels[0]] * 2.0
    r = ro_label_order.search(q, eps_or_ef=30.0, k=1)
    idx, dist = r.indices, r.distances
    assert int(idx[0]) == int(dense_labels[0])
    assert np.isclose(dist[0], 0.0, atol=1e-5)


def test_sizebounded_graph_from_graph_capacity():
    """Test to_mutable() with capacity parameter."""
    samples = 20
    dims = 4
    edges_per_vertex = 4

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)

    graph = deglib.build(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)

    # Convert to mutable with larger max_size
    mutable_graph = graph.to_mutable(max_size=100)
    assert mutable_graph.is_mutable()
    assert mutable_graph.size() == samples

    # Should be able to add a vertex via builder
    builder = deglib.GraphBuilder(mutable_graph)
    builder.add_entry(samples, data[0])
    builder.build()
    assert mutable_graph.size() == samples + 1
    assert mutable_graph.has_vertex(samples)


def test_sizebounded_graph_from_graph_search_matches():
    """Test that search on a graph converted via to_mutable returns same results as original."""
    samples = 50
    dims = 16
    edges_per_vertex = 10

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)

    graph = deglib.build(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)
    readonly_graph = graph.to_readonly()
    mutable_graph = readonly_graph.to_mutable()

    # Search on both should return the same results
    query = data[0:1]
    r_orig = readonly_graph.search(query, eps_or_ef=0.1, k=5)
    r_new = mutable_graph.search(query, eps_or_ef=0.1, k=5)
    orig_indices, orig_dists = r_orig.indices, r_orig.distances
    new_indices, new_dists = r_new.indices, r_new.distances

    assert orig_indices.shape == new_indices.shape
    assert np.allclose(orig_dists, new_dists, atol=1e-5)


def test_sizebounded_graph_from_graph_inner_product():
    """Test to_mutable() with InnerProduct metric."""
    samples = 30
    dims = 8
    edges_per_vertex = 6

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    graph = deglib.build(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_InnerProduct)
    mutable_graph = graph.to_mutable()

    assert mutable_graph.is_mutable()
    assert mutable_graph.size() == samples
    assert mutable_graph.get_metric() == Metric.FP32_InnerProduct

    # Search should work
    query = data[0:1]
    res = mutable_graph.search(query, eps_or_ef=0.1, k=5)
    indices, distances = res.indices, res.distances
    assert indices.shape == (1, 5)
    assert distances.shape == (1, 5)


def test_dynamic_graph_create_empty_and_build():
    """Test creating an empty DynamicGraph and populating it with GraphBuilder."""
    samples = 50
    dims = 16
    edges_per_vertex = 8
    chunk_size = 32

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)

    graph = deglib.create_dynamic_empty(
        dim=dims, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2, chunk_size=chunk_size
    )
    assert graph.is_mutable()
    assert graph.size() == 0
    assert graph.get_edges_per_vertex() == edges_per_vertex

    builder = deglib.GraphBuilder(graph)
    labels = np.arange(samples, dtype=np.uint32)
    builder.add_entry(labels, data)
    builder.build()

    assert graph.size() == samples
    for i in range(samples):
        assert graph.has_vertex(i)

    # Search
    query = data[0:1]
    r = graph.search(query, eps_or_ef=0.1, k=5)
    indices, distances = r.indices, r.distances
    assert indices.shape == (1, 5)
    assert indices[0, 0] == 0
    assert np.isclose(distances[0, 0], 0.0, atol=1e-5)


def test_to_dynamic_conversion_and_mutation():
    """Test converting a graph to DynamicGraph via to_dynamic() and modifying it."""
    samples = 40
    dims = 16
    edges_per_vertex = 8

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)
    graph = deglib.build(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)

    dyn_graph = graph.to_dynamic(chunk_size=64)
    assert dyn_graph.is_mutable()
    assert dyn_graph.size() == samples

    # Search on converted graph
    query = data[0:1]
    r_orig = graph.search(query, eps_or_ef=0.1, k=5)
    r_dyn = dyn_graph.search(query, eps_or_ef=0.1, k=5)
    orig_indices, orig_dists = r_orig.indices, r_orig.distances
    dyn_indices, dyn_dists = r_dyn.indices, r_dyn.distances
    assert np.array_equal(orig_indices, dyn_indices)
    assert np.allclose(orig_dists, dyn_dists, atol=1e-5)

    # Further mutation with GraphBuilder
    new_data = np.random.default_rng(123).standard_normal((5, dims)).astype(np.float32)
    new_labels = np.arange(samples, samples + 5, dtype=np.uint32)
    builder = deglib.GraphBuilder(dyn_graph)
    builder.add_entry(new_labels, new_data)
    builder.build()

    assert dyn_graph.size() == samples + 5
    assert dyn_graph.has_vertex(samples + 4)


def test_save_and_load_dynamic_graph(tmp_path):
    """Test saving a graph and reloading it via load_dynamic_graph."""
    samples = 30
    dims = 8
    edges_per_vertex = 6

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)
    graph = _build_mutable(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)

    file_path = tmp_path / "test_dyn_graph.deg"
    graph.save_graph(file_path)

    loaded_graph = deglib.load_dynamic_graph(file_path, chunk_size=32)
    assert loaded_graph.is_mutable()
    assert loaded_graph.size() == samples
    assert loaded_graph.get_edges_per_vertex() == edges_per_vertex

    query = data[0:1]
    r_orig = graph.search(query, eps_or_ef=0.1, k=5)
    r_loaded = loaded_graph.search(query, eps_or_ef=0.1, k=5)
    orig_indices, orig_dists = r_orig.indices, r_orig.distances
    loaded_indices, loaded_dists = r_loaded.indices, r_loaded.distances
    assert np.array_equal(orig_indices, loaded_indices)
    assert np.allclose(orig_dists, loaded_dists, atol=1e-5)


def test_save_and_load_mutable_graph(tmp_path):
    """Test saving a graph and reloading it via load_mutable_graph."""
    samples = 30
    dims = 8
    edges_per_vertex = 6

    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)
    graph = _build_mutable(data, edges_per_vertex=edges_per_vertex, metric=Metric.FP32_L2)

    file_path = tmp_path / "test_mutable_graph.deg"
    graph.save_graph(file_path)

    # 1. Load with default capacity (size of file)
    loaded_graph = deglib.load_mutable_graph(file_path)
    assert loaded_graph.is_mutable()
    assert loaded_graph.size() == samples
    assert loaded_graph.get_edges_per_vertex() == edges_per_vertex

    query = data[0:1]
    r_orig = graph.search(query, eps_or_ef=0.1, k=5)
    r_loaded = loaded_graph.search(query, eps_or_ef=0.1, k=5)
    orig_indices, orig_dists = r_orig.indices, r_orig.distances
    loaded_indices, loaded_dists = r_loaded.indices, r_loaded.distances
    assert np.array_equal(orig_indices, loaded_indices)
    assert np.allclose(orig_dists, loaded_dists, atol=1e-5)

    # 2. Load with extended capacity and add entries
    loaded_with_cap = deglib.load_mutable_graph(file_path, capacity=samples + 10)
    assert loaded_with_cap.is_mutable()
    assert loaded_with_cap.size() == samples

    builder = deglib.GraphBuilder(loaded_with_cap)
    new_data = np.random.default_rng(99).standard_normal((5, dims)).astype(np.float32)
    new_labels = np.arange(samples, samples + 5, dtype=np.uint32)
    builder.add_entry(new_labels, new_data)
    builder.build()
    assert loaded_with_cap.size() == samples + 5


def test_create_helpers():
    """Test module-level graph creation helpers and graph metadata getters."""
    g1 = deglib.create_empty(100, 16, 16, metric=Metric.FP32_L2)
    assert g1.is_mutable()
    assert g1.size() == 0
    assert g1.get_dim() == 16
    assert g1.get_metric() == Metric.FP32_L2
    assert g1.get_data_size() == 16 * 4
    assert isinstance(g1.get_instruction(), deglib.InstructionSet)

    g2 = deglib.create_dynamic_empty(16, 16, metric=Metric.FP32_L2, chunk_size=128)
    assert g2.is_mutable()
    assert g2.size() == 0
    assert g2.get_dim() == 16

    features = np.random.default_rng(42).standard_normal((20, 16)).astype(np.float32)
    g3 = deglib.create_random_graph(features, edges_per_vertex=4, metric=Metric.FP32_L2)
    assert g3.size() == 20


def test_to_readonly_self_quantization():
    """FP32 graph converts to Int8 / EVP automatically when metric changes without explicit transforms."""
    rng = np.random.default_rng(9)
    raw = rng.standard_normal((40, 16)).astype(np.float32)
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    g_l2 = _build_mutable(raw, edges_per_vertex=8, metric=Metric.FP32_L2)
    ro_int8 = g_l2.to_readonly().with_metric(Metric.Int8_L2)
    assert not ro_int8.is_mutable()
    assert ro_int8.get_metric() == Metric.Int8_L2
    idx = ro_int8.search(raw[0], eps_or_ef=0.2, k=5).indices
    assert len(idx) == 5

    mut_int8 = g_l2.with_metric(Metric.Int8_L2)
    assert mut_int8.is_mutable()
    assert mut_int8.get_metric() == Metric.Int8_L2

    g_ip = deglib.build(unit, edges_per_vertex=8, extend_k=16, metric=Metric.FP32_InnerProduct)
    ro_evp = g_ip.to_readonly().with_metric(Metric.EVP_InnerProduct)
    assert not ro_evp.is_mutable()
    assert ro_evp.get_metric() == Metric.EVP_InnerProduct
    idx_evp = ro_evp.search(unit[0], eps_or_ef=0.2, k=5).indices
    assert len(idx_evp) == 5


def test_graph_conversion_validation():
    """Test Python-side validation in with_metric, with_transformations, with_features, to_mutable, and to_dynamic."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((24, 16)).astype(np.float32)
    g = deglib.build(data, edges_per_vertex=4, metric=Metric.FP32_L2)
    q_int8 = deglib.make_scalar_quantizer_int8(data)

    # 1. Invalid metric string / type
    with pytest.raises(ValueError, match="Unknown metric"):
        g.with_metric("Int8_L2_INVALID")
    with pytest.raises(TypeError, match="Metric"):
        g.with_metric(object())

    # 2. Invalid or empty transformations
    with pytest.raises(ValueError, match="at least one"):
        g.with_transformations([])
    with pytest.raises((TypeError, AttributeError)):
        g.with_transformations(["not_a_transform"])

    # 3. Transformation input_dim mismatch against graph dim
    wrong_dim_q = deglib.make_scalar_quantizer_int8(rng.standard_normal((10, 8)).astype(np.float32))
    with pytest.raises(ValueError, match="input_dim"):
        g.with_transformations([wrong_dim_q])

    # 4. Transformation output_byte_size mismatch against target metric
    with pytest.raises(ValueError, match="output_byte_size"):
        g.to_dynamic().with_transformations([q_int8], metric=Metric.FP16_L2)

    # 5. with_features wrong shape / row count / dtype
    with pytest.raises(ValueError, match="2D array"):
        g.with_features(data.ravel())
    with pytest.raises(ValueError, match="row count"):
        g.with_features(data[:10])
    with pytest.raises((ValueError, deglib.utils.WrongDtypeException)):
        g.to_dynamic().with_features(data.astype(np.float16))

    # 6. Self-transform from non-float graph
    ro_int8 = g.to_readonly().with_transformations([q_int8], metric=Metric.Int8_L2)
    assert not ro_int8.is_mutable()
    q_uint8 = deglib.make_scalar_quantizer_uint8(data)
    with pytest.raises(ValueError, match="non-float"):
        ro_int8.with_metric(Metric.Uint8_L2)
    with pytest.raises((ValueError, RuntimeError)):
        ro_int8.with_transformations([q_uint8], metric=Metric.Uint8_L2)

    # 7. Invalid max_size / chunk_size
    with pytest.raises(ValueError, match="max_size"):
        g.to_mutable(max_size=-1)
    with pytest.raises(ValueError, match="chunk_size"):
        g.to_dynamic(chunk_size=0)


def test_graph_with_metric_same_dtype_retag_preserves_backend():
    """with_metric(metric) with same data type preserves features and current backend."""
    rng = np.random.default_rng(13)
    data = rng.standard_normal((24, 8)).astype(np.float32)
    g_l2 = _build_mutable(data, edges_per_vertex=4, metric=Metric.FP32_L2)

    ro_ip = g_l2.to_readonly().with_metric(Metric.FP32_InnerProduct)
    assert not ro_ip.is_mutable()
    assert ro_ip.get_metric() == Metric.FP32_InnerProduct
    assert ro_ip.get_neighbors(0) == g_l2.get_neighbors(0)

    mut_ip = g_l2.with_metric(Metric.FP32_InnerProduct, threads=2)
    assert mut_ip.is_mutable()
    assert mut_ip.get_metric() == Metric.FP32_InnerProduct
    assert mut_ip.get_neighbors(0) == g_l2.get_neighbors(0)

    dyn_ip = g_l2.to_dynamic(chunk_size=32).with_metric(Metric.FP32_InnerProduct, threads=2)
    assert dyn_ip.is_mutable()
    assert dyn_ip.get_metric() == Metric.FP32_InnerProduct
    assert dyn_ip.get_neighbors(0) == g_l2.get_neighbors(0)


def test_graph_search_results_zero_copy_lifetime_and_bounds():
    """SearchResults from graph.search and graph.explore keep NumPy views valid after wrapper deletion."""
    import gc

    rng = np.random.default_rng(11)
    data = rng.standard_normal((100, 16)).astype(np.float32)
    g = deglib.build(data, edges_per_vertex=8, extend_k=16, metric=Metric.FP32_L2, optimize=False)

    # Single query SearchResults + lifetime after `del res`
    res = g.search(data[0], eps_or_ef=0.2, k=5)
    assert isinstance(res, deglib.SearchResults)
    assert res.graph_distance_count > 0
    assert res.graph_hop_count > 0
    assert res.rerank_distance_count == 0
    idx = res.indices
    dist = res.distances
    assert not idx.flags["OWNDATA"]
    assert not dist.flags["OWNDATA"]
    idx_copy = idx.copy()
    dist_copy = dist.copy()
    del res
    gc.collect()
    np.testing.assert_array_equal(idx, idx_copy)
    np.testing.assert_allclose(dist, dist_copy)

    # Budget bounds for both eps and ef modes
    bounded_eps = g.search(data[0], eps_or_ef=0.2, k=10, max_distance_computation_count=4)
    assert 1 <= bounded_eps.graph_distance_count <= 4
    bounded_ef = g.search(data[0], eps_or_ef=32.0, k=10, max_distance_computation_count=4)
    assert 1 <= bounded_ef.graph_distance_count <= 4

    # Batch explore with distances=False and telemetry
    exp_res = g.explore(
        np.array([0, 1, 2], dtype=np.uint32), k=6, max_distance_computation_count=30, return_distances=False
    )
    assert isinstance(exp_res, deglib.SearchResults)
    assert exp_res.distances is None
    assert exp_res.graph_distance_count > 0
    assert exp_res.graph_hop_count > 0
    assert exp_res.rerank_distance_count == 0
    exp_idx = exp_res.indices
    assert exp_idx.shape == (3, 6)
    assert not exp_idx.flags["OWNDATA"]
    exp_copy = exp_idx.copy()
    del exp_res
    gc.collect()
    np.testing.assert_array_equal(exp_idx, exp_copy)
