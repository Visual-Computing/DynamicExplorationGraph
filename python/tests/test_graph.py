import os
import platform
import random
from typing import Optional

import pytest
import pathlib
import tempfile
import numpy as np

import deglib
from deglib.search import Filter
from deglib.distances import Metric, FloatSpace

IS_MACOS_M1 = platform.system() == "Darwin" and platform.machine() == "arm64"


def get_tmp_graph_file(samples: int, dims: int) -> pathlib.Path:
    tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_test')
    os.makedirs(tmpdir, exist_ok=True)

    return pathlib.Path(os.path.join(tmpdir, 'test_graph_S{}_D{}.deg'.format(samples, dims)))


def get_ranking(features: np.ndarray, graph: deglib.DynamicExplorationGraph, query: np.ndarray) -> np.ndarray:
    """
    Returns the ranking for each feature vector in the graph
    """
    query = query.reshape(1, graph.get_feature_space().dim())

    if features.dtype == np.uint8:
        features = features.astype(np.float32)
    if query.dtype == np.uint8:
        query = query.astype(np.float32)

    if graph.get_feature_space().metric() in (Metric.FP32_L2, Metric.Uint8_L2):
        distances = np.sum(np.square(features - query), axis=1)
    elif graph.get_feature_space().metric() == Metric.FP32_InnerProduct:
        distances = 1.0 - np.dot(features, query.T).flatten()
    else:
        raise ValueError(f'unknown metric: {graph.get_feature_space().metric()}')
    return np.argsort(distances)


class Configuration:
    def __init__(
        self, edges_per_vertex: int, samples: int, dims: int, data: np.ndarray, graph: deglib.DynamicExplorationGraph,
            graph_path: Optional[pathlib.Path], query: np.ndarray, metric: Metric
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
        if metric == Metric.FP32_InnerProduct:
            # normalize data
            data = np.random.random((samples, dims)).astype(np.float32)
            data /= np.linalg.norm(data, axis=1).reshape(-1, 1)

            query = np.random.random((dims,)).astype(np.float32)
            query /= np.linalg.norm(query)
        elif metric == Metric.FP32_L2:
            data = np.random.normal(size=(samples, dims)).astype(np.float32)
            query = np.random.normal(size=(dims,)).astype(np.float32)
        elif metric == Metric.Uint8_L2:
            data = np.random.randint(0, 256, size=(samples, dims)).astype(np.uint8)
            query = np.random.randint(0, 256, size=(dims,)).astype(np.uint8)
        else:
            raise ValueError(f'Unsupported metric: {metric}')

        size_bounded_graph = deglib.build_from_data(
            data, edges_per_vertex=edges_per_vertex, metric=metric, optimization_target=deglib.builder.OptimizationTarget.LowLID,
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
        return deglib.build_from_data(
            self.data, edges_per_vertex=self.edges_per_vertex, metric=self.metric
        )

    def __repr__(self):
        return f'Conf({type(self.graph).__name__}, metric={self.metric.name})'


configurations = [
    *Configuration.generate(100, 128, Metric.FP32_L2, 10),
    *Configuration.generate(100, 128, Metric.Uint8_L2, 10),
    *Configuration.generate(100, 128, Metric.FP32_InnerProduct, 10),
]

large_configurations = [
    *Configuration.generate(20_000, 2, Metric.Uint8_L2, 10),
]

mutable_configurations = [c for c in configurations if c.graph.is_mutable()]


@pytest.mark.parametrize('conf', configurations)
def test_search(conf: Configuration):
    if IS_MACOS_M1 and conf.metric == Metric.InnerProduct:
        pytest.skip('This test is skipped on macOS with M1 chip, as avx2 is not supported on m1 chip.')

    k = 10
    graph_result, dists = conf.graph.search(conf.query, eps=0.1, k=k)
    dists = dists.flatten()
    graph_result = graph_result.flatten()
    correct_result = get_ranking(conf.data, conf.graph, conf.query)[:k]

    assert graph_result.shape[-1] == k, 'expected {} results, but got {}'.format(k, graph_result.shape[-1])

    # test matches are good
    matches = set(graph_result).intersection(set(correct_result))
    assert len(matches) >= k-2, 'expected at least {} matching results, but got only {}'.format(k-2, len(matches))

    # test result is sorted
    last_distance = -1.0
    for index, distance in enumerate(dists):
        assert last_distance <= distance, (
            'ResultSet is not sorted.\ndistance {} at index {} larger than\ndistance {} at index {}'.format(
                last_distance, index-1, distance, index
            )
        )


@pytest.mark.parametrize('conf', mutable_configurations)
def test_remove_non_mrng_edges(conf: Configuration):
    graph = conf.create_new_size_bounded_graph()
    deglib.optimization.remove_non_mrng_edges(graph)


@pytest.mark.parametrize('conf', configurations)
def test_threaded_search(conf: Configuration):
    k = 10
    graph_result, dists = conf.graph.search(conf.query, eps=0.1, k=k)
    for n_threads in range(2, 8):
        threaded_graph_result, threaded_dists = conf.graph.search(conf.query, eps=0.1, k=k, threads=n_threads)
        assert np.all(np.equal(threaded_graph_result, graph_result)), \
            'Threaded and non threaded results differ (n_threads={})'.format(n_threads)
        assert np.allclose(threaded_dists, dists), \
            'Threaded and non threaded dists differ (n_threads={})'.format(n_threads)


@pytest.mark.parametrize('conf', configurations)
def test_explore(conf: Configuration):
    k = 10
    include_entry = True
    entry_vertex_index = random.randint(0, conf.samples-1)
    indices, distances = conf.graph.explore(entry_vertex_index, k, max_distance_computation_count=k*10, include_entry=include_entry)
    assert len(indices) == k
    assert len(distances) == k
    # check that valid (non-NaN) distances are sorted (ascending)
    valid_distances = distances[~np.isnan(distances)]
    for i in range(len(valid_distances) - 1):
        assert valid_distances[i] <= valid_distances[i + 1], 'Distances should be sorted in ascending order'


@pytest.mark.parametrize('conf', configurations)
def test_get_edges_per_vertex(conf: Configuration):
    assert conf.graph.get_edges_per_vertex() == conf.edges_per_vertex


@pytest.mark.parametrize('conf', configurations)
def test_has_vertex(conf: Configuration):
    assert conf.graph.has_vertex(0)
    assert not conf.graph.has_vertex(conf.graph.size())


@pytest.mark.parametrize('conf', mutable_configurations)
def test_modify_graph(conf: Configuration):
    graph = conf.create_new_size_bounded_graph()
    # remove vertex via builder
    builder = deglib.GraphBuilder(graph)
    builder.remove_entry(graph.size()-1)
    builder.build()
    assert graph.size() == conf.samples - 1
    # add vertex back via builder
    builder2 = deglib.GraphBuilder(graph)
    builder2.add_entry(conf.samples-1, conf.data[-1])
    builder2.build()


@pytest.mark.parametrize('conf', mutable_configurations)
def test_load_graph(conf: Configuration):
    graph = deglib.load_readonly_graph(conf.graph_path)
    assert not graph.is_mutable()

    with pytest.raises(FileNotFoundError):
        _graph = deglib.load_readonly_graph(pathlib.Path('path') / 'does' / 'not' / 'exist')


@pytest.mark.parametrize('conf', mutable_configurations)
def test_save_graph(tmp_path, conf):
    target_path = tmp_path / "save_path.deg"
    if target_path.is_file():
        os.remove(target_path)
    conf.graph.save_graph(target_path)
    assert target_path.is_file()
    os.remove(target_path)


@pytest.mark.parametrize('conf', mutable_configurations)
def test_convert_graph(conf: Configuration):
    rd_graph = conf.graph.to_readonly()
    assert not rd_graph.is_mutable()


@pytest.mark.parametrize('conf', large_configurations)
def test_filters(conf: Configuration):
    k = 400

    valid_labels = np.random.choice(conf.graph.size(), size=12_000, replace=False)
    results, _dists = conf.graph.search(conf.query, filter_labels=Filter(valid_labels), eps=0.01, k=k)

    if not np.all(np.isin(results, valid_labels)):
        raise ValueError('Found results that should have been filtered out.')


@pytest.mark.parametrize('conf', large_configurations)
def test_small_filters(conf: Configuration):
    for n_valid in [200, 400, 600]:
        k = 400

        valid_labels = np.random.choice(conf.graph.size(), size=n_valid, replace=False)
        # if less valid labels are present, than k, a warning is expected
        if n_valid < k:
            with pytest.warns(UserWarning):
                results, _dists = conf.graph.search(conf.query, filter_labels=Filter(valid_labels), eps=0.01, k=k)
        else:
            results, _dists = conf.graph.search(conf.query, filter_labels=Filter(valid_labels), eps=0.01, k=k)

        if results.shape[-1] != min(k, n_valid):
            raise ValueError('expected {} results, but got {}'.format(min(k, n_valid), results.shape[-1]))

        if not np.all(np.isin(results, valid_labels)):
            raise ValueError('Found results that should have been filtered out.')


@pytest.mark.parametrize('conf', configurations[:1])
def test_filter_edge_cases(conf: Configuration):
    k = 5
    # All valid labels filter
    all_labels = np.arange(conf.graph.size(), dtype=np.int32)
    results_all, _ = conf.graph.search(conf.query, filter_labels=Filter(all_labels), eps=0.1, k=k)
    assert results_all.shape[-1] == k

    # No valid labels filter with max_value specified
    no_labels = np.array([], dtype=np.int32)
    with pytest.warns(UserWarning):
        results_none, _ = conf.graph.search(conf.query, filter_labels=Filter(no_labels, max_value=0), eps=0.1, k=k)


def _build_test_graph_with_distinct_labels(samples=50, dims=16, edges_per_vertex=10):
    """Build a graph with external labels that differ from internal indices."""
    data = np.random.default_rng(42).standard_normal((samples, dims)).astype(np.float32)
    labels = np.arange(samples, dtype=np.uint32) * 10
    graph = deglib.build_from_data(
        data, labels=labels, edges_per_vertex=edges_per_vertex,
        optimization_target=deglib.builder.OptimizationTarget.LowLID,
        extend_k=10, extend_eps=0.001
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
    indices, distances = g.search(data[0:1], eps=0.1, k=5)
    assert indices.shape == (1, 5)
    # search_batch returns external labels (multiples of 10)
    for idx in indices.flatten():
        if idx != 0xFFFFFFFF:
            assert idx % 10 == 0, \
                f"Search result {idx} should be an external label (multiple of 10)"


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
    indices, distances = g.explore(0, k=5, max_distance_computation_count=50)
    assert len(indices) == 5
    assert len(distances) == 5
    for idx in indices:
        if idx != 0xFFFFFFFF:
            assert idx % 10 == 0, \
                f"Facade explore result should be external label (multiple of 10), got {idx}"


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
    indices, distances = g.search(data[0:1], eps=0.1, k=5)
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
