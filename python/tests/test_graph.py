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

IS_MACOS_M1 = platform.system() == "Darwin" and platform.machine() == "arm64"


def get_tmp_graph_file(samples: int, dims: int) -> pathlib.Path:
    tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_test')
    os.makedirs(tmpdir, exist_ok=True)

    return pathlib.Path(os.path.join(tmpdir, 'test_graph_S{}_D{}.deg'.format(samples, dims)))


def get_ranking(features: np.ndarray, graph: deglib.graph.SearchGraph, query: np.ndarray) -> np.ndarray:
    """
    Returns the ranking for each feature vector in the graph
    """
    query = query.reshape(1, graph.get_feature_space().dim())

    if features.dtype == np.uint8:
        features = features.astype(np.float32)
    if query.dtype == np.uint8:
        query = query.astype(np.float32)

    if graph.get_feature_space().metric() in (deglib.Metric.L2, deglib.Metric.L2_Uint8):
        distances = np.sum(np.square(features - query), axis=1)
    elif graph.get_feature_space().metric() == deglib.Metric.InnerProduct:
        distances = 1.0 - np.dot(features, query.T).flatten()
    else:
        raise ValueError(f'unknown metric: {graph.get_feature_space().metric()}')
    return np.argsort(distances)


class Configuration:
    def __init__(
            self, edges_per_vertex: int, samples: int, dims: int, data: np.ndarray, graph: deglib.graph.SearchGraph,
            graph_path: Optional[pathlib.Path], query: np.ndarray, metric: deglib.Metric
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
        if metric == deglib.Metric.InnerProduct:
            # normalize data
            data = np.random.random((samples, dims)).astype(np.float32)
            data /= np.linalg.norm(data, axis=1).reshape(-1, 1)

            query = np.random.random((dims,)).astype(np.float32)
            query /= np.linalg.norm(query)
        elif metric == deglib.Metric.L2:
            data = np.random.normal(size=(samples, dims)).astype(np.float32)
            query = np.random.normal(size=(dims,)).astype(np.float32)
        elif metric == deglib.Metric.L2_Uint8:
            data = np.random.randint(0, 256, size=(samples, dims)).astype(np.uint8)
            query = np.random.randint(0, 256, size=(dims,)).astype(np.uint8)
        else:
            raise ValueError(f'Unsupported metric: {metric}')

        size_bounded_graph = deglib.builder.build_from_data(
            data, edges_per_vertex=edges_per_vertex, metric=metric, lid=deglib.builder.LID.Low,
        )

        graph_path = get_tmp_graph_file(samples, dims)
        size_bounded_graph.save_graph(graph_path)
        read_only_graph = deglib.graph.load_readonly_graph(graph_path)
        read_only_graph_converted = deglib.graph.ReadOnlyGraph.from_graph(size_bounded_graph)

        return [
            Configuration(edges_per_vertex, samples, dims, data, size_bounded_graph, graph_path, query, metric),
            Configuration(edges_per_vertex, samples, dims, data, read_only_graph, None, query, metric),
            Configuration(edges_per_vertex, samples, dims, data, read_only_graph_converted, None, query, metric),
        ]

    def create_new_size_bounded_graph(self):
        return deglib.builder.build_from_data(
            self.data, edges_per_vertex=self.edges_per_vertex, metric=self.metric
        )

    def __repr__(self):
        return f'Conf({type(self.graph).__name__}, metric={self.metric.name})'


configurations = [
    *Configuration.generate(100, 128, deglib.Metric.L2, 10),
    *Configuration.generate(100, 128, deglib.Metric.L2_Uint8, 10),
    *Configuration.generate(100, 128, deglib.Metric.InnerProduct, 10),
]

large_configurations = [
    *Configuration.generate(20_000, 2, deglib.Metric.L2_Uint8, 10),
]

mutable_configurations = [c for c in configurations if isinstance(c.graph, deglib.graph.MutableGraph)]


@pytest.mark.parametrize('conf', configurations)
def test_get_feature_vector(conf: Configuration):
    for i in range(conf.graph.size()):
        l = conf.graph.get_external_label(i)
        fv = conf.graph.get_feature_vector(i)
        assert fv.shape == (conf.dims,)
        assert np.allclose(fv, conf.data[l])

    with pytest.raises(IndexError):
        _fv = conf.graph.get_feature_vector(conf.graph.size())


@pytest.mark.parametrize('conf', configurations)
def test_search(conf: Configuration):
    if IS_MACOS_M1 and conf.metric == deglib.Metric.InnerProduct:
        pytest.skip('This test is skipped on macOS with M1 chip, as avx2 is not supported on m1 chip.')

    k = 10
    graph_result, dists = conf.graph.search(conf.query, eps=0.1, k=k)
    dists = dists.flatten()
    graph_result = graph_result.flatten()
    correct_result = get_ranking(conf.data, conf.graph, conf.query)[:k]

    assert graph_result.shape[-1] == k, 'expected {} results, but got {}'.format(k, graph_result.shape[1])

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
        last_distance = distance


@pytest.mark.parametrize('conf', mutable_configurations)
def test_remove_non_mrng_edges(conf: Configuration):
    graph = conf.create_new_size_bounded_graph()
    graph.remove_non_mrng_edges()


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
def test_has_path(conf: Configuration):
    entry_vertex_indices = conf.graph.get_entry_vertex_indices()
    path = conf.graph.has_path(entry_vertex_indices, 70, 0.001, 10)
    for p in path:
        assert isinstance(p, deglib.search.ObjectDistance)


@pytest.mark.parametrize('conf', configurations)
def test_explore(conf: Configuration):
    k = 10
    entry_vertex_index = random.randint(0, conf.samples-1)
    result = conf.graph.explore(entry_vertex_index, k, max_distance_computation_count=k*10)
    assert len(result) == k
    assert all(isinstance(od, deglib.search.ObjectDistance) for od in result)


@pytest.mark.parametrize('conf', configurations)
def test_get_edges_per_vertex(conf: Configuration):
    assert conf.graph.get_edges_per_vertex() == conf.edges_per_vertex


@pytest.mark.parametrize('conf', configurations)
def test_get_neighbor_indices(conf: Configuration):
    for i in range(conf.graph.size()):
        neighbor_indices = conf.graph.get_neighbor_indices(i)
        assert isinstance(neighbor_indices, np.ndarray)
        assert len(neighbor_indices) == conf.edges_per_vertex
        assert neighbor_indices.dtype == np.uint32


@pytest.mark.parametrize('conf', configurations)
def test_has_vertex(conf: Configuration):
    assert conf.graph.has_vertex(0)
    assert not conf.graph.has_vertex(conf.graph.size())


@pytest.mark.parametrize('conf', configurations)
def test_has_edge(conf: Configuration):
    counter = 0
    for e in range(conf.graph.size()):
        if conf.graph.has_edge(0, e):
            counter += 1
    assert counter == conf.graph.get_edges_per_vertex()


@pytest.mark.parametrize('conf', mutable_configurations)
def test_get_neighbor_weights(conf: Configuration):
    weights = conf.graph.get_neighbor_weights(0)
    assert isinstance(weights, np.ndarray)
    assert len(weights) == conf.edges_per_vertex


@pytest.mark.parametrize('conf', mutable_configurations)
def test_modify_graph(conf: Configuration):
    graph = conf.create_new_size_bounded_graph()
    graph.remove_vertex(graph.size()-1)
    assert graph.size() == conf.samples - 1
    graph.add_vertex(conf.samples-1, conf.data[-1])


@pytest.mark.parametrize('conf', mutable_configurations)
def test_load_graph(conf: Configuration):
    graph = deglib.graph.load_readonly_graph(conf.graph_path)
    assert isinstance(graph, deglib.graph.ReadOnlyGraph)

    with pytest.raises(FileNotFoundError):
        _graph = deglib.graph.load_readonly_graph(pathlib.Path('path') / 'does' / 'not' / 'exist')


@pytest.mark.parametrize('conf', mutable_configurations)
def test_save_graph(tmp_path, conf):
    target_path = tmp_path / "save_path.deg"
    if target_path.is_file():
        os.remove(target_path)
    conf.graph.save_graph(target_path)
    assert target_path.is_file()
    os.remove(target_path)


@pytest.mark.parametrize('conf', mutable_configurations)
def test_del_graph(conf: Configuration):
    graph = deglib.graph.load_readonly_graph(conf.graph_path)
    fv = graph.get_feature_vector(0, copy=True)

    del graph

    print(np.sum(fv))  # try to access data, after graph is deleted


@pytest.mark.parametrize('conf', mutable_configurations)
def test_convert_graph(conf: Configuration):
    rd_graph = deglib.graph.ReadOnlyGraph.from_graph(conf.graph)
    assert isinstance(rd_graph, deglib.graph.ReadOnlyGraph)


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
