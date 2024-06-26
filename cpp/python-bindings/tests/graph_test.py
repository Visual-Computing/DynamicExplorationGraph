import os
import random
from typing import Callable, Self

import pytest
import pathlib
import tempfile
import numpy as np

import deglib
from deglib.graph import SearchGraph


def get_tmp_graph_file(samples: int, dims: int) -> pathlib.Path:
    tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_test')
    os.makedirs(tmpdir, exist_ok=True)

    return pathlib.Path(os.path.join(tmpdir, 'test_graph_S{}_D{}.deg'.format(samples, dims)))


def create_size_bounded_graph(data: np.ndarray, edges_per_vertex: int):
    samples, dims = data.shape

    rnd = deglib.Mt19937()
    feature_space = deglib.FloatSpace(dims, deglib.Metric.L2)
    graph = deglib.graph.SizeBoundedGraph(samples, edges_per_vertex, feature_space)

    builder = deglib.builder.EvenRegularGraphBuilder(graph, rnd, extend_k=30, extend_eps=0.2, improve_k=30)

    for i, vec in enumerate(data):
        vec: np.ndarray
        builder.add_entry(i, vec)

    def improvement_callback(_status):
        pass

    builder.build(improvement_callback, False)

    return graph


def get_ranking(graph: deglib.graph.SearchGraph, query: np.ndarray) -> np.ndarray:
    """
    Returns the ranking for each feature vector in the graph
    """
    features = np.empty((graph.size(), graph.get_feature_space().dim()), dtype=np.float32)

    for i in range(graph.size()):
        features[i] = graph.get_feature_vector(i)

    query = query.reshape(1, graph.get_feature_space().dim())

    l2_distances = np.sum(np.square(features - query), axis=1)
    return np.argsort(l2_distances)


def get_read_only_graph(test_graphs):
    return test_graphs.read_only_graph


def get_size_bounded_graph(test_graphs):
    return test_graphs.size_bounded_graph


class TestGraphs:
    def setup_method(self):
        self.samples = 100
        self.dims = 128
        self.edges_per_vertex = self.samples // 10

        self.data = np.random.random((self.samples, self.dims)).astype(np.float32)
        self.size_bounded_graph = create_size_bounded_graph(self.data, edges_per_vertex=self.edges_per_vertex)

        self.graph_path = get_tmp_graph_file(self.samples, self.dims)
        self.size_bounded_graph.save_graph(self.graph_path)
        self.read_only_graph = deglib.graph.load_readonly_graph(self.graph_path)

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_get_feature_vector(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        for i in range(graph.size()):
            fv = graph.get_feature_vector(i)
            assert fv.shape == (self.dims,)
            assert fv.dtype == np.float32

        with pytest.raises(IndexError):
            _fv = self.read_only_graph.get_feature_vector(self.read_only_graph.size())

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_search(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        k = 10
        query = np.random.random((self.dims,)).astype(np.float32)
        graph_result = graph.search(query, eps=0.1, k=k)
        correct_result = get_ranking(graph, query)[:k]

        matches = set(g.get_internal_index() for g in graph_result).intersection(set(correct_result))
        assert len(matches) >= k-2, 'expected at least {} matching results, but got only {}'.format(k-2, len(matches))

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_has_path(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        entry_vertex_indices = graph.get_entry_vertex_indices()
        path = graph.has_path(entry_vertex_indices, 70, 0.001, 10)
        for p in path:
            assert isinstance(p, deglib.search.ObjectDistance)

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_explore(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph: SearchGraph = graph_getter(self)
        k = 10
        entry_vertex_index = random.randint(0, self.samples)
        result = graph.explore(entry_vertex_index, k, max_distance_computation_count=k*10)
        assert len(result) == k
        assert all(isinstance(od, deglib.search.ObjectDistance) for od in result)

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_get_edges_per_vertex(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        assert graph.get_edges_per_vertex() == self.edges_per_vertex

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_get_neighbor_indices(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        for i in range(graph.size()):
            neighbor_indices = graph.get_neighbor_indices(i)
            assert isinstance(neighbor_indices, np.ndarray)
            assert len(neighbor_indices) == self.edges_per_vertex
            assert neighbor_indices.dtype == np.uint32

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_has_vertex(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        assert graph.has_vertex(0)
        assert not graph.has_vertex(graph.size())

    @pytest.mark.parametrize('graph_getter', [get_read_only_graph, get_size_bounded_graph])
    def test_has_edge(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        counter = 0
        for e in range(graph.size()):
            if graph.has_edge(0, e):
                counter += 1
        assert counter == graph.get_edges_per_vertex()

    def test_get_neighbor_weights(self):
        weights = self.size_bounded_graph.get_neighbor_weights(0)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == self.edges_per_vertex

    def test_modify_graph(self):
        self.size_bounded_graph.remove_vertex(self.size_bounded_graph.size()-1)
        assert self.size_bounded_graph.size() == self.samples - 1
        self.size_bounded_graph.add_vertex(self.samples-1, self.data[-1])

    def test_load_graph(self):
        graph = deglib.graph.load_readonly_graph(self.graph_path)
        assert isinstance(graph, deglib.graph.ReadOnlyGraph)

        with pytest.raises(FileNotFoundError):
            _graph = deglib.graph.load_readonly_graph(pathlib.Path('path') / 'does' / 'not' / 'exist')

    def test_save_graph(self, tmp_path):
        target_path = tmp_path / "save_path.deg"
        if target_path.is_file():
            os.remove(target_path)
        self.size_bounded_graph.save_graph(target_path)
        assert target_path.is_file()
        os.remove(target_path)

    def test_del_graph(self):
        graph = deglib.graph.load_readonly_graph(self.graph_path)
        fv = graph.get_feature_vector(0, copy=True)

        del graph

        print(np.sum(fv))  # try to access data, after graph is deleted
