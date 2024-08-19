import os
import random
from typing import Callable, Self

import pytest
import pathlib
import tempfile
import numpy as np

import deglib


def get_tmp_graph_file(samples: int, dims: int) -> pathlib.Path:
    tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_test')
    os.makedirs(tmpdir, exist_ok=True)

    return pathlib.Path(os.path.join(tmpdir, 'test_graph_S{}_D{}.deg'.format(samples, dims)))


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


def get_read_only_graph_converted(test_graphs):
    return test_graphs.read_only_graph_converted


all_graph_getters = [get_read_only_graph, get_size_bounded_graph, get_read_only_graph_converted]


class TestGraphs:
    def setup_method(self):
        self.samples = 100
        self.dims = 128
        self.edges_per_vertex = self.samples // 10

        self.data = np.random.random((self.samples, self.dims)).astype(np.float32)
        self.size_bounded_graph = deglib.builder.build_from_data(
            self.data, edges_per_vertex=self.edges_per_vertex
        )

        self.graph_path = get_tmp_graph_file(self.samples, self.dims)
        self.size_bounded_graph.save_graph(self.graph_path)
        self.read_only_graph = deglib.graph.load_readonly_graph(self.graph_path)
        self.read_only_graph_converted = deglib.graph.ReadOnlyGraph.from_graph(self.size_bounded_graph)

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_get_feature_vector(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        for i in range(graph.size()):
            fv = graph.get_feature_vector(i)
            assert fv.shape == (self.dims,)
            assert fv.dtype == np.float32

        with pytest.raises(IndexError):
            _fv = self.read_only_graph.get_feature_vector(self.read_only_graph.size())

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_search(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        k = 10
        query = np.random.random((self.dims,)).astype(np.float32)
        graph_result, dists = graph.search(query, eps=0.1, k=k)
        dists = dists.flatten()
        graph_result = graph_result.flatten()
        correct_result = get_ranking(graph, query)[:k]

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

    def test_remove_non_mrng_edges(self):
        self.size_bounded_graph.remove_non_mrng_edges()

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_threaded_search(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        k = 10
        query = np.random.random((self.dims,)).astype(np.float32)
        graph_result, dists = graph.search(query, eps=0.1, k=k)
        for n_threads in range(2, 8):
            threaded_graph_result, threaded_dists = graph.search(query, eps=0.1, k=k, threads=n_threads)
            assert np.allclose(threaded_graph_result, graph_result), \
                'Threaded and non threaded results differ (n_threads={})'.format(n_threads)
            assert np.allclose(threaded_dists, dists), \
                'Threaded and non threaded dists differ (n_threads={})'.format(n_threads)

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_has_path(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        entry_vertex_indices = graph.get_entry_vertex_indices()
        path = graph.has_path(entry_vertex_indices, 70, 0.001, 10)
        for p in path:
            assert isinstance(p, deglib.search.ObjectDistance)

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_explore(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph: deglib.graph.SearchGraph = graph_getter(self)
        k = 10
        entry_vertex_index = random.randint(0, self.samples-1)
        result = graph.explore(entry_vertex_index, k, max_distance_computation_count=k*10)
        assert len(result) == k
        assert all(isinstance(od, deglib.search.ObjectDistance) for od in result)

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_get_edges_per_vertex(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        assert graph.get_edges_per_vertex() == self.edges_per_vertex

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_get_neighbor_indices(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        for i in range(graph.size()):
            neighbor_indices = graph.get_neighbor_indices(i)
            assert isinstance(neighbor_indices, np.ndarray)
            assert len(neighbor_indices) == self.edges_per_vertex
            assert neighbor_indices.dtype == np.uint32

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
    def test_has_vertex(self, graph_getter: Callable[[Self], deglib.graph.SearchGraph]):
        graph = graph_getter(self)
        assert graph.has_vertex(0)
        assert not graph.has_vertex(graph.size())

    @pytest.mark.parametrize('graph_getter', all_graph_getters)
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

    def test_convert_graph(self):
        rd_graph = deglib.graph.ReadOnlyGraph.from_graph(self.size_bounded_graph)
        assert isinstance(rd_graph, deglib.graph.ReadOnlyGraph)
