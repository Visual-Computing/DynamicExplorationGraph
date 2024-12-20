import os
import multiprocessing
import warnings
from typing import List, Optional, Tuple, Self, Union
from abc import ABC, abstractmethod

import numpy as np

import deglib_cpp
import pathlib

from .distances import FloatSpace, Metric, SpaceInterface
from .search import ResultSet, ObjectDistance, Filter
from .utils import assure_array, InvalidShapeException, assure_contiguous


class SearchGraph(ABC):
    def __init__(self, graph_cpp: deglib_cpp.SearchGraph):
        self.graph_cpp = graph_cpp

    @abstractmethod
    def size(self) -> int:
        """
        :return: the number of vertices in the graph
        """
        raise NotImplementedError()

    @abstractmethod
    def get_edges_per_vertex(self) -> int:
        """
        :return: the number of edges of each vertex
        """
        raise NotImplementedError()

    @abstractmethod
    def get_feature_space(self) -> SpaceInterface:
        """
        :return: the feature space
        """
        raise NotImplementedError()

    @abstractmethod
    def get_external_label(self, internal_index: int) -> int:
        """
        Translates external labels to internal index

        :param internal_index: The internal index to translate
        :returns: The external label
        """
        raise NotImplementedError()

    @abstractmethod
    def get_internal_index(self, external_label: int) -> int:
        """
        Translates internal index to external label

        :param external_label: The external label to translate
        :returns: The internal index
        """
        raise NotImplementedError()

    @abstractmethod
    def get_neighbor_indices(self, internal_index: int, copy: bool = False) -> np.ndarray:
        """
        Get the indices (internal index) of the given vertex.

        :param internal_index: The internal index to get the neighbors of
        :param copy: If True the returned neighbor indices are a copy, otherwise they reference internal graph data.
        :returns: The internal index
        """
        raise NotImplementedError()

    def get_feature_vector(self, index: int, copy: bool = False) -> np.ndarray:
        """
        Get the feature vector of the given internal index.

        :param index: The internal index to get the feature vector of
        :param copy: If True the returned feature vector is a copy, otherwise a reference to the graph data is returned.
        :returns: The feature vector of the given index
        """
        if index < 0 or index >= self.size():
            raise IndexError("Index {} out of range for size {}".format(index, self.size()))
        memory_view = self.graph_cpp.get_feature_vector(index)
        feature_vector = np.asarray(memory_view)
        if copy:
            feature_vector = np.copy(feature_vector)
        return feature_vector

    @abstractmethod
    def has_vertex(self, external_label: int) -> bool:
        """
        :returns: whether the given external label is present in the graph.
        """
        raise NotImplementedError()

    @abstractmethod
    def has_edge(self, internal_index: int, neighbor_index: int) -> bool:
        """
        :returns: whether the vertex at internal_index has an edge to the vertex at neighbor_index.
        """
        raise NotImplementedError()

    def get_entry_vertex_indices(self) -> List[int]:
        """
        Creates a list of internal indices that can be used as starting point for an anns search.
        """
        return [self.get_internal_index(0)]

    @abstractmethod
    def has_path(self, entry_vertex_indices: List[int], to_vertex: int, eps: float, k: int) -> List[ObjectDistance]:
        """
        Returns a path from one of the entry vertex indices to the given to_vertex.

        :param entry_vertex_indices: List of start vertices
        :param to_vertex: The vertex to find a path to
        :param eps: Controls how many nodes are checked during search. Lower eps values like 0.001 are faster but less
                    accurate. Higher eps values like 0.1 are slower but more accurate. Should always be greater 0.
        :param k: TODO
        """
        raise NotImplementedError()

    def search(
            self, query: np.ndarray, eps: float, k: int, filter_labels: Union[None, np.ndarray, Filter] = None,
            max_distance_computation_count: int = 0, entry_vertex_indices: Optional[List[int]] = None, threads: int = 1,
            thread_batch_size: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.

        Eps greater 0 extends the search range and takes additional graph vertices into account.

        It is possible to limit the amount of work by specifying a maximal number of distances to be calculated.
        For lower numbers it is recommended to set eps to 0 since its very unlikely the method can make use of the
        extended the search range.

        :param query: A feature vector for which similar feature vectors should searched.
        :param eps: Controls how many nodes are checked during search. Lower eps values like 0.001 are faster but less
                    accurate. Higher eps values like 0.1 are slower but more accurate. Should always be greater 0.
        :param k: The number of results that will be returned. If k is smaller than the number of vertices in the graph,
                  k is set to the number of vertices in the graph.
        :param filter_labels: A numpy array with dtype int32, that contains all labels that can be returned or an object
                              of type Filter, that limits the possible results to a given set.
                              All other labels will not be included in the result set.
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        :param entry_vertex_indices: Start point for exploratory search. If None, a reasonable default is used.
        :param threads: The number of threads to use for parallel processing. It should not excel the number of queries.
                        If set to 0, the minimum of the number of cores of this machine and the number of queries is
                        used.
        :param thread_batch_size: If threads != 1, the number of queries to search in the same thread.
        :returns: A tuple containing (indices, distances) where indices is a numpy-array of shape [n_queries, k]
                  containing the indices to the closest found neighbors to the queries.
                  Distances is a numpy-array of shape [n_queries, k] containing the distances to the closest found
                  neighbors.
        """
        # handle query shapes
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        if len(query.shape) != 2:
            raise InvalidShapeException('invalid query shape: {}'.format(query.shape))

        if k > self.size():
            warnings.warn(
                'k={} is smaller than number of vertices in graph={}. Setting k={}'.format(k, self.size(), self.size()))
            k = self.size()

        valid_dtype = self.get_feature_space().metric().get_dtype()

        query = assure_array(query, 'query', valid_dtype)
        if entry_vertex_indices is None:
            entry_vertex_indices = self.get_entry_vertex_indices()

        filter_obj = Filter.create_filter(filter_labels, self.size())

        threads = get_num_useful_threads(threads, query.shape[0])

        if thread_batch_size <= 0:
            thread_batch_size = max(query.shape[0] // (threads * 4), 1)

        indices, distances, num_results = self.graph_cpp.search(
            entry_vertex_indices, query, eps, k, filter_obj, max_distance_computation_count, threads,
            thread_batch_size
        )
        if num_results != k:
            warnings.warn('Number of results ({}) is smaller than k ({})'.format(num_results, k))
            indices = indices[:, :num_results]
            distances = distances[:, :num_results]
        return indices, distances

    @abstractmethod
    def explore(self, entry_vertex_index: int, k: int, max_distance_computation_count: int) -> ResultSet:
        """
        An exploration for similar element, limited by max_distance_computation_count

        :param entry_vertex_index: The start point for which similar feature vectors should be searched
        :param k: The number of similar feature vectors to return
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_cpp(self):
        raise NotImplementedError()


class ReadOnlyGraph(SearchGraph):
    def __init__(self, graph_cpp: deglib_cpp.ReadOnlyGraph):
        super().__init__(graph_cpp)
        if not isinstance(graph_cpp, deglib_cpp.ReadOnlyGraph):
            raise TypeError("expected ReadOnlyGraph but got {}".format(type(graph_cpp)))

    @classmethod
    def from_graph(
            cls, input_graph: SearchGraph, max_vertex_count: int = -1, feature_space: FloatSpace | None = None,
            edges_per_vertex: int = -1
    ) -> Self:
        """
        Create a read only graph from the given graph by only keeping information that is useful for searching.

        :param input_graph: The graph to build from
        :param max_vertex_count: If given the new size of the returned graph, otherwise will be taken from input graph
        :param feature_space: If given the feature space for the graph, otherwise the same as the feature space of the
                              input graph
        :param edges_per_vertex: The number of edges for the new graph. Should not be smaller than the edges of the
                                 input graph
        """
        if max_vertex_count == -1:
            max_vertex_count = input_graph.size()
        if feature_space is None:
            feature_space = input_graph.get_feature_space()
        if edges_per_vertex == -1:
            edges_per_vertex = input_graph.get_edges_per_vertex()
        return ReadOnlyGraph(deglib_cpp.read_only_graph_from_graph(
            input_graph.to_cpp(), max_vertex_count, feature_space.to_cpp(), edges_per_vertex
        ))

    def size(self) -> int:
        """
        :return: the number of vertices in the graph
        """
        return self.graph_cpp.size()

    def get_feature_space(self) -> FloatSpace:
        """
        :return: the feature space
        """
        # first two parameters get ignored
        return FloatSpace(float_space_cpp=self.graph_cpp.get_feature_space())

    def get_internal_index(self, external_label: int) -> int:
        """
        Translates internal index to external label

        :param external_label: The external label to translate
        :returns: The internal index
        """
        return self.graph_cpp.get_internal_index(external_label)

    def has_path(self, entry_vertex_indices: List[int], to_vertex: int, eps: float, k: int) -> List[ObjectDistance]:
        """
        Returns a path from one of the entry vertex indices to the given to_vertex.

        :param entry_vertex_indices: List of start vertices
        :param to_vertex: The vertex to find a path to
        :param eps: Controls how many nodes are checked during search. Lower eps values like 0.001 are faster but less
                    accurate. Higher eps values like 0.1 are slower but more accurate. Should always be greater 0.
        :param k: TODO
        """
        return [ObjectDistance(od) for od in self.graph_cpp.has_path(entry_vertex_indices, to_vertex, eps, k)]

    def get_entry_vertex_indices(self) -> List[int]:
        """
        Creates a list of internal indices that can be used as starting point for an anns search.
        """
        return self.graph_cpp.get_entry_vertex_indices()

    def get_external_label(self, internal_index: int) -> int:
        """
        Translates external labels to internal index

        :param internal_index: The internal index to translate
        :returns: The external label
        """
        return self.graph_cpp.get_external_label(internal_index)

    def explore(self, entry_vertex_index: int, k: int, max_distance_computation_count: int) -> ResultSet:
        """
        An exploration for similar element, limited by max_distance_computation_count

        :param entry_vertex_index: The start point for which similar feature vectors should be searched
        :param k: The number of similar feature vectors to return
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        """
        return ResultSet(self.graph_cpp.explore(entry_vertex_index, k, max_distance_computation_count))

    def get_edges_per_vertex(self) -> int:
        """
        :return: the number of edges of each vertex
        """
        return self.graph_cpp.get_edges_per_vertex()

    def get_neighbor_indices(self, internal_index: int, copy: bool = False) -> np.ndarray:
        """
        Get the neighbor indices (internal index) of the given vertex.

        :param internal_index: The internal index to get the neighbors of
        :param copy: If True the returned neighbor indices are a copy, otherwise they reference internal graph data.
        """
        if internal_index < 0 or internal_index >= self.size():
            raise IndexError("Index {} out of range for size {}".format(internal_index, self.size()))
        memory_view = self.graph_cpp.get_neighbor_indices(internal_index)
        neighbors = np.asarray(memory_view)
        if copy:
            neighbors = np.copy(neighbors)
        return neighbors

    def has_vertex(self, external_label: int) -> bool:
        """
        :returns: whether the given external label is present in the graph.
        """
        return self.graph_cpp.has_vertex(external_label)

    def has_edge(self, internal_index: int, neighbor_index: int) -> bool:
        """
        :returns: whether the vertex at internal_index has an edge to the vertex at neighbor_index.
        """
        return self.graph_cpp.has_edge(internal_index, neighbor_index)

    def to_cpp(self) -> deglib_cpp.ReadOnlyGraph:
        return self.graph_cpp

    def __repr__(self) -> str:
        return (f'ReadOnlyGraph(size={self.size()} edges_per_vertex={self.get_edges_per_vertex()} '
                f'dim={self.get_feature_space().dim()})')


def load_readonly_graph(path: pathlib.Path | str) -> ReadOnlyGraph:
    """
    Read a saved ReadOnlyGraph from given file. The file can be created by calling SizeBoundedGraph.save_graph().

    :param path: The path where to look for the file
    :raises FileNotFoundError: If the given file does not exist
    """
    if not os.path.isfile(path):
        raise FileNotFoundError('File "{}" could not be found'.format(path))
    return ReadOnlyGraph(deglib_cpp.load_readonly_graph(str(path)))


class MutableGraph(SearchGraph, ABC):
    @abstractmethod
    def add_vertex(self, external_label: int, feature_vector: np.ndarray) -> int:
        """
        Add a new vertex. The neighbor indices will be prefilled with a self-loop, the weights will be 0.

        :param external_label: The label for the new vertex
        :param feature_vector: The feature vector to add. This numpy array should be c-contiguous, otherwise it has to
                               be reallocated.
        :return: the internal index of the new vertex
        """
        raise NotImplementedError()

    @abstractmethod
    def remove_vertex(self, external_label: int):
        """
        Remove an existing vertex.

        :param external_label: The external label of the vertex that should be removed.
        """
        raise NotImplementedError()

    @abstractmethod
    def change_edge(self, internal_index: int, from_neighbor_index: int, to_neighbor_index: int,
                    to_neighbor_weight: float) -> bool:
        """
        Swap a neighbor with another neighbor and its weight.

        :param internal_index: vertex index which neighbors should be changed
        :param from_neighbor_index: neighbor index to remove
        :param to_neighbor_index: neighbor index to add
        :param to_neighbor_weight: weight of the neighbor to add
        :return: True if the from_neighbor_index was found and changed
        """
        raise NotImplementedError()

    @abstractmethod
    def change_edges(self, internal_index: int, neighbor_indices: np.ndarray, neighbor_weights: np.ndarray):
        """
        Change all edges of a vertex.
        The neighbor indices/weights and feature vectors will be copied.
        The neighbor array need to have enough neighbors to match the edge-per-vertex count of the graph.
        The indices in the neighbor_indices array must be sorted.

        :param internal_index: The index of the vertex for which edges should change
        :param neighbor_indices: These neighbors will be set as the new neighbors of the specified vertex
        :param neighbor_weights: These weights will be set as the new weights for the neighbors.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_neighbor_weights(self, internal_index: int, copy: bool = False) -> np.ndarray:
        """
        Get weights for each neighbor of the vertex defined by the given index.

        :param internal_index: The index that specifies the vertex
        :param copy: If True the returned neighbor weights are copied, otherwise they reference internal graph data.
        :returns: The weights of the neighbors
        """
        raise NotImplementedError()

    @abstractmethod
    def get_edge_weight(self, from_neighbor_index: int, to_neighbor_index: int) -> float:
        """
        Get the weight from vertex to another vertex. If start vertex is not a neighbor of end vertex, -1.0 is returned.

        :param from_neighbor_index: Internal index of the start vertex
        :param to_neighbor_index: Internal index of the target vertex
        :returns: If present the weight between start and target vertex, -1.0 otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def save_graph(self, path: str | pathlib.Path) -> bool:
        """
        Save graph to specified file. Creates necessary directories.

        :param path: The path where to save the file.
        """
        raise NotImplementedError()

    @abstractmethod
    def to_cpp(self):
        raise NotImplementedError()

    def remove_non_mrng_edges(self):
        """
        Remove all edges which are not MRNG conform.
        """
        deglib_cpp.remove_non_mrng_edges(self.to_cpp())


class SizeBoundedGraph(MutableGraph):
    def __init__(
            self, max_vertex_count: int, edges_per_vertex: int, feature_space: FloatSpace,
            graph_cpp: Optional[deglib_cpp.SizeBoundedGraph] = None
    ):
        """
        Create a new SizeBoundedGraph. This can be used to fit a graph to a dataset and save it for later use.

        :param max_vertex_count: The maximum number of vertices, that can be added to this graph
        :param edges_per_vertex: The number of neighbors for each vertex
        :param feature_space: The space definition for all feature vectors
        :param graph_cpp: For internal use, do not use except you know what you are doing
        """
        if graph_cpp is None:
            graph_cpp = deglib_cpp.SizeBoundedGraph(max_vertex_count, edges_per_vertex, feature_space.to_cpp())
        super().__init__(graph_cpp)
        self.feature_space = feature_space

    @staticmethod
    def create_empty(capacity: int, dims: int, edges_per_vertex: int = 32, metric: Metric = Metric.L2):
        """
        Create an empty SizeBoundedGraph.

        :param capacity: The maximal number of vertices of this graph.
        :param dims: The number of dimensions of each feature vector
        :param edges_per_vertex: Number of neighbors for each vertex. Defaults to 32.
        :param metric: The metric to measure distances between features. Defaults to L2-Metric.
        """
        return SizeBoundedGraph(capacity, edges_per_vertex, FloatSpace.create(dims, metric))

    def size(self) -> int:
        """
        :return: the number of vertices in the graph
        """
        return self.graph_cpp.size()

    def get_feature_space(self) -> FloatSpace:
        """
        :return: the feature space
        """
        return FloatSpace(self.graph_cpp.get_feature_space())

    def get_internal_index(self, external_label: int) -> int:
        """
        Translates internal index to external label

        :param external_label: The external label to translate
        :returns: The internal index
        """
        return self.graph_cpp.get_internal_index(external_label)

    def has_path(self, entry_vertex_indices: List[int], to_vertex: int, eps: float, k: int) -> List[ObjectDistance]:
        """
        Returns a path from one of the entry vertex indices to the given to_vertex.

        :param entry_vertex_indices: List of start vertices
        :param to_vertex: The vertex to find a path to
        :param eps: Controls how many nodes are checked during search. Lower eps values like 0.001 are faster but less
                    accurate. Higher eps values like 0.1 are slower but more accurate. Should always be greater 0.
        :param k: TODO
        """
        return [ObjectDistance(od) for od in self.graph_cpp.has_path(entry_vertex_indices, to_vertex, eps, k)]

    def get_entry_vertex_indices(self) -> List[int]:
        """
        Creates a list of internal indices that can be used as starting point for an anns search.
        """
        return self.graph_cpp.get_entry_vertex_indices()

    def get_external_label(self, internal_index: int) -> int:
        """
        Translates external labels to internal index

        :param internal_index: The internal index to translate
        :returns: The external label
        """
        return self.graph_cpp.get_external_label(internal_index)

    def save_graph(self, path: pathlib.Path | str):
        """
        Save graph to specified file. Creates necessary directories.

        :param path: The path where to save the file.
        """
        self.graph_cpp.save_graph(str(path))

    def get_edges_per_vertex(self) -> int:
        """
        :return: the number of edges of each vertex
        """
        return self.graph_cpp.get_edges_per_vertex()

    def add_vertex(self, external_label: int, feature_vector: np.ndarray) -> int:
        """
        Add a new vertex. The neighbor indices will be prefilled with a self-loop, the weights will be 0.

        :param external_label: The label for the new vertex
        :param feature_vector: The feature vector to add. This numpy array should be c-contiguous, otherwise it has to
                               be reallocated.
        :return: the internal index of the new vertex
        """
        valid_dtype = self.get_feature_space().metric().get_dtype()
        feature_vector = assure_array(feature_vector, 'feature_vector', valid_dtype)
        return self.graph_cpp.add_vertex(external_label, feature_vector)

    def remove_vertex(self, external_label: int):
        """
        Remove an existing vertex.

        :param external_label: The external label of the vertex that should be removed.
        """
        self.graph_cpp.remove_vertex(external_label)

    def change_edge(
            self, internal_index: int, from_neighbor_index: int, to_neighbor_index: int, to_neighbor_weight: float
    ) -> bool:
        """
        Swap a neighbor with another neighbor and its weight.

        :param internal_index: vertex index which neighbors should be changed
        :param from_neighbor_index: neighbor index to remove
        :param to_neighbor_index: neighbor index to add
        :param to_neighbor_weight: weight of the neighbor to add
        :return: True if the from_neighbor_index was found and changed
        """
        return self.graph_cpp.change_edge(internal_index, from_neighbor_index, to_neighbor_index, to_neighbor_weight)

    def change_edges(self, internal_index: int, neighbor_indices: np.ndarray, neighbor_weights: np.ndarray):
        """
        Change all edges of a vertex.
        The neighbor indices/weights and feature vectors will be copied.
        The neighbor array need to have enough neighbors to match the edge-per-vertex count of the graph.
        The indices in the neighbor_indices array must be sorted.

        :param internal_index: The index of the vertex for which edges should change
        :param neighbor_indices: These neighbors will be set as the new neighbors of the specified vertex
        :param neighbor_weights: These weights will be set as the new weights for the neighbors.
        """
        neighbor_indices = assure_array(neighbor_indices, 'neighbor_indices', np.uint32)
        neighbor_weights = assure_array(neighbor_weights, 'neighbor_weights', np.float32)
        return self.graph_cpp.change_edges(internal_index, neighbor_indices, neighbor_weights)

    def get_neighbor_weights(self, internal_index: int, copy: bool = False) -> np.ndarray:
        """
        Get weights for each neighbor of the vertex defined by the given index.

        :param internal_index: The index that specifies the vertex
        :param copy: If True the returned neighbor weights are copied, otherwise they reference internal graph data.
        :returns: The weights of the neighbors
        """
        if internal_index < 0 or internal_index >= self.size():
            raise IndexError("Index {} out of range for size {}".format(internal_index, self.size()))
        memory_view = self.graph_cpp.get_neighbor_weights(internal_index)
        weights = np.asarray(memory_view)
        if copy:
            weights = np.copy(weights)
        return weights

    def get_edge_weight(self, from_neighbor_index: int, to_neighbor_index: int) -> float:
        """
        Get the weight from vertex to another vertex. If start vertex is not a neighbor of end vertex, -1.0 is returned.

        :param from_neighbor_index: Internal index of the start vertex
        :param to_neighbor_index: Internal index of the target vertex
        :returns: If present the weight between start and target vertex, -1.0 otherwise
        """
        return self.graph_cpp.get_edge_weight(from_neighbor_index, to_neighbor_index)

    def get_neighbor_indices(self, internal_index: int, copy: bool = False) -> np.ndarray:
        """
        Get the indices (internal index) of the given vertex.

        :param internal_index: The internal index to get the neighbors of
        :param copy: If True the returned neighbor indices are a copy, otherwise they reference internal graph data.
        :returns: The internal index
        """
        if internal_index < 0 or internal_index >= self.size():
            raise IndexError("Index {} out of range for size {}".format(internal_index, self.size()))
        memory_view = self.graph_cpp.get_neighbor_indices(internal_index)
        indices = np.asarray(memory_view)
        if copy:
            indices = np.copy(indices)
        return indices

    def has_vertex(self, external_label: int) -> bool:
        """
        :returns: whether the given external label is present in the graph.
        """
        return self.graph_cpp.has_vertex(external_label)

    def has_edge(self, internal_index: int, neighbor_index: int) -> bool:
        """
        :returns: whether the vertex at internal_index has an edge to the vertex at neighbor_index.
        """
        return self.graph_cpp.has_edge(internal_index, neighbor_index)

    def explore(self, entry_vertex_index: int, k: int, max_distance_computation_count: int) -> ResultSet:
        """
        An exploration for similar element, limited by max_distance_computation_count

        :param entry_vertex_index: The start point for which similar feature vectors should be searched
        :param k: The number of similar feature vectors to return
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        """
        return ResultSet(self.graph_cpp.explore(entry_vertex_index, k, max_distance_computation_count))

    def to_cpp(self) -> deglib_cpp.SizeBoundedGraph:
        return self.graph_cpp

    def __repr__(self) -> str:
        return (f'SizeBoundedGraph(size={self.size()} edges_per_vertex={self.get_edges_per_vertex()} '
                f'dim={self.get_feature_space().dim()})')


def get_num_useful_threads(requested: int, max_limit: int):
    if requested == 0:
        requested = multiprocessing.cpu_count()
    return min(requested, max_limit)  # dont use more threads than queries


__all__ = ['load_readonly_graph', 'ReadOnlyGraph', 'SizeBoundedGraph', 'MutableGraph', 'SearchGraph']
