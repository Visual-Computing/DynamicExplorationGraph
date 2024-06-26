import os
from typing import List, Optional
from abc import ABC, abstractmethod

import numpy as np

import deglib_cpp
import pathlib

from .distances import FloatSpace, Metric, SpaceInterface
from .search import ResultSet, ObjectDistance
from .utils import assure_array


# TODO: safety checks


class SearchGraph(ABC):
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

    @abstractmethod
    def get_feature_vector(self, internal_index: int, copy: bool = False) -> np.ndarray:
        """
        Get the feature vector of the given internal index.

        :param internal_index: The internal index to get the feature vector of.
        :param copy: If True the returned feature vector is a copy, otherwise a reference to the graph data is returned.
        :returns: The feature vector of the given index
        """
        raise NotImplementedError()

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
        """
        raise NotImplementedError()

    @abstractmethod
    def search(self, query: np.ndarray, eps: float, k: int, max_distance_computation_count: int = 0,
               entry_vertex_indices: Optional[List[int]] = None) -> ResultSet:
        """
        Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.

        Eps greater 0 extends the search range and takes additional graph vertices into account.

        It is possible to limit the amount of work by specifying a maximal number of distances to be calculated.
        For lower numbers it is recommended to set eps to 0 since its very unlikely the method can make use of the
        extended the search range.

        :param query: A feature vector for which similar feature vectors should searched.
        :param eps: TODO
        :param k: The number of results that will be returned
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        :param entry_vertex_indices: Start point for exploratory search. If None, a reasonable default is used.
        """
        raise NotImplementedError()

    @abstractmethod
    def explore(self, entry_vertex_index: int, k: int, max_distance_computation_count: int) -> ResultSet:
        """
        A exploration for similar element, limited by max_distance_computation_count

        :param entry_vertex_index: The start point for which similar feature vectors should be searched
        :param k: The number of similar feature vectors to return
        :param max_distance_computation_count: TODO
        """
        raise NotImplementedError()

    @abstractmethod
    def to_cpp(self):
        raise NotImplementedError()


class ReadOnlyGraph(SearchGraph):
    def __init__(self, graph_cpp: deglib_cpp.ReadOnlyGraph):
        if not isinstance(graph_cpp, deglib_cpp.ReadOnlyGraph):
            raise TypeError("expected ReadOnlyGraph but got {}".format(type(graph_cpp)))
        self.graph_cpp = graph_cpp

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
        return FloatSpace(0, Metric.L2, float_space_cpp=self.graph_cpp.get_feature_space())

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

    def get_internal_index(self, external_label: int) -> int:
        """
        Translates internal index to external label

        :param external_label: The external label to translate
        :returns: The internal index
        """
        return self.graph_cpp.get_internal_index(external_label)

    def search(self, query: np.ndarray, eps: float, k: int, max_distance_computation_count: int = 0,
               entry_vertex_indices: Optional[List[int]] = None) -> ResultSet:
        """
        Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.

        Eps greater 0 extends the search range and takes additional graph vertices into account.

        It is possible to limit the amount of work by specifying a maximal number of distances to be calculated.
        For lower numbers it is recommended to set eps to 0 since its very unlikely the method can make use of the
        extended the search range.

        :param query: A feature vector for which similar feature vectors should searched.
        :param eps: TODO
        :param k: The number of results that will be returned
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        :param entry_vertex_indices: Start point for exploratory search. If None, a reasonable default is used.
        """
        query = assure_array(query, 'query', np.float32)
        if entry_vertex_indices is None:
            entry_vertex_indices = self.get_entry_vertex_indices()
        return ResultSet(self.graph_cpp.search(entry_vertex_indices, query, eps, k, max_distance_computation_count))

    def has_path(self, entry_vertex_indices: List[int], to_vertex: int, eps: float, k: int) -> List[ObjectDistance]:
        """
        Returns a path from one of the entry vertex indices to the given to_vertex.

        :param entry_vertex_indices: List of start vertices
        :param to_vertex: The vertex to find a path to
        :param eps: TODO
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
        self.graph_cpp = graph_cpp

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
        return FloatSpace(0, Metric.L2, float_space_cpp=self.graph_cpp.get_feature_space())

    def get_feature_vector(self, index: int, copy: bool = False) -> np.ndarray:
        """
        Get the feature vector of the given internal index.

        :param index: The internal index to get the feature vector of
        :param copy: If True the returned feature vector is a copy, otherwise a reference to the graph data is returned.
        :returns: The feature vector of the given index
        """
        memory_view = self.graph_cpp.get_feature_vector(index)
        feature_vector = np.asarray(memory_view)
        if copy:
            feature_vector = np.copy(feature_vector)
        return feature_vector

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
        :param eps: TODO
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
        feature_vector = assure_array(feature_vector, 'feature_vector', np.float32)
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

    def search(
            self, query: np.ndarray, eps: float, k: int, max_distance_computation_count: int = 0,
            entry_vertex_indices: Optional[List[int]] = None
    ) -> ResultSet:
        """
        Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.

        Eps greater 0 extends the search range and takes additional graph vertices into account.

        It is possible to limit the amount of work by specifying a maximal number of distances to be calculated.
        For lower numbers it is recommended to set eps to 0 since its very unlikely the method can make use of the
        extended the search range.

        :param query: A feature vector for which similar feature vectors should searched.
        :param eps: TODO
        :param k: The number of results that will be returned
        :param max_distance_computation_count: Limit the number of distance calculations. If set to 0 this is ignored.
        :param entry_vertex_indices: Start point for exploratory search. If None, a reasonable default is used.
        """
        query = assure_array(query, 'query', np.float32)
        if entry_vertex_indices is None:
            entry_vertex_indices = self.get_entry_vertex_indices()
        return ResultSet(self.graph_cpp.search(entry_vertex_indices, query, eps, k, max_distance_computation_count))

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


__all__ = ['load_readonly_graph', 'ReadOnlyGraph', 'SizeBoundedGraph', 'MutableGraph', 'SearchGraph']
