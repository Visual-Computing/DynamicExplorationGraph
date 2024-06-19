from typing import List, Optional
from abc import ABC, abstractmethod

import numpy as np

import deglib_cpp
import pathlib

from .distances import FloatSpace, Metric, SpaceInterface
from .utils import assure_array


# TODO: safety checks
# TODO: wrap ResultSet in python


class SearchGraph(ABC):
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_edges_per_vertex(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_feature_space(self) -> SpaceInterface:
        raise NotImplementedError()

    @abstractmethod
    def get_external_label(self, internal_index: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_internal_index(self, external_label: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_neighbor_indices(self, internal_index: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_feature_vector(self, internal_index: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def has_vertex(self, external_label: int) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def has_edge(self, internal_index: int, neighbor_index: int) -> bool:
        raise NotImplementedError()

    def get_entry_vertex_indices(self) -> List[int]:
        return [self.get_internal_index(0)]

    # @abstractmethod
    # def has_path(self, _entry_vertex_indices: List[int], _to_vertex: int, _eps: float, _k: int) -> List[deglib_cpp.ObjectDistance]:
    #     """
    #     Perform a search but stops when the to_vertex was found.
    #     """
    #     return NotImplemented()

    @abstractmethod
    def search(self, query: np.ndarray, eps: float, k: int, max_distance_computation_count: int = 0, entry_vertex_indices: Optional[List[int]] = None) -> deglib_cpp.ResultSet:
        """
        Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.

        Eps greater 0 extends the search range and takes additional graph vertices into account.

        It is possible to limit the amount of work by specifing a maximal number of distances to be calculated.
        For lower numbers it is recommended to set eps to 0 since its very unlikly the method can make use of the extended the search range.
        """
        raise NotImplementedError()

    @abstractmethod
    def explore(self, entry_vertex_index: int, k: int, max_distance_computation_count: int) -> deglib_cpp.ResultSet:
        """
        A exploration for similar element, limited by max_distance_computation_count
        """
        raise NotImplementedError()


class ReadOnlyGraph(SearchGraph):
    def __init__(self, graph_cpp: deglib_cpp.ReadOnlyGraph):
        self.graph_cpp = graph_cpp

    def size(self) -> int:
        return self.graph_cpp.size()

    def get_feature_space(self) -> FloatSpace:
        # first two parameters get ignored
        return FloatSpace(0, Metric.L2, float_space_cpp=self.graph_cpp.get_feature_space())

    # TODO: copy=True parameter
    def get_feature_vector(self, index) -> np.ndarray:
        memory_view = self.graph_cpp.get_feature_vector(index)
        feature_vector = np.asarray(memory_view)
        return feature_vector

    def get_internal_index(self, index: int) -> int:
        return self.graph_cpp.get_internal_index(index)

    def search(self, query: np.ndarray, eps: float, k: int, max_computation_count: int = 0,
               entry_vertex_indices: Optional[List[int]] = None) -> deglib_cpp.ResultSet:
        query = assure_array(query, 'query', np.float32)
        if entry_vertex_indices is None:
            entry_vertex_indices = self.get_entry_vertex_indices()
        return self.graph_cpp.search(entry_vertex_indices, query, eps, k, max_computation_count)

    def get_entry_vertex_indices(self) -> List[int]:
        return self.graph_cpp.get_entry_vertex_indices()

    def get_external_label(self, index: int) -> int:
        return self.graph_cpp.get_external_label(index)

    def explore(self, entry_vertex_index: int, k: int, max_distance_count: int):
        return self.graph_cpp.explore(entry_vertex_index, k, max_distance_count)

    def get_edges_per_vertex(self) -> int:
        return self.graph_cpp.get_edges_per_vertex()

    def get_neighbor_indices(self, internal_index: int) -> np.ndarray:
        return self.graph_cpp.get_neighbor_indices(internal_index)

    def has_vertex(self, external_label: int) -> bool:
        return self.graph_cpp.has_vertex(external_label)

    def has_edge(self, internal_index: int, neighbor_index: int) -> bool:
        return self.graph_cpp.has_edge(internal_index, neighbor_index)


def load_readonly_graph(path: pathlib.Path | str) -> ReadOnlyGraph:
    return ReadOnlyGraph(deglib_cpp.load_readonly_graph(str(path)))


class MutableGraph(SearchGraph, ABC):
    @abstractmethod
    def add_vertex(self, external_label: int, feature_vector: np.ndarray) -> int:
        """
        Add a new vertex. The neighbor indices will be prefilled with a self-loop, the weights will be 0.
        :return: the internal index of the new vertex
        """
        raise NotImplementedError()

    @abstractmethod
    def remove_vertex(self, external_label: int):
        """
        Remove an existing vertex.
        """
        raise NotImplementedError()

    @abstractmethod
    def change_edge(self, internal_index: int, from_neighbor_index: int, to_neighbor_index: int, to_neighbor_weight: float) -> bool:
        """
        Swap a neighbor with another neighbor and its weight.

        @param internal_index vertex index which neighbors should be changed
        @param from_neighbor_index neighbor index to remove
        @param to_neighbor_index neighbor index to add
        @param to_neighbor_weight weight of the neighbor to add
        @return true if the from_neighbor_index was found and changed
        """
        raise NotImplementedError()

    @abstractmethod
    def change_edges(self, internal_index: int, neighbor_indices: np.ndarray, neighbor_weights: np.ndarray):
        """
        Change all edges of a vertex.
        The neighbor indices/weights and feature vectors will be copied.
        The neighbor array need to have enough neighbors to match the edge-per-vertex count of the graph.
        The indices in the neighbor_indices array must be sorted.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_neighbor_weights(self, internal_index: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_edge_weight(self, from_neighbor_index: int, to_neighbor_index: int) -> float:
        raise NotImplementedError()

    @abstractmethod
    def save_graph(self, path_to_graph: str | pathlib.Path) -> bool:
        raise NotImplementedError()


class SizeBoundedGraph(MutableGraph):
    def __init__(
            self, max_vertex_count: int, edges_per_vertex: int, feature_space: FloatSpace,
            graph_cpp: Optional[deglib_cpp.SizeBoundedGraph] = None
    ):
        if graph_cpp is None:
            graph_cpp = deglib_cpp.SizeBoundedGraph(max_vertex_count, edges_per_vertex, feature_space.to_cpp())
        self.graph_cpp = graph_cpp

    def size(self) -> int:
        return self.graph_cpp.size()

    def get_feature_space(self) -> FloatSpace:
        # first two parameters get ignored
        return FloatSpace(0, Metric.L2, float_space_cpp=self.graph_cpp.get_feature_space())

    # TODO: copy=True parameter
    def get_feature_vector(self, index) -> np.ndarray:
        memory_view = self.graph_cpp.get_feature_vector(index)
        feature_vector = np.asarray(memory_view)
        return feature_vector

    def get_internal_index(self, index: int) -> int:
        return self.graph_cpp.get_internal_index(index)

    # TODO: add search function
    # def search(
    #         self, entry_vertex_indices: List[int], query: np.ndarray, eps: float, k: int,
    #         max_computation_count: int = 0
    # ) -> deglib_cpp.ResultSet:
    #     query = assure_array(query, 'query', np.float32)
    #     return self.graph_cpp.search(entry_vertex_indices, query, eps, k, max_computation_count)

    def get_entry_vertex_indices(self) -> List[int]:
        return self.graph_cpp.get_entry_vertex_indices()

    def get_external_label(self, index: int) -> int:
        return self.graph_cpp.get_external_label(index)

    def to_cpp(self) -> deglib_cpp.SizeBoundedGraph:
        return self.graph_cpp

    def save_graph(self, graph_file: pathlib.Path | str):
        print('save to:', graph_file)
        self.graph_cpp.save_graph(str(graph_file))

    def get_edges_per_vertex(self) -> int:
        return self.graph_cpp.get_edges_per_vertex()

    def add_vertex(self, external_label: int, feature_vector: np.ndarray) -> int:
        return self.graph_cpp.add_vertex(external_label, feature_vector)

    def remove_vertex(self, external_label: int):
        self.graph_cpp.remove_vertex(external_label)

    def change_edge(self, internal_index: int, from_neighbor_index: int, to_neighbor_index: int,
                    to_neighbor_weight: float) -> bool:
        return self.graph_cpp.change_edge(internal_index, from_neighbor_index, to_neighbor_index)

    def change_edges(self, internal_index: int, neighbor_indices: np.ndarray, neighbor_weights: np.ndarray):
        neighbor_indices = assure_array(neighbor_indices, 'neighbor_indices', np.uint32)
        neighbor_weights = assure_array(neighbor_weights, 'neighbor_weights', np.float32)
        return self.graph_cpp.change_edges(internal_index, neighbor_indices, neighbor_weights)

    def get_neighbor_weights(self, internal_index: int) -> np.ndarray:
        return self.graph_cpp.get_neighbor_weights(internal_index)

    def get_edge_weight(self, from_neighbor_index: int, to_neighbor_index: int) -> float:
        return self.graph_cpp.get_edge_weight(from_neighbor_index, to_neighbor_index)

    def get_neighbor_indices(self, internal_index: int) -> np.ndarray:
        return self.graph_cpp.get_neighbor_indices(internal_index)

    def has_vertex(self, external_label: int) -> bool:
        return self.graph_cpp.has_vertex(external_label)

    def has_edge(self, internal_index: int, neighbor_index: int) -> bool:
        return self.graph_cpp.has_edge(internal_index, neighbor_index)

    def search(self, query: np.ndarray, eps: float, k: int, max_computation_count: int = 0,
               entry_vertex_indices: Optional[List[int]] = None) -> deglib_cpp.ResultSet:
        query = assure_array(query, 'query', np.float32)
        if entry_vertex_indices is None:
            entry_vertex_indices = self.get_entry_vertex_indices()
        return self.graph_cpp.search(entry_vertex_indices, query, eps, k, max_computation_count)

    def explore(self, entry_vertex_index: int, k: int, max_distance_computation_count: int) -> deglib_cpp.ResultSet:
        return self.graph_cpp.explore(entry_vertex_index, k, max_distance_computation_count)


__all__ = ['load_readonly_graph', 'ReadOnlyGraph', 'SizeBoundedGraph']
