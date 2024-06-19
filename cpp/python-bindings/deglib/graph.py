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
        return NotImplemented()

    @abstractmethod
    def get_edges_per_vertex(self) -> int:
        return NotImplemented()

    @abstractmethod
    def get_feature_space(self) -> SpaceInterface:
        return NotImplemented()

    @abstractmethod
    def get_external_label(self, _internal_index: int) -> int:
        return NotImplemented()

    @abstractmethod
    def get_internal_index(self, _external_label: int) -> int:
        return NotImplemented()

    @abstractmethod
    def get_neighbor_indices(self, _internal_index: int) -> np.ndarray:
        return NotImplemented()

    @abstractmethod
    def get_feature_vector(self, _internal_index: int) -> np.ndarray:
        return NotImplemented()

    @abstractmethod
    def has_vertex(self, _external_label: int) -> bool:
        return NotImplemented()

    @abstractmethod
    def has_edge(self, _internal_index: int, _neighbor_index: int) -> bool:
        return NotImplemented()

    @abstractmethod
    def get_entry_vertex_indices(self) -> List[int]:
        return [self.get_internal_index(0)]

    # @abstractmethod
    # def has_path(self, _entry_vertex_indices: List[int], _to_vertex: int, _eps: float, _k: int) -> List[deglib_cpp.ObjectDistance]:
    #     """
    #     Perform a search but stops when the to_vertex was found.
    #     """
    #     return NotImplemented()

    @abstractmethod
    def search(self, _query: np.ndarray, _eps: float, _k: int, _max_distance_computation_count: int = 0, _entry_vertex_indices: Optional[List[int]] = None) -> deglib_cpp.ResultSet:
        """
        Approximate nearest neighbor search based on yahoo's range search algorithm for graphs.

        Eps greater 0 extends the search range and takes additional graph vertices into account.

        It is possible to limit the amount of work by specifing a maximal number of distances to be calculated.
        For lower numbers it is recommended to set eps to 0 since its very unlikly the method can make use of the extended the search range.
        """
        return NotImplemented()

    @abstractmethod
    def explore(self, _entry_vertex_index: int, _k: int, _max_distance_computation_count: int) -> deglib_cpp.ResultSet:
        """
        A exploration for similar element, limited by max_distance_computation_count
        """
        return NotImplemented()


class ReadOnlyGraph:
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

    def search(
            self, entry_vertex_indices: List[int], query: np.ndarray, eps: float, k: int,
            max_computation_count: int = 0
    ) -> deglib_cpp.ResultSet:
        query = assure_array(query, 'query', np.float32)
        return self.graph_cpp.search(entry_vertex_indices, query, eps, k, max_computation_count)

    def get_entry_vertex_indices(self) -> List[int]:
        return self.graph_cpp.get_entry_vertex_indices()

    def get_external_label(self, index: int) -> int:
        return self.graph_cpp.get_external_label(index)

    def explore(self, entry_vertex_index: int, k: int, max_distance_count: int):
        return self.graph_cpp.explore(entry_vertex_index, k, max_distance_count)


def load_readonly_graph(path: pathlib.Path | str) -> ReadOnlyGraph:
    return ReadOnlyGraph(deglib_cpp.load_readonly_graph(str(path)))


class MutableGraph():
    pass


# TODO: add MutableGraph and SearchGraph interfaces
class SizeBoundedGraph:
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


__all__ = ['load_readonly_graph', 'ReadOnlyGraph', 'SizeBoundedGraph']
