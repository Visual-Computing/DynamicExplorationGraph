from typing import List, Optional

import numpy as np

import deglib_cpp
import pathlib

from .distances import FloatSpace, Metric
from .utils import assure_array


# TODO: safety checks
# TODO: wrap ResultSet in python


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


def load_readonly_graph(path: pathlib.Path | str) -> ReadOnlyGraph:
    return ReadOnlyGraph(deglib_cpp.load_readonly_graph(str(path)))


# TODO: add MutableGraph and SearchGraph interfaces
class SizeBoundedGraph:
    def __init__(
            self, max_vertex_count: int, edges_per_vertex: int, feature_space: FloatSpace,
            graph_cpp: Optional[deglib_cpp.SizeBoundedGraph] = None
    ):
        if graph_cpp is None:
            graph_cpp = deglib_cpp.SizeBoundedGraph(max_vertex_count, edges_per_vertex, feature_space._to_cpp())
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

    def _to_cpp(self) -> deglib_cpp.SizeBoundedGraph:
        return self.graph_cpp

    def save_graph(self, graph_file: pathlib.Path | str):
        print('save to:', graph_file)
        print('self graph cpp type:', type(self.graph_cpp))
        self.graph_cpp.save_graph(str(graph_file))


__all__ = ['load_readonly_graph', 'ReadOnlyGraph', 'SizeBoundedGraph']
