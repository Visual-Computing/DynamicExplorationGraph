from typing import List

import numpy as np

import deglib_cpp
import pathlib

from .utils import assure_array


# TODO: safety checks
# TODO: wrap ResultSet in python


class ReadOnlyGraph:
    def __init__(self, graph_cpp: deglib_cpp.ReadOnlyGraph):
        self.graph_cpp = graph_cpp

    def size(self) -> int:
        return self.graph_cpp.size()

    def get_feature_space(self) -> deglib_cpp.FloatSpace:
        return self.graph_cpp.get_feature_space()

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


__all__ = ['load_readonly_graph']
