from typing import Optional

import numpy as np
import deglib_cpp
from .std import Mt19937
from .graph import SizeBoundedGraph


class EvenRegularGraphBuilder:
    # TODO: add MutableGraph Interface
    def __init__(
            self, graph: SizeBoundedGraph, rng: Mt19937, extend_k: Optional[int] = None, extend_eps: float = 0.2,
            improve_k: Optional[int] = None, improve_eps: float = 0.001, max_path_length: int = 10, swap_tries: int = 3,
            additional_swap_tries: int = 3
    ):
        if improve_k is None:
            improve_k = graph.get_edges_per_vertex()
        if extend_k is None:
            extend_k = graph.get_edges_per_vertex()
        self.builder_cpp = deglib_cpp.EvenRegularGraphBuilder(
            graph.to_cpp(), rng.to_cpp(), extend_k, extend_eps, improve_k, improve_eps, max_path_length, swap_tries,
            additional_swap_tries
        )

    def add_entry(self, label: int, feature: np.ndarray):
        self.builder_cpp.add_entry(label, feature)

    def remove_entry(self, label: int):
        self.builder_cpp.remove_entry(label)

    def build(self, improvement_callback, infinite: bool = False):
        self.builder_cpp.build(improvement_callback, infinite)
