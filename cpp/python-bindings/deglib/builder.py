import numpy as np
import deglib_cpp
from .std import Mt19937
from .graph import SizeBoundedGraph


class EvenRegularGraphBuilder:
    # TODO: add MutableGraph Interface
    def __init__(
            self, graph: SizeBoundedGraph, rng: Mt19937, extend_k: int, extend_eps: float, improve_k: int,
            improve_eps: float, max_path_length: int = 10, swap_tries: int = 3, additional_swap_tries: int = 3
    ):
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
