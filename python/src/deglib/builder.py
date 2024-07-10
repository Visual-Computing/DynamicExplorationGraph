from typing import Optional, Callable

import numpy as np
import deglib_cpp

from .distances import Metric
from .std import Mt19937
from .graph import MutableGraph, SizeBoundedGraph
from .utils import assure_array


class EvenRegularGraphBuilder:
    def __init__(
            self, graph: MutableGraph, rng: Mt19937 | None = None, extend_k: Optional[int] = None,
            extend_eps: float = 0.2, improve_k: Optional[int] = None, improve_eps: float = 0.001,
            max_path_length: int = 10, swap_tries: int = 3, additional_swap_tries: int = 3
    ):
        """
        Create an EvenRegularBuilder that can be used to construct a graph.

        :param graph: The preallocated graph to build
        :param rng: An rng generator. Will be constructed, if set to None (default)
        :param extend_k: TODO
        :param extend_eps: TODO
        :param improve_k: TODO
        :param improve_eps: TODO
        :param max_path_length: TODO
        :param swap_tries: TODO
        :param additional_swap_tries: TODO
        """
        if rng is None:
            rng = Mt19937()
        if improve_k is None:
            improve_k = graph.get_edges_per_vertex()
        if extend_k is None:
            extend_k = graph.get_edges_per_vertex()
        self.builder_cpp = deglib_cpp.EvenRegularGraphBuilder(
            graph.to_cpp(), rng.to_cpp(), extend_k, extend_eps, improve_k, improve_eps, max_path_length, swap_tries,
            additional_swap_tries
        )
        self.graph = graph
        self.rng = rng

    @staticmethod
    def build_from_data(
            data: np.ndarray, edges_per_vertex: int = 32, capacity: int = -1, metric: Metric = Metric.L2,
            rng: Mt19937 | None = None, extend_k: Optional[int] = None, extend_eps: float = 0.2,
            improve_k: Optional[int] = None, improve_eps: float = 0.001, max_path_length: int = 10, swap_tries: int = 3,
            additional_swap_tries: int = 3
    ) -> SizeBoundedGraph:
        """
        Create a new graph built from the given data.

        :param data: numpy array with shape [N, D], where N is the number of samples and D is the number of dimensions
                     per feature.
        :param capacity: The maximal number of vertices of this graph. Defaults to the number of samples in data.
        :param edges_per_vertex: The number of edges per vertex for the graph. Defaults to 32.
        :param metric: The metric to measure distances between features. Defaults to L2-Metric.
        :param rng: An rng generator. Will be constructed, if set to None (default)
        :param extend_k: TODO
        :param extend_eps: TODO
        :param improve_k: TODO
        :param improve_eps: TODO
        :param max_path_length: TODO
        :param swap_tries: TODO
        :param additional_swap_tries: TODO
        """
        if capacity <= 0:
            capacity = data.shape[0]
        graph = SizeBoundedGraph.create_empty(capacity, data.shape[1], edges_per_vertex, metric)
        builder = EvenRegularGraphBuilder(
            graph, rng, extend_k=extend_k, extend_eps=extend_eps, improve_k=improve_k, improve_eps=improve_eps,
            max_path_length=max_path_length, swap_tries=swap_tries, additional_swap_tries=additional_swap_tries
        )

        for i, vec in enumerate(data):
            vec: np.ndarray
            builder.add_entry(i, vec)

        builder.build()

        return graph

    def add_entry(self, external_label: int, feature: np.ndarray):
        """
        Add entry that should be added to the graph.

        :param external_label: The label, that names the added vertex
        :param feature: The feature that should be added to the graph.
        """
        feature = assure_array(feature, 'feature', np.float32)
        self.builder_cpp.add_entry(external_label, feature)

    def remove_entry(self, label: int):
        """
        Remove an entry from the graph.

        :param label: The label, that names the graph vertex
        """
        self.builder_cpp.remove_entry(label)

    def build(
            self, improvement_callback: Callable[[deglib_cpp.BuilderStatus], None] | None = None, infinite: bool = False
    ):
        """
        Build the graph. This could be run on a separate thread in an infinite loop. Call stop() to end this process.

        :param improvement_callback: The callback that is called after each step of the build process. A BuilderStatus
                                     is the only argument to the function.
        :param infinite: If set to True, blocks indefinitely, until the stop() function is called. Can be used, if
                         build() is run in a separate thread.
        """
        if improvement_callback is None:
            self.builder_cpp.build_silent(infinite)
        else:
            self.builder_cpp.build(improvement_callback, infinite)

    def stop(self):
        self.builder_cpp.stop()

    def __repr__(self):
        return 'EvenRegularGraphBuilder(vertices_added={})'.format(self.graph.size())
