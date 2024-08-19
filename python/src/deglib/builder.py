import enum
import sys
import time
from typing import Optional, Callable, List, Iterable

import numpy as np
import deglib_cpp

from .distances import Metric
from .std import Mt19937
from .graph import MutableGraph, SizeBoundedGraph
from .utils import assure_array, InvalidShapeException


class LID(enum.IntEnum):
    Unknown = deglib_cpp.LID.Unknown
    High = deglib_cpp.LID.High
    Low = deglib_cpp.LID.Low

    def to_cpp(self) -> deglib_cpp.LID:
        if self == LID.Unknown:
            return deglib_cpp.LID.Unknown
        elif self == LID.High:
            return deglib_cpp.LID.High
        elif self == LID.Low:
            return deglib_cpp.LID.Low


class EvenRegularGraphBuilder:
    def __init__(
            self, graph: MutableGraph, rng: Mt19937 | None = None, lid: LID = LID.Unknown,
            extend_k: Optional[int] = None, extend_eps: float = 0.2, improve_k: Optional[int] = None,
            improve_eps: float = 0.001, max_path_length: int = 10, swap_tries: int = 3, additional_swap_tries: int = 3
    ):
        """
        Create an EvenRegularBuilder that can be used to construct a graph.

        :param graph: The preallocated graph to build
        :param rng: An rng generator. Will be constructed, if set to None (default)
        :param lid: TODO
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
            graph.to_cpp(), rng.to_cpp(), lid.to_cpp(), extend_k, extend_eps, improve_k, improve_eps, max_path_length,
            swap_tries, additional_swap_tries
        )
        self.graph = graph
        self.rng = rng
        self.lid = lid

    def add_entry(self, external_label: int | Iterable[int] | np.ndarray, feature: np.ndarray):
        """
        Add entry that should be added to the graph. Can also add a batch of entries with shape [N, D], where N is the
        number of entries, D is the dimensionality. In this case external_label should be a list of ints (or numpy
        array) with length N.

        :param external_label: The label or the batch of labels, that names the added vertex. If this is a numpy array,
        it should have dtype uint32.
        :param feature: The feature or the batch of features, that should be added to the graph.
        """
        # standardize feature shape
        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)
        if len(feature.shape) != 2:
            raise InvalidShapeException('invalid feature shape: {}'.format(feature.shape))
        valid_dtype = self.graph.get_feature_space().metric().get_dtype()
        feature = assure_array(feature, 'feature', valid_dtype)

        # standardize external label
        if isinstance(external_label, int):
            external_label = np.array([external_label], dtype=np.uint32)
        elif isinstance(external_label, Iterable) and not isinstance(external_label, np.ndarray):
            external_label = np.array(external_label, dtype=np.uint32)
        assure_array(external_label, 'external_label', np.uint32)

        assert feature.shape[0] == external_label.shape[0], 'Got {} features, but {} labels'.format(
            feature.shape[0], external_label.shape[0]
        )

        self.builder_cpp.add_entry(external_label, feature)

    def remove_entry(self, external_label: int):
        """
        Remove an entry from the graph.

        :param external_label: The external label, that names the graph vertex
        """
        self.builder_cpp.remove_entry(external_label)

    def get_num_new_entries(self) -> int:
        """
        :returns: the number of entries to add
        """
        return self.builder_cpp.get_num_new_entries()

    def get_num_remove_entries(self) -> int:
        """
        :returns: the number of entries to remove
        """
        return self.builder_cpp.get_num_remove_entries()

    def set_thread_count(self, thread_count: int):
        """
        Set the number of thread used to build the graph.

        :param thread_count: number of thread used to build the graph
        """
        return self.builder_cpp.set_thread_count(thread_count)

    def set_batch_size(self, batch_size: int):
        """
        During construction several thread can build the graph. To minimize synchronization batches are used.
        The batch size defines how many vertices should be added at once without calling the callback of the build method.

        :param batch_size: size of the batch used to build the graph
        """
        return self.builder_cpp.set_batch_size(batch_size)

    def build(
            self, callback: Callable[[deglib_cpp.BuilderStatus], None] | str | None = None,
            infinite: bool = False
    ):
        """
        Build the graph. This could be run on a separate thread in an infinite loop. Call stop() to end this process.

        :param callback: The callback that is called after each step of the build process. A BuilderStatus
                                     is the only argument to the function.
                                     If None nothing is printed.
                                     If callback is the string "progress", a simple progress bar is printed to stdout.
        :param infinite: If set to True, blocks indefinitely, until the stop() function is called. Can be used, if
                         build() is run in a separate thread.
        """
        if callback is None:
            self.builder_cpp.build_silent(infinite)
        else:
            if not infinite and callback == 'progress':
                callback = ProgressCallback(self.get_num_new_entries(), self.get_num_remove_entries())
            self.builder_cpp.build(callback, infinite)

    def stop(self):
        self.builder_cpp.stop()

    def __repr__(self):
        return 'EvenRegularGraphBuilder(vertices_added={})'.format(self.graph.size())


def build_from_data(
        data: np.ndarray, labels: Iterable[int] | None = None, edges_per_vertex: int = 32, capacity: int = -1,
        metric: Metric = Metric.L2, rng: Mt19937 | None = None, lid: LID = LID.Unknown, extend_k: Optional[int] = None,
        extend_eps: float = 0.2, improve_k: Optional[int] = None, improve_eps: float = 0.001, max_path_length: int = 10,
        swap_tries: int = 3, additional_swap_tries: int = 3,
        callback: Callable[[deglib_cpp.BuilderStatus], None] | str | None = None
) -> SizeBoundedGraph:
    """
    Create a new graph built from the given data.
    Infinite building is not supported, as you dont have any reference to the builder to stop it using this
    function.

    :param data: numpy array with shape [N, D], where N is the number of samples and D is the number of dimensions
                 per feature.
    :param labels: The labels for each data entry. Should have shape [N]. If this is a numpy array it should have dtype
                   uint32. If None labels starting from 0 will be created.
    :param capacity: The maximal number of vertices of this graph. Defaults to the number of samples in data.
    :param edges_per_vertex: The number of edges per vertex for the graph. Defaults to 32.
    :param metric: The metric to measure distances between features. Defaults to L2-Metric.
    :param rng: An rng generator. Will be constructed, if set to None (default)
    :param lid: TODO
    :param extend_k: TODO
    :param extend_eps: TODO
    :param improve_k: TODO
    :param improve_eps: TODO
    :param max_path_length: TODO
    :param swap_tries: TODO
    :param additional_swap_tries: TODO
    :param callback: The callback that is called after each step of the build process. A BuilderStatus
                                 is the only argument to the function.
                                 If None nothing is printed.
                                 If callback is the string "progress", a simple progress bar is printed to stdout.
    """
    if capacity <= 0:
        capacity = data.shape[0]
    graph = SizeBoundedGraph.create_empty(capacity, data.shape[1], edges_per_vertex, metric)
    builder = EvenRegularGraphBuilder(
        graph, rng, lid=lid, extend_k=extend_k, extend_eps=extend_eps, improve_k=improve_k,
        improve_eps=improve_eps, max_path_length=max_path_length, swap_tries=swap_tries,
        additional_swap_tries=additional_swap_tries
    )

    if labels is None:
        labels = np.arange(data.shape[0], dtype=np.uint32)

    builder.add_entry(labels, data)

    builder.build(callback=callback)

    return graph


class ProgressCallback:
    def __init__(
            self, num_new_entries: int, num_remove_entries: int, bar_length: int = 60, min_print_interval: float = 0.1
    ):
        self.num_new_entries = num_new_entries
        self.num_remove_entries = num_remove_entries
        self.bar_length = bar_length
        self.maximal = self.num_new_entries + self.num_remove_entries
        self.len_max = len(str(self.maximal))
        self.last_print_time = 0
        self.min_print_interval = min_print_interval

    def __call__(self, builder_status: deglib_cpp.BuilderStatus):
        current_time = time.time()
        num_steps = builder_status.added + builder_status.deleted
        last_step = num_steps == self.maximal
        if current_time - self.last_print_time >= self.min_print_interval or last_step:
            self.last_print_time = current_time

            progress = num_steps / self.maximal
            block = int(self.bar_length * progress)
            bar = '#' * block + '-' * (self.bar_length - block)
            percentage = progress * 100
            sys.stdout.write(f'\r{percentage:6.2f}% [{bar}] ({num_steps:{self.len_max}} / {self.maximal})')
            if last_step:
                sys.stdout.write('\n')  # newline at the end
            sys.stdout.flush()
