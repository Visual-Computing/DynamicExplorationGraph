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


class OptimizationTarget(enum.IntEnum):
    StreamingData = deglib_cpp.OptimizationTarget.StreamingData
    HighLID = deglib_cpp.OptimizationTarget.HighLID
    LowLID = deglib_cpp.OptimizationTarget.LowLID

    def to_cpp(self) -> deglib_cpp.OptimizationTarget:
        if self == OptimizationTarget.StreamingData:
            return deglib_cpp.OptimizationTarget.StreamingData
        elif self == OptimizationTarget.HighLID:
            return deglib_cpp.OptimizationTarget.HighLID
        elif self == OptimizationTarget.LowLID:
            return deglib_cpp.OptimizationTarget.LowLID


class EvenRegularGraphBuilder:
    def __init__(
            self, graph: MutableGraph, rng: Mt19937 | None = None,
            optimization_target: OptimizationTarget = OptimizationTarget.LowLID,
            extend_k: int = 0, extend_eps: float = 0.1,
            improve_k: int = 0, improve_eps: float = 0.0,
            max_path_length: int = 5, swap_tries: int = 0, additional_swap_tries: int = 0
    ):
        """
        Create an EvenRegularBuilder that can be used to construct a graph.

        :param graph: The preallocated graph to build
        :param rng: An rng generator. Will be constructed, if set to None (default)
        :param optimization_target: TODO
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
        if extend_k < graph.get_edges_per_vertex():
            extend_k = graph.get_edges_per_vertex()
        self.builder_cpp = deglib_cpp.EvenRegularGraphBuilder(
            graph.to_cpp(), rng.to_cpp(), optimization_target.to_cpp(), extend_k, extend_eps, improve_k, improve_eps, max_path_length,
            swap_tries, additional_swap_tries
        )
        self.graph = graph
        self.rng = rng
        self.optimization_target = optimization_target

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
        Set the number of threads used to extend the graph during building.

        When the thread count is greater than 1 and the optimization target is not StreamingData,
        the builder will utilize multiple threads to add elements to the graph in parallel.
        By default, all available CPU cores/threads are used unless specified.
        Note: The order in which elements are added is not guaranteed when using multiple threads.

        :param thread_count: Number of threads to use for graph extension.
        :type thread_count: int
        :return: None
        """
        return self.builder_cpp.set_thread_count(thread_count)

    def set_batch_size(self, tasks_per_batch: int, task_size: int):
        """
        Set the batch size parameters for parallel graph construction.

        The builder processes elements in batches to minimize synchronization between threads.
        The total batch size is calculated as:
            batch_size = thread_count * tasks_per_batch * task_size

        Effects of thread count and batch size:
          - thread_count = 1 and batch_size = 1: low throughput, medium latency, order of elements is guaranteed
          - thread_count > 1 and batch_size = 1: high throughput, low latency, order of elements is not guaranteed
          - thread_count > 1 and batch_size > 1: highest throughput, highest latency, order of elements is not guaranteed

        Note: The optimization target `StreamingData` always uses a thread count of 1.

        :param tasks_per_batch: Number of tasks for each thread in one batch (default: 32).
        :type tasks_per_batch: int
        :param task_size: Number of elements each thread processes in one task (default: 10).
        :type task_size: int
        :return: None
        """
        return self.builder_cpp.set_batch_size(tasks_per_batch, task_size)

    def get_batch_size(self) -> int:
        """
        :returns: the batch size used for parallel graph construction.
        """
        return self.builder_cpp.get_batch_size()

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
        metric: Metric = Metric.L2, rng: Mt19937 | None = None, optimization_target: OptimizationTarget = OptimizationTarget.LowLID,
        extend_k: Optional[int] = None, extend_eps: float = 0.2,
        improve_k: Optional[int] = None, improve_eps: float = 0.001, max_path_length: int = 10,
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
    :param optimization_target: TODO
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
        graph, rng, optimization_target=optimization_target, extend_k=extend_k, extend_eps=extend_eps, improve_k=improve_k,
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
