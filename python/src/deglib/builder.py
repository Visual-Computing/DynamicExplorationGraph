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
    """
    Information about the data distribution to help switch between different graph extension strategies.

    - StreamingData: Optimized for streaming or shifting distributions
    - HighLID: Optimized for datasets with high local intrinsic dimensionality (above 15)
    - LowLID: Optimized for datasets with low local intrinsic dimensionality (below 15)
    """
    StreamingData = deglib_cpp.OptimizationTarget.StreamingData
    HighLID = deglib_cpp.OptimizationTarget.HighLID
    LowLID = deglib_cpp.OptimizationTarget.LowLID

    def to_cpp(self) -> deglib_cpp.OptimizationTarget:
        """
        Convert the Python OptimizationTarget enum to its C++ equivalent.

        :return: The corresponding C++ OptimizationTarget enum value
        :rtype: deglib_cpp.OptimizationTarget
        """
        if self == OptimizationTarget.StreamingData:
            return deglib_cpp.OptimizationTarget.StreamingData
        elif self == OptimizationTarget.HighLID:
            return deglib_cpp.OptimizationTarget.HighLID
        elif self == OptimizationTarget.LowLID:
            return deglib_cpp.OptimizationTarget.LowLID


class EvenRegularGraphBuilder:
    """
    Constructs an EvenRegularGraphBuilder for building and optimizing a regular graph.

    This class provides functionality to incrementally build and optimize a graph structure
    by adding/removing vertices and improving edge connections through various strategies.
    """
    def __init__(
            self, graph: MutableGraph, rng: Mt19937 | None = None,
            optimization_target: OptimizationTarget = OptimizationTarget.LowLID,
            extend_k: int = 0, extend_eps: float = 0.1,
            improve_k: int = 0, improve_eps: float = 0.001, max_path_length: int = 5,
            swap_tries: int = 0, additional_swap_tries: int = 0
    ):
        """
        Initialize an EvenRegularGraphBuilder with the specified parameters.

        :param graph: The preallocated mutable graph to build and optimize
        :type graph: MutableGraph
        :param rng: Random number generator used for randomized operations. If None, a new Mt19937 will be created
        :type rng: Mt19937 | None
        :param optimization_target: Optimization strategy based on data distribution characteristics
        :type optimization_target: OptimizationTarget
        :param extend_k: Number of neighbors to consider when extending the graph. Defaults to graph's edges_per_vertex if smaller
        :type extend_k: int
        :param extend_eps: Epsilon value for neighbor search during graph extension
        :type extend_eps: float
        :param improve_k: Number of neighbors to consider when improving the graph
        :type improve_k: int
        :param improve_eps: Epsilon value for neighbor search during graph improvement
        :type improve_eps: float
        :param max_path_length: Maximum number of edge swaps in a single improvement attempt
        :type max_path_length: int
        :param swap_tries: Number of improvement attempts per build step
        :type swap_tries: int
        :param additional_swap_tries: Additional improvement attempts after a successful improvement
        :type additional_swap_tries: int
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
        Provide the builder with new entries to be added to the graph during the build process.

        Can add a single entry or a batch of entries. For batch processing, the feature array
        should have shape [N, D] where N is the number of entries and D is the dimensionality.

        :param external_label: The label(s) that name the added vertex/vertices. For batch processing,
                              should be a list or array with length N matching the number of features
        :type external_label: int | Iterable[int] | np.ndarray
        :param feature: The feature vector(s) to be added. Shape [D] for single entry or [N, D] for batch
        :type feature: np.ndarray
        :raises InvalidShapeException: If feature array has invalid shape (not 1D or 2D)
        :raises AssertionError: If number of features doesn't match number of labels in batch processing
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
        Command the builder to remove a vertex from the graph as quickly as possible.

        :param external_label: The external label that identifies the graph vertex to remove
        :type external_label: int
        """
        self.builder_cpp.remove_entry(external_label)

    def get_num_new_entries(self) -> int:
        """
        Get the number of entries that will be added to the graph.

        :return: Number of entries queued for addition
        :rtype: int
        """
        return self.builder_cpp.get_num_new_entries()

    def get_num_remove_entries(self) -> int:
        """
        Get the number of entries that will be removed from the graph.

        :return: Number of entries queued for removal
        :rtype: int
        """
        return self.builder_cpp.get_num_remove_entries()

    def set_thread_count(self, thread_count: int):
        """
        Set the number of threads used to extend the graph during building.

        When the thread count is greater than 1 and the optimization target is not StreamingData,
        the builder will utilize multiple threads to add elements to the graph in parallel.
        By default, all available CPU cores/threads are used unless specified.
        Note: The order in which elements are added is not guaranteed when using multiple threads.

        :param thread_count: Number of threads to use for graph extension
        :type thread_count: int
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
        """
        return self.builder_cpp.set_batch_size(tasks_per_batch, task_size)

    def get_batch_size(self) -> int:
        """
        Get the current batch size used for parallel graph construction.

        :return: The current batch size
        :rtype: int
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
        """
        Stop the build process.

        The build process can only be stopped between steps, not during step execution.
        """
        self.builder_cpp.stop()

    def __repr__(self):
        """
        Return a string representation of the EvenRegularGraphBuilder.

        :return: String representation showing the number of vertices added
        :rtype: str
        """
        return 'EvenRegularGraphBuilder(vertices_added={})'.format(self.graph.size())


def build_from_data(
        data: np.ndarray, labels: Iterable[int] | None = None, edges_per_vertex: int = 32, capacity: int = -1,
        metric: Metric = Metric.L2, rng: Mt19937 | None = None, optimization_target: OptimizationTarget = OptimizationTarget.LowLID,
        extend_k: int = 0, extend_eps: float = 0.2,
        improve_k: int = 0, improve_eps: float = 0.001, max_path_length: int = 5,
        swap_tries: int = 0, additional_swap_tries: int = 0,
        callback: Callable[[deglib_cpp.BuilderStatus], None] | str | None = None
) -> SizeBoundedGraph:
    """
    Create a new graph built from the given data using an EvenRegularGraphBuilder.

    This is a convenience function that creates a graph, builder, adds all data entries,
    and builds the complete graph in one call. Infinite building is not supported.

    :param data: Feature data with shape [N, D] where N is number of samples, D is dimensionality
    :type data: np.ndarray
    :param labels: Labels for each data entry with shape [N]. If None, labels 0 to N-1 are created.
                  If numpy array, should have dtype uint32
    :type labels: Iterable[int] | None
    :param edges_per_vertex: Number of edges per vertex in the graph
    :type edges_per_vertex: int
    :param capacity: Maximum number of vertices the graph can hold. If <= 0, defaults to data.shape[0]
    :type capacity: int
    :param metric: Distance metric for measuring feature similarity
    :type metric: Metric
    :param rng: Random number generator. If None, a new Mt19937 will be created
    :type rng: Mt19937 | None
    :param optimization_target: Optimization strategy based on data distribution characteristics
    :type optimization_target: OptimizationTarget
    :param extend_k: Number of neighbors to consider when extending the graph
    :type extend_k: int
    :param extend_eps: Epsilon value for neighbor search during graph extension
    :type extend_eps: float
    :param improve_k: Number of neighbors to consider when improving the graph
    :type improve_k: int
    :param improve_eps: Epsilon value for neighbor search during graph improvement
    :type improve_eps: float
    :param max_path_length: Maximum number of edge swaps in a single improvement attempt
    :type max_path_length: int
    :param swap_tries: Number of improvement attempts per build step
    :type swap_tries: int
    :param additional_swap_tries: Additional improvement attempts after a successful improvement
    :type additional_swap_tries: int
    :param callback: Callback function for build progress reporting. If "progress", shows progress bar
    :type callback: Callable[[deglib_cpp.BuilderStatus], None] | str | None
    :return: The constructed and optimized graph
    :rtype: SizeBoundedGraph
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
    """
    A callback class for displaying build progress as a progress bar.

    Provides a visual progress bar showing the completion status of graph building operations,
    including both addition and removal of entries.
    """
    def __init__(
            self, num_new_entries: int, num_remove_entries: int, bar_length: int = 60, min_print_interval: float = 0.1
    ):
        """
        Initialize the progress callback with build parameters.

        :param num_new_entries: Total number of entries to be added to the graph
        :type num_new_entries: int
        :param num_remove_entries: Total number of entries to be removed from the graph
        :type num_remove_entries: int
        :param bar_length: Length of the progress bar in characters
        :type bar_length: int
        :param min_print_interval: Minimum time interval between progress updates in seconds
        :type min_print_interval: float
        """
        self.num_new_entries = num_new_entries
        self.num_remove_entries = num_remove_entries
        self.bar_length = bar_length
        self.maximal = self.num_new_entries + self.num_remove_entries
        self.len_max = len(str(self.maximal))
        self.last_print_time = 0
        self.min_print_interval = min_print_interval

    def __call__(self, builder_status: deglib_cpp.BuilderStatus):
        """
        Display the current build progress as a formatted progress bar.

        Called by the builder during the build process to report status. Updates are throttled
        by min_print_interval to avoid excessive output, except for the final step.

        :param builder_status: Current status of the build process containing step counts
        :type builder_status: deglib_cpp.BuilderStatus
        """
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
