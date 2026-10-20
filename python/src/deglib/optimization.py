import sys
import typing
import numpy as np
import deglib_cpp
import deglib_cpp.distances as cpp_distances

from .graph import DynamicExplorationGraph
from .distances import FloatSpace, Metric


def prune_non_mrng_edges(graph: DynamicExplorationGraph, num_threads: int = 0) -> int:
    """
    Remove all graph edges that violate the Monotonic Relative Neighborhood Graph (MRNG) rule.

    An edge between vertices `u` and `v` is pruned if there exists another vertex `w`
    such that distance(u, w) < distance(u, v) and distance(v, w) < distance(u, v).
    Pruning non-MRNG edges reduces graph redundancy and accelerates query traversal.

    :param graph: The graph to optimize. Must be mutable.
    :param num_threads: Number of worker threads (0 uses all available CPU cores).
    :return: Total number of edges removed.
    """
    return deglib_cpp.prune_non_mrng_edges(graph.dynamic_exploration_graph_cpp, num_threads)


def prune_worst_edges(graph: DynamicExplorationGraph, prune_worst: int, num_threads: int = 0):
    """
    Prune the longest (highest-weight) edges of each vertex by replacing them with self-loops.

    :param graph: The graph to optimize. Must be mutable.
    :param prune_worst: Number of worst neighbor edges to replace per vertex.
    :param num_threads: Number of worker threads (0 uses all available CPU cores).
    """
    deglib_cpp.prune_worst_edges(graph.dynamic_exploration_graph_cpp, prune_worst, num_threads)


def presort(
    vectors: np.ndarray,
    space_or_metric: FloatSpace | Metric | None = None,
    radius_decay: float = 0.9,
    threads: int = 0,
    callback: typing.Callable[[float], typing.Union[bool, None]] | None = None,
    *,
    metric: Metric | None = None,
    space: FloatSpace | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Perform 1D pre-sorting of dataset feature vectors using Fast Linear Alignment Scheme (FLAS).

    Sorting high-dimensional vectors onto a 1D curve improves data locality and index construction speed.
    FLAS optimization is designed for FP32 feature spaces (FP32_L2, FP32_InnerProduct).

    :param vectors: 2D float32 NumPy array of shape (count, dim).
    :param space_or_metric: FloatSpace instance or Metric type used for distance computation during sorting.
    :param radius_decay: Decay factor per iteration for neighborhood radius (default 0.9).
    :param threads: Number of worker threads (0 uses all available CPU cores).
    :param callback: Optional callback for reporting sorting progress. Receives a progress float in range
                     [0.0, 1.0]. Returning True cancels sorting.
    :param metric: Explicit Metric enum (cannot be combined with space or space_or_metric).
    :param space: Explicit FloatSpace instance (cannot be combined with metric or space_or_metric).
    :param show_progress: If True and no callback is given, prints progress to stdout.
    :return: 1D uint32 NumPy array containing the sorted permutation of original vector indices.
    """
    # Count how many of the three ways to specify distance/space were provided
    specified = sum(x is not None for x in (space_or_metric, metric, space))
    if specified > 1:
        raise ValueError("Cannot specify more than one of 'space_or_metric', 'metric', or 'space'")

    target = (
        Metric.FP32_L2
        if specified == 0
        else (space_or_metric if space_or_metric is not None else (metric if metric is not None else space))
    )

    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = vectors_f32.shape[1] if vectors_f32.ndim == 2 else 0

    if isinstance(target, FloatSpace):
        cpp_space = target.float_space_cpp
    elif isinstance(target, cpp_distances.FloatSpace):
        cpp_space = target
    elif isinstance(target, Metric):
        cpp_space = cpp_distances.FloatSpace(dim, cpp_distances.Metric(int(target)))
    elif isinstance(target, cpp_distances.Metric):
        cpp_space = cpp_distances.FloatSpace(dim, target)
    else:
        raise TypeError(f"Expected Metric or FloatSpace, got {type(target).__name__}")

    cb_fn = None
    if callback is not None:
        cb_fn = callback
    elif show_progress:
        last_pct = [-1]

        def progress_cb(prog: float) -> bool:
            pct = int(prog * 100.0)
            if pct != last_pct[0]:
                last_pct[0] = pct
                sys.stdout.write(f"\rFLAS Presort... {pct}%")
                sys.stdout.flush()
                if pct >= 100:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
            return False

        cb_fn = progress_cb

    return deglib_cpp.presort(vectors_f32, cpp_space, radius_decay, threads, cb_fn)


__all__ = [
    "prune_non_mrng_edges",
    "prune_worst_edges",
    "presort",
]
