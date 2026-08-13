import sys
import typing
import numpy as np
import deglib_cpp
import deglib_cpp.distances as cpp_distances

from .graph import DynamicExplorationGraph
from .distances import Metric


def remove_non_mrng_edges(graph: DynamicExplorationGraph, num_threads: int = 0) -> int:
    """
    Remove all edges which do not satisfy the MRNG condition.

    :param graph: The graph to optimize. Must be mutable.
    :param num_threads: Number of threads to use for parallel processing. If 0, uses hardware concurrency.
    :return: Number of edges removed.
    """
    return deglib_cpp.remove_non_mrng_edges(graph.dynamic_exploration_graph_cpp, num_threads)


def prune_worst_edges(graph: DynamicExplorationGraph, prune_worst: int, num_threads: int = 0):
    """
    Prune the worst (highest-weight) `prune_worst` neighbors of each vertex
    by replacing them with self-loops.

    :param graph: The graph to optimize. Must be mutable.
    :param prune_worst: Number of worst neighbors to replace with self-loops per vertex.
    :param num_threads: Number of threads to use for parallel processing. If 0, uses hardware concurrency.
    """
    deglib_cpp.prune_worst_edges(graph.dynamic_exploration_graph_cpp, prune_worst, num_threads)


def presort(
    vectors: np.ndarray,
    metric: Metric | str = Metric.FP32_L2,
    radius_decay: float = 0.9,
    threads: int = 0,
    callback: typing.Callable[[float], typing.Union[bool, None]] | str | None = None
) -> np.ndarray:
    """
    Perform 1D pre-sorting of dataset feature vectors using FLAS.

    :param vectors: 2D float32 NumPy array of shape (count, dim).
    :param metric: Metric type used for distance computation during sorting.
    :param radius_decay: Decay factor per iteration for neighborhood radius (default 0.9).
    :param threads: Number of worker threads (0 = use hardware concurrency).
    :param callback: Optional callback for reporting sorting progress. If 'progress', prints progress to stdout.
                     If a function, receives progress float in range [0.0, 1.0]. Returning True cancels sorting early.
    :return: 1D uint32 NumPy array containing the sorted permutation of original vector indices [0..count-1].
    """
    if isinstance(metric, str):
        metric_type = getattr(cpp_distances.Metric, metric)
    elif isinstance(metric, Metric):
        metric_type = cpp_distances.Metric(int(metric))
    elif isinstance(metric, cpp_distances.Metric):
        metric_type = metric
    else:
        metric_type = cpp_distances.Metric(int(metric))

    cb_fn = None
    if callback == "progress":
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
    elif callable(callback):
        cb_fn = callback

    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    return deglib_cpp.presort(vectors_f32, metric_type, radius_decay, threads, cb_fn)


def mips_l2_transform(database: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Transforms database vectors from d-dimensional space to (d+1)-dimensional space
    for Maximum Inner Product Search (MIPS) using L2 distance.

    For each vector x_i in R^d:
        x'_i = [x_i, sqrt(M^2 - ||x_i||^2)] in R^(d+1)
    where M^2 = max_i ||x_i||^2.

    :param database: 2D float32 NumPy array of shape (N, d).
    :return: Tuple of (transformed_database array of shape (N, d+1), max_norm float M).
    """
    db_f32 = np.ascontiguousarray(database, dtype=np.float32)
    return deglib_cpp.mips_l2_transform(db_f32)


def mips_l2_transform_query(queries: np.ndarray) -> np.ndarray:
    """
    Pads query vectors from d-dimensional space to (d+1)-dimensional space
    for MIPS queries against an L2-transformed database.

    For each query vector q_i in R^d:
        q'_i = [q_i, 0] in R^(d+1).

    :param queries: 1D or 2D float32 NumPy array.
    :return: Transformed queries array of shape (d+1) or (Q, d+1).
    """
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)
    return deglib_cpp.mips_l2_transform_query(queries_f32)


__all__ = [
    'remove_non_mrng_edges',
    'prune_worst_edges',
    'presort',
    'mips_l2_transform',
    'mips_l2_transform_query',
]
