import sys
import typing
import numpy as np
import deglib_cpp
import sys
import typing
import numpy as np
import deglib_cpp
import deglib_cpp.distances as cpp_distances

from .graph import DynamicExplorationGraph
from .distances import FloatSpace, Metric


def prune_non_rng_edges(graph: DynamicExplorationGraph, num_threads: int = 0) -> int:
    """
    Remove all graph edges that violate the Relative Neighborhood Graph (RNG) rule.

    An edge between vertices `u` and `v` is pruned if there exists another vertex `w`
    such that distance(u, w) < distance(u, v) and distance(v, w) < distance(u, v).
    Pruning non-RNG edges reduces graph redundancy and accelerates query traversal.

    :param graph: The graph to optimize. Must be mutable.
    :param num_threads: Number of worker threads (0 uses all available CPU cores).
    :return: Total number of edges removed.
    """
    return deglib_cpp.prune_non_rng_edges(graph.dynamic_exploration_graph_cpp, num_threads)


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
    space_or_metric: FloatSpace | Metric | str | None = None,
    radius_decay: float = 0.9,
    threads: int = 0,
    callback: typing.Callable[[float], typing.Union[bool, None]] | str | None = None,
    *,
    metric: FloatSpace | Metric | str | None = None,
    space: FloatSpace | None = None,
) -> np.ndarray:
    """
    Perform 1D pre-sorting of dataset feature vectors using Fast Linear Alignment Scheme (FLAS).

    Sorting high-dimensional vectors onto a 1D curve improves data locality and index construction speed.

    :param vectors: 2D float32 NumPy array of shape (count, dim).
    :param space_or_metric: FloatSpace instance or Metric type used for distance computation during sorting.
    :param radius_decay: Decay factor per iteration for neighborhood radius (default 0.9).
    :param threads: Number of worker threads (0 uses all available CPU cores).
    :param callback: Optional callback for reporting sorting progress.
                     If ``'progress'``, prints progress to stdout.
                     If a function, receives progress float in range [0.0, 1.0]. Returning True cancels sorting.
    :param metric: Alias for space_or_metric.
    :param space: Alias for space_or_metric.
    :return: 1D uint32 NumPy array containing the sorted permutation of original vector indices.
    """
    target = (
        space
        if space is not None
        else (metric if metric is not None else (space_or_metric if space_or_metric is not None else Metric.FP32_L2))
    )

    vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = vectors_f32.shape[1] if vectors_f32.ndim == 2 else 0

    if isinstance(target, FloatSpace):
        cpp_space = target.float_space_cpp
    elif isinstance(target, cpp_distances.FloatSpace):
        cpp_space = target
    elif isinstance(target, str):
        metric_val = getattr(cpp_distances.Metric, target)
        cpp_space = cpp_distances.FloatSpace(dim, cpp_distances.Metric(int(metric_val)))
    elif isinstance(target, Metric):
        cpp_space = cpp_distances.FloatSpace(dim, cpp_distances.Metric(int(target)))
    elif isinstance(target, cpp_distances.Metric):
        cpp_space = cpp_distances.FloatSpace(dim, target)
    else:
        cpp_space = cpp_distances.FloatSpace(dim, cpp_distances.Metric(int(target)))

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

    return deglib_cpp.presort(vectors_f32, cpp_space, radius_decay, threads, cb_fn)


def mips_l2_transform(database: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Transform database vectors from d-dimensional space to (d+1)-dimensional space
    for Maximum Inner Product Search (MIPS) using L2 distance.

    For each vector :math:`x_i \\in \\mathbb{R}^d`:

    .. math::

        x'_i = [x_i, \\sqrt{M^2 - \\|x_i\\|^2}] \\in \\mathbb{R}^{d+1}

    where :math:`M^2 = \\max_i \\|x_i\\|^2`.

    :param database: 2D float32 NumPy array of shape (N, d).
    :return: Tuple of (transformed_database array of shape (N, d+1), max_norm float M).
    """
    db_f32 = np.ascontiguousarray(database, dtype=np.float32)
    return deglib_cpp.mips_l2_transform(db_f32)


def quantize_batch(vectors: np.ndarray, non_zeros: int, num_threads: int = 0) -> np.ndarray:
    """
    Quantize float32 or float16 vectors into byte-packed extreme value representation (EVP).

    :param vectors: 2D float32 or float16 NumPy array of vectors.
    :param non_zeros: Target number of non-zero entries to retain per quantized vector.
    :param num_threads: Number of worker threads (0 uses all available CPU cores).
    :return: 2D uint8 NumPy array of quantized vectors.
    """
    if vectors.dtype == np.float16:
        vectors = vectors.view(np.uint16)
    return deglib_cpp.optimization.quantize_batch(vectors, non_zeros, num_threads)


def mips_l2_transform_query(queries: np.ndarray) -> np.ndarray:
    """
    Pad query vectors from d-dimensional space to (d+1)-dimensional space
    for MIPS queries against an L2-transformed database.

    For each query vector :math:`q_i \\in \\mathbb{R}^d`:

    .. math::

        q'_i = [q_i, 0] \\in \\mathbb{R}^{d+1}

    :param queries: 1D or 2D float32 NumPy array.
    :return: Transformed queries array of shape (d+1) or (Q, d+1).
    """
    queries_f32 = np.ascontiguousarray(queries, dtype=np.float32)
    return deglib_cpp.mips_l2_transform_query(queries_f32)


__all__ = [
    "prune_non_rng_edges",
    "prune_worst_edges",
    "presort",
    "mips_l2_transform",
    "mips_l2_transform_query",
    "quantize_batch",
]
