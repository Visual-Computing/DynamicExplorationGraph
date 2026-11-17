import sys
from typing import Callable, Optional
import numpy as np
import deglib_cpp

from deglib.distances import Metric
from deglib.graph import DynamicExplorationGraph
from deglib.utils import assure_array

cpp_opt = deglib_cpp.optimization


def prune_non_mrng_edges(graph: DynamicExplorationGraph, threads: int = 0) -> int:
    """
    Prune edges that violate the Monotonic Relative Neighborhood Graph (MRNG) condition.

    Args:
        graph: Mutable graph instance to prune in-place.
        threads: Number of worker threads (``0`` = all CPU cores).

    Returns:
        Number of vertices whose edge lists changed.
    """
    return cpp_opt.prune_non_mrng_edges(getattr(graph, "dynamic_exploration_graph_cpp", graph), threads)


def prune_worst_edges(graph: DynamicExplorationGraph, prune_worst: int, threads: int = 0) -> None:
    """
    Replace the ``prune_worst`` longest edges of each vertex with self-loops.

    Args:
        graph: Mutable graph instance to prune in-place.
        prune_worst: Number of longest outgoing edges per vertex to prune.
        threads: Number of worker threads (``0`` = all CPU cores).
    """
    cpp_opt.prune_worst_edges(getattr(graph, "dynamic_exploration_graph_cpp", graph), prune_worst, threads)


def presort(
    vectors: np.ndarray,
    metric: Metric | str = Metric.FP32_L2,
    radius_decay: float = 0.9,
    threads: int = 0,
    callback: Optional[Callable[[float], Optional[bool]]] = None,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Perform 1D pre-sorting of dataset feature vectors using FLAS (Fast Linear Alignment Scheme).

    Args:
        vectors: 2D ``float32`` NumPy array of shape ``(N, dim)``.
        metric: ``Metric.FP32_L2`` or ``Metric.FP32_InnerProduct`` (string or ``Metric``;
            optionally bound via ``.with_instruction(...)``).
        radius_decay: Decay rate for FLAS neighborhood radius per iteration in ``(0, 1)``.
        threads: Number of worker threads (``0`` = all CPU cores).
        callback: Optional callback receiving progress in ``[0.0, 1.0]``; returning ``False`` cancels.
        show_progress: If True, prints a live progress bar to stdout.

    Returns:
        1D ``uint32`` array of sorted dataset indices of length ``N``.
    """
    resolved_metric = Metric.resolve(metric)
    base_metric = Metric(int(resolved_metric))
    if base_metric not in (Metric.FP32_L2, Metric.FP32_InnerProduct):
        raise ValueError(f"presort only supports FP32_L2 and FP32_InnerProduct, got {resolved_metric.name}")

    vectors = assure_array(vectors, "vectors", np.dtype(np.float32))
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a 2D float32 array, got shape {vectors.shape}")
    if vectors.shape[0] == 0:
        return np.empty(0, dtype=np.uint32)

    cpp_space = resolved_metric._to_cpp_space(vectors.shape[1])

    wrapped_callback = None
    if show_progress or callback is not None:
        bar_len = 30
        last_pct = -1

        def _progress_cb(progress: float) -> bool:
            nonlocal last_pct
            if show_progress:
                pct = int(progress * 100)
                if pct != last_pct:
                    last_pct = pct
                    filled = int(bar_len * progress)
                    bar = "=" * filled + ">" * (1 if filled < bar_len else 0) + " " * max(0, bar_len - filled - 1)
                    sys.stdout.write(f"\rPresorting [{bar}] {pct:3d}%")
                    sys.stdout.flush()
            if callback is not None:
                res = callback(progress)
                return False if res is None else bool(res)
            return False

        wrapped_callback = _progress_cb

    try:
        return cpp_opt.presort(
            vectors,
            cpp_space,
            radius_decay=radius_decay,
            threads=threads,
            callback=wrapped_callback,
        )
    finally:
        if show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()


__all__ = ["presort", "prune_non_mrng_edges", "prune_worst_edges"]
