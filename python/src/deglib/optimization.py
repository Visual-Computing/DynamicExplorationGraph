import sys
from typing import Callable, Optional
import numpy as np
import deglib_cpp

from deglib.distances import Metric
from deglib.graph import DynamicExplorationGraph
from deglib.transformation import is_high_level_metric_name
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
        vectors: 2D ``float32`` or ``float16`` NumPy array of shape ``(N, dim)``.
        metric: Metric used for FLAS distance comparisons. Accepts:
            - High-level metric strings: ``"l2"``, ``"euclidean"``, ``"cosine"``,
              ``"normalized"``, ``"ip"``, ``"innerproduct"``, ``"inner_product"``, ``"mips"``.
            - Explicit ``Metric`` or metric string: ``Metric.FP32_L2``, ``Metric.FP16_L2``,
              ``Metric.FP32_InnerProduct``, ``Metric.FP16_InnerProduct`` (optionally bound via
              ``.with_instruction(...)``).
        radius_decay: Decay rate for FLAS neighborhood radius per iteration in ``(0, 1)``.
        threads: Number of worker threads (``0`` = all CPU cores).
        callback: Optional callback receiving progress in ``[0.0, 1.0]``; returning ``False`` cancels.
        show_progress: If True, prints a live progress bar to stdout.

    Returns:
        1D ``uint32`` array of sorted dataset indices of length ``N``.

    See Also:
        :func:`deglib.build`: High-level graph build with ``presort=True``.
        :func:`deglib.optimization.prune_non_mrng_edges`: Low-level MRNG edge pruning.
        :func:`deglib.optimization.prune_worst_edges`: Low-level worst-edge pruning.
    """
    if is_high_level_metric_name(metric):
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2D array, got shape {getattr(vectors, 'shape', type(vectors))}")
        if vectors.dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
            raise ValueError(f"presort requires float32 or float16 vectors, got dtype {vectors.dtype}")
        vectors = np.ascontiguousarray(vectors)
        if vectors.shape[0] == 0:
            return np.empty(0, dtype=np.uint32)
        if metric.strip().lower() in ("ip", "innerproduct", "inner_product", "mips"):
            from deglib.transformation import apply_on_database, make_mips_transform

            mips = make_mips_transform(vectors)
            vectors = apply_on_database(mips, vectors, threads=threads)
            resolved_metric = Metric.FP32_L2
        else:
            resolved_metric = Metric.FP16_L2 if vectors.dtype == np.float16 else Metric.FP32_L2
        base_metric = resolved_metric
    else:
        resolved_metric = Metric.resolve(metric)
        base_metric = Metric(int(resolved_metric))
        if base_metric not in (
            Metric.FP32_L2,
            Metric.FP32_InnerProduct,
            Metric.FP16_L2,
            Metric.FP16_InnerProduct,
        ):
            raise ValueError(f"presort only supports FP32/FP16 L2 and InnerProduct metrics, got {resolved_metric.name}")

        expected_dtype = np.dtype(np.float16) if "FP16" in base_metric.name else np.dtype(np.float32)
        vectors = assure_array(vectors, "vectors", expected_dtype)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2D {expected_dtype} array, got shape {vectors.shape}")
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
