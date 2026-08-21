"""
main.py — Maximum Inner Product Search (MIPS) Example (SISAP Challenge 2026 Task 2 mode5_flas).

Demonstrates fast construction and exploration of a MIPS graph on high-dimensional vectors
(Llama embeddings) using Dynamic Exploration Graph (DEG), (d+1)-dimensional L2 transformation,
1D FLAS pre-sorting, and FP16 inner product search.

Usage
-----
    uv run main.py                                      # Runs benchmark with default dataset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import deglib
import deglib_cpp
from deglib.distances import FloatSpace, Metric
from deglib.cpu import InstructionSet
from deglib.optimization import mips_l2_transform, presort

from dataset_utils import (
    ensure_mips_dataset,
    load_hdf5_dataset,
    compute_recall,
)

DEFAULT_K_TOP = 30
DEFAULT_K_GRAPH = 32
DEFAULT_K_EXT = 64
DEFAULT_EPS_EXT = 0.001
DEFAULT_EPS_SEARCH = 0.18
DEFAULT_MAX_DIST_LIST = [6000, 6500, 7000, 7500, 8000, 9000]
DEFAULT_USE_FLAS = True
DEFAULT_FLAS_DECAY = 0.9
DEFAULT_BUILD_THREADS = 1
DEFAULT_SEARCH_THREADS = 8
DEFAULT_OPT_TARGET = deglib.builder.OptimizationTarget.LowLID
DEFAULT_NUM_RUNS = 100
DEFAULT_INSTRUCTION_SET = InstructionSet.Auto


def compute_1d_distortion(features: np.ndarray, sorted_indices: np.ndarray | None = None) -> float:
    """
    Compute the mean 1D L2 distance between adjacent vectors in the given order.

    If sorted_indices is provided, vectors are accessed in that permutation order;
    otherwise they are accessed in natural sequential order.
    """
    if features.shape[0] < 2:
        return 0.0
    if sorted_indices is not None:
        ordered = features[sorted_indices]
    else:
        ordered = features
    diffs = np.diff(ordered, axis=0)
    dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    return float(dists.mean())


def plot_mips_results(
    recalls: list[float],
    search_times_ms: list[float],
    max_dists: list[int],
    k_top: int,
    eps_search: float,
    output_plot: str | None = None,
    no_show: bool = False,
):
    """
    Renders a line plot with Total Search Time (ms) on the X-axis and Recall@k_top on the Y-axis.
    """
    if not recalls or all(r < 0 for r in recalls):
        print("No valid recall data available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        search_times_ms,
        recalls,
        marker="o",
        color="#C44E52",
        linewidth=2,
        markersize=7,
        label=f"DEG MIPS (eps={eps_search})",
    )

    for t, r, md in zip(search_times_ms, recalls, max_dists):
        ax.annotate(
            f"max_dist={md}\n({r * 100.0:.1f}%)",
            xy=(t, r),
            xytext=(7, -10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#2B2B2B",
        )

    ax.set_xlabel("Total Search Time (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Recall@{k_top}", fontsize=11, fontweight="bold")
    ax.set_title(f"DEG MIPS Trade-off: Search Time vs. Recall@{k_top}", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if output_plot:
        plt.savefig(output_plot, dpi=300)
        print(f"Plot saved to {output_plot}")

    if not no_show:
        plt.show()


def _make_build_callback(graph, total_count: int, t_build_start: float):
    """
    Create a build callback that prints progress every 10,000 vertices.
    """
    log_interval = 10000

    def callback(status: deglib_cpp.BuilderStatus):
        current_size = graph.size()
        if current_size % log_interval == 0 or current_size == total_count:
            now = time.perf_counter()
            elapsed_s = now - t_build_start
            vps = current_size / elapsed_s if elapsed_s > 0.001 else 0.0
            pct = (current_size / total_count) * 100.0 if total_count > 0 else 0.0
            print(
                f"  [Build] {current_size:7d} / {total_count:7d} vertices ({pct:.1f}%) | "
                f"{elapsed_s:.1f}s elapsed | {vps:6.0f} vert/s | improved: {status.improved}"
            )
            sys.stdout.flush()

    return callback


def construct_and_search_mips(
    train_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray | None = None,
    k_top: int = DEFAULT_K_TOP,
    k_graph: int = DEFAULT_K_GRAPH,
    k_ext: int = DEFAULT_K_EXT,
    eps_ext: float = DEFAULT_EPS_EXT,
    use_flas: bool = DEFAULT_USE_FLAS,
    flas_radius_decay: float = DEFAULT_FLAS_DECAY,
    build_threads: int = DEFAULT_BUILD_THREADS,
    search_threads: int = DEFAULT_SEARCH_THREADS,
    eps_search: float = DEFAULT_EPS_SEARCH,
    max_dist_list: list[int] | None = None,
    optimization_target: deglib.builder.OptimizationTarget = DEFAULT_OPT_TARGET,
    num_runs: int = DEFAULT_NUM_RUNS,
    instruction: InstructionSet = DEFAULT_INSTRUCTION_SET,
) -> dict[str, list[float] | float | list[int]]:
    """
    Constructs and evaluates a MIPS graph using DEG Mode 5 (L2 Build d+1, FP16 IP Search + FLAS).
    """
    if max_dist_list is None:
        max_dist_list = DEFAULT_MAX_DIST_LIST

    n_vecs, dims = train_vectors.shape
    n_queries = query_vectors.shape[0]
    print(
        f"Dataset shape: {n_vecs} train vectors, {n_queries} queries, {dims} dimensions (dtype: {train_vectors.dtype})"
    )
    print(
        f"MIPS Config: k_top={k_top}, k_graph={k_graph}, k_ext={k_ext}, eps_ext={eps_ext}, use_flas={use_flas}, "
        f"opt_target={optimization_target.name}, "
        f"build_threads={build_threads}, search_threads={search_threads}, num_runs={num_runs}, "
        f"instruction={instruction.name}"
    )

    # 1. (d+1)-dimensional L2 Transformation Phase
    t0 = time.perf_counter()
    train_transformed, max_norm = mips_l2_transform(train_vectors)
    t_transform = time.perf_counter() - t0
    new_dims = train_transformed.shape[1]
    print(
        f"L2 Transformation completed in {t_transform * 1000:.2f}ms (Max norm M = {max_norm:.6f}, new_dims = {new_dims})"
    )

    # 2. 1D FLAS Pre-Sorting Phase
    t0 = time.perf_counter()
    sorted_indices = None

    if use_flas:
        flas_space = FloatSpace.create(new_dims, Metric.FP32_L2)
        dist_before = compute_1d_distortion(train_transformed)
        print(
            f"Running FLAS 1D Pre-sorting: N={n_vecs}, dim={new_dims}, decay={flas_radius_decay}, metric={flas_space.metric().name} ({flas_space.get_instruction().name}), threads={build_threads}..."
        )

        def flas_progress_cb(prog: float) -> bool:
            pct = int(prog * 100.0)
            if pct % 5 == 0 or prog >= 1.0:
                print(f"\r  FLAS Progress: {pct:5.1f} %", end="", flush=True)
            if prog >= 1.0:
                print()
            return False

        sorted_indices = presort(
            train_transformed,
            space=FloatSpace.create(new_dims, Metric.FP32_L2, instruction),
            radius_decay=flas_radius_decay,
            threads=build_threads,
            callback=flas_progress_cb,
        )
        t_flas = time.perf_counter() - t0

        dist_after = compute_1d_distortion(train_transformed, sorted_indices)
        improv = dist_before / dist_after if dist_after > 1e-7 else 0.0
        print(
            f"FLAS 1D Pre-sorting completed in {t_flas:.3f}s (Distortion: {dist_before:.4f} -> {dist_after:.4f}, improvement: {improv:.2f}x)"
        )
    else:
        t_flas = time.perf_counter() - t0

    # 3. Graph Construction Phase (in FP32_InnerProduct space for d+1 dims)
    t0 = time.perf_counter()
    build_space = FloatSpace.create(new_dims, Metric.FP32_L2, instruction)
    print(
        f"Building graph: k_graph={k_graph}, k_ext={k_ext}, eps_ext={eps_ext:.4f}, opt_target={optimization_target.name}, metric={build_space.metric().name} ({build_space.get_instruction().name}), build_threads={build_threads}..."
    )
    graph = deglib.create_empty(n_vecs, build_space, k_graph)
    builder = deglib.GraphBuilder(
        graph,
        extend_k=k_ext,
        extend_eps=eps_ext,
        improve_k=0,
        improve_eps=0.0,
        max_path_length=5,
        optimization_target=optimization_target,
    )
    builder.set_thread_count(build_threads)
    builder.set_batch_size(100, 100)

    if use_flas and sorted_indices is not None:
        builder.add_entry(sorted_indices, train_transformed[sorted_indices])
    else:
        indices = np.arange(n_vecs, dtype=np.uint32)
        builder.add_entry(indices, train_transformed)

    build_callback = _make_build_callback(graph, n_vecs, time.perf_counter())
    builder.build(callback=build_callback)

    t_build = time.perf_counter() - t0
    print(f"Graph construction completed in {t_build:.3f}s")

    # 5. FP16 Feature Convert & Swap Phase
    t0 = time.perf_counter()
    database_fp16 = deglib.distances.floats_to_fp16(train_vectors)
    fp16_space = FloatSpace.create(dims, Metric.FP16_InnerProduct, instruction)
    fp16_graph = graph.to_readonly(feature_space=fp16_space, custom_features=database_fp16)
    t_swap = time.perf_counter() - t0
    print(f"FP16 feature conversion & ReadOnlyGraph swap completed in {t_swap * 1000:.2f}ms")

    # Convert queries to FP16
    t0 = time.perf_counter()
    queries_fp16 = deglib.distances.floats_to_fp16(query_vectors)
    t_qconvert = time.perf_counter() - t0

    # 6. FP16 Inner Product Search Phase (Sweep over max_dist_list with num_runs averaging)
    print(
        f"\n--- FP16 Inner Product Search Sweep (eps_search={eps_search}, search_threads={search_threads}, num_runs={num_runs}) ---"
    )
    recalls = []
    search_times_ms = []
    qps_list = []
    max_dists = []

    best_recall = -1.0
    best_search_time = 0.0
    best_max_dist = max_dist_list[0]

    for max_dist in max_dist_list:
        run_times = []
        for run in range(num_runs):
            t0 = time.perf_counter()
            indices = fp16_graph.search(
                queries_fp16,
                eps=eps_search,
                k=k_top,
                max_distance_computation_count=max_dist,
                threads=search_threads,
                return_distances=False,
                unsorted=False,
            )
            t_search = time.perf_counter() - t0
            run_times.append(t_search)

        avg_search_s = sum(run_times) / len(run_times)
        total_search_s = avg_search_s + t_qconvert
        search_time_ms = total_search_s * 1000.0

        recall = compute_recall(ground_truth, indices, k_top) if ground_truth is not None else -1.0
        qps = n_queries / max(total_search_s, 1e-9)

        recalls.append(recall)
        search_times_ms.append(search_time_ms)
        qps_list.append(qps)
        max_dists.append(max_dist)

        if recall >= 0:
            print(
                f"  max_dist={max_dist:5d} | Recall@{k_top}: {recall * 100.0:6.2f}% | Search Time: {search_time_ms:6.1f}ms | QPS: {qps:8.1f}"
            )
        else:
            print(f"  max_dist={max_dist:5d} | Search Time: {search_time_ms:6.1f}ms | QPS: {qps:8.1f}")

        if recall > best_recall:
            best_recall = recall
            best_search_time = total_search_s
            best_max_dist = max_dist

    t_total_index = t_transform + t_flas + t_build + t_swap

    print(f"\n--- Timing Summary ---")
    if t_transform > 0:
        print(f"Transform Time:        {t_transform * 1000:8.2f} ms")
    if use_flas and t_flas > 0:
        print(f"FLAS Presort Time:     {t_flas:8.3f} s")
    print(f"Graph Build Time:      {t_build:8.3f} s")
    print(f"FP16 Swap Time:        {t_swap * 1000:8.2f} ms")
    print(f"Total Index Time:      {t_total_index:8.3f} s")
    if best_recall >= 0:
        print(
            f"Best Search Time:      {best_search_time * 1000.0:8.1f} ms (max_dist={best_max_dist}, Recall@{k_top}={best_recall * 100.0:.2f}%)"
        )

    # FINAL SUMMARY table
    total_elapsed_s = t_total_index + (t_qconvert / 1000.0) + (best_search_time / 1000.0)

    print(f"\n========================================================================")
    print(f"  FINAL SUMMARY (L2-structured Build, FP16 IP Search (FLAS) - Mode 5)")
    print(f"========================================================================")
    print(f"Load Time:             {0.0:8.1f} ms")
    print(f"Quantize Time:         {t_transform * 1000:8.1f} ms")
    print(f"Graph Build Time:      {t_build:8.1f} s")
    print(f"Graph Conversion Time: {t_swap * 1000:8.1f} ms")
    print(f"Explore Time:          {best_search_time * 1000:8.1f} ms")
    print(f"Rerank Time:           {0.0:8.1f} ms")
    print(f"FLAS Time:             {t_flas:8.1f} s")
    print(f"Total Elapsed Time:    {total_elapsed_s:8.1f} s")
    if best_recall >= 0:
        print(f"Recall@{k_top}:             {best_recall * 100.0:8.2f} %")
    print(f"------------------------------------------------------------------------")
    print(f"Hyperparameters:")
    print(f"  K_TOP:                 {k_top}")
    print(f"  K_GRAPH:               {k_graph}")
    print(f"  K_EXT:                 {k_ext}")
    print(f"  EPS_EXT:               {eps_ext:.3f}")
    print(f"  OPT_TARGET:            {optimization_target.name}")
    print(f"  max_dist:              {best_max_dist}")
    print(f"  threads:               {search_threads}")
    print(f"------------------------------------------------------------------------")
    print(f"Dataset Info:")
    print(f"  Vectors:               {n_vecs}")
    print(f"  Dimensions:            {dims}")
    print(f"========================================================================")

    return {
        "transform_time": t_transform,
        "flas_time": t_flas,
        "build_time": t_build,
        "swap_time": t_swap,
        "qconvert_time": t_qconvert,
        "total_index_time": t_total_index,
        "total_elapsed_time": total_elapsed_s,
        "recalls": recalls,
        "search_times_ms": search_times_ms,
        "qps_list": qps_list,
        "max_dists": max_dists,
        "best_recall": best_recall,
        "best_search_time": best_search_time,
    }


def parse_opt_target(s: str) -> deglib.builder.OptimizationTarget:
    """Parse optimization target string."""
    s_lower = s.lower()
    if s_lower in ("streamingdata", "streaming", "stream"):
        return deglib.builder.OptimizationTarget.StreamingData
    if s_lower in ("lowlid", "low"):
        return deglib.builder.OptimizationTarget.LowLID
    if s_lower in ("highlid", "high"):
        return deglib.builder.OptimizationTarget.HighLID
    raise ValueError(f"Unknown optimization target: {s}. Use StreamingData, LowLID, or HighLID.")


def parse_instruction(s: str) -> InstructionSet:
    """Parse instruction set string."""
    try:
        return InstructionSet[s]
    except KeyError:
        valid = ", ".join(i.name for i in InstructionSet)
        raise ValueError(f"Unknown instruction set: {s}. Valid options: {valid}")


def parse_max_dist(s: str) -> list[int]:
    """Parse comma-separated list of max distance budgets."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Maximum Inner Product Search (MIPS) Example (SISAP Task 2 mode5_flas)"
    )
    parser.add_argument(
        "--dataset", type=str, default="llama-dev", help="Path to HDF5 dataset file or 'llama-dev' (default: llama-dev)"
    )
    parser.add_argument("--k-graph", type=int, default=DEFAULT_K_GRAPH, help="Graph degree per vertex")
    parser.add_argument("--k-ext", type=int, default=DEFAULT_K_EXT, help="Builder search size parameter")
    parser.add_argument("--eps-ext", type=float, default=DEFAULT_EPS_EXT, help="Builder search expansion factor")
    parser.add_argument("--eps-search", type=float, default=DEFAULT_EPS_SEARCH, help="Epsilon search factor")
    parser.add_argument(
        "--flas",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_FLAS,
        help="Enable or disable FLAS 1D pre-sorting",
    )
    parser.add_argument(
        "--flas-decay", type=float, default=DEFAULT_FLAS_DECAY, help="FLAS neighborhood radius decay rate"
    )
    parser.add_argument(
        "--build-threads", type=int, default=DEFAULT_BUILD_THREADS, help="Number of threads for graph building"
    )
    parser.add_argument(
        "--search-threads", type=int, default=DEFAULT_SEARCH_THREADS, help="Number of parallel threads for query search"
    )
    opt_target_choices = [t.name for t in deglib.builder.OptimizationTarget]
    parser.add_argument(
        "--opt-target",
        type=parse_opt_target,
        default=DEFAULT_OPT_TARGET,
        choices=opt_target_choices,
        help="Optimization target: StreamingData, LowLID, HighLID (default: LowLID)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help="Number of search repetitions for averaging (default: 100)",
    )
    parser.add_argument(
        "--max-dist",
        type=parse_max_dist,
        default=DEFAULT_MAX_DIST_LIST,
        help="Comma-separated list of max distance budgets (default: 6000,6500,7000,7500,8000,9000)",
    )
    instruction_choices = [i.name for i in InstructionSet]
    parser.add_argument(
        "--instruction",
        type=parse_instruction,
        default=DEFAULT_INSTRUCTION_SET,
        choices=instruction_choices,
        help="SIMD instruction set: Auto, Scalar, AVX2, AVX512 (default: Auto)",
    )
    parser.add_argument("--output-plot", type=str, default=None, help="Path to save timing chart PNG")
    parser.add_argument("--no-show", action="store_true", help="Disable GUI plot display")

    args = parser.parse_args()

    if args.dataset in (None, "llama-dev", "llama"):
        dataset_path = ensure_mips_dataset()
    else:
        dataset_path = Path(args.dataset)

    print(f"Loading dataset from {dataset_path}...")
    train_data, query_data, gt_data = load_hdf5_dataset(dataset_path, k_top=DEFAULT_K_TOP)

    stats = construct_and_search_mips(
        train_data,
        query_data,
        gt_data,
        k_top=DEFAULT_K_TOP,
        k_graph=args.k_graph,
        k_ext=args.k_ext,
        eps_ext=args.eps_ext,
        use_flas=args.flas,
        flas_radius_decay=args.flas_decay,
        build_threads=args.build_threads,
        search_threads=args.search_threads,
        eps_search=args.eps_search,
        max_dist_list=args.max_dist,
        optimization_target=args.opt_target,
        num_runs=args.num_runs,
        instruction=args.instruction,
    )

    if args.output_plot or not args.no_show:
        plot_mips_results(
            recalls=stats["recalls"],
            search_times_ms=stats["search_times_ms"],
            max_dists=stats["max_dists"],
            k_top=DEFAULT_K_TOP,
            eps_search=args.eps_search,
            output_plot=args.output_plot,
            no_show=args.no_show,
        )


if __name__ == "__main__":
    main()
