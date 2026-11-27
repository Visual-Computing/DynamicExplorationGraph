"""
main.py — Maximum Inner Product Search (MIPS) Example (SISAP Challenge 2026 Task 2).

Demonstrates fast construction and exploration of a MIPS graph using the high-level
`deglib` API (`metric="ip"`, `compress="16bit"` or `"8bit"`, FLAS pre-sorting, and `deglib.searcher`).

Usage
-----
    uv run main.py
    uv run main.py --compress 8bit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import deglib
import matplotlib.pyplot as plt
import numpy as np

from dataset import compute_recall, ensure_mips_dataset, load_hdf5_dataset

# --- 1. Shared defaults (apply to both 16bit and 8bit) ---
DEFAULT_K_TOP = 30
DEFAULT_K_GRAPH = 48
DEFAULT_K_EXT = 96
DEFAULT_EPS_EXT = 0.1
DEFAULT_BUILD_THREADS = 1
DEFAULT_SEARCH_THREADS = 8
DEFAULT_OPT_TARGET = deglib.OptimizationTarget.LowLID
DEFAULT_SEED = 7
DEFAULT_NUM_RUNS = 20
DEFAULT_COMPRESS = "16bit"

# --- 2. 16bit search defaults (FP16_InnerProduct + auto MIPS_SCALED, no reranker) ---
DEFAULT_16BIT_EPS_LIST = [0.0056, 0.0057, 0.0058, 0.0059, 0.0060]
DEFAULT_16BIT_RERANKER = None
DEFAULT_16BIT_RERANK_FACTOR = 1.0
DEFAULT_16BIT_MAX_DIST = 0

# --- 3. 8bit search defaults (Int8_InnerProduct + FP16 reranker, ~60k QPS @ ~80% Recall@30) ---
DEFAULT_8BIT_EPS_LIST = [0.11, 0.12, 0.13, 0.14]
DEFAULT_8BIT_RERANKER = "16bit"
DEFAULT_8BIT_RERANK_FACTOR = 1.6
DEFAULT_8BIT_MAX_DIST = 6000


def get_mode_defaults(compress: str) -> tuple[list[float], str | None, float, int]:
    """Return (eps_list, reranker, rerank_factor, max_dist) defaults for the chosen precision."""
    if compress == "8bit":
        return (
            DEFAULT_8BIT_EPS_LIST,
            DEFAULT_8BIT_RERANKER,
            DEFAULT_8BIT_RERANK_FACTOR,
            DEFAULT_8BIT_MAX_DIST,
        )
    return (
        DEFAULT_16BIT_EPS_LIST,
        DEFAULT_16BIT_RERANKER,
        DEFAULT_16BIT_RERANK_FACTOR,
        DEFAULT_16BIT_MAX_DIST,
    )


def plot_mips_results(
    recalls: list[float],
    qps_list: list[float],
    eps_list: list[float],
    k_top: int,
    compress: str = DEFAULT_COMPRESS,
    output_plot: str | None = None,
    no_show: bool = False,
) -> None:
    """Plot Recall@k_top (X-axis) against Queries Per Second (Y-axis)."""
    if not recalls or all(r < 0 for r in recalls):
        print("No valid recall data available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, qps_list, marker="o", color="#C44E52", linewidth=2, markersize=7, label=f"DEG MIPS ({compress})")
    for r, qps, eps in zip(recalls, qps_list, eps_list):
        ax.annotate(
            f"eps={eps:.4f}\n({r * 100.0:.2f}%)",
            xy=(r, qps),
            xytext=(7, -10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlabel(f"Recall@{k_top}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Queries Per Second (QPS)", fontsize=11, fontweight="bold")
    ax.set_title(f"DEG MIPS Trade-off: Recall@{k_top} vs. QPS ({compress})", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if output_plot:
        plt.savefig(output_plot, dpi=300)
        print(f"Plot saved to {output_plot}")
    if not no_show:
        plt.show()


def construct_and_search_mips(
    train_vectors: np.ndarray,
    query_vectors: np.ndarray,
    ground_truth: np.ndarray | None = None,
    k_top: int = DEFAULT_K_TOP,
    k_graph: int = DEFAULT_K_GRAPH,
    k_ext: int = DEFAULT_K_EXT,
    eps_ext: float = DEFAULT_EPS_EXT,
    use_flas: bool = True,
    compress: str = DEFAULT_COMPRESS,
    build_threads: int = DEFAULT_BUILD_THREADS,
    search_threads: int = DEFAULT_SEARCH_THREADS,
    eps_list: list[float] | None = None,
    rerank_factor: float | None = None,
    max_dist: int | None = None,
    optimization_target: deglib.OptimizationTarget = DEFAULT_OPT_TARGET,
    seed: int = DEFAULT_SEED,
    num_runs: int = DEFAULT_NUM_RUNS,
) -> dict[str, list[float]]:
    """Build a MIPS graph in FP32, convert to a compressed searcher, and sweep eps values."""
    def_eps, reranker, def_rf, def_md = get_mode_defaults(compress)
    eps_list = def_eps if eps_list is None else eps_list
    rerank_factor = def_rf if rerank_factor is None else rerank_factor
    max_dist = def_md if max_dist is None else max_dist

    # 1. Build graph in MIPS_UNSCALED space (handled internally by metric="ip"), then swap back to IP
    t0 = time.perf_counter()
    graph = deglib.build(
        train_vectors,
        metric="ip",
        edges_per_vertex=k_graph,
        optimization_target=optimization_target,
        extend_k=k_ext,
        extend_eps=eps_ext,
        seed=seed,
        presort=use_flas,
        optimize=False,
        threads=build_threads,
        show_progress=True,
    )
    t_build = time.perf_counter() - t0

    # 2. Create optimized ReadOnlyGraph searcher (compress="8bit" selects Int8_InnerProduct + FP16 reranker)
    t0 = time.perf_counter()
    searcher = deglib.searcher(graph, compress=compress, reranker=reranker, optimize=True, threads=search_threads)
    t_opt = time.perf_counter() - t0
    print(
        f"Build: {t_build:.2f}s | Optimize: {t_opt * 1000:.1f}ms | "
        f"metric={searcher.graph.get_metric().name}, reranker={reranker}, "
        f"instruction={searcher.graph.get_instruction().name}, dim={searcher.graph.get_dim()}"
    )

    # 3. Warmup run before timing so thread pool and caches are primed
    _ = searcher.search(
        query_vectors,
        k=k_top,
        eps_or_ef=eps_list[0],
        rerank_factor=rerank_factor,
        max_distance_computation_count=max_dist,
        threads=search_threads,
        return_distances=False,
    )

    # 4. Sweep eps values and report median QPS across num_runs repetitions
    recalls: list[float] = []
    qps_list: list[float] = []
    n_queries = query_vectors.shape[0]
    for eps in eps_list:
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            indices = searcher.search(
                query_vectors,
                k=k_top,
                eps_or_ef=eps,
                rerank_factor=rerank_factor,
                max_distance_computation_count=max_dist,
                threads=search_threads,
                return_distances=False,
            ).indices
            times.append(time.perf_counter() - t0)
        times.sort()
        med_s = times[len(times) // 2]
        recall = compute_recall(ground_truth, indices, k_top) if ground_truth is not None else -1.0
        qps = n_queries / max(med_s, 1e-9)
        recalls.append(recall)
        qps_list.append(qps)
        print(
            f"  eps={eps:.4f} (md={max_dist}, rf={rerank_factor:.2f}) | "
            f"Recall@{k_top}: {recall * 100.0:6.2f}% | {med_s * 1000.0:5.1f}ms | QPS: {qps:8.1f}"
        )

    return {"recalls": recalls, "qps_list": qps_list, "eps_list": eps_list}


def parse_eps_list(s: str) -> list[float]:
    """Parse a comma-separated list of eps search thresholds."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Maximum Inner Product Search (MIPS) Example")
    parser.add_argument("--dataset", type=str, default="llama-dev", help="Path to HDF5 dataset or 'llama-dev'")
    parser.add_argument("--k-graph", type=int, default=DEFAULT_K_GRAPH, help="Graph degree per vertex")
    parser.add_argument("--k-ext", type=int, default=DEFAULT_K_EXT, help="Builder search size parameter")
    parser.add_argument("--eps-ext", type=float, default=DEFAULT_EPS_EXT, help="Builder search expansion factor")
    parser.add_argument("--eps-list", type=parse_eps_list, default=None, help="Comma-separated eps values (default per mode)")
    parser.add_argument("--compress", type=str, default=DEFAULT_COMPRESS, choices=["32bit", "16bit", "8bit"], help="Target precision")
    parser.add_argument("--rerank-factor", type=float, default=None, help="Rerank candidate multiplier (default per mode)")
    parser.add_argument("--max-dist", type=int, default=None, help="Max distance computations per query (default per mode)")
    parser.add_argument("--flas", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable FLAS pre-sorting")
    parser.add_argument("--build-threads", type=int, default=DEFAULT_BUILD_THREADS, help="Threads for graph build")
    parser.add_argument("--search-threads", type=int, default=DEFAULT_SEARCH_THREADS, help="Threads for query search")
    parser.add_argument("--opt-target", type=deglib.OptimizationTarget.resolve, default=DEFAULT_OPT_TARGET, help="Optimization target")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for FLAS and graph build")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS, help="Repetitions per eps for median QPS")
    parser.add_argument("--output-plot", type=str, default=None, help="Path to save Recall vs. QPS PNG")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot window")
    args = parser.parse_args()

    dataset_path = ensure_mips_dataset() if args.dataset in (None, "llama-dev", "llama") else Path(args.dataset)
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
        compress=args.compress,
        build_threads=args.build_threads,
        search_threads=args.search_threads,
        eps_list=args.eps_list,
        rerank_factor=args.rerank_factor,
        max_dist=args.max_dist,
        optimization_target=args.opt_target,
        seed=args.seed,
        num_runs=args.num_runs,
    )

    if args.output_plot or not args.no_show:
        plot_mips_results(
            recalls=stats["recalls"],
            qps_list=stats["qps_list"],
            eps_list=stats["eps_list"],
            k_top=DEFAULT_K_TOP,
            compress=args.compress,
            output_plot=args.output_plot,
            no_show=args.no_show,
        )


if __name__ == "__main__":
    main()
