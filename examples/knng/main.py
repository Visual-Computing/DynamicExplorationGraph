"""
main.py — k-Nearest Neighbor Graph (k-NNG) Construction Example (SISAP Challenge 2026 Task 1).

Demonstrates fast construction and exploration of a k-NN graph on high-dimensional vectors
using Dynamic Exploration Graph (DEG), EVP feature quantization, and FP16 candidate reranking.

Uses deglib_cpp C++ bindings for:
  - quantize_batch: Fast EVP quantization of float32 vectors to byte-packed EVP format
  - floats_to_fp16 / fp16_to_floats: IEEE 754 half-precision conversion for reranking
  - EVP_InnerProduct metric: Bit-level inner product distance for quantized graph search

Usage
-----
    uv run main.py                                      # Uses small dataset (10000 vectors by default)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import deglib
from deglib.distances import FloatSpace, Metric
from deglib.optimization import quantize_batch
from deglib.search import rerank
from dataset_utils import load_hdf5_dataset, ensure_small_dataset, DEFAULT_CACHE_DIR

DEFAULT_K_TOP = 15
DEFAULT_K_GRAPH = 16
DEFAULT_K_EXT = 32
DEFAULT_NON_ZEROS = 700
DEFAULT_MAX_DIST = 400
DEFAULT_EVP_K = 50
DEFAULT_PRUNE_WORST = 6
DEFAULT_THREADS = 8


def prune_worst_edges(graph: deglib.DynamicExplorationGraph, prune_worst: int) -> None:
    """
    Prunes the worst (highest-weight) `prune_worst` neighbors of each vertex
    by replacing them with self-loops, using the C++ implementation in deglib::optimization::pruning.
    """
    if prune_worst <= 0:
        return
    deglib.optimization.prune_worst_edges(graph, prune_worst)


def quantize_vectors(
    vectors: np.ndarray, non_zeros: int = DEFAULT_NON_ZEROS, num_threads: int = DEFAULT_THREADS
) -> np.ndarray:
    """
    Quantizes floating point vectors to sparse EVP-bit representation using deglib.
    Applies top `non_zeros` active component feature projection via C++ implementation.
    """
    dims = vectors.shape[1]
    effective_non_zeros = min(non_zeros, dims - 1)
    return quantize_batch(vectors, effective_non_zeros, num_threads)


def construct_knng(
    train_vectors: np.ndarray,
    ground_truth: np.ndarray | None = None,
    k_top: int = DEFAULT_K_TOP,
    k_graph: int = DEFAULT_K_GRAPH,
    k_ext: int = DEFAULT_K_EXT,
    non_zeros: int = DEFAULT_NON_ZEROS,
    max_dist: int = DEFAULT_MAX_DIST,
    evp_k: int = DEFAULT_EVP_K,
    prune_worst: int = DEFAULT_PRUNE_WORST,
    threads: int = DEFAULT_THREADS,
) -> dict[str, float]:
    """
    Constructs and evaluates a k-Nearest Neighbor Graph (k-NNG) using DEG.
    """
    n_vecs, dims = train_vectors.shape
    print(f"Dataset shape: {n_vecs} vectors, {dims} dimensions (dtype: {train_vectors.dtype})")
    print(
        f"k-NNG Construction Config: k_top={k_top}, k_graph={k_graph}, non_zeros={non_zeros}, max_dist={max_dist}, evp_k={evp_k}, prune_worst={prune_worst}, threads={threads}"
    )

    # 1. Quantization Phase
    t0 = time.perf_counter()
    quant_vectors = quantize_vectors(train_vectors, non_zeros=non_zeros, num_threads=threads)
    t_quant = time.perf_counter() - t0
    print(f"Quantization completed in {t_quant:.3f}s")

    # 2. Graph Construction Phase (using EVP_InnerProduct metric for fast quantized search)
    t0 = time.perf_counter()
    space = FloatSpace.create(dims, Metric.EVP_InnerProduct)
    print(f"Graph Construction: {space.metric().name} ({space.get_instruction().name})")
    graph = deglib.create_empty(n_vecs, space, k_graph)
    builder = deglib.GraphBuilder(graph, extend_k=k_ext, extend_eps=0.001)
    builder.set_thread_count(threads)

    labels = np.arange(n_vecs, dtype=np.uint32)
    builder.add_entry(labels, quant_vectors)
    builder.build()
    t_build = time.perf_counter() - t0
    print(f"Graph construction completed in {t_build:.3f}s")

    t0 = time.perf_counter()
    if prune_worst > 0:
        prune_worst_edges(graph, prune_worst)
    t_prune = time.perf_counter() - t0
    print(f"Edge pruning completed in {t_prune:.3f}s")

    # 3. Graph Exploration Phase (Self-Join for k-NNG neighbor retrieval)
    t0 = time.perf_counter()
    entry_labels = np.ascontiguousarray(labels, dtype=np.uint32)
    indices = graph.explore(
        entry_labels,
        k=evp_k,
        max_distance_computation_count=max_dist,
        include_entry=False,
        threads=threads,
        return_distances=False,
        unsorted=True,
    )
    t_explore = time.perf_counter() - t0
    print(f"Graph candidate search completed in {t_explore:.3f}s")

    # 4. FP16 Candidate Rerank Phase
    t0 = time.perf_counter()
    rerank_space = FloatSpace.create(dims, Metric.FP16_InnerProduct)
    print(f"Rerank Space: {rerank_space.metric().name} ({rerank_space.get_instruction().name})")
    final_knng_edges = rerank(
        space=rerank_space,
        queries=train_vectors,
        candidate_indices=indices,
        base_vectors=train_vectors,
        k_top=k_top,
        num_threads=threads,
        return_distances=False,
        unsorted=True,
    )
    t_rerank = time.perf_counter() - t0
    print(f"FP16 Reranking completed in {t_rerank:.3f}s")

    t_total = t_quant + t_build + t_prune + t_explore + t_rerank

    # 5. Recall@15 Evaluation against Ground Truth
    recall = 0.0
    if ground_truth is not None:
        correct = 0
        total_gt = 0
        for i in range(n_vecs):
            # Ground truth list clean-up: remove self-references (i) if present
            gt_row = [idx for idx in ground_truth[i] if idx != i][:k_top]
            gt_set = set(gt_row)
            pred_set = set(final_knng_edges[i])
            correct += len(gt_set.intersection(pred_set))
            total_gt += len(gt_set)
        recall = correct / max(total_gt, 1)
        print(f"\n---> k-NNG Quality (Recall@{k_top}): {recall * 100.0:.2f}%")

    print(f"\n--- Timing Summary ---")
    print(f"Quantization Time: {t_quant:.3f}s")
    print(f"Graph Build Time:  {t_build:.3f}s")
    print(f"Prune Time:        {t_prune:.3f}s")
    print(f"Search Time:       {t_explore:.3f}s")
    print(f"Rerank Time:       {t_rerank:.3f}s")
    print(f"Total Time:        {t_total:.3f}s")

    return {
        "quant_time": t_quant,
        "build_time": t_build,
        "prune_time": t_prune,
        "explore_time": t_explore,
        "rerank_time": t_rerank,
        "total_time": t_total,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="k-Nearest Neighbor Graph (k-NNG) Construction Example (SISAP Task 1)")
    parser.add_argument(
        "--dataset", type=str, default="small", help="Path to HDF5 dataset file or 'small' (default: small dataset)"
    )
    parser.add_argument("--non-zeros", type=int, default=DEFAULT_NON_ZEROS, help="EVP quantization non-zero components")
    parser.add_argument("--k-graph", type=int, default=DEFAULT_K_GRAPH, help="Graph degree per vertex")
    parser.add_argument("--k-ext", type=int, default=DEFAULT_K_EXT, help="Builder search size parameter")
    parser.add_argument("--max-dist", type=int, default=DEFAULT_MAX_DIST, help="Search step budget per query")
    parser.add_argument("--evpK", type=int, default=DEFAULT_EVP_K, help="Candidate list size before FP16 reranking")
    parser.add_argument(
        "--prune-worst",
        type=int,
        default=DEFAULT_PRUNE_WORST,
        help="Number of worst neighbors to replace with self-loops",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Number of parallel threads")
    parser.add_argument("--output-plot", type=str, default=None, help="Path to save timing chart PNG")
    parser.add_argument("--no-show", action="store_true", help="Disable GUI plot display")

    args = parser.parse_args()

    if args.dataset in (None, "small", "small_dataset"):
        dataset_path = ensure_small_dataset()
    else:
        dataset_path = Path(args.dataset)

    print(f"Loading dataset from {dataset_path}...")
    train_data, gt_data = load_hdf5_dataset(dataset_path)

    stats = construct_knng(
        train_data,
        gt_data,
        k_top=15,
        k_graph=args.k_graph,
        k_ext=args.k_ext,
        non_zeros=args.non_zeros,
        max_dist=args.max_dist,
        evp_k=args.evpK,
        prune_worst=args.prune_worst,
        threads=args.threads,
    )

    if args.output_plot or not args.no_show:
        stages = ["Quantization", "Graph Build", "Exploration", "Rerank"]
        times = [stats["quant_time"], stats["build_time"], stats["explore_time"], stats["rerank_time"]]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(stages, times, color=["#4C72B0", "#55A868", "#C44E52", "#8172B1"])
        ax.set_ylabel("Time (seconds)")
        ax.set_title("k-NNG Construction Execution Time Breakdown (SISAP Task 1 Mode 4)")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}s",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        if args.output_plot:
            plt.savefig(args.output_plot)
            print(f"Plot saved to {args.output_plot}")

        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()
