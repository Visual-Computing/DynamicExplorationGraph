"""
====================================================================================================
DEG Sliding Window Batched Update Benchmark (CleANN Paper Baseline Reproduction)
====================================================================================================

Paper Reference:
  "CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search"
  by Ziyu Zhang, Yuanhao Wei, Joshua Engels, Julian Shun (MIT CSAIL, arXiv:2507.19802v1)
  
Specific Section & Figures Reproduced:
  - Section 6.1 (Experiment Setup: Sliding Window Batched Update)
  - Section 6.2.3 (Comparison with Sequential Baseline DEG)
  - Figure 20: Recall 10@10 over updates for RedCaps with index size 500k over 100 rounds
  - Figure 21: Single-thread update and search throughput on RedCaps with index size 500k (DEG with eps=0.03)

Experiment Protocol:
  1. Initial Graph Construction:
     - Load RedCaps-512 embeddings (downloaded automatically from Zenodo: https://zenodo.org/records/13137120).
     - Build an initial DEG graph of size N = 500,000 using `deglib.builder.GraphBuilder` with
       `OptimizationTarget.StreamingData`.
  2. Streaming Sliding-Window Loop (100 Rounds):
     - In each round r in 1..100:
       a. Batch Insert: Insert 1% (5,000 new vectors) via `builder.add_entry()`.
       b. Batch Delete: Delete 1% (5,000 oldest active vectors) via `builder.remove_entry()`.
          * Note: DEG performs neighborhood repair automatically upon vertex removal.
       c. Graph Update Execution: Call `builder.build()` to process pending mutations.
       d. Update Throughput: Measure total elapsed time for (insert + delete + repair) in updates/ms.
       e. Dynamic Ground Truth: Compute exact Top-10 nearest neighbors for the 800 RedCaps test queries
          over the currently active sliding window of 500,000 vectors.
       f. Query Evaluation: Execute single-threaded batch search with eps=0.03 and k=10.
          Measure Search QPS, query latency, and Recall 10@10.
  3. Visualizations & Metrics:
     - Generate dual-panel plot reproducing Figure 20 (Recall over rounds) and Figure 21 (Throughput).
"""

import argparse
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import deglib
from dataset import (
    DATASET_METADATA,
    fbin_read,
    fvecs_read,
    load_benchmark_dataset,
    compute_exact_topk_labels,
)


def run_sliding_window_benchmark(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    metric: deglib.Metric = deglib.Metric.FP32_L2,
    window_size: int = 500_000,
    batch_size: int = 5_000,
    rounds: int = 100,
    edges_per_vertex: int = 32,
    search_k: int = 10,
    search_eps: float = 0.03,
    no_show: bool = False,
    save_plot: str | None = "sliding_window_benchmark.png",
):
    """
    Executes the sliding window benchmark reproducing the DEG baseline evaluation
    from the CleANN paper (Section 6.1, 6.2.3, Fig 20 & 21).

    :param base_vectors: Full dataset of feature embeddings (at least window_size + rounds * batch_size).
    :param query_vectors: Query embeddings (e.g. 800 test queries for RedCaps).
    :param metric: Distance metric (FP32_L2 for normalized cosine embeddings or L2 spaces).
    :param window_size: Number of active vectors maintained in the sliding window (paper uses 500,000).
    :param batch_size: Number of vectors inserted/deleted per round (paper uses 5,000 = 1% of 500k).
    :param rounds: Number of sliding window update rounds (paper uses 100 rounds).
    :param edges_per_vertex: DEG graph degree bound (edges per vertex, default: 32).
    :param search_k: Top-k nearest neighbors to retrieve (paper uses k=10).
    :param search_eps: DEG query exploration epsilon (paper uses eps=0.03).
    :param no_show: If True, skips displaying the interactive Matplotlib window.
    :param save_plot: Path to save the dual-panel summary plot.
    """
    total_vectors = len(base_vectors)
    needed_vectors = window_size + (rounds * batch_size)

    if total_vectors < needed_vectors:
        raise ValueError(
            f"Dataset has {total_vectors:,} vectors, but {needed_vectors:,} are required "
            f"for window_size={window_size:,}, batch_size={batch_size:,}, rounds={rounds}."
        )

    dims = base_vectors.shape[1]
    print("=" * 75)
    print("  DEG Sliding Window Benchmark (CleANN Paper Baseline Experiment)")
    print("=" * 75)
    print(f"  Reference:          CleANN Paper (arXiv:2507.19802v1), Section 6.1 & 6.2.3")
    print(f"  Figures:            Figure 20 (Recall@10) & Figure 21 (Throughput)")
    print(f"  Dimensions:         {dims}")
    print(f"  Metric:             {metric.name}")
    print(f"  Window Size (N):    {window_size:,} vectors")
    print(f"  Batch Size (M):     {batch_size:,} ({batch_size / window_size * 100:.1f}% per round)")
    print(f"  Total Rounds:       {rounds}")
    print(f"  Search k:           {search_k}")
    print(f"  Search eps:         {search_eps}")
    print(f"  Edges per vertex:   {edges_per_vertex}")
    print(f"  Total Queries:      {len(query_vectors):,}")
    print("=" * 75)

    # 1. Initialize Space and Preallocated DEG Graph
    # Preallocate graph with capacity for window_size + batch_size buffer
    max_capacity = window_size + batch_size + 1000
    space = deglib.FloatSpace.create(dim=dims, metric=metric)
    graph = deglib.create_empty(
        capacity=max_capacity,
        feature_space=space,
        edges_per_vertex=edges_per_vertex,
    )

    # Builder configured with StreamingData target (single-threaded deterministic streaming)
    builder = deglib.builder.GraphBuilder(
        graph,
        optimization_target=deglib.builder.OptimizationTarget.StreamingData,
        extend_k=edges_per_vertex * 2,
        extend_eps=0.1,
    )
    builder.set_thread_count(1)

    # 2. Initial Graph Population (first N vectors)
    print(f"\n[Step 1/2] Building initial graph with {window_size:,} vectors...")
    t0 = time.perf_counter()

    chunk_size = 50_000
    for i in range(0, window_size, chunk_size):
        end_i = min(i + chunk_size, window_size)
        labels = np.arange(i, end_i, dtype=np.uint32)
        features = base_vectors[i:end_i]
        builder.add_entry(labels, features)
        builder.build()

    build_time = time.perf_counter() - t0
    print(f"Initial graph built in {build_time:.2f}s (Throughput: {window_size / build_time:,.0f} vec/s)\n")

    # Tracking metrics across rounds
    round_recalls = []
    round_search_qps = []
    round_update_throughput = []  # updates / ms

    active_start_idx = 0
    next_insert_idx = window_size

    print(f"[Step 2/2] Running {rounds} sliding window update rounds...")
    print(f"{'Round':>6} | {'Recall@' + str(search_k):>11} | {'Search QPS':>12} | {'Search ms':>10} | {'Update/ms':>10}")
    print("-" * 62)

    for r in range(rounds):
        # ---------------------------------------------------------------------
        # 1. Update Phase: Insert new batch & Delete oldest batch
        # ---------------------------------------------------------------------
        t_update_start = time.perf_counter()

        # Add M new vectors
        new_labels = np.arange(next_insert_idx, next_insert_idx + batch_size, dtype=np.uint32)
        new_features = base_vectors[next_insert_idx : next_insert_idx + batch_size]
        builder.add_entry(new_labels, new_features)

        # Remove M oldest active vectors
        # DEG handles edge reconnection and navigability repair automatically upon removal
        for remove_lbl in range(active_start_idx, active_start_idx + batch_size):
            builder.remove_entry(remove_lbl)

        # Process pending insertions and deletions
        builder.build()
        t_update_end = time.perf_counter()
        update_duration_sec = t_update_end - t_update_start

        # Total operations in update batch = insertions + deletions
        total_ops = batch_size * 2
        update_ops_per_ms = (total_ops / update_duration_sec) / 1000.0
        round_update_throughput.append(update_ops_per_ms)

        active_start_idx += batch_size
        next_insert_idx += batch_size

        # ---------------------------------------------------------------------
        # 2. Dynamic Ground Truth Calculation for the Current Active Window
        # ---------------------------------------------------------------------
        current_active_labels = np.arange(active_start_idx, next_insert_idx, dtype=np.uint32)
        current_active_features = base_vectors[active_start_idx:next_insert_idx]
        gt_labels = compute_exact_topk_labels(
            current_active_labels,
            current_active_features,
            query_vectors,
            k=search_k,
            metric=metric,
        )

        # ---------------------------------------------------------------------
        # 3. Query Benchmark Phase (Single-Threaded Batch Search)
        # ---------------------------------------------------------------------
        t_query_start = time.perf_counter()
        returned_labels = graph.search(
            query_vectors,
            k=search_k,
            eps=search_eps,
            threads=1,
            return_distances=False,
        )
        t_query_end = time.perf_counter()

        query_duration_sec = t_query_end - t_query_start
        search_qps = len(query_vectors) / query_duration_sec
        search_ms = (query_duration_sec / len(query_vectors)) * 1000.0

        # Calculate Recall@k
        num_matches = 0
        total_k = len(query_vectors) * search_k
        for q_idx in range(len(query_vectors)):
            gt_set = set(gt_labels[q_idx])
            for lbl in returned_labels[q_idx]:
                if lbl in gt_set:
                    num_matches += 1

        recall = num_matches / total_k
        round_recalls.append(recall)
        round_search_qps.append(search_qps)

        if (r + 1) % 5 == 0 or r == 0 or r == rounds - 1:
            print(f"{r + 1:6d} | {recall * 100:10.2f}% | {search_qps:12,.1f} | {search_ms:10.3f} | {update_ops_per_ms:10.2f}")

    print("-" * 62)
    print("\nBenchmark Summary:")
    print(f"  Average Recall@{search_k}:       {np.mean(round_recalls) * 100:.2f}% (std: {np.std(round_recalls) * 100:.2f}%)")
    print(f"  Average Search QPS:        {np.mean(round_search_qps):,.1f}")
    print(f"  Average Update Throughput: {np.mean(round_update_throughput):.2f} updates/ms")

    # -------------------------------------------------------------------------
    # Plotting Results: Reproducing CleANN Paper Fig. 20 & Fig. 21
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Recall over rounds (reproduces Figure 20 in CleANN paper)
    ax1.plot(range(1, rounds + 1), round_recalls, color="#1f77b4", linewidth=2, label="DEG (eps=0.03)")
    ax1.set_xlabel("Round (Sliding Window Batch)")
    ax1.set_ylabel(f"Recall@{search_k}")
    ax1.set_title(f"Figure 20 Reproduction: Recall@{search_k} Over Updates\n(Window Size: {window_size:,}, {update_fraction * 100:.0f}% update/round)")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Panel 2: Throughput over rounds (reproduces Figure 21 in CleANN paper)
    ax2.plot(range(1, rounds + 1), round_search_qps, color="#ff7f0e", linewidth=2, label="Search QPS")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Search QPS", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(1, rounds + 1), round_update_throughput, color="#2ca02c", linestyle=":", linewidth=2, label="Update Throughput (ops/ms)")
    ax2_twin.set_ylabel("Update Throughput (ops/ms)", color="#2ca02c")
    ax2_twin.tick_params(axis="y", labelcolor="#2ca02c")

    ax2.set_title("Figure 21 Reproduction: Search & Update Throughput\n(Single-Threaded Execution)")
    fig.tight_layout()

    if save_plot:
        out_path = Path(save_plot)
        fig.savefig(out_path, dpi=200)
        print(f"\nPlot saved to: {out_path.resolve()}")

    if not no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="DEG Sliding Window Benchmark (CleANN Paper RedCaps Baseline Experiment Reproduction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="redcaps",
        choices=list(DATASET_METADATA.keys()) + ["custom"],
        help="Benchmark dataset to automatically download & use (default: 'redcaps').",
    )
    parser.add_argument(
        "--base-file",
        type=str,
        default=None,
        help="Path to custom .fvecs / .fbin / .hdf5 base vector file.",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="Path to custom .fvecs / .fbin / .hdf5 query vector file.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=500_000,
        help="Number of active vectors maintained in the sliding window (CleANN paper: 500,000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        help="Number of vectors inserted and deleted per round (CleANN paper: 5,000 = 1%% of 500k).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="Number of sliding window update rounds (CleANN paper: 100).",
    )
    parser.add_argument(
        "--edges-per-vertex",
        "-k",
        type=int,
        default=32,
        help="Number of edges per vertex in the DEG graph.",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=10,
        help="k nearest neighbors to search (CleANN paper: k=10).",
    )
    parser.add_argument(
        "--search-eps",
        type=float,
        default=0.03,
        help="DEG search exploration epsilon (CleANN paper: eps=0.03).",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=800,
        help="Number of test queries to evaluate per round (RedCaps has 800 test queries).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive matplotlib plot window.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="sliding_window_benchmark.png",
        help="Path to save the benchmark plot.",
    )

    args = parser.parse_args()

    # Load custom dataset if provided, else download standard dataset
    if args.base_file and Path(args.base_file).is_file():
        print(f"Loading base vectors from {args.base_file}...")
        if args.base_file.endswith(".fbin"):
            base = fbin_read(args.base_file)
        else:
            base = fvecs_read(args.base_file)

        if args.query_file and Path(args.query_file).is_file():
            print(f"Loading query vectors from {args.query_file}...")
            if args.query_file.endswith(".fbin"):
                queries = fbin_read(args.query_file)
            else:
                queries = fvecs_read(args.query_file)
        else:
            print(f"No query file provided; sampling {args.num_queries} random vectors from base as queries...")
            queries = base[: args.num_queries].copy()
        metric = deglib.Metric.FP32_L2
    else:
        print(f"Dataset '{args.dataset}' requested. Ensuring download from Zenodo/server...")
        base, queries, metric, dataset_dir = load_benchmark_dataset(args.dataset)
        if len(queries) > args.num_queries:
            queries = queries[: args.num_queries]

    run_sliding_window_benchmark(
        base_vectors=base,
        query_vectors=queries,
        metric=metric,
        window_size=args.window_size,
        batch_size=args.batch_size,
        rounds=args.rounds,
        edges_per_vertex=args.edges_per_vertex,
        search_k=args.search_k,
        search_eps=args.search_eps,
        no_show=args.no_show,
        save_plot=args.save_plot,
    )


if __name__ == "__main__":
    main()
