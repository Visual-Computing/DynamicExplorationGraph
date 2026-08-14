import argparse
import os
import sys
import time
import multiprocessing
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import deglib
from deglib_cpp import avx_usable, avx512_usable
from dataset_utils import (
    load_dataset,
    get_default_cache_dir,
    resolve_dataset_key,
    build_graph_filename,
    DATASET_METADATA,
)
from presets import get_preset
from graph_analysis import analyze_graph

def _render_anns_plot(dataset_name: str, instruction_set: str, search_k: int, anns_recalls: list, anns_qps: list):
    plt.figure(figsize=(8, 6))
    plt.plot(anns_recalls, anns_qps, marker='o', linewidth=2, color='tab:blue', label=f"DEG ({dataset_name})")
    plt.xlabel(f"Recall@{search_k}")
    plt.ylabel("Queries Per Second (QPS)")
    plt.title(f"DEG ANNS Search Benchmark on {dataset_name} ({instruction_set})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_anns_results(
    dataset_name: str,
    instruction_set: str,
    search_k: int,
    anns_recalls: list,
    anns_qps: list,
    no_show: bool = False
):
    if not no_show and anns_recalls:
        print("\nDisplaying interactive ANNS plot window (in separate GUI process)...")
        p = multiprocessing.Process(
            target=_render_anns_plot,
            args=(dataset_name, instruction_set, search_k, list(anns_recalls), list(anns_qps))
        )
        p.start()

def _render_explore_plot(dataset_name: str, instruction_set: str, explore_k: int, explore_recalls: list, explore_qps: list):
    plt.figure(figsize=(8, 6))
    plt.plot(explore_recalls, explore_qps, marker='s', linewidth=2, color='tab:orange', label=f"DEG Explore ({dataset_name})")
    plt.xlabel(f"Explore Recall@{explore_k}")
    plt.ylabel("Queries Per Second (QPS)")
    plt.title(f"DEG Graph Exploration Benchmark on {dataset_name} ({instruction_set})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_explore_results(
    dataset_name: str,
    instruction_set: str,
    explore_k: int,
    explore_recalls: list,
    explore_qps: list,
    no_show: bool = False
):
    if not no_show and explore_recalls:
        print("\nDisplaying interactive Exploration plot window...")
        p = multiprocessing.Process(
            target=_render_explore_plot,
            args=(dataset_name, instruction_set, explore_k, list(explore_recalls), list(explore_qps))
        )
        p.start()

def compute_linear_search_baseline(base_vecs: np.ndarray, float_space: deglib.distances.FloatSpace, sample_size: int = 100) -> float:
    n_base = len(base_vecs)
    query_count = min(sample_size, n_base)
    rng = np.random.default_rng(7)
    sample_indices = rng.choice(n_base, size=query_count, replace=False)

    print(f"\n--- Computing Linear Search Baseline ---")
    print(f"Computing linear search baseline with {query_count} random queries on {n_base} base vectors using C++ {float_space.metric().name} ({float_space.get_instruction().name})...")

    start_time = time.perf_counter()
    min_dist = float('inf')

    for idx in sample_indices:
        dists = float_space.compute_distances(base_vecs[idx], base_vecs)
        d_min = float(np.min(dists))
        if d_min < min_dist:
            min_dist = d_min

    total_time_us = (time.perf_counter() - start_time) * 1e6
    time_per_query_us = total_time_us / max(query_count, 1)
    linear_baseline_us = time_per_query_us * 2
    print(f"Linear search baseline: {int(time_per_query_us)}us per query (total: {int(total_time_us / 1000)}ms for {query_count} queries) with min distance {min_dist:.6f}")
    return linear_baseline_us

def run_static_benchmark(
    dataset_key: str,
    cache_dir: Path,
    graph_path_arg: str | None,
    instruction_set: str,
    build_threads: int,
    max_base_vecs: int | None,
    no_show: bool
):
    resolved_key = resolve_dataset_key(dataset_key)
    preset = get_preset(resolved_key)
    
    # Load dataset
    base_vecs, query_vecs, gt_vecs, explore_entry, explore_gt, meta = load_dataset(resolved_key, cache_dir)

    if max_base_vecs and max_base_vecs < len(base_vecs):
        print(f"Limiting base vectors to first {max_base_vecs} samples for fast run.")
        base_vecs = base_vecs[:max_base_vecs]

    dims = base_vecs.shape[1]
    metric = preset["metric"]

    # Map instruction_set string to deglib.cpu.InstructionSet
    inst_lower = instruction_set.lower()
    if inst_lower == "scalar":
        instruction_enum = deglib.cpu.InstructionSet.Scalar
    elif inst_lower == "avx2":
        instruction_enum = deglib.cpu.InstructionSet.AVX2
    elif inst_lower == "avx512":
        instruction_enum = deglib.cpu.InstructionSet.AVX512
    else:
        instruction_enum = deglib.cpu.InstructionSet.Auto

    target_str = preset.get("optimization_target", "LowLID")
    optimization_target = (
        deglib.builder.OptimizationTarget.HighLID
        if target_str == "HighLID"
        else deglib.builder.OptimizationTarget.LowLID
    )

    print(f"\n=== Benchmarking Static Graph Creation & Evaluation for Dataset: {meta['name']} ===")
    print(f"Selected Instruction Set: {instruction_set}")
    print(f"Build Threads: {build_threads}")
    print(f"Base vectors: {base_vecs.shape[0]} | Dimension: {dims}")
    print(f"Query vectors: {query_vecs.shape[0]} | GT shape: {gt_vecs.shape}")

    feature_space = deglib.distances.FloatSpace.create(dims, metric, instruction_enum)
    actual_inst = str(feature_space.get_instruction().name)
    print(f"Distance Metric: {metric.name} ({actual_inst})")

    linear_baseline_us = compute_linear_search_baseline(base_vecs, feature_space, sample_size=100)

    graph_path = Path(graph_path_arg) if graph_path_arg else None
    graph = None

    # Load from file if --graph-path was explicitly provided and file exists
    if graph_path and graph_path.exists():
        print(f"\n=== Loading Graph from File: {graph_path} ===")
        graph = deglib.load_readonly_graph(str(graph_path))
        print(f"Graph loaded: {graph.size()} vertices")
    else:
        if graph_path:
            print(f"\n=== Building Graph with {build_threads} threads (saving to {graph_path}) ===")
        else:
            print(f"\n=== Building Graph with {build_threads} threads (in RAM, no file path specified) ===")

        build_start = time.perf_counter()
        graph_mut = deglib.DynamicExplorationGraph.create_empty(
            capacity=base_vecs.shape[0],
            feature_space=deglib.distances.FloatSpace.create(dims, metric, instruction_enum),
            edges_per_vertex=preset["k"],
        )
        builder = deglib.GraphBuilder(
            graph_mut,
            seed=7,
            optimization_target=optimization_target,
            extend_k=preset.get("extend_k", preset["k"]),
            extend_eps=preset["build_eps"],
            improve_k=preset.get("improve_k", 0),
            improve_eps=preset.get("improve_eps", 0.0),
        )
        builder.set_batch_size(10, 10)
        if build_threads > 1:
            builder.set_thread_count(build_threads)

        labels = np.arange(base_vecs.shape[0], dtype=np.uint32)
        builder.add_entry(labels, base_vecs)
        builder.build(callback="progress")

        build_time = time.perf_counter() - build_start
        print(f"Graph built in {build_time:.2f} seconds ({graph_mut.size()} vertices).")

        if graph_path:
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_mut.save_graph(str(graph_path))
            print(f"Graph saved to file: {graph_path}")
            graph = deglib.load_readonly_graph(str(graph_path))
        else:
            graph = graph_mut.to_readonly()

    # Graph Analysis
    analyze_graph(graph)

    # ANNS Test
    search_k = min(preset["anns_k"], gt_vecs.shape[1])
    repeat = preset.get("anns_repeat", 1)
    eps_list = sorted(preset["search_eps_list"])

    print(f"\n--- ANNS Test (k={search_k}) ---")
    print(f"Compute TOP{search_k} for eps {', '.join(f'{e:.3f}' for e in eps_list)}")

    anns_recalls = []
    anns_qps = []
    n_queries = len(query_vecs)

    for eps in eps_list:
        start_time = time.perf_counter()
        for _ in range(repeat):
            indices_batch, _ = graph.search(query_vecs, eps=eps, k=search_k, threads=1)
        elapsed_sec = time.perf_counter() - start_time

        search_time_us = elapsed_sec * 1e6
        time_us_per_query = int((search_time_us / max(n_queries, 1)) / repeat)
        qps = n_queries / max(elapsed_sec / repeat, 1e-9)

        hits = 0
        total_returned = 0
        for i in range(n_queries):
            gt_set = set(gt_vecs[i, :search_k])
            ret_set = set(indices_batch[i])
            hits += len(gt_set.intersection(ret_set))
            total_returned += len(gt_set)

        recall = hits / max(total_returned, 1)
        anns_recalls.append(recall)
        anns_qps.append(qps)

        print(f"eps {eps:6.3f} \trecall {recall:.5f} \ttime_us_per_query {time_us_per_query:6d}us \t{qps:9.1f} qps \tsearch time: {int(search_time_us / 1000):6d}ms")

        if linear_baseline_us > 0 and time_us_per_query > linear_baseline_us:
            print(f"eps {eps:.3f} \t ABORTED ({time_us_per_query}us/query > {int(linear_baseline_us)}us baseline)")
            break

        if recall > 0.997:
            print("Reached recall > 0.997, stopping further tests.")
            break

    plot_anns_results(
        dataset_name=meta['name'],
        instruction_set=instruction_set,
        search_k=search_k,
        anns_recalls=anns_recalls,
        anns_qps=anns_qps,
        no_show=no_show,
    )

    # Exploration Test
    if explore_entry is not None and explore_gt is not None:
        explore_k = preset.get("explore_k", 1000)
        actual_explore_k = min(explore_k, explore_gt.shape[1])
        explore_depth = preset.get("explore_depth", 3)
        explore_repeat = preset.get("explore_repeat", 1)

        print(f"\n--- Exploration Test (k={actual_explore_k}) ---")

        entry_labels = np.ascontiguousarray(explore_entry, dtype=np.uint32)
        explore_recalls = []
        explore_qps = []
        last_recall = -1.0
        query_count = len(entry_labels)

        k_factor = 100
        max_dist_steps = []
        for f in range(explore_depth + 1):
            start_i = 1 if f == 0 else 2
            for i in range(start_i, 11):
                max_dist = (actual_explore_k + k_factor * (i - 1)) if f == 0 else (k_factor * i)
                max_dist_steps.append(max_dist)
            k_factor *= 10

        for max_dist in max_dist_steps:
            start_time = time.perf_counter()
            for _ in range(explore_repeat):
                indices_batch, _ = graph.explore(
                    entry_labels,
                    k=actual_explore_k,
                    max_distance_computation_count=max_dist,
                    include_entry=True,
                    threads=1,
                )
            elapsed_sec = time.perf_counter() - start_time

            search_time_us = elapsed_sec * 1e6
            time_us_per_query = int(search_time_us / max(query_count * explore_repeat, 1))
            qps = query_count / max(elapsed_sec / explore_repeat, 1e-9)

            hits = 0
            total_returned = 0
            for i in range(query_count):
                gt_set = set(explore_gt[i, :actual_explore_k])
                ret_set = set(indices_batch[i])
                hits += len(gt_set.intersection(ret_set))
                total_returned += len(gt_set)

            recall = hits / max(total_returned, 1)
            explore_recalls.append(recall)
            explore_qps.append(qps)

            print(f"k {actual_explore_k:5d}, max_distance_count {max_dist:6d}, recall {recall:.4f}, time_us_per_query {time_us_per_query:6d}us, {qps:9.1f} qps")

            if linear_baseline_us > 0 and time_us_per_query > linear_baseline_us:
                print(f"max_distance_count {max_dist:5d}, k {actual_explore_k:4d}, ABORTED ({time_us_per_query}us/query > {int(linear_baseline_us)}us baseline)")
                break

            if recall == last_recall:
                print(f"Recall stabilized at {recall:.4f}, stopping exploration sweep")
                break
            last_recall = recall

            if recall >= 0.997:
                print(f"Recall target 0.997 reached, stopping exploration sweep")
                break

        plot_explore_results(
            dataset_name=meta['name'],
            instruction_set=instruction_set,
            explore_k=actual_explore_k,
            explore_recalls=explore_recalls,
            explore_qps=explore_qps,
            no_show=no_show,
        )

def main():
    parser = argparse.ArgumentParser(description="DEG Benchmark Tool: bench_static_data (Python)")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="audio",
        choices=["sift1m", "deep1m", "glove", "audio", "enron", "all"],
        help="Dataset name (e.g. sift1m, deep1m, glove, audio, enron, all) (default: audio)",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=None,
        help="Save generated .deg graph files to this path. (default: none)")
    parser.add_argument(
        "--instruction",
        choices=["auto", "scalar", "avx2", "avx512"],
        default="auto",
        help="SIMD instruction set to use (default: auto)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding graph files even if they exist",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads used for building the graph"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom directory to cache datasets (default: ~/.cache/deg_datasets)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive plot window"
    )
    parser.add_argument(
        "--max-base-vecs",
        type=int,
        default=None,
        help="Optional limit on base vectors for quick debugging/testing"
    )

    args = parser.parse_args()

    dataset_name = args.dataset 
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_default_cache_dir()

    # Detect CPU Instruction Set
    if args.instruction != "auto":
        instruction_set = args.instruction.upper()
    elif avx512_usable():
        instruction_set = "AVX512"
    elif avx_usable():
        instruction_set = "AVX2"
    else:
        instruction_set = "Scalar"

    print("=== bench_static_data (Python) ===")
    print(f"Data Cache Directory: {cache_dir.resolve()}")

    if dataset_name.lower() == "all":
        datasets_to_run = ["sift1m", "deep1m", "glove-100", "audio", "enron"]
    else:
        datasets_to_run = [dataset_name]

    for ds in datasets_to_run:
        run_static_benchmark(
            dataset_key=ds,
            cache_dir=cache_dir,
            graph_path_arg=args.graph_path,
            instruction_set=instruction_set,
            build_threads=args.threads,
            max_base_vecs=args.max_base_vecs,
            no_show=args.no_show,
        )

    print("\nStatic Benchmark Finished Successfully.")

if __name__ == "__main__":
    main()
