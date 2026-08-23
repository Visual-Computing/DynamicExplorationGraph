import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import deglib
from deglib_cpp import avx_usable, avx512_usable

from dataset_utils import (
    VIBE_DATASETS,
    build_graph_filename,
    get_default_cache_dir,
    load_vibe_dataset,
    resolve_dataset_key,
)
from module import DegANN
from presets import get_config_grid_presets, get_default_config_preset, load_vibe_config


def _render_anns_plot(dataset_name: str, instruction_set: str, search_k: int, results_series: list):
    plt.figure(figsize=(9, 6.5))
    for entry in results_series:
        label = entry["label"]
        recalls = entry["recalls"]
        qps = entry["qps"]
        if recalls and qps:
            plt.plot(recalls, qps, marker="o", linewidth=2, label=label)

    plt.xlabel(f"Recall@{search_k}")
    plt.ylabel("Queries Per Second (QPS)")
    plt.title(f"DEG ANNS Benchmark on {dataset_name} ({instruction_set})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_anns_results(
    dataset_name: str, instruction_set: str, search_k: int, results_series: list, no_show: bool = False
):
    if not no_show and results_series:
        print("\nDisplaying interactive ANNS plot window (in separate GUI process)...")
        p = multiprocessing.Process(
            target=_render_anns_plot, args=(dataset_name, instruction_set, search_k, list(results_series))
        )
        p.start()


def compute_linear_search_baseline(
    base_vecs: np.ndarray, float_space: deglib.distances.FloatSpace, sample_size: int = 100
) -> float:
    n_base = len(base_vecs)
    query_count = min(sample_size, n_base)
    rng = np.random.default_rng(7)
    sample_indices = rng.choice(n_base, size=query_count, replace=False)

    print("\n--- Computing Linear Search Baseline ---")
    print(
        f"Computing linear search baseline with {query_count} random queries on {n_base:,} base vectors using C++ {float_space.metric().name} ({float_space.get_instruction().name})..."
    )

    start_time = time.perf_counter()
    min_dist = float("inf")

    for idx in sample_indices:
        dists = float_space.compute_distances(base_vecs[idx], base_vecs)
        d_min = float(np.min(dists))
        if d_min < min_dist:
            min_dist = d_min

    elapsed_time = time.perf_counter() - start_time
    time_per_query_sec = elapsed_time / query_count
    time_per_query_us = time_per_query_sec * 1e6
    qps = 1.0 / time_per_query_sec if time_per_query_sec > 0 else 0

    print(
        f"Linear Scan Baseline: {time_per_query_us:.1f} us/query ({qps:.1f} QPS), "
        f"Total time for {query_count} queries: {elapsed_time * 1000:.1f} ms\n"
    )
    return time_per_query_us


def main():
    parser = argparse.ArgumentParser(
        description="VIBE (Vector Index Benchmark for Embeddings) ANNS benchmark with DEG"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="agnews-mxbai",
        help=f"Dataset name. Available: {', '.join(VIBE_DATASETS.keys())} (default: agnews-mxbai)",
    )
    parser.add_argument(
        "--cache-dir",
        "-c",
        type=Path,
        default=None,
        help="Directory to store datasets and built graphs (default: D:/Data/VIBE or ~/.cache/vibe)",
    )
    default_build_threads = max(1, (os.cpu_count() or 2) // 2)
    parser.add_argument(
        "--build-threads",
        "-t",
        type=int,
        default=default_build_threads,
        help=f"Number of threads for graph building (default: {default_build_threads} [half of CPU cores])",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Graph degree k (edges per vertex). Defaults to preset.",
    )
    parser.add_argument(
        "--extend-k",
        type=int,
        default=None,
        help="Additional edges checked during construction. Defaults to preset.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Build eps factor during graph construction. Defaults to preset.",
    )
    parser.add_argument(
        "--force-rebuild",
        "--rebuild-graph",
        action="store_true",
        dest="rebuild_graph",
        help="Force rebuild graph files even if cached graph exists.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open GUI plot windows at completion.",
    )
    parser.add_argument(
        "--anns-k",
        type=int,
        default=100,
        help="Target number of top-k nearest neighbors to evaluate (default: 100).",
    )
    parser.add_argument(
        "--anns-repeat",
        type=int,
        default=None,
        help="Repeat ANNS query run N times to average query timing (default: from preset).",
    )
    parser.add_argument(
        "--search-eps-list",
        type=float,
        nargs="+",
        default=None,
        help="Custom list of eps values for ANNS search benchmark.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = resolve_dataset_key(args.dataset)
    if dataset_key not in VIBE_DATASETS:
        print(f"Error: Unknown dataset '{args.dataset}'.", file=sys.stderr)
        print(f"Available VIBE datasets: {', '.join(VIBE_DATASETS.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"Selected VIBE Dataset: {dataset_key}")
    print(f"Cache directory: {cache_dir}")

    # Load dataset
    base_vecs, query_vecs, gt_vecs, meta = load_vibe_dataset(dataset_key, cache_dir)
    dims = base_vecs.shape[1]
    metric = meta["metric"]
    float_space = deglib.distances.FloatSpace.create(dims, metric)
    instruction_set = float_space.get_instruction().name

    print(f"\n--- System & Distance Config ---")
    print(f"Metric: {metric.name}, Dimensions: {dims}")
    print(f"Vector Space Type: FloatSpace ({instruction_set})")
    print(f"Hardware AVX: {avx_usable()}, AVX-512: {avx512_usable()}")

    # Linear scan baseline
    linear_baseline_us = compute_linear_search_baseline(base_vecs, float_space)

    # Get configurations to run (either user-overridden single config or full grid from config.yml)
    if args.k is not None:
        configs_to_run = [get_default_config_preset(dataset_key)]
        configs_to_run[0]["k"] = args.k
    else:
        configs_to_run = get_config_grid_presets(dataset_key)

    actual_k = min(args.anns_k, gt_vecs.shape[1])
    repeat = args.anns_repeat if args.anns_repeat is not None else 1
    results_series = []

    print(f"\n=======================================================")
    print(f"Running VIBE ANNS Benchmark on {dataset_key} ({len(configs_to_run)} graph config(s))")
    print(f"=======================================================")

    for cfg_idx, cfg in enumerate(configs_to_run):
        k = cfg["k"]
        extend_k = args.extend_k if args.extend_k is not None else cfg.get("extend_k", 2 * k)
        build_eps = args.eps if args.eps is not None else cfg["build_eps"]
        opt_target = cfg["optimization_target"]
        imp_k = cfg.get("improve_k", 0)
        imp_eps = cfg.get("improve_eps", 0.0)
        eps_list = sorted(args.search_eps_list if args.search_eps_list is not None else cfg["search_eps_list"])

        graph_path = build_graph_filename(
            dataset_key=dataset_key,
            cache_dir=cache_dir,
            dims=dims,
            k=k,
            extend_k=extend_k,
            extend_eps=build_eps,
            optimization_target_str=opt_target,
            metric_str=metric.name,
        )

        if args.rebuild_graph and graph_path and graph_path.is_file():
            graph_path.unlink()

        adapter = DegANN(
            metric=meta.get("hdf5_distance", metric.name),
            k=k,
            extend_k=extend_k,
            build_eps=build_eps,
            opt_target=opt_target,
            improve_k=imp_k,
            improve_eps=imp_eps,
            threads=args.build_threads,
            graph_path=str(graph_path),
        )

        print(f"\n--- [{cfg_idx + 1}/{len(configs_to_run)}] Fitting / Loading Index: K={k}, ExtendK={extend_k}, Eps={build_eps:.2f}, Opt={opt_target}, Threads={args.build_threads} ---")
        adapter.fit(base_vecs)

        deglib.analysis.analyze_graph(adapter.graph)

        print(f"Evaluating top-{actual_k} search for eps: {', '.join(f'{e:.3f}' for e in eps_list)}")

        anns_recalls = []
        anns_qps = []
        n_queries = len(query_vecs)

        for eps in eps_list:
            adapter.set_query_arguments(eps)
            start_time = time.perf_counter()
            for _ in range(repeat):
                adapter.batch_query(query_vecs, n=actual_k)
            elapsed_sec = time.perf_counter() - start_time
            indices_batch = adapter.get_batch_results()

            search_time_us = elapsed_sec * 1e6
            time_us_per_query = int((search_time_us / max(n_queries, 1)) / repeat)
            qps = n_queries / max(elapsed_sec / repeat, 1e-9)

            hits = 0
            total_returned = 0
            for i in range(n_queries):
                gt_set = set(gt_vecs[i, :actual_k])
                ret_set = set(indices_batch[i])
                hits += len(gt_set.intersection(ret_set))
                total_returned += len(gt_set)

            recall = hits / max(total_returned, 1)
            anns_recalls.append(recall)
            anns_qps.append(qps)

            print(
                f"  eps {eps:6.3f} \trecall@{actual_k}: {recall:.5f} \t{time_us_per_query:6d} us/query \t{qps:9.1f} QPS \tsearch time: {int(search_time_us / 1000):6d}ms"
            )

            if linear_baseline_us > 0 and time_us_per_query > linear_baseline_us:
                print(f"  eps {eps:.3f} \t ABORTED ({time_us_per_query}us/query > {int(linear_baseline_us)}us baseline)")
                break

            if recall > 0.999:
                print("  Reached recall > 0.999, stopping further test iterations for this graph.")
                break

        results_series.append(
            {
                "label": f"DEG ({opt_target}, K={k})",
                "recalls": anns_recalls,
                "qps": anns_qps,
            }
        )

    # Plot all curve series in a single combined window
    plot_anns_results(
        dataset_name=meta["name"],
        instruction_set=instruction_set,
        search_k=actual_k,
        results_series=results_series,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
