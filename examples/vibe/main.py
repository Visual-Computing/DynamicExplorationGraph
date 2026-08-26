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
    ensure_dataset,
    get_default_cache_dir,
    load_vibe_dataset,
    resolve_dataset_key,
)
from module import DegANN
from presets import get_config_grid_presets, get_default_config_preset, load_vibe_config


class TeeLogger:
    """Duplicates stream writes to both the original terminal stream and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, message):
        self.stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def set_cpu_affinity(cpu_ids: list[int] | None = None) -> list[int] | None:
    """Pins the current process to specific CPU core(s) (cross-platform: Windows/Linux)."""
    if cpu_ids is None:
        return None
    try:
        import psutil

        p = psutil.Process()
        p.cpu_affinity(cpu_ids)
        current = p.cpu_affinity()
        print(f"Process CPU affinity pinned to core(s): {current}")
        return current
    except Exception as e:
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, cpu_ids)
                current = list(os.sched_getaffinity(0))
                print(f"Process CPU affinity pinned to core(s): {current}")
                return current
            except Exception as e2:
                print(f"Warning: Could not set CPU affinity via os.sched_setaffinity: {e2}", file=sys.stderr)
        else:
            print(f"Warning: Could not set CPU affinity: {e}", file=sys.stderr)
    return None


import matplotlib.colors as mcolors


def save_and_show_anns_plot(
    dataset_name: str,
    instruction_set: str,
    search_k: int,
    results_series: list,
    output_path: Path = None,
    no_show: bool = False,
):
    if not results_series:
        return

    plt.figure(figsize=(10, 7))

    # Color hues per optimization target (Red, Green, Blue)
    # LowLID -> Blue, HighLID -> Red, StreamingData -> Green
    OPT_BASE_COLORS = {
        "lowlid": "Blues",
        "highlid": "Reds",
        "streamingdata": "Greens",
        "streaming": "Greens",
    }

    # Collect all unique k values to compute intensity/saturation gradient
    all_k_values = sorted(list({entry.get("k", 30) for entry in results_series}))
    min_k = min(all_k_values) if all_k_values else 16
    max_k = max(all_k_values) if all_k_values else 48

    for entry in results_series:
        label = entry["label"]
        recalls = entry["recalls"]
        qps = entry["qps"]
        if not (recalls and qps):
            continue

        opt_target = str(entry.get("opt_target", "")).lower().strip()
        k = entry.get("k", 30)
        is_pruned = bool(entry.get("prune_non_rng", False))

        # Determine colormap based on opt_target
        cmap_name = OPT_BASE_COLORS.get(opt_target, "Purples")
        cmap = plt.get_cmap(cmap_name)

        # Scale intensity with k from 0.45 (light/pastel for small k) to 0.95 (deep/saturated for large k)
        if max_k > min_k:
            norm_factor = 0.45 + 0.50 * ((k - min_k) / (max_k - min_k))
        else:
            norm_factor = 0.75
        color = cmap(norm_factor)

        # Line style: dashed for pruned, solid for unpruned
        linestyle = "--" if is_pruned else "-"
        marker = "s" if is_pruned else "o"

        plt.plot(
            recalls,
            qps,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            linewidth=2,
            label=label,
        )

    plt.xlabel(f"Recall@{search_k}", fontsize=12)
    plt.ylabel("Queries Per Second (QPS)", fontsize=12)
    plt.title(f"DEG ANNS Benchmark on {dataset_name} ({instruction_set})", fontsize=14, pad=10)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.yscale("log")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path.resolve()}")

    if not no_show:
        plt.show()
    plt.close()

def compute_linear_search_baseline(
    base_vecs: np.ndarray, float_space: deglib.distances.FloatSpace, sample_size: int = 10
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
    return time_per_query_us / 10


def main():
    parser = argparse.ArgumentParser(description="VIBE (Vector Index Benchmark for Embeddings) ANNS benchmark with DEG")
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
    parser.add_argument(
        "--cpu",
        "--cpu-affinity",
        dest="cpu_affinity",
        type=int,
        nargs="+",
        default=None,
        help="Pin benchmark process to specific CPU core ID(s) (e.g. --cpu 0 or --cpu 0 1 2 3).",
    )
    parser.add_argument(
        "--query-dtype",
        "--dtype",
        dest="query_dtype",
        type=str,
        default="float32",
        choices=["float32", "fp32", "float16", "fp16", "int8", "sq8", "uint8", "u8"],
        help="Data type used to query the DEG graph (default: float32).",
    )
    parser.add_argument(
        "--k-rerank",
        "--rerank",
        dest="k_rerank",
        type=int,
        default=None,
        help="Number of candidates to retrieve using quantized graph before reranking with FP16 features.",
    )
    parser.add_argument(
        "--prune-non-rng",
        "--mrng",
        action="store_true",
        dest="prune_non_rng",
        help="Prune all non-RNG edges (converting graph to MRNG) before searching or after building.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or get_default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = resolve_dataset_key(args.dataset)
    if dataset_key not in VIBE_DATASETS:
        print(f"Error: Unknown dataset '{args.dataset}'.", file=sys.stderr)
        print(f"Available VIBE datasets: {', '.join(VIBE_DATASETS.keys())}", file=sys.stderr)
        sys.exit(1)

    dataset_dir, _, _ = ensure_dataset(dataset_key, cache_dir)
    deg_dir = dataset_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)

    # Set up console logging to both terminal and a log file in the graph/deg output directory
    log_file_path = deg_dir / f"{dataset_key}_benchmark.log"
    log_file = open(log_file_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(original_stdout, log_file)
    sys.stderr = TeeLogger(original_stderr, log_file)

    try:
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting VIBE benchmark run")
        print(f"Log file: {log_file_path.resolve()}")
        print(f"Selected VIBE Dataset: {dataset_key}")
        print(f"Cache directory: {cache_dir}")

        # Pin CPU core affinity if requested
        if args.cpu_affinity is not None:
            set_cpu_affinity(args.cpu_affinity)

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
            # Base graph is always FP32 on disk with FLAS enabled
            is_l2 = metric in (deglib.Metric.FP32_L2, deglib.Metric.FP16_L2, deglib.Metric.Int8_L2, deglib.Metric.Uint8_L2)
            build_metric_str = deglib.Metric.FP32_L2.name if is_l2 else deglib.Metric.FP32_InnerProduct.name
            graph_path = build_graph_filename(
                dataset_key=dataset_key,
                cache_dir=cache_dir,
                dims=dims,
                k=k,
                extend_k=extend_k,
                extend_eps=build_eps,
                optimization_target_str=opt_target,
                metric_str=build_metric_str,
                use_flas=True,
            )

            if args.rebuild_graph and graph_path and graph_path.is_file():
                graph_path.unlink()

            adapter = DegANN(
                metric=meta.get("hdf5_distance", metric.name),
                k=k,
                opt_target=opt_target,
                extend_k=extend_k,
                build_eps=build_eps,
                improve_k=imp_k,
                improve_eps=imp_eps,
                prune_non_rng=args.prune_non_rng or cfg.get("prune_non_rng", False),
                threads=args.build_threads,
                query_dtype=args.query_dtype,
                k_rerank=args.k_rerank,
                graph_path=str(graph_path),
            )
            print(
                f"\n--- [{cfg_idx + 1}/{len(configs_to_run)}] Fitting / Loading Index: K={k}, ExtendK={extend_k}, Eps={build_eps:.2f}, Opt={opt_target}, Threads={args.build_threads} ---"
            )
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
                    print(
                        f"  eps {eps:.3f} \t ABORTED ({time_us_per_query}us/query > {int(linear_baseline_us)}us baseline)"
                    )
                    break

                if recall > 0.999:
                    print("  Reached recall > 0.999, stopping further test iterations for this graph.")
                    break

            is_pruned = args.prune_non_rng or cfg.get("prune_non_rng", False)
            prune_label = ", MRNG" if is_pruned else ""
            results_series.append(
                {
                    "label": f"DEG ({opt_target}, K={k}{prune_label})",
                    "opt_target": opt_target,
                    "k": k,
                    "prune_non_rng": is_pruned,
                    "recalls": anns_recalls,
                    "qps": anns_qps,
                }
            )

            # Explicitly free loaded graph and adapter memory between test iterations
            # del adapter
            # import gc
            # gc.collect()

        # Plot and save all curve series in a combined figure
        plot_output_path = deg_dir / f"{dataset_key}_anns_benchmark.png"
        save_and_show_anns_plot(
            dataset_name=meta["name"],
            instruction_set=instruction_set,
            search_k=actual_k,
            results_series=results_series,
            output_path=plot_output_path,
            no_show=args.no_show,
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
