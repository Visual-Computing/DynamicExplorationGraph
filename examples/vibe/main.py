import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path
import numpy as np

import deglib
from deglib_cpp import avx_usable, avx512_usable

from dataset import (
    VIBE_DATASETS,
    build_graph_filename,
    ensure_dataset,
    get_default_cache_dir,
    load_vibe_dataset,
    resolve_dataset_key,
)
from module import DEG, QG
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


from plot import export_interactive_html


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1 and b.ndim == 2:
        return np.linalg.norm(b - a, axis=1)
    if a.ndim == 2 and b.ndim == 1:
        return np.linalg.norm(a - b, axis=1)
    return np.linalg.norm(a - b)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1 and b.ndim == 2:
        return np.clip(1.0 - np.dot(b, a) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1)), a_min=0.0, a_max=2.0)
    if a.ndim == 2 and b.ndim == 1:
        return np.clip(1.0 - np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b)), a_min=0.0, a_max=2.0)
    return np.clip(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), a_min=0.0, a_max=2.0)


def _ip_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1 and b.ndim == 2:
        return -np.dot(b, a)
    if a.ndim == 2 and b.ndim == 1:
        return -np.dot(a, b)
    return -np.dot(a, b)


def _normalized_cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # cosine distance assuming a and b are already normalized
    return np.clip(1.0 + _ip_distance(a, b), a_min=0.0, a_max=2.0)


_VIBE_METRICS = {
    "euclidean": _euclidean_distance,
    "l2": _euclidean_distance,
    "cosine": _cosine_distance,
    "ip": _ip_distance,
    "normalized": _normalized_cosine_distance,
}
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
    return time_per_query_us / 40


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
        help="Directory to store datasets and built graphs (default: ~/.cache/vibe)",
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
        "--no-show",
        action="store_true",
        help="Do not open GUI plot windows at completion.",
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
        "--algorithm",
        "-a",
        type=str,
        default=None,
        choices=["deg", "deg-qg", "qg", "all"],
        help="Select specific algorithm to benchmark ('deg', 'deg-qg' / 'qg', or 'all'). Default: run all enabled in config.yml.",
    )
    parser.add_argument(
        "--query-dtype",
        type=str,
        default=None,
        choices=["float32", "int8"],
        help="Override feature storage and search vector dtype ('float32', 'int8'). Defaults to config.yml.",
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
        base_vecs, query_vecs, gt_vecs, gt_distances, meta = load_vibe_dataset(dataset_key, cache_dir)
        dims = base_vecs.shape[1]
        metric_str = meta["metric"]
        is_l2 = metric_str.lower().strip() in ("euclidean", "l2", "fp32_l2")
        metric_enum = deglib.Metric.FP32_L2 if is_l2 else deglib.Metric.FP32_InnerProduct
        float_space = deglib.distances.FloatSpace.create(dims, metric_enum)
        instruction_set = float_space.get_instruction().name

        print(f"\n--- System & Distance Config ---")
        print(f"Metric: {metric_str} ({metric_enum.name}), Dimensions: {dims}")
        print(f"Vector Space Type: FloatSpace ({instruction_set})")
        print(f"Query Dtype: {args.query_dtype.upper() if args.query_dtype else 'FROM CONFIG.YML'}")
        print(f"Hardware AVX: {avx_usable()}, AVX-512: {avx512_usable()}")

        # Linear scan baseline
        linear_baseline_us = compute_linear_search_baseline(base_vecs, float_space)

        # Filter algorithm if requested (e.g. --algorithm deg or --algorithm deg-qg)
        alg_filter = args.algorithm
        if alg_filter == "qg":
            alg_filter = "deg-qg"
        elif alg_filter == "all":
            alg_filter = None

        configs_to_run = get_config_grid_presets(dataset_key, algorithm_name=alg_filter)


        actual_k = min(cfg_k_target := 100, gt_vecs.shape[1])
        results_series = []

        print(f"\n=======================================================")
        print(f"Running VIBE ANNS Benchmark on {dataset_key} ({len(configs_to_run)} graph config(s))")
        print(f"=======================================================")

        for cfg_idx, cfg in enumerate(configs_to_run):
            k = cfg["k"]
            opt_target = cfg["optimization_target"]
            is_pruned = cfg.get("prune_non_rng", False)
            constructor_name = cfg.get("constructor", "DEG")
            alg_name = cfg.get("alg_name", "deg")

            if constructor_name == "QG":
                adapter = QG(
                    metric=metric_str,
                    k=k,
                    opt_target=opt_target,
                    prune_non_rng=is_pruned,
                    threads=args.build_threads,
                )
                query_dtype = "int8"
            else:
                adapter = DEG(
                    metric=metric_str,
                    k=k,
                    opt_target=opt_target,
                    prune_non_rng=is_pruned,
                    threads=args.build_threads,
                )
                query_dtype = "float32"

            print(f"\n--- [{cfg_idx + 1}/{len(configs_to_run)}] Processing Algorithm '{alg_name}' ({constructor_name}) ---")
            adapter.fit(base_vecs, cache_dir=cache_dir)

            if adapter.graph is not None:
                deglib.analysis.analyze_graph(adapter.graph)

            # Determine query evaluation parameter grid
            factors_to_eval = cfg.get("rerank_size_factors", [1.0])
            eps_list = sorted(cfg.get("search_eps_list", []))
            ef_list = sorted(cfg.get("ef_list", []))

            prune_label = ", MRNG" if is_pruned else ""

            for r_factor in factors_to_eval:
                rerank_label = f", Rerank={r_factor:.2f}x" if r_factor > 1.0 else ""
                eval_params = [(eps, 0) for eps in eps_list] if eps_list else [(0.0, ef) for ef in ef_list]
                param_str = f"eps: {', '.join(f'{e:.3f}' for e in eps_list)}" if eps_list else f"ef: {', '.join(str(ef) for ef in ef_list)}"

                print(f"\nEvaluating top-{actual_k} search (rerank_factor={r_factor:.2f}) for {param_str}")

                anns_recalls = []
                anns_qps = []
                anns_params = []
                n_queries = len(query_vecs)

                for (eps, ef) in eval_params:
                    if ef > 0:
                        adapter.set_query_arguments(rerank_size_factor=r_factor, ef=ef)
                    else:
                        adapter.set_query_arguments(rerank_size_factor=r_factor, search_eps=eps)

                    results_indices = []
                    start_time = time.perf_counter()
                    for q_idx in range(n_queries):
                        results_indices.append(adapter.query(query_vecs[q_idx], actual_k))
                    elapsed_sec = time.perf_counter() - start_time
                    indices_batch = np.array(results_indices, dtype=np.int64)

                    search_time_us = elapsed_sec * 1e6
                    time_us_per_query = int(search_time_us / max(n_queries, 1))
                    qps = n_queries / max(elapsed_sec, 1e-9)
                    hits = 0
                    total_returned = n_queries * actual_k
                    if gt_distances is not None:
                        # Official VIBE distance-tolerance recall calculation
                        dist_fn = _VIBE_METRICS.get(metric_str.lower().strip(), _ip_distance)
                        for i in range(n_queries):
                            t = gt_distances[i, actual_k - 1] + 1e-3
                            ret_vecs = base_vecs[indices_batch[i, :actual_k]]
                            pred_dists = dist_fn(query_vecs[i], ret_vecs)
                            hits += int((pred_dists <= t).sum())
                    else:
                        for i in range(n_queries):
                            gt_set = set(gt_vecs[i, :actual_k])
                            ret_set = set(indices_batch[i, :actual_k])
                            hits += len(gt_set.intersection(ret_set))

                    recall = hits / max(total_returned, 1)
                    anns_recalls.append(recall)
                    anns_qps.append(qps)
                    anns_params.append(f"ef={ef}" if ef > 0 else f"eps={eps:g}")

                    lbl = f"ef {ef:4d}" if ef > 0 else f"eps {eps:6.3f}"
                    print(
                        f"  {lbl} \trecall@{actual_k}: {recall:.5f} \t{time_us_per_query:6d} us/query \t{qps:9.1f} QPS \tsearch time: {int(search_time_us / 1000):6d}ms"
                    )

                    if linear_baseline_us > 0 and time_us_per_query > linear_baseline_us:
                        print(
                            f"  ABORTED ({time_us_per_query}us/query > {int(linear_baseline_us)}us baseline)"
                        )
                        break

                    if recall > 0.999:
                        print("  Reached recall > 0.999, stopping further test iterations for this configuration.")
                        break

                results_series.append(
                    {
                        "label": f"{alg_name.upper()} ({opt_target}, K={k}{prune_label}{rerank_label})",
                        "opt_target": opt_target,
                        "k": k,
                        "query_dtype": query_dtype,
                        "prune_non_rng": is_pruned,
                        "rerank_factor": r_factor,
                        "recalls": anns_recalls,
                        "qps": anns_qps,
                        "search_params": anns_params,
                    }
                )

            # Explicitly free loaded graph and adapter memory between test iterations
            # del adapter
            # import gc
            # gc.collect()

        # Export interactive HTML plot
        html_output_path = deg_dir / f"{dataset_key}_anns_benchmark.html"
        export_interactive_html(
            dataset_name=meta["name"],
            instruction_set=instruction_set,
            search_k=actual_k,
            results_series=results_series,
            html_path=html_output_path,
        )
        if not args.no_show:
            import webbrowser

            webbrowser.open(html_output_path.as_uri())
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
