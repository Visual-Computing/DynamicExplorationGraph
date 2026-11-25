import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

import deglib
import numpy as np
from dataset import (
    VIBE_DATASETS,
    ensure_dataset,
    get_default_cache_dir,
    load_vibe_dataset,
    resolve_dataset_key,
)
from module import DEG, QG
from presets import get_config_grid_presets


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


def get_physical_core_ids() -> list[int]:
    """Returns the primary logical processor ID for each physical CPU core (excluding SMT/Hyper-Threading siblings)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class SYSTEM_LOGICAL_PROCESSOR_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("ProcessorMask", ctypes.c_size_t),
                    ("Relationship", ctypes.c_int),
                    ("Reserved", ctypes.c_ulonglong * 2),
                ]

            buf_len = wintypes.DWORD(0)
            ctypes.windll.kernel32.GetLogicalProcessorInformation(None, ctypes.byref(buf_len))
            if buf_len.value > 0:
                count = buf_len.value // ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)
                buf = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION * count)()
                if ctypes.windll.kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(buf_len)):
                    cores = sorted(
                        (info.ProcessorMask & -info.ProcessorMask).bit_length() - 1
                        for info in buf
                        if info.Relationship == 0 and info.ProcessorMask
                    )
                    if cores:
                        return cores
        except Exception:
            pass
    else:
        try:
            topo_glob = Path("/sys/devices/system/cpu").glob("cpu[0-9]*/topology/thread_siblings_list")
            first_siblings: set[int] = set()
            for p in topo_glob:
                text = p.read_text().strip()
                first_id = int(text.split(",")[0].split("-")[0])
                first_siblings.add(first_id)
            if first_siblings:
                return sorted(first_siblings)
        except Exception:
            pass

    try:
        import psutil

        logical = psutil.cpu_count(logical=True) or 1
        physical = psutil.cpu_count(logical=False) or logical
        if logical == 2 * physical:
            return list(range(0, logical, 2))
        return list(range(physical))
    except Exception:
        total = os.cpu_count() or 2
        return list(range(0, total, 2)) if total > 1 else [0]


def get_performance_core_ids() -> list[int]:
    """Returns physical core IDs with the highest efficiency/performance class (e.g. P-cores / Zen 5 cores)."""
    physical_cores = get_physical_core_ids()
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class GROUP_AFFINITY(ctypes.Structure):
                _fields_ = [
                    ("Mask", ctypes.c_size_t),
                    ("Group", wintypes.WORD),
                    ("Reserved", wintypes.WORD * 3),
                ]

            class PROCESSOR_RELATIONSHIP(ctypes.Structure):
                _fields_ = [
                    ("Flags", ctypes.c_byte),
                    ("EfficiencyClass", ctypes.c_byte),
                    ("Reserved", ctypes.c_byte * 20),
                    ("GroupCount", wintypes.WORD),
                    ("GroupMask", GROUP_AFFINITY * 1),
                ]

            kernel32 = ctypes.windll.kernel32
            buf_len = wintypes.DWORD(0)
            kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(buf_len))
            if buf_len.value > 0:
                buf = (ctypes.c_byte * buf_len.value)()
                if kernel32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(buf_len)):
                    cores: list[tuple[int, int]] = []
                    off = 0
                    while off < buf_len.value:
                        rel = ctypes.c_int.from_buffer(buf, off).value
                        size = ctypes.c_uint32.from_buffer(buf, off + 4).value
                        if rel == 0:
                            pr = PROCESSOR_RELATIONSHIP.from_buffer(buf, off + 8)
                            mask = pr.GroupMask[0].Mask
                            if mask:
                                first_cpu = (mask & -mask).bit_length() - 1
                                cores.append((first_cpu, int(pr.EfficiencyClass)))
                        off += size
                    if cores:
                        max_ec = max(ec for _, ec in cores)
                        perf = sorted(cpu for cpu, ec in cores if ec == max_ec)
                        if perf:
                            return perf
        except Exception:
            pass
    else:
        try:
            caps: list[tuple[int, int]] = []
            for cpu_id in physical_cores:
                cap_path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpu_capacity")
                if cap_path.exists():
                    caps.append((cpu_id, int(cap_path.read_text().strip())))
            if len(caps) == len(physical_cores):
                max_cap = max(cap for _, cap in caps)
                perf = sorted(cpu for cpu, cap in caps if cap == max_cap)
                if perf:
                    return perf
        except Exception:
            pass
    return physical_cores


def set_cpu_affinity(
    cpu_ids: list[int] | None = None,
    label: str = "physical core(s) (no HT)",
) -> list[int] | None:
    """Pins the current process to `cpu_ids` (defaults to physical CPU cores)."""
    if cpu_ids is None:
        cpu_ids = get_physical_core_ids()
    try:
        import psutil

        p = psutil.Process()
        p.cpu_affinity(cpu_ids)
        current = p.cpu_affinity()
        print(f"Process CPU affinity pinned to {label}: {current}")
        return current
    except Exception as e:
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, cpu_ids)
                current = list(os.sched_getaffinity(0))
                print(f"Process CPU affinity pinned to {label}: {current}")
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
    base_vecs: np.ndarray, metric: deglib.distances.Metric, sample_size: int = 10
) -> float:
    n_base = len(base_vecs)
    query_count = min(sample_size, n_base)
    rng = np.random.default_rng(7)
    sample_indices = rng.choice(n_base, size=query_count, replace=False)

    print("\n--- Computing Linear Search Baseline ---")
    print(
        f"Computing linear search baseline with {query_count} random queries on {n_base:,} base vectors using C++ {metric.name} ({metric.get_instruction().name})..."
    )

    start_time = time.perf_counter()
    min_dist = float("inf")

    for idx in sample_indices:
        dists = metric.compute_distances(base_vecs[idx], base_vecs)
        d_min = float(np.min(dists))
        min_dist = min(min_dist, d_min)

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
        help="Directory to store datasets and built graphs (default: ~/.cache/deg_datasets, env: DEG_DATA_PATH)",
    )
    physical_cores = get_physical_core_ids()
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
        help=f"Pin benchmark process to specific CPU core ID(s) (default: physical cores without HT {physical_cores}).",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default=None,
        choices=["deg", "deg-qg", "qg", "deg-evp", "evp", "all"],
        help="Select specific algorithm to benchmark ('deg', 'deg-qg' / 'qg', 'deg-evp' / 'evp', or 'all'). Default: run all enabled in config.yml.",
    )
    parser.add_argument(
        "--query-dtype",
        type=str,
        default=None,
        choices=["float32", "int8"],
        help="Override feature storage and search vector dtype ('float32', 'int8'). Defaults to config.yml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Benchmark config YAML (default: examples/vibe/config.yml).",
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        help="Do not abort benchmark iterations when query time exceeds the linear baseline.",
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

        # Pin CPU core affinity (defaults to all physical cores without Hyper-Threading)
        physical_cores = get_physical_core_ids()
        perf_cores = get_performance_core_ids()
        set_cpu_affinity(args.cpu_affinity if args.cpu_affinity is not None else physical_cores)

        # Load dataset
        base_vecs, query_vecs, gt_vecs, gt_distances, meta = load_vibe_dataset(dataset_key, cache_dir)
        dims = base_vecs.shape[1]
        metric_str = meta["metric"]
        is_l2 = metric_str.lower().strip() in ("euclidean", "l2", "fp32_l2")
        metric_enum = (
            (deglib.Metric.FP16_L2 if is_l2 else deglib.Metric.FP16_InnerProduct)
            if base_vecs.dtype == np.float16
            else (deglib.Metric.FP32_L2 if is_l2 else deglib.Metric.FP32_InnerProduct)
        )
        instruction_set = metric_enum.get_instruction().name

        print("\n--- System & Distance Config ---")
        print(f"Metric: {metric_str} ({metric_enum.name}), Dimensions: {dims}")
        print(f"Vector Space Type: Metric ({instruction_set})")
        int8_metric_enum = deglib.Metric.Int8_L2 if is_l2 else deglib.Metric.Int8_InnerProduct
        int8_instruction_set = int8_metric_enum.get_instruction().name
        print(f"Int8 Feature Space: {int8_metric_enum.get_data_size(dims)} bytes ({int8_instruction_set})")
        print(f"Query Dtype: {args.query_dtype.upper() if args.query_dtype else 'FROM CONFIG.YML'}")
        print(
            f"Hardware AVX: {deglib.cpu.has_avx2()}, AVX-512: {deglib.cpu.has_avx512()}, AVX2-VNNI: {deglib.cpu.has_avx2_vnni()}, AVX-512-VNNI: {deglib.cpu.has_avx512_vnni()}"
        )

        # Linear scan baseline
        linear_baseline_us = compute_linear_search_baseline(base_vecs, metric_enum)

        # Filter algorithm if requested (e.g. --algorithm deg or --algorithm deg-qg)
        alg_filter = args.algorithm
        if alg_filter in ("qg", "deg-qg"):
            alg_filter = "deg-qg"
        elif alg_filter in ("evp", "deg-evp"):
            alg_filter = "deg-evp"
        elif alg_filter == "all":
            alg_filter = None

        configs_to_run = get_config_grid_presets(dataset_key, algorithm_name=alg_filter, config_path=args.config)

        actual_k = min(cfg_k_target := 100, gt_vecs.shape[1])
        results_series = []

        print("\n=======================================================")
        print(f"Running VIBE ANNS Benchmark on {dataset_key} ({len(configs_to_run)} graph config(s))")
        print("=======================================================")

        for cfg_idx, cfg in enumerate(configs_to_run):
            cfg_args = cfg["args"]
            cfg_query_args = cfg["query_args"]
            k = cfg_args["k"]
            opt_target = cfg_args["opt_target"]
            constructor_name = cfg.get("constructor", "DEG")
            alg_name = cfg.get("alg_name", "deg")

            if constructor_name == "QG":
                if not cfg_args.get("quants"):
                    raise ValueError(f"Missing args.quants for algorithm '{alg_name}'")
                adapter = QG(
                    metric=metric_str,
                    **cfg_args,
                )
            else:
                adapter = DEG(
                    metric=metric_str,
                    **cfg_args,
                )

            print(
                f"\n--- [{cfg_idx + 1}/{len(configs_to_run)}] Processing Algorithm '{alg_name}' ({constructor_name}) ---"
            )
            if args.cpu_affinity is None and cfg_idx > 0 and perf_cores != physical_cores:
                set_cpu_affinity(physical_cores, "physical core(s) (no HT)")
            adapter.fit(base_vecs, cache_dir=cache_dir)
            if args.cpu_affinity is None and perf_cores != physical_cores:
                set_cpu_affinity(perf_cores, "performance core(s) (max EfficiencyClass)")
            # Allow CPU thermals and worker threads to settle after graph build/optimization before query timing.
            time.sleep(3.0)

            # Determine query evaluation parameter grid
            quants_to_eval = (cfg_query_args.get("quants") or list(adapter.quant_metrics.keys())) if constructor_name == "QG" else [None]
            factors_to_eval = cfg_query_args.get("rerank_size_factor", [1.0])
            raw_param_list = cfg_query_args.get("eps_or_ef", [])

            # Separate and sort ef and eps parameter lists independently
            ef_values = sorted([v for v in raw_param_list if v >= 1.0])
            eps_values = sorted([v for v in raw_param_list if v < 1.0])
            param_groups = []
            if ef_values:
                param_groups.append(("ef", ef_values))
            if eps_values:
                param_groups.append(("eps", eps_values))

            for quant_name in quants_to_eval:
                if constructor_name == "QG":
                    quant_metric_obj = adapter.quant_metrics[quant_name]
                    query_dtype, query_metric = quant_metric_obj.name.split("_", 1)
                    quant_type = query_dtype
                    active_searcher = adapter.searchers[quant_name]
                else:
                    query_metric = metric_str
                    query_dtype = "float32"
                    quant_type = "fp32"
                    active_searcher = adapter.searcher

                feature_instruction = f", Features={active_searcher.graph.get_instruction().name}"
                print(
                    f"Index Config: K={k}, Opt={opt_target}, "
                    f"QueryMetric={query_metric}, "
                    f"QueryType={query_dtype}{feature_instruction}"
                )

                # Warm up thread-local workspace, CPU boost, and caches before timing queries.
                if param_groups and len(query_vecs) > 0:
                    warmup_param = param_groups[0][1][0]
                    warmup_factor = factors_to_eval[0] if factors_to_eval else 1.0
                    if constructor_name == "QG":
                        adapter.set_query_arguments(
                            eps_or_ef=warmup_param, rerank_size_factor=warmup_factor, quants=quant_name
                        )
                    else:
                        adapter.set_query_arguments(eps_or_ef=warmup_param)
                    for q_idx in range(min(200, len(query_vecs))):
                        adapter.query(query_vecs[q_idx], actual_k)

                for r_factor in factors_to_eval:
                    rerank_label = f", Rerank={r_factor:.2f}x" if r_factor > 1.0 else ""
                    formatted_params = [str(int(round(v))) for v in ef_values] + [f"{v:g}" for v in eps_values]
                    param_str = ", ".join(formatted_params)

                    print(f"\nEvaluating top-{actual_k} search (rerank_factor={r_factor:.2f}) for eps_or_ef: {param_str}")

                    anns_recalls = []
                    anns_qps = []
                    anns_params = []
                    n_queries = len(query_vecs)

                    for param_type, group_vals in param_groups:
                        for val in group_vals:
                            if constructor_name == "QG":
                                adapter.set_query_arguments(eps_or_ef=val, rerank_size_factor=r_factor, quants=quant_name)
                            else:
                                adapter.set_query_arguments(eps_or_ef=val)

                            results_indices = []
                            total_query_sec = 0.0
                            for q_idx in range(n_queries):
                                t_start = time.perf_counter()
                                candidates = adapter.query(query_vecs[q_idx], actual_k)
                                total_query_sec += time.perf_counter() - t_start
                                results_indices.append(np.array(candidates, dtype=np.int64, copy=True))

                            elapsed_sec = total_query_sec

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
                                    ret_vecs = base_vecs[results_indices[i]]
                                    pred_dists = dist_fn(query_vecs[i], ret_vecs)
                                    hits += int((pred_dists <= t).sum())
                            else:
                                for i in range(n_queries):
                                    gt_set = set(gt_vecs[i, :actual_k])
                                    ret_set = set(results_indices[i])
                                    hits += len(gt_set.intersection(ret_set))

                            recall = hits / max(total_returned, 1)
                            anns_recalls.append(recall)
                            anns_qps.append(qps)
                            is_ef = param_type == "ef"
                            param_tag = f"ef={int(round(val))}" if is_ef else f"eps={val:g}"
                            anns_params.append(param_tag)

                            lbl = f"ef  {int(round(val)):4d}" if is_ef else f"eps {val:6.3f}"
                            print(
                                f"  {lbl} \trecall@{actual_k}: {recall:.5f} \t{time_us_per_query:6d} us/query \t{qps:9.1f} QPS \tsearch time: {int(search_time_us / 1000):6d}ms"
                            )

                            if not args.no_stop and linear_baseline_us > 0 and time_us_per_query > linear_baseline_us and qps <= 1000:
                                print(
                                    f"  ABORTED {param_type.upper()} ({time_us_per_query}us/query > {int(linear_baseline_us)}us baseline, {qps:.1f} QPS <= 1000)"
                                )
                                break

                            if recall > 0.999:
                                print(
                                    f"  Reached recall > 0.999 for {param_type.upper()}, stopping further {param_type.upper()} iterations."
                                )
                                break

                    results_series.append(
                        {
                            "label": (
                                f"{alg_name.upper()} ({opt_target}, K={k}, Quant={quant_type}, "
                                f"QueryMetric={query_metric}{rerank_label})"
                            ),
                            "algo": alg_name,
                            "opt_target": opt_target,
                            "k": k,
                            "quant": quant_type,
                            "query_metric": query_metric,
                            "query_dtype": query_dtype,
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
