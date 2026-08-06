import argparse
import multiprocessing
import time
from enum import Enum
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import deglib
from dataset_utils import (
    load_dataset_for_dynamic,
    get_default_cache_dir,
    resolve_dataset_key,
    DATASET_METADATA,
)
from presets import get_preset
from graph_analysis import analyze_graph


class DataStreamType(Enum):
    """
    Mirrors the C++ bench_dynamic_data.cpp DataStreamType enum.

    - AddHalf:                       Insert only the first half of base vectors.
    - AddHalfRemoveAndAddOneAtATime: Interleaved insertion and deletion operations.
    - AddAllRemoveHalf:              Insert all vectors, then remove the second half.
    """
    AddHalf = "AddHalf"
    AddHalfRemoveAndAddOneAtATime = "AddHalfRemoveAndAddOneAtATime"
    AddAllRemoveHalf = "AddAllRemoveHalf"


# Stream types tested in bench_dynamic_data.cpp (in order)
STREAM_TYPES = [
    DataStreamType.AddHalf,
    DataStreamType.AddHalfRemoveAndAddOneAtATime,
    DataStreamType.AddAllRemoveHalf,
]


# ---------------------------------------------------------------------------
# Plotting – single combined window updated live via Queue
# ---------------------------------------------------------------------------

_STREAM_COLORS = ["tab:blue", "tab:orange", "tab:green"]
_STREAM_MARKERS = ["o", "s", "^"]

# Give up opening the plot window if no benchmark data arrived within this many seconds.
_FIRST_DATA_TIMEOUT = 3600.0

def _render_combined_anns_plot(
    queue: multiprocessing.Queue,
    dataset_name: str,
    instruction_set: str,
    search_k: int,
):
    """Plot subprocess: waits for the first result, then opens one figure it updates live.

    The window is only opened once the first curve is ready, so the user does not stare at
    an empty figure during the (possibly slow) first graph build.
    """
    import matplotlib.pyplot as plt

    # On Windows, Ctrl+C in the console sends CTRL_C_EVENT to *every* process attached to
    # the console, including this GUI-only subprocess. A KeyboardInterrupt in the Tk event
    # loop (or while blocked on the queue) must not crash the plot or dump a traceback.
    try:
        first = queue.get(timeout=_FIRST_DATA_TIMEOUT)
    except KeyboardInterrupt:
        return  # user aborted before any results arrived
    except Exception:
        print(f"\nNo benchmark data within {_FIRST_DATA_TIMEOUT:.0f}s; not opening plot window.", flush=True)
        return

    if first is None:
        return  # all data already sent - nothing to plot

    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlabel(f"Recall@{search_k}")
        ax.set_ylabel("Queries Per Second (QPS)")
        ax.set_title(f"DEG Dynamic ANNS Benchmark\n{dataset_name} ({instruction_set})")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        def draw_curve(stream_idx: int, stream_type_str: str, recalls, qps):
            color = _STREAM_COLORS[stream_idx % len(_STREAM_COLORS)]
            marker = _STREAM_MARKERS[stream_idx % len(_STREAM_MARKERS)]
            ax.plot(recalls, qps, marker=marker, linewidth=2, color=color, label=stream_type_str)
            ax.legend()

        def draw_msg(stream_idx: int, msg):
            stream_type_str, recalls, qps = msg
            draw_curve(stream_idx, stream_type_str, recalls, qps)
            fig.canvas.draw_idle()

        draw_msg(0, first)  # draw the first curve right away
        stream_idx = 1
        done = False

        while not done and plt.fignum_exists(fig.number):
            # Drain all pending messages
            while True:
                try:
                    msg = queue.get_nowait()
                except Exception:
                    break

                if msg is None:  # sentinel – all data has been sent
                    done = True
                    break

                draw_msg(stream_idx, msg)
                stream_idx += 1

            plt.pause(0.2)  # keep GUI event loop alive while waiting for next curve

        # Keep window open until user closes it
        if plt.fignum_exists(fig.number):
            plt.ioff()
            plt.show(block=True)
    except KeyboardInterrupt:
        # Ctrl+C while the plot is up: exit gracefully, no traceback.
        pass
    finally:
        plt.close("all")




# ---------------------------------------------------------------------------
# Linear search baseline
# ---------------------------------------------------------------------------

def compute_linear_search_baseline(base_vecs: np.ndarray, float_space: deglib.distances.FloatSpace, sample_size: int = 100) -> float:
    n_base = len(base_vecs)
    query_count = min(sample_size, n_base)
    rng = np.random.default_rng(7)
    sample_indices = rng.choice(n_base, size=query_count, replace=False)

    print(f"\n--- Computing Linear Search Baseline ---")
    print(f"Computing linear search baseline with {query_count} random queries on {n_base} base vectors "
          f"using C++ {float_space.metric().name} ({float_space.float_space_cpp.get_instruction()})...")

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
    print(f"Linear search baseline: {int(time_per_query_us)}us per query "
          f"(total: {int(total_time_us / 1000)}ms for {query_count} queries) with min distance {min_dist:.6f}")
    return linear_baseline_us


# ---------------------------------------------------------------------------
# Graph filename (mirrors C++ bench_dynamic_data.cpp naming)
# {dims}D_K{k}_{stream_type_str}.deg  in  <graph_dir>/
# ---------------------------------------------------------------------------

def build_dynamic_graph_filename(graph_dir: Path, dims: int, k: int, stream_type: DataStreamType) -> Path:
    """
    Returns the graph file path matching C++ naming:
        {dims}D_K{k}_{stream_type}.deg
    """
    graph_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dims}D_K{k}_{stream_type.value}.deg"
    return graph_dir / filename


# ---------------------------------------------------------------------------
# Graph building – simulation of the three DataStreamTypes
# ---------------------------------------------------------------------------

def build_dynamic_graph(
    base_vecs: np.ndarray,
    stream_type: DataStreamType,
    graph_path: Path | None,
    preset: dict,
    instruction_enum: deglib.distances.InstructionSet,
    build_threads: int,
):
    """
    Builds a DEG graph using the given DataStreamType strategy and saves it to graph_path.

    Mirrors the C++ create_graph() call with the respective DataStreamType in build.h.
    """
    n = len(base_vecs)
    half = n // 2
    dims = base_vecs.shape[1]
    metric = preset["metric"]
    k = preset["k"]

    print(f"\n=== Building Graph: {stream_type.value} ===")
    if graph_path:
        print(f"Output: {graph_path}")
    else:
        print("Output: (in RAM, no file)")

    graph_mut = deglib.builder.SizeBoundedGraph.create_empty(
        capacity=n,
        dims=dims,
        edges_per_vertex=k,
        metric=metric,
        instruction=instruction_enum,
    )
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph_mut,
        rng=deglib.Mt19937(7),
        optimization_target=deglib.builder.OptimizationTarget.StreamingData,
        extend_k=preset.get("extend_k", k),
        extend_eps=preset["build_eps"],
        improve_k=preset.get("improve_k", 0),
        improve_eps=preset.get("improve_eps", 0.0),
    )
    builder.set_batch_size(10, 10)
    if build_threads > 1:
        builder.set_thread_count(build_threads)

    all_labels = np.arange(n, dtype=np.uint32)

    if stream_type == DataStreamType.AddHalf:
        # Insert only the first half of base vectors.
        # Mirrors: base_size /= 2; for i in [0..base_size): addEntry(i)
        builder.add_entry(all_labels[:half], base_vecs[:half])

    elif stream_type == DataStreamType.AddAllRemoveHalf:
        # Insert all, then remove the second half.
        # Mirrors: for i in [0..n): addEntry(i); for i in [n/2..n): removeEntry(i)
        builder.add_entry(all_labels, base_vecs)
        for i in range(half, n):
            builder.remove_entry(int(all_labels[i]))

    elif stream_type == DataStreamType.AddHalfRemoveAndAddOneAtATime:
        # Interleaved add/remove matching C++ build.h.
        quarter = n // 4

        # 1st loop: add base_size_quarter pairs from [0, quarter) and [half, half + quarter)
        for i in range(quarter):
            builder.add_entry(int(all_labels[i]), base_vecs[i : i + 1])
            builder.add_entry(int(all_labels[half + i]), base_vecs[half + i : half + i + 1])

        # 2nd loop: add base_size_quarter pairs from [quarter, half) and remove same number
        for i in range(quarter):
            builder.add_entry(int(all_labels[quarter + i]), base_vecs[quarter + i : quarter + i + 1])
            builder.add_entry(int(all_labels[half + quarter + i]), base_vecs[half + quarter + i : half + quarter + i + 1])
            builder.remove_entry(int(all_labels[half + (i * 2) + 0]))
            builder.remove_entry(int(all_labels[half + (i * 2) + 1]))

        # Remainder wave: If base_size is not divisible by 4, add remaining entries to reach exactly half vertices
        remainder = half - (quarter * 2)
        for i in range(remainder):
            rem_idx = quarter * 2 + i
            builder.add_entry(int(all_labels[rem_idx]), base_vecs[rem_idx : rem_idx + 1])

    build_start = time.perf_counter()
    builder.build(callback="progress")
    build_time = time.perf_counter() - build_start
    print(f"Graph built in {build_time:.2f} seconds ({graph_mut.size()} vertices).")

    if graph_path:
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_mut.save_graph(str(graph_path))
        print(f"Graph saved to: {graph_path}")
        return deglib.graph.load_readonly_graph(str(graph_path))
    else:
        return deglib.graph.ReadOnlyGraph.from_graph(graph_mut)


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def run_dynamic_benchmark(
    dataset_key: str,
    cache_dir: Path,
    graph_dir: Path | None,
    instruction_set: str,
    build_threads: int,
    max_base_vecs: int | None,
    no_show: bool,
    force_rebuild: bool,
):
    resolved_key = resolve_dataset_key(dataset_key)
    preset = get_preset(resolved_key)

    # Load dataset (no exploration data needed)
    base_vecs, query_vecs, gt_vecs_full, gt_vecs_half, meta = load_dataset_for_dynamic(resolved_key, cache_dir)

    if max_base_vecs and max_base_vecs < len(base_vecs):
        print(f"Limiting base vectors to first {max_base_vecs} samples for fast run.")
        base_vecs = base_vecs[:max_base_vecs]

    dims = base_vecs.shape[1]
    metric = preset["metric"]

    # Map instruction_set string to deglib.distances.InstructionSet
    inst_lower = instruction_set.lower()
    if inst_lower == "scalar":
        instruction_enum = deglib.distances.InstructionSet.Scalar
    elif inst_lower == "avx2":
        instruction_enum = deglib.distances.InstructionSet.AVX2
    elif inst_lower == "avx512":
        instruction_enum = deglib.distances.InstructionSet.AVX512
    else:
        instruction_enum = deglib.distances.InstructionSet.Auto

    print(f"\n=== Benchmarking Dynamic Data Streams for Dataset: {meta['name']} ===")
    print(f"Selected Instruction Set: {instruction_set}")
    print(f"Build Threads: {build_threads}")
    print(f"Base vectors: {base_vecs.shape[0]} | Dimension: {dims}")
    print(f"Query vectors: {query_vecs.shape[0]} | Full-GT shape: {gt_vecs_full.shape} | Half-GT shape: {gt_vecs_half.shape}")

    feature_space = deglib.distances.FloatSpace.create(dims, metric, instruction_enum)
    actual_inst = str(feature_space.float_space_cpp.get_instruction()).split('.')[-1]
    print(f"Distance Metric: {metric.name} ({actual_inst})")

    linear_baseline_us = compute_linear_search_baseline(base_vecs, feature_space, sample_size=100)

    # Compute search_k once – it does not vary across stream types
    search_k = min(preset["anns_k"], gt_vecs_half.shape[1])

    # Start a single combined plot process that live-updates as each curve arrives
    plot_queue: multiprocessing.Queue | None = None
    plot_process: multiprocessing.Process | None = None
    if not no_show:
        plot_queue = multiprocessing.Queue()
        plot_process = multiprocessing.Process(
            target=_render_combined_anns_plot,
            args=(plot_queue, meta["name"], instruction_set, search_k),
        )
        plot_process.start()
        print("\nPlot window will open once the first results are ready.")

    # Iterate over all three stream types (mirrors C++ bench_dynamic_data.cpp)
    for stream_type in STREAM_TYPES:
        print(f"\n=== Testing DataStreamType: {stream_type.value} ===")

        # Resolve graph file path
        if graph_dir is not None:
            graph_path = build_dynamic_graph_filename(graph_dir, dims, preset["k"], stream_type)
        else:
            graph_path = None

        graph = None

        # Load from file if it exists and force_rebuild is not set
        if graph_path and graph_path.exists() and not force_rebuild:
            print(f"Loading existing graph: {graph_path}")
            graph = deglib.graph.load_readonly_graph(str(graph_path))
            print(f"Loaded dynamic graph with {graph.size()} vertices")
        else:
            if graph_path and force_rebuild:
                print(f"Force rebuild requested, rebuilding graph: {graph_path}")
            elif graph_path:
                print(f"Graph not found, building: {graph_path}")
            else:
                print("No --graph-dir specified, building graph in RAM only.")

            graph = build_dynamic_graph(
                base_vecs=base_vecs,
                stream_type=stream_type,
                graph_path=graph_path,
                preset=preset,
                instruction_enum=instruction_enum,
                build_threads=build_threads,
            )

        if graph is None:
            print(f"ERROR: Could not build or load graph for {stream_type.value}")
            continue

        print(f"Graph size: {graph.size()} vertices")

        # Graph analysis
        analyze_graph(graph)

        # ANNS Test using half ground truth
        # C++: use_half = (ds_type != DataStreamType::AddAll) -> always True for our 3 types
        search_k = min(preset["anns_k"], gt_vecs_half.shape[1])
        repeat = preset.get("anns_repeat", 1)
        eps_list = sorted(preset["search_eps_list"])

        print(f"\n--- ANNS Test (k={search_k}, using half-dataset GT) ---")
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
                gt_set = set(gt_vecs_half[i, :search_k])
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

        # Send this stream type's results to the combined plot process
        if plot_queue is not None and anns_recalls:
            plot_queue.put((stream_type.value, list(anns_recalls), list(anns_qps)))

        # Release graph memory before building the next one
        del graph

    # Signal the plot process that all data has been sent, then wait for the user to close it
    if plot_queue is not None:
        plot_queue.put(None)
        if plot_process is not None:
            plot_process.join()


def main():
    parser = argparse.ArgumentParser(description="DEG Benchmark Tool: bench_dynamic_data (Python)")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="sift1m",
        choices=["sift1m", "deep1m", "glove", "audio", "enron", "all"],
        help="Dataset name (e.g. sift1m, deep1m, glove, audio, enron, all) (default: sift1m)",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        default=None,
        help=(
            "Directory to save/load graph files. Graph filenames follow C++ naming: "
            "{dims}D_K{k}_{stream_type}.deg. If omitted, graphs are built in RAM only."
        ),
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the graphs even if graph files already exist",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="auto",
        choices=["auto", "avx512", "avx2", "scalar"],
        help="Select distance instruction set (default: auto)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads used for building the graph",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom directory to cache datasets (default: ~/.cache/deg_datasets)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive plot window",
    )
    parser.add_argument(
        "--max-base-vecs",
        type=int,
        default=None,
        help="Optional limit on base vectors for quick debugging/testing",
    )

    args = parser.parse_args()

    dataset_name = args.dataset or "sift1m"
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_default_cache_dir()
    graph_dir = Path(args.graph_dir) if args.graph_dir else None

    # Detect CPU Instruction Set
    if args.instruction != "auto":
        instruction_set = args.instruction.upper()
    elif deglib.avx512_usable():
        instruction_set = "AVX512"
    elif deglib.avx_usable():
        instruction_set = "AVX2"
    else:
        instruction_set = "Scalar"

    print("=== bench_dynamic_data (Python) ===")
    print(f"Data Cache Directory: {cache_dir.resolve()}")
    if graph_dir:
        print(f"Graph Directory: {graph_dir.resolve()}")
        print(f"Graph filename format: {{dims}}D_K{{k}}_{{stream_type}}.deg")

    if dataset_name.lower() == "all":
        datasets_to_run = ["sift1m", "deep1m", "glove-100", "audio", "enron"]
    else:
        datasets_to_run = [dataset_name]

    for ds in datasets_to_run:
        run_dynamic_benchmark(
            dataset_key=ds,
            cache_dir=cache_dir,
            graph_dir=graph_dir,
            instruction_set=instruction_set,
            build_threads=args.threads,
            max_base_vecs=args.max_base_vecs,
            no_show=args.no_show,
            force_rebuild=args.force_rebuild,
        )

    print("\nDynamic Benchmark Finished Successfully.")


if __name__ == "__main__":
    main()
