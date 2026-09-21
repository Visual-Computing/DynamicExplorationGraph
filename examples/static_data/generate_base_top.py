#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm

import deglib
from dataset import (
    load_dataset,
    get_default_cache_dir,
    resolve_dataset_key,
    ivecs_write,
    ivecs_read,
    DATASET_METADATA,
)
from presets import get_preset


def find_default_graph(dataset_key: str, dataset_dir: Path) -> Optional[Path]:
    """
    Search for the most suitable existing graph (.deg) for the given dataset.
    Prioritizes optimized/sorted graphs, then mainBranch, then any .deg file.
    """
    # 1. Look in <dataset_dir>/deg/
    deg_dir = dataset_dir / "deg"
    if deg_dir.is_dir():
        # Prefer FlasRamSorted, FlasSorted, RamSorted, or standard build
        for pattern in ["*_FlasRamSorted.deg", "*_FlasSorted.deg", "*_RamSorted.deg", "*LowLID.deg", "*HighLID.deg"]:
            matches = list(deg_dir.glob(pattern))
            if matches:
                return matches[0]
        # Any .deg directly in deg/
        deg_files = [f for f in deg_dir.glob("*.deg") if not f.name.startswith("Eval_")]
        if deg_files:
            return deg_files[0]

    # 2. Look in <dataset_dir>/deg_mainBranch/
    main_dir = dataset_dir / "deg_mainBranch"
    if main_dir.is_dir():
        matches = list(main_dir.glob("*.deg"))
        if matches:
            return matches[0]

    # 3. Recursive search in dataset_dir (excluding dynamic/eval variations if possible)
    all_deg = list(dataset_dir.rglob("*.deg"))
    if all_deg:
        # Prefer non-dynamic, non-eval
        candidates = [f for f in all_deg if "dynamic" not in str(f).lower() and not f.name.startswith("Eval_")]
        if candidates:
            return candidates[0]
        return all_deg[0]

    return None


def tune_explore_eps(
    graph,
    explore_entry: np.ndarray,
    explore_gt: np.ndarray,
    k: int,
    target_recall: float,
    eps_candidates: List[float],
    threads: int = 1,
) -> Tuple[float, float]:
    """
    Evaluates different eps values on explore entries to find the smallest eps
    that achieves target_recall for top-k.
    """
    entry_labels = np.ascontiguousarray(explore_entry, dtype=np.uint32)
    if entry_labels.ndim == 2:
        entry_labels = entry_labels[:, 0]

    n_queries = len(entry_labels)
    eval_gt = explore_gt[:, :k]

    print(f"\n--- Tuning Search-Eps on {n_queries} Explore Queries (Target Recall@{k}: {target_recall:.3f}) ---")

    best_eps = eps_candidates[-1]
    best_recall = 0.0

    for eps in eps_candidates:
        t0 = time.perf_counter()
        indices_batch = graph.explore(
            entry_labels,
            k=k,
            eps=eps,
            include_entry=True,
            threads=threads,
            return_distances=False,
        )
        elapsed = time.perf_counter() - t0
        qps = n_queries / max(elapsed, 1e-9)

        hits = 0
        total_items = n_queries * k
        for i in range(n_queries):
            gt_set = set(eval_gt[i])
            res_set = set(indices_batch[i])
            hits += len(gt_set.intersection(res_set))

        recall = hits / total_items
        print(f"  eps: {eps:6.3f} | Recall@{k}: {recall:7.4f} | QPS: {qps:8.1f} ({elapsed*1000/n_queries:6.2f} ms/query)")

        if recall >= target_recall:
            best_eps = eps
            best_recall = recall
            print(f"--> Target recall reached with eps = {best_eps:.3f} (Recall: {best_recall:.4f})")
            return best_eps, best_recall

        if recall > best_recall:
            best_recall = recall
            best_eps = eps

    print(f"--> Target recall not fully reached in sweep. Best eps: {best_eps:.3f} (Recall: {best_recall:.4f})")
    return best_eps, best_recall


def generate_base_top(
    dataset_key: str,
    cache_dir: Path,
    k: int = 100,
    target_recall: float = 0.99,
    graph_path_arg: Optional[str] = None,
    output_path_arg: Optional[str] = None,
    chunk_size: int = 50000,
    threads: int = 0,
    eps_override: Optional[float] = None,
):
    resolved_key = resolve_dataset_key(dataset_key)
    if resolved_key not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASET_METADATA.keys())}")

    print(f"\n=======================================================")
    print(f"Processing Dataset: {DATASET_METADATA[resolved_key]['name']} ({resolved_key})")
    print(f"=======================================================")

    # 1. Load dataset metadata & exploration GT
    base_vecs, query_vecs, gt_vecs, explore_entry, explore_gt, meta = load_dataset(resolved_key, cache_dir)
    n_base = len(base_vecs)
    preset = get_preset(resolved_key)

    # 2. Determine graph path
    if graph_path_arg:
        graph_path = Path(graph_path_arg)
        if not graph_path.is_file():
            raise FileNotFoundError(f"Specified graph file does not exist: {graph_path}")
    else:
        dataset_folder = cache_dir / meta["folder"]
        graph_path = find_default_graph(resolved_key, dataset_folder)
        if not graph_path:
            raise FileNotFoundError(f"Could not find any existing .deg graph for {resolved_key} in {dataset_folder}")

    print(f"Using Graph: {graph_path}")
    t_load = time.perf_counter()
    graph = deglib.load_readonly_graph(str(graph_path))
    print(f"Graph loaded in {time.perf_counter() - t_load:.2f}s | Vertices: {graph.size()}")

    if graph.size() < n_base:
        print(f"Warning: Graph size ({graph.size()}) is smaller than base vector count ({n_base})!")

    # 3. Determine search threads
    if threads <= 0:
        import os
        threads = os.cpu_count() or 4
    print(f"Using {threads} threads for search and exploration.")

    # 4. Tune or override search eps
    if eps_override is not None:
        chosen_eps = eps_override
        print(f"\nUsing manual eps override: {chosen_eps}")
    else:
        eps_candidates = sorted(preset.get("search_eps_list", [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]))
        chosen_eps, tuned_recall = tune_explore_eps(
            graph=graph,
            explore_entry=explore_entry,
            explore_gt=explore_gt,
            k=k,
            target_recall=target_recall,
            eps_candidates=eps_candidates,
            threads=threads,
        )

    # 5. Determine output file path
    if output_path_arg:
        out_file = Path(output_path_arg)
    else:
        files_dir = cache_dir / meta["folder"]
        if (files_dir / meta["folder"]).is_dir():
            files_dir = files_dir / meta["folder"]
        out_file = files_dir / f"{meta['folder']}_base_top{k}.ivecs"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nTarget output file: {out_file}")

    # 6. Execute Batch Exploration for all 0 .. n_base - 1
    total_chunks = (n_base + chunk_size - 1) // chunk_size
    print(f"\nStarting Exploration for all {n_base} elements (k={k}, eps={chosen_eps:.3f}, {total_chunks} chunks)...")

    start_explore_time = time.perf_counter()
    
    # We will write directly in chunks into an output file in ivecs format
    # In ivecs: each vector is preceded by int32(k), followed by k * int32(id)
    k_int32 = np.int32(k)

    with open(out_file, "wb") as f_out:
        with tqdm(total=n_base, desc="Exploration Progress", unit="vec") as pbar:
            for start_idx in range(0, n_base, chunk_size):
                end_idx = min(start_idx + chunk_size, n_base)
                batch_labels = np.arange(start_idx, end_idx, dtype=np.uint32)

                indices_chunk = graph.explore(
                    batch_labels,
                    k=k,
                    eps=chosen_eps,
                    include_entry=True,
                    threads=threads,
                    return_distances=False,
                )

                # Format as ivecs: [k, id_0, id_1, ..., id_k-1] per row
                n_chunk = len(batch_labels)
                k_col = np.full((n_chunk, 1), k_int32, dtype=np.int32)
                ivecs_chunk = np.hstack([k_col, indices_chunk.astype(np.int32)])
                f_out.write(ivecs_chunk.tobytes())

                pbar.update(n_chunk)

    total_duration = time.perf_counter() - start_explore_time
    qps = n_base / max(total_duration, 1e-9)
    file_size_mb = out_file.stat().st_size / (1024 * 1024)

    print(f"\nDone! Explored {n_base} vectors in {total_duration:.2f}s ({qps:.1f} vectors/s).")
    print(f"Saved: {out_file} ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal search eps on explore groundtruth and generate base_topK.ivecs via graph exploration."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="audio",
        help="Dataset name (e.g. audio, enron, deep1m, glove, sift1m, or 'all').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of nearest neighbors to retrieve per base entry (default: 100).",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.99,
        help="Target Recall@k on explore groundtruth for selecting optimal eps (default: 0.99).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Optional manual search eps override (bypasses auto-tuning).",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Optional explicit path to the .deg graph file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional explicit path for the output .ivecs file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Batch chunk size for exploration and streaming file write (default: 50000).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of threads to use (default: 0 = all CPU cores).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Dataset root/cache directory (defaults to DEG_CACHE_DIR or ~/.cache/deg_datasets).",
    )

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else get_default_cache_dir()

    if args.dataset.lower() == "all":
        datasets_to_process = ["audio", "enron", "deep1m", "glove"]
    else:
        datasets_to_process = [args.dataset]

    for d in datasets_to_process:
        try:
            generate_base_top(
                dataset_key=d,
                cache_dir=cache_dir,
                k=args.k,
                target_recall=args.target_recall,
                graph_path_arg=args.graph_path,
                output_path_arg=args.out,
                chunk_size=args.chunk_size,
                threads=args.threads,
                eps_override=args.eps,
            )
        except Exception as e:
            print(f"\n[ERROR] Failed processing dataset '{d}': {e}", file=sys.stderr)
            if len(datasets_to_process) == 1:
                raise


if __name__ == "__main__":
    main()
