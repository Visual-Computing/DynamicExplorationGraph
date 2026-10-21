import os
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import deglib
from deglib.distances import Metric, FloatSpace


def ivecs_read(filename: str | Path, mmap: bool = False) -> np.ndarray:
    if mmap:
        with open(filename, "rb") as f:
            d = np.fromfile(f, dtype=np.int32, count=1)[0]
        mm = np.memmap(filename, dtype=np.int32, mode="r")
        return mm.reshape(-1, d + 1)[:, 1:]
    a = np.fromfile(filename, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(filename: str | Path) -> np.ndarray:
    return ivecs_read(filename).view(np.float32)


DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "sift1m": {
        "name": "SIFT1M",
        "url": "https://static.visual-computing.com/paper/DEG/sift.tar.gz",
        "archive": "sift.tar.gz",
        "folder": "sift",
        "metric": Metric.FP32_L2,
        "dim": 128,
        "base_count": 1000000,
        "base_file": "sift1m_base.fvecs",
        "query_file": "sift1m_query.fvecs",
        "gt_file": "sift1m_groundtruth_top100_nb1000000.ivecs",
        "explore_entry_file": "sift1m_explore_entry_vertex.ivecs",
        "explore_query_file": "sift1m_explore_query.fvecs",
        "explore_gt_file": "sift1m_explore_groundtruth_top1000.ivecs",
    },
    "audio": {
        "name": "Audio",
        "url": "https://static.visual-computing.com/paper/DEG/audio.tar.gz",
        "archive": "audio.tar.gz",
        "folder": "audio",
        "metric": Metric.FP32_L2,
        "dim": 192,
        "base_count": 53387,
        "base_file": "audio_base.fvecs",
        "query_file": "audio_query.fvecs",
        "gt_file": "audio_groundtruth_top100_nb53387.ivecs",
        "explore_entry_file": "audio_explore_entry_vertex.ivecs",
        "explore_query_file": "audio_explore_query.fvecs",
        "explore_gt_file": "audio_explore_groundtruth_top1000.ivecs",
    },
    "enron": {
        "name": "Enron",
        "url": "https://static.visual-computing.com/paper/DEG/enron.tar.gz",
        "archive": "enron.tar.gz",
        "folder": "enron",
        "metric": Metric.FP32_L2,
        "dim": 1369,
        "base_count": 94987,
        "base_file": "enron_base.fvecs",
        "query_file": "enron_query.fvecs",
        "gt_file": "enron_groundtruth_top100_nb94987.ivecs",
        "explore_entry_file": "enron_explore_entry_vertex.ivecs",
        "explore_query_file": "enron_explore_query.fvecs",
        "explore_gt_file": "enron_explore_groundtruth_top1000.ivecs",
    },
    "deep1m": {
        "name": "DEEP1M",
        "url": "https://static.visual-computing.com/paper/DEG/deep1m.tar.gz",
        "archive": "deep1m.tar.gz",
        "folder": "deep1m",
        "metric": Metric.FP32_L2,
        "dim": 96,
        "base_count": 1000000,
        "base_file": "deep1m_base.fvecs",
        "query_file": "deep1m_query.fvecs",
        "gt_file": "deep1m_groundtruth_top100_nb1000000.ivecs",
        "explore_entry_file": "deep1m_explore_entry_vertex.ivecs",
        "explore_query_file": "deep1m_explore_query.fvecs",
        "explore_gt_file": "deep1m_explore_groundtruth_top1000.ivecs",
    },
    "glove": {
        "name": "GloVe-100",
        "url": "https://static.visual-computing.com/paper/DEG/glove-100.tar.gz",
        "archive": "glove-100.tar.gz",
        "folder": "glove",
        "metric": Metric.FP32_L2,
        "dim": 100,
        "base_count": 1183514,
        "base_file": "glove_base.fvecs",
        "query_file": "glove_query.fvecs",
        "gt_file": "glove_groundtruth_top100_nb1183514.ivecs",
        "explore_entry_file": "glove_explore_entry_vertex.ivecs",
        "explore_query_file": "glove_explore_query.fvecs",
        "explore_gt_file": "glove_explore_groundtruth_top1000.ivecs",
    },
}


def resolve_dataset_key(key: str) -> str:
    return key.lower()


def get_default_cache_dir() -> Path:
    """Returns the default dataset cache directory (~/.cache/deg_datasets or DEG_CACHE_DIR)."""
    env_dir = os.environ.get("DEG_CACHE_DIR")
    if env_dir:
        path = Path(env_dir)
    else:
        path = Path.home() / ".cache" / "deg_datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest_path: Path):
    """Downloads a file from url to dest_path with progress indication."""
    print(f"Downloading {url} to {dest_path}...")
    temp_path = dest_path.with_suffix(".tmp")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB chunks

        with open(temp_path, "wb") as out_file:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)
                if total_size > 0:
                    percent = downloaded * 100 // total_size
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\rProgress: {percent}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    print()
    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)
    print(f"Download complete: {dest_path}")


def ivecs_write(filename: str | Path, data: np.ndarray):
    """Writes a 2D int32 numpy array to ivecs format."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    d = np.int32(data.shape[1])
    n = data.shape[0]
    d_col = np.full((n, 1), d, dtype=np.int32)
    with_d = np.hstack([d_col, data.astype(np.int32)])
    with_d.tofile(filename)


def fvecs_write(filename: str | Path, data: np.ndarray):
    """Writes a 2D float32 numpy array to fvecs format."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    d = np.int32(data.shape[1])
    n = data.shape[0]
    d_float = np.frombuffer(d.tobytes(), dtype=np.float32)[0]
    d_col = np.full((n, 1), d_float, dtype=np.float32)
    with_d = np.hstack([d_col, data.astype(np.float32)])
    with_d.tofile(filename)


def cleanup_legacy_gt(folder: Path):
    """Deletes legacy/broken groundtruth files extracted from old dataset archives."""
    legacy_files = [
        "sift_groundtruth.ivecs",
        "sift_explore_ground_truth.ivecs",
        "audio_groundtruth.ivecs",
        "audio_explore_ground_truth.ivecs",
        "enron_groundtruth.ivecs",
        "enron_explore_ground_truth.ivecs",
        "deep1m_groundtruth.ivecs",
        "deep1m_explore_ground_truth.ivecs",
        "glove-100_groundtruth.ivecs",
        "glove-100_explore_ground_truth.ivecs",
        "glove_groundtruth.ivecs",
        "glove_explore_ground_truth.ivecs",
    ]
    for filename in legacy_files:
        for p in folder.rglob(filename):
            try:
                print(f"Removing legacy groundtruth file: {p}")
                p.unlink()
            except Exception as e:
                print(f"Warning: Failed to remove legacy GT file {p}: {e}")


def ensure_dataset(dataset_key: str, cache_dir: Path) -> Path:
    """Ensures the dataset is downloaded and extracted in cache_dir, returning folder path."""
    key = resolve_dataset_key(dataset_key)
    if key not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASET_METADATA.keys())}")

    meta = DATASET_METADATA[key]
    archive_path = cache_dir / meta["archive"]

    # 1. Check if direct folder (e.g., D:\Data\DEG\sift1m or D:\Data\DEG\sift) exists
    extracted_folder = cache_dir / meta["folder"]
    if not extracted_folder.is_dir():
        # Check case-insensitive / fallback matches before triggering a download
        subdirs = [p for p in cache_dir.iterdir() if p.is_dir() and meta["folder"].lower() in p.name.lower()]
        if subdirs:
            extracted_folder = subdirs[0]

    # 2. If folder is still not found, check/download archive and extract
    if not extracted_folder.is_dir():
        if not archive_path.is_file():
            # Also check if archive exists inside a subfolder or cache_dir
            archive_matches = list(cache_dir.rglob(meta["archive"]))
            if archive_matches:
                archive_path = archive_matches[0]
            else:
                download_file(meta["url"], archive_path)

        print(f"Extracting {archive_path} into {cache_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)
        print("Extraction complete.")

        extracted_folder = cache_dir / meta["folder"]
        if not extracted_folder.is_dir():
            subdirs = [p for p in cache_dir.iterdir() if p.is_dir() and meta["folder"].lower() in p.name.lower()]
            if subdirs:
                extracted_folder = subdirs[0]
            else:
                extracted_folder = cache_dir

    return extracted_folder


def build_graph_filename(
    dataset_key: str,
    cache_dir: Path,
    dims: int,
    k: int,
    extend_k: int,
    extend_eps: float,
    optimization_target_str: str,
    metric_str: str,
) -> Path:
    """
    Builds the graph path matching C++ filename formatting:
    <dataset_dir>/deg/{dims}D_{metric}_K{k}_AddK{k_ext}Eps{eps_ext:.1f}_{lid}.deg
    """
    dataset_dir = ensure_dataset(dataset_key, cache_dir)
    deg_dir = dataset_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dims}D_{metric_str}_K{k}_AddK{extend_k}Eps{extend_eps:.1f}_{optimization_target_str}.deg"
    return deg_dir / filename


def compute_and_save_anns_gt(
    base_vecs: np.ndarray, query_vecs: np.ndarray, float_space: FloatSpace, k: int, out_path: Path
):
    n_queries = len(query_vecs)
    gt_matrix = np.zeros((n_queries, k), dtype=np.int32)
    print(f"Generating ANNS Ground Truth (Full Dataset, top-{k})...")
    start_t = time.perf_counter()

    for q in range(n_queries):
        dists = float_space.compute_distances(query_vecs[q], base_vecs)
        top_k = np.argpartition(dists, k)[:k]
        top_k = top_k[np.argsort(dists[top_k])]
        gt_matrix[q, :] = top_k
        if (q + 1) % 100 == 0 or (q + 1) == n_queries:
            elapsed_ms = int((time.perf_counter() - start_t) * 1000)
            print(
                f"\r  Ground truth progress: {q + 1}/{n_queries} ({100.0 * (q + 1) / n_queries:.1f}%) after {elapsed_ms}ms",
                end="",
                flush=True,
            )
    print()
    ivecs_write(out_path, gt_matrix)
    print(f"ANNS Ground Truth generated and saved to {out_path}")


def generate_explore_entry_and_query(
    base_vecs: np.ndarray, entry_path: Path, query_path: Path, sample_count: int = 10000
):
    print("Generating exploration entry vertices and explore queries...")
    half_base_size = len(base_vecs) // 2
    step = float(half_base_size) / sample_count
    entry_ids = np.zeros((sample_count, 1), dtype=np.int32)
    for i in range(sample_count):
        idx = int(i * step)
        if idx >= half_base_size:
            idx = half_base_size - 1
        entry_ids[i, 0] = idx

    ivecs_write(entry_path, entry_ids)

    explore_features = base_vecs[entry_ids[:, 0]]
    fvecs_write(query_path, explore_features)


def compute_and_save_explore_gt(
    base_vecs: np.ndarray, explore_query_vecs: np.ndarray, float_space: FloatSpace, k: int, out_path: Path
):
    n_queries = len(explore_query_vecs)
    gt_matrix = np.zeros((n_queries, k), dtype=np.int32)
    print(f"Generating Exploration Ground Truth (Full Dataset, top-{k})...")
    start_t = time.perf_counter()

    for q in range(n_queries):
        dists = float_space.compute_distances(explore_query_vecs[q], base_vecs)
        top_k = np.argpartition(dists, k)[:k]
        top_k = top_k[np.argsort(dists[top_k])]
        gt_matrix[q, :] = top_k
        if (q + 1) % 500 == 0 or (q + 1) == n_queries:
            elapsed_ms = int((time.perf_counter() - start_t) * 1000)
            print(
                f"\r  Exploration ground truth progress: {q + 1}/{n_queries} ({100.0 * (q + 1) / n_queries:.1f}%) after {elapsed_ms}ms",
                end="",
                flush=True,
            )
    print()
    ivecs_write(out_path, gt_matrix)
    print(f"Exploration Ground Truth generated and saved to {out_path}")


def load_dataset(
    dataset_key: str, cache_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any, Dict[str, Any]]:
    """
    Downloads/extracts dataset if needed and loads (base, query, groundtruth) numpy arrays.
    Returns (base_features, query_features, ground_truth, explore_entry, explore_gt, metadata).
    """
    key = resolve_dataset_key(dataset_key)
    meta = DATASET_METADATA[key]
    folder = ensure_dataset(key, cache_dir)

    cleanup_legacy_gt(folder)

    files_dir = folder / meta["folder"] if (folder / meta["folder"]).is_dir() else folder

    base_path = files_dir / meta["base_file"]
    query_path = files_dir / meta["query_file"]

    print(f"Loading base features from {base_path}...")
    base_vecs = fvecs_read(base_path)

    print(f"Loading query features from {query_path}...")
    query_vecs = fvecs_read(query_path)

    dims = base_vecs.shape[1]
    metric = meta["metric"]
    float_space = FloatSpace.create(dims, metric)

    # 1. ANNS Ground Truth
    gt_path = files_dir / meta["gt_file"]
    if not gt_path.is_file():
        print(f"ANNS Ground Truth not found at {gt_path}.")
        compute_and_save_anns_gt(base_vecs, query_vecs, float_space, 100, gt_path)

    print(f"Loading groundtruth indices from {gt_path}...")
    gt_vecs = ivecs_read(gt_path)

    # 2. Exploration Entry Vertices, Exploration Query, and Exploration Ground Truth
    explore_entry_path = files_dir / meta["explore_entry_file"]
    explore_query_path = files_dir / meta["explore_query_file"]
    explore_gt_path = files_dir / meta["explore_gt_file"]

    if not explore_entry_path.is_file() or not explore_query_path.is_file():
        generate_explore_entry_and_query(base_vecs, explore_entry_path, explore_query_path)

    print(f"Loading explore entry vertices from {explore_entry_path}...")
    explore_entry = ivecs_read(explore_entry_path)

    if not explore_gt_path.is_file():
        print(f"Exploration Ground Truth not found at {explore_gt_path}.")
        explore_query_vecs = fvecs_read(explore_query_path)
        compute_and_save_explore_gt(base_vecs, explore_query_vecs, float_space, 1000, explore_gt_path)

    print(f"Loading explore groundtruth from {explore_gt_path}...")
    explore_gt = ivecs_read(explore_gt_path)

    return base_vecs, query_vecs, gt_vecs, explore_entry, explore_gt, meta


def find_default_graph(dataset_key: str, dataset_folder: Path) -> Optional[Path]:
    """Finds the default or best matching .deg graph in dataset directory."""
    deg_dir = dataset_folder / "deg"
    if not deg_dir.is_dir():
        deg_dir = dataset_folder

    deg_files = list(deg_dir.rglob("*.deg"))
    if not deg_files:
        return None

    for f in deg_files:
        if dataset_key.lower() in f.name.lower():
            return f
    return deg_files[0]


def tune_explore_eps(
    graph,
    explore_entry: np.ndarray,
    explore_gt: np.ndarray,
    k: int = 100,
    target_recall: float = 0.99,
    eps_candidates: Optional[list] = None,
    threads: int = 4,
) -> Tuple[float, float]:
    """Finds the minimum exploration eps achieving target recall on explore groundtruth."""
    if eps_candidates is None:
        eps_candidates = [0.01, 0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4]

    query_count = min(len(explore_entry), 2000)
    eval_entry = np.ascontiguousarray(explore_entry[:query_count], dtype=np.uint32)
    eval_gt = explore_gt[:query_count, :k]

    print(f"\n--- Tuning Exploration eps on {query_count} explore queries (Target Recall: {target_recall}) ---")

    best_eps = eps_candidates[-1]
    best_recall = 0.0

    for eps in eps_candidates:
        indices_batch, _ = graph.explore(
            eval_entry,
            k=k,
            eps=eps,
            include_entry=True,
            threads=threads,
            return_distances=False,
        )

        hits = 0
        total_returned = 0
        for i in range(query_count):
            gt_set = set(eval_gt[i])
            ret_set = set(indices_batch[i])
            hits += len(gt_set.intersection(ret_set))
            total_returned += len(gt_set)

        recall = hits / max(total_returned, 1)
        print(f"  eps {eps:6.3f} -> Recall@{k}: {recall:.4f}")

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


def compute_linear_groundtruth_chunk(
    query_chunk: np.ndarray,
    base_target: np.ndarray,
    metric: Metric,
    k: int,
    pool: ThreadPoolExecutor,
    query_offset: Optional[int] = None,
) -> np.ndarray:
    """Computes exact top-k nearest neighbor indices for query_chunk against base_target (excluding self if query_offset provided)."""
    n_queries = len(query_chunk)
    n_base = len(base_target)
    actual_k = min(k, n_base if query_offset is None else max(0, n_base - 1))

    if metric == Metric.FP32_InnerProduct:
        sims = np.dot(query_chunk, base_target.T)
        if query_offset is not None:
            for i in range(n_queries):
                self_idx = query_offset + i
                if self_idx < n_base:
                    sims[i, self_idx] = -np.inf

        def _sort_ip(idx):
            row = sims[idx]
            part = np.argpartition(-row, actual_k)[:actual_k]
            return part[np.argsort(-row[part])]

        results = list(pool.map(_sort_ip, range(n_queries)))
    else:
        q_norm = np.sum(query_chunk**2, axis=1, keepdims=True)
        b_norm = np.sum(base_target**2, axis=1, keepdims=True).T
        dists_sq = q_norm + b_norm - 2.0 * np.dot(query_chunk, base_target.T)
        if query_offset is not None:
            for i in range(n_queries):
                self_idx = query_offset + i
                if self_idx < n_base:
                    dists_sq[i, self_idx] = np.inf

        def _sort_l2(idx):
            row = dists_sq[idx]
            part = np.argpartition(row, actual_k)[:actual_k]
            return part[np.argsort(row[part])]

        results = list(pool.map(_sort_l2, range(n_queries)))

    return np.asarray(results, dtype=np.int32)


def generate_base_top_linear(
    base_vecs: np.ndarray,
    metric: Metric,
    k: int,
    out_file: Path,
    total_queries: int,
    base_limit: int = 0,
    chunk_size: int = 1000,
    num_threads: int = 0,
):
    effective_threads = num_threads if num_threads > 0 else (os.cpu_count() or 4)
    base_target = base_vecs[:base_limit] if base_limit > 0 else base_vecs
    n_base = len(base_target)
    actual_k = min(k, n_base)

    print(
        f"\nStarting Exact Linear Search: {total_queries} queries against {n_base} base vectors "
        f"(k={actual_k}, metric={metric.name}, {effective_threads} threads, chunk={chunk_size})..."
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    if temp_file.exists():
        temp_file.unlink()

    k_int32 = np.int32(actual_k)
    start_t = time.perf_counter()

    with ThreadPoolExecutor(max_workers=effective_threads) as pool:
        with open(temp_file, "wb") as f_out:
            for start_idx in range(0, total_queries, chunk_size):
                end_idx = min(start_idx + chunk_size, total_queries)
                chunk_q = base_vecs[start_idx:end_idx]

                top_indices = compute_linear_groundtruth_chunk(
                    chunk_q, base_target, metric, actual_k, pool, query_offset=start_idx
                )

                n_chunk = len(chunk_q)
                d_col = np.full((n_chunk, 1), k_int32, dtype=np.int32)
                with_d = np.hstack([d_col, top_indices])
                f_out.write(with_d.tobytes())

                elapsed = time.perf_counter() - start_t
                qps = end_idx / max(elapsed, 1e-9)
                pct = 100.0 * end_idx / total_queries
                print(
                    f"\r  Progress: {end_idx}/{total_queries} ({pct:.1f}%) in {elapsed:.1f}s ({qps:.1f} vec/s)",
                    end="",
                    flush=True,
                )

    print()
    if out_file.exists():
        out_file.unlink()
    temp_file.rename(out_file)

    total_time = time.perf_counter() - start_t
    mb = out_file.stat().st_size / (1024 * 1024)
    print(f"Linear search complete: {out_file} ({mb:.2f} MB in {total_time:.1f}s)")


def generate_base_top_anns(
    base_vecs: np.ndarray,
    dataset_key: str,
    cache_dir: Path,
    k: int,
    out_file: Path,
    total_queries: int,
    graph_path_arg: Optional[str] = None,
    chunk_size: int = 25000,
    threads: int = 0,
    eps_override: Optional[float] = None,
    target_recall: float = 0.99,
):
    effective_threads = threads if threads > 0 else (os.cpu_count() or 4)
    meta = DATASET_METADATA[dataset_key]

    if graph_path_arg:
        graph_path = Path(graph_path_arg)
        if not graph_path.is_file():
            raise FileNotFoundError(f"Specified graph file does not exist: {graph_path}")
    else:
        dataset_folder = cache_dir / meta["folder"]
        graph_path = find_default_graph(dataset_key, dataset_folder)
        if not graph_path:
            raise FileNotFoundError(f"Could not find any existing .deg graph for {dataset_key} in {dataset_folder}")

    print(f"Using Graph: {graph_path}")
    t_load = time.perf_counter()
    graph = deglib.load_readonly_graph(str(graph_path))
    print(f"Graph loaded in {time.perf_counter() - t_load:.2f}s | Vertices: {graph.size()}")

    if eps_override is not None:
        chosen_eps = eps_override
    else:
        from presets import get_preset

        preset = get_preset(dataset_key)
        eps_candidates = sorted(preset.get("search_eps_list", [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]))
        files_dir = cache_dir / meta["folder"]
        if (files_dir / meta["folder"]).is_dir():
            files_dir = files_dir / meta["folder"]
        explore_entry = ivecs_read(files_dir / meta["explore_entry_file"])
        explore_gt = ivecs_read(files_dir / meta["explore_gt_file"])

        chosen_eps, _ = tune_explore_eps(
            graph=graph,
            explore_entry=explore_entry,
            explore_gt=explore_gt,
            k=k,
            target_recall=target_recall,
            eps_candidates=eps_candidates,
            threads=effective_threads,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    if temp_file.exists():
        temp_file.unlink()

    k_int32 = np.int32(k)
    start_t = time.perf_counter()

    with open(temp_file, "wb") as f_out:
        for start_idx in range(0, total_queries, chunk_size):
            end_idx = min(start_idx + chunk_size, total_queries)
            batch_labels = np.arange(start_idx, end_idx, dtype=np.uint32)

            indices_chunk = graph.explore(
                batch_labels,
                k=k,
                eps=chosen_eps,
                include_entry=False,
                threads=effective_threads,
                return_distances=False,
            )

            n_chunk = len(batch_labels)
            d_col = np.full((n_chunk, 1), k_int32, dtype=np.int32)
            with_d = np.hstack([d_col, indices_chunk.astype(np.int32)])
            f_out.write(with_d.tobytes())

            elapsed = time.perf_counter() - start_t
            qps = end_idx / max(elapsed, 1e-9)
            pct = 100.0 * end_idx / total_queries
            print(
                f"\r  Progress: {end_idx}/{total_queries} ({pct:.1f}%) in {elapsed:.1f}s ({qps:.1f} vec/s)",
                end="",
                flush=True,
            )

    print()
    if out_file.exists():
        out_file.unlink()
    temp_file.rename(out_file)

    total_time = time.perf_counter() - start_t
    mb = out_file.stat().st_size / (1024 * 1024)
    print(f"ANNS explore complete: {out_file} ({mb:.2f} MB in {total_time:.1f}s)")


def generate_base_groundtruth(
    dataset_key: str,
    cache_dir: Path,
    k: int = 1000,
    quality_mode: str = "exact",
    graph_path: Optional[Union[str, Path]] = None,
    include_half: bool = False,
    threads: int = 0,
):
    key = resolve_dataset_key(dataset_key)
    meta = DATASET_METADATA[key]
    folder = ensure_dataset(key, cache_dir)
    files_dir = folder / meta["folder"] if (folder / meta["folder"]).is_dir() else folder

    base_path = files_dir / meta["base_file"]
    base_vecs = fvecs_read(base_path)
    n_base = len(base_vecs)
    half_base = n_base // 2
    metric = meta["metric"]
    use_anns = quality_mode.lower() in ("approximated", "approx", "anns")

    out_file = files_dir / f"{meta['folder']}_base_top{k}.ivecs"
    if not out_file.is_file():
        if use_anns:
            generate_base_top_anns(
                base_vecs=base_vecs,
                dataset_key=key,
                cache_dir=cache_dir,
                k=k,
                out_file=out_file,
                total_queries=n_base,
                graph_path_arg=str(graph_path) if graph_path else None,
                threads=threads,
            )
        else:
            generate_base_top_linear(
                base_vecs=base_vecs,
                metric=metric,
                k=k,
                out_file=out_file,
                total_queries=n_base,
                base_limit=0,
                num_threads=threads,
            )

    if include_half:
        half_out = files_dir / f"{meta['folder']}_base_half_top{k}.ivecs"
        if not half_out.is_file():
            if use_anns:
                generate_base_top_anns(
                    base_vecs=base_vecs,
                    dataset_key=key,
                    cache_dir=cache_dir,
                    k=k,
                    out_file=half_out,
                    total_queries=half_base,
                    graph_path_arg=str(graph_path) if graph_path else None,
                    threads=threads,
                )
            else:
                generate_base_top_linear(
                    base_vecs=base_vecs,
                    metric=metric,
                    k=k,
                    out_file=half_out,
                    total_queries=half_base,
                    base_limit=half_base,
                    num_threads=threads,
                )


def load_base_groundtruth(
    dataset_key: str,
    cache_dir: Path,
    graph_quality: str = "none",
    graph_path: Optional[Union[str, Path]] = None,
) -> Optional[np.ndarray]:
    """
    Loads base_top1000 ground truth for graph quality analysis.
    If the file does not exist and graph_quality is 'exact' or 'approximated',
    it is automatically computed and cached.
    """
    if not graph_quality or graph_quality.lower() == "none":
        return None

    key = resolve_dataset_key(dataset_key)
    meta = DATASET_METADATA[key]
    folder = ensure_dataset(key, cache_dir)
    files_dir = folder / meta["folder"] if (folder / meta["folder"]).is_dir() else folder
    base_top_path = files_dir / f"{meta['folder']}_base_top1000.ivecs"

    if not base_top_path.is_file():
        print(f"Base Top Ground Truth not found at {base_top_path}. Generating...")
        generate_base_groundtruth(
            dataset_key=key,
            cache_dir=cache_dir,
            k=1000,
            quality_mode=graph_quality,
            graph_path=graph_path,
        )

    if base_top_path.is_file():
        print(f"Loading base ground truth for graph quality from {base_top_path} (mmap)...")
        return ivecs_read(base_top_path, mmap=True)

    return None
