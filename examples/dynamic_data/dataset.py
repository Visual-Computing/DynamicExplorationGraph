import os
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
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
        "folder": "sift1m",
        "metric": Metric.FP32_L2,
        "dim": 128,
        "base_count": 1000000,
        "base_file": "sift1m_base.fvecs",
        "query_file": "sift1m_query.fvecs",
        "gt_file": "sift1m_groundtruth_top100_nb1000000.ivecs",
        "gt_half_file": "sift1m_groundtruth_top100_nb500000.ivecs",
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
        "gt_half_file": "audio_groundtruth_top100_nb26693.ivecs",
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
        "gt_half_file": "enron_groundtruth_top100_nb47493.ivecs",
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
        "gt_half_file": "deep1m_groundtruth_top100_nb500000.ivecs",
    },
    "glove": {
        "name": "GloVe-100",
        "url": "https://static.visual-computing.com/paper/DEG/glove-100.tar.gz",
        "archive": "glove-100.tar.gz",
        "folder": "glove",
        "metric": Metric.FP32_InnerProduct,
        "dim": 100,
        "base_count": 1183514,
        "base_file": "glove_base.fvecs",
        "query_file": "glove_query.fvecs",
        "gt_file": "glove_groundtruth_top100_nb1183514.ivecs",
        "gt_half_file": "glove_groundtruth_top100_nb591757.ivecs",
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


def compute_and_save_anns_gt(
    base_vecs: np.ndarray, query_vecs: np.ndarray, float_space: FloatSpace, k: int, out_path: Path
):
    """Computes ANNS ground truth against the full base dataset."""
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


def compute_and_save_anns_gt_half(
    base_vecs: np.ndarray, query_vecs: np.ndarray, float_space: FloatSpace, k: int, out_path: Path
):
    """Computes ANNS ground truth against only the first half of the base dataset.

    Mirrors the C++ generate_anns_groundtruth_files() with include_half=true,
    used for dynamic benchmarks where only half the base vectors are active.
    """
    half_base = base_vecs[: len(base_vecs) // 2]
    n_queries = len(query_vecs)
    gt_matrix = np.zeros((n_queries, k), dtype=np.int32)
    print(f"Generating ANNS Ground Truth (Half Dataset = {len(half_base)} vectors, top-{k})...")
    start_t = time.perf_counter()

    for q in range(n_queries):
        dists = float_space.compute_distances(query_vecs[q], half_base)
        top_k = np.argpartition(dists, k)[:k]
        top_k = top_k[np.argsort(dists[top_k])]
        gt_matrix[q, :] = top_k
        if (q + 1) % 100 == 0 or (q + 1) == n_queries:
            elapsed_ms = int((time.perf_counter() - start_t) * 1000)
            print(
                f"\r  Half GT progress: {q + 1}/{n_queries} ({100.0 * (q + 1) / n_queries:.1f}%) after {elapsed_ms}ms",
                end="",
                flush=True,
            )
    print()
    ivecs_write(out_path, gt_matrix)
    print(f"ANNS Half Ground Truth generated and saved to {out_path}")


def load_dataset_for_dynamic(
    dataset_key: str, cache_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Downloads/extracts dataset if needed and loads arrays for the dynamic benchmark.

    Unlike load_dataset() in static_data, this function:
    - Does NOT load exploration data (not needed for dynamic benchmark)
    - Loads (or generates) BOTH full and half ANNS ground truth

    Returns: (base_vecs, query_vecs, gt_vecs_full, gt_vecs_half, metadata)
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

    # Full ANNS Ground Truth
    gt_path = files_dir / meta["gt_file"]
    if not gt_path.is_file():
        print(f"ANNS Full Ground Truth not found at {gt_path}.")
        compute_and_save_anns_gt(base_vecs, query_vecs, float_space, 100, gt_path)

    print(f"Loading full groundtruth indices from {gt_path}...")
    gt_vecs_full = ivecs_read(gt_path)

    # Half ANNS Ground Truth (against first base_count/2 vectors)
    gt_half_path = files_dir / meta["gt_half_file"]
    if not gt_half_path.is_file():
        print(f"ANNS Half Ground Truth not found at {gt_half_path}.")
        compute_and_save_anns_gt_half(base_vecs, query_vecs, float_space, 100, gt_half_path)

    print(f"Loading half groundtruth indices from {gt_half_path}...")
    gt_vecs_half = ivecs_read(gt_half_path)

    return base_vecs, query_vecs, gt_vecs_full, gt_vecs_half, meta


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


def generate_base_half_top_linear(
    half_vecs: np.ndarray,
    metric: Metric,
    k: int,
    out_file: Optional[Path] = None,
    chunk_size: int = 1000,
    num_threads: int = 0,
) -> np.ndarray:
    """Computes exact base-to-base top-k ground truth over half_vecs (excluding self)."""
    effective_threads = num_threads if num_threads > 0 else (os.cpu_count() or 4)
    n_half = len(half_vecs)
    actual_k = min(k, max(1, n_half - 1))

    print(
        f"\nStarting Exact Linear Search (Half Base): {n_half} queries against {n_half} vectors "
        f"(k={actual_k}, metric={metric.name}, {effective_threads} threads, chunk={chunk_size})..."
    )

    start_t = time.perf_counter()

    if out_file is None:
        chunks = []
        with ThreadPoolExecutor(max_workers=effective_threads) as pool:
            for start_idx in range(0, n_half, chunk_size):
                end_idx = min(start_idx + chunk_size, n_half)
                chunks.append(
                    compute_linear_groundtruth_chunk(
                        half_vecs[start_idx:end_idx], half_vecs, metric, actual_k, pool, query_offset=start_idx
                    )
                )
        return np.vstack(chunks).view(np.uint32)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    if temp_file.exists():
        temp_file.unlink()

    k_int32 = np.int32(actual_k)
    with ThreadPoolExecutor(max_workers=effective_threads) as pool:
        with open(temp_file, "wb") as f_out:
            for start_idx in range(0, n_half, chunk_size):
                end_idx = min(start_idx + chunk_size, n_half)
                chunk_q = half_vecs[start_idx:end_idx]
                top_indices = compute_linear_groundtruth_chunk(
                    chunk_q, half_vecs, metric, actual_k, pool, query_offset=start_idx
                )
                d_col = np.full((len(chunk_q), 1), k_int32, dtype=np.int32)
                f_out.write(np.hstack([d_col, top_indices]).tobytes())

                elapsed = time.perf_counter() - start_t
                qps = end_idx / max(elapsed, 1e-9)
                pct = 100.0 * end_idx / n_half
                print(
                    f"\r  Progress: {end_idx}/{n_half} ({pct:.1f}%) in {elapsed:.1f}s ({qps:.1f} vec/s)",
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
    return ivecs_read(out_file, mmap=True)


def generate_base_half_top_anns(
    graph: Any,
    half_vecs: np.ndarray,
    metric: Metric,
    k: int,
    out_file: Optional[Path] = None,
    chunk_size: int = 25000,
    num_threads: int = 0,
    target_recall: float = 0.99,
) -> np.ndarray:
    """Computes approximated base-to-base top-k ground truth over half_vecs via graph exploration."""
    effective_threads = num_threads if num_threads > 0 else (os.cpu_count() or 4)
    n_half = len(half_vecs)
    actual_k = min(k, max(1, n_half - 1))

    # Tune exploration eps on a small random sample against exact linear top-k
    sample_count = min(500, n_half)
    rng = np.random.default_rng(7)
    sample_labels = np.sort(rng.choice(n_half, size=sample_count, replace=False)).astype(np.uint32)
    tune_k = min(100, actual_k)

    with ThreadPoolExecutor(max_workers=effective_threads) as pool:
        sample_gt = compute_linear_groundtruth_chunk(half_vecs[sample_labels], half_vecs, metric, tune_k + 1, pool)
    # Remove self from sample_gt rows
    filtered_gt = []
    for row_i, lbl in enumerate(sample_labels):
        row = sample_gt[row_i]
        row = row[row != int(lbl)][:tune_k]
        filtered_gt.append(set(int(x) for x in row))

    eps_candidates = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    chosen_eps = eps_candidates[-1]
    best_recall = 0.0
    print(f"\n--- Tuning Exploration eps on {sample_count} half-base samples (Target Recall: {target_recall}) ---")
    for eps in eps_candidates:
        indices_batch = graph.explore(
            sample_labels,
            k=tune_k,
            eps_or_ef=eps,
            include_entry=False,
            threads=effective_threads,
            return_distances=False,
        )
        hits = sum(len(filtered_gt[i].intersection(indices_batch[i])) for i in range(sample_count))
        recall = hits / max(sample_count * tune_k, 1)
        print(f"  eps {eps:6.3f} -> Recall@{tune_k}: {recall:.4f}")
        if recall >= target_recall:
            chosen_eps = eps
            best_recall = recall
            break
        if recall > best_recall:
            best_recall = recall
            chosen_eps = eps

    print(f"Generating half-base ground truth via graph.explore (eps={chosen_eps:.3f}, k={actual_k})...")
    start_t = time.perf_counter()

    if out_file is None:
        all_labels = np.arange(n_half, dtype=np.uint32)
        indices = graph.explore(
            all_labels,
            k=actual_k,
            eps_or_ef=chosen_eps,
            include_entry=False,
            threads=effective_threads,
            return_distances=False,
        )
        return np.asarray(indices, dtype=np.uint32)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_suffix(".tmp")
    if temp_file.exists():
        temp_file.unlink()

    k_int32 = np.int32(actual_k)
    with open(temp_file, "wb") as f_out:
        for start_idx in range(0, n_half, chunk_size):
            end_idx = min(start_idx + chunk_size, n_half)
            batch_labels = np.arange(start_idx, end_idx, dtype=np.uint32)
            indices_chunk = graph.explore(
                batch_labels,
                k=actual_k,
                eps_or_ef=chosen_eps,
                include_entry=False,
                threads=effective_threads,
                return_distances=False,
            )
            d_col = np.full((len(batch_labels), 1), k_int32, dtype=np.int32)
            f_out.write(np.hstack([d_col, indices_chunk.astype(np.int32)]).tobytes())

            elapsed = time.perf_counter() - start_t
            qps = end_idx / max(elapsed, 1e-9)
            pct = 100.0 * end_idx / n_half
            print(
                f"\r  Progress: {end_idx}/{n_half} ({pct:.1f}%) in {elapsed:.1f}s ({qps:.1f} vec/s)",
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
    return ivecs_read(out_file, mmap=True)


def load_base_half_groundtruth(
    dataset_key: str,
    cache_dir: Path,
    base_vecs: np.ndarray,
    graph_quality: str = "auto",
    graph: Optional[Any] = None,
    threads: int = 0,
) -> Optional[np.ndarray]:
    """
    Loads (or generates) half-dataset base_half_top1000 ground truth for dynamic graph quality analysis.

    Modes:
    - 'auto' (default): Loads existing ground truth file if present; skips if not found.
    - 'exact': Loads cached file if present, otherwise computes exact linear search ground truth.
    - 'approximated': Loads cached file if present, otherwise computes ANNS exploration ground truth using `graph`.
    """
    mode = (graph_quality or "auto").lower()

    key = resolve_dataset_key(dataset_key)
    meta = DATASET_METADATA[key]
    metric = meta["metric"]
    n_half = len(base_vecs) // 2
    if n_half <= 1:
        return None

    # Truncated debug run (--max-base-vecs): compute in-memory without reading/polluting full-dataset cache
    if len(base_vecs) < meta["base_count"]:
        if mode == "auto":
            # In debug/truncated runs, do not silently compute heavy GT unless explicitly requested
            return None
        half_vecs = base_vecs[:n_half]
        if mode in ("approximated", "approx", "anns") and graph is not None:
            return generate_base_half_top_anns(graph, half_vecs, metric, k=1000, out_file=None, num_threads=threads)
        return generate_base_half_top_linear(half_vecs, metric, k=1000, out_file=None, num_threads=threads)

    folder = ensure_dataset(key, cache_dir)
    files_dir = folder / meta["folder"] if (folder / meta["folder"]).is_dir() else folder
    base_half_path = files_dir / f"{meta['folder']}_base_half_top1000.ivecs"
    alt_half_path = files_dir / f"{key}_base_half_top1000.ivecs"

    existing_path = base_half_path if base_half_path.is_file() else (alt_half_path if alt_half_path.is_file() else None)
    if existing_path is not None:
        print(f"Loading half-base ground truth for graph quality from {existing_path} (mmap)...")
        return ivecs_read(existing_path, mmap=True)

    if mode == "auto":
        print(f"Half-base ground truth not found at {base_half_path}. Skipping graph quality evaluation (mode: auto).")
        return None

    half_vecs = base_vecs[:n_half]
    if mode in ("approximated", "approx", "anns"):
        if graph is None:
            raise ValueError("Graph instance is required to generate approximated half-base ground truth.")
        return generate_base_half_top_anns(graph, half_vecs, metric, k=1000, out_file=base_half_path, num_threads=threads)

    return generate_base_half_top_linear(half_vecs, metric, k=1000, out_file=base_half_path, num_threads=threads)

