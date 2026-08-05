import os
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import deglib.repository as repo
from deglib.distances import Metric, FloatSpace

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "sift1m": {
        "name": "SIFT1M",
        "url": "https://static.visual-computing.com/paper/DEG/sift.tar.gz",
        "archive": "sift.tar.gz",
        "folder": "sift",
        "metric": Metric.FP32_L2,
        "dim": 128,
        "base_count": 1000000,
        "base_file": "sift_base.fvecs",
        "query_file": "sift_query.fvecs",
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
    "glove-100": {
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

# Aliases
DATASET_ALIASES = {
    "glove": "glove-100",
}

def resolve_dataset_key(key: str) -> str:
    key = key.lower()
    return DATASET_ALIASES.get(key, key)

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
    extracted_folder = cache_dir / meta["folder"]

    if not extracted_folder.is_dir():
        if not archive_path.is_file():
            download_file(meta["url"], archive_path)
        
        print(f"Extracting {archive_path} into {cache_dir}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)
        print("Extraction complete.")
    
    if not extracted_folder.is_dir():
        subdirs = [p for p in cache_dir.iterdir() if p.is_dir() and meta["folder"].lower() in p.name.lower()]
        if subdirs:
            extracted_folder = subdirs[0]
        else:
            extracted_folder = cache_dir

    return extracted_folder

def find_file(directory: Path, expected_name: str, pattern: str) -> Path:
    """Finds a file by exact name or matching pattern in directory tree."""
    target = directory / expected_name
    if target.is_file():
        return target
    
    matches = list(directory.rglob(expected_name))
    if matches:
        return matches[0]

    matches = list(directory.rglob(f"*{pattern}*"))
    if matches:
        return matches[0]
    
    raise FileNotFoundError(f"Could not find file '{expected_name}' or pattern '{pattern}' in {directory}")

def compute_and_save_anns_gt(base_vecs: np.ndarray, query_vecs: np.ndarray, float_space: FloatSpace, k: int, out_path: Path):
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
            print(f"\r  Ground truth progress: {q + 1}/{n_queries} ({100.0 * (q + 1) / n_queries:.1f}%) after {elapsed_ms}ms", end="", flush=True)
    print()
    ivecs_write(out_path, gt_matrix)
    print(f"ANNS Ground Truth generated and saved to {out_path}")

def compute_and_save_anns_gt_half(base_vecs: np.ndarray, query_vecs: np.ndarray, float_space: FloatSpace, k: int, out_path: Path):
    """Computes ANNS ground truth against only the first half of the base dataset.

    Mirrors the C++ generate_anns_groundtruth_files() with include_half=true,
    used for dynamic benchmarks where only half the base vectors are active.
    """
    half_base = base_vecs[:len(base_vecs) // 2]
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
            print(f"\r  Half GT progress: {q + 1}/{n_queries} ({100.0 * (q + 1) / n_queries:.1f}%) after {elapsed_ms}ms", end="", flush=True)
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

    base_path = find_file(folder, meta["base_file"], "base")
    query_path = find_file(folder, meta["query_file"], "query")

    print(f"Loading base features from {base_path}...")
    base_vecs = repo.fvecs_read(base_path)

    print(f"Loading query features from {query_path}...")
    query_vecs = repo.fvecs_read(query_path)

    dims = base_vecs.shape[1]
    metric = meta["metric"]
    float_space = FloatSpace.create(dims, metric)

    # Full ANNS Ground Truth
    gt_file_name = meta["gt_file"]
    gt_path = folder / gt_file_name
    if not gt_path.is_file():
        sub_matches = list(folder.rglob(gt_file_name))
        if sub_matches:
            gt_path = sub_matches[0]
        else:
            print(f"ANNS Full Ground Truth not found at {gt_path}.")
            compute_and_save_anns_gt(base_vecs, query_vecs, float_space, 100, gt_path)

    print(f"Loading full groundtruth indices from {gt_path}...")
    gt_vecs_full = repo.ivecs_read(gt_path)

    # Half ANNS Ground Truth (against first base_count/2 vectors)
    gt_half_file_name = meta["gt_half_file"]
    gt_half_path = folder / gt_half_file_name
    if not gt_half_path.is_file():
        sub_matches = list(folder.rglob(gt_half_file_name))
        if sub_matches:
            gt_half_path = sub_matches[0]
        else:
            print(f"ANNS Half Ground Truth not found at {gt_half_path}.")
            compute_and_save_anns_gt_half(base_vecs, query_vecs, float_space, 100, gt_half_path)

    print(f"Loading half groundtruth indices from {gt_half_path}...")
    gt_vecs_half = repo.ivecs_read(gt_half_path)

    return base_vecs, query_vecs, gt_vecs_full, gt_vecs_half, meta
