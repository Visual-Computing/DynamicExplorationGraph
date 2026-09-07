import os
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from deglib.distances import Metric, FloatSpace


def ivecs_read(filename: str | Path) -> np.ndarray:
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
        "metric": Metric.FP32_InnerProduct,
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
