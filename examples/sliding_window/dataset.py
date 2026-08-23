import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np

import h5py
from deglib.distances import Metric


def ivecs_read(filename: str | Path) -> np.ndarray:
    """Read .ivecs vector file."""
    a = np.fromfile(filename, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(filename: str | Path) -> np.ndarray:
    """Read .fvecs vector file."""
    return ivecs_read(filename).view(np.float32)


def fbin_read(filename: str | Path) -> np.ndarray:
    """Read .fbin binary file (count: uint32, dim: uint32, data: float32[count * dim])."""
    with open(filename, "rb") as f:
        count = np.fromfile(f, dtype=np.uint32, count=1)[0]
        dim = np.fromfile(f, dtype=np.uint32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=int(count) * int(dim))
        return data.reshape(int(count), int(dim))


DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "redcaps": {
        "name": "RedCaps-512",
        "url": "https://zenodo.org/records/13137120/files/redcaps-512-angular.hdf5?download=1",
        "file": "redcaps-512-angular.hdf5",
        "folder": "redcaps",
        "format": "hdf5",
        "metric": Metric.FP32_L2,  # Normalized angular embeddings -> L2 space
        "dim": 512,
    },
    "sift1m": {
        "name": "SIFT1M",
        "url": "https://static.visual-computing.com/paper/DEG/sift.tar.gz",
        "archive": "sift.tar.gz",
        "folder": "sift1m",
        "format": "tar_fvecs",
        "metric": Metric.FP32_L2,
        "dim": 128,
        "base_file": "sift1m_base.fvecs",
        "query_file": "sift1m_query.fvecs",
    },
    "glove": {
        "name": "GloVe-100",
        "url": "https://static.visual-computing.com/paper/DEG/glove-100.tar.gz",
        "archive": "glove-100.tar.gz",
        "folder": "glove",
        "format": "tar_fvecs",
        "metric": Metric.FP32_InnerProduct,
        "dim": 100,
        "base_file": "glove_base.fvecs",
        "query_file": "glove_query.fvecs",
    },
    "deep1m": {
        "name": "DEEP1M",
        "url": "https://static.visual-computing.com/paper/DEG/deep1m.tar.gz",
        "archive": "deep1m.tar.gz",
        "folder": "deep1m",
        "format": "tar_fvecs",
        "metric": Metric.FP32_L2,
        "dim": 96,
        "base_file": "deep1m_base.fvecs",
        "query_file": "deep1m_query.fvecs",
    },
}


def get_default_cache_dir() -> Path:
    """Returns the default dataset cache directory (~/.cache/deg_datasets, DEG_CACHE_DIR, or D:/Data/DEG)."""
    env_dir = os.environ.get("DEG_CACHE_DIR") or os.environ.get("DEG_DATA_PATH")
    if env_dir:
        path = Path(env_dir)
    else:
        d_drive = Path("D:/Data/DEG")
        try:
            if d_drive.parent.exists():
                d_drive.mkdir(parents=True, exist_ok=True)
                return d_drive
        except (PermissionError, OSError):
            pass
        path = Path.home() / ".cache" / "deg_datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest_path: Path):
    """
    Downloads a file from url to dest_path with automatic resume support.
    Uses system curl with resume flag (-C -) if available, or Python requests with Range header.
    """
    import shutil
    import subprocess

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    existing_bytes = temp_path.stat().st_size if temp_path.is_file() else 0
    if existing_bytes > 0:
        print(f"Resuming download from byte offset {existing_bytes:,} ({existing_bytes / (1024 * 1024):.1f} MB)...")

    # 1. First priority: System curl (supports automatic resume, retries, speed & bytes stats)
    if shutil.which("curl"):
        print(f"Downloading {url}\n  -> {dest_path} (via curl with resume)...")
        cmd = [
            "curl",
            "-L",
            "-C", "-",
            "--retry", "10",
            "--retry-delay", "3",
            "-o", str(temp_path),
            url,
        ]
        ret = subprocess.run(cmd)
        if ret.returncode == 0 and temp_path.is_file():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"\nDownload complete: {dest_path}")
            return
        elif ret.returncode != 0:
            print(f"curl exited with code {ret.returncode}, attempting Python resume fallback...")

    # 2. Fallback: Python requests with HTTP Range header for resume
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Referer": "https://zenodo.org/",
    }

    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    session = requests.Session()
    response = session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60)

    if response.status_code == 206:
        mode = "ab"
        downloaded = existing_bytes
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            total_size = int(content_range.split("/")[-1])
        else:
            total_size = existing_bytes + int(response.headers.get("Content-Length", 0))
    elif response.status_code == 200:
        mode = "wb"
        downloaded = 0
        total_size = int(response.headers.get("Content-Length", 0))
    else:
        response.raise_for_status()
        return

    block_size = 4 * 1024 * 1024  # 4MB chunks
    with open(temp_path, mode) as out_file:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = downloaded * 100 // total_size
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\rProgress: {percent}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
                else:
                    mb = downloaded / (1024 * 1024)
                    print(f"\rDownloaded: {mb:.1f} MB", end="", flush=True)

    print()
    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)
    print(f"Download complete: {dest_path}")


def ensure_dataset(dataset_key: str, cache_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Ensures that the dataset directory and file exist under <cache_dir>/<dataset_folder>/.
    Returns (dataset_dir, data_file_path, meta).
    """
    key = dataset_key.lower()
    if key not in DATASET_METADATA:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choose from: {list(DATASET_METADATA.keys())}")

    meta = DATASET_METADATA[key]
    dataset_dir = cache_dir / meta["folder"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    fmt = meta["format"]

    if fmt == "hdf5":
        target_file = dataset_dir / meta["file"]
        part_file = dataset_dir / (meta["file"] + ".part")

        # Migrate legacy files from cache_dir root if present
        legacy_file = cache_dir / meta["file"]
        legacy_part = cache_dir / (meta["file"] + ".part")
        if not target_file.is_file() and legacy_file.is_file():
            print(f"Moving {legacy_file} -> {target_file}...")
            legacy_file.rename(target_file)
        if not target_file.is_file() and not part_file.is_file() and legacy_part.is_file():
            print(f"Moving {legacy_part} -> {part_file}...")
            legacy_part.rename(part_file)

        if not target_file.is_file():
            download_file(meta["url"], target_file)

        return dataset_dir, target_file, meta

    elif fmt == "tar_fvecs":
        archive_path = cache_dir / meta["archive"]
        extracted_folder = dataset_dir

        if not (extracted_folder / meta["base_file"]).is_file():
            if not archive_path.is_file():
                download_file(meta["url"], archive_path)

            print(f"Extracting {archive_path} into {cache_dir}...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=cache_dir)
            print("Extraction complete.")

        base_path = extracted_folder / meta["base_file"] if (extracted_folder / meta["base_file"]).is_file() else (cache_dir / meta["folder"] / meta["base_file"])
        return dataset_dir, base_path, meta


def build_graph_filename(
    dataset_dir: Path,
    dims: int,
    k: int,
    extend_k: int,
    extend_eps: float,
    optimization_target_str: str = "StreamingData",
    metric_str: str = "L2",
) -> Path:
    """
    Builds the graph path matching standard DEG format:
    <dataset_dir>/deg/{dims}D_{metric}_K{k}_AddK{extend_k}Eps{extend_eps:.1f}_{optimization_target_str}.deg
    """
    deg_dir = dataset_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dims}D_{metric_str}_K{k}_AddK{extend_k}Eps{extend_eps:.1f}_{optimization_target_str}.deg"
    return deg_dir / filename


def load_benchmark_dataset(
    dataset_key: str = "redcaps",
    cache_dir: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, Metric, Path]:
    """
    Downloads (if necessary) and loads base vectors, query vectors, and dataset directory.
    Returns: (base_vecs, query_vecs, metric, dataset_dir)
    """
    if cache_dir is None:
        cache_dir = get_default_cache_dir()

    key = dataset_key.lower()
    dataset_dir, file_path, meta = ensure_dataset(key, cache_dir)
    fmt = meta["format"]

    if fmt == "hdf5":
        print(f"Reading HDF5 dataset from {file_path}...")
        with h5py.File(file_path, "r") as h5:
            base_vecs = np.array(h5["train"], dtype=np.float32)
            query_vecs = np.array(h5["test"], dtype=np.float32)

        # Normalize if not already unit vectors
        norms = np.linalg.norm(base_vecs[:100], axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            print("Normalizing base and query embeddings to unit length...")
            base_vecs /= np.linalg.norm(base_vecs, axis=1, keepdims=True)
            query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

        return base_vecs, query_vecs, meta["metric"], dataset_dir

    elif fmt == "tar_fvecs":
        files_dir = dataset_dir
        base_path = files_dir / meta["base_file"]
        query_path = files_dir / meta["query_file"]

        print(f"Loading base features from {base_path}...")
        base_vecs = fvecs_read(base_path)

        print(f"Loading query features from {query_path}...")
        query_vecs = fvecs_read(query_path)

        return base_vecs, query_vecs, meta["metric"], dataset_dir


def compute_exact_topk_labels(
    active_labels: np.ndarray,
    active_features: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    metric: Metric = Metric.FP32_L2,
) -> np.ndarray:
    """
    Compute exact top-k ground truth labels for queries over the current active vectors.
    """
    num_queries = len(queries)
    gt_labels = np.empty((num_queries, k), dtype=np.uint32)

    # Process queries in chunks to bound memory usage
    chunk_size = 200
    for q_start in range(0, num_queries, chunk_size):
        q_end = min(q_start + chunk_size, num_queries)
        q_chunk = queries[q_start:q_end]

        if metric == Metric.FP32_InnerProduct:
            sims = np.dot(q_chunk, active_features.T)
            topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]
            row_indices = np.arange(q_end - q_start)[:, None]
            sorted_order = np.argsort(-sims[row_indices, topk_idx], axis=1)
        else:
            # Minimizing L2 distance
            q_norm_sq = np.sum(q_chunk**2, axis=1, keepdims=True)
            act_norm_sq = np.sum(active_features**2, axis=1, keepdims=True).T
            dists = q_norm_sq + act_norm_sq - 2.0 * np.dot(q_chunk, active_features.T)
            topk_idx = np.argpartition(dists, k, axis=1)[:, :k]
            row_indices = np.arange(q_end - q_start)[:, None]
            sorted_order = np.argsort(dists[row_indices, topk_idx], axis=1)

        sorted_topk_idx = topk_idx[row_indices, sorted_order]
        gt_labels[q_start:q_end] = active_labels[sorted_topk_idx]

    return gt_labels
