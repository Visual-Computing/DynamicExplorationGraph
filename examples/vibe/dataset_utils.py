import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple
import h5py
import numpy as np
from tqdm import tqdm

from deglib.distances import Metric, FloatSpace

# Base URL for VIBE dataset downloads on Hugging Face
HF_VIBE_BASE_URL = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main"

# Metadata mapping for the 10 requested VIBE datasets
VIBE_DATASETS: Dict[str, Dict[str, Any]] = {
    "agnews-mxbai": {
        "name": "AGNews-mxbai",
        "file": "agnews-mxbai-1024-euclidean.hdf5",
        "metric": Metric.FP32_L2,
        "type": "in-distribution",
        "dim": 1024,
        "size": 120000,
    },
    "arxiv-nomic": {
        "name": "ArXiv-nomic",
        "file": "arxiv-nomic-768-normalized.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "type": "in-distribution",
        "dim": 768,
        "size": 2000000,
    },
    "landmark-dino": {
        "name": "Landmark-dino",
        "file": "landmark-dino-768-cosine.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "normalize": True,
        "type": "in-distribution",
        "dim": 768,
        "size": 760757,
    },
    "msmarco-qwen": {
        "name": "MSMARCO-qwen",
        "file": "msmarco-qwen-1024-normalized.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "type": "in-distribution",
        "dim": 1024,
        "size": 8841823,
    },
    "gooaq-distilroberta": {
        "name": "GooAQ-distilroberta",
        "file": "gooaq-distilroberta-768-normalized.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "type": "in-distribution",
        "dim": 768,
        "size": 1471375,
    },
    "laion-clip": {
        "name": "LAION-clip",
        "file": "laion-clip-512-normalized.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "type": "out-of-distribution",
        "dim": 512,
        "size": 1000000,
    },
    "imagenet-align": {
        "name": "ImageNet-align",
        "file": "imagenet-align-640-normalized.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "type": "out-of-distribution",
        "dim": 640,
        "size": 1281167,
    },
    "imagenet-clip": {
        "name": "ImageNet-clip",
        "file": "imagenet-clip-512-normalized.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "type": "in-distribution",
        "dim": 512,
        "size": 1281167,
    },
    "yandex": {
        "name": "Yandex-200",
        "file": "yandex-200-cosine.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "normalize": True,
        "type": "out-of-distribution",
        "dim": 200,
        "size": 1000000,
    },
    "yahoo-minilm": {
        "name": "Yahoo-minilm",
        "file": "yahoo-minilm-384-normalized.hdf5",
        "metric": Metric.FP32_InnerProduct,
        "type": "in-distribution",
        "dim": 384,
        "size": 677305,
    },
}

ALIAS_MAP: Dict[str, str] = {
    "agnews": "agnews-mxbai",
    "agnews-mxbai": "agnews-mxbai",
    "arxiv": "arxiv-nomic",
    "arxiv-nomic": "arxiv-nomic",
    "landmark": "landmark-dino",
    "landmark-dino": "landmark-dino",
    "msmarco": "msmarco-qwen",
    "msmarco-qwen": "msmarco-qwen",
    "gooaq": "gooaq-distilroberta",
    "gooaq-distilroberta": "gooaq-distilroberta",
    "laion": "laion-clip",
    "laion-clip": "laion-clip",
    "imagenet-align": "imagenet-align",
    "imagenet-clip": "imagenet-clip",
    "imagenet": "imagenet-clip",
    "yandex": "yandex",
    "yandex-200": "yandex",
    "yahoo": "yahoo-minilm",
    "yahoo-minilm": "yahoo-minilm",
}


def resolve_dataset_key(key: str) -> str:
    """Normalizes dataset input key."""
    norm = key.strip().lower()
    if norm in ALIAS_MAP:
        return ALIAS_MAP[norm]
    for k in VIBE_DATASETS:
        if k in norm or norm in k:
            return k
    return norm


def get_default_cache_dir() -> Path:
    """Returns the default dataset cache directory (~/.cache/deg_datasets, DEG_CACHE_DIR, or D:/Data/DEG)."""
    env_dir = os.environ.get("DEG_CACHE_DIR") or os.environ.get("VIBE_CACHE_DIR") or os.environ.get("VIBE_CACHE")
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
    """Downloads a file with an interactive progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    print(f"Downloading {url} -> {dest_path}...")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )

    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB

        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name,
            miniters=1,
        ) as bar:
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))

    tmp_path.rename(dest_path)
    print(f"Download complete: {dest_path}")


def ensure_dataset(dataset_key: str, cache_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Ensures that the dataset folder and HDF5 file for dataset_key exist in cache_dir.
    Returns (dataset_dir, hdf5_path, meta).
    """
    key = resolve_dataset_key(dataset_key)
    if key not in VIBE_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Choose from: {list(VIBE_DATASETS.keys())}")

    meta = VIBE_DATASETS[key]
    dataset_dir = cache_dir / key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = dataset_dir / meta["file"]
    # Check if file exists in cache_dir root from a previous run and move it into dataset folder
    legacy_file = cache_dir / meta["file"]
    if not hdf5_path.is_file() and legacy_file.is_file():
        legacy_file.rename(hdf5_path)

    if not hdf5_path.is_file():
        url = f"{HF_VIBE_BASE_URL}/{meta['file']}"
        download_file(url, hdf5_path)

    return dataset_dir, hdf5_path, meta


def load_vibe_dataset(
    dataset_key: str, cache_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads a VIBE dataset HDF5 file.
    Returns:
        (train_base_vectors, test_query_vectors, gt_neighbor_indices, metadata)
    """
    key = resolve_dataset_key(dataset_key)
    dataset_dir, hdf5_path, meta = ensure_dataset(key, cache_dir)

    print(f"Opening HDF5 dataset file: {hdf5_path}...")
    with h5py.File(hdf5_path, "r") as f:
        # Check datasets inside HDF5
        print(f"  Available keys in HDF5: {list(f.keys())}")

        # Load train (corpus) vectors as float32 C-contiguous
        train_ds = f["train"]
        print(f"  Reading train embeddings ({train_ds.shape}, dtype={train_ds.dtype})...")
        base_vecs = np.ascontiguousarray(train_ds[:], dtype=np.float32)

        # Load test (query) vectors
        test_ds = f["test"]
        print(f"  Reading test query embeddings ({test_ds.shape}, dtype={test_ds.dtype})...")
        query_vecs = np.ascontiguousarray(test_ds[:], dtype=np.float32)

        # Load top-100 ground truth neighbors
        neighbors_ds = f["neighbors"]
        print(f"  Reading ground truth neighbors ({neighbors_ds.shape}, dtype={neighbors_ds.dtype})...")
        gt_neighbors = np.ascontiguousarray(neighbors_ds[:], dtype=np.int32)

        # Update metadata if actual attributes are in HDF5
        if "dimension" in f.attrs:
            meta["hdf5_dim"] = int(f.attrs["dimension"])
        if "distance" in f.attrs:
            meta["hdf5_distance"] = str(f.attrs["distance"])
        if "point_type" in f.attrs:
            meta["hdf5_point_type"] = str(f.attrs["point_type"])

    # If cosine distance is required, normalize to unit sphere for Inner Product metric
    if meta.get("normalize", False):
        print("  Normalizing vectors to unit L2-norm for Cosine / InnerProduct metric...")
        norm_b = np.linalg.norm(base_vecs, axis=1, keepdims=True)
        norm_b[norm_b == 0] = 1.0
        base_vecs = np.ascontiguousarray(base_vecs / norm_b, dtype=np.float32)

        norm_q = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        norm_q[norm_q == 0] = 1.0
        query_vecs = np.ascontiguousarray(query_vecs / norm_q, dtype=np.float32)

    print(
        f"Dataset '{meta['name']}' loaded successfully: {base_vecs.shape[0]:,} base vectors, "
        f"{query_vecs.shape[0]:,} query vectors, {base_vecs.shape[1]} dimensions, "
        f"metric={meta['metric'].name}."
    )
    return base_vecs, query_vecs, gt_neighbors, meta


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
    Builds the graph path matching static_data / C++ conventions:
    <dataset_dir>/deg/{dims}D_{metric}_K{k}_AddK{extend_k}Eps{extend_eps:.1f}_{optimization_target_str}.deg
    """
    dataset_dir, _, _ = ensure_dataset(dataset_key, cache_dir)
    deg_dir = dataset_dir / "deg"
    deg_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dims}D_{metric_str}_K{k}_AddK{extend_k}Eps{extend_eps:.1f}_{optimization_target_str}.deg"
    return deg_dir / filename
