"""
dataset_utils.py — Helper utilities for downloading and loading SISAP 2026 Task 1 datasets (HDF5 format).
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
import numpy as np
import h5py

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deg_datasets"

HF_REPO_ID = "SISAP-Challenges/SISAP2026"
HF_SMALL_FILE = "wikipedia-small/benchmark-dev-wikipedia-bge-m3-small.h5"
HF_SMALL_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_SMALL_FILE}"


def ensure_small_dataset(cache_dir: Path | None = None) -> Path:
    """
    Ensures that benchmark-dev-wikipedia-bge-m3-small.h5 exists in local cache directory.
    Uses Hugging Face Hub native caching to avoid duplicate file downloads or copy errors.
    """
    # 1. Try HuggingFace Hub native download/cache
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SMALL_FILE,
            repo_type="dataset",
        )
        path = Path(downloaded)
        if path.is_file() and path.stat().st_size > 0:
            print(f"Using cached dataset: {path}")
            return path
    except Exception as err:
        print(f"huggingface_hub check ({err}), trying fallback location...")

    # 2. Check custom cache directory / direct HTTP download fallback
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    target_path = cache_dir / "wikipedia-small" / "benchmark-dev-wikipedia-bge-m3-small.h5"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.is_file() and target_path.stat().st_size > 0:
        print(f"Using cached dataset: {target_path}")
        return target_path

    # Fallback to direct HTTP download with progress reporter
    print(f"Downloading small dataset from {HF_SMALL_URL} to {target_path}...")
    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / max(total_size, 1))
        sys.stdout.write(f"\rDownloading dataset... {percent}% ({count * block_size / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(HF_SMALL_URL, target_path, _progress)
    print("\nDownload complete.")
    return target_path


def load_hdf5_dataset(file_path: str | Path, max_vecs: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Loads dataset vectors and optional ground-truth knns from a SISAP HDF5 dataset file.
    If max_vecs is specified, ground truth is re-computed for the sub-dataset.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with h5py.File(path, "r") as f:
        if "train" not in f:
            raise KeyError(f"Expected dataset key 'train' not found in {path}. Keys found: {list(f.keys())}")
        
        dataset = f["train"]
        if max_vecs is not None and max_vecs < dataset.shape[0]:
            train_data = dataset[:max_vecs]
            is_subset = True
        else:
            train_data = dataset[:]
            is_subset = False

        gt_data = None
        if not is_subset:
            if "allknn/knns" in f:
                gt_ds = f["allknn/knns"]
                gt_data = gt_ds[:]
            elif "knns" in f:
                gt_ds = f["knns"]
                gt_data = gt_ds[:]

            # Handle 1-based indexing in ground truth if present
            if gt_data is not None and gt_data.size > 0:
                if np.min(gt_data) == 1 and np.max(gt_data) >= train_data.shape[0]:
                    gt_data = gt_data - 1
        else:
            # Recompute exact top-15 inner-product ground truth for the subset
            print(f"Recomputing subset ground truth for first {max_vecs} vectors...")
            train_data_f32 = train_data.astype(np.float32)
            sim_matrix = np.dot(train_data_f32, train_data_f32.T)
            np.fill_diagonal(sim_matrix, -np.inf)
            gt_data = np.argsort(-sim_matrix, axis=1)[:, :15]

    return train_data, gt_data


def generate_synthetic_dataset(num_vecs: int = 2000, dims: int = 1024, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates normalized synthetic vectors and exact inner-product ground truth for self-join testing.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((num_vecs, dims)).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    data = data / norms

    # Compute exact inner product matrix for ground truth top 15
    sim_matrix = np.dot(data, data.T)
    np.fill_diagonal(sim_matrix, -np.inf) # Exclude self
    gt_data = np.argsort(-sim_matrix, axis=1)[:, :15]

    return data, gt_data
