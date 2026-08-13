"""
dataset_utils.py — Helper utilities for downloading and loading SISAP 2026 Task 2 MIPS dataset (HDF5 format).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
import numpy as np
import h5py

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deg_datasets"

HF_REPO_ID = "SISAP-Challenges/SISAP2026"
HF_MIPS_FILE = "llama-dev/llama-dev.h5"
HF_MIPS_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_MIPS_FILE}"


def ensure_mips_dataset(cache_dir: Path | None = None) -> Path:
    """
    Ensures that llama-dev/llama-dev.h5 exists in local cache directory.
    Uses Hugging Face Hub native caching to avoid duplicate downloads.
    """
    # 1. Try HuggingFace Hub native download/cache
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MIPS_FILE,
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
    target_path = cache_dir / "llama-dev" / "llama-dev.h5"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.is_file() and target_path.stat().st_size > 0:
        print(f"Using cached dataset: {target_path}")
        return target_path

    # Fallback to direct HTTP download
    print(f"Downloading llama-dev dataset from {HF_MIPS_URL} to {target_path}...")
    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / max(total_size, 1))
        sys.stdout.write(f"\rDownloading dataset... {percent}% ({count * block_size / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(HF_MIPS_URL, target_path, _progress)
    print("\nDownload complete.")
    return target_path


def load_hdf5_dataset(file_path: str | Path, k_top: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Loads train vectors, test query vectors, and ground truth from a SISAP Task 2 HDF5 dataset file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with h5py.File(path, "r") as f:
        # Read train vectors
        if "train" in f:
            train_data = f["train"][:]
        elif "emb" in f:
            train_data = f["emb"][:]
        else:
            first_key = list(f.keys())[0]
            train_data = f[first_key][:]

        # Read query vectors
        if "test/queries" in f:
            query_data = f["test/queries"][:]
        elif "queries" in f:
            query_data = f["queries"][:]
        elif "query" in f:
            query_data = f["query"][:]
        else:
            query_data = train_data[:100]

        # Read ground truth
        gt_data = None
        if "test/knns" in f:
            gt_data = f["test/knns"][:, :k_top]
        elif "knns" in f:
            gt_data = f["knns"][:, :k_top]

    return train_data.astype(np.float32), query_data.astype(np.float32), gt_data


def compute_recall(gt_knns: np.ndarray, result_knns: np.ndarray, k_top: int) -> float:
    """
    Computes average recall@k_top between ground truth and returned search indices.
    SISAP ground truth indices are 1-based (1..N), while DEG search result indices are 0-based (0..N-1).
    """
    if gt_knns is None or result_knns is None:
        return -1.0

    n_queries = min(gt_knns.shape[0], result_knns.shape[0])
    total_hits = 0

    for i in range(n_queries):
        gt_set_0based = {int(x) - 1 for x in gt_knns[i, :k_top] if int(x) > 0}
        res_set = set(result_knns[i, :k_top])

        total_hits += len(gt_set_0based.intersection(res_set))

    return total_hits / float(n_queries * k_top)

