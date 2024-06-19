from pathlib import Path
import numpy as np


def ivecs_read(filename: str | Path) -> np.ndarray:
    """
    Taken from https://github.com/facebookresearch/faiss/blob/main/benchs/datasets.py#L12
    """
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(filename: str | Path) -> np.ndarray:
    """
    Taken from https://github.com/facebookresearch/faiss/blob/main/benchs/datasets.py#L12
    """
    return ivecs_read(filename).view('float32')
