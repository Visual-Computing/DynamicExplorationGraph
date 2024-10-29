from pathlib import Path
import numpy as np


def ivecs_read(filename: str | Path) -> np.ndarray:
    """
    Taken from https://github.com/facebookresearch/faiss/blob/main/benchs/datasets.py#L12.
    The loaded dataset should be in the format described here: http://corpus-texmex.irisa.fr/
    """
    a = np.fromfile(filename, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def u8vecs_read(filename: str | Path) -> np.ndarray:
    """
    The loaded dataset should be in the format described here: http://corpus-texmex.irisa.fr/
    """
    a = np.fromfile(filename, dtype=np.int32)
    b = a.view(np.uint8).reshape(-1, a[0] + 4)
    return b[:, 4:].copy()


def fvecs_read(filename: str | Path) -> np.ndarray:
    """
    Taken from https://github.com/facebookresearch/faiss/blob/main/benchs/datasets.py#L12
    The loaded dataset should be in the format described here: http://corpus-texmex.irisa.fr/
    """
    return ivecs_read(filename).view(np.int32)
