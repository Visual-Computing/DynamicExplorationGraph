import numpy as np
from pathlib import Path

from deglib_cpp import avx_usable, sse_usable, distance_float_l2


def ivecs_read(fname: str | Path) -> np.ndarray:
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname: str | Path) -> np.ndarray:
    return ivecs_read(fname).view('float32')


__all__ = ['avx_usable', 'sse_usable', 'distance_float_l2', 'ivecs_read', 'fvecs_read']
