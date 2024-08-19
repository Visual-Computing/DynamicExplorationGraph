import time
import warnings
import numpy as np
import psutil


class NonContiguousWarning(Warning):
    pass


def assure_contiguous(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Assures that the returned array is c contiguous. If it is not, a new c-contiguous array will be created.
    Furthermore, the user is warned about poor performance.
    """
    if not arr.flags['C_CONTIGUOUS']:
        warnings.warn('{} is not c contiguous. This will lead to poor performance.'.format(name), NonContiguousWarning)
        arr = np.ascontiguousarray(arr)  # copy to create c-contiguous array
    return arr


class WrongDtypeException(Exception):
    pass


class InvalidShapeException(Exception):
    pass


def assure_dtype(arr: np.ndarray, name: str, dtype: np.dtype):
    """
    Raise an exception, if the given numpy array has the wrong dtype
    """
    if arr.dtype != dtype:
        raise WrongDtypeException('{} has wrong dtype \"{}\", expected was \"{}\"'.format(name, arr.dtype, dtype))


def assure_array(arr: np.ndarray, name: str, dtype: np.dtype) -> np.ndarray:
    """
    Makes sure, that the given array fulfills the requirements for c++ api.
    """
    assure_dtype(arr, name, dtype)
    return assure_contiguous(arr, name)


class StopWatch:
    def __init__(self):
        self.start_time = time.perf_counter_ns()

    def get_elapsed_time_micro(self) -> int:
        end_time = time.perf_counter_ns()
        return (end_time - self.start_time) // 1000


def get_current_rss_mb():
    return psutil.Process().memory_info().rss // 1000_000
