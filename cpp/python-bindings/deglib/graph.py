import deglib_cpp
import pathlib


def load_readonly_graph(path: pathlib.Path | str):
    return deglib_cpp.load_readonly_graph(str(path))


__all__ = ['load_readonly_graph']
