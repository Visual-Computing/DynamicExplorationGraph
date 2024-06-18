from deglib_cpp import avx_usable, sse_usable
from . import datasets
from . import graph
from . import benchmark
from .distances import Metric, FloatSpace
from .repository import load_static_repository


__all__ = [
    'avx_usable', 'sse_usable', 'datasets', 'graph', 'load_static_repository', 'benchmark', 'Metric', 'FloatSpace'
]
