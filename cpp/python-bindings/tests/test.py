import gc
import pathlib
import sys

import deglib_cpp
import deglib
import numpy as np


def print_helper(a):
    print('dtype:', a.dtype)
    deglib_cpp.print_py_buffer(a)


def test_simple():
    a = np.arange(12, dtype=np.int32).reshape(4, 3)

    print_helper(a.astype(np.int32))
    print_helper(a.astype(np.uint32))
    print_helper(a.astype(np.float32))
    print_helper(a.astype(np.int64))
    print_helper(a.astype(np.uint64))
    print_helper(a.astype(np.float64))


def test_transpose():
    a = np.arange(12, dtype=np.int32).reshape(4, 3)
    # print(a)
    # a = a[1:, 1:]
    print(a)
    deglib_cpp.print_py_buffer(a)
    print()

    print(a.T)
    deglib_cpp.print_py_buffer(a.T)


def test_slice():
    shape = (6, 7)
    a = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    print(a)
    deglib_cpp.print_py_buffer(a)
    print()

    a_sliced = a[:, 1:]
    print(a_sliced)
    deglib_cpp.print_py_buffer(a_sliced)
    print()

    a_sliced = a[::2, 1:5]
    print(a_sliced)
    deglib_cpp.print_py_buffer(a_sliced)


def test_buffer():
    buffer = deglib_cpp.MyBuffer(10)
    buffer.print_buffer()
    memview = buffer.get_memory_view()
    arr = np.asarray(memview)
    print(arr)
    arr[0] = 42
    print(arr)
    buffer.print_buffer()


def test_free_graph():
    data_path: pathlib.Path = pathlib.Path(sys.argv[1])
    graph_file: pathlib.Path = (data_path / "deg" / "best_distortion_decisions" /
                                "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0"
                                "+0_improveEvery2ndNonPerfectEdge.deg")

    graph = deglib.graph.load_readonly_graph(graph_file)
    fv = graph.get_feature_vector(10)
    print(fv)
    del graph

    print(fv)


def test_take_graph():
    data_path: pathlib.Path = pathlib.Path(sys.argv[1])
    repository_file = data_path / "SIFT1M" / "sift_base.fvecs"
    repository = deglib.repository.fvecs_read(repository_file)
    d = 30
    max_vertex_count, dims = repository.shape
    feature_space = deglib.FloatSpace(dims, deglib.Metric.L2)
    graph = deglib.graph.SizeBoundedGraph(max_vertex_count, d, feature_space)

    deglib_cpp.test_take_graph(graph.to_cpp())


def test_callback():
    build_infos = []

    def stateful_callback(build_status):
        build_infos.append(build_status)

    deglib_cpp.test_callback(stateful_callback, 1, 3)

    for bi in build_infos:
        print(bi.added)
        print(type(bi.added))
        print(sys.getsizeof(bi.added))


if __name__ == '__main__':
    test_simple()
    # test_transpose()
    # test_slice()
    # test_buffer()
    # test_free_graph()
    # test_take_graph()
    # test_callback()
