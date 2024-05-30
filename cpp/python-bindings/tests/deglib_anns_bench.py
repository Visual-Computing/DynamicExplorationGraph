"""
Mirror of the c++ benchmark "cpp/benchmark/src/deglib_anns_bench.cpp".
"""
import sys
import pathlib

import deglib


def test_deglib_anns_bench():
    print("Testing ...")

    if deglib.avx_usable():
        print("use AVX2  ...")
    elif deglib.sse_usable():
        print("use SSE  ...")
    else:
        print("use arch  ...")

    data_path: pathlib.Path = pathlib.Path(sys.argv[1])

    query_file: pathlib.Path = data_path / "SIFT1M" / "sift_query.fvecs"
    graph_file: pathlib.Path = (data_path / "deg" / "best_distortion_decisions" /
                                "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0"
                                "+0_improveEvery2ndNonPerfectEdge.deg")
    gt_file: pathlib.Path = data_path / "SIFT1M" / "sift_groundtruth.ivecs"

    assert query_file.is_file(), 'Could not find query file: {}'.format(query_file)
    assert graph_file.is_file(), 'Could not find graph file: {}'.format(graph_file)
    assert gt_file.is_file(), 'Could not find ground truth file: {}'.format(gt_file)

    graph = deglib.graph.load_readonly_graph(graph_file)

    print("Graph with {} vertices".format(graph.size()))

    query_repository = deglib::load_static_repository(query_file.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());


if __name__ == '__main__':
    test_deglib_anns_bench()  # TODO: replace with test framework
