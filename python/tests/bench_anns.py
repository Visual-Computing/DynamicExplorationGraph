"""
Mirror of the c++ benchmark "cpp/benchmark/src/deglib_anns_bench.cpp".
"""
import sys
import pathlib

import deglib
from deglib.utils import StopWatch, get_current_rss_mb


# TODO: check slow performance


def test_deglib_anns_bench():
    print("Testing ...")

    if deglib.avx_usable():
        print("use AVX2  ...")
    elif deglib.sse_usable():
        print("use SSE  ...")
    else:
        print("use arch  ...")

    data_path: pathlib.Path = pathlib.Path(sys.argv[1])
    repeat_test = 1
    k = 100

    query_file: pathlib.Path = data_path / "SIFT1M" / "sift_query.fvecs"
    graph_file: pathlib.Path = (data_path / "deg" / "best_distortion_decisions" /
                                "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0"
                                "+0_improveEvery2ndNonPerfectEdge.deg")
    gt_file: pathlib.Path = data_path / "SIFT1M" / "sift_groundtruth.ivecs"

    assert query_file.is_file(), 'Could not find query file: {}'.format(query_file)
    assert graph_file.is_file(), 'Could not find graph file: {}'.format(graph_file)
    assert gt_file.is_file(), 'Could not find ground truth file: {}'.format(gt_file)

    print("Load graph {}".format(graph_file))
    print("Actual memory usage: {} Mb".format(get_current_rss_mb()))
    stop_watch = StopWatch()
    graph = deglib.graph.load_readonly_graph(graph_file)
    elapsed_us = stop_watch.get_elapsed_time_micro()
    print("Graph with {} vertices".format(graph.size()))
    print("Actual memory usage: {} Mb".format(get_current_rss_mb()))
    print("Loading Graph took {} us".format(elapsed_us))

    query_repository = deglib.repository.fvecs_read(query_file)
    print("{} Query Features with {} dimensions".format(query_repository.shape[0], query_repository.shape[1]))

    ground_truth = deglib.repository.ivecs_read(gt_file)
    print("{} ground truth {} dimensions".format(ground_truth.shape[0], ground_truth.shape[1]))

    print("Test with k={} and repeat_test={}".format(k, repeat_test))

    deglib.benchmark.test_graph_anns(graph, query_repository, ground_truth, repeat_test, k)


if __name__ == '__main__':
    test_deglib_anns_bench()  # TODO: replace with test framework
