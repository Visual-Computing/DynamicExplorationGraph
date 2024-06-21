import enum
import pathlib
import sys

import deglib


class DatasetName(enum.Enum):
    Audio = enum.auto()
    Enron = enum.auto()
    Sift1m = enum.auto()
    Glove = enum.auto()


def main():
    print("Testing ...")

    if deglib.avx_usable():
        print("use AVX2  ...")
    elif deglib.sse_usable():
        print("use SSE  ...")
    else:
        print("use arch  ...")

    repeat_test = 1

    data_path: pathlib.Path = pathlib.Path(sys.argv[1])

    k = 1000

    dataset_name = DatasetName.Audio

    if dataset_name == DatasetName.Sift1m:
        graph_file = data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg"
        gt_file = data_path / "SIFT1M" / "sift_explore_ground_truth.ivecs"
        query_file = data_path / "SIFT1M" / "sift_explore_entry_vertex.ivecs"
    elif dataset_name == DatasetName.Glove:
        graph_file = data_path / "deg" / "100D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg"
        gt_file = data_path / "glove-100" / "glove-100_explore_ground_truth.ivecs"
        query_file = data_path / "glove-100" / "glove-100_explore_entry_vertex.ivecs"
    elif dataset_name == DatasetName.Glove:
        graph_file = data_path / "deg" / "1369D_L2_K30_AddK60Eps0.3High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg"
        gt_file = data_path / "enron" / "enron_explore_ground_truth.ivecs"
        query_file = data_path / "enron" / "enron_explore_entry_vertex.ivecs"
    elif dataset_name == DatasetName.Audio:
        graph_file = data_path / "deg" / "neighbor_choice" / "192D_L2_K20_AddK40Eps0.3Low_schemeA.deg"
        # graph_file = data_path / "deg" / "192D_L2_K20_AddK40Eps0.3High_SwapK20-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg"
        gt_file = data_path / "audio" / "audio_explore_ground_truth.ivecs"
        query_file = data_path / "audio" / "audio_explore_entry_vertex.ivecs"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # load graph
    print("Load graph {}".format(graph_file))
    graph = deglib.graph.load_readonly_graph(graph_file)

    # load starting vertex data
    entry_vertex = deglib.repository.ivecs_read(query_file)
    print("{} entry vertex {} dimensions".format(entry_vertex.shape[0], entry_vertex.shape[1]))

    # load ground truth data (nearest neighbors of the starting vertices)
    ground_truth = deglib.repository.ivecs_read(gt_file)
    print("{} ground truth {} dimensions".format(ground_truth.shape[0], ground_truth.shape[1]))

    # explore the graph
    deglib.benchmark.test_graph_explore(graph, ground_truth, entry_vertex, repeat_test, k)

    print("Test OK")


if __name__ == '__main__':
    main()
