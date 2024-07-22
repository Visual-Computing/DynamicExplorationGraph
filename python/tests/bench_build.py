import pathlib
import sys
import enum
import time

import deglib
from deglib.utils import get_current_rss_mb, StopWatch


class DataStreamType(enum.Enum):
    AddAll = enum.auto()
    AddHalf = enum.auto()
    AddAllRemoveHalf = enum.auto()
    AddHalfRemoveAndAddOneAtATime = enum.auto()


class DatasetName(enum.Enum):
    Audio = enum.auto()
    Enron = enum.auto()
    Sift1m = enum.auto()
    Glove = enum.auto()


def main():
    if deglib.avx_usable():
        print("use AVX2  ...")
    elif deglib.sse_usable():
        print("use SSE  ...")
    else:
        print("use arch  ...")

    print("Actual memory usage: {} Mb".format(get_current_rss_mb()))

    data_path: pathlib.Path = pathlib.Path(sys.argv[1])

    dataset_name = DatasetName.Audio
    if dataset_name == DatasetName.Audio:
        repository_file = data_path / "audio" / "audio_base.fvecs"
        query_file = data_path / "audio" / "audio_query.fvecs"
        gt_file = data_path / "audio" / "audio_groundtruth.ivecs"
        graph_file = data_path / "deg" / "neighbor_choice" / "192D_L2_K20_AddK40Eps0.3Low_schemeA_2.deg"

        if not graph_file.is_file():
            create_graph(repository_file, DataStreamType.AddAll, graph_file, d=20, k_ext=40, eps_ext=0.3, k_opt=20, eps_opt=0.001, i_opt=5)
        test_graph(query_file, gt_file, graph_file, repeat=50, k=20)

    elif dataset_name == DatasetName.Enron:
        repository_file = data_path / "enron" / "enron_base.fvecs"
        query_file = data_path / "enron" / "enron_query.fvecs"
        gt_file = data_path / "enron" / "enron_groundtruth.ivecs"
        graph_file = data_path / "deg" / "neighbor_choice" / "1369D_L2_K30_AddK60Eps0.3High_schemeC.deg"

        if not graph_file.is_file():
            create_graph(repository_file, DataStreamType.AddAll, graph_file, d=30, k_ext=60, eps_ext=0.3, k_opt=30,
                         eps_opt=0.001, i_opt=5)
        test_graph(query_file, gt_file, graph_file, repeat=20, k=20)

    elif dataset_name == DatasetName.Sift1m:
        data_stream_type = DataStreamType.AddAllRemoveHalf
        repository_file = data_path / "SIFT1M" / "sift_base.fvecs"
        query_file = data_path / "SIFT1M" / "sift_query.fvecs"
        gt_file = (data_path / "SIFT1M" /
                   ("sift_groundtruth.ivecs" if
                    data_stream_type == DataStreamType.AddAll else
                    "sift_groundtruth_base500000.ivecs"))
        graph_file = (data_path / "deg" /
                      "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0"
                      "+0_improveEvery2ndNonPerfectEdge3.deg")

        if not graph_file.is_file():
            create_graph(repository_file, data_stream_type, graph_file, d=30, k_ext=60, eps_ext=0.2, k_opt=30, eps_opt=0.001, i_opt=5)
        test_graph(query_file, gt_file, graph_file, repeat=1, k=100)

    elif dataset_name == DatasetName.Glove:
        data_stream_type = DataStreamType.AddAll
        repository_file = data_path / "glove-100" / "glove-100_base.fvecs"
        query_file = data_path / "glove-100" / "glove-100_query.fvecs"
        gt_file = data_path / "glove-100" / (
            "glove-100_groundtruth.ivecs"
            if data_stream_type == DataStreamType.AddAll else "glove-100_groundtruth_base591757.ivecs")
        graph_file = data_path / "deg" / ("100D_L2_K40_AddK40Eps0.2High_SwapK40-0StepEps0.001LowPath5Rnd0"
                                          "+0_improveEvery2ndNonPerfectEdge.deg")

        if not graph_file.is_file():
            create_graph(repository_file, data_stream_type, graph_file, d=30, k_ext=30, eps_ext=0.2, k_opt=30,
                         eps_opt=0.001, i_opt=5)
        test_graph(query_file, gt_file, graph_file, repeat=1, k=100)


def create_graph(
        repository_file: pathlib.Path, data_stream_type: DataStreamType, graph_file: pathlib.Path, d: int, k_ext: int,
        eps_ext: float, k_opt: int, eps_opt: float, i_opt: int
):
    rnd = deglib.Mt19937()  # default 7
    metric = deglib.Metric.L2  # default metric
    swap_tries = 0  # additional swap tries between the next graph extension
    additional_swap_tries = 0  # increase swap try count for each successful swap
    # load data
    print("Load Data")
    repository = deglib.repository.fvecs_read(repository_file)
    # TODO: report actual mem usage
    print("Actual memory usage: {} Mb after loading data".format(get_current_rss_mb()))

    # create a new graph
    print("Setup empty graph with {} vertices in {}D feature space".format(repository.shape[0], repository.shape[1]))
    dims = repository.shape[1]
    max_vertex_count = repository.shape[0]
    feature_space = deglib.FloatSpace(dims, metric)
    graph = deglib.graph.SizeBoundedGraph(max_vertex_count, d, feature_space)
    # TODO: report actual mem usage
    print("Actual memory usage: {} Mb after setup empty graph".format(get_current_rss_mb()))

    # create a graph builder to add vertices to the new graph and improve its edges
    print("Start graph builder")
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, rnd, extend_k=k_ext, extend_eps=eps_ext, improve_k=k_opt, improve_eps=eps_opt, max_path_length=i_opt,
        swap_tries=swap_tries, additional_swap_tries=additional_swap_tries
    )

    # provide all features to the graph builder at once. In an online system this will be called multiple times
    base_size = repository.shape[0]

    def add_entry(label):
        feature = repository[label]
        # feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.add_entry(label, feature)

    if data_stream_type == DataStreamType.AddHalfRemoveAndAddOneAtATime:
        base_size_half = base_size // 2
        base_size_fourth = base_size // 4
        for i in range(base_size_fourth):
            add_entry(0 + i)
            add_entry(base_size_half + i)
        for i in range(base_size_fourth):
            add_entry(base_size_fourth + i)
            add_entry(base_size_half + base_size_fourth + i)
            builder.remove_entry(base_size_half + (i * 2) + 0)
            builder.remove_entry(base_size_half + (i * 2) + 1)

    else:
        base_size //= 2 if (data_stream_type == DataStreamType.AddHalf) else 1
        for i in range(base_size):
            add_entry(i)

        if data_stream_type == DataStreamType.AddAllRemoveHalf:
            for i in range(base_size//2, base_size):
                builder.remove_entry(i)

    del repository
    print("Actual memory usage: {} Mb after setup graph builder".format(get_current_rss_mb()))

    # check the integrity of the graph during the graph build process
    log_after = 100000

    print("Start building")
    start = time.perf_counter()
    duration = 0

    def improvement_callback(status):
        size = graph.size()
        nonlocal duration
        nonlocal start

        if status.step % log_after == 0 or size == base_size:
            duration += time.perf_counter() - start
            avg_edge_weight = deglib.analysis.calc_avg_edge_weight(graph, 1)
            weight_histogram_sorted = deglib.analysis.calc_edge_weight_histogram(graph, True, 1)
            weight_histogram = deglib.analysis.calc_edge_weight_histogram(graph, False, 1)
            valid_weights = deglib.analysis.check_graph_weights(graph) and deglib.analysis.check_graph_regularity(graph, size, True)
            connected = deglib.analysis.check_graph_connectivity(graph)

            print("{:7} vertices, {:6.3f}s, {:4} / {:4} improv, Q: {:.3f} -> Sorted:{}, InOrder:{}, {} connected & {}, RSS {}".format(
                    size, duration, status.improved, status.tries, avg_edge_weight,
                    " ".join(str(h) for h in weight_histogram_sorted), " ".join(str(h) for h in weight_histogram),
                    "" if connected else "not", "valid" if valid_weights else "invalid",
                    get_current_rss_mb()
                )
            )
            start = time.perf_counter()
        elif status.step % (log_after//10) == 0:
            duration += time.perf_counter() - start
            avg_edge_weight = deglib.analysis.calc_avg_edge_weight(graph, 1)
            connected = deglib.analysis.check_graph_connectivity(graph)

            print("{:7} vertices, {:6.3f}s, {:4} / {:4} improv, AEW: {:.3f}, {} connected, RSS {}".format(
                size, duration, status.improved, status.tries, avg_edge_weight,
                "" if connected else "not", get_current_rss_mb())
            )
            start = time.perf_counter()

    # start the build process
    stopwatch = StopWatch()
    builder.build(improvement_callback, False)
    duration = stopwatch.get_elapsed_time_micro() / 1000000
    print("Actual memory usage: {} Mb after building the graph in {:.4} secs".format(
        get_current_rss_mb(), float(duration)
    ))

    # store the graph
    graph.save_graph(graph_file)

    print("The graph contains {} non-RNG edges".format(deglib.analysis.calc_non_rng_edges(graph)))


def test_graph(query_file: pathlib.Path, gt_file: pathlib.Path, graph_file: pathlib.Path, repeat: int, k: int):
    # load an existing graph
    print("Load graph {}".format(graph_file))
    graph = deglib.graph.load_readonly_graph(graph_file)
    print("Actual memory usage: {} Mb after loading the graph".format(get_current_rss_mb()))

    query_repository = deglib.repository.fvecs_read(query_file)
    print("{} Query Features with {} dimensions".format(query_repository.shape[0], query_repository.shape[1]))

    ground_truth = deglib.repository.ivecs_read(gt_file)
    print("{} ground truth {} dimensions".format(ground_truth.shape[0], ground_truth.shape[1]))

    deglib.benchmark.test_graph_anns(graph, query_repository, ground_truth, repeat, k)


if __name__ == '__main__':
    main()
