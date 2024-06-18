import pathlib
import sys
import enum
import time

import deglib


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

    print("Actual memory usage: {} Mb, Max memory usage: {} Mb".format(0, 0))

    # omp_set_num_threads(8);
    # std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;

    data_path: pathlib.Path = pathlib.Path(sys.argv[1])

    dataset_name = DatasetName.Audio
    if dataset_name == DatasetName.Audio:
        repository_file = data_path / "audio" / "audio_base.fvecs"
        query_file = data_path / "audio" / "audio_query.fvecs"
        gt_file = data_path / "audio" / "audio_groundtruth.ivecs"
        graph_file = data_path / "deg" / "neighbor_choice" / "192D_L2_K20_AddK40Eps0.3Low_schemeA.deg"

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
    repository = deglib.load_static_repository(repository_file)
    # TODO: report actual mem usage
    print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", 0, 0)

    # create a new graph
    print("Setup empty graph with {} vertices in {}D feature space".format(repository.size(), repository.dims()))
    dims = repository.dims()
    max_vertex_count = repository.size()
    feature_space = deglib.FloatSpace(dims, metric)
    graph = deglib.graph.SizeBoundedGraph(max_vertex_count, d, feature_space)
    # TODO: report actual mem usage
    print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph".format(0, 0))

    # create a graph builder to add vertices to the new graph and improve its edges
    print("Start graph builder")
    builder = deglib.builder.EvenRegularGraphBuilder(
        graph, rnd, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries
    )

    # provide all features to the graph builder at once. In an online system this will be called multiple times
    base_size = repository.size()

    def add_entry(label):
        feature = repository.get_feature(label)
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

    repository.clear()
    print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup graph builder".format(0, 0))

    # check the integrity of the graph during the graph build process
    log_after = 100000

    print("Start building")
    start = time.perf_counter()
    duration_ms = 0  # TODO: rename to duration

    def improvement_callback(status):
        size = graph.size()
        nonlocal duration_ms
        nonlocal start

        if status.step % log_after == 0 or size == base_size:
            duration_ms += time.perf_counter() - start
            avg_edge_weight = deglib.analysis.calc_avg_edge_weight(graph, 1)
            weight_histogram_sorted = deglib.analysis.calc_edge_weight_histogram(graph, True, 1)
            weight_histogram = deglib.analysis.calc_edge_weight_histogram(graph, False, 1)
            valid_weights = deglib.analysis.check_graph_weights(graph) and deglib.analysis.check_graph_regularity(graph, size, True)
            connected = deglib.analysis.check_graph_connectivity(graph)
            # duration = duration_ms / 1000
            currRSS = 0
            peakRSS = 0
            print("{:7} vertices, {:5}s, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{}, InOrder:{}, {} connected & {}, RSS {} & peakRSS {}".format(
                    size, duration_ms, status.improved, status.tries, avg_edge_weight,
                    " ".join(str(h) for h in weight_histogram_sorted), " ".join(str(h) for h in weight_histogram),
                    "" if connected else "not", "valid" if valid_weights else "invalid",
                    currRSS, peakRSS
                )
            )
            start = time.perf_counter()
        elif status.step % (log_after//10) == 0:
            duration_ms += time.perf_counter() - start
            avg_edge_weight = deglib.analysis.calc_avg_edge_weight(graph, 1)
            connected = deglib.analysis.check_graph_connectivity(graph)

            duration = duration_ms
            currRSS = 0
            peakRSS = 0
            print("{:7} vertices, {:5}s, {:8} / {:8} improv, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}".format(size, duration, status.improved, status.tries, avg_edge_weight, "" if connected else "not", currRSS, peakRSS))
            start = time.perf_counter()

    # start the build process
    builder.build(improvement_callback, False)
    print("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs".format(0, 0, duration_ms))

    # store the graph
    graph.save_graph(graph_file)

    print("The graph contains {} non-RNG edges".format(deglib.analysis.calc_non_rng_edges(graph)))


def test_graph(query_file: pathlib.Path, gt_file: pathlib.Path, graph_file: pathlib.Path, repeat: int, k: int):
    print('test_graph')
    # load an existing graph
    """
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading the graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    const auto query_repository = deglib::load_static_repository(query_file.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

    size_t dims_out;
    size_t count_out;
    const auto ground_truth_f = deglib::fvecs_read(gt_file.c_str(), dims_out, count_out);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", count_out, dims_out);

    deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t)dims_out, repeat, k);
    """


if __name__ == '__main__':
    main()
