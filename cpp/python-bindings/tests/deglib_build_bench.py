import pathlib
import random
import sys
import enum

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

    dataset_name = DatasetName.Sift1m
    if dataset_name == DatasetName.Audio:
        repository_file = data_path / "audio" / "audio_base.fvecs"
        query_file = data_path / "enron" / "enron_query.fvecs"
        gt_file = data_path / "enron" / "enron_groundtruth.ivecs"
        graph_file = data_path / "deg" / "neighbor_choice" / "192D_L2_K20_AddK40Eps0.3Low_schemeA.deg"

        if not graph_file.is_file():
            create_graph(repository_file, DataStreamType.AddAll, graph_file, d=20, k_ext=40, eps_ext=0.3, k_opt=20,
                         eps_opt=0.001, i_opt=5)
        test_graph(query_file, gt_file, graph_file, repeat=50, k=20)

    elif dataset_name == DatasetName.Enron:
        repository_file = data_path / "enron" / "enron_base.fvecs"
        query_file = data_path / "audio" / "audio_query.fvecs"
        gt_file = data_path / "audio" / "audio_groundtruth.ivecs"
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
            create_graph(repository_file, data_stream_type, graph_file, d=30, k_ext=60, eps_ext=0.2, k_opt=30,
                         eps_opt=0.001, i_opt=5)
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
    rnd = random.Random(7)  # default 7
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
    # graph = deglib.graph.SizeBoundedGraph(max_vertex_count, d, feature_space)
    # TODO: report actual mem usage
    print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph".format(0, 0))

    """
    // create a graph builder to add vertices to the new graph and improve its edges
    fmt::print("Start graph builder \n");   
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries);
    
    // provide all features to the graph builder at once. In an online system this will be called multiple times
    auto base_size = uint32_t(repository.size());
    auto addEntry = [&builder, &repository, dims] (auto label)
    {
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(label, std::move(feature_vector));
    };
    if(data_stream_type == AddHalfRemoveAndAddOneAtATime) {
        auto base_size_half = base_size / 2;
        auto base_size_fourth = base_size / 4;
        for (uint32_t i = 0; i < base_size_fourth; i++) { 
            addEntry(0 + i);
            addEntry(base_size_half + i);
        }
        for (uint32_t i = 0; i < base_size_fourth; i++) { 
            addEntry(base_size_fourth + i);
            addEntry(base_size_half + base_size_fourth + i);
            builder.removeEntry(base_size_half + (i * 2) + 0);
            builder.removeEntry(base_size_half + (i * 2) + 1);
        }
    } else {
        base_size /= (data_stream_type == AddHalf) ? 2 : 1;
        for (uint32_t i = 0; i < base_size; i++) 
            addEntry(i);

        if(data_stream_type == AddAllRemoveHalf) 
            for (uint32_t i = base_size/2; i < base_size; i++) 
                builder.removeEntry(i);
    }
    repository.clear();
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup graph builder\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // check the integrity of the graph during the graph build process
    const auto log_after = 100000;

    fmt::print("Start building \n");    
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        if(status.step % log_after == 0 || size == base_size) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
            auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true, 1);
            auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false, 1);
            auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:5}s, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                        size, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
        else if(status.step % (log_after/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
                        auto connected = deglib::analysis::check_graph_connectivity(graph);

            auto duration = duration_ms / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:5}s, {:8} / {:8} improv, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}\n", size, duration, status.improved, status.tries, avg_edge_weight, connected ? "" : "not", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
    };

    // start the build process
    builder.build(improvement_callback, false);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000, duration_ms / 1000);

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
    """


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
