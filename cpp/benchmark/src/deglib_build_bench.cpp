#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"

/**
 * The last three types are used to reproduce the deletion tests in our paper
 */
enum DataStreamType { AddAll, AddHalf, AddAllRemoveHalf, AddHalfRemoveAndAddOneAtATime };


/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void create_graph(const std::string repository_file, const DataStreamType data_stream_type, const std::string graph_file, const uint8_t d, const uint8_t k_ext, const float eps_ext, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt) {
    
    auto rnd = std::mt19937(7);                         // default 7
    const deglib::Metric metric = deglib::Metric::L2;   // defaul metric
    const uint32_t swap_tries = 0;                      // additional swap tries between the next graph extension
    const uint32_t additional_swap_tries = 0;           // increse swap try count for each successful swap

    // load data
    fmt::print("Load Data \n");
    auto repository = deglib::load_static_repository(repository_file.c_str());   
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a new graph
    fmt::print("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, d, feature_space);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

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
}

/**
 * Load the graph from the drive and test it against the SIFT query data.
 */
void test_graph(const std::string query_file, const std::string gt_file, const std::string graph_file, const uint32_t repeat, const uint32_t k) {

    // load an existing graph
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
}

int main() {

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb \n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    omp_set_num_threads(8);
    std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;

    const auto data_path = std::filesystem::path(DATA_PATH);
  
    // // ------------------------------- audio -----------------------------------------
    // const auto repository_file      = (data_path / "audio" / "audio_base.fvecs").string();
    // const auto query_file           = (data_path / "audio" / "audio_query.fvecs").string();
    // const auto gt_file              = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "neighbor_choice" / "192D_L2_K20_AddK40Eps0.3Low_schemeA.deg").string();

    // if(std::filesystem::exists(graph_file.c_str()) == false)
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, 20, 40, 0.3f, 20, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    // test_graph(query_file, gt_file, graph_file, 50, 20); // repeat_test, k



    // // ------------------------------- enron -----------------------------------------
    // const auto repository_file      = (data_path / "enron" / "enron_base.fvecs").string();
    // const auto query_file           = (data_path / "enron" / "enron_query.fvecs").string();
    // const auto gt_file              = (data_path / "enron" / "enron_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "neighbor_choice" / "1369D_L2_K30_AddK60Eps0.3High_schemeC.deg").string();

    // if(std::filesystem::exists(graph_file.c_str()) == false)
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, 30, 60, 0.3f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    // test_graph(query_file, gt_file, graph_file, 20, 20); // repeat_test, k



    // ------------------------------- SIFT1M -----------------------------------------
    const auto data_stream_type     = DataStreamType::AddAllRemoveHalf;
    const auto repository_file      = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    const auto query_file           = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    const auto gt_file              = (data_path / "SIFT1M" / (data_stream_type == AddAll ? "sift_groundtruth.ivecs" : "sift_groundtruth_base500000.ivecs" )).string();
    const auto graph_file           = (data_path / "deg" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge3.deg").string();

    if(std::filesystem::exists(graph_file.c_str()) == false)
        create_graph(repository_file, data_stream_type, graph_file, 30, 60, 0.2f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    test_graph(query_file, gt_file, graph_file, 1, 100); // repeat_test, k



    // // ------------------------------- GLOVE -----------------------------------------
    // const auto data_stream_type = DataStreamType::AddAll;
    // const auto repository_file  = (data_path / "glove-100" / "glove-100_base.fvecs").string();    
    // const auto query_file       = (data_path / "glove-100" / "glove-100_query.fvecs").string();
    // const auto gt_file          = (data_path / "glove-100" / (data_stream_type == AddAll ? "glove-100_groundtruth.ivecs" : "glove-100_groundtruth_base591757.ivecs" )).string();
    // const auto graph_file       = (data_path / "deg" / "100D_L2_K40_AddK40Eps0.2High_SwapK40-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();

    // if(std::filesystem::exists(graph_file.c_str()) == false)
    //     create_graph(repository_file, data_stream_type, graph_file, 30, 30, 0.2f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    // test_graph(query_file, gt_file, graph_file, 1, 100); // repeat_test, k

 

    fmt::print("Test OK\n");
    return 0;
}
