#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"


/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void opt_graph(const std::string initial_graph_file, const std::string graph_file, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t iterations) {
    
    auto rnd = std::mt19937(7);                         // default 7
    const deglib::Metric metric = deglib::Metric::L2;   // defaul metric

    fmt::print("Load graph {} \n", initial_graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    fmt::print("Graph with {} vertices and an avg edge weight of {} \n", graph.size(), deglib::analysis::calc_avg_edge_weight(graph, 1));

    // create a graph builder to add vertices to the new graph and improve its edges
    fmt::print("Start graph builder \n");   
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, 0, 0, false, k_opt, eps_opt, i_opt, 1, 0);
    
    // check the integrity of the graph during the graph build process
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        if(status.step % (iterations/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);

            auto duration = duration_ms / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} step, {:5}s, AEW: {:4.2f}, {} connected, {}, RSS {} & peakRSS {}\n", status.step, duration, avg_edge_weight, connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }

        if(status.step > iterations)
            builder.stop();
    };

    // start the build process
    builder.build(improvement_callback, true);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000, duration_ms / 1000);

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}

/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void create_graph(const std::string repository_file, const std::string graph_file, const uint8_t d, const uint8_t k_ext, const float eps_ext, const bool schemeC) {
    
    auto rnd = std::mt19937(7);                         // default 7
    const deglib::Metric metric = deglib::Metric::L2;   // defaul metric

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
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, k_ext, eps_ext, schemeC, 0, 0, 0, 0, 0);
    
    // provide all features to the graph builder at once. In an online system this will be called multiple times
    auto base_size = uint32_t(repository.size());
    for (uint32_t i = 0; i < base_size; i++) {
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(i));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(i, std::move(feature_vector));
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
            fmt::print("{:7} vertices, {:5}s, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                        size, duration, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
        else if(status.step % (log_after/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
                        auto connected = deglib::analysis::check_graph_connectivity(graph);

            auto duration = duration_ms / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:5}s, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}\n", size, duration, avg_edge_weight, connected ? "" : "not", currRSS, peakRSS);
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

    const auto data_path = std::filesystem::path(DATA_PATH);
  


    // // ------------------------------- SIFT1M -----------------------------------------
    // const auto repository_file      = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    // const auto query_file           = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    // const auto gt_file              = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();
    // const uint32_t repeat_test      = 1;

    // const auto eg_graph_file        = (data_path / "crEG" / "sift_128D_L2_EG30.deg").string();
    // const uint8_t d                 = 30;
    // const uint8_t k_ext             = 60;
    // const float eps_ext             = 0.1f;
    // const bool schemeC              = false;

    // const auto creg_graph_file      = (data_path / "crEG" / "sift_128D_L2_crEG30.deg").string();
    // const uint8_t k_opt             = 30;
    // const float eps_opt             = 0.001f;
    // const uint8_t i_opt             = 5;
    // const uint32_t iterations       = 200000;


    // // ------------------------------- GLOVE -----------------------------------------
    // const auto repository_file      = (data_path / "glove-100" / "glove-100_base.fvecs").string();
    // const auto query_file           = (data_path / "glove-100" / "glove-100_query.fvecs").string();
    // const auto gt_file              = (data_path / "glove-100" / "glove-100_groundtruth.ivecs").string();
    // const uint32_t repeat_test      = 1;
    
    // const auto eg_graph_file        = (data_path / "crEG" / "glove_100D_L2_EG30.deg").string();
    // const uint8_t d                 = 30;
    // const uint8_t k_ext             = 60;
    // const float eps_ext             = 0.1f;
    // const bool schemeC              = false;

    // const auto creg_graph_file      = (data_path / "crEG" / "glove_100D_L2_crEG30.deg").string();
    // const uint8_t k_opt             = 30;
    // const float eps_opt             = 0.001f;
    // const uint8_t i_opt             = 5;
    // const uint32_t iterations       = 2000000;


    // ------------------------------- Deep1M -----------------------------------------
    const auto repository_file      = (data_path / "deep1m" / "deep1m_base.fvecs").string();
    const auto query_file           = (data_path / "deep1m" / "deep1m_query.fvecs").string();
    const auto gt_file              = (data_path / "deep1m" / "deep1m_groundtruth.ivecs").string();
    const uint32_t repeat_test      = 1;

    const auto eg_graph_file        = (data_path / "crEG" / "deep1m_96D_L2_EG30.deg").string();
    const uint8_t d                 = 30;
    const uint8_t k_ext             = 60;
    const float eps_ext             = 0.1f;
    const bool schemeC              = false;

    const auto creg_graph_file      = (data_path / "crEG" / "deep1m_96D_L2_crEG30.deg").string();
    const uint8_t k_opt             = 30;
    const float eps_opt             = 0.001f;
    const uint8_t i_opt             = 5;
    const uint32_t iterations       = 400000;


    // // ------------------------------- audio -----------------------------------------
    // const auto repository_file      = (data_path / "audio" / "audio_base.fvecs").string();
    // const auto query_file           = (data_path / "audio" / "audio_query.fvecs").string();
    // const auto gt_file              = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const uint32_t repeat_test      = 100;

    // const auto eg_graph_file        = (data_path / "crEG" / "audio_192D_L2_EG20.deg").string();
    // const uint8_t d                 = 20;
    // const uint8_t k_ext             = 40;
    // const float eps_ext             = 0.1f;
    // const bool schemeC              = false;

    // const auto creg_graph_file      = (data_path / "crEG" / "audio_192D_L2_crEG20.deg").string();
    // const uint8_t k_opt             = 20;
    // const float eps_opt             = 0.001f;
    // const uint8_t i_opt             = 5;
    // const uint32_t iterations       = 20000;


    // build, optimize and test
    if(std::filesystem::exists(eg_graph_file.c_str()) == false)
        create_graph(repository_file, eg_graph_file, d, k_ext, eps_ext, schemeC); 
    test_graph(query_file, gt_file, eg_graph_file, repeat_test, 100); // repeat_test, k

    if(std::filesystem::exists(creg_graph_file.c_str()) == false)
        opt_graph(eg_graph_file, creg_graph_file, k_opt, eps_opt, i_opt, iterations); 
    test_graph(query_file, gt_file, creg_graph_file, repeat_test, 100); // repeat_test, k

    fmt::print("Test OK\n");
    return 0;
}
