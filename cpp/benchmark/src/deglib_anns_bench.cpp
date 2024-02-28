#include <random>

#include <fmt/core.h>
#include <cmath>

#include "benchmark.h"
#include "deglib.h"

int main(int argc, char *argv[]) {
    fmt::print("Testing ...\n");

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif


    const auto data_path = std::filesystem::path(DATA_PATH);
    uint32_t k = 100; 
    uint32_t repeat_test = 1;

    // ------------------------------------------ SIFT1M ---------------------------------------------
    const auto graph_file   = (data_path / "crEG" / "sift_128D_L2_crEG30.deg").string(); 
    const auto gt_file      = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();
    const auto query_file   = (data_path / "SIFT1M" / "sift_query.fvecs").string();

    // ------------------------------------------ Glove ---------------------------------------------
    // const auto graph_file   = (data_path / "crEG" / "glove_100D_L2_crEG30.deg").string(); 
    // const auto gt_file      = (data_path / "glove-100" / "glove-100_groundtruth.ivecs").string();
    // const auto query_file   = (data_path / "glove-100" / "glove-100_query.fvecs").string();

    // ------------------------------------------ Deep1M ---------------------------------------------
    // const auto graph_file    = (data_path / "crEG" / "deep1m_96D_L2_crEG30.deg").string(); 
    // const auto gt_file       = (data_path / "deep1m" / "deep1m_groundtruth.ivecs").string();
    // const auto query_file    = (data_path / "deep1m" / "deep1m_query.fvecs").string();

    // ------------------------------------------ Audio ---------------------------------------------
    // const auto graph_file    = (data_path / "crEG" / "audio_192D_L2_crEG20.deg").string(); 
    // const auto gt_file       = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const auto query_file    = (data_path / "audio" / "audio_query.fvecs").string();

    fmt::print("Load graph {} \n", graph_file);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    fmt::print("Graph with {} vertices \n", graph.size());
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);

    // load the test data and run several ANNS on the graph   
    const auto query_repository = deglib::load_static_repository(query_file.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

    size_t ground_truth_dims;
    size_t ground_truth_count;
    const auto ground_truth_f = deglib::fvecs_read(gt_file.c_str(), ground_truth_dims, ground_truth_count);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

    fmt::print("Test with k={} and repeat_test={}\n", k, repeat_test);
    deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t) ground_truth_dims, repeat_test, k);
  
    fmt::print("Test OK\n");
    return 0;
}