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
    uint32_t test_threads = 1;

    // SIFT1M
    const auto graph_file   = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    const auto query_file   = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    const auto gt_file      = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();

    fmt::print("Load graph {} \n", graph_file);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
    StopW stopW;
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    long long int elapsed_us = stopW.getElapsedTimeMicro();
    fmt::print("Graph with {} vertices \n", graph.size());
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
    fmt::print("Loading Graph took {} us\n", elapsed_us);

    // load the test data and run several ANNS on the graph   
    const auto query_repository = deglib::load_static_repository(query_file.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

    size_t ground_truth_dims;
    size_t ground_truth_count;
    const auto ground_truth_f = deglib::fvecs_read(gt_file.c_str(), ground_truth_dims, ground_truth_count);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

    fmt::print("Test with k={} and repeat_test={}\n", k, repeat_test);
    deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t) ground_truth_dims, repeat_test, test_threads, k);
  
    fmt::print("Test OK\n");
    return 0;
}