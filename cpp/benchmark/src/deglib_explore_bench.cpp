#include <random>

#include <fmt/core.h>

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

   
    const uint32_t repeat_test = 1;
    const auto data_path = std::filesystem::path(DATA_PATH);
    
    const uint32_t k = 1000; 

    // ------------------------------------------ SIFT1M ---------------------------------------------
    const auto graph_file               = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string(); 
    const auto gt_file   = (data_path / "SIFT1M" / "sift_explore_ground_truth.ivecs").string();
    const auto query_file        = (data_path / "SIFT1M" / "sift_explore_entry_vertex.ivecs").string();

    // ------------------------------------------ Glove ---------------------------------------------
    // const auto graph_file               = (data_path / "deg" / "100D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string(); 
    // const auto gt_file   = (data_path / "glove-100" / "glove-100_explore_ground_truth.ivecs").string();
    // const auto query_file        = (data_path / "glove-100" / "glove-100_explore_entry_vertex.ivecs").string();

    // ------------------------------------------ Enron ---------------------------------------------
    // const auto graph_file               = (data_path / "deg" / "1369D_L2_K30_AddK60Eps0.3High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string(); 
    // const auto gt_file   = (data_path / "enron" / "enron_explore_ground_truth.ivecs").string();
    // const auto query_file        = (data_path / "enron" / "enron_explore_entry_vertex.ivecs").string();

    // ------------------------------------------ Audio ---------------------------------------------
    // const auto graph_file               = (data_path / "deg" / "192D_L2_K20_AddK40Eps0.3High_SwapK20-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string(); 
    // const auto gt_file   = (data_path / "audio" / "audio_explore_ground_truth.ivecs").string();
    // const auto query_file        = (data_path / "audio" / "audio_explore_entry_vertex.ivecs").string();
    
    // load graph
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());

    // load starting vertex data
    size_t entry_vertex_dims;
    size_t entry_vertex_count;
    const auto entry_vertex_f = deglib::fvecs_read(query_file.c_str(), entry_vertex_dims, entry_vertex_count);
    const auto entry_vertex = (uint32_t*)entry_vertex_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} entry vertex {} dimensions \n", entry_vertex_count, entry_vertex_dims);

    // load ground truth data (nearest neighbors of the starting vertices)
    size_t ground_truth_dims;
    size_t ground_truth_count;
    const auto ground_truth_f = deglib::fvecs_read(gt_file.c_str(), ground_truth_dims, ground_truth_count);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

    // explore the graph
    deglib::benchmark::test_graph_explore(graph, (uint32_t) ground_truth_count, ground_truth, (uint32_t) ground_truth_dims, entry_vertex, (uint32_t) entry_vertex_dims, repeat_test, k);

    fmt::print("Test OK\n");
    return 0;
}