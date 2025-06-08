#include <random>

#include <fmt/core.h>
#include <cmath>
#include <vector>
#include <chrono>

#include "benchmark.h"
#include "deglib.h"



void filter_test() {
        constexpr size_t max_value = 999'999;
    constexpr size_t max_id_count = 1'000'000;
    std::vector<int> valid_ids;

    // Pre-generate random query IDs for is_valid tests
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, static_cast<int>(max_value));
    size_t num_queries = 100'000;
    std::vector<int> query_ids(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        query_ids[i] = dist(rng);
    }

    for (size_t fill_rate = 0; fill_rate <= 1'000'000; fill_rate += 100'000) {
        valid_ids.clear();
        valid_ids.reserve(fill_rate);
        for (size_t i = 0; i < fill_rate; ++i) {
            valid_ids.push_back(static_cast<int>(i));
        }

        fmt::print("\nTesting with fill rate: {} IDs\n", fill_rate);

        // Measure the time for Filter creation
        auto start_creation = std::chrono::high_resolution_clock::now();
        deglib::graph::Filter filter(valid_ids.data(), valid_ids.size(), max_value, max_id_count);
        auto end_creation = std::chrono::high_resolution_clock::now();

        double creation_time = std::chrono::duration<double, std::milli>(end_creation - start_creation).count();
        fmt::print("Filter creation time: {:.3f} ms\n", creation_time);

        // Measure the time for for_each_valid_id
        auto start_retrieval = std::chrono::high_resolution_clock::now();
        uint64_t retrieved_ids_count = 0;
        filter.for_each_valid_label([&retrieved_ids_count](uint32_t valid_label) {
            retrieved_ids_count++;
        });
        auto end_retrieval = std::chrono::high_resolution_clock::now();

        double retrieval_time = std::chrono::duration<double, std::milli>(end_retrieval - start_retrieval).count();
        fmt::print("for_each_valid_id time: {:.3f} ms, retrieved_ids_count: {}\n", retrieval_time, retrieved_ids_count);

        // Perform 100k random is_valid checks using pre-generated query IDs
        size_t valid_count = 0;
        auto start_is_valid = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_queries; ++i) {
            if (filter.is_valid(query_ids[i])) {
                valid_count++;
            }
        }
        auto end_is_valid = std::chrono::high_resolution_clock::now();

        double is_valid_time = std::chrono::duration<double, std::milli>(end_is_valid - start_is_valid).count();
        fmt::print("is_valid time for 100k queries: {:.3f} ms\n", is_valid_time);
        fmt::print("Number of valid IDs in queries: {}\n", valid_count);

        fmt::print("Number of valid IDs: {}\n", filter.size());
        fmt::print("Inclusion rate: {:.6f}\n", filter.get_inclusion_rate());
    }

}

int main(int argc, char *argv[]) {
    fmt::print("Testing ...\n");

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif

    // filter_test();

    const auto data_path = std::filesystem::path(DATA_PATH);
    uint32_t k = 100; 
    uint32_t repeat_test = 1;
    uint32_t test_threads = 1;

    // // SIFT1M
    // const auto graph_file   = (data_path / "deg" / "schemes" / "128D_L2_K30_AddK60Eps0.1_schemeUnkown.deg").string();
    // const auto query_file   = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    // const auto gt_file      = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();

    // SIFT1000K
    const auto graph_file   = (data_path / "deg" / "128D_L2_K16_AddK32Eps0.1_schemeLow.deg").string();
    const auto query_file   = (data_path / "SIFT100K" / "sift_query.fvecs").string();
    const auto gt_file      = (data_path / "SIFT100K" / "sift_groundtruth.ivecs").string();

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

    // create filter    
    auto graph_size = graph.size();
    auto valid_label_rate = graph_size; // graph_size=all allowed
    auto fill_rate = static_cast<size_t>(graph_size * (valid_label_rate / 100));
    auto step_size = static_cast<double>(graph_size) / fill_rate;
    std::vector<int> valid_labels;
    valid_labels.reserve(fill_rate);
    for (double i = 0; i < graph_size; i += step_size) {
        valid_labels.push_back(static_cast<int>(i));
    }
    deglib::graph::Filter filter(valid_labels.data(), valid_labels.size(), graph_size-1, graph_size);
    fmt::print("{} valid label in filter, {} valid labels, {} fill_rate, {} step_size \n", filter.size(), valid_labels.size(), fill_rate, step_size);

    fmt::print("Test with k={} and repeat_test={}\n", k, repeat_test);
    // deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t) ground_truth_dims, repeat_test, test_threads, k, &filter);
    deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t) ground_truth_dims, repeat_test, test_threads, k);
  
    fmt::print("Test OK\n");
    return 0;
}