#include "common/test_helpers.h"
#include <span>

// ============================================================================
// ReadOnlyGraph Search & Explore Regression Benchmarks (100x averaged)
// ============================================================================
// Measures average throughput (QPS), latency (ms), and recall for search()
// and explore() across 100 iterations on ReadOnlyGraph instances.
// ============================================================================

TEST(ReadOnlyGraphRegression, SearchAndExplore_FP32_L2)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;
    const int benchmark_runs = 100;
    const uint32_t edges_per_vertex = 32;
    const float extend_eps = 0.1f;
    const float search_eps = 0.05f;
    const uint32_t search_k = 10;
    const uint32_t explore_max_calcs = 2000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, search_k);

    deglib::FloatSpace feature_space(dim, deglib::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph mutable_graph(static_cast<uint32_t>(base_count), edges_per_vertex, feature_space);

    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(
        mutable_graph, rnd, deglib::builder::OptimizationTarget::LowLID, edges_per_vertex, extend_eps, 0, 0.0f
    );
    builder.setThreadCount(1);

    for (size_t i = 0; i < base_count; ++i) {
        std::vector<std::byte> feat(dim * sizeof(float));
        std::memcpy(feat.data(), &base_data[i * dim], dim * sizeof(float));
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat));
    }
    auto dummy_callback = [](deglib::builder::BuilderStatus&) {};
    builder.build(dummy_callback, false);

    // Convert to ReadOnlyGraph
    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), edges_per_vertex, feature_space, mutable_graph);

    // 1. Search Benchmark (100x runs)
    uint32_t correct_search = 0;
    uint32_t total_gt = 0;
    auto t_start_search = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t q = 0; q < query_count; ++q) {
            auto results = graph.search(std::span<const float>(&query_data[q * dim], dim), search_k, search_eps);

            if (run == 0) {
                std::unordered_set<uint32_t> gt_set(gt_data[q].begin(), gt_data[q].end());
                total_gt += static_cast<uint32_t>(gt_set.size());
                while (!results.empty()) {
                    uint32_t ext_label = graph.getExternalLabel(results.top().getInternalIndex());
                    if (gt_set.count(ext_label)) {
                        correct_search++;
                    }
                    results.pop();
                }
            }
        }
    }
    auto t_end_search = std::chrono::high_resolution_clock::now();
    double total_search_ms = std::chrono::duration<double, std::milli>(t_end_search - t_start_search).count();
    size_t total_queries = query_count * benchmark_runs;
    double search_qps = (static_cast<double>(total_queries) / total_search_ms) * 1000.0;
    float search_recall = static_cast<float>(correct_search) / static_cast<float>(total_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_L2 search(): "
              << total_search_ms << " ms total for " << total_queries << " queries (" 
              << (total_search_ms / total_queries) << " ms/q), "
              << search_qps << " QPS, recall=" << (search_recall * 100.0f) << "%\n";

    // 2. Explore Benchmark (100x runs)
    size_t explore_count_per_run = 1000;
    size_t total_explorations = explore_count_per_run * benchmark_runs;
    uint32_t correct_explore = 0;
    uint32_t total_explore_gt = 0;
    auto t_start_explore = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < benchmark_runs/10; ++run) {
        for (size_t i = 0; i < explore_count_per_run; ++i) {
            uint32_t entry_node = static_cast<uint32_t>((run * 13 + i) % base_count);
            auto results = graph.explore(entry_node, search_k, explore_max_calcs);

            if (run == 0) {
                // Compute ground truth for this entry_node vector
                const float* entry_vec = &base_data[entry_node * dim];
                std::vector<std::pair<float, uint32_t>> dists(base_count);
                for (size_t b = 0; b < base_count; ++b) {
                    float d = deglib::distances::fp32_l2::L2Float::compare(entry_vec, &base_data[b * dim], &dim);
                    dists[b] = {d, static_cast<uint32_t>(b)};
                }
                std::partial_sort(dists.begin(), dists.begin() + search_k, dists.end());
                std::unordered_set<uint32_t> gt_set;
                for (size_t k = 0; k < search_k; ++k) gt_set.insert(dists[k].second);
                total_explore_gt += static_cast<uint32_t>(gt_set.size());

                while (!results.empty()) {
                    uint32_t ext_label = graph.getExternalLabel(results.top().getInternalIndex());
                    if (gt_set.count(ext_label)) {
                        correct_explore++;
                    }
                    results.pop();
                }
            }
        }
    }
    auto t_end_explore = std::chrono::high_resolution_clock::now();
    double total_explore_ms = std::chrono::duration<double, std::milli>(t_end_explore - t_start_explore).count();
    double explore_qps = (static_cast<double>(total_explorations) / total_explore_ms) * 1000.0;
    float explore_recall = static_cast<float>(correct_explore) / static_cast<float>(total_explore_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_L2 explore(): "
              << total_explore_ms << " ms total for " << total_explorations << " explorations (" 
              << (total_explore_ms / total_explorations) << " ms/q), "
              << explore_qps << " QPS, recall=" << (explore_recall * 100.0f) << "%\n";
}

TEST(ReadOnlyGraphRegression, SearchAndExplore_FP32_InnerProduct)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;
    const int benchmark_runs = 100;
    const uint32_t edges_per_vertex = 32;
    const float extend_eps = 0.1f;
    const float search_eps = 0.05f;
    const uint32_t search_k = 10;
    const uint32_t explore_max_calcs = 2000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, search_k);

    deglib::FloatSpace feature_space(dim, deglib::Metric::FP32_InnerProduct);
    deglib::graph::SizeBoundedGraph mutable_graph(static_cast<uint32_t>(base_count), edges_per_vertex, feature_space);

    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(
        mutable_graph, rnd, deglib::builder::OptimizationTarget::LowLID, edges_per_vertex, extend_eps, 0, 0.0f
    );
    builder.setThreadCount(1);

    for (size_t i = 0; i < base_count; ++i) {
        std::vector<std::byte> feat(dim * sizeof(float));
        std::memcpy(feat.data(), &base_data[i * dim], dim * sizeof(float));
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat));
    }
    auto dummy_callback = [](deglib::builder::BuilderStatus&) {};
    builder.build(dummy_callback, false);

    // Convert to ReadOnlyGraph
    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), edges_per_vertex, feature_space, mutable_graph);

    // Search Benchmark (100x runs)
    uint32_t correct_search = 0;
    uint32_t total_gt = 0;
    size_t total_queries = query_count * benchmark_runs;
    auto t_start_search = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t q = 0; q < query_count; ++q) {
            auto results = graph.search(std::span<const float>(&query_data[q * dim], dim), search_k, search_eps);

            if (run == 0) {
                std::unordered_set<uint32_t> gt_set(gt_data[q].begin(), gt_data[q].end());
                total_gt += static_cast<uint32_t>(gt_set.size());
                while (!results.empty()) {
                    uint32_t ext_label = graph.getExternalLabel(results.top().getInternalIndex());
                    if (gt_set.count(ext_label)) {
                        correct_search++;
                    }
                    results.pop();
                }
            }
        }
    }
    auto t_end_search = std::chrono::high_resolution_clock::now();
    double total_search_ms = std::chrono::duration<double, std::milli>(t_end_search - t_start_search).count();
    double search_qps = (static_cast<double>(total_queries) / total_search_ms) * 1000.0;
    float search_recall = static_cast<float>(correct_search) / static_cast<float>(total_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_InnerProduct search(): "
              << total_search_ms << " ms total (" << (total_search_ms / total_queries) << " ms/q), "
              << search_qps << " QPS, recall=" << (search_recall * 100.0f) << "%\n";

    // Explore Benchmark (100x runs)
    size_t explore_count_per_run = 1000;
    size_t total_explorations = explore_count_per_run * benchmark_runs;
    uint32_t correct_explore = 0;
    uint32_t total_explore_gt = 0;
    auto t_start_explore = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < benchmark_runs/10; ++run) {
        for (size_t i = 0; i < explore_count_per_run; ++i) {
            uint32_t entry_node = static_cast<uint32_t>((run * 13 + i) % base_count);
            auto results = graph.explore(entry_node, search_k, explore_max_calcs);

            if (run == 0) {
                // Compute ground truth for this entry_node vector
                const float* entry_vec = &base_data[entry_node * dim];
                std::vector<std::pair<float, uint32_t>> dists(base_count);
                for (size_t b = 0; b < base_count; ++b) {
                    float d = deglib::distances::fp32_ip::InnerProductFloat::compare(entry_vec, &base_data[b * dim], &dim);
                    dists[b] = {d, static_cast<uint32_t>(b)};
                }
                std::partial_sort(dists.begin(), dists.begin() + search_k, dists.end());
                std::unordered_set<uint32_t> gt_set;
                for (size_t k = 0; k < search_k; ++k) gt_set.insert(dists[k].second);
                total_explore_gt += static_cast<uint32_t>(gt_set.size());

                while (!results.empty()) {
                    uint32_t ext_label = graph.getExternalLabel(results.top().getInternalIndex());
                    if (gt_set.count(ext_label)) {
                        correct_explore++;
                    }
                    results.pop();
                }
            }
        }
    }
    auto t_end_explore = std::chrono::high_resolution_clock::now();
    double total_explore_ms = std::chrono::duration<double, std::milli>(t_end_explore - t_start_explore).count();
    double explore_qps = (static_cast<double>(total_explorations) / total_explore_ms) * 1000.0;
    float explore_recall = static_cast<float>(correct_explore) / static_cast<float>(total_explore_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_InnerProduct explore(): "
              << total_explore_ms << " ms total (" << (total_explore_ms / total_explorations) << " ms/q), "
              << explore_qps << " QPS, recall=" << (explore_recall * 100.0f) << "%\n";
}
