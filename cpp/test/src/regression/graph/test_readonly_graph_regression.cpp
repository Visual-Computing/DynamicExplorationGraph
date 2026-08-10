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
    const size_t dim = 1024;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t num_clusters = 100;
    const uint32_t edges_per_vertex = 16;
    const float extend_eps = 0.1f;
    const float search_eps = 0.001f;
    const uint32_t search_k = 100;
    const int benchmark_runs = 100;
    const uint32_t explore_max_calcs = 100;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    std::vector<float> explore_data(query_count * dim);
    for (size_t i = 0; i < query_count; ++i) {
        uint32_t entry_node = static_cast<uint32_t>(i % base_count);
        std::memcpy(&explore_data[i * dim], &base_data[entry_node * dim], dim * sizeof(float));
    }

    deglib::distances::FloatSpace feature_space(dim, deglib::distances::Metric::FP32_L2);
    auto dist_func = feature_space.get_dist_func();

    auto search_gt_data = compute_groundtruth(base_data, base_count, query_data, query_count, dim, search_k, dist_func);
    auto explore_gt_data = compute_groundtruth(base_data, base_count, explore_data, query_count, dim, search_k, dist_func);

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

    // 1. Search Benchmark 
    std::vector<decltype(graph.search(std::span<const float>(), 0, 0.0f))> last_search_results(query_count);
    auto t_start_search = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t q = 0; q < query_count; ++q) {
            auto results = graph.search(std::span<const float>(&query_data[q * dim], dim), search_k, search_eps);
            if (run == 0) {
                last_search_results[q] = std::move(results);
            }
        }
    }
    auto t_end_search = std::chrono::high_resolution_clock::now();
    double total_search_ms = std::chrono::duration<double, std::milli>(t_end_search - t_start_search).count();

    // Ground truth & recall calculation for search (outside timer)
    uint32_t correct_search = 0;
    uint32_t total_gt = 0;
    for (size_t q = 0; q < query_count; ++q) {
        std::unordered_set<uint32_t> gt_set(search_gt_data[q].begin(), search_gt_data[q].end());
        total_gt += static_cast<uint32_t>(gt_set.size());
        auto results = std::move(last_search_results[q]);
        while (!results.empty()) {
            uint32_t ext_label = graph.getExternalLabel(results.top().getIdentifier());
            if (gt_set.count(ext_label)) {
                correct_search++;
            }
            results.pop();
        }
    }

    size_t total_queries = query_count * benchmark_runs;
    double search_qps = (static_cast<double>(total_queries) / total_search_ms) * 1000.0;
    float search_recall = static_cast<float>(correct_search) / static_cast<float>(total_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_L2 search(): "
              << total_search_ms << " ms total for " << total_queries << " queries (" 
              << (total_search_ms / total_queries) << " ms/q), "
              << search_qps << " QPS, recall=" << (search_recall * 100.0f) << "%\n";

    // 2. Explore Benchmark 
    std::vector<decltype(graph.explore(0, 0, 0))> last_explore_results(query_count);
    auto t_start_explore = std::chrono::high_resolution_clock::now();

    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t i = 0; i < query_count; ++i) {
            uint32_t entry_node = static_cast<uint32_t>((run * 13 + i) % base_count);
            auto results = graph.explore(entry_node, search_k, explore_max_calcs);
            if (run == 0) {
                last_explore_results[i] = std::move(results);
            }
        }
    }
    auto t_end_explore = std::chrono::high_resolution_clock::now();
    double total_explore_ms = std::chrono::duration<double, std::milli>(t_end_explore - t_start_explore).count();

    // Ground truth & recall calculation for explore (outside timer)
    uint32_t correct_explore = 0;
    uint32_t total_explore_gt = 0;
    for (size_t i = 0; i < query_count; ++i) {
        std::unordered_set<uint32_t> gt_set(explore_gt_data[i].begin(), explore_gt_data[i].end());
        total_explore_gt += static_cast<uint32_t>(gt_set.size());

        auto results = std::move(last_explore_results[i]);
        while (!results.empty()) {
            uint32_t ext_label = graph.getExternalLabel(results.top().getIdentifier());
            if (gt_set.count(ext_label)) {
                correct_explore++;
            }
            results.pop();
        }
    }

    size_t total_explorations = query_count * benchmark_runs;
    double explore_qps = (static_cast<double>(total_explorations) / total_explore_ms) * 1000.0;
    float explore_recall = static_cast<float>(correct_explore) / static_cast<float>(total_explore_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_L2 explore(): "
              << total_explore_ms << " ms total for " << total_explorations << " explorations (" 
              << (total_explore_ms / total_explorations) << " ms/q), "
              << explore_qps << " QPS, recall=" << (explore_recall * 100.0f) << "%\n";
}

TEST(ReadOnlyGraphRegression, SearchAndExplore_FP32_InnerProduct)
{
    const size_t dim = 1024;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t num_clusters = 100;
    const uint32_t edges_per_vertex = 16;
    const float extend_eps = 0.1f;
    const float search_eps = 0.001f;
    const uint32_t search_k = 100;
    const int benchmark_runs = 100;
    const uint32_t explore_max_calcs = 100;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    std::vector<float> explore_data(query_count * dim);
    for (size_t i = 0; i < query_count; ++i) {
        uint32_t entry_node = static_cast<uint32_t>(i % base_count);
        std::memcpy(&explore_data[i * dim], &base_data[entry_node * dim], dim * sizeof(float));
    }

    deglib::distances::FloatSpace feature_space(dim, deglib::distances::Metric::FP32_InnerProduct);
    auto dist_func = feature_space.get_dist_func();

    auto search_gt_data = compute_groundtruth(base_data, base_count, query_data, query_count, dim, search_k, dist_func);
    auto explore_gt_data = compute_groundtruth(base_data, base_count, explore_data, query_count, dim, search_k, dist_func);

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

    // Search Benchmark 
    std::vector<decltype(graph.search(std::span<const float>(), 0, 0.0f))> last_search_results(query_count);
    auto t_start_search = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t q = 0; q < query_count; ++q) {
            auto results = graph.search(std::span<const float>(&query_data[q * dim], dim), search_k, search_eps);
            if (run == 0) {
                last_search_results[q] = std::move(results);
            }
        }
    }
    auto t_end_search = std::chrono::high_resolution_clock::now();
    double total_search_ms = std::chrono::duration<double, std::milli>(t_end_search - t_start_search).count();

    // Ground truth & recall calculation for search (outside timer)
    uint32_t correct_search = 0;
    uint32_t total_gt = 0;
    for (size_t q = 0; q < query_count; ++q) {
        std::unordered_set<uint32_t> gt_set(search_gt_data[q].begin(), search_gt_data[q].end());
        total_gt += static_cast<uint32_t>(gt_set.size());
        auto results = std::move(last_search_results[q]);
        while (!results.empty()) {
            uint32_t ext_label = graph.getExternalLabel(results.top().getIdentifier());
            if (gt_set.count(ext_label)) {
                correct_search++;
            }
            results.pop();
        }
    }

    size_t total_queries = query_count * benchmark_runs;
    double search_qps = (static_cast<double>(total_queries) / total_search_ms) * 1000.0;
    float search_recall = static_cast<float>(correct_search) / static_cast<float>(total_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_InnerProduct search(): "
              << total_search_ms << " ms total for " << total_queries << " queries (" 
              << (total_search_ms / total_queries) << " ms/q), "
              << search_qps << " QPS, recall=" << (search_recall * 100.0f) << "%\n";

    // Explore Benchmark
    std::vector<decltype(graph.explore(0, 0, 0))> last_explore_results(query_count);
    auto t_start_explore = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t i = 0; i < query_count; ++i) {
            uint32_t entry_node = static_cast<uint32_t>((run * 13 + i) % base_count);
            auto results = graph.explore(entry_node, search_k, explore_max_calcs);
            if (run == 0) {
                last_explore_results[i] = std::move(results);
            }
        }
    }
    auto t_end_explore = std::chrono::high_resolution_clock::now();
    double total_explore_ms = std::chrono::duration<double, std::milli>(t_end_explore - t_start_explore).count();

    // Ground truth & recall calculation for explore (outside timer)
    uint32_t correct_explore = 0;
    uint32_t total_explore_gt = 0;
    for (size_t i = 0; i < query_count; ++i) {
        std::unordered_set<uint32_t> gt_set(explore_gt_data[i].begin(), explore_gt_data[i].end());
        total_explore_gt += static_cast<uint32_t>(gt_set.size());

        auto results = std::move(last_explore_results[i]);
        while (!results.empty()) {
            uint32_t ext_label = graph.getExternalLabel(results.top().getIdentifier());
            if (gt_set.count(ext_label)) {
                correct_explore++;
            }
            results.pop();
        }
    }

    size_t total_explorations = query_count * benchmark_runs;
    double explore_qps = (static_cast<double>(total_explorations) / total_explore_ms) * 1000.0;
    float explore_recall = static_cast<float>(correct_explore) / static_cast<float>(total_explore_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph FP32_InnerProduct explore(): "
              << total_explore_ms << " ms total for " << total_explorations << " explorations (" 
              << (total_explore_ms / total_explorations) << " ms/q), "
              << explore_qps << " QPS, recall=" << (explore_recall * 100.0f) << "%\n";
}

TEST(ReadOnlyGraphRegression, SearchAndExplore_EVP_InnerProduct)
{
    const size_t dim = 1024;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t num_clusters = 100;
    const uint32_t edges_per_vertex = 16;
    const float extend_eps = 0.1f;
    const float search_eps = 0.001f;
    const uint32_t search_k = 100;
    const int benchmark_runs = 100;
    const uint32_t explore_max_calcs = 100;
    const size_t non_zeros = 512;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    std::vector<float> explore_data(query_count * dim);
    for (size_t i = 0; i < query_count; ++i) {
        uint32_t entry_node = static_cast<uint32_t>(i % base_count);
        std::memcpy(&explore_data[i * dim], &base_data[entry_node * dim], dim * sizeof(float));
    }

    deglib::distances::FloatSpace float_feature_space(dim, deglib::distances::Metric::FP32_InnerProduct);
    auto float_dist_func = float_feature_space.get_dist_func();

    // Compute groundtruth using original float data (InnerProduct)
    auto search_gt_data = compute_groundtruth(base_data, base_count, query_data, query_count, dim, search_k, float_dist_func);
    auto explore_gt_data = compute_groundtruth(base_data, base_count, explore_data, query_count, dim, search_k, float_dist_func);

    // Quantize data to EVP
    auto base_quant = deglib::quantization::evp::quantize_batch(base_data.data(), base_count, dim, non_zeros, 8);
    auto query_quant = deglib::quantization::evp::quantize_batch(query_data.data(), query_count, dim, non_zeros, 8);
    size_t vec_bytes = 2 * (dim / 8);

    deglib::distances::FloatSpace feature_space(dim, deglib::distances::Metric::EVP_InnerProduct);
    deglib::graph::SizeBoundedGraph mutable_graph(static_cast<uint32_t>(base_count), edges_per_vertex, feature_space);

    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(
        mutable_graph, rnd, deglib::builder::OptimizationTarget::LowLID, edges_per_vertex, extend_eps, 0, 0.0f
    );
    builder.setThreadCount(1);

    for (size_t i = 0; i < base_count; ++i) {
        std::vector<std::byte> feat(vec_bytes);
        std::memcpy(feat.data(), base_quant.data() + i * vec_bytes, vec_bytes);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat));
    }
    auto dummy_callback = [](deglib::builder::BuilderStatus&) {};
    builder.build(dummy_callback, false);

    // Convert to ReadOnlyGraph
    deglib::graph::ReadOnlyGraph graph(mutable_graph.size(), edges_per_vertex, feature_space, mutable_graph);

    // Search Benchmark 
    std::vector<decltype(graph.search(std::span<const std::byte>(), 0, 0.0f))> last_search_results(query_count);
    auto t_start_search = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t q = 0; q < query_count; ++q) {
            std::span<const std::byte> q_span(reinterpret_cast<const std::byte*>(query_quant.data() + q * vec_bytes), vec_bytes);
            auto results = graph.search(q_span, search_k, search_eps);
            if (run == 0) {
                last_search_results[q] = std::move(results);
            }
        }
    }
    auto t_end_search = std::chrono::high_resolution_clock::now();
    double total_search_ms = std::chrono::duration<double, std::milli>(t_end_search - t_start_search).count();

    // Ground truth & recall calculation for search using original float GT (outside timer)
    uint32_t correct_search = 0;
    uint32_t total_gt = 0;
    for (size_t q = 0; q < query_count; ++q) {
        std::unordered_set<uint32_t> gt_set(search_gt_data[q].begin(), search_gt_data[q].end());
        total_gt += static_cast<uint32_t>(gt_set.size());
        auto results = std::move(last_search_results[q]);
        while (!results.empty()) {
            uint32_t ext_label = graph.getExternalLabel(results.top().getIdentifier());
            if (gt_set.count(ext_label)) {
                correct_search++;
            }
            results.pop();
        }
    }

    size_t total_queries = query_count * benchmark_runs;
    double search_qps = (static_cast<double>(total_queries) / total_search_ms) * 1000.0;
    float search_recall = static_cast<float>(correct_search) / static_cast<float>(total_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph EVP_InnerProduct search(): "
              << total_search_ms << " ms total for " << total_queries << " queries (" 
              << (total_search_ms / total_queries) << " ms/q), "
              << search_qps << " QPS, recall=" << (search_recall * 100.0f) << "%\n";

    // Explore Benchmark
    std::vector<decltype(graph.explore(0, 0, 0))> last_explore_results(query_count);
    auto t_start_explore = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < benchmark_runs; ++run) {
        for (size_t i = 0; i < query_count; ++i) {
            uint32_t entry_node = static_cast<uint32_t>((run * 13 + i) % base_count);
            auto results = graph.explore(entry_node, search_k, explore_max_calcs);
            if (run == 0) {
                last_explore_results[i] = std::move(results);
            }
        }
    }
    auto t_end_explore = std::chrono::high_resolution_clock::now();
    double total_explore_ms = std::chrono::duration<double, std::milli>(t_end_explore - t_start_explore).count();

    // Ground truth & recall calculation for explore using original float data (outside timer)
    uint32_t correct_explore = 0;
    uint32_t total_explore_gt = 0;
    for (size_t i = 0; i < query_count; ++i) {
        std::unordered_set<uint32_t> gt_set(explore_gt_data[i].begin(), explore_gt_data[i].end());
        total_explore_gt += static_cast<uint32_t>(gt_set.size());

        auto results = std::move(last_explore_results[i]);
        while (!results.empty()) {
            uint32_t ext_label = graph.getExternalLabel(results.top().getIdentifier());
            if (gt_set.count(ext_label)) {
                correct_explore++;
            }
            results.pop();
        }
    }

    size_t total_explorations = query_count * benchmark_runs;
    double explore_qps = (static_cast<double>(total_explorations) / total_explore_ms) * 1000.0;
    float explore_recall = static_cast<float>(correct_explore) / static_cast<float>(total_explore_gt);

    std::cout << "[BENCHMARK 100x] ReadOnlyGraph EVP_InnerProduct explore(): "
              << total_explore_ms << " ms total for " << total_explorations << " explorations (" 
              << (total_explore_ms / total_explorations) << " ms/q), "
              << explore_qps << " QPS, recall=" << (explore_recall * 100.0f) << "%\n";
}
