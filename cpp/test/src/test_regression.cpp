#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <algorithm>
#include <unordered_set>

#include <builder.h>
#include <deglib.h>

// Compute exact brute-force L2 groundtruth for top-K neighbors
static std::vector<std::vector<uint32_t>> compute_groundtruth_l2(
    const std::vector<float>& base, size_t base_count,
    const std::vector<float>& query, size_t query_count,
    size_t dim, uint32_t k) 
{
    std::vector<std::vector<uint32_t>> gt(query_count);

    for (int q = 0; q < static_cast<int>(query_count); ++q) {
        std::vector<std::pair<float, uint32_t>> dists(base_count);
        const float* q_vec = &query[q * dim];

        for (size_t i = 0; i < base_count; ++i) {
            const float* b_vec = &base[i * dim];
            float sum = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float diff = q_vec[d] - b_vec[d];
                sum += diff * diff;
            }
            dists[i] = {sum, static_cast<uint32_t>(i)};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].reserve(k);
        for (uint32_t i = 0; i < k; ++i) {
            gt[q].push_back(dists[i].second);
        }
    }

    return gt;
}

// Generate cross-platform deterministic clustered dataset (Gaussian Mixture with fixed seed)
static void generate_synthetic_clustered_dataset(size_t count, size_t dim, std::vector<float>& base, std::vector<float>& query, size_t query_count) {
    base.resize(count * dim);
    query.resize(query_count * dim);

    std::mt19937 rng(42);
    size_t num_clusters = 20;
    std::vector<std::vector<float>> centroids(num_clusters, std::vector<float>(dim));
    std::normal_distribution<float> cent_dist(0.0f, 10.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.5f);

    for (size_t c = 0; c < num_clusters; ++c) {
        for (size_t d = 0; d < dim; ++d) {
            centroids[c][d] = cent_dist(rng);
        }
    }

    for (size_t i = 0; i < count; ++i) {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d) {
            base[i * dim + d] = centroids[c][d] + noise_dist(rng);
        }
    }

    for (size_t q = 0; q < query_count; ++q) {
        size_t c = q % num_clusters;
        for (size_t d = 0; d < dim; ++d) {
            query[q * dim + d] = centroids[c][d] + noise_dist(rng);
        }
    }
}

TEST(DeglibRegression, SyntheticClusteredDataset) {
    size_t dim = 128;
    size_t base_count = 1000;
    size_t query_count = 100;
    const uint32_t k = 10;
    const float eps = 0.001f;

    std::vector<float> base_data;
    std::vector<float> query_data;

    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count);
    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    // Build DEG Graph
    deglib::FloatSpace feature_space(dim, deglib::Metric::L2);
    const uint32_t edges_per_vertex = 16;
    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(base_count), edges_per_vertex, std::move(feature_space));

    std::mt19937 rng(1337);
    deglib::builder::EvenRegularGraphBuilder builder(
        graph, rng, deglib::builder::OptimizationTarget::LowLID,
        edges_per_vertex, 0.1f, 4, 0.05f);

    auto t_build_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < base_count; ++i) {
        const std::byte* ptr = reinterpret_cast<const std::byte*>(&base_data[i * dim]);
        std::vector<std::byte> feat_vec(ptr, ptr + dim * sizeof(float));
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
    }

    auto build_callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(build_callback);

    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_secs = std::chrono::duration<double>(t_build_end - t_build_start).count();

    // Search evaluation for eps = 0.001f
    auto entry_vertex_indices = graph.getEntryVertexIndices();
    size_t total_correct = 0;

    auto t_search_start = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < query_count; ++q) {
        const std::byte* q_ptr = reinterpret_cast<const std::byte*>(&query_data[q * dim]);
        auto result = graph.search(entry_vertex_indices, q_ptr, eps, k);

        std::unordered_set<uint32_t> gt_set;
        if (!gt_data.empty() && q < gt_data.size()) {
            size_t eval_k = std::min(static_cast<size_t>(k), gt_data[q].size());
            for (size_t i = 0; i < eval_k; ++i) {
                gt_set.insert(gt_data[q][i]);
            }
        }

        while (!result.empty()) {
            auto top_item = result.top();
            result.pop();
            uint32_t ext_label = graph.getExternalLabel(top_item.getInternalIndex());
            if (gt_set.count(ext_label)) {
                total_correct++;
            }
        }
    }

    auto t_search_end = std::chrono::high_resolution_clock::now();
    double search_secs = std::chrono::duration<double>(t_search_end - t_search_start).count();
    double qps = static_cast<double>(query_count) / search_secs;
    double recall = static_cast<double>(total_correct) / static_cast<double>(query_count * k);

    // Assertions for regression testing
    EXPECT_GE(recall, 0.85) << "Recall@10 dropped below 85% threshold";
    EXPECT_GT(qps, 5000.0) << "Search QPS dropped below 5000 QPS threshold";
    EXPECT_GT(build_secs, 0.0);
}
