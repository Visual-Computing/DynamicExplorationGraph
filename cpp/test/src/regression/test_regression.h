#pragma once

#include <deglib.h>
#include <gtest/gtest.h>

#include "deterministic_normal_distribution.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

// Compute exact brute-force groundtruth for top-K neighbors using a custom distance evaluator.
// Works for both float and uint8_t element types via the ElemType template parameter.
template <typename ElemType, typename DistFunc>
inline static std::vector<std::vector<uint32_t>> compute_groundtruth_custom(const std::vector<ElemType>& base, size_t base_count,
                                                                       const std::vector<ElemType>& query, size_t query_count,
                                                                       size_t dim, uint32_t k, DistFunc dist_func)
{
    std::vector<std::vector<uint32_t>> gt(query_count);

    for (int q = 0; q < static_cast<int>(query_count); ++q)
    {
        std::vector<std::pair<float, uint32_t>> dists(base_count);
        const ElemType* q_vec = &query[q * dim];

        for (size_t i = 0; i < base_count; ++i)
        {
            const ElemType* b_vec = &base[i * dim];
            float d = dist_func(q_vec, b_vec, &dim);
            dists[i] = {d, static_cast<uint32_t>(i)};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].reserve(k);
        for (uint32_t i = 0; i < k; ++i)
        {
            gt[q].push_back(dists[i].second);
        }
    }

    return gt;
}

// Compute exact brute-force L2 groundtruth for top-K neighbors.
// Uses the scalar L2Float::compare() implementation from deglib to ensure
// the ground-truth distances match the actual distance computation exactly.
inline static std::vector<std::vector<uint32_t>> compute_groundtruth_l2(const std::vector<float>& base, size_t base_count,
                                                                 const std::vector<float>& query, size_t query_count,
                                                                 size_t dim, uint32_t k)
{
    return compute_groundtruth_custom<float>(base, base_count, query, query_count, dim, k,
                                      [](const float* q_vec, const float* b_vec, const void* qty_ptr)
                                      {
                                          return deglib::distances::fp32_l2::L2Float::compare(q_vec, b_vec, qty_ptr);
                                      });
}

// Compute exact brute-force InnerProduct groundtruth for top-K neighbors (distance = 1 - dot_product).
// Uses the scalar InnerProductFloat::compare() implementation from deglib to ensure
// the ground-truth distances match the actual distance computation exactly.
inline static std::vector<std::vector<uint32_t>> compute_groundtruth_innerproduct(const std::vector<float>& base, size_t base_count,
                                                                             const std::vector<float>& query, size_t query_count,
                                                                             size_t dim, uint32_t k)
{
    return compute_groundtruth_custom<float>(base, base_count, query, query_count, dim, k,
                                      [](const float* q_vec, const float* b_vec, const void* qty_ptr)
                                      {
                                          return deglib::distances::fp32_ip::InnerProductFloat::compare(q_vec, b_vec, qty_ptr);
                                      });
}

// Generate cross-platform deterministic clustered dataset (Gaussian Mixture with fixed seed)
// Uses DeterministicNormalDistribution instead of std::normal_distribution for portability.
inline static void generate_synthetic_clustered_dataset(size_t count, size_t dim, std::vector<float>& base,
                                                 std::vector<float>& query, size_t query_count,
                                                 size_t num_clusters = 20)
{
    base.resize(count * dim);
    query.resize(query_count * dim);

    std::mt19937 rng(42);
    std::vector<std::vector<float>> centroids(num_clusters, std::vector<float>(dim));
    DeterministicNormalDistribution cent_dist(-1000.0f, 1000.0f);
    DeterministicNormalDistribution noise_dist(-100.0f, 100.0f);

    for (size_t c = 0; c < num_clusters; ++c)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            centroids[c][d] = cent_dist(rng);
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            base[i * dim + d] = centroids[c][d] + noise_dist(rng);
        }
    }

    for (size_t q = 0; q < query_count; ++q)
    {
        size_t c = q % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            query[q * dim + d] = centroids[c][d] + noise_dist(rng);
        }
    }
}

// Compute exact brute-force L2 groundtruth for uint8 vectors.
// Uses the scalar L2Uint8::compare() from deglib to ensure the ground-truth
// distances match the actual distance computation exactly.
inline static std::vector<std::vector<uint32_t>> compute_groundtruth_l2_uint8(const std::vector<uint8_t>& base, size_t base_count,
                                                                          const std::vector<uint8_t>& query, size_t query_count,
                                                                          size_t dim, uint32_t k)
{
    return compute_groundtruth_custom<uint8_t>(base, base_count, query, query_count, dim, k,
                                      [](const uint8_t* q_vec, const uint8_t* b_vec, const void* qty_ptr)
                                      {
                                          return deglib::distances::uint8_l2::L2Uint8::compare(q_vec, b_vec, qty_ptr);
                                      });
}

// Generate uint8 clustered synthetic dataset (values clamped to 0..255)
// Uses DeterministicNormalDistribution instead of std::normal_distribution for portability.
inline static void generate_synthetic_clustered_dataset_uint8(size_t count, size_t dim, std::vector<uint8_t>& base,
                                                              std::vector<uint8_t>& query, size_t query_count,
                                                              size_t num_clusters = 20)
{
    base.resize(count * dim);
    query.resize(query_count * dim);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> cent_dist(20, 235);
    DeterministicNormalDistribution noise_dist(-15.0f, 15.0f);

    std::vector<std::vector<int>> centroids(num_clusters, std::vector<int>(dim));
    for (size_t c = 0; c < num_clusters; ++c)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            centroids[c][d] = cent_dist(rng);
        }
    }

    auto clamp_u8 = [](float val) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
    };

    for (size_t i = 0; i < count; ++i)
    {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            base[i * dim + d] = clamp_u8(centroids[c][d] + noise_dist(rng));
        }
    }

    for (size_t q = 0; q < query_count; ++q)
    {
        size_t c = q % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            query[q * dim + d] = clamp_u8(centroids[c][d] + noise_dist(rng));
        }
    }
}


// Universal regression benchmark runner function for any metric
// num_runs: number of measured search runs (averaged for QPS/recall).
//           Higher values extend the total search measurement window, reducing QPS noise.
inline static void run_regression_test(const char* name, deglib::Metric metric, double min_qps, double max_build_secs,
                                double min_recall, const void* base_data,
                                const void* query_data, size_t base_count, size_t query_count,
                                size_t dim, const std::vector<std::vector<uint32_t>>& gt_data,
                                std::optional<deglib::DistanceVariant> dist_variant = std::nullopt,
                                size_t num_runs = 5)
{
    std::cout << "--- Testing Instruction Variant: " << name << " ---" << std::endl;

    const uint32_t search_k = 10;
    const float search_eps = 0.05f;

    const uint32_t edges_per_vertex = 32;
    const deglib::builder::OptimizationTarget optimization_target = deglib::builder::OptimizationTarget::LowLID;
    const uint8_t extend_k = static_cast<uint8_t>(edges_per_vertex);
    const float extend_eps = 0.1f;
    const uint8_t improve_k = 0;
    const float improve_eps = 0.0f;
    const uint8_t max_path_length = 5;
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;
    const uint32_t thread_count = 1;

    // Compute byte size per vector based on metric type (0x10 flag indicates 8-bit integer)
    const size_t feature_bytes = (static_cast<int>(metric) & 0x10) ? dim * sizeof(uint8_t) : dim * sizeof(float);

    // Build DEG Graph using the specified metric feature space
    const deglib::FloatSpace feature_space = dist_variant.has_value()
        ? deglib::FloatSpace(dim, metric, dist_variant.value())
        : deglib::FloatSpace(dim, metric);

    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(base_count), edges_per_vertex,
                                          std::move(feature_space));

    std::mt19937 rng(1337);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rng, optimization_target, extend_k, extend_eps, improve_k,
                                                     improve_eps, max_path_length, swap_tries, additional_swap_tries);
    builder.setThreadCount(thread_count);
    auto t_build_start = std::chrono::high_resolution_clock::now();

    const std::byte* base_bytes = reinterpret_cast<const std::byte*>(base_data);
    for (size_t i = 0; i < base_count; ++i)
    {
        const std::byte* ptr = base_bytes + i * feature_bytes;
        std::vector<std::byte> feat_vec(ptr, ptr + feature_bytes);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
    }

    auto build_callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(build_callback);

    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_secs = std::chrono::duration<double>(t_build_end - t_build_start).count();

    auto entry_vertex_indices = graph.getEntryVertexIndices();
    const std::byte* query_bytes = reinterpret_cast<const std::byte*>(query_data);

    auto run_search = [&]() -> std::pair<double, double>
    {
        size_t total_correct = 0;
        auto t_search_start = std::chrono::high_resolution_clock::now();

        for (size_t q = 0; q < query_count; ++q)
        {
            const std::byte* q_ptr = query_bytes + q * feature_bytes;
            auto result = graph.search(entry_vertex_indices, q_ptr, search_eps, search_k, nullptr, 0);

            std::unordered_set<uint32_t> gt_set;
            if (!gt_data.empty() && q < gt_data.size())
            {
                size_t eval_k = std::min(static_cast<size_t>(search_k), gt_data[q].size());
                for (size_t i = 0; i < eval_k; ++i)
                {
                    gt_set.insert(gt_data[q][i]);
                }
            }

            while (!result.empty())
            {
                auto top_item = result.top();
                result.pop();
                uint32_t ext_label = graph.getExternalLabel(top_item.getInternalIndex());
                if (gt_set.count(ext_label))
                {
                    total_correct++;
                }
            }
        }

        auto t_search_end = std::chrono::high_resolution_clock::now();
        double search_secs = std::chrono::duration<double>(t_search_end - t_search_start).count();
        double qps = static_cast<double>(query_count) / search_secs;
        double recall = static_cast<double>(total_correct) / static_cast<double>(query_count * search_k);
        return {qps, recall};
    };

    // Warm-up run
    run_search();

    // Measured average runs — higher num_runs extends the measurement window,
    // reducing QPS noise from OS jitter and CPU power-state transitions.
    double total_qps = 0.0;
    double total_recall = 0.0;
    for (size_t r = 0; r < num_runs; ++r)
    {
        auto [qps, recall] = run_search();
        total_qps += qps;
        total_recall += recall;
    }
    double qps = total_qps / num_runs;
    double recall = total_recall / num_runs;

    std::cout << "[" << name << "] build_secs=" << build_secs << "  qps=" << qps << "  recall=" << recall << std::endl;

    EXPECT_GE(recall + 1e-5, min_recall) << name << ": Recall@10 dropped below threshold";
    if (std::getenv("SKIP_PERFORMANCE_TESTS") == nullptr) {
        EXPECT_GT(qps, min_qps) << name << ": Search QPS dropped below threshold";
        EXPECT_LE(build_secs, max_build_secs) << name << ": Build time exceeded threshold";
    }
}
