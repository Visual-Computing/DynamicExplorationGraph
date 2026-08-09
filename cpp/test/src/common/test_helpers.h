#pragma once

#include <deglib/deglib.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/distance/evp_inner_product.h"

// ============================================================================
// Shared Test Utilities
// ============================================================================
// Centralized helpers for both integration and regression tests.
// All dataset generation uses a bit-exact 32-bit PRNG so results are
// reproducible across MSVC, GCC, and Clang on all CPU architectures.
// ============================================================================

// ---------------------------------------------------------------------------
// PRNG — pure 32-bit integer xorshift, bit-exact across MSVC, GCC, Clang.
// ---------------------------------------------------------------------------

inline static uint32_t deglib_prng_next(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

inline static float deglib_prng_float(uint32_t& state, float min_val, float max_val) {
    uint32_t val = deglib_prng_next(state) >> 8; // 24-bit integer
    float u = static_cast<float>(val) / 16777215.0f;
    float range = max_val - min_val;
    return min_val + u * range;
}

// ---------------------------------------------------------------------------
// Dataset generation (float, for L2 and InnerProduct)
// ---------------------------------------------------------------------------

// Generate cross-platform deterministic clustered dataset (Gaussian Mixture with fixed seed)
// Uses pure 32-bit integer arithmetic and exact integer-to-float conversion to guarantee
// 100% bit-exact float vectors across MSVC, GCC, and Clang on all CPU architectures.
inline static void generate_synthetic_clustered_dataset(size_t count, size_t dim, std::vector<float>& base,
                                                 std::vector<float>& query, size_t query_count,
                                                 size_t num_clusters = 20)
{
    base.resize(count * dim);
    query.resize(query_count * dim);

    uint32_t rng_state = 42;
    std::vector<std::vector<int32_t>> centroids(num_clusters, std::vector<int32_t>(dim));

    for (size_t c = 0; c < num_clusters; ++c)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            centroids[c][d] = static_cast<int32_t>(val % 2001) - 1000; // [-1000, 1000]
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            int32_t noise = static_cast<int32_t>(val % 201) - 100; // [-100, 100]
            base[i * dim + d] = static_cast<float>(centroids[c][d] + noise);
        }
    }

    for (size_t q = 0; q < query_count; ++q)
    {
        size_t c = q % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            int32_t noise = static_cast<int32_t>(val % 201) - 100; // [-100, 100]
            query[q * dim + d] = static_cast<float>(centroids[c][d] + noise);
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset generation (uint8, for L2_Uint8)
// ---------------------------------------------------------------------------

// Generate cross-platform deterministic uint8 clustered dataset
inline static void generate_synthetic_clustered_dataset_uint8(size_t count, size_t dim, std::vector<uint8_t>& base,
                                                       std::vector<uint8_t>& query, size_t query_count,
                                                       size_t num_clusters = 20)
{
    base.resize(count * dim);
    query.resize(query_count * dim);

    uint32_t rng_state = 42;
    std::vector<std::vector<int>> centroids(num_clusters, std::vector<int>(dim));

    for (size_t c = 0; c < num_clusters; ++c)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            centroids[c][d] = 20 + static_cast<int>(val % 216); // [20, 235]
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            int noise = static_cast<int>(val % 31) - 15; // [-15, 15]
            int res = centroids[c][d] + noise;
            base[i * dim + d] = static_cast<uint8_t>(std::clamp(res, 0, 255));
        }
    }

    for (size_t q = 0; q < query_count; ++q)
    {
        size_t c = q % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            int noise = static_cast<int>(val % 31) - 15; // [-15, 15]
            int res = centroids[c][d] + noise;
            query[q * dim + d] = static_cast<uint8_t>(std::clamp(res, 0, 255));
        }
    }
}

// ---------------------------------------------------------------------------
// Dataset generation (FP16, for FP16InnerProduct)
// ---------------------------------------------------------------------------

// Generate cross-platform deterministic clustered FP16 dataset.
// Generates float vectors using the same PRNG as the float generator,
// then converts them to FP16 (uint16_t) using bit-exact conversion.
// The float centroids are computed using the same logic as generate_synthetic_clustered_dataset,
// ensuring the FP16 dataset is derived from the same deterministic float data.
inline static void generate_synthetic_clustered_dataset_fp16(size_t count, size_t dim, std::vector<uint16_t>& base,
                                                       std::vector<uint16_t>& query, size_t query_count,
                                                       size_t num_clusters = 20)
{
    // Generate float data using the same logic as the float generator
    std::vector<float> base_float(count * dim);
    std::vector<float> query_float(query_count * dim);

    uint32_t rng_state = 42;
    std::vector<std::vector<int32_t>> centroids(num_clusters, std::vector<int32_t>(dim));

    for (size_t c = 0; c < num_clusters; ++c)
    {
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            centroids[c][d] = static_cast<int32_t>(val % 2001) - 1000; // [-1000, 1000]
        }
    }

    for (size_t i = 0; i < count; ++i)
    {
        size_t c = i % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            int32_t noise = static_cast<int32_t>(val % 201) - 100; // [-100, 100]
            base_float[i * dim + d] = static_cast<float>(centroids[c][d] + noise);
        }
    }

    for (size_t q = 0; q < query_count; ++q)
    {
        size_t c = q % num_clusters;
        for (size_t d = 0; d < dim; ++d)
        {
            uint32_t val = deglib_prng_next(rng_state);
            int32_t noise = static_cast<int32_t>(val % 201) - 100; // [-100, 100]
            query_float[q * dim + d] = static_cast<float>(centroids[c][d] + noise);
        }
    }

    // Convert float vectors to FP16 (uint16_t) using bit-exact conversion
    base.resize(count * dim);
    query.resize(query_count * dim);
    deglib::distances::fp16::floats_to_fp16(base_float.data(), base.data(), count * dim);
    deglib::distances::fp16::floats_to_fp16(query_float.data(), query.data(), query_count * dim);
}

// ---------------------------------------------------------------------------
// Groundtruth computation
// ---------------------------------------------------------------------------

// Compute exact brute-force groundtruth for top-K neighbors using a custom distance evaluator.
// Works for all vector element types (float, uint8_t, uint16_t, std::byte).
template <typename ElemType, typename DistFunc>
inline static std::vector<std::vector<uint32_t>> compute_groundtruth(const std::vector<ElemType>& base, size_t base_count,
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
    return compute_groundtruth<float>(base, base_count, query, query_count, dim, k,
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
    return compute_groundtruth<float>(base, base_count, query, query_count, dim, k,
                                      [](const float* q_vec, const float* b_vec, const void* qty_ptr)
                                      {
                                          return deglib::distances::fp32_ip::InnerProductFloat::compare(q_vec, b_vec, qty_ptr);
                                      });
}

// Compute exact brute-force FP16 InnerProduct groundtruth for top-K neighbors (distance = 1.f - dot_product).
// Uses the scalar InnerProductFP16::compare() implementation from deglib.
inline static std::vector<std::vector<uint32_t>> compute_groundtruth_fp16_ip(const std::vector<uint16_t>& base, size_t base_count,
                                                                         const std::vector<uint16_t>& query, size_t query_count,
                                                                         size_t dim, uint32_t k)
{
    return compute_groundtruth<uint16_t>(base, base_count, query, query_count, dim, k,
                                      [](const uint16_t* q_vec, const uint16_t* b_vec, const void* qty_ptr)
                                      {
                                          return deglib::distances::fp16_ip::InnerProductFP16::compare(q_vec, b_vec, qty_ptr);
                                      });
}

// Compute exact brute-force L2 groundtruth for uint8 vectors.
// Uses the scalar L2Uint8::compare() from deglib to ensure the ground-truth
// distances match the actual distance computation exactly.
inline static std::vector<std::vector<uint32_t>> compute_groundtruth_l2_uint8(const std::vector<uint8_t>& base, size_t base_count,
                                                                          const std::vector<uint8_t>& query, size_t query_count,
                                                                          size_t dim, uint32_t k)
{
    return compute_groundtruth<uint8_t>(base, base_count, query, query_count, dim, k,
                                      [](const uint8_t* q_vec, const uint8_t* b_vec, const void* qty_ptr)
                                      {
                                          return deglib::distances::uint8_l2::L2Uint8::compare(q_vec, b_vec, qty_ptr);
                                      });
}

// Compute exact brute-force EVP InnerProduct groundtruth for top-K neighbors.
// Uses the scalar EvpInnerProduct::compare() from deglib to ensure
// the ground-truth distances match the actual distance computation exactly.
// For EVP, a single vector is 2 * (dim / 8) bytes (ones mask + negative_ones mask).
inline static std::vector<std::vector<uint32_t>> compute_groundtruth_evp(const std::vector<std::byte>& base, size_t base_count,
                                                                     const std::vector<std::byte>& query, size_t query_count,
                                                                     size_t dim, uint32_t k)
{
    const size_t bytes_per_vec = 2 * (dim / 8);
    return compute_groundtruth<std::byte>(base, base_count, query, query_count, bytes_per_vec, k,
        [dim](const std::byte* q_vec, const std::byte* b_vec, const void*)
        {
            uint32_t d = static_cast<uint32_t>(dim);
            return deglib::distances::evp_ip::EvpInnerProduct::compare(q_vec, b_vec, &d);
        });
}

// ---------------------------------------------------------------------------
// Dataset generation (EVP, for EVPInnerProduct)
// ---------------------------------------------------------------------------

// Generate cross-platform deterministic clustered EVP dataset.
// Generates float vectors using generate_synthetic_clustered_dataset(),
// then quantizes them to EVP bytes via deglib::quantization::evp::quantize_batch().
inline static void generate_synthetic_clustered_dataset_evp(size_t count, size_t dim, std::vector<std::byte>& base_evp,
                                                        std::vector<std::byte>& query_evp, size_t query_count,
                                                        size_t num_clusters = 20, uint32_t non_zeros = 0)
{
    if (non_zeros == 0) non_zeros = static_cast<uint32_t>(dim / 4);
    std::vector<float> base_float, query_float;
    generate_synthetic_clustered_dataset(count, dim, base_float, query_float, query_count, num_clusters);

    base_evp  = deglib::quantization::evp::quantize_batch(base_float.data(), count, dim, non_zeros);
    query_evp = deglib::quantization::evp::quantize_batch(query_float.data(), query_count, dim, non_zeros);
}

// ---------------------------------------------------------------------------
// Distance recall verification (SIMD variant vs scalar ground truth)
// ---------------------------------------------------------------------------

template <typename ElemType, typename DistFunc>
inline static void check_distance_recall(const char* name, const std::vector<ElemType>& base_data, size_t base_count,
                                         const std::vector<ElemType>& query_data, size_t query_count,
                                         size_t dim, uint32_t k,
                                         const std::vector<std::vector<uint32_t>>& gt_scalar,
                                         DistFunc dist_func)
{
    std::vector<std::vector<uint32_t>> gt(query_count);
    for (int q = 0; q < static_cast<int>(query_count); ++q)
    {
        std::vector<std::pair<float, uint32_t>> dists(base_count);
        const void* q_vec = static_cast<const void*>(&query_data[q * dim]);
        for (size_t i = 0; i < base_count; ++i)
        {
            const void* b_vec = static_cast<const void*>(&base_data[i * dim]);
            size_t qty = dim;
            float d = dist_func(q_vec, b_vec, &qty);
            dists[i] = {d, static_cast<uint32_t>(i)};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].reserve(k);
        for (uint32_t i = 0; i < k; ++i) gt[q].push_back(dists[i].second);
    }

    size_t correct = 0;
    for (size_t q = 0; q < query_count; ++q)
    {
        std::unordered_set<uint32_t> gt_set(gt_scalar[q].begin(), gt_scalar[q].end());
        for (uint32_t idx : gt[q])
        {
            if (gt_set.count(idx)) ++correct;
        }
    }
    double recall = static_cast<double>(correct) / static_cast<double>(query_count * k);
    std::cout << "[DistanceRecall " << name << "] recall=" << recall << "  correct=" << correct << "/"
              << (query_count * k) << std::endl;
    EXPECT_EQ(recall, 1.0) << "Distance recall between scalar and " << name << " must be exactly 1.0";
}

// ---------------------------------------------------------------------------
// Distance recall verification for EVP (SIMD variant vs scalar ground truth)
// ---------------------------------------------------------------------------
// EVP vectors are bit-packed: 2 * (dim / 8) bytes per vector. The dim parameter
// passed to check_distance_recall controls vector indexing, but EVP distance
// functions need the original float dimension as qty_ptr. This wrapper handles
// the type mismatch by using bytes_per_vec for indexing and the float dim for qty.
template <typename DistFunc>
inline static void check_distance_recall_evp(const char* name, const std::vector<std::byte>& base_data, size_t base_count,
                                             const std::vector<std::byte>& query_data, size_t query_count,
                                             size_t dim, uint32_t k,
                                             const std::vector<std::vector<uint32_t>>& gt_scalar,
                                             DistFunc dist_func)
{
    const size_t bytes_per_vec = 2 * (dim / 8);
    std::vector<std::vector<uint32_t>> gt(query_count);
    for (int q = 0; q < static_cast<int>(query_count); ++q)
    {
        std::vector<std::pair<float, uint32_t>> dists(base_count);
        const std::byte* q_vec = &query_data[q * bytes_per_vec];
        for (size_t i = 0; i < base_count; ++i)
        {
            const std::byte* b_vec = &base_data[i * bytes_per_vec];
            uint32_t qty = static_cast<uint32_t>(dim);
            float d = dist_func(q_vec, b_vec, &qty);
            dists[i] = {d, static_cast<uint32_t>(i)};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        gt[q].reserve(k);
        for (uint32_t i = 0; i < k; ++i) gt[q].push_back(dists[i].second);
    }

    size_t correct = 0;
    for (size_t q = 0; q < query_count; ++q)
    {
        std::unordered_set<uint32_t> gt_set(gt_scalar[q].begin(), gt_scalar[q].end());
        for (uint32_t idx : gt[q])
        {
            if (gt_set.count(idx)) ++correct;
        }
    }
    double recall = static_cast<double>(correct) / static_cast<double>(query_count * k);
    std::cout << "[DistanceRecall " << name << "] recall=" << recall << "  correct=" << correct << "/"
              << (query_count * k) << std::endl;
    EXPECT_EQ(recall, 1.0) << "Distance recall between scalar and " << name << " must be exactly 1.0";
}

// ---------------------------------------------------------------------------
// Graph builder + search recall runner (integration: no QPS / build-time checks)
// ---------------------------------------------------------------------------

inline static void run_integration_test(const char* name, deglib::Metric metric, double min_recall,
                                const void* base_data,
                                const void* query_data, size_t base_count, size_t query_count,
                                size_t dim, const std::vector<std::vector<uint32_t>>& gt_data,
                                std::optional<deglib::DistanceVariant> dist_variant = std::nullopt,
                                deglib::builder::OptimizationTarget optimization_target = deglib::builder::OptimizationTarget::LowLID)
{
    const uint32_t search_k = 10;
    const float search_eps = 0.05f;

    const uint32_t edges_per_vertex = 32;
    const uint8_t extend_k = static_cast<uint8_t>(edges_per_vertex);
    const float extend_eps = 0.1f;
    const uint8_t improve_k = 0;
    const float improve_eps = 0.0f;
    const uint8_t max_path_length = 5;
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;
    const uint32_t thread_count = 1;

    // Build DEG Graph using the specified metric feature space
    const deglib::FloatSpace feature_space = dist_variant.has_value()
        ? deglib::FloatSpace(dim, metric, dist_variant.value())
        : deglib::FloatSpace(dim, metric);

    const size_t feature_bytes = feature_space.get_data_size();

    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(base_count), edges_per_vertex,
                                          std::move(feature_space));

    std::mt19937 rng(1337);
    deglib::builder::EvenRegularGraphBuilder builder(graph, rng, optimization_target, extend_k, extend_eps, improve_k,
                                                     improve_eps, max_path_length, swap_tries, additional_swap_tries);
    builder.setThreadCount(thread_count);

    const std::byte* base_bytes = reinterpret_cast<const std::byte*>(base_data);
    for (size_t i = 0; i < base_count; ++i)
    {
        const std::byte* ptr = base_bytes + i * feature_bytes;
        std::vector<std::byte> feat_vec(ptr, ptr + feature_bytes);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
    }

    auto build_callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(build_callback);

    auto entry_vertex_indices = graph.getEntryVertexIndices();
    const std::byte* query_bytes = reinterpret_cast<const std::byte*>(query_data);

    size_t total_correct = 0;
    for (size_t q = 0; q < query_count; ++q)
    {
        const std::byte* q_ptr = query_bytes + q * feature_bytes;
        std::span<const float> q_span(reinterpret_cast<const float*>(q_ptr), dim);
        auto result = graph.search(q_span, search_k, search_eps, nullptr, 0);

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
            uint32_t ext_label = graph.getExternalLabel(top_item.getIdentifier());
            if (gt_set.count(ext_label))
            {
                total_correct++;
            }
        }
    }

    double recall = static_cast<double>(total_correct) / static_cast<double>(query_count * search_k);
    std::cout << "[" << name << "] recall=" << recall << std::endl;

    EXPECT_GE(recall + 1e-5, min_recall);
}

// ---------------------------------------------------------------------------
// Builder helper for determinism tests — builds a graph and returns all
// neighbor indices as a flat vector for comparison.
// ---------------------------------------------------------------------------

inline static std::vector<uint32_t> build_graph_for_determinism(
    deglib::builder::OptimizationTarget optimization_target,
    size_t dim, size_t base_count, uint32_t edges_per_vertex,
    const std::vector<float>& base_data)
{
    const deglib::FloatSpace feature_space(dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(base_count), edges_per_vertex,
                                          std::move(feature_space));

    std::mt19937 rng(1337);
    const uint8_t extend_k = static_cast<uint8_t>(edges_per_vertex);
    const float extend_eps = 0.1f;
    const uint8_t improve_k = 0;
    const float improve_eps = 0.0f;
    const uint8_t max_path_length = 5;
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;

    deglib::builder::EvenRegularGraphBuilder builder(graph, rng, optimization_target,
                                                     extend_k, extend_eps, improve_k, improve_eps,
                                                     max_path_length, swap_tries, additional_swap_tries);
    builder.setThreadCount(1);

    const size_t feature_bytes = dim * sizeof(float);
    const std::byte* base_bytes = reinterpret_cast<const std::byte*>(base_data.data());
    for (size_t i = 0; i < base_count; ++i)
    {
        const std::byte* ptr = base_bytes + i * feature_bytes;
        std::vector<std::byte> feat_vec(ptr, ptr + feature_bytes);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
    }

    auto build_callback = [](deglib::builder::BuilderStatus& status) {};
    builder.build(build_callback);

    std::vector<uint32_t> neighbors;
    neighbors.reserve(base_count * edges_per_vertex);
    for (uint32_t v = 0; v < base_count; ++v)
    {
        const uint32_t* nb = graph.getNeighborIndices(v);
        for (uint32_t e = 0; e < edges_per_vertex; ++e)
        {
            neighbors.push_back(nb[e]);
        }
    }
    return neighbors;
}

// ---------------------------------------------------------------------------
// Builder integration test runner — wraps run_integration_test with dataset
// generation and groundtruth for a given metric, variant, and optimization target.
// ---------------------------------------------------------------------------

inline static void run_builder_integration_test(const char* name, deglib::Metric metric, double min_recall,
                                                size_t dim, size_t base_count, size_t query_count, size_t num_clusters,
                                                std::optional<deglib::DistanceVariant> dist_variant,
                                                deglib::builder::OptimizationTarget optimization_target)
{
    if (metric.get_data_type() == deglib::MetricDataType::Uint8)
    {
        // uint8 metric
        std::vector<uint8_t> base_data;
        std::vector<uint8_t> query_data;
        generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

        auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

        run_integration_test(name, metric, min_recall,
                             base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                             dist_variant, optimization_target);
    }
    else if (metric == deglib::Metric::FP16InnerProduct)
    {
        // FP16 metric
        std::vector<uint16_t> base_data;
        std::vector<uint16_t> query_data;
        generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

        auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

        run_integration_test(name, metric, min_recall,
                             base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                             dist_variant, optimization_target);
    }
    else if (metric == deglib::Metric::EVPInnerProduct)
    {
        // EVP metric
        std::vector<std::byte> base_data;
        std::vector<std::byte> query_data;
        generate_synthetic_clustered_dataset_evp(base_count, dim, base_data, query_data, query_count, num_clusters);

        auto gt_data = compute_groundtruth_evp(base_data, base_count, query_data, query_count, dim, 10);

        run_integration_test(name, metric, min_recall,
                             base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                             dist_variant, optimization_target);
    }
    else
    {
        // float metric (L2 or InnerProduct)
        std::vector<float> base_data;
        std::vector<float> query_data;
        generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

        std::vector<std::vector<uint32_t>> gt_data;
        if (metric == deglib::Metric::InnerProduct)
        {
            gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);
        }
        else
        {
            gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);
        }

        run_integration_test(name, metric, min_recall,
                             base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                             dist_variant, optimization_target);
    }
}

// ---------------------------------------------------------------------------
// Universal regression benchmark runner function for any metric
// num_runs: number of measured search runs (averaged for QPS/recall).
//           Higher values extend the total search measurement window, reducing QPS noise.
// optimization_target: controls the graph build strategy (LowLID, HighLID, StreamingData).
// ---------------------------------------------------------------------------

inline static void run_regression_test(const char* name, deglib::Metric metric, double min_qps, double max_build_secs,
                                double min_recall, const void* base_data,
                                const void* query_data, size_t base_count, size_t query_count,
                                size_t dim, const std::vector<std::vector<uint32_t>>& gt_data,
                                std::optional<deglib::DistanceVariant> dist_variant = std::nullopt,
                                size_t num_runs = 5,
                                deglib::builder::OptimizationTarget optimization_target = deglib::builder::OptimizationTarget::LowLID,
                                uint32_t edges_per_vertex = 32,
                                uint8_t extend_k = 0,
                                float extend_eps = 0.1f)
{
    const uint32_t search_k = 10;
    const float search_eps = 0.05f;

    if (extend_k == 0) extend_k = static_cast<uint8_t>(edges_per_vertex);
    const uint8_t improve_k = 0;
    const float improve_eps = 0.0f;
    const uint8_t max_path_length = 5;
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;
    const uint32_t thread_count = 1;

    // Build DEG Graph using the specified metric feature space
    const deglib::FloatSpace feature_space = dist_variant.has_value()
        ? deglib::FloatSpace(dim, metric, dist_variant.value())
        : deglib::FloatSpace(dim, metric);

    const size_t feature_bytes = feature_space.get_data_size();

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
            std::span<const float> q_span(reinterpret_cast<const float*>(q_ptr), dim);
            auto result = graph.search(q_span, search_k, search_eps, nullptr, 0);

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
                uint32_t ext_label = graph.getExternalLabel(top_item.getIdentifier());
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

    EXPECT_GE(recall + 1e-5, min_recall);
    if (std::getenv("SKIP_PERFORMANCE_TESTS") == nullptr) {
        EXPECT_GT(qps, min_qps);
        EXPECT_LE(build_secs, max_build_secs);
    }
}

// ---------------------------------------------------------------------------
// Checksums for dataset determinism verification
// ---------------------------------------------------------------------------

// FNV-1a 64-bit hash for byte buffers
inline static uint64_t fnv1a_64(const void* data, size_t bytes) {
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= ptr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Compute checksum of a float vector
inline static uint64_t float_vector_checksum(const std::vector<float>& vec) {
    return fnv1a_64(vec.data(), vec.size() * sizeof(float));
}

// Compute checksum of a byte vector
inline static uint64_t byte_vector_checksum(const std::vector<std::byte>& vec) {
    return fnv1a_64(vec.data(), vec.size());
}

// Compute checksum of groundtruth 2D vector
inline static uint64_t groundtruth_checksum(const std::vector<std::vector<uint32_t>>& gt) {
    uint64_t hash = 14695981039346656037ULL;
    for (const auto& row : gt) {
        uint64_t row_hash = fnv1a_64(row.data(), row.size() * sizeof(uint32_t));
        hash ^= row_hash;
        hash *= 1099511628211ULL;
    }
    return hash;
}
