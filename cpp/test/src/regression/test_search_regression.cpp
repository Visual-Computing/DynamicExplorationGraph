#include "deglib/distances.h"
#include "deglib/search.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

// Deterministic PRNG for generating benchmark data
inline uint32_t benchmark_prng(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

inline float benchmark_float(uint32_t& state, float min_val = -1.0f, float max_val = 1.0f) {
    uint32_t val = benchmark_prng(state) >> 8;
    float u = static_cast<float>(val) / 16777215.0f;
    return min_val + u * (max_val - min_val);
}

}  // namespace

TEST(SearchRegression, Rerank_FP32_L2_Benchmark) {
    const size_t dim = 512;
    const size_t num_queries = 5000;
    const size_t num_base = 100000;
    const size_t candidates_per_query = 200;
    const size_t k_top = 10;
    const size_t num_threads = 0;  // auto-detect

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    // Generate random base vectors and query vectors
    std::vector<float> base_vectors(num_base * dim);
    std::vector<float> query_vectors(num_queries * dim);
    uint32_t rng = 1337;

    for (auto& v : base_vectors) v = benchmark_float(rng);
    for (auto& v : query_vectors) v = benchmark_float(rng);

    // Generate candidate indices per query (random indices in [0, num_base))
    std::vector<uint32_t> candidates(num_queries * candidates_per_query);
    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t c = 0; c < candidates_per_query; ++c) {
            candidates[q * candidates_per_query + c] = benchmark_prng(rng) % num_base;
        }
    }

    // Warm-up run
    auto warm_results = deglib::search::rerank(
        space, query_vectors.data(), num_queries, base_vectors.data(), num_base, candidates.data(), candidates_per_query, k_top, num_threads
    );
    EXPECT_EQ(warm_results.size(), num_queries);

    // Benchmark run
    const int iterations = 30;
    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it) {
        auto results = deglib::search::rerank(
            space, query_vectors.data(), num_queries, base_vectors.data(), num_base, candidates.data(), candidates_per_query, k_top, num_threads
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    double total_comps = static_cast<double>(num_queries * candidates_per_query * iterations);
    double comps_per_sec = (total_comps / (total_ms / 1000.0)) / 1e6;  // MComps/sec
    double gelements_per_sec = (comps_per_sec * dim) / 1000.0;         // GElements/sec
    double qps = (num_queries * iterations) / (total_ms / 1000.0);

    std::cout << "\n=== Rerank Benchmark (FP32_L2, Instruction: " << space.get_instruction() << ") ===\n"
              << "Queries: " << num_queries << ", Base Vectors: " << num_base << ", Dim: " << dim << ", Cands/Query: " << candidates_per_query
              << ", K-Top: " << k_top << ", Iterations: " << iterations << "\n"
              << std::fixed << std::setprecision(3) << "Average latency: " << avg_ms << " ms / batch\n"
              << "Total time: " << total_ms << " ms\n"
              << "Throughput: " << std::setprecision(2) << qps << " QPS\n"
              << "Distance Comparisons: " << comps_per_sec << " MComps/sec\n"
              << "Element Throughput: " << gelements_per_sec << " GElements/sec\n\n";
}

TEST(SearchRegression, Rerank_FP16_InnerProduct_Benchmark) {
    const size_t dim = 512;
    const size_t num_queries = 5000;
    const size_t num_base = 100000;
    const size_t candidates_per_query = 200;
    const size_t k_top = 10;
    const size_t num_threads = 0;  // auto-detect

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP16_InnerProduct);

    // Generate random base vectors and query vectors in FP16 (stored as uint16_t)
    std::vector<uint16_t> base_vectors(num_base * dim);
    std::vector<uint16_t> query_vectors(num_queries * dim);
    uint32_t rng = 4242;

    for (auto& v : base_vectors) v = static_cast<uint16_t>(benchmark_prng(rng) & 0xFFFF);
    for (auto& v : query_vectors) v = static_cast<uint16_t>(benchmark_prng(rng) & 0xFFFF);

    std::vector<uint32_t> candidates(num_queries * candidates_per_query);
    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t c = 0; c < candidates_per_query; ++c) {
            candidates[q * candidates_per_query + c] = benchmark_prng(rng) % num_base;
        }
    }

    // Warm-up run
    auto warm_results = deglib::search::rerank(
        space, query_vectors.data(), num_queries, base_vectors.data(), num_base, candidates.data(), candidates_per_query, k_top, num_threads
    );
    EXPECT_EQ(warm_results.size(), num_queries);

    // Benchmark run
    const int iterations = 30;
    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it) {
        auto results = deglib::search::rerank(
            space, query_vectors.data(), num_queries, base_vectors.data(), num_base, candidates.data(), candidates_per_query, k_top, num_threads
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    double total_comps = static_cast<double>(num_queries * candidates_per_query * iterations);
    double comps_per_sec = (total_comps / (total_ms / 1000.0)) / 1e6;  // MComps/sec
    double gelements_per_sec = (comps_per_sec * dim) / 1000.0;         // GElements/sec
    double qps = (num_queries * iterations) / (total_ms / 1000.0);

    std::cout << "\n=== Rerank Benchmark (FP16_InnerProduct, Instruction: " << space.get_instruction() << ") ===\n"
              << "Queries: " << num_queries << ", Base Vectors: " << num_base << ", Dim: " << dim << ", Cands/Query: " << candidates_per_query
              << ", K-Top: " << k_top << ", Iterations: " << iterations << "\n"
              << std::fixed << std::setprecision(3) << "Average latency: " << avg_ms << " ms / batch\n"
              << "Total time: " << total_ms << " ms\n"
              << "Throughput: " << std::setprecision(2) << qps << " QPS\n"
              << "Distance Comparisons: " << comps_per_sec << " MComps/sec\n"
              << "Element Throughput: " << gelements_per_sec << " GElements/sec\n\n";
}
