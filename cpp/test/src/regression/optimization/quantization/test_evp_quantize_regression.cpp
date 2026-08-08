#include <cstdint>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>

#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/distance/evp_inner_product.h"
#include "deglib/graph/sizebounded_graph.h"
#include "deglib/builder.h"
#include "common/test_helpers.h"
#include "gtest/gtest.h"

// ============================================================================
// EVP Quantization & EVP_InnerProduct Search Regression Benchmarks
// ============================================================================

TEST(EvpQuantizeRegression, QuantizeBatchThroughput) {
    const size_t count = 10000;
    const uint32_t dim = 1024;
    const uint32_t non_zeros = 512;

    std::vector<float> data(count * dim);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = dist(rng);
    }

    // Single-thread quantization benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    auto single_res = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Multi-thread quantization benchmark (8 threads)
    t0 = std::chrono::high_resolution_clock::now();
    auto multi_res = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 8);
    t1 = std::chrono::high_resolution_clock::now();
    double multi_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_EQ(single_res, multi_res);

    std::cout << "[REGRESSION BENCHMARK] EVP QuantizeBatch (" << count << " vecs, dim=" << dim << ", non_zeros=" << non_zeros << "):\n"
              << "  Single-thread: " << single_ms << " ms (" << (static_cast<double>(count) / single_ms * 1000.0) << " vecs/sec)\n"
              << "  8-thread:      " << multi_ms << " ms (" << (static_cast<double>(count) / multi_ms * 1000.0) << " vecs/sec)\n";
}

TEST(EvpQuantizeRegression, EvpDistanceCalculationSpeed) {
    const size_t count = 10000;
    const uint32_t dim = 1024;
    const uint32_t non_zeros = 512;

    std::vector<float> data(count * dim);
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = dist(rng);
    }

    auto quantized = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 8);
    size_t vec_bytes = 2 * dim / 8; // byte size per quantized vector

    const size_t num_dists = 5000000; // 5 million distance comparisons
    auto t0 = std::chrono::high_resolution_clock::now();

    double sum_dists = 0.0;
    for (size_t i = 0; i < num_dists; ++i) {
        size_t idx1 = i % count;
        size_t idx2 = (i * 7 + 13) % count;
        const auto* vec1 = reinterpret_cast<const float*>(quantized.data() + idx1 * vec_bytes);
        const auto* vec2 = reinterpret_cast<const float*>(quantized.data() + idx2 * vec_bytes);
        float d = deglib::distances::evp_ip::EvpInnerProduct::compare(vec1, vec2, &dim);
        sum_dists += d;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mops = (static_cast<double>(num_dists) / total_ms) / 1000.0;

    std::cout << "[REGRESSION BENCHMARK] EVP Distance Calculations (" << num_dists << " dists, dim=" << dim << "): "
              << total_ms << " ms total (" << mops << " Million ops/sec, sum=" << sum_dists << ")\n";
}