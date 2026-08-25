#include "common/test_helpers.h"
#include "deglib/distances.h"
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

// ============================================================================
// Distance Calculation Performance Regression Benchmarks
// ============================================================================
// Measures throughput (Distance Comparisons/sec, Element Throughput, QPS, Latency)
// for all distance metrics and instruction sets:
//   - FP32_L2 (Scalar, AVX2, AVX512)
//   - FP32_InnerProduct (Scalar, AVX2, AVX512)
//   - FP16_L2 (Scalar, AVX2, AVX512)
//   - FP16_InnerProduct (Scalar, AVX2, AVX512)
//   - Uint8_L2 (Scalar, AVX2, AVX512)
//   - Uint8_InnerProduct (Scalar, AVX2, AVX512)
//   - EVP_InnerProduct (Scalar, AVX2, AVX512)
//
// These benchmarks detect performance regressions in raw distance kernels.
// Correctness is verified in unit tests.
// ============================================================================

namespace {

struct DistBenchmarkResult {
    double qps;        // Queries / sec (single compare)
    double batch_qps;  // Queries / sec (compare_batch)
};

template <typename VectorType>
inline DistBenchmarkResult benchmark_distance_metric(
    const char* name,
    deglib::distances::Metric metric,
    deglib::cpu::InstructionSet instruction,
    const std::vector<VectorType>& base_data,
    size_t base_count,
    const std::vector<VectorType>& query_data,
    size_t query_count,
    size_t dim,
    size_t iterations
) {
    const deglib::distances::FloatSpace space(dim, metric, instruction);
    const auto dist_func = space.get_dist_func();
    const auto batch_dist_func = space.get_batch_dist_func();
    const auto* param = space.get_dist_func_param();
    const size_t elements_per_vec = space.get_data_size() / sizeof(VectorType);
    volatile float sink = 0.0f;

    // Prepare pointer array for base data for batch testing
    std::vector<const void*> base_ptrs(base_count);
    for (size_t i = 0; i < base_count; ++i) {
        base_ptrs[i] = &base_data[i * elements_per_vec];
    }
    std::vector<float> dists_buffer(base_count);

    // Warmup single compare
    for (size_t q = 0; q < std::min<size_t>(query_count, 10); ++q) {
        const void* q_vec = &query_data[q * elements_per_vec];
        for (size_t i = 0; i < std::min<size_t>(base_count, 100); ++i) {
            const void* b_vec = &base_data[i * elements_per_vec];
            sink += dist_func(q_vec, b_vec, param);
        }
    }

    // Benchmark single compare
    auto start_single = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        for (size_t q = 0; q < query_count; ++q) {
            const void* q_vec = &query_data[q * elements_per_vec];
            for (size_t i = 0; i < base_count; ++i) {
                const void* b_vec = &base_data[i * elements_per_vec];
                sink += dist_func(q_vec, b_vec, param);
            }
        }
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    double total_ms_single = std::chrono::duration<double, std::milli>(end_single - start_single).count();
    double total_queries = static_cast<double>(query_count * iterations);
    double single_qps = total_queries / (total_ms_single / 1000.0);

    // Warmup compare_batch
    for (size_t q = 0; q < std::min<size_t>(query_count, 10); ++q) {
        const void* q_vec = &query_data[q * elements_per_vec];
        batch_dist_func(q_vec, base_ptrs.data(), base_count, param, dists_buffer.data());
        sink += dists_buffer[0];
    }

    // Benchmark compare_batch
    auto start_batch = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        for (size_t q = 0; q < query_count; ++q) {
            const void* q_vec = &query_data[q * elements_per_vec];
            batch_dist_func(q_vec, base_ptrs.data(), base_count, param, dists_buffer.data());
            sink += dists_buffer[0];
        }
    }
    auto end_batch = std::chrono::high_resolution_clock::now();
    double total_ms_batch = std::chrono::duration<double, std::milli>(end_batch - start_batch).count();
    double batch_qps = total_queries / (total_ms_batch / 1000.0);

    DistBenchmarkResult result;
    result.qps = single_qps;
    result.batch_qps = batch_qps;

    std::cout << std::left << std::setw(32) << name << " | " << std::right << std::setw(6) << dim << " dim"
              << " | " << std::setw(12) << std::fixed << std::setprecision(2) << result.qps << " QPS"
              << " | " << std::setw(12) << std::fixed << std::setprecision(2) << result.batch_qps << " Batch QPS" << std::endl;

    return result;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Main Regression Test Suites for Distance Kernels
// ---------------------------------------------------------------------------

TEST(DistancesRegression, FP32_L2_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 20;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n================ FP32 L2 Distance Throughput (10k Base, 100 Queries) ================\n";
    for (size_t dim : dims) {
        std::vector<float> base_data, query_data;
        generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            "FP32_L2 (Scalar)", deglib::distances::Metric::FP32_L2, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count, dim,
            iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                "FP32_L2 (AVX2)", deglib::distances::Metric::FP32_L2, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count, dim,
                iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                "FP32_L2 (AVX512)", deglib::distances::Metric::FP32_L2, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data, query_count,
                dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

TEST(DistancesRegression, FP32_InnerProduct_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 20;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n============= FP32 InnerProduct Distance Throughput (10k Base, 100 Queries) =============\n";
    for (size_t dim : dims) {
        std::vector<float> base_data, query_data;
        generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            "FP32_IP (Scalar)", deglib::distances::Metric::FP32_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data,
            query_count, dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                "FP32_IP (AVX2)", deglib::distances::Metric::FP32_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                "FP32_IP (AVX512)", deglib::distances::Metric::FP32_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

TEST(DistancesRegression, FP16_L2_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 20;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n================ FP16 L2 Distance Throughput (10k Base, 100 Queries) ================\n";
    for (size_t dim : dims) {
        std::vector<uint16_t> base_data, query_data;
        generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            "FP16_L2 (Scalar)", deglib::distances::Metric::FP16_L2, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count, dim,
            iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                "FP16_L2 (AVX2)", deglib::distances::Metric::FP16_L2, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count, dim,
                iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                "FP16_L2 (AVX512)", deglib::distances::Metric::FP16_L2, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data, query_count,
                dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

TEST(DistancesRegression, FP16_InnerProduct_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 20;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n============= FP16 InnerProduct Distance Throughput (10k Base, 100 Queries) =============\n";
    for (size_t dim : dims) {
        std::vector<uint16_t> base_data, query_data;
        generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            "FP16_IP (Scalar)", deglib::distances::Metric::FP16_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data,
            query_count, dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                "FP16_IP (AVX2)", deglib::distances::Metric::FP16_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                "FP16_IP (AVX512)", deglib::distances::Metric::FP16_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

TEST(DistancesRegression, Uint8_L2_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 20;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n================ Uint8 L2 Distance Throughput (10k Base, 100 Queries) ================\n";
    for (size_t dim : dims) {
        std::vector<uint8_t> base_data, query_data;
        generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            "Uint8_L2 (Scalar)", deglib::distances::Metric::Uint8_L2, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count, dim,
            iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                "Uint8_L2 (AVX2)", deglib::distances::Metric::Uint8_L2, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count, dim,
                iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                "Uint8_L2 (AVX512)", deglib::distances::Metric::Uint8_L2, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data, query_count,
                dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

TEST(DistancesRegression, Uint8_InnerProduct_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 20;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n============= Uint8 InnerProduct Distance Throughput (10k Base, 100 Queries) =============\n";
    for (size_t dim : dims) {
        std::vector<uint8_t> base_data, query_data;
        generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            "Uint8_IP (Scalar)", deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data,
            query_count, dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                "Uint8_IP (AVX2)", deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                "Uint8_IP (AVX512)", deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

TEST(DistancesRegression, EVP_InnerProduct_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 20;
    const size_t iterations_simd = 20;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n============== EVP InnerProduct Distance Throughput (10k Base, 100 Queries) ==============\n";
    for (size_t dim : dims) {
        std::vector<std::byte> base_data, query_data;
        generate_synthetic_clustered_dataset_evp(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            "EVP_IP (Scalar)", deglib::distances::Metric::EVP_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count,
            dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                "EVP_IP (AVX2)", deglib::distances::Metric::EVP_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count,
                dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                "EVP_IP (AVX512)", deglib::distances::Metric::EVP_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}
