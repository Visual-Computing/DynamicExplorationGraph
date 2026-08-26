#include "common/test_helpers.h"
#include "deglib/distances.h"
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif defined(__linux__)
    #include <pthread.h>
    #include <sched.h>
#endif
// ============================================================================
// Distance Calculation Performance Regression Benchmarks
// ============================================================================
// Measures throughput (Distance Comparisons/sec, Element Throughput, QPS, Latency)
// for all distance metrics and instruction sets:
//   - Int8_InnerProduct (Scalar, AVX2, AVX2_VNNI, AVX512, AVX512_VNNI)
//   - FP32_L2 (Scalar, AVX2, AVX512)
//   - FP32_InnerProduct (Scalar, AVX2, AVX512)
//   - FP16_L2 (Scalar, AVX2, AVX512)
//   - FP16_InnerProduct (Scalar, AVX2, AVX512)
//   - Uint8_L2 (Scalar, AVX2, AVX512)
//   - Uint8_InnerProduct (Scalar, AVX2, AVX512)
//   - EVP_InnerProduct (Scalar, AVX2, AVX512)
// These benchmarks detect performance regressions in raw distance kernels.
// Correctness is verified in unit tests.
// ============================================================================

namespace {

inline void pin_current_thread_to_core(int core_id = 0) {
#if defined(_WIN32)
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = static_cast<DWORD_PTR>(1) << (core_id % (sizeof(DWORD_PTR) * 8));
    SetThreadAffinityMask(thread, mask);
    SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id % CPU_SETSIZE, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

struct DistBenchmarkResult {
    double qps;        // Queries / sec (single compare)
    double batch_qps;  // Queries / sec (compare_batch)
};

template <typename VectorType>
inline DistBenchmarkResult benchmark_distance_metric(
    deglib::distances::Metric metric,
    deglib::cpu::InstructionSet instruction,
    const std::vector<VectorType>& base_data,
    size_t base_count,
    const std::vector<VectorType>& query_data,
    size_t query_count,
    size_t dim,
    size_t iterations
) {
    pin_current_thread_to_core(0);
    const deglib::distances::FloatSpace space(dim, metric, instruction);
    const auto dist_func = space.get_dist_func();
    const auto batch_dist_func = space.get_batch_dist_func();
    const auto* param = space.get_dist_func_param();
    const size_t raw_element_count = space.get_data_size() / sizeof(VectorType);
    const size_t raw_byte_size = space.get_data_size();
    // Align stride of each vector to a 64-byte cache line boundary
    const size_t aligned_byte_stride = (raw_byte_size + 63) & ~size_t(63);
    const size_t stride_elements = aligned_byte_stride / sizeof(VectorType);

    // Allocate 64-byte aligned contiguous memory for base and query data
    std::vector<VectorType> aligned_base(base_count * stride_elements, VectorType(0));
    for (size_t i = 0; i < base_count; ++i) {
        std::memcpy(&aligned_base[i * stride_elements], &base_data[i * raw_element_count], raw_byte_size);
    }

    std::vector<VectorType> aligned_query(query_count * stride_elements, VectorType(0));
    for (size_t q = 0; q < query_count; ++q) {
        std::memcpy(&aligned_query[q * stride_elements], &query_data[q * raw_element_count], raw_byte_size);
    }

    volatile float sink = 0.0f;

    // Prepare pointer array for base data for batch testing (every pointer is 64-byte aligned)
    std::vector<const void*> base_ptrs(base_count);
    for (size_t i = 0; i < base_count; ++i) {
        base_ptrs[i] = &aligned_base[i * stride_elements];
    }
    std::vector<float> dists_buffer(base_count);

    // Warmup single compare
    for (size_t q = 0; q < std::min<size_t>(query_count, 10); ++q) {
        const void* q_vec = &aligned_query[q * stride_elements];
        for (size_t i = 0; i < std::min<size_t>(base_count, 100); ++i) {
            const void* b_vec = &aligned_base[i * stride_elements];
            sink += dist_func(q_vec, b_vec, param);
        }
    }

    // Benchmark single compare
    auto start_single = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        for (size_t q = 0; q < query_count; ++q) {
            const void* q_vec = &aligned_query[q * stride_elements];
            for (size_t i = 0; i < base_count; ++i) {
                const void* b_vec = &aligned_base[i * stride_elements];
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
        const void* q_vec = &aligned_query[q * stride_elements];
        batch_dist_func(q_vec, base_ptrs.data(), base_count, param, dists_buffer.data());
        sink += dists_buffer[0];
    }

    // Benchmark compare_batch
    auto start_batch = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        for (size_t q = 0; q < query_count; ++q) {
            const void* q_vec = &aligned_query[q * stride_elements];
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

    std::string metric_str;
    switch (metric) {
        case deglib::distances::Metric::FP32_L2: metric_str = "FP32_L2"; break;
        case deglib::distances::Metric::FP32_InnerProduct: metric_str = "FP32_IP"; break;
        case deglib::distances::Metric::FP16_L2: metric_str = "FP16_L2"; break;
        case deglib::distances::Metric::FP16_InnerProduct: metric_str = "FP16_IP"; break;
        case deglib::distances::Metric::Uint8_L2: metric_str = "Uint8_L2"; break;
        case deglib::distances::Metric::Uint8_InnerProduct: metric_str = "Uint8_IP"; break;
        case deglib::distances::Metric::Int8_InnerProduct: metric_str = "Int8_IP"; break;
        case deglib::distances::Metric::EVP_InnerProduct: metric_str = "EVP_IP"; break;
        default: metric_str = "Unknown"; break;
    }
    std::string display_name = metric_str + " (" + space.get_instruction() + ")";
    std::cout << std::left << std::setw(30) << display_name << " | " << std::right << std::setw(6) << dim << " dim"
              << " | " << std::setw(12) << std::fixed << std::setprecision(2) << result.qps << " QPS"
              << " | " << std::setw(12) << std::fixed << std::setprecision(2) << result.batch_qps << " Batch QPS" << std::endl;
    return result;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Main Regression Test Suites for Distance Kernels
// ---------------------------------------------------------------------------

TEST(DistancesRegression, Uint8_InnerProduct_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 50;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n============= Uint8 InnerProduct Distance Throughput (10k Base, 100 Queries) =============\n";
    for (size_t dim : dims) {
        std::vector<uint8_t> base_data, query_data;
        generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data,
            query_count, dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx_vnni()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::AVX2_VNNI, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512_vnni()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Uint8_InnerProduct, deglib::cpu::InstructionSet::AVX512_VNNI, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

TEST(DistancesRegression, Int8_InnerProduct_Throughput) {
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const size_t iterations_scalar = 5;
    const size_t iterations_simd = 50;
    std::vector<size_t> dims = {64, 128, 200, 256, 384, 512, 640, 768, 1024};

    std::cout << "\n============= Int8 InnerProduct Distance Throughput (10k Base, 100 Queries) =============\n";
    for (size_t dim : dims) {
        std::vector<int8_t> base_data, query_data;
        generate_synthetic_clustered_dataset_int8(base_count, dim, base_data, query_data, query_count, 1000);

        benchmark_distance_metric(
            deglib::distances::Metric::Int8_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data,
            query_count, dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Int8_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx_vnni()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Int8_InnerProduct, deglib::cpu::InstructionSet::AVX2_VNNI, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Int8_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512_vnni()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Int8_InnerProduct, deglib::cpu::InstructionSet::AVX512_VNNI, base_data, base_count,
                query_data, query_count, dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}

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
            deglib::distances::Metric::FP32_L2, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count, dim,
            iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP32_L2, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count, dim,
                iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP32_L2, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data, query_count,
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
            deglib::distances::Metric::FP32_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data,
            query_count, dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP32_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP32_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
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
            deglib::distances::Metric::FP16_L2, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count, dim,
            iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP16_L2, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count, dim,
                iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP16_L2, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data, query_count,
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
            deglib::distances::Metric::FP16_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data,
            query_count, dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP16_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::FP16_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
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
            deglib::distances::Metric::Uint8_L2, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count, dim,
            iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Uint8_L2, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count, dim,
                iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::Uint8_L2, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data, query_count,
                dim, iterations_simd
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
            deglib::distances::Metric::EVP_InnerProduct, deglib::cpu::InstructionSet::Scalar, base_data, base_count, query_data, query_count,
            dim, iterations_scalar
        );

#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            benchmark_distance_metric(
                deglib::distances::Metric::EVP_InnerProduct, deglib::cpu::InstructionSet::AVX2, base_data, base_count, query_data, query_count,
                dim, iterations_simd
            );
        }
        if (deglib::cpu::has_avx512()) {
            benchmark_distance_metric(
                deglib::distances::Metric::EVP_InnerProduct, deglib::cpu::InstructionSet::AVX512, base_data, base_count, query_data,
                query_count, dim, iterations_simd
            );
        }
#endif
        std::cout << "--------------------------------------------------------------------------------------\n";
    }
}
