#include "common/test_helpers.h"
#include "deglib/builder.h"
#include "deglib/distance/evp_ip.h"
#include "deglib/graph/sizebounded_graph.h"
#include "deglib/transformation/evp.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

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

    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(data);
    std::vector<std::byte> single_res(count * q.output_byte_size());
    std::vector<std::byte> multi_res(count * q.output_byte_size());

    // Single-thread quantization benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    q.transform_database(std::span<const float>(data), std::span<std::byte>(single_res), 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Multi-thread quantization benchmark (8 threads)
    t0 = std::chrono::high_resolution_clock::now();
    q.transform_database(std::span<const float>(data), std::span<std::byte>(multi_res), 8);
    auto t2 = std::chrono::high_resolution_clock::now();
    double multi_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

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

    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(data);
    std::vector<std::byte> quantized(count * q.output_byte_size());
    q.transform_database(std::span<const float>(data), std::span<std::byte>(quantized), 8);
    size_t vec_bytes = q.output_byte_size();  // byte size per quantized vector

    const size_t num_dists = 5000000;  // 5 million distance comparisons
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

    std::cout << "[REGRESSION BENCHMARK] EVP Distance Calculations (" << num_dists << " dists, dim=" << dim << "): " << total_ms << " ms total (" << mops
              << " Million ops/sec, sum=" << sum_dists << ")\n";
}

TEST(EvpQuantizeRegression, CompareNativeFp16VsConvertFp32FitAndTransform) {
    const size_t count = 10000;
    const uint32_t dim = 1024;
    const size_t total_elements = count * dim;

    // Generate test FP16 vectors
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> original_f32(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
        original_f32[i] = dist(rng);
    }
    std::vector<uint16_t> data_fp16(total_elements);
    deglib::distances::fp16::floats_to_fp16(original_f32.data(), data_fp16.data(), total_elements);

    std::cout << "\n======================================================================\n";
    std::cout << "[BENCHMARK] EVP Native FP16 vs (C++ SIMD FP16->FP32 + FP32 EVP)\n";
    std::cout << "Data: " << count << " vectors, dim = " << dim << " (" << (total_elements * sizeof(uint16_t)) / (1024 * 1024) << " MB FP16)\n";
    std::cout << "======================================================================\n";

    // -------------------------------------------------------------------------
    // 1. Separate Breakdown: compute_column_mean & calibrate_best_non_zeros
    // -------------------------------------------------------------------------
    std::cout << "\n--- 1. SEPARATE BREAKDOWN: compute_column_mean ---\n";
    std::vector<float> mean_f32(dim);
    std::vector<float> mean_fp16_out(dim);
    std::vector<float> mean_conv_out(dim);

    // 1.1 FP32 native
    double t_mean_f32 = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        deglib::transformation::detail::compute_column_mean(std::span<const float>(original_f32), dim, std::span<float>(mean_f32), 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        t_mean_f32 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // 1.2 FP16 native
    double t_mean_fp16 = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        deglib::transformation::detail::compute_column_mean(std::span<const uint16_t>(data_fp16), dim, std::span<float>(mean_fp16_out), 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        t_mean_fp16 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // 1.3 FP16 -> Convert to FP32 (SIMD) -> FP32 compute_column_mean
    double t_conv_mean = 0.0;
    double t_mean_after_conv = 0.0;
    {
        std::vector<float> temp_f32(total_elements);
        auto t0 = std::chrono::high_resolution_clock::now();
        deglib::distances::fp16::fp16_to_floats(data_fp16.data(), temp_f32.data(), total_elements);
        auto t1 = std::chrono::high_resolution_clock::now();
        t_conv_mean = std::chrono::duration<double, std::milli>(t1 - t0).count();

        deglib::transformation::detail::compute_column_mean(std::span<const float>(temp_f32), dim, std::span<float>(mean_conv_out), 0);
        auto t2 = std::chrono::high_resolution_clock::now();
        t_mean_after_conv = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    double t_total_conv_mean = t_conv_mean + t_mean_after_conv;

    std::cout << "  (A) FP32 native compute_column_mean:               " << t_mean_f32 << " ms\n";
    std::cout << "  (B) FP16 native compute_column_mean:               " << t_mean_fp16 << " ms\n";
    std::cout << "  (C) FP16 -> FP32 (SIMD Convert " << t_conv_mean << " ms + FP32 mean " << t_mean_after_conv << " ms): "
              << t_total_conv_mean << " ms\n";
    std::cout << "  >>> Speedup: FP16 native is " << (t_total_conv_mean / t_mean_fp16) << "x faster than Convert+FP32, and "
              << (t_mean_f32 / t_mean_fp16) << "x vs pure FP32\n";

    std::cout << "\n--- 2. SEPARATE BREAKDOWN: calibrate_best_non_zeros ---\n";
    std::vector<uint16_t> mean_fp16_vec(dim);
    deglib::distances::fp16::floats_to_fp16(mean_f32.data(), mean_fp16_vec.data(), dim);

    // 2.1 FP32 native calibration
    double t_calib_f32 = 0.0;
    uint32_t nz_f32 = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        nz_f32 = deglib::transformation::detail::calibrate_best_non_zeros(std::span<const float>(original_f32), dim, std::span<const float>(mean_f32), 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        t_calib_f32 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // 2.2 FP16 native calibration
    double t_calib_fp16 = 0.0;
    uint32_t nz_fp16 = 0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        nz_fp16 = deglib::transformation::detail::calibrate_best_non_zeros(std::span<const uint16_t>(data_fp16), dim, std::span<const uint16_t>(mean_fp16_vec), 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        t_calib_fp16 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // 2.3 FP16 -> Convert to FP32 (SIMD) -> FP32 calibrate_best_non_zeros
    double t_conv_calib = 0.0;
    double t_calib_after_conv = 0.0;
    uint32_t nz_conv = 0;
    {
        std::vector<float> temp_f32(total_elements);
        auto t0 = std::chrono::high_resolution_clock::now();
        deglib::distances::fp16::fp16_to_floats(data_fp16.data(), temp_f32.data(), total_elements);
        auto t1 = std::chrono::high_resolution_clock::now();
        t_conv_calib = std::chrono::duration<double, std::milli>(t1 - t0).count();

        nz_conv = deglib::transformation::detail::calibrate_best_non_zeros(std::span<const float>(temp_f32), dim, std::span<const float>(mean_f32), 0);
        auto t2 = std::chrono::high_resolution_clock::now();
        t_calib_after_conv = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    double t_total_conv_calib = t_conv_calib + t_calib_after_conv;

    std::cout << "  (A) FP32 native calibrate_best_non_zeros:          " << t_calib_f32 << " ms (non_zeros=" << nz_f32 << ")\n";
    std::cout << "  (B) FP16 native calibrate_best_non_zeros:          " << t_calib_fp16 << " ms (non_zeros=" << nz_fp16 << ")\n";
    std::cout << "  (C) FP16 -> FP32 (SIMD Convert " << t_conv_calib << " ms + FP32 calib " << t_calib_after_conv << " ms): "
              << t_total_conv_calib << " ms (non_zeros=" << nz_conv << ")\n";
    std::cout << "  >>> Speedup: FP16 native is " << (t_total_conv_calib / t_calib_fp16) << "x faster than Convert+FP32, and "
              << (t_calib_f32 / t_calib_fp16) << "x vs pure FP32\n";

    // -------------------------------------------------------------------------
    // 3. Overall Benchmark: Full FIT (Mean + Calibration combined)
    // -------------------------------------------------------------------------
    std::cout << "\n--- 3. FULL FIT (Mean + Calibration combined) ---\n";

    // 3A. Native FP16 Fit
    double native_fit_ms = 0.0;
    uint32_t native_nz = 0;
    {
        deglib::transformation::EvpQuantizer q_native(dim, 0);
        auto t0 = std::chrono::high_resolution_clock::now();
        q_native.fit(std::span<const uint16_t>(data_fp16));
        auto t1 = std::chrono::high_resolution_clock::now();
        native_fit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        native_nz = q_native.non_zeros();
    }

    // 3B. Convert FP16 -> FP32 (C++ SIMD) + FP32 Fit
    double conv_time_fit_ms = 0.0;
    double f32_fit_ms = 0.0;
    double total_converted_fit_ms = 0.0;
    uint32_t converted_nz = 0;
    {
        std::vector<float> temp_f32(total_elements);
        auto t0 = std::chrono::high_resolution_clock::now();
        deglib::distances::fp16::fp16_to_floats(data_fp16.data(), temp_f32.data(), total_elements);
        auto t1 = std::chrono::high_resolution_clock::now();
        conv_time_fit_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        deglib::transformation::EvpQuantizer q_f32(dim, 0);
        auto t2 = std::chrono::high_resolution_clock::now();
        q_f32.fit(std::span<const float>(temp_f32));
        auto t3 = std::chrono::high_resolution_clock::now();
        f32_fit_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        total_converted_fit_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
        converted_nz = q_f32.non_zeros();
    }

    std::cout << "Native FP16 Fit:                 " << native_fit_ms << " ms (non_zeros=" << native_nz << ")\n";
    std::cout << "Convert to FP32 (SIMD):          " << conv_time_fit_ms << " ms\n";
    std::cout << "FP32 Fit:                        " << f32_fit_ms << " ms (non_zeros=" << converted_nz << ")\n";
    std::cout << "Total (SIMD Convert + FP32 Fit):   " << total_converted_fit_ms << " ms\n";
    if (native_fit_ms < total_converted_fit_ms) {
        std::cout << ">>> Native FP16 Fit is " << (total_converted_fit_ms / native_fit_ms) << "x FASTER than Convert + FP32 Fit\n";
    } else {
        std::cout << ">>> Convert + FP32 Fit is " << (native_fit_ms / total_converted_fit_ms) << "x faster than Native FP16 Fit\n";
    }

    // -------------------------------------------------------------------------
    // 2. Benchmark: TRANSFORM / QUANTIZE (single-thread and multi-thread)
    // -------------------------------------------------------------------------
    const uint32_t chosen_nz = 512;
    deglib::transformation::EvpQuantizer q_eval(dim, chosen_nz);
    q_eval.fit(std::span<const uint16_t>(data_fp16));
    const size_t out_bytes_size = count * q_eval.output_byte_size();

    std::vector<std::byte> out_native(out_bytes_size);
    std::vector<std::byte> out_converted(out_bytes_size);

    for (size_t threads : {size_t{1}, size_t{8}}) {
        std::cout << "\n--- 2. TRANSFORM / QUANTIZE (" << threads << " thread" << (threads > 1 ? "s" : "") << ") ---\n";

        // 2A. Native FP16 Transform
        const int iters = 5;
        double native_transform_ms = 0.0;
        for (int it = 0; it < iters; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            q_eval.transform_database(std::span<const uint16_t>(data_fp16), std::span<std::byte>(out_native), threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            native_transform_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        native_transform_ms /= iters;

        // 2B. C++ SIMD Convert FP16 -> FP32 + FP32 Transform
        std::vector<float> temp_f32(total_elements);
        double conv_ms = 0.0;
        double f32_transform_ms = 0.0;
        for (int it = 0; it < iters; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            deglib::distances::fp16::fp16_to_floats(data_fp16.data(), temp_f32.data(), total_elements);
            auto t1 = std::chrono::high_resolution_clock::now();
            conv_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

            q_eval.transform_database(std::span<const float>(temp_f32), std::span<std::byte>(out_converted), threads);
            auto t2 = std::chrono::high_resolution_clock::now();
            f32_transform_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        conv_ms /= iters;
        f32_transform_ms /= iters;
        double total_converted_transform_ms = conv_ms + f32_transform_ms;

        std::cout << "Native FP16 Transform:               " << native_transform_ms << " ms ("
                  << (static_cast<double>(count) / native_transform_ms * 1000.0) << " vecs/sec)\n";
        std::cout << "Convert to FP32 (SIMD):              " << conv_ms << " ms\n";
        std::cout << "FP32 Transform:                      " << f32_transform_ms << " ms ("
                  << (static_cast<double>(count) / f32_transform_ms * 1000.0) << " vecs/sec)\n";
        std::cout << "Total (SIMD Convert + FP32 Transform): " << total_converted_transform_ms << " ms ("
                  << (static_cast<double>(count) / total_converted_transform_ms * 1000.0) << " vecs/sec)\n";

        if (native_transform_ms < total_converted_transform_ms) {
            std::cout << ">>> Native FP16 Transform is " << (total_converted_transform_ms / native_transform_ms)
                      << "x FASTER than Convert + FP32 Transform\n";
        } else {
            std::cout << ">>> Convert + FP32 Transform is " << (native_transform_ms / total_converted_transform_ms)
                      << "x faster than Native FP16 Transform\n";
        }

        // Even without counting conversion time:
        std::cout << "Pure Transform comparison (FP16 kernel vs FP32 kernel): ";
        if (native_transform_ms < f32_transform_ms) {
            std::cout << "FP16 is " << (f32_transform_ms / native_transform_ms) << "x faster than FP32 kernel\n";
        } else {
            std::cout << "FP32 kernel is " << (native_transform_ms / f32_transform_ms) << "x faster than FP16 kernel\n";
        }
    }
    std::cout << "======================================================================\n\n";
}

