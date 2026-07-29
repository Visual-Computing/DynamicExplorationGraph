#include "common/test_helpers.h"

// ============================================================================
// Builder Performance Regression Benchmarks
// ============================================================================
// Consolidated 100k performance benchmark tests covering:
//   - L2 Float metric (AVX512, AVX2, SSE, Scalar variants)
//   - InnerProduct Float metric (AVX512, AVX2, SSE, Scalar variants)
//   - L2 Uint8 metric (AVX512, AVX2, SSE, Scalar variants)
//   - InnerProduct FP16 metric (AVX512, AVX2 32Ext, AVX2 16Ext, AVX2 8Ext, Scalar variants)
//   - OptimizationTarget modes (LowLID, HighLID, StreamingData)
//
// Each benchmark measures QPS, build time, and recall on 100,000 base vectors.
// QPS and build-time assertions are skipped when SKIP_PERFORMANCE_TESTS is set.
// ============================================================================

// ---------------------------------------------------------------------------
// L2 Float Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, L2Float_Benchmark_AVX512_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Float_AVX512_16Ext", deglib::Metric::L2, 46000.0, 6.0, 0.961,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float16Ext_AVX512{}, 100);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, L2Float_Benchmark_AVX2_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Float_AVX2_16Ext", deglib::Metric::L2, 43000.0, 6.0, 0.961,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float16Ext_AVX2{}, 100);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderRegression, L2Float_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Float_Scalar", deglib::Metric::L2, 21000.0, 14.0, 0.971,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100);
}

// ---------------------------------------------------------------------------
// InnerProduct Float Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, InnerProductFloat_Benchmark_AVX512_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFloat_AVX512_16Ext", deglib::Metric::InnerProduct, 38000.0, 4.5, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX512{}, 50);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, InnerProductFloat_Benchmark_AVX2_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFloat_AVX2_16Ext", deglib::Metric::InnerProduct, 38000.0, 4.5, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX2{}, 50);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderRegression, InnerProductFloat_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFloat_Scalar", deglib::Metric::InnerProduct, 18000.0, 9.8, 0.867,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat{}, 50);
}

// ---------------------------------------------------------------------------
// L2 Uint8 Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, L2Uint8_Benchmark_AVX512_32Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Uint8_AVX512_32Ext", deglib::Metric::L2_Uint8, 60000.0, 4.8, 0.99,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8Ext32_AVX512{}, 500);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, L2Uint8_Benchmark_AVX2_32Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Uint8_AVX2_32Ext", deglib::Metric::L2_Uint8, 60000.0, 4.9, 0.99,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8Ext32_AVX2{}, 500);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderRegression, L2Uint8_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("L2Uint8_Scalar", deglib::Metric::L2_Uint8, 50000.0, 5.8, 0.99,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8{}, 500);
}


// ---------------------------------------------------------------------------
// InnerProduct FP16 Metric Benchmarks (100k)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, InnerProductFP16_Benchmark_AVX512_32Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFP16_AVX512_32Ext", deglib::Metric::FP16InnerProduct, 47000.0, 4.2, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp16_ip::InnerProductFP16_32Ext_AVX512{}, 50);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, InnerProductFP16_Benchmark_AVX2_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFP16_AVX2_16Ext", deglib::Metric::FP16InnerProduct, 47000.0, 4.2, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp16_ip::InnerProductFP16_16Ext_AVX2{}, 50);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibBuilderRegression, InnerProductFP16_Benchmark_AVX2_8Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFP16_AVX2_8Ext", deglib::Metric::FP16InnerProduct, 45000.0, 4.4, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp16_ip::InnerProductFP16_8Ext_AVX2{}, 50);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}


TEST(DeglibBuilderRegression, InnerProductFP16_Benchmark_Scalar)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("InnerProductFP16_Scalar", deglib::Metric::FP16InnerProduct, 8000.0, 25.0, 0.867,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp16_ip::InnerProductFP16{}, 50);
}

// ---------------------------------------------------------------------------
// OptimizationTarget Benchmarks (100k, L2 Float)
// ---------------------------------------------------------------------------

TEST(DeglibBuilderRegression, Benchmark_LowLID)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("LowLID", deglib::Metric::L2, 20000.0, 14, 0.971,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::LowLID);
}

TEST(DeglibBuilderRegression, Benchmark_HighLID)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("HighLID", deglib::Metric::L2, 12000.0, 14.4, 0.96,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::HighLID);
}

TEST(DeglibBuilderRegression, Benchmark_StreamingData)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("StreamingData", deglib::Metric::L2, 16000.0, 30, 0.91,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100,
                        deglib::builder::OptimizationTarget::StreamingData);
}

