#include "test_regression.h"

// Regression tests for InnerProduct metric: each SIMD variant gets its own benchmark test.
// num_runs=50 extends the search measurement window to ~280ms per run,
// reducing QPS noise from OS jitter and CPU power-state transitions.

TEST(DeglibRegressionIP, Benchmark_AVX512_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("AVX512_16Ext", deglib::Metric::InnerProduct, 18000.0, 10.3, 0.774,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX512{}, 50);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionIP, Benchmark_AVX2_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("AVX2_16Ext", deglib::Metric::InnerProduct, 17000.0, 10.3, 0.774,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX2{}, 50);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionIP, Benchmark_SSE_16Ext)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("SSE_16Ext", deglib::Metric::InnerProduct, 16000.0, 11.8, 0.774,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat16Ext_SSE{}, 50);
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionIP, Benchmark_Scalar)
{
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("Scalar", deglib::Metric::InnerProduct, 13500.0, 15.8, 0.78,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_ip::InnerProductFloat{}, 50);
}

// Distance recall tests: each SIMD variant is verified against scalar ground truth in its own test.

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat16Ext_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat16Ext_AVX512", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX512::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat16ExtResiduals_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat16ExtResiduals_AVX512", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat16ExtResiduals_AVX512::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat16Ext_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat16Ext_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat16Ext_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat8Ext_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat8Ext_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat8Ext_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat16ExtResiduals_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat16ExtResiduals_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat16ExtResiduals_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat16Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat16Ext_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat16Ext_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat8Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat8Ext_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat8Ext_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat4Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat4Ext_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat4Ext_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat16ExtResiduals_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat16ExtResiduals_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat16ExtResiduals_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat4ExtResiduals_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat4ExtResiduals_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat4ExtResiduals_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionIP, DistanceRecall_InnerProductFloat)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFloat", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_ip::InnerProductFloat::compare(a, b, qty); });
}
