#include "test_regression.h"

TEST(DeglibRegressionL2, Benchmark_AVX512_16Ext)
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

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("AVX512_16Ext", deglib::Metric::L2, 42000.0, 6.0, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float16Ext_AVX512{}, 100);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionL2, Benchmark_AVX2_16Ext)
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

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("AVX2_16Ext", deglib::Metric::L2, 39000.0, 6.2, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float16Ext_AVX2{}, 100);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionL2, Benchmark_SSE_16Ext)
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

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("SSE_16Ext", deglib::Metric::L2, 35000.0, 7.0, 0.866,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float16Ext_SSE{}, 100);
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2, Benchmark_Scalar)
{
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("Scalar", deglib::Metric::L2, 22000.0, 12.5, 0.867,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::fp32_l2::L2Float{}, 100);
}

// Distance recall tests: each SIMD variant is verified against scalar ground truth in its own test.

TEST(DeglibRegressionL2, DistanceRecall_L2Float16Ext_AVX512)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float16Ext_AVX512", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float16Ext_AVX512::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float16ExtResiduals_AVX512)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float16ExtResiduals_AVX512", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float16ExtResiduals_AVX512::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float16Ext_AVX2)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float16Ext_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float16Ext_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float8Ext_AVX2)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float8Ext_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float8Ext_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float16ExtResiduals_AVX2)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float16ExtResiduals_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float16ExtResiduals_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float16Ext_SSE)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float16Ext_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float16Ext_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float8Ext_SSE)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float8Ext_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float8Ext_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float4Ext_SSE)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float4Ext_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float4Ext_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float16ExtResiduals_SSE)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float16ExtResiduals_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float16ExtResiduals_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float4ExtResiduals_SSE)
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

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float4ExtResiduals_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float4ExtResiduals_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2, DistanceRecall_L2Float)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Float", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp32_l2::L2Float::compare(a, b, qty); });
}
