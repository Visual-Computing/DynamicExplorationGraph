#include "test_regression.h"

TEST(DeglibRegressionL2Uint8, Benchmark_AVX512_Ext32)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("AVX512_Ext32", deglib::Metric::L2_Uint8, 77000.0, 4.6, 0.97,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8Ext32_AVX512{}, 500);
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, Benchmark_AVX2_Ext32)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("AVX2_Ext32", deglib::Metric::L2_Uint8, 78000.0, 4.6, 0.97,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8Ext32_AVX2{}, 500);
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, Benchmark_SSE_Ext32)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("SSE_Ext32", deglib::Metric::L2_Uint8, 73000.0, 5.1, 0.97,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8Ext32_SSE{}, 500);
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, Benchmark_Scalar)
{
    size_t dim = 128;
    size_t base_count = 100000;
    size_t query_count = 100;
    size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    auto gt_data = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);

    run_regression_test("Scalar", deglib::Metric::L2_Uint8, 66000.0, 5.5, 0.97,
                        base_data.data(), query_data.data(), base_count, query_count, dim, gt_data,
                        deglib::distances::uint8_l2::L2Uint8{}, 500);
}

// Distance recall tests: each SIMD variant is verified against scalar ground truth in its own test.

TEST(DeglibRegressionL2Uint8, DistanceRecall_L2Uint8Ext32_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Uint8Ext32_AVX512", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::uint8_l2::L2Uint8Ext32_AVX512::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, DistanceRecall_L2Uint8Ext32_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Uint8Ext32_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::uint8_l2::L2Uint8Ext32_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, DistanceRecall_L2Uint8Ext16_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Uint8Ext16_AVX2", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::uint8_l2::L2Uint8Ext16_AVX2::compare(a, b, qty); });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, DistanceRecall_L2Uint8Ext32_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Uint8Ext32_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::uint8_l2::L2Uint8Ext32_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, DistanceRecall_L2Uint8Ext16_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Uint8Ext16_SSE", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::uint8_l2::L2Uint8Ext16_SSE::compare(a, b, qty); });
#else
    GTEST_SKIP() << "SSE not available on this platform";
#endif
}

TEST(DeglibRegressionL2Uint8, DistanceRecall_L2Uint8)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("L2Uint8", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::uint8_l2::L2Uint8::compare(a, b, qty); });
}
