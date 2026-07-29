#include "test_builder_integration.h"

// ============================================================================
// Distance Calculation Recall Integration Tests
// ============================================================================
// Verifies that every SIMD distance variant produces identical top-K results
// to the scalar ground truth.  Uses 10,000 base vectors and 100 queries.
// ============================================================================

// ---------------------------------------------------------------------------
// L2 Float distance recall tests
// ---------------------------------------------------------------------------

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float16Ext_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float16ExtResiduals_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float16Ext_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float8Ext_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float16ExtResiduals_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float16Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float8Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float4Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float16ExtResiduals_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float4ExtResiduals_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
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

// ---------------------------------------------------------------------------
// InnerProduct Float distance recall tests
// ---------------------------------------------------------------------------

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat16Ext_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat16ExtResiduals_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat16Ext_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat8Ext_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat16ExtResiduals_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat16Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat8Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat4Ext_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat16ExtResiduals_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat4ExtResiduals_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
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

// ---------------------------------------------------------------------------
// L2 Uint8 distance recall tests
// ---------------------------------------------------------------------------

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8Ext32_AVX512)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8Ext32_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8Ext16_AVX2)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8Ext32_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8Ext16_SSE)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_sse42()) {
        GTEST_SKIP() << "SSE4.2 not available on this CPU";
    }
    const size_t dim = 128;
    const size_t base_count = 10000;
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
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
