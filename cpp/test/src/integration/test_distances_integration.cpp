#include "common/test_helpers.h"

// ============================================================================
// Distance Calculation Recall Integration Tests
// ============================================================================
// Verifies that every SIMD distance variant and ResidualMode preset produces
// identical top-K results to the scalar ground truth.
// ============================================================================

using deglib::distances::ResidualMode;

// ---------------------------------------------------------------------------
// L2 Float distance recall tests
// ---------------------------------------------------------------------------

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float_AVX512_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<float> base_data, query_data;
        generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("L2Float_AVX512_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX512_DualPlusSimd", 112, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX512_DualTail", 125, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualTail>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX512_DualOnly", 128, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX512_SimdTail", 25, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdTail>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX512_SimdOnly", 16, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX512_TailOnly", 7, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibDistanceIntegration, DistanceRecall_L2Float_AVX2_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<float> base_data, query_data;
        generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("L2Float_AVX2_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX2_DualPlusSimd", 24, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX2_DualTail", 21, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualTail>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX2_DualOnly", 16, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX2_SimdTail", 13, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdTail>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX2_SimdOnly", 8, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("L2Float_AVX2_TailOnly", 7, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
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

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat_AVX512_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<float> base_data, query_data;
        generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("IPFloat_AVX512_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX512_DualOnly", 128, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX512_DualPlusSimd", 112, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX512_SimdOnly", 16, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX512_TailOnly", 7, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFloat_AVX2_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<float> base_data, query_data;
        generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("IPFloat_AVX2_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX2_DualOnly", 16, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX2_DualPlusSimd", 24, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX2_SimdOnly", 8, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("IPFloat_AVX2_TailOnly", 7, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
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

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8_AVX512_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<uint8_t> base_data, query_data;
        generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("L2Uint8_AVX512_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX512_DualOnly", 128, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX512_DualPlusSimd", 96, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX512_SimdOnly", 32, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX512_TailOnly", 15, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibDistanceIntegration, DistanceRecall_L2Uint8_AVX2_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<uint8_t> base_data, query_data;
        generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("L2Uint8_AVX2_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX2_DualOnly", 32, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX2_DualPlusSimd", 48, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX2_SimdOnly", 16, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("L2Uint8_AVX2_TailOnly", 15, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
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

// ---------------------------------------------------------------------------
// FP16 Inner Product distance recall tests
// ---------------------------------------------------------------------------

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFP16_AVX512_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<uint16_t> base_data, query_data;
        generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("FP16_IP_AVX512_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX512_DualOnly", 128, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX512_DualPlusSimd", 112, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX512_SimdOnly", 16, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX512_TailOnly", 7, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX512 not available on this platform";
#endif
}

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFP16_AVX2_Modes)
{
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    auto test_mode = [&](const char* name, size_t dim, auto dist_fn) {
        std::vector<uint16_t> base_data, query_data;
        generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, 1000);
        auto gt_scalar = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, k);
        check_distance_recall(name, base_data, base_count, query_data, query_count, dim, k, gt_scalar, dist_fn);
    };

    test_mode("FP16_IP_AVX2_Full", 127, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::Full>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX2_DualOnly", 16, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualOnly>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX2_DualPlusSimd", 24, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualPlusSimd>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX2_SimdOnly", 8, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::SimdOnly>::compare(a, b, qty);
    });
    test_mode("FP16_IP_AVX2_TailOnly", 7, [](const void* a, const void* b, const void* qty) {
        return deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::TailOnly>::compare(a, b, qty);
    });
#else
    GTEST_SKIP() << "AVX2 not available on this platform";
#endif
}

TEST(DeglibDistanceIntegration, DistanceRecall_InnerProductFP16)
{
    const size_t dim = 128;
    const size_t base_count = 10000;
    const size_t query_count = 100;
    const uint32_t k = 10;

    std::vector<uint16_t> base_data;
    std::vector<uint16_t> query_data;
    generate_synthetic_clustered_dataset_fp16(base_count, dim, base_data, query_data, query_count, 1000);

    auto gt_scalar = compute_groundtruth_fp16_ip(base_data, base_count, query_data, query_count, dim, k);

    check_distance_recall("InnerProductFP16", base_data, base_count, query_data, query_count, dim, k, gt_scalar,
                          [](const void* a, const void* b, const void* qty)
                          { return deglib::distances::fp16_ip::InnerProductFP16::compare(a, b, qty); });
}
