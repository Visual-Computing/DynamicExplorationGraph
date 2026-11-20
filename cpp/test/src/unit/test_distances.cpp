#include "deglib/distances.h"

#include <gtest/gtest.h>

using deglib::distances::ResidualMode;

// ============================================================================
// Distance Selection Unit Tests
// ============================================================================
// Verifies that select_dist(dim, instruction) returns the expected distance variant
// and ResidualMode specialization for explicit InstructionSets.
// ============================================================================

TEST(DeglibDistanceSelection, FP32_L2_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::TailOnly>>(
                deglib::distances::fp32_l2::select_dist(7, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdOnly>>(
                deglib::distances::fp32_l2::select_dist(16, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdTail>>(
                deglib::distances::fp32_l2::select_dist(25, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::fp32_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp32_l2::select_dist(112, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::Full>>(
                deglib::distances::fp32_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::TailOnly>>(
                deglib::distances::fp32_l2::select_dist(7, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdOnly>>(
                deglib::distances::fp32_l2::select_dist(8, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdTail>>(
                deglib::distances::fp32_l2::select_dist(13, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::fp32_l2::select_dist(16, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp32_l2::select_dist(24, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::Full>>(
                deglib::distances::fp32_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, FP32_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::TailOnly>>(
                deglib::distances::fp32_ip::select_dist(7, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::SimdOnly>>(
                deglib::distances::fp32_ip::select_dist(16, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::SimdTail>>(
                deglib::distances::fp32_ip::select_dist(25, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::fp32_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp32_ip::select_dist(112, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::Full>>(
                deglib::distances::fp32_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::TailOnly>>(
                deglib::distances::fp32_ip::select_dist(7, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::SimdOnly>>(
                deglib::distances::fp32_ip::select_dist(8, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::SimdTail>>(
                deglib::distances::fp32_ip::select_dist(13, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::fp32_ip::select_dist(16, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp32_ip::select_dist(24, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::Full>>(
                deglib::distances::fp32_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, Uint8_L2_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::TailOnly>>(
                deglib::distances::uint8_l2::select_dist(15, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::SimdOnly>>(
                deglib::distances::uint8_l2::select_dist(32, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::SimdTail>>(
                deglib::distances::uint8_l2::select_dist(45, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::uint8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::uint8_l2::select_dist(96, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::Full>>(
                deglib::distances::uint8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::TailOnly>>(
                deglib::distances::uint8_l2::select_dist(15, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::SimdOnly>>(
                deglib::distances::uint8_l2::select_dist(16, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::SimdTail>>(
                deglib::distances::uint8_l2::select_dist(25, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::uint8_l2::select_dist(32, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualPlusSimd>>(
                deglib::distances::uint8_l2::select_dist(48, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::Full>>(
                deglib::distances::uint8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, Uint8_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::TailOnly>>(
                deglib::distances::uint8_ip::select_dist(15, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::SimdOnly>>(
                deglib::distances::uint8_ip::select_dist(32, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::SimdTail>>(
                deglib::distances::uint8_ip::select_dist(45, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::uint8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::uint8_ip::select_dist(96, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::Full>>(
                deglib::distances::uint8_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::TailOnly>>(
                deglib::distances::uint8_ip::select_dist(15, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::SimdOnly>>(
                deglib::distances::uint8_ip::select_dist(16, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::SimdTail>>(
                deglib::distances::uint8_ip::select_dist(25, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::uint8_ip::select_dist(32, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::DualPlusSimd>>(
                deglib::distances::uint8_ip::select_dist(48, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::Full>>(
                deglib::distances::uint8_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, FP16_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::TailOnly>>(
                deglib::distances::fp16_ip::select_dist(7, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::SimdOnly>>(
                deglib::distances::fp16_ip::select_dist(16, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::SimdTail>>(
                deglib::distances::fp16_ip::select_dist(25, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::fp16_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp16_ip::select_dist(112, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::Full>>(
                deglib::distances::fp16_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::TailOnly>>(
                deglib::distances::fp16_ip::select_dist(7, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::SimdOnly>>(
                deglib::distances::fp16_ip::select_dist(8, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::SimdTail>>(
                deglib::distances::fp16_ip::select_dist(13, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::fp16_ip::select_dist(16, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp16_ip::select_dist(24, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::Full>>(
                deglib::distances::fp16_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, FP16_L2_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::TailOnly>>(
                deglib::distances::fp16_l2::select_dist(7, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::SimdOnly>>(
                deglib::distances::fp16_l2::select_dist(16, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::SimdTail>>(
                deglib::distances::fp16_l2::select_dist(25, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::fp16_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp16_l2::select_dist(112, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::Full>>(
                deglib::distances::fp16_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::TailOnly>>(
                deglib::distances::fp16_l2::select_dist(7, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::SimdOnly>>(
                deglib::distances::fp16_l2::select_dist(8, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::SimdTail>>(
                deglib::distances::fp16_l2::select_dist(13, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::fp16_l2::select_dist(16, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp16_l2::select_dist(24, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::Full>>(
                deglib::distances::fp16_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, Int8_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512_vnni()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX512_VNNI<ResidualMode::SimdOnly>>(
                deglib::distances::int8_ip::select_dist(256, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX512_VNNI<ResidualMode::Full>>(
                deglib::distances::int8_ip::select_dist(255, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
    }
    if (deglib::cpu::has_avx2_vnni()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX2_VNNI<ResidualMode::SimdOnly>>(
                deglib::distances::int8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX2_VNNI<ResidualMode::Full>>(
                deglib::distances::int8_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
    }
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::int8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::int8_ip::select_dist(64, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, Int8_L2_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_l2::L2Int8_AVX512<ResidualMode::DualOnly>>(
                deglib::distances::int8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512)
            )
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_l2::L2Int8_AVX2<ResidualMode::DualOnly>>(
                deglib::distances::int8_l2::select_dist(64, deglib::cpu::InstructionSet::AVX2)
            )
        );
    }
#endif
}

TEST(DeglibDistanceSelection, VNNIFallbackAcrossAllMetrics) {
#if defined(DEGLIB_X86)
    // If VNNI is requested on a metric that has no dedicated VNNI kernel,
    // it MUST return the AVX512 (or AVX2) specialization, NEVER Scalar!
    if (deglib::cpu::has_avx512_vnni()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::Full>>(
                deglib::distances::fp32_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::Full>>(
                deglib::distances::fp32_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::Full>>(
                deglib::distances::fp16_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::Full>>(
                deglib::distances::fp16_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::Full>>(
                deglib::distances::uint8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_l2::L2Int8_AVX512<ResidualMode::Full>>(
                deglib::distances::int8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
        if (deglib::cpu::has_avx512_vpopcntdq()) {
            EXPECT_TRUE(
                std::holds_alternative<deglib::distances::evp_ip::EvpInnerProduct_AVX512<ResidualMode::Full>>(
                    deglib::distances::evp_ip::select_dist(1536 + 128, deglib::cpu::InstructionSet::AVX512_VNNI)
                )
            );
        } else {
            EXPECT_TRUE(
                std::holds_alternative<deglib::distances::evp_ip::EvpInnerProduct_AVX2<ResidualMode::Full>>(
                    deglib::distances::evp_ip::select_dist(1536 + 128, deglib::cpu::InstructionSet::AVX512_VNNI)
                )
            );
        }
    }
    if (deglib::cpu::has_avx2_vnni()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::Full>>(
                deglib::distances::fp32_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::Full>>(
                deglib::distances::fp32_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::Full>>(
                deglib::distances::fp16_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::Full>>(
                deglib::distances::fp16_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::Full>>(
                deglib::distances::uint8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_l2::L2Int8_AVX2<ResidualMode::Full>>(
                deglib::distances::int8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::evp_ip::EvpInnerProduct_AVX2<ResidualMode::Full>>(
                deglib::distances::evp_ip::select_dist(768 + 128, deglib::cpu::InstructionSet::AVX2_VNNI)
            )
        );
    }
#endif
}

TEST(DeglibDistances, ComputeDistancesVectorOverload) {
    deglib::distances::FloatSpace space(3, deglib::distances::Metric::FP32_L2);
    const std::vector<float> queries = {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> targets = {1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 3.0f};

    const auto dists = deglib::distances::compute_distances(space, queries.data(), 2, targets.data(), 2);
    ASSERT_EQ(dists.size(), 4u);
    EXPECT_FLOAT_EQ(dists[0], 0.0f);
    EXPECT_FLOAT_EQ(dists[1], 25.0f);
    EXPECT_FLOAT_EQ(dists[2], 14.0f);
    EXPECT_FLOAT_EQ(dists[3], 61.0f);
}

TEST(DeglibDistances, BatchAndStridedKernelMatchSingleCompare) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    const float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float targets[3][4] = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {2.0f, 4.0f, 6.0f, 8.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };
    const void* ptrs[3] = {targets[0], targets[1], targets[2]};

    const auto single = space.get_dist_func();
    const auto batch = space.get_batch_dist_func();
    const auto strided = space.get_strided_dist_func();
    const auto param = space.get_dist_func_param();

    float batch_no_pf[3] = {};
    float batch_pf[3] = {};
    float strided_no_pf[3] = {};
    float strided_pf[3] = {};

    batch(query, ptrs, 3, param, batch_no_pf, 0, 0);
    batch(query, ptrs, 3, param, batch_pf, 2, space.get_data_size());
    strided(query, targets, space.get_data_size(), 3, param, strided_no_pf, 0, 0);
    strided(query, targets, space.get_data_size(), 3, param, strided_pf, 2, space.get_data_size());

    for (size_t i = 0; i < 3; ++i) {
        const float expected = single(query, targets[i], param);
        EXPECT_FLOAT_EQ(batch_no_pf[i], expected);
        EXPECT_FLOAT_EQ(batch_pf[i], expected);
        EXPECT_FLOAT_EQ(strided_no_pf[i], expected);
        EXPECT_FLOAT_EQ(strided_pf[i], expected);
    }
}
