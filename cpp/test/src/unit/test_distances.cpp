#include "deglib/distances.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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

TEST(DeglibDistances, UInt7L2MaskedTailMatchesScalar) {
#if defined(DEGLIB_X86)
    if (!deglib::cpu::has_avx512()) GTEST_SKIP() << "AVX-512 not supported";
    for (const size_t dim : {33u, 63u, 64u, 72u, 127u, 200u}) {
        std::vector<int8_t> a(dim), b(dim);
        for (size_t i = 0; i < dim; ++i) {
            a[i] = static_cast<int8_t>((i * 17 + 3) % 128);
            b[i] = static_cast<int8_t>((i * 31 + 11) % 128);
        }
        const deglib::distances::FloatSpace scalar_space(dim, deglib::distances::Metric::UInt7_L2, deglib::cpu::InstructionSet::Scalar);
        const deglib::distances::FloatSpace avx512_space(dim, deglib::distances::Metric::UInt7_L2, deglib::cpu::InstructionSet::AVX512);
        const float expected = deglib::distances::compute_distance(scalar_space, a.data(), b.data());
        const float actual = deglib::distances::compute_distance(avx512_space, a.data(), b.data());
        EXPECT_FLOAT_EQ(actual, expected) << "dim=" << dim;
    }
#endif
}

namespace {

inline void fill_metric_test_buffer(const deglib::distances::Metric metric, const size_t dim, const size_t seed, std::vector<uint8_t>& buf) {
    using deglib::distances::Metric;
    std::fill(buf.begin(), buf.end(), uint8_t{0});

    switch (metric) {
        case Metric::FP32_L2:
        case Metric::FP32_InnerProduct: {
            for (size_t i = 0; i < dim; ++i) {
                const int centered = static_cast<int>((i * 37 + seed * 19 + 7) % 401) - 200;
                const float val = static_cast<float>(centered) * 0.01f;
                std::memcpy(buf.data() + i * sizeof(float), &val, sizeof(float));
            }
            break;
        }
        case Metric::FP16_L2:
        case Metric::FP16_InnerProduct: {
            std::vector<float> tmp(dim);
            for (size_t i = 0; i < dim; ++i) {
                const int centered = static_cast<int>((i * 41 + seed * 23 + 11) % 401) - 200;
                tmp[i] = static_cast<float>(centered) * 0.01f;
            }
            deglib::distances::fp16::floats_to_fp16(tmp.data(), reinterpret_cast<uint16_t*>(buf.data()), dim);
            break;
        }
        case Metric::Int8_InnerProduct: {
            auto* p = reinterpret_cast<int8_t*>(buf.data());
            for (size_t i = 0; i < dim; ++i) {
                // Keep values in [-127, 127] so VNNI vpdpbusd sign_epi8 avoids int8(-128) negation overflow.
                p[i] = static_cast<int8_t>(static_cast<int>((i * 53 + seed * 29 + 3) % 255) - 127);
            }
            break;
        }
        case Metric::Int8_L2: {
            auto* p = reinterpret_cast<int8_t*>(buf.data());
            for (size_t i = 0; i < dim; ++i) {
                p[i] = static_cast<int8_t>(static_cast<int>((i * 53 + seed * 29 + 3) % 256) - 128);
            }
            break;
        }
        case Metric::Uint8_L2:
        case Metric::Uint8_InnerProduct: {
            for (size_t i = 0; i < dim; ++i) {
                buf[i] = static_cast<uint8_t>((i * 67 + seed * 41 + 5) % 256);
            }
            break;
        }
        case Metric::UInt7_L2:
        case Metric::UInt7_InnerProduct: {
            for (size_t i = 0; i < dim; ++i) {
                buf[i] = static_cast<uint8_t>((i * 47 + seed * 31 + 11) % 128);
            }
            break;
        }
        case Metric::Uint4_L2:
        case Metric::Uint4_InnerProduct: {
            // High nibble of final byte for odd dim remains 0x0 per zero-padded packing contract.
            for (size_t i = 0; i < dim; ++i) {
                const uint8_t nib = static_cast<uint8_t>((i * 11 + seed * 7 + 3) % 16);
                if ((i & 1) == 0) {
                    buf[i / 2] |= nib;
                } else {
                    buf[i / 2] |= static_cast<uint8_t>(nib << 4);
                }
            }
            break;
        }
        case Metric::Int4_L2:
        case Metric::Int4_InnerProduct: {
            // High nibble of final byte for odd dim remains 0x0 per zero-padded packing contract.
            for (size_t i = 0; i < dim; ++i) {
                const int8_t sval = static_cast<int8_t>(static_cast<int>((i * 13 + seed * 5 + 1) % 16) - 8);
                const uint8_t nib = static_cast<uint8_t>(sval) & 0x0F;
                if ((i & 1) == 0) {
                    buf[i / 2] |= nib;
                } else {
                    buf[i / 2] |= static_cast<uint8_t>(nib << 4);
                }
            }
            break;
        }
        case Metric::EVP_InnerProduct: {
            const size_t mask_bytes = (dim + 7) / 8;
            uint8_t* ones = buf.data();
            uint8_t* negs = buf.data() + mask_bytes;
            for (size_t j = 0; j < dim; ++j) {
                const size_t choice = (j * 29 + seed * 17 + 1) % 5;
                const uint8_t bit = static_cast<uint8_t>(1u << (j & 7));
                if (choice < 2) {
                    ones[j >> 3] |= bit;
                } else if (choice < 4) {
                    negs[j >> 3] |= bit;
                }
            }
            break;
        }
    }
}

}  // namespace

TEST(DeglibDistances, AllMetricsMaskedTailMatchesScalar) {
#if defined(DEGLIB_X86)
    using deglib::cpu::InstructionSet;
    using deglib::distances::Metric;

    std::vector<InstructionSet> isas;
    if (deglib::cpu::has_avx512_vnni()) isas.push_back(InstructionSet::AVX512_VNNI);
    if (deglib::cpu::has_avx512()) isas.push_back(InstructionSet::AVX512);
    if (deglib::cpu::has_avx2_vnni()) isas.push_back(InstructionSet::AVX2_VNNI);
    if (deglib::cpu::has_avx2()) isas.push_back(InstructionSet::AVX2);
    if (isas.empty()) GTEST_SKIP() << "No SIMD ISA supported on host";

    constexpr std::array<Metric, 15> metrics = {
        Metric::FP32_L2,  Metric::FP32_InnerProduct,  Metric::FP16_L2,          Metric::FP16_InnerProduct,  Metric::Int8_L2,  Metric::Int8_InnerProduct,
        Metric::Uint8_L2, Metric::Uint8_InnerProduct, Metric::UInt7_L2,         Metric::UInt7_InnerProduct, Metric::Uint4_L2, Metric::Uint4_InnerProduct,
        Metric::Int4_L2,  Metric::Int4_InnerProduct,  Metric::EVP_InnerProduct,
    };

    constexpr std::array<size_t, 39> dims = {
        1,   2,   3,   4,   7,   8,   9,   15,  16,  17,  24,  31,  32,  33,  48,  63,  64,  65,  80,   96,
        127, 128, 129, 199, 200, 229, 253, 255, 256, 257, 260, 272, 288, 384, 511, 512, 513, 768, 1024,
    };

    for (const Metric metric : metrics) {
        for (const size_t dim : dims) {
            const deglib::distances::FloatSpace scalar_space(dim, metric, InstructionSet::Scalar);
            std::vector<uint8_t> a(scalar_space.get_data_size());
            std::vector<uint8_t> b(scalar_space.get_data_size());
            fill_metric_test_buffer(metric, dim, 1, a);
            fill_metric_test_buffer(metric, dim, 2, b);

            const float expected = deglib::distances::compute_distance(scalar_space, a.data(), b.data());

            for (const InstructionSet isa : isas) {
                const deglib::distances::FloatSpace simd_space(dim, metric, isa);
                ASSERT_EQ(simd_space.get_data_size(), scalar_space.get_data_size()) << "metric=" << metric.to_string() << " dim=" << dim;
                const float actual = deglib::distances::compute_distance(simd_space, a.data(), b.data());

                if (metric == Metric::FP32_L2 || metric == Metric::FP32_InnerProduct) {
                    EXPECT_NEAR(actual, expected, std::max(1e-4f, 1e-4f * std::abs(expected)))
                        << "metric=" << metric.to_string() << " dim=" << dim << " isa=" << static_cast<int>(isa);
                } else if (metric == Metric::FP16_L2 || metric == Metric::FP16_InnerProduct) {
                    EXPECT_NEAR(actual, expected, std::max(1e-3f, 1e-3f * std::abs(expected)))
                        << "metric=" << metric.to_string() << " dim=" << dim << " isa=" << static_cast<int>(isa);
                } else if (metric == Metric::EVP_InnerProduct) {
                    EXPECT_NEAR(actual, expected, 1e-5f) << "metric=" << metric.to_string() << " dim=" << dim << " isa=" << static_cast<int>(isa);
                } else {
                    EXPECT_FLOAT_EQ(actual, expected) << "metric=" << metric.to_string() << " dim=" << dim << " isa=" << static_cast<int>(isa);
                }
            }
        }
    }

    // Verify specialized Optimal Int8 L2 kernels directly across all dims.
    for (const size_t dim : dims) {
        std::vector<int8_t> a_opt(dim), b_opt(dim);
        for (size_t i = 0; i < dim; ++i) {
            a_opt[i] = static_cast<int8_t>(static_cast<int>((i * 53 + 17) % 256) - 128);
            b_opt[i] = static_cast<int8_t>(static_cast<int>((i * 71 + 43) % 256) - 128);
        }
        const float expected_opt = deglib::distances::int8_l2::L2Int8::compare(a_opt.data(), b_opt.data(), &dim);

        if (deglib::cpu::has_avx512()) {
            EXPECT_FLOAT_EQ(deglib::distances::int8_l2::L2Int8_AVX512_Optimal<ResidualMode::Full>::compare(a_opt.data(), b_opt.data(), &dim), expected_opt)
                << "AVX512_Optimal dim=" << dim;
        }
        if (deglib::cpu::has_avx2()) {
            EXPECT_FLOAT_EQ(deglib::distances::int8_l2::L2Int8_AVX2_Optimal<ResidualMode::Full>::compare(a_opt.data(), b_opt.data(), &dim), expected_opt)
                << "AVX2_Optimal dim=" << dim;
        }
    }
#endif
}
