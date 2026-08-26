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
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::TailOnly>>(deglib::distances::fp32_l2::select_dist(7, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdOnly>>(deglib::distances::fp32_l2::select_dist(16, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdTail>>(deglib::distances::fp32_l2::select_dist(25, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualOnly>>(deglib::distances::fp32_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualPlusSimd>>(deglib::distances::fp32_l2::select_dist(112, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::Full>>(deglib::distances::fp32_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512)));
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::TailOnly>>(deglib::distances::fp32_l2::select_dist(7, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdOnly>>(deglib::distances::fp32_l2::select_dist(8, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdTail>>(deglib::distances::fp32_l2::select_dist(13, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualOnly>>(deglib::distances::fp32_l2::select_dist(16, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::fp32_l2::select_dist(24, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::Full>>(deglib::distances::fp32_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2)));
    }
#endif
}

TEST(DeglibDistanceSelection, FP32_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::TailOnly>>(deglib::distances::fp32_ip::select_dist(7, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::SimdOnly>>(deglib::distances::fp32_ip::select_dist(16, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::SimdTail>>(deglib::distances::fp32_ip::select_dist(25, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualOnly>>(deglib::distances::fp32_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp32_ip::select_dist(112, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::Full>>(deglib::distances::fp32_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512))
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::TailOnly>>(deglib::distances::fp32_ip::select_dist(7, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::SimdOnly>>(deglib::distances::fp32_ip::select_dist(8, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::SimdTail>>(deglib::distances::fp32_ip::select_dist(13, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualOnly>>(deglib::distances::fp32_ip::select_dist(16, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::fp32_ip::select_dist(24, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::Full>>(deglib::distances::fp32_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2))
        );
    }
#endif
}

TEST(DeglibDistanceSelection, Uint8_L2_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::TailOnly>>(deglib::distances::uint8_l2::select_dist(15, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::SimdOnly>>(deglib::distances::uint8_l2::select_dist(32, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::SimdTail>>(deglib::distances::uint8_l2::select_dist(45, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualOnly>>(deglib::distances::uint8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualPlusSimd>>(deglib::distances::uint8_l2::select_dist(96, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::Full>>(deglib::distances::uint8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512)));
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::TailOnly>>(deglib::distances::uint8_l2::select_dist(15, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::SimdOnly>>(deglib::distances::uint8_l2::select_dist(16, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::SimdTail>>(deglib::distances::uint8_l2::select_dist(25, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualOnly>>(deglib::distances::uint8_l2::select_dist(32, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::uint8_l2::select_dist(48, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::Full>>(deglib::distances::uint8_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2)));
    }
#endif
}

TEST(DeglibDistanceSelection, Uint8_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::TailOnly>>(deglib::distances::uint8_ip::select_dist(15, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::SimdOnly>>(deglib::distances::uint8_ip::select_dist(32, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::SimdTail>>(deglib::distances::uint8_ip::select_dist(45, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::DualOnly>>(deglib::distances::uint8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::uint8_ip::select_dist(96, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX512<ResidualMode::Full>>(deglib::distances::uint8_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512))
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::TailOnly>>(deglib::distances::uint8_ip::select_dist(15, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::SimdOnly>>(deglib::distances::uint8_ip::select_dist(16, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::SimdTail>>(deglib::distances::uint8_ip::select_dist(25, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::DualOnly>>(deglib::distances::uint8_ip::select_dist(32, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::DualPlusSimd>>(
                deglib::distances::uint8_ip::select_dist(48, deglib::cpu::InstructionSet::AVX2)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::uint8_ip::InnerProductUint8_AVX2<ResidualMode::Full>>(deglib::distances::uint8_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2))
        );
    }
#endif
}

TEST(DeglibDistanceSelection, FP16_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::TailOnly>>(deglib::distances::fp16_ip::select_dist(7, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::SimdOnly>>(deglib::distances::fp16_ip::select_dist(16, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::SimdTail>>(deglib::distances::fp16_ip::select_dist(25, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualOnly>>(deglib::distances::fp16_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualPlusSimd>>(
                deglib::distances::fp16_ip::select_dist(112, deglib::cpu::InstructionSet::AVX512)
            )
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::Full>>(deglib::distances::fp16_ip::select_dist(127, deglib::cpu::InstructionSet::AVX512))
        );
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::TailOnly>>(deglib::distances::fp16_ip::select_dist(7, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::SimdOnly>>(deglib::distances::fp16_ip::select_dist(8, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::SimdTail>>(deglib::distances::fp16_ip::select_dist(13, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualOnly>>(deglib::distances::fp16_ip::select_dist(16, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::fp16_ip::select_dist(24, deglib::cpu::InstructionSet::AVX2))
        );
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::Full>>(deglib::distances::fp16_ip::select_dist(127, deglib::cpu::InstructionSet::AVX2))
        );
    }
#endif
}

TEST(DeglibDistanceSelection, FP16_L2_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::TailOnly>>(deglib::distances::fp16_l2::select_dist(7, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::SimdOnly>>(deglib::distances::fp16_l2::select_dist(16, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::SimdTail>>(deglib::distances::fp16_l2::select_dist(25, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::DualOnly>>(deglib::distances::fp16_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512)));
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::DualPlusSimd>>(deglib::distances::fp16_l2::select_dist(112, deglib::cpu::InstructionSet::AVX512))
        );
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX512<ResidualMode::Full>>(deglib::distances::fp16_l2::select_dist(127, deglib::cpu::InstructionSet::AVX512)));
    }
    if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::TailOnly>>(deglib::distances::fp16_l2::select_dist(7, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::SimdOnly>>(deglib::distances::fp16_l2::select_dist(8, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::SimdTail>>(deglib::distances::fp16_l2::select_dist(13, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::DualOnly>>(deglib::distances::fp16_l2::select_dist(16, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::fp16_l2::select_dist(24, deglib::cpu::InstructionSet::AVX2)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_l2::L2FP16_AVX2<ResidualMode::Full>>(deglib::distances::fp16_l2::select_dist(127, deglib::cpu::InstructionSet::AVX2)));
    }
#endif
}

TEST(DeglibDistanceSelection, Int8_IP_SelectDist) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512_vnni()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX512_VNNI<ResidualMode::DualOnly>>(
                deglib::distances::int8_ip::select_dist(256, deglib::cpu::InstructionSet::AVX512_VNNI)
            )
        );
    }
    if (deglib::cpu::has_avx_vnni()) {
        EXPECT_TRUE(
            std::holds_alternative<deglib::distances::int8_ip::InnerProductInt8_AVX2_VNNI<ResidualMode::DualOnly>>(
                deglib::distances::int8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI)
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
    if (deglib::cpu::has_avx512_vnni()) {
        EXPECT_NO_THROW(deglib::distances::fp32_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
        EXPECT_NO_THROW(deglib::distances::fp32_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
        EXPECT_NO_THROW(deglib::distances::fp16_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
        EXPECT_NO_THROW(deglib::distances::uint8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
        EXPECT_NO_THROW(deglib::distances::uint8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
        EXPECT_NO_THROW(deglib::distances::evp_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
        EXPECT_NO_THROW(deglib::distances::int8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
        EXPECT_NO_THROW(deglib::distances::int8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
    }
    if (deglib::cpu::has_avx_vnni()) {
        EXPECT_NO_THROW(deglib::distances::fp32_l2::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
        EXPECT_NO_THROW(deglib::distances::fp32_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
        EXPECT_NO_THROW(deglib::distances::fp16_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
        EXPECT_NO_THROW(deglib::distances::uint8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
        EXPECT_NO_THROW(deglib::distances::uint8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
        EXPECT_NO_THROW(deglib::distances::evp_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
        EXPECT_NO_THROW(deglib::distances::int8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
        EXPECT_NO_THROW(deglib::distances::int8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
    }
#endif
}
