#include <gtest/gtest.h>
#include "distances.h"

using deglib::distances::ResidualMode;

// ============================================================================
// Distance Selection Unit Tests
// ============================================================================
// Verifies that select_dist(dim) chooses the exact expected distance variant
// and ResidualMode specialization based on vector dimension alignment.
// ============================================================================

TEST(DeglibDistanceSelection, FP32_L2_SelectDist)
{
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::TailOnly>>(deglib::distances::fp32_l2::select_dist(7)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdOnly>>(deglib::distances::fp32_l2::select_dist(16)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::SimdTail>>(deglib::distances::fp32_l2::select_dist(25)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualOnly>>(deglib::distances::fp32_l2::select_dist(128)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::DualPlusSimd>>(deglib::distances::fp32_l2::select_dist(112)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX512<ResidualMode::Full>>(deglib::distances::fp32_l2::select_dist(127)));
    } else if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::TailOnly>>(deglib::distances::fp32_l2::select_dist(7)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdOnly>>(deglib::distances::fp32_l2::select_dist(8)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::SimdTail>>(deglib::distances::fp32_l2::select_dist(13)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualOnly>>(deglib::distances::fp32_l2::select_dist(16)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::fp32_l2::select_dist(24)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_l2::L2Float_AVX2<ResidualMode::Full>>(deglib::distances::fp32_l2::select_dist(127)));
    }
#endif
}

TEST(DeglibDistanceSelection, FP32_IP_SelectDist)
{
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::TailOnly>>(deglib::distances::fp32_ip::select_dist(7)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::SimdOnly>>(deglib::distances::fp32_ip::select_dist(16)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::SimdTail>>(deglib::distances::fp32_ip::select_dist(25)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualOnly>>(deglib::distances::fp32_ip::select_dist(128)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>>(deglib::distances::fp32_ip::select_dist(112)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX512<ResidualMode::Full>>(deglib::distances::fp32_ip::select_dist(127)));
    } else if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::TailOnly>>(deglib::distances::fp32_ip::select_dist(7)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::SimdOnly>>(deglib::distances::fp32_ip::select_dist(8)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::SimdTail>>(deglib::distances::fp32_ip::select_dist(13)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualOnly>>(deglib::distances::fp32_ip::select_dist(16)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::fp32_ip::select_dist(24)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp32_ip::InnerProductFloat_AVX2<ResidualMode::Full>>(deglib::distances::fp32_ip::select_dist(127)));
    }
#endif
}

TEST(DeglibDistanceSelection, Uint8_L2_SelectDist)
{
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::TailOnly>>(deglib::distances::uint8_l2::select_dist(15)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::SimdOnly>>(deglib::distances::uint8_l2::select_dist(32)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::SimdTail>>(deglib::distances::uint8_l2::select_dist(45)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualOnly>>(deglib::distances::uint8_l2::select_dist(128)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::DualPlusSimd>>(deglib::distances::uint8_l2::select_dist(96)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX512<ResidualMode::Full>>(deglib::distances::uint8_l2::select_dist(127)));
    } else if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::TailOnly>>(deglib::distances::uint8_l2::select_dist(15)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::SimdOnly>>(deglib::distances::uint8_l2::select_dist(16)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::SimdTail>>(deglib::distances::uint8_l2::select_dist(25)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualOnly>>(deglib::distances::uint8_l2::select_dist(32)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::uint8_l2::select_dist(48)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::uint8_l2::L2Uint8_AVX2<ResidualMode::Full>>(deglib::distances::uint8_l2::select_dist(127)));
    }
#endif
}

TEST(DeglibDistanceSelection, FP16_IP_SelectDist)
{
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::TailOnly>>(deglib::distances::fp16_ip::select_dist(7)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::SimdOnly>>(deglib::distances::fp16_ip::select_dist(16)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::SimdTail>>(deglib::distances::fp16_ip::select_dist(25)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualOnly>>(deglib::distances::fp16_ip::select_dist(128)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::DualPlusSimd>>(deglib::distances::fp16_ip::select_dist(112)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX512<ResidualMode::Full>>(deglib::distances::fp16_ip::select_dist(127)));
    } else if (deglib::cpu::has_avx2()) {
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::TailOnly>>(deglib::distances::fp16_ip::select_dist(7)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::SimdOnly>>(deglib::distances::fp16_ip::select_dist(8)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::SimdTail>>(deglib::distances::fp16_ip::select_dist(13)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualOnly>>(deglib::distances::fp16_ip::select_dist(16)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::DualPlusSimd>>(deglib::distances::fp16_ip::select_dist(24)));
        EXPECT_TRUE(std::holds_alternative<deglib::distances::fp16_ip::InnerProductFP16_AVX2<ResidualMode::Full>>(deglib::distances::fp16_ip::select_dist(127)));
    }
#endif
}
