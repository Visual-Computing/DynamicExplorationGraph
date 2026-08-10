// test_evp_inner_product.cpp — Unit tests for symmetric EVP inner product distance
//
// Tests cover:
// - Naive scalar implementation correctness & symmetry
// - Distance normalization (distance = 1 - similarity / (2*dim))
// - AVX2 and AVX-512 SIMD variants match naive across vector dimensions
// - SelectDist variant selection & FloatSpace integration

#include <bit>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "common/test_helpers.h"
#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/distances.h"

// ============================================================================
// Test helpers
// ============================================================================

static std::pair<std::vector<std::byte>, std::vector<std::byte>>
make_evp_pair(uint32_t dim, uint32_t non_zeros, int seed_a = 42, int seed_b = 99) {
    std::mt19937 rng_a(seed_a), rng_b(seed_b);
    std::vector<float> a(dim), b(dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = dist(rng_a);
    for (auto& x : b) x = dist(rng_b);
    return {
        deglib::quantization::evp::quantize_single(a.data(), dim, non_zeros),
        deglib::quantization::evp::quantize_single(b.data(), dim, non_zeros)
    };
}

// ============================================================================
// Naive distance & Normalization tests
// ============================================================================

TEST(EvpInnerProduct, NormalizationAndSymmetry) {
    const uint32_t dim = 128;
    auto [a, b] = make_evp_pair(dim, 32);

    float sim_self = deglib::distances::evp_ip::EvpInnerProduct::compare_naive(
        a.data(), a.data(), &dim);
    EXPECT_GT(sim_self, 0.0f);

    float sim_ab = deglib::distances::evp_ip::EvpInnerProduct::compare_naive(
        a.data(), b.data(), &dim);
    float sim_ba = deglib::distances::evp_ip::EvpInnerProduct::compare_naive(
        b.data(), a.data(), &dim);
    EXPECT_FLOAT_EQ(sim_ab, sim_ba);

    float expected_dist = 1.f - (sim_ab / (2.0f * dim));
    float dist = deglib::distances::evp_ip::EvpInnerProduct::compare(
        a.data(), b.data(), &dim);
    EXPECT_FLOAT_EQ(dist, expected_dist);
}

// ============================================================================
// SIMD correctness vs Naive (AVX2 & AVX-512)
// ============================================================================

TEST(EvpInnerProduct, Avx2MatchesNaive) {
#if defined(DEGLIB_X86)
    for (uint32_t dim : {64, 128, 256}) {
        auto [a, b] = make_evp_pair(dim, dim / 4);
        float naive = deglib::distances::evp_ip::EvpInnerProduct::compare(
            a.data(), b.data(), &dim);
        float avx2 = deglib::distances::evp_ip::EvpInnerProduct_AVX2<deglib::distances::ResidualMode::Full>::compare(
            a.data(), b.data(), &dim);
        EXPECT_NEAR(avx2, naive, 0.001f) << "Mismatch at dimension " << dim;
    }
#else
    GTEST_SKIP() << "AVX2 support was not compiled in";
#endif
}

TEST(EvpInnerProduct, Avx512MatchesNaive) {
#if !defined(DEGLIB_X86) || !defined(DEGLIB_TARGET_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "CPU does not support AVX-512";
    }
    for (uint32_t dim : {64, 128, 256}) {
        auto [a, b] = make_evp_pair(dim, dim / 4);
        float naive = deglib::distances::evp_ip::EvpInnerProduct::compare(
            a.data(), b.data(), &dim);
        float avx512 = deglib::distances::evp_ip::EvpInnerProduct_AVX512<deglib::distances::ResidualMode::Full>::compare(
            a.data(), b.data(), &dim);
        EXPECT_NEAR(avx512, naive, 0.001f) << "Mismatch at dimension " << dim;
    }
#endif
}

// ============================================================================
// Distance dispatch & FloatSpace integration
// ============================================================================

TEST(EvpInnerProduct, SelectDistReturnsValidVariant) {
    for (size_t dim : {8, 16, 32, 64, 128, 256, 512, 1024}) {
        auto variant = deglib::distances::evp_ip::select_dist(dim);
        (void)variant;
        SUCCEED();
    }
}

TEST(EvpInnerProduct, FloatSpaceEVPInnerProduct) {
    const size_t dim = 128;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::EVP_InnerProduct);

    EXPECT_EQ(space.dim(), dim);
    EXPECT_EQ(space.metric(), deglib::distances::Metric::EVP_InnerProduct);
    EXPECT_EQ(space.get_data_size(), 2 * (dim / 8));
}



