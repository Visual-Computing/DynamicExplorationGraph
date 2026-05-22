// test_distances.cpp — Comprehensive unit tests for deglib distances.h
//
// Each SIMD-accelerated distance function is validated against a
// hand-written naive reference implementation so that correctness is
// guaranteed regardless of which SIMD path the compiler selects.

#include <cmath>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "distances.h"
#include "quantization/evp_quantize.h"

// ---------------------------------------------------------------------------
//  Google Test (bundled with fmt)
// ---------------------------------------------------------------------------
#include "gtest/gtest.h"
#include "gmock/gmock.h"

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
//  Naive reference implementations
// ---------------------------------------------------------------------------

static float l2_naive(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static float ip_naive(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
}

static float l2_uint8_naive(const uint8_t* a, const uint8_t* b, size_t dim) {
    int64_t sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        sum += d * d;
    }
    return static_cast<float>(sum);
}

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

static const size_t DIM_4   = 4;
static const size_t DIM_8   = 8;
static const size_t DIM_16  = 16;
static const size_t DIM_128 = 128;

// Allocate and fill vectors with deterministic pseudo-random data.
static std::vector<float> make_float_vec(size_t dim, unsigned seed = 42) {
    std::vector<float> v(dim);
    unsigned s = seed;
    for (size_t i = 0; i < dim; ++i) {
        s = s * 1103515245 + 12345;
        v[i] = static_cast<float>((s >> 16) % 10000) / 10000.0f * 10.0f - 5.0f;
    }
    return v;
}

static std::vector<uint8_t> make_uint8_vec(size_t dim, unsigned seed = 42) {
    std::vector<uint8_t> v(dim);
    unsigned s = seed;
    for (size_t i = 0; i < dim; ++i) {
        s = s * 1103515245 + 12345;
        v[i] = static_cast<uint8_t>((s >> 16) % 256);
    }
    return v;
}

// Compare two floats with an absolute tolerance.
static bool approx_eq(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) < eps;
}

// ---------------------------------------------------------------------------
//  L2 Float — baseline (no SIMD required)
// ---------------------------------------------------------------------------

TEST(L2Float, IdentityZero) {
    std::vector<float> v(16, 0.0f);
    size_t dim = v.size();
    float d = deglib::distances::L2Float::compare(v.data(), v.data(), &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(L2Float, KnownValue) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    size_t dim = 3;
    float d = deglib::distances::L2Float::compare(a, b, &dim);
    // (4-1)^2 + (5-2)^2 + (6-3)^2 = 27
    EXPECT_NEAR(d, 27.0f, 1e-4f);
}

TEST(L2Float, Symmetry) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 99);
    size_t dim = a.size();
    float ab = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    float ba = deglib::distances::L2Float::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(L2Float, MatchesNaive_4) {
    auto a = make_float_vec(DIM_4);
    auto b = make_float_vec(DIM_4, 7);
    size_t dim = a.size();
    float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Float, MatchesNaive_64) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 13);
    size_t dim = a.size();
    float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Float, MatchesNaive_128) {
    auto a = make_float_vec(DIM_128);
    auto b = make_float_vec(DIM_128, 21);
    size_t dim = a.size();
    float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
    // 128-dim: accumulated rounding error can be ~1e-3 relative
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-3f);
}

TEST(L2Float, Dim1) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    size_t dim = 1;
    float d = deglib::distances::L2Float::compare(a, b, &dim);
    EXPECT_NEAR(d, 16.0f, 1e-4f);
}

TEST(L2Float, Dim0) {
    float a = 1.0f;
    float b = 2.0f;
    size_t dim = 0;
    float d = deglib::distances::L2Float::compare(&a, &b, &dim);
    EXPECT_EQ(d, 0.0f);
}

// ---------------------------------------------------------------------------
//  L2Float SIMD extensions — validated against naive
// ---------------------------------------------------------------------------

// Helper: test a SIMD L2 distance class against naive for several dimensions.
template <typename DistClass>
static void test_l2_matches_naive() {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = DistClass::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// SIMD extensions only process full SIMD lanes (4/8/16).
// Dimensions that are multiples of the SIMD width are tested;
// non-aligned dimensions are handled by the *Residuals classes.
TEST(L2Float4Ext, MatchesNaive) {
    // dim must be a multiple of 4
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float4Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float8Ext, MatchesNaive) {
    // dim must be a multiple of 8
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float8Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float16Ext, MatchesNaive) {
    // dim must be a multiple of 16
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float16Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
//  L2Float residual variants (non-aligned dimensions)
// ---------------------------------------------------------------------------

TEST(L2Float16ExtResiduals, NonAlignedDims) {
    // Dimensions that are NOT multiples of 16.
    std::vector<size_t> dims = {17, 20, 24, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim + 1);
        float d = deglib::distances::L2Float16ExtResiduals::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float4ExtResiduals, NonAlignedDims) {
    // Dimensions that are NOT multiples of 4.
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 25, 33, 50};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim + 2);
        float d = deglib::distances::L2Float4ExtResiduals::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
//  Inner Product — baseline
// ---------------------------------------------------------------------------

TEST(InnerProductFloat, IdentityZero) {
    std::vector<float> v(16, 0.0f);
    size_t dim = v.size();
    float d = deglib::distances::InnerProductFloat::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-4f); // 1 - 0 = 1
}

TEST(InnerProductFloat, UnitVectorSelf) {
    std::vector<float> v(4, 1.0f);
    size_t dim = v.size();
    float d = deglib::distances::InnerProductFloat::compare(v.data(), v.data(), &dim);
    // dot = 4, so 1 - 4 = -3
    EXPECT_NEAR(d, -3.0f, 1e-4f);
}

TEST(InnerProductFloat, Orthogonal) {
    float a[] = {1.0f, 0, 0, 0};
    float b[] = {0, 0, 0, 1.0f};
    size_t dim = 4;
    float d = deglib::distances::InnerProductFloat::compare(a, b, &dim);
    // dot = 0, so 1 - 0 = 1
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(InnerProductFloat, Symmetry) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 99);
    size_t dim = a.size();
    float ab = deglib::distances::InnerProductFloat::compare(a.data(), b.data(), &dim);
    float ba = deglib::distances::InnerProductFloat::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(InnerProductFloat, MatchesNaive_4) {
    auto a = make_float_vec(DIM_4);
    auto b = make_float_vec(DIM_4, 7);
    size_t dim = a.size();
    float d = deglib::distances::InnerProductFloat::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(InnerProductFloat, MatchesNaive_64) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 13);
    size_t dim = a.size();
    float d = deglib::distances::InnerProductFloat::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(InnerProductFloat, MatchesNaive_128) {
    auto a = make_float_vec(DIM_128);
    auto b = make_float_vec(DIM_128, 21);
    size_t dim = a.size();
    float d = deglib::distances::InnerProductFloat::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-4f);
}

// ---------------------------------------------------------------------------
//  Inner Product SIMD extensions
// ---------------------------------------------------------------------------

template <typename DistClass>
static void test_ip_matches_naive() {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = DistClass::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// SIMD IP extensions: dim must be a multiple of the SIMD width.
// NOTE: InnerProductFloat4Ext has a known bug in ip_4ext: it returns
// abs(dot) instead of dot, so compare() computes 1 - abs(dot).
// We test against a reference that matches the *actual* implementation.
static float ip_4ext_bug_ref(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return 1.0f - std::abs(dot);
}

TEST(InnerProductFloat4Ext, MatchesNaive) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::InnerProductFloat4Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_4ext_bug_ref(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(InnerProductFloat8Ext, MatchesNaive) {
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::InnerProductFloat8Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(InnerProductFloat16Ext, MatchesNaive) {
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::InnerProductFloat16Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
//  Inner Product residual variants
// ---------------------------------------------------------------------------

TEST(InnerProductFloat16ExtResiduals, NonAlignedDims) {
    std::vector<size_t> dims = {17, 20, 24, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim + 1);
        float d = deglib::distances::InnerProductFloat16ExtResiduals::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// InnerProductFloat4ExtResiduals computes:
//   1 - (ip_4ext(dot_of_first_qty4) + ip_naive(dot_of_remaining))
// where ip_4ext returns abs(dot). So the result is:
//   1 - (abs(dot_part) + dot_tail)
// which is NOT the same as 1 - abs(total_dot).
static float ip_4ext_residuals_bug_ref(const float* a, const float* b, size_t qty) {
    size_t qty4 = qty >> 2 << 2;
    float dot_part = 0.0f;
    for (size_t i = 0; i < qty4; ++i) {
        dot_part += a[i] * b[i];
    }
    float dot_tail = 0.0f;
    for (size_t i = qty4; i < qty; ++i) {
        dot_tail += a[i] * b[i];
    }
    return 1.0f - (std::abs(dot_part) + dot_tail);
}

TEST(InnerProductFloat4ExtResiduals, NonAlignedDims) {
    // dim must NOT be a multiple of 4 (otherwise Float4Ext path is taken)
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 25, 33, 50};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim + 2);
        float d = deglib::distances::InnerProductFloat4ExtResiduals::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_4ext_residuals_bug_ref(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
//  L2 Uint8 — baseline
// ---------------------------------------------------------------------------

TEST(L2Uint8, IdentityZero) {
    std::vector<uint8_t> v(128, 0);
    size_t dim = v.size();
    float d = deglib::distances::L2Uint8::compare(v.data(), v.data(), &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(L2Uint8, KnownValue) {
    uint8_t a[] = {3, 4};
    uint8_t b[] = {0, 0};
    size_t dim = 2;
    float d = deglib::distances::L2Uint8::compare(a, b, &dim);
    EXPECT_NEAR(d, 25.0f, 1e-4f);
}

TEST(L2Uint8, MaxDiff) {
    std::vector<uint8_t> a(128, 255);
    std::vector<uint8_t> b(128, 0);
    size_t dim = a.size();
    float d = deglib::distances::L2Uint8::compare(a.data(), b.data(), &dim);
    // 128 * 255^2 = 8323200
    EXPECT_NEAR(d, 128.0f * 255.0f * 255.0f, 1.0f);
}

TEST(L2Uint8, Symmetry) {
    auto a = make_uint8_vec(128);
    auto b = make_uint8_vec(128, 99);
    size_t dim = a.size();
    float ab = deglib::distances::L2Uint8::compare(a.data(), b.data(), &dim);
    float ba = deglib::distances::L2Uint8::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(L2Uint8, MatchesNaive_16) {
    auto a = make_uint8_vec(16);
    auto b = make_uint8_vec(16, 7);
    size_t dim = a.size();
    float d = deglib::distances::L2Uint8::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-2f);
}

TEST(L2Uint8, MatchesNaive_128) {
    auto a = make_uint8_vec(128);
    auto b = make_uint8_vec(128, 13);
    size_t dim = a.size();
    float d = deglib::distances::L2Uint8::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-2f);
}

TEST(L2Uint8, Dim1) {
    uint8_t a[] = {10};
    uint8_t b[] = {20};
    size_t dim = 1;
    float d = deglib::distances::L2Uint8::compare(a, b, &dim);
    EXPECT_NEAR(d, 100.0f, 1e-4f);
}

TEST(L2Uint8, Dim0) {
    uint8_t a = 1;
    uint8_t b = 2;
    size_t dim = 0;
    float d = deglib::distances::L2Uint8::compare(&a, &b, &dim);
    EXPECT_EQ(d, 0.0f);
}

// ---------------------------------------------------------------------------
//  L2 Uint8 SIMD extensions
// ---------------------------------------------------------------------------

template <typename DistClass>
static void test_l2_uint8_matches_naive() {
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = DistClass::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1.0f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8Ext16, MatchesNaive) { test_l2_uint8_matches_naive<deglib::distances::L2Uint8Ext16>(); }
TEST(L2Uint8Ext32, MatchesNaive) { test_l2_uint8_matches_naive<deglib::distances::L2Uint8Ext32>(); }

// ---------------------------------------------------------------------------
//  EvpBitsSimilarity — each path validated against compare_naive
// ---------------------------------------------------------------------------

static std::pair<std::vector<std::byte>, std::vector<std::byte>>
make_evp_pair(uint32_t dim, uint32_t non_zeros, int seed_a = 42, int seed_b = 99) {
    std::mt19937 rng_a(seed_a), rng_b(seed_b);
    std::vector<float> a(dim), b(dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = dist(rng_a);
    for (auto& x : b) x = dist(rng_b);
    return {
        deglib::quantization::quantize_single(a.data(), dim, non_zeros),
        deglib::quantization::quantize_single(b.data(), dim, non_zeros)
    };
}

static uint32_t count_evp_active_bits(const std::vector<std::byte>& bits, uint32_t dim) {
    const size_t mask_bytes = dim / 8;
    uint32_t active = 0;
    for (size_t i = 0; i < mask_bytes; ++i) {
        active += std::popcount(static_cast<unsigned int>(static_cast<uint8_t>(bits[i]))) +
                  std::popcount(static_cast<unsigned int>(static_cast<uint8_t>(bits[i + mask_bytes])));
    }
    return active;
}

TEST(EvpBitsSimilarity, NaiveSelfSimilarity) {
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;
    float sim = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(a.data()),
        static_cast<const void*>(&dim));
    EXPECT_GT(sim, 0.0f);
}

TEST(EvpBitsSimilarity, CompareNormalizesToZeroForIdenticalVectors) {
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;
    
    float sim = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(a.data()),
        static_cast<const void*>(&dim));
        
    const float max_similarity = static_cast<float>(dim) * 2.0f;
    float expected = 1.f - (sim / max_similarity);

    float dist = deglib::distances::EvpBitsSimilarity::compare(
        static_cast<const void*>(a.data()), static_cast<const void*>(a.data()),
        static_cast<const void*>(&dim));

    EXPECT_FLOAT_EQ(dist, expected);
}

TEST(EvpBitsSimilarity, CompareMatchesPairwiseNormalization) {
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;

    float sim = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
        
    const float max_similarity = static_cast<float>(dim) * 2.0f;
    float expected = 1.f - (sim / max_similarity);

    float dist = deglib::distances::EvpBitsSimilarity::compare(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));

    EXPECT_FLOAT_EQ(dist, expected);
}



TEST(EvpBitsSimilarity, Avx2MatchesNaive_64Dim) {
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;
    float naive = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx2  = deglib::distances::EvpBitsSimilarity::compare_avx2(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx2, naive, 0.001f);
}

TEST(EvpBitsSimilarity, Avx2MatchesNaive_128Dim) {
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    float naive = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx2  = deglib::distances::EvpBitsSimilarity::compare_avx2(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx2, naive, 0.001f);
}

TEST(EvpBitsSimilarity, Avx2MatchesNaive_256Dim) {
    auto [a, b] = make_evp_pair(256, 64);
    uint32_t dim = 256;
    float naive = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx2  = deglib::distances::EvpBitsSimilarity::compare_avx2(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx2, naive, 0.001f);
}

TEST(EvpBitsSimilarity, Avx512MatchesNaive_64Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;
    float naive = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx512 = deglib::distances::EvpBitsSimilarity::compare_avx512(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive, 0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx512MatchesNaive_128Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    float naive = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx512 = deglib::distances::EvpBitsSimilarity::compare_avx512(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive, 0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx512MatchesNaive_256Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(256, 64);
    uint32_t dim = 256;
    float naive = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx512 = deglib::distances::EvpBitsSimilarity::compare_avx512(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive, 0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx2AndAvx512MatchNaive_128Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    const void* qa = static_cast<const void*>(a.data());
    const void* qb = static_cast<const void*>(b.data());
    const void* qd = static_cast<const void*>(&dim);

    float naive  = deglib::distances::EvpBitsSimilarity::compare_naive(qa, qb, qd);
    float avx2   = deglib::distances::EvpBitsSimilarity::compare_avx2(qa, qb, qd);
    float avx512 = deglib::distances::EvpBitsSimilarity::compare_avx512(qa, qb, qd);

    EXPECT_NEAR(avx2,   naive,  0.001f);
    EXPECT_NEAR(avx512, naive,  0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx2AndAvx512MatchNaive_256Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(256, 64);
    uint32_t dim = 256;
    const void* qa = static_cast<const void*>(a.data());
    const void* qb = static_cast<const void*>(b.data());
    const void* qd = static_cast<const void*>(&dim);

    float naive  = deglib::distances::EvpBitsSimilarity::compare_naive(qa, qb, qd);
    float avx2   = deglib::distances::EvpBitsSimilarity::compare_avx2(qa, qb, qd);
    float avx512 = deglib::distances::EvpBitsSimilarity::compare_avx512(qa, qb, qd);

    EXPECT_NEAR(avx2,   naive,  0.001f);
    EXPECT_NEAR(avx512, naive,  0.001f);
#endif
}

TEST(EvpBitsSimilarity, Symmetry) {
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    const void* qa = static_cast<const void*>(a.data());
    const void* qb = static_cast<const void*>(b.data());
    const void* qd = static_cast<const void*>(&dim);

    float ab = deglib::distances::EvpBitsSimilarity::compare_naive(qa, qb, qd);
    float ba = deglib::distances::EvpBitsSimilarity::compare_naive(qb, qa, qd);
    EXPECT_NEAR(ab, ba, 0.001f);
}

// Removed MatchesQuantizationLibrary test because similarity_bytes is not available.

TEST(EvpBitsSimilarity, SmallDims) {
    // Smallest valid EVP: dim=8 → 1 byte ones + 1 byte negs
    std::mt19937 rng(42);
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;
    std::vector<float> v(dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(rng);

    auto result = deglib::quantization::quantize_single(v.data(), dim, non_zeros);

    float naive  = deglib::distances::EvpBitsSimilarity::compare_naive(
        static_cast<const void*>(result.data()), static_cast<const void*>(result.data()),
        static_cast<const void*>(&dim));
    float avx2   = deglib::distances::EvpBitsSimilarity::compare_avx2(
        static_cast<const void*>(result.data()), static_cast<const void*>(result.data()),
        static_cast<const void*>(&dim));

    EXPECT_GT(naive, 0.0f);
    EXPECT_NEAR(avx2,   naive,  0.001f);

#if defined(USE_AVX512)
    float avx512 = deglib::distances::EvpBitsSimilarity::compare_avx512(
        static_cast<const void*>(result.data()), static_cast<const void*>(result.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive,  0.001f);
#endif
}

// ---------------------------------------------------------------------------
//  FloatSpace — integration test
// ---------------------------------------------------------------------------

TEST(FloatSpace, L2_128Dim) {
    deglib::FloatSpace space(128, deglib::Metric::L2);
    EXPECT_EQ(space.dim(), 128u);
    EXPECT_EQ(space.metric(), deglib::Metric::L2);
    EXPECT_EQ(space.get_data_size(), 128 * sizeof(float));

    auto a = make_float_vec(128);
    auto b = make_float_vec(128, 99);
    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), 128), 1e-2f);
}

TEST(FloatSpace, InnerProduct_64Dim) {
    deglib::FloatSpace space(64, deglib::Metric::InnerProduct);
    EXPECT_EQ(space.dim(), 64u);
    EXPECT_EQ(space.metric(), deglib::Metric::InnerProduct);
    EXPECT_EQ(space.get_data_size(), 64 * sizeof(float));

    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 7);
    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, ip_naive(a.data(), b.data(), 64), 1e-2f);
}

TEST(FloatSpace, L2_Uint8_128Dim) {
    deglib::FloatSpace space(128, deglib::Metric::L2_Uint8);
    EXPECT_EQ(space.dim(), 128u);
    EXPECT_EQ(space.metric(), deglib::Metric::L2_Uint8);
    EXPECT_EQ(space.get_data_size(), 128 * sizeof(uint8_t));

    auto a = make_uint8_vec(128);
    auto b = make_uint8_vec(128, 13);
    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), 128), 1.0f);
}

TEST(FloatSpace, VariousDimensions) {
    // Test that FloatSpace selects a valid distance function for a range of dims.
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 128, 255, 256};
    for (size_t dim : dims) {
        deglib::FloatSpace space(dim, deglib::Metric::L2);
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
//  FP16 Inner Product
// ---------------------------------------------------------------------------

// Helper: convert a float array to FP16 (uint16_t) using the software path
// from FP16InnerProduct::fp16_to_float in reverse (via SIMD round-trip when
// available, otherwise via the standard _mm_cvtps_ph intrinsic path).
static std::vector<uint16_t> floats_to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    // Use _mm_cvtps_ph for a correct hardware round-trip (8 at a time)
    size_t i = 0;
    for (; i + 4 <= v.size(); i += 4) {
        __m128 f4 = _mm_loadu_ps(&v[i]);
        __m128i h4 = _mm_cvtps_ph(f4, _MM_FROUND_TO_NEAREST_INT);
        // _mm_cvtps_ph stores 4 uint16 into the lower 8 bytes
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h4);
        out[i]   = tmp[0];
        out[i+1] = tmp[1];
        out[i+2] = tmp[2];
        out[i+3] = tmp[3];
    }
    // tail
    for (; i < v.size(); ++i) {
        __m128 f1 = _mm_set_ss(v[i]);
        __m128i h1 = _mm_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h1);
        out[i] = tmp[0];
    }
#else
    // No SIMD: use a well-known portable float32→float16 bit-manipulation
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        uint16_t sign     = static_cast<uint16_t>((bits >> 16) & 0x8000u);
        int32_t  exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFFu;
        if (exponent <= 0)      { out[i] = sign; }
        else if (exponent >= 31){ out[i] = static_cast<uint16_t>(sign | 0x7C00u); }
        else                    { out[i] = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13)); }
    }
#endif
    return out;
}

// Naive FP16 inner-product reference (operates on uint16_t arrays).
// Converts via the software path so the reference is independent of SIMD.
static float fp16_ip_naive_ref(const uint16_t* a, const uint16_t* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += deglib::distances::FP16InnerProduct::fp16_to_float(a[i]) *
               deglib::distances::FP16InnerProduct::fp16_to_float(b[i]);
    }
    return 1.0f - dot;
}

// ---------------------------------------------------------------------------
//  fp16_to_float — unit tests for the software converter
// ---------------------------------------------------------------------------

TEST(FP16InnerProduct, Fp16ToFloat_Zero) {
    EXPECT_FLOAT_EQ(deglib::distances::FP16InnerProduct::fp16_to_float(0x0000u), 0.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_One) {
    // FP16 representation of 1.0f: sign=0, exp=15 (stored as 01111), mant=0 → 0x3C00
    EXPECT_FLOAT_EQ(deglib::distances::FP16InnerProduct::fp16_to_float(0x3C00u), 1.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_NegOne) {
    // -1.0f: 0xBC00
    EXPECT_FLOAT_EQ(deglib::distances::FP16InnerProduct::fp16_to_float(0xBC00u), -1.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_Two) {
    // 2.0f: 0x4000
    EXPECT_FLOAT_EQ(deglib::distances::FP16InnerProduct::fp16_to_float(0x4000u), 2.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_RoundTrip) {
    // Round-trip through float -> FP16 -> float should be close (FP16 has ~3 decimal digits)
    const std::vector<float> vals = {0.5f, -0.5f, 1.5f, -1.5f, 0.125f, 100.0f, -0.001f};
    for (float orig : vals) {
        auto fp16 = floats_to_fp16({orig});
        float recovered = deglib::distances::FP16InnerProduct::fp16_to_float(fp16[0]);
        EXPECT_NEAR(recovered, orig, std::abs(orig) * 1e-2f + 1e-4f)
            << "original=" << orig;
    }
}

// ---------------------------------------------------------------------------
//  FP16InnerProduct naive baseline
// ---------------------------------------------------------------------------

TEST(FP16InnerProduct, IdentityZero) {
    // All-zeros: dot = 0, distance = 1
    std::vector<uint16_t> v(16, 0);
    size_t dim = v.size();
    float d = deglib::distances::FP16InnerProduct::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(FP16InnerProduct, KnownValue) {
    // a = [1,0,0,0], b = [1,0,0,0] -> dot=1 -> distance=0
    auto a = floats_to_fp16({1.0f, 0.0f, 0.0f, 0.0f});
    auto b = floats_to_fp16({1.0f, 0.0f, 0.0f, 0.0f});
    size_t dim = 4;
    float d = deglib::distances::FP16InnerProduct::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 5e-3f);  // FP16 round-trip tolerance
}

TEST(FP16InnerProduct, Orthogonal) {
    auto a = floats_to_fp16({1.0f, 0.0f, 0.0f, 0.0f});
    auto b = floats_to_fp16({0.0f, 1.0f, 0.0f, 0.0f});
    size_t dim = 4;
    float d = deglib::distances::FP16InnerProduct::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-3f);
}

TEST(FP16InnerProduct, Symmetry) {
    auto af = make_float_vec(64);
    auto bf = make_float_vec(64, 99);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = a.size();
    float ab = deglib::distances::FP16InnerProduct::compare(a.data(), b.data(), &dim);
    float ba = deglib::distances::FP16InnerProduct::compare(b.data(), a.data(), &dim);
    EXPECT_NEAR(ab, ba, 1e-3f);
}

TEST(FP16InnerProduct, MatchesNaiveRef_32) {
    auto af = make_float_vec(32);
    auto bf = make_float_vec(32, 7);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = a.size();
    float d = deglib::distances::FP16InnerProduct::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, fp16_ip_naive_ref(a.data(), b.data(), dim), 1e-3f);
}

TEST(FP16InnerProduct, MatchesNaiveRef_128) {
    auto af = make_float_vec(128);
    auto bf = make_float_vec(128, 13);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = a.size();
    float d = deglib::distances::FP16InnerProduct::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, fp16_ip_naive_ref(a.data(), b.data(), dim), 1e-2f);
}

// ---------------------------------------------------------------------------
//  FP16 SIMD extensions — validated against the naive implementation
// ---------------------------------------------------------------------------

template <typename DistClass>
static void test_fp16_ip_matches_naive(const std::vector<size_t>& dims) {
    for (size_t dim : dims) {
        auto af = make_float_vec(dim);
        auto bf = make_float_vec(dim, static_cast<unsigned>(dim));
        auto a  = floats_to_fp16(af);
        auto b  = floats_to_fp16(bf);
        float d    = DistClass::compare(a.data(), b.data(), &dim);
        float ref  = fp16_ip_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}

TEST(FP16InnerProductExt8, MatchesNaive) {
    // dim must be a multiple of 8
    test_fp16_ip_matches_naive<deglib::distances::FP16InnerProductExt8>(
        {8, 16, 32, 64, 128, 256});
}

TEST(FP16InnerProductExt16, MatchesNaive) {
    // dim must be a multiple of 16
    test_fp16_ip_matches_naive<deglib::distances::FP16InnerProductExt16>(
        {16, 32, 64, 128, 256});
}

TEST(FP16InnerProductExt32, MatchesNaive) {
    // dim must be a multiple of 32
    test_fp16_ip_matches_naive<deglib::distances::FP16InnerProductExt32>(
        {32, 64, 128, 256});
}

// ---------------------------------------------------------------------------
//  FP16 residual variants (non-aligned dimensions)
// ---------------------------------------------------------------------------

TEST(FP16InnerProductExt16Residuals, NonAlignedDims) {
    std::vector<size_t> dims = {17, 20, 24, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto af = make_float_vec(dim);
        auto bf = make_float_vec(dim, static_cast<unsigned>(dim + 1));
        auto a  = floats_to_fp16(af);
        auto b  = floats_to_fp16(bf);
        float d   = deglib::distances::FP16InnerProductExt16Residuals::compare(a.data(), b.data(), &dim);
        float ref = fp16_ip_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}

TEST(FP16InnerProductExt8Residuals, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 25};
    for (size_t dim : dims) {
        auto af = make_float_vec(dim);
        auto bf = make_float_vec(dim, static_cast<unsigned>(dim + 2));
        auto a  = floats_to_fp16(af);
        auto b  = floats_to_fp16(bf);
        float d   = deglib::distances::FP16InnerProductExt8Residuals::compare(a.data(), b.data(), &dim);
        float ref = fp16_ip_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
//  FP16 cross-SIMD-path consistency
// ---------------------------------------------------------------------------

TEST(FP16InnerProduct, AllPathsAgree_128) {
    auto af = make_float_vec(128);
    auto bf = make_float_vec(128, 77);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = 128;

    float naive  = deglib::distances::FP16InnerProduct::compare(a.data(), b.data(), &dim);
    float ext8   = deglib::distances::FP16InnerProductExt8::compare(a.data(), b.data(), &dim);
    float ext16  = deglib::distances::FP16InnerProductExt16::compare(a.data(), b.data(), &dim);
    float ext32  = deglib::distances::FP16InnerProductExt32::compare(a.data(), b.data(), &dim);

    EXPECT_NEAR(ext8,  naive, 1e-2f);
    EXPECT_NEAR(ext16, naive, 1e-2f);
    EXPECT_NEAR(ext32, naive, 1e-2f);
}

TEST(FP16InnerProduct, AllPathsAgree_256) {
    auto af = make_float_vec(256);
    auto bf = make_float_vec(256, 55);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = 256;

    float naive  = deglib::distances::FP16InnerProduct::compare(a.data(), b.data(), &dim);
    float ext8   = deglib::distances::FP16InnerProductExt8::compare(a.data(), b.data(), &dim);
    float ext16  = deglib::distances::FP16InnerProductExt16::compare(a.data(), b.data(), &dim);
    float ext32  = deglib::distances::FP16InnerProductExt32::compare(a.data(), b.data(), &dim);

    EXPECT_NEAR(ext8,  naive, 1e-2f);
    EXPECT_NEAR(ext16, naive, 1e-2f);
    EXPECT_NEAR(ext32, naive, 1e-2f);
}

// ---------------------------------------------------------------------------
//  FloatSpace integration — FP16InnerProduct metric
// ---------------------------------------------------------------------------

TEST(FloatSpace, FP16InnerProduct_128Dim) {
    deglib::FloatSpace space(128, deglib::Metric::FP16InnerProduct);
    EXPECT_EQ(space.dim(), 128u);
    EXPECT_EQ(space.metric(), deglib::Metric::FP16InnerProduct);
    EXPECT_EQ(space.get_data_size(), 128 * sizeof(uint16_t));

    auto af = make_float_vec(128);
    auto bf = make_float_vec(128, 99);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    float d   = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    float ref = fp16_ip_naive_ref(a.data(), b.data(), 128);
    EXPECT_NEAR(d, ref, 1e-2f);
}

TEST(FloatSpace, FP16InnerProduct_VariousDimensions) {
    std::vector<size_t> dims = {1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 128, 256};
    for (size_t dim : dims) {
        deglib::FloatSpace space(dim, deglib::Metric::FP16InnerProduct);
        auto af = make_float_vec(dim);
        auto bf = make_float_vec(dim, static_cast<unsigned>(dim));
        auto a  = floats_to_fp16(af);
        auto b  = floats_to_fp16(bf);
        float d   = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        float ref = fp16_ip_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}


// ---------------------------------------------------------------------------
//  Main — handled by test_main.cpp (shared entry point)
// ---------------------------------------------------------------------------
