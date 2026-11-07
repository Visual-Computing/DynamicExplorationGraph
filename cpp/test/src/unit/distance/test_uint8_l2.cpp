// test_uint8_l2.cpp — Unit tests for L2 distance on uint8_t vectors
//
// Tests scalar and SIMD L2 distance implementations for uint8_t vectors.
// HEAD API uses L2Uint8, L2Uint8_AVX512<Mode>, L2Uint8_AVX2<Mode>.

#include "deglib/distance/uint8_l2.h"
#include "deglib/distances.h"
#include "gtest/gtest.h"

#include <chrono>
#include <vector>

namespace {

inline std::vector<uint8_t> make_uint8_vec(size_t n, int seed = 0) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<uint8_t>((seed + static_cast<int>(i)) % 256);
    }
    return v;
}

inline float l2_uint8_naive(const uint8_t* a, const uint8_t* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += d * d;
    }
    return sum;
}

using deglib::distances::uint8_l2::L2Uint8;

TEST(L2Uint8, IdentityZero) {
    std::vector<uint8_t> v(16, 42);
    size_t dim = v.size();
    float d = L2Uint8::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 1e-4f);
}

TEST(L2Uint8, KnownValue) {
    uint8_t a[] = {0, 100, 200};
    uint8_t b[] = {50, 150, 250};
    size_t dim = 3;
    float d = L2Uint8::compare(a, b, &dim);
    EXPECT_NEAR(d, 7500.0f, 1e-4f);
}

TEST(L2Uint8, Symmetry) {
    auto a = make_uint8_vec(64);
    auto b = make_uint8_vec(64, 99);
    size_t dim = a.size();
    float ab = L2Uint8::compare(a.data(), b.data(), &dim);
    float ba = L2Uint8::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(L2Uint8, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(L2Uint8, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim + 1);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(L2Uint8, ExtremeValues) {
    std::vector<size_t> dims = {16, 64, 128};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim, 0);
        auto b = make_uint8_vec(dim, 1);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(L2Uint8, LargeDimension) {
    size_t dim = 1000;
    auto a = make_uint8_vec(dim, 42);
    auto b = make_uint8_vec(dim, 123);
    float d = L2Uint8::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f);
}

}  // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::ResidualMode;
using deglib::distances::uint8_l2::L2Uint8_AVX2;
using deglib::distances::uint8_l2::L2Uint8_AVX512;

TEST(L2Uint8_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    // DualOnly: dim must be a multiple of 64 (at least 64)
    for (size_t dim : {64, 128, 256}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 64*k + 32 (e.g. 96, 160, 224)
    for (size_t dim : {96, 160, 224}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 64*k + rem (rem in 1..31, e.g. 65, 70, 95)
    for (size_t dim : {65, 70, 80, 95}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 32
    {
        size_t dim = 32;
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 32 + rem (rem in 1..31, e.g. 33, 40, 63)
    for (size_t dim : {33, 40, 63}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..31
    for (size_t dim : {1, 4, 15, 31}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: requires all 3 parts (dim = 64*k + 32 + rem, where rem in 1..31)
    for (size_t dim : {97, 105, 127, 161}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(L2Uint8_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    // DualOnly: dim must be a multiple of 32 (at least 32)
    for (size_t dim : {32, 64, 128, 256}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 32*k + 16 (e.g. 48, 80, 112)
    for (size_t dim : {48, 80, 112, 144}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 32*k + rem (rem in 1..15, e.g. 33, 40, 47)
    for (size_t dim : {33, 40, 47, 65, 70}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 16
    {
        size_t dim = 16;
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 16 + rem (rem in 1..15, e.g. 17, 24, 31)
    for (size_t dim : {17, 24, 31}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..15
    for (size_t dim : {1, 4, 7, 15}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: requires all 3 parts (dim = 32*k + 16 + rem, where rem in 1..15)
    for (size_t dim : {49, 55, 63, 81, 115}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(L2Uint8_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::uint8_l2::select_dist(dim);
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = std::visit([&](auto&& dist) { return dist.compare(a.data(), b.data(), &dim); }, dist_variant);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#endif  // DEGLIB_X86

namespace {

// FloatSpace integration tests
TEST(L2Uint8_FloatSpace, L2Uint8Metric) {
    size_t dim = 64;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_L2);

    auto a = make_uint8_vec(dim, 42);
    auto b = make_uint8_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Uint8_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 48, 64, 80, 100, 128, 256, 784};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_L2);

        auto a = make_uint8_vec(dim, dim);
        auto b = make_uint8_vec(dim, dim + 1);

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

TEST(L2Uint8_SelectDist, AcceptsVNNIFallback) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512_vnni()) {
        EXPECT_NO_THROW(deglib::distances::uint8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
    }
    if (deglib::cpu::has_avx_vnni()) {
        EXPECT_NO_THROW(deglib::distances::uint8_l2::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
    }
#endif
}
