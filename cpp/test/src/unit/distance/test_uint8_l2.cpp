// test_uint8_l2.cpp — Unit tests for L2 distance on uint8_t vectors
//
// Tests scalar and SIMD L2 distance implementations for uint8_t vectors.
// HEAD API uses L2Uint8, L2Uint8_AVX512<Mode>, L2Uint8_AVX2<Mode>.

#include <vector>

#include "gtest/gtest.h"
#include "distance/uint8_l2.h"
#include "distances.h"

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
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim + 1);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8, ExtremeValues) {
    std::vector<size_t> dims = {16, 64, 128};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim, 0);
        auto b = make_uint8_vec(dim, 1);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8, LargeDimension) {
    size_t dim = 1000;
    auto a = make_uint8_vec(dim, 42);
    auto b = make_uint8_vec(dim, 123);
    float d = L2Uint8::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f);
}

} // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::uint8_l2::L2Uint8_AVX512;
using deglib::distances::uint8_l2::L2Uint8_AVX2;
using deglib::distances::ResidualMode;

TEST(L2Uint8_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::uint8_l2::select_dist(dim);
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = std::visit([&](auto&& dist) {
            return dist.compare(a.data(), b.data(), &dim);
        }, dist_variant);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

} // anonymous namespace

#endif // DEGLIB_X86

namespace {

// FloatSpace integration tests
TEST(L2Uint8_FloatSpace, L2Uint8Metric) {
    size_t dim = 64;
    deglib::FloatSpace space(dim, deglib::Metric::L2_Uint8);

    auto a = make_uint8_vec(dim, 42);
    auto b = make_uint8_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Uint8_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        deglib::FloatSpace space(dim, deglib::Metric::L2_Uint8);

        auto a = make_uint8_vec(dim, dim);
        auto b = make_uint8_vec(dim, dim + 1);

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

} // anonymous namespace
