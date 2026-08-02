// test_fp32_inner_product.cpp — Unit tests for InnerProduct distance computations
//
// Tests scalar and SIMD inner product distance implementations for various
// dimensions. HEAD API uses InnerProductFloat, InnerProductFloat_AVX512<Mode>,
// InnerProductFloat_AVX2<Mode> instead of InnerProductFloat4Ext/etc.

#include <vector>

#include "gtest/gtest.h"
#include "distance/fp32_ip.h"
#include "distances.h"

namespace {

inline std::vector<float> make_float_vec(size_t n, int seed = 0) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>((seed + static_cast<int>(i)) % 100);
    }
    return v;
}

inline float ip_naive(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

using deglib::distances::fp32_ip::InnerProductFloat;

TEST(InnerProductFloat, IdentityZero) {
    std::vector<float> v(16, 0.0f);
    size_t dim = v.size();
    float d = InnerProductFloat::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(InnerProductFloat, UnitVectorSelf) {
    std::vector<float> v(4, 1.0f);
    size_t dim = v.size();
    float d = InnerProductFloat::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, -3.0f, 1e-4f);
}

TEST(InnerProductFloat, Orthogonal) {
    float a[] = {1.0f, 0, 0, 0};
    float b[] = {0, 0, 0, 1.0f};
    size_t dim = 4;
    float d = InnerProductFloat::compare(a, b, &dim);
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(InnerProductFloat, Symmetry) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 99);
    size_t dim = a.size();
    float ab = InnerProductFloat::compare(a.data(), b.data(), &dim);
    float ba = InnerProductFloat::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(InnerProductFloat, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = InnerProductFloat::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(InnerProductFloat, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim + 1);
        float d = InnerProductFloat::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(InnerProductFloat, DotProduct) {
    size_t dim = 8;
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> b(dim, 1.0f);
    float dot = InnerProductFloat::dot(a.data(), b.data(), &dim);
    EXPECT_NEAR(dot, 36.0f, 1e-5f);
}

TEST(InnerProductFloat, ZeroDistance) {
    size_t dim = 8;
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    std::vector<float> a(dim, scale);
    std::vector<float> b = a;
    float d = InnerProductFloat::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 1e-5f);
}

TEST(InnerProductFloat, LargeDimension) {
    size_t dim = 1000;
    auto a = make_float_vec(dim, 42);
    auto b = make_float_vec(dim, 123);
    float d = InnerProductFloat::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f);
}

} // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::fp32_ip::InnerProductFloat_AVX512;
using deglib::distances::fp32_ip::InnerProductFloat_AVX2;
using deglib::distances::ResidualMode;

TEST(InnerProductFloat_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = InnerProductFloat_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(InnerProductFloat_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = InnerProductFloat_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(InnerProductFloat_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::fp32_ip::select_dist(dim);
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = std::visit([&](auto&& dist) {
            return dist.compare(a.data(), b.data(), &dim);
        }, dist_variant);
        EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

} // anonymous namespace

#endif // DEGLIB_X86

namespace {

// FloatSpace integration tests
TEST(InnerProductFloat_FloatSpace, InnerProductMetric) {
    size_t dim = 64;
    deglib::FloatSpace space(dim, deglib::Metric::InnerProduct);

    auto a = make_float_vec(dim, 42);
    auto b = make_float_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f);
}

TEST(InnerProductFloat_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256, 512};
    for (size_t dim : dims) {
        deglib::FloatSpace space(dim, deglib::Metric::InnerProduct);

        auto a = make_float_vec(dim, dim);
        auto b = make_float_vec(dim, dim + 1);

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, 1.0f - ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

} // anonymous namespace
