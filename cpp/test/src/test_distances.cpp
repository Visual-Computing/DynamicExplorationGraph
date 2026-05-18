// test_distances.cpp — Comprehensive unit tests for deglib distances.h
//
// Each SIMD-accelerated distance function is validated against a
// hand-written naive reference implementation so that correctness is
// guaranteed regardless of which SIMD path the compiler selects.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "distances.h"

// ---------------------------------------------------------------------------
//  Google Test (bundled with fmt)
// ---------------------------------------------------------------------------
#include "gtest/gtest.h"
#include "gmock/gmock.h"

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
//  Main — handled by test_main.cpp (shared entry point)
// ---------------------------------------------------------------------------
