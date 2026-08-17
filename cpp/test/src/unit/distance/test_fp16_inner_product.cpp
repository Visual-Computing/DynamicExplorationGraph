// test_fp16_inner_product.cpp — Unit tests for FP16 inner product distance computations
//
// Tests scalar and SIMD inner product distance implementations for various
// dimensions. Uses InnerProductFP16, InnerProductFP16_AVX512<Mode>,
// InnerProductFP16_AVX2<Mode> and select_dist() for dispatch.
// FP16 vectors are stored as uint16_t arrays (IEEE 754 half-precision bit patterns).
// The distance is computed as 1.f - dot_product, where dot_product is the raw
// inner product of the float-converted vectors.

#include "deglib/distance/fp16_ip.h"
#include "deglib/distances.h"
#include "gtest/gtest.h"

#include <vector>

namespace {

using deglib::distances::fp16::float_to_fp16;
using deglib::distances::fp16_ip::InnerProductFP16;

// Convert a float vector to a uint16_t (FP16) vector for inner product testing.
inline std::vector<uint16_t> make_fp16_vec(const std::vector<float>& floats) {
    std::vector<uint16_t> fp16(floats.size());
    for (size_t i = 0; i < floats.size(); ++i) {
        fp16[i] = float_to_fp16(floats[i]);
    }
    return fp16;
}

// Generate a float vector with deterministic values based on seed and dimension.
inline std::vector<float> make_float_vec(size_t n, int seed = 0) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>((seed + static_cast<int>(i)) % 100);
    }
    return v;
}

// Naive scalar inner product distance: 1 - sum(a*b)
inline float ip_naive(const uint16_t* a, const uint16_t* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float fa = deglib::distances::fp16::fp16_to_float(a[i]);
        float fb = deglib::distances::fp16::fp16_to_float(b[i]);
        sum += fa * fb;
    }
    return 1.0f - sum;
}

}  // anonymous namespace

// ============================================================================
// Scalar correctness tests
// ============================================================================

TEST(InnerProductFP16, IdentityZero) {
    std::vector<float> v(16, 0.0f);
    auto fp16 = make_fp16_vec(v);
    size_t dim = fp16.size();
    float d = InnerProductFP16::compare(fp16.data(), fp16.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(InnerProductFP16, UnitVectorSelf) {
    std::vector<float> v(4, 1.0f);
    auto fp16 = make_fp16_vec(v);
    size_t dim = fp16.size();
    float d = InnerProductFP16::compare(fp16.data(), fp16.data(), &dim);
    // dot = 4, distance = 1 - 4 = -3
    EXPECT_NEAR(d, -3.0f, 1e-4f);
}

TEST(InnerProductFP16, Orthogonal) {
    std::vector<float> a = {1.0f, 0, 0, 0};
    std::vector<float> b = {0, 0, 0, 1.0f};
    auto fa = make_fp16_vec(a);
    auto fb = make_fp16_vec(b);
    size_t dim = 4;
    float d = InnerProductFP16::compare(fa.data(), fb.data(), &dim);
    // dot = 0, distance = 1 - 0 = 1
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(InnerProductFP16, Symmetry) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 99);
    auto fa = make_fp16_vec(a);
    auto fb = make_fp16_vec(b);
    size_t dim = fa.size();
    float ab = InnerProductFP16::compare(fa.data(), fb.data(), &dim);
    float ba = InnerProductFP16::compare(fb.data(), fa.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(InnerProductFP16, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, static_cast<int>(dim));
        auto fa = make_fp16_vec(a);
        auto fb = make_fp16_vec(b);
        float d = InnerProductFP16::compare(fa.data(), fb.data(), &dim);
        EXPECT_NEAR(d, ip_naive(fa.data(), fb.data(), dim), 1e-2f) << "dim=" << dim;
    }
}

TEST(InnerProductFP16, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, static_cast<int>(dim + 1));
        auto fa = make_fp16_vec(a);
        auto fb = make_fp16_vec(b);
        float d = InnerProductFP16::compare(fa.data(), fb.data(), &dim);
        EXPECT_NEAR(d, ip_naive(fa.data(), fb.data(), dim), 1e-2f) << "dim=" << dim;
    }
}

TEST(InnerProductFP16, LargeDimension) {
    size_t dim = 1000;
    auto a = make_float_vec(dim, 42);
    auto b = make_float_vec(dim, 123);
    auto fa = make_fp16_vec(a);
    auto fb = make_fp16_vec(b);
    float d = InnerProductFP16::compare(fa.data(), fb.data(), &dim);
    EXPECT_NEAR(d, ip_naive(fa.data(), fb.data(), dim), 1e-2f);
}

// ============================================================================
// SIMD correctness tests (AVX2 & AVX-512)
// ============================================================================

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::ResidualMode;
using deglib::distances::fp16_ip::InnerProductFP16_AVX2;
using deglib::distances::fp16_ip::InnerProductFP16_AVX512;

TEST(InnerProductFP16_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX-512 not supported";
    }
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, static_cast<int>(dim));
        auto fa = make_fp16_vec(a);
        auto fb = make_fp16_vec(b);
        float d = InnerProductFP16_AVX512<ResidualMode::Full>::compare(fa.data(), fb.data(), &dim);
        EXPECT_NEAR(d, ip_naive(fa.data(), fb.data(), dim), 1e-2f) << "dim=" << dim;
    }
}

TEST(InnerProductFP16_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, static_cast<int>(dim));
        auto fa = make_fp16_vec(a);
        auto fb = make_fp16_vec(b);
        float d = InnerProductFP16_AVX2<ResidualMode::Full>::compare(fa.data(), fb.data(), &dim);
        EXPECT_NEAR(d, ip_naive(fa.data(), fb.data(), dim), 1e-2f) << "dim=" << dim;
    }
}

TEST(InnerProductFP16_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::fp16_ip::select_dist(dim);
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, static_cast<int>(dim));
        auto fa = make_fp16_vec(a);
        auto fb = make_fp16_vec(b);
        float d = std::visit([&](auto&& dist) { return dist.compare(fa.data(), fb.data(), &dim); }, dist_variant);
        EXPECT_NEAR(d, ip_naive(fa.data(), fb.data(), dim), 1e-2f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#endif  // DEGLIB_X86

// ============================================================================
// FloatSpace integration tests
// ============================================================================

namespace {

TEST(InnerProductFP16_FloatSpace, FP16InnerProductMetric) {
    size_t dim = 64;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP16_InnerProduct);

    EXPECT_EQ(space.dim(), dim);
    EXPECT_EQ(space.metric(), deglib::distances::Metric::FP16_InnerProduct);
    EXPECT_EQ(space.get_data_size(), dim * sizeof(uint16_t));
}

TEST(InnerProductFP16_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256, 512};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP16_InnerProduct);

        auto a = make_float_vec(dim, static_cast<int>(dim));
        auto b = make_float_vec(dim, static_cast<int>(dim + 1));
        auto fa = make_fp16_vec(a);
        auto fb = make_fp16_vec(b);

        float d = space.get_dist_func()(fa.data(), fb.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, ip_naive(fa.data(), fb.data(), dim), 1e-2f) << "dim=" << dim;
    }
}

TEST(InnerProductFP16_FloatSpace, SelectDistMatchesScalar) {
    std::vector<size_t> dims = {1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP16_InnerProduct);

        auto a = make_float_vec(dim, 42);
        auto b = make_float_vec(dim, 123);
        auto fa = make_fp16_vec(a);
        auto fb = make_fp16_vec(b);

        float d = space.get_dist_func()(fa.data(), fb.data(), space.get_dist_func_param());
        float expected = ip_naive(fa.data(), fb.data(), dim);
        EXPECT_NEAR(d, expected, 1e-2f) << "dim=" << dim;
    }
}

TEST(InnerProductFP16_Batch, MatchesSingleCompare) {
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256, 768};
    std::vector<size_t> counts = {1, 3, 4, 7, 8, 9, 15, 16, 25};

    for (size_t dim : dims) {
        for (size_t count : counts) {
            auto q_floats = make_float_vec(dim, 77);
            auto q_fp16 = make_fp16_vec(q_floats);

            std::vector<std::vector<uint16_t>> db_fp16(count);
            std::vector<const void*> db_ptrs(count);
            for (size_t i = 0; i < count; ++i) {
                auto db_floats = make_float_vec(dim, static_cast<int>(i * 10 + 1));
                db_fp16[i] = make_fp16_vec(db_floats);
                db_ptrs[i] = db_fp16[i].data();
            }

            std::vector<float> batch_dists(count, 0.0f);
            auto dist_variant = deglib::distances::fp16_ip::select_dist(dim);

            std::visit(
                [&](auto&& dist) {
                    using DistType = std::decay_t<decltype(dist)>;
                    DistType::compare_batch(q_fp16.data(), db_ptrs.data(), count, &dim, batch_dists.data());
                },
                dist_variant
            );

            for (size_t i = 0; i < count; ++i) {
                float single_dist = std::visit(
                    [&](auto&& dist) {
                        using DistType = std::decay_t<decltype(dist)>;
                        return DistType::compare(q_fp16.data(), db_ptrs[i], &dim);
                    },
                    dist_variant
                );

                EXPECT_NEAR(batch_dists[i], single_dist, 1e-4f) << "dim=" << dim << ", count=" << count << ", index=" << i;
            }
        }
    }
}

}  // anonymous namespace
