// test_int8_l2.cpp — Unit tests for L2 distance on int8_t vectors
//
// Tests scalar and SIMD L2 distance implementations for int8_t vectors.

#include "deglib/distance/int8_l2.h"
#include "deglib/distances.h"
#include "gtest/gtest.h"

#include <vector>

namespace {

inline std::vector<int8_t> make_int8_vec(size_t n, int seed = 0) {
    std::vector<int8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        int val = (seed + static_cast<int>(i) * 17) % 256;
        if (val > 127) val -= 256;
        v[i] = static_cast<int8_t>(val);
    }
    return v;
}

inline float l2_int8_naive(const int8_t* a, const int8_t* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += d * d;
    }
    return sum;
}

using deglib::distances::int8_l2::L2Int8;

TEST(L2Int8, IdentityZero) {
    std::vector<int8_t> v(16, 42);
    size_t dim = v.size();
    float d = L2Int8::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 1e-4f);
}

TEST(L2Int8, KnownValue) {
    int8_t a[] = {-50, 0, 100};
    int8_t b[] = {50, -50, -28};
    size_t dim = 3;
    float d = L2Int8::compare(a, b, &dim);
    // (-50 - 50)^2 + (0 - (-50))^2 + (100 - (-28))^2 = 100^2 + 50^2 + 128^2 = 10000 + 2500 + 16384 = 28884
    EXPECT_NEAR(d, 28884.0f, 1e-4f);
}

TEST(L2Int8, MinMaxValues) {
    int8_t a[] = {-128, 127, -128, 127};
    int8_t b[] = {127, -128, -128, 127};
    size_t dim = 4;
    float d = L2Int8::compare(a, b, &dim);
    // (-128 - 127)^2 + (127 - (-128))^2 + 0 + 0 = 255^2 + 255^2 = 65025 + 65025 = 130050
    EXPECT_NEAR(d, 130050.0f, 1e-4f);
}

TEST(L2Int8, Symmetry) {
    auto a = make_int8_vec(64, -50);
    auto b = make_int8_vec(64, 99);
    size_t dim = a.size();
    float ab = L2Int8::compare(a.data(), b.data(), &dim);
    float ba = L2Int8::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(L2Int8, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_int8_vec(dim, -100);
        auto b = make_int8_vec(dim, static_cast<int>(dim) + 5);
        float d = L2Int8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(L2Int8, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_int8_vec(dim, 10);
        auto b = make_int8_vec(dim, -20);
        float d = L2Int8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(L2Int8, LargeDimension) {
    size_t dim = 1000;
    auto a = make_int8_vec(dim, -42);
    auto b = make_int8_vec(dim, 123);
    float d = L2Int8::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f);
}

}  // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::ResidualMode;
using deglib::distances::int8_l2::L2Int8_AVX2;
using deglib::distances::int8_l2::L2Int8_AVX512;

TEST(L2Int8_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    // DualOnly: dim must be a multiple of 64 (at least 64)
    for (size_t dim : {64, 128, 256}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX512<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 64*k + 32 (e.g. 96, 160, 224)
    for (size_t dim : {96, 160, 224}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX512<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 64*k + rem (rem in 1..31, e.g. 65, 70, 95)
    for (size_t dim : {65, 70, 80, 95}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX512<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 32
    {
        size_t dim = 32;
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX512<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 32 + rem (rem in 1..31, e.g. 33, 40, 63)
    for (size_t dim : {33, 40, 63}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX512<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..31
    for (size_t dim : {1, 4, 15, 31}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX512<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: requires all 3 parts (dim = 64*k + 32 + rem, where rem in 1..31)
    for (size_t dim : {97, 105, 127, 161}) {
        auto a = make_int8_vec(dim, -55);
        auto b = make_int8_vec(dim, 88);
        float d = L2Int8_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(L2Int8_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    // DualOnly: dim must be a multiple of 32 (at least 32)
    for (size_t dim : {32, 64, 128, 256}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX2<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 32*k + 16 (e.g. 48, 80, 112)
    for (size_t dim : {48, 80, 112, 144}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX2<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 32*k + rem (rem in 1..15, e.g. 33, 40, 47)
    for (size_t dim : {33, 40, 47, 65, 70}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX2<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 16
    {
        size_t dim = 16;
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX2<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 16 + rem (rem in 1..15, e.g. 17, 24, 31)
    for (size_t dim : {17, 24, 31}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX2<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..15
    for (size_t dim : {1, 4, 7, 15}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX2<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: requires all 3 parts (dim = 32*k + 16 + rem, where rem in 1..15)
    for (size_t dim : {49, 55, 63, 81, 115}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = L2Int8_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(L2Int8_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::int8_l2::select_dist(dim);
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = std::visit([&](auto&& dist) { return dist.compare(a.data(), b.data(), &dim); }, dist_variant);
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#endif  // DEGLIB_X86

namespace {

// FloatSpace integration tests
TEST(L2Int8_FloatSpace, L2Int8Metric) {
    size_t dim = 64;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Int8_L2);

    auto a = make_int8_vec(dim, -42);
    auto b = make_int8_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Int8_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 48, 64, 80, 100, 128, 256, 784};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Int8_L2);

        auto a = make_int8_vec(dim, static_cast<int>(dim));
        auto b = make_int8_vec(dim, static_cast<int>(dim + 1));

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, l2_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(L2Int8_Batch, MatchesSingleCompare) {
    std::vector<size_t> dims = {16, 32, 48, 64, 80, 100, 128, 256, 768, 784};
    std::vector<size_t> counts = {1, 3, 4, 7, 8, 9, 15, 16, 25};

    for (size_t dim : dims) {
        for (size_t count : counts) {
            auto q = make_int8_vec(dim, -77);

            std::vector<std::vector<int8_t>> db(count);
            std::vector<const void*> db_ptrs(count);
            for (size_t i = 0; i < count; ++i) {
                db[i] = make_int8_vec(dim, static_cast<int>(i * 10 - 50));
                db_ptrs[i] = db[i].data();
            }

            std::vector<float> batch_dists(count, 0.0f);
            auto dist_variant = deglib::distances::int8_l2::select_dist(dim);

            std::visit(
                [&](auto&& dist) {
                    using DistType = std::decay_t<decltype(dist)>;
                    DistType::compare_batch(q.data(), db_ptrs.data(), count, &dim, batch_dists.data());
                },
                dist_variant
            );

            for (size_t i = 0; i < count; ++i) {
                float single_dist = std::visit(
                    [&](auto&& dist) {
                        using DistType = std::decay_t<decltype(dist)>;
                        return DistType::compare(q.data(), db_ptrs[i], &dim);
                    },
                    dist_variant
                );

                EXPECT_NEAR(batch_dists[i], single_dist, 1e-4f) << "dim=" << dim << ", count=" << count << ", index=" << i;
            }
        }
    }
}

}  // anonymous namespace
