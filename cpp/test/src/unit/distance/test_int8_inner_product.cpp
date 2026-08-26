// test_int8_inner_product.cpp — Unit tests for InnerProduct distance on int8_t vectors
//
// Tests scalar, AVX2, AVX2_VNNI, AVX512, and AVX512_VNNI inner product distance implementations for int8_t vectors.

#include "deglib/distance/int8_ip.h"
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

inline float ip_int8_naive(const int8_t* a, const int8_t* b, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
    }
    return -static_cast<float>(sum);
}

using deglib::distances::int8_ip::InnerProductInt8;

TEST(InnerProductInt8, IdentityZero) {
    std::vector<int8_t> v(16, 0);
    size_t dim = v.size();
    float d = InnerProductInt8::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 1e-4f);
}

TEST(InnerProductInt8, KnownValue) {
    int8_t a[] = {-1, 2, -3};
    int8_t b[] = {4, -5, 6};
    size_t dim = 3;
    float d = InnerProductInt8::compare(a, b, &dim);
    // (-1)*4 + 2*(-5) + (-3)*6 = -4 - 10 - 18 = -32 -> return -(-32.0f) = 32.0f
    EXPECT_NEAR(d, 32.0f, 1e-4f);
}

TEST(InnerProductInt8, MinMaxValues) {
    int8_t a[] = {-128, 127, -128, 127};
    int8_t b[] = {-128, 127, 127, -128};
    size_t dim = 4;
    float d = InnerProductInt8::compare(a, b, &dim);
    EXPECT_NEAR(d, -1.0f, 1e-4f);
}

TEST(InnerProductInt8, Symmetry) {
    auto a = make_int8_vec(64, -50);
    auto b = make_int8_vec(64, 99);
    size_t dim = a.size();
    float ab = InnerProductInt8::compare(a.data(), b.data(), &dim);
    float ba = InnerProductInt8::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(InnerProductInt8, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_int8_vec(dim, -100);
        auto b = make_int8_vec(dim, static_cast<int>(dim) + 5);
        float d = InnerProductInt8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(InnerProductInt8, NonAlignedDims) {
    std::vector<size_t> dims = {1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129};
    for (size_t dim : dims) {
        auto a = make_int8_vec(dim, 10);
        auto b = make_int8_vec(dim, -20);
        float d = InnerProductInt8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::ResidualMode;
using deglib::distances::int8_ip::InnerProductInt8_AVX2;
using deglib::distances::int8_ip::InnerProductInt8_AVX2_VNNI;
using deglib::distances::int8_ip::InnerProductInt8_AVX512;
using deglib::distances::int8_ip::InnerProductInt8_AVX512_VNNI;

TEST(InnerProductInt8_AVX512_VNNI, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512_vnni()) {
        GTEST_SKIP() << "AVX512_VNNI not supported";
    }
    // DualOnly: dim must be a multiple of 128
    for (size_t dim : {128, 256}) {
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, 75);
        float d = InnerProductInt8_AVX512_VNNI<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 128*k + 64 (e.g. 64, 192)
    for (size_t dim : {192, 320}) {
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, 75);
        float d = InnerProductInt8_AVX512_VNNI<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 128*k + rem (rem in 1..63)
    for (size_t dim : {129, 140, 191}) {
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, 75);
        float d = InnerProductInt8_AVX512_VNNI<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 64
    {
        size_t dim = 64;
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, 75);
        float d = InnerProductInt8_AVX512_VNNI<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 64 + rem (rem in 1..63)
    for (size_t dim : {65, 80, 127}) {
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, 75);
        float d = InnerProductInt8_AVX512_VNNI<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..63
    for (size_t dim : {1, 7, 32, 63}) {
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, 75);
        float d = InnerProductInt8_AVX512_VNNI<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: dim = 128*k + 64 + rem (rem in 1..63)
    for (size_t dim : {193, 220, 255}) {
        auto a = make_int8_vec(dim, -40);
        auto b = make_int8_vec(dim, 75);
        float d = InnerProductInt8_AVX512_VNNI<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(InnerProductInt8_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    // DualOnly: dim must be a multiple of 64
    for (size_t dim : {64, 128, 256}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX512<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 64*k + 32
    for (size_t dim : {96, 160, 224}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX512<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 64*k + rem (rem in 1..31)
    for (size_t dim : {65, 70, 80, 95}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX512<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 32
    {
        size_t dim = 32;
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX512<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 32 + rem (rem in 1..31)
    for (size_t dim : {33, 40, 63}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX512<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..31
    for (size_t dim : {1, 4, 15, 31}) {
        auto a = make_int8_vec(dim, -30);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX512<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: dim = 64*k + 32 + rem (rem in 1..31)
    for (size_t dim : {97, 105, 127, 161}) {
        auto a = make_int8_vec(dim, -55);
        auto b = make_int8_vec(dim, 88);
        float d = InnerProductInt8_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(InnerProductInt8_AVX2_VNNI, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx_vnni()) {
        GTEST_SKIP() << "AVX_VNNI not supported";
    }
    // DualOnly: dim must be a multiple of 64
    for (size_t dim : {64, 128, 256}) {
        auto a = make_int8_vec(dim, -80);
        auto b = make_int8_vec(dim, 45);
        float d = InnerProductInt8_AVX2_VNNI<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 64*k + 32
    for (size_t dim : {96, 160, 224}) {
        auto a = make_int8_vec(dim, -80);
        auto b = make_int8_vec(dim, 45);
        float d = InnerProductInt8_AVX2_VNNI<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 64*k + rem (rem in 1..31)
    for (size_t dim : {65, 70, 80, 95}) {
        auto a = make_int8_vec(dim, -80);
        auto b = make_int8_vec(dim, 45);
        float d = InnerProductInt8_AVX2_VNNI<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 32
    {
        size_t dim = 32;
        auto a = make_int8_vec(dim, -80);
        auto b = make_int8_vec(dim, 45);
        float d = InnerProductInt8_AVX2_VNNI<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 32 + rem (rem in 1..31)
    for (size_t dim : {33, 40, 63}) {
        auto a = make_int8_vec(dim, -80);
        auto b = make_int8_vec(dim, 45);
        float d = InnerProductInt8_AVX2_VNNI<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..31
    for (size_t dim : {1, 7, 16, 31}) {
        auto a = make_int8_vec(dim, -80);
        auto b = make_int8_vec(dim, 45);
        float d = InnerProductInt8_AVX2_VNNI<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: dim = 64*k + 32 + rem (rem in 1..31)
    for (size_t dim : {97, 105, 127, 161}) {
        auto a = make_int8_vec(dim, -80);
        auto b = make_int8_vec(dim, 45);
        float d = InnerProductInt8_AVX2_VNNI<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(InnerProductInt8_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    // DualOnly: dim must be a multiple of 32 (at least 32)
    for (size_t dim : {32, 64, 128, 256}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX2<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 32*k + 16 (e.g. 48, 80, 112)
    for (size_t dim : {48, 80, 112}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX2<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 32*k + rem (rem in 1..15, e.g. 33, 40, 47)
    for (size_t dim : {33, 40, 47}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX2<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 16
    {
        size_t dim = 16;
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX2<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 16 + rem (rem in 1..15, e.g. 17, 20, 31)
    for (size_t dim : {17, 20, 31}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX2<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..15
    for (size_t dim : {1, 4, 7, 15}) {
        auto a = make_int8_vec(dim, -20);
        auto b = make_int8_vec(dim, static_cast<int>(dim));
        float d = InnerProductInt8_AVX2<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: requires all 3 parts (dim = 32*k + 16 + rem, where rem in 1..15)
    for (size_t dim : {49, 55, 63, 81}) {
        auto a = make_int8_vec(dim, -70);
        auto b = make_int8_vec(dim, 33);
        float d = InnerProductInt8_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(InnerProductInt8_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 7, 16, 31, 32, 33, 63, 64, 65, 128, 256};
    for (size_t dim : dims) {
        auto dist_var = deglib::distances::int8_ip::select_dist(dim, deglib::cpu::InstructionSet::Auto);
        auto a = make_int8_vec(dim, -12);
        auto b = make_int8_vec(dim, 34);
        float d = std::visit([&](const auto& fn) { return fn.compare(a.data(), b.data(), &dim); }, dist_var);
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#endif  // DEGLIB_X86

namespace {

TEST(FloatSpace_Int8_InnerProduct, CompareMatchesNaive) {
    size_t dim = 64;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Int8_InnerProduct);

    auto a = make_int8_vec(dim, -42);
    auto b = make_int8_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(FloatSpace_Int8_InnerProduct, AllDimsMatchNaive) {
    std::vector<size_t> dims = {1, 3, 7, 16, 31, 32, 33, 63, 64, 65, 128, 129, 256};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Int8_InnerProduct);

        auto a = make_int8_vec(dim, static_cast<int>(dim));
        auto b = make_int8_vec(dim, static_cast<int>(dim) + 1);

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, ip_int8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(FloatSpace_Int8_InnerProduct, BatchMatchesSingle) {
    std::vector<size_t> test_dims = {8, 16, 32, 64, 128, 256};
    std::vector<size_t> batch_counts = {1, 4, 7, 8, 9, 16, 23, 24, 32};

    for (size_t dim : test_dims) {
        for (size_t count : batch_counts) {
            deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Int8_InnerProduct);
            auto dist_func = space.get_dist_func();
            auto batch_func = space.get_batch_dist_func();
            auto param = space.get_dist_func_param();

            auto query = make_int8_vec(dim, -99);
            std::vector<std::vector<int8_t>> db_vecs(count);
            std::vector<const void*> db_ptrs(count);

            for (size_t j = 0; j < count; ++j) {
                db_vecs[j] = make_int8_vec(dim, static_cast<int>(j * 100) - 50);
                db_ptrs[j] = db_vecs[j].data();
            }

            std::vector<float> single_dists(count);
            for (size_t j = 0; j < count; ++j) {
                single_dists[j] = dist_func(query.data(), db_ptrs[j], param);
            }

            std::vector<float> batch_dists(count, 0.0f);
            batch_func(query.data(), db_ptrs.data(), count, param, batch_dists.data());

            for (size_t j = 0; j < count; ++j) {
                EXPECT_FLOAT_EQ(batch_dists[j], single_dists[j]) << "dim=" << dim << " count=" << count << " idx=" << j;
            }
        }
    }
}

}  // anonymous namespace
