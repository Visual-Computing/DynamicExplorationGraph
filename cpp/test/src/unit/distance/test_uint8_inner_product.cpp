// test_uint8_inner_product.cpp — Unit tests for InnerProduct distance on uint8_t vectors
//
// Tests scalar, AVX2, AVX512, and VNNI inner product distance implementations for uint8_t vectors.

#include "deglib/distance/uint8_ip.h"
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

inline float ip_uint8_naive(const uint8_t* a, const uint8_t* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += static_cast<float>(a[i]) * static_cast<float>(b[i]);
    }
    return -sum;
}

using deglib::distances::uint8_ip::InnerProductUint8;

TEST(InnerProductUint8, IdentityZero) {
    std::vector<uint8_t> v(16, 0);
    size_t dim = v.size();
    float d = InnerProductUint8::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 1e-4f);
}

TEST(InnerProductUint8, KnownValue) {
    uint8_t a[] = {1, 2, 3};
    uint8_t b[] = {4, 5, 6};
    size_t dim = 3;
    float d = InnerProductUint8::compare(a, b, &dim);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32 -> return -32.0f
    EXPECT_NEAR(d, -32.0f, 1e-4f);
}

TEST(InnerProductUint8, Symmetry) {
    auto a = make_uint8_vec(64);
    auto b = make_uint8_vec(64, 99);
    size_t dim = a.size();
    float ab = InnerProductUint8::compare(a.data(), b.data(), &dim);
    float ba = InnerProductUint8::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(InnerProductUint8, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(InnerProductUint8, NonAlignedDims) {
    std::vector<size_t> dims = {1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim, 10);
        auto b = make_uint8_vec(dim, 20);
        float d = InnerProductUint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::ResidualMode;
using deglib::distances::uint8_ip::InnerProductUint8_AVX2;
using deglib::distances::uint8_ip::InnerProductUint8_AVX512;

TEST(InnerProductUint8_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    // DualOnly: dim must be a multiple of 64 (at least 64)
    for (size_t dim : {64, 128, 256}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX512<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 64*k + 32 (e.g. 96, 160, 224)
    for (size_t dim : {96, 160, 224}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX512<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 64*k + rem (rem in 1..31, e.g. 65, 70, 95)
    for (size_t dim : {65, 70, 80, 95}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX512<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 32
    {
        size_t dim = 32;
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX512<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 32 + rem (rem in 1..31, e.g. 33, 40, 63)
    for (size_t dim : {33, 40, 63}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX512<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..31
    for (size_t dim : {1, 4, 15, 31}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX512<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: requires all 3 parts (dim = 64*k + 32 + rem, where rem in 1..31)
    for (size_t dim : {97, 105, 127, 161}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(InnerProductUint8_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    // DualOnly: dim must be a multiple of 32 (at least 32)
    for (size_t dim : {32, 64, 128, 256}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX2<ResidualMode::DualOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualOnly dim=" << dim;
    }
    // DualPlusSimd: dim = 32*k + 16 (e.g. 48, 80, 112)
    for (size_t dim : {48, 80, 112, 144}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX2<ResidualMode::DualPlusSimd>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualPlusSimd dim=" << dim;
    }
    // DualTail: dim = 32*k + rem (rem in 1..15, e.g. 33, 40, 47)
    for (size_t dim : {33, 40, 47, 65, 70}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX2<ResidualMode::DualTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "DualTail dim=" << dim;
    }
    // SimdOnly: dim = 16
    {
        size_t dim = 16;
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX2<ResidualMode::SimdOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdOnly dim=" << dim;
    }
    // SimdTail: dim = 16 + rem (rem in 1..15, e.g. 17, 24, 31)
    for (size_t dim : {17, 24, 31}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX2<ResidualMode::SimdTail>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "SimdTail dim=" << dim;
    }
    // TailOnly: dim in 1..15
    for (size_t dim : {1, 4, 7, 15}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX2<ResidualMode::TailOnly>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "TailOnly dim=" << dim;
    }
    // Full: requires all 3 parts (dim = 32*k + 16 + rem, where rem in 1..15)
    for (size_t dim : {49, 55, 63, 81, 115}) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = InnerProductUint8_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "Full dim=" << dim;
    }
}

TEST(InnerProductUint8_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::uint8_ip::select_dist(dim);
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = std::visit([&](auto&& dist) { return dist.compare(a.data(), b.data(), &dim); }, dist_variant);
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

}  // anonymous namespace

#endif  // DEGLIB_X86

namespace {

TEST(InnerProductUint8_FloatSpace, Uint8Metric) {
    size_t dim = 64;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_InnerProduct);

    auto a = make_uint8_vec(dim, 42);
    auto b = make_uint8_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(InnerProductUint8_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_InnerProduct);

        auto a = make_uint8_vec(dim, dim);
        auto b = make_uint8_vec(dim, dim + 1);

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, ip_uint8_naive(a.data(), b.data(), dim), 1e-4f) << "dim=" << dim;
    }
}

TEST(InnerProductUint8_Batch, MatchesSingleCompare) {
    std::vector<size_t> dims = {16, 32, 64, 128, 256, 768};
    std::vector<size_t> counts = {1, 3, 4, 7, 8, 9, 15, 16, 25};

    for (size_t dim : dims) {
        for (size_t count : counts) {
            auto q = make_uint8_vec(dim, 77);
            std::vector<std::vector<uint8_t>> db(count);
            std::vector<const void*> db_ptrs(count);
            for (size_t i = 0; i < count; ++i) {
                db[i] = make_uint8_vec(dim, static_cast<int>(i * 13 + 1));
                db_ptrs[i] = db[i].data();
            }

            deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_InnerProduct);
            auto dist_func = space.get_dist_func();
            auto batch_func = space.get_batch_dist_func();
            auto param = space.get_dist_func_param();

            std::vector<float> single_dists(count);
            for (size_t i = 0; i < count; ++i) {
                single_dists[i] = dist_func(q.data(), db_ptrs[i], param);
            }

            std::vector<float> batch_dists(count, 0.0f);
            batch_func(q.data(), db_ptrs.data(), count, param, batch_dists.data());

            for (size_t i = 0; i < count; ++i) {
                EXPECT_FLOAT_EQ(batch_dists[i], single_dists[i]) << "dim=" << dim << " count=" << count << " index=" << i;
            }
        }
    }
}

TEST(InnerProductUint8_SelectDist, AcceptsVNNIFallback) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512_vnni()) {
        EXPECT_NO_THROW(deglib::distances::uint8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX512_VNNI));
    }
    if (deglib::cpu::has_avx_vnni()) {
        EXPECT_NO_THROW(deglib::distances::uint8_ip::select_dist(128, deglib::cpu::InstructionSet::AVX2_VNNI));
    }
#endif
}

}  // anonymous namespace
