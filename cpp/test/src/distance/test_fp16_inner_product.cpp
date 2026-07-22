#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "test_helpers.h"

TEST(FP16InnerProduct, Fp16ToFloat_Zero) {
    EXPECT_FLOAT_EQ(deglib::distances::fp16_to_float(0x0000u), 0.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_One) {
    EXPECT_FLOAT_EQ(deglib::distances::fp16_to_float(0x3C00u), 1.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_NegOne) {
    EXPECT_FLOAT_EQ(deglib::distances::fp16_to_float(0xBC00u), -1.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_Two) {
    EXPECT_FLOAT_EQ(deglib::distances::fp16_to_float(0x4000u), 2.0f);
}

TEST(FP16InnerProduct, Fp16ToFloat_RoundTrip) {
    const std::vector<float> vals = {0.5f, -0.5f, 1.5f, -1.5f, 0.125f, 100.0f, -0.001f};
    for (float orig : vals) {
        auto fp16 = floats_to_fp16({orig});
        float recovered = deglib::distances::fp16_to_float(fp16[0]);
        EXPECT_NEAR(recovered, orig, std::abs(orig) * 1e-2f + 1e-4f)
            << "original=" << orig;
    }
}

TEST(FP16InnerProduct, IdentityZero) {
    std::vector<uint16_t> v(16, 0);
    size_t dim = v.size();
    float d = deglib::distances::FP16InnerProductDefault::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(FP16InnerProduct, KnownValue) {
    auto a = floats_to_fp16({1.0f, 0.0f, 0.0f, 0.0f});
    auto b = floats_to_fp16({1.0f, 0.0f, 0.0f, 0.0f});
    size_t dim = 4;
    float d = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 5e-3f);
}

TEST(FP16InnerProduct, Orthogonal) {
    auto a = floats_to_fp16({1.0f, 0.0f, 0.0f, 0.0f});
    auto b = floats_to_fp16({0.0f, 1.0f, 0.0f, 0.0f});
    size_t dim = 4;
    float d = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-3f);
}

TEST(FP16InnerProduct, Symmetry) {
    auto af = make_float_vec(64);
    auto bf = make_float_vec(64, 99);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = a.size();
    float ab = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
    float ba = deglib::distances::FP16InnerProductDefault::compare(b.data(), a.data(), &dim);
    EXPECT_NEAR(ab, ba, 1e-3f);
}

TEST(FP16InnerProduct, MatchesNaiveRef_32) {
    auto af = make_float_vec(32);
    auto bf = make_float_vec(32, 7);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = a.size();
    float d = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, fp16_ip_naive_ref(a.data(), b.data(), dim), 1e-3f);
}

TEST(FP16InnerProduct, MatchesNaiveRef_128) {
    auto af = make_float_vec(128);
    auto bf = make_float_vec(128, 13);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = a.size();
    float d = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, fp16_ip_naive_ref(a.data(), b.data(), dim), 1e-2f);
}

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
    test_fp16_ip_matches_naive<deglib::distances::FP16InnerProductExt8>(
        {8, 16, 32, 64, 128, 256});
}

TEST(FP16InnerProductExt16, MatchesNaive) {
    test_fp16_ip_matches_naive<deglib::distances::FP16InnerProductExt16>(
        {16, 32, 64, 128, 256});
}

TEST(FP16InnerProductExt32, MatchesNaive) {
    test_fp16_ip_matches_naive<deglib::distances::FP16InnerProductExt32>(
        {32, 64, 128, 256});
}

TEST(FP16InnerProductDefault, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto af = make_float_vec(dim);
        auto bf = make_float_vec(dim, static_cast<unsigned>(dim + 1));
        auto a  = floats_to_fp16(af);
        auto b  = floats_to_fp16(bf);
        float d   = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
        float ref = fp16_ip_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}

TEST(FP16InnerProduct, AllPathsAgree_128) {
    auto af = make_float_vec(128);
    auto bf = make_float_vec(128, 77);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    size_t dim = 128;

    float naive  = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
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

    float naive  = deglib::distances::FP16InnerProductDefault::compare(a.data(), b.data(), &dim);
    float ext8   = deglib::distances::FP16InnerProductExt8::compare(a.data(), b.data(), &dim);
    float ext16  = deglib::distances::FP16InnerProductExt16::compare(a.data(), b.data(), &dim);
    float ext32  = deglib::distances::FP16InnerProductExt32::compare(a.data(), b.data(), &dim);

    EXPECT_NEAR(ext8,  naive, 1e-2f);
    EXPECT_NEAR(ext16, naive, 1e-2f);
    EXPECT_NEAR(ext32, naive, 1e-2f);
}
