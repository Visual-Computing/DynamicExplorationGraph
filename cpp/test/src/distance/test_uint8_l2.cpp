#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "test_helpers.h"

static void test_l2_uint8_matches_naive() {
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = deglib::distances::Uint8L2Default::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1.0f)
            << "dim=" << dim;
    }
}

TEST(Uint8L2Default, IdentityZero) {
    std::vector<uint8_t> v(128, 0);
    size_t dim = v.size();
    float d = deglib::distances::Uint8L2Default::compare(v.data(), v.data(), &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(Uint8L2Default, KnownValue) {
    uint8_t a[] = {3, 4};
    uint8_t b[] = {0, 0};
    size_t dim = 2;
    float d = deglib::distances::Uint8L2Default::compare(a, b, &dim);
    EXPECT_NEAR(d, 25.0f, 1e-4f);
}

TEST(Uint8L2Default, MaxDiff) {
    std::vector<uint8_t> a(128, 255);
    std::vector<uint8_t> b(128, 0);
    size_t dim = a.size();
    float d = deglib::distances::Uint8L2Default::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, 128.0f * 255.0f * 255.0f, 1.0f);
}

TEST(Uint8L2Default, Symmetry) {
    auto a = make_uint8_vec(128);
    auto b = make_uint8_vec(128, 99);
    size_t dim = a.size();
    float ab = deglib::distances::Uint8L2Default::compare(a.data(), b.data(), &dim);
    float ba = deglib::distances::Uint8L2Default::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(Uint8L2Default, MatchesNaive_16) {
    auto a = make_uint8_vec(16);
    auto b = make_uint8_vec(16, 7);
    size_t dim = a.size();
    float d = deglib::distances::Uint8L2Default::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-2f);
}

TEST(Uint8L2Default, MatchesNaive_128) {
    auto a = make_uint8_vec(128);
    auto b = make_uint8_vec(128, 13);
    size_t dim = a.size();
    float d = deglib::distances::Uint8L2Default::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-2f);
}

TEST(Uint8L2Default, Dim1) {
    uint8_t a[] = {10};
    uint8_t b[] = {20};
    size_t dim = 1;
    float d = deglib::distances::Uint8L2Default::compare(a, b, &dim);
    EXPECT_NEAR(d, 100.0f, 1e-4f);
}

TEST(Uint8L2Default, Dim0) {
    uint8_t a = 1;
    uint8_t b = 2;
    size_t dim = 0;
    float d = deglib::distances::Uint8L2Default::compare(&a, &b, &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(L2Uint8Ext16, MatchesNaive) {  // backward-compatible alias
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = deglib::distances::L2Uint8Ext16::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1.0f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8Ext32, MatchesNaive) {  // backward-compatible alias
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = deglib::distances::L2Uint8Ext32::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1.0f)
            << "dim=" << dim;
    }
}

TEST(Uint8L2, BatchMatchesNaive) {
    for (size_t dim : std::vector<size_t>{16, 32, 64, 128}) {
        auto query = make_uint8_vec(dim, 42);
        std::vector<std::vector<uint8_t>> db_vecs(12);
        std::vector<const void*> db_ptrs(12);
        for (size_t j = 0; j < 12; ++j) {
            db_vecs[j] = make_uint8_vec(dim, 100 + j);
            db_ptrs[j] = db_vecs[j].data();
        }

        std::vector<float> expected(12);
        for (size_t j = 0; j < 12; ++j) {
            expected[j] = deglib::distances::Uint8L2Default::compare(query.data(), db_ptrs[j], &dim);
        }

        std::vector<float> dists(12, 0.0f);
        deglib::distances::Uint8L2Default::compare_batch(query.data(), db_ptrs.data(), 12, &dim, dists.data());
        for (size_t j = 0; j < 12; ++j) {
            EXPECT_NEAR(dists[j], expected[j], 1.0f) << "dim=" << dim << " j=" << j;
        }
    }
}
