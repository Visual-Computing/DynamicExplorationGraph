#include <vector>
#include <array>
#include <memory>

#include "gtest/gtest.h"
#include "test_helpers.h"

TEST(InnerProductFloat, IdentityZero) {
    std::vector<float> v(16, 0.0f);
    size_t dim = v.size();
    float d = deglib::distances::InnerProductFloat::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 1.0f, 1e-4f);
}

TEST(InnerProductFloat, UnitVectorSelf) {
    std::vector<float> v(4, 1.0f);
    size_t dim = v.size();
    float d = deglib::distances::InnerProductFloat::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, -3.0f, 1e-4f);
}

TEST(InnerProductFloat, Orthogonal) {
    float a[] = {1.0f, 0, 0, 0};
    float b[] = {0, 0, 0, 1.0f};
    size_t dim = 4;
    float d = deglib::distances::InnerProductFloat::compare(a, b, &dim);
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

TEST(InnerProductFloat, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::InnerProductFloat::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(InnerProductFloat4Ext, MatchesNaive) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::InnerProductFloat4Ext::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-2f)
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

TEST(InnerProductFloat, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim + 1);
        float d = deglib::distances::InnerProductFloat::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// ---------------------------------------------------------------------------
// compare_batch tests
// ---------------------------------------------------------------------------

TEST(FP32InnerProduct_Batch, Batch4) {
    size_t dim = 64;
    auto a = make_float_vec(dim);
    auto b0 = make_float_vec(dim, 1);
    auto b1 = make_float_vec(dim, 2);
    auto b2 = make_float_vec(dim, 3);
    auto b3 = make_float_vec(dim, 4);
    const void* db[4] = {b0.data(), b1.data(), b2.data(), b3.data()};
    float dists[4];
    deglib::distances::InnerProductFloat::compare_batch(a.data(), db, 4, &dim, dists);
    for (int j = 0; j < 4; ++j) {
        auto b = make_float_vec(dim, j + 1);
        EXPECT_NEAR(dists[j], ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "j=" << j;
    }
}

TEST(FP32InnerProduct_Batch, Batch8) {
    size_t dim = 64;
    auto a = make_float_vec(dim);
    std::array<std::vector<float>, 8> bv;
    std::array<const void*, 8> db;
    for (int j = 0; j < 8; ++j) {
        bv[j] = make_float_vec(dim, j + 1);
        db[j] = bv[j].data();
    }
    float dists[8];
    deglib::distances::InnerProductFloat::compare_batch(a.data(), db.data(), 8, &dim, dists);
    for (int j = 0; j < 8; ++j) {
        auto b = make_float_vec(dim, j + 1);
        EXPECT_NEAR(dists[j], ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "j=" << j;
    }
}

TEST(FP32InnerProduct_Batch, Count13) {
    size_t dim = 64;
    auto a = make_float_vec(dim);
    std::vector<std::vector<float>> bv(13);
    std::vector<const void*> db(13);
    for (int j = 0; j < 13; ++j) {
        bv[j] = make_float_vec(dim, j + 1);
        db[j] = bv[j].data();
    }
    float dists[13];
    deglib::distances::InnerProductFloat::compare_batch(a.data(), db.data(), 13, &dim, dists);
    for (int j = 0; j < 13; ++j) {
        auto b = make_float_vec(dim, j + 1);
        EXPECT_NEAR(dists[j], ip_naive(a.data(), b.data(), dim), 1e-2f)
            << "j=" << j;
    }
}

TEST(FP32InnerProduct_Batch, CascadeDims) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        std::vector<std::vector<float>> bv(5);
        std::vector<const void*> db(5);
        for (int j = 0; j < 5; ++j) {
            bv[j] = make_float_vec(dim, j + 1);
            db[j] = bv[j].data();
        }
        float dists[5];
        deglib::distances::InnerProductFloat::compare_batch(a.data(), db.data(), 5, &dim, dists);
        for (int j = 0; j < 5; ++j) {
            auto b = make_float_vec(dim, j + 1);
            EXPECT_NEAR(dists[j], ip_naive(a.data(), b.data(), dim), 1e-2f)
                << "dim=" << dim << " j=" << j;
        }
    }
}
