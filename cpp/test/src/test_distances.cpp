#include <vector>

#include "gtest/gtest.h"
#include "distance/test_helpers.h"

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

TEST(FloatSpace, FP16InnerProduct_128Dim) {
    deglib::FloatSpace space(128, deglib::Metric::FP16InnerProduct);
    EXPECT_EQ(space.dim(), 128u);
    EXPECT_EQ(space.metric(), deglib::Metric::FP16InnerProduct);
    EXPECT_EQ(space.get_data_size(), 128 * sizeof(uint16_t));

    auto af = make_float_vec(128);
    auto bf = make_float_vec(128, 99);
    auto a  = floats_to_fp16(af);
    auto b  = floats_to_fp16(bf);
    float d   = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    float ref = fp16_ip_naive_ref(a.data(), b.data(), 128);
    EXPECT_NEAR(d, ref, 1e-2f);
}

TEST(FloatSpace, FP16InnerProduct_VariousDimensions) {
    std::vector<size_t> dims = {1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 128, 256};
    for (size_t dim : dims) {
        deglib::FloatSpace space(dim, deglib::Metric::FP16InnerProduct);
        auto af = make_float_vec(dim);
        auto bf = make_float_vec(dim, static_cast<unsigned>(dim));
        auto a  = floats_to_fp16(af);
        auto b  = floats_to_fp16(bf);
        float d   = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        float ref = fp16_ip_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}

TEST(DistancesBatch, compare_batch_L2) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim, 42);
        std::vector<std::vector<float>> db(20);
        std::vector<const void*> db_arr(20);
        for (size_t j = 0; j < 20; ++j) {
            db[j] = make_float_vec(dim, 100 + j);
            db_arr[j] = db[j].data();
        }

        std::vector<float> expected_dists(20);
        for (size_t j = 0; j < 20; ++j) {
            expected_dists[j] = deglib::distances::L2Float::compare(query.data(), db_arr[j], &dim);
        }

        {
            std::vector<float> dists(20, 0.0f);
            deglib::distances::compare_batch<deglib::distances::L2Float>(query.data(), db_arr.data(), 20, &dim, dists.data());
            for (size_t j = 0; j < 20; ++j) {
                EXPECT_NEAR(dists[j], expected_dists[j], 1e-4f) << "L2Float dim=" << dim << " j=" << j;
            }
        }

        if (dim % 4 == 0) {
            std::vector<float> dists(20, 0.0f);
            deglib::distances::compare_batch<deglib::distances::L2Float4Ext>(query.data(), db_arr.data(), 20, &dim, dists.data());
            for (size_t j = 0; j < 20; ++j) {
                EXPECT_NEAR(dists[j], expected_dists[j], 1e-3f) << "L2Float4Ext dim=" << dim << " j=" << j;
            }
        }

        if (dim % 8 == 0) {
            std::vector<float> dists(20, 0.0f);
            deglib::distances::compare_batch<deglib::distances::L2Float8Ext>(query.data(), db_arr.data(), 20, &dim, dists.data());
            for (size_t j = 0; j < 20; ++j) {
                EXPECT_NEAR(dists[j], expected_dists[j], 1e-3f) << "L2Float8Ext dim=" << dim << " j=" << j;
            }
        }

        if (dim % 16 == 0) {
            std::vector<float> dists(20, 0.0f);
            deglib::distances::compare_batch<deglib::distances::L2Float16Ext>(query.data(), db_arr.data(), 20, &dim, dists.data());
            for (size_t j = 0; j < 20; ++j) {
                EXPECT_NEAR(dists[j], expected_dists[j], 1e-3f) << "L2Float16Ext dim=" << dim << " j=" << j;
            }
        }
    }
}
