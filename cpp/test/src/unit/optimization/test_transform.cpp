// test_transform.cpp — Unit tests for deglib::optimization::mips_l2_transform and mips_l2_transform_query
//
// Covers: mips_l2_transform, mips_l2_transform_query

#include "deglib/optimization/transform.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <vector>

TEST(TransformTest, MipsL2TransformBasic) {
    const size_t count = 3;
    const size_t dim = 2;
    std::vector<float> input = {
        3.0f, 4.0f,  // norm^2 = 25 (norm = 5)
        0.0f, 0.0f,  // norm^2 = 0
        1.0f, 2.0f   // norm^2 = 5
    };
    std::vector<float> output(count * (dim + 1));

    float max_norm = deglib::optimization::mips_l2_transform(input.data(), count, dim, output.data());

    EXPECT_NEAR(max_norm, 5.0f, 1e-5f);

    const size_t new_dim = dim + 1;

    // First vector: extra dim = sqrt(25 - 25) = 0
    EXPECT_FLOAT_EQ(output[0 * new_dim + 0], 3.0f);
    EXPECT_FLOAT_EQ(output[0 * new_dim + 1], 4.0f);
    EXPECT_NEAR(output[0 * new_dim + 2], 0.0f, 1e-5f);

    // Second vector: extra dim = sqrt(25 - 0) = 5
    EXPECT_FLOAT_EQ(output[1 * new_dim + 0], 0.0f);
    EXPECT_FLOAT_EQ(output[1 * new_dim + 1], 0.0f);
    EXPECT_NEAR(output[1 * new_dim + 2], 5.0f, 1e-5f);

    // Third vector: extra dim = sqrt(25 - 5) = sqrt(20) ~ 4.472136
    EXPECT_FLOAT_EQ(output[2 * new_dim + 0], 1.0f);
    EXPECT_FLOAT_EQ(output[2 * new_dim + 1], 2.0f);
    EXPECT_NEAR(output[2 * new_dim + 2], std::sqrt(20.0f), 1e-5f);

    // All transformed vectors must have norm^2 == 25
    for (size_t i = 0; i < count; ++i) {
        float norm_sq = 0.0f;
        for (size_t j = 0; j < new_dim; ++j) {
            float val = output[i * new_dim + j];
            norm_sq += val * val;
        }
        EXPECT_NEAR(norm_sq, 25.0f, 1e-4f);
    }
}

TEST(TransformTest, MipsL2TransformQueryPadsZero) {
    const size_t count = 2;
    const size_t dim = 3;
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> output(count * (dim + 1));

    deglib::optimization::mips_l2_transform_query(input.data(), count, dim, output.data());

    const size_t new_dim = dim + 1;
    EXPECT_FLOAT_EQ(output[0 * new_dim + 0], 1.0f);
    EXPECT_FLOAT_EQ(output[0 * new_dim + 1], 2.0f);
    EXPECT_FLOAT_EQ(output[0 * new_dim + 2], 3.0f);
    EXPECT_FLOAT_EQ(output[0 * new_dim + 3], 0.0f);

    EXPECT_FLOAT_EQ(output[1 * new_dim + 0], 4.0f);
    EXPECT_FLOAT_EQ(output[1 * new_dim + 1], 5.0f);
    EXPECT_FLOAT_EQ(output[1 * new_dim + 2], 6.0f);
    EXPECT_FLOAT_EQ(output[1 * new_dim + 3], 0.0f);
}

TEST(TransformTest, MipsL2TransformDistanceEquivalence) {
    const size_t db_count = 3;
    const size_t query_count = 2;
    const size_t dim = 4;
    std::vector<float> db_input = {1.0f, 0.5f, -0.2f, 0.8f, 0.0f, 1.2f, 0.5f, 0.1f, -0.5f, 0.2f, 1.0f, 0.0f};
    std::vector<float> query_input = {0.5f, -0.3f, 0.8f, 0.1f, 1.2f, 0.0f, 0.0f, 0.4f};

    std::vector<float> db_output(db_count * (dim + 1));
    std::vector<float> query_output(query_count * (dim + 1));

    float max_norm = deglib::optimization::mips_l2_transform(db_input.data(), db_count, dim, db_output.data());
    deglib::optimization::mips_l2_transform_query(query_input.data(), query_count, dim, query_output.data());

    float M_sq = max_norm * max_norm;
    const size_t new_dim = dim + 1;

    // Verify L2 distance relationship: ||q' - x'_i||^2 == ||q||^2 + M^2 - 2 * <q, x_i>
    for (size_t q = 0; q < query_count; ++q) {
        float q_norm_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float q_val = query_input[q * dim + d];
            q_norm_sq += q_val * q_val;
        }

        for (size_t i = 0; i < db_count; ++i) {
            float l2_dist_sq = 0.0f;
            for (size_t d = 0; d < new_dim; ++d) {
                float diff = query_output[q * new_dim + d] - db_output[i * new_dim + d];
                l2_dist_sq += diff * diff;
            }

            float ip_val = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                ip_val += query_input[q * dim + d] * db_input[i * dim + d];
            }

            float expected_l2_dist_sq = q_norm_sq + M_sq - 2.0f * ip_val;
            EXPECT_NEAR(l2_dist_sq, expected_l2_dist_sq, 1e-4f);
        }
    }
}

TEST(TransformTest, MipsL2TransformVectorOverloads) {
    const size_t count = 2;
    const size_t dim = 2;
    std::vector<float> input = {3.0f, 4.0f, 0.0f, 0.0f};

    auto [output, max_norm] = deglib::optimization::mips_l2_transform(input, count, dim);
    EXPECT_NEAR(max_norm, 5.0f, 1e-5f);
    EXPECT_EQ(output.size(), count * (dim + 1));

    auto q_output = deglib::optimization::mips_l2_transform_query(input, count, dim);
    EXPECT_EQ(q_output.size(), count * (dim + 1));
    EXPECT_FLOAT_EQ(q_output[2], 0.0f);
}
