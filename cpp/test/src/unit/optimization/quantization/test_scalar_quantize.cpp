// test_scalar_quantize.cpp — Unit tests for deglib::quantization::scalar (Int8 and Uint8 scalar quantization)

#include "deglib/distance/fp16.h"
#include "deglib/optimization/quantization/scalar_quantize.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// Helper to generate normalized random float vectors
std::vector<float> generate_random_floats(size_t count, uint32_t dim, unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            float v = dist(rng);
            data[i * dim + d] = v;
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        if (norm > 1e-12f) {
            for (uint32_t d = 0; d < dim; ++d) {
                data[i * dim + d] /= norm;
            }
        }
    }
    return data;
}

}  // namespace

// ============================================================================
// Int8 Symmetric Quantization Tests
// ============================================================================

TEST(ScalarQuantize, Int8SymmetricBasic) {
    std::vector<float> vec = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    const uint32_t dim = 5;
    const size_t count = 1;

    deglib::quantization::scalar::SymCalibratorInt8 cal;
    auto result = deglib::quantization::scalar::quantize_int8_symmetric(vec.data(), count, dim, 0.0f, 1, &cal);

    ASSERT_EQ(result.size(), 5u);
    EXPECT_NEAR(cal.abs_max, 1.0f, 1e-5f);
    EXPECT_EQ(result[0], -127);
    EXPECT_EQ(result[1], -64); // std::round(-0.5 * 127) = -64
    EXPECT_EQ(result[2], 0);
    EXPECT_EQ(result[3], 64);
    EXPECT_EQ(result[4], 127);

    // Test dequantization
    EXPECT_NEAR(cal.transform_back(result[0]), -1.0f, 1e-2f);
    EXPECT_NEAR(cal.transform_back(result[4]), 1.0f, 1e-2f);
}

TEST(ScalarQuantize, Int8SymmetricBatchMultiThreadConsistent) {
    const size_t count = 1000;
    const uint32_t dim = 128;
    auto data = generate_random_floats(count, dim, 123);

    auto single_thread = deglib::quantization::scalar::quantize_int8_symmetric(data.data(), count, dim, 0.0f, 1);
    auto multi_thread = deglib::quantization::scalar::quantize_int8_symmetric(data.data(), count, dim, 0.0f, 4);

    ASSERT_EQ(single_thread.size(), multi_thread.size());
    for (size_t i = 0; i < single_thread.size(); ++i) {
        EXPECT_EQ(single_thread[i], multi_thread[i]);
    }
}

TEST(ScalarQuantize, Int8FP16Equivalence) {
    const size_t count = 100;
    const uint32_t dim = 64;
    auto data_f32 = generate_random_floats(count, dim, 777);

    std::vector<uint16_t> data_fp16(count * dim);
    deglib::distances::fp16::floats_to_fp16(data_f32.data(), data_fp16.data(), count * dim);

    auto result_from_f32 = deglib::quantization::scalar::quantize_int8_symmetric(data_f32.data(), count, dim, 0.0f, 2);
    auto result_from_fp16 = deglib::quantization::scalar::quantize_int8_symmetric(data_fp16.data(), count, dim, 0.0f, 2);

    ASSERT_EQ(result_from_f32.size(), result_from_fp16.size());
    // Minor rounding differences at half precision boundaries (+/- 1) are acceptable, but most should be identical
    size_t exact_matches = 0;
    for (size_t i = 0; i < result_from_f32.size(); ++i) {
        int diff = std::abs(int(result_from_f32[i]) - int(result_from_fp16[i]));
        EXPECT_LE(diff, 1);
        if (diff == 0) exact_matches++;
    }
    EXPECT_GT(exact_matches, result_from_f32.size() * 0.95);
}

// ============================================================================
// Uint8 Affine Quantization Tests
// ============================================================================

TEST(ScalarQuantize, Uint8AffineGlobalBasic) {
    std::vector<float> vec = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    const uint32_t dim = 5;
    const size_t count = 1;

    deglib::quantization::scalar::AffineCalibratorUint8 cal;
    auto result = deglib::quantization::scalar::quantize_uint8_affine(vec.data(), count, dim, 0.0f, 1, &cal);

    ASSERT_EQ(result.size(), 5u);
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 64u);
    EXPECT_EQ(result[2], 128u);
    EXPECT_EQ(result[3], 191u);
    EXPECT_EQ(result[4], 255u);

    EXPECT_NEAR(cal.transform_back(result[0]), 0.0f, 1e-2f);
    EXPECT_NEAR(cal.transform_back(result[4]), 1.0f, 1e-2f);
}

TEST(ScalarQuantize, Uint8AffinePerDimBasic) {
    // 2 vectors, dim = 2
    // dim 0 in [0, 10], dim 1 in [-5, 5]
    std::vector<float> vec = {
        0.0f, -5.0f,
        10.0f, 5.0f
    };
    const uint32_t dim = 2;
    const size_t count = 2;

    deglib::quantization::scalar::AffinePerDimCalibratorUint8 cal(dim);
    auto result = deglib::quantization::scalar::quantize_uint8_affine_perdim(vec.data(), count, dim, 0.0f, 1, &cal);

    ASSERT_EQ(result.size(), 4u);
    // vector 0: (0.0 -> 0), (-5.0 -> 0)
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 0u);
    // vector 1: (10.0 -> 255), (5.0 -> 255)
    EXPECT_EQ(result[2], 255u);
    EXPECT_EQ(result[3], 255u);
}
