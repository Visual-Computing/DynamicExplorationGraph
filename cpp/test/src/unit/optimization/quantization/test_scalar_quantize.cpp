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

    deglib::quantization::scalar::ScalarQuantizerInt8 quantizer;
    quantizer.fit(vec.data(), count, dim, 0.0f);
    auto result = quantizer.quantize(vec.data(), count, dim, 1);

    ASSERT_EQ(result.size(), 5u);
    EXPECT_NEAR(quantizer.abs_max, 1.0f, 1e-5f);
    EXPECT_EQ(result[0], -127);
    EXPECT_EQ(result[1], -64); // std::round(-0.5 * 127) = -64
    EXPECT_EQ(result[2], 0);
    EXPECT_EQ(result[3], 64);
    EXPECT_EQ(result[4], 127);

    // Test dequantization
    auto dequant = quantizer.dequantize(result.data(), count, dim, 1);
    EXPECT_NEAR(dequant[0], -1.0f, 1e-2f);
    EXPECT_NEAR(dequant[4], 1.0f, 1e-2f);
}

TEST(ScalarQuantize, Int8FitThenQuantizeDistributionConsistency) {
    // Database has vectors with values in [-2.0, 2.0]
    // Query has smaller vectors in [-0.5, 0.5]
    // Quantizing query with the database-fitted quantizer must preserve the database's scale (abs_max = 2.0).
    std::vector<float> db_vec = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> query_vec = {-0.5f, 0.0f, 0.5f, -0.25f, 0.25f};
    const uint32_t dim = 5;

    deglib::quantization::scalar::ScalarQuantizerInt8 quantizer;
    quantizer.fit(db_vec.data(), 1, dim);
    EXPECT_NEAR(quantizer.abs_max, 2.0f, 1e-5f);

    auto quant_db = quantizer.quantize(db_vec.data(), 1, dim);
    auto quant_query = quantizer.quantize(query_vec.data(), 1, dim);

    // In db, 2.0 maps to 127
    EXPECT_EQ(quant_db[4], 127);

    // In query, 0.5 maps to round(0.5 / 2.0 * 127) = round(31.75) = 32
    EXPECT_EQ(quant_query[2], 32);

    // Dequantizing query gives back ~0.5039f
    auto dequant_query = quantizer.dequantize(quant_query.data(), 1, dim);
    EXPECT_NEAR(dequant_query[2], 0.5f, 0.02f);
}

TEST(ScalarQuantize, Int8SymmetricBatchMultiThreadConsistent) {
    const size_t count = 1000;
    const uint32_t dim = 128;
    auto data = generate_random_floats(count, dim, 123);

    deglib::quantization::scalar::ScalarQuantizerInt8 quantizer;
    quantizer.fit(data.data(), count, dim, 0.0f);

    auto single_thread = quantizer.quantize(data.data(), count, dim, 1);
    auto multi_thread = quantizer.quantize(data.data(), count, dim, 4);

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

    deglib::quantization::scalar::ScalarQuantizerInt8 q_f32;
    q_f32.fit(data_f32.data(), count, dim, 0.0f);

    deglib::quantization::scalar::ScalarQuantizerInt8 q_fp16;
    q_fp16.fit(data_fp16.data(), count, dim, 0.0f);

    auto result_from_f32 = q_f32.quantize(data_f32.data(), count, dim, 2);
    auto result_from_fp16 = q_fp16.quantize(data_fp16.data(), count, dim, 2);

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

    deglib::quantization::scalar::ScalarQuantizerUint8 quantizer;
    quantizer.fit(vec.data(), count, dim, 0.0f);
    auto result = quantizer.quantize(vec.data(), count, dim, 1);

    ASSERT_EQ(result.size(), 5u);
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 64u);
    EXPECT_EQ(result[2], 128u);
    EXPECT_EQ(result[3], 191u);
    EXPECT_EQ(result[4], 255u);

    auto dequant = quantizer.dequantize(result.data(), count, dim, 1);
    EXPECT_NEAR(dequant[0], 0.0f, 1e-2f);
    EXPECT_NEAR(dequant[4], 1.0f, 1e-2f);
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

    deglib::quantization::scalar::ScalarQuantizerUint8PerDim quantizer(dim);
    quantizer.fit(vec.data(), count, dim, 0.0f);
    auto result = quantizer.quantize(vec.data(), count, dim, 1);

    ASSERT_EQ(result.size(), 4u);
    // vector 0: (0.0 -> 0), (-5.0 -> 0)
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 0u);
    // vector 1: (10.0 -> 255), (5.0 -> 255)
    EXPECT_EQ(result[2], 255u);
    EXPECT_EQ(result[3], 255u);

    auto dequant = quantizer.dequantize(result.data(), count, dim, 1);
    EXPECT_NEAR(dequant[0], 0.0f, 1e-2f);
    EXPECT_NEAR(dequant[1], -5.0f, 1e-2f);
    EXPECT_NEAR(dequant[2], 10.0f, 1e-2f);
    EXPECT_NEAR(dequant[3], 5.0f, 1e-2f);
}

// ============================================================================
// Factory Function Tests (make_scalar_quantizer_*)
// ============================================================================

#include "deglib/optimization.h"

TEST(ScalarQuantize, FactoryFunctionsMakeAndQuantize) {
    const size_t count = 50;
    const uint32_t dim = 16;
    auto data = generate_random_floats(count, dim, 42);

    auto q_int8 = deglib::optimization::make_scalar_quantizer_int8(data.data(), count, dim);
    EXPECT_TRUE(q_int8.is_fitted);
    auto quant_int8 = q_int8.quantize(data.data(), count, dim);
    EXPECT_EQ(quant_int8.size(), count * dim);

    auto q_uint8 = deglib::optimization::make_scalar_quantizer_uint8(data.data(), count, dim);
    EXPECT_TRUE(q_uint8.is_fitted);
    auto quant_uint8 = q_uint8.quantize(data.data(), count, dim);
    EXPECT_EQ(quant_uint8.size(), count * dim);

    auto q_perdim = deglib::optimization::make_scalar_quantizer_uint8_perdim(data.data(), count, dim);
    EXPECT_TRUE(q_perdim.is_fitted);
    auto quant_perdim = q_perdim.quantize(data.data(), count, dim);
    EXPECT_EQ(quant_perdim.size(), count * dim);

    auto q_int8_pdim = deglib::optimization::make_scalar_quantizer_int8_perdim(data.data(), count, dim);
    EXPECT_TRUE(q_int8_pdim.is_fitted);
    auto quant_int8_pdim = q_int8_pdim.quantize(data.data(), count, dim);
    EXPECT_EQ(quant_int8_pdim.size(), count * dim);
}

TEST(ScalarQuantize, Int8PerDimBasic) {
    // 2 vectors of dim 2
    // Vector 0: [-100.0, -1.0]
    // Vector 1: [ 100.0,  1.0]
    std::vector<float> vec = {-100.0f, -1.0f, 100.0f, 1.0f};
    const uint32_t dim = 2;
    const size_t count = 2;

    deglib::quantization::scalar::ScalarQuantizerInt8PerDim quantizer;
    quantizer.fit(vec.data(), count, dim, 0.0f);
    EXPECT_NEAR(quantizer.abs_maxs[0], 100.0f, 1e-4f);
    EXPECT_NEAR(quantizer.abs_maxs[1], 1.0f, 1e-4f);

    auto result = quantizer.quantize(vec.data(), count, dim);
    EXPECT_EQ(result[0], -127);
    EXPECT_EQ(result[1], -127);
    EXPECT_EQ(result[2], 127);
    EXPECT_EQ(result[3], 127);

    auto dequant = quantizer.dequantize(result.data(), count, dim);
    EXPECT_NEAR(dequant[0], -100.0f, 1.0f);
    EXPECT_NEAR(dequant[1], -1.0f, 0.02f);
}

TEST(ScalarQuantize, Uint8AndUint8PerDimFP16) {
    const size_t count = 20;
    const uint32_t dim = 8;
    auto data_f32 = generate_random_floats(count, dim, 123);
    std::vector<uint16_t> data_fp16(count * dim);
    for (size_t i = 0; i < count * dim; ++i) {
        data_fp16[i] = deglib::distances::fp16::float_to_fp16(data_f32[i]);
    }

    // ScalarQuantizerUint8 with FP16
    deglib::quantization::scalar::ScalarQuantizerUint8 q_u8;
    q_u8.fit(data_fp16.data(), count, dim);
    auto quant_u8 = q_u8.quantize(data_fp16.data(), count, dim);
    EXPECT_EQ(quant_u8.size(), count * dim);

    // ScalarQuantizerUint8PerDim with FP16
    deglib::quantization::scalar::ScalarQuantizerUint8PerDim q_pdim;
    q_pdim.fit(data_fp16.data(), count, dim);
    auto quant_pdim = q_pdim.quantize(data_fp16.data(), count, dim);
    EXPECT_EQ(quant_pdim.size(), count * dim);
}
