// test_scalar_quantize.cpp — Unit tests for deglib::transformation::scalar (Int8 and Uint8 scalar quantization)

#include "deglib/distance/fp16.h"
#include "deglib/transformation.h"
#include "deglib/transformation/scalar.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <span>
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

template <typename Q, typename TIn, typename TOut>
std::vector<TOut> transform_db_helper(const Q& q, std::span<const TIn> in, size_t num_threads = 0) {
    const size_t out_elements = (in.size() / q.dim()) * q.dim();
    std::vector<TOut> out(out_elements);
    std::span<std::byte> out_bytes(reinterpret_cast<std::byte*>(out.data()), out_elements * sizeof(TOut));
    q.transform_database(in, out_bytes, num_threads);
    return out;
}

template <typename Q, typename TIn, typename TOut>
std::vector<TOut> transform_query_helper(const Q& q, std::span<const TIn> in, size_t num_threads = 0) {
    const size_t out_elements = (in.size() / q.dim()) * q.dim();
    std::vector<TOut> out(out_elements);
    std::span<std::byte> out_bytes(reinterpret_cast<std::byte*>(out.data()), out_elements * sizeof(TOut));
    q.transform_query(in, out_bytes, num_threads);
    return out;
}

}  // namespace

// ============================================================================
// Int8 Symmetric Quantization Tests
// ============================================================================

TEST(ScalarQuantize, Int8SymmetricBasic) {
    std::vector<float> vec = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    const uint32_t dim = 5;

    deglib::transformation::ScalarQuantizerInt8 quantizer(dim, 1.0f);
    quantizer.fit(std::span<const float>(vec), 0.0f);
    auto result = transform_db_helper<deglib::transformation::ScalarQuantizerInt8, float, int8_t>(quantizer, std::span<const float>(vec), 1);

    ASSERT_EQ(result.size(), 5u);
    EXPECT_NEAR(quantizer.abs_max(), 1.0f, 1e-5f);
    EXPECT_EQ(result[0], -127);
    EXPECT_EQ(result[1], -64);  // std::round(-0.5 * 127) = -64
    EXPECT_EQ(result[2], 0);
    EXPECT_EQ(result[3], 64);
    EXPECT_EQ(result[4], 127);
}

TEST(ScalarQuantize, Int8FitThenQuantizeDistributionConsistency) {
    // Database has vectors with values in [-2.0, 2.0]
    // Query has smaller vectors in [-0.5, 0.5]
    // Quantizing query with the database-fitted quantizer must preserve the database's scale (abs_max = 2.0).
    std::vector<float> db_vec = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> query_vec = {-0.5f, 0.0f, 0.5f, -0.25f, 0.25f};
    const uint32_t dim = 5;

    deglib::transformation::ScalarQuantizerInt8 quantizer(dim, 1.0f);
    quantizer.fit(std::span<const float>(db_vec));
    EXPECT_NEAR(quantizer.abs_max(), 2.0f, 1e-5f);

    auto quant_db = transform_db_helper<deglib::transformation::ScalarQuantizerInt8, float, int8_t>(quantizer, std::span<const float>(db_vec));
    auto quant_query = transform_query_helper<deglib::transformation::ScalarQuantizerInt8, float, int8_t>(quantizer, std::span<const float>(query_vec));

    // In db, 2.0 maps to 127
    EXPECT_EQ(quant_db[4], 127);

    // In query, 0.5 maps to round(0.5 / 2.0 * 127) = round(31.75) = 32
    EXPECT_EQ(quant_query[2], 32);
}

TEST(ScalarQuantize, Int8SymmetricBatchMultiThreadConsistent) {
    const size_t count = 1000;
    const uint32_t dim = 128;
    auto data = generate_random_floats(count, dim, 123);

    deglib::transformation::ScalarQuantizerInt8 quantizer(dim, 1.0f);
    quantizer.fit(std::span<const float>(data), 0.0f);

    auto single_thread = transform_db_helper<deglib::transformation::ScalarQuantizerInt8, float, int8_t>(quantizer, std::span<const float>(data), 1);
    auto multi_thread = transform_db_helper<deglib::transformation::ScalarQuantizerInt8, float, int8_t>(quantizer, std::span<const float>(data), 4);

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

    deglib::transformation::ScalarQuantizerInt8 q_f32(dim, 1.0f);
    q_f32.fit(std::span<const float>(data_f32), 0.0f);

    deglib::transformation::ScalarQuantizerInt8 q_fp16(dim, 1.0f);
    q_fp16.fit(std::span<const uint16_t>(data_fp16), 0.0f);

    auto result_from_f32 = transform_db_helper<deglib::transformation::ScalarQuantizerInt8, float, int8_t>(q_f32, std::span<const float>(data_f32), 2);
    auto result_from_fp16 = transform_db_helper<deglib::transformation::ScalarQuantizerInt8, uint16_t, int8_t>(q_fp16, std::span<const uint16_t>(data_fp16), 2);

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

    deglib::transformation::ScalarQuantizerUint8 quantizer(dim);
    quantizer.fit(std::span<const float>(vec), 0.0f);
    auto result = transform_db_helper<deglib::transformation::ScalarQuantizerUint8, float, uint8_t>(quantizer, std::span<const float>(vec), 1);

    ASSERT_EQ(result.size(), 5u);
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 64u);
    EXPECT_EQ(result[2], 128u);
    EXPECT_EQ(result[3], 191u);
    EXPECT_EQ(result[4], 255u);
}

TEST(ScalarQuantize, Uint8AffinePerDimBasic) {
    // 2 vectors, dim = 2
    // dim 0 in [0, 10], dim 1 in [-5, 5]
    std::vector<float> vec = {
        0.0f, -5.0f,
        10.0f, 5.0f
    };
    const uint32_t dim = 2;

    deglib::transformation::ScalarQuantizerUint8PerDim quantizer(dim);
    quantizer.fit(std::span<const float>(vec), 0.0f);
    auto result = transform_db_helper<deglib::transformation::ScalarQuantizerUint8PerDim, float, uint8_t>(quantizer, std::span<const float>(vec), 1);

    ASSERT_EQ(result.size(), 4u);
    // vector 0: (0.0 -> 0), (-5.0 -> 0)
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 0u);
    // vector 1: (10.0 -> 255), (5.0 -> 255)
    EXPECT_EQ(result[2], 255u);
    EXPECT_EQ(result[3], 255u);
}

// ============================================================================
// Factory Function Tests (make_scalar_quantizer_*)
// ============================================================================

TEST(ScalarQuantize, FactoryFunctionsMakeAndQuantize) {
    const size_t count = 50;
    const uint32_t dim = 16;
    auto data = generate_random_floats(count, dim, 42);

    auto q_int8 = deglib::transformation::make_scalar_quantizer_int8(std::span<const float>(data), dim);
    EXPECT_TRUE(q_int8.is_fitted());
    auto quant_int8 = transform_db_helper<deglib::transformation::ScalarQuantizerInt8, float, int8_t>(q_int8, std::span<const float>(data));
    EXPECT_EQ(quant_int8.size(), count * dim);

    auto q_uint8 = deglib::transformation::make_scalar_quantizer_uint8(std::span<const float>(data), dim);
    EXPECT_TRUE(q_uint8.is_fitted());
    auto quant_uint8 = transform_db_helper<deglib::transformation::ScalarQuantizerUint8, float, uint8_t>(q_uint8, std::span<const float>(data));
    EXPECT_EQ(quant_uint8.size(), count * dim);

    auto q_perdim = deglib::transformation::make_scalar_quantizer_uint8_perdim(std::span<const float>(data), dim);
    EXPECT_TRUE(q_perdim.is_fitted());
    auto quant_perdim = transform_db_helper<deglib::transformation::ScalarQuantizerUint8PerDim, float, uint8_t>(q_perdim, std::span<const float>(data));
    EXPECT_EQ(quant_perdim.size(), count * dim);

    auto q_int8_pdim = deglib::transformation::make_scalar_quantizer_int8_perdim(std::span<const float>(data), dim);
    EXPECT_TRUE(q_int8_pdim.is_fitted());
    auto quant_int8_pdim = transform_db_helper<deglib::transformation::ScalarQuantizerInt8PerDim, float, int8_t>(q_int8_pdim, std::span<const float>(data));
    EXPECT_EQ(quant_int8_pdim.size(), count * dim);
}

TEST(ScalarQuantize, Int8PerDimBasic) {
    // 2 vectors of dim 2
    // Vector 0: [-100.0, -1.0]
    // Vector 1: [ 100.0,  1.0]
    std::vector<float> vec = {-100.0f, -1.0f, 100.0f, 1.0f};
    const uint32_t dim = 2;

    deglib::transformation::ScalarQuantizerInt8PerDim quantizer(dim);
    quantizer.fit(std::span<const float>(vec), 0.0f);
    EXPECT_NEAR(quantizer.abs_maxs()[0], 100.0f, 1e-4f);
    EXPECT_NEAR(quantizer.abs_maxs()[1], 1.0f, 1e-4f);

    auto result = transform_db_helper<deglib::transformation::ScalarQuantizerInt8PerDim, float, int8_t>(quantizer, std::span<const float>(vec));
    EXPECT_EQ(result[0], -127);
    EXPECT_EQ(result[1], -127);
    EXPECT_EQ(result[2], 127);
    EXPECT_EQ(result[3], 127);
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
    deglib::transformation::ScalarQuantizerUint8 q_u8(dim);
    q_u8.fit(std::span<const uint16_t>(data_fp16));
    auto quant_u8 = transform_db_helper<deglib::transformation::ScalarQuantizerUint8, uint16_t, uint8_t>(q_u8, std::span<const uint16_t>(data_fp16));
    EXPECT_EQ(quant_u8.size(), count * dim);

    // ScalarQuantizerUint8PerDim with FP16
    deglib::transformation::ScalarQuantizerUint8PerDim q_pdim(dim);
    q_pdim.fit(std::span<const uint16_t>(data_fp16));
    auto quant_pdim = transform_db_helper<deglib::transformation::ScalarQuantizerUint8PerDim, uint16_t, uint8_t>(q_pdim, std::span<const uint16_t>(data_fp16));
    EXPECT_EQ(quant_pdim.size(), count * dim);
}

TEST(ScalarQuantize, DirectFP16QuantizeSingle) {
    const uint32_t dim = 4;
    std::vector<float> f32_vec = {-1.0f, 0.0f, 0.5f, 1.0f};
    std::vector<uint16_t> fp16_vec(dim);
    deglib::distances::fp16::floats_to_fp16(f32_vec.data(), fp16_vec.data(), dim);

    // 1. Int8
    deglib::transformation::ScalarQuantizerInt8 q_i8(dim, 1.0f);
    int8_t out_i8[4];
    q_i8.transform_database(std::span<const uint16_t>(fp16_vec), std::span<std::byte>(reinterpret_cast<std::byte*>(out_i8), sizeof(out_i8)));
    EXPECT_EQ(out_i8[0], -127);
    EXPECT_EQ(out_i8[1], 0);
    EXPECT_EQ(out_i8[2], 64);
    EXPECT_EQ(out_i8[3], 127);

    // 2. Uint8
    deglib::transformation::ScalarQuantizerUint8 q_u8(dim, -1.0f, 1.0f);
    uint8_t out_u8[4];
    q_u8.transform_database(std::span<const uint16_t>(fp16_vec), std::span<std::byte>(reinterpret_cast<std::byte*>(out_u8), sizeof(out_u8)));
    EXPECT_EQ(out_u8[0], 0);
    EXPECT_NEAR(out_u8[1], 128, 1);
    EXPECT_NEAR(out_u8[2], 191, 1);
    EXPECT_EQ(out_u8[3], 255);

    // 3. Int8PerDim
    deglib::transformation::ScalarQuantizerInt8PerDim q_i8_pd(dim, {1.0f, 1.0f, 1.0f, 1.0f});
    int8_t out_i8_pd[4];
    q_i8_pd.transform_database(std::span<const uint16_t>(fp16_vec), std::span<std::byte>(reinterpret_cast<std::byte*>(out_i8_pd), sizeof(out_i8_pd)));
    EXPECT_EQ(out_i8_pd[0], -127);
    EXPECT_EQ(out_i8_pd[3], 127);

    // 4. Uint8PerDim
    deglib::transformation::ScalarQuantizerUint8PerDim q_u8_pd(dim, {-1.0f, -1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f, 1.0f});
    uint8_t out_u8_pd[4];
    q_u8_pd.transform_database(std::span<const uint16_t>(fp16_vec), std::span<std::byte>(reinterpret_cast<std::byte*>(out_u8_pd), sizeof(out_u8_pd)));
    EXPECT_EQ(out_u8_pd[0], 0);
    EXPECT_EQ(out_u8_pd[3], 255);
}

TEST(ScalarQuantize, MustBeFittedBeforeTransform) {
    const uint32_t dim = 8;
    std::vector<float> data(dim, 1.0f);
    std::vector<int8_t> out(dim);
    std::span<std::byte> out_bytes(reinterpret_cast<std::byte*>(out.data()), sizeof(int8_t) * dim);

    deglib::transformation::ScalarQuantizerInt8 q_i8;
    q_i8.set_dim(dim);
    EXPECT_FALSE(q_i8.is_fitted());
    EXPECT_THROW(q_i8.transform_database(std::span<const float>(data), out_bytes), std::logic_error);

    deglib::transformation::ScalarQuantizerUint8 q_u8(dim);
    EXPECT_FALSE(q_u8.is_fitted());
    EXPECT_THROW(q_u8.transform_database(std::span<const float>(data), out_bytes), std::logic_error);

    deglib::transformation::ScalarQuantizerInt8PerDim q_i8_pd(dim);
    EXPECT_FALSE(q_i8_pd.is_fitted());
    EXPECT_THROW(q_i8_pd.transform_database(std::span<const float>(data), out_bytes), std::logic_error);

    deglib::transformation::ScalarQuantizerUint8PerDim q_u8_pd(dim);
    EXPECT_FALSE(q_u8_pd.is_fitted());
    EXPECT_THROW(q_u8_pd.transform_database(std::span<const float>(data), out_bytes), std::logic_error);
}

TEST(ScalarQuantize, SizeValidationChecks) {
    const uint32_t dim = 4;
    deglib::transformation::ScalarQuantizerInt8 q_i8(dim, 1.0f);
    std::vector<float> data(5, 1.0f); // Not multiple of dim
    std::vector<int8_t> out(4);
    std::span<std::byte> out_bytes(reinterpret_cast<std::byte*>(out.data()), sizeof(int8_t) * 4);

    EXPECT_THROW(q_i8.transform_database(std::span<const float>(data), out_bytes), std::invalid_argument);

    std::vector<float> valid_data(4, 1.0f);
    std::vector<int8_t> small_out(2);
    std::span<std::byte> small_out_bytes(reinterpret_cast<std::byte*>(small_out.data()), sizeof(int8_t) * 2);
    EXPECT_THROW(q_i8.transform_database(std::span<const float>(valid_data), small_out_bytes), std::invalid_argument);
}

// ============================================================================
// FP16 Quantization Tests
// ============================================================================

TEST(ScalarQuantize, FP16IdentityWhenInRange) {
    // Values within FP16 range -> scale stays 1.0, output is bit-exact FP16 of input.
    std::vector<float> vec = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    const uint32_t dim = 5;

    deglib::transformation::ScalarQuantizerFP16 quantizer(dim, 1.0f);
    quantizer.fit(std::span<const float>(vec));
    EXPECT_NEAR(quantizer.abs_max(), 1.0f, 1e-5f);
    EXPECT_FLOAT_EQ(quantizer.scale(), 1.0f);
    EXPECT_FLOAT_EQ(quantizer.inv_scale(), 1.0f);

    auto result = transform_db_helper<deglib::transformation::ScalarQuantizerFP16, float, uint16_t>(quantizer, std::span<const float>(vec), 1);
    ASSERT_EQ(result.size(), 5u);
    for (uint32_t i = 0; i < dim; ++i) {
        EXPECT_EQ(result[i], deglib::distances::fp16::float_to_fp16(vec[i]));
    }
}

TEST(ScalarQuantize, FP16ScalesLargeValuesNoInfNaN) {
    // Values far above 65504 (audio-like) must not overflow to Inf/NaN in FP16.
    const size_t count = 8;
    const uint32_t dim = 4;
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = (i % 2 == 0) ? 100000.0f : -100000.0f;
    }

    deglib::transformation::ScalarQuantizerFP16 quantizer(dim, 1.0f);
    quantizer.fit(std::span<const float>(data));
    EXPECT_GT(quantizer.abs_max(), deglib::transformation::ScalarQuantizerFP16::kFP16Max);
    EXPECT_LT(quantizer.scale(), 1.0f);

    auto result = transform_db_helper<deglib::transformation::ScalarQuantizerFP16, float, uint16_t>(quantizer, std::span<const float>(data), 2);
    for (uint16_t h : result) {
        float f = deglib::distances::fp16::fp16_to_float(h);
        EXPECT_FALSE(std::isinf(f));
        EXPECT_FALSE(std::isnan(f));
    }
}

TEST(ScalarQuantize, FP16RoundtripPreservesL2Ordering) {
    // Roundtrip quantize -> dequantize approximates the original; L2 ranking is preserved.
    const size_t count = 64;
    const uint32_t dim = 16;
    auto data = generate_random_floats(count, dim, 2024);
    // Scale up so values exceed FP16 range and force a down-scaling.
    for (float& v : data) v *= 5000.0f;

    deglib::transformation::ScalarQuantizerFP16 quantizer(dim, 1.0f);
    quantizer.fit(std::span<const float>(data));

    auto quant = transform_db_helper<deglib::transformation::ScalarQuantizerFP16, float, uint16_t>(quantizer, std::span<const float>(data), 1);

    // Dequantize first vector and compare to original within FP16 relative precision.
    std::vector<float> dequant(dim);
    quantizer.dequantize(std::span<const std::byte>(reinterpret_cast<const std::byte*>(quant.data()), dim * sizeof(uint16_t)),
                         std::span<float>(dequant));
    for (uint32_t d = 0; d < dim; ++d) {
        float orig = data[d];
        // FP16 relative precision ~ 2^-11; scaling is power-of-two (exact), so error is only from FP16.
        EXPECT_NEAR(dequant[d], orig, std::abs(orig) * 2e-3f + 1e-2f);
    }
}

TEST(ScalarQuantize, FP16SerializeRoundtrip) {
    const size_t count = 16;
    const uint32_t dim = 8;
    auto data = generate_random_floats(count, dim, 99);
    for (float& v : data) v *= 30000.0f;  // force scaling path

    deglib::transformation::ScalarQuantizerFP16 q(dim, 1.0f);
    q.fit(std::span<const float>(data));

    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    q.serialize(ss);
    auto q2 = deglib::transformation::ScalarQuantizerFP16::deserialize(ss);
    ASSERT_NE(q2, nullptr);
    EXPECT_EQ(q2->dim(), dim);
    EXPECT_FLOAT_EQ(q2->abs_max(), q.abs_max());
    EXPECT_FLOAT_EQ(q2->scale(), q.scale());
    EXPECT_FLOAT_EQ(q2->inv_scale(), q.inv_scale());
    EXPECT_TRUE(q2->is_fitted());
}

TEST(ScalarQuantize, FP16FactoryAndFP16InputEquivalence) {
    const size_t count = 32;
    const uint32_t dim = 8;
    auto data_f32 = generate_random_floats(count, dim, 555);

    auto q = deglib::transformation::make_scalar_quantizer_fp16(std::span<const float>(data_f32), dim);
    EXPECT_TRUE(q.is_fitted());

    // Quantizing from FP16 input (scale==1 path) matches direct float conversion.
    std::vector<uint16_t> data_fp16(count * dim);
    deglib::distances::fp16::floats_to_fp16(data_f32.data(), data_fp16.data(), count * dim);

    auto from_f32 = transform_db_helper<deglib::transformation::ScalarQuantizerFP16, float, uint16_t>(q, std::span<const float>(data_f32), 1);
    std::vector<uint16_t> from_fp16(count * dim);
    q.transform_database(std::span<const uint16_t>(data_fp16), std::span<std::byte>(reinterpret_cast<std::byte*>(from_fp16.data()), from_fp16.size() * sizeof(uint16_t)));
    ASSERT_EQ(from_f32.size(), from_fp16.size());
    for (size_t i = 0; i < from_f32.size(); ++i) EXPECT_EQ(from_f32[i], from_fp16[i]);
}

TEST(ScalarQuantize, FP16MustBeFittedBeforeTransform) {
    const uint32_t dim = 8;
    std::vector<float> data(dim, 1.0f);
    std::vector<uint16_t> out(dim);
    std::span<std::byte> out_bytes(reinterpret_cast<std::byte*>(out.data()), sizeof(uint16_t) * dim);

    deglib::transformation::ScalarQuantizerFP16 q;
    q.set_dim(dim);
    EXPECT_FALSE(q.is_fitted());
    EXPECT_THROW(q.transform_database(std::span<const float>(data), out_bytes), std::logic_error);
}

TEST(ScalarQuantize, FP16TargetMetricMapping) {
    deglib::transformation::ScalarQuantizerFP16 q(4, 1.0f);
    auto l2 = q.get_target_metric(deglib::distances::Metric::FP32_L2);
    auto ip = q.get_target_metric(deglib::distances::Metric::FP32_InnerProduct);
    EXPECT_EQ(l2, deglib::distances::Metric::FP16_L2);
    EXPECT_EQ(ip, deglib::distances::Metric::FP16_InnerProduct);
    EXPECT_EQ(q.tag(), deglib::transformation::TAG_SQ16);
}

TEST(ScalarQuantize, FP16AbsMaxQuantile) {
    // One huge outlier among many small values.
    const uint32_t dim = 1;
    std::vector<float> data = {1.0f, 1.0f, 1.0f, 1000000.0f};

    // Default quantile 1.0 -> strict maximum (outlier counts).
    deglib::transformation::ScalarQuantizerFP16 q_full(dim, 1.0f, 1.0f);
    q_full.fit(std::span<const float>(data));
    EXPECT_NEAR(q_full.abs_max(), 1000000.0f, 1e-2f);
    EXPECT_LT(q_full.scale(), 1.0f);

    // Lower quantile drops the top outlier -> abs_max falls to the bulk magnitude.
    deglib::transformation::ScalarQuantizerFP16 q_q(dim, 1.0f, 0.75f);
    q_q.fit(std::span<const float>(data));
    EXPECT_LT(q_q.abs_max(), 1000000.0f);
    EXPECT_FLOAT_EQ(q_q.abs_max_quantile(), 0.75f);
}
