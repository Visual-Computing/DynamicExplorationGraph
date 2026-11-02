// test_evp_quantize.cpp — Unit tests for deglib::transformation::EvpQuantizer

#include "deglib/distance/fp16.h"
#include "deglib/transformation.h"
#include "deglib/transformation/evp.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

// ============================================================================
// Single vector quantization (float)
// ============================================================================

TEST(EvpQuantize, SingleBasic) {
    float vec[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    // A dataset with mean 0
    std::vector<float> train_data(dim, 0.0f);
    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(train_data);

    std::vector<std::byte> result(q.output_byte_size());
    q.transform_database(std::span<const float>(vec, dim), std::span<std::byte>(result));

    EXPECT_EQ(result.size(), 2u);  // 2 * (8+7)/8 = 2

    // abs: [1,1,2,2,3,3,4,4] -> top 4 = idx 4,5,6,7
    // idx 4,6 positive -> ones bits 4,6 = 0x50
    // idx 5,7 negative -> negs bits 5,7 = 0xA0
    EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x50u);
    EXPECT_EQ(static_cast<uint8_t>(result[1]), 0xA0u);
}

TEST(EvpQuantize, SingleMaskNoOverlap) {
    // 16-dim: positive and negative values in top 8
    float vec[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t dim = 16;
    const uint32_t non_zeros = 8;

    std::vector<float> train_data(dim, 0.0f);
    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(train_data);

    std::vector<std::byte> result(q.output_byte_size());
    q.transform_database(std::span<const float>(vec, dim), std::span<std::byte>(result));

    EXPECT_EQ(result.size(), 4u);  // 2 * (16+7)/8 = 4

    // ones: idx 0,2,4,6 positive -> byte 0: 0x55, byte 1: 0x00
    // negs: idx 1,3,5,7 negative -> byte 2: 0xAA, byte 3: 0x00
    EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x55u);
    EXPECT_EQ(static_cast<uint8_t>(result[1]), 0x00u);
    EXPECT_EQ(static_cast<uint8_t>(result[2]), 0xAAu);
    EXPECT_EQ(static_cast<uint8_t>(result[3]), 0x00u);
}

TEST(EvpQuantize, SingleWithMeanMatchesManualSub) {
    float vec[] = {5.0f, 1.0f, 6.0f, 0.0f, 7.0f, -1.0f, 8.0f, -2.0f};
    float mean[] = {4.0f, 2.0f, 4.0f, 2.0f, 4.0f, 2.0f, 4.0f, 2.0f};
    // Centered: [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(std::span<const float>(mean, dim));  // mean of a single vector is the vector itself
    std::vector<std::byte> class_res(q.output_byte_size());
    q.transform_database(std::span<const float>(vec, dim), std::span<std::byte>(class_res));

    EXPECT_EQ(class_res.size(), 2u);
    EXPECT_EQ(static_cast<uint8_t>(class_res[0]), 0x50u);
    EXPECT_EQ(static_cast<uint8_t>(class_res[1]), 0xA0u);
}

// ============================================================================
// Arbitrary dimension (not divisible by 8)
// ============================================================================

TEST(EvpQuantize, ArbitraryDimensionsNotDivisibleBy8) {
    for (uint32_t dim : {7u, 9u, 13u, 17u, 77u, 125u}) {
        const uint32_t non_zeros = std::min(dim - 1, std::max(1u, dim / 4));
        const size_t count = 30;

        std::mt19937 rng(1337 + dim);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> data(count * dim);
        for (auto& v : data) v = dist(rng);

        deglib::transformation::EvpQuantizer q(dim, non_zeros);
        q.fit(std::span<const float>(data));

        const size_t expected_mask_bytes = (dim + 7) / 8;
        EXPECT_EQ(q.output_byte_size(), 2 * expected_mask_bytes);

        std::vector<std::byte> result(count * q.output_byte_size());
        q.transform_database(std::span<const float>(data), std::span<std::byte>(result));

        // Verify no overlap between positive and negative masks
        for (size_t i = 0; i < count; ++i) {
            const std::byte* ones = result.data() + i * q.output_byte_size();
            const std::byte* negs = ones + expected_mask_bytes;
            for (size_t b = 0; b < expected_mask_bytes; ++b) {
                EXPECT_EQ(static_cast<uint8_t>(ones[b] & negs[b]), 0u);
            }
            // Unused high bits in the last byte must be zero
            const uint32_t remainder_bits = dim % 8;
            if (remainder_bits != 0) {
                const uint8_t valid_mask = static_cast<uint8_t>((1u << remainder_bits) - 1u);
                EXPECT_EQ(static_cast<uint8_t>(ones[expected_mask_bytes - 1]) & ~valid_mask, 0u);
                EXPECT_EQ(static_cast<uint8_t>(negs[expected_mask_bytes - 1]) & ~valid_mask, 0u);
            }
        }

        // Test FP16 equivalence on arbitrary dimension (fit with fp16_data for float16-native precision)
        std::vector<uint16_t> fp16_data(count * dim);
        deglib::distances::fp16::floats_to_fp16(data.data(), fp16_data.data(), count * dim);
        deglib::distances::fp16::fp16_to_floats(fp16_data.data(), data.data(), count * dim);

        deglib::transformation::EvpQuantizer q_fp16_fit(dim, non_zeros);
        q_fp16_fit.fit(std::span<const float>(data));
        std::vector<std::byte> result_f32_from_f16(count * q_fp16_fit.output_byte_size());
        q_fp16_fit.transform_database(std::span<const float>(data), std::span<std::byte>(result_f32_from_f16));

        std::vector<std::byte> result_fp16(count * q_fp16_fit.output_byte_size());
        q_fp16_fit.transform_database(std::span<const uint16_t>(fp16_data), std::span<std::byte>(result_fp16));
        EXPECT_EQ(result_f32_from_f16, result_fp16);
    }
}

// ============================================================================
// Batch quantization (float & uint16_t FP16)
// ============================================================================

TEST(EvpQuantize, BatchNoOverlap) {
    std::mt19937 rng(42);
    const size_t count = 100;
    const uint32_t dim = 128;
    const uint32_t non_zeros = 32;

    std::vector<float> data(count * dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = dist(rng);
    }

    std::vector<float> zero_mean(dim, 0.0f);
    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(zero_mean);

    std::vector<std::byte> result(count * q.output_byte_size());
    q.transform_database(std::span<const float>(data), std::span<std::byte>(result));

    const size_t mask_bytes = (dim + 7) / 8;
    for (size_t i = 0; i < count; ++i) {
        const std::byte* ones = result.data() + i * 2 * mask_bytes;
        const std::byte* negs = ones + mask_bytes;
        const uint64_t* o = reinterpret_cast<const uint64_t*>(ones);
        const uint64_t* n = reinterpret_cast<const uint64_t*>(negs);
        size_t num_uint64 = mask_bytes / sizeof(uint64_t);
        for (size_t j = 0; j < num_uint64; ++j) {
            EXPECT_EQ(o[j] & n[j], 0ULL) << "Overlap at vector " << i << ", uint64 " << j;
        }
    }
}

TEST(EvpQuantize, BatchMultiThreadConsistent) {
    const size_t count = 4096;
    const uint32_t dim = 256;
    const uint32_t non_zeros = 64;

    std::vector<float> data(count * dim);
    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = dist(rng);
    }

    std::vector<float> zero_mean(dim, 0.0f);
    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(zero_mean);

    std::vector<std::byte> single(count * q.output_byte_size());
    std::vector<std::byte> multi(count * q.output_byte_size());

    q.transform_database(std::span<const float>(data), std::span<std::byte>(single), 1);
    q.transform_database(std::span<const float>(data), std::span<std::byte>(multi), 4);

    EXPECT_EQ(single.size(), multi.size());
    EXPECT_EQ(single, multi);
}

TEST(EvpQuantize, SingleVsBatchConsistency) {
    std::mt19937 rng(7);
    const size_t count = 50;
    const uint32_t dim = 64;
    const uint32_t non_zeros = 16;

    std::vector<float> data(count * dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = dist(rng);
    }

    std::vector<float> zero_mean(dim, 0.0f);
    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(zero_mean);
    std::vector<std::byte> batch_result(count * q.output_byte_size());
    q.transform_database(std::span<const float>(data), std::span<std::byte>(batch_result));

    for (size_t i = 0; i < count; ++i) {
        std::vector<std::byte> single_result(q.output_byte_size());
        q.transform_database(std::span<const float>(&data[i * dim], dim), std::span<std::byte>(single_result));

        size_t offset = i * single_result.size();
        for (size_t b = 0; b < single_result.size(); ++b) {
            EXPECT_EQ(batch_result[offset + b], single_result[b]) << "Mismatch at vector " << i << ", byte " << b;
        }
    }
}

TEST(EvpQuantize, FP32VsFP16BitEquivalence) {
    std::mt19937 rng(12345);
    const size_t count = 20;
    const uint32_t dim = 128;
    const uint32_t non_zeros = 32;

    std::vector<float> float_data(count * dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : float_data) {
        v = dist(rng);
    }

    std::vector<float> zero_mean(dim, 0.0f);
    deglib::transformation::EvpQuantizer q(dim, non_zeros);
    q.fit(zero_mean);
    std::vector<std::byte> fp32_evp_bits(count * q.output_byte_size());
    q.transform_database(std::span<const float>(float_data), std::span<std::byte>(fp32_evp_bits));

    // Convert floats to FP16 (uint16_t)
    std::vector<uint16_t> fp16_data(count * dim);
    deglib::distances::fp16::floats_to_fp16(float_data.data(), fp16_data.data(), count * dim);

    std::vector<std::byte> fp16_evp_bits(count * q.output_byte_size());
    q.transform_database(std::span<const uint16_t>(fp16_data), std::span<std::byte>(fp16_evp_bits));

    // Compare bitmasks for 100% exact equality
    EXPECT_EQ(fp32_evp_bits.size(), fp16_evp_bits.size());
    EXPECT_EQ(fp32_evp_bits, fp16_evp_bits);
}

// ============================================================================
// Invalid argument handling
// ============================================================================

TEST(EvpQuantize, InvalidArguments) {
    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const uint32_t dim = 8;

    std::vector<float> zero_mean(dim, 0.0f);

    // non_zeros >= dim
    deglib::transformation::EvpQuantizer q_invalid_nz(dim, dim);
    q_invalid_nz.fit(zero_mean);
    std::vector<std::byte> out(q_invalid_nz.output_byte_size());
    EXPECT_THROW(q_invalid_nz.transform_database(std::span<const float>(vec, dim), std::span<std::byte>(out)), std::invalid_argument);

    // Buffer too small
    deglib::transformation::EvpQuantizer q(dim, 4);
    q.fit(zero_mean);
    std::vector<std::byte> too_small(1);
    EXPECT_THROW(q.transform_database(std::span<const float>(vec, dim), std::span<std::byte>(too_small)), std::invalid_argument);

    // Input size not multiple of dim
    EXPECT_THROW(q.transform_database(std::span<const float>(vec, 7), std::span<std::byte>(out)), std::invalid_argument);
}

TEST(EvpQuantize, ComputeAndSetMeanFloat) {
    const uint32_t dim = 8;
    const size_t count = 4;
    std::vector<float> data = {1.0f,  2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,  9.0f,  10.0f,
                               -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> expected_mean = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    deglib::transformation::EvpQuantizer q(dim, 4);
    EXPECT_FALSE(q.has_mean());
    q.fit(std::span<const float>(data), 1);
    EXPECT_TRUE(q.has_mean());
    ASSERT_EQ(q.mean().size(), dim);

    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_NEAR(q.mean()[d], expected_mean[d], 1e-6f);
    }

    // Multi-threaded execution should produce the exact same mean
    deglib::transformation::EvpQuantizer q_mt(dim, 4);
    q_mt.fit(std::span<const float>(data), 4);
    EXPECT_EQ(q_mt.mean().size(), q.mean().size());
    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_EQ(q_mt.mean()[d], q.mean()[d]);
    }
}

TEST(EvpQuantize, ComputeAndSetMeanFP16) {
    const uint32_t dim = 8;
    const size_t count = 4;
    std::vector<float> data = {1.0f,  2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,  9.0f,  10.0f,
                               -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<uint16_t> fp16_data(count * dim);
    deglib::distances::fp16::floats_to_fp16(data.data(), fp16_data.data(), count * dim);

    std::vector<float> expected_mean = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    deglib::transformation::EvpQuantizer q(dim, 4);
    q.fit(std::span<const uint16_t>(fp16_data), 2);
    EXPECT_TRUE(q.has_mean());
    ASSERT_EQ(q.mean().size(), dim);

    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_NEAR(q.mean()[d], expected_mean[d], 1e-3f);
    }
}

TEST(EvpQuantize, FP16WithMeanMatchesFP32WithMean) {
    const uint32_t dim = 16;
    const size_t count = 10;
    const uint32_t non_zeros = 6;
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    std::vector<uint16_t> fp16_data(count * dim);
    deglib::distances::fp16::floats_to_fp16(data.data(), fp16_data.data(), count * dim);

    deglib::transformation::EvpQuantizer q32(dim, non_zeros);
    q32.fit(std::span<const float>(data));

    deglib::transformation::EvpQuantizer q16(dim, non_zeros);
    q16.fit(std::span<const uint16_t>(fp16_data));

    std::vector<std::byte> bits32(count * q32.output_byte_size());
    std::vector<std::byte> bits16(count * q16.output_byte_size());

    q32.transform_database(std::span<const float>(data), std::span<std::byte>(bits32), 2);
    q16.transform_database(std::span<const uint16_t>(fp16_data), std::span<std::byte>(bits16), 2);

    EXPECT_EQ(bits32, bits16);
}

TEST(EvpQuantize, AutoCalibrateNonZeros) {
    const uint32_t dim = 32;
    const size_t count = 200;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    // non_zeros = 0 triggers automatic calibration in fit()
    deglib::transformation::EvpQuantizer q(dim, 0);
    EXPECT_EQ(q.non_zeros(), 0u);

    q.fit(std::span<const float>(data));
    EXPECT_GT(q.non_zeros(), 0u);
    EXPECT_LT(q.non_zeros(), dim);

    // Quantizing with auto-calibrated non_zeros succeeds
    std::vector<std::byte> bits(count * q.output_byte_size());
    q.transform_database(std::span<const float>(data), std::span<std::byte>(bits));
    EXPECT_EQ(bits.size(), count * 2 * ((dim + 7) / 8));

    // Calling transform on an un-fitted quantizer throws std::logic_error
    deglib::transformation::EvpQuantizer q_unfitted(dim, 0);
    EXPECT_EQ(q_unfitted.non_zeros(), 0u);
    EXPECT_FALSE(q_unfitted.has_mean());
    EXPECT_FALSE(q_unfitted.is_fitted());

    std::vector<std::byte> bits_unfitted(count * q_unfitted.output_byte_size());
    EXPECT_THROW(q_unfitted.transform_database(std::span<const float>(data), std::span<std::byte>(bits_unfitted)), std::logic_error);
}
