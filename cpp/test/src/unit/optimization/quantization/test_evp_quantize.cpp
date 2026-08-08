// test_evp_quantize.cpp — Unit tests for deglib::quantization (quantize_single, quantize_batch for float & fp16 uint16_t)

#include <cstdint>
#include <random>
#include <vector>

#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/distance/fp16.h"
#include "gtest/gtest.h"

// ============================================================================
// Single vector quantization (float)
// ============================================================================

TEST(EvpQuantize, SingleBasic) {
    float vec[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    auto result = deglib::quantization::evp::quantize_single(vec, dim, non_zeros);

    EXPECT_EQ(result.size(), 2u);  // 2 * 8/8

    // abs: [1,1,2,2,3,3,4,4] -> top 4 = idx 4,5,6,7
    // idx 4,6 positive -> ones bits 4,6 = 0x50
    // idx 5,7 negative -> negs bits 5,7 = 0xA0
    EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x50u);
    EXPECT_EQ(static_cast<uint8_t>(result[1]), 0xA0u);
}

TEST(EvpQuantize, SingleMaskNoOverlap) {
    // 16-dim: positive and negative values in top 8
    float vec[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f,
                   0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t dim = 16;
    const uint32_t non_zeros = 8;

    auto result = deglib::quantization::evp::quantize_single(vec, dim, non_zeros);

    EXPECT_EQ(result.size(), 4u);  // 2 * 16/8

    // ones: idx 0,2,4,6 positive -> byte 0: 0x55, byte 1: 0x00
    // negs: idx 1,3,5,7 negative -> byte 2: 0xAA, byte 3: 0x00
    EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x55u);
    EXPECT_EQ(static_cast<uint8_t>(result[1]), 0x00u);
    EXPECT_EQ(static_cast<uint8_t>(result[2]), 0xAAu);
    EXPECT_EQ(static_cast<uint8_t>(result[3]), 0x00u);
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

    auto result = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros);

    for (size_t i = 0; i < count; ++i) {
        const std::byte* ones = result.data() + i * 2 * dim / 8;
        const std::byte* negs = ones + dim / 8;
        const uint64_t* o = reinterpret_cast<const uint64_t*>(ones);
        const uint64_t* n = reinterpret_cast<const uint64_t*>(negs);
        size_t num_uint64 = (dim / 8) / sizeof(uint64_t);
        for (size_t j = 0; j < num_uint64; ++j) {
            EXPECT_EQ(o[j] & n[j], 0ULL)
                << "Overlap at vector " << i << ", uint64 " << j;
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

    auto single = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 1);
    auto multi  = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 4);

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

    auto batch_result = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros);

    for (size_t i = 0; i < count; ++i) {
        auto single_result = deglib::quantization::evp::quantize_single(&data[i * dim], dim, non_zeros);

        size_t offset = i * single_result.size();
        for (size_t b = 0; b < single_result.size(); ++b) {
            EXPECT_EQ(batch_result[offset + b], single_result[b])
                << "Mismatch at vector " << i << ", byte " << b;
        }
    }
}

TEST(EvpQuantize, FP32VsFP16BitEquivalence) {
    // Verifies that quantizing float vectors directly vs. converting float vectors to FP16
    // and then quantizing produces 100% identical EVP bitmasks.
    std::mt19937 rng(12345);
    const size_t count = 20;
    const uint32_t dim = 128;
    const uint32_t non_zeros = 32;

    std::vector<float> float_data(count * dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : float_data) {
        v = dist(rng);
    }

    // FP32 quantization
    auto fp32_evp_bits = deglib::quantization::evp::quantize_batch(float_data.data(), count, dim, non_zeros);

    // Convert floats to FP16 (uint16_t)
    std::vector<uint16_t> fp16_data(count * dim);
    deglib::distances::fp16::floats_to_fp16(float_data.data(), fp16_data.data(), count * dim);

    // FP16 quantization
    auto fp16_evp_bits = deglib::quantization::evp::quantize_batch(fp16_data.data(), count, dim, non_zeros);

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

    // Dim not divisible by 8
    EXPECT_THROW(deglib::quantization::evp::quantize_single(vec, 6, 3), std::invalid_argument);
    EXPECT_THROW(deglib::quantization::evp::quantize_batch(vec, 1, 6, 3), std::invalid_argument);

    // non_zeros >= dim
    EXPECT_THROW(deglib::quantization::evp::quantize_single(vec, dim, dim), std::invalid_argument);
    EXPECT_THROW(deglib::quantization::evp::quantize_batch(vec, 1, dim, dim), std::invalid_argument);
}
