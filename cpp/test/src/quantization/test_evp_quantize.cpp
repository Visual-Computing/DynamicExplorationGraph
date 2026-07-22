// test_evp_quantize.cpp — Unit tests for deglib::quantization (quantize_single, quantize_batch)

#include <cstdint>
#include <random>
#include <vector>

#include "quantization/evp_quantize.h"
#include "distances.h"
#include "gtest/gtest.h"

// ============================================================================
// quantize_single
// ============================================================================

TEST(EvpQuantize, SingleBasic) {
    float vec[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    auto result = deglib::quantization::quantize_single(vec, dim, non_zeros);

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

    auto result = deglib::quantization::quantize_single(vec, dim, non_zeros);

    EXPECT_EQ(result.size(), 4u);  // 2 * 16/8

    // ones: idx 0,2,4,6 positive -> byte 0: 0x55, byte 1: 0x00
    // negs: idx 1,3,5,7 negative -> byte 2: 0xAA, byte 3: 0x00
    EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x55u);
    EXPECT_EQ(static_cast<uint8_t>(result[1]), 0x00u);
    EXPECT_EQ(static_cast<uint8_t>(result[2]), 0xAAu);
    EXPECT_EQ(static_cast<uint8_t>(result[3]), 0x00u);
}

// ============================================================================
// quantize_batch
// ============================================================================

TEST(EvpQuantize, BatchBasic) {
    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    };
    const size_t count = 2;
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    auto result = deglib::quantization::quantize_batch(data, count, dim, non_zeros);

    EXPECT_EQ(result.size(), count * 2u);  // 2 vectors * 2 bytes each
}

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

    auto result = deglib::quantization::quantize_batch(data.data(), count, dim, non_zeros);

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

    auto single = deglib::quantization::quantize_batch(data.data(), count, dim, non_zeros, 1);
    auto multi  = deglib::quantization::quantize_batch(data.data(), count, dim, non_zeros, 4);

    EXPECT_EQ(single.size(), multi.size());
    EXPECT_EQ(single, multi);
}

// ============================================================================
// Single vs Batch consistency
// ============================================================================

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

    auto batch_result = deglib::quantization::quantize_batch(data.data(), count, dim, non_zeros);

    for (size_t i = 0; i < count; ++i) {
        auto single_result = deglib::quantization::quantize_single(&data[i * dim], dim, non_zeros);

        size_t offset = i * single_result.size();
        for (size_t b = 0; b < single_result.size(); ++b) {
            EXPECT_EQ(batch_result[offset + b], single_result[b])
                << "Mismatch at vector " << i << ", byte " << b;
        }
    }
}

// ============================================================================
// Integration with distances.h EvpBitsSimilarity
// ============================================================================

TEST(EvpQuantize, EvpSimilarityNoOverlap) {
    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
    };
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    auto r0 = deglib::quantization::quantize_single(data, dim, non_zeros);
    auto r1 = deglib::quantization::quantize_single(data + dim, dim, non_zeros);

    float sim = deglib::distances::EvpBitsSimilarity::compare_naive(
        r0.data(), r1.data(), &dim);

    EXPECT_NEAR(sim, 8.0f, 0.01f);  // dim, no overlap
}

TEST(EvpQuantize, EvpSimilaritySelf) {
    float vec[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    auto result = deglib::quantization::quantize_single(vec, dim, non_zeros);

    float sim = deglib::distances::EvpBitsSimilarity::compare_naive(
        result.data(), result.data(), &dim);

    EXPECT_NEAR(sim, 12.0f, 0.01f); // 2 * non_zeros_active + dim = 4 + 8 = 12
}

// ============================================================================
// Invalid args
// ============================================================================

TEST(EvpQuantize, InvalidDimSingle) {
    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(deglib::quantization::quantize_single(vec, 6, 3), std::invalid_argument);
}

TEST(EvpQuantize, InvalidDimBatch) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(deglib::quantization::quantize_batch(data, 1, 6, 3), std::invalid_argument);
}

TEST(EvpQuantize, InvalidNonZeros) {
    float vec[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const uint32_t dim = 8;
    EXPECT_THROW(deglib::quantization::quantize_single(vec, dim, dim), std::invalid_argument);
    EXPECT_THROW(deglib::quantization::quantize_batch(vec, 1, dim, dim), std::invalid_argument);
}
