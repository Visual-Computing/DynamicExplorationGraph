// test_evp_quantize.cpp — Unit tests for deglib::quantization (quantize_single, quantize_batch for float & fp16 uint16_t)

#include "deglib/distance/fp16.h"
#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/optimization/quantization/quantizer_concept.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

static_assert(deglib::quantization::Quantizer<deglib::quantization::evp::EvpQuantizer>);

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
    float vec[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
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

TEST(EvpQuantize, SingleWithMeanMatchesManualSub) {
    float vec[] = {5.0f, 1.0f, 6.0f, 0.0f, 7.0f, -1.0f, 8.0f, -2.0f};
    float mean[] = {4.0f, 2.0f, 4.0f, 2.0f, 4.0f, 2.0f, 4.0f, 2.0f};
    // Centered: [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    auto result = deglib::quantization::evp::quantize_single(vec, dim, non_zeros, mean);

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(static_cast<uint8_t>(result[0]), 0x50u);
    EXPECT_EQ(static_cast<uint8_t>(result[1]), 0xA0u);

    // Also test EvpQuantizer class
    deglib::quantization::evp::EvpQuantizer q(non_zeros);
    q.set_mean(std::span<const float>(mean, dim));
    auto class_res = q.quantize(vec, 1, dim);
    EXPECT_EQ(class_res, result);
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

    auto single = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 1);
    auto multi = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 4);

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

    deglib::quantization::evp::EvpQuantizer q(non_zeros);
    auto batch_result = q.quantize(data.data(), count, dim);

    for (size_t i = 0; i < count; ++i) {
        auto single_result = deglib::quantization::evp::quantize_single(&data[i * dim], dim, non_zeros, q.mean_ptr());

        size_t offset = i * single_result.size();
        for (size_t b = 0; b < single_result.size(); ++b) {
            EXPECT_EQ(batch_result[offset + b], single_result[b]) << "Mismatch at vector " << i << ", byte " << b;
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

TEST(EvpQuantize, ClassSpanOverloadsMatchPointer) {
    const uint32_t dim = 16;
    const size_t count = 4;
    const uint32_t non_zeros = 4;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    deglib::quantization::evp::EvpQuantizer q(non_zeros);
    const std::span<const float> src(data.data(), data.size());
    EXPECT_EQ(q.quantize(src, dim), q.quantize(data.data(), count, dim));

    std::vector<std::byte> dst(count * 2 * dim / 8);
    q.quantize(src, std::span<std::byte>(dst), dim);
    EXPECT_EQ(dst, q.quantize(data.data(), count, dim));

    EXPECT_THROW(q.quantize(src, 0), std::invalid_argument);
}

TEST(EvpQuantize, ComputeAndSetMeanFloat) {
    const uint32_t dim = 8;
    const size_t count = 4;
    std::vector<float> data = {1.0f,  2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,  9.0f,  10.0f,
                               -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<float> expected_mean = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    deglib::quantization::evp::EvpQuantizer q(4);
    EXPECT_FALSE(q.has_mean());
    q.compute_and_set_mean(data.data(), count, dim, 1);
    EXPECT_TRUE(q.has_mean());
    ASSERT_EQ(q.get_mean().size(), dim);

    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_NEAR(q.get_mean()[d], expected_mean[d], 1e-6f);
    }

    // Multi-threaded execution should produce the exact same mean
    deglib::quantization::evp::EvpQuantizer q_mt(4);
    q_mt.compute_and_set_mean(data.data(), count, dim, 4);
    EXPECT_EQ(q_mt.get_mean(), q.get_mean());

    q.clear_mean();
    EXPECT_FALSE(q.has_mean());
    EXPECT_EQ(q.mean_ptr(), nullptr);
}

TEST(EvpQuantize, ComputeAndSetMeanFP16) {
    const uint32_t dim = 8;
    const size_t count = 4;
    std::vector<float> data = {1.0f,  2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,  9.0f,  10.0f,
                               -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    std::vector<uint16_t> fp16_data(count * dim);
    deglib::distances::fp16::floats_to_fp16(data.data(), fp16_data.data(), count * dim);

    std::vector<float> expected_mean = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    deglib::quantization::evp::EvpQuantizer q(4);
    q.compute_and_set_mean(fp16_data.data(), count, dim, 2);
    EXPECT_TRUE(q.has_mean());
    ASSERT_EQ(q.get_mean().size(), dim);

    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_NEAR(q.get_mean()[d], expected_mean[d], 1e-3f);
    }
}

TEST(EvpQuantize, QuantizeBatchIntoMatchesQuantizeBatch) {
    const uint32_t dim = 64;
    const size_t count = 32;
    const uint32_t non_zeros = 16;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    const size_t bytes_per_evp = 2 * (dim / 8);

    // 1. With auto-fitted mean via EvpQuantizer
    deglib::quantization::evp::EvpQuantizer q(non_zeros);
    auto batch_auto = q.quantize(data.data(), count, dim, 2);
    std::vector<std::byte> into_auto(count * bytes_per_evp);
    deglib::quantization::evp::detail::quantize_batch_into(data.data(), count, dim, non_zeros, into_auto.data(), 2, q.mean_ptr());
    EXPECT_EQ(batch_auto, into_auto);

    // 2. With explicitly provided mean
    std::vector<float> mean(dim);
    for (auto& m : mean) m = dist(rng);
    auto batch_with_mean = deglib::quantization::evp::quantize_batch(data.data(), count, dim, non_zeros, 2, mean.data());
    std::vector<std::byte> into_with_mean(count * bytes_per_evp);
    deglib::quantization::evp::detail::quantize_batch_into(data.data(), count, dim, non_zeros, into_with_mean.data(), 2, mean.data());
    EXPECT_EQ(batch_with_mean, into_with_mean);
}

TEST(EvpQuantize, FP16WithMeanMatchesFP32WithMean) {
    const uint32_t dim = 16;
    const size_t count = 10;
    const uint32_t non_zeros = 6;
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    std::vector<float> mean(dim);
    for (auto& m : mean) m = dist(rng);

    std::vector<uint16_t> fp16_data(count * dim);
    deglib::distances::fp16::floats_to_fp16(data.data(), fp16_data.data(), count * dim);

    deglib::quantization::evp::EvpQuantizer q32(non_zeros, mean);
    deglib::quantization::evp::EvpQuantizer q16(non_zeros, mean);

    auto bits32 = q32.quantize(data.data(), count, dim, 2);
    auto bits16 = q16.quantize(fp16_data.data(), count, dim, 2);

    EXPECT_EQ(bits32, bits16);
}

