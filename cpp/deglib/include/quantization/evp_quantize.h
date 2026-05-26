#pragma once

#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "concurrent.h"

namespace deglib::quantization {

// ============================================================================
// Conversion: fp32 → EVP bytes
// ============================================================================

/**
 * Quantizes a single fp32 vector to EVP bytes.
 *
 * Layout: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
 *
 * @param embedding  Pointer to dim float values (row-major)
 * @param dim        Dimension (must be divisible by 8)
 * @param non_zeros  Number of top-K elements by absolute value
 * @return std::vector<std::byte> with 2 * dim/8 bytes
 * @throws std::invalid_argument if dim % 8 != 0 or non_zeros >= dim
 */
inline std::vector<std::byte> quantize_single(const float* embedding, uint32_t dim, uint32_t non_zeros) {
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_single: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_single: non_zeros must be < dim");
    }

    const size_t mask_bytes = dim / 8;
    std::vector<std::byte> result(2 * mask_bytes);

    std::vector<std::pair<float, uint32_t>> abs_vals(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        abs_vals[i] = {std::abs(embedding[i]), i};
    }
    std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<uint8_t> is_top(dim, 0);
    for (uint32_t j = dim - non_zeros; j < dim; ++j) {
        is_top[abs_vals[j].second] = 1;
    }

    std::byte* ones_dst = result.data();
    std::byte* negs_dst = result.data() + mask_bytes;

    for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
        int byte_val = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t bit_idx = byte_idx * 8 + bit;
            if (bit_idx >= dim) break;
            if (is_top[bit_idx] && embedding[bit_idx] > 0.0f) {
                byte_val |= (1 << bit);
            }
        }
        ones_dst[byte_idx] = static_cast<std::byte>(byte_val);
    }

    for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
        int byte_val = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t bit_idx = byte_idx * 8 + bit;
            if (bit_idx >= dim) break;
            if (is_top[bit_idx] && embedding[bit_idx] < 0.0f) {
                byte_val |= (1 << bit);
            }
        }
        negs_dst[byte_idx] = static_cast<std::byte>(byte_val);
    }

    return result;
}

/**
 * Quantizes a batch of fp32 vectors to EVP bytes, multi-threaded.
 *
 * Layout: [count][ones (dim/8)][negative_ones (dim/8)]
 *
 * @param data       Pointer to [count x dim] float data (row-major)
 * @param count      Number of vectors
 * @param dim        Dimension (must be divisible by 8)
 * @param non_zeros  Number of top-K elements per vector
 * @param numThreads Number of threads (0 = auto-detect)
 * @return std::vector<std::byte> with count * 2 * dim/8 bytes
 * @throws std::invalid_argument if dim % 8 != 0 or non_zeros >= dim
 */
inline std::vector<std::byte> quantize_batch(const float* data, size_t count, uint32_t dim,
                                              uint32_t non_zeros, size_t numThreads = 0) {
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_batch: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_batch: non_zeros must be < dim");
    }

    const size_t mask_bytes = dim / 8;
    const size_t bytes_per_evp = 2 * mask_bytes;
    std::vector<std::byte> result(count * bytes_per_evp);

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, numThreads, 1,
        [data, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, &result](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, count);

            // Thread-local buffers
            std::vector<std::pair<float, uint32_t>> abs_vals(dim);
            std::vector<uint8_t> is_top(dim, 0);

            for (size_t i = start; i < end; ++i) {
                const float* emb = &data[i * dim];
                std::byte* dst = &result[i * bytes_per_evp];

                // Step 1: top non_zeros by absolute value
                for (uint32_t k = 0; k < dim; ++k) {
                    abs_vals[k] = {std::abs(emb[k]), k};
                }
                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

                is_top.assign(dim, 0);
                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    is_top[abs_vals[j].second] = 1;
                }

                // Step 2: write ones + negs masks
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;

                for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
                    int bv = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const uint32_t bit_idx = byte_idx * 8 + bit;
                        if (bit_idx >= dim) break;
                        if (is_top[bit_idx] && emb[bit_idx] > 0.0f) bv |= (1 << bit);
                    }
                    ones_dst[byte_idx] = static_cast<std::byte>(bv);
                }

                for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
                    int bv = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const uint32_t bit_idx = byte_idx * 8 + bit;
                        if (bit_idx >= dim) break;
                        if (is_top[bit_idx] && emb[bit_idx] < 0.0f) bv |= (1 << bit);
                    }
                    negs_dst[byte_idx] = static_cast<std::byte>(bv);
                }
            }
        });

    return result;
}

/**
 * Quantizes a batch of FP16 (uint16_t) vectors to EVP bytes, multi-threaded.
 *
 * --- Why this works directly on uint16_t ---
 * IEEE 754 half-precision floats (binary16 format) are structured as:
 *   [1 sign bit (bit 15)] [5 exponent bits (bits 10-14)] [10 mantissa bits (bits 0-9)]
 *
 * Floating-point numbers formatted in standard IEEE 754 binary representation
 * exhibit a highly useful property: their absolute lexicographical ordering.
 * By clearing the sign bit (bit 15) via a bitwise AND (x & 0x7FFFu), the remaining
 * 15 bits representing the magnitude (exponent and mantissa) sort as unsigned
 * integers in the exact same monotonic order as their real absolute floating-point
 * values.
 *
 * Consequently:
 *   abs(float_value_A) < abs(float_value_B)
 * is mathematically identical to:
 *   (uint16_value_A & 0x7FFFu) < (uint16_value_B & 0x7FFFu)
 *
 * Note that the sign of each selected element is not ignored; the sign bit (bit 15)
 * is checked using a fast bitwise test: (x & 0x8000u) != 0 for negative, and == 0 for positive.
 * This routes positive elements into the 'ones' mask and negative elements into the 'negative_ones'
 * mask, perfectly matching the mathematical behavior of standard float quantization.
 *
 * This allows us to perform the Top-K selection and sign routing using single-cycle integer
 * bitwise operations. This entirely avoids slow floating-point conversions and float arithmetic,
 * yielding substantial performance benefits.
 *
 * --- Requirements ---
 * 1. The input data must be structured in standard IEEE 754 half-precision float format
 *    (binary16), passed as uint16_t.
 * 2. The data must not contain NaNs (since NaNs violate the strict ordering assumption
 *    of floating-point values). Standard embedding datasets normally do not have NaNs.
 *
 * Layout: [count][ones (dim/8)][negative_ones (dim/8)]
 *
 * @param data       Pointer to [count x dim] uint16_t FP16 data (row-major)
 * @param count      Number of vectors
 * @param dim        Dimension (must be divisible by 8)
 * @param non_zeros  Number of top-K elements per vector
 * @param numThreads Number of threads (0 = auto-detect)
 * @return std::vector<std::byte> with count * 2 * dim/8 bytes
 * @throws std::invalid_argument if dim % 8 != 0 or non_zeros >= dim
 */
inline std::vector<std::byte> quantize_batch(const uint16_t* data, size_t count, uint32_t dim,
                                              uint32_t non_zeros, size_t numThreads = 0) {
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_batch: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_batch: non_zeros must be < dim");
    }

    const size_t mask_bytes = dim / 8;
    const size_t bytes_per_evp = 2 * mask_bytes;
    std::vector<std::byte> result(count * bytes_per_evp);

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, numThreads, 1,
        [data, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, &result](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, count);

            // Thread-local buffers
            std::vector<std::pair<uint16_t, uint32_t>> abs_vals(dim);
            std::vector<uint8_t> is_top(dim, 0);

            for (size_t i = start; i < end; ++i) {
                const uint16_t* emb = &data[i * dim];
                std::byte* dst = &result[i * bytes_per_evp];

                // Step 1: top non_zeros by absolute value (ignoring sign bit in FP16)
                for (uint32_t k = 0; k < dim; ++k) {
                    abs_vals[k] = {static_cast<uint16_t>(emb[k] & 0x7FFFu), k};
                }
                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

                is_top.assign(dim, 0);
                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    is_top[abs_vals[j].second] = 1;
                }

                // Step 2: write ones + negs masks
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;

                for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
                    int bv = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const uint32_t bit_idx = byte_idx * 8 + bit;
                        if (bit_idx >= dim) break;
                        // Positive: sign bit (0x8000) is 0 and absolute value is > 0
                        if (is_top[bit_idx] && (emb[bit_idx] & 0x8000u) == 0 && (emb[bit_idx] & 0x7FFFu) > 0) {
                            bv |= (1 << bit);
                        }
                    }
                    ones_dst[byte_idx] = static_cast<std::byte>(bv);
                }

                for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
                    int bv = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const uint32_t bit_idx = byte_idx * 8 + bit;
                        if (bit_idx >= dim) break;
                        // Negative: sign bit (0x8000) is 1 and absolute value is > 0
                        if (is_top[bit_idx] && (emb[bit_idx] & 0x8000u) != 0 && (emb[bit_idx] & 0x7FFFu) > 0) {
                            bv |= (1 << bit);
                        }
                    }
                    negs_dst[byte_idx] = static_cast<std::byte>(bv);
                }
            }
        });

    return result;
}

/**
 * Quantizes a batch of FP16 (uint16_t) vectors stored as a vector of vectors (std::vector<std::vector<std::byte>>)
 * to EVP bytes, multi-threaded. This completely avoids contiguous allocation copies of the raw dataset.
 *
 * Layout of output: [count][ones (dim/8)][negative_ones (dim/8)]
 *
 * @param data       Vector of vectors containing FP16 float data
 * @param dim        Dimension (must be divisible by 8)
 * @param non_zeros  Number of top-K elements per vector
 * @param numThreads Number of threads (0 = auto-detect)
 * @return std::vector<std::byte> with count * 2 * dim/8 bytes
 * @throws std::invalid_argument if dim % 8 != 0 or non_zeros >= dim
 */
inline std::vector<std::byte> quantize_batch(const std::vector<std::vector<std::byte>>& data, uint32_t dim,
                                              uint32_t non_zeros, size_t numThreads = 0) {
    const size_t count = data.size();
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_batch: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_batch: non_zeros must be < dim");
    }

    const size_t mask_bytes = dim / 8;
    const size_t bytes_per_evp = 2 * mask_bytes;
    std::vector<std::byte> result(count * bytes_per_evp);

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, numThreads, 1,
        [&data, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, &result](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, count);

            // Thread-local buffers
            std::vector<std::pair<uint16_t, uint32_t>> abs_vals(dim);
            std::vector<uint8_t> is_top(dim, 0);

            for (size_t i = start; i < end; ++i) {
                const uint16_t* emb = reinterpret_cast<const uint16_t*>(data[i].data());
                std::byte* dst = &result[i * bytes_per_evp];

                // Step 1: top non_zeros by absolute value (ignoring sign bit in FP16)
                for (uint32_t k = 0; k < dim; ++k) {
                    abs_vals[k] = {static_cast<uint16_t>(emb[k] & 0x7FFFu), k};
                }
                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

                is_top.assign(dim, 0);
                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    is_top[abs_vals[j].second] = 1;
                }

                // Step 2: write ones + negs masks
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;

                for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
                    int bv = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const uint32_t bit_idx = byte_idx * 8 + bit;
                        if (bit_idx >= dim) break;
                        // Positive: sign bit (0x8000) is 0 and absolute value is > 0
                        if (is_top[bit_idx] && (emb[bit_idx] & 0x8000u) == 0 && (emb[bit_idx] & 0x7FFFu) > 0) {
                            bv |= (1 << bit);
                        }
                    }
                    ones_dst[byte_idx] = static_cast<std::byte>(bv);
                }

                for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
                    int bv = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        const uint32_t bit_idx = byte_idx * 8 + bit;
                        if (bit_idx >= dim) break;
                        // Negative: sign bit (0x8000) is 1 and absolute value is > 0
                        if (is_top[bit_idx] && (emb[bit_idx] & 0x8000u) != 0 && (emb[bit_idx] & 0x7FFFu) > 0) {
                            bv |= (1 << bit);
                        }
                    }
                    negs_dst[byte_idx] = static_cast<std::byte>(bv);
                }
            }
        });

    return result;
}
}  // namespace deglib::quantization
