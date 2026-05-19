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
        static_cast<size_t>(0), num_chunks, numThreads,
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

}  // namespace deglib::quantization
