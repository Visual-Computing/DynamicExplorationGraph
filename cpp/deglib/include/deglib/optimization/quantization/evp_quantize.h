#pragma once

#include "deglib/concurrent.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace deglib::quantization::evp {

// ============================================================================
// Internal Implementation Details (Low-level kernels & buffer helpers)
// ============================================================================
namespace detail {

// Validates span-based quantize inputs, returns the vector count.
inline size_t checked_span_count(size_t src_size, size_t dst_size, uint32_t dim) {
    if (dim == 0) {
        if (src_size == 0 && dst_size == 0) return 0;
        throw std::invalid_argument("quantize: dim must be > 0");
    }
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize: dim must be divisible by 8");
    }
    if (src_size % dim != 0) {
        throw std::invalid_argument("quantize: src span size must be a multiple of dim");
    }
    const size_t count = src_size / dim;
    if (dst_size < count * 2 * (dim / 8)) {
        throw std::invalid_argument("quantize: dst span too small for src span and dim");
    }
    return count;
}

/**
 * Quantizes a single fp32 vector directly into a pre-allocated EVP byte buffer.
 *
 * Layout: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
 */
inline void quantize_single_into(const float* embedding, uint32_t dim, uint32_t non_zeros, std::byte* out_bytes) {
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_single: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_single: non_zeros must be < dim");
    }
    if (embedding == nullptr || out_bytes == nullptr) {
        throw std::invalid_argument("quantize_single: embedding and out_bytes must not be null");
    }

    const size_t mask_bytes = dim / 8;
    std::byte* ones_dst = out_bytes;
    std::byte* negs_dst = out_bytes + mask_bytes;
    std::memset(out_bytes, 0, 2 * mask_bytes);

    std::vector<std::pair<float, uint32_t>> abs_vals(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        abs_vals[i] = {std::abs(embedding[i]), i};
    }
    std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });
    for (uint32_t j = dim - non_zeros; j < dim; ++j) {
        const uint32_t bit_idx = abs_vals[j].second;
        const float val = embedding[bit_idx];
        if (val > 0.0f) {
            ones_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
        } else if (val < 0.0f) {
            negs_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
        }
    }
}

/**
 * Quantizes a batch of fp32 vectors directly into a pre-allocated destination buffer, multi-threaded.
 *
 * Layout: [count][ones (dim/8)][negative_ones (dim/8)]
 */
inline void quantize_batch_into(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, std::byte* out_bytes, size_t numThreads = 0) {
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_batch: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_batch: non_zeros must be < dim");
    }
    if (count == 0) return;
    if (data == nullptr || out_bytes == nullptr) {
        throw std::invalid_argument("quantize_batch: data and out_bytes must not be null");
    }

    const size_t mask_bytes = dim / 8;
    const size_t bytes_per_evp = 2 * mask_bytes;
    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, numThreads,
        [data, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, out_bytes](size_t chunk_id, size_t) {
            const size_t start = chunk_id * chunk_size;
            const size_t end = std::min(start + chunk_size, count);
            std::vector<std::pair<float, uint32_t>> abs_vals(dim);

            for (size_t i = start; i < end; ++i) {
                const float* emb = &data[i * dim];
                std::byte* dst = &out_bytes[i * bytes_per_evp];
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;
                std::memset(dst, 0, bytes_per_evp);

                for (uint32_t k = 0; k < dim; ++k) {
                    abs_vals[k] = {std::abs(emb[k]), k};
                }
                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
                    return a.first != b.first ? a.first < b.first : a.second < b.second;
                });
                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    const uint32_t bit_idx = abs_vals[j].second;
                    const float val = emb[bit_idx];
                    if (val > 0.0f) {
                        ones_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
                    } else if (val < 0.0f) {
                        negs_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
                    }
                }
            }
        }
    );
}

/**
 * Quantizes a single FP16 (uint16_t) vector directly into a pre-allocated EVP byte buffer.
 *
 * Layout: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
 */
inline void quantize_single_into(const uint16_t* embedding, uint32_t dim, uint32_t non_zeros, std::byte* out_bytes) {
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_single: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_single: non_zeros must be < dim");
    }
    if (embedding == nullptr || out_bytes == nullptr) {
        throw std::invalid_argument("quantize_single: embedding and out_bytes must not be null");
    }

    const size_t mask_bytes = dim / 8;
    std::byte* ones_dst = out_bytes;
    std::byte* negs_dst = out_bytes + mask_bytes;
    std::memset(out_bytes, 0, 2 * mask_bytes);

    std::vector<std::pair<uint16_t, uint32_t>> abs_vals(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        abs_vals[i] = {static_cast<uint16_t>(embedding[i] & 0x7FFFu), i};
    }
    std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });

    for (uint32_t j = dim - non_zeros; j < dim; ++j) {
        const uint32_t bit_idx = abs_vals[j].second;
        const uint16_t h = embedding[bit_idx];
        if ((h & 0x7FFFu) > 0) {
            if ((h & 0x8000u) == 0) {
                ones_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
            } else {
                negs_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
            }
        }
    }
}

/**
 * Quantizes a batch of FP16 (uint16_t) vectors directly into a pre-allocated destination buffer, multi-threaded.
 *
 * Layout: [count][ones (dim/8)][negative_ones (dim/8)]
 */
inline void quantize_batch_into(const uint16_t* data, size_t count, uint32_t dim, uint32_t non_zeros, std::byte* out_bytes, size_t numThreads = 0) {
    if (dim % 8 != 0) {
        throw std::invalid_argument("quantize_batch: dim must be divisible by 8, got " + std::to_string(dim));
    }
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_batch: non_zeros must be < dim");
    }
    if (count == 0) return;
    if (data == nullptr || out_bytes == nullptr) {
        throw std::invalid_argument("quantize_batch: data and out_bytes must not be null");
    }

    const size_t mask_bytes = dim / 8;
    const size_t bytes_per_evp = 2 * mask_bytes;
    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, numThreads,
        [data, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, out_bytes](size_t chunk_id, size_t) {
            const size_t start = chunk_id * chunk_size;
            const size_t end = std::min(start + chunk_size, count);
            std::vector<std::pair<uint16_t, uint32_t>> abs_vals(dim);

            for (size_t i = start; i < end; ++i) {
                const uint16_t* emb = &data[i * dim];
                std::byte* dst = &out_bytes[i * bytes_per_evp];
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;
                std::memset(dst, 0, bytes_per_evp);

                for (uint32_t k = 0; k < dim; ++k) {
                    abs_vals[k] = {static_cast<uint16_t>(emb[k] & 0x7FFFu), k};
                }
                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
                    return a.first != b.first ? a.first < b.first : a.second < b.second;
                });

                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    const uint32_t bit_idx = abs_vals[j].second;
                    const uint16_t h = emb[bit_idx];
                    if ((h & 0x7FFFu) > 0) {
                        if ((h & 0x8000u) == 0) {
                            ones_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
                        } else {
                            negs_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
                        }
                    }
                }
            }
        }
    );
}

}  // namespace detail

// ============================================================================
// Public API: EvpQuantizer
// Satisfies deglib::quantization::Quantizer concept
// ============================================================================

/**
 * EVP quantizer holding the non_zeros hyperparameter.
 *
 * Converts floating point vectors into ternary EVP byte-packed format:
 * [ones (dim/8 bytes)][negative_ones (dim/8 bytes)].
 *
 * Purely stateless: requires no mean vector storage or loading.
 */
class EvpQuantizer {
  public:
    using output_type = std::byte;
    uint32_t non_zeros = 0;

    EvpQuantizer() = default;
    explicit EvpQuantizer(uint32_t nz) : non_zeros(nz) {}

    // --- Quantization ---
    void quantize(const float* src, std::byte* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (count == 1) {
            detail::quantize_single_into(src, dim, non_zeros, dst);
            return;
        }
        detail::quantize_batch_into(src, count, dim, non_zeros, dst, numThreads);
    }

    void quantize(const uint16_t* src_fp16, std::byte* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (count == 1) {
            detail::quantize_single_into(src_fp16, dim, non_zeros, dst);
            return;
        }
        detail::quantize_batch_into(src_fp16, count, dim, non_zeros, dst, numThreads);
    }

    std::vector<std::byte> quantize(const float* src, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<std::byte> result(count * 2 * (dim / 8));
        quantize(src, result.data(), count, dim, numThreads);
        return result;
    }

    std::vector<std::byte> quantize(const uint16_t* src_fp16, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<std::byte> result(count * 2 * (dim / 8));
        quantize(src_fp16, result.data(), count, dim, numThreads);
        return result;
    }

    void quantize(std::span<const float> src, std::span<std::byte> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src.data(), dst.data(), detail::checked_span_count(src.size(), dst.size(), dim), dim, numThreads);
    }

    void quantize(std::span<const uint16_t> src_fp16, std::span<std::byte> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src_fp16.data(), dst.data(), detail::checked_span_count(src_fp16.size(), dst.size(), dim), dim, numThreads);
    }

    std::vector<std::byte> quantize(std::span<const float> src, uint32_t dim, size_t numThreads = 0) const {
        if (dim == 0) {
            if (src.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src.data(), src.size() / dim, dim, numThreads);
    }

    std::vector<std::byte> quantize(std::span<const uint16_t> src_fp16, uint32_t dim, size_t numThreads = 0) const {
        if (dim == 0) {
            if (src_fp16.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src_fp16.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src_fp16.data(), src_fp16.size() / dim, dim, numThreads);
    }
};

// ============================================================================
// Convenience Free Functions
// ============================================================================

inline std::vector<std::byte> quantize_single(const float* embedding, uint32_t dim, uint32_t non_zeros) {
    return EvpQuantizer(non_zeros).quantize(embedding, 1, dim);
}

inline std::vector<std::byte> quantize_single(const uint16_t* embedding, uint32_t dim, uint32_t non_zeros) {
    return EvpQuantizer(non_zeros).quantize(embedding, 1, dim);
}

inline std::vector<std::byte> quantize_batch(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0) {
    return EvpQuantizer(non_zeros).quantize(data, count, dim, numThreads);
}

inline std::vector<std::byte> quantize_batch(const uint16_t* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0) {
    return EvpQuantizer(non_zeros).quantize(data, count, dim, numThreads);
}

}  // namespace deglib::quantization::evp
