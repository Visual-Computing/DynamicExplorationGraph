#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace deglib::quantization::evp {

// ============================================================================
// Internal Implementation Details (Low-level SIMD kernels & buffer helpers)
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
 * Computes dimension-wise (column-wise) mean of a batch of FP32 vectors.
 * Accumulates across threads in double precision without mutex locks for numerical precision.
 */
inline void compute_column_mean(const float* src, size_t count, uint32_t dim, float* out_mean, size_t numThreads = 0) {
    if (count == 0 || dim == 0 || src == nullptr || out_mean == nullptr) return;

    const size_t actual_threads = (numThreads == 0)
        ? std::clamp(static_cast<size_t>(std::thread::hardware_concurrency()), size_t{1}, size_t{64})
        : numThreads;
    std::vector<double> thread_sums(actual_threads * dim, 0.0);

    deglib::concurrent::parallel_batch_for(static_cast<size_t>(0), count, actual_threads, [&](size_t begin, size_t end, size_t thread_id) {
        double* my_sum = thread_sums.data() + thread_id * dim;
#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            for (size_t i = begin; i < end; ++i) {
                const float* vec = src + i * dim;
                uint32_t d = 0;
                for (; d + 8 <= dim; d += 8) {
                    __m128 vf1 = _mm_loadu_ps(vec + d);
                    __m128 vf2 = _mm_loadu_ps(vec + d + 4);
                    __m256d vd1 = _mm256_cvtps_pd(vf1);
                    __m256d vd2 = _mm256_cvtps_pd(vf2);
                    __m256d cur1 = _mm256_loadu_pd(my_sum + d);
                    __m256d cur2 = _mm256_loadu_pd(my_sum + d + 4);
                    _mm256_storeu_pd(my_sum + d, _mm256_add_pd(cur1, vd1));
                    _mm256_storeu_pd(my_sum + d + 4, _mm256_add_pd(cur2, vd2));
                }
                for (; d < dim; ++d) {
                    my_sum[d] += static_cast<double>(vec[d]);
                }
            }
            return;
        }
#endif
        for (size_t i = begin; i < end; ++i) {
            const float* vec = src + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                my_sum[d] += static_cast<double>(vec[d]);
            }
        }
    });

    const double inv_count = 1.0 / static_cast<double>(count);
    for (uint32_t d = 0; d < dim; ++d) {
        double total = 0.0;
        for (size_t t = 0; t < actual_threads; ++t) {
            total += thread_sums[t * dim + d];
        }
        out_mean[d] = static_cast<float>(total * inv_count);
    }
}

/**
 * Quantizes a single FP32 vector directly into a pre-allocated EVP byte buffer.
 * If mean is provided, it is subtracted from the embedding prior to quantization.
 *
 * Layout: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
 */
inline void quantize_single_into(const float* embedding, uint32_t dim, uint32_t non_zeros, std::byte* out_bytes, const float* mean = nullptr) {
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
    if (mean != nullptr) {
        for (uint32_t i = 0; i < dim; ++i) {
            abs_vals[i] = {std::abs(embedding[i] - mean[i]), i};
        }
    } else {
        for (uint32_t i = 0; i < dim; ++i) {
            abs_vals[i] = {std::abs(embedding[i]), i};
        }
    }

    std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });

    for (uint32_t j = dim - non_zeros; j < dim; ++j) {
        const uint32_t bit_idx = abs_vals[j].second;
        const float val = (mean != nullptr) ? (embedding[bit_idx] - mean[bit_idx]) : embedding[bit_idx];
        if (val > 0.0f) {
            ones_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
        } else if (val < 0.0f) {
            negs_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
        }
    }
}

/**
 * Quantizes a batch of FP32 vectors directly into a pre-allocated destination buffer, multi-threaded.
 * If mean is provided, it is subtracted from each vector prior to quantization.
 *
 * Layout: [count][ones (dim/8)][negative_ones (dim/8)]
 */
inline void quantize_batch_into(
    const float* data,
    size_t count,
    uint32_t dim,
    uint32_t non_zeros,
    std::byte* out_bytes,
    size_t numThreads = 0,
    const float* mean = nullptr
) {
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
        [data, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, mean, out_bytes](size_t chunk_id, size_t) {
            const size_t start = chunk_id * chunk_size;
            const size_t end = std::min(start + chunk_size, count);
            std::vector<std::pair<float, uint32_t>> abs_vals(dim);

            for (size_t i = start; i < end; ++i) {
                const float* emb = &data[i * dim];
                std::byte* dst = &out_bytes[i * bytes_per_evp];
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;
                std::memset(dst, 0, bytes_per_evp);

                if (mean != nullptr) {
                    for (uint32_t k = 0; k < dim; ++k) {
                        abs_vals[k] = {std::abs(emb[k] - mean[k]), k};
                    }
                } else {
                    for (uint32_t k = 0; k < dim; ++k) {
                        abs_vals[k] = {std::abs(emb[k]), k};
                    }
                }

                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
                    return a.first != b.first ? a.first < b.first : a.second < b.second;
                });

                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    const uint32_t bit_idx = abs_vals[j].second;
                    const float val = (mean != nullptr) ? (emb[bit_idx] - mean[bit_idx]) : emb[bit_idx];
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
 * If mean_fp16 is provided, it is subtracted in SIMD native FP16 hardware prior to quantization.
 *
 * Layout: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
 */
inline void quantize_single_into(
    const uint16_t* embedding,
    uint32_t dim,
    uint32_t non_zeros,
    std::byte* out_bytes,
    const uint16_t* mean_fp16 = nullptr
) {
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
    std::vector<uint16_t> diff_buf(dim);

    if (mean_fp16 != nullptr) {
        deglib::distances::fp16::sub_and_abs_fp16(embedding, mean_fp16, dim, diff_buf.data(), abs_vals.data());
    } else {
        for (uint32_t i = 0; i < dim; ++i) {
            abs_vals[i] = {static_cast<uint16_t>(embedding[i] & 0x7FFFu), i};
        }
    }

    std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });

    for (uint32_t j = dim - non_zeros; j < dim; ++j) {
        const uint32_t bit_idx = abs_vals[j].second;
        const uint16_t h = (mean_fp16 != nullptr) ? diff_buf[bit_idx] : embedding[bit_idx];
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
 * If mean_fp16 is provided, it is subtracted in SIMD native FP16 hardware prior to quantization.
 *
 * Layout: [count][ones (dim/8)][negative_ones (dim/8)]
 */
inline void quantize_batch_into(
    const uint16_t* data,
    size_t count,
    uint32_t dim,
    uint32_t non_zeros,
    std::byte* out_bytes,
    size_t numThreads = 0,
    const uint16_t* mean_fp16 = nullptr
) {
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
        [data, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, mean_fp16, out_bytes](size_t chunk_id, size_t) {
            const size_t start = chunk_id * chunk_size;
            const size_t end = std::min(start + chunk_size, count);

            std::vector<std::pair<uint16_t, uint32_t>> abs_vals(dim);
            std::vector<uint16_t> diff_buf(dim);

            for (size_t i = start; i < end; ++i) {
                const uint16_t* emb = &data[i * dim];
                std::byte* dst = &out_bytes[i * bytes_per_evp];
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;
                std::memset(dst, 0, bytes_per_evp);

                if (mean_fp16 != nullptr) {
                    deglib::distances::fp16::sub_and_abs_fp16(emb, mean_fp16, dim, diff_buf.data(), abs_vals.data());
                } else {
                    for (uint32_t k = 0; k < dim; ++k) {
                        abs_vals[k] = {static_cast<uint16_t>(emb[k] & 0x7FFFu), k};
                    }
                }

                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
                    return a.first != b.first ? a.first < b.first : a.second < b.second;
                });

                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    const uint32_t bit_idx = abs_vals[j].second;
                    const uint16_t h = (mean_fp16 != nullptr) ? diff_buf[bit_idx] : emb[bit_idx];
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
 * Stateful EVP quantizer holding the non_zeros hyperparameter and the dataset centroid (mean).
 *
 * EVP quantization automatically learns and subtracts the mean vector upon the first quantize()
 * call (or via fit()), ensuring optimal bit-level feature variance without manual configuration.
 */
class EvpQuantizer {
  public:
    using output_type = std::byte;
    uint32_t non_zeros = 0;

    EvpQuantizer() = default;
    explicit EvpQuantizer(uint32_t nz) : non_zeros(nz) {}
    EvpQuantizer(uint32_t nz, std::span<const float> m) : non_zeros(nz) { set_mean(m); }
    EvpQuantizer(uint32_t nz, const std::vector<float>& m) : non_zeros(nz) { set_mean(m); }

    // --- State & Mean Access ---
    bool has_mean() const noexcept { return !mean_.empty(); }
    const std::vector<float>& get_mean() const noexcept { return mean_; }
    const std::vector<uint16_t>& get_mean_fp16() const noexcept { return mean_fp16_; }
    const float* mean_ptr() const noexcept { return mean_.empty() ? nullptr : mean_.data(); }
    const uint16_t* mean_fp16_ptr() const noexcept { return mean_fp16_.empty() ? nullptr : mean_fp16_.data(); }

    void clear_mean() noexcept {
        mean_.clear();
        mean_fp16_.clear();
    }

    void set_mean(std::span<const float> m) {
        mean_.assign(m.begin(), m.end());
        mean_fp16_.resize(mean_.size());
        deglib::distances::fp16::floats_to_fp16(mean_.data(), mean_fp16_.data(), mean_.size());
    }

    void set_mean_fp16(std::span<const uint16_t> m) {
        mean_fp16_.assign(m.begin(), m.end());
        mean_.resize(mean_fp16_.size());
        deglib::distances::fp16::fp16_to_floats(mean_fp16_.data(), mean_.data(), mean_fp16_.size());
    }

    // --- Training / Parameter Fitting ---
    void fit(const float* src, size_t count, uint32_t dim, size_t numThreads = 0) {
        if (count == 0 || dim == 0) return;
        mean_.resize(dim);
        detail::compute_column_mean(src, count, dim, mean_.data(), numThreads);
        mean_fp16_.resize(dim);
        deglib::distances::fp16::floats_to_fp16(mean_.data(), mean_fp16_.data(), dim);
    }

    void fit(const uint16_t* src_fp16, size_t count, uint32_t dim, size_t numThreads = 0) {
        if (count == 0 || dim == 0) return;
        mean_.resize(dim);
        deglib::distances::fp16::compute_column_mean(src_fp16, count, dim, mean_.data(), numThreads);
        mean_fp16_.resize(dim);
        deglib::distances::fp16::floats_to_fp16(mean_.data(), mean_fp16_.data(), dim);
    }

    void compute_and_set_mean(const float* src, size_t count, uint32_t dim, size_t numThreads = 0) {
        fit(src, count, dim, numThreads);
    }

    void compute_and_set_mean(const uint16_t* src_fp16, size_t count, uint32_t dim, size_t numThreads = 0) {
        fit(src_fp16, count, dim, numThreads);
    }

    // --- Quantization (non-const: automatically fits mean if empty) ---
    void quantize(const float* src, std::byte* dst, size_t count, uint32_t dim, size_t numThreads = 0) {
        if (count == 0 || dim == 0) return;
        if (mean_.empty()) {
            if (count > 1) {
                fit(src, count, dim, numThreads);
            } else {
                mean_.assign(dim, 0.0f);
                mean_fp16_.assign(dim, 0);
            }
        }
        if (mean_.size() != dim) {
            throw std::invalid_argument("EvpQuantizer: mean vector dimension does not match vectors dimension");
        }
        if (count == 1) {
            detail::quantize_single_into(src, dim, non_zeros, dst, mean_ptr());
            return;
        }
        detail::quantize_batch_into(src, count, dim, non_zeros, dst, numThreads, mean_ptr());
    }

    void quantize(const uint16_t* src_fp16, std::byte* dst, size_t count, uint32_t dim, size_t numThreads = 0) {
        if (count == 0 || dim == 0) return;
        if (mean_fp16_.empty()) {
            if (count > 1) {
                fit(src_fp16, count, dim, numThreads);
            } else {
                mean_.assign(dim, 0.0f);
                mean_fp16_.assign(dim, 0);
            }
        }
        if (mean_fp16_.size() != dim) {
            throw std::invalid_argument("EvpQuantizer: mean vector dimension does not match vectors dimension");
        }
        if (count == 1) {
            detail::quantize_single_into(src_fp16, dim, non_zeros, dst, mean_fp16_ptr());
            return;
        }
        detail::quantize_batch_into(src_fp16, count, dim, non_zeros, dst, numThreads, mean_fp16_ptr());
    }

    std::vector<std::byte> quantize(const float* src, size_t count, uint32_t dim, size_t numThreads = 0) {
        if (count == 0 || dim == 0) return {};
        std::vector<std::byte> result(count * 2 * (dim / 8));
        quantize(src, result.data(), count, dim, numThreads);
        return result;
    }

    std::vector<std::byte> quantize(const uint16_t* src_fp16, size_t count, uint32_t dim, size_t numThreads = 0) {
        if (count == 0 || dim == 0) return {};
        std::vector<std::byte> result(count * 2 * (dim / 8));
        quantize(src_fp16, result.data(), count, dim, numThreads);
        return result;
    }

    void quantize(std::span<const float> src, std::span<std::byte> dst, uint32_t dim, size_t numThreads = 0) {
        quantize(src.data(), dst.data(), detail::checked_span_count(src.size(), dst.size(), dim), dim, numThreads);
    }

    void quantize(std::span<const uint16_t> src_fp16, std::span<std::byte> dst, uint32_t dim, size_t numThreads = 0) {
        quantize(src_fp16.data(), dst.data(), detail::checked_span_count(src_fp16.size(), dst.size(), dim), dim, numThreads);
    }

    std::vector<std::byte> quantize(std::span<const float> src, uint32_t dim, size_t numThreads = 0) {
        if (dim == 0) {
            if (src.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src.data(), src.size() / dim, dim, numThreads);
    }

    std::vector<std::byte> quantize(std::span<const uint16_t> src_fp16, uint32_t dim, size_t numThreads = 0) {
        if (dim == 0) {
            if (src_fp16.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src_fp16.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src_fp16.data(), src_fp16.size() / dim, dim, numThreads);
    }

    // --- Quantization (const: requires mean to be already computed or set) ---
    void quantize(const float* src, std::byte* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (mean_.empty()) {
            throw std::invalid_argument("EvpQuantizer: mean must be computed or set before const quantize");
        }
        if (mean_.size() != dim) {
            throw std::invalid_argument("EvpQuantizer: mean vector dimension does not match vectors dimension");
        }
        if (count == 1) {
            detail::quantize_single_into(src, dim, non_zeros, dst, mean_ptr());
            return;
        }
        detail::quantize_batch_into(src, count, dim, non_zeros, dst, numThreads, mean_ptr());
    }

    void quantize(const uint16_t* src_fp16, std::byte* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (mean_.empty()) {
            throw std::invalid_argument("EvpQuantizer: mean must be computed or set before const quantize");
        }
        if (mean_fp16_.size() != dim) {
            throw std::invalid_argument("EvpQuantizer: mean vector dimension does not match vectors dimension");
        }
        if (count == 1) {
            detail::quantize_single_into(src_fp16, dim, non_zeros, dst, mean_fp16_ptr());
            return;
        }
        detail::quantize_batch_into(src_fp16, count, dim, non_zeros, dst, numThreads, mean_fp16_ptr());
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

  private:
    std::vector<float> mean_;
    std::vector<uint16_t> mean_fp16_;
};

// ============================================================================
// Convenience Free Functions (delegate directly to EvpQuantizer)
// ============================================================================

inline std::vector<std::byte> quantize_single(const float* embedding, uint32_t dim, uint32_t non_zeros, const float* mean = nullptr) {
    EvpQuantizer q(non_zeros);
    if (mean != nullptr) {
        q.set_mean(std::span<const float>(mean, dim));
    }
    return q.quantize(embedding, 1, dim);
}

inline std::vector<std::byte> quantize_single(const uint16_t* embedding, uint32_t dim, uint32_t non_zeros, const uint16_t* mean_fp16 = nullptr) {
    EvpQuantizer q(non_zeros);
    if (mean_fp16 != nullptr) {
        q.set_mean_fp16(std::span<const uint16_t>(mean_fp16, dim));
    }
    return q.quantize(embedding, 1, dim);
}

inline std::vector<std::byte> quantize_single(const uint16_t* embedding, uint32_t dim, uint32_t non_zeros, const float* mean) {
    EvpQuantizer q(non_zeros);
    if (mean != nullptr) {
        q.set_mean(std::span<const float>(mean, dim));
    }
    return q.quantize(embedding, 1, dim);
}

inline std::vector<std::byte>
quantize_batch(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0, const float* mean = nullptr) {
    EvpQuantizer q(non_zeros);
    if (mean != nullptr) {
        q.set_mean(std::span<const float>(mean, dim));
    }
    return q.quantize(data, count, dim, numThreads);
}

inline std::vector<std::byte>
quantize_batch(const uint16_t* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0, const uint16_t* mean_fp16 = nullptr) {
    EvpQuantizer q(non_zeros);
    if (mean_fp16 != nullptr) {
        q.set_mean_fp16(std::span<const uint16_t>(mean_fp16, dim));
    }
    return q.quantize(data, count, dim, numThreads);
}

inline std::vector<std::byte>
quantize_batch(const uint16_t* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads, const float* mean) {
    EvpQuantizer q(non_zeros);
    if (mean != nullptr) {
        q.set_mean(std::span<const float>(mean, dim));
    }
    return q.quantize(data, count, dim, numThreads);
}

}  // namespace deglib::quantization::evp
