#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace deglib::transformation {

namespace detail {

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

inline void quantize_single_into(const uint16_t* embedding_fp16, uint32_t dim, uint32_t non_zeros, std::byte* out_bytes, const uint16_t* mean_fp16 = nullptr) {
    std::vector<float> f_emb(dim);
    deglib::distances::fp16::fp16_to_floats(embedding_fp16, f_emb.data(), dim);
    if (mean_fp16 != nullptr) {
        std::vector<float> f_mean(dim);
        deglib::distances::fp16::fp16_to_floats(mean_fp16, f_mean.data(), dim);
        quantize_single_into(f_emb.data(), dim, non_zeros, out_bytes, f_mean.data());
    } else {
        quantize_single_into(f_emb.data(), dim, non_zeros, out_bytes, nullptr);
    }
}

inline void quantize_batch_into(
    const uint16_t* data_fp16,
    size_t count,
    uint32_t dim,
    uint32_t non_zeros,
    std::byte* out_bytes,
    size_t numThreads = 0,
    const uint16_t* mean_fp16 = nullptr
) {
    if (count == 0) return;
    const size_t total = count * dim;
    std::vector<float> f_data(total);
    deglib::distances::fp16::fp16_to_floats(data_fp16, f_data.data(), total);
    if (mean_fp16 != nullptr) {
        std::vector<float> f_mean(dim);
        deglib::distances::fp16::fp16_to_floats(mean_fp16, f_mean.data(), dim);
        quantize_batch_into(f_data.data(), count, dim, non_zeros, out_bytes, numThreads, f_mean.data());
    } else {
        quantize_batch_into(f_data.data(), count, dim, non_zeros, out_bytes, numThreads, nullptr);
    }
}

}  // namespace detail

class FeatureTransformation;

/**
 * Enhanced Vector Packing (EVP) Quantizer.
 */
class EvpQuantizer : public FeatureTransformation {
  public:
    using output_type = std::byte;
    uint32_t dim = 0;
    uint32_t non_zeros = 16;

    EvpQuantizer() = default;
    explicit EvpQuantizer(uint32_t non_zeros) : non_zeros(non_zeros) {}
    EvpQuantizer(uint32_t dim, uint32_t non_zeros) : dim(dim), non_zeros(non_zeros) {}
    EvpQuantizer(uint32_t non_zeros, const std::vector<float>& m) : dim(static_cast<uint32_t>(m.size())), non_zeros(non_zeros) { set_mean(m); }
    EvpQuantizer(uint32_t dim, uint32_t non_zeros, const std::vector<float>& m) : dim(dim), non_zeros(non_zeros) {
        if (!m.empty()) set_mean(m);
    }

    uint32_t tag() const noexcept { return 0x51505645; }  // 'EVPQ'
    std::string_view name() const noexcept { return "EvpQuantizer"; }
    uint32_t input_dim() const noexcept { return dim; }
    uint32_t output_dim() const noexcept { return dim; }
    size_t input_bytes() const noexcept { return dim * sizeof(float); }
    size_t output_bytes() const noexcept { return 2 * (dim / 8); }

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
        if (dim == 0) dim = static_cast<uint32_t>(mean_.size());
    }

    void set_mean_fp16(std::span<const uint16_t> m) {
        mean_fp16_.assign(m.begin(), m.end());
        mean_.resize(mean_fp16_.size());
        deglib::distances::fp16::fp16_to_floats(mean_fp16_.data(), mean_.data(), mean_fp16_.size());
        if (dim == 0) dim = static_cast<uint32_t>(mean_fp16_.size());
    }

    void fit(const float* src, size_t count, uint32_t d, size_t numThreads = 0) {
        if (count == 0 || d == 0) return;
        dim = d;
        mean_.resize(dim);
        detail::compute_column_mean(src, count, dim, mean_.data(), numThreads);
        mean_fp16_.resize(dim);
        deglib::distances::fp16::floats_to_fp16(mean_.data(), mean_fp16_.data(), dim);
    }

    void fit(const float* src, size_t count, size_t numThreads = 0) {
        fit(src, count, dim, numThreads);
    }

    void fit(const uint16_t* src_fp16, size_t count, uint32_t d, size_t numThreads = 0) {
        if (count == 0 || d == 0) return;
        dim = d;
        mean_.resize(dim);
        deglib::distances::fp16::compute_column_mean(src_fp16, count, dim, mean_.data(), numThreads);
        mean_fp16_.resize(dim);
        deglib::distances::fp16::floats_to_fp16(mean_.data(), mean_fp16_.data(), dim);
    }

    void fit(const uint16_t* src_fp16, size_t count, size_t numThreads = 0) {
        fit(src_fp16, count, dim, numThreads);
    }

    void compute_and_set_mean(const float* src, size_t count, uint32_t d, size_t numThreads = 0) {
        fit(src, count, d, numThreads);
    }

    void compute_and_set_mean(const uint16_t* src_fp16, size_t count, uint32_t d, size_t numThreads = 0) {
        fit(src_fp16, count, d, numThreads);
    }

    void quantize(const float* src, std::byte* dst, size_t count, uint32_t d, size_t numThreads = 0) {
        if (count == 0 || d == 0) return;
        if (!mean_.empty() && mean_.size() != d) {
            throw std::invalid_argument(std::format("EvpQuantizer: mean dimension ({}) does not match vector dimension ({})", mean_.size(), d));
        }
        if (non_zeros >= d) {
            throw std::invalid_argument(std::format("EvpQuantizer: non_zeros ({}) must be < dim ({})", non_zeros, d));
        }
        if (mean_.empty()) {
            if (count > 1) {
                fit(src, count, d, numThreads);
            } else {
                mean_.assign(d, 0.0f);
                mean_fp16_.assign(d, 0);
            }
        }
        if (count == 1) {
            detail::quantize_single_into(src, d, non_zeros, dst, mean_ptr());
            return;
        }
        detail::quantize_batch_into(src, count, d, non_zeros, dst, numThreads, mean_ptr());
    }

    void quantize(const float* src, std::byte* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (!mean_.empty() && mean_.size() != d) {
            throw std::invalid_argument(std::format("EvpQuantizer: mean dimension ({}) does not match vector dimension ({})", mean_.size(), d));
        }
        if (non_zeros >= d) {
            throw std::invalid_argument(std::format("EvpQuantizer: non_zeros ({}) must be < dim ({})", non_zeros, d));
        }
        if (count == 1) {
            detail::quantize_single_into(src, d, non_zeros, dst, mean_ptr());
            return;
        }
        detail::quantize_batch_into(src, count, d, non_zeros, dst, numThreads, mean_ptr());
    }

    std::vector<std::byte> quantize(const float* src, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<std::byte> result(count * 2 * (d / 8));
        quantize(src, result.data(), count, d, numThreads);
        return result;
    }

    void quantize(const uint16_t* src_fp16, std::byte* dst, size_t count, uint32_t d, size_t numThreads = 0) {
        if (count == 0 || d == 0) return;
        if (!mean_.empty() && mean_.size() != d) {
            throw std::invalid_argument(std::format("EvpQuantizer: mean dimension ({}) does not match vector dimension ({})", mean_.size(), d));
        }
        if (non_zeros >= d) {
            throw std::invalid_argument(std::format("EvpQuantizer: non_zeros ({}) must be < dim ({})", non_zeros, d));
        }
        if (mean_.empty()) {
            if (count > 1) {
                fit(src_fp16, count, d, numThreads);
            } else {
                mean_.assign(d, 0.0f);
                mean_fp16_.assign(d, 0);
            }
        }
        if (count == 1) {
            detail::quantize_single_into(src_fp16, d, non_zeros, dst, mean_fp16_ptr());
            return;
        }
        detail::quantize_batch_into(src_fp16, count, d, non_zeros, dst, numThreads, mean_fp16_ptr());
    }

    void quantize(const uint16_t* src_fp16, std::byte* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (!mean_.empty() && mean_.size() != d) {
            throw std::invalid_argument(std::format("EvpQuantizer: mean dimension ({}) does not match vector dimension ({})", mean_.size(), d));
        }
        if (non_zeros >= d) {
            throw std::invalid_argument(std::format("EvpQuantizer: non_zeros ({}) must be < dim ({})", non_zeros, d));
        }
        if (count == 1) {
            detail::quantize_single_into(src_fp16, d, non_zeros, dst, mean_fp16_ptr());
            return;
        }
        detail::quantize_batch_into(src_fp16, count, d, non_zeros, dst, numThreads, mean_fp16_ptr());
    }

    std::vector<std::byte> quantize(const uint16_t* src_fp16, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<std::byte> result(count * 2 * (d / 8));
        quantize(src_fp16, result.data(), count, d, numThreads);
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

    void transform_database(const float* in, std::byte* out, size_t count) const {
        quantize(in, out, count, dim);
    }

    void transform_query(const float* in, std::byte* out, size_t count) const {
        quantize(in, out, count, dim);
    }

    void serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        os.write(reinterpret_cast<const char*>(&non_zeros), sizeof(non_zeros));
        uint32_t mean_len = static_cast<uint32_t>(mean_.size());
        os.write(reinterpret_cast<const char*>(&mean_len), sizeof(mean_len));
        if (mean_len > 0) {
            os.write(reinterpret_cast<const char*>(mean_.data()), mean_len * sizeof(float));
        }
    }

    static std::unique_ptr<EvpQuantizer> deserialize(std::istream& is) {
        uint32_t dim = 0;
        uint32_t non_zeros = 0;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&non_zeros), sizeof(non_zeros));
        uint32_t mean_len = 0;
        is.read(reinterpret_cast<char*>(&mean_len), sizeof(mean_len));
        std::vector<float> mean(mean_len);
        if (mean_len > 0) {
            is.read(reinterpret_cast<char*>(mean.data()), mean_len * sizeof(float));
        }
        return std::make_unique<EvpQuantizer>(dim, non_zeros, std::move(mean));
    }

  private:
    std::vector<float> mean_;
    std::vector<uint16_t> mean_fp16_;
};

inline std::vector<std::byte> quantize_single(const float* embedding, uint32_t dim, uint32_t non_zeros, const float* mean = nullptr) {
    EvpQuantizer q(dim, non_zeros);
    if (mean != nullptr) q.set_mean(std::span<const float>(mean, dim));
    return q.quantize(embedding, 1, dim);
}

inline std::vector<std::byte>
quantize_batch(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0, const float* mean = nullptr) {
    EvpQuantizer q(dim, non_zeros);
    if (mean != nullptr) q.set_mean(std::span<const float>(mean, dim));
    return q.quantize(data, count, dim, numThreads);
}

inline std::vector<std::byte>
quantize_batch(const uint16_t* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0, const uint16_t* mean_fp16 = nullptr) {
    EvpQuantizer q(dim, non_zeros);
    if (mean_fp16 != nullptr) q.set_mean_fp16(std::span<const uint16_t>(mean_fp16, dim));
    return q.quantize(data, count, dim, numThreads);
}

inline EvpQuantizer make_evp_quantizer(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0) {
    EvpQuantizer q(dim, non_zeros);
    q.fit(data, count, dim, numThreads);
    return q;
}

}  // namespace deglib::transformation
