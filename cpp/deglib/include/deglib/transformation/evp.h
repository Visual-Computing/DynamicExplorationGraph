#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/evp_ip.h"
#include "deglib/distance/fp16.h"
#include "deglib/distance/fp16_ip.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace deglib::transformation {

namespace detail {

inline void compute_column_mean(std::span<const float> src, uint32_t dim, std::span<float> out_mean, size_t numThreads = 0) {
    if (src.empty() || dim == 0) return;
    if (src.size() % dim != 0) throw std::invalid_argument("compute_column_mean: src size must be a multiple of dim");
    const size_t count = src.size() / dim;
    if (out_mean.size() < dim) throw std::invalid_argument("compute_column_mean: out_mean span too small");

    const size_t actual_threads = (numThreads == 0) ? std::clamp(static_cast<size_t>(std::thread::hardware_concurrency()), size_t{1}, size_t{64}) : numThreads;
    std::vector<double> thread_sums(actual_threads * dim, 0.0);

    deglib::concurrent::parallel_batch_for(static_cast<size_t>(0), count, actual_threads, [&](size_t begin, size_t end, size_t thread_id) {
        double* my_sum = thread_sums.data() + thread_id * dim;
#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            for (size_t i = begin; i < end; ++i) {
                const float* vec = src.data() + i * dim;
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
            const float* vec = src.data() + i * dim;
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

inline void compute_column_mean(std::span<const uint16_t> src_fp16, uint32_t dim, std::span<float> out_mean, size_t numThreads = 0) {
    if (src_fp16.empty() || dim == 0) return;
    if (src_fp16.size() % dim != 0) throw std::invalid_argument("compute_column_mean: src_fp16 size must be a multiple of dim");
    const size_t count = src_fp16.size() / dim;
    if (out_mean.size() < dim) throw std::invalid_argument("compute_column_mean: out_mean span too small");

    const size_t actual_threads = (numThreads == 0) ? std::clamp(static_cast<size_t>(std::thread::hardware_concurrency()), size_t{1}, size_t{64}) : numThreads;
    std::vector<double> thread_sums(actual_threads * dim, 0.0);

    deglib::concurrent::parallel_batch_for(static_cast<size_t>(0), count, actual_threads, [&](size_t begin, size_t end, size_t thread_id) {
        double* my_sum = thread_sums.data() + thread_id * dim;
#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            for (size_t i = begin; i < end; ++i) {
                const uint16_t* vec = src_fp16.data() + i * dim;
                uint32_t d = 0;
                for (; d + 8 <= dim; d += 8) {
                    __m128i vhp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vec + d));
                    __m256 va = _mm256_cvtph_ps(vhp);
                    __m128 vf1 = _mm256_castps256_ps128(va);
                    __m128 vf2 = _mm256_extractf128_ps(va, 1);
                    __m256d vd1 = _mm256_cvtps_pd(vf1);
                    __m256d vd2 = _mm256_cvtps_pd(vf2);
                    __m256d cur1 = _mm256_loadu_pd(my_sum + d);
                    __m256d cur2 = _mm256_loadu_pd(my_sum + d + 4);
                    _mm256_storeu_pd(my_sum + d, _mm256_add_pd(cur1, vd1));
                    _mm256_storeu_pd(my_sum + d + 4, _mm256_add_pd(cur2, vd2));
                }
                for (; d < dim; ++d) {
                    my_sum[d] += static_cast<double>(deglib::distances::fp16::fp16_to_float(vec[d]));
                }
            }
            return;
        }
#endif
        for (size_t i = begin; i < end; ++i) {
            const uint16_t* vec = src_fp16.data() + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                my_sum[d] += static_cast<double>(deglib::distances::fp16::fp16_to_float(vec[d]));
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

inline void quantize_single_into(std::span<const float> embedding, uint32_t non_zeros, std::span<std::byte> out_bytes, std::span<const float> mean = {}) {
    const uint32_t dim = static_cast<uint32_t>(embedding.size());
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_single: non_zeros must be < dim");
    }
    const size_t mask_bytes = (dim + 7) / 8;
    if (out_bytes.size() < 2 * mask_bytes) {
        throw std::invalid_argument("quantize_single: out_bytes too small");
    }
    if (!mean.empty() && mean.size() != dim) {
        throw std::invalid_argument("quantize_single: mean size must match embedding dim");
    }

    std::byte* ones_dst = out_bytes.data();
    std::byte* negs_dst = out_bytes.data() + mask_bytes;
    std::memset(out_bytes.data(), 0, 2 * mask_bytes);

    std::vector<std::pair<float, uint32_t>> abs_vals(dim);
    if (!mean.empty()) {
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
        const float val = (!mean.empty()) ? (embedding[bit_idx] - mean[bit_idx]) : embedding[bit_idx];
        if (val > 0.0f) {
            ones_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
        } else if (val < 0.0f) {
            negs_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
        }
    }
}

inline void quantize_batch_into(
    std::span<const float> data,
    uint32_t dim,
    uint32_t non_zeros,
    std::span<std::byte> out_bytes,
    size_t numThreads = 0,
    std::span<const float> mean = {}
) {
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_batch: non_zeros must be < dim");
    }
    if (data.empty()) return;
    if (data.size() % dim != 0) {
        throw std::invalid_argument("quantize_batch: data size must be multiple of dim");
    }
    const size_t count = data.size() / dim;
    const size_t mask_bytes = (dim + 7) / 8;
    const size_t bytes_per_evp = 2 * mask_bytes;
    if (out_bytes.size() < count * bytes_per_evp) {
        throw std::invalid_argument("quantize_batch: out_bytes too small");
    }
    if (!mean.empty() && mean.size() != dim) {
        throw std::invalid_argument("quantize_batch: mean size must match dim");
    }

    const float* data_ptr = data.data();
    std::byte* out_ptr = out_bytes.data();
    const float* mean_ptr = mean.empty() ? nullptr : mean.data();

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, numThreads,
        [data_ptr, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, mean_ptr, out_ptr](size_t chunk_id, size_t) {
            const size_t start = chunk_id * chunk_size;
            const size_t end = std::min(start + chunk_size, count);
            std::vector<std::pair<float, uint32_t>> abs_vals(dim);

            for (size_t i = start; i < end; ++i) {
                const float* emb = &data_ptr[i * dim];
                std::byte* dst = &out_ptr[i * bytes_per_evp];
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;
                std::memset(dst, 0, bytes_per_evp);

                if (mean_ptr != nullptr) {
                    for (uint32_t k = 0; k < dim; ++k) {
                        abs_vals[k] = {std::abs(emb[k] - mean_ptr[k]), k};
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
                    const float val = (mean_ptr != nullptr) ? (emb[bit_idx] - mean_ptr[bit_idx]) : emb[bit_idx];
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

inline void
quantize_single_into(std::span<const uint16_t> embedding_fp16, uint32_t non_zeros, std::span<std::byte> out_bytes, std::span<const uint16_t> mean_fp16 = {}) {
    const uint32_t dim = static_cast<uint32_t>(embedding_fp16.size());
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_single: non_zeros must be < dim");
    }
    const size_t mask_bytes = (dim + 7) / 8;
    if (out_bytes.size() < 2 * mask_bytes) {
        throw std::invalid_argument("quantize_single: out_bytes too small");
    }
    if (!mean_fp16.empty() && mean_fp16.size() != dim) {
        throw std::invalid_argument("quantize_single: mean_fp16 size must match embedding_fp16 dim");
    }

    std::byte* ones_dst = out_bytes.data();
    std::byte* negs_dst = out_bytes.data() + mask_bytes;
    std::memset(out_bytes.data(), 0, 2 * mask_bytes);

    std::vector<uint16_t> diff(dim);
    std::vector<std::pair<uint16_t, uint32_t>> abs_vals(dim);

    if (!mean_fp16.empty()) {
        deglib::distances::fp16::sub_and_abs_fp16(embedding_fp16.data(), mean_fp16.data(), dim, diff.data(), abs_vals.data());
    } else {
        for (uint32_t i = 0; i < dim; ++i) {
            uint16_t val = embedding_fp16[i];
            diff[i] = val;
            abs_vals[i] = {static_cast<uint16_t>(val & 0x7FFFu), i};
        }
    }

    std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });

    for (uint32_t j = dim - non_zeros; j < dim; ++j) {
        const uint32_t bit_idx = abs_vals[j].second;
        const uint16_t val = diff[bit_idx];
        const bool is_zero = (val & 0x7FFFu) == 0;
        if (!is_zero) {
            const bool is_neg = (val & 0x8000u) != 0;
            if (!is_neg) {
                ones_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
            } else {
                negs_dst[bit_idx >> 3] |= static_cast<std::byte>(1 << (bit_idx & 7));
            }
        }
    }
}

inline void quantize_batch_into(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    uint32_t non_zeros,
    std::span<std::byte> out_bytes,
    size_t numThreads = 0,
    std::span<const uint16_t> mean_fp16 = {}
) {
    if (data_fp16.empty()) return;
    if (data_fp16.size() % dim != 0) {
        throw std::invalid_argument("quantize_batch: data_fp16 size must be multiple of dim");
    }
    const size_t count = data_fp16.size() / dim;
    if (non_zeros >= dim) {
        throw std::invalid_argument("quantize_batch: non_zeros must be < dim");
    }
    const size_t mask_bytes = (dim + 7) / 8;
    const size_t bytes_per_evp = 2 * mask_bytes;
    if (out_bytes.size() < count * bytes_per_evp) {
        throw std::invalid_argument("quantize_batch: out_bytes too small");
    }
    if (!mean_fp16.empty() && mean_fp16.size() != dim) {
        throw std::invalid_argument("quantize_batch: mean_fp16 size must match dim");
    }

    const uint16_t* data_ptr = data_fp16.data();
    std::byte* out_ptr = out_bytes.data();
    const uint16_t* mean_ptr = mean_fp16.empty() ? nullptr : mean_fp16.data();

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_chunks, numThreads,
        [data_ptr, count, dim, non_zeros, mask_bytes, bytes_per_evp, chunk_size, mean_ptr, out_ptr](size_t chunk_id, size_t) {
            const size_t start = chunk_id * chunk_size;
            const size_t end = std::min(start + chunk_size, count);
            std::vector<uint16_t> diff(dim);
            std::vector<std::pair<uint16_t, uint32_t>> abs_vals(dim);

            for (size_t i = start; i < end; ++i) {
                const uint16_t* emb = &data_ptr[i * dim];
                std::byte* dst = &out_ptr[i * bytes_per_evp];
                std::byte* ones_dst = dst;
                std::byte* negs_dst = dst + mask_bytes;
                std::memset(dst, 0, bytes_per_evp);

                if (mean_ptr != nullptr) {
                    deglib::distances::fp16::sub_and_abs_fp16(emb, mean_ptr, dim, diff.data(), abs_vals.data());
                } else {
                    for (uint32_t k = 0; k < dim; ++k) {
                        uint16_t val = emb[k];
                        diff[k] = val;
                        abs_vals[k] = {static_cast<uint16_t>(val & 0x7FFFu), k};
                    }
                }

                std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(), [](const auto& a, const auto& b) {
                    return a.first != b.first ? a.first < b.first : a.second < b.second;
                });

                for (uint32_t j = dim - non_zeros; j < dim; ++j) {
                    const uint32_t bit_idx = abs_vals[j].second;
                    const uint16_t val = diff[bit_idx];
                    const bool is_zero = (val & 0x7FFFu) == 0;
                    if (!is_zero) {
                        const bool is_neg = (val & 0x8000u) != 0;
                        if (!is_neg) {
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

/**
 * @brief Calibrates the optimal non_zeros parameter for EVP quantization on the given dataset.
 */
inline uint32_t calibrate_best_non_zeros(std::span<const float> src, uint32_t dim, std::span<const float> mean = {}, size_t numThreads = 0) {
    if (src.empty() || dim == 0) return std::max(uint32_t{1}, std::min(dim - 1, uint32_t{16}));
    const size_t count = src.size() / dim;
    if (count < 2) return std::max(uint32_t{1}, std::min(dim - 1, uint32_t{16}));

    const size_t calib_base_count = std::min(count, size_t{2000});
    const size_t available_q = count - calib_base_count;
    const size_t calib_query_count = (available_q >= 50) ? std::min(available_q, size_t{100}) : std::min(calib_base_count, size_t{50});
    std::span<const float> calib_base = src.subspan(0, calib_base_count * dim);
    std::span<const float> calib_queries =
        (available_q >= 50) ? src.subspan(calib_base_count * dim, calib_query_count * dim) : calib_base.subspan(0, calib_query_count * dim);

    // Build candidate list of non_zeros values to evaluate
    std::vector<uint32_t> raw_candidates = {
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        640,
        768,
        1024,
        1352,
        static_cast<uint32_t>(dim * 0.01f),
        static_cast<uint32_t>(dim * 0.02f),
        static_cast<uint32_t>(dim * 0.05f),
        static_cast<uint32_t>(dim * 0.10f),
        static_cast<uint32_t>(dim * 0.25f),
        static_cast<uint32_t>(dim * 0.50f),
        static_cast<uint32_t>(dim * 0.66f)
    };
    std::set<uint32_t> sorted_unique;
    for (uint32_t c : raw_candidates) {
        if (c >= 4 && c < dim) {
            sorted_unique.insert(c);
        }
    }
    if (sorted_unique.empty()) {
        sorted_unique.insert(std::max(uint32_t{1}, std::min(dim - 1, uint32_t{16})));
    }

    const size_t top_k = std::min(calib_base_count, size_t{100});

    // 1. Compute exact FP32 inner product matrix: calib_queries (Q) x calib_base (N)
    std::vector<std::vector<uint32_t>> gt_top_indices(calib_query_count);
    {
        std::vector<std::pair<float, uint32_t>> sim_row(calib_base_count);
        for (size_t q = 0; q < calib_query_count; ++q) {
            const float* q_vec = calib_queries.data() + q * dim;
            for (size_t b = 0; b < calib_base_count; ++b) {
                const float* b_vec = calib_base.data() + b * dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) {
                    dot += q_vec[d] * b_vec[d];
                }
                sim_row[b] = {dot, static_cast<uint32_t>(b)};
            }
            std::nth_element(sim_row.begin(), sim_row.begin() + top_k, sim_row.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            gt_top_indices[q].resize(top_k);
            for (size_t k = 0; k < top_k; ++k) {
                gt_top_indices[q][k] = sim_row[k].second;
            }
            std::sort(gt_top_indices[q].begin(), gt_top_indices[q].end());
        }
    }

    // 2. Evaluate each nz candidate using bitwise popcount inner product
    const size_t evp_stride = 2 * ((dim + 7) / 8);
    std::vector<std::byte> base_evp(calib_base_count * evp_stride);
    std::vector<std::byte> query_evp(calib_query_count * evp_stride);

    uint32_t best_nz = *sorted_unique.begin();
    float best_overlap = -1.0f;

    std::vector<std::pair<float, uint32_t>> evp_row(calib_base_count);
    std::vector<uint32_t> evp_top(top_k);

    for (uint32_t nz : sorted_unique) {
        quantize_batch_into(calib_base, dim, nz, base_evp, numThreads, mean);
        if (calib_queries.data() == calib_base.data()) {
            std::memcpy(query_evp.data(), base_evp.data(), calib_query_count * evp_stride);
        } else {
            quantize_batch_into(calib_queries, dim, nz, query_evp, numThreads, mean);
        }

        size_t total_overlap = 0;
        for (size_t q = 0; q < calib_query_count; ++q) {
            const std::byte* q_ptr = query_evp.data() + q * evp_stride;
            for (size_t b = 0; b < calib_base_count; ++b) {
                const std::byte* b_ptr = base_evp.data() + b * evp_stride;
                const float dist = deglib::distances::evp_ip::EvpInnerProduct::compare(q_ptr, b_ptr, &dim);
                evp_row[b] = {dist, static_cast<uint32_t>(b)};
            }
            std::nth_element(evp_row.begin(), evp_row.begin() + top_k, evp_row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (size_t k = 0; k < top_k; ++k) {
                evp_top[k] = evp_row[k].second;
            }
            std::sort(evp_top.begin(), evp_top.end());

            size_t i = 0, j = 0;
            const auto& gt_row = gt_top_indices[q];
            while (i < top_k && j < top_k) {
                if (evp_top[i] == gt_row[j]) {
                    ++total_overlap;
                    ++i;
                    ++j;
                } else if (evp_top[i] < gt_row[j]) {
                    ++i;
                } else {
                    ++j;
                }
            }
        }

        const float avg_overlap = static_cast<float>(total_overlap) / static_cast<float>(calib_query_count * top_k);
        if (avg_overlap > best_overlap) {
            best_overlap = avg_overlap;
            best_nz = nz;
        }
    }

    return best_nz;
}

/**
 * @brief Calibrates the optimal non_zeros parameter directly from FP16 data.
 */
inline uint32_t calibrate_best_non_zeros(std::span<const uint16_t> src_fp16, uint32_t dim, std::span<const uint16_t> mean_fp16 = {}, size_t numThreads = 0) {
    if (src_fp16.empty() || dim == 0) return std::max(uint32_t{1}, std::min(dim - 1, uint32_t{16}));
    const size_t count = src_fp16.size() / dim;
    if (count < 2) return std::max(uint32_t{1}, std::min(dim - 1, uint32_t{16}));

    const size_t calib_base_count = std::min(count, size_t{2000});
    const size_t available_q = count - calib_base_count;
    const size_t calib_query_count = (available_q >= 50) ? std::min(available_q, size_t{100}) : std::min(calib_base_count, size_t{50});
    std::span<const uint16_t> calib_base = src_fp16.subspan(0, calib_base_count * dim);
    std::span<const uint16_t> calib_queries =
        (available_q >= 50) ? src_fp16.subspan(calib_base_count * dim, calib_query_count * dim) : calib_base.subspan(0, calib_query_count * dim);

    // Build candidate list of non_zeros values to evaluate
    std::vector<uint32_t> raw_candidates = {
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        640,
        768,
        1024,
        1352,
        static_cast<uint32_t>(dim * 0.01f),
        static_cast<uint32_t>(dim * 0.02f),
        static_cast<uint32_t>(dim * 0.05f),
        static_cast<uint32_t>(dim * 0.10f),
        static_cast<uint32_t>(dim * 0.25f),
        static_cast<uint32_t>(dim * 0.50f),
        static_cast<uint32_t>(dim * 0.66f)
    };
    std::set<uint32_t> sorted_unique;
    for (uint32_t c : raw_candidates) {
        if (c >= 4 && c < dim) {
            sorted_unique.insert(c);
        }
    }
    if (sorted_unique.empty()) {
        sorted_unique.insert(std::max(uint32_t{1}, std::min(dim - 1, uint32_t{16})));
    }

    const size_t top_k = std::min(calib_base_count, size_t{100});
    auto dist_func = deglib::distances::fp16_ip::select_dist(dim);
    const size_t dim_size_t = dim;

    // 1. Compute exact FP16 ground truth inner product rankings
    std::vector<std::vector<uint32_t>> gt_top_indices(calib_query_count);
    {
        std::vector<std::pair<float, uint32_t>> sim_row(calib_base_count);
        for (size_t q = 0; q < calib_query_count; ++q) {
            const uint16_t* q_vec = calib_queries.data() + q * dim;
            for (size_t b = 0; b < calib_base_count; ++b) {
                const uint16_t* b_vec = calib_base.data() + b * dim;
                float d = std::visit([&](const auto& d_obj) { return d_obj.compare(q_vec, b_vec, &dim_size_t); }, dist_func);
                sim_row[b] = {d, static_cast<uint32_t>(b)};
            }
            std::nth_element(sim_row.begin(), sim_row.begin() + top_k, sim_row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            gt_top_indices[q].resize(top_k);
            for (size_t k = 0; k < top_k; ++k) {
                gt_top_indices[q][k] = sim_row[k].second;
            }
            std::sort(gt_top_indices[q].begin(), gt_top_indices[q].end());
        }
    }

    // 2. Evaluate each nz candidate using bitwise popcount inner product
    const size_t evp_stride = 2 * ((dim + 7) / 8);
    std::vector<std::byte> base_evp(calib_base_count * evp_stride);
    std::vector<std::byte> query_evp(calib_query_count * evp_stride);

    uint32_t best_nz = *sorted_unique.begin();
    float best_overlap = -1.0f;

    std::vector<std::pair<float, uint32_t>> evp_row(calib_base_count);
    std::vector<uint32_t> evp_top(top_k);

    for (uint32_t nz : sorted_unique) {
        quantize_batch_into(calib_base, dim, nz, base_evp, numThreads, mean_fp16);
        if (calib_queries.data() == calib_base.data()) {
            std::memcpy(query_evp.data(), base_evp.data(), calib_query_count * evp_stride);
        } else {
            quantize_batch_into(calib_queries, dim, nz, query_evp, numThreads, mean_fp16);
        }

        size_t total_overlap = 0;
        for (size_t q = 0; q < calib_query_count; ++q) {
            const std::byte* q_ptr = query_evp.data() + q * evp_stride;
            for (size_t b = 0; b < calib_base_count; ++b) {
                const std::byte* b_ptr = base_evp.data() + b * evp_stride;
                const float dist = deglib::distances::evp_ip::EvpInnerProduct::compare(q_ptr, b_ptr, &dim);
                evp_row[b] = {dist, static_cast<uint32_t>(b)};
            }
            std::nth_element(evp_row.begin(), evp_row.begin() + top_k, evp_row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (size_t k = 0; k < top_k; ++k) {
                evp_top[k] = evp_row[k].second;
            }
            std::sort(evp_top.begin(), evp_top.end());

            size_t i = 0, j = 0;
            const auto& gt_row = gt_top_indices[q];
            while (i < top_k && j < top_k) {
                if (evp_top[i] == gt_row[j]) {
                    ++total_overlap;
                    ++i;
                    ++j;
                } else if (evp_top[i] < gt_row[j]) {
                    ++i;
                } else {
                    ++j;
                }
            }
        }

        const float avg_overlap = static_cast<float>(total_overlap) / static_cast<float>(calib_query_count * top_k);
        if (avg_overlap > best_overlap) {
            best_overlap = avg_overlap;
            best_nz = nz;
        }
    }

    return best_nz;
}

}  // namespace detail

/**
 * @brief Equi-Voronoi Polytopes (EVP) Quantizer.
 *
 * Quantizes high-dimensional floating-point vectors into compact bit-packed representations
 * (dim / 8 bytes for positive entries, dim / 8 bytes for negative entries) suitable for
 * fast bitwise popcount symmetric inner product search (Metric::EVP_InnerProduct).
 *
 * @param non_zeros The number of extreme dimensions (largest magnitude components) retained
 *                  as +/-1 bits. If set to 0 (default), EvpQuantizer automatically calibrates
 *                  the optimal non_zeros value during fit() by measuring top-100 neighborhood
 *                  preservation on a data sample.
 */
class EvpQuantizer : public FeatureTransformation {
  public:
    using output_type = std::byte;

    /**
     * @brief Constructs an EvpQuantizer.
     *
     * @param dim Vector dimension (must match vector size).
     * @param non_zeros (Optional) Number of extreme dimensions retained as +/-1 bits. Defaults to 0 (auto-calibrated during fit()).
     */
    EvpQuantizer() = default;
    explicit EvpQuantizer(uint32_t dim, uint32_t non_zeros = 0) : dim_(dim), non_zeros_(non_zeros) {}

    uint32_t tag() const noexcept override { return 0x51505645; }  // 'EVPQ'
    std::string_view name() const noexcept override { return "EvpQuantizer"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return 2 * ((dim_ + 7) / 8); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric) const override { return deglib::distances::Metric::EVP_InnerProduct; }

    uint32_t dim() const noexcept { return dim_; }
    uint32_t non_zeros() const noexcept { return non_zeros_; }
    bool is_fitted() const noexcept override { return non_zeros_ > 0 && has_mean(); }
    bool has_mean() const noexcept { return !mean_.empty(); }

    std::span<const float> mean() const noexcept { return mean_; }
    std::span<const uint16_t> mean_fp16() const noexcept { return mean_fp16_; }

    void fit(std::span<const float> data) override { fit(data, 0); }

    void fit(std::span<const float> data, size_t numThreads) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("EvpQuantizer: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("EvpQuantizer: data size must be multiple of dim");
        const size_t count = data.size() / dim_;

        mean_.resize(dim_);
        detail::compute_column_mean(data, dim_, mean_, numThreads);
        mean_fp16_.resize(dim_);
        deglib::distances::fp16::floats_to_fp16(mean_.data(), mean_fp16_.data(), dim_);

        if (non_zeros_ == 0) {
            non_zeros_ =
                (count > 1) ? detail::calibrate_best_non_zeros(data, dim_, mean_, numThreads) : std::max(uint32_t{1}, std::min(dim_ - 1, uint32_t{16}));
        }
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, 0); }

    void fit(std::span<const uint16_t> data_fp16, size_t numThreads) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("EvpQuantizer: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("EvpQuantizer: data size must be multiple of dim");
        const size_t count = data_fp16.size() / dim_;

        mean_.resize(dim_);
        detail::compute_column_mean(data_fp16, dim_, mean_, numThreads);
        mean_fp16_.resize(dim_);
        deglib::distances::fp16::floats_to_fp16(mean_.data(), mean_fp16_.data(), dim_);

        if (non_zeros_ == 0) {
            non_zeros_ = (count > 1) ? detail::calibrate_best_non_zeros(data_fp16, dim_, mean_fp16_, numThreads)
                                     : std::max(uint32_t{1}, std::min(dim_ - 1, uint32_t{16}));
        }
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override { transform_vectors(in_fp16, out, 0); }

    void transform_query(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override { transform_vectors(in_fp16, out, 0); }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const { transform_vectors(in, out, numThreads); }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const { transform_vectors(in, out, numThreads); }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const { transform_vectors(in_fp16, out, numThreads); }

    void dequantize(std::span<const std::byte> in, std::span<float> out) const override {
        if (dim_ == 0) throw std::invalid_argument("EvpQuantizer: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("EvpQuantizer: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("EvpQuantizer: dequantize output too small");

        const size_t mask_bytes = (dim_ + 7) / 8;
        const std::byte* ones_src = in.data();
        const std::byte* negs_src = in.data() + mask_bytes;
        const float* mean_ptr = mean_.empty() ? nullptr : mean_.data();

        for (uint32_t d = 0; d < dim_; ++d) {
            const size_t byte_idx = d >> 3;
            const uint8_t bit_mask = static_cast<uint8_t>(1 << (d & 7));
            const bool is_pos = (static_cast<uint8_t>(ones_src[byte_idx]) & bit_mask) != 0;
            const bool is_neg = (static_cast<uint8_t>(negs_src[byte_idx]) & bit_mask) != 0;

            float val = 0.0f;
            if (is_pos) {
                val = 1.0f;
            } else if (is_neg) {
                val = -1.0f;
            }

            out[d] = mean_ptr != nullptr ? (mean_ptr[d] + val) : val;
        }
    }

  private:
    void transform_vectors(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("EvpQuantizer: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("EvpQuantizer: input size must be multiple of dim");
        if (!is_fitted()) throw std::logic_error("EvpQuantizer: transform called before fit() (non_zeros is 0 or mean is not set)");
        if (non_zeros_ >= dim_) throw std::invalid_argument(std::format("EvpQuantizer: non_zeros ({}) must be < dim ({})", non_zeros_, dim_));

        const size_t count = in.size() / dim_;
        const size_t evp_stride = output_byte_size();
        if (out.size() < count * evp_stride) throw std::invalid_argument("EvpQuantizer: output buffer too small");

        if (count == 1) {
            detail::quantize_single_into(in, non_zeros_, out, mean_);
            return;
        }
        detail::quantize_batch_into(in, dim_, non_zeros_, out, numThreads, mean_);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("EvpQuantizer: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("EvpQuantizer: input size must be multiple of dim");
        if (!is_fitted()) throw std::logic_error("EvpQuantizer: transform called before fit() (non_zeros is 0 or mean is not set)");
        if (non_zeros_ >= dim_) throw std::invalid_argument(std::format("EvpQuantizer: non_zeros ({}) must be < dim ({})", non_zeros_, dim_));

        const size_t count = in_fp16.size() / dim_;
        const size_t evp_stride = output_byte_size();
        if (out.size() < count * evp_stride) throw std::invalid_argument("EvpQuantizer: output buffer too small");

        if (count == 1) {
            detail::quantize_single_into(in_fp16, non_zeros_, out, mean_fp16_);
            return;
        }
        detail::quantize_batch_into(in_fp16, dim_, non_zeros_, out, numThreads, mean_fp16_);
    }

  public:
    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&non_zeros_), sizeof(non_zeros_));
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
        auto q = std::make_unique<EvpQuantizer>(dim, non_zeros);
        if (mean_len > 0) {
            q->mean_.resize(mean_len);
            is.read(reinterpret_cast<char*>(q->mean_.data()), mean_len * sizeof(float));
            q->mean_fp16_.resize(mean_len);
            deglib::distances::fp16::floats_to_fp16(q->mean_.data(), q->mean_fp16_.data(), mean_len);
        }
        return q;
    }

  private:
    uint32_t dim_ = 0;
    uint32_t non_zeros_ = 0;
    std::vector<float> mean_;
    std::vector<uint16_t> mean_fp16_;
};

inline EvpQuantizer make_evp_quantizer(std::span<const float> data, uint32_t dim, uint32_t non_zeros = 0, size_t numThreads = 0) {
    EvpQuantizer q(dim, non_zeros);
    q.fit(data, numThreads);
    return q;
}

}  // namespace deglib::transformation
