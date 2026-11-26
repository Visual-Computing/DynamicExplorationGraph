#pragma once

#include "deglib/concurrent.h"
#include "deglib/config.h"
#include "deglib/distance/fp16.h"
#include "deglib/transformation.h"
#include "deglib/transformation/scalar/common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace deglib::transformation {

namespace detail {

/**
 * Optimizes the quantile for 4-bit scalar quantizers using a 2-stage approach:
 * 1. Computes the statistical MSE-optimal quantile (q_stat) by minimizing reconstruction error on base vectors.
 * 2. Spans an adaptive grid between q_stat and 1.0f and evaluates pseudo-query recall against exact FP32 distances.
 * 3. Chooses the clipped quantile only if it yields strictly higher recall than 1.0f (safety-first).
 *
 * @param data Array of float values (interleaved vectors)
 * @param total_items Total number of float items (num_vectors * dim)
 * @param dim Dimensionality of vectors
 * @param is_inner_product Whether metric is inner product (true) or L2 (false)
 * @return Optimal quantile in [q_stat, 1.0]
 */
inline float optimize_quantile_uint4(const float* data, size_t total_items, uint32_t dim, bool is_inner_product = true) {
    if (total_items == 0 || dim == 0) return 1.0f;
    const size_t num_vectors = total_items / dim;
    if (num_vectors < 50) return 1.0f;

    // Use up to 50 pseudo-queries and up to 1000 pseudo-base vectors
    const size_t num_queries = std::min<size_t>(50, num_vectors / 4);
    const size_t num_base = std::min<size_t>(1000, num_vectors - num_queries);
    if (num_queries == 0 || num_base == 0) return 1.0f;

    const float* queries = data;
    const float* base = data + num_queries * dim;

    // Step 1: Find statistical MSE-optimal quantile on base vectors
    const float mse_candidates[] = {0.980f, 0.985f, 0.990f, 0.995f, 0.997f, 0.999f, 0.9995f, 1.0f};
    float q_stat = 1.0f;
    double min_mse = std::numeric_limits<double>::infinity();

    std::vector<float> dequant_base(num_base * dim);

    for (float q : mse_candidates) {
        const auto [min_val, max_val] = find_minmax(base, num_base * dim, q);
        const float range = std::max(1e-12f, max_val - min_val);
        const float scale = 15.0f / range;
        const float inv_scale = range / 15.0f;

        double mse = 0.0;
        for (size_t i = 0; i < num_base * dim; ++i) {
            float v = std::clamp(base[i], min_val, max_val);
            float quantized = std::round((v - min_val) * scale);
            float dequant = min_val + quantized * inv_scale;
            float diff = base[i] - dequant;
            mse += static_cast<double>(diff) * diff;
        }
        if (mse < min_mse) {
            min_mse = mse;
            q_stat = q;
        }
    }

    // Step 2: Evaluate exact Top-K (K=20) for each pseudo-query against exact base
    const size_t k = std::min<size_t>(20, num_base);
    std::vector<std::vector<uint32_t>> exact_topk(num_queries, std::vector<uint32_t>(k));

    for (size_t q = 0; q < num_queries; ++q) {
        const float* q_vec = queries + q * dim;
        std::vector<std::pair<float, uint32_t>> dists(num_base);
        for (size_t b = 0; b < num_base; ++b) {
            const float* b_vec = base + b * dim;
            float score = 0.0f;
            if (is_inner_product) {
                for (uint32_t d = 0; d < dim; ++d) score += q_vec[d] * b_vec[d];
            } else {
                for (uint32_t d = 0; d < dim; ++d) {
                    float diff = q_vec[d] - b_vec[d];
                    score += diff * diff;
                }
            }
            dists[b] = {score, static_cast<uint32_t>(b)};
        }

        if (is_inner_product) {
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        for (size_t i = 0; i < k; ++i) {
            exact_topk[q][i] = dists[i].second;
        }
    }

    // Step 3: Grid search between q_stat and 1.0f on pseudo-query recall
    std::vector<float> grid_candidates;
    grid_candidates.push_back(1.0f);
    if (q_stat < 1.0f) {
        const int num_steps = 6;
        for (int step = 0; step < num_steps; ++step) {
            float val = q_stat + static_cast<float>(step) * (1.0f - q_stat) / static_cast<float>(num_steps);
            grid_candidates.push_back(val);
        }
    }

    float best_q = 1.0f;
    size_t max_hits = 0;

    for (float q : grid_candidates) {
        const auto [min_val, max_val] = find_minmax(base, num_base * dim, q);
        const float range = std::max(1e-12f, max_val - min_val);
        const float scale = 15.0f / range;
        const float inv_scale = range / 15.0f;

        for (size_t i = 0; i < num_base * dim; ++i) {
            float v = std::clamp(base[i], min_val, max_val);
            float quantized = std::round((v - min_val) * scale);
            dequant_base[i] = min_val + quantized * inv_scale;
        }

        size_t hits = 0;
        for (size_t q_idx = 0; q_idx < num_queries; ++q_idx) {
            const float* q_vec = queries + q_idx * dim;
            std::vector<std::pair<float, uint32_t>> dists(num_base);
            for (size_t b = 0; b < num_base; ++b) {
                const float* b_vec = dequant_base.data() + b * dim;
                float score = 0.0f;
                if (is_inner_product) {
                    for (uint32_t d = 0; d < dim; ++d) score += q_vec[d] * b_vec[d];
                } else {
                    for (uint32_t d = 0; d < dim; ++d) {
                        float diff = q_vec[d] - b_vec[d];
                        score += diff * diff;
                    }
                }
                dists[b] = {score, static_cast<uint32_t>(b)};
            }

            if (is_inner_product) {
                std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            for (size_t i = 0; i < k; ++i) {
                uint32_t cand = dists[i].second;
                for (size_t j = 0; j < k; ++j) {
                    if (exact_topk[q_idx][j] == cand) {
                        ++hits;
                        break;
                    }
                }
            }
        }

        if (q == 1.0f) {
            max_hits = hits;
            best_q = 1.0f;
        } else if (hits > max_hits) {
            max_hits = hits;
            best_q = q;
        }
    }

    return best_q;
}

}  // namespace detail

/**
 * @brief Uniform 4-bit scalar quantizer (SQ4U / Uint4).
 * Quantizes float/fp16 values into 4-bit integers [0, 15] packed 2 per uint8_t byte:
 *   byte = (dim[2*j] & 0x0F) | ((dim[2*j+1] & 0x0F) << 4)
 */
class ScalarQuantizerUint4 : public FeatureTransformation {
  public:
    explicit ScalarQuantizerUint4(uint32_t dim) : dim_(dim) {}

    uint32_t tag() const noexcept override { return TAG_SQU4; }
    TransformationType type() const noexcept override { return TransformationType::Quantizer; }
    std::string_view name() const noexcept override { return "ScalarQuantizerUint4"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return (dim_ + 1) / 2; }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct) ? deglib::distances::Metric::Uint4_InnerProduct
                                                                             : deglib::distances::Metric::Uint4_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float min_val() const noexcept { return min_val_; }
    float max_val() const noexcept { return max_val_; }
    float dif() const noexcept { return dif_; }
    float scale() const noexcept { return scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override { fit(data, -1.0f); }

    /**
     * Calibrate affine quantizer.
     * @param data Interleaved float training data
     * @param quantile Quantile in (0, 1] for outlier clipping, or < 0 for automatic self-supervised calibration
     */
    void fit(std::span<const float> data, float quantile) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint4: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint4: data size must be multiple of dim");
        if ((quantile <= 0.0f || quantile >= 1.0f) && detail::is_exact_integer_range(data.data(), data.size(), 0.0f, 15.0f)) {
            set_range(0.0f, 15.0f);
            return;
        }
        const size_t total_rows = data.size() / dim_;
        const size_t target_rows = std::min({total_rows, std::max<size_t>(1, total_rows / 10), size_t{10000}});
        std::vector<float> sampled;
        std::span<const float> train_span = data;
        if (target_rows < total_rows) {
            sampled = deglib::transformation::detail::sample_strided_rows<float>(data, dim_, target_rows);
            train_span = sampled;
        }
        float actual_q = quantile;
        if (actual_q <= 0.0f) {
            actual_q = detail::optimize_quantile_uint4(train_span.data(), train_span.size(), dim_, true);
        }
        auto [mi, ma] = detail::find_minmax(train_span.data(), train_span.size(), actual_q);
        set_range(mi, ma);
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, -1.0f); }

    void fit(std::span<const uint16_t> data_fp16, float quantile) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint4: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint4: data size must be multiple of dim");
        if ((quantile <= 0.0f || quantile >= 1.0f) && detail::is_exact_integer_range_fp16(data_fp16, 0.0f, 15.0f)) {
            set_range(0.0f, 15.0f);
            return;
        }
        const size_t total_rows = data_fp16.size() / dim_;
        const size_t target_rows = std::min({total_rows, std::max<size_t>(1, total_rows / 10), size_t{10000}});
        if (quantile > 0.0f) {
            std::vector<uint16_t> sampled_fp16;
            std::span<const uint16_t> train_fp16 = data_fp16;
            if (target_rows < total_rows) {
                sampled_fp16 = deglib::transformation::detail::sample_strided_rows<uint16_t>(data_fp16, dim_, target_rows);
                train_fp16 = sampled_fp16;
            }
            auto [mi, ma] = detail::find_minmax_fp16(train_fp16, quantile);
            set_range(mi, ma);
            return;
        }
        std::vector<float> converted(target_rows * dim_);
        for (size_t s = 0; s < target_rows; ++s) {
            const size_t src_row = (target_rows == total_rows) ? s : (s * total_rows) / target_rows;
            deglib::distances::fp16::fp16_to_floats(data_fp16.data() + src_row * dim_, converted.data() + s * dim_, dim_);
        }
        fit(std::span<const float>(converted), quantile);
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint4: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerUint4: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerUint4: dequantize output too small");
        const uint8_t* src = reinterpret_cast<const uint8_t*>(in.data());
        const float inv_scale = 1.0f / scale_;
        for (uint32_t j = 0; j < dim_; ++j) {
            uint8_t byte_val = src[j / 2];
            uint8_t nibble = (j & 1) ? ((byte_val >> 4) & 0x0F) : (byte_val & 0x0F);
            out[j] = min_val_ + static_cast<float>(nibble) * inv_scale;
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&min_val_), sizeof(min_val_));
        os.write(reinterpret_cast<const char*>(&max_val_), sizeof(max_val_));
        os.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
    }

    static std::unique_ptr<ScalarQuantizerUint4> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float min_v = 0.0f, max_v = 1.0f, scale = 15.0f;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&min_v), sizeof(min_v));
        is.read(reinterpret_cast<char*>(&max_v), sizeof(max_v));
        is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        auto quant = std::make_unique<ScalarQuantizerUint4>(dim);
        quant->set_range(min_v, max_v);
        quant->scale_ = scale;
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    void set_range(float min_v, float max_v) {
        min_val_ = min_v;
        max_val_ = max_v;
        dif_ = max_val_ - min_val_;
        if (dif_ <= 1e-12f) dif_ = 1.0f;
        scale_ = 15.0f / dif_;
        is_fitted_ = true;
    }
    inline uint8_t transform(float x) const {
        float scaled = std::round((x - min_val_) * scale_);
        if (scaled < 0.0f) return 0;
        if (scaled > 15.0f) return 15;
        return static_cast<uint8_t>(scaled);
    }

    using TransformRowFn = void (*)(const float*, uint8_t*, uint32_t, float, float) noexcept;

    static void transform_row_scalar(const float* src, uint8_t* dst, uint32_t d, float min_val, float scale) noexcept {
        auto q_one = [min_val, scale](float x) noexcept -> uint8_t {
            float scaled = std::round((x - min_val) * scale);
            if (scaled < 0.0f) return 0;
            if (scaled > 15.0f) return 15;
            return static_cast<uint8_t>(scaled);
        };
        uint32_t j = 0;
        for (; j + 1 < d; j += 2) {
            dst[j >> 1] = static_cast<uint8_t>(q_one(src[j]) | (q_one(src[j + 1]) << 4));
        }
        if (j < d) {
            dst[j >> 1] = q_one(src[j]);
        }
    }

#if defined(DEGLIB_X86)
    DEGLIB_TARGET_AVX2 static void transform_row_avx2(const float* src, uint8_t* dst, uint32_t d, float min_val, float scale) noexcept {
        const __m256 v_min = _mm256_set1_ps(min_val);
        const __m256 v_scale = _mm256_set1_ps(scale);
        const __m256 v_zero = _mm256_setzero_ps();
        const __m256 v_15 = _mm256_set1_ps(15.0f);
        const __m128i pack_mul = _mm_set1_epi16(0x1001);

        uint32_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 scaled = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(src + j), v_min), v_scale);
            scaled = _mm256_min_ps(_mm256_max_ps(scaled, v_zero), v_15);
            __m256i i32 = _mm256_cvtps_epi32(scaled);
            __m128i lo = _mm256_castsi256_si128(i32);
            __m128i hi = _mm256_extracti128_si256(i32, 1);
            __m128i i16 = _mm_packus_epi32(lo, hi);
            __m128i u8 = _mm_packus_epi16(i16, i16);
            __m128i paired16 = _mm_maddubs_epi16(u8, pack_mul);
            __m128i packed = _mm_packus_epi16(paired16, paired16);
            uint32_t val = static_cast<uint32_t>(_mm_cvtsi128_si32(packed));
            std::memcpy(dst + (j >> 1), &val, 4);
        }
        if (j < d) {
            transform_row_scalar(src + j, dst + (j >> 1), d - j, min_val, scale);
        }
    }

    DEGLIB_TARGET_AVX512 static void transform_row_avx512(const float* src, uint8_t* dst, uint32_t d, float min_val, float scale) noexcept {
        const __m512 v_min = _mm512_set1_ps(min_val);
        const __m512 v_scale = _mm512_set1_ps(scale);
        const __m512 v_zero = _mm512_setzero_ps();
        const __m512 v_15 = _mm512_set1_ps(15.0f);
        const __m128i pack_mul = _mm_set1_epi16(0x1001);

        uint32_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 scaled = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(src + j), v_min), v_scale);
            scaled = _mm512_min_ps(_mm512_max_ps(scaled, v_zero), v_15);
            __m512i i32 = _mm512_cvtps_epi32(scaled);
            __m128i u8 = _mm512_cvtusepi32_epi8(i32);
            __m128i paired16 = _mm_maddubs_epi16(u8, pack_mul);
            __m128i packed = _mm_packus_epi16(paired16, paired16);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + (j >> 1)), packed);
        }
        if (j < d) {
            transform_row_scalar(src + j, dst + (j >> 1), d - j, min_val, scale);
        }
    }
#endif

    static TransformRowFn select_transform_row_fn() noexcept {
#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx512()) return &transform_row_avx512;
        if (deglib::cpu::has_avx2()) return &transform_row_avx2;
#endif
        return &transform_row_scalar;
    }

    inline void transform_row(const float* src, uint8_t* dst, uint32_t d) const {
        transform_row_fn_(src, dst, d, min_val_, scale_);
    }

    inline void transform_row(const uint16_t* src_fp16, uint8_t* dst, uint32_t dim) const {
        constexpr uint32_t CHUNK = 64;
        float chunk_buf[CHUNK];
        for (uint32_t i = 0; i < dim; i += CHUNK) {
            uint32_t cur = std::min(CHUNK, dim - i);
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, cur);
            uint32_t j = 0;
            uint8_t* dst_chunk = dst + (i >> 1);
            for (; j + 1 < cur; j += 2) {
                dst_chunk[j >> 1] = static_cast<uint8_t>(transform(chunk_buf[j]) | (transform(chunk_buf[j + 1]) << 4));
            }
            if (j < cur) {
                dst_chunk[j >> 1] = transform(chunk_buf[j]);
            }
        }
    }

    void quantize_internal(const float* src, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        const size_t out_stride = (d + 1) / 2;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src + i * d, dst + i * out_stride, d);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, d, out_stride, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src + i * d, dst + i * out_stride, d);
            }
        });
    }

    void quantize_internal(const uint16_t* src_fp16, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        const size_t out_stride = (d + 1) / 2;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src_fp16 + i * d, dst + i * out_stride, d);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, d, out_stride, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src_fp16 + i * d, dst + i * out_stride, d);
            }
        });
    }

    void transform_vectors(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint4: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint4: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerUint4: transform called before fit()");
        const size_t count = in.size() / dim_;
        const size_t out_stride = (dim_ + 1) / 2;
        const size_t req_bytes = count * out_stride;
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerUint4: output buffer too small");
        quantize_internal(in.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint4: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint4: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerUint4: transform called before fit()");
        const size_t count = in_fp16.size() / dim_;
        const size_t out_stride = (dim_ + 1) / 2;
        const size_t req_bytes = count * out_stride;
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerUint4: output buffer too small");
        quantize_internal(in_fp16.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    mutable uint32_t dim_ = 0;
    float min_val_ = 0.0f;
    float max_val_ = 1.0f;
    float dif_ = 1.0f;
    float scale_ = 15.0f;
    bool is_fitted_ = false;
    TransformRowFn transform_row_fn_ = select_transform_row_fn();
};

inline ScalarQuantizerUint4 make_scalar_quantizer_uint4(std::span<const float> data, uint32_t dim, float quantile = -1.0f) {
    ScalarQuantizerUint4 q(dim);
    q.fit(data, quantile);
    return q;
}

}  // namespace deglib::transformation
