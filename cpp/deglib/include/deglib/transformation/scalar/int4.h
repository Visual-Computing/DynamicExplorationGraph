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

/**
 * @brief Symmetric 4-bit scalar quantizer (SQI4 / Int4).
 * Quantizes float/fp16 values into signed 4-bit integers [-7, 7] (or [-8, 7]) packed 2 per uint8_t byte:
 *   byte = (nibble0 & 0x0F) | ((nibble1 & 0x0F) << 4)
 */
class ScalarQuantizerInt4 : public FeatureTransformation {
  public:
    explicit ScalarQuantizerInt4(uint32_t dim) : dim_(dim) {}

    uint32_t tag() const noexcept override { return TAG_SQI4; }
    TransformationType type() const noexcept override { return TransformationType::Quantizer; }
    std::string_view name() const noexcept override { return "ScalarQuantizerInt4"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return (dim_ + 1) / 2; }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct) ? deglib::distances::Metric::Int4_InnerProduct
                                                                             : deglib::distances::Metric::Int4_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float abs_max() const noexcept { return abs_max_; }
    float scale() const noexcept { return scale_; }
    float inv_scale() const noexcept { return inv_scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override { fit(data, -1.0f); }

    void fit(std::span<const float> data, float quantile) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt4: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt4: data size must be multiple of dim");
        if (quantile <= 0.0f) quantile = 1.0f;
        if (quantile >= 1.0f && detail::is_exact_integer_range(data.data(), data.size(), -8.0f, 7.0f)) {
            set_abs_max(8.0f);
            return;
        }
        float am = detail::find_absmax(data.data(), data.size(), quantile);
        set_abs_max(am);
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, -1.0f); }

    void fit(std::span<const uint16_t> data_fp16, float quantile) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt4: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt4: data size must be multiple of dim");
        if ((quantile <= 0.0f || quantile >= 1.0f) && detail::is_exact_integer_range_fp16(data_fp16, -8.0f, 7.0f)) {
            set_abs_max(8.0f);
            return;
        }
        set_abs_max(detail::find_absmax_fp16(data_fp16, quantile));
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt4: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerInt4: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerInt4: dequantize output too small");
        const uint8_t* src = reinterpret_cast<const uint8_t*>(in.data());
        for (uint32_t j = 0; j < dim_; ++j) {
            uint8_t byte_val = src[j / 2];
            uint8_t raw_nibble = (j & 1) ? ((byte_val >> 4) & 0x0F) : (byte_val & 0x0F);
            int8_t s = static_cast<int8_t>(raw_nibble);
            if (s & 0x08) s |= 0xF0;
            out[j] = static_cast<float>(s) * inv_scale_;
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&abs_max_), sizeof(abs_max_));
        os.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
        os.write(reinterpret_cast<const char*>(&inv_scale_), sizeof(inv_scale_));
    }

    static std::unique_ptr<ScalarQuantizerInt4> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float abs_m = 1.0f, scale = 7.0f, inv_scale = 1.0f / 7.0f;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&abs_m), sizeof(abs_m));
        is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        is.read(reinterpret_cast<char*>(&inv_scale), sizeof(inv_scale));
        auto quant = std::make_unique<ScalarQuantizerInt4>(dim);
        quant->abs_max_ = abs_m;
        quant->scale_ = scale;
        quant->inv_scale_ = inv_scale;
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    void set_abs_max(float abs_m) {
        abs_max_ = abs_m;
        if (abs_max_ <= 1e-12f) abs_max_ = 1.0f;
        scale_ = 8.0f / abs_max_;
        inv_scale_ = abs_max_ / 8.0f;
        is_fitted_ = true;
    }
    inline uint8_t transform(float x) const {
        float scaled = std::round(x * scale_);
        int32_t v = static_cast<int32_t>(scaled);
        if (v < -8) v = -8;
        if (v > 7) v = 7;
        return static_cast<uint8_t>(v & 0x0F);
    }

    using TransformRowFn = void (*)(const float*, uint8_t*, uint32_t, float) noexcept;

    static void transform_row_scalar(const float* src, uint8_t* dst, uint32_t d, float scale) noexcept {
        auto q_one = [scale](float x) noexcept -> uint8_t {
            float scaled = std::round(x * scale);
            int32_t v = static_cast<int32_t>(scaled);
            if (v < -8) v = -8;
            if (v > 7) v = 7;
            return static_cast<uint8_t>(v & 0x0F);
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
    DEGLIB_TARGET_AVX2 static void transform_row_avx2(const float* src, uint8_t* dst, uint32_t d, float scale) noexcept {
        const __m256 v_scale = _mm256_set1_ps(scale);
        const __m256 v_min = _mm256_set1_ps(-8.0f);
        const __m256 v_max = _mm256_set1_ps(7.0f);
        const __m256i v_mask = _mm256_set1_epi32(0x0F);
        const __m128i pack_mul = _mm_set1_epi16(0x1001);

        uint32_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(src + j), v_scale);
            scaled = _mm256_min_ps(_mm256_max_ps(scaled, v_min), v_max);
            __m256i i32 = _mm256_and_si256(_mm256_cvtps_epi32(scaled), v_mask);
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
            transform_row_scalar(src + j, dst + (j >> 1), d - j, scale);
        }
    }

    DEGLIB_TARGET_AVX512 static void transform_row_avx512(const float* src, uint8_t* dst, uint32_t d, float scale) noexcept {
        const __m512 v_scale = _mm512_set1_ps(scale);
        const __m512 v_min = _mm512_set1_ps(-8.0f);
        const __m512 v_max = _mm512_set1_ps(7.0f);
        const __m512i v_mask = _mm512_set1_epi32(0x0F);
        const __m128i pack_mul = _mm_set1_epi16(0x1001);

        uint32_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 scaled = _mm512_mul_ps(_mm512_loadu_ps(src + j), v_scale);
            scaled = _mm512_min_ps(_mm512_max_ps(scaled, v_min), v_max);
            __m512i i32 = _mm512_and_si512(_mm512_cvtps_epi32(scaled), v_mask);
            __m128i u8 = _mm512_cvtusepi32_epi8(i32);
            __m128i paired16 = _mm_maddubs_epi16(u8, pack_mul);
            __m128i packed = _mm_packus_epi16(paired16, paired16);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + (j >> 1)), packed);
        }
        if (j < d) {
            transform_row_scalar(src + j, dst + (j >> 1), d - j, scale);
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
        transform_row_fn_(src, dst, d, scale_);
    }

    inline void transform_row(const uint16_t* src_fp16, uint8_t* dst, uint32_t dim) const {
        constexpr uint32_t CHUNK = 64;
        float chunk_buf[CHUNK];
        const size_t num_bytes = (dim + 1) / 2;
        std::memset(dst, 0, num_bytes);
        for (uint32_t i = 0; i < dim; i += CHUNK) {
            uint32_t cur = std::min(CHUNK, dim - i);
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, cur);
            for (uint32_t j = 0; j < cur; ++j) {
                uint32_t glob_idx = i + j;
                uint8_t y = transform(chunk_buf[j]);
                if (glob_idx & 1) {
                    dst[glob_idx / 2] |= (y << 4);
                } else {
                    dst[glob_idx / 2] |= (y & 0x0F);
                }
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt4: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt4: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerInt4: transform called before fit()");
        const size_t count = in.size() / dim_;
        const size_t out_stride = (dim_ + 1) / 2;
        const size_t req_bytes = count * out_stride;
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerInt4: output buffer too small");
        quantize_internal(in.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt4: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt4: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerInt4: transform called before fit()");
        const size_t count = in_fp16.size() / dim_;
        const size_t out_stride = (dim_ + 1) / 2;
        const size_t req_bytes = count * out_stride;
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerInt4: output buffer too small");
        quantize_internal(in_fp16.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    mutable uint32_t dim_ = 0;
    float abs_max_ = 1.0f;
    float scale_ = 7.0f;
    float inv_scale_ = 1.0f / 7.0f;
    bool is_fitted_ = false;
    TransformRowFn transform_row_fn_ = select_transform_row_fn();
};

inline ScalarQuantizerInt4 make_scalar_quantizer_int4(std::span<const float> data, uint32_t dim, float quantile = -1.0f) {
    ScalarQuantizerInt4 q(dim);
    q.fit(data, quantile);
    return q;
}

}  // namespace deglib::transformation
