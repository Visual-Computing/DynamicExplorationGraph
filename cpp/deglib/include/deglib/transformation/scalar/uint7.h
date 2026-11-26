#pragma once

#include "deglib/concurrent.h"
#include "deglib/config.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/transformation.h"
#include "deglib/transformation/scalar/common.h"

#include <algorithm>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace deglib::transformation {

// Affine uniform quantizer restricted to the signed range [0, 127].
class ScalarQuantizerUint7 : public FeatureTransformation {
  public:
    explicit ScalarQuantizerUint7(uint32_t dim) : dim_(dim) {}

    uint32_t tag() const noexcept override { return TAG_SQU7; }
    TransformationType type() const noexcept override { return TransformationType::Quantizer; }
    std::string_view name() const noexcept override { return "ScalarQuantizerUint7"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(int8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        return input_metric.get_distance_kind() == deglib::distances::MetricDistanceKind::InnerProduct ? deglib::distances::Metric::Uint7_InnerProduct
                                                                                                       : deglib::distances::Metric::Uint7_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float min() const noexcept { return min_; }
    float max() const noexcept { return max_; }
    float scale() const noexcept { return scale_; }
    float inv_scale() const noexcept { return inv_scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override { fit(data, -1.0f); }

    void fit(std::span<const float> data, float quantile) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint7: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint7: data size must be multiple of dim");
        if (quantile <= 0.0f) quantile = 1.0f;
        if (quantile >= 1.0f && detail::is_exact_integer_range(data.data(), data.size(), 0.0f, 127.0f)) {
            set_min_max(0.0f, 127.0f);
            return;
        }
        const size_t total_rows = data.size() / dim_;
        const size_t target_rows = std::min(total_rows, std::max<size_t>(1, total_rows / 10));
        std::vector<float> sampled;
        std::span<const float> train_span = data;
        if (target_rows < total_rows) {
            sampled = deglib::transformation::detail::sample_strided_rows<float>(data, dim_, target_rows);
            train_span = sampled;
        }
        const auto [mn, mx] = detail::find_minmax(train_span.data(), train_span.size(), quantile);
        set_min_max(mn, mx);
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, -1.0f); }

    void fit(std::span<const uint16_t> data_fp16, float quantile) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint7: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint7: data size must be multiple of dim");
        if ((quantile <= 0.0f || quantile >= 1.0f) && detail::is_exact_integer_range_fp16(data_fp16, 0.0f, 127.0f)) {
            set_min_max(0.0f, 127.0f);
            return;
        }
        const size_t total_rows = data_fp16.size() / dim_;
        const size_t target_rows = std::min(total_rows, std::max<size_t>(1, total_rows / 10));
        std::vector<uint16_t> sampled;
        std::span<const uint16_t> train_span = data_fp16;
        if (target_rows < total_rows) {
            sampled = deglib::transformation::detail::sample_strided_rows<uint16_t>(data_fp16, dim_, target_rows);
            train_span = sampled;
        }
        auto [mn, mx] = detail::find_minmax_fp16(train_span, quantile);
        set_min_max(mn, mx);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }
    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override { transform_vectors(in_fp16, out, 0); }
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }
    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override { transform_vectors(in_fp16, out, 0); }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t num_threads) const { transform_vectors(in, out, num_threads); }
    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t num_threads) const {
        transform_vectors(in_fp16, out, num_threads);
    }
    void transform_query(std::span<const float> in, std::span<std::byte> out, size_t num_threads) const { transform_vectors(in, out, num_threads); }
    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t num_threads) const {
        transform_vectors(in_fp16, out, num_threads);
    }

    void dequantize(std::span<const std::byte> in, std::span<float> out) const override {
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint7: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerUint7: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerUint7: dequantize output too small");
        const int8_t* src = reinterpret_cast<const int8_t*>(in.data());
        for (uint32_t d = 0; d < dim_; ++d) {
            out[d] = static_cast<float>(src[d]) * inv_scale_ + min_;
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&min_), sizeof(min_));
        os.write(reinterpret_cast<const char*>(&max_), sizeof(max_));
        os.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
        os.write(reinterpret_cast<const char*>(&inv_scale_), sizeof(inv_scale_));
    }

    static std::unique_ptr<ScalarQuantizerUint7> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float min_val = 0.0f;
        float max_val = 1.0f;
        float scale = 127.0f;
        float inv_scale = 1.0f / 127.0f;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&min_val), sizeof(min_val));
        is.read(reinterpret_cast<char*>(&max_val), sizeof(max_val));
        is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        is.read(reinterpret_cast<char*>(&inv_scale), sizeof(inv_scale));
        auto quant = std::make_unique<ScalarQuantizerUint7>(dim);
        quant->min_ = min_val;
        quant->max_ = max_val;
        quant->scale_ = scale;
        quant->inv_scale_ = inv_scale;
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    void set_min_max(float min_val, float max_val) {
        min_ = min_val;
        max_ = max_val;
        const float range = std::max(1e-12f, max_ - min_);
        scale_ = 127.0f / range;
        inv_scale_ = range / 127.0f;
        is_fitted_ = true;
    }
    inline int8_t transform(float x) const {
        float value = (x - min_) * scale_;
        value += 0.5f;
        if (value > 127.0f) return 127;
        if (value < 0.0f) return 0;
        return static_cast<int8_t>(static_cast<int32_t>(value));
    }

    using TransformRowFn = void (*)(const float*, int8_t*, uint32_t, float, float) noexcept;

    static void transform_row_scalar(const float* src, int8_t* dst, uint32_t d, float min_val, float scale) noexcept {
        for (uint32_t j = 0; j < d; ++j) {
            float value = (src[j] - min_val) * scale + 0.5f;
            if (value > 127.0f) {
                dst[j] = 127;
            } else if (value < 0.0f) {
                dst[j] = 0;
            } else {
                dst[j] = static_cast<int8_t>(static_cast<int32_t>(value));
            }
        }
    }

#if defined(DEGLIB_X86)
    DEGLIB_TARGET_AVX2 static void transform_row_avx2(const float* src, int8_t* dst, uint32_t d, float min_val, float scale) noexcept {
        const __m256 v_min = _mm256_set1_ps(min_val);
        const __m256 v_scale = _mm256_set1_ps(scale);
        const __m256 v_half = _mm256_set1_ps(0.5f);
        const __m256 v_zero = _mm256_setzero_ps();
        const __m256 v_127 = _mm256_set1_ps(127.0f);

        uint32_t j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(src + j), v_min), v_scale), v_half);
            v = _mm256_min_ps(_mm256_max_ps(v, v_zero), v_127);
            __m256i i32 = _mm256_cvttps_epi32(v);
            __m128i lo = _mm256_castsi256_si128(i32);
            __m128i hi = _mm256_extracti128_si256(i32, 1);
            __m128i i16 = _mm_packus_epi32(lo, hi);
            __m128i i8 = _mm_packus_epi16(i16, i16);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + j), i8);
        }
        if (j < d) {
            transform_row_scalar(src + j, dst + j, d - j, min_val, scale);
        }
    }

    DEGLIB_TARGET_AVX512 static void transform_row_avx512(const float* src, int8_t* dst, uint32_t d, float min_val, float scale) noexcept {
        const __m512 v_min = _mm512_set1_ps(min_val);
        const __m512 v_scale = _mm512_set1_ps(scale);
        const __m512 v_half = _mm512_set1_ps(0.5f);
        const __m512 v_zero = _mm512_setzero_ps();
        const __m512 v_127 = _mm512_set1_ps(127.0f);

        uint32_t j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 v = _mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(src + j), v_min), v_scale), v_half);
            v = _mm512_min_ps(_mm512_max_ps(v, v_zero), v_127);
            __m512i i32 = _mm512_cvttps_epi32(v);
            __m128i i8 = _mm512_cvtusepi32_epi8(i32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + j), i8);
        }
        if (j < d) {
            transform_row_scalar(src + j, dst + j, d - j, min_val, scale);
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

    inline void transform_row(const float* src, int8_t* dst, uint32_t d) const {
        transform_row_fn_(src, dst, d, min_, scale_);
    }

    inline void transform_row(const uint16_t* src_fp16, int8_t* dst, uint32_t d) const {
        constexpr uint32_t chunk_size = 64;
        float chunk[chunk_size];
        uint32_t i = 0;
        for (; i + chunk_size <= d; i += chunk_size) {
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk, chunk_size);
            transform_row(chunk, dst + i, chunk_size);
        }
        if (i < d) {
            const uint32_t rem = d - i;
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk, rem);
            transform_row(chunk, dst + i, rem);
        }
    }

    template <typename T>
    void transform_internal(const T* src, int8_t* dst, size_t count, uint32_t d, size_t num_threads) const {
        if (count == 0 || d == 0) return;
        deglib::concurrent::parallel_batch_for(0, count, num_threads, [src, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src + i * d, dst + i * d, d);
            }
        });
    }

    template <typename T>
    void transform_vectors(std::span<const T> in, std::span<std::byte> out, size_t num_threads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint7: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint7: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerUint7: transform called before fit()");

        const size_t count = in.size() / dim_;
        const size_t required_bytes = count * output_byte_size();
        if (out.size() < required_bytes) throw std::invalid_argument("ScalarQuantizerUint7: output buffer too small");
        transform_internal(in.data(), reinterpret_cast<int8_t*>(out.data()), count, dim_, num_threads);
    }

    uint32_t dim_ = 0;
    float min_ = 0.0f;
    float max_ = 1.0f;
    float scale_ = 127.0f;
    float inv_scale_ = 1.0f / 127.0f;
    bool is_fitted_ = false;
    TransformRowFn transform_row_fn_ = select_transform_row_fn();
};

inline ScalarQuantizerUint7 make_scalar_quantizer_uint7(std::span<const float> data, uint32_t dim, float quantile = -1.0f) {
    ScalarQuantizerUint7 q(dim);
    q.fit(data, quantile);
    return q;
}

}  // namespace deglib::transformation
