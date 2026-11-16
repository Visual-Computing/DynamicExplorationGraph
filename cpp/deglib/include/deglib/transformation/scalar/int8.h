#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/transformation.h"
#include "deglib/transformation/scalar/common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace deglib::transformation {

/**
 * Symmetric Signed INT8 Quantizer [-127, 127] (global scale).
 */
class ScalarQuantizerInt8 : public FeatureTransformation {
  public:
    explicit ScalarQuantizerInt8(uint32_t dim) : dim_(dim) {}

    uint32_t tag() const noexcept override { return TAG_SQI8; }
    std::string_view name() const noexcept override { return "ScalarQuantizerInt8"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(int8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct) ? deglib::distances::Metric::Int8_InnerProduct
                                                                             : deglib::distances::Metric::Int8_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float abs_max() const noexcept { return abs_max_; }
    float scale() const noexcept { return scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override { fit(data, -1.0f); }

    void fit(std::span<const float> data, float quantile) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8: data size must be multiple of dim");
        if (quantile <= 0.0f) quantile = 1.0f;
        const size_t count = data.size() / dim_;
        float found_max = detail::find_absmax(data.data(), count * dim_, quantile);
        set_abs_max(found_max);
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, -1.0f); }

    void fit(std::span<const uint16_t> data_fp16, float quantile) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8: data size must be multiple of dim");
        if (quantile <= 0.0f) quantile = 1.0f;
        const size_t total = data_fp16.size();
        if (quantile >= 1.0f) {
            uint16_t max_half = 0;
            for (size_t i = 0; i < total; ++i) {
                uint16_t mag = data_fp16[i] & 0x7FFFu;
                if (mag > max_half) max_half = mag;
            }
            float val = deglib::distances::fp16::fp16_to_float(max_half);
            set_abs_max(val);
        } else {
            std::vector<float> converted(total);
            for (size_t i = 0; i < total; ++i) {
                converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
            }
            fit(std::span<const float>(converted), quantile);
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerInt8: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerInt8: dequantize output too small");
        const int8_t* src = reinterpret_cast<const int8_t*>(in.data());
        const float inv_scale = 1.0f / scale_;
        for (uint32_t d = 0; d < dim_; ++d) {
            out[d] = static_cast<float>(src[d]) * inv_scale;
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&abs_max_), sizeof(abs_max_));
        os.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
    }

    static std::unique_ptr<ScalarQuantizerInt8> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float abs_max = 1.0f;
        float scale = 127.0f;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&abs_max), sizeof(abs_max));
        is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        auto quant = std::make_unique<ScalarQuantizerInt8>(dim);
        quant->abs_max_ = abs_max;
        quant->scale_ = scale;
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    void set_abs_max(float val) {
        abs_max_ = (val <= 1e-12f) ? 1.0f : val;
        scale_ = 127.0f / abs_max_;
        is_fitted_ = true;
    }
    inline int8_t transform(float x) const {
        float scaled = std::round(x * scale_);
        if (scaled < -127.0f) return -127;
        if (scaled > 127.0f) return 127;
        return static_cast<int8_t>(scaled);
    }

    inline void transform_row(const float* src, int8_t* dst, uint32_t d) const {
        for (uint32_t j = 0; j < d; ++j) {
            dst[j] = transform(src[j]);
        }
    }

    inline void transform_row(const uint16_t* src_fp16, int8_t* dst, uint32_t d) const {
        constexpr uint32_t CHUNK = 64;
        float chunk_buf[CHUNK];
        uint32_t i = 0;
        for (; i + CHUNK <= d; i += CHUNK) {
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, CHUNK);
            transform_row(chunk_buf, dst + i, CHUNK);
        }
        if (i < d) {
            uint32_t rem = d - i;
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, rem);
            transform_row(chunk_buf, dst + i, rem);
        }
    }

    void quantize_internal(const float* src, int8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src + i * d, dst + i * d, d);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src + i * d, dst + i * d, d);
            }
        });
    }

    void quantize_internal(const uint16_t* src_fp16, int8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src_fp16 + i * d, dst + i * d, d);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src_fp16 + i * d, dst + i * d, d);
            }
        });
    }

    void transform_vectors(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerInt8: transform called before fit()");
        const size_t count = in.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(int8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerInt8: output buffer too small");
        quantize_internal(in.data(), reinterpret_cast<int8_t*>(out.data()), count, dim_, numThreads);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerInt8: transform called before fit()");
        const size_t count = in_fp16.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(int8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerInt8: output buffer too small");
        quantize_internal(in_fp16.data(), reinterpret_cast<int8_t*>(out.data()), count, dim_, numThreads);
    }

    mutable uint32_t dim_ = 0;
    float abs_max_ = 1.0f;
    float scale_ = 127.0f;
    bool is_fitted_ = false;
};

inline ScalarQuantizerInt8 make_scalar_quantizer_int8(std::span<const float> data, uint32_t dim, float quantile = -1.0f) {
    ScalarQuantizerInt8 q(dim);
    q.fit(data, quantile);
    return q;
}

}  // namespace deglib::transformation
