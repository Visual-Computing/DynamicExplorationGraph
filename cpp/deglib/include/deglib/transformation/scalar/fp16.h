#pragma once

#include "deglib/concurrent.h"
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
 * Symmetric FP16 (IEEE 754 half-precision) quantizer: float32 -> FP16 (uint16_t).
 */
class ScalarQuantizerFP16 : public FeatureTransformation {
  public:
    static constexpr float kFP16Max = 65504.0f;  // largest finite half-precision value

    explicit ScalarQuantizerFP16(uint32_t dim) : dim_(dim) {}
    ScalarQuantizerFP16(uint32_t dim, float abs_max) : dim_(dim) { set_abs_max(abs_max); }

    uint32_t tag() const noexcept override { return TAG_SQ16; }
    TransformationType type() const noexcept override { return TransformationType::Quantizer; }
    std::string_view name() const noexcept override { return "ScalarQuantizerFP16"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(uint16_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct) ? deglib::distances::Metric::FP16_InnerProduct
                                                                             : deglib::distances::Metric::FP16_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float abs_max() const noexcept { return abs_max_; }
    float quantile() const noexcept { return quantile_; }
    float scale() const noexcept { return scale_; }
    float inv_scale() const noexcept { return inv_scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override { fit(data, -1.0f); }

    void fit(std::span<const float> data, float quantile) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerFP16: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerFP16: data size must be multiple of dim");
        if (quantile <= 0.0f) quantile = 1.0f;
        const size_t count = data.size() / dim_;
        quantile_ = quantile;
        float found_max = detail::find_absmax(data.data(), count * dim_, quantile_);
        set_abs_max(found_max);
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, -1.0f); }

    void fit(std::span<const uint16_t> data_fp16, float) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerFP16: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerFP16: data size must be multiple of dim");
        abs_max_ = kFP16Max;
        scale_ = 1.0f;
        inv_scale_ = 1.0f;
        is_fitted_ = true;
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }

    void transform_query(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override { transform_vectors(in_fp16, out); }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override { transform_vectors(in_fp16, out); }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const { transform_vectors(in, out, numThreads); }

    void transform_query(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const { transform_vectors(in, out, numThreads); }

    void dequantize(std::span<const std::byte> in, std::span<float> out) const override {
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerFP16: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerFP16: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerFP16: dequantize output too small");
        const uint16_t* src = reinterpret_cast<const uint16_t*>(in.data());
        deglib::distances::fp16::fp16_to_floats(src, out.data(), dim_);
        if (inv_scale_ != 1.0f) {
            for (uint32_t d = 0; d < dim_; ++d) {
                out[d] *= inv_scale_;
            }
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&abs_max_), sizeof(abs_max_));
        os.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
        os.write(reinterpret_cast<const char*>(&inv_scale_), sizeof(inv_scale_));
    }

    static std::unique_ptr<ScalarQuantizerFP16> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float abs_max = 1.0f, scale = 1.0f, inv_scale = 1.0f;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&abs_max), sizeof(abs_max));
        is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        is.read(reinterpret_cast<char*>(&inv_scale), sizeof(inv_scale));
        auto quant = std::make_unique<ScalarQuantizerFP16>(dim);
        quant->abs_max_ = abs_max;
        quant->scale_ = scale;
        quant->inv_scale_ = inv_scale;
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    void set_abs_max(float val) {
        abs_max_ = (val <= 1e-12f) ? 1.0f : val;
        compute_scale();
        is_fitted_ = true;
    }
    void compute_scale() {
        if (abs_max_ <= kFP16Max) {
            scale_ = 1.0f;
            inv_scale_ = 1.0f;
        } else {
            const int p = static_cast<int>(std::ceil(std::log2(abs_max_)));
            scale_ = std::ldexp(1.0f, -p);
            inv_scale_ = std::ldexp(1.0f, p);
        }
    }

    inline void transform_row(const float* src, uint16_t* dst, uint32_t d) const {
        if (scale_ == 1.0f) {
            deglib::distances::fp16::floats_to_fp16(src, dst, d);
            return;
        }
        constexpr uint32_t CHUNK = 64;
        float buf[CHUNK];
        for (uint32_t i = 0; i < d; i += CHUNK) {
            const uint32_t n = std::min<uint32_t>(CHUNK, d - i);
            for (uint32_t j = 0; j < n; ++j) buf[j] = src[i + j] * scale_;
            deglib::distances::fp16::floats_to_fp16(buf, dst + i, n);
        }
    }

    void quantize_internal(const float* src, uint16_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
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

    void transform_vectors(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerFP16: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerFP16: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerFP16: transform called before fit()");
        const size_t count = in.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(uint16_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerFP16: output buffer too small");
        quantize_internal(in.data(), reinterpret_cast<uint16_t*>(out.data()), count, dim_, numThreads);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerFP16: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerFP16: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerFP16: transform called before fit()");
        const size_t req_bytes = in_fp16.size() * sizeof(uint16_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerFP16: output buffer too small");
        std::memcpy(out.data(), in_fp16.data(), req_bytes);
    }

    mutable uint32_t dim_ = 0;
    float abs_max_ = 1.0f;
    float scale_ = 1.0f;
    float inv_scale_ = 1.0f;
    float quantile_ = 1.0f;
    bool is_fitted_ = false;
};

inline ScalarQuantizerFP16 make_scalar_quantizer_fp16(std::span<const float> data, uint32_t dim, float quantile = -1.0f) {
    ScalarQuantizerFP16 q(dim);
    q.fit(data, quantile);
    return q;
}

}  // namespace deglib::transformation
