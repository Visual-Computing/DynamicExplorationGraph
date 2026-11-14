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
 * Asymmetric Unsigned UINT8 Quantizer [0, 255] (global min / max).
 */
class ScalarQuantizerUint8 : public FeatureTransformation {
  public:
    using output_type = uint8_t;

    ScalarQuantizerUint8() = default;
    explicit ScalarQuantizerUint8(uint32_t dim) : dim_(dim) {}
    ScalarQuantizerUint8(float min_v, float max_v) { set_range(min_v, max_v); }
    ScalarQuantizerUint8(uint32_t dim, float min_v, float max_v) : dim_(dim) { set_range(min_v, max_v); }

    void set_range(float min_v, float max_v) {
        min_val_ = min_v;
        max_val_ = max_v;
        dif_ = max_val_ - min_val_;
        if (dif_ <= 1e-12f) dif_ = 1.0f;
        scale_ = 255.0f / dif_;
        is_fitted_ = true;
    }

    void set_dim(uint32_t d) noexcept { dim_ = d; }

    uint32_t tag() const noexcept override { return TAG_SQU8; }
    std::string_view name() const noexcept override { return "ScalarQuantizerUint8"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(uint8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct) ? deglib::distances::Metric::Uint8_InnerProduct
                                                                             : deglib::distances::Metric::Uint8_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float min_val() const noexcept { return min_val_; }
    float max_val() const noexcept { return max_val_; }
    float dif() const noexcept { return dif_; }
    float scale() const noexcept { return scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override { fit(data, 1.0f); }

    void fit(std::span<const float> data, float quantile) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8: data size must be multiple of dim");
        const size_t count = data.size() / dim_;
        auto [mi, ma] = detail::find_minmax(data.data(), count * dim_, quantile);
        set_range(mi, ma);
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, 1.0f); }

    void fit(std::span<const uint16_t> data_fp16, float quantile) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8: data size must be multiple of dim");
        const size_t total = data_fp16.size();
        std::vector<float> converted(total);
        for (size_t i = 0; i < total; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerUint8: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerUint8: dequantize output too small");
        const uint8_t* src = reinterpret_cast<const uint8_t*>(in.data());
        const float inv_scale = 1.0f / scale_;
        for (uint32_t d = 0; d < dim_; ++d) {
            out[d] = min_val_ + static_cast<float>(src[d]) * inv_scale;
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&min_val_), sizeof(min_val_));
        os.write(reinterpret_cast<const char*>(&max_val_), sizeof(max_val_));
        os.write(reinterpret_cast<const char*>(&scale_), sizeof(scale_));
    }

    static std::unique_ptr<ScalarQuantizerUint8> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float min_v = 0.0f, max_v = 1.0f, scale = 255.0f;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&min_v), sizeof(min_v));
        is.read(reinterpret_cast<char*>(&max_v), sizeof(max_v));
        is.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        auto quant = std::make_unique<ScalarQuantizerUint8>(dim, min_v, max_v);
        quant->scale_ = scale;
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    inline uint8_t transform(float x) const {
        float scaled = std::round((x - min_val_) * scale_);
        if (scaled < 0.0f) return 0;
        if (scaled > 255.0f) return 255;
        return static_cast<uint8_t>(scaled);
    }

    inline void transform_row(const float* src, uint8_t* dst, uint32_t d) const {
        for (uint32_t j = 0; j < d; ++j) {
            dst[j] = transform(src[j]);
        }
    }

    inline void transform_row(const uint16_t* src_fp16, uint8_t* dst, uint32_t dim) const {
        constexpr uint32_t CHUNK = 64;
        float chunk_buf[CHUNK];
        uint32_t i = 0;
        for (; i + CHUNK <= dim; i += CHUNK) {
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, CHUNK);
            transform_row(chunk_buf, dst + i, CHUNK);
        }
        if (i < dim) {
            uint32_t rem = dim - i;
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, rem);
            transform_row(chunk_buf, dst + i, rem);
        }
    }

    void quantize_internal(const float* src, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
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

    void quantize_internal(const uint16_t* src_fp16, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerUint8: transform called before fit()");
        const size_t count = in.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(uint8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerUint8: output buffer too small");
        quantize_internal(in.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerUint8: transform called before fit()");
        const size_t count = in_fp16.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(uint8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerUint8: output buffer too small");
        quantize_internal(in_fp16.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    mutable uint32_t dim_ = 0;
    float min_val_ = 0.0f;
    float max_val_ = 1.0f;
    float dif_ = 1.0f;
    float scale_ = 255.0f;
    bool is_fitted_ = false;
};

/**
 * Asymmetric Unsigned UINT8 Quantizer with per-dimension scaling.
 */
class ScalarQuantizerUint8PerDim : public FeatureTransformation {
  public:
    using output_type = uint8_t;

    ScalarQuantizerUint8PerDim() = default;
    explicit ScalarQuantizerUint8PerDim(uint32_t d) : dim_(d), mins_(d), maxs_(d), scales_(d) {}
    ScalarQuantizerUint8PerDim(const std::vector<float>& min_vals, const std::vector<float>& max_vals) { set_ranges(min_vals, max_vals); }
    ScalarQuantizerUint8PerDim(uint32_t d, const std::vector<float>& min_vals, const std::vector<float>& max_vals) : dim_(d) {
        if (!min_vals.empty() && !max_vals.empty()) set_ranges(min_vals, max_vals);
    }

    void set_ranges(const std::vector<float>& min_vals, const std::vector<float>& max_vals) {
        if (min_vals.size() != max_vals.size()) {
            throw std::invalid_argument("min_vals and max_vals size mismatch");
        }
        dim_ = static_cast<uint32_t>(min_vals.size());
        mins_ = min_vals;
        maxs_ = max_vals;
        scales_.resize(dim_);
        for (uint32_t j = 0; j < dim_; ++j) {
            float diff = maxs_[j] - mins_[j];
            if (diff <= 1e-12f) diff = 1.0f;
            scales_[j] = 255.0f / diff;
        }
        is_fitted_ = true;
    }

    void set_dim(uint32_t d) noexcept { dim_ = d; }

    uint32_t tag() const noexcept override { return TAG_SQUP; }
    std::string_view name() const noexcept override { return "ScalarQuantizerUint8PerDim"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(uint8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct) ? deglib::distances::Metric::Uint8_InnerProduct
                                                                             : deglib::distances::Metric::Uint8_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    const std::vector<float>& mins() const noexcept { return mins_; }
    const std::vector<float>& maxs() const noexcept { return maxs_; }
    const std::vector<float>& scales() const noexcept { return scales_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override { fit(data, 1.0f); }

    void fit(std::span<const float> data, float quantile) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: data size must be multiple of dim");
        const size_t count = data.size() / dim_;
        detail::find_minmax_perdim(mins_, maxs_, data.data(), count, dim_, quantile);
        scales_.resize(dim_);
        for (uint32_t j = 0; j < dim_; ++j) {
            float diff = maxs_[j] - mins_[j];
            if (diff <= 1e-12f) diff = 1.0f;
            scales_[j] = 255.0f / diff;
        }
        is_fitted_ = true;
    }

    void fit(std::span<const uint16_t> data_fp16) override { fit(data_fp16, 1.0f); }

    void fit(std::span<const uint16_t> data_fp16, float quantile) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: data size must be multiple of dim");
        const size_t total = data_fp16.size();
        std::vector<float> converted(total);
        for (size_t i = 0; i < total; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dequantize output too small");
        const uint8_t* src = reinterpret_cast<const uint8_t*>(in.data());
        for (uint32_t d = 0; d < dim_; ++d) {
            out[d] = mins_[d] + static_cast<float>(src[d]) / scales_[d];
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(mins_.data()), dim_ * sizeof(float));
        os.write(reinterpret_cast<const char*>(maxs_.data()), dim_ * sizeof(float));
    }

    static std::unique_ptr<ScalarQuantizerUint8PerDim> deserialize(std::istream& is) {
        uint32_t dim = 0;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        std::vector<float> mins(dim);
        std::vector<float> maxs(dim);
        is.read(reinterpret_cast<char*>(mins.data()), dim * sizeof(float));
        is.read(reinterpret_cast<char*>(maxs.data()), dim * sizeof(float));
        auto quant = std::make_unique<ScalarQuantizerUint8PerDim>(mins, maxs);
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    inline uint8_t transform(float x, uint32_t d) const {
        float scaled = std::round((x - mins_[d]) * scales_[d]);
        if (scaled < 0.0f) return 0;
        if (scaled > 255.0f) return 255;
        return static_cast<uint8_t>(scaled);
    }

    inline void transform_row(const float* src, uint8_t* dst, uint32_t d, uint32_t d_offset = 0) const {
        for (uint32_t j = 0; j < d; ++j) {
            dst[j] = transform(src[j], d_offset + j);
        }
    }

    inline void transform_row(const uint16_t* src_fp16, uint8_t* dst, uint32_t d) const {
        constexpr uint32_t CHUNK = 64;
        float chunk_buf[CHUNK];
        uint32_t i = 0;
        for (; i + CHUNK <= d; i += CHUNK) {
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, CHUNK);
            transform_row(chunk_buf, dst + i, CHUNK, i);
        }
        if (i < d) {
            uint32_t rem = d - i;
            deglib::distances::fp16::fp16_to_floats(src_fp16 + i, chunk_buf, rem);
            transform_row(chunk_buf, dst + i, rem, i);
        }
    }

    void quantize_internal(const float* src, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
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

    void quantize_internal(const uint16_t* src_fp16, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src_fp16 + i * dim_, dst + i * dim_, d);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src_fp16 + i * dim_, dst + i * dim_, d);
            }
        });
    }

    void transform_vectors(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerUint8PerDim: transform called before fit()");
        const size_t count = in.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(uint8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerUint8PerDim: output buffer too small");
        quantize_internal(in.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerUint8PerDim: transform called before fit()");
        const size_t count = in_fp16.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(uint8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerUint8PerDim: output buffer too small");
        quantize_internal(in_fp16.data(), reinterpret_cast<uint8_t*>(out.data()), count, dim_, numThreads);
    }

    mutable uint32_t dim_ = 0;
    std::vector<float> mins_;
    std::vector<float> maxs_;
    std::vector<float> scales_;
    bool is_fitted_ = false;
};

inline ScalarQuantizerUint8 make_scalar_quantizer_uint8(std::span<const float> data, uint32_t dim, float quantile = 1.0f) {
    ScalarQuantizerUint8 q(dim);
    q.fit(data, quantile);
    return q;
}

inline ScalarQuantizerUint8PerDim make_scalar_quantizer_uint8_perdim(std::span<const float> data, uint32_t dim, float quantile = 1.0f) {
    ScalarQuantizerUint8PerDim q(dim);
    q.fit(data, quantile);
    return q;
}

}  // namespace deglib::transformation
