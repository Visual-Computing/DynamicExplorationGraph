#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/transformation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace deglib::transformation {

namespace detail {

inline std::pair<float, float> find_minmax(const float* data, size_t total_items, float drop_ratio = 0.0f) {
    if (total_items == 0) return {0.0f, 0.0f};

    if (drop_ratio <= 0.0f) {
        float min_val = data[0];
        float max_val = data[0];
        for (size_t i = 1; i < total_items; ++i) {
            float v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        return {min_val, max_val};
    }

    size_t top = static_cast<size_t>(static_cast<double>(total_items) * drop_ratio) + 1;
    std::priority_queue<float, std::vector<float>, std::less<float>> max_heap;
    std::priority_queue<float, std::vector<float>, std::greater<float>> min_heap;

    for (size_t i = 0; i < total_items; ++i) {
        float x = data[i];
        if (min_heap.size() < top) {
            min_heap.push(x);
        } else if (x > min_heap.top()) {
            min_heap.pop();
            min_heap.push(x);
        }

        if (max_heap.size() < top) {
            max_heap.push(x);
        } else if (x < max_heap.top()) {
            max_heap.pop();
            max_heap.push(x);
        }
    }

    return {max_heap.top(), min_heap.top()};
}

inline float find_absmax(const float* data, size_t total_items, float drop_ratio = 0.0f) {
    if (total_items == 0) return 0.0f;

    if (drop_ratio <= 0.0f) {
        float abs_max = 0.0f;
        for (size_t i = 0; i < total_items; ++i) {
            float v = std::abs(data[i]);
            if (v > abs_max) abs_max = v;
        }
        return abs_max;
    }

    size_t top = static_cast<size_t>(static_cast<double>(total_items) * drop_ratio) + 1;
    std::priority_queue<float, std::vector<float>, std::greater<float>> min_heap;

    for (size_t i = 0; i < total_items; ++i) {
        float x = std::abs(data[i]);
        if (min_heap.size() < top) {
            min_heap.push(x);
        } else if (x > min_heap.top()) {
            min_heap.pop();
            min_heap.push(x);
        }
    }

    return min_heap.top();
}

inline void find_absmax_perdim(std::vector<float>& abs_maxs, const float* data, size_t n, uint32_t dim, float drop_ratio = 0.0f) {
    abs_maxs.assign(dim, 0.0f);
    if (n == 0 || dim == 0) return;

    if (drop_ratio <= 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            const float* row = data + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                float v = std::abs(row[d]);
                if (v > abs_maxs[d]) abs_maxs[d] = v;
            }
        }
        return;
    }

    size_t top = static_cast<size_t>(static_cast<double>(n) * drop_ratio) + 1;
    std::vector<std::priority_queue<float, std::vector<float>, std::greater<float>>> min_heaps(dim);

    for (size_t i = 0; i < n; ++i) {
        const float* row = data + i * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            float x = std::abs(row[d]);
            auto& mi_h = min_heaps[d];
            if (mi_h.size() < top) {
                mi_h.push(x);
            } else if (x > mi_h.top()) {
                mi_h.pop();
                mi_h.push(x);
            }
        }
    }

    for (uint32_t d = 0; d < dim; ++d) {
        abs_maxs[d] = min_heaps[d].top();
    }
}

inline void find_minmax_perdim(std::vector<float>& mins, std::vector<float>& maxs, const float* data, size_t n, uint32_t dim, float drop_ratio = 0.0f) {
    mins.assign(dim, std::numeric_limits<float>::max());
    maxs.assign(dim, std::numeric_limits<float>::lowest());

    if (n == 0 || dim == 0) return;

    if (drop_ratio <= 0.0f) {
        for (size_t i = 0; i < n; ++i) {
            const float* row = data + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                float v = row[d];
                if (v < mins[d]) mins[d] = v;
                if (v > maxs[d]) maxs[d] = v;
            }
        }
        return;
    }

    size_t top = static_cast<size_t>(static_cast<double>(n) * drop_ratio) + 1;
    std::vector<std::priority_queue<float, std::vector<float>, std::less<float>>> max_heaps(dim);
    std::vector<std::priority_queue<float, std::vector<float>, std::greater<float>>> min_heaps(dim);

    for (size_t i = 0; i < n; ++i) {
        const float* row = data + i * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            float x = row[d];
            auto& mi_h = min_heaps[d];
            if (mi_h.size() < top) {
                mi_h.push(x);
            } else if (x > mi_h.top()) {
                mi_h.pop();
                mi_h.push(x);
            }

            auto& ma_h = max_heaps[d];
            if (ma_h.size() < top) {
                ma_h.push(x);
            } else if (x < ma_h.top()) {
                ma_h.pop();
                ma_h.push(x);
            }
        }
    }

    for (uint32_t d = 0; d < dim; ++d) {
        mins[d] = max_heaps[d].top();
        maxs[d] = min_heaps[d].top();
    }
}

}  // namespace detail

/**
 * Symmetric Signed INT8 Quantizer [-127, 127] (global scale).
 */
class ScalarQuantizerInt8 : public FeatureTransformation {
  public:
    using output_type = int8_t;

    ScalarQuantizerInt8() = default;
    explicit ScalarQuantizerInt8(float abs_max_val) { set_abs_max(abs_max_val); }
    ScalarQuantizerInt8(uint32_t dim, float abs_max_val) : dim_(dim) { set_abs_max(abs_max_val); }

    void set_abs_max(float val) {
        abs_max_ = (val <= 1e-12f) ? 1.0f : val;
        scale_ = 127.0f / abs_max_;
        is_fitted_ = true;
    }

    void set_dim(uint32_t d) noexcept { dim_ = d; }

    uint32_t tag() const noexcept override { return 0x38495153; }  // 'SQI8'
    std::string_view name() const noexcept override { return "ScalarQuantizerInt8"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(int8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct)
            ? deglib::distances::Metric::Int8_InnerProduct
            : deglib::distances::Metric::Int8_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float abs_max() const noexcept { return abs_max_; }
    float scale() const noexcept { return scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override {
        fit(data, 0.0f);
    }

    void fit(std::span<const float> data, float drop_ratio) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8: data size must be multiple of dim");
        const size_t count = data.size() / dim_;
        float found_max = detail::find_absmax(data.data(), count * dim_, drop_ratio);
        set_abs_max(found_max);
    }

    void fit(std::span<const uint16_t> data_fp16) override {
        fit(data_fp16, 0.0f);
    }

    void fit(std::span<const uint16_t> data_fp16, float drop_ratio) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8: data size must be multiple of dim");
        const size_t total = data_fp16.size();
        if (drop_ratio <= 0.0f) {
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
            fit(std::span<const float>(converted), drop_ratio);
        }
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void dequantize(std::span<const std::byte> in, std::span<float> out) const {
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
        auto quant = std::make_unique<ScalarQuantizerInt8>(dim, abs_max);
        quant->scale_ = scale;
        quant->is_fitted_ = true;
        return quant;
    }

  private:
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

    uint32_t tag() const noexcept override { return 0x38555153; }  // 'SQU8'
    std::string_view name() const noexcept override { return "ScalarQuantizerUint8"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(uint8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct)
            ? deglib::distances::Metric::Uint8_InnerProduct
            : deglib::distances::Metric::Uint8_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    float min_val() const noexcept { return min_val_; }
    float max_val() const noexcept { return max_val_; }
    float dif() const noexcept { return dif_; }
    float scale() const noexcept { return scale_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override {
        fit(data, 0.0f);
    }

    void fit(std::span<const float> data, float drop_ratio) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8: data size must be multiple of dim");
        const size_t count = data.size() / dim_;
        auto [mi, ma] = detail::find_minmax(data.data(), count * dim_, drop_ratio);
        set_range(mi, ma);
    }

    void fit(std::span<const uint16_t> data_fp16) override {
        fit(data_fp16, 0.0f);
    }

    void fit(std::span<const uint16_t> data_fp16, float drop_ratio) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8: data size must be multiple of dim");
        const size_t total = data_fp16.size();
        std::vector<float> converted(total);
        for (size_t i = 0; i < total; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
        }
        fit(std::span<const float>(converted), drop_ratio);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void dequantize(std::span<const std::byte> in, std::span<float> out) const {
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
 * Symmetric Signed INT8 Quantizer with per-dimension scaling.
 */
class ScalarQuantizerInt8PerDim : public FeatureTransformation {
  public:
    using output_type = int8_t;

    ScalarQuantizerInt8PerDim() = default;
    explicit ScalarQuantizerInt8PerDim(uint32_t d) : dim_(d), abs_maxs_(d), scales_(d) {}
    ScalarQuantizerInt8PerDim(const std::vector<float>& max_vals) { set_abs_maxs(max_vals); }
    ScalarQuantizerInt8PerDim(uint32_t d, const std::vector<float>& max_vals) : dim_(d) {
        if (!max_vals.empty()) set_abs_maxs(max_vals);
    }

    void set_abs_maxs(const std::vector<float>& max_vals) {
        dim_ = static_cast<uint32_t>(max_vals.size());
        abs_maxs_ = max_vals;
        scales_.resize(dim_);
        for (uint32_t j = 0; j < dim_; ++j) {
            float v = abs_maxs_[j];
            if (v <= 1e-12f) v = 1.0f;
            scales_[j] = 127.0f / v;
        }
        is_fitted_ = true;
    }

    void set_dim(uint32_t d) noexcept { dim_ = d; }

    uint32_t tag() const noexcept override { return 0x50385153; }  // 'SQ8P'
    std::string_view name() const noexcept override { return "ScalarQuantizerInt8PerDim"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(int8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct)
            ? deglib::distances::Metric::Int8_InnerProduct
            : deglib::distances::Metric::Int8_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    const std::vector<float>& abs_maxs() const noexcept { return abs_maxs_; }
    const std::vector<float>& scales() const noexcept { return scales_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override {
        fit(data, 0.0f);
    }

    void fit(std::span<const float> data, float drop_ratio) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: data size must be multiple of dim");
        const size_t count = data.size() / dim_;
        detail::find_absmax_perdim(abs_maxs_, data.data(), count, dim_, drop_ratio);
        scales_.resize(dim_);
        for (uint32_t j = 0; j < dim_; ++j) {
            float v = abs_maxs_[j];
            if (v <= 1e-12f) v = 1.0f;
            scales_[j] = 127.0f / v;
        }
        is_fitted_ = true;
    }

    void fit(std::span<const uint16_t> data_fp16) override {
        fit(data_fp16, 0.0f);
    }

    void fit(std::span<const uint16_t> data_fp16, float drop_ratio) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: data size must be multiple of dim");
        const size_t total = data_fp16.size();
        std::vector<float> converted(total);
        for (size_t i = 0; i < total; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
        }
        fit(std::span<const float>(converted), drop_ratio);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void dequantize(std::span<const std::byte> in, std::span<float> out) const {
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: dim must be > 0");
        if (in.size() < output_byte_size()) throw std::invalid_argument("ScalarQuantizerInt8PerDim: dequantize input too small");
        if (out.size() < dim_) throw std::invalid_argument("ScalarQuantizerInt8PerDim: dequantize output too small");
        const int8_t* src = reinterpret_cast<const int8_t*>(in.data());
        for (uint32_t d = 0; d < dim_; ++d) {
            out[d] = static_cast<float>(src[d]) / scales_[d];
        }
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(abs_maxs_.data()), dim_ * sizeof(float));
    }

    static std::unique_ptr<ScalarQuantizerInt8PerDim> deserialize(std::istream& is) {
        uint32_t dim = 0;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        std::vector<float> abs_maxs(dim);
        is.read(reinterpret_cast<char*>(abs_maxs.data()), dim * sizeof(float));
        auto quant = std::make_unique<ScalarQuantizerInt8PerDim>(abs_maxs);
        quant->is_fitted_ = true;
        return quant;
    }

  private:
    inline int8_t transform(float x, uint32_t d) const {
        float scaled = std::round(x * scales_[d]);
        if (scaled < -127.0f) return -127;
        if (scaled > 127.0f) return 127;
        return static_cast<int8_t>(scaled);
    }

    inline void transform_row(const float* src, int8_t* dst, uint32_t d, uint32_t d_offset = 0) const {
        for (uint32_t j = 0; j < d; ++j) {
            dst[j] = transform(src[j], d_offset + j);
        }
    }

    inline void transform_row(const uint16_t* src_fp16, int8_t* dst, uint32_t d) const {
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
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerInt8PerDim: transform called before fit()");
        const size_t count = in.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(int8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerInt8PerDim: output buffer too small");
        quantize_internal(in.data(), reinterpret_cast<int8_t*>(out.data()), count, dim_, numThreads);
    }

    void transform_vectors(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        if (in_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: dim must be > 0");
        if (in_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerInt8PerDim: input size must be multiple of dim");
        if (!is_fitted_) throw std::logic_error("ScalarQuantizerInt8PerDim: transform called before fit()");
        const size_t count = in_fp16.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(int8_t);
        if (out.size() < req_bytes) throw std::invalid_argument("ScalarQuantizerInt8PerDim: output buffer too small");
        quantize_internal(in_fp16.data(), reinterpret_cast<int8_t*>(out.data()), count, dim_, numThreads);
    }

    mutable uint32_t dim_ = 0;
    std::vector<float> abs_maxs_;
    std::vector<float> scales_;
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
    ScalarQuantizerUint8PerDim(const std::vector<float>& min_vals, const std::vector<float>& max_vals) {
        set_ranges(min_vals, max_vals);
    }
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

    uint32_t tag() const noexcept override { return 0x50555153; }  // 'SQUP'
    std::string_view name() const noexcept override { return "ScalarQuantizerUint8PerDim"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return dim_ * sizeof(uint8_t); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        auto kind = input_metric.get_distance_kind();
        return (kind == deglib::distances::MetricDistanceKind::InnerProduct)
            ? deglib::distances::Metric::Uint8_InnerProduct
            : deglib::distances::Metric::Uint8_L2;
    }

    uint32_t dim() const noexcept { return dim_; }
    const std::vector<float>& mins() const noexcept { return mins_; }
    const std::vector<float>& maxs() const noexcept { return maxs_; }
    const std::vector<float>& scales() const noexcept { return scales_; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    void fit(std::span<const float> data) override {
        fit(data, 0.0f);
    }

    void fit(std::span<const float> data, float drop_ratio) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: data size must be multiple of dim");
        const size_t count = data.size() / dim_;
        detail::find_minmax_perdim(mins_, maxs_, data.data(), count, dim_, drop_ratio);
        scales_.resize(dim_);
        for (uint32_t j = 0; j < dim_; ++j) {
            float diff = maxs_[j] - mins_[j];
            if (diff <= 1e-12f) diff = 1.0f;
            scales_[j] = 255.0f / diff;
        }
        is_fitted_ = true;
    }

    void fit(std::span<const uint16_t> data_fp16) override {
        fit(data_fp16, 0.0f);
    }

    void fit(std::span<const uint16_t> data_fp16, float drop_ratio) {
        if (data_fp16.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: dim must be > 0 prior to fit()");
        if (data_fp16.size() % dim_ != 0) throw std::invalid_argument("ScalarQuantizerUint8PerDim: data size must be multiple of dim");
        const size_t total = data_fp16.size();
        std::vector<float> converted(total);
        for (size_t i = 0; i < total; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
        }
        fit(std::span<const float>(converted), drop_ratio);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const override {
        transform_vectors(in_fp16, out, 0);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in_fp16, out, numThreads);
    }

    void dequantize(std::span<const std::byte> in, std::span<float> out) const {
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

inline ScalarQuantizerInt8 make_scalar_quantizer_int8(std::span<const float> data, uint32_t dim, float drop_ratio = 0.0f) {
    ScalarQuantizerInt8 q(dim, 1.0f);
    q.fit(data, drop_ratio);
    return q;
}

inline ScalarQuantizerUint8 make_scalar_quantizer_uint8(std::span<const float> data, uint32_t dim, float drop_ratio = 0.0f) {
    ScalarQuantizerUint8 q(dim);
    q.fit(data, drop_ratio);
    return q;
}

inline ScalarQuantizerInt8PerDim make_scalar_quantizer_int8_perdim(std::span<const float> data, uint32_t dim, float drop_ratio = 0.0f) {
    ScalarQuantizerInt8PerDim q(dim);
    q.fit(data, drop_ratio);
    return q;
}

inline ScalarQuantizerUint8PerDim make_scalar_quantizer_uint8_perdim(std::span<const float> data, uint32_t dim, float drop_ratio = 0.0f) {
    ScalarQuantizerUint8PerDim q(dim);
    q.fit(data, drop_ratio);
    return q;
}

}  // namespace deglib::transformation
