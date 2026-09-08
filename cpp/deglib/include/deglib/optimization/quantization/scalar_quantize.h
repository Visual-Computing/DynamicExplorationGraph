#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deglib::quantization::scalar {

// Validates span-based quantize inputs, returns the vector count.
inline size_t checked_span_count(size_t src_size, size_t dst_size, uint32_t dim) {
    if (dim == 0) {
        if (src_size == 0 && dst_size == 0) return 0;
        throw std::invalid_argument("quantize: dim must be > 0");
    }
    if (src_size % dim != 0) {
        throw std::invalid_argument("quantize: src span size must be a multiple of dim");
    }
    const size_t count = src_size / dim;
    if (dst_size < count * dim) {
        throw std::invalid_argument("quantize: dst span too small for src span and dim");
    }
    return count;
}

// ============================================================================
// Calibration Helpers (internal percentile / min / max scanning)
// ============================================================================

inline float limit_range(float x, float low = 0.0f, float high = 1.0f) {
    if (x < low) return low;
    if (x > high) return high;
    return x;
}

inline float limit_range_sym(float x, float bound = 1.0f) {
    if (x < -bound) return -bound;
    if (x > bound) return bound;
    return x;
}

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

// ============================================================================
// Object-Oriented State-Aware Quantizers
// ============================================================================

/**
 * Symmetric Signed INT8 Quantizer [-127, 127] (global scale).
 * Supports FP32 and FP16 (uint16_t) inputs.
 */
class ScalarQuantizerInt8 {
public:
    using output_type = int8_t;
    float abs_max = 1.0f;
    float scale = 127.0f;
    bool is_fitted = false;

    ScalarQuantizerInt8() = default;
    explicit ScalarQuantizerInt8(float abs_max_val) {
        set_abs_max(abs_max_val);
    }

    void set_abs_max(float val) {
        abs_max = (val <= 1e-12f) ? 1.0f : val;
        scale = 127.0f / abs_max;
        is_fitted = true;
    }

    void fit(const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f) {
        if (count == 0 || dim == 0) return;
        float found_max = find_absmax(data, count * dim, drop_ratio);
        set_abs_max(found_max);
    }

    void fit(const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f) {
        if (count == 0 || dim == 0) return;
        if (drop_ratio <= 0.0f) {
            uint16_t max_half = 0;
            for (size_t i = 0; i < count * dim; ++i) {
                uint16_t mag = data_fp16[i] & 0x7FFFu;
                if (mag > max_half) max_half = mag;
            }
            float val = deglib::distances::fp16::fp16_to_float(max_half);
            set_abs_max(val);
        } else {
            std::vector<float> converted(count * dim);
            for (size_t i = 0; i < count * dim; ++i) {
                converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
            }
            fit(converted.data(), count, dim, drop_ratio);
        }
    }

    void quantize(const float* src, int8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src + i * dim, dst + i * dim, dim);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src + i * dim, dst + i * dim, dim);
            }
        });
    }

    void quantize(const uint16_t* src_fp16, int8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, dim);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, dim);
            }
        });
    }

    std::vector<int8_t> quantize(const float* src, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<int8_t> result(count * dim);
        quantize(src, result.data(), count, dim, numThreads);
        return result;
    }

    std::vector<int8_t> quantize(const uint16_t* src_fp16, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<int8_t> result(count * dim);
        quantize(src_fp16, result.data(), count, dim, numThreads);
        return result;
    }

    void quantize(std::span<const float> src, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src.data(), dst.data(), checked_span_count(src.size(), dst.size(), dim), dim, numThreads);
    }

    void quantize(std::span<const uint16_t> src_fp16, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src_fp16.data(), dst.data(), checked_span_count(src_fp16.size(), dst.size(), dim), dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const float> src, uint32_t dim, size_t numThreads = 0) const {
        if (dim == 0) {
            if (src.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src.data(), src.size() / dim, dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const uint16_t> src_fp16, uint32_t dim, size_t numThreads = 0) const {
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
    inline int8_t transform(float x) const {
        float scaled = std::round(x * scale);
        if (scaled < -127.0f) return -127;
        if (scaled > 127.0f) return 127;
        return static_cast<int8_t>(scaled);
    }

    inline int8_t transform(uint16_t fp16_bits) const {
        return transform(deglib::distances::fp16::fp16_to_float(fp16_bits));
    }

    inline void transform_row(const float* src, int8_t* dst, uint32_t dim) const {
        for (uint32_t j = 0; j < dim; ++j) {
            dst[j] = transform(src[j]);
        }
    }

    inline void transform_row(const uint16_t* src_fp16, int8_t* dst, uint32_t dim) const {
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
};

/**
 * Per-dimension Symmetric Signed INT8 Quantizer [-127, 127].
 * Calibrates abs_max per dimension. Supports FP32 and FP16 (uint16_t) inputs.
 */
class ScalarQuantizerInt8PerDim {
public:
    using output_type = int8_t;
    uint32_t dim = 0;
    std::vector<float> abs_maxs;
    std::vector<float> scales;
    bool is_fitted = false;

    ScalarQuantizerInt8PerDim() = default;
    explicit ScalarQuantizerInt8PerDim(uint32_t d) : dim(d), abs_maxs(d), scales(d) {}

    void set_abs_maxs(const std::vector<float>& max_vals) {
        dim = static_cast<uint32_t>(max_vals.size());
        abs_maxs = max_vals;
        scales.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            float m = (abs_maxs[j] <= 1e-12f) ? 1.0f : abs_maxs[j];
            abs_maxs[j] = m;
            scales[j] = 127.0f / m;
        }
        is_fitted = true;
    }

    void fit(const float* data, size_t count, uint32_t d, float drop_ratio = 0.0f) {
        if (count == 0 || d == 0) return;
        dim = d;
        find_absmax_perdim(abs_maxs, data, count, dim, drop_ratio);
        scales.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            float m = (abs_maxs[j] <= 1e-12f) ? 1.0f : abs_maxs[j];
            abs_maxs[j] = m;
            scales[j] = 127.0f / m;
        }
        is_fitted = true;
    }

    void fit(const uint16_t* data_fp16, size_t count, uint32_t d, float drop_ratio = 0.0f) {
        if (count == 0 || d == 0) return;
        std::vector<float> converted(count * d);
        for (size_t i = 0; i < count * d; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
        }
        fit(converted.data(), count, d, drop_ratio);
    }

    void quantize(const float* src, int8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerInt8PerDim: expected {}, got {}", dim, d));
        }
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

    void quantize(const uint16_t* src_fp16, int8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerInt8PerDim: expected {}, got {}", dim, d));
        }
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, d);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, d);
            }
        });
    }

    std::vector<int8_t> quantize(const float* src, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<int8_t> result(count * d);
        quantize(src, result.data(), count, d, numThreads);
        return result;
    }

    std::vector<int8_t> quantize(const uint16_t* src_fp16, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<int8_t> result(count * d);
        quantize(src_fp16, result.data(), count, d, numThreads);
        return result;
    }

    void quantize(std::span<const float> src, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src.data(), dst.data(), checked_span_count(src.size(), dst.size(), dim), dim, numThreads);
    }

    void quantize(std::span<const uint16_t> src_fp16, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src_fp16.data(), dst.data(), checked_span_count(src_fp16.size(), dst.size(), dim), dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const float> src, uint32_t dim, size_t numThreads = 0) const {
        if (dim == 0) {
            if (src.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src.data(), src.size() / dim, dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const uint16_t> src_fp16, uint32_t dim, size_t numThreads = 0) const {
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
    inline int8_t transform(float x, uint32_t d) const {
        float scaled = std::round(x * scales[d]);
        if (scaled < -127.0f) return -127;
        if (scaled > 127.0f) return 127;
        return static_cast<int8_t>(scaled);
    }

    inline int8_t transform(uint16_t fp16_bits, uint32_t d) const {
        return transform(deglib::distances::fp16::fp16_to_float(fp16_bits), d);
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
};

/**
 * Global Affine UINT8 Quantizer [0, 255] (ideal for Euclidean L2).
 * Supports FP32 and FP16 (uint16_t) inputs.
 */
class ScalarQuantizerUint8 {
public:
    using output_type = uint8_t;
    float min_val = 0.0f;
    float max_val = 1.0f;
    float dif = 1.0f;
    float scale = 255.0f;
    bool is_fitted = false;

    ScalarQuantizerUint8() = default;
    ScalarQuantizerUint8(float min_v, float max_v) {
        set_range(min_v, max_v);
    }

    void set_range(float min_v, float max_v) {
        min_val = min_v;
        max_val = max_v;
        dif = max_val - min_val;
        if (dif <= 1e-12f) {
            dif = 1.0f;
        }
        scale = 255.0f / dif;
        is_fitted = true;
    }

    void fit(const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f) {
        if (count == 0 || dim == 0) return;
        auto [mi, ma] = find_minmax(data, count * dim, drop_ratio);
        set_range(mi, ma);
    }

    void fit(const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f) {
        if (count == 0 || dim == 0) return;
        std::vector<float> converted(count * dim);
        for (size_t i = 0; i < count * dim; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
        }
        fit(converted.data(), count, dim, drop_ratio);
    }

    void quantize(const float* src, uint8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src + i * dim, dst + i * dim, dim);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src + i * dim, dst + i * dim, dim);
            }
        });
    }

    void quantize(const uint16_t* src_fp16, uint8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, dim);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, dim);
            }
        });
    }

    std::vector<uint8_t> quantize(const float* src, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<uint8_t> result(count * dim);
        quantize(src, result.data(), count, dim, numThreads);
        return result;
    }

    std::vector<uint8_t> quantize(const uint16_t* src_fp16, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<uint8_t> result(count * dim);
        quantize(src_fp16, result.data(), count, dim, numThreads);
        return result;
    }

    void quantize(std::span<const float> src, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src.data(), dst.data(), checked_span_count(src.size(), dst.size(), dim), dim, numThreads);
    }

    void quantize(std::span<const uint16_t> src_fp16, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src_fp16.data(), dst.data(), checked_span_count(src_fp16.size(), dst.size(), dim), dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const float> src, uint32_t dim, size_t numThreads = 0) const {
        if (dim == 0) {
            if (src.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src.data(), src.size() / dim, dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const uint16_t> src_fp16, uint32_t dim, size_t numThreads = 0) const {
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
    inline uint8_t transform(float x) const {
        float scaled = std::round((x - min_val) * scale);
        if (scaled < 0.0f) return 0;
        if (scaled > 255.0f) return 255;
        return static_cast<uint8_t>(scaled);
    }

    inline uint8_t transform(uint16_t fp16_bits) const {
        return transform(deglib::distances::fp16::fp16_to_float(fp16_bits));
    }

    inline void transform_row(const float* src, uint8_t* dst, uint32_t dim) const {
        for (uint32_t j = 0; j < dim; ++j) {
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
};

/**
 * Per-dimension Affine UINT8 Quantizer [0, 255] (ideal for Euclidean L2 with varying dimensional scales).
 * Supports FP32 and FP16 (uint16_t) inputs.
 */
class ScalarQuantizerUint8PerDim {
public:
    using output_type = uint8_t;
    uint32_t dim = 0;
    std::vector<float> mins;
    std::vector<float> maxs;
    std::vector<float> difs;
    std::vector<float> scales;
    bool is_fitted = false;

    ScalarQuantizerUint8PerDim() = default;
    explicit ScalarQuantizerUint8PerDim(uint32_t d) : dim(d), mins(d), maxs(d), difs(d), scales(d) {}

    void set_ranges(const std::vector<float>& min_vals, const std::vector<float>& max_vals) {
        if (min_vals.size() != max_vals.size()) {
            throw std::invalid_argument("min_vals and max_vals size mismatch");
        }
        dim = static_cast<uint32_t>(min_vals.size());
        mins = min_vals;
        maxs = max_vals;
        difs.resize(dim);
        scales.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            difs[j] = maxs[j] - mins[j];
            if (difs[j] <= 1e-12f) difs[j] = 1.0f;
            scales[j] = 255.0f / difs[j];
        }
        is_fitted = true;
    }

    void fit(const float* data, size_t count, uint32_t d, float drop_ratio = 0.0f) {
        if (count == 0 || d == 0) return;
        dim = d;
        find_minmax_perdim(mins, maxs, data, count, dim, drop_ratio);
        difs.resize(dim);
        scales.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            difs[j] = maxs[j] - mins[j];
            if (difs[j] <= 1e-12f) difs[j] = 1.0f;
            scales[j] = 255.0f / difs[j];
        }
        is_fitted = true;
    }

    void fit(const uint16_t* data_fp16, size_t count, uint32_t d, float drop_ratio = 0.0f) {
        if (count == 0 || d == 0) return;
        std::vector<float> converted(count * d);
        for (size_t i = 0; i < count * d; ++i) {
            converted[i] = deglib::distances::fp16::fp16_to_float(data_fp16[i]);
        }
        fit(converted.data(), count, d, drop_ratio);
    }

    void quantize(const float* src, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerUint8PerDim: expected {}, got {}", dim, d));
        }
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

    void quantize(const uint16_t* src_fp16, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerUint8PerDim: expected {}, got {}", dim, d));
        }
        if (numThreads == 1 || count == 1) {
            for (size_t i = 0; i < count; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, d);
            }
            return;
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                this->transform_row(src_fp16 + i * dim, dst + i * dim, d);
            }
        });
    }

    std::vector<uint8_t> quantize(const float* src, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<uint8_t> result(count * d);
        quantize(src, result.data(), count, d, numThreads);
        return result;
    }

    std::vector<uint8_t> quantize(const uint16_t* src_fp16, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<uint8_t> result(count * d);
        quantize(src_fp16, result.data(), count, d, numThreads);
        return result;
    }

    void quantize(std::span<const float> src, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src.data(), dst.data(), checked_span_count(src.size(), dst.size(), dim), dim, numThreads);
    }

    void quantize(std::span<const uint16_t> src_fp16, std::span<output_type> dst, uint32_t dim, size_t numThreads = 0) const {
        quantize(src_fp16.data(), dst.data(), checked_span_count(src_fp16.size(), dst.size(), dim), dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const float> src, uint32_t dim, size_t numThreads = 0) const {
        if (dim == 0) {
            if (src.empty()) return {};
            throw std::invalid_argument("quantize: dim must be > 0");
        }
        if (src.size() % dim != 0) {
            throw std::invalid_argument("quantize: src span size must be a multiple of dim");
        }
        return quantize(src.data(), src.size() / dim, dim, numThreads);
    }

    std::vector<output_type> quantize(std::span<const uint16_t> src_fp16, uint32_t dim, size_t numThreads = 0) const {
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
    inline uint8_t transform(float x, uint32_t d) const {
        float scaled = std::round((x - mins[d]) * scales[d]);
        if (scaled < 0.0f) return 0;
        if (scaled > 255.0f) return 255;
        return static_cast<uint8_t>(scaled);
    }

    inline uint8_t transform(uint16_t fp16_bits, uint32_t d) const {
        return transform(deglib::distances::fp16::fp16_to_float(fp16_bits), d);
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
};

}  // namespace deglib::quantization::scalar
