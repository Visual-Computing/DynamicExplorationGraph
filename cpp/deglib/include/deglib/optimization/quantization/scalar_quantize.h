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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deglib::quantization::scalar {

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
    float abs_max = 1.0f;
    float scale = 127.0f;
    float inv_scale = 1.0f / 127.0f;
    bool is_fitted = false;

    ScalarQuantizerInt8() = default;
    explicit ScalarQuantizerInt8(float abs_max_val) {
        set_abs_max(abs_max_val);
    }

    void set_abs_max(float val) {
        abs_max = (val <= 1e-12f) ? 1.0f : val;
        scale = 127.0f / abs_max;
        inv_scale = abs_max / 127.0f;
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

    inline int8_t transform(float x) const {
        float scaled = std::round(x * scale);
        if (scaled < -127.0f) return -127;
        if (scaled > 127.0f) return 127;
        return static_cast<int8_t>(scaled);
    }

    inline float transform_back(int8_t x) const {
        return static_cast<float>(x) * inv_scale;
    }

    void quantize(const float* src, int8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const float* s = src + i * dim;
                int8_t* d = dst + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    d[j] = this->transform(s[j]);
                }
            }
        });
    }

    void quantize(const uint16_t* src_fp16, int8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint16_t* s = src_fp16 + i * dim;
                int8_t* d = dst + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    float val = deglib::distances::fp16::fp16_to_float(s[j]);
                    d[j] = this->transform(val);
                }
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

    void dequantize(const int8_t* src, float* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const int8_t* s = src + i * dim;
                float* d = dst + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    d[j] = this->transform_back(s[j]);
                }
            }
        });
    }

    std::vector<float> dequantize(const int8_t* src, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<float> result(count * dim);
        dequantize(src, result.data(), count, dim, numThreads);
        return result;
    }

    std::vector<int8_t> fit_quantize(const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data, count, dim, drop_ratio);
        return quantize(data, count, dim, numThreads);
    }

    std::vector<int8_t> fit_quantize(const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data_fp16, count, dim, drop_ratio);
        return quantize(data_fp16, count, dim, numThreads);
    }
};

/**
 * Per-dimension Symmetric Signed INT8 Quantizer [-127, 127].
 * Calibrates abs_max per dimension. Supports FP32 and FP16 (uint16_t) inputs.
 */
class ScalarQuantizerInt8PerDim {
public:
    uint32_t dim = 0;
    std::vector<float> abs_maxs;
    std::vector<float> scales;
    std::vector<float> inv_scales;
    bool is_fitted = false;

    ScalarQuantizerInt8PerDim() = default;
    explicit ScalarQuantizerInt8PerDim(uint32_t d) : dim(d), abs_maxs(d), scales(d), inv_scales(d) {}

    void set_abs_maxs(const std::vector<float>& max_vals) {
        dim = static_cast<uint32_t>(max_vals.size());
        abs_maxs = max_vals;
        scales.resize(dim);
        inv_scales.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            float m = (abs_maxs[j] <= 1e-12f) ? 1.0f : abs_maxs[j];
            abs_maxs[j] = m;
            scales[j] = 127.0f / m;
            inv_scales[j] = m / 127.0f;
        }
        is_fitted = true;
    }

    void fit(const float* data, size_t count, uint32_t d, float drop_ratio = 0.0f) {
        if (count == 0 || d == 0) return;
        dim = d;
        find_absmax_perdim(abs_maxs, data, count, dim, drop_ratio);
        scales.resize(dim);
        inv_scales.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            float m = (abs_maxs[j] <= 1e-12f) ? 1.0f : abs_maxs[j];
            abs_maxs[j] = m;
            scales[j] = 127.0f / m;
            inv_scales[j] = m / 127.0f;
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

    inline int8_t transform(float x, uint32_t d) const {
        float scaled = std::round(x * scales[d]);
        if (scaled < -127.0f) return -127;
        if (scaled > 127.0f) return 127;
        return static_cast<int8_t>(scaled);
    }

    inline float transform_back(int8_t x, uint32_t d) const {
        return static_cast<float>(x) * inv_scales[d];
    }

    void quantize(const float* src, int8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerInt8PerDim: expected {}, got {}", dim, d));
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const float* s = src + i * d;
                int8_t* dst_row = dst + i * d;
                for (uint32_t j = 0; j < d; ++j) {
                    dst_row[j] = this->transform(s[j], j);
                }
            }
        });
    }

    void quantize(const uint16_t* src_fp16, int8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerInt8PerDim: expected {}, got {}", dim, d));
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint16_t* s = src_fp16 + i * d;
                int8_t* dst_row = dst + i * d;
                for (uint32_t j = 0; j < d; ++j) {
                    float val = deglib::distances::fp16::fp16_to_float(s[j]);
                    dst_row[j] = this->transform(val, j);
                }
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

    void dequantize(const int8_t* src, float* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerInt8PerDim: expected {}, got {}", dim, d));
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const int8_t* s = src + i * d;
                float* dst_row = dst + i * d;
                for (uint32_t j = 0; j < d; ++j) {
                    dst_row[j] = this->transform_back(s[j], j);
                }
            }
        });
    }

    std::vector<float> dequantize(const int8_t* src, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<float> result(count * d);
        dequantize(src, result.data(), count, d, numThreads);
        return result;
    }

    std::vector<int8_t> fit_quantize(const float* data, size_t count, uint32_t d, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data, count, d, drop_ratio);
        return quantize(data, count, d, numThreads);
    }

    std::vector<int8_t> fit_quantize(const uint16_t* data_fp16, size_t count, uint32_t d, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data_fp16, count, d, drop_ratio);
        return quantize(data_fp16, count, d, numThreads);
    }
};

/**
 * Global Affine UINT8 Quantizer [0, 255] (ideal for Euclidean L2).
 * Supports FP32 and FP16 (uint16_t) inputs.
 */
class ScalarQuantizerUint8 {
public:
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

    inline uint8_t transform(float x) const {
        float scaled = std::round((x - min_val) * scale);
        if (scaled < 0.0f) return 0;
        if (scaled > 255.0f) return 255;
        return static_cast<uint8_t>(scaled);
    }

    inline float transform_back(uint8_t x) const {
        return static_cast<float>(x) / 255.0f * dif + min_val;
    }

    void quantize(const float* src, uint8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const float* s = src + i * dim;
                uint8_t* d = dst + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    d[j] = this->transform(s[j]);
                }
            }
        });
    }

    void quantize(const uint16_t* src_fp16, uint8_t* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint16_t* s = src_fp16 + i * dim;
                uint8_t* d = dst + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    float val = deglib::distances::fp16::fp16_to_float(s[j]);
                    d[j] = this->transform(val);
                }
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

    void dequantize(const uint8_t* src, float* dst, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return;
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, dim, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* s = src + i * dim;
                float* d = dst + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    d[j] = this->transform_back(s[j]);
                }
            }
        });
    }

    std::vector<float> dequantize(const uint8_t* src, size_t count, uint32_t dim, size_t numThreads = 0) const {
        if (count == 0 || dim == 0) return {};
        std::vector<float> result(count * dim);
        dequantize(src, result.data(), count, dim, numThreads);
        return result;
    }

    std::vector<uint8_t> fit_quantize(const float* data, size_t count, uint32_t dim, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data, count, dim, drop_ratio);
        return quantize(data, count, dim, numThreads);
    }

    std::vector<uint8_t> fit_quantize(const uint16_t* data_fp16, size_t count, uint32_t dim, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data_fp16, count, dim, drop_ratio);
        return quantize(data_fp16, count, dim, numThreads);
    }
};

/**
 * Per-dimension Affine UINT8 Quantizer [0, 255] (ideal for Euclidean L2 with varying dimensional scales).
 * Supports FP32 and FP16 (uint16_t) inputs.
 */
class ScalarQuantizerUint8PerDim {
public:
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

    inline uint8_t transform(float x, uint32_t d) const {
        float scaled = std::round((x - mins[d]) * scales[d]);
        if (scaled < 0.0f) return 0;
        if (scaled > 255.0f) return 255;
        return static_cast<uint8_t>(scaled);
    }

    inline float transform_back(uint8_t x, uint32_t d) const {
        return static_cast<float>(x) / 255.0f * difs[d] + mins[d];
    }

    void quantize(const float* src, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerUint8PerDim: expected {}, got {}", dim, d));
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const float* s = src + i * d;
                uint8_t* dst_row = dst + i * d;
                for (uint32_t j = 0; j < d; ++j) {
                    dst_row[j] = this->transform(s[j], j);
                }
            }
        });
    }

    void quantize(const uint16_t* src_fp16, uint8_t* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerUint8PerDim: expected {}, got {}", dim, d));
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src_fp16, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint16_t* s = src_fp16 + i * d;
                uint8_t* dst_row = dst + i * d;
                for (uint32_t j = 0; j < d; ++j) {
                    float val = deglib::distances::fp16::fp16_to_float(s[j]);
                    dst_row[j] = this->transform(val, j);
                }
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

    void dequantize(const uint8_t* src, float* dst, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return;
        if (d != dim) {
            throw std::invalid_argument(std::format("Dimension mismatch in ScalarQuantizerUint8PerDim: expected {}, got {}", dim, d));
        }
        deglib::concurrent::parallel_batch_for(0, count, numThreads, [src, dst, d, this](size_t begin, size_t end, size_t) {
            for (size_t i = begin; i < end; ++i) {
                const uint8_t* s = src + i * d;
                float* dst_row = dst + i * d;
                for (uint32_t j = 0; j < d; ++j) {
                    dst_row[j] = this->transform_back(s[j], j);
                }
            }
        });
    }

    std::vector<float> dequantize(const uint8_t* src, size_t count, uint32_t d, size_t numThreads = 0) const {
        if (count == 0 || d == 0) return {};
        std::vector<float> result(count * d);
        dequantize(src, result.data(), count, d, numThreads);
        return result;
    }

    std::vector<uint8_t> fit_quantize(const float* data, size_t count, uint32_t d, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data, count, d, drop_ratio);
        return quantize(data, count, d, numThreads);
    }

    std::vector<uint8_t> fit_quantize(const uint16_t* data_fp16, size_t count, uint32_t d, float drop_ratio = 0.0f, size_t numThreads = 0) {
        fit(data_fp16, count, d, drop_ratio);
        return quantize(data_fp16, count, d, numThreads);
    }
};

}  // namespace deglib::quantization::scalar
