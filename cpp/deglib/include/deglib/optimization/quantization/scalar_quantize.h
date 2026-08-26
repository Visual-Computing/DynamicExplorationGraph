#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deglib::quantization::scalar {

// ============================================================================
// Calibration Helpers (finding min/max/absmax with optional drop_ratio / clipping)
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

/**
 * Finds global min and max across all items, with optional drop_ratio (percentile clipping).
 * If drop_ratio == 0.0f, computes exact min and max via linear scan.
 */
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
    std::priority_queue<float, std::vector<float>, std::less<float>> max_heap;     // keeps smallest top elements
    std::priority_queue<float, std::vector<float>, std::greater<float>> min_heap;  // keeps largest top elements

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

/**
 * Finds absolute maximum across all items, with optional drop_ratio (percentile clipping).
 * If drop_ratio == 0.0f, computes exact abs_max via linear scan.
 */
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

/**
 * Per-dimension min and max calibration.
 */
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
// Calibrators
// ============================================================================

struct SymCalibratorInt8 {
    float abs_max = 1.0f;
    float scale = 127.0f;       // scale = 127.0f / abs_max
    float inv_scale = 1.0f / 127.0f; // inv_scale = abs_max / 127.0f

    SymCalibratorInt8() = default;

    void calibrate(const float* data, size_t total_items, float drop_ratio = 0.0f) {
        abs_max = find_absmax(data, total_items, drop_ratio);
        if (abs_max <= 1e-12f) {
            abs_max = 1.0f;
        }
        scale = 127.0f / abs_max;
        inv_scale = abs_max / 127.0f;
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
};

struct AffineCalibratorUint8 {
    float min_val = 0.0f;
    float max_val = 1.0f;
    float dif = 1.0f;
    float scale = 255.0f;

    AffineCalibratorUint8() = default;

    void calibrate(const float* data, size_t total_items, float drop_ratio = 0.0f) {
        auto [mi, ma] = find_minmax(data, total_items, drop_ratio);
        min_val = mi;
        max_val = ma;
        dif = max_val - min_val;
        if (dif <= 1e-12f) {
            dif = 1.0f;
        }
        scale = 255.0f / dif;
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
};

struct AffinePerDimCalibratorUint8 {
    uint32_t dim = 0;
    std::vector<float> mins;
    std::vector<float> maxs;
    std::vector<float> difs;
    std::vector<float> scales;

    AffinePerDimCalibratorUint8() = default;
    explicit AffinePerDimCalibratorUint8(uint32_t d) : dim(d), mins(d), maxs(d), difs(d), scales(d) {}

    void calibrate(const float* data, size_t n, uint32_t d, float drop_ratio = 0.0f) {
        dim = d;
        find_minmax_perdim(mins, maxs, data, n, dim, drop_ratio);
        difs.resize(dim);
        scales.resize(dim);
        for (uint32_t j = 0; j < dim; ++j) {
            difs[j] = maxs[j] - mins[j];
            if (difs[j] <= 1e-12f) difs[j] = 1.0f;
            scales[j] = 255.0f / difs[j];
        }
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
};

// ============================================================================
// Batch & Single Quantization APIs
// ============================================================================

/**
 * Quantize FP32 vectors to symmetric signed INT8 [-127, 127] (ideal for InnerProduct / Cosine).
 * Multi-threaded using deglib::concurrent::parallel_batch_for.
 *
 * @param data       Pointer to [count x dim] float data
 * @param count      Number of vectors
 * @param dim        Dimension of each vector
 * @param drop_ratio Optional percentile clipping for calibration (default 0.0)
 * @param numThreads Worker threads (0 = auto-detect hardware concurrency)
 * @param out_calibrator Optional pointer to receive the fitted calibrator
 * @return std::vector<int8_t> with count * dim bytes
 */
inline std::vector<int8_t> quantize_int8_symmetric(
    const float* data,
    size_t count,
    uint32_t dim,
    float drop_ratio = 0.0f,
    size_t numThreads = 0,
    SymCalibratorInt8* out_calibrator = nullptr
) {
    if (count == 0 || dim == 0) return {};

    SymCalibratorInt8 cal;
    cal.calibrate(data, count * dim, drop_ratio);
    if (out_calibrator) {
        *out_calibrator = cal;
    }

    std::vector<int8_t> result(count * dim);

    deglib::concurrent::parallel_batch_for(0, count, numThreads, [data, dim, &cal, &result](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) {
            const float* src = data + i * dim;
            int8_t* dst = result.data() + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                dst[d] = cal.transform(src[d]);
            }
        }
    });

    return result;
}

/**
 * Quantize FP16 (uint16_t) vectors to symmetric signed INT8 [-127, 127].
 * First converts FP16 to FP32 during processing.
 */
inline std::vector<int8_t> quantize_int8_symmetric(
    const uint16_t* data,
    size_t count,
    uint32_t dim,
    float drop_ratio = 0.0f,
    size_t numThreads = 0,
    SymCalibratorInt8* out_calibrator = nullptr
) {
    if (count == 0 || dim == 0) return {};

    // Convert sample/all to float for calibration
    // For calibration, find absmax across half floats:
    // IEEE half-precision: abs(x) magnitude is monotonic in (x & 0x7FFF)
    uint16_t max_half = 0;
    for (size_t i = 0; i < count * dim; ++i) {
        uint16_t mag = data[i] & 0x7FFFu;
        if (mag > max_half) max_half = mag;
    }
    float abs_max = deglib::distances::fp16::fp16_to_float(max_half);

    SymCalibratorInt8 cal;
    if (abs_max <= 1e-12f) abs_max = 1.0f;
    cal.abs_max = abs_max;
    cal.scale = 127.0f / abs_max;
    cal.inv_scale = abs_max / 127.0f;
    if (out_calibrator) {
        *out_calibrator = cal;
    }

    std::vector<int8_t> result(count * dim);

    deglib::concurrent::parallel_batch_for(0, count, numThreads, [data, dim, &cal, &result](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) {
            const uint16_t* src = data + i * dim;
            int8_t* dst = result.data() + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                float val = deglib::distances::fp16::fp16_to_float(src[d]);
                dst[d] = cal.transform(val);
            }
        }
    });

    return result;
}

/**
 * Quantize FP32 vectors to unsigned UINT8 [0, 255] using global affine mapping (ideal for L2).
 */
inline std::vector<uint8_t> quantize_uint8_affine(
    const float* data,
    size_t count,
    uint32_t dim,
    float drop_ratio = 0.0f,
    size_t numThreads = 0,
    AffineCalibratorUint8* out_calibrator = nullptr
) {
    if (count == 0 || dim == 0) return {};

    AffineCalibratorUint8 cal;
    cal.calibrate(data, count * dim, drop_ratio);
    if (out_calibrator) {
        *out_calibrator = cal;
    }

    std::vector<uint8_t> result(count * dim);

    deglib::concurrent::parallel_batch_for(0, count, numThreads, [data, dim, &cal, &result](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) {
            const float* src = data + i * dim;
            uint8_t* dst = result.data() + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                dst[d] = cal.transform(src[d]);
            }
        }
    });

    return result;
}

/**
 * Quantize FP32 vectors to unsigned UINT8 [0, 255] using per-dimension affine mapping.
 */
inline std::vector<uint8_t> quantize_uint8_affine_perdim(
    const float* data,
    size_t count,
    uint32_t dim,
    float drop_ratio = 0.0f,
    size_t numThreads = 0,
    AffinePerDimCalibratorUint8* out_calibrator = nullptr
) {
    if (count == 0 || dim == 0) return {};

    AffinePerDimCalibratorUint8 cal(dim);
    cal.calibrate(data, count, dim, drop_ratio);
    if (out_calibrator) {
        *out_calibrator = cal;
    }

    std::vector<uint8_t> result(count * dim);

    deglib::concurrent::parallel_batch_for(0, count, numThreads, [data, dim, &cal, &result](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; ++i) {
            const float* src = data + i * dim;
            uint8_t* dst = result.data() + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                dst[d] = cal.transform(src[d], d);
            }
        }
    });

    return result;
}

}  // namespace deglib::quantization::scalar
