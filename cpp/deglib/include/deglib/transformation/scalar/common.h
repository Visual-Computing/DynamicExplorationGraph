#pragma once

#include "deglib/distance/fp16.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

namespace deglib::transformation::detail {

template <typename T>
inline std::vector<T> sample_strided_rows(std::span<const T> data, uint32_t dim, size_t max_rows) {
    const size_t total_rows = data.size() / dim;
    const size_t sample_rows = std::min(total_rows, max_rows);
    std::vector<T> sampled(sample_rows * dim);
    for (size_t s = 0; s < sample_rows; ++s) {
        const size_t src_row = (sample_rows == total_rows) ? s : (s * total_rows) / sample_rows;
        std::memcpy(sampled.data() + s * dim, data.data() + src_row * dim, static_cast<size_t>(dim) * sizeof(T));
    }
    return sampled;
}

inline std::pair<float, float> find_minmax(const float* data, size_t total_items, float quantile = -1.0f) {
    if (total_items == 0) return {0.0f, 0.0f};

    if (quantile <= 0.0f || quantile >= 1.0f) {
        float min_val = data[0];
        float max_val = data[0];
        for (size_t i = 1; i < total_items; ++i) {
            float v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        return {min_val, max_val};
    }

    const float drop_ratio = std::max(0.0f, 1.0f - quantile);
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

inline float find_absmax(const float* data, size_t total_items, float quantile = -1.0f) {
    if (total_items == 0) return 0.0f;

    if (quantile <= 0.0f || quantile >= 1.0f) {
        float abs_max = 0.0f;
        for (size_t i = 0; i < total_items; ++i) {
            float v = std::abs(data[i]);
            if (v > abs_max) abs_max = v;
        }
        return abs_max;
    }

    const float drop_ratio = std::max(0.0f, 1.0f - quantile);
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

inline float find_absmax_fp16(std::span<const uint16_t> data_fp16, float quantile = -1.0f) {
    if (data_fp16.empty()) return 0.0f;
    if (quantile <= 0.0f || quantile >= 1.0f) {
        uint16_t max_half = 0;
        for (uint16_t h : data_fp16) {
            uint16_t mag = h & 0x7FFFu;
            if (mag > max_half) max_half = mag;
        }
        return deglib::distances::fp16::fp16_to_float(max_half);
    }

    const float drop_ratio = std::max(0.0f, 1.0f - quantile);
    const size_t top = static_cast<size_t>(static_cast<double>(data_fp16.size()) * drop_ratio) + 1;
    std::priority_queue<uint16_t, std::vector<uint16_t>, std::greater<uint16_t>> min_heap;
    for (uint16_t h : data_fp16) {
        uint16_t mag = h & 0x7FFFu;
        if (min_heap.size() < top) {
            min_heap.push(mag);
        } else if (mag > min_heap.top()) {
            min_heap.pop();
            min_heap.push(mag);
        }
    }
    return deglib::distances::fp16::fp16_to_float(min_heap.top());
}

inline std::pair<float, float> find_minmax_fp16(std::span<const uint16_t> data_fp16, float quantile = -1.0f) {
    if (data_fp16.empty()) return {0.0f, 0.0f};
    auto to_ordered_key = [](uint16_t h) noexcept -> int16_t {
        return (h & 0x8000u) ? static_cast<int16_t>(-static_cast<int32_t>(h & 0x7FFFu))
                             : static_cast<int16_t>(h);
    };
    auto from_ordered_key = [](int16_t key) noexcept -> uint16_t {
        return (key < 0) ? static_cast<uint16_t>(0x8000u | static_cast<uint16_t>(-static_cast<int32_t>(key)))
                         : static_cast<uint16_t>(key);
    };

    if (quantile <= 0.0f || quantile >= 1.0f) {
        int16_t min_key = to_ordered_key(data_fp16[0]);
        int16_t max_key = min_key;
        for (size_t i = 1; i < data_fp16.size(); ++i) {
            int16_t k = to_ordered_key(data_fp16[i]);
            if (k < min_key) min_key = k;
            if (k > max_key) max_key = k;
        }
        return {
            deglib::distances::fp16::fp16_to_float(from_ordered_key(min_key)),
            deglib::distances::fp16::fp16_to_float(from_ordered_key(max_key))
        };
    }

    const float drop_ratio = std::max(0.0f, 1.0f - quantile);
    const size_t top = static_cast<size_t>(static_cast<double>(data_fp16.size()) * drop_ratio) + 1;
    std::priority_queue<int16_t, std::vector<int16_t>, std::less<int16_t>> max_heap;
    std::priority_queue<int16_t, std::vector<int16_t>, std::greater<int16_t>> min_heap;
    for (uint16_t h : data_fp16) {
        int16_t k = to_ordered_key(h);
        if (min_heap.size() < top) {
            min_heap.push(k);
        } else if (k > min_heap.top()) {
            min_heap.pop();
            min_heap.push(k);
        }
        if (max_heap.size() < top) {
            max_heap.push(k);
        } else if (k < max_heap.top()) {
            max_heap.pop();
            max_heap.push(k);
        }
    }
    return {
        deglib::distances::fp16::fp16_to_float(from_ordered_key(max_heap.top())),
        deglib::distances::fp16::fp16_to_float(from_ordered_key(min_heap.top()))
    };
}

}  // namespace deglib::transformation::detail
