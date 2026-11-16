#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace deglib::transformation::detail {

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

/**
 * Optimizes the quantile for 4-bit scalar quantizers using a 2-stage approach:
 * 1. Computes the statistical MSE-optimal quantile (q_stat) by minimizing reconstruction error on base vectors.
 * 2. Spans an adaptive grid between q_stat and 1.0f and evaluates pseudo-query recall against exact FP32 distances.
 * 3. Chooses the clipped quantile only if it yields strictly higher recall than 1.0f (safety-first).
 *
 * @param data Array of float values (interleaved vectors)
 * @param total_items Total number of float items (num_vectors * dim)
 * @param dim Dimensionality of vectors
 * @param is_inner_product Whether metric is inner product (true) or L2 (false)
 * @return Optimal quantile in [q_stat, 1.0]
 */
inline float optimize_quantile_uint4(const float* data, size_t total_items, uint32_t dim, bool is_inner_product = true) {
    if (total_items == 0 || dim == 0) return 1.0f;
    const size_t num_vectors = total_items / dim;
    if (num_vectors < 50) return 1.0f;

    // Use up to 50 pseudo-queries and up to 1000 pseudo-base vectors
    const size_t num_queries = std::min<size_t>(50, num_vectors / 4);
    const size_t num_base = std::min<size_t>(1000, num_vectors - num_queries);
    if (num_queries == 0 || num_base == 0) return 1.0f;

    const float* queries = data;
    const float* base = data + num_queries * dim;

    // Step 1: Find statistical MSE-optimal quantile on base vectors
    const float mse_candidates[] = {0.980f, 0.985f, 0.990f, 0.995f, 0.997f, 0.999f, 0.9995f, 1.0f};
    float q_stat = 1.0f;
    double min_mse = std::numeric_limits<double>::infinity();

    std::vector<float> dequant_base(num_base * dim);

    for (float q : mse_candidates) {
        const auto [min_val, max_val] = find_minmax(base, num_base * dim, q);
        const float range = std::max(1e-12f, max_val - min_val);
        const float scale = 15.0f / range;
        const float inv_scale = range / 15.0f;

        double mse = 0.0;
        for (size_t i = 0; i < num_base * dim; ++i) {
            float v = std::clamp(base[i], min_val, max_val);
            float quantized = std::round((v - min_val) * scale);
            float dequant = min_val + quantized * inv_scale;
            float diff = base[i] - dequant;
            mse += static_cast<double>(diff) * diff;
        }
        if (mse < min_mse) {
            min_mse = mse;
            q_stat = q;
        }
    }

    // Step 2: Evaluate exact Top-K (K=20) for each pseudo-query against exact base
    const size_t k = std::min<size_t>(20, num_base);
    std::vector<std::vector<uint32_t>> exact_topk(num_queries, std::vector<uint32_t>(k));

    for (size_t q = 0; q < num_queries; ++q) {
        const float* q_vec = queries + q * dim;
        std::vector<std::pair<float, uint32_t>> dists(num_base);
        for (size_t b = 0; b < num_base; ++b) {
            const float* b_vec = base + b * dim;
            float score = 0.0f;
            if (is_inner_product) {
                for (uint32_t d = 0; d < dim; ++d) score += q_vec[d] * b_vec[d];
            } else {
                for (uint32_t d = 0; d < dim; ++d) {
                    float diff = q_vec[d] - b_vec[d];
                    score += diff * diff;
                }
            }
            dists[b] = {score, static_cast<uint32_t>(b)};
        }

        if (is_inner_product) {
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        } else {
            std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        for (size_t i = 0; i < k; ++i) {
            exact_topk[q][i] = dists[i].second;
        }
    }

    // Step 3: Grid search between q_stat and 1.0f on pseudo-query recall
    std::vector<float> grid_candidates;
    grid_candidates.push_back(1.0f);
    if (q_stat < 1.0f) {
        const int num_steps = 6;
        for (int step = 0; step < num_steps; ++step) {
            float val = q_stat + static_cast<float>(step) * (1.0f - q_stat) / static_cast<float>(num_steps);
            grid_candidates.push_back(val);
        }
    }

    float best_q = 1.0f;
    size_t max_hits = 0;

    for (float q : grid_candidates) {
        const auto [min_val, max_val] = find_minmax(base, num_base * dim, q);
        const float range = std::max(1e-12f, max_val - min_val);
        const float scale = 15.0f / range;
        const float inv_scale = range / 15.0f;

        for (size_t i = 0; i < num_base * dim; ++i) {
            float v = std::clamp(base[i], min_val, max_val);
            float quantized = std::round((v - min_val) * scale);
            dequant_base[i] = min_val + quantized * inv_scale;
        }

        size_t hits = 0;
        for (size_t q_idx = 0; q_idx < num_queries; ++q_idx) {
            const float* q_vec = queries + q_idx * dim;
            std::vector<std::pair<float, uint32_t>> dists(num_base);
            for (size_t b = 0; b < num_base; ++b) {
                const float* b_vec = dequant_base.data() + b * dim;
                float score = 0.0f;
                if (is_inner_product) {
                    for (uint32_t d = 0; d < dim; ++d) score += q_vec[d] * b_vec[d];
                } else {
                    for (uint32_t d = 0; d < dim; ++d) {
                        float diff = q_vec[d] - b_vec[d];
                        score += diff * diff;
                    }
                }
                dists[b] = {score, static_cast<uint32_t>(b)};
            }

            if (is_inner_product) {
                std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::partial_sort(dists.begin(), dists.begin() + k, dists.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            for (size_t i = 0; i < k; ++i) {
                uint32_t cand = dists[i].second;
                for (size_t j = 0; j < k; ++j) {
                    if (exact_topk[q_idx][j] == cand) {
                        ++hits;
                        break;
                    }
                }
            }
        }

        if (q == 1.0f) {
            max_hits = hits;
            best_q = 1.0f;
        } else if (hits > max_hits) {
            max_hits = hits;
            best_q = q;
        }
    }

    return best_q;
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

}  // namespace deglib::transformation::detail
