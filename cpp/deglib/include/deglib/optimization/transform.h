#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace deglib::optimization {

/**
 * @brief Transforms database vectors from d-dimensional space to (d+1)-dimensional space
 * for Maximum Inner Product Search (MIPS) using L2 distance.
 *
 * For each vector x_i in R^d:
 *   x'_i = [x_i, sqrt(M^2 - ||x_i||^2)] in R^(d+1)
 * where M^2 = max_i ||x_i||^2.
 *
 * All transformed database vectors x'_i will have identical norm ||x'_i|| = M.
 * Minimizing ||q' - x'_i||^2 (where q' = [q, 0]) under L2 distance is mathematically
 * equivalent to maximizing the inner product <q, x_i>.
 *
 * @param input Pointer to contiguous FP32 input array of shape (count, dim).
 * @param count Number of vectors.
 * @param dim Dimension of input vectors (d).
 * @param output Pointer to pre-allocated FP32 output array of shape (count, dim + 1).
 * @return Max vector norm M = sqrt(max_norm_sq).
 */
inline float mips_l2_transform(const float* input, size_t count, size_t dim, float* output) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("mips_l2_transform: input and output pointers must not be null");
    }
    if (count == 0 || dim == 0) {
        return 0.0f;
    }

    // 1. Compute squared norms and track maximum
    double max_norm_sq = 0.0;
    std::vector<double> norms_sq(count, 0.0);

    for (size_t i = 0; i < count; ++i) {
        double sum = 0.0;
        const float* vec = input + i * dim;
        for (size_t j = 0; j < dim; ++j) {
            double val = static_cast<double>(vec[j]);
            sum += val * val;
        }
        norms_sq[i] = sum;
        if (sum > max_norm_sq) {
            max_norm_sq = sum;
        }
    }

    // 2. Append extra dimension: sqrt(max_norm_sq - ||x_i||^2)
    const size_t new_dim = dim + 1;
    for (size_t i = 0; i < count; ++i) {
        const float* in_vec = input + i * dim;
        float* out_vec = output + i * new_dim;

        std::memcpy(out_vec, in_vec, dim * sizeof(float));

        double diff = max_norm_sq - norms_sq[i];
        float extra = (diff > 0.0) ? static_cast<float>(std::sqrt(diff)) : 0.0f;
        out_vec[dim] = extra;
    }

    return static_cast<float>(std::sqrt(max_norm_sq));
}

/**
 * @brief Pads query vectors from d-dimensional space to (d+1)-dimensional space
 * for MIPS queries against an L2-transformed database.
 *
 * For each query vector q_i in R^d:
 *   q'_i = [q_i, 0] in R^(d+1).
 *
 * @param input Pointer to contiguous FP32 input array of shape (count, dim).
 * @param count Number of query vectors.
 * @param dim Dimension of query vectors (d).
 * @param output Pointer to pre-allocated FP32 output array of shape (count, dim + 1).
 */
inline void mips_l2_transform_query(const float* input, size_t count, size_t dim, float* output) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("mips_l2_transform_query: input and output pointers must not be null");
    }
    if (count == 0 || dim == 0) {
        return;
    }

    const size_t new_dim = dim + 1;
    for (size_t i = 0; i < count; ++i) {
        const float* in_vec = input + i * dim;
        float* out_vec = output + i * new_dim;

        std::memcpy(out_vec, in_vec, dim * sizeof(float));
        out_vec[dim] = 0.0f;
    }
}

/**
 * @brief Convenience overload for std::vector input.
 */
inline std::pair<std::vector<float>, float> mips_l2_transform(const std::vector<float>& input, size_t count, size_t dim) {
    std::vector<float> output(count * (dim + 1));
    float max_norm = mips_l2_transform(input.data(), count, dim, output.data());
    return {output, max_norm};
}

/**
 * @brief Convenience overload for std::vector query input.
 */
inline std::vector<float> mips_l2_transform_query(const std::vector<float>& input, size_t count, size_t dim) {
    std::vector<float> output(count * (dim + 1));
    mips_l2_transform_query(input.data(), count, dim, output.data());
    return output;
}

} // namespace deglib::optimization
