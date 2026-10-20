#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace deglib::transformation {

/**
 * @brief Transforms database vectors from d-dimensional space to (d+1)-dimensional space
 * for Maximum Inner Product Search (MIPS) using L2 distance.
 *
 * For each vector x_i in R^d:
 *   x'_i = [x_i, sqrt(M^2 - ||x_i||^2)] in R^(d+1)
 * where M^2 = max_i ||x_i||^2.
 */
inline float mips_l2_transform(const float* input, size_t count, size_t dim, float* output) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("mips_l2_transform: input and output pointers must not be null");
    }
    if (count == 0 || dim == 0) return 0.0f;

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
        if (sum > max_norm_sq) max_norm_sq = sum;
    }

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
 * for MIPS queries against an L2-transformed database (q' = [q, 0]).
 */
inline void mips_l2_transform_query(const float* input, size_t count, size_t dim, float* output) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("mips_l2_transform_query: input and output pointers must not be null");
    }
    if (count == 0 || dim == 0) return;

    const size_t new_dim = dim + 1;
    for (size_t i = 0; i < count; ++i) {
        const float* in_vec = input + i * dim;
        float* out_vec = output + i * new_dim;
        std::memcpy(out_vec, in_vec, dim * sizeof(float));
        out_vec[dim] = 0.0f;
    }
}

inline std::pair<std::vector<float>, float> mips_l2_transform(const std::vector<float>& input, size_t count, size_t dim) {
    std::vector<float> output(count * (dim + 1));
    float max_norm = mips_l2_transform(input.data(), count, dim, output.data());
    return {output, max_norm};
}

inline std::vector<float> mips_l2_transform_query(const std::vector<float>& input, size_t count, size_t dim) {
    std::vector<float> output(count * (dim + 1));
    mips_l2_transform_query(input.data(), count, dim, output.data());
    return output;
}

// Forward declare FeatureTransformation if not yet defined
class FeatureTransformation;

class MipsTransform : public FeatureTransformation {
  private:
    uint32_t dim_ = 0;
    float max_norm_ = 0.0f;

  public:
    MipsTransform() = default;
    MipsTransform(uint32_t dim, float max_norm = 0.0f) : dim_(dim), max_norm_(max_norm) {}

    uint32_t tag() const noexcept { return 0x5350494D; }  // 'MIPS'
    std::string_view name() const noexcept { return "MipsTransform"; }
    uint32_t input_dim() const noexcept { return dim_; }
    uint32_t output_dim() const noexcept { return dim_ + 1; }
    size_t input_bytes() const noexcept { return dim_ * sizeof(float); }
    size_t output_bytes() const noexcept { return (dim_ + 1) * sizeof(float); }
    float max_norm() const noexcept { return max_norm_; }

    void fit(const float* data, size_t count) {
        if (count == 0 || dim_ == 0) return;
        double max_norm_sq = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double sum = 0.0;
            const float* vec = data + i * dim_;
            for (size_t j = 0; j < dim_; ++j) {
                double val = static_cast<double>(vec[j]);
                sum += val * val;
            }
            if (sum > max_norm_sq) max_norm_sq = sum;
        }
        max_norm_ = static_cast<float>(std::sqrt(max_norm_sq));
    }

    void transform_database(const float* in, std::byte* out, size_t count) const {
        float* f_out = reinterpret_cast<float*>(out);
        const size_t new_dim = dim_ + 1;
        const double max_norm_sq = double(max_norm_) * double(max_norm_);
        for (size_t i = 0; i < count; ++i) {
            const float* in_vec = in + i * dim_;
            float* out_vec = f_out + i * new_dim;
            std::memcpy(out_vec, in_vec, dim_ * sizeof(float));
            double sum = 0.0;
            for (size_t j = 0; j < dim_; ++j) {
                double val = static_cast<double>(in_vec[j]);
                sum += val * val;
            }
            double diff = max_norm_sq - sum;
            float extra = (diff > 0.0) ? static_cast<float>(std::sqrt(diff)) : 0.0f;
            out_vec[dim_] = extra;
        }
    }

    void transform_query(const float* in, std::byte* out, size_t count) const {
        mips_l2_transform_query(in, count, dim_, reinterpret_cast<float*>(out));
    }

    void serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&max_norm_), sizeof(max_norm_));
    }

    static std::unique_ptr<MipsTransform> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float max_norm = 0.0f;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&max_norm), sizeof(max_norm));
        return std::make_unique<MipsTransform>(dim, max_norm);
    }
};

}  // namespace deglib::transformation
