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

namespace detail {

inline float mips_l2_transform_raw(const float* input, size_t count, size_t dim, float* output) {
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

    const double max_norm = std::sqrt(max_norm_sq);
    const double inv_max_norm = (max_norm > 0.0) ? (1.0 / max_norm) : 0.0;
    const size_t new_dim = dim + 1;
    for (size_t i = 0; i < count; ++i) {
        const float* in_vec = input + i * dim;
        float* out_vec = output + i * new_dim;
        for (size_t j = 0; j < dim; ++j) {
            out_vec[j] = static_cast<float>(static_cast<double>(in_vec[j]) * inv_max_norm);
        }
        const double scaled_norm_sq = norms_sq[i] * inv_max_norm * inv_max_norm;
        const double diff = 1.0 - scaled_norm_sq;
        out_vec[dim] = (diff > 0.0) ? static_cast<float>(std::sqrt(diff)) : 0.0f;
    }

    return static_cast<float>(max_norm);
}

inline void mips_l2_transform_query_raw(const float* input, size_t count, size_t dim, float* output) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("mips_l2_transform_query: input and output pointers must not be null");
    }
    if (count == 0 || dim == 0) return;

    const size_t new_dim = dim + 1;
    for (size_t i = 0; i < count; ++i) {
        const float* in_vec = input + i * dim;
        float* out_vec = output + i * new_dim;
        double norm_sq = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            const double val = static_cast<double>(in_vec[j]);
            norm_sq += val * val;
        }
        const double inv_norm = (norm_sq > 0.0) ? (1.0 / std::sqrt(norm_sq)) : 0.0;
        for (size_t j = 0; j < dim; ++j) {
            out_vec[j] = static_cast<float>(static_cast<double>(in_vec[j]) * inv_norm);
        }
        out_vec[dim] = 0.0f;
    }
}

}  // namespace detail

/**
 * @brief Transforms database vectors from d-dimensional space to (d+1)-dimensional space
 * for Maximum Inner Product Search (MIPS) using L2 distance.
 */
inline float mips_l2_transform(std::span<const float> input, size_t count, size_t dim, std::span<float> output) {
    if (count == 0 || dim == 0) return 0.0f;
    if (input.size() < count * dim) {
        throw std::invalid_argument("mips_l2_transform: input span too small");
    }
    if (output.size() < count * (dim + 1)) {
        throw std::invalid_argument("mips_l2_transform: output span too small");
    }
    return detail::mips_l2_transform_raw(input.data(), count, dim, output.data());
}

/**
 * @brief Pads query vectors from d-dimensional space to (d+1)-dimensional space
 * for MIPS queries against an L2-transformed database (q' = [q, 0]).
 */
inline void mips_l2_transform_query(std::span<const float> input, size_t count, size_t dim, std::span<float> output) {
    if (count == 0 || dim == 0) return;
    if (input.size() < count * dim) {
        throw std::invalid_argument("mips_l2_transform_query: input span too small");
    }
    if (output.size() < count * (dim + 1)) {
        throw std::invalid_argument("mips_l2_transform_query: output span too small");
    }
    detail::mips_l2_transform_query_raw(input.data(), count, dim, output.data());
}

inline std::pair<std::vector<float>, float> mips_l2_transform(std::span<const float> input, size_t count, size_t dim) {
    std::vector<float> output(count * (dim + 1));
    float max_norm = mips_l2_transform(input, count, dim, output);
    return {output, max_norm};
}

inline std::vector<float> mips_l2_transform_query(std::span<const float> input, size_t count, size_t dim) {
    std::vector<float> output(count * (dim + 1));
    mips_l2_transform_query(input, count, dim, output);
    return output;
}

/**
 * @brief Spherical MIPS-to-L2 / Cosine reduction (Neyshabur & Srebro, 2015 / Simple-LSH)
 *        lifting vectors from d dimensions to (d + 1) dimensions.
 *
 * Why this is needed:
 *   Unnormalized Inner Product (IP) is not a metric: it violates the triangle inequality
 *   and self-distances vary with vector norms, which degrades graph topology and greedy
 *   routing. Lifting every database vector onto a (d + 1)-dimensional sphere gives all
 *   database vectors identical L2 norm so that maximizing <x, q> is strictly equivalent
 *   to minimizing ||x' - q'||_2 (or maximizing <x', q'> on the unit sphere).
 *
 * Two database lifting modes are supported (controlled by `scale`):
 *   - `scale = true`  (MIPS_SCALED, default):
 *       x' = [x / M, sqrt(1 - ||x / M||^2)],   ||x'||_2 = 1
 *       q' = [q / ||q||, 0],                   ||q'||_2 = 1
 *     Because both x' and q' have unit norm, ||x' - q'||_2^2 = 2 - 2 * <x', q'>, so both
 *     L2 and InnerProduct spaces preserve the exact MIPS ranking. Used for FLAS presort
 *     (`PRESORT_SCALED`) and `FP32_IP | MIPS_SCALED` / `FP16_IP | MIPS_SCALED` search.
 *   - `scale = false` (MIPS_UNSCALED):
 *       x' = [x, sqrt(M^2 - ||x||^2)],         ||x'||_2 = M
 *       q' = [q / ||q||, 0]
 *     Avoids `1 / M` scaling while keeping ||x'||_2 = M constant across all database
 *     vectors. Used during graph construction (`BUILD_UNSCALED`).
 *
 * Example:
 * @code
 *   deglib::transformation::MipsTransform mips(dim, 0.0f, true);
 *   mips.fit(database_f32);
 *   mips.transform_database(database_f32, out_db_bytes);
 *   mips.transform_query(queries_f32, out_q_bytes);
 * @endcode
 */
class MipsTransform : public FeatureTransformation {
  private:
    uint32_t dim_ = 0;       ///< Input vector dimensionality `d` (output is `d + 1`).
    float max_norm_ = 0.0f;  ///< Calibration constant `M >= max_i ||x_i||_2` (fitted by `fit()`).
    bool scale_ = true;      ///< `true` = MIPS_SCALED (`1/M`), `false` = MIPS_UNSCALED (raw `x`).

  public:
    MipsTransform() = default;

    /**
     * @brief Construct a MipsTransform.
     * @param dim       Input vector dimensionality `d` (must be > 0 before `fit()` / `transform_*()`).
     * @param max_norm  Optional precomputed maximum L2 norm `M`. Pass `0.0f` to calibrate via `fit()`.
     * @param scale     `true` for `MIPS_SCALED` (`[x / M, sqrt(1 - ||x / M||^2)]`),
     *                  `false` for `MIPS_UNSCALED` (`[x, sqrt(M^2 - ||x||^2)]`).
     */
    MipsTransform(uint32_t dim, float max_norm = 0.0f, bool scale = true) : dim_(dim), max_norm_(max_norm), scale_(scale) {}

    uint32_t tag() const noexcept override { return 0x5350494D; }  // 'MIPS'
    TransformationType type() const noexcept override { return TransformationType::Spatial; }
    std::string_view name() const noexcept override { return "MipsTransform"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_ + 1; }
    size_t input_byte_size() const noexcept override { return dim_ * sizeof(float); }
    size_t output_byte_size() const noexcept override { return (dim_ + 1) * sizeof(float); }

    /// @brief Maximum L2 norm `M` used for spherical lifting.
    float max_norm() const noexcept { return max_norm_; }

    /// @brief Whether `transform_database()` scales by `1 / max_norm` (`true` = MIPS_SCALED, `false` = MIPS_UNSCALED).
    bool scale() const noexcept { return scale_; }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric) const override { return deglib::distances::Metric::FP32_L2; }

    /// @brief Returns true if `dim > 0` and `max_norm > 0`.
    bool is_fitted() const noexcept override { return dim_ > 0 && max_norm_ > 0.0f; }

    /**
     * @brief Calibrate `max_norm_` as `max_i ||x_i||_2` over all database vectors in `data`.
     * @param data Contiguous `float32` array of length `count * dim_`.
     */
    void fit(std::span<const float> data) override {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("MipsTransform: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("MipsTransform: data size must be multiple of dim");
        const size_t count = data.size() / dim_;
        double max_norm_sq = 0.0;
        for (size_t i = 0; i < count; ++i) {
            double sum = 0.0;
            const float* vec = data.data() + i * dim_;
            for (size_t j = 0; j < dim_; ++j) {
                double val = static_cast<double>(vec[j]);
                sum += val * val;
            }
            if (sum > max_norm_sq) max_norm_sq = sum;
        }
        max_norm_ = static_cast<float>(std::sqrt(max_norm_sq));
    }

    /**
     * @brief Lift `count` database vectors from `d` to `d + 1` dimensions.
     * @param in   Contiguous `float32` input span of length `count * dim_`.
     * @param out  Writable byte span of size at least `count * (dim_ + 1) * sizeof(float)`.
     */
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("MipsTransform: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("MipsTransform: input size must be multiple of input_dim");
        if (!is_fitted()) throw std::logic_error("MipsTransform: transform_database called before fit()");
        const size_t count = in.size() / dim_;
        const size_t new_dim = dim_ + 1;
        if (out.size() < count * new_dim * sizeof(float)) throw std::invalid_argument("MipsTransform: output buffer too small");

        float* f_out = reinterpret_cast<float*>(out.data());
        if (scale_) {
            const double inv_max_norm = (max_norm_ > 0.0f) ? (1.0 / static_cast<double>(max_norm_)) : 0.0;
            for (size_t i = 0; i < count; ++i) {
                const float* in_vec = in.data() + i * dim_;
                float* out_vec = f_out + i * new_dim;
                double sum = 0.0;
                for (size_t j = 0; j < dim_; ++j) {
                    const double val = static_cast<double>(in_vec[j]);
                    sum += val * val;
                    out_vec[j] = static_cast<float>(val * inv_max_norm);
                }
                const double scaled_norm_sq = sum * inv_max_norm * inv_max_norm;
                const double diff = 1.0 - scaled_norm_sq;
                out_vec[dim_] = (diff > 0.0) ? static_cast<float>(std::sqrt(diff)) : 0.0f;
            }
        } else {
            const double max_norm_sq = static_cast<double>(max_norm_) * static_cast<double>(max_norm_);
            for (size_t i = 0; i < count; ++i) {
                const float* in_vec = in.data() + i * dim_;
                float* out_vec = f_out + i * new_dim;
                double sum = 0.0;
                for (size_t j = 0; j < dim_; ++j) {
                    const double val = static_cast<double>(in_vec[j]);
                    sum += val * val;
                    out_vec[j] = in_vec[j];
                }
                const double diff = max_norm_sq - sum;
                out_vec[dim_] = (diff > 0.0) ? static_cast<float>(std::sqrt(diff)) : 0.0f;
            }
        }
    }

    /**
     * @brief Normalize `count` query vectors to unit length and append `0.0f` (`q' = [q / ||q||, 0]`).
     * @param in   Contiguous `float32` input span of length `count * dim_`.
     * @param out  Writable byte span of size at least `count * (dim_ + 1) * sizeof(float)`.
     */
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("MipsTransform: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("MipsTransform: input size must be multiple of input_dim");
        const size_t count = in.size() / dim_;
        const size_t new_dim = dim_ + 1;
        if (out.size() < count * new_dim * sizeof(float)) throw std::invalid_argument("MipsTransform: output buffer too small");
        detail::mips_l2_transform_query_raw(in.data(), count, dim_, reinterpret_cast<float*>(out.data()));
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&max_norm_), sizeof(max_norm_));
        const uint8_t scale_u8 = scale_ ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&scale_u8), sizeof(scale_u8));
    }

    static std::unique_ptr<MipsTransform> deserialize(std::istream& is) {
        uint32_t dim = 0;
        float max_norm = 0.0f;
        uint8_t scale_u8 = 1;
        is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        is.read(reinterpret_cast<char*>(&max_norm), sizeof(max_norm));
        is.read(reinterpret_cast<char*>(&scale_u8), sizeof(scale_u8));
        return std::make_unique<MipsTransform>(dim, max_norm, scale_u8 != 0);
    }
};

}  // namespace deglib::transformation
