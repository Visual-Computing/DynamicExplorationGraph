#pragma once

#include "deglib/concurrent.h"
#include "deglib/transformation.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace deglib::transformation {

namespace detail {

/**
 * Normalizes a single vector in-place or to a destination buffer: dst = src / ||src||_2.
 * If norm <= 1e-12, zeroes out the vector or leaves it as-is.
 */
inline void normalize_vector_l2(const float* src, float* dst, uint32_t dim) noexcept {
    double sum = 0.0;
    for (uint32_t j = 0; j < dim; ++j) {
        double val = static_cast<double>(src[j]);
        sum += val * val;
    }

    if (sum > 1e-24) {
        float inv_norm = static_cast<float>(1.0 / std::sqrt(sum));
        for (uint32_t j = 0; j < dim; ++j) {
            dst[j] = src[j] * inv_norm;
        }
    } else {
        if (dst != src) {
            std::memset(dst, 0, dim * sizeof(float));
        }
    }
}

}  // namespace detail

/**
 * L2 Normalization Feature Transformation.
 *
 * Maps vectors x to unit length x / ||x||_2.
 * Output dimension is identical to input dimension (dim -> dim, float32 -> float32).
 */
class L2NormalizeTransform : public FeatureTransformation {
  public:
    using output_type = float;

    L2NormalizeTransform() = default;
    explicit L2NormalizeTransform(uint32_t dim) : dim_(dim) {}

    uint32_t dim() const noexcept { return dim_; }
    void set_dim(uint32_t d) noexcept { dim_ = d; }

    uint32_t tag() const noexcept override { return TAG_NORM; }
    std::string_view name() const noexcept override { return "L2NormalizeTransform"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return static_cast<size_t>(dim_) * sizeof(float); }
    size_t output_byte_size() const noexcept override { return static_cast<size_t>(dim_) * sizeof(float); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override {
        return input_metric;
    }

    bool is_fitted() const noexcept override { return dim_ > 0; }

    void fit(std::span<const float> data) override {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("L2NormalizeTransform: dim must be > 0 prior to fit()");
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }

    void transform_query(std::span<const float> in, std::span<std::byte> out) const override {
        transform_vectors(in, out, 0);
    }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    }

    static std::shared_ptr<L2NormalizeTransform> deserialize(std::istream& is) {
        uint32_t dim = 0;
        if (!is.read(reinterpret_cast<char*>(&dim), sizeof(dim))) return nullptr;
        return std::make_shared<L2NormalizeTransform>(dim);
    }

  private:
    void transform_vectors(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("L2NormalizeTransform: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("L2NormalizeTransform: input size must be multiple of dim");
        const size_t count = in.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(float);
        if (out.size() < req_bytes) throw std::invalid_argument("L2NormalizeTransform: output buffer too small");

        float* dst = reinterpret_cast<float*>(out.data());
        const float* src = in.data();

        if (numThreads <= 1 || count < 1000) {
            for (size_t i = 0; i < count; ++i) {
                detail::normalize_vector_l2(src + i * dim_, dst + i * dim_, dim_);
            }
        } else {
            deglib::concurrent::parallel_for(size_t{0}, count, numThreads, [&](size_t i, size_t /*thread_id*/) {
                detail::normalize_vector_l2(src + i * dim_, dst + i * dim_, dim_);
            });
        }
    }

    uint32_t dim_ = 0;
};

inline L2NormalizeTransform make_l2_normalize_transform(uint32_t dim = 0) {
    return L2NormalizeTransform(dim);
}

}  // namespace deglib::transformation
