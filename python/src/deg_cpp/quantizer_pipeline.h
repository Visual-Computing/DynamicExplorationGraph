#pragma once

#include <deglib/concurrent.h>
#include <deglib/distance/fp16.h>
#include <deglib/distances.h>
#include <deglib/transformation.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace deglib_py {

namespace detail {

/**
 * Loads one FP32 or FP16 vector as an FP32 span, using `scratch` only for FP16 conversion.
 */
inline std::span<const float> load_as_f32_span(const std::byte* src, uint32_t dim, bool src_fp16, std::vector<float>& scratch) {
    if (src_fp16) {
        if (scratch.size() < dim) scratch.resize(dim);
        deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(src), scratch.data(), dim);
        return std::span<const float>(scratch.data(), dim);
    }
    return std::span<const float>(reinterpret_cast<const float*>(src), dim);
}

/**
 * Decides whether an InnerProduct graph needs a MipsTransform (dim -> dim + 1) and fits its max_norm.
 *
 * Why scan all `n` vectors:
 *   MipsTransform maps x -> [x / M, sqrt(1 - ||x / M||^2)] where M = max_i ||x_i||.
 *   A subsample could miss high-norm outliers and produce ||x / M|| > 1 (NaN in sqrt).
 * Why skip MipsTransform for unit-norm vectors:
 *   When all vectors already have ||x|| ~= 1, Inner Product ordering equals L2 ordering
 *   (||a - b||^2 = 2 - 2*dot(a,b)), so an extra MIPS dimension is unnecessary.
 * Why use FloatSpace(InnerProduct):
 *   Reuses the SIMD AVX2/AVX512 FP16/FP32 inner-product kernel (dist = 1 - dot(v, v))
 *   instead of a manual scalar loop or FP16-to-FP32 scratch buffer.
 */
template <typename InternalGraphT>
inline std::shared_ptr<deglib::transformation::MipsTransform> fit_mips_transform_from_graph(
    const InternalGraphT& internal, uint32_t dim, bool src_fp16
) {
    const auto ip_metric = src_fp16 ? deglib::distances::Metric::FP16_InnerProduct : deglib::distances::Metric::FP32_InnerProduct;
    const deglib::distances::FloatSpace ip_space(dim, ip_metric);
    const auto ip_dist = ip_space.get_dist_func();
    const void* dist_param = ip_space.get_dist_func_param();

    float min_norm_sq = std::numeric_limits<float>::max();
    float max_norm_sq = 0.0f;
    const size_t n = internal.size();
    for (size_t i = 0; i < n; ++i) {
        const std::byte* src = internal.getFeatureVector(static_cast<uint32_t>(i));
        const float norm_sq = 1.0f - ip_dist(src, src, dist_param);
        if (norm_sq < min_norm_sq) min_norm_sq = norm_sq;
        if (norm_sq > max_norm_sq) max_norm_sq = norm_sq;
    }

    const float tol = src_fp16 ? 1e-2f : 1e-3f;
    const bool is_unit_norm = (std::abs(min_norm_sq - 1.0f) <= tol) && (std::abs(max_norm_sq - 1.0f) <= tol);
    if (is_unit_norm) {
        return nullptr;
    }
    return std::make_shared<deglib::transformation::MipsTransform>(dim, std::sqrt(max_norm_sq));
}

}  // namespace detail

/**
 * Calibrates a quantizer pipeline directly from an FP32/FP16 graph's stored feature vectors.
 * When `mips` is true, checks whether stored vectors are already unit-norm; if not, fits and
 * prepends a `MipsTransform` (`dim -> dim + 1`). When `normalize` is true, prepends `L2NormalizeTransform`.
 */
template <typename InternalGraphT>
inline std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> make_quantizer_pipeline_from_graph(
    const InternalGraphT& internal,
    deglib::distances::MetricDataType target_dt,
    float quantile = -1.0f,
    bool use_rerank = false,
    size_t num_threads = 0,
    uint64_t rotation_seed = 7,
    bool normalize = false,
    bool mips = false
) {
    if (internal.size() == 0) {
        throw std::invalid_argument("Cannot calibrate quantizer pipeline from an empty graph");
    }
    if (normalize && mips) {
        throw std::invalid_argument("Cannot enable both normalize and mips in make_quantizer_pipeline_from_graph");
    }
    const auto& src_space = internal.getFeatureSpace();
    const auto src_dt = src_space.metric().get_data_type();
    if (src_dt != deglib::distances::MetricDataType::FP32 && src_dt != deglib::distances::MetricDataType::FP16) {
        throw std::invalid_argument("Cannot calibrate quantizer pipeline from a non-float graph without external features");
    }

    const size_t n = internal.size();
    const uint32_t dim = static_cast<uint32_t>(src_space.dim());
    const size_t max_sample = (target_dt == deglib::distances::MetricDataType::EVP) ? size_t{65536} : size_t{10000};
    const size_t sample_count = std::min(n, max_sample);
    const bool src_fp16 = (src_dt == deglib::distances::MetricDataType::FP16);

    auto l2n = normalize ? std::make_shared<deglib::transformation::L2NormalizeTransform>(dim) : nullptr;
    auto mips_tr = mips ? detail::fit_mips_transform_from_graph(internal, dim, src_fp16) : nullptr;

    // Downstream quantizers (rotation, scalar/binary quantizers) must calibrate on the
    // post-prefix vectors (`sample_f32`), not on raw vectors. The full graph is transformed
    // later when the caller applies the returned pipeline.
    const bool use_mips = static_cast<bool>(mips_tr);
    const uint32_t prep_dim = use_mips ? (dim + 1) : dim;
    std::vector<float> sample_f32(sample_count * prep_dim);

    deglib::concurrent::parallel_batch_for(
        size_t{0}, sample_count, num_threads,
        [&](size_t begin, size_t end, size_t) {
            std::vector<float> fp16_scratch;
            for (size_t s = begin; s < end; ++s) {
                const uint32_t v_idx = static_cast<uint32_t>((sample_count == n) ? s : (s * n) / sample_count);
                const std::byte* src = internal.getFeatureVector(v_idx);
                float* dst = sample_f32.data() + s * prep_dim;
                auto dst_bytes = std::as_writable_bytes(std::span<float>(dst, prep_dim));

                if (use_mips) {
                    auto raw_f32 = detail::load_as_f32_span(src, dim, src_fp16, fp16_scratch);
                    mips_tr->transform_database(raw_f32, dst_bytes);
                } else {
                    if (src_fp16) {
                        deglib::distances::fp16::fp16_to_floats(reinterpret_cast<const uint16_t*>(src), dst, dim);
                    } else {
                        std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
                    }
                    if (l2n) {
                        l2n->transform_database(std::span<const float>(dst, dim), dst_bytes);
                    }
                }
            }
        },
        64
    );

    auto pipeline = deglib::transformation::make_quantizer_pipeline(
        std::span<const float>(sample_f32), prep_dim, target_dt, quantile, use_rerank, num_threads, rotation_seed
    );
    if (l2n) {
        pipeline.insert(pipeline.begin(), std::move(l2n));
    } else if (mips_tr) {
        pipeline.insert(pipeline.begin(), std::move(mips_tr));
    }
    return pipeline;
}

}  // namespace deglib_py
