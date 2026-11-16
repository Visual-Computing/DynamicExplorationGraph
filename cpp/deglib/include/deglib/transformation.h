#pragma once

#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace deglib::transformation {

// FourCC Identifiers for Feature Transformations
inline constexpr uint32_t TAG_MIPS = uint32_t('M') | (uint32_t('I') << 8) | (uint32_t('P') << 16) | (uint32_t('S') << 24);
inline constexpr uint32_t TAG_SQI8 = uint32_t('S') | (uint32_t('Q') << 8) | (uint32_t('I') << 16) | (uint32_t('8') << 24);
inline constexpr uint32_t TAG_SQU8 = uint32_t('S') | (uint32_t('Q') << 8) | (uint32_t('U') << 16) | (uint32_t('8') << 24);
inline constexpr uint32_t TAG_SQU4 = uint32_t('S') | (uint32_t('Q') << 8) | (uint32_t('U') << 16) | (uint32_t('4') << 24);
inline constexpr uint32_t TAG_SQI4 = uint32_t('S') | (uint32_t('Q') << 8) | (uint32_t('I') << 16) | (uint32_t('4') << 24);
inline constexpr uint32_t TAG_SQU7 = uint32_t('S') | (uint32_t('Q') << 8) | (uint32_t('U') << 16) | (uint32_t('7') << 24);
inline constexpr uint32_t TAG_SQ16 = uint32_t('S') | (uint32_t('Q') << 8) | (uint32_t('1') << 16) | (uint32_t('6') << 24);
inline constexpr uint32_t TAG_EVPQ = uint32_t('E') | (uint32_t('V') << 8) | (uint32_t('P') << 16) | (uint32_t('Q') << 24);
inline constexpr uint32_t TAG_NORM = uint32_t('N') | (uint32_t('O') << 8) | (uint32_t('R') << 16) | (uint32_t('M') << 24);
inline constexpr uint32_t TAG_FWHT = uint32_t('F') | (uint32_t('W') << 8) | (uint32_t('H') << 16) | (uint32_t('T') << 24);

/**
 * @brief Abstract base class for all feature and vector transformations.
 */
class FeatureTransformation {
  public:
    virtual ~FeatureTransformation() = default;

    virtual uint32_t tag() const noexcept = 0;
    virtual std::string_view name() const noexcept = 0;
    virtual uint32_t input_dim() const noexcept = 0;
    virtual uint32_t output_dim() const noexcept = 0;
    virtual size_t input_byte_size() const noexcept = 0;
    virtual size_t output_byte_size() const noexcept = 0;
    virtual size_t query_byte_size() const noexcept { return output_byte_size(); }

    virtual deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const { return input_metric; }

    // Indicates if transformation has been fitted and is ready to transform
    virtual bool is_fitted() const noexcept { return true; }

    // Fit methods (default no-op for parameter-free transformations)
    virtual void fit(std::span<const float>) {}
    virtual void fit(std::span<const uint16_t>) {}

    // Transform database / training vectors (e.g. at ingestion time in Builder)
    virtual void transform_database(std::span<const float> in, std::span<std::byte> out) const = 0;
    virtual void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const {}

    // Transform query vectors (at search time)
    virtual void transform_query(std::span<const float> in, std::span<std::byte> out) const = 0;
    virtual void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const {}

    /** Approximate inverse for native transformed feature rows. Supported by scalar quantizers and EVP. */
    virtual void dequantize(std::span<const std::byte>, std::span<float>) const {
        throw std::logic_error("FeatureTransformation::dequantize is not implemented");
    }

    // Binary serialization into stream
    virtual void serialize(std::ostream& os) const = 0;
};

}  // namespace deglib::transformation

#include "deglib/transformation/evp.h"
#include "deglib/transformation/mips.h"
#include "deglib/transformation/normalize.h"
#include "deglib/transformation/rotation.h"
#include "deglib/transformation/scalar/common.h"
#include "deglib/transformation/scalar/fp16.h"
#include "deglib/transformation/scalar/int4.h"
#include "deglib/transformation/scalar/int8.h"
#include "deglib/transformation/scalar/uint4.h"
#include "deglib/transformation/scalar/uint7.h"
#include "deglib/transformation/scalar/uint8.h"

namespace deglib::transformation {

namespace detail {

constexpr float default_rotation_kurtosis_threshold(deglib::distances::MetricDataType data_type, bool use_rerank = false) noexcept {
    switch (data_type) {
        case deglib::distances::MetricDataType::Uint4:
        case deglib::distances::MetricDataType::Int4:
            return 0.8f;
        case deglib::distances::MetricDataType::UInt7:
            return use_rerank ? 5.0f : 0.8f;
        case deglib::distances::MetricDataType::Uint8:
            return use_rerank ? 5.0f : 1.85f;
        case deglib::distances::MetricDataType::Int8:
            return use_rerank ? 25.0f : 1.85f;
        default:
            return -1.0f;
    }
}

template <typename QuantT>
inline std::vector<std::shared_ptr<FeatureTransformation>> build_quantizer_pipeline(
    std::span<const float> data, uint32_t dim, float quantile, float kurtosis_threshold,
    size_t num_threads = 0, uint64_t rotation_seed = 42
) {
    if (dim == 0) throw std::invalid_argument("make_quantizer_pipeline: dim must be > 0");
    if (data.size() % dim != 0) throw std::invalid_argument("make_quantizer_pipeline: data size must be multiple of dim");

    const size_t sample_items = std::min(data.size(), static_cast<size_t>(10000) * dim);
    std::span<const float> sample_data = data.first(sample_items);

    std::vector<std::shared_ptr<FeatureTransformation>> pipeline;
    std::span<const float> fit_data = sample_data;
    std::vector<float> rotated;
    if (kurtosis_threshold >= 0.0f) {
        auto rot = std::make_shared<FastHadamardRotation>(dim, rotation_seed);
        rot->fit_with_kurtosis(sample_data, kurtosis_threshold);
        if (rot->is_enabled()) {
            rotated.resize(sample_data.size());
            rot->transform_database(sample_data, std::as_writable_bytes(std::span<float>(rotated)), num_threads);
            fit_data = rotated;
            pipeline.push_back(std::move(rot));
        }
    }

    auto quant = std::make_shared<QuantT>(dim);
    quant->fit(fit_data, quantile);
    pipeline.push_back(std::move(quant));
    return pipeline;
}

}  // namespace detail

class TransformationWorkspace {
  private:
    std::vector<std::byte> buf_a_;
    std::vector<std::byte> buf_b_;
    std::vector<float> f_buf_;

  public:
    std::byte* get_buffer_a(size_t size) {
        if (buf_a_.size() < size) buf_a_.resize(size);
        return buf_a_.data();
    }
    std::byte* get_buffer_b(size_t size) {
        if (buf_b_.size() < size) buf_b_.resize(size);
        return buf_b_.data();
    }
    float* get_float_buffer(size_t count) {
        if (f_buf_.size() < count) f_buf_.resize(count);
        return f_buf_.data();
    }
};

inline void serialize_transformations(const std::vector<std::shared_ptr<FeatureTransformation>>& transformations, std::ostream& os) {
    uint32_t count = static_cast<uint32_t>(transformations.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& t : transformations) {
        if (!t) continue;
        uint32_t tag = t->tag();
        std::ostringstream ss(std::ios::binary);
        t->serialize(ss);
        std::string payload = ss.str();
        uint32_t len = static_cast<uint32_t>(payload.size());
        os.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(payload.data(), len);
    }
}

inline std::vector<std::shared_ptr<FeatureTransformation>> deserialize_transformations(std::istream& is) {
    uint32_t count = 0;
    if (!is.read(reinterpret_cast<char*>(&count), sizeof(count))) return {};
    std::vector<std::shared_ptr<FeatureTransformation>> transformations;
    transformations.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t tag = 0;
        uint32_t len = 0;
        if (!is.read(reinterpret_cast<char*>(&tag), sizeof(tag))) break;
        if (!is.read(reinterpret_cast<char*>(&len), sizeof(len))) break;

        std::string payload(len, '\0');
        is.read(payload.data(), len);
        std::stringstream ss(payload, std::ios::in | std::ios::binary);

        std::shared_ptr<FeatureTransformation> t;
        switch (tag) {
            case TAG_MIPS:
                t = MipsTransform::deserialize(ss);
                break;
            case TAG_SQI8:
                t = ScalarQuantizerInt8::deserialize(ss);
                break;
            case TAG_SQU7:
                t = ScalarQuantizerUInt7::deserialize(ss);
                break;
            case TAG_SQU8:
                t = ScalarQuantizerUint8::deserialize(ss);
                break;
            case TAG_SQ16:
                t = ScalarQuantizerFP16::deserialize(ss);
                break;
            case TAG_EVPQ:
                t = EvpQuantizer::deserialize(ss);
                break;
            case TAG_NORM:
                t = L2NormalizeTransform::deserialize(ss);
                break;
            case TAG_FWHT:
                t = FastHadamardRotation::deserialize(ss);
                break;
            default:
                break;
        }
        if (t) transformations.push_back(std::move(t));
    }
    return transformations;
}

inline void apply_query(std::span<const std::shared_ptr<FeatureTransformation>> transformations, const float* query, std::byte* out, size_t count = 1) {
    if (transformations.empty()) {
        std::memcpy(out, query, count * sizeof(float));
        return;
    }
    if (transformations.size() == 1) {
        const auto& t = transformations[0];
        t->transform_query(std::span<const float>(query, count * t->input_dim()), std::span<std::byte>(out, count * t->query_byte_size()));
        return;
    }

    thread_local TransformationWorkspace workspace;
    const float* cur_in = query;
    const size_t buf_a_size = transformations[0]->output_byte_size() * count;
    std::byte* cur_buf = workspace.get_buffer_a(buf_a_size);
    transformations[0]->transform_query(std::span<const float>(cur_in, count * transformations[0]->input_dim()), std::span<std::byte>(cur_buf, buf_a_size));

    for (size_t i = 1; i < transformations.size(); ++i) {
        const bool is_last = (i == transformations.size() - 1);
        const size_t cur_out_size = transformations[i]->output_byte_size() * count;
        std::byte* next_buf = is_last ? out : ((i % 2 == 1) ? workspace.get_buffer_b(cur_out_size) : workspace.get_buffer_a(cur_out_size));
        transformations[i]->transform_query(
            std::span<const float>(reinterpret_cast<const float*>(cur_buf), count * transformations[i]->input_dim()),
            std::span<std::byte>(next_buf, cur_out_size)
        );
        cur_buf = next_buf;
    }
}

namespace detail {

inline void apply_pipeline_one_float(
    std::span<const std::shared_ptr<FeatureTransformation>> transformations,
    const float* in_f32,
    std::byte* out,
    std::vector<std::byte>& buf_a,
    std::vector<std::byte>& buf_b
) {
    const size_t n_xfrms = transformations.size();
    if (n_xfrms == 1) {
        const auto& t = transformations[0];
        t->transform_database(std::span<const float>(in_f32, t->input_dim()), std::span<std::byte>(out, t->output_byte_size()));
        return;
    }

    const size_t s0 = transformations[0]->output_byte_size();
    if (buf_a.size() < s0) buf_a.resize(s0);
    transformations[0]->transform_database(
        std::span<const float>(in_f32, transformations[0]->input_dim()),
        std::span<std::byte>(buf_a.data(), s0)
    );
    std::byte* cur_buf = buf_a.data();

    for (size_t i = 1; i < n_xfrms; ++i) {
        const bool is_last = (i == n_xfrms - 1);
        const size_t out_bytes = transformations[i]->output_byte_size();
        std::byte* next_buf = out;
        if (!is_last) {
            auto& target_vec = (i % 2 == 1) ? buf_b : buf_a;
            if (target_vec.size() < out_bytes) target_vec.resize(out_bytes);
            next_buf = target_vec.data();
        }
        transformations[i]->transform_database(
            std::span<const float>(reinterpret_cast<const float*>(cur_buf), transformations[i]->input_dim()),
            std::span<std::byte>(next_buf, out_bytes)
        );
        cur_buf = next_buf;
    }
}

}  // namespace detail

inline void apply_database(
    std::span<const std::shared_ptr<FeatureTransformation>> transformations,
    const float* data,
    std::byte* out,
    size_t count,
    size_t num_threads = 0
) {
    if (transformations.empty()) {
        std::memcpy(out, data, count * sizeof(float));
        return;
    }
    if (count == 0) return;
    if (transformations.size() == 1) {
        const auto& t = transformations[0];
        t->transform_database(std::span<const float>(data, count * t->input_dim()), std::span<std::byte>(out, count * t->output_byte_size()));
        return;
    }

    const uint32_t in_dim = transformations.front()->input_dim();
    const size_t out_stride = transformations.back()->output_byte_size();
    if (num_threads == 1 || count == 1) {
        thread_local std::vector<std::byte> tls_a;
        thread_local std::vector<std::byte> tls_b;
        for (size_t v = 0; v < count; ++v) {
            detail::apply_pipeline_one_float(transformations, data + v * in_dim, out + v * out_stride, tls_a, tls_b);
        }
        return;
    }

    deglib::concurrent::parallel_batch_for(
        static_cast<size_t>(0), count, num_threads,
        [&](size_t begin, size_t end, size_t) {
            std::vector<std::byte> buf_a;
            std::vector<std::byte> buf_b;
            for (size_t v = begin; v < end; ++v) {
                detail::apply_pipeline_one_float(transformations, data + v * in_dim, out + v * out_stride, buf_a, buf_b);
            }
        },
        4
    );
}

inline void apply_database(
    std::span<const std::shared_ptr<FeatureTransformation>> transformations,
    const uint16_t* data_fp16,
    std::byte* out,
    size_t count,
    size_t num_threads = 0
) {
    if (count == 0 || transformations.empty()) return;
    const uint32_t in_dim = transformations.front()->input_dim();
    const size_t out_stride = transformations.back()->output_byte_size();

    if (num_threads == 1 || count == 1) {
        thread_local std::vector<float> tls_f;
        thread_local std::vector<std::byte> tls_a;
        thread_local std::vector<std::byte> tls_b;
        if (tls_f.size() < in_dim) tls_f.resize(in_dim);
        for (size_t v = 0; v < count; ++v) {
            deglib::distances::fp16::fp16_to_floats(data_fp16 + v * in_dim, tls_f.data(), in_dim);
            detail::apply_pipeline_one_float(transformations, tls_f.data(), out + v * out_stride, tls_a, tls_b);
        }
        return;
    }

    deglib::concurrent::parallel_batch_for(
        static_cast<size_t>(0), count, num_threads,
        [&](size_t begin, size_t end, size_t) {
            std::vector<float> f_buf(in_dim);
            std::vector<std::byte> buf_a;
            std::vector<std::byte> buf_b;
            for (size_t v = begin; v < end; ++v) {
                deglib::distances::fp16::fp16_to_floats(data_fp16 + v * in_dim, f_buf.data(), in_dim);
                detail::apply_pipeline_one_float(transformations, f_buf.data(), out + v * out_stride, buf_a, buf_b);
            }
        },
        4
    );
}

inline uint32_t get_input_dim(std::span<const std::shared_ptr<FeatureTransformation>> transformations) noexcept {
    return transformations.empty() ? 0 : transformations.front()->input_dim();
}

inline uint32_t get_output_dim(std::span<const std::shared_ptr<FeatureTransformation>> transformations) noexcept {
    return transformations.empty() ? 0 : transformations.back()->output_dim();
}

inline size_t get_output_byte_size(std::span<const std::shared_ptr<FeatureTransformation>> transformations) noexcept {
    return transformations.empty() ? 0 : transformations.back()->output_byte_size();
}

inline deglib::distances::Metric
get_target_metric(std::span<const std::shared_ptr<FeatureTransformation>> transformations, deglib::distances::Metric input_metric) {
    deglib::distances::Metric cur = input_metric;
    for (const auto& t : transformations) {
        if (t) cur = t->get_target_metric(cur);
    }
    return cur;
}

inline std::vector<std::shared_ptr<FeatureTransformation>> make_quantizer_pipeline(
    std::span<const float> data, uint32_t dim, deglib::distances::MetricDataType data_type,
    float quantile = -1.0f, bool use_rerank = false, size_t num_threads = 0, uint64_t rotation_seed = 42
) {
    const float threshold = detail::default_rotation_kurtosis_threshold(data_type, use_rerank);
    switch (data_type) {
        case deglib::distances::MetricDataType::Uint4:
            return detail::build_quantizer_pipeline<ScalarQuantizerUint4>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::Int4:
            return detail::build_quantizer_pipeline<ScalarQuantizerInt4>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::UInt7:
            return detail::build_quantizer_pipeline<ScalarQuantizerUInt7>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::Uint8:
            return detail::build_quantizer_pipeline<ScalarQuantizerUint8>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::Int8:
            return detail::build_quantizer_pipeline<ScalarQuantizerInt8>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::FP16:
            return detail::build_quantizer_pipeline<ScalarQuantizerFP16>(data, dim, quantile, -1.0f, num_threads, rotation_seed);
        default:
            throw std::invalid_argument("make_quantizer_pipeline: unsupported MetricDataType");
    }
}

inline std::vector<std::shared_ptr<FeatureTransformation>> make_quantizer_pipeline(
    std::span<const float> data, uint32_t dim, deglib::distances::Metric metric,
    float quantile = -1.0f, bool use_rerank = false, size_t num_threads = 0, uint64_t rotation_seed = 42
) {
    return make_quantizer_pipeline(data, dim, metric.get_data_type(), quantile, use_rerank, num_threads, rotation_seed);
}

inline std::vector<std::shared_ptr<FeatureTransformation>> make_quantizer_pipeline(
    std::span<const uint16_t> data_fp16, uint32_t dim, deglib::distances::MetricDataType data_type,
    float quantile = -1.0f, bool use_rerank = false, size_t num_threads = 0, uint64_t rotation_seed = 42
) {
    if (dim == 0) throw std::invalid_argument("make_quantizer_pipeline: dim must be > 0");
    if (data_fp16.size() % dim != 0) throw std::invalid_argument("make_quantizer_pipeline: data size must be multiple of dim");
    const size_t sample_items = std::min(data_fp16.size(), static_cast<size_t>(10000) * dim);
    std::vector<float> sample_f32(sample_items);
    deglib::distances::fp16::fp16_to_floats(data_fp16.data(), sample_f32.data(), sample_items);
    return make_quantizer_pipeline(std::span<const float>(sample_f32), dim, data_type, quantile, use_rerank, num_threads, rotation_seed);
}

inline std::vector<std::shared_ptr<FeatureTransformation>> make_quantizer_pipeline(
    std::span<const uint16_t> data_fp16, uint32_t dim, deglib::distances::Metric metric,
    float quantile = -1.0f, bool use_rerank = false, size_t num_threads = 0, uint64_t rotation_seed = 42
) {
    return make_quantizer_pipeline(data_fp16, dim, metric.get_data_type(), quantile, use_rerank, num_threads, rotation_seed);
}

}  // namespace deglib::transformation
