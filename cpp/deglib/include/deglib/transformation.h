#pragma once

#include "deglib/distances.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace deglib::transformation {

constexpr uint32_t make_fourcc(char a, char b, char c, char d) noexcept {
    return (uint32_t(uint8_t(a))) | (uint32_t(uint8_t(b)) << 8) | (uint32_t(uint8_t(c)) << 16) | (uint32_t(uint8_t(d)) << 24);
}

// FourCC Identifiers for Feature Transformations
inline constexpr uint32_t TAG_MIPS = make_fourcc('M', 'I', 'P', 'S');
inline constexpr uint32_t TAG_SQI8 = make_fourcc('S', 'Q', 'I', '8');
inline constexpr uint32_t TAG_SQU8 = make_fourcc('S', 'Q', 'U', '8');
inline constexpr uint32_t TAG_SQU4 = make_fourcc('S', 'Q', 'U', '4');
inline constexpr uint32_t TAG_SQI4 = make_fourcc('S', 'Q', 'I', '4');
inline constexpr uint32_t TAG_SQU7 = make_fourcc('S', 'Q', 'U', '7');
inline constexpr uint32_t TAG_SQ16 = make_fourcc('S', 'Q', '1', '6');
inline constexpr uint32_t TAG_EVPQ = make_fourcc('E', 'V', 'P', 'Q');
inline constexpr uint32_t TAG_NORM = make_fourcc('N', 'O', 'R', 'M');
inline constexpr uint32_t TAG_FWHT = make_fourcc('F', 'W', 'H', 'T');

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

    // Backward compatibility aliases
    size_t input_bytes() const noexcept { return input_byte_size(); }
    size_t output_bytes() const noexcept { return output_byte_size(); }

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

inline std::shared_ptr<FeatureTransformation> deserialize_transformation(std::istream& is) {
    uint32_t tag = 0;
    uint32_t len = 0;
    if (!is.read(reinterpret_cast<char*>(&tag), sizeof(tag))) return nullptr;
    if (!is.read(reinterpret_cast<char*>(&len), sizeof(len))) return nullptr;

    std::string payload(len, '\0');
    is.read(payload.data(), len);
    std::stringstream ss(payload, std::ios::in | std::ios::binary);

    switch (tag) {
        case TAG_MIPS:
            return MipsTransform::deserialize(ss);
        case TAG_SQI8:
            return ScalarQuantizerInt8::deserialize(ss);
        case TAG_SQU7:
            return ScalarQuantizerUInt7::deserialize(ss);
        case TAG_SQU8:
            return ScalarQuantizerUint8::deserialize(ss);
        case TAG_SQ16:
            return ScalarQuantizerFP16::deserialize(ss);
        case TAG_EVPQ:
            return EvpQuantizer::deserialize(ss);
        case TAG_NORM:
            return L2NormalizeTransform::deserialize(ss);
        case TAG_FWHT:
            return FastHadamardRotation::deserialize(ss);
        default:
            return nullptr;
    }
}

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
        auto t = deserialize_transformation(is);
        if (t) transformations.push_back(std::move(t));
    }
    return transformations;
}

class QueryWorkspace {
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

    thread_local QueryWorkspace workspace;
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

inline void apply_database(std::span<const std::shared_ptr<FeatureTransformation>> transformations, const float* data, std::byte* out, size_t count) {
    if (transformations.empty()) {
        std::memcpy(out, data, count * sizeof(float));
        return;
    }
    if (transformations.size() == 1) {
        const auto& t = transformations[0];
        t->transform_database(std::span<const float>(data, count * t->input_dim()), std::span<std::byte>(out, count * t->output_byte_size()));
        return;
    }

    const size_t buf_a_size = transformations[0]->output_byte_size() * count;
    std::vector<std::byte> buf_a(buf_a_size);
    std::vector<std::byte> buf_b;
    const float* cur_in = data;
    std::byte* cur_buf = buf_a.data();
    transformations[0]->transform_database(std::span<const float>(cur_in, count * transformations[0]->input_dim()), std::span<std::byte>(cur_buf, buf_a_size));

    for (size_t i = 1; i < transformations.size(); ++i) {
        const bool is_last = (i == transformations.size() - 1);
        const size_t cur_out_size = transformations[i]->output_byte_size() * count;
        if (!is_last) {
            if (i % 2 == 1)
                buf_b.resize(cur_out_size);
            else
                buf_a.resize(cur_out_size);
        }
        std::byte* next_buf = is_last ? out : ((i % 2 == 1) ? buf_b.data() : buf_a.data());
        transformations[i]->transform_database(
            std::span<const float>(reinterpret_cast<const float*>(cur_buf), count * transformations[i]->input_dim()),
            std::span<std::byte>(next_buf, cur_out_size)
        );
        cur_buf = next_buf;
    }
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

// Backward compatibility alias
inline size_t get_output_bytes(std::span<const std::shared_ptr<FeatureTransformation>> transformations) noexcept {
    return get_output_byte_size(transformations);
}

inline deglib::distances::Metric
get_target_metric(std::span<const std::shared_ptr<FeatureTransformation>> transformations, deglib::distances::Metric input_metric) {
    deglib::distances::Metric cur = input_metric;
    for (const auto& t : transformations) {
        if (t) cur = t->get_target_metric(cur);
    }
    return cur;
}

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

    std::vector<std::shared_ptr<FeatureTransformation>> pipeline;
    std::span<const float> fit_data = data;
    std::vector<float> rotated;
    if (kurtosis_threshold >= 0.0f) {
        auto rot = std::make_shared<FastHadamardRotation>(dim, rotation_seed);
        rot->fit_with_kurtosis(data, kurtosis_threshold);
        if (rot->is_enabled()) {
            rotated.resize(data.size());
            rot->transform_database(data, std::as_writable_bytes(std::span<float>(rotated)), num_threads);
            fit_data = rotated;
            pipeline.push_back(std::move(rot));
        }
    }

    auto quant = std::make_shared<QuantT>(dim);
    quant->fit(fit_data, quantile);
    pipeline.push_back(std::move(quant));
    return pipeline;
}

inline std::vector<std::shared_ptr<FeatureTransformation>> make_quantizer_pipeline(
    std::span<const float> data, uint32_t dim, deglib::distances::MetricDataType data_type,
    float quantile = -1.0f, bool use_rerank = false, size_t num_threads = 0, uint64_t rotation_seed = 42
) {
    const float threshold = default_rotation_kurtosis_threshold(data_type, use_rerank);
    switch (data_type) {
        case deglib::distances::MetricDataType::Uint4:
            return build_quantizer_pipeline<ScalarQuantizerUint4>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::Int4:
            return build_quantizer_pipeline<ScalarQuantizerInt4>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::UInt7:
            return build_quantizer_pipeline<ScalarQuantizerUInt7>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::Uint8:
            return build_quantizer_pipeline<ScalarQuantizerUint8>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::Int8:
            return build_quantizer_pipeline<ScalarQuantizerInt8>(data, dim, quantile, threshold, num_threads, rotation_seed);
        case deglib::distances::MetricDataType::FP16:
            return build_quantizer_pipeline<ScalarQuantizerFP16>(data, dim, quantile, -1.0f, num_threads, rotation_seed);
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

}  // namespace deglib::transformation
