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
inline constexpr uint32_t TAG_SQ8P = make_fourcc('S', 'Q', '8', 'P');
inline constexpr uint32_t TAG_SQUP = make_fourcc('S', 'Q', 'U', 'P');
inline constexpr uint32_t TAG_EVPQ = make_fourcc('E', 'V', 'P', 'Q');

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

    // Backward compatibility aliases
    size_t input_bytes() const noexcept { return input_byte_size(); }
    size_t output_bytes() const noexcept { return output_byte_size(); }

    virtual deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const {
        return input_metric;
    }

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
#include "deglib/transformation/scalar.h"

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
        case TAG_SQU8:
            return ScalarQuantizerUint8::deserialize(ss);
        case TAG_SQ8P:
            return ScalarQuantizerInt8PerDim::deserialize(ss);
        case TAG_SQUP:
            return ScalarQuantizerUint8PerDim::deserialize(ss);
        case TAG_EVPQ:
            return EvpQuantizer::deserialize(ss);
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
        t->transform_query(std::span<const float>(query, count * t->input_dim()), std::span<std::byte>(out, count * t->output_byte_size()));
        return;
    }

    thread_local QueryWorkspace workspace;
    const float* cur_in = query;
    const size_t buf_a_size = transformations[0]->output_byte_size() * count;
    std::byte* cur_buf = workspace.get_buffer_a(buf_a_size);
    transformations[0]->transform_query(std::span<const float>(cur_in, count * transformations[0]->input_dim()),
                                       std::span<std::byte>(cur_buf, buf_a_size));

    for (size_t i = 1; i < transformations.size(); ++i) {
        const bool is_last = (i == transformations.size() - 1);
        const size_t cur_out_size = transformations[i]->output_byte_size() * count;
        std::byte* next_buf = is_last ? out
                                      : ((i % 2 == 1) ? workspace.get_buffer_b(cur_out_size)
                                                      : workspace.get_buffer_a(cur_out_size));
        transformations[i]->transform_query(
            std::span<const float>(reinterpret_cast<const float*>(cur_buf), count * transformations[i]->input_dim()),
            std::span<std::byte>(next_buf, cur_out_size));
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
    transformations[0]->transform_database(std::span<const float>(cur_in, count * transformations[0]->input_dim()),
                                          std::span<std::byte>(cur_buf, buf_a_size));

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
            std::span<std::byte>(next_buf, cur_out_size));
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

inline deglib::distances::Metric get_target_metric(
    std::span<const std::shared_ptr<FeatureTransformation>> transformations,
    deglib::distances::Metric input_metric
) {
    deglib::distances::Metric cur = input_metric;
    for (const auto& t : transformations) {
        if (t) cur = t->get_target_metric(cur);
    }
    return cur;
}

}  // namespace deglib::transformation
