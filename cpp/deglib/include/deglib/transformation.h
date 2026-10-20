#pragma once

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
    virtual size_t input_bytes() const noexcept = 0;
    virtual size_t output_bytes() const noexcept = 0;

    // Transform database / training vectors (e.g. at ingestion time in Builder)
    virtual void transform_database(const float* in, std::byte* out, size_t count) const = 0;

    // Transform query vectors (at search time)
    virtual void transform_query(const float* in, std::byte* out, size_t count) const = 0;

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

inline void apply_query(const std::vector<std::shared_ptr<FeatureTransformation>>& transformations, const float* query, std::byte* out, size_t count = 1) {
    if (transformations.empty()) {
        std::memcpy(out, query, count * sizeof(float));
        return;
    }
    if (transformations.size() == 1) {
        transformations[0]->transform_query(query, out, count);
        return;
    }

    thread_local QueryWorkspace workspace;
    const float* cur_in = query;
    std::byte* cur_buf = workspace.get_buffer_a(transformations[0]->output_bytes() * count);
    transformations[0]->transform_query(cur_in, cur_buf, count);

    for (size_t i = 1; i < transformations.size(); ++i) {
        const bool is_last = (i == transformations.size() - 1);
        std::byte* next_buf = is_last ? out
                                      : ((i % 2 == 1) ? workspace.get_buffer_b(transformations[i]->output_bytes() * count)
                                                      : workspace.get_buffer_a(transformations[i]->output_bytes() * count));
        transformations[i]->transform_query(reinterpret_cast<const float*>(cur_buf), next_buf, count);
        cur_buf = next_buf;
    }
}

inline void apply_database(const std::vector<std::shared_ptr<FeatureTransformation>>& transformations, const float* data, std::byte* out, size_t count) {
    if (transformations.empty()) {
        std::memcpy(out, data, count * sizeof(float));
        return;
    }
    if (transformations.size() == 1) {
        transformations[0]->transform_database(data, out, count);
        return;
    }

    std::vector<std::byte> buf_a(transformations[0]->output_bytes() * count);
    std::vector<std::byte> buf_b;
    const float* cur_in = data;
    std::byte* cur_buf = buf_a.data();
    transformations[0]->transform_database(cur_in, cur_buf, count);

    for (size_t i = 1; i < transformations.size(); ++i) {
        const bool is_last = (i == transformations.size() - 1);
        if (!is_last) {
            if (i % 2 == 1)
                buf_b.resize(transformations[i]->output_bytes() * count);
            else
                buf_a.resize(transformations[i]->output_bytes() * count);
        }
        std::byte* next_buf = is_last ? out : ((i % 2 == 1) ? buf_b.data() : buf_a.data());
        transformations[i]->transform_database(reinterpret_cast<const float*>(cur_buf), next_buf, count);
        cur_buf = next_buf;
    }
}

inline uint32_t get_input_dim(const std::vector<std::shared_ptr<FeatureTransformation>>& transformations) noexcept {
    return transformations.empty() ? 0 : transformations.front()->input_dim();
}

inline uint32_t get_output_dim(const std::vector<std::shared_ptr<FeatureTransformation>>& transformations) noexcept {
    return transformations.empty() ? 0 : transformations.back()->output_dim();
}

inline size_t get_output_bytes(const std::vector<std::shared_ptr<FeatureTransformation>>& transformations) noexcept {
    return transformations.empty() ? 0 : transformations.back()->output_bytes();
}

}  // namespace deglib::transformation

namespace deglib::quantization {

template <typename Q>
concept Quantizer = requires(
    const Q& q,
    const float* f32,
    const uint16_t* f16,
    typename Q::output_type* dst,
    std::span<const float> f32_span,
    std::span<const uint16_t> f16_span,
    std::span<typename Q::output_type> dst_span,
    size_t count,
    uint32_t dim
) {
    typename Q::output_type;
    { q.quantize(f32, dst, count, dim) } -> std::same_as<void>;
    { q.quantize(f16, dst, count, dim) } -> std::same_as<void>;
    { q.quantize(f32, count, dim) } -> std::same_as<std::vector<typename Q::output_type>>;
    { q.quantize(f16, count, dim) } -> std::same_as<std::vector<typename Q::output_type>>;
    { q.quantize(f32_span, dst_span, dim) } -> std::same_as<void>;
    { q.quantize(f16_span, dst_span, dim) } -> std::same_as<void>;
    { q.quantize(f32_span, dim) } -> std::same_as<std::vector<typename Q::output_type>>;
    { q.quantize(f16_span, dim) } -> std::same_as<std::vector<typename Q::output_type>>;
};

}  // namespace deglib::quantization
