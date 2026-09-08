#pragma once

// C++20 concept describing the quantizer interface shared by all quantizer
// classes (ScalarQuantizerInt8/Int8PerDim/Uint8/Uint8PerDim, EvpQuantizer).
//
// Every quantizer exposes its packed element type via output_type and four
// quantize flavors per input precision (float / uint16_t fp16):
// pointer in-place, pointer returning, span in-place, span returning.

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

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
