#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <config.h>

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
#include <immintrin.h>
#endif

namespace deglib {
namespace distances {

static inline float fp16_to_float(uint16_t h) {
#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    __m128i h_val = _mm_cvtsi32_si128(h);
    __m128 f_val = _mm_cvtph_ps(h_val);
    return _mm_cvtss_f32(f_val);
#else
    const uint32_t sign     = (h & 0x8000u) << 16;
    const uint32_t exponent = (h & 0x7C00u) >> 10;
    const uint32_t mantissa = (h & 0x03FFu);
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t e = 0;
            uint32_t m = mantissa << 1;
            while (!(m & 0x0400u)) { m <<= 1; ++e; }
            bits = sign | ((127 - 15 - e + 1) << 23) | ((m & 0x03FFu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static inline uint16_t float_to_fp16(float f) {
#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    __m128 f_val = _mm_set_ss(f);
    __m128i h_val = _mm_cvtps_ph(f_val, _MM_FROUND_TO_NEAREST_INT);
    return (uint16_t)_mm_cvtsi128_si32(h_val);
#else
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint16_t sign     = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    int32_t  exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0)      { return sign; }
    else if (exponent >= 31){ return static_cast<uint16_t>(sign | 0x7C00u); }
    else                    { return static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13)); }
#endif
}

static inline std::vector<uint16_t> floats_to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    size_t i = 0;
    for (; i + 4 <= v.size(); i += 4) {
        __m128 f4 = _mm_loadu_ps(&v[i]);
        __m128i h4 = _mm_cvtps_ph(f4, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h4);
        out[i]   = tmp[0];
        out[i+1] = tmp[1];
        out[i+2] = tmp[2];
        out[i+3] = tmp[3];
    }
    for (; i < v.size(); ++i) {
        out[i] = float_to_fp16(v[i]);
    }
#else
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = float_to_fp16(v[i]);
    }
#endif
    return out;
}

} // namespace distances
} // namespace deglib
