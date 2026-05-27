#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <config.h>
#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

#include "distances.h"

static float l2_naive(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static float ip_naive(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
}

static float l2_uint8_naive(const uint8_t* a, const uint8_t* b, size_t dim) {
    int64_t sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        sum += d * d;
    }
    return static_cast<float>(sum);
}

static const size_t DIM_4   = 4;
static const size_t DIM_8   = 8;
static const size_t DIM_16  = 16;
static const size_t DIM_128 = 128;

static std::vector<float> make_float_vec(size_t dim, unsigned seed = 42) {
    std::vector<float> v(dim);
    unsigned s = seed;
    for (size_t i = 0; i < dim; ++i) {
        s = s * 1103515245 + 12345;
        v[i] = static_cast<float>((s >> 16) % 10000) / 10000.0f * 10.0f - 5.0f;
    }
    return v;
}

static std::vector<uint8_t> make_uint8_vec(size_t dim, unsigned seed = 42) {
    std::vector<uint8_t> v(dim);
    unsigned s = seed;
    for (size_t i = 0; i < dim; ++i) {
        s = s * 1103515245 + 12345;
        v[i] = static_cast<uint8_t>((s >> 16) % 256);
    }
    return v;
}

static bool approx_eq(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) < eps;
}

static std::vector<uint16_t> floats_to_fp16(const std::vector<float>& v) {
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
        __m128 f1 = _mm_set_ss(v[i]);
        __m128i h1 = _mm_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h1);
        out[i] = tmp[0];
    }
#else
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        uint16_t sign     = static_cast<uint16_t>((bits >> 16) & 0x8000u);
        int32_t  exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFFu;
        if (exponent <= 0)      { out[i] = sign; }
        else if (exponent >= 31){ out[i] = static_cast<uint16_t>(sign | 0x7C00u); }
        else                    { out[i] = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13)); }
    }
#endif
    return out;
}

static float fp16_ip_naive_ref(const uint16_t* a, const uint16_t* b, size_t dim) {
    float dot = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += deglib::distances::fp16_to_float(a[i]) *
               deglib::distances::fp16_to_float(b[i]);
    }
    return 1.0f - dot;
}
