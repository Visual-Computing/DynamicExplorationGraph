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
