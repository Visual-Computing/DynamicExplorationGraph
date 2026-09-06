#pragma once

#include "deglib/config.h"
#include "deglib/utils/cpu.h"

#include <cmath>
#include <cstdint>

namespace deglib::distances {

// Shared FP32 distance utilities and declarations.
// Base header for all FP32 metric modules (fp32_l2.h, fp32_ip.h).
//
// Note: ResidualMode has been moved to distance/residual_mode.h.
// Include that header directly if you need the enum.

// ---------------------------------------------------------------------------
// L2 normalization (for Cosine metric)
// ---------------------------------------------------------------------------
// Normalizes a float vector in-place or to a separate buffer.
// Uses AVX2 for the dot product and scaling when available.
// ---------------------------------------------------------------------------

DEGLIB_TARGET_AVX2 inline void normalize_f32_avx2(const float* in, float* out, size_t dim) {
    // Compute sum of squares using AVX2
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        // Convert 8 floats to 8 doubles (4+4)
        __m128 v_lo128 = _mm256_castps256_ps128(v);
        __m128 v_hi128 = _mm256_extractf128_ps(v, 1);
        __m256d v_lo = _mm256_cvtps_pd(v_lo128);
        __m256d v_hi = _mm256_cvtps_pd(v_hi128);
        sum = _mm256_add_pd(sum, _mm256_mul_pd(v_hi, v_hi));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(v_lo, v_lo));
    }
    // Horizontal sum
    __m128d sum128 = _mm256_castpd256_pd128(sum);
    __m128d hi = _mm256_extractf128_pd(sum, 1);
    sum128 = _mm_add_pd(sum128, hi);
    double tmp[2];
    _mm_storeu_pd(tmp, sum128);
    double norm_sq = tmp[0] + tmp[1];
    // Handle remaining elements
    for (; i < dim; ++i) {
        norm_sq += in[i] * in[i];
    }
    float inv_norm = 1.0f / std::sqrt(static_cast<float>(norm_sq));
    // Scale
    i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        __m256 scale = _mm256_set1_ps(inv_norm);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, scale));
    }
    for (; i < dim; ++i) {
        out[i] = in[i] * inv_norm;
    }
}

inline void normalize_f32(const float* in, float* out, size_t dim) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx2()) {
        normalize_f32_avx2(in, out, dim);
        return;
    }
#endif
    // Scalar fallback
    double norm_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        norm_sq += in[i] * in[i];
    }
    float inv_norm = 1.0f / std::sqrt(static_cast<float>(norm_sq));
    for (size_t i = 0; i < dim; ++i) {
        out[i] = in[i] * inv_norm;
    }
}

}  // end namespace deglib::distances
