#pragma once

#include "deglib/config.h"
#include "deglib/concurrent.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

namespace deglib::distances {

// Shared FP16 distance utilities and declarations.
// Base header for future FP16 metric modules (fp16_l2.h, fp16_ip.h).

// ---------------------------------------------------------------------------
// FP16 <-> Float conversion utilities
// ---------------------------------------------------------------------------
// IEEE 754 half-precision (binary16) conversion functions.
// FP16 vectors are stored as uint16_t bit patterns (IEEE 754 half-precision).
// These functions provide bit-exact conversion between float (binary32)
// and half-precision, using hardware F16C when available and a
// bit-manipulation scalar fallback otherwise.
// ---------------------------------------------------------------------------

namespace fp16 {

// ---------------------------------------------------------------------------
// GCC/Clang F16C intrinsics (target-attributed)
// ---------------------------------------------------------------------------
// On GCC/Clang, __attribute__((target("f16c,avx"))) forces the compiler to
// generate F16C instructions without requiring global -mf16c flags.
// On MSVC, these intrinsics are not available in the same form, so we use
// _mm_cvtps_ph / _mm_cvtph_ps (SSE intrinsics) instead.
// ---------------------------------------------------------------------------

#if defined(DEGLIB_X86) && (defined(__GNUC__) || defined(__clang__))

DEGLIB_TARGET_AVX2 inline uint16_t float_to_fp16_gcc(float f) { return _cvtss_sh(f, 0); }

DEGLIB_TARGET_AVX2 inline float fp16_to_float_gcc(uint16_t h) { return _cvtsh_ss(h); }

DEGLIB_TARGET_AVX2 inline void floats_to_fp16_gcc(const float* floats, uint16_t* fp16_vals, size_t count) {
    size_t i = 0;
    // Process 8 floats per step with _mm256_cvtps_ph
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(floats + i);
        __m128i vhp = _mm256_cvtps_ph(va, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(fp16_vals + i), vhp);
    }
    // Process 4 floats per step with _mm_cvtps_ph
    for (; i + 4 <= count; i += 4) {
        __m128 va = _mm_loadu_ps(floats + i);
        __m128i vhp = _mm_cvtps_ph(va, 0);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(fp16_vals + i), vhp);
    }
    // Scalar fallback for remaining 0-3 elements
    for (; i < count; ++i) {
        fp16_vals[i] = _cvtss_sh(floats[i], 0);
    }
}

DEGLIB_TARGET_AVX2 inline void fp16_to_floats_gcc(const uint16_t* fp16_vals, float* floats, size_t count) {
    size_t i = 0;
    // Process 8 uint16_t per step with _mm256_cvtph_ps
    for (; i + 8 <= count; i += 8) {
        __m128i vhp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16_vals + i));
        __m256 va = _mm256_cvtph_ps(vhp);
        _mm256_storeu_ps(floats + i, va);
    }
    // Process 4 uint16_t per step with _mm_cvtph_ps
    for (; i + 4 <= count; i += 4) {
        __m128i vhp = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(fp16_vals + i));
        __m128 va = _mm_cvtph_ps(vhp);
        _mm_storeu_ps(floats + i, va);
    }
    // Scalar fallback for remaining 0-3 elements
    for (; i < count; ++i) {
        floats[i] = _cvtsh_ss(fp16_vals[i]);
    }
}

#endif  // defined(DEGLIB_X86) && (defined(__GNUC__) || defined(__clang__))

// ---------------------------------------------------------------------------
// MSVC F16C intrinsics (using SSE _mm_cvtps_ph / _mm_cvtph_ps)
// ---------------------------------------------------------------------------
// MSVC's <immintrin.h> doesn't define GCC-style F16C intrinsics
// (_cvtss_sh, _cvtsh_ss). Instead, we use _mm_cvtps_ph / _mm_cvtph_ps
// which operate on __m128/__m128i and are available with /arch:AVX2.
// ---------------------------------------------------------------------------

#if defined(DEGLIB_X86) && defined(_MSC_VER)

inline uint16_t float_to_fp16_msvc(float f) {
    __m128 f_val = _mm_set_ss(f);
    __m128i h_val = _mm_cvtps_ph(f_val, 0);
    return static_cast<uint16_t>(_mm_cvtsi128_si32(h_val));
}

inline float fp16_to_float_msvc(uint16_t h) {
    __m128i h_val = _mm_cvtsi32_si128(static_cast<int>(h));
    __m128 f_val = _mm_cvtph_ps(h_val);
    return _mm_cvtss_f32(f_val);
}

inline void floats_to_fp16_msvc(const float* floats, uint16_t* fp16_vals, size_t count) {
    size_t i = 0;
    // Process 8 floats per step with _mm256_cvtps_ph
    for (; i + 8 <= count; i += 8) {
        __m256 va = _mm256_loadu_ps(floats + i);
        __m128i vhp = _mm256_cvtps_ph(va, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(fp16_vals + i), vhp);
    }
    // Process 4 floats per step with _mm_cvtps_ph
    for (; i + 4 <= count; i += 4) {
        __m128 va = _mm_loadu_ps(floats + i);
        __m128i vhp = _mm_cvtps_ph(va, 0);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(fp16_vals + i), vhp);
    }
    // Scalar fallback for remaining 0-3 elements
    for (; i < count; ++i) {
        fp16_vals[i] = float_to_fp16_msvc(floats[i]);
    }
}

inline void fp16_to_floats_msvc(const uint16_t* fp16_vals, float* floats, size_t count) {
    size_t i = 0;
    // Process 8 uint16_t per step with _mm256_cvtph_ps
    for (; i + 8 <= count; i += 8) {
        __m128i vhp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fp16_vals + i));
        __m256 va = _mm256_cvtph_ps(vhp);
        _mm256_storeu_ps(floats + i, va);
    }
    // Process 4 uint16_t per step with _mm_cvtph_ps
    for (; i + 4 <= count; i += 4) {
        __m128i vhp = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(fp16_vals + i));
        __m128 va = _mm_cvtph_ps(vhp);
        _mm_storeu_ps(floats + i, va);
    }
    // Scalar fallback for remaining 0-3 elements
    for (; i < count; ++i) {
        floats[i] = fp16_to_float_msvc(fp16_vals[i]);
    }
}

#endif  // defined(DEGLIB_X86) && defined(_MSC_VER)

// ---------------------------------------------------------------------------
// Scalar IEEE 754 Round-to-Nearest-Even fallback
// ---------------------------------------------------------------------------
// Used when F16C is not available at runtime (or not compiled in).
// Implements proper round-to-nearest-even rounding per IEEE 754.
// ---------------------------------------------------------------------------

inline uint16_t float_to_fp16_scalar(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(f));
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t mantissa = x & 0x7FFFFF;
    int32_t exponent = ((x >> 23) & 0xFF) - 127;
    uint32_t fp16_exp;

    if (exponent < -24) {
        // Too small, underflow to zero
        return static_cast<uint16_t>(sign << 15);
    } else if (exponent < -14) {
        // Subnormal FP16
        int32_t shift = -14 - exponent;
        uint32_t mantissa_with_hidden = mantissa | 0x800000;
        uint32_t fp16_mantissa = mantissa_with_hidden >> (23 + shift);
        // Round to nearest even
        uint32_t remainder = mantissa_with_hidden & ((1u << (23 + shift)) - 1);
        if (remainder > (1u << (22 + shift)) || (remainder == (1u << (22 + shift)) && (fp16_mantissa & 1))) {
            fp16_mantissa++;
        }
        fp16_exp = 0;
        return static_cast<uint16_t>((sign << 15) | (fp16_exp << 10) | (fp16_mantissa & 0x3FF));
    } else if (exponent <= 15) {
        // Normal FP16
        fp16_exp = static_cast<uint32_t>(exponent + 15);
        uint32_t fp16_mantissa = mantissa >> 13;
        // Round to nearest even
        uint32_t remainder = mantissa & 0x1FFF;
        if (remainder > 0x1000 || (remainder == 0x1000 && (fp16_mantissa & 1))) {
            fp16_mantissa++;
            if (fp16_mantissa > 0x3FF) {
                fp16_mantissa = 0;
                fp16_exp++;
            }
        }
        return static_cast<uint16_t>((sign << 15) | (fp16_exp << 10) | (fp16_mantissa & 0x3FF));
    } else if (exponent >= 128) {
        // NaN or Inf (exponent field is 0xFF in the float)
        if (mantissa == 0) {
            return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
        } else {
            return static_cast<uint16_t>((sign << 15) | (0x1F << 10) | 0x200 | (mantissa >> 13));
        }
    } else {
        // Too large, overflow to infinity
        return static_cast<uint16_t>((sign << 15) | (0x1F << 10));
    }
}

inline float fp16_to_float_scalar(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t fp16_exp = (h >> 10) & 0x1F;
    uint32_t fp16_mantissa = h & 0x3FF;
    uint32_t float_bits;

    if (fp16_exp == 0) {
        // Zero or subnormal
        if (fp16_mantissa == 0) {
            float_bits = sign << 31;
        } else {
            // Subnormal: normalize
            uint32_t mantissa = fp16_mantissa;
            int32_t shift = 0;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                shift--;
            }
            mantissa &= 0x3FF;  // Remove the implicit leading 1
            float_bits = (sign << 31) | ((127 + (-14) + shift) << 23) | (mantissa << 13);
        }
    } else if (fp16_exp == 0x1F) {
        // Inf or NaN
        if (fp16_mantissa == 0) {
            float_bits = (sign << 31) | (0xFF << 23);
        } else {
            float_bits = (sign << 31) | (0xFF << 23) | 0x7FFFFF;
        }
    } else {
        // Normal number
        float_bits = (sign << 31) | ((fp16_exp + 127 - 15) << 23) | (fp16_mantissa << 13);
    }

    float result;
    std::memcpy(&result, &float_bits, sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Public API: float_to_fp16, fp16_to_float, floats_to_fp16, fp16_to_floats
// ---------------------------------------------------------------------------
// Runtime dispatch via deglib::cpu::has_avx2().
// On GCC/Clang: uses DEGLIB_TARGET_AVX2-attributed intrinsics.
// On MSVC: uses _mm_cvtps_ph / _mm_cvtph_ps (SSE intrinsics).
// Fallback: scalar IEEE 754 Round-to-Nearest-Even.
// ---------------------------------------------------------------------------

inline uint16_t float_to_fp16(float f) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx2()) {
    #if defined(__GNUC__) || defined(__clang__)
        return float_to_fp16_gcc(f);
    #elif defined(_MSC_VER)
        return float_to_fp16_msvc(f);
    #endif
    }
#endif
    return float_to_fp16_scalar(f);
}

inline float fp16_to_float(uint16_t h) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx2()) {
    #if defined(__GNUC__) || defined(__clang__)
        return fp16_to_float_gcc(h);
    #elif defined(_MSC_VER)
        return fp16_to_float_msvc(h);
    #endif
    }
#endif
    return fp16_to_float_scalar(h);
}

inline void floats_to_fp16(const float* floats, uint16_t* fp16_vals, size_t count) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx2()) {
    #if defined(__GNUC__) || defined(__clang__)
        floats_to_fp16_gcc(floats, fp16_vals, count);
        return;
    #elif defined(_MSC_VER)
        floats_to_fp16_msvc(floats, fp16_vals, count);
        return;
    #endif
    }
#endif
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        fp16_vals[i] = float_to_fp16_scalar(floats[i]);
    }
}

inline void fp16_to_floats(const uint16_t* fp16_vals, float* floats, size_t count) {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx2()) {
    #if defined(__GNUC__) || defined(__clang__)
        fp16_to_floats_gcc(fp16_vals, floats, count);
        return;
    #elif defined(_MSC_VER)
        fp16_to_floats_msvc(fp16_vals, floats, count);
        return;
    #endif
    }
#endif
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        floats[i] = fp16_to_float_scalar(fp16_vals[i]);
    }
}

/**
 * Computes centered vector (fp16 - mean) in float, and populates abs_vals for sorting.
 */
inline void center_and_abs(
    const uint16_t* src_fp16,
    const float* mean,
    uint32_t dim,
    float* out_centered,
    std::pair<float, uint32_t>* out_abs_vals
) {
    uint32_t d = 0;
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx2()) {
        for (; d + 8 <= dim; d += 8) {
            __m128i vhp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_fp16 + d));
            __m256 val = _mm256_cvtph_ps(vhp);
            __m256 m = _mm256_loadu_ps(mean + d);
            __m256 diff = _mm256_sub_ps(val, m);
            _mm256_storeu_ps(out_centered + d, diff);
            __m256 abs_diff = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), diff);
            alignas(32) float abs_arr[8];
            _mm256_store_ps(abs_arr, abs_diff);
            for (int k = 0; k < 8; ++k) {
                out_abs_vals[d + k] = {abs_arr[k], d + k};
            }
        }
    }
#endif
    for (; d < dim; ++d) {
        float diff = fp16_to_float(src_fp16[d]) - mean[d];
        out_centered[d] = diff;
        out_abs_vals[d] = {std::abs(diff), d};
    }
}

/**
 * Directly subtracts mean_fp16 from src_fp16 in native FP16 hardware without converting to float,
 * writes the FP16 difference into out_diff_fp16, and writes {abs_val (uint16_t), index} pairs.
 */
inline void sub_and_abs_fp16(
    const uint16_t* src_fp16,
    const uint16_t* mean_fp16,
    uint32_t dim,
    uint16_t* out_diff_fp16,
    std::pair<uint16_t, uint32_t>* out_abs_vals
) {
    uint32_t d = 0;
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx2()) {
        for (; d + 8 <= dim; d += 8) {
            __m128i a_h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_fp16 + d));
            __m128i b_h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mean_fp16 + d));
            __m256 a_f = _mm256_cvtph_ps(a_h);
            __m256 b_f = _mm256_cvtph_ps(b_h);
            __m256 diff_f = _mm256_sub_ps(a_f, b_f);
            __m128i diff_h = _mm256_cvtps_ph(diff_f, 0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out_diff_fp16 + d), diff_h);

            __m128i abs_h = _mm_and_si128(diff_h, _mm_set1_epi16(0x7fff));
            alignas(16) uint16_t abs_arr[8];
            _mm_store_si128(reinterpret_cast<__m128i*>(abs_arr), abs_h);
            for (int k = 0; k < 8; ++k) {
                out_abs_vals[d + k] = {abs_arr[k], d + k};
            }
        }
    }
#endif
    for (; d < dim; ++d) {
        float diff = fp16_to_float(src_fp16[d]) - fp16_to_float(mean_fp16[d]);
        uint16_t h = float_to_fp16(diff);
        out_diff_fp16[d] = h;
        out_abs_vals[d] = {static_cast<uint16_t>(h & 0x7FFFu), d};
    }
}

/**
 * Computes dimension-wise (column-wise) mean of a batch of FP16 vectors.
 * Accumulates directly from FP16 in SIMD registers into per-thread sums without
 * allocating intermediate float arrays and without any mutex locks.
 */
inline void compute_column_mean(const uint16_t* src_fp16, size_t count, uint32_t dim, float* out_mean, size_t numThreads = 0) {
    if (count == 0 || dim == 0 || src_fp16 == nullptr || out_mean == nullptr) return;

    const size_t actual_threads = (numThreads == 0) ? std::clamp(static_cast<size_t>(std::thread::hardware_concurrency()), size_t{1}, size_t{64}) : numThreads;
    std::vector<double> thread_sums(actual_threads * dim, 0.0);

    deglib::concurrent::parallel_batch_for(static_cast<size_t>(0), count, actual_threads, [&](size_t begin, size_t end, size_t thread_id) {
        double* my_sum = thread_sums.data() + thread_id * dim;
#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx2()) {
            for (size_t i = begin; i < end; ++i) {
                const uint16_t* vec = src_fp16 + i * dim;
                uint32_t d = 0;
                for (; d + 8 <= dim; d += 8) {
                    __m128i vhp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vec + d));
                    __m256 va = _mm256_cvtph_ps(vhp);
                    alignas(32) float f_vals[8];
                    _mm256_store_ps(f_vals, va);
                    for (int k = 0; k < 8; ++k) {
                        my_sum[d + k] += f_vals[k];
                    }
                }
                for (; d < dim; ++d) {
                    my_sum[d] += fp16_to_float(vec[d]);
                }
            }
            return;
        }
#endif
        for (size_t i = begin; i < end; ++i) {
            const uint16_t* vec = src_fp16 + i * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                my_sum[d] += fp16_to_float(vec[d]);
            }
        }
    });

    const double inv_count = 1.0 / static_cast<double>(count);
    for (uint32_t d = 0; d < dim; ++d) {
        double total = 0.0;
        for (size_t t = 0; t < actual_threads; ++t) {
            total += thread_sums[t * dim + d];
        }
        out_mean[d] = static_cast<float>(total * inv_count);
    }
}

}  // namespace fp16

}  // end namespace deglib::distances
