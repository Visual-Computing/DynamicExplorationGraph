#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include "deglib/config.h"

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

        DEGLIB_TARGET_AVX2 inline uint16_t float_to_fp16_gcc(float f) {
            return _cvtss_sh(f, 0);
        }

        DEGLIB_TARGET_AVX2 inline float fp16_to_float_gcc(uint16_t h) {
            return _cvtsh_ss(h);
        }

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

#endif // defined(DEGLIB_X86) && (defined(__GNUC__) || defined(__clang__))

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
            // Process 4 floats per step with _mm_cvtps_ph
            for (; i + 4 <= count; i += 4) {
                __m128 va = _mm_loadu_ps(floats + i);
                __m128i vhp = _mm_cvtps_ph(va, 0);
                // Store 4 uint16_t values from the __m128i
                alignas(16) uint16_t temp[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(temp), vhp);
                fp16_vals[i] = temp[0];
                fp16_vals[i + 1] = temp[1];
                fp16_vals[i + 2] = temp[2];
                fp16_vals[i + 3] = temp[3];
            }
            // Scalar fallback for remaining 0-3 elements
            for (; i < count; ++i) {
                fp16_vals[i] = float_to_fp16_msvc(floats[i]);
            }
        }

        inline void fp16_to_floats_msvc(const uint16_t* fp16_vals, float* floats, size_t count) {
            size_t i = 0;
            // Process 4 uint16_t per step with _mm_cvtph_ps
            for (; i + 4 <= count; i += 4) {
                // Load 4 uint16_t values into __m128i
                alignas(16) uint16_t temp[4] = {fp16_vals[i], fp16_vals[i + 1], fp16_vals[i + 2], fp16_vals[i + 3]};
                __m128i vhp = _mm_load_si128(reinterpret_cast<const __m128i*>(temp));
                __m128 va = _mm_cvtph_ps(vhp);
                _mm_storeu_ps(floats + i, va);
            }
            // Scalar fallback for remaining 0-3 elements
            for (; i < count; ++i) {
                floats[i] = fp16_to_float_msvc(fp16_vals[i]);
            }
        }

#endif // defined(DEGLIB_X86) && defined(_MSC_VER)

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
                if (remainder > (1u << (22 + shift)) ||
                    (remainder == (1u << (22 + shift)) && (fp16_mantissa & 1))) {
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
                if (remainder > 0x1000 ||
                    (remainder == 0x1000 && (fp16_mantissa & 1))) {
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
                    mantissa &= 0x3FF; // Remove the implicit leading 1
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

    } // namespace fp16

} // end namespace deglib::distances
