#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <config.h>

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <distance/fp16.h>

namespace deglib {
namespace distances {

// =============================================================================
// L2 (Euclidean squared) distance for float16 vectors stored as uint16_t.
//
// Single cascading SIMD template: FP16L2<MIN_ALIGN>
//   Always cascades from widest available SIMD down to scalar:
//     AVX-512 (32 fp16 -> 32 fp32) -> AVX (16 fp16) -> SSE (8 fp16) -> scalar
//
// Strategy: load N uint16_t elements, convert to float via _mm_cvtph_ps /
//   _mm256_cvtph_ps / _mm512_cvtph_ps, compute squared differences, accumulate.
//
// MIN_ALIGN specifies minimum guaranteed element-count alignment:
//   32 -> dim%32==0, 16 -> dim%16==0, 8 -> dim%8==0, 1 -> no guarantee
//
// compare() returns sum of squared differences in FP32.
//
// if constexpr eliminates dead tail code at compile time.
//
// Backward-compatible aliases:
//   FP16L2Ext32, FP16L2Ext16, FP16L2Ext8, FP16L2Default
// =============================================================================


// -----------------------------------------------------------------------------
// SIMD horizontal-sum helpers (named fp16l2_* to avoid collisions)
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static inline float fp16l2_hsum128(__m128 v) {
    alignas(16) float f[4];
    _mm_store_ps(f, v);
    return f[0] + f[1] + f[2] + f[3];
}
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
static inline float fp16l2_hsum256(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return fp16l2_hsum128(_mm_add_ps(lo, hi));
}
#endif

#if defined(USE_AVX512)
static inline float fp16l2_hsum512(__m512 v) {
    __m256 lo = _mm512_extractf32x8_ps(v, 0);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return fp16l2_hsum256(_mm256_add_ps(lo, hi));
}
#endif


// =============================================================================
// FP16L2<MIN_ALIGN> — cascading SIMD FP16 squared L2 distance
// =============================================================================

/**
 * Template parameter MIN_ALIGN (element alignment, not byte alignment):
 *   32 -> dim%32==0  (widest SIMD loop, no tail)
 *   16 -> dim%16==0
 *    8 -> dim%8==0
 *    1 -> arbitrary dimension
 *
 * compare() returns squared Euclidean distance computed in FP32 after
 * expanding both FP16 inputs element-wise.
 *
 * Equivalent to:
 *   sum_i (fp16_to_float(a[i]) - fp16_to_float(b[i]))^2
 */
template <size_t MIN_ALIGN>
class FP16L2 {
public:
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const uint16_t* b = (const uint16_t*)pVect2v;
        size_t size = *((const size_t*)qty_ptr);
        const uint16_t* last = a + size;

#if defined(USE_AVX512)
        // ---- AVX-512 main loop: 32 fp16 per iteration ----
        __m512 sum512 = _mm512_setzero_ps();
        while (a + 32 <= last) {
            __m256i ra_lo = _mm256_loadu_si256((const __m256i*)a);
            __m256i ra_hi = _mm256_loadu_si256((const __m256i*)(a + 16));
            __m256i rb_lo = _mm256_loadu_si256((const __m256i*)b);
            __m256i rb_hi = _mm256_loadu_si256((const __m256i*)(b + 16));
            __m512 fa_lo = _mm512_cvtph_ps(ra_lo);
            __m512 fa_hi = _mm512_cvtph_ps(ra_hi);
            __m512 fb_lo = _mm512_cvtph_ps(rb_lo);
            __m512 fb_hi = _mm512_cvtph_ps(rb_hi);
            __m512 d_lo = _mm512_sub_ps(fa_lo, fb_lo);
            __m512 d_hi = _mm512_sub_ps(fa_hi, fb_hi);
            sum512 = _mm512_fmadd_ps(d_lo, d_lo, sum512);
            sum512 = _mm512_fmadd_ps(d_hi, d_hi, sum512);
            a += 32; b += 32;
        }
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0),
                                       _mm512_extractf32x8_ps(sum512, 1));

        // ---- AVX tail: 16 fp16 ----
        if constexpr (MIN_ALIGN < 32) {
            while (a + 16 <= last) {
                __m128i ra_lo = _mm_loadu_si128((const __m128i*)a);
                __m128i ra_hi = _mm_loadu_si128((const __m128i*)(a + 8));
                __m128i rb_lo = _mm_loadu_si128((const __m128i*)b);
                __m128i rb_hi = _mm_loadu_si128((const __m128i*)(b + 8));
                __m256 fa_lo = _mm256_cvtph_ps(ra_lo);
                __m256 fa_hi = _mm256_cvtph_ps(ra_hi);
                __m256 fb_lo = _mm256_cvtph_ps(rb_lo);
                __m256 fb_hi = _mm256_cvtph_ps(rb_hi);
                __m256 d_lo = _mm256_sub_ps(fa_lo, fb_lo);
                __m256 d_hi = _mm256_sub_ps(fa_hi, fb_hi);
                sum256 = _mm256_fmadd_ps(d_lo, d_lo, sum256);
                sum256 = _mm256_fmadd_ps(d_hi, d_hi, sum256);
                a += 16; b += 16;
            }
        }

        // ---- SSE tail: 8 fp16 ----
        float result = fp16l2_hsum256(sum256);
        if constexpr (MIN_ALIGN < 16) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 8 <= last) {
                __m128i ra = _mm_loadu_si128((const __m128i*)a);
                __m128i rb = _mm_loadu_si128((const __m128i*)b);
                __m128 fa_lo = _mm_cvtph_ps(ra);
                __m128 fa_hi = _mm_cvtph_ps(_mm_srli_si128(ra, 8));
                __m128 fb_lo = _mm_cvtph_ps(rb);
                __m128 fb_hi = _mm_cvtph_ps(_mm_srli_si128(rb, 8));
                __m128 d_lo = _mm_sub_ps(fa_lo, fb_lo);
                __m128 d_hi = _mm_sub_ps(fa_hi, fb_hi);
                sum128 = _mm_fmadd_ps(d_lo, d_lo, sum128);
                sum128 = _mm_fmadd_ps(d_hi, d_hi, sum128);
                a += 8; b += 8;
            }
            result += fp16l2_hsum128(sum128);
        }

        // ---- Scalar tail ----
        if constexpr (MIN_ALIGN < 8) {
            while (a < last) {
                float fa = fp16_to_float(*a++);
                float fb = fp16_to_float(*b++);
                float d = fa - fb;
                result += d * d;
            }
        }
        return result;

#elif defined(USE_AVX)
        // ---- AVX main loop: 16 fp16 per iteration ----
        __m256 sum256 = _mm256_setzero_ps();
        while (a + 16 <= last) {
            __m128i ra_lo = _mm_loadu_si128((const __m128i*)a);
            __m128i ra_hi = _mm_loadu_si128((const __m128i*)(a + 8));
            __m128i rb_lo = _mm_loadu_si128((const __m128i*)b);
            __m128i rb_hi = _mm_loadu_si128((const __m128i*)(b + 8));
            __m256 fa_lo = _mm256_cvtph_ps(ra_lo);
            __m256 fa_hi = _mm256_cvtph_ps(ra_hi);
            __m256 fb_lo = _mm256_cvtph_ps(rb_lo);
            __m256 fb_hi = _mm256_cvtph_ps(rb_hi);
            __m256 d_lo = _mm256_sub_ps(fa_lo, fb_lo);
            __m256 d_hi = _mm256_sub_ps(fa_hi, fb_hi);
            sum256 = _mm256_fmadd_ps(d_lo, d_lo, sum256);
            sum256 = _mm256_fmadd_ps(d_hi, d_hi, sum256);
            a += 16; b += 16;
        }

        // ---- SSE tail: 8 fp16 ----
        float result = fp16l2_hsum256(sum256);
        if constexpr (MIN_ALIGN < 16) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 8 <= last) {
                __m128i ra = _mm_loadu_si128((const __m128i*)a);
                __m128i rb = _mm_loadu_si128((const __m128i*)b);
                __m128 fa_lo = _mm_cvtph_ps(ra);
                __m128 fa_hi = _mm_cvtph_ps(_mm_srli_si128(ra, 8));
                __m128 fb_lo = _mm_cvtph_ps(rb);
                __m128 fb_hi = _mm_cvtph_ps(_mm_srli_si128(rb, 8));
                __m128 d_lo = _mm_sub_ps(fa_lo, fb_lo);
                __m128 d_hi = _mm_sub_ps(fa_hi, fb_hi);
                sum128 = _mm_fmadd_ps(d_lo, d_lo, sum128);
                sum128 = _mm_fmadd_ps(d_hi, d_hi, sum128);
                a += 8; b += 8;
            }
            result += fp16l2_hsum128(sum128);
        }

        // ---- Scalar tail ----
        if constexpr (MIN_ALIGN < 8) {
            while (a < last) {
                float fa = fp16_to_float(*a++);
                float fb = fp16_to_float(*b++);
                float d = fa - fb;
                result += d * d;
            }
        }
        return result;

#elif defined(USE_SSE)
        // ---- SSE main loop: 8 fp16 per iteration ----
        __m128 sum128 = _mm_setzero_ps();
        while (a + 8 <= last) {
            __m128i ra = _mm_loadu_si128((const __m128i*)a);
            __m128i rb = _mm_loadu_si128((const __m128i*)b);
            __m128 fa_lo = _mm_cvtph_ps(ra);
            __m128 fa_hi = _mm_cvtph_ps(_mm_srli_si128(ra, 8));
            __m128 fb_lo = _mm_cvtph_ps(rb);
            __m128 fb_hi = _mm_cvtph_ps(_mm_srli_si128(rb, 8));
            __m128 d_lo = _mm_sub_ps(fa_lo, fb_lo);
            __m128 d_hi = _mm_sub_ps(fa_hi, fb_hi);
            sum128 = _mm_fmadd_ps(d_lo, d_lo, sum128);
            sum128 = _mm_fmadd_ps(d_hi, d_hi, sum128);
            a += 8; b += 8;
        }

        float result = fp16l2_hsum128(sum128);
        // ---- Scalar tail ----
        if constexpr (MIN_ALIGN < 8) {
            while (a < last) {
                float fa = fp16_to_float(*a++);
                float fb = fp16_to_float(*b++);
                float d = fa - fb;
                result += d * d;
            }
        }
        return result;

#else
        // ---- Pure scalar fallback ----
        float result = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float fa = fp16_to_float(a[i]);
            float fb = fp16_to_float(b[i]);
            float d = fa - fb;
            result += d * d;
        }
        return result;
#endif
    }

private:
    // -------------------------------------------------------------------------
    // Tail helper: residual FP16 elements that do not fill a full SIMD chunk
    // -------------------------------------------------------------------------
    inline static float compare_tail(const uint16_t* a, const uint16_t* b, size_t tail_elems) {
        float result = 0.0f;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        if (tail_elems >= 8) {
            __m128i ra = _mm_loadu_si128((const __m128i*)a);
            __m128i rb = _mm_loadu_si128((const __m128i*)b);
            __m128 fa_lo = _mm_cvtph_ps(ra);
            __m128 fa_hi = _mm_cvtph_ps(_mm_srli_si128(ra, 8));
            __m128 fb_lo = _mm_cvtph_ps(rb);
            __m128 fb_hi = _mm_cvtph_ps(_mm_srli_si128(rb, 8));
            __m128 d_lo = _mm_sub_ps(fa_lo, fb_lo);
            __m128 d_hi = _mm_sub_ps(fa_hi, fb_hi);
            __m128 s = _mm_fmadd_ps(d_lo, d_lo, _mm_fmadd_ps(d_hi, d_hi, _mm_setzero_ps()));
            result = fp16l2_hsum128(s);
            a += 8; b += 8;
            tail_elems -= 8;
        }
#endif
        for (size_t i = 0; i < tail_elems; ++i) {
            float fa = fp16_to_float(a[i]);
            float fb = fp16_to_float(b[i]);
            float d = fa - fb;
            result += d * d;
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Batch helpers: compute squared L2 from one query to N db vectors
    // -------------------------------------------------------------------------

#if defined(USE_AVX) || defined(USE_AVX512)
    inline static void batch8_avx(const uint16_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 16;
        const size_t tail_start = nc * 16;
        const size_t tail_dim = dim - tail_start;

        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps(), s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps(), s7 = _mm256_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            __m128i q_raw_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
            __m128i q_raw_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
            __m256 q_lo = _mm256_cvtph_ps(q_raw_lo);
            __m256 q_hi = _mm256_cvtph_ps(q_raw_hi);

            #define FP16L2_B8_AVX(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[c * 16]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[c * 16 + 8]); \
                __m256 d_lo_ = _mm256_sub_ps(q_lo, _mm256_cvtph_ps(r_lo_)); \
                __m256 d_hi_ = _mm256_sub_ps(q_hi, _mm256_cvtph_ps(r_hi_)); \
                reg = _mm256_fmadd_ps(d_lo_, d_lo_, reg); \
                reg = _mm256_fmadd_ps(d_hi_, d_hi_, reg); \
            } while(0)

            FP16L2_B8_AVX(0, s0); FP16L2_B8_AVX(1, s1);
            FP16L2_B8_AVX(2, s2); FP16L2_B8_AVX(3, s3);
            FP16L2_B8_AVX(4, s4); FP16L2_B8_AVX(5, s5);
            FP16L2_B8_AVX(6, s6); FP16L2_B8_AVX(7, s7);
            #undef FP16L2_B8_AVX
        }

        dists[0] = fp16l2_hsum256(s0); dists[1] = fp16l2_hsum256(s1);
        dists[2] = fp16l2_hsum256(s2); dists[3] = fp16l2_hsum256(s3);
        dists[4] = fp16l2_hsum256(s4); dists[5] = fp16l2_hsum256(s5);
        dists[6] = fp16l2_hsum256(s6); dists[7] = fp16l2_hsum256(s7);

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 8; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += compare_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    inline static void batch4_avx(const uint16_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 16;
        const size_t tail_start = nc * 16;
        const size_t tail_dim = dim - tail_start;

        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            __m128i q_raw_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
            __m128i q_raw_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
            __m256 q_lo = _mm256_cvtph_ps(q_raw_lo);
            __m256 q_hi = _mm256_cvtph_ps(q_raw_hi);

            #define FP16L2_B4_AVX(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[c * 16]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[c * 16 + 8]); \
                __m256 d_lo_ = _mm256_sub_ps(q_lo, _mm256_cvtph_ps(r_lo_)); \
                __m256 d_hi_ = _mm256_sub_ps(q_hi, _mm256_cvtph_ps(r_hi_)); \
                reg = _mm256_fmadd_ps(d_lo_, d_lo_, reg); \
                reg = _mm256_fmadd_ps(d_hi_, d_hi_, reg); \
            } while(0)

            FP16L2_B4_AVX(0, s0); FP16L2_B4_AVX(1, s1);
            FP16L2_B4_AVX(2, s2); FP16L2_B4_AVX(3, s3);
            #undef FP16L2_B4_AVX
        }

        dists[0] = fp16l2_hsum256(s0); dists[1] = fp16l2_hsum256(s1);
        dists[2] = fp16l2_hsum256(s2); dists[3] = fp16l2_hsum256(s3);

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 4; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += compare_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_AVX512)
    inline static void batch8_avx512(const uint16_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc32 = dim / 32;
        const size_t tail16_start = nc32 * 32;
        const size_t tail_start = (dim / 16) * 16;
        const size_t tail_dim = dim - tail_start;

        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps(), s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps(), s7 = _mm512_setzero_ps();

        for (size_t c = 0; c < nc32; ++c) {
            __m256i q_raw_lo = _mm256_loadu_si256((const __m256i*)&query[c * 32]);
            __m256i q_raw_hi = _mm256_loadu_si256((const __m256i*)&query[c * 32 + 16]);
            __m512 q_lo = _mm512_cvtph_ps(q_raw_lo);
            __m512 q_hi = _mm512_cvtph_ps(q_raw_hi);

            #define FP16L2_B8_512(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m256i r_lo_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32]); \
                __m256i r_hi_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32 + 16]); \
                __m512 d_lo_ = _mm512_sub_ps(q_lo, _mm512_cvtph_ps(r_lo_)); \
                __m512 d_hi_ = _mm512_sub_ps(q_hi, _mm512_cvtph_ps(r_hi_)); \
                reg = _mm512_fmadd_ps(d_lo_, d_lo_, reg); \
                reg = _mm512_fmadd_ps(d_hi_, d_hi_, reg); \
            } while(0)

            FP16L2_B8_512(0, s0); FP16L2_B8_512(1, s1);
            FP16L2_B8_512(2, s2); FP16L2_B8_512(3, s3);
            FP16L2_B8_512(4, s4); FP16L2_B8_512(5, s5);
            FP16L2_B8_512(6, s6); FP16L2_B8_512(7, s7);
            #undef FP16L2_B8_512
        }

        dists[0] = fp16l2_hsum512(s0); dists[1] = fp16l2_hsum512(s1);
        dists[2] = fp16l2_hsum512(s2); dists[3] = fp16l2_hsum512(s3);
        dists[4] = fp16l2_hsum512(s4); dists[5] = fp16l2_hsum512(s5);
        dists[6] = fp16l2_hsum512(s6); dists[7] = fp16l2_hsum512(s7);

        // AVX-512 remainder: up to 1 AVX chunk (16 fp16) between tail16_start and tail_start
        if (tail16_start + 16 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();
            __m256 t4 = _mm256_setzero_ps(), t5 = _mm256_setzero_ps();
            __m256 t6 = _mm256_setzero_ps(), t7 = _mm256_setzero_ps();

            __m128i q16_lo = _mm_loadu_si128((const __m128i*)&query[tail16_start]);
            __m128i q16_hi = _mm_loadu_si128((const __m128i*)&query[tail16_start + 8]);
            __m256 qf_lo = _mm256_cvtph_ps(q16_lo);
            __m256 qf_hi = _mm256_cvtph_ps(q16_hi);

            #define FP16L2_B8_AVX8(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[tail16_start]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[tail16_start + 8]); \
                __m256 d_lo_ = _mm256_sub_ps(qf_lo, _mm256_cvtph_ps(r_lo_)); \
                __m256 d_hi_ = _mm256_sub_ps(qf_hi, _mm256_cvtph_ps(r_hi_)); \
                reg = _mm256_fmadd_ps(d_lo_, d_lo_, reg); \
                reg = _mm256_fmadd_ps(d_hi_, d_hi_, reg); \
            } while(0)

            FP16L2_B8_AVX8(0, t0); FP16L2_B8_AVX8(1, t1);
            FP16L2_B8_AVX8(2, t2); FP16L2_B8_AVX8(3, t3);
            FP16L2_B8_AVX8(4, t4); FP16L2_B8_AVX8(5, t5);
            FP16L2_B8_AVX8(6, t6); FP16L2_B8_AVX8(7, t7);
            #undef FP16L2_B8_AVX8

            dists[0] += fp16l2_hsum256(t0); dists[1] += fp16l2_hsum256(t1);
            dists[2] += fp16l2_hsum256(t2); dists[3] += fp16l2_hsum256(t3);
            dists[4] += fp16l2_hsum256(t4); dists[5] += fp16l2_hsum256(t5);
            dists[6] += fp16l2_hsum256(t6); dists[7] += fp16l2_hsum256(t7);
        }

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 8; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += compare_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    inline static void batch4_avx512(const uint16_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc32 = dim / 32;
        const size_t tail16_start = nc32 * 32;
        const size_t tail_start = (dim / 16) * 16;
        const size_t tail_dim = dim - tail_start;

        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();

        for (size_t c = 0; c < nc32; ++c) {
            __m256i q_raw_lo = _mm256_loadu_si256((const __m256i*)&query[c * 32]);
            __m256i q_raw_hi = _mm256_loadu_si256((const __m256i*)&query[c * 32 + 16]);
            __m512 q_lo = _mm512_cvtph_ps(q_raw_lo);
            __m512 q_hi = _mm512_cvtph_ps(q_raw_hi);

            #define FP16L2_B4_512(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m256i r_lo_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32]); \
                __m256i r_hi_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32 + 16]); \
                __m512 d_lo_ = _mm512_sub_ps(q_lo, _mm512_cvtph_ps(r_lo_)); \
                __m512 d_hi_ = _mm512_sub_ps(q_hi, _mm512_cvtph_ps(r_hi_)); \
                reg = _mm512_fmadd_ps(d_lo_, d_lo_, reg); \
                reg = _mm512_fmadd_ps(d_hi_, d_hi_, reg); \
            } while(0)

            FP16L2_B4_512(0, s0); FP16L2_B4_512(1, s1);
            FP16L2_B4_512(2, s2); FP16L2_B4_512(3, s3);
            #undef FP16L2_B4_512
        }

        dists[0] = fp16l2_hsum512(s0); dists[1] = fp16l2_hsum512(s1);
        dists[2] = fp16l2_hsum512(s2); dists[3] = fp16l2_hsum512(s3);

        if (tail16_start + 16 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();

            __m128i q16_lo = _mm_loadu_si128((const __m128i*)&query[tail16_start]);
            __m128i q16_hi = _mm_loadu_si128((const __m128i*)&query[tail16_start + 8]);
            __m256 qf_lo = _mm256_cvtph_ps(q16_lo);
            __m256 qf_hi = _mm256_cvtph_ps(q16_hi);

            #define FP16L2_B4_AVX8(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[tail16_start]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[tail16_start + 8]); \
                __m256 d_lo_ = _mm256_sub_ps(qf_lo, _mm256_cvtph_ps(r_lo_)); \
                __m256 d_hi_ = _mm256_sub_ps(qf_hi, _mm256_cvtph_ps(r_hi_)); \
                reg = _mm256_fmadd_ps(d_lo_, d_lo_, reg); \
                reg = _mm256_fmadd_ps(d_hi_, d_hi_, reg); \
            } while(0)

            FP16L2_B4_AVX8(0, t0); FP16L2_B4_AVX8(1, t1);
            FP16L2_B4_AVX8(2, t2); FP16L2_B4_AVX8(3, t3);
            #undef FP16L2_B4_AVX8

            dists[0] += fp16l2_hsum256(t0); dists[1] += fp16l2_hsum256(t1);
            dists[2] += fp16l2_hsum256(t2); dists[3] += fp16l2_hsum256(t3);
        }

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 4; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += compare_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    inline static void batch4_sse(const uint16_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 8;
        const size_t tail_start = nc * 8;
        const size_t tail_dim = dim - tail_start;

        __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
        __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            __m128i q_raw = _mm_loadu_si128((const __m128i*)&query[c * 8]);
            __m128 q_lo = _mm_cvtph_ps(q_raw);
            __m128 q_hi = _mm_cvtph_ps(_mm_srli_si128(q_raw, 8));

            #define FP16L2_B4_SSE(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_raw_ = _mm_loadu_si128((const __m128i*)&db_[c * 8]); \
                __m128 d_lo_ = _mm_sub_ps(q_lo, _mm_cvtph_ps(r_raw_)); \
                __m128 d_hi_ = _mm_sub_ps(q_hi, _mm_cvtph_ps(_mm_srli_si128(r_raw_, 8))); \
                reg = _mm_fmadd_ps(d_lo_, d_lo_, reg); \
                reg = _mm_fmadd_ps(d_hi_, d_hi_, reg); \
            } while(0)

            FP16L2_B4_SSE(0, s0); FP16L2_B4_SSE(1, s1);
            FP16L2_B4_SSE(2, s2); FP16L2_B4_SSE(3, s3);
            #undef FP16L2_B4_SSE
        }

        dists[0] = fp16l2_hsum128(s0); dists[1] = fp16l2_hsum128(s1);
        dists[2] = fp16l2_hsum128(s2); dists[3] = fp16l2_hsum128(s3);

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 4; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += compare_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

public:
    /**
     * Batch squared FP16 L2 distance: one query to many database vectors.
     *
     * Uses AVX-512 -> AVX -> SSE batch helpers, then falls back to compare()
     * for any remainder not fitting a complete batch slot.
     */
    inline static void compare_batch(const void* query_ptr, const void* const* db_arr,
                                     size_t count, const void* qty_ptr, float* dists) {
        const uint16_t* query = (const uint16_t*)query_ptr;
        const size_t dim = *((const size_t*)qty_ptr);

        size_t i = 0;
#if defined(USE_AVX512)
        for (; i + 8 <= count; i += 8)
            batch8_avx512(query, &db_arr[i], dim, &dists[i]);
        if (i + 4 <= count) {
            batch4_avx512(query, &db_arr[i], dim, &dists[i]);
            i += 4;
        }
#elif defined(USE_AVX)
        for (; i + 8 <= count; i += 8)
            batch8_avx(query, &db_arr[i], dim, &dists[i]);
        if (i + 4 <= count) {
            batch4_avx(query, &db_arr[i], dim, &dists[i]);
            i += 4;
        }
#else
        for (; i + 4 <= count; i += 4)
            batch4_sse(query, &db_arr[i], dim, &dists[i]);
#endif
        for (; i < count; ++i)
            dists[i] = compare(query, db_arr[i], qty_ptr);
    }
};


// =============================================================================
// Backward-compatible aliases
// =============================================================================

using FP16L2Ext32   = FP16L2<32>;
using FP16L2Ext16   = FP16L2<16>;
using FP16L2Ext8    = FP16L2<8>;
using FP16L2Default = FP16L2<1>;

} // namespace distances
} // namespace deglib
