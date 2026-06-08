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
// FP16 Inner Product (dot product) for float16 vectors.
//
// Single cascading SIMD template: FP16InnerProduct<MIN_ALIGN>
//   Always cascades from widest available SIMD down to scalar:
//     AVX-512 (32 fp16) -> AVX (16 fp16) -> SSE (8 fp16) -> scalar
//
// Each SIMD iteration loads uint16_t elements, converts them to float via
// _mm_cvtph_ps, then multiplies and accumulates.  Both halves of a 16-element
// AVX chunk fold into one __m256 accumulator, so register pressure is the same
// as for FP32: 8 vectors fit in 8 __m256 registers.
//
// MIN_ALIGN specifies the minimum guaranteed element count alignment:
//   32 -> dim%32==0, 16 -> dim%16==0, 8 -> dim%8==0, 1 -> no guarantee
//
// compare() returns 1.f - dot(a, b)  (distance metric: 1 - similarity)
// dot()     returns the raw dot product
//
// if constexpr eliminates dead tail code at compile time.
//
// Backward-compatible aliases:
//   FP16InnerProductExt32, FP16InnerProductExt16, FP16InnerProductExt8,
//   FP16InnerProduct
// =============================================================================


// -----------------------------------------------------------------------------
// SIMD horizontal-sum helpers
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static inline float fp16_hsum128(__m128 v) {
    alignas(16) float f[4];
    _mm_store_ps(f, v);
    return f[0] + f[1] + f[2] + f[3];
}
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
static inline float fp16_hsum256(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return fp16_hsum128(_mm_add_ps(lo, hi));
}
#endif

#if defined(USE_AVX512)
static inline float fp16_hsum512(__m512 v) {
    __m256 lo = _mm512_extractf32x8_ps(v, 0);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return fp16_hsum256(_mm256_add_ps(lo, hi));
}
#endif


// -----------------------------------------------------------------------------
// SIMD iteration macros for FP16 inner product accumulation
//
// Each iteration loads N uint16 elements, converts to float in two halves,
// and accumulates both halves into a single SIMD register.
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#define FP16_SSE_ITER(a, b, sum) do {                                      \
    __m128i raw_a_ = _mm_loadu_si128((const __m128i*)(a));                 \
    __m128  a_lo_  = _mm_cvtph_ps(raw_a_);                                 \
    __m128  a_hi_  = _mm_cvtph_ps(_mm_srli_si128(raw_a_, 8));             \
    __m128i raw_b_ = _mm_loadu_si128((const __m128i*)(b));                 \
    __m128  b_lo_  = _mm_cvtph_ps(raw_b_);                                 \
    __m128  b_hi_  = _mm_cvtph_ps(_mm_srli_si128(raw_b_, 8));             \
    (sum) = _mm_fmadd_ps(a_lo_, b_lo_, (sum));                             \
    (sum) = _mm_fmadd_ps(a_hi_, b_hi_, (sum));                             \
    (a) += 8; (b) += 8;                                                    \
} while(0)
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
#define FP16_AVX_ITER(a, b, sum) do {                                      \
    __m128i raw_a_lo_ = _mm_loadu_si128((const __m128i*)(a));              \
    __m128i raw_a_hi_ = _mm_loadu_si128((const __m128i*)((a) + 8));        \
    __m256  a_lo_     = _mm256_cvtph_ps(raw_a_lo_);                        \
    __m256  a_hi_     = _mm256_cvtph_ps(raw_a_hi_);                        \
    __m128i raw_b_lo_ = _mm_loadu_si128((const __m128i*)(b));              \
    __m128i raw_b_hi_ = _mm_loadu_si128((const __m128i*)((b) + 8));        \
    __m256  b_lo_     = _mm256_cvtph_ps(raw_b_lo_);                        \
    __m256  b_hi_     = _mm256_cvtph_ps(raw_b_hi_);                        \
    (sum) = _mm256_fmadd_ps(a_lo_, b_lo_, (sum));                          \
    (sum) = _mm256_fmadd_ps(a_hi_, b_hi_, (sum));                          \
    (a) += 16; (b) += 16;                                                  \
} while(0)
#endif

#if defined(USE_AVX512)
#define FP16_AVX512_ITER(a, b, sum) do {                                    \
    __m256i raw_a_lo_ = _mm256_loadu_si256((const __m256i*)(a));            \
    __m256i raw_a_hi_ = _mm256_loadu_si256((const __m256i*)((a) + 16));     \
    __m512  a_lo_     = _mm512_cvtph_ps(raw_a_lo_);                         \
    __m512  a_hi_     = _mm512_cvtph_ps(raw_a_hi_);                         \
    __m256i raw_b_lo_ = _mm256_loadu_si256((const __m256i*)(b));            \
    __m256i raw_b_hi_ = _mm256_loadu_si256((const __m256i*)((b) + 16));     \
    __m512  b_lo_     = _mm512_cvtph_ps(raw_b_lo_);                         \
    __m512  b_hi_     = _mm512_cvtph_ps(raw_b_hi_);                         \
    (sum) = _mm512_fmadd_ps(a_lo_, b_lo_, (sum));                           \
    (sum) = _mm512_fmadd_ps(a_hi_, b_hi_, (sum));                           \
    (a) += 32; (b) += 32;                                                  \
} while(0)
#endif


// =============================================================================
// FP16InnerProduct<MIN_ALIGN> — cascading SIMD FP16 inner product
// =============================================================================

template <size_t MIN_ALIGN>
class FP16InnerProduct {
public:

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        return 1.f - dot(pVect1v, pVect2v, qty_ptr);
    }

    inline static float dot(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const uint16_t* b = (const uint16_t*)pVect2v;
        size_t size = *((size_t*)qty_ptr);
        const uint16_t* last = a + size;

#if defined(USE_AVX512)
        __m512 sum512 = _mm512_setzero_ps();
        while (a + 32 <= last) {
            FP16_AVX512_ITER(a, b, sum512);
        }
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));

        if constexpr (MIN_ALIGN < 32) {
            while (a + 16 <= last) {
                FP16_AVX_ITER(a, b, sum256);
            }
        }

        float result = fp16_hsum256(sum256);
        if constexpr (MIN_ALIGN < 16) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 8 <= last) {
                FP16_SSE_ITER(a, b, sum128);
            }
            result += fp16_hsum128(sum128);
        }
        if constexpr (MIN_ALIGN < 8) {
            while (a < last) {
                result += fp16_to_float(*a++) * fp16_to_float(*b++);
            }
        }
        return result;

#elif defined(USE_AVX)
        __m256 sum256 = _mm256_setzero_ps();

        while (a + 16 <= last) {
            FP16_AVX_ITER(a, b, sum256);
        }

        float result = fp16_hsum256(sum256);
        if constexpr (MIN_ALIGN < 16) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 8 <= last) {
                FP16_SSE_ITER(a, b, sum128);
            }
            result += fp16_hsum128(sum128);
        }
        if constexpr (MIN_ALIGN < 8) {
            while (a < last) {
                result += fp16_to_float(*a++) * fp16_to_float(*b++);
            }
        }
        return result;

#elif defined(USE_SSE)
        __m128 sum128 = _mm_setzero_ps();

        if constexpr (MIN_ALIGN >= 8) {
            while (a + 8 <= last) {
                FP16_SSE_ITER(a, b, sum128);
            }
        }

        float result = fp16_hsum128(sum128);
        if constexpr (MIN_ALIGN < 8) {
            while (a < last) {
                result += fp16_to_float(*a++) * fp16_to_float(*b++);
            }
        }
        return result;

#else
        float result = 0.f;
        for (size_t i = 0; i < size; ++i) {
            result += fp16_to_float(a[i]) * fp16_to_float(b[i]);
        }
        return result;
#endif
    }

private:
    inline static float dot_impl_tail(const uint16_t* a, const uint16_t* b, size_t tail_elems) {
        float result = 0.0f;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        if (tail_elems >= 8) {
            __m128 s = _mm_setzero_ps();
            FP16_SSE_ITER(a, b, s);
            result = fp16_hsum128(s);
            tail_elems -= 8;
        }
#endif
        for (size_t i = 0; i < tail_elems; ++i) {
            result += fp16_to_float(a[i]) * fp16_to_float(b[i]);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Batch helpers
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

            #define FP16_B8_AVX(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[c * 16]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[c * 16 + 8]); \
                reg = _mm256_fmadd_ps(q_lo, _mm256_cvtph_ps(r_lo_), reg); \
                reg = _mm256_fmadd_ps(q_hi, _mm256_cvtph_ps(r_hi_), reg); \
            } while(0)

            FP16_B8_AVX(0, s0); FP16_B8_AVX(1, s1);
            FP16_B8_AVX(2, s2); FP16_B8_AVX(3, s3);
            FP16_B8_AVX(4, s4); FP16_B8_AVX(5, s5);
            FP16_B8_AVX(6, s6); FP16_B8_AVX(7, s7);
            #undef FP16_B8_AVX
        }

        dists[0] = fp16_hsum256(s0); dists[1] = fp16_hsum256(s1);
        dists[2] = fp16_hsum256(s2); dists[3] = fp16_hsum256(s3);
        dists[4] = fp16_hsum256(s4); dists[5] = fp16_hsum256(s5);
        dists[6] = fp16_hsum256(s6); dists[7] = fp16_hsum256(s7);

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 8; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
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

            #define FP16_B4_AVX(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[c * 16]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[c * 16 + 8]); \
                reg = _mm256_fmadd_ps(q_lo, _mm256_cvtph_ps(r_lo_), reg); \
                reg = _mm256_fmadd_ps(q_hi, _mm256_cvtph_ps(r_hi_), reg); \
            } while(0)

            FP16_B4_AVX(0, s0); FP16_B4_AVX(1, s1);
            FP16_B4_AVX(2, s2); FP16_B4_AVX(3, s3);
            #undef FP16_B4_AVX
        }

        dists[0] = fp16_hsum256(s0); dists[1] = fp16_hsum256(s1);
        dists[2] = fp16_hsum256(s2); dists[3] = fp16_hsum256(s3);

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 4; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_AVX512)
    inline static void batch8_avx512(const uint16_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc32 = dim / 32;
        const size_t tail16 = nc32 * 32;
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

            #define FP16_B8_512(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m256i r_lo_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32]); \
                __m256i r_hi_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32 + 16]); \
                reg = _mm512_fmadd_ps(q_lo, _mm512_cvtph_ps(r_lo_), reg); \
                reg = _mm512_fmadd_ps(q_hi, _mm512_cvtph_ps(r_hi_), reg); \
            } while(0)

            FP16_B8_512(0, s0); FP16_B8_512(1, s1);
            FP16_B8_512(2, s2); FP16_B8_512(3, s3);
            FP16_B8_512(4, s4); FP16_B8_512(5, s5);
            FP16_B8_512(6, s6); FP16_B8_512(7, s7);
            #undef FP16_B8_512
        }

        dists[0] = fp16_hsum512(s0); dists[1] = fp16_hsum512(s1);
        dists[2] = fp16_hsum512(s2); dists[3] = fp16_hsum512(s3);
        dists[4] = fp16_hsum512(s4); dists[5] = fp16_hsum512(s5);
        dists[6] = fp16_hsum512(s6); dists[7] = fp16_hsum512(s7);

        // AVX-512 remainder: up to 1 AVX chunk (16 fp16)
        if (tail16 + 16 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();
            __m256 t4 = _mm256_setzero_ps(), t5 = _mm256_setzero_ps();
            __m256 t6 = _mm256_setzero_ps(), t7 = _mm256_setzero_ps();

            __m128i q16_lo = _mm_loadu_si128((const __m128i*)&query[tail16]);
            __m128i q16_hi = _mm_loadu_si128((const __m128i*)&query[tail16 + 8]);
            __m256 qf_lo = _mm256_cvtph_ps(q16_lo);
            __m256 qf_hi = _mm256_cvtph_ps(q16_hi);

            #define FP16_B8_AVX8(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[tail16]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[tail16 + 8]); \
                reg = _mm256_fmadd_ps(qf_lo, _mm256_cvtph_ps(r_lo_), reg); \
                reg = _mm256_fmadd_ps(qf_hi, _mm256_cvtph_ps(r_hi_), reg); \
            } while(0)

            FP16_B8_AVX8(0, t0); FP16_B8_AVX8(1, t1);
            FP16_B8_AVX8(2, t2); FP16_B8_AVX8(3, t3);
            FP16_B8_AVX8(4, t4); FP16_B8_AVX8(5, t5);
            FP16_B8_AVX8(6, t6); FP16_B8_AVX8(7, t7);
            #undef FP16_B8_AVX8

            dists[0] += fp16_hsum256(t0); dists[1] += fp16_hsum256(t1);
            dists[2] += fp16_hsum256(t2); dists[3] += fp16_hsum256(t3);
            dists[4] += fp16_hsum256(t4); dists[5] += fp16_hsum256(t5);
            dists[6] += fp16_hsum256(t6); dists[7] += fp16_hsum256(t7);
        }

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 8; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    inline static void batch4_avx512(const uint16_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc32 = dim / 32;
        const size_t tail16 = nc32 * 32;
        const size_t tail_start = (dim / 16) * 16;
        const size_t tail_dim = dim - tail_start;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();

        for (size_t c = 0; c < nc32; ++c) {
            __m256i q_raw_lo = _mm256_loadu_si256((const __m256i*)&query[c * 32]);
            __m256i q_raw_hi = _mm256_loadu_si256((const __m256i*)&query[c * 32 + 16]);
            __m512 q_lo = _mm512_cvtph_ps(q_raw_lo);
            __m512 q_hi = _mm512_cvtph_ps(q_raw_hi);

            #define FP16_B4_512(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m256i r_lo_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32]); \
                __m256i r_hi_ = _mm256_loadu_si256((const __m256i*)&db_[c * 32 + 16]); \
                reg = _mm512_fmadd_ps(q_lo, _mm512_cvtph_ps(r_lo_), reg); \
                reg = _mm512_fmadd_ps(q_hi, _mm512_cvtph_ps(r_hi_), reg); \
            } while(0)

            FP16_B4_512(0, s0); FP16_B4_512(1, s1);
            FP16_B4_512(2, s2); FP16_B4_512(3, s3);
            #undef FP16_B4_512
        }

        dists[0] = fp16_hsum512(s0); dists[1] = fp16_hsum512(s1);
        dists[2] = fp16_hsum512(s2); dists[3] = fp16_hsum512(s3);

        if (tail16 + 16 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();

            __m128i q16_lo = _mm_loadu_si128((const __m128i*)&query[tail16]);
            __m128i q16_hi = _mm_loadu_si128((const __m128i*)&query[tail16 + 8]);
            __m256 qf_lo = _mm256_cvtph_ps(q16_lo);
            __m256 qf_hi = _mm256_cvtph_ps(q16_hi);

            #define FP16_B4_AVX8(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_lo_ = _mm_loadu_si128((const __m128i*)&db_[tail16]); \
                __m128i r_hi_ = _mm_loadu_si128((const __m128i*)&db_[tail16 + 8]); \
                reg = _mm256_fmadd_ps(qf_lo, _mm256_cvtph_ps(r_lo_), reg); \
                reg = _mm256_fmadd_ps(qf_hi, _mm256_cvtph_ps(r_hi_), reg); \
            } while(0)

            FP16_B4_AVX8(0, t0); FP16_B4_AVX8(1, t1);
            FP16_B4_AVX8(2, t2); FP16_B4_AVX8(3, t3);
            #undef FP16_B4_AVX8

            dists[0] += fp16_hsum256(t0); dists[1] += fp16_hsum256(t1);
            dists[2] += fp16_hsum256(t2); dists[3] += fp16_hsum256(t3);
        }

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 4; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
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

            #define FP16_B4_SSE(j, reg) do { \
                const uint16_t* db_ = (const uint16_t*)db_arr[j]; \
                __m128i r_raw_ = _mm_loadu_si128((const __m128i*)&db_[c * 8]); \
                reg = _mm_fmadd_ps(q_lo, _mm_cvtph_ps(r_raw_), reg); \
                reg = _mm_fmadd_ps(q_hi, _mm_cvtph_ps(_mm_srli_si128(r_raw_, 8)), reg); \
            } while(0)

            FP16_B4_SSE(0, s0); FP16_B4_SSE(1, s1);
            FP16_B4_SSE(2, s2); FP16_B4_SSE(3, s3);
            #undef FP16_B4_SSE
        }

        dists[0] = fp16_hsum128(s0); dists[1] = fp16_hsum128(s1);
        dists[2] = fp16_hsum128(s2); dists[3] = fp16_hsum128(s3);

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 4; ++j) {
                const uint16_t* db = (const uint16_t*)db_arr[j];
                float tail = 0.0f;
                for (size_t k = 0; k < tail_dim; ++k)
                    tail += fp16_to_float(query[tail_start + k]) * fp16_to_float(db[tail_start + k]);
                dists[j] += tail;
            }
        }
    }
#endif

public:
    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
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
            dists[i] = dot(query, db_arr[i], qty_ptr);

        for (size_t j = 0; j < count; ++j)
            dists[j] = 1.0f - dists[j];
    }
};


// =============================================================================
// Backward-compatible aliases
// =============================================================================

using FP16InnerProductExt32          = FP16InnerProduct<32>;
using FP16InnerProductExt16          = FP16InnerProduct<16>;
using FP16InnerProductExt8           = FP16InnerProduct<8>;
using FP16InnerProductDefault        = FP16InnerProduct<1>;

} // namespace distances
} // namespace deglib

// Clean up internal macros
#ifdef FP16_SSE_ITER
#undef FP16_SSE_ITER
#endif
#ifdef FP16_AVX_ITER
#undef FP16_AVX_ITER
#endif
#ifdef FP16_AVX512_ITER
#undef FP16_AVX512_ITER
#endif
