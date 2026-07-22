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

namespace deglib {
namespace distances {

// =============================================================================
// L2 (Euclidean squared) distance for uint8 vectors.
//
// Single cascading SIMD template: Uint8L2<MIN_ALIGN>
//   Always cascades from widest available SIMD down to scalar:
//     AVX (16 uint8) -> SSE (16 uint8) -> scalar
//
// MIN_ALIGN specifies the minimum guaranteed alignment of the dimension:
//   32 -> dim%32==0, 16 -> dim%16==0, 1 -> no guarantee
//
// if constexpr eliminates dead tail code at compile time.
//
// Backward-compatible aliases:
//   L2Uint8Ext32   = Uint8L2<32>
//   L2Uint8Ext16   = Uint8L2<16>
//   Uint8L2Default = Uint8L2<1>
//
// Performance guarantee: All abstractions resolved at compile time.
// No vtables, no function pointers, no dynamic dispatch.
// =============================================================================

// -----------------------------------------------------------------------------
// SIMD horizontal-sum helpers (int32 accumulators)
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static inline int hsum128_epi32(__m128i v) {
    alignas(16) int buf[4];
    _mm_store_si128((__m128i*)buf, v);
    return buf[0] + buf[1] + buf[2] + buf[3];
}
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
static inline int hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_extracti128_si256(v, 0);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    return hsum128_epi32(_mm_add_epi32(lo, hi));
}
#endif

// =============================================================================
// Uint8L2<MIN_ALIGN> — cascading SIMD distance
// =============================================================================

template <size_t MIN_ALIGN>
class Uint8L2 {
public:
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        return float(dot(pVect1v, pVect2v, qty_ptr));
    }

    inline static int64_t dot(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint8_t* a = (const uint8_t*)pVect1v;
        const uint8_t* b = (const uint8_t*)pVect2v;
        size_t size = *((const size_t*)qty_ptr);
        const uint8_t* last = a + size;

#if defined(USE_AVX512) || defined(USE_AVX)
        // ---- AVX main loop: 16 uint8 per iteration ----
        __m256i sum = _mm256_setzero_si256();
        while (a + 16 <= last) {
            __m128i va = _mm_loadu_si128((const __m128i*)(a));
            __m128i vb = _mm_loadu_si128((const __m128i*)(b));
            __m256i va16 = _mm256_cvtepu8_epi16(va);
            __m256i vb16 = _mm256_cvtepu8_epi16(vb);
            __m256i diff = _mm256_sub_epi16(va16, vb16);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(diff, diff));
            a += 16; b += 16;
        }
        int64_t result = (int64_t)hsum256_epi32(sum);

        // ---- Scalar tail (if MIN_ALIGN < 16) ----
        if constexpr (MIN_ALIGN < 16) {
            while (a < last) {
                int32_t d = (int32_t)(*a) - (int32_t)(*b);
                result += (int64_t)(d * d);
                ++a; ++b;
            }
        }
        return result;

#elif defined(USE_SSE)
        // ---- SSE main loop: 16 uint8 per iteration (lo+hi halves) ----
        // _mm_cvtepu8_epi16 only processes 8 bytes at a time, so we split
        __m128i sse_lo = _mm_setzero_si128();
        __m128i sse_hi = _mm_setzero_si128();
        while (a + 16 <= last) {
            __m128i va = _mm_loadu_si128((const __m128i*)(a));
            __m128i vb = _mm_loadu_si128((const __m128i*)(b));

            __m128i va_lo = _mm_cvtepu8_epi16(va);
            __m128i va_hi = _mm_cvtepu8_epi16(_mm_srli_si128(va, 8));
            __m128i vb_lo = _mm_cvtepu8_epi16(vb);
            __m128i vb_hi = _mm_cvtepu8_epi16(_mm_srli_si128(vb, 8));

            __m128i d_lo = _mm_sub_epi16(va_lo, vb_lo);
            __m128i d_hi = _mm_sub_epi16(va_hi, vb_hi);

            sse_lo = _mm_add_epi32(sse_lo, _mm_madd_epi16(d_lo, d_lo));
            sse_hi = _mm_add_epi32(sse_hi, _mm_madd_epi16(d_hi, d_hi));

            a += 16; b += 16;
        }

        __m128i sse_sum = _mm_add_epi32(sse_lo, sse_hi);
        int64_t result = (int64_t)hsum128_epi32(sse_sum);

        // ---- Scalar tail (if MIN_ALIGN < 16) ----
        if constexpr (MIN_ALIGN < 16) {
            while (a < last) {
                int32_t d = (int32_t)(*a) - (int32_t)(*b);
                result += (int64_t)(d * d);
                ++a; ++b;
            }
        }
        return result;

#else
        // ---- Pure scalar fallback ----
        int64_t result = 0;
        for (size_t i = 0; i < size; ++i) {
            int32_t d = (int32_t)a[i] - (int32_t)b[i];
            result += (int64_t)(d * d);
        }
        return result;
#endif
    }

private:
    // -------------------------------------------------------------------------
    // Tail helper: accumulates residual elements that do not fill a full
    // 16-byte SIMD chunk
    // -------------------------------------------------------------------------

    inline static int64_t compare_impl_tail(const uint8_t* a, const uint8_t* b, size_t tail_dim) {
        int64_t result = 0;
        for (size_t i = 0; i < tail_dim; ++i) {
            int32_t d = (int32_t)a[i] - (int32_t)b[i];
            result += (int64_t)(d * d);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // SIMD batch helpers (private)
    //
    // Each function processes all full 16-byte SIMD chunks and, when
    // MIN_ALIGN does not guarantee full chunks, applies compare_impl_tail
    // for the residual elements.
    // -------------------------------------------------------------------------

#if defined(USE_AVX) || defined(USE_AVX512)
    inline static void batch8_avx(const uint8_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 16;
        const size_t tail_start = nc * 16;
        const size_t tail_dim = dim - tail_start;
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        __m256i s2 = _mm256_setzero_si256(), s3 = _mm256_setzero_si256();
        __m256i s4 = _mm256_setzero_si256(), s5 = _mm256_setzero_si256();
        __m256i s6 = _mm256_setzero_si256(), s7 = _mm256_setzero_si256();

        for (size_t c = 0; c < nc; ++c) {
            __m128i qv = _mm_loadu_si128((const __m128i*)(query + c * 16));
            __m256i q16 = _mm256_cvtepu8_epi16(qv);

            #define U8_B8_AVX(j, reg) do { \
                const uint8_t* db_ = (const uint8_t*)db_arr[j]; \
                __m128i dv = _mm_loadu_si128((const __m128i*)(db_ + c * 16)); \
                __m256i d16 = _mm256_sub_epi16(q16, _mm256_cvtepu8_epi16(dv)); \
                reg = _mm256_add_epi32(reg, _mm256_madd_epi16(d16, d16)); \
            } while(0)

            U8_B8_AVX(0, s0); U8_B8_AVX(1, s1);
            U8_B8_AVX(2, s2); U8_B8_AVX(3, s3);
            U8_B8_AVX(4, s4); U8_B8_AVX(5, s5);
            U8_B8_AVX(6, s6); U8_B8_AVX(7, s7);
            #undef U8_B8_AVX
        }

        dists[0] = (float)hsum256_epi32(s0); dists[1] = (float)hsum256_epi32(s1);
        dists[2] = (float)hsum256_epi32(s2); dists[3] = (float)hsum256_epi32(s3);
        dists[4] = (float)hsum256_epi32(s4); dists[5] = (float)hsum256_epi32(s5);
        dists[6] = (float)hsum256_epi32(s6); dists[7] = (float)hsum256_epi32(s7);

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 8; ++j) {
                const uint8_t* db = (const uint8_t*)db_arr[j];
                dists[j] += (float)compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    inline static void batch4_avx(const uint8_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 16;
        const size_t tail_start = nc * 16;
        const size_t tail_dim = dim - tail_start;
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        __m256i s2 = _mm256_setzero_si256(), s3 = _mm256_setzero_si256();

        for (size_t c = 0; c < nc; ++c) {
            __m128i qv = _mm_loadu_si128((const __m128i*)(query + c * 16));
            __m256i q16 = _mm256_cvtepu8_epi16(qv);

            #define U8_B4_AVX(j, reg) do { \
                const uint8_t* db_ = (const uint8_t*)db_arr[j]; \
                __m128i dv = _mm_loadu_si128((const __m128i*)(db_ + c * 16)); \
                __m256i d16 = _mm256_sub_epi16(q16, _mm256_cvtepu8_epi16(dv)); \
                reg = _mm256_add_epi32(reg, _mm256_madd_epi16(d16, d16)); \
            } while(0)

            U8_B4_AVX(0, s0); U8_B4_AVX(1, s1);
            U8_B4_AVX(2, s2); U8_B4_AVX(3, s3);
            #undef U8_B4_AVX
        }

        dists[0] = (float)hsum256_epi32(s0); dists[1] = (float)hsum256_epi32(s1);
        dists[2] = (float)hsum256_epi32(s2); dists[3] = (float)hsum256_epi32(s3);

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 4; ++j) {
                const uint8_t* db = (const uint8_t*)db_arr[j];
                dists[j] += (float)compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    inline static void batch4_sse(const uint8_t* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 16;
        const size_t tail_start = nc * 16;
        const size_t tail_dim = dim - tail_start;
        __m128i s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
        __m128i s2 = _mm_setzero_si128(), s3 = _mm_setzero_si128();

        for (size_t c = 0; c < nc; ++c) {
            __m128i qv = _mm_loadu_si128((const __m128i*)(query + c * 16));
            __m128i q_lo = _mm_cvtepu8_epi16(qv);
            __m128i q_hi = _mm_cvtepu8_epi16(_mm_srli_si128(qv, 8));

            #define U8_B4_SSE(j, reg) do { \
                const uint8_t* db_ = (const uint8_t*)db_arr[j]; \
                __m128i dv = _mm_loadu_si128((const __m128i*)(db_ + c * 16)); \
                __m128i d_lo = _mm_sub_epi16(q_lo, _mm_cvtepu8_epi16(dv)); \
                __m128i d_hi = _mm_sub_epi16(q_hi, _mm_cvtepu8_epi16(_mm_srli_si128(dv, 8))); \
                reg = _mm_add_epi32(reg, _mm_madd_epi16(d_lo, d_lo)); \
                reg = _mm_add_epi32(reg, _mm_madd_epi16(d_hi, d_hi)); \
            } while(0)

            U8_B4_SSE(0, s0); U8_B4_SSE(1, s1);
            U8_B4_SSE(2, s2); U8_B4_SSE(3, s3);
            #undef U8_B4_SSE
        }

        dists[0] = (float)hsum128_epi32(s0); dists[1] = (float)hsum128_epi32(s1);
        dists[2] = (float)hsum128_epi32(s2); dists[3] = (float)hsum128_epi32(s3);

        if constexpr (MIN_ALIGN < 16) {
            for (int j = 0; j < 4; ++j) {
                const uint8_t* db = (const uint8_t*)db_arr[j];
                dists[j] += (float)compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

public:
    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        const uint8_t* query = (const uint8_t*)query_ptr;
        const size_t dim = *((const size_t*)qty_ptr);

        size_t i = 0;
#if defined(USE_AVX512) || defined(USE_AVX)
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

using L2Uint8Ext32   = Uint8L2<32>;
using L2Uint8Ext16   = Uint8L2<16>;
using Uint8L2Default = Uint8L2<1>;

} // namespace distances
} // namespace deglib
