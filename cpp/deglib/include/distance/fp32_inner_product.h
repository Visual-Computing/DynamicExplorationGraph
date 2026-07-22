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
// Inner Product (dot product) distance for float32 vectors.
//
// Single cascading SIMD template: FP32InnerProduct<MIN_ALIGN>
//   Always cascades from widest available SIMD down to scalar:
//     AVX-512 (16 floats) -> AVX (8 floats) -> SSE (4 floats) -> scalar
//
// MIN_ALIGN specifies the minimum guaranteed alignment of the dimension:
//   16 -> dim%16==0, 8 -> dim%8==0, 4 -> dim%4==0, 1 -> no guarantee
//
// compare() returns 1.f - dot(a, b)  (distance metric: 1 - similarity)
// dot()     returns the raw dot product (for benchmarking / internal use)
//
// if constexpr eliminates dead tail code at compile time.
//
// Backward-compatible aliases:
//   InnerProductFloat16Ext, InnerProductFloat8Ext, InnerProductFloat4Ext,
//   InnerProductFloat, InnerProductFloat16ExtResiduals, InnerProductFloat4ExtResiduals
//
// Performance guarantee: All abstractions resolved at compile time.
// No vtables, no function pointers, no dynamic dispatch.
// =============================================================================


// -----------------------------------------------------------------------------
// SIMD horizontal-sum helpers
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static inline float ip_hsum128(__m128 v) {
    alignas(16) float f[4];
    _mm_store_ps(f, v);
    return f[0] + f[1] + f[2] + f[3];
}
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
static inline float ip_hsum256(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return ip_hsum128(_mm_add_ps(lo, hi));
}
#endif

#if defined(USE_AVX512)
static inline float ip_hsum512(__m512 v) {
    __m256 lo = _mm512_extractf32x8_ps(v, 0);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return ip_hsum256(_mm256_add_ps(lo, hi));
}
#endif


// -----------------------------------------------------------------------------
// SIMD iteration macros for inner product (dot) accumulation
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#define IP_SSE_ITER(a, b, sum) do {                                       \
    (sum) = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), (sum));       \
    (a) += 4; (b) += 4;                                                   \
} while(0)
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
#define IP_AVX_ITER(a, b, sum) do {                                       \
    (sum) = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), (sum)); \
    (a) += 8; (b) += 8;                                                   \
} while(0)
#endif

#if defined(USE_AVX512)
#define IP_AVX512_ITER(a, b, sum) do {                                     \
    (sum) = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), (sum)); \
    (a) += 16; (b) += 16;                                                 \
} while(0)
#endif


// =============================================================================
// FP32InnerProduct<MIN_ALIGN> — cascading SIMD dot product
// =============================================================================

template <size_t MIN_ALIGN>
class FP32InnerProduct {
public:
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        return 1.f - dot(pVect1v, pVect2v, qty_ptr);
    }

    inline static float dot(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);
        const float* last = a + size;

#if defined(USE_AVX512)
        __m512 sum512 = _mm512_setzero_ps();
        while (a + 16 <= last) {
            IP_AVX512_ITER(a, b, sum512);
        }
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));

        if constexpr (MIN_ALIGN < 16) {
            while (a + 8 <= last) {
                IP_AVX_ITER(a, b, sum256);
            }
        }

        float result = ip_hsum256(sum256);
        if constexpr (MIN_ALIGN < 8) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 4 <= last) {
                IP_SSE_ITER(a, b, sum128);
            }
            result += ip_hsum128(sum128);
        }

        if constexpr (MIN_ALIGN < 4) {
            while (a < last) {
                result += *a++ * *b++;
            }
        }
        return result;

#elif defined(USE_AVX)
        __m256 sum256 = _mm256_setzero_ps();

        if constexpr (MIN_ALIGN >= 16) {
            while (a + 16 <= last) {
                IP_AVX_ITER(a, b, sum256);
                IP_AVX_ITER(a, b, sum256);
            }
        } else {
            while (a + 8 <= last) {
                IP_AVX_ITER(a, b, sum256);
            }
        }

        float result = ip_hsum256(sum256);
        if constexpr (MIN_ALIGN < 8) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 4 <= last) {
                IP_SSE_ITER(a, b, sum128);
            }
            result += ip_hsum128(sum128);
        }

        if constexpr (MIN_ALIGN < 4) {
            while (a < last) {
                result += *a++ * *b++;
            }
        }
        return result;

#elif defined(USE_SSE)
        __m128 sum128 = _mm_setzero_ps();

        if constexpr (MIN_ALIGN >= 16) {
            while (a + 16 <= last) {
                IP_SSE_ITER(a, b, sum128);
                IP_SSE_ITER(a, b, sum128);
                IP_SSE_ITER(a, b, sum128);
                IP_SSE_ITER(a, b, sum128);
            }
        } else if constexpr (MIN_ALIGN >= 8) {
            while (a + 8 <= last) {
                IP_SSE_ITER(a, b, sum128);
                IP_SSE_ITER(a, b, sum128);
            }
        } else if constexpr (MIN_ALIGN >= 4) {
            while (a + 4 <= last) {
                IP_SSE_ITER(a, b, sum128);
            }
        }

        float result = ip_hsum128(sum128);
        if constexpr (MIN_ALIGN < 4) {
            while (a < last) {
                result += *a++ * *b++;
            }
        }
        return result;

#else
        float dot0, dot1, dot2, dot3;
        const float* unroll_group = last - 3;
        float result = 0;
        while (a < unroll_group) {
            dot0 = a[0] * b[0];
            dot1 = a[1] * b[1];
            dot2 = a[2] * b[2];
            dot3 = a[3] * b[3];
            result += dot0 + dot1 + dot2 + dot3;
            a += 4; b += 4;
        }
        while (a < last) {
            result += *a++ * *b++;
        }
        return result;
#endif
    }

private:
    inline static float dot_impl_tail(const float* a, const float* b, size_t tail_dim) {
        float result = 0.0f;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        if (tail_dim >= 4) {
            __m128 s = _mm_setzero_ps();
            IP_SSE_ITER(a, b, s);
            result = ip_hsum128(s);
            tail_dim -= 4;
        }
#endif
        for (size_t i = 0; i < tail_dim; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

#if defined(USE_AVX) || defined(USE_AVX512)
    inline static void batch8_avx(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 8;
        const size_t tail_start = nc * 8;
        const size_t tail_dim = dim - tail_start;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps(), s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps(), s7 = _mm256_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

            #define IP_BATCH_AVX(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                reg = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db[c * 8]), reg); \
            } while(0)

            IP_BATCH_AVX(0, s0); IP_BATCH_AVX(1, s1); IP_BATCH_AVX(2, s2); IP_BATCH_AVX(3, s3);
            IP_BATCH_AVX(4, s4); IP_BATCH_AVX(5, s5); IP_BATCH_AVX(6, s6); IP_BATCH_AVX(7, s7);
            #undef IP_BATCH_AVX
        }

        dists[0] = ip_hsum256(s0); dists[1] = ip_hsum256(s1);
        dists[2] = ip_hsum256(s2); dists[3] = ip_hsum256(s3);
        dists[4] = ip_hsum256(s4); dists[5] = ip_hsum256(s5);
        dists[6] = ip_hsum256(s6); dists[7] = ip_hsum256(s7);

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 8; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    inline static void batch4_avx(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 8;
        const size_t tail_start = nc * 8;
        const size_t tail_dim = dim - tail_start;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

            #define IP_BATCH_AVX4(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                reg = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db[c * 8]), reg); \
            } while(0)

            IP_BATCH_AVX4(0, s0); IP_BATCH_AVX4(1, s1);
            IP_BATCH_AVX4(2, s2); IP_BATCH_AVX4(3, s3);
            #undef IP_BATCH_AVX4
        }

        dists[0] = ip_hsum256(s0); dists[1] = ip_hsum256(s1);
        dists[2] = ip_hsum256(s2); dists[3] = ip_hsum256(s3);

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 4; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_AVX512)
    inline static void batch8_avx512(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc16 = dim / 16;
        const size_t tail8 = nc16 * 16;
        const size_t tail_start = (dim / 8) * 8;
        const size_t tail_dim = dim - tail_start;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        __m512 s4 = _mm512_setzero_ps(), s5 = _mm512_setzero_ps();
        __m512 s6 = _mm512_setzero_ps(), s7 = _mm512_setzero_ps();

        for (size_t c = 0; c < nc16; ++c) {
            const __m512 qf = _mm512_loadu_ps(&query[c * 16]);

            #define IP_BATCH_AVX512(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                reg = _mm512_fmadd_ps(qf, _mm512_loadu_ps(&db[c * 16]), reg); \
            } while(0)

            IP_BATCH_AVX512(0, s0); IP_BATCH_AVX512(1, s1); IP_BATCH_AVX512(2, s2); IP_BATCH_AVX512(3, s3);
            IP_BATCH_AVX512(4, s4); IP_BATCH_AVX512(5, s5); IP_BATCH_AVX512(6, s6); IP_BATCH_AVX512(7, s7);
            #undef IP_BATCH_AVX512
        }

        dists[0] = ip_hsum512(s0); dists[1] = ip_hsum512(s1);
        dists[2] = ip_hsum512(s2); dists[3] = ip_hsum512(s3);
        dists[4] = ip_hsum512(s4); dists[5] = ip_hsum512(s5);
        dists[6] = ip_hsum512(s6); dists[7] = ip_hsum512(s7);

        if (tail8 + 8 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();
            __m256 t4 = _mm256_setzero_ps(), t5 = _mm256_setzero_ps();
            __m256 t6 = _mm256_setzero_ps(), t7 = _mm256_setzero_ps();

            const __m256 qf8 = _mm256_loadu_ps(&query[tail8]);

            #define IP_BATCH_AVX8(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                reg = _mm256_fmadd_ps(qf8, _mm256_loadu_ps(&db[tail8]), reg); \
            } while(0)

            IP_BATCH_AVX8(0, t0); IP_BATCH_AVX8(1, t1); IP_BATCH_AVX8(2, t2); IP_BATCH_AVX8(3, t3);
            IP_BATCH_AVX8(4, t4); IP_BATCH_AVX8(5, t5); IP_BATCH_AVX8(6, t6); IP_BATCH_AVX8(7, t7);
            #undef IP_BATCH_AVX8

            dists[0] += ip_hsum256(t0); dists[1] += ip_hsum256(t1);
            dists[2] += ip_hsum256(t2); dists[3] += ip_hsum256(t3);
            dists[4] += ip_hsum256(t4); dists[5] += ip_hsum256(t5);
            dists[6] += ip_hsum256(t6); dists[7] += ip_hsum256(t7);
        }

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 8; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    inline static void batch4_avx512(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc16 = dim / 16;
        const size_t tail8 = nc16 * 16;
        const size_t tail_start = (dim / 8) * 8;
        const size_t tail_dim = dim - tail_start;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();

        for (size_t c = 0; c < nc16; ++c) {
            const __m512 qf = _mm512_loadu_ps(&query[c * 16]);

            #define IP_BATCH4_AVX512(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                reg = _mm512_fmadd_ps(qf, _mm512_loadu_ps(&db[c * 16]), reg); \
            } while(0)

            IP_BATCH4_AVX512(0, s0); IP_BATCH4_AVX512(1, s1);
            IP_BATCH4_AVX512(2, s2); IP_BATCH4_AVX512(3, s3);
            #undef IP_BATCH4_AVX512
        }

        dists[0] = ip_hsum512(s0); dists[1] = ip_hsum512(s1);
        dists[2] = ip_hsum512(s2); dists[3] = ip_hsum512(s3);

        if (tail8 + 8 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();

            const __m256 qf8 = _mm256_loadu_ps(&query[tail8]);

            #define IP_BATCH4_AVX8(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                reg = _mm256_fmadd_ps(qf8, _mm256_loadu_ps(&db[tail8]), reg); \
            } while(0)

            IP_BATCH4_AVX8(0, t0); IP_BATCH4_AVX8(1, t1);
            IP_BATCH4_AVX8(2, t2); IP_BATCH4_AVX8(3, t3);
            #undef IP_BATCH4_AVX8

            dists[0] += ip_hsum256(t0); dists[1] += ip_hsum256(t1);
            dists[2] += ip_hsum256(t2); dists[3] += ip_hsum256(t3);
        }

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 4; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    inline static void batch4_sse(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 4;
        const size_t tail_start = nc * 4;
        const size_t tail_dim = dim - tail_start;
        __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
        __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            const __m128 qf = _mm_loadu_ps(&query[c * 4]);

            #define IP_BATCH_SSE(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                reg = _mm_fmadd_ps(qf, _mm_loadu_ps(&db[c * 4]), reg); \
            } while(0)

            IP_BATCH_SSE(0, s0); IP_BATCH_SSE(1, s1);
            IP_BATCH_SSE(2, s2); IP_BATCH_SSE(3, s3);
            #undef IP_BATCH_SSE
        }

        dists[0] = ip_hsum128(s0); dists[1] = ip_hsum128(s1);
        dists[2] = ip_hsum128(s2); dists[3] = ip_hsum128(s3);

        if constexpr (MIN_ALIGN < 4) {
            for (int j = 0; j < 4; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += dot_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

public:
    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        const float* query = (const float*)query_ptr;
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

        // All batch helpers compute raw dot product; convert to distance metric
        for (size_t j = 0; j < count; ++j)
            dists[j] = 1.0f - dists[j];
    }
};


// =============================================================================
// Backward-compatible aliases
// =============================================================================

using InnerProductFloat16Ext = FP32InnerProduct<16>;
using InnerProductFloat8Ext  = FP32InnerProduct<8>;
using InnerProductFloat4Ext  = FP32InnerProduct<4>;
using InnerProductFloat      = FP32InnerProduct<1>;

} // namespace distances
} // namespace deglib

// Clean up internal macros to avoid polluting the global namespace
#ifdef IP_SSE_ITER
#undef IP_SSE_ITER
#endif
#ifdef IP_AVX_ITER
#undef IP_AVX_ITER
#endif
#ifdef IP_AVX512_ITER
#undef IP_AVX512_ITER
#endif
