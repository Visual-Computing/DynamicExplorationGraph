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
// L2 (Euclidean squared) distance for float32 vectors.
//
// Single cascading SIMD template: FP32L2<MIN_ALIGN>
//   Always cascades from widest available SIMD down to scalar:
//     AVX-512 (16 floats) -> AVX (8 floats) -> SSE (4 floats) -> scalar
//
// MIN_ALIGN specifies the minimum guaranteed alignment of the dimension:
//   16 -> dim%16==0, 8 -> dim%8==0, 4 -> dim%4==0, 1 -> no guarantee
//
// if constexpr eliminates dead tail code at compile time.
//
// Backward-compatible aliases:
//   L2Float16Ext, L2Float8Ext, L2Float4Ext, L2Float,
//   L2Float16ExtResiduals, L2Float4ExtResiduals
//
// Performance guarantee: All abstractions resolved at compile time.
// No vtables, no function pointers, no dynamic dispatch.
// =============================================================================


// -----------------------------------------------------------------------------
// SIMD horizontal-sum helpers
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
/**
 * Horizontal sum of a 128-bit SSE register (4 floats -> 1 scalar float).
 */
static inline float hsum128(__m128 v) {
    alignas(16) float f[4];
    _mm_store_ps(f, v);
    return f[0] + f[1] + f[2] + f[3];
}
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
/**
 * Horizontal sum of a 256-bit AVX register (8 floats -> 1 scalar float).
 * Splits into two 128-bit halves, adds them, then delegates to hsum128.
 */
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum128(_mm_add_ps(lo, hi));
}
#endif

#if defined(USE_AVX512)
/**
 * Horizontal sum of a 512-bit AVX-512 register (16 floats -> 1 scalar float).
 * Splits into two 256-bit halves, adds them, then delegates to hsum256.
 */
static inline float hsum512(__m512 v) {
    __m256 lo = _mm512_extractf32x8_ps(v, 0);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return hsum256(_mm256_add_ps(lo, hi));
}
#endif


// -----------------------------------------------------------------------------
// SIMD iteration macros for L2 distance accumulation
// -----------------------------------------------------------------------------

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
/** One SSE iteration: process 4 floats of L2 squared difference. */
#define L2_SSE_ITER(a, b, sum) do {                                     \
    __m128 v_ = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));          \
    (sum) = _mm_fmadd_ps(v_, v_, (sum));                                \
    (a) += 4; (b) += 4;                                                 \
} while(0)
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
/** One AVX iteration: process 8 floats of L2 squared difference. */
#define L2_AVX_ITER(a, b, sum) do {                                     \
    __m256 v_ = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));  \
    (sum) = _mm256_fmadd_ps(v_, v_, (sum));                             \
    (a) += 8; (b) += 8;                                                 \
} while(0)
#endif

#if defined(USE_AVX512)
/** One AVX-512 iteration: process 16 floats of L2 squared difference. */
#define L2_AVX512_ITER(a, b, sum) do {                                   \
    __m512 v_ = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));   \
    (sum) = _mm512_fmadd_ps(v_, v_, (sum));                               \
    (a) += 16; (b) += 16;                                                \
} while(0)
#endif


// =============================================================================
// FP32L2<MIN_ALIGN> — cascading SIMD distance
// =============================================================================

/**
 * Template parameter MIN_ALIGN:
 *   16 -> dim%16==0  (tightest loop, widest SIMD)
 *   8  -> dim%8==0
 *   4  -> dim%4==0
 *   1  -> arbitrary dimension
 *
 * The compare() cascades: widest available SIMD main loop -> narrower tail -> scalar tail.
 * if constexpr (MIN_ALIGN >= N) eliminates dead code.
 *
 * Batch helpers (batch8_*, batch4_*) are private static members.  Each one
 * processes all full SIMD chunks for its batch size and, if MIN_ALIGN < 8/4,
 * also applies compare_impl_tail for the residual floats at the end of each vector.
 */
template <size_t MIN_ALIGN>
class FP32L2 {
public:
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);
        const float* last = a + size;

#if defined(USE_AVX512)
        // ---- AVX-512 main loop (16 floats per iter) ----
        __m512 sum512 = _mm512_setzero_ps();
        while (a + 16 <= last) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
            sum512 = _mm512_fmadd_ps(v, v, sum512);
            a += 16; b += 16;
        }
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));

        // ---- AVX tail (if MIN_ALIGN < 16, up to 1 iter of 8 floats) ----
        if constexpr (MIN_ALIGN < 16) {
            while (a + 8 <= last) {
                L2_AVX_ITER(a, b, sum256);
            }
        }

        // ---- SSE tail (if MIN_ALIGN < 8, up to 1 iter of 4 floats) ----
        float result = hsum256(sum256);
        if constexpr (MIN_ALIGN < 8) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 4 <= last) {
                L2_SSE_ITER(a, b, sum128);
            }
            result += hsum128(sum128);
        }

        // ---- Scalar tail (if MIN_ALIGN < 4) ----
        if constexpr (MIN_ALIGN < 4) {
            float diff;
            while (a < last) {
                diff = *a++ - *b++;
                result += diff * diff;
            }
        }
        return result;

#elif defined(USE_AVX)
        // ---- AVX main loop ----
        // Unroll 2x when MIN_ALIGN >= 16 (processes 16 floats per iter = 1 cache line)
        __m256 sum256 = _mm256_setzero_ps();

        if constexpr (MIN_ALIGN >= 16) {
            while (a + 16 <= last) {
                L2_AVX_ITER(a, b, sum256);
                L2_AVX_ITER(a, b, sum256);
            }
        } else {
            while (a + 8 <= last) {
                L2_AVX_ITER(a, b, sum256);
            }
        }

        // ---- SSE tail (if MIN_ALIGN < 8, up to 1 iter of 4 floats) ----
        float result = hsum256(sum256);
        if constexpr (MIN_ALIGN < 8) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 4 <= last) {
                L2_SSE_ITER(a, b, sum128);
            }
            result += hsum128(sum128);
        }

        // ---- Scalar tail (if MIN_ALIGN < 4) ----
        if constexpr (MIN_ALIGN < 4) {
            float diff;
            while (a < last) {
                diff = *a++ - *b++;
                result += diff * diff;
            }
        }
        return result;

#elif defined(USE_SSE)
        // ---- SSE main loop ----
        // Unroll based on MIN_ALIGN: 4x for >=16, 2x for >=8, 1x for >=4
        __m128 sum128 = _mm_setzero_ps();

        if constexpr (MIN_ALIGN >= 16) {
            while (a + 16 <= last) {
                L2_SSE_ITER(a, b, sum128);
                L2_SSE_ITER(a, b, sum128);
                L2_SSE_ITER(a, b, sum128);
                L2_SSE_ITER(a, b, sum128);
            }
        } else if constexpr (MIN_ALIGN >= 8) {
            while (a + 8 <= last) {
                L2_SSE_ITER(a, b, sum128);
                L2_SSE_ITER(a, b, sum128);
            }
        } else if constexpr (MIN_ALIGN >= 4) {
            while (a + 4 <= last) {
                L2_SSE_ITER(a, b, sum128);
            }
        }

        // ---- Scalar tail (if MIN_ALIGN < 4) ----
        float result = hsum128(sum128);
        if constexpr (MIN_ALIGN < 4) {
            float diff;
            while (a < last) {
                diff = *a++ - *b++;
                result += diff * diff;
            }
        }
        return result;

#else
        // ---- Pure scalar fallback with 4x unrolling ----
        float diff0, diff1, diff2, diff3;
        const float* unroll_group = last - 3;
        float result = 0;
        while (a < unroll_group) {
            diff0 = a[0] - b[0];
            diff1 = a[1] - b[1];
            diff2 = a[2] - b[2];
            diff3 = a[3] - b[3];
            result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
            a += 4; b += 4;
        }
        while (a < last) {
            diff0 = *a++ - *b++;
            result += diff0 * diff0;
        }
        return result;
#endif
    }

private:
    // -------------------------------------------------------------------------
    // Tail helper: accumulates residual floats that do not fill a full SIMD chunk
    // -------------------------------------------------------------------------

    /**
     * Computes the L2 residual for the tail end of a vector pair.
     *
     * Batch functions operate on full SIMD chunks and leave any remainder
     * (dim % 8 or dim % 4) to this helper:
     *   - tail_dim >= 4: one SSE iteration (4 floats via L2_SSE_ITER)
     *   - tail_dim 0..3: plain scalar loop
     */
    inline static float compare_impl_tail(const float* a, const float* b, size_t tail_dim) {
        float result = 0.0f;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        if (tail_dim >= 4) {
            __m128 s = _mm_setzero_ps();
            L2_SSE_ITER(a, b, s);
            result = hsum128(s);
            tail_dim -= 4;
        }
#endif
        for (size_t i = 0; i < tail_dim; ++i) {
            float d = a[i] - b[i];
            result += d * d;
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // SIMD batch helpers (private)
    //
    // Each function processes all full SIMD chunks for its batch size and,
    // when the MIN_ALIGN guarantee does not cover the chunk size, also
    // applies compare_impl_tail for the residual floats.  This keeps the
    // tail logic local to each batch function and eliminates the need for
    // a separate post-pass in compare_batch.
    // -------------------------------------------------------------------------

#if defined(USE_AVX) || defined(USE_AVX512)
    /** Batch-8 AVX: L2 from query to 8 db vectors, dim/8*8 floats + tail. */
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

            #define L2_BATCH_AVX(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                __m256 v = _mm256_sub_ps(qf, _mm256_loadu_ps(&db[c * 8])); \
                reg = _mm256_fmadd_ps(v, v, reg); \
            } while(0)

            L2_BATCH_AVX(0, s0); L2_BATCH_AVX(1, s1); L2_BATCH_AVX(2, s2); L2_BATCH_AVX(3, s3);
            L2_BATCH_AVX(4, s4); L2_BATCH_AVX(5, s5); L2_BATCH_AVX(6, s6); L2_BATCH_AVX(7, s7);
            #undef L2_BATCH_AVX
        }

        dists[0] = hsum256(s0); dists[1] = hsum256(s1);
        dists[2] = hsum256(s2); dists[3] = hsum256(s3);
        dists[4] = hsum256(s4); dists[5] = hsum256(s5);
        dists[6] = hsum256(s6); dists[7] = hsum256(s7);

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 8; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    /** Batch-4 AVX: L2 from query to 4 db vectors, dim/8*8 floats + tail. */
    inline static void batch4_avx(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 8;
        const size_t tail_start = nc * 8;
        const size_t tail_dim = dim - tail_start;
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

            #define L2_BATCH_AVX4(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                __m256 v = _mm256_sub_ps(qf, _mm256_loadu_ps(&db[c * 8])); \
                reg = _mm256_fmadd_ps(v, v, reg); \
            } while(0)

            L2_BATCH_AVX4(0, s0); L2_BATCH_AVX4(1, s1);
            L2_BATCH_AVX4(2, s2); L2_BATCH_AVX4(3, s3);
            #undef L2_BATCH_AVX4
        }

        dists[0] = hsum256(s0); dists[1] = hsum256(s1);
        dists[2] = hsum256(s2); dists[3] = hsum256(s3);

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 4; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_AVX512)
    /** Batch-8 AVX-512: L2 from query to 8 db vectors, 16+8 floats + tail. */
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

            #define L2_BATCH_AVX512(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                __m512 v = _mm512_sub_ps(qf, _mm512_loadu_ps(&db[c * 16])); \
                reg = _mm512_fmadd_ps(v, v, reg); \
            } while(0)

            L2_BATCH_AVX512(0, s0); L2_BATCH_AVX512(1, s1); L2_BATCH_AVX512(2, s2); L2_BATCH_AVX512(3, s3);
            L2_BATCH_AVX512(4, s4); L2_BATCH_AVX512(5, s5); L2_BATCH_AVX512(6, s6); L2_BATCH_AVX512(7, s7);
            #undef L2_BATCH_AVX512
        }

        dists[0] = hsum512(s0); dists[1] = hsum512(s1);
        dists[2] = hsum512(s2); dists[3] = hsum512(s3);
        dists[4] = hsum512(s4); dists[5] = hsum512(s5);
        dists[6] = hsum512(s6); dists[7] = hsum512(s7);

        // AVX-512 remainder: one extra 8-float chunk if dim straddles a 16-float boundary
        if (tail8 + 8 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();
            __m256 t4 = _mm256_setzero_ps(), t5 = _mm256_setzero_ps();
            __m256 t6 = _mm256_setzero_ps(), t7 = _mm256_setzero_ps();

            const __m256 qf8 = _mm256_loadu_ps(&query[tail8]);

            #define L2_BATCH_AVX8(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                __m256 v = _mm256_sub_ps(qf8, _mm256_loadu_ps(&db[tail8])); \
                reg = _mm256_fmadd_ps(v, v, reg); \
            } while(0)

            L2_BATCH_AVX8(0, t0); L2_BATCH_AVX8(1, t1); L2_BATCH_AVX8(2, t2); L2_BATCH_AVX8(3, t3);
            L2_BATCH_AVX8(4, t4); L2_BATCH_AVX8(5, t5); L2_BATCH_AVX8(6, t6); L2_BATCH_AVX8(7, t7);
            #undef L2_BATCH_AVX8

            dists[0] += hsum256(t0); dists[1] += hsum256(t1);
            dists[2] += hsum256(t2); dists[3] += hsum256(t3);
            dists[4] += hsum256(t4); dists[5] += hsum256(t5);
            dists[6] += hsum256(t6); dists[7] += hsum256(t7);
        }

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 8; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }

    /** Batch-4 AVX-512: L2 from query to 4 db vectors, 16+8 floats + tail. */
    inline static void batch4_avx512(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc16 = dim / 16;
        const size_t tail8 = nc16 * 16;
        const size_t tail_start = (dim / 8) * 8;
        const size_t tail_dim = dim - tail_start;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();

        for (size_t c = 0; c < nc16; ++c) {
            const __m512 qf = _mm512_loadu_ps(&query[c * 16]);

            #define L2_BATCH4_AVX512(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                __m512 v = _mm512_sub_ps(qf, _mm512_loadu_ps(&db[c * 16])); \
                reg = _mm512_fmadd_ps(v, v, reg); \
            } while(0)

            L2_BATCH4_AVX512(0, s0); L2_BATCH4_AVX512(1, s1);
            L2_BATCH4_AVX512(2, s2); L2_BATCH4_AVX512(3, s3);
            #undef L2_BATCH4_AVX512
        }

        dists[0] = hsum512(s0); dists[1] = hsum512(s1);
        dists[2] = hsum512(s2); dists[3] = hsum512(s3);

        // AVX-512 remainder: one extra 8-float chunk if dim straddles a 16-float boundary
        if (tail8 + 8 <= dim) {
            __m256 t0 = _mm256_setzero_ps(), t1 = _mm256_setzero_ps();
            __m256 t2 = _mm256_setzero_ps(), t3 = _mm256_setzero_ps();

            const __m256 qf8 = _mm256_loadu_ps(&query[tail8]);

            #define L2_BATCH4_AVX8(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                __m256 v = _mm256_sub_ps(qf8, _mm256_loadu_ps(&db[tail8])); \
                reg = _mm256_fmadd_ps(v, v, reg); \
            } while(0)

            L2_BATCH4_AVX8(0, t0); L2_BATCH4_AVX8(1, t1);
            L2_BATCH4_AVX8(2, t2); L2_BATCH4_AVX8(3, t3);
            #undef L2_BATCH4_AVX8

            dists[0] += hsum256(t0); dists[1] += hsum256(t1);
            dists[2] += hsum256(t2); dists[3] += hsum256(t3);
        }

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 4; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    /** Batch-4 SSE: L2 from query to 4 db vectors, dim/4*4 floats + tail. */
    inline static void batch4_sse(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 4;
        const size_t tail_start = nc * 4;
        const size_t tail_dim = dim - tail_start;
        __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
        __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            const __m128 qf = _mm_loadu_ps(&query[c * 4]);

            #define L2_BATCH_SSE(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                __m128 v = _mm_sub_ps(qf, _mm_loadu_ps(&db[c * 4])); \
                reg = _mm_fmadd_ps(v, v, reg); \
            } while(0)

            L2_BATCH_SSE(0, s0); L2_BATCH_SSE(1, s1);
            L2_BATCH_SSE(2, s2); L2_BATCH_SSE(3, s3);
            #undef L2_BATCH_SSE
        }

        dists[0] = hsum128(s0); dists[1] = hsum128(s1);
        dists[2] = hsum128(s2); dists[3] = hsum128(s3);

        if constexpr (MIN_ALIGN < 4) {
            for (int j = 0; j < 4; ++j) {
                const float* db = (const float*)db_arr[j];
                dists[j] += compare_impl_tail(query + tail_start, db + tail_start, tail_dim);
            }
        }
    }
#endif

public:
    /**
     * Batch L2 distance: computes squared L2 from one query to many database vectors.
     *
     * Strategy
     * --------
     * 1. Process as many vectors as possible in SIMD batches of 8 or 4 using the
     *    widest available ISA (AVX-512 -> AVX -> SSE).  Each batch function
     *    internally handles its own residual tail (dim % chunk_size) via
     *    compare_impl_tail, gated by `if constexpr` so it is compile-time
     *    eliminated when MIN_ALIGN guarantees full chunks.
     * 2. Any remaining vectors that did not fit into a full batch are processed
     *    individually via compare(), which has its own cascading SIMD fallback.
     *
     * @param query_ptr  Pointer to the query vector (float32).
     * @param db_arr     Array of `count` pointers to database vectors.
     * @param count      Number of database vectors.
     * @param qty_ptr    Pointer to size_t storing the vector dimensionality.
     * @param dists      Output array of `count` squared L2 distances.
     */
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
        // Remaining vectors that did not fill a complete batch slot
        for (; i < count; ++i)
            dists[i] = compare(query, db_arr[i], qty_ptr);
    }
};


// =============================================================================
// Backward-compatible aliases
// =============================================================================

using L2Float16Ext            = FP32L2<16>;
using L2Float8Ext             = FP32L2<8>;
using L2Float4Ext             = FP32L2<4>;
using L2Float                 = FP32L2<1>;

} // namespace distances
} // namespace deglib

// Clean up internal macros to avoid polluting the global namespace
#ifdef L2_SSE_ITER
#undef L2_SSE_ITER
#endif
#ifdef L2_AVX_ITER
#undef L2_AVX_ITER
#endif
#ifdef L2_AVX512_ITER
#undef L2_AVX512_ITER
#endif
