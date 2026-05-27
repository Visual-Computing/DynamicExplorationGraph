#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <algorithm>
#include <thread>
#include <vector>
#include <numeric>
#include <filesystem>

#include <fmt/core.h>

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#include <immintrin.h>
#endif

#include "deglib/include/builder.h"
#include "deglib/include/concurrent.h"
#include "deglib/include/distances.h"
#include "deglib/include/graph/sizebounded_graph.h"
#include "deglib/include/quantization/evp_quantize.h"
#include "deglib/include/repository.h"

#include "evp_common.h"

// ============================================================================
// Microbench: FP16 Inner Product — deglib vs custom unrolled/batched variants
// ============================================================================
//
// Benchmarks FP16 inner product implementations:
//   1. Deglib built-in distance functions (ip_naive, ip_8ext, ip_16ext, ip_32ext, compare)
//   2. Custom unrolled single-query variants (avx2 unroll by 2 and 4)
//   3. Custom batch variants (batch=4, batch=8, with and without unroll)
//
// Batch variants amortize the FP16→float conversion across multiple database
// vectors, yielding significant speedups for KNN-style workloads.
//
// Output format: "name    ms    ns/op  (best-of-5, sink=...)"
//
// Hardware: AMD 5600G with AVX2 support
// Dataset: 200000 vectors, dim=1024, FP16 (train.hvecs)
//
// Results (100K comparisons, best-of-5):
// ---------------------------------------------------------------
//   ip_naive (deglib)          1339 ns/op    1.0x baseline
//   ip_8ext  (deglib)           633 ns/op    2.1x
//   ip_16ext (deglib)           517 ns/op    2.6x
//   compare  (deglib)          1294 ns/op    1.0x
//   ip_custom_unroll2           468 ns/op    2.9x
//   ip_custom_unroll4           350 ns/op    3.8x
//   ip_custom_batch4            222 ns/op    6.0x   (batch=4)
//   ip_custom_batch8            175 ns/op    7.7x   (batch=8)
//   ip_custom_batch4_u2         232 ns/op    5.8x   (batch=4, unroll2)
//   ip_custom_batch8_u2         169 ns/op    7.9x   (batch=8, unroll2)  ← BEST
// ---------------------------------------------------------------
//
// Key takeaways:
//   * Batch variants amortize query FP16→float conversion across neighbors.
//   * Batch=8 beats batch=4: wider amortization.
//   * Unrolling provides modest gains for batch=8, negligible for batch=4.
//   * ip_custom_unroll4 (single-query, 350 ns/op) is already faster than
//     deglib's best ext variant (ip_16ext, 517 ns/op).
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: horizontal sum of a 256-bit register (4+4 → scalar)
// ---------------------------------------------------------------------------
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    alignas(16) float f[4];
    _mm_store_ps(f, sum);
    return f[0] + f[1] + f[2] + f[3];
}

// ---------------------------------------------------------------------------
// Custom single-query implementations
// ---------------------------------------------------------------------------

// ip_custom_unroll2: AVX2 unroll by 2 iterations (32 FP16 per outer iteration)
// Uses 4 accumulators for ILP. Tail falls back to ip_16ext.
static float ip_custom_unroll2(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const uint16_t* a = (const uint16_t*)pVect1v;
    const uint16_t* b = (const uint16_t*)pVect2v;
    size_t size = *((size_t*)qty_ptr);

    size_t num_chunks = size / 16;  // each chunk = 16 FP16 elements
    size_t full_chunks = num_chunks & ~3ULL;  // round down to multiple of 4

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (size_t c = 0; c < full_chunks; c += 4) {
        // Iteration 0: elements [c..c+15]
        {
            __m128i raw_a_lo = _mm_loadu_si128((const __m128i*)&a[c * 16]);
            __m128i raw_a_hi = _mm_loadu_si128((const __m128i*)&a[c * 16 + 8]);
            __m128i raw_b_lo = _mm_loadu_si128((const __m128i*)&b[c * 16]);
            __m128i raw_b_hi = _mm_loadu_si128((const __m128i*)&b[c * 16 + 8]);
            __m256 va = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_lo), _mm256_cvtph_ps(raw_b_lo), sum0);
            sum0 = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_hi), _mm256_cvtph_ps(raw_b_hi), va);
        }
        // Iteration 1: elements [c+16..c+31]
        {
            size_t c2 = c + 16;
            __m128i raw_a_lo = _mm_loadu_si128((const __m128i*)&a[c2]);
            __m128i raw_a_hi = _mm_loadu_si128((const __m128i*)&a[c2 + 8]);
            __m128i raw_b_lo = _mm_loadu_si128((const __m128i*)&b[c2]);
            __m128i raw_b_hi = _mm_loadu_si128((const __m128i*)&b[c2 + 8]);
            __m256 va = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_lo), _mm256_cvtph_ps(raw_b_lo), sum1);
            sum1 = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_hi), _mm256_cvtph_ps(raw_b_hi), va);
        }
        // Iteration 2: elements [c+32..c+47]
        {
            size_t c2 = c + 32;
            __m128i raw_a_lo = _mm_loadu_si128((const __m128i*)&a[c2]);
            __m128i raw_a_hi = _mm_loadu_si128((const __m128i*)&a[c2 + 8]);
            __m128i raw_b_lo = _mm_loadu_si128((const __m128i*)&b[c2]);
            __m128i raw_b_hi = _mm_loadu_si128((const __m128i*)&b[c2 + 8]);
            __m256 va = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_lo), _mm256_cvtph_ps(raw_b_lo), sum2);
            sum2 = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_hi), _mm256_cvtph_ps(raw_b_hi), va);
        }
        // Iteration 3: elements [c+48..c+63]
        {
            size_t c2 = c + 48;
            __m128i raw_a_lo = _mm_loadu_si128((const __m128i*)&a[c2]);
            __m128i raw_a_hi = _mm_loadu_si128((const __m128i*)&a[c2 + 8]);
            __m128i raw_b_lo = _mm_loadu_si128((const __m128i*)&b[c2]);
            __m128i raw_b_hi = _mm_loadu_si128((const __m128i*)&b[c2 + 8]);
            __m256 va = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_lo), _mm256_cvtph_ps(raw_b_lo), sum3);
            sum3 = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_hi), _mm256_cvtph_ps(raw_b_hi), va);
        }
    }

    // Tail: remaining chunks handled by ip_16ext
    float result = hsum256(sum0) + hsum256(sum1) + hsum256(sum2) + hsum256(sum3);
    if (full_chunks < num_chunks) {
        result += deglib::distances::FP16InnerProductExt16::dot(&a[full_chunks * 16], &b[full_chunks * 16], qty_ptr);
    }
    return result;
#else
    return deglib::distances::FP16InnerProductDefault::dot(pVect1v, pVect2v, qty_ptr);
#endif
}

// ip_custom_unroll4: AVX2 unroll by 4 iterations (64 FP16 per outer iteration)
// Uses 8 accumulators for higher ILP, more register pressure.
static float ip_custom_unroll4(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const uint16_t* a = (const uint16_t*)pVect1v;
    const uint16_t* b = (const uint16_t*)pVect2v;
    size_t size = *((size_t*)qty_ptr);

    size_t num_chunks = size / 16;
    size_t full_chunks = num_chunks & ~7ULL;  // round down to multiple of 8

    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    __m256 s4 = _mm256_setzero_ps();
    __m256 s5 = _mm256_setzero_ps();
    __m256 s6 = _mm256_setzero_ps();
    __m256 s7 = _mm256_setzero_ps();

    for (size_t c = 0; c < full_chunks; c += 8) {
        auto accumulate = [&](size_t offset, __m256& dst) {
            __m128i raw_a_lo = _mm_loadu_si128((const __m128i*)&a[c + offset]);
            __m128i raw_a_hi = _mm_loadu_si128((const __m128i*)&a[c + offset + 8]);
            __m128i raw_b_lo = _mm_loadu_si128((const __m128i*)&b[c + offset]);
            __m128i raw_b_hi = _mm_loadu_si128((const __m128i*)&b[c + offset + 8]);
            __m256 t = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_lo), _mm256_cvtph_ps(raw_b_lo), dst);
            dst = _mm256_fmadd_ps(_mm256_cvtph_ps(raw_a_hi), _mm256_cvtph_ps(raw_b_hi), t);
        };
        accumulate(0, s0);
        accumulate(16, s1);
        accumulate(32, s2);
        accumulate(48, s3);
        accumulate(64, s4);
        accumulate(80, s5);
        accumulate(96, s6);
        accumulate(112, s7);
    }

    float result = hsum256(s0) + hsum256(s1) + hsum256(s2) + hsum256(s3)
                 + hsum256(s4) + hsum256(s5) + hsum256(s6) + hsum256(s7);
    if (full_chunks < num_chunks) {
        result += deglib::distances::FP16InnerProductExt16::dot(&a[full_chunks * 16], &b[full_chunks * 16], qty_ptr);
    }
    return result;
#else
    return deglib::distances::FP16InnerProductDefault::dot(pVect1v, pVect2v, qty_ptr);
#endif
}

// ---------------------------------------------------------------------------
// Custom batch implementations
// Signature: void fn(const uint16_t* query, const uint16_t* const* db_arr, const size_t* dim, float* dists)
// ---------------------------------------------------------------------------

// ip_custom_batch4: Batch of 4, chunk-first pattern (convert query per chunk, accumulate all neighbors)
static void ip_custom_batch4(const uint16_t* query, const uint16_t* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t num_chunks = *dim / 16;

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (size_t c = 0; c < num_chunks; ++c) {
        const __m128i q_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
        const __m128i q_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
        const __m256 qf_lo = _mm256_cvtph_ps(q_lo);
        const __m256 qf_hi = _mm256_cvtph_ps(q_hi);

        #define ACCUM(j, sum_reg) do { \
            const __m128i b_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16]); \
            const __m128i b_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16 + 8]); \
            const __m256 bf_lo = _mm256_cvtph_ps(b_lo); \
            const __m256 bf_hi = _mm256_cvtph_ps(b_hi); \
            sum_reg = _mm256_fmadd_ps(qf_lo, bf_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf_hi, bf_hi, sum_reg); \
        } while(0)

        ACCUM(0, sum0);
        ACCUM(1, sum1);
        ACCUM(2, sum2);
        ACCUM(3, sum3);
        #undef ACCUM
    }

    dists[0] = 1.0f - hsum256(sum0);
    dists[1] = 1.0f - hsum256(sum1);
    dists[2] = 1.0f - hsum256(sum2);
    dists[3] = 1.0f - hsum256(sum3);
#else
    for (int j = 0; j < 4; ++j) {
        dists[j] = 1.0f - deglib::distances::FP16InnerProductDefault::dot(query, db_arr[j], dim);
    }
#endif
}

// ip_custom_batch8: Batch of 8, chunk-first pattern
static void ip_custom_batch8(const uint16_t* query, const uint16_t* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t num_chunks = *dim / 16;

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();

    for (size_t c = 0; c < num_chunks; ++c) {
        const __m128i q_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
        const __m128i q_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
        const __m256 qf_lo = _mm256_cvtph_ps(q_lo);
        const __m256 qf_hi = _mm256_cvtph_ps(q_hi);

        #define ACCUM(j, sum_reg) do { \
            const __m128i b_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16]); \
            const __m128i b_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16 + 8]); \
            const __m256 bf_lo = _mm256_cvtph_ps(b_lo); \
            const __m256 bf_hi = _mm256_cvtph_ps(b_hi); \
            sum_reg = _mm256_fmadd_ps(qf_lo, bf_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf_hi, bf_hi, sum_reg); \
        } while(0)

        ACCUM(0, sum0);
        ACCUM(1, sum1);
        ACCUM(2, sum2);
        ACCUM(3, sum3);
        ACCUM(4, sum4);
        ACCUM(5, sum5);
        ACCUM(6, sum6);
        ACCUM(7, sum7);
        #undef ACCUM
    }

    dists[0] = 1.0f - hsum256(sum0);
    dists[1] = 1.0f - hsum256(sum1);
    dists[2] = 1.0f - hsum256(sum2);
    dists[3] = 1.0f - hsum256(sum3);
    dists[4] = 1.0f - hsum256(sum4);
    dists[5] = 1.0f - hsum256(sum5);
    dists[6] = 1.0f - hsum256(sum6);
    dists[7] = 1.0f - hsum256(sum7);
#else
    for (int j = 0; j < 8; ++j) {
        dists[j] = 1.0f - deglib::distances::FP16InnerProductDefault::dot(query, db_arr[j], dim);
    }
#endif
}

// ip_custom_batch4_unroll2: Batch=4, process 2 query chunks per outer iteration
static void ip_custom_batch4_unroll2(const uint16_t* query, const uint16_t* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t num_chunks = *dim / 16;

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t c = 0;
    for (; c + 1 < num_chunks; c += 2) {
        const __m128i q0_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
        const __m128i q0_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
        const __m256 qf0_lo = _mm256_cvtph_ps(q0_lo);
        const __m256 qf0_hi = _mm256_cvtph_ps(q0_hi);

        const __m128i q1_lo = _mm_loadu_si128((const __m128i*)&query[(c + 1) * 16]);
        const __m128i q1_hi = _mm_loadu_si128((const __m128i*)&query[(c + 1) * 16 + 8]);
        const __m256 qf1_lo = _mm256_cvtph_ps(q1_lo);
        const __m256 qf1_hi = _mm256_cvtph_ps(q1_hi);

        #define ACCUM(j, sum_reg) do { \
            const __m128i b0_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16]); \
            const __m128i b0_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16 + 8]); \
            const __m256 bf0_lo = _mm256_cvtph_ps(b0_lo); \
            const __m256 bf0_hi = _mm256_cvtph_ps(b0_hi); \
            sum_reg = _mm256_fmadd_ps(qf0_lo, bf0_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf0_hi, bf0_hi, sum_reg); \
            const __m128i b1_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][(c + 1) * 16]); \
            const __m128i b1_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][(c + 1) * 16 + 8]); \
            const __m256 bf1_lo = _mm256_cvtph_ps(b1_lo); \
            const __m256 bf1_hi = _mm256_cvtph_ps(b1_hi); \
            sum_reg = _mm256_fmadd_ps(qf1_lo, bf1_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf1_hi, bf1_hi, sum_reg); \
        } while(0)

        ACCUM(0, sum0);
        ACCUM(1, sum1);
        ACCUM(2, sum2);
        ACCUM(3, sum3);
        #undef ACCUM
    }
    for (; c < num_chunks; ++c) {
        const __m128i q_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
        const __m128i q_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
        const __m256 qf_lo = _mm256_cvtph_ps(q_lo);
        const __m256 qf_hi = _mm256_cvtph_ps(q_hi);

        #define ACCUM(j, sum_reg) do { \
            const __m128i b_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16]); \
            const __m128i b_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16 + 8]); \
            const __m256 bf_lo = _mm256_cvtph_ps(b_lo); \
            const __m256 bf_hi = _mm256_cvtph_ps(b_hi); \
            sum_reg = _mm256_fmadd_ps(qf_lo, bf_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf_hi, bf_hi, sum_reg); \
        } while(0)

        ACCUM(0, sum0);
        ACCUM(1, sum1);
        ACCUM(2, sum2);
        ACCUM(3, sum3);
        #undef ACCUM
    }

    dists[0] = 1.0f - hsum256(sum0);
    dists[1] = 1.0f - hsum256(sum1);
    dists[2] = 1.0f - hsum256(sum2);
    dists[3] = 1.0f - hsum256(sum3);
#else
    for (int j = 0; j < 4; ++j) {
        dists[j] = 1.0f - deglib::distances::FP16InnerProductDefault::dot(query, db_arr[j], dim);
    }
#endif
}

// ip_custom_batch8_unroll2: Batch=8, process 2 query chunks per outer iteration
static void ip_custom_batch8_unroll2(const uint16_t* query, const uint16_t* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t num_chunks = *dim / 16;

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    __m256 sum4 = _mm256_setzero_ps();
    __m256 sum5 = _mm256_setzero_ps();
    __m256 sum6 = _mm256_setzero_ps();
    __m256 sum7 = _mm256_setzero_ps();

    size_t c = 0;
    for (; c + 1 < num_chunks; c += 2) {
        const __m128i q0_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
        const __m128i q0_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
        const __m256 qf0_lo = _mm256_cvtph_ps(q0_lo);
        const __m256 qf0_hi = _mm256_cvtph_ps(q0_hi);

        const __m128i q1_lo = _mm_loadu_si128((const __m128i*)&query[(c + 1) * 16]);
        const __m128i q1_hi = _mm_loadu_si128((const __m128i*)&query[(c + 1) * 16 + 8]);
        const __m256 qf1_lo = _mm256_cvtph_ps(q1_lo);
        const __m256 qf1_hi = _mm256_cvtph_ps(q1_hi);

        #define ACCUM(j, sum_reg) do { \
            const __m128i b0_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16]); \
            const __m128i b0_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16 + 8]); \
            const __m256 bf0_lo = _mm256_cvtph_ps(b0_lo); \
            const __m256 bf0_hi = _mm256_cvtph_ps(b0_hi); \
            sum_reg = _mm256_fmadd_ps(qf0_lo, bf0_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf0_hi, bf0_hi, sum_reg); \
            const __m128i b1_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][(c + 1) * 16]); \
            const __m128i b1_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][(c + 1) * 16 + 8]); \
            const __m256 bf1_lo = _mm256_cvtph_ps(b1_lo); \
            const __m256 bf1_hi = _mm256_cvtph_ps(b1_hi); \
            sum_reg = _mm256_fmadd_ps(qf1_lo, bf1_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf1_hi, bf1_hi, sum_reg); \
        } while(0)

        ACCUM(0, sum0); ACCUM(1, sum1); ACCUM(2, sum2); ACCUM(3, sum3);
        ACCUM(4, sum4); ACCUM(5, sum5); ACCUM(6, sum6); ACCUM(7, sum7);
        #undef ACCUM
    }
    for (; c < num_chunks; ++c) {
        const __m128i q_lo = _mm_loadu_si128((const __m128i*)&query[c * 16]);
        const __m128i q_hi = _mm_loadu_si128((const __m128i*)&query[c * 16 + 8]);
        const __m256 qf_lo = _mm256_cvtph_ps(q_lo);
        const __m256 qf_hi = _mm256_cvtph_ps(q_hi);

        #define ACCUM(j, sum_reg) do { \
            const __m128i b_lo = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16]); \
            const __m128i b_hi = _mm_loadu_si128((const __m128i*)&db_arr[j][c * 16 + 8]); \
            const __m256 bf_lo = _mm256_cvtph_ps(b_lo); \
            const __m256 bf_hi = _mm256_cvtph_ps(b_hi); \
            sum_reg = _mm256_fmadd_ps(qf_lo, bf_lo, sum_reg); \
            sum_reg = _mm256_fmadd_ps(qf_hi, bf_hi, sum_reg); \
        } while(0)

        ACCUM(0, sum0); ACCUM(1, sum1); ACCUM(2, sum2); ACCUM(3, sum3);
        ACCUM(4, sum4); ACCUM(5, sum5); ACCUM(6, sum6); ACCUM(7, sum7);
        #undef ACCUM
    }

    dists[0] = 1.0f - hsum256(sum0);
    dists[1] = 1.0f - hsum256(sum1);
    dists[2] = 1.0f - hsum256(sum2);
    dists[3] = 1.0f - hsum256(sum3);
    dists[4] = 1.0f - hsum256(sum4);
    dists[5] = 1.0f - hsum256(sum5);
    dists[6] = 1.0f - hsum256(sum6);
    dists[7] = 1.0f - hsum256(sum7);
#else
    for (int j = 0; j < 8; ++j) {
        dists[j] = 1.0f - deglib::distances::FP16InnerProductDefault::dot(query, db_arr[j], dim);
    }
#endif
}
// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

static void bench_fp16_ip(
    const std::vector<std::vector<std::byte>>& fp16_data_vecs,
    size_t count,
    uint32_t dim,
    uint32_t threads)
{
    (void)threads;
    if (count == 0 || dim == 0) {
        std::printf("FP16 IP bench: no data (count=%zu, dim=%u)\n", count, dim);
        return;
    }

    std::vector<uint32_t> shuffled_indices(count);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0u);
    std::mt19937 g(1337);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), g);

    size_t dim_sz = dim;

    size_t comparisons = 100'000;
    comparisons = std::min(comparisons, std::max<size_t>(count * 256, 10'000));

    std::printf("=== FP16 Inner Product Microbenchmark ===\n");
    std::printf("Vectors: %zu, dim=%u\n", count, dim);
    std::printf("bytes/vector=%zu\n", static_cast<size_t>(dim) * sizeof(uint16_t));
    std::printf("Comparisons: %zu\n", comparisons);

    struct BenchResult {
        const char* name;
        double ms;
        double ns_per_op;
    };

    BenchResult best{ "<none>", 0.0, std::numeric_limits<double>::infinity() };

    auto run_case_single = [&](const char* name, auto&& fn) {
        constexpr int trials = 5;

        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>(comparisons / 10, 5'000);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const uint32_t c = static_cast<uint32_t>(((i * 40503ULL) + 17ULL) % count);
            const uint32_t db_idx = shuffled_indices[c];
            const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());
            const uint16_t* b = reinterpret_cast<const uint16_t*>(fp16_data_vecs[db_idx].data());
            sink += fn(a, b, &dim_sz);
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const uint32_t c = static_cast<uint32_t>(((i * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());
                const uint16_t* b = reinterpret_cast<const uint16_t*>(fp16_data_vecs[db_idx].data());
                sink += fn(a, b, &dim_sz);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double ns_per_op = (ms * 1e6) / static_cast<double>(comparisons);

            if (ns_per_op < best_ns_per_op) {
                best_ns_per_op = ns_per_op;
                best_ms = ms;
            }
        }

        std::printf("  %-14s  %10.2f ms  %8.2f ns/op  (best-of-%d, sink=%.3f)\n",
                    name, best_ms, best_ns_per_op, trials, (float)sink);

        if (best_ns_per_op < best.ns_per_op) {
            best = BenchResult{ name, best_ms, best_ns_per_op };
        }
    };

    auto run_case_batch = [&](const char* name, auto&& fn_batch, int batch_size) {
        constexpr int trials = 5;

        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>((comparisons / batch_size) / 10, 50'000);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());

            const uint16_t* b_arr[8];
            for (int j = 0; j < batch_size; ++j) {
                const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                b_arr[j] = reinterpret_cast<const uint16_t*>(fp16_data_vecs[db_idx].data());
            }

            alignas(32) float dists[8];
                fn_batch(a, b_arr, &dim_sz, dists);
            for (int j = 0; j < batch_size; ++j) {
                sink += dists[j];
            }
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons / batch_size; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());

                const uint16_t* b_arr[8];
                for (int j = 0; j < batch_size; ++j) {
                    const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                    const uint32_t db_idx = shuffled_indices[c];
                    b_arr[j] = reinterpret_cast<const uint16_t*>(fp16_data_vecs[db_idx].data());
                }

                alignas(32) float dists[8];
            fn_batch(a, b_arr, &dim_sz, dists);
                for (int j = 0; j < batch_size; ++j) {
                    sink += dists[j];
                }
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double ns_per_op = (ms * 1e6) / static_cast<double>(comparisons);

            if (ns_per_op < best_ns_per_op) {
                best_ns_per_op = ns_per_op;
                best_ms = ms;
            }
        }

        std::printf("  %-14s  %10.2f ms  %8.2f ns/op  (best-of-%d, sink=%.3f)\n",
                    name, best_ms, best_ns_per_op, trials, (float)sink);

        if (best_ns_per_op < best.ns_per_op) {
            best = BenchResult{ name, best_ms, best_ns_per_op };
        }
    };

    // ------------------------------------------------------------------
    // Deglib baseline implementations (single-query) — using dot()
    // ------------------------------------------------------------------
    std::printf("\n--- Deglib baselines (single-query) ---\n");

    run_case_single("ip_naive", deglib::distances::FP16InnerProductDefault::dot);

#if defined(USE_SSE)
    run_case_single("ip_8ext", deglib::distances::FP16InnerProductExt8::dot);
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
    run_case_single("ip_16ext", deglib::distances::FP16InnerProductExt16::dot);
#endif

#if defined(USE_AVX512)
    run_case_single("ip_32ext", deglib::distances::FP16InnerProductExt32::dot);
#endif

    run_case_single("compare", deglib::distances::FP16InnerProductDefault::compare);

    // ------------------------------------------------------------------
    // Custom unrolled variants (single-query)
    // ------------------------------------------------------------------
#if defined(USE_AVX) || defined(USE_AVX512)
    std::printf("\n--- Custom unrolled (single-query) ---\n");

    run_case_single("ip_custom_unroll2", ip_custom_unroll2);
    run_case_single("ip_custom_unroll4", ip_custom_unroll4);
#endif

    // ------------------------------------------------------------------
    // Custom batch implementations
    // ------------------------------------------------------------------
    std::printf("\n--- Custom batch variants ---\n");

    run_case_batch("ip_custom_batch4", ip_custom_batch4, 4);
    run_case_batch("ip_custom_batch8", ip_custom_batch8, 8);
    run_case_batch("ip_custom_batch4_u2", ip_custom_batch4_unroll2, 4);
    run_case_batch("ip_custom_batch8_u2", ip_custom_batch8_unroll2, 8);

    // ------------------------------------------------------------------
    // Deglib compare_batch via FP16InnerProductExt16 (dim%16==0)
    // ------------------------------------------------------------------
    std::printf("\n--- Deglib compare_batch (via FP16InnerProductExt16) ---\n");
    run_case_batch("deglib_batch4",
        [](const uint16_t* q, const uint16_t* const* db, const size_t* d, float* dists) {
            const void* vdb[4];
            for (int j = 0; j < 4; ++j) vdb[j] = db[j];
            deglib::distances::FP16InnerProductExt16::compare_batch(q, vdb, 4, d, dists);
        }, 4);
    run_case_batch("deglib_batch8",
        [](const uint16_t* q, const uint16_t* const* db, const size_t* d, float* dists) {
            const void* vdb[8];
            for (int j = 0; j < 8; ++j) vdb[j] = db[j];
            deglib::distances::FP16InnerProductExt16::compare_batch(q, vdb, 8, d, dists);
        }, 8);

    // ------------------------------------------------------------------
    // Deglib compare_batch via FP16InnerProductExt8 (dim%8==0)
    // ------------------------------------------------------------------
    std::printf("\n--- Deglib compare_batch (via FP16InnerProductExt8) ---\n");
    run_case_batch("deglib_batch4_ext8",
        [](const uint16_t* q, const uint16_t* const* db, const size_t* d, float* dists) {
            const void* vdb[4];
            for (int j = 0; j < 4; ++j) vdb[j] = db[j];
            deglib::distances::FP16InnerProductExt8::compare_batch(q, vdb, 4, d, dists);
        }, 4);
    run_case_batch("deglib_batch8_ext8",
        [](const uint16_t* q, const uint16_t* const* db, const size_t* d, float* dists) {
            const void* vdb[8];
            for (int j = 0; j < 8; ++j) vdb[j] = db[j];
            deglib::distances::FP16InnerProductExt8::compare_batch(q, vdb, 8, d, dists);
        }, 8);

    // ------------------------------------------------------------------
    // Deglib compare_batch via FP16InnerProduct<1> (cascading template)
    // ------------------------------------------------------------------
    std::printf("\n--- Deglib compare_batch (via FP16InnerProduct<1> cascading) ---\n");
    run_case_batch("deglib_batch4_cascade",
        [](const uint16_t* q, const uint16_t* const* db, const size_t* d, float* dists) {
            const void* vdb[4];
            for (int j = 0; j < 4; ++j) vdb[j] = db[j];
            deglib::distances::FP16InnerProduct<1>::compare_batch(q, vdb, 4, d, dists);
        }, 4);
    run_case_batch("deglib_batch8_cascade",
        [](const uint16_t* q, const uint16_t* const* db, const size_t* d, float* dists) {
            const void* vdb[8];
            for (int j = 0; j < 8; ++j) vdb[j] = db[j];
            deglib::distances::FP16InnerProduct<1>::compare_batch(q, vdb, 8, d, dists);
        }, 8);

    std::printf("\nBEST: %s (%.2f ns/op)\n", best.name, best.ns_per_op);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
#if defined(USE_AVX512)
    std::printf("Using AVX-512...\n");
#elif defined(USE_AVX)
    std::printf("Using AVX2...\n");
#elif defined(USE_SSE)
    std::printf("Using SSE...\n");
#else
    std::printf("Using arch...\n");
#endif

    const size_t threads = 6;

    std::filesystem::path data_path;
    if (argc >= 2) {
        data_path = argv[1];
    } else {
#ifdef DATA_PATH
        data_path = DATA_PATH;
#else
        std::fprintf(stderr, "Usage: %s <data_path>\n", argv[0]);
        return 1;
#endif
    }

    std::printf("Data path: %S\n", data_path.c_str());

    const auto train_hvecs = data_path / "train.hvecs";
    std::printf("Looking for: %S\n", train_hvecs.c_str());
    if (!std::filesystem::exists(train_hvecs)) {
        std::fprintf(stderr, "Error: %S not found\n", train_hvecs.c_str());
        return 1;
    }

    double t_load = evp_common::now_ms();
    size_t dims = 0, count = 0;
    std::printf("Loading hvecs...\n");
    fflush(stdout);
    auto train_vectors = evp_common::hvecs_read(train_hvecs.string().c_str(), dims, count);
    std::printf("Loaded: %zu vectors, dim=%zu\n", count, dims);
    fflush(stdout);
    double load_ms = evp_common::now_ms() - t_load;

    std::printf("Loaded %S: %zu vectors, dim=%zu\n", train_hvecs.c_str(), count, dims);
    std::printf("Load time: %.2f ms\n\n", load_ms);

    bench_fp16_ip(train_vectors, count, static_cast<uint32_t>(dims), threads);

    return 0;
}
