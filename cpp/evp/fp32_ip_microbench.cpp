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
// Microbench: FP32 Inner Product — deglib vs custom unrolled/batched variants
// ============================================================================
//
// Benchmarks FP32 inner product implementations:
//   1. Deglib built-in distance functions (ip_naive, ip_16ext, ip_8ext, ip_4ext, compare)
//   2. Custom unrolled single-query variants (avx2 unroll by 2 and 4)
//   3. Custom batch variants (batch=4, batch=8, with and without unroll)
//
// Data: train.hvecs (FP16) is loaded and converted to FP32 at startup.
//
// Output format: "name    ms    ns/op  (best-of-5, sink=...)"
//
// Hardware: AMD 5600G with AVX2 support
// Dataset: 200000 vectors, dim=1024, FP32 (converted from train.hvecs)
//
// Results (100K comparisons, best-of-5):
// ---------------------------------------------------------------
//   ip_naive (deglib)           916 ns/op    1.0x baseline
//   ip_16ext (deglib)           635 ns/op    1.4x
//   ip_8ext  (deglib)           630 ns/op    1.5x
//   ip_4ext  (deglib)           771 ns/op    1.2x
//   compare  (deglib)           921 ns/op    1.0x
//   ip_custom_unroll2           616 ns/op    1.5x
//   ip_custom_unroll4           628 ns/op    1.5x   (register pressure)
//   ip_custom_batch4            285 ns/op    3.2x   (batch=4)
//   ip_custom_batch8            231 ns/op    4.0x   (batch=8)
//   ip_custom_batch4_u2         288 ns/op    3.2x   (batch=4, unroll2)
//   ip_custom_batch8_u2         230 ns/op    4.0x   (batch=8, unroll2)  ← BEST
// ---------------------------------------------------------------
//
// Key takeaways:
//   * Batch variants dominate: batch=8 achieves 4.0× over ip_naive.
//   * Single-query custom unrolled roughly match deglib's ip_8ext/16ext
//     — indexed addressing (a[c*8]) vs pointer-chasing (a += 8) is close.
//   * ip_custom_unroll4 ties unroll2 (register pressure balances throughput).
//   * FP32 batch8_u2 (230 ns/op) is ~1.4× slower than FP16 batch8_u2 (169 ns/op)
//     due to 2× memory bandwidth and no conversion bottleneck to amortize.
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: horizontal sum of a 256-bit register (4+4 → scalar)
// ---------------------------------------------------------------------------
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    alignas(16) float f[4];
    _mm_store_ps(f, s);
    return f[0] + f[1] + f[2] + f[3];
}

// ---------------------------------------------------------------------------
// Custom single-query implementations (matching deglib signature:
//   float fn(const void* a, const void* b, const void* qty_ptr))
// ---------------------------------------------------------------------------

// ip_custom_unroll2: AVX2 unroll by 2 (4 accumulators, 32 floats per outer iter)
static float ip_custom_unroll2(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const float* a = (const float*)pVect1v;
    const float* b = (const float*)pVect2v;
    size_t size = *((size_t*)qty_ptr);

    size_t num_chunks = size / 8;
    size_t full = num_chunks & ~3ULL;

    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    for (size_t c = 0; c < full; c += 4) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[c * 8]), _mm256_loadu_ps(&b[c * 8]), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[(c+1) * 8]), _mm256_loadu_ps(&b[(c+1) * 8]), sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[(c+2) * 8]), _mm256_loadu_ps(&b[(c+2) * 8]), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(&a[(c+3) * 8]), _mm256_loadu_ps(&b[(c+3) * 8]), sum3);
    }

    float result = hsum256(sum0) + hsum256(sum1) + hsum256(sum2) + hsum256(sum3);
    for (size_t c = full; c < num_chunks; ++c) {
        result += hsum256(_mm256_mul_ps(_mm256_loadu_ps(&a[c * 8]), _mm256_loadu_ps(&b[c * 8])));
    }
    return result;
#else
    return deglib::distances::InnerProductFloat::ip_naive(pVect1v, pVect2v, qty_ptr);
#endif
}

// ip_custom_unroll4: AVX2 unroll by 4 (8 accumulators, 64 floats per outer iter)
static float ip_custom_unroll4(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const float* a = (const float*)pVect1v;
    const float* b = (const float*)pVect2v;
    size_t size = *((size_t*)qty_ptr);

    size_t num_chunks = size / 8;
    size_t full = num_chunks & ~7ULL;

    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    __m256 s4 = _mm256_setzero_ps();
    __m256 s5 = _mm256_setzero_ps();
    __m256 s6 = _mm256_setzero_ps();
    __m256 s7 = _mm256_setzero_ps();

    for (size_t c = 0; c < full; c += 8) {
        #define LOAD_ACCUM(offset, reg) \
            reg = _mm256_fmadd_ps(_mm256_loadu_ps(&a[(c+offset) * 8]), _mm256_loadu_ps(&b[(c+offset) * 8]), reg)
        LOAD_ACCUM(0, s0); LOAD_ACCUM(1, s1); LOAD_ACCUM(2, s2); LOAD_ACCUM(3, s3);
        LOAD_ACCUM(4, s4); LOAD_ACCUM(5, s5); LOAD_ACCUM(6, s6); LOAD_ACCUM(7, s7);
        #undef LOAD_ACCUM
    }

    float result = hsum256(s0) + hsum256(s1) + hsum256(s2) + hsum256(s3)
                 + hsum256(s4) + hsum256(s5) + hsum256(s6) + hsum256(s7);
    for (size_t c = full; c < num_chunks; ++c) {
        result += hsum256(_mm256_mul_ps(_mm256_loadu_ps(&a[c * 8]), _mm256_loadu_ps(&b[c * 8])));
    }
    return result;
#else
    return deglib::distances::InnerProductFloat::ip_naive(pVect1v, pVect2v, qty_ptr);
#endif
}

// ---------------------------------------------------------------------------
// Custom batch implementations
// Signature: void fn(const float* query, const float* const* db_arr, const size_t* dim, float* dists)
// ---------------------------------------------------------------------------

// ip_custom_batch4: Batch of 4
static void ip_custom_batch4(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t nc = *dim / 8;
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();

    for (size_t c = 0; c < nc; ++c) {
        __m256 qf = _mm256_loadu_ps(&query[c * 8]);
        s0 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[0][c * 8]), s0);
        s1 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[1][c * 8]), s1);
        s2 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[2][c * 8]), s2);
        s3 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[3][c * 8]), s3);
    }

    dists[0] = 1.0f - hsum256(s0);
    dists[1] = 1.0f - hsum256(s1);
    dists[2] = 1.0f - hsum256(s2);
    dists[3] = 1.0f - hsum256(s3);
#else
    for (int j = 0; j < 4; ++j)
        dists[j] = 1.0f - deglib::distances::InnerProductFloat::ip_naive(query, db_arr[j], dim);
#endif
}

// ip_custom_batch8: Batch of 8
static void ip_custom_batch8(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t nc = *dim / 8;
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    __m256 s4 = _mm256_setzero_ps();
    __m256 s5 = _mm256_setzero_ps();
    __m256 s6 = _mm256_setzero_ps();
    __m256 s7 = _mm256_setzero_ps();

    for (size_t c = 0; c < nc; ++c) {
        __m256 qf = _mm256_loadu_ps(&query[c * 8]);
        s0 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[0][c * 8]), s0);
        s1 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[1][c * 8]), s1);
        s2 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[2][c * 8]), s2);
        s3 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[3][c * 8]), s3);
        s4 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[4][c * 8]), s4);
        s5 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[5][c * 8]), s5);
        s6 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[6][c * 8]), s6);
        s7 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[7][c * 8]), s7);
    }

    dists[0] = 1.0f - hsum256(s0);
    dists[1] = 1.0f - hsum256(s1);
    dists[2] = 1.0f - hsum256(s2);
    dists[3] = 1.0f - hsum256(s3);
    dists[4] = 1.0f - hsum256(s4);
    dists[5] = 1.0f - hsum256(s5);
    dists[6] = 1.0f - hsum256(s6);
    dists[7] = 1.0f - hsum256(s7);
#else
    for (int j = 0; j < 8; ++j)
        dists[j] = 1.0f - deglib::distances::InnerProductFloat::ip_naive(query, db_arr[j], dim);
#endif
}

// ip_custom_batch4_unroll2: Batch=4, 2 chunks per outer iteration
static void ip_custom_batch4_unroll2(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t nc = *dim / 8;
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();

    size_t c = 0;
    for (; c + 1 < nc; c += 2) {
        __m256 qf0 = _mm256_loadu_ps(&query[c * 8]);
        __m256 qf1 = _mm256_loadu_ps(&query[(c + 1) * 8]);

        s0 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[0][c * 8]), s0);
        s0 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[0][(c+1) * 8]), s0);
        s1 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[1][c * 8]), s1);
        s1 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[1][(c+1) * 8]), s1);
        s2 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[2][c * 8]), s2);
        s2 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[2][(c+1) * 8]), s2);
        s3 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[3][c * 8]), s3);
        s3 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[3][(c+1) * 8]), s3);
    }
    for (; c < nc; ++c) {
        __m256 qf = _mm256_loadu_ps(&query[c * 8]);
        s0 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[0][c * 8]), s0);
        s1 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[1][c * 8]), s1);
        s2 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[2][c * 8]), s2);
        s3 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[3][c * 8]), s3);
    }

    dists[0] = 1.0f - hsum256(s0);
    dists[1] = 1.0f - hsum256(s1);
    dists[2] = 1.0f - hsum256(s2);
    dists[3] = 1.0f - hsum256(s3);
#else
    for (int j = 0; j < 4; ++j)
        dists[j] = 1.0f - deglib::distances::InnerProductFloat::ip_naive(query, db_arr[j], dim);
#endif
}

// ip_custom_batch8_unroll2: Batch=8, 2 chunks per outer iteration
static void ip_custom_batch8_unroll2(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t nc = *dim / 8;
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    __m256 s4 = _mm256_setzero_ps();
    __m256 s5 = _mm256_setzero_ps();
    __m256 s6 = _mm256_setzero_ps();
    __m256 s7 = _mm256_setzero_ps();

    size_t c = 0;
    for (; c + 1 < nc; c += 2) {
        __m256 qf0 = _mm256_loadu_ps(&query[c * 8]);
        __m256 qf1 = _mm256_loadu_ps(&query[(c + 1) * 8]);

        s0 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[0][c * 8]), s0);
        s0 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[0][(c+1) * 8]), s0);
        s1 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[1][c * 8]), s1);
        s1 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[1][(c+1) * 8]), s1);
        s2 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[2][c * 8]), s2);
        s2 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[2][(c+1) * 8]), s2);
        s3 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[3][c * 8]), s3);
        s3 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[3][(c+1) * 8]), s3);
        s4 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[4][c * 8]), s4);
        s4 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[4][(c+1) * 8]), s4);
        s5 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[5][c * 8]), s5);
        s5 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[5][(c+1) * 8]), s5);
        s6 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[6][c * 8]), s6);
        s6 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[6][(c+1) * 8]), s6);
        s7 = _mm256_fmadd_ps(qf0, _mm256_loadu_ps(&db_arr[7][c * 8]), s7);
        s7 = _mm256_fmadd_ps(qf1, _mm256_loadu_ps(&db_arr[7][(c+1) * 8]), s7);
    }
    for (; c < nc; ++c) {
        __m256 qf = _mm256_loadu_ps(&query[c * 8]);
        s0 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[0][c * 8]), s0);
        s1 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[1][c * 8]), s1);
        s2 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[2][c * 8]), s2);
        s3 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[3][c * 8]), s3);
        s4 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[4][c * 8]), s4);
        s5 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[5][c * 8]), s5);
        s6 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[6][c * 8]), s6);
        s7 = _mm256_fmadd_ps(qf, _mm256_loadu_ps(&db_arr[7][c * 8]), s7);
    }

    dists[0] = 1.0f - hsum256(s0);
    dists[1] = 1.0f - hsum256(s1);
    dists[2] = 1.0f - hsum256(s2);
    dists[3] = 1.0f - hsum256(s3);
    dists[4] = 1.0f - hsum256(s4);
    dists[5] = 1.0f - hsum256(s5);
    dists[6] = 1.0f - hsum256(s6);
    dists[7] = 1.0f - hsum256(s7);
#else
    for (int j = 0; j < 8; ++j)
        dists[j] = 1.0f - deglib::distances::InnerProductFloat::ip_naive(query, db_arr[j], dim);
#endif
}

// ---------------------------------------------------------------------------
// FP16 → FP32 converter
// ---------------------------------------------------------------------------
static std::vector<std::vector<float>> convert_to_float(
    const std::vector<std::vector<std::byte>>& fp16_data, size_t dim)
{
    std::vector<std::vector<float>> result(fp16_data.size());
    for (size_t i = 0; i < fp16_data.size(); ++i) {
        result[i].resize(dim);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(fp16_data[i].data());
        float* dst = result[i].data();
#if defined(USE_AVX) || defined(USE_AVX512)
        size_t j = 0;
        for (; j + 16 <= dim; j += 16) {
            __m128i lo = _mm_loadu_si128((const __m128i*)&src[j]);
            __m128i hi = _mm_loadu_si128((const __m128i*)&src[j + 8]);
            _mm256_storeu_ps(&dst[j], _mm256_cvtph_ps(lo));
            _mm256_storeu_ps(&dst[j + 8], _mm256_cvtph_ps(hi));
        }
        for (; j < dim; ++j)
            dst[j] = deglib::distances::FP16InnerProduct::fp16_to_float(src[j]);
#else
        for (size_t j = 0; j < dim; ++j)
            dst[j] = deglib::distances::FP16InnerProduct::fp16_to_float(src[j]);
#endif
    }
    return result;
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

static void bench_fp32_ip(
    const std::vector<std::vector<float>>& float_data_vecs,
    size_t count,
    uint32_t dim,
    uint32_t threads)
{
    (void)threads;
    if (count == 0 || dim == 0) {
        std::printf("FP32 IP bench: no data (count=%zu, dim=%u)\n", count, dim);
        return;
    }

    std::vector<uint32_t> shuffled_indices(count);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0u);
    std::mt19937 g(1337);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), g);

    size_t dim_sz = dim;

    size_t comparisons = 100'000;
    comparisons = std::min(comparisons, std::max<size_t>(count * 256, 10'000));

    std::printf("=== FP32 Inner Product Microbenchmark ===\n");
    std::printf("Vectors: %zu, dim=%u\n", count, dim);
    std::printf("bytes/vector=%zu\n", static_cast<size_t>(dim) * sizeof(float));
    std::printf("Comparisons: %zu\n", comparisons);

    struct BenchResult { const char* name; double ms; double ns_per_op; };
    BenchResult best{ "<none>", 0.0, std::numeric_limits<double>::infinity() };

    auto run_case_single = [&](const char* name, auto&& fn) {
        constexpr int trials = 5;
        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>(comparisons / 10, 5'000);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const uint32_t c = static_cast<uint32_t>(((i * 40503ULL) + 17ULL) % count);
            const uint32_t db_idx = shuffled_indices[c];
            const float* a = float_data_vecs[q].data();
            const float* b = float_data_vecs[db_idx].data();
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
                const float* a = float_data_vecs[q].data();
                const float* b = float_data_vecs[db_idx].data();
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

        if (best_ns_per_op < best.ns_per_op)
            best = BenchResult{ name, best_ms, best_ns_per_op };
    };

    auto run_case_batch = [&](const char* name, auto&& fn_batch, int batch_size) {
        constexpr int trials = 5;
        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>((comparisons / batch_size) / 10, 50'000);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const float* a = float_data_vecs[q].data();
            const float* b_arr[8];
            for (int j = 0; j < batch_size; ++j) {
                const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                b_arr[j] = float_data_vecs[db_idx].data();
            }
            alignas(32) float dists[8];
            fn_batch(a, b_arr, &dim_sz, dists);
            for (int j = 0; j < batch_size; ++j)
                sink += dists[j];
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons / batch_size; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const float* a = float_data_vecs[q].data();
                const float* b_arr[8];
                for (int j = 0; j < batch_size; ++j) {
                    const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                    const uint32_t db_idx = shuffled_indices[c];
                    b_arr[j] = float_data_vecs[db_idx].data();
                }
                alignas(32) float dists[8];
                fn_batch(a, b_arr, &dim_sz, dists);
                for (int j = 0; j < batch_size; ++j)
                    sink += dists[j];
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

        if (best_ns_per_op < best.ns_per_op)
            best = BenchResult{ name, best_ms, best_ns_per_op };
    };

    // ------------------------------------------------------------------
    // Deglib baselines (single-query)
    // ------------------------------------------------------------------
    std::printf("\n--- Deglib baselines (single-query) ---\n");
    run_case_single("ip_naive", deglib::distances::InnerProductFloat::ip_naive);

#if defined(USE_AVX) || defined(USE_AVX512)
    run_case_single("ip_16ext", deglib::distances::InnerProductFloat16Ext::ip_16ext);
#endif

#if defined(USE_AVX) || defined(USE_SSE)
    run_case_single("ip_8ext", deglib::distances::InnerProductFloat8Ext::ip_8ext);
#endif

#if defined(USE_SSE)
    run_case_single("ip_4ext", deglib::distances::InnerProductFloat4Ext::ip_4ext);
#endif

    run_case_single("compare", deglib::distances::InnerProductFloat::compare);

    // ------------------------------------------------------------------
    // Custom unrolled (single-query)
    // ------------------------------------------------------------------
#if defined(USE_AVX) || defined(USE_AVX512)
    std::printf("\n--- Custom unrolled (single-query) ---\n");
    run_case_single("ip_custom_unroll2", ip_custom_unroll2);
    run_case_single("ip_custom_unroll4", ip_custom_unroll4);
#endif

    // ------------------------------------------------------------------
    // Custom batch variants
    // ------------------------------------------------------------------
    std::printf("\n--- Custom batch variants ---\n");
    run_case_batch("ip_custom_batch4", ip_custom_batch4, 4);
    run_case_batch("ip_custom_batch8", ip_custom_batch8, 8);
    run_case_batch("ip_custom_batch4_u2", ip_custom_batch4_unroll2, 4);
    run_case_batch("ip_custom_batch8_u2", ip_custom_batch8_unroll2, 8);

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
    auto fp16_vectors = evp_common::hvecs_read(train_hvecs.string().c_str(), dims, count);
    std::printf("Loaded: %zu vectors, dim=%zu\n", count, dims);
    fflush(stdout);

    std::printf("Converting FP16->FP32...\n");
    fflush(stdout);
    auto float_vectors = convert_to_float(fp16_vectors, dims);
    double load_ms = evp_common::now_ms() - t_load;

    std::printf("Loaded %S: %zu vectors, dim=%zu\n", train_hvecs.c_str(), count, dims);
    std::printf("Load + convert time: %.2f ms\n\n", load_ms);

    fp16_vectors.clear();
    fp16_vectors.shrink_to_fit();

    bench_fp32_ip(float_vectors, count, static_cast<uint32_t>(dims), threads);

    return 0;
}
