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
// Microbench: FP32 L2 Distance — deglib vs custom unrolled/batched variants
// ============================================================================
//
// Benchmarks FP32 L2 (Euclidean) distance implementations:
//   1. Deglib built-in functions (compare, L2Float16Ext, L2Float8Ext, L2Float4Ext)
//   2. Custom unrolled single-query variants (avx2 unroll by 2 and 4)
//   3. Custom batch variants (batch=4, batch=8, with and without unroll)
//
// Formula: sum((a[i] - b[i])^2)
//
// Data: train.hvecs (FP16) is loaded and converted to FP32 at startup.
//
// Output format: "name    ms    ns/op  (best-of-5, sink=...)"
//
// Hardware: 13th Gen Intel Core i7-13850HX (Raptor Cove / Gracemont), AVX2
// Dataset: 200000 vectors, dim=1024, FP32 (converted from train.hvecs)
//
// Results (100K comparisons, best-of-5):
// ---------------------------------------------------------------
//   L2_naive (deglib)          923 ns/op    1.0x baseline
//   L2_16ext (deglib)          644 ns/op    1.4x
//   L2_8ext  (deglib)          631 ns/op    1.5x
//   L2_4ext  (deglib)          727 ns/op    1.3x
//   l2_custom_u2               641 ns/op    1.4x   (unroll2)
//   l2_custom_u4               653 ns/op    1.4x   (unroll4, register pressure)
//   l2_custom_b4               318 ns/op    2.9x   (batch=4)
//   l2_custom_b8               253 ns/op    3.6x   (batch=8)
//   l2_custom_b4_u2            310 ns/op    3.0x   (batch=4, unroll2)
//   l2_custom_b8_u2            253 ns/op    3.6x   (batch=8, unroll2)  ← BEST
// ---------------------------------------------------------------
//
// Key takeaways:
//   * Batch=8 achieves 3.6× over scalar naive (923 → 253 ns/op).
//   * Single-query custom unrolled match deglib ext variants (~640 ns/op).
//   * Unrolling provides no benefit for batch (indexed addressing vs pointers).
//   * L2 performance is very similar to FP32 inner product: same memory
//     bandwidth bottleneck, just sub+FMA instead of load+FMA.
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

// l2_custom_unroll2: AVX2 unroll by 2 (4 accumulators, 32 floats per outer iter)
static float l2_custom_unroll2(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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
        __m256 v0 = _mm256_sub_ps(_mm256_loadu_ps(&a[c * 8]), _mm256_loadu_ps(&b[c * 8]));
        __m256 v1 = _mm256_sub_ps(_mm256_loadu_ps(&a[(c+1) * 8]), _mm256_loadu_ps(&b[(c+1) * 8]));
        __m256 v2 = _mm256_sub_ps(_mm256_loadu_ps(&a[(c+2) * 8]), _mm256_loadu_ps(&b[(c+2) * 8]));
        __m256 v3 = _mm256_sub_ps(_mm256_loadu_ps(&a[(c+3) * 8]), _mm256_loadu_ps(&b[(c+3) * 8]));
        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
        sum2 = _mm256_fmadd_ps(v2, v2, sum2);
        sum3 = _mm256_fmadd_ps(v3, v3, sum3);
    }

    float result = hsum256(sum0) + hsum256(sum1) + hsum256(sum2) + hsum256(sum3);
    for (size_t c = full; c < num_chunks; ++c) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(&a[c * 8]), _mm256_loadu_ps(&b[c * 8]));
        result += hsum256(_mm256_mul_ps(v, v));
    }
    return result;
#else
    return deglib::distances::L2Float::compare(pVect1v, pVect2v, qty_ptr);
#endif
}

// l2_custom_unroll4: AVX2 unroll by 4 (8 accumulators, 64 floats per outer iter)
static float l2_custom_unroll4(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
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
        #define L2_ACCUM(offset, reg) do { \
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(&a[(c+offset) * 8]), _mm256_loadu_ps(&b[(c+offset) * 8])); \
            reg = _mm256_fmadd_ps(v, v, reg); \
        } while(0)

        L2_ACCUM(0, s0); L2_ACCUM(1, s1); L2_ACCUM(2, s2); L2_ACCUM(3, s3);
        L2_ACCUM(4, s4); L2_ACCUM(5, s5); L2_ACCUM(6, s6); L2_ACCUM(7, s7);
        #undef L2_ACCUM
    }

    float result = hsum256(s0) + hsum256(s1) + hsum256(s2) + hsum256(s3)
                 + hsum256(s4) + hsum256(s5) + hsum256(s6) + hsum256(s7);
    for (size_t c = full; c < num_chunks; ++c) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(&a[c * 8]), _mm256_loadu_ps(&b[c * 8]));
        result += hsum256(_mm256_mul_ps(v, v));
    }
    return result;
#else
    return deglib::distances::L2Float::compare(pVect1v, pVect2v, qty_ptr);
#endif
}

// ---------------------------------------------------------------------------
// Custom batch implementations
// Signature: void fn(const float* query, const float* const* db_arr, const size_t* dim, float* dists)
// ---------------------------------------------------------------------------

// l2_custom_batch4: Batch of 4
static void l2_custom_batch4(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t nc = *dim / 8;
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();

    for (size_t c = 0; c < nc; ++c) {
        const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

        #define L2_BATCH(j, reg) do { \
            __m256 v = _mm256_sub_ps(qf, _mm256_loadu_ps(&db_arr[j][c * 8])); \
            reg = _mm256_fmadd_ps(v, v, reg); \
        } while(0)

        L2_BATCH(0, s0); L2_BATCH(1, s1); L2_BATCH(2, s2); L2_BATCH(3, s3);
        #undef L2_BATCH
    }

    dists[0] = hsum256(s0);
    dists[1] = hsum256(s1);
    dists[2] = hsum256(s2);
    dists[3] = hsum256(s3);
#else
    for (int j = 0; j < 4; ++j)
        dists[j] = deglib::distances::L2Float::compare(query, db_arr[j], dim);
#endif
}

// l2_custom_batch8: Batch of 8
static void l2_custom_batch8(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
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
        const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

        #define L2_BATCH(j, reg) do { \
            __m256 v = _mm256_sub_ps(qf, _mm256_loadu_ps(&db_arr[j][c * 8])); \
            reg = _mm256_fmadd_ps(v, v, reg); \
        } while(0)

        L2_BATCH(0, s0); L2_BATCH(1, s1); L2_BATCH(2, s2); L2_BATCH(3, s3);
        L2_BATCH(4, s4); L2_BATCH(5, s5); L2_BATCH(6, s6); L2_BATCH(7, s7);
        #undef L2_BATCH
    }

    dists[0] = hsum256(s0);
    dists[1] = hsum256(s1);
    dists[2] = hsum256(s2);
    dists[3] = hsum256(s3);
    dists[4] = hsum256(s4);
    dists[5] = hsum256(s5);
    dists[6] = hsum256(s6);
    dists[7] = hsum256(s7);
#else
    for (int j = 0; j < 8; ++j)
        dists[j] = deglib::distances::L2Float::compare(query, db_arr[j], dim);
#endif
}

// l2_custom_batch4_unroll2: Batch=4, 2 chunks per outer iteration
static void l2_custom_batch4_unroll2(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const size_t nc = *dim / 8;
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();

    size_t c = 0;
    for (; c + 1 < nc; c += 2) {
        const __m256 qf0 = _mm256_loadu_ps(&query[c * 8]);
        const __m256 qf1 = _mm256_loadu_ps(&query[(c+1) * 8]);

        #define L2_BATCH2(j, reg) do { \
            __m256 v0 = _mm256_sub_ps(qf0, _mm256_loadu_ps(&db_arr[j][c * 8])); \
            __m256 v1 = _mm256_sub_ps(qf1, _mm256_loadu_ps(&db_arr[j][(c+1) * 8])); \
            reg = _mm256_fmadd_ps(v0, v0, reg); \
            reg = _mm256_fmadd_ps(v1, v1, reg); \
        } while(0)

        L2_BATCH2(0, s0); L2_BATCH2(1, s1); L2_BATCH2(2, s2); L2_BATCH2(3, s3);
        #undef L2_BATCH2
    }
    for (; c < nc; ++c) {
        const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

        #define L2_BATCH(j, reg) do { \
            __m256 v = _mm256_sub_ps(qf, _mm256_loadu_ps(&db_arr[j][c * 8])); \
            reg = _mm256_fmadd_ps(v, v, reg); \
        } while(0)

        L2_BATCH(0, s0); L2_BATCH(1, s1); L2_BATCH(2, s2); L2_BATCH(3, s3);
        #undef L2_BATCH
    }

    dists[0] = hsum256(s0);
    dists[1] = hsum256(s1);
    dists[2] = hsum256(s2);
    dists[3] = hsum256(s3);
#else
    for (int j = 0; j < 4; ++j)
        dists[j] = deglib::distances::L2Float::compare(query, db_arr[j], dim);
#endif
}

// l2_custom_batch8_unroll2: Batch=8, 2 chunks per outer iteration
static void l2_custom_batch8_unroll2(const float* query, const float* const* db_arr, const size_t* dim, float* dists) {
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
        const __m256 qf0 = _mm256_loadu_ps(&query[c * 8]);
        const __m256 qf1 = _mm256_loadu_ps(&query[(c+1) * 8]);

        #define L2_BATCH2(j, reg) do { \
            __m256 v0 = _mm256_sub_ps(qf0, _mm256_loadu_ps(&db_arr[j][c * 8])); \
            __m256 v1 = _mm256_sub_ps(qf1, _mm256_loadu_ps(&db_arr[j][(c+1) * 8])); \
            reg = _mm256_fmadd_ps(v0, v0, reg); \
            reg = _mm256_fmadd_ps(v1, v1, reg); \
        } while(0)

        L2_BATCH2(0, s0); L2_BATCH2(1, s1); L2_BATCH2(2, s2); L2_BATCH2(3, s3);
        L2_BATCH2(4, s4); L2_BATCH2(5, s5); L2_BATCH2(6, s6); L2_BATCH2(7, s7);
        #undef L2_BATCH2
    }
    for (; c < nc; ++c) {
        const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

        #define L2_BATCH(j, reg) do { \
            __m256 v = _mm256_sub_ps(qf, _mm256_loadu_ps(&db_arr[j][c * 8])); \
            reg = _mm256_fmadd_ps(v, v, reg); \
        } while(0)

        L2_BATCH(0, s0); L2_BATCH(1, s1); L2_BATCH(2, s2); L2_BATCH(3, s3);
        L2_BATCH(4, s4); L2_BATCH(5, s5); L2_BATCH(6, s6); L2_BATCH(7, s7);
        #undef L2_BATCH
    }

    dists[0] = hsum256(s0);
    dists[1] = hsum256(s1);
    dists[2] = hsum256(s2);
    dists[3] = hsum256(s3);
    dists[4] = hsum256(s4);
    dists[5] = hsum256(s5);
    dists[6] = hsum256(s6);
    dists[7] = hsum256(s7);
#else
    for (int j = 0; j < 8; ++j)
        dists[j] = deglib::distances::L2Float::compare(query, db_arr[j], dim);
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

static void bench_fp32_l2(
    const std::vector<std::vector<float>>& float_data_vecs,
    size_t count,
    uint32_t dim,
    uint32_t threads)
{
    (void)threads;
    if (count == 0 || dim == 0) {
        std::printf("FP32 L2 bench: no data (count=%zu, dim=%u)\n", count, dim);
        return;
    }

    std::vector<uint32_t> shuffled_indices(count);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0u);
    std::mt19937 g(1337);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), g);

    size_t dim_sz = dim;

    size_t comparisons = 100'000;
    comparisons = std::min(comparisons, std::max<size_t>(count * 256, 10'000));

    std::printf("=== FP32 L2 Distance Microbenchmark ===\n");
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
    run_case_single("L2_naive", deglib::distances::L2Float::compare);

#if defined(USE_AVX) || defined(USE_AVX512)
    run_case_single("L2_16ext", deglib::distances::L2Float16Ext::compare);
#endif

#if defined(USE_AVX) || defined(USE_SSE)
    run_case_single("L2_8ext", deglib::distances::L2Float8Ext::compare);
#endif

#if defined(USE_SSE)
    run_case_single("L2_4ext", deglib::distances::L2Float4Ext::compare);
#endif

    // ------------------------------------------------------------------
    // Custom unrolled (single-query)
    // ------------------------------------------------------------------
#if defined(USE_AVX) || defined(USE_AVX512)
    std::printf("\n--- Custom unrolled (single-query) ---\n");
    run_case_single("l2_custom_u2", l2_custom_unroll2);
    run_case_single("l2_custom_u4", l2_custom_unroll4);
#endif

    // ------------------------------------------------------------------
    // Custom batch variants
    // ------------------------------------------------------------------
    std::printf("\n--- Custom batch variants ---\n");
    run_case_batch("l2_custom_b4", l2_custom_batch4, 4);
    run_case_batch("l2_custom_b8", l2_custom_batch8, 8);
    run_case_batch("l2_custom_b4_u2", l2_custom_batch4_unroll2, 4);
    run_case_batch("l2_custom_b8_u2", l2_custom_batch8_unroll2, 8);

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

    bench_fp32_l2(float_vectors, count, static_cast<uint32_t>(dims), threads);

    return 0;
}
