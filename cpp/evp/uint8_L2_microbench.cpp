#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <bit>
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
// Microbench: Uint8 L2 Distance — deglib vs custom unrolled/batched variants
// ============================================================================
//
// Benchmarks uint8 L2 (Euclidean) distance implementations:
//   1. Deglib built-in functions (L2Uint8, L2Uint8Ext32, L2Uint8Ext16)
//   2. Custom unrolled single-query variants (avx2 unroll by 2 and 4)
//   3. Custom batch variants (batch=4, batch=8, with and without unroll)
//
// Formula: sum((a[i] - b[i])^2)
//
// Data: train.hvecs (FP16) is converted to FP32, linearly quantized to uint8
//       using global min/max across all float values.
//
// Output format: "name    ms    ns/op  (best-of-5, sink=...)"
//
// Hardware: 13th Gen Intel Core i7-13850HX (Raptor Cove / Gracemont), AVX2
// Dataset: 200000 vectors, dim=1024, uint8 (quantized from train.hvecs)
//
// Results (100K comparisons, best-of-5):
// ---------------------------------------------------------------
//   u8_naive  (deglib)        545 ns/op    1.0x baseline
//   u8_ext32  (deglib)        361 ns/op    1.5x
//   u8_ext16  (deglib)        492 ns/op    1.1x
//   u8_custom_u2              362 ns/op    1.5x
//   u8_custom_u4              354 ns/op    1.5x   ← best single-query
//   u8_custom_b4              193 ns/op    2.8x   (batch=4)  ← BEST overall
//   u8_custom_b8              427 ns/op    1.3x   (batch=8, scalar — slower!)
//   u8_custom_b4_u2           443 ns/op    1.2x   (batch=4, unroll2 — slower!)
//   u8_custom_b8_u2           386 ns/op    1.4x   (batch=8, unroll2 — slower!)
// ---------------------------------------------------------------
//
// Key takeaways:
//   * Batch=4 wins at 193 ns/op (2.8× over naive) — scalar batch loads query
//     once, amortizing memory traffic across 4 DB vectors.
//   * Batch=8 is SLOWER than batch=4: 8-way scalar loop creates register
//     pressure and 8× the integer work per element without SIMD.
//   * Custom unrolled SIMD (u4, 354 ns/op) matches deglib ext32 but doesn't
//     beat it — the bottleneck shifts away from uint8 computation.
//   * Uint8 is fast: even the naive scalar (545 ns) is 1.7× faster than
//     FP32 naive L2 (923 ns) due to 4× less memory bandwidth.
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: horizontal sum of 4 int32 from __m128i
// ---------------------------------------------------------------------------
static inline int hsum_epi32(__m128i v) {
    alignas(16) int buf[4];
    _mm_store_si128(reinterpret_cast<__m128i*>(buf), v);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

// ---------------------------------------------------------------------------
// FP16 → uint8 quantizer (linear, global min/max)
// ---------------------------------------------------------------------------
static std::vector<std::vector<uint8_t>> quantize_to_uint8(
    const std::vector<std::vector<std::byte>>& fp16_data, size_t dim)
{
    // Step 1: find global min/max of all float values
    float g_min = std::numeric_limits<float>::infinity();
    float g_max = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < fp16_data.size(); ++i) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(fp16_data[i].data());
        for (size_t j = 0; j < dim; ++j) {
            float v = deglib::distances::FP16InnerProduct::fp16_to_float(src[j]);
            if (v < g_min) g_min = v;
            if (v > g_max) g_max = v;
        }
    }

    const float scale = 255.0f / (g_max - g_min);
    std::printf("Uint8 quantize: global range [%.4f, %.4f], scale=%.4f\n", g_min, g_max, scale);

    // Step 2: quantize
    std::vector<std::vector<uint8_t>> result(fp16_data.size());
    for (size_t i = 0; i < fp16_data.size(); ++i) {
        result[i].resize(dim);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(fp16_data[i].data());
        uint8_t* dst = result[i].data();
        for (size_t j = 0; j < dim; ++j) {
            float v = deglib::distances::FP16InnerProduct::fp16_to_float(src[j]);
            float q = (v - g_min) * scale;
            dst[j] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, q + 0.5f)));
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Custom single-query implementations (matching deglib signature:
//   float fn(const void* a, const void* b, const void* qty_ptr))
// ---------------------------------------------------------------------------

// u8_custom_unroll2: AVX2, 32 uint8/iteration, 2 accumulators
static float u8_custom_unroll2(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
    const uint8_t* b = static_cast<const uint8_t*>(pVect2v);
    const size_t size = *static_cast<const size_t*>(qty_ptr);

    __m256i sum0 = _mm256_setzero_si256();
    __m256i sum1 = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        __m128i v1a = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i v2a = _mm_loadu_si128((const __m128i*)(b + i));
        __m128i v1b = _mm_loadu_si128((const __m128i*)(a + i + 16));
        __m128i v2b = _mm_loadu_si128((const __m128i*)(b + i + 16));

        __m256i da = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v1a), _mm256_cvtepu8_epi16(v2a));
        __m256i db = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v1b), _mm256_cvtepu8_epi16(v2b));

        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(da, da));
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(db, db));
    }

    __m256i sum = _mm256_add_epi32(sum0, sum1);
    __m128i lo = _mm256_extracti128_si256(sum, 0);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    int result = hsum_epi32(_mm_add_epi32(lo, hi));

    for (; i < size; ++i) {
        int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
        result += d * d;
    }
    return static_cast<float>(result);
#else
    return deglib::distances::L2Uint8::compare(pVect1v, pVect2v, qty_ptr);
#endif
}

// u8_custom_unroll4: AVX2, 64 uint8/iteration, 4 accumulators
static float u8_custom_unroll4(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX) || defined(USE_AVX512)
    const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
    const uint8_t* b = static_cast<const uint8_t*>(pVect2v);
    const size_t size = *static_cast<const size_t*>(qty_ptr);

    __m256i s0 = _mm256_setzero_si256();
    __m256i s1 = _mm256_setzero_si256();
    __m256i s2 = _mm256_setzero_si256();
    __m256i s3 = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 64 <= size; i += 64) {
        #define U8_L2_BLOCK(off, reg) do { \
            __m128i va = _mm_loadu_si128((const __m128i*)(a + i + off)); \
            __m128i vb = _mm_loadu_si128((const __m128i*)(b + i + off)); \
            __m256i d = _mm256_sub_epi16(_mm256_cvtepu8_epi16(va), _mm256_cvtepu8_epi16(vb)); \
            reg = _mm256_add_epi32(reg, _mm256_madd_epi16(d, d)); \
        } while(0)

        U8_L2_BLOCK(0,  s0);
        U8_L2_BLOCK(16, s1);
        U8_L2_BLOCK(32, s2);
        U8_L2_BLOCK(48, s3);
        #undef U8_L2_BLOCK
    }

    __m256i sum = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
    __m128i lo = _mm256_extracti128_si256(sum, 0);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    int result = hsum_epi32(_mm_add_epi32(lo, hi));

    for (; i < size; ++i) {
        int d = static_cast<int>(a[i]) - static_cast<int>(b[i]);
        result += d * d;
    }
    return static_cast<float>(result);
#else
    return deglib::distances::L2Uint8::compare(pVect1v, pVect2v, qty_ptr);
#endif
}

// ---------------------------------------------------------------------------
// Custom batch implementations
// Signature: void fn(const uint8_t* query, const uint8_t* const* db_arr, const size_t* dim, float* dists)
// ---------------------------------------------------------------------------

// u8_custom_batch4: Batch of 4
static void u8_custom_batch4(const uint8_t* query, const uint8_t* const* db_arr, const size_t* dim, float* dists) {
    const size_t size = *dim;
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    for (size_t i = 0; i < size; ++i) {
        int qv = static_cast<int>(query[i]);
        int d0 = qv - static_cast<int>(db_arr[0][i]); s0 += d0 * d0;
        int d1 = qv - static_cast<int>(db_arr[1][i]); s1 += d1 * d1;
        int d2 = qv - static_cast<int>(db_arr[2][i]); s2 += d2 * d2;
        int d3 = qv - static_cast<int>(db_arr[3][i]); s3 += d3 * d3;
    }

    dists[0] = static_cast<float>(s0);
    dists[1] = static_cast<float>(s1);
    dists[2] = static_cast<float>(s2);
    dists[3] = static_cast<float>(s3);
}

// u8_custom_batch8: Batch of 8
static void u8_custom_batch8(const uint8_t* query, const uint8_t* const* db_arr, const size_t* dim, float* dists) {
    const size_t size = *dim;
    int32_t s[8] = {0};

    for (size_t i = 0; i < size; ++i) {
        int qv = static_cast<int>(query[i]);
        for (int j = 0; j < 8; ++j) {
            int d = qv - static_cast<int>(db_arr[j][i]);
            s[j] += d * d;
        }
    }

    for (int j = 0; j < 8; ++j)
        dists[j] = static_cast<float>(s[j]);
}

// u8_custom_batch4_u2: Batch=4, 2 unrolled inner loop
static void u8_custom_batch4_u2(const uint8_t* query, const uint8_t* const* db_arr, const size_t* dim, float* dists) {
    const size_t size = *dim;
    int32_t s[4] = {0};

    size_t i = 0;
    for (; i + 1 < size; i += 2) {
        int q0 = static_cast<int>(query[i]);
        int q1 = static_cast<int>(query[i + 1]);
        for (int j = 0; j < 4; ++j) {
            int d0 = q0 - static_cast<int>(db_arr[j][i]);
            int d1 = q1 - static_cast<int>(db_arr[j][i + 1]);
            s[j] += d0 * d0 + d1 * d1;
        }
    }
    for (; i < size; ++i) {
        int qv = static_cast<int>(query[i]);
        for (int j = 0; j < 4; ++j) {
            int d = qv - static_cast<int>(db_arr[j][i]);
            s[j] += d * d;
        }
    }

    for (int j = 0; j < 4; ++j)
        dists[j] = static_cast<float>(s[j]);
}

// u8_custom_batch8_u2: Batch=8, 2 unrolled inner loop
static void u8_custom_batch8_u2(const uint8_t* query, const uint8_t* const* db_arr, const size_t* dim, float* dists) {
    const size_t size = *dim;
    int32_t s[8] = {0};

    size_t i = 0;
    for (; i + 1 < size; i += 2) {
        int q0 = static_cast<int>(query[i]);
        int q1 = static_cast<int>(query[i + 1]);
        for (int j = 0; j < 8; ++j) {
            int d0 = q0 - static_cast<int>(db_arr[j][i]);
            int d1 = q1 - static_cast<int>(db_arr[j][i + 1]);
            s[j] += d0 * d0 + d1 * d1;
        }
    }
    for (; i < size; ++i) {
        int qv = static_cast<int>(query[i]);
        for (int j = 0; j < 8; ++j) {
            int d = qv - static_cast<int>(db_arr[j][i]);
            s[j] += d * d;
        }
    }

    for (int j = 0; j < 8; ++j)
        dists[j] = static_cast<float>(s[j]);
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

static void bench_u8_l2(
    const std::vector<std::vector<uint8_t>>& u8_data_vecs,
    size_t count,
    uint32_t dim,
    uint32_t threads)
{
    (void)threads;
    if (count == 0 || dim == 0) {
        std::printf("Uint8 L2 bench: no data (count=%zu, dim=%u)\n", count, dim);
        return;
    }

    std::vector<uint32_t> shuffled_indices(count);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0u);
    std::mt19937 g(1337);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), g);

    size_t dim_sz = dim;

    size_t comparisons = 100'000;
    comparisons = std::min(comparisons, std::max<size_t>(count * 256, 10'000));

    std::printf("=== Uint8 L2 Distance Microbenchmark ===\n");
    std::printf("Vectors: %zu, dim=%u\n", count, dim);
    std::printf("bytes/vector=%zu\n", static_cast<size_t>(dim) * sizeof(uint8_t));
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
            const uint8_t* a = u8_data_vecs[q].data();
            const uint8_t* b = u8_data_vecs[db_idx].data();
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
                const uint8_t* a = u8_data_vecs[q].data();
                const uint8_t* b = u8_data_vecs[db_idx].data();
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
            const uint8_t* a = u8_data_vecs[q].data();
            const uint8_t* b_arr[8];
            for (int j = 0; j < batch_size; ++j) {
                const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                b_arr[j] = u8_data_vecs[db_idx].data();
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
                const uint8_t* a = u8_data_vecs[q].data();
                const uint8_t* b_arr[8];
                for (int j = 0; j < batch_size; ++j) {
                    const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                    const uint32_t db_idx = shuffled_indices[c];
                    b_arr[j] = u8_data_vecs[db_idx].data();
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
    run_case_single("u8_naive", deglib::distances::L2Uint8::compare);

#if defined(USE_AVX) || defined(USE_AVX512) || defined(USE_SSE)
    run_case_single("u8_ext32", deglib::distances::L2Uint8Ext32::compare);
#endif

#if defined(USE_SSE)
    run_case_single("u8_ext16", deglib::distances::L2Uint8Ext16::compare);
#endif

    // ------------------------------------------------------------------
    // Custom unrolled (single-query)
    // ------------------------------------------------------------------
#if defined(USE_AVX) || defined(USE_AVX512)
    std::printf("\n--- Custom unrolled (single-query) ---\n");
    run_case_single("u8_custom_u2", u8_custom_unroll2);
    run_case_single("u8_custom_u4", u8_custom_unroll4);
#endif

    // ------------------------------------------------------------------
    // Custom batch variants
    // ------------------------------------------------------------------
    std::printf("\n--- Custom batch variants ---\n");
    run_case_batch("u8_custom_b4", u8_custom_batch4, 4);
    run_case_batch("u8_custom_b8", u8_custom_batch8, 8);
    run_case_batch("u8_custom_b4_u2", u8_custom_batch4_u2, 4);
    run_case_batch("u8_custom_b8_u2", u8_custom_batch8_u2, 8);

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

    std::printf("Quantizing FP16->uint8 (linear, global min/max)...\n");
    fflush(stdout);
    auto u8_vectors = quantize_to_uint8(fp16_vectors, dims);
    double load_ms = evp_common::now_ms() - t_load;

    std::printf("Loaded %S: %zu vectors, dim=%zu\n", train_hvecs.c_str(), count, dims);
    std::printf("Load + quantize time: %.2f ms\n\n", load_ms);

    fp16_vectors.clear();
    fp16_vectors.shrink_to_fit();

    bench_u8_l2(u8_vectors, count, static_cast<uint32_t>(dims), threads);

    return 0;
}
