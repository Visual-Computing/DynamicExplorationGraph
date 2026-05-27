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
#include <bit>

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
// Microbench: EVP Bits Similarity — deglib vs custom unrolled/batched variants
// ============================================================================
//
// Benchmarks EVP bit-vector similarity:
//   1. Deglib built-in functions (naive, avx2, compare)
//   2. Custom unrolled single-query variants (avx2 unroll by 2 and 4)
//   3. Custom batch variants (batch=4, batch=8)
//
// Data: train.hvecs (FP16) is quantized to EVP bits (NON_ZEROS=512) at startup.
//
// Each EVP vector layout: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
//
// Formula: sim = (aa + bb + dim) - (cc + dd)
//   aa = popcount(ones(a) & ones(b))
//   bb = popcount(negs(a) & negs(b))
//   cc = popcount(ones(a) & negs(b))
//   dd = popcount(ones(b) & negs(a))
// distance = 1.0f - sim / (2.0f * dim)
//
// Output format: "name    ms    ns/op  (best-of-5, sink=...)"
//
// Hardware: 13th Gen Intel Core i7-13850HX (Raptor Cove / Gracemont), AVX2
// Dataset: 200000 vectors, dim=1024, NON_ZEROS=512, EVP=256 bytes/vector
//
// Results (100K comparisons, best-of-5):
// ---------------------------------------------------------------
//   naive  (deglib)           182 ns/op    1.0x baseline
//   avx2   (deglib)           177 ns/op    1.0x
//   compare(deglib)           176 ns/op    1.0x
//   evp_custom_u2             178 ns/op    1.0x   (unroll2, AVX2)
//   evp_custom_u4             184 ns/op    1.0x   (unroll4, AVX2)
//   evp_custom_b4              87 ns/op    2.1x   (batch=4)
//   evp_custom_b8              71 ns/op    2.6x   (batch=8)  ← BEST
// ---------------------------------------------------------------
//
// Key takeaways:
//   * Scalar naive (uint64_t + std::popcount) matches AVX2 shuffle popcount.
//     With only 256 bytes per vector and fast hardware popcnt, SIMD offers
//     no single-query advantage.
//   * Batch variants still win by amortizing query memory loads:
//     batch=4 → 2.1×, batch=8 → 2.6×.
//   * Batch gains are smaller than for FP16 (8.4×) because there is no
//     FP16→float conversion cost to amortize.
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: horizontal sum of 4 int64 from __m256i
// ---------------------------------------------------------------------------
static inline uint64_t hsum_epi64(__m256i v) {
    alignas(32) int64_t buf[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), v);
    return static_cast<uint64_t>(buf[0] + buf[1] + buf[2] + buf[3]);
}

// ---------------------------------------------------------------------------
// Custom single-query implementations (matching deglib EVP signature:
//   float fn(const void* a, const void* b, const void* qty_ptr))
// ---------------------------------------------------------------------------

// evp_custom_unroll2: 64 bytes/iteration, 8 accumulators
static float evp_custom_unroll2(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX)
    const std::byte* a = static_cast<const std::byte*>(pVect1v);
    const std::byte* b = static_cast<const std::byte*>(pVect2v);
    const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
    const size_t mask_bytes = dim / 8;

    const std::byte* ones_a = a;
    const std::byte* negs_a = a + mask_bytes;
    const std::byte* ones_b = b;
    const std::byte* negs_b = b + mask_bytes;

    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();

    auto popcnt = [&](__m256i vec) -> __m256i {
        __m256i lo = _mm256_and_si256(vec, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
        return _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
    };

    __m256i aa0 = zero, bb0 = zero, cc0 = zero, dd0 = zero;
    __m256i aa1 = zero, bb1 = zero, cc1 = zero, dd1 = zero;

    size_t i = 0;
    for (; i + 64 <= mask_bytes; i += 64) {
        __m256i o1 = _mm256_loadu_si256((const __m256i*)&ones_a[i]);
        __m256i o2 = _mm256_loadu_si256((const __m256i*)&ones_b[i]);
        __m256i n1 = _mm256_loadu_si256((const __m256i*)&negs_a[i]);
        __m256i n2 = _mm256_loadu_si256((const __m256i*)&negs_b[i]);
        aa0 = _mm256_add_epi64(aa0, _mm256_sad_epu8(popcnt(_mm256_and_si256(o1, o2)), zero));
        bb0 = _mm256_add_epi64(bb0, _mm256_sad_epu8(popcnt(_mm256_and_si256(n1, n2)), zero));
        cc0 = _mm256_add_epi64(cc0, _mm256_sad_epu8(popcnt(_mm256_and_si256(o1, n2)), zero));
        dd0 = _mm256_add_epi64(dd0, _mm256_sad_epu8(popcnt(_mm256_and_si256(o2, n1)), zero));

        o1 = _mm256_loadu_si256((const __m256i*)&ones_a[i + 32]);
        o2 = _mm256_loadu_si256((const __m256i*)&ones_b[i + 32]);
        n1 = _mm256_loadu_si256((const __m256i*)&negs_a[i + 32]);
        n2 = _mm256_loadu_si256((const __m256i*)&negs_b[i + 32]);
        aa1 = _mm256_add_epi64(aa1, _mm256_sad_epu8(popcnt(_mm256_and_si256(o1, o2)), zero));
        bb1 = _mm256_add_epi64(bb1, _mm256_sad_epu8(popcnt(_mm256_and_si256(n1, n2)), zero));
        cc1 = _mm256_add_epi64(cc1, _mm256_sad_epu8(popcnt(_mm256_and_si256(o1, n2)), zero));
        dd1 = _mm256_add_epi64(dd1, _mm256_sad_epu8(popcnt(_mm256_and_si256(o2, n1)), zero));
    }

    __m256i aa = _mm256_add_epi64(aa0, aa1);
    __m256i bb = _mm256_add_epi64(bb0, bb1);
    __m256i cc = _mm256_add_epi64(cc0, cc1);
    __m256i dd = _mm256_add_epi64(dd0, dd1);

    uint64_t aa_u = hsum_epi64(aa);
    uint64_t bb_u = hsum_epi64(bb);
    uint64_t cc_u = hsum_epi64(cc);
    uint64_t dd_u = hsum_epi64(dd);

    for (; i < mask_bytes; ++i) {
        unsigned int o1b = static_cast<unsigned int>(static_cast<uint8_t>(ones_a[i]));
        unsigned int o2b = static_cast<unsigned int>(static_cast<uint8_t>(ones_b[i]));
        unsigned int n1b = static_cast<unsigned int>(static_cast<uint8_t>(negs_a[i]));
        unsigned int n2b = static_cast<unsigned int>(static_cast<uint8_t>(negs_b[i]));
        aa_u += std::popcount(o1b & o2b);
        bb_u += std::popcount(n1b & n2b);
        cc_u += std::popcount(o1b & n2b);
        dd_u += std::popcount(o2b & n1b);
    }

    const float similarity = static_cast<float>(aa_u + bb_u + dim) - static_cast<float>(cc_u + dd_u);
    return 1.0f - similarity / (2.0f * static_cast<float>(dim));
#else
    return deglib::distances::EvpInnerProduct::compare_naive(pVect1v, pVect2v, qty_ptr);
#endif
}

// evp_custom_unroll4: 128 bytes/iteration, 16 accumulators
static float evp_custom_unroll4(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX)
    const std::byte* a = static_cast<const std::byte*>(pVect1v);
    const std::byte* b = static_cast<const std::byte*>(pVect2v);
    const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
    const size_t mask_bytes = dim / 8;

    const std::byte* ones_a = a;
    const std::byte* negs_a = a + mask_bytes;
    const std::byte* ones_b = b;
    const std::byte* negs_b = b + mask_bytes;

    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();

    auto popcnt = [&](__m256i vec) -> __m256i {
        __m256i lo = _mm256_and_si256(vec, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
        return _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));
    };

    __m256i aa0 = zero, bb0 = zero, cc0 = zero, dd0 = zero;
    __m256i aa1 = zero, bb1 = zero, cc1 = zero, dd1 = zero;
    __m256i aa2 = zero, bb2 = zero, cc2 = zero, dd2 = zero;
    __m256i aa3 = zero, bb3 = zero, cc3 = zero, dd3 = zero;

    size_t i = 0;
    for (; i + 128 <= mask_bytes; i += 128) {
        #define BLOCK(off, aa_r, bb_r, cc_r, dd_r) do { \
            __m256i o1 = _mm256_loadu_si256((const __m256i*)&ones_a[i + off]); \
            __m256i o2 = _mm256_loadu_si256((const __m256i*)&ones_b[i + off]); \
            __m256i n1 = _mm256_loadu_si256((const __m256i*)&negs_a[i + off]); \
            __m256i n2 = _mm256_loadu_si256((const __m256i*)&negs_b[i + off]); \
            aa_r = _mm256_add_epi64(aa_r, _mm256_sad_epu8(popcnt(_mm256_and_si256(o1, o2)), zero)); \
            bb_r = _mm256_add_epi64(bb_r, _mm256_sad_epu8(popcnt(_mm256_and_si256(n1, n2)), zero)); \
            cc_r = _mm256_add_epi64(cc_r, _mm256_sad_epu8(popcnt(_mm256_and_si256(o1, n2)), zero)); \
            dd_r = _mm256_add_epi64(dd_r, _mm256_sad_epu8(popcnt(_mm256_and_si256(o2, n1)), zero)); \
        } while(0)

        BLOCK(0,   aa0, bb0, cc0, dd0);
        BLOCK(32,  aa1, bb1, cc1, dd1);
        BLOCK(64,  aa2, bb2, cc2, dd2);
        BLOCK(96,  aa3, bb3, cc3, dd3);
        #undef BLOCK
    }

    __m256i aa = _mm256_add_epi64(_mm256_add_epi64(aa0, aa1), _mm256_add_epi64(aa2, aa3));
    __m256i bb = _mm256_add_epi64(_mm256_add_epi64(bb0, bb1), _mm256_add_epi64(bb2, bb3));
    __m256i cc = _mm256_add_epi64(_mm256_add_epi64(cc0, cc1), _mm256_add_epi64(cc2, cc3));
    __m256i dd = _mm256_add_epi64(_mm256_add_epi64(dd0, dd1), _mm256_add_epi64(dd2, dd3));

    uint64_t aa_u = hsum_epi64(aa);
    uint64_t bb_u = hsum_epi64(bb);
    uint64_t cc_u = hsum_epi64(cc);
    uint64_t dd_u = hsum_epi64(dd);

    for (; i < mask_bytes; ++i) {
        unsigned int o1b = static_cast<unsigned int>(static_cast<uint8_t>(ones_a[i]));
        unsigned int o2b = static_cast<unsigned int>(static_cast<uint8_t>(ones_b[i]));
        unsigned int n1b = static_cast<unsigned int>(static_cast<uint8_t>(negs_a[i]));
        unsigned int n2b = static_cast<unsigned int>(static_cast<uint8_t>(negs_b[i]));
        aa_u += std::popcount(o1b & o2b);
        bb_u += std::popcount(n1b & n2b);
        cc_u += std::popcount(o1b & n2b);
        dd_u += std::popcount(o2b & n1b);
    }

    const float similarity = static_cast<float>(aa_u + bb_u + dim) - static_cast<float>(cc_u + dd_u);
    return 1.0f - similarity / (2.0f * static_cast<float>(dim));
#else
    return deglib::distances::EvpInnerProduct::compare_naive(pVect1v, pVect2v, qty_ptr);
#endif
}

// ---------------------------------------------------------------------------
// Custom batch implementations
// Signature:
//   void fn(const std::byte* query, const std::byte* const* db_arr, const uint32_t* dim, float* dists)
// ---------------------------------------------------------------------------

// evp_custom_batch4: Query vs 4 DB vectors
static void evp_custom_batch4(const std::byte* query, const std::byte* const* db_arr, const uint32_t* dim, float* dists) {
    const uint32_t d = *dim;
    const size_t mask_bytes = d / 8;

    const std::byte* q_ones = query;
    const std::byte* q_negs = query + mask_bytes;

    uint64_t aa[4] = {0}, bb[4] = {0}, cc[4] = {0}, dd[4] = {0};

    for (size_t i = 0; i + 8 <= mask_bytes; i += 8) {
        const uint64_t oq = *reinterpret_cast<const uint64_t*>(&q_ones[i]);
        const uint64_t nq = *reinterpret_cast<const uint64_t*>(&q_negs[i]);

        for (int j = 0; j < 4; ++j) {
            const std::byte* db = db_arr[j];
            const uint64_t od = *reinterpret_cast<const uint64_t*>(&db[i]);
            const uint64_t nd = *reinterpret_cast<const uint64_t*>(&db[i + mask_bytes]);
            aa[j] += std::popcount(oq & od);
            bb[j] += std::popcount(nq & nd);
            cc[j] += std::popcount(oq & nd);
            dd[j] += std::popcount(od & nq);
        }
    }

    for (size_t i = mask_bytes & ~7ULL; i < mask_bytes; ++i) {
        const uint8_t oq = static_cast<uint8_t>(q_ones[i]);
        const uint8_t nq = static_cast<uint8_t>(q_negs[i]);
        for (int j = 0; j < 4; ++j) {
            const std::byte* db = db_arr[j];
            const uint8_t od = static_cast<uint8_t>(db[i]);
            const uint8_t nd = static_cast<uint8_t>(db[i + mask_bytes]);
            aa[j] += std::popcount(static_cast<unsigned int>(oq & od));
            bb[j] += std::popcount(static_cast<unsigned int>(nq & nd));
            cc[j] += std::popcount(static_cast<unsigned int>(oq & nd));
            dd[j] += std::popcount(static_cast<unsigned int>(od & nq));
        }
    }

    for (int j = 0; j < 4; ++j) {
        const float sim = static_cast<float>(aa[j] + bb[j] + d) - static_cast<float>(cc[j] + dd[j]);
        dists[j] = 1.0f - sim / (2.0f * static_cast<float>(d));
    }
}

// evp_custom_batch8: Query vs 8 DB vectors
static void evp_custom_batch8(const std::byte* query, const std::byte* const* db_arr, const uint32_t* dim, float* dists) {
    const uint32_t d = *dim;
    const size_t mask_bytes = d / 8;

    const std::byte* q_ones = query;
    const std::byte* q_negs = query + mask_bytes;

    uint64_t aa[8] = {0}, bb[8] = {0}, cc[8] = {0}, dd[8] = {0};

    for (size_t i = 0; i + 8 <= mask_bytes; i += 8) {
        const uint64_t oq = *reinterpret_cast<const uint64_t*>(&q_ones[i]);
        const uint64_t nq = *reinterpret_cast<const uint64_t*>(&q_negs[i]);

        for (int j = 0; j < 8; ++j) {
            const std::byte* db = db_arr[j];
            const uint64_t od = *reinterpret_cast<const uint64_t*>(&db[i]);
            const uint64_t nd = *reinterpret_cast<const uint64_t*>(&db[i + mask_bytes]);
            aa[j] += std::popcount(oq & od);
            bb[j] += std::popcount(nq & nd);
            cc[j] += std::popcount(oq & nd);
            dd[j] += std::popcount(od & nq);
        }
    }

    for (size_t i = mask_bytes & ~7ULL; i < mask_bytes; ++i) {
        const uint8_t oq = static_cast<uint8_t>(q_ones[i]);
        const uint8_t nq = static_cast<uint8_t>(q_negs[i]);
        for (int j = 0; j < 8; ++j) {
            const std::byte* db = db_arr[j];
            const uint8_t od = static_cast<uint8_t>(db[i]);
            const uint8_t nd = static_cast<uint8_t>(db[i + mask_bytes]);
            aa[j] += std::popcount(static_cast<unsigned int>(oq & od));
            bb[j] += std::popcount(static_cast<unsigned int>(nq & nd));
            cc[j] += std::popcount(static_cast<unsigned int>(oq & nd));
            dd[j] += std::popcount(static_cast<unsigned int>(od & nq));
        }
    }

    for (int j = 0; j < 8; ++j) {
        const float sim = static_cast<float>(aa[j] + bb[j] + d) - static_cast<float>(cc[j] + dd[j]);
        dists[j] = 1.0f - sim / (2.0f * static_cast<float>(d));
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

static void bench_evp(
    const std::byte* evp_data,
    size_t count,
    uint32_t dim,
    uint32_t non_zeros,
    uint32_t threads)
{
    (void)threads;
    (void)non_zeros;
    const size_t bytes_per_evp = static_cast<size_t>(dim) / 4;

    if (count == 0 || dim == 0) {
        std::printf("EVP bench: no data (count=%zu, dim=%u)\n", count, dim);
        return;
    }

    std::vector<uint32_t> shuffled_indices(count);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0u);
    std::mt19937 g(1337);
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), g);

    size_t comparisons = 100'000;
    comparisons = std::min(comparisons, std::max<size_t>(count * 256, 10'000));

    std::printf("=== EVP Bits Similarity Microbenchmark ===\n");
    std::printf("Vectors: %zu, dim=%u, NON_ZEROS=%u\n", count, dim, non_zeros);
    std::printf("bytes/evp=%zu (ones+negs)\n", bytes_per_evp);
    std::printf("Comparisons: %zu\n", comparisons);

    struct BenchResult { const char* name; double ms; double ns_per_op; };
    BenchResult best{ "<none>", 0.0, std::numeric_limits<double>::infinity() };

    auto run_case = [&](const char* name, auto&& fn) {
        constexpr int trials = 5;
        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>(comparisons / 10, 5'000);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const uint32_t c = static_cast<uint32_t>(((i * 40503ULL) + 17ULL) % count);
            const uint32_t db_idx = shuffled_indices[c];
            const std::byte* a = evp_data + static_cast<size_t>(q) * bytes_per_evp;
            const std::byte* b = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
            sink += fn(a, b, &dim);
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const uint32_t c = static_cast<uint32_t>(((i * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                const std::byte* a = evp_data + static_cast<size_t>(q) * bytes_per_evp;
                const std::byte* b = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
                sink += fn(a, b, &dim);
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
            const std::byte* a = evp_data + static_cast<size_t>(q) * bytes_per_evp;
            const std::byte* b_arr[8];
            for (int j = 0; j < batch_size; ++j) {
                const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                b_arr[j] = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
            }
            alignas(32) float dists[8];
            fn_batch(a, b_arr, &dim, dists);
            for (int j = 0; j < batch_size; ++j)
                sink += dists[j];
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons / batch_size; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const std::byte* a = evp_data + static_cast<size_t>(q) * bytes_per_evp;
                const std::byte* b_arr[8];
                for (int j = 0; j < batch_size; ++j) {
                    const uint32_t c = static_cast<uint32_t>((( (i * batch_size + j) * 40503ULL) + 17ULL) % count);
                    const uint32_t db_idx = shuffled_indices[c];
                    b_arr[j] = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
                }
                alignas(32) float dists[8];
                fn_batch(a, b_arr, &dim, dists);
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
    run_case("naive", deglib::distances::EvpInnerProduct::compare_naive);
#if defined(USE_AVX) || defined(USE_AVX512)
    run_case("avx2", deglib::distances::EvpInnerProduct::compare_avx2);
#endif
#if defined(USE_AVX512) && defined(__AVX512VPOPCNTDQ__)
    run_case("avx512", deglib::distances::EvpInnerProduct::compare_avx512);
#endif
    run_case("compare", deglib::distances::EvpInnerProduct::compare);

    // ------------------------------------------------------------------
    // Custom unrolled (single-query)
    // ------------------------------------------------------------------
#if defined(USE_AVX) || defined(USE_AVX512)
    std::printf("\n--- Custom unrolled (single-query) ---\n");
    run_case("evp_custom_u2", evp_custom_unroll2);
    run_case("evp_custom_u4", evp_custom_unroll4);
#endif

    // ------------------------------------------------------------------
    // Custom batch variants
    // ------------------------------------------------------------------
    std::printf("\n--- Custom batch variants ---\n");
    run_case_batch("evp_custom_b4", evp_custom_batch4, 4);
    run_case_batch("evp_custom_b8", evp_custom_batch8, 8);

    // ------------------------------------------------------------------
    // Deglib compare_batch
    // ------------------------------------------------------------------
    {
        auto deglib_batch4 = [](const std::byte* query, const std::byte* const* db_arr, const uint32_t* dim, float* dists) {
            deglib::distances::EvpInnerProduct::compare_batch(
                static_cast<const void*>(query),
                reinterpret_cast<const void* const*>(db_arr),
                4,
                static_cast<const void*>(dim),
                dists);
        };
        auto deglib_batch8 = [](const std::byte* query, const std::byte* const* db_arr, const uint32_t* dim, float* dists) {
            deglib::distances::EvpInnerProduct::compare_batch(
                static_cast<const void*>(query),
                reinterpret_cast<const void* const*>(db_arr),
                8,
                static_cast<const void*>(dim),
                dists);
        };
        std::printf("\n--- Deglib compare_batch ---\n");
        run_case_batch("deglib_batch4", deglib_batch4, 4);
        run_case_batch("deglib_batch8", deglib_batch8, 8);
    }

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

    constexpr uint32_t NON_ZEROS = 512;
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

    std::printf("Quantizing to EVP bits (NON_ZEROS=%u)...\n", NON_ZEROS);
    fflush(stdout);
    auto quantized = deglib::quantization::quantize_batch(
        train_vectors, static_cast<uint32_t>(dims), NON_ZEROS, threads);
    double load_ms = evp_common::now_ms() - t_load;

    std::printf("Loaded %S: %zu vectors, dim=%zu\n", train_hvecs.c_str(), count, dims);
    std::printf("Load + quantize time: %.2f ms (produced %.2f MB)\n\n",
                load_ms,
                static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

    train_vectors.clear();
    train_vectors.shrink_to_fit();

    bench_evp(
        quantized.data(),
        count,
        static_cast<uint32_t>(dims),
        NON_ZEROS,
        threads);

    return 0;
}
