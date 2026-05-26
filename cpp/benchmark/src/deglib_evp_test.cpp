/**
 * @file deglib_evp_test.cpp
 * @brief Graph Benchmark: build a DEG graph and evaluate exploration recall@15.
 *
 * Uses only deglib headers (no benchmark.h / dataset.h / build.h).
 * Loads raw .fvecs and .ivecs, builds a SizeBoundedGraph, explores for Top-15,
 * and compares against allknn ground truth.
 *
 * Modes:
 *   evp   - quantize with deglib::quantization, build graph with Metric::EvpBits
 *   raw   - use original float feature vectors directly with Metric::InnerProduct
 *   fp16  - convert FP32 to FP16, build graph with Metric::FP16InnerProduct
 *
 * Usage:
 *   deglib_evp_test <data_path> [evp|evp-no-rerank|evp-build-fp16-search|evp-build-fp16-proxy|raw|fp16|evp-linear]
 *
 * Expected files in data_path:
 *   train.fvecs      - float feature vectors
 *   allknn.ivecs     - ground truth (1-indexed, dims=32)
 */

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

#include <fmt/core.h>

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#include <immintrin.h>
#endif

#include "builder.h"
#include "concurrent.h"
#include "distances.h"
#include "graph/sizebounded_graph.h"
#include "graph/readonly_graph.h"
#include "graph/readonly_graph_external.h"
#include "quantization/evp_quantize.h"
#include "repository.h"

// ============================================================================
// Helpers
// ============================================================================

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
           .count();
}

// ============================================================================
// Consolidated Modes
// ============================================================================

enum class BenchmarkMode {
    EvpBuildEvpExploreFP16Rerank,
    EvpBuildEvpExplore,
    EvpBuildFP16ExternalSearch,
    FP16BuildFP16Explore,
    EvpLinearSearch,
    EvpBuildFP16AsymmetricSearch,
    EvpBuildFP16AsymmetricSearchRerank,
    FP16EvpAsymmetricMicrobench
};

// ============================================================================
// Microbench: FP16-EVP Asymmetric Similarity
// ============================================================================

static void bench_fp16_evp_asymmetric(
    const uint16_t* fp16_data,
    size_t count,
    uint32_t dim,
    const std::byte* evp_data,
    uint32_t non_zeros,
    uint32_t threads)
{
    (void)threads;
    const uint32_t dim_u32 = dim;
    const size_t bytes_per_evp = static_cast<size_t>(dim) / 4; // dim/8 ones + dim/8 negs

    // Keep this compute-heavy but not unbounded.
    size_t comparisons = 1'000'000;
    if (count == 0 || dim == 0) {
        std::printf("Asymmetric bench: no data (count=%zu, dim=%u)\n", count, dim);
        return;
    }
    // Scale down for tiny datasets.
    comparisons = std::min(comparisons, std::max<size_t>(count * 256, 10'000));

    std::printf("=== FP16-EVP Asymmetric Similarity Microbenchmark ===\n");
    std::printf("Vectors: %zu, dim=%u, NON_ZEROS=%u\n", count, dim, non_zeros);
    std::printf("bytes/query=%zu (FP16), bytes/evp=%zu (ones+negs)\n",
                static_cast<size_t>(dim) * sizeof(uint16_t), bytes_per_evp);
    std::printf("Comparisons: %zu\n", comparisons);

    struct BenchResult {
        const char* name;
        double ms;
        double ns_per_op;
    };

    BenchResult best{ "<none>", 0.0, std::numeric_limits<double>::infinity() };

    auto run_case = [&](const char* name, auto&& fn) {
        constexpr int trials = 5;

        // Warmup
        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>(comparisons / 10, 50'000);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const uint32_t c = static_cast<uint32_t>(((i * 40503ULL) + 17ULL) % count);
            const uint16_t* a = fp16_data + static_cast<size_t>(q) * dim;
            const std::byte* b = evp_data + static_cast<size_t>(c) * bytes_per_evp;
            sink += fn(a, b, &dim_u32);
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const uint32_t c = static_cast<uint32_t>(((i * 40503ULL) + 17ULL) % count);
                const uint16_t* a = fp16_data + static_cast<size_t>(q) * dim;
                const std::byte* b = evp_data + static_cast<size_t>(c) * bytes_per_evp;
                sink += fn(a, b, &dim_u32);
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

    // ----------------------------------------------------------------------
    // Candidate implementations (benchmark-only)
    // ----------------------------------------------------------------------

#if defined(USE_SSE)
    struct SseNibbleTables {
        alignas(16) float ones[16][4];
        alignas(16) float negs[16][4];
        SseNibbleTables() {
            for (int m = 0; m < 16; ++m) {
                for (int i = 0; i < 4; ++i) {
                    ones[m][i] = ((m >> i) & 1) ? 1.0f : 0.0f;
                    negs[m][i] = ((m >> i) & 1) ? 1.0f : 0.0f;
                }
            }
        }
    };
    static const SseNibbleTables sse_tbl;

    auto ip_sse_lut = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m128 sum128 = _mm_setzero_ps();
        for (size_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            const uint8_t ones_byte = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t negs_byte = static_cast<uint8_t>(negs_b[byte_idx]);
            if (ones_byte == 0 && negs_byte == 0) {
                continue;
            }

            __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            __m128 a_lo = _mm_cvtph_ps(raw_a);
            __m128 a_hi = _mm_cvtph_ps(_mm_srli_si128(raw_a, 8));

            const uint8_t o0 = ones_byte & 0x0F;
            const uint8_t n0 = negs_byte & 0x0F;
            if ((o0 | n0) != 0) {
                const __m128 m_ones = _mm_load_ps(sse_tbl.ones[o0]);
                const __m128 m_negs = _mm_load_ps(sse_tbl.negs[n0]);
                const __m128 mask = _mm_sub_ps(m_ones, m_negs);
                sum128 = _mm_add_ps(sum128, _mm_mul_ps(a_lo, mask));
            }

            const uint8_t o1 = (ones_byte >> 4) & 0x0F;
            const uint8_t n1 = (negs_byte >> 4) & 0x0F;
            if ((o1 | n1) != 0) {
                const __m128 m_ones = _mm_load_ps(sse_tbl.ones[o1]);
                const __m128 m_negs = _mm_load_ps(sse_tbl.negs[n1]);
                const __m128 mask = _mm_sub_ps(m_ones, m_negs);
                sum128 = _mm_add_ps(sum128, _mm_mul_ps(a_hi, mask));
            }
        }

        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };
#endif

#if defined(USE_AVX)
    struct Avx2ByteTables {
        alignas(32) float ones[256][8];
        alignas(32) float negs[256][8];
        Avx2ByteTables() {
            for (int m = 0; m < 256; ++m) {
                for (int i = 0; i < 8; ++i) {
                    ones[m][i] = ((m >> i) & 1) ? 1.0f : 0.0f;
                    negs[m][i] = ((m >> i) & 1) ? 1.0f : 0.0f;
                }
            }
        }
    };
    static const Avx2ByteTables avx2_tbl;

    // Combined table: mask = ones - negs for each (ones_byte, negs_byte).
    // Size: 256*256*8 floats = 2MB. Benchmark-only candidate.
    struct Avx2DiffTables {
        alignas(32) float diff[256][256][8];
        Avx2DiffTables() {
            for (int o = 0; o < 256; ++o) {
                for (int n = 0; n < 256; ++n) {
                    for (int i = 0; i < 8; ++i) {
                        const int oi = (o >> i) & 1;
                        const int ni = (n >> i) & 1;
                        diff[o][n][i] = static_cast<float>(oi - ni);
                    }
                }
            }
        }
    };
    static const Avx2DiffTables avx2_diff_tbl;

    auto ip_avx2_lut = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256 sum256 = _mm256_setzero_ps();
        for (size_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            const uint8_t ones_byte = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t negs_byte = static_cast<uint8_t>(negs_b[byte_idx]);
            if (ones_byte == 0 && negs_byte == 0) {
                continue;
            }

            __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw_a);

            const __m256 m_ones = _mm256_load_ps(avx2_tbl.ones[ones_byte]);
            const __m256 m_negs = _mm256_load_ps(avx2_tbl.negs[negs_byte]);
            const __m256 mask = _mm256_sub_ps(m_ones, m_negs);

            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(a_ps, mask));
        }

        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };

    // Unroll by 2 bytes (16 FP16 per iteration). This reduces per-iteration overhead.
    auto ip_avx2_lut2 = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256 sum256 = _mm256_setzero_ps();
        __m256 sum256_2 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 1 < mask_bytes; byte_idx += 2) {
            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);

            if ((o0 | n0 | o1 | n1) == 0) {
                continue;
            }

            // Load 16 FP16 values corresponding to two bytes.
            const __m256i raw16 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m128i raw0 = _mm256_extractf128_si256(raw16, 0);
            const __m128i raw1 = _mm256_extractf128_si256(raw16, 1);

            const __m256 a0 = _mm256_cvtph_ps(raw0);
            const __m256 a1 = _mm256_cvtph_ps(raw1);

            if ((o0 | n0) != 0) {
                const __m256 m_ones = _mm256_load_ps(avx2_tbl.ones[o0]);
                const __m256 m_negs = _mm256_load_ps(avx2_tbl.negs[n0]);
                const __m256 mask = _mm256_sub_ps(m_ones, m_negs);
                sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(a0, mask));
            }
            if ((o1 | n1) != 0) {
                const __m256 m_ones = _mm256_load_ps(avx2_tbl.ones[o1]);
                const __m256 m_negs = _mm256_load_ps(avx2_tbl.negs[n1]);
                const __m256 mask = _mm256_sub_ps(m_ones, m_negs);
                sum256_2 = _mm256_add_ps(sum256_2, _mm256_mul_ps(a1, mask));
            }
        }

        // Tail (odd mask_bytes)
        if (byte_idx < mask_bytes) {
            const uint8_t ones_byte = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t negs_byte = static_cast<uint8_t>(negs_b[byte_idx]);
            if ((ones_byte | negs_byte) != 0) {
                __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
                __m256 a_ps = _mm256_cvtph_ps(raw_a);
                const __m256 m_ones = _mm256_load_ps(avx2_tbl.ones[ones_byte]);
                const __m256 m_negs = _mm256_load_ps(avx2_tbl.negs[negs_byte]);
                const __m256 mask = _mm256_sub_ps(m_ones, m_negs);
                sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(a_ps, mask));
            }
        }

        sum256 = _mm256_add_ps(sum256, sum256_2);
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };

    // Same as ip_avx2_lut2 but branchless: always loads masks and multiplies.
    // Useful when most bytes are non-zero (e.g., NON_ZEROS around dim/2).
    auto ip_avx2_lut2_nb = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 1 < mask_bytes; byte_idx += 2) {
            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);

            const __m256i raw16 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m128i raw_a0 = _mm256_extractf128_si256(raw16, 0);
            const __m128i raw_a1 = _mm256_extractf128_si256(raw16, 1);

            const __m256 a0 = _mm256_cvtph_ps(raw_a0);
            const __m256 a1 = _mm256_cvtph_ps(raw_a1);

            const __m256 mask0 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0]));
            const __m256 mask1 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1]));

            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a0, mask0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(a1, mask1));
        }

        if (byte_idx < mask_bytes) {
            const uint8_t o = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n = static_cast<uint8_t>(negs_b[byte_idx]);

            __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw_a);
            const __m256 mask = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n]));
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a_ps, mask));
        }

        const __m256 sum256 = _mm256_add_ps(sum0, sum1);
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };

    // Unroll by 4 bytes (32 FP16 per iteration).
    auto ip_avx2_lut4 = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 3 < mask_bytes; byte_idx += 4) {
            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);
            const uint8_t o2 = static_cast<uint8_t>(ones_b[byte_idx + 2]);
            const uint8_t n2 = static_cast<uint8_t>(negs_b[byte_idx + 2]);
            const uint8_t o3 = static_cast<uint8_t>(ones_b[byte_idx + 3]);
            const uint8_t n3 = static_cast<uint8_t>(negs_b[byte_idx + 3]);

            if ((o0 | n0 | o1 | n1 | o2 | n2 | o3 | n3) == 0) {
                continue;
            }

            const __m256i raw16_0 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m256i raw16_1 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 2) * 8]);

            const __m128i r0 = _mm256_extractf128_si256(raw16_0, 0);
            const __m128i r1 = _mm256_extractf128_si256(raw16_0, 1);
            const __m128i r2 = _mm256_extractf128_si256(raw16_1, 0);
            const __m128i r3 = _mm256_extractf128_si256(raw16_1, 1);

            const __m256 a0 = _mm256_cvtph_ps(r0);
            const __m256 a1 = _mm256_cvtph_ps(r1);
            const __m256 a2 = _mm256_cvtph_ps(r2);
            const __m256 a3 = _mm256_cvtph_ps(r3);

            if ((o0 | n0) != 0) {
                const __m256 mask = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0]));
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(a0, mask));
            }
            if ((o1 | n1) != 0) {
                const __m256 mask = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1]));
                s1 = _mm256_add_ps(s1, _mm256_mul_ps(a1, mask));
            }
            if ((o2 | n2) != 0) {
                const __m256 mask = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o2]), _mm256_load_ps(avx2_tbl.negs[n2]));
                s2 = _mm256_add_ps(s2, _mm256_mul_ps(a2, mask));
            }
            if ((o3 | n3) != 0) {
                const __m256 mask = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o3]), _mm256_load_ps(avx2_tbl.negs[n3]));
                s3 = _mm256_add_ps(s3, _mm256_mul_ps(a3, mask));
            }
        }

        // Remainder bytes
        for (; byte_idx < mask_bytes; ++byte_idx) {
            const uint8_t ones_byte = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t negs_byte = static_cast<uint8_t>(negs_b[byte_idx]);
            if ((ones_byte | negs_byte) == 0) {
                continue;
            }
            __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw_a);
            const __m256 mask = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[ones_byte]), _mm256_load_ps(avx2_tbl.negs[negs_byte]));
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(a_ps, mask));
        }

        __m256 sum256 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };

    // Branchless 4-byte unroll (always processes masks).
    auto ip_avx2_lut4_nb = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps();
        __m256 s3 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 3 < mask_bytes; byte_idx += 4) {
            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);
            const uint8_t o2 = static_cast<uint8_t>(ones_b[byte_idx + 2]);
            const uint8_t n2 = static_cast<uint8_t>(negs_b[byte_idx + 2]);
            const uint8_t o3 = static_cast<uint8_t>(ones_b[byte_idx + 3]);
            const uint8_t n3 = static_cast<uint8_t>(negs_b[byte_idx + 3]);

            const __m256i raw16_0 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m256i raw16_1 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 2) * 8]);

            const __m128i r0 = _mm256_extractf128_si256(raw16_0, 0);
            const __m128i r1 = _mm256_extractf128_si256(raw16_0, 1);
            const __m128i r2 = _mm256_extractf128_si256(raw16_1, 0);
            const __m128i r3 = _mm256_extractf128_si256(raw16_1, 1);

            const __m256 a0 = _mm256_cvtph_ps(r0);
            const __m256 a1 = _mm256_cvtph_ps(r1);
            const __m256 a2 = _mm256_cvtph_ps(r2);
            const __m256 a3 = _mm256_cvtph_ps(r3);

            const __m256 m0 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0]));
            const __m256 m1 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1]));
            const __m256 m2 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o2]), _mm256_load_ps(avx2_tbl.negs[n2]));
            const __m256 m3 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o3]), _mm256_load_ps(avx2_tbl.negs[n3]));

            s0 = _mm256_add_ps(s0, _mm256_mul_ps(a0, m0));
            s1 = _mm256_add_ps(s1, _mm256_mul_ps(a1, m1));
            s2 = _mm256_add_ps(s2, _mm256_mul_ps(a2, m2));
            s3 = _mm256_add_ps(s3, _mm256_mul_ps(a3, m3));
        }

        for (; byte_idx < mask_bytes; ++byte_idx) {
            const uint8_t o = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n = static_cast<uint8_t>(negs_b[byte_idx]);
            __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw_a);
            const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n]));
            s0 = _mm256_add_ps(s0, _mm256_mul_ps(a_ps, m));
        }

        const __m256 sum256 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };

    // Branchless 8-byte unroll (64 FP16 per iteration).
    auto ip_avx2_lut8_nb = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256 sumA = _mm256_setzero_ps();
        __m256 sumB = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 7 < mask_bytes; byte_idx += 8) {
            // 8 bytes -> 4 loads of 16 FP16 each
            const __m256i raw16_0 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 0) * 8]);
            const __m256i raw16_1 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 2) * 8]);
            const __m256i raw16_2 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 4) * 8]);
            const __m256i raw16_3 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 6) * 8]);

            const __m128i r0 = _mm256_extractf128_si256(raw16_0, 0);
            const __m128i r1 = _mm256_extractf128_si256(raw16_0, 1);
            const __m128i r2 = _mm256_extractf128_si256(raw16_1, 0);
            const __m128i r3 = _mm256_extractf128_si256(raw16_1, 1);
            const __m128i r4 = _mm256_extractf128_si256(raw16_2, 0);
            const __m128i r5 = _mm256_extractf128_si256(raw16_2, 1);
            const __m128i r6 = _mm256_extractf128_si256(raw16_3, 0);
            const __m128i r7 = _mm256_extractf128_si256(raw16_3, 1);

            const __m256 a0 = _mm256_cvtph_ps(r0);
            const __m256 a1 = _mm256_cvtph_ps(r1);
            const __m256 a2 = _mm256_cvtph_ps(r2);
            const __m256 a3 = _mm256_cvtph_ps(r3);
            const __m256 a4 = _mm256_cvtph_ps(r4);
            const __m256 a5 = _mm256_cvtph_ps(r5);
            const __m256 a6 = _mm256_cvtph_ps(r6);
            const __m256 a7 = _mm256_cvtph_ps(r7);

            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx + 0]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx + 0]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);
            const uint8_t o2 = static_cast<uint8_t>(ones_b[byte_idx + 2]);
            const uint8_t n2 = static_cast<uint8_t>(negs_b[byte_idx + 2]);
            const uint8_t o3 = static_cast<uint8_t>(ones_b[byte_idx + 3]);
            const uint8_t n3 = static_cast<uint8_t>(negs_b[byte_idx + 3]);
            const uint8_t o4 = static_cast<uint8_t>(ones_b[byte_idx + 4]);
            const uint8_t n4 = static_cast<uint8_t>(negs_b[byte_idx + 4]);
            const uint8_t o5 = static_cast<uint8_t>(ones_b[byte_idx + 5]);
            const uint8_t n5 = static_cast<uint8_t>(negs_b[byte_idx + 5]);
            const uint8_t o6 = static_cast<uint8_t>(ones_b[byte_idx + 6]);
            const uint8_t n6 = static_cast<uint8_t>(negs_b[byte_idx + 6]);
            const uint8_t o7 = static_cast<uint8_t>(ones_b[byte_idx + 7]);
            const uint8_t n7 = static_cast<uint8_t>(negs_b[byte_idx + 7]);

            const __m256 m0 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0]));
            const __m256 m1 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1]));
            const __m256 m2 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o2]), _mm256_load_ps(avx2_tbl.negs[n2]));
            const __m256 m3 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o3]), _mm256_load_ps(avx2_tbl.negs[n3]));
            const __m256 m4 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o4]), _mm256_load_ps(avx2_tbl.negs[n4]));
            const __m256 m5 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o5]), _mm256_load_ps(avx2_tbl.negs[n5]));
            const __m256 m6 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o6]), _mm256_load_ps(avx2_tbl.negs[n6]));
            const __m256 m7 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o7]), _mm256_load_ps(avx2_tbl.negs[n7]));

            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a0, m0));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a1, m1));
            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a2, m2));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a3, m3));
            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a4, m4));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a5, m5));
            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a6, m6));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a7, m7));
        }

        // Fall back to 4-byte chunks
        for (; byte_idx + 3 < mask_bytes; byte_idx += 4) {
            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);
            const uint8_t o2 = static_cast<uint8_t>(ones_b[byte_idx + 2]);
            const uint8_t n2 = static_cast<uint8_t>(negs_b[byte_idx + 2]);
            const uint8_t o3 = static_cast<uint8_t>(ones_b[byte_idx + 3]);
            const uint8_t n3 = static_cast<uint8_t>(negs_b[byte_idx + 3]);

            const __m256i raw16_0 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m256i raw16_1 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 2) * 8]);

            const __m128i r0 = _mm256_extractf128_si256(raw16_0, 0);
            const __m128i r1 = _mm256_extractf128_si256(raw16_0, 1);
            const __m128i r2 = _mm256_extractf128_si256(raw16_1, 0);
            const __m128i r3 = _mm256_extractf128_si256(raw16_1, 1);

            const __m256 a0 = _mm256_cvtph_ps(r0);
            const __m256 a1 = _mm256_cvtph_ps(r1);
            const __m256 a2 = _mm256_cvtph_ps(r2);
            const __m256 a3 = _mm256_cvtph_ps(r3);

            const __m256 m0 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0]));
            const __m256 m1 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1]));
            const __m256 m2 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o2]), _mm256_load_ps(avx2_tbl.negs[n2]));
            const __m256 m3 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o3]), _mm256_load_ps(avx2_tbl.negs[n3]));

            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a0, m0));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a1, m1));
            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a2, m2));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a3, m3));
        }

        // Tail bytes
        for (; byte_idx < mask_bytes; ++byte_idx) {
            const uint8_t o = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n = static_cast<uint8_t>(negs_b[byte_idx]);
            __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw_a);
            const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n]));
            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a_ps, m));
        }

        const __m256 sum256 = _mm256_add_ps(sumA, sumB);
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };

    // Unroll by 2 bytes, but use a single combined diff LUT for the mask.
    auto ip_avx2_diff2 = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim_local = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim_local / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 1 < mask_bytes; byte_idx += 2) {
            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);

            if ((o0 | n0 | o1 | n1) == 0) {
                continue;
            }

            const __m256i raw16 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m128i raw0 = _mm256_extractf128_si256(raw16, 0);
            const __m128i raw1 = _mm256_extractf128_si256(raw16, 1);

            const __m256 a0 = _mm256_cvtph_ps(raw0);
            const __m256 a1 = _mm256_cvtph_ps(raw1);

            if ((o0 | n0) != 0) {
                const __m256 mask = _mm256_load_ps(avx2_diff_tbl.diff[o0][n0]);
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(a0, mask));
            }
            if ((o1 | n1) != 0) {
                const __m256 mask = _mm256_load_ps(avx2_diff_tbl.diff[o1][n1]);
                s1 = _mm256_add_ps(s1, _mm256_mul_ps(a1, mask));
            }
        }

        if (byte_idx < mask_bytes) {
            const uint8_t o = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n = static_cast<uint8_t>(negs_b[byte_idx]);
            if ((o | n) != 0) {
                __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
                __m256 a_ps = _mm256_cvtph_ps(raw_a);
                const __m256 mask = _mm256_load_ps(avx2_diff_tbl.diff[o][n]);
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(a_ps, mask));
            }
        }

        const __m256 sum256 = _mm256_add_ps(s0, s1);
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    };
#endif

    std::printf("\nAvailable implementations in this build:\n");
    std::printf("  - ip_naive (always)\n");
#if defined(USE_SSE)
    std::printf("  - ip_sse\n");
    std::printf("  - ip_sse_lut (candidate)\n");
#endif
#if defined(USE_AVX)
    std::printf("  - ip_avx2\n");
    std::printf("  - ip_avx2_lut (candidate)\n");
    std::printf("  - ip_avx2_lut2 (candidate)\n");
    std::printf("  - ip_avx2_lut2_nb (candidate)\n");
    std::printf("  - ip_avx2_lut4 (candidate)\n");
    std::printf("  - ip_avx2_lut4_nb (candidate)\n");
    std::printf("  - ip_avx2_lut8_nb (candidate)\n");
    std::printf("  - ip_avx2_diff2 (candidate)\n");
#endif
#if defined(USE_AVX512)
    std::printf("  - ip_avx512\n");
#endif
    std::printf("  - compare() dispatch\n\n");

    run_case("ip_naive", deglib::distances::FP16EvpAsymmetricSimilarity::ip_naive);
#if defined(USE_SSE)
    run_case("ip_sse", deglib::distances::FP16EvpAsymmetricSimilarity::ip_sse);
    run_case("ip_sse_lut", ip_sse_lut);
#endif
#if defined(USE_AVX)
    run_case("ip_avx2", deglib::distances::FP16EvpAsymmetricSimilarity::ip_avx2);
    run_case("ip_avx2_lut", ip_avx2_lut);
    run_case("ip_avx2_lut2", ip_avx2_lut2);
    run_case("ip_avx2_lut2_nb", ip_avx2_lut2_nb);
    run_case("ip_avx2_lut4", ip_avx2_lut4);
    run_case("ip_avx2_lut4_nb", ip_avx2_lut4_nb);
    run_case("ip_avx2_lut8_nb", ip_avx2_lut8_nb);
    run_case("ip_avx2_diff2", ip_avx2_diff2);
#endif
#if defined(USE_AVX512)
    run_case("ip_avx512", deglib::distances::FP16EvpAsymmetricSimilarity::ip_avx512);
#endif
    run_case("compare", deglib::distances::FP16EvpAsymmetricSimilarity::compare);

    std::printf("\nBEST: %s (%.2f ns/op)\n", best.name, best.ns_per_op);

    std::printf("\n=============================================\n");
}

// ============================================================================
// Shared helpers
// ============================================================================

static void run_exploration_sweep(
    const deglib::search::SearchGraph& graph,
    const int32_t* gt_data,
    size_t count,
    size_t gt_dims,
    uint32_t k_top,
    uint8_t threads,
    const char* label,
    bool use_search = false,
    const void* rerank_data = nullptr,
    size_t rerank_feature_bytes = 0,
    deglib::DISTFUNC<float> rerank_dist_func = nullptr,
    size_t dims = 0,
    bool use_asymmetric_search = false,
    const void* query_data = nullptr)
{
    std::printf("\n--- Exploration sweep (%s) ---\n", label);

    const uint32_t k_explore = k_top;

    // Fixed max_distance_count values
    const uint32_t max_distances[] = { k_explore, 50, 100, 200, 300, 400 };
    float best_recall = -1.0f;
    uint32_t best_max_dist = 0;
    double best_explore_ms = 0.0;
    bool stop_sweep = false;

    for (uint32_t md_idx = 0; md_idx < sizeof(max_distances) / sizeof(max_distances[0]) && !stop_sweep; md_idx++) {
        const uint32_t max_distance_count = max_distances[md_idx];

        // --- Phase 1: Graph Exploration ---
        double t_explore_start = now_ms();
        std::vector<std::vector<uint32_t>> explore_candidates(count);

        deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
            [&](size_t label, size_t thread_id) {
                (void)thread_id;
                size_t entry_idx = graph.getInternalIndex(static_cast<uint32_t>(label));
                uint32_t k_search = (rerank_data != nullptr) ? (use_asymmetric_search ? 25 : std::max(k_top, max_distance_count)) : k_explore;

                deglib::search::ResultSet result_queue;
                if (use_asymmetric_search) {
                    std::vector<uint32_t> entry_indices = { static_cast<uint32_t>(entry_idx) };
                    const std::byte* query = static_cast<const std::byte*>(query_data) + label * dims * sizeof(uint16_t);
                    result_queue = graph.search(
                        entry_indices,
                        query,
                        1000.0f,
                        k_search + 1,
                        nullptr,
                        max_distance_count + 1);
                } else if (use_search) {
                    std::vector<uint32_t> entry_indices = { static_cast<uint32_t>(entry_idx) };
                    const std::byte* query = graph.getFeatureVector(static_cast<uint32_t>(entry_idx));
                    result_queue = graph.search(
                        entry_indices,
                        query,
                        1000.0f,
                        k_search + 1,
                        nullptr,
                        max_distance_count + 1);
                } else {
                    result_queue = graph.explore(
                        static_cast<uint32_t>(entry_idx),
                        k_search,
                        false,  // include_entry (false: exclude self)
                        max_distance_count);
                }

                auto& cands = explore_candidates[label];
                cands.reserve(result_queue.size());
                while (!result_queue.empty()) {
                    const auto cand_idx = result_queue.top().getInternalIndex();
                    if (cand_idx != entry_idx) {
                        cands.push_back(graph.getExternalLabel(cand_idx));
                    }
                    result_queue.pop();
                }
                if ((use_search || use_asymmetric_search) && cands.size() > k_search) {
                    cands.resize(k_search);
                }
            });
        double explore_ms = now_ms() - t_explore_start;

        // --- Phase 2: Reranking (if rerank_data is provided) ---
        double rerank_ms = 0.0;
        std::vector<std::vector<uint32_t>> results(count);

        if (rerank_data != nullptr) {
            double t_rerank_start = now_ms();
            const auto* base = static_cast<const std::byte*>(rerank_data);
            deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
                [&](size_t label, size_t thread_id) {
                    (void)thread_id;
                    struct Candidate {
                        uint32_t label;
                        float distance;
                    };
                    const auto& cands = explore_candidates[label];
                    std::vector<Candidate> candidates;
                    candidates.reserve(cands.size());

                    size_t dims_size = dims;
                    const std::byte* query_ptr = base + label * rerank_feature_bytes;

                    for (uint32_t cand_label : cands) {
                        const std::byte* cand_ptr = base + cand_label * rerank_feature_bytes;
                        float exact_dist = rerank_dist_func(query_ptr, cand_ptr, &dims_size);
                        candidates.push_back({cand_label, exact_dist});
                    }

                    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                        return a.distance < b.distance;
                    });

                    auto& result = results[label];
                    result.resize(k_top);
                    std::fill(result.begin(), result.end(), std::numeric_limits<uint32_t>::max());
                    for (uint32_t j = 0; j < k_top && j < candidates.size(); ++j) {
                        result[j] = candidates[j].label;
                    }
                });
            rerank_ms = now_ms() - t_rerank_start;
        } else {
            results = std::move(explore_candidates);
        }

        double explore_ms_single = explore_ms + rerank_ms;

        double wall_explore = explore_ms;
        double wall_rerank = rerank_ms;

        // Calculate recall
        int total_hits = 0;
        for (size_t i = 0; i < count; ++i) {
            const int32_t* gt_row = &gt_data[i * gt_dims];
            for (uint32_t k = 1; k <= k_top && k < gt_dims; ++k) {
                uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1→0 indexed
                const auto& row = results[i];
                const uint32_t row_len = static_cast<uint32_t>(std::min(row.size(), static_cast<size_t>(k_top)));
                for (uint32_t j = 0; j < row_len; ++j) {
                    if (row[j] == gt_idx) {
                        total_hits++;
                        break;
                    }
                }
            }
        }

        float recall = static_cast<float>(total_hits) / (count * k_top);

        if (recall > best_recall) {
            best_recall = recall;
            best_max_dist = max_distance_count;
            best_explore_ms = explore_ms_single;
        }

        if (rerank_data != nullptr) {
            std::printf("  max_dist=%6u  recall=%.4f  time=%.2f ms (explore=%.2f ms, rerank=%.2f ms)\n",
                        max_distance_count, recall, explore_ms_single, wall_explore, wall_rerank);
        } else {
            std::printf("  max_dist=%6u  recall=%.4f  time=%.2f ms\n",
                        max_distance_count, recall, explore_ms_single);
        }

        if (recall >= 0.8f) {
            std::printf("  Reached recall target 0.8, stopping sweep\n");
            stop_sweep = true;
        }
    }

    std::printf("\nBest recall@%d: %.4f (max_distance_count=%u, explore_time=%.2f ms)\n",
                k_top, best_recall, best_max_dist, best_explore_ms);
}



/**
 * Reads an hvecs file containing FP16 feature vectors.
 * Follows the standard vecs format (4-byte uint32_t dimension header followed by vector data).
 */
static auto hvecs_read(const char* fname, size_t& d_out, size_t& n_out) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing file %s, size is: %ju message: %s \n", fname, file_size, ec.message().c_str());
        perror("");
        abort();
    }

    // open as binary
    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open()) {
        std::fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    // read dimension header
    uint32_t dims = 0;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    assert((dims > 0 && dims < 1'000'000) && "unreasonable dimension");

    // compute number of rows
    size_t row_bytes = sizeof(uint32_t) + dims * sizeof(uint16_t);
    assert(file_size % row_bytes == 0 || !"weird file size");
    size_t n = (size_t)file_size / row_bytes;
    d_out = dims;
    n_out = n;

    // read data rows
    auto x = std::make_unique<std::byte[]>(file_size);
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), file_size);
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(file_size) || !"could not read whole file");

    // shift array to remove row headers (the headers are 4 bytes at the start of each row)
    for (size_t i = 0; i < n; i++) {
        std::memmove(&x[i * dims * sizeof(uint16_t)], &x[sizeof(uint32_t) + i * row_bytes], dims * sizeof(uint16_t));
    }

    ifstream.close();
    return x;
}

// ============================================================================
// Unified Benchmark Entry Point
// ============================================================================

static int run_benchmark(
    BenchmarkMode mode,
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t non_zeros,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top)
{
    // --------------------------------------------------------------------------
    // Load data
    // --------------------------------------------------------------------------
    const auto train_hvecs = data_path / "train.hvecs";
    const auto allknn_ivecs = data_path / "allknn.ivecs";

    const bool needs_ground_truth = (mode != BenchmarkMode::FP16EvpAsymmetricMicrobench);

    if (!std::filesystem::exists(train_hvecs)) {
        std::fprintf(stderr, "Error: %ls not found\n", train_hvecs.wstring().c_str());
        return 1;
    }
    if (needs_ground_truth) {
        if (!std::filesystem::exists(allknn_ivecs)) {
            std::fprintf(stderr, "Error: %ls not found\n", allknn_ivecs.wstring().c_str());
            return 1;
        }
    }

    double t_load = now_ms();

    size_t dims = 0, count = 0;
    auto train_bytes = hvecs_read(train_hvecs.string().c_str(), dims, count);
    const uint16_t* train_data_fp16 = reinterpret_cast<const uint16_t*>(train_bytes.get());

    size_t gt_dims = 0, gt_count = 0;
    std::unique_ptr<std::byte[]> gt_bytes;
    const int32_t* gt_data = nullptr;
    if (needs_ground_truth) {
        gt_bytes = deglib::ivecs_read(allknn_ivecs.string().c_str(), gt_dims, gt_count);
        gt_data = reinterpret_cast<const int32_t*>(gt_bytes.get());
    }

    double load_ms = now_ms() - t_load;
    std::printf("Loaded %ls: %zu vectors, dim=%zu\n", train_hvecs.filename().wstring().c_str(), count, dims);
    if (needs_ground_truth) {
        std::printf("Ground truth: %zu queries, top-%zu\n", gt_count, gt_dims);
    }
    std::printf("Load time: %.2f ms\n\n", load_ms);

    if (needs_ground_truth) {
        if (count != gt_count) {
            std::fprintf(stderr,
                "Error: train.hvecs and allknn.ivecs must contain the same number of entries (%zu vs %zu)\n",
                count, gt_count);
            return 1;
        }
    }

    if (mode == BenchmarkMode::FP16EvpAsymmetricMicrobench) {
        // Quantize once to EVP and then time the asymmetric dot-product paths.
        double t_q = now_ms();
        auto quantized = deglib::quantization::quantize_batch(
            train_data_fp16, count, static_cast<uint32_t>(dims), non_zeros, threads);
        double quantize_ms = now_ms() - t_q;

        std::printf("Quantize time: %.2f ms (produced %.2f MB)\n\n",
                    quantize_ms,
                    static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

        bench_fp16_evp_asymmetric(
            train_data_fp16,
            count,
            static_cast<uint32_t>(dims),
            quantized.data(),
            non_zeros,
            threads);
        return 0;
    }



    if (mode == BenchmarkMode::FP16BuildFP16Explore) {
        std::printf("=== FP16 Build FP16 Explore Graph Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f\n",
                    k_top, k_graph, k_ext, eps_ext);
        std::printf("Threads: %u\n\n", threads);

        std::printf("FP16 data size: %.2f MB\n\n",
                    static_cast<double>(count * dims * sizeof(uint16_t)) / (1024.0 * 1024.0));

        // --------------------------------------------------------------------------
        // Build graph with Metric::FP16InnerProduct
        // --------------------------------------------------------------------------
        double t2 = now_ms();

        deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
        deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(count), k_graph, feature_space);

        std::mt19937 rnd(42);
        deglib::builder::EvenRegularGraphBuilder builder(
            graph, rnd,
            deglib::builder::OptimizationTarget::LowLID,
            k_ext, eps_ext,
            0, 0.0f,
            5,
            0, 0,
            true,
            false,
            false
        );
        builder.setThreadCount(static_cast<uint32_t>(threads));

        // Add all entries from FP16 data directly
        const size_t bytes_per_fp16 = dims * sizeof(uint16_t);
        const std::byte* train_data_bytes = train_bytes.get();
        for (size_t i = 0; i < count; ++i) {
            std::vector<std::byte> feature(train_data_bytes + i * bytes_per_fp16, train_data_bytes + (i + 1) * bytes_per_fp16);
            builder.addEntry(static_cast<uint32_t>(i), std::move(feature));
        }

        auto build_callback = [](deglib::builder::BuilderStatus& status) {
            std::printf("  Build step %llu: added=%llu, improved=%llu\n",
                        (unsigned long long)status.step,
                        (unsigned long long)status.added,
                        (unsigned long long)status.improved);
        };

        builder.build(build_callback, false);

        double build_ms = now_ms() - t2;
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        // --------------------------------------------------------------------------
        // Exploration sweep
        // --------------------------------------------------------------------------
        run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads), "FP16 Build FP16 Explore");

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (FP16 Build FP16 Explore)\n");
        std::printf("=============================================\n");
        std::printf("Graph Build:             %.2f ms\n", build_ms);
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("FP16 bytes/vector:       %zu\n", bytes_per_fp16);
        std::printf("=============================================\n\n");

        return 0;
    } else if (mode == BenchmarkMode::EvpBuildEvpExploreFP16Rerank || mode == BenchmarkMode::EvpBuildEvpExplore || mode == BenchmarkMode::EvpBuildFP16ExternalSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank) {
        std::printf("=== EVP Bits Graph Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f, MODE=%s\n",
                    non_zeros, k_top, k_graph, k_ext, eps_ext,
                    mode == BenchmarkMode::EvpBuildFP16ExternalSearch ? "build-fp16-external-search" :
                    (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch ? "build-fp16-asymmetric-search" :
                    (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank ? "build-fp16-asymmetric-search-rerank" :
                    (mode == BenchmarkMode::EvpBuildEvpExploreFP16Rerank ? "build-evp-explore-fp16-rerank" : "build-evp-explore"))));
        std::printf("Threads: %u\n\n", threads);

        // --------------------------------------------------------------------------
        // Quantize
        // --------------------------------------------------------------------------
        double t1 = now_ms();

        auto quantized = deglib::quantization::quantize_batch(
            train_data_fp16, count, static_cast<uint32_t>(dims), non_zeros, threads);

        double quantize_ms = now_ms() - t1;
        std::printf("Quantize time: %.2f ms\n", quantize_ms);
        std::printf("Quantized size: %.2f MB\n\n",
                    static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

        // --------------------------------------------------------------------------
        // Build graph with Metric::EvpBits
        // --------------------------------------------------------------------------
        double t2 = now_ms();

        deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::EvpBits);
        deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(count), k_graph, feature_space);

        std::mt19937 rnd(42);
        deglib::builder::EvenRegularGraphBuilder builder(
            graph, rnd,
            deglib::builder::OptimizationTarget::LowLID,
            k_ext, eps_ext,
            0, 0.0f,
            5,
            0, 0,
            true,
            false,
            false
        );
        builder.setThreadCount(static_cast<uint32_t>(threads));

        const size_t bytes_per_evp = dims / 4;
        for (size_t i = 0; i < count; ++i) {
            std::vector<std::byte> feature(quantized.data() + i * bytes_per_evp, quantized.data() + (i + 1) * bytes_per_evp);
            builder.addEntry(static_cast<uint32_t>(i), std::move(feature));
        }

        auto build_callback = [](deglib::builder::BuilderStatus& status) {
            std::printf("  Build step %llu: added=%llu, improved=%llu\n",
                        (unsigned long long)status.step,
                        (unsigned long long)status.added,
                        (unsigned long long)status.improved);
        };

        builder.build(build_callback, false);

        double build_ms = now_ms() - t2;
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        // --------------------------------------------------------------------------
        // Exploration sweep
        // --------------------------------------------------------------------------
        double conversion_ms = 0.0;
        if (mode == BenchmarkMode::EvpBuildFP16ExternalSearch) {
            // Copy loaded FP16 data to a mutable vector for in-place reordering
            double t_copy_start = now_ms();
            std::vector<uint16_t> fp16_data(train_data_fp16, train_data_fp16 + count * dims);
            conversion_ms = now_ms() - t_copy_start;

            // Permute FP16 array in-place so that fp16_data[internal_index] holds the
            // features for that internal index (no proxy needed, no extra allocation).
            double t_reorder_start = now_ms();
            deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(
                graph, fp16_data.data(), dims);
            double reorder_ms = now_ms() - t_reorder_start;
            std::printf("In-place feature reorder time: %.2f ms\n", reorder_ms);
            conversion_ms += reorder_ms;

            // Build the external graph — references fp16_data directly, no copies.
            double t_ext_start = now_ms();
            deglib::FloatSpace fp16_ext_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            deglib::graph::ReadOnlyGraphExternal fp16_ext_graph(fp16_ext_space, graph, fp16_data.data());
            conversion_ms += now_ms() - t_ext_start;
            std::printf("ReadOnlyGraphExternal construction time: %.2f ms\n", now_ms() - t_ext_start);

            run_exploration_sweep(fp16_ext_graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build FP16 External Search (ReadOnlyGraphExternal)",
                                  true);
        } else if (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch) {
            double t_copy_start = now_ms();
            deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
            deglib::graph::ReadOnlyGraph readonly_asym_graph(static_cast<uint32_t>(count), k_graph, asymmetric_space, graph);
            conversion_ms = now_ms() - t_copy_start;
            std::printf("ReadOnlyGraph construction (copy) time: %.2f ms\n", conversion_ms);

            run_exploration_sweep(readonly_asym_graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build FP16 Asymmetric Search (ReadOnlyGraph)",
                                  false,
                                  nullptr,
                                  0,
                                  nullptr,
                                  dims,
                                  true,
                                  train_data_fp16);
        } else if (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank) {
            double t_copy_start = now_ms();
            deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
            deglib::graph::ReadOnlyGraph readonly_asym_graph(static_cast<uint32_t>(count), k_graph, asymmetric_space, graph);
            conversion_ms = now_ms() - t_copy_start;
            std::printf("ReadOnlyGraph construction (copy) time: %.2f ms\n", conversion_ms);

            deglib::FloatSpace fp16_rerank_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            run_exploration_sweep(readonly_asym_graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build FP16 Asymmetric Search FP16 Rerank (ReadOnlyGraph)",
                                  false,
                                  train_data_fp16,
                                  dims * sizeof(uint16_t),
                                  fp16_rerank_space.get_dist_func(),
                                  dims,
                                  true,
                                  train_data_fp16);
        } else if (mode == BenchmarkMode::EvpBuildEvpExploreFP16Rerank) {
            deglib::FloatSpace fp16_rerank_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build EVP Explore FP16 Rerank",
                                  false,
                                  train_data_fp16,
                                  dims * sizeof(uint16_t),
                                  fp16_rerank_space.get_dist_func(),
                                  dims);
        } else {
            run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build EVP Explore");
        }

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (EVP Bits)\n");
        std::printf("=============================================\n");
        std::printf("Quantize:                %.2f ms\n", quantize_ms);
        std::printf("Graph Build:             %.2f ms\n", build_ms);
        if (mode == BenchmarkMode::EvpBuildFP16ExternalSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank) {
            std::printf("Graph Conversion:        %.2f ms\n", conversion_ms);
        }
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("NON_ZEROS:               %u\n", non_zeros);
        std::printf("=============================================\n\n");

        return 0;
    } else if (mode == BenchmarkMode::EvpLinearSearch) {
        std::printf("=== EVP Bits Linear Search Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("NON_ZEROS=%u, K_TOP=%u\n", non_zeros, k_top);
        std::printf("Threads: %u\n\n", threads);

        // --------------------------------------------------------------------------
        // Quantize
        // --------------------------------------------------------------------------
        double t1 = now_ms();

        auto quantized = deglib::quantization::quantize_batch(
            train_data_fp16, count, static_cast<uint32_t>(dims), non_zeros, threads);

        double quantize_ms = now_ms() - t1;
        std::printf("Quantize time: %.2f ms\n", quantize_ms);
        std::printf("Quantized size: %.2f MB\n\n",
                    static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

        // --------------------------------------------------------------------------
        // Linear Search using distances.h
        // --------------------------------------------------------------------------
        double t_search_start = now_ms();

        std::vector<std::vector<uint32_t>> results(count, std::vector<uint32_t>(k_top));
        const size_t bytes_per_evp = dims / 4;
        const uint32_t dims_u32 = static_cast<uint32_t>(dims);

        deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
            [&](size_t i, size_t thread_id) {
                (void)thread_id;
                const std::byte* query_ptr = quantized.data() + i * bytes_per_evp;

                std::vector<std::pair<float, uint32_t>> top;
                top.reserve(k_top + 1);

                for (size_t j = 0; j < count; ++j) {
                    if (i == j) {
                        continue;
                    }

                    const std::byte* cand_ptr = quantized.data() + j * bytes_per_evp;
                    float dist = deglib::distances::EvpBitsSimilarity::compare(query_ptr, cand_ptr, &dims_u32);

                    if (top.size() < k_top) {
                        top.push_back({dist, static_cast<uint32_t>(j)});
                        std::push_heap(top.begin(), top.end());
                    } else if (dist < top.front().first) {
                        std::pop_heap(top.begin(), top.end());
                        top.back() = {dist, static_cast<uint32_t>(j)};
                        std::push_heap(top.begin(), top.end());
                    }
                }

                std::sort_heap(top.begin(), top.end());

                for (uint32_t k = 0; k < k_top && k < top.size(); ++k) {
                    results[i][k] = top[k].second;
                }
            });

        double search_ms = now_ms() - t_search_start;
        std::printf("Linear Search time: %.2f ms\n", search_ms);

        // --------------------------------------------------------------------------
        // Calculate Recall
        // --------------------------------------------------------------------------
        int total_hits = 0;
        for (size_t i = 0; i < count; ++i) {
            const int32_t* gt_row = &gt_data[i * gt_dims];
            for (uint32_t k = 1; k <= k_top && k < gt_dims; ++k) {
                uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1→0 indexed
                for (uint32_t j = 0; j < k_top; ++j) {
                    if (results[i][j] == gt_idx) {
                        total_hits++;
                        break;
                    }
                }
            }
        }

        float recall = static_cast<float>(total_hits) / (count * k_top);
        std::printf("Recall@%u: %.4f\n", k_top, recall);

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (EVP Bits Linear Search)\n");
        std::printf("=============================================\n");
        std::printf("Quantize:                %.2f ms\n", quantize_ms);
        std::printf("Linear Search:           %.2f ms\n", search_ms);
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("Recall@%u:                %.4f\n", k_top, recall);
        std::printf("=============================================\n\n");

        return 0;
    }
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
#if defined(USE_AVX)
    fmt::print("Using AVX2...\n");
#elif defined(USE_SSE)
    fmt::print("Using SSE...\n");
#else
    fmt::print("Using arch...\n");
#endif

    // --------------------------------------------------------------------------
    // Parameters
    // --------------------------------------------------------------------------
    constexpr uint32_t NON_ZEROS = 512;
    constexpr uint32_t K_TOP     = 15;
    constexpr uint8_t  K_GRAPH   = 32;
    constexpr uint8_t  K_EXT     = 32;
    constexpr float    EPS_EXT   = 0.001f;

    // --------------------------------------------------------------------------
    // Parse data path and mode
    // --------------------------------------------------------------------------
    std::filesystem::path data_path;
    // std::string mode = "fp16-build-fp16-explore"; 
    std::string mode = "evp-build-fp16-asymmetric-search";  // default 
    // std::string mode = "evp-build-fp16-external-search"; 
    // std::string mode = "evp-build-evp-explore"; 
    
    

    if (argc >= 2) {
        data_path = argv[1];
    } else {
#ifdef DATA_PATH
        data_path = DATA_PATH;
#else
    std::fprintf(stderr, "Usage: %s <data_path> [evp-build-evp-explore-fp16-rerank|evp-build-evp-explore|evp-build-fp16-external-search|fp16-build-fp16-explore|evp-linear-search|evp-build-fp16-asymmetric-search|evp-build-fp16-asymmetric-search-rerank|bench-asymmetric]\n", argv[0]);
        return 1;
#endif
    }

    if (argc >= 3) {
        mode = argv[2];
    }

    // Support old short names as aliases for ease of use
    if (mode == "fp16") mode = "fp16-build-fp16-explore";
    if (mode == "evp") mode = "evp-build-evp-explore-fp16-rerank";
    if (mode == "evp-no-rerank") mode = "evp-build-evp-explore";
    if (mode == "evp-linear") mode = "evp-linear-search";
    if (mode == "evp-asymmetric") mode = "evp-build-fp16-asymmetric-search";
    if (mode == "evp-asymmetric-rerank") mode = "evp-build-fp16-asymmetric-search-rerank";
    if (mode == "bench-asymmetric") mode = "fp16-evp-asymmetric-bench";

    if (mode != "evp-build-evp-explore-fp16-rerank" && mode != "evp-build-evp-explore" && mode != "evp-build-fp16-external-search" && mode != "fp16-build-fp16-explore" && mode != "evp-linear-search" && mode != "evp-build-fp16-asymmetric-search" && mode != "evp-build-fp16-asymmetric-search-rerank" && mode != "fp16-evp-asymmetric-bench") {
        std::fprintf(stderr, "Unknown mode '%s'. Use 'evp-build-evp-explore-fp16-rerank', 'evp-build-evp-explore', 'evp-build-fp16-external-search', 'fp16-build-fp16-explore', 'evp-linear-search', 'evp-build-fp16-asymmetric-search', 'evp-build-fp16-asymmetric-search-rerank', or 'bench-asymmetric'.\n", mode.c_str());
        return 1;
    }

    const size_t threads = 6;

    // --------------------------------------------------------------------------
    // Dispatch to selected mode
    // --------------------------------------------------------------------------
    BenchmarkMode benchmark_mode = BenchmarkMode::FP16BuildFP16Explore;
    if (mode == "evp-build-evp-explore-fp16-rerank") {
        benchmark_mode = BenchmarkMode::EvpBuildEvpExploreFP16Rerank;
    } else if (mode == "evp-build-evp-explore") {
        benchmark_mode = BenchmarkMode::EvpBuildEvpExplore;
    } else if (mode == "evp-build-fp16-external-search") {
        benchmark_mode = BenchmarkMode::EvpBuildFP16ExternalSearch;
    } else if (mode == "evp-linear-search") {
        benchmark_mode = BenchmarkMode::EvpLinearSearch;
    } else if (mode == "evp-build-fp16-asymmetric-search") {
        benchmark_mode = BenchmarkMode::EvpBuildFP16AsymmetricSearch;
    } else if (mode == "evp-build-fp16-asymmetric-search-rerank") {
        benchmark_mode = BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank;
    } else if (mode == "fp16-evp-asymmetric-bench") {
        benchmark_mode = BenchmarkMode::FP16EvpAsymmetricMicrobench;
    }

    return run_benchmark(benchmark_mode, data_path, threads, NON_ZEROS, K_GRAPH, K_EXT, EPS_EXT, K_TOP);
}
