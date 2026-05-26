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

#include "builder.h"
#include "concurrent.h"
#include "distances.h"
#include "graph/sizebounded_graph.h"
#include "quantization/evp_quantize.h"
#include "repository.h"

#include "evp_common.h"

// ============================================================================
// Microbench: FP16-EVP Asymmetric Similarity
// ============================================================================

static void bench_fp16_evp_asymmetric(
    const std::vector<std::vector<std::byte>>& fp16_data_vecs,
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

    // Generate once a vector with numbers 0 to count-1, shuffle it to simulate random memory access
    std::vector<uint32_t> shuffled_indices(count);
    std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0u);
    std::mt19937 g(1337); // Deterministic seed for fair reproducible benchmark
    std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), g);

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
            const uint32_t db_idx = shuffled_indices[c];
            const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());
            const std::byte* b = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
            sink += fn(a, b, &dim_u32);
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
                const std::byte* b = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
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
            const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n]));
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(a_ps, m));
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

    auto ip_avx2_batch4_ptr = [](const uint16_t* a, const std::byte* const* b_arr, const uint32_t* qty_ptr, float* dists) {
        const uint32_t dim = *qty_ptr;
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b[4];
        const std::byte* negs_b[4];
        for (int j = 0; j < 4; ++j) {
            ones_b[j] = b_arr[j];
            negs_b[j] = b_arr[j] + mask_bytes;
        }

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        for (size_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            const __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            const __m256 a_ps = _mm256_cvtph_ps(raw_a);

            #define ACCUMULATE_NEIGHBOR(j, sum_reg) { \
                const uint8_t o = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n])); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps, m)); \
            }

            ACCUMULATE_NEIGHBOR(0, sum0);
            ACCUMULATE_NEIGHBOR(1, sum1);
            ACCUMULATE_NEIGHBOR(2, sum2);
            ACCUMULATE_NEIGHBOR(3, sum3);

            #undef ACCUMULATE_NEIGHBOR
        }

        auto hsum256 = [](const __m256 sum256) -> float {
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(16) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        };

        dists[0] = 1.0f - hsum256(sum0);
        dists[1] = 1.0f - hsum256(sum1);
        dists[2] = 1.0f - hsum256(sum2);
        dists[3] = 1.0f - hsum256(sum3);
    };

    auto ip_avx2_batch8_ptr = [](const uint16_t* a, const std::byte* const* b_arr, const uint32_t* qty_ptr, float* dists) {
        const uint32_t dim = *qty_ptr;
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b[8];
        const std::byte* negs_b[8];
        for (int j = 0; j < 8; ++j) {
            ones_b[j] = b_arr[j];
            negs_b[j] = b_arr[j] + mask_bytes;
        }

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();
        __m256 sum5 = _mm256_setzero_ps();
        __m256 sum6 = _mm256_setzero_ps();
        __m256 sum7 = _mm256_setzero_ps();

        for (size_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            const __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            const __m256 a_ps = _mm256_cvtph_ps(raw_a);

            #define ACCUMULATE_NEIGHBOR(j, sum_reg) { \
                const uint8_t o = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n])); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps, m)); \
            }

            ACCUMULATE_NEIGHBOR(0, sum0);
            ACCUMULATE_NEIGHBOR(1, sum1);
            ACCUMULATE_NEIGHBOR(2, sum2);
            ACCUMULATE_NEIGHBOR(3, sum3);
            ACCUMULATE_NEIGHBOR(4, sum4);
            ACCUMULATE_NEIGHBOR(5, sum5);
            ACCUMULATE_NEIGHBOR(6, sum6);
            ACCUMULATE_NEIGHBOR(7, sum7);

            #undef ACCUMULATE_NEIGHBOR
        }

        auto hsum256 = [](const __m256 sum256) -> float {
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(16) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        };

        dists[0] = 1.0f - hsum256(sum0);
        dists[1] = 1.0f - hsum256(sum1);
        dists[2] = 1.0f - hsum256(sum2);
        dists[3] = 1.0f - hsum256(sum3);
        dists[4] = 1.0f - hsum256(sum4);
        dists[5] = 1.0f - hsum256(sum5);
        dists[6] = 1.0f - hsum256(sum6);
        dists[7] = 1.0f - hsum256(sum7);
    };

    auto ip_avx2_batch4_unroll2 = [](const uint16_t* a, const std::byte* const* b_arr, const uint32_t* qty_ptr, float* dists) {
        const uint32_t dim = *qty_ptr;
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b[4];
        const std::byte* negs_b[4];
        for (int j = 0; j < 4; ++j) {
            ones_b[j] = b_arr[j];
            negs_b[j] = b_arr[j] + mask_bytes;
        }

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 1 < mask_bytes; byte_idx += 2) {
            const __m256i raw16 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m128i raw_a0 = _mm256_extractf128_si256(raw16, 0);
            const __m128i raw_a1 = _mm256_extractf128_si256(raw16, 1);
            const __m256 a_ps_0 = _mm256_cvtph_ps(raw_a0);
            const __m256 a_ps_1 = _mm256_cvtph_ps(raw_a1);

            #define ACCUMULATE_NEIGHBOR(j, sum_reg) { \
                const uint8_t o0 = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n0 = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const uint8_t o1 = static_cast<uint8_t>(ones_b[j][byte_idx + 1]); \
                const uint8_t n1 = static_cast<uint8_t>(negs_b[j][byte_idx + 1]); \
                const __m256 m0 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0])); \
                const __m256 m1 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1])); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_0, m0)); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_1, m1)); \
            }

            ACCUMULATE_NEIGHBOR(0, sum0);
            ACCUMULATE_NEIGHBOR(1, sum1);
            ACCUMULATE_NEIGHBOR(2, sum2);
            ACCUMULATE_NEIGHBOR(3, sum3);

            #undef ACCUMULATE_NEIGHBOR
        }

        if (byte_idx < mask_bytes) {
            const __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            const __m256 a_ps = _mm256_cvtph_ps(raw_a);

            #define ACCUMULATE_NEIGHBOR_REMAINDER(j, sum_reg) { \
                const uint8_t o = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n])); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps, m)); \
            }

            ACCUMULATE_NEIGHBOR_REMAINDER(0, sum0);
            ACCUMULATE_NEIGHBOR_REMAINDER(1, sum1);
            ACCUMULATE_NEIGHBOR_REMAINDER(2, sum2);
            ACCUMULATE_NEIGHBOR_REMAINDER(3, sum3);

            #undef ACCUMULATE_NEIGHBOR_REMAINDER
        }

        auto hsum256 = [](const __m256 sum256) -> float {
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(16) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        };

        dists[0] = 1.0f - hsum256(sum0);
        dists[1] = 1.0f - hsum256(sum1);
        dists[2] = 1.0f - hsum256(sum2);
        dists[3] = 1.0f - hsum256(sum3);
    };

    auto ip_avx2_batch8_unroll2 = [](const uint16_t* a, const std::byte* const* b_arr, const uint32_t* qty_ptr, float* dists) {
        const uint32_t dim = *qty_ptr;
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b[8];
        const std::byte* negs_b[8];
        for (int j = 0; j < 8; ++j) {
            ones_b[j] = b_arr[j];
            negs_b[j] = b_arr[j] + mask_bytes;
        }

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();
        __m256 sum5 = _mm256_setzero_ps();
        __m256 sum6 = _mm256_setzero_ps();
        __m256 sum7 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 1 < mask_bytes; byte_idx += 2) {
            const __m256i raw16 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m128i raw_a0 = _mm256_extractf128_si256(raw16, 0);
            const __m128i raw_a1 = _mm256_extractf128_si256(raw16, 1);
            const __m256 a_ps_0 = _mm256_cvtph_ps(raw_a0);
            const __m256 a_ps_1 = _mm256_cvtph_ps(raw_a1);

            #define ACCUMULATE_NEIGHBOR(j, sum_reg) { \
                const uint8_t o0 = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n0 = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const uint8_t o1 = static_cast<uint8_t>(ones_b[j][byte_idx + 1]); \
                const uint8_t n1 = static_cast<uint8_t>(negs_b[j][byte_idx + 1]); \
                const __m256 m0 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0])); \
                const __m256 m1 = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1])); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_0, m0)); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_1, m1)); \
            }

            ACCUMULATE_NEIGHBOR(0, sum0);
            ACCUMULATE_NEIGHBOR(1, sum1);
            ACCUMULATE_NEIGHBOR(2, sum2);
            ACCUMULATE_NEIGHBOR(3, sum3);
            ACCUMULATE_NEIGHBOR(4, sum4);
            ACCUMULATE_NEIGHBOR(5, sum5);
            ACCUMULATE_NEIGHBOR(6, sum6);
            ACCUMULATE_NEIGHBOR(7, sum7);

            #undef ACCUMULATE_NEIGHBOR
        }

        if (byte_idx < mask_bytes) {
            const __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            const __m256 a_ps = _mm256_cvtph_ps(raw_a);

            #define ACCUMULATE_NEIGHBOR_REMAINDER(j, sum_reg) { \
                const uint8_t o = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n])); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps, m)); \
            }

            ACCUMULATE_NEIGHBOR_REMAINDER(0, sum0);
            ACCUMULATE_NEIGHBOR_REMAINDER(1, sum1);
            ACCUMULATE_NEIGHBOR_REMAINDER(2, sum2);
            ACCUMULATE_NEIGHBOR_REMAINDER(3, sum3);
            ACCUMULATE_NEIGHBOR_REMAINDER(4, sum4);
            ACCUMULATE_NEIGHBOR_REMAINDER(5, sum5);
            ACCUMULATE_NEIGHBOR_REMAINDER(6, sum6);
            ACCUMULATE_NEIGHBOR_REMAINDER(7, sum7);

            #undef ACCUMULATE_NEIGHBOR_REMAINDER
        }

        auto hsum256 = [](const __m256 sum256) -> float {
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(16) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        };

        dists[0] = 1.0f - hsum256(sum0);
        dists[1] = 1.0f - hsum256(sum1);
        dists[2] = 1.0f - hsum256(sum2);
        dists[3] = 1.0f - hsum256(sum3);
        dists[4] = 1.0f - hsum256(sum4);
        dists[5] = 1.0f - hsum256(sum5);
        dists[6] = 1.0f - hsum256(sum6);
        dists[7] = 1.0f - hsum256(sum7);
    };

    auto ip_avx2_batch4_unroll4 = [](const uint16_t* a, const std::byte* const* b_arr, const uint32_t* qty_ptr, float* dists) {
        const uint32_t dim = *qty_ptr;
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b[4];
        const std::byte* negs_b[4];
        for (int j = 0; j < 4; ++j) {
            ones_b[j] = b_arr[j];
            negs_b[j] = b_arr[j] + mask_bytes;
        }

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 3 < mask_bytes; byte_idx += 4) {
            const __m256i raw16_0 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m256i raw16_1 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 2) * 8]);

            const __m128i r0 = _mm256_extractf128_si256(raw16_0, 0);
            const __m128i r1 = _mm256_extractf128_si256(raw16_0, 1);
            const __m128i r2 = _mm256_extractf128_si256(raw16_1, 0);
            const __m128i r3 = _mm256_extractf128_si256(raw16_1, 1);

            const __m256 a_ps_0 = _mm256_cvtph_ps(r0);
            const __m256 a_ps_1 = _mm256_cvtph_ps(r1);
            const __m256 a_ps_2 = _mm256_cvtph_ps(r2);
            const __m256 a_ps_3 = _mm256_cvtph_ps(r3);

            #define ACCUMULATE_NEIGHBOR(j, sum_reg) { \
                const uint8_t o0 = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n0 = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const uint8_t o1 = static_cast<uint8_t>(ones_b[j][byte_idx + 1]); \
                const uint8_t n1 = static_cast<uint8_t>(negs_b[j][byte_idx + 1]); \
                const uint8_t o2 = static_cast<uint8_t>(ones_b[j][byte_idx + 2]); \
                const uint8_t n2 = static_cast<uint8_t>(negs_b[j][byte_idx + 2]); \
                const uint8_t o3 = static_cast<uint8_t>(ones_b[j][byte_idx + 3]); \
                const uint8_t n3 = static_cast<uint8_t>(negs_b[j][byte_idx + 3]); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_0, _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o0]), _mm256_load_ps(avx2_tbl.negs[n0])))); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_1, _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o1]), _mm256_load_ps(avx2_tbl.negs[n1])))); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_2, _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o2]), _mm256_load_ps(avx2_tbl.negs[n2])))); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps_3, _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o3]), _mm256_load_ps(avx2_tbl.negs[n3])))); \
            }

            ACCUMULATE_NEIGHBOR(0, sum0);
            ACCUMULATE_NEIGHBOR(1, sum1);
            ACCUMULATE_NEIGHBOR(2, sum2);
            ACCUMULATE_NEIGHBOR(3, sum3);

            #undef ACCUMULATE_NEIGHBOR
        }

        // Remainder
        for (; byte_idx < mask_bytes; ++byte_idx) {
            const __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            const __m256 a_ps = _mm256_cvtph_ps(raw_a);

            #define ACCUMULATE_NEIGHBOR_REMAINDER(j, sum_reg) { \
                const uint8_t o = static_cast<uint8_t>(ones_b[j][byte_idx]); \
                const uint8_t n = static_cast<uint8_t>(negs_b[j][byte_idx]); \
                const __m256 m = _mm256_sub_ps(_mm256_load_ps(avx2_tbl.ones[o]), _mm256_load_ps(avx2_tbl.negs[n])); \
                sum_reg = _mm256_add_ps(sum_reg, _mm256_mul_ps(a_ps, m)); \
            }

            ACCUMULATE_NEIGHBOR_REMAINDER(0, sum0);
            ACCUMULATE_NEIGHBOR_REMAINDER(1, sum1);
            ACCUMULATE_NEIGHBOR_REMAINDER(2, sum2);
            ACCUMULATE_NEIGHBOR_REMAINDER(3, sum3);

            #undef ACCUMULATE_NEIGHBOR_REMAINDER
        }

        auto hsum256 = [](const __m256 sum256) -> float {
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(16) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        };

        dists[0] = 1.0f - hsum256(sum0);
        dists[1] = 1.0f - hsum256(sum1);
        dists[2] = 1.0f - hsum256(sum2);
        dists[3] = 1.0f - hsum256(sum3);
    };


    auto run_case_batch4 = [&](const char* name, auto&& fn_batch4) {
        constexpr int trials = 5;

        // Warmup
        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>((comparisons / 4) / 10, 12500);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());
            
            const std::byte* b[4];
            for (int j = 0; j < 4; ++j) {
                const uint32_t c = static_cast<uint32_t>((( (i * 4 + j) * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                b[j] = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
            }
            
            alignas(32) float dists[4];
            fn_batch4(a, b, &dim_u32, dists);
            for (int j = 0; j < 4; ++j) {
                sink += dists[j];
            }
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons / 4; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());
                
                const std::byte* b[4];
                for (int j = 0; j < 4; ++j) {
                    const uint32_t c = static_cast<uint32_t>((( (i * 4 + j) * 40503ULL) + 17ULL) % count);
                    const uint32_t db_idx = shuffled_indices[c];
                    b[j] = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
                }
                
                alignas(32) float dists[4];
                fn_batch4(a, b, &dim_u32, dists);
                for (int j = 0; j < 4; ++j) {
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

    auto run_case_batch8 = [&](const char* name, auto&& fn_batch8) {
        constexpr int trials = 5;

        // Warmup
        volatile float sink = 0.0f;
        const size_t warm = std::min<size_t>((comparisons / 8) / 10, 6250);
        for (size_t i = 0; i < warm; ++i) {
            const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
            const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());
            
            const std::byte* b[8];
            for (int j = 0; j < 8; ++j) {
                const uint32_t c = static_cast<uint32_t>((( (i * 8 + j) * 40503ULL) + 17ULL) % count);
                const uint32_t db_idx = shuffled_indices[c];
                b[j] = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
            }
            
            alignas(32) float dists[8];
            fn_batch8(a, b, &dim_u32, dists);
            for (int j = 0; j < 8; ++j) {
                sink += dists[j];
            }
        }

        double best_ms = std::numeric_limits<double>::infinity();
        double best_ns_per_op = std::numeric_limits<double>::infinity();

        for (int t = 0; t < trials; ++t) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < comparisons / 8; ++i) {
                const uint32_t q = static_cast<uint32_t>((i * 2654435761ULL) % count);
                const uint16_t* a = reinterpret_cast<const uint16_t*>(fp16_data_vecs[q].data());
                
                const std::byte* b[8];
                for (int j = 0; j < 8; ++j) {
                    const uint32_t c = static_cast<uint32_t>((( (i * 8 + j) * 40503ULL) + 17ULL) % count);
                    const uint32_t db_idx = shuffled_indices[c];
                    b[j] = evp_data + static_cast<size_t>(db_idx) * bytes_per_evp;
                }
                
                alignas(32) float dists[8];
                fn_batch8(a, b, &dim_u32, dists);
                for (int j = 0; j < 8; ++j) {
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
    std::printf("  - ip_avx2_batch4_ptr (batch candidate)\n");
    std::printf("  - ip_avx2_batch8_ptr (batch candidate)\n");
    std::printf("  - ip_avx2_batch4_unroll2 (batch candidate)\n");
    std::printf("  - ip_avx2_batch8_unroll2 (batch candidate)\n");
    std::printf("  - ip_avx2_batch4_unroll4 (batch candidate)\n");
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
    run_case_batch4("ip_avx2_b4_ptr", ip_avx2_batch4_ptr);
    run_case_batch8("ip_avx2_b8_ptr", ip_avx2_batch8_ptr);
    run_case_batch4("ip_avx2_b4_u2", ip_avx2_batch4_unroll2);
    run_case_batch8("ip_avx2_b8_u2", ip_avx2_batch8_unroll2);
    run_case_batch4("ip_avx2_b4_u4", ip_avx2_batch4_unroll4);
#endif
#if defined(USE_AVX512)
    run_case("ip_avx512", deglib::distances::FP16EvpAsymmetricSimilarity::ip_avx512);
#endif
    run_case("compare", deglib::distances::FP16EvpAsymmetricSimilarity::compare);

    std::printf("\nBEST: %s (%.2f ns/op)\n", best.name, best.ns_per_op);

    std::printf("\n=============================================\n");
}

int main(int argc, char* argv[]) {
#if defined(USE_AVX)
    fmt::print("Using AVX2...\n");
#elif defined(USE_SSE)
    fmt::print("Using SSE...\n");
#else
    fmt::print("Using arch...\n");
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

    const auto train_hvecs = data_path / "train.hvecs";
    if (!std::filesystem::exists(train_hvecs)) {
        std::fprintf(stderr, "Error: %ls not found\n", train_hvecs.wstring().c_str());
        return 1;
    }

    double t_load = now_ms();
    size_t dims = 0, count = 0;
    auto train_vectors = hvecs_read(train_hvecs.string().c_str(), dims, count);
    double load_ms = now_ms() - t_load;

    std::printf("Loaded %ls: %zu vectors, dim=%zu\n", train_hvecs.filename().wstring().c_str(), count, dims);
    std::printf("Load time: %.2f ms\n\n", load_ms);

    double t_q = now_ms();
    auto quantized = deglib::quantization::quantize_batch(
        train_vectors, static_cast<uint32_t>(dims), NON_ZEROS, threads);
    double quantize_ms = now_ms() - t_q;

    std::printf("Quantize time: %.2f ms (produced %.2f MB)\n\n",
                quantize_ms,
                static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

    bench_fp16_evp_asymmetric(
        train_vectors,
        count,
        static_cast<uint32_t>(dims),
        quantized.data(),
        NON_ZEROS,
        threads);

    return 0;
}
