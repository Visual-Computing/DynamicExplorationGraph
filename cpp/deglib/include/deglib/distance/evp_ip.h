#pragma once

#include "deglib/config.h"
#include "deglib/distance/residual_mode.h"

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <variant>
#include <vector>

#if defined(DEGLIB_X86)
    #include <immintrin.h>
#endif

namespace deglib::distances::evp_ip {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------- EVP Inner Product Dists ---------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// EVP (Explicit Value Product) quantization converts a float vector into a bit-packed
// representation: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
//
// The symmetric EVP inner product distance computes popcount intersections:
//   aa = popcount(ones(a) AND ones(b))   — same-sign positive overlap
//   bb = popcount(neg(a) AND neg(b))     — same-sign negative overlap
//   cc = popcount(ones(a) AND neg(b))    — cross-sign overlap
//   dd = popcount(ones(b) AND neg(a))    — cross-sign overlap
//   similarity = aa + bb + dim - cc - dd
//   distance = 1 - similarity / (2 * dim)
//
// qty_ptr points to the original float dimension (must be divisible by 8).
// ---------------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------
// Scalar fallback — no SIMD required.
// Uses std::popcount for portable bit counting.
// -------------------------------------------------------------------

class EvpInnerProduct {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        return 1.f - (compare_naive(pVect1v, pVect2v, qty_ptr) / (2.f * static_cast<float>(*static_cast<const uint32_t*>(qty_ptr))));
    }

    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        for (size_t i = 0; i < count; ++i) {
            dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
        }
    }

    inline static float compare_naive(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const std::byte* a = static_cast<const std::byte*>(pVect1v);
        const std::byte* b = static_cast<const std::byte*>(pVect2v);
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_a = a;
        const std::byte* negs_a = a + mask_bytes;
        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        uint32_t same_sign = 0;
        uint32_t diff_sign = 0;

        size_t i = 0;
        for (; i + sizeof(uint64_t) <= mask_bytes; i += sizeof(uint64_t)) {
            const uint64_t o1 = *reinterpret_cast<const uint64_t*>(&ones_a[i]);
            const uint64_t o2 = *reinterpret_cast<const uint64_t*>(&ones_b[i]);
            const uint64_t n1 = *reinterpret_cast<const uint64_t*>(&negs_a[i]);
            const uint64_t n2 = *reinterpret_cast<const uint64_t*>(&negs_b[i]);

            same_sign += std::popcount((o1 & o2) | (n1 & n2));
            diff_sign += std::popcount((o1 & n2) | (o2 & n1));
        }

        for (; i < mask_bytes; ++i) {
            const unsigned int o1 = static_cast<uint8_t>(ones_a[i]);
            const unsigned int o2 = static_cast<uint8_t>(ones_b[i]);
            const unsigned int n1 = static_cast<uint8_t>(negs_a[i]);
            const unsigned int n2 = static_cast<uint8_t>(negs_b[i]);

            same_sign += std::popcount((o1 & o2) | (n1 & n2));
            diff_sign += std::popcount((o1 & n2) | (o2 & n1));
        }

        return static_cast<float>(same_sign + dim) - static_cast<float>(diff_sign);
    }
};  // EvpInnerProduct

#if defined(DEGLIB_X86)
    // -------------------------------------------------------------------
    // EvpInnerProduct SIMD implementations — process EVP bit-packed
    // vectors with aligned SIMD portions plus scalar residuals for any
    // unaligned tail. Separate classes per SIMD width so that compare()
    // has zero runtime dispatch overhead — select_dist() chooses the class.
    // The ResidualMode template parameter controls which loop variants
    // are compiled in. When a mode is false, the corresponding loop is
    // eliminated at compile time, producing a faster path for dimensions
    // that are known to be SIMD-aligned.
    // -------------------------------------------------------------------

DEGLIB_TARGET_AVX512 inline static uint64_t evp_hsum512(__m512i s) {
    alignas(64) int64_t arr[8];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(arr), s);
    return static_cast<uint64_t>(arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]);
}

DEGLIB_TARGET_AVX2 inline static uint64_t evp_hsum256(__m256i s) {
    alignas(32) int64_t arr[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), s);
    return static_cast<uint64_t>(arr[0] + arr[1] + arr[2] + arr[3]);
}
template <ResidualMode Mode = ResidualMode::Full>
class EvpInnerProduct_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const std::byte* a = static_cast<const std::byte*>(pVect1v);
        const std::byte* b = static_cast<const std::byte*>(pVect2v);
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_a = a;
        const std::byte* negs_a = a + mask_bytes;
        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m512i acc_same = _mm512_setzero_si512();
        __m512i acc_diff = _mm512_setzero_si512();

        const size_t block = 64;
        size_t i = 0;
        if constexpr (HasDualSimd) {
            for (; i + 2 * block <= mask_bytes; i += 2 * block) {
                __m512i o1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_a[i]));
                __m512i o2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_b[i]));
                __m512i n1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_a[i]));
                __m512i n2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_b[i]));

                __m512i same = _mm512_or_si512(_mm512_and_si512(o1, o2), _mm512_and_si512(n1, n2));
                __m512i diff = _mm512_or_si512(_mm512_and_si512(o1, n2), _mm512_and_si512(o2, n1));

                acc_same = _mm512_add_epi64(acc_same, _mm512_popcnt_epi64(same));
                acc_diff = _mm512_add_epi64(acc_diff, _mm512_popcnt_epi64(diff));

                __m512i o1b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_a[i + block]));
                __m512i o2b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_b[i + block]));
                __m512i n1b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_a[i + block]));
                __m512i n2b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_b[i + block]));

                __m512i same_b = _mm512_or_si512(_mm512_and_si512(o1b, o2b), _mm512_and_si512(n1b, n2b));
                __m512i diff_b = _mm512_or_si512(_mm512_and_si512(o1b, n2b), _mm512_and_si512(o2b, n1b));

                acc_same = _mm512_add_epi64(acc_same, _mm512_popcnt_epi64(same_b));
                acc_diff = _mm512_add_epi64(acc_diff, _mm512_popcnt_epi64(diff_b));
            }
        }
        if constexpr (HasSimd) {
            for (; i + block <= mask_bytes; i += block) {
                __m512i o1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_a[i]));
                __m512i o2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_b[i]));
                __m512i n1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_a[i]));
                __m512i n2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_b[i]));

                __m512i same = _mm512_or_si512(_mm512_and_si512(o1, o2), _mm512_and_si512(n1, n2));
                __m512i diff = _mm512_or_si512(_mm512_and_si512(o1, n2), _mm512_and_si512(o2, n1));

                acc_same = _mm512_add_epi64(acc_same, _mm512_popcnt_epi64(same));
                acc_diff = _mm512_add_epi64(acc_diff, _mm512_popcnt_epi64(diff));
            }
        }

        // Horizontal reduce of SIMD accumulators
        alignas(64) int64_t arr[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(arr), acc_same);
        uint64_t same_count = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(arr), acc_diff);
        uint64_t diff_count = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            for (; i < mask_bytes; ++i) {
                const unsigned int o1 = static_cast<uint8_t>(ones_a[i]);
                const unsigned int o2 = static_cast<uint8_t>(ones_b[i]);
                const unsigned int n1 = static_cast<uint8_t>(negs_a[i]);
                const unsigned int n2 = static_cast<uint8_t>(negs_b[i]);
                same_count += std::popcount((o1 & o2) | (n1 & n2));
                diff_count += std::popcount((o1 & n2) | (o2 & n1));
            }
        }

        const float sim = static_cast<float>(same_count + dim) - static_cast<float>(diff_count);
        return 1.f - sim / (2.f * static_cast<float>(dim));
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 4;
        const std::byte* query = static_cast<const std::byte*>(query_ptr);
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        auto batch_impl = [query, mask_bytes, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512 {
            const std::byte* q_ones = query;
            const std::byte* q_negs = query + mask_bytes;

            alignas(64) __m512i acc_same[BATCH_SIZE];
            alignas(64) __m512i acc_diff[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                acc_same[j] = _mm512_setzero_si512();
                acc_diff[j] = _mm512_setzero_si512();
            }

            const size_t block = 64;
            size_t i = 0;
            if constexpr (HasDualSimd) {
                for (; i + 2 * block <= mask_bytes; i += 2 * block) {
                    __m512i q_o1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&q_ones[i]));
                    __m512i q_n1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&q_negs[i]));
                    __m512i q_o1b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&q_ones[i + block]));
                    __m512i q_n1b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&q_negs[i + block]));

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const std::byte* db_ = static_cast<const std::byte*>(db[j]);
                        __m512i o2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&db_[i]));
                        __m512i n2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&db_[i + mask_bytes]));
                        __m512i o2b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&db_[i + block]));
                        __m512i n2b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&db_[i + block + mask_bytes]));

                        __m512i same = _mm512_or_si512(_mm512_and_si512(q_o1, o2), _mm512_and_si512(q_n1, n2));
                        __m512i diff = _mm512_or_si512(_mm512_and_si512(q_o1, n2), _mm512_and_si512(o2, q_n1));
                        acc_same[j] = _mm512_add_epi64(acc_same[j], _mm512_popcnt_epi64(same));
                        acc_diff[j] = _mm512_add_epi64(acc_diff[j], _mm512_popcnt_epi64(diff));

                        __m512i same_b = _mm512_or_si512(_mm512_and_si512(q_o1b, o2b), _mm512_and_si512(q_n1b, n2b));
                        __m512i diff_b = _mm512_or_si512(_mm512_and_si512(q_o1b, n2b), _mm512_and_si512(o2b, q_n1b));
                        acc_same[j] = _mm512_add_epi64(acc_same[j], _mm512_popcnt_epi64(same_b));
                        acc_diff[j] = _mm512_add_epi64(acc_diff[j], _mm512_popcnt_epi64(diff_b));
                    }
                }
            }
            if constexpr (HasSimd) {
                for (; i + block <= mask_bytes; i += block) {
                    __m512i q_o1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&q_ones[i]));
                    __m512i q_n1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&q_negs[i]));

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const std::byte* db_ = static_cast<const std::byte*>(db[j]);
                        __m512i o2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&db_[i]));
                        __m512i n2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&db_[i + mask_bytes]));

                        __m512i same = _mm512_or_si512(_mm512_and_si512(q_o1, o2), _mm512_and_si512(q_n1, n2));
                        __m512i diff = _mm512_or_si512(_mm512_and_si512(q_o1, n2), _mm512_and_si512(o2, q_n1));
                        acc_same[j] = _mm512_add_epi64(acc_same[j], _mm512_popcnt_epi64(same));
                        acc_diff[j] = _mm512_add_epi64(acc_diff[j], _mm512_popcnt_epi64(diff));
                    }
                }
            }

            alignas(64) uint64_t same_res[BATCH_SIZE], diff_res[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                same_res[j] = evp_hsum512(acc_same[j]);
                diff_res[j] = evp_hsum512(acc_diff[j]);
            }

            if constexpr (HasTail) {
                for (; i < mask_bytes; ++i) {
                    const unsigned int q_b1 = static_cast<uint8_t>(q_ones[i]);
                    const unsigned int q_n1 = static_cast<uint8_t>(q_negs[i]);
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const std::byte* db_ = static_cast<const std::byte*>(db[j]);
                        const unsigned int b2 = static_cast<uint8_t>(db_[i]);
                        const unsigned int n2 = static_cast<uint8_t>(db_[i + mask_bytes]);
                        same_res[j] += std::popcount((q_b1 & b2) | (q_n1 & n2));
                        diff_res[j] += std::popcount((q_b1 & n2) | (b2 & q_n1));
                    }
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                const float sim = static_cast<float>(same_res[j] + dim) - static_cast<float>(diff_res[j]);
                out_dists[j] = 1.f - sim / (2.f * static_cast<float>(dim));
            }
        };

        size_t i = 0;
        for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
            batch_impl(&db_arr[i], &dists[i]);
        }
        for (; i < count; ++i) {
            dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
        }
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class EvpInnerProduct_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }
    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const std::byte* a = static_cast<const std::byte*>(pVect1v);
        const std::byte* b = static_cast<const std::byte*>(pVect2v);
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_a = a;
        const std::byte* negs_a = a + mask_bytes;
        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256i acc_same = _mm256_setzero_si256();
        __m256i acc_diff = _mm256_setzero_si256();

        const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        const __m256i low_mask = _mm256_set1_epi8(0x0F);
        const __m256i zero = _mm256_setzero_si256();

        auto vector_popcnt = [&](__m256i vec) {
            __m256i lo = _mm256_and_si256(vec, low_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
            __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
            __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
            return _mm256_add_epi8(popcnt1, popcnt2);
        };

        const size_t block = 32;
        size_t i = 0;
        if constexpr (HasDualSimd) {
            for (; i + 2 * block <= mask_bytes; i += 2 * block) {
                __m256i o1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_a[i]));
                __m256i o2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_b[i]));
                __m256i n1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_a[i]));
                __m256i n2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_b[i]));

                __m256i same = _mm256_or_si256(_mm256_and_si256(o1, o2), _mm256_and_si256(n1, n2));
                __m256i diff = _mm256_or_si256(_mm256_and_si256(o1, n2), _mm256_and_si256(o2, n1));

                acc_same = _mm256_add_epi64(acc_same, _mm256_sad_epu8(vector_popcnt(same), zero));
                acc_diff = _mm256_add_epi64(acc_diff, _mm256_sad_epu8(vector_popcnt(diff), zero));

                __m256i o1b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_a[i + block]));
                __m256i o2b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_b[i + block]));
                __m256i n1b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_a[i + block]));
                __m256i n2b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_b[i + block]));

                __m256i same_b = _mm256_or_si256(_mm256_and_si256(o1b, o2b), _mm256_and_si256(n1b, n2b));
                __m256i diff_b = _mm256_or_si256(_mm256_and_si256(o1b, n2b), _mm256_and_si256(o2b, n1b));

                acc_same = _mm256_add_epi64(acc_same, _mm256_sad_epu8(vector_popcnt(same_b), zero));
                acc_diff = _mm256_add_epi64(acc_diff, _mm256_sad_epu8(vector_popcnt(diff_b), zero));
            }
        }
        if constexpr (HasSimd) {
            for (; i + block <= mask_bytes; i += block) {
                __m256i o1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_a[i]));
                __m256i o2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_b[i]));
                __m256i n1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_a[i]));
                __m256i n2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_b[i]));

                __m256i same = _mm256_or_si256(_mm256_and_si256(o1, o2), _mm256_and_si256(n1, n2));
                __m256i diff = _mm256_or_si256(_mm256_and_si256(o1, n2), _mm256_and_si256(o2, n1));

                acc_same = _mm256_add_epi64(acc_same, _mm256_sad_epu8(vector_popcnt(same), zero));
                acc_diff = _mm256_add_epi64(acc_diff, _mm256_sad_epu8(vector_popcnt(diff), zero));
            }
        }

        // Horizontal reduce of SIMD accumulators
        alignas(32) int64_t arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), acc_same);
        uint64_t same_count = arr[0] + arr[1] + arr[2] + arr[3];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), acc_diff);
        uint64_t diff_count = arr[0] + arr[1] + arr[2] + arr[3];

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            for (; i < mask_bytes; ++i) {
                const unsigned int o1 = static_cast<uint8_t>(ones_a[i]);
                const unsigned int o2 = static_cast<uint8_t>(ones_b[i]);
                const unsigned int n1 = static_cast<uint8_t>(negs_a[i]);
                const unsigned int n2 = static_cast<uint8_t>(negs_b[i]);
                same_count += std::popcount((o1 & o2) | (n1 & n2));
                diff_count += std::popcount((o1 & n2) | (o2 & n1));
            }
        }

        const float sim = static_cast<float>(same_count + dim) - static_cast<float>(diff_count);
        return 1.f - sim / (2.f * static_cast<float>(dim));
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 4;
        const std::byte* query = static_cast<const std::byte*>(query_ptr);
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        auto batch_impl = [query, mask_bytes, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2 {
            const std::byte* q_ones = query;
            const std::byte* q_negs = query + mask_bytes;

            const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
            const __m256i low_mask = _mm256_set1_epi8(0x0F);
            const __m256i zero = _mm256_setzero_si256();

            auto vector_popcnt = [&](__m256i vec) {
                __m256i lo = _mm256_and_si256(vec, low_mask);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
                __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
                __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
                return _mm256_add_epi8(popcnt1, popcnt2);
            };

            alignas(32) __m256i acc_same[BATCH_SIZE];
            alignas(32) __m256i acc_diff[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                acc_same[j] = _mm256_setzero_si256();
                acc_diff[j] = _mm256_setzero_si256();
            }

            const size_t block = 32;
            size_t i = 0;
            if constexpr (HasDualSimd) {
                for (; i + 2 * block <= mask_bytes; i += 2 * block) {
                    __m256i q_o1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&q_ones[i]));
                    __m256i q_n1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&q_negs[i]));
                    __m256i q_o1b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&q_ones[i + block]));
                    __m256i q_n1b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&q_negs[i + block]));

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const std::byte* db_ = static_cast<const std::byte*>(db[j]);
                        __m256i o2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[i]));
                        __m256i n2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[i + mask_bytes]));
                        __m256i o2b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[i + block]));
                        __m256i n2b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[i + block + mask_bytes]));

                        __m256i same = _mm256_or_si256(_mm256_and_si256(q_o1, o2), _mm256_and_si256(q_n1, n2));
                        __m256i diff = _mm256_or_si256(_mm256_and_si256(q_o1, n2), _mm256_and_si256(o2, q_n1));
                        acc_same[j] = _mm256_add_epi64(acc_same[j], _mm256_sad_epu8(vector_popcnt(same), zero));
                        acc_diff[j] = _mm256_add_epi64(acc_diff[j], _mm256_sad_epu8(vector_popcnt(diff), zero));

                        __m256i same_b = _mm256_or_si256(_mm256_and_si256(q_o1b, o2b), _mm256_and_si256(q_n1b, n2b));
                        __m256i diff_b = _mm256_or_si256(_mm256_and_si256(q_o1b, n2b), _mm256_and_si256(o2b, q_n1b));
                        acc_same[j] = _mm256_add_epi64(acc_same[j], _mm256_sad_epu8(vector_popcnt(same_b), zero));
                        acc_diff[j] = _mm256_add_epi64(acc_diff[j], _mm256_sad_epu8(vector_popcnt(diff_b), zero));
                    }
                }
            }
            if constexpr (HasSimd) {
                for (; i + block <= mask_bytes; i += block) {
                    __m256i q_o1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&q_ones[i]));
                    __m256i q_n1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&q_negs[i]));

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const std::byte* db_ = static_cast<const std::byte*>(db[j]);
                        __m256i o2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[i]));
                        __m256i n2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[i + mask_bytes]));

                        __m256i same = _mm256_or_si256(_mm256_and_si256(q_o1, o2), _mm256_and_si256(q_n1, n2));
                        __m256i diff = _mm256_or_si256(_mm256_and_si256(q_o1, n2), _mm256_and_si256(o2, q_n1));
                        acc_same[j] = _mm256_add_epi64(acc_same[j], _mm256_sad_epu8(vector_popcnt(same), zero));
                        acc_diff[j] = _mm256_add_epi64(acc_diff[j], _mm256_sad_epu8(vector_popcnt(diff), zero));
                    }
                }
            }

            alignas(32) uint64_t same_res[BATCH_SIZE], diff_res[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                same_res[j] = evp_hsum256(acc_same[j]);
                diff_res[j] = evp_hsum256(acc_diff[j]);
            }

            if constexpr (HasTail) {
                for (; i < mask_bytes; ++i) {
                    const unsigned int q_b1 = static_cast<uint8_t>(q_ones[i]);
                    const unsigned int q_n1 = static_cast<uint8_t>(q_negs[i]);
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const std::byte* db_ = static_cast<const std::byte*>(db[j]);
                        const unsigned int b2 = static_cast<uint8_t>(db_[i]);
                        const unsigned int n2 = static_cast<uint8_t>(db_[i + mask_bytes]);
                        same_res[j] += std::popcount((q_b1 & b2) | (q_n1 & n2));
                        diff_res[j] += std::popcount((q_b1 & n2) | (b2 & q_n1));
                    }
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                const float sim = static_cast<float>(same_res[j] + dim) - static_cast<float>(diff_res[j]);
                out_dists[j] = 1.f - sim / (2.f * static_cast<float>(dim));
            }
        };

        size_t i = 0;
        for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
            batch_impl(&db_arr[i], &dists[i]);
        }
        for (; i < count; ++i) {
            dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
        }
    }
};
#endif

using DistanceVariant = std::variant<
    EvpInnerProduct
#if defined(DEGLIB_X86)
    ,
    EvpInnerProduct_AVX512<ResidualMode::Full>,
    EvpInnerProduct_AVX512<ResidualMode::DualPlusSimd>,
    EvpInnerProduct_AVX512<ResidualMode::DualTail>,
    EvpInnerProduct_AVX512<ResidualMode::DualOnly>,
    EvpInnerProduct_AVX512<ResidualMode::SimdTail>,
    EvpInnerProduct_AVX512<ResidualMode::SimdOnly>,
    EvpInnerProduct_AVX512<ResidualMode::TailOnly>,
    EvpInnerProduct_AVX2<ResidualMode::Full>,
    EvpInnerProduct_AVX2<ResidualMode::DualPlusSimd>,
    EvpInnerProduct_AVX2<ResidualMode::DualTail>,
    EvpInnerProduct_AVX2<ResidualMode::DualOnly>,
    EvpInnerProduct_AVX2<ResidualMode::SimdTail>,
    EvpInnerProduct_AVX2<ResidualMode::SimdOnly>,
    EvpInnerProduct_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 512) {
            return EvpInnerProduct_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 1024) {
            if (dim == 512)
                return EvpInnerProduct_AVX512<ResidualMode::SimdOnly>{};
            else
                return EvpInnerProduct_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 1024;
            if (rem == 0)
                return EvpInnerProduct_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 512)
                return EvpInnerProduct_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 512)
                return EvpInnerProduct_AVX512<ResidualMode::DualTail>{};
            else
                return EvpInnerProduct_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 256) {
            return EvpInnerProduct_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 512) {
            if (dim == 256)
                return EvpInnerProduct_AVX2<ResidualMode::SimdOnly>{};
            else
                return EvpInnerProduct_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 512;
            if (rem == 0)
                return EvpInnerProduct_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 256)
                return EvpInnerProduct_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 256)
                return EvpInnerProduct_AVX2<ResidualMode::DualTail>{};
            else
                return EvpInnerProduct_AVX2<ResidualMode::Full>{};
        }
    }
#endif
    return EvpInnerProduct{};
}

}  // namespace deglib::distances::evp_ip
