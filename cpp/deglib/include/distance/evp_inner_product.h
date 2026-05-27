#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <config.h>

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif



namespace deglib {
namespace distances {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------- EVP Bits ------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

/**
 * Computes EvpBits similarity between two byte arrays.
 *
 * Each vector layout: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
 * qty_ptr points to the original float dimension (divisible by 8).
 *
 * Formula: (aa + bb + dim*2) - (cc + dd)
 *   aa = popcount(ones(a) AND ones(b))
 *   bb = popcount(neg(a) AND neg(b))
 *   cc = popcount(ones(a) AND neg(b))
 *   dd = popcount(ones(b) AND neg(a))
 */
class EvpInnerProduct {
public:
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);

    #if defined(USE_AVX512) && defined(__AVX512VPOPCNTDQ__)
        const float similarity = compare_avx512(pVect1v, pVect2v, qty_ptr);
    #elif defined(USE_AVX)
        const float similarity = compare_avx2(pVect1v, pVect2v, qty_ptr);
    #else
        const float similarity = compare_naive(pVect1v, pVect2v, qty_ptr);
    #endif

        const float max_similarity = 2.f * dim;
        return 1.f - (similarity / max_similarity);
    }

    inline static float compare_naive(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const std::byte* a = (const std::byte*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        uint32_t dim = *((uint32_t*)qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_a = a;
        const std::byte* negs_a = a + mask_bytes;
        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        uint32_t aa = 0, bb = 0, cc = 0, dd = 0;

        size_t i = 0;
        for (; i + sizeof(uint64_t) <= mask_bytes; i += sizeof(uint64_t)) {
            const uint64_t o1 = *reinterpret_cast<const uint64_t*>(&ones_a[i]);
            const uint64_t o2 = *reinterpret_cast<const uint64_t*>(&ones_b[i]);
            const uint64_t n1 = *reinterpret_cast<const uint64_t*>(&negs_a[i]);
            const uint64_t n2 = *reinterpret_cast<const uint64_t*>(&negs_b[i]);
            aa += std::popcount(o1 & o2);
            bb += std::popcount(n1 & n2);
            cc += std::popcount(o1 & n2);
            dd += std::popcount(o2 & n1);
        }

        for (; i < mask_bytes; ++i) {
            unsigned int b1 = static_cast<unsigned int>(static_cast<uint8_t>(ones_a[i]));
            unsigned int b2 = static_cast<unsigned int>(static_cast<uint8_t>(ones_b[i]));
            unsigned int n1 = static_cast<unsigned int>(static_cast<uint8_t>(negs_a[i]));
            unsigned int n2 = static_cast<unsigned int>(static_cast<uint8_t>(negs_b[i]));
            aa += std::popcount(b1 & b2);
            bb += std::popcount(n1 & n2);
            cc += std::popcount(b1 & n2);
            dd += std::popcount(b2 & n1);
        }

        return static_cast<float>(aa + bb + dim) - static_cast<float>(cc + dd);
    }

    inline static float compare_avx2(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const std::byte* a = (const std::byte*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        uint32_t dim = *((uint32_t*)qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_a = a;
        const std::byte* negs_a = a + mask_bytes;
        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m256i acc_aa = _mm256_setzero_si256();
        __m256i acc_bb = _mm256_setzero_si256();
        __m256i acc_cc = _mm256_setzero_si256();
        __m256i acc_dd = _mm256_setzero_si256();

        const __m256i lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        const __m256i low_mask = _mm256_set1_epi8(0x0F);
        const __m256i zero = _mm256_setzero_si256();

        const size_t block = 32;
        size_t i = 0;
        for (; i + block <= mask_bytes; i += block) {
            __m256i o1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_a[i]));
            __m256i o2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&ones_b[i]));
            __m256i n1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_a[i]));
            __m256i n2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&negs_b[i]));

            __m256i and_aa = _mm256_and_si256(o1, o2);
            __m256i and_bb = _mm256_and_si256(n1, n2);
            __m256i and_cc = _mm256_and_si256(o1, n2);
            __m256i and_dd = _mm256_and_si256(o2, n1);

            auto vector_popcnt = [&](__m256i vec) {
                __m256i lo = _mm256_and_si256(vec, low_mask);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
                __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
                __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
                return _mm256_add_epi8(popcnt1, popcnt2);
            };

            acc_aa = _mm256_add_epi64(acc_aa, _mm256_sad_epu8(vector_popcnt(and_aa), zero));
            acc_bb = _mm256_add_epi64(acc_bb, _mm256_sad_epu8(vector_popcnt(and_bb), zero));
            acc_cc = _mm256_add_epi64(acc_cc, _mm256_sad_epu8(vector_popcnt(and_cc), zero));
            acc_dd = _mm256_add_epi64(acc_dd, _mm256_sad_epu8(vector_popcnt(and_dd), zero));
        }

        alignas(32) int64_t arr[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), acc_aa);
        uint64_t aa = arr[0] + arr[1] + arr[2] + arr[3];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), acc_bb);
        uint64_t bb = arr[0] + arr[1] + arr[2] + arr[3];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), acc_cc);
        uint64_t cc = arr[0] + arr[1] + arr[2] + arr[3];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(arr), acc_dd);
        uint64_t dd = arr[0] + arr[1] + arr[2] + arr[3];

        for (; i < mask_bytes; ++i) {
            unsigned int b1 = static_cast<unsigned int>(static_cast<uint8_t>(ones_a[i]));
            unsigned int b2 = static_cast<unsigned int>(static_cast<uint8_t>(ones_b[i]));
            unsigned int n1 = static_cast<unsigned int>(static_cast<uint8_t>(negs_a[i]));
            unsigned int n2 = static_cast<unsigned int>(static_cast<uint8_t>(negs_b[i]));
            aa += std::popcount(b1 & b2);
            bb += std::popcount(n1 & n2);
            cc += std::popcount(b1 & n2);
            dd += std::popcount(b2 & n1);
        }

        return static_cast<float>(aa + bb + dim) - static_cast<float>(cc + dd);
    }

    inline static float compare_avx512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const std::byte* a = (const std::byte*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        uint32_t dim = *((uint32_t*)qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_a = a;
        const std::byte* negs_a = a + mask_bytes;
        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m512i acc_aa = _mm512_setzero_si512();
        __m512i acc_bb = _mm512_setzero_si512();
        __m512i acc_cc = _mm512_setzero_si512();
        __m512i acc_dd = _mm512_setzero_si512();

        const size_t block = 64;
        size_t i = 0;
        for (; i + block <= mask_bytes; i += block) {
            __m512i o1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_a[i]));
            __m512i o2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&ones_b[i]));
            __m512i n1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_a[i]));
            __m512i n2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&negs_b[i]));

            acc_aa = _mm512_add_epi64(acc_aa, _mm512_popcnt_epi64(_mm512_and_si512(o1, o2)));
            acc_bb = _mm512_add_epi64(acc_bb, _mm512_popcnt_epi64(_mm512_and_si512(n1, n2)));
            acc_cc = _mm512_add_epi64(acc_cc, _mm512_popcnt_epi64(_mm512_and_si512(o1, n2)));
            acc_dd = _mm512_add_epi64(acc_dd, _mm512_popcnt_epi64(_mm512_and_si512(o2, n1)));
        }

        alignas(64) int64_t arr[8];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(arr), acc_aa);
        uint64_t aa = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(arr), acc_bb);
        uint64_t bb = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(arr), acc_cc);
        uint64_t cc = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(arr), acc_dd);
        uint64_t dd = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

        for (; i < mask_bytes; ++i) {
            unsigned int b1 = static_cast<unsigned int>(static_cast<uint8_t>(ones_a[i]));
            unsigned int b2 = static_cast<unsigned int>(static_cast<uint8_t>(ones_b[i]));
            unsigned int n1 = static_cast<unsigned int>(static_cast<uint8_t>(negs_a[i]));
            unsigned int n2 = static_cast<unsigned int>(static_cast<uint8_t>(negs_b[i]));
            aa += std::popcount(b1 & b2);
            bb += std::popcount(n1 & n2);
            cc += std::popcount(b1 & n2);
            dd += std::popcount(b2 & n1);
        }

        return static_cast<float>(aa + bb + dim) - static_cast<float>(cc + dd);
    }

    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;
        const std::byte* query = static_cast<const std::byte*>(query_ptr);

    #if defined(USE_AVX512) && defined(__AVX512VPOPCNTDQ__)
        while (count >= 8) {
            batch8_avx512(query, db_arr, mask_bytes, dim, dists);
            db_arr += 8; dists += 8; count -= 8;
        }
        while (count >= 4) {
            batch4_avx512(query, db_arr, mask_bytes, dim, dists);
            db_arr += 4; dists += 4; count -= 4;
        }
    #elif defined(USE_AVX)
        while (count >= 8) {
            batch8_avx2(query, db_arr, mask_bytes, dim, dists);
            db_arr += 8; dists += 8; count -= 8;
        }
        while (count >= 4) {
            batch4_avx2(query, db_arr, mask_bytes, dim, dists);
            db_arr += 4; dists += 4; count -= 4;
        }
    #endif

        for (size_t i = 0; i < count; ++i) {
            dists[i] = compare(query, db_arr[i], qty_ptr);
        }
    }

private:
    template <size_t BatchN>
    static void batch_scalar_impl(const std::byte* query, const void* const* db_arr, size_t mask_bytes, uint32_t dim, float* dists) {
        uint64_t aa[BatchN] = {0}, bb[BatchN] = {0}, cc[BatchN] = {0}, dd[BatchN] = {0};

        size_t i = 0;
        for (; i + 8 <= mask_bytes; i += 8) {
            const uint64_t oq = *reinterpret_cast<const uint64_t*>(&query[i]);
            const uint64_t nq = *reinterpret_cast<const uint64_t*>(&query[i + mask_bytes]);
            for (size_t j = 0; j < BatchN; ++j) {
                const std::byte* db = static_cast<const std::byte*>(db_arr[j]);
                const uint64_t od = *reinterpret_cast<const uint64_t*>(&db[i]);
                const uint64_t nd = *reinterpret_cast<const uint64_t*>(&db[i + mask_bytes]);
                aa[j] += std::popcount(oq & od);
                bb[j] += std::popcount(nq & nd);
                cc[j] += std::popcount(oq & nd);
                dd[j] += std::popcount(od & nq);
            }
        }

        for (; i < mask_bytes; ++i) {
            const uint8_t oq = static_cast<uint8_t>(query[i]);
            const uint8_t nq = static_cast<uint8_t>(query[i + mask_bytes]);
            for (size_t j = 0; j < BatchN; ++j) {
                const std::byte* db = static_cast<const std::byte*>(db_arr[j]);
                const uint8_t od = static_cast<uint8_t>(db[i]);
                const uint8_t nd = static_cast<uint8_t>(db[i + mask_bytes]);
                aa[j] += std::popcount(static_cast<unsigned int>(oq & od));
                bb[j] += std::popcount(static_cast<unsigned int>(nq & nd));
                cc[j] += std::popcount(static_cast<unsigned int>(oq & nd));
                dd[j] += std::popcount(static_cast<unsigned int>(od & nq));
            }
        }

        for (size_t j = 0; j < BatchN; ++j) {
            const float sim = static_cast<float>(aa[j] + bb[j] + dim) - static_cast<float>(cc[j] + dd[j]);
            dists[j] = 1.0f - sim / (2.0f * static_cast<float>(dim));
        }
    }

    static void batch8_avx2(const std::byte* query, const void* const* db_arr, size_t mask_bytes, uint32_t dim, float* dists) {
        batch_scalar_impl<8>(query, db_arr, mask_bytes, dim, dists);
    }

    static void batch4_avx2(const std::byte* query, const void* const* db_arr, size_t mask_bytes, uint32_t dim, float* dists) {
        batch_scalar_impl<4>(query, db_arr, mask_bytes, dim, dists);
    }

    static void batch8_avx512(const std::byte* query, const void* const* db_arr, size_t mask_bytes, uint32_t dim, float* dists) {
        batch_scalar_impl<8>(query, db_arr, mask_bytes, dim, dists);
    }

    static void batch4_avx512(const std::byte* query, const void* const* db_arr, size_t mask_bytes, uint32_t dim, float* dists) {
        batch_scalar_impl<4>(query, db_arr, mask_bytes, dim, dists);
    }
};

using EvpBitsSimilarity = EvpInnerProduct;

} // namespace distances
} // namespace deglib
