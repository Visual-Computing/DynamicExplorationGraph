#pragma once

#include "deglib/distance/residual_mode.h"
#include "deglib/distance/uint8.h"

#include <stdexcept>
#include <cstring>
#include <variant>

namespace deglib::distances::uint8_ip {

// ---------------------------------------------------------------------------------------------------------------------
// ------------------------------------------ Uint8 InnerProduct Dists --------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

// Scalar fallback — no SIMD required.
class InnerProductUint8 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        int64_t result = 0;
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        size_t size = *((size_t*)qty_ptr);
        for (size_t i = 0; i < size; i++) {
            result += int64_t(a[i]) * int64_t(b[i]);
        }

        return -float(result);
    }

    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        for (size_t i = 0; i < count; ++i) {
            dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
        }
    }
};

#if defined(DEGLIB_X86)

DEGLIB_TARGET_AVX2 inline static int64_t uint8_ip_hsum256(__m256i s) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(sum128));
}

DEGLIB_TARGET_AVX512 inline static int64_t uint8_ip_hsum512(__m512i s) {
    #if defined(DEGLIB_X86)
    return static_cast<int64_t>(_mm512_reduce_add_epi32(s));
    #else
    __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(s), _mm512_extracti32x8_epi32(s, 1));
    return uint8_ip_hsum256(sum256);
    #endif
}

template <size_t BATCH_SIZE = 8>
inline static void process_batch_tail(
    const uint8_t* query,
    const void* const* db,
    size_t offset,
    size_t dim,
    float* out_dists
) {
    const size_t tail_len = dim - offset;
    if (tail_len >= 8) {
        uint64_t q64;
        std::memcpy(&q64, query + offset, sizeof(uint64_t));
        const uint8_t* q_bytes = reinterpret_cast<const uint8_t*>(&q64);
        for (size_t j = 0; j < BATCH_SIZE; ++j) {
            const uint8_t* db_ptr = static_cast<const uint8_t*>(db[j]);
            uint64_t db64;
            std::memcpy(&db64, db_ptr + offset, sizeof(uint64_t));
            const uint8_t* db_bytes = reinterpret_cast<const uint8_t*>(&db64);
            int32_t tsum = 0;
            for (size_t k = 0; k < 8; ++k) {
                tsum += int32_t(q_bytes[k]) * int32_t(db_bytes[k]);
            }
            out_dists[j] -= static_cast<float>(tsum);
        }
        offset += 8;
    }
    for (size_t k = offset; k < dim; ++k) {
        const int32_t q_val = static_cast<int32_t>(query[k]);
        for (size_t j = 0; j < BATCH_SIZE; ++j) {
            const uint8_t* db_ptr = static_cast<const uint8_t*>(db[j]);
            out_dists[j] -= static_cast<float>(q_val * static_cast<int32_t>(db_ptr[k]));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// AVX512-VNNI (vpdpbusd with signed transform)
//
// Mathematical Background for Unsigned x Unsigned dot product via vpdpbusd:
// Hardware vpdpbusd natively computes unsigned int8 (u) * signed int8 (s):
//   dot(u, s) = sum(u_i * s_i)
// For uint8 dot product sum(a_i * b_i):
//   1. a_i is already unsigned in [0, 255].
//   2. Map b_i (uint8) to signed int8 range [-128, 127] by subtracting 128 (via XOR 0x80):
//      s_b_i = b_i - 128
//   3. Execute vpdpbusd(a, s_b):
//      sum(a_i * (b_i - 128)) = sum(a_i * b_i) - 128 * sum(a_i)
//   4. Add the query bias compensation (q_correction = 128 * sum(a_i)):
//      sum(a_i * b_i) = vpdpbusd_result + 128 * sum(a_i)
//
// In compare_batch(), q_correction depends solely on the query vector and is
// precomputed once across SIMD chunks, avoiding redundant calculations per DB vector.
// ---------------------------------------------------------------------------------------------------------------------
template <ResidualMode Mode = ResidualMode::Full>
class InnerProductUint8_AVX512_VNNI {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512_VNNI"; }

    DEGLIB_TARGET_AVX512_VNNI inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const uint8_t* last = a + size;

        const __m512i xor_mask = _mm512_set1_epi8(static_cast<char>(0x80));
        __m512i sum512_1 = _mm512_setzero_si512();
        __m512i sum512_2 = _mm512_setzero_si512();
        __m512i q_comp_1 = _mm512_setzero_si512();
        __m512i q_comp_2 = _mm512_setzero_si512();

        if constexpr (HasDualSimd) {
            while (a + 127 < last) {
                __m512i raw_a1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
                __m512i raw_b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
                __m512i s_b1 = _mm512_xor_si512(raw_b1, xor_mask);
                sum512_1 = _mm512_dpbusd_epi32(sum512_1, raw_a1, s_b1);
                q_comp_1 = _mm512_add_epi32(q_comp_1, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(raw_a1)), _mm512_set1_epi16(1)));
                q_comp_1 = _mm512_add_epi32(q_comp_1, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(raw_a1, 1)), _mm512_set1_epi16(1)));

                a += 64;
                b += 64;

                __m512i raw_a2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
                __m512i raw_b2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
                __m512i s_b2 = _mm512_xor_si512(raw_b2, xor_mask);
                sum512_2 = _mm512_dpbusd_epi32(sum512_2, raw_a2, s_b2);
                q_comp_2 = _mm512_add_epi32(q_comp_2, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(raw_a2)), _mm512_set1_epi16(1)));
                q_comp_2 = _mm512_add_epi32(q_comp_2, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(raw_a2, 1)), _mm512_set1_epi16(1)));

                a += 64;
                b += 64;
            }
        }
        if constexpr (HasSimd) {
            __m512i raw_a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
            __m512i raw_b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
            __m512i s_b = _mm512_xor_si512(raw_b, xor_mask);
            sum512_1 = _mm512_dpbusd_epi32(sum512_1, raw_a, s_b);
            q_comp_1 = _mm512_add_epi32(q_comp_1, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(raw_a)), _mm512_set1_epi16(1)));
            q_comp_1 = _mm512_add_epi32(q_comp_1, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(raw_a, 1)), _mm512_set1_epi16(1)));

            a += 64;
            b += 64;
        }

        // Horizontal reduce
        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        __m512i q_comp = _mm512_add_epi32(q_comp_1, q_comp_2);
        int64_t result = uint8_ip_hsum512(sum512) + (uint8_ip_hsum512(q_comp) * 128);

        if constexpr (HasTail) {
            const size_t rem = last - a;
            if (rem >= 32) {
                __m256i raw_a256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i raw_b256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i prod1 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(raw_a256)), _mm256_cvtepu8_epi16(_mm256_castsi256_si128(raw_b256)));
                __m256i prod2 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(raw_a256, 1)), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(raw_b256, 1)));
                result += uint8_ip_hsum256(_mm256_add_epi32(prod1, prod2));
                a += 32;
                b += 32;
            }
            if ((last - a) >= 16) {
                __m128i raw_a128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i raw_b128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                __m256i prod = _mm256_madd_epi16(_mm256_cvtepu8_epi16(raw_a128), _mm256_cvtepu8_epi16(raw_b128));
                result += uint8_ip_hsum256(prod);
                a += 16;
                b += 16;
            }
            if ((last - a) >= 8) {
                __m128i raw_a64 = _mm_loadu_si64(a);
                __m128i raw_b64 = _mm_loadu_si64(b);
                __m128i prod = _mm_madd_epi16(_mm_cvtepu8_epi16(raw_a64), _mm_cvtepu8_epi16(raw_b64));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(1, 0, 3, 2)));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(2, 3, 0, 1)));
                result += static_cast<int64_t>(_mm_cvtsi128_si32(prod));
                a += 8;
                b += 8;
            }
            while (a < last) {
                result += int64_t(*a++) * int64_t(*b++);
            }
        }

        return -float(result);
    }

    DEGLIB_TARGET_AVX512_VNNI inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const uint8_t* query = static_cast<const uint8_t*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        // Precalculate q_correction once for the entire batch of DB vectors
        int64_t q_correction = 0;
        if constexpr (HasDualSimd || HasSimd) {
            __m512i q_comp = _mm512_setzero_si512();
            const size_t simd_limit = dim >= 64 ? (dim - 63) : 0;
            for (size_t offset = 0; offset < simd_limit; offset += 64) {
                __m512i q_raw = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(query + offset));
                q_comp = _mm512_add_epi32(q_comp, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(q_raw)), _mm512_set1_epi16(1)));
                q_comp = _mm512_add_epi32(q_comp, _mm512_madd_epi16(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(q_raw, 1)), _mm512_set1_epi16(1)));
            }
            q_correction = uint8_ip_hsum512(q_comp) * 128;
        }

        auto batch_impl = [query, dim, q_correction](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512_VNNI {
            const __m512i xor_mask = _mm512_set1_epi8(static_cast<char>(0x80));
            __m512i acc[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                acc[j] = _mm512_setzero_si512();
            }

            size_t offset = 0;
            if constexpr (HasDualSimd || HasSimd) {
                const size_t simd_limit = dim >= 64 ? (dim - 63) : 0;
                for (; offset < simd_limit; offset += 64) {
                    __m512i q_raw = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(query + offset));
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const uint8_t* db_ptr = static_cast<const uint8_t*>(db[j]);
                        __m512i r_raw = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(db_ptr + offset));
                        __m512i s_r = _mm512_xor_si512(r_raw, xor_mask);
                        acc[j] = _mm512_dpbusd_epi32(acc[j], q_raw, s_r);
                    }
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                int64_t total = uint8_ip_hsum512(acc[j]) + q_correction;
                out_dists[j] = -static_cast<float>(total);
            }

            if constexpr (HasTail) {
                process_batch_tail<BATCH_SIZE>(query, db, offset, dim, out_dists);
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
class InnerProductUint8_AVX2_VNNI {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2_VNNI"; }

    DEGLIB_TARGET_AVX2_VNNI inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const uint8_t* last = a + size;

        const __m256i xor_mask = _mm256_set1_epi8(static_cast<char>(0x80));
        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();
        __m256i q_comp_1 = _mm256_setzero_si256();
        __m256i q_comp_2 = _mm256_setzero_si256();

        if constexpr (HasDualSimd) {
            while (a + 63 < last) {
                __m256i raw_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i raw_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i s_b1 = _mm256_xor_si256(raw_b1, xor_mask);
                sum256_1 = _mm256_dpbusd_epi32(sum256_1, raw_a1, s_b1);
                q_comp_1 = _mm256_add_epi32(q_comp_1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(raw_a1)), _mm256_set1_epi16(1)));
                q_comp_1 = _mm256_add_epi32(q_comp_1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(raw_a1, 1)), _mm256_set1_epi16(1)));

                a += 32;
                b += 32;

                __m256i raw_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i raw_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i s_b2 = _mm256_xor_si256(raw_b2, xor_mask);
                sum256_2 = _mm256_dpbusd_epi32(sum256_2, raw_a2, s_b2);
                q_comp_2 = _mm256_add_epi32(q_comp_2, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(raw_a2)), _mm256_set1_epi16(1)));
                q_comp_2 = _mm256_add_epi32(q_comp_2, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(raw_a2, 1)), _mm256_set1_epi16(1)));

                a += 32;
                b += 32;
            }
        }
        if constexpr (HasSimd) {
            __m256i raw_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i raw_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m256i s_b = _mm256_xor_si256(raw_b, xor_mask);
            sum256_1 = _mm256_dpbusd_epi32(sum256_1, raw_a, s_b);
            q_comp_1 = _mm256_add_epi32(q_comp_1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(raw_a)), _mm256_set1_epi16(1)));
            q_comp_1 = _mm256_add_epi32(q_comp_1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(raw_a, 1)), _mm256_set1_epi16(1)));

            a += 32;
            b += 32;
        }

        // Horizontal reduce
        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        __m256i q_comp = _mm256_add_epi32(q_comp_1, q_comp_2);
        int64_t result = uint8_ip_hsum256(sum256) + (uint8_ip_hsum256(q_comp) * 128);

        if constexpr (HasTail) {
            if ((last - a) >= 16) {
                __m128i raw_a128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i raw_b128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                __m256i prod = _mm256_madd_epi16(_mm256_cvtepu8_epi16(raw_a128), _mm256_cvtepu8_epi16(raw_b128));
                result += uint8_ip_hsum256(prod);
                a += 16;
                b += 16;
            }
            if ((last - a) >= 8) {
                __m128i raw_a64 = _mm_loadu_si64(a);
                __m128i raw_b64 = _mm_loadu_si64(b);
                __m128i prod = _mm_madd_epi16(_mm_cvtepu8_epi16(raw_a64), _mm_cvtepu8_epi16(raw_b64));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(1, 0, 3, 2)));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(2, 3, 0, 1)));
                result += static_cast<int64_t>(_mm_cvtsi128_si32(prod));
                a += 8;
                b += 8;
            }
            while (a < last) {
                result += int64_t(*a++) * int64_t(*b++);
            }
        }

        return -float(result);
    }

    DEGLIB_TARGET_AVX2_VNNI inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const uint8_t* query = static_cast<const uint8_t*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        // Precalculate q_correction once for the entire batch of DB vectors
        int64_t q_correction = 0;
        if constexpr (HasDualSimd || HasSimd) {
            __m256i q_comp = _mm256_setzero_si256();
            const size_t simd_limit = dim >= 32 ? (dim - 31) : 0;
            for (size_t offset = 0; offset < simd_limit; offset += 32) {
                __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + offset));
                q_comp = _mm256_add_epi32(q_comp, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(q_raw)), _mm256_set1_epi16(1)));
                q_comp = _mm256_add_epi32(q_comp, _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(q_raw, 1)), _mm256_set1_epi16(1)));
            }
            q_correction = uint8_ip_hsum256(q_comp) * 128;
        }

        auto batch_impl = [query, dim, q_correction](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2_VNNI {
            const __m256i xor_mask = _mm256_set1_epi8(static_cast<char>(0x80));
            __m256i acc[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                acc[j] = _mm256_setzero_si256();
            }

            size_t offset = 0;
            if constexpr (HasDualSimd || HasSimd) {
                const size_t simd_limit = dim >= 32 ? (dim - 31) : 0;
                for (; offset < simd_limit; offset += 32) {
                    __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + offset));
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const uint8_t* db_ptr = static_cast<const uint8_t*>(db[j]);
                        __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db_ptr + offset));
                        __m256i s_r = _mm256_xor_si256(r_raw, xor_mask);
                        acc[j] = _mm256_dpbusd_epi32(acc[j], q_raw, s_r);
                    }
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                int64_t total = uint8_ip_hsum256(acc[j]) + q_correction;
                out_dists[j] = -static_cast<float>(total);
            }

            if constexpr (HasTail) {
                process_batch_tail<BATCH_SIZE>(query, db, offset, dim, out_dists);
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
class InnerProductUint8_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const unsigned char* a = (const unsigned char*)pVect1v;
        const unsigned char* b = (const unsigned char*)pVect2v;

        const unsigned char* last = a + size;

        __m512i sum512_1 = _mm512_setzero_si512();
        __m512i sum512_2 = _mm512_setzero_si512();
        if constexpr (HasDualSimd) {
            while (a + 63 < last) {
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(_mm512_cvtepu8_epi16(v1), _mm512_cvtepu8_epi16(v2)));
                a += 32;
                b += 32;
                __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                sum512_2 = _mm512_add_epi32(sum512_2, _mm512_madd_epi16(_mm512_cvtepu8_epi16(v3), _mm512_cvtepu8_epi16(v4)));
                a += 32;
                b += 32;
            }
        }
        if constexpr (HasSimd) {
            __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(_mm512_cvtepu8_epi16(q_raw), _mm512_cvtepu8_epi16(r_raw)));
            a += 32;
            b += 32;
        }

        // Horizontal reduce
        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        int64_t result = uint8_ip_hsum512(sum512);

        if constexpr (HasTail) {
            const size_t rem = last - a;
            if (rem >= 32) {
                __m256i raw_a256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i raw_b256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i prod1 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(raw_a256)), _mm256_cvtepu8_epi16(_mm256_castsi256_si128(raw_b256)));
                __m256i prod2 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(_mm256_extracti128_si256(raw_a256, 1)), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(raw_b256, 1)));
                result += uint8_ip_hsum256(_mm256_add_epi32(prod1, prod2));
                a += 32;
                b += 32;
            }
            if ((last - a) >= 16) {
                __m128i raw_a128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i raw_b128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                __m256i prod = _mm256_madd_epi16(_mm256_cvtepu8_epi16(raw_a128), _mm256_cvtepu8_epi16(raw_b128));
                result += uint8_ip_hsum256(prod);
                a += 16;
                b += 16;
            }
            if ((last - a) >= 8) {
                __m128i raw_a64 = _mm_loadu_si64(a);
                __m128i raw_b64 = _mm_loadu_si64(b);
                __m128i prod = _mm_madd_epi16(_mm_cvtepu8_epi16(raw_a64), _mm_cvtepu8_epi16(raw_b64));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(1, 0, 3, 2)));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(2, 3, 0, 1)));
                result += static_cast<int64_t>(_mm_cvtsi128_si32(prod));
                a += 8;
                b += 8;
            }
            while (a < last) {
                result += int64_t(*a++) * int64_t(*b++);
            }
        }

        return -float(result);
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const unsigned char* query = static_cast<const unsigned char*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512 {
            __m512i acc[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                acc[j] = _mm512_setzero_si512();
            }

            size_t offset = 0;
            if constexpr (HasDualSimd || HasSimd) {
                const size_t simd_limit = dim >= 32 ? (dim - 31) : 0;
                for (; offset < simd_limit; offset += 32) {
                    __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + offset));
                    __m512i q_vec = _mm512_cvtepu8_epi16(q_raw);
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const unsigned char* db_ptr = static_cast<const unsigned char*>(db[j]);
                        __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db_ptr + offset));
                        acc[j] = _mm512_add_epi32(acc[j], _mm512_madd_epi16(q_vec, _mm512_cvtepu8_epi16(r_raw)));
                    }
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = -static_cast<float>(uint8_ip_hsum512(acc[j]));
            }

            if constexpr (HasTail) {
                process_batch_tail<BATCH_SIZE>(query, db, offset, dim, out_dists);
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
class InnerProductUint8_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const unsigned char* a = (const unsigned char*)pVect1v;
        const unsigned char* b = (const unsigned char*)pVect2v;

        const unsigned char* last = a + size;

        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();
        if constexpr (HasDualSimd) {
            while (a + 31 < last) {
                __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(v1), _mm256_cvtepu8_epi16(v2)));
                a += 16;
                b += 16;
                __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(_mm256_cvtepu8_epi16(v3), _mm256_cvtepu8_epi16(v4)));
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
            sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(_mm256_cvtepu8_epi16(v1), _mm256_cvtepu8_epi16(v2)));
            a += 16;
            b += 16;
        }

        // Horizontal reduce
        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        int64_t result = uint8_ip_hsum256(sum256);

        if constexpr (HasTail) {
            if ((last - a) >= 8) {
                __m128i raw_a64 = _mm_loadu_si64(a);
                __m128i raw_b64 = _mm_loadu_si64(b);
                __m128i prod = _mm_madd_epi16(_mm_cvtepu8_epi16(raw_a64), _mm_cvtepu8_epi16(raw_b64));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(1, 0, 3, 2)));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(2, 3, 0, 1)));
                result += static_cast<int64_t>(_mm_cvtsi128_si32(prod));
                a += 8;
                b += 8;
            }
            while (a < last) {
                result += int64_t(*a++) * int64_t(*b++);
            }
        }

        return -float(result);
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const unsigned char* query = static_cast<const unsigned char*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2 {
            __m256i acc[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                acc[j] = _mm256_setzero_si256();
            }

            size_t offset = 0;
            if constexpr (HasDualSimd || HasSimd) {
                const size_t simd_limit = dim >= 16 ? (dim - 15) : 0;
                for (; offset < simd_limit; offset += 16) {
                    __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(query + offset));
                    __m256i q_vec = _mm256_cvtepu8_epi16(q_raw);
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const unsigned char* db_ptr = static_cast<const unsigned char*>(db[j]);
                        __m128i r_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(db_ptr + offset));
                        acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(q_vec, _mm256_cvtepu8_epi16(r_raw)));
                    }
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = -static_cast<float>(uint8_ip_hsum256(acc[j]));
            }

            if constexpr (HasTail) {
                process_batch_tail<BATCH_SIZE>(query, db, offset, dim, out_dists);
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
    InnerProductUint8
#if defined(DEGLIB_X86)
    ,
    InnerProductUint8_AVX512_VNNI<ResidualMode::Full>,
    InnerProductUint8_AVX512_VNNI<ResidualMode::DualPlusSimd>,
    InnerProductUint8_AVX512_VNNI<ResidualMode::DualTail>,
    InnerProductUint8_AVX512_VNNI<ResidualMode::DualOnly>,
    InnerProductUint8_AVX512_VNNI<ResidualMode::SimdTail>,
    InnerProductUint8_AVX512_VNNI<ResidualMode::SimdOnly>,
    InnerProductUint8_AVX512_VNNI<ResidualMode::TailOnly>,
    InnerProductUint8_AVX512<ResidualMode::Full>,
    InnerProductUint8_AVX512<ResidualMode::DualPlusSimd>,
    InnerProductUint8_AVX512<ResidualMode::DualTail>,
    InnerProductUint8_AVX512<ResidualMode::DualOnly>,
    InnerProductUint8_AVX512<ResidualMode::SimdTail>,
    InnerProductUint8_AVX512<ResidualMode::SimdOnly>,
    InnerProductUint8_AVX512<ResidualMode::TailOnly>,
    InnerProductUint8_AVX2_VNNI<ResidualMode::Full>,
    InnerProductUint8_AVX2_VNNI<ResidualMode::DualPlusSimd>,
    InnerProductUint8_AVX2_VNNI<ResidualMode::DualTail>,
    InnerProductUint8_AVX2_VNNI<ResidualMode::DualOnly>,
    InnerProductUint8_AVX2_VNNI<ResidualMode::SimdTail>,
    InnerProductUint8_AVX2_VNNI<ResidualMode::SimdOnly>,
    InnerProductUint8_AVX2_VNNI<ResidualMode::TailOnly>,
    InnerProductUint8_AVX2<ResidualMode::Full>,
    InnerProductUint8_AVX2<ResidualMode::DualPlusSimd>,
    InnerProductUint8_AVX2<ResidualMode::DualTail>,
    InnerProductUint8_AVX2<ResidualMode::DualOnly>,
    InnerProductUint8_AVX2<ResidualMode::SimdTail>,
    InnerProductUint8_AVX2<ResidualMode::SimdOnly>,
    InnerProductUint8_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 64) {
            return InnerProductUint8_AVX512_VNNI<ResidualMode::TailOnly>{};
        } else if (dim < 128) {
            if (dim == 64)
                return InnerProductUint8_AVX512_VNNI<ResidualMode::SimdOnly>{};
            else
                return InnerProductUint8_AVX512_VNNI<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 128;
            if (rem == 0)
                return InnerProductUint8_AVX512_VNNI<ResidualMode::DualOnly>{};
            else if (rem == 64)
                return InnerProductUint8_AVX512_VNNI<ResidualMode::DualPlusSimd>{};
            else if (rem < 64)
                return InnerProductUint8_AVX512_VNNI<ResidualMode::DualTail>{};
            else
                return InnerProductUint8_AVX512_VNNI<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX512) {
        if (dim < 32) {
            return InnerProductUint8_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 64) {
            if (dim == 32)
                return InnerProductUint8_AVX512<ResidualMode::SimdOnly>{};
            else
                return InnerProductUint8_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 64;
            if (rem == 0)
                return InnerProductUint8_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 32)
                return InnerProductUint8_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 32)
                return InnerProductUint8_AVX512<ResidualMode::DualTail>{};
            else
                return InnerProductUint8_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 32) {
            return InnerProductUint8_AVX2_VNNI<ResidualMode::TailOnly>{};
        } else if (dim < 64) {
            if (dim == 32)
                return InnerProductUint8_AVX2_VNNI<ResidualMode::SimdOnly>{};
            else
                return InnerProductUint8_AVX2_VNNI<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 64;
            if (rem == 0)
                return InnerProductUint8_AVX2_VNNI<ResidualMode::DualOnly>{};
            else if (rem == 32)
                return InnerProductUint8_AVX2_VNNI<ResidualMode::DualPlusSimd>{};
            else if (rem < 32)
                return InnerProductUint8_AVX2_VNNI<ResidualMode::DualTail>{};
            else
                return InnerProductUint8_AVX2_VNNI<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2) {
        if (dim < 16) {
            return InnerProductUint8_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return InnerProductUint8_AVX2<ResidualMode::SimdOnly>{};
            else
                return InnerProductUint8_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 32;
            if (rem == 0)
                return InnerProductUint8_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 16)
                return InnerProductUint8_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 16)
                return InnerProductUint8_AVX2<ResidualMode::DualTail>{};
            else
                return InnerProductUint8_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return InnerProductUint8{};
}
}  // namespace deglib::distances::uint8_ip
