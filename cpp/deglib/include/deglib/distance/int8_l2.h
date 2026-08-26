#pragma once

#include "deglib/distance/residual_mode.h"
#include "deglib/distance/int8.h"

#include <stdexcept>
#include <cstring>
#include <variant>

namespace deglib::distances::int8_l2 {

// ---------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------ Int8 L2 Dists ------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

// Scalar fallback — no SIMD required.
class L2Int8 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        int64_t result = 0;
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);

        size_t size = *((size_t*)qty_ptr);
        for (size_t i = 0; i < size; i++) {
            int32_t diff0 = int32_t(a[i]) - int32_t(b[i]);
            result += diff0 * diff0;
        }

        return float(result);
    }

    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        for (size_t i = 0; i < count; ++i) {
            dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
        }
    }
};

#if defined(DEGLIB_X86)
// -------------------------------------------------------------------
// L2Int8 SIMD implementations — process vectors with
// aligned SIMD portions plus scalar residuals for any unaligned tail.
// Separate classes per SIMD width so that compare() has zero
// runtime dispatch overhead — select_dist() chooses the class.

DEGLIB_TARGET_AVX2 inline static int64_t int8_l2_hsum256(__m256i s) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(sum128));
}

DEGLIB_TARGET_AVX512 inline static int64_t int8_l2_hsum512(__m512i s) { return static_cast<int64_t>(_mm512_reduce_add_epi32(s)); }

template <size_t BATCH_SIZE = 8>
inline static void process_batch_tail(
    const int8_t* query,
    const void* const* db,
    size_t offset,
    size_t dim,
    float* out_dists
) {
    const size_t tail_len = dim - offset;
    if (tail_len >= 8) {
        int64_t q64;
        std::memcpy(&q64, query + offset, sizeof(int64_t));
        const int8_t* q_bytes = reinterpret_cast<const int8_t*>(&q64);
        for (size_t j = 0; j < BATCH_SIZE; ++j) {
            const int8_t* db_ptr = static_cast<const int8_t*>(db[j]);
            int64_t db64;
            std::memcpy(&db64, db_ptr + offset, sizeof(int64_t));
            const int8_t* db_bytes = reinterpret_cast<const int8_t*>(&db64);
            int32_t tsum = 0;
            for (size_t k = 0; k < 8; ++k) {
                int32_t diff = int32_t(q_bytes[k]) - int32_t(db_bytes[k]);
                tsum += diff * diff;
            }
            out_dists[j] += static_cast<float>(tsum);
        }
        offset += 8;
    }
    for (size_t k = offset; k < dim; ++k) {
        const int32_t q_val = static_cast<int32_t>(query[k]);
        for (size_t j = 0; j < BATCH_SIZE; ++j) {
            const int8_t* db_ptr = static_cast<const int8_t*>(db[j]);
            int32_t diff = q_val - static_cast<int32_t>(db_ptr[k]);
            out_dists[j] += static_cast<float>(diff * diff);
        }
    }
}

template <ResidualMode Mode = ResidualMode::Full>
class L2Int8_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);

        const int8_t* last = a + size;

        __m512i sum512_1 = _mm512_setzero_si512();
        __m512i sum512_2 = _mm512_setzero_si512();
        if constexpr (HasDualSimd) {
            while (a + 63 < last) {
                __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m512i diff = _mm512_sub_epi16(_mm512_cvtepi8_epi16(v1), _mm512_cvtepi8_epi16(v2));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(diff, diff));
                a += 32;
                b += 32;
                __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m512i diff2 = _mm512_sub_epi16(_mm512_cvtepi8_epi16(v3), _mm512_cvtepi8_epi16(v4));
                sum512_2 = _mm512_add_epi32(sum512_2, _mm512_madd_epi16(diff2, diff2));
                a += 32;
                b += 32;
            }
        }

        if constexpr (HasSimd) {
            __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m512i diff = _mm512_sub_epi16(_mm512_cvtepi8_epi16(q_raw), _mm512_cvtepi8_epi16(r_raw));
            sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(diff, diff));
            a += 32;
            b += 32;
        }

        // Horizontal reduce of SIMD accumulators
        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(sum512), _mm512_extracti64x4_epi64(sum512, 1));
        __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));

        alignas(16) int sum_array[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
        int64_t result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            while (a < last) {
                int32_t diff = int32_t(*a++) - int32_t(*b++);
                result += int64_t(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const int8_t* query = static_cast<const int8_t*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512 {
            size_t offset = 0;
            alignas(64) __m512i s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm512_setzero_si512();
            }

            if constexpr (HasDualSimd) {
                const size_t nc32 = dim / 32;
                offset = nc32 * 32;

                for (size_t c = 0; c < nc32; ++c) {
                    size_t idx = c * 32;
                    __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&query[idx]));
                    __m512i q_vec = _mm512_cvtepi8_epi16(q_raw);

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const int8_t* db_ = static_cast<const int8_t*>(db[j]);
                        __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[idx]));
                        __m512i diff = _mm512_sub_epi16(q_vec, _mm512_cvtepi8_epi16(r_raw));
                        s[j] = _mm512_add_epi32(s[j], _mm512_madd_epi16(diff, diff));
                    }
                }
            }

            if constexpr (HasSimd) {
                __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&query[offset]));
                __m512i q_vec = _mm512_cvtepi8_epi16(q_raw);
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const int8_t* db_ = static_cast<const int8_t*>(db[j]);
                    __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[offset]));
                    __m512i diff = _mm512_sub_epi16(q_vec, _mm512_cvtepi8_epi16(r_raw));
                    s[j] = _mm512_add_epi32(s[j], _mm512_madd_epi16(diff, diff));
                }
                offset += 32;
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = static_cast<float>(int8_l2_hsum512(s[j]));
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
class L2Int8_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }
    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);

        const int8_t* last = a + size;

        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();
        if constexpr (HasDualSimd) {
            while (a + 31 < last) {
                __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                __m256i diff = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(diff, diff));
                a += 16;
                b += 16;
                __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                __m256i diff2 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v3), _mm256_cvtepi8_epi16(v4));
                sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(diff2, diff2));
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
            __m256i diff = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2));
            sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(diff, diff));
            a += 16;
            b += 16;
        }

        // Horizontal reduce of SIMD accumulators
        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum256, 0), _mm256_extracti128_si256(sum256, 1));

        alignas(16) int sum_array[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
        int64_t result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            while (a < last) {
                int32_t diff = int32_t(*a++) - int32_t(*b++);
                result += int64_t(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const int8_t* query = static_cast<const int8_t*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2 {
            size_t offset = 0;
            alignas(32) __m256i s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm256_setzero_si256();
            }

            if constexpr (HasDualSimd) {
                const size_t nc16 = dim / 16;
                offset = nc16 * 16;

                for (size_t c = 0; c < nc16; ++c) {
                    size_t idx = c * 16;
                    __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[idx]));
                    __m256i q_vec = _mm256_cvtepi8_epi16(q_raw);

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const int8_t* db_ = static_cast<const int8_t*>(db[j]);
                        __m128i r_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[idx]));
                        __m256i diff = _mm256_sub_epi16(q_vec, _mm256_cvtepi8_epi16(r_raw));
                        s[j] = _mm256_add_epi32(s[j], _mm256_madd_epi16(diff, diff));
                    }
                }
            }

            if constexpr (HasSimd) {
                __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[offset]));
                __m256i q_vec = _mm256_cvtepi8_epi16(q_raw);
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const int8_t* db_ = static_cast<const int8_t*>(db[j]);
                    __m128i r_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[offset]));
                    __m256i diff = _mm256_sub_epi16(q_vec, _mm256_cvtepi8_epi16(r_raw));
                    s[j] = _mm256_add_epi32(s[j], _mm256_madd_epi16(diff, diff));
                }
                offset += 16;
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = static_cast<float>(int8_l2_hsum256(s[j]));
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
    L2Int8
#if defined(DEGLIB_X86)
    ,
    L2Int8_AVX512<ResidualMode::Full>,
    L2Int8_AVX512<ResidualMode::DualPlusSimd>,
    L2Int8_AVX512<ResidualMode::DualTail>,
    L2Int8_AVX512<ResidualMode::DualOnly>,
    L2Int8_AVX512<ResidualMode::SimdTail>,
    L2Int8_AVX512<ResidualMode::SimdOnly>,
    L2Int8_AVX512<ResidualMode::TailOnly>,
    L2Int8_AVX2<ResidualMode::Full>,
    L2Int8_AVX2<ResidualMode::DualPlusSimd>,
    L2Int8_AVX2<ResidualMode::DualTail>,
    L2Int8_AVX2<ResidualMode::DualOnly>,
    L2Int8_AVX2<ResidualMode::SimdTail>,
    L2Int8_AVX2<ResidualMode::SimdOnly>,
    L2Int8_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512) {
        if (dim < 32) {
            return L2Int8_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 64) {
            if (dim == 32)
                return L2Int8_AVX512<ResidualMode::SimdOnly>{};
            else
                return L2Int8_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 64;
            if (rem == 0)
                return L2Int8_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 32)
                return L2Int8_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 32)
                return L2Int8_AVX512<ResidualMode::DualTail>{};
            else
                return L2Int8_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2) {
        if (dim < 16) {
            return L2Int8_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return L2Int8_AVX2<ResidualMode::SimdOnly>{};
            else
                return L2Int8_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 32;
            if (rem == 0)
                return L2Int8_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 16)
                return L2Int8_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 16)
                return L2Int8_AVX2<ResidualMode::DualTail>{};
            else
                return L2Int8_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return L2Int8{};
}

}  // namespace deglib::distances::int8_l2
