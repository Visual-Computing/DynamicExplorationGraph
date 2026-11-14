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
            while (a + 31 < last) {
                __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m512i diff = _mm512_sub_epi16(_mm512_cvtepi8_epi16(q_raw), _mm512_cvtepi8_epi16(r_raw));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(diff, diff));
                a += 32;
                b += 32;
            }
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
            while (a + 15 < last) {
                __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                __m256i diff = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(diff, diff));
                a += 16;
                b += 16;
            }
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
};

template <ResidualMode Mode = ResidualMode::Full>
class L2Int8_AVX512_Optimal {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512_Optimal"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const int8_t* last = a + size;

        __m512i sum512_1 = _mm512_setzero_si512();
        __m512i sum512_2 = _mm512_setzero_si512();

        while (a + 63 < last) {
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m512i diff1 = _mm512_sub_epi16(_mm512_cvtepi8_epi16(v1), _mm512_cvtepi8_epi16(v2));
            sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(diff1, diff1));
            a += 32;
            b += 32;

            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m512i diff2 = _mm512_sub_epi16(_mm512_cvtepi8_epi16(v3), _mm512_cvtepi8_epi16(v4));
            sum512_2 = _mm512_add_epi32(sum512_2, _mm512_madd_epi16(diff2, diff2));
            a += 32;
            b += 32;
        }

        if (a + 31 < last) {
            __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m512i diff = _mm512_sub_epi16(_mm512_cvtepi8_epi16(q_raw), _mm512_cvtepi8_epi16(r_raw));
            sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(diff, diff));
            a += 32;
            b += 32;
        }

        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        int64_t result = int8_l2_hsum512(sum512);

        if constexpr (HasTail) {
            if ((last - a) >= 16) {
                __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                __m256i diff = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2));
                result += int8_l2_hsum256(_mm256_madd_epi16(diff, diff));
                a += 16;
                b += 16;
            }
            if ((last - a) >= 8) {
                __m128i v1 = _mm_loadu_si64(a);
                __m128i v2 = _mm_loadu_si64(b);
                __m128i diff = _mm_sub_epi16(_mm_cvtepi8_epi16(v1), _mm_cvtepi8_epi16(v2));
                __m128i prod = _mm_madd_epi16(diff, diff);
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(1, 0, 3, 2)));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(2, 3, 0, 1)));
                result += static_cast<int64_t>(_mm_cvtsi128_si32(prod));
                a += 8;
                b += 8;
            }
            while (a < last) {
                int32_t diff = int32_t(*a++) - int32_t(*b++);
                result += int64_t(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class L2Int8_AVX2_Optimal {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2_Optimal"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const int8_t* last = a + size;

        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();

        while (a + 31 < last) {
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
            __m256i diff1 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2));
            sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(diff1, diff1));
            a += 16;
            b += 16;

            __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
            __m256i diff2 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v3), _mm256_cvtepi8_epi16(v4));
            sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(diff2, diff2));
            a += 16;
            b += 16;
        }

        if (a + 15 < last) {
            __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
            __m256i diff = _mm256_sub_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2));
            sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(diff, diff));
            a += 16;
            b += 16;
        }

        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        int64_t result = int8_l2_hsum256(sum256);

        if constexpr (HasTail) {
            if ((last - a) >= 8) {
                __m128i v1 = _mm_loadu_si64(a);
                __m128i v2 = _mm_loadu_si64(b);
                __m128i diff = _mm_sub_epi16(_mm_cvtepi8_epi16(v1), _mm_cvtepi8_epi16(v2));
                __m128i prod = _mm_madd_epi16(diff, diff);
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(1, 0, 3, 2)));
                prod = _mm_add_epi32(prod, _mm_shuffle_epi32(prod, _MM_SHUFFLE(2, 3, 0, 1)));
                result += static_cast<int64_t>(_mm_cvtsi128_si32(prod));
                a += 8;
                b += 8;
            }
            while (a < last) {
                int32_t diff = int32_t(*a++) - int32_t(*b++);
                result += int64_t(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }
};

DEGLIB_TARGET_AVX2 inline int64_t int8_l2_hsum128(__m128i s) {
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(s));
}

DEGLIB_TARGET_AVX2 inline void l2_int8_s7_acc256(__m256i& sum, const int8_t* a, const int8_t* b, __m256i ones) {
    const __m256i xx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i yy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    const __m256i d = _mm256_abs_epi8(_mm256_sub_epi8(xx, yy));
    const __m256i prod = _mm256_maddubs_epi16(d, d);
    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod, ones));
}

DEGLIB_TARGET_AVX2 inline int64_t l2_int8_s7_acc128(const int8_t* a, const int8_t* b) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i xx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    const __m128i yy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    const __m128i d = _mm_abs_epi8(_mm_sub_epi8(xx, yy));
    const __m128i prod = _mm_maddubs_epi16(d, d);
    return int8_l2_hsum128(_mm_madd_epi16(prod, ones));
}

DEGLIB_TARGET_AVX2 inline int64_t l2_int8_s7_acc64(const int8_t* a, const int8_t* b) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i xx = _mm_loadu_si64(a);
    const __m128i yy = _mm_loadu_si64(b);
    const __m128i d = _mm_abs_epi8(_mm_sub_epi8(xx, yy));
    const __m128i prod = _mm_maddubs_epi16(d, d);
    return int8_l2_hsum128(_mm_madd_epi16(prod, ones));
}

DEGLIB_TARGET_AVX512 inline void l2_int8_s7_acc512(__m512i& sum, const int8_t* a, const int8_t* b, __m512i ones) {
    const __m512i xx = _mm512_loadu_si512(a);
    const __m512i yy = _mm512_loadu_si512(b);
    const __m512i d = _mm512_abs_epi8(_mm512_sub_epi8(xx, yy));
    const __m512i prod = _mm512_maddubs_epi16(d, d);
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(prod, ones));
}

template <ResidualMode Mode = ResidualMode::Full>
class L2Int8_AVX512_S7 {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512_S7"; }

    // Exact for S7 values in [0, 127]: the byte difference fits in int8.
    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = *static_cast<const size_t*>(qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const __m512i ones = _mm512_set1_epi16(1);

        __m512i sum0 = _mm512_setzero_si512();
        __m512i sum1 = _mm512_setzero_si512();

        size_t i = 0;
        for (; i + 128 <= dim; i += 128) {
            l2_int8_s7_acc512(sum0, a + i, b + i, ones);
            l2_int8_s7_acc512(sum1, a + i + 64, b + i + 64, ones);
        }
        if (i + 64 <= dim) {
            l2_int8_s7_acc512(sum0, a + i, b + i, ones);
            i += 64;
        }

        int64_t result = int8_l2_hsum512(_mm512_add_epi32(sum0, sum1));

        if constexpr (HasTail) {
            size_t rem = dim - i;
            if (rem >= 32) {
                __m256i tail256 = _mm256_setzero_si256();
                l2_int8_s7_acc256(tail256, a + i, b + i, _mm256_set1_epi16(1));
                result += int8_l2_hsum256(tail256);
                i += 32;
                rem -= 32;
            }
            if (rem >= 16) {
                result += l2_int8_s7_acc128(a + i, b + i);
                i += 16;
                rem -= 16;
            }
            if (rem >= 8) {
                result += l2_int8_s7_acc64(a + i, b + i);
                i += 8;
                rem -= 8;
            }
            for (; i < dim; ++i) {
                int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
                result += static_cast<int64_t>(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class L2Int8_AVX2_S7 {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2_S7"; }

    // Exact for S7 values in [0, 127]: the byte difference fits in int8.
    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = *static_cast<const size_t*>(qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const __m256i ones = _mm256_set1_epi16(1);

        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();

        size_t i = 0;
        for (; i + 64 <= dim; i += 64) {
            l2_int8_s7_acc256(sum0, a + i, b + i, ones);
            l2_int8_s7_acc256(sum1, a + i + 32, b + i + 32, ones);
        }
        if (i + 32 <= dim) {
            l2_int8_s7_acc256(sum0, a + i, b + i, ones);
            i += 32;
        }

        int64_t result = int8_l2_hsum256(_mm256_add_epi32(sum0, sum1));

        if constexpr (HasTail) {
            size_t rem = dim - i;
            if (rem >= 16) {
                result += l2_int8_s7_acc128(a + i, b + i);
                i += 16;
                rem -= 16;
            }
            if (rem >= 8) {
                result += l2_int8_s7_acc64(a + i, b + i);
                i += 8;
                rem -= 8;
            }
            for (; i < dim; ++i) {
                int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
                result += static_cast<int64_t>(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }
};

#endif

using DistanceVariant = std::variant<
    L2Int8
#if defined(DEGLIB_X86)
    ,
    L2Int8_AVX512_S7<ResidualMode::Full>,
    L2Int8_AVX512_S7<ResidualMode::SimdOnly>,
    L2Int8_AVX2_S7<ResidualMode::Full>,
    L2Int8_AVX2_S7<ResidualMode::SimdOnly>,
    L2Int8_AVX512_Optimal<ResidualMode::Full>,
    L2Int8_AVX512_Optimal<ResidualMode::SimdOnly>,
    L2Int8_AVX2_Optimal<ResidualMode::Full>,
    L2Int8_AVX2_Optimal<ResidualMode::SimdOnly>,
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
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
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
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
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
