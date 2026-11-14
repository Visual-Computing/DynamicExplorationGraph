#pragma once

#include "deglib/config.h"
#include "deglib/distance/residual_mode.h"

#include <cstddef>
#include <cstdint>
#include <variant>

namespace deglib::distances::uint7_l2 {

// Scalar fallback for INT7 codes in [0, 127].
class L2UInt7 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = static_cast<size_t>(*static_cast<const int32_t*>(qty_ptr));
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);

        int64_t result = 0;
        for (size_t i = 0; i < dim; ++i) {
            const int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
            result += static_cast<int64_t>(diff) * diff;
        }
        return static_cast<float>(result);
    }
};

#if defined(DEGLIB_X86)

DEGLIB_TARGET_AVX2 inline static int64_t uint7_hsum256(__m256i x) {
    const __m128i sumh = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    const __m128i tmp = _mm_hadd_epi32(sumh, sumh);
    return static_cast<int64_t>(_mm_extract_epi32(tmp, 0)) + static_cast<int64_t>(_mm_extract_epi32(tmp, 1));
}

DEGLIB_TARGET_AVX512 inline static int64_t uint7_hsum512(__m512i s) { return static_cast<int64_t>(_mm512_reduce_add_epi32(s)); }

DEGLIB_TARGET_AVX2 inline static int64_t uint7_hsum128(__m128i s) {
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(s));
}

DEGLIB_TARGET_AVX2 inline static void l2_uint7_acc256(__m256i& sum, const int8_t* a, const int8_t* b, __m256i ones) {
    const __m256i xx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i yy = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    const __m256i d = _mm256_abs_epi8(_mm256_sub_epi8(xx, yy));
    const __m256i prod = _mm256_maddubs_epi16(d, d);
    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod, ones));
}

DEGLIB_TARGET_AVX512 inline static void l2_uint7_acc512(__m512i& sum, const int8_t* a, const int8_t* b, __m512i ones) {
    const __m512i xx = _mm512_loadu_si512(a);
    const __m512i yy = _mm512_loadu_si512(b);
    const __m512i d = _mm512_abs_epi8(_mm512_sub_epi8(xx, yy));
    const __m512i prod = _mm512_maddubs_epi16(d, d);
    sum = _mm512_add_epi32(sum, _mm512_madd_epi16(prod, ones));
}

DEGLIB_TARGET_AVX2 inline static int64_t l2_uint7_acc128(const int8_t* a, const int8_t* b) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i xx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    const __m128i yy = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    const __m128i d = _mm_abs_epi8(_mm_sub_epi8(xx, yy));
    const __m128i prod = _mm_maddubs_epi16(d, d);
    return uint7_hsum128(_mm_madd_epi16(prod, ones));
}

DEGLIB_TARGET_AVX2 inline static int64_t l2_uint7_acc64(const int8_t* a, const int8_t* b) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i xx = _mm_loadu_si64(a);
    const __m128i yy = _mm_loadu_si64(b);
    const __m128i d = _mm_abs_epi8(_mm_sub_epi8(xx, yy));
    const __m128i prod = _mm_maddubs_epi16(d, d);
    return uint7_hsum128(_mm_madd_epi16(prod, ones));
}

template <ResidualMode Mode = ResidualMode::Full>
class L2UInt7_AVX512 {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const int32_t dim = *static_cast<const int32_t*>(qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const __m512i ones = _mm512_set1_epi16(1);

        __m512i sum0 = _mm512_setzero_si512();
        __m512i sum1 = _mm512_setzero_si512();

        int32_t i = 0;
        for (; i + 128 <= dim; i += 128) {
            l2_uint7_acc512(sum0, a + i, b + i, ones);
            l2_uint7_acc512(sum1, a + i + 64, b + i + 64, ones);
        }
        if (i + 64 <= dim) {
            l2_uint7_acc512(sum0, a + i, b + i, ones);
            i += 64;
        }

        int64_t result = uint7_hsum512(_mm512_add_epi32(sum0, sum1));

        if constexpr (HasTail) {
            int32_t rem = dim - i;
            if (rem >= 32) {
                __m256i tail256 = _mm256_setzero_si256();
                l2_uint7_acc256(tail256, a + i, b + i, _mm256_set1_epi16(1));
                result += uint7_hsum256(tail256);
                i += 32;
                rem -= 32;
            }
            if (rem >= 16) {
                result += l2_uint7_acc128(a + i, b + i);
                i += 16;
                rem -= 16;
            }
            if (rem >= 8) {
                result += l2_uint7_acc64(a + i, b + i);
                i += 8;
                rem -= 8;
            }
            for (; i < dim; ++i) {
                const int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
                result += static_cast<int64_t>(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class L2UInt7_AVX2 {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const int32_t dim = *static_cast<const int32_t*>(qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const __m256i ones = _mm256_set1_epi16(1);

        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();

        int32_t i = 0;
        for (; i + 64 <= dim; i += 64) {
            l2_uint7_acc256(sum0, a + i, b + i, ones);
            l2_uint7_acc256(sum1, a + i + 32, b + i + 32, ones);
        }
        if (i + 32 <= dim) {
            l2_uint7_acc256(sum0, a + i, b + i, ones);
            i += 32;
        }

        int64_t result = uint7_hsum256(_mm256_add_epi32(sum0, sum1));

        if constexpr (HasTail) {
            int32_t rem = dim - i;
            if (rem >= 16) {
                result += l2_uint7_acc128(a + i, b + i);
                i += 16;
                rem -= 16;
            }
            if (rem >= 8) {
                result += l2_uint7_acc64(a + i, b + i);
                i += 8;
                rem -= 8;
            }
            for (; i < dim; ++i) {
                const int32_t diff = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
                result += static_cast<int64_t>(diff) * diff;
            }
        }

        return static_cast<float>(result);
    }
};

#endif

using DistanceVariant = std::variant<
    L2UInt7
#if defined(DEGLIB_X86)
    ,
    L2UInt7_AVX512<ResidualMode::Full>,
    L2UInt7_AVX512<ResidualMode::SimdOnly>,
    L2UInt7_AVX2<ResidualMode::Full>,
    L2UInt7_AVX2<ResidualMode::SimdOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 32) {
            return L2UInt7{};
        }
        const size_t rem = dim % 64;
        if (rem == 0) {
            return L2UInt7_AVX512<ResidualMode::SimdOnly>{};
        }
        return L2UInt7_AVX512<ResidualMode::Full>{};
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 16) {
            return L2UInt7{};
        }
        const size_t rem = dim % 32;
        if (rem == 0) {
            return L2UInt7_AVX2<ResidualMode::SimdOnly>{};
        }
        return L2UInt7_AVX2<ResidualMode::Full>{};
    }
#endif

    return L2UInt7{};
}

}  // namespace deglib::distances::uint7_l2
