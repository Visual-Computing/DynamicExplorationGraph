#pragma once

#include "deglib/distance/int8.h"
#include "deglib/distance/residual_mode.h"
#include "deglib/distance/tail_mask.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace deglib::distances::int8_ip {

// ---------------------------------------------------------------------------------------------------------------------
// ------------------------------------------- Int8 InnerProduct Dists -------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

// Scalar fallback — no SIMD required.
class InnerProductInt8 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        int64_t result = 0;
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);

        size_t size = *((size_t*)qty_ptr);
        for (size_t i = 0; i < size; i++) {
            result += int64_t(a[i]) * int64_t(b[i]);
        }

        return -float(result);
    }
};

#if defined(DEGLIB_X86)

DEGLIB_TARGET_AVX512 inline static __m512i int8_ip_sign_epi8_512(__m512i a, __m512i b) {
    const __m512i zero = _mm512_setzero_si512();
    const __mmask64 blt0 = _mm512_movepi8_mask(b);
    const __mmask64 ble0 = _mm512_cmple_epi8_mask(b, zero);
    const __m512i a_blt0 = _mm512_mask_mov_epi8(zero, blt0, a);
    return _mm512_mask_sub_epi8(a, ble0, zero, a_blt0);
}

DEGLIB_TARGET_AVX2 inline static int64_t int8_ip_hsum256(__m256i s) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(sum128));
}

DEGLIB_TARGET_AVX512 inline static int64_t int8_ip_hsum512(__m512i s) {
    #if defined(DEGLIB_X86)
    return static_cast<int64_t>(_mm512_reduce_add_epi32(s));
    #else
    __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(s), _mm512_extracti32x8_epi32(s, 1));
    return int8_ip_hsum256(sum256);
    #endif
}
// ---------------------------------------------------------------------------------------------------------------------
// AVX512-VNNI (vpdpbusd with signed transform)
// ---------------------------------------------------------------------------------------------------------------------
template <ResidualMode Mode = ResidualMode::Full>
class InnerProductInt8_AVX512_VNNI {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512_VNNI"; }

    DEGLIB_TARGET_AVX512_VNNI inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const int8_t* last = a + size;

        __m512i sum512_1 = _mm512_setzero_si512();
        __m512i sum512_2 = _mm512_setzero_si512();

        while (a + 127 < last) {
            __m512i raw_a1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
            __m512i raw_b1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
            __m512i abs_a1 = _mm512_abs_epi8(raw_a1);
            __m512i sgn_b1 = int8_ip_sign_epi8_512(raw_b1, raw_a1);
            sum512_1 = _mm512_dpbusd_epi32(sum512_1, abs_a1, sgn_b1);
            a += 64;
            b += 64;

            __m512i raw_a2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
            __m512i raw_b2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
            __m512i abs_a2 = _mm512_abs_epi8(raw_a2);
            __m512i sgn_b2 = int8_ip_sign_epi8_512(raw_b2, raw_a2);
            sum512_2 = _mm512_dpbusd_epi32(sum512_2, abs_a2, sgn_b2);
            a += 64;
            b += 64;
        }

        if (a + 63 < last) {
            __m512i raw_a = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
            __m512i raw_b = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
            __m512i abs_a = _mm512_abs_epi8(raw_a);
            __m512i sgn_b = int8_ip_sign_epi8_512(raw_b, raw_a);
            sum512_1 = _mm512_dpbusd_epi32(sum512_1, abs_a, sgn_b);
            a += 64;
            b += 64;
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __mmask64 tail_mask = tail_mask::mask64(static_cast<size_t>(rem));
                __m512i raw_a = _mm512_maskz_loadu_epi8(tail_mask, a);
                __m512i raw_b = _mm512_maskz_loadu_epi8(tail_mask, b);
                __m512i abs_a = _mm512_abs_epi8(raw_a);
                __m512i sgn_b = int8_ip_sign_epi8_512(raw_b, raw_a);
                sum512_1 = _mm512_dpbusd_epi32(sum512_1, abs_a, sgn_b);
            }
        }

        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        int64_t result = int8_ip_hsum512(sum512);

        return -float(result);
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// AVX512 (Standard AVX-512 BW/DQ/F)
// ---------------------------------------------------------------------------------------------------------------------
template <ResidualMode Mode = ResidualMode::Full>
class InnerProductInt8_AVX512 {
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
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(_mm512_cvtepi8_epi16(v1), _mm512_cvtepi8_epi16(v2)));
                a += 32;
                b += 32;
                __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                sum512_2 = _mm512_add_epi32(sum512_2, _mm512_madd_epi16(_mm512_cvtepi8_epi16(v3), _mm512_cvtepi8_epi16(v4)));
                a += 32;
                b += 32;
            }
        }
        if constexpr (HasSimd) {
            while (a + 31 < last) {
                __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(_mm512_cvtepi8_epi16(q_raw), _mm512_cvtepi8_epi16(r_raw)));
                a += 32;
                b += 32;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __mmask32 mask = tail_mask::mask32(static_cast<size_t>(rem));
                __m256i v1 = _mm256_maskz_loadu_epi8(mask, a);
                __m256i v2 = _mm256_maskz_loadu_epi8(mask, b);
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(_mm512_cvtepi8_epi16(v1), _mm512_cvtepi8_epi16(v2)));
            }
        }

        // Horizontal reduce
        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        int64_t result = int8_ip_hsum512(sum512);

        return -float(result);
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// AVX2-VNNI (AVX_VNNI: _mm256_dpbusd_epi32 with sign_epi8)
// ---------------------------------------------------------------------------------------------------------------------
template <ResidualMode Mode = ResidualMode::Full>
class InnerProductInt8_AVX2_VNNI {
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2_VNNI"; }

    DEGLIB_TARGET_AVX2_VNNI inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        size_t size = *((size_t*)qty_ptr);
        const int8_t* a = static_cast<const int8_t*>(pVect1v);
        const int8_t* b = static_cast<const int8_t*>(pVect2v);
        const int8_t* last = a + size;

        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();

        while (a + 63 < last) {
            __m256i raw_a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i raw_b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m256i abs_a1 = _mm256_sign_epi8(raw_a1, raw_a1);
            __m256i sgn_b1 = _mm256_sign_epi8(raw_b1, raw_a1);
            sum256_1 = _mm256_dpbusd_epi32(sum256_1, abs_a1, sgn_b1);
            a += 32;
            b += 32;

            __m256i raw_a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i raw_b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m256i abs_a2 = _mm256_sign_epi8(raw_a2, raw_a2);
            __m256i sgn_b2 = _mm256_sign_epi8(raw_b2, raw_a2);
            sum256_2 = _mm256_dpbusd_epi32(sum256_2, abs_a2, sgn_b2);
            a += 32;
            b += 32;
        }

        if (a + 31 < last) {
            __m256i raw_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
            __m256i raw_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
            __m256i abs_a = _mm256_sign_epi8(raw_a, raw_a);
            __m256i sgn_b = _mm256_sign_epi8(raw_b, raw_a);
            sum256_1 = _mm256_dpbusd_epi32(sum256_1, abs_a, sgn_b);
            a += 32;
            b += 32;
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __m256i raw_a = tail_mask::load_tail_bytes_32(a, static_cast<size_t>(rem), size);
                const __m256i raw_b = tail_mask::load_tail_bytes_32(b, static_cast<size_t>(rem), size);
                const __m256i abs_a = _mm256_sign_epi8(raw_a, raw_a);
                const __m256i sgn_b = _mm256_sign_epi8(raw_b, raw_a);
                sum256_1 = _mm256_dpbusd_epi32(sum256_1, abs_a, sgn_b);
            }
        }

        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        int64_t result = int8_ip_hsum256(sum256);

        return -float(result);
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// AVX2 (Standard AVX2)
// ---------------------------------------------------------------------------------------------------------------------
template <ResidualMode Mode = ResidualMode::Full>
class InnerProductInt8_AVX2 {
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
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2)));
                a += 16;
                b += 16;
                __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(_mm256_cvtepi8_epi16(v3), _mm256_cvtepi8_epi16(v4)));
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            while (a + 15 < last) {
                __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2)));
                a += 16;
                b += 16;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __m128i v1 = tail_mask::load_tail_bytes_16(a, static_cast<size_t>(rem), size);
                const __m128i v2 = tail_mask::load_tail_bytes_16(b, static_cast<size_t>(rem), size);
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(_mm256_cvtepi8_epi16(v1), _mm256_cvtepi8_epi16(v2)));
            }
        }

        // Horizontal reduce
        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        int64_t result = int8_ip_hsum256(sum256);

        return -float(result);
    }
};

#endif

using DistanceVariant = std::variant<
    InnerProductInt8
#if defined(DEGLIB_X86)
    ,
    InnerProductInt8_AVX512_VNNI<ResidualMode::Full>,
    InnerProductInt8_AVX512_VNNI<ResidualMode::SimdOnly>,

    InnerProductInt8_AVX512<ResidualMode::Full>,
    InnerProductInt8_AVX512<ResidualMode::DualPlusSimd>,
    InnerProductInt8_AVX512<ResidualMode::DualTail>,
    InnerProductInt8_AVX512<ResidualMode::DualOnly>,
    InnerProductInt8_AVX512<ResidualMode::SimdTail>,
    InnerProductInt8_AVX512<ResidualMode::SimdOnly>,
    InnerProductInt8_AVX512<ResidualMode::TailOnly>,

    InnerProductInt8_AVX2_VNNI<ResidualMode::Full>,
    InnerProductInt8_AVX2_VNNI<ResidualMode::SimdOnly>,

    InnerProductInt8_AVX2<ResidualMode::Full>,
    InnerProductInt8_AVX2<ResidualMode::DualPlusSimd>,
    InnerProductInt8_AVX2<ResidualMode::DualTail>,
    InnerProductInt8_AVX2<ResidualMode::DualOnly>,
    InnerProductInt8_AVX2<ResidualMode::SimdTail>,
    InnerProductInt8_AVX2<ResidualMode::SimdOnly>,
    InnerProductInt8_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim % 64 == 0)
            return InnerProductInt8_AVX512_VNNI<ResidualMode::SimdOnly>{};
        else
            return InnerProductInt8_AVX512_VNNI<ResidualMode::Full>{};
    } else if (target == deglib::cpu::InstructionSet::AVX512) {
        if (dim < 32) {
            return InnerProductInt8_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 64) {
            if (dim == 32)
                return InnerProductInt8_AVX512<ResidualMode::SimdOnly>{};
            else
                return InnerProductInt8_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 64;
            if (rem == 0)
                return InnerProductInt8_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 32)
                return InnerProductInt8_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 32)
                return InnerProductInt8_AVX512<ResidualMode::DualTail>{};
            else
                return InnerProductInt8_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim % 32 == 0)
            return InnerProductInt8_AVX2_VNNI<ResidualMode::SimdOnly>{};
        else
            return InnerProductInt8_AVX2_VNNI<ResidualMode::Full>{};
    } else if (target == deglib::cpu::InstructionSet::AVX2) {
        if (dim < 16) {
            return InnerProductInt8_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return InnerProductInt8_AVX2<ResidualMode::SimdOnly>{};
            else
                return InnerProductInt8_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 32;
            if (rem == 0)
                return InnerProductInt8_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 16)
                return InnerProductInt8_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 16)
                return InnerProductInt8_AVX2<ResidualMode::DualTail>{};
            else
                return InnerProductInt8_AVX2<ResidualMode::Full>{};
        }
    }
#else
    if (instruction != deglib::cpu::InstructionSet::Auto && instruction != deglib::cpu::InstructionSet::Scalar) {
        throw std::runtime_error("Requested SIMD instruction set is not supported on this platform");
    }
#endif

    return InnerProductInt8{};
}

}  // namespace deglib::distances::int8_ip
