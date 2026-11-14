#pragma once

#include "deglib/distance/residual_mode.h"
#include "deglib/distance/uint4.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace deglib::distances::uint4_ip {

// ---------------------------------------------------------------------------------------------------------------------
// ------------------------------------------ Uint4 InnerProduct Dists -------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

// Scalar fallback — no SIMD required.
class InnerProductUint4 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        int64_t result = 0;
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const size_t dim = *((size_t*)qty_ptr);
        const size_t num_bytes = dim / 2;

        for (size_t i = 0; i < num_bytes; ++i) {
            uint8_t byte_a = a[i];
            uint8_t byte_b = b[i];

            int32_t a0 = int32_t(byte_a & 0x0F);
            int32_t a1 = int32_t((byte_a >> 4) & 0x0F);
            int32_t b0 = int32_t(byte_b & 0x0F);
            int32_t b1 = int32_t((byte_b >> 4) & 0x0F);

            result += a0 * b0 + a1 * b1;
        }

        if (dim & 1) {
            uint8_t byte_a = a[num_bytes];
            uint8_t byte_b = b[num_bytes];
            int32_t a0 = int32_t(byte_a & 0x0F);
            int32_t b0 = int32_t(byte_b & 0x0F);
            result += a0 * b0;
        }

        return -static_cast<float>(result);
    }
};

#if defined(DEGLIB_X86)

DEGLIB_TARGET_AVX2 inline static int64_t uint4_ip_hsum256(__m256i s) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(sum128));
}

DEGLIB_TARGET_AVX512 inline static int64_t uint4_ip_hsum512(__m512i s) {
    return static_cast<int64_t>(_mm512_reduce_add_epi32(s));
}

template <ResidualMode Mode = ResidualMode::Full>
class InnerProductUint4_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = *((size_t*)qty_ptr);
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const size_t num_bytes = dim / 2;
        const uint8_t* last_full_byte = a + num_bytes;

        __m512i sum512_1 = _mm512_setzero_si512();
        __m512i sum512_2 = _mm512_setzero_si512();
        const __m512i mask = _mm512_set1_epi8(0x0F);
        const __m512i ones = _mm512_set1_epi16(1);

        if constexpr (HasDualSimd) {
            while (a + 127 < last_full_byte) {
                __m512i xx1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
                __m512i yy1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
                __m512i xx1_lo = _mm512_and_si512(xx1, mask);
                __m512i xx1_hi = _mm512_and_si512(_mm512_srli_epi16(xx1, 4), mask);
                __m512i yy1_lo = _mm512_and_si512(yy1, mask);
                __m512i yy1_hi = _mm512_and_si512(_mm512_srli_epi16(yy1, 4), mask);

                // Both operands in [0, 15] <= 127, so signed multiplication with _mm512_maddubs_epi16 is exact!
                __m512i prod1_lo = _mm512_maddubs_epi16(xx1_lo, yy1_lo);
                __m512i prod1_hi = _mm512_maddubs_epi16(xx1_hi, yy1_hi);
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_lo, ones));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_hi, ones));

                a += 64;
                b += 64;

                __m512i xx2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
                __m512i yy2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
                __m512i xx2_lo = _mm512_and_si512(xx2, mask);
                __m512i xx2_hi = _mm512_and_si512(_mm512_srli_epi16(xx2, 4), mask);
                __m512i yy2_lo = _mm512_and_si512(yy2, mask);
                __m512i yy2_hi = _mm512_and_si512(_mm512_srli_epi16(yy2, 4), mask);

                __m512i prod2_lo = _mm512_maddubs_epi16(xx2_lo, yy2_lo);
                __m512i prod2_hi = _mm512_maddubs_epi16(xx2_hi, yy2_hi);
                sum512_2 = _mm512_add_epi32(sum512_2, _mm512_madd_epi16(prod2_lo, ones));
                sum512_2 = _mm512_add_epi32(sum512_2, _mm512_madd_epi16(prod2_hi, ones));

                a += 64;
                b += 64;
            }
        }

        if constexpr (HasSimd) {
            while (a + 63 < last_full_byte) {
                __m512i xx1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
                __m512i yy1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
                __m512i xx1_lo = _mm512_and_si512(xx1, mask);
                __m512i xx1_hi = _mm512_and_si512(_mm512_srli_epi16(xx1, 4), mask);
                __m512i yy1_lo = _mm512_and_si512(yy1, mask);
                __m512i yy1_hi = _mm512_and_si512(_mm512_srli_epi16(yy1, 4), mask);

                __m512i prod1_lo = _mm512_maddubs_epi16(xx1_lo, yy1_lo);
                __m512i prod1_hi = _mm512_maddubs_epi16(xx1_hi, yy1_hi);
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_lo, ones));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_hi, ones));

                a += 64;
                b += 64;
            }
        }

        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        int64_t result = uint4_ip_hsum512(sum512);

        if constexpr (HasTail) {
            while (a < last_full_byte) {
                uint8_t byte_a = *a++;
                uint8_t byte_b = *b++;
                int32_t a0 = int32_t(byte_a & 0x0F);
                int32_t a1 = int32_t((byte_a >> 4) & 0x0F);
                int32_t b0 = int32_t(byte_b & 0x0F);
                int32_t b1 = int32_t((byte_b >> 4) & 0x0F);
                result += a0 * b0 + a1 * b1;
            }
            if (dim & 1) {
                uint8_t byte_a = *a;
                uint8_t byte_b = *b;
                int32_t a0 = int32_t(byte_a & 0x0F);
                int32_t b0 = int32_t(byte_b & 0x0F);
                result += a0 * b0;
            }
        } else if (dim & 1) {
            uint8_t byte_a = *a;
            uint8_t byte_b = *b;
            int32_t a0 = int32_t(byte_a & 0x0F);
            int32_t b0 = int32_t(byte_b & 0x0F);
            result += a0 * b0;
        }

        return -static_cast<float>(result);
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class InnerProductUint4_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = *((size_t*)qty_ptr);
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const size_t num_bytes = dim / 2;
        const uint8_t* last_full_byte = a + num_bytes;

        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();
        const __m256i mask = _mm256_set1_epi8(0x0F);
        const __m256i ones = _mm256_set1_epi16(1);

        if constexpr (HasDualSimd) {
            while (a + 63 < last_full_byte) {
                __m256i xx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i yy1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i xx1_lo = _mm256_and_si256(xx1, mask);
                __m256i xx1_hi = _mm256_and_si256(_mm256_srli_epi16(xx1, 4), mask);
                __m256i yy1_lo = _mm256_and_si256(yy1, mask);
                __m256i yy1_hi = _mm256_and_si256(_mm256_srli_epi16(yy1, 4), mask);

                __m256i prod1_lo = _mm256_maddubs_epi16(xx1_lo, yy1_lo);
                __m256i prod1_hi = _mm256_maddubs_epi16(xx1_hi, yy1_hi);
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_lo, ones));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_hi, ones));

                a += 32;
                b += 32;

                __m256i xx2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i yy2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i xx2_lo = _mm256_and_si256(xx2, mask);
                __m256i xx2_hi = _mm256_and_si256(_mm256_srli_epi16(xx2, 4), mask);
                __m256i yy2_lo = _mm256_and_si256(yy2, mask);
                __m256i yy2_hi = _mm256_and_si256(_mm256_srli_epi16(yy2, 4), mask);

                __m256i prod2_lo = _mm256_maddubs_epi16(xx2_lo, yy2_lo);
                __m256i prod2_hi = _mm256_maddubs_epi16(xx2_hi, yy2_hi);
                sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(prod2_lo, ones));
                sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(prod2_hi, ones));

                a += 32;
                b += 32;
            }
        }

        if constexpr (HasSimd) {
            while (a + 31 < last_full_byte) {
                __m256i xx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i yy1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i xx1_lo = _mm256_and_si256(xx1, mask);
                __m256i xx1_hi = _mm256_and_si256(_mm256_srli_epi16(xx1, 4), mask);
                __m256i yy1_lo = _mm256_and_si256(yy1, mask);
                __m256i yy1_hi = _mm256_and_si256(_mm256_srli_epi16(yy1, 4), mask);

                __m256i prod1_lo = _mm256_maddubs_epi16(xx1_lo, yy1_lo);
                __m256i prod1_hi = _mm256_maddubs_epi16(xx1_hi, yy1_hi);
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_lo, ones));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_hi, ones));

                a += 32;
                b += 32;
            }
        }

        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        int64_t result = uint4_ip_hsum256(sum256);

        if constexpr (HasTail) {
            while (a < last_full_byte) {
                uint8_t byte_a = *a++;
                uint8_t byte_b = *b++;
                int32_t a0 = int32_t(byte_a & 0x0F);
                int32_t a1 = int32_t((byte_a >> 4) & 0x0F);
                int32_t b0 = int32_t(byte_b & 0x0F);
                int32_t b1 = int32_t((byte_b >> 4) & 0x0F);
                result += a0 * b0 + a1 * b1;
            }
            if (dim & 1) {
                uint8_t byte_a = *a;
                uint8_t byte_b = *b;
                int32_t a0 = int32_t(byte_a & 0x0F);
                int32_t b0 = int32_t(byte_b & 0x0F);
                result += a0 * b0;
            }
        } else if (dim & 1) {
            uint8_t byte_a = *a;
            uint8_t byte_b = *b;
            int32_t a0 = int32_t(byte_a & 0x0F);
            int32_t b0 = int32_t(byte_b & 0x0F);
            result += a0 * b0;
        }

        return -static_cast<float>(result);
    }
};

#endif  // DEGLIB_X86

using DistanceVariant = std::variant<
    InnerProductUint4
#if defined(DEGLIB_X86)
    ,
    InnerProductUint4_AVX512<ResidualMode::Full>,
    InnerProductUint4_AVX512<ResidualMode::DualPlusSimd>,
    InnerProductUint4_AVX512<ResidualMode::DualTail>,
    InnerProductUint4_AVX512<ResidualMode::DualOnly>,
    InnerProductUint4_AVX512<ResidualMode::SimdTail>,
    InnerProductUint4_AVX512<ResidualMode::SimdOnly>,
    InnerProductUint4_AVX512<ResidualMode::TailOnly>,
    InnerProductUint4_AVX2<ResidualMode::Full>,
    InnerProductUint4_AVX2<ResidualMode::DualPlusSimd>,
    InnerProductUint4_AVX2<ResidualMode::DualTail>,
    InnerProductUint4_AVX2<ResidualMode::DualOnly>,
    InnerProductUint4_AVX2<ResidualMode::SimdTail>,
    InnerProductUint4_AVX2<ResidualMode::SimdOnly>,
    InnerProductUint4_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 128) {
            return InnerProductUint4_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 256) {
            if (dim == 128)
                return InnerProductUint4_AVX512<ResidualMode::SimdOnly>{};
            else
                return InnerProductUint4_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 256;
            if (rem == 0)
                return InnerProductUint4_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 128)
                return InnerProductUint4_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 128)
                return InnerProductUint4_AVX512<ResidualMode::DualTail>{};
            else
                return InnerProductUint4_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 64) {
            return InnerProductUint4_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 128) {
            if (dim == 64)
                return InnerProductUint4_AVX2<ResidualMode::SimdOnly>{};
            else
                return InnerProductUint4_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 128;
            if (rem == 0)
                return InnerProductUint4_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 64)
                return InnerProductUint4_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 64)
                return InnerProductUint4_AVX2<ResidualMode::DualTail>{};
            else
                return InnerProductUint4_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return InnerProductUint4{};
}

}  // namespace deglib::distances::uint4_ip

