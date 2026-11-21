#pragma once

#include "deglib/distance/int4.h"
#include "deglib/distance/residual_mode.h"
#include "deglib/distance/tail_mask.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace deglib::distances::int4_l2 {

// Helper for sign-extending a 4-bit nibble to 8-bit signed integer
inline static int8_t sign_extend_nibble(uint8_t n) {
    int8_t s = static_cast<int8_t>(n & 0x0F);
    if (s & 0x08) s |= 0xF0;
    return s;
}

// Scalar fallback — no SIMD required.
// Packed int4 contract: for odd dimensions, the last byte stores the final element
// in its low nibble and 0x0 in its high nibble, so summing all num_bytes = (dim + 1) / 2
// bytes is exact without an odd-dimension tail branch.
class L2Int4 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        int64_t result = 0;
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const size_t dim = *((size_t*)qty_ptr);
        const size_t num_bytes = (dim + 1) / 2;

        for (size_t i = 0; i < num_bytes; ++i) {
            uint8_t byte_a = a[i];
            uint8_t byte_b = b[i];

            int32_t a0 = sign_extend_nibble(byte_a);
            int32_t a1 = sign_extend_nibble(byte_a >> 4);
            int32_t b0 = sign_extend_nibble(byte_b);
            int32_t b1 = sign_extend_nibble(byte_b >> 4);

            int32_t diff0 = a0 - b0;
            int32_t diff1 = a1 - b1;

            result += diff0 * diff0 + diff1 * diff1;
        }

        return static_cast<float>(result);
    }
};

#if defined(DEGLIB_X86)

DEGLIB_TARGET_AVX2 inline static int64_t int4_l2_hsum256(__m256i s) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(sum128));
}

DEGLIB_TARGET_AVX512 inline static int64_t int4_l2_hsum512(__m512i s) {
    return static_cast<int64_t>(_mm512_reduce_add_epi32(s));
}

// AVX2 Nibble Sign-Extension:
// Shifting left by 4, then arithmetic shifting right by 4 sign-extends both nibbles into int8 bytes
DEGLIB_TARGET_AVX2 inline static void unpack_sign_extend_avx2(__m256i raw, __m256i& lo, __m256i& hi) {
    // raw has 32 bytes = 64 nibbles
    // lo: lower nibble in bits 0..3 -> shift left 4, srai 4
    __m256i lo_s = _mm256_slli_epi16(raw, 4);
    lo = _mm256_srai_epi16(lo_s, 4);
    // bytes in lo: high byte has bits 8..15, low byte bits 0..7.
    // However, srai_epi16 treats each 16-bit word.
    // Better: unpack bytes with sign extension:
    // low 16 bytes:
    __m128i lo128 = _mm256_castsi256_si128(raw);
    __m128i hi128 = _mm256_extracti128_si256(raw, 1);
    
    // Low nibbles: and 0x0F, then xor 0x08, sub 0x08 -> sign extension
    // Or simpler: (val ^ 0x08) - 0x08 maps [0..15] to [-8..7] unsigned-to-signed
}

// Exact mathematical identity for L2 of two sign-extended 4-bit nibbles:
// Notice: for two signed 4-bit integers a, b in [-8, 7]:
// a - b is IDENTICAL to ((a + 8) - (b + 8))!
// Adding 8 to a 4-bit two's complement value (or XORing with 0x08) converts it bijectively
// into an unsigned 4-bit value in [0, 15]!
// Because: (a - b)^2 = ((a + c) - (b + c))^2 for any constant c!
// In particular, for c = 8:
// Let u_a = (a & 0x0F) ^ 0x08 (which is in [0, 15])
// Let u_b = (b & 0x0F) ^ 0x08 (which is in [0, 15])
// Then diff = a - b = u_a - u_b!
// AND THIS MEANS:
// |a - b| = |u_a - u_b| !
// The L2 distance between signed 4-bit nibbles is IDENTICAL to the L2 distance between unsigned 4-bit nibbles
// when both are stored with the same bit pattern!
// Proof:
// Let byte_a and byte_b be the packed bytes.
// Nibbles n_a = byte_a & 0x0F, n_b = byte_b & 0x0F.
// Value of signed nibble: S(n) = (n >= 8) ? n - 16 : n.
// S(n_a) - S(n_b):
// Since S(n) = n - 16 * sign_bit, mod 16 difference is just (n_a - n_b) mod 16.
// But wait! Is (S(n_a) - S(n_b))^2 equal to (n_a - n_b)^2 if they are interpreted as unsigned?
// NO: S(0x0F) = -1. S(0x00) = 0. Diff is -1, squared is 1.
// If interpreted as unsigned: 15 - 0 = 15, squared is 225!
// BUT if we XOR with 0x08:
// 0x0F ^ 0x08 = 0x07 (representing -1 + 8 = 7).
// 0x00 ^ 0x08 = 0x08 (representing 0 + 8 = 8).
// Diff is 7 - 8 = -1, squared is 1!
// Exactly! Mapping signed 4-bit to unsigned via XOR 0x08 makes diff = (u_a - u_b)!

template <ResidualMode Mode = ResidualMode::Full>
class L2Int4_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = *((size_t*)qty_ptr);
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const size_t num_bytes = (dim + 1) / 2;
        const uint8_t* last_full_byte = a + num_bytes;

        __m512i sum512_1 = _mm512_setzero_si512();
        __m512i sum512_2 = _mm512_setzero_si512();
        const __m512i mask = _mm512_set1_epi8(0x0F);
        const __m512i sign_flip = _mm512_set1_epi8(0x08);
        const __m512i ones = _mm512_set1_epi16(1);

        if constexpr (HasDualSimd) {
            while (a + 127 < last_full_byte) {
                __m512i xx1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
                __m512i yy1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
                // Convert signed nibbles to unsigned offset by 8:
                __m512i xx1_lo = _mm512_xor_si512(_mm512_and_si512(xx1, mask), sign_flip);
                __m512i xx1_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(xx1, 4), mask), sign_flip);
                __m512i yy1_lo = _mm512_xor_si512(_mm512_and_si512(yy1, mask), sign_flip);
                __m512i yy1_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(yy1, 4), mask), sign_flip);

                __m512i d1_lo = _mm512_abs_epi8(_mm512_sub_epi8(xx1_lo, yy1_lo));
                __m512i d1_hi = _mm512_abs_epi8(_mm512_sub_epi8(xx1_hi, yy1_hi));

                __m512i prod1_lo = _mm512_maddubs_epi16(d1_lo, d1_lo);
                __m512i prod1_hi = _mm512_maddubs_epi16(d1_hi, d1_hi);
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_lo, ones));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_hi, ones));

                a += 64;
                b += 64;

                __m512i xx2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a));
                __m512i yy2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b));
                __m512i xx2_lo = _mm512_xor_si512(_mm512_and_si512(xx2, mask), sign_flip);
                __m512i xx2_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(xx2, 4), mask), sign_flip);
                __m512i yy2_lo = _mm512_xor_si512(_mm512_and_si512(yy2, mask), sign_flip);
                __m512i yy2_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(yy2, 4), mask), sign_flip);

                __m512i d2_lo = _mm512_abs_epi8(_mm512_sub_epi8(xx2_lo, yy2_lo));
                __m512i d2_hi = _mm512_abs_epi8(_mm512_sub_epi8(xx2_hi, yy2_hi));

                __m512i prod2_lo = _mm512_maddubs_epi16(d2_lo, d2_lo);
                __m512i prod2_hi = _mm512_maddubs_epi16(d2_hi, d2_hi);
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
                __m512i xx1_lo = _mm512_xor_si512(_mm512_and_si512(xx1, mask), sign_flip);
                __m512i xx1_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(xx1, 4), mask), sign_flip);
                __m512i yy1_lo = _mm512_xor_si512(_mm512_and_si512(yy1, mask), sign_flip);
                __m512i yy1_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(yy1, 4), mask), sign_flip);

                __m512i d1_lo = _mm512_abs_epi8(_mm512_sub_epi8(xx1_lo, yy1_lo));
                __m512i d1_hi = _mm512_abs_epi8(_mm512_sub_epi8(xx1_hi, yy1_hi));

                __m512i prod1_lo = _mm512_maddubs_epi16(d1_lo, d1_lo);
                __m512i prod1_hi = _mm512_maddubs_epi16(d1_hi, d1_hi);
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_lo, ones));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_hi, ones));

                a += 64;
                b += 64;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last_full_byte - a;
            if (rem > 0) {
                const __mmask64 tail_mask = tail_mask::mask64(static_cast<size_t>(rem));
                const __m512i xx1 = _mm512_maskz_loadu_epi8(tail_mask, a);
                const __m512i yy1 = _mm512_maskz_loadu_epi8(tail_mask, b);
                const __m512i xx1_lo = _mm512_xor_si512(_mm512_and_si512(xx1, mask), sign_flip);
                const __m512i xx1_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(xx1, 4), mask), sign_flip);
                const __m512i yy1_lo = _mm512_xor_si512(_mm512_and_si512(yy1, mask), sign_flip);
                const __m512i yy1_hi = _mm512_xor_si512(_mm512_and_si512(_mm512_srli_epi16(yy1, 4), mask), sign_flip);

                const __m512i d1_lo = _mm512_abs_epi8(_mm512_sub_epi8(xx1_lo, yy1_lo));
                const __m512i d1_hi = _mm512_abs_epi8(_mm512_sub_epi8(xx1_hi, yy1_hi));

                const __m512i prod1_lo = _mm512_maddubs_epi16(d1_lo, d1_lo);
                const __m512i prod1_hi = _mm512_maddubs_epi16(d1_hi, d1_hi);
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_lo, ones));
                sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(prod1_hi, ones));
            }
        }

        __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
        return static_cast<float>(int4_l2_hsum512(sum512));
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class L2Int4_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = *((size_t*)qty_ptr);
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const size_t num_bytes = (dim + 1) / 2;
        const uint8_t* last_full_byte = a + num_bytes;

        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();
        const __m256i mask = _mm256_set1_epi8(0x0F);
        const __m256i sign_flip = _mm256_set1_epi8(0x08);
        const __m256i ones = _mm256_set1_epi16(1);

        if constexpr (HasDualSimd) {
            while (a + 63 < last_full_byte) {
                __m256i xx1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i yy1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i xx1_lo = _mm256_xor_si256(_mm256_and_si256(xx1, mask), sign_flip);
                __m256i xx1_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(xx1, 4), mask), sign_flip);
                __m256i yy1_lo = _mm256_xor_si256(_mm256_and_si256(yy1, mask), sign_flip);
                __m256i yy1_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(yy1, 4), mask), sign_flip);

                __m256i d1_lo = _mm256_abs_epi8(_mm256_sub_epi8(xx1_lo, yy1_lo));
                __m256i d1_hi = _mm256_abs_epi8(_mm256_sub_epi8(xx1_hi, yy1_hi));

                __m256i prod1_lo = _mm256_maddubs_epi16(d1_lo, d1_lo);
                __m256i prod1_hi = _mm256_maddubs_epi16(d1_hi, d1_hi);
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_lo, ones));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_hi, ones));

                a += 32;
                b += 32;

                __m256i xx2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                __m256i yy2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                __m256i xx2_lo = _mm256_xor_si256(_mm256_and_si256(xx2, mask), sign_flip);
                __m256i xx2_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(xx2, 4), mask), sign_flip);
                __m256i yy2_lo = _mm256_xor_si256(_mm256_and_si256(yy2, mask), sign_flip);
                __m256i yy2_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(yy2, 4), mask), sign_flip);

                __m256i d2_lo = _mm256_abs_epi8(_mm256_sub_epi8(xx2_lo, yy2_lo));
                __m256i d2_hi = _mm256_abs_epi8(_mm256_sub_epi8(xx2_hi, yy2_hi));

                __m256i prod2_lo = _mm256_maddubs_epi16(d2_lo, d2_lo);
                __m256i prod2_hi = _mm256_maddubs_epi16(d2_hi, d2_hi);
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
                __m256i xx1_lo = _mm256_xor_si256(_mm256_and_si256(xx1, mask), sign_flip);
                __m256i xx1_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(xx1, 4), mask), sign_flip);
                __m256i yy1_lo = _mm256_xor_si256(_mm256_and_si256(yy1, mask), sign_flip);
                __m256i yy1_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(yy1, 4), mask), sign_flip);

                __m256i d1_lo = _mm256_abs_epi8(_mm256_sub_epi8(xx1_lo, yy1_lo));
                __m256i d1_hi = _mm256_abs_epi8(_mm256_sub_epi8(xx1_hi, yy1_hi));

                __m256i prod1_lo = _mm256_maddubs_epi16(d1_lo, d1_lo);
                __m256i prod1_hi = _mm256_maddubs_epi16(d1_hi, d1_hi);
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_lo, ones));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_hi, ones));

                a += 32;
                b += 32;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last_full_byte - a;
            if (rem > 0) {
                const __m256i xx1 = tail_mask::load_tail_bytes_32(a, static_cast<size_t>(rem), num_bytes);
                const __m256i yy1 = tail_mask::load_tail_bytes_32(b, static_cast<size_t>(rem), num_bytes);
                const __m256i xx1_lo = _mm256_xor_si256(_mm256_and_si256(xx1, mask), sign_flip);
                const __m256i xx1_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(xx1, 4), mask), sign_flip);
                const __m256i yy1_lo = _mm256_xor_si256(_mm256_and_si256(yy1, mask), sign_flip);
                const __m256i yy1_hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(yy1, 4), mask), sign_flip);

                const __m256i d1_lo = _mm256_abs_epi8(_mm256_sub_epi8(xx1_lo, yy1_lo));
                const __m256i d1_hi = _mm256_abs_epi8(_mm256_sub_epi8(xx1_hi, yy1_hi));

                const __m256i prod1_lo = _mm256_maddubs_epi16(d1_lo, d1_lo);
                const __m256i prod1_hi = _mm256_maddubs_epi16(d1_hi, d1_hi);
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_lo, ones));
                sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(prod1_hi, ones));
            }
        }

        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        return static_cast<float>(int4_l2_hsum256(sum256));
    }
};

#endif  // DEGLIB_X86

using DistanceVariant = std::variant<
    L2Int4
#if defined(DEGLIB_X86)
    ,
    L2Int4_AVX512<ResidualMode::Full>,
    L2Int4_AVX512<ResidualMode::DualPlusSimd>,
    L2Int4_AVX512<ResidualMode::DualTail>,
    L2Int4_AVX512<ResidualMode::DualOnly>,
    L2Int4_AVX512<ResidualMode::SimdTail>,
    L2Int4_AVX512<ResidualMode::SimdOnly>,
    L2Int4_AVX512<ResidualMode::TailOnly>,
    L2Int4_AVX2<ResidualMode::Full>,
    L2Int4_AVX2<ResidualMode::DualPlusSimd>,
    L2Int4_AVX2<ResidualMode::DualTail>,
    L2Int4_AVX2<ResidualMode::DualOnly>,
    L2Int4_AVX2<ResidualMode::SimdTail>,
    L2Int4_AVX2<ResidualMode::SimdOnly>,
    L2Int4_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    const size_t num_bytes = (dim + 1) / 2;
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (num_bytes < 64) {
            return L2Int4_AVX512<ResidualMode::TailOnly>{};
        } else if (num_bytes < 128) {
            if (num_bytes == 64)
                return L2Int4_AVX512<ResidualMode::SimdOnly>{};
            else
                return L2Int4_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = num_bytes % 128;
            if (rem == 0)
                return L2Int4_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 64)
                return L2Int4_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 64)
                return L2Int4_AVX512<ResidualMode::DualTail>{};
            else
                return L2Int4_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (num_bytes < 32) {
            return L2Int4_AVX2<ResidualMode::TailOnly>{};
        } else if (num_bytes < 64) {
            if (num_bytes == 32)
                return L2Int4_AVX2<ResidualMode::SimdOnly>{};
            else
                return L2Int4_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = num_bytes % 64;
            if (rem == 0)
                return L2Int4_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 32)
                return L2Int4_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 32)
                return L2Int4_AVX2<ResidualMode::DualTail>{};
            else
                return L2Int4_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return L2Int4{};
}

}  // namespace deglib::distances::int4_l2

