#pragma once

#include "deglib/distance/residual_mode.h"
#include "deglib/distance/int4.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <variant>

namespace deglib::distances::int4_ip {

inline static int8_t sign_extend_nibble(uint8_t n) {
    int8_t s = static_cast<int8_t>(n & 0x0F);
    if (s & 0x08) s |= 0xF0;
    return s;
}

// Scalar fallback — no SIMD required.
class InnerProductInt4 {
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

            int32_t a0 = sign_extend_nibble(byte_a);
            int32_t a1 = sign_extend_nibble(byte_a >> 4);
            int32_t b0 = sign_extend_nibble(byte_b);
            int32_t b1 = sign_extend_nibble(byte_b >> 4);

            result += a0 * b0 + a1 * b1;
        }

        if (dim & 1) {
            uint8_t byte_a = a[num_bytes];
            uint8_t byte_b = b[num_bytes];
            int32_t a0 = sign_extend_nibble(byte_a);
            int32_t b0 = sign_extend_nibble(byte_b);
            result += a0 * b0;
        }

        return -static_cast<float>(result);
    }
};

#if defined(DEGLIB_X86)

DEGLIB_TARGET_AVX2 inline static int64_t int4_ip_hsum256(__m256i s) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
    return static_cast<int64_t>(_mm_cvtsi128_si32(sum128));
}

DEGLIB_TARGET_AVX512 inline static int64_t int4_ip_hsum512(__m512i s) {
    return static_cast<int64_t>(_mm512_reduce_add_epi32(s));
}

// In AVX2/AVX512: unpack nibbles to 16-bit signed integers and multiply with _mm_madd_epi16
template <ResidualMode Mode = ResidualMode::Full>
class InnerProductInt4_AVX512 {
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

        const __m128i mask = _mm_set1_epi8(0x0F);
        const __m128i s8 = _mm_set1_epi8(0x08);
        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();

        while (a + 15 < last_full_byte) {
            __m128i ra = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            __m128i rb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

            // Low nibbles:
            __m128i a_lo = _mm_and_si128(ra, mask);
            __m128i b_lo = _mm_and_si128(rb, mask);
            // Sign extend 4-bit to 8-bit: (x ^ 0x08) - 8
            __m128i a_lo_s8 = _mm_sub_epi8(_mm_xor_si128(a_lo, s8), s8);
            __m128i b_lo_s8 = _mm_sub_epi8(_mm_xor_si128(b_lo, s8), s8);

            // High nibbles:
            __m128i a_hi = _mm_and_si128(_mm_srli_epi16(ra, 4), mask);
            __m128i b_hi = _mm_and_si128(_mm_srli_epi16(rb, 4), mask);
            __m128i a_hi_s8 = _mm_sub_epi8(_mm_xor_si128(a_hi, s8), s8);
            __m128i b_hi_s8 = _mm_sub_epi8(_mm_xor_si128(b_hi, s8), s8);

            // Sign extend int8 to int16:
            __m256i a_lo_16 = _mm256_cvtepi8_epi16(a_lo_s8);
            __m256i b_lo_16 = _mm256_cvtepi8_epi16(b_lo_s8);
            __m256i a_hi_16 = _mm256_cvtepi8_epi16(a_hi_s8);
            __m256i b_hi_16 = _mm256_cvtepi8_epi16(b_hi_s8);

            sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(a_lo_16, b_lo_16));
            sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(a_hi_16, b_hi_16));

            a += 16;
            b += 16;
        }

        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        int64_t result = int4_ip_hsum256(sum256);

        while (a < last_full_byte) {
            uint8_t byte_a = *a++;
            uint8_t byte_b = *b++;
            int32_t a0 = sign_extend_nibble(byte_a);
            int32_t a1 = sign_extend_nibble(byte_a >> 4);
            int32_t b0 = sign_extend_nibble(byte_b);
            int32_t b1 = sign_extend_nibble(byte_b >> 4);
            result += a0 * b0 + a1 * b1;
        }
        if (dim & 1) {
            uint8_t byte_a = *a;
            uint8_t byte_b = *b;
            int32_t a0 = sign_extend_nibble(byte_a);
            int32_t b0 = sign_extend_nibble(byte_b);
            result += a0 * b0;
        }

        return -static_cast<float>(result);
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class InnerProductInt4_AVX2 {
  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const size_t dim = *((size_t*)qty_ptr);
        const uint8_t* a = static_cast<const uint8_t*>(pVect1v);
        const uint8_t* b = static_cast<const uint8_t*>(pVect2v);

        const size_t num_bytes = dim / 2;
        const uint8_t* last_full_byte = a + num_bytes;

        const __m128i mask = _mm_set1_epi8(0x0F);
        const __m128i s8 = _mm_set1_epi8(0x08);
        __m256i sum256_1 = _mm256_setzero_si256();
        __m256i sum256_2 = _mm256_setzero_si256();

        while (a + 15 < last_full_byte) {
            __m128i ra = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
            __m128i rb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));

            // Low nibbles:
            __m128i a_lo = _mm_and_si128(ra, mask);
            __m128i b_lo = _mm_and_si128(rb, mask);
            __m128i a_lo_s8 = _mm_sub_epi8(_mm_xor_si128(a_lo, s8), s8);
            __m128i b_lo_s8 = _mm_sub_epi8(_mm_xor_si128(b_lo, s8), s8);

            // High nibbles:
            __m128i a_hi = _mm_and_si128(_mm_srli_epi16(ra, 4), mask);
            __m128i b_hi = _mm_and_si128(_mm_srli_epi16(rb, 4), mask);
            __m128i a_hi_s8 = _mm_sub_epi8(_mm_xor_si128(a_hi, s8), s8);
            __m128i b_hi_s8 = _mm_sub_epi8(_mm_xor_si128(b_hi, s8), s8);

            __m256i a_lo_16 = _mm256_cvtepi8_epi16(a_lo_s8);
            __m256i b_lo_16 = _mm256_cvtepi8_epi16(b_lo_s8);
            __m256i a_hi_16 = _mm256_cvtepi8_epi16(a_hi_s8);
            __m256i b_hi_16 = _mm256_cvtepi8_epi16(b_hi_s8);

            sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(a_lo_16, b_lo_16));
            sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(a_hi_16, b_hi_16));

            a += 16;
            b += 16;
        }

        __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
        int64_t result = int4_ip_hsum256(sum256);

        while (a < last_full_byte) {
            uint8_t byte_a = *a++;
            uint8_t byte_b = *b++;
            int32_t a0 = sign_extend_nibble(byte_a);
            int32_t a1 = sign_extend_nibble(byte_a >> 4);
            int32_t b0 = sign_extend_nibble(byte_b);
            int32_t b1 = sign_extend_nibble(byte_b >> 4);
            result += a0 * b0 + a1 * b1;
        }
        if (dim & 1) {
            uint8_t byte_a = *a;
            uint8_t byte_b = *b;
            int32_t a0 = sign_extend_nibble(byte_a);
            int32_t b0 = sign_extend_nibble(byte_b);
            result += a0 * b0;
        }

        return -static_cast<float>(result);
    }
};

#endif  // DEGLIB_X86

using DistanceVariant = std::variant<
    InnerProductInt4
#if defined(DEGLIB_X86)
    ,
    InnerProductInt4_AVX512<ResidualMode::Full>,
    InnerProductInt4_AVX2<ResidualMode::Full>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        return InnerProductInt4_AVX512<ResidualMode::Full>{};
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        return InnerProductInt4_AVX2<ResidualMode::Full>{};
    }
#endif

    return InnerProductInt4{};
}

}  // namespace deglib::distances::int4_ip
