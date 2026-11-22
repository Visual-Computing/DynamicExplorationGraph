#pragma once

#include "deglib/distance/fp16.h"
#include "deglib/distance/residual_mode.h"
#include "deglib/distance/tail_mask.h"

#include <cmath>
#include <stdexcept>
#include <variant>

namespace deglib::distances::fp16_l2 {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------- FP16 L2 Dists -------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// FP16 vectors are stored as uint16_t arrays (IEEE 754 half-precision bit patterns).
// The L2 distance is computed as the sum of squared differences: sum((a[i] - b[i])^2).
// ---------------------------------------------------------------------------------------------------------------------

// Scalar fallback — no SIMD required.
// Uses std::fma for precise accumulation, converting each FP16 value to float.
class L2FP16 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = static_cast<const uint16_t*>(pVect1v);
        const uint16_t* b = static_cast<const uint16_t*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        float result = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float fa = deglib::distances::fp16::fp16_to_float(a[i]);
            float fb = deglib::distances::fp16::fp16_to_float(b[i]);
            float diff = fa - fb;
            result = std::fma(diff, diff, result);
        }
        return result;
    }
};

#if defined(DEGLIB_X86)
DEGLIB_TARGET_AVX2 inline static float fp16_l2_hsum256(__m256 s) {
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

DEGLIB_TARGET_AVX512 inline static float fp16_l2_hsum512(__m512 s) { return _mm512_reduce_add_ps(s); }

// -------------------------------------------------------------------
// L2FP16 SIMD implementations — process vectors with
// aligned SIMD portions plus scalar residuals for any unaligned tail.
// Separate classes per SIMD width so that compare() has zero
// runtime dispatch overhead — select_dist() chooses the class.
// The HasResidual template parameter controls whether the scalar
// residual tail loop is compiled in. When HasResidual == false,
// the residual loop is eliminated at compile time, producing a
// faster path for dimensions that are known to be SIMD-aligned.
// -------------------------------------------------------------------

template <ResidualMode Mode = ResidualMode::Full>
class L2FP16_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = static_cast<const uint16_t*>(pVect1v);
        const uint16_t* b = static_cast<const uint16_t*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        const uint16_t* last = a + size;

        __m512 sum512_1 = _mm512_setzero_ps();
        __m512 sum512_2 = _mm512_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 31 < last) {
                __m512 va1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
                __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
                __m512 diff1 = _mm512_sub_ps(va1, vb1);
                sum512_1 = _mm512_fmadd_ps(diff1, diff1, sum512_1);
                a += 16;
                b += 16;
                __m512 va2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
                __m512 vb2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
                __m512 diff2 = _mm512_sub_ps(va2, vb2);
                sum512_2 = _mm512_fmadd_ps(diff2, diff2, sum512_2);
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            while (a + 15 < last) {
                __m512 va1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
                __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
                __m512 diff1 = _mm512_sub_ps(va1, vb1);
                sum512_1 = _mm512_fmadd_ps(diff1, diff1, sum512_1);
                a += 16;
                b += 16;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __mmask16 mask = tail_mask::mask16(static_cast<size_t>(rem));
                const __m512 va = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, a));
                const __m512 vb = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b));
                const __m512 diff = _mm512_sub_ps(va, vb);
                sum512_1 = _mm512_fmadd_ps(diff, diff, sum512_1);
            }
        }

        // Horizontal reduce of SIMD accumulators
        __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
        return fp16_l2_hsum512(sum512);
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch4(
        const void* qv, const void* b0v, const void* b1v, const void* b2v, const void* b3v, const void* qty_ptr, float* out
    ) {
        const size_t size = *static_cast<const size_t*>(qty_ptr);
        const uint16_t* q = static_cast<const uint16_t*>(qv);
        const uint16_t* b0 = static_cast<const uint16_t*>(b0v);
        const uint16_t* b1 = static_cast<const uint16_t*>(b1v);
        const uint16_t* b2 = static_cast<const uint16_t*>(b2v);
        const uint16_t* b3 = static_cast<const uint16_t*>(b3v);
        const uint16_t* last = q + size;

        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        if constexpr (HasDualSimd) {
            while (q + 31 < last) {
                const __m512 vq0 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)));
                const __m512 d00 = _mm512_sub_ps(vq0, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0))));
                const __m512 d10 = _mm512_sub_ps(vq0, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1))));
                const __m512 d20 = _mm512_sub_ps(vq0, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2))));
                const __m512 d30 = _mm512_sub_ps(vq0, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3))));
                sum0 = _mm512_fmadd_ps(d00, d00, sum0);
                sum1 = _mm512_fmadd_ps(d10, d10, sum1);
                sum2 = _mm512_fmadd_ps(d20, d20, sum2);
                sum3 = _mm512_fmadd_ps(d30, d30, sum3);

                const __m512 vq1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 16)));
                const __m512 d01 = _mm512_sub_ps(vq1, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0 + 16))));
                const __m512 d11 = _mm512_sub_ps(vq1, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1 + 16))));
                const __m512 d21 = _mm512_sub_ps(vq1, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2 + 16))));
                const __m512 d31 = _mm512_sub_ps(vq1, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3 + 16))));
                sum0 = _mm512_fmadd_ps(d01, d01, sum0);
                sum1 = _mm512_fmadd_ps(d11, d11, sum1);
                sum2 = _mm512_fmadd_ps(d21, d21, sum2);
                sum3 = _mm512_fmadd_ps(d31, d31, sum3);

                q += 32;
                b0 += 32;
                b1 += 32;
                b2 += 32;
                b3 += 32;
            }
        }

        if constexpr (HasSimd) {
            while (q + 15 < last) {
                const __m512 vq = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)));
                const __m512 d0 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b0))));
                const __m512 d1 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b1))));
                const __m512 d2 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b2))));
                const __m512 d3 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b3))));
                sum0 = _mm512_fmadd_ps(d0, d0, sum0);
                sum1 = _mm512_fmadd_ps(d1, d1, sum1);
                sum2 = _mm512_fmadd_ps(d2, d2, sum2);
                sum3 = _mm512_fmadd_ps(d3, d3, sum3);
                q += 16;
                b0 += 16;
                b1 += 16;
                b2 += 16;
                b3 += 16;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - q;
            if (rem > 0) {
                const __mmask16 mask = tail_mask::mask16(static_cast<size_t>(rem));
                const __m512 vq = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, q));
                const __m512 d0 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b0)));
                const __m512 d1 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b1)));
                const __m512 d2 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b2)));
                const __m512 d3 = _mm512_sub_ps(vq, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, b3)));
                sum0 = _mm512_fmadd_ps(d0, d0, sum0);
                sum1 = _mm512_fmadd_ps(d1, d1, sum1);
                sum2 = _mm512_fmadd_ps(d2, d2, sum2);
                sum3 = _mm512_fmadd_ps(d3, d3, sum3);
            }
        }

        out[0] = fp16_l2_hsum512(sum0);
        out[1] = fp16_l2_hsum512(sum1);
        out[2] = fp16_l2_hsum512(sum2);
        out[3] = fp16_l2_hsum512(sum3);
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class L2FP16_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = static_cast<const uint16_t*>(pVect1v);
        const uint16_t* b = static_cast<const uint16_t*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        const uint16_t* last = a + size;

        __m256 sum256_1 = _mm256_setzero_ps();
        __m256 sum256_2 = _mm256_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 15 < last) {
                __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
                __m256 vb1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
                __m256 diff1 = _mm256_sub_ps(va1, vb1);
                sum256_1 = _mm256_fmadd_ps(diff1, diff1, sum256_1);
                a += 8;
                b += 8;
                __m256 va2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
                __m256 vb2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
                __m256 diff2 = _mm256_sub_ps(va2, vb2);
                sum256_2 = _mm256_fmadd_ps(diff2, diff2, sum256_2);
                a += 8;
                b += 8;
            }
        }
        if constexpr (HasSimd) {
            while (a + 7 < last) {
                __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
                __m256 vb1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
                __m256 diff1 = _mm256_sub_ps(va1, vb1);
                sum256_1 = _mm256_fmadd_ps(diff1, diff1, sum256_1);
                a += 8;
                b += 8;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __m256 va = _mm256_cvtph_ps(tail_mask::load_tail_bytes_16(a, static_cast<size_t>(rem) * 2, size * 2));
                const __m256 vb = _mm256_cvtph_ps(tail_mask::load_tail_bytes_16(b, static_cast<size_t>(rem) * 2, size * 2));
                const __m256 diff = _mm256_sub_ps(va, vb);
                sum256_1 = _mm256_fmadd_ps(diff, diff, sum256_1);
            }
        }

        // Horizontal reduce of SIMD accumulators
        __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
        return fp16_l2_hsum256(sum256);
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch4(
        const void* qv, const void* b0v, const void* b1v, const void* b2v, const void* b3v, const void* qty_ptr, float* out
    ) {
        const size_t size = *static_cast<const size_t*>(qty_ptr);
        const uint16_t* q = static_cast<const uint16_t*>(qv);
        const uint16_t* b0 = static_cast<const uint16_t*>(b0v);
        const uint16_t* b1 = static_cast<const uint16_t*>(b1v);
        const uint16_t* b2 = static_cast<const uint16_t*>(b2v);
        const uint16_t* b3 = static_cast<const uint16_t*>(b3v);
        const uint16_t* last = q + size;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        if constexpr (HasDualSimd || HasSimd) {
            while (q + 7 < last) {
                const __m256 vq = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q)));
                const __m256 d0 = _mm256_sub_ps(vq, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b0))));
                const __m256 d1 = _mm256_sub_ps(vq, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b1))));
                const __m256 d2 = _mm256_sub_ps(vq, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b2))));
                const __m256 d3 = _mm256_sub_ps(vq, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b3))));
                sum0 = _mm256_fmadd_ps(d0, d0, sum0);
                sum1 = _mm256_fmadd_ps(d1, d1, sum1);
                sum2 = _mm256_fmadd_ps(d2, d2, sum2);
                sum3 = _mm256_fmadd_ps(d3, d3, sum3);
                q += 8;
                b0 += 8;
                b1 += 8;
                b2 += 8;
                b3 += 8;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - q;
            if (rem > 0) {
                const size_t rem_bytes = static_cast<size_t>(rem) * 2;
                const size_t total_bytes = size * 2;
                const __m256 vq = _mm256_cvtph_ps(tail_mask::load_tail_bytes_16(q, rem_bytes, total_bytes));
                const __m256 d0 = _mm256_sub_ps(vq, _mm256_cvtph_ps(tail_mask::load_tail_bytes_16(b0, rem_bytes, total_bytes)));
                const __m256 d1 = _mm256_sub_ps(vq, _mm256_cvtph_ps(tail_mask::load_tail_bytes_16(b1, rem_bytes, total_bytes)));
                const __m256 d2 = _mm256_sub_ps(vq, _mm256_cvtph_ps(tail_mask::load_tail_bytes_16(b2, rem_bytes, total_bytes)));
                const __m256 d3 = _mm256_sub_ps(vq, _mm256_cvtph_ps(tail_mask::load_tail_bytes_16(b3, rem_bytes, total_bytes)));
                sum0 = _mm256_fmadd_ps(d0, d0, sum0);
                sum1 = _mm256_fmadd_ps(d1, d1, sum1);
                sum2 = _mm256_fmadd_ps(d2, d2, sum2);
                sum3 = _mm256_fmadd_ps(d3, d3, sum3);
            }
        }

        out[0] = fp16_l2_hsum256(sum0);
        out[1] = fp16_l2_hsum256(sum1);
        out[2] = fp16_l2_hsum256(sum2);
        out[3] = fp16_l2_hsum256(sum3);
    }
};
#endif

using DistanceVariant = std::variant<
    L2FP16
#if defined(DEGLIB_X86)
    ,
    L2FP16_AVX512<ResidualMode::Full>,
    L2FP16_AVX512<ResidualMode::DualPlusSimd>,
    L2FP16_AVX512<ResidualMode::DualTail>,
    L2FP16_AVX512<ResidualMode::DualOnly>,
    L2FP16_AVX512<ResidualMode::SimdTail>,
    L2FP16_AVX512<ResidualMode::SimdOnly>,
    L2FP16_AVX512<ResidualMode::TailOnly>,
    L2FP16_AVX2<ResidualMode::Full>,
    L2FP16_AVX2<ResidualMode::DualPlusSimd>,
    L2FP16_AVX2<ResidualMode::DualTail>,
    L2FP16_AVX2<ResidualMode::DualOnly>,
    L2FP16_AVX2<ResidualMode::SimdTail>,
    L2FP16_AVX2<ResidualMode::SimdOnly>,
    L2FP16_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 16) {
            return L2FP16_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return L2FP16_AVX512<ResidualMode::SimdOnly>{};
            else
                return L2FP16_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 32;
            if (rem == 0)
                return L2FP16_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 16)
                return L2FP16_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 16)
                return L2FP16_AVX512<ResidualMode::DualTail>{};
            else
                return L2FP16_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 8) {
            return L2FP16_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 16) {
            if (dim == 8)
                return L2FP16_AVX2<ResidualMode::SimdOnly>{};
            else
                return L2FP16_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 16;
            if (rem == 0)
                return L2FP16_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 8)
                return L2FP16_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 8)
                return L2FP16_AVX2<ResidualMode::DualTail>{};
            else
                return L2FP16_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return L2FP16{};
}

}  // namespace deglib::distances::fp16_l2
