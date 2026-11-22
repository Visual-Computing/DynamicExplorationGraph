#pragma once

#include "deglib/distance/residual_mode.h"
#include "deglib/distance/tail_mask.h"

#include <cmath>
#include <stdexcept>
#include <variant>

namespace deglib::distances::fp32_ip {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------- Float Inner Product Dists ---------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

class InnerProductFloat {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) { return 1.f - dot(pVect1v, pVect2v, qty_ptr); }
    inline static float dot(const void* pVect1v, const void* pVect2v, const void* qty_ptr) { return ip_naive(pVect1v, pVect2v, qty_ptr); }

    inline static float ip_naive(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);

        const float* last = a + size;
        const float* unroll_group = last - 3;

        // Process 4 entries at each loop for efficiency.
        float result = 0;
        while (a < unroll_group) {
            result = std::fma(a[3], b[3], std::fma(a[2], b[2], std::fma(a[1], b[1], std::fma(a[0], b[0], result))));
            a += 4;
            b += 4;
        }

        // Process last 0-3 entries
        while (a < last) {
            result = std::fma(*a, *b, result);
            a++;
            b++;
        }

        return result;
    }
};

#if defined(DEGLIB_X86)
// -------------------------------------------------------------------
// InnerProductFloat SIMD implementations — process vectors with
// -------------------------------------------------------------------
// InnerProductFloat SIMD implementations — process vectors with
// aligned SIMD portions plus scalar residuals for any unaligned tail.
// Separate classes per SIMD width so that compare() has zero
// runtime dispatch overhead — select_dist() chooses the class.
// The HasResidual template parameter controls whether the scalar
// residual tail loop is compiled in. When HasResidual == false,
// the residual loop is eliminated at compile time, producing a
// faster path for dimensions that are known to be SIMD-aligned.
// -------------------------------------------------------------------

DEGLIB_TARGET_AVX2 inline static float fp32_hsum256(__m256 s) {
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

DEGLIB_TARGET_AVX512 inline static float fp32_hsum512(__m512 s) { return _mm512_reduce_add_ps(s); }

template <ResidualMode Mode = ResidualMode::Full>
class InnerProductFloat_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);

        const float* last = a + size;

        __m512 sum512_1 = _mm512_setzero_ps();
        __m512 sum512_2 = _mm512_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 31 < last) {
                sum512_1 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512_1);
                a += 16;
                b += 16;
                sum512_2 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512_2);
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            while (a + 15 < last) {
                __m512 q_vec = _mm512_loadu_ps(a);
                __m512 db_vec = _mm512_loadu_ps(b);
                sum512_1 = _mm512_fmadd_ps(q_vec, db_vec, sum512_1);
                a += 16;
                b += 16;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __mmask16 mask = tail_mask::mask16(static_cast<size_t>(rem));
                sum512_1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, a), _mm512_maskz_loadu_ps(mask, b), sum512_1);
            }
        }

        // Horizontal reduce of SIMD accumulators
        __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(32) float f[4];
        _mm_store_ps(f, sum128);
        float result = f[0] + f[1] + f[2] + f[3];

        return 1.f - result;
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch4(
        const void* qv, const void* b0v, const void* b1v, const void* b2v, const void* b3v, const void* qty_ptr, float* out
    ) {
        const size_t size = *static_cast<const size_t*>(qty_ptr);
        const float* q = static_cast<const float*>(qv);
        const float* b0 = static_cast<const float*>(b0v);
        const float* b1 = static_cast<const float*>(b1v);
        const float* b2 = static_cast<const float*>(b2v);
        const float* b3 = static_cast<const float*>(b3v);
        const float* last = q + size;

        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        __m512 sum2 = _mm512_setzero_ps();
        __m512 sum3 = _mm512_setzero_ps();

        if constexpr (HasDualSimd) {
            while (q + 31 < last) {
                const __m512 vq0 = _mm512_loadu_ps(q);
                sum0 = _mm512_fmadd_ps(vq0, _mm512_loadu_ps(b0), sum0);
                sum1 = _mm512_fmadd_ps(vq0, _mm512_loadu_ps(b1), sum1);
                sum2 = _mm512_fmadd_ps(vq0, _mm512_loadu_ps(b2), sum2);
                sum3 = _mm512_fmadd_ps(vq0, _mm512_loadu_ps(b3), sum3);

                const __m512 vq1 = _mm512_loadu_ps(q + 16);
                sum0 = _mm512_fmadd_ps(vq1, _mm512_loadu_ps(b0 + 16), sum0);
                sum1 = _mm512_fmadd_ps(vq1, _mm512_loadu_ps(b1 + 16), sum1);
                sum2 = _mm512_fmadd_ps(vq1, _mm512_loadu_ps(b2 + 16), sum2);
                sum3 = _mm512_fmadd_ps(vq1, _mm512_loadu_ps(b3 + 16), sum3);

                q += 32;
                b0 += 32;
                b1 += 32;
                b2 += 32;
                b3 += 32;
            }
        }

        if constexpr (HasSimd) {
            while (q + 15 < last) {
                const __m512 vq = _mm512_loadu_ps(q);
                sum0 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(b0), sum0);
                sum1 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(b1), sum1);
                sum2 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(b2), sum2);
                sum3 = _mm512_fmadd_ps(vq, _mm512_loadu_ps(b3), sum3);
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
                const __m512 vq = _mm512_maskz_loadu_ps(mask, q);
                sum0 = _mm512_fmadd_ps(vq, _mm512_maskz_loadu_ps(mask, b0), sum0);
                sum1 = _mm512_fmadd_ps(vq, _mm512_maskz_loadu_ps(mask, b1), sum1);
                sum2 = _mm512_fmadd_ps(vq, _mm512_maskz_loadu_ps(mask, b2), sum2);
                sum3 = _mm512_fmadd_ps(vq, _mm512_maskz_loadu_ps(mask, b3), sum3);
            }
        }

        out[0] = 1.f - fp32_hsum512(sum0);
        out[1] = 1.f - fp32_hsum512(sum1);
        out[2] = 1.f - fp32_hsum512(sum2);
        out[3] = 1.f - fp32_hsum512(sum3);
    }
};

template <ResidualMode Mode = ResidualMode::Full>
class InnerProductFloat_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const float* a = static_cast<const float*>(pVect1v);
        const float* b = static_cast<const float*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        const float* last = a + size;

        __m256 sum256_1 = _mm256_setzero_ps();
        __m256 sum256_2 = _mm256_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 15 < last) {
                __m256 va1 = _mm256_loadu_ps(a);
                __m256 vb1 = _mm256_loadu_ps(b);
                sum256_1 = _mm256_fmadd_ps(va1, vb1, sum256_1);
                a += 8;
                b += 8;
                __m256 va2 = _mm256_loadu_ps(a);
                __m256 vb2 = _mm256_loadu_ps(b);
                sum256_2 = _mm256_fmadd_ps(va2, vb2, sum256_2);
                a += 8;
                b += 8;
            }
        }
        if constexpr (HasSimd) {
            while (a + 7 < last) {
                __m256 va1 = _mm256_loadu_ps(a);
                __m256 vb1 = _mm256_loadu_ps(b);
                sum256_1 = _mm256_fmadd_ps(va1, vb1, sum256_1);
                a += 8;
                b += 8;
            }
        }

        if constexpr (HasTail) {
            const ptrdiff_t rem = last - a;
            if (rem > 0) {
                const __m256i mask = tail_mask::load_mask_ps(static_cast<size_t>(rem));
                sum256_1 = _mm256_fmadd_ps(_mm256_maskload_ps(a, mask), _mm256_maskload_ps(b, mask), sum256_1);
            }
        }

        // Horizontal reduce of SIMD accumulators
        __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
        float result = fp32_hsum256(sum256);

        return 1.f - result;
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch4(
        const void* qv, const void* b0v, const void* b1v, const void* b2v, const void* b3v, const void* qty_ptr, float* out
    ) {
        const size_t size = *static_cast<const size_t*>(qty_ptr);
        const float* q = static_cast<const float*>(qv);
        const float* b0 = static_cast<const float*>(b0v);
        const float* b1 = static_cast<const float*>(b1v);
        const float* b2 = static_cast<const float*>(b2v);
        const float* b3 = static_cast<const float*>(b3v);
        const float* last = q + size;

        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();

        if constexpr (HasDualSimd || HasSimd) {
            while (q + 7 < last) {
                const __m256 vq = _mm256_loadu_ps(q);
                sum0 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(b0), sum0);
                sum1 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(b1), sum1);
                sum2 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(b2), sum2);
                sum3 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(b3), sum3);
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
                const __m256i mask = tail_mask::load_mask_ps(static_cast<size_t>(rem));
                const __m256 vq = _mm256_maskload_ps(q, mask);
                sum0 = _mm256_fmadd_ps(vq, _mm256_maskload_ps(b0, mask), sum0);
                sum1 = _mm256_fmadd_ps(vq, _mm256_maskload_ps(b1, mask), sum1);
                sum2 = _mm256_fmadd_ps(vq, _mm256_maskload_ps(b2, mask), sum2);
                sum3 = _mm256_fmadd_ps(vq, _mm256_maskload_ps(b3, mask), sum3);
            }
        }

        out[0] = 1.f - fp32_hsum256(sum0);
        out[1] = 1.f - fp32_hsum256(sum1);
        out[2] = 1.f - fp32_hsum256(sum2);
        out[3] = 1.f - fp32_hsum256(sum3);
    }
};
#endif

using DistanceVariant = std::variant<
    InnerProductFloat
#if defined(DEGLIB_X86)
    ,
    InnerProductFloat_AVX512<ResidualMode::Full>,
    InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>,
    InnerProductFloat_AVX512<ResidualMode::DualTail>,
    InnerProductFloat_AVX512<ResidualMode::DualOnly>,
    InnerProductFloat_AVX512<ResidualMode::SimdTail>,
    InnerProductFloat_AVX512<ResidualMode::SimdOnly>,
    InnerProductFloat_AVX512<ResidualMode::TailOnly>,
    InnerProductFloat_AVX2<ResidualMode::Full>,
    InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>,
    InnerProductFloat_AVX2<ResidualMode::DualTail>,
    InnerProductFloat_AVX2<ResidualMode::DualOnly>,
    InnerProductFloat_AVX2<ResidualMode::SimdTail>,
    InnerProductFloat_AVX2<ResidualMode::SimdOnly>,
    InnerProductFloat_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 16) {
            return InnerProductFloat_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return InnerProductFloat_AVX512<ResidualMode::SimdOnly>{};
            else
                return InnerProductFloat_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 32;
            if (rem == 0)
                return InnerProductFloat_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 16)
                return InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 16)
                return InnerProductFloat_AVX512<ResidualMode::DualTail>{};
            else
                return InnerProductFloat_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 8) {
            return InnerProductFloat_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 16) {
            if (dim == 8)
                return InnerProductFloat_AVX2<ResidualMode::SimdOnly>{};
            else
                return InnerProductFloat_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 16;
            if (rem == 0)
                return InnerProductFloat_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 8)
                return InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 8)
                return InnerProductFloat_AVX2<ResidualMode::DualTail>{};
            else
                return InnerProductFloat_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return InnerProductFloat{};
}

}  // namespace deglib::distances::fp32_ip
