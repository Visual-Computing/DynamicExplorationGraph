#pragma once

#include <cmath>
#include <variant>
#include "deglib/distance/fp16.h"
#include "deglib/distance/residual_mode.h"

namespace deglib::distances::fp16_ip {

    // ---------------------------------------------------------------------------------------------------------------------
    // ----------------------------------------------- FP16 Inner Product Dists ---------------------------------------------
    // ---------------------------------------------------------------------------------------------------------------------
    // FP16 vectors are stored as uint16_t arrays (IEEE 754 half-precision bit patterns).
    // The distance is computed as 1.f - dot_product, where dot_product is the raw
    // inner product of the float-converted vectors.
    // ---------------------------------------------------------------------------------------------------------------------

    // Scalar fallback — no SIMD required.
    // Uses std::fma for precise accumulation, converting each FP16 value to float.
    class InnerProductFP16 {
    public:
        static constexpr const char* get_instruction() { return "Scalar"; }

        inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            const uint16_t* a = static_cast<const uint16_t*>(pVect1v);
            const uint16_t* b = static_cast<const uint16_t*>(pVect2v);
            size_t size = *((size_t*)qty_ptr);

            float result = 0.0f;
            for (size_t i = 0; i < size; ++i) {
                float fa = deglib::distances::fp16::fp16_to_float(a[i]);
                float fb = deglib::distances::fp16::fp16_to_float(b[i]);
                result = std::fma(fa, fb, result);
            }
            return 1.f - result;
        }
    };

#if defined(DEGLIB_X86)
    // -------------------------------------------------------------------
    // InnerProductFP16 SIMD implementations — process vectors with
    // aligned SIMD portions plus scalar residuals for any unaligned tail.
    // Separate classes per SIMD width so that compare() has zero
    // runtime dispatch overhead — select_dist() chooses the class.
    // The HasResidual template parameter controls whether the scalar
    // residual tail loop is compiled in. When HasResidual == false,
    // the residual loop is eliminated at compile time, producing a
    // faster path for dimensions that are known to be SIMD-aligned.
    // -------------------------------------------------------------------

    template <ResidualMode Mode = ResidualMode::Full>
    class InnerProductFP16_AVX512 {
        static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
        static constexpr bool HasSimd     = has_flag(Mode, ResidualMode::Simd);
        static constexpr bool HasTail     = has_flag(Mode, ResidualMode::Tail);

    public:
        static constexpr const char* get_instruction() { return "AVX512"; }

        DEGLIB_TARGET_AVX512 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            const uint16_t *a = static_cast<const uint16_t *>(pVect1v);
            const uint16_t *b = static_cast<const uint16_t *>(pVect2v);
            size_t size = *((size_t *) qty_ptr);

            const uint16_t *last = a + size;

            __m512 sum512_1 = _mm512_setzero_ps();
            __m512 sum512_2 = _mm512_setzero_ps();
            if constexpr (HasDualSimd) {
                while (a + 31 < last) {
                    __m512 va1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a)));
                    __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b)));
                    sum512_1 = _mm512_fmadd_ps(va1, vb1, sum512_1);
                    a += 16;
                    b += 16;
                    __m512 va2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a)));
                    __m512 vb2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b)));
                    sum512_2 = _mm512_fmadd_ps(va2, vb2, sum512_2);
                    a += 16;
                    b += 16;
                }
            }
            if constexpr (HasSimd) {
                while (a + 15 < last) {
                    __m512 va1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a)));
                    __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b)));
                    sum512_1 = _mm512_fmadd_ps(va1, vb1, sum512_1);
                    a += 16;
                    b += 16;
                }
            }

            // Horizontal reduce of SIMD accumulators
            __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
            __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(32) float f[4];
            _mm_store_ps(f, sum128);
            float result = f[0] + f[1] + f[2] + f[3];

            // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
            if constexpr (HasTail) {
                while (a < last) {
                    float fa = deglib::distances::fp16::fp16_to_float(*a++);
                    float fb = deglib::distances::fp16::fp16_to_float(*b++);
                    result = std::fma(fa, fb, result);
                }
            }

            return 1.f - result;
        }
    };

    template <ResidualMode Mode = ResidualMode::Full>
    class InnerProductFP16_AVX2 {
        static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
        static constexpr bool HasSimd     = has_flag(Mode, ResidualMode::Simd);
        static constexpr bool HasTail     = has_flag(Mode, ResidualMode::Tail);

    public:
        static constexpr const char* get_instruction() { return "AVX2"; }
        DEGLIB_TARGET_AVX2 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            const uint16_t *a = static_cast<const uint16_t *>(pVect1v);
            const uint16_t *b = static_cast<const uint16_t *>(pVect2v);
            size_t size = *((size_t *) qty_ptr);

            const uint16_t *last = a + size;

            __m256 sum256_1 = _mm256_setzero_ps();
            __m256 sum256_2 = _mm256_setzero_ps();
            if constexpr (HasDualSimd) {
                while (a + 15 < last) {
                    __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a)));
                    __m256 vb1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
                    sum256_1 = _mm256_fmadd_ps(va1, vb1, sum256_1);
                    a += 8;
                    b += 8;
                    __m256 va2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a)));
                    __m256 vb2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
                    sum256_2 = _mm256_fmadd_ps(va2, vb2, sum256_2);
                    a += 8;
                    b += 8;
                }
            }
            if constexpr (HasSimd) {
                while (a + 7 < last) {
                    __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a)));
                    __m256 vb1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
                    sum256_1 = _mm256_fmadd_ps(va1, vb1, sum256_1);
                    a += 8;
                    b += 8;
                }
            }

            // Horizontal reduce of SIMD accumulators
            __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(32) float f[4];
            _mm_store_ps(f, sum128);
            float result = f[0] + f[1] + f[2] + f[3];

            // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
            if constexpr (HasTail) {
                while (a < last) {
                    float fa = deglib::distances::fp16::fp16_to_float(*a++);
                    float fb = deglib::distances::fp16::fp16_to_float(*b++);
                    result = std::fma(fa, fb, result);
                }
            }

            return 1.f - result;
        }
    };
#endif

    using DistanceVariant = std::variant<
        InnerProductFP16
#if defined(DEGLIB_X86)
        , InnerProductFP16_AVX512<ResidualMode::Full>
        , InnerProductFP16_AVX512<ResidualMode::DualPlusSimd>
        , InnerProductFP16_AVX512<ResidualMode::DualTail>
        , InnerProductFP16_AVX512<ResidualMode::DualOnly>
        , InnerProductFP16_AVX512<ResidualMode::SimdTail>
        , InnerProductFP16_AVX512<ResidualMode::SimdOnly>
        , InnerProductFP16_AVX512<ResidualMode::TailOnly>
        , InnerProductFP16_AVX2<ResidualMode::Full>
        , InnerProductFP16_AVX2<ResidualMode::DualPlusSimd>
        , InnerProductFP16_AVX2<ResidualMode::DualTail>
        , InnerProductFP16_AVX2<ResidualMode::DualOnly>
        , InnerProductFP16_AVX2<ResidualMode::SimdTail>
        , InnerProductFP16_AVX2<ResidualMode::SimdOnly>
        , InnerProductFP16_AVX2<ResidualMode::TailOnly>
#endif
    >;

    inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
        if (instruction == deglib::cpu::InstructionSet::Scalar) {
            return InnerProductFP16{};
        }

#if defined(DEGLIB_X86)
        if (instruction == deglib::cpu::InstructionSet::AVX512 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx512())) {
            if (instruction == deglib::cpu::InstructionSet::AVX512 && !deglib::cpu::has_avx512()) {
                throw std::runtime_error("AVX512 instruction set requested, but not supported by CPU");
            }
            if (dim < 16) {
                return InnerProductFP16_AVX512<ResidualMode::TailOnly>{};
            } else if (dim < 32) {
                if (dim == 16) return InnerProductFP16_AVX512<ResidualMode::SimdOnly>{};
                else return InnerProductFP16_AVX512<ResidualMode::SimdTail>{};
            } else {
                if (dim % 32 == 0) return InnerProductFP16_AVX512<ResidualMode::DualOnly>{};
                else if (dim % 16 == 0) return InnerProductFP16_AVX512<ResidualMode::DualPlusSimd>{};
                else return InnerProductFP16_AVX512<ResidualMode::Full>{};
            }
        }
        else if (instruction == deglib::cpu::InstructionSet::AVX2 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx2())) {
            if (instruction == deglib::cpu::InstructionSet::AVX2 && !deglib::cpu::has_avx2()) {
                throw std::runtime_error("AVX2 instruction set requested, but not supported by CPU");
            }
            if (dim < 8) {
                return InnerProductFP16_AVX2<ResidualMode::TailOnly>{};
            } else if (dim < 16) {
                if (dim == 8) return InnerProductFP16_AVX2<ResidualMode::SimdOnly>{};
                else return InnerProductFP16_AVX2<ResidualMode::SimdTail>{};
            } else {
                if (dim % 16 == 0) return InnerProductFP16_AVX2<ResidualMode::DualOnly>{};
                else if (dim % 8 == 0) return InnerProductFP16_AVX2<ResidualMode::DualPlusSimd>{};
                else return InnerProductFP16_AVX2<ResidualMode::Full>{};
            }
        }
#else
        if (instruction != deglib::cpu::InstructionSet::Auto && instruction != deglib::cpu::InstructionSet::Scalar) {
            throw std::runtime_error("Requested SIMD instruction set is not supported on this platform");
        }
#endif

        return InnerProductFP16{};
    }

} // namespace deglib::distances::fp16_ip
