#pragma once

#include <cmath>
#include <variant>
#include "deglib/distance/residual_mode.h"

namespace deglib::distances::fp32_ip {

        // ---------------------------------------------------------------------------------------------------------------------
        // ----------------------------------------------- Float Inner Product Dists ---------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------

        class InnerProductFloat {
        public:
            static constexpr const char* get_instruction() { return "Scalar"; }

            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - dot(pVect1v, pVect2v, qty_ptr);
            }

            inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return ip_naive(pVect1v, pVect2v, qty_ptr);
            }

            inline static float ip_naive(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

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
        // aligned SIMD portions plus scalar residuals for any unaligned tail.
        // Separate classes per SIMD width so that compare() has zero
        // runtime dispatch overhead — select_dist() chooses the class.
        // The HasResidual template parameter controls whether the scalar
        // residual tail loop is compiled in. When HasResidual == false,
        // the residual loop is eliminated at compile time, producing a
        // faster path for dimensions that are known to be SIMD-aligned.
        // -------------------------------------------------------------------

        template <ResidualMode Mode = ResidualMode::Full>
        class InnerProductFloat_AVX512 {
            static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
            static constexpr bool HasSimd     = has_flag(Mode, ResidualMode::Simd);
            static constexpr bool HasTail     = has_flag(Mode, ResidualMode::Tail);

        public:
            static constexpr const char* get_instruction() { return "AVX512"; }

            DEGLIB_TARGET_AVX512 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;

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
                        sum512_1 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512_1);
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
                        result = std::fma(*a++, *b++, result);
                    }
                }

                return 1.f - result;
            }
        };

        template <ResidualMode Mode = ResidualMode::Full>
        class InnerProductFloat_AVX2 {
            static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
            static constexpr bool HasSimd     = has_flag(Mode, ResidualMode::Simd);
            static constexpr bool HasTail     = has_flag(Mode, ResidualMode::Tail);

        public:
            static constexpr const char* get_instruction() { return "AVX2"; }
            DEGLIB_TARGET_AVX2 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;

                __m256 sum256_1 = _mm256_setzero_ps();
                __m256 sum256_2 = _mm256_setzero_ps();
                if constexpr (HasDualSimd) {
                    while (a + 15 < last) {
                        sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_1);
                        a += 8;
                        b += 8;
                        sum256_2 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_2);
                        a += 8;
                        b += 8;
                    }
                }
                if constexpr (HasSimd) {
                    while (a + 7 < last) {
                        sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_1);
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
                        result = std::fma(*a++, *b++, result);
                    }
                }

                return 1.f - result;
            }
        };
#endif

    using DistanceVariant = std::variant<
        InnerProductFloat
#if defined(DEGLIB_X86)
        , InnerProductFloat_AVX512<ResidualMode::Full>
        , InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>
        , InnerProductFloat_AVX512<ResidualMode::DualTail>
        , InnerProductFloat_AVX512<ResidualMode::DualOnly>
        , InnerProductFloat_AVX512<ResidualMode::SimdTail>
        , InnerProductFloat_AVX512<ResidualMode::SimdOnly>
        , InnerProductFloat_AVX512<ResidualMode::TailOnly>
        , InnerProductFloat_AVX2<ResidualMode::Full>
        , InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>
        , InnerProductFloat_AVX2<ResidualMode::DualTail>
        , InnerProductFloat_AVX2<ResidualMode::DualOnly>
        , InnerProductFloat_AVX2<ResidualMode::SimdTail>
        , InnerProductFloat_AVX2<ResidualMode::SimdOnly>
        , InnerProductFloat_AVX2<ResidualMode::TailOnly>
#endif
    >;

    inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
        if (instruction == deglib::cpu::InstructionSet::Scalar) {
            return InnerProductFloat{};
        }

#if defined(DEGLIB_X86)
        if (instruction == deglib::cpu::InstructionSet::AVX512 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx512())) {
            if (instruction == deglib::cpu::InstructionSet::AVX512 && !deglib::cpu::has_avx512()) {
                throw std::runtime_error("AVX512 instruction set requested, but not supported by CPU");
            }
            if (dim < 16) {
                return InnerProductFloat_AVX512<ResidualMode::TailOnly>{};
            } else if (dim < 32) {
                if (dim == 16) return InnerProductFloat_AVX512<ResidualMode::SimdOnly>{};
                else return InnerProductFloat_AVX512<ResidualMode::SimdTail>{};
            } else {
                if (dim % 32 == 0) return InnerProductFloat_AVX512<ResidualMode::DualOnly>{};
                else if (dim % 16 == 0) return InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>{};
                else return InnerProductFloat_AVX512<ResidualMode::Full>{};
            }
        }
        else if (instruction == deglib::cpu::InstructionSet::AVX2 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx2())) {
            if (instruction == deglib::cpu::InstructionSet::AVX2 && !deglib::cpu::has_avx2()) {
                throw std::runtime_error("AVX2 instruction set requested, but not supported by CPU");
            }
            if (dim < 8) {
                return InnerProductFloat_AVX2<ResidualMode::TailOnly>{};
            } else if (dim < 16) {
                if (dim == 8) return InnerProductFloat_AVX2<ResidualMode::SimdOnly>{};
                else return InnerProductFloat_AVX2<ResidualMode::SimdTail>{};
            } else {
                if (dim % 16 == 0) return InnerProductFloat_AVX2<ResidualMode::DualOnly>{};
                else if (dim % 8 == 0) return InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>{};
                else return InnerProductFloat_AVX2<ResidualMode::Full>{};
            }
        }
#else
        if (instruction != deglib::cpu::InstructionSet::Auto && instruction != deglib::cpu::InstructionSet::Scalar) {
            throw std::runtime_error("Requested SIMD instruction set is not supported on this platform");
        }
#endif

        return InnerProductFloat{};
    }

} // namespace deglib::distances::fp32_ip
