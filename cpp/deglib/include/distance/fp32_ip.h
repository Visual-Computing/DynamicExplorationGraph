#pragma once

#include <cmath>
#include "distance/fp32.h"

namespace deglib::distances::fp32_ip {

        // ---------------------------------------------------------------------------------------------------------------------
        // ----------------------------------------------- Float Inner Product Dists ---------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------

        class InnerProductFloat {
        public:
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
        // InnerProductFloat16Ext — processes 16 floats (64 bytes) per iteration.
        // Separate classes per SIMD width so that compare() has zero
        // runtime dispatch overhead — select_dist() chooses the class.
        // -------------------------------------------------------------------

        class InnerProductFloat16Ext_AVX512 {
        public:
            DEGLIB_TARGET_AVX512 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - dot(pVect1v, pVect2v, qty_ptr);
            }

            DEGLIB_TARGET_AVX512 inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return ip_16ext(pVect1v, pVect2v, qty_ptr);
            }

            DEGLIB_TARGET_AVX512 inline static float ip_16ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;

                __m512 sum512 = _mm512_setzero_ps();
                while (a < last) {
                    sum512 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512);
                    a += 16;
                    b += 16;
                }

                __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        class InnerProductFloat16Ext_AVX2 {
        public:
            DEGLIB_TARGET_AVX2 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - dot(pVect1v, pVect2v, qty_ptr);
            }

            DEGLIB_TARGET_AVX2 inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return ip_16ext(pVect1v, pVect2v, qty_ptr);
            }

            DEGLIB_TARGET_AVX2 inline static float ip_16ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;

                __m256 sum256_1 = _mm256_setzero_ps();
                __m256 sum256_2 = _mm256_setzero_ps();
                while (a < last) {
                    sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_1);
                    a += 8;
                    b += 8;        
                    sum256_2 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_2);
                    a += 8;
                    b += 8;
                }
                __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        class InnerProductFloat16Ext_SSE {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - dot(pVect1v, pVect2v, qty_ptr);
            }

            inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return ip_16ext(pVect1v, pVect2v, qty_ptr);
            }

            inline static float ip_16ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;

                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
                    a += 4;
                    b += 4;
                    sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
                    a += 4;
                    b += 4;
                    sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
                    a += 4;
                    b += 4;
                    sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
                    a += 4;
                    b += 4;
                }
                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        // -------------------------------------------------------------------
        // InnerProductFloat8Ext — processes 8 floats (32 bytes) per iteration.
        // -------------------------------------------------------------------

        class InnerProductFloat8Ext_AVX2 {
        public:
            DEGLIB_TARGET_AVX2 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - dot(pVect1v, pVect2v, qty_ptr);
            }

            DEGLIB_TARGET_AVX2 inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return ip_8ext(pVect1v, pVect2v, qty_ptr);
            }

            DEGLIB_TARGET_AVX2 inline static float ip_8ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;

                __m256 sum256 = _mm256_setzero_ps();
                while (a < last) {
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        class InnerProductFloat8Ext_SSE {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - dot(pVect1v, pVect2v, qty_ptr);
            }

            inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return ip_8ext(pVect1v, pVect2v, qty_ptr);
            }

            inline static float ip_8ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;

                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
                    a += 4;
                    b += 4;
                    sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
                    a += 4;
                    b += 4;
                }
                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        // -------------------------------------------------------------------
        // InnerProductFloat4Ext — processes 4 floats (16 bytes) per iteration.
        // -------------------------------------------------------------------

        class InnerProductFloat4Ext_SSE {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - dot(pVect1v, pVect2v, qty_ptr);
            }

            inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return ip_4ext(pVect1v, pVect2v, qty_ptr);
            }

            inline static float ip_4ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;

                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
                    a += 4;
                    b += 4;
                }

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return abs(f[0] + f[1] + f[2] + f[3]);
            }
        };

        // -------------------------------------------------------------------
        // Residual classes — process the aligned portion with the SIMD
        // variant and the remainder with the scalar fallback.
        // -------------------------------------------------------------------

        class InnerProductFloat16ExtResiduals_AVX512 {
        public:
            DEGLIB_TARGET_AVX512 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty16 = qty >> 4 << 4;
                float res = InnerProductFloat16Ext_AVX512::dot(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16;
                float res_tail = InnerProductFloat::dot(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };

        class InnerProductFloat16ExtResiduals_AVX2 {
        public:
            DEGLIB_TARGET_AVX2 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty16 = qty >> 4 << 4;
                float res = InnerProductFloat16Ext_AVX2::dot(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16;
                float res_tail = InnerProductFloat::dot(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };

        class InnerProductFloat16ExtResiduals_SSE {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty16 = qty >> 4 << 4;
                float res = InnerProductFloat16Ext_SSE::dot(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16;
                float res_tail = InnerProductFloat::dot(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };

        class InnerProductFloat4ExtResiduals_SSE {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty4 = qty >> 2 << 2;
                float res = InnerProductFloat4Ext_SSE::dot(pVect1v, pVect2v, &qty4);
                float *pVect1 = (float *) pVect1v + qty4;
                float *pVect2 = (float *) pVect2v + qty4;

                size_t qty_left = qty - qty4;
                float res_tail = InnerProductFloat::dot(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };
#endif

    using DistanceVariant = std::variant<
        InnerProductFloat
#if defined(DEGLIB_X86)
        , InnerProductFloat16Ext_AVX512,
        InnerProductFloat16Ext_AVX2,
        InnerProductFloat16Ext_SSE,
        InnerProductFloat8Ext_AVX2,
        InnerProductFloat8Ext_SSE,
        InnerProductFloat4Ext_SSE,
        InnerProductFloat16ExtResiduals_AVX512,
        InnerProductFloat16ExtResiduals_AVX2,
        InnerProductFloat16ExtResiduals_SSE,
        InnerProductFloat4ExtResiduals_SSE
#endif
    >;

    inline DistanceVariant select_dist(const size_t dim) {
#if defined(DEGLIB_X86)
            if (deglib::cpu::has_avx512()) {
                if (dim % 16 == 0)
                    return InnerProductFloat16Ext_AVX512{};
                else if (dim % 8 == 0)
                    return InnerProductFloat8Ext_AVX2{};
                else if (dim % 4 == 0)
                    return InnerProductFloat4Ext_SSE{};
                else if (dim > 16)
                    return InnerProductFloat16ExtResiduals_AVX512{};
                else if (dim > 4)
                    return InnerProductFloat4ExtResiduals_SSE{};
            }
            else if (deglib::cpu::has_avx2()) {
                if (dim % 16 == 0)
                    return InnerProductFloat16Ext_AVX2{};
                else if (dim % 8 == 0)
                    return InnerProductFloat8Ext_AVX2{};
                else if (dim % 4 == 0)
                    return InnerProductFloat4Ext_SSE{};
                else if (dim > 16)
                    return InnerProductFloat16ExtResiduals_AVX2{};
                else if (dim > 4)
                    return InnerProductFloat4ExtResiduals_SSE{};
            }
            else if (deglib::cpu::has_sse42()) {
                if (dim % 16 == 0)
                    return InnerProductFloat16Ext_SSE{};
                else if (dim % 8 == 0)
                    return InnerProductFloat8Ext_SSE{};
                else if (dim % 4 == 0)
                    return InnerProductFloat4Ext_SSE{};
                else if (dim > 16)
                    return InnerProductFloat16ExtResiduals_SSE{};
                else if (dim > 4)
                    return InnerProductFloat4ExtResiduals_SSE{};
            }
#endif
            return InnerProductFloat{};
        }

} // namespace deglib::distances::fp32_ip
