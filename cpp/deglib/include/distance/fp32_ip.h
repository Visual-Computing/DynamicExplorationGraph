#pragma once

#include "distance/fp32.h"

namespace deglib::distances::fp32_ip {

        // ---------------------------------------------------------------------------------------------------------------------
        // ----------------------------------------------- Float Inner Product Dists ---------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------

        class InnerProductFloat {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - ip_naive(pVect1v, pVect2v, qty_ptr);
            }

            inline static float ip_naive(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                float dot0, dot1, dot2, dot3;
                const float* last = a + size;
                const float* unroll_group = last - 3;

                // Process 4 entries at each loop for efficiency. 
                float result = 0;
                while (a < unroll_group) {
                    dot0 = a[0] * b[0];
                    dot1 = a[1] * b[1];
                    dot2 = a[2] * b[2];
                    dot3 = a[3] * b[3];
                    result += dot0 + dot1 + dot2 + dot3;
                    a += 4;
                    b += 4;
                }

                // Process last 0-3 entries
                while (a < last) {
                    result += *a++ * *b++;
                }

                return result;
            }
        };

        class InnerProductFloat16Ext {
        public:

            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - ip_16ext(pVect1v, pVect2v, qty_ptr);
            }

            // AVX instructions don't require their memory operands to be aligned, but SSE does
            // https://stackoverflow.com/questions/52147378/choice-between-aligned-vs-unaligned-x86-simd-instructions
            inline static float ip_16ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            #if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;
            #if defined(USE_AVX512)
                __m512 sum512 = _mm512_setzero_ps();
                while (a < last) {
                    sum512 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512);
                    a += 16;
                    b += 16;
                }

                __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_AVX2)
                __m256 sum256 = _mm256_setzero_ps();
                while (a < last) {
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;        
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE42)
                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            #else
                return InnerProductFloat::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };
        
        class InnerProductFloat8Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - ip_8ext(pVect1v, pVect2v, qty_ptr);
            }

            inline static float ip_8ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            #if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
            #if defined(USE_AVX2)
                __m256 sum256 = _mm256_setzero_ps();
                while (a < last) {
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE42)
                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            #else
                return InnerProductFloat::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };

        class InnerProductFloat4Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                return 1.f - ip_4ext(pVect1v, pVect2v, qty_ptr);
            }

            inline static float ip_4ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            #if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                }

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return abs(f[0] + f[1] + f[2] + f[3]);
            #else
                return InnerProductFloat::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };

        class InnerProductFloat16ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty16 = qty >> 4 << 4;
                float res = InnerProductFloat16Ext::ip_16ext(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16;
                float res_tail = InnerProductFloat::ip_naive(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };

        class InnerProductFloat4ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty4 = qty >> 2 << 2;
                float res = InnerProductFloat4Ext::ip_4ext(pVect1v, pVect2v, &qty4);
                float *pVect1 = (float *) pVect1v + qty4;
                float *pVect2 = (float *) pVect2v + qty4;

                size_t qty_left = qty - qty4;
                float res_tail = InnerProductFloat::ip_naive(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };

    using DistanceVariant = std::variant<
        InnerProductFloat,
        InnerProductFloat16Ext,
        InnerProductFloat8Ext,
        InnerProductFloat4Ext,
        InnerProductFloat16ExtResiduals,
        InnerProductFloat4ExtResiduals
    >;

    inline DistanceVariant select_dist(const size_t dim) {
            #if defined(USE_SSE42) || defined(USE_AVX2) || defined(USE_AVX512)
                if (dim % 16 == 0)
                    return InnerProductFloat16Ext{};
                else if (dim % 8 == 0)
                    return InnerProductFloat8Ext{};
                else if (dim % 4 == 0)
                    return InnerProductFloat4Ext{};
                else if (dim > 16)
                    return InnerProductFloat16ExtResiduals{};
                else if (dim > 4)
                    return InnerProductFloat4ExtResiduals{};
            #endif
            return InnerProductFloat{};
        }

} // namespace deglib::distances::fp32_ip
