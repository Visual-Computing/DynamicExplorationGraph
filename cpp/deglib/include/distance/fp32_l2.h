#pragma once

#include "distance/fp32.h"

namespace deglib::distances::fp32_l2 {

        // ---------------------------------------------------------------------------------------------------------------------
        // ----------------------------------------------- Float L2 Dists ---------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------
        class L2Float {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                float diff0, diff1, diff2, diff3;
                const float* last = a + size;
                const float* unroll_group = last - 3;

                // Process 4 items with each loop for efficiency. 
                float result = 0;
                while (a < unroll_group) {
                    diff0 = a[0] - b[0];
                    diff1 = a[1] - b[1];
                    diff2 = a[2] - b[2];
                    diff3 = a[3] - b[3];
                    result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                    a += 4;
                    b += 4;
                }
                // Process last 0-3 elements.  Not needed for standard vector lengths. 
                while (a < last) {
                    diff0 = *a++ - *b++;
                    result += diff0 * diff0;
                }

                return result;
            }
        };

        class L2Float16Ext {
        public:
            // AVX instructions don't require their memory operands to be aligned, but SSE does
            // https://stackoverflow.com/questions/52147378/choice-between-aligned-vs-unaligned-x86-simd-instructions
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
            #if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;
            #if defined(USE_AVX512)
                // Dual accumulator unrolling: two independent FMADD chains
                // hide the latency of _mm512_fmadd_ps on modern CPUs.
                __m512 sum512_1 = _mm512_setzero_ps();
                __m512 sum512_2 = _mm512_setzero_ps();
                while (a + 31 < last)
                {
                    __m512 v1 = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
                    sum512_1 = _mm512_fmadd_ps(v1, v1, sum512_1);
                    a += 16;
                    b += 16;
                    __m512 v2 = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
                    sum512_2 = _mm512_fmadd_ps(v2, v2, sum512_2);
                    a += 16;
                    b += 16;
                }
                while (a < last)
                {
                    __m512 v = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
                    sum512_1 = _mm512_fmadd_ps(v, v, sum512_1);
                    a += 16;
                    b += 16;
                }
                __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
                __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_AVX2)
                
                // Two sum acculumator is slower
                // newer CPUs have reciprocal throughput less than its latency -> performance can be improved if multiple instructions are executed in parallel
                // https://stackoverflow.com/questions/65818232/improving-performance-of-floating-point-dot-product-of-an-array-with-simd/65827668#65827668
                __m256 sum256 = _mm256_setzero_ps();
                __m256 v;
                while (a < last) {
                    v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                    sum256 = _mm256_fmadd_ps(v, v, sum256);
                    a += 8;
                    b += 8;        
                    v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                    sum256 = _mm256_fmadd_ps(v, v, sum256);
                    a += 8;
                    b += 8;
                }

                // TODO cast faster then extract?
                //__m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
                //sum128 = _mm_add_ps(sum128, _mm_unpackhi_ps(sum128, sum128));

                // TODO horizontal add faster?
                // https://doc.rust-lang.org/core/arch/x86/fn._mm256_hadd_ps.html
                // https://stackoverflow.com/questions/51274287/computing-8-horizontal-sums-of-eight-avx-single-precision-floating-point-vectors/51275249#51275249
                // _mm256_hadd_ps(sum256)

                // TODO down to a single number 
                // https://www.py4u.net/discuss/73145

                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE42)
                __m128 sum128 = _mm_setzero_ps();
                __m128 v;
                while (a < last) {
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            #else
                return L2Float::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };
        
        class L2Float8Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {

            #if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
            #if defined(USE_AVX512) || defined(USE_AVX2)
                __m256 sum256 = _mm256_setzero_ps();
                __m256 v;
                while (a < last) {
                    v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                    sum256 = _mm256_fmadd_ps(v, v, sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE42)
                __m128 sum128 = _mm_setzero_ps();
                __m128 v;
                while (a < last) {
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            #else
                return L2Float::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };

        class L2Float4Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            #if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                // TODO add NEON: https://github.com/ashvardanian/SimSIMD/blob/main/include/simsimd/spatial.h#L180
                // https://github.com/ashvardanian/SimSIMD/blob/main/include/simsimd/types.h#L156
                // #include <arm_neon.h>
                // const float *last = a + size;
                // float32x4_t sum128 = _mm_setzero_ps();
                // float32x4_t v;
                // while (a < last) {
                //     v = vsubq_f32(vld1q_f32(a), vld1q_f32(b));
                //     sum128 = vfmaq_f32(sum128, v, v);
                //     a += 4;
                //     b += 4;
                // }
                // return vaddvq_f32(sum128);

                const float *last = a + size;
                __m128 sum128 = _mm_setzero_ps();
                __m128 v;
                while (a < last) {
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                }

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            #else
                return L2Float::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };

        class L2Float16ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty16 = qty >> 4 << 4;
                float res = L2Float16Ext::compare(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16 ;
                float res_tail = L2Float::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };

        class L2Float4ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty4 = qty >> 2 << 2;
                float res = L2Float4Ext::compare(pVect1v, pVect2v, &qty4);
                float *pVect1 = (float *) pVect1v + qty4;
                float *pVect2 = (float *) pVect2v + qty4;

                size_t qty_left = qty - qty4;
                float res_tail = L2Float::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };

    using DistanceVariant = std::variant<
        L2Float,
        L2Float16Ext,
        L2Float8Ext,
        L2Float4Ext,
        L2Float16ExtResiduals,
        L2Float4ExtResiduals
    >;

    inline DistanceVariant select_dist(const size_t dim) {
            #if defined(USE_SSE42) || defined(USE_AVX2) || defined(USE_AVX512)
                if (dim % 16 == 0)
                    return L2Float16Ext{};
                else if (dim % 8 == 0)
                    return L2Float8Ext{};
                else if (dim % 4 == 0)
                    return L2Float4Ext{};
                else if (dim > 16)
                    return L2Float16ExtResiduals{};
                else if (dim > 4)
                    return L2Float4ExtResiduals{};
            #endif
            return L2Float{};
        }

} // namespace deglib::distances::fp32_l2