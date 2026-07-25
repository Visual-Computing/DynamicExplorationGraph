#pragma once

#include <variant>
#include <concepts>

#include "config.h"

namespace deglib {
    
    namespace distances {

        // ---------------------------------------------------------------------------------------------------------------------
        // ----------------------------------------------- Float Dists ---------------------------------------------------------
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
                float res = deglib::distances::L2Float16Ext::compare(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16 ;
                float res_tail = deglib::distances::L2Float::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };

        class L2Float4ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty4 = qty >> 2 << 2;
                float res = deglib::distances::L2Float4Ext::compare(pVect1v, pVect2v, &qty4);
                float *pVect1 = (float *) pVect1v + qty4;
                float *pVect2 = (float *) pVect2v + qty4;

                size_t qty_left = qty - qty4;
                float res_tail = deglib::distances::L2Float::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };



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
                float res = deglib::distances::InnerProductFloat16Ext::ip_16ext(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16;
                float res_tail = deglib::distances::InnerProductFloat::ip_naive(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };

        class InnerProductFloat4ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty4 = qty >> 2 << 2;
                float res = deglib::distances::InnerProductFloat4Ext::ip_4ext(pVect1v, pVect2v, &qty4);
                float *pVect1 = (float *) pVect1v + qty4;
                float *pVect2 = (float *) pVect2v + qty4;

                size_t qty_left = qty - qty4;
                float res_tail = deglib::distances::InnerProductFloat::ip_naive(pVect1, pVect2, &qty_left);
                return 1.f - (res + res_tail);
            }
        };



        // ---------------------------------------------------------------------------------------------------------------------
        // ----------------------------------------------- Uint8 Dists ---------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------

        class L2Uint8 {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                int64_t result = 0;
                uint8_t *a = (uint8_t *) pVect1v;
                uint8_t *b = (uint8_t *) pVect2v;

                size_t size = *((size_t *) qty_ptr);
                for(size_t i = 0; i < size; i++) {
                    int32_t diff0 = int32_t(a[i]) - int32_t(b[i]);
                    result += diff0 * diff0;
                }

                return float(result);
            }
        };

        class L2Uint8Ext32 {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {

            #if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_SSE42)
                size_t size = *((size_t *) qty_ptr);
                const unsigned char *a = (const unsigned char *) pVect1v;
                const unsigned char *b = (const unsigned char *) pVect2v;

             #if defined(USE_AVX512)
                __m512i sum512 = _mm512_setzero_si512();

                size_t i = 0;
                for (; i + 32 <= size; i += 32) {
                    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
                    __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
                    __m512i diff = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v1), _mm512_cvtepu8_epi16(v2));
                    sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(diff, diff));
                }
                __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(sum512), _mm512_extracti64x4_epi64(sum512, 1));
                __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));

             #elif defined(USE_AVX2)

                __m256i sum256 = _mm256_setzero_si256();
                for (size_t i = 0; i + 16 <= size; i += 16) {
                    __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                    __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));

                    __m256i v1_lo = _mm256_cvtepu8_epi16(v1);
                    __m256i v2_lo = _mm256_cvtepu8_epi16(v2);

                    __m256i diff_lo = _mm256_sub_epi16(v1_lo, v2_lo);
                    __m256i sqr_lo = _mm256_madd_epi16(diff_lo, diff_lo);
                    sum256 = _mm256_add_epi32(sum256, sqr_lo);
                }
                __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum256, 0), _mm256_extracti128_si256(sum256, 1));

            #elif defined(USE_SSE42)

                __m128i d2_low_vec = _mm_setzero_si128();
                __m128i d2_high_vec = _mm_setzero_si128();
                for (size_t i = 0; i + 16 <= size; i += 16) {
                    __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                    __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
                    
                     // Sign extend int8 to int16
                    __m128i v1_lo = _mm_cvtepu8_epi16(v1);
                    __m128i v1_hi = _mm_cvtepu8_epi16(_mm_srli_si128(v1, 8));
                    __m128i v2_lo = _mm_cvtepu8_epi16(v2);
                    __m128i v2_hi = _mm_cvtepu8_epi16(_mm_srli_si128(v2, 8));

                    // Subtract and multiply
                    __m128i diff_lo = _mm_sub_epi16(v1_lo, v2_lo);
                    __m128i diff_hi = _mm_sub_epi16(v1_hi, v2_hi);
                    __m128i sqr_lo = _mm_madd_epi16(diff_lo, diff_lo);
                    __m128i sqr_hi = _mm_madd_epi16(diff_hi, diff_hi);

                    d2_low_vec = _mm_add_epi32(d2_low_vec, sqr_lo);
                    d2_high_vec = _mm_add_epi32(d2_high_vec, sqr_hi);
                }
                __m128i sum128 = _mm_add_epi32(d2_low_vec, d2_high_vec);
            #endif 

                alignas(16) int sum_array[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
                return static_cast<float>(sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]);

            #else
                return L2Uint8::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };

        class L2Uint8Ext16 {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {

            #if defined(USE_AVX2)
                size_t size = *((size_t *) qty_ptr);
                const unsigned char *a = (const unsigned char *) pVect1v;
                const unsigned char *b = (const unsigned char *) pVect2v;

                __m256i sum256 = _mm256_setzero_si256();
                for (size_t i = 0; i + 16 <= size; i += 16) {
                    __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                    __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));

                    __m256i v1_ext = _mm256_cvtepu8_epi16(v1);
                    __m256i v2_ext = _mm256_cvtepu8_epi16(v2);

                    __m256i diff = _mm256_sub_epi16(v1_ext, v2_ext);
                    __m256i sqr = _mm256_madd_epi16(diff, diff);
                    sum256 = _mm256_add_epi32(sum256, sqr);
                }
                __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));
                alignas(16) int sum_array[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
                return static_cast<float>(sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]);
            #elif defined(USE_SSE42)
                size_t size = *((size_t *) qty_ptr);
                const unsigned char *a = (const unsigned char *) pVect1v;
                const unsigned char *b = (const unsigned char *) pVect2v;
                
                __m128i d2_low_vec = _mm_setzero_si128();
                __m128i d2_high_vec = _mm_setzero_si128();
                for (size_t i = 0; i + 16 <= size; i += 16) {
                    __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                    __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
                    
                    __m128i v1_lo = _mm_cvtepu8_epi16(v1);
                    __m128i v1_hi = _mm_cvtepu8_epi16(_mm_srli_si128(v1, 8));
                    __m128i v2_lo = _mm_cvtepu8_epi16(v2);
                    __m128i v2_hi = _mm_cvtepu8_epi16(_mm_srli_si128(v2, 8));

                    __m128i diff_lo = _mm_sub_epi16(v1_lo, v2_lo);
                    __m128i diff_hi = _mm_sub_epi16(v1_hi, v2_hi);

                    __m128i sqr_lo = _mm_madd_epi16(diff_lo, diff_lo);
                    __m128i sqr_hi = _mm_madd_epi16(diff_hi, diff_hi);

                    d2_low_vec = _mm_add_epi32(d2_low_vec, sqr_lo);
                    d2_high_vec = _mm_add_epi32(d2_high_vec, sqr_hi);
                }
                __m128i sum128 = _mm_add_epi32(d2_low_vec, d2_high_vec);

                alignas(16) int sum_array[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
                return static_cast<float>(sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]);
            #else
                return L2Uint8::compare(pVect1v, pVect2v, qty_ptr);
            #endif 
            }
        };

    } // end namespace Distances

    enum class Metric {
        // 0x00 = float
        //L1 = 0x00 | 0,
        L2 = 0x00 | 1,
        InnerProduct = 0x00 | 2,

        // 0x10 = uint8
        L2_Uint8 = 0x10 | 1
    };

    /**
     * Function pointer signature for distance comparison functions.
     */
    template <typename MTYPE>
    using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);

    /**
     * Concept for distance function implementations that provide a static compare method
     * compatible with DISTFUNC<float>.
     */
    template <typename T>
    concept DistanceFunction = requires(const void* a, const void* b, const void* param) {
        { T::compare(a, b, param) } -> std::same_as<float>;
    } && std::is_convertible_v<decltype(&T::compare), DISTFUNC<float>>;

    /**
     * Variant containing all supported concrete distance function implementations.
     */
    using DistanceVariant = std::variant<
        deglib::distances::L2Float,
        deglib::distances::L2Float16Ext,
        deglib::distances::L2Float8Ext,
        deglib::distances::L2Float4Ext,
        deglib::distances::L2Float16ExtResiduals,
        deglib::distances::L2Float4ExtResiduals,
        deglib::distances::InnerProductFloat,
        deglib::distances::InnerProductFloat16Ext,
        deglib::distances::InnerProductFloat8Ext,
        deglib::distances::InnerProductFloat4Ext,
        deglib::distances::InnerProductFloat16ExtResiduals,
        deglib::distances::InnerProductFloat4ExtResiduals,
        deglib::distances::L2Uint8,
        deglib::distances::L2Uint8Ext32,
        deglib::distances::L2Uint8Ext16
    >;

    // Compile-time verification that every type in DistanceVariant fulfills the DistanceFunction concept
    static_assert([]<typename... Ts>(std::variant<Ts...>*) {
        return (deglib::DistanceFunction<Ts> && ...);
    }(static_cast<DistanceVariant*>(nullptr)), "All types in DistanceVariant must satisfy DistanceFunction concept");

    /**
     * Extracts the DISTFUNC<float> function pointer from a DistanceVariant object.
     */
    inline DISTFUNC<float> to_dist_func(const DistanceVariant& variant) {
        return std::visit([](auto&& dist) -> DISTFUNC<float> {
            using DistType = std::decay_t<decltype(dist)>;
            static_assert(deglib::DistanceFunction<DistType>, "Selected distance variant must satisfy DistanceFunction concept");
            return &DistType::compare;
        }, variant);
    }

    /**
     * Represents a metric feature space for vector distance computations.
     * Manages dimension, metric type, byte size, and distance evaluation logic.
     */
    class FloatSpace  {

        static DistanceVariant select_dist_variant(const size_t dim, const deglib::Metric metric) {
            if(metric == deglib::Metric::L2) {
                #if defined(USE_SSE42) || defined(USE_AVX2) || defined(USE_AVX512)
                    if (dim % 16 == 0)
                        return deglib::distances::L2Float16Ext{};
                    else if (dim % 8 == 0)
                        return deglib::distances::L2Float8Ext{};
                    else if (dim % 4 == 0)
                        return deglib::distances::L2Float4Ext{};
                    else if (dim > 16)
                        return deglib::distances::L2Float16ExtResiduals{};
                    else if (dim > 4)
                        return deglib::distances::L2Float4ExtResiduals{};
                #else
                    return deglib::distances::L2Float{};
                #endif
            }
            else if(metric == deglib::Metric::InnerProduct) 
            {
                #if defined(USE_SSE42) || defined(USE_AVX2) || defined(USE_AVX512)
                    if (dim % 16 == 0)
                        return deglib::distances::InnerProductFloat16Ext{};
                    else if (dim % 8 == 0)
                        return deglib::distances::InnerProductFloat8Ext{};
                    else if (dim % 4 == 0)
                        return deglib::distances::InnerProductFloat4Ext{};
                    else if (dim > 16)
                        return deglib::distances::InnerProductFloat16ExtResiduals{};
                    else if (dim > 4)
                        return deglib::distances::InnerProductFloat4ExtResiduals{};
                #else
                    return deglib::distances::InnerProductFloat{};
                #endif
            } 
            else if(metric == deglib::Metric::L2_Uint8) 
            {
                #if defined(USE_SSE42) || defined(USE_AVX2) || defined(USE_AVX512)
                    if (dim % 32 == 0)
                        return deglib::distances::L2Uint8Ext32{};
                    else if (dim % 16 == 0)
                        return deglib::distances::L2Uint8Ext16{};
                    else
                        return deglib::distances::L2Uint8{};
                #else
                    return deglib::distances::L2Uint8{};
                #endif
            }
            return deglib::distances::L2Float{};
        }

        static size_t calculate_data_size(const size_t dim, const deglib::Metric metric) {
            return (static_cast<int>(metric) & 0x10) ? dim * sizeof(uint8_t) : dim * sizeof(float);
        }

        const DistanceVariant dist_variant_;
        const size_t data_size_;
        const size_t dim_;
        const deglib::Metric metric_;

    public:
        FloatSpace(const size_t dim, const deglib::Metric metric) 
            : dist_variant_(select_dist_variant(dim, metric)),
              data_size_(calculate_data_size(dim, metric)),
              dim_(dim),
              metric_(metric) {
        }

        /**
         * Returns the dimension of feature vectors in this space.
         */
        const size_t dim() const {
            return dim_;
        }

        /**
         * Returns the metric type used for distance computations.
         */
        const deglib::Metric metric() const {
            return metric_;
        }

        /**
         * Returns the size in bytes of a single feature vector in this space.
         */
        const size_t get_data_size() const {
            return data_size_;
        }

        /**
         * Returns a function pointer to the selected distance comparison function.
         */
        const DISTFUNC<float> get_dist_func() const {
            return to_dist_func(dist_variant_);
        }

        /**
         * Returns the parameter required by the distance function (pointer to vector dimension).
         */
        const void *get_dist_func_param() const {
            return &dim_;
        }

        /**
         * Executes a visitor function with the concrete compile-time DistanceFunction type
         * selected for this feature space (static dispatch).
         */
        template <typename Visitor>
        decltype(auto) compute(Visitor&& visitor) const {
            return std::visit(std::forward<Visitor>(visitor), dist_variant_);
        }

        ~FloatSpace() {}
    };

}  // end namespace deglib
