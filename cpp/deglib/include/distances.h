#pragma once

#include <config.h>

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
            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;
            #if defined(USE_AVX512)
                __m512 sum512 = _mm512_setzero_ps();
                while (a < last) {
                    __m512 v = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
                    sum512 = _mm512_fmadd_ps(v, v, sum512);
                    a += 16;
                    b += 16;
                }

                __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_AVX)
                
                // TODO two sum and v's to increase throughput
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
            #elif defined(USE_SSE)
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

            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
            #if defined(USE_AVX)
                __m256 sum256 = _mm256_setzero_ps();
                __m256 v;
                while (a < last) {
                    v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                    sum256 = _mm256_fmadd_ps(v, v, sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE)
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
            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
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
            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
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
            #elif defined(USE_AVX)
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
            #elif defined(USE_SSE)
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
            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
            #if defined(USE_AVX)
                __m256 sum256 = _mm256_setzero_ps();
                while (a < last) {
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE)
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
            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
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

            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
                size_t size = *((size_t *) qty_ptr);
                const unsigned char *a = (const unsigned char *) pVect1v;
                const unsigned char *b = (const unsigned char *) pVect2v;

             #if defined(USE_AVX)
            
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

                // __m256i d2_high_vec = _mm256_setzero_si256();
                // __m256i d2_low_vec = _mm256_setzero_si256();
                // for (size_t i = 0; i + 32 <= size; i += 32) {
                //     __m256i a_vec = _mm256_loadu_si256((__m256i const*)(a + i));
                //     __m256i b_vec = _mm256_loadu_si256((__m256i const*)(b + i));

                //     // Sign extend int8 to int16
                //     __m256i a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_vec));
                //     __m256i a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_vec, 1));
                //     __m256i b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_vec));
                //     __m256i b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_vec, 1));

                //     // Subtract and multiply
                //     __m256i d_low = _mm256_sub_epi16(a_low, b_low);
                //     __m256i d_high = _mm256_sub_epi16(a_high, b_high);
                //     __m256i d2_low_part = _mm256_madd_epi16(d_low, d_low);
                //     __m256i d2_high_part = _mm256_madd_epi16(d_high, d_high);

                //     // Accumulate into int32 vectors
                //     d2_low_vec = _mm256_add_epi32(d2_low_vec, d2_low_part);
                //     d2_high_vec = _mm256_add_epi32(d2_high_vec, d2_high_part);
                // }

                // // Accumulate the 32-bit integers from `d2_high_vec` and `d2_low_vec`
                // __m256i d2_vec = _mm256_add_epi32(d2_low_vec, d2_high_vec);
                // __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(d2_vec, 0), _mm256_extracti128_si256(d2_vec, 1));
   
            #elif defined(USE_SSE)

                // __m128i sum128 = _mm_setzero_si128();
                // for (size_t i = 0; i + 8 <= size; i += 8) {
                //     __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
                //     __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));

                //     __m128i v1_lo = _mm_cvtepu8_epi16(v1);
                //     __m128i v2_lo = _mm_cvtepu8_epi16(v2);

                //     __m128i diff_lo = _mm_sub_epi16(v1_lo, v2_lo);
                //     __m128i sqr_lo = _mm_madd_epi16(diff_lo, diff_lo);
                //     sum128 = _mm_add_epi32(sum128, sqr_lo);
                // }

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

            #if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
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

    template <typename MTYPE>
    using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);

    template <typename MTYPE>
    class SpaceInterface
    {
    public:
        virtual const size_t dim() const = 0;
        virtual const deglib::Metric metric() const = 0;
        virtual const size_t get_data_size() const = 0;
        virtual const DISTFUNC<MTYPE> get_dist_func() const = 0;
        virtual const void* get_dist_func_param() const = 0;
		virtual ~SpaceInterface() = default;
    };


    class FloatSpace : public SpaceInterface<float> {

        static DISTFUNC<float> select_dist_func(const size_t dim, const deglib::Metric metric) {
            DISTFUNC<float> distfunc = deglib::distances::L2Float::compare;

            if(metric == deglib::Metric::L2) {
                #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                    if (dim % 16 == 0)
                        distfunc = deglib::distances::L2Float16Ext::compare;
                    else if (dim % 8 == 0)
                        distfunc = deglib::distances::L2Float8Ext::compare;
                    else if (dim % 4 == 0)
                        distfunc = deglib::distances::L2Float4Ext::compare;
                    else if (dim > 16)
                        distfunc = deglib::distances::L2Float16ExtResiduals::compare;
                    else if (dim > 4)
                        distfunc = deglib::distances::L2Float4ExtResiduals::compare;
                #else
                    distfunc = deglib::distances::L2Float::compare;
                #endif
            }
            else if(metric == deglib::Metric::InnerProduct) 
            {
                #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                    if (dim % 16 == 0)
                        distfunc = deglib::distances::InnerProductFloat16Ext::compare;
                    else if (dim % 8 == 0)
                        distfunc = deglib::distances::InnerProductFloat8Ext::compare;
                    else if (dim % 4 == 0)
                        distfunc = deglib::distances::InnerProductFloat4Ext::compare;
                    else if (dim > 16)
                        distfunc = deglib::distances::InnerProductFloat16ExtResiduals::compare;
                    else if (dim > 4)
                        distfunc = deglib::distances::InnerProductFloat4ExtResiduals::compare;
                #else
                    distfunc = deglib::distances::InnerProductFloat::compare;
                #endif
            } 
            else if(metric == deglib::Metric::L2_Uint8) 
            {
                #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                    if (dim % 32 == 0)
                        distfunc = deglib::distances::L2Uint8Ext32::compare;
                    else if (dim % 16 == 0)
                        distfunc = deglib::distances::L2Uint8Ext16::compare;
                    else
                        distfunc = deglib::distances::L2Uint8::compare;
                #else
                    distfunc = deglib::distances::L2Uint8::compare;
                #endif
            }

            // TODO add cosine but convert to a distance = 2 - (cosine + 1)
            // https://www.kaggle.com/cdabakoglu/word-vectors-cosine-similarity
            // https://github.com/yahoojapan/NGT/blob/master/lib/NGT/PrimitiveComparator.h#L431

            return distfunc;
        }

        static size_t calculate_data_size(const size_t dim, const deglib::Metric metric) {
            return (static_cast<int>(metric) & 0x10) ? dim * sizeof(uint8_t) : dim * sizeof(float);
        }

        const DISTFUNC<float> fstdistfunc_;
        const size_t data_size_;
        const size_t dim_;
        const deglib::Metric metric_;

    public:
        FloatSpace(const size_t dim, const deglib::Metric metric) 
            : fstdistfunc_(select_dist_func(dim, metric)), data_size_(calculate_data_size(dim, metric)), dim_(dim), metric_(metric) {
        }

        const size_t dim() const {
            return dim_;
        }

        const deglib::Metric metric() const {
            return metric_;
        }


        const size_t get_data_size() const {
            return data_size_;
        }

        const DISTFUNC<float> get_dist_func() const {
            return fstdistfunc_;
        }

        const void *get_dist_func_param() const {
            return &dim_;
        }

        ~FloatSpace() {}
    };

}  // end namespace deglib
