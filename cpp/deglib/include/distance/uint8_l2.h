#pragma once

#include "distance/uint8.h"

namespace deglib::distances::uint8_l2 {

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

    using DistanceVariant = std::variant<
        L2Uint8,
        L2Uint8Ext32,
        L2Uint8Ext16
    >;

    inline DistanceVariant select_dist(const size_t dim) {
            #if defined(USE_SSE42) || defined(USE_AVX2) || defined(USE_AVX512)
                if (dim % 32 == 0)
                    return L2Uint8Ext32{};
                else if (dim % 16 == 0)
                    return L2Uint8Ext16{};
            #endif
            return L2Uint8{};
        }

} // namespace deglib::distances::uint8_l2
