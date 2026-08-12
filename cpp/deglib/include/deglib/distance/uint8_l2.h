#pragma once

#include <variant>
#include "deglib/distance/uint8.h"
#include "deglib/distance/residual_mode.h"

namespace deglib::distances::uint8_l2 {

        // ---------------------------------------------------------------------------------------------------------------------
        // ----------------------------------------------- Uint8 L2 Dists ---------------------------------------------------------
        // ---------------------------------------------------------------------------------------------------------------------

        // Scalar fallback — no SIMD required.
        class L2Uint8 {
        public:
            static constexpr const char* get_instruction() { return "Scalar"; }

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

           inline static void compare_batch(const void *query_ptr, const void * const *db_arr, size_t count, const void *qty_ptr, float *dists) {
               for (size_t i = 0; i < count; ++i) {
                   dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
               }
           }
        };

#if defined(DEGLIB_X86)
        // -------------------------------------------------------------------
        // L2Uint8 SIMD implementations — process vectors with
        // aligned SIMD portions plus scalar residuals for any unaligned tail.
        // Separate classes per SIMD width so that compare() has zero
        // runtime dispatch overhead — select_dist() chooses the class.
        // The HasResidual template parameter controls whether the scalar
        // residual tail loop is compiled in. When HasResidual == false,
        // the residual loop is eliminated at compile time, producing a
        // faster path for dimensions that are known to be SIMD-aligned.


       DEGLIB_TARGET_AVX2 inline static int64_t uint8_l2_hsum256(__m256i s) {
           __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(s, 0), _mm256_extracti128_si256(s, 1));
           alignas(16) int sum_array[4];
           _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
           return static_cast<int64_t>(sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]);
       }

       DEGLIB_TARGET_AVX512 inline static int64_t uint8_l2_hsum512(__m512i s) {
           __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(s), _mm512_extracti64x4_epi64(s, 1));
           return uint8_l2_hsum256(sum256);
       }
        template <ResidualMode Mode = ResidualMode::Full>
        class L2Uint8_AVX512 {
            static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
            static constexpr bool HasSimd     = has_flag(Mode, ResidualMode::Simd);
            static constexpr bool HasTail     = has_flag(Mode, ResidualMode::Tail);

        public:
            static constexpr const char* get_instruction() { return "AVX512"; }

            DEGLIB_TARGET_AVX512 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t size = *((size_t *) qty_ptr);
                const unsigned char *a = (const unsigned char *) pVect1v;
                const unsigned char *b = (const unsigned char *) pVect2v;

                const unsigned char *last = a + size;

                __m512i sum512_1 = _mm512_setzero_si512();
                __m512i sum512_2 = _mm512_setzero_si512();
                if constexpr (HasDualSimd) {
                    while (a + 63 < last) {
                        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                        __m512i diff = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v1), _mm512_cvtepu8_epi16(v2));
                        sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(diff, diff));
                        a += 32;
                        b += 32;
                        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                        __m256i v4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                        __m512i diff2 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v3), _mm512_cvtepu8_epi16(v4));
                        sum512_2 = _mm512_add_epi32(sum512_2, _mm512_madd_epi16(diff2, diff2));
                        a += 32;
                        b += 32;
                    }
                }
                if constexpr (HasSimd) {
                    while (a + 31 < last) {
                        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
                        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
                        __m512i diff = _mm512_sub_epi16(_mm512_cvtepu8_epi16(v1), _mm512_cvtepu8_epi16(v2));
                        sum512_1 = _mm512_add_epi32(sum512_1, _mm512_madd_epi16(diff, diff));
                        a += 32;
                        b += 32;
                    }
                }

                // Horizontal reduce of SIMD accumulators
                __m512i sum512 = _mm512_add_epi32(sum512_1, sum512_2);
                __m256i sum256 = _mm256_add_epi32(_mm512_castsi512_si256(sum512), _mm512_extracti64x4_epi64(sum512, 1));
                __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum256), _mm256_extracti128_si256(sum256, 1));

                alignas(16) int sum_array[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
                int64_t result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

                // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
                if constexpr (HasTail) {
                    while (a < last) {
                        int32_t diff = int32_t(*a++) - int32_t(*b++);
                        result += int64_t(diff) * diff;
                    }
                }

                return static_cast<float>(result);
            }

           DEGLIB_TARGET_AVX512 inline static void compare_batch(const void *query_ptr, const void * const *db_arr, size_t count, const void *qty_ptr, float *dists) {
               static constexpr size_t BATCH_SIZE = 8;
               const unsigned char* query = static_cast<const unsigned char*>(query_ptr);
               const size_t dim = *((const size_t*)qty_ptr);

               auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512 {
                   const size_t nc32 = dim / 32;
                   size_t offset = nc32 * 32;

                   alignas(64) __m512i s[BATCH_SIZE];
                   for (size_t j = 0; j < BATCH_SIZE; ++j) {
                       s[j] = _mm512_setzero_si512();
                   }

                   for (size_t c = 0; c < nc32; ++c) {
                       size_t idx = c * 32;
                       __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&query[idx]));
                       __m512i q_vec = _mm512_cvtepu8_epi16(q_raw);

                       for (size_t j = 0; j < BATCH_SIZE; ++j) {
                           const unsigned char* db_ = static_cast<const unsigned char*>(db[j]);
                           __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[idx]));
                           __m512i diff = _mm512_sub_epi16(q_vec, _mm512_cvtepu8_epi16(r_raw));
                           s[j] = _mm512_add_epi32(s[j], _mm512_madd_epi16(diff, diff));
                       }
                   }

                   for (size_t j = 0; j < BATCH_SIZE; ++j) {
                       out_dists[j] = static_cast<float>(uint8_l2_hsum512(s[j]));
                   }

                   if constexpr (HasSimd) {
                       if (offset + 32 <= dim) {
                           __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&query[offset]));
                           __m512i q_vec = _mm512_cvtepu8_epi16(q_raw);
                           for (size_t j = 0; j < BATCH_SIZE; ++j) {
                               const unsigned char* db_ = static_cast<const unsigned char*>(db[j]);
                               __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[offset]));
                               __m512i diff = _mm512_sub_epi16(q_vec, _mm512_cvtepu8_epi16(r_raw));
                               out_dists[j] += static_cast<float>(uint8_l2_hsum512(_mm512_madd_epi16(diff, diff)));
                           }
                           offset += 32;
                       }
                   }

                   if constexpr (HasTail) {
                       if (offset < dim) {
                           for (size_t j = 0; j < BATCH_SIZE; ++j) {
                               const unsigned char* db_ptr = static_cast<const unsigned char*>(db[j]);
                               int64_t tail_sum = 0;
                               for (size_t k = offset; k < dim; ++k) {
                                   int32_t diff = int32_t(query[k]) - int32_t(db_ptr[k]);
                                   tail_sum += int64_t(diff) * diff;
                               }
                               out_dists[j] += static_cast<float>(tail_sum);
                           }
                       }
                   }
               };

               size_t i = 0;
               for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
                   batch_impl(&db_arr[i], &dists[i]);
               }
               for (; i < count; ++i) {
                   dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
               }
           }
        };

        template <ResidualMode Mode = ResidualMode::Full>
        class L2Uint8_AVX2 {
            static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
            static constexpr bool HasSimd     = has_flag(Mode, ResidualMode::Simd);
            static constexpr bool HasTail     = has_flag(Mode, ResidualMode::Tail);

        public:
            static constexpr const char* get_instruction() { return "AVX2"; }
            DEGLIB_TARGET_AVX2 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t size = *((size_t *) qty_ptr);
                const unsigned char *a = (const unsigned char *) pVect1v;
                const unsigned char *b = (const unsigned char *) pVect2v;

                const unsigned char *last = a + size;

                __m256i sum256_1 = _mm256_setzero_si256();
                __m256i sum256_2 = _mm256_setzero_si256();
                if constexpr (HasDualSimd) {
                    while (a + 31 < last) {
                        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                        __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                        __m256i diff = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v1), _mm256_cvtepu8_epi16(v2));
                        sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(diff, diff));
                        a += 16;
                        b += 16;
                        __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                        __m128i v4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                        __m256i diff2 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v3), _mm256_cvtepu8_epi16(v4));
                        sum256_2 = _mm256_add_epi32(sum256_2, _mm256_madd_epi16(diff2, diff2));
                        a += 16;
                        b += 16;
                    }
                }
                if constexpr (HasSimd) {
                    while (a + 15 < last) {
                        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
                        __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
                        __m256i diff = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v1), _mm256_cvtepu8_epi16(v2));
                        sum256_1 = _mm256_add_epi32(sum256_1, _mm256_madd_epi16(diff, diff));
                        a += 16;
                        b += 16;
                    }
                }

                // Horizontal reduce of SIMD accumulators
                __m256i sum256 = _mm256_add_epi32(sum256_1, sum256_2);
                __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum256, 0), _mm256_extracti128_si256(sum256, 1));

                alignas(16) int sum_array[4];
                _mm_store_si128(reinterpret_cast<__m128i*>(sum_array), sum128);
                int64_t result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
               // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
               if constexpr (HasTail) {
                   while (a < last) {
                       int32_t diff = int32_t(*a++) - int32_t(*b++);
                       result += int64_t(diff) * diff;
                   }
               }

               return static_cast<float>(result);
           }

           DEGLIB_TARGET_AVX2 inline static void compare_batch(const void *query_ptr, const void * const *db_arr, size_t count, const void *qty_ptr, float *dists) {
               static constexpr size_t BATCH_SIZE = 8;
               const unsigned char* query = static_cast<const unsigned char*>(query_ptr);
               const size_t dim = *((const size_t*)qty_ptr);

               auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2 {
                   const size_t nc16 = dim / 16;
                   size_t offset = nc16 * 16;

                   alignas(32) __m256i s[BATCH_SIZE];
                   for (size_t j = 0; j < BATCH_SIZE; ++j) {
                       s[j] = _mm256_setzero_si256();
                   }

                   for (size_t c = 0; c < nc16; ++c) {
                       size_t idx = c * 16;
                       __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[idx]));
                       __m256i q_vec = _mm256_cvtepu8_epi16(q_raw);

                       for (size_t j = 0; j < BATCH_SIZE; ++j) {
                           const unsigned char* db_ = static_cast<const unsigned char*>(db[j]);
                           __m128i r_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[idx]));
                           __m256i diff = _mm256_sub_epi16(q_vec, _mm256_cvtepu8_epi16(r_raw));
                           s[j] = _mm256_add_epi32(s[j], _mm256_madd_epi16(diff, diff));
                       }
                   }

                   for (size_t j = 0; j < BATCH_SIZE; ++j) {
                       out_dists[j] = static_cast<float>(uint8_l2_hsum256(s[j]));
                   }

                   if constexpr (HasSimd) {
                       if (offset + 16 <= dim) {
                           __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[offset]));
                           __m256i q_vec = _mm256_cvtepu8_epi16(q_raw);
                           for (size_t j = 0; j < BATCH_SIZE; ++j) {
                               const unsigned char* db_ = static_cast<const unsigned char*>(db[j]);
                               __m128i r_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[offset]));
                               __m256i diff = _mm256_sub_epi16(q_vec, _mm256_cvtepu8_epi16(r_raw));
                               out_dists[j] += static_cast<float>(uint8_l2_hsum256(_mm256_madd_epi16(diff, diff)));
                           }
                           offset += 16;
                       }
                   }

                   if constexpr (HasTail) {
                       if (offset < dim) {
                           for (size_t j = 0; j < BATCH_SIZE; ++j) {
                               const unsigned char* db_ptr = static_cast<const unsigned char*>(db[j]);
                               int64_t tail_sum = 0;
                               for (size_t k = offset; k < dim; ++k) {
                                   int32_t diff = int32_t(query[k]) - int32_t(db_ptr[k]);
                                   tail_sum += int64_t(diff) * diff;
                               }
                               out_dists[j] += static_cast<float>(tail_sum);
                           }
                       }
                   }
               };

               size_t i = 0;
               for (; i + BATCH_SIZE <= count; i += BATCH_SIZE) {
                   batch_impl(&db_arr[i], &dists[i]);
               }
               for (; i < count; ++i) {
                   dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
               }
           }
       };
#endif

    using DistanceVariant = std::variant<
        L2Uint8
#if defined(DEGLIB_X86)
        , L2Uint8_AVX512<ResidualMode::Full>
        , L2Uint8_AVX512<ResidualMode::DualPlusSimd>
        , L2Uint8_AVX512<ResidualMode::DualTail>
        , L2Uint8_AVX512<ResidualMode::DualOnly>
        , L2Uint8_AVX512<ResidualMode::SimdTail>
        , L2Uint8_AVX512<ResidualMode::SimdOnly>
        , L2Uint8_AVX512<ResidualMode::TailOnly>
        , L2Uint8_AVX2<ResidualMode::Full>
        , L2Uint8_AVX2<ResidualMode::DualPlusSimd>
        , L2Uint8_AVX2<ResidualMode::DualTail>
        , L2Uint8_AVX2<ResidualMode::DualOnly>
        , L2Uint8_AVX2<ResidualMode::SimdTail>
        , L2Uint8_AVX2<ResidualMode::SimdOnly>
        , L2Uint8_AVX2<ResidualMode::TailOnly>
#endif
    >;

    inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
        if (instruction == deglib::cpu::InstructionSet::Scalar) {
            return L2Uint8{};
        }

#if defined(DEGLIB_X86)
        if (instruction == deglib::cpu::InstructionSet::AVX512 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx512())) {
            if (instruction == deglib::cpu::InstructionSet::AVX512 && !deglib::cpu::has_avx512()) {
                throw std::runtime_error("AVX512 instruction set requested, but not supported by CPU");
            }
            if (dim < 32) {
                return L2Uint8_AVX512<ResidualMode::TailOnly>{};
            } else if (dim < 64) {
                if (dim == 32) return L2Uint8_AVX512<ResidualMode::SimdOnly>{};
                else return L2Uint8_AVX512<ResidualMode::SimdTail>{};
            } else {
                if (dim % 64 == 0) return L2Uint8_AVX512<ResidualMode::DualOnly>{};
                else if (dim % 32 == 0) return L2Uint8_AVX512<ResidualMode::DualPlusSimd>{};
                else return L2Uint8_AVX512<ResidualMode::Full>{};
            }
        }
        else if (instruction == deglib::cpu::InstructionSet::AVX2 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx2())) {
            if (instruction == deglib::cpu::InstructionSet::AVX2 && !deglib::cpu::has_avx2()) {
                throw std::runtime_error("AVX2 instruction set requested, but not supported by CPU");
            }
            if (dim < 16) {
                return L2Uint8_AVX2<ResidualMode::TailOnly>{};
            } else if (dim < 32) {
                if (dim == 16) return L2Uint8_AVX2<ResidualMode::SimdOnly>{};
                else return L2Uint8_AVX2<ResidualMode::SimdTail>{};
            } else {
                if (dim % 32 == 0) return L2Uint8_AVX2<ResidualMode::DualOnly>{};
                else if (dim % 16 == 0) return L2Uint8_AVX2<ResidualMode::DualPlusSimd>{};
                else return L2Uint8_AVX2<ResidualMode::Full>{};
            }
        }
#else
        if (instruction != deglib::cpu::InstructionSet::Auto && instruction != deglib::cpu::InstructionSet::Scalar) {
            throw std::runtime_error("Requested SIMD instruction set is not supported on this platform");
        }
#endif

        return L2Uint8{};
    }

} // namespace deglib::distances::uint8_l2
