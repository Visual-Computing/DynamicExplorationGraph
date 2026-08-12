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
        inline static void compare_batch(const void *query_ptr, const void * const *db_arr, size_t count, const void *qty_ptr, float *dists) {
            for (size_t i = 0; i < count; ++i) {
                dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
            }
        }
    };

#if defined(DEGLIB_X86)
    DEGLIB_TARGET_AVX2 inline static float fp16_hsum256(__m256 s) {
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(s, 0), _mm256_extractf128_ps(s, 1));
        alignas(32) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    }

    DEGLIB_TARGET_AVX512 inline static float fp16_hsum512(__m512 s) {
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(s, 0), _mm512_extractf32x8_ps(s, 1));
        return fp16_hsum256(sum256);
    }

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
            float result = fp16_hsum512(sum512);

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

        DEGLIB_TARGET_AVX512 inline static void compare_batch(const void *query_ptr, const void * const *db_arr, size_t count, const void *qty_ptr, float *dists) {
            static constexpr size_t BATCH_SIZE = 8;
            const uint16_t* query = static_cast<const uint16_t*>(query_ptr);
            const size_t dim = *((const size_t*)qty_ptr);

            auto batch_impl = [query, dim](const void* const* db, float* out_dists) {
                const size_t nc32 = dim / 32;
                size_t offset = nc32 * 32;

                alignas(64) __m512 s[BATCH_SIZE];
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    s[j] = _mm512_setzero_ps();
                }

                for (size_t c = 0; c < nc32; ++c) {
                    size_t idx = c * 32;
                    __m256i q_raw_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&query[idx]));
                    __m256i q_raw_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&query[idx + 16]));
                    __m512 q_lo = _mm512_cvtph_ps(q_raw_lo);
                    __m512 q_hi = _mm512_cvtph_ps(q_raw_hi);

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const uint16_t* db_ = static_cast<const uint16_t*>(db[j]);
                        __m256i r_lo_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[idx]));
                        __m256i r_hi_ = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[idx + 16]));
                        s[j] = _mm512_fmadd_ps(q_lo, _mm512_cvtph_ps(r_lo_), s[j]);
                        s[j] = _mm512_fmadd_ps(q_hi, _mm512_cvtph_ps(r_hi_), s[j]);
                    }
                }

                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    out_dists[j] = fp16_hsum512(s[j]);
                }

                if constexpr (HasSimd) {
                    if (offset + 16 <= dim) {
                        __m128i q16_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[offset]));
                        __m128i q16_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[offset + 8]));
                        __m256 qf_lo = _mm256_cvtph_ps(q16_lo);
                        __m256 qf_hi = _mm256_cvtph_ps(q16_hi);

                        for (size_t j = 0; j < BATCH_SIZE; ++j) {
                            const uint16_t* db_ = static_cast<const uint16_t*>(db[j]);
                            __m128i r_lo_ = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[offset]));
                            __m128i r_hi_ = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[offset + 8]));
                            __m256 t_lo = _mm256_fmadd_ps(qf_lo, _mm256_cvtph_ps(r_lo_), _mm256_setzero_ps());
                            __m256 t_hi = _mm256_fmadd_ps(qf_hi, _mm256_cvtph_ps(r_hi_), _mm256_setzero_ps());
                            out_dists[j] += fp16_hsum256(_mm256_add_ps(t_lo, t_hi));
                        }
                        offset += 16;
                    }
                }

                if constexpr (HasTail) {
                    if (offset < dim) {
                        for (size_t j = 0; j < BATCH_SIZE; ++j) {
                            const uint16_t* db_ptr = static_cast<const uint16_t*>(db[j]);
                            float tail_dot = 0.0f;
                            for (size_t k = offset; k < dim; ++k) {
                                float fa = deglib::distances::fp16::fp16_to_float(query[k]);
                                float fb = deglib::distances::fp16::fp16_to_float(db_ptr[k]);
                                tail_dot = std::fma(fa, fb, tail_dot);
                            }
                            out_dists[j] += tail_dot;
                        }
                    }
                }

                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    out_dists[j] = 1.0f - out_dists[j];
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
            float result = fp16_hsum256(sum256);

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

        DEGLIB_TARGET_AVX2 inline static void compare_batch(const void *query_ptr, const void * const *db_arr, size_t count, const void *qty_ptr, float *dists) {
            static constexpr size_t BATCH_SIZE = 8;
            const uint16_t* query = static_cast<const uint16_t*>(query_ptr);
            const size_t dim = *((const size_t*)qty_ptr);

            auto batch_impl = [query, dim](const void* const* db, float* out_dists) {
                const size_t nc = dim / 16;
                size_t offset = nc * 16;

                alignas(32) __m256 s[BATCH_SIZE];
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    s[j] = _mm256_setzero_ps();
                }

                for (size_t c = 0; c < nc; ++c) {
                    size_t idx = c * 16;
                    __m128i q_raw_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[idx]));
                    __m128i q_raw_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[idx + 8]));
                    __m256 q_lo = _mm256_cvtph_ps(q_raw_lo);
                    __m256 q_hi = _mm256_cvtph_ps(q_raw_hi);

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const uint16_t* db_ = static_cast<const uint16_t*>(db[j]);
                        __m128i r_lo_ = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[idx]));
                        __m128i r_hi_ = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[idx + 8]));
                        s[j] = _mm256_fmadd_ps(q_lo, _mm256_cvtph_ps(r_lo_), s[j]);
                        s[j] = _mm256_fmadd_ps(q_hi, _mm256_cvtph_ps(r_hi_), s[j]);
                    }
                }

                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    out_dists[j] = fp16_hsum256(s[j]);
                }

                if constexpr (HasSimd) {
                    if (offset + 8 <= dim) {
                        __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[offset]));
                        __m256 q_vec = _mm256_cvtph_ps(q_raw);
                        for (size_t j = 0; j < BATCH_SIZE; ++j) {
                            const uint16_t* db_ptr = static_cast<const uint16_t*>(db[j]);
                            __m128i r_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_ptr[offset]));
                            __m256 r_vec = _mm256_cvtph_ps(r_raw);
                            __m256 prod = _mm256_mul_ps(q_vec, r_vec);
                            out_dists[j] += fp16_hsum256(prod);
                        }
                        offset += 8;
                    }
                }

                if constexpr (HasTail) {
                    if (offset < dim) {
                        for (size_t j = 0; j < BATCH_SIZE; ++j) {
                            const uint16_t* db_ptr = static_cast<const uint16_t*>(db[j]);
                            float tail_dot = 0.0f;
                            for (size_t k = offset; k < dim; ++k) {
                                float fa = deglib::distances::fp16::fp16_to_float(query[k]);
                                float fb = deglib::distances::fp16::fp16_to_float(db_ptr[k]);
                                tail_dot = std::fma(fa, fb, tail_dot);
                            }
                            out_dists[j] += tail_dot;
                        }
                    }
                }

                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    out_dists[j] = 1.0f - out_dists[j];
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
