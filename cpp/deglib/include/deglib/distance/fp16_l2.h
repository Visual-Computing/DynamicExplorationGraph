#pragma once

#include "deglib/distance/fp16.h"
#include "deglib/distance/residual_mode.h"

#include <cmath>
#include <stdexcept>
#include <variant>

namespace deglib::distances::fp16_l2 {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------- FP16 L2 Dists -------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// FP16 vectors are stored as uint16_t arrays (IEEE 754 half-precision bit patterns).
// The L2 distance is computed as the sum of squared differences: sum((a[i] - b[i])^2).
// ---------------------------------------------------------------------------------------------------------------------

// Scalar fallback — no SIMD required.
// Uses std::fma for precise accumulation, converting each FP16 value to float.
class L2FP16 {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = static_cast<const uint16_t*>(pVect1v);
        const uint16_t* b = static_cast<const uint16_t*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        float result = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            float fa = deglib::distances::fp16::fp16_to_float(a[i]);
            float fb = deglib::distances::fp16::fp16_to_float(b[i]);
            float diff = fa - fb;
            result = std::fma(diff, diff, result);
        }
        return result;
    }

    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        for (size_t i = 0; i < count; ++i) {
            dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
        }
    }
};

#if defined(DEGLIB_X86)
DEGLIB_TARGET_AVX2 inline static float fp16_l2_hsum256(__m256 s) {
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

DEGLIB_TARGET_AVX512 inline static float fp16_l2_hsum512(__m512 s) { return _mm512_reduce_add_ps(s); }

// -------------------------------------------------------------------
// L2FP16 SIMD implementations — process vectors with
// aligned SIMD portions plus scalar residuals for any unaligned tail.
// Separate classes per SIMD width so that compare() has zero
// runtime dispatch overhead — select_dist() chooses the class.
// The HasResidual template parameter controls whether the scalar
// residual tail loop is compiled in. When HasResidual == false,
// the residual loop is eliminated at compile time, producing a
// faster path for dimensions that are known to be SIMD-aligned.
// -------------------------------------------------------------------

template <ResidualMode Mode = ResidualMode::Full>
class L2FP16_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = static_cast<const uint16_t*>(pVect1v);
        const uint16_t* b = static_cast<const uint16_t*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        const uint16_t* last = a + size;

        __m512 sum512_1 = _mm512_setzero_ps();
        __m512 sum512_2 = _mm512_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 31 < last) {
                __m512 va1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
                __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
                __m512 diff1 = _mm512_sub_ps(va1, vb1);
                sum512_1 = _mm512_fmadd_ps(diff1, diff1, sum512_1);
                a += 16;
                b += 16;
                __m512 va2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
                __m512 vb2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
                __m512 diff2 = _mm512_sub_ps(va2, vb2);
                sum512_2 = _mm512_fmadd_ps(diff2, diff2, sum512_2);
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            __m512 va1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
            __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
            __m512 diff1 = _mm512_sub_ps(va1, vb1);
            sum512_1 = _mm512_fmadd_ps(diff1, diff1, sum512_1);
            a += 16;
            b += 16;
        }

        // Horizontal reduce of SIMD accumulators
        __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
        float result = fp16_l2_hsum512(sum512);

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        // Vectorized residual tail for arbitrary dimensions
        if constexpr (HasTail) {
            if ((last - a) >= 8) {
                __m256 va8 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
                __m256 vb8 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
                __m256 diff8 = _mm256_sub_ps(va8, vb8);
                result += fp16_l2_hsum256(_mm256_mul_ps(diff8, diff8));
                a += 8;
                b += 8;
            }
            if ((last - a) >= 4) {
                __m128 va4 = _mm_cvtph_ps(_mm_loadu_si64(a));
                __m128 vb4 = _mm_cvtph_ps(_mm_loadu_si64(b));
                __m128 diff4 = _mm_sub_ps(va4, vb4);
                __m128 prod4 = _mm_mul_ps(diff4, diff4);
                prod4 = _mm_add_ps(prod4, _mm_movehl_ps(prod4, prod4));
                prod4 = _mm_add_ss(prod4, _mm_shuffle_ps(prod4, prod4, 1));
                result += _mm_cvtss_f32(prod4);
                a += 4;
                b += 4;
            }
            while (a < last) {
                float fa = deglib::distances::fp16::fp16_to_float(*a++);
                float fb = deglib::distances::fp16::fp16_to_float(*b++);
                float diff = fa - fb;
                result = std::fma(diff, diff, result);
            }
        }

        return result;
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const uint16_t* query = static_cast<const uint16_t*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512 {
            size_t offset = 0;
            alignas(64) __m512 s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm512_setzero_ps();
            }

            if constexpr (HasDualSimd) {
                const size_t nc32 = dim / 32;
                offset = nc32 * 32;

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
                        __m512 diff_lo = _mm512_sub_ps(q_lo, _mm512_cvtph_ps(r_lo_));
                        __m512 diff_hi = _mm512_sub_ps(q_hi, _mm512_cvtph_ps(r_hi_));
                        s[j] = _mm512_fmadd_ps(diff_lo, diff_lo, s[j]);
                        s[j] = _mm512_fmadd_ps(diff_hi, diff_hi, s[j]);
                    }
                }
            }

            if constexpr (HasSimd) {
                __m256i q_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&query[offset]));
                __m512 qf = _mm512_cvtph_ps(q_raw);

                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const uint16_t* db_ = static_cast<const uint16_t*>(db[j]);
                    __m256i r_raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&db_[offset]));
                    __m512 diff = _mm512_sub_ps(qf, _mm512_cvtph_ps(r_raw));
                    s[j] = _mm512_fmadd_ps(diff, diff, s[j]);
                }
                offset += 16;
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = fp16_l2_hsum512(s[j]);
            }

            if constexpr (HasTail) {
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const uint16_t* db_ptr = static_cast<const uint16_t*>(db[j]);
                    float tail_sum = 0.0f;
                    for (size_t k = offset; k < dim; ++k) {
                        float fa = deglib::distances::fp16::fp16_to_float(query[k]);
                        float fb = deglib::distances::fp16::fp16_to_float(db_ptr[k]);
                        float diff = fa - fb;
                        tail_sum = std::fma(diff, diff, tail_sum);
                    }
                    out_dists[j] += tail_sum;
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
class L2FP16_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = static_cast<const uint16_t*>(pVect1v);
        const uint16_t* b = static_cast<const uint16_t*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        const uint16_t* last = a + size;

        __m256 sum256_1 = _mm256_setzero_ps();
        __m256 sum256_2 = _mm256_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 15 < last) {
                __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
                __m256 vb1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
                __m256 diff1 = _mm256_sub_ps(va1, vb1);
                sum256_1 = _mm256_fmadd_ps(diff1, diff1, sum256_1);
                a += 8;
                b += 8;
                __m256 va2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
                __m256 vb2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
                __m256 diff2 = _mm256_sub_ps(va2, vb2);
                sum256_2 = _mm256_fmadd_ps(diff2, diff2, sum256_2);
                a += 8;
                b += 8;
            }
        }
        if constexpr (HasSimd) {
            __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
            __m256 vb1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            sum256_1 = _mm256_fmadd_ps(diff1, diff1, sum256_1);
            a += 8;
            b += 8;
        }

        // Horizontal reduce of SIMD accumulators
        __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
        float result = fp16_l2_hsum256(sum256);

        // Vectorized residual tail for arbitrary dimensions
        if constexpr (HasTail) {
            if ((last - a) >= 4) {
                __m128 va4 = _mm_cvtph_ps(_mm_loadu_si64(a));
                __m128 vb4 = _mm_cvtph_ps(_mm_loadu_si64(b));
                __m128 diff4 = _mm_sub_ps(va4, vb4);
                __m128 prod4 = _mm_mul_ps(diff4, diff4);
                prod4 = _mm_add_ps(prod4, _mm_movehl_ps(prod4, prod4));
                prod4 = _mm_add_ss(prod4, _mm_shuffle_ps(prod4, prod4, 1));
                result += _mm_cvtss_f32(prod4);
                a += 4;
                b += 4;
            }
            while (a < last) {
                float fa = deglib::distances::fp16::fp16_to_float(*a++);
                float fb = deglib::distances::fp16::fp16_to_float(*b++);
                float diff = fa - fb;
                result = std::fma(diff, diff, result);
            }
        }

        return result;
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const uint16_t* query = static_cast<const uint16_t*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2 {
            size_t offset = 0;
            alignas(32) __m256 s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm256_setzero_ps();
            }

            if constexpr (HasDualSimd) {
                const size_t nc16 = dim / 16;
                offset = nc16 * 16;

                for (size_t c = 0; c < nc16; ++c) {
                    size_t idx = c * 16;
                    __m128i q_raw_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[idx]));
                    __m128i q_raw_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[idx + 8]));
                    __m256 qf_lo = _mm256_cvtph_ps(q_raw_lo);
                    __m256 qf_hi = _mm256_cvtph_ps(q_raw_hi);

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const uint16_t* db_ = static_cast<const uint16_t*>(db[j]);
                        __m128i r_lo_ = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[idx]));
                        __m128i r_hi_ = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[idx + 8]));
                        __m256 diff_lo = _mm256_sub_ps(qf_lo, _mm256_cvtph_ps(r_lo_));
                        __m256 diff_hi = _mm256_sub_ps(qf_hi, _mm256_cvtph_ps(r_hi_));
                        s[j] = _mm256_fmadd_ps(diff_lo, diff_lo, s[j]);
                        s[j] = _mm256_fmadd_ps(diff_hi, diff_hi, s[j]);
                    }
                }
            }

            if constexpr (HasSimd) {
                __m128i q_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&query[offset]));
                __m256 qf = _mm256_cvtph_ps(q_raw);
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const uint16_t* db_ = static_cast<const uint16_t*>(db[j]);
                    __m128i r_raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&db_[offset]));
                    __m256 diff = _mm256_sub_ps(qf, _mm256_cvtph_ps(r_raw));
                    s[j] = _mm256_fmadd_ps(diff, diff, s[j]);
                }
                offset += 8;
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = fp16_l2_hsum256(s[j]);
            }

            if constexpr (HasTail) {
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const uint16_t* db_ptr = static_cast<const uint16_t*>(db[j]);
                    float tail_sum = 0.0f;
                    for (size_t k = offset; k < dim; ++k) {
                        float fa = deglib::distances::fp16::fp16_to_float(query[k]);
                        float fb = deglib::distances::fp16::fp16_to_float(db_ptr[k]);
                        float diff = fa - fb;
                        tail_sum = std::fma(diff, diff, tail_sum);
                    }
                    out_dists[j] += tail_sum;
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
    L2FP16
#if defined(DEGLIB_X86)
    ,
    L2FP16_AVX512<ResidualMode::Full>,
    L2FP16_AVX512<ResidualMode::DualPlusSimd>,
    L2FP16_AVX512<ResidualMode::DualTail>,
    L2FP16_AVX512<ResidualMode::DualOnly>,
    L2FP16_AVX512<ResidualMode::SimdTail>,
    L2FP16_AVX512<ResidualMode::SimdOnly>,
    L2FP16_AVX512<ResidualMode::TailOnly>,
    L2FP16_AVX2<ResidualMode::Full>,
    L2FP16_AVX2<ResidualMode::DualPlusSimd>,
    L2FP16_AVX2<ResidualMode::DualTail>,
    L2FP16_AVX2<ResidualMode::DualOnly>,
    L2FP16_AVX2<ResidualMode::SimdTail>,
    L2FP16_AVX2<ResidualMode::SimdOnly>,
    L2FP16_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 16) {
            return L2FP16_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return L2FP16_AVX512<ResidualMode::SimdOnly>{};
            else
                return L2FP16_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 32;
            if (rem == 0)
                return L2FP16_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 16)
                return L2FP16_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 16)
                return L2FP16_AVX512<ResidualMode::DualTail>{};
            else
                return L2FP16_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 8) {
            return L2FP16_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 16) {
            if (dim == 8)
                return L2FP16_AVX2<ResidualMode::SimdOnly>{};
            else
                return L2FP16_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 16;
            if (rem == 0)
                return L2FP16_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 8)
                return L2FP16_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 8)
                return L2FP16_AVX2<ResidualMode::DualTail>{};
            else
                return L2FP16_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return L2FP16{};
}

}  // namespace deglib::distances::fp16_l2
