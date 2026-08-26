#pragma once

#include "deglib/distance/residual_mode.h"

#include <cmath>
#include <stdexcept>
#include <variant>

namespace deglib::distances::fp32_l2 {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------- Float L2 Dists ---------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------
// Scalar fallback — no SIMD required.
class L2Float {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }

    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);

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
            result = std::fma(diff3, diff3, std::fma(diff2, diff2, std::fma(diff1, diff1, std::fma(diff0, diff0, result))));
            a += 4;
            b += 4;
        }
        // Process last 0-3 elements. Not needed for standard vector lengths.
        while (a < last) {
            diff0 = *a++ - *b++;
            result = std::fma(diff0, diff0, result);
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
// -------------------------------------------------------------------
// L2Float SIMD implementations — process vectors with aligned SIMD
// portions plus scalar residuals for any unaligned tail.
// Separate classes per SIMD width so that compare() has zero
// runtime dispatch overhead — select_dist() chooses the class.
// The HasResidual template parameter controls whether the scalar
// residual tail loop is compiled in. When HasResidual == false,
// the residual loop is eliminated at compile time, producing a
// faster path for dimensions that are known to be SIMD-aligned.
// -------------------------------------------------------------------

DEGLIB_TARGET_AVX2 inline static float fp32_l2_hsum256(__m256 s) {
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

DEGLIB_TARGET_AVX512 inline static float fp32_l2_hsum512(__m512 s) { return _mm512_reduce_add_ps(s); }
template <ResidualMode Mode = ResidualMode::Full>
class L2Float_AVX512 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX512"; }

    DEGLIB_TARGET_AVX512 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);

        const float* last = a + size;

        // Dual accumulator unrolling: two independent FMADD chains
        // hide the latency of _mm512_fmadd_ps on modern CPUs.
        __m512 sum512_1 = _mm512_setzero_ps();
        __m512 sum512_2 = _mm512_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 31 < last) {
                __m512 v1 = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
                sum512_1 = _mm512_fmadd_ps(v1, v1, sum512_1);
                a += 16;
                b += 16;
                __m512 v2 = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
                sum512_2 = _mm512_fmadd_ps(v2, v2, sum512_2);
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
            sum512_1 = _mm512_fmadd_ps(v, v, sum512_1);
            a += 16;
            b += 16;
        }
        __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(32) float f[4];
        _mm_store_ps(f, sum128);
        float result = f[0] + f[1] + f[2] + f[3];

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            while (a < last) {
                float diff = *a++ - *b++;
                result = std::fma(diff, diff, result);
            }
        }

        return result;
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const float* query = static_cast<const float*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512 {
            size_t offset = 0;
            alignas(64) __m512 s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm512_setzero_ps();
            }

            if constexpr (HasDualSimd) {
                const size_t nc16 = dim / 16;
                offset = nc16 * 16;

                for (size_t c = 0; c < nc16; ++c) {
                    size_t idx = c * 16;
                    __m512 q_vec = _mm512_loadu_ps(&query[idx]);

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const float* db_ = static_cast<const float*>(db[j]);
                        __m512 diff = _mm512_sub_ps(q_vec, _mm512_loadu_ps(&db_[idx]));
                        s[j] = _mm512_fmadd_ps(diff, diff, s[j]);
                    }
                }
            }
            if constexpr (HasSimd) {
                __m512 q_vec = _mm512_loadu_ps(&query[offset]);
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const float* db_ = static_cast<const float*>(db[j]);
                    __m512 diff = _mm512_sub_ps(q_vec, _mm512_loadu_ps(&db_[offset]));
                    s[j] = _mm512_fmadd_ps(diff, diff, s[j]);
                }
                offset += 16;
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = fp32_l2_hsum512(s[j]);
            }

            if constexpr (HasTail) {
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const float* db_ptr = static_cast<const float*>(db[j]);
                    float tail_sum = 0.0f;
                    for (size_t k = offset; k < dim; ++k) {
                        float diff = query[k] - db_ptr[k];
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
class L2Float_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }

    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const float* a = static_cast<const float*>(pVect1v);
        const float* b = static_cast<const float*>(pVect2v);
        size_t size = *((size_t*)qty_ptr);

        const float* last = a + size;

        __m256 sum256_1 = _mm256_setzero_ps();
        __m256 sum256_2 = _mm256_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 15 < last) {
                __m256 va1 = _mm256_loadu_ps(a);
                __m256 vb1 = _mm256_loadu_ps(b);
                __m256 diff1 = _mm256_sub_ps(va1, vb1);
                sum256_1 = _mm256_fmadd_ps(diff1, diff1, sum256_1);
                a += 8;
                b += 8;
                __m256 va2 = _mm256_loadu_ps(a);
                __m256 vb2 = _mm256_loadu_ps(b);
                __m256 diff2 = _mm256_sub_ps(va2, vb2);
                sum256_2 = _mm256_fmadd_ps(diff2, diff2, sum256_2);
                a += 8;
                b += 8;
            }
        }
        if constexpr (HasSimd) {
            __m256 va1 = _mm256_loadu_ps(a);
            __m256 vb1 = _mm256_loadu_ps(b);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            sum256_1 = _mm256_fmadd_ps(diff1, diff1, sum256_1);
            a += 8;
            b += 8;
        }

        // Horizontal reduce of SIMD accumulators
        __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
        float result = fp32_l2_hsum256(sum256);

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            while (a < last) {
                float diff = *a++ - *b++;
                result = std::fma(diff, diff, result);
            }
        }

        return result;
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const float* query = static_cast<const float*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2 {
            size_t offset = 0;
            alignas(32) __m256 s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm256_setzero_ps();
            }

            if constexpr (HasDualSimd) {
                const size_t nc8 = dim / 8;
                offset = nc8 * 8;

                for (size_t c = 0; c < nc8; ++c) {
                    size_t idx = c * 8;
                    __m256 q_vec = _mm256_loadu_ps(&query[idx]);

                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const float* db_ = static_cast<const float*>(db[j]);
                        __m256 diff = _mm256_sub_ps(q_vec, _mm256_loadu_ps(&db_[idx]));
                        s[j] = _mm256_fmadd_ps(diff, diff, s[j]);
                    }
                }
            }

            if constexpr (HasSimd) {
                __m256 q_vec = _mm256_loadu_ps(&query[offset]);
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const float* db_ = static_cast<const float*>(db[j]);
                    __m256 diff = _mm256_sub_ps(q_vec, _mm256_loadu_ps(&db_[offset]));
                    s[j] = _mm256_fmadd_ps(diff, diff, s[j]);
                }
                offset += 8;
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = fp32_l2_hsum256(s[j]);
            }

            if constexpr (HasTail) {
                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const float* db_ptr = static_cast<const float*>(db[j]);
                    float tail_sum = 0.0f;
                    for (size_t k = offset; k < dim; ++k) {
                        float diff = query[k] - db_ptr[k];
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
    L2Float
#if defined(DEGLIB_X86)
    ,
    L2Float_AVX512<ResidualMode::Full>,
    L2Float_AVX512<ResidualMode::DualPlusSimd>,
    L2Float_AVX512<ResidualMode::DualTail>,
    L2Float_AVX512<ResidualMode::DualOnly>,
    L2Float_AVX512<ResidualMode::SimdTail>,
    L2Float_AVX512<ResidualMode::SimdOnly>,
    L2Float_AVX512<ResidualMode::TailOnly>,
    L2Float_AVX2<ResidualMode::Full>,
    L2Float_AVX2<ResidualMode::DualPlusSimd>,
    L2Float_AVX2<ResidualMode::DualTail>,
    L2Float_AVX2<ResidualMode::DualOnly>,
    L2Float_AVX2<ResidualMode::SimdTail>,
    L2Float_AVX2<ResidualMode::SimdOnly>,
    L2Float_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto target = deglib::cpu::resolve_instruction_set(instruction);

#if defined(DEGLIB_X86)
    if (target == deglib::cpu::InstructionSet::AVX512 || target == deglib::cpu::InstructionSet::AVX512_VNNI) {
        if (dim < 16) {
            return L2Float_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return L2Float_AVX512<ResidualMode::SimdOnly>{};
            else
                return L2Float_AVX512<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 32;
            if (rem == 0)
                return L2Float_AVX512<ResidualMode::DualOnly>{};
            else if (rem == 16)
                return L2Float_AVX512<ResidualMode::DualPlusSimd>{};
            else if (rem < 16)
                return L2Float_AVX512<ResidualMode::DualTail>{};
            else
                return L2Float_AVX512<ResidualMode::Full>{};
        }
    } else if (target == deglib::cpu::InstructionSet::AVX2 || target == deglib::cpu::InstructionSet::AVX2_VNNI) {
        if (dim < 8) {
            return L2Float_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 16) {
            if (dim == 8)
                return L2Float_AVX2<ResidualMode::SimdOnly>{};
            else
                return L2Float_AVX2<ResidualMode::SimdTail>{};
        } else {
            const size_t rem = dim % 16;
            if (rem == 0)
                return L2Float_AVX2<ResidualMode::DualOnly>{};
            else if (rem == 8)
                return L2Float_AVX2<ResidualMode::DualPlusSimd>{};
            else if (rem < 8)
                return L2Float_AVX2<ResidualMode::DualTail>{};
            else
                return L2Float_AVX2<ResidualMode::Full>{};
        }
    }
#endif

    return L2Float{};
}

}  // namespace deglib::distances::fp32_l2
