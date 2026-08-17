#pragma once

#include "deglib/distance/residual_mode.h"

#include <cmath>
#include <variant>

namespace deglib::distances::fp32_ip {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------------- Float Inner Product Dists ---------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

class InnerProductFloat {
  public:
    static constexpr const char* get_instruction() { return "Scalar"; }
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) { return 1.f - dot(pVect1v, pVect2v, qty_ptr); }
    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        for (size_t i = 0; i < count; ++i) {
            dists[i] = compare(query_ptr, db_arr[i], qty_ptr);
        }
    }

    inline static float dot(const void* pVect1v, const void* pVect2v, const void* qty_ptr) { return ip_naive(pVect1v, pVect2v, qty_ptr); }

    inline static float ip_naive(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);

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
// InnerProductFloat SIMD implementations — process vectors with
// -------------------------------------------------------------------
// InnerProductFloat SIMD implementations — process vectors with
// aligned SIMD portions plus scalar residuals for any unaligned tail.
// Separate classes per SIMD width so that compare() has zero
// runtime dispatch overhead — select_dist() chooses the class.
// The HasResidual template parameter controls whether the scalar
// residual tail loop is compiled in. When HasResidual == false,
// the residual loop is eliminated at compile time, producing a
// faster path for dimensions that are known to be SIMD-aligned.
// -------------------------------------------------------------------

DEGLIB_TARGET_AVX2 inline static float fp32_hsum256(__m256 s) {
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(s, 0), _mm256_extractf128_ps(s, 1));
    alignas(32) float f[4];
    _mm_store_ps(f, sum128);
    return f[0] + f[1] + f[2] + f[3];
}

DEGLIB_TARGET_AVX512 inline static float fp32_hsum512(__m512 s) {
    __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(s, 0), _mm512_extractf32x8_ps(s, 1));
    return fp32_hsum256(sum256);
}

template <ResidualMode Mode = ResidualMode::Full>
class InnerProductFloat_AVX512 {
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

        __m512 sum512_1 = _mm512_setzero_ps();
        __m512 sum512_2 = _mm512_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 31 < last) {
                sum512_1 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512_1);
                a += 16;
                b += 16;
                sum512_2 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512_2);
                a += 16;
                b += 16;
            }
        }
        if constexpr (HasSimd) {
            while (a + 15 < last) {
                sum512_1 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512_1);
                a += 16;
                b += 16;
            }
        }

        // Horizontal reduce of SIMD accumulators
        __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(32) float f[4];
        _mm_store_ps(f, sum128);
        float result = f[0] + f[1] + f[2] + f[3];

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            while (a < last) {
                result = std::fma(*a++, *b++, result);
            }
        }

        return 1.f - result;
    }

    DEGLIB_TARGET_AVX512 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const float* query = static_cast<const float*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX512 {
            const size_t nc16 = dim / 16;
            size_t offset = nc16 * 16;

            alignas(64) __m512 s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm512_setzero_ps();
            }

            for (size_t c = 0; c < nc16; ++c) {
                size_t idx = c * 16;
                __m512 q_vec = _mm512_loadu_ps(&query[idx]);

                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const float* db_ = static_cast<const float*>(db[j]);
                    __m512 db_vec = _mm512_loadu_ps(&db_[idx]);
                    s[j] = _mm512_fmadd_ps(q_vec, db_vec, s[j]);
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = fp32_hsum512(s[j]);
            }

            if constexpr (HasSimd) {
                if (offset + 16 <= dim) {
                    __m512 q_vec = _mm512_loadu_ps(&query[offset]);
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const float* db_ = static_cast<const float*>(db[j]);
                        __m512 db_vec = _mm512_loadu_ps(&db_[offset]);
                        out_dists[j] += fp32_hsum512(_mm512_mul_ps(q_vec, db_vec));
                    }
                    offset += 16;
                }
            }

            if constexpr (HasTail) {
                if (offset < dim) {
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const float* db_ptr = static_cast<const float*>(db[j]);
                        float tail_dot = 0.0f;
                        for (size_t k = offset; k < dim; ++k) {
                            tail_dot = std::fma(query[k], db_ptr[k], tail_dot);
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
class InnerProductFloat_AVX2 {
    static constexpr bool HasDualSimd = has_flag(Mode, ResidualMode::DualSimd);
    static constexpr bool HasSimd = has_flag(Mode, ResidualMode::Simd);
    static constexpr bool HasTail = has_flag(Mode, ResidualMode::Tail);

  public:
    static constexpr const char* get_instruction() { return "AVX2"; }
    DEGLIB_TARGET_AVX2 inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);

        const float* last = a + size;

        __m256 sum256_1 = _mm256_setzero_ps();
        __m256 sum256_2 = _mm256_setzero_ps();
        if constexpr (HasDualSimd) {
            while (a + 15 < last) {
                sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_1);
                a += 8;
                b += 8;
                sum256_2 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_2);
                a += 8;
                b += 8;
            }
        }
        if constexpr (HasSimd) {
            while (a + 7 < last) {
                sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256_1);
                a += 8;
                b += 8;
            }
        }

        // Horizontal reduce of SIMD accumulators
        __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(32) float f[4];
        _mm_store_ps(f, sum128);
        float result = f[0] + f[1] + f[2] + f[3];

        // Scalar residual for the unaligned tail — eliminated at compile-time if HasTail == false
        if constexpr (HasTail) {
            while (a < last) {
                result = std::fma(*a++, *b++, result);
            }
        }

        return 1.f - result;
    }

    DEGLIB_TARGET_AVX2 inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        static constexpr size_t BATCH_SIZE = 8;
        const float* query = static_cast<const float*>(query_ptr);
        const size_t dim = *((const size_t*)qty_ptr);

        auto batch_impl = [query, dim](const void* const* db, float* out_dists) DEGLIB_TARGET_AVX2 {
            const size_t nc8 = dim / 8;
            size_t offset = nc8 * 8;

            alignas(32) __m256 s[BATCH_SIZE];
            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                s[j] = _mm256_setzero_ps();
            }

            for (size_t c = 0; c < nc8; ++c) {
                size_t idx = c * 8;
                __m256 q_vec = _mm256_loadu_ps(&query[idx]);

                for (size_t j = 0; j < BATCH_SIZE; ++j) {
                    const float* db_ = static_cast<const float*>(db[j]);
                    __m256 db_vec = _mm256_loadu_ps(&db_[idx]);
                    s[j] = _mm256_fmadd_ps(q_vec, db_vec, s[j]);
                }
            }

            for (size_t j = 0; j < BATCH_SIZE; ++j) {
                out_dists[j] = fp32_hsum256(s[j]);
            }

            if constexpr (HasSimd) {
                if (offset + 8 <= dim) {
                    __m256 q_vec = _mm256_loadu_ps(&query[offset]);
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const float* db_ = static_cast<const float*>(db[j]);
                        __m256 db_vec = _mm256_loadu_ps(&db_[offset]);
                        out_dists[j] += fp32_hsum256(_mm256_mul_ps(q_vec, db_vec));
                    }
                    offset += 8;
                }
            }

            if constexpr (HasTail) {
                if (offset < dim) {
                    for (size_t j = 0; j < BATCH_SIZE; ++j) {
                        const float* db_ptr = static_cast<const float*>(db[j]);
                        float tail_dot = 0.0f;
                        for (size_t k = offset; k < dim; ++k) {
                            tail_dot = std::fma(query[k], db_ptr[k], tail_dot);
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
    InnerProductFloat
#if defined(DEGLIB_X86)
    ,
    InnerProductFloat_AVX512<ResidualMode::Full>,
    InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>,
    InnerProductFloat_AVX512<ResidualMode::DualTail>,
    InnerProductFloat_AVX512<ResidualMode::DualOnly>,
    InnerProductFloat_AVX512<ResidualMode::SimdTail>,
    InnerProductFloat_AVX512<ResidualMode::SimdOnly>,
    InnerProductFloat_AVX512<ResidualMode::TailOnly>,
    InnerProductFloat_AVX2<ResidualMode::Full>,
    InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>,
    InnerProductFloat_AVX2<ResidualMode::DualTail>,
    InnerProductFloat_AVX2<ResidualMode::DualOnly>,
    InnerProductFloat_AVX2<ResidualMode::SimdTail>,
    InnerProductFloat_AVX2<ResidualMode::SimdOnly>,
    InnerProductFloat_AVX2<ResidualMode::TailOnly>
#endif
    >;

inline DistanceVariant select_dist(const size_t dim, const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    if (instruction == deglib::cpu::InstructionSet::Scalar) {
        return InnerProductFloat{};
    }

#if defined(DEGLIB_X86)
    if (instruction == deglib::cpu::InstructionSet::AVX512 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx512())) {
        if (instruction == deglib::cpu::InstructionSet::AVX512 && !deglib::cpu::has_avx512()) {
            throw std::runtime_error("AVX512 instruction set requested, but not supported by CPU");
        }
        if (dim < 16) {
            return InnerProductFloat_AVX512<ResidualMode::TailOnly>{};
        } else if (dim < 32) {
            if (dim == 16)
                return InnerProductFloat_AVX512<ResidualMode::SimdOnly>{};
            else
                return InnerProductFloat_AVX512<ResidualMode::SimdTail>{};
        } else {
            if (dim % 32 == 0)
                return InnerProductFloat_AVX512<ResidualMode::DualOnly>{};
            else if (dim % 16 == 0)
                return InnerProductFloat_AVX512<ResidualMode::DualPlusSimd>{};
            else
                return InnerProductFloat_AVX512<ResidualMode::Full>{};
        }
    } else if (instruction == deglib::cpu::InstructionSet::AVX2 || (instruction == deglib::cpu::InstructionSet::Auto && deglib::cpu::has_avx2())) {
        if (instruction == deglib::cpu::InstructionSet::AVX2 && !deglib::cpu::has_avx2()) {
            throw std::runtime_error("AVX2 instruction set requested, but not supported by CPU");
        }
        if (dim < 8) {
            return InnerProductFloat_AVX2<ResidualMode::TailOnly>{};
        } else if (dim < 16) {
            if (dim == 8)
                return InnerProductFloat_AVX2<ResidualMode::SimdOnly>{};
            else
                return InnerProductFloat_AVX2<ResidualMode::SimdTail>{};
        } else {
            if (dim % 16 == 0)
                return InnerProductFloat_AVX2<ResidualMode::DualOnly>{};
            else if (dim % 8 == 0)
                return InnerProductFloat_AVX2<ResidualMode::DualPlusSimd>{};
            else
                return InnerProductFloat_AVX2<ResidualMode::Full>{};
        }
    }
#else
    if (instruction != deglib::cpu::InstructionSet::Auto && instruction != deglib::cpu::InstructionSet::Scalar) {
        throw std::runtime_error("Requested SIMD instruction set is not supported on this platform");
    }
#endif

    return InnerProductFloat{};
}

}  // namespace deglib::distances::fp32_ip
