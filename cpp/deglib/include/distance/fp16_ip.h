#pragma once

#include <cmath>
#include "distance/fp16.h"

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
        inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            return 1.f - dot(pVect1v, pVect2v, qty_ptr);
        }

        inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            return deglib::distances::fp16::fp16_ip_naive(pVect1v, pVect2v, qty_ptr);
        }
    };

#if defined(DEGLIB_X86)
    // -------------------------------------------------------------------
    // InnerProductFP16_32Ext — processes 32 FP16 values (64 bytes) per iteration.
    // Uses AVX-512 F16C (_mm512_cvtph_ps) to convert 32 half-precision
    // values to 32 single-precision floats in one instruction, then
    // accumulates with _mm512_fmadd_ps.
    // -------------------------------------------------------------------

    class InnerProductFP16_32Ext_AVX512 {
    public:
        DEGLIB_TARGET_AVX512 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            return 1.f - dot(pVect1v, pVect2v, qty_ptr);
        }

        DEGLIB_TARGET_AVX512 inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            const uint16_t *a = static_cast<const uint16_t *>(pVect1v);
            const uint16_t *b = static_cast<const uint16_t *>(pVect2v);
            size_t size = *((size_t *) qty_ptr);

            const uint16_t *last = a + size;

            __m512 sum512_1 = _mm512_setzero_ps();
            __m512 sum512_2 = _mm512_setzero_ps();
            while (a < last) {
                __m512 va1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a)));
                __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b)));
                sum512_1 = _mm512_fmadd_ps(va1, vb1, sum512_1);
                a += 16;
                b += 16;

                if (a < last) {
                    __m512 va2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a)));
                    __m512 vb2 = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b)));
                    sum512_2 = _mm512_fmadd_ps(va2, vb2, sum512_2);
                    a += 16;
                    b += 16;
                }
            }

            __m512 sum512 = _mm512_add_ps(sum512_1, sum512_2);
            __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(32) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        }
    };

    // -------------------------------------------------------------------
    // InnerProductFP16_16Ext — processes 16 FP16 values (32 bytes) per iteration.
    // Uses AVX2 F16C (_mm256_cvtph_ps) to convert 16 half-precision
    // values to 16 single-precision floats, then accumulates with
    // _mm256_fmadd_ps.
    // -------------------------------------------------------------------

    class InnerProductFP16_16Ext_AVX2 {
    public:
        inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            return 1.f - dot(pVect1v, pVect2v, qty_ptr);
        }

        inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            const uint16_t *a = static_cast<const uint16_t *>(pVect1v);
            const uint16_t *b = static_cast<const uint16_t *>(pVect2v);
            size_t size = *((size_t *) qty_ptr);

            const uint16_t *last = a + size;

            __m256 sum256_1 = _mm256_setzero_ps();
            __m256 sum256_2 = _mm256_setzero_ps();
            while (a < last) {
                __m256 va1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a)));
                __m256 vb1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
                sum256_1 = _mm256_fmadd_ps(va1, vb1, sum256_1);
                a += 8;
                b += 8;

                if (a < last) {
                    __m256 va2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a)));
                    __m256 vb2 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
                    sum256_2 = _mm256_fmadd_ps(va2, vb2, sum256_2);
                    a += 8;
                    b += 8;
                }
            }
            __m256 sum256 = _mm256_add_ps(sum256_1, sum256_2);
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            alignas(32) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        }
    };

    // -------------------------------------------------------------------
    // InnerProductFP16_8Ext — processes 8 FP16 values (16 bytes) per iteration.
    // Uses SSE F16C (_mm_cvtph_ps) to convert 8 half-precision values to
    // 8 single-precision floats, then accumulates with _mm256_fmadd_ps
    // via two SSE loads combined into an AVX2 register.
    // -------------------------------------------------------------------

    class InnerProductFP16_8Ext_SSE {
    public:
        inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            return 1.f - dot(pVect1v, pVect2v, qty_ptr);
        }

        inline static float dot(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            const uint16_t *a = static_cast<const uint16_t *>(pVect1v);
            const uint16_t *b = static_cast<const uint16_t *>(pVect2v);
            size_t size = *((size_t *) qty_ptr);

            const uint16_t *last = a + size;

            __m128 sum128 = _mm_setzero_ps();
            while (a < last) {
                __m128 va = _mm_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a)));
                __m128 vb = _mm_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
                sum128 = _mm_fmadd_ps(va, vb, sum128);
                a += 8;
                b += 8;
            }

            alignas(32) float f[4];
            _mm_store_ps(f, sum128);
            return f[0] + f[1] + f[2] + f[3];
        }
    };

    // -------------------------------------------------------------------
    // Residual classes — process the aligned portion with the SIMD
    // variant and the remainder with the scalar fallback.
    // -------------------------------------------------------------------

    class InnerProductFP16_32ExtResiduals_AVX512 {
    public:
        DEGLIB_TARGET_AVX512 inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            size_t qty = *((size_t *) qty_ptr);

            size_t qty32 = qty >> 5 << 5;
            float res = InnerProductFP16_32Ext_AVX512::dot(pVect1v, pVect2v, &qty32);
            const uint16_t *pVect1 = static_cast<const uint16_t *>(pVect1v) + qty32;
            const uint16_t *pVect2 = static_cast<const uint16_t *>(pVect2v) + qty32;

            size_t qty_left = qty - qty32;
            float res_tail = InnerProductFP16::dot(pVect1, pVect2, &qty_left);
            return 1.f - (res + res_tail);
        }
    };

    class InnerProductFP16_16ExtResiduals_AVX2 {
    public:
        inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            size_t qty = *((size_t *) qty_ptr);

            size_t qty16 = qty >> 4 << 4;
            float res = InnerProductFP16_16Ext_AVX2::dot(pVect1v, pVect2v, &qty16);
            const uint16_t *pVect1 = static_cast<const uint16_t *>(pVect1v) + qty16;
            const uint16_t *pVect2 = static_cast<const uint16_t *>(pVect2v) + qty16;

            size_t qty_left = qty - qty16;
            float res_tail = InnerProductFP16::dot(pVect1, pVect2, &qty_left);
            return 1.f - (res + res_tail);
        }
    };

    class InnerProductFP16_8ExtResiduals_SSE {
    public:
        inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
            size_t qty = *((size_t *) qty_ptr);

            size_t qty8 = qty >> 3 << 3;
            float res = InnerProductFP16_8Ext_SSE::dot(pVect1v, pVect2v, &qty8);
            const uint16_t *pVect1 = static_cast<const uint16_t *>(pVect1v) + qty8;
            const uint16_t *pVect2 = static_cast<const uint16_t *>(pVect2v) + qty8;

            size_t qty_left = qty - qty8;
            float res_tail = InnerProductFP16::dot(pVect1, pVect2, &qty_left);
            return 1.f - (res + res_tail);
        }
    };
#endif

    using DistanceVariant = std::variant<
        InnerProductFP16
#if defined(DEGLIB_X86)
        , InnerProductFP16_32Ext_AVX512,
        InnerProductFP16_16Ext_AVX2,
        InnerProductFP16_8Ext_SSE,
        InnerProductFP16_32ExtResiduals_AVX512,
        InnerProductFP16_16ExtResiduals_AVX2,
        InnerProductFP16_8ExtResiduals_SSE
#endif
    >;

    inline DistanceVariant select_dist(const size_t dim) {
#if defined(DEGLIB_X86)
        if (deglib::cpu::has_avx512()) {
            if (dim % 32 == 0)
                return InnerProductFP16_32Ext_AVX512{};
            else if (dim % 16 == 0)
                return InnerProductFP16_16Ext_AVX2{};
            else if (dim % 8 == 0)
                return InnerProductFP16_8Ext_SSE{};
            else if (dim > 32)
                return InnerProductFP16_32ExtResiduals_AVX512{};
            else if (dim > 8)
                return InnerProductFP16_8ExtResiduals_SSE{};
        }
        else if (deglib::cpu::has_avx2()) {
            if (dim % 16 == 0)
                return InnerProductFP16_16Ext_AVX2{};
            else if (dim % 8 == 0)
                return InnerProductFP16_8Ext_SSE{};
            else if (dim > 16)
                return InnerProductFP16_16ExtResiduals_AVX2{};
            else if (dim > 8)
                return InnerProductFP16_8ExtResiduals_SSE{};
        }
        else if (deglib::cpu::has_sse42()) {
            if (dim % 8 == 0)
                return InnerProductFP16_8Ext_SSE{};
            else if (dim > 8)
                return InnerProductFP16_8ExtResiduals_SSE{};
        }
#endif
        return InnerProductFP16{};
    }

} // namespace deglib::distances::fp16_ip
