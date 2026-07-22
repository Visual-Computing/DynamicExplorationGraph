#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <config.h>
#include "distance/fp16_inner_product.h"

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace deglib {
namespace distances {

// ---------------------------------------------------------------------------------------------------------------------
// ----------------------------------------- FP16-EVP Asymmetric -------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------

class FP16EvpAsymInnerProduct {
public:
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(USE_AVX512)
        return 1.f - ip_avx512(pVect1v, pVect2v, qty_ptr);
#elif defined(USE_AVX)
    return 1.f - ip_avx2(pVect1v, pVect2v, qty_ptr);
#elif defined(USE_SSE)
    return 1.f - ip_sse(pVect1v, pVect2v, qty_ptr);
#else
        return 1.f - ip_naive(pVect1v, pVect2v, qty_ptr);
#endif
    }

    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        const uint16_t* query = (const uint16_t*)query_ptr;
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;
        size_t idx = 0;
#if defined(USE_AVX512)
        for (; idx + 8 <= count; idx += 8) {
            const std::byte* ones[8], * negs[8];
            for (int j = 0; j < 8; ++j) {
                const std::byte* db = (const std::byte*)db_arr[idx + j];
                ones[j] = db; negs[j] = db + mask_bytes;
            }
            batch8_avx512(query, ones, negs, mask_bytes, dists + idx);
        }
        for (; idx + 4 <= count; idx += 4) {
            const std::byte* ones[4], * negs[4];
            for (int j = 0; j < 4; ++j) {
                const std::byte* db = (const std::byte*)db_arr[idx + j];
                ones[j] = db; negs[j] = db + mask_bytes;
            }
            batch4_avx512(query, ones, negs, mask_bytes, dists + idx);
        }
#elif defined(USE_AVX)
        for (; idx + 8 <= count; idx += 8) {
            const std::byte* ones[8], * negs[8];
            for (int j = 0; j < 8; ++j) {
                const std::byte* db = (const std::byte*)db_arr[idx + j];
                ones[j] = db; negs[j] = db + mask_bytes;
            }
            batch8_avx2(query, ones, negs, mask_bytes, dists + idx);
        }
        for (; idx + 4 <= count; idx += 4) {
            const std::byte* ones[4], * negs[4];
            for (int j = 0; j < 4; ++j) {
                const std::byte* db = (const std::byte*)db_arr[idx + j];
                ones[j] = db; negs[j] = db + mask_bytes;
            }
            batch4_avx2(query, ones, negs, mask_bytes, dists + idx);
        }
#endif
        for (; idx < count; ++idx) {
            dists[idx] = compare(query_ptr, db_arr[idx], qty_ptr);
        }
    }

    inline static float ip_naive(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        float ip = 0.f;
        for (size_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            uint8_t ones_byte = static_cast<uint8_t>(ones_b[byte_idx]);
            uint8_t negs_byte = static_cast<uint8_t>(negs_b[byte_idx]);

            if (ones_byte == 0 && negs_byte == 0) {
                continue;
            }

            for (int bit = 0; bit < 8; ++bit) {
                const size_t i = byte_idx * 8 + static_cast<size_t>(bit);
                if ((ones_byte & (1 << bit)) != 0) {
                    ip += deglib::distances::fp16_to_float(a[i]);
                }
                if ((negs_byte & (1 << bit)) != 0) {
                    ip -= deglib::distances::fp16_to_float(a[i]);
                }
            }
        }

        return ip;
    }

#if defined(USE_SSE)
    inline static float ip_sse(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        struct alignas(16) SseNibbleMaskLut {
            alignas(16) float ones[16][4];
            alignas(16) float negs[16][4];

            constexpr SseNibbleMaskLut() : ones{}, negs{} {
                for (int v = 0; v < 16; ++v) {
                    for (int bit = 0; bit < 4; ++bit) {
                        ones[v][bit] = ((v >> bit) & 1) ? 1.0f : 0.0f;
                        negs[v][bit] = ((v >> bit) & 1) ? 1.0f : 0.0f;
                    }
                }
            }
        };

        static constexpr SseNibbleMaskLut lut{};

        __m128 sum128 = _mm_setzero_ps();

        for (size_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            const uint8_t ones_byte = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t negs_byte = static_cast<uint8_t>(negs_b[byte_idx]);

            const __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            const __m128 a_lo = _mm_cvtph_ps(raw_a);
            const __m128 a_hi = _mm_cvtph_ps(_mm_srli_si128(raw_a, 8));

            const uint8_t o0 = ones_byte & 0x0F;
            const uint8_t n0 = negs_byte & 0x0F;
            const uint8_t o1 = (ones_byte >> 4) & 0x0F;
            const uint8_t n1 = (negs_byte >> 4) & 0x0F;

            const __m128 m0 = _mm_sub_ps(_mm_load_ps(lut.ones[o0]), _mm_load_ps(lut.negs[n0]));
            const __m128 m1 = _mm_sub_ps(_mm_load_ps(lut.ones[o1]), _mm_load_ps(lut.negs[n1]));

            sum128 = _mm_add_ps(sum128, _mm_mul_ps(a_lo, m0));
            sum128 = _mm_add_ps(sum128, _mm_mul_ps(a_hi, m1));
        }

        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    }
#endif

#if defined(USE_AVX)
    inline static float ip_avx2(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        struct alignas(32) Avx2ByteMaskLut {
            alignas(32) float ones[256][8];
            alignas(32) float negs[256][8];

            constexpr Avx2ByteMaskLut() : ones{}, negs{} {
                for (int v = 0; v < 256; ++v) {
                    for (int bit = 0; bit < 8; ++bit) {
                        const float f = ((v >> bit) & 1) ? 1.0f : 0.0f;
                        ones[v][bit] = f;
                        negs[v][bit] = f;
                    }
                }
            }
        };

        static constexpr Avx2ByteMaskLut lut{};

        __m256 sumA = _mm256_setzero_ps();
        __m256 sumB = _mm256_setzero_ps();

        size_t byte_idx = 0;
        for (; byte_idx + 3 < mask_bytes; byte_idx += 4) {
            const uint8_t o0 = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n0 = static_cast<uint8_t>(negs_b[byte_idx]);
            const uint8_t o1 = static_cast<uint8_t>(ones_b[byte_idx + 1]);
            const uint8_t n1 = static_cast<uint8_t>(negs_b[byte_idx + 1]);
            const uint8_t o2 = static_cast<uint8_t>(ones_b[byte_idx + 2]);
            const uint8_t n2 = static_cast<uint8_t>(negs_b[byte_idx + 2]);
            const uint8_t o3 = static_cast<uint8_t>(ones_b[byte_idx + 3]);
            const uint8_t n3 = static_cast<uint8_t>(negs_b[byte_idx + 3]);

            const __m256i raw16_0 = _mm256_loadu_si256((const __m256i*)&a[byte_idx * 8]);
            const __m256i raw16_1 = _mm256_loadu_si256((const __m256i*)&a[(byte_idx + 2) * 8]);

            const __m128i r0 = _mm256_extractf128_si256(raw16_0, 0);
            const __m128i r1 = _mm256_extractf128_si256(raw16_0, 1);
            const __m128i r2 = _mm256_extractf128_si256(raw16_1, 0);
            const __m128i r3 = _mm256_extractf128_si256(raw16_1, 1);

            const __m256 a0 = _mm256_cvtph_ps(r0);
            const __m256 a1 = _mm256_cvtph_ps(r1);
            const __m256 a2 = _mm256_cvtph_ps(r2);
            const __m256 a3 = _mm256_cvtph_ps(r3);

            const __m256 m0 = _mm256_sub_ps(_mm256_load_ps(lut.ones[o0]), _mm256_load_ps(lut.negs[n0]));
            const __m256 m1 = _mm256_sub_ps(_mm256_load_ps(lut.ones[o1]), _mm256_load_ps(lut.negs[n1]));
            const __m256 m2 = _mm256_sub_ps(_mm256_load_ps(lut.ones[o2]), _mm256_load_ps(lut.negs[n2]));
            const __m256 m3 = _mm256_sub_ps(_mm256_load_ps(lut.ones[o3]), _mm256_load_ps(lut.negs[n3]));

            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a0, m0));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a1, m1));
            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a2, m2));
            sumB = _mm256_add_ps(sumB, _mm256_mul_ps(a3, m3));
        }

        for (; byte_idx < mask_bytes; ++byte_idx) {
            const uint8_t o = static_cast<uint8_t>(ones_b[byte_idx]);
            const uint8_t n = static_cast<uint8_t>(negs_b[byte_idx]);
            const __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[byte_idx * 8]);
            const __m256 a_ps = _mm256_cvtph_ps(raw_a);
            const __m256 m = _mm256_sub_ps(_mm256_load_ps(lut.ones[o]), _mm256_load_ps(lut.negs[n]));
            sumA = _mm256_add_ps(sumA, _mm256_mul_ps(a_ps, m));
        }

        const __m256 sum256 = _mm256_add_ps(sumA, sumB);
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    }
#endif

#if defined(USE_AVX512)
    inline static float ip_avx512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        const uint16_t* a = (const uint16_t*)pVect1v;
        const std::byte* b = (const std::byte*)pVect2v;
        const uint32_t dim = *static_cast<const uint32_t*>(qty_ptr);
        const size_t mask_bytes = dim / 8;

        const std::byte* ones_b = b;
        const std::byte* negs_b = b + mask_bytes;

        __m512 sum512 = _mm512_setzero_ps();

        size_t i = 0;
        for (; i + 2 <= mask_bytes; i += 2) {
            __m256i raw_a = _mm256_loadu_si256((const __m256i*)&a[i * 8]);
            __m512 a_ps = _mm512_cvtph_ps(raw_a);

            uint16_t m_ones = *reinterpret_cast<const uint16_t*>(&ones_b[i]);
            uint16_t m_negs = *reinterpret_cast<const uint16_t*>(&negs_b[i]);

            sum512 = _mm512_mask_add_ps(sum512, m_ones, sum512, a_ps);
            sum512 = _mm512_mask_sub_ps(sum512, m_negs, sum512, a_ps);
        }

        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        float ip = f[0] + f[1] + f[2] + f[3];

        if (i < mask_bytes) {
            __m128i raw_a = _mm_loadu_si128((const __m128i*)&a[i * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw_a);

            uint8_t ones_byte = static_cast<uint8_t>(ones_b[i]);
            uint8_t negs_byte = static_cast<uint8_t>(negs_b[i]);

            __m256i shift = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i v_ones = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(ones_byte), shift), _mm256_set1_epi32(1));
            __m256i v_negs = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(negs_byte), shift), _mm256_set1_epi32(1));
            __m256i v_diff = _mm256_sub_epi32(v_ones, v_negs);
            __m256 mask_ps = _mm256_cvtepi32_ps(v_diff);

            __m256 sum256_tail = _mm256_mul_ps(a_ps, mask_ps);
            __m128 sum128_tail = _mm_add_ps(_mm256_extractf128_ps(sum256_tail, 0), _mm256_extractf128_ps(sum256_tail, 1));
            alignas(16) float f_tail[4];
            _mm_store_ps(f_tail, sum128_tail);
            ip += f_tail[0] + f_tail[1] + f_tail[2] + f_tail[3];
        }

        return ip;
    }
#endif

private:
    struct alignas(32) Avx2ByteMaskLut {
        alignas(32) float ones[256][8];
        alignas(32) float negs[256][8];
        constexpr Avx2ByteMaskLut() : ones{}, negs{} {
            for (int v = 0; v < 256; ++v) {
                for (int bit = 0; bit < 8; ++bit) {
                    const float f = ((v >> bit) & 1) ? 1.0f : 0.0f;
                    ones[v][bit] = f;
                    negs[v][bit] = f;
                }
            }
        }
    };

    static float hsum256_ps(__m256 v) {
        __m128 lo = _mm256_extractf128_ps(v, 0);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        alignas(16) float f[4];
        _mm_store_ps(f, sum128);
        return f[0] + f[1] + f[2] + f[3];
    }

    static void batch8_avx2(const uint16_t* a, const std::byte* const* ones_db, const std::byte* const* negs_db, size_t mask_bytes, float* dists) {
        static constexpr Avx2ByteMaskLut lut{};
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        __m256 s4 = _mm256_setzero_ps(), s5 = _mm256_setzero_ps();
        __m256 s6 = _mm256_setzero_ps(), s7 = _mm256_setzero_ps();
        for (size_t i = 0; i < mask_bytes; ++i) {
            __m128i raw = _mm_loadu_si128((const __m128i*)&a[i * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw);
            #define B8(j, reg) do { \
                uint8_t ob = (uint8_t)ones_db[j][i], nb = (uint8_t)negs_db[j][i]; \
                reg = _mm256_add_ps(reg, _mm256_mul_ps(a_ps, \
                    _mm256_sub_ps(_mm256_load_ps(lut.ones[ob]), _mm256_load_ps(lut.negs[nb])))); \
            } while(0)
            B8(0,s0); B8(1,s1); B8(2,s2); B8(3,s3);
            B8(4,s4); B8(5,s5); B8(6,s6); B8(7,s7);
            #undef B8
        }
        dists[0] = 1.f - hsum256_ps(s0); dists[1] = 1.f - hsum256_ps(s1);
        dists[2] = 1.f - hsum256_ps(s2); dists[3] = 1.f - hsum256_ps(s3);
        dists[4] = 1.f - hsum256_ps(s4); dists[5] = 1.f - hsum256_ps(s5);
        dists[6] = 1.f - hsum256_ps(s6); dists[7] = 1.f - hsum256_ps(s7);
    }

    static void batch4_avx2(const uint16_t* a, const std::byte* const* ones_db, const std::byte* const* negs_db, size_t mask_bytes, float* dists) {
        static constexpr Avx2ByteMaskLut lut{};
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        for (size_t i = 0; i < mask_bytes; ++i) {
            __m128i raw = _mm_loadu_si128((const __m128i*)&a[i * 8]);
            __m256 a_ps = _mm256_cvtph_ps(raw);
            #define B4(j, reg) do { \
                uint8_t ob = (uint8_t)ones_db[j][i], nb = (uint8_t)negs_db[j][i]; \
                reg = _mm256_add_ps(reg, _mm256_mul_ps(a_ps, \
                    _mm256_sub_ps(_mm256_load_ps(lut.ones[ob]), _mm256_load_ps(lut.negs[nb])))); \
            } while(0)
            B4(0,s0); B4(1,s1); B4(2,s2); B4(3,s3);
            #undef B4
        }
        dists[0] = 1.f - hsum256_ps(s0); dists[1] = 1.f - hsum256_ps(s1);
        dists[2] = 1.f - hsum256_ps(s2); dists[3] = 1.f - hsum256_ps(s3);
    }

    static void batch8_avx512(const uint16_t* a, const std::byte* const* ones_db, const std::byte* const* negs_db, size_t mask_bytes, float* dists) {
        // AVX-512 batch: same pattern but _mm512_cvtph_ps converts 16 fp16 → 16 floats
        // For now, fall back to AVX2 implementation
        batch8_avx2(a, ones_db, negs_db, mask_bytes, dists);
    }

    static void batch4_avx512(const uint16_t* a, const std::byte* const* ones_db, const std::byte* const* negs_db, size_t mask_bytes, float* dists) {
        batch4_avx2(a, ones_db, negs_db, mask_bytes, dists);
    }

    static float compare_impl_tail(const uint16_t* query, const std::byte* db_ones, const std::byte* db_negs, size_t tail_bytes) {
        // dim is always a multiple of 8 for EVP — no tail needed
        (void)query; (void)db_ones; (void)db_negs; (void)tail_bytes;
        return 0.0f;
    }
};

using FP16EvpAsymmetricSimilarity = FP16EvpAsymInnerProduct;

} // namespace distances
} // namespace deglib
