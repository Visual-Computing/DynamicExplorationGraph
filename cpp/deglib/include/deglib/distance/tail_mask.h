#pragma once

#include "deglib/config.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(DEGLIB_X86)

namespace deglib::distances::tail_mask {

// Shift-UB-safe AVX-512 bitmasks for tail lengths in [0, W].
inline constexpr __mmask64 mask64(size_t rem) noexcept {
    return rem >= 64 ? ~__mmask64{0} : static_cast<__mmask64>((uint64_t{1} << rem) - 1);
}

inline constexpr __mmask32 mask32(size_t rem) noexcept {
    return rem >= 32 ? ~__mmask32{0} : static_cast<__mmask32>((uint32_t{1} << rem) - 1);
}

inline constexpr __mmask16 mask16(size_t rem) noexcept {
    return rem >= 16 ? ~__mmask16{0} : static_cast<__mmask16>((uint32_t{1} << rem) - 1);
}

namespace detail {

struct alignas(32) MaskTablePS {
    int32_t rows[9][8];
};

consteval MaskTablePS make_mask_table_ps() noexcept {
    MaskTablePS t{};
    for (size_t rem = 0; rem <= 8; ++rem) {
        for (size_t lane = 0; lane < 8; ++lane) {
            t.rows[rem][lane] = (lane < rem) ? -1 : 0;
        }
    }
    return t;
}

struct alignas(32) OverlapMaskTable32 {
    uint8_t rows[33][32];
};

consteval OverlapMaskTable32 make_overlap_mask_table_32() noexcept {
    OverlapMaskTable32 t{};
    for (size_t rem = 0; rem <= 32; ++rem) {
        const size_t zero_prefix = 32 - rem;
        for (size_t i = zero_prefix; i < 32; ++i) {
            t.rows[rem][i] = 0xFF;
        }
    }
    return t;
}

struct alignas(16) OverlapMaskTable16 {
    uint8_t rows[17][16];
};

consteval OverlapMaskTable16 make_overlap_mask_table_16() noexcept {
    OverlapMaskTable16 t{};
    for (size_t rem = 0; rem <= 16; ++rem) {
        const size_t zero_prefix = 16 - rem;
        for (size_t i = zero_prefix; i < 16; ++i) {
            t.rows[rem][i] = 0xFF;
        }
    }
    return t;
}

inline constexpr MaskTablePS kMaskTablePS = make_mask_table_ps();
inline constexpr OverlapMaskTable32 kOverlapMask32 = make_overlap_mask_table_32();
inline constexpr OverlapMaskTable16 kOverlapMask16 = make_overlap_mask_table_16();

}  // namespace detail

DEGLIB_TARGET_AVX2 inline __m256i load_mask_ps(size_t rem) noexcept {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(detail::kMaskTablePS.rows[rem]));
}

DEGLIB_TARGET_AVX2 inline __m256i load_overlap_mask_32(size_t rem) noexcept {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(detail::kOverlapMask32.rows[rem]));
}

DEGLIB_TARGET_AVX2 inline __m128i load_overlap_mask_16(size_t rem) noexcept {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(detail::kOverlapMask16.rows[rem]));
}

/**
 * Loads `rem` remaining bytes (0 <= rem <= 32) zero-extended into a 256-bit register.
 * When `total_bytes >= 32`, uses an in-bounds overlapping load ending at `cur + rem`
 * masked by `kOverlapMask32[rem]` (single unaligned load + bitwise AND).
 * Otherwise falls back to a zero-padded stack copy to avoid reading before the buffer.
 */
DEGLIB_TARGET_AVX2 inline __m256i load_tail_bytes_32(const void* cur, size_t rem, size_t total_bytes) noexcept {
    if (total_bytes >= 32) {
        const auto* p = (static_cast<const uint8_t*>(cur) + rem) - 32;
        return _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), load_overlap_mask_32(rem));
    }
    alignas(32) uint8_t buf[32] = {};
    std::memcpy(buf, cur, rem);
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(buf));
}

/**
 * Loads `rem` remaining bytes (0 <= rem <= 16) zero-extended into a 128-bit register.
 * When `total_bytes >= 16`, uses an in-bounds overlapping load ending at `cur + rem`
 * masked by `kOverlapMask16[rem]`. Otherwise uses a zero-padded stack copy.
 */
DEGLIB_TARGET_AVX2 inline __m128i load_tail_bytes_16(const void* cur, size_t rem, size_t total_bytes) noexcept {
    if (total_bytes >= 16) {
        const auto* p = (static_cast<const uint8_t*>(cur) + rem) - 16;
        return _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)), load_overlap_mask_16(rem));
    }
    alignas(16) uint8_t buf[16] = {};
    std::memcpy(buf, cur, rem);
    return _mm_load_si128(reinterpret_cast<const __m128i*>(buf));
}

}  // namespace deglib::distances::tail_mask

#endif  // DEGLIB_X86
