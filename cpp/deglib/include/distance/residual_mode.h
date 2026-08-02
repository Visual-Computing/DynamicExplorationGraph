#pragma once

#include "config.h"

namespace deglib::distances {

    // Shared residual mode for SIMD distance implementations.
    // Controls which loop variants are compiled into SIMD distance functions.
    // Used by all SIMD-accelerated distance classes (fp32_l2, fp32_ip, fp16_ip, uint8_l2, evp_ip).

    enum class ResidualMode : uint8_t {
        DualOnly         = 0,      // 0x00: Dual SIMD accumulator loop only (legacy <false> compatible)
        DualSimd         = 1 << 0, // 0x01: Dual SIMD accumulator loop (2x vector unroll)
        Simd             = 1 << 1, // 0x02: Single SIMD vector loop (1x vector unroll)
        Tail             = 1 << 2, // 0x04: Scalar tail loop (1 float at a time)

        // Presets:
        TailOnly         = Tail,
        SimdOnly         = Simd,
        SimdTail         = Simd | Tail,
        DualTail         = DualSimd | Tail,
        DualPlusSimd     = DualSimd | Simd,
        Full             = DualSimd | Simd | Tail // Default: all active
    };

    constexpr ResidualMode operator|(ResidualMode a, ResidualMode b) {
        return static_cast<ResidualMode>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    constexpr bool has_flag(ResidualMode mode, ResidualMode flag) {
        if (mode == ResidualMode::DualOnly) {
            return flag == ResidualMode::DualSimd;
        }
        return (static_cast<uint8_t>(mode) & static_cast<uint8_t>(flag)) != 0;
    }

} // end namespace deglib::distances
