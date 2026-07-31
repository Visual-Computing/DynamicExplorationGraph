#pragma once

#include "config.h"

namespace deglib::distances {

    // Shared FP32 distance utilities and declarations.
    // Base header for all FP32 metric modules (fp32_l2.h, fp32_ip.h).

    enum class ResidualMode : uint8_t {
        DualOnly         = 0,      // 0x00 / false: Dual SIMD accumulator loop only (legacy <false> compatible)
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
