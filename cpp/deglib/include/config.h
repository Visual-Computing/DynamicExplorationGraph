#pragma once

#include <cstdint>

// Always include the x86 intrinsic headers so that SIMD code in the distance
// headers compiles regardless of the target architecture flags.
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#include <xmmintrin.h>  // for _mm_prefetch
#endif


// ---------------------------------------------------------------------------
// Runtime CPU feature detection
// ---------------------------------------------------------------------------
// Uses CPUID (via __cpuidex on MSVC, __cpuid_count on GCC/Clang) to detect
// SSE4.2, F16C, AVX, AVX2, and AVX-512F at runtime. Results are cached in a
// function-local static so the CPUID query runs only once.
//
// The distance headers compile all SIMD code paths unconditionally (the
// intrinsic headers are always included below). At runtime, select_dist()
// uses deglib::cpu::has_*() to dispatch to the best variant. This means there
// is zero loop overhead — capability checks happen during variant selection,
// not inside distance calculation loops.
// ---------------------------------------------------------------------------

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace deglib::cpu {

    namespace detail {

        // Query CPUID leaf/subleaf into a 4-element int array.
        // Uses __cpuidex on MSVC and __cpuid_count on GCC/Clang.
        inline void cpuid(int leaf, int subleaf, int cpu_info[4]) {
#if defined(_MSC_VER)
            __cpuidex(cpu_info, leaf, subleaf);
#else
            __cpuid_count(leaf, subleaf, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif
        }

        // Cached hardware feature flags, populated on first call via a function-local static.
        struct CpuFeatures {
            bool sse42;
            bool f16c;
            bool avx;
            bool avx2;
            bool avx512f;

            CpuFeatures() {
                int cpu_info[4] = {0};

                // Leaf 1: feature flags in ECX and EDX
                cpuid(1, 0, cpu_info);
                sse42 = (cpu_info[2] & (1 << 20)) != 0;
                f16c  = (cpu_info[2] & (1 << 29)) != 0;
                avx   = (cpu_info[2] & (1 << 28)) != 0;

                // Leaf 7, subleaf 0: extended feature flags in EBX and ECX
                cpuid(7, 0, cpu_info);
                avx2    = (cpu_info[1] & (1 << 5)) != 0;
                avx512f = (cpu_info[1] & (1 << 16)) != 0;
            }
        };

        inline const CpuFeatures& features() {
            static CpuFeatures cached;
            return cached;
        }

    } // namespace detail

    // Runtime CPU feature detection — safe to call from any translation unit.
    // These checks are performed once (cached) and have zero cost per call thereafter.

    inline bool has_sse42()  { return detail::features().sse42; }
    inline bool has_f16c()   { return detail::features().f16c; }
    inline bool has_avx()    { return detail::features().avx; }
    inline bool has_avx2()   { return detail::features().avx2; }
    inline bool has_avx512() { return detail::features().avx512f; }

} // namespace deglib::cpu

