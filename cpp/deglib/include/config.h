#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DEGLIB_X86 1
#endif

// Target attribute for AVX-512 functions on GCC/Clang
#if defined(DEGLIB_X86) && (defined(__GNUC__) || defined(__clang__))
#define DEGLIB_TARGET_AVX512 __attribute__((target("avx512f,avx512dq,fma")))
#else
#define DEGLIB_TARGET_AVX512
#endif

// Architecture intrinsic headers
#if defined(DEGLIB_X86)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#include <xmmintrin.h>  // for _mm_prefetch
#include <cpuid.h>
#endif
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

namespace deglib::cpu {

    namespace detail {

#if defined(DEGLIB_X86)
        // Query CPUID leaf/subleaf into a 4-element int array.
        // Uses __cpuidex on MSVC and __cpuid_count on GCC/Clang.
        inline void cpuid(int leaf, int subleaf, int cpu_info[4]) {
#if defined(_MSC_VER)
            __cpuidex(cpu_info, leaf, subleaf);
#else
            __cpuid_count(leaf, subleaf, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif
        }
#endif

        // Cached hardware feature flags, populated on first call via a function-local static.
        struct CpuFeatures {
            bool sse42{false};
            bool f16c{false};
            bool avx{false};
            bool avx2{false};
            bool avx512f{false};

            CpuFeatures() {
#if defined(DEGLIB_X86)
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
#endif
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

