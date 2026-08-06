#pragma once

#include <cstdint>

// Architecture flags. Define DEGLIB_X86 if the code is for x86 machines. ARM often has separate coding pathes.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DEGLIB_X86 1
#endif

// Compile methods with this attribute for AVX-512 functions on GCC/Clang
#if defined(DEGLIB_X86) && (defined(__GNUC__) || defined(__clang__))
#define DEGLIB_TARGET_AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vpopcntdq,fma")))
#else
#define DEGLIB_TARGET_AVX512
#endif

// Compile methods with this attribute for AVX2 F16C functions on GCC/Clang.
// Covers AVX2, F16C, and FMA intrinsics used in fp16_ip.h and fp32_ip.h.
// F16C is assumed available on all CPUs supporting AVX2.
#if defined(DEGLIB_X86) && (defined(__GNUC__) || defined(__clang__))
#define DEGLIB_TARGET_AVX2 __attribute__((target("avx2,f16c,fma")))
#else
#define DEGLIB_TARGET_AVX2
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
// AVX2 and AVX-512F at runtime. Results are cached in a function-local static
// so the CPUID query runs only once.
//
// AVX2 is the minimum x86 baseline requirement. F16C is always available on
// CPUs that support AVX2, so it is not tracked separately.
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
            bool avx2{false};
            bool avx512f{false};

            CpuFeatures() {
#if defined(DEGLIB_X86)
                int cpu_info[4] = {0};

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

    enum class InstructionSet : uint8_t {
        Auto   = 0,
        Scalar = 1,
        AVX2   = 2,
        AVX512 = 3
    };

    inline const char* instruction_set_to_string(InstructionSet inst) {
        switch (inst) {
            case InstructionSet::Auto:   return "Auto";
            case InstructionSet::Scalar: return "Scalar";
            case InstructionSet::AVX2:   return "AVX2";
            case InstructionSet::AVX512: return "AVX512";
        }
        return "Unknown";
    }

    // Runtime CPU feature detection — safe to call from any translation unit.
    // These checks are performed once (cached) and have zero cost per call thereafter.

    inline bool has_avx2()   { return detail::features().avx2; }
    inline bool has_avx512() { return detail::features().avx512f; }

} // namespace deglib::cpu

