#pragma once

#include "deglib/config.h"

#include <cstdint>

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
        avx2 = (cpu_info[1] & (1 << 5)) != 0;
        avx512f = (cpu_info[1] & (1 << 16)) != 0;
#endif
    }
};

inline const CpuFeatures& features() {
    static CpuFeatures cached;
    return cached;
}

}  // namespace detail

enum class InstructionSet : uint8_t { Auto = 0, Scalar = 1, AVX2 = 2, AVX512 = 3 };

inline const char* instruction_set_to_string(InstructionSet inst) {
    switch (inst) {
        case InstructionSet::Auto:
            return "Auto";
        case InstructionSet::Scalar:
            return "Scalar";
        case InstructionSet::AVX2:
            return "AVX2";
        case InstructionSet::AVX512:
            return "AVX512";
    }
    return "Unknown";
}

// Runtime CPU feature detection — safe to call from any translation unit.
// These checks are performed once (cached) and have zero cost per call thereafter.

inline bool has_avx2() { return detail::features().avx2; }
inline bool has_avx512() { return detail::features().avx512f; }

}  // namespace deglib::cpu
