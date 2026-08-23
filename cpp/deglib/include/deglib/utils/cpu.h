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
    bool avx_vnni{false};
    bool avx512f{false};
    bool avx512_vnni{false};

    CpuFeatures() {
#if defined(DEGLIB_X86)
        int cpu_info[4] = {0};

        // Leaf 7, subleaf 0: extended feature flags in EBX and ECX
        cpuid(7, 0, cpu_info);
        avx2 = (cpu_info[1] & (1 << 5)) != 0;
        avx512f = (cpu_info[1] & (1 << 16)) != 0;
        avx512_vnni = (cpu_info[2] & (1 << 11)) != 0;

        // Leaf 7, subleaf 1: extended feature flags in EAX (AVX_VNNI is Bit 4)
        cpuid(7, 1, cpu_info);
        avx_vnni = (cpu_info[0] & (1 << 4)) != 0;
#endif
    }
};

inline const CpuFeatures& features() {
    static CpuFeatures cached;
    return cached;
}

}  // namespace detail

enum class InstructionSet : uint8_t { Auto = 0, Scalar = 1, AVX2 = 2, AVX2_VNNI = 3, AVX512 = 4, AVX512_VNNI = 5 };

inline const char* instruction_set_to_string(InstructionSet inst) {
    switch (inst) {
        case InstructionSet::Auto:
            return "Auto";
        case InstructionSet::Scalar:
            return "Scalar";
        case InstructionSet::AVX2:
            return "AVX2";
        case InstructionSet::AVX2_VNNI:
            return "AVX2_VNNI";
        case InstructionSet::AVX512:
            return "AVX512";
        case InstructionSet::AVX512_VNNI:
            return "AVX512_VNNI";
    }
    return "Unknown";
}

// Runtime CPU feature detection — safe to call from any translation unit.
// These checks are performed once (cached) and have zero cost per call thereafter.

inline bool has_avx2() { return detail::features().avx2; }
inline bool has_avx_vnni() { return detail::features().avx_vnni; }
inline bool has_avx512() { return detail::features().avx512f; }
inline bool has_avx512_vnni() { return detail::features().avx512_vnni; }

// Validates the requested InstructionSet against CPU capabilities
// and resolves Auto / VNNI aliases to the concrete InstructionSet (AVX512, AVX2, or Scalar).
inline InstructionSet resolve_instruction_set(InstructionSet instruction) {
    if (instruction == InstructionSet::Scalar) {
        return InstructionSet::Scalar;
    }

#if defined(DEGLIB_X86)
    if (instruction == InstructionSet::AVX512 && !has_avx512()) {
        throw std::runtime_error("AVX512 instruction set requested, but not supported by CPU");
    }
    if (instruction == InstructionSet::AVX512_VNNI && !has_avx512_vnni()) {
        throw std::runtime_error("AVX512-VNNI instruction set requested, but not supported by CPU");
    }
    if (instruction == InstructionSet::AVX2 && !has_avx2()) {
        throw std::runtime_error("AVX2 instruction set requested, but not supported by CPU");
    }
    if (instruction == InstructionSet::AVX2_VNNI && !has_avx_vnni()) {
        throw std::runtime_error("AVX-VNNI instruction set requested, but not supported by CPU");
    }

    if (instruction == InstructionSet::AVX512 || instruction == InstructionSet::AVX512_VNNI ||
        (instruction == InstructionSet::Auto && has_avx512())) {
        return InstructionSet::AVX512;
    }
    if (instruction == InstructionSet::AVX2 || instruction == InstructionSet::AVX2_VNNI ||
        (instruction == InstructionSet::Auto && has_avx2())) {
        return InstructionSet::AVX2;
    }
#else
    if (instruction != InstructionSet::Auto && instruction != InstructionSet::Scalar) {
        throw std::runtime_error("Requested SIMD instruction set is not supported on this platform");
    }
#endif

    return InstructionSet::Scalar;
}

}  // namespace deglib::cpu
