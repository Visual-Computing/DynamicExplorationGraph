#pragma once

#include <cstdint>

// Architecture flags. Define DEGLIB_X86 if the code is for x86 machines. ARM often has separate coding pathes.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define DEGLIB_X86 1
#endif

// Compile methods with this attribute for AVX-512 base functions on GCC/Clang
#if defined(DEGLIB_X86) && (defined(__GNUC__) || defined(__clang__))
    #define DEGLIB_TARGET_AVX512 __attribute__((target("avx512f,avx512dq,avx512bw,avx512vpopcntdq")))
    #define DEGLIB_TARGET_AVX512_VNNI __attribute__((target("avx512f,avx512dq,avx512bw,avx512vpopcntdq,avx512vnni")))
    #define DEGLIB_TARGET_AVX2_VNNI __attribute__((target("avx2,f16c,fma,avxvnni")))
    #define DEGLIB_TARGET_AVX2 __attribute__((target("avx2,f16c,fma")))
#else
    #define DEGLIB_TARGET_AVX512
    #define DEGLIB_TARGET_AVX512_VNNI
    #define DEGLIB_TARGET_AVX2_VNNI
    #define DEGLIB_TARGET_AVX2
#endif

// Architecture intrinsic headers
#if defined(DEGLIB_X86)
    #ifdef _MSC_VER
        #include <intrin.h>

        #include <stdexcept>
    #else
        #include <cpuid.h>
        #include <x86intrin.h>
        #include <xmmintrin.h>  // for _mm_prefetch
    #endif
#endif

// CPU detection utilities
#include "deglib/utils/cpu.h"
