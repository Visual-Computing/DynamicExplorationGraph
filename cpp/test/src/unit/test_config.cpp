#include <gtest/gtest.h>
#include <iostream>

#include <deglib.h>

#if defined(DEGLIB_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// Helper to query CPUID features safely on MSVC and GCC/Clang
static void get_cpuid(int leaf, int subleaf, int cpu_info[4]) {
#if defined(_MSC_VER)
    __cpuidex(cpu_info, leaf, subleaf);
#else
    __cpuid_count(leaf, subleaf, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#endif
}
#endif

TEST(CPUFeaturesTest, CpuHelperFunctionsMatchRawCPUID) {
#if defined(DEGLIB_X86)
    int cpu_info[4] = {0};
    get_cpuid(1, 0, cpu_info);

    get_cpuid(7, 0, cpu_info);
    bool hw_avx2    = (cpu_info[1] & (1 << 5)) != 0;
    bool hw_avx512f = (cpu_info[1] & (1 << 16)) != 0;

    // Verify deglib::cpu helper functions match raw CPUID queries
    EXPECT_EQ(deglib::cpu::has_avx2(),   hw_avx2)  << "deglib::cpu::has_avx2() does not match raw CPUID";
    EXPECT_EQ(deglib::cpu::has_avx512(), hw_avx512f) << "deglib::cpu::has_avx512() does not match raw CPUID";

    std::cout << "[CPU Test] deglib::cpu helper functions verified against raw CPUID" << std::endl;
#else
    // On non-x86 architectures (e.g. ARM), x86 feature flags must all be false
    EXPECT_FALSE(deglib::cpu::has_avx2());
    EXPECT_FALSE(deglib::cpu::has_avx512());

    std::cout << "[CPU Test] Non-x86 architecture detected: all x86 SIMD features set to false as expected" << std::endl;
#endif
}