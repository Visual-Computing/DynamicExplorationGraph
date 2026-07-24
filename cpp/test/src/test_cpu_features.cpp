#include <gtest/gtest.h>
#include <iostream>

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

TEST(CPUFeaturesTest, QueryHardwareSupport) {
    int cpu_info[4] = {0};
    get_cpuid(1, 0, cpu_info);

    bool hw_sse42 = (cpu_info[2] & (1 << 20)) != 0;
    bool hw_f16c  = (cpu_info[2] & (1 << 29)) != 0;
    bool hw_avx   = (cpu_info[2] & (1 << 28)) != 0;

    get_cpuid(7, 0, cpu_info);
    bool hw_avx2    = (cpu_info[1] & (1 << 5)) != 0;
    bool hw_avx512f = (cpu_info[1] & (1 << 16)) != 0;

    std::cout << "[CPU Test] Hardware Feature Support:" << std::endl;
    std::cout << "  SSE4.2  : " << (hw_sse42 ? "YES" : "NO") << std::endl;
    std::cout << "  F16C    : " << (hw_f16c  ? "YES" : "NO") << std::endl;
    std::cout << "  AVX     : " << (hw_avx   ? "YES" : "NO") << std::endl;
    std::cout << "  AVX2    : " << (hw_avx2  ? "YES" : "NO") << std::endl;
    std::cout << "  AVX512F : " << (hw_avx512f ? "YES" : "NO") << std::endl;

#if defined(__AVX512F__)
    EXPECT_TRUE(hw_avx512f) << "Binary compiled with AVX512F, but CPU hardware does not support it!";
#endif

#if defined(__AVX2__)
    EXPECT_TRUE(hw_avx2) << "Binary compiled with AVX2, but CPU hardware does not support it!";
#endif

#if defined(__AVX__)
    EXPECT_TRUE(hw_avx) << "Binary compiled with AVX, but CPU hardware does not support it!";
#endif
}
