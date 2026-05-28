#pragma once

#include <vector>

#include "gtest/gtest.h"
#include "test_helpers.h"

// ===========================================================================
// Config cascade verification: ensure USE_* macros are set correctly.
//
// Expected cascade (from config.h):
//   __AVX512F__  -> USE_AVX512, USE_AVX, USE_SSE
//   __AVX2__/__AVX__ -> USE_AVX, USE_SSE
//   __SSE__/__SSE2__  -> USE_SSE
// ===========================================================================

// --- Compile-time cascade assertions ---

TEST(ConfigCascade, USE_SSE_AlwaysSet_When_Wider_SIMD) {
    // USE_SSE must be defined whenever USE_AVX or USE_AVX512 is defined.
#ifdef USE_AVX512
    static_assert(true, "USE_AVX512 defined");
    #ifndef USE_AVX
        #error "USE_AVX512 defined but USE_AVX not defined — cascade broken"
    #endif
    #ifndef USE_SSE
        #error "USE_AVX512 defined but USE_SSE not defined — cascade broken"
    #endif
#elif defined(USE_AVX)
    static_assert(true, "USE_AVX defined");
    #ifndef USE_SSE
        #error "USE_AVX defined but USE_SSE not defined — cascade broken"
    #endif
#endif
    // If only USE_SSE is defined (no AVX/AVX512), that's fine.
    SUCCEED() << "Cascade check passed. USE_AVX512="
#ifdef USE_AVX512
        << "ON"
#else
        << "OFF"
#endif
        ", USE_AVX="
#ifdef USE_AVX
        << "ON"
#else
        << "OFF"
#endif
        ", USE_SSE="
#ifdef USE_SSE
        << "ON"
#else
        << "OFF"
#endif
        ;
}

// ===========================================================================
// Functional tests: verify that each SIMD path produces correct results
// when its corresponding USE_ macro is active.
// ===========================================================================

TEST(ConfigCascade, SSE_Path_Correctness) {
    // When SSE/AVX is available, verify that the SIMD path produces correct results
    // for dimensions that force SSE tail processing.
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    std::vector<size_t> dims = {4, 5, 6, 7, 8, 9, 12, 13, 16, 20, 100};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
#else
    GTEST_SKIP() << "No SIMD support available";
#endif
}

TEST(ConfigCascade, AVX_Path_Correctness) {
    // When USE_AVX is set, AVX main loop + SSE tail must both work.
#ifdef USE_AVX
    std::vector<size_t> dims = {8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
#else
    GTEST_SKIP() << "USE_AVX not defined";
#endif
}

TEST(ConfigCascade, AVX512_Path_Correctness) {
#ifdef USE_AVX512
    std::vector<size_t> dims = {16, 24, 32, 48, 64, 80, 96, 112, 128, 200, 256, 512};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
#else
    GTEST_SKIP() << "USE_AVX512 not defined";
#endif
}

// ===========================================================================
// Batch tests: verify that batch paths work correctly with cascading SIMD
// ===========================================================================

TEST(ConfigCascade, Batch_AvxPath_WhenAvxAvailable) {
    // l2_batch8_avx and l2_batch4_avx require USE_AVX.
    // With cascading, these are available whenever USE_AVX is set (even if USE_AVX512).
#if defined(USE_AVX) || defined(USE_AVX512)
    std::vector<size_t> dims = {8, 12, 16, 20, 24, 32, 40, 100, 128};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        std::vector<std::vector<float>> dbs(13);
        const void* db_ptrs[13];
        for (int j = 0; j < 13; ++j) {
            dbs[j] = make_float_vec(dim, (j + 1) * 7);
            db_ptrs[j] = dbs[j].data();
        }

        std::vector<float> dists(13);
        deglib::distances::L2Float4Ext::compare_batch(query.data(), db_ptrs, 13, &dim, dists.data());

        for (int j = 0; j < 13; ++j) {
            EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-2f)
                << "dim=" << dim << " db=" << j;
        }
    }
#else
    GTEST_SKIP() << "Neither USE_AVX nor USE_AVX512 defined";
#endif
}

TEST(ConfigCascade, Batch_SseOnly_WhenNoAvx) {
    // When USE_SSE is set but USE_AVX is NOT set, batch should fall back to SSE.
#if defined(USE_SSE) && !defined(USE_AVX)
    std::vector<size_t> dims = {4, 8, 12, 16, 20, 32, 64};
    for (size_t dim : dims) {
        auto query = make_float_vec(dim);
        std::vector<std::vector<float>> dbs(5);
        const void* db_ptrs[5];
        for (int j = 0; j < 5; ++j) {
            dbs[j] = make_float_vec(dim, (j + 1) * 3);
            db_ptrs[j] = dbs[j].data();
        }

        std::vector<float> dists(5);
        deglib::distances::L2Float4Ext::compare_batch(query.data(), db_ptrs, 5, &dim, dists.data());

        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(dists[j], l2_naive(query.data(), (const float*)db_ptrs[j], dim), 1e-2f)
                << "dim=" << dim << " db=" << j;
        }
    }
#else
    GTEST_SKIP() << "Not SSE-only";
#endif
}

// ===========================================================================
// Tail cascade tests: verify that dim%8 remainder is handled correctly
// when wider SIMD is active
// ===========================================================================

TEST(ConfigCascade, TailDimMod8_Correctness) {
    // Test dimensions that require tail handling: dim % 8 != 0
    // Under USE_AVX, the main loop processes dim/8*8, tail handles the rest.
#if defined(USE_AVX) || defined(USE_AVX512)
    std::vector<size_t> dims = {5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 21, 25, 33, 100, 125};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
#else
    GTEST_SKIP() << "Neither USE_AVX nor USE_AVX512 defined";
#endif
}

TEST(ConfigCascade, TailDimMod16_Correctness_Avx512) {
    // Under USE_AVX512, the main loop processes dim/16*16, tail handles the rest.
#ifdef USE_AVX512
    std::vector<size_t> dims = {17, 20, 24, 32, 33, 48, 49, 64, 65, 100, 112, 128, 129, 256, 257};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
#else
    GTEST_SKIP() << "USE_AVX512 not defined";
#endif
}

// ===========================================================================
// Consistency: all SIMD aliases must produce same results as naive
// ===========================================================================

TEST(ConfigCascade, AllAliases_ConsistentWithNaive) {
    std::vector<size_t> dims = {4, 8, 12, 16, 20, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float naive = l2_naive(a.data(), b.data(), dim);

        float d0 = deglib::distances::L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d0, naive, 1e-2f) << "L2Float dim=" << dim;

        if (dim % 4 == 0) {
            float d4 = deglib::distances::L2Float4Ext::compare(a.data(), b.data(), &dim);
            EXPECT_NEAR(d4, naive, 1e-2f) << "L2Float4Ext dim=" << dim;
        }
        if (dim % 8 == 0) {
            float d8 = deglib::distances::L2Float8Ext::compare(a.data(), b.data(), &dim);
            EXPECT_NEAR(d8, naive, 1e-2f) << "L2Float8Ext dim=" << dim;
        }
        if (dim % 16 == 0) {
            float d16 = deglib::distances::L2Float16Ext::compare(a.data(), b.data(), &dim);
            EXPECT_NEAR(d16, naive, 1e-2f) << "L2Float16Ext dim=" << dim;
        }
    }
}
