#include <bit>
#include <cstdint>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test_helpers.h"
#include "quantization/evp_quantize.h"

static std::vector<std::vector<std::byte>>
make_evp_vecs(uint32_t dim, uint32_t non_zeros, size_t count, int seed_base = 42) {
    std::vector<std::vector<std::byte>> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        std::mt19937 rng(seed_base + static_cast<int>(i * 7));
        std::vector<float> v(dim);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : v) x = dist(rng);
        result.push_back(deglib::quantization::quantize_single(v.data(), dim, non_zeros));
    }
    return result;
}

static std::pair<std::vector<std::byte>, std::vector<std::byte>>
make_evp_pair(uint32_t dim, uint32_t non_zeros, int seed_a = 42, int seed_b = 99) {
    std::mt19937 rng_a(seed_a), rng_b(seed_b);
    std::vector<float> a(dim), b(dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = dist(rng_a);
    for (auto& x : b) x = dist(rng_b);
    return {
        deglib::quantization::quantize_single(a.data(), dim, non_zeros),
        deglib::quantization::quantize_single(b.data(), dim, non_zeros)
    };
}

TEST(EvpBitsSimilarity, NaiveSelfSimilarity) {
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;
    float sim = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(a.data()),
        static_cast<const void*>(&dim));
    EXPECT_GT(sim, 0.0f);
}

TEST(EvpBitsSimilarity, CompareNormalizesToZeroForIdenticalVectors) {
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;

    float sim = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(a.data()),
        static_cast<const void*>(&dim));

    const float max_similarity = static_cast<float>(dim) * 2.0f;
    float expected = 1.f - (sim / max_similarity);

    float dist = deglib::distances::EvpInnerProduct::compare(
        static_cast<const void*>(a.data()), static_cast<const void*>(a.data()),
        static_cast<const void*>(&dim));

    EXPECT_FLOAT_EQ(dist, expected);
}

TEST(EvpBitsSimilarity, CompareMatchesPairwiseNormalization) {
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;

    float sim = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));

    const float max_similarity = static_cast<float>(dim) * 2.0f;
    float expected = 1.f - (sim / max_similarity);

    float dist = deglib::distances::EvpInnerProduct::compare(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));

    EXPECT_FLOAT_EQ(dist, expected);
}

TEST(EvpBitsSimilarity, Avx2MatchesNaive_64Dim) {
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;
    float naive = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx2  = deglib::distances::EvpInnerProduct::compare_avx2(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx2, naive, 0.001f);
}

TEST(EvpBitsSimilarity, Avx2MatchesNaive_128Dim) {
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    float naive = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx2  = deglib::distances::EvpInnerProduct::compare_avx2(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx2, naive, 0.001f);
}

TEST(EvpBitsSimilarity, Avx2MatchesNaive_256Dim) {
    auto [a, b] = make_evp_pair(256, 64);
    uint32_t dim = 256;
    float naive = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx2  = deglib::distances::EvpInnerProduct::compare_avx2(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx2, naive, 0.001f);
}

TEST(EvpBitsSimilarity, Avx512MatchesNaive_64Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(64, 16);
    uint32_t dim = 64;
    float naive = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx512 = deglib::distances::EvpInnerProduct::compare_avx512(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive, 0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx512MatchesNaive_128Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    float naive = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx512 = deglib::distances::EvpInnerProduct::compare_avx512(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive, 0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx512MatchesNaive_256Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(256, 64);
    uint32_t dim = 256;
    float naive = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    float avx512 = deglib::distances::EvpInnerProduct::compare_avx512(
        static_cast<const void*>(a.data()), static_cast<const void*>(b.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive, 0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx2AndAvx512MatchNaive_128Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    const void* qa = static_cast<const void*>(a.data());
    const void* qb = static_cast<const void*>(b.data());
    const void* qd = static_cast<const void*>(&dim);

    float naive  = deglib::distances::EvpInnerProduct::compare_naive(qa, qb, qd);
    float avx2   = deglib::distances::EvpInnerProduct::compare_avx2(qa, qb, qd);
    float avx512 = deglib::distances::EvpInnerProduct::compare_avx512(qa, qb, qd);

    EXPECT_NEAR(avx2,   naive,  0.001f);
    EXPECT_NEAR(avx512, naive,  0.001f);
#endif
}

TEST(EvpBitsSimilarity, Avx2AndAvx512MatchNaive_256Dim) {
#if !defined(USE_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    auto [a, b] = make_evp_pair(256, 64);
    uint32_t dim = 256;
    const void* qa = static_cast<const void*>(a.data());
    const void* qb = static_cast<const void*>(b.data());
    const void* qd = static_cast<const void*>(&dim);

    float naive  = deglib::distances::EvpInnerProduct::compare_naive(qa, qb, qd);
    float avx2   = deglib::distances::EvpInnerProduct::compare_avx2(qa, qb, qd);
    float avx512 = deglib::distances::EvpInnerProduct::compare_avx512(qa, qb, qd);

    EXPECT_NEAR(avx2,   naive,  0.001f);
    EXPECT_NEAR(avx512, naive,  0.001f);
#endif
}

TEST(EvpBitsSimilarity, Symmetry) {
    auto [a, b] = make_evp_pair(128, 32);
    uint32_t dim = 128;
    const void* qa = static_cast<const void*>(a.data());
    const void* qb = static_cast<const void*>(b.data());
    const void* qd = static_cast<const void*>(&dim);

    float ab = deglib::distances::EvpInnerProduct::compare_naive(qa, qb, qd);
    float ba = deglib::distances::EvpInnerProduct::compare_naive(qb, qa, qd);
    EXPECT_NEAR(ab, ba, 0.001f);
}

TEST(EvpInnerProduct, CompareBatch4MatchesCompare) {
    auto vecs = make_evp_vecs(128, 32, 5);
    uint32_t dim = 128;
    const void* query = static_cast<const void*>(vecs[0].data());
    const void* arr[4] = {
        static_cast<const void*>(vecs[1].data()),
        static_cast<const void*>(vecs[2].data()),
        static_cast<const void*>(vecs[3].data()),
        static_cast<const void*>(vecs[4].data()),
    };
    float batch_dists[4];
    deglib::distances::EvpInnerProduct::compare_batch(
        query, arr, 4, static_cast<const void*>(&dim), batch_dists);

    for (int i = 0; i < 4; ++i) {
        float single = deglib::distances::EvpInnerProduct::compare(
            query, arr[i], static_cast<const void*>(&dim));
        EXPECT_NEAR(batch_dists[i], single, 0.001f);
    }
}

TEST(EvpInnerProduct, CompareBatch8MatchesCompare) {
    auto vecs = make_evp_vecs(128, 32, 9);
    uint32_t dim = 128;
    const void* query = static_cast<const void*>(vecs[0].data());
    const void* arr[8] = {
        static_cast<const void*>(vecs[1].data()),
        static_cast<const void*>(vecs[2].data()),
        static_cast<const void*>(vecs[3].data()),
        static_cast<const void*>(vecs[4].data()),
        static_cast<const void*>(vecs[5].data()),
        static_cast<const void*>(vecs[6].data()),
        static_cast<const void*>(vecs[7].data()),
        static_cast<const void*>(vecs[8].data()),
    };
    float batch_dists[8];
    deglib::distances::EvpInnerProduct::compare_batch(
        query, arr, 8, static_cast<const void*>(&dim), batch_dists);

    for (int i = 0; i < 8; ++i) {
        float single = deglib::distances::EvpInnerProduct::compare(
            query, arr[i], static_cast<const void*>(&dim));
        EXPECT_NEAR(batch_dists[i], single, 0.001f);
    }
}

TEST(EvpInnerProduct, CompareBatch5UsesBatch4PlusRemainder) {
    auto vecs = make_evp_vecs(128, 32, 6);
    uint32_t dim = 128;
    const void* query = static_cast<const void*>(vecs[0].data());
    const void* arr[5] = {
        static_cast<const void*>(vecs[1].data()),
        static_cast<const void*>(vecs[2].data()),
        static_cast<const void*>(vecs[3].data()),
        static_cast<const void*>(vecs[4].data()),
        static_cast<const void*>(vecs[5].data()),
    };
    float batch_dists[5];
    deglib::distances::EvpInnerProduct::compare_batch(
        query, arr, 5, static_cast<const void*>(&dim), batch_dists);

    for (int i = 0; i < 5; ++i) {
        float single = deglib::distances::EvpInnerProduct::compare(
            query, arr[i], static_cast<const void*>(&dim));
        EXPECT_NEAR(batch_dists[i], single, 0.001f);
    }
}

TEST(EvpBitsSimilarity, SmallDims) {
    std::mt19937 rng(42);
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;
    std::vector<float> v(dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(rng);

    auto result = deglib::quantization::quantize_single(v.data(), dim, non_zeros);

    float naive  = deglib::distances::EvpInnerProduct::compare_naive(
        static_cast<const void*>(result.data()), static_cast<const void*>(result.data()),
        static_cast<const void*>(&dim));
    float avx2   = deglib::distances::EvpInnerProduct::compare_avx2(
        static_cast<const void*>(result.data()), static_cast<const void*>(result.data()),
        static_cast<const void*>(&dim));

    EXPECT_GT(naive, 0.0f);
    EXPECT_NEAR(avx2,   naive,  0.001f);

#if defined(USE_AVX512)
    float avx512 = deglib::distances::EvpInnerProduct::compare_avx512(
        static_cast<const void*>(result.data()), static_cast<const void*>(result.data()),
        static_cast<const void*>(&dim));
    EXPECT_NEAR(avx512, naive,  0.001f);
#endif
}
