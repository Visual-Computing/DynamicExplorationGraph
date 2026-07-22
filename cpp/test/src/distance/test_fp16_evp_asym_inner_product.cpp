#include <cstdint>
#include <random>
#include <vector>

#include <config.h>

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

#include "gtest/gtest.h"
#include "test_helpers.h"
#include "quantization/evp_quantize.h"

static std::pair<std::vector<uint16_t>, std::vector<std::byte>>
make_asymmetric_pair(uint32_t dim, uint32_t non_zeros, int seed_a = 42, int seed_b = 99) {
    std::mt19937 rng_a(seed_a), rng_b(seed_b);
    std::vector<float> a(dim), b(dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = dist(rng_a);
    for (auto& x : b) x = dist(rng_b);
    return {
        floats_to_fp16(a),
        deglib::quantization::quantize_single(b.data(), dim, non_zeros)
    };
}

static float fp16_evp_asymmetric_naive_ref(const uint16_t* a, const std::byte* b, uint32_t dim) {
    std::vector<float> reconstructed_b(dim, 0.0f);
    const size_t mask_bytes = dim / 8;
    const std::byte* ones_b = b;
    const std::byte* negs_b = b + mask_bytes;

    for (size_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
        uint8_t ones_byte = static_cast<uint8_t>(ones_b[byte_idx]);
        uint8_t negs_byte = static_cast<uint8_t>(negs_b[byte_idx]);

        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t i = byte_idx * 8 + bit;
            if (i >= dim) break;

            float val = 0.0f;
            if ((ones_byte & (1 << bit)) != 0) {
                val = 1.0f;
            } else if ((negs_byte & (1 << bit)) != 0) {
                val = -1.0f;
            }
            reconstructed_b[i] = val;
        }
    }

    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float fa = deglib::distances::fp16_to_float(a[i]);
        dot += fa * reconstructed_b[i];
    }

    return 1.0f - dot;
}

TEST(FP16EvpAsymInnerProduct, HandCraftedCalculation) {
    uint32_t dim = 8;
    auto a = floats_to_fp16({1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f});

    std::vector<std::byte> b(2);
    b[0] = static_cast<std::byte>(13);
    b[1] = static_cast<std::byte>(82);

    float d = deglib::distances::FP16EvpAsymInnerProduct::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, -21.f, 1e-2f);
}

TEST(FP16EvpAsymInnerProduct, MatchesNaive_64Dim) {
    uint32_t dim = 64;
    auto [a, b] = make_asymmetric_pair(dim, 16, 12, 34);
    float d = deglib::distances::FP16EvpAsymInnerProduct::compare(a.data(), b.data(), &dim);
    float ref = fp16_evp_asymmetric_naive_ref(a.data(), b.data(), dim);
    EXPECT_NEAR(d, ref, 1e-2f);
}

TEST(FP16EvpAsymInnerProduct, MatchesNaive_128Dim) {
    uint32_t dim = 128;
    auto [a, b] = make_asymmetric_pair(dim, 32, 56, 78);
    float d = deglib::distances::FP16EvpAsymInnerProduct::compare(a.data(), b.data(), &dim);
    float ref = fp16_evp_asymmetric_naive_ref(a.data(), b.data(), dim);
    EXPECT_NEAR(d, ref, 1e-2f);
}

TEST(FP16EvpAsymInnerProduct, MatchesNaive_256Dim) {
    uint32_t dim = 256;
    auto [a, b] = make_asymmetric_pair(dim, 64, 90, 12);
    float d = deglib::distances::FP16EvpAsymInnerProduct::compare(a.data(), b.data(), &dim);
    float ref = fp16_evp_asymmetric_naive_ref(a.data(), b.data(), dim);
    EXPECT_NEAR(d, ref, 1e-2f);
}

#if defined(USE_SSE)
TEST(FP16EvpAsymInnerProduct, SSEMatchesRef) {
    std::vector<uint32_t> dims = {8, 16, 32, 64, 128, 256};
    for (uint32_t dim : dims) {
        auto [a, b] = make_asymmetric_pair(dim, dim / 4, 12, 34);
        float d = 1.f - deglib::distances::FP16EvpAsymInnerProduct::ip_sse(a.data(), b.data(), &dim);
        float ref = fp16_evp_asymmetric_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}
#endif

#if defined(USE_AVX)
TEST(FP16EvpAsymInnerProduct, AVX2MatchesRef) {
    std::vector<uint32_t> dims = {8, 16, 32, 64, 128, 256};
    for (uint32_t dim : dims) {
        auto [a, b] = make_asymmetric_pair(dim, dim / 4, 56, 78);
        float d = 1.f - deglib::distances::FP16EvpAsymInnerProduct::ip_avx2(a.data(), b.data(), &dim);
        float ref = fp16_evp_asymmetric_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}
#endif

#if defined(USE_AVX512)
TEST(FP16EvpAsymInnerProduct, AVX512MatchesRef) {
    std::vector<uint32_t> dims = {8, 16, 24, 32, 48, 64, 128, 256};
    for (uint32_t dim : dims) {
        auto [a, b] = make_asymmetric_pair(dim, dim / 4, 90, 12);
        float d = 1.f - deglib::distances::FP16EvpAsymInnerProduct::ip_avx512(a.data(), b.data(), &dim);
        float ref = fp16_evp_asymmetric_naive_ref(a.data(), b.data(), dim);
        EXPECT_NEAR(d, ref, 1e-2f) << "dim=" << dim;
    }
}
#endif

TEST(FP16EvpAsymInnerProduct, PathsAgree) {
    std::vector<uint32_t> dims = {8, 16, 32, 64, 128, 256};
    for (uint32_t dim : dims) {
        auto [a, b] = make_asymmetric_pair(dim, dim / 4, 77, 88);
        float naive = 1.f - deglib::distances::FP16EvpAsymInnerProduct::ip_naive(a.data(), b.data(), &dim);
        float dispatch = deglib::distances::FP16EvpAsymInnerProduct::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(dispatch, naive, 1e-2f) << "dispatch vs naive dim=" << dim;

#if defined(USE_SSE)
        float sse = 1.f - deglib::distances::FP16EvpAsymInnerProduct::ip_sse(a.data(), b.data(), &dim);
        EXPECT_NEAR(sse, naive, 1e-2f) << "sse vs naive dim=" << dim;
#endif

#if defined(USE_AVX)
        float avx2 = 1.f - deglib::distances::FP16EvpAsymInnerProduct::ip_avx2(a.data(), b.data(), &dim);
        EXPECT_NEAR(avx2, naive, 1e-2f) << "avx2 vs naive dim=" << dim;
#endif

#if defined(USE_AVX512)
        float avx512 = 1.f - deglib::distances::FP16EvpAsymInnerProduct::ip_avx512(a.data(), b.data(), &dim);
        EXPECT_NEAR(avx512, naive, 1e-2f) << "avx512 vs naive dim=" << dim;
#endif
    }
}

TEST(FP16EvpAsymInnerProduct, CompareBatchMatchesSingle) {
    uint32_t dim = 64;
    constexpr int N = 12;
    std::vector<std::vector<uint16_t>> queries(N);
    std::vector<std::vector<std::byte>> dbs(N);
    for (int i = 0; i < N; ++i) {
        auto [a, b] = make_asymmetric_pair(dim, 16, i * 10, i * 10 + 5);
        queries[i] = a;
        dbs[i] = b;
    }

    const void* db_arr[N];
    for (int i = 0; i < N; ++i) db_arr[i] = dbs[i].data();

    auto check = [&](size_t count) {
        float batch_dists[16];
        deglib::distances::FP16EvpAsymInnerProduct::compare_batch(
            queries[0].data(), db_arr, count, &dim, batch_dists);
        for (size_t i = 0; i < count; ++i) {
            float expected = deglib::distances::FP16EvpAsymInnerProduct::compare(
                queries[0].data(), db_arr[i], &dim);
            EXPECT_NEAR(batch_dists[i], expected, 1e-5f) << "i=" << i << " count=" << count;
        }
    };

    check(1);    // pure tail
    check(3);    // tail
    check(4);    // batch4 + tail
    check(5);    // batch4 + tail
    check(7);    // batch4 + tail
    check(8);    // batch8
    check(9);    // batch8 + tail
    check(12);   // batch8 + batch4
}
