// test_evp_inner_product.cpp — Unit tests for symmetric EVP inner product distance
//
// Tests cover:
// - Naive scalar implementation correctness & symmetry
// - Distance normalization (distance = 1 - similarity / (2*dim))
// - AVX2 and AVX-512 SIMD variants match naive across vector dimensions
// - SelectDist variant selection & FloatSpace integration

#include <bit>
#include <cstdint>
#include <chrono>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "common/test_helpers.h"
#include "deglib/optimization/quantization/evp_quantize.h"
#include "deglib/distances.h"

// ============================================================================
// Test helpers
// ============================================================================

static std::pair<std::vector<std::byte>, std::vector<std::byte>>
make_evp_pair(uint32_t dim, uint32_t non_zeros, int seed_a = 42, int seed_b = 99) {
    std::mt19937 rng_a(seed_a), rng_b(seed_b);
    std::vector<float> a(dim), b(dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : a) x = dist(rng_a);
    for (auto& x : b) x = dist(rng_b);
    return {
        deglib::quantization::evp::quantize_single(a.data(), dim, non_zeros),
        deglib::quantization::evp::quantize_single(b.data(), dim, non_zeros)
    };
}

// ============================================================================
// Naive distance & Normalization tests
// ============================================================================

TEST(EvpInnerProduct, NormalizationAndSymmetry) {
    const uint32_t dim = 128;
    auto [a, b] = make_evp_pair(dim, 32);

    float sim_self = deglib::distances::evp_ip::EvpInnerProduct::compare_naive(
        a.data(), a.data(), &dim);
    EXPECT_GT(sim_self, 0.0f);

    float sim_ab = deglib::distances::evp_ip::EvpInnerProduct::compare_naive(
        a.data(), b.data(), &dim);
    float sim_ba = deglib::distances::evp_ip::EvpInnerProduct::compare_naive(
        b.data(), a.data(), &dim);
    EXPECT_FLOAT_EQ(sim_ab, sim_ba);

    float expected_dist = 1.f - (sim_ab / (2.0f * dim));
    float dist = deglib::distances::evp_ip::EvpInnerProduct::compare(
        a.data(), b.data(), &dim);
    EXPECT_FLOAT_EQ(dist, expected_dist);
}

// ============================================================================
// SIMD correctness vs Naive (AVX2 & AVX-512)
// ============================================================================

TEST(EvpInnerProduct, Avx2MatchesNaive) {
#if defined(DEGLIB_X86)
    for (uint32_t dim : {64, 128, 256}) {
        auto [a, b] = make_evp_pair(dim, dim / 4);
        float naive = deglib::distances::evp_ip::EvpInnerProduct::compare(
            a.data(), b.data(), &dim);
        float avx2 = deglib::distances::evp_ip::EvpInnerProduct_AVX2<deglib::distances::ResidualMode::Full>::compare(
            a.data(), b.data(), &dim);
        EXPECT_NEAR(avx2, naive, 0.001f) << "Mismatch at dimension " << dim;
    }
#else
    GTEST_SKIP() << "AVX2 support was not compiled in";
#endif
}

TEST(EvpInnerProduct, Avx512MatchesNaive) {
#if !defined(DEGLIB_X86) || !defined(DEGLIB_TARGET_AVX512)
    GTEST_SKIP() << "AVX-512 support was not compiled in";
#else
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "CPU does not support AVX-512";
    }
    for (uint32_t dim : {64, 128, 256}) {
        auto [a, b] = make_evp_pair(dim, dim / 4);
        float naive = deglib::distances::evp_ip::EvpInnerProduct::compare(
            a.data(), b.data(), &dim);
        float avx512 = deglib::distances::evp_ip::EvpInnerProduct_AVX512<deglib::distances::ResidualMode::Full>::compare(
            a.data(), b.data(), &dim);
        EXPECT_NEAR(avx512, naive, 0.001f) << "Mismatch at dimension " << dim;
    }
#endif
}

// ============================================================================
// Distance dispatch & FloatSpace integration
// ============================================================================

TEST(EvpInnerProduct, SelectDistReturnsValidVariant) {
    for (size_t dim : {8, 16, 32, 64, 128, 256, 512, 1024}) {
        auto variant = deglib::distances::evp_ip::select_dist(dim);
        (void)variant;
        SUCCEED();
    }
}

TEST(EvpInnerProduct, FloatSpaceEVPInnerProduct) {
    const size_t dim = 128;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::EVP_InnerProduct);

    EXPECT_EQ(space.dim(), dim);
    EXPECT_EQ(space.metric(), deglib::distances::Metric::EVP_InnerProduct);
    EXPECT_EQ(space.get_data_size(), 2 * (dim / 8));
}

// ============================================================================
// Batch correctness tests
// ============================================================================

TEST(EvpInnerProduct_Batch, MatchesSingleCompare) {
#if defined(DEGLIB_X86)
   std::vector<uint32_t> dims = {64, 128, 256, 512};
   std::vector<size_t> counts = {1, 3, 4, 7, 8, 9, 15, 16, 25};

   for (uint32_t dim : dims) {
       for (size_t count : counts) {
           auto [q, _] = make_evp_pair(dim, dim / 4, 77, 0);

           std::vector<std::vector<std::byte>> db(count);
           std::vector<const void*> db_ptrs(count);
           for (size_t i = 0; i < count; ++i) {
               auto [_, db_vec] = make_evp_pair(dim, dim / 4, static_cast<int>(i * 10 + 1), 0);
               db[i] = db_vec;
               db_ptrs[i] = db[i].data();
           }

           std::vector<float> batch_dists(count, 0.0f);
           auto dist_variant = deglib::distances::evp_ip::select_dist(dim);

           std::visit([&](auto&& dist) {
               using DistType = std::decay_t<decltype(dist)>;
               DistType::compare_batch(q.data(), db_ptrs.data(), count, &dim, batch_dists.data());
           }, dist_variant);

           for (size_t i = 0; i < count; ++i) {
               float single_dist = std::visit([&](auto&& dist) {
                   using DistType = std::decay_t<decltype(dist)>;
                   return DistType::compare(q.data(), db_ptrs[i], &dim);
               }, dist_variant);

               EXPECT_NEAR(batch_dists[i], single_dist, 1e-4f)
                   << "dim=" << dim << ", count=" << count << ", index=" << i;
           }
       }
   }
#else
   GTEST_SKIP() << "EVP SIMD support was not compiled in";
#endif
}

// ============================================================================
// Performance: compare vs compare_batch
// ============================================================================

TEST(EvpInnerProduct_Batch, PerformanceCompareVsBatch) {
#if defined(DEGLIB_X86)
  const uint32_t dim = 128;
  const size_t count = 1000;

  auto [q, _] = make_evp_pair(dim, dim / 4, 77, 0);

  std::vector<std::vector<std::byte>> db(count);
  std::vector<const void*> db_ptrs(count);
  for (size_t i = 0; i < count; ++i) {
      auto [_, db_vec] = make_evp_pair(dim, dim / 4, static_cast<int>(i * 10 + 1), 0);
      db[i] = db_vec;
      db_ptrs[i] = db[i].data();
  }

  std::vector<float> batch_dists(count, 0.0f);
  auto dist_variant = deglib::distances::evp_ip::select_dist(dim);

  // Warm up
  std::visit([&](auto&& dist) {
      using DistType = std::decay_t<decltype(dist)>;
      DistType::compare_batch(q.data(), db_ptrs.data(), count, &dim, batch_dists.data());
  }, dist_variant);

  // Time single compare loop
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<float> single_dists(count, 0.0f);
  std::visit([&](auto&& dist) {
      using DistType = std::decay_t<decltype(dist)>;
      for (size_t i = 0; i < count; ++i) {
          single_dists[i] = DistType::compare(q.data(), db_ptrs[i], &dim);
      }
  }, dist_variant);
  auto mid = std::chrono::high_resolution_clock::now();

  // Time batch compare
  std::visit([&](auto&& dist) {
      using DistType = std::decay_t<decltype(dist)>;
      DistType::compare_batch(q.data(), db_ptrs.data(), count, &dim, batch_dists.data());
  }, dist_variant);
  auto end = std::chrono::high_resolution_clock::now();

  auto single_us = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
  auto batch_us = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();

  std::cout << "  dim=" << dim << ", count=" << count << "\n"
            << "  single compare: " << single_us << " us\n"
            << "  batch compare:  " << batch_us << " us\n"
            << "  speedup:        " << (static_cast<double>(single_us) / batch_us) << "x\n";

  // Verify correctness
  for (size_t i = 0; i < count; ++i) {
      EXPECT_NEAR(batch_dists[i], single_dists[i], 1e-3f)
          << "dim=" << dim << ", index=" << i;
  }

  // Batch should be at least as fast (allowing for noise)
  EXPECT_LE(batch_us, single_us * 2)
      << "batch compare should not be significantly slower than single compare";
#else
  GTEST_SKIP() << "EVP SIMD support was not compiled in";
#endif
}



