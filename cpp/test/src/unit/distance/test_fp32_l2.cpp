// test_fp32_l2.cpp — Unit tests for L2 distance computations
//
// Tests scalar and SIMD L2 distance implementations for various dimensions.
// HEAD API uses L2Float, L2Float_AVX512<Mode>, L2Float_AVX2<Mode> instead of
// L2Float4Ext/L2Float8Ext/L2Float16Ext from older commits.

#include <chrono>

#include "gtest/gtest.h"
#include "deglib/distance/fp32_l2.h"
#include "deglib/distances.h"

namespace {

inline std::vector<float> make_float_vec(size_t n, int seed = 0) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>((seed + static_cast<int>(i)) % 100);
    }
    return v;
}

inline float l2_naive(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

using deglib::distances::fp32_l2::L2Float;

TEST(L2Float, IdentityZero) {
    std::vector<float> v(16, 0.0f);
    size_t dim = v.size();
    float d = L2Float::compare(v.data(), v.data(), &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(L2Float, KnownValue) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    size_t dim = 3;
    float d = L2Float::compare(a, b, &dim);
    EXPECT_NEAR(d, 27.0f, 1e-4f);
}

TEST(L2Float, Symmetry) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 99);
    size_t dim = a.size();
    float ab = L2Float::compare(a.data(), b.data(), &dim);
    float ba = L2Float::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(L2Float, MatchesNaive_4) {
    auto a = make_float_vec(4);
    auto b = make_float_vec(4, 7);
    size_t dim = a.size();
    float d = L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Float, MatchesNaive_64) {
    auto a = make_float_vec(64);
    auto b = make_float_vec(64, 13);
    size_t dim = a.size();
    float d = L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-3f);
}

TEST(L2Float, MatchesNaive_128) {
    auto a = make_float_vec(128);
    auto b = make_float_vec(128, 21);
    size_t dim = a.size();
    float d = L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-3f);
}

TEST(L2Float, Dim1) {
    float a[] = {3.0f};
    float b[] = {7.0f};
    size_t dim = 1;
    float d = L2Float::compare(a, b, &dim);
    EXPECT_NEAR(d, 16.0f, 1e-4f);
}

TEST(L2Float, Dim0) {
    float a = 1.0f;
    float b = 2.0f;
    size_t dim = 0;
    float d = L2Float::compare(&a, &b, &dim);
    EXPECT_EQ(d, 0.0f);
}

TEST(L2Float, MatchesNaive_MultipleDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = L2Float::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float, LargeDimension) {
    size_t dim = 1000;
    auto a = make_float_vec(dim, 42);
    auto b = make_float_vec(dim, 123);
    float d = L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f);
}

TEST(L2Float, NegativeValues) {
    size_t dim = 6;
    std::vector<float> a = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float d = L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-5f);
}

TEST(L2Float, LargeValues) {
    size_t dim = 16;
    std::vector<float> a(dim, 1000.0f);
    std::vector<float> b(dim, -1000.0f);
    float d = L2Float::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f);
}

} // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::fp32_l2::L2Float_AVX512;
using deglib::distances::fp32_l2::L2Float_AVX2;
using deglib::distances::ResidualMode;

// Test AVX512 variants against naive reference (if CPU supports AVX512)
TEST(L2Float_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = L2Float_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// Test AVX2 variants against naive reference (if CPU supports AVX2)
TEST(L2Float_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = L2Float_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

// Test that select_dist returns valid distance functions for various dimensions
TEST(L2Float_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::fp32_l2::select_dist(dim);
        auto a = make_float_vec(dim);
        auto b = make_float_vec(dim, dim);
        float d = std::visit([&](auto&& dist) {
            return dist.compare(a.data(), b.data(), &dim);
        }, dist_variant);
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

} // anonymous namespace

#endif // DEGLIB_X86

namespace {

// FloatSpace integration tests
TEST(L2Float_FloatSpace, L2Metric) {
    size_t dim = 64;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    auto a = make_float_vec(dim, 42);
    auto b = make_float_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f);
}

TEST(L2Float_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256, 512};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

        auto a = make_float_vec(dim, dim);
        auto b = make_float_vec(dim, dim + 1);

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, l2_naive(a.data(), b.data(), dim), 1e-2f)
            << "dim=" << dim;
    }
}

TEST(L2Float_Batch, MatchesSingleCompare) {
   std::vector<size_t> dims = {8, 16, 32, 64, 128, 256, 768};
   std::vector<size_t> counts = {1, 3, 4, 7, 8, 9, 15, 16, 25};

   for (size_t dim : dims) {
       for (size_t count : counts) {
           auto q = make_float_vec(dim, 77);

           std::vector<std::vector<float>> db(count);
           std::vector<const void*> db_ptrs(count);
           for (size_t i = 0; i < count; ++i) {
               db[i] = make_float_vec(dim, static_cast<int>(i * 10 + 1));
               db_ptrs[i] = db[i].data();
           }

           std::vector<float> batch_dists(count, 0.0f);
           auto dist_variant = deglib::distances::fp32_l2::select_dist(dim);

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
}

} // anonymous namespace

// ============================================================================
// Performance: compare vs compare_batch
// ============================================================================

TEST(L2Float_Batch, PerformanceCompareVsBatch) {
  const size_t dim = 128;
  const size_t count = 1000;

  auto q = make_float_vec(dim, 77);

  std::vector<std::vector<float>> db(count);
  std::vector<const void*> db_ptrs(count);
  for (size_t i = 0; i < count; ++i) {
      db[i] = make_float_vec(dim, static_cast<int>(i * 10 + 1));
      db_ptrs[i] = db[i].data();
  }

  std::vector<float> batch_dists(count, 0.0f);
  auto dist_variant = deglib::distances::fp32_l2::select_dist(dim);

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
}
