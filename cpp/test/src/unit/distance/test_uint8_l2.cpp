// test_uint8_l2.cpp — Unit tests for L2 distance on uint8_t vectors
//
// Tests scalar and SIMD L2 distance implementations for uint8_t vectors.
// HEAD API uses L2Uint8, L2Uint8_AVX512<Mode>, L2Uint8_AVX2<Mode>.

#include <vector>
#include <chrono>
#include "gtest/gtest.h"
#include "deglib/distance/uint8_l2.h"
#include "deglib/distances.h"

namespace {

inline std::vector<uint8_t> make_uint8_vec(size_t n, int seed = 0) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<uint8_t>((seed + static_cast<int>(i)) % 256);
    }
    return v;
}

inline float l2_uint8_naive(const uint8_t* a, const uint8_t* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        sum += d * d;
    }
    return sum;
}

using deglib::distances::uint8_l2::L2Uint8;

TEST(L2Uint8, IdentityZero) {
    std::vector<uint8_t> v(16, 42);
    size_t dim = v.size();
    float d = L2Uint8::compare(v.data(), v.data(), &dim);
    EXPECT_NEAR(d, 0.0f, 1e-4f);
}

TEST(L2Uint8, KnownValue) {
    uint8_t a[] = {0, 100, 200};
    uint8_t b[] = {50, 150, 250};
    size_t dim = 3;
    float d = L2Uint8::compare(a, b, &dim);
    EXPECT_NEAR(d, 7500.0f, 1e-4f);
}

TEST(L2Uint8, Symmetry) {
    auto a = make_uint8_vec(64);
    auto b = make_uint8_vec(64, 99);
    size_t dim = a.size();
    float ab = L2Uint8::compare(a.data(), b.data(), &dim);
    float ba = L2Uint8::compare(b.data(), a.data(), &dim);
    EXPECT_EQ(ab, ba);
}

TEST(L2Uint8, MatchesNaive) {
    std::vector<size_t> dims = {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8, NonAlignedDims) {
    std::vector<size_t> dims = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 17, 20, 24, 25, 33, 50, 100, 129, 200};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim + 1);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8, ExtremeValues) {
    std::vector<size_t> dims = {16, 64, 128};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim, 0);
        auto b = make_uint8_vec(dim, 1);
        float d = L2Uint8::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8, LargeDimension) {
    size_t dim = 1000;
    auto a = make_uint8_vec(dim, 42);
    auto b = make_uint8_vec(dim, 123);
    float d = L2Uint8::compare(a.data(), b.data(), &dim);
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f);
}

} // anonymous namespace

#if defined(DEGLIB_X86)

namespace {

using deglib::distances::uint8_l2::L2Uint8_AVX512;
using deglib::distances::uint8_l2::L2Uint8_AVX2;
using deglib::distances::ResidualMode;

TEST(L2Uint8_AVX512, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx512()) {
        GTEST_SKIP() << "AVX512 not supported";
    }
    std::vector<size_t> dims = {16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX512<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8_AVX2, MatchesNaive_IfSupported) {
    if (!deglib::cpu::has_avx2()) {
        GTEST_SKIP() << "AVX2 not supported";
    }
    std::vector<size_t> dims = {8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = L2Uint8_AVX2<ResidualMode::Full>::compare(a.data(), b.data(), &dim);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8_SelectDist, ReturnsValidDistance) {
    std::vector<size_t> dims = {1, 4, 8, 16, 32, 64, 100, 128, 256};
    for (size_t dim : dims) {
        auto dist_variant = deglib::distances::uint8_l2::select_dist(dim);
        auto a = make_uint8_vec(dim);
        auto b = make_uint8_vec(dim, dim);
        float d = std::visit([&](auto&& dist) {
            return dist.compare(a.data(), b.data(), &dim);
        }, dist_variant);
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

} // anonymous namespace

#endif // DEGLIB_X86

namespace {

// FloatSpace integration tests
TEST(L2Uint8_FloatSpace, L2Uint8Metric) {
    size_t dim = 64;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_L2);

    auto a = make_uint8_vec(dim, 42);
    auto b = make_uint8_vec(dim, 123);

    float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
    EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f);
}

TEST(L2Uint8_FloatSpace, VariousDims) {
    std::vector<size_t> dims = {4, 8, 16, 32, 64, 128, 256};
    for (size_t dim : dims) {
        deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Uint8_L2);

        auto a = make_uint8_vec(dim, dim);
        auto b = make_uint8_vec(dim, dim + 1);

        float d = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        EXPECT_NEAR(d, l2_uint8_naive(a.data(), b.data(), dim), 1e-4f)
            << "dim=" << dim;
    }
}

TEST(L2Uint8_Batch, MatchesSingleCompare) {
   std::vector<size_t> dims = {16, 32, 64, 128, 256, 768};
   std::vector<size_t> counts = {1, 3, 4, 7, 8, 9, 15, 16, 25};

   for (size_t dim : dims) {
       for (size_t count : counts) {
           auto q = make_uint8_vec(dim, 77);

           std::vector<std::vector<uint8_t>> db(count);
           std::vector<const void*> db_ptrs(count);
           for (size_t i = 0; i < count; ++i) {
               db[i] = make_uint8_vec(dim, static_cast<int>(i * 10 + 1));
               db_ptrs[i] = db[i].data();
           }

           std::vector<float> batch_dists(count, 0.0f);
           auto dist_variant = deglib::distances::uint8_l2::select_dist(dim);

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

TEST(L2Uint8_Batch, PerformanceCompareVsBatch) {
  const size_t dim = 256;
  const size_t count = 1000;

  auto q = make_uint8_vec(dim, 77);

  std::vector<std::vector<uint8_t>> db(count);
  std::vector<const void*> db_ptrs(count);
  for (size_t i = 0; i < count; ++i) {
      db[i] = make_uint8_vec(dim, static_cast<int>(i * 10 + 1));
      db_ptrs[i] = db[i].data();
  }

  std::vector<float> batch_dists(count, 0.0f);
  auto dist_variant = deglib::distances::uint8_l2::select_dist(dim);

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
