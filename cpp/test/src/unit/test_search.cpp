// test_search.cpp — Unit tests for deglib::search::Reranker (search.h)
//
// Tests the Reranker class which re-ranks candidate neighbor indices
// for queries using exact FloatSpace distances.

#include "deglib/distances.h"
#include "deglib/search.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <random>
#include <span>
#include <vector>

// ---------------------------------------------------------------------------
//  Reranker — basic functionality
// ---------------------------------------------------------------------------

TEST(Rerank, SingleQuerySingleCandidate) {
   deglib::distances::FloatSpace fs(4, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
   std::vector<float> base = {1.0f, 0.0f, 0.0f, 0.0f};
   uint32_t candidates[] = {0};

   deglib::search::Reranker<float> reranker(fs, base.data(), 1);
   auto results = reranker.rerank(query.data(), 1, candidates, 1, 1, 1);

   ASSERT_EQ(results.size(), 1u);
   auto res = std::move(results[0]);
   EXPECT_EQ(res.size(), 1u);
   EXPECT_EQ(res.top().getIdentifier(), 0u);
}

TEST(Rerank, SingleQueryMultipleCandidates) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
       1.0f, 0.0f,  // idx 0, dist = 1
       3.0f, 0.0f,  // idx 1, dist = 9
       0.5f, 0.0f,  // idx 2, dist = 0.25
   };
   uint32_t candidates[] = {0, 1, 2};

   deglib::search::Reranker<float> reranker(fs, base.data(), 3);
   auto results = reranker.rerank(query.data(), 1, candidates, 3, 2, 1);

   ASSERT_EQ(results.size(), 1u);
   auto res = std::move(results[0]);
   EXPECT_EQ(res.size(), 2u);

   // Closest 2: idx 2 (dist=0.25), idx 0 (dist=1.0)
   // ResultSet is a max-heap (largest dist on top): top() is idx 0 (1.0), popping leaves idx 2 (0.25)
   EXPECT_EQ(res.top().getIdentifier(), 0u);
   res.pop();
   EXPECT_EQ(res.top().getIdentifier(), 2u);
}

TEST(Rerank, MultipleQueries) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> queries = {
       0.0f, 0.0f,  // query 0
       5.0f, 5.0f,  // query 1
   };
   std::vector<float> base = {
       0.0f, 0.0f,  // idx 0
       1.0f, 1.0f,  // idx 1
   };
   uint32_t candidates[] = {
       0, 1,  // candidates for query 0
       0, 1,  // candidates for query 1
   };

   deglib::search::Reranker<float> reranker(fs, base.data(), 2);
   auto results = reranker.rerank(queries.data(), 2, candidates, 2, 2, 1);

   ASSERT_EQ(results.size(), 2u);

   // Query 0: idx 0 (dist=0), idx 1 (dist=2). Top (worst) is idx 1.
   auto res0 = std::move(results[0]);
   EXPECT_EQ(res0.top().getIdentifier(), 1u);
   res0.pop();
   EXPECT_EQ(res0.top().getIdentifier(), 0u);

   // Query 1: idx 1 is closer (dist=8 vs dist=50). Top (worst) is idx 0.
   auto res1 = std::move(results[1]);
   EXPECT_EQ(res1.top().getIdentifier(), 0u);
   res1.pop();
   EXPECT_EQ(res1.top().getIdentifier(), 1u);
}

TEST(Rerank, KTopZeroReturnsAllCandidates) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
       1.0f, 0.0f,  // idx 0, dist = 1
       2.0f, 0.0f,  // idx 1, dist = 4
       3.0f, 0.0f,  // idx 2, dist = 9
   };
   uint32_t candidates[] = {0, 1, 2};

   deglib::search::Reranker<float> reranker(fs, base.data(), 3);
   // k_top=0 should use all candidates
   auto results = reranker.rerank(query.data(), 1, candidates, 3, 0, 1);

   ASSERT_EQ(results.size(), 1u);
   EXPECT_EQ(results[0].size(), 3u);
}

TEST(Rerank, KTopLargerThanCandidates) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
       1.0f, 0.0f,  // idx 0
   };
   uint32_t candidates[] = {0};

   deglib::search::Reranker<float> reranker(fs, base.data(), 1);
   // k_top=3 but only 1 candidate — k_top is clamped to candidates_per_query
   auto results = reranker.rerank(query.data(), 1, candidates, 1, 3, 1);

   ASSERT_EQ(results.size(), 1u);
   EXPECT_EQ(results[0].size(), 1u);
   EXPECT_EQ(results[0].top().getIdentifier(), 0u);
}

TEST(Rerank, NullQueriesThrows) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   uint32_t candidates[] = {0};
   deglib::search::Reranker<float> reranker(fs, nullptr, 0);

   EXPECT_THROW(reranker.rerank(nullptr, 1, candidates, 1, 1, 1), std::invalid_argument);
}

TEST(Rerank, NullCandidatesThrows) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   deglib::search::Reranker<float> reranker(fs, nullptr, 0);

   EXPECT_THROW(reranker.rerank(query.data(), 1, nullptr, 1, 1, 1), std::invalid_argument);
}

TEST(Rerank, InvalidCandidateIndexSkipped) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
       1.0f, 0.0f,  // idx 0, valid
   };
   // idx 99 is out of bounds, should be skipped
   uint32_t candidates[] = {0, 99};

   deglib::search::Reranker<float> reranker(fs, base.data(), 1);
   auto results = reranker.rerank(query.data(), 1, candidates, 2, 2, 1);

   ASSERT_EQ(results.size(), 1u);
   EXPECT_EQ(results[0].size(), 1u);
   EXPECT_EQ(results[0].top().getIdentifier(), 0u);
}

TEST(Rerank, SelfRerankWhenBaseVectorsAreQueries) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {
       0.0f, 0.0f,  // query 0
       1.0f, 0.0f,  // query 1
   };
   uint32_t candidates[] = {
       0, 1,  // candidates for query 0
       0, 1,  // candidates for query 1
   };

   // queries used as target base vectors
   deglib::search::Reranker<float> reranker(fs, query.data(), 2);
   auto results = reranker.rerank(query.data(), 2, candidates, 2, 2, 1);

   ASSERT_EQ(results.size(), 2u);

   // Query 0: idx 0 (dist=0), idx 1 (dist=1). Worst is idx 1.
   auto res0 = std::move(results[0]);
   EXPECT_EQ(res0.top().getIdentifier(), 1u);
   res0.pop();
   EXPECT_EQ(res0.top().getIdentifier(), 0u);

   // Query 1: idx 1 (dist=0), idx 0 (dist=1). Worst is idx 0.
   auto res1 = std::move(results[1]);
   EXPECT_EQ(res1.top().getIdentifier(), 0u);
   res1.pop();
   EXPECT_EQ(res1.top().getIdentifier(), 1u);
}

TEST(Rerank, InnerProductMetric) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_InnerProduct);
   std::vector<float> query = {1.0f, 0.0f};
   std::vector<float> base = {
       1.0f, 0.0f,  // idx 0, inner product = 1
       2.0f, 0.0f,  // idx 1, inner product = 2
       0.5f, 0.0f,  // idx 2, inner product = 0.5
   };
   uint32_t candidates[] = {0, 1, 2};

   deglib::search::Reranker<float> reranker(fs, base.data(), 3);
   auto results = reranker.rerank(query.data(), 1, candidates, 3, 2, 1);

   ASSERT_EQ(results.size(), 1u);
   auto res = std::move(results[0]);
   EXPECT_EQ(res.size(), 2u);

   // Inner product: lower distance value stored in FloatSpace (e.g. -2.0 vs -1.0 vs -0.5).
   // Closest/best: idx 1 (ip=2, converted dist=-2), idx 0 (ip=1, converted dist=-1).
   // Max-heap top() returns the worst distance in top 2 (idx 0).
   EXPECT_EQ(res.top().getIdentifier(), 0u);
   res.pop();
   EXPECT_EQ(res.top().getIdentifier(), 1u);
}

TEST(Rerank, SingleQuerySpanOverloads) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
       0.0f, 1.0f,  // dist^2 = 1
       0.0f, 3.0f,  // dist^2 = 9
       0.0f, 2.0f,  // dist^2 = 4
   };
   std::vector<uint32_t> candidates = {0, 1, 2};

   deglib::search::Reranker<float> reranker(fs, base.data(), 3);
   auto q_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(query.data()), query.size() * sizeof(float));

   // 1. Single query returning ResultSet
   auto heap = reranker.rerank(q_span, std::span<const uint32_t>(candidates), 2);
   EXPECT_EQ(heap.size(), 2u);

   // 2. Single query in-place span (sorted, with distances)
   std::vector<uint32_t> out_indices(2);
   std::vector<float> out_dists(2);
   uint32_t count = reranker.rerank(
       q_span, std::span<const uint32_t>(candidates), 2, std::span<uint32_t>(out_indices), std::span<float>(out_dists),
       /*return_distances=*/true, /*unsorted=*/false
   );

   EXPECT_EQ(count, 2u);
   EXPECT_EQ(out_indices[0], 0u);
   EXPECT_NEAR(out_dists[0], 1.0f, 1e-5f);
   EXPECT_EQ(out_indices[1], 2u);
   EXPECT_NEAR(out_dists[1], 4.0f, 1e-5f);
}

TEST(Rerank, RerankerClassDirectUsage) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
       0.0f, 1.0f,  // dist^2 = 1
       0.0f, 3.0f,  // dist^2 = 9
       0.0f, 2.0f,  // dist^2 = 4
   };
   std::vector<uint32_t> candidates = {0, 1, 2};

   deglib::search::Reranker<float> reranker(fs, base.data(), 3);
   EXPECT_EQ(reranker.getNumBaseVectors(), 3u);
   EXPECT_EQ(reranker.getBaseVectors(), base.data());

   std::vector<uint32_t> out_indices(2);
   std::vector<float> out_dists(2);
   uint32_t count = reranker.rerank(query.data(), 2, candidates.data(), candidates.size(), 2, out_indices.data(), out_dists.data(), true, false);
   EXPECT_EQ(count, 2u);
   EXPECT_EQ(out_indices[0], 0u);
   EXPECT_NEAR(out_dists[0], 1.0f, 1e-5f);
   EXPECT_EQ(out_indices[1], 2u);
   EXPECT_NEAR(out_dists[1], 4.0f, 1e-5f);
}

TEST(Rerank, PrefetchEnabledByDefaultAndTogglesCleanly) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> base = {0.0f, 1.0f, 0.0f, 3.0f, 0.0f, 2.0f};
   std::vector<uint32_t> candidates = {0, 1, 2};
   std::vector<float> query = {0.0f, 0.0f};

   deglib::search::Reranker<float> reranker(fs, base.data(), 3);

   // Default must be the prefetching config, otherwise large out-of-cache rerank runs silently lose throughput.
   EXPECT_GT(reranker.getPo(), 0);

   reranker.setPo(0);
   EXPECT_EQ(reranker.getPo(), 0);
   reranker.setPl(-4);
   EXPECT_EQ(reranker.getPl(), 0);

   reranker.setPrefetch(12, 3);
   EXPECT_EQ(reranker.getPo(), 12);
   EXPECT_EQ(reranker.getPl(), 3);

   // Toggling the prefetch config must not change the ranking.
   std::vector<uint32_t> out_indices(2);
   std::vector<float> out_dists(2);
   uint32_t count = reranker.rerank(query.data(), 2, candidates.data(), candidates.size(), 2, out_indices.data(), out_dists.data(), true, false);
   EXPECT_EQ(count, 2u);
   EXPECT_EQ(out_indices[0], 0u);
   EXPECT_NEAR(out_dists[0], 1.0f, 1e-5f);
   EXPECT_EQ(out_indices[1], 2u);
   EXPECT_NEAR(out_dists[1], 4.0f, 1e-5f);
}

TEST(Rerank, PrefetchAndNoPrefetchPathsAgreeOnOutOfCacheWorkingSet) {
   // 40k x 128-dim float32 = 20 MB of base vectors with random candidate order: every candidate
   // misses cache, which is the regime the lookahead prefetcher exists for. Both paths must
   // produce byte-identical top-k, so the prefetcher is a pure speed knob and never a semantic one.
   const uint32_t dim = 128;
   const size_t num_base = 40000;
   const size_t num_queries = 200;
   const size_t candidates_per_query = 200;
   const uint32_t k_top = 10;

   std::mt19937 rng(1234);
   std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

   std::vector<float> base(num_base * dim);
   for (auto& v : base) v = uniform(rng);
   std::vector<float> queries(num_queries * dim);
   for (auto& v : queries) v = uniform(rng);
   std::vector<uint32_t> candidates(num_queries * candidates_per_query);
   for (auto& c : candidates) c = rng() % num_base;

   deglib::distances::FloatSpace fs(dim, deglib::distances::Metric::FP32_L2);
   deglib::search::Reranker<float> reranker(fs, base.data(), num_base);

   reranker.setPrefetch(0, 0);
   auto unprefetched = reranker.rerank(queries.data(), num_queries, candidates.data(), candidates_per_query, k_top, 1);
   reranker.setPrefetch(8, 0);
   auto prefetched = reranker.rerank(queries.data(), num_queries, candidates.data(), candidates_per_query, k_top, 1);

   ASSERT_EQ(unprefetched.size(), num_queries);
   ASSERT_EQ(prefetched.size(), num_queries);
   for (size_t q = 0; q < num_queries; ++q) {
       ASSERT_EQ(unprefetched[q].size(), k_top) << "query " << q;
       for (size_t i = 0; i < k_top; ++i) {
           EXPECT_EQ(unprefetched[q][i].getIdentifier(), prefetched[q][i].getIdentifier()) << "query " << q << " rank " << i;
           EXPECT_FLOAT_EQ(unprefetched[q][i].getDistance(), prefetched[q][i].getDistance()) << "query " << q << " rank " << i;
       }
   }
}
