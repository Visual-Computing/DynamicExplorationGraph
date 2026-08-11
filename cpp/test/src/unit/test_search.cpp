// test_search.cpp — Unit tests for deglib::search::rerank (search.h)
//
// Tests the rerank function which re-ranks candidate neighbor indices
// for queries using exact FloatSpace distances.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "deglib/search.h"
#include "deglib/distances.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
//  rerank — basic functionality
// ---------------------------------------------------------------------------// ---------------------------------------------------------------------------
//  rerank — basic functionality
// ---------------------------------------------------------------------------

TEST(Rerank, SingleQuerySingleCandidate) {
   deglib::distances::FloatSpace fs(4, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
   std::vector<float> base = {1.0f, 0.0f, 0.0f, 0.0f};
   uint32_t candidates[] = {0};

   auto results = deglib::search::rerank(fs, query.data(), 1, base.data(), 1,
                                         candidates, 1, 1, 1);

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

   auto results = deglib::search::rerank(fs, query.data(), 1, base.data(), 3,
                                         candidates, 3, 2, 1);

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

   auto results = deglib::search::rerank(fs, queries.data(), 2, base.data(), 2,
                                         candidates, 2, 2, 1);

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

   // k_top=0 should use all candidates
   auto results = deglib::search::rerank(fs, query.data(), 1, base.data(), 3,
                                         candidates, 3, 0, 1);

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

   // k_top=3 but only 1 candidate — k_top is clamped to candidates_per_query
   auto results = deglib::search::rerank(fs, query.data(), 1, base.data(), 1,
                                         candidates, 1, 3, 1);

   ASSERT_EQ(results.size(), 1u);
   EXPECT_EQ(results[0].size(), 1u);
   EXPECT_EQ(results[0].top().getIdentifier(), 0u);
}

TEST(Rerank, NullQueriesThrows) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   uint32_t candidates[] = {0};

   EXPECT_THROW(deglib::search::rerank(fs, nullptr, 1, nullptr, 1,
                                        candidates, 1, 1, 1),
                std::invalid_argument);
}

TEST(Rerank, NullCandidatesThrows) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};

   EXPECT_THROW(deglib::search::rerank(fs, query.data(), 1, nullptr, 1,
                                        nullptr, 1, 1, 1),
                std::invalid_argument);
}

TEST(Rerank, InvalidCandidateIndexSkipped) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
      1.0f, 0.0f,  // idx 0, valid
   };
   // idx 99 is out of bounds, should be skipped
   uint32_t candidates[] = {0, 99};

   auto results = deglib::search::rerank(fs, query.data(), 1, base.data(), 1,
                                         candidates, 2, 2, 1);

   ASSERT_EQ(results.size(), 1u);
   EXPECT_EQ(results[0].size(), 1u);
   EXPECT_EQ(results[0].top().getIdentifier(), 0u);
}

TEST(Rerank, UsesQueriesAsTargetsWhenBaseNull) {
   deglib::distances::FloatSpace fs(2, deglib::distances::Metric::FP32_L2);
   std::vector<float> query = {
      0.0f, 0.0f,  // query 0
      1.0f, 0.0f,  // query 1
   };
   uint32_t candidates[] = {
      0, 1,  // candidates for query 0
      0, 1,  // candidates for query 1
   };

   // base_vectors=null → queries used as targets
   auto results = deglib::search::rerank(fs, query.data(), 2, nullptr, 0,
                                         candidates, 2, 2, 1);

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

   auto results = deglib::search::rerank(fs, query.data(), 1, base.data(), 3,
                                         candidates, 3, 2, 1);

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
