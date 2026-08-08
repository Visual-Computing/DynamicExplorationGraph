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
// ---------------------------------------------------------------------------

TEST(Rerank, SingleQuerySingleCandidate) {
   deglib::FloatSpace fs(4, deglib::Metric::L2);
   std::vector<float> query = {0.0f, 0.0f, 0.0f, 0.0f};
   std::vector<float> base = {1.0f, 0.0f, 0.0f, 0.0f};
   uint32_t candidates[] = {0};
   uint32_t results[1];

   deglib::search::rerank(fs, query.data(), 1, base.data(), 1,
                          candidates, 1, 1, 1, results);

   EXPECT_EQ(results[0], 0u);
}

TEST(Rerank, SingleQueryMultipleCandidates) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
      1.0f, 0.0f,  // idx 0, dist = 1
      3.0f, 0.0f,  // idx 1, dist = 9
      0.5f, 0.0f,  // idx 2, dist = 0.25
   };
   uint32_t candidates[] = {0, 1, 2};
   uint32_t results[2];

   deglib::search::rerank(fs, query.data(), 1, base.data(), 3,
                          candidates, 3, 2, 1, results);

   // Closest 2: idx 2 (dist=0.25), idx 0 (dist=1.0)
   EXPECT_EQ(results[0], 2u);
   EXPECT_EQ(results[1], 0u);
}

TEST(Rerank, MultipleQueries) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
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
   uint32_t results[4]; // 2 queries x 2 k_top

   deglib::search::rerank(fs, queries.data(), 2, base.data(), 2,
                          candidates, 2, 2, 1, results);

   // Query 0: both equidistant, but idx 0 comes first
   EXPECT_EQ(results[0], 0u);
   EXPECT_EQ(results[1], 1u);

   // Query 1: idx 1 is closer (dist=8 vs dist=50)
   EXPECT_EQ(results[2], 1u);
   EXPECT_EQ(results[3], 0u);
}

TEST(Rerank, KTopZeroReturnsAllCandidates) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
      1.0f, 0.0f,  // idx 0, dist = 1
      2.0f, 0.0f,  // idx 1, dist = 4
      3.0f, 0.0f,  // idx 2, dist = 9
   };
   uint32_t candidates[] = {0, 1, 2};
   uint32_t results[3];

   // k_top=0 should use all candidates
   deglib::search::rerank(fs, query.data(), 1, base.data(), 3,
                          candidates, 3, 0, 1, results);

   EXPECT_EQ(results[0], 0u);
   EXPECT_EQ(results[1], 1u);
   EXPECT_EQ(results[2], 2u);
}

TEST(Rerank, KTopLargerThanCandidates) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
      1.0f, 0.0f,  // idx 0
   };
   uint32_t candidates[] = {0};
   uint32_t results[3];

   // k_top=3 but only 1 candidate — k_top is clamped to candidates_per_query
   deglib::search::rerank(fs, query.data(), 1, base.data(), 1,
                          candidates, 1, 3, 1, results);

   EXPECT_EQ(results[0], 0u);
}
TEST(Rerank, NullQueriesThrows) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   uint32_t candidates[] = {0};
   uint32_t results[1];

   EXPECT_THROW(deglib::search::rerank(fs, nullptr, 1, nullptr, 1,
                                        candidates, 1, 1, 1, results),
                std::invalid_argument);
}

TEST(Rerank, NullCandidatesThrows) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   std::vector<float> query = {0.0f, 0.0f};
   uint32_t results[1];

   EXPECT_THROW(deglib::search::rerank(fs, query.data(), 1, nullptr, 1,
                                        nullptr, 1, 1, 1, results),
                std::invalid_argument);
}

TEST(Rerank, NullResultsThrows) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {1.0f, 0.0f};
   uint32_t candidates[] = {0};

   EXPECT_THROW(deglib::search::rerank(fs, query.data(), 1, base.data(), 1,
                                        candidates, 1, 1, 1, nullptr),
                std::invalid_argument);
}

TEST(Rerank, InvalidCandidateIndexSkipped) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   std::vector<float> query = {0.0f, 0.0f};
   std::vector<float> base = {
      1.0f, 0.0f,  // idx 0, valid
   };
   // idx 99 is out of bounds, should be skipped
   uint32_t candidates[] = {0, 99};
   uint32_t results[2];

   deglib::search::rerank(fs, query.data(), 1, base.data(), 1,
                          candidates, 2, 2, 1, results);

   EXPECT_EQ(results[0], 0u);
   EXPECT_EQ(results[1], 0u);  // filled with query index since only 1 valid candidate
}

TEST(Rerank, UsesQueriesAsTargetsWhenBaseNull) {
   deglib::FloatSpace fs(2, deglib::Metric::L2);
   std::vector<float> query = {
      0.0f, 0.0f,  // query 0
      1.0f, 0.0f,  // query 1
   };
   uint32_t candidates[] = {
      0, 1,  // candidates for query 0
      0, 1,  // candidates for query 1
   };
   uint32_t results[4];

   // base_vectors=null → queries used as targets
   deglib::search::rerank(fs, query.data(), 2, nullptr, 0,
                          candidates, 2, 2, 1, results);

   // Query 0: idx 0 (dist=0), idx 1 (dist=1)
   EXPECT_EQ(results[0], 0u);
   EXPECT_EQ(results[1], 1u);

   // Query 1: idx 1 (dist=0), idx 0 (dist=1)
   EXPECT_EQ(results[2], 1u);
   EXPECT_EQ(results[3], 0u);
}

TEST(Rerank, InnerProductMetric) {
   deglib::FloatSpace fs(2, deglib::Metric::InnerProduct);
   std::vector<float> query = {1.0f, 0.0f};
   std::vector<float> base = {
      1.0f, 0.0f,  // idx 0, inner product = 1
      2.0f, 0.0f,  // idx 1, inner product = 2
      0.5f, 0.0f,  // idx 2, inner product = 0.5
   };
   uint32_t candidates[] = {0, 1, 2};
   uint32_t results[2];

   deglib::search::rerank(fs, query.data(), 1, base.data(), 3,
                          candidates, 3, 2, 1, results);

   // Inner product: higher is better → idx 1 (ip=2), idx 0 (ip=1)
   EXPECT_EQ(results[0], 1u);
   EXPECT_EQ(results[1], 0u);
}
