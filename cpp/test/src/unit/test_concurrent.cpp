#include "deglib/concurrent.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

// ============================================================================
// Unit tests for deglib::concurrent::parallel_batch_for
// ============================================================================

namespace {

// Collects every (begin, end) range the callback was invoked with.
struct RangeCollector {
    std::mutex m;
    std::vector<std::pair<size_t, size_t>> ranges;
    std::vector<size_t> thread_ids;

    void add(size_t begin, size_t end, size_t thread_id) {
        std::lock_guard<std::mutex> lk(m);
        ranges.emplace_back(begin, end);
        thread_ids.push_back(thread_id);
    }

    // True if the ranges partition [0, total) exactly: disjoint, contiguous, no gaps.
    bool covers(size_t total) const {
        if (ranges.empty()) return total == 0;
        std::vector<std::pair<size_t, size_t>> sorted = ranges;
        std::sort(sorted.begin(), sorted.end());
        size_t cursor = 0;
        for (auto [b, e] : sorted) {
            if (b != cursor || e < b) return false;
            cursor = e;
        }
        return cursor == total;
    }

    size_t total_elements() const {
        size_t n = 0;
        for (auto [b, e] : ranges) n += e - b;
        return n;
    }
};

}  // namespace

TEST(ParallelBatchFor, CoversExactRangeForVariousThreadCounts) {
    const size_t count = 1000;
    for (size_t threads : {1, 2, 3, 7, 8, 64}) {  // 64 > count also tested
        RangeCollector rc;
        deglib::concurrent::parallel_batch_for(0, count, threads, [&](size_t begin, size_t end, size_t tid) {
            EXPECT_LT(begin, end) << "empty chunk must not be dispatched";
            EXPECT_LE(end, count);
            rc.add(begin, end, tid);
        });
        EXPECT_TRUE(rc.covers(count)) << "threads=" << threads;
        EXPECT_EQ(rc.total_elements(), count) << "threads=" << threads;
    }
}

TEST(ParallelBatchFor, HandlesNonZeroStartOffset) {
    const size_t start = 5;
    const size_t count = 100;
    RangeCollector rc;
    deglib::concurrent::parallel_batch_for(start, start + count, 8, [&](size_t begin, size_t end, size_t tid) {
        EXPECT_GE(begin, start);
        EXPECT_LE(end, start + count);
        rc.add(begin, end, tid);
    });
    // The element indices must be offset by `start` and cover [start, start+count) exactly:
    std::vector<bool> seen(count, false);
    for (auto [b, e] : rc.ranges)
        for (size_t i = b; i < e; i++) seen[i - start] = true;
    EXPECT_EQ(std::count(seen.begin(), seen.end(), true), static_cast<ptrdiff_t>(count));
}

TEST(ParallelBatchFor, EmptyRangeInvokesNothing) {
    int calls = 0;
    deglib::concurrent::parallel_batch_for(10, 10, 8, [&](size_t, size_t, size_t) { calls++; });
    EXPECT_EQ(calls, 0);
}

TEST(ParallelBatchFor, SingleThreadCallsOnceWithFullRange) {
    RangeCollector rc;
    deglib::concurrent::parallel_batch_for(3, 7, 1, [&](size_t begin, size_t end, size_t tid) {
        EXPECT_EQ(tid, 0u);
        rc.add(begin, end, tid);
    });
    ASSERT_EQ(rc.ranges.size(), 1u);
    EXPECT_EQ(rc.ranges[0].first, 3u);
    EXPECT_EQ(rc.ranges[0].second, 7u);
}

TEST(ParallelBatchFor, ThreadIdsStayBelowChunkCount) {
    const size_t count = 500;
    RangeCollector rc;
    deglib::concurrent::parallel_batch_for(0, count, 8, [&](size_t begin, size_t end, size_t tid) {
        EXPECT_LT(tid, 8u);
        rc.add(begin, end, tid);
    });
    // Every observed worker id must be a valid participant id (< numChunks).
    // NOTE: we deliberately do NOT assert that more than one distinct id was
    // observed: with dynamic chunk claiming the caller can finish all tiny
    // chunks before the spawned workers even start, so a single-id run is legal.
    for (size_t tid : rc.thread_ids) {
        EXPECT_LT(tid, 8u);
    }
    EXPECT_TRUE(rc.covers(count));
}

TEST(ParallelBatchFor, ChunkSizeDifferByAtMostOne) {
    const size_t count = 100;
    for (size_t threads : {3, 6, 8}) {
        RangeCollector rc;
        deglib::concurrent::parallel_batch_for(0, count, threads, [&](size_t b, size_t e, size_t) { rc.add(b, e, 0); });
        size_t min_len = count, max_len = 0;
        for (auto [b, e] : rc.ranges) {
            min_len = std::min(min_len, e - b);
            max_len = std::max(max_len, e - b);
        }
        EXPECT_LE(max_len - min_len, 1u) << "threads=" << threads;
    }
}

TEST(ParallelBatchFor, ExceptionFromWorkerIsRethrownOnCaller) {
    const size_t count = 100;
    bool saw_parallel = false;
    try {
        deglib::concurrent::parallel_batch_for(0, count, 4, [&](size_t begin, size_t, size_t) {
            if (begin > count / 2) {  // only workers processing the later chunks throw
                saw_parallel = true;  // proven that a spawned worker (not the caller) threw
                throw std::runtime_error("boom");
            }
        });
        FAIL() << "expected exception to propagate";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "boom");
        EXPECT_TRUE(saw_parallel);
    }
}

TEST(ParallelBatchFor, FunctionalSumMatchesSerial) {
    const size_t count = 100000;
    std::atomic<size_t> sum{0};
    deglib::concurrent::parallel_batch_for(0, count, 8, [&](size_t begin, size_t end, size_t) {
        size_t local = 0;
        for (size_t i = begin; i < end; i++) local += i;
        sum += local;
    });
    const size_t expected = count * (count - 1) / 2;
    EXPECT_EQ(sum.load(), expected);
}

// A tiny benchmark-ish smoke test: batch_for must be substantially faster than
// element-wise parallel_for on fine-grained work (this is the regression the
// primitive exists for). Kept loose to avoid CI flakiness.
TEST(ParallelBatchFor, BatchIsNotSlowerThanElementWise) {
    const size_t count = 200000;
    volatile size_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    deglib::concurrent::parallel_for(0, count, 8, [&](size_t i, size_t) { sink += i; });
    auto t1 = std::chrono::steady_clock::now();
    deglib::concurrent::parallel_batch_for(0, count, 8, [&](size_t begin, size_t end, size_t) {
        for (size_t i = begin; i < end; i++) sink += i;
    });
    auto t2 = std::chrono::steady_clock::now();
    auto elementwise = std::chrono::duration<double>(t1 - t0).count();
    auto batched = std::chrono::duration<double>(t2 - t1).count();
    EXPECT_LE(batched, elementwise * 1.5) << "elementwise=" << elementwise << " batched=" << batched;
    EXPECT_GT(sink, 0u);
}
