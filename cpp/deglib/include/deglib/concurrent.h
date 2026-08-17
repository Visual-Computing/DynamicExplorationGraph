#pragma once

#include "deglib/config.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace deglib::concurrent {

// Multithreaded executor
// The helper function copied from https://github.com/nmslib/hnswlib/blob/master/examples/cpp/example_mt_search.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template <class Function>
inline void parallel_for(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

// ---------------------------------------------------------------------------
// parallel_batch_for: parallel loop over contiguous batches.
//
// Splits [start, end) into min(numThreads, end-start) contiguous chunks and
// invokes
//     fn(begin, end, threadId)
// exactly once per non-empty chunk, with the half-open range [begin, end).
//
// Unlike parallel_for (which calls fn once per element and increments a
// shared atomic counter per element), the counter is only touched once per
// CHUNK. On fine-grained work (short fn bodies) that removes the cache-line
// ping-pong between workers, which is what makes element-wise loops with a
// shared counter scale poorly. Chunks are still claimed dynamically by the
// workers, so load stays balanced even when the work per element varies.
// Contiguous chunks also give better cache locality than interleaved
// element-wise claiming.
//
// The caller thread participates as worker 0 (like parallel_for). The range
// [begin, end) is passed so callers can write `for (size_t i = begin; i < end;
// ++i) { ... }` or use std::copy_n / std::memcpy on the whole slice.
// Exceptions thrown by fn are captured and re-thrown on the caller thread.
// ---------------------------------------------------------------------------
template <class Function>
inline void parallel_batch_for(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    const size_t count = end - start;
    if (count == 0) {
        return;
    }

    const size_t numChunks = std::min(numThreads, count);
    if (numChunks == 1) {
        fn(start, end, 0);
        return;
    }

    // Static partition of [0, count) into numChunks contiguous chunks:
    //   base = count / numChunks, rem = count % numChunks
    //   chunk c = [c*base + min(c, rem),  c*base + min(c, rem) + base + (c < rem))
    // The first `rem` chunks have base+1 elements, the rest have base.
    const size_t base = count / numChunks;
    const size_t rem = count % numChunks;

    std::atomic<size_t> current(0);
    std::vector<std::thread> threads;

    // keep track of exceptions in threads (same scheme as parallel_for)
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    auto worker = [&](size_t threadId) {
        while (true) {
            size_t c = current.fetch_add(1);
            if (c >= numChunks) {
                break;
            }
            const size_t cbegin = start + c * base + std::min(c, rem);
            const size_t clen = base + (c < rem ? 1 : 0);
            try {
                fn(cbegin, cbegin + clen, threadId);
            } catch (...) {
                std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                lastException = std::current_exception();
                current = numChunks;  // stop the remaining workers
                break;
            }
        }
    };

    threads.reserve(numChunks - 1);
    for (size_t threadId = 1; threadId < numChunks; ++threadId) {
        threads.emplace_back(worker, threadId);
    }
    worker(0);  // caller participates
    for (auto& thread : threads) {
        thread.join();
    }
    if (lastException) {
        std::rethrow_exception(lastException);
    }
}

}  // namespace deglib::concurrent
