#pragma once

#include "deglib/config.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace deglib::concurrent {

namespace detail {

/**
 * Lightweight persistent worker pool backing `parallel_for` and `parallel_batch_for`.
 *
 * Keeps OS threads (and their `thread_local` workspaces) alive across calls so repeated
 * short parallel batches avoid `std::thread` spawn/join and workspace reallocation overhead.
 * Re-entrant or concurrent callers fall back to ad-hoc `std::thread` execution without blocking.
 */
class PersistentThreadPool {
  public:
    PersistentThreadPool() = default;
    PersistentThreadPool(const PersistentThreadPool&) = delete;
    PersistentThreadPool& operator=(const PersistentThreadPool&) = delete;

    template <class WorkerFn>
    bool try_run(size_t actual_threads, WorkerFn&& worker) {
        if (busy_.exchange(true, std::memory_order_acquire)) {
            return false;
        }
        struct BusyGuard {
            std::atomic<bool>& flag;
            ~BusyGuard() noexcept { flag.store(false, std::memory_order_release); }
        } guard{busy_};

        std::atomic<size_t> remaining(actual_threads - 1);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            while (workers_.size() < actual_threads - 1) {
                const size_t tid = workers_.size() + 1;
                workers_.emplace_back([this, tid] { worker_loop(tid); });
            }
            job_fn_ = [&](size_t tid) { worker(tid); };
            active_threads_ = actual_threads;
            remaining_ = &remaining;
            ++generation_;
        }
        work_cv_.notify_all();

        worker(0);  // caller participates as worker 0

        if (remaining.load(std::memory_order_acquire) > 0) {
            std::unique_lock<std::mutex> lk(mtx_);
            done_cv_.wait(lk, [&] { return remaining.load(std::memory_order_acquire) == 0; });
        }
        return true;
    }

  private:
    void worker_loop(size_t tid) {
        size_t my_gen = 0;
        while (true) {
            std::function<void(size_t)> fn;
            std::atomic<size_t>* rem_ptr = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                work_cv_.wait(lk, [&] { return generation_ != my_gen && tid < active_threads_; });
                my_gen = generation_;
                fn = job_fn_;
                rem_ptr = remaining_;
            }
            fn(tid);
            if (rem_ptr->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(mtx_);
                done_cv_.notify_one();
            }
        }
    }

    std::atomic<bool> busy_{false};
    std::mutex mtx_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    std::function<void(size_t)> job_fn_;
    std::atomic<size_t>* remaining_ = nullptr;
    size_t active_threads_ = 1;
    size_t generation_ = 0;
};

inline PersistentThreadPool& default_pool() {
    // Intentional process-lifetime singleton (avoids Windows DLL_PROCESS_DETACH join deadlocks).
    static auto* pool = new PersistentThreadPool();
    return *pool;
}

template <class WorkerFn>
inline void run_workers(size_t actual_threads, WorkerFn&& worker) {
    if (default_pool().try_run(actual_threads, worker)) {
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(actual_threads - 1);
    for (size_t thread_id = 1; thread_id < actual_threads; ++thread_id) {
        threads.emplace_back(worker, thread_id);
    }
    worker(0);
    for (auto& thread : threads) {
        thread.join();
    }
}

}  // namespace detail

// Multithreaded executor
// The helper function copied from https://github.com/nmslib/hnswlib/blob/master/examples/cpp/example_mt_search.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template <class Function>
inline void parallel_for(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    const size_t count = end - start;
    if (count == 0) {
        return;
    }

    const size_t actualThreads = std::min(std::max<size_t>(1, numThreads), count);
    if (actualThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
        return;
    }

    std::atomic<size_t> current(start);
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    auto worker = [&](size_t threadId) {
        while (true) {
            const size_t id = current.fetch_add(1, std::memory_order_relaxed);
            if (id >= end) {
                break;
            }
            try {
                fn(id, threadId);
            } catch (...) {
                std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                lastException = std::current_exception();
                current.store(end, std::memory_order_relaxed);
                break;
            }
        }
    };

    detail::run_workers(actualThreads, worker);
    if (lastException) {
        std::rethrow_exception(lastException);
    }
}

// ---------------------------------------------------------------------------
// parallel_batch_for: parallel loop over contiguous batches.
//
// Splits [start, end) into min(end - start, actualThreads * chunksPerThread)
// contiguous chunks and invokes
//     fn(begin, end, threadId)
// once per non-empty chunk, with the half-open range [begin, end) and
// threadId in [0, actualThreads).
//
// Unlike parallel_for (which calls fn once per element and increments a
// shared atomic counter per element), the counter is only touched once per
// CHUNK. On fine-grained work (short fn bodies) that removes the cache-line
// ping-pong between workers, which is what makes element-wise loops with a
// shared counter scale poorly. When chunksPerThread > 1, chunks are claimed
// dynamically by the workers, balancing variable-cost workloads and OS
// preemption while keeping atomic contention low.
//
// The caller thread participates as worker 0 (like parallel_for). The range
// [begin, end) is passed so callers can write `for (size_t i = begin; i < end;
// ++i) { ... }` or use std::copy_n / std::memcpy on the whole slice.
// Exceptions thrown by fn are captured and re-thrown on the caller thread.
// ---------------------------------------------------------------------------
template <class Function>
inline void parallel_batch_for(size_t start, size_t end, size_t numThreads, Function fn, size_t chunksPerThread = 1) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    const size_t count = end - start;
    if (count == 0) {
        return;
    }

    const size_t actualThreads = std::min(std::max<size_t>(1, numThreads), count);
    if (actualThreads == 1) {
        fn(start, end, 0);
        return;
    }

    const size_t mult = std::max<size_t>(1, chunksPerThread);
    const size_t numChunks = std::min(count, actualThreads * mult);

    // Static partition of [0, count) into numChunks contiguous chunks:
    //   base = count / numChunks, rem = count % numChunks
    //   chunk c = [c*base + min(c, rem),  c*base + min(c, rem) + base + (c < rem))
    // The first `rem` chunks have base+1 elements, the rest have base.
    const size_t base = count / numChunks;
    const size_t rem = count % numChunks;

    std::atomic<size_t> current(0);
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    auto worker = [&](size_t threadId) {
        while (true) {
            const size_t c = current.fetch_add(1, std::memory_order_relaxed);
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
                current.store(numChunks, std::memory_order_relaxed);
                break;
            }
        }
    };

    detail::run_workers(actualThreads, worker);
    if (lastException) {
        std::rethrow_exception(lastException);
    }
}

}  // namespace deglib::concurrent
