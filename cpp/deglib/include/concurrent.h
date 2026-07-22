#pragma once

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "config.h"


namespace deglib::concurrent {

// Multithreaded executor
template <class FuncType>
inline void parallel_for(size_t start, size_t end, size_t numThreads, size_t batchSize, FuncType fn) {

    // default number of threads is half of the hardware concurrency
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency() / 2;
    }

    // default batch size is 1% of the total number of elements per thread
    if (batchSize <= 0) {
        batchSize =  std::max(static_cast<size_t>(1), (end-start) / (static_cast<size_t>(numThreads) * 100));
    }

    if (numThreads <= 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t batchStart = current.fetch_add(batchSize);

                    if (batchStart >= end) {
                        break;
                    }

                    size_t batchEnd = std::min(batchStart + batchSize, end);

                    try {
                        for (size_t id = batchStart; id < batchEnd; id++) {
                            fn(id, threadId);
                        }
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
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

}  // namespace deglib::concurrent
