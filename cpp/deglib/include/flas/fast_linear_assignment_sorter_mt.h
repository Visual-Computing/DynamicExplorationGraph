#ifndef FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_MT_H
#define FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_MT_H

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <array>

#include "fast_linear_assignment_sorter.h"

// ---------------------------------------------------------------------------
// Multi-Threaded 1D-FLAS Sorter
//
// Architecture & Components:
//  - Persistent ThreadPool: Worker threads are spawned once during the initial
//    call to `do_sorting_1d` and reused across all iterations to eliminate
//    thread spawn overhead.
//  - Thread-Local Isolation: Each worker owns a dedicated `ThreadWorker` containing
//    an independent `RandomEngine` (preventing lock contention) and a local
//    `SwapBuffers` instance (holding candidate positions, quantized distance LUT,
//    and Jonker-Volgenant JVScratch solver state).
//  - Non-Blocking Position Locks: Fine-grained spin-free atomic flags (`PositionLocks`)
//    prevent data races when swapping elements in the shared `map_fields` array.
//  - Kernel Reuse: Delegates directly to core FLAS kernels (`flas::do_swaps`,
//    `filter_weighted_som_1d`).
// ---------------------------------------------------------------------------

namespace flas {

// ---------------------------------------------------------------------------
// ThreadWorker: Holds thread-local scratch memory and RNG for a worker thread.
//   - `rng`: Independent per-thread pseudo-random engine.
//   - `swaps`: Thread-local SwapBuffers instance encapsulating candidate positions,
//     distance LUT, and Jonker-Volgenant solver state for allocation-free inner loops.
// ---------------------------------------------------------------------------
struct ThreadWorker {
  RandomEngine rng;
  SwapBuffers swaps;
};

// ---------------------------------------------------------------------------
// PositionLocks: Fine-grained atomic flag lock pool using `std::atomic_flag`.
// Prevents data races when multiple threads swap elements in `map_fields` concurrently.
// ---------------------------------------------------------------------------
class PositionLocks {
public:
  explicit PositionLocks(int count) {
    pool_size_ = std::min(count, kPoolCap);
    flags_ = std::make_unique<std::atomic_flag[]>(static_cast<size_t>(pool_size_));
    for (int i = 0; i < pool_size_; i++)
      flags_[static_cast<size_t>(i)].clear(std::memory_order_relaxed);
  }

  // Attempts non-blocking acquisition of target position flags in sorted order.
  inline bool try_lock_positions(const int *positions, int n) {
    for (int i = 0; i < n; i++) {
      std::atomic_flag *flag = &flags_[static_cast<size_t>(positions[i] % pool_size_)];
      if (flag->test_and_set(std::memory_order_acquire)) {
        for (int k = 0; k < i; k++)
          flags_[static_cast<size_t>(positions[k] % pool_size_)].clear(std::memory_order_release);
        return false;
      }
    }
    return true;
  }

  // Releases locked position flags.
  inline void unlock_positions(const int *positions, int n) {
    for (int i = 0; i < n; i++)
      flags_[static_cast<size_t>(positions[i] % pool_size_)].clear(std::memory_order_release);
  }

private:
  static constexpr int kPoolCap = 1 << 20; // 1M slots (~1MB contiguous array)
  int pool_size_;
  std::unique_ptr<std::atomic_flag[]> flags_;
};

// ---------------------------------------------------------------------------
// SwapThreadPool: Persistent worker thread pool.
// Uses C++20 `std::barrier` to synchronize phase execution across iterations.
// ---------------------------------------------------------------------------
class SwapThreadPool {
public:
  explicit SwapThreadPool(int num_threads)
      : n_(num_threads < 1 ? 1 : num_threads), bar_(n_, ResetGo{&go_}) {
    for (int t = 1; t < n_; ++t)
      workers_.emplace_back([this, t] { worker_main(t); });
  }

  ~SwapThreadPool() {
    {
      std::lock_guard lk(m_);
      stop_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto &w : workers_)
      if (w.joinable()) w.join();
  }

  SwapThreadPool(const SwapThreadPool &) = delete;
  SwapThreadPool &operator=(const SwapThreadPool &) = delete;

  // Executes `fn(index, thread_id)` in parallel across the index range [0, end).
  template <class Fn>
  void parallel_for(size_t end, const Fn &fn) {
    const int n = n_;
    if (n <= 1 || end == 0) {
      for (size_t id = 0; id < end; ++id) fn(id, 0);
      return;
    }
    {
      std::lock_guard lk(m_);
      cfg_end_ = end;
      cfg_fn_ = fn;
      go_.store(true, std::memory_order_release);
    }
    cv_.notify_all();

    run_chunk(fn, 0, end, n);
    bar_.arrive_and_wait();
  }

private:
  struct ResetGo {
    std::atomic<bool> *flag = nullptr;
    void operator()() const noexcept { flag->store(false, std::memory_order_release); }
  };

  void worker_main(int tid) {
    for (;;) {
      size_t end;
      std::function<void(size_t, size_t)> fn;
      {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] {
          return stop_.load(std::memory_order_acquire) ||
                 go_.load(std::memory_order_acquire);
        });
        if (stop_.load(std::memory_order_acquire)) return;
        end = cfg_end_;
        fn = cfg_fn_;
      }
      run_chunk(fn, tid, end, n_);
      bar_.arrive_and_wait();
    }
  }

  template <class Fn>
  static void run_chunk(const Fn &fn, int tid, size_t end, int n) {
    size_t base = end / static_cast<size_t>(n);
    size_t rem = end % static_cast<size_t>(n);
    size_t start = (static_cast<size_t>(tid) < rem)
                       ? static_cast<size_t>(tid) * (base + 1)
                       : rem * (base + 1) + (static_cast<size_t>(tid) - rem) * base;
    size_t len = base + (static_cast<size_t>(tid) < rem ? 1 : 0);
    for (size_t i = 0; i < len; ++i)
      fn(start + i, static_cast<size_t>(tid));
  }

private:
  int n_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<bool> go_{false};
  std::atomic<bool> stop_{false};
  size_t cfg_end_ = 0;
  std::function<void(size_t, size_t)> cfg_fn_;
  std::barrier<ResetGo> bar_;
};

// ---------------------------------------------------------------------------
// Thread-safe overload of `find_swap_positions_1d`:
// Samples candidate swap positions within a localized random window using an explicit RNG.
// ---------------------------------------------------------------------------
inline int find_swap_positions_1d(const FlasContext &ctx, SwapBuffers &swaps, std::span<const int> swap_indices, int num_swap_indices, RandomEngine &rng) {
  std::uniform_int_distribution<int> pos_dist(0, ctx.count - 1);
  int x0 = pos_dist(rng);

  int x_start = std::max(0, std::min(x0 - num_swap_indices / 2, ctx.count - num_swap_indices));

  int max_sp = swaps.num_swap_positions();
  int start_index = 0;
  if (num_swap_indices > max_sp) {
    std::uniform_int_distribution<int> index_dist(0, num_swap_indices - max_sp - 1);
    start_index = index_dist(rng);
  }

  int num_swap_positions = 0;
  for (int j = start_index; j < num_swap_indices && num_swap_positions < max_sp; j++) {
    int dx = swap_indices[j];
    int pos = std::min(ctx.count - 1, std::max(0, x_start + dx));
    swaps.swap_positions[num_swap_positions++] = pos;
  }
  return num_swap_positions;
}

// ---------------------------------------------------------------------------
// Helper: Performs try-locking on target candidate positions and executes swaps.
//  1. Copies and deduplicates candidate positions into a stack buffer (`std::array<int, 32>`).
//  2. Attempts atomic try-locking via `PositionLocks`. Aborts probe if target positions are locked.
//  3. Executes `flas::do_swaps` from the base header.
//  4. Releases all locked target positions.
// ---------------------------------------------------------------------------
inline bool try_lock_and_swap(const FlasContext &ctx, SomGrid &grid, ThreadWorker &worker, int num_swaps, PositionLocks &locks) {
  if (num_swaps == 0) return false;

  std::array<int, 32> lock_pos;
  int n = std::min(num_swaps, static_cast<int>(lock_pos.size()));
  for (int i = 0; i < n; i++) {
    lock_pos[static_cast<size_t>(i)] = worker.swaps.swap_positions[i];
  }
  std::sort(lock_pos.begin(), lock_pos.begin() + n);
  int unique = 0;
  for (int i = 0; i < n; i++) {
    if (unique == 0 || lock_pos[static_cast<size_t>(i)] != lock_pos[static_cast<size_t>(unique - 1)]) {
      lock_pos[static_cast<size_t>(unique++)] = lock_pos[static_cast<size_t>(i)];
    }
  }

  if (!locks.try_lock_positions(lock_pos.data(), unique)) {
    return false;
  }

  do_swaps(ctx, grid, worker.swaps, num_swaps);

  locks.unlock_positions(lock_pos.data(), unique);
  return true;
}

// ---------------------------------------------------------------------------
// Main Entry Point: Multi-Threaded 1D FLAS Sorting
// Executes 5 steps per neighborhood radius iteration:
//   1. Parallel copy of feature vectors to `SomGrid`
//   2. Serial 1D sliding-window moving-average filter (`filter_weighted_som_1d`)
//   3. Shuffle active index window per worker
//   4. Parallel candidate swap probes with non-blocking try-locking
//   5. Progress callback and termination evaluation
// ---------------------------------------------------------------------------
inline void do_sorting_1d(
  std::span<MapField> map_fields, int dim, const FlasSettings &settings, RandomEngine &rng,
  const std::function<bool(float)>& callback, int num_threads
) {
  int count = static_cast<int>(map_fields.size());
  if (count <= 0) return;

  if (num_threads <= 1) {
    do_sorting_1d(map_fields, dim, settings, rng, callback);
    return;
  }

  float rad = static_cast<float>(count) * settings.initial_radius_factor;
  const int num_iterations = static_cast<int>(ceil(-log(rad / settings.radius_end) / log(settings.radius_decay)));
  int iteration_counter = 0;
  if (callback && callback(0.f))
    return;

  FlasContext ctx(map_fields, count, dim, rng, settings.metric);
  SomGrid grid(count, dim);

  SwapThreadPool pool(num_threads);
  PositionLocks position_locks(ctx.count);

  const int max_sp = std::min(count, settings.max_swap_positions);
  std::vector<ThreadWorker> workers(static_cast<size_t>(num_threads));
  std::uniform_int_distribution<unsigned int> seed_dist;
  const unsigned int base_seed = seed_dist(ctx.rng);

  const int init_radius = std::max(1, std::min(count / 2, static_cast<int>(std::round(rad))));
  const int max_num_swap_indices = std::max(std::min(2 * init_radius + 1, count), max_sp);

  for (int t = 0; t < num_threads; t++) {
    workers[static_cast<size_t>(t)].rng = RandomEngine(base_seed + static_cast<unsigned int>(t));
    workers[static_cast<size_t>(t)].swaps.resize(static_cast<size_t>(max_sp), static_cast<size_t>(max_num_swap_indices));
  }

  do {
    // 1. Parallel copy of current feature vectors to SOM grid
    pool.parallel_for(static_cast<size_t>(count),
        [&](size_t i, size_t /*tid*/) {
          std::copy_n(
              ctx.map_fields[static_cast<int>(i)].feature,
              static_cast<size_t>(dim),
              grid.row(static_cast<int>(i), dim));
        });

    int radius = std::max(1, static_cast<int>(std::round(rad)));
    int radius_1d = std::max(1, std::min(count / 2, radius));
    rad *= settings.radius_decay;

    // 2. Serial 1D sliding-window moving-average filter
    for (int i = 0; i < settings.num_filters; i++)
      filter_weighted_som_1d(radius_1d, ctx, grid);

    // 3. Parallel prepare and shuffle active index window per worker
    int num_swap_indices = std::min(2 * radius + 1, ctx.count);
    while (num_swap_indices < max_sp && num_swap_indices < ctx.count) num_swap_indices++;

    pool.parallel_for(static_cast<size_t>(num_threads), [&](size_t t, size_t) {
      ThreadWorker &worker = workers[t];
      if (worker.swaps.swap_indices.size() < static_cast<size_t>(num_swap_indices)) {
        worker.swaps.swap_indices.resize(static_cast<size_t>(num_swap_indices));
      }
      for (int i = 0; i < num_swap_indices; i++)
        worker.swaps.swap_indices[i] = i;
      shuffle_array(std::span<int>(worker.swaps.swap_indices.data(), static_cast<size_t>(num_swap_indices)), worker.rng);
    });

    // 4. Parallel candidate swap probes with position try-locking
    const int num_swap_tries = std::max(1, static_cast<int>(settings.sample_factor * static_cast<float>(ctx.count) / static_cast<float>(max_sp)));

    if (num_swap_tries > 1) {
      pool.parallel_for(
          static_cast<size_t>(num_swap_tries),
          [&](size_t /*try_id*/, size_t thread_id) {
            ThreadWorker &worker = workers[thread_id];
            std::span<const int> active_indices(worker.swaps.swap_indices.data(), static_cast<size_t>(num_swap_indices));
            int num_swaps = find_swap_positions_1d(ctx, worker.swaps, active_indices, num_swap_indices, worker.rng);
            try_lock_and_swap(ctx, grid, worker, num_swaps, position_locks);
          });
    } else {
      std::span<const int> active_indices(workers[0].swaps.swap_indices.data(), static_cast<size_t>(num_swap_indices));
      int num_swaps = find_swap_positions_1d(ctx, workers[0].swaps, active_indices, num_swap_indices, workers[0].rng);
      do_swaps(ctx, grid, workers[0].swaps, num_swaps);
    }

    // 5. Progress reporting and early termination evaluation
    iteration_counter++;
    float progress = static_cast<float>(iteration_counter) / static_cast<float>(num_iterations);
    if (callback && callback(progress))
      break;
  } while (rad > settings.radius_end);
}

} // namespace flas

#endif // FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_MT_H
