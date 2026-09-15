#include "deglib/distances.h"
#include "deglib/search.h"
#include "deglib/builder.h"
#include "deglib/graph.h"
#include "deglib/search/searcher.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include <span>

namespace {

// Deterministic PRNG for generating benchmark data
inline uint32_t benchmark_prng(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

inline float benchmark_float(uint32_t& state, float min_val = -1.0f, float max_val = 1.0f) {
    uint32_t val = benchmark_prng(state) >> 8;
    float u = static_cast<float>(val) / 16777215.0f;
    return min_val + u * (max_val - min_val);
}

}  // namespace

TEST(SearchRegression, Rerank_FP32_L2_Benchmark) {
    const size_t dim = 512;
    const size_t num_queries = 5000;
    const size_t num_base = 100000;
    const size_t candidates_per_query = 200;
    const size_t k_top = 10;
    const size_t num_threads = 0;  // auto-detect

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    // Generate random base vectors and query vectors
    std::vector<float> base_vectors(num_base * dim);
    std::vector<float> query_vectors(num_queries * dim);
    uint32_t rng = 1337;

    for (auto& v : base_vectors) v = benchmark_float(rng);
    for (auto& v : query_vectors) v = benchmark_float(rng);

    // Generate candidate indices per query (random indices in [0, num_base))
    std::vector<uint32_t> candidates(num_queries * candidates_per_query);
    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t c = 0; c < candidates_per_query; ++c) {
            candidates[q * candidates_per_query + c] = benchmark_prng(rng) % num_base;
        }
    }

   deglib::search::Reranker<float> reranker(space, base_vectors.data(), num_base);

   // Warm-up run
   auto warm_results = reranker.rerank(query_vectors.data(), num_queries, candidates.data(), candidates_per_query, k_top, num_threads);
    EXPECT_EQ(warm_results.size(), num_queries);

    // Benchmark run
    const int iterations = 30;
    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it) {
       auto results = reranker.rerank(query_vectors.data(), num_queries, candidates.data(), candidates_per_query, k_top, num_threads);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    double total_comps = static_cast<double>(num_queries * candidates_per_query * iterations);
    double comps_per_sec = (total_comps / (total_ms / 1000.0)) / 1e6;  // MComps/sec
    double gelements_per_sec = (comps_per_sec * dim) / 1000.0;         // GElements/sec
    double qps = (num_queries * iterations) / (total_ms / 1000.0);

    std::cout << "\n=== Rerank Benchmark (FP32_L2, Instruction: " << space.get_instruction() << ") ===\n"
              << "Queries: " << num_queries << ", Base Vectors: " << num_base << ", Dim: " << dim << ", Cands/Query: " << candidates_per_query
              << ", K-Top: " << k_top << ", Iterations: " << iterations << "\n"
              << std::fixed << std::setprecision(3) << "Average latency: " << avg_ms << " ms / batch\n"
              << "Total time: " << total_ms << " ms\n"
              << "Throughput: " << std::setprecision(2) << qps << " QPS\n"
              << "Distance Comparisons: " << comps_per_sec << " MComps/sec\n"
              << "Element Throughput: " << gelements_per_sec << " GElements/sec\n\n";
}

TEST(SearchRegression, Rerank_FP16_InnerProduct_Benchmark) {
    const size_t dim = 512;
    const size_t num_queries = 5000;
    const size_t num_base = 100000;
    const size_t candidates_per_query = 200;
    const size_t k_top = 10;
    const size_t num_threads = 0;  // auto-detect

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP16_InnerProduct);

    // Generate random base vectors and query vectors in FP16 (stored as uint16_t)
    std::vector<uint16_t> base_vectors(num_base * dim);
    std::vector<uint16_t> query_vectors(num_queries * dim);
    uint32_t rng = 4242;

    for (auto& v : base_vectors) v = static_cast<uint16_t>(benchmark_prng(rng) & 0xFFFF);
    for (auto& v : query_vectors) v = static_cast<uint16_t>(benchmark_prng(rng) & 0xFFFF);

    std::vector<uint32_t> candidates(num_queries * candidates_per_query);
    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t c = 0; c < candidates_per_query; ++c) {
            candidates[q * candidates_per_query + c] = benchmark_prng(rng) % num_base;
        }
    }

   deglib::search::Reranker<uint16_t> reranker(space, base_vectors.data(), num_base);

   // Warm-up run
   auto warm_results = reranker.rerank(query_vectors.data(), num_queries, candidates.data(), candidates_per_query, k_top, num_threads);
    EXPECT_EQ(warm_results.size(), num_queries);

    // Benchmark run
    const int iterations = 30;
    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iterations; ++it) {
       auto results = reranker.rerank(query_vectors.data(), num_queries, candidates.data(), candidates_per_query, k_top, num_threads);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;
    double total_comps = static_cast<double>(num_queries * candidates_per_query * iterations);
    double comps_per_sec = (total_comps / (total_ms / 1000.0)) / 1e6;  // MComps/sec
    double gelements_per_sec = (comps_per_sec * dim) / 1000.0;         // GElements/sec
    double qps = (num_queries * iterations) / (total_ms / 1000.0);

    std::cout << "\n=== Rerank Benchmark (FP16_InnerProduct, Instruction: " << space.get_instruction() << ") ===\n"
              << "Queries: " << num_queries << ", Base Vectors: " << num_base << ", Dim: " << dim << ", Cands/Query: " << candidates_per_query
              << ", K-Top: " << k_top << ", Iterations: " << iterations << "\n"
              << std::fixed << std::setprecision(3) << "Average latency: " << avg_ms << " ms / batch\n"
              << "Total time: " << total_ms << " ms\n"
              << "Throughput: " << std::setprecision(2) << qps << " QPS\n"
              << "Distance Comparisons: " << comps_per_sec << " MComps/sec\n"
              << "Element Throughput: " << gelements_per_sec << " GElements/sec\n\n";
}

TEST(SearchRegression, Reranker_Prefetch_Optimize_Benchmark) {
    // Reports rerank throughput for the prefetch config chosen by optimize() versus the Reranker's
    // built-in default (po=8, pl=0). The default lies inside the tuner's search grid (po in {0,4,8,12,16},
    // pl in {0,2,4}), so the tuned config is never worse in the tuner's own measurement; the real speedup
    // is workload and hardware dependent and only appears once the base vectors exceed cache. This
    // benchmark reports the measured ratio and guards against a regression (tuned must not be materially
    // slower). A default that silently disables prefetching shows up here as the default line collapsing.
    const size_t dim = 512;
    const size_t num_queries = 2000;
    const size_t num_base = 100000;
    const size_t candidates_per_query = 200;
    const uint32_t k_top = 10;
    const int iterations = 5;

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    std::vector<float> base_vectors(num_base * dim);
    std::vector<float> query_vectors(num_queries * dim);
    uint32_t rng = 9999;
    for (auto& v : base_vectors) v = benchmark_float(rng);
    for (auto& v : query_vectors) v = benchmark_float(rng);

    std::vector<uint32_t> candidates(num_queries * candidates_per_query);
    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t c = 0; c < candidates_per_query; ++c) {
            candidates[q * candidates_per_query + c] = benchmark_prng(rng) % num_base;
        }
    }

    deglib::search::Reranker<float> reranker(space, base_vectors.data(), num_base);

    std::vector<uint32_t> out_indices(k_top);
    std::vector<float> out_distances(k_top);
    auto sweep_ms = [&]() {
        for (size_t q = 0; q < 8; ++q) {
            reranker.rerank(
                query_vectors.data() + q * dim, dim, candidates.data() + q * candidates_per_query, candidates_per_query, k_top, out_indices.data(),
                out_distances.data(), true, true
            );
        }
        double best = std::numeric_limits<double>::max();
        for (int it = 0; it < iterations; ++it) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t q = 0; q < num_queries; ++q) {
                reranker.rerank(
                    query_vectors.data() + q * dim, dim, candidates.data() + q * candidates_per_query, candidates_per_query, k_top, out_indices.data(),
                    out_distances.data(), true, true
                );
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };

    // Default config (inside the tuner grid) vs the tuned config, same queries and candidates.
    const int32_t default_po = reranker.getPo();
    const int32_t default_pl = reranker.getPl();
    const double default_ms = sweep_ms();

    reranker.optimize(query_vectors.data(), /*n_queries=*/30, candidates_per_query, k_top);
    const int32_t tuned_po = reranker.getPo();
    const int32_t tuned_pl = reranker.getPl();
    const double tuned_ms = sweep_ms();

    const double speedup = default_ms / tuned_ms;
    std::cout << "\n=== Reranker Prefetch Benchmark (FP32_L2, Instruction: " << space.get_instruction() << ") ===\n"
              << "Base Vectors: " << num_base << ", Dim: " << dim << " (" << (static_cast<double>(num_base) * dim * 4 / 1e6) << " MB working set), "
              << "Queries: " << num_queries << ", Cands/Query: " << candidates_per_query << ", K-Top: " << k_top << ", Iterations: " << iterations << "\n"
              << "Default prefetch (po=" << default_po << ",pl=" << default_pl << "): " << std::fixed << std::setprecision(3) << default_ms << " ms/sweep ("
              << std::setprecision(0) << (num_queries / (default_ms / 1000.0)) << " QPS)\n"
              << "Tuned prefetch (po=" << tuned_po << ",pl=" << tuned_pl << "): " << std::setprecision(3) << tuned_ms << " ms/sweep (" << std::setprecision(0)
              << (num_queries / (tuned_ms / 1000.0)) << " QPS)\n"
              << "Speedup: " << std::setprecision(3) << speedup << "x\n\n";

    // Regression guard: optimize() must not make reranking materially slower than the default, and the
    // default must remain a prefetching config (po > 0) so out-of-cache reranking never silently loses throughput.
    EXPECT_GE(speedup, 0.85);
    EXPECT_GT(default_po, 0);
}

TEST(SearchRegression, SearcherTraversal_Optimize_Benchmark) {
   // Reports traversal throughput for the prefetch config chosen by optimize() versus the graph's
   // built-in default (po=8,pl=3,nl=3). The default lies inside the tuner's search grid, so the
   // tuned config is never worse in the tuner's own measurement; the real speedup is workload and
   // hardware dependent (it only appears once the working set exceeds cache). This benchmark reports
   // the measured ratio and guards against a regression (tuned must not be materially slower).
   const uint32_t dim = 128;
   const uint32_t num_base = 60000;
   const uint32_t num_queries = 512;
   const int iterations = 5;

   std::vector<float> data(num_base * dim);
   uint32_t rng = 9001;
   for (auto& v : data) v = benchmark_float(rng);

   deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);
   auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 16, deglib::distances::Metric::FP32_L2);
   auto searcher = deglib::search::make_searcher(graph.internal());

   searcher->optimize(/*n_clusters=*/64, /*n_iter=*/5, /*sample_size=*/0, /*k=*/10, /*seed=*/7, /*threads=*/1);
   const int32_t tuned_po = graph.getPo();
   const int32_t tuned_pl = graph.getPl();
   const int32_t tuned_nl = graph.getNl();

   std::vector<uint32_t> idx(10);
   std::vector<float> d(10);
   auto sweep_us = [&]() {
       double best = std::numeric_limits<double>::max();
       for (int it = 0; it < iterations; ++it) {
           const auto t0 = std::chrono::high_resolution_clock::now();
           for (uint32_t q = 0; q < num_queries; ++q) {
               searcher->search_f32(data.data() + static_cast<size_t>(q) * dim, 10, 0.1f, 1.0f, idx.data(), d.data());
           }
           const auto t1 = std::chrono::high_resolution_clock::now();
           best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count());
       }
       return best;
   };

   // Default config (inside the tuner grid) vs the tuned config, entry points held constant.
   graph.setPrefetch(8, 3, 3);
   const double default_us = sweep_us();
   graph.setPrefetch(tuned_po, tuned_pl, tuned_nl);
   const double tuned_us = sweep_us();

   const double speedup = default_us / tuned_us;
   std::cout << "\n=== Searcher Traversal Prefetch Benchmark (FP32_L2, Instruction: " << space.get_instruction() << ") ===\n"
             << "Vertices: " << num_base << ", Dim: " << dim << " (" << (static_cast<double>(num_base) * dim * 4 / 1e6) << " MB working set), Queries: " << num_queries
             << ", Iterations: " << iterations << "\n"
             << "Default prefetch (po=8,pl=3,nl=3): " << std::fixed << std::setprecision(1) << default_us << " us/sweep\n"
             << "Tuned prefetch (po=" << tuned_po << ",pl=" << tuned_pl << ",nl=" << tuned_nl << "): " << tuned_us << " us/sweep\n"
             << "Speedup: " << std::setprecision(3) << speedup << "x\n\n";

   // Regression guard: optimize() must not make traversal materially slower than the default.
   EXPECT_GE(speedup, 0.85);
}
