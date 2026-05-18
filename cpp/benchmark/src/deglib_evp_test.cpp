/**
 * @file deglib_evp_test.cpp
 * @brief EVP Bits Graph Benchmark: build a DEG graph from quantized EvpBits and evaluate exploration recall@15.
 *
 * Uses only deglib headers (no benchmark.h / dataset.h / build.h).
 * Loads raw .fvecs and .ivecs, quantizes with deglib::quantization, builds a SizeBoundedGraph
 * with Metric::EvpBits, explores the graph for Top-15, and compares against allknn ground truth.
 *
 * Usage:
 *   deglib_evp_test <data_path>
 *
 * Expected files in data_path:
 *   train.fvecs      - float feature vectors
 *   allknn.ivecs     - ground truth (1-indexed, dims=32)
 */

#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <thread>
#include <vector>

#include <fmt/core.h>

#include "builder.h"
#include "concurrent.h"
#include "distances.h"
#include "graph/sizebounded_graph.h"
#include "quantization/evp_quantize.h"
#include "repository.h"

// ============================================================================
// Helpers
// ============================================================================

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
           .count();
}

static size_t hardware_threads() {
    return std::max(std::thread::hardware_concurrency(), static_cast<unsigned int>(1));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
#if defined(USE_AVX)
    fmt::print("Using AVX2...\n");
#elif defined(USE_SSE)
    fmt::print("Using SSE...\n");
#else
    fmt::print("Using arch...\n");
#endif

    // --------------------------------------------------------------------------
    // Parameters
    // --------------------------------------------------------------------------
    constexpr uint32_t NON_ZEROS = 512;
    constexpr uint32_t K_TOP     = 15;
    constexpr uint8_t  K_GRAPH   = 32;
    constexpr uint8_t  K_EXT     = 32;
    constexpr float    EPS_EXT   = 0.001f;

    // --------------------------------------------------------------------------
    // Parse data path
    // --------------------------------------------------------------------------
    std::filesystem::path data_path;
    if (argc >= 2) {
        data_path = argv[1];
    } else {
        // Try DATA_PATH compile-time define
#ifdef DATA_PATH
        data_path = DATA_PATH;
#else
        std::fprintf(stderr, "Usage: %s <data_path>\n", argv[0]);
        return 1;
#endif
    }

    const auto train_fvecs = data_path / "SISAP" / "train.fvecs";
    const auto allknn_ivecs = data_path / "SISAP" / "allknn.ivecs";

    if (!std::filesystem::exists(train_fvecs)) {
        std::fprintf(stderr, "Error: %ls not found\n", train_fvecs.wstring().c_str());
        return 1;
    }
    if (!std::filesystem::exists(allknn_ivecs)) {
        std::fprintf(stderr, "Error: %ls not found\n", allknn_ivecs.wstring().c_str());
        return 1;
    }

    const size_t threads = 1; //hardware_threads();

    std::printf("=== EVP Bits Graph Benchmark ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f\n",
                NON_ZEROS, K_TOP, K_GRAPH, K_EXT, EPS_EXT);
    std::printf("Threads: %zu\n\n", threads);

    // --------------------------------------------------------------------------
    // Step 1: Load train.fvecs
    // --------------------------------------------------------------------------
    double t0 = now_ms();

    size_t dims = 0, count = 0;
    auto train_bytes = deglib::fvecs_read(train_fvecs.string().c_str(), dims, count);
    const float* train_data = reinterpret_cast<const float*>(train_bytes.get());

    // --------------------------------------------------------------------------
    // Step 2: Load allknn.ivecs (ground truth, 1-indexed)
    // --------------------------------------------------------------------------
    size_t gt_dims = 0, gt_count = 0;
    auto gt_bytes = deglib::ivecs_read(allknn_ivecs.string().c_str(), gt_dims, gt_count);
    const int32_t* gt_data = reinterpret_cast<const int32_t*>(gt_bytes.get());

    double load_ms = now_ms() - t0;
    std::printf("Loaded: %zu vectors, dim=%zu\n", count, dims);
    std::printf("Ground truth: %zu queries, top-%zu\n", gt_count, gt_dims);
    std::printf("Load time: %.2f ms\n\n", load_ms);

    // Check that dataset and ground truth sizes match
    if (count != gt_count) {
        std::fprintf(stderr,
            "Error: train.fvecs and allknn.ivecs must contain the same number of entries (%zu vs %zu)\n",
            count, gt_count);
        return 1;
    }

    // --------------------------------------------------------------------------
    // Step 3: Quantize with deglib::quantization
    // --------------------------------------------------------------------------
    double t1 = now_ms();

    auto quantized = deglib::quantization::quantize_batch(
        train_data, count, static_cast<uint32_t>(dims), NON_ZEROS, threads);

    double quantize_ms = now_ms() - t1;
    std::printf("Quantize time: %.2f ms\n", quantize_ms);
    std::printf("Quantized size: %.2f MB\n\n",
                static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

    // --------------------------------------------------------------------------
    // Step 4: Build DEG graph with Metric::EvpBits
    // --------------------------------------------------------------------------
    double t2 = now_ms();

    // Create feature space for EvpBits
    deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::EvpBits);

    // Create graph
    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(count), K_GRAPH, feature_space);

    // Create builder
    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(
        graph, rnd,
        deglib::builder::OptimizationTarget::LowLID,
        K_EXT, EPS_EXT,
        0, 0.0f,    // improve_k, improve_eps (0 = no improvement during build)
        5,          // max_path_length
        0, 0,       // swap_tries, additional_swap_tries
        true,       // use_rng
        false,      // use_path_verification
        false       // use_simple_edge_swaps
    );
    builder.setThreadCount(static_cast<uint32_t>(threads));

    // Add all entries from quantized data
    const size_t bytes_per_evp = dims / 4; // 2 * dim/8 = dim/4
    for (size_t i = 0; i < count; ++i) {
        std::vector<std::byte> feature(quantized.data() + i * bytes_per_evp,
                                       quantized.data() + (i + 1) * bytes_per_evp);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feature));
    }

    // Build
    auto build_callback = [](deglib::builder::BuilderStatus& status) {
        std::printf("  Build step %llu: added=%llu, improved=%llu\n",
                    (unsigned long long)status.step,
                    (unsigned long long)status.added,
                    (unsigned long long)status.improved);
    };

    builder.build(build_callback, false);

    double build_ms = now_ms() - t2;
    std::printf("Graph built: %zu vertices, %u edges/vertex\n", graph.size(), graph.getEdgesPerVertex());
    std::printf("Build time: %.2f ms\n\n", build_ms);

    // --------------------------------------------------------------------------
    // Step 5: Exploration for Recall@K_TOP
    // --------------------------------------------------------------------------
    const uint32_t k_explore = K_TOP;

    // Sweep max_distance_count like in deglib_test.cpp
    uint32_t k_factor = 100;
    float best_recall = -1.0f;
    uint32_t best_max_dist = 0;
    double best_explore_ms = 0.0;
    std::vector<std::vector<uint32_t>> best_results;

    for (uint32_t f = 0; f <= 3; f++, k_factor *= 10) {
        for (uint32_t c = (f == 0) ? 1 : 2; c <= 10; c++) {
            const uint32_t max_distance_count = (f == 0) ? (k_explore + k_factor * (c - 1)) : (k_factor * c);

            // Measure per-configuration timing
            double t_explore_start = now_ms();

            // Explore all vertices
            std::vector<std::vector<uint32_t>> results(count);

            deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads,
                [&](size_t entry_idx, size_t) {
                    auto result_queue = graph.explore(
                        static_cast<uint32_t>(entry_idx),
                        k_explore,
                        true,   // include_entry
                        max_distance_count);

                    auto& result = results[entry_idx];
                    result.resize(k_explore);
                    for (uint32_t j = 0; j < k_explore && !result_queue.empty(); ++j) {
                        result[j] = graph.getExternalLabel(result_queue.top().getInternalIndex());
                        result_queue.pop();
                    }
                });

            double explore_ms_single = now_ms() - t_explore_start;

            // Calculate recall
            int total_hits = 0;
            for (size_t i = 0; i < count; ++i) {                
                const int32_t* gt_row = &gt_data[i * gt_dims];
                for (uint32_t k = 0; k < K_TOP && k < gt_dims; ++k) {
                    uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1→0 indexed
                    for (uint32_t j = 0; j < K_TOP; ++j) {
                        if (results[i][j] == gt_idx) {
                            total_hits++;
                            break;
                        }
                    }
                }
            }

            float recall = static_cast<float>(total_hits) / (count * K_TOP);

            if (recall > best_recall) {
                best_recall = recall;
                best_max_dist = max_distance_count;
                best_explore_ms = explore_ms_single;
                best_results = std::move(results);
            }

            std::printf("  max_dist=%6u  recall=%.4f  time=%.2f ms\n", max_distance_count, recall, explore_ms_single);

            if (recall >= 0.997f) {
                std::printf("  Reached recall target 0.997, stopping sweep\n");
                goto recall_done;
            }
        }
    }

recall_done:;

    std::printf("\nBest recall@%d: %.4f (max_distance_count=%u, explore_time=%.2f ms)\n",
                K_TOP, best_recall, best_max_dist, best_explore_ms);

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    double total_ms = (now_ms() - t0);

    std::printf("=============================================\n");
    std::printf("          FINAL SUMMARY\n");
    std::printf("=============================================\n");
    std::printf("Load:                    %.2f ms\n", load_ms);
    std::printf("Quantize:                %.2f ms\n", quantize_ms);
    std::printf("Graph Build:             %.2f ms\n", build_ms);
    std::printf("Exploration (best):      %.2f ms\n", best_explore_ms);
    std::printf("---------------------------------------------\n");
    std::printf("TOTAL:                   %.2f ms\n", total_ms);
    std::printf("Vectors:                 %zu\n", count);
    std::printf("Dimensions:              %zu\n", dims);
    std::printf("NON_ZEROS:               %u\n", NON_ZEROS);
    std::printf("Recall@%d:               %.4f\n", K_TOP, best_recall);
    std::printf("Max distance count:      %u\n", best_max_dist);
    std::printf("=============================================\n");

    return 0;
}
