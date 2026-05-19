/**
 * @file deglib_evp_test.cpp
 * @brief Graph Benchmark: build a DEG graph and evaluate exploration recall@15.
 *
 * Uses only deglib headers (no benchmark.h / dataset.h / build.h).
 * Loads raw .fvecs and .ivecs, builds a SizeBoundedGraph, explores for Top-15,
 * and compares against allknn ground truth.
 *
 * Two modes:
 *   evp   - quantize with deglib::quantization, build graph with Metric::EvpBits
 *   raw   - use original float feature vectors directly with Metric::L2
 *
 * Usage:
 *   deglib_evp_test <data_path> [mode]
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
#include <cstring>
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

// ============================================================================
// Shared helpers
// ============================================================================

static void run_exploration_sweep(
    const deglib::graph::SizeBoundedGraph& graph,
    const int32_t* gt_data,
    size_t count,
    size_t gt_dims,
    uint32_t k_top,
    uint8_t threads,
    std::string_view label)
{
    std::printf("\n--- Exploration sweep (%s) ---\n", label);

    const uint32_t k_explore = k_top;

    // Fixed max_distance_count values
    const uint32_t max_distances[] = { k_explore, 50, 100, 200, 300, 400, 500, 750, 1000 };
    float best_recall = -1.0f;
    uint32_t best_max_dist = 0;
    double best_explore_ms = 0.0;
    bool stop_sweep = false;

    for (uint32_t md_idx = 0; md_idx < sizeof(max_distances) / sizeof(max_distances[0]) && !stop_sweep; md_idx++) {
        const uint32_t max_distance_count = max_distances[md_idx];

        // Measure per-configuration timing
        double t_explore_start = now_ms();

        // Explore all vertices
        std::vector<std::vector<uint32_t>> results(count);

        deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads,
            [&](size_t label, size_t) {
                size_t entry_idx = graph.getInternalIndex(static_cast<uint32_t>(label));
                auto result_queue = graph.explore(
                    static_cast<uint32_t>(entry_idx),
                    k_explore,
                    false,  // include_entry (false: exclude self)
                    max_distance_count);

                auto& result = results[label];
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
            for (uint32_t k = 1; k <= k_top && k < gt_dims; ++k) {
                uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1→0 indexed
                for (uint32_t j = 0; j < k_top; ++j) {
                    if (results[i][j] == gt_idx) {
                        total_hits++;
                        break;
                    }
                }
            }
        }

        float recall = static_cast<float>(total_hits) / (count * k_top);

        if (recall > best_recall) {
            best_recall = recall;
            best_max_dist = max_distance_count;
            best_explore_ms = explore_ms_single;
        }

        std::printf("  max_dist=%6u  recall=%.4f  time=%.2f ms\n", max_distance_count, recall, explore_ms_single);

        if (recall >= 0.8f) {
            std::printf("  Reached recall target 0.8, stopping sweep\n");
            stop_sweep = true;
        }
    }

    std::printf("\nBest recall@%d: %.4f (max_distance_count=%u, explore_time=%.2f ms)\n",
                k_top, best_recall, best_max_dist, best_explore_ms);
}

// ============================================================================
// Mode 1: EVP Bits path (quantize → EvpBits graph)
// ============================================================================

static int run_evp_mode(
    const std::filesystem::path& data_path,
    size_t count, size_t dims,
    const float* train_data,
    const int32_t* gt_data, size_t gt_dims,
    uint32_t threads,
    uint32_t non_zeros,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top)
{
    std::printf("=== EVP Bits Graph Benchmark ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f\n",
                non_zeros, k_top, k_graph, k_ext, eps_ext);
    std::printf("Threads: %zu\n\n", threads);

    // --------------------------------------------------------------------------
    // Quantize
    // --------------------------------------------------------------------------
    double t1 = now_ms();

    auto quantized = deglib::quantization::quantize_batch(
        train_data, count, static_cast<uint32_t>(dims), non_zeros, threads);

    double quantize_ms = now_ms() - t1;
    std::printf("Quantize time: %.2f ms\n", quantize_ms);
    std::printf("Quantized size: %.2f MB\n\n",
                static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

    // --------------------------------------------------------------------------
    // Build graph with Metric::EvpBits
    // --------------------------------------------------------------------------
    double t2 = now_ms();

    deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::EvpBits);
    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(count), k_graph, feature_space);

    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(
        graph, rnd,
        deglib::builder::OptimizationTarget::LowLID,
        k_ext, eps_ext,
        0, 0.0f,
        5,
        0, 0,
        true,
        false,
        false
    );
    builder.setThreadCount(static_cast<uint32_t>(threads));

    const size_t bytes_per_evp = dims / 4;
    for (size_t i = 0; i < count; ++i) {
        std::vector<std::byte> feature(quantized.data() + i * bytes_per_evp,
                                       quantized.data() + (i + 1) * bytes_per_evp);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feature));
    }

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
    // Exploration sweep
    // --------------------------------------------------------------------------
    run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, threads, "EVP Bits");

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    double total_ms = (now_ms() - 0.0); // caller handles t0

    std::printf("=============================================\n");
    std::printf("          FINAL SUMMARY (EVP Bits)\n");
    std::printf("=============================================\n");
    std::printf("Quantize:                %.2f ms\n", quantize_ms);
    std::printf("Graph Build:             %.2f ms\n", build_ms);
    std::printf("---------------------------------------------\n");
    std::printf("Vectors:                 %zu\n", count);
    std::printf("Dimensions:              %zu\n", dims);
    std::printf("NON_ZEROS:               %u\n", non_zeros);
    std::printf("=============================================\n\n");

    return 0;
}

// ============================================================================
// Mode 2: Raw float path (no quantization, Metric::L2)
// ============================================================================

static int run_raw_mode(
    const std::filesystem::path& data_path,
    size_t count, size_t dims,
    const float* train_data,
    const int32_t* gt_data, size_t gt_dims,
    uint32_t threads,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top)
{
    std::printf("=== FP32 InnerProduct Graph Benchmark ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f\n",
                k_top, k_graph, k_ext, eps_ext);
    std::printf("Threads: %zu\n\n", threads);

    // --------------------------------------------------------------------------
    // Build graph with Metric::InnerProduct (no quantization)
    // --------------------------------------------------------------------------
    double t2 = now_ms();

    deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::InnerProduct);
    deglib::graph::SizeBoundedGraph graph(static_cast<uint32_t>(count), k_graph, feature_space);

    std::mt19937 rnd(42);
    deglib::builder::EvenRegularGraphBuilder builder(
        graph, rnd,
        deglib::builder::OptimizationTarget::LowLID,
        k_ext, eps_ext,
        0, 0.0f,
        5,
        0, 0,
        true,
        false,
        false
    );
    builder.setThreadCount(static_cast<uint32_t>(threads));

    // Add all entries from raw float data
    for (size_t i = 0; i < count; ++i) {
        std::vector<std::byte> feature(dims * sizeof(float));
        std::memcpy(feature.data(), train_data + i * dims, dims * sizeof(float));
        builder.addEntry(static_cast<uint32_t>(i), std::move(feature));
    }

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
    // Exploration sweep
    // --------------------------------------------------------------------------
    run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, threads, "FP32 InnerProduct");

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    std::printf("=============================================\n");
    std::printf("          FINAL SUMMARY (FP32 InnerProduct)\n");
    std::printf("=============================================\n");
    std::printf("Graph Build:             %.2f ms\n", build_ms);
    std::printf("---------------------------------------------\n");
    std::printf("Vectors:                 %zu\n", count);
    std::printf("Dimensions:              %zu\n", dims);
    std::printf("=============================================\n\n");

    return 0;
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
    // Parse data path and mode
    // --------------------------------------------------------------------------
    std::filesystem::path data_path;
    std::string mode = "evp";  // default: EVP Bits mode

    if (argc >= 2) {
        data_path = argv[1];
    } else {
#ifdef DATA_PATH
        data_path = DATA_PATH;
#else
        std::fprintf(stderr, "Usage: %s <data_path> [evp|raw]\n", argv[0]);
        return 1;
#endif
    }

    if (argc >= 3) {
        mode = argv[2];
    }

    if (mode != "evp" && mode != "raw") {
        std::fprintf(stderr, "Unknown mode '%s'. Use 'evp' or 'raw'.\n", mode.c_str());
        return 1;
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

    const size_t threads = 6; // hardware_threads();

    // --------------------------------------------------------------------------
    // Load data
    // --------------------------------------------------------------------------
    double t0 = now_ms();

    size_t dims = 0, count = 0;
    auto train_bytes = deglib::fvecs_read(train_fvecs.string().c_str(), dims, count);
    const float* train_data = reinterpret_cast<const float*>(train_bytes.get());

    size_t gt_dims = 0, gt_count = 0;
    auto gt_bytes = deglib::ivecs_read(allknn_ivecs.string().c_str(), gt_dims, gt_count);
    const int32_t* gt_data = reinterpret_cast<const int32_t*>(gt_bytes.get());

    double load_ms = now_ms() - t0;
    std::printf("Loaded: %zu vectors, dim=%zu\n", count, dims);
    std::printf("Ground truth: %zu queries, top-%zu\n", gt_count, gt_dims);
    std::printf("Load time: %.2f ms\n\n", load_ms);

    if (count != gt_count) {
        std::fprintf(stderr,
            "Error: train.fvecs and allknn.ivecs must contain the same number of entries (%zu vs %zu)\n",
            count, gt_count);
        return 1;
    }

    // --------------------------------------------------------------------------
    // Dispatch to selected mode
    // --------------------------------------------------------------------------
    if (mode == "evp") {
        return run_evp_mode(data_path, count, dims, train_data, gt_data, gt_dims,
                            threads, NON_ZEROS, K_GRAPH, K_EXT, EPS_EXT, K_TOP);
    } else {
        return run_raw_mode(data_path, count, dims, train_data, gt_data, gt_dims,
                            threads, K_GRAPH, K_EXT, EPS_EXT, K_TOP);
    }
}
