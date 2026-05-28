#pragma once

/**
 * @file modi1.h
 * @brief Benchmark Mode 1: FP16 Build, FP16 Explore (fp16-build-fp16-explore)
 * 
 * Behavior:
 * 1. Uses exact FP16 training vectors directly.
 * 2. Builds a SizeBoundedGraph using the Metric::FP16InnerProduct metric.
 * 3. Explores the FP16 graph directly using FP16 Inner Product similarity.
 * 4. Acts as a baseline high-quality exact search topology.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <algorithm>
#include <thread>
#include <vector>
#include <numeric>
#include <filesystem>
#include <fstream>

#include <fmt/core.h>

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#include <immintrin.h>
#endif

#include "builder.h"
#include "concurrent.h"
#include "distances.h"
#include "graph/sizebounded_graph.h"
#include "graph/readonly_graph.h"
#include "graph/readonly_graph_external.h"
#include "quantization/evp_quantize.h"
#include "repository.h"

#include "../evp_common.h"
#include "../hdf5_reader.h"

namespace task1::mode1 {

static void run_exploration_sweep(
    const deglib::search::SearchGraph& graph,
    uint32_t k_top,
    uint32_t max_distance_count,
    uint8_t threads,
    bool compute_recall,
    const std::vector<std::vector<int32_t>>& gt_data,
    const std::string& output_path)
{
    size_t count = graph.size();
    std::printf("\n--- Exploration (FP16 Build FP16 Explore, max_dist=%u) ---\n", max_distance_count);
    const size_t batch_size = evp_common::calc_batch_size(count, threads);

    double t_explore_start = evp_common::now_ms();
    std::vector<std::vector<uint32_t>> results(count);

    deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, batch_size,
        [&](size_t label, size_t thread_id) {
            (void)thread_id;
            size_t entry_idx = graph.getInternalIndex(static_cast<uint32_t>(label));

            deglib::search::ResultSet result_queue = graph.explore(
                static_cast<uint32_t>(entry_idx),
                k_top,
                false,
                max_distance_count);

            auto& res = results[label];
            res.reserve(result_queue.size());
            while (!result_queue.empty()) {
                const auto cand_idx = result_queue.top().getInternalIndex();
                if (cand_idx != entry_idx) {
                    res.push_back(graph.getExternalLabel(cand_idx));
                }
                result_queue.pop();
            }
        });
    double explore_ms = evp_common::now_ms() - t_explore_start;
    std::printf("Exploration complete in %.2f ms\n", explore_ms);

    if (compute_recall) {
        float recall = evp_common::compute_recall(gt_data, results, k_top);
        std::printf("Recall@%u: %.4f  (max_dist=%u, time=%.2f ms)\n",
                    k_top, recall, max_distance_count, explore_ms);
    } else {
        evp_common::ivecs_write(output_path, results);
    }
}

static int run(
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t non_zeros,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top,
    uint32_t max_distance_count,
    bool compute_recall,
    const std::string& output_path)
{
    (void)non_zeros;
    const std::string h5path = data_path.string();
    std::printf("HDF5 mode (modi1): scanning '%s'\n", h5path.c_str());

    auto datasets = hdf5_reader::scan_datasets(h5path);
    auto& train_info = hdf5_reader::find_dataset(datasets, "train");

    double t_load = evp_common::now_ms();
    std::vector<std::vector<std::byte>> train_vectors = hdf5_reader::read_matrix_bytes(h5path, train_info);
    size_t dims = static_cast<size_t>(train_info.num_cols);
    size_t count = static_cast<size_t>(train_info.num_rows);

    std::vector<std::vector<int32_t>> gt_data;
    if (compute_recall) {
        gt_data = evp_common::load_ground_truth(h5path, datasets, k_top);
        if (count != gt_data.size()) {
            std::fprintf(stderr,
                "Error: train and allknn must contain the same number of entries (%zu vs %zu)\n",
                count, gt_data.size());
            return 1;
        }
    }
    double load_ms = evp_common::now_ms() - t_load;

    std::printf("Loaded train:  %zu vectors, dim=%zu\n", count, dims);
    if (compute_recall) {
        std::printf("Ground truth:  %zu queries, top-%u\n", gt_data.size(), k_top);
    }
    std::printf("Load time:     %.2f ms\n\n", load_ms);

    std::printf("=== FP16 Build FP16 Explore Graph Benchmark (Mode 1) ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.3f\n",
                k_top, k_graph, k_ext, eps_ext);
    std::printf("Threads: %u\n\n", threads);

    std::printf("FP16 data size: %.2f MB\n\n",
                static_cast<double>(count * dims * sizeof(uint16_t)) / (1024.0 * 1024.0));

    // --------------------------------------------------------------------------
    // Build graph with Metric::FP16InnerProduct
    // --------------------------------------------------------------------------
    double t2 = evp_common::now_ms();

    deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
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
    builder.setBatchSize(64, 128);

    // Add all entries from FP16 data directly using zero-copy std::move
    const size_t bytes_per_fp16 = dims * sizeof(uint16_t);
    for (size_t i = 0; i < count; ++i) {
        builder.addEntry(static_cast<uint32_t>(i), std::move(train_vectors[i]));
    }

    auto build_callback = [&](deglib::builder::BuilderStatus& status) {
        double elapsed = evp_common::now_ms() - t2;
        std::printf("  Build step %llu: added=%llu, improved=%llu, elapsed=%.2f ms\n",
                    (unsigned long long)status.step,
                    (unsigned long long)status.added,
                    (unsigned long long)status.improved,
                    elapsed);
    };

    builder.build(build_callback, false);

    double build_ms = evp_common::now_ms() - t2;
    std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
    std::printf("Build time: %.2f ms\n\n", build_ms);

    // --------------------------------------------------------------------------
    // Exploration sweep
    // --------------------------------------------------------------------------
    run_exploration_sweep(graph, k_top, max_distance_count, static_cast<uint8_t>(threads),
                          compute_recall, gt_data, output_path);

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    std::printf("=============================================\n");
    std::printf("          FINAL SUMMARY (FP16 Build FP16 Explore - Mode 1)\n");
    std::printf("=============================================\n");
    std::printf("Graph Build:             %.2f ms\n", build_ms);
    std::printf("---------------------------------------------\n");
    std::printf("Vectors:                 %zu\n", count);
    std::printf("Dimensions:              %zu\n", dims);
    std::printf("FP16 bytes/vector:       %zu\n", bytes_per_fp16);
    std::printf("=============================================\n\n");

    return 0;
}

} // namespace task1::mode1
