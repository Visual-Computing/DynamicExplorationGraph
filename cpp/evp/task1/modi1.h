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

struct ExplorationTimings {
    double explore_ms = 0.0;
    float recall = -1.0f;
};

static ExplorationTimings run_exploration_sweep(
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
    std::vector<std::vector<uint32_t>> results(count);

    std::printf("  Chunk-based dataset mode: unified explore via chunks with timing measurements\n");
    double t_start = evp_common::now_ms();

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    std::vector<double> chunk_search_times(num_chunks, 0.0);

    deglib::concurrent::parallel_for(static_cast<size_t>(0), num_chunks, threads, 1,
        [&](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, count);
            size_t num_items = end - start;

            double t_search_start = evp_common::now_ms();
            for (size_t i = 0; i < num_items; ++i) {
                size_t label = start + i;
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
            }
            chunk_search_times[chunk_id] = evp_common::now_ms() - t_search_start;
        });

    double total_ms = evp_common::now_ms() - t_start;
    double sum_search_ms = std::accumulate(chunk_search_times.begin(), chunk_search_times.end(), 0.0) / threads;

    std::printf("Exploration complete in %.2f ms (explore=%.2f ms)\n",
                total_ms, sum_search_ms);

    float recall = -1.0f;
    if (compute_recall) {
        recall = evp_common::compute_recall(gt_data, results, k_top);
        std::printf("Recall@%u: %.4f  (max_dist=%u, explore=%.2f ms)\n",
                    k_top, recall, max_distance_count, sum_search_ms);
    } else {
        evp_common::ivecs_write(output_path, results);
    }

    return { sum_search_ms, recall };
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

    double t_load_start = evp_common::now_ms();
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
    double total_load_ms = evp_common::now_ms() - t_load_start;

    std::printf("=== FP16 Build FP16 Explore Graph Benchmark (Mode 1 - Chunked Build) ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.3f\n",
                k_top, k_graph, k_ext, eps_ext);
    std::printf("Threads: %u\n\n", threads);

    std::printf("FP16 data size: %.2f MB\n\n",
                static_cast<double>(count * dims * sizeof(uint16_t)) / (1024.0 * 1024.0));

    // --------------------------------------------------------------------------
    // Build graph with Metric::FP16InnerProduct
    // --------------------------------------------------------------------------
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

    double total_build_ms = 0.0;
    const size_t load_chunk_size = 100000;
    double t_build_start = evp_common::now_ms();

    for (size_t start_row = 0; start_row < count; start_row += load_chunk_size) {
        size_t current_chunk_size = std::min(load_chunk_size, count - start_row);

        // Load chunk
        double t_chunk_load = evp_common::now_ms();
        std::vector<std::vector<std::byte>> chunk_vectors = hdf5_reader::read_matrix_bytes(h5path, train_info, start_row, current_chunk_size);
        double chunk_load_ms = evp_common::now_ms() - t_chunk_load;
        total_load_ms += chunk_load_ms;

        // Add chunk entries
        for (size_t i = 0; i < current_chunk_size; ++i) {
            builder.addEntry(static_cast<uint32_t>(start_row + i), std::move(chunk_vectors[i]));
        }
        chunk_vectors.clear();
        chunk_vectors.shrink_to_fit();

        // Build chunk
        double t_chunk_build = evp_common::now_ms();
        auto dummy_callback = [](deglib::builder::BuilderStatus&) {};
        builder.build(dummy_callback, false);
        double chunk_build_ms = evp_common::now_ms() - t_chunk_build;
        total_build_ms += chunk_build_ms;

        double elapsed_s = (evp_common::now_ms() - t_build_start) / 1000.0;
        std::printf("  Chunk [%6zuk - %6zuk): Load = %.2fs, Build = %.2fs | Elapsed = %.2fs\n", 
                    start_row / 1000, (start_row + current_chunk_size) / 1000, 
                    chunk_load_ms / 1000.0, chunk_build_ms / 1000.0, elapsed_s);
    }

    std::printf("\nLoaded train:  %zu vectors, dim=%zu\n", count, dims);
    if (compute_recall) {
        std::printf("Ground truth:  %zu queries, top-%u\n", gt_data.size(), k_top);
    }
    std::printf("Total Load time:     %.2fs\n", total_load_ms / 1000.0);
    std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
    std::printf("Total Build time:    %.2fs\n\n", total_build_ms / 1000.0);

    // --------------------------------------------------------------------------
    // Exploration sweep
    // --------------------------------------------------------------------------
    auto timings = run_exploration_sweep(graph, k_top, max_distance_count, static_cast<uint8_t>(threads),
                                         compute_recall, gt_data, output_path);

    const size_t bytes_per_fp16 = dims * sizeof(uint16_t);
    double total_time_ms = total_load_ms + total_build_ms + timings.explore_ms;

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    std::printf("========================================================================\n");
    std::printf("  FINAL SUMMARY (FP16 Build, FP16 Explore - Mode 1)                     \n");
    std::printf("========================================================================\n");
    std::printf("Load Time:               %.2f ms\n", total_load_ms);
    std::printf("Graph Build Time:        %.2f ms\n", total_build_ms);
    std::printf("Explore Time:            %.2f ms\n", timings.explore_ms);
    std::printf("Exploration Wall Time:   %.2f ms\n", timings.explore_ms);
    std::printf("Total Elapsed Time:      %.2f ms\n", total_time_ms);
    if (compute_recall) {
        std::printf("Recall@%u:                %.4f\n", k_top, timings.recall);
    }
    std::printf("------------------------------------------------------------------------\n");
    std::printf("Hyperparameters:\n");
    std::printf("  K_TOP:                 %u\n", k_top);
    std::printf("  K_GRAPH:               %u\n", (uint32_t)k_graph);
    std::printf("  K_EXT:                 %u\n", (uint32_t)k_ext);
    std::printf("  EPS_EXT:               %.3f\n", eps_ext);
    std::printf("  max_dist:              %u\n", max_distance_count);
    std::printf("  threads:               %u\n", threads);
    std::printf("------------------------------------------------------------------------\n");
    std::printf("Dataset Info:\n");
    std::printf("  Vectors:               %zu\n", count);
    std::printf("  Dimensions:            %zu\n", dims);
    std::printf("  FP16 bytes/vector:     %zu\n", bytes_per_fp16);
    std::printf("========================================================================\n\n");

    return 0;
}

} // namespace task1::mode1
