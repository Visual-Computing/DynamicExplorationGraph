#pragma once

/**
 * @file modi6.h
 * @brief Benchmark Mode 6: EVP Build, FP16 Asymmetric Search (evp-build-fp16-asymmetric-search)
 * 
 * Behavior:
 * 1. Quantizes training features to EVP bits representation.
 * 2. Builds a SizeBoundedGraph using the EvpBits metric.
 * 3. Converts to a ReadOnlyGraph using the Metric::FP16EvpAsymmetric metric.
 * 4. Performs asymmetric search where query is FP16 representation and database is EVP bits.
 * 5. No candidate reranking is performed.
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

#include <fmt/core.h>

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#include <immintrin.h>
#endif

#include "builder.h"
#include "concurrent.h"
#include "distances.h"
#include "graph/sizebounded_graph.h"
#include "graph/readonly_graph.h"
#include "quantization/evp_quantize.h"
#include "repository.h"

#include "../evp_common.h"
#include "../hdf5_reader.h"

namespace task1::mode6 {

struct ExplorationTimings {
    double explore_ms = 0.0;
    float recall = -1.0f;
};

static ExplorationTimings run_exploration(
    const deglib::search::SearchGraph& graph,
    uint32_t k_top,
    uint32_t max_distance_count,
    uint8_t threads,
    bool compute_recall,
    const std::vector<std::vector<int32_t>>& gt_data,
    const std::string& output_path,
    const std::vector<std::vector<std::byte>>& query_data_vecs)
{
    size_t count = graph.size();
    std::printf("\n--- Exploration (EVP Build FP16 Asymmetric Search, max_dist=%u) ---\n", max_distance_count);
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

                std::vector<uint32_t> entry_indices = { static_cast<uint32_t>(entry_idx) };
                const std::byte* query = query_data_vecs[label].data();
                deglib::search::ResultSet result_queue = graph.search(
                    entry_indices,
                    query,
                    1000.0f,
                    k_top + 1,
                    nullptr,
                    max_distance_count + 1);

                auto& res = results[label];
                res.reserve(result_queue.size());
                while (!result_queue.empty()) {
                    const auto cand_idx = result_queue.top().getInternalIndex();
                    if (cand_idx != entry_idx) {
                        res.push_back(graph.getExternalLabel(cand_idx));
                    }
                    result_queue.pop();
                }
                std::reverse(res.begin(), res.end());
                if (res.size() > k_top) {
                    res.resize(k_top);
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
    const std::string& output_path,
    const std::string& graph_path,
    uint32_t prune_worst = 0)
{
    const std::string h5path = data_path.string();
    std::printf("HDF5 mode (modi6): scanning '%s'\n", h5path.c_str());

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

    std::printf("=== EVP Bits Graph Benchmark (Mode 6: EvpBuildFP16AsymmetricSearch) ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.3f, MODE=build-fp16-asymmetric-search\n",
                non_zeros, k_top, k_graph, k_ext, eps_ext);
    std::printf("Threads: %u\n\n", threads);

    // --------------------------------------------------------------------------
    // Build/Load graph with Metric::EvpBits
    // --------------------------------------------------------------------------
    deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::EvpBits);
    std::unique_ptr<deglib::graph::SizeBoundedGraph> graph_ptr;
    bool loaded = false;
    double quantize_ms = 0.0;
    double build_ms = 0.0;

    if (!graph_path.empty() && std::filesystem::exists(graph_path)) {
        double t_load_graph_start = evp_common::now_ms();
        std::printf("Loading existing graph from %s...\n", graph_path.c_str());
        auto g = deglib::graph::load_sizebounded_graph(graph_path.c_str());
        const auto& fs = g.getFeatureSpace();
        if (fs.metric() == deglib::Metric::EvpBits && fs.dim() == dims && g.size() == count) {
            graph_ptr = std::make_unique<deglib::graph::SizeBoundedGraph>(std::move(g));
            loaded = true;
            double load_graph_ms = evp_common::now_ms() - t_load_graph_start;
            std::printf("Graph loaded successfully in %.2f ms\n", load_graph_ms);
        } else {
            std::fprintf(stderr, "Warning: Saved graph properties do not match dataset: metric=%d vs %d, dim=%u vs %zu, size=%u vs %zu. Rebuilding.\n",
                         (int)fs.metric(), (int)deglib::Metric::EvpBits, (unsigned)fs.dim(), dims, g.size(), count);
        }
    }

    if (!loaded) {
        // --------------------------------------------------------------------------
        // Quantize
        // --------------------------------------------------------------------------
        double t1 = evp_common::now_ms();

        auto quantized = deglib::quantization::quantize_batch(
            train_vectors, static_cast<uint32_t>(dims), non_zeros, threads);

        quantize_ms = evp_common::now_ms() - t1;
        std::printf("Quantize time: %.2f ms\n", quantize_ms);
        std::printf("Quantized size: %.2f MB\n\n",
                    static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

        double t2 = evp_common::now_ms();
        graph_ptr = std::make_unique<deglib::graph::SizeBoundedGraph>(static_cast<uint32_t>(count), k_graph, feature_space);
        deglib::graph::SizeBoundedGraph& graph = *graph_ptr;

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

        const size_t bytes_per_evp = dims / 4;
        for (size_t i = 0; i < count; ++i) {
            std::vector<std::byte> feature(quantized.data() + i * bytes_per_evp, quantized.data() + (i + 1) * bytes_per_evp);
            builder.addEntry(static_cast<uint32_t>(i), std::move(feature));
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

        build_ms = evp_common::now_ms() - t2;
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        if (!graph_path.empty()) {
            std::printf("Saving built graph to %s...\n", graph_path.c_str());
            graph.saveGraph(graph_path.c_str());
        }
    }

    deglib::graph::SizeBoundedGraph& graph = *graph_ptr;

    evp_common::prune_worst_neighbors(graph, prune_worst, threads);

    // --------------------------------------------------------------------------
    // Exploration
    // --------------------------------------------------------------------------
    double conversion_ms = 0.0;
    double t_copy_start = evp_common::now_ms();
    deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
    deglib::graph::ReadOnlyGraph readonly_asym_graph(static_cast<uint32_t>(count), graph.getEdgesPerVertex(), asymmetric_space, graph);
    conversion_ms = evp_common::now_ms() - t_copy_start;
    std::printf("ReadOnlyGraph construction (copy) time: %.2f ms\n", conversion_ms);

    auto timings = run_exploration(readonly_asym_graph, k_top, max_distance_count, static_cast<uint8_t>(threads),
                                         compute_recall, gt_data, output_path, train_vectors);

    double total_time_ms = load_ms + quantize_ms + build_ms + conversion_ms + timings.explore_ms;

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    std::printf("========================================================================\n");
    std::printf("  FINAL SUMMARY (EVP Build, FP16 Asymmetric Search - Mode 6)            \n");
    std::printf("========================================================================\n");
    std::printf("Load Time:               %.2f ms\n", load_ms);
    std::printf("Quantize Time:           %.2f ms\n", quantize_ms);
    std::printf("Graph Build Time:        %.2f ms\n", build_ms);
    std::printf("Graph Conversion Time:   %.2f ms\n", conversion_ms);
    std::printf("Explore Time:            %.2f ms\n", timings.explore_ms);
    std::printf("Exploration Wall Time:   %.2f ms\n", timings.explore_ms);
    std::printf("Total Elapsed Time:      %.2f ms\n", total_time_ms);
    if (compute_recall) {
        std::printf("Recall@%u:                %.4f\n", k_top, timings.recall);
    }
    std::printf("------------------------------------------------------------------------\n");
    std::printf("Hyperparameters:\n");
    std::printf("  NON_ZEROS:             %u\n", non_zeros);
    std::printf("  K_TOP:                 %u\n", k_top);
    std::printf("  K_GRAPH:               %u\n", (uint32_t)k_graph);
    std::printf("  K_EXT:                 %u\n", (uint32_t)k_ext);
    std::printf("  EPS_EXT:               %.3f\n", eps_ext);
    std::printf("  max_dist:              %u\n", max_distance_count);
    std::printf("  threads:               %u\n", threads);
    std::printf("  prune_worst:           %u\n", prune_worst);
    std::printf("------------------------------------------------------------------------\n");
    std::printf("Dataset Info:\n");
    std::printf("  Vectors:               %zu\n", count);
    std::printf("  Dimensions:            %zu\n", dims);
    std::printf("========================================================================\n\n");

    return 0;
}

} // namespace task1::mode6
