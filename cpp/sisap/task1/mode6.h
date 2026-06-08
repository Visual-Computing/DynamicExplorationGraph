#pragma once

/**
 * @file mode6.h
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

#include "../sisap_common.h"
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
    std::vector<std::vector<uint32_t>> results(count);

    double t_start = sisap_common::now_ms();

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    std::vector<double> chunk_search_times(num_chunks, 0.0);

    deglib::concurrent::parallel_for(static_cast<size_t>(0), num_chunks, threads, 1,
        [&](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, count);
            size_t num_items = end - start;

            double t_search_start = sisap_common::now_ms();
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
            chunk_search_times[chunk_id] = sisap_common::now_ms() - t_search_start;
        });

    double total_ms = sisap_common::now_ms() - t_start;
    double sum_search_ms = std::accumulate(chunk_search_times.begin(), chunk_search_times.end(), 0.0) / threads;

    float recall = -1.0f;
    if (compute_recall) {
        recall = sisap_common::compute_recall(gt_data, results, k_top);
    } else {
        sisap_common::ivecs_write(output_path, results);
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
    const std::vector<uint32_t>& max_dist_list,
    bool compute_recall,
    float goal_recall,
    const std::string& output_path,
    const std::string& graph_path,
    uint32_t prune_worst = 0)
{
    const std::string h5path = data_path.string();
    auto datasets = hdf5_reader::scan_datasets(h5path);
    auto& train_info = hdf5_reader::find_dataset(datasets, "train");

    double t_load = sisap_common::now_ms();
    std::vector<std::vector<std::byte>> train_vectors = hdf5_reader::read_matrix_bytes(h5path, train_info);
    size_t dims = static_cast<size_t>(train_info.num_cols);
    size_t count = static_cast<size_t>(train_info.num_rows);

    std::vector<std::vector<int32_t>> gt_data;
    if (compute_recall) {
        gt_data = sisap_common::load_ground_truth(h5path, datasets, k_top);
        if (count != gt_data.size()) {
            std::fprintf(stderr,
                "Error: train and allknn must contain the same number of entries (%zu vs %zu)\n",
                count, gt_data.size());
            return 1;
        }
    }
    double load_ms = sisap_common::now_ms() - t_load;

    std::printf("=== EVP Build, Asymmetric FP16 & EVP Explore - Mode 6 ===\n");

    // --------------------------------------------------------------------------
    // Build/Load graph with Metric::EvpBits
    // --------------------------------------------------------------------------
    deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::EvpBits);
    std::unique_ptr<deglib::graph::SizeBoundedGraph> graph_ptr;
    bool loaded = false;
    double quantize_ms = 0.0;
    double build_ms = 0.0;

    if (!graph_path.empty() && std::filesystem::exists(graph_path)) {
        double t_load_graph_start = sisap_common::now_ms();
        std::printf("Loading existing graph from %s...\n", graph_path.c_str());
        auto g = deglib::graph::load_sizebounded_graph(graph_path.c_str());
        const auto& fs = g.getFeatureSpace();
        if (fs.metric() == deglib::Metric::EvpBits && fs.dim() == dims && g.size() == count) {
            graph_ptr = std::make_unique<deglib::graph::SizeBoundedGraph>(std::move(g));
            loaded = true;
            double load_graph_ms = sisap_common::now_ms() - t_load_graph_start;
            std::printf("Graph loaded successfully in %.2f ms\n", load_graph_ms);
        } else {
            std::fprintf(stderr, "Warning: Saved graph properties do not match dataset: metric=%d vs %d, dim=%u vs %zu, size=%u vs %zu. Rebuilding.\n",
                         (int)fs.metric(), (int)deglib::Metric::EvpBits, (unsigned)fs.dim(), dims, g.size(), count);
        }
    }

    if (!loaded) {
        std::printf("Building graph: k_graph=%u, k_ext=%u, eps_ext=%.3f, non_zeros=%u, threads=%u\n", k_graph, k_ext, eps_ext, non_zeros, threads);

        // Quantize all upfront once (highly optimized, parallel, zero raw copy)
        double t1 = sisap_common::now_ms();
        auto quantized = deglib::quantization::quantize_batch(
            train_vectors, static_cast<uint32_t>(dims), non_zeros, threads);
        quantize_ms = sisap_common::now_ms() - t1;

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

        const size_t load_chunk_size = 200000;
        const size_t bytes_per_evp = dims / 4;
        double t_build_start = sisap_common::now_ms();

        for (size_t start_row = 0; start_row < count; start_row += load_chunk_size) {
            size_t current_chunk_size = std::min(load_chunk_size, count - start_row);

            // Add chunk entries directly from quantized buffer
            for (size_t i = 0; i < current_chunk_size; ++i) {
                size_t idx = start_row + i;
                std::vector<std::byte> feature(quantized.data() + idx * bytes_per_evp, quantized.data() + (idx + 1) * bytes_per_evp);
                builder.addEntry(static_cast<uint32_t>(idx), std::move(feature));
            }

            // Build chunk
            double t_chunk_build = sisap_common::now_ms();
            auto dummy_callback = [](deglib::builder::BuilderStatus&) {};
            builder.build(dummy_callback, false);
            double chunk_build_ms = sisap_common::now_ms() - t_chunk_build;
            build_ms += chunk_build_ms;

            double elapsed_s = (sisap_common::now_ms() - t_build_start) / 1000.0;
            std::printf("  Chunk [%6zuk - %6zuk): Build = %.2fs | Elapsed = %.2fs\n", 
                        start_row / 1000, (start_row + current_chunk_size) / 1000, 
                        chunk_build_ms / 1000.0, elapsed_s);
        }

        if (!graph_path.empty()) {
            std::printf("Saving built graph to %s...\n", graph_path.c_str());
            graph.saveGraph(graph_path.c_str());
        }
    }

    deglib::graph::SizeBoundedGraph& graph = *graph_ptr;

    double prune_ms = sisap_common::prune_worst_neighbors(graph, prune_worst, threads);

    // --------------------------------------------------------------------------
    // Exploration
    // --------------------------------------------------------------------------
    double conversion_ms = 0.0;
    double t_copy_start = sisap_common::now_ms();
    deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
    deglib::graph::ReadOnlyGraph readonly_asym_graph(static_cast<uint32_t>(count), graph.getEdgesPerVertex(), asymmetric_space, graph);
    conversion_ms = sisap_common::now_ms() - t_copy_start;

    std::printf("Starting exploration: k_top=%u, prune_worst=%u, threads=%u\n", k_top, prune_worst, threads);
    uint32_t best_max_dist = 0;
    float best_recall = -1.0f;
    ExplorationTimings best_timings;

    for (uint32_t max_dist_val : max_dist_list) {
        auto timings = run_exploration(readonly_asym_graph, k_top, max_dist_val, static_cast<uint8_t>(threads),
                                             compute_recall, gt_data, output_path, train_vectors);

        std::printf("  max_dist=%u has recall %.2f %% and time %.1f s\n",
                    max_dist_val, timings.recall * 100.0f, timings.explore_ms / 1000.0);

        bool is_better = false;
        if (timings.recall >= goal_recall) {
            is_better = (best_recall < goal_recall) || (timings.explore_ms < best_timings.explore_ms);
        } else {
            is_better = (best_recall < goal_recall) && (timings.recall > best_recall);
        }

        if (is_better) {
            best_recall = timings.recall;
            best_max_dist = max_dist_val;
            best_timings = timings;
        }
    }

    std::printf("\n");
    double total_time_ms = load_ms + quantize_ms + build_ms + conversion_ms + best_timings.explore_ms;

    sisap_common::print_summary(
        "EVP Build, Asymmetric FP16 & EVP Explore", 6,
        load_ms, quantize_ms, build_ms, conversion_ms, prune_ms,
        best_timings.explore_ms, 0.0, total_time_ms,
        compute_recall, k_top, best_timings.recall,
        threads, best_max_dist, prune_worst,
        k_graph, k_ext, eps_ext, non_zeros, count, dims, 0
    );

    return 0;
}

} // namespace task1::mode6
