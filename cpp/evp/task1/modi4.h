#pragma once

/**
 * @file modi4.h
 * @brief Benchmark Mode 4: EVP Build, EVP Explore, FP16 Rerank (evp-build-evp-explore-fp16-rerank)
 * 
 * Behavior:
 * 1. Quantizes training features to EVP bits representation.
 * 2. Builds a SizeBoundedGraph using the EvpBits metric.
 * 3. Explores the EVP graph topology using EVP Bits similarity.
 * 4. Reranks the explore candidates using exact FP16 Inner Product distances.
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

namespace task1::mode4 {

/**
 * @brief Performs search/exploration sweep over different graph search limits, calculates recall or exports Top-K results.
 */
struct ExplorationTimings {
    double search_ms = 0.0;
    double rerank_ms = 0.0;
    double total_ms = 0.0;
    float recall = -1.0f;
};

static ExplorationTimings run_exploration_sweep(
    const deglib::search::SearchGraph& graph,
    uint32_t evpK,
    uint32_t max_distance_count,
    uint8_t threads,
    uint32_t k_top,
    size_t dims,
    const std::vector<std::vector<std::byte>>& train_vectors,
    bool compute_recall,
    const std::vector<std::vector<int32_t>>& gt_data,
    const std::string& output_path)
{
    size_t count = graph.size();
    const uint32_t k_search = evpK > 0 ? evpK : std::max(k_top, max_distance_count);
    std::printf("\n--- Exploration (EVP Build EVP Explore FP16 Rerank, max_dist=%u) ---\n", max_distance_count);
    std::printf("  Dataset size: %zu vectors\n", count);

    deglib::FloatSpace fp16_rerank_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
    std::vector<std::vector<uint32_t>> results(count);

    std::printf("  Chunk-based dataset mode: unified explore+rerank via chunks with separate timing measurements\n");
    double t_start = evp_common::now_ms();

    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;

    std::vector<double> chunk_search_times(num_chunks, 0.0);
    std::vector<double> chunk_rerank_times(num_chunks, 0.0);

    deglib::concurrent::parallel_for(static_cast<size_t>(0), num_chunks, threads, 1,
        [&](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, count);
            size_t num_items = end - start;

            std::vector<std::vector<uint32_t>> chunk_cands(num_items);

            // --- Explore Phase for Chunk ---
            double t_search_start = evp_common::now_ms();
            for (size_t i = 0; i < num_items; ++i) {
                size_t label = start + i;
                size_t entry_idx = graph.getInternalIndex(static_cast<uint32_t>(label));

                deglib::search::ResultSet result_queue = graph.explore(
                    static_cast<uint32_t>(entry_idx),
                    k_search,
                    false,
                    max_distance_count);

                auto& cands = chunk_cands[i];
                cands.reserve(result_queue.size());
                while (!result_queue.empty()) {
                    const auto cand_idx = result_queue.top().getInternalIndex();
                    if (cand_idx != entry_idx) {
                        cands.push_back(graph.getExternalLabel(cand_idx));
                    }
                    result_queue.pop();
                }
            }
            chunk_search_times[chunk_id] = evp_common::now_ms() - t_search_start;

            // --- Rerank Phase for Chunk ---
            double t_rerank_start = evp_common::now_ms();
            for (size_t i = 0; i < num_items; ++i) {
                size_t label = start + i;
                const auto& cands = chunk_cands[i];
                const size_t num_cands = cands.size();

                const std::byte* query_ptr = train_vectors[label].data();
                std::vector<const void*> cand_ptrs(num_cands);
                for (size_t c = 0; c < num_cands; ++c) {
                    cand_ptrs[c] = train_vectors[cands[c]].data();
                }

                std::vector<float> exact_dists(num_cands);
                const void* dist_func_param = fp16_rerank_space.get_dist_func_param();
                deglib::distances::dispatch_distance(fp16_rerank_space, [&]<typename COMPARATOR>() {
                    deglib::distances::compare_batch<COMPARATOR>(query_ptr, cand_ptrs.data(), num_cands, dist_func_param, exact_dists.data());
                });

                struct Candidate {
                    uint32_t label;
                    float distance;
                };
                std::vector<Candidate> candidates(num_cands);
                for (size_t c = 0; c < num_cands; ++c) {
                    candidates[c] = {cands[c], exact_dists[c]};
                }

                const size_t n = std::min<size_t>(k_top, num_cands);
                std::partial_sort(candidates.begin(), candidates.begin() + n, candidates.begin() + num_cands,
                    [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });

                auto& result = results[label];
                result.resize(k_top);
                std::fill(result.begin(), result.end(), std::numeric_limits<uint32_t>::max());
                for (uint32_t j = 0; j < n; ++j) {
                    result[j] = candidates[j].label;
                }
            }
            chunk_rerank_times[chunk_id] = evp_common::now_ms() - t_rerank_start;
        });

    double total_ms = evp_common::now_ms() - t_start;
    double sum_search_ms = std::accumulate(chunk_search_times.begin(), chunk_search_times.end(), 0.0) / threads;
    double sum_rerank_ms = std::accumulate(chunk_rerank_times.begin(), chunk_rerank_times.end(), 0.0) / threads;

    std::printf("Exploration complete in %.2f ms (explore=%.2f ms, rerank=%.2f ms)\n",
                total_ms, sum_search_ms, sum_rerank_ms);

    float recall = -1.0f;
    if (compute_recall) {
        recall = evp_common::compute_recall(gt_data, results, k_top);
        std::printf("Recall@%u: %.4f  (max_dist=%u, explore=%.2f ms, rerank=%.2f ms)\n",
                    k_top, recall, max_distance_count, sum_search_ms, sum_rerank_ms);
    } else {
        evp_common::ivecs_write(output_path, results);
    }

    return { sum_search_ms, sum_rerank_ms, total_ms, recall };
}

/**
 * @brief Run function for Mode 4.
 */
static int run(
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t non_zeros,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top,
    uint32_t max_distance_count,
    uint32_t evpK,
    bool compute_recall,
    const std::string& output_path)
{
    const std::string h5path = data_path.string();
    std::printf("HDF5 mode (modi4): scanning '%s'\n", h5path.c_str());

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

    std::printf("=== EVP Bits Graph Benchmark (Mode 4: EvpBuildEvpExploreFP16Rerank) ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.3f, MODE=build-evp-explore-fp16-rerank\n",
                non_zeros, k_top, k_graph, k_ext, eps_ext);
    std::printf("Threads: %u\n\n", threads);

    // --------------------------------------------------------------------------
    // Quantize
    // --------------------------------------------------------------------------
    double t1 = evp_common::now_ms();

    auto quantized = deglib::quantization::quantize_batch(
        train_vectors, static_cast<uint32_t>(dims), non_zeros, threads);

    double quantize_ms = evp_common::now_ms() - t1;
    std::printf("Quantize time: %.2f ms\n", quantize_ms);
    std::printf("Quantized size: %.2f MB\n\n",
                static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

    // --------------------------------------------------------------------------
    // Build graph with Metric::EvpBits
    // --------------------------------------------------------------------------
    double t2 = evp_common::now_ms();

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

    double build_ms = evp_common::now_ms() - t2;
    std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
    std::printf("Build time: %.2f ms\n\n", build_ms);

    // --------------------------------------------------------------------------
    // Exploration sweep
    // --------------------------------------------------------------------------
    auto timings = run_exploration_sweep(graph, evpK, max_distance_count, static_cast<uint8_t>(threads),
                                         k_top, dims, train_vectors,
                                         compute_recall, gt_data, output_path);

    double total_time_ms = load_ms + quantize_ms + build_ms + timings.total_ms;

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    std::printf("========================================================================\n");
    std::printf("  FINAL SUMMARY (EVP Build, EVP Explore, FP16 Rerank - Mode 4)          \n");
    std::printf("========================================================================\n");
    std::printf("Load Time:               %.2f ms\n", load_ms);
    std::printf("Quantize Time:           %.2f ms\n", quantize_ms);
    std::printf("Graph Build Time:        %.2f ms\n", build_ms);
    std::printf("Explore Time:            %.2f ms\n", timings.search_ms);
    std::printf("Rerank Time:             %.2f ms\n", timings.rerank_ms);
    std::printf("Exploration Wall Time:   %.2f ms\n", timings.total_ms);
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
    std::printf("  evpK:                  %u\n", evpK);
    std::printf("  max_dist:              %u\n", max_distance_count);
    std::printf("  threads:               %u\n", threads);
    std::printf("---------------------------------------------\n");
    std::printf("Dataset Info:\n");
    std::printf("  Vectors:               %zu\n", count);
    std::printf("  Dimensions:            %zu\n", dims);
    std::printf("=============================================\n\n");

    return 0;
}

} // namespace task1::mode4
