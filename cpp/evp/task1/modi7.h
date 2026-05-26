#pragma once

/**
 * @file modi7.h
 * @brief Benchmark Mode 7: EVP Build, Asymmetric Search, FP16 Rerank (evp-build-fp16-asymmetric-search-rerank)
 * 
 * Behavior:
 * 1. Quantizes training features to EVP bits representation.
 * 2. Builds a SizeBoundedGraph using the EvpBits metric.
 * 3. Converts to a ReadOnlyGraph using the Metric::FP16EvpAsymmetric metric.
 * 4. Performs asymmetric search (FP16 queries vs EVP database items in graph).
 * 5. Reranks the retrieved candidates using exact FP16 Inner Product distances.
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

namespace task1::mode7 {

/**
 * @brief Performs search/exploration sweep over different graph search limits, calculates recall or exports Top-K results.
 */
static void run_exploration_sweep(
    const deglib::search::SearchGraph& graph,
    uint32_t k_top,
    uint32_t max_distance_count,
    uint8_t threads,
    uint32_t evpK,
    size_t dims,
    const std::vector<std::vector<std::byte>>& train_vectors,
    bool compute_recall,
    const std::vector<std::vector<int32_t>>& gt_data,
    const std::string& output_path)
{
    deglib::FloatSpace fp16_rerank_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
    deglib::DISTFUNC<float> rerank_dist_func = fp16_rerank_space.get_dist_func();
    size_t count = graph.size();
    const uint32_t k_search = evpK > 0 ? evpK : 50; 
    std::printf("\n--- Exploration (EVP Build FP16 Asymmetric Search FP16 Rerank, max_dist=%u) ---\n", max_distance_count);

    // --- Search ---
    double t_explore_start = evp_common::now_ms();
    std::vector<std::vector<uint32_t>> explore_candidates(count);

    deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
        [&](size_t label, size_t thread_id) {
            (void)thread_id;
            size_t entry_idx = graph.getInternalIndex(static_cast<uint32_t>(label));

            std::vector<uint32_t> entry_indices = { static_cast<uint32_t>(entry_idx) };
            const std::byte* query = train_vectors[label].data();
            deglib::search::ResultSet result_queue = graph.search(
                entry_indices,
                query,
                1000.0f,
                k_search + 1,
                nullptr,
                max_distance_count + 1);

            auto& cands = explore_candidates[label];
            cands.reserve(result_queue.size());
            while (!result_queue.empty()) {
                const auto cand_idx = result_queue.top().getInternalIndex();
                if (cand_idx != entry_idx) {
                    cands.push_back(graph.getExternalLabel(cand_idx));
                }
                result_queue.pop();
            }
            if (cands.size() > k_search) {
                cands.resize(k_search);
            }
        });
    double explore_ms = evp_common::now_ms() - t_explore_start;

    // --- Rerank ---
    double t_rerank_start = evp_common::now_ms();
    std::vector<std::vector<uint32_t>> results(count);
    deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
        [&](size_t label, size_t thread_id) {
            (void)thread_id;
            struct Candidate {
                uint32_t label;
                float distance;
            };
            const auto& cands = explore_candidates[label];
            std::vector<Candidate> candidates;
            candidates.reserve(cands.size());

            const std::byte* query_ptr = train_vectors[label].data();
            size_t dims_size = dims;

            for (uint32_t cand_label : cands) {
                const std::byte* cand_ptr = train_vectors[cand_label].data();
                float exact_dist = rerank_dist_func(query_ptr, cand_ptr, &dims_size);
                candidates.push_back({cand_label, exact_dist});
            }

            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                return a.distance < b.distance;
            });

            auto& result = results[label];
            result.resize(k_top);
            std::fill(result.begin(), result.end(), std::numeric_limits<uint32_t>::max());
            for (uint32_t j = 0; j < k_top && j < candidates.size(); ++j) {
                result[j] = candidates[j].label;
            }
        });
    double rerank_ms = evp_common::now_ms() - t_rerank_start;
    double total_ms = explore_ms + rerank_ms;
    std::printf("Exploration complete in %.2f ms (search=%.2f ms, rerank=%.2f ms)\n",
                total_ms, explore_ms, rerank_ms);

    if (compute_recall) {
        float recall = evp_common::compute_recall(results, gt_data, count, k_top);
        std::printf("Recall@%u: %.4f  (max_dist=%u, search=%.2f ms, rerank=%.2f ms)\n",
                    k_top, recall, max_distance_count, explore_ms, rerank_ms);
    } else {
        evp_common::ivecs_write(output_path, results);
    }
}

/**
 * @brief Run function for Mode 7.
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
    std::printf("HDF5 mode (modi7): scanning '%s'\n", h5path.c_str());

    auto datasets = hdf5_reader::scan_datasets(h5path);
    auto& train_info = hdf5_reader::find_dataset(datasets, "train");

    double t_load = evp_common::now_ms();
    std::vector<std::vector<std::byte>> train_vectors = hdf5_reader::read_matrix_bytes(h5path, train_info);
    size_t dims = static_cast<size_t>(train_info.num_cols);
    size_t count = static_cast<size_t>(train_info.num_rows);

    std::vector<std::vector<int32_t>> gt_data;
    size_t gt_dims = 0, gt_count = 0;
    if (compute_recall) {
        auto& allknn_info = hdf5_reader::find_dataset(datasets, "allknn/knns");
        std::printf("  allknn/knns: %llu x %llu (elem=%uB)\n",
            (unsigned long long)allknn_info.num_rows,
            (unsigned long long)allknn_info.num_cols,
            allknn_info.element_size);
        gt_data = hdf5_reader::read_matrix_int32(h5path, allknn_info);
        gt_dims = static_cast<size_t>(allknn_info.num_cols);
        gt_count = static_cast<size_t>(allknn_info.num_rows);
    }
    double load_ms = evp_common::now_ms() - t_load;

    std::printf("Loaded train:  %zu vectors, dim=%zu\n", count, dims);
    if (compute_recall) {
        std::printf("Ground truth:  %zu queries, top-%zu\n", gt_count, gt_dims);
    }
    std::printf("Load time:     %.2f ms\n\n", load_ms);

    if (compute_recall && count != gt_count) {
        std::fprintf(stderr,
            "Error: train and allknn must contain the same number of entries (%zu vs %zu)\n",
            count, gt_count);
        return 1;
    }

    std::printf("=== EVP Bits Graph Benchmark (Mode 7: EvpBuildFP16AsymmetricSearchRerank) ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f, MODE=build-fp16-asymmetric-search-rerank\n",
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

    const size_t bytes_per_evp = dims / 4;
    for (size_t i = 0; i < count; ++i) {
        std::vector<std::byte> feature(quantized.data() + i * bytes_per_evp, quantized.data() + (i + 1) * bytes_per_evp);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feature));
    }

    auto build_callback = [](deglib::builder::BuilderStatus& status) {
        std::printf("  Build step %llu: added=%llu, improved=%llu\n",
                    (unsigned long long)status.step,
                    (unsigned long long)status.added,
                    (unsigned long long)status.improved);
    };

    builder.build(build_callback, false);

    double build_ms = evp_common::now_ms() - t2;
    std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
    std::printf("Build time: %.2f ms\n\n", build_ms);

    // --------------------------------------------------------------------------
    // Exploration sweep
    // --------------------------------------------------------------------------
    double conversion_ms = 0.0;
    double t_copy_start = evp_common::now_ms();
    deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
    deglib::graph::ReadOnlyGraph readonly_asym_graph(static_cast<uint32_t>(count), k_graph, asymmetric_space, graph);
    conversion_ms = evp_common::now_ms() - t_copy_start;
    std::printf("ReadOnlyGraph construction (copy) time: %.2f ms\n", conversion_ms);

    run_exploration_sweep(readonly_asym_graph, k_top, max_distance_count, static_cast<uint8_t>(threads),
                          evpK, dims, train_vectors,
                          compute_recall, gt_data, output_path);

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    std::printf("=============================================\n");
    std::printf("          FINAL SUMMARY (EVP Bits - Mode 7)\n");
    std::printf("=============================================\n");
    std::printf("Quantize:                %.2f ms\n", quantize_ms);
    std::printf("Graph Build:             %.2f ms\n", build_ms);
    std::printf("Graph Conversion:        %.2f ms\n", conversion_ms);
    std::printf("---------------------------------------------\n");
    std::printf("Vectors:                 %zu\n", count);
    std::printf("Dimensions:              %zu\n", dims);
    std::printf("NON_ZEROS:               %u\n", non_zeros);
    std::printf("=============================================\n\n");

    return 0;
}

} // namespace task1::mode7
