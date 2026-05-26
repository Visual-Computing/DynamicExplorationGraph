/**
 * @file deglib_evp_test.cpp
 * @brief Graph Benchmark: build a DEG graph and evaluate exploration recall@15.
 *
 * Uses only deglib headers (no benchmark.h / dataset.h / build.h).
 * Loads raw .fvecs and .ivecs, builds a SizeBoundedGraph, explores for Top-15,
 * and compares against allknn ground truth.
 *
 * Modes:
 *   evp   - quantize with deglib::quantization, build graph with Metric::EvpBits
 *   raw   - use original float feature vectors directly with Metric::InnerProduct
 *   fp16  - convert FP32 to FP16, build graph with Metric::FP16InnerProduct
 *
 * Usage:
 *   deglib_evp_test <data_path> [evp|evp-no-rerank|evp-build-fp16-search|evp-build-fp16-proxy|raw|fp16|evp-linear]
 *
 * When data_path is a .h5 file: loads train and allknn/knns directly from HDF5.
 * When data_path is a directory: loads train.hvecs and allknn.ivecs (legacy).
 */

#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

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

#include "evp_common.h"
#include "hdf5_reader.h"

// ============================================================================
// Consolidated Modes
// ============================================================================

enum class BenchmarkMode {
    EvpBuildEvpExploreFP16Rerank,
    EvpBuildEvpExplore,
    EvpBuildFP16ExternalSearch,
    FP16BuildFP16Explore,
    EvpLinearSearch,
    EvpBuildFP16AsymmetricSearch,
    EvpBuildFP16AsymmetricSearchRerank
};

// ============================================================================
// Shared helpers
// ============================================================================

static void run_exploration_sweep(
    const deglib::search::SearchGraph& graph,
    const std::vector<std::vector<int32_t>>& gt_data,
    size_t count,
    size_t gt_dims,
    uint32_t k_top,
    uint8_t threads,
    const char* label,
    bool use_search = false,
    const void* rerank_data = nullptr,
    size_t rerank_feature_bytes = 0,
    deglib::DISTFUNC<float> rerank_dist_func = nullptr,
    size_t dims = 0,
    bool use_asymmetric_search = false,
    const void* query_data = nullptr,
    const std::vector<std::vector<std::byte>>* query_data_vecs = nullptr,
    const std::vector<std::vector<std::byte>>* rerank_data_vecs = nullptr)
{
    std::printf("\n--- Exploration sweep (%s) ---\n", label);

    const uint32_t k_explore = k_top;

    // Fixed max_distance_count values
    const uint32_t max_distances[] = { k_explore, 50, 100, 200, 300, 400 };
    float best_recall = -1.0f;
    uint32_t best_max_dist = 0;
    double best_explore_ms = 0.0;
    bool stop_sweep = false;

    for (uint32_t md_idx = 0; md_idx < sizeof(max_distances) / sizeof(max_distances[0]) && !stop_sweep; md_idx++) {
        const uint32_t max_distance_count = max_distances[md_idx];

        // --- Phase 1: Graph Exploration ---
        double t_explore_start = now_ms();
        std::vector<std::vector<uint32_t>> explore_candidates(count);

        deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
            [&](size_t label, size_t thread_id) {
                (void)thread_id;
                size_t entry_idx = graph.getInternalIndex(static_cast<uint32_t>(label));
                uint32_t k_search = (rerank_data != nullptr || rerank_data_vecs != nullptr) ? (use_asymmetric_search ? 25 : std::max(k_top, max_distance_count)) : k_explore;

                deglib::search::ResultSet result_queue;
                if (use_asymmetric_search) {
                    std::vector<uint32_t> entry_indices = { static_cast<uint32_t>(entry_idx) };
                    const std::byte* query = query_data_vecs
                        ? (*query_data_vecs)[label].data()
                        : (static_cast<const std::byte*>(query_data) + label * dims * sizeof(uint16_t));
                    result_queue = graph.search(
                        entry_indices,
                        query,
                        1000.0f,
                        k_search + 1,
                        nullptr,
                        max_distance_count + 1);
                } else if (use_search) {
                    std::vector<uint32_t> entry_indices = { static_cast<uint32_t>(entry_idx) };
                    const std::byte* query = graph.getFeatureVector(static_cast<uint32_t>(entry_idx));
                    result_queue = graph.search(
                        entry_indices,
                        query,
                        1000.0f,
                        k_search + 1,
                        nullptr,
                        max_distance_count + 1);
                } else {
                    result_queue = graph.explore(
                        static_cast<uint32_t>(entry_idx),
                        k_search,
                        false,  // include_entry (false: exclude self)
                        max_distance_count);
                }

                auto& cands = explore_candidates[label];
                cands.reserve(result_queue.size());
                while (!result_queue.empty()) {
                    const auto cand_idx = result_queue.top().getInternalIndex();
                    if (cand_idx != entry_idx) {
                        cands.push_back(graph.getExternalLabel(cand_idx));
                    }
                    result_queue.pop();
                }
                if ((use_search || use_asymmetric_search) && cands.size() > k_search) {
                    cands.resize(k_search);
                }
            });
        double explore_ms = now_ms() - t_explore_start;

        // --- Phase 2: Reranking (if rerank_data or rerank_data_vecs is provided) ---
        double rerank_ms = 0.0;
        std::vector<std::vector<uint32_t>> results(count);

        if (rerank_data != nullptr || rerank_data_vecs != nullptr) {
            double t_rerank_start = now_ms();
            const auto* base = static_cast<const std::byte*>(rerank_data);
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

                    size_t dims_size = dims;
                    const std::byte* query_ptr = rerank_data_vecs
                        ? (*rerank_data_vecs)[label].data()
                        : (base + label * rerank_feature_bytes);

                    for (uint32_t cand_label : cands) {
                        const std::byte* cand_ptr = rerank_data_vecs
                            ? (*rerank_data_vecs)[cand_label].data()
                            : (base + cand_label * rerank_feature_bytes);
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
            rerank_ms = now_ms() - t_rerank_start;
        } else {
            results = std::move(explore_candidates);
        }

        double explore_ms_single = explore_ms + rerank_ms;

        double wall_explore = explore_ms;
        double wall_rerank = rerank_ms;

        // Calculate recall
        int total_hits = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto& gt_row = gt_data[i];
            for (uint32_t k = 1; k <= k_top && k < (uint32_t)gt_row.size(); ++k) {
                uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1->0 indexed
                const auto& row = results[i];
                const uint32_t row_len = static_cast<uint32_t>(std::min(row.size(), static_cast<size_t>(k_top)));
                for (uint32_t j = 0; j < row_len; ++j) {
                    if (row[j] == gt_idx) {
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

        if (rerank_data != nullptr) {
            std::printf("  max_dist=%6u  recall=%.4f  time=%.2f ms (explore=%.2f ms, rerank=%.2f ms)\n",
                        max_distance_count, recall, explore_ms_single, wall_explore, wall_rerank);
        } else {
            std::printf("  max_dist=%6u  recall=%.4f  time=%.2f ms\n",
                        max_distance_count, recall, explore_ms_single);
        }

        if (recall >= 0.8f) {
            std::printf("  Reached recall target 0.8, stopping sweep\n");
            stop_sweep = true;
        }
    }

    std::printf("\nBest recall@%d: %.4f (max_distance_count=%u, explore_time=%.2f ms)\n",
                k_top, best_recall, best_max_dist, best_explore_ms);
}

// ============================================================================
// Unified Benchmark Entry Point
// ============================================================================

// Pre-loaded data variant: ALL data loading happens in main().
static int run_benchmark(
    BenchmarkMode mode,
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t non_zeros,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top,
    std::vector<std::vector<std::byte>> train_vectors_in,
    std::vector<std::vector<int32_t>> gt_data_in,
    size_t dims_in,
    size_t count_in,
    size_t gt_dims_in,
    size_t gt_count_in)
{
    std::vector<std::vector<std::byte>>& train_vectors = train_vectors_in;
    std::vector<std::vector<int32_t>>& gt_data = gt_data_in;
    size_t dims = dims_in;
    size_t count = count_in;
    size_t gt_dims = gt_dims_in;
    size_t gt_count = gt_count_in;

    if (count != gt_count) {
        std::fprintf(stderr,
            "Error: train and allknn must contain the same number of entries (%zu vs %zu)\n",
            count, gt_count);
        return 1;
    }

    if (mode == BenchmarkMode::FP16BuildFP16Explore) {
        std::printf("=== FP16 Build FP16 Explore Graph Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f\n",
                    k_top, k_graph, k_ext, eps_ext);
        std::printf("Threads: %u\n\n", threads);

        std::printf("FP16 data size: %.2f MB\n\n",
                    static_cast<double>(count * dims * sizeof(uint16_t)) / (1024.0 * 1024.0));

        // --------------------------------------------------------------------------
        // Build graph with Metric::FP16InnerProduct
        // --------------------------------------------------------------------------
        double t2 = now_ms();

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

        // Add all entries from FP16 data directly using zero-copy std::move
        const size_t bytes_per_fp16 = dims * sizeof(uint16_t);
        for (size_t i = 0; i < count; ++i) {
            builder.addEntry(static_cast<uint32_t>(i), std::move(train_vectors[i]));
        }

        auto build_callback = [](deglib::builder::BuilderStatus& status) {
            std::printf("  Build step %llu: added=%llu, improved=%llu\n",
                        (unsigned long long)status.step,
                        (unsigned long long)status.added,
                        (unsigned long long)status.improved);
        };

        builder.build(build_callback, false);

        double build_ms = now_ms() - t2;
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        // --------------------------------------------------------------------------
        // Exploration sweep
        // --------------------------------------------------------------------------
        run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads), "FP16 Build FP16 Explore");

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (FP16 Build FP16 Explore)\n");
        std::printf("=============================================\n");
        std::printf("Graph Build:             %.2f ms\n", build_ms);
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("FP16 bytes/vector:       %zu\n", bytes_per_fp16);
        std::printf("=============================================\n\n");

        return 0;
    } else if (mode == BenchmarkMode::EvpBuildEvpExploreFP16Rerank || mode == BenchmarkMode::EvpBuildEvpExplore || mode == BenchmarkMode::EvpBuildFP16ExternalSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank) {
        std::printf("=== EVP Bits Graph Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f, MODE=%s\n",
                    non_zeros, k_top, k_graph, k_ext, eps_ext,
                    mode == BenchmarkMode::EvpBuildFP16ExternalSearch ? "build-fp16-external-search" :
                    (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch ? "build-fp16-asymmetric-search" :
                    (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank ? "build-fp16-asymmetric-search-rerank" :
                    (mode == BenchmarkMode::EvpBuildEvpExploreFP16Rerank ? "build-evp-explore-fp16-rerank" : "build-evp-explore"))));
        std::printf("Threads: %u\n\n", threads);

        // --------------------------------------------------------------------------
        // Quantize
        // --------------------------------------------------------------------------
        double t1 = now_ms();

        auto quantized = deglib::quantization::quantize_batch(
            train_vectors, static_cast<uint32_t>(dims), non_zeros, threads);

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

        double build_ms = now_ms() - t2;
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", static_cast<size_t>(graph.size()), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        // --------------------------------------------------------------------------
        // Exploration sweep
        // --------------------------------------------------------------------------
        double conversion_ms = 0.0;
        if (mode == BenchmarkMode::EvpBuildFP16ExternalSearch) {
            // Copy loaded FP16 data to a mutable vector for in-place reordering (specifically required by ReadOnlyGraphExternal)
            double t_copy_start = now_ms();
            std::vector<uint16_t> fp16_data(count * dims);
            for (size_t i = 0; i < count; ++i) {
                std::memcpy(fp16_data.data() + i * dims, train_vectors[i].data(), dims * sizeof(uint16_t));
            }
            conversion_ms = now_ms() - t_copy_start;

            // Permute FP16 array in-place so that fp16_data[internal_index] holds the
            // features for that internal index (no proxy needed, no extra allocation).
            double t_reorder_start = now_ms();
            deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(
                graph, fp16_data.data(), dims);
            double reorder_ms = now_ms() - t_reorder_start;
            std::printf("In-place feature reorder time: %.2f ms\n", reorder_ms);
            conversion_ms += reorder_ms;

            // Build the external graph — references fp16_data directly, no copies.
            double t_ext_start = now_ms();
            deglib::FloatSpace fp16_ext_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            deglib::graph::ReadOnlyGraphExternal fp16_ext_graph(fp16_ext_space, graph, fp16_data.data());
            conversion_ms += now_ms() - t_ext_start;
            std::printf("ReadOnlyGraphExternal construction time: %.2f ms\n", now_ms() - t_ext_start);

            run_exploration_sweep(fp16_ext_graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build FP16 External Search (ReadOnlyGraphExternal)",
                                  true);
        } else if (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch) {
            double t_copy_start = now_ms();
            deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
            deglib::graph::ReadOnlyGraph readonly_asym_graph(static_cast<uint32_t>(count), k_graph, asymmetric_space, graph);
            conversion_ms = now_ms() - t_copy_start;
            std::printf("ReadOnlyGraph construction (copy) time: %.2f ms\n", conversion_ms);

            run_exploration_sweep(readonly_asym_graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build FP16 Asymmetric Search (ReadOnlyGraph)",
                                  false,
                                  nullptr,
                                  0,
                                  nullptr,
                                  dims,
                                  true,
                                  nullptr,
                                  &train_vectors);
        } else if (mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank) {
            double t_copy_start = now_ms();
            deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
            deglib::graph::ReadOnlyGraph readonly_asym_graph(static_cast<uint32_t>(count), k_graph, asymmetric_space, graph);
            conversion_ms = now_ms() - t_copy_start;
            std::printf("ReadOnlyGraph construction (copy) time: %.2f ms\n", conversion_ms);

            deglib::FloatSpace fp16_rerank_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            run_exploration_sweep(readonly_asym_graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build FP16 Asymmetric Search FP16 Rerank (ReadOnlyGraph)",
                                  false,
                                  nullptr,
                                  dims * sizeof(uint16_t),
                                  fp16_rerank_space.get_dist_func(),
                                  dims,
                                  true,
                                  nullptr,
                                  &train_vectors,
                                  &train_vectors);
        } else if (mode == BenchmarkMode::EvpBuildEvpExploreFP16Rerank) {
            deglib::FloatSpace fp16_rerank_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build EVP Explore FP16 Rerank",
                                  false,
                                  nullptr,
                                  dims * sizeof(uint16_t),
                                  fp16_rerank_space.get_dist_func(),
                                  dims,
                                  false,
                                  nullptr,
                                  nullptr,
                                  &train_vectors);
        } else {
            run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, static_cast<uint8_t>(threads),
                                  "EVP Build EVP Explore");
        }

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (EVP Bits)\n");
        std::printf("=============================================\n");
        std::printf("Quantize:                %.2f ms\n", quantize_ms);
        std::printf("Graph Build:             %.2f ms\n", build_ms);
        if (mode == BenchmarkMode::EvpBuildFP16ExternalSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearch || mode == BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank) {
            std::printf("Graph Conversion:        %.2f ms\n", conversion_ms);
        }
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("NON_ZEROS:               %u\n", non_zeros);
        std::printf("=============================================\n\n");

        return 0;
    } else if (mode == BenchmarkMode::EvpLinearSearch) {
        std::printf("=== EVP Bits Linear Search Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("NON_ZEROS=%u, K_TOP=%u\n", non_zeros, k_top);
        std::printf("Threads: %u\n\n", threads);

        // --------------------------------------------------------------------------
        // Quantize
        // --------------------------------------------------------------------------
        double t1 = now_ms();

        auto quantized = deglib::quantization::quantize_batch(
            train_vectors, static_cast<uint32_t>(dims), non_zeros, threads);

        double quantize_ms = now_ms() - t1;
        std::printf("Quantize time: %.2f ms\n", quantize_ms);
        std::printf("Quantized size: %.2f MB\n\n",
                    static_cast<double>(quantized.size()) / (1024.0 * 1024.0));

        // --------------------------------------------------------------------------
        // Linear Search using distances.h
        // --------------------------------------------------------------------------
        double t_search_start = now_ms();

        std::vector<std::vector<uint32_t>> results(count, std::vector<uint32_t>(k_top));
        const size_t bytes_per_evp = dims / 4;
        const uint32_t dims_u32 = static_cast<uint32_t>(dims);

        deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
            [&](size_t i, size_t thread_id) {
                (void)thread_id;
                const std::byte* query_ptr = quantized.data() + i * bytes_per_evp;

                std::vector<std::pair<float, uint32_t>> top;
                top.reserve(k_top + 1);

                for (size_t j = 0; j < count; ++j) {
                    if (i == j) {
                        continue;
                    }

                    const std::byte* cand_ptr = quantized.data() + j * bytes_per_evp;
                    float dist = deglib::distances::EvpBitsSimilarity::compare(query_ptr, cand_ptr, &dims_u32);

                    if (top.size() < k_top) {
                        top.push_back({dist, static_cast<uint32_t>(j)});
                        std::push_heap(top.begin(), top.end());
                    } else if (dist < top.front().first) {
                        std::pop_heap(top.begin(), top.end());
                        top.back() = {dist, static_cast<uint32_t>(j)};
                        std::push_heap(top.begin(), top.end());
                    }
                }

                std::sort_heap(top.begin(), top.end());

                for (uint32_t k = 0; k < k_top && k < top.size(); ++k) {
                    results[i][k] = top[k].second;
                }
            });

        double search_ms = now_ms() - t_search_start;
        std::printf("Linear Search time: %.2f ms\n", search_ms);

        // --------------------------------------------------------------------------
        // Calculate Recall
        // --------------------------------------------------------------------------
        int total_hits = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto& gt_row = gt_data[i];
            for (uint32_t k = 1; k <= k_top && k < (uint32_t)gt_row.size(); ++k) {
                uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1->0 indexed
                for (uint32_t j = 0; j < k_top; ++j) {
                    if (results[i][j] == gt_idx) {
                        total_hits++;
                        break;
                    }
                }
            }
        }

        float recall = static_cast<float>(total_hits) / (count * k_top);
        std::printf("Recall@%u: %.4f\n", k_top, recall);

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (EVP Bits Linear Search)\n");
        std::printf("=============================================\n");
        std::printf("Quantize:                %.2f ms\n", quantize_ms);
        std::printf("Linear Search:           %.2f ms\n", search_ms);
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("Recall@%u:                %.4f\n", k_top, recall);
        std::printf("=============================================\n\n");

        return 0;
    }
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
  try {
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
    std::string mode = "evp-build-fp16-asymmetric-search";  // default

    if (argc >= 2) {
        data_path = argv[1];
    } else {
#ifdef DATA_PATH
        data_path = DATA_PATH;
#else
        std::fprintf(stderr, "Usage: %s <data_path_or_h5_file> [evp-build-evp-explore-fp16-rerank|evp-build-evp-explore|evp-build-fp16-external-search|fp16-build-fp16-explore|evp-linear-search|evp-build-fp16-asymmetric-search|evp-build-fp16-asymmetric-search-rerank]\n", argv[0]);
        return 1;
#endif
    }

    if (argc >= 3) {
        mode = argv[2];
    }

    // Support old short names as aliases for ease of use
    if (mode == "fp16") mode = "fp16-build-fp16-explore";
    if (mode == "evp") mode = "evp-build-evp-explore-fp16-rerank";
    if (mode == "evp-no-rerank") mode = "evp-build-evp-explore";
    if (mode == "evp-linear") mode = "evp-linear-search";
    if (mode == "evp-asymmetric") mode = "evp-build-fp16-asymmetric-search";
    if (mode == "evp-asymmetric-rerank") mode = "evp-build-fp16-asymmetric-search-rerank";

    if (mode != "evp-build-evp-explore-fp16-rerank" && mode != "evp-build-evp-explore" && mode != "evp-build-fp16-external-search" && mode != "fp16-build-fp16-explore" && mode != "evp-linear-search" && mode != "evp-build-fp16-asymmetric-search" && mode != "evp-build-fp16-asymmetric-search-rerank") {
        std::fprintf(stderr, "Unknown mode '%s'. Use 'evp-build-evp-explore-fp16-rerank', 'evp-build-evp-explore', 'evp-build-fp16-external-search', 'fp16-build-fp16-explore', 'evp-linear-search', 'evp-build-fp16-asymmetric-search', or 'evp-build-fp16-asymmetric-search-rerank'.\n", mode.c_str());
        return 1;
    }

    const size_t threads = 6;

    // --------------------------------------------------------------------------
    // Dispatch to selected mode
    // --------------------------------------------------------------------------
    BenchmarkMode benchmark_mode = BenchmarkMode::FP16BuildFP16Explore;
    if (mode == "evp-build-evp-explore-fp16-rerank") {
        benchmark_mode = BenchmarkMode::EvpBuildEvpExploreFP16Rerank;
    } else if (mode == "evp-build-evp-explore") {
        benchmark_mode = BenchmarkMode::EvpBuildEvpExplore;
    } else if (mode == "evp-build-fp16-external-search") {
        benchmark_mode = BenchmarkMode::EvpBuildFP16ExternalSearch;
    } else if (mode == "evp-linear-search") {
        benchmark_mode = BenchmarkMode::EvpLinearSearch;
    } else if (mode == "evp-build-fp16-asymmetric-search") {
        benchmark_mode = BenchmarkMode::EvpBuildFP16AsymmetricSearch;
    } else if (mode == "evp-build-fp16-asymmetric-search-rerank") {
        benchmark_mode = BenchmarkMode::EvpBuildFP16AsymmetricSearchRerank;
    }

    // --------------------------------------------------------------------------
    // Load data (always in main, never in run_benchmark)
    // --------------------------------------------------------------------------
    const std::string ext = data_path.extension().string();
    const bool is_hdf5 = (ext == ".h5" || ext == ".hdf5");

    std::vector<std::vector<std::byte>> train_vectors;
    std::vector<std::vector<int32_t>> gt_data;
    size_t dims = 0, count = 0, gt_dims = 0, gt_count = 0;

    if (is_hdf5) {
        const std::string h5path = data_path.string();
        std::printf("HDF5 mode: scanning '%s'\n", h5path.c_str());

        auto datasets = hdf5_reader::scan_datasets(h5path);
        auto& train_info = hdf5_reader::find_dataset(datasets, "train");
        auto& allknn_info = hdf5_reader::find_dataset(datasets, "allknn/knns");

        std::printf("  train:       %llu x %llu (elem=%uB)\n",
            (unsigned long long)train_info.num_rows,
            (unsigned long long)train_info.num_cols,
            train_info.element_size);
        std::printf("  allknn/knns: %llu x %llu (elem=%uB)\n",
            (unsigned long long)allknn_info.num_rows,
            (unsigned long long)allknn_info.num_cols,
            allknn_info.element_size);

        double t_load = now_ms();
        train_vectors = hdf5_reader::read_dataset_as_vecs(h5path, train_info);
        gt_data = hdf5_reader::read_dataset_as_ints(h5path, allknn_info, gt_dims, gt_count);
        dims = static_cast<size_t>(train_info.num_cols);
        count = static_cast<size_t>(train_info.num_rows);
        double load_ms = now_ms() - t_load;

        std::printf("Loaded train:  %zu vectors, dim=%zu\n", count, dims);
        std::printf("Ground truth:  %zu queries, top-%zu\n", gt_count, gt_dims);
        std::printf("Load time:     %.2f ms\n\n", load_ms);
    }
    else {
        // Legacy .hvecs / .ivecs directory path
        const auto train_hvecs  = data_path / "train.hvecs";
        const auto allknn_ivecs = data_path / "allknn.ivecs";

        if (!std::filesystem::exists(train_hvecs)) {
            std::fprintf(stderr, "Error: %ls not found\n", train_hvecs.wstring().c_str());
            return 1;
        }
        if (!std::filesystem::exists(allknn_ivecs)) {
            std::fprintf(stderr, "Error: %ls not found\n", allknn_ivecs.wstring().c_str());
            return 1;
        }

        double t_load = now_ms();
        train_vectors = hvecs_read(train_hvecs.string().c_str(), dims, count);
        auto gt_raw = deglib::ivecs_read(allknn_ivecs.string().c_str(), gt_dims, gt_count);
        const int32_t* raw = reinterpret_cast<const int32_t*>(gt_raw.get());
        gt_data.resize(gt_count);
        for (size_t i = 0; i < gt_count; ++i) {
            gt_data[i] = std::vector<int32_t>(raw + i * gt_dims, raw + (i + 1) * gt_dims);
        }
        double load_ms = now_ms() - t_load;

        std::printf("Loaded %ls: %zu vectors, dim=%zu\n", train_hvecs.filename().wstring().c_str(), count, dims);
        std::printf("Ground truth: %zu queries, top-%zu\n", gt_count, gt_dims);
        std::printf("Load time: %.2f ms\n\n", load_ms);
    }

    return run_benchmark(
        benchmark_mode, data_path, static_cast<uint32_t>(threads),
        NON_ZEROS, K_GRAPH, K_EXT, EPS_EXT, K_TOP,
        std::move(train_vectors), std::move(gt_data),
        dims, count, gt_dims, gt_count);
  } catch (const std::exception& ex) {
      std::fprintf(stderr, "Fatal error: %s\n", ex.what());
      return 1;
  }
}
