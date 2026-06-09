#pragma once

/**
 * @file mode5.h
 * @brief Task 2 Mode 5: FP32 Build (L2-converted) + FP16 Inner Product Search
 *
 *  .\deglib_sisap.exe task2  "C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5" mode5 --eps-ext 0.001 --k-ext 64 --k-graph 32 --build-threads 1 --max-dist 5000,6000,7000,8000 --eps-search 0.15,0.18,0.19,0.2,0.3 --num-runs 10 --flas


  --- eps_search=0.150 ---
    max_dist=5000 has recall 73.75 % and search time 19.8 ms
    max_dist=6000 has recall 76.01 % and search time 22.4 ms
    max_dist=7000 has recall 77.30 % and search time 24.6 ms
    max_dist=8000 has recall 78.17 % and search time 25.5 ms

  --- eps_search=0.180 ---
    max_dist=5000 has recall 75.39 % and search time 22.2 ms
    max_dist=6000 has recall 78.46 % and search time 25.5 ms
    max_dist=7000 has recall 80.54 % and search time 28.8 ms
    max_dist=8000 has recall 82.14 % and search time 31.8 ms

  --- eps_search=0.190 ---
    max_dist=5000 has recall 75.75 % and search time 22.1 ms
    max_dist=6000 has recall 78.90 % and search time 26.0 ms
    max_dist=7000 has recall 81.13 % and search time 29.8 ms
    max_dist=8000 has recall 82.96 % and search time 32.8 ms
    
========================================================================
  FINAL SUMMARY (L2-structured Build, FP16 IP Search (FLAS) - Mode 5)
========================================================================
Load Time:               105.7 ms
Quantize Time:            49.7 ms
Graph Build Time:         23.8 s
Graph Conversion Time:   148.6 ms
Pruning Time:              0.0 ms
Explore Time:             28.8 ms
Rerank Time:               0.0 ms
FLAS Time:                39.2 s
Total Elapsed Time:       24.1 s
Recall@30:               80.54 %
------------------------------------------------------------------------
Hyperparameters:
  K_TOP:                 30
  K_GRAPH:               32
  K_EXT:                 64
  EPS_EXT:               0.001
  OPT_TARGET:            LowLID
  max_dist:              7000
  threads:               8
  prune_worst:           0
------------------------------------------------------------------------
Dataset Info:
  Vectors:               256921
  Dimensions:            128
========================================================================

 *
 * Behavior:
 * 1. Loads FP32 training vectors from "train" as the database.
 * 2. Computes the maximum squared norm M^2 of database vectors.
 * 3. Appends sqrt(M^2 - ||x||^2) to each database vector (d+1 dimensions).
 * 4. Builds a SizeBoundedGraph using Metric::L2 with FloatSpace(dims + 1, Metric::L2).
 * 5. Converts original d-dimensional database vectors to FP16.
 * 6. Creates a ReadOnlyGraph using Metric::FP16InnerProduct with the FP16 database,
 *    mapping the graph structure built in L2-space.
 * 7. Loads FP32 query vectors, converts them to FP16 (d dimensions).
 * 8. For each query, uses fp16_graph.search() with Metric::FP16InnerProduct.
 * 9. Tracks build time, conversion time, and search time separately.
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
#include "repository.h"

#include "../sisap_common.h"
#include "../hdf5_reader.h"

#include "distance/fp16.h"

namespace task2::mode_l2_fp16_ip {

struct ExplorationTimings {
    double search_ms = 0.0;
    float recall = -1.0f;
};

using deglib::distances::floats_to_fp16;

static ExplorationTimings run_search(
    const deglib::graph::ReadOnlyGraph& graph,
    const std::vector<std::vector<uint16_t>>& queries,
    uint32_t k_top,
    float eps_search,
    uint32_t max_dist,
    uint8_t threads,
    bool compute_recall,
    int num_runs = 1,
    const std::vector<std::vector<int32_t>>& gt_data = {},
    const std::string& output_path = "")
{
    size_t count = queries.size();
    const size_t chunk_size = 8192;
    const size_t num_chunks = (count + chunk_size - 1) / chunk_size;
    std::vector<std::vector<uint32_t>> results(count);
    std::vector<double> run_times;

    for (int run = 0; run < num_runs; ++run) {
        std::fill(results.begin(), results.end(), std::vector<uint32_t>());
        std::vector<double> chunk_search_times(num_chunks, 0.0);

        deglib::concurrent::parallel_for(static_cast<size_t>(0), num_chunks, static_cast<uint32_t>(threads), 1,
            [&](size_t chunk_id, size_t) {
                size_t start = chunk_id * chunk_size;
                size_t end = std::min(start + chunk_size, count);
                size_t num_items = end - start;

                double t_search_start = sisap_common::now_ms();
                for (size_t i = 0; i < num_items; ++i) {
                    size_t q_idx = start + i;

                    const std::byte* query_bytes = reinterpret_cast<const std::byte*>(queries[q_idx].data());
                    deglib::search::ResultSet result_queue = graph.search(
                        {0},
                        query_bytes,
                        eps_search,
                        k_top,
                        nullptr,
                        max_dist);

                    auto& res = results[q_idx];
                    res.reserve(result_queue.size());
                    while (!result_queue.empty()) {
                        const auto cand_idx = result_queue.top().getInternalIndex();
                        uint32_t external_label = graph.getExternalLabel(cand_idx) + 1;
                        res.push_back(external_label);
                        result_queue.pop();
                    }
                }
                chunk_search_times[chunk_id] = sisap_common::now_ms() - t_search_start;
            });

        double sum_search_ms = std::accumulate(chunk_search_times.begin(), chunk_search_times.end(), 0.0) / static_cast<double>(threads);
        run_times.push_back(sum_search_ms);
    }

    double avg_ms = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();

    float recall = -1.0f;
    if (compute_recall) {
        recall = sisap_common::compute_recall(gt_data, results, k_top);
    } else {
        sisap_common::ivecs_write(output_path, results);
    }

    return { avg_ms, recall };
}

static int run(
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t build_threads,
    bool use_flas,
    FlasMetric flas_metric,
    float flas_radius_decay,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    deglib::builder::OptimizationTarget opt_target,
    uint64_t opt_iterations,
    uint32_t prune_worst,
    uint32_t k_top,
    int num_runs,
    const std::vector<uint32_t>& max_dist_list,
    const std::vector<float>& eps_search_list,
    bool compute_recall,
    float goal_recall,
    const std::string& output_path = "",
    const std::string& graph_path = "")
{
    double opt_ms = 0.0;
    const std::string h5path = data_path.string();
    std::printf("\n");

    auto datasets = hdf5_reader::scan_datasets(h5path);
    auto& train_info = hdf5_reader::find_dataset(datasets, "train");

    const char* query_name = "test/queries";
    const char* knn_name = "test/knns";

    const hdf5_reader::DatasetInfo* query_info_ptr = nullptr;
    const hdf5_reader::DatasetInfo* gt_info_ptr = nullptr;
    size_t query_count = 0;

    if (query_name) {
        query_info_ptr = &hdf5_reader::find_dataset(datasets, query_name);
        query_count = static_cast<size_t>(query_info_ptr->num_rows);
    }

    double t_load_start = sisap_common::now_ms();
    size_t dims = static_cast<size_t>(train_info.num_cols);
    size_t count = static_cast<size_t>(train_info.num_rows);

    std::printf("Dataset: train = %zu vectors, queries = %zu (%s), dims = %zu\n",
                count, query_count, query_name ? query_name : "none", dims);

    // --------------------------------------------------------------------------
    // Load ground truth
    // --------------------------------------------------------------------------
    std::vector<std::vector<int32_t>> gt_data;
    if (compute_recall) {
        if (knn_name) {
            gt_info_ptr = &hdf5_reader::find_dataset(datasets, knn_name);
            auto gt_matrix = hdf5_reader::read_matrix_int64(h5path, *gt_info_ptr);

            if (gt_matrix.size() != query_count) {
                std::fprintf(stderr,
                    "Error: queries (%zu) and %s (%zu) must have the same number of rows\n",
                    query_count, knn_name, gt_matrix.size());
                return 1;
            }

            gt_data.resize(gt_matrix.size());
            for (size_t i = 0; i < gt_data.size(); ++i) {
                const auto& row = gt_matrix[i];
                size_t n = std::min(static_cast<size_t>(k_top), row.size());
                for (size_t j = 0; j < n; ++j) {
                    gt_data[i].push_back(static_cast<int32_t>(row[j]));
                }
            }
        } else {
            std::fprintf(stderr, "No ground truth dataset found for recall computation.\n");
            return 1;
        }
    }
    double load_ms = sisap_common::now_ms() - t_load_start;

    std::printf("=== L2-structured FP32 Build, FP16 IP Search - Task 2 Mode 5 (opt_target=%s) ===\n", sisap_common::opt_target_str(opt_target));

    // --------------------------------------------------------------------------
    // Load ALL FP32 training vectors once
    // --------------------------------------------------------------------------
    double t_load_fp32 = sisap_common::now_ms();
    std::vector<float> database_fp32 = hdf5_reader::read_flat_fp32(h5path, train_info);
    double load_fp32_ms = sisap_common::now_ms() - t_load_fp32;
    load_ms += load_fp32_ms;

    // --------------------------------------------------------------------------
    // Perform (d+1)-dimensional L2 transformation on database for building
    // --------------------------------------------------------------------------
    double t_transform_start = sisap_common::now_ms();
    
    // 1. Compute squared norms and find max
    double max_norm_sq = 0.0;
    std::vector<double> norms_sq(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < dims; ++j) {
            float val = database_fp32[i * dims + j];
            sum += double(val) * double(val);
        }
        norms_sq[i] = sum;
        if (sum > max_norm_sq) {
            max_norm_sq = sum;
        }
    }

    std::printf("Max norm squared M^2 = %.6f (M = %.6f)\n", max_norm_sq, std::sqrt(max_norm_sq));

    // 2. Append extra dimension
    size_t new_dims = dims + 1;
    std::vector<float> database_transformed(count * new_dims);
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(&database_transformed[i * new_dims], &database_fp32[i * dims], dims * sizeof(float));
        double diff = max_norm_sq - norms_sq[i];
        float extra = (diff > 0.0) ? static_cast<float>(std::sqrt(diff)) : 0.0f;
        database_transformed[i * new_dims + dims] = extra;
    }

    double transform_ms = sisap_common::now_ms() - t_transform_start;
    std::printf("Transformed database to %zu dimensions for graph building in %.2f ms\n", new_dims, transform_ms);

    // --------------------------------------------------------------------------
    // Optional FLAS pre-sort (performed on transformed vectors)
    // --------------------------------------------------------------------------
    std::vector<uint32_t> sorted_indices;
    double flas_ms = 0.0;
    if (use_flas) {
        sorted_indices = sisap_common::run_flas_presort(database_transformed.data(), count, new_dims, flas_metric, flas_ms, flas_radius_decay);
        if (sorted_indices.empty()) return 1;
    }

    // --------------------------------------------------------------------------
    // Build/Load graph with Metric::L2 (dimension d+1)
    // --------------------------------------------------------------------------
    deglib::FloatSpace feature_space(static_cast<uint32_t>(new_dims), deglib::Metric::L2);
    std::unique_ptr<deglib::graph::SizeBoundedGraph> graph_ptr;
    bool loaded = false;
    double build_ms = 0.0;

    if (!graph_path.empty() && std::filesystem::exists(graph_path)) {
        double t_load_graph_start = sisap_common::now_ms();
        std::printf("Loading existing graph from %s...\n", graph_path.c_str());
        auto g = deglib::graph::load_sizebounded_graph(graph_path.c_str());
        const auto& fs = g.getFeatureSpace();
        const uint32_t graph_size = g.size();

        if (fs.metric() == feature_space.metric() && fs.dim() == new_dims && graph_size == count) {
            graph_ptr = std::make_unique<deglib::graph::SizeBoundedGraph>(std::move(g));
            loaded = true;
            double load_graph_ms = sisap_common::now_ms() - t_load_graph_start;
            std::printf("Graph loaded successfully in %.2f ms (%u vertices)\n", load_graph_ms, graph_size);
        } else {
            std::fprintf(stderr, "Warning: Saved graph properties do not match dataset: metric=%d vs %d, dim=%u vs %zu, size=%u vs %zu. Rebuilding.\n",
                         (int)fs.metric(), (int)feature_space.metric(), (unsigned)fs.dim(), (unsigned)new_dims, graph_size, count);
        }
    }

    if (!loaded) {
        if (use_flas)
            std::printf("Building graph (FLAS order): k_graph=%u, k_ext=%u, eps_ext=%.3f, threads=%u, opt_target=%s\n", k_graph, k_ext, eps_ext, build_threads, sisap_common::opt_target_str(opt_target));
        else
            std::printf("Building graph: k_graph=%u, k_ext=%u, eps_ext=%.3f, threads=%u, opt_target=%s\n", k_graph, k_ext, eps_ext, build_threads, sisap_common::opt_target_str(opt_target));

        graph_ptr = std::make_unique<deglib::graph::SizeBoundedGraph>(static_cast<uint32_t>(count), k_graph, feature_space);
        deglib::graph::SizeBoundedGraph& graph = *graph_ptr;

        std::mt19937 rnd(42);
        deglib::builder::EvenRegularGraphBuilder builder(
            graph, rnd,
            opt_target,
            k_ext, eps_ext,
            0, 0.0f,
            5,
            0, 0,
            true,
            false,
            false
        );
        builder.setThreadCount(static_cast<uint32_t>(build_threads));
        builder.setBatchSize(64, 128);

        const size_t load_chunk_size = 20000;
        double t_build_start = sisap_common::now_ms();

        for (size_t start_row = 0; start_row < count; start_row += load_chunk_size) {
            size_t current_chunk_size = std::min(load_chunk_size, count - start_row);

            size_t bytes_per_vector = new_dims * sizeof(float);
            for (size_t j = 0; j < current_chunk_size; ++j) {
                size_t idx = use_flas ? sorted_indices[start_row + j] : (start_row + j);
                std::vector<std::byte> feature(bytes_per_vector);
                std::memcpy(feature.data(), &database_transformed[idx * new_dims], bytes_per_vector);
                builder.addEntry(static_cast<uint32_t>(idx), std::move(feature));
            }

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

        if (opt_iterations > 0) {
            double t_opt_start = sisap_common::now_ms();
            std::printf("Optimizing graph: k_opt=%u, eps_opt=%.4f, i_opt=%u, iterations=%zu\n", k_ext, 0.01f, 5, (size_t)opt_iterations);
            sisap_common::optimize_graph(graph, k_ext, 0.01f, 5, opt_iterations, 10000);
            opt_ms = sisap_common::now_ms() - t_opt_start;
        }

        if (!graph_path.empty()) {
            std::printf("Saving built graph to %s...\n", graph_path.c_str());
            graph.saveGraph(graph_path.c_str());
        }
    }

    deglib::graph::SizeBoundedGraph& graph = *graph_ptr;

    // Prune worst neighbors (prune_worst=0 = no pruning, default)
    double prune_ms = sisap_common::prune_worst_neighbors(graph, prune_worst, static_cast<uint32_t>(build_threads));

    // --------------------------------------------------------------------------
    // Convert original database features (d dims) to FP16 and build ReadOnlyGraph
    // --------------------------------------------------------------------------
    double t_convert_start = sisap_common::now_ms();
    std::printf("Converting original features (dims=%zu) to FP16...\n", dims);

    std::vector<uint16_t> database_fp16(count * dims);
    for (size_t i = 0; i < count; ++i) {
        std::vector<float> fp32_vec(dims);
        std::memcpy(fp32_vec.data(), &database_fp32[i * dims], dims * sizeof(float));
        std::vector<uint16_t> fp16_vec = floats_to_fp16(fp32_vec);
        std::memcpy(&database_fp16[i * dims], fp16_vec.data(), dims * sizeof(uint16_t));
    }

    std::printf("Building ReadOnlyGraph with FP16InnerProduct features...\n");
    deglib::FloatSpace fp16_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
    deglib::graph::ReadOnlyGraph fp16_graph(fp16_space, graph, database_fp16.data());
    
    // Free the original graph, temporary buffers and transformed vectors
    graph_ptr.reset();
    database_fp32.clear();
    database_fp32.shrink_to_fit();
    database_fp16.clear();
    database_fp16.shrink_to_fit();
    database_transformed.clear();
    database_transformed.shrink_to_fit();

    double convert_ms = sisap_common::now_ms() - t_convert_start;
    std::printf("Converted database and swapped features in %.2f ms\n", convert_ms);

    // --------------------------------------------------------------------------
    // Load query vectors and convert to FP16 (d dims)
    // --------------------------------------------------------------------------
    std::vector<std::vector<uint16_t>> queries;
    if (!query_info_ptr) {
        std::fprintf(stderr, "Error: No query dataset found in HDF5 file.\n");
        return 1;
    }
    double t_query_load = sisap_common::now_ms();
    auto queries_fp32 = hdf5_reader::read_matrix_fp32(h5path, *query_info_ptr);
    double query_load_ms = sisap_common::now_ms() - t_query_load;
    load_ms += query_load_ms;

    double t_query_convert = sisap_common::now_ms();
    queries.reserve(queries_fp32.size());
    for (const auto& q : queries_fp32) {
        queries.push_back(floats_to_fp16(q));
    }
    double query_convert_ms = sisap_common::now_ms() - t_query_convert;
    std::printf("Loaded %zu queries (%.2f ms) and converted to FP16 (%.2f ms, dims=%zu)\n", queries.size(), query_load_ms, query_convert_ms, dims);

    // --------------------------------------------------------------------------
    // Exploration with parameter sweep: eps_search × max_dist
    // --------------------------------------------------------------------------
    std::printf("Starting exploration: k_top=%u, threads=%u\n", k_top, threads);

    float best_recall = -1.0f;
    float best_eps_search = 0.1f;
    uint32_t best_max_dist = 200;
    ExplorationTimings best_timings;

    for (float eps_search : eps_search_list) {
        std::printf("\n  --- eps_search=%.3f ---\n", eps_search);
        
        for (uint32_t max_dist_val : max_dist_list) {
            auto timings = run_search(fp16_graph, queries, k_top, eps_search, max_dist_val, static_cast<uint8_t>(threads),
                                           compute_recall, num_runs, gt_data, output_path);
            timings.search_ms += query_convert_ms;

            std::printf("    max_dist=%u has recall %.2f %% and search time %.1f ms\n",
                        max_dist_val, timings.recall * 100.0f, timings.search_ms);

            bool is_better = false;
            if (timings.recall >= goal_recall) {
                is_better = (best_recall < goal_recall) || (timings.search_ms < best_timings.search_ms);
            } else {
                is_better = (best_recall < goal_recall) && (timings.recall > best_recall);
            }

            if (is_better) {
                best_recall = timings.recall;
                best_eps_search = eps_search;
                best_max_dist = max_dist_val;
                best_timings = timings;
            }
        }
    }

    std::printf("\n");
    double total_time_ms = load_ms + build_ms + transform_ms + opt_ms + convert_ms + best_timings.search_ms;

    sisap_common::print_summary(
        (use_flas ? "L2-structured Build, FP16 IP Search (FLAS)" : "L2-structured Build, FP16 IP Search"), 5,
        load_ms, transform_ms, build_ms, convert_ms, prune_ms,
        best_timings.search_ms, 0.0, total_time_ms,
        compute_recall, k_top, best_timings.recall,
        threads, best_max_dist, 0,
        k_graph, k_ext, eps_ext, 0, count, dims, 0, opt_target,
        flas_ms, opt_ms
    );

    return 0;
}

} // namespace task2::mode_l2_fp16_ip
