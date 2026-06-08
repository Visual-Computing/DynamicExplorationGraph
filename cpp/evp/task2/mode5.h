#pragma once

/**
 * @file mode5.h
 * @brief Task 2 Mode 5: FP32 Build (L2-converted) + FP16 Inner Product Search
 *
 * Building graph: k_graph=30, k_ext=60, eps_ext=0.001, threads=1, opt_target=LowLID
 *  --- eps_search=0.150 ---
 *   max_dist=50000 has recall 77.25 % and search time 39.5 ms
 * --- eps_search=0.180 ---
 *   max_dist=50000 has recall 83.93 % and search time 56.6 ms
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

#include "../evp_common.h"
#include "../hdf5_reader.h"
#include "../flas_common.h"

namespace task2::mode_l2_fp16_ip {

struct ExplorationTimings {
    double search_ms = 0.0;
    float recall = -1.0f;
};

static std::vector<uint16_t> floats_to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    size_t i = 0;
    for (; i + 4 <= v.size(); i += 4) {
        __m128 f4 = _mm_loadu_ps(&v[i]);
        __m128i h4 = _mm_cvtps_ph(f4, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h4);
        out[i]   = tmp[0];
        out[i+1] = tmp[1];
        out[i+2] = tmp[2];
        out[i+3] = tmp[3];
    }
    for (; i < v.size(); ++i) {
        __m128 f1 = _mm_set_ss(v[i]);
        __m128i h1 = _mm_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h1);
        out[i] = tmp[0];
    }
#else
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        uint16_t sign     = static_cast<uint16_t>((bits >> 16) & 0x8000u);
        int32_t  exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFFu;
        if (exponent <= 0)      { out[i] = sign; }
        else if (exponent >= 31){ out[i] = static_cast<uint16_t>(sign | 0x7C00u); }
        else                    { out[i] = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13)); }
    }
#endif
    return out;
}

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

                double t_search_start = evp_common::now_ms();
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
                chunk_search_times[chunk_id] = evp_common::now_ms() - t_search_start;
            });

        double sum_search_ms = std::accumulate(chunk_search_times.begin(), chunk_search_times.end(), 0.0) / static_cast<double>(threads);
        run_times.push_back(sum_search_ms);
    }

    double avg_ms = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();

    float recall = -1.0f;
    if (compute_recall) {
        recall = evp_common::compute_recall(gt_data, results, k_top);
    } else {
        evp_common::ivecs_write(output_path, results);
    }

    return { avg_ms, recall };
}

static int run(
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t build_threads,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top,
    const std::vector<uint32_t>& max_dist_list,
    bool compute_recall,
    int num_runs = 1,
    const std::string& output_path = "",
    const std::string& graph_path = "",
    uint32_t prune_worst = 0,
    const std::vector<float>& eps_search_list = {0.1f},
    deglib::builder::OptimizationTarget opt_target = deglib::builder::OptimizationTarget::LowLID,
    bool use_flas = false,
    FlasMetric flas_metric = FlasMetric::L2,
    float flas_radius_decay = 0.93f)
{
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

    double t_load_start = evp_common::now_ms();
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
    double load_ms = evp_common::now_ms() - t_load_start;

    std::printf("=== L2-structured FP32 Build, FP16 IP Search - Task 2 Mode 5 (opt_target=%s) ===\n", evp_common::opt_target_str(opt_target));

    // --------------------------------------------------------------------------
    // Load ALL FP32 training vectors once
    // --------------------------------------------------------------------------
    double t_load_fp32 = evp_common::now_ms();
    std::vector<float> database_fp32 = hdf5_reader::read_flat_fp32(h5path, train_info);
    double load_fp32_ms = evp_common::now_ms() - t_load_fp32;
    load_ms += load_fp32_ms;

    // --------------------------------------------------------------------------
    // Perform (d+1)-dimensional L2 transformation on database for building
    // --------------------------------------------------------------------------
    double t_transform_start = evp_common::now_ms();
    
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

    double transform_ms = evp_common::now_ms() - t_transform_start;
    std::printf("Transformed database to %zu dimensions for graph building in %.2f ms\n", new_dims, transform_ms);

    // --------------------------------------------------------------------------
    // Optional FLAS pre-sort (performed on transformed vectors)
    // --------------------------------------------------------------------------
    std::vector<uint32_t> sorted_indices;
    double flas_ms = 0.0;
    if (use_flas) {
        sorted_indices = flas_common::run_flas_presort(database_transformed.data(), count, new_dims, flas_metric, flas_ms, flas_radius_decay);
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
        double t_load_graph_start = evp_common::now_ms();
        std::printf("Loading existing graph from %s...\n", graph_path.c_str());
        auto g = deglib::graph::load_sizebounded_graph(graph_path.c_str());
        const auto& fs = g.getFeatureSpace();
        const uint32_t graph_size = g.size();

        if (fs.metric() == feature_space.metric() && fs.dim() == new_dims && graph_size == count) {
            graph_ptr = std::make_unique<deglib::graph::SizeBoundedGraph>(std::move(g));
            loaded = true;
            double load_graph_ms = evp_common::now_ms() - t_load_graph_start;
            std::printf("Graph loaded successfully in %.2f ms (%u vertices)\n", load_graph_ms, graph_size);
        } else {
            std::fprintf(stderr, "Warning: Saved graph properties do not match dataset: metric=%d vs %d, dim=%u vs %zu, size=%u vs %zu. Rebuilding.\n",
                         (int)fs.metric(), (int)feature_space.metric(), (unsigned)fs.dim(), (unsigned)new_dims, graph_size, count);
        }
    }

    if (!loaded) {
        if (use_flas)
            std::printf("Building graph (FLAS order): k_graph=%u, k_ext=%u, eps_ext=%.3f, threads=%u, opt_target=%s\n", k_graph, k_ext, eps_ext, build_threads, evp_common::opt_target_str(opt_target));
        else
            std::printf("Building graph: k_graph=%u, k_ext=%u, eps_ext=%.3f, threads=%u, opt_target=%s\n", k_graph, k_ext, eps_ext, build_threads, evp_common::opt_target_str(opt_target));

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
        double t_build_start = evp_common::now_ms();

        for (size_t start_row = 0; start_row < count; start_row += load_chunk_size) {
            size_t current_chunk_size = std::min(load_chunk_size, count - start_row);

            size_t bytes_per_vector = new_dims * sizeof(float);
            for (size_t j = 0; j < current_chunk_size; ++j) {
                size_t idx = use_flas ? sorted_indices[start_row + j] : (start_row + j);
                std::vector<std::byte> feature(bytes_per_vector);
                std::memcpy(feature.data(), &database_transformed[idx * new_dims], bytes_per_vector);
                builder.addEntry(static_cast<uint32_t>(idx), std::move(feature));
            }

            double t_chunk_build = evp_common::now_ms();
            auto dummy_callback = [](deglib::builder::BuilderStatus&) {};
            builder.build(dummy_callback, false);
            double chunk_build_ms = evp_common::now_ms() - t_chunk_build;
            build_ms += chunk_build_ms;

            double elapsed_s = (evp_common::now_ms() - t_build_start) / 1000.0;
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

    // Prune worst neighbors (prune_worst=0 = no pruning, default)
    double prune_ms = evp_common::prune_worst_neighbors(graph, prune_worst, static_cast<uint32_t>(build_threads));

    // --------------------------------------------------------------------------
    // Convert original database features (d dims) to FP16 and build ReadOnlyGraph
    // --------------------------------------------------------------------------
    double t_convert_start = evp_common::now_ms();
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

    double convert_ms = evp_common::now_ms() - t_convert_start;
    std::printf("Converted database and swapped features in %.2f ms\n", convert_ms);

    // --------------------------------------------------------------------------
    // Load query vectors and convert to FP16 (d dims)
    // --------------------------------------------------------------------------
    std::vector<std::vector<uint16_t>> queries;
    if (!query_info_ptr) {
        std::fprintf(stderr, "Error: No query dataset found in HDF5 file.\n");
        return 1;
    }
    double t_query_load = evp_common::now_ms();
    auto queries_fp32 = hdf5_reader::read_matrix_fp32(h5path, *query_info_ptr);
    queries.reserve(queries_fp32.size());
    for (const auto& q : queries_fp32) {
        queries.push_back(floats_to_fp16(q));
    }
    double query_load_ms = evp_common::now_ms() - t_query_load;
    load_ms += query_load_ms;
    std::printf("Loaded %zu queries and converted to FP16 (dims=%zu)\n", queries.size(), dims);

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

            std::printf("    max_dist=%u has recall %.2f %% and search time %.1f ms\n",
                        max_dist_val, timings.recall * 100.0f, timings.search_ms);

            if (timings.recall > best_recall) {
                best_recall = timings.recall;
                best_eps_search = eps_search;
                best_max_dist = max_dist_val;
                best_timings = timings;
            }
        }
    }

    std::printf("\n");
    double total_time_ms = load_ms + build_ms + transform_ms + convert_ms + best_timings.search_ms;

    evp_common::print_summary(
        (use_flas ? "L2-structured Build, FP16 IP Search (FLAS)" : "L2-structured Build, FP16 IP Search"), 5,
        load_ms, transform_ms, build_ms, convert_ms, prune_ms,
        best_timings.search_ms, flas_ms, total_time_ms,
        compute_recall, k_top, best_timings.recall,
        threads, best_max_dist, 0,
        k_graph, k_ext, eps_ext, 0, count, dims, 0, opt_target
    );

    return 0;
}

} // namespace task2::mode_l2_fp16_ip
