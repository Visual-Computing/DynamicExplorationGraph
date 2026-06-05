#pragma once

/**
 * @file mode4.h
 * @brief Task 2 Mode 4: FP32 Build + Asymmetric FP16×EVP Search
 *
 * Behavior:
 * 1. Loads FP32 training vectors from "train" as the database.
 * 2. Builds a SizeBoundedGraph using Metric::InnerProduct with FloatSpace(dims, InnerProduct).
 * 3. Quantizes all database vectors to EVP bits.
 * 4. Creates a ReadOnlyGraph using Metric::FP16EvpAsymmetric with the EVP-quantized features.
 * 5. Loads FP32 query vectors, converts them to FP16.
 * 6. For each query, uses the asymmetric search (FP16 query × EVP database).
 * 7. Tracks build time, quantization time, and search time separately.
 * 8. Requires --non-zeros parameter for EVP quantization sparsity.
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
#include "../flas_common.h"

namespace task2::mode_evp_asym_search {

// ---------------------------------------------------------------------------
// FP32 → FP16 conversion (same as mode2/mode3)
// ---------------------------------------------------------------------------
static std::vector<uint16_t> floats_to_fp16(const std::vector<float>& v, size_t dims) {
    std::vector<uint16_t> out(dims);
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    size_t i = 0;
    for (; i + 4 <= dims; i += 4) {
        __m128 f4 = _mm_loadu_ps(&v[i]);
        __m128i h4 = _mm_cvtps_ph(f4, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h4);
        out[i]   = tmp[0];
        out[i+1] = tmp[1];
        out[i+2] = tmp[2];
        out[i+3] = tmp[3];
    }
    for (; i < dims; ++i) {
        __m128 f1 = _mm_set_ss(v[i]);
        __m128i h1 = _mm_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h1);
        out[i] = tmp[0];
    }
#else
    for (size_t i = 0; i < dims; ++i) {
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

struct ExplorationTimings {
    double search_ms = 0.0;
    float recall = -1.0f;
};

static ExplorationTimings run_search(
    const deglib::graph::ReadOnlyGraph& graph,
    const std::vector<std::vector<uint16_t>>& fp16_queries,
    uint32_t k_top,
    float eps_search,
    uint32_t max_dist,
    uint8_t threads,
    bool compute_recall,
    int num_runs = 1,
    const std::vector<std::vector<int32_t>>& gt_data = {},
    const std::string& output_path = "")
{
    size_t count = fp16_queries.size();
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

                    // FP16 query bytes — the comparator reads uint16_t*
                    const std::byte* query_bytes = reinterpret_cast<const std::byte*>(fp16_queries[q_idx].data());
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
    uint32_t non_zeros = 64,
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

    // Validate EVP requirements
    if (dims % 8 != 0) {
        std::fprintf(stderr, "Error: EVP quantization requires dim %% 8 == 0, but dim=%zu\n", dims);
        return 1;
    }
    if (non_zeros >= dims) {
        std::fprintf(stderr, "Error: --non-zeros (%u) must be < dim (%zu)\n", non_zeros, dims);
        return 1;
    }

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

    std::printf("=== FP32 Build, Asymmetric FP16×EVP Search - Task 2 Mode 4 (opt_target=%s, non_zeros=%u) ===\n",
                evp_common::opt_target_str(opt_target), non_zeros);

    // --------------------------------------------------------------------------
    // Load ALL FP32 training vectors once — used for graph building AND EVP quantization
    // --------------------------------------------------------------------------
    double t_load_fp32 = evp_common::now_ms();
    std::vector<float> database_fp32 = hdf5_reader::read_flat_fp32(h5path, train_info);
    double load_fp32_ms = evp_common::now_ms() - t_load_fp32;
    load_ms += load_fp32_ms;

    // --------------------------------------------------------------------------
    // Optional FLAS pre-sort
    // --------------------------------------------------------------------------
    std::vector<uint32_t> sorted_indices;
    double flas_ms = 0.0;
    if (use_flas) {
        sorted_indices = flas_common::run_flas_presort(database_fp32.data(), count, dims, flas_metric, flas_ms, flas_radius_decay);
        if (sorted_indices.empty()) return 1;
    }

    // --------------------------------------------------------------------------
    // Build/Load graph with Metric::InnerProduct (FP32)
    // --------------------------------------------------------------------------
    deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::InnerProduct);
    std::unique_ptr<deglib::graph::SizeBoundedGraph> graph_ptr;
    bool loaded = false;
    double build_ms = 0.0;

    if (!graph_path.empty() && std::filesystem::exists(graph_path)) {
        double t_load_graph_start = evp_common::now_ms();
        std::printf("Loading existing graph from %s...\n", graph_path.c_str());
        auto g = deglib::graph::load_sizebounded_graph(graph_path.c_str());
        const auto& fs = g.getFeatureSpace();
        const uint32_t graph_size = g.size();

        if (fs.metric() == feature_space.metric() && fs.dim() == dims && graph_size == count) {
            graph_ptr = std::make_unique<deglib::graph::SizeBoundedGraph>(std::move(g));
            loaded = true;
            double load_graph_ms = evp_common::now_ms() - t_load_graph_start;
            std::printf("Graph loaded successfully in %.2f ms (%u vertices)\n", load_graph_ms, graph_size);
        } else {
            std::fprintf(stderr, "Warning: Saved graph properties do not match dataset: metric=%d vs %d, dim=%u vs %zu, size=%u vs %zu. Rebuilding.\n",
                         (int)fs.metric(), (int)feature_space.metric(), (unsigned)fs.dim(), dims, graph_size, count);
        }
    }

    if (!loaded) {
        if (use_flas)
            std::printf("Building graph (FLAS order): k_graph=%u, k_ext=%u, eps_ext=%.3f, threads=%u, opt_target=%s\n",
                        k_graph, k_ext, eps_ext, build_threads, evp_common::opt_target_str(opt_target));
        else
            std::printf("Building graph: k_graph=%u, k_ext=%u, eps_ext=%.3f, threads=%u, opt_target=%s\n",
                        k_graph, k_ext, eps_ext, build_threads, evp_common::opt_target_str(opt_target));

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

            size_t bytes_per_vector = dims * sizeof(float);
            for (size_t j = 0; j < current_chunk_size; ++j) {
                size_t idx = use_flas ? sorted_indices[start_row + j] : (start_row + j);
                std::vector<std::byte> feature(bytes_per_vector);
                std::memcpy(feature.data(), &database_fp32[idx * dims], bytes_per_vector);
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

    double prune_ms = evp_common::prune_worst_neighbors(graph, prune_worst, static_cast<uint32_t>(build_threads));

    // --------------------------------------------------------------------------
    // Quantize training features to EVP bits and build ReadOnlyGraph
    // --------------------------------------------------------------------------
    double t_quantize_start = evp_common::now_ms();
    std::printf("Quantizing features to EVP bits (non_zeros=%u)...\n", non_zeros);

    auto quantized_evp = deglib::quantization::quantize_batch(
        database_fp32.data(), count, static_cast<uint32_t>(dims), non_zeros, threads);

    std::printf("Building ReadOnlyGraph with FP16EvpAsymmetric features...\n");
    deglib::FloatSpace asymmetric_space(static_cast<uint32_t>(dims), deglib::Metric::FP16EvpAsymmetric);
    deglib::graph::ReadOnlyGraph evp_graph(asymmetric_space, graph, quantized_evp.data());

    graph_ptr.reset();

    double quantize_ms = evp_common::now_ms() - t_quantize_start;
    std::printf("Quantized database and swapped features in %.2f ms\n", quantize_ms);

    quantized_evp.clear();
    quantized_evp.shrink_to_fit();

    // --------------------------------------------------------------------------
    // Load query vectors and convert to FP16
    // --------------------------------------------------------------------------
    if (!query_info_ptr) {
        std::fprintf(stderr, "Error: No query dataset found in HDF5 file.\n");
        return 1;
    }
    double t_query_load = evp_common::now_ms();
    auto queries_fp32 = hdf5_reader::read_matrix_fp32(h5path, *query_info_ptr);
    std::vector<std::vector<uint16_t>> fp16_queries;
    fp16_queries.reserve(queries_fp32.size());
    for (size_t i = 0; i < queries_fp32.size(); ++i) {
        fp16_queries.push_back(floats_to_fp16(queries_fp32[i], dims));
    }
    double query_load_ms = evp_common::now_ms() - t_query_load;
    load_ms += query_load_ms;
    std::printf("Loaded %zu queries and converted to FP16 (dims=%zu)\n", fp16_queries.size(), dims);

    // --------------------------------------------------------------------------
    // Diagnostic: compare asymmetric EVP distance vs FP32 IP for query 0
    // --------------------------------------------------------------------------
    if (compute_recall && query_count > 0 && !fp16_queries.empty() && !gt_data.empty()) {
        const size_t dq = 0;
        const auto& q_fp16 = fp16_queries[dq];
        const auto& gt_row = gt_data[dq];
        int32_t raw_gt0 = gt_row.size() > 0 ? gt_row[0] : -1;
        uint32_t gt_label = (raw_gt0 > 0) ? static_cast<uint32_t>(raw_gt0) - 1 : 0;
        uint32_t last_label = static_cast<uint32_t>(count - 1);

        float d_gt = 0.0f, d_rand = 0.0f;
        try {
            uint32_t gt_internal = evp_graph.getInternalIndex(gt_label);
            uint32_t rand_internal = evp_graph.getInternalIndex(last_label);
            d_gt = deglib::distances::FP16EvpAsymInnerProduct::compare(
                q_fp16.data(), evp_graph.getFeatureVector(gt_internal), &dims);
            d_rand = deglib::distances::FP16EvpAsymInnerProduct::compare(
                q_fp16.data(), evp_graph.getFeatureVector(rand_internal), &dims);
        } catch (...) {
            std::fprintf(stderr, "  [DIAG] Error accessing graph features\n");
        }

        float ip_gt = 0.0f, ip_rand = 0.0f;
        for (size_t k = 0; k < dims; ++k) {
            ip_gt   += queries_fp32[dq][k] * database_fp32[gt_label * dims + k];
            ip_rand += queries_fp32[dq][k] * database_fp32[last_label * dims + k];
        }
        std::printf("  [DIAG] Query 0: asym_dist(gt0)=%.4f asym_dist(rand)=%.4f (lower=better) | FP32_IP(gt0)=%.2f FP32_IP(rand)=%.2f (higher=better)\n",
                    d_gt, d_rand, ip_gt, ip_rand);
    }

    // Free FP32 data — no longer needed
    database_fp32.clear();
    database_fp32.shrink_to_fit();
    queries_fp32.clear();
    queries_fp32.shrink_to_fit();

    // --------------------------------------------------------------------------
    // Exploration with parameter sweep: eps_search × max_dist
    // --------------------------------------------------------------------------
    std::printf("Starting exploration: k_top=%u, threads=%u\n", k_top, threads);

    float best_recall = -1.0f;
    float best_eps_search = 0.1f;
    uint32_t best_max_dist = 200;
    ExplorationTimings best_timings;

    for (float eps_search : eps_search_list) {
        std::printf("\n  --- eps_search=%.2f ---\n", eps_search);
        
        for (uint32_t max_dist_val : max_dist_list) {
            auto timings = run_search(evp_graph, fp16_queries, k_top, eps_search, max_dist_val, static_cast<uint8_t>(threads),
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
    double total_time_ms = load_ms + build_ms + quantize_ms + best_timings.search_ms;

    evp_common::print_summary(
        (use_flas ? "FP32 Build, Asymmetric FP16×EVP Search (FLAS)" : "FP32 Build, Asymmetric FP16×EVP Search"), 4,
        load_ms, 0.0, build_ms, quantize_ms, prune_ms,
        best_timings.search_ms, flas_ms, total_time_ms,
        compute_recall, k_top, best_timings.recall,
        threads, best_max_dist, 0,
        k_graph, k_ext, eps_ext, non_zeros, count, dims, 0, opt_target
    );

    return 0;
}

} // namespace task2::mode_evp_asym_search
