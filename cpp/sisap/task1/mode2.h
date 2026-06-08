#pragma once

/**
 * @file mode2.h
 * @brief Benchmark Mode 2: EVP Linear Search (evp-linear-search)
 * 
 * Behavior:
 * 1. Quantizes training features to EVP bits representation.
 * 2. Runs an exact brute-force linear search over all quantized vector pairs.
 * 3. Uses distances comparison via EvpBitsSimilarity::compare.
 * 4. Acts as an exact similarity search baseline (no graph built).
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
#include "quantization/evp_quantize.h"
#include "repository.h"

#include "../sisap_common.h"
#include "../hdf5_reader.h"

namespace task1::mode2 {

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
    const std::string& output_path)
{
    (void)k_graph;
    (void)k_ext;
    (void)eps_ext;
    (void)max_dist_list;

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

    std::printf("=== EVP Linear Search - Mode 2 ===\n");

    // --------------------------------------------------------------------------
    // Quantize
    // --------------------------------------------------------------------------
    double t1 = sisap_common::now_ms();

    auto quantized = deglib::quantization::quantize_batch(
        train_vectors, static_cast<uint32_t>(dims), non_zeros, threads);

    double quantize_ms = sisap_common::now_ms() - t1;

    std::printf("Starting exploration: k_top=%u, prune_worst=0, threads=%u\n", k_top, threads);
    uint32_t best_max_dist = 0;
    float best_recall = -1.0f;
    double best_search_ms = 0.0;

    for (uint32_t max_dist_val : max_dist_list) {
        double t_start = sisap_common::now_ms();

        std::vector<std::vector<uint32_t>> results(count, std::vector<uint32_t>(k_top));
        const size_t bytes_per_evp = dims / 4;
        const uint32_t dims_u32 = static_cast<uint32_t>(dims);

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
                    const std::byte* query_ptr = quantized.data() + label * bytes_per_evp;

                    std::vector<std::pair<float, uint32_t>> top;
                    top.reserve(k_top + 1);

                    for (size_t j = 0; j < count; ++j) {
                        if (label == j) {
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
                        results[label][k] = top[k].second;
                    }
                }
                chunk_search_times[chunk_id] = sisap_common::now_ms() - t_search_start;
            });

        double total_ms = sisap_common::now_ms() - t_start;
        double search_ms = std::accumulate(chunk_search_times.begin(), chunk_search_times.end(), 0.0) / threads;
        // Write results to output file if needed
        sisap_common::ivecs_write(output_path, results);

        // --------------------------------------------------------------------------
        // Calculate Recall
        // --------------------------------------------------------------------------
        float recall = 0.0f;
        if (compute_recall) {
            recall = sisap_common::compute_recall(gt_data, results, k_top);
        }

        std::printf("  max_dist=%u has recall %.2f %% and time %.1f s\n",
                    max_dist_val, recall * 100.0f, search_ms / 1000.0);

        bool is_better = false;
        if (recall >= goal_recall) {
            is_better = (best_recall < goal_recall) || (search_ms < best_search_ms);
        } else {
            is_better = (best_recall < goal_recall) && (recall > best_recall);
        }

        if (is_better) {
            best_recall = recall;
            best_max_dist = max_dist_val;
            best_search_ms = search_ms;
        }
    }

    std::printf("\n");
    double total_time_ms = load_ms + quantize_ms + best_search_ms;

    sisap_common::print_summary(
        "EVP Linear Search", 2,
        load_ms, quantize_ms, 0.0, 0.0, 0.0,
        best_search_ms, 0.0, total_time_ms,
        compute_recall, k_top, best_recall,
        threads, best_max_dist, 0,
        0, 0, 0.0f, non_zeros, count, dims, 0
    );

    return 0;
}

} // namespace task1::mode2
