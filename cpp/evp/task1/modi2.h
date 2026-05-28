#pragma once

/**
 * @file modi2.h
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
#include "graph/readonly_graph.h"
#include "graph/readonly_graph_external.h"
#include "quantization/evp_quantize.h"
#include "repository.h"

#include "../evp_common.h"
#include "../hdf5_reader.h"

namespace task1::mode2 {

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
    (void)k_graph;
    (void)k_ext;
    (void)eps_ext;
    (void)max_distance_count;

    const std::string h5path = data_path.string();
    std::printf("HDF5 mode (modi2): scanning '%s'\n", h5path.c_str());

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

    std::printf("=== EVP Bits Linear Search Benchmark (Mode 2) ===\n");
    std::printf("Data path: %ls\n", data_path.wstring().c_str());
    std::printf("NON_ZEROS=%u, K_TOP=%u\n", non_zeros, k_top);
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
    // Linear Search using distances.h
    // --------------------------------------------------------------------------
    double t_search_start = evp_common::now_ms();

    std::vector<std::vector<uint32_t>> results(count, std::vector<uint32_t>(k_top));
    const size_t bytes_per_evp = dims / 4;
    const uint32_t dims_u32 = static_cast<uint32_t>(dims);
    const size_t batch_size = evp_common::calc_batch_size(count, static_cast<uint8_t>(threads));

    deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, batch_size,
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

    double search_ms = evp_common::now_ms() - t_search_start;
    std::printf("Linear Search time: %.2f ms\n", search_ms);

    // Write results to output file if needed
    evp_common::ivecs_write(output_path, results);

    // --------------------------------------------------------------------------
    // Calculate Recall
    // --------------------------------------------------------------------------
    float recall = 0.0f;
    if (compute_recall) {
        recall = evp_common::compute_recall(gt_data, results, k_top);
        std::printf("Recall@%u: %.4f\n", k_top, recall);
    }

    // --------------------------------------------------------------------------
    // Summary
    // --------------------------------------------------------------------------
    std::printf("=============================================\n");
    std::printf("          FINAL SUMMARY (EVP Bits Linear Search - Mode 2)\n");
    std::printf("=============================================\n");
    std::printf("Quantize:                %.2f ms\n", quantize_ms);
    std::printf("Linear Search:           %.2f ms\n", search_ms);
    std::printf("---------------------------------------------\n");
    std::printf("Vectors:                 %zu\n", count);
    std::printf("Dimensions:              %zu\n", dims);
    if (compute_recall) {
        std::printf("Recall@%u:                %.4f\n", k_top, recall);
    }
    std::printf("=============================================\n\n");

    return 0;
}

} // namespace task1::mode2
