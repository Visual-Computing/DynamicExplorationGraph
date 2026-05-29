#pragma once

#include <chrono>
#include <cstdint>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "hdf5_reader.h"
#include "graph/sizebounded_graph.h"
#include "concurrent.h"
#include <algorithm>

namespace evp_common {

inline size_t calc_batch_size(size_t count, uint8_t threads) {
    return std::max(static_cast<size_t>(1), count / (static_cast<size_t>(threads) * 100));
}

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
           .count();
}

inline void prune_worst_neighbors(deglib::graph::SizeBoundedGraph& graph, uint32_t prune_worst, uint32_t threads) {
    if (prune_worst == 0) {
        return;
    }

    std::printf("Pruning %u worst neighbors per vertex and replacing them with self-loops using %u threads...\n", prune_worst, threads);
    double t_prune_start = now_ms();
    const size_t num_vertices = graph.size();
    const uint32_t k = graph.getEdgesPerVertex();

    const size_t chunk_size = 8192;
    const size_t num_chunks = (num_vertices + chunk_size - 1) / chunk_size;

    deglib::concurrent::parallel_for(static_cast<size_t>(0), num_chunks, threads, 1,
        [&](size_t chunk_id, size_t) {
            size_t start = chunk_id * chunk_size;
            size_t end = std::min(start + chunk_size, num_vertices);

            struct NeighborInfo {
                uint32_t index;
                float weight;
            };
            std::vector<NeighborInfo> neighbors;
            neighbors.reserve(k);

            std::vector<uint32_t> new_indices(k);
            std::vector<float> new_weights(k);

            for (size_t u = start; u < end; ++u) {
                const uint32_t* neighbor_indices = graph.getNeighborIndices(static_cast<uint32_t>(u));
                const float* neighbor_weights = graph.getNeighborWeights(static_cast<uint32_t>(u));

                neighbors.clear();
                for (uint32_t i = 0; i < k; ++i) {
                    neighbors.push_back({neighbor_indices[i], neighbor_weights[i]});
                }

                // Sort neighbors in descending order of weights (worst neighbor first)
                std::sort(neighbors.begin(), neighbors.end(), [](const NeighborInfo& a, const NeighborInfo& b) {
                    return a.weight > b.weight;
                });

                // Overwrite index to the vertex itself and weight to 0.f for the prune_worst worst neighbors
                for (uint32_t i = 0; i < prune_worst; ++i) {
                    neighbors[i].index = static_cast<uint32_t>(u);
                    neighbors[i].weight = 0.0f;
                }

                // Re-sort the entire neighborhood by neighbor index (internal index ID)
                std::sort(neighbors.begin(), neighbors.end(), [](const NeighborInfo& a, const NeighborInfo& b) {
                    return a.index < b.index;
                });

                // Write back to graph
                for (uint32_t i = 0; i < k; ++i) {
                    new_indices[i] = neighbors[i].index;
                    new_weights[i] = neighbors[i].weight;
                }

                graph.changeEdges(static_cast<uint32_t>(u), new_indices.data(), new_weights.data());
            }
        });

    double prune_ms = now_ms() - t_prune_start;
    std::printf("Pruning complete in %.2f ms\n", prune_ms);
}

inline std::vector<std::vector<std::byte>> hvecs_read(const char* fname, size_t& d_out, size_t& n_out) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing file %s, size is: %ju message: %s \n", fname, file_size, ec.message().c_str());
        perror("");
        abort();
    }

    // open as binary
    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open()) {
        std::fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }

    // read dimension header
    uint32_t dims = 0;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    assert((dims > 0 && dims < 1'000'000) && "unreasonable dimension");

    // compute number of rows
    size_t row_bytes = sizeof(uint32_t) + dims * sizeof(uint16_t);
    assert(file_size % row_bytes == 0 || !"weird file size");
    size_t n = (size_t)file_size / row_bytes;
    d_out = dims;
    n_out = n;

    std::vector<std::vector<std::byte>> vectors(n);
    for (size_t i = 0; i < n; ++i) {
        vectors[i].resize(dims * sizeof(uint16_t));
        // Seek to the start of this row's features (skip the 4-byte dimension header)
        ifstream.seekg(i * row_bytes + sizeof(uint32_t));
        ifstream.read(reinterpret_cast<char*>(vectors[i].data()), dims * sizeof(uint16_t));
    }

    ifstream.close();
    return vectors;
}

/**
 * @brief Writes a result list to a file in the ivecs binary format.
 *
 * Each row is serialised as:
 *   - 4 bytes : uint32_t  – number of elements in this row (= d)
 *   - d × 4 bytes : uint32_t[] – the element values
 *
 * Each row is written in full (row.size() elements). Callers are expected
 * to have already trimmed the rows to the desired length before calling.
 * If the output path is empty or the file cannot be opened, an error is
 * printed to stderr and the function returns without writing.
 *
 * @param output_path  Destination file path (binary).
 * @param results      2-D result list, one inner vector per query.
 */
inline void ivecs_write(
    const std::string& output_path,
    const std::vector<std::vector<uint32_t>>& results)
{
    if (output_path.empty()) {
        return;
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
        std::fprintf(stderr, "Error: Could not open output file '%s' for writing.\n",
                      output_path.c_str());
        return;
    }

    for (const auto& row : results) {
        const uint32_t d = static_cast<uint32_t>(row.size());
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
        out.write(reinterpret_cast<const char*>(row.data()), d * sizeof(uint32_t));
    }

    std::printf("Successfully wrote %zu result rows (ivecs) to '%s'\n",
                results.size(), output_path.c_str());
}

inline std::vector<std::vector<int32_t>> load_ground_truth(
    const std::string& h5path,
    const std::map<std::string, hdf5_reader::DatasetInfo>& datasets,
    uint32_t k_top)
{
    auto& allknn_info = hdf5_reader::find_dataset(datasets, "allknn/knns");
    std::printf("  allknn/knns: %llu x %llu (elem=%uB)\n",
        (unsigned long long)allknn_info.num_rows,
        (unsigned long long)allknn_info.num_cols,
        allknn_info.element_size);

    auto gt_data = hdf5_reader::read_matrix_int32(h5path, allknn_info);
    for (auto& row : gt_data) {
        size_t n = row.size() > 1 ? std::min(static_cast<size_t>(k_top), row.size() - 1) : 0;
        for (size_t i = 0; i < n; ++i) {
            row[i] = row[i + 1] - 1;
        }
        row.resize(n);
    }
    return gt_data;
}

inline float compute_recall(
    const std::vector<std::vector<int32_t>>& gt_data,
    const std::vector<std::vector<uint32_t>>& results,
    uint32_t k_top)
{
    int total_hits = 0;
    size_t count = gt_data.size();
    if (count != results.size()) {
        std::fprintf(stderr, "Error: gt_data size (%zu) != results size (%zu)\n",
                    count, results.size());
        std::exit(1);
    }
    for (size_t i = 0; i < count; ++i) {
        const auto& gt_row = gt_data[i];
        const auto& row = results[i];
        if (gt_row.size() < k_top) {
            std::fprintf(stderr, "Error: ground truth row %zu has only %zu entries, expected %u\n",
                        i, gt_row.size(), k_top);
            std::exit(1);
        }
        const uint32_t row_len = static_cast<uint32_t>(std::min(row.size(), static_cast<size_t>(k_top)));
        for (uint32_t k = 0; k < k_top; ++k) {
            int32_t gt_idx = gt_row[k];
            for (uint32_t j = 0; j < row_len; ++j) {
                if (static_cast<int32_t>(row[j]) == gt_idx) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    return static_cast<float>(total_hits) / (count * k_top);
}

} // namespace evp_common
