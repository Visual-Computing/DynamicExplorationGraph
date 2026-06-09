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
#include <algorithm>
#include <cstdio>
#include <random>

#include "hdf5_reader.h"
#include "graph/sizebounded_graph.h"
#include "builder.h"
#include "concurrent.h"
#include "flas/fast_linear_assignment_sorter.hpp"

namespace sisap_common {

inline size_t calc_batch_size(size_t count, uint8_t threads) {
    return std::max(static_cast<size_t>(1), count / (static_cast<size_t>(threads) * 100));
}

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
           .count();
}

inline std::vector<uint32_t> parse_list(const std::string& s) {
    std::vector<uint32_t> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            res.push_back(static_cast<uint32_t>(std::stoul(item)));
        }
    }
    return res;
}

inline std::vector<float> parse_list_f(const std::string& s) {
    std::vector<float> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            res.push_back(std::stof(item));
        }
    }
    return res;
}

inline const char* opt_target_str(deglib::builder::OptimizationTarget t) {
    switch (t) {
        case deglib::builder::OptimizationTarget::LowLID:                  return "LowLID";
        case deglib::builder::OptimizationTarget::HighLID:                 return "HighLID";
        case deglib::builder::OptimizationTarget::StreamingData_SchemeA:   return "StreamingData_SchemeA";
        case deglib::builder::OptimizationTarget::StreamingData_SchemeB:   return "StreamingData_SchemeB";
        case deglib::builder::OptimizationTarget::StreamingData_SchemeC:   return "StreamingData_SchemeC";
        case deglib::builder::OptimizationTarget::StreamingData_SchemeD:   return "StreamingData_SchemeD";
        case deglib::builder::OptimizationTarget::SchemeA:                 return "SchemeA";
        case deglib::builder::OptimizationTarget::SchemeB:                 return "SchemeB";
        default:                                                            return "Unknown";
    }
}

// Format a duration (given in ms) as "XX.X ms" when < 10000 ms, else "XX.X s".
inline void print_time(const char* label, double ms) {
    if (ms < 10000.0) {
        std::printf("%-24s%6.1f ms\n", label, ms);
    } else {
        std::printf("%-24s%6.1f s\n",  label, ms / 1000.0);
    }
}

inline void print_summary(
    const char* mode_name,
    uint32_t mode_number,
    double load_ms,
    double quantize_ms,
    double build_ms,
    double convert_ms,
    double prune_ms,
    double explore_ms,
    double rerank_ms,
    double total_elapsed_ms,
    bool compute_recall,
    uint32_t k_top,
    float recall,
    uint32_t threads,
    uint32_t max_dist,
    uint32_t prune_worst,
    uint8_t k_graph,
    uint8_t k_ext,
    float eps_ext,
    uint32_t non_zeros,
    size_t count,
    size_t dims,
    uint32_t evpK,
    deglib::builder::OptimizationTarget opt_target = deglib::builder::OptimizationTarget::LowLID,
    double flas_ms = 0.0,
    double opt_ms = 0.0)
{
    std::printf("========================================================================\n");
    std::printf("  FINAL SUMMARY (%s - Mode %u)\n", mode_name, mode_number);
    std::printf("========================================================================\n");
    print_time("Load Time:",             load_ms);
    print_time("Quantize Time:",         quantize_ms);
    print_time("Graph Build Time:",      build_ms);
    print_time("Graph Conversion Time:", convert_ms);
    print_time("Pruning Time:",          prune_ms);
    print_time("Explore Time:",          explore_ms);
    print_time("Rerank Time:",           rerank_ms);
    if (flas_ms > 0.0) {
        print_time("FLAS Time:",         flas_ms);
    }
    if (opt_ms > 0.0) {
        print_time("Graph Opt Time:",     opt_ms);
    }
    print_time("Total Elapsed Time:",    total_elapsed_ms);
    if (compute_recall) {
        std::printf("Recall@%u:              %6.2f %%\n", k_top, recall * 100.0f);
    }
    std::printf("------------------------------------------------------------------------\n");
    std::printf("Hyperparameters:\n");
    if (non_zeros > 0) {
        std::printf("  NON_ZEROS:             %u\n", non_zeros);
    }
    std::printf("  K_TOP:                 %u\n", k_top);
    std::printf("  K_GRAPH:               %u\n", (uint32_t)k_graph);
    std::printf("  K_EXT:                 %u\n", (uint32_t)k_ext);
    std::printf("  EPS_EXT:               %.3f\n", eps_ext);
    std::printf("  OPT_TARGET:            %s\n", opt_target_str(opt_target));
    if (evpK > 0) {
        std::printf("  evpK:                  %u\n", evpK);
    }
    std::printf("  max_dist:              %u\n", max_dist);
    std::printf("  threads:               %u\n", threads);
    std::printf("  prune_worst:           %u\n", prune_worst);
    std::printf("------------------------------------------------------------------------\n");
    std::printf("Dataset Info:\n");
    std::printf("  Vectors:               %zu\n", count);
    std::printf("  Dimensions:            %zu\n", dims);
    std::printf("========================================================================\n\n");
}

inline double prune_worst_neighbors(deglib::graph::SizeBoundedGraph& graph, uint32_t prune_worst, uint32_t threads) {
    if (prune_worst == 0) {
        return 0.0;
    }

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

    return now_ms() - t_prune_start;
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

/**
 * @brief Run FLAS pre-sort on a flat FP32 array of N×D vectors.
 *
 * @param database_fp32  Flat array of count * dims float values (row-major).
 * @param count          Number of vectors.
 * @param dims           Dimensionality of each vector.
 * @param metric         Distance metric for FLAS (L2 or InnerProduct).
 * @param flas_ms        [out] FLAS elapsed time in milliseconds.
 * @return std::vector<uint32_t>  Sorted permutation: result[i] = original index at grid position i.
 *                                Empty on failure.
 */
inline std::vector<uint32_t> run_flas_presort(
    const float* database_fp32, size_t count, size_t dims,
    FlasMetric metric, double& flas_ms,
    float radius_decay = 0.93f)
{
    double t_start = now_ms();

    std::vector<MapField> map_fields(count);
    for (size_t i = 0; i < count; ++i) {
        init_map_field(&map_fields[i], static_cast<int>(i),
                       &database_fp32[i * dims], true);
    }

    FlasSettings settings = default_flas_settings();
    settings.metric = metric;
    settings.radius_decay = radius_decay;

    std::mt19937 rng(42);  // deterministic seed

    auto progress_callback = [t_start](float progress) mutable -> bool {
        static int last_pct = -1;
        int pct = static_cast<int>(progress * 100.0f);
        if (pct > last_pct) {
            last_pct = pct;
            double elapsed_s = (now_ms() - t_start) / 1000.0;
            if (progress > 0.0f) {
                double total_est = elapsed_s / progress;
                double remaining = total_est - elapsed_s;
                std::printf("\r  FLAS progress: %3d%% - elapsed %.0fs, est. remaining %.0fs   ",
                            pct, elapsed_s, remaining);
            } else {
                std::printf("\r  FLAS progress: %3d%% - elapsed %.0fs   ", pct, elapsed_s);
            }
            std::fflush(stdout);
        }
        return false;
    };

    do_sorting_full(
        map_fields.data(),
        static_cast<int>(dims),
        1,                              // columns = 1 (1D)
        static_cast<int>(count),        // rows = N
        &settings,
        &rng,
        progress_callback
    );
    std::printf("\r  FLAS progress: 100%% - done.                    \n");

    std::vector<uint32_t> sorted_indices(count);
    for (size_t i = 0; i < count; ++i) {
        sorted_indices[i] = static_cast<uint32_t>(map_fields[i].id);
    }

    // Validate permutation
    std::vector<bool> seen(count, false);
    bool valid = true;
    for (size_t i = 0; i < count; ++i) {
        if (sorted_indices[i] >= count) {
            std::fprintf(stderr, "Error: FLAS returned invalid index %u at position %zu\n",
                         sorted_indices[i], i);
            valid = false;
            break;
        }
        if (seen[sorted_indices[i]]) {
            std::fprintf(stderr, "Error: FLAS returned duplicate index %u at position %zu\n",
                         sorted_indices[i], i);
            valid = false;
            break;
        }
        seen[sorted_indices[i]] = true;
    }

    flas_ms = now_ms() - t_start;

    if (!valid) {
        return {};
    }
    std::printf("FLAS permutation valid: %zu unique indices | Time: %.2f s\n",
                count, flas_ms / 1000.0);
    return sorted_indices;
}

inline void optimize_graph(deglib::graph::SizeBoundedGraph& graph,
                           const uint8_t k_opt,
                           const float eps_opt,
                           const uint8_t i_opt,
                           const uint64_t total_iterations,
                           const uint64_t log_interval = 10000) {
    auto rnd = std::mt19937(7);
    auto builder = deglib::builder::EvenRegularGraphBuilder(
        graph, rnd, deglib::builder::OptimizationTarget::LowLID, 0, 0, k_opt, eps_opt, i_opt, 1, 0);

    auto start = std::chrono::steady_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    uint64_t duration_ms = 0;

    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto tries = status.tries;
        const auto improved = status.improved;

        // Log progress at intervals
        if (log_interval > 0 && tries > 0 && tries % log_interval == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto duration = duration_ms / 1000;
            auto avg_improv = uint32_t((improved - last_status.improved));

            std::printf("%5jds, %8ju / %8ju iterations (%2u improvements)\n",
                        (uint64_t)duration,
                        (uint64_t)improved,
                        (uint64_t)tries,
                        (uint32_t)avg_improv);

            last_status = status;
            start = std::chrono::steady_clock::now();
        }

        // stop after total_iterations
        if (tries >= total_iterations) builder.stop();
    };

    builder.build(improvement_callback, true);
}

} // namespace sisap_common
