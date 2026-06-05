#pragma once

/**
 * @file flas_common.h
 * @brief Shared FLAS pre-sort helper for EVP Task 2 modes.
 *
 * Provides a single function run_flas_presort() that all mode files
 * call with the same interface, eliminating code duplication.
 */

#include <cstdio>
#include <random>
#include <vector>

#include "flas/fast_linear_assignment_sorter.hpp"
#include "evp_common.h"

namespace flas_common {

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
    FlasMetric metric, double& flas_ms)
{
    double t_start = evp_common::now_ms();

    std::vector<MapField> map_fields(count);
    for (size_t i = 0; i < count; ++i) {
        init_map_field(&map_fields[i], static_cast<int>(i),
                       &database_fp32[i * dims], true);
    }

    FlasSettings settings = default_flas_settings();
    settings.metric = metric;

    std::mt19937 rng(42);  // deterministic seed

    auto progress_callback = [t_start](float progress) mutable -> bool {
        static int last_pct = -1;
        int pct = static_cast<int>(progress * 100.0f);
        if (pct > last_pct) {
            last_pct = pct;
            double elapsed_s = (evp_common::now_ms() - t_start) / 1000.0;
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

    flas_ms = evp_common::now_ms() - t_start;

    if (!valid) {
        return {};
    }
    std::printf("FLAS permutation valid: %zu unique indices | Time: %.2f s\n",
                count, flas_ms / 1000.0);
    return sorted_indices;
}

} // namespace flas_common
