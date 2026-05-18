#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "evp_bits.h"
#include "repository.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <array>
#include <random>
#include <vector>
#include <cstdlib>

// ============================================================================
// SISAP-Benchmark: Replicate task1_evp.py
// ============================================================================

int main() {
    // DATA_PATH defined at compile-time via CMake (or empty string if not set)
    const std::filesystem::path data_path(DATA_PATH);

    const auto train_fvecs = data_path / "SISAP" / "train.fvecs";
    const auto allknn_ivecs = data_path / "SISAP" / "allknn.ivecs";

    // Parameter
    constexpr uint32_t NON_ZEROS = 512;
    constexpr uint32_t K_TOP = 15;
    constexpr uint32_t K_RECALL = 15;

    printf("=== SISAP EVP Benchmark ===\n");
    printf("Data path: %ls\n", data_path.wstring().c_str());
    printf("NON_ZEROS: %u, K_TOP: %u, K_RECALL: %u\n", NON_ZEROS, K_TOP, K_RECALL);

    // Thread count via env DEG_THREADS (0 = autodetect)
    size_t threads = 0;
    if (const char* threads_env = std::getenv("DEG_THREADS"); threads_env != nullptr) {
        try {
            long long t = std::stoll(threads_env);
            if (t > 0) {
                threads = static_cast<size_t>(t);
            }
        } catch (...) {
            // Ignore parse errors and keep autodetect.
        }
    }

    // Limit vectors for faster iterative profiling. Default: 20k vectors.
    size_t max_vectors = 20000;
    if (const char* max_vec_env = std::getenv("DEG_MAX_VECTORS"); max_vec_env != nullptr) {
        try {
            long long v = std::stoll(max_vec_env);
            if (v >= 0) {
                max_vectors = static_cast<size_t>(v);
            }
        } catch (...) {
            // Ignore parse errors and keep default.
        }
    }

    // ---- Step 1: Load train.fvecs ----
    printf("\nLoading train.fvecs...\n");
    auto t_start = std::chrono::high_resolution_clock::now();

    size_t dims, count;
    auto train_bytes = deglib::fvecs_read(train_fvecs.string().c_str(), dims, count);
    const float* train_data = reinterpret_cast<const float*>(train_bytes.get());

    printf("  Loaded: %zu vectors, dim=%zu\n", count, dims);

    // ---- Step 2: Load allknn.ivecs ----
    printf("Loading allknn.ivecs...\n");
    size_t gt_dims, gt_count;
    auto gt_bytes = deglib::ivecs_read(allknn_ivecs.string().c_str(), gt_dims, gt_count);
    const int32_t* gt_data = reinterpret_cast<const int32_t*>(gt_bytes.get());

    printf("  Loaded: %zu ground truth vectors, dim=%zu\n", gt_count, gt_dims);

    size_t eval_count = std::min(count, gt_count);
    if (max_vectors > 0) {
        eval_count = std::min(eval_count, max_vectors);
    }
    printf("Using %zu vectors for evaluation", eval_count);
    if (max_vectors == 0) {
        printf(" (no limit)\n");
    } else {
        printf(" (limit=%zu, set DEG_MAX_VECTORS=0 for full run)\n", max_vectors);
    }

    auto t_end_load = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double>(t_end_load - t_start).count();
    printf("Loading & File I/O: %.2f s\n", load_time);

    // ---- Step 3: Create EvpBitsArray ----
    printf("\nConverting to EvpBits (non_zeros=%u)...\n", NON_ZEROS);
    auto t_convert_start = std::chrono::high_resolution_clock::now();

    deglib::EvpBitsArray array(train_data, eval_count, static_cast<uint32_t>(dims), NON_ZEROS, threads);

    auto t_convert_end = std::chrono::high_resolution_clock::now();
    double convert_time = std::chrono::duration<double>(t_convert_end - t_convert_start).count();
    printf("Conversion: %.2f s\n", convert_time);
    printf("  bytes_per_evp: %zu\n", array.bytes_per_evp());
    printf("  total memory: %.2f MB\n",
           static_cast<double>(array.size() * array.bytes_per_evp()) / (1024.0 * 1024.0));

    // ---- Step 4: All-Pairs Similarity + Top-K ----
    printf("\nComputing all-pairs similarity and Top-%d...\n", K_TOP);
    auto t_sim_start = std::chrono::high_resolution_clock::now();

    // For each vector: find Top-K neighbors (excluding self)
    std::vector<std::vector<uint32_t>> topk_results(eval_count, std::vector<uint32_t>(K_TOP));

    // Cache pointers once to reduce repeated index arithmetic in the hot path.
    std::vector<const std::byte*> ones_ptrs(eval_count);
    std::vector<const std::byte*> negs_ptrs(eval_count);
    for (size_t i = 0; i < eval_count; ++i) {
        ones_ptrs[i] = array.ones_ptr(static_cast<uint32_t>(i));
        negs_ptrs[i] = array.negs_ptr(static_cast<uint32_t>(i));
    }

    // Use deglib::concurrent::parallel_for with batching for similarity computation
    const size_t batch_size = 256;
    const size_t num_batches = (eval_count + batch_size - 1) / batch_size;
    deglib::concurrent::parallel_for(
        static_cast<size_t>(0), num_batches, threads,
        [&topk_results, &ones_ptrs, &negs_ptrs, eval_count, dims, batch_size](size_t batch_id, size_t /*threadId*/) {
            size_t start = batch_id * batch_size;
            size_t end = std::min(start + batch_size, eval_count);
            for (size_t i = start; i < end; ++i) {
                // For small K (15), a fixed Top-K buffer is faster than a heap.
                size_t available = (eval_count > 0) ? (eval_count - 1) : 0;
                size_t k = K_TOP < available ? K_TOP : available;
                if (k == 0) {
                    continue;
                }

                const std::byte* ones_i = ones_ptrs[i];
                const std::byte* negs_i = negs_ptrs[i];

                std::array<std::pair<float, uint32_t>, K_TOP> top{};
                size_t top_size = 0;

                for (size_t j = 0; j < eval_count; ++j) {
                    // Exclude self-match (with recall is ~0.68, without 0.7270)
                    if (i == j) {
                        continue;
                    }

                    float sim = deglib::evp_similarity_bytes(
                        ones_i,
                        negs_i,
                        ones_ptrs[j],
                        negs_ptrs[j],
                        static_cast<uint32_t>(dims));

                    if (top_size < k) {
                        top[top_size++] = std::make_pair(sim, static_cast<uint32_t>(j));
                    } else {
                        size_t min_pos = 0;
                        float min_val = top[0].first;
                        for (size_t m = 1; m < k; ++m) {
                            if (top[m].first < min_val) {
                                min_val = top[m].first;
                                min_pos = m;
                            }
                        }
                        if (sim > min_val) {
                            top[min_pos] = std::make_pair(sim, static_cast<uint32_t>(j));
                        }
                    }
                }

                std::sort(top.begin(), top.begin() + static_cast<long long>(k),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

                for (size_t j = 0; j < k; ++j) {
                    topk_results[i][j] = top[j].second;
                }
        }
    });

    auto t_sim_end = std::chrono::high_resolution_clock::now();
    double sim_time = std::chrono::duration<double>(t_sim_end - t_sim_start).count();
    printf("Similarity Computation: %.2f s\n", sim_time);

    // ---- Step 5: Calculate Recall@K ----
    printf("Calculating recall...\n");
    auto t_recall_start = std::chrono::high_resolution_clock::now();

    // Ground Truth is 1-indexed -> subtract 1
    // Use only the first K_RECALL entries from GT (matching Rust/Python)
    int total_hits = 0;
    int total_queries = 0;

    for (size_t i = 0; i < eval_count; ++i) {
        // Ground Truth for vector i: only the first K_RECALL entries
        const int32_t* gt_row = &gt_data[i * gt_dims];
        int hits = 0;

        for (uint32_t k = 0; k < K_RECALL && k < static_cast<uint32_t>(gt_dims); ++k) {
            int32_t gt_idx = gt_row[k] - 1;  // 1-indexed -> 0-indexed

            // Check if gt_idx is in the Top-K results
            for (uint32_t j = 0; j < K_TOP && j < static_cast<uint32_t>(topk_results[i].size()); ++j) {
                if (topk_results[i][j] == static_cast<uint32_t>(gt_idx)) {
                    hits++;
                    break;
                }
            }
        }

        total_hits += hits;
        // Normalize by K_RECALL, not gt_dims
        total_queries += K_RECALL;
    }

    auto t_recall_end = std::chrono::high_resolution_clock::now();
    double recall_time = std::chrono::duration<double>(t_recall_end - t_recall_start).count();

    double recall_at_15 = static_cast<double>(total_hits) / total_queries;

    // ---- Summary ----
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    printf("\n=============================================\n");
    printf("          FINAL EVALUATION SUMMARY\n");
    printf("=============================================\n");
    printf("Loading & Conversion:              %.2f s\n", load_time + convert_time);
    printf("Similarity Computation:          %.2f s\n", sim_time);
    printf("Recall Calculation:                %.2f s\n", recall_time);
    printf("---------------------------------------------\n");
    printf("TOTAL:                            %.2f s\n", total_time);
    printf("AVERAGE RECALL@%d:                 %.4f\n", K_RECALL, recall_at_15);
    printf("=============================================\n");

    return 0;
}
