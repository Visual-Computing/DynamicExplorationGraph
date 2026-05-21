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
 *   deglib_evp_test <data_path> [evp|evp-no-rerank|evp-asymmetric|raw|fp16]
 *
 * Expected files in data_path:
 *   train.fvecs      - float feature vectors
 *   allknn.ivecs     - ground truth (1-indexed, dims=32)
 */

#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <algorithm>
#include <thread>
#include <vector>

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

// ============================================================================
// Helpers
// ============================================================================

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
           .count();
}

// ============================================================================
// Consolidated Modes
// ============================================================================

enum class BenchmarkMode {
    EvpWithRerank,
    EvpNoRerank,
    EvpAsymmetric,
    Raw,
    FP16,
    EvpLinear
};

// ============================================================================
// Shared helpers
// ============================================================================

static void run_exploration_sweep(
    const deglib::search::SearchGraph& graph,
    const int32_t* gt_data,
    size_t count,
    size_t gt_dims,
    uint32_t k_top,
    uint8_t threads,
    const char* label,
    const void* rerank_data = nullptr,
    size_t rerank_feature_bytes = 0,
    deglib::DISTFUNC<float> rerank_dist_func = nullptr,
    size_t dims = 0)
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
                size_t entry_idx = graph.getInternalIndex(static_cast<uint32_t>(label));
                uint32_t k_search = (rerank_data != nullptr) ? std::max(k_top, max_distance_count) : k_explore;

                auto result_queue = graph.explore(
                    static_cast<uint32_t>(entry_idx),
                    k_search,
                    false,  // include_entry (false: exclude self)
                    max_distance_count);

                auto& cands = explore_candidates[label];
                cands.reserve(result_queue.size());
                while (!result_queue.empty()) {
                    cands.push_back(graph.getExternalLabel(result_queue.top().getInternalIndex()));
                    result_queue.pop();
                }
            });
        double explore_ms = now_ms() - t_explore_start;

        // --- Phase 2: Reranking (if rerank_data is provided) ---
        double rerank_ms = 0.0;
        std::vector<std::vector<uint32_t>> results(count);

        if (rerank_data != nullptr) {
            double t_rerank_start = now_ms();
            const auto* base = static_cast<const std::byte*>(rerank_data);
            deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
                [&](size_t label, size_t thread_id) {
                    struct Candidate {
                        uint32_t label;
                        float distance;
                    };
                    const auto& cands = explore_candidates[label];
                    std::vector<Candidate> candidates;
                    candidates.reserve(cands.size());

                    size_t dims_size = dims;
                    const std::byte* query_ptr = base + label * rerank_feature_bytes;

                    for (uint32_t cand_label : cands) {
                        const std::byte* cand_ptr = base + cand_label * rerank_feature_bytes;
                        float exact_dist = rerank_dist_func(query_ptr, cand_ptr, &dims_size);
                        candidates.push_back({cand_label, exact_dist});
                    }

                    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                        return a.distance < b.distance;
                    });

                    auto& result = results[label];
                    result.resize(k_top);
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
            const int32_t* gt_row = &gt_data[i * gt_dims];
            for (uint32_t k = 1; k <= k_top && k < gt_dims; ++k) {
                uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1→0 indexed
                for (uint32_t j = 0; j < k_top; ++j) {
                    if (results[i][j] == gt_idx) {
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
// FP32 -> FP16 batch conversion helper
// ============================================================================

/**
 * Converts a packed FP32 array (count × dims floats) to FP16 (uint16_t) in
 * parallel. Uses F16C intrinsics when available, falls back to a portable
 * software implementation otherwise.
 */
static std::vector<uint16_t> fp32_to_fp16_batch(
    const float* src, size_t count, size_t dims, uint32_t threads)
{
    std::vector<uint16_t> dst(count * dims);
    deglib::concurrent::parallel_for(static_cast<size_t>(0), count, threads, 1,
        [&](size_t i, size_t /*thread_id*/) {
            const float* s = src + i * dims;
            uint16_t*    d = dst.data() + i * dims;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
            for (size_t k = 0; k < dims; ++k) {
                __m128  f32 = _mm_set_ss(s[k]);
                __m128i h   = _mm_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                d[k] = static_cast<uint16_t>(_mm_cvtsi128_si32(h));
            }
#else
            for (size_t k = 0; k < dims; ++k) {
                uint32_t bits;
                std::memcpy(&bits, &s[k], sizeof(bits));
                const uint32_t sign     = (bits & 0x80000000u) >> 16;
                const uint32_t exponent = (bits & 0x7F800000u) >> 23;
                const uint32_t mantissa = (bits & 0x007FFFFFu);
                uint16_t h;
                if (exponent == 255) {
                    h = static_cast<uint16_t>(sign | 0x7C00u | (mantissa ? 0x0200u : 0u));
                } else {
                    const int32_t e = static_cast<int32_t>(exponent) - 127 + 15;
                    if      (e >= 31) h = static_cast<uint16_t>(sign | 0x7C00u); // overflow → inf
                    else if (e <= 0)  h = static_cast<uint16_t>(sign);            // underflow → 0
                    else              h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (mantissa >> 13));
                }
                d[k] = h;
            }
#endif
        });
    return dst;
}

// ============================================================================
// Unified Benchmark Entry Point
// ============================================================================

static int run_benchmark(
    BenchmarkMode mode,
    const std::filesystem::path& data_path,
    uint32_t threads,
    uint32_t non_zeros,
    uint8_t k_graph, uint8_t k_ext,
    float eps_ext,
    uint32_t k_top)
{
    // --------------------------------------------------------------------------
    // Load data
    // --------------------------------------------------------------------------
    const auto train_fvecs  = data_path / "SISAP" / "train.fvecs";
    const auto allknn_ivecs = data_path / "SISAP" / "allknn.ivecs";

    if (!std::filesystem::exists(train_fvecs)) {
        std::fprintf(stderr, "Error: %ls not found\n", train_fvecs.wstring().c_str());
        return 1;
    }
    if (!std::filesystem::exists(allknn_ivecs)) {
        std::fprintf(stderr, "Error: %ls not found\n", allknn_ivecs.wstring().c_str());
        return 1;
    }

    double t_load = now_ms();

    size_t dims = 0, count = 0;
    auto train_bytes = deglib::fvecs_read(train_fvecs.string().c_str(), dims, count);
    const float* train_data = reinterpret_cast<const float*>(train_bytes.get());

    size_t gt_dims = 0, gt_count = 0;
    auto gt_bytes = deglib::ivecs_read(allknn_ivecs.string().c_str(), gt_dims, gt_count);
    const int32_t* gt_data = reinterpret_cast<const int32_t*>(gt_bytes.get());

    double load_ms = now_ms() - t_load;
    std::printf("Loaded: %zu vectors, dim=%zu\n", count, dims);
    std::printf("Ground truth: %zu queries, top-%zu\n", gt_count, gt_dims);
    std::printf("Load time: %.2f ms\n\n", load_ms);

    if (count != gt_count) {
        std::fprintf(stderr,
            "Error: train.fvecs and allknn.ivecs must contain the same number of entries (%zu vs %zu)\n",
            count, gt_count);
        return 1;
    }

    if (mode == BenchmarkMode::Raw) {
        std::printf("=== FP32 InnerProduct Graph Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f\n",
                    k_top, k_graph, k_ext, eps_ext);
        std::printf("Threads: %zu\n\n", threads);

        // --------------------------------------------------------------------------
        // Build graph with Metric::InnerProduct (no quantization)
        // --------------------------------------------------------------------------
        double t2 = now_ms();

        deglib::FloatSpace feature_space(static_cast<uint32_t>(dims), deglib::Metric::InnerProduct);
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

        // Add all entries from raw float data
        for (size_t i = 0; i < count; ++i) {
            std::vector<std::byte> feature(dims * sizeof(float));
            std::memcpy(feature.data(), train_data + i * dims, dims * sizeof(float));
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
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", graph.size(), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        // --------------------------------------------------------------------------
        // Exploration sweep
        // --------------------------------------------------------------------------
        run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, threads, "FP32 InnerProduct");

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (FP32 InnerProduct)\n");
        std::printf("=============================================\n");
        std::printf("Graph Build:             %.2f ms\n", build_ms);
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("=============================================\n\n");

        return 0;
    } else if (mode == BenchmarkMode::FP16) {
        std::printf("=== FP16 InnerProduct Graph Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f\n",
                    k_top, k_graph, k_ext, eps_ext);
        std::printf("Threads: %zu\n\n", threads);

        // --------------------------------------------------------------------------
        // Convert FP32 → FP16
        // --------------------------------------------------------------------------
        double t_conv_start = now_ms();
        auto fp16_data = fp32_to_fp16_batch(train_data, count, dims, threads);
        double convert_ms = now_ms() - t_conv_start;

        std::printf("FP32→FP16 conversion time: %.2f ms\n", convert_ms);
        std::printf("FP16 data size: %.2f MB\n\n",
                    static_cast<double>(fp16_data.size() * sizeof(uint16_t)) / (1024.0 * 1024.0));

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

        // Add all entries from FP16 data
        const size_t bytes_per_fp16 = dims * sizeof(uint16_t);
        for (size_t i = 0; i < count; ++i) {
            std::vector<std::byte> feature(bytes_per_fp16);
            std::memcpy(feature.data(), fp16_data.data() + i * dims, bytes_per_fp16);
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
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", graph.size(), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        // --------------------------------------------------------------------------
        // Exploration sweep
        // --------------------------------------------------------------------------
        run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, threads, "FP16 InnerProduct");

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (FP16 InnerProduct)\n");
        std::printf("=============================================\n");
        std::printf("FP32→FP16 Conversion:    %.2f ms\n", convert_ms);
        std::printf("Graph Build:             %.2f ms\n", build_ms);
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("FP16 bytes/vector:       %zu\n", bytes_per_fp16);
        std::printf("=============================================\n\n");

        return 0;
    } else if (mode == BenchmarkMode::EvpWithRerank || mode == BenchmarkMode::EvpNoRerank || mode == BenchmarkMode::EvpAsymmetric) {
        std::printf("=== EVP Bits Graph Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("NON_ZEROS=%u, K_TOP=%u, K_GRAPH=%u, K_EXT=%u, EPS_EXT=%.2f, MODE=%s\n",
                    non_zeros, k_top, k_graph, k_ext, eps_ext,
                    mode == BenchmarkMode::EvpAsymmetric ? "asymmetric" :
                    (mode == BenchmarkMode::EvpWithRerank ? "rerank" : "no-rerank"));
        std::printf("Threads: %zu\n\n", threads);

        // --------------------------------------------------------------------------
        // Quantize
        // --------------------------------------------------------------------------
        double t1 = now_ms();

        auto quantized = deglib::quantization::quantize_batch(
            train_data, count, static_cast<uint32_t>(dims), non_zeros, threads);

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
            std::vector<std::byte> feature(quantized.data() + i * bytes_per_evp,
                                           quantized.data() + (i + 1) * bytes_per_evp);
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
        std::printf("Graph built: %zu vertices, %u edges/vertex\n", graph.size(), graph.getEdgesPerVertex());
        std::printf("Build time: %.2f ms\n\n", build_ms);

        // --------------------------------------------------------------------------
        // Exploration sweep
        // --------------------------------------------------------------------------
        double conversion_ms = 0.0;
        if (mode == BenchmarkMode::EvpAsymmetric) {
            // Convert FP32 → FP16 for asymmetric graph
            double t_conv_start = now_ms();
            auto fp16_data = fp32_to_fp16_batch(train_data, count, dims, threads);
            conversion_ms = now_ms() - t_conv_start;
            std::printf("FP32→FP16 conversion time: %.2f ms\n", conversion_ms);

            deglib::FloatSpace fp16_feature_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            deglib::graph::ReadOnlyGraph fp16_graph(fp16_feature_space, graph, fp16_data.data());
            conversion_ms += now_ms() - t_conv_start;
            std::printf("Asymmetric graph conversion time: %.2f ms\n", conversion_ms);

            run_exploration_sweep(fp16_graph, gt_data, count, gt_dims, k_top, threads,
                                  "EVP Asymmetric (ReadOnly FP16)");
        } else if (mode == BenchmarkMode::EvpWithRerank) {
            // Convert FP32 → FP16 for reranking
            double t_conv_start = now_ms();
            auto fp16_data = fp32_to_fp16_batch(train_data, count, dims, threads);
            conversion_ms = now_ms() - t_conv_start;
            std::printf("FP32→FP16 conversion time: %.2f ms\n", conversion_ms);

            deglib::FloatSpace fp16_rerank_space(static_cast<uint32_t>(dims), deglib::Metric::FP16InnerProduct);
            run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, threads,
                                  "EVP Bits (with FP16 Reranking)",
                                  fp16_data.data(),
                                  dims * sizeof(uint16_t),
                                  fp16_rerank_space.get_dist_func(),
                                  dims);
        } else {
            run_exploration_sweep(graph, gt_data, count, gt_dims, k_top, threads,
                                  "EVP Bits (no Reranking)");
        }

        // --------------------------------------------------------------------------
        // Summary
        // --------------------------------------------------------------------------
        std::printf("=============================================\n");
        std::printf("          FINAL SUMMARY (EVP Bits)\n");
        std::printf("=============================================\n");
        std::printf("Quantize:                %.2f ms\n", quantize_ms);
        std::printf("Graph Build:             %.2f ms\n", build_ms);
        if (mode == BenchmarkMode::EvpAsymmetric) {
            std::printf("Graph Conversion:        %.2f ms\n", conversion_ms);
        }
        std::printf("---------------------------------------------\n");
        std::printf("Vectors:                 %zu\n", count);
        std::printf("Dimensions:              %zu\n", dims);
        std::printf("NON_ZEROS:               %u\n", non_zeros);
        std::printf("=============================================\n\n");

        return 0;
    } else if (mode == BenchmarkMode::EvpLinear) {
        std::printf("=== EVP Bits Linear Search Benchmark ===\n");
        std::printf("Data path: %ls\n", data_path.wstring().c_str());
        std::printf("NON_ZEROS=%u, K_TOP=%u\n", non_zeros, k_top);
        std::printf("Threads: %zu\n\n", threads);

        // --------------------------------------------------------------------------
        // Quantize
        // --------------------------------------------------------------------------
        double t1 = now_ms();

        auto quantized = deglib::quantization::quantize_batch(
            train_data, count, static_cast<uint32_t>(dims), non_zeros, threads);

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
            const int32_t* gt_row = &gt_data[i * gt_dims];
            for (uint32_t k = 1; k <= k_top && k < gt_dims; ++k) {
                uint32_t gt_idx = static_cast<uint32_t>(gt_row[k] - 1); // 1→0 indexed
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
    std::string mode = "evp";  // default: EVP Bits mode

    if (argc >= 2) {
        data_path = argv[1];
    } else {
#ifdef DATA_PATH
        data_path = DATA_PATH;
#else
        std::fprintf(stderr, "Usage: %s <data_path> [evp|evp-no-rerank|evp-asymmetric|raw|fp16|evp-linear]\n", argv[0]);
        return 1;
#endif
    }

    if (argc >= 3) {
        mode = argv[2];
    }

    if (mode != "evp" && mode != "evp-no-rerank" && mode != "evp-asymmetric" && mode != "raw" && mode != "fp16" && mode != "evp-linear") {
        std::fprintf(stderr, "Unknown mode '%s'. Use 'evp', 'evp-no-rerank', 'evp-asymmetric', 'raw', 'fp16', or 'evp-linear'.\n", mode.c_str());
        return 1;
    }

    const size_t threads = 6;

    // --------------------------------------------------------------------------
    // Dispatch to selected mode
    // --------------------------------------------------------------------------
    BenchmarkMode benchmark_mode = BenchmarkMode::EvpWithRerank;
    if (mode == "evp-no-rerank") {
        benchmark_mode = BenchmarkMode::EvpNoRerank;
    } else if (mode == "evp-asymmetric") {
        benchmark_mode = BenchmarkMode::EvpAsymmetric;
    } else if (mode == "raw") {
        benchmark_mode = BenchmarkMode::Raw;
    } else if (mode == "fp16") {
        benchmark_mode = BenchmarkMode::FP16;
    } else if (mode == "evp-linear") {
        benchmark_mode = BenchmarkMode::EvpLinear;
    }

    return run_benchmark(benchmark_mode, data_path, threads, NON_ZEROS, K_GRAPH, K_EXT, EPS_EXT, K_TOP);
}

