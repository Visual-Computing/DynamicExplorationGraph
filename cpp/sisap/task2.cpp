/**
 * @file task2.cpp
 * @brief EVP Benchmark Task 2 — Entrypoint to dispatch HDF5-based MIPS ANN benchmarks for LLM attention workloads.
 *
 * Overview
 * --------
 * This executable implements the baseline benchmark mode (mode1) exercising FP32 graph construction
 * and FP32 search.
 * It accepts a single HDF5 input file (llama-dev.h5 ONLY — no other HDF5 files are supported for task2) containing at least the "train" dataset and
 * the "test/knns" ground-truth for recall evaluation.
 *
 * Input Format
 * ------------
 * The input HDF5 file must contain:
 *   - "train"         : N × D dataset of FP32 (float) feature vectors (database).
 *   - "test/queries"  : Q × D query vectors (separate from database).
 *   - "test/knns"     : Q × K ground-truth nearest-neighbor indices (0-based, int32),
 *                       required when running in recall mode (the default).
 *
 * Output Modes
 * ------------
 * Two mutually exclusive output behaviours are supported:
 *
 *   1. Recall mode (default):
 *      Loads the "test/knns" ground truth, runs the benchmark, and prints Recall@K.
 *
 *   2. Save mode (--no-recall --output <path>):
 *      Skips ground-truth loading, runs the search once, and writes the retrieved
 *      neighbor indices to a binary ivecs file (one row per query; each row:
 *      uint32_t count followed by count × uint32_t label values).
 *
 * Benchmark Modes
 * ---------------
 *   mode1  (baseline | fp32-build-fp32-explore)
 *     Builds a SizeBoundedGraph using FP32 features (Metric::InnerProduct) and
 *     explores via FP32 inner-product similarity. Serves as the high-quality baseline.
 *
 *   mode2  (fp16-build-fp16-explore)
 *     Builds the graph with FP16 features and explores via FP16 inner-product similarity.
 *
 *   mode3  (baseline-fp16 | fp32-build-fp16-explore)
 *     Builds the graph with FP32 features and explores via FP16 inner-product similarity.
 *
 *   mode4  (l2-converted | fp32-build-l2-explore)
 *     Builds the graph with FP32 features converted to L2 space (dimension d+1)
 *     and explores via FP32 L2 similarity.
 *
 *   mode5  (l2-fp16-ip | l2-build-fp16-ip-explore)
 *     Builds the graph with FP32 features converted to L2 space (dimension d+1)
 *     and explores via FP16 Inner Product similarity.
 *
 *   mode6  (l2-fp16-l2 | l2-build-fp16-l2-explore)
 *     Builds the graph with FP32 features converted to L2 space (dimension d+1)
 *     and explores via FP16 L2 similarity.
 *
 *   mode7  (l2-fp16-d2 | l2-build-fp16-d2-explore)
 *     Builds the graph with FP32 features converted to L2-converted space (dimension d+2)
 *     and explores via FP16 L2 similarity.
 *
 * FLAS Settings
 * -------------
 *   --flas                    Enable FLAS pre-sort (default: off). Works with any mode.
 *   --flas-metric <l2|ip>    FLAS distance metric (default: l2).
 *   --flas-radius-decay <f>  FLAS swap radius decay factor (default: 0.93).
 *
 * CLI Usage
 * ---------
 *   deglib_sisap_task2.exe <hdf5_file> <mode> [options...]
 *
 *   <mode> accepts:
 *     - "baseline"
 *     - "fp32-build-fp32-explore"
 *     - "mode1"
 *
 * Key Options
 * -----------
 *   --threads <n>      Parallel thread count (default: 6).
 *   --k-top <n>        Number of nearest neighbors to retrieve (default: 30).
 *   --k-graph <n>      Graph degree / edges per vertex (default: 32).
 *   --k-ext <n>        Builder extension degree (default: 32).
 *   --eps-ext <f>      Builder entry-search expansion coefficient (default: 0.001).
 *   --max-dist <n>     Maximum distance computations per query during exploration (default: 200).
 *   --no-recall        Skip recall computation; requires --output.
 *   --output <path>    Write results to a binary ivecs file.
 *   --graph <path>     File path to save the pre-built graph to, or load a pre-built graph from.
 *   --prune-worst <n>  Number of worst (least similar) neighbors per vertex to replace with self-loops.
 *
 * Examples
 * --------
 *   # FP32 baseline (mode1)
 *   .\deglib_sisap_task2.exe llama-dev.h5 mode1 --max-dist 100
 *   # Or via combined entry point
 *   .\deglib_sisap.exe task2 llama-dev.h5 mode1 --max-dist 100
 */

#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <string>
#include <filesystem>
#include <iostream>

#include "sisap_common.h"
#include "task2/mode1.h"
#include "task2/mode2.h"
#include "task2/mode3.h"
#include "task2/mode4.h"
#include "task2/mode5.h"
#include "task2/mode6.h"
#include "task2/mode7.h"
#include "flas/fast_linear_assignment_sorter.hpp"

#ifdef BUILD_COMBINED_SISAP
int run_task2(int argc, char* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    try {
        std::string data_path;
        std::string mode;
        uint32_t threads = 8;
        uint32_t build_threads = 1;
        uint32_t k_top = 30;
        uint8_t k_graph = 30;
        uint8_t k_ext = 30;
        float eps_ext = 0.001f;
        bool run_recall = true;
        std::string output_path;
        std::string graph_path;
        std::string max_dist_str = "20000";
        uint32_t prune_worst = 0;
        std::string eps_search_str = "0.3";
        uint32_t num_runs = 1;
        bool use_flas = false;
        float goal_recall = 0.8f;
        std::string flas_metric_str = "l2";
        float flas_radius_decay = 0.93f;
        deglib::builder::OptimizationTarget opt_target = deglib::builder::OptimizationTarget::LowLID;
        uint64_t opt_iterations = 0;

        if (argc < 3) {
            std::fprintf(stderr, "Usage: %s <hdf5_file_path> <mode_name> [options...]\n", argv[0]);
            std::fprintf(stderr, "Modes:\n");
            std::fprintf(stderr, "  baseline | fp32-build-fp32-explore | mode1            : FP32 build + FP32 explore\n");
            std::fprintf(stderr, "  fp16-build-fp16-explore | mode2                      : FP16 build + FP16 explore\n");
            std::fprintf(stderr, "  baseline-fp16 | fp32-build-fp16-explore | mode3       : FP32 build + FP16 explore\n");
            std::fprintf(stderr, "  l2-converted | fp32-build-l2-explore | mode4              : FP32 build (L2 converted) + FP32 L2 search\n");
            std::fprintf(stderr, "  l2-fp16-ip | l2-build-fp16-ip-explore | mode5             : FP32 build (L2 converted) + FP16 InnerProduct search\n");
            std::fprintf(stderr, "  l2-fp16-l2 | l2-build-fp16-l2-explore | mode6             : FP32 build (L2 converted) + FP16 L2 search\n");
            std::fprintf(stderr, "  l2-fp16-d2 | l2-build-fp16-d2-explore | mode7             : FP32 build (L2-converted d+2) + FP16 L2 search\n");
            std::fprintf(stderr, "  Use --flas with any FP32 mode to enable FLAS pre-sort.\n\n");
            std::fprintf(stderr, "  --threads <n>      Number of CPU worker threads used for query exploration (default: 6).\n");
            std::fprintf(stderr, "  --build-threads <n> Number of CPU worker threads used for graph construction (default: same as --threads).\n");
            std::fprintf(stderr, "  --k-top <n>        The final number of nearest neighbors (top-K) retrieved per query\n");
            std::fprintf(stderr, "                     and evaluated for recall or written to the output file (default: 30).\n");
            std::fprintf(stderr, "  --k-graph <n>      The degree of the regular graph. Specifies the exact number of edges\n");
            std::fprintf(stderr, "                     (neighbors) per vertex. Must be an even integer >= 4 (default: 32).\n");
            std::fprintf(stderr, "  --k-ext <n>        The search size (k-top parameter) used during graph construction. Decides\n");
            std::fprintf(stderr, "                     how many good neighbors are shown to each newly added node, from which it\n");
            std::fprintf(stderr, "                     selects nodes to connect with up to k-graph (default: 32).\n");
            std::fprintf(stderr, "  --eps-ext <f>      Search expansion parameter used together with k-ext during graph construction.\n");
            std::fprintf(stderr, "                     Decides if nodes whose distance is slightly worse (e.g. 0.01 = 1%%) than the\n");
            std::fprintf(stderr, "                     current worst in the search list should be explored (default: 0.001).\n");
            std::fprintf(stderr, "  --no-recall        Disables loading ground-truth datasets and calculating Recall@K metrics.\n");
            std::fprintf(stderr, "                     Required when exporting search results to an output file.\n");
            std::fprintf(stderr, "  --output <path>    Path to a binary `.ivecs` file where the retrieved nearest-neighbor indices\n");
            std::fprintf(stderr, "                     will be saved (one row per query; uint32_t count followed by indices).\n");
            std::fprintf(stderr, "  --max-dist <list>  Exploration search budget or comma-separated list of budgets. Specifies the maximum number of\n");
            std::fprintf(stderr, "                     distance computations allowed per query. Main parameter to trade search speed for recall (default: 20000).\n");
            std::fprintf(stderr, "  --graph <path>     File path to save the pre-built graph to, or load a pre-built graph from\n");
            std::fprintf(stderr, "                     to bypass the construction phase.\n");
            std::fprintf(stderr, "  --prune-worst <n>  Number of worst (least similar) neighbors per vertex to replace with self-loops (default: 0).\n");
            std::fprintf(stderr, "  --flas                    Enable FLAS pre-sort of training vectors before graph construction (default: off).\n");
            std::fprintf(stderr, "  --flas-metric <l2|ip>   FLAS distance metric: l2 (Euclidean) or ip (Inner Product) (default: l2).\n");
            std::fprintf(stderr, "  --flas-radius-decay <f>  FLAS swap radius decay factor per iteration (default: 0.93).\n");
            std::fprintf(stderr, "  --num-runs <n>           Number of query explorations to perform and average (default: 1).\n");
            std::fprintf(stderr, "  --opt-target <str> Optimization target for graph builder (LowLID, HighLID, StreamingData_SchemeA, ..., default: LowLID).\n");
            std::fprintf(stderr, "  --goal-recall <f>  Recall threshold. Sweeps will select configuration minimizing search time\n");
            std::fprintf(stderr, "                     among those with recall >= threshold (default: 0.8).\n");
            std::fprintf(stderr, "  --opt-iterations <n> Number of graph optimization iterations to perform after building (default: 0).\n");
            return 1;
        }

        data_path = argv[1];
        mode = argv[2];

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--threads" && i + 1 < argc) {
                threads = std::stoul(argv[++i]);
            } else if (arg == "--build-threads" && i + 1 < argc) {
                build_threads = std::stoul(argv[++i]);
            } else if (arg == "--k-top" && i + 1 < argc) {
                k_top = std::stoul(argv[++i]);
            } else if (arg == "--k-graph" && i + 1 < argc) {
                k_graph = static_cast<uint8_t>(std::stoul(argv[++i]));
            } else if (arg == "--k-ext" && i + 1 < argc) {
                k_ext = static_cast<uint8_t>(std::stoul(argv[++i]));
            } else if (arg == "--eps-ext" && i + 1 < argc) {
                eps_ext = std::stof(argv[++i]);
            } else if (arg == "--no-recall") {
                run_recall = false;
            } else if (arg == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (arg == "--max-dist" && i + 1 < argc) {
                max_dist_str = argv[++i];
            } else if (arg == "--graph" && i + 1 < argc) {
                graph_path = argv[++i];
            } else if (arg == "--prune-worst" && i + 1 < argc) {
                prune_worst = std::stoul(argv[++i]);
            } else if (arg == "--eps-search" && i + 1 < argc) {
                eps_search_str = argv[++i];
            } else if (arg == "--flas") {
                use_flas = true;
            } else if (arg == "--flas-metric" && i + 1 < argc) {
                flas_metric_str = argv[++i];
            } else if (arg == "--flas-radius-decay" && i + 1 < argc) {
                flas_radius_decay = std::stof(argv[++i]);
            } else if (arg == "--num-runs" && i + 1 < argc) {
                num_runs = std::stoul(argv[++i]);
            } else if (arg == "--goal-recall" && i + 1 < argc) {
                goal_recall = std::stof(argv[++i]);
            } else if (arg == "--opt-iterations" && i + 1 < argc) {
                opt_iterations = std::stoull(argv[++i]);
            } else if (arg == "--opt-target" && i + 1 < argc) {
                std::string target_str = argv[++i];
                if (target_str == "LowLID") {
                    opt_target = deglib::builder::OptimizationTarget::LowLID;
                } else if (target_str == "HighLID") {
                    opt_target = deglib::builder::OptimizationTarget::HighLID;
                } else if (target_str == "StreamingData_SchemeA") {
                    opt_target = deglib::builder::OptimizationTarget::StreamingData_SchemeA;
                } else if (target_str == "StreamingData_SchemeB") {
                    opt_target = deglib::builder::OptimizationTarget::StreamingData_SchemeB;
                } else if (target_str == "StreamingData_SchemeC") {
                    opt_target = deglib::builder::OptimizationTarget::StreamingData_SchemeC;
                } else if (target_str == "StreamingData_SchemeD") {
                    opt_target = deglib::builder::OptimizationTarget::StreamingData_SchemeD;
                } else if (target_str == "SchemeA") {
                    opt_target = deglib::builder::OptimizationTarget::SchemeA;
                } else if (target_str == "SchemeB") {
                    opt_target = deglib::builder::OptimizationTarget::SchemeB;
                } else {
                    std::fprintf(stderr, "Error: Unknown optimization target '%s'\n", target_str.c_str());
                    return 1;
                }
            } else {
                std::fprintf(stderr, "Warning: Unknown or malformed option '%s'\n", arg.c_str());
            }
        }

        if (build_threads == 0) {
            build_threads = threads;
        }

        std::vector<uint32_t> max_dist_list = sisap_common::parse_list(max_dist_str);
        std::vector<float> eps_search_list = sisap_common::parse_list_f(eps_search_str);

        // Validate --prune-worst
        if (prune_worst >= k_graph) {
            std::fprintf(stderr, "Error: --prune-worst (%u) must be less than --k-graph (%u)\n", prune_worst, (uint32_t)k_graph);
            return 1;
        }

        // Validate --no-recall restrictions
        if (!run_recall) {
            if (output_path.empty()) {
                std::fprintf(stderr, "Error: --output path must be specified when --no-recall is set.\n");
                return 1;
            }
            if (max_dist_list.size() > 1) {
                std::fprintf(stderr, "Error: Multiple parameters for --max-dist are not allowed when --no-recall is set.\n");
                return 1;
            }
        }

        std::setvbuf(stdout, NULL, _IONBF, 0);
        std::setvbuf(stderr, NULL, _IONBF, 0);

        // Validate that data_path has an HDF5 extension
        std::filesystem::path path(data_path);
        std::string ext = path.extension().string();
        if (ext != ".h5" && ext != ".hdf5") {
            std::fprintf(stderr, "Error: Only HDF5 (.h5 or .hdf5) input files are accepted by task2.\n");
            return 1;
        }

        // Resolve FLAS metric string to enum
        FlasMetric flas_metric = FlasMetric::L2;
        if (use_flas) {
            if (flas_metric_str == "ip" || flas_metric_str == "innerproduct") {
                flas_metric = FlasMetric::InnerProduct;
            } else if (flas_metric_str != "l2") {
                std::fprintf(stderr, "Warning: Unknown --flas-metric '%s', using l2\n", flas_metric_str.c_str());
            }
        }

        // Print compiled-in SIMD instruction set info
#ifdef USE_AVX512
        std::fprintf(stderr, "SIMD: AVX-512, AVX2, SSE\n");
#elif defined(USE_AVX)
        std::fprintf(stderr, "SIMD: AVX2, SSE\n");
#elif defined(USE_SSE)
        std::fprintf(stderr, "SIMD: SSE\n");
#else
        std::fprintf(stderr, "SIMD: none (scalar)\n");
#endif

          if (mode == "fp32-build-fp32-explore" || mode == "baseline" || mode == "mode1") {
             return task2::mode_baseline::run(path, threads, build_threads, use_flas, flas_metric, flas_radius_decay, k_graph, k_ext, eps_ext, opt_target, opt_iterations, prune_worst, k_top, num_runs, max_dist_list, eps_search_list, run_recall, goal_recall, output_path, graph_path);
          } else if (mode == "fp16-build-fp16-explore" || mode == "mode2") {
             return task2::mode_fp16_build::run(path, threads, build_threads, use_flas, flas_metric, flas_radius_decay, k_graph, k_ext, eps_ext, opt_target, opt_iterations, prune_worst, k_top, num_runs, max_dist_list, eps_search_list, run_recall, goal_recall, output_path, graph_path);
          } else if (mode == "fp32-build-fp16-explore" || mode == "baseline-fp16" || mode == "mode3") {
             return task2::mode_fp16::run(path, threads, build_threads, use_flas, flas_metric, flas_radius_decay, k_graph, k_ext, eps_ext, opt_target, opt_iterations, prune_worst, k_top, num_runs, max_dist_list, eps_search_list, run_recall, goal_recall, output_path, graph_path);
           } else if (mode == "l2-converted" || mode == "fp32-build-l2-explore" || mode == "mode4") {
              return task2::mode_l2_converted::run(path, threads, build_threads, use_flas, flas_metric, flas_radius_decay, k_graph, k_ext, eps_ext, opt_target, opt_iterations, prune_worst, k_top, num_runs, max_dist_list, eps_search_list, run_recall, goal_recall, output_path, graph_path);
           } else if (mode == "l2-fp16-ip" || mode == "l2-build-fp16-ip-explore" || mode == "mode5") {
              return task2::mode_l2_fp16_ip::run(path, threads, build_threads, use_flas, flas_metric, flas_radius_decay, k_graph, k_ext, eps_ext, opt_target, opt_iterations, prune_worst, k_top, num_runs, max_dist_list, eps_search_list, run_recall, goal_recall, output_path, graph_path);
           } else if (mode == "l2-fp16-l2" || mode == "l2-build-fp16-l2-explore" || mode == "mode6") {
              return task2::mode_l2_fp16::run(path, threads, build_threads, use_flas, flas_metric, flas_radius_decay, k_graph, k_ext, eps_ext, opt_target, opt_iterations, prune_worst, k_top, num_runs, max_dist_list, eps_search_list, run_recall, goal_recall, output_path, graph_path);
            } else if (mode == "l2-fp16-d2" || mode == "l2-build-fp16-d2-explore" || mode == "mode7") {
               return task2::mode_l2_fp16_d2::run(path, threads, build_threads, use_flas, flas_metric, flas_radius_decay, k_graph, k_ext, eps_ext, opt_target, opt_iterations, prune_worst, k_top, num_runs, max_dist_list, eps_search_list, run_recall, goal_recall, output_path, graph_path);
            } else {
               std::fprintf(stderr, "Error: Unknown mode '%s'. Supported modes: mode1, mode2, mode3, mode4, mode5, mode6, mode7\n", mode.c_str());
               return 1;
           }

    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Fatal error: %s\n", ex.what());
        return 1;
    }
}
