/**
 * @file task1.cpp
 * @brief EVP Benchmark Task 1 — Entrypoint to dispatch HDF5-based ANN benchmarks to modularized modes.
 *
 * Overview
 * --------
 * This executable implements 7 distinct benchmark modes (mode1–mode7), each exercising a different
 * combination of graph construction strategy (FP16 vs. EVP-quantized), search metric (EVP bits,
 * FP16 inner product, asymmetric FP16-vs-EVP), and optional post-processing (FP16 reranking).
 * All modes accept a single HDF5 input file (SISAP benchmark files only, e.g. benchmark-dev-wikipedia-bge-m3-small.h5 or benchmark-dev-wikipedia-bge-m3.h5) containing at least the "train" dataset and
 * the "allknn/knns" ground-truth for recall evaluation.
 *
 * Input Format
 * ------------
 * The input HDF5 file must contain:
 *   - "train"       : N × D dataset of FP16 (float16) feature vectors.
 *   - "allknn/knns" : N × K ground-truth nearest-neighbor indices (1-based, int32),
 *                     required when running in recall mode (the default).
 *
 * Output Modes
 * ------------
 * Two mutually exclusive output behaviours are supported:
 *
 *   1. Recall mode (default):
 *      Loads the "allknn/knns" ground truth, runs the benchmark, and prints Recall@K.
 *
 *   2. Save mode (--no-recall --output <path>):
 *      Skips ground-truth loading, runs the search once, and writes the retrieved
 *      neighbor indices to a binary ivecs file (one row per query; each row:
 *      uint32_t count followed by count × uint32_t label values).
 *
 * Benchmark Modes
 * ---------------
 *   mode1  (fp16, fp16-build-fp16-explore)
 *     Builds a SizeBoundedGraph using FP16 features (Metric::FP16InnerProduct) and
 *     explores via FP16 inner-product similarity. Serves as a high-quality baseline.
 *
 *   mode2  (evp-linear, evp-linear-search)
 *     Quantizes all vectors to EVP bits and performs exact brute-force (linear) search.
 *     No graph is built. Serves as an exact similarity baseline.
 *
 *   mode3  (evp, evp-build-evp-explore)
 *     Builds the graph using EVP-quantized features (Metric::EvpBits) and explores
 *     directly using EVP bit similarity. No reranking.
 *
 *   mode4  (evp-rerank, evp-build-evp-explore-fp16-rerank)
 *     Builds the EVP graph, explores using EVP bits to retrieve candidates, then
 *     reranks the candidates using exact FP16 inner-product distances.
 *
 *   mode5  (evp-build-fp16-external-search)
 *     Builds the EVP graph, then constructs a ReadOnlyGraphExternal that references
 *     the permuted FP16 data in-place. Search uses FP16 inner product over the
 *     EVP-derived graph topology.
 *
 *   mode6  (evp-asymmetric, evp-build-fp16-asymmetric-search)
 *     Builds the EVP graph, converts to a ReadOnlyGraph with Metric::FP16EvpAsymmetric.
 *     Queries are FP16, database items remain EVP-quantized. No reranking.
 *
 *   mode7  (evp-asymmetric-rerank, evp-build-fp16-asymmetric-search-rerank)
 *     Same asymmetric search as mode6, but adds a candidate reranking phase using
 *     exact FP16 inner-product distances.
 *
 * CLI Usage
 * ---------
 *   deglib_sisap_task1.exe <hdf5_file> <mode> [options...]
 *
 *   <mode> accepts:
 *     - The symbolic mode name (e.g. "fp16", "evp-rerank", "evp-linear", "evp-asymmetric", etc.)
 *     - The descriptive alias (e.g. "fp16-build-fp16-explore", "evp-build-evp-explore-fp16-rerank")
 *     - The positional identifier "mode1" through "mode7"
 *
 * Key Options
 * -----------
 *   --threads <n>      Parallel thread count (default: 6).
 *   --non-zeros <n>    EVP quantization sparsity parameter (default: 512).
 *   --k-top <n>        Number of nearest neighbors to retrieve (default: 15).
 *   --k-graph <n>      Graph degree / edges per vertex (default: 32).
 *   --k-ext <n>        Builder extension degree (default: 32).
 *   --eps-ext <f>      Builder entry-search expansion coefficient (default: 0.001).
 *   --max-dist <n>     Maximum distance computations per query during exploration (default: 200).
 *   --evpK <n>         Override internal candidate list size (default: auto).
 *   --no-recall        Skip recall computation; requires --output.
 *   --output <path>    Write results to a binary ivecs file.
 *
 * Examples
 * --------
 *   # FP16 baseline (mode1)
 *   .\deglib_sisap_task1.exe dataset.h5 mode1 --max-dist 100
 *
 *   # EVP build + explore + FP16 rerank (mode4)
 *   .\deglib_sisap_task1.exe dataset.h5 evp-rerank --max-dist 200
 *
 *   # Asymmetric search + rerank, save results (mode7)
 *   .\deglib_sisap_task1.exe dataset.h5 mode7 --max-dist 200 --evpK 50 --no-recall --output results.ivecs
 *
 *   # Or run via combined entry point
 *   .\deglib_sisap.exe task1 dataset.h5 mode4 --max-dist 200
 */

#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <string>
#include <filesystem>
#include <iostream>

#include "sisap_common.h"
#include "task1/mode1.h"
#include "task1/mode2.h"
#include "task1/mode3.h"
#include "task1/mode4.h"
#include "task1/mode5.h"
#include "task1/mode6.h"
#include "task1/mode7.h"

#ifdef BUILD_COMBINED_SISAP
int run_task1(int argc, char* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    try {
        std::string data_path;
        std::string mode;
        uint32_t threads = 8;
        uint32_t non_zeros = 600;
        uint32_t k_top = 15;
        uint8_t k_graph = 32;
        uint8_t k_ext = 32;
        float eps_ext = 0.001f;
        bool run_recall = true;
        std::string output_path;
        std::string graph_path;
        std::string evpK_str = "50"; 
        std::string max_dist_str = "200";
        uint32_t prune_worst = 16;
        float goal_recall = 0.8f;

        if (argc < 3) {
            std::fprintf(stderr, "Usage: %s <hdf5_file_path> <mode_name> [options...]\n", argv[0]);
            std::fprintf(stderr, "Modes:\n");
            std::fprintf(stderr, "  fp16 | fp16-build-fp16-explore | mode1                        : FP16 build + FP16 explore\n");
            std::fprintf(stderr, "  evp-linear | evp-linear-search | mode2                        : EVP quantization + brute-force linear search\n");
            std::fprintf(stderr, "  evp | evp-build-evp-explore | mode3                  : EVP build + EVP explore (no rerank)\n");
            std::fprintf(stderr, "  evp-rerank | evp-build-evp-explore-fp16-rerank | mode4               : EVP build + EVP explore + FP16 rerank\n");
            std::fprintf(stderr, "  evp-build-fp16-external-search | mode5                        : EVP build + FP16 external graph search\n");
            std::fprintf(stderr, "  evp-asymmetric | evp-build-fp16-asymmetric-search | mode6     : EVP build + asymmetric FP16-vs-EVP search\n");
            std::fprintf(stderr, "  evp-asymmetric-rerank | evp-build-fp16-asymmetric-search-rerank | mode7\n");
            std::fprintf(stderr, "                                                     : EVP build + asymmetric search + FP16 rerank\n\n");
            std::fprintf(stderr, "Options:\n");
            std::fprintf(stderr, "  --threads <n>      Number of CPU worker threads used for parallel EVP quantization,\n");
            std::fprintf(stderr, "                     even-regular graph construction, and query exploration (default: 6).\n");
            std::fprintf(stderr, "  --non-zeros <n>    EVP Quantization sparsity parameter. Specifies the exact number of non-zero\n");
            std::fprintf(stderr, "                     elements in each quantized sparse vector (default: 600).\n");
            std::fprintf(stderr, "  --k-top <n>        The final number of nearest neighbors (top-K) retrieved per query\n");
            std::fprintf(stderr, "                     and evaluated for recall or written to the output file (default: 15).\n");
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
            std::fprintf(stderr, "  --evpK <list>      Candidate pool size or comma-separated list of sizes. Graph search retrieves `evpK` candidates\n");
            std::fprintf(stderr, "                     which are then reranked using exact FP16 inner product. Used in Mode 4 and Mode 7 (default: 50).\n");
            std::fprintf(stderr, "  --max-dist <list>  Exploration search budget or comma-separated list of budgets. Specifies the maximum number of\n");
            std::fprintf(stderr, "                     distance computations allowed per query. Main parameter to trade search speed for recall (default: 200).\n");
            std::fprintf(stderr, "  --graph <path>     File path to save the pre-built graph to, or load a pre-built graph from\n");
            std::fprintf(stderr, "                     to bypass the construction phase.\n");
            std::fprintf(stderr, "  --prune-worst <n>  Number of worst (least similar) neighbors per vertex to replace with self-loops.\n");
            std::fprintf(stderr, "                     Applies to all modes (default: 16).\n");
            std::fprintf(stderr, "  --goal-recall <f>  Recall threshold. Sweeps will select configuration minimizing search time\n");
            std::fprintf(stderr, "                     among those with recall >= threshold (default: 0.8).\n");
            return 1;
        }

        data_path = argv[1];
        mode = argv[2];

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--threads" && i + 1 < argc) {
                threads = std::stoul(argv[++i]);
            } else if (arg == "--non-zeros" && i + 1 < argc) {
                non_zeros = std::stoul(argv[++i]);
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
            } else if (arg == "--evpK" && i + 1 < argc) {
                evpK_str = argv[++i];
            } else if (arg == "--max-dist" && i + 1 < argc) {
                max_dist_str = argv[++i];
            } else if (arg == "--graph" && i + 1 < argc) {
                graph_path = argv[++i];
            } else if (arg == "--prune-worst" && i + 1 < argc) {
                prune_worst = std::stoul(argv[++i]);
            } else if (arg == "--goal-recall" && i + 1 < argc) {
                goal_recall = std::stof(argv[++i]);
            } else {
                std::fprintf(stderr, "Warning: Unknown or malformed option '%s'\n", arg.c_str());
            }
        }

        std::vector<uint32_t> evpK_list = sisap_common::parse_list(evpK_str);
        std::vector<uint32_t> max_dist_list = sisap_common::parse_list(max_dist_str);

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
            if (max_dist_list.size() > 1 || evpK_list.size() > 1) {
                std::fprintf(stderr, "Error: Multiple parameters for --max-dist or --evpK are not allowed when --no-recall is set.\n");
                return 1;
            }
        }

        // Validate --evpK for reranking modes (Mode 4, Mode 7)
        if (mode == "evp-rerank" || mode == "evp-build-evp-explore-fp16-rerank" || mode == "mode4" ||
            mode == "evp-asymmetric-rerank" || mode == "evp-build-fp16-asymmetric-search-rerank" || mode == "mode7") {
            for (uint32_t evpK_val : evpK_list) {
                if (evpK_val < k_top) {
                    std::fprintf(stderr, "Error: --evpK (%u) must be greater than or equal to --k-top (%u) for reranking modes.\n", evpK_val, k_top);
                    return 1;
                }
                for (uint32_t max_dist_val : max_dist_list) {
                    if (evpK_val > max_dist_val) {
                        std::fprintf(stderr, "Error: --evpK (%u) must be less than or equal to --max-dist (%u) for reranking modes.\n", evpK_val, max_dist_val);
                        return 1;
                    }
                }
            }
        }

        // Validate that data_path has an HDF5 extension
        std::filesystem::path path(data_path);
        std::string ext = path.extension().string();
        if (ext != ".h5" && ext != ".hdf5") {
            std::fprintf(stderr, "Error: Only HDF5 (.h5 or .hdf5) input files are accepted by task1.\n");
            return 1;
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

        // Support both old mode names/aliases and modeX names
        if (mode == "fp16-build-fp16-explore" || mode == "fp16" || mode == "mode1") {
            return task1::mode1::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_dist_list, run_recall, goal_recall, output_path, graph_path, prune_worst);
        } else if (mode == "evp-linear-search" || mode == "evp-linear" || mode == "mode2") {
            if (!graph_path.empty()) {
                std::fprintf(stderr, "Warning: --graph-path is not supported by mode2 (EVP linear search) and will be ignored.\n");
            }
            return task1::mode2::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_dist_list, run_recall, goal_recall, output_path);
        } else if (mode == "evp-build-evp-explore" || mode == "evp" || mode == "mode3") {
            return task1::mode3::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_dist_list, run_recall, goal_recall, output_path, graph_path, prune_worst);
        } else if (mode == "evp-build-evp-explore-fp16-rerank" || mode == "evp-rerank" || mode == "mode4") {
            return task1::mode4::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_dist_list, evpK_list, run_recall, goal_recall, output_path, graph_path, prune_worst);
        } else if (mode == "evp-build-fp16-external-search" || mode == "mode5") {
            return task1::mode5::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_dist_list, run_recall, goal_recall, output_path, graph_path, prune_worst);
        } else if (mode == "evp-build-fp16-asymmetric-search" || mode == "evp-asymmetric" || mode == "mode6") {
            return task1::mode6::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_dist_list, run_recall, goal_recall, output_path, graph_path, prune_worst);
        } else if (mode == "evp-build-fp16-asymmetric-search-rerank" || mode == "evp-asymmetric-rerank" || mode == "mode7") {
            return task1::mode7::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_dist_list, evpK_list, run_recall, goal_recall, output_path, graph_path, prune_worst);
        } else {
            std::fprintf(stderr, "Error: Unknown mode '%s'. Choose a valid benchmark mode (e.g. 'evp-rerank', 'evp', etc.) or 'mode1'-'mode7'.\n", mode.c_str());
            return 1;
        }

    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Fatal error: %s\n", ex.what());
        return 1;
    }
}
