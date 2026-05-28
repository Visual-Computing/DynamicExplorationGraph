/**
 * @file task1.cpp
 * @brief EVP Benchmark Task 1 — Entrypoint to dispatch HDF5-based ANN benchmarks to modularized modes.
 *
 * Overview
 * --------
 * This executable implements 7 distinct benchmark modes (modi1–modi7), each exercising a different
 * combination of graph construction strategy (FP16 vs. EVP-quantized), search metric (EVP bits,
 * FP16 inner product, asymmetric FP16-vs-EVP), and optional post-processing (FP16 reranking).
 * All modes accept a single HDF5 input file containing at least the "train" dataset and
 * optionally the "allknn/knns" ground-truth for recall evaluation.
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
 *   modi1  (fp16, fp16-build-fp16-explore)
 *     Builds a SizeBoundedGraph using FP16 features (Metric::FP16InnerProduct) and
 *     explores via FP16 inner-product similarity. Serves as a high-quality baseline.
 *
 *   modi2  (evp-linear, evp-linear-search)
 *     Quantizes all vectors to EVP bits and performs exact brute-force (linear) search.
 *     No graph is built. Serves as an exact similarity baseline.
 *
 *   modi3  (evp-no-rerank, evp-build-evp-explore)
 *     Builds the graph using EVP-quantized features (Metric::EvpBits) and explores
 *     directly using EVP bit similarity. No reranking.
 *
 *   modi4  (evp, evp-build-evp-explore-fp16-rerank)
 *     Builds the EVP graph, explores using EVP bits to retrieve candidates, then
 *     reranks the candidates using exact FP16 inner-product distances.
 *
 *   modi5  (evp-build-fp16-external-search)
 *     Builds the EVP graph, then constructs a ReadOnlyGraphExternal that references
 *     the permuted FP16 data in-place. Search uses FP16 inner product over the
 *     EVP-derived graph topology.
 *
 *   modi6  (evp-asymmetric, evp-build-fp16-asymmetric-search)
 *     Builds the EVP graph, converts to a ReadOnlyGraph with Metric::FP16EvpAsymmetric.
 *     Queries are FP16, database items remain EVP-quantized. No reranking.
 *
 *   modi7  (evp-asymmetric-rerank, evp-build-fp16-asymmetric-search-rerank)
 *     Same asymmetric search as modi6, but adds a candidate reranking phase using
 *     exact FP16 inner-product distances.
 *
 * CLI Usage
 * ---------
 *   deglib_evp_task1.exe <hdf5_file> <mode> [options...]
 *
 *   <mode> accepts:
 *     - The symbolic mode name (e.g. "fp16", "evp", "evp-linear", "evp-asymmetric", etc.)
 *     - The descriptive alias (e.g. "fp16-build-fp16-explore", "evp-build-evp-explore-fp16-rerank")
 *     - The positional identifier "modi1" through "modi7"
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
 *   # FP16 baseline (modi1)
 *   .\deglib_evp_task1.exe dataset.h5 modi1 --max-dist 100
 *
 *   # EVP build + explore + FP16 rerank (modi4)
 *   .\deglib_evp_task1.exe dataset.h5 evp --max-dist 200
 *
 *   # Asymmetric search + rerank, save results (modi7)
 *   .\deglib_evp_task1.exe dataset.h5 modi7 --max-dist 200 --evpK 50 --no-recall --output results.ivecs
 */

#if defined(_WIN32)
    #define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstdio>
#include <string>
#include <filesystem>
#include <iostream>

#include "task1/modi1.h"
#include "task1/modi2.h"
#include "task1/modi3.h"
#include "task1/modi4.h"
#include "task1/modi5.h"
#include "task1/modi6.h"
#include "task1/modi7.h"

int main(int argc, char* argv[]) {
    try {
        std::string data_path;
        std::string mode;
        uint32_t threads = 6;
        uint32_t non_zeros = 512;
        uint32_t k_top = 15;
        uint8_t k_graph = 32;
        uint8_t k_ext = 32;
        float eps_ext = 0.001f;
        bool run_recall = true;
        std::string output_path;
        std::string graph_path;
        uint32_t evpK = 0; // 0 means use standard defaults
        uint32_t max_distance_count = 200;

        if (argc < 3) {
            std::fprintf(stderr, "Usage: %s <hdf5_file_path> <mode_name> [options...]\n", argv[0]);
            std::fprintf(stderr, "Modes:\n");
            std::fprintf(stderr, "  fp16 | fp16-build-fp16-explore | modi1                        : FP16 build + FP16 explore\n");
            std::fprintf(stderr, "  evp-linear | evp-linear-search | modi2                        : EVP quantization + brute-force linear search\n");
            std::fprintf(stderr, "  evp-no-rerank | evp-build-evp-explore | modi3                  : EVP build + EVP explore (no rerank)\n");
            std::fprintf(stderr, "  evp | evp-build-evp-explore-fp16-rerank | modi4               : EVP build + EVP explore + FP16 rerank\n");
            std::fprintf(stderr, "  evp-build-fp16-external-search | modi5                        : EVP build + FP16 external graph search\n");
            std::fprintf(stderr, "  evp-asymmetric | evp-build-fp16-asymmetric-search | modi6     : EVP build + asymmetric FP16-vs-EVP search\n");
            std::fprintf(stderr, "  evp-asymmetric-rerank | evp-build-fp16-asymmetric-search-rerank | modi7\n");
            std::fprintf(stderr, "                                                     : EVP build + asymmetric search + FP16 rerank\n\n");
            std::fprintf(stderr, "Options:\n");
            std::fprintf(stderr, "  --threads <n>      Number of threads (default: 6)\n");
            std::fprintf(stderr, "  --non-zeros <n>    Number of non-zeros for EVP (default: 512)\n");
            std::fprintf(stderr, "  --k-top <n>        Top-K nearest neighbors to retrieve (default: 15)\n");
            std::fprintf(stderr, "  --k-graph <n>      k_graph for graph structure (default: 32)\n");
            std::fprintf(stderr, "  --k-ext <n>        k_ext for graph extension (default: 32)\n");
            std::fprintf(stderr, "  --eps-ext <f>      eps_ext for search expansion (default: 0.001)\n");
            std::fprintf(stderr, "  --no-recall        Skip recall computation\n");
            std::fprintf(stderr, "  --output <path>    Output file path to save retrieved neighbor labels\n");
            std::fprintf(stderr, "  --evpK <n>         Search parameter to override graph search size (defaults: 50 for asym rerank (modi7), std::max(k_top, max_dist) for EVP rerank (modi4))\n");
            std::fprintf(stderr, "  --max-dist <n>     Max distance count for graph exploration (default: 200)\n");
            std::fprintf(stderr, "  --graph <path>     Path to load/save the pre-built graph\n");
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
                evpK = std::stoul(argv[++i]);
            } else if (arg == "--max-dist" && i + 1 < argc) {
                max_distance_count = std::stoul(argv[++i]);
            } else if (arg == "--graph" && i + 1 < argc) {
                graph_path = argv[++i];
            } else {
                std::fprintf(stderr, "Warning: Unknown or malformed option '%s'\n", arg.c_str());
            }
        }

        // Validate that data_path has an HDF5 extension
        std::filesystem::path path(data_path);
        std::string ext = path.extension().string();
        if (ext != ".h5" && ext != ".hdf5") {
            std::fprintf(stderr, "Error: Only HDF5 (.h5 or .hdf5) input files are accepted by task1.\n");
            return 1;
        }

        if (!run_recall && output_path.empty()) {
            std::fprintf(stderr, "Warning: --no-recall is set but no --output file path is specified. Top-K results will not be saved.\n");
        }

        // Print compiled-in SIMD instruction set info
#ifdef USE_AVX512
        std::fprintf(stderr, "SIMD: AVX-512, AVX, SSE\n");
#elif defined(USE_AVX)
        std::fprintf(stderr, "SIMD: AVX, SSE\n");
#elif defined(USE_SSE)
        std::fprintf(stderr, "SIMD: SSE\n");
#else
        std::fprintf(stderr, "SIMD: none (scalar)\n");
#endif

        // Support both old mode names/aliases and modiX names
        if (mode == "fp16-build-fp16-explore" || mode == "fp16" || mode == "modi1") {
            return task1::mode1::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_distance_count, run_recall, output_path, graph_path);
        } else if (mode == "evp-linear-search" || mode == "evp-linear" || mode == "modi2") {
            if (!graph_path.empty()) {
                std::fprintf(stderr, "Warning: --graph-path is not supported by modi2 (EVP linear search) and will be ignored.\n");
            }
            return task1::mode2::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_distance_count, run_recall, output_path);
        } else if (mode == "evp-build-evp-explore" || mode == "evp-no-rerank" || mode == "modi3") {
            return task1::mode3::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_distance_count, run_recall, output_path, graph_path);
        } else if (mode == "evp-build-evp-explore-fp16-rerank" || mode == "evp" || mode == "modi4") {
            return task1::mode4::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_distance_count, evpK, run_recall, output_path, graph_path);
        } else if (mode == "evp-build-fp16-external-search" || mode == "modi5") {
            return task1::mode5::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_distance_count, run_recall, output_path, graph_path);
        } else if (mode == "evp-build-fp16-asymmetric-search" || mode == "evp-asymmetric" || mode == "modi6") {
            return task1::mode6::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_distance_count, run_recall, output_path, graph_path);
        } else if (mode == "evp-build-fp16-asymmetric-search-rerank" || mode == "evp-asymmetric-rerank" || mode == "modi7") {
            return task1::mode7::run(path, threads, non_zeros, k_graph, k_ext, eps_ext, k_top, max_distance_count, evpK, run_recall, output_path, graph_path);
        } else {
            std::fprintf(stderr, "Error: Unknown mode '%s'. Choose a valid benchmark mode (e.g. 'evp', 'evp-no-rerank', etc.) or 'modi1'-'modi7'.\n", mode.c_str());
            return 1;
        }

    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Fatal error: %s\n", ex.what());
        return 1;
    }
}
