/**
 * @file deglib_phd_bench.cpp
 * @brief Standalone benchmark tool for running benchmarks on existing DEG graphs.
 * 
 * This tool allows running any benchmark (search, explore, graph analysis) on any
 * existing graph file. Output goes to console only (no log files).
 * 
 * Configuration:
 *   Edit the default parameters in main() to change dataset, graph file, and benchmark type.
 *   Command-line options can override numeric parameters.
 * 
 * Usage:
 *   deglib_phd_bench [options]
 * 
 * Options:
 *   --k <value>           - Number of nearest neighbors (default: 100)
 *   --repeat <value>      - Number of test repetitions (default: 1)
 *   --threads <value>     - Number of threads (default: 1)
 *   --eps <values>        - Comma-separated eps values for ANNS
 *   --explore-k <value>   - k for exploration test (default: 1000)
 *   --half-gt             - Use half dataset ground truth
 *   --reachability        - Compute search reachability (expensive)
 *   --reach               - Compute exploration reachability (expensive)
 * 
 * Examples:
 *   deglib_phd_bench                          # Run with defaults from source code
 *   deglib_phd_bench --k 50 --threads 4       # Override k and threads
 *   deglib_phd_bench --reachability           # Also compute reachability
 */

#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>

#include "benchmark.h"
#include "deglib.h"

using namespace deglib::benchmark;

// Parse comma-separated float values
static std::vector<float> parse_eps_values(const std::string& str) {
    std::vector<float> values;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        values.push_back(std::stof(item));
    }
    return values;
}

static void print_usage(const char* program_name) {
    fmt::print("\nUsage: {} [dataset] [graph_file] [benchmark_type] [options]\n\n", program_name);
    fmt::print("Arguments:\n");
    fmt::print("  dataset       - Dataset name (sift1m, deep1m, glove, audio) - default: sift1m\n");
    fmt::print("  graph_file    - Path to the graph file (.deg) - default: auto from dataset\n");
    fmt::print("  benchmark     - Benchmark type: anns, explore, stats, all (default: all)\n\n");
    fmt::print("Options:\n");
    fmt::print("  --k <value>           - Number of nearest neighbors (default: 100)\n");
    fmt::print("  --repeat <value>      - Number of test repetitions (default: 1)\n");
    fmt::print("  --threads <value>     - Number of threads (default: 1)\n");
    fmt::print("  --eps <values>        - Comma-separated eps values for ANNS (default: 0.1,0.12,0.14,0.16,0.18,0.2,0.3)\n");
    fmt::print("  --explore-k <value>   - k for exploration test (default: 1000)\n");
    fmt::print("  --half-gt             - Use half dataset ground truth\n");
    fmt::print("  --reachability        - Compute search reachability (expensive)\n");
    fmt::print("  --reach               - Compute exploration reachability (expensive)\n\n");
    fmt::print("Examples:\n");
    fmt::print("  {}                                       # defaults: sift1m, all benchmarks\n", program_name);
    fmt::print("  {} sift1m path/to/graph.deg\n", program_name);
    fmt::print("  {} sift1m path/to/graph.deg anns --k 100 --eps 0.1,0.2,0.3\n", program_name);
    fmt::print("  {} deep1m path/to/graph.deg stats --reachability\n", program_name);
    fmt::print("  {} glove path/to/graph.deg explore --explore-k 30\n", program_name);
}

int main(int argc, char* argv[]) {
    #if defined(USE_AVX)
        fmt::print("Using AVX2...\n");
    #elif defined(USE_SSE)
        fmt::print("Using SSE...\n");
    #else
        fmt::print("Using arch...\n");
    #endif

    const auto data_path = std::filesystem::path(DATA_PATH);
    
    // ==========================================================================
    // Default parameters - change these in the IDE to run different benchmarks
    // ==========================================================================
    std::string dataset_str = "sift1m";                 // Dataset: sift1m, deep1m, glove, audio
    std::string graph_file = "c:/Data/phd/sift1m/deg/128D_L2_K30_AddK60Eps0.1_LowLID.deg";                        // Graph file path (empty = auto-generate)
    std::string benchmark_type = "stats";                 // Benchmark: anns, explore, stats, all
    uint32_t k = 100;                                   // ANNS k
    uint32_t explore_k = 1000;                          // Exploration k
    uint32_t repeat = 1;                                // Test repetitions
    uint32_t threads = 1;                               // Thread count
    std::vector<float> eps_parameter = { 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f };
    bool use_half_gt = false;                           // Use half dataset ground truth
    bool compute_reachability = false;                  // Compute search reachability (expensive)
    bool compute_reach = false;                         // Compute exploration reachability (expensive)
    // ==========================================================================
    
    // Parse command-line arguments
    // Positional args: [dataset] [graph_file] [benchmark_type]
    // Then optional --key value pairs
    int positional_index = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Check for optional flags first
        if (arg == "--k" && i + 1 < argc) {
            k = std::stoul(argv[++i]);
        } else if (arg == "--explore-k" && i + 1 < argc) {
            explore_k = std::stoul(argv[++i]);
        } else if (arg == "--repeat" && i + 1 < argc) {
            repeat = std::stoul(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoul(argv[++i]);
        } else if (arg == "--eps" && i + 1 < argc) {
            eps_parameter = parse_eps_values(argv[++i]);
        } else if (arg == "--half-gt") {
            use_half_gt = true;
        } else if (arg == "--reachability") {
            compute_reachability = true;
        } else if (arg == "--reach") {
            compute_reach = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg.rfind("--", 0) == 0) {
            // Unknown option starting with --
            fmt::print(stderr, "Unknown option: {}\n", arg);
            print_usage(argv[0]);
            return 1;
        } else {
            // Positional argument
            if (positional_index == 0) {
                dataset_str = arg;
            } else if (positional_index == 1) {
                graph_file = arg;
            } else if (positional_index == 2) {
                benchmark_type = arg;
            } else {
                fmt::print(stderr, "Too many positional arguments: {}\n", arg);
                print_usage(argv[0]);
                return 1;
            }
            positional_index++;
        }
    }
    
    // Parse dataset name
    DatasetName dataset_name = DatasetName::from_string(dataset_str);
    if (!dataset_name.is_valid()) {
        fmt::print(stderr, "Error: Unknown dataset: {}\n", dataset_str);
        fmt::print(stderr, "Valid datasets: sift1m, deep1m, glove, audio\n");
        return 1;
    }
    
    // Create dataset object
    Dataset ds(dataset_name, data_path);
    
    // Validate graph file exists
    if (!std::filesystem::exists(graph_file)) {
        fmt::print(stderr, "Error: Graph file not found: {}\n", graph_file);
        return 1;
    }
    
    // Set up OpenMP threads
    omp_set_num_threads(threads);
    
    fmt::print("\n=== DEG Benchmark Tool ===\n");
    fmt::print("Dataset: {}\n", dataset_str);
    fmt::print("Graph: {}\n", graph_file);
    fmt::print("Benchmark: {}\n", benchmark_type);
    fmt::print("k={}, explore_k={}, repeat={}, threads={}\n", k, explore_k, repeat, threads);
    fmt::print("eps_parameter: {}\n", fmt::join(eps_parameter, ", "));
    fmt::print("use_half_gt: {}, compute_reachability: {}, compute_reach: {}\n\n", 
               use_half_gt, compute_reachability, compute_reach);
    
    // Load graph
    fmt::print("Loading graph: {}\n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    fmt::print("Graph loaded: {} vertices, {} edges per vertex\n\n", graph.size(), graph.getEdgesPerVertex());
    
    // Run requested benchmarks
    bool run_stats = (benchmark_type == "stats" || benchmark_type == "all");
    bool run_anns = (benchmark_type == "anns" || benchmark_type == "all");
    bool run_explore = (benchmark_type == "explore" || benchmark_type == "all");
    
    // Stats / Graph Analysis
    if (run_stats) {
        fmt::print("=== Graph Analysis ===\n");
        deglib::benchmark::analyze_graph(graph, {}, compute_reachability, compute_reach, threads);
        fmt::print("\n");
    }
    
    // ANNS Benchmark
    if (run_anns) {
        fmt::print("=== ANNS Benchmark (k={}) ===\n", k);
        
        // Load query repository
        fmt::print("Loading query repository...\n");
        auto query_repository = ds.load_query();
        fmt::print("Loaded {} query vectors\n", query_repository.size());
        
        // Load ground truth
        fmt::print("Loading ground truth (use_half={})...\n", use_half_gt);
        auto ground_truth = ds.load_groundtruth(k, use_half_gt);
        if (ground_truth.empty()) {
            fmt::print(stderr, "Error: Failed to load ground truth\n");
            return 1;
        }
        fmt::print("Loaded ground truth for {} queries\n", ground_truth.size());
        
        deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, 
            repeat, threads, k, eps_parameter);
        fmt::print("\n");
    }
    
    // Exploration Benchmark
    if (run_explore) {
        fmt::print("=== Exploration Benchmark (k={}) ===\n", explore_k);
        
        // Load entry vertices
        fmt::print("Loading exploration entry vertices...\n");
        auto entry_vertices = ds.load_explore_entry_vertices();
        if (entry_vertices.empty()) {
            fmt::print(stderr, "Error: Failed to load exploration entry vertices\n");
            return 1;
        }
        fmt::print("Loaded {} entry vertices\n", entry_vertices.size());
        
        // Load exploration ground truth
        fmt::print("Loading exploration ground truth...\n");
        auto explore_gt = ds.load_explore_groundtruth(explore_k);
        if (explore_gt.empty()) {
            fmt::print(stderr, "Error: Failed to load exploration ground truth\n");
            return 1;
        }
        fmt::print("Loaded exploration ground truth for {} entries\n", explore_gt.size());
        
        deglib::benchmark::test_graph_explore(graph, entry_vertices, explore_gt, 
            false, repeat, explore_k, threads);
        fmt::print("\n");
    }
    
    fmt::print("=== Benchmark Complete ===\n");
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
    
    return 0;
}
