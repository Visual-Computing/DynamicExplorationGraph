#include <random>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <optional>
#include <filesystem>
#include <atomic>
#include <map>

#include <fmt/core.h>
#include <omp.h>

#include "benchmark.h"
#include "stats.h"
#include "dataset.h"
#include "build.h"
#include "deglib.h"
#include "analysis.h"



// Use types from benchmark.h and dataset.h
using deglib::benchmark::DataStreamType;
using deglib::benchmark::Dataset;
using deglib::benchmark::DatasetName;
using deglib::benchmark::log;

// Test type configurations - each test type only runs if present in config

/**
 * CreateGraphTest: Creates a graph and runs comprehensive tests on it.
 * 
 * This test:
 * 1. Creates a graph with the specified parameters
 * 2. Computes graph statistics (connectivity, regularity, etc.)
 * 3. Tests ANNS with Top-100 recall
 * 4. Tests exploration with Top-1000 from entry vertices
 * 5. Runs k-sweep test for k values from 1 to 1024
 */
struct CreateGraphTest {
    // Graph construction parameters
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::LowLID;
    uint8_t k = 30;
    uint8_t k_ext = 60;
    float eps_ext = 0.1f;
    uint32_t build_threads = 1;
    
    // ANNS test parameters
    uint32_t anns_k = 100;                    // k for ANNS recall test
    uint32_t anns_repeat = 1;                 // Number of test repetitions
    uint32_t anns_threads = 1;                // Threads for ANNS test
    
    // Exploration test parameters  
    uint32_t explore_k = 1000;                // k for exploration test
    uint32_t explore_repeat = 1;              // Number of test repetitions
    uint32_t explore_threads = 1;             // Threads for ANNS test

    // k-sweep parameters (tests recall for k=1,2,4,8,...,1024)
    std::vector<uint32_t> k_sweep_values = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    // Test eps parameters for ANNS benchmark (search)
    std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f};
};

/**
 * OptimizeGraphTest: Optimizes an existing graph using the EvenRegularGraphBuilder.
 * 
 * This test:
 * 1. Loads the graph created by CreateGraphTest
 * 2. Optimizes it using improve_and_test with the specified parameters
 * 3. Saves checkpoints at save_interval iterations in a subdirectory
 * 4. Tests all checkpoint graphs with ANNS recall
 * 5. Logs everything to a single log file in the subdirectory
 */
struct OptimizeGraphTest {
    uint8_t k_opt = 30;                       // Optimization neighborhood size
    float eps_opt = 0.001f;                   // Optimization epsilon for search
    uint8_t i_opt = 5;                        // Optimization path length
    uint64_t total_iterations = 1000000;       // Total optimization iterations
    uint64_t save_interval = 100000;          // Save graph every N iterations
};

/**
 * AllSchemesTest: Tests all 5 optimization schemes with same k.
 * Uses k, k_ext, eps_ext from CreateGraphTest.
 * Each scheme has its own eps_parameter for testing.
 * Replaces LIDComparisonTest.
 */
struct AllSchemesTest {
    // Per-scheme eps parameters for testing (indexed by OptimizationTarget enum)
    std::map<deglib::builder::OptimizationTarget, std::vector<float>> eps_parameter;
};

/**
 * KSweepTest: Sweep different k values for graph construction.
 * Builds graphs in "kScaling" subdirectory.
 */
struct KSweepTest {
    std::vector<uint8_t> k_values = {20, 30, 40, 50, 60, 70, 80, 90};  // k values to test
    std::vector<float> eps_parameter = { 0.01f, 0.1f, 0.12f, 0.16f, 0.2f, 0.3f, 0.4f };
};

/**
 * KExtSweepTest: Sweep different k_ext values for graph construction.
 * Builds graphs in "kExtScaling" subdirectory.
 */
struct KExtSweepTest {
    std::vector<uint8_t> k_ext_values = {30, 40, 50, 60, 90};  // k_ext values to test
    std::vector<float> eps_parameter = { 0.01f, 0.1f, 0.12f, 0.16f, 0.2f, 0.3f, 0.4f };
};

/**
 * EpsExtSweepTest: Sweep different eps_ext values for graph construction.
 * Builds graphs in "epsExtScaling" subdirectory.
 */
struct EpsExtSweepTest {
    std::vector<float> eps_ext_values = {0.0f, 0.05f, 0.1f, 0.2f, 0.3f};  // eps_ext values to test
    std::vector<float> eps_parameter = { 0.01f, 0.1f, 0.12f, 0.16f, 0.2f, 0.3f, 0.4f };
};

/**
 * SizeScalingTest: Build and test graphs with different amounts of data.
 * Builds graphs in "sizeScaling" subdirectory using create_incremental_graphs.
 * Graphs are built incrementally at size_interval intervals.
 * Uses a single log file for all size variants.
 */
struct SizeScalingTest {
    uint32_t size_interval = 100000;  // Build graphs at this interval (e.g., 200k, 400k, ...)
    std::vector<float> eps_parameter = { 0.01f, 0.1f, 0.12f, 0.16f, 0.2f, 0.3f, 0.4f };
};

/**
 * OptScalingTest: Build a random graph and optimize it at different iteration intervals.
 * Builds graphs in "optScaling" subdirectory.
 * First creates a random graph, then optimizes it at specified iteration intervals.
 * All graphs are tested afterwards with a single log file for the directory.
 */
struct OptScalingTest {
    uint8_t k_opt = 30;                       // Optimization neighborhood size
    float eps_opt = 0.001f;                   // Optimization epsilon for search
    uint8_t i_opt = 5;                        // Optimization path length
    uint64_t iteration_interval = 100000;     // Save checkpoint every N iterations
    uint64_t total_iterations = 1000000;      // Total optimization iterations
    std::vector<float> eps_parameter = { 0.01f, 0.1f, 0.12f, 0.16f, 0.2f, 0.3f, 0.4f };
};

/**
 * ThreadScalingTest: Build graphs with different thread counts.
 * Builds graphs in "threadScaling" subdirectory.
 * Each graph has its own log file for detailed build timing analysis.
 * Uses CreateGraphTest parameters for graph construction.
 */
struct ThreadScalingTest {
    std::vector<uint32_t> thread_counts = {1, 2, 4, 8, 16};  // Number of threads to test
};

/**
 * RNGDisabledTest: Build a single graph with RNG pruning disabled.
 * Uses CreateGraphTest parameters for graph construction.
 * Stores graph in normal graph_directory (not a subdirectory).
 */
struct RNGDisabledTest {
};

/**
 * DynamicDataTest: Build graphs with different DataStreamTypes.
 * Builds graphs in "dynamic" subdirectory.
 * Each DataStreamType gets its own graph and log file.
 * Uses CreateGraphTest build parameters with StreamingData optimization target.
 * Ground truth is selected based on DataStreamType (full or half dataset).
 */
struct DynamicDataTest {
    std::vector<DataStreamType> data_stream_types = {
        DataStreamType::AddAll,
        DataStreamType::AddHalf,
        DataStreamType::AddAllRemoveHalf,
        DataStreamType::AddHalfRemoveAndAddOneAtATime
    };
};

// Main dataset configuration
struct DatasetConfig {
    // Dataset identity (using DatasetName class)
    DatasetName dataset_name = DatasetName::Invalid;
    
    // Common settings
    DataStreamType data_stream_type = DataStreamType::AddAll;
    deglib::Metric metric = deglib::Metric::L2;
    
    // ============================================================================
    // Static helper functions for enum to string conversion
    // ============================================================================
    static std::string data_stream_type_str(DataStreamType ds) {
        switch(ds) {
            case DataStreamType::AddAll: return "AddAll";
            case DataStreamType::AddHalf: return "AddHalf";
            case DataStreamType::AddAllRemoveHalf: return "AddAllRemoveHalf";
            case DataStreamType::AddHalfRemoveAndAddOneAtATime: return "AddHalfRemoveAndAddOneAtATime";
            default: return "Unknown";
        }
    }
    
    static std::string metric_str(deglib::Metric m) {
        switch(m) {
            case deglib::Metric::L2: return "L2";
            case deglib::Metric::L2_Uint8: return "L2_Uint8";
            default: return "UnknownMetric";
        }
    }
    
    static std::string optimization_target_str(deglib::builder::OptimizationTarget t) {
        switch(t) {
            case deglib::builder::OptimizationTarget::LowLID: return "LowLID";
            case deglib::builder::OptimizationTarget::HighLID: return "HighLID";
            case deglib::builder::OptimizationTarget::StreamingData: return "StreamingData";
            case deglib::builder::OptimizationTarget::SchemeA: return "SchemeA";
            case deglib::builder::OptimizationTarget::SchemeB: return "SchemeB";
            default: return "UnknownLID";
        }
    }
    
    // ============================================================================
    // Test configurations - always present with default values
    // Modify specific values in get_dataset_config as needed
    // ============================================================================
    CreateGraphTest create_graph;
    OptimizeGraphTest optimize_graph;
    AllSchemesTest all_schemes_test;
    KSweepTest k_sweep_test;
    KExtSweepTest k_ext_sweep_test;
    EpsExtSweepTest eps_ext_sweep_test;
    SizeScalingTest size_scaling_test;
    OptScalingTest opt_scaling_test;
    ThreadScalingTest thread_scaling_test;
    RNGDisabledTest rng_disabled_test;
    DynamicDataTest dynamic_data_test;
};


// Path building utilities for graph files
struct GraphPaths {
    std::filesystem::path graph_dir;
    
    GraphPaths(const Dataset& ds) 
        : graph_dir(ds.data_root() / ds.name() / "deg") {}
    
    // ============================================================================
    // Helper methods to reduce code duplication
    // ============================================================================
private:
    static std::string metric_str(deglib::Metric m) {
        return (m == deglib::Metric::L2) ? "L2" : "L2_Uint8";
    }
    
    // Base name: {dims}D_{metric}_K{k}_AddK{k_ext}Eps{eps_ext}_{scheme}
    std::string base_name(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                          deglib::builder::OptimizationTarget lid) const {
        return fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}", 
                           dims, metric_str(metric), k, k_ext, eps_ext, DatasetConfig::optimization_target_str(lid));
    }
    
    // Opt suffix: _OptK{k_opt}Eps{eps_opt}Path{i_opt}
    static std::string opt_suffix(uint8_t k_opt, float eps_opt, uint8_t i_opt) {
        return fmt::format("_OptK{}Eps{:.4f}Path{}", k_opt, eps_opt, i_opt);
    }
    
public:
    // ============================================================================
    // Main directory
    // ============================================================================
    std::string graph_directory() const {
        return graph_dir.string();
    }
    
    // ============================================================================
    // CreateGraphTest: main graph files
    // ============================================================================
    std::string graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                           deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / (base_name(dims, metric, k, k_ext, eps_ext, lid) + ".deg")).string();
    }
    
    std::string graph_log_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                               deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / (base_name(dims, metric, k, k_ext, eps_ext, lid) + ".log")).string();
    }
    
    // ============================================================================
    // OptimizeGraphTest: optimization subdirectory with checkpoints
    // ============================================================================
    std::string opt_directory(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                              deglib::builder::OptimizationTarget lid, uint8_t k_opt, float eps_opt, uint8_t i_opt) const {
        return (graph_dir / (base_name(dims, metric, k, k_ext, eps_ext, lid) + opt_suffix(k_opt, eps_opt, i_opt))).string();
    }
    
    std::string opt_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                               deglib::builder::OptimizationTarget lid, uint8_t k_opt, float eps_opt, uint8_t i_opt, 
                               uint64_t iterations) const {
        return (std::filesystem::path(opt_directory(dims, metric, k, k_ext, eps_ext, lid, k_opt, eps_opt, i_opt)) 
                / fmt::format("it{}.deg", iterations)).string();
    }
    
    std::string opt_log_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                             deglib::builder::OptimizationTarget lid, uint8_t k_opt, float eps_opt, uint8_t i_opt) const {
        return (std::filesystem::path(opt_directory(dims, metric, k, k_ext, eps_ext, lid, k_opt, eps_opt, i_opt)) / "log.txt").string();
    }
    
    // ============================================================================
    // Scaling test directories
    // ============================================================================
    std::string k_scaling_directory() const       { return (graph_dir / "kScaling").string(); }
    std::string k_ext_scaling_directory() const   { return (graph_dir / "kExtScaling").string(); }
    std::string eps_ext_scaling_directory() const { return (graph_dir / "epsExtScaling").string(); }
    std::string size_scaling_directory() const    { return (graph_dir / "sizeScaling").string(); }
    std::string opt_scaling_directory() const     { return (graph_dir / "optScaling").string(); }
    std::string thread_scaling_directory() const  { return (graph_dir / "threadScaling").string(); }
    std::string dynamic_directory() const         { return (graph_dir / "dynamic").string(); }
    
    // ============================================================================
    // Generic scaling file helpers (used by k_sweep, k_ext_sweep, eps_ext_sweep)
    // ============================================================================
    std::string scaling_graph_file(const std::string& dir, uint32_t dims, deglib::Metric metric, 
                                   uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        return (std::filesystem::path(dir) / (base_name(dims, metric, k, k_ext, eps_ext, lid) + ".deg")).string();
    }
    
    std::string scaling_log_file(const std::string& dir, uint32_t dims, deglib::Metric metric,
                                 uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        return (std::filesystem::path(dir) / (base_name(dims, metric, k, k_ext, eps_ext, lid) + ".log")).string();
    }
    
    // ============================================================================
    // ThreadScalingTest: per-thread graph and log files
    // ============================================================================
    std::string thread_scaling_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                                          deglib::builder::OptimizationTarget lid, uint32_t threads) const {
        return (std::filesystem::path(thread_scaling_directory()) 
                / fmt::format("{}_T{}.deg", base_name(dims, metric, k, k_ext, eps_ext, lid), threads)).string();
    }
    
    std::string thread_scaling_log_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext,
                                        deglib::builder::OptimizationTarget lid, uint32_t threads) const {
        return (std::filesystem::path(thread_scaling_directory()) 
                / fmt::format("{}_T{}.log", base_name(dims, metric, k, k_ext, eps_ext, lid), threads)).string();
    }
    
    // ============================================================================
    // RNGDisabledTest: NoRNG graph and log files in main directory
    // ============================================================================
    std::string rng_disabled_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext,
                                        deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / fmt::format("{}_NoRNG.deg", base_name(dims, metric, k, k_ext, eps_ext, lid))).string();
    }
    
    std::string rng_disabled_log_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext,
                                      deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / fmt::format("{}_NoRNG.log", base_name(dims, metric, k, k_ext, eps_ext, lid))).string();
    }
    
    // ============================================================================
    // DynamicDataTest: per-DataStreamType graph and log files with opt params
    // ============================================================================
    std::string dynamic_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext,
                                   uint8_t k_opt, float eps_opt, uint8_t i_opt, DataStreamType ds_type) const {
        std::string ds_str = DatasetConfig::data_stream_type_str(ds_type);
        return (std::filesystem::path(dynamic_directory()) 
                / fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_StreamingData{}_{}_.deg", 
                              dims, metric_str(metric), k, k_ext, eps_ext, opt_suffix(k_opt, eps_opt, i_opt), ds_str)).string();
    }
    
    std::string dynamic_log_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext,
                                 uint8_t k_opt, float eps_opt, uint8_t i_opt, DataStreamType ds_type) const {
        std::string ds_str = DatasetConfig::data_stream_type_str(ds_type);
        return (std::filesystem::path(dynamic_directory()) 
                / fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_StreamingData{}_{}_.log", 
                              dims, metric_str(metric), k, k_ext, eps_ext, opt_suffix(k_opt, eps_opt, i_opt), ds_str)).string();
    }
    
    // ============================================================================
    // SizeScalingTest: incremental size graph files
    // ============================================================================
    std::string size_scaling_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                                        deglib::builder::OptimizationTarget lid, uint32_t size) const {
        return (std::filesystem::path(size_scaling_directory()) 
                / fmt::format("{}_N{}.deg", base_name(dims, metric, k, k_ext, eps_ext, lid), size)).string();
    }
    
    std::string size_scaling_log_file() const {
        return (std::filesystem::path(size_scaling_directory()) / "log.txt").string();
    }
    
    // ============================================================================
    // OptScalingTest: random graph + optimized checkpoints
    // ============================================================================
    std::string opt_scaling_random_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k) const {
        return (std::filesystem::path(opt_scaling_directory()) 
                / fmt::format("{}D_{}_K{}_random.deg", dims, metric_str(metric), k)).string();
    }
    
    std::string opt_scaling_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, 
                                       uint8_t k_opt, float eps_opt, uint8_t i_opt, uint64_t iterations) const {
        return (std::filesystem::path(opt_scaling_directory()) 
                / fmt::format("{}D_{}_K{}{}_it{}.deg", dims, metric_str(metric), k, opt_suffix(k_opt, eps_opt, i_opt), iterations)).string();
    }
    
    std::string opt_scaling_log_file() const {
        return (std::filesystem::path(opt_scaling_directory()) / "log.txt").string();
    }
};

// Dataset configuration factory
// All test structs have default values, just modify what's different per dataset
static DatasetConfig get_dataset_config(const DatasetName& dataset_name) {
    DatasetConfig conf{};
    conf.dataset_name = dataset_name;

    if (dataset_name == DatasetName::SIFT1M) {
        conf.create_graph.eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f };

        conf.optimize_graph.total_iterations = 2000000;

        conf.all_schemes_test.eps_parameter = {
            { deglib::builder::OptimizationTarget::LowLID,    { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f } },
            { deglib::builder::OptimizationTarget::HighLID,   { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f } },
            { deglib::builder::OptimizationTarget::SchemeA,   { 0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f } },
            { deglib::builder::OptimizationTarget::SchemeB,   { 0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f } }
        };
        
    } else if (dataset_name == DatasetName::GLOVE) {
        std::vector<float> eps = { 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f, 1.2f };
        conf.create_graph.lid = deglib::builder::OptimizationTarget::HighLID;
        conf.create_graph.eps_parameter = eps;
        
    } else if (dataset_name == DatasetName::DEEP1M) {
        std::vector<float> eps = { 0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f };
        conf.create_graph.eps_parameter = eps;
        
    } else if (dataset_name == DatasetName::AUDIO) {
        std::vector<float> eps = { 0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f };
        conf.create_graph.k = 20;
        conf.create_graph.k_ext = 40;
        conf.create_graph.anns_repeat = 50;
        conf.create_graph.eps_parameter = eps;
        conf.optimize_graph.k_opt = 20;
    }

    return conf;
}

int main(int argc, char *argv[]) {
    log("Testing ...\n");

    #if defined(USE_AVX)
        log("use AVX2  ...\n");
    #elif defined(USE_SSE)
        log("use SSE  ...\n");
    #else
        log("use arch  ...\n");
    #endif

    const auto data_path = std::filesystem::path(DATA_PATH);
    log("data_path {} \n", data_path.string());

    // Parse command-line arguments
    // Usage: deglib_phd <dataset> [test_type] [--run]
    DatasetName ds_name = DatasetName::SIFT1M;
    std::string test_type_arg = "create_graph";
    bool do_run = true;
    
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "help" || arg == "--help") {
            log("Usage: deglib_phd <dataset> [test_type] [--run]\n");
            log("Datasets: sift1m, deep1m, audio, glove\n");
            log("Test types:\n");
            log("  create_graph    - Build graph, run stats, ANNS, explore, k-sweep\n");
            log("  optimize_graph  - Optimize existing graph\n");
            log("  dynamic_data    - Build graphs with different DataStreamTypes\n");
            log("  all_schemes     - Test all 5 OptimizationTargets\n");
            log("  k_sweep         - Sweep k values for graph construction\n");
            log("  k_ext_sweep     - Sweep k_ext values for graph construction\n");
            log("  eps_ext_sweep   - Sweep eps_ext values for graph construction\n");
            log("  size_scaling    - Build/test graphs with different data sizes (incremental)\n");
            log("  opt_scaling     - Build random graph, optimize at intervals, test\n");
            log("  thread_scaling  - Build graphs with different thread counts\n");
            log("  rng_disabled    - Build graph with RNG pruning disabled\n");
            log("  all             - Run all available tests\n");
            log("Options: --run  Actually execute tests (otherwise just show config)\n");
            return 0;
        }
        if(arg == "--run") { do_run = true; continue; }
        
        // Try to parse as dataset name
        auto parsed_ds = DatasetName::from_string(arg);
        if (parsed_ds.is_valid()) {
            ds_name = parsed_ds;
        } else if(arg == "create_graph" || arg == "optimize_graph" || arg == "dynamic_data" || 
                arg == "all_schemes" || arg == "k_sweep" || arg == "k_ext_sweep" || 
                arg == "eps_ext_sweep" || arg == "size_scaling" || arg == "opt_scaling" || 
                arg == "thread_scaling" || arg == "rng_disabled" || arg == "all") {
            test_type_arg = arg;
        }
    }

    // Create Dataset object (combines name with data_path)
    Dataset ds(ds_name, data_path);
    
    // Get dataset config and create path utilities
    auto config = get_dataset_config(ds_name);
    GraphPaths graph_paths(ds);  // GraphPaths for graph files

    // Print config overview (always to console)
    log("\n=== Dataset: {} ===\n", ds.name());
    log("Repository file: {}\n", ds.base_file());
    log("Query file: {}\n", ds.query_file());
    log("Exploration query file: {}\n", ds.explore_query_file());
    log("Graph directory: {}\n", graph_paths.graph_directory());
    log("Ground truth (full): {}\n", ds.groundtruth_file_full());
    log("Ground truth (half): {}\n", ds.groundtruth_file_half());
    log("Metric: {}\n", DatasetConfig::metric_str(config.metric));
    
    // List available test types for this dataset
    log("\nAvailable test types for '{}':\n", ds.name());
    {
        const auto& cg = config.create_graph;
        log(" - create_graph (k={}, k_ext={}, eps_ext={:.2f}, lid={})\n", 
            cg.k, cg.k_ext, cg.eps_ext, DatasetConfig::optimization_target_str(cg.lid));
        log("     ANNS: k={}, repeat={}, threads={}\n", cg.anns_k, cg.anns_repeat, cg.anns_threads);
        log("     Explore: k={}\n", cg.explore_k);
        log("     k-sweep: [{}]\n", fmt::join(cg.k_sweep_values, ", "));
    }
    log(" - optimize_graph (k_opt={}, eps_opt={:.4f}, i_opt={}, total_it={}, save_interval={})\n",
        config.optimize_graph.k_opt, config.optimize_graph.eps_opt, config.optimize_graph.i_opt,
        config.optimize_graph.total_iterations, config.optimize_graph.save_interval);
    
    // Additional test types
    log(" - all_schemes (tests 5 OptimizationTargets)\n");
    log(" - k_sweep (k_values=[{}], dir=kScaling)\n", 
        fmt::join(config.k_sweep_test.k_values, ", "));
    log(" - k_ext_sweep (k_ext_values=[{}], dir=kExtScaling)\n",
        fmt::join(config.k_ext_sweep_test.k_ext_values, ", "));
    log(" - eps_ext_sweep (eps_ext_values=[{}], dir=epsExtScaling)\n",
        fmt::join(config.eps_ext_sweep_test.eps_ext_values, ", "));
    log(" - size_scaling (interval={}, dir=sizeScaling)\n",
        config.size_scaling_test.size_interval);
    {
        const auto& os = config.opt_scaling_test;
        const auto& og = config.optimize_graph;
        log(" - opt_scaling (interval={}, total={}, k_opt={}, dir=optScaling)\n",
            os.iteration_interval, os.total_iterations, og.k_opt);
    }
    log(" - thread_scaling (thread_counts=[{}], dir=threadScaling)\n",
        fmt::join(config.thread_scaling_test.thread_counts, ", "));
    log(" - rng_disabled (uses create_graph params, RNG=off)\n");
    log(" - dynamic_data (DataStreamTypes, StreamingData opt, dir=dynamic)\n");

    // Load data once if we're running tests
    std::unique_ptr<deglib::StaticFeatureRepository> base_repository;
    std::unique_ptr<deglib::StaticFeatureRepository> query_repository;
    uint32_t dims = 0;
    
    if(do_run) {
        // Setup dataset (downloads, extracts, generates ground truth if needed)
        log("\nSetting up dataset...\n");
        if (!deglib::benchmark::setup_dataset(ds)) {
            log("ERROR: Failed to setup dataset {}\n", ds.name());
            return 1;
        }
        
        log("\nLoading data...\n");
        
        // Ensure graph directory exists
        deglib::benchmark::ensure_directory(graph_paths.graph_directory());
        
        // Load repositories using Dataset convenience methods
        base_repository = std::make_unique<deglib::StaticFeatureRepository>(ds.load_base());
        query_repository = std::make_unique<deglib::StaticFeatureRepository>(ds.load_query());
        dims = base_repository->dims();
        
        log("Loaded {} features with {} dimensions\n", base_repository->size(), dims);
        log("Loaded {} queries\n", query_repository->size());
    }

    // Helper: determine if we need half dataset ground truth
    const bool use_half_gt = (config.data_stream_type != DataStreamType::AddAll);

    // Execute tests based on test_type_arg
    bool run_all = (test_type_arg == "all");
    
    // CREATE_GRAPH test
    if(run_all || test_type_arg == "create_graph") {
        const auto& cg = config.create_graph;
        
        log("\n=== CREATE_GRAPH Test ===\n");
        log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
            cg.k, cg.k_ext, cg.eps_ext, DatasetConfig::optimization_target_str(cg.lid), cg.build_threads);
        
        if(do_run && base_repository) {
            std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string log_path = graph_paths.graph_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            
            // Set up log file for this graph (all tests logged here)
            deglib::benchmark::set_log_file(log_path, true);
            
            // Build graph if it doesn't exist
            if(std::filesystem::exists(graph_path)) {
                log("Graph already exists: {}\n", graph_path);
            } else {
                log("\n=== Building Graph ===\n");
                log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                    cg.k, cg.k_ext, cg.eps_ext, DatasetConfig::optimization_target_str(cg.lid), cg.build_threads);
                log("Output graph: {}\n", graph_path);
                
                deglib::benchmark::create_graph(*base_repository, config.data_stream_type, graph_path, 
                    config.metric, cg.lid, cg.k, cg.k_ext, cg.eps_ext, 0, 0, 0, cg.build_threads);
            }
            
            // Run comprehensive tests on the graph (existing or newly created)
            if(std::filesystem::exists(graph_path)) {
                const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                log("Graph loaded: {} vertices\n", graph.size());
                
                // 1. Compute and log graph stats
                log("\n--- Graph Statistics ---\n");
                auto stats = deglib::benchmark::collect_graph_stats(graph);
                deglib::benchmark::log_graph_stats(stats);
                
                // 2. ANNS Test with Top-k
                log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
                auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
                deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                    cg.anns_repeat, cg.anns_threads, cg.anns_k, cg.eps_parameter);
                
                // 3. Exploration Test
                log("\n--- Exploration Test (k={}) ---\n", cg.explore_k);
                auto entry_vertices = ds.load_explore_entry_vertices();
                auto explore_gt = ds.load_explore_groundtruth(cg.explore_k);
                deglib::benchmark::test_graph_explore(graph, entry_vertices, explore_gt, 
                    false, cg.explore_repeat, cg.explore_k, cg.explore_threads);
                
                // 4. k-Sweep Test
                log("\n--- k-Sweep Test ---\n");
                for(uint32_t k_val : cg.k_sweep_values) {
                    log("\n-- k={} --\n", k_val);
                    auto gt_k = ds.load_groundtruth(k_val, use_half_gt);
                    deglib::benchmark::test_graph_anns(graph, *query_repository, gt_k, 
                        cg.anns_repeat, cg.anns_threads, k_val, cg.eps_parameter);
                }
            }
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    // OPTIMIZE_GRAPH test (optimize graph using improve_and_test, produce checkpoints)
    if(run_all || test_type_arg == "optimize_graph") {
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;
        
        log("\n=== OPTIMIZE_GRAPH Test ===\n");
        log("Settings: k_opt={}, eps_opt={:.4f}, i_opt={}, total_iterations={}, save_interval={}\n",
            og.k_opt, og.eps_opt, og.i_opt, og.total_iterations, og.save_interval);
        
        if(do_run && base_repository && query_repository) {
            std::string input_graph = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string opt_dir = graph_paths.opt_directory(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, og.k_opt, og.eps_opt, og.i_opt);
            std::string log_path = graph_paths.opt_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, og.k_opt, og.eps_opt, og.i_opt);
            
            // Create optimization subdirectory
            deglib::benchmark::ensure_directory(opt_dir);
            
            // Set up log file for all optimization runs
            deglib::benchmark::set_log_file(log_path, false);
            log("\n=== OPTIMIZE_GRAPH Test ===\n");
            log("Input graph: {}\n", input_graph);
            log("Output directory: {}\n", opt_dir);
            log("Settings: k_opt={}, eps_opt={:.4f}, i_opt={}, total_iterations={}, save_interval={}\n",
                og.k_opt, og.eps_opt, og.i_opt, og.total_iterations, og.save_interval);
            
            // Check if source graph exists
            if(!std::filesystem::exists(input_graph)) {
                log("ERROR: Source graph does not exist: {}\n", input_graph);
            } else {
                // Load ground truth for testing during optimization
                auto ground_truth = ds.load_groundtruth(cg.anns_k, false);
                
                // Check if final checkpoint already exists
                std::string final_checkpoint = graph_paths.opt_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, 
                                                                           cg.lid, og.k_opt, og.eps_opt, og.i_opt, og.total_iterations);
                
                std::vector<std::pair<std::string, uint64_t>> created_files;
                
                if(std::filesystem::exists(final_checkpoint)) {
                    log("All checkpoints already exist, skipping optimization phase\n");
                    // Collect existing checkpoint files for testing
                    for(uint64_t it = og.save_interval; it <= og.total_iterations; it += og.save_interval) {
                        std::string cp = graph_paths.opt_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, 
                                                                     cg.lid, og.k_opt, og.eps_opt, og.i_opt, it);
                        if(std::filesystem::exists(cp)) {
                            created_files.push_back({cp, it});
                        }
                    }
                } else {
                    // Load the source graph and optimize it
                    log("\n--- Loading source graph ---\n");
                    auto graph = deglib::graph::load_sizebounded_graph(input_graph.c_str());
                    log("Loaded graph: {} vertices\n", graph.size());
                    
                    // Generate base name for checkpoint files
                    std::string metric_str = (config.metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
                    std::string graph_name_base = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}", 
                                                              dims, metric_str, cg.k, cg.k_ext, cg.eps_ext, 
                                                              DatasetConfig::optimization_target_str(cg.lid));
                    
                    // Run optimization with checkpoints
                    log("\n--- Optimizing graph ---\n");
                    created_files = deglib::benchmark::improve_and_test(
                        graph, opt_dir, graph_name_base,
                        og.k_opt, og.eps_opt, og.i_opt,
                        og.save_interval, og.total_iterations,
                        *query_repository, ground_truth, cg.anns_k);
                    log("Created {} checkpoint graphs\n", created_files.size());
                }
                
                // Phase 2: Test all checkpoint graphs with full ANNS test
                log("\n--- Testing all checkpoints ---\n");
                for(const auto& [cp_path, iterations] : created_files) {
                    log("\n=== Checkpoint: iterations={} ===\n", iterations);
                    log("Graph: {}\n", cp_path);
                    
                    if(std::filesystem::exists(cp_path)) {
                        const auto cp_graph = deglib::graph::load_readonly_graph(cp_path.c_str());
                        
                        // Graph statistics
                        auto stats = deglib::benchmark::collect_graph_stats(cp_graph);
                        deglib::benchmark::log_graph_stats(stats);
                        
                        // ANNS Test
                        log("\n-- ANNS Test (k={}) --\n", cg.anns_k);
                        deglib::benchmark::test_graph_anns(cp_graph, *query_repository, ground_truth, 
                            cg.anns_repeat, cg.anns_threads, cg.anns_k, cg.eps_parameter);
                    } else {
                        log("Checkpoint file not found: {}\n", cp_path);
                    }
                }
            }
            
            deglib::benchmark::reset_log_to_console();
            log("OPTIMIZE_GRAPH: Log written to: {}\n", log_path);
        }
    }
    
    // ALL_SCHEMES test (tests all 5 OptimizationTargets with CreateGraphTest parameters)
    if(run_all || test_type_arg == "all_schemes") {
        const auto& as = config.all_schemes_test;
        const auto& cg = config.create_graph;
        
        log("\n=== ALL_SCHEMES Test ===\n");
        log("Testing all 5 OptimizationTargets with k={}, k_ext={}, eps_ext={:.2f}\n", cg.k, cg.k_ext, cg.eps_ext);
        
        if(do_run && base_repository && query_repository) {
            // Load ground truth for this test
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);
            
            std::vector<deglib::builder::OptimizationTarget> all_targets = {
                deglib::builder::OptimizationTarget::LowLID,
                deglib::builder::OptimizationTarget::HighLID,
                deglib::builder::OptimizationTarget::SchemeA,
                deglib::builder::OptimizationTarget::SchemeB
            };
            
            for(auto lid : all_targets) {
                std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, lid);
                std::string log_path = graph_paths.graph_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, lid);
                deglib::benchmark::set_log_file(log_path, true);
                
                log("\n=== ALL_SCHEMES Test: {} ===\n", DatasetConfig::optimization_target_str(lid));
                log("Graph: {}\n", graph_path);
                
                // Get per-scheme eps_parameter, fall back to CreateGraphTest default
                auto eps_it = as.eps_parameter.find(lid);
                const auto& scheme_eps = (eps_it != as.eps_parameter.end()) ? eps_it->second : cg.eps_parameter;
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, scheme_eps);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("{}: Log written to: {}\n", DatasetConfig::optimization_target_str(lid), log_path);
            }
        }
    }
    
    // K_SWEEP test (sweep k values for graph construction in kScaling directory)
    if(run_all || test_type_arg == "k_sweep") {
        const auto& ks = config.k_sweep_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.k_scaling_directory();
        
        log("\n=== K_SWEEP Test ===\n");
        log("Testing k values: [{}]\n", fmt::join(ks.k_values, ", "));
        log("Directory: {}\n", scaling_dir);
        
        if(do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);
            
            // Load ground truth once for all k tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
            
            for(uint8_t k : ks.k_values) {
                uint8_t k_ext = k * 2;  // k_ext scales with k
                std::string graph_path = graph_paths.scaling_graph_file(scaling_dir, dims, config.metric, k, k_ext, cg.eps_ext, cg.lid);
                std::string log_path = graph_paths.scaling_log_file(scaling_dir, dims, config.metric, k, k_ext, cg.eps_ext, cg.lid);
                    
                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== K_SWEEP Test: k={} ===\n", k);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, ks.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("k={}: Log written to: {}\n", k, log_path);
            }
        }
    }
    
    // K_EXT_SWEEP test (sweep k_ext values for graph construction in kExtScaling directory)
    if(run_all || test_type_arg == "k_ext_sweep") {
        const auto& kes = config.k_ext_sweep_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.k_ext_scaling_directory();
        
        log("\n=== K_EXT_SWEEP Test ===\n");
        log("Testing k_ext values: [{}]\n", fmt::join(kes.k_ext_values, ", "));
        log("Directory: {}\n", scaling_dir);
        
        if(do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);
            
            // Load ground truth once for all k_ext tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
            
            for(uint8_t k_ext : kes.k_ext_values) {
                std::string graph_path = graph_paths.scaling_graph_file(scaling_dir, dims, config.metric, cg.k, k_ext, cg.eps_ext, cg.lid);
                std::string log_path = graph_paths.scaling_log_file(scaling_dir, dims, config.metric, cg.k, k_ext, cg.eps_ext, cg.lid);
                    
                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== K_EXT_SWEEP Test: k_ext={} ===\n", k_ext);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, kes.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("k_ext={}: Log written to: {}\n", k_ext, log_path);
            }
        }
    }
    
    // EPS_EXT_SWEEP test (sweep eps_ext values for graph construction in epsExtScaling directory)
    if(run_all || test_type_arg == "eps_ext_sweep") {
        const auto& ees = config.eps_ext_sweep_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.eps_ext_scaling_directory();
        
        log("\n=== EPS_EXT_SWEEP Test ===\n");
        log("Testing eps_ext values: [{}]\n", fmt::join(ees.eps_ext_values, ", "));
        log("Directory: {}\n", scaling_dir);
        
        if(do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);
            
            // Load ground truth once for all eps_ext tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
            
            for(float eps_ext : ees.eps_ext_values) {
                std::string graph_path = graph_paths.scaling_graph_file(scaling_dir, dims, config.metric, cg.k, cg.k_ext, eps_ext, cg.lid);
                std::string log_path = graph_paths.scaling_log_file(scaling_dir, dims, config.metric, cg.k, cg.k_ext, eps_ext, cg.lid);
                    
                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== EPS_EXT_SWEEP Test: eps_ext={:.2f} ===\n", eps_ext);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, ees.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("eps_ext={:.2f}: Log written to: {}\n", eps_ext, log_path);
            }
        }
    }
    
    // SIZE_SCALING test (build and test graphs with different data sizes using incremental building)
    if(run_all || test_type_arg == "size_scaling") {
        const auto& ss = config.size_scaling_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.size_scaling_directory();
        std::string log_path = graph_paths.size_scaling_log_file();
        
        log("\n=== SIZE_SCALING Test ===\n");
        log("Size interval: {}\n", ss.size_interval);
        log("Directory: {}\n", scaling_dir);
        log("Log file: {}\n", log_path);
        
        if(do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);
            
            // Set log file for all size scaling operations
            deglib::benchmark::set_log_file(log_path, false);
            log("\n=== SIZE_SCALING Test ===\n");
            log("Size interval: {}\n", ss.size_interval);
            
            // Generate base name for graph files
            std::string metric_str = (config.metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
            std::string scheme = DatasetConfig::optimization_target_str(cg.lid);
            std::string graph_name_base = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}", 
                                                       dims, metric_str, cg.k, cg.k_ext, cg.eps_ext, scheme);
            
            // Phase 1: Build all graphs incrementally using create_incremental_graphs
            log("\n--- Phase 1: Building graphs incrementally ---\n");
            
            // Check if we need to build (no graphs exist yet)
            bool needs_build = true;
            uint32_t total_size = uint32_t(base_repository->size());
            for(uint32_t size = ss.size_interval; size <= total_size; size += ss.size_interval) {
                std::string size_str = std::to_string(size / 1000) + "k";
                std::string graph_path = fmt::format("{}/{}_{}.deg", scaling_dir, graph_name_base, size_str);
                if(std::filesystem::exists(graph_path)) {
                    log("Graph for size={} already exists: {}\n", size, graph_path);
                    needs_build = false;  // At least one exists, skip full rebuild
                }
            }
            
            if(needs_build) {
                auto created_files = deglib::benchmark::create_incremental_graphs(
                    *base_repository, scaling_dir, graph_name_base, ss.size_interval,
                    config.metric, cg.lid, cg.k, cg.k_ext, cg.eps_ext, 
                    30, 0.001f, 5,  // k_opt, eps_opt, i_opt (defaults)
                    cg.build_threads);
                log("Created {} incremental graphs\n", created_files.size());
            } else {
                log("Some graphs exist, skipping incremental build. Delete existing graphs to rebuild.\n");
            }
            
            // Phase 2: Test all graphs
            log("\n--- Phase 2: Testing graphs ---\n");
            for(uint32_t size = ss.size_interval; size <= total_size; size += ss.size_interval) {
                std::string size_str = std::to_string(size / 1000) + "k";
                std::string graph_path = fmt::format("{}/{}_{}.deg", scaling_dir, graph_name_base, size_str);
                log("\n=== SIZE_SCALING Test: size={} ===\n", size);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    
                    // Load ground truth (full dataset, queries search against subset)
                    // Note: For size scaling, we use full query set against partial graph
                    auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
                    
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, ss.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
            }
            
            deglib::benchmark::reset_log_to_console();
            log("SIZE_SCALING: Log written to: {}\n", log_path);
        }
    }
    
    // OPT_SCALING test (build random graph, optimize at intervals, test all)
    if(run_all || test_type_arg == "opt_scaling") {
        const auto& os = config.opt_scaling_test;
        const auto& og = config.optimize_graph;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.opt_scaling_directory();
        std::string log_path = graph_paths.opt_scaling_log_file();
        
        log("\n=== OPT_SCALING Test ===\n");
        log("Iteration interval: {}, total: {}\n", os.iteration_interval, os.total_iterations);
        log("Optimization: k_opt={}, eps_opt={:.4f}, i_opt={}\n", og.k_opt, og.eps_opt, og.i_opt);
        log("Directory: {}\n", scaling_dir);
        log("Log file: {}\n", log_path);
        
        if(do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);
            
            // Set log file for all opt scaling operations
            deglib::benchmark::set_log_file(log_path, false);
            log("\n=== OPT_SCALING Test ===\n");
            log("Iteration interval: {}, total: {}\n", os.iteration_interval, os.total_iterations);
            log("Optimization: k_opt={}, eps_opt={:.4f}, i_opt={}\n", og.k_opt, og.eps_opt, og.i_opt);
            
            // Random graph path
            std::string random_graph_path = graph_paths.opt_scaling_random_graph_file(dims, config.metric, cg.k);
            
            // Generate base name for optimized graph files
            std::string metric_str = (config.metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
            std::string graph_name_base = fmt::format("{}D_{}_K{}_OptK{}Eps{:.4f}Path{}", 
                                                       dims, metric_str, cg.k, og.k_opt, og.eps_opt, og.i_opt);
            
            // Phase 1: Build random graph
            log("\n--- Phase 1: Building random graph ---\n");
            log("Random graph: {}\n", random_graph_path);
            
            bool random_graph_built = std::filesystem::exists(random_graph_path);
            if(random_graph_built) {
                log("Random graph already exists, skipping build\n");
            } else {
                // Build random graph using create_random_exploration_graph (returns graph)
                auto random_graph = deglib::benchmark::create_random_graph(
                    *base_repository, config.metric, cg.k);
                random_graph.saveGraph(random_graph_path.c_str());
                log("Saved random graph: {}\n", random_graph_path);
                random_graph_built = true;
            }
            
            // Phase 2: Optimize graph at each iteration checkpoint
            log("\n--- Phase 2: Optimizing graph at intervals ---\n");
            
            // Check if all checkpoint graphs exist
            bool all_checkpoints_exist = true;
            for(uint64_t iterations = os.iteration_interval; iterations <= os.total_iterations; iterations += os.iteration_interval) {
                std::string opt_graph_path = graph_paths.opt_scaling_graph_file(dims, config.metric, cg.k, og.k_opt, og.eps_opt, og.i_opt, iterations);
                if(!std::filesystem::exists(opt_graph_path)) {
                    all_checkpoints_exist = false;
                    break;
                }
            }
            
            if(all_checkpoints_exist) {
                log("All checkpoint graphs exist, skipping optimization\n");
            } else if(random_graph_built) {
                // Load random graph and optimize it
                auto graph = deglib::graph::load_sizebounded_graph(random_graph_path.c_str());
                log("Loaded random graph: {} vertices\n", graph.size());
                
                // Load ground truth for recall testing during optimization
                auto gt_for_opt = ds.load_groundtruth(cg.anns_k, use_half_gt);
                
                // Optimize and save checkpoints with recall testing
                auto created_files = deglib::benchmark::improve_and_test(
                    graph, scaling_dir, graph_name_base,
                    os.k_opt, os.eps_opt, os.i_opt,
                    os.iteration_interval, os.total_iterations,
                    *query_repository, gt_for_opt, cg.anns_k);
                log("Created {} optimized checkpoint graphs\n", created_files.size());
            }
            
            // Phase 3: Test all optimized graphs
            log("\n--- Phase 3: Testing graphs ---\n");
            auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
            
            // Test random graph (iteration 0)
            log("\n=== OPT_SCALING Test: iterations=0 (random) ===\n");
            log("Graph: {}\n", random_graph_path);
            if(std::filesystem::exists(random_graph_path)) {
                const auto graph = deglib::graph::load_readonly_graph(random_graph_path.c_str());
                deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                    cg.anns_repeat, cg.anns_threads, cg.anns_k, os.eps_parameter);
            } else {
                log("Graph file not found: {}\n", random_graph_path);
            }
            
            // Test optimized graphs at each checkpoint
            for(uint64_t iterations = os.iteration_interval; iterations <= os.total_iterations; iterations += os.iteration_interval) {
                std::string opt_graph_path = graph_paths.opt_scaling_graph_file(dims, config.metric, cg.k, og.k_opt, og.eps_opt, og.i_opt, iterations);
                log("\n=== OPT_SCALING Test: iterations={} ===\n", iterations);
                log("Graph: {}\n", opt_graph_path);
                
                if(std::filesystem::exists(opt_graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(opt_graph_path.c_str());
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, os.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", opt_graph_path);
                }
            }
            
            deglib::benchmark::reset_log_to_console();
            log("OPT_SCALING: Log written to: {}\n", log_path);
        }
    }
    
    // THREAD_SCALING test (build graphs with different thread counts)
    if(run_all || test_type_arg == "thread_scaling") {
        const auto& ts = config.thread_scaling_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.thread_scaling_directory();
        
        log("\n=== THREAD_SCALING Test ===\n");
        log("Testing thread counts: [{}]\n", fmt::join(ts.thread_counts, ", "));
        log("Directory: {}\n", scaling_dir);
        
        if(do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);
            
            // Load ground truth once for all thread tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
            
            for(uint32_t threads : ts.thread_counts) {
                std::string graph_path = graph_paths.thread_scaling_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, threads);
                std::string log_path = graph_paths.thread_scaling_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, threads);
                
                // Set up log file for this thread count (each gets its own log for timing analysis)
                deglib::benchmark::set_log_file(log_path, false);
                log("\n=== THREAD_SCALING Test: threads={} ===\n", threads);
                log("Graph: {}\n", graph_path);
                
                // Phase 1: Build graph if it doesn't exist
                if(std::filesystem::exists(graph_path)) {
                    log("Graph already exists, skipping build\n");
                } else {
                    log("\n--- Building graph with {} threads ---\n", threads);
                    deglib::benchmark::create_graph(
                        *base_repository, config.data_stream_type, graph_path,
                        config.metric, cg.lid, cg.k, cg.k_ext, cg.eps_ext,
                        0, 0, 0,  // k_opt, eps_opt, i_opt (defaults)
                        threads);
                    log("Graph built and saved: {}\n", graph_path);
                }
                
                // Phase 2: Test the graph
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("\n--- Testing graph ---\n");
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, cg.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("threads={}: Log written to: {}\n", threads, log_path);
            }
        }
    }
    
    // RNG_DISABLED test (build graph with RNG pruning disabled)
    if(run_all || test_type_arg == "rng_disabled") {
        const auto& rd = config.rng_disabled_test;
        const auto& cg = config.create_graph;
        
        std::string graph_path = graph_paths.rng_disabled_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
        std::string log_path = graph_paths.rng_disabled_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
        
        log("\n=== RNG_DISABLED Test ===\n");
        log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, RNG=disabled\n",
            cg.k, cg.k_ext, cg.eps_ext, DatasetConfig::optimization_target_str(cg.lid));
        log("Graph: {}\n", graph_path);
        log("Log: {}\n", log_path);
        
        if(do_run && base_repository && query_repository) {
            deglib::benchmark::set_log_file(log_path, true);
            log("\n=== RNG_DISABLED Test ===\n");
            
            if(std::filesystem::exists(graph_path)) {
                log("Graph already exists: {}\n", graph_path);
            } else {
                log("\n=== Building Graph with RNG Disabled ===\n");
                log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                    cg.k, cg.k_ext, cg.eps_ext, DatasetConfig::optimization_target_str(cg.lid), cg.build_threads);
                log("Output graph: {}\n", graph_path);
                
                // Build graph with RNG pruning disabled (use_rng=false)
                deglib::benchmark::create_graph(*base_repository, config.data_stream_type, graph_path, 
                    config.metric, cg.lid, cg.k, cg.k_ext, cg.eps_ext, 0, 0, 0, 
                    cg.build_threads, /*use_rng=*/false);
            }
            
            // Test the graph
            if(std::filesystem::exists(graph_path)) {
                const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                log("Graph loaded: {} vertices\n", graph.size());
                
                log("\n--- Graph Statistics ---\n");
                auto stats = deglib::benchmark::collect_graph_stats(graph);
                deglib::benchmark::log_graph_stats(stats);
                
                // ANNS Test
                log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
                auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);
                deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                    cg.anns_repeat, cg.anns_threads, cg.anns_k, cg.eps_parameter);
            }
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }

    
    // DYNAMIC_DATA test (builds graphs with different DataStreamTypes in dynamic directory)
    if(run_all || test_type_arg == "dynamic_data") {
        const auto& dd = config.dynamic_data_test;
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;
        std::string dynamic_dir = graph_paths.dynamic_directory();
        
        log("\n=== DYNAMIC_DATA Test ===\n");
        log("Testing DataStreamTypes with StreamingData optimization target\n");
        log("Optimization: k_opt={}, eps_opt={:.4f}, i_opt={}\n", og.k_opt, og.eps_opt, og.i_opt);
        log("Directory: {}\n", dynamic_dir);
        
        if(do_run && base_repository && query_repository) {
            // Ensure dynamic directory exists
            std::filesystem::create_directories(dynamic_dir);
            
            for(DataStreamType ds_type : dd.data_stream_types) {
                std::string graph_path = graph_paths.dynamic_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, 
                                                                         og.k_opt, og.eps_opt, og.i_opt, ds_type);
                std::string log_path = graph_paths.dynamic_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, 
                                                                     og.k_opt, og.eps_opt, og.i_opt, ds_type);
                
                // Set up log file for this DataStreamType
                deglib::benchmark::set_log_file(log_path, false);
                log("\n=== DYNAMIC_DATA Test: {} ===\n", DatasetConfig::data_stream_type_str(ds_type));
                log("Graph: {}\n", graph_path);
                
                // Phase 1: Build graph if it doesn't exist
                if(std::filesystem::exists(graph_path)) {
                    log("Graph already exists, skipping build\n");
                } else {
                    log("\n--- Building graph with DataStreamType={} ---\n", DatasetConfig::data_stream_type_str(ds_type));
                    deglib::benchmark::create_graph(
                        *base_repository, ds_type, graph_path,
                        config.metric, deglib::builder::OptimizationTarget::StreamingData,
                        cg.k, cg.k_ext, cg.eps_ext,
                        og.k_opt, og.eps_opt, og.i_opt,
                        cg.build_threads);
                    log("Graph built and saved: {}\n", graph_path);
                }
                
                // Phase 2: Test the graph
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} vertices\n", graph.size());
                    
                    // Select ground truth based on DataStreamType
                    // AddAll uses full dataset, others use half dataset ground truth
                    bool use_half = (ds_type != DataStreamType::AddAll);
                    auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half);
                    
                    // Graph statistics
                    log("\n--- Graph Statistics ---\n");
                    auto stats = deglib::benchmark::collect_graph_stats(graph);
                    deglib::benchmark::log_graph_stats(stats);
                    
                    // ANNS Test
                    log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, cg.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("{}: Log written to: {}\n", DatasetConfig::data_stream_type_str(ds_type), log_path);
            }
        }
    }
    
    
    log("\nTest OK\n");
    return 0;
}
