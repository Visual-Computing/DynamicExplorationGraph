/**
 * @file deglib_build_and_test.cpp
 * @brief Comprehensive graph building and testing tool for DEG.
 *
 * This tool builds DEG graphs with various configurations and runs comprehensive
 * tests including ANNS, exploration, and graph quality analysis.
 *
 * Usage:
 *   deglib_build_and_test [dataset] [tests...]
 *
 * See DatasetConfig and test structs for available configurations.
 */

#if defined(_WIN32)
    #include <windows.h>
#endif

#include <fmt/core.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <thread>

#include "analysis.h"
#include "benchmark.h"
#include "build.h"
#include "dataset.h"
#include "deglib.h"
#include "stats.h"

// Use types from benchmark.h and dataset.h
using deglib::benchmark::Dataset;
using deglib::benchmark::DatasetInfo;
using deglib::benchmark::DatasetName;
using deglib::benchmark::DataStreamType;
using deglib::benchmark::log;

// Helper function to wait before benchmark tests to let the machine settle down
static void wait_before_test(int seconds = 10) {
    log("Waiting {} seconds for machine to settle...\n", seconds);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

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
    uint32_t analysis_threads = std::thread::hardware_concurrency() / 2;  // Threads for graph analysis (default: all CPU threads)

    // ANNS test parameters
    uint32_t anns_k = 100;      // k for ANNS recall test
    uint32_t anns_repeat = 1;   // Number of test repetitions
    uint32_t anns_threads = 1;  // Threads for ANNS test

    // Exploration test parameters
    uint32_t explore_k = 1000;     // k for exploration test
    uint32_t explore_repeat = 1;   // Number of test repetitions
    uint32_t explore_threads = 1;  // Threads for ANNS test

    // k-sweep parameters (tests recall for k=1,2,4,8,...,1024)
    std::vector<uint32_t> k_sweep_values = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    // Test eps parameters for ANNS benchmark (search)
    std::vector<float> eps_parameter = {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f};
};

/**
 * OptimizeGraphTest: Optimizes an existing graph using the EvenRegularGraphBuilder.
 *
 * This test:
 * 1. Loads the graph created by CreateGraphTest
 * 2. Optimizes it for total_iterations
 * 3. Saves the optimized graph in the main graph directory
 * 4. Tests with graph stats, ANNS top-100, and exploration top-1000
 */
struct OptimizeGraphTest {
    uint8_t k_opt = 30;                  // Optimization neighborhood size
    float eps_opt = 0.001f;              // Optimization epsilon for search
    uint8_t i_opt = 5;                   // Optimization path length
    uint64_t total_iterations = 100000;  // Total optimization iterations
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
};

/**
 * KExtSweepTest: Sweep different k_ext values for graph construction.
 * Builds graphs in "kExtScaling" subdirectory.
 */
struct KExtSweepTest {
    std::vector<uint8_t> k_ext_values = {30, 40, 50, 60, 90};  // k_ext values to test
};

/**
 * EpsExtSweepTest: Sweep different eps_ext values for graph construction.
 * Builds graphs in "epsExtScaling" subdirectory.
 */
struct EpsExtSweepTest {
    std::vector<float> eps_ext_values = {0.0f, 0.1f, 0.2f, 0.3f};  // eps_ext values to test
};

/**
 * SizeScalingTest: Build and test graphs with different amounts of data.
 * Builds graphs in "sizeScaling" subdirectory using create_incremental_graphs.
 * Graphs are built incrementally at size_interval intervals.
 * Uses a single log file for all size variants.
 */
struct SizeScalingTest {
    uint32_t size_interval = 10000;  // Build graphs at this interval (e.g., 200k, 400k, ...)
};

/**
 * ReduceScalingTest: Build a full graph and reduce it incrementally.
 * Builds graphs in "reduceScaling" subdirectory using reduce_graph_incremental.
 * Graphs are reduced to size_interval intervals.
 * Uses a single log file for all size variants.
 */
struct ReduceScalingTest {
    uint32_t size_interval = 100000;  // Reduce graphs to this interval
};

/**
 * OptScalingTest: Build a random graph and optimize it at different iteration intervals.
 * Builds graphs in "optScaling" subdirectory.
 * First creates a random graph, then optimizes it at specified iteration intervals.
 * All graphs are tested afterwards with a single log file for the directory.
 */
struct OptScalingTest {
    uint8_t k_opt = 30;                     // Optimization neighborhood size
    float eps_opt = 0.001f;                 // Optimization epsilon for search
    uint8_t i_opt = 5;                      // Optimization path length
    uint64_t iteration_interval = 1000000;  // Save checkpoint every N iterations
    uint64_t total_iterations = 10000000;   // Total optimization iterations
    std::vector<float> eps_parameter = {0.01f, 0.05f, 0.1f, 0.12f, 0.16f, 0.2f, 0.3f, 0.4f};
};

/**
 * ThreadScalingTest: Build graphs with different thread counts.
 * Builds graphs in "threadScaling" subdirectory.
 * Each graph has its own log file for detailed build timing analysis.
 * Uses CreateGraphTest parameters for graph construction.
 */
struct ThreadScalingTest {
    std::vector<uint32_t> thread_counts = {1, 2, 4, 8, 16, 32};  // Number of threads to test
};

/**
 * RNGDisabledTest: Build a single graph with RNG pruning disabled.
 * Uses CreateGraphTest parameters for graph construction.
 * Stores graph in normal graph_directory (not a subdirectory).
 */
struct RNGDisabledTest {};

/**
 * DynamicDataTest: Build graphs with different DataStreamTypes.
 * Builds graphs in "dynamic" subdirectory.
 * Each DataStreamType gets its own graph and log file.
 * Uses CreateGraphTest build parameters with StreamingData optimization target.
 * Ground truth is selected based on DataStreamType (full or half dataset).
 */
struct DynamicDataTest {
    std::vector<DataStreamType> data_stream_types = {
        DataStreamType::AddAll, DataStreamType::AddHalf, DataStreamType::AddAllRemoveHalf, DataStreamType::AddHalfRemoveAndAddOneAtATime};
};

// Main dataset configuration
struct DatasetConfig {
    // Dataset identity (using DatasetName class)
    DatasetName dataset_name = DatasetName::Invalid;

    // Common settings
    deglib::Metric metric = deglib::Metric::L2;

    // ============================================================================
    // Static helper functions for enum to string conversion
    // ============================================================================
    static std::string data_stream_type_str(DataStreamType ds) {
        switch (ds) {
            case DataStreamType::AddAll:
                return "AddAll";
            case DataStreamType::AddHalf:
                return "AddHalf";
            case DataStreamType::AddAllRemoveHalf:
                return "AddAllRemoveHalf";
            case DataStreamType::AddHalfRemoveAndAddOneAtATime:
                return "AddHalfRemoveAndAddOneAtATime";
            default:
                return "Unknown";
        }
    }

    static std::string metric_str(deglib::Metric m) {
        switch (m) {
            case deglib::Metric::L2:
                return "L2";
            case deglib::Metric::L2_Uint8:
                return "L2_Uint8";
            default:
                return "UnknownMetric";
        }
    }

    static std::string optimization_target_str(deglib::builder::OptimizationTarget t) {
        switch (t) {
            case deglib::builder::OptimizationTarget::SchemeA:
                return "SchemeA";
            case deglib::builder::OptimizationTarget::SchemeB:
                return "SchemeB";
            case deglib::builder::OptimizationTarget::HighLID:
                return "HighLID";
            case deglib::builder::OptimizationTarget::LowLID:
                return "LowLID";
            case deglib::builder::OptimizationTarget::StreamingData_SchemeA:
                return "StreamingData_SchemeA";
            case deglib::builder::OptimizationTarget::StreamingData_SchemeB:
                return "StreamingData_SchemeB";
            case deglib::builder::OptimizationTarget::StreamingData_SchemeC:
                return "StreamingData_SchemeC";
            case deglib::builder::OptimizationTarget::StreamingData_SchemeD:
                return "StreamingData_SchemeD";
            default:
                return "UnknownLID";
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
    ReduceScalingTest reduce_scaling_test;
    OptScalingTest opt_scaling_test;
    ThreadScalingTest thread_scaling_test;
    RNGDisabledTest rng_disabled_test;
    DynamicDataTest dynamic_data_test;
};

// Path building utilities for graph files
struct GraphPaths {
    std::filesystem::path graph_dir;

    GraphPaths(const Dataset& ds) : graph_dir(ds.data_root() / ds.name() / "deg") {}

    // ============================================================================
    // Helper methods to reduce code duplication
    // ============================================================================
private:
    static std::string metric_str(deglib::Metric m) { return (m == deglib::Metric::L2) ? "L2" : "L2_Uint8"; }

    // Base name: {dims}D_{metric}_K{k}_AddK{k_ext}Eps{eps_ext}_{scheme}
    std::string base_name(
        uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        return fmt::format(
            "{}D_{}_K{}_AddK{}Eps{:.1f}_{}", dims, metric_str(metric), k, k_ext, eps_ext, DatasetConfig::optimization_target_str(lid));
    }

    // Opt suffix: _OptK{k_opt}Eps{eps_opt}Path{i_opt}
    static std::string opt_suffix(uint8_t k_opt, float eps_opt, uint8_t i_opt) {
        return fmt::format("_OptK{}Eps{:.4f}Path{}", k_opt, eps_opt, i_opt);
    }

public:
    // ============================================================================
    // Main directory
    // ============================================================================
    std::string graph_directory() const { return graph_dir.string(); }

    // ============================================================================
    // CreateGraphTest: main graph files (with optional optimization parameters)
    // ============================================================================
    std::string graph_file(uint32_t dims,
                           deglib::Metric metric,
                           uint8_t k,
                           uint8_t k_ext,
                           float eps_ext,
                           deglib::builder::OptimizationTarget lid,
                           uint8_t k_opt = 0,
                           float eps_opt = 0,
                           uint8_t i_opt = 0) const {
        std::string suffix = (k_opt == 0 && eps_opt == 0 && i_opt == 0) ? "" : opt_suffix(k_opt, eps_opt, i_opt);
        return (graph_dir / (base_name(dims, metric, k, k_ext, eps_ext, lid) + suffix + ".deg")).string();
    }

    std::string graph_log_file(uint32_t dims,
                               deglib::Metric metric,
                               uint8_t k,
                               uint8_t k_ext,
                               float eps_ext,
                               deglib::builder::OptimizationTarget lid,
                               uint8_t k_opt = 0,
                               float eps_opt = 0,
                               uint8_t i_opt = 0) const {
        std::string suffix = (k_opt == 0 && eps_opt == 0 && i_opt == 0) ? "" : opt_suffix(k_opt, eps_opt, i_opt);
        return (graph_dir / (base_name(dims, metric, k, k_ext, eps_ext, lid) + suffix + ".log")).string();
    }

    // ============================================================================
    // OptimizeGraphTest: optimized graph files in main directory
    // ============================================================================
    std::string optimized_graph_file(uint32_t dims,
                                     deglib::Metric metric,
                                     uint8_t k,
                                     uint8_t k_ext,
                                     float eps_ext,
                                     deglib::builder::OptimizationTarget lid,
                                     uint8_t k_opt,
                                     float eps_opt,
                                     uint8_t i_opt,
                                     uint64_t iterations) const {
        return (graph_dir / (base_name(dims, metric, k, k_ext, eps_ext, lid) + opt_suffix(k_opt, eps_opt, i_opt) +
                             fmt::format("_it{}.deg", iterations)))
            .string();
    }

    std::string optimized_log_file(uint32_t dims,
                                   deglib::Metric metric,
                                   uint8_t k,
                                   uint8_t k_ext,
                                   float eps_ext,
                                   deglib::builder::OptimizationTarget lid,
                                   uint8_t k_opt,
                                   float eps_opt,
                                   uint8_t i_opt,
                                   uint64_t iterations) const {
        return (graph_dir / (base_name(dims, metric, k, k_ext, eps_ext, lid) + opt_suffix(k_opt, eps_opt, i_opt) +
                             fmt::format("_it{}.log", iterations)))
            .string();
    }

    // ============================================================================
    // Scaling test directories
    // ============================================================================
    std::string k_scaling_directory() const { return (graph_dir / "kScaling").string(); }
    std::string k_ext_scaling_directory() const { return (graph_dir / "kExtScaling").string(); }
    std::string eps_ext_scaling_directory() const { return (graph_dir / "epsExtScaling").string(); }
    std::string size_scaling_directory(deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / ("sizeScaling_" + DatasetConfig::optimization_target_str(lid))).string();
    }
    std::string reduce_scaling_directory(deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / ("reduceScaling_" + DatasetConfig::optimization_target_str(lid))).string();
    }
    std::string opt_scaling_directory() const { return (graph_dir / "optScaling").string(); }
    std::string thread_scaling_directory() const { return (graph_dir / "threadScaling").string(); }
    std::string dynamic_directory() const { return (graph_dir / "dynamic").string(); }

    // ============================================================================
    // Generic scaling file helpers (used by k_sweep, k_ext_sweep, eps_ext_sweep)
    // ============================================================================
    std::string scaling_graph_file(const std::string& dir,
                                   uint32_t dims,
                                   deglib::Metric metric,
                                   uint8_t k,
                                   uint8_t k_ext,
                                   float eps_ext,
                                   deglib::builder::OptimizationTarget lid) const {
        return (std::filesystem::path(dir) / (base_name(dims, metric, k, k_ext, eps_ext, lid) + ".deg")).string();
    }

    std::string scaling_log_file(const std::string& dir,
                                 uint32_t dims,
                                 deglib::Metric metric,
                                 uint8_t k,
                                 uint8_t k_ext,
                                 float eps_ext,
                                 deglib::builder::OptimizationTarget lid) const {
        return (std::filesystem::path(dir) / (base_name(dims, metric, k, k_ext, eps_ext, lid) + ".log")).string();
    }

    // ============================================================================
    // ThreadScalingTest: per-thread graph and log files
    // ============================================================================
    std::string thread_scaling_graph_file(uint32_t dims,
                                          deglib::Metric metric,
                                          uint8_t k,
                                          uint8_t k_ext,
                                          float eps_ext,
                                          deglib::builder::OptimizationTarget lid,
                                          uint32_t threads) const {
        return (std::filesystem::path(thread_scaling_directory()) /
                fmt::format("{}_T{}.deg", base_name(dims, metric, k, k_ext, eps_ext, lid), threads))
            .string();
    }

    std::string thread_scaling_log_file(uint32_t dims,
                                        deglib::Metric metric,
                                        uint8_t k,
                                        uint8_t k_ext,
                                        float eps_ext,
                                        deglib::builder::OptimizationTarget lid,
                                        uint32_t threads) const {
        return (std::filesystem::path(thread_scaling_directory()) /
                fmt::format("{}_T{}.log", base_name(dims, metric, k, k_ext, eps_ext, lid), threads))
            .string();
    }

    // ============================================================================
    // RNGDisabledTest: NoRNG graph and log files in main directory
    // ============================================================================
    std::string rng_disabled_graph_file(
        uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / fmt::format("{}_NoRNG.deg", base_name(dims, metric, k, k_ext, eps_ext, lid))).string();
    }

    std::string rng_disabled_log_file(
        uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        return (graph_dir / fmt::format("{}_NoRNG.log", base_name(dims, metric, k, k_ext, eps_ext, lid))).string();
    }

    // ============================================================================
    // DynamicDataTest: per-DataStreamType graph and log files with opt params
    // ============================================================================
private:
    std::string dynamic_file_name(uint32_t dims,
                                  deglib::Metric metric,
                                  deglib::builder::OptimizationTarget lid,
                                  uint8_t k,
                                  uint8_t k_ext,
                                  float eps_ext,
                                  uint8_t k_opt,
                                  float eps_opt,
                                  uint8_t i_opt,
                                  uint64_t opt_iterations,
                                  DataStreamType ds_type,
                                  const std::string& ext,
                                  bool use_path_verification) const {
        std::string ds_str = DatasetConfig::data_stream_type_str(ds_type);
        std::string lid_str = DatasetConfig::optimization_target_str(lid);
        std::string opt_part = (k_opt == 0 && eps_opt == 0 && i_opt == 0) ? "" : opt_suffix(k_opt, eps_opt, i_opt);
        std::string it_part = (opt_iterations == 0) ? "" : fmt::format("_it{}", opt_iterations);
        std::string path_part = use_path_verification ? "_UsingPathSearch" : "";
        return (std::filesystem::path(dynamic_directory()) / fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}{}{}_{}{}{}",
                                                                         dims,
                                                                         metric_str(metric),
                                                                         k,
                                                                         k_ext,
                                                                         eps_ext,
                                                                         lid_str,
                                                                         opt_part,
                                                                         it_part,
                                                                         ds_str,
                                                                         path_part,
                                                                         ext))
            .string();
    }

public:
    std::string dynamic_graph_file(uint32_t dims,
                                   deglib::Metric metric,
                                   deglib::builder::OptimizationTarget lid,
                                   uint8_t k,
                                   uint8_t k_ext,
                                   float eps_ext,
                                   uint8_t k_opt,
                                   float eps_opt,
                                   uint8_t i_opt,
                                   DataStreamType ds_type,
                                   uint64_t opt_iterations = 0,
                                   bool use_path_verification = false) const {
        return dynamic_file_name(
            dims, metric, lid, k, k_ext, eps_ext, k_opt, eps_opt, i_opt, opt_iterations, ds_type, ".deg", use_path_verification);
    }

    std::string dynamic_log_file(uint32_t dims,
                                 deglib::Metric metric,
                                 deglib::builder::OptimizationTarget lid,
                                 uint8_t k,
                                 uint8_t k_ext,
                                 float eps_ext,
                                 uint8_t k_opt,
                                 float eps_opt,
                                 uint8_t i_opt,
                                 DataStreamType ds_type,
                                 uint64_t opt_iterations = 0,
                                 bool use_path_verification = false) const {
        return dynamic_file_name(
            dims, metric, lid, k, k_ext, eps_ext, k_opt, eps_opt, i_opt, opt_iterations, ds_type, ".log", use_path_verification);
    }

    // ============================================================================
    // SizeScalingTest: incremental size graph files
    // ============================================================================
    std::string size_scaling_graph_file(uint32_t dims,
                                        deglib::Metric metric,
                                        uint8_t k,
                                        uint8_t k_ext,
                                        float eps_ext,
                                        deglib::builder::OptimizationTarget lid,
                                        uint32_t size) const {
        return (std::filesystem::path(size_scaling_directory(lid)) /
                fmt::format("{}_N{}.deg", base_name(dims, metric, k, k_ext, eps_ext, lid), size))
            .string();
    }

    std::string size_scaling_log_file(deglib::builder::OptimizationTarget lid) const {
        return (std::filesystem::path(size_scaling_directory(lid)) / "log.txt").string();
    }

    // ============================================================================
    // ReduceScalingTest: incremental reduce graph files
    // ============================================================================
    std::string reduce_scaling_log_file(deglib::builder::OptimizationTarget lid) const {
        return (std::filesystem::path(reduce_scaling_directory(lid)) / "log.txt").string();
    }

    // ============================================================================
    // OptScalingTest: random graph + optimized checkpoints
    // ============================================================================
    std::string opt_scaling_random_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k) const {
        return (std::filesystem::path(opt_scaling_directory()) / fmt::format("{}D_{}_K{}_random.deg", dims, metric_str(metric), k))
            .string();
    }

    std::string opt_scaling_graph_file(
        uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_opt, float eps_opt, uint8_t i_opt, uint64_t iterations) const {
        return (std::filesystem::path(opt_scaling_directory()) /
                fmt::format("{}D_{}_K{}{}_it{}.deg", dims, metric_str(metric), k, opt_suffix(k_opt, eps_opt, i_opt), iterations))
            .string();
    }

    std::string opt_scaling_log_file() const { return (std::filesystem::path(opt_scaling_directory()) / "log.txt").string(); }
};

// Dataset configuration factory
// All test structs have default values, just modify what's different per dataset
static DatasetConfig get_dataset_config(const DatasetName& dataset_name) {
    DatasetConfig conf{};
    conf.dataset_name = dataset_name;

    if (dataset_name == DatasetName::SIFT1M) {
        conf.create_graph.eps_parameter = {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f};

        conf.optimize_graph.total_iterations = 200000;

        conf.all_schemes_test.eps_parameter = {
            {deglib::builder::OptimizationTarget::SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::HighLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::LowLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeC, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeD, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}}};

    } else if (dataset_name == DatasetName::GLOVE) {
        conf.create_graph.lid = deglib::builder::OptimizationTarget::HighLID;
        conf.create_graph.eps_parameter = {0.001f, 0.05f, 0.10f, 0.125f, 0.15f, 0.175f, 0.2f, 0.25f, 0.3f};

        conf.optimize_graph.total_iterations = 2000000;
        conf.opt_scaling_test.total_iterations = 30000000;
        conf.opt_scaling_test.iteration_interval = 3000000;

        conf.all_schemes_test.eps_parameter = {
            {deglib::builder::OptimizationTarget::SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f}},
            {deglib::builder::OptimizationTarget::SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f}},
            {deglib::builder::OptimizationTarget::HighLID, {0.01f, 0.05f, 0.10f, 0.15f, 0.2f, 0.3f}},
            {deglib::builder::OptimizationTarget::LowLID, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeC, {0.01f, 0.05f, 0.10f, 0.15f, 0.2f, 0.3f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeD, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f}}};

    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.create_graph.eps_parameter = {0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f};

        conf.optimize_graph.total_iterations = 400000;
        conf.opt_scaling_test.total_iterations = 20000000;
        conf.opt_scaling_test.iteration_interval = 2000000;

        conf.all_schemes_test.eps_parameter = {
            {deglib::builder::OptimizationTarget::SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::HighLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::LowLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeC, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeD, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}}};

    } else if (dataset_name == DatasetName::AUDIO) {
        conf.create_graph.k = 20;
        conf.create_graph.k_ext = 40;
        conf.create_graph.anns_repeat = 50;
        conf.create_graph.eps_parameter = {0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f};

        conf.optimize_graph.k_opt = 20;
        conf.optimize_graph.total_iterations = 20000;

        conf.all_schemes_test.eps_parameter = {
            {deglib::builder::OptimizationTarget::SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::HighLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::LowLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeC, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeD, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}}};

        conf.opt_scaling_test.iteration_interval = 100000;
        conf.opt_scaling_test.total_iterations = 1000000;
    } else if (dataset_name == DatasetName::ENRON) {
        conf.create_graph.k = 30;
        conf.create_graph.k_ext = 60;
        conf.create_graph.anns_repeat = 50;
        conf.create_graph.eps_parameter = {0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f};

        conf.optimize_graph.k_opt = 30;
        conf.optimize_graph.total_iterations = 40000;

        conf.all_schemes_test.eps_parameter = {
            {deglib::builder::OptimizationTarget::SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::HighLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::LowLID, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeA, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeB, {0.01f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeC, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}},
            {deglib::builder::OptimizationTarget::StreamingData_SchemeD, {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f}}};

        conf.opt_scaling_test.iteration_interval = 100000;
        conf.opt_scaling_test.total_iterations = 1000000;
    }

    return conf;
}

int main(int argc, char* argv[]) {
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
    DatasetName ds_name = DatasetName::AUDIO;
    std::string test_type_arg = "all";
    bool do_run = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "help" || arg == "--help") {
            log("Usage: deglib_phd <dataset> [test_type] [--run]\n");
            log("Datasets: sift1m, deep1m, audio, glove, enron\n");
            log("Test types:\n");
            log("  create_graph    - Build graph, run stats, ANNS, explore, k-sweep\n");
            log("  optimize_graph  - Optimize existing graph\n");
            log("  dynamic_data    - Build graphs with different DataStreamTypes\n");
            log("  all_schemes     - Test all 5 OptimizationTargets\n");
            log("  k_sweep         - Sweep k values for graph construction\n");
            log("  k_ext_sweep     - Sweep k_ext values for graph construction\n");
            log("  eps_ext_sweep   - Sweep eps_ext values for graph construction\n");
            log("  size_scaling    - Build/test graphs with different data sizes (incremental)\n");
            log("  reduce_scaling  - Build full graph and reduce it incrementally\n");
            log("  opt_scaling     - Build random graph, optimize at intervals, test\n");
            log("  thread_scaling  - Build graphs with different thread counts\n");
            log("  rng_disabled    - Build graph with RNG pruning disabled\n");
            log("  all             - Run all available tests\n");
            log("Options: --run  Actually execute tests (otherwise just show config)\n");
            return 0;
        }
        if (arg == "--run") {
            do_run = true;
            continue;
        }

        // Try to parse as dataset name
        auto parsed_ds = DatasetName::from_string(arg);
        if (parsed_ds.is_valid()) {
            ds_name = parsed_ds;
        } else if (arg == "create_graph" || arg == "optimize_graph" || arg == "dynamic_data" || arg == "all_schemes" || arg == "k_sweep" ||
                   arg == "k_ext_sweep" || arg == "eps_ext_sweep" || arg == "size_scaling" || arg == "reduce_scaling" ||
                   arg == "opt_scaling" || arg == "thread_scaling" || arg == "rng_disabled" || arg == "all") {
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
            cg.k,
            cg.k_ext,
            cg.eps_ext,
            DatasetConfig::optimization_target_str(cg.lid));
        log("     ANNS: k={}, repeat={}, threads={}\n", cg.anns_k, cg.anns_repeat, cg.anns_threads);
        log("     Explore: k={}\n", cg.explore_k);
        log("     k-sweep: [{}]\n", fmt::join(cg.k_sweep_values, ", "));
    }
    log(" - optimize_graph (k_opt={}, eps_opt={:.4f}, i_opt={}, total_it={})\n",
        config.optimize_graph.k_opt,
        config.optimize_graph.eps_opt,
        config.optimize_graph.i_opt,
        config.optimize_graph.total_iterations);

    // Additional test types
    log(" - all_schemes (tests 5 OptimizationTargets)\n");
    log(" - k_sweep (k_values=[{}], dir=kScaling)\n", fmt::join(config.k_sweep_test.k_values, ", "));
    log(" - k_ext_sweep (k_ext_values=[{}], dir=kExtScaling)\n", fmt::join(config.k_ext_sweep_test.k_ext_values, ", "));
    log(" - eps_ext_sweep (eps_ext_values=[{}], dir=epsExtScaling)\n", fmt::join(config.eps_ext_sweep_test.eps_ext_values, ", "));
    log(" - size_scaling (interval={}, dir=sizeScaling)\n", config.size_scaling_test.size_interval);
    log(" - reduce_scaling (interval={}, dir=reduceScaling)\n", config.reduce_scaling_test.size_interval);
    {
        const auto& os = config.opt_scaling_test;
        const auto& og = config.optimize_graph;
        log(" - opt_scaling (interval={}, total={}, k_opt={}, dir=optScaling)\n", os.iteration_interval, os.total_iterations, og.k_opt);
    }
    log(" - thread_scaling (thread_counts=[{}], dir=threadScaling)\n", fmt::join(config.thread_scaling_test.thread_counts, ", "));
    log(" - rng_disabled (uses create_graph params, RNG=off)\n");
    log(" - dynamic_data (DataStreamTypes, StreamingData opt, dir=dynamic)\n");

    // Load data once if we're running tests
    std::unique_ptr<deglib::StaticFeatureRepository> base_repository;
    std::unique_ptr<deglib::StaticFeatureRepository> query_repository;
    uint32_t dims = 0;
    uint64_t linear_baseline_us = 0;  // Baseline time per query for early abort

    if (do_run) {
        // Setup dataset (downloads, extracts, generates ground truth if needed)
        auto setup_threads = std::thread::hardware_concurrency() / 2;
        log("\nSetting up dataset with {} threads...\n", setup_threads);
        if (!deglib::benchmark::setup_dataset(ds, setup_threads)) {
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

        // Compute linear search baseline for early abort
        log("\n--- Computing Linear Search Baseline ---\n");
        linear_baseline_us = deglib::benchmark::compute_linear_search_baseline(*base_repository, config.metric, 100) * 2;
    }

    // Execute tests based on test_type_arg
    bool run_all = (test_type_arg == "all");

    // CREATE_GRAPH test
    if (run_all || test_type_arg == "create_graph") {
        const auto& cg = config.create_graph;

        log("\n=== CREATE_GRAPH Test ===\n");
        log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
            cg.k,
            cg.k_ext,
            cg.eps_ext,
            DatasetConfig::optimization_target_str(cg.lid),
            cg.build_threads);

        if (do_run && base_repository) {
            std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string log_path = graph_paths.graph_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);

            // Skip entire scenario if log file already exists
            if (std::filesystem::exists(log_path)) {
                log("CREATE_GRAPH: Skipping - log file already exists: {}\n", log_path);
            } else {
                // Set up log file for this graph (all tests logged here)
                deglib::benchmark::set_log_file(log_path, true);

                // Build graph if it doesn't exist
                if (std::filesystem::exists(graph_path)) {
                    log("Graph already exists: {}\n", graph_path);
                } else {
                    log("\n=== Building Graph ===\n");
                    log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                        cg.k,
                        cg.k_ext,
                        cg.eps_ext,
                        DatasetConfig::optimization_target_str(cg.lid),
                        cg.build_threads);
                    log("Output graph: {}\n", graph_path);

                    deglib::benchmark::create_graph(*base_repository,
                                                    DataStreamType::AddAll,
                                                    graph_path,
                                                    config.metric,
                                                    cg.lid,
                                                    cg.k,
                                                    cg.k_ext,
                                                    cg.eps_ext,
                                                    0,
                                                    0,
                                                    0,
                                                    cg.build_threads,
                                                    true,
                                                    ds.info().scale);
                }

                // Run comprehensive tests on the graph (existing or newly created)
                if (std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} vertices\n", graph.size());

                    // 1. Analyze graph and log stats (with graph quality using base GT)
                    log("\n--- Graph Analysis ---\n");
                    {
                        auto base_gt = ds.load_base_groundtruth(graph.getEdgesPerVertex());
                        deglib::benchmark::analyze_graph(graph, base_gt, true, true, cg.analysis_threads);
                    }

                    // 2. ANNS Test with Top-k
                    log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
                    {
                        auto ground_truth = ds.load_groundtruth(cg.anns_k, false);
                        wait_before_test();
                        deglib::benchmark::test_graph_anns(graph,
                                                           *query_repository,
                                                           ground_truth,
                                                           cg.anns_repeat,
                                                           cg.anns_threads,
                                                           cg.anns_k,
                                                           cg.eps_parameter,
                                                           nullptr,
                                                           linear_baseline_us);
                    }

                    // 3. Exploration Test
                    log("\n--- Exploration Test (k={}) ---\n", cg.explore_k);
                    {
                        auto entry_vertices = ds.load_explore_entry_vertices();
                        auto explore_gt = ds.load_explore_groundtruth(cg.explore_k);
                        wait_before_test();
                        deglib::benchmark::test_graph_explore(graph,
                                                              entry_vertices,
                                                              explore_gt,
                                                              true,
                                                              cg.explore_repeat,
                                                              cg.explore_k,
                                                              cg.explore_threads,
                                                              nullptr,
                                                              ds.info().explore_depth,
                                                              linear_baseline_us);
                    }

                    // 4. k-Sweep Test
                    log("\n--- k-Sweep Test ---\n");
                    for (uint32_t k_val : cg.k_sweep_values) {
                        log("\n-- k={} --\n", k_val);
                        auto gt_k = ds.load_groundtruth(k_val, false);
                        wait_before_test();
                        deglib::benchmark::test_graph_anns(graph,
                                                           *query_repository,
                                                           gt_k,
                                                           cg.anns_repeat,
                                                           cg.anns_threads,
                                                           k_val,
                                                           cg.eps_parameter,
                                                           nullptr,
                                                           linear_baseline_us);
                    }
                }

                deglib::benchmark::reset_log_to_console();
                log("Log written to: {}\n", log_path);
            }
        }
    }

    // OPTIMIZE_GRAPH test (optimize graph and run comprehensive tests)
    if (run_all || test_type_arg == "optimize_graph") {
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;

        log("\n=== OPTIMIZE_GRAPH Test ===\n");
        log("Settings: k_opt={}, eps_opt={:.4f}, i_opt={}, total_iterations={}\n", og.k_opt, og.eps_opt, og.i_opt, og.total_iterations);

        if (do_run && base_repository && query_repository) {
            std::string input_graph = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string output_graph = graph_paths.optimized_graph_file(
                dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, og.k_opt, og.eps_opt, og.i_opt, og.total_iterations);
            std::string log_path = graph_paths.optimized_log_file(
                dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, og.k_opt, og.eps_opt, og.i_opt, og.total_iterations);

            // Skip entire scenario if log file already exists
            if (std::filesystem::exists(log_path)) {
                log("OPTIMIZE_GRAPH: Skipping - log file already exists: {}\n", log_path);
            } else {
                // Set up log file for this optimized graph
                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== OPTIMIZE_GRAPH Test ===\n");
                log("Input graph: {}\n", input_graph);
                log("Output graph: {}\n", output_graph);
                log("Settings: k_opt={}, eps_opt={:.4f}, i_opt={}, total_iterations={}\n",
                    og.k_opt,
                    og.eps_opt,
                    og.i_opt,
                    og.total_iterations);

                // Check if source graph exists
                if (!std::filesystem::exists(input_graph)) {
                    log("ERROR: Source graph does not exist: {}\n", input_graph);
                } else {
                    // Build optimized graph if it doesn't exist
                    if (std::filesystem::exists(output_graph)) {
                        log("Optimized graph already exists: {}\n", output_graph);
                    } else {
                        // Load the source graph and optimize it
                        log("\n--- Loading source graph ---\n");
                        auto graph = deglib::graph::load_sizebounded_graph(input_graph.c_str());
                        log("Loaded graph: {} vertices\n", graph.size());

                        // Run optimization
                        log("\n--- Optimizing graph for {} iterations ---\n", og.total_iterations);
                        deglib::benchmark::optimize_graph(
                            graph, og.k_opt, og.eps_opt, og.i_opt, og.total_iterations, 10000, ds.info().scale);

                        // Save optimized graph
                        graph.saveGraph(output_graph.c_str());
                        log("Saved optimized graph: {}\n", output_graph);
                    }

                    // Run comprehensive tests on the optimized graph
                    if (std::filesystem::exists(output_graph)) {
                        const auto graph = deglib::graph::load_readonly_graph(output_graph.c_str());
                        log("Graph loaded: {} vertices\n", graph.size());

                        // 1. Analyze graph and log stats (with graph quality using base GT)
                        log("\n--- Graph Analysis ---\n");
                        {
                            auto base_gt = ds.load_base_groundtruth(graph.getEdgesPerVertex());
                            deglib::benchmark::analyze_graph(graph, base_gt, true, true, cg.analysis_threads);
                        }

                        // 2. ANNS Test with Top-100
                        log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
                        {
                            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);
                            wait_before_test();
                            deglib::benchmark::test_graph_anns(graph,
                                                               *query_repository,
                                                               ground_truth,
                                                               cg.anns_repeat,
                                                               cg.anns_threads,
                                                               cg.anns_k,
                                                               cg.eps_parameter,
                                                               nullptr,
                                                               linear_baseline_us);
                        }

                        // 3. Exploration Test with Top-1000
                        log("\n--- Exploration Test (k={}) ---\n", cg.explore_k);
                        {
                            auto entry_vertices = ds.load_explore_entry_vertices();
                            auto explore_gt = ds.load_explore_groundtruth(cg.explore_k);
                            wait_before_test();
                            deglib::benchmark::test_graph_explore(graph,
                                                                  entry_vertices,
                                                                  explore_gt,
                                                                  true,
                                                                  cg.explore_repeat,
                                                                  cg.explore_k,
                                                                  cg.explore_threads,
                                                                  nullptr,
                                                                  ds.info().explore_depth,
                                                                  linear_baseline_us);
                        }
                    }
                }

                deglib::benchmark::reset_log_to_console();
                log("OPTIMIZE_GRAPH: Log written to: {}\n", log_path);
            }
        }
    }

    // ALL_SCHEMES test (tests all OptimizationTargets)
    if (run_all || test_type_arg == "all_schemes") {
        const auto& as = config.all_schemes_test;
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;

        log("\n=== ALL_SCHEMES Test ===\n");
        log("Testing all schemes with k={}, k_ext={}, eps_ext={:.2f}\n", cg.k, cg.k_ext, cg.eps_ext);

        if (do_run && base_repository && query_repository) {
            // Load ground truth for this test
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);

            // Iterate over all schemes configured in eps_parameter
            for (const auto& [lid, scheme_eps] : as.eps_parameter) {
                std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, lid);
                std::string log_path = graph_paths.graph_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, lid);

                // Skip if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("{}: Skipping - log file already exists: {}\n", DatasetConfig::optimization_target_str(lid), log_path);
                    continue;
                }

                deglib::benchmark::set_log_file(log_path, true);

                log("Testing scheme {} \n", DatasetConfig::optimization_target_str(lid));

                // Build graph if it doesn't exist
                if (!std::filesystem::exists(graph_path)) {
                    log("Building Graph ...\n");
                    deglib::benchmark::create_graph(*base_repository,
                                                    DataStreamType::AddAll,
                                                    graph_path,
                                                    config.metric,
                                                    lid,
                                                    cg.k,
                                                    cg.k_ext,
                                                    cg.eps_ext,
                                                    0,
                                                    0,
                                                    0,
                                                    cg.build_threads,
                                                    true,
                                                    ds.info().scale);
                }

                // Test the graph
                if (std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Testing Graph ({} vertices) ...\n", graph.size());

                    wait_before_test();
                    deglib::benchmark::test_graph_anns(graph,
                                                       *query_repository,
                                                       ground_truth,
                                                       cg.anns_repeat,
                                                       cg.anns_threads,
                                                       cg.anns_k,
                                                       scheme_eps,
                                                       nullptr,
                                                       linear_baseline_us);
                } else {
                    log("ERROR: Graph file not found after build attempt: {}\n", graph_path);
                }

                deglib::benchmark::reset_log_to_console();
                log("{}: Log written to: {}\n", DatasetConfig::optimization_target_str(lid), log_path);
            }
        }
    }

    // K_SWEEP test (sweep k values for graph construction in kScaling directory)
    if (run_all || test_type_arg == "k_sweep") {
        const auto& ks = config.k_sweep_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.k_scaling_directory();

        log("\n=== K_SWEEP Test ===\n");
        log("Testing k values: [{}]\n", fmt::join(ks.k_values, ", "));
        log("Directory: {}\n", scaling_dir);

        if (do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);

            // Load ground truth once for all k tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);

            // Use the highest k value as k_ext for all graphs
            uint8_t k_ext = *std::max_element(ks.k_values.begin(), ks.k_values.end());

            for (uint8_t k : ks.k_values) {
                std::string graph_path = graph_paths.scaling_graph_file(scaling_dir, dims, config.metric, k, k_ext, cg.eps_ext, cg.lid);
                std::string log_path = graph_paths.scaling_log_file(scaling_dir, dims, config.metric, k, k_ext, cg.eps_ext, cg.lid);

                // Skip if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("k={}: Skipping - log file already exists: {}\n", k, log_path);
                    continue;
                }

                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== K_SWEEP Test: k={} ===\n", k);
                log("Graph: {}\n", graph_path);

                // Build graph if it doesn't exist
                if (!std::filesystem::exists(graph_path)) {
                    log("\n--- Building Graph with k={} ---\n", k);
                    log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                        k,
                        k_ext,
                        cg.eps_ext,
                        DatasetConfig::optimization_target_str(cg.lid),
                        cg.build_threads);

                    deglib::benchmark::create_graph(*base_repository,
                                                    DataStreamType::AddAll,
                                                    graph_path,
                                                    config.metric,
                                                    cg.lid,
                                                    k,
                                                    k_ext,
                                                    cg.eps_ext,
                                                    0,
                                                    0,
                                                    0,
                                                    cg.build_threads,
                                                    true,
                                                    ds.info().scale);
                }

                // Test the graph
                if (std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} vertices\n", graph.size());
                    wait_before_test();
                    deglib::benchmark::test_graph_anns(graph,
                                                       *query_repository,
                                                       ground_truth,
                                                       cg.anns_repeat,
                                                       cg.anns_threads,
                                                       cg.anns_k,
                                                       cg.eps_parameter,
                                                       nullptr,
                                                       linear_baseline_us);
                } else {
                    log("ERROR: Graph file not found after build attempt: {}\n", graph_path);
                }

                deglib::benchmark::reset_log_to_console();
                log("k={}: Log written to: {}\n", k, log_path);
            }
        }
    }

    // K_EXT_SWEEP test (sweep k_ext values for graph construction in kExtScaling directory)
    if (run_all || test_type_arg == "k_ext_sweep") {
        const auto& kes = config.k_ext_sweep_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.k_ext_scaling_directory();

        log("\n=== K_EXT_SWEEP Test ===\n");
        log("Testing k_ext values: [{}]\n", fmt::join(kes.k_ext_values, ", "));
        log("Directory: {}\n", scaling_dir);

        if (do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);

            // Load ground truth once for all k_ext tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);

            for (uint8_t k_ext : kes.k_ext_values) {
                std::string graph_path = graph_paths.scaling_graph_file(scaling_dir, dims, config.metric, cg.k, k_ext, cg.eps_ext, cg.lid);
                std::string log_path = graph_paths.scaling_log_file(scaling_dir, dims, config.metric, cg.k, k_ext, cg.eps_ext, cg.lid);

                // Skip if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("k_ext={}: Skipping - log file already exists: {}\n", k_ext, log_path);
                    continue;
                }

                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== K_EXT_SWEEP Test: k_ext={} ===\n", k_ext);
                log("Graph: {}\n", graph_path);

                // Build graph if it doesn't exist
                if (!std::filesystem::exists(graph_path)) {
                    log("\n--- Building Graph with k_ext={} ---\n", k_ext);
                    log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                        cg.k,
                        k_ext,
                        cg.eps_ext,
                        DatasetConfig::optimization_target_str(cg.lid),
                        cg.build_threads);

                    deglib::benchmark::create_graph(*base_repository,
                                                    DataStreamType::AddAll,
                                                    graph_path,
                                                    config.metric,
                                                    cg.lid,
                                                    cg.k,
                                                    k_ext,
                                                    cg.eps_ext,
                                                    0,
                                                    0,
                                                    0,
                                                    cg.build_threads,
                                                    true,
                                                    ds.info().scale);
                }

                // Test the graph
                if (std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} vertices\n", graph.size());
                    wait_before_test();
                    deglib::benchmark::test_graph_anns(graph,
                                                       *query_repository,
                                                       ground_truth,
                                                       cg.anns_repeat,
                                                       cg.anns_threads,
                                                       cg.anns_k,
                                                       cg.eps_parameter,
                                                       nullptr,
                                                       linear_baseline_us);
                } else {
                    log("ERROR: Graph file not found after build attempt: {}\n", graph_path);
                }

                deglib::benchmark::reset_log_to_console();
                log("k_ext={}: Log written to: {}\n", k_ext, log_path);
            }
        }
    }

    // EPS_EXT_SWEEP test (sweep eps_ext values for graph construction in epsExtScaling directory)
    if (run_all || test_type_arg == "eps_ext_sweep") {
        const auto& ees = config.eps_ext_sweep_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.eps_ext_scaling_directory();

        log("\n=== EPS_EXT_SWEEP Test ===\n");
        log("Testing eps_ext values: [{}]\n", fmt::join(ees.eps_ext_values, ", "));
        log("Directory: {}\n", scaling_dir);

        if (do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);

            // Load ground truth once for all eps_ext tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);

            for (float eps_ext : ees.eps_ext_values) {
                std::string graph_path = graph_paths.scaling_graph_file(scaling_dir, dims, config.metric, cg.k, cg.k_ext, eps_ext, cg.lid);
                std::string log_path = graph_paths.scaling_log_file(scaling_dir, dims, config.metric, cg.k, cg.k_ext, eps_ext, cg.lid);

                // Skip if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("eps_ext={:.2f}: Skipping - log file already exists: {}\n", eps_ext, log_path);
                    continue;
                }

                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== EPS_EXT_SWEEP Test: eps_ext={:.2f} ===\n", eps_ext);
                log("Graph: {}\n", graph_path);

                // Build graph if it doesn't exist
                if (!std::filesystem::exists(graph_path)) {
                    log("\n--- Building Graph with eps_ext={:.2f} ---\n", eps_ext);
                    log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                        cg.k,
                        cg.k_ext,
                        eps_ext,
                        DatasetConfig::optimization_target_str(cg.lid),
                        cg.build_threads);

                    deglib::benchmark::create_graph(*base_repository,
                                                    DataStreamType::AddAll,
                                                    graph_path,
                                                    config.metric,
                                                    cg.lid,
                                                    cg.k,
                                                    cg.k_ext,
                                                    eps_ext,
                                                    0,
                                                    0,
                                                    0,
                                                    cg.build_threads,
                                                    true,
                                                    ds.info().scale);
                }

                // Test the graph
                if (std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} vertices\n", graph.size());
                    wait_before_test();
                    deglib::benchmark::test_graph_anns(graph,
                                                       *query_repository,
                                                       ground_truth,
                                                       cg.anns_repeat,
                                                       cg.anns_threads,
                                                       cg.anns_k,
                                                       cg.eps_parameter,
                                                       nullptr,
                                                       linear_baseline_us);
                } else {
                    log("ERROR: Graph file not found after build attempt: {}\n", graph_path);
                }

                deglib::benchmark::reset_log_to_console();
                log("eps_ext={:.2f}: Log written to: {}\n", eps_ext, log_path);
            }
        }
    }

    // SIZE_SCALING test (build and test graphs with different data sizes using incremental building)
    if (run_all || test_type_arg == "size_scaling") {
        const auto& ss = config.size_scaling_test;
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;

        // Test both schemes: original cg.lid and StreamingData
        for (deglib::builder::OptimizationTarget lid_scheme : {cg.lid, deglib::builder::OptimizationTarget::StreamingData_SchemeC}) {
            std::string scaling_dir = graph_paths.size_scaling_directory(lid_scheme);
            std::string log_path = graph_paths.size_scaling_log_file(lid_scheme);
            std::string scheme_name = DatasetConfig::optimization_target_str(lid_scheme);

            log("\n=== SIZE_SCALING Test: {} ===\n", scheme_name);
            log("Size interval: {}\n", ss.size_interval);
            log("Directory: {}\n", scaling_dir);
            log("Log file: {}\n", log_path);

            if (do_run && base_repository && query_repository) {
                // Skip entire scenario if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("SIZE_SCALING {}: Skipping - log file already exists: {}\n", scheme_name, log_path);
                    continue;
                } else {
                    // Ensure scaling directory exists
                    std::filesystem::create_directories(scaling_dir);

                    // Set log file for all size scaling operations
                    deglib::benchmark::set_log_file(log_path, false);
                    log("\n=== SIZE_SCALING Test: {} ===\n", scheme_name);
                    log("Size interval: {}\n", ss.size_interval);

                    // Generate base name for graph files
                    std::string metric_str = (config.metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
                    std::string scheme = DatasetConfig::optimization_target_str(lid_scheme);
                    std::string graph_name_base = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}_OptK{}Eps{:.3f}Path{}",
                                                              dims,
                                                              metric_str,
                                                              cg.k,
                                                              cg.k_ext,
                                                              cg.eps_ext,
                                                              scheme,
                                                              og.k_opt,
                                                              og.eps_opt,
                                                              og.i_opt);

                    // Build all graphs incrementally (skips existing graphs internally)
                    log("\n--- Building graphs incrementally ---\n");
                    auto created_files = deglib::benchmark::create_incremental_graphs(*base_repository,
                                                                                      scaling_dir,
                                                                                      graph_name_base,
                                                                                      ss.size_interval,
                                                                                      config.metric,
                                                                                      lid_scheme,
                                                                                      cg.k,
                                                                                      cg.k_ext,
                                                                                      cg.eps_ext,
                                                                                      og.k_opt,
                                                                                      og.eps_opt,
                                                                                      og.i_opt,
                                                                                      cg.build_threads,
                                                                                      true,
                                                                                      ds.info().scale);
                    log("Created {} incremental graphs\n", created_files.size());

                    // Test all graphs
                    log("\n--- Testing graphs ---\n");
                    for (const auto& [graph_path, vertex_count] : created_files) {
                        if (std::filesystem::exists(graph_path)) {
                            // Check for ground truth for this specific size
                            std::string gt_file = ds.groundtruth_file(vertex_count);
                            if (std::filesystem::exists(gt_file)) {
                                log("\n=== SIZE_SCALING Test: {} size={} ===\n", scheme_name, vertex_count);
                                log("Graph: {}\n", graph_path);

                                const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                                auto ground_truth = ds.load_groundtruth_for_size(cg.anns_k, vertex_count);

                                wait_before_test();
                                deglib::benchmark::test_graph_anns(graph,
                                                                   *query_repository,
                                                                   ground_truth,
                                                                   cg.anns_repeat,
                                                                   cg.anns_threads,
                                                                   cg.anns_k,
                                                                   cg.eps_parameter,
                                                                   nullptr,
                                                                   linear_baseline_us);
                            }
                        } else {
                            log("Graph file not found: {}\n", graph_path);
                        }
                    }

                    deglib::benchmark::reset_log_to_console();
                    log("SIZE_SCALING {}: Log written to: {}\n", scheme_name, log_path);
                }
            }
        }
    }

    // OPT_SCALING test (build random graph, optimize at intervals, test all)
    if (run_all || test_type_arg == "opt_scaling") {
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

        if (do_run && base_repository && query_repository) {
            // Skip entire scenario if log file already exists
            if (std::filesystem::exists(log_path)) {
                log("OPT_SCALING: Skipping - log file already exists: {}\n", log_path);
            } else {
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
                std::string graph_name_base =
                    fmt::format("{}D_{}_K{}_OptK{}Eps{:.4f}Path{}", dims, metric_str, cg.k, og.k_opt, og.eps_opt, og.i_opt);

                // Phase 1: Build random graph if needed
                log("\n--- Phase 1: Building random graph ---\n");
                log("Random graph: {}\n", random_graph_path);

                if (std::filesystem::exists(random_graph_path)) {
                    log("Random graph already exists, skipping build\n");
                } else {
                    auto random_graph = deglib::benchmark::create_random_graph(*base_repository, config.metric, cg.k, 0, ds.info().scale);
                    random_graph.saveGraph(random_graph_path.c_str());
                    log("Saved random graph: {}\n", random_graph_path);
                }

                // Phase 2: Optimize graph at each iteration checkpoint
                log("\n--- Phase 2: Optimizing graph at intervals ---\n");

                // Load random graph and optimize it
                auto graph = deglib::graph::load_sizebounded_graph(random_graph_path.c_str());
                log("Loaded random graph: {} vertices\n", graph.size());

                // Load ground truth for recall testing during optimization
                auto ground_truth = ds.load_groundtruth(cg.anns_k, false);

                // Optimize and save checkpoints with recall testing
                auto created_files = deglib::benchmark::improve_and_test(graph,
                                                                         scaling_dir,
                                                                         graph_name_base,
                                                                         og.k_opt,
                                                                         og.eps_opt,
                                                                         og.i_opt,
                                                                         os.iteration_interval,
                                                                         os.total_iterations,
                                                                         *query_repository,
                                                                         ground_truth,
                                                                         cg.anns_k,
                                                                         2000,
                                                                         ds.info().scale);
                log("Created {} optimized checkpoint graphs\n", created_files.size());

                // Phase 3: Test all optimized graphs
                log("\n--- Phase 3: Testing graphs ---\n");

                // Test random graph (iteration 0)
                log("\n=== OPT_SCALING Test: iterations=0 (random) ===\n");
                log("Graph: {}\n", random_graph_path);
                if (std::filesystem::exists(random_graph_path)) {
                    const auto random_graph = deglib::graph::load_readonly_graph(random_graph_path.c_str());
                    wait_before_test();
                    deglib::benchmark::test_graph_anns(random_graph,
                                                       *query_repository,
                                                       ground_truth,
                                                       cg.anns_repeat,
                                                       cg.anns_threads,
                                                       cg.anns_k,
                                                       os.eps_parameter,
                                                       nullptr,
                                                       linear_baseline_us);
                } else {
                    log("Graph file not found: {}\n", random_graph_path);
                }

                // Test optimized graphs at each checkpoint
                for (const auto& [opt_graph_path, iterations] : created_files) {
                    log("\n=== OPT_SCALING Test: iterations={} ===\n", iterations);
                    log("Graph: {}\n", opt_graph_path);

                    if (std::filesystem::exists(opt_graph_path)) {
                        const auto opt_graph = deglib::graph::load_readonly_graph(opt_graph_path.c_str());
                        wait_before_test();
                        deglib::benchmark::test_graph_anns(opt_graph,
                                                           *query_repository,
                                                           ground_truth,
                                                           cg.anns_repeat,
                                                           cg.anns_threads,
                                                           cg.anns_k,
                                                           os.eps_parameter,
                                                           nullptr,
                                                           linear_baseline_us);
                    } else {
                        log("Graph file not found: {}\n", opt_graph_path);
                    }
                }

                deglib::benchmark::reset_log_to_console();
                log("OPT_SCALING: Log written to: {}\n", log_path);
            }
        }
    }

    // THREAD_SCALING test (build graphs with different thread counts)
    if (run_all || test_type_arg == "thread_scaling") {
        const auto& ts = config.thread_scaling_test;
        const auto& cg = config.create_graph;
        std::string scaling_dir = graph_paths.thread_scaling_directory();

        log("\n=== THREAD_SCALING Test ===\n");
        log("Testing thread counts: [{}]\n", fmt::join(ts.thread_counts, ", "));
        log("Directory: {}\n", scaling_dir);

        if (do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);

            // Load ground truth once for all thread tests
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);

            for (uint32_t threads : ts.thread_counts) {
                std::string graph_path =
                    graph_paths.thread_scaling_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, threads);
                std::string log_path =
                    graph_paths.thread_scaling_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, threads);

                // Skip if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("threads={}: Skipping - log file already exists: {}\n", threads, log_path);
                    continue;
                }

                // Set up log file for this thread count (each gets its own log for timing analysis)
                deglib::benchmark::set_log_file(log_path, false);
                log("\n=== THREAD_SCALING Test: threads={} ===\n", threads);
                log("Graph: {}\n", graph_path);

                // Phase 1: Build graph if it doesn't exist
                if (std::filesystem::exists(graph_path)) {
                    log("Graph already exists, skipping build\n");
                } else {
                    log("\n--- Building graph with {} threads ---\n", threads);
                    deglib::benchmark::create_graph(*base_repository,
                                                    DataStreamType::AddAll,
                                                    graph_path,
                                                    config.metric,
                                                    cg.lid,
                                                    cg.k,
                                                    cg.k_ext,
                                                    cg.eps_ext,
                                                    0,
                                                    0,
                                                    0,  // k_opt, eps_opt, i_opt (defaults)
                                                    threads,
                                                    true,
                                                    ds.info().scale);
                    log("Graph built and saved: {}\n", graph_path);
                }

                // Phase 2: Test the graph
                if (std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("\n--- Testing graph ---\n");
                    wait_before_test();
                    deglib::benchmark::test_graph_anns(graph,
                                                       *query_repository,
                                                       ground_truth,
                                                       cg.anns_repeat,
                                                       cg.anns_threads,
                                                       cg.anns_k,
                                                       cg.eps_parameter,
                                                       nullptr,
                                                       linear_baseline_us);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }

                deglib::benchmark::reset_log_to_console();
                log("threads={}: Log written to: {}\n", threads, log_path);
            }
        }
    }

    // RNG_DISABLED test (build graph with RNG pruning disabled)
    if (run_all || test_type_arg == "rng_disabled") {
        const auto& rd = config.rng_disabled_test;
        const auto& cg = config.create_graph;

        std::string graph_path = graph_paths.rng_disabled_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
        std::string log_path = graph_paths.rng_disabled_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);

        log("\n=== RNG_DISABLED Test ===\n");
        log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, RNG=disabled\n",
            cg.k,
            cg.k_ext,
            cg.eps_ext,
            DatasetConfig::optimization_target_str(cg.lid));
        log("Graph: {}\n", graph_path);
        log("Log: {}\n", log_path);

        if (do_run && base_repository && query_repository) {
            // Skip entire scenario if log file already exists
            if (std::filesystem::exists(log_path)) {
                log("RNG_DISABLED: Skipping - log file already exists: {}\n", log_path);
            } else {
                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== RNG_DISABLED Test ===\n");

                if (std::filesystem::exists(graph_path)) {
                    log("Graph already exists: {}\n", graph_path);
                } else {
                    log("\n=== Building Graph with RNG Disabled ===\n");
                    log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                        cg.k,
                        cg.k_ext,
                        cg.eps_ext,
                        DatasetConfig::optimization_target_str(cg.lid),
                        cg.build_threads);
                    log("Output graph: {}\n", graph_path);

                    // Build graph with RNG pruning disabled (use_rng=false)
                    deglib::benchmark::create_graph(*base_repository,
                                                    DataStreamType::AddAll,
                                                    graph_path,
                                                    config.metric,
                                                    cg.lid,
                                                    cg.k,
                                                    cg.k_ext,
                                                    cg.eps_ext,
                                                    0,
                                                    0,
                                                    0,
                                                    cg.build_threads,
                                                    /*use_rng=*/false,
                                                    ds.info().scale);
                }

                // Test the graph
                if (std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} vertices\n", graph.size());

                    // ANNS Test
                    log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
                    auto ground_truth = ds.load_groundtruth(cg.anns_k, false);
                    wait_before_test();
                    deglib::benchmark::test_graph_anns(graph,
                                                       *query_repository,
                                                       ground_truth,
                                                       cg.anns_repeat,
                                                       cg.anns_threads,
                                                       cg.anns_k,
                                                       cg.eps_parameter,
                                                       nullptr,
                                                       linear_baseline_us);
                }

                deglib::benchmark::reset_log_to_console();
                log("Log written to: {}\n", log_path);
            }
        }
    }

    // DYNAMIC_DATA test (builds graphs with different DataStreamTypes in dynamic directory)
    if (run_all || test_type_arg == "dynamic_data") {
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;
        std::string dynamic_dir = graph_paths.dynamic_directory();

        log("\n=== DYNAMIC_DATA Test ===\n");
        log("Testing DataStreamTypes with StreamingData optimization target\n");
        log("Optimization: k_opt={}, eps_opt={:.4f}, i_opt={}\n", og.k_opt, og.eps_opt, og.i_opt);
        log("Directory: {}\n", dynamic_dir);

        if (do_run && base_repository && query_repository) {
            // Ensure dynamic directory exists
            std::filesystem::create_directories(dynamic_dir);

            const std::vector<std::tuple<uint8_t, float, uint8_t, uint32_t, deglib::builder::OptimizationTarget, bool>> settings = {
                // TODO: add version which uses RangeSearch in the RestoreGraph phase
                std::make_tuple(og.k_opt, og.eps_opt, og.i_opt, 0, deglib::builder::OptimizationTarget::StreamingData_SchemeC, false),
                std::make_tuple(og.k_opt, og.eps_opt, og.i_opt, 0, deglib::builder::OptimizationTarget::StreamingData_SchemeC, true),
                std::make_tuple(0, 0, 0, 0, deglib::builder::OptimizationTarget::StreamingData_SchemeC, false),
                std::make_tuple(og.k_opt, og.eps_opt, og.i_opt, 0, cg.lid, false),
                std::make_tuple(og.k_opt, og.eps_opt, og.i_opt, og.total_iterations / 2, cg.lid, false),
            };
            for (const auto& [k_opt, eps_opt, i_opt, opt_iterations, lid, use_path_verification] : settings) {
                for (DataStreamType ds_type :
                     {DataStreamType::AddHalf, DataStreamType::AddHalfRemoveAndAddOneAtATime, DataStreamType::AddAllRemoveHalf}) {
                    std::string graph_path = graph_paths.dynamic_graph_file(dims,
                                                                            config.metric,
                                                                            lid,
                                                                            cg.k,
                                                                            cg.k_ext,
                                                                            cg.eps_ext,
                                                                            k_opt,
                                                                            eps_opt,
                                                                            i_opt,
                                                                            ds_type,
                                                                            opt_iterations,
                                                                            use_path_verification);
                    std::string log_path = graph_paths.dynamic_log_file(dims,
                                                                        config.metric,
                                                                        lid,
                                                                        cg.k,
                                                                        cg.k_ext,
                                                                        cg.eps_ext,
                                                                        k_opt,
                                                                        eps_opt,
                                                                        i_opt,
                                                                        ds_type,
                                                                        opt_iterations,
                                                                        use_path_verification);

                    // Skip if log file already exists
                    if (std::filesystem::exists(log_path)) {
                        log("{}: Skipping - log file already exists: {}\n", DatasetConfig::data_stream_type_str(ds_type), log_path);
                        continue;
                    }

                    // Set up log file for this DataStreamType
                    deglib::benchmark::set_log_file(log_path, false);
                    log("\n=== DYNAMIC_DATA Test: {} ===\n", DatasetConfig::data_stream_type_str(ds_type));
                    log("Graph: {}\n", graph_path);

                    // Phase 1: Build graph if it doesn't exist
                    if (std::filesystem::exists(graph_path)) {
                        log("Graph already exists, skipping build\n");
                    } else {
                        log("\n--- Building graph with DataStreamType={} ---\n", DatasetConfig::data_stream_type_str(ds_type));
                        deglib::benchmark::create_graph(*base_repository,
                                                        ds_type,
                                                        graph_path,
                                                        config.metric,
                                                        lid,
                                                        cg.k,
                                                        cg.k_ext,
                                                        cg.eps_ext,
                                                        k_opt,
                                                        eps_opt,
                                                        i_opt,
                                                        cg.build_threads,
                                                        true,
                                                        ds.info().scale,
                                                        use_path_verification);
                        log("Graph built and saved: {}\n", graph_path);

                        // optimize graph
                        if (opt_iterations > 0) {
                            log("\n--- Optimizing graph for {} iterations ---\n", opt_iterations);
                            auto graph = deglib::graph::load_sizebounded_graph(graph_path.c_str());
                            deglib::benchmark::optimize_graph(graph, k_opt, eps_opt, i_opt, opt_iterations, 10000, ds.info().scale);
                            graph.saveGraph(graph_path.c_str());
                        }
                    }

                    // Phase 2: Test the graph
                    if (std::filesystem::exists(graph_path)) {
                        const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                        log("Graph loaded: {} vertices\n", graph.size());

                        // Determine if we need half dataset ground truth based on DataStreamType
                        bool use_half = (ds_type != DataStreamType::AddAll);

                        // Graph analysis with base GT (use half for dynamic data)
                        log("\n--- Graph Analysis ---\n");
                        {
                            auto base_gt = ds.load_base_groundtruth(graph.getEdgesPerVertex(), use_half);
                            deglib::benchmark::analyze_graph(graph, base_gt, true, true, cg.analysis_threads);
                        }

                        // ANNS Test
                        log("\n--- ANNS Test (k={}) ---\n", cg.anns_k);
                        {
                            auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half);
                            wait_before_test();
                            deglib::benchmark::test_graph_anns(graph,
                                                               *query_repository,
                                                               ground_truth,
                                                               cg.anns_repeat,
                                                               cg.anns_threads,
                                                               cg.anns_k,
                                                               cg.eps_parameter,
                                                               nullptr,
                                                               linear_baseline_us);
                        }

                        // Exploration Test
                        log("\n--- Exploration Test (k={}) ---\n", cg.explore_k);
                        {
                            auto entry_vertices = ds.load_explore_entry_vertices();
                            auto explore_gt = ds.load_explore_groundtruth(cg.explore_k, use_half);
                            wait_before_test();
                            deglib::benchmark::test_graph_explore(graph,
                                                                  entry_vertices,
                                                                  explore_gt,
                                                                  true,
                                                                  cg.explore_repeat,
                                                                  cg.explore_k,
                                                                  cg.explore_threads,
                                                                  nullptr,
                                                                  ds.info().explore_depth,
                                                                  linear_baseline_us);
                        }
                    } else {
                        log("Graph file not found: {}\n", graph_path);
                    }

                    deglib::benchmark::reset_log_to_console();
                    log("{}: Log written to: {}\n", DatasetConfig::data_stream_type_str(ds_type), log_path);
                }
            }
        }
    }

    // REDUCE_SCALING test (build full graph and reduce it incrementally)
    if (run_all || test_type_arg == "reduce_scaling") {
        const auto& rs = config.reduce_scaling_test;
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;

        // Test both schemes: original cg.lid and StreamingData
        for (deglib::builder::OptimizationTarget lid_scheme : {cg.lid, deglib::builder::OptimizationTarget::StreamingData_SchemeC}) {
            std::string scaling_dir = graph_paths.reduce_scaling_directory(lid_scheme);
            std::string log_path = graph_paths.reduce_scaling_log_file(lid_scheme);
            std::string scheme = DatasetConfig::optimization_target_str(lid_scheme);

            log("\n=== REDUCE_SCALING Test: {} ===\n", scheme);
            log("Size interval: {}\n", rs.size_interval);
            log("Directory: {}\n", scaling_dir);
            log("Log file: {}\n", log_path);

            if (do_run && base_repository && query_repository) {
                // Skip entire scenario if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("REDUCE_SCALING {}: Skipping - log file already exists: {}\n", scheme, log_path);
                    continue;
                } else {
                    // Ensure scaling directory exists
                    std::filesystem::create_directories(scaling_dir);

                    // Set log file for all reduce scaling operations
                    deglib::benchmark::set_log_file(log_path, false);
                    log("\n=== REDUCE_SCALING Test: {} ===\n", scheme);
                    log("Size interval: {}\n", rs.size_interval);

                    for (const auto& [k_opt, eps_opt, i_opt] : std::array<std::tuple<uint8_t, float, uint8_t>, 2>{
                             std::make_tuple(og.k_opt, og.eps_opt, og.i_opt),
                             std::make_tuple(static_cast<uint8_t>(0), 0.0f, static_cast<uint8_t>(0))}) {
                        log("----------------------------------------------------------------\n");
                        log("---- REDUCE_SCALING with k_opt={}, eps_opt={:.4f}, i_opt={} ----\n", k_opt, eps_opt, i_opt);

                        // Generate base name for graph files
                        std::string metric_str = (config.metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
                        std::string graph_name_base = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}_ReduceWith_OptK{}Eps{:.3f}Path{}",
                                                                  dims,
                                                                  metric_str,
                                                                  cg.k,
                                                                  cg.k_ext,
                                                                  cg.eps_ext,
                                                                  scheme,
                                                                  k_opt,
                                                                  eps_opt,
                                                                  i_opt);

                        // Load or build the full graph first
                        std::string full_graph_path = graph_paths.dynamic_graph_file(dims,
                                                                                     config.metric,
                                                                                     lid_scheme,
                                                                                     cg.k,
                                                                                     cg.k_ext,
                                                                                     cg.eps_ext,
                                                                                     og.k_opt,
                                                                                     og.eps_opt,
                                                                                     og.i_opt,
                                                                                     DataStreamType::AddAll);
                        if (!std::filesystem::exists(full_graph_path)) {
                            log("\n--- Building full graph first ---\n");
                            deglib::benchmark::create_graph(*base_repository,
                                                            DataStreamType::AddAll,
                                                            full_graph_path,
                                                            config.metric,
                                                            lid_scheme,
                                                            cg.k,
                                                            cg.k_ext,
                                                            cg.eps_ext,
                                                            og.k_opt,
                                                            og.eps_opt,
                                                            og.i_opt,
                                                            cg.build_threads,
                                                            true,
                                                            ds.info().scale);
                        }

                        // Reduce graph incrementally
                        auto created_files = [&]() {
                            log("\n--- Loading full graph ---\n");
                            auto graph = deglib::graph::load_sizebounded_graph(full_graph_path.c_str());
                            log("Loaded graph with {} vertices\n", graph.size());

                            log("\n--- Reducing graph incrementally ---\n");
                            auto files = deglib::benchmark::reduce_graph_incremental(graph,
                                                                                     scaling_dir,
                                                                                     graph_name_base,
                                                                                     rs.size_interval,
                                                                                     k_opt,
                                                                                     eps_opt,
                                                                                     i_opt,
                                                                                     cg.build_threads,
                                                                                     true,
                                                                                     ds.info().scale);
                            log("Created {} reduced graphs\n", files.size());
                            return files;
                        }();

                        // Test all graphs
                        log("\n--- Testing graphs ---\n");
                        for (const auto& [graph_path, vertex_count] : created_files) {
                            log("\n=== REDUCE_SCALING Test: {} size={} ===\n", scheme, vertex_count);
                            log("Graph: {}\n", graph_path);

                            if (std::filesystem::exists(graph_path)) {
                                const auto test_graph = deglib::graph::load_readonly_graph(graph_path.c_str());

                                // dont test an empty graph
                                if (test_graph.size() == 0) continue;

                                // Load ground truth for this specific size
                                auto ground_truth = ds.load_groundtruth_for_size(cg.anns_k, vertex_count);

                                wait_before_test();
                                deglib::benchmark::test_graph_anns(test_graph,
                                                                   *query_repository,
                                                                   ground_truth,
                                                                   cg.anns_repeat,
                                                                   cg.anns_threads,
                                                                   cg.anns_k,
                                                                   cg.eps_parameter,
                                                                   nullptr,
                                                                   linear_baseline_us);
                            } else {
                                log("Graph file not found: {}\n", graph_path);
                            }
                        }
                    }

                    deglib::benchmark::reset_log_to_console();
                    log("REDUCE_SCALING {}: Log written to: {}\n", scheme, log_path);
                }
            }
        }
    }

    // REDUCE_SCALING_MEASURE test
    // This reuses the reduce_scaling configuration but runs the incremental measurement in a separate log file.
    if (run_all || test_type_arg == "reduce_scaling") {
        const auto& cg = config.create_graph;
        const auto& og = config.optimize_graph;
        const auto& ss = config.size_scaling_test;

        for (deglib::builder::OptimizationTarget lid_scheme : {cg.lid, deglib::builder::OptimizationTarget::StreamingData_SchemeC}) {
            std::string reduce_dir = graph_paths.reduce_scaling_directory(lid_scheme);
            std::string log_path = reduce_dir + "/log_measured.txt";
            std::string scheme = DatasetConfig::optimization_target_str(lid_scheme);

            // Get size scaling directory where the pre-built graphs are
            std::string size_scaling_dir = graph_paths.size_scaling_directory(lid_scheme);

            log("\n=== REDUCE_SCALING_MEASURE Test: {} ===\n", scheme);
            log("Size interval: {}\n", ss.size_interval);
            log("Directory: {}\n", reduce_dir);
            log("Log file: {}\n", log_path);
            log("Loading graphs from: {}\n", size_scaling_dir);

            if (do_run && base_repository) {
                // Skip if log file already exists
                if (std::filesystem::exists(log_path)) {
                    log("REDUCE_SCALING_MEASURE {}: Skipping - log file already exists: {}\n", scheme, log_path);
                    continue;
                } else {
                    // Ensure scaling directory exists
                    std::filesystem::create_directories(reduce_dir);

                    // Set log file for measurement
                    deglib::benchmark::set_log_file(log_path, false);
                    log("\n=== REDUCE_SCALING_MEASURE Test: {} ===\n", scheme);

                    // Generate base name for expected size scaling graph files
                    std::string metric_str = (config.metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
                    std::string graph_name_base = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}_OptK{}Eps{:.3f}Path{}",
                                                              dims,
                                                              metric_str,
                                                              cg.k,
                                                              cg.k_ext,
                                                              cg.eps_ext,
                                                              scheme,
                                                              og.k_opt,
                                                              og.eps_opt,
                                                              og.i_opt);

                    for (const auto& [k_opt, eps_opt, i_opt] : std::array<std::tuple<uint8_t, float, uint8_t>, 2>{
                             std::make_tuple(og.k_opt, og.eps_opt, og.i_opt),
                             std::make_tuple(static_cast<uint8_t>(0), 0.0f, static_cast<uint8_t>(0))}) {
                        log("----------------------------------------------------------------\n");
                        log("---- REDUCE_SCALING_MEASURE with k_opt={}, eps_opt={:.4f}, i_opt={} ----\n", k_opt, eps_opt, i_opt);

                        deglib::benchmark::measure_reduction_on_prebuilt_graphs(size_scaling_dir,
                                                                                graph_name_base,
                                                                                ss.size_interval,
                                                                                (uint32_t)base_repository->size(),
                                                                                k_opt,
                                                                                eps_opt,
                                                                                i_opt,
                                                                                cg.build_threads,
                                                                                lid_scheme,
                                                                                true,
                                                                                ds.info().scale);
                    }

                    deglib::benchmark::reset_log_to_console();
                    log("REDUCE_SCALING_MEASURE {}: Log written to: {}\n", scheme, log_path);
                }
            }
        }
    }

    log("\nTest OK\n");
    return 0;
}
