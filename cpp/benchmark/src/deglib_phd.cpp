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
#include "deglib.h"
#include "analysis.h"



// Use the shared DataStreamType from benchmark.h
using deglib::benchmark::DataStreamType;
using deglib::benchmark::log;

// ============================================================================
// Helper functions for enum to string conversion
// ============================================================================
static inline std::string to_str_data_stream(const DataStreamType ds) {
    switch(ds) {
        case DataStreamType::AddAll: return "AddAll";
        case DataStreamType::AddHalf: return "AddHalf";
        case DataStreamType::AddAllRemoveHalf: return "AddAllRemoveHalf";
        case DataStreamType::AddHalfRemoveAndAddOneAtATime: return "AddHalfRemoveAndAddOneAtATime";
        default: return "Unknown";
    }
}

static inline std::string to_str_metric(const deglib::Metric m) {
    switch(m) {
        case deglib::Metric::L2: return "L2";
        case deglib::Metric::L2_Uint8: return "L2_Uint8";
        default: return "UnknownMetric";
    }
}

static inline std::string to_str_lid(const deglib::builder::OptimizationTarget t) {
    switch(t) {
        case deglib::builder::OptimizationTarget::LowLID: return "LowLID";
        case deglib::builder::OptimizationTarget::HighLID: return "HighLID";
        case deglib::builder::OptimizationTarget::StreamingData: return "StreamingData";
        case deglib::builder::OptimizationTarget::SchemeA: return "SchemeA";
        case deglib::builder::OptimizationTarget::SchemeB: return "SchemeB";
        default: return "UnknownLID";
    }
}

// Use Dataset and DatasetName from dataset.h
using deglib::benchmark::Dataset;
using deglib::benchmark::DatasetName;

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
};

struct OptimizeGraphTest {
    uint8_t k_opt = 30;
    float eps_opt = 0.001f;
    uint8_t i_opt = 5;
    uint32_t total_iterations = 100000;     // Total optimization iterations
    uint32_t save_interval = 20000;          // Save graph every N iterations
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
    std::vector<uint8_t> k_values;  // e.g., {10, 20, 30, 40, 50}
    std::vector<float> eps_parameter = { 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f };
};

/**
 * KOptSweepTest: Sweep different k_ext values for graph construction.
 * Builds graphs in "kExtScaling" subdirectory.
 */
struct KExtSweepTest {
    std::vector<uint8_t> k_ext_values;  // e.g., {30, 60, 90, 120}
    std::vector<float> eps_parameter = { 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f };
};

/**
 * EpsExtSweepTest: Sweep different eps_ext values for graph construction.
 * Builds graphs in "epsExtScaling" subdirectory.
 */
struct EpsExtSweepTest {
    std::vector<float> eps_ext_values;  // e.g., {0.05, 0.1, 0.15, 0.2}
    std::vector<float> eps_parameter = { 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f };
};

/**
 * SizeScalingTest: Build and test graphs with different amounts of data.
 * Builds graphs in "sizeScaling" subdirectory.
 * All graphs are built first, then tested with their respective ground truth.
 * Uses a single log file for all size variants.
 */
struct SizeScalingTest {
    std::vector<uint32_t> size_values;  // e.g., {100000, 200000, 500000, 1000000}
    std::vector<float> eps_parameter = { 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f };
};

// Test with RNG disabled (deterministic graph construction)
struct RNGDisabledTest {
    uint8_t k = 30;
    uint8_t k_ext = 60;
    float eps_ext = 0.1f;
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::LowLID;
    bool rng_disabled = true;
};



struct ReduceGraphTest {
    uint8_t target_k = 20;  // Reduce graph to this degree
};

// Main dataset configuration
struct DatasetConfig {
    // Dataset identity (using DatasetName class)
    DatasetName dataset_name = DatasetName::Invalid;
    
    // Common settings
    DataStreamType data_stream_type = DataStreamType::AddAll;
    deglib::Metric metric = deglib::Metric::L2;
    
    // Test eps parameters for ANNS benchmark (search)
    std::vector<float> eps_parameter = { 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f };
    
    // Optional test types - test runs only if optional has value
    std::optional<CreateGraphTest> create_graph;
    std::optional<OptimizeGraphTest> optimize_graph;
    std::optional<ReduceGraphTest> reduce_graph;
    
    // Additional comprehensive test types
    std::optional<AllSchemesTest> all_schemes_test;
    std::optional<KSweepTest> k_sweep_test;
    std::optional<KExtSweepTest> k_ext_sweep_test;
    std::optional<EpsExtSweepTest> eps_ext_sweep_test;
    std::optional<SizeScalingTest> size_scaling_test;
    std::optional<RNGDisabledTest> rng_disabled_test;
};


// Path building utilities for graph files
struct GraphPaths {
    std::filesystem::path graph_dir;
    
    GraphPaths(const Dataset& ds) 
        : graph_dir(ds.data_root() / ds.name() / "deg") {}
    
    std::string graph_directory() const {
        return graph_dir.string();
    }
    
    std::string graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string filename = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}.deg", dims, metric_str, k, k_ext, eps_ext, scheme);
        return (graph_dir / filename).string();
    }
    
    // Subdirectory for optimization checkpoints
    std::string opt_directory(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                              deglib::builder::OptimizationTarget lid, uint8_t k_opt, float eps_opt, uint8_t i_opt) const {
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string dir_name = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}_OptK{}Eps{:.4f}Path{}", 
                                            dims, metric_str, k, k_ext, eps_ext, scheme, k_opt, eps_opt, i_opt);
        return (graph_dir / dir_name).string();
    }
    
    std::string opt_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                                deglib::builder::OptimizationTarget lid, uint8_t k_opt, float eps_opt, uint8_t i_opt, uint32_t iterations) const {
        std::string opt_dir = opt_directory(dims, metric, k, k_ext, eps_ext, lid, k_opt, eps_opt, i_opt);
        std::string filename = fmt::format("it{}.deg", iterations);
        return (std::filesystem::path(opt_dir) / filename).string();
    }
    
    // Log file for a graph (same directory as graph)
    std::string graph_log_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string filename = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}_log.txt", dims, metric_str, k, k_ext, eps_ext, scheme);
        return (graph_dir / filename).string();
    }
    
    // Log file for optimization checkpoints (in opt subdirectory)
    std::string opt_log_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                              deglib::builder::OptimizationTarget lid, uint8_t k_opt, float eps_opt, uint8_t i_opt) const {
        std::string opt_dir = opt_directory(dims, metric, k, k_ext, eps_ext, lid, k_opt, eps_opt, i_opt);
        return (std::filesystem::path(opt_dir) / "log.txt").string();
    }
    
    // Scaling test directories
    std::string k_scaling_directory() const {
        return (graph_dir / "kScaling").string();
    }
    
    std::string k_ext_scaling_directory() const {
        return (graph_dir / "kExtScaling").string();
    }
    
    std::string eps_ext_scaling_directory() const {
        return (graph_dir / "epsExtScaling").string();
    }
    
    std::string size_scaling_directory() const {
        return (graph_dir / "sizeScaling").string();
    }
    
    // Graph file in scaling directory with size parameter
    std::string size_scaling_graph_file(uint32_t dims, deglib::Metric metric, 
                                         uint8_t k, uint8_t k_ext, float eps_ext, 
                                         deglib::builder::OptimizationTarget lid, uint32_t size) const {
        std::string scaling_dir = size_scaling_directory();
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string filename = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}_N{}.deg", dims, metric_str, k, k_ext, eps_ext, scheme, size);
        return (std::filesystem::path(scaling_dir) / filename).string();
    }
    
    // Single log file for all size scaling tests
    std::string size_scaling_log_file() const {
        return (std::filesystem::path(size_scaling_directory()) / "log.txt").string();
    }
    
    // Graph file in scaling directory
    std::string scaling_graph_file(const std::string& scaling_dir, uint32_t dims, deglib::Metric metric, 
                                    uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string filename = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}.deg", dims, metric_str, k, k_ext, eps_ext, scheme);
        return (std::filesystem::path(scaling_dir) / filename).string();
    }
    
    // Log file in scaling directory
    std::string scaling_log_file(const std::string& scaling_dir, uint32_t dims, deglib::Metric metric,
                                  uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string filename = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}_log.txt", dims, metric_str, k, k_ext, eps_ext, scheme);
        return (std::filesystem::path(scaling_dir) / filename).string();
    }
};

// Dataset configuration factory
static DatasetConfig get_dataset_config(const DatasetName& dataset_name) {
    DatasetConfig conf{};
    conf.dataset_name = dataset_name;
    conf.data_stream_type = DataStreamType::AddAll;
    conf.metric = deglib::Metric::L2;

    if (dataset_name == DatasetName::GLOVE) {
        conf.eps_parameter = { 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f, 1.2f };
        
        // Create graph test with HighLID default for glove
        CreateGraphTest cg;
        cg.lid = deglib::builder::OptimizationTarget::HighLID;
        cg.k = 30;
        cg.k_ext = 60;
        conf.create_graph = cg;
        
        // Optimize graph test
        OptimizeGraphTest og;
        og.k_opt = 30;
        conf.optimize_graph = og;
        
    } else if (dataset_name == DatasetName::SIFT1M) {
        conf.eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f };
        
        CreateGraphTest cg;
        cg.k = 30;
        cg.k_ext = 60;
        conf.create_graph = cg;
        
        OptimizeGraphTest og;
        og.k_opt = 30;
        conf.optimize_graph = og;
        
        // All 5 schemes test with per-scheme eps_parameter
        AllSchemesTest as;
        as.eps_parameter = {
            {deglib::builder::OptimizationTarget::LowLID, conf.eps_parameter},
            {deglib::builder::OptimizationTarget::HighLID, conf.eps_parameter},
            {deglib::builder::OptimizationTarget::StreamingData, conf.eps_parameter},
            {deglib::builder::OptimizationTarget::SchemeA, conf.eps_parameter},
            {deglib::builder::OptimizationTarget::SchemeB, conf.eps_parameter}
        };
        conf.all_schemes_test = as;
        
        // K sweep test (varies k values)
        KSweepTest ks;
        ks.k_values = {10, 20, 30, 40, 50};
        ks.eps_parameter = conf.eps_parameter;
        conf.k_sweep_test = ks;
        
        // K_ext sweep test (varies k_ext values)
        KExtSweepTest kes;
        kes.k_ext_values = {30, 60, 90, 120};
        kes.eps_parameter = conf.eps_parameter;
        conf.k_ext_sweep_test = kes;
        
        // Eps_ext sweep test (varies eps_ext values)
        EpsExtSweepTest ees;
        ees.eps_ext_values = {0.0f, 0.1f, 0.2f, 0.3f};
        ees.eps_parameter = conf.eps_parameter;
        conf.eps_ext_sweep_test = ees;
        
        // Size scaling test (varies dataset size)
        SizeScalingTest ss;
        ss.size_values = {100000, 200000, 500000, 1000000};
        ss.eps_parameter = conf.eps_parameter;
        conf.size_scaling_test = ss;
        
        // RNG disabled test
        RNGDisabledTest rng;
        rng.k = 30;
        conf.rng_disabled_test = rng;
        
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.eps_parameter = { 0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f };
        
        CreateGraphTest cg;
        cg.k = 30;
        cg.k_ext = 60;
        conf.create_graph = cg;
        
        OptimizeGraphTest og;
        og.k_opt = 30;
        conf.optimize_graph = og;
        
    } else if (dataset_name == DatasetName::AUDIO) {
        conf.eps_parameter = { 0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f };
        
        CreateGraphTest cg;
        cg.k = 20;
        cg.k_ext = 40;
        cg.anns_repeat = 50;  // More repeats for small dataset
        conf.create_graph = cg;
        
        OptimizeGraphTest og;
        og.k_opt = 20;
        conf.optimize_graph = og;
        
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
    std::string test_type_arg = "all";
    bool do_run = false;
    
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "help" || arg == "--help") {
            log("Usage: deglib_phd <dataset> [test_type] [--run]\n");
            log("Datasets: sift1m, deep1m, audio, glove\n");
            log("Test types:\n");
            log("  create_graph    - Build graph, run stats, ANNS, explore, k-sweep\n");
            log("  optimize_graph  - Optimize existing graph\n");
            log("  reduce_graph    - Reduce graph degree\n");
            log("  all_schemes     - Test all 5 OptimizationTargets\n");
            log("  k_sweep         - Sweep k values for graph construction\n");
            log("  k_ext_sweep     - Sweep k_ext values for graph construction\n");
            log("  eps_ext_sweep   - Sweep eps_ext values for graph construction\n");
            log("  size_scaling    - Build/test graphs with different data sizes\n");
            log("  all             - Run all available tests\n");
            log("Options: --run  Actually execute tests (otherwise just show config)\n");
            return 0;
        }
        if(arg == "--run") { do_run = true; continue; }
        
        // Try to parse as dataset name
        auto parsed_ds = DatasetName::from_string(arg);
        if (parsed_ds.is_valid()) {
            ds_name = parsed_ds;
        } else if(arg == "create_graph" || arg == "optimize_graph" || arg == "reduce_graph" || 
                arg == "all_schemes" || arg == "k_sweep" || arg == "k_ext_sweep" || 
                arg == "eps_ext_sweep" || arg == "size_scaling" || arg == "all") {
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
    log("Metric: {}\n", to_str_metric(config.metric));
    log("DataStreamType: {}\n", to_str_data_stream(config.data_stream_type));
    log("Eps parameters: [{}]\n", fmt::join(config.eps_parameter, ", "));
    
    // List available test types for this dataset
    log("\nAvailable test types for '{}':\n", ds.name());
    if(config.create_graph) {
        const auto& cg = *config.create_graph;
        log(" - create_graph (k={}, k_ext={}, eps_ext={:.2f}, lid={})\n", 
            cg.k, cg.k_ext, cg.eps_ext, to_str_lid(cg.lid));
        log("     ANNS: k={}, repeat={}, threads={}\n", cg.anns_k, cg.anns_repeat, cg.anns_threads);
        log("     Explore: k={}\n", cg.explore_k);
        log("     k-sweep: [{}]\n", fmt::join(cg.k_sweep_values, ", "));
    }
    if(config.optimize_graph) log(" - optimize_graph (k_opt={}, eps_opt={:.4f}, i_opt={}, total_it={}, save_interval={})\n",
        config.optimize_graph->k_opt, config.optimize_graph->eps_opt, config.optimize_graph->i_opt,
        config.optimize_graph->total_iterations, config.optimize_graph->save_interval);
    if(config.reduce_graph) log(" - reduce_graph (target_k={})\n", config.reduce_graph->target_k);
    
    // Additional test types
    if(config.all_schemes_test) log(" - all_schemes (tests 5 OptimizationTargets)\n");
    if(config.k_sweep_test) log(" - k_sweep (k_values=[{}], dir=kScaling)\n", 
        fmt::join(config.k_sweep_test->k_values, ", "));
    if(config.k_ext_sweep_test) log(" - k_ext_sweep (k_ext_values=[{}], dir=kExtScaling)\n",
        fmt::join(config.k_ext_sweep_test->k_ext_values, ", "));
    if(config.eps_ext_sweep_test) log(" - eps_ext_sweep (eps_ext_values=[{}], dir=epsExtScaling)\n",
        fmt::join(config.eps_ext_sweep_test->eps_ext_values, ", "));
    if(config.size_scaling_test) log(" - size_scaling (size_values=[{}], dir=sizeScaling)\n",
        fmt::join(config.size_scaling_test->size_values, ", "));
    if(config.rng_disabled_test) log(" - rng_disabled (k={}, rng=off)\n", config.rng_disabled_test->k);

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
    if((run_all || test_type_arg == "create_graph") && config.create_graph) {
        const auto& cg = *config.create_graph;
        
        log("\n=== CREATE_GRAPH Test ===\n");
        log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
            cg.k, cg.k_ext, cg.eps_ext, to_str_lid(cg.lid), cg.build_threads);
        
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
                    cg.k, cg.k_ext, cg.eps_ext, to_str_lid(cg.lid), cg.build_threads);
                log("Output graph: {}\n", graph_path);
                // TODO: Implement create_graph logic
                log("create_graph execution not yet implemented\n");
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
                    cg.anns_repeat, cg.anns_threads, cg.anns_k, config.eps_parameter);
                
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
                        cg.anns_repeat, cg.anns_threads, k_val, config.eps_parameter);
                }
            }
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    // OPTIMIZE_GRAPH test (interval-based optimization with checkpoints)
    if((run_all || test_type_arg == "optimize_graph") && config.optimize_graph && config.create_graph) {
        const auto& cg = *config.create_graph;
        const auto& og = *config.optimize_graph;
        
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
            deglib::benchmark::set_log_file(log_path, true);
            log("\n=== OPTIMIZE_GRAPH Test ===\n");
            log("Input graph: {}\n", input_graph);
            log("Output directory: {}\n", opt_dir);
            log("Settings: k_opt={}, eps_opt={:.4f}, i_opt={}, total_iterations={}, save_interval={}\n",
                og.k_opt, og.eps_opt, og.i_opt, og.total_iterations, og.save_interval);

            // Load ground truth for testing
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);
            
            // Run optimization in intervals
            for(uint32_t it = og.save_interval; it <= og.total_iterations; it += og.save_interval) {
                std::string checkpoint_graph = graph_paths.opt_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, og.k_opt, og.eps_opt, og.i_opt, it);
                
                log("\n--- Iteration {} ---\n", it);
                
                if(std::filesystem::exists(checkpoint_graph)) {
                    log("Checkpoint exists: {}\n", checkpoint_graph);
                } else {
                    // TODO: Run optimization from previous checkpoint to current iteration
                    log("Optimization not yet implemented\n");
                }
                
                // Test checkpoint if it exists
                if(std::filesystem::exists(checkpoint_graph)) {
                    const auto graph = deglib::graph::load_readonly_graph(checkpoint_graph.c_str());
                    log("Loaded checkpoint: {} vertices\n", graph.size());
                    
                    // 2. ANNS Test with Top-k
                    log("\n-- ANNS Test (k=100) --\n");
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, config.eps_parameter);
                }
            }
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    // REDUCE_GRAPH test
    if((run_all || test_type_arg == "reduce_graph") && config.reduce_graph) {
        const auto& rg = *config.reduce_graph;
        
        log("\n=== REDUCE_GRAPH Test ===\n");
        log("Settings: target_k={}\n", rg.target_k);
        
        if(do_run && config.create_graph) {
            const auto& cg = *config.create_graph;
            std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string log_path = graph_paths.graph_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            
            deglib::benchmark::set_log_file(log_path, true);
            log("\n=== REDUCE_GRAPH Test ===\n");
            log("Input graph: {}\n", graph_path);
            log("Target k: {}\n", rg.target_k);
            // TODO: Implement reduce_graph logic
            log("reduce_graph execution not yet implemented\n");
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    // ALL_SCHEMES test (tests all 5 OptimizationTargets with CreateGraphTest parameters)
    if((run_all || test_type_arg == "all_schemes") && config.all_schemes_test && config.create_graph) {
        const auto& as = *config.all_schemes_test;
        const auto& cg = *config.create_graph;
        
        log("\n=== ALL_SCHEMES Test ===\n");
        log("Testing all 5 OptimizationTargets with k={}, k_ext={}, eps_ext={:.2f}\n", cg.k, cg.k_ext, cg.eps_ext);
        
        if(do_run && base_repository && query_repository) {
            // Load ground truth for this test
            auto ground_truth = ds.load_groundtruth(cg.anns_k, false);
            
            std::vector<deglib::builder::OptimizationTarget> all_targets = {
                deglib::builder::OptimizationTarget::LowLID,
                deglib::builder::OptimizationTarget::HighLID,
                deglib::builder::OptimizationTarget::StreamingData,
                deglib::builder::OptimizationTarget::SchemeA,
                deglib::builder::OptimizationTarget::SchemeB
            };
            
            for(auto lid : all_targets) {
                std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, lid);
                std::string log_path = graph_paths.graph_log_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, lid);
                deglib::benchmark::set_log_file(log_path, true);
                
                log("\n=== ALL_SCHEMES Test: {} ===\n", to_str_lid(lid));
                log("Graph: {}\n", graph_path);
                
                // Get per-scheme eps_parameter, fall back to config default
                auto eps_it = as.eps_parameter.find(lid);
                const auto& scheme_eps = (eps_it != as.eps_parameter.end()) ? eps_it->second : config.eps_parameter;
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, 
                        cg.anns_repeat, cg.anns_threads, cg.anns_k, scheme_eps);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("{}: Log written to: {}\n", to_str_lid(lid), log_path);
            }
        }
    }
    
    // K_SWEEP test (sweep k values for graph construction in kScaling directory)
    if((run_all || test_type_arg == "k_sweep") && config.k_sweep_test && config.create_graph) {
        const auto& ks = *config.k_sweep_test;
        const auto& cg = *config.create_graph;
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
    if((run_all || test_type_arg == "k_ext_sweep") && config.k_ext_sweep_test && config.create_graph) {
        const auto& kes = *config.k_ext_sweep_test;
        const auto& cg = *config.create_graph;
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
    if((run_all || test_type_arg == "eps_ext_sweep") && config.eps_ext_sweep_test && config.create_graph) {
        const auto& ees = *config.eps_ext_sweep_test;
        const auto& cg = *config.create_graph;
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
    
    // SIZE_SCALING test (build and test graphs with different data sizes)
    if((run_all || test_type_arg == "size_scaling") && config.size_scaling_test && config.create_graph) {
        const auto& ss = *config.size_scaling_test;
        const auto& cg = *config.create_graph;
        std::string scaling_dir = graph_paths.size_scaling_directory();
        std::string log_path = graph_paths.size_scaling_log_file();
        
        log("\n=== SIZE_SCALING Test ===\n");
        log("Testing size values: [{}]\n", fmt::join(ss.size_values, ", "));
        log("Directory: {}\n", scaling_dir);
        log("Log file: {}\n", log_path);
        
        if(do_run && base_repository && query_repository) {
            // Ensure scaling directory exists
            std::filesystem::create_directories(scaling_dir);
            
            // Set log file for all size scaling operations
            deglib::benchmark::set_log_file(log_path, false);
            log("\n=== SIZE_SCALING Test ===\n");
            log("Testing size values: [{}]\n", fmt::join(ss.size_values, ", "));
            
            // Phase 1: Build all graphs
            log("\n--- Phase 1: Building graphs ---\n");
            for(uint32_t size : ss.size_values) {
                std::string graph_path = graph_paths.size_scaling_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, size);
                log("\nBuilding graph for size={}: {}\n", size, graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    log("Graph already exists, skipping build\n");
                } else {
                    // TODO: Build graph with limited data
                    // This requires creating a subset repository and building the graph
                    log("Graph building not yet implemented for size scaling\n");
                }
            }
            
            // Phase 2: Test all graphs
            log("\n--- Phase 2: Testing graphs ---\n");
            for(uint32_t size : ss.size_values) {
                std::string graph_path = graph_paths.size_scaling_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid, size);
                log("\n=== SIZE_SCALING Test: size={} ===\n", size);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    
                    // Load ground truth specific to this size
                    // Ground truth file pattern: groundtruth_N{size}.ivecs or similar
                    auto ground_truth = ds.load_groundtruth(cg.anns_k, use_half_gt);  // TODO: Load size-specific ground truth
                    
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
    
    log("\nTest OK\n");
    return 0;
}
