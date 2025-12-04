#include <random>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <optional>
#include <filesystem>
#include <atomic>

#include <fmt/core.h>
#include <omp.h>

#include "benchmark.h"
#include "stats.h"
#include "dataset.h"
#include "deglib.h"
#include "analysis.h">



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
struct CreateGraphTest {
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::LowLID;
    uint8_t k = 30;
    uint8_t k_ext = 60;
    float eps_ext = 0.1f;
    uint32_t thread_count = 1;
};

struct OptimizeGraphTest {
    uint8_t k_opt = 30;
    float eps_opt = 0.001f;
    uint8_t i_opt = 5;
    uint32_t iterations = 20000;
};

struct ReduceGraphTest {
    uint8_t target_k = 20;  // Reduce graph to this degree
};

struct TestGraphTest {
    uint32_t repeat = 1;
    uint32_t test_threads = 1;
    uint32_t k_test = 100;
};

struct ScalingTest {
    std::vector<uint8_t> k_values;  // e.g., {20, 30, 40, 60, 80}
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::LowLID;
    float eps_ext = 0.1f;
    uint32_t thread_count = 1;
    uint32_t repeat = 1;
    uint32_t test_threads = 1;
    uint32_t k_test = 100;
};

struct KSweepTest {
    std::vector<uint8_t> k_values;  // k values to test
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::LowLID;
    float eps_ext = 0.1f;
};

struct LIDComparisonTest {
    uint8_t k = 30;
    float eps_ext = 0.1f;
    // Tests both LowLID and HighLID schemes
};

// New comprehensive test types

// Test all 5 optimization schemes with same k
struct AllSchemesTest {
    uint8_t k = 30;
    uint8_t k_ext = 60;
    float eps_ext = 0.1f;
    uint32_t thread_count = 1;
    // Will test: LowLID, HighLID, StreamingData, SchemeA, SchemeB
};

// Sweep different k_opt values for optimization
struct KOptSweepTest {
    std::vector<uint8_t> k_opt_values;  // e.g., {10, 20, 30, 40, 50}
    float eps_opt = 0.001f;
    uint8_t i_opt = 5;
    uint32_t iterations = 20000;
};

// Sweep different eps_opt values for optimization
struct EpsOptSweepTest {
    std::vector<float> eps_opt_values;  // e.g., {0.0001, 0.001, 0.01, 0.1}
    uint8_t k_opt = 30;
    uint8_t i_opt = 5;
    uint32_t iterations = 20000;
};

// Iterative optimization - save graph at intervals
struct IterativeOptTest {
    uint8_t k_opt = 30;
    float eps_opt = 0.001f;
    uint8_t i_opt = 5;
    uint32_t total_iterations = 100000;
    uint32_t save_interval = 10000;  // Save every N iterations
};

// Test with RNG disabled (deterministic graph construction)
struct RNGDisabledTest {
    uint8_t k = 30;
    uint8_t k_ext = 60;
    float eps_ext = 0.1f;
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::LowLID;
    bool rng_disabled = true;
};

// Exploration benchmark test
struct ExploreTest {
    uint32_t k_explore = 1000;  // Top-k for exploration
    uint32_t query_count = 1000;
    bool include_entry = false;
    uint32_t repeat = 1;
    uint32_t threads = 1;
};

// Comprehensive statistics test
struct StatsTest {
    bool compute_graph_quality = true;      // Needs top-list file
    bool compute_seed_reachability = true;  // Expensive
    bool compute_avg_reach = true;          // Very expensive
    bool check_connectivity = true;
    bool check_regularity = true;
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
    std::optional<TestGraphTest> test_graph;
    std::optional<ScalingTest> scaling_test;
    std::optional<KSweepTest> k_sweep_test;
    std::optional<LIDComparisonTest> lid_comparison_test;
    
    // New comprehensive test types
    std::optional<AllSchemesTest> all_schemes_test;
    std::optional<KOptSweepTest> k_opt_sweep_test;
    std::optional<EpsOptSweepTest> eps_opt_sweep_test;
    std::optional<IterativeOptTest> iterative_opt_test;
    std::optional<RNGDisabledTest> rng_disabled_test;
    std::optional<ExploreTest> explore_test;
    std::optional<StatsTest> stats_test;
};


// Path building utilities for graph files
struct GraphPaths {
    std::filesystem::path data_path;
    std::string name;
    
    GraphPaths(const std::filesystem::path& base, const DatasetName& ds) 
        : data_path(base), name(ds.name()) {}
    
    GraphPaths(const std::filesystem::path& base, const Dataset& ds) 
        : data_path(base), name(ds.name()) {}
    
    std::string graph_directory() const {
        return (data_path / "deg" / name).string();
    }
    
    std::string graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, deglib::builder::OptimizationTarget lid) const {
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string filename = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}.deg", dims, metric_str, k, k_ext, eps_ext, scheme);
        return (data_path / "deg" / name / filename).string();
    }
    
    std::string opt_graph_file(uint32_t dims, deglib::Metric metric, uint8_t k, uint8_t k_ext, float eps_ext, 
                                deglib::builder::OptimizationTarget lid, uint8_t k_opt, float eps_opt, uint8_t i_opt, uint32_t iterations) const {
        std::string metric_str = (metric == deglib::Metric::L2) ? "L2" : "L2_Uint8";
        std::string scheme = to_str_lid(lid);
        std::string base_name = fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}", dims, metric_str, k, k_ext, eps_ext, scheme);
        std::string opt_suffix = fmt::format("_OptK{}Eps{:.3f}Path{}_it{}.deg", k_opt, eps_opt, i_opt, iterations);
        return (data_path / "deg" / name / (base_name + opt_suffix)).string();
    }
    
    // Log file path for a specific test
    std::string log_file(const std::string& test_name) const {
        return (data_path / "deg" / name / fmt::format("{}_log.txt", test_name)).string();
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
        
        // Test graph
        TestGraphTest tg;
        conf.test_graph = tg;
        
        // K-sweep test
        KSweepTest ks;
        ks.k_values = {20, 30, 40, 80};
        ks.lid = deglib::builder::OptimizationTarget::HighLID;
        conf.k_sweep_test = ks;
        
        // LID comparison test
        LIDComparisonTest lc;
        lc.k = 30;
        conf.lid_comparison_test = lc;
    } else if (dataset_name == DatasetName::SIFT1M) {
        conf.eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f };
        
        CreateGraphTest cg;
        cg.k = 30;
        cg.k_ext = 60;
        conf.create_graph = cg;
            
            OptimizeGraphTest og;
            og.k_opt = 30;
            conf.optimize_graph = og;
            
            TestGraphTest tg;
            conf.test_graph = tg;
            
            // Scaling test: test different k values
            ScalingTest st;
            st.k_values = {20, 30, 40, 80};
            conf.scaling_test = st;
            
            // LID comparison test
            LIDComparisonTest lc;
            lc.k = 30;
            conf.lid_comparison_test = lc;
            
            // All 5 schemes test
            AllSchemesTest as;
            as.k = 30;
            as.k_ext = 60;
            conf.all_schemes_test = as;
            
            // K_opt sweep test
            KOptSweepTest kos;
            kos.k_opt_values = {10, 20, 30, 40, 50};
            conf.k_opt_sweep_test = kos;
            
            // Eps_opt sweep test
            EpsOptSweepTest eos;
            eos.eps_opt_values = {0.0001f, 0.001f, 0.01f, 0.1f};
            conf.eps_opt_sweep_test = eos;
            
            // Iterative optimization test
            IterativeOptTest io;
            io.total_iterations = 100000;
            io.save_interval = 20000;
            conf.iterative_opt_test = io;
            
            // RNG disabled test
            RNGDisabledTest rng;
            rng.k = 30;
            conf.rng_disabled_test = rng;
            
            // Exploration test
            ExploreTest et;
            et.k_explore = 1000;
            et.query_count = 1000;
            conf.explore_test = et;
            
            // Statistics test
            StatsTest st_test;
            st_test.compute_graph_quality = true;
            st_test.compute_seed_reachability = true;
            st_test.compute_avg_reach = false;  // Very expensive, disable by default
            conf.stats_test = st_test;
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.eps_parameter = { 0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f };
        
        CreateGraphTest cg;
        cg.k = 30;
        cg.k_ext = 60;
        conf.create_graph = cg;
        
        OptimizeGraphTest og;
        og.k_opt = 30;
        conf.optimize_graph = og;
        
        TestGraphTest tg;
        conf.test_graph = tg;
        
        ScalingTest st;
        st.k_values = {30, 60, 90};
        conf.scaling_test = st;
        
        LIDComparisonTest lc;
        lc.k = 30;
        conf.lid_comparison_test = lc;
    } else if (dataset_name == DatasetName::AUDIO) {
        conf.eps_parameter = { 0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f };
        
        CreateGraphTest cg;
        cg.k = 20;
        cg.k_ext = 40;
        conf.create_graph = cg;
        
        OptimizeGraphTest og;
        og.k_opt = 20;
        conf.optimize_graph = og;
        
        TestGraphTest tg;
        tg.repeat = 50;  // More repeats for small dataset
        conf.test_graph = tg;
        
        ScalingTest st;
        st.k_values = {10, 20, 40};
        st.repeat = 50;
        conf.scaling_test = st;
        
        LIDComparisonTest lc;
        lc.k = 20;
        conf.lid_comparison_test = lc;
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
    // test_type: create_graph | optimize_graph | reduce_graph | test_graph | scaling | k_sweep | lid_compare | 
    //            all_schemes | k_opt_sweep | eps_opt_sweep | explore | stats | all
    DatasetName ds_name = DatasetName::SIFT1M;
    std::string test_type_arg = "all";
    bool do_run = false;
    
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "help" || arg == "--help") {
            log("Usage: deglib_phd <dataset> [test_type] [--run]\n");
            log("Datasets: sift1m, deep1m, audio, glove\n");
            log("Test types:\n");
            log("  create_graph    - Build graph with default settings\n");
            log("  optimize_graph  - Optimize existing graph\n");
            log("  reduce_graph    - Reduce graph degree\n");
            log("  test_graph      - Benchmark search on graph\n");
            log("  scaling         - Test different k values\n");
            log("  k_sweep         - Sweep k values with fixed LID\n");
            log("  lid_compare     - Compare LowLID vs HighLID\n");
            log("  all_schemes     - Test all 5 OptimizationTargets\n");
            log("  k_opt_sweep     - Sweep k_opt values for optimization\n");
            log("  eps_opt_sweep   - Sweep eps_opt values for optimization\n");
            log("  explore         - Run exploration benchmark (top-k from entry vertex)\n");
            log("  stats           - Compute comprehensive graph statistics\n");
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
                arg == "test_graph" || arg == "scaling" || arg == "k_sweep" || arg == "lid_compare" ||
                arg == "all_schemes" || arg == "k_opt_sweep" || arg == "eps_opt_sweep" || arg == "explore" || 
                arg == "stats" || arg == "all") {
            test_type_arg = arg;
        }
    }

    // Create Dataset object (combines name with data_path)
    Dataset ds(ds_name, data_path);
    
    // Get dataset config and create path utilities
    auto config = get_dataset_config(ds_name);
    GraphPaths graph_paths(data_path, ds);  // GraphPaths for graph files

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
    if(config.create_graph) log(" - create_graph (k={}, k_ext={}, eps_ext={:.2f}, lid={})\n", 
        config.create_graph->k, config.create_graph->k_ext, config.create_graph->eps_ext, to_str_lid(config.create_graph->lid));
    if(config.optimize_graph) log(" - optimize_graph (k_opt={}, eps_opt={:.4f}, i_opt={})\n",
        config.optimize_graph->k_opt, config.optimize_graph->eps_opt, config.optimize_graph->i_opt);
    if(config.reduce_graph) log(" - reduce_graph (target_k={})\n", config.reduce_graph->target_k);
    if(config.test_graph) log(" - test_graph (repeat={}, test_threads={}, k_test={})\n",
        config.test_graph->repeat, config.test_graph->test_threads, config.test_graph->k_test);
    if(config.scaling_test) {
        log(" - scaling (k_values=[{}])\n", fmt::join(config.scaling_test->k_values, ", "));
    }
    if(config.k_sweep_test) {
        log(" - k_sweep (k_values=[{}])\n", fmt::join(config.k_sweep_test->k_values, ", "));
    }
    if(config.lid_comparison_test) log(" - lid_compare (k={})\n", config.lid_comparison_test->k);
    
    // New test types
    if(config.all_schemes_test) log(" - all_schemes (k={}, k_ext={}, tests 5 OptimizationTargets)\n",
        config.all_schemes_test->k, config.all_schemes_test->k_ext);
    if(config.k_opt_sweep_test) log(" - k_opt_sweep (k_opt_values=[{}])\n", 
        fmt::join(config.k_opt_sweep_test->k_opt_values, ", "));
    if(config.eps_opt_sweep_test) log(" - eps_opt_sweep (eps_opt_values=[{}])\n",
        fmt::join(config.eps_opt_sweep_test->eps_opt_values, ", "));
    if(config.iterative_opt_test) log(" - iterative_opt (total_it={}, save_interval={})\n",
        config.iterative_opt_test->total_iterations, config.iterative_opt_test->save_interval);
    if(config.rng_disabled_test) log(" - rng_disabled (k={}, rng=off)\n", config.rng_disabled_test->k);
    if(config.explore_test) log(" - explore (k={}, query_count={})\n",
        config.explore_test->k_explore, config.explore_test->query_count);
    if(config.stats_test) log(" - stats (GQ={}, reachability={}, avg_reach={})\n",
        config.stats_test->compute_graph_quality, config.stats_test->compute_seed_reachability, 
        config.stats_test->compute_avg_reach);

    // Load data once if we're running tests
    std::unique_ptr<deglib::StaticFeatureRepository> base_repository;
    std::unique_ptr<deglib::StaticFeatureRepository> query_repository;
    std::vector<std::unordered_set<uint32_t>> ground_truth;
    uint32_t dims = 0;
    
    if(do_run) {
        log("\nLoading data...\n");
        
        // Ensure graph directory exists
        deglib::benchmark::ensure_directory(graph_paths.graph_directory());
        
        // Load repositories using Dataset convenience methods
        base_repository = std::make_unique<deglib::StaticFeatureRepository>(ds.load_base());
        query_repository = std::make_unique<deglib::StaticFeatureRepository>(ds.load_query());
        dims = base_repository->dims();
        
        // Get ground truth file based on data stream type
        bool use_half = (config.data_stream_type != DataStreamType::AddAll);
        std::string gt_file = use_half ? ds.groundtruth_file_half() : ds.groundtruth_file_full();
        
        // Check if ground truth file exists, create if needed
        if (!std::filesystem::exists(gt_file)) {
            log("Ground truth file not found, computing...\n");
            const uint32_t k_gt = 100; // Standard k for ground truth
            auto gt_vec = deglib::benchmark::compute_knn_groundtruth(*base_repository, *query_repository, config.metric, k_gt, 0, omp_get_max_threads());
            deglib::benchmark::ivecs_write(gt_file.c_str(), k_gt, query_repository->size(), gt_vec.data());
            log("Ground truth saved to: {}\n", gt_file);
        }
        
        // Load ground truth as unordered_sets
        const uint32_t k_gt = 100;
        ground_truth = ds.load_groundtruth(k_gt, use_half);
        
        log("Loaded {} features with {} dimensions\n", base_repository->size(), dims);
        log("Loaded {} queries\n", query_repository->size());
        log("Loaded {} ground truth entries\n", ground_truth.size());
    }

    // Execute tests based on test_type_arg
    bool run_all = (test_type_arg == "all");
    
    // CREATE_GRAPH test
    if((run_all || test_type_arg == "create_graph") && config.create_graph) {
        const auto& cg = *config.create_graph;
        std::string test_name = fmt::format("create_graph_K{}_{}",  cg.k, to_str_lid(cg.lid));
        
        log("\n=== CREATE_GRAPH Test ===\n");
        log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
            cg.k, cg.k_ext, cg.eps_ext, to_str_lid(cg.lid), cg.thread_count);
        
        if(do_run && base_repository) {
            std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            
            // Check if graph already exists
            if(std::filesystem::exists(graph_path)) {
                log("Graph already exists, skipping creation: {}\n", graph_path);
            } else {
                // Set up log file for this test (append mode for re-runs)
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== CREATE_GRAPH Test: {} ===\n", test_name);
                log("Settings: k={}, k_ext={}, eps_ext={:.2f}, lid={}, threads={}\n",
                    cg.k, cg.k_ext, cg.eps_ext, to_str_lid(cg.lid), cg.thread_count);
                log("Output graph: {}\n", graph_path);
                // TODO: Implement create_graph logic
                log("create_graph execution not yet implemented\n");
                
                deglib::benchmark::reset_log_to_console();
                log("Log written to: {}\n", log_path);
            }
        }
    }
    
    // OPTIMIZE_GRAPH test
    if((run_all || test_type_arg == "optimize_graph") && config.optimize_graph && config.create_graph) {
        const auto& og = *config.optimize_graph;
        const auto& cg = *config.create_graph;
        std::string test_name = fmt::format("optimize_graph_K{}_OptK{}", cg.k, og.k_opt);
        
        log("\n=== OPTIMIZE_GRAPH Test ===\n");
        log("Settings: k_opt={}, eps_opt={:.4f}, i_opt={}, iterations={}\n",
            og.k_opt, og.eps_opt, og.i_opt, og.iterations);
        
        if(do_run && base_repository) {
            std::string input_graph = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string output_graph = graph_paths.opt_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, 
                cg.lid, og.k_opt, og.eps_opt, og.i_opt, og.iterations);
            
            // Check if optimized graph already exists
            if(std::filesystem::exists(output_graph)) {
                log("Optimized graph already exists, skipping: {}\n", output_graph);
            } else {
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                log("\n=== OPTIMIZE_GRAPH Test: {} ===\n", test_name);
                log("Input graph: {}\n", input_graph);
                log("Output graph: {}\n", output_graph);
                // TODO: Implement optimize_graph logic
                log("optimize_graph execution not yet implemented\n");
                
                deglib::benchmark::reset_log_to_console();
                log("Log written to: {}\n", log_path);
            }
        }
    }
    
    // REDUCE_GRAPH test
    if((run_all || test_type_arg == "reduce_graph") && config.reduce_graph) {
        const auto& rg = *config.reduce_graph;
        std::string test_name = fmt::format("reduce_graph_K{}", rg.target_k);
        
        log("\n=== REDUCE_GRAPH Test ===\n");
        log("Settings: target_k={}\n", rg.target_k);
        
        if(do_run) {
            std::string log_path = graph_paths.log_file(test_name);
            deglib::benchmark::set_log_file(log_path, true);
            log("\n=== REDUCE_GRAPH Test: {} ===\n", test_name);
            // TODO: Implement reduce_graph logic
            log("reduce_graph execution not yet implemented\n");
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    // TEST_GRAPH test
    if((run_all || test_type_arg == "test_graph") && config.test_graph && config.create_graph) {
        const auto& tg = *config.test_graph;
        const auto& cg = *config.create_graph;
        std::string test_name = fmt::format("test_graph_K{}_{}", cg.k, to_str_lid(cg.lid));
        
        log("\n=== TEST_GRAPH Test ===\n");
        log("Settings: repeat={}, test_threads={}, k_test={}\n", tg.repeat, tg.test_threads, tg.k_test);
        
        if(do_run && base_repository && query_repository) {
            std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            
            // Always append to log file for benchmarks (can re-run benchmarks multiple times)
            std::string log_path = graph_paths.log_file(test_name);
            deglib::benchmark::set_log_file(log_path, true);
            
            log("\n=== TEST_GRAPH Test: {} ===\n", test_name);
            log("Graph: {}\n", graph_path);
            
            if(std::filesystem::exists(graph_path)) {
                const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                log("Graph loaded: {} nodes\n", graph.size());
                
                // Log comprehensive graph statistics
                auto stats = deglib::benchmark::collect_graph_stats(graph);
                deglib::benchmark::log_graph_stats(stats);
                
                deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, tg.repeat, tg.test_threads, tg.k_test, config.eps_parameter);
            } else {
                log("Graph file not found: {}\n", graph_path);
            }
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    // SCALING test (tests multiple k values)
    if((run_all || test_type_arg == "scaling") && config.scaling_test) {
        const auto& st = *config.scaling_test;
        
        log("\n=== SCALING Test ===\n");
        log("Testing k values: [{}]\n", fmt::join(st.k_values, ", "));
        
        if(do_run && base_repository && query_repository) {
            for(uint8_t k : st.k_values) {
                uint8_t k_ext = k * 2;
                std::string test_name = fmt::format("scaling_K{}_{}", k, to_str_lid(st.lid));
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                
                std::string graph_path = graph_paths.graph_file(dims, config.metric, k, k_ext, st.eps_ext, st.lid);
                log("\n=== SCALING Test: k={} ===\n", k);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} nodes\n", graph.size());
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, st.repeat, st.test_threads, st.k_test, config.eps_parameter);
                } else {
                    log("Graph file not found, skipping.\n");
                }
                
                deglib::benchmark::reset_log_to_console();
                log("k={}: Log written to: {}\n", k, log_path);
            }
        }
    }
    
    // K_SWEEP test
    if((run_all || test_type_arg == "k_sweep") && config.k_sweep_test) {
        const auto& ks = *config.k_sweep_test;
        
        log("\n=== K_SWEEP Test ===\n");
        log("Testing k values: [{}] with LID={}\n", fmt::join(ks.k_values, ", "), to_str_lid(ks.lid));
        
        if(do_run && base_repository && query_repository) {
            for(uint8_t k : ks.k_values) {
                uint8_t k_ext = k * 2;
                std::string test_name = fmt::format("k_sweep_K{}_{}", k, to_str_lid(ks.lid));
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                
                std::string graph_path = graph_paths.graph_file(dims, config.metric, k, k_ext, ks.eps_ext, ks.lid);
                log("\n=== K_SWEEP Test: k={} ===\n", k);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} nodes\n", graph.size());
                    uint32_t repeat = config.test_graph ? config.test_graph->repeat : 1;
                    uint32_t test_threads = config.test_graph ? config.test_graph->test_threads : 1;
                    uint32_t k_test = config.test_graph ? config.test_graph->k_test : 100;
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, repeat, test_threads, k_test, config.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("k={}: Log written to: {}\n", k, log_path);
            }
        }
    }
    
    // LID_COMPARE test (tests both LowLID and HighLID)
    if((run_all || test_type_arg == "lid_compare") && config.lid_comparison_test) {
        const auto& lc = *config.lid_comparison_test;
        uint8_t k_ext = lc.k * 2;
        
        log("\n=== LID_COMPARE Test ===\n");
        log("Comparing LowLID vs HighLID with k={}\n", lc.k);
        
        if(do_run && base_repository && query_repository) {
            for(auto lid : {deglib::builder::OptimizationTarget::LowLID, deglib::builder::OptimizationTarget::HighLID}) {
                std::string test_name = fmt::format("lid_compare_K{}_{}", lc.k, to_str_lid(lid));
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                
                std::string graph_path = graph_paths.graph_file(dims, config.metric, lc.k, k_ext, lc.eps_ext, lid);
                log("\n=== LID_COMPARE Test: {} ===\n", to_str_lid(lid));
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    log("Graph loaded: {} nodes\n", graph.size());
                    auto stats = deglib::benchmark::collect_graph_stats(graph);
                    deglib::benchmark::log_graph_stats(stats);
                    uint32_t repeat = config.test_graph ? config.test_graph->repeat : 1;
                    uint32_t test_threads = config.test_graph ? config.test_graph->test_threads : 1;
                    uint32_t k_test = config.test_graph ? config.test_graph->k_test : 100;
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, repeat, test_threads, k_test, config.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("{}: Log written to: {}\n", to_str_lid(lid), log_path);
            }
        }
    }
    
    // ALL_SCHEMES test (tests all 5 OptimizationTargets with same k)
    if((run_all || test_type_arg == "all_schemes") && config.all_schemes_test) {
        const auto& as = *config.all_schemes_test;
        
        log("\n=== ALL_SCHEMES Test ===\n");
        log("Testing all 5 OptimizationTargets with k={}\n", as.k);
        
        if(do_run && base_repository && query_repository) {
            std::vector<deglib::builder::OptimizationTarget> all_targets = {
                deglib::builder::OptimizationTarget::LowLID,
                deglib::builder::OptimizationTarget::HighLID,
                deglib::builder::OptimizationTarget::StreamingData,
                deglib::builder::OptimizationTarget::SchemeA,
                deglib::builder::OptimizationTarget::SchemeB
            };
            
            for(auto lid : all_targets) {
                std::string test_name = fmt::format("all_schemes_K{}_{}", as.k, to_str_lid(lid));
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                
                std::string graph_path = graph_paths.graph_file(dims, config.metric, as.k, as.k_ext, as.eps_ext, lid);
                log("\n=== ALL_SCHEMES Test: {} ===\n", to_str_lid(lid));
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    auto stats = deglib::benchmark::collect_graph_stats(graph);
                    deglib::benchmark::log_graph_stats(stats);
                    uint32_t repeat = config.test_graph ? config.test_graph->repeat : 1;
                    uint32_t test_threads = config.test_graph ? config.test_graph->test_threads : 1;
                    uint32_t k_test = config.test_graph ? config.test_graph->k_test : 100;
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, repeat, test_threads, k_test, config.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("{}: Log written to: {}\n", to_str_lid(lid), log_path);
            }
        }
    }
    
    // K_OPT_SWEEP test (sweep k_opt values for optimization)
    if((run_all || test_type_arg == "k_opt_sweep") && config.k_opt_sweep_test && config.create_graph) {
        const auto& kos = *config.k_opt_sweep_test;
        const auto& cg = *config.create_graph;
        
        log("\n=== K_OPT_SWEEP Test ===\n");
        log("Testing k_opt values: [{}]\n", fmt::join(kos.k_opt_values, ", "));
        
        if(do_run && base_repository && query_repository) {
            for(uint8_t k_opt : kos.k_opt_values) {
                std::string test_name = fmt::format("k_opt_sweep_K{}_OptK{}", cg.k, k_opt);
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                
                std::string graph_path = graph_paths.opt_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, 
                    cg.lid, k_opt, kos.eps_opt, kos.i_opt, kos.iterations);
                log("\n=== K_OPT_SWEEP Test: k_opt={} ===\n", k_opt);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    auto stats = deglib::benchmark::collect_graph_stats(graph);
                    deglib::benchmark::log_graph_stats(stats);
                    uint32_t repeat = config.test_graph ? config.test_graph->repeat : 1;
                    uint32_t test_threads = config.test_graph ? config.test_graph->test_threads : 1;
                    uint32_t k_test = config.test_graph ? config.test_graph->k_test : 100;
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, repeat, test_threads, k_test, config.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("k_opt={}: Log written to: {}\n", k_opt, log_path);
            }
        }
    }
    
    // EPS_OPT_SWEEP test (sweep eps_opt values for optimization)
    if((run_all || test_type_arg == "eps_opt_sweep") && config.eps_opt_sweep_test && config.create_graph) {
        const auto& eos = *config.eps_opt_sweep_test;
        const auto& cg = *config.create_graph;
        
        log("\n=== EPS_OPT_SWEEP Test ===\n");
        log("Testing eps_opt values: [{}]\n", fmt::join(eos.eps_opt_values, ", "));
        
        if(do_run && base_repository && query_repository) {
            for(float eps_opt : eos.eps_opt_values) {
                std::string test_name = fmt::format("eps_opt_sweep_K{}_Eps{:.4f}", cg.k, eps_opt);
                std::string log_path = graph_paths.log_file(test_name);
                deglib::benchmark::set_log_file(log_path, true);
                
                std::string graph_path = graph_paths.opt_graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, 
                    cg.lid, eos.k_opt, eps_opt, eos.i_opt, eos.iterations);
                log("\n=== EPS_OPT_SWEEP Test: eps_opt={:.4f} ===\n", eps_opt);
                log("Graph: {}\n", graph_path);
                
                if(std::filesystem::exists(graph_path)) {
                    const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                    auto stats = deglib::benchmark::collect_graph_stats(graph);
                    deglib::benchmark::log_graph_stats(stats);
                    uint32_t repeat = config.test_graph ? config.test_graph->repeat : 1;
                    uint32_t test_threads = config.test_graph ? config.test_graph->test_threads : 1;
                    uint32_t k_test = config.test_graph ? config.test_graph->k_test : 100;
                    deglib::benchmark::test_graph_anns(graph, *query_repository, ground_truth, repeat, test_threads, k_test, config.eps_parameter);
                } else {
                    log("Graph file not found: {}\n", graph_path);
                }
                
                deglib::benchmark::reset_log_to_console();
                log("eps_opt={:.4f}: Log written to: {}\n", eps_opt, log_path);
            }
        }
    }
    
    // EXPLORE test (exploration benchmark)
    if((run_all || test_type_arg == "explore") && config.explore_test && config.create_graph) {
        const auto& et = *config.explore_test;
        const auto& cg = *config.create_graph;
        
        log("\n=== EXPLORE Test ===\n");
        log("Settings: k_explore={}, query_count={}, include_entry={}\n", et.k_explore, et.query_count, et.include_entry);
        
        if(do_run && base_repository) {
            std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string test_name = fmt::format("explore_K{}_top{}", cg.k, et.k_explore);
            std::string log_path = graph_paths.log_file(test_name);
            deglib::benchmark::set_log_file(log_path, true);
            
            log("\n=== EXPLORE Test: {} ===\n", test_name);
            log("Graph: {}\n", graph_path);
            
            if(std::filesystem::exists(graph_path)) {
                const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                auto stats = deglib::benchmark::collect_graph_stats(graph);
                deglib::benchmark::log_graph_stats(stats);
                
                // Load entry vertex labels (should exist from setup_dataset)
                std::string entry_labels_file = ds.explore_entry_vertex_file();
                std::unique_ptr<std::byte[]> entry_labels_data;
                size_t entry_dims_out = 0;
                size_t entry_count_out = 0;
                
                if(std::filesystem::exists(entry_labels_file)) {
                    log("Loading entry vertex labels: {}\n", entry_labels_file);
                    entry_labels_data = deglib::fvecs_read(entry_labels_file.c_str(), entry_dims_out, entry_count_out);
                } else {
                    log("Error: Entry vertex labels file not found: {}\n", entry_labels_file);
                    log("Please run setup_dataset first to generate all required files.\n");
                    deglib::benchmark::reset_log_to_console();
                    return 1;
                }
                
                // Load exploration ground truth (should exist from setup_dataset)
                std::string explore_gt_file = ds.explore_groundtruth_file();
                std::unique_ptr<std::byte[]> explore_gt_data;
                size_t explore_gt_dims_out = 0;
                size_t explore_gt_count_out = 0;
                
                if(std::filesystem::exists(explore_gt_file)) {
                    log("Loading exploration ground truth: {}\n", explore_gt_file);
                    explore_gt_data = deglib::fvecs_read(explore_gt_file.c_str(), explore_gt_dims_out, explore_gt_count_out);
                } else {
                    log("Error: Exploration ground truth file not found: {}\n", explore_gt_file);
                    log("Please run setup_dataset first to generate all required files.\n");
                    deglib::benchmark::reset_log_to_console();
                    return 1;
                }
                
                const auto explore_gt_ptr = (uint32_t*)explore_gt_data.get();
                const auto entry_labels_ptr = (uint32_t*)entry_labels_data.get();
                
                deglib::benchmark::test_graph_explore(graph, (uint32_t)entry_count_out, explore_gt_ptr, (uint32_t)explore_gt_dims_out, 
                    entry_labels_ptr, (uint32_t)entry_dims_out, et.include_entry, et.repeat, et.k_explore, et.threads);
            } else {
                log("Graph file not found: {}\n", graph_path);
            }
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    // STATS test (comprehensive graph statistics)
    if((run_all || test_type_arg == "stats") && config.stats_test && config.create_graph) {
        const auto& st = *config.stats_test;
        const auto& cg = *config.create_graph;
        
        log("\n=== STATS Test ===\n");
        log("Settings: GQ={}, seed_reachability={}, avg_reach={}, connectivity={}, regularity={}\n",
            st.compute_graph_quality, st.compute_seed_reachability, st.compute_avg_reach,
            st.check_connectivity, st.check_regularity);
        
        if(do_run && base_repository) {
            std::string graph_path = graph_paths.graph_file(dims, config.metric, cg.k, cg.k_ext, cg.eps_ext, cg.lid);
            std::string test_name = fmt::format("stats_K{}_{}", cg.k, to_str_lid(cg.lid));
            std::string log_path = graph_paths.log_file(test_name);
            deglib::benchmark::set_log_file(log_path, true);
            
            log("\n=== STATS Test: {} ===\n", test_name);
            log("Graph: {}\n", graph_path);
            
            if(std::filesystem::exists(graph_path)) {
                const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
                
                // Collect full statistics
                auto stats = deglib::benchmark::collect_graph_stats(graph);
                
                // Check graph regularity
                if (st.check_regularity) {
                    log("\nChecking graph regularity...\n");
                    bool regular = deglib::analysis::check_graph_regularity(graph, (uint32_t)graph.size(), true);
                    log("Graph regularity: {}\n", regular ? "PASS" : "FAIL");
                }
                
                // Check graph connectivity
                if (st.check_connectivity) {
                    log("\nChecking graph connectivity...\n");
                    bool connected = deglib::analysis::check_graph_connectivity(graph);
                    log("Graph connectivity: {}\n", connected ? "CONNECTED" : "DISCONNECTED");
                    
                    if (!connected) {
                        uint32_t components = deglib::analysis::count_graph_components(graph);
                        log("Number of components: {}\n", components);
                    }
                }
                
                // Compute graph quality (needs ground truth top-list)
                if (st.compute_graph_quality && !ground_truth.empty()) {
                    log("\nComputing graph quality...\n");
                    stats.graph_quality = deglib::benchmark::compute_graph_quality(graph, ground_truth);
                }
                
                // Compute seed reachability (expensive)
                if (st.compute_seed_reachability) {
                    log("\nComputing seed reachability...\n");
                    stats.search_reachability = deglib::benchmark::compute_search_reachability(graph);
                }
                
                // Compute average reach (very expensive)
                if (st.compute_avg_reach) {
                    log("\nComputing average reach...\n");
                    stats.exploration_reachability = deglib::benchmark::compute_exploration_reach(graph);
                }
                
                // Log all statistics
                log("\n");
                deglib::benchmark::log_graph_stats(stats);
                
            } else {
                log("Graph file not found: {}\n", graph_path);
            }
            
            deglib::benchmark::reset_log_to_console();
            log("Log written to: {}\n", log_path);
        }
    }
    
    log("\nTest OK\n");
    return 0;
}
