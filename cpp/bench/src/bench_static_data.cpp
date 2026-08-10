#include <fmt/core.h>
#include <fmt/format.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "benchmark.h"
#include "build.h"
#include "dataset.h"
#include <deglib/deglib.h>
#include "file_io.h"
#include "logging.h"
#include "stats.h"
#include "stopwatch.h"

using namespace deglib::benchmark;

struct Config {
    uint8_t k = 30;
    uint8_t k_ext = 60;
    float eps_ext = 0.1f;
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::LowLID;
    uint32_t build_threads = std::thread::hardware_concurrency() / 2;
    uint32_t anns_k = 100;
    uint32_t anns_repeat = 1;
    uint32_t anns_threads = 1;
    uint32_t explore_k = 1000;
    uint32_t explore_repeat = 1;
    uint32_t explore_threads = 1;
    std::vector<float> eps_parameter = {0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};
};

static Config get_dataset_config(const DatasetName& dataset_name) {
    Config conf{};
    if (dataset_name == DatasetName::AUDIO) {
        conf.k = 20;
        conf.k_ext = 40;
        conf.anns_repeat = 50;
        conf.eps_parameter = {0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.2f, 1.6f, 2.0f};
    } else if (dataset_name == DatasetName::ENRON) {
        conf.k = 30;
        conf.k_ext = 60;
        conf.anns_repeat = 50;
        conf.eps_parameter = {0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.2f, 1.6f, 2.0f};
    } else if (dataset_name == DatasetName::GLOVE) {
        conf.lid = deglib::builder::OptimizationTarget::HighLID;
        conf.eps_parameter = {0.001f, 0.05f, 0.10f, 0.125f, 0.15f, 0.175f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.eps_parameter = {0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};
    } else if (dataset_name == DatasetName::SIFT1M) {
        // Uses default parameters (k=30, k_ext=60, LowLID)
    }
    return conf;
}

static std::string build_graph_filename(const Dataset& ds, const Config& cg, uint32_t dims) {
    std::string metric_str = ds.info().metric.to_string();
    std::string lid_str;
    switch (cg.lid) {
        case deglib::builder::OptimizationTarget::HighLID: lid_str = "HighLID"; break;
        case deglib::builder::OptimizationTarget::LowLID: lid_str = "LowLID"; break;
        case deglib::builder::OptimizationTarget::StreamingData: lid_str = "StreamingData"; break;
        default: lid_str = "UnknownLID"; break;
    }
    return (ds.dataset_dir() / "deg" / fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}.deg", dims, metric_str, cg.k, cg.k_ext, cg.eps_ext, lid_str)).string();
}

void print_help(const char* program_name) {
    log("=== DEG Benchmark Tool: bench_static_data ===\n\n");
    log("Description:\n");
    log("  Builds a DEG graph for static dataset(s) using EvenRegularGraphBuilder (AddAll mode),\n");
    log("  computes graph statistics (in/out degree, reachability, quality), and evaluates\n");
    log("  ANNS recall vs QPS performance as well as exploration search.\n\n");
    log("Usage:\n");
    log("  {} [dataset] [options]\n\n", program_name);
    log("Datasets:\n");
    log("  sift1m   - SIFT1M (1M vectors, 128D, default)\n");
    log("  deep1m   - DEEP1M (1M vectors, 96D)\n");
    log("  glove    - GloVe (1.18M vectors, 100D, HighLID optimization)\n");
    log("  audio    - Audio (53.3k vectors, 192D, K=20)\n");
    log("  enron    - Enron (94.9k vectors, 1369D, K=30)\n");
    log("  all      - Execute benchmarks for all datasets sequentially\n\n");
    log("Options:\n");
    log("  --force-rebuild       Force rebuilding the graph even if the graph file already exists\n");
    log("  --instruction <inst>  Select distance instruction set (auto, avx512, avx2, scalar)\n");
    log("  --threads <count>     Number of threads used for building the graph (default: hardware_concurrency / 2)\n");
    log("  --help, -h            Display this detailed help message\n\n");
    log("Data Path:\n");
    log("  Data is loaded from / saved to DATA_PATH defined at build time (e.g. \"{}\")\n", DATA_PATH);
}

static deglib::cpu::InstructionSet parse_instruction_set(const std::string& str) {
    if (str == "auto" || str == "Auto") return deglib::cpu::InstructionSet::Auto;
    if (str == "scalar" || str == "Scalar") return deglib::cpu::InstructionSet::Scalar;
    if (str == "avx2" || str == "AVX2") return deglib::cpu::InstructionSet::AVX2;
    if (str == "avx512" || str == "AVX512") return deglib::cpu::InstructionSet::AVX512;
    log("Unknown instruction set '{}', falling back to Auto\n", str);
    return deglib::cpu::InstructionSet::Auto;
}

void run_static_benchmark(const DatasetName& ds_name, const std::filesystem::path& data_path, bool force_rebuild, deglib::cpu::InstructionSet instruction, uint32_t build_threads) {
    Dataset ds(ds_name, data_path);
    auto config = get_dataset_config(ds_name);
    config.build_threads = build_threads;

    log("\n=== Benchmarking Static Graph Creation & Evaluation for Dataset: {} ===\n", ds.name());

    auto setup_threads = std::thread::hardware_concurrency() / 2;
    if (!setup_dataset(ds, setup_threads, /*include_half=*/false, /*include_exploration=*/true)) {
        log("ERROR: Failed to setup dataset {}\n", ds.name());
        return;
    }

    auto base_repository = ds.load_base();
    auto query_repository = ds.load_query();
    const uint32_t dims = (uint32_t)base_repository.dims();

    std::string graph_path = build_graph_filename(ds, config, dims);
    ensure_directory(std::filesystem::path(graph_path).parent_path());

    std::string log_path = graph_path + ".log";
    set_log_file(log_path, false);
    log("Logging to file: {}\n", log_path);

    log("\n--- Computing Linear Search Baseline ---\n");
    uint64_t linear_baseline_us = compute_linear_search_baseline(base_repository, ds.info().metric, 100, instruction) * 2;

    if (!std::filesystem::exists(graph_path) || force_rebuild) {
        log("\n=== Building Graph ===\n");
        log("Output graph: {}\n", graph_path);
        create_graph(base_repository,
                     DataStreamType::AddAll,
                     graph_path,
                     ds.info().metric,
                     config.lid,
                     config.k,
                     config.k_ext,
                     config.eps_ext,
                     0, 0, 0,
                     config.build_threads,
                     true,
                     ds.info().scale,
                     false,
                     instruction);
    } else {
        log("Graph already exists: {}\n", graph_path);
    }

    if (std::filesystem::exists(graph_path)) {
        const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
        log("Graph loaded: {} vertices\n", graph.size());

        log("\n--- Graph Analysis ---\n");
        {
            analyze_graph(graph);
        }

        log("\n--- ANNS Test (k={}) ---\n", config.anns_k);
        {
            auto ground_truth = ds.load_groundtruth(config.anns_k, false);
            test_graph_anns(graph,
                            query_repository,
                            ground_truth,
                            config.anns_repeat,
                            config.anns_threads,
                            config.anns_k,
                            config.eps_parameter,
                            nullptr,
                            linear_baseline_us);
        }

        log("\n--- Exploration Test (k={}) ---\n", config.explore_k);
        {
            auto entry_vertices = ds.load_explore_entry_vertices();
            auto explore_gt = ds.load_explore_groundtruth(config.explore_k);
            test_graph_explore(graph,
                               entry_vertices,
                               explore_gt,
                               true,
                               config.explore_repeat,
                               config.explore_k,
                               config.explore_threads,
                               nullptr,
                               ds.info().explore_depth,
                               linear_baseline_us);
        }
    } else {
        log("ERROR: Graph file not found: {}\n", graph_path);
    }
}

int main(int argc, char* argv[]) {
    const auto data_path = std::filesystem::path(DATA_PATH);
    DatasetName ds_name = DatasetName::DEEP1M;
    bool force_rebuild = false;
    deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::AVX2;
    uint32_t build_threads = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--force-rebuild") {
            force_rebuild = true;
        } else if (arg == "--instruction" && i + 1 < argc) {
            instruction = parse_instruction_set(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            build_threads = (uint32_t)std::stoul(argv[++i]);
        } else {
            auto parsed = DatasetName::from_string(arg);
            if (parsed.is_valid()) {
                ds_name = parsed;
            } else {
                log("Unknown argument or dataset: {}\n\n", arg);
                print_help(argv[0]);
                return 1;
            }
        }
    }

    log("=== bench_static_data ===\n");

    std::vector<DatasetName> datasets_to_run;
    if (ds_name == DatasetName::ALL) {
        for (const auto& d : DatasetName::all()) datasets_to_run.push_back(d);
    } else {
        datasets_to_run.push_back(ds_name);
    }

    for (const auto& current_ds : datasets_to_run) {
        run_static_benchmark(current_ds, data_path, force_rebuild, instruction, build_threads);
    }

    log("\nStatic Benchmark Finished Successfully.\n");
    return 0;
}
