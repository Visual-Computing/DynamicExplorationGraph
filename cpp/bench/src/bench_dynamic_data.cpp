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

struct DynamicConfig {
    uint8_t k = 30;
    uint8_t k_ext = 60;
    float eps_ext = 0.1f;
    deglib::builder::OptimizationTarget lid = deglib::builder::OptimizationTarget::StreamingData;
    uint32_t anns_k = 100;
    uint32_t anns_repeat = 1;
    uint32_t anns_threads = 1;
    std::vector<float> eps_parameter = {0.01f, 0.05f, 0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.8f, 1.0f};
    std::vector<DataStreamType> stream_types = {
        DataStreamType::AddHalf,
        DataStreamType::AddHalfRemoveAndAddOneAtATime,
        DataStreamType::AddAllRemoveHalf
    };
};

static std::string ds_type_str(DataStreamType ds) {
    switch (ds) {
        case DataStreamType::AddAll: return "AddAll";
        case DataStreamType::AddHalf: return "AddHalf";
        case DataStreamType::AddAllRemoveHalf: return "AddAllRemoveHalf";
        case DataStreamType::AddHalfRemoveAndAddOneAtATime: return "AddHalfRemoveAndAddOneAtATime";
        default: return "Unknown";
    }
}

static DynamicConfig get_dataset_config(const DatasetName& dataset_name) {
    DynamicConfig conf{};
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
        conf.eps_parameter = {0.001f, 0.05f, 0.10f, 0.125f, 0.15f, 0.175f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.eps_parameter = {0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};
    } else if (dataset_name == DatasetName::SIFT1M) {
        // Uses default parameters (k=30, k_ext=60, StreamingData target)
    }
    return conf;
}

void print_help(const char* program_name) {
    log("=== DEG Benchmark Tool: bench_dynamic_data ===\n\n");
    log("Description:\n");
    log("  Builds and evaluates DEG graphs under various dynamic data streaming conditions\n");
    log("  (e.g., partial insertions, streaming additions/deletions, interleaved operations)\n");
    log("  using the StreamingData optimization target.\n\n");
    log("Tested Data Stream Patterns:\n");
    log("  1. AddHalf                        - Insert only the first half of base vectors\n");
    log("  2. AddHalfRemoveAndAddOneAtATime - Interleaved insertion and deletion operations\n");
    log("  3. AddAllRemoveHalf               - Insert all vectors, then remove the second half\n\n");
    log("Usage:\n");
    log("  {} [dataset] [options]\n\n", program_name);
    log("Datasets:\n");
    log("  sift1m   - SIFT1M (1M vectors, 128D, default)\n");
    log("  deep1m   - DEEP1M (1M vectors, 96D)\n");
    log("  glove    - GloVe (1.18M vectors, 100D)\n");
    log("  audio    - Audio (53.3k vectors, 192D)\n");
    log("  enron    - Enron (94.9k vectors, 1369D)\n");
    log("  all      - Execute dynamic benchmarks for all datasets sequentially\n\n");
    log("Options:\n");
    log("  --force-rebuild       Force rebuilding the graphs even if graph files already exist\n");
    log("  --instruction <inst>  Select distance instruction set (auto, avx512, avx2, scalar)\n");
    log("  --help, -h            Display this detailed help message\n\n");
    log("Data Path:\n");
    log("  Data is loaded from / saved to DATA_PATH defined at build time (e.g. \"{}\")\n", DATA_PATH);
}

static deglib::distances::InstructionSet parse_instruction_set(const std::string& str) {
    if (str == "auto" || str == "Auto") return deglib::distances::InstructionSet::Auto;
    if (str == "scalar" || str == "Scalar") return deglib::distances::InstructionSet::Scalar;
    if (str == "avx2" || str == "AVX2") return deglib::distances::InstructionSet::AVX2;
    if (str == "avx512" || str == "AVX512") return deglib::distances::InstructionSet::AVX512;
    log("Unknown instruction set '{}', falling back to Auto\n", str);
    return deglib::distances::InstructionSet::Auto;
}

void run_dynamic_benchmark(const DatasetName& ds_name, const std::filesystem::path& data_path, bool force_rebuild, deglib::distances::InstructionSet instruction) {
    Dataset ds(ds_name, data_path);
    auto config = get_dataset_config(ds_name);

    log("\n=== Benchmarking Dynamic Data Streams for Dataset: {} ===\n", ds.name());

    auto setup_threads = std::thread::hardware_concurrency() / 2;
    if (!setup_dataset(ds, setup_threads, /*include_half=*/true, /*include_exploration=*/false)) {
        log("ERROR: Failed to setup dataset {}\n", ds.name());
        return;
    }

    auto base_repository = ds.load_base();
    auto query_repository = ds.load_query();
    const uint32_t dims = (uint32_t)base_repository.dims();

    std::filesystem::path dyn_dir = ds.dataset_dir() / "deg" / "dynamic";
    ensure_directory(dyn_dir);

    log("\n--- Computing Linear Search Baseline ---\n");
    uint64_t linear_baseline_us = compute_linear_search_baseline(base_repository, ds.info().metric, 100, instruction) * 2;

    for (DataStreamType ds_type : config.stream_types) {
        std::string graph_path = (dyn_dir / fmt::format("{}D_K{}_{}.deg", dims, config.k, ds_type_str(ds_type))).string();
        std::string log_path = graph_path + ".log";
        set_log_file(log_path, false);

        log("\n=== Testing DataStreamType: {} ===\n", ds_type_str(ds_type));
        log("Graph path: {}\n", graph_path);
        log("Logging to file: {}\n", log_path);

        if (!std::filesystem::exists(graph_path) || force_rebuild) {
            log("Building dynamic graph...\n");
            create_graph(base_repository,
                         ds_type,
                         graph_path,
                         ds.info().metric,
                         config.lid,
                         config.k,
                         config.k_ext,
                         config.eps_ext,
                         0, 0, 0,
                         1,
                         true,
                         ds.info().scale,
                         false,
                         instruction);
        } else {
            log("Graph already exists: {}\n", graph_path);
        }

        if (std::filesystem::exists(graph_path)) {
            const auto graph = deglib::graph::load_readonly_graph(graph_path.c_str());
            log("Loaded dynamic graph with {} vertices\n", graph.size());

            bool use_half = (ds_type != DataStreamType::AddAll);

            log("\n--- Graph Analysis ---\n");
            {
                analyze_graph(graph);
            }

            log("\n--- ANNS Test (k={}) ---\n", config.anns_k);
            {
                auto ground_truth = ds.load_groundtruth(config.anns_k, use_half);
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
        } else {
            log("ERROR: Graph file not found: {}\n", graph_path);
        }
    }
}

int main(int argc, char* argv[]) {
    const auto data_path = std::filesystem::path(DATA_PATH);
    DatasetName ds_name = DatasetName::AUDIO;
    bool force_rebuild = false;
    deglib::distances::InstructionSet instruction = deglib::distances::InstructionSet::AVX2;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "--force-rebuild") {
            force_rebuild = true;
        } else if (arg == "--instruction" && i + 1 < argc) {
            instruction = parse_instruction_set(argv[++i]);
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

    log("=== bench_dynamic_data ===\n");

    std::vector<DatasetName> datasets_to_run;
    if (ds_name == DatasetName::ALL) {
        for (const auto& d : DatasetName::all()) datasets_to_run.push_back(d);
    } else {
        datasets_to_run.push_back(ds_name);
    }

    for (const auto& current_ds : datasets_to_run) {
        run_dynamic_benchmark(current_ds, data_path, force_rebuild, instruction);
    }

    log("\nDynamic Benchmark Finished Successfully.\n");
    return 0;
}
