#include "benchmark.h"
#include "build.h"
#include "dataset.h"
#include "file_io.h"
#include "logging.h"
#include "stats.h"
#include "stopwatch.h"

#include <deglib/deglib.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace deglib::benchmark;

struct Config {
    uint8_t k = 30;
    uint8_t k_opt = 30;
    float eps_opt = 0.001f;
    uint8_t max_path_length = 5;
    uint64_t log_after = 100000;
    uint64_t max_iterations = 1000000;
    uint32_t max_distance_count_test = 2000;
    uint32_t k_test = 100;
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
        conf.eps_parameter = {0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.2f, 1.6f, 2.0f};
        conf.max_iterations = 300000;
    } else if (dataset_name == DatasetName::ENRON) {
        conf.eps_parameter = {0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.2f, 1.6f, 2.0f};
    } else if (dataset_name == DatasetName::GLOVE) {
        conf.eps_parameter = {0.001f, 0.05f, 0.10f, 0.125f, 0.15f, 0.175f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};
    } else if (dataset_name == DatasetName::DEEP1M) {
        conf.eps_parameter = {0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f, 1.5f, 2.0f};
    } else if (dataset_name == DatasetName::SIFT1M) {
        // Uses default parameters (k=30, LowLID)
    }
    return conf;
}

void print_help(const char* program_name) {
    log("=== DEG Benchmark Tool: bench_edge_optimization ===\n\n");
    log("Description:\n");
    log("  Fully in-memory edge-optimization benchmark. Builds a random even-regular graph in RAM,\n");
    log("  optimizes its edges with EvenRegularGraphBuilder (LowLID target), and evaluates the\n");
    log("  result (graph statistics, ANNS recall vs QPS, exploration search). No graph files are\n");
    log("  written to disk; output goes to the console only.\n\n");
    log("Usage:\n");
    log("  {} [dataset] [options]\n\n", program_name);
    log("Datasets:\n");
    log("  sift1m   - SIFT1M (1M vectors, 128D, default)\n");
    log("  deep1m   - DEEP1M (1M vectors, 96D)\n");
    log("  glove    - GloVe (1.18M vectors, 100D)\n");
    log("  audio    - Audio (53.3k vectors, 192D, K=20)\n");
    log("  enron    - Enron (94.9k vectors, 1369D, K=30)\n");
    log("  all      - Execute benchmarks for all datasets sequentially\n\n");
    log("Options:\n");
    log("  --data-path, -d <path> Path to datasets directory (default: env DEG_DATA_PATH or ./data)\n");
    log("  --instruction <inst>  Select distance instruction set (auto, avx512, avx2, scalar, default: auto)\n");
    log("  --k-opt <uint>        Number of neighbors considered for edge improvement (default: 30)\n");
    log("  --eps-opt <float>     Epsilon for neighbor search during edge improvement (default: 0.001)\n");
    log("  --log-after <uint>    Log optimization progress every N improvement tries (default: 100000)\n");
    log("  --iterations <uint>   Stop optimization after N improvement tries (default: 1000000)\n");
    log("  --help, -h            Display this detailed help message\n\n");
}

static deglib::cpu::InstructionSet parse_instruction_set(const std::string& str) {
    if (str == "auto" || str == "Auto") return deglib::cpu::InstructionSet::Auto;
    if (str == "scalar" || str == "Scalar") return deglib::cpu::InstructionSet::Scalar;
    if (str == "avx2" || str == "AVX2") return deglib::cpu::InstructionSet::AVX2;
    if (str == "avx512" || str == "AVX512") return deglib::cpu::InstructionSet::AVX512;
    log("Unknown instruction set '{}', falling back to Auto\n", str);
    return deglib::cpu::InstructionSet::Auto;
}

void run_edge_optimization_benchmark(
    const DatasetName& ds_name,
    const std::filesystem::path& data_path,
    const Config& config,
    deglib::cpu::InstructionSet instruction
) {
    Dataset ds(ds_name, data_path);

    log("\n=== Edge Optimization Benchmark for Dataset: {} ===\n", ds.name());

    auto setup_threads = std::thread::hardware_concurrency() / 2;
    if (!setup_dataset(ds, setup_threads, /*include_half=*/false, /*include_exploration=*/true)) {
        log("ERROR: Failed to setup dataset {}\n", ds.name());
        return;
    }

    auto base_repository = ds.load_base();
    auto query_repository = ds.load_query();
    auto ground_truth = ds.load_groundtruth(config.anns_k, false);

    log("\n--- Computing Linear Search Baseline ---\n");
    uint64_t linear_baseline_us = compute_linear_search_baseline(base_repository, ds.info().metric, 100, instruction) * 2;

    log("\n=== Building Initial Random Graph (in memory) ===\n");
    auto graph = create_random_graph(base_repository, ds.info().metric, config.k, 0, ds.info().scale, instruction);
    auto initial_recall = estimate_recall(graph, query_repository, ground_truth, config.max_distance_count_test, config.k_test);
    auto initial_avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, ds.info().scale);
    log("Initial random graph: AEW {:.2f}, Recall [{}]\n", initial_avg_edge_weight, fmt::join(initial_recall, ", "));

    log("\n=== Optimizing Graph Edges (in memory) ===\n");
    auto rnd = std::mt19937(7);
    auto builder = deglib::builder::EvenRegularGraphBuilder(
        graph, rnd, deglib::builder::OptimizationTarget::LowLID, 0, 0, config.k_opt, config.eps_opt, config.max_path_length, 1, 0
    );

    auto start = std::chrono::steady_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    uint64_t duration_ms = 0;

    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto tries = status.tries;
        const auto improved = status.improved;
        if (tries > 0 && tries % config.log_after == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto duration = duration_ms / 1000;
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, ds.info().scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto diff = tries - last_status.tries;
            auto interval_improvements = improved - last_status.improved;
            auto improv_rate = (diff > 0) ? 100.0f * (float)interval_improvements / (float)diff : 0.0f;

            auto valid = deglib::analysis::check_graph_regularity(graph, (uint32_t)graph.size(), true);
            if (!valid) {
                builder.stop();
                log("Invalid graph, optimization process stopped\n");
            }

            auto recall = estimate_recall(graph, query_repository, ground_truth, config.max_distance_count_test, config.k_test);
            log("{:5}s, with {:8} / {:8} improvements ({:.1f}% rate), AEW {:.2f}, Recall [{}], connected {}\n", duration, improved, tries, improv_rate,
                avg_edge_weight, fmt::join(recall, ", "), connected);

            last_status = status;
            start = std::chrono::steady_clock::now();
        }

        if (tries >= config.max_iterations) {
            builder.stop();
        }
    };

    builder.build(improvement_callback, true);

    log("\n--- Graph Analysis ---\n");
    analyze_graph(graph);

    log("\n--- ANNS Test (k={}) ---\n", config.anns_k);
    test_graph_anns(
        graph, query_repository, ground_truth, config.anns_repeat, config.anns_threads, config.anns_k, config.eps_parameter, nullptr, linear_baseline_us
    );

    log("\n--- Exploration Test (k={}) ---\n", config.explore_k);
    auto entry_vertices = ds.load_explore_entry_vertices();
    auto explore_gt = ds.load_explore_groundtruth(config.explore_k);
    test_graph_explore(
        graph, entry_vertices, explore_gt, true, config.explore_repeat, config.explore_k, config.explore_threads, nullptr, ds.info().explore_depth,
        linear_baseline_us
    );
}

int main(int argc, char* argv[]) {
    auto data_path = deglib::benchmark::get_default_data_path();
    DatasetName ds_name = DatasetName::AUDIO;
    deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto;
    uint8_t k_opt = 30;
    float eps_opt = 0.001f;
    uint64_t log_after = 100000;
    uint64_t max_iterations = 1000000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if ((arg == "--data-path" || arg == "-d") && i + 1 < argc) {
            data_path = std::filesystem::path(argv[++i]);
        } else if (arg == "--instruction" && i + 1 < argc) {
            instruction = parse_instruction_set(argv[++i]);
        } else if (arg == "--k-opt" && i + 1 < argc) {
            k_opt = (uint8_t)std::stoul(argv[++i]);
        } else if (arg == "--eps-opt" && i + 1 < argc) {
            eps_opt = std::stof(argv[++i]);
        } else if (arg == "--log-after" && i + 1 < argc) {
            log_after = std::stoull(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            max_iterations = std::stoull(argv[++i]);
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

    log("=== bench_edge_optimization ===\n");

    std::vector<DatasetName> datasets_to_run;
    if (ds_name == DatasetName::ALL) {
        for (const auto& d : DatasetName::all()) datasets_to_run.push_back(d);
    } else {
        datasets_to_run.push_back(ds_name);
    }

    for (const auto& current_ds : datasets_to_run) {
        auto current_config = get_dataset_config(current_ds);
        current_config.k_opt = k_opt;
        current_config.eps_opt = eps_opt;
        current_config.log_after = log_after;
        current_config.max_iterations = max_iterations;
        run_edge_optimization_benchmark(current_ds, data_path, current_config, instruction);
    }

    log("\nEdge Optimization Benchmark Finished Successfully.\n");
    return 0;
}
