#include "benchmark.h"
#include "build.h"
#include "dataset.h"
#include "file_io.h"
#include "logging.h"
#include "stopwatch.h"

#include <deglib/deglib.h>
#include <deglib/optimization.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

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
    float flas_radius_decay = 0.9f;
    std::vector<float> eps_parameter = {0.0f, 0.05f, 0.1f, 0.2f};
};

static Config get_dataset_config(const DatasetName& dataset_name) {
    Config conf{};
    if (dataset_name == DatasetName::AUDIO) {
        conf.k = 20;
        conf.k_ext = 40;
        conf.anns_repeat = 50;
    } else if (dataset_name == DatasetName::ENRON) {
        conf.anns_repeat = 50;
    } else if (dataset_name == DatasetName::GLOVE) {
        conf.lid = deglib::builder::OptimizationTarget::HighLID;
    } else if (dataset_name == DatasetName::DEEP1M || dataset_name == DatasetName::SIFT1M) {
        // Uses default parameters (k=30, k_ext=60, LowLID)
    }
    return conf;
}

static std::string build_graph_filename(const Dataset& ds, const Config& cg, const uint32_t dims, const bool flas_sorted) {
    std::string metric_str = ds.info().metric.to_string();
    std::string lid_str;
    switch (cg.lid) {
        case deglib::builder::OptimizationTarget::HighLID:
            lid_str = "HighLID";
            break;
        case deglib::builder::OptimizationTarget::LowLID:
            lid_str = "LowLID";
            break;
        case deglib::builder::OptimizationTarget::StreamingData:
            lid_str = "StreamingData";
            break;
        default:
            lid_str = "UnknownLID";
            break;
    }
    const char* suffix = flas_sorted ? "_FlasSorted" : "";
    return (ds.dataset_dir() / "deg" / fmt::format("{}D_{}_K{}_AddK{}Eps{:.1f}_{}{}.deg", dims, metric_str, cg.k, cg.k_ext, cg.eps_ext, lid_str, suffix))
        .string();
}

static deglib::cpu::InstructionSet parse_instruction_set(const std::string& str) {
    if (str == "auto" || str == "Auto") return deglib::cpu::InstructionSet::Auto;
    if (str == "scalar" || str == "Scalar") return deglib::cpu::InstructionSet::Scalar;
    if (str == "avx2" || str == "AVX2") return deglib::cpu::InstructionSet::AVX2;
    if (str == "avx512" || str == "AVX512") return deglib::cpu::InstructionSet::AVX512;
    log("Unknown instruction set '{}', falling back to Auto\n", str);
    return deglib::cpu::InstructionSet::Auto;
}

void print_help(const char* program_name) {
    log("=== DEG Benchmark Tool: bench_flas_presort ===\n\n");
    log("Description:\n");
    log("  Builds a DEG graph twice with identical parameters: once with natural insertion\n");
    log("  order and once with FLAS 1D pre-sorted insertion order. Compares build time and\n");
    log("  search quality (ANNS recall vs query time for a few eps values).\n\n");
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
    log("  --data-path, -d <path> Path to datasets directory (default: env DEG_DATA_PATH or ./data)\n");
    log("  --force-rebuild        Force rebuilding both graphs even if the graph files already exist\n");
    log("  --instruction <inst>   Select distance instruction set (auto, avx512, avx2, scalar)\n");
    log("  --threads <count>      Number of threads used for building the graph and FLAS presorting\n");
    log("                         (default: hardware_concurrency / 2)\n");
    log("  --flas-decay <value>   FLAS neighborhood radius decay rate (default: 0.9)\n");
    log("  --help, -h             Display this detailed help message\n\n");
}

struct AnnsRow {
    float eps = 0;
    float recall = 0;
    uint64_t time_us_per_query = 0;
};

/**
 * Evaluate ANNS recall and query time for every eps value. Mirrors test_graph_anns
 * (early abort when slower than the linear search baseline or the recall target is
 * reached) but returns the results for a direct comparison between two graphs.
 */
static std::vector<AnnsRow> evaluate_anns(
    const deglib::graph::InternalGraph& graph,
    const deglib::FeatureRepository& query_repository,
    const std::vector<std::vector<uint32_t>>& ground_truth,
    const Config& config,
    const uint64_t linear_baseline_us
) {
    const auto entry_vertex_indices = graph.getEntryVertexIndices();
    const auto test_size = uint32_t(query_repository.size());
    const uint32_t abort_sample_size = 100;

    std::vector<float> eps_parameter_sorted = config.eps_parameter;
    std::sort(eps_parameter_sorted.begin(), eps_parameter_sorted.end());
    log("Compute TOP{} for eps {}\n", config.anns_k, fmt::join(eps_parameter_sorted, ", "));

    std::vector<AnnsRow> rows;
    for (float eps : eps_parameter_sorted) {
        if (linear_baseline_us > 0 && test_size > abort_sample_size) {
            const auto sample_size = std::min(abort_sample_size, test_size);
            StopW sample_stopw = StopW();
            deglib::benchmark::test_approx_anns(
                graph, entry_vertex_indices, query_repository, ground_truth, eps, config.anns_k, sample_size, config.anns_threads
            );
            const auto sample_time_per_query = sample_stopw.getElapsedTimeMicro() / sample_size;
            if (sample_time_per_query > linear_baseline_us) {
                log("eps {:.3f} \t ABORTED ({}us/query > {}us linear search baseline after {} queries)\n", eps, sample_time_per_query, linear_baseline_us,
                    sample_size);
                break;
            }
        }

        StopW stopw = StopW();
        float recall = 0;
        for (size_t i = 0; i < config.anns_repeat; i++)
            recall = deglib::benchmark::test_approx_anns(
                graph, entry_vertex_indices, query_repository, ground_truth, eps, config.anns_k, test_size, config.anns_threads
            );
        const auto search_time_us = stopw.getElapsedTimeMicro();

        AnnsRow row;
        row.eps = eps;
        row.recall = recall;
        row.time_us_per_query = (search_time_us / test_size) / config.anns_repeat;
        rows.push_back(row);

        log("eps {:.3f} \t recall {:.5f} \t time_us_per_query {:6}us \t search time: {:6}ms\n", row.eps, row.recall, row.time_us_per_query,
            search_time_us / 1000);
        if (recall > 0.997) {
            log("Reached recall > 0.997, stopping further tests.\n");
            break;
        }
    }
    return rows;
}

/**
 * Print a side-by-side comparison of two ANNS evaluations. Rows are matched by their
 * eps value, so differently aborted sweeps still produce a valid table.
 */
static void print_anns_comparison(const std::vector<AnnsRow>& normal, const std::vector<AnnsRow>& flas_sorted, const uint32_t k) {
    log("\n--- ANNS Comparison (k={}) ---\n", k);
    log("{:>7} | {:>13} | {:>13} | {:>9} | {:>12} | {:>12} | {:>8}\n", "eps", "recall normal", "recall flas", "d recall", "us/q normal", "us/q flas",
        "speedup");

    size_t i = 0, j = 0;
    while (i < normal.size() || j < flas_sorted.size()) {
        if (j >= flas_sorted.size() || (i < normal.size() && normal[i].eps < flas_sorted[j].eps)) {
            log("{:7.3f} | {:13.5f} | {:>13} | {:>9} | {:12} | {:>12} | {:>8}\n", normal[i].eps, normal[i].recall, "-", "-", normal[i].time_us_per_query, "-",
                "-");
            i++;
        } else if (i >= normal.size() || flas_sorted[j].eps < normal[i].eps) {
            log("{:7.3f} | {:>13} | {:13.5f} | {:>9} | {:>12} | {:12} | {:>8}\n", flas_sorted[j].eps, "-", flas_sorted[j].recall, "-", "-",
                flas_sorted[j].time_us_per_query, "-");
            j++;
        } else {
            const auto d_recall = flas_sorted[j].recall - normal[i].recall;
            const auto speedup = (flas_sorted[j].time_us_per_query > 0) ? ((float)normal[i].time_us_per_query / (float)flas_sorted[j].time_us_per_query) : 0.0f;
            log("{:7.3f} | {:13.5f} | {:13.5f} | {:+9.5f} | {:12} | {:12} | {:7.2f}x\n", normal[i].eps, normal[i].recall, flas_sorted[j].recall, d_recall,
                normal[i].time_us_per_query, flas_sorted[j].time_us_per_query, speedup);
            i++;
            j++;
        }
    }
}

void run_flas_benchmark(
    const DatasetName& ds_name,
    const std::filesystem::path& data_path,
    const bool force_rebuild,
    const deglib::cpu::InstructionSet instruction,
    const uint32_t build_threads,
    const float flas_radius_decay
) {
    Dataset ds(ds_name, data_path);
    auto config = get_dataset_config(ds_name);
    config.build_threads = build_threads;
    if (flas_radius_decay > 0) config.flas_radius_decay = flas_radius_decay;

    log("\n=== Benchmarking FLAS Presorted Graph Creation & Evaluation for Dataset: {} ===\n", ds.name());

    auto setup_threads = std::thread::hardware_concurrency() / 2;
    if (!setup_dataset(ds, setup_threads, /*include_half=*/false, /*include_exploration=*/false)) {
        log("ERROR: Failed to setup dataset {}\n", ds.name());
        return;
    }

    auto base_repository = ds.load_base();
    auto query_repository = ds.load_query();
    const uint32_t dims = (uint32_t)base_repository.dims();

    const auto graph_path_normal = build_graph_filename(ds, config, dims, false);
    const auto graph_path_flas = build_graph_filename(ds, config, dims, true);
    ensure_directory(std::filesystem::path(graph_path_normal).parent_path());

    const auto log_path = (ds.dataset_dir() / "deg" / fmt::format("bench_flas_presort_{}.log", ds.name())).string();
    set_log_file(log_path, false);
    log("Logging to file: {}\n", log_path);

    log("\n--- Computing Linear Search Baseline ---\n");
    uint64_t linear_baseline_us = compute_linear_search_baseline(base_repository, ds.info().metric, 100, instruction) * 2;

    // -----------------------------------------------------------------------
    // Variant A: build the graph with natural insertion order
    // -----------------------------------------------------------------------
    bool built_normal = false;
    uint64_t build_time_normal_us = 0;
    if (!std::filesystem::exists(graph_path_normal) || force_rebuild) {
        log("\n=== Building Graph (natural insertion order) ===\n");
        log("Output graph: {}\n", graph_path_normal);
        StopW stopw = StopW();
        create_graph(
            base_repository, DataStreamType::AddAll, graph_path_normal, ds.info().metric, config.lid, config.k, config.k_ext, config.eps_ext, 0, 0, 0,
            config.build_threads, true, ds.info().scale, false, instruction
        );
        build_time_normal_us = stopw.getElapsedTimeMicro();
        built_normal = true;
        log("Graph build (natural order) took {:.1f}s\n", build_time_normal_us / 1000000.0);
    } else {
        log("\nGraph already exists: {}\n", graph_path_normal);
    }

    // -----------------------------------------------------------------------
    // Evaluate variant A
    // -----------------------------------------------------------------------
    std::vector<AnnsRow> anns_normal;
    if (std::filesystem::exists(graph_path_normal)) {
        const auto graph = deglib::graph::load_readonly_graph(graph_path_normal.c_str());
        log("Graph loaded: {} vertices\n", graph.size());

        log("\n--- ANNS Test (natural order, k={}) ---\n", config.anns_k);
        auto ground_truth = ds.load_groundtruth(config.anns_k, false);
        anns_normal = evaluate_anns(graph, query_repository, ground_truth, config, linear_baseline_us);
    } else {
        log("ERROR: Graph file not found: {}\n", graph_path_normal);
    }

    // -----------------------------------------------------------------------
    // FLAS 1D pre-sorting of the base data
    // -----------------------------------------------------------------------
    log("\n=== FLAS 1D Pre-sorting ===\n");
    const auto base_size = base_repository.size();
    if (base_size > 1) {
        const auto feature_byte_size = size_t(base_repository.getFeature(1) - base_repository.getFeature(0));
        if (feature_byte_size != size_t(dims) * sizeof(float)) {
            log("ERROR: FLAS presorting requires FP32 features, but the repository uses {} bytes per feature ({} dims)\n", feature_byte_size, dims);
            return;
        }
    }

    const auto flas_space = deglib::distances::FloatSpace(dims, ds.info().metric, instruction);
    StopW flas_stopw = StopW();
    auto sorted_indices = deglib::optimization::presort(
        reinterpret_cast<const float*>(base_repository.getFeature(0)), base_size, flas_space, config.flas_radius_decay, config.build_threads
    );
    const uint64_t flas_time_us = flas_stopw.getElapsedTimeMicro();
    log("FLAS presorting took {:.1f}s (radius decay {}, {} threads)\n", flas_time_us / 1000000.0, config.flas_radius_decay, config.build_threads);

    // verify that the result is a valid permutation of [0, base_size)
    {
        std::vector<bool> seen(base_size, false);
        bool valid = sorted_indices.size() == base_size;
        for (const auto idx : sorted_indices) {
            if (idx >= base_size || seen[idx]) {
                valid = false;
                break;
            }
            seen[idx] = true;
        }
        if (!valid) {
            log("ERROR: FLAS returned an invalid permutation\n");
            return;
        }
    }

    // -----------------------------------------------------------------------
    // Variant B: build the same graph with FLAS pre-sorted insertion order.
    // The external labels stay untouched, so all ground truth data remains valid.
    // -----------------------------------------------------------------------
    bool built_flas = false;
    uint64_t build_time_flas_us = 0;
    if (!std::filesystem::exists(graph_path_flas) || force_rebuild) {
        log("\n=== Building Graph (FLAS pre-sorted insertion order) ===\n");
        log("Output graph: {}\n", graph_path_flas);
        StopW stopw = StopW();
        create_graph(
            base_repository, DataStreamType::AddAll, graph_path_flas, ds.info().metric, config.lid, config.k, config.k_ext, config.eps_ext, 0, 0, 0,
            config.build_threads, true, ds.info().scale, false, instruction, sorted_indices
        );
        build_time_flas_us = stopw.getElapsedTimeMicro();
        built_flas = true;
        log("Graph build (FLAS sorted) took {:.1f}s\n", build_time_flas_us / 1000000.0);
    } else {
        log("\nGraph already exists: {}\n", graph_path_flas);
    }

    // -----------------------------------------------------------------------
    // Evaluate variant B
    // -----------------------------------------------------------------------
    std::vector<AnnsRow> anns_flas;
    if (std::filesystem::exists(graph_path_flas)) {
        const auto graph = deglib::graph::load_readonly_graph(graph_path_flas.c_str());
        log("Graph loaded: {} vertices\n", graph.size());

        log("\n--- ANNS Test (FLAS sorted, k={}) ---\n", config.anns_k);
        auto ground_truth = ds.load_groundtruth(config.anns_k, false);
        anns_flas = evaluate_anns(graph, query_repository, ground_truth, config, linear_baseline_us);
    } else {
        log("ERROR: Graph file not found: {}\n", graph_path_flas);
    }

    // -----------------------------------------------------------------------
    // Comparison
    // -----------------------------------------------------------------------
    print_anns_comparison(anns_normal, anns_flas, config.anns_k);

    // aggregate search performance over the commonly measured eps values
    size_t common_eps_count = 0;
    uint64_t search_time_normal_us = 0;
    uint64_t search_time_flas_us = 0;
    float recall_normal_sum = 0;
    float recall_flas_sum = 0;
    {
        size_t i = 0, j = 0;
        while (i < anns_normal.size() && j < anns_flas.size()) {
            if (anns_normal[i].eps < anns_flas[j].eps) {
                i++;
            } else if (anns_flas[j].eps < anns_normal[i].eps) {
                j++;
            } else {
                search_time_normal_us += anns_normal[i].time_us_per_query;
                search_time_flas_us += anns_flas[j].time_us_per_query;
                recall_normal_sum += anns_normal[i].recall;
                recall_flas_sum += anns_flas[j].recall;
                common_eps_count++;
                i++;
                j++;
            }
        }
    }

    log("\n=== Summary ===\n");
    if (built_normal)
        log("Build time (natural order):   {:10.1f} s\n", build_time_normal_us / 1000000.0);
    else
        log("Build time (natural order):   {:>10} (graph loaded from cache)\n", "-");
    log("FLAS presort time:            {:10.1f} s\n", flas_time_us / 1000000.0);
    if (built_flas)
        log("Build time (FLAS sorted):     {:10.1f} s\n", build_time_flas_us / 1000000.0);
    else
        log("Build time (FLAS sorted):     {:>10} (graph loaded from cache)\n", "-");

    if (common_eps_count > 0) {
        log("Search time (natural order):  {:10.1f} us/q (avg recall {:.5f}, {} eps values)\n", (float)search_time_normal_us / common_eps_count,
            recall_normal_sum / common_eps_count, common_eps_count);
        log("Search time (FLAS sorted):    {:10.1f} us/q (avg recall {:.5f})\n", (float)search_time_flas_us / common_eps_count,
            recall_flas_sum / common_eps_count);
    }

    if (built_normal && built_flas && build_time_flas_us > 0) {
        const auto total_flas_us = flas_time_us + build_time_flas_us;
        log("Total (presort + build):      {:10.1f} s\n", total_flas_us / 1000000.0);
        log("Build speedup:                {:10.2f} x\n", (float)build_time_normal_us / (float)build_time_flas_us);
        log("Total speedup:                {:10.2f} x\n", (float)build_time_normal_us / (float)total_flas_us);
    }

    if (common_eps_count > 0 && search_time_flas_us > 0) {
        log("Search speedup:               {:10.2f} x\n", (float)search_time_normal_us / (float)search_time_flas_us);
    }
}

int main(int argc, char* argv[]) {
    auto data_path = deglib::benchmark::get_default_data_path();
    DatasetName ds_name = DatasetName::SIFT1M;
    bool force_rebuild = false;
    deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto;
    uint32_t build_threads = std::thread::hardware_concurrency() / 2;
    float flas_radius_decay = 0.0f;  // 0 = use default from Config

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        } else if ((arg == "--data-path" || arg == "-d") && i + 1 < argc) {
            data_path = std::filesystem::path(argv[++i]);
        } else if (arg == "--force-rebuild") {
            force_rebuild = true;
        } else if (arg == "--instruction" && i + 1 < argc) {
            instruction = parse_instruction_set(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            build_threads = (uint32_t)std::stoul(argv[++i]);
        } else if (arg == "--flas-decay" && i + 1 < argc) {
            flas_radius_decay = std::stof(argv[++i]);
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

    log("=== bench_flas_presort ===\n");

    std::vector<DatasetName> datasets_to_run;
    if (ds_name == DatasetName::ALL) {
        for (const auto& d : DatasetName::all()) datasets_to_run.push_back(d);
    } else {
        datasets_to_run.push_back(ds_name);
    }

    for (const auto& current_ds : datasets_to_run) {
        run_flas_benchmark(current_ds, data_path, force_rebuild, instruction, build_threads, flas_radius_decay);
    }

    log("\nFLAS Presort Benchmark Finished Successfully.\n");
    return 0;
}
