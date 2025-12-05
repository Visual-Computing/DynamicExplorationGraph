/**
 * @file deglib_setup_dataset.cpp
 * @brief Tool to download and set up benchmark datasets.
 * 
 * Usage: deglib_setup_dataset <dataset> [--threads N]
 * Datasets: sift1m, deep1m, glove, audio, all
 */

#include <filesystem>
#include <string>

#include <fmt/core.h>

#include "dataset.h"

int main(int argc, char* argv[]) {
    fmt::print("=== DEG Dataset Setup Tool ===\n\n");
    
    #if defined(USE_AVX)
        fmt::print("CPU: AVX2\n");
    #elif defined(USE_SSE)
        fmt::print("CPU: SSE\n");
    #else
        fmt::print("CPU: generic\n");
    #endif
    
    const auto data_path = std::filesystem::path(DATA_PATH);
    fmt::print("DATA_PATH: {}\n\n", data_path.string());
    
    // Parse arguments
    std::string dataset_arg = "sift1m";
    uint32_t thread_count = std::thread::hardware_concurrency();
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            fmt::print("Usage: {} <dataset> [--threads N]\n", argv[0]);
            fmt::print("\nDatasets:\n");
            fmt::print("  sift1m  - SIFT1M (1M vectors, 128D)\n");
            fmt::print("  deep1m  - Deep1M (1M vectors, 96D)\n");
            fmt::print("  glove   - GloVe (1.18M vectors, 100D)\n");
            fmt::print("  audio   - Audio (53K vectors, 192D)\n");
            fmt::print("  all     - Set up all datasets\n");
            fmt::print("\nOptions:\n");
            fmt::print("  --threads N  Number of threads for ground truth computation (default: 4)\n");
            return 0;
        }
        
        if (arg == "--threads" && i + 1 < argc) {
            thread_count = std::stoi(argv[++i]);
            continue;
        }
        
        // Dataset argument
        dataset_arg = arg;
    }
    
    fmt::print("Thread count: {}\n\n", thread_count);
    
    // Set up dataset(s)
    if (dataset_arg == "all") {
        fmt::print("Setting up all datasets...\n\n");
        deglib::benchmark::setup_all_datasets(data_path, thread_count);
    } else {
        auto ds = deglib::benchmark::DatasetName::from_string(dataset_arg);
        if (!ds.is_valid()) {
            fmt::print(stderr, "Unknown dataset: {}\n", dataset_arg);
            fmt::print(stderr, "Valid datasets: sift1m, deep1m, glove, audio, all\n");
            return 1;
        }
        
        if (!deglib::benchmark::setup_dataset(ds, data_path, thread_count)) {
            fmt::print(stderr, "Failed to set up dataset: {}\n", dataset_arg);
            return 1;
        }
    }
    
    fmt::print("\nDone!\n");
    return 0;
}
