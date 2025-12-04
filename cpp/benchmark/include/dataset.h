#pragma once

/**
 * @file dataset.h
 * @brief Dataset management utilities for deglib benchmarks.
 * 
 * This header provides:
 * - Dataset enum and metadata
 * - Download, extraction, and file renaming utilities
 * - Ground truth computation for ANNS and exploration queries
 * - Automatic generation of missing files
 */

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <atomic>
#include <chrono>
#include <limits>

#include <fmt/core.h>
#include <fmt/format.h>

#include "deglib.h"
#include "file_io.h"

namespace deglib::benchmark
{

// ============================================================================
// Forward Declarations
// ============================================================================

struct DatasetInfo;
struct DatasetPaths;

// ============================================================================
// Dataset Class
// ============================================================================

/**
 * Dataset wrapper class providing name, info, and path utilities.
 * Uses static instances for type-safe dataset identification.
 */
class Dataset {
public:
    // Static dataset instances
    static const Dataset SIFT1M;
    static const Dataset DEEP1M;
    static const Dataset GLOVE;
    static const Dataset AUDIO;
    static const Dataset Invalid;
    
    // All valid datasets for iteration
    static const std::array<Dataset, 4>& all() {
        static const std::array<Dataset, 4> datasets = {SIFT1M, DEEP1M, GLOVE, AUDIO};
        return datasets;
    }
    
    // Parse from string (case-insensitive)
    static Dataset from_string(const std::string& str) {
        std::string lower = str;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        
        for (const auto& ds : all()) {
            if (lower == ds.name()) {
                return ds;
            }
        }
        return Invalid;
    }
    
    // Accessors
    const char* name() const { return name_; }
    bool is_valid() const { return name_ != Invalid.name_; }
    
    // Convert to string (same as name())
    const char* to_string() const { return name_; }
    
    // Get dataset info (defined after DatasetInfo)
    DatasetInfo info() const;
    
    // Get dataset paths for a given data root (defined after DatasetPaths)
    DatasetPaths paths(const std::filesystem::path& data_root) const;
    
    // Comparison operators (compare by name pointer - works because of static instances)
    bool operator==(const Dataset& other) const { return name_ == other.name_; }
    bool operator!=(const Dataset& other) const { return name_ != other.name_; }
    
private:
    constexpr Dataset(const char* name) : name_(name) {}
    
    const char* name_;
};

// Static dataset definitions
inline constexpr Dataset Dataset::SIFT1M{"sift1m"};
inline constexpr Dataset Dataset::DEEP1M{"deep1m"};
inline constexpr Dataset Dataset::GLOVE{"glove"};
inline constexpr Dataset Dataset::AUDIO{"audio"};
inline constexpr Dataset Dataset::Invalid{"invalid"};

// ============================================================================
// Dataset Info Structure
// ============================================================================

struct DatasetInfo {
    Dataset dataset;                // The dataset this info belongs to
    std::string download_url;       // URL to download the dataset
    deglib::Metric metric;          // Distance metric (L2, etc.)
    size_t base_count;              // Number of base vectors
    size_t query_count;             // Number of query vectors
    uint32_t dims;                  // Feature dimensions
    
    // File names (canonical)
    std::string base_file;          // e.g., "sift1m_base.fvecs"
    std::string query_file;         // e.g., "sift1m_query.fvecs"
    
    // Exploration files
    std::string explore_query_file;           // e.g., "sift1m_explore_query.fvecs"
    std::string explore_entry_vertex_file;    // e.g., "sift1m_explore_entry_vertex.ivecs"
    std::string explore_groundtruth_file;     // e.g., "sift1m_explore_groundtruth_top1000.ivecs"
    
    // Exploration parameters
    static constexpr size_t EXPLORE_SAMPLE_COUNT = 10000;
    static constexpr uint32_t EXPLORE_TOPK = 1000;
    
    // Ground truth parameters
    static constexpr uint32_t GROUNDTRUTH_TOPK = 1024;
    static constexpr size_t GROUNDTRUTH_STEP = 100000;  // Steps for partial ground truth
    
    // Convenience accessor for name
    const char* name() const { return dataset.name(); }
};

// ============================================================================
// Dataset Configuration Factory
// ============================================================================

inline DatasetInfo make_dataset_info(const Dataset& ds) {
    DatasetInfo info{ds, {}, deglib::Metric::L2, 0, 0, 0, {}, {}, {}, {}, {}};
    
    std::string name = ds.name();
    
    // Set canonical file names
    info.base_file = name + "_base.fvecs";
    info.query_file = name + "_query.fvecs";
    info.explore_query_file = name + "_explore_query.fvecs";
    info.explore_entry_vertex_file = name + "_explore_entry_vertex.ivecs";
    info.explore_groundtruth_file = name + "_explore_groundtruth_top1000.ivecs";
    
    if (ds == Dataset::SIFT1M) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/sift.tar.gz";
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 128;
    } else if (ds == Dataset::DEEP1M) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/deep1m.tar.gz";
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 96;
    } else if (ds == Dataset::GLOVE) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/glove.tar.gz";
        info.base_count = 1183514;
        info.query_count = 10000;
        info.dims = 100;
    } else if (ds == Dataset::AUDIO) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/audio.tar.gz";
        info.base_count = 53387;
        info.query_count = 200;
        info.dims = 192;
    }
    
    return info;
}

// Dataset::info() implementation
inline DatasetInfo Dataset::info() const {
    return make_dataset_info(*this);
}

// ============================================================================
// Path Utilities
// ============================================================================

struct DatasetPaths {
    std::filesystem::path data_root;     // DATA_PATH
    std::filesystem::path dataset_dir;   // DATA_PATH/sift1m
    std::filesystem::path files_dir;     // DATA_PATH/sift1m/sift1m
    DatasetInfo info;                    // Cached dataset info
    
    DatasetPaths(const std::filesystem::path& root, const Dataset& dataset)
        : data_root(root)
        , dataset_dir(root / dataset.name())
        , files_dir(root / dataset.name() / dataset.name())
        , info(dataset.info())
    {}
    
    std::string base_file() const {
        return (files_dir / info.base_file).string();
    }
    
    std::string query_file() const {
        return (files_dir / info.query_file).string();
    }
    
    std::string explore_query_file() const {
        return (files_dir / info.explore_query_file).string();
    }
    
    std::string explore_entry_vertex_file() const {
        return (files_dir / info.explore_entry_vertex_file).string();
    }
    
    std::string explore_groundtruth_file() const {
        return (files_dir / info.explore_groundtruth_file).string();
    }
    
    // Ground truth file for specific base count
    std::string groundtruth_file(size_t nb) const {
        return (files_dir / fmt::format("{}_groundtruth_top{}_nb{}.ivecs", 
                info.name(), DatasetInfo::GROUNDTRUTH_TOPK, nb)).string();
    }
    
    // Ground truth file for full dataset
    std::string groundtruth_file_full() const {
        return groundtruth_file(info.base_count);
    }
    
    // Ground truth file for half dataset
    std::string groundtruth_file_half() const {
        return groundtruth_file(info.base_count / 2);
    } 
};

// Dataset::paths() implementation
inline DatasetPaths Dataset::paths(const std::filesystem::path& data_root) const {
    return DatasetPaths(data_root, *this);
}


// ============================================================================
// Ground Truth Computation
// ============================================================================

/**
 * Compute ground truth (brute force k-NN) for queries against base vectors.
 * Returns a flat array of [query_count * k] indices.
 */
inline std::vector<uint32_t> compute_knn_groundtruth(
    const deglib::FeatureRepository& base_repo,
    const deglib::FeatureRepository& query_repo,
    const deglib::Metric metric,
    const uint32_t k_target,
    const size_t base_limit = 0,  // 0 means use all
    const uint32_t thread_count = 1)
{
    const auto base_size = base_limit > 0 ? (uint32_t)std::min(base_limit, base_repo.size()) : (uint32_t)base_repo.size();
    const auto query_size = (uint32_t)query_repo.size();
    const auto dims = base_repo.dims();

    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    auto topLists = std::vector<uint32_t>(k_target * query_size);
    std::atomic<uint32_t> progress{0};
    const auto start = std::chrono::steady_clock::now();

    deglib::concurrent::parallel_for(0, query_size, thread_count, [&](size_t q, size_t) {
        const auto query = query_repo.getFeature((uint32_t)q);

        auto worst_distance = (std::numeric_limits<float>::max)();
        auto results = deglib::search::ResultSet();
        
        for (uint32_t b = 0; b < base_size; b++) {
            const auto distance = dist_func(query, base_repo.getFeature(b), dist_func_param);
            if (distance < worst_distance) {
                results.emplace(b, distance);
                if (results.size() > k_target) {
                    results.pop();
                    worst_distance = results.top().getDistance();
                }
            }
        }

        auto topList = topLists.data() + (k_target * q);
        for (int32_t i = k_target - 1; i >= 0; i--) {
            if (!results.empty()) {
                topList[i] = results.top().getInternalIndex();
                results.pop();
            } else {
                topList[i] = (std::numeric_limits<uint32_t>::max)();
            }
        }

        uint32_t count = ++progress;
        if (count % 100 == 0 || count == query_size) {
            const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            fmt::print("  Ground truth progress: {}/{} queries ({:.1f}%) after {}ms\n", 
                       count, query_size, 100.0f * count / query_size, duration_ms);
        }
    });

    return topLists;
}

// ============================================================================
// Dataset-specific Setup Functions
// ============================================================================

namespace detail {

/**
 * Setup SIFT1M dataset - download, extract, rename files
 */
inline bool setup_sift1m_files(const DatasetPaths& paths) {
    fmt::print("\n=== Setting up SIFT1M dataset ===\n");
    
    const auto& info = paths.info;
    const auto archive_file = paths.dataset_dir / "sift.tar.gz";
    const auto tmp_dir = paths.dataset_dir / "_tmp_extract";
    
    // Ensure directories exist
    ensure_directory(paths.dataset_dir);
    ensure_directory(paths.files_dir);
    
    // Check if already setup (base file exists with correct name)
    if (file_exists(paths.base_file())) {
        fmt::print("SIFT1M already set up at {}\n", paths.files_dir.string());
        return true;
    }
    
    // Download if archive doesn't exist
    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) {
            fmt::print(stderr, "Failed to download SIFT1M\n");
            return false;
        }
    }
    
    // Clean up any previous tmp directory and create fresh one
    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);
    
    // Extract archive to tmp directory
    if (!extract_tar_gz(archive_file, tmp_dir)) {
        fmt::print(stderr, "Failed to extract SIFT1M archive\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    // Find extracted directory containing base file
    auto extracted_dir = find_directory_with_file(tmp_dir, "sift_base.fvecs");
    if (extracted_dir.empty()) {
        fmt::print(stderr, "Could not find extracted SIFT1M files\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    fmt::print("Found extracted files in: {}\n", extracted_dir.string());
    
    // Move files to canonical names
    move_file(extracted_dir / "sift_base.fvecs", paths.files_dir / info.base_file);
    move_file(extracted_dir / "sift_query.fvecs", paths.files_dir / info.query_file);
    move_file(extracted_dir / "sift_explore_query.fvecs", paths.files_dir / info.explore_query_file);
    move_file(extracted_dir / "sift_explore_entry_vertex.ivecs", paths.files_dir / info.explore_entry_vertex_file);
    // Rename explore ground truth (different naming convention)
    move_file(extracted_dir / "sift_explore_ground_truth.ivecs", paths.files_dir / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("SIFT1M files set up in: {}\n", paths.files_dir.string());
    return true;
}

/**
 * Setup DEEP1M dataset
 */
inline bool setup_deep1m_files(const DatasetPaths& paths) {
    fmt::print("\n=== Setting up DEEP1M dataset ===\n");
    
    const auto& info = paths.info;
    const auto archive_file = paths.dataset_dir / "deep1m.tar.gz";
    const auto tmp_dir = paths.dataset_dir / "_tmp_extract";
    
    ensure_directory(paths.dataset_dir);
    ensure_directory(paths.files_dir);
    
    if (file_exists(paths.base_file())) {
        fmt::print("DEEP1M already set up at {}\n", paths.files_dir.string());
        return true;
    }
    
    // Download if archive doesn't exist
    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) {
            fmt::print(stderr, "Failed to download DEEP1M\n");
            return false;
        }
    }
    
    // Clean up any previous tmp directory and create fresh one
    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);
    
    // Extract archive to tmp directory
    if (!extract_tar_gz(archive_file, tmp_dir)) {
        fmt::print(stderr, "Failed to extract DEEP1M archive\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    // Find extracted directory containing base file
    auto extracted_dir = find_directory_with_file(tmp_dir, "deep1m_base.fvecs");
    if (extracted_dir.empty()) {
        fmt::print(stderr, "Could not find extracted DEEP1M files\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    fmt::print("Found extracted files in: {}\n", extracted_dir.string());
    
    // Move files to canonical names
    move_file(extracted_dir / "deep1m_base.fvecs", paths.files_dir / info.base_file);
    move_file(extracted_dir / "deep1m_query.fvecs", paths.files_dir / info.query_file);
    
    // Handle exploration files if they exist (with possible different naming)
    move_file(extracted_dir / "deep1m_explore_query.fvecs", paths.files_dir / info.explore_query_file);
    move_file(extracted_dir / "deep1m_explore_entry_vertex.ivecs", paths.files_dir / info.explore_entry_vertex_file);
    move_file(extracted_dir / "deep1m_explore_ground_truth.ivecs", paths.files_dir / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("DEEP1M files set up in: {}\n", paths.files_dir.string());
    return true;
}

/**
 * Setup GLOVE dataset
 */
inline bool setup_glove_files(const DatasetPaths& paths) {
    fmt::print("\n=== Setting up GLOVE dataset ===\n");
    
    const auto& info = paths.info;
    const auto archive_file = paths.dataset_dir / "glove.tar.gz";
    const auto tmp_dir = paths.dataset_dir / "_tmp_extract";
    
    ensure_directory(paths.dataset_dir);
    ensure_directory(paths.files_dir);
    
    if (file_exists(paths.base_file())) {
        fmt::print("GLOVE already set up at {}\n", paths.files_dir.string());
        return true;
    }
    
    // Download if archive doesn't exist
    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) {
            fmt::print(stderr, "Failed to download GLOVE\n");
            return false;
        }
    }
    
    // Clean up any previous tmp directory and create fresh one
    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);
    
    // Extract archive to tmp directory
    if (!extract_tar_gz(archive_file, tmp_dir)) {
        fmt::print(stderr, "Failed to extract GLOVE archive\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    // Find extracted directory containing base file (GLOVE uses "glove-100_" prefix)
    auto extracted_dir = find_directory_with_file(tmp_dir, "glove-100_base.fvecs");
    if (extracted_dir.empty()) {
        fmt::print(stderr, "Could not find extracted GLOVE files\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    fmt::print("Found extracted files in: {}\n", extracted_dir.string());
    
    // Move files to canonical names (GLOVE uses "glove-100_" prefix in archive)
    move_file(extracted_dir / "glove-100_base.fvecs", paths.files_dir / info.base_file);
    move_file(extracted_dir / "glove-100_query.fvecs", paths.files_dir / info.query_file);
    
    // Handle exploration files if they exist
    move_file(extracted_dir / "glove-100_explore_query.fvecs", paths.files_dir / info.explore_query_file);
    move_file(extracted_dir / "glove_explore_query.fvecs", paths.files_dir / info.explore_query_file);
    move_file(extracted_dir / "glove-100_explore_entry_vertex.ivecs", paths.files_dir / info.explore_entry_vertex_file);
    move_file(extracted_dir / "glove_explore_entry_vertex.ivecs", paths.files_dir / info.explore_entry_vertex_file);
    move_file(extracted_dir / "glove-100_explore_ground_truth.ivecs", paths.files_dir / info.explore_groundtruth_file);
    move_file(extracted_dir / "glove_explore_ground_truth.ivecs", paths.files_dir / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("GLOVE files set up in: {}\n", paths.files_dir.string());
    return true;
}

/**
 * Setup AUDIO dataset
 */
inline bool setup_audio_files(const DatasetPaths& paths) {
    fmt::print("\n=== Setting up AUDIO dataset ===\n");
    
    const auto& info = paths.info;
    const auto archive_file = paths.dataset_dir / "audio.tar.gz";
    const auto tmp_dir = paths.dataset_dir / "_tmp_extract";
    
    ensure_directory(paths.dataset_dir);
    ensure_directory(paths.files_dir);
    
    if (file_exists(paths.base_file())) {
        fmt::print("AUDIO already set up at {}\n", paths.files_dir.string());
        return true;
    }
    
    // Download if archive doesn't exist
    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) {
            fmt::print(stderr, "Failed to download AUDIO\n");
            return false;
        }
    }
    
    // Clean up any previous tmp directory and create fresh one
    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);
    
    // Extract archive to tmp directory
    if (!extract_tar_gz(archive_file, tmp_dir)) {
        fmt::print(stderr, "Failed to extract AUDIO archive\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    // Find extracted directory containing base file
    auto extracted_dir = find_directory_with_file(tmp_dir, "audio_base.fvecs");
    if (extracted_dir.empty()) {
        fmt::print(stderr, "Could not find extracted AUDIO files\n");
        remove_directory(tmp_dir);
        return false;
    }
    
    fmt::print("Found extracted files in: {}\n", extracted_dir.string());
    
    // Move files to canonical names
    move_file(extracted_dir / "audio_base.fvecs", paths.files_dir / info.base_file);
    move_file(extracted_dir / "audio_query.fvecs", paths.files_dir / info.query_file);
    
    // Handle exploration files if they exist
    move_file(extracted_dir / "audio_explore_query.fvecs", paths.files_dir / info.explore_query_file);
    move_file(extracted_dir / "audio_explore_entry_vertex.ivecs", paths.files_dir / info.explore_entry_vertex_file);
    move_file(extracted_dir / "audio_explore_ground_truth.ivecs", paths.files_dir / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("AUDIO files set up in: {}\n", paths.files_dir.string());
    return true;
}

} // namespace detail

// ============================================================================
// Exploration File Generation
// ============================================================================

/**
 * Generate exploration files by subsampling the base dataset.
 * Creates:
 * - explore_query.fvecs: Features of sampled entry vertices
 * - explore_entry_vertex.ivecs: IDs of sampled entry vertices
 * - explore_groundtruth_top1000.ivecs: Top-1000 nearest neighbors for each entry
 */
inline bool generate_exploration_files(
    const DatasetPaths& paths,
    const deglib::FeatureRepository& base_repo,
    const uint32_t thread_count = 4)
{
    const auto& info = paths.info;
    const size_t base_size = base_repo.size();
    const size_t sample_count = DatasetInfo::EXPLORE_SAMPLE_COUNT;
    const uint32_t topk = DatasetInfo::EXPLORE_TOPK;
    const uint32_t dims = (uint32_t)base_repo.dims();
    
    // Check if files already exist
    if (file_exists(paths.explore_query_file()) &&
        file_exists(paths.explore_entry_vertex_file()) &&
        file_exists(paths.explore_groundtruth_file())) {
        fmt::print("Exploration files already exist\n");
        return true;
    }
    
    fmt::print("\n=== Generating exploration files ===\n");
    fmt::print("Base size: {}, Sample count: {}, TopK: {}\n", base_size, sample_count, topk);
    
    // Subsample: step through base data to get exactly sample_count elements
    std::vector<uint32_t> entry_ids(sample_count);
    std::vector<float> entry_features(sample_count * dims);
    
    const double step = static_cast<double>(base_size) / sample_count;
    for (size_t i = 0; i < sample_count; i++) {
        uint32_t idx = static_cast<uint32_t>(i * step);
        if (idx >= base_size) idx = (uint32_t)base_size - 1;
        
        entry_ids[i] = idx;
        const float* src = reinterpret_cast<const float*>(base_repo.getFeature(idx));
        std::copy(src, src + dims, entry_features.data() + i * dims);
    }
    
    fmt::print("Selected {} entry vertices with step {:.2f}\n", sample_count, step);
    
    // Write entry vertex IDs
    ivecs_write(paths.explore_entry_vertex_file().c_str(), 1, sample_count, entry_ids.data());
    fmt::print("Wrote: {}\n", paths.explore_entry_vertex_file());
    
    // Write entry features
    fvecs_write(paths.explore_query_file().c_str(), dims, sample_count, entry_features.data());
    fmt::print("Wrote: {}\n", paths.explore_query_file());
    
    // Create a temporary repository for queries
    auto query_features = std::make_unique<std::byte[]>(sample_count * dims * sizeof(float));
    std::memcpy(query_features.get(), entry_features.data(), sample_count * dims * sizeof(float));
    deglib::StaticFeatureRepository query_repo(std::move(query_features), dims, sample_count, sizeof(float));
    
    // Compute ground truth for exploration (top-1000 from base for each entry)
    fmt::print("Computing exploration ground truth (this may take a while)...\n");
    auto groundtruth = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, 0, thread_count);
    
    // Write ground truth
    ivecs_write(paths.explore_groundtruth_file().c_str(), topk, sample_count, groundtruth.data());
    fmt::print("Wrote: {}\n", paths.explore_groundtruth_file());
    
    return true;
}

// ============================================================================
// Ground Truth File Generation
// ============================================================================

/**
 * Generate ground truth files for ANNS benchmark.
 * Creates ground truth files for:
 * - Every GROUNDTRUTH_STEP base vectors (100k, 200k, etc.)
 * - Half of the base vectors
 * - All base vectors
 */
inline bool generate_groundtruth_files(
    const DatasetPaths& paths,
    const deglib::FeatureRepository& base_repo,
    const deglib::FeatureRepository& query_repo,
    const uint32_t thread_count = 4)
{
    const auto& info = paths.info;
    const size_t base_size = base_repo.size();
    const uint32_t topk = DatasetInfo::GROUNDTRUTH_TOPK;
    const size_t step = DatasetInfo::GROUNDTRUTH_STEP;
    
    fmt::print("\n=== Generating ground truth files ===\n");
    fmt::print("Base size: {}, Query size: {}, TopK: {}, Step: {}\n", 
               base_size, query_repo.size(), topk, step);
    
    // Generate for each step
    for (size_t nb = step; nb <= base_size; nb += step) {
        std::string gt_file = paths.groundtruth_file(nb);
        
        if (file_exists(gt_file)) {
            fmt::print("Ground truth exists: {}\n", gt_file);
            continue;
        }
        
        fmt::print("\nComputing ground truth for nb={} ...\n", nb);
        auto groundtruth = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, nb, thread_count);
        
        ivecs_write(gt_file.c_str(), topk, query_repo.size(), groundtruth.data());
        fmt::print("Wrote: {}\n", gt_file);
    }
    
    // Generate for half of base (if not already covered by step)
    size_t half_count = base_size / 2;
    if (half_count % step != 0) {
        std::string gt_half_file = paths.groundtruth_file_half();
        
        if (!file_exists(gt_half_file)) {
            fmt::print("\nComputing ground truth for half (nb={}) ...\n", half_count);
            auto groundtruth = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, half_count, thread_count);
            
            ivecs_write(gt_half_file.c_str(), topk, query_repo.size(), groundtruth.data());
            fmt::print("Wrote: {}\n", gt_half_file);
        }
    }
    
    // Generate for full base (if not already covered by step)
    if (base_size % step != 0) {
        std::string gt_full_file = paths.groundtruth_file_full();
        
        if (!file_exists(gt_full_file)) {
            fmt::print("\nComputing ground truth for full (nb={}) ...\n", base_size);
            auto groundtruth = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, 0, thread_count);
            
            ivecs_write(gt_full_file.c_str(), topk, query_repo.size(), groundtruth.data());
            fmt::print("Wrote: {}\n", gt_full_file);
        }
    }
    
    return true;
}

// ============================================================================
// Main Setup Function
// ============================================================================

/**
 * Set up a dataset: download, extract, rename files, generate missing data.
 * 
 * @param data_path The root data path (DATA_PATH)
 * @param dataset The dataset to set up
 * @param thread_count Number of threads for ground truth computation
 * @return true if successful
 */
inline bool setup_dataset(
    const std::filesystem::path& data_path,
    const Dataset& dataset,
    const uint32_t thread_count = 4)
{
    if (!dataset.is_valid()) {
        fmt::print(stderr, "Invalid dataset\n");
        return false;
    }
    
    auto paths = dataset.paths(data_path);
    const auto& info = paths.info;
    
    fmt::print("\n============================================================\n");
    fmt::print("Setting up dataset: {}\n", dataset.name());
    fmt::print("Data path: {}\n", data_path.string());
    fmt::print("Dataset directory: {}\n", paths.dataset_dir.string());
    fmt::print("Files directory: {}\n", paths.files_dir.string());
    fmt::print("============================================================\n");
    
    // Step 1: Download and extract dataset files
    bool setup_ok = false;
    if (dataset == Dataset::SIFT1M) {
        setup_ok = detail::setup_sift1m_files(paths);
    } else if (dataset == Dataset::DEEP1M) {
        setup_ok = detail::setup_deep1m_files(paths);
    } else if (dataset == Dataset::GLOVE) {
        setup_ok = detail::setup_glove_files(paths);
    } else if (dataset == Dataset::AUDIO) {
        setup_ok = detail::setup_audio_files(paths);
    } else {
        fmt::print(stderr, "Unknown dataset\n");
        return false;
    }
    
    if (!setup_ok) {
        fmt::print(stderr, "Failed to set up dataset files\n");
        return false;
    }
    
    // Step 2: Verify required files exist
    if (!file_exists(paths.base_file())) {
        fmt::print(stderr, "Base file not found: {}\n", paths.base_file());
        return false;
    }
    if (!file_exists(paths.query_file())) {
        fmt::print(stderr, "Query file not found: {}\n", paths.query_file());
        return false;
    }
    
    fmt::print("\nLoading base repository...\n");
    auto base_repo = deglib::load_static_repository(paths.base_file().c_str());
    fmt::print("Loaded {} vectors of dimension {}\n", base_repo.size(), base_repo.dims());
    
    fmt::print("\nLoading query repository...\n");
    auto query_repo = deglib::load_static_repository(paths.query_file().c_str());
    fmt::print("Loaded {} vectors of dimension {}\n", query_repo.size(), query_repo.dims());
    
    // Step 3: Generate exploration files if missing
    generate_exploration_files(paths, base_repo, thread_count);
    
    // Step 4: Generate ground truth files if missing
    generate_groundtruth_files(paths, base_repo, query_repo, thread_count);
    
    fmt::print("\n============================================================\n");
    fmt::print("Dataset {} setup complete!\n", dataset.name());
    fmt::print("\nFiles:\n");
    fmt::print("  Base:     {}\n", paths.base_file());
    fmt::print("  Query:    {}\n", paths.query_file());
    fmt::print("  Explore Query:  {}\n", paths.explore_query_file());
    fmt::print("  Explore Entry:  {}\n", paths.explore_entry_vertex_file());
    fmt::print("  Explore GT:     {}\n", paths.explore_groundtruth_file());
    fmt::print("  GT Full:  {}\n", paths.groundtruth_file_full());
    fmt::print("  GT Half:  {}\n", paths.groundtruth_file_half());
    fmt::print("============================================================\n\n");
    
    return true;
}

/**
 * Convenience function to set up all datasets.
 */
inline bool setup_all_datasets(
    const std::filesystem::path& data_path,
    const uint32_t thread_count = 4)
{
    bool all_ok = true;
    
    for (const auto& ds : Dataset::all()) {
        if (!setup_dataset(data_path, ds, thread_count)) {
            fmt::print(stderr, "Failed to set up {}\n", ds.name());
            all_ok = false;
        }
    }
    
    return all_ok;
}

} // namespace deglib::benchmark
