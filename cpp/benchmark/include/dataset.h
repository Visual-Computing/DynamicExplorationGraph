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
#include <unordered_set>

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
class Dataset;

// ============================================================================
// DatasetName - Enum-like class for dataset identification
// ============================================================================

/**
 * Lightweight enum-like class identifying a dataset by name.
 * Uses static instances for type-safe dataset identification.
 */
class DatasetName {
public:
    // Static dataset instances
    static const DatasetName SIFT1M;
    static const DatasetName DEEP1M;
    static const DatasetName GLOVE;
    static const DatasetName AUDIO;
    static const DatasetName Invalid;
    
    // All valid datasets for iteration
    static const std::array<DatasetName, 4>& all() {
        static const std::array<DatasetName, 4> datasets = {SIFT1M, DEEP1M, GLOVE, AUDIO};
        return datasets;
    }
    
    // Parse from string (case-insensitive)
    static DatasetName from_string(const std::string& str) {
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
    const char* to_string() const { return name_; }
    
    // Get dataset info
    DatasetInfo info() const;
    
    // Comparison operators
    bool operator==(const DatasetName& other) const { return name_ == other.name_; }
    bool operator!=(const DatasetName& other) const { return name_ != other.name_; }
    
private:
    constexpr DatasetName(const char* name) : name_(name) {}
    const char* name_;
};

// Static DatasetName definitions
inline constexpr DatasetName DatasetName::SIFT1M{"sift1m"};
inline constexpr DatasetName DatasetName::DEEP1M{"deep1m"};
inline constexpr DatasetName DatasetName::GLOVE{"glove"};
inline constexpr DatasetName DatasetName::AUDIO{"audio"};
inline constexpr DatasetName DatasetName::Invalid{"invalid"};

// ============================================================================
// Dataset Info Structure
// ============================================================================

struct DatasetInfo {
    DatasetName dataset_name;       // The dataset name this info belongs to
    std::string download_url;       // URL to download the dataset
    deglib::Metric metric;          // Distance metric (L2, etc.)
    size_t base_count;              // Number of base vectors
    size_t query_count;             // Number of query vectors
    uint32_t dims;                  // Feature dimensions
    
    // File names (canonical)
    std::string base_file;          // e.g., "sift1m_base.fvecs"
    std::string query_file;         // e.g., "sift1m_query.fvecs"
    
    // Exploration files (sampled subset)
    std::string explore_query_file;           // e.g., "sift1m_explore_query.fvecs"
    std::string explore_entry_vertex_file;    // e.g., "sift1m_explore_entry_vertex.ivecs"
    std::string explore_groundtruth_file;     // e.g., "sift1m_explore_groundtruth_top1000.ivecs"
    
    // Full exploration ground truth (all base elements)
    std::string full_explore_groundtruth_file;  // e.g., "sift1m_full_explore_groundtruth_top1000.ivecs"
    
    // Exploration parameters
    static constexpr size_t EXPLORE_SAMPLE_COUNT = 10000;
    static constexpr uint32_t EXPLORE_TOPK = 1000;
    
    // Ground truth parameters
    static constexpr uint32_t GROUNDTRUTH_TOPK = 1024;
    static constexpr size_t GROUNDTRUTH_STEP = 100000;  // Steps for partial ground truth
    
    // Convenience accessor for name
    const char* name() const { return dataset_name.name(); }
};

// ============================================================================
// Dataset Configuration Factory
// ============================================================================

inline DatasetInfo make_dataset_info(const DatasetName& ds) {
    DatasetInfo info{ds, {}, deglib::Metric::L2, 0, 0, 0, {}, {}, {}, {}, {}, {}};
    
    std::string name = ds.name();
    
    // Set canonical file names
    info.base_file = name + "_base.fvecs";
    info.query_file = name + "_query.fvecs";
    info.explore_query_file = name + "_explore_query.fvecs";
    info.explore_entry_vertex_file = name + "_explore_entry_vertex.ivecs";
    info.explore_groundtruth_file = name + "_explore_groundtruth_top1000.ivecs";
    info.full_explore_groundtruth_file = name + "_full_explore_groundtruth_top1000.ivecs";
    
    if (ds == DatasetName::SIFT1M) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/sift.tar.gz";
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 128;
    } else if (ds == DatasetName::DEEP1M) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/deep1m.tar.gz";
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 96;
    } else if (ds == DatasetName::GLOVE) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/glove.tar.gz";
        info.base_count = 1183514;
        info.query_count = 10000;
        info.dims = 100;
    } else if (ds == DatasetName::AUDIO) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/audio.tar.gz";
        info.base_count = 53387;
        info.query_count = 200;
        info.dims = 192;
    }
    
    return info;
}

// DatasetName::info() implementation
inline DatasetInfo DatasetName::info() const {
    return make_dataset_info(*this);
}

// ============================================================================
// Dataset Class - Main class holding data_root and providing all utilities
// ============================================================================

/**
 * Dataset class combining dataset identification with data root path.
 * Provides all path utilities and data loading methods.
 */
class Dataset {
public:
    Dataset(const DatasetName& name, const std::filesystem::path& data_root)
        : name_(name)
        , data_root_(data_root)
        , dataset_dir_(data_root / name.name())
        , files_dir_(data_root / name.name() / name.name())
        , info_(name.info())
    {}
    
    // ========== Accessors ==========
    
    const DatasetName& dataset_name() const { return name_; }
    const char* name() const { return name_.name(); }
    bool is_valid() const { return name_.is_valid(); }
    const DatasetInfo& info() const { return info_; }
    
    const std::filesystem::path& data_root() const { return data_root_; }
    const std::filesystem::path& dataset_dir() const { return dataset_dir_; }
    const std::filesystem::path& files_dir() const { return files_dir_; }
    
    // ========== Path Methods ==========
    
    std::string base_file() const {
        return (files_dir_ / info_.base_file).string();
    }
    
    std::string query_file() const {
        return (files_dir_ / info_.query_file).string();
    }
    
    std::string explore_query_file() const {
        return (files_dir_ / info_.explore_query_file).string();
    }
    
    std::string explore_entry_vertex_file() const {
        return (files_dir_ / info_.explore_entry_vertex_file).string();
    }
    
    std::string explore_groundtruth_file() const {
        return (files_dir_ / info_.explore_groundtruth_file).string();
    }
    
    // Full exploration ground truth file (top-k for ALL base elements)
    std::string full_explore_groundtruth_file() const {
        return (files_dir_ / info_.full_explore_groundtruth_file).string();
    }
    
    // Ground truth file for specific base count
    std::string groundtruth_file(size_t nb) const {
        return (files_dir_ / fmt::format("{}_groundtruth_top{}_nb{}.ivecs", 
                info_.name(), DatasetInfo::GROUNDTRUTH_TOPK, nb)).string();
    }
    
    // Ground truth file for full dataset
    std::string groundtruth_file_full() const {
        return groundtruth_file(info_.base_count);
    }
    
    // Ground truth file for half dataset
    std::string groundtruth_file_half() const {
        return groundtruth_file(info_.base_count / 2);
    }
    
    // ========== Data Loading Methods ==========
    
    /**
     * @brief Load the base feature repository.
     */
    deglib::StaticFeatureRepository load_base() const {
        return deglib::load_static_repository(base_file().c_str());
    }
    
    /**
     * @brief Load the query feature repository.
     */
    deglib::StaticFeatureRepository load_query() const {
        return deglib::load_static_repository(query_file().c_str());
    }
    
    /**
     * @brief Load ground truth and convert to vector of unordered_sets.
     * 
     * @param k Number of nearest neighbors to include in each set
     * @param use_half_dataset Whether to use half dataset ground truth
     * @return Vector of unordered_sets, one per query
     */
    std::vector<std::unordered_set<uint32_t>> load_groundtruth(size_t k, bool use_half_dataset = false) const {
        std::string gt_file = use_half_dataset ? groundtruth_file_half() : groundtruth_file_full();
        
        size_t ground_truth_dims = 0;
        size_t ground_truth_size = 0;
        auto gt_data = deglib::fvecs_read(gt_file.c_str(), ground_truth_dims, ground_truth_size);
        const uint32_t* ground_truth = reinterpret_cast<const uint32_t*>(gt_data.get());
        
        // Does the ground truth data provide enough top elements to check for k elements?
        if (ground_truth_dims < k) {
            fmt::print(stderr, "Ground truth data has only {} elements but need {}\n", ground_truth_dims, k);
            abort();
        }

        auto answers = std::vector<std::unordered_set<uint32_t>>(ground_truth_size);
        for (size_t i = 0; i < ground_truth_size; i++) {
            auto& gt = answers[i];
            gt.reserve(k);
            for (size_t j = 0; j < k; j++) {
                gt.insert(ground_truth[ground_truth_dims * i + j]);
            }
        }

        return answers;
    }
    
    /**
     * @brief Load exploration entry vertex IDs.
     * @return Vector of entry vertex IDs (external labels)
     */
    std::vector<uint32_t> load_explore_entry_vertices() const {
        size_t dims = 0, count = 0;
        auto data = deglib::fvecs_read(explore_entry_vertex_file().c_str(), dims, count);
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(data.get());
        
        // Entry vertices are stored as ivecs with dims=1, so total count is count
        std::vector<uint32_t> entry_vertices(count);
        for (size_t i = 0; i < count; i++) {
            entry_vertices[i] = ptr[i * dims];  // Each entry has 'dims' elements, take first
        }
        return entry_vertices;
    }
    
    /**
     * @brief Load exploration ground truth and convert to vector of unordered_sets.
     * 
     * @param k Number of nearest neighbors to include in each set (default: EXPLORE_TOPK = 1000)
     * @return Vector of unordered_sets, one per entry vertex
     */
    std::vector<std::unordered_set<uint32_t>> load_explore_groundtruth(size_t k = DatasetInfo::EXPLORE_TOPK) const {
        size_t dims = 0, count = 0;
        auto data = deglib::fvecs_read(explore_groundtruth_file().c_str(), dims, count);
        const uint32_t* gt_ptr = reinterpret_cast<const uint32_t*>(data.get());
        
        // Clamp k to available dimensions
        size_t actual_k = std::min(k, dims);
        
        std::vector<std::unordered_set<uint32_t>> answers(count);
        for (size_t i = 0; i < count; i++) {
            auto& gt = answers[i];
            gt.reserve(actual_k);
            for (size_t j = 0; j < actual_k; j++) {
                gt.insert(gt_ptr[dims * i + j]);
            }
        }
        return answers;
    }
    
    /**
     * @brief Load full exploration ground truth (for all base elements) as vector of unordered_sets.
     * 
     * This ground truth contains the top-k nearest neighbors for EVERY element in the base dataset.
     * Used for computing graph quality metrics.
     * 
     * @param k Number of nearest neighbors to include in each set (default: EXPLORE_TOPK = 1000)
     * @return Vector of unordered_sets, one per base element
     */
    std::vector<std::unordered_set<uint32_t>> load_full_explore_groundtruth(size_t k = DatasetInfo::EXPLORE_TOPK) const {
        size_t dims = 0, count = 0;
        auto data = deglib::fvecs_read(full_explore_groundtruth_file().c_str(), dims, count);
        const uint32_t* gt_ptr = reinterpret_cast<const uint32_t*>(data.get());
        
        // Clamp k to available dimensions
        size_t actual_k = std::min(k, dims);
        
        std::vector<std::unordered_set<uint32_t>> answers(count);
        for (size_t i = 0; i < count; i++) {
            auto& gt = answers[i];
            gt.reserve(actual_k);
            for (size_t j = 0; j < actual_k; j++) {
                gt.insert(gt_ptr[dims * i + j]);
            }
        }
        return answers;
    }
    
    // ========== Comparison ==========
    
    bool operator==(const Dataset& other) const { 
        return name_ == other.name_ && data_root_ == other.data_root_; 
    }
    bool operator!=(const Dataset& other) const { return !(*this == other); }
    
private:
    DatasetName name_;
    std::filesystem::path data_root_;
    std::filesystem::path dataset_dir_;
    std::filesystem::path files_dir_;
    DatasetInfo info_;
};


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
inline bool setup_sift1m_files(const Dataset& ds) {
    fmt::print("\n=== Setting up SIFT1M dataset ===\n");
    
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "sift.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";
    
    // Ensure directories exist
    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());
    
    // Check if already setup (base file exists with correct name)
    if (file_exists(ds.base_file())) {
        fmt::print("SIFT1M already set up at {}\n", ds.files_dir().string());
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
    move_file(extracted_dir / "sift_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "sift_query.fvecs", ds.files_dir() / info.query_file);
    move_file(extracted_dir / "sift_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "sift_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);
    // Rename explore ground truth (different naming convention)
    move_file(extracted_dir / "sift_explore_ground_truth.ivecs", ds.files_dir() / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("SIFT1M files set up in: {}\n", ds.files_dir().string());
    return true;
}

/**
 * Setup DEEP1M dataset
 */
inline bool setup_deep1m_files(const Dataset& ds) {
    fmt::print("\n=== Setting up DEEP1M dataset ===\n");
    
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "deep1m.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";
    
    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());
    
    if (file_exists(ds.base_file())) {
        fmt::print("DEEP1M already set up at {}\n", ds.files_dir().string());
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
    move_file(extracted_dir / "deep1m_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "deep1m_query.fvecs", ds.files_dir() / info.query_file);
    
    // Handle exploration files if they exist (with possible different naming)
    move_file(extracted_dir / "deep1m_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "deep1m_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);
    move_file(extracted_dir / "deep1m_explore_ground_truth.ivecs", ds.files_dir() / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("DEEP1M files set up in: {}\n", ds.files_dir().string());
    return true;
}

/**
 * Setup GLOVE dataset
 */
inline bool setup_glove_files(const Dataset& ds) {
    fmt::print("\n=== Setting up GLOVE dataset ===\n");
    
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "glove.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";
    
    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());
    
    if (file_exists(ds.base_file())) {
        fmt::print("GLOVE already set up at {}\n", ds.files_dir().string());
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
    move_file(extracted_dir / "glove-100_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "glove-100_query.fvecs", ds.files_dir() / info.query_file);
    
    // Handle exploration files if they exist
    move_file(extracted_dir / "glove-100_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "glove_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "glove-100_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);
    move_file(extracted_dir / "glove_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);
    move_file(extracted_dir / "glove-100_explore_ground_truth.ivecs", ds.files_dir() / info.explore_groundtruth_file);
    move_file(extracted_dir / "glove_explore_ground_truth.ivecs", ds.files_dir() / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("GLOVE files set up in: {}\n", ds.files_dir().string());
    return true;
}

/**
 * Setup AUDIO dataset
 */
inline bool setup_audio_files(const Dataset& ds) {
    fmt::print("\n=== Setting up AUDIO dataset ===\n");
    
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "audio.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";
    
    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());
    
    if (file_exists(ds.base_file())) {
        fmt::print("AUDIO already set up at {}\n", ds.files_dir().string());
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
    move_file(extracted_dir / "audio_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "audio_query.fvecs", ds.files_dir() / info.query_file);
    
    // Handle exploration files if they exist
    move_file(extracted_dir / "audio_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "audio_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);
    move_file(extracted_dir / "audio_explore_ground_truth.ivecs", ds.files_dir() / info.explore_groundtruth_file);
    
    // Clean up tmp directory completely
    remove_directory(tmp_dir);
    
    fmt::print("AUDIO files set up in: {}\n", ds.files_dir().string());
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
    const Dataset& ds,
    const deglib::FeatureRepository& base_repo,
    const uint32_t thread_count = 4)
{
    const auto& info = ds.info();
    const size_t base_size = base_repo.size();
    const size_t sample_count = DatasetInfo::EXPLORE_SAMPLE_COUNT;
    const uint32_t topk = DatasetInfo::EXPLORE_TOPK;
    const uint32_t dims = (uint32_t)base_repo.dims();
    
    // Check if files already exist
    if (file_exists(ds.explore_query_file()) &&
        file_exists(ds.explore_entry_vertex_file()) &&
        file_exists(ds.explore_groundtruth_file())) {
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
    ivecs_write(ds.explore_entry_vertex_file().c_str(), 1, sample_count, entry_ids.data());
    fmt::print("Wrote: {}\n", ds.explore_entry_vertex_file());
    
    // Write entry features
    fvecs_write(ds.explore_query_file().c_str(), dims, sample_count, entry_features.data());
    fmt::print("Wrote: {}\n", ds.explore_query_file());
    
    // Create a temporary repository for queries
    auto query_features = std::make_unique<std::byte[]>(sample_count * dims * sizeof(float));
    std::memcpy(query_features.get(), entry_features.data(), sample_count * dims * sizeof(float));
    deglib::StaticFeatureRepository query_repo(std::move(query_features), dims, sample_count, sizeof(float));
    
    // Compute ground truth for exploration (top-1000 from base for each entry)
    fmt::print("Computing exploration ground truth (this may take a while)...\n");
    auto groundtruth = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, 0, thread_count);
    
    // Write ground truth
    ivecs_write(ds.explore_groundtruth_file().c_str(), topk, sample_count, groundtruth.data());
    fmt::print("Wrote: {}\n", ds.explore_groundtruth_file());
    
    return true;
}

// ============================================================================
// Full Exploration Ground Truth Generation
// ============================================================================

/**
 * Generate full exploration ground truth for ALL base elements.
 * Creates:
 * - full_explore_groundtruth_top1000.ivecs: Top-1000 nearest neighbors for EVERY base element
 * 
 * This is used for computing graph quality metrics where we need the ideal neighbors
 * for every vertex in the graph, not just a sampled subset.
 * 
 * WARNING: This is very expensive for large datasets (O(n^2) distance computations).
 */
inline bool generate_full_exploration_groundtruth(
    const Dataset& ds,
    const deglib::FeatureRepository& base_repo,
    const uint32_t thread_count = 4)
{
    const auto& info = ds.info();
    const size_t base_size = base_repo.size();
    const uint32_t topk = DatasetInfo::EXPLORE_TOPK;
    
    // Check if file already exists
    if (file_exists(ds.full_explore_groundtruth_file())) {
        fmt::print("Full exploration ground truth already exists: {}\n", ds.full_explore_groundtruth_file());
        return true;
    }
    
    fmt::print("\n=== Generating full exploration ground truth ===\n");
    fmt::print("Base size: {}, TopK: {}, Threads: {}\n", base_size, topk, thread_count);
    fmt::print("WARNING: This computes top-{} for ALL {} elements (expensive)...\n", topk, base_size);
    
    // Compute ground truth: for each base element, find its top-k nearest neighbors in the base set
    // This is essentially computing KNN with base_repo as both query and base
    auto groundtruth = compute_knn_groundtruth(base_repo, base_repo, info.metric, topk, 0, thread_count);
    
    // Write ground truth
    ivecs_write(ds.full_explore_groundtruth_file().c_str(), topk, base_size, groundtruth.data());
    fmt::print("Wrote: {}\n", ds.full_explore_groundtruth_file());
    
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
    const Dataset& ds,
    const deglib::FeatureRepository& base_repo,
    const deglib::FeatureRepository& query_repo,
    const uint32_t thread_count = 4)
{
    const auto& info = ds.info();
    const size_t base_size = base_repo.size();
    const uint32_t topk = DatasetInfo::GROUNDTRUTH_TOPK;
    const size_t step = DatasetInfo::GROUNDTRUTH_STEP;
    
    fmt::print("\n=== Generating ground truth files ===\n");
    fmt::print("Base size: {}, Query size: {}, TopK: {}, Step: {}\n", 
               base_size, query_repo.size(), topk, step);
    
    // Generate for each step
    for (size_t nb = step; nb <= base_size; nb += step) {
        std::string gt_file = ds.groundtruth_file(nb);
        
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
        std::string gt_half_file = ds.groundtruth_file_half();
        
        if (!file_exists(gt_half_file)) {
            fmt::print("\nComputing ground truth for half (nb={}) ...\n", half_count);
            auto groundtruth = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, half_count, thread_count);
            
            ivecs_write(gt_half_file.c_str(), topk, query_repo.size(), groundtruth.data());
            fmt::print("Wrote: {}\n", gt_half_file);
        }
    }
    
    // Generate for full base (if not already covered by step)
    if (base_size % step != 0) {
        std::string gt_full_file = ds.groundtruth_file_full();
        
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
 * @param ds The dataset to set up (already contains data_root)
 * @param thread_count Number of threads for ground truth computation
 * @return true if successful
 */
inline bool setup_dataset(
    const Dataset& ds,
    const uint32_t thread_count = 4)
{
    if (!ds.is_valid()) {
        fmt::print(stderr, "Invalid dataset\n");
        return false;
    }
    
    fmt::print("\n============================================================\n");
    fmt::print("Setting up dataset: {}\n", ds.name());
    fmt::print("Data path: {}\n", ds.data_root().string());
    fmt::print("Dataset directory: {}\n", ds.dataset_dir().string());
    fmt::print("Files directory: {}\n", ds.files_dir().string());
    fmt::print("============================================================\n");
    
    // Step 1: Download and extract dataset files
    bool setup_ok = false;
    if (ds.dataset_name() == DatasetName::SIFT1M) {
        setup_ok = detail::setup_sift1m_files(ds);
    } else if (ds.dataset_name() == DatasetName::DEEP1M) {
        setup_ok = detail::setup_deep1m_files(ds);
    } else if (ds.dataset_name() == DatasetName::GLOVE) {
        setup_ok = detail::setup_glove_files(ds);
    } else if (ds.dataset_name() == DatasetName::AUDIO) {
        setup_ok = detail::setup_audio_files(ds);
    } else {
        fmt::print(stderr, "Unknown dataset\n");
        return false;
    }
    
    if (!setup_ok) {
        fmt::print(stderr, "Failed to set up dataset files\n");
        return false;
    }
    
    // Step 2: Verify required files exist
    if (!file_exists(ds.base_file())) {
        fmt::print(stderr, "Base file not found: {}\n", ds.base_file());
        return false;
    }
    if (!file_exists(ds.query_file())) {
        fmt::print(stderr, "Query file not found: {}\n", ds.query_file());
        return false;
    }
    
    fmt::print("\nLoading base repository...\n");
    auto base_repo = ds.load_base();
    fmt::print("Loaded {} vectors of dimension {}\n", base_repo.size(), base_repo.dims());
    
    fmt::print("\nLoading query repository...\n");
    auto query_repo = ds.load_query();
    fmt::print("Loaded {} vectors of dimension {}\n", query_repo.size(), query_repo.dims());
    
    // Step 3: Generate exploration files if missing (sampled subset)
    generate_exploration_files(ds, base_repo, thread_count);
    
    // Step 4: Generate ground truth files if missing
    generate_groundtruth_files(ds, base_repo, query_repo, thread_count);
    
    // Step 5: Generate full exploration ground truth if missing (all base elements)
    generate_full_exploration_groundtruth(ds, base_repo, thread_count);
    
    fmt::print("\n============================================================\n");
    fmt::print("Dataset {} setup complete!\n", ds.name());
    fmt::print("\nFiles:\n");
    fmt::print("  Base:     {}\n", ds.base_file());
    fmt::print("  Query:    {}\n", ds.query_file());
    fmt::print("  Explore Query:  {}\n", ds.explore_query_file());
    fmt::print("  Explore Entry:  {}\n", ds.explore_entry_vertex_file());
    fmt::print("  Explore GT:     {}\n", ds.explore_groundtruth_file());
    fmt::print("  Full Explore GT: {}\n", ds.full_explore_groundtruth_file());
    fmt::print("  GT Full:  {}\n", ds.groundtruth_file_full());
    fmt::print("  GT Half:  {}\n", ds.groundtruth_file_half());
    fmt::print("============================================================\n\n");
    
    return true;
}

/**
 * Convenience overload taking DatasetName and data_root separately.
 */
inline bool setup_dataset(
    const DatasetName& name,
    const std::filesystem::path& data_root,
    const uint32_t thread_count = 4)
{
    Dataset ds(name, data_root);
    return setup_dataset(ds, thread_count);
}

/**
 * Convenience function to set up all datasets.
 */
inline bool setup_all_datasets(
    const std::filesystem::path& data_root,
    const uint32_t thread_count = 4)
{
    bool all_ok = true;
    
    for (const auto& name : DatasetName::all()) {
        Dataset ds(name, data_root);
        if (!setup_dataset(ds, thread_count)) {
            fmt::print(stderr, "Failed to set up {}\n", ds.name());
            all_ok = false;
        }
    }
    
    return all_ok;
}

} // namespace deglib::benchmark
