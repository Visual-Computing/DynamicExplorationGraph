#pragma once

/**
 * @file dataset.h
 * @brief Dataset management utilities for deglib benchmarks.
 */

#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "deglib.h"
#include "file_io.h"

namespace deglib::benchmark {

struct DatasetInfo;
class Dataset;

class DatasetName {
public:
    static const DatasetName SIFT1M;
    static const DatasetName DEEP1M;
    static const DatasetName GLOVE;
    static const DatasetName AUDIO;
    static const DatasetName ENRON;
    static const DatasetName ALL;
    static const DatasetName Invalid;

    static const std::array<DatasetName, 5>& all() {
        static const std::array<DatasetName, 5> datasets = {AUDIO, ENRON, SIFT1M, DEEP1M, GLOVE};
        return datasets;
    }

    static DatasetName from_string(const std::string& str) {
        std::string lower = str;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower == "all") return ALL;

        for (const auto& ds : all()) {
            if (lower == ds.name()) return ds;
        }
        return Invalid;
    }

    const char* name() const { return name_; }
    bool is_valid() const { return name_ != Invalid.name_; }
    const char* to_string() const { return name_; }

    DatasetInfo info() const;

    bool operator==(const DatasetName& other) const { return name_ == other.name_; }
    bool operator!=(const DatasetName& other) const { return name_ != other.name_; }

private:
    constexpr DatasetName(const char* name) : name_(name) {}
    const char* name_;
};

inline constexpr DatasetName DatasetName::SIFT1M{"sift1m"};
inline constexpr DatasetName DatasetName::DEEP1M{"deep1m"};
inline constexpr DatasetName DatasetName::GLOVE{"glove"};
inline constexpr DatasetName DatasetName::AUDIO{"audio"};
inline constexpr DatasetName DatasetName::ENRON{"enron"};
inline constexpr DatasetName DatasetName::ALL{"all"};
inline constexpr DatasetName DatasetName::Invalid{"invalid"};

struct DatasetInfo {
    DatasetName dataset_name;
    std::string download_url;
    deglib::Metric metric;
    size_t base_count;
    size_t query_count;
    uint32_t dims;
    uint32_t scale;
    uint32_t explore_depth;

    std::string base_file;
    std::string query_file;

    std::string explore_query_file;
    std::string explore_entry_vertex_file;
    std::string explore_groundtruth_file;
    std::string explore_groundtruth_half_file;

    static constexpr size_t EXPLORE_SAMPLE_COUNT = 10000;
    static constexpr uint32_t EXPLORE_TOPK = 1000;
    static constexpr uint32_t GROUNDTRUTH_TOPK = 100;

    const char* name() const { return dataset_name.name(); }
};

inline DatasetInfo make_dataset_info(const DatasetName& ds) {
    DatasetInfo info{ds, {}, deglib::Metric::L2, 0, 0, 0, 1, 2, {}, {}, {}, {}, {}, {}};

    std::string name = ds.name();

    info.base_file = name + "_base.fvecs";
    info.query_file = name + "_query.fvecs";
    info.explore_query_file = name + "_explore_query.fvecs";
    info.explore_entry_vertex_file = name + "_explore_entry_vertex.ivecs";
    info.explore_groundtruth_file = name + "_explore_groundtruth_top1000.ivecs";
    info.explore_groundtruth_half_file = name + "_explore_groundtruth_half_top1000.ivecs";

    if (ds == DatasetName::SIFT1M) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/sift.tar.gz";
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 128;
        info.scale = 1;
        info.explore_depth = 2;
    } else if (ds == DatasetName::DEEP1M) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/deep1m.tar.gz";
        info.base_count = 1000000;
        info.query_count = 10000;
        info.dims = 96;
        info.scale = 100;
        info.explore_depth = 2;
    } else if (ds == DatasetName::GLOVE) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/glove-100.tar.gz";
        info.base_count = 1183514;
        info.query_count = 10000;
        info.dims = 100;
        info.scale = 100;
        info.explore_depth = 2;
    } else if (ds == DatasetName::AUDIO) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/audio.tar.gz";
        info.base_count = 53387;
        info.query_count = 200;
        info.dims = 192;
        info.scale = 1;
        info.explore_depth = 1;
    } else if (ds == DatasetName::ENRON) {
        info.download_url = "https://static.visual-computing.com/paper/DEG/enron.tar.gz";
        info.base_count = 94987;
        info.query_count = 200;
        info.dims = 1369;
        info.scale = 1;
        info.explore_depth = 1;
    }

    return info;
}

inline DatasetInfo DatasetName::info() const {
    return make_dataset_info(*this);
}

class Dataset {
public:
    Dataset(const DatasetName& name, const std::filesystem::path& data_root)
        : name_(name),
          data_root_(data_root),
          dataset_dir_(data_root / name.name()),
          files_dir_(data_root / name.name() / name.name()),
          info_(name.info()) {}

    const DatasetName& dataset_name() const { return name_; }
    const char* name() const { return name_.name(); }
    bool is_valid() const { return name_.is_valid(); }
    const DatasetInfo& info() const { return info_; }

    const std::filesystem::path& data_root() const { return data_root_; }
    const std::filesystem::path& dataset_dir() const { return dataset_dir_; }
    const std::filesystem::path& files_dir() const { return files_dir_; }

    std::string base_file() const { return (files_dir_ / info_.base_file).string(); }
    std::string query_file() const { return (files_dir_ / info_.query_file).string(); }
    std::string explore_query_file() const { return (files_dir_ / info_.explore_query_file).string(); }
    std::string explore_entry_vertex_file() const { return (files_dir_ / info_.explore_entry_vertex_file).string(); }
    std::string explore_groundtruth_file() const { return (files_dir_ / info_.explore_groundtruth_file).string(); }
    std::string explore_groundtruth_half_file() const { return (files_dir_ / info_.explore_groundtruth_half_file).string(); }

    std::string groundtruth_file(size_t nb) const {
        return (files_dir_ / fmt::format("{}_groundtruth_top{}_nb{}.ivecs", info_.name(), DatasetInfo::GROUNDTRUTH_TOPK, nb)).string();
    }
    std::string groundtruth_file_full() const { return groundtruth_file(info_.base_count); }
    std::string groundtruth_file_half() const { return groundtruth_file(info_.base_count / 2); }

    deglib::StaticFeatureRepository load_base() const { return deglib::load_static_repository(base_file().c_str()); }
    deglib::StaticFeatureRepository load_query() const { return deglib::load_static_repository(query_file().c_str()); }

    std::vector<std::vector<uint32_t>> load_groundtruth(size_t k, bool use_half_dataset = false) const {
        std::string gt_file = use_half_dataset ? groundtruth_file_half() : groundtruth_file_full();
        return load_groundtruth_from_file(gt_file, k);
    }

private:
    std::vector<std::vector<uint32_t>> load_groundtruth_from_file(const std::string& gt_file, size_t k) const {
        size_t ground_truth_dims = 0;
        size_t ground_truth_size = 0;
        auto gt_data = deglib::fvecs_read(gt_file.c_str(), ground_truth_dims, ground_truth_size);
        const uint32_t* ground_truth = reinterpret_cast<const uint32_t*>(gt_data.get());

        if (ground_truth_dims < k) {
            fmt::print(stderr, "Ground truth data has only {} elements but need {}\n", ground_truth_dims, k);
            abort();
        }

        auto answers = std::vector<std::vector<uint32_t>>(ground_truth_size);
        for (size_t i = 0; i < ground_truth_size; i++) {
            auto& gt = answers[i];
            gt.resize(k);
            for (size_t j = 0; j < k; j++) {
                gt[j] = ground_truth[ground_truth_dims * i + j];
            }
            std::sort(gt.begin(), gt.end());
        }

        return answers;
    }

public:
    std::vector<uint32_t> load_explore_entry_vertices() const {
        size_t dims = 0, count = 0;
        auto data = deglib::fvecs_read(explore_entry_vertex_file().c_str(), dims, count);
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(data.get());

        std::vector<uint32_t> entry_vertices(count);
        for (size_t i = 0; i < count; i++) {
            entry_vertices[i] = ptr[i * dims];
        }
        return entry_vertices;
    }

    std::vector<std::vector<uint32_t>> load_explore_groundtruth(size_t k = DatasetInfo::EXPLORE_TOPK, bool use_half_dataset = false) const {
        std::string gt_file = use_half_dataset ? explore_groundtruth_half_file() : explore_groundtruth_file();

        size_t dims = 0, count = 0;
        auto data = deglib::fvecs_read(gt_file.c_str(), dims, count);
        const uint32_t* gt_ptr = reinterpret_cast<const uint32_t*>(data.get());

        size_t actual_k = std::min(k, dims);

        std::vector<std::vector<uint32_t>> answers(count);
        for (size_t i = 0; i < count; i++) {
            auto& gt = answers[i];
            gt.resize(actual_k);
            for (size_t j = 0; j < actual_k; j++) {
                gt[j] = gt_ptr[dims * i + j];
            }
            std::sort(gt.begin(), gt.end());
        }
        return answers;
    }

private:
    DatasetName name_;
    std::filesystem::path data_root_;
    std::filesystem::path dataset_dir_;
    std::filesystem::path files_dir_;
    DatasetInfo info_;
};

inline std::vector<uint32_t> compute_knn_groundtruth(const deglib::FeatureRepository& base_repo,
                                                     const deglib::FeatureRepository& query_repo,
                                                     const deglib::Metric metric,
                                                     const uint32_t k_target,
                                                     const size_t base_limit = 0,
                                                     const uint32_t thread_count = 1) {
    const auto base_size = base_limit > 0 ? (uint32_t)std::min(base_limit, base_repo.size()) : (uint32_t)base_repo.size();
    const auto query_size = (uint32_t)query_repo.size();
    const auto dims = base_repo.dims();

    const auto feature_space = deglib::FloatSpace((uint32_t)dims, metric);
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
            const auto duration_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            fmt::print("  Ground truth progress: {}/{} queries ({:.1f}%) after {}ms\n",
                       count,
                       query_size,
                       100.0f * count / query_size,
                       duration_ms);
        }
    });

    return topLists;
}

namespace detail {

inline bool setup_sift1m_files(const Dataset& ds) {
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "sift.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";

    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());

    if (file_exists(ds.base_file())) return true;

    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) return false;
    }

    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);

    if (!extract_tar_gz(archive_file, tmp_dir)) {
        remove_directory(tmp_dir);
        return false;
    }

    auto extracted_dir = find_directory_with_file(tmp_dir, "sift_base.fvecs");
    if (extracted_dir.empty()) {
        remove_directory(tmp_dir);
        return false;
    }

    move_file(extracted_dir / "sift_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "sift_query.fvecs", ds.files_dir() / info.query_file);
    move_file(extracted_dir / "sift_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "sift_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);

    remove_directory(tmp_dir);
    return true;
}

inline bool setup_deep1m_files(const Dataset& ds) {
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "deep1m.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";

    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());

    if (file_exists(ds.base_file())) return true;

    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) return false;
    }

    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);

    if (!extract_tar_gz(archive_file, tmp_dir)) {
        remove_directory(tmp_dir);
        return false;
    }

    auto extracted_dir = find_directory_with_file(tmp_dir, "deep1m_base.fvecs");
    if (extracted_dir.empty()) {
        remove_directory(tmp_dir);
        return false;
    }

    move_file(extracted_dir / "deep1m_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "deep1m_query.fvecs", ds.files_dir() / info.query_file);
    move_file(extracted_dir / "deep1m_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "deep1m_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);

    remove_directory(tmp_dir);
    return true;
}

inline bool setup_glove_files(const Dataset& ds) {
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "glove.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";

    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());

    if (file_exists(ds.base_file())) return true;

    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) return false;
    }

    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);

    if (!extract_tar_gz(archive_file, tmp_dir)) {
        remove_directory(tmp_dir);
        return false;
    }

    auto extracted_dir = find_directory_with_file(tmp_dir, "glove_base.fvecs");
    if (extracted_dir.empty()) {
        remove_directory(tmp_dir);
        return false;
    }

    move_file(extracted_dir / "glove_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "glove_query.fvecs", ds.files_dir() / info.query_file);
    move_file(extracted_dir / "glove_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "glove_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);

    remove_directory(tmp_dir);
    return true;
}

inline bool setup_audio_files(const Dataset& ds) {
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "audio.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";

    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());

    if (file_exists(ds.base_file())) return true;

    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) return false;
    }

    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);

    if (!extract_tar_gz(archive_file, tmp_dir)) {
        remove_directory(tmp_dir);
        return false;
    }

    auto extracted_dir = find_directory_with_file(tmp_dir, "audio_base.fvecs");
    if (extracted_dir.empty()) {
        remove_directory(tmp_dir);
        return false;
    }

    move_file(extracted_dir / "audio_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "audio_query.fvecs", ds.files_dir() / info.query_file);
    move_file(extracted_dir / "audio_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "audio_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);

    remove_directory(tmp_dir);
    return true;
}

inline bool setup_enron_files(const Dataset& ds) {
    const auto& info = ds.info();
    const auto archive_file = ds.dataset_dir() / "enron.tar.gz";
    const auto tmp_dir = ds.dataset_dir() / "_tmp_extract";

    ensure_directory(ds.dataset_dir());
    ensure_directory(ds.files_dir());

    if (file_exists(ds.base_file())) return true;

    if (!file_exists(archive_file)) {
        if (!download_file(info.download_url, archive_file)) return false;
    }

    remove_directory(tmp_dir);
    ensure_directory(tmp_dir);

    if (!extract_tar_gz(archive_file, tmp_dir)) {
        remove_directory(tmp_dir);
        return false;
    }

    auto extracted_dir = find_directory_with_file(tmp_dir, "enron_base.fvecs");
    if (extracted_dir.empty()) {
        remove_directory(tmp_dir);
        return false;
    }

    move_file(extracted_dir / "enron_base.fvecs", ds.files_dir() / info.base_file);
    move_file(extracted_dir / "enron_query.fvecs", ds.files_dir() / info.query_file);
    move_file(extracted_dir / "enron_explore_query.fvecs", ds.files_dir() / info.explore_query_file);
    move_file(extracted_dir / "enron_explore_entry_vertex.ivecs", ds.files_dir() / info.explore_entry_vertex_file);

    remove_directory(tmp_dir);
    return true;
}

}  // namespace detail

inline bool generate_exploration_files(const Dataset& ds, const deglib::FeatureRepository& base_repo, const uint32_t thread_count = 4, bool include_half = true) {
    const auto& info = ds.info();
    const size_t base_size = base_repo.size();
    const size_t half_base_size = base_size / 2;
    const size_t sample_count = DatasetInfo::EXPLORE_SAMPLE_COUNT;
    const uint32_t topk = DatasetInfo::EXPLORE_TOPK;
    const uint32_t dims = (uint32_t)base_repo.dims();

    bool all_exist = file_exists(ds.explore_query_file()) && file_exists(ds.explore_entry_vertex_file()) &&
                     file_exists(ds.explore_groundtruth_file()) && (!include_half || file_exists(ds.explore_groundtruth_half_file()));

    if (all_exist) return true;

    std::vector<uint32_t> entry_ids;
    std::vector<float> entry_features;
    size_t actual_sample_count = sample_count;

    bool entry_ids_exist = file_exists(ds.explore_entry_vertex_file());
    bool entry_features_exist = file_exists(ds.explore_query_file());

    if (entry_ids_exist) {
        size_t id_dims = 0, id_count = 0;
        auto id_data = deglib::fvecs_read(ds.explore_entry_vertex_file().c_str(), id_dims, id_count);
        const uint32_t* id_ptr = reinterpret_cast<const uint32_t*>(id_data.get());
        entry_ids.resize(id_count);
        for (size_t i = 0; i < id_count; i++) {
            entry_ids[i] = id_ptr[i * id_dims];
        }
        actual_sample_count = id_count;
    } else {
        entry_ids.resize(sample_count);
        const double step = static_cast<double>(half_base_size) / sample_count;
        for (size_t i = 0; i < sample_count; i++) {
            uint32_t idx = static_cast<uint32_t>(i * step);
            if (idx >= half_base_size) idx = (uint32_t)half_base_size - 1;
            entry_ids[i] = idx;
        }
        ivecs_write(ds.explore_entry_vertex_file().c_str(), 1, sample_count, entry_ids.data());
    }

    if (entry_features_exist) {
        size_t feat_dims = 0, feat_count = 0;
        auto feat_data = deglib::fvecs_read(ds.explore_query_file().c_str(), feat_dims, feat_count);
        const float* feat_ptr = reinterpret_cast<const float*>(feat_data.get());
        entry_features.resize(feat_count * feat_dims);
        std::copy(feat_ptr, feat_ptr + feat_count * feat_dims, entry_features.data());
    } else {
        entry_features.resize(actual_sample_count * dims);
        for (size_t i = 0; i < actual_sample_count; i++) {
            const float* src = reinterpret_cast<const float*>(base_repo.getFeature(entry_ids[i]));
            std::copy(src, src + dims, entry_features.data() + i * dims);
        }
        fvecs_write(ds.explore_query_file().c_str(), dims, actual_sample_count, entry_features.data());
    }

    auto feature_bytes = std::make_unique<std::byte[]>(entry_features.size() * sizeof(float));
    std::memcpy(feature_bytes.get(), entry_features.data(), entry_features.size() * sizeof(float));
    deglib::StaticFeatureRepository explore_query_repo(std::move(feature_bytes), dims, actual_sample_count, sizeof(float));

    if (!file_exists(ds.explore_groundtruth_file())) {
        fmt::print("\nGenerating Exploration Ground Truth (Full Dataset)...\n");
        auto gt = compute_knn_groundtruth(base_repo, explore_query_repo, info.metric, topk, 0, thread_count);
        ivecs_write(ds.explore_groundtruth_file().c_str(), topk, actual_sample_count, gt.data());
    }

    if (include_half && !file_exists(ds.explore_groundtruth_half_file())) {
        fmt::print("\nGenerating Exploration Ground Truth (Half Dataset)...\n");
        auto gt_half = compute_knn_groundtruth(base_repo, explore_query_repo, info.metric, topk, half_base_size, thread_count);
        ivecs_write(ds.explore_groundtruth_half_file().c_str(), topk, actual_sample_count, gt_half.data());
    }

    return true;
}

inline bool generate_anns_groundtruth_files(const Dataset& ds, const deglib::FeatureRepository& base_repo, const uint32_t thread_count = 4, bool include_half = true) {
    const auto& info = ds.info();
    const size_t base_size = base_repo.size();
    const size_t half_base_size = base_size / 2;
    const uint32_t topk = DatasetInfo::GROUNDTRUTH_TOPK;

    bool full_exists = file_exists(ds.groundtruth_file_full());
    bool half_exists = file_exists(ds.groundtruth_file_half());

    if (full_exists && (!include_half || half_exists)) return true;

    auto query_repo = ds.load_query();

    if (!full_exists) {
        fmt::print("\nGenerating ANNS Ground Truth (Full Dataset)...\n");
        auto gt = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, 0, thread_count);
        ivecs_write(ds.groundtruth_file_full().c_str(), topk, query_repo.size(), gt.data());
    }

    if (include_half && !half_exists) {
        fmt::print("\nGenerating ANNS Ground Truth (Half Dataset)...\n");
        auto gt_half = compute_knn_groundtruth(base_repo, query_repo, info.metric, topk, half_base_size, thread_count);
        ivecs_write(ds.groundtruth_file_half().c_str(), topk, query_repo.size(), gt_half.data());
    }

    return true;
}

inline bool setup_dataset(const Dataset& ds, const uint32_t thread_count = 4, bool include_half = true, bool include_exploration = true) {
    bool ok = false;
    if (ds.dataset_name() == DatasetName::SIFT1M) {
        ok = detail::setup_sift1m_files(ds);
    } else if (ds.dataset_name() == DatasetName::DEEP1M) {
        ok = detail::setup_deep1m_files(ds);
    } else if (ds.dataset_name() == DatasetName::GLOVE) {
        ok = detail::setup_glove_files(ds);
    } else if (ds.dataset_name() == DatasetName::AUDIO) {
        ok = detail::setup_audio_files(ds);
    } else if (ds.dataset_name() == DatasetName::ENRON) {
        ok = detail::setup_enron_files(ds);
    }

    if (!ok) return false;

    auto base_repo = ds.load_base();

    if (!generate_anns_groundtruth_files(ds, base_repo, thread_count, include_half)) return false;
    if (include_exploration) {
        if (!generate_exploration_files(ds, base_repo, thread_count, include_half)) return false;
    }

    return true;
}

}  // namespace deglib::benchmark
