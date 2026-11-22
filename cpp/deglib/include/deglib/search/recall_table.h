#pragma once

#include "deglib/concurrent.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/utils/memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iomanip>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deglib::search {

/**
 * Calibrated fixed-beam entry mapping a target recall to an `ef` parameter.
 */
struct EfRecallEntry {
    float target_recall = 0.0f;
    uint32_t ef = 0;
    float measured_recall = 0.0f;
};

/**
 * Calibrated exploration-margin entry mapping a target recall to an `eps` parameter.
 */
struct EpsRecallEntry {
    float target_recall = 0.0f;
    float eps = 0.0f;
    float measured_recall = 0.0f;
};

class RecallTable;

namespace detail {
RecallTable build_recall_table_impl(
    const deglib::graph::InternalGraph& graph,
    uint32_t k,
    std::span<const float> resolved_targets,
    size_t sample_count,
    size_t threads,
    double bf_budget_override_us = -1.0
);
}  // namespace detail

/**
 * Calibration table mapping target recall@k levels to graph search parameters (`ef` and `eps`).
 *
 * Produced by `DynamicExplorationGraph::to_recall_table()` / `deglib::search::build_recall_table()`.
 * Stores calibrated entries for achievable target recalls (replacing unreachable targets with
 * the achieved lower/upper boundary entries) as well as monotonic recall bounds and lookup
 * interpolation helpers.
 */
class RecallTable {
  public:
    static constexpr uint32_t kMinEf = 10;
    static constexpr uint32_t kMaxEf = 300;
    static constexpr float kMinEps = 0.01f;
    static constexpr float kMaxEps = 0.20f;
    static constexpr float kEpsResolution = 0.0001f;

    static constexpr size_t kDefaultSampleCount = 300;
    static constexpr float kDefaultMinRecall = 0.80f;
    static constexpr float kDefaultMaxRecall = 0.999f;
    static constexpr float kDefaultRecallStep = 0.005f;

    RecallTable() = default;

    [[nodiscard]] uint32_t k() const noexcept { return k_; }
    [[nodiscard]] size_t sample_count() const noexcept { return sample_count_; }

    [[nodiscard]] bool has_ef() const noexcept { return !ef_entries_.empty() || ef_recall_bounds_.has_value(); }
    [[nodiscard]] bool has_eps() const noexcept { return !eps_entries_.empty() || eps_recall_bounds_.has_value(); }

    [[nodiscard]] std::optional<std::pair<float, float>> ef_recall_bounds() const noexcept { return ef_recall_bounds_; }
    [[nodiscard]] std::optional<std::pair<float, float>> eps_recall_bounds() const noexcept { return eps_recall_bounds_; }

    [[nodiscard]] const std::vector<EfRecallEntry>& ef_entries() const noexcept { return ef_entries_; }
    [[nodiscard]] const std::vector<EpsRecallEntry>& eps_entries() const noexcept { return eps_entries_; }

    /**
     * Returns deduplicated `ef` values from calibrated entries in ascending order.
     */
    [[nodiscard]] std::vector<uint32_t> ef_values() const {
        std::vector<uint32_t> vals;
        vals.reserve(ef_entries_.size());
        for (const auto& entry : ef_entries_) {
            vals.push_back(entry.ef);
        }
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        return vals;
    }

    /**
     * Returns deduplicated `eps` values from calibrated entries in ascending order.
     */
    [[nodiscard]] std::vector<float> eps_values() const {
        std::vector<float> vals;
        vals.reserve(eps_entries_.size());
        for (const auto& entry : eps_entries_) {
            vals.push_back(entry.eps);
        }
        std::sort(vals.begin(), vals.end());
        std::vector<float> dedup;
        dedup.reserve(vals.size());
        for (const float v : vals) {
            if (dedup.empty() || std::abs(v - dedup.back()) > 1e-6f) {
                dedup.push_back(v);
            }
        }
        return dedup;
    }

    /**
     * Returns deduplicated `ef` values (as floats) followed by deduplicated `eps` values,
     * directly usable with `eps_or_ef` search sweeps.
     */
    [[nodiscard]] std::vector<float> eps_or_ef_values() const {
        const auto efs = ef_values();
        const auto epss = eps_values();
        std::vector<float> combined;
        combined.reserve(efs.size() + epss.size());
        for (const uint32_t ef : efs) {
            combined.push_back(static_cast<float>(ef));
        }
        for (const float eps : epss) {
            combined.push_back(eps);
        }
        return combined;
    }

    /**
     * Looks up or linearly interpolates an `ef` parameter for `target_recall`.
     *
     * Returns `std::nullopt` if `ef` was not calibrated or `target_recall` lies outside the
     * calibrated bounds (beyond tolerance) unless `clamp == true`.
     */
    [[nodiscard]] std::optional<uint32_t> ef_for(float target_recall, bool clamp = false) const {
        return lookup_param<uint32_t>(
            target_recall, clamp, ef_recall_bounds_, ef_curve_, ef_entries_, [](const EfRecallEntry& e) { return e.ef; },
            [](uint32_t lo, uint32_t hi, float alpha) {
                const long long rounded =
                    static_cast<long long>(std::ceil(static_cast<double>(lo) + static_cast<double>(alpha) * static_cast<double>(hi - lo) - 1e-9));
                return std::clamp(static_cast<uint32_t>(rounded), lo, hi);
            }
        );
    }

    /**
     * Looks up or linearly interpolates an `eps` parameter for `target_recall`.
     *
     * Returns `std::nullopt` if `eps` was not calibrated or `target_recall` lies outside the
     * calibrated bounds (beyond tolerance) unless `clamp == true`.
     */
    [[nodiscard]] std::optional<float> eps_for(float target_recall, bool clamp = false) const {
        return lookup_param<float>(
            target_recall, clamp, eps_recall_bounds_, eps_curve_, eps_entries_, [](const EpsRecallEntry& e) { return e.eps; },
            [](float lo, float hi, float alpha) {
                const float raw = lo + alpha * (hi - lo);
                const float rounded = std::ceil(raw * 10000.0f - 1e-6f) / 10000.0f;
                return std::clamp(rounded, lo, hi);
            }
        );
    }

    /**
     * Formats a human-readable ASCII summary of calibrated recall bounds and entries.
     */
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << std::fixed;
        oss << "RecallTable(k=" << k_ << ", sample_count=" << sample_count_ << ")\n";

        if (!has_ef() && !has_eps()) {
            oss << "  <empty>";
            return oss.str();
        }

        if (has_ef()) {
            oss << "  ef";
            if (ef_recall_bounds_.has_value()) {
                oss << " [recall bounds: " << std::setprecision(4) << ef_recall_bounds_->first << " .. " << ef_recall_bounds_->second << "]";
            }
            oss << ":\n";
            if (ef_entries_.empty()) {
                oss << "    (no entries)\n";
            } else {
                oss << "    target_recall |   ef | measured_recall\n";
                oss << "    --------------+------+----------------\n";
                for (const auto& entry : ef_entries_) {
                    oss << "    " << std::setw(13) << std::setprecision(4) << entry.target_recall << " | " << std::setw(4) << entry.ef << " | " << std::setw(15)
                        << std::setprecision(4) << entry.measured_recall << "\n";
                }
            }
        }

        if (has_eps()) {
            oss << "  eps";
            if (eps_recall_bounds_.has_value()) {
                oss << " [recall bounds: " << std::setprecision(4) << eps_recall_bounds_->first << " .. " << eps_recall_bounds_->second << "]";
            }
            oss << ":\n";
            if (eps_entries_.empty()) {
                oss << "    (no entries)\n";
            } else {
                oss << "    target_recall |    eps | measured_recall\n";
                oss << "    --------------+--------+----------------\n";
                for (const auto& entry : eps_entries_) {
                    oss << "    " << std::setw(13) << std::setprecision(4) << entry.target_recall << " | " << std::setw(6) << std::setprecision(4) << entry.eps
                        << " | " << std::setw(15) << std::setprecision(4) << entry.measured_recall << "\n";
                }
            }
        }

        std::string out = oss.str();
        if (!out.empty() && out.back() == '\n') {
            out.pop_back();
        }
        return out;
    }

  private:
    template <typename ParamT, typename EntryT, typename GetParamFn, typename InterpFn>
    static std::optional<ParamT> lookup_param(
        float target_recall,
        bool clamp,
        const std::optional<std::pair<float, float>>& bounds,
        const std::vector<std::pair<ParamT, float>>& curve,
        const std::vector<EntryT>& entries,
        GetParamFn&& get_param,
        InterpFn&& interp_param
    ) {
        if (!bounds.has_value() || curve.empty() || !std::isfinite(target_recall)) {
            return std::nullopt;
        }

        const float r_min = bounds->first;
        const float r_max = bounds->second;
        const ParamT p_min = curve.front().first;

        // Determine the earliest safe parameter achieving r_max (matches upper-bound entry).
        ParamT p_max = curve.front().first;
        for (const auto& pt : curve) {
            if (pt.second >= r_max - 1e-6f) {
                p_max = pt.first;
                break;
            }
        }

        if (target_recall < r_min - 1e-6f) {
            return clamp ? std::optional<ParamT>(p_min) : std::nullopt;
        }
        if (target_recall > r_max + 1e-6f) {
            return clamp ? std::optional<ParamT>(p_max) : std::nullopt;
        }

        for (const auto& entry : entries) {
            if (std::abs(entry.target_recall - target_recall) <= 1e-6f) {
                return get_param(entry);
            }
        }

        if (target_recall <= r_min) {
            return p_min;
        }
        if (target_recall >= r_max) {
            return p_max;
        }

        // Build monotonic non-decreasing recall envelope along ascending parameter order.
        float prev_mono_r = curve.front().second;
        ParamT prev_p = curve.front().first;
        for (size_t i = 1; i < curve.size(); ++i) {
            const float cur_mono_r = (std::max)(prev_mono_r, curve[i].second);
            const ParamT cur_p = curve[i].first;
            if (cur_mono_r >= target_recall && cur_mono_r > prev_mono_r) {
                const float alpha = (target_recall - prev_mono_r) / (cur_mono_r - prev_mono_r);
                return interp_param(prev_p, cur_p, std::clamp(alpha, 0.0f, 1.0f));
            }
            prev_mono_r = cur_mono_r;
            prev_p = cur_p;
        }
        return p_max;
    }

    uint32_t k_ = 0;
    size_t sample_count_ = 0;
    std::optional<std::pair<float, float>> ef_recall_bounds_;
    std::optional<std::pair<float, float>> eps_recall_bounds_;
    std::vector<EfRecallEntry> ef_entries_;
    std::vector<EpsRecallEntry> eps_entries_;
    std::vector<std::pair<uint32_t, float>> ef_curve_;
    std::vector<std::pair<float, float>> eps_curve_;

    friend RecallTable detail::build_recall_table_impl(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        std::span<const float> resolved_targets,
        size_t sample_count,
        size_t threads,
        double bf_budget_override_us
    );
};

namespace detail {

inline void validate_common_recall_table_args(uint32_t k, size_t sample_count) {
    if (k == 0) {
        throw std::invalid_argument("to_recall_table: k must be >= 1");
    }
    if (sample_count == 0) {
        throw std::invalid_argument("to_recall_table: sample_count must be >= 1");
    }
}

inline std::vector<float> resolve_recall_targets_from_range(std::pair<float, float> recall_range, float recall_step) {
    const float min_r = recall_range.first;
    const float max_r = recall_range.second;
    if (!std::isfinite(min_r) || !std::isfinite(max_r) || min_r <= 0.0f || max_r > 1.0f || min_r > max_r) {
        throw std::invalid_argument(std::format("to_recall_table: recall_range ({}, {}) must satisfy 0.0 < min <= max <= 1.0", min_r, max_r));
    }
    if (!std::isfinite(recall_step) || recall_step <= 0.0f) {
        throw std::invalid_argument(std::format("to_recall_table: recall_step ({}) must be finite and > 0.0", recall_step));
    }

    std::vector<float> targets;
    const double min_d = static_cast<double>(min_r);
    const double max_d = static_cast<double>(max_r);
    const double step_d = static_cast<double>(recall_step);

    for (size_t i = 0;; ++i) {
        const double v = min_d + static_cast<double>(i) * step_d;
        if (v > max_d + 1e-6) {
            break;
        }
        targets.push_back(static_cast<float>((std::min)(v, max_d)));
    }
    if (targets.empty() || (max_d - static_cast<double>(targets.back())) > 1e-5) {
        targets.push_back(max_r);
    }
    return targets;
}

inline std::vector<float> resolve_explicit_recall_targets(std::span<const float> target_recalls) {
    if (target_recalls.empty()) {
        throw std::invalid_argument("to_recall_table: target_recalls must not be empty");
    }
    std::vector<float> targets;
    targets.reserve(target_recalls.size());
    for (const float r : target_recalls) {
        if (!std::isfinite(r) || r <= 0.0f || r > 1.0f) {
            throw std::invalid_argument(std::format("to_recall_table: each target recall ({}) must be finite in (0.0, 1.0]", r));
        }
        targets.push_back(r);
    }
    std::sort(targets.begin(), targets.end());
    std::vector<float> dedup;
    dedup.reserve(targets.size());
    for (const float r : targets) {
        if (dedup.empty() || (r - dedup.back()) > 1e-6f) {
            dedup.push_back(r);
        }
    }
    return dedup;
}

/**
 * Deterministically samples `q_count` unique internal vertex indices in `[0, n)`,
 * preferring vertices that are not entry points so searches do not start on the query itself.
 */
inline std::vector<uint32_t> sample_distinct_query_indices(const deglib::graph::InternalGraph& graph, size_t q_count, uint32_t n) {
    std::unordered_set<uint32_t> entry_set;
    for (const uint32_t ep : graph.getEntryVertexIndices()) {
        if (ep < n) {
            entry_set.insert(ep);
        }
    }

    std::vector<uint32_t> selected;
    selected.reserve(q_count);
    std::mt19937 rng(42u);

    const size_t non_entry_count = static_cast<size_t>(n) - entry_set.size();
    const size_t primary_target = (std::min)(q_count, non_entry_count);

    if (primary_target > 0) {
        if (primary_target * 4 >= non_entry_count) {
            std::vector<uint32_t> pool;
            pool.reserve(non_entry_count);
            for (uint32_t i = 0; i < n; ++i) {
                if (entry_set.find(i) == entry_set.end()) {
                    pool.push_back(i);
                }
            }
            for (size_t i = 0; i < primary_target; ++i) {
                std::uniform_int_distribution<size_t> dist(i, pool.size() - 1);
                const size_t j = dist(rng);
                std::swap(pool[i], pool[j]);
                selected.push_back(pool[i]);
            }
        } else {
            std::unordered_set<uint32_t> chosen;
            chosen.reserve(primary_target * 2);
            std::uniform_int_distribution<uint32_t> dist(0, n - 1);
            while (selected.size() < primary_target) {
                const uint32_t cand = dist(rng);
                if (entry_set.find(cand) == entry_set.end() && chosen.insert(cand).second) {
                    selected.push_back(cand);
                }
            }
        }
    }

    if (selected.size() < q_count) {
        std::vector<uint32_t> entries(entry_set.begin(), entry_set.end());
        std::sort(entries.begin(), entries.end());
        std::shuffle(entries.begin(), entries.end(), rng);
        for (const uint32_t ep : entries) {
            if (selected.size() >= q_count) {
                break;
            }
            selected.push_back(ep);
        }
    }

    return selected;
}

/**
 * Computes exact k-th nearest neighbor distance thresholds for all sampled queries
 * via a single cache-blocked parallel pass over the graph's stored feature vectors.
 */
inline std::vector<float>
compute_exact_gt_thresholds(const deglib::graph::InternalGraph& graph, std::span<const void* const> query_ptrs, uint32_t effective_k, size_t actual_threads) {
    const uint32_t n = graph.size();
    const size_t q_count = query_ptrs.size();
    const size_t gt_threads = (std::min)(actual_threads, static_cast<size_t>(n));
    const auto& space = graph.getFeatureSpace();
    const auto batch_dist_func = space.get_batch_dist_func();
    const void* dist_param = space.get_dist_func_param();

    const size_t stride_q = q_count * static_cast<size_t>(effective_k);
    std::vector<float> thread_heaps(gt_threads * stride_q);
    std::vector<uint32_t> thread_counts(gt_threads * q_count, 0);
    std::vector<float> thread_radii(gt_threads * q_count, std::numeric_limits<float>::max());

    deglib::concurrent::parallel_batch_for(
        size_t{0}, static_cast<size_t>(n), gt_threads,
        [&](size_t begin, size_t end, size_t thread_id) {
            std::vector<float> dists(q_count);
            float* local_heaps = thread_heaps.data() + thread_id * stride_q;
            uint32_t* local_counts = thread_counts.data() + thread_id * q_count;
            float* local_radii = thread_radii.data() + thread_id * q_count;

            for (size_t v = begin; v < end; ++v) {
                const void* v_feat = graph.getFeatureVector(static_cast<uint32_t>(v));
                batch_dist_func(v_feat, query_ptrs.data(), q_count, dist_param, dists.data(), 0, 0);

                for (size_t q = 0; q < q_count; ++q) {
                    const float d = dists[q];
                    uint32_t cnt = local_counts[q];
                    float* heap = local_heaps + q * effective_k;
                    if (cnt < effective_k) {
                        heap[cnt] = d;
                        ++cnt;
                        local_counts[q] = cnt;
                        std::push_heap(heap, heap + cnt);
                        if (cnt == effective_k) {
                            local_radii[q] = heap[0];
                        }
                    } else if (d < local_radii[q]) {
                        std::pop_heap(heap, heap + effective_k);
                        heap[effective_k - 1] = d;
                        std::push_heap(heap, heap + effective_k);
                        local_radii[q] = heap[0];
                    }
                }
            }
        },
        8
    );

    // Distance kernels on integer quantized codes are exact; floating-point metrics allow tiny ULP rounding tolerance.
    const auto dt = space.metric().get_data_type();
    const bool is_float_space =
        (dt == deglib::distances::MetricDataType::FP32 || dt == deglib::distances::MetricDataType::FP16 || dt == deglib::distances::MetricDataType::EVP);

    std::vector<float> thresholds(q_count);
    std::vector<float> merged;
    merged.reserve(gt_threads * effective_k);

    for (size_t q = 0; q < q_count; ++q) {
        merged.clear();
        for (size_t t = 0; t < gt_threads; ++t) {
            const uint32_t cnt = thread_counts[t * q_count + q];
            const float* src = thread_heaps.data() + t * stride_q + q * effective_k;
            merged.insert(merged.end(), src, src + cnt);
        }
        const size_t kth_idx = static_cast<size_t>(effective_k) - 1;
        std::nth_element(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(kth_idx), merged.end());
        const float kth_dist = merged[kth_idx];
        thresholds[q] = is_float_space ? (kth_dist + 1e-6f * std::abs(kth_dist) + 1e-7f) : kth_dist;
    }

    return thresholds;
}

/**
 * Estimates the wall-clock time (in microseconds) of evaluating all `q_count` queries
 * via parallel brute-force scan across `eval_threads`.
 */
inline double estimate_brute_force_batch_us(const deglib::graph::InternalGraph& graph, std::span<const void* const> query_ptrs, size_t eval_threads) {
    const size_t n = graph.size();
    const size_t q_count = query_ptrs.size();
    const size_t bf_q = eval_threads;
    const auto& space = graph.getFeatureSpace();
    const size_t feature_bytes = (std::max)(size_t{1}, space.get_data_size());

    size_t m_t = n;
    if (n >= bf_q * 1024) {
        const size_t bytes_per_thread = (64ull * 1024 * 1024) / bf_q;
        const size_t target_rows = (std::max)(size_t{1024}, bytes_per_thread / feature_bytes);
        m_t = (std::min)(n / bf_q, target_rows);
    }

    const auto batch_dist_func = space.get_batch_dist_func();
    const void* dist_param = space.get_dist_func_param();
    const int32_t po = graph.getPo();
    const size_t prefetch_bytes = static_cast<size_t>(graph.getPl()) * deglib::memory::L1_CACHE_LINE_SIZE;

    std::vector<float> thread_sinks(bf_q, std::numeric_limits<float>::max());
    const bool disjoint_slices = (n >= bf_q * 1024);

    const auto t_start = std::chrono::steady_clock::now();
    deglib::concurrent::parallel_batch_for(
        size_t{0}, bf_q, bf_q,
        [&](size_t begin, size_t end, size_t) {
            constexpr size_t kChunk = 64;
            const void* ptrs[kChunk];
            float dists[kChunk];

            for (size_t tid = begin; tid < end; ++tid) {
                const void* q_ptr = query_ptrs[tid];
                const size_t slice_start = disjoint_slices ? (tid * m_t) : 0;
                const size_t slice_end = slice_start + m_t;
                float min_d = std::numeric_limits<float>::max();

                for (size_t pos = slice_start; pos < slice_end; pos += kChunk) {
                    const size_t chunk = (std::min)(kChunk, slice_end - pos);
                    for (size_t i = 0; i < chunk; ++i) {
                        ptrs[i] = graph.getFeatureVector(static_cast<uint32_t>(pos + i));
                    }
                    batch_dist_func(q_ptr, ptrs, chunk, dist_param, dists, po, prefetch_bytes);
                    for (size_t i = 0; i < chunk; ++i) {
                        if (dists[i] < min_d) {
                            min_d = dists[i];
                        }
                    }
                }
                thread_sinks[tid] = min_d;
            }
        },
        1
    );
    const double sample_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();

    volatile float sink = 0.0f;
    for (const float v : thread_sinks) {
        sink = sink + v;
    }
    (void)sink;

    const double bf_batch_us = sample_us * (static_cast<double>(q_count) / static_cast<double>(bf_q)) * (static_cast<double>(n) / static_cast<double>(m_t));
    return (std::max)(bf_batch_us, 10000.0);
}

struct EvalResult {
    float recall = 0.0f;
    double elapsed_us = 0.0;
    bool aborted = false;
};

template <bool UseEf>
inline EvalResult evaluate_recall_batch(
    const deglib::graph::InternalGraph& graph,
    std::span<const uint32_t> sample_indices,
    std::span<const float> gt_thresholds,
    uint32_t effective_k,
    uint32_t ef,
    float eps,
    size_t eval_threads,
    double hard_limit_us
) {
    const size_t q_count = sample_indices.size();
    const size_t feature_bytes = graph.getFeatureSpace().get_data_size();

    std::atomic<bool> aborted{false};
    std::vector<uint64_t> thread_hits(eval_threads, 0);

    const auto t_start = std::chrono::steady_clock::now();
    deglib::concurrent::parallel_batch_for(
        size_t{0}, q_count, eval_threads,
        [&](size_t begin, size_t end, size_t thread_id) {
            thread_local deglib::graph::SearchWorkspace ws;
            uint64_t local_hits = 0;

            for (size_t q = begin; q < end; ++q) {
                if (((q - begin) & 3u) == 0u) {
                    if (aborted.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const double elapsed_so_far = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
                    if (elapsed_so_far >= hard_limit_us) {
                        aborted.store(true, std::memory_order_relaxed);
                        break;
                    }
                }

                const std::span<const std::byte> q_span(graph.getFeatureVector(sample_indices[q]), feature_bytes);
                if constexpr (UseEf) {
                    graph.search_ef(ws, q_span, effective_k, ef);
                } else {
                    graph.search(ws, q_span, effective_k, eps);
                }

                const float threshold = gt_thresholds[q];
                if (ws.used_pool) {
                    const size_t count = (std::min)(static_cast<size_t>(effective_k), ws.pool.size());
                    for (size_t i = 0; i < count; ++i) {
                        if (ws.pool[i].getDistance() <= threshold) {
                            ++local_hits;
                        }
                    }
                } else {
                    while (ws.results.size() > effective_k) {
                        ws.results.pop();
                    }
                    for (size_t i = 0; i < ws.results.size(); ++i) {
                        if (ws.results[i].getDistance() <= threshold) {
                            ++local_hits;
                        }
                    }
                }
            }

            thread_hits[thread_id] += local_hits;
        },
        4
    );

    const double elapsed_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
    if (aborted.load(std::memory_order_relaxed)) {
        return {0.0f, elapsed_us, true};
    }

    uint64_t total_hits = 0;
    for (const uint64_t h : thread_hits) {
        total_hits += h;
    }
    const double total_expected = static_cast<double>(q_count) * static_cast<double>(effective_k);
    const float recall = static_cast<float>(static_cast<double>(total_hits) / total_expected);
    return {recall, elapsed_us, false};
}

struct AxisProbe {
    int32_t param = 0;
    float recall = 0.0f;
};

struct CalibratedAxisResult {
    std::optional<std::pair<float, float>> bounds;
    std::vector<AxisProbe> probes;
    std::vector<std::tuple<float, int32_t, float>> entries;  // (target_recall, param, measured_recall)
};

template <typename EvalParamFn>
inline CalibratedAxisResult
calibrate_axis(std::span<const int32_t> ladder, std::span<const float> targets, double soft_budget_us, double hard_abort_us, EvalParamFn&& eval_param) {
    CalibratedAxisResult out;
    if (ladder.empty() || targets.empty()) {
        return out;
    }

    const int32_t param_min = ladder.front();
    std::vector<AxisProbe> probes;

    auto find_probe = [&](int32_t param) -> const AxisProbe* {
        for (const auto& p : probes) {
            if (p.param == param) {
                return &p;
            }
        }
        return nullptr;
    };

    auto insert_probe = [&](int32_t param, float recall) {
        auto it = std::lower_bound(probes.begin(), probes.end(), param, [](const AxisProbe& a, int32_t val) { return a.param < val; });
        if (it != probes.end() && it->param == param) {
            it->recall = recall;
        } else {
            probes.insert(it, AxisProbe{param, recall});
        }
    };

    // 1. Exponential upward ladder guarded by brute-force time budgets.
    const float max_target = targets.back();
    for (const int32_t p : ladder) {
        const double limit_us = (p == param_min) ? (2.0 * hard_abort_us) : hard_abort_us;
        const EvalResult res = eval_param(p, limit_us);
        if (res.aborted || (p > param_min && res.elapsed_us >= hard_abort_us)) {
            break;
        }
        insert_probe(p, res.recall);
        if (res.elapsed_us >= soft_budget_us) {
            break;
        }
        if (res.recall >= max_target) {
            break;
        }
    }

    if (probes.empty()) {
        return out;
    }

    auto compute_bounds = [&]() {
        const float r_min = probes.front().recall;
        const int32_t p_min = probes.front().param;
        float r_max = r_min;
        int32_t p_max = p_min;
        for (const auto& pr : probes) {
            if (pr.recall > r_max + 1e-6f) {
                r_max = pr.recall;
                p_max = pr.param;
            }
        }
        return std::tuple<float, int32_t, float, int32_t>{r_min, p_min, r_max, p_max};
    };

    const auto [init_r_min, init_p_min, init_r_max, init_p_max] = compute_bounds();

    bool had_below_min = false;
    bool had_above_max = false;
    std::vector<float> in_range_targets;
    in_range_targets.reserve(targets.size());

    for (const float t : targets) {
        if (t < init_r_min - 1e-6f) {
            had_below_min = true;
        } else if (t > init_r_max + 1e-6f) {
            had_above_max = true;
        } else {
            in_range_targets.push_back(t);
        }
    }

    // 2. Bracketed binary search for each achievable target using the shared sorted probe cache.
    for (const float t : in_range_targets) {
        if (probes.front().recall >= t - 1e-6f) {
            out.entries.emplace_back(t, probes.front().param, probes.front().recall);
            continue;
        }

        size_t hi_idx = probes.size();
        for (size_t i = 1; i < probes.size(); ++i) {
            if (probes[i].recall >= t - 1e-6f) {
                hi_idx = i;
                break;
            }
        }

        if (hi_idx == probes.size()) {
            const auto [cur_r_min, cur_p_min, cur_r_max, cur_p_max] = compute_bounds();
            (void)cur_r_min;
            (void)cur_p_min;
            out.entries.emplace_back(t, cur_p_max, cur_r_max);
            continue;
        }

        int32_t lo = probes[hi_idx - 1].param;
        int32_t hi = probes[hi_idx].param;
        float r_hi = probes[hi_idx].recall;

        while (hi - lo > 1) {
            const int32_t mid = lo + (hi - lo) / 2;
            float r_mid = 0.0f;
            bool stop_early = false;

            if (const auto* cached = find_probe(mid)) {
                r_mid = cached->recall;
            } else {
                const EvalResult res = eval_param(mid, hard_abort_us);
                if (res.aborted) {
                    break;
                }
                insert_probe(mid, res.recall);
                r_mid = res.recall;
                if (res.elapsed_us >= soft_budget_us) {
                    stop_early = true;
                }
            }

            if (r_mid < t - 1e-6f) {
                lo = mid;
            } else {
                hi = mid;
                r_hi = r_mid;
            }

            if (stop_early) {
                break;
            }
        }

        out.entries.emplace_back(t, hi, r_hi);
    }

    // 3. Finalize recall bounds and replace unreachable target groups with boundary entries (no duplicates).
    const auto [final_r_min, final_p_min, final_r_max, final_p_max] = compute_bounds();
    if (had_below_min && (out.entries.empty() || (std::get<1>(out.entries.front()) > final_p_min && std::get<0>(out.entries.front()) > final_r_min + 1e-6f))) {
        out.entries.insert(out.entries.begin(), std::tuple<float, int32_t, float>{final_r_min, final_p_min, final_r_min});
    }
    if (had_above_max && (out.entries.empty() || (std::get<1>(out.entries.back()) < final_p_max && std::get<0>(out.entries.back()) < final_r_max - 1e-6f))) {
        out.entries.emplace_back(final_r_max, final_p_max, final_r_max);
    }

    out.bounds = std::pair<float, float>{final_r_min, final_r_max};
    out.probes = std::move(probes);
    return out;
}

inline RecallTable build_recall_table_impl(
    const deglib::graph::InternalGraph& graph,
    uint32_t k,
    std::span<const float> resolved_targets,
    size_t sample_count,
    size_t threads,
    double bf_budget_override_us
) {
    validate_common_recall_table_args(k, sample_count);

    RecallTable table;
    table.k_ = k;

    const uint32_t n = graph.size();
    if (n == 0) {
        table.sample_count_ = 0;
        return table;
    }

    const size_t q_count = (std::min)(sample_count, static_cast<size_t>(n));
    const uint32_t effective_k = (std::min)(k, n);
    table.sample_count_ = q_count;

    const size_t actual_threads = (threads == 0) ? (std::max)(size_t{1}, static_cast<size_t>(std::thread::hardware_concurrency())) : threads;
    const size_t eval_threads = (std::min)(actual_threads, q_count);

    const std::vector<uint32_t> sample_indices = sample_distinct_query_indices(graph, q_count, n);
    std::vector<const void*> query_ptrs(q_count);
    for (size_t q = 0; q < q_count; ++q) {
        query_ptrs[q] = graph.getFeatureVector(sample_indices[q]);
    }

    const std::vector<float> gt_thresholds = compute_exact_gt_thresholds(graph, query_ptrs, effective_k, actual_threads);

    const double effective_bf_budget_us =
        (bf_budget_override_us > 0.0) ? bf_budget_override_us : estimate_brute_force_batch_us(graph, query_ptrs, eval_threads);
    const double soft_budget_us = 0.75 * effective_bf_budget_us;
    const double hard_abort_us = 1.0 * effective_bf_budget_us;

    static constexpr int32_t kEfLadder[] = {10, 20, 40, 80, 160, 300};
    auto ef_res = calibrate_axis(kEfLadder, resolved_targets, soft_budget_us, hard_abort_us, [&](int32_t ef, double limit_us) {
        return evaluate_recall_batch<true>(graph, sample_indices, gt_thresholds, effective_k, static_cast<uint32_t>(ef), 0.0f, eval_threads, limit_us);
    });
    table.ef_recall_bounds_ = ef_res.bounds;
    table.ef_entries_.reserve(ef_res.entries.size());
    for (const auto& [target_r, ef_val, meas_r] : ef_res.entries) {
        table.ef_entries_.push_back(EfRecallEntry{target_r, static_cast<uint32_t>(ef_val), meas_r});
    }
    table.ef_curve_.reserve(ef_res.probes.size());
    for (const auto& pr : ef_res.probes) {
        table.ef_curve_.emplace_back(static_cast<uint32_t>(pr.param), pr.recall);
    }

    static constexpr int32_t kEpsLadder[] = {100, 200, 400, 800, 1500, 2000};
    auto eps_res = calibrate_axis(kEpsLadder, resolved_targets, soft_budget_us, hard_abort_us, [&](int32_t eps_scaled, double limit_us) {
        const float eps = static_cast<float>(eps_scaled) / 10000.0f;
        return evaluate_recall_batch<false>(graph, sample_indices, gt_thresholds, effective_k, 0, eps, eval_threads, limit_us);
    });
    table.eps_recall_bounds_ = eps_res.bounds;
    table.eps_entries_.reserve(eps_res.entries.size());
    for (const auto& [target_r, eps_scaled, meas_r] : eps_res.entries) {
        table.eps_entries_.push_back(EpsRecallEntry{target_r, static_cast<float>(eps_scaled) / 10000.0f, meas_r});
    }
    table.eps_curve_.reserve(eps_res.probes.size());
    for (const auto& pr : eps_res.probes) {
        table.eps_curve_.emplace_back(static_cast<float>(pr.param) / 10000.0f, pr.recall);
    }

    return table;
}

}  // namespace detail

/**
 * @brief Calibrates a `RecallTable` for `graph` using a target recall range and step size.
 */
[[nodiscard]] inline RecallTable build_recall_table(
    const deglib::graph::InternalGraph& graph,
    uint32_t k,
    size_t sample_count = RecallTable::kDefaultSampleCount,
    std::pair<float, float> recall_range = {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall},
    float recall_step = RecallTable::kDefaultRecallStep,
    size_t threads = 0
) {
    detail::validate_common_recall_table_args(k, sample_count);
    const std::vector<float> targets = detail::resolve_recall_targets_from_range(recall_range, recall_step);
    return detail::build_recall_table_impl(graph, k, targets, sample_count, threads);
}

/**
 * @brief Calibrates a `RecallTable` for `graph` using explicit target recall levels.
 */
[[nodiscard]] inline RecallTable build_recall_table(
    const deglib::graph::InternalGraph& graph,
    uint32_t k,
    std::span<const float> target_recalls,
    size_t sample_count = RecallTable::kDefaultSampleCount,
    size_t threads = 0
) {
    detail::validate_common_recall_table_args(k, sample_count);
    const std::vector<float> targets = detail::resolve_explicit_recall_targets(target_recalls);
    return detail::build_recall_table_impl(graph, k, targets, sample_count, threads);
}

}  // namespace deglib::search
