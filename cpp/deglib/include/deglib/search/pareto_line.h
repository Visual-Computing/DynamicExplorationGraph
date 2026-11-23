#pragma once

#include "deglib/graph/internal_graph.h"
#include "deglib/search/recall_table.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iomanip>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace deglib::search {

/**
 * Unified recall-latency Pareto frontier across one or more `RecallTable`s.
 *
 * Produced by `DynamicExplorationGraph::to_pareto_line()` / `ParetoLine::build()`.
 * For each target recall across the underlying `RecallTable`s, selects the fastest parameter/regime
 * using single-threaded median latency, removes Pareto-dominated points, and keeps the underlying
 * `RecallTable`s accessible via `recall_tables()`.
 */
class ParetoLine {
  public:
    /**
     * Single non-dominated operating point on a `ParetoLine`.
     *
     * `eps_or_ef >= 1.0f` represents a fixed beam size (`ef`); `eps_or_ef < 1.0f`
     * represents a relative exploration margin (`eps`). Both share the direct
     * `eps_or_ef` convention accepted by `search()`.
     */
    struct Entry {
        float target_recall = 0.0f;
        float eps_or_ef = 0.0f;
        float measured_recall = 0.0f;
        double latency_us = 0.0;
        double qps = 0.0;
        double avg_distance_count = 0.0;

        [[nodiscard]] bool is_ef() const noexcept { return eps_or_ef >= 1.0f; }
        [[nodiscard]] bool is_eps() const noexcept { return eps_or_ef < 1.0f; }
    };

    static constexpr size_t kDefaultRounds = 3;

    ParetoLine() = default;

    /**
     * @brief Builds a `ParetoLine` from one or more precomputed `RecallTable`s.
     */
    [[nodiscard]] static ParetoLine
    build(const deglib::graph::InternalGraph& graph, std::span<const RecallTable> recall_tables, size_t sample_count = 0, size_t rounds = kDefaultRounds) {
        return build_impl(graph, recall_tables, sample_count, rounds);
    }

    /**
     * @brief Builds a `ParetoLine` from a single precomputed `RecallTable`.
     */
    [[nodiscard]] static ParetoLine
    build(const deglib::graph::InternalGraph& graph, const RecallTable& recall_table, size_t sample_count = 0, size_t rounds = kDefaultRounds) {
        return build(graph, std::span<const RecallTable>(&recall_table, 1), sample_count, rounds);
    }

    /**
     * @brief Calibrates `Ef` and `Eps` `RecallTable`s for `graph` and reduces them to a `ParetoLine`.
     */
    [[nodiscard]] static ParetoLine build(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall},
        float recall_step = RecallTable::kDefaultRecallStep,
        size_t threads = 0,
        size_t rounds = kDefaultRounds
    ) {
        if (rounds == 0) {
            throw std::invalid_argument("to_pareto_line: rounds must be >= 1");
        }
        const std::vector<RecallTable> tables = RecallTable::build_all(graph, k, sample_count, recall_range, recall_step, threads);
        return build_impl(graph, tables, 0, rounds);
    }

    /**
     * @brief Calibrates `Ef` and `Eps` `RecallTable`s for explicit target recalls and reduces them to a `ParetoLine`.
     */
    [[nodiscard]] static ParetoLine build(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = RecallTable::kDefaultSampleCount,
        size_t threads = 0,
        size_t rounds = kDefaultRounds
    ) {
        if (rounds == 0) {
            throw std::invalid_argument("to_pareto_line: rounds must be >= 1");
        }
        const std::vector<RecallTable> tables = RecallTable::build_all(graph, k, target_recalls, sample_count, threads);
        return build_impl(graph, tables, 0, rounds);
    }

    [[nodiscard]] uint32_t k() const noexcept { return k_; }
    [[nodiscard]] size_t sample_count() const noexcept { return sample_count_; }
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] size_t size() const noexcept { return entries_.size(); }

    [[nodiscard]] std::optional<std::pair<float, float>> recall_bounds() const noexcept {
        if (entries_.empty()) {
            return std::nullopt;
        }
        return std::pair<float, float>{entries_.front().measured_recall, entries_.back().measured_recall};
    }

    [[nodiscard]] const std::vector<Entry>& entries() const noexcept { return entries_; }
    [[nodiscard]] const std::vector<RecallTable>& recall_tables() const noexcept { return recall_tables_; }

    /**
     * Returns deduplicated `eps_or_ef` parameters along the Pareto frontier in ascending recall order.
     */
    [[nodiscard]] std::vector<float> eps_or_ef_values() const {
        std::vector<float> vals;
        vals.reserve(entries_.size());
        for (const auto& entry : entries_) {
            const float p = entry.eps_or_ef;
            const bool seen = std::any_of(vals.begin(), vals.end(), [p](float existing) { return std::abs(existing - p) <= 1e-6f; });
            if (!seen) {
                vals.push_back(p);
            }
        }
        return vals;
    }

    /**
     * Looks up or interpolates an `eps_or_ef` parameter for `target_recall`.
     *
     * Interpolates only within the same parameter regime (`ef`-`ef` or `eps`-`eps`) using
     * upper-rounding to step `1` / `0.0001`. Across a regime crossover (`ef` -> `eps`), returns
     * the upper frontier parameter without mixing incompatible scales.
     */
    [[nodiscard]] std::optional<float> param_for(float target_recall, bool clamp = false) const {
        if (entries_.empty() || !std::isfinite(target_recall)) {
            return std::nullopt;
        }

        const float lo_bound = (std::min)(entries_.front().target_recall, entries_.front().measured_recall);
        const float hi_bound = (std::max)(entries_.back().target_recall, entries_.back().measured_recall);

        if (target_recall < lo_bound - 1e-6f) {
            return clamp ? std::optional<float>(entries_.front().eps_or_ef) : std::nullopt;
        }
        if (target_recall > hi_bound + 1e-6f) {
            return clamp ? std::optional<float>(entries_.back().eps_or_ef) : std::nullopt;
        }

        for (const auto& entry : entries_) {
            if (std::abs(entry.target_recall - target_recall) <= 1e-6f) {
                return entry.eps_or_ef;
            }
        }

        if (target_recall <= entries_.front().measured_recall + 1e-6f) {
            return entries_.front().eps_or_ef;
        }
        if (target_recall >= entries_.back().measured_recall - 1e-6f) {
            return entries_.back().eps_or_ef;
        }

        for (size_t i = 1; i < entries_.size(); ++i) {
            if (entries_[i].measured_recall >= target_recall - 1e-6f) {
                const auto& a = entries_[i - 1];
                const auto& b = entries_[i];
                if (a.is_ef() && b.is_ef() && b.measured_recall > a.measured_recall + 1e-7f && b.eps_or_ef >= a.eps_or_ef) {
                    const double alpha = std::clamp(
                        static_cast<double>(target_recall - a.measured_recall) / static_cast<double>(b.measured_recall - a.measured_recall), 0.0, 1.0
                    );
                    const double raw = static_cast<double>(a.eps_or_ef) + alpha * static_cast<double>(b.eps_or_ef - a.eps_or_ef);
                    const long long rounded = static_cast<long long>(std::ceil(raw - 1e-3));
                    return std::clamp(static_cast<float>(rounded), a.eps_or_ef, b.eps_or_ef);
                }
                if (a.is_eps() && b.is_eps() && b.measured_recall > a.measured_recall + 1e-7f && b.eps_or_ef >= a.eps_or_ef) {
                    const double alpha = std::clamp(
                        static_cast<double>(target_recall - a.measured_recall) / static_cast<double>(b.measured_recall - a.measured_recall), 0.0, 1.0
                    );
                    const double raw = static_cast<double>(a.eps_or_ef) + alpha * static_cast<double>(b.eps_or_ef - a.eps_or_ef);
                    const double rounded = std::ceil(raw * 10000.0 - 1e-3) / 10000.0;
                    return std::clamp(static_cast<float>(rounded), a.eps_or_ef, b.eps_or_ef);
                }
                return b.eps_or_ef;
            }
        }
        return entries_.back().eps_or_ef;
    }

    /**
     * Formats a human-readable ASCII summary of the Pareto frontier.
     */
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << std::fixed;
        oss << "ParetoLine(k=" << k_ << ", sample_count=" << sample_count_ << ", entries=" << entries_.size() << ")\n";

        const auto bounds = recall_bounds();
        if (bounds.has_value()) {
            oss << "  [recall bounds: " << std::setprecision(4) << bounds->first << " .. " << bounds->second << "]\n";
        }
        if (entries_.empty()) {
            oss << "  <empty>";
            return oss.str();
        }

        oss << "  target_recall | mode |  param | measured_recall | latency_us |       qps | avg_dists\n";
        oss << "  --------------+------+--------+-----------------+------------+-----------+----------\n";
        for (const auto& e : entries_) {
            oss << "  " << std::setw(13) << std::setprecision(4) << e.target_recall << " | " << std::setw(4) << (e.is_ef() ? "ef" : "eps") << " | ";
            if (e.is_ef()) {
                oss << std::setw(6) << static_cast<uint32_t>(std::llround(e.eps_or_ef));
            } else {
                oss << std::setw(6) << std::setprecision(4) << e.eps_or_ef;
            }
            oss << " | " << std::setw(15) << std::setprecision(4) << e.measured_recall << " | " << std::setw(10) << std::setprecision(1) << e.latency_us
                << " | " << std::setw(9) << std::setprecision(1) << e.qps << " | " << std::setw(9) << std::setprecision(1) << e.avg_distance_count << "\n";
        }

        std::string out = oss.str();
        if (!out.empty() && out.back() == '\n') {
            out.pop_back();
        }
        return out;
    }

  private:
    struct TimingSample {
        double latency_us = 0.0;
        double avg_distance_count = 0.0;
    };

    struct CandidateSpec {
        bool use_ef = true;
        int32_t param = 0;  // raw ef for ef, 10000*eps for eps
        float measured_recall = 0.0f;
    };

    static ParetoLine make(uint32_t k, size_t sample_count, std::vector<RecallTable> recall_tables, std::vector<Entry> entries) {
        ParetoLine line;
        line.k_ = k;
        line.sample_count_ = sample_count;
        line.recall_tables_ = std::move(recall_tables);
        line.entries_ = std::move(entries);
        return line;
    }

    static int32_t encode_param(bool use_ef, float eps_or_ef) noexcept {
        return use_ef ? static_cast<int32_t>(std::llround(eps_or_ef)) : static_cast<int32_t>(std::lround(static_cast<double>(eps_or_ef) * 10000.0));
    }

    template <typename MeasureSingleFn, typename MeasureGroupFn>
    static std::vector<Entry> select_entries(std::span<const RecallTable> recall_tables, MeasureSingleFn&& measure_single, MeasureGroupFn&& measure_group) {
        std::vector<float> all_targets;
        size_t total_entries = 0;
        for (const auto& t : recall_tables) {
            total_entries += t.entries().size();
        }
        all_targets.reserve(total_entries);
        for (const auto& t : recall_tables) {
            for (const auto& e : t.entries()) {
                all_targets.push_back(e.target_recall);
            }
        }
        if (all_targets.empty()) {
            return {};
        }

        std::sort(all_targets.begin(), all_targets.end());
        std::vector<float> unique_targets;
        unique_targets.reserve(all_targets.size());
        for (const float t : all_targets) {
            if (unique_targets.empty() || (t - unique_targets.back()) > 1e-6f) {
                unique_targets.push_back(t);
            }
        }

        auto find_table_candidate = [](const RecallTable& table, float target) -> std::optional<CandidateSpec> {
            const bool use_ef = (table.mode() == RecallTable::Mode::Ef);
            for (const auto& e : table.entries()) {
                if (std::abs(e.target_recall - target) <= 1e-6f) {
                    return CandidateSpec{use_ef, encode_param(use_ef, e.eps_or_ef), e.measured_recall};
                }
            }
            const auto param_opt = table.param_for(target, /*clamp=*/false);
            if (!param_opt.has_value()) {
                return std::nullopt;
            }
            return CandidateSpec{use_ef, encode_param(use_ef, *param_opt), target};
        };

        std::unordered_map<int32_t, TimingSample> ef_cache;
        std::unordered_map<int32_t, TimingSample> eps_cache;

        auto make_entry = [](float target, const CandidateSpec& cand, const TimingSample& timing) {
            const float eps_or_ef = cand.use_ef ? static_cast<float>(cand.param) : (static_cast<float>(cand.param) / 10000.0f);
            const double safe_latency = (std::max)(timing.latency_us, 1e-6);
            return Entry{target, eps_or_ef, cand.measured_recall, safe_latency, 1e6 / safe_latency, timing.avg_distance_count};
        };

        std::vector<Entry> chosen;
        chosen.reserve(unique_targets.size());

        for (const float target : unique_targets) {
            std::vector<CandidateSpec> raw_candidates;
            raw_candidates.reserve(recall_tables.size());
            for (const auto& table : recall_tables) {
                if (const auto cand = find_table_candidate(table, target); cand.has_value()) {
                    raw_candidates.push_back(*cand);
                }
            }
            if (raw_candidates.empty()) {
                continue;
            }

            // Deduplicate identical (mode, param), preserving the highest measured recall.
            std::vector<CandidateSpec> dedup;
            dedup.reserve(raw_candidates.size());
            for (const auto& c : raw_candidates) {
                auto it = std::find_if(dedup.begin(), dedup.end(), [&](const CandidateSpec& existing) {
                    return existing.use_ef == c.use_ef && existing.param == c.param;
                });
                if (it != dedup.end()) {
                    it->measured_recall = (std::max)(it->measured_recall, c.measured_recall);
                } else {
                    dedup.push_back(c);
                }
            }

            // Same-mode dominance pre-filter: smaller param does strictly less traversal work.
            std::vector<CandidateSpec> candidates;
            candidates.reserve(dedup.size());
            for (const auto& c : dedup) {
                auto it = std::find_if(candidates.begin(), candidates.end(), [&](const CandidateSpec& existing) { return existing.use_ef == c.use_ef; });
                if (it == candidates.end()) {
                    candidates.push_back(c);
                } else if (c.param < it->param) {
                    *it = c;
                }
            }

            std::vector<CandidateSpec> uncached;
            uncached.reserve(candidates.size());
            for (const auto& c : candidates) {
                const auto& cache = c.use_ef ? ef_cache : eps_cache;
                if (cache.find(c.param) == cache.end()) {
                    uncached.push_back(c);
                }
            }

            if (uncached.size() >= 2) {
                const std::vector<TimingSample> samples = measure_group(std::span<const CandidateSpec>(uncached));
                for (size_t i = 0; i < uncached.size(); ++i) {
                    auto& cache = uncached[i].use_ef ? ef_cache : eps_cache;
                    cache.emplace(uncached[i].param, samples[i]);
                }
            } else if (uncached.size() == 1) {
                const CandidateSpec& c = uncached.front();
                auto& cache = c.use_ef ? ef_cache : eps_cache;
                cache.emplace(c.param, measure_single(c.use_ef, c.param));
            }

            auto get_cached = [&](const CandidateSpec& c) -> TimingSample {
                const auto& cache = c.use_ef ? ef_cache : eps_cache;
                return cache.at(c.param);
            };

            size_t best_idx = 0;
            TimingSample best_timing = get_cached(candidates[0]);
            for (size_t i = 1; i < candidates.size(); ++i) {
                const TimingSample cur_timing = get_cached(candidates[i]);
                const double max_lat = (std::max)(cur_timing.latency_us, best_timing.latency_us);
                bool cur_better = false;
                if (std::abs(cur_timing.latency_us - best_timing.latency_us) <= 0.01 * max_lat) {
                    if (std::abs(cur_timing.avg_distance_count - best_timing.avg_distance_count) > 1e-9) {
                        cur_better = (cur_timing.avg_distance_count < best_timing.avg_distance_count);
                    } else if (candidates[i].use_ef != candidates[best_idx].use_ef) {
                        cur_better = candidates[i].use_ef;  // prefer ef on exact tie
                    } else {
                        cur_better = (candidates[i].param < candidates[best_idx].param);
                    }
                } else {
                    cur_better = (cur_timing.latency_us < best_timing.latency_us);
                }
                if (cur_better) {
                    best_idx = i;
                    best_timing = cur_timing;
                }
            }
            chosen.push_back(make_entry(target, candidates[best_idx], best_timing));
        }

        if (chosen.empty()) {
            return {};
        }

        // Sort by (measured_recall ASC, latency_us DESC, target_recall ASC) so that scanning backward keeps
        // strictly increasing (measured_recall, latency_us) and drops slower or lower-target ties at equal recall.
        std::sort(chosen.begin(), chosen.end(), [](const Entry& a, const Entry& b) {
            if (std::abs(a.measured_recall - b.measured_recall) > 1e-6f) {
                return a.measured_recall < b.measured_recall;
            }
            if (std::abs(a.latency_us - b.latency_us) > 1e-9) {
                return a.latency_us > b.latency_us;
            }
            return a.target_recall < b.target_recall;
        });

        std::vector<Entry> frontier;
        frontier.reserve(chosen.size());
        double best_latency = std::numeric_limits<double>::infinity();
        for (size_t idx = chosen.size(); idx-- > 0;) {
            if (chosen[idx].latency_us < best_latency - 1e-9) {
                frontier.push_back(chosen[idx]);
                best_latency = chosen[idx].latency_us;
            }
        }
        std::reverse(frontier.begin(), frontier.end());
        return frontier;
    }

    static ParetoLine build_impl(const deglib::graph::InternalGraph& graph, std::span<const RecallTable> recall_tables, size_t sample_count, size_t rounds) {
        if (rounds == 0) {
            throw std::invalid_argument("to_pareto_line: rounds must be >= 1");
        }

        uint32_t k = 0;
        if (!recall_tables.empty()) {
            k = recall_tables.front().k();
            for (const auto& t : recall_tables) {
                if (t.k() != k) {
                    throw std::invalid_argument(std::format("to_pareto_line: recall_tables have mismatched k ({} vs {})", k, t.k()));
                }
            }
        }

        ParetoLine line;
        line.k_ = k;
        line.recall_tables_.assign(recall_tables.begin(), recall_tables.end());

        const uint32_t n = graph.size();
        const RecallTable* first_non_empty = nullptr;
        for (const auto& t : recall_tables) {
            if (!t.empty()) {
                first_non_empty = &t;
                break;
            }
        }
        if (n == 0 || k == 0 || first_non_empty == nullptr) {
            line.sample_count_ = 0;
            return line;
        }

        size_t table_samples = 0;
        for (const auto& t : recall_tables) {
            if (t.sample_count() > 0) {
                table_samples = t.sample_count();
                break;
            }
        }
        const size_t base_samples = (sample_count == 0) ? (table_samples > 0 ? table_samples : RecallTable::kDefaultSampleCount) : sample_count;
        const size_t q_count = (std::min)(base_samples, static_cast<size_t>(n));
        const uint32_t effective_k = (std::min)(k, n);
        line.sample_count_ = q_count;

        const std::vector<uint32_t> sample_indices = RecallTable::sample_distinct_query_indices(graph, q_count, n);
        const size_t feature_bytes = graph.getFeatureSpace().get_data_size();
        deglib::graph::SearchWorkspace ws;

        auto run_single_pass = [&](bool use_ef, int32_t param, size_t count) -> TimingSample {
            uint64_t total_dists = 0;
            size_t anti_dce = 0;
            const auto t_start = std::chrono::steady_clock::now();
            for (size_t q = 0; q < count; ++q) {
                const std::span<const std::byte> q_span(graph.getFeatureVector(sample_indices[q]), feature_bytes);
                if (use_ef) {
                    graph.search_ef(ws, q_span, effective_k, static_cast<uint32_t>(param));
                } else {
                    graph.search(ws, q_span, effective_k, static_cast<float>(param) / 10000.0f);
                }
                total_dists += ws.graph_distance_count;
                anti_dce += ws.size();
            }
            const double total_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
            (void)anti_dce;
            const double denom = static_cast<double>((std::max)(count, size_t{1}));
            return {(std::max)(total_us / denom, 1e-6), static_cast<double>(total_dists) / denom};
        };

        // Short warmup using the first non-empty table's first entry to stabilize caches and frequency.
        const size_t warmup_count = (std::min)(q_count, size_t{64});
        const bool warmup_is_ef = (first_non_empty->mode() == RecallTable::Mode::Ef);
        const int32_t warmup_param = encode_param(warmup_is_ef, first_non_empty->entries().front().eps_or_ef);
        (void)run_single_pass(warmup_is_ef, warmup_param, warmup_count);

        auto measure_single = [&](bool use_ef, int32_t param) -> TimingSample {
            std::vector<double> lats;
            lats.reserve(rounds);
            double dist_sum = 0.0;
            for (size_t r = 0; r < rounds; ++r) {
                const TimingSample s = run_single_pass(use_ef, param, q_count);
                lats.push_back(s.latency_us);
                dist_sum += s.avg_distance_count;
            }
            std::sort(lats.begin(), lats.end());
            return {lats[rounds / 2], dist_sum / static_cast<double>(rounds)};
        };

        auto measure_group = [&](std::span<const CandidateSpec> group) -> std::vector<TimingSample> {
            const size_t m = group.size();
            std::vector<std::vector<double>> lats(m);
            std::vector<double> dist_sums(m, 0.0);
            for (auto& v : lats) {
                v.reserve(rounds);
            }
            for (size_t r = 0; r < rounds; ++r) {
                const bool reverse = ((r & size_t{1}) != 0);
                for (size_t step = 0; step < m; ++step) {
                    const size_t idx = reverse ? (m - 1 - step) : step;
                    const TimingSample s = run_single_pass(group[idx].use_ef, group[idx].param, q_count);
                    lats[idx].push_back(s.latency_us);
                    dist_sums[idx] += s.avg_distance_count;
                }
            }
            std::vector<TimingSample> out(m);
            const double r_d = static_cast<double>(rounds);
            for (size_t idx = 0; idx < m; ++idx) {
                std::sort(lats[idx].begin(), lats[idx].end());
                out[idx] = TimingSample{lats[idx][rounds / 2], dist_sums[idx] / r_d};
            }
            return out;
        };

        line.entries_ = select_entries(recall_tables, measure_single, measure_group);
        return line;
    }

    uint32_t k_ = 0;
    size_t sample_count_ = 0;
    std::vector<RecallTable> recall_tables_;
    std::vector<Entry> entries_;

    friend struct ParetoLineTestAccess;
};

}  // namespace deglib::search
