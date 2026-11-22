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
 * Produced by `DynamicExplorationGraph::to_pareto_line()` / `Searcher::to_pareto_line()` / `ParetoLine::build()`.
 * Gathers real measured operating points across the underlying `RecallTable`s in `(measured_recall, latency_us)`
 * space using single-threaded median latency, removes Pareto-dominated points, and keeps the underlying
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
        float rerank_factor = 1.0f;
        float measured_recall = 0.0f;
        double latency_us = 0.0;
        double qps = 0.0;
        double avg_distance_count = 0.0;
        double avg_rerank_distance_count = 0.0;

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
     * Returns deduplicated `(eps_or_ef, rerank_factor)` parameter pairs along the Pareto frontier in ascending recall order.
     */
    [[nodiscard]] std::vector<std::pair<float, float>> params() const {
        std::vector<std::pair<float, float>> vals;
        vals.reserve(entries_.size());
        for (const auto& entry : entries_) {
            const float p = entry.eps_or_ef;
            const float rf = entry.rerank_factor;
            const bool seen = std::any_of(vals.begin(), vals.end(), [p, rf](const std::pair<float, float>& existing) {
                return std::abs(existing.first - p) <= 1e-6f && std::abs(existing.second - rf) <= 1e-6f;
            });
            if (!seen) {
                vals.emplace_back(p, rf);
            }
        }
        return vals;
    }

    /**
     * Returns `(eps_or_ef, rerank_factor)` of the frontier entry whose `measured_recall` is closest
     * to `target_recall` (breaking ties within `1e-6` toward the higher-recall entry).
     *
     * Returns `std::nullopt` if empty, non-finite, or outside `recall_bounds()` unless `clamp == true`.
     */
    [[nodiscard]] std::optional<std::pair<float, float>> param_for(float target_recall, bool clamp = false) const {
        if (entries_.empty() || !std::isfinite(target_recall)) {
            return std::nullopt;
        }

        const float r_min = entries_.front().measured_recall;
        const float r_max = entries_.back().measured_recall;

        if (target_recall < r_min - 1e-6f) {
            return clamp ? std::optional<std::pair<float, float>>({entries_.front().eps_or_ef, entries_.front().rerank_factor}) : std::nullopt;
        }
        if (target_recall > r_max + 1e-6f) {
            return clamp ? std::optional<std::pair<float, float>>({entries_.back().eps_or_ef, entries_.back().rerank_factor}) : std::nullopt;
        }

        if (entries_.size() == 1 || target_recall <= r_min) {
            return std::pair<float, float>{entries_.front().eps_or_ef, entries_.front().rerank_factor};
        }
        if (target_recall >= r_max) {
            return std::pair<float, float>{entries_.back().eps_or_ef, entries_.back().rerank_factor};
        }

        const auto it = std::lower_bound(entries_.begin(), entries_.end(), target_recall, [](const Entry& e, float val) { return e.measured_recall < val; });
        if (it == entries_.begin()) {
            return std::pair<float, float>{entries_.front().eps_or_ef, entries_.front().rerank_factor};
        }
        if (it == entries_.end()) {
            return std::pair<float, float>{entries_.back().eps_or_ef, entries_.back().rerank_factor};
        }

        const auto& a = *(it - 1);
        const auto& b = *it;
        const double dist_a = static_cast<double>(target_recall) - static_cast<double>(a.measured_recall);
        const double dist_b = static_cast<double>(b.measured_recall) - static_cast<double>(target_recall);
        const Entry& best = (dist_a < dist_b - 1e-6) ? a : b;
        return std::pair<float, float>{best.eps_or_ef, best.rerank_factor};
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

        const bool has_rerank =
            std::any_of(entries_.begin(), entries_.end(), [](const Entry& e) { return e.rerank_factor > 1.0f + 1e-6f || e.avg_rerank_distance_count > 0.0; });

        if (has_rerank) {
            oss << "  target_recall | mode |  param |    rf | measured_recall | latency_us |       qps | graph_dists | rerank_dists\n";
            oss << "  --------------+------+--------+-------+-----------------+------------+-----------+-------------+-------------\n";
        } else {
            oss << "  target_recall | mode |  param | measured_recall | latency_us |       qps | avg_dists\n";
            oss << "  --------------+------+--------+-----------------+------------+-----------+----------\n";
        }
        for (const auto& e : entries_) {
            oss << "  " << std::setw(13) << std::setprecision(4) << e.target_recall << " | " << std::setw(4) << (e.is_ef() ? "ef" : "eps") << " | ";
            if (e.is_ef()) {
                oss << std::setw(6) << static_cast<uint32_t>(std::llround(e.eps_or_ef));
            } else {
                oss << std::setw(6) << std::setprecision(4) << e.eps_or_ef;
            }
            if (has_rerank) {
                oss << " | " << std::setw(5) << std::setprecision(2) << e.rerank_factor;
            }
            oss << " | " << std::setw(15) << std::setprecision(4) << e.measured_recall << " | " << std::setw(10) << std::setprecision(1) << e.latency_us
                << " | " << std::setw(9) << std::setprecision(1) << e.qps;
            if (has_rerank) {
                oss << " | " << std::setw(11) << std::setprecision(1) << e.avg_distance_count << " | " << std::setw(12) << std::setprecision(1)
                    << e.avg_rerank_distance_count << "\n";
            } else {
                oss << " | " << std::setw(9) << std::setprecision(1) << e.avg_distance_count << "\n";
            }
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
        double avg_rerank_distance_count = 0.0;
    };

    struct CandidateSpec {
        bool use_ef = true;
        int32_t param = 0;  // raw ef for ef, 10000*eps for eps
        float rerank_factor = 1.0f;
        uint32_t num_candidates = 0;  // k for Graph / rf <= 1.0, ceil(k * rf) for Searcher with reranker
        float measured_recall = 0.0f;
        float target_recall = 0.0f;
    };

    struct TimingKey {
        bool use_ef = true;
        int32_t param = 0;
        uint32_t num_candidates = 0;

        bool operator==(const TimingKey&) const noexcept = default;
    };

    struct TimingKeyHash {
        size_t operator()(const TimingKey& key) const noexcept {
            size_t h = std::hash<bool>{}(key.use_ef);
            h ^= std::hash<int32_t>{}(key.param) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(key.num_candidates) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
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

    static uint32_t compute_num_candidates(uint32_t effective_k, float rerank_factor, bool has_reranker) noexcept {
        if (!has_reranker) {
            return effective_k;
        }
        return static_cast<uint32_t>(std::ceil(static_cast<double>(effective_k) * static_cast<double>((std::max)(1.0f, rerank_factor))));
    }

    template <typename MeasureSingleFn, typename MeasureGroupFn>
    static std::vector<Entry> select_entries(
        uint32_t effective_k,
        bool has_reranker,
        std::span<const RecallTable> recall_tables,
        size_t rounds,
        MeasureSingleFn&& measure_single,
        MeasureGroupFn&& measure_group
    ) {
        size_t total_entries = 0;
        for (const auto& t : recall_tables) {
            total_entries += t.entries().size();
        }
        if (total_entries == 0) {
            return {};
        }

        // Step A: Gather real measured operating points only (no synthetic interpolation) and deduplicate identical execution configs.
        std::unordered_map<TimingKey, CandidateSpec, TimingKeyHash> unique_by_key;
        unique_by_key.reserve(total_entries);
        for (const auto& table : recall_tables) {
            const bool use_ef = (table.mode() == RecallTable::Mode::Ef);
            const float rf = table.rerank_factor();
            const uint32_t num_cands = compute_num_candidates(effective_k, rf, has_reranker);
            for (const auto& e : table.entries()) {
                const int32_t param = encode_param(use_ef, e.eps_or_ef);
                const TimingKey key{use_ef, param, num_cands};
                const CandidateSpec cand{use_ef, param, rf, num_cands, e.measured_recall, e.target_recall};
                auto [it, inserted] = unique_by_key.emplace(key, cand);
                if (!inserted) {
                    CandidateSpec& ex = it->second;
                    if (cand.measured_recall > ex.measured_recall + 1e-6f) {
                        ex.measured_recall = cand.measured_recall;
                        ex.target_recall = cand.target_recall;
                    } else if (std::abs(cand.measured_recall - ex.measured_recall) <= 1e-6f) {
                        ex.measured_recall = (std::max)(ex.measured_recall, cand.measured_recall);
                        ex.target_recall = (std::max)(ex.target_recall, cand.target_recall);
                    }
                    ex.rerank_factor = (std::min)(ex.rerank_factor, cand.rerank_factor);
                }
            }
        }

        // Step B: Deterministic sort & exact within-axis dominance pre-filter per (use_ef, num_candidates).
        std::vector<CandidateSpec> raw_candidates;
        raw_candidates.reserve(unique_by_key.size());
        for (const auto& kv : unique_by_key) {
            raw_candidates.push_back(kv.second);
        }
        std::sort(raw_candidates.begin(), raw_candidates.end(), [](const CandidateSpec& a, const CandidateSpec& b) {
            if (a.use_ef != b.use_ef) {
                return a.use_ef;
            }
            if (a.num_candidates != b.num_candidates) {
                return a.num_candidates < b.num_candidates;
            }
            if (a.param != b.param) {
                return a.param < b.param;
            }
            return a.measured_recall > b.measured_recall;
        });

        std::vector<CandidateSpec> candidates;
        candidates.reserve(raw_candidates.size());
        size_t grp_start = 0;
        while (grp_start < raw_candidates.size()) {
            const bool grp_ef = raw_candidates[grp_start].use_ef;
            const uint32_t grp_cands = raw_candidates[grp_start].num_candidates;
            float best_recall = -1.0f;
            size_t idx = grp_start;
            while (idx < raw_candidates.size() && raw_candidates[idx].use_ef == grp_ef && raw_candidates[idx].num_candidates == grp_cands) {
                if (raw_candidates[idx].measured_recall > best_recall + 1e-6f) {
                    candidates.push_back(raw_candidates[idx]);
                    best_recall = raw_candidates[idx].measured_recall;
                }
                ++idx;
            }
            grp_start = idx;
        }
        if (candidates.empty()) {
            return {};
        }

        std::sort(candidates.begin(), candidates.end(), [](const CandidateSpec& a, const CandidateSpec& b) {
            if (a.measured_recall != b.measured_recall) {
                return a.measured_recall < b.measured_recall;
            }
            if (a.use_ef != b.use_ef) {
                return a.use_ef;
            }
            if (a.num_candidates != b.num_candidates) {
                return a.num_candidates < b.num_candidates;
            }
            return a.param < b.param;
        });

        // Step C: Measure latencies using 2-stage global Pareto screening when worthwhile.
        std::vector<CandidateSpec> final_candidates;
        std::vector<TimingSample> final_timings;
        if (candidates.size() == 1) {
            final_candidates = std::move(candidates);
            final_timings.push_back(measure_single(final_candidates.front(), rounds));
        } else if (rounds >= 3 && candidates.size() > 2) {
            const std::vector<TimingSample> screen_samples = measure_group(std::span<const CandidateSpec>(candidates), size_t{1});
            const size_t m = candidates.size();
            std::vector<double> min_lat_at_or_above(m, std::numeric_limits<double>::infinity());
            double suffix_min = std::numeric_limits<double>::infinity();
            size_t grp_end = m;
            while (grp_end > 0) {
                const float grp_max_r = candidates[grp_end - 1].measured_recall;
                size_t grp_begin = grp_end - 1;
                while (grp_begin > 0 && (grp_max_r - candidates[grp_begin - 1].measured_recall) <= 1e-6f) {
                    --grp_begin;
                }
                for (size_t j = grp_begin; j < grp_end; ++j) {
                    suffix_min = (std::min)(suffix_min, screen_samples[j].latency_us);
                }
                for (size_t j = grp_begin; j < grp_end; ++j) {
                    min_lat_at_or_above[j] = suffix_min;
                }
                grp_end = grp_begin;
            }

            final_candidates.reserve(m);
            for (size_t i = 0; i < m; ++i) {
                if (screen_samples[i].latency_us <= min_lat_at_or_above[i] * 1.20) {
                    final_candidates.push_back(candidates[i]);
                }
            }

            if (final_candidates.size() >= 2) {
                final_timings = measure_group(std::span<const CandidateSpec>(final_candidates), rounds);
            } else if (final_candidates.size() == 1) {
                final_timings.push_back(measure_single(final_candidates.front(), rounds));
            }
        } else {
            final_candidates = std::move(candidates);
            final_timings = measure_group(std::span<const CandidateSpec>(final_candidates), rounds);
        }

        // Step D: Reduce equal measured_recall runs with 1% latency tolerance + distance tie-breaker, then strictly Pareto prune in 2D.
        std::vector<Entry> scored;
        scored.reserve(final_candidates.size());
        for (size_t i = 0; i < final_candidates.size(); ++i) {
            const CandidateSpec& cand = final_candidates[i];
            const TimingSample& timing = final_timings[i];
            const float eps_or_ef = cand.use_ef ? static_cast<float>(cand.param) : (static_cast<float>(cand.param) / 10000.0f);
            const double safe_latency = (std::max)(timing.latency_us, 1e-6);
            scored.push_back(
                Entry{
                    cand.target_recall, eps_or_ef, cand.rerank_factor, cand.measured_recall, safe_latency, 1e6 / safe_latency, timing.avg_distance_count,
                    timing.avg_rerank_distance_count
                }
            );
        }

        std::vector<Entry> reduced;
        reduced.reserve(scored.size());
        size_t i = 0;
        while (i < scored.size()) {
            size_t j = i + 1;
            while (j < scored.size() && (scored[j].measured_recall - scored[i].measured_recall) <= 1e-6f) {
                ++j;
            }
            if (j - i == 1) {
                reduced.push_back(scored[i]);
            } else {
                double min_lat = scored[i].latency_us;
                for (size_t k = i + 1; k < j; ++k) {
                    min_lat = (std::min)(min_lat, scored[k].latency_us);
                }
                const double tie_cutoff = min_lat * 1.01;
                std::optional<size_t> best_idx;
                for (size_t k = i; k < j; ++k) {
                    if (scored[k].latency_us > tie_cutoff) {
                        continue;
                    }
                    if (!best_idx.has_value()) {
                        best_idx = k;
                        continue;
                    }
                    const Entry& cur = scored[k];
                    const Entry& best = scored[*best_idx];
                    const double cur_dists = cur.avg_distance_count + cur.avg_rerank_distance_count;
                    const double best_dists = best.avg_distance_count + best.avg_rerank_distance_count;
                    bool cur_better = false;
                    if (std::abs(cur_dists - best_dists) > 1e-9) {
                        cur_better = (cur_dists < best_dists);
                    } else if (cur.is_ef() != best.is_ef()) {
                        cur_better = cur.is_ef();
                    } else if (std::abs(cur.rerank_factor - best.rerank_factor) > 1e-6f) {
                        cur_better = (cur.rerank_factor < best.rerank_factor);
                    } else if (std::abs(cur.eps_or_ef - best.eps_or_ef) > 1e-6f) {
                        cur_better = (cur.eps_or_ef < best.eps_or_ef);
                    } else {
                        cur_better = (cur.target_recall > best.target_recall);
                    }
                    if (cur_better) {
                        best_idx = k;
                    }
                }
                reduced.push_back(scored[*best_idx]);
            }
            i = j;
        }

        std::vector<Entry> frontier;
        frontier.reserve(reduced.size());
        double best_latency = std::numeric_limits<double>::infinity();
        for (size_t idx = reduced.size(); idx-- > 0;) {
            if (reduced[idx].latency_us < best_latency - 1e-9) {
                frontier.push_back(reduced[idx]);
                best_latency = reduced[idx].latency_us;
            }
        }
        std::reverse(frontier.begin(), frontier.end());
        return frontier;
    }

    static std::pair<ParetoLine, size_t>
    validate_and_init(size_t collection_size, std::span<const RecallTable> recall_tables, size_t sample_count, size_t rounds) {
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

        const bool has_non_empty = std::any_of(recall_tables.begin(), recall_tables.end(), [](const RecallTable& t) { return !t.empty(); });
        if (collection_size == 0 || k == 0 || !has_non_empty) {
            line.sample_count_ = 0;
            return {std::move(line), 0};
        }

        size_t table_samples = 0;
        for (const auto& t : recall_tables) {
            if (t.sample_count() > 0) {
                table_samples = t.sample_count();
                break;
            }
        }
        const size_t base_samples = (sample_count == 0) ? (table_samples > 0 ? table_samples : RecallTable::kDefaultSampleCount) : sample_count;
        const size_t q_count = (std::min)(base_samples, collection_size);
        line.sample_count_ = q_count;
        return {std::move(line), q_count};
    }

    template <typename RunPassFn>
    static ParetoLine run_pareto_selection(ParetoLine line, size_t collection_size, size_t rounds, bool has_reranker, RunPassFn&& run_single_pass) {
        const size_t q_count = line.sample_count_;
        const uint32_t effective_k = static_cast<uint32_t>((std::min)(static_cast<size_t>(line.k_), collection_size));
        const RecallTable* first_non_empty = nullptr;
        for (const auto& t : line.recall_tables_) {
            if (!t.empty()) {
                first_non_empty = &t;
                break;
            }
        }

        // Short warmup using the first non-empty table's first entry to stabilize caches and frequency.
        const size_t warmup_count = (std::min)(q_count, size_t{64});
        const bool warmup_is_ef = (first_non_empty->mode() == RecallTable::Mode::Ef);
        const float warmup_rf = first_non_empty->rerank_factor();
        const CandidateSpec warmup_cand{
            warmup_is_ef,
            encode_param(warmup_is_ef, first_non_empty->entries().front().eps_or_ef),
            warmup_rf,
            compute_num_candidates(effective_k, warmup_rf, has_reranker),
            first_non_empty->entries().front().measured_recall,
            first_non_empty->entries().front().target_recall
        };
        (void)run_single_pass(warmup_cand, warmup_count);

        auto measure_single = [&](const CandidateSpec& cand, size_t single_rounds) -> TimingSample {
            std::vector<double> lats;
            lats.reserve(single_rounds);
            double dist_sum = 0.0;
            double rerank_sum = 0.0;
            for (size_t r = 0; r < single_rounds; ++r) {
                const TimingSample s = run_single_pass(cand, q_count);
                lats.push_back(s.latency_us);
                dist_sum += s.avg_distance_count;
                rerank_sum += s.avg_rerank_distance_count;
            }
            std::sort(lats.begin(), lats.end());
            const double denom = static_cast<double>(single_rounds);
            return {lats[single_rounds / 2], dist_sum / denom, rerank_sum / denom};
        };

        auto measure_group = [&](std::span<const CandidateSpec> group, size_t group_rounds) -> std::vector<TimingSample> {
            const size_t m = group.size();
            std::vector<std::vector<double>> lats(m);
            std::vector<double> dist_sums(m, 0.0);
            std::vector<double> rerank_sums(m, 0.0);
            for (auto& v : lats) {
                v.reserve(group_rounds);
            }
            for (size_t r = 0; r < group_rounds; ++r) {
                const bool reverse = ((r & size_t{1}) != 0);
                for (size_t step = 0; step < m; ++step) {
                    const size_t idx = reverse ? (m - 1 - step) : step;
                    const TimingSample s = run_single_pass(group[idx], q_count);
                    lats[idx].push_back(s.latency_us);
                    dist_sums[idx] += s.avg_distance_count;
                    rerank_sums[idx] += s.avg_rerank_distance_count;
                }
            }
            std::vector<TimingSample> out(m);
            const double r_d = static_cast<double>(group_rounds);
            for (size_t idx = 0; idx < m; ++idx) {
                std::sort(lats[idx].begin(), lats[idx].end());
                out[idx] = TimingSample{lats[idx][group_rounds / 2], dist_sums[idx] / r_d, rerank_sums[idx] / r_d};
            }
            return out;
        };

        line.entries_ = select_entries(effective_k, has_reranker, line.recall_tables_, rounds, measure_single, measure_group);
        return line;
    }

    static ParetoLine build_impl(const deglib::graph::InternalGraph& graph, std::span<const RecallTable> recall_tables, size_t sample_count, size_t rounds) {
        const uint32_t n = graph.size();
        auto [line, q_count] = validate_and_init(n, recall_tables, sample_count, rounds);
        if (q_count == 0) {
            return line;
        }

        const uint32_t effective_k = (std::min)(line.k_, n);
        const std::vector<uint32_t> sample_indices = RecallTable::sample_distinct_query_indices(graph, q_count, n);
        const size_t feature_bytes = graph.getFeatureSpace().get_data_size();
        deglib::graph::SearchWorkspace ws;

        auto run_single_pass = [&](const CandidateSpec& cand, size_t count) -> TimingSample {
            uint64_t total_dists = 0;
            size_t anti_dce = 0;
            const auto t_start = std::chrono::steady_clock::now();
            for (size_t q = 0; q < count; ++q) {
                const std::span<const std::byte> q_span(graph.getFeatureVector(sample_indices[q]), feature_bytes);
                if (cand.use_ef) {
                    graph.search_ef(ws, q_span, effective_k, static_cast<uint32_t>(cand.param));
                } else {
                    graph.search(ws, q_span, effective_k, static_cast<float>(cand.param) / 10000.0f);
                }
                total_dists += ws.graph_distance_count;
                anti_dce += ws.size();
            }
            const double total_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_start).count();
            (void)anti_dce;
            const double denom = static_cast<double>((std::max)(count, size_t{1}));
            return {(std::max)(total_us / denom, 1e-6), static_cast<double>(total_dists) / denom, 0.0};
        };

        return run_pareto_selection(std::move(line), n, rounds, /*has_reranker=*/false, run_single_pass);
    }

    uint32_t k_ = 0;
    size_t sample_count_ = 0;
    std::vector<RecallTable> recall_tables_;
    std::vector<Entry> entries_;

    friend class Searcher;
    friend struct ParetoLineTestAccess;
};

}  // namespace deglib::search
