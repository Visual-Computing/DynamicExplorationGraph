// test_pareto_line.cpp — Unit tests for deglib::search::ParetoLine and DynamicExplorationGraph::to_pareto_line

#include "deglib/builder.h"
#include "deglib/deglib.h"
#include "deglib/distances.h"
#include "deglib/graph.h"
#include "deglib/search/pareto_line.h"
#include "deglib/search/searcher.h"
#include "deglib/transformation.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

namespace deglib::search {
struct ParetoLineTestAccess {
    using TimingSample = ParetoLine::TimingSample;
    using CandidateSpec = ParetoLine::CandidateSpec;

    static ParetoLine make(uint32_t k, size_t sample_count, std::vector<RecallTable> recall_tables, std::vector<ParetoLine::Entry> entries) {
        return ParetoLine::make(k, sample_count, std::move(recall_tables), std::move(entries));
    }

    template <typename MeasureSingleFn, typename MeasureGroupFn>
    static std::vector<ParetoLine::Entry> select_entries(
        uint32_t effective_k,
        bool has_reranker,
        std::span<const RecallTable> recall_tables,
        size_t rounds,
        MeasureSingleFn&& measure_single,
        MeasureGroupFn&& measure_group
    ) {
        return ParetoLine::select_entries(
            effective_k, has_reranker, recall_tables, rounds, std::forward<MeasureSingleFn>(measure_single), std::forward<MeasureGroupFn>(measure_group)
        );
    }
};
}  // namespace deglib::search

namespace {

using deglib::builder::OptimizationTarget;
using deglib::distances::Metric;
using deglib::search::ParetoLine;
using deglib::search::RecallTable;

std::vector<float> make_random_data(uint32_t count, uint32_t dim, uint32_t seed, bool normalize = false) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(static_cast<size_t>(count) * dim);
    for (float& v : data) {
        v = dist(rng);
    }
    if (normalize) {
        for (uint32_t i = 0; i < count; ++i) {
            float* row = data.data() + static_cast<size_t>(i) * dim;
            float norm_sq = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) {
                norm_sq += row[d] * row[d];
            }
            const float inv = 1.0f / std::sqrt((std::max)(norm_sq, 1e-12f));
            for (uint32_t d = 0; d < dim; ++d) {
                row[d] *= inv;
            }
        }
    }
    return data;
}

std::vector<float> make_clustered_data(uint32_t count, uint32_t dim, uint32_t num_clusters, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> center_dist(0.0f, 3.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.35f);

    std::vector<float> centers(static_cast<size_t>(num_clusters) * dim);
    for (float& v : centers) {
        v = center_dist(rng);
    }

    std::vector<float> data(static_cast<size_t>(count) * dim);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t c = i % num_clusters;
        const float* c_ptr = centers.data() + static_cast<size_t>(c) * dim;
        float* dst = data.data() + static_cast<size_t>(i) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            dst[d] = c_ptr[d] + noise_dist(rng);
        }
    }
    return data;
}

deglib::DynamicExplorationGraph build_fp32_graph(
    std::span<const float> data,
    uint32_t dim,
    uint8_t edges_per_vertex = 16,
    Metric metric = Metric::FP32_L2,
    OptimizationTarget opt = OptimizationTarget::LowLID
) {
    return deglib::builder::build_from_data(data, dim, {}, edges_per_vertex, metric, opt, edges_per_vertex * 2, 0.1f, 0, 0.001f, 5, 0, 2, 42);
}

void verify_pareto_invariants(const ParetoLine& line) {
    EXPECT_EQ(line.size(), line.entries().size());
    EXPECT_EQ(line.empty(), line.entries().empty());

    const auto& entries = line.entries();
    if (entries.empty()) {
        EXPECT_FALSE(line.recall_bounds().has_value());
        EXPECT_TRUE(line.eps_or_ef_values().empty());
        return;
    }

    ASSERT_TRUE(line.recall_bounds().has_value());
    EXPECT_FLOAT_EQ(line.recall_bounds()->first, entries.front().measured_recall);
    EXPECT_FLOAT_EQ(line.recall_bounds()->second, entries.back().measured_recall);

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        EXPECT_TRUE(e.is_ef() != e.is_eps());
        if (e.is_ef()) {
            EXPECT_GE(e.eps_or_ef, static_cast<float>(RecallTable::kMinEf));
            EXPECT_LE(e.eps_or_ef, static_cast<float>(RecallTable::kMaxEf));
        } else {
            EXPECT_GE(e.eps_or_ef, RecallTable::kMinEps - 1e-6f);
            EXPECT_LE(e.eps_or_ef, RecallTable::kMaxEps + 1e-6f);
        }
        EXPECT_GT(e.latency_us, 0.0);
        EXPECT_NEAR(e.qps, 1e6 / e.latency_us, 1e-3 * (std::max)(1.0, e.qps));
        EXPECT_GE(e.rerank_factor, 1.0f);
        EXPECT_GE(e.avg_distance_count, 0.0);
        EXPECT_GE(e.avg_rerank_distance_count, 0.0);
        if (i > 0) {
            EXPECT_GT(e.measured_recall, entries[i - 1].measured_recall);
            EXPECT_GT(e.latency_us, entries[i - 1].latency_us);
            EXPECT_LT(e.qps, entries[i - 1].qps);
        }

        const auto found = line.param_for(e.measured_recall);
        ASSERT_TRUE(found.has_value());
        EXPECT_NEAR(found->first, e.eps_or_ef, 1e-6f);
        EXPECT_FLOAT_EQ(found->second, e.rerank_factor);
    }

    const auto params = line.eps_or_ef_values();
    EXPECT_FALSE(params.empty());
    EXPECT_LE(params.size(), entries.size());
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t j = i + 1; j < params.size(); ++j) {
            EXPECT_GT(std::abs(params[i] - params[j]), 1e-6f);
        }
    }

    const auto pairs = line.params();
    EXPECT_FALSE(pairs.empty());
    EXPECT_LE(pairs.size(), entries.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
        for (size_t j = i + 1; j < pairs.size(); ++j) {
            EXPECT_TRUE(std::abs(pairs[i].first - pairs[j].first) > 1e-6f || std::abs(pairs[i].second - pairs[j].second) > 1e-6f);
        }
    }
}

TEST(ParetoLineTest, SyntheticCrossoverPruningAndParamLookup) {
    using deglib::search::ParetoLineTestAccess;
    using CandidateSpec = ParetoLineTestAccess::CandidateSpec;
    using Entry = ParetoLine::Entry;
    using TimingSample = ParetoLineTestAccess::TimingSample;

    // 1. Calibrate independent Ef and Eps RecallTables over [0.40, 0.99] on smooth random data.
    const uint32_t dim = 16;
    const uint32_t count = 600;
    const auto data = make_random_data(count, dim, 23);
    auto graph = build_fp32_graph(data, dim, 8, Metric::FP32_L2);
    const std::vector<RecallTable> tables = graph.to_recall_tables(10, 80, {0.40f, 0.99f}, 0.05f, 1);
    ASSERT_EQ(tables.size(), 2u);
    const RecallTable& ef_table = tables[0];
    const RecallTable& eps_table = tables[1];

    ASSERT_GE(ef_table.entries().size(), 3u);
    ASSERT_GE(eps_table.entries().size(), 2u);
    const size_t mid_eps_idx = (std::max)(size_t{1}, eps_table.entries().size() / 2);
    const float crossover_recall = eps_table.entries()[mid_eps_idx].target_recall;
    const auto crossover_ef_opt = ef_table.param_for(crossover_recall);
    ASSERT_TRUE(crossover_ef_opt.has_value());
    const int32_t crossover_ef = static_cast<int32_t>(std::llround(*crossover_ef_opt));
    const int32_t first_ef = static_cast<int32_t>(std::llround(ef_table.entries().front().eps_or_ef));
    ASSERT_GT(crossover_ef, first_ef);

    int32_t dominated_ef = -1;
    for (const auto& e : ef_table.entries()) {
        const int32_t ef_val = static_cast<int32_t>(std::llround(e.eps_or_ef));
        if (ef_val < crossover_ef && ef_val > first_ef) {
            dominated_ef = ef_val;
        }
    }
    ASSERT_GT(dominated_ef, 0);

    auto timing_for = [&](bool use_ef, int32_t param) -> TimingSample {
        if (use_ef) {
            if (dominated_ef > 0 && param == dominated_ef) {
                return {180.0, 200.0, 0.0};  // slower than crossover eps (~105 us) -> must be Pareto-pruned
            }
            if (param >= crossover_ef) {
                return {500.0 + static_cast<double>(param), 400.0 + static_cast<double>(param), 0.0};
            }
            return {20.0 + static_cast<double>(param) * 0.5, 100.0 + static_cast<double>(param), 0.0};
        }
        return {100.0 + static_cast<double>(param) * 0.05, 150.0 + static_cast<double>(param) * 0.1, 0.0};
    };

    auto measure_single = [&](const CandidateSpec& c, size_t) -> TimingSample { return timing_for(c.use_ef, c.param); };

    auto measure_group = [&](std::span<const CandidateSpec> group, size_t) -> std::vector<TimingSample> {
        std::vector<TimingSample> out;
        out.reserve(group.size());
        for (const auto& c : group) {
            out.push_back(timing_for(c.use_ef, c.param));
        }
        return out;
    };

    const auto selected = ParetoLineTestAccess::select_entries(10, false, tables, 1, measure_single, measure_group);
    const ParetoLine synth_line = ParetoLineTestAccess::make(10, 80, tables, selected);
    verify_pareto_invariants(synth_line);

    bool has_ef = false;
    bool has_eps = false;
    bool has_dominated_ef = false;
    for (const auto& e : selected) {
        if (e.is_ef()) {
            has_ef = true;
            if (dominated_ef > 0 && static_cast<int32_t>(std::llround(e.eps_or_ef)) == dominated_ef) {
                has_dominated_ef = true;
            }
        } else {
            has_eps = true;
        }
    }
    EXPECT_TRUE(has_ef);
    EXPECT_TRUE(has_eps);
    EXPECT_FALSE(has_dominated_ef);

    // 2. Within-axis dominance pre-filter: smaller param with >= measured_recall dominates larger param in the same (mode, num_candidates) group.
    {
        auto dense_graph = build_fp32_graph(data, dim, 16, Metric::FP32_L2);
        const std::vector<float> dense_target = {0.96f};
        const std::vector<float> sparse_target = {0.90f};
        const RecallTable tight_dense = dense_graph.to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(dense_target), 80, 1);
        const RecallTable tight_sparse = graph.to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(sparse_target), 80, 1);
        ASSERT_EQ(tight_dense.size(), 1u);
        ASSERT_EQ(tight_sparse.size(), 1u);
        ASSERT_LT(tight_dense.entries()[0].eps_or_ef, tight_sparse.entries()[0].eps_or_ef);
        ASSERT_GE(tight_dense.entries()[0].measured_recall, tight_sparse.entries()[0].measured_recall);

        const float expected_min_ef = tight_dense.entries()[0].eps_or_ef;
        size_t group_calls = 0;
        size_t single_calls = 0;
        const std::vector<RecallTable> pair_ef = {tight_sparse, tight_dense};
        const auto filtered_entries = ParetoLineTestAccess::select_entries(
            10, false, pair_ef, 3,
            [&](const CandidateSpec& c, size_t) -> TimingSample {
                ++single_calls;
                return {50.0 + static_cast<double>(c.param), static_cast<double>(c.use_ef ? 100 : 200), 0.0};
            },
            [&](std::span<const CandidateSpec>, size_t) -> std::vector<TimingSample> {
                ++group_calls;
                return {};
            }
        );
        EXPECT_EQ(group_calls, 0u);
        EXPECT_EQ(single_calls, 1u);
        ASSERT_EQ(filtered_entries.size(), 1u);
        EXPECT_FLOAT_EQ(filtered_entries[0].eps_or_ef, expected_min_ef);
    }

    // 3. Verify 1% latency tie-breaker chooses lower total distance count at equal measured_recall.
    {
        const auto easy_data = make_clustered_data(150, dim, 10, 7);
        auto easy_graph = build_fp32_graph(easy_data, dim, 16, Metric::FP32_L2);
        const std::vector<float> max_target = {1.0f};
        const auto tie_tables = easy_graph.to_recall_tables(10, std::span<const float>(max_target), 30, 1);
        ASSERT_EQ(tie_tables.size(), 2u);
        ASSERT_EQ(tie_tables[0].size(), 1u);
        ASSERT_EQ(tie_tables[1].size(), 1u);
        ASSERT_NEAR(tie_tables[0].entries()[0].measured_recall, 1.0f, 1e-6f);
        ASSERT_NEAR(tie_tables[1].entries()[0].measured_recall, 1.0f, 1e-6f);

        const auto tie_entries = ParetoLineTestAccess::select_entries(
            10, false, tie_tables, 3,
            [&](const CandidateSpec& c, size_t) -> TimingSample { return c.use_ef ? TimingSample{100.0, 500.0, 0.0} : TimingSample{100.5, 250.0, 0.0}; },
            [&](std::span<const CandidateSpec> group, size_t) -> std::vector<TimingSample> {
                std::vector<TimingSample> out;
                for (const auto& c : group) {
                    out.push_back(c.use_ef ? TimingSample{100.0, 500.0, 0.0} : TimingSample{100.5, 250.0, 0.0});
                }
                return out;
            }
        );
        ASSERT_EQ(tie_entries.size(), 1u);
        EXPECT_TRUE(tie_entries[0].is_eps());
    }

    // 4. Explicit param_for tests: nearest measured_recall lookup (no interpolation, ignores target_recall, upper tie-break), clamping, params().
    {
        const std::vector<Entry> explicit_entries = {
            {0.75f, 30.0f, 1.0f, 0.80f, 50.0, 20000.0, 100.0, 0.0},    {0.80f, 50.0f, 1.0f, 0.85f, 80.0, 12500.0, 180.0, 0.0},
            {0.85f, 40.0f, 1.5f, 0.88f, 90.0, 11111.1, 150.0, 15.0},   {0.88f, 0.0200f, 1.5f, 0.90f, 100.0, 10000.0, 220.0, 15.0},
            {0.92f, 0.0600f, 1.5f, 0.95f, 140.0, 7142.8, 300.0, 15.0},
        };
        const ParetoLine explicit_line = ParetoLineTestAccess::make(10, 80, tables, explicit_entries);

        // Exact measured_recall match ignores target_recall (at 0.80, entries[0] has measured_recall=0.80 whereas entries[1] has target_recall=0.80).
        const auto p80 = explicit_line.param_for(0.80f);
        ASSERT_TRUE(p80.has_value());
        EXPECT_FLOAT_EQ(p80->first, 30.0f);
        EXPECT_FLOAT_EQ(p80->second, 1.0f);

        // Closer-lower vs closer-upper nearest measured_recall (no interpolation).
        const auto p81 = explicit_line.param_for(0.81f);
        ASSERT_TRUE(p81.has_value());
        EXPECT_FLOAT_EQ(p81->first, 30.0f);
        EXPECT_FLOAT_EQ(p81->second, 1.0f);

        const auto p84 = explicit_line.param_for(0.84f);
        ASSERT_TRUE(p84.has_value());
        EXPECT_FLOAT_EQ(p84->first, 50.0f);
        EXPECT_FLOAT_EQ(p84->second, 1.0f);

        // Exact midpoint tie breaks toward the higher-recall entry.
        const auto p825 = explicit_line.param_for(0.825f);
        ASSERT_TRUE(p825.has_value());
        EXPECT_FLOAT_EQ(p825->first, 50.0f);
        EXPECT_FLOAT_EQ(p825->second, 1.0f);

        const auto p89 = explicit_line.param_for(0.89f);
        ASSERT_TRUE(p89.has_value());
        EXPECT_FLOAT_EQ(p89->first, 0.0200f);
        EXPECT_FLOAT_EQ(p89->second, 1.5f);

        const auto p92 = explicit_line.param_for(0.92f);
        ASSERT_TRUE(p92.has_value());
        EXPECT_FLOAT_EQ(p92->first, 0.0200f);
        EXPECT_FLOAT_EQ(p92->second, 1.5f);

        const auto p93 = explicit_line.param_for(0.93f);
        ASSERT_TRUE(p93.has_value());
        EXPECT_FLOAT_EQ(p93->first, 0.0600f);
        EXPECT_FLOAT_EQ(p93->second, 1.5f);

        // Bounds checked strictly against measured_recall (0.75/0.78 < measured_recall_min=0.80 even though entries[0].target_recall == 0.75).
        EXPECT_FALSE(explicit_line.param_for(0.75f, /*clamp=*/false).has_value());
        EXPECT_FALSE(explicit_line.param_for(0.78f, /*clamp=*/false).has_value());
        const auto clamp_lo = explicit_line.param_for(0.78f, /*clamp=*/true);
        ASSERT_TRUE(clamp_lo.has_value());
        EXPECT_FLOAT_EQ(clamp_lo->first, 30.0f);
        EXPECT_FLOAT_EQ(clamp_lo->second, 1.0f);

        EXPECT_FALSE(explicit_line.param_for(0.99f, /*clamp=*/false).has_value());
        const auto clamp_hi = explicit_line.param_for(0.99f, /*clamp=*/true);
        ASSERT_TRUE(clamp_hi.has_value());
        EXPECT_FLOAT_EQ(clamp_hi->first, 0.0600f);
        EXPECT_FLOAT_EQ(clamp_hi->second, 1.5f);

        EXPECT_FALSE(explicit_line.param_for(std::numeric_limits<float>::quiet_NaN(), /*clamp=*/true).has_value());
        EXPECT_FALSE(explicit_line.param_for(std::numeric_limits<float>::infinity(), /*clamp=*/true).has_value());

        const std::vector<std::pair<float, float>> expected_pairs = {{30.0f, 1.0f}, {50.0f, 1.0f}, {40.0f, 1.5f}, {0.0200f, 1.5f}, {0.0600f, 1.5f}};
        const auto got_pairs = explicit_line.params();
        ASSERT_EQ(got_pairs.size(), expected_pairs.size());
        for (size_t i = 0; i < expected_pairs.size(); ++i) {
            EXPECT_FLOAT_EQ(got_pairs[i].first, expected_pairs[i].first);
            EXPECT_FLOAT_EQ(got_pairs[i].second, expected_pairs[i].second);
        }
    }
}

TEST(ParetoLineTest, EndToEndGraphOverloadsAndMetrics) {
    const uint32_t dim = 16;
    const uint32_t count = 400;
    const auto data = make_clustered_data(count, dim, 20, 7);

    auto mutable_graph = build_fp32_graph(data, dim, 16, Metric::FP32_L2);
    auto ro_graph = mutable_graph.to_readonly();
    auto dyn_graph = mutable_graph.to_dynamic();

    // Overload 1: from a single precomputed RecallTable (Ef-only)
    const RecallTable ef_only = ro_graph.to_recall_table(10, RecallTable::Mode::Ef, 60, {0.80f, 0.99f}, 0.05f, 1);
    const ParetoLine line_ef_only = ro_graph.to_pareto_line(ef_only);
    EXPECT_EQ(line_ef_only.k(), 10u);
    EXPECT_EQ(line_ef_only.sample_count(), 60u);
    ASSERT_EQ(line_ef_only.recall_tables().size(), 1u);
    EXPECT_EQ(line_ef_only.recall_tables()[0].mode(), RecallTable::Mode::Ef);
    verify_pareto_invariants(line_ef_only);
    for (const auto& e : line_ef_only.entries()) {
        EXPECT_TRUE(e.is_ef());
    }

    // Overload 2: from precomputed span<const RecallTable>
    const std::vector<RecallTable> precomputed = ro_graph.to_recall_tables(10, 60, {0.80f, 0.99f}, 0.05f, 1);
    const ParetoLine line_from_tables = ro_graph.to_pareto_line(std::span<const RecallTable>(precomputed));
    EXPECT_EQ(line_from_tables.k(), 10u);
    EXPECT_EQ(line_from_tables.sample_count(), 60u);
    ASSERT_EQ(line_from_tables.recall_tables().size(), 2u);
    EXPECT_EQ(line_from_tables.recall_tables()[0].entries().size(), precomputed[0].entries().size());
    EXPECT_EQ(line_from_tables.recall_tables()[1].entries().size(), precomputed[1].entries().size());
    verify_pareto_invariants(line_from_tables);

    // Overload 3: direct range + step on mutable & dynamic graphs
    const ParetoLine line_mut = mutable_graph.to_pareto_line(10, 60, {0.80f, 0.99f}, 0.05f, 1);
    ASSERT_EQ(line_mut.recall_tables().size(), 2u);
    verify_pareto_invariants(line_mut);

    const ParetoLine line_dyn = dyn_graph.to_pareto_line(10, 60, {0.80f, 0.99f}, 0.05f, 1);
    ASSERT_EQ(line_dyn.recall_tables().size(), 2u);
    verify_pareto_invariants(line_dyn);

    // Overload 4: explicit target_recalls
    const std::vector<float> custom_targets = {0.95f, 0.85f, 0.90f};
    const ParetoLine line_explicit = ro_graph.to_pareto_line(10, std::span<const float>(custom_targets), 60, 1);
    ASSERT_EQ(line_explicit.recall_tables().size(), 2u);
    verify_pareto_invariants(line_explicit);

    // InnerProduct + quantized graphs
    const uint32_t qdim = 32;
    const auto ip_data = make_random_data(count, qdim, 11, /*normalize=*/true);
    auto ip_graph = build_fp32_graph(ip_data, qdim, 16, Metric::FP32_InnerProduct, OptimizationTarget::HighLID);
    const ParetoLine ip_line = ip_graph.to_pareto_line(10, 50, {0.70f, 0.99f}, 0.10f, 1);
    verify_pareto_invariants(ip_line);

    auto int8_xfrms = deglib::transformation::make_quantizer_pipeline(ip_data, qdim, Metric::Int8_InnerProduct);
    auto int8_space = deglib::distances::FloatSpace(qdim, Metric::Int8_InnerProduct);
    auto int8_graph = ip_graph.with_transformations(int8_xfrms, int8_space);
    const ParetoLine int8_line = int8_graph.to_pareto_line(10, 50, {0.70f, 0.99f}, 0.10f, 1);
    verify_pareto_invariants(int8_line);

    const auto l2_data = make_clustered_data(count, qdim, 16, 19);
    auto l2_graph = build_fp32_graph(l2_data, qdim, 16, Metric::FP32_L2);
    auto u4_xfrms = deglib::transformation::make_quantizer_pipeline(l2_data, qdim, Metric::Uint4_L2);
    auto u4_space = deglib::distances::FloatSpace(qdim, Metric::Uint4_L2);
    auto u4_graph = l2_graph.with_transformations(u4_xfrms, u4_space);
    const ParetoLine u4_line = u4_graph.to_pareto_line(10, 50, {0.70f, 0.99f}, 0.10f, 1);
    verify_pareto_invariants(u4_line);
}

TEST(ParetoLineTest, EmptyGraphAndValidationErrors) {
    const auto space = deglib::distances::FloatSpace(8, Metric::FP32_L2);
    auto empty_graph = deglib::create_empty(16, 4, space);
    const ParetoLine empty_line = empty_graph.to_pareto_line(10);
    EXPECT_TRUE(empty_line.empty());
    EXPECT_EQ(empty_line.size(), 0u);
    EXPECT_EQ(empty_line.k(), 10u);
    EXPECT_EQ(empty_line.sample_count(), 0u);
    EXPECT_EQ(empty_line.recall_tables().size(), 2u);
    EXPECT_FALSE(empty_line.recall_bounds().has_value());
    EXPECT_FALSE(empty_line.param_for(0.95f, /*clamp=*/true).has_value());
    EXPECT_TRUE(empty_line.eps_or_ef_values().empty());
    EXPECT_FALSE(empty_line.to_string().empty());

    // Empty RecallTable / empty span passed to non-empty graph returns empty ParetoLine
    const auto tiny_data = make_random_data(15, 8, 61);
    auto tiny_graph = build_fp32_graph(tiny_data, 8, 4, Metric::FP32_L2);
    const ParetoLine from_empty_tables = tiny_graph.to_pareto_line(std::span<const RecallTable>(empty_line.recall_tables()));
    EXPECT_TRUE(from_empty_tables.empty());
    EXPECT_EQ(from_empty_tables.k(), 10u);

    const ParetoLine from_zero_tables = tiny_graph.to_pareto_line(std::span<const RecallTable>{});
    EXPECT_TRUE(from_zero_tables.empty());
    EXPECT_EQ(from_zero_tables.k(), 0u);

    // Mismatched k across tables throws invalid_argument
    const RecallTable t_k5 = tiny_graph.to_recall_table(5, RecallTable::Mode::Ef, 10, {0.80f, 0.95f}, 0.05f, 1);
    const RecallTable t_k8 = tiny_graph.to_recall_table(8, RecallTable::Mode::Eps, 10, {0.80f, 0.95f}, 0.05f, 1);
    const std::vector<RecallTable> mismatched = {t_k5, t_k8};
    EXPECT_THROW((void)tiny_graph.to_pareto_line(std::span<const RecallTable>(mismatched)), std::invalid_argument);

    // Graph smaller than sample_count and k
    const ParetoLine tiny_line = tiny_graph.to_pareto_line(50, 300);
    EXPECT_EQ(tiny_line.sample_count(), 15u);
    verify_pareto_invariants(tiny_line);
    EXPECT_FALSE(tiny_line.to_string().empty());

    // rounds == 0 throws invalid_argument across all overloads
    EXPECT_THROW((void)tiny_graph.to_pareto_line(t_k5, 0, 0), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_pareto_line(std::span<const RecallTable>(&t_k5, 1), 0, 0), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_pareto_line(5, 10, {0.80f, 0.95f}, 0.05f, 1, 0), std::invalid_argument);
    const std::vector<float> targets = {0.90f};
    EXPECT_THROW((void)tiny_graph.to_pareto_line(5, std::span<const float>(targets), 10, 1, 0), std::invalid_argument);
}

TEST(ParetoLineTest, MultiRerankDominanceTwoStageAndNoInterpolation) {
    using deglib::search::ParetoLineTestAccess;
    using CandidateSpec = ParetoLineTestAccess::CandidateSpec;
    using TimingSample = ParetoLineTestAccess::TimingSample;

    const uint32_t dim = 32;
    const uint32_t count = 600;
    const auto rand_data = make_random_data(count, dim, 23);
    auto graph_mid = build_fp32_graph(rand_data, dim, 10, Metric::FP32_L2);
    auto ro_mid = graph_mid.to_readonly();
    auto reranker = std::make_shared<deglib::search::Reranker>(
        deglib::search::Reranker::from_graph(ro_mid.internal(), deglib::distances::FloatSpace(dim, Metric::FP32_L2))
    );
    auto searcher = deglib::search::make_searcher(ro_mid.internal(), reranker);

    // 1. Per-(use_ef, num_candidates) dominance pre-filter and 2-stage global Pareto screening.
    const std::vector<float> high_target = {0.94f};
    const std::vector<float> low_target = {0.85f};
    const RecallTable t_ef_1 = searcher->to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(high_target), 1.0f, 50, 1);
    const RecallTable t_eps_1 = searcher->to_recall_table(10, RecallTable::Mode::Eps, std::span<const float>(high_target), 1.0f, 50, 1);
    const RecallTable t_ef_15 = searcher->to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(low_target), 1.5f, 50, 1);
    const RecallTable t_eps_20 = searcher->to_recall_table(10, RecallTable::Mode::Eps, std::span<const float>(low_target), 2.0f, 50, 1);
    ASSERT_EQ(t_ef_1.size(), 1u);
    ASSERT_EQ(t_eps_1.size(), 1u);
    ASSERT_EQ(t_ef_15.size(), 1u);
    ASSERT_EQ(t_eps_20.size(), 1u);
    const float min_hi_meas = (std::min)(t_ef_1.entries()[0].measured_recall, t_eps_1.entries()[0].measured_recall);
    const float max_lo_meas = (std::max)(t_ef_15.entries()[0].measured_recall, t_eps_20.entries()[0].measured_recall);
    ASSERT_GT(min_hi_meas, max_lo_meas + 1e-4f);

    {
        auto sparse_graph = build_fp32_graph(rand_data, dim, 6, Metric::FP32_L2);
        auto sparse_searcher = deglib::search::make_searcher(sparse_graph.internal(), reranker);
        const std::vector<float> sparse_target = {0.82f};
        const RecallTable t_ef_1_slow = sparse_searcher->to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(sparse_target), 1.0f, 50, 1);
        ASSERT_EQ(t_ef_1_slow.size(), 1u);
        ASSERT_LT(t_ef_1.entries()[0].eps_or_ef, t_ef_1_slow.entries()[0].eps_or_ef);
        ASSERT_GE(t_ef_1.entries()[0].measured_recall, t_ef_1_slow.entries()[0].measured_recall);

        const std::vector<RecallTable> mix = {t_ef_1_slow, t_ef_1, t_eps_1, t_ef_15, t_eps_20};
        std::vector<std::pair<size_t, size_t>> group_calls;  // (group_size, group_rounds)
        size_t single_calls = 0;

        auto latency_for = [](const CandidateSpec& c) -> double {
            if (c.use_ef && c.num_candidates == 10) return 50.0;
            if (!c.use_ef && c.num_candidates == 10) return 55.0;
            return 100.0 + static_cast<double>(c.num_candidates);  // 115.0 / 120.0 -> pruned in Stage 1 against higher-recall rf=1.0
        };

        const auto selected = ParetoLineTestAccess::select_entries(
            10, true, mix, 3,
            [&](const CandidateSpec& c, size_t r) -> TimingSample {
                ++single_calls;
                EXPECT_EQ(r, 3u);
                return {latency_for(c), 100.0, c.num_candidates > 10 ? static_cast<double>(c.num_candidates) : 0.0};
            },
            [&](std::span<const CandidateSpec> group, size_t r) -> std::vector<TimingSample> {
                group_calls.emplace_back(group.size(), r);
                std::vector<TimingSample> out;
                out.reserve(group.size());
                for (const auto& c : group) {
                    out.push_back({latency_for(c), 100.0, c.num_candidates > 10 ? static_cast<double>(c.num_candidates) : 0.0});
                }
                return out;
            }
        );

        ASSERT_EQ(group_calls.size(), 2u);
        EXPECT_EQ(group_calls[0], (std::pair<size_t, size_t>{4u, 1u}));  // Stage 1: all 4 uncached, 1 round
        EXPECT_EQ(group_calls[1], (std::pair<size_t, size_t>{2u, 3u}));  // Stage 2: only 2 rf=1.0 finalists, full 3 rounds
        EXPECT_EQ(single_calls, 0u);
        ASSERT_GE(selected.size(), 1u);
        EXPECT_TRUE(selected[0].is_ef());
        EXPECT_FLOAT_EQ(selected[0].rerank_factor, 1.0f);
    }

    // 2. No synthetic interpolation: only exact RecallTable entries (2 + 1 = 3 candidates) are evaluated.
    {
        const std::vector<float> a_targets = {0.70f, 0.92f};
        const std::vector<float> b_targets = {0.82f};
        const RecallTable t_a = searcher->to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(a_targets), 1.0f, 50, 1);
        const RecallTable t_b = searcher->to_recall_table(10, RecallTable::Mode::Eps, std::span<const float>(b_targets), 1.5f, 50, 1);
        ASSERT_EQ(t_a.size(), 2u);
        ASSERT_EQ(t_b.size(), 1u);

        const std::vector<RecallTable> pair_tables = {t_a, t_b};
        size_t evaluated_candidates = 0;
        const auto selected = ParetoLineTestAccess::select_entries(
            10, true, pair_tables, 1,
            [&](const CandidateSpec&, size_t) -> TimingSample {
                ADD_FAILURE() << "Unexpected measure_single call";
                return {100.0, 100.0, 0.0};
            },
            [&](std::span<const CandidateSpec> group, size_t) -> std::vector<TimingSample> {
                evaluated_candidates = group.size();
                std::vector<TimingSample> out;
                out.reserve(group.size());
                for (size_t i = 0; i < group.size(); ++i) {
                    out.push_back({40.0 + static_cast<double>(i) * 15.0, 100.0, group[i].num_candidates > 10 ? 15.0 : 0.0});
                }
                return out;
            }
        );
        EXPECT_EQ(evaluated_candidates, 3u);
        EXPECT_EQ(selected.size(), 3u);
    }

    // 3. Same-target preservation & 2D Pareto dominance over measured_recall.
    {
        std::optional<RecallTable> same_tgt_a;
        std::optional<RecallTable> same_tgt_b;
        for (const float t : {0.80f, 0.85f, 0.88f, 0.90f}) {
            const std::vector<float> single_t = {t};
            const RecallTable ta = searcher->to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(single_t), 1.0f, 50, 1);
            const RecallTable tb = searcher->to_recall_table(10, RecallTable::Mode::Eps, std::span<const float>(single_t), 2.0f, 50, 1);
            if (ta.size() == 1u && tb.size() == 1u && std::abs(ta.entries()[0].measured_recall - tb.entries()[0].measured_recall) > 1e-4f) {
                same_tgt_a = ta;
                same_tgt_b = tb;
                break;
            }
        }
        ASSERT_TRUE(same_tgt_a.has_value());
        ASSERT_TRUE(same_tgt_b.has_value());

        const std::vector<RecallTable> same_tgt_tables = {*same_tgt_a, *same_tgt_b};
        const float r_a = same_tgt_a->entries()[0].measured_recall;
        const float r_b = same_tgt_b->entries()[0].measured_recall;
        const bool a_is_higher = (r_a > r_b);

        // Case 3a: Higher measured_recall is faster (100 vs 120) -> dominates and prunes lower measured_recall.
        const auto pruned = ParetoLineTestAccess::select_entries(
            10, true, same_tgt_tables, 1, [&](const CandidateSpec&, size_t) -> TimingSample { return {100.0, 100.0, 0.0}; },
            [&](std::span<const CandidateSpec> group, size_t) -> std::vector<TimingSample> {
                std::vector<TimingSample> out;
                for (const auto& c : group) {
                    const bool is_high = a_is_higher ? c.use_ef : !c.use_ef;
                    out.push_back({is_high ? 100.0 : 120.0, 100.0, c.num_candidates > 10 ? 20.0 : 0.0});
                }
                return out;
            }
        );
        ASSERT_EQ(pruned.size(), 1u);
        EXPECT_FLOAT_EQ(pruned[0].measured_recall, (std::max)(r_a, r_b));

        // Case 3b: Higher measured_recall is slower (130 vs 100) -> both survive (not collapsed per target).
        const auto kept_both = ParetoLineTestAccess::select_entries(
            10, true, same_tgt_tables, 1, [&](const CandidateSpec&, size_t) -> TimingSample { return {100.0, 100.0, 0.0}; },
            [&](std::span<const CandidateSpec> group, size_t) -> std::vector<TimingSample> {
                std::vector<TimingSample> out;
                for (const auto& c : group) {
                    const bool is_high = a_is_higher ? c.use_ef : !c.use_ef;
                    out.push_back({is_high ? 130.0 : 100.0, 100.0, c.num_candidates > 10 ? 20.0 : 0.0});
                }
                return out;
            }
        );
        ASSERT_EQ(kept_both.size(), 2u);
        EXPECT_FLOAT_EQ(kept_both[0].measured_recall, (std::min)(r_a, r_b));
        EXPECT_FLOAT_EQ(kept_both[1].measured_recall, (std::max)(r_a, r_b));
    }
}

}  // namespace
