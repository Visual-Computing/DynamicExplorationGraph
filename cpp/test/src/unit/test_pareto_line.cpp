// test_pareto_line.cpp — Unit tests for deglib::search::ParetoLine and DynamicExplorationGraph::to_pareto_line

#include "deglib/builder.h"
#include "deglib/deglib.h"
#include "deglib/distances.h"
#include "deglib/graph.h"
#include "deglib/search/pareto_line.h"
#include "deglib/transformation.h"
#include "gtest/gtest.h"

#include <algorithm>
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
    static std::vector<ParetoLine::Entry>
    select_entries(std::span<const RecallTable> recall_tables, MeasureSingleFn&& measure_single, MeasureGroupFn&& measure_group) {
        return ParetoLine::select_entries(recall_tables, std::forward<MeasureSingleFn>(measure_single), std::forward<MeasureGroupFn>(measure_group));
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
        EXPECT_GE(e.avg_distance_count, 0.0);
        if (i > 0) {
            EXPECT_GT(e.measured_recall, entries[i - 1].measured_recall);
            EXPECT_GT(e.latency_us, entries[i - 1].latency_us);
            EXPECT_LT(e.qps, entries[i - 1].qps);
        }

        const auto found = line.param_for(e.target_recall);
        ASSERT_TRUE(found.has_value());
        EXPECT_NEAR(*found, e.eps_or_ef, 1e-6f);
    }

    const auto params = line.eps_or_ef_values();
    EXPECT_FALSE(params.empty());
    EXPECT_LE(params.size(), entries.size());
    for (size_t i = 0; i < params.size(); ++i) {
        for (size_t j = i + 1; j < params.size(); ++j) {
            EXPECT_GT(std::abs(params[i] - params[j]), 1e-6f);
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
                return {180.0, 200.0};  // slower than crossover eps (~105 us) -> must be Pareto-pruned
            }
            if (param >= crossover_ef) {
                return {500.0 + static_cast<double>(param), 400.0 + static_cast<double>(param)};
            }
            return {20.0 + static_cast<double>(param) * 0.5, 100.0 + static_cast<double>(param)};
        }
        return {100.0 + static_cast<double>(param) * 0.05, 150.0 + static_cast<double>(param) * 0.1};
    };

    auto measure_group = [&](std::span<const CandidateSpec> group) -> std::vector<TimingSample> {
        std::vector<TimingSample> out;
        out.reserve(group.size());
        for (const auto& c : group) {
            out.push_back(timing_for(c.use_ef, c.param));
        }
        return out;
    };

    const auto selected = ParetoLineTestAccess::select_entries(tables, timing_for, measure_group);
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

    // 2. Same-mode dominance pre-filtering: two Mode::Ef tables at the same target -> smaller ef kept without group timing.
    {
        auto mid_graph = build_fp32_graph(data, dim, 12, Metric::FP32_L2);
        const std::vector<float> target_vec = {0.95f};
        const RecallTable tight_mid = mid_graph.to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(target_vec), 80, 1);
        const RecallTable tight_sparse = graph.to_recall_table(10, RecallTable::Mode::Ef, std::span<const float>(target_vec), 80, 1);
        ASSERT_EQ(tight_mid.size(), 1u);
        ASSERT_EQ(tight_sparse.size(), 1u);
        ASSERT_FLOAT_EQ(tight_mid.entries()[0].target_recall, 0.95f);
        ASSERT_FLOAT_EQ(tight_sparse.entries()[0].target_recall, 0.95f);
        ASSERT_LT(tight_mid.entries()[0].eps_or_ef, tight_sparse.entries()[0].eps_or_ef);

        const float expected_min_ef = tight_mid.entries()[0].eps_or_ef;
        size_t group_calls = 0;
        size_t single_calls = 0;
        const std::vector<RecallTable> pair_ef = {tight_sparse, tight_mid};
        const auto filtered_entries = ParetoLineTestAccess::select_entries(
            pair_ef,
            [&](bool use_ef, int32_t param) -> TimingSample {
                ++single_calls;
                return {50.0 + static_cast<double>(param), static_cast<double>(use_ef ? 100 : 200)};
            },
            [&](std::span<const CandidateSpec>) -> std::vector<TimingSample> {
                ++group_calls;
                return {};
            }
        );
        EXPECT_EQ(group_calls, 0u);
        EXPECT_EQ(single_calls, 1u);
        ASSERT_EQ(filtered_entries.size(), 1u);
        EXPECT_FLOAT_EQ(filtered_entries[0].eps_or_ef, expected_min_ef);
    }

    // 3. Verify 1% latency tie-breaker chooses lower avg_distance_count.
    {
        const auto tie_tables = graph.to_recall_tables(10, std::vector<float>{eps_table.entries().front().target_recall}, 80, 1);
        const auto tie_entries = ParetoLineTestAccess::select_entries(
            tie_tables, [&](bool use_ef, int32_t) -> TimingSample { return use_ef ? TimingSample{100.0, 500.0} : TimingSample{100.5, 250.0}; },
            [&](std::span<const CandidateSpec> group) -> std::vector<TimingSample> {
                std::vector<TimingSample> out;
                for (const auto& c : group) {
                    out.push_back(c.use_ef ? TimingSample{100.0, 500.0} : TimingSample{100.5, 250.0});
                }
                return out;
            }
        );
        ASSERT_EQ(tie_entries.size(), 1u);
        EXPECT_TRUE(tie_entries[0].is_eps());
    }

    // 4. Explicit param_for tests: exact match, same-regime interpolation, regime boundary (no mixed interpolation), clamping.
    {
        const std::vector<Entry> explicit_entries = {
            {0.80f, 30.0f, 0.80f, 50.0, 20000.0, 100.0},
            {0.85f, 50.0f, 0.85f, 80.0, 12500.0, 180.0},
            {0.90f, 0.0200f, 0.90f, 100.0, 10000.0, 220.0},
            {0.95f, 0.0600f, 0.95f, 140.0, 7142.8, 300.0},
        };
        const ParetoLine explicit_line = ParetoLineTestAccess::make(10, 80, tables, explicit_entries);
        EXPECT_FLOAT_EQ(*explicit_line.param_for(0.80f), 30.0f);
        EXPECT_FLOAT_EQ(*explicit_line.param_for(0.825f), 40.0f);    // ef-ef ceiling interpolation
        EXPECT_FLOAT_EQ(*explicit_line.param_for(0.875f), 0.0200f);  // regime boundary ef -> eps picks upper eps
        EXPECT_FLOAT_EQ(*explicit_line.param_for(0.925f), 0.0400f);  // eps-eps ceiling interpolation

        EXPECT_FALSE(explicit_line.param_for(0.70f, /*clamp=*/false).has_value());
        EXPECT_FLOAT_EQ(*explicit_line.param_for(0.70f, /*clamp=*/true), 30.0f);
        EXPECT_FALSE(explicit_line.param_for(0.99f, /*clamp=*/false).has_value());
        EXPECT_FLOAT_EQ(*explicit_line.param_for(0.99f, /*clamp=*/true), 0.0600f);

        const std::vector<float> expected_order = {30.0f, 50.0f, 0.0200f, 0.0600f};
        const auto got = explicit_line.eps_or_ef_values();
        ASSERT_EQ(got.size(), expected_order.size());
        for (size_t i = 0; i < expected_order.size(); ++i) {
            EXPECT_FLOAT_EQ(got[i], expected_order[i]);
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
    auto int8_graph = ip_graph.to_readonly(int8_space, nullptr, int8_xfrms);
    const ParetoLine int8_line = int8_graph.to_pareto_line(10, 50, {0.70f, 0.99f}, 0.10f, 1);
    verify_pareto_invariants(int8_line);

    const auto l2_data = make_clustered_data(count, qdim, 16, 19);
    auto l2_graph = build_fp32_graph(l2_data, qdim, 16, Metric::FP32_L2);
    auto u4_xfrms = deglib::transformation::make_quantizer_pipeline(l2_data, qdim, Metric::Uint4_L2);
    auto u4_space = deglib::distances::FloatSpace(qdim, Metric::Uint4_L2);
    auto u4_graph = l2_graph.to_readonly(u4_space, nullptr, u4_xfrms);
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

}  // namespace
