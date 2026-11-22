// test_recall_table.cpp — Unit tests for deglib::search::RecallTable and DynamicExplorationGraph::to_recall_table / to_recall_tables

#include "deglib/builder.h"
#include "deglib/deglib.h"
#include "deglib/distances.h"
#include "deglib/graph.h"
#include "deglib/search/recall_table.h"
#include "deglib/transformation.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <vector>

namespace deglib::search {
struct RecallTableTestAccess {
    using EvalResult = RecallTable::EvalResult;

    static auto sample_distinct_query_indices(const deglib::graph::InternalGraph& graph, size_t q_count, uint32_t n) {
        return RecallTable::sample_distinct_query_indices(graph, q_count, n);
    }

    static auto compute_exact_gt_thresholds(
        const deglib::graph::InternalGraph& graph,
        std::span<const void* const> query_ptrs,
        uint32_t effective_k,
        size_t actual_threads
    ) {
        return RecallTable::compute_exact_gt_thresholds(graph, query_ptrs, effective_k, actual_threads);
    }

    template <bool UseEf>
    static EvalResult evaluate_recall_batch(
        const deglib::graph::InternalGraph& graph,
        std::span<const uint32_t> sample_indices,
        std::span<const float> gt_thresholds,
        uint32_t effective_k,
        uint32_t ef,
        float eps,
        size_t eval_threads,
        double hard_limit_us
    ) {
        return RecallTable::evaluate_recall_batch<UseEf>(graph, sample_indices, gt_thresholds, effective_k, ef, eps, eval_threads, hard_limit_us);
    }

    template <typename EvalParamFn>
    static auto
    calibrate_axis(std::span<const int32_t> ladder, std::span<const float> targets, double soft_budget_us, double hard_abort_us, EvalParamFn&& eval_param) {
        return RecallTable::calibrate_axis(ladder, targets, soft_budget_us, hard_abort_us, std::forward<EvalParamFn>(eval_param));
    }

    static std::vector<RecallTable> build_modes_impl(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        std::span<const RecallTable::Mode> modes,
        std::span<const float> resolved_targets,
        size_t sample_count,
        size_t threads,
        double bf_budget_override_us = -1.0
    ) {
        return RecallTable::build_modes_impl(graph, k, modes, resolved_targets, sample_count, threads, bf_budget_override_us);
    }
};
}  // namespace deglib::search

namespace {

using deglib::builder::OptimizationTarget;
using deglib::distances::Metric;
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

// Clustered data gives smooth recall progression across ef/eps so multiple targets are achievable.
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

void verify_table_invariants(const RecallTable& table) {
    EXPECT_EQ(table.empty(), table.entries().empty());
    EXPECT_EQ(table.size(), table.entries().size());
    EXPECT_TRUE(table.mode() == RecallTable::Mode::Ef || table.mode() == RecallTable::Mode::Eps);
    EXPECT_FLOAT_EQ(table.rerank_factor(), 1.0f);

    if (table.empty()) {
        return;
    }

    ASSERT_TRUE(table.recall_bounds().has_value());
    EXPECT_LE(table.recall_bounds()->first, table.recall_bounds()->second);

    const auto& entries = table.entries();
    for (size_t i = 0; i < entries.size(); ++i) {
        if (table.mode() == RecallTable::Mode::Ef) {
            EXPECT_GE(entries[i].eps_or_ef, static_cast<float>(RecallTable::kMinEf));
            EXPECT_LE(entries[i].eps_or_ef, static_cast<float>(RecallTable::kMaxEf));
            EXPECT_LE(std::abs(entries[i].eps_or_ef - std::round(entries[i].eps_or_ef)), 1e-6f);
        } else {
            EXPECT_GE(entries[i].eps_or_ef, RecallTable::kMinEps - 1e-6f);
            EXPECT_LE(entries[i].eps_or_ef, RecallTable::kMaxEps + 1e-6f);
        }
        EXPECT_GE(entries[i].measured_recall, 0.0f);
        EXPECT_LE(entries[i].measured_recall, 1.0f);
        EXPECT_GE(entries[i].measured_recall, entries[i].target_recall - 1e-5f);
        if (i > 0) {
            EXPECT_GT(entries[i].target_recall, entries[i - 1].target_recall);
        }
    }

    const auto vals = table.eps_or_ef_values();
    EXPECT_FALSE(vals.empty());
    for (size_t i = 1; i < vals.size(); ++i) {
        EXPECT_LT(vals[i - 1] + 1e-6f, vals[i]);
    }
}

TEST(RecallTableTest, FP32L2GraphTypesAndDefaults) {
    const uint32_t dim = 16;
    const uint32_t count = 400;
    const auto data = make_clustered_data(count, dim, 20, 7);

    auto mutable_graph = build_fp32_graph(data, dim, 16, Metric::FP32_L2);
    auto ro_graph = mutable_graph.to_readonly();
    auto dyn_graph = mutable_graph.to_dynamic();

    const uint32_t k = 10;
    const size_t sample_count = 60;

    EXPECT_EQ(RecallTable::kMinEf, 10u);
    EXPECT_EQ(RecallTable::kMaxEf, 500u);
    EXPECT_EQ(RecallTable::kEfResolution, 1u);
    EXPECT_FLOAT_EQ(RecallTable::kMinEps, 0.001f);
    EXPECT_FLOAT_EQ(RecallTable::kMaxEps, 0.20f);
    EXPECT_FLOAT_EQ(RecallTable::kEpsResolution, 0.0001f);
    EXPECT_FLOAT_EQ(RecallTable::kDefaultMinRecall, 0.80f);
    EXPECT_FLOAT_EQ(RecallTable::kDefaultMaxRecall, 0.999f);
    EXPECT_FLOAT_EQ(RecallTable::kDefaultRecallStep, 0.005f);

    // Single-mode calls on ReadOnlyGraph
    const auto ef_ro = ro_graph.to_recall_table(k, RecallTable::Mode::Ef, sample_count);
    const auto eps_ro = ro_graph.to_recall_table(k, RecallTable::Mode::Eps, sample_count);
    EXPECT_EQ(ef_ro.k(), k);
    EXPECT_EQ(ef_ro.sample_count(), sample_count);
    EXPECT_EQ(ef_ro.mode(), RecallTable::Mode::Ef);
    EXPECT_EQ(eps_ro.mode(), RecallTable::Mode::Eps);
    verify_table_invariants(ef_ro);
    verify_table_invariants(eps_ro);

    // Shared build_all / to_recall_tables matches single-mode calls
    const auto tables_ro = ro_graph.to_recall_tables(k, sample_count);
    ASSERT_EQ(tables_ro.size(), 2u);
    EXPECT_EQ(tables_ro[0].mode(), RecallTable::Mode::Ef);
    EXPECT_EQ(tables_ro[1].mode(), RecallTable::Mode::Eps);
    verify_table_invariants(tables_ro[0]);
    verify_table_invariants(tables_ro[1]);
    ASSERT_EQ(tables_ro[0].entries().size(), ef_ro.entries().size());
    ASSERT_EQ(tables_ro[1].entries().size(), eps_ro.entries().size());
    for (size_t i = 0; i < ef_ro.entries().size(); ++i) {
        EXPECT_FLOAT_EQ(tables_ro[0].entries()[i].eps_or_ef, ef_ro.entries()[i].eps_or_ef);
        EXPECT_FLOAT_EQ(tables_ro[0].entries()[i].measured_recall, ef_ro.entries()[i].measured_recall);
    }
    for (size_t i = 0; i < eps_ro.entries().size(); ++i) {
        EXPECT_FLOAT_EQ(tables_ro[1].entries()[i].eps_or_ef, eps_ro.entries()[i].eps_or_ef);
        EXPECT_FLOAT_EQ(tables_ro[1].entries()[i].measured_recall, eps_ro.entries()[i].measured_recall);
    }

    // Verify 0.5% default step granularity on mutable graph over [0.80, 0.95] default step=0.005
    const auto tables_mut = mutable_graph.to_recall_tables(k, sample_count, {0.80f, 0.95f});
    ASSERT_EQ(tables_mut.size(), 2u);
    verify_table_invariants(tables_mut[0]);
    verify_table_invariants(tables_mut[1]);
    if (tables_mut[0].recall_bounds()->first <= 0.85f && tables_mut[0].recall_bounds()->second >= 0.92f) {
        bool found_half_percent_step = false;
        const auto& entries = tables_mut[0].entries();
        for (size_t i = 1; i < entries.size(); ++i) {
            if (std::abs((entries[i].target_recall - entries[i - 1].target_recall) - 0.005f) <= 1e-4f) {
                found_half_percent_step = true;
                break;
            }
        }
        EXPECT_TRUE(found_half_percent_step);
    }

    // DynamicGraph
    const auto tables_dyn = dyn_graph.to_recall_tables(k, sample_count);
    ASSERT_EQ(tables_dyn.size(), 2u);
    verify_table_invariants(tables_dyn[0]);
    verify_table_invariants(tables_dyn[1]);
}

TEST(RecallTableTest, FP32InnerProductAndQuantizedGraphs) {
    const uint32_t dim = 32;
    const uint32_t count = 400;
    const auto ip_data = make_random_data(count, dim, 11, /*normalize=*/true);

    // 1. FP32 InnerProduct (negative distances)
    auto ip_graph = build_fp32_graph(ip_data, dim, 16, Metric::FP32_InnerProduct, OptimizationTarget::HighLID);
    const auto ip_tables = ip_graph.to_recall_tables(10, 50, {0.70f, 0.99f}, 0.10f);
    ASSERT_EQ(ip_tables.size(), 2u);
    verify_table_invariants(ip_tables[0]);
    verify_table_invariants(ip_tables[1]);
    ASSERT_TRUE(ip_tables[0].recall_bounds().has_value());
    EXPECT_GT(ip_tables[0].recall_bounds()->second, 0.5f);

    // 2. Quantized Int8_InnerProduct via make_quantizer_pipeline + to_readonly
    auto int8_xfrms = deglib::transformation::make_quantizer_pipeline(ip_data, dim, Metric::Int8_InnerProduct);
    auto int8_space = deglib::distances::FloatSpace(dim, Metric::Int8_InnerProduct);
    auto int8_graph = ip_graph.to_readonly(int8_space, nullptr, int8_xfrms);
    const auto int8_tables = int8_graph.to_recall_tables(10, 50, {0.70f, 0.99f}, 0.10f);
    ASSERT_EQ(int8_tables.size(), 2u);
    verify_table_invariants(int8_tables[0]);
    verify_table_invariants(int8_tables[1]);

    // 3. Quantized Uint4_L2
    const auto l2_data = make_clustered_data(count, dim, 16, 19);
    auto l2_graph = build_fp32_graph(l2_data, dim, 16, Metric::FP32_L2);
    auto u4_xfrms = deglib::transformation::make_quantizer_pipeline(l2_data, dim, Metric::Uint4_L2);
    auto u4_space = deglib::distances::FloatSpace(dim, Metric::Uint4_L2);
    auto u4_graph = l2_graph.to_readonly(u4_space, nullptr, u4_xfrms);
    const auto u4_tables = u4_graph.to_recall_tables(10, 50, {0.70f, 0.99f}, 0.10f);
    ASSERT_EQ(u4_tables.size(), 2u);
    verify_table_invariants(u4_tables[0]);
    verify_table_invariants(u4_tables[1]);
}

TEST(RecallTableTest, ExplicitTargetsRangeStepAndInterpolation) {
    const uint32_t dim = 16;
    const uint32_t count = 500;
    const auto data = make_clustered_data(count, dim, 25, 23);
    auto graph = build_fp32_graph(data, dim, 16, Metric::FP32_L2);

    // Single-mode range + step overload
    const auto range_ef = graph.to_recall_table(10, RecallTable::Mode::Ef, 80, {0.80f, 0.99f}, 0.05f, 1);
    verify_table_invariants(range_ef);
    ASSERT_TRUE(range_ef.recall_bounds().has_value());
    const auto [r_min, r_max] = *range_ef.recall_bounds();

    // Explicit target_recalls overload with unsorted duplicates
    const std::vector<float> custom_targets = {0.95f, 0.85f, 0.95f, 0.90f};
    const auto explicit_tables = graph.to_recall_tables(10, std::span<const float>(custom_targets), 80, 1);
    ASSERT_EQ(explicit_tables.size(), 2u);
    const auto& explicit_ef = explicit_tables[0];
    verify_table_invariants(explicit_ef);
    verify_table_invariants(explicit_tables[1]);

    // Exact match for every entry target
    for (const auto& entry : explicit_ef.entries()) {
        const auto found = explicit_ef.param_for(entry.target_recall);
        ASSERT_TRUE(found.has_value());
        EXPECT_FLOAT_EQ(*found, entry.eps_or_ef);
        if (entry.target_recall >= r_min - 1e-6f && entry.target_recall <= r_max + 1e-6f) {
            EXPECT_NEAR(entry.measured_recall, entry.target_recall, 0.03f);
        }
    }

    // Lookup behavior for intermediate and out-of-bounds targets
    if (r_max - r_min > 0.02f) {
        const float mid_target = 0.5f * (r_min + r_max);
        const auto mid_ef = explicit_ef.param_for(mid_target);
        ASSERT_TRUE(mid_ef.has_value());
        EXPECT_GE(*mid_ef, static_cast<float>(RecallTable::kMinEf));
        EXPECT_LE(*mid_ef, static_cast<float>(RecallTable::kMaxEf));
    }

    if (r_min > 0.10f) {
        EXPECT_FALSE(explicit_ef.param_for(0.05f, /*clamp=*/false).has_value());
        const auto clamped_lo = explicit_ef.param_for(0.05f, /*clamp=*/true);
        ASSERT_TRUE(clamped_lo.has_value());
        EXPECT_FLOAT_EQ(*clamped_lo, static_cast<float>(RecallTable::kMinEf));
    }

    if (r_max < 0.995f) {
        EXPECT_FALSE(explicit_ef.param_for(1.0f, /*clamp=*/false).has_value());
        const auto clamped_hi = explicit_ef.param_for(1.0f, /*clamp=*/true);
        ASSERT_TRUE(clamped_hi.has_value());
        EXPECT_LE(*clamped_hi, static_cast<float>(RecallTable::kMaxEf));
    }
}

TEST(RecallTableTest, UnreachableTargetsReplacedByBoundsWithoutDuplicates) {
    // 1. Small/easy graph where ef=10 already achieves near-1.0 recall -> low targets below r_min collapse into single lower bound.
    {
        const uint32_t dim = 8;
        const uint32_t count = 120;
        const auto data = make_clustered_data(count, dim, 6, 31);
        auto easy_graph = build_fp32_graph(data, dim, 24, Metric::FP32_L2);

        const std::vector<float> low_targets = {0.10f, 0.20f, 0.30f};
        const auto table = easy_graph.to_recall_table(5, RecallTable::Mode::Ef, std::span<const float>(low_targets), 40, 1);
        ASSERT_TRUE(table.recall_bounds().has_value());
        const float r_min = table.recall_bounds()->first;
        if (r_min > 0.30f + 1e-6f) {
            ASSERT_EQ(table.entries().size(), 1u);
            EXPECT_FLOAT_EQ(table.entries()[0].eps_or_ef, static_cast<float>(RecallTable::kMinEf));
            EXPECT_FLOAT_EQ(table.entries()[0].target_recall, r_min);
            EXPECT_FLOAT_EQ(table.entries()[0].measured_recall, r_min);
        }
    }

    // 2. Hard sparse random graph where ef=500 cannot reach very high targets -> high targets collapse into single upper bound.
    {
        const uint32_t dim = 64;
        const uint32_t count = 3000;
        const auto data = make_random_data(count, dim, 37);
        const auto space = deglib::distances::FloatSpace(dim, Metric::FP32_L2);
        auto raw = deglib::create_random_graph(reinterpret_cast<const std::byte*>(data.data()), count, 6, space, 37);

        const std::vector<float> high_targets = {0.95f, 0.98f, 0.999f};
        const auto table = raw.to_recall_table(30, RecallTable::Mode::Ef, std::span<const float>(high_targets), 50, 1);
        ASSERT_TRUE(table.recall_bounds().has_value());
        const float r_max = table.recall_bounds()->second;
        ASSERT_LT(r_max, 0.94f);
        ASSERT_EQ(table.entries().size(), 1u);
        EXPECT_FLOAT_EQ(table.entries().back().target_recall, r_max);
        EXPECT_FLOAT_EQ(table.entries().back().measured_recall, r_max);

        const auto clamped_hi = table.param_for(0.999f, /*clamp=*/true);
        ASSERT_TRUE(clamped_hi.has_value());
        EXPECT_FLOAT_EQ(*clamped_hi, table.entries().back().eps_or_ef);
    }
}

TEST(RecallTableTest, BruteForceTimeGuardStopsUpwardSearchEarly) {
    const uint32_t dim = 32;
    const uint32_t count = 800;
    const auto data = make_random_data(count, dim, 41);
    const auto space = deglib::distances::FloatSpace(dim, Metric::FP32_L2);
    auto graph = deglib::create_random_graph(reinterpret_cast<const std::byte*>(data.data()), count, 12, space, 41);

    const std::vector<float> targets = {0.90f, 0.95f, 0.999f};
    const size_t q_count = 150;
    const uint32_t k = 20;

    // Measure ef=10 duration on the exact same sampled queries so the guard allows ef=10
    // yet stops before reaching larger ef values (up to 500).
    const auto sample_indices = deglib::search::RecallTableTestAccess::sample_distinct_query_indices(graph.internal(), q_count, count);
    std::vector<const void*> q_ptrs(q_count);
    for (size_t q = 0; q < q_count; ++q) {
        q_ptrs[q] = graph.internal().getFeatureVector(sample_indices[q]);
    }
    const auto thresholds = deglib::search::RecallTableTestAccess::compute_exact_gt_thresholds(graph.internal(), q_ptrs, k, 1);
    const auto ef10_eval =
        deglib::search::RecallTableTestAccess::evaluate_recall_batch<true>(graph.internal(), sample_indices, thresholds, k, 10, 0.0f, 1, 1e9);
    ASSERT_FALSE(ef10_eval.aborted);

    const double tight_budget_us = (std::max)(20.0, ef10_eval.elapsed_us * 1.10);
    const auto only_ef = std::array<RecallTable::Mode, 1>{RecallTable::Mode::Ef};
    const auto guarded_tables = deglib::search::RecallTableTestAccess::build_modes_impl(graph.internal(), k, only_ef, targets, q_count, 1, tight_budget_us);
    ASSERT_EQ(guarded_tables.size(), 1u);
    const auto& guarded_table = guarded_tables.front();

    ASSERT_EQ(guarded_table.mode(), RecallTable::Mode::Ef);
    ASSERT_FALSE(guarded_table.empty());
    EXPECT_LT(guarded_table.entries().back().eps_or_ef, static_cast<float>(RecallTable::kMaxEf));
    const auto clamped_hi = guarded_table.param_for(1.0f, /*clamp=*/true);
    ASSERT_TRUE(clamped_hi.has_value());
    EXPECT_LT(*clamped_hi, static_cast<float>(RecallTable::kMaxEf));

    // Ultra-tiny budget: hard-abort must exit cleanly without hanging
    const auto ultra_tables = deglib::search::RecallTableTestAccess::build_modes_impl(graph.internal(), k, only_ef, targets, q_count, 1, 0.01);
    ASSERT_EQ(ultra_tables.size(), 1u);
    EXPECT_EQ(ultra_tables[0].k(), k);
    EXPECT_TRUE(ultra_tables[0].empty() || ultra_tables[0].entries().back().eps_or_ef < static_cast<float>(RecallTable::kMaxEf));
}

TEST(RecallTableTest, DeterminismAndThreadEquivalence) {
    const uint32_t dim = 16;
    const uint32_t count = 350;
    const auto data = make_clustered_data(count, dim, 18, 53);
    auto graph = build_fp32_graph(data, dim, 16, Metric::FP32_L2);

    const auto t1 = graph.to_recall_tables(10, 50, {0.85f, 0.999f}, 0.05f, 1);
    const auto t2 = graph.to_recall_tables(10, 50, {0.85f, 0.999f}, 0.05f, 1);
    const auto t_mt = graph.to_recall_tables(10, 50, {0.85f, 0.999f}, 0.05f, 4);

    ASSERT_EQ(t1.size(), 2u);
    ASSERT_EQ(t2.size(), 2u);
    ASSERT_EQ(t_mt.size(), 2u);

    for (size_t m = 0; m < 2; ++m) {
        ASSERT_EQ(t1[m].entries().size(), t2[m].entries().size());
        ASSERT_EQ(t1[m].entries().size(), t_mt[m].entries().size());
        for (size_t i = 0; i < t1[m].entries().size(); ++i) {
            EXPECT_FLOAT_EQ(t1[m].entries()[i].eps_or_ef, t2[m].entries()[i].eps_or_ef);
            EXPECT_FLOAT_EQ(t1[m].entries()[i].eps_or_ef, t_mt[m].entries()[i].eps_or_ef);
            EXPECT_FLOAT_EQ(t1[m].entries()[i].measured_recall, t_mt[m].entries()[i].measured_recall);
        }
    }
}

TEST(RecallTableTest, EdgeCasesValidationAndFormatting) {
    // Empty graph returns empty RecallTable cleanly
    const auto space = deglib::distances::FloatSpace(8, Metric::FP32_L2);
    auto empty_graph = deglib::create_empty(16, 4, space);

    const auto empty_ef = empty_graph.to_recall_table(10, RecallTable::Mode::Ef);
    EXPECT_EQ(empty_ef.k(), 10u);
    EXPECT_EQ(empty_ef.sample_count(), 0u);
    EXPECT_EQ(empty_ef.mode(), RecallTable::Mode::Ef);
    EXPECT_TRUE(empty_ef.empty());
    EXPECT_EQ(empty_ef.size(), 0u);
    EXPECT_FALSE(empty_ef.recall_bounds().has_value());
    EXPECT_FALSE(empty_ef.param_for(0.95f, true).has_value());
    EXPECT_NE(empty_ef.to_string().find("mode=ef"), std::string::npos);
    EXPECT_NE(empty_ef.to_string().find("<empty>"), std::string::npos);

    const auto empty_both = empty_graph.to_recall_tables(10);
    ASSERT_EQ(empty_both.size(), 2u);
    EXPECT_EQ(empty_both[0].mode(), RecallTable::Mode::Ef);
    EXPECT_EQ(empty_both[1].mode(), RecallTable::Mode::Eps);
    EXPECT_TRUE(empty_both[0].empty());
    EXPECT_TRUE(empty_both[1].empty());
    EXPECT_NE(empty_both[1].to_string().find("mode=eps"), std::string::npos);

    // Graph smaller than requested sample_count and k
    const auto tiny_data = make_random_data(15, 8, 61);
    auto tiny_graph = build_fp32_graph(tiny_data, 8, 4, Metric::FP32_L2);
    const auto tiny_tables = tiny_graph.to_recall_tables(50, 300);
    ASSERT_EQ(tiny_tables.size(), 2u);
    EXPECT_EQ(tiny_tables[0].k(), 50u);
    EXPECT_EQ(tiny_tables[0].sample_count(), 15u);
    EXPECT_FALSE(tiny_tables[0].to_string().empty());

    // Validation errors
    EXPECT_THROW((void)tiny_graph.to_recall_table(0, RecallTable::Mode::Ef), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, RecallTable::Mode::Ef, 0), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(0), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, 0), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, 10, {0.0f, 0.95f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, 10, {0.95f, 0.90f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, 10, {0.80f, 1.05f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, 10, {std::numeric_limits<float>::quiet_NaN(), 0.95f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, 10, {0.80f, 0.95f}, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, 10, {0.80f, 0.95f}, -0.05f), std::invalid_argument);

    const std::vector<float> empty_targets;
    const std::vector<float> zero_target = {0.0f};
    const std::vector<float> over_one = {1.1f};
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, RecallTable::Mode::Eps, std::span<const float>(empty_targets)), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, std::span<const float>(empty_targets)), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, std::span<const float>(zero_target)), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_tables(5, std::span<const float>(over_one)), std::invalid_argument);
}

TEST(RecallTableTest, ExactResolutionAndUpperBoundSelection) {
    using deglib::search::RecallTableTestAccess;
    using EvalResult = RecallTableTestAccess::EvalResult;

    static constexpr int32_t kEfLadder[] = {10, 20, 40, 80, 160, 320, 500};
    static constexpr int32_t kEpsLadder[] = {10, 50, 100, 200, 400, 800, 1500, 2000};
    const std::vector<float> targets = {0.90f};

    // Case 1: Adjacent ef bracket (50 -> 0.89, 51 -> 0.91): must pick ef=51 (>= 0.90) after probing 50 and 51
    {
        std::vector<int32_t> probed;
        const auto res = RecallTableTestAccess::calibrate_axis(kEfLadder, targets, 1e9, 1e9, [&](int32_t ef, double) {
            probed.push_back(ef);
            const float r = (ef <= 50) ? 0.89f : 0.91f;
            return EvalResult{r, 1.0, false};
        });
        ASSERT_EQ(res.entries.size(), 1u);
        EXPECT_EQ(std::get<1>(res.entries[0]), 51);
        EXPECT_FLOAT_EQ(std::get<2>(res.entries[0]), 0.91f);
        EXPECT_NE(std::find(probed.begin(), probed.end(), 50), probed.end());
        EXPECT_NE(std::find(probed.begin(), probed.end(), 51), probed.end());
    }

    // Case 2: Intermediates exist (51 -> 0.894, 52 -> 0.898, 53 -> 0.901, 54 -> 0.91): must refine to ef=53
    {
        const auto res = RecallTableTestAccess::calibrate_axis(kEfLadder, targets, 1e9, 1e9, [&](int32_t ef, double) {
            float r = 0.91f;
            if (ef <= 50) {
                r = 0.89f;
            } else if (ef == 51) {
                r = 0.894f;
            } else if (ef == 52) {
                r = 0.898f;
            } else if (ef == 53) {
                r = 0.901f;
            }
            return EvalResult{r, 1.0, false};
        });
        ASSERT_EQ(res.entries.size(), 1u);
        EXPECT_EQ(std::get<1>(res.entries[0]), 53);
        EXPECT_FLOAT_EQ(std::get<2>(res.entries[0]), 0.901f);
    }

    // Case 3: Extended ef upper range (380 -> 0.895, 381 -> 0.902) up to kMaxEf=500
    {
        const auto res = RecallTableTestAccess::calibrate_axis(kEfLadder, targets, 1e9, 1e9, [&](int32_t ef, double) {
            const float r = (ef <= 380) ? 0.895f : 0.902f;
            return EvalResult{r, 1.0, false};
        });
        ASSERT_EQ(res.entries.size(), 1u);
        EXPECT_EQ(std::get<1>(res.entries[0]), 381);
        EXPECT_FLOAT_EQ(std::get<2>(res.entries[0]), 0.902f);
    }

    // Case 4: Extended eps lower range down to 0.005 (scaled 73 -> 0.899, 74 -> 0.901)
    {
        const auto res = RecallTableTestAccess::calibrate_axis(kEpsLadder, targets, 1e9, 1e9, [&](int32_t eps_scaled, double) {
            const float r = (eps_scaled <= 73) ? 0.899f : 0.901f;
            return EvalResult{r, 1.0, false};
        });
        ASSERT_EQ(res.entries.size(), 1u);
        EXPECT_EQ(std::get<1>(res.entries[0]), 74);
        EXPECT_FLOAT_EQ(std::get<2>(res.entries[0]), 0.901f);
    }

    // Case 5: eps 0.0001 resolution mid-range (scaled 437 -> 0.8998, 438 -> 0.9002)
    {
        std::vector<int32_t> probed;
        const auto res = RecallTableTestAccess::calibrate_axis(kEpsLadder, targets, 1e9, 1e9, [&](int32_t eps_scaled, double) {
            probed.push_back(eps_scaled);
            const float r = (eps_scaled <= 437) ? 0.8998f : 0.9002f;
            return EvalResult{r, 1.0, false};
        });
        ASSERT_EQ(res.entries.size(), 1u);
        EXPECT_EQ(std::get<1>(res.entries[0]), 438);
        EXPECT_FLOAT_EQ(std::get<2>(res.entries[0]), 0.9002f);
        EXPECT_NE(std::find(probed.begin(), probed.end(), 437), probed.end());
        EXPECT_NE(std::find(probed.begin(), probed.end(), 438), probed.end());
    }

    // Case 6: Sub-0.005 eps bisection down to kMinEps=0.001 (scaled 23 -> 0.895, 24 -> 0.902)
    {
        std::vector<int32_t> probed;
        const auto res = RecallTableTestAccess::calibrate_axis(kEpsLadder, targets, 1e9, 1e9, [&](int32_t eps_scaled, double) {
            probed.push_back(eps_scaled);
            const float r = (eps_scaled <= 23) ? 0.895f : 0.902f;
            return EvalResult{r, 1.0, false};
        });
        ASSERT_EQ(res.entries.size(), 1u);
        EXPECT_EQ(std::get<1>(res.entries[0]), 24);
        EXPECT_FLOAT_EQ(std::get<2>(res.entries[0]), 0.902f);
        EXPECT_NE(std::find(probed.begin(), probed.end(), 10), probed.end());
        EXPECT_NE(std::find(probed.begin(), probed.end(), 24), probed.end());
    }
}

}  // namespace
