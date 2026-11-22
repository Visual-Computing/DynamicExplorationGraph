// test_recall_table.cpp — Unit tests for deglib::search::RecallTable and DynamicExplorationGraph::to_recall_table

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
    if (table.has_ef() && !table.ef_entries().empty()) {
        const auto& entries = table.ef_entries();
        for (size_t i = 0; i < entries.size(); ++i) {
            EXPECT_GE(entries[i].ef, RecallTable::kMinEf);
            EXPECT_LE(entries[i].ef, RecallTable::kMaxEf);
            EXPECT_GE(entries[i].measured_recall, 0.0f);
            EXPECT_LE(entries[i].measured_recall, 1.0f);
            EXPECT_GE(entries[i].measured_recall, entries[i].target_recall - 1e-5f);
            if (i > 0) {
                EXPECT_GT(entries[i].target_recall, entries[i - 1].target_recall);
            }
        }
        const auto efs = table.ef_values();
        EXPECT_FALSE(efs.empty());
        for (size_t i = 1; i < efs.size(); ++i) {
            EXPECT_LT(efs[i - 1], efs[i]);
        }
    }

    if (table.has_eps() && !table.eps_entries().empty()) {
        const auto& entries = table.eps_entries();
        for (size_t i = 0; i < entries.size(); ++i) {
            EXPECT_GE(entries[i].eps, RecallTable::kMinEps - 1e-6f);
            EXPECT_LE(entries[i].eps, RecallTable::kMaxEps + 1e-6f);
            EXPECT_GE(entries[i].measured_recall, 0.0f);
            EXPECT_LE(entries[i].measured_recall, 1.0f);
            EXPECT_GE(entries[i].measured_recall, entries[i].target_recall - 1e-5f);
            if (i > 0) {
                EXPECT_GT(entries[i].target_recall, entries[i - 1].target_recall);
            }
        }
        const auto epss = table.eps_values();
        EXPECT_FALSE(epss.empty());
        for (size_t i = 1; i < epss.size(); ++i) {
            EXPECT_LT(epss[i - 1], epss[i]);
        }
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

    EXPECT_FLOAT_EQ(RecallTable::kDefaultMinRecall, 0.80f);
    EXPECT_FLOAT_EQ(RecallTable::kDefaultMaxRecall, 0.999f);
    EXPECT_FLOAT_EQ(RecallTable::kDefaultRecallStep, 0.005f);

    // Default call on ReadOnlyGraph: calibrates both ef and eps with default range=(0.80, 0.999), step=0.005
    const auto table_ro = ro_graph.to_recall_table(k, sample_count);
    EXPECT_EQ(table_ro.k(), k);
    EXPECT_EQ(table_ro.sample_count(), sample_count);
    EXPECT_TRUE(table_ro.has_ef());
    EXPECT_TRUE(table_ro.has_eps());
    ASSERT_TRUE(table_ro.ef_recall_bounds().has_value());
    ASSERT_TRUE(table_ro.eps_recall_bounds().has_value());
    EXPECT_LE(table_ro.ef_recall_bounds()->first, table_ro.ef_recall_bounds()->second);
    EXPECT_LE(table_ro.eps_recall_bounds()->first, table_ro.eps_recall_bounds()->second);
    verify_table_invariants(table_ro);

    // Combined eps_or_ef_values preserves ef_values followed by eps_values
    const auto combined = table_ro.eps_or_ef_values();
    EXPECT_EQ(combined.size(), table_ro.ef_values().size() + table_ro.eps_values().size());
    for (size_t i = 0; i < table_ro.ef_values().size(); ++i) {
        EXPECT_FLOAT_EQ(combined[i], static_cast<float>(table_ro.ef_values()[i]));
    }
    for (size_t i = 0; i < table_ro.eps_values().size(); ++i) {
        EXPECT_FLOAT_EQ(combined[table_ro.ef_values().size() + i], table_ro.eps_values()[i]);
    }

    // Verify 0.5% default step granularity on mutable graph over [0.80, 0.95] default step=0.005
    const auto table_mut = mutable_graph.to_recall_table(k, sample_count, {0.80f, 0.95f});
    EXPECT_TRUE(table_mut.has_ef());
    EXPECT_TRUE(table_mut.has_eps());
    verify_table_invariants(table_mut);
    if (table_mut.ef_recall_bounds()->first <= 0.85f && table_mut.ef_recall_bounds()->second >= 0.92f) {
        bool found_half_percent_step = false;
        const auto& entries = table_mut.ef_entries();
        for (size_t i = 1; i < entries.size(); ++i) {
            if (std::abs((entries[i].target_recall - entries[i - 1].target_recall) - 0.005f) <= 1e-4f) {
                found_half_percent_step = true;
                break;
            }
        }
        EXPECT_TRUE(found_half_percent_step);
    }

    // DynamicGraph
    const auto table_dyn = dyn_graph.to_recall_table(k, sample_count);
    EXPECT_TRUE(table_dyn.has_ef());
    EXPECT_TRUE(table_dyn.has_eps());
    verify_table_invariants(table_dyn);
}

TEST(RecallTableTest, FP32InnerProductAndQuantizedGraphs) {
    const uint32_t dim = 32;
    const uint32_t count = 400;
    const auto ip_data = make_random_data(count, dim, 11, /*normalize=*/true);

    // 1. FP32 InnerProduct (negative distances)
    auto ip_graph = build_fp32_graph(ip_data, dim, 16, Metric::FP32_InnerProduct, OptimizationTarget::HighLID);
    const auto ip_table = ip_graph.to_recall_table(10, 50, {0.70f, 0.99f}, 0.10f);
    EXPECT_TRUE(ip_table.has_ef());
    EXPECT_TRUE(ip_table.has_eps());
    verify_table_invariants(ip_table);
    ASSERT_TRUE(ip_table.ef_recall_bounds().has_value());
    EXPECT_GT(ip_table.ef_recall_bounds()->second, 0.5f);

    // 2. Quantized Int8_InnerProduct via make_quantizer_pipeline + to_readonly
    auto int8_xfrms = deglib::transformation::make_quantizer_pipeline(ip_data, dim, Metric::Int8_InnerProduct);
    auto int8_space = deglib::distances::FloatSpace(dim, Metric::Int8_InnerProduct);
    auto int8_graph = ip_graph.to_readonly(int8_space, nullptr, int8_xfrms);
    const auto int8_table = int8_graph.to_recall_table(10, 50, {0.70f, 0.99f}, 0.10f);
    EXPECT_TRUE(int8_table.has_ef());
    EXPECT_TRUE(int8_table.has_eps());
    verify_table_invariants(int8_table);

    // 3. Quantized Uint4_L2
    const auto l2_data = make_clustered_data(count, dim, 16, 19);
    auto l2_graph = build_fp32_graph(l2_data, dim, 16, Metric::FP32_L2);
    auto u4_xfrms = deglib::transformation::make_quantizer_pipeline(l2_data, dim, Metric::Uint4_L2);
    auto u4_space = deglib::distances::FloatSpace(dim, Metric::Uint4_L2);
    auto u4_graph = l2_graph.to_readonly(u4_space, nullptr, u4_xfrms);
    const auto u4_table = u4_graph.to_recall_table(10, 50, {0.70f, 0.99f}, 0.10f);
    EXPECT_TRUE(u4_table.has_ef());
    EXPECT_TRUE(u4_table.has_eps());
    verify_table_invariants(u4_table);
}

TEST(RecallTableTest, ExplicitTargetsRangeStepAndInterpolation) {
    const uint32_t dim = 16;
    const uint32_t count = 500;
    const auto data = make_clustered_data(count, dim, 25, 23);
    auto graph = build_fp32_graph(data, dim, 16, Metric::FP32_L2);

    // Range + step overload
    const auto range_table = graph.to_recall_table(10, 80, {0.80f, 0.99f}, 0.05f, 1);
    verify_table_invariants(range_table);
    ASSERT_TRUE(range_table.ef_recall_bounds().has_value());
    const auto [r_min, r_max] = *range_table.ef_recall_bounds();

    // Explicit target_recalls overload with unsorted duplicates
    const std::vector<float> custom_targets = {0.95f, 0.85f, 0.95f, 0.90f};
    const auto explicit_table = graph.to_recall_table(10, std::span<const float>(custom_targets), 80, 1);
    verify_table_invariants(explicit_table);

    // Exact match for every entry target
    for (const auto& entry : explicit_table.ef_entries()) {
        const auto found = explicit_table.ef_for(entry.target_recall);
        ASSERT_TRUE(found.has_value());
        EXPECT_EQ(*found, entry.ef);
        // Measured recall must stay close to achievable targets
        if (entry.target_recall >= r_min - 1e-6f && entry.target_recall <= r_max + 1e-6f) {
            EXPECT_NEAR(entry.measured_recall, entry.target_recall, 0.03f);
        }
    }

    // Lookup behavior for intermediate and out-of-bounds targets
    if (r_max - r_min > 0.02f) {
        const float mid_target = 0.5f * (r_min + r_max);
        const auto mid_ef = explicit_table.ef_for(mid_target);
        ASSERT_TRUE(mid_ef.has_value());
        EXPECT_GE(*mid_ef, RecallTable::kMinEf);
        EXPECT_LE(*mid_ef, RecallTable::kMaxEf);
    }

    if (r_min > 0.10f) {
        EXPECT_FALSE(explicit_table.ef_for(0.05f, /*clamp=*/false).has_value());
        const auto clamped_lo = explicit_table.ef_for(0.05f, /*clamp=*/true);
        ASSERT_TRUE(clamped_lo.has_value());
        EXPECT_EQ(*clamped_lo, RecallTable::kMinEf);
    }

    if (r_max < 0.995f) {
        EXPECT_FALSE(explicit_table.ef_for(1.0f, /*clamp=*/false).has_value());
        const auto clamped_hi = explicit_table.ef_for(1.0f, /*clamp=*/true);
        ASSERT_TRUE(clamped_hi.has_value());
        EXPECT_LE(*clamped_hi, RecallTable::kMaxEf);
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
        const auto table = easy_graph.to_recall_table(5, std::span<const float>(low_targets), 40, 1);
        ASSERT_TRUE(table.ef_recall_bounds().has_value());
        const float r_min = table.ef_recall_bounds()->first;
        if (r_min > 0.30f + 1e-6f) {
            ASSERT_EQ(table.ef_entries().size(), 1u);
            EXPECT_EQ(table.ef_entries()[0].ef, RecallTable::kMinEf);
            EXPECT_FLOAT_EQ(table.ef_entries()[0].target_recall, r_min);
            EXPECT_FLOAT_EQ(table.ef_entries()[0].measured_recall, r_min);
        }
    }

    // 2. Hard sparse random graph where ef=300 cannot reach very high targets -> high targets collapse into single upper bound.
    {
        const uint32_t dim = 32;
        const uint32_t count = 2000;
        const auto data = make_random_data(count, dim, 37);
        const auto space = deglib::distances::FloatSpace(dim, Metric::FP32_L2);
        auto raw = deglib::create_random_graph(reinterpret_cast<const std::byte*>(data.data()), count, 6, space, 37);

        const std::vector<float> high_targets = {0.95f, 0.98f, 0.999f};
        const auto table = raw.to_recall_table(30, std::span<const float>(high_targets), 50, 1);
        ASSERT_TRUE(table.ef_recall_bounds().has_value());
        const float r_max = table.ef_recall_bounds()->second;
        ASSERT_LT(r_max, 0.94f);
        // Should contain at most 1 bound entry rather than 3 unachievable entries
        ASSERT_EQ(table.ef_entries().size(), 1u);
        EXPECT_FLOAT_EQ(table.ef_entries().back().target_recall, r_max);
        EXPECT_FLOAT_EQ(table.ef_entries().back().measured_recall, r_max);

        // Clamp=true must match upper-bound entry
        const auto clamped_hi = table.ef_for(0.999f, /*clamp=*/true);
        ASSERT_TRUE(clamped_hi.has_value());
        EXPECT_EQ(*clamped_hi, table.ef_entries().back().ef);
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
    // yet stops before reaching larger ef values (such as 160 or 300).
    const auto sample_indices = deglib::search::detail::sample_distinct_query_indices(graph.internal(), q_count, count);
    std::vector<const void*> q_ptrs(q_count);
    for (size_t q = 0; q < q_count; ++q) {
        q_ptrs[q] = graph.internal().getFeatureVector(sample_indices[q]);
    }
    const auto thresholds = deglib::search::detail::compute_exact_gt_thresholds(graph.internal(), q_ptrs, k, 1);
    const auto ef10_eval = deglib::search::detail::evaluate_recall_batch<true>(graph.internal(), sample_indices, thresholds, k, 10, 0.0f, 1, 1e9);
    ASSERT_FALSE(ef10_eval.aborted);

    // Set bf_budget so soft_budget = 0.75 * bf_budget <= ef10_eval.elapsed_us and hard_abort = bf_budget > ef10_eval.elapsed_us * 0.5
    const double tight_budget_us = (std::max)(20.0, ef10_eval.elapsed_us * 1.10);
    const auto guarded_table = deglib::search::detail::build_recall_table_impl(graph.internal(), k, targets, q_count, 1, tight_budget_us);

    ASSERT_TRUE(guarded_table.has_ef());
    ASSERT_FALSE(guarded_table.ef_entries().empty());
    // Must stop well short of kMaxEf=300
    EXPECT_LT(guarded_table.ef_entries().back().ef, RecallTable::kMaxEf);
    const auto clamped_hi = guarded_table.ef_for(1.0f, /*clamp=*/true);
    ASSERT_TRUE(clamped_hi.has_value());
    EXPECT_LT(*clamped_hi, RecallTable::kMaxEf);

    // Ultra-tiny budget: hard-abort must exit cleanly without hanging
    const auto ultra_tight = deglib::search::detail::build_recall_table_impl(graph.internal(), k, targets, q_count, 1, 0.01);
    EXPECT_EQ(ultra_tight.k(), k);
    EXPECT_TRUE(ultra_tight.ef_entries().empty() || ultra_tight.ef_entries().back().ef < RecallTable::kMaxEf);
}

TEST(RecallTableTest, DeterminismAndThreadEquivalence) {
    const uint32_t dim = 16;
    const uint32_t count = 350;
    const auto data = make_clustered_data(count, dim, 18, 53);
    auto graph = build_fp32_graph(data, dim, 16, Metric::FP32_L2);

    const auto t1 = graph.to_recall_table(10, 50, {0.85f, 0.999f}, 0.05f, 1);
    const auto t2 = graph.to_recall_table(10, 50, {0.85f, 0.999f}, 0.05f, 1);
    const auto t_mt = graph.to_recall_table(10, 50, {0.85f, 0.999f}, 0.05f, 4);

    ASSERT_EQ(t1.ef_entries().size(), t2.ef_entries().size());
    ASSERT_EQ(t1.ef_entries().size(), t_mt.ef_entries().size());
    for (size_t i = 0; i < t1.ef_entries().size(); ++i) {
        EXPECT_EQ(t1.ef_entries()[i].ef, t2.ef_entries()[i].ef);
        EXPECT_EQ(t1.ef_entries()[i].ef, t_mt.ef_entries()[i].ef);
        EXPECT_FLOAT_EQ(t1.ef_entries()[i].measured_recall, t_mt.ef_entries()[i].measured_recall);
    }

    ASSERT_EQ(t1.eps_entries().size(), t2.eps_entries().size());
    ASSERT_EQ(t1.eps_entries().size(), t_mt.eps_entries().size());
    for (size_t i = 0; i < t1.eps_entries().size(); ++i) {
        EXPECT_FLOAT_EQ(t1.eps_entries()[i].eps, t2.eps_entries()[i].eps);
        EXPECT_FLOAT_EQ(t1.eps_entries()[i].eps, t_mt.eps_entries()[i].eps);
        EXPECT_FLOAT_EQ(t1.eps_entries()[i].measured_recall, t_mt.eps_entries()[i].measured_recall);
    }
}

TEST(RecallTableTest, EdgeCasesValidationAndFormatting) {
    // Empty graph returns empty RecallTable cleanly
    const auto space = deglib::distances::FloatSpace(8, Metric::FP32_L2);
    auto empty_graph = deglib::create_empty(16, 4, space);
    const auto empty_table = empty_graph.to_recall_table(10);
    EXPECT_EQ(empty_table.k(), 10u);
    EXPECT_EQ(empty_table.sample_count(), 0u);
    EXPECT_FALSE(empty_table.has_ef());
    EXPECT_FALSE(empty_table.has_eps());
    EXPECT_FALSE(empty_table.ef_for(0.95f, true).has_value());
    EXPECT_FALSE(empty_table.eps_for(0.95f, true).has_value());
    EXPECT_FALSE(empty_table.to_string().empty());

    // Graph smaller than requested sample_count and k
    const auto tiny_data = make_random_data(15, 8, 61);
    auto tiny_graph = build_fp32_graph(tiny_data, 8, 4, Metric::FP32_L2);
    const auto tiny_table = tiny_graph.to_recall_table(50, 300);
    EXPECT_EQ(tiny_table.k(), 50u);
    EXPECT_EQ(tiny_table.sample_count(), 15u);
    EXPECT_FALSE(tiny_table.to_string().empty());

    // Validation errors
    EXPECT_THROW((void)tiny_graph.to_recall_table(0), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, 0), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, 10, {0.0f, 0.95f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, 10, {0.95f, 0.90f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, 10, {0.80f, 1.05f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, 10, {std::numeric_limits<float>::quiet_NaN(), 0.95f}, 0.05f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, 10, {0.80f, 0.95f}, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, 10, {0.80f, 0.95f}, -0.05f), std::invalid_argument);

    const std::vector<float> empty_targets;
    const std::vector<float> zero_target = {0.0f};
    const std::vector<float> over_one = {1.1f};
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, std::span<const float>(empty_targets)), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, std::span<const float>(zero_target)), std::invalid_argument);
    EXPECT_THROW((void)tiny_graph.to_recall_table(5, std::span<const float>(over_one)), std::invalid_argument);
}

TEST(RecallTableTest, ExactResolutionAndUpperBoundSelection) {
    using deglib::search::detail::calibrate_axis;
    using deglib::search::detail::EvalResult;

    static constexpr int32_t kEfLadder[] = {10, 20, 40, 80, 160, 300};
    static constexpr int32_t kEpsLadder[] = {100, 200, 400, 800, 1500, 2000};
    const std::vector<float> targets = {0.90f};

    // Case 1: Adjacent ef bracket (50 -> 0.89, 51 -> 0.91): must pick ef=51 (>= 0.90) after probing 50 and 51
    {
        std::vector<int32_t> probed;
        const auto res = calibrate_axis(kEfLadder, targets, 1e9, 1e9, [&](int32_t ef, double) {
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
        const auto res = calibrate_axis(kEfLadder, targets, 1e9, 1e9, [&](int32_t ef, double) {
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

    // Case 3: eps 0.0001 resolution (scaled 437 -> 0.8998, 438 -> 0.9002): must bisect down to 438 (0.0438)
    {
        std::vector<int32_t> probed;
        const auto res = calibrate_axis(kEpsLadder, targets, 1e9, 1e9, [&](int32_t eps_scaled, double) {
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
}

}  // namespace
