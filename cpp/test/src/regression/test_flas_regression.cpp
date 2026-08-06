#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cmath>

#include "deglib/flas/fast_linear_assignment_sorter.h"
#include "deglib/flas/fast_linear_assignment_sorter_mt.h"
#include "common/test_helpers.h"

// ============================================================================
// FLAS Regression Tests
// ============================================================================
// Measures FLAS sorting runtime and grid topology quality, and compares
// building and searching a DEG graph on default raw data vs FLAS 1D
// pre-sorted data to measure build speed improvement and search QPS/accuracy
// due to improved cache locality from memory-aligned layout.
// ============================================================================

// ---------------------------------------------------------------------------
// Helper: compute 1D locality metric — average distance between consecutive
// elements in the sorted 1D grid. Lower values indicate better locality.
// Uses L2 distance on the original feature vectors.
// ---------------------------------------------------------------------------
static float compute_1d_locality(const std::vector<float>& features, int N, int dim,
                                 const std::vector<int>& sorted_ids) {
    const size_t dim_sz = static_cast<size_t>(dim);
    float total_dist = 0.0f;
    int count = 0;
    for (int i = 0; i < N - 1; i++) {
        const float* a = &features[static_cast<size_t>(sorted_ids[i]) * dim_sz];
        const float* b = &features[static_cast<size_t>(sorted_ids[i + 1]) * dim_sz];
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = a[d] - b[d];
            dist += diff * diff;
        }
        total_dist += std::sqrt(dist);
        count++;
    }
    return total_dist / static_cast<float>(count);
}

// ---------------------------------------------------------------------------
// Helper: run FLAS sorting and return the sorted ID permutation
// ---------------------------------------------------------------------------
static std::vector<int> run_flas_sort(const float* features, int N, int dim,
                                      unsigned int seed, const flas::FlasSettings& settings) {
    auto map_fields = flas::make_map_fields(features, N, dim);

    flas::RandomEngine rng(seed);
    flas::do_sorting_1d(map_fields, dim, settings, rng, [](float) { return false; });

    std::vector<int> result(N);
    for (int i = 0; i < N; ++i) result[i] = map_fields[i].id;
    return result;
}

// ---------------------------------------------------------------------------
// Helper: build a DEG graph and measure build time
// Returns build time in seconds and the built graph
// ---------------------------------------------------------------------------
static double build_deg_graph(deglib::graph::SizeBoundedGraph& graph,
                              const std::vector<std::byte>& feature_bytes,
                              size_t feature_bytes_per_vec,
                              size_t base_count,
                              deglib::builder::OptimizationTarget optimization_target,
                              uint32_t /*edges_per_vertex*/,
                              uint8_t extend_k,
                              float extend_eps) {
    std::mt19937 rng(1337);
    const uint8_t improve_k = 0;
    const float improve_eps = 0.0f;
    const uint8_t max_path_length = 5;
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;

    deglib::builder::EvenRegularGraphBuilder builder(graph, rng, optimization_target,
                                                     extend_k, extend_eps, improve_k, improve_eps,
                                                     max_path_length, swap_tries, additional_swap_tries);
    builder.setThreadCount(1);

    for (size_t i = 0; i < base_count; ++i) {
        const std::byte* ptr = &feature_bytes[i * feature_bytes_per_vec];
        std::vector<std::byte> feat_vec(ptr, ptr + feature_bytes_per_vec);
        builder.addEntry(static_cast<uint32_t>(i), std::move(feat_vec));
    }

    auto build_callback = [](deglib::builder::BuilderStatus&) {};

    auto t_build_start = std::chrono::high_resolution_clock::now();
    builder.build(build_callback);
    auto t_build_end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(t_build_end - t_build_start).count();
}

// ---------------------------------------------------------------------------
// Helper: search a DEG graph and return QPS and recall
// ---------------------------------------------------------------------------
static std::pair<double, double> search_deg_graph(
    const deglib::graph::SizeBoundedGraph& graph,
    const std::vector<std::byte>& query_bytes,
    size_t feature_bytes_per_vec,
    size_t query_count,
    const std::vector<std::vector<uint32_t>>& gt_data,
    uint32_t search_k,
    float search_eps) {

    auto entry_vertex_indices = graph.getEntryVertexIndices();

    size_t total_correct = 0;
    auto t_search_start = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < query_count; ++q) {
        const std::byte* q_ptr = &query_bytes[q * feature_bytes_per_vec];
        auto result = graph.search(entry_vertex_indices, q_ptr, search_eps, search_k, nullptr, 0);

        std::unordered_set<uint32_t> gt_set;
        if (!gt_data.empty() && q < gt_data.size()) {
            size_t eval_k = std::min(static_cast<size_t>(search_k), gt_data[q].size());
            for (size_t i = 0; i < eval_k; ++i) {
                gt_set.insert(gt_data[q][i]);
            }
        }

        while (!result.empty()) {
            auto top_item = result.top();
            result.pop();
            uint32_t ext_label = graph.getExternalLabel(top_item.getInternalIndex());
            if (gt_set.count(ext_label)) {
                total_correct++;
            }
        }
    }

    auto t_search_end = std::chrono::high_resolution_clock::now();
    double search_secs = std::chrono::duration<double>(t_search_end - t_search_start).count();
    double qps = static_cast<double>(query_count) / search_secs;
    double recall = static_cast<double>(total_correct) / static_cast<double>(query_count * search_k);

    return {qps, recall};
}

// ============================================================================
// FLAS Sorting Runtime and Quality Tests
// ============================================================================

TEST(FlasRegression, SortingRuntimeAndQuality_1D_N500_D8) {
    const int N = 500;
    const int D = 8;
    const size_t dim = D;

    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : features) v = dist(rng);

    flas::FlasSettings settings = flas::FlasSettings{};

    // Measure FLAS sorting runtime
    auto t_start = std::chrono::high_resolution_clock::now();
    auto sorted_ids = run_flas_sort(features.data(), N, D, 42, settings);
    auto t_end = std::chrono::high_resolution_clock::now();
    double sort_secs = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "[FLAS_Sort_1D_N500_D8] sort_secs=" << sort_secs << std::endl;

    // Verify permutation validity
    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        ASSERT_GE(sorted_ids[i], 0);
        ASSERT_LT(sorted_ids[i], N);
        EXPECT_FALSE(seen[sorted_ids[i]]);
        seen[sorted_ids[i]] = true;
    }

    // Compute locality metrics
    float flas_locality = compute_1d_locality(features, N, D, sorted_ids);

    // Compute raw (unsorted) locality for comparison
    std::vector<int> raw_ids(N);
    for (int i = 0; i < N; ++i) raw_ids[i] = i;
    float raw_locality = compute_1d_locality(features, N, D, raw_ids);

    std::cout << "[FLAS_Sort_1D_N500_D8] flas_locality=" << flas_locality
              << "  raw_locality=" << raw_locality
              << "  improvement=" << (raw_locality / flas_locality) << "x" << std::endl;

    // FLAS should produce a permutation (all elements present)
    EXPECT_EQ(sorted_ids.size(), static_cast<size_t>(N));

    // FLAS sorting should complete in reasonable time
    if (std::getenv("SKIP_PERFORMANCE_TESTS") == nullptr) {
        EXPECT_LT(sort_secs, 10.0);
    }
}

TEST(FlasRegression, SortingRuntimeAndQuality_2D_N500_D8) {
    const int cols = 25;
    const int rows = 20;
    const int N = cols * rows;
    const int D = 8;

    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : features) v = dist(rng);

    auto map_fields = flas::make_map_fields(features.data(), N, D);

    flas::FlasSettings settings = flas::FlasSettings{};
    flas::RandomEngine flas_rng(42);

    auto t_start = std::chrono::high_resolution_clock::now();
    flas::do_sorting_1d(map_fields, D, settings, flas_rng, [](float) { return false; });
    auto t_end = std::chrono::high_resolution_clock::now();
    double sort_secs = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "[FLAS_Sort_2D_N500_D8] sort_secs=" << sort_secs << std::endl;

    // Verify permutation validity
    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        int id = map_fields[i].id;
        ASSERT_GE(id, 0);
        ASSERT_LT(id, N);
        EXPECT_FALSE(seen[id]);
        seen[id] = true;
    }

    EXPECT_LT(sort_secs, 10.0);
}

// ---------------------------------------------------------------------------
// Helper: run FLAS sorting with an explicit thread count and return
// the sorted ID permutation. The single-thread path is exercised via
// run_flas_sort above; this overload drives the parallel swap path.
// ---------------------------------------------------------------------------
static std::vector<int> run_flas_sort_mt(const float* features, int N, int dim,
                                                  unsigned int seed, const flas::FlasSettings& settings,
                                                  int num_threads) {
    auto map_fields = flas::make_map_fields(features, N, dim);

    flas::RandomEngine rng(seed);
    flas::do_sorting_1d(map_fields, dim, settings, rng,
                                 [](float) { return false; }, num_threads);

    std::vector<int> result(N);
    for (int i = 0; i < N; ++i) result[i] = map_fields[i].id;
    return result;
}

// ============================================================================
// DEG Graph Build & Search Performance Benchmark
// Compares building and searching a graph on:
// 1) RAW (unsorted) data
// 2) FLAS-sorted (presorted) data
// ============================================================================

TEST(FlasRegression, DEGGraphBuildAndSearch_RawVsFlasSorted) {
    const size_t dim = 128;
    const size_t base_count = 5000;
    const size_t query_count = 100;
    const size_t num_clusters = 100;
    const uint32_t edges_per_vertex = 16;
    const uint8_t extend_k = static_cast<uint8_t>(edges_per_vertex);
    const float extend_eps = 0.1f;
    const uint32_t search_k = 10;
    const float search_eps = 0.05f;
    const float radius_decay = 0.8f;

    // Generate clustered dataset
    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    // Compute groundtruth using scalar L2 distance
    auto gt_data = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, search_k);

    const size_t feature_bytes_per_vec = dim * sizeof(float);
    // -----------------------------------------------------------------------
    // 1. Build DEG graph on RAW (unsorted) data
    // -----------------------------------------------------------------------
    std::vector<std::byte> base_bytes_raw(base_count * feature_bytes_per_vec);
    std::memcpy(base_bytes_raw.data(), base_data.data(), base_count * feature_bytes_per_vec);

    std::vector<std::byte> query_bytes_raw(query_count * feature_bytes_per_vec);
    std::memcpy(query_bytes_raw.data(), query_data.data(), query_count * feature_bytes_per_vec);

    deglib::FloatSpace feature_space_raw(dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph_raw(static_cast<uint32_t>(base_count), edges_per_vertex,
                                              std::move(feature_space_raw));

    double build_secs_raw = build_deg_graph(graph_raw, base_bytes_raw, feature_bytes_per_vec,
                                            base_count,
                                            deglib::builder::OptimizationTarget::LowLID,
                                            edges_per_vertex, extend_k, extend_eps);

    auto [qps_raw, recall_raw] = search_deg_graph(graph_raw, query_bytes_raw, feature_bytes_per_vec,
                                                   query_count, gt_data, search_k, search_eps);

    std::vector<int> sorted_ids_raw(base_count);
    for (size_t i = 0; i < base_count; ++i) sorted_ids_raw[i] = static_cast<int>(i);
    float locality_raw = compute_1d_locality(base_data, static_cast<int>(base_count), static_cast<int>(dim), sorted_ids_raw);

    // -----------------------------------------------------------------------
    // 2. Build DEG graph on FLAS Clean (serial) sorted data
    // -----------------------------------------------------------------------
    flas::FlasSettings flas_settings = flas::FlasSettings{};
    flas_settings.radius_decay = radius_decay;
    auto t_flas_start = std::chrono::high_resolution_clock::now();
    auto sorted_ids_clean = run_flas_sort(base_data.data(), static_cast<int>(base_count),
                                           static_cast<int>(dim), 42, flas_settings);
    auto t_flas_end = std::chrono::high_resolution_clock::now();
    double flas_secs_clean = std::chrono::duration<double>(t_flas_end - t_flas_start).count();

    std::vector<float> base_data_clean(base_count * dim);
    for (size_t i = 0; i < base_count; ++i) {
        std::memcpy(&base_data_clean[i * dim], &base_data[sorted_ids_clean[i] * dim], dim * sizeof(float));
    }

    std::vector<std::vector<uint32_t>> gt_data_clean(query_count);
    for (size_t q = 0; q < query_count; ++q) {
        gt_data_clean[q].resize(search_k);
        for (size_t k = 0; k < search_k && k < gt_data[q].size(); ++k) {
            int original_id = static_cast<int>(gt_data[q][k]);
            for (size_t pos = 0; pos < base_count; ++pos) {
                if (sorted_ids_clean[pos] == original_id) {
                    gt_data_clean[q][k] = static_cast<uint32_t>(pos);
                    break;
                }
            }
        }
    }

    std::vector<std::byte> base_bytes_clean(base_count * feature_bytes_per_vec);
    std::memcpy(base_bytes_clean.data(), base_data_clean.data(), base_count * feature_bytes_per_vec);

    deglib::FloatSpace feature_space_clean(dim, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph_clean(static_cast<uint32_t>(base_count), edges_per_vertex,
                                                   std::move(feature_space_clean));

    double build_secs_clean = build_deg_graph(graph_clean, base_bytes_clean, feature_bytes_per_vec,
                                                base_count,
                                                deglib::builder::OptimizationTarget::LowLID,
                                                edges_per_vertex, extend_k, extend_eps);

    auto [qps_clean, recall_clean] = search_deg_graph(graph_clean, query_bytes_raw, feature_bytes_per_vec,
                                                         query_count, gt_data_clean, search_k, search_eps);
    float locality_clean = compute_1d_locality(base_data, static_cast<int>(base_count), static_cast<int>(dim), sorted_ids_clean);

    // Print clean formatted results
    char buf[256];
    std::snprintf(buf, sizeof(buf), "[DEGGraph_1_RAW]              locality=%-8.2f                   build_secs=%-7.4f  qps=%-8.1f  recall=%.3f",
                  locality_raw, build_secs_raw, qps_raw, recall_raw);
    std::cout << buf << std::endl;

    std::snprintf(buf, sizeof(buf), "[DEGGraph_2_FlasSorted]         locality=%-8.2f  flas_secs=%-7.4f  build_secs=%-7.4f  qps=%-8.1f  recall=%.3f",
                  locality_clean, flas_secs_clean, build_secs_clean, qps_clean, recall_clean);
    std::cout << buf << std::endl;

    EXPECT_GE(recall_raw + 1e-5, 0.5);
    EXPECT_GE(recall_clean + 1e-5, 0.5);
}

TEST(FlasRegression, InnerProductMetric_N200_D16) {
    const int N = 200;
    const int D = 16;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    flas::FlasSettings settings = flas::FlasSettings{};
    settings.metric = flas::FlasMetric::InnerProduct;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto sorted_ids = run_flas_sort(features.data(), N, D, 42, settings);
    auto t_end = std::chrono::high_resolution_clock::now();
    double sort_secs = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "[FLAS_InnerProduct_N200_D16] sort_secs=" << sort_secs << std::endl;

    // Verify permutation validity
    std::vector<bool> seen(N, false);
    for (int i = 0; i < N; ++i) {
        ASSERT_GE(sorted_ids[i], 0);
        ASSERT_LT(sorted_ids[i], N);
        EXPECT_FALSE(seen[sorted_ids[i]]);
        seen[sorted_ids[i]] = true;
    }
}

// ============================================================================
// FLAS determinism test — same seed should produce same result
// ============================================================================

TEST(FlasRegression, Determinism_SameSeedSameResult) {
    const int N = 100;
    const int D = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    for (auto& v : features) v = dist(rng);

    flas::FlasSettings settings = flas::FlasSettings{};

    auto sorted1 = run_flas_sort(features.data(), N, D, 42, settings);
    auto sorted2 = run_flas_sort(features.data(), N, D, 42, settings);

    EXPECT_EQ(sorted1, sorted2) << "Same seed should produce same sorted result";
}

// ============================================================================
// FLAS 1D — parallel swap path: validity, locality, single-thread equivalence
// ============================================================================
// Drives check_random_swaps_1d_mt through do_sorting_1d with N (2, 4, 8)
// threads and verifies:
//   1. Every run produces a valid permutation (no dropped/duplicate ids) —
//      this catches data races on map_fields.
//   2. Parallel locality is at least as good as the raw unsorted input.
//   3. With num_threads == 1 the MT entry point matches the serial result
//      bit-for-bit (the single-thread fast path is preserved).
// ============================================================================
TEST(FlasRegression, ParallelSwaps_ValidityAndLocality) {
    const int N = 50000;
    const int D = 16;

    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : features) v = dist(rng);

    flas::FlasSettings settings;

    // Raw (unsorted) locality as the lower bound the parallel sort must beat.
    std::vector<int> raw_ids(N);
    for (int i = 0; i < N; ++i) raw_ids[i] = i;
    const float locality_raw = compute_1d_locality(features, N, D, raw_ids);

    // Serial reference (single thread) — timed for a speedup reference.
    auto t0 = std::chrono::high_resolution_clock::now();
    auto sorted_serial = run_flas_sort(features.data(), N, D, 42, settings);
    auto t1 = std::chrono::high_resolution_clock::now();
    const double secs_serial = std::chrono::duration<double>(t1 - t0).count();

    auto check_valid_permutation = [&](const std::vector<int>& ids, const char* label) {
        ASSERT_EQ(ids.size(), static_cast<size_t>(N));
        std::vector<bool> seen(N, false);
        for (int i = 0; i < N; ++i) {
            ASSERT_GE(ids[i], 0) << label;
            ASSERT_LT(ids[i], N) << label;
            EXPECT_FALSE(seen[ids[i]]) << label << " duplicate id at position " << i;
            seen[ids[i]] = true;
        }
    };

    check_valid_permutation(sorted_serial, "serial");
    const float locality_serial = compute_1d_locality(features, N, D, sorted_serial);

    std::cout << "[ParallelSwaps_Serial] locality=" << locality_serial
              << "  raw=" << locality_raw << "  secs=" << secs_serial << std::endl;

    for (int tc : {2, 4, 8}) {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto sorted_mt = run_flas_sort_mt(features.data(), N, D, 42, settings, tc);
        auto t3 = std::chrono::high_resolution_clock::now();
        const double secs_mt = std::chrono::duration<double>(t3 - t2).count();

        check_valid_permutation(sorted_mt, ("mt-t" + std::to_string(tc)).c_str());
        const float locality_mt = compute_1d_locality(features, N, D, sorted_mt);

        std::cout << "[ParallelSwaps_MT_" << tc << "] locality=" << locality_mt
                  << " (serial=" << locality_serial << ", raw=" << locality_raw << ")"
                  << "  secs=" << secs_mt << "  speedup=" << (secs_serial / secs_mt) << "x"
                  << std::endl;

        // MT must improve over the raw input (the whole point of FLAS).
        EXPECT_LT(locality_mt, locality_raw)
            << "MT run with " << tc << " threads did not improve locality vs raw";
    }
}

TEST(FlasRegression, ParallelSwaps_SingleThreadMatchesSerial) {
    const int N = 1000;
    const int D = 8;

    std::vector<float> features(static_cast<size_t>(N) * static_cast<size_t>(D));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : features) v = dist(rng);

    flas::FlasSettings settings;

    const auto sorted_serial = run_flas_sort(features.data(), N, D, 42, settings);
    const auto sorted_mt1   = run_flas_sort_mt(features.data(), N, D, 42, settings, 1);

    // num_threads == 1 must take the identical serial fast path.
    EXPECT_EQ(sorted_serial, sorted_mt1);
}


