#pragma once

/**
 * @file benchmark.h
 * @brief Main benchmark utilities for deglib.
 * 
 * This header includes:
 * - Logging utilities (dual console + file output)
 * - Graph statistics collection
 * - Ground truth computation
 * - ANNS and exploration testing functions
 */

#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <random>
#include <limits>

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

#include "deglib.h"
#include "stopwatch.h"
#include "logging.h"
#include "file_io.h"
#include "stats.h"
#include "dataset.h"

namespace deglib::benchmark
{

// ============================================================================
// Ground Truth computation
// ============================================================================

// Compute ground truth (brute force k-NN) for ANNS queries
inline std::vector<uint32_t> compute_ground_truth(
    const deglib::FeatureRepository& base_repo,
    const deglib::FeatureRepository& query_repo,
    const deglib::Metric metric,
    const uint32_t k_target,
    const uint32_t thread_count = 1)
{
    const auto base_size = (uint32_t)base_repo.size();
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
            topList[i] = results.top().getInternalIndex();
            results.pop();
        }

        uint32_t count = ++progress;
        if (count % 100 == 0) {
            const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            log("  Computed {}/{} ground truth lists after {}ms\n", count, query_size, duration_ms);
        }
    });

    return topLists;
}

// Compute ground truth for exploration (k-NN from entry vertices in graph)
inline std::vector<uint32_t> compute_explore_ground_truth(
    const deglib::search::SearchGraph& graph,
    const std::vector<uint32_t>& entry_vertex_labels,
    const uint32_t k_target,
    const bool include_entry,
    const uint32_t thread_count = 1)
{
    const auto query_count = entry_vertex_labels.size();
    auto topLists = std::vector<uint32_t>(k_target * query_count);
    std::atomic<uint32_t> progress{0};
    const auto start = std::chrono::steady_clock::now();

    deglib::concurrent::parallel_for(0, query_count, thread_count, [&](size_t q, size_t) {
        const auto entry_label = entry_vertex_labels[q];
        const auto entry_index = graph.getInternalIndex(entry_label);
        
        // Use a large max_distance_count for ground truth
        auto result_queue = graph.explore(entry_index, k_target, include_entry, 100000);
        
        auto topList = topLists.data() + (k_target * q);
        for (int32_t i = k_target - 1; i >= 0 && !result_queue.empty(); i--) {
            topList[i] = graph.getExternalLabel(result_queue.top().getInternalIndex());
            result_queue.pop();
        }

        uint32_t count = ++progress;
        if (count % 100 == 0) {
            const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            log("  Computed {}/{} explore ground truth lists after {}ms\n", count, query_count, duration_ms);
        }
    });

    return topLists;
}

/*
 * Stream types for the benchmarks. These are shared across the benchmark programs and
 * were previously declared inside multiple files. Move the canonical definition here
 * so we can reuse it across the tools.
 */
enum DataStreamType { AddAll, AddHalf, AddAllRemoveHalf, AddHalfRemoveAndAddOneAtATime };

/**
 * @brief Convert raw ground truth data to vector of unordered_sets.
 * @deprecated Use DatasetPaths::to_groundtruth_sets() instead.
 */
[[deprecated("Use DatasetPaths::to_groundtruth_sets() instead")]]
static std::vector<std::unordered_set<uint32_t>> get_ground_truth(
    const uint32_t* ground_truth, 
    const size_t ground_truth_size, 
    const uint32_t ground_truth_dims, 
    const size_t k) 
{
    return DatasetPaths::to_groundtruth_sets(ground_truth, ground_truth_size, ground_truth_dims, k);
}

static float test_approx_anns(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_vertex_indices,
                         const deglib::FeatureRepository& query_repository, const std::vector<std::unordered_set<uint32_t>>& ground_truth, 
                         const float eps, const uint32_t k, const uint32_t test_size, const uint32_t threads, const deglib::graph::Filter* filter = nullptr)
{
    auto corrects = std::vector<float>(threads);
    deglib::concurrent::parallel_for(0, test_size, threads, [&] (size_t i, size_t thread_id) {

        auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(uint32_t(i)));
        auto result_queue = graph.search(entry_vertex_indices, query, eps, k, filter);

        if (result_queue.size() != k) {
            fmt::print(stderr, "ANNS with k={} got only {} results for query {}\n", k, result_queue.size(), i);
            abort();
        }

        uint32_t correct = 0;
        const auto& gt = ground_truth[i];
        while (result_queue.empty() == false)
        {
            const auto& result = result_queue.top();
            const auto external_id = graph.getExternalLabel(result.getInternalIndex());
            if (gt.find(external_id) != gt.end()) correct++;
            result_queue.pop();
        }

        corrects[thread_id] += correct;
    });

    // calc recall
    float total_correct = 0;
    for (size_t i = 0; i < threads; i++) 
        total_correct += corrects[i];
    return total_correct / (test_size*k);
}

static float test_approx_explore(const deglib::search::SearchGraph& graph, const std::vector<std::vector<uint32_t>>& entry_vertex_indices, const boolean include_entry,
                                  const std::vector<std::unordered_set<uint32_t>>& ground_truth, const uint32_t k, const uint32_t max_distance_count,
                                  const uint32_t threads, const deglib::graph::Filter* filter = nullptr)
{    
    auto corrects = std::vector<float>(threads);
    deglib::concurrent::parallel_for(0, entry_vertex_indices.size(), threads, [&] (size_t i, size_t thread_id) {

        const auto entry_vertex_index = entry_vertex_indices[i][0];
        auto result_queue = graph.explore(entry_vertex_index, k, include_entry, max_distance_count); // TODO missing filter

        if (result_queue.size() != k) {
            fmt::print(stderr, "Exploration with k={} got only {} results for query {}\n", k, result_queue.size(), i);
            abort();
        }

        uint32_t correct = 0;
        const auto& gt = ground_truth[i];
        while (result_queue.empty() == false)
        {
            const auto& result = result_queue.top();
            const auto external_id = graph.getExternalLabel(result.getInternalIndex());
            if (gt.find(external_id) != gt.end()) correct++;
            result_queue.pop();
        }

        corrects[thread_id] += correct;
    });

    // calc recall
    float total_correct = 0;
    for (size_t i = 0; i < threads; i++) 
        total_correct += corrects[i];
    return total_correct / (entry_vertex_indices.size()*k);
}

static void test_graph_anns(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, 
                            const uint32_t* ground_truth, const uint32_t ground_truth_dims, 
                            const uint32_t repeat, const uint32_t threads, const uint32_t k, 
                            const std::vector<float>& eps_parameter,
                            const deglib::graph::Filter* filter = nullptr)
{
    // reproduceable entry point for the graph search
    const auto entry_vertex_indices = graph.getEntryVertexIndices();
    log("internal id {} \n", entry_vertex_indices[0]);

    // test ground truth
    log("Parsing gt:\n");
    auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), ground_truth_dims, k);
    log("Loaded gt:\n");

    log("Compute TOP{} for eps {}\n", k, fmt::join(eps_parameter, ", "));
    const auto test_size = uint32_t(query_repository.size());
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = 0;
        for (size_t i = 0; i < repeat; i++) 
            recall = deglib::benchmark::test_approx_anns(graph, entry_vertex_indices, query_repository, answer, eps, k, test_size, threads, filter);
        uint64_t search_time_us = stopw.getElapsedTimeMicro();
        uint64_t time_us_per_query = (search_time_us / test_size) / repeat;

        log("eps {:.3f} \t recall {:.5f} \t time_us_per_query {:6}us \t search time: {:6}ms\n", eps, recall, time_us_per_query, search_time_us / 1000);
        if (recall > 1.0)
            break;
    }

    log("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    log("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

/**
 * @brief Test ANNS performance using pre-computed ground truth sets.
 * 
 * @param graph The search graph
 * @param query_repository Query feature repository
 * @param ground_truth Pre-computed ground truth as vector of unordered_sets
 * @param repeat Number of repetitions for timing
 * @param threads Number of threads
 * @param k Number of nearest neighbors to find
 * @param eps_parameter Vector of epsilon values to test
 * @param filter Optional filter
 */
static void test_graph_anns(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, 
                            const std::vector<std::unordered_set<uint32_t>>& ground_truth,
                            const uint32_t repeat, const uint32_t threads, const uint32_t k, 
                            const std::vector<float>& eps_parameter,
                            const deglib::graph::Filter* filter = nullptr)
{
    // reproduceable entry point for the graph search
    const auto entry_vertex_indices = graph.getEntryVertexIndices();
    log("internal id {} \n", entry_vertex_indices[0]);

    log("Compute TOP{} for eps {}\n", k, fmt::join(eps_parameter, ", "));
    const auto test_size = uint32_t(query_repository.size());
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = 0;
        for (size_t i = 0; i < repeat; i++) 
            recall = deglib::benchmark::test_approx_anns(graph, entry_vertex_indices, query_repository, ground_truth, eps, k, test_size, threads, filter);
        uint64_t search_time_us = stopw.getElapsedTimeMicro();
        uint64_t time_us_per_query = (search_time_us / test_size) / repeat;

        log("eps {:.3f} \t recall {:.5f} \t time_us_per_query {:6}us \t search time: {:6}ms\n", eps, recall, time_us_per_query, search_time_us / 1000);
        if (recall > 1.0)
            break;
    }

    log("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    log("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

static void test_graph_explore(const deglib::search::SearchGraph& graph, const uint32_t query_count, const uint32_t* ground_truth, const uint32_t ground_truth_dims, const uint32_t* entry_vertex_labels, const uint32_t entry_vertex_dims, const boolean include_entry, const uint32_t repeat, const uint32_t k, const uint32_t threads, const deglib::graph::Filter* filter = nullptr)
{
    if (ground_truth_dims < k)
    {
        fmt::print(stderr, "ground thruth data does not have enough dimensions, expected {} got {} \n", k, ground_truth_dims);
        perror("");
        abort();
    }
      // reproduceable entry point for the graph search
    auto entry_vertex_indices = std::vector<std::vector<uint32_t>>();
    for (size_t i = 0; i < query_count; i++) {
        auto entry_vertex = std::vector<uint32_t>();
        entry_vertex.reserve(entry_vertex_dims);

        // use the entry vertex labels to get the internal index
        for (size_t v = 0; v < entry_vertex_dims; v++)             
            entry_vertex.emplace_back(graph.getInternalIndex(entry_vertex_labels[i * entry_vertex_dims + v]));
        entry_vertex_indices.emplace_back(entry_vertex);
    }

    // ground truth data
    const auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_count, ground_truth_dims, k);

    // try different k values
    uint32_t k_factor = 16;
    for (uint32_t f = 0; f <= 3; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {         
           const auto max_distance_count = ((f == 0) ? (k + k_factor * (i-1)) : (k_factor * i));

        //  for (uint32_t i = 1; i < 14; i++) {
        //      const auto max_distance_count = i;

            StopW stopw = StopW();
            float recall = 0;
            for (size_t r = 0; r < repeat; r++) 
                recall = deglib::benchmark::test_approx_explore(graph, entry_vertex_indices, include_entry, answer, k, max_distance_count, threads, filter);
            uint64_t search_time_us = stopw.getElapsedTimeMicro();
            uint64_t time_us_per_query = search_time_us / (query_count * repeat);

            fmt::print("max_distance_count {:5}, k {:4}, recall {:.5f}, time_us_per_query {:4}us \t search time: {:6}ms\n", max_distance_count, k, recall, time_us_per_query, search_time_us / 1000);
            if (recall > 1.0)
                break;
        }
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

}  // namespace deglib::benchmark