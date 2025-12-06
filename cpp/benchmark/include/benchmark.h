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

static float test_approx_anns(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_vertex_indices,
                         const deglib::FeatureRepository& query_repository, const std::vector<std::vector<uint32_t>>& ground_truth, 
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
            if (std::binary_search(gt.begin(), gt.end(), external_id)) correct++;
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
                                  const std::vector<std::vector<uint32_t>>& ground_truth, const uint32_t k, const uint32_t max_distance_count,
                                  const uint32_t threads, const deglib::graph::Filter* filter = nullptr)
{    
    auto corrects = std::vector<float>(threads);
    deglib::concurrent::parallel_for(0, entry_vertex_indices.size(), threads, [&] (size_t i, size_t thread_id) {

        const auto entry_vertex_index = entry_vertex_indices[i][0];
        auto result_queue = graph.explore(entry_vertex_index, k, include_entry, max_distance_count); // TODO missing filter

        if (result_queue.size() != k) {
            fmt::print(stderr, "Exploration with k={} got only {} results for query {} and max_distance_count {}\n", k, result_queue.size(), i, max_distance_count);
            abort();
        }

        uint32_t correct = 0;
        const auto& gt = ground_truth[i];
        while (result_queue.empty() == false)
        {
            const auto& result = result_queue.top();
            const auto external_id = graph.getExternalLabel(result.getInternalIndex());
            if (std::binary_search(gt.begin(), gt.end(), external_id)) correct++;
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

static std::vector<float> estimate_recall(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, const std::vector<std::vector<uint32_t>>& answer, const uint32_t max_distance_count, const uint32_t k) {

    const auto entry_vertex_indices = std::vector<uint32_t> { graph.getInternalIndex(0) };

    std::vector<float> recalls;
    std::vector<float> eps_parameter = { 0.1f, 0.2f };

    for (float eps : eps_parameter)
    {
        size_t total = 0;
        size_t correct = 0;
        
        #pragma omp parallel for reduction(+:total) reduction(+:correct)
        for (int i = 0; i < (int)query_repository.size(); i++)
        {
            auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(uint32_t(i)));
            auto result_queue = graph.search(entry_vertex_indices, query, eps, k, nullptr, max_distance_count);

            const auto& gt = answer[i];
            total += result_queue.size();
            
            size_t local_correct = 0;
            while (result_queue.empty() == false)
            {
                const auto internal_index = result_queue.top().getInternalIndex();
                const auto external_id = graph.getExternalLabel(internal_index);
                if (std::binary_search(gt.begin(), gt.end(), external_id)) local_correct++;
                result_queue.pop();
            }
            correct += local_correct;
        }

        const auto precision = ((float)correct) / total;
        recalls.push_back(precision);
    }

    return recalls;
}

/**
 * @brief Test ANNS performance using pre-computed ground truth sorted vectors.
 * 
 * @param graph The search graph
 * @param query_repository Query feature repository
 * @param ground_truth Pre-computed ground truth as vector of sorted vectors
 * @param repeat Number of repetitions for timing
 * @param threads Number of threads
 * @param k Number of nearest neighbors to find
 * @param eps_parameter Vector of epsilon values to test
 * @param filter Optional filter
 */
static void test_graph_anns(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, 
                            const std::vector<std::vector<uint32_t>>& ground_truth,
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


/**
 * @brief Test exploration performance using pre-computed ground truth sorted vectors.
 * 
 * @param graph The search graph
 * @param entry_vertex_labels External labels of entry vertices
 * @param ground_truth Pre-computed exploration ground truth as vector of sorted vectors
 * @param include_entry Whether to include entry vertex in results
 * @param repeat Number of repetitions for timing
 * @param k Number of nearest neighbors to find
 * @param threads Number of threads
 * @param filter Optional filter
 * @param explore_depth Exploration depth for max_distance_count scaling
 */
static void test_graph_explore(const deglib::search::SearchGraph& graph,
                               const std::vector<uint32_t>& entry_vertex_labels,
                               const std::vector<std::vector<uint32_t>>& ground_truth,
                               const boolean include_entry,
                               const uint32_t repeat, const uint32_t k, const uint32_t threads,
                               const deglib::graph::Filter* filter = nullptr,
                               const uint32_t explore_depth = 3)
{
    if (entry_vertex_labels.size() != ground_truth.size()) {
        fmt::print(stderr, "Entry vertex count {} does not match ground truth count {}\n", 
                   entry_vertex_labels.size(), ground_truth.size());
        abort();
    }
    
    const uint32_t query_count = (uint32_t)entry_vertex_labels.size();
    
    // Convert external labels to internal indices
    auto entry_vertex_indices = std::vector<std::vector<uint32_t>>(query_count);
    for (size_t i = 0; i < query_count; i++) {
        entry_vertex_indices[i].push_back(graph.getInternalIndex(entry_vertex_labels[i]));
    }

    // try different max_distance_count values
    uint32_t k_factor = 100;
    for (uint32_t f = 0; f <= explore_depth; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {         
            const auto max_distance_count = ((f == 0) ? (k + k_factor * (i-1)) : (k_factor * i));

            StopW stopw = StopW();
            float recall = 0;
            for (size_t r = 0; r < repeat; r++) 
                recall = deglib::benchmark::test_approx_explore(graph, entry_vertex_indices, include_entry, 
                                                                 ground_truth, k, max_distance_count, threads, filter);
            uint64_t search_time_us = stopw.getElapsedTimeMicro();
            uint64_t time_us_per_query = search_time_us / (query_count * repeat);

            log("max_distance_count {:5}, k {:4}, recall {:.5f}, time_us_per_query {:4}us \t search time: {:6}ms\n", 
                max_distance_count, k, recall, time_us_per_query, search_time_us / 1000);
            if (recall > 1.0)
                break;
        }
    }

    log("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    log("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}


}  // namespace deglib::benchmark