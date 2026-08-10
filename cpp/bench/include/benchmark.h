#pragma once

/**
 * @file benchmark.h
 * @brief Main benchmark utilities for deglib without OpenMP dependency.
 */

#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <atomic>
#include <span>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>

#include "dataset.h"
#include <deglib/deglib.h>
#include "file_io.h"
#include "logging.h"
#include "stats.h"
#include "stopwatch.h"

namespace deglib::benchmark {

/**
 * @brief Compute baseline time per query using linear distance computation.
 */
static uint64_t compute_linear_search_baseline(const deglib::FeatureRepository& base_repository,
                                               const deglib::distances::Metric metric,
                                               const uint32_t sample_size = 100,
                                               const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    const auto dims = base_repository.dims();
    const auto base_size = base_repository.size();
    const auto feature_space = deglib::distances::FloatSpace(dims, metric, instruction);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    const auto query_count = std::min(sample_size, (uint32_t)base_size);

    std::vector<uint32_t> query_indices(query_count);
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)base_size - 1);
    for (uint32_t i = 0; i < query_count; i++) {
        query_indices[i] = dist(rng);
    }

    log("Computing linear search baseline with {} random queries on {} base vectors...\n", query_count, base_size);

    StopW stopw = StopW();

    float min_dist = std::numeric_limits<float>::max();
    for (uint32_t q = 0; q < query_count; q++) {
        const auto query = base_repository.getFeature(query_indices[q]);

        for (uint32_t i = 0; i < base_size; i++) {
            const auto base_feature = base_repository.getFeature(i);
            const auto d = dist_func(query, base_feature, dist_func_param);
            if (d < min_dist) min_dist = d;
        }
    }

    const uint64_t total_time_us = stopw.getElapsedTimeMicro();
    const uint64_t time_per_query_us = total_time_us / query_count;

    log("Linear search baseline: {}us per query (total: {}ms for {} queries) with min distance {}\n",
        time_per_query_us,
        total_time_us / 1000,
        query_count,
        min_dist);

    return time_per_query_us;
}

static float test_approx_anns(const deglib::graph::InternalGraph& graph,
                              const std::vector<uint32_t>& entry_vertex_indices,
                              const deglib::FeatureRepository& query_repository,
                              const std::vector<std::vector<uint32_t>>& ground_truth,
                              const float eps,
                              const uint32_t k,
                              const uint32_t test_size,
                              const uint32_t threads,
                              const deglib::search::Filter* filter = nullptr) {
    auto corrects = std::vector<float>(threads);
    deglib::concurrent::parallel_for(0, test_size, threads, [&](size_t i, size_t thread_id) {
        auto query = reinterpret_cast<const float*>(query_repository.getFeature(uint32_t(i)));
        auto result_queue = graph.search(std::span<const float>(query, graph.getFeatureSpace().dim()), k, eps, filter);

        if (result_queue.size() != k) {
            fmt::print(stderr, "ANNS with k={} got only {} results for query {}\n", k, result_queue.size(), i);
            abort();
        }

        uint32_t correct = 0;
        const auto& gt = ground_truth[i];
        while (result_queue.empty() == false) {
            const auto& result = result_queue.top();
            const auto external_id = graph.getExternalLabel(result.getIdentifier());
            if (std::binary_search(gt.begin(), gt.end(), external_id)) correct++;
            result_queue.pop();
        }

        corrects[thread_id] += correct;
    });

    float total_correct = 0;
    for (size_t i = 0; i < threads; i++) total_correct += corrects[i];
    return total_correct / (test_size * k);
}

static float test_approx_explore(const deglib::graph::InternalGraph& graph,
                                 const std::vector<std::vector<uint32_t>>& entry_vertex_indices,
                                 const bool include_entry,
                                 const std::vector<std::vector<uint32_t>>& ground_truth,
                                 const uint32_t k,
                                 const uint32_t max_distance_count,
                                 const uint32_t threads,
                              const deglib::search::Filter* filter = nullptr) {
    auto corrects = std::vector<float>(threads);
    deglib::concurrent::parallel_for(0, entry_vertex_indices.size(), threads, [&](size_t i, size_t thread_id) {
        const auto entry_vertex_index = entry_vertex_indices[i][0];
        auto result_queue = graph.explore(entry_vertex_index, k, max_distance_count, 0.0f, include_entry);

        if (result_queue.size() != k) {
            fmt::print(stderr,
                       "Exploration with k={} got only {} results for query {} and max_distance_count {}\n",
                       k,
                       result_queue.size(),
                       i,
                       max_distance_count);
            abort();
        }

        uint32_t correct = 0;
        const auto& gt = ground_truth[i];
        while (result_queue.empty() == false) {
            const auto& result = result_queue.top();
            const auto external_id = graph.getExternalLabel(result.getIdentifier());
            if (std::binary_search(gt.begin(), gt.end(), external_id)) correct++;
            result_queue.pop();
        }

        corrects[thread_id] += correct;
    });

    float total_correct = 0;
    for (size_t i = 0; i < threads; i++) total_correct += corrects[i];
    return total_correct / (entry_vertex_indices.size() * k);
}

static std::vector<float> estimate_recall(const deglib::graph::InternalGraph& graph,
                                          const deglib::FeatureRepository& query_repository,
                                          const std::vector<std::vector<uint32_t>>& answer,
                                          const uint32_t max_distance_count,
                                          const uint32_t k) {
    const auto entry_vertex_indices = std::vector<uint32_t>{graph.getInternalIndex(0)};

    std::vector<float> recalls;
    std::vector<float> eps_parameter = {0.1f, 0.2f};
    const uint32_t threads = std::thread::hardware_concurrency() / 2;

    for (float eps : eps_parameter) {
        std::atomic<size_t> total{0};
        std::atomic<size_t> correct{0};

        deglib::concurrent::parallel_for(0, query_repository.size(), threads, [&](size_t i, size_t thread_id) {
            auto query = reinterpret_cast<const float*>(query_repository.getFeature(uint32_t(i)));
            auto result_queue = graph.search(std::span<const float>(query, graph.getFeatureSpace().dim()), k, eps, nullptr, max_distance_count);

            const auto& gt = answer[i];
            total += result_queue.size();

            size_t local_correct = 0;
            while (result_queue.empty() == false) {
                const auto internal_index = result_queue.top().getIdentifier();
                const auto external_id = graph.getExternalLabel(internal_index);
                if (std::binary_search(gt.begin(), gt.end(), external_id)) local_correct++;
                result_queue.pop();
            }
            correct += local_correct;
        });

        const auto precision = ((float)correct.load()) / total.load();
        recalls.push_back(precision);
    }

    return recalls;
}

static void test_graph_anns(const deglib::graph::InternalGraph& graph,
                            const deglib::FeatureRepository& query_repository,
                            const std::vector<std::vector<uint32_t>>& ground_truth,
                            const uint32_t repeat,
                            const uint32_t threads,
                            const uint32_t k,
                            const std::vector<float>& eps_parameter,
                            const deglib::search::Filter* filter = nullptr,
                            const uint64_t linear_baseline_us = 0,
                            const uint32_t abort_sample_size = 100) {
    const auto entry_vertex_indices = graph.getEntryVertexIndices();

    std::vector<float> eps_parameter_sorted = eps_parameter;
    std::sort(eps_parameter_sorted.begin(), eps_parameter_sorted.end());
    log("Compute TOP{} for eps {}\n", k, fmt::join(eps_parameter_sorted, ", "));
    if (linear_baseline_us > 0) {
        log("Early abort enabled: baseline {}us/query, checking after {} queries\n", linear_baseline_us, abort_sample_size);
    }

    log("Internal seed id {} \n", entry_vertex_indices[0]);
    log("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    log("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);

    const auto test_size = uint32_t(query_repository.size());
    for (float eps : eps_parameter_sorted) {
        if (linear_baseline_us > 0 && test_size > abort_sample_size) {
            const auto sample_size = std::min(abort_sample_size, test_size);
            StopW sample_stopw = StopW();
            deglib::benchmark::test_approx_anns(
                graph, entry_vertex_indices, query_repository, ground_truth, eps, k, sample_size, threads, filter);
            const uint64_t sample_time_us = sample_stopw.getElapsedTimeMicro();
            const uint64_t sample_time_per_query = sample_time_us / sample_size;

            if (sample_time_per_query > linear_baseline_us) {
                log("eps {:.3f} \t ABORTED ({}us/query > {}us baseline after {} queries)\n",
                    eps,
                    sample_time_per_query,
                    linear_baseline_us,
                    sample_size);
                return;
            }
        }

        StopW stopw = StopW();
        float recall = 0;
        for (size_t i = 0; i < repeat; i++)
            recall = deglib::benchmark::test_approx_anns(
                graph, entry_vertex_indices, query_repository, ground_truth, eps, k, test_size, threads, filter);
        uint64_t search_time_us = stopw.getElapsedTimeMicro();
        uint64_t time_us_per_query = (search_time_us / test_size) / repeat;

        log("eps {:.3f} \t recall {:.5f} \t time_us_per_query {:6}us \t search time: {:6}ms\n",
            eps,
            recall,
            time_us_per_query,
            search_time_us / 1000);
        if (recall > 0.997) {
            log("Reached recall > 0.997, stopping further tests.\n");
            return;
        }
    }
}

static void test_graph_explore(const deglib::graph::InternalGraph& graph,
                               const std::vector<uint32_t>& entry_vertex_labels,
                               const std::vector<std::vector<uint32_t>>& ground_truth,
                               const bool include_entry,
                               const uint32_t repeat,
                               const uint32_t k,
                               const uint32_t threads,
                            const deglib::search::Filter* filter = nullptr,
                               const uint32_t explore_depth = 3,
                               const uint64_t linear_baseline_us = 0,
                               const uint32_t abort_sample_size = 100,
                               const float recall_target = 0.997f) {
    if (entry_vertex_labels.size() != ground_truth.size()) {
        fmt::print(stderr, "Entry vertex count {} does not match ground truth count {}\n", entry_vertex_labels.size(), ground_truth.size());
        abort();
    }

    const uint32_t query_count = (uint32_t)entry_vertex_labels.size();

    auto entry_vertex_indices = std::vector<std::vector<uint32_t>>(query_count);
    for (size_t i = 0; i < query_count; i++) {
        entry_vertex_indices[i].push_back(graph.getInternalIndex(entry_vertex_labels[i]));
    }

    if (linear_baseline_us > 0) {
        log("Early abort enabled: baseline {}us/query, checking after {} queries\n", linear_baseline_us, abort_sample_size);
    }

    log("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    log("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);

    uint32_t k_factor = 100;
    float last_recall = -1.0f;
    for (uint32_t f = 0; f <= explore_depth; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {
            const auto max_distance_count = ((f == 0) ? (k + k_factor * (i - 1)) : (k_factor * i));

            if (linear_baseline_us > 0 && query_count > abort_sample_size) {
                const auto sample_size = std::min(abort_sample_size, query_count);

                auto sample_entry_indices =
                    std::vector<std::vector<uint32_t>>(entry_vertex_indices.begin(), entry_vertex_indices.begin() + sample_size);
                auto sample_ground_truth = std::vector<std::vector<uint32_t>>(ground_truth.begin(), ground_truth.begin() + sample_size);
                StopW sample_stopw = StopW();
                deglib::benchmark::test_approx_explore(
                    graph, sample_entry_indices, include_entry, sample_ground_truth, k, max_distance_count, threads, filter);
                const uint64_t sample_time_us = sample_stopw.getElapsedTimeMicro();
                const uint64_t sample_time_per_query = sample_time_us / sample_size;

                if (sample_time_per_query > linear_baseline_us) {
                    log("max_distance_count {:5}, k {:4}, ABORTED ({}us/query > {}us baseline after {} queries)\n",
                        max_distance_count,
                        k,
                        sample_time_per_query,
                        linear_baseline_us,
                        sample_size);
                    return;
                }
            }

            StopW stopw = StopW();
            float recall = 0;
            for (size_t r = 0; r < repeat; r++)
                recall = deglib::benchmark::test_approx_explore(
                    graph, entry_vertex_indices, include_entry, ground_truth, k, max_distance_count, threads, filter);
            uint64_t search_time_us = stopw.getElapsedTimeMicro();
            uint64_t time_us_per_query = search_time_us / (query_count * repeat);

            log("k {:5}, max_distance_count {:6}, recall {:.4f}, time_us_per_query {:6}\n",
                k,
                max_distance_count,
                recall,
                time_us_per_query);

            if (recall == last_recall) {
                log("Recall stabilized at {:.4f}, stopping exploration sweep\n", recall);
                return;
            }
            last_recall = recall;

            if (recall >= recall_target) {
                log("Recall target {:.3f} reached, stopping exploration sweep\n", recall_target);
                return;
            }
        }
    }
}

}  // namespace deglib::benchmark

