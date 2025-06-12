#pragma once

#include <filesystem>
#include <unordered_set>

#include "deglib.h"
#include "stopwatch.h"

namespace deglib::benchmark
{

static std::vector<std::unordered_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const uint32_t ground_truth_dims, const size_t k)
{
    // does the ground truth data provide enough top elements the check for k elements?
    if(ground_truth_dims < k) {
        fmt::print(stderr, "Ground truth data has only {} elements but need {}\n", ground_truth_dims, k);
        abort();
    }

    auto answers = std::vector<std::unordered_set<uint32_t>>(ground_truth_size);
    for (uint32_t i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
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

static void test_graph_anns(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, const uint32_t* ground_truth, const uint32_t ground_truth_dims, const uint32_t repeat, const uint32_t threads, const uint32_t k, const deglib::graph::Filter* filter = nullptr)
{
    // reproduceable entry point for the graph search
    const auto entry_vertex_indices = graph.getEntryVertexIndices();
    fmt::print("internal id {} \n", entry_vertex_indices[0]);

    // test ground truth
    fmt::print("Parsing gt:\n");
    auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), ground_truth_dims, k);
    fmt::print("Loaded gt:\n");

    // try different eps values for the search radius
    // std::vector<float> eps_parameter = { 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.11f, 0.15f, 0.2f };    // crawl
    // std::vector<float> eps_parameter = { 0.05f, 0.06f, 0.07f, 0.08f, 0.1f, 0.12f, 0.18f, 0.2f,   };  // enron
    // std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.8f  };                    // UQ-V
    // std::vector<float> eps_parameter = { 0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f, };    // audio
    // std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f  };       // SIFT1M k=100
    std::vector<float> eps_parameter = { 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f };             // GloVe
    // std::vector<float> eps_parameter = { 0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f, };           // Deep1M
    // std::vector<float> eps_parameter = { 0.01f, 0.02f, 0.05f };           // ccnews-small

    fmt::print("Compute TOP{} for eps {}us\n", k, fmt::join(eps_parameter, ", "));
    const auto test_size = uint32_t(query_repository.size());
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = 0;
        for (size_t i = 0; i < repeat; i++) 
            recall = deglib::benchmark::test_approx_anns(graph, entry_vertex_indices, query_repository, answer, eps, k, test_size, threads, filter);
        uint64_t search_time_us = stopw.getElapsedTimeMicro();
        uint64_t time_us_per_query = (search_time_us / test_size) / repeat;

        fmt::print("eps {:.3f} \t recall {:.5f} \t time_us_per_query {:6}us \t search time: {:6}ms\n", eps, recall, time_us_per_query, search_time_us / 1000);
        if (recall > 1.0)
            break;
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
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