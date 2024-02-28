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
                         const float eps, const uint32_t k, const uint32_t test_size)
{
    size_t total = 0;
    size_t correct = 0;
    for (uint32_t i = 0; i < test_size; i++)
    {
        auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(i));
        auto result_queue = graph.search(entry_vertex_indices, query, eps, k);        

        if (result_queue.size() != k) {
            fmt::print(stderr, "ANNS with k={} got only {} results for query {}\n", k, result_queue.size(), i);
            abort();
        }

        total += result_queue.size();
        const auto gt = ground_truth[i];
        while (result_queue.empty() == false)
        {
            const auto internal_index = result_queue.top().getInternalIndex();
            const auto external_id = graph.getExternalLabel(internal_index);
            if (gt.find(external_id) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static float test_approx_explore(const deglib::search::SearchGraph& graph, const std::vector<std::vector<uint32_t>>& entry_vertex_indices, 
                                  const std::vector<std::unordered_set<uint32_t>>& ground_truth, const uint32_t k, const uint32_t max_distance_count)
{    
    size_t total = 0;
    size_t correct = 0;
    for (uint32_t i = 0; i < entry_vertex_indices.size(); i++) {
        const auto entry_vertex_index = entry_vertex_indices[i][0];
        auto result_queue = graph.explore(entry_vertex_index, k, max_distance_count);

        total += k;
        const auto& gt = ground_truth[i];
        while (result_queue.empty() == false)
        {
            const auto internal_index = result_queue.top().getInternalIndex();
            const auto external_id = graph.getExternalLabel(internal_index);
            if (gt.find(external_id) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static void test_graph_anns(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, const uint32_t* ground_truth, const uint32_t ground_truth_dims, const uint32_t repeat, const uint32_t k)
{
    // reproduceable entry point for the graph search
    const auto entry_vertex_indices = graph.getEntryVertexIndices();
    fmt::print("external id {} \n", graph.getInternalIndex(entry_vertex_indices[0]));

    // test ground truth
    fmt::print("Parsing gt:\n");
    auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), ground_truth_dims, k);
    fmt::print("Loaded gt:\n");

    // try different eps values for the search radius
    // std::vector<float> eps_parameter = { 0.00f, 0.03f, 0.05f, 0.07f, 0.09f, 0.12f, 0.2f, 0.3f, };    // audio
    std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f  };       // SIFT1M k=100
    // std::vector<float> eps_parameter = { 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f };             // GloVe
    // std::vector<float> eps_parameter = { 0.01f, 0.02f, 0.03f, 0.04f, 0.06f, 0.1f, 0.2f, };          // Deep1M

    const auto test_size = uint32_t(query_repository.size());
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = 0;
        for (size_t i = 0; i < repeat; i++) 
            recall = deglib::benchmark::test_approx_anns(graph, entry_vertex_indices, query_repository, answer, eps, k, test_size);
        uint64_t time_us_per_query = (stopw.getElapsedTimeMicro() / test_size) / repeat;

        fmt::print("eps {:.2f} \t recall {:.5f} \t time_us_per_query {:6}us\n", eps, recall, time_us_per_query);
        if (recall > 1.0)
            break;
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

static void test_graph_explore(const deglib::search::SearchGraph& graph, const uint32_t query_count, const uint32_t* ground_truth, const uint32_t ground_truth_dims, const uint32_t* entry_vertices, const uint32_t entry_vertex_dims, const uint32_t repeat, const uint32_t k)
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
        auto entry_vertex = std::vector<uint32_t>(entry_vertices + i * entry_vertex_dims, entry_vertices + (i+1) * entry_vertex_dims);
        entry_vertex_indices.emplace_back(entry_vertex);
    }

    // ground truth data
    const auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_count, ground_truth_dims, k);

    // try different k values
    uint32_t k_factor = 100;
    for (uint32_t f = 0; f <= 3; f++, k_factor *= 10) {
        for (uint32_t i = (f == 0) ? 1 : 2; i < 11; i++) {         
           const auto max_distance_count = ((f == 0) ? (k + k_factor * (i-1)) : (k_factor * i));

        //  for (uint32_t i = 1; i < 14; i++) {
        //      const auto max_distance_count = i;

            StopW stopw = StopW();
            float recall = 0;
            for (size_t r = 0; r < repeat; r++) 
                recall = deglib::benchmark::test_approx_explore(graph, entry_vertex_indices, answer, k, max_distance_count);
            uint64_t time_us_per_query = stopw.getElapsedTimeMicro() / (query_count * repeat);

            fmt::print("max_distance_count {:5}, k {:4}, recall {:.5f}, time_us_per_query {:4}us\n", max_distance_count, k, recall, time_us_per_query);
            if (recall > 1.0)
                break;
        }
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

}  // namespace deglib::benchmark