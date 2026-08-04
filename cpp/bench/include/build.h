#pragma once

/**
 * @file build.h
 * @brief Graph building utilities for deglib benchmarks.
 */

#include <fmt/core.h>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <random>
#include <vector>

#include "analysis.h"
#include "benchmark.h"
#include "deglib.h"

namespace deglib::benchmark {

enum DataStreamType { AddAll, AddHalf, AddAllRemoveHalf, AddHalfRemoveAndAddOneAtATime };

inline deglib::graph::SizeBoundedGraph create_random_graph(const deglib::StaticFeatureRepository& repository,
                                                           deglib::Metric metric,
                                                           const uint8_t k,
                                                           const uint32_t max_size = 0,
                                                           const uint32_t scale = 1,
                                                           const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    log("Build a random EG{}\n", k);

    const auto dims = repository.dims();
    const auto feature_space = deglib::FloatSpace(dims, metric, instruction);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    const auto start = std::chrono::system_clock::now();
    const uint8_t edges_per_vertex = k;
    const uint32_t vertex_count = (max_size > 0 && max_size < repository.size()) ? max_size : uint32_t(repository.size());
    auto graph = deglib::graph::SizeBoundedGraph(vertex_count, edges_per_vertex, feature_space);

    {
        const auto size = (uint32_t)(edges_per_vertex + 1);
        for (uint32_t y = 0; y < size; y++) {
            const auto query = repository.getFeature(y);
            const auto internal_index = graph.addVertex(y, query);

            auto neighbor_indices = std::vector<uint32_t>();
            auto neighbor_weights = std::vector<float>();
            for (uint32_t x = 0; x < size; x++) {
                if (x == internal_index) continue;
                neighbor_indices.emplace_back(x);
                neighbor_weights.emplace_back(dist_func(query, repository.getFeature(x), dist_func_param));
            }
            graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());
        }
    }

    auto rnd = std::mt19937(7);
    auto rnd_neighbor = std::uniform_int_distribution<uint32_t>(0, edges_per_vertex - 1);

    for (uint32_t label = edges_per_vertex + 1; label < vertex_count; label++) {
        const auto new_vertex_feature = repository.getFeature(label);
        const auto internal_index = graph.addVertex(label, new_vertex_feature);
        auto top_list = std::uniform_int_distribution<uint32_t>(0, label - 1);

        auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
        while (new_neighbors.size() < edges_per_vertex) {
            const auto candidate_index = (uint32_t)top_list(rnd);
            const auto new_neighbor_index = graph.getNeighborIndices(candidate_index)[rnd_neighbor(rnd)];

            bool duplicate = false;
            for (const auto& neighbor : new_neighbors) {
                if (neighbor.first == candidate_index || neighbor.first == new_neighbor_index) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;

            const auto candidate_dist = dist_func(new_vertex_feature, graph.getFeatureVector(candidate_index), dist_func_param);
            graph.changeEdge(candidate_index, new_neighbor_index, internal_index, candidate_dist);
            new_neighbors.emplace_back(candidate_index, candidate_dist);

            const auto new_neighbor_dist = dist_func(new_vertex_feature, graph.getFeatureVector(new_neighbor_index), dist_func_param);
            graph.changeEdge(new_neighbor_index, candidate_index, internal_index, new_neighbor_dist);
            new_neighbors.emplace_back(new_neighbor_index, new_neighbor_dist);
        }

        std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        auto neighbor_indices = std::vector<uint32_t>();
        auto neighbor_weights = std::vector<float>();
        for (auto&& neighbor : new_neighbors) {
            neighbor_indices.emplace_back(neighbor.first);
            neighbor_weights.emplace_back(neighbor.second);
        }
        graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());

        if ((label + 1) % 100000 == 0 || (label + 1) == vertex_count) {
            auto quality = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
            log("{:7} elements, in {:5}s, AEW {:4.2f}, connected {} \n", (label + 1), duration, quality, connected);
        }
    }

    const auto valid = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, vertex_count, true);
    if (valid == false) log("WARNING: Invalid graph detected during build\n");

    return graph;
}

inline void create_graph(const deglib::StaticFeatureRepository& repository,
                         const DataStreamType data_stream_type,
                         const std::string& graph_file,
                         deglib::Metric metric,
                         deglib::builder::OptimizationTarget lid,
                         const uint8_t k,
                         const uint8_t k_ext,
                         const float eps_ext,
                         const uint8_t k_opt,
                         const float eps_opt,
                         const uint8_t i_opt,
                         const uint32_t thread_count,
                         const bool use_rng = true,
                         const uint32_t scale = 1,
                         const bool use_path_verification = false,
                         const deglib::cpu::InstructionSet instruction = deglib::cpu::InstructionSet::Auto) {
    auto rnd = std::mt19937(7);
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;

    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric, instruction);
    const auto feature_byte_size = feature_space.get_data_size();

    log("Initializing empty graph (capacity: {} vertices, {}D {} {} feature space using {})\n",
        repository.size(),
        repository.dims(),
        metric.get_distance_name(),
        metric.get_data_type_name(),
        feature_space.get_instruction());

    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, k, feature_space);

    auto builder = deglib::builder::EvenRegularGraphBuilder(
        graph, rnd, lid, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries);
    builder.setThreadCount(thread_count);
    builder.setBatchSize(10, 10);

    auto base_size = uint32_t(repository.size());
    auto addEntry = [&builder, &repository, feature_byte_size](auto idx) {
        auto feature = repository.getFeature(idx);
        auto feature_vector = std::vector<std::byte>{feature, feature + feature_byte_size};
        builder.addEntry(idx, std::move(feature_vector));
    };

    if (data_stream_type == DataStreamType::AddHalfRemoveAndAddOneAtATime) {
        auto base_size_half = base_size / 2;
        auto base_size_fourth = base_size / 4;
        for (uint32_t i = 0; i < base_size_fourth; i++) {
            addEntry(0 + i);
            addEntry(base_size_half + i);
        }
        for (uint32_t i = 0; i < base_size_fourth; i++) {
            addEntry(base_size_fourth + i);
            addEntry(base_size_half + base_size_fourth + i);
            builder.removeEntry(base_size_half + (i * 2) + 0);
            builder.removeEntry(base_size_half + (i * 2) + 1);
        }
    } else {
        base_size /= (data_stream_type == DataStreamType::AddHalf) ? 2 : 1;
        for (uint32_t i = 0; i < base_size; i++) addEntry(i);

        if (data_stream_type == DataStreamType::AddAllRemoveHalf)
            for (uint32_t i = base_size / 2; i < base_size; i++) builder.removeEntry(i);
    }

    const auto log_after = (base_size <= 100000) ? 10000 : 100000;

    log("Start building graph with {} threads\n", thread_count);
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        if (status.added % log_after == 0 || size == base_size) {
            duration_ms +=
                uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto valid_weights =
                deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            log("{:7} vertices, {:8}ms, {:8} / {:8} improv, AEW: {:4.2f}, {} connected & {}, RSS {} & peakRSS {}\n",
                size,
                duration,
                status.improved,
                status.tries,
                avg_edge_weight,
                connected ? "" : "not",
                valid_weights ? "valid" : "invalid",
                currRSS,
                peakRSS);
            start = std::chrono::steady_clock::now();
        } else if (status.added % (log_after / 10) == 0) {
            duration_ms +=
                uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            log("{:7} vertices, {:8}ms, {:8} / {:8} improv, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}\n",
                size,
                duration,
                status.improved,
                status.tries,
                avg_edge_weight,
                connected ? "" : "not",
                currRSS,
                peakRSS);
            start = std::chrono::steady_clock::now();
        }
    };

    builder.build(improvement_callback, false);
    log("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n",
        getCurrentRSS() / 1000000,
        getPeakRSS() / 1000000,
        duration_ms / 1000);

    graph.saveGraph(graph_file.c_str());
    log("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}

inline void optimize_graph(deglib::graph::SizeBoundedGraph& graph,
                           const uint8_t k_opt,
                           const float eps_opt,
                           const uint8_t i_opt,
                           const uint64_t total_iterations,
                           const uint64_t log_interval = 10000,
                           const uint32_t scale = 1) {
    auto rnd = std::mt19937(7);
    auto builder = deglib::builder::EvenRegularGraphBuilder(
        graph, rnd, deglib::builder::OptimizationTarget::LowLID, 0, 0, k_opt, eps_opt, i_opt, 1, 0);

    auto initial_avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
    log("Optimizing graph with initial AEW {:.2f}\n", initial_avg_edge_weight);

    auto start = std::chrono::steady_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    uint64_t duration_ms = 0;

    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto tries = status.tries;
        const auto improved = status.improved;

        if (log_interval > 0 && tries > 0 && tries % log_interval == 0) {
            duration_ms +=
                uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto duration = duration_ms / 1000;
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto diff = tries - last_status.tries;
            auto avg_improv = (diff > 0) ? uint32_t((improved - last_status.improved) / diff) : 0;

            log("{:5}s, {:8} / {:8} iterations (avg {:2} improvements), AEW {:.2f}, connected {}\n",
                duration,
                improved,
                tries,
                avg_improv,
                avg_edge_weight,
                connected);

            last_status = status;
            start = std::chrono::steady_clock::now();
        }

        if (tries >= total_iterations) builder.stop();
    };

    builder.build(improvement_callback, true);

    log("Optimization complete. Final AEW: {:.2f}, non-RNG edges: {}\n",
        deglib::analysis::calc_avg_edge_weight(graph, scale),
        deglib::analysis::calc_non_rng_edges(graph));
}

}  // namespace deglib::benchmark
