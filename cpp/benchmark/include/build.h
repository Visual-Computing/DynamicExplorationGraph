#pragma once

/**
 * @file build.h
 * @brief Graph building utilities for deglib benchmarks.
 * 
 * This header provides functions for building and optimizing graphs:
 * - create_random_exploration_graph: Build a random regular graph (returns graph)
 * - optimize_graph: Optimize an existing graph for a specified number of iterations
 * - improve_and_save_checkpoints: Optimize graph and save at intervals
 * - create_graph: Build a DEG graph with the builder
 * - create_incremental_graphs: Build graphs incrementally with increasing data sizes
 */

#include <random>
#include <chrono>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <functional>

#include <fmt/core.h>
#include <fmt/format.h>

#include "deglib.h"
#include "analysis.h"
#include "benchmark.h"

namespace deglib::benchmark
{

    
/*
 * Stream types for the benchmarks. These are shared across the benchmark programs and
 * were previously declared inside multiple files. Move the canonical definition here
 * so we can reuse it across the tools.
 */
enum DataStreamType { AddAll, AddHalf, AddAllRemoveHalf, AddHalfRemoveAndAddOneAtATime };



/**
 * Create a random regular graph from the repository.
 * This creates a baseline graph without optimization, useful for testing
 * the effect of optimization on search performance.
 * 
 * @param repository The feature repository containing the data.
 * @param metric Distance metric to use.
 * @param k Edges per vertex (graph degree).
 * @param max_size Maximum number of vertices (0 = use all).
 * @return The created SizeBoundedGraph.
 */
inline deglib::graph::SizeBoundedGraph create_random_graph(
    const deglib::StaticFeatureRepository& repository, 
    deglib::Metric metric, 
    const uint8_t k, 
    const uint32_t max_size = 0) 
{
    log("Build a random EG{}\n", k);

    const auto dims = repository.dims();
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    // create the graph and add all vertices, without any edges
    const auto start = std::chrono::system_clock::now();
    const uint8_t edges_per_vertex = k;
    const uint32_t vertex_count = (max_size > 0 && max_size < repository.size()) ? max_size : uint32_t(repository.size());
    auto graph = deglib::graph::SizeBoundedGraph(vertex_count, edges_per_vertex, feature_space);

    // add the initial vertices (edges_per_vertex + 1)
    {
        const auto size = (uint32_t)(edges_per_vertex + 1);
        for (uint32_t y = 0; y < size; y++) {
            const auto query = repository.getFeature(y);
            const auto internal_index = graph.addVertex(y, query);

            auto neighbor_indices = std::vector<uint32_t>();
            auto neighbor_weights = std::vector<float>();
            for (uint32_t x = 0; x < size; x++) {
                if(x == internal_index) continue;
                neighbor_indices.emplace_back(x);
                neighbor_weights.emplace_back(dist_func(query, repository.getFeature(x), dist_func_param));
            }
            graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());
        }
    }

    // random order of vertices
    auto rnd = std::mt19937(7);
    auto rnd_neighbor = std::uniform_int_distribution<uint32_t>(0, edges_per_vertex - 1);

    // add the remaining vertices
    for (uint32_t label = edges_per_vertex + 1; label < vertex_count; label++) {
        const auto new_vertex_feature = repository.getFeature(label);
        const auto internal_index = graph.addVertex(label, new_vertex_feature);
        auto top_list = std::uniform_int_distribution<uint32_t>(0, label - 1);
        
        // remove the worst edge of the good neighbors and connect them with this new vertex
        auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
        while (new_neighbors.size() < edges_per_vertex) {
            const auto candidate_index = (uint32_t) top_list(rnd);

            if(graph.hasEdge(candidate_index, internal_index))
                continue;

            uint32_t new_neighbor_index = 0;
            float new_neighbor_weight = -1;
            const auto neighbor_weights = graph.getNeighborWeights(candidate_index);
            const auto neighbor_indices = graph.getNeighborIndices(candidate_index);
            while(new_neighbor_weight < 0) {
                const auto edge_idx = (uint32_t) rnd_neighbor(rnd);
                const auto neighbor_index = neighbor_indices[edge_idx];
                const auto neighbor_weight = neighbor_weights[edge_idx];

                if(graph.hasEdge(neighbor_index, internal_index) == false) {
                    new_neighbor_index = neighbor_index;
                    new_neighbor_weight = neighbor_weight;
                }          
            }

            if(new_neighbor_weight < 0) {
                log("ERROR: could not find edge in neighbor list of vertex {} to connect to vertex {}\n", candidate_index, internal_index);
                abort();
            }

            const auto candidate_dist = dist_func(new_vertex_feature, graph.getFeatureVector(candidate_index), dist_func_param); 
            graph.changeEdge(candidate_index, new_neighbor_index, internal_index, candidate_dist);
            new_neighbors.emplace_back(candidate_index, candidate_dist);

            const auto new_neighbor_dist = dist_func(new_vertex_feature, graph.getFeatureVector(new_neighbor_index), dist_func_param);
            graph.changeEdge(new_neighbor_index, candidate_index, internal_index, new_neighbor_dist);
            new_neighbors.emplace_back(new_neighbor_index, new_neighbor_dist);
        }

        std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        auto neighbor_indices = std::vector<uint32_t>();
        auto neighbor_weights = std::vector<float>();
        for (auto &&neighbor : new_neighbors) {
            neighbor_indices.emplace_back(neighbor.first);
            neighbor_weights.emplace_back(neighbor.second);
        }
        graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());

        if((label+1) % 100000 == 0 || (label+1) == vertex_count) {
            auto quality = deglib::analysis::calc_avg_edge_weight(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
            log("{:7} elements, in {:5}s, AEW {:4.2f}, connected {} \n", (label+1), duration, quality, connected);
        }
    }

    const auto valid = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, vertex_count, true);
    if(valid == false) 
        log("WARNING: Invalid graph detected during build\n");

    return graph;
}


/**
 * Build a DEG graph using the EvenRegularGraphBuilder.
 * 
 * @param repository The feature repository containing the data.
 * @param data_stream_type How to add/remove data during build.
 * @param graph_file Output graph file path.
 * @param metric Distance metric to use.
 * @param lid Optimization target (LowLID, HighLID, etc.).
 * @param k Edges per vertex (graph degree).
 * @param k_ext Extended search neighborhood for building.
 * @param eps_ext Epsilon for extended search.
 * @param k_opt Optimization neighborhood size.
 * @param eps_opt Optimization epsilon.
 * @param i_opt Optimization path length.
 * @param thread_count Number of threads to use.
 * @param use_rng Enable RNG (Relative Neighborhood Graph) pruning during graph extension (default: true).
 */
inline void create_graph(
    const deglib::StaticFeatureRepository& repository, 
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
    const bool use_rng = true) 
{
    auto rnd = std::mt19937(7);
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;
    const uint32_t scale = 1000;

    log("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto feature_byte_size = feature_space.get_data_size();
    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, k, feature_space);

    log("Start graph builder (RNG pruning: {})\n", use_rng ? "enabled" : "disabled");   
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, lid, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries, use_rng);
    builder.setThreadCount(thread_count);
    if(thread_count == 1)
        builder.setBatchSize(1,1);
    
    auto base_size = uint32_t(repository.size());
    auto addEntry = [&builder, &repository, feature_byte_size] (auto idx) {
        auto feature = repository.getFeature(idx);
        auto feature_vector = std::vector<std::byte>{feature, feature + feature_byte_size};
        builder.addEntry(idx, std::move(feature_vector));
    };
    
    if(data_stream_type == DataStreamType::AddHalfRemoveAndAddOneAtATime) {
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
        for (uint32_t i = 0; i < base_size; i++) 
            addEntry(i);

        if(data_stream_type == DataStreamType::AddAllRemoveHalf) 
            for (uint32_t i = base_size/2; i < base_size; i++) 
                builder.removeEntry(i);
    }

    const auto log_after = 100000;

    log("Start building \n");    
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        if(status.step % log_after == 0 || size == base_size) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true, scale);
            auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false, scale);
            auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            log("{:7} vertices, {:8}ms, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                        size, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
        else if(status.step % (log_after/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            log("{:7} vertices, {:8}ms, {:8} / {:8} improv, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}\n", 
                size, duration, status.improved, status.tries, avg_edge_weight, connected ? "" : "not", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
    };

    builder.build(improvement_callback, false);
    log("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000, duration_ms / 1000);

    graph.saveGraph(graph_file.c_str());
    log("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}

/**
 * Create multiple graphs incrementally using more data from the repository.
 * Builds graphs at specified size intervals, all in the same directory.
 * 
 * @param repository The feature repository containing the data.
 * @param output_dir Directory to save graphs.
 * @param graph_name_base Base name for graph files.
 * @param interval Add this many vertices between each saved graph.
 * @param metric Distance metric to use.
 * @param lid Optimization target.
 * @param k Edges per vertex (graph degree).
 * @param k_ext Extended search neighborhood for building.
 * @param eps_ext Epsilon for extended search.
 * @param k_opt Optimization neighborhood size.
 * @param eps_opt Optimization epsilon.
 * @param i_opt Optimization path length.
 * @param thread_count Number of threads to use.
 * @param use_rng Enable RNG (Relative Neighborhood Graph) pruning during graph extension (default: true).
 * @return Vector of pairs (graph_path, vertex_count) for each saved graph.
 */
inline std::vector<std::pair<std::string, uint32_t>> create_incremental_graphs(
    const deglib::StaticFeatureRepository& repository, 
    const std::string& output_dir,
    const std::string& graph_name_base,
    const uint32_t step_size,
    deglib::Metric metric, 
    deglib::builder::OptimizationTarget lid, 
    const uint8_t k, 
    const uint8_t k_ext, 
    const float eps_ext, 
    const uint8_t k_opt, 
    const float eps_opt, 
    const uint8_t i_opt, 
    const uint32_t thread_count,
    const bool use_rng = true) 
{
    std::vector<std::pair<std::string, uint32_t>> created_files;
    auto rnd = std::mt19937(7);
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;
    const uint32_t scale = 1000;

    log("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto feature_byte_size = feature_space.get_data_size();
    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, k, feature_space);

    log("Start graph builder (RNG pruning: {})\n", use_rng ? "enabled" : "disabled");   
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, lid, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries, use_rng);
    builder.setThreadCount(thread_count);
    if(thread_count == 1)
        builder.setBatchSize(1,1);
    
    const auto log_after = 100000;

    log("Start building\n");    
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        if(status.step % log_after == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true, scale);
            auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false, scale);
            auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;
            log("{:7} vertices, {:8}ms, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}\n", 
                        size, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid");
            start = std::chrono::steady_clock::now();
        }
    };

    uint32_t current_count = 0;
    uint32_t total_size = uint32_t(repository.size());

    while(current_count < total_size) {
        uint32_t target_size = std::min(current_count + step_size, total_size);
        
        log("Adding data from {} to {}\n", current_count, target_size);

        for (uint32_t i = current_count; i < target_size; i++) {
            auto feature = repository.getFeature(i);
            auto feature_vector = std::vector<std::byte>{feature, feature + feature_byte_size};
            builder.addEntry(i, std::move(feature_vector)); 
        }
        current_count = target_size;

        // Start the build process
        builder.build(improvement_callback, false);
        log("Built graph in {} secs\n", duration_ms / 1000);

        // Construct filename with size suffix (e.g., _100k)
        std::string size_str = std::to_string(target_size / 1000) + "k";
        std::string graph_file = fmt::format("{}/{}_{}.deg", output_dir, graph_name_base, size_str);
        
        graph.saveGraph(graph_file.c_str());
        created_files.push_back({graph_file, target_size});

        log("Saved graph: {}, non-RNG edges: {}\n", graph_file, deglib::analysis::calc_non_rng_edges(graph));
    }

    return created_files;
}


/**
 * Optimize an existing graph by running the builder's improve step.
 * This is a simple optimization function that runs for a specified number of iterations.
 * 
 * @param graph The graph to optimize (modified in place).
 * @param k_opt Optimization neighborhood size.
 * @param eps_opt Optimization epsilon for search.
 * @param i_opt Optimization path length.
 * @param total_iterations Total optimization iterations to run.
 * @param log_interval Log progress every N iterations (0 = disable, default: 10000).
 */
inline void optimize_graph(
    deglib::graph::SizeBoundedGraph& graph,
    const uint8_t k_opt, 
    const float eps_opt, 
    const uint8_t i_opt,
    const uint64_t total_iterations,
    const uint64_t log_interval = 10000)
{
    auto rnd = std::mt19937(7);
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, deglib::builder::OptimizationTarget::LowLID, 0, 0, k_opt, eps_opt, i_opt, 1, 0);

    auto initial_avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
    log("Optimizing graph with initial AEW {:.2f}\n", initial_avg_edge_weight);

    auto start = std::chrono::steady_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    uint64_t duration_ms = 0;
    
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto tries = status.tries;
        const auto improved = status.improved;
        
        // Log progress at intervals
        if(log_interval > 0 && tries > 0 && tries % log_interval == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto duration = duration_ms / 1000;
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto diff = tries - last_status.tries;
            auto avg_improv = (diff > 0) ? uint32_t((improved - last_status.improved) / diff) : 0;

            log("{:5}s, {:8} / {:8} iterations (avg {:2} improvements), AEW {:.2f}, connected {}\n", 
                duration, improved, tries, avg_improv, avg_edge_weight, connected);
            
            last_status = status;
            start = std::chrono::steady_clock::now();
        }
        
        // stop after total_iterations
        if(tries >= total_iterations) 
            builder.stop();
    };
    
    builder.build(improvement_callback, true);
    
    log("Optimization complete. Final AEW: {:.2f}, non-RNG edges: {}\n", 
        deglib::analysis::calc_avg_edge_weight(graph, 1), deglib::analysis::calc_non_rng_edges(graph));
}


/**
 * Improve a graph and test it at regular intervals, saving checkpoints.
 * This is based on improve_and_test from deglib_build_bench.cpp.
 * 
 * @param graph The graph to optimize (modified in place).
 * @param output_dir Directory to save checkpoint graphs.
 * @param graph_name_base Base name for checkpoint files (without extension).
 * @param k_opt Optimization neighborhood size.
 * @param eps_opt Optimization epsilon for search.
 * @param i_opt Optimization path length.
 * @param save_interval Save checkpoint every N iterations.
 * @param max_iterations Total optimization iterations (0 = 10 * save_interval).
 * @param query_repository Query features for recall testing.
 * @param ground_truth Ground truth as vector of unordered_sets for recall estimation.
 * @param k_test k parameter for testing.
 * @param max_distance_count_test Max distance count for recall estimation.
 * @return Vector of pairs (graph_path, iteration_count) for each saved checkpoint.
 */
inline std::vector<std::pair<std::string, uint64_t>> improve_and_test(
    deglib::graph::SizeBoundedGraph& graph,
    const std::string& output_dir,
    const std::string& graph_name_base,
    const uint8_t k_opt, 
    const float eps_opt, 
    const uint8_t i_opt,
    const uint64_t save_interval,
    const uint64_t max_iterations,
    const deglib::StaticFeatureRepository& query_repository,
    const std::vector<std::unordered_set<uint32_t>>& ground_truth,
    const uint32_t k_test,
    const uint32_t max_distance_count_test = 2000) 
{
    std::vector<std::pair<std::string, uint64_t>> created_files;
    
    auto rnd = std::mt19937(7);
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, deglib::builder::OptimizationTarget::LowLID, 0, 0, k_opt, eps_opt, i_opt, 1, 0);

    auto initial_recall = deglib::benchmark::estimate_recall(graph, query_repository, ground_truth, max_distance_count_test, k_test);
    auto initial_avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
    log("Improve and test graph with initial AEW {:.2f}, Recall {}\n", 
        initial_avg_edge_weight, fmt::join(initial_recall, ", "));

    auto start = std::chrono::steady_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    uint64_t duration_ms = 0;
    const uint64_t total_iterations = (max_iterations > 0) ? max_iterations : (10 * save_interval);
    
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto tries = status.tries;
        const auto improved = status.improved;
        
        if(tries > 0 && tries % save_interval == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto duration = duration_ms / 1000;
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto diff = tries - last_status.tries;
            auto avg_improv = (diff > 0) ? uint32_t((improved - last_status.improved) / diff) : 0;
            auto avg_tries = (diff > 0) ? uint32_t((tries - last_status.tries) / diff) : 0;

            // check the graph from time to time
            auto valid = deglib::analysis::check_graph_regularity(graph, (uint32_t)graph.size(), true);
            if(valid == false) {
                builder.stop();
                log("Invalid graph, build process is stopped\n");
                return;
            } 

            // Save checkpoint
            auto graph_file = fmt::format("{}/{}_it{}.deg", output_dir, graph_name_base, tries);
            graph.saveGraph(graph_file.c_str());
            created_files.push_back({graph_file, tries});

            // Test recall
            auto recall = deglib::benchmark::estimate_recall(graph, query_repository, ground_truth, max_distance_count_test, k_test);

            log("{:5}s, with {:8} / {:8} improvements (avg {:2}/{:3}), AEW {:.2f}, Recall {}, connected {}\n", 
                duration, improved, tries, avg_improv, avg_tries, avg_edge_weight, fmt::join(recall, ", "), connected);
            
            last_status = status;
            start = std::chrono::steady_clock::now();
        }
        
        // stop after total_iterations
        if(tries >= total_iterations) 
            builder.stop();
    };
    
    builder.build(improvement_callback, true);
    
    log("Optimization complete. Final AEW: {:.2f}, non-RNG edges: {}\n", 
        deglib::analysis::calc_avg_edge_weight(graph, 1), deglib::analysis::calc_non_rng_edges(graph));
    
    return created_files;
}


} // namespace deglib::benchmark
