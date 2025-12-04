#include <random>
#include <chrono>
#include <unordered_set>
#include <filesystem>
#include <atomic>

#include <fmt/core.h>
#include <omp.h>

#include "benchmark.h"
#include "deglib.h"

/**
 * The last three types are used to reproduce the deletion tests in our paper
 */
enum DataStreamType { AddAll, AddHalf, AddAllRemoveHalf, AddHalfRemoveAndAddOneAtATime };




void change_features(const std::string initial_graph_file, const std::string repository_file, const deglib::Metric metric, const std::string graph_file) {
    const auto start = std::chrono::steady_clock::now();

    fmt::print("Load graph {} \n", initial_graph_file);
    auto initial_graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    const auto vertex_count = initial_graph.size();
    const auto edge_per_vertex = initial_graph.getEdgesPerVertex();
    fmt::print("Graph with {} vertices and containing {} non-RNG edges\n", initial_graph.size(), deglib::analysis::calc_non_rng_edges(initial_graph));

    // load data
    fmt::print("Load Data \n");
    auto repository = deglib::load_static_repository(repository_file.c_str());   
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a new graph
    fmt::print("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, edge_per_vertex, feature_space);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto label = initial_graph.getExternalLabel(i);
        const auto neighbor_indices = initial_graph.getNeighborIndices(i);
        const auto neighbor_weights = initial_graph.getNeighborWeights(i);

        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(i));
        const auto internal_index = graph.addVertex(label, feature);

        graph.changeEdges(internal_index, neighbor_indices, neighbor_weights);
    }
    const auto duration_ms = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("Changing feature in {} ms. Final graph contains {} non-RNG edges\n", duration_ms, deglib::analysis::calc_non_rng_edges(graph));
}

void remove_non_mrng_edges_2(const std::string initial_graph_file, const std::string graph_file) {
    fmt::print("Load graph {} \n", initial_graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    fmt::print("Graph with {} vertices and containing {} non-RNG edges\n", graph.size(), deglib::analysis::calc_non_rng_edges(graph));

    const auto vertex_count = graph.size();
    const auto edge_per_vertex = graph.getEdgesPerVertex();

    const auto start = std::chrono::steady_clock::now();
    std::vector<deglib::builder::GraphEdge> nonMRNG_edges;
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto vertex_index = i;
        const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
        const auto neighbor_weights = graph.getNeighborWeights(vertex_index);

        // find all none rng conform neighbors
        for (uint32_t n = 0; n < edge_per_vertex; n++) {
            const auto neighbor_index = neighbor_indices[n];
            const auto neighbor_weight = neighbor_weights[n];
            if(deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) 
                nonMRNG_edges.emplace_back(vertex_index, neighbor_index, neighbor_weight);
        }
    }
    std::sort(nonMRNG_edges.begin(), nonMRNG_edges.end(), [](const auto& x, const auto& y){return x.weight < y.weight;});

    size_t removed_rng_edges = 0;
    for (size_t i = 0; i < nonMRNG_edges.size(); i++) {
        const deglib::builder::GraphEdge& edge = nonMRNG_edges[i];
        if(deglib::analysis::checkRNG(graph, edge_per_vertex, edge.from_vertex, edge.to_vertex, edge.weight) == false) {
            graph.changeEdge(edge.from_vertex, edge.to_vertex, edge.from_vertex, 0);
            removed_rng_edges++;
        }
    }
    const auto duration_ms = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("Removed {} edges in {} ms. Final graph contains {} non-RNG edges\n", removed_rng_edges, duration_ms, deglib::analysis::calc_non_rng_edges(graph));
}

void remove_non_mrng_edges_1(const std::string initial_graph_file, const std::string graph_file) {
    fmt::print("Load graph {} \n", initial_graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    fmt::print("Graph with {} vertices and containing {} non-RNG edges\n", graph.size(), deglib::analysis::calc_non_rng_edges(graph));

    const auto vertex_count = graph.size();
    const auto edge_per_vertex = graph.getEdgesPerVertex();

    const auto start = std::chrono::steady_clock::now();
    size_t removed_rng_edges = 0;
    for (uint32_t i = 0; i < vertex_count; i++) {
        const auto vertex_index = i;

        // sort neighbors by their weight (highest to lowest)
        std::vector<std::pair<uint32_t, float>> neighbors;
        {
            const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
            const auto neighbor_weights = graph.getNeighborWeights(vertex_index);
            for (uint32_t n = 0; n < edge_per_vertex; n++) {
                const auto neighbor_index = neighbor_indices[n];
                const auto neighbor_weight = neighbor_weights[n];
                neighbors.emplace_back(neighbor_index, neighbor_weight);
            }
            std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y){return x.second < y.second;});
        }

        // find all none rng conform neighbors
        std::vector<uint32_t> nonMRNG_edges;
        for (uint32_t n = 0; n < neighbors.size(); n++) {
            const auto neighbor_index = neighbors[n].first;
            const auto neighbor_weight = neighbors[n].second;
            if(deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) 
                nonMRNG_edges.emplace_back(n);
        }

        bool removed_edge = false;
        do {
            removed_edge = false;
            for (uint32_t n = 0; n < nonMRNG_edges.size(); n++) {
                const auto neighbor_index = neighbors[nonMRNG_edges[n]].first;
                const auto neighbor_weight = neighbors[nonMRNG_edges[n]].second;

                if(deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) {
                    nonMRNG_edges.erase(nonMRNG_edges.begin() + n);
                    graph.changeEdge(vertex_index, neighbor_index, vertex_index, 0);
                    removed_rng_edges++;
                    removed_edge = true;
                    break;
                }
            }
        } while(removed_edge);
    }
    const auto duration_ms = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("Removed {} edges in {} ms. Final graph contains {} non-RNG edges\n", removed_rng_edges, duration_ms, deglib::analysis::calc_non_rng_edges(graph));
}


/**
 * Remove all none MRNG conform edges
 */
void remove_non_mrng_edges(const std::string initial_graph_file, const std::string graph_file) {
    fmt::print("Load graph {} \n", initial_graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    fmt::print("Graph with {} vertices and containing {} non-RNG edges\n", graph.size(), deglib::analysis::calc_non_rng_edges(graph));

    deglib::builder::remove_non_mrng_edges(graph);

    // store the graph
    graph.saveGraph(graph_file.c_str());
}

/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void optimize_graph(const std::string initial_graph_file, const std::string graph_file, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t iterations) {
    
    auto rnd = std::mt19937(7);                         // default 7

    fmt::print("Load graph {} \n", initial_graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    fmt::print("Graph with {} vertices and an avg edge weight of {} \n", graph.size(), deglib::analysis::calc_avg_edge_weight(graph, 100));

    deglib::builder::optimize_edges(graph, k_opt, eps_opt, i_opt, iterations);

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}

/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void reduce_graph(const std::string graph_file, deglib::builder::OptimizationTarget lid, const uint8_t d, const uint8_t k_ext, const float eps_ext, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t thread_count) {
    
    auto rnd = std::mt19937(7);                         // default 7
    const uint32_t swap_tries = 0;                      // additional swap tries between the next graph extension
    const uint32_t additional_swap_tries = 0;           // increse swap try count for each successful swap
    const uint32_t scale = 10;

    // create a new graph 
    fmt::print("Load graph {} \n", graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a graph builder to add vertices to the new graph and improve its edges
    fmt::print("Start graph builder \n");   
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, lid, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries);
    builder.setThreadCount(thread_count);

    // queue all vertices to be removed
    const uint32_t max_vertex_count = graph.size();
    for (uint32_t i = 0; i < max_vertex_count; i++) 
        builder.removeEntry(i);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup graph builder\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // check the integrity of the graph during the graph build process
    const auto log_after = 100000;

    fmt::print("Start building \n");    
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        if(status.step % log_after == 0 || size == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true, scale);
            auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false, scale);
            auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:8}ms, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                        size, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
        else 
        if(status.step % (log_after/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:8}ms, {:8} / {:8} improv, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}\n", size, duration, status.improved, status.tries, avg_edge_weight, connected ? "" : "not", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
    };

    // start the build process
    builder.build(improvement_callback, false);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000, duration_ms / 1000);

    fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}


/**
 * Load the data repository and create a random regular graph.
 * Store the graph in the graph file.
 */
void create_random_exploration_graph(const std::string repository_file, const std::string graph_file, deglib::Metric metric, const uint8_t d, const uint32_t max_size = 0) {
    fmt::print("Build and store a random EG{}\n", d);

    // load data
    fmt::print("Load Data \n");
    auto repository = deglib::load_static_repository(repository_file.c_str());   
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    const auto dims = repository.dims();
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    // create the graph and add all vertices, without any edges
    const auto start = std::chrono::system_clock::now();
    const uint8_t edges_per_vertex = d;
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

            // check if the vertex is already in the edge list of the new vertex (added during a previous loop-run)
            // since all edges are undirected and the edge information of the new vertex does not yet exist, we search the other way around.
            if(graph.hasEdge(candidate_index, internal_index))
                continue;

            // find a new random neighbor
            uint32_t new_neighbor_index = 0;
            float new_neighbor_weight = -1;
            const auto neighbor_weights = graph.getNeighborWeights(candidate_index);
            const auto neighbor_indices = graph.getNeighborIndices(candidate_index);
            while(new_neighbor_weight < 0) {
                const auto edge_idx = (uint32_t) rnd_neighbor(rnd);
                const auto neighbor_index = neighbor_indices[edge_idx];
                const auto neighbor_weight = neighbor_weights[edge_idx];

                // the suggest neighbors might already be in the edge list of the new vertex
                if(graph.hasEdge(neighbor_index, internal_index) == false) {
                    new_neighbor_index = neighbor_index;
                    new_neighbor_weight = neighbor_weight;
                }          
            }

            // this should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the result-vertex and still has space for more
            if(new_neighbor_weight < 0) {
                fmt::print("it was not possible to find an edge (best weight {}) in the neighbor list of vertex {} which would connect to vertex {} \n", new_neighbor_weight, candidate_index, internal_index);
                perror("");
                abort();
            }

            // place the new vertex in the edge list of the result-vertex
            const auto candidate_dist = dist_func(new_vertex_feature, graph.getFeatureVector(candidate_index), dist_func_param); 
            graph.changeEdge(candidate_index, new_neighbor_index, internal_index, candidate_dist);
            new_neighbors.emplace_back(candidate_index, candidate_dist);

            // place the new vertex in the edge list of the worst edge neighbor
            const auto new_neighbor_dist = dist_func(new_vertex_feature, graph.getFeatureVector(new_neighbor_index), dist_func_param);
            graph.changeEdge(new_neighbor_index, candidate_index, internal_index, new_neighbor_dist);
            new_neighbors.emplace_back(new_neighbor_index, new_neighbor_dist);
        }

        // sort the neighbors by their neighbor indices and store them in the new vertex
        std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        auto neighbor_indices = std::vector<uint32_t>();
        auto neighbor_weights = std::vector<float>();
        for (auto &&neighbor : new_neighbors) {
            neighbor_indices.emplace_back(neighbor.first);
            neighbor_weights.emplace_back(neighbor.second);
        }
        graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());

        if((label+1) % 10000 == 0 || (label+1) == vertex_count) {
            auto quality = deglib::analysis::calc_avg_edge_weight(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
            fmt::print("{:7} elements, in {:5}s, AEW {:4.2f}, connected {} \n", (label+1), duration, quality, connected);
        }
    }

    // check the graph from time to time
    const auto valid = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, vertex_count, true);
    if(valid == false) 
        fmt::print("Invalid graph, build process is stopped \n");

    // store the graph
    graph.saveGraph(graph_file.c_str());
    fmt::print("Write graph {} \n\n", graph_file.c_str());
}

/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void create_graph(const std::string repository_file, const DataStreamType data_stream_type, const std::string graph_file, deglib::Metric metric, deglib::builder::OptimizationTarget lid, const uint8_t d, const uint8_t k_ext, const float eps_ext, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t thread_count) {
    
    auto rnd = std::mt19937(7);                         // default 7
    const uint32_t swap_tries = 0;                      // additional swap tries between the next graph extension
    const uint32_t additional_swap_tries = 0;           // increse swap try count for each successful swap
    const uint32_t scale = 1000;

    // load data
    fmt::print("Load Data \n");
    auto repository = deglib::load_static_repository(repository_file.c_str());   
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a new graph
    fmt::print("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto feature_byte_size = feature_space.get_data_size();
    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, d, feature_space);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a graph builder to add vertices to the new graph and improve its edges
    fmt::print("Start graph builder \n");   
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, lid, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries);
    builder.setThreadCount(thread_count);
    if(thread_count == 1)
        builder.setBatchSize(1,1);
    
    // provide all features to the graph builder at once. In an online system this will be called multiple times
    auto base_size = uint32_t(repository.size());
    auto addEntry = [&builder, &repository, feature_byte_size] (auto idx)
    {
        auto feature = repository.getFeature(idx);
        auto feature_vector = std::vector<std::byte>{feature, feature + feature_byte_size};
        builder.addEntry(idx, std::move(feature_vector)); // TODO label offset +1 for the laion dataset
    };
    if(data_stream_type == AddHalfRemoveAndAddOneAtATime) {
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
        base_size /= (data_stream_type == AddHalf) ? 2 : 1;
        for (uint32_t i = 0; i < base_size; i++) 
            addEntry(i);

        if(data_stream_type == AddAllRemoveHalf) 
            for (uint32_t i = base_size/2; i < base_size; i++) 
                builder.removeEntry(i);
    }
    repository.clear();
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup graph builder\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // check the integrity of the graph during the graph build process
    const auto log_after = 100000;

    fmt::print("Start building \n");    
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
            auto duration = duration_ms;// / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:8}ms, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                        size, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
        else 
        if(status.step % (log_after/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms;// / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:8}ms, {:8} / {:8} improv, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}\n", size, duration, status.improved, status.tries, avg_edge_weight, connected ? "" : "not", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
    };

    // start the build process
    builder.build(improvement_callback, false);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000, duration_ms / 1000);

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}

/**
 * Create multiple graphs incrementally using more data from the repository file.
 * Returns a list of the created graph files and their sizes.
 */
std::vector<std::pair<std::string, uint32_t>> create_incremental_graphs(const std::string repository_file, const std::string graph_file_base, const uint32_t step_size, deglib::Metric metric, deglib::builder::OptimizationTarget lid, const uint8_t d, const uint8_t k_ext, const float eps_ext, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t thread_count) {
    
    std::vector<std::pair<std::string, uint32_t>> created_files;
    auto rnd = std::mt19937(7);                         // default 7
    const uint32_t swap_tries = 0;                      // additional swap tries between the next graph extension
    const uint32_t additional_swap_tries = 0;           // increse swap try count for each successful swap
    const uint32_t scale = 1000;

    // load data
    fmt::print("Load Data \n");
    auto repository = deglib::load_static_repository(repository_file.c_str());   
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a new graph
    fmt::print("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto feature_byte_size = feature_space.get_data_size();
    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, d, feature_space);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a graph builder to add vertices to the new graph and improve its edges
    fmt::print("Start graph builder \n");   
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, lid, k_ext, eps_ext, k_opt, eps_opt, i_opt, swap_tries, additional_swap_tries);
    builder.setThreadCount(thread_count);
    if(thread_count == 1)
        builder.setBatchSize(1,1);
    
    // check the integrity of the graph during the graph build process
    const auto log_after = 100000;

    fmt::print("Start building \n");    
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
            auto duration = duration_ms;// / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:8}ms, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                        size, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
    };

    uint32_t current_count = 0;
    uint32_t total_size = uint32_t(repository.size());

    while(current_count < total_size) {
        uint32_t target_size = std::min(current_count + step_size, total_size);
        
        fmt::print("Adding data from {} to {}\n", current_count, target_size);

        for (uint32_t i = current_count; i < target_size; i++) {
            auto feature = repository.getFeature(i);
            auto feature_vector = std::vector<std::byte>{feature, feature + feature_byte_size};
            builder.addEntry(i, std::move(feature_vector)); 
        }
        current_count = target_size;

        // start the build process
        builder.build(improvement_callback, false);
        fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000, duration_ms / 1000);

        // construct filename
        std::filesystem::path p(graph_file_base);
        std::string extension = p.extension().string();
        std::string stem = p.stem().string();
        std::string size_str = std::to_string(target_size / 1000) + "k";
        std::string new_filename = (p.parent_path() / (stem + "_" + size_str + extension)).string();

        // store the graph
        graph.saveGraph(new_filename.c_str());
        created_files.push_back({new_filename, target_size});

        fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
    }

    return created_files;
}

/**
 * Load the graph from the drive and test it against the SIFT query data.
 */
void test_graph(const std::string query_file, const std::string gt_file, const std::string graph_file, const uint32_t repeat, const uint32_t threads, const uint32_t k) {

    // load an existing graph
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    // const auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading the graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    const auto query_repository = deglib::load_static_repository(query_file.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

    size_t dims_out;
    size_t count_out;
    const auto ground_truth_f = deglib::fvecs_read(gt_file.c_str(), dims_out, count_out);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", count_out, dims_out);

    deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t)dims_out, repeat, threads, k);
}

static std::vector<float> estimate_recall(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, const std::vector<std::unordered_set<uint32_t>>& answer, const uint32_t max_distance_count, const uint32_t k) {

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
                if (gt.find(external_id) != gt.end()) local_correct++;
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
 * Load the graph from the drive, improve it and test it against the query data.
 * The tests are only used to give an estimate of the current quality, 
 * which is why the max_distance_count_test ideally is kept low (e.g. 2000).
 * 
 * @param initial_graph_file The initial graph to be improved.
 * @param query_path The path to the query repository.
 * @param gt_path The path to the ground truth file.
 * @param k_opt The optimization parameter k.
 * @param eps_opt The optimization parameter eps.
 * @param max_distance_count_test The maximum distance count for testing.
 * @param k_test The k parameter for testing.
 */
static void improve_and_test(const std::string initial_graph_file, const std::string query_path, const std::string gt_path, const uint8_t k_opt, const float eps_opt, const uint64_t log_after, const uint32_t max_distance_count_test, const uint32_t k_test) {

    // Load query repository
    auto query_repository = deglib::load_static_repository(query_path.c_str());
    
    // Load ground truth
    size_t dims_out;
    size_t count_out;
    const auto ground_truth_f = deglib::fvecs_read(gt_path.c_str(), dims_out, count_out);
    const auto ground_truth = (uint32_t*)ground_truth_f.get();
    auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), (uint32_t)dims_out, k_test);

    auto rnd = std::mt19937(7);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    
    // Adapted constructor call
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, deglib::builder::OptimizationTarget::LowLID, 0, 0, k_opt, eps_opt, 5, 1, 0);

    auto initial_recall = estimate_recall(graph, query_repository, answer, max_distance_count_test, k_test);
    auto initial_graph_quality = 0.0f;//deglib::analysis::calc_graph_quality(graph);
    auto initial_avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
    auto initial_avg_neighbor_rank = 0.0f;//deglib::analysis::calc_avg_neighbor_rank(graph);
    fmt::print("Improve and test graph {} with initial GQ {:.4f}, AEW {:.2f}, ANR {:.2f}, Recall {}, \n", initial_graph_file, initial_graph_quality, initial_avg_edge_weight, initial_avg_neighbor_rank, fmt::join(initial_recall, ", "));

    auto start = std::chrono::steady_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto tries = status.tries;
        const auto improved = status.improved;
        if(tries > 0 && tries % log_after == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto duration = duration_ms / 1000;
            auto graph_quality = 0.0f;//deglib::analysis::calc_graph_quality(graph);
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, 1);
            auto avg_neighbor_rank = 0.0f;//deglib::analysis::calc_avg_neighbor_rank(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto diff = tries - last_status.tries;
            auto avg_improv = uint32_t((improved - last_status.improved) / diff);
            auto avg_tries = uint32_t((tries - last_status.tries) / diff);

            // check the graph from time to time
            auto valid = deglib::analysis::check_graph_regularity(graph, (uint32_t)graph.size(), true);
            if(valid == false) {
                builder.stop();
                fmt::print("Invalid graph, build process is stopped \n");
            } 

            // test the graph quality
            std::filesystem::path p(initial_graph_file);
            std::string extension = p.extension().string();
            std::string stem = p.stem().string();
            auto graph_file = (p.parent_path() / fmt::format("{}_OptK{}Eps{:.3f}Path5_it{}{}", stem, k_opt, eps_opt, tries, extension)).string();

            graph.saveGraph(graph_file.c_str());
            auto recall = estimate_recall(graph, query_repository, answer, max_distance_count_test, k_test);

            fmt::print("{:5}s, with {:8} / {:8} improvements (avg {:2}/{:3}), GQ {:.4f}, AEW {:.2f}, ANR {:.2f}, Recall {}, connected {} \n", duration, improved, tries, avg_improv, avg_tries, graph_quality, avg_edge_weight, avg_neighbor_rank, fmt::join(recall, ", "), connected);
            last_status = status;
            start = std::chrono::steady_clock::now();
        }
        
        // stop after 10 * log_after
        if(tries >= 10 * log_after) 
            builder.stop();

    };
    builder.build(improvement_callback, true);
}

int main() {

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb \n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    omp_set_num_threads(8);
    std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;

    const auto data_path = std::filesystem::path(DATA_PATH);
  
    // // ------------------------------- UQ-V -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "uqv" / "uqv_base.fvecs").string();
    // const auto query_file           = (data_path / "uqv" / "uqv_query.fvecs").string();
    // const auto gt_file              = (data_path / "uqv" / "uqv_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "crEG" / "256D_L2_K20_AddK40Eps0.1_schemeD.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "crEG" / "256D_L2_K20_AddK40Eps0.1_schemeD_OptK30Eps0.001Path5_it200000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::LowLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
// 
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, metric, lid, 20, 40, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    //     //optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 200000); // k_opt, eps_opt, i_opt, iteration
    // }
    // test_graph(query_file, gt_file, graph_file, 1, 1, 100); // repeat_test, test_threads, k
    

    // // ------------------------------- msong -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "msong" / "msong_base.fvecs").string();
    // const auto query_file           = (data_path / "msong" / "msong_query.fvecs").string();
    // const auto gt_file              = (data_path / "msong" / "msong_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "crEG" / "420D_L2_K30_AddK30Eps0.1_schemeC.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "crEG" / "420D_L2_K30_AddK30Eps0.1_schemeC_OptK30Eps0.001Path5_it200000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::HighLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
// 
    // if(std::filesystem::exists(graph_file.c_str()) == false) 
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, metric, lid, 30, 30, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    // if(std::filesystem::exists(opt_graph_file.c_str()) == false) 
    //     optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 200000); // k_opt, eps_opt, i_opt, iteration
    // test_graph(query_file, gt_file, opt_graph_file, 1, 1, 100); // repeat_test, test_threads, k


    // // ------------------------------- audio -----------------------------------------
    // const auto repository_file      = (data_path / "audio" / "audio_base.fvecs").string();
    // const auto query_file           = (data_path / "audio" / "audio_query.fvecs").string();
    // const auto gt_file              = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "192D_L2_K20_AddK40Eps0.1_schemeD.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "192D_L2_K20_AddK40Eps0.1_schemeD_OptK20Eps0.001Path5_it20000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::LowLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
    
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, metric, lid, 20, 40, 0.1f, 20, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //     optimze_graph(graph_file, opt_graph_file, 20, 0.001f, 5, 20000); // k_opt, eps_opt, i_opt, iteration
    // }
    // test_graph(query_file, gt_file, opt_graph_file, 50, 1, 20); // repeat_test, test_threads, k



    // // ------------------------------- enron -----------------------------------------
    // const auto repository_file      = (data_path / "enron" / "enron_base.fvecs").string();
    // const auto query_file           = (data_path / "enron" / "enron_query.fvecs").string();
    // const auto gt_file              = (data_path / "enron" / "enron_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "1369D_L2_K30_AddK60Eps0.1_schemeD.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "1369D_L2_K30_AddK60Eps0.1_schemeD_OptK30Eps0.001Path5_it400000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::LowLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
    
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //     optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 20000); // k_opt, eps_opt, i_opt, iteration
    // }
    // test_graph(query_file, gt_file, opt_graph_file, 20, 1, 20); //  // repeat_test, test_threads, k


    // // ------------------------------- Deep1M -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "deep1m" / "deep1m_base.fvecs").string();
    // const auto query_file           = (data_path / "deep1m" / "deep1m_query.fvecs").string();
    // const auto gt_file              = (data_path / "deep1m" / (data_stream_type == AddAll ? "deep1m_groundtruth_top1024_nb1000000.ivecs" : "deep1m_groundtruth_base500000.ivecs" )).string();
    // const auto reduce_graph_file    = (data_path / "deg" / "crEG" / "96D_L2_K30_AddK60Eps0.1_schemeD.deg").string();
    // const auto graph_file           = (data_path / "deg" / "crEG" / "scaling" / "96D_L2_K30_AddK60Eps0.1_schemeD_1000k.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "dynamic" / "96D_L2_K30_AddK60Eps0.1_add500k_schemeD_OptAfterwardsWith_SwapK30-0StepEps0.001LowPath5_it100000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::LowLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
    
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    //     //optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 100000); // k_opt, eps_opt, i_opt, iteration
    // }
    // reduce_graph(reduce_graph_file, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    //test_graph(query_file, gt_file, graph_file, 1, 1, 100);  // repeat_test, test_threads, k
    //for (uint32_t k = 128; k <= 1024; k *= 2) {
    //    fmt::print("Testing for k = {} \n", k);
    //    test_graph(query_file, gt_file, graph_file, 1, 1, k);
    //}

    // scaling test
    //const int step_size = 100000;
    //const auto graph_files = create_incremental_graphs(repository_file, graph_file, step_size, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); 
    //for (const auto& [g_file, size] : graph_files) {
    //    fmt::print("Testing incremental graph {} \n", g_file);
//
    //    // construct gt_file based on size
    //    std::filesystem::path gt_path(gt_file);
    //    std::string gt_filename = "deep1m_groundtruth_top1024_nb" + std::to_string(size) + ".ivecs";
    //    std::string current_gt_file = (gt_path.parent_path() / gt_filename).string();
//
    //    test_graph(query_file, current_gt_file, g_file, 1, 1, 100);
    //}

    // ------------------------------- Gist1m -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "gist" / "gist_base.fvecs").string();
    // const auto query_file           = (data_path / "gist" / "gist_query.fvecs").string();
    // const auto gt_file              = (data_path / "gist" / (data_stream_type == AddAll ? "gist_groundtruth.ivecs" : "gist_groundtruth_base500000.ivecs" )).string();
    // const auto graph_file           = (data_path / "deg" / "crEG" / "960D_L2_K30_AddK30Eps0.1_schemeC.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "crEG" / "960D_L2_K30_AddK60Eps0.1_schemeD_OptAfterwardsWith_SwapK30-0StepEps0.001LowPath5_it100000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::HighLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
// 
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 30, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    //     //optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 100000); // k_opt, eps_opt, i_opt, iteration
    // }
    // // reduce_graph(reduce_graph_file, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    // test_graph(query_file, gt_file, graph_file, 1, 1, 100);  // repeat_test, test_threads, k


    // // ------------------------------- crawl -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "crawl" / "crawl_base.fvecs").string();
    // const auto query_file           = (data_path / "crawl" / "crawl_query.fvecs").string();
    // const auto gt_file              = (data_path / "crawl" / (data_stream_type == AddAll ? "crawl_groundtruth.ivecs" : "crawl_groundtruth_base500000.ivecs" )).string();
    // const auto graph_file           = (data_path / "deg" / "crEG" / "300D_L2_K30_AddK60Eps0.1_schemeC.deg").string();
    // //const auto graph_file           = (data_path / "deg" / "300D_L2_K30_AddK30Eps0.2High.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "crEG" / "300D_L2_K30_AddK60Eps0.1_schemeD_OptAfterwardsWith_SwapK30-0StepEps0.001LowPath5_it100000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::HighLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
// 
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    //     //optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 100000); // k_opt, eps_opt, i_opt, iteration
    // }
    // // reduce_graph(reduce_graph_file, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    // test_graph(query_file, gt_file, graph_file, 1, 1, 100);  // repeat_test, test_threads, k


    // ------------------------------- SIFT1M -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    // const auto query_file           = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    // const auto gt_file              = (data_path / "SIFT1M" / (data_stream_type == AddAll ? "sift_groundtruth_top1024.ivecs" : "sift_groundtruth_base500000.ivecs" )).string();
    // const auto graph_file           = (data_path / "deg" / "crEG" / "schemes" / "128D_L2_K30_AddK60Eps0.1_schemeB.deg").string();
    // //const auto graph_file           = (data_path / "deg" / "crEG" / "128D_L2_K30_AddK60Eps0.1Low_schemeD_OptAfterwardsWith_SwapK30-0StepEps0.001LowPath5_it200000.deg").string();
    // // const auto reduce_graph_file    = (data_path / "deg" / "online" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    // // const auto opt_graph_file       = (data_path / "deg" / "dynamic" / "128D_L2_K30_AddK60Eps0.1_add500k_schemeD_OptAfterwardsWith_SwapK30-0StepEps0.001LowPath5_it100000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::LowLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;
//
    //if(std::filesystem::exists(graph_file.c_str()) == false) {
    ////    create_random_exploration_graph(repository_file, graph_file, metric, 30, 100000);
    //    create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //}

    // optimize graph
    // optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 100000); // k_opt, eps_opt, i_opt, iteration
    //improve_and_test(graph_file, query_file, gt_file, 30, 0.001f, 2000000, 2000, 100);

    // reduce_graph(reduce_graph_file, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //test_graph(query_file, gt_file, graph_file, 1, 1, 100);  // repeat_test, test_threads, k
    //for (uint32_t k = 1; k <= 1024; k *= 2) {
    //    fmt::print("Testing for k = {} \n", k);
    //    test_graph(query_file, gt_file, graph_file, 1, 1, k);
    //}

    //// scaling test
    //const int step_size = 100000;
    //const auto graph_files = create_incremental_graphs(repository_file, graph_file, step_size, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); 
    //for (const auto& [g_file, size] : graph_files) {
    //    fmt::print("Testing incremental graph {} \n", g_file);
//
    //    // construct gt_file based on size
    //    std::filesystem::path gt_path(gt_file);
    //    std::string gt_filename = "sift_groundtruth_top1024_nb" + std::to_string(size) + ".ivecs";
    //    std::string current_gt_file = (gt_path.parent_path() / gt_filename).string();
//
    //    test_graph(query_file, current_gt_file, g_file, 1, 1, 100);
    //}
    
    // // ------------------------------- SIFT100K -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "SIFT100K" / "sift_base.fvecs").string();
    // const auto query_file           = (data_path / "SIFT100K" / "sift_query.fvecs").string();
    // const auto gt_file              = (data_path / "SIFT100K" / (data_stream_type == AddAll ? "sift_groundtruth.ivecs" : "sift_groundtruth_base500000.ivecs" )).string();
    // const auto graph_file           = (data_path / "deg" / "128D_L2_K16_AddK32Eps0.1_schemeLow.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::LowLID : deglib::builder::OptimizationTarget::StreamingData;
    // const deglib::Metric metric     = deglib::Metric::L2;

    // if(std::filesystem::exists(graph_file.c_str()) == false) 
    //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 16, 32, 0.1f, 0, 0, 0, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    // test_graph(query_file, gt_file, graph_file, 1, 1, 100);  // repeat_test, test_threads, k

    
    // ------------------------------- ccnews-small -----------------------------------------
    /*
    const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "ccnews-small" / "ccnews-small_base.fvecs").string();
    // const auto query_file           = (data_path / "ccnews-small" / "ccnews-small_base.fvecs").string();
    const auto repository_file      = (data_path / "ccnews-small" / "ccnews-small_base.u8vecs").string();
    const auto query_file           = (data_path / "ccnews-small" / "ccnews-small_base.u8vecs").string();
    const auto gt_file              = (data_path / "ccnews-small" / "ccnews-small_explore_groundtruth.ivecs").string();
    const auto graph_file           = (data_path / "deg" / "384D_uint8_L2_K16_AddK32Eps0.05_LowLID.deg").string();
    const auto opt_graph_file       = (data_path / "deg" / "384D_uint8_L2_K16_AddK32Eps0.05_LowLID_opt500k.deg").string();
    const auto lid                  = deglib::builder::OptimizationTarget::LowLID;
    const deglib::Metric metric     = deglib::Metric::L2_Uint8;

    if(std::filesystem::exists(graph_file.c_str()) == false) 
        create_graph(repository_file, data_stream_type, graph_file, metric, lid, 16, 32, 0.1f, 0, 0, 0, 8); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, threads
    optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 500000); // k_opt, eps_opt, i_opt, iteration
    test_graph(query_file, gt_file, opt_graph_file, 1, 8, 16);  // repeat_test, test_threads, k
    */

    // ------------------------------- GLOVE -----------------------------------------
    const auto data_stream_type     = DataStreamType::AddAll;
    const auto repository_file      = (data_path / "glove-100" / "glove-100_base.fvecs").string();    
    const auto query_file           = (data_path / "glove-100" / "glove-100_query.fvecs").string();
    const auto gt_file              = (data_path / "glove-100" / (data_stream_type == AddAll ? "glove-100_groundtruth_top1024_nb1183514.ivecs" : "glove-100_groundtruth_base591757.ivecs" )).string();
    const auto graph_file           = (data_path / "deg" / "crEG" / "schemes" / "100D_L2_K30_AddK60Eps0.1_schemeD.deg").string();
    const auto lid                  = deglib::builder::OptimizationTarget::LowLID;
    // const auto reduce_graph_file    = (data_path / "deg" / "schemes" / "100D_L2_K30_AddK60Eps0.1_schemeC.deg").string();
    // const auto graph_file           = (data_path / "deg" / "dynamic" / "100D_L2_K30_AddK60Eps0.1_add500k_schemeC.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "dynamic" / "100D_L2_K30_AddK60Eps0.1_add500k_schemeC_OptAfterwardsWith_SwapK30-0StepEps0.001LowPath5_it1000000.deg").string();
    //  const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::OptimizationTarget::HighLID : deglib::builder::OptimizationTarget::StreamingData;
    const deglib::Metric metric     = deglib::Metric::L2;
    
    if(std::filesystem::exists(graph_file.c_str()) == false) {
        create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 0, 0, 0, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, thread_count
        // create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, thread_count
        // optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 1000000); // k_opt, eps_opt, i_opt, iteration
    }
    // reduce_graph(reduce_graph_file, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    test_graph(query_file, gt_file, graph_file, 1, 1, 100);  // repeat_test, test_threads, k
    //for (uint32_t k = 128; k <= 1024; k *= 2) {
    //    fmt::print("Testing for k = {} \n", k);
    //    test_graph(query_file, gt_file, graph_file, 1, 1, k);
    //}

    // scaling test
    //const int step_size = 100000;
    //const auto graph_files = create_incremental_graphs(repository_file, graph_file, step_size, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 1); 
    //for (const auto& [g_file, size] : graph_files) {
    //    fmt::print("Testing incremental graph {} \n", g_file);
//
    //    // construct gt_file based on size
    //    std::filesystem::path gt_path(gt_file);
    //    std::string gt_filename = "glove-100_groundtruth_top1024_nb" + std::to_string(size) + ".ivecs";
    //    std::string current_gt_file = (gt_path.parent_path() / gt_filename).string();
//
    //    test_graph(query_file, current_gt_file, g_file, 1, 1, 100);
    //}

    // // ------------------------------- laion2B -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddAll;
    // // const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K.fvecs").string();           // 300K 768float
    // // const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K_512float.fvecs").string();  // 300K 768float
    // const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K_512byte.u8vecs").string();  // 300K 512uint8
    // // const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=10M_512byte.u8vecs").string();   // 10M 512uint8

    // // const auto query_file           = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k.fvecs").string();         // 768float
    // // const auto query_file           = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k_512float.fvecs").string();   // 512float
    // const auto query_file           = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k_512byte.u8vecs").string();   // 512uint8

    // const auto gt_file              = (data_path / "laion2B" / "gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.ivecs").string(); // 300K
    // // const auto gt_file              = (data_path / "laion2B" / "gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.ivecs").string();  // 10M
    // const auto graph_file           = (data_path / "deg" / "300k" / "768D_L2_K30_AddK60Eps0.1_schemeD_t4_512byte.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "300k" / "768D_L2_K30_AddK60Eps0.1_schemeD_t1_512byte_200kAll.deg").string(); 
    // const auto mrng_graph_file      = (data_path / "deg" / "300k" / "768D_L2_K30_AddK60Eps0.1_schemeD_t12_512byte_removedNonMRNG.deg").string(); 
    // const auto lid                  = deglib::builder::OptimizationTarget::LowLID; // low=schemeD, high=schemeC
    // const deglib::Metric metric     = deglib::Metric::L2_Uint8;

    // if(std::filesystem::exists(graph_file.c_str()) == false) 
    //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 4); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, thread_count
    // test_graph(query_file, gt_file, graph_file, 1, 4, 30); // repeat_test, test_threads, k

    // if(std::filesystem::exists(opt_graph_file.c_str()) == false) 
    //     optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 200000); // k_opt, eps_opt, i_opt, iteration
    // test_graph(query_file, gt_file, opt_graph_file, 1, 1, 30);  // repeat_test, test_threads, k

    // if(std::filesystem::exists(mrng_graph_file.c_str()) == false) {
    //     remove_non_mrng_edges(graph_file, mrng_graph_file);
    //     // remove_non_mrng_edges(opt_graph_file, mrng_graph_file);
    //     // change_features(graph_file, repository_file, metric, opt_graph_file);
    // }
    // test_graph(query_file, gt_file, mrng_graph_file, 1, 1, 30);  // repeat_test, test_threads, k

    
    // ------------------------------- 2D graph -----------------------------------------
    // fmt::print("Data path: {}\n", data_path.string());
    // const auto graph_file = (data_path / "2dgraph.deg").string();
    // const auto opt_graph_file = (data_path / "2dgraph_opt.deg").string();
    // optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 20); // k_opt, eps_opt, i_opt, iteration

    fmt::print("Data path: {}\n", data_path.string());
    const auto graph_file = (data_path / "fashion.deg").string();
    const auto opt_graph_file = (data_path / "fashion_opt.deg").string();
    optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 1000); // k_opt, eps_opt, i_opt, iteration


    fmt::print("Test OK\n");
    return 0;
}
