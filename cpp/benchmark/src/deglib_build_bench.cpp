#include <random>
#include <chrono>

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
void optimze_graph(const std::string initial_graph_file, const std::string graph_file, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t iterations) {
    
    auto rnd = std::mt19937(7);                         // default 7

    fmt::print("Load graph {} \n", initial_graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file.c_str());
    fmt::print("Graph with {} vertices and an avg edge weight of {} \n", graph.size(), deglib::analysis::calc_avg_edge_weight(graph, 100));

    deglib::builder::optimze_edges(graph, k_opt, eps_opt, i_opt, iterations);

    // store the graph
    graph.saveGraph(graph_file.c_str());

    fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}

/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void create_graph(const std::string repository_file, const DataStreamType data_stream_type, const std::string graph_file, deglib::Metric metric, deglib::builder::LID lid, const uint8_t d, const uint8_t k_ext, const float eps_ext, const uint8_t k_opt, const float eps_opt, const uint8_t i_opt, const uint32_t thread_count) {
    
    auto rnd = std::mt19937(7);                         // default 7
    const uint32_t swap_tries = 0;                      // additional swap tries between the next graph extension
    const uint32_t additional_swap_tries = 0;           // increse swap try count for each successful swap
    const uint32_t scale = 10;

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
    builder.setBatchSize(10000);
    builder.setThreadCount(thread_count);
    
    // provide all features to the graph builder at once. In an online system this will be called multiple times
    auto base_size = uint32_t(repository.size());
    auto addEntry = [&builder, &repository, feature_byte_size] (auto idx)
    {
        auto feature = repository.getFeature(idx);
        auto feature_vector = std::vector<std::byte>{feature, feature + feature_byte_size};
        builder.addEntry(idx+1, std::move(feature_vector)); // TODO label offset +1 for the laion dataset
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
    const auto log_after = 100;

    fmt::print("Start building \n");    
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        const auto size = graph.size();

        // if(status.step % log_after == 0 || size == base_size) {    
        //     duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
        //     auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
        //     auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true, scale);
        //     auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false, scale);
        //     auto valid_weights = deglib::analysis::check_graph_weights(graph) && deglib::analysis::check_graph_regularity(graph, uint32_t(size), true);
        //     auto connected = deglib::analysis::check_graph_connectivity(graph);
        //     auto duration = duration_ms / 1000;
        //     auto currRSS = getCurrentRSS() / 1000000;
        //     auto peakRSS = getPeakRSS() / 1000000;
        //     fmt::print("{:7} vertices, {:5}s, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
        //                 size, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
        //     start = std::chrono::steady_clock::now();
        // }
        // else 
        if(status.step % (log_after/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, scale);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:5}s, {:8} / {:8} improv, AEW: {:4.2f}, {} connected, RSS {} & peakRSS {}\n", size, duration, status.improved, status.tries, avg_edge_weight, connected ? "" : "not", currRSS, peakRSS);
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
  
    // // ------------------------------- audio -----------------------------------------
    // const auto repository_file      = (data_path / "audio" / "audio_base.fvecs").string();
    // const auto query_file           = (data_path / "audio" / "audio_query.fvecs").string();
    // const auto gt_file              = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "192D_L2_K20_AddK40Eps0.1_schemeD.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "192D_L2_K20_AddK40Eps0.1_schemeD_OptK20Eps0.001Path5_it20000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::LID::Low : deglib::builder::LID::Unknown;
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
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::LID::Low : deglib::builder::LID::Unknown;
    // const deglib::Metric metric     = deglib::Metric::L2;
    
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //     optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 20000); // k_opt, eps_opt, i_opt, iteration
    // }
    // test_graph(query_file, gt_file, opt_graph_file, 20, 1, 20); //  // repeat_test, test_threads, k


    // // ------------------------------- Deep1M -----------------------------------------
    // const auto repository_file      = (data_path / "deep1m" / "deep1m_base.fvecs").string();
    // const auto query_file           = (data_path / "deep1m" / "deep1m_query.fvecs").string();
    // const auto gt_file              = (data_path / "deep1m" / "deep1m_groundtruth.ivecs").string();
    // const auto graph_file           = (data_path / "deg" / "96D_L2_K30_AddK60Eps0.1_schemeD.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "96D_L2_K30_AddK60Eps0.1_schemeD_OptK30Eps0.001Path5_it400000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::LID::Low : deglib::builder::LID::Unknown;
    // const deglib::Metric metric     = deglib::Metric::L2;
    
    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, DataStreamType::AddAll, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //     optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 400000); // k_opt, eps_opt, i_opt, iteration
    // }
    // test_graph(query_file, gt_file, opt_graph_file, 20, 1, 20);  // repeat_test, test_threads, k



    // // ------------------------------- SIFT1M -----------------------------------------
    // const auto data_stream_type     = DataStreamType::AddHalfRemoveAndAddOneAtATime;
    // const auto repository_file      = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    // const auto query_file           = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    // const auto gt_file              = (data_path / "SIFT1M" / (data_stream_type == AddAll ? "sift_groundtruth.ivecs" : "sift_groundtruth_base500000.ivecs" )).string();
    // const auto graph_file           = (data_path / "deg" / "test" / "K30_AddK60Eps0.1_SwapK30Eps0.001_add500k_add2_remove2_until500k_2.deg").string();
    // const auto opt_graph_file       = (data_path / "deg" / "test" / "128D_L2_K30_AddK60Eps0.2_schemeD_OptK30Eps0.001Path5_it200000.deg").string();
    // const auto lid                  = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::LID::Low : deglib::builder::LID::Unknown;
    // const deglib::Metric metric     = deglib::Metric::L2;

    // if(std::filesystem::exists(graph_file.c_str()) == false) {
    //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //     // optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 200000); // k_opt, eps_opt, i_opt, iteration
    // }
    // test_graph(query_file, gt_file, graph_file, 1, 1, 100);  // repeat_test, test_threads, k


    // ------------------------------- SIFT1M -----------------------------------------
    const auto data_stream_type     = DataStreamType::AddAll;
    // const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K.fvecs").string();           // 300K 768float
    // const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K_512float.fvecs").string();  // 300K 768float
    const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K_512byte.u8vecs").string();  // 300K 512uint8
    // const auto repository_file      = (data_path / "laion2B" / "laion2B-en-clip768v2-n=10M_512byte.u8vecs").string();   // 10M 512uint8

    // const auto query_file           = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k.fvecs").string();         // 768float
    // const auto query_file           = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k_512float.fvecs").string();   // 512float
    const auto query_file           = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k_512byte.u8vecs").string();   // 512uint8

    const auto gt_file              = (data_path / "laion2B" / "gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.ivecs").string(); // 300K
    // const auto gt_file              = (data_path / "laion2B" / "gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.ivecs").string();  // 10M
    const auto graph_file           = (data_path / "deg" / "300k" / "768D_L2_K30_AddK60Eps0.1_schemeD_t4_512byte.deg").string();
    const auto opt_graph_file       = (data_path / "deg" / "300k" / "768D_L2_K30_AddK60Eps0.1_schemeD_t1_512byte_200kAll.deg").string(); 
    const auto mrng_graph_file      = (data_path / "deg" / "300k" / "768D_L2_K30_AddK60Eps0.1_schemeD_t12_512byte_removedNonMRNG.deg").string(); 
    const auto lid                  = deglib::builder::LID::Low; // low=schemeD, high=schemeC
    const deglib::Metric metric     = deglib::Metric::L2_Uint8;

    // if(std::filesystem::exists(graph_file.c_str()) == false) 
    //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5, 4); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt, thread_count
    test_graph(query_file, gt_file, graph_file, 1, 4, 30); // repeat_test, test_threads, k

    // if(std::filesystem::exists(opt_graph_file.c_str()) == false) 
    //     optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 200000); // k_opt, eps_opt, i_opt, iteration
    // test_graph(query_file, gt_file, opt_graph_file, 1, 1, 30);  // repeat_test, test_threads, k

    // if(std::filesystem::exists(mrng_graph_file.c_str()) == false) {
    //     remove_non_mrng_edges(graph_file, mrng_graph_file);
    //     // remove_non_mrng_edges(opt_graph_file, mrng_graph_file);
    //     // change_features(graph_file, repository_file, metric, opt_graph_file);
    // }
    // test_graph(query_file, gt_file, mrng_graph_file, 1, 1, 30);  // repeat_test, test_threads, k


    // // ------------------------------- GLOVE -----------------------------------------
    // const auto data_stream_type = DataStreamType::AddAll;
    // const auto repository_file  = (data_path / "glove-100" / "glove-100_base.fvecs").string();    
    // const auto query_file       = (data_path / "glove-100" / "glove-100_query.fvecs").string();
    // const auto gt_file          = (data_path / "glove-100" / (data_stream_type == AddAll ? "glove-100_groundtruth.ivecs" : "glove-100_groundtruth_base591757.ivecs" )).string();
    // const auto graph_file       = (data_path / "deg" / "test" / "100D_L2_K30_AddK60Eps0.1_schemeC.deg").string();
    // const auto opt_graph_file   = (data_path / "deg" / "test" / "100D_L2_K30_AddK60Eps0.1_schemeC_OptK30Eps0.001Path5_it1000000.deg").string();
    // const auto lid              = (data_stream_type == AddAll || data_stream_type == AddHalf) ? deglib::builder::LID::High : deglib::builder::LID::Unknown;
    // const deglib::Metric metric = deglib::Metric::L2;
    
    // // if(std::filesystem::exists(graph_file.c_str()) == false) {
    // //     create_graph(repository_file, data_stream_type, graph_file, metric, lid, 30, 60, 0.1f, 30, 0.001f, 5); // d, k_ext, eps_ext, k_opt, eps_opt, i_opt
    //     optimze_graph(graph_file, opt_graph_file, 30, 0.001f, 5, 1000000); // k_opt, eps_opt, i_opt, iteration
    // // }
    // test_graph(query_file, gt_file, opt_graph_file, 1, 1, 100);  // repeat_test, test_threads, k

 

    fmt::print("Test OK\n");
    return 0;
}
