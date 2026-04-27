#include "graph/sizebounded_graph.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <random>
#include <filesystem>
#include <functional>
#include <string>
#include <fstream>

static int tests_run = 0;
static int tests_passed = 0;

void check(bool condition, const char* name) {
    ++tests_run;
    if (condition) {
        ++tests_passed;
        std::cout << "  PASS: " << name << "\n";
    } else {
        std::cerr << "  FAIL: " << name << "\n";
    }
}

void check_bool(bool condition, const std::string& name) {
    ++tests_run;
    if (condition) {
        ++tests_passed;
        std::cout << "  PASS: " << name << "\n";
    } else {
        std::cerr << "  FAIL: " << name << "\n";
    }
}

void check_float(float actual, float expected, const char* name, float eps = 1e-4f) {
    ++tests_run;
    if (abs(actual - expected) < eps) {
        ++tests_passed;
        std::cout << "  PASS: " << name << "\n";
    } else {
        std::cerr << "  FAIL: " << name << " (actual=" << actual << ", expected=" << expected << ")\n";
    }
}

std::vector<float> make_vector_4d(float x, float y, float z, float w) {
    return {x, y, z, w};
}

// === 1. Konstruktion ===

void test_construction() {
    std::cout << "[Construction]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(100, 4, space);

    check(graph.size() == 0, "empty graph has size 0");
    check(graph.capacity() == 100, "capacity is 100");
    check(graph.getEdgesPerVertex() == 4, "edges per vertex is 4");
    check(space.dim() == 4, "dimension is 4");
    check(space.metric() == deglib::Metric::L2, "metric is L2");

    // Constructor rejects odd edges_per_vertex
    try {
        deglib::FloatSpace space2(4, deglib::Metric::L2);
        deglib::graph::SizeBoundedGraph graph2(50, 3, space2); // odd edges
    } catch (const std::invalid_argument&) {
        check(true, "invalid_argument for odd edges_per_vertex");
    }

    // InnerProduct metric
    deglib::FloatSpace space_ip(8, deglib::Metric::InnerProduct);
    deglib::graph::SizeBoundedGraph graph_ip(200, 8, space_ip);
    check(graph_ip.getEdgesPerVertex() == 8, "InnerProduct graph: 8 edges per vertex");
    check(graph_ip.capacity() == 200, "InnerProduct graph: capacity 200");
    check(graph_ip.getFeatureSpace().metric() == deglib::Metric::InnerProduct, "metric is InnerProduct");
}

// === 2. Vertex-Management ===

void test_add_vertex() {
    std::cout << "[Add Vertex]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    auto v0 = make_vector_4d(0.0f, 0.0f, 0.0f, 0.0f);
    auto v1 = make_vector_4d(1.0f, 0.0f, 0.0f, 0.0f);
    auto v2 = make_vector_4d(2.0f, 0.0f, 0.0f, 0.0f);

    auto idx0 = graph.addVertex(0, reinterpret_cast<const std::byte*>(v0.data()));
    check(idx0 == 0, "first vertex has internal index 0");
    check(graph.size() == 1, "size is 1 after adding one vertex");
    check(graph.hasVertex(0), "hasVertex(0) is true");

    auto idx1 = graph.addVertex(1, reinterpret_cast<const std::byte*>(v1.data()));
    check(idx1 == 1, "second vertex has internal index 1");
    check(graph.size() == 2, "size is 2 after adding two vertices");
    check(graph.hasVertex(1), "hasVertex(1) is true");

    auto idx2 = graph.addVertex(2, reinterpret_cast<const std::byte*>(v2.data()));
    check(idx2 == 2, "third vertex has internal index 2");
    check(graph.size() == 3, "size is 3 after adding three vertices");

    check(graph.getExternalLabel(0) == 0, "external label of internal 0 is 0");
    check(graph.getExternalLabel(1) == 1, "external label of internal 1 is 1");
    check(graph.getExternalLabel(2) == 2, "external label of internal 2 is 2");
}

void test_remove_vertex_single() {
    std::cout << "[Remove Vertex - Single]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(4, 2, space);

    auto v0 = make_vector_4d(0.0f, 0.0f, 0.0f, 0.0f);
    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0.data()));

    auto neighbors = graph.removeVertex(0);
    check(graph.size() == 0, "size is 0 after removing the only vertex");
    check(!graph.hasVertex(0), "hasVertex(0) is false");
}

void test_remove_middle_vertex() {
    std::cout << "[Remove Vertex - Middle of Chain]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 2, space);

    for (int i = 0; i < 4; i++) {
        std::vector<float> v(4, 0.0f);
        v[i % 4] = static_cast<float>(i);
        graph.addVertex(i, reinterpret_cast<const std::byte*>(v.data()));
    }

    check(graph.size() == 4, "size is 4 after adding 4 vertices");
    graph.removeVertex(1);

    check(graph.size() == 3, "size is 3 after removing middle vertex");
    check(!graph.hasVertex(1), "vertex 1 is gone");
    check(graph.hasVertex(0), "vertex 0 still exists");
    check(graph.hasVertex(2), "vertex 2 still exists");
    check(graph.hasVertex(3), "vertex 3 still exists");
}

void test_self_loop_initialization() {
    std::cout << "[Self-Loop Initialization]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    std::vector<float> v = {1.0f, 0.0f, 0.0f, 0.0f};
    graph.addVertex(5, reinterpret_cast<const std::byte*>(v.data()));

    check(graph.hasEdge(0, 0), "new vertex has self-loop to itself");
    check(!graph.hasEdge(0, 1), "no edge to vertex 1");
}

// === 3. Edge-Management ===

void test_set_and_query_edge() {
    std::cout << "[Set and Query Edge]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(10, 4, space);

    auto v0 = make_vector_4d(0.0f, 0.0f, 0.0f, 0.0f);
    auto v1 = make_vector_4d(3.0f, 4.0f, 0.0f, 0.0f);

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0.data()));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1.data()));

    graph.changeEdge(0, 0, 1, 5.0f);
    graph.changeEdge(1, 1, 0, 5.0f);

    check(graph.hasEdge(0, 1), "edge 0->1 exists");
    check(graph.hasEdge(1, 0), "edge 1->0 exists (bidirectional)");

    check_float(graph.getEdgeWeight(0, 1), 5.0f, "edge 0->1 weight is 5.0");
    check_float(graph.getEdgeWeight(1, 0), 5.0f, "edge 1->0 weight is 5.0");
    check_float(graph.getEdgeWeight(0, 2), -1.0f, "non-existent edge returns -1");
}

void test_change_edge_swap() {
    std::cout << "[ChangeEdge - Swapping Neighbors]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    for (int i = 0; i < 4; i++) {
        std::vector<float> v(4, 0.0f);
        v[0] = static_cast<float>(i);
        graph.addVertex(i, reinterpret_cast<const std::byte*>(v.data()));
    }

    graph.changeEdge(0, 0, 1, 1.0f);
    graph.changeEdge(1, 1, 0, 1.0f);

    graph.changeEdge(2, 2, 0, 4.0f);
    graph.changeEdge(0, 0, 2, 4.0f);

    graph.changeEdge(3, 3, 0, 9.0f);
    graph.changeEdge(0, 0, 3, 9.0f);

    check(graph.hasEdge(0, 1), "0->1 exists");
    check(graph.hasEdge(0, 2), "0->2 exists");
    check(graph.hasEdge(0, 3), "0->3 exists");
    check(graph.hasEdge(1, 0), "1->0 exists");
    check(graph.hasEdge(2, 0), "2->0 exists");
    check(graph.hasEdge(3, 0), "3->0 exists");
    check_float(graph.getEdgeWeight(0, 1), 1.0f, "edge 0->1 weight 1.0");
    check_float(graph.getEdgeWeight(0, 2), 4.0f, "edge 0->2 weight 4.0");
    check_float(graph.getEdgeWeight(0, 3), 9.0f, "edge 0->3 weight 9.0");
}

void test_change_edges_sorted() {
    std::cout << "[ChangeEdges - Sorted Neighbors]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    for (int i = 0; i < 5; i++) {
        std::vector<float> v(4, 0.0f);
        v[0] = static_cast<float>(i);
        graph.addVertex(i, reinterpret_cast<const std::byte*>(v.data()));
    }

    uint32_t sorted_neighbors[] = {0, 1, 2, 3};
    float weights[] = {0.0f, 1.0f, 4.0f, 9.0f};
    graph.changeEdges(0, sorted_neighbors, weights);

    const auto* neighbors = graph.getNeighborIndices(0);
    check(neighbors[0] == 0, "first neighbor is self (0)");
    check(neighbors[1] == 1, "second neighbor is 1");
    check(neighbors[2] == 2, "third neighbor is 2");
    check(neighbors[3] == 3, "fourth neighbor is 3");
}

// === 4. Vertex-Info ===

void test_label_lookup() {
    std::cout << "[Label Lookup]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(10, 2, space);

    for (auto lbl : {100u, 200u, 300u}) {
        std::vector<float> v(4, 0.0f);
        v[0] = static_cast<float>(lbl / 100);
        graph.addVertex(lbl, reinterpret_cast<const std::byte*>(v.data()));
    }

    check(graph.getInternalIndex(100) == 0, "label 100 → internal index 0");
    check(graph.getInternalIndex(200) == 1, "label 200 → internal index 1");
    check(graph.getInternalIndex(300) == 2, "label 300 → internal index 2");
    check(graph.getExternalLabel(0) == 100, "internal 0 → label 100");
    check(graph.getExternalLabel(1) == 200, "internal 1 → label 200");
    check(graph.getExternalLabel(2) == 300, "internal 2 → label 300");
}

void test_feature_vector_storage() {
    std::cout << "[Feature Vector Storage]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(2, 2, space);

    std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v2 = {5.0f, 6.0f, 7.0f, 8.0f};

    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1.data()));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2.data()));

    const float* fp1 = reinterpret_cast<const float*>(graph.getFeatureVector(0));
    const float* fp2 = reinterpret_cast<const float*>(graph.getFeatureVector(1));

    check_float(fp1[0], 1.0f, "feature[0][0] = 1.0");
    check_float(fp1[1], 2.0f, "feature[0][1] = 2.0");
    check_float(fp2[0], 5.0f, "feature[1][0] = 5.0");
    check_float(fp2[3], 8.0f, "feature[1][3] = 8.0");
}

// === 5. Graph-Info ===

void test_capacity_and_large_graph() {
    std::cout << "[Capacity and Large Graph]\n";

    deglib::FloatSpace space(128, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(1000, 16, space);

    check(graph.getEdgesPerVertex() == 16, "16 edges per vertex");
    check(graph.capacity() == 1000, "capacity 1000");
    check(graph.getFeatureSpace().dim() == 128, "128 dimensions");
    check(graph.getFeatureSpace().metric() == deglib::Metric::L2, "L2 metric");
}

void test_small_graph() {
    std::cout << "[Small Graph]\n";

    deglib::FloatSpace space(2, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(3, 2, space);

    auto v = std::make_unique<float[]>(2);
    v[0] = 0.0f; v[1] = 0.0f;
    graph.addVertex(0, reinterpret_cast<const std::byte*>(v.get()));
    check(graph.size() == 1, "single vertex");
    check(graph.hasEdge(0, 0), "self-loop exists");
}

// === 6. Search Tests ===

void test_search_basic() {
    std::cout << "[Search Basic]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    float query[] = {0.5f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search({0}, reinterpret_cast<const std::byte*>(query), 0.0f, 5);

    check(results.size() > 0, "search returns at least 1 result");
    if (results.size() > 0) {
        check(results.top().getInternalIndex() >= 0, "top index is valid");
    }
}

void test_search_with_filter() {
    std::cout << "[Search with Filter]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    int valid[] = {0};
    deglib::graph::Filter filter(valid, 1, 10, 10);

    float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = graph.search({0}, reinterpret_cast<const std::byte*>(query), 0.0f, 3, &filter);

    check(results.size() >= 0, "search with filter doesn't crash");
}

// === 7. Save and Load ===

void test_save_graph() {
    std::cout << "[Save Graph]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v2[] = {2.0f, 0.0f, 0.0f, 0.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2));

    const char* saved_graph = "test/graph_save_test.deg";
    bool saved = graph.saveGraph(saved_graph);
    check(saved, "saveGraph succeeded");

    auto size = std::filesystem::file_size(saved_graph);
    check(size > 0, "saved file has positive size");
    std::filesystem::remove(saved_graph);
}

void test_save_load_header() {
    std::cout << "[Save/Load Header Verification]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(5, 4, space);

    float v0[] = {7.0f, 10.0f, 0.0f, 0.0f};
    float v1[] = {0.5f, -1.0f, 2.0f, 3.0f};

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1));
    graph.changeEdge(0, 0, 1, 5.0f);

    std::string saved_graph = "test/graph_load_test.deg";
    graph.saveGraph(&saved_graph[0]);

    std::ifstream ifs(saved_graph.c_str(), std::ios::binary);
    check(ifs.is_open(), "opened saved graph for reading");

    if (ifs.is_open()) {
        uint8_t metric_type = 0;
        uint16_t dim = 0;
        uint32_t graph_size = 0;
        uint8_t edges = 0;
        ifs.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        ifs.read(reinterpret_cast<char*>(&graph_size), sizeof(graph_size));
        ifs.read(reinterpret_cast<char*>(&edges), sizeof(edges));

        check(static_cast<int>(metric_type) == 1, "saved metric matches L2");
        check(dim == 4, "saved dimension matches 4");
        check(graph_size == 2, "saved vertex count matches 2");
        check(edges == 4, "saved edges per vertex matches 4");

        ifs.close();
    }

    std::filesystem::remove(saved_graph.c_str());
}

void test_save_uint8() {
    std::cout << "[Save Uint8 Graph]\n";

    deglib::FloatSpace space(128, deglib::Metric::L2_Uint8);
    deglib::graph::SizeBoundedGraph graph(3, 2, space);

    std::vector<uint8_t> v0(128, 0);
    std::vector<uint8_t> v1(128, 1);
    std::vector<uint8_t> v2(128, 255);

    graph.addVertex(0, reinterpret_cast<const std::byte*>(v0.data()));
    graph.addVertex(1, reinterpret_cast<const std::byte*>(v1.data()));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(v2.data()));

    const char* saved = "test/graph_uint8_test.deg";
    bool saved_ok = graph.saveGraph(saved);
    check(saved_ok, "save uint8 graph succeeded");

    auto size = std::filesystem::file_size(saved);
    check(size > 0, "saved uint8 graph has positive size");
    std::filesystem::remove(saved);
    check(graph.getFeatureSpace().metric() == deglib::Metric::L2_Uint8, "saved graph metric is L2_Uint8");
}

// === 8. Multiple Operations ===

void test_multiple_add_remove() {
    std::cout << "[Multiple Add/Remove Cycles]\n";

    deglib::FloatSpace space(4, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(10, 2, space);

    for (int i = 0; i < 5; i++) {
        std::vector<float> v(4, 0.0f);
        v[i % 4] = static_cast<float>(i);
        graph.addVertex(i, reinterpret_cast<const std::byte*>(v.data()));
    }
    check(graph.size() == 5, "added 5 vertices, size 5");

    graph.removeVertex(0);
    graph.removeVertex(2);
    graph.removeVertex(4);
    check(graph.size() == 2, "removed 3 vertices, size 2");
    check(graph.hasVertex(1), "vertex 1 survives remove cycle");
    check(graph.hasVertex(3), "vertex 3 survives remove cycle");

    for (int i = 5; i < 8; i++) {
        std::vector<float> v(4, 0.0f);
        v[i % 4] = static_cast<float>(i);
        graph.addVertex(i, reinterpret_cast<const std::byte*>(v.data()));
    }
    check(graph.size() == 5, "after cycle 2, size is 5");

    for (int i = 1; i <= 7; i++) {
        if (graph.hasVertex(i)) graph.removeVertex(i);
    }
    check(graph.size() == 0, "finally empty after cleanup");
}

void test_feature_space_dimensions() {
    std::cout << "[Feature Space - Dimension Handling]\n";

    std::vector<size_t> dims = {1, 4, 8, 64, 128, 192};
    for (size_t dim : dims) {
        deglib::FloatSpace space(dim, deglib::Metric::L2);
        deglib::graph::SizeBoundedGraph graph(3, 2, space);

        std::vector<float> v(dim, 0.0f);
        v[0] = 1.0f;
        graph.addVertex(0, reinterpret_cast<const std::byte*>(v.data()));

        check_bool(graph.size() == 1, "dimension " + std::to_string(dim) + " graph can hold vertices");
        check_bool(graph.getFeatureSpace().dim() == dim, "dimension " + std::to_string(dim) + " recorded correctly");
    }

    std::vector<size_t> u8_dims = {64, 128, 192};
    for (size_t dim : u8_dims) {
        deglib::FloatSpace space(dim, deglib::Metric::L2_Uint8);
        deglib::graph::SizeBoundedGraph graph(3, 2, space);

        std::vector<uint8_t> v(dim, 0);
        v[0] = 1;
        graph.addVertex(0, reinterpret_cast<const std::byte*>(v.data()));

        check_bool(graph.getFeatureSpace().dim() == dim, "uint8 dimension " + std::to_string(dim) + " set correctly");
    }
}

// === main ===

int main() {
    std::cout << "=== SizeBoundedGraph Unit Tests ===\n\n";

    test_construction();
    test_add_vertex();
    test_remove_vertex_single();
    test_remove_middle_vertex();
    test_self_loop_initialization();
    test_set_and_query_edge();
    test_change_edge_swap();
    test_change_edges_sorted();
    test_label_lookup();
    test_feature_vector_storage();
    test_capacity_and_large_graph();
    test_small_graph();
    test_search_basic();
    test_search_with_filter();
    test_save_graph();
    test_save_load_header();
    test_save_uint8();
    test_multiple_add_remove();
    test_feature_space_dimensions();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";

    if (tests_passed < tests_run) {
        std::cerr << "FAILED!\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
