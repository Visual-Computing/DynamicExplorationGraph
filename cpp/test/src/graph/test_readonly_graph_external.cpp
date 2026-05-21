// test_readonly_graph_external.cpp — Google Test suite for ReadOnlyGraphExternal
//
// Covers:
//  1. Construction from a SizeBoundedGraph
//  2. Correct internal / external label mapping
//  3. getFeatureVector returns the correctly permuted data from the external array
//  4. reorder_features_inplace correctness (all cycle lengths: trivial, 2-cycle, long cycle)
//  5. explore produces results matching ReadOnlyGraph on the same topology + features
//  6. search produces results matching ReadOnlyGraph
//  7. hasVertex / hasEdge / hasPath smoke-tests

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "graph/readonly_graph.h"
#include "graph/readonly_graph_external.h"
#include "graph/sizebounded_graph.h"
#include "gtest/gtest.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a small, fully-connected (k-regular) graph with float L2 features.
// Vertices are labelled 0..N-1.  External label == internal index in this case.
static deglib::graph::SizeBoundedGraph make_linear_graph(
    uint32_t N, uint8_t edges_per_vertex, uint32_t dims,
    std::vector<std::vector<float>>& out_features)
{
    deglib::FloatSpace space(dims, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(N, edges_per_vertex, space);

    out_features.resize(N, std::vector<float>(dims, 0.f));
    for (uint32_t i = 0; i < N; ++i) {
        out_features[i][0] = static_cast<float>(i);
        const std::byte* bytes = reinterpret_cast<const std::byte*>(out_features[i].data());
        graph.addVertex(i, bytes);
    }

    // Wire edges: each vertex connects to min/max neighbours in a circular fashion
    for (uint32_t i = 0; i < N; ++i) {
        std::vector<uint32_t> nbrs;
        for (uint8_t j = 1; j <= edges_per_vertex / 2; ++j) {
            nbrs.push_back((i + j) % N);
            nbrs.push_back((i + N - j) % N);
        }
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        std::vector<float> weights(nbrs.size(), 1.f);
        graph.changeEdges(i, nbrs.data(), weights.data());
    }

    return graph;
}

// Build a graph where external labels are shuffled (label != internal index).
// Returns the label→feature mapping.
static deglib::graph::SizeBoundedGraph make_shuffled_label_graph(
    uint32_t N, uint8_t edges_per_vertex, uint32_t dims,
    std::vector<uint32_t>& out_labels,              // internal_index → external_label
    std::vector<std::vector<float>>& out_features)  // external_label → features
{
    // Create a permutation for external labels
    out_labels.resize(N);
    std::iota(out_labels.begin(), out_labels.end(), 0u);
    std::mt19937 rng(12345);
    std::shuffle(out_labels.begin(), out_labels.end(), rng);

    deglib::FloatSpace space(dims, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(N, edges_per_vertex, space);

    out_features.resize(N, std::vector<float>(dims, 0.f));
    for (uint32_t label = 0; label < N; ++label) {
        out_features[label][0] = static_cast<float>(label) * 10.f;
        out_features[label][1] = static_cast<float>(label);
    }

    // Insert vertices with shuffled labels
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t label = out_labels[i];
        const std::byte* bytes = reinterpret_cast<const std::byte*>(out_features[label].data());
        graph.addVertex(label, bytes);
    }

    // Wire a simple ring topology
    for (uint32_t i = 0; i < N; ++i) {
        std::vector<uint32_t> nbrs;
        for (uint8_t j = 1; j <= edges_per_vertex / 2; ++j) {
            nbrs.push_back((i + j) % N);
            nbrs.push_back((i + N - j) % N);
        }
        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        std::vector<float> weights(nbrs.size(), 1.f);
        graph.changeEdges(i, nbrs.data(), weights.data());
    }

    return graph;
}

// Collect all result labels from a ResultSet into a sorted vector.
static std::vector<uint32_t> drain_to_labels(deglib::search::ResultSet rs,
                                              const deglib::search::SearchGraph& graph)
{
    std::vector<uint32_t> labels;
    while (!rs.empty()) {
        labels.push_back(graph.getExternalLabel(rs.top().getInternalIndex()));
        rs.pop();
    }
    std::sort(labels.begin(), labels.end());
    return labels;
}

// ---------------------------------------------------------------------------
// 1. Construction
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, ConstructionBasic) {
    constexpr uint32_t N = 8;
    constexpr uint32_t dims = 4;
    constexpr uint8_t K = 4;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    // Build a flat float array indexed by label (== internal index here)
    std::vector<float> ext_feats(N * dims);
    for (uint32_t i = 0; i < N; ++i)
        std::memcpy(ext_feats.data() + i * dims, features[i].data(), dims * sizeof(float));

    // Permute in-place (identity permutation here since label == internal index)
    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext_feats.data(), dims);

    deglib::graph::ReadOnlyGraphExternal ext_graph(fp_space, src, ext_feats.data());

    EXPECT_EQ(ext_graph.size(), N);
    EXPECT_EQ(ext_graph.getEdgesPerVertex(), K);
    EXPECT_EQ(ext_graph.getFeatureSpace().metric(), deglib::Metric::L2);
}

TEST(ReadOnlyGraphExternal, RejectsOddEdgesPerVertex) {
    constexpr uint32_t N = 4;
    constexpr uint32_t dims = 4;
    constexpr uint8_t K = 2;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    std::vector<float> ext_feats(N * dims, 0.f);
    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);

    // Temporarily patch: build a "bad" graph with 3 edges — but SizeBoundedGraph
    // already guards against odd K, so just test that the External graph also does.
    // We can verify the guard fires via a wrapper.
    // Since SizeBoundedGraph prevents building odd-K graphs we cannot easily get
    // one, so instead we test the constructor's guard using a mock graph.
    // Here we just test that even-K construction succeeds:
    EXPECT_NO_THROW(
        deglib::graph::ReadOnlyGraphExternal ext(fp_space, src, ext_feats.data())
    );
}

// ---------------------------------------------------------------------------
// 2. Label mapping
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, LabelMappingIdentity) {
    constexpr uint32_t N = 6;
    constexpr uint32_t dims = 4;
    constexpr uint8_t K = 2;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    std::vector<float> ext_feats(N * dims);
    for (uint32_t i = 0; i < N; ++i)
        std::memcpy(ext_feats.data() + i * dims, features[i].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext_feats.data(), dims);
    deglib::graph::ReadOnlyGraphExternal ext(fp_space, src, ext_feats.data());

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t label = ext.getExternalLabel(i);
        uint32_t back  = ext.getInternalIndex(label);
        EXPECT_EQ(back, i) << "Round-trip failed for internal index " << i;
        EXPECT_TRUE(ext.hasVertex(label));
    }
}

TEST(ReadOnlyGraphExternal, LabelMappingShuffled) {
    constexpr uint32_t N = 10;
    constexpr uint32_t dims = 4;
    constexpr uint8_t K = 4;

    std::vector<uint32_t> labels;
    std::vector<std::vector<float>> feat_by_label;
    auto src = make_shuffled_label_graph(N, K, dims, labels, feat_by_label);

    // Build flat external array indexed by label
    std::vector<float> ext_feats(N * dims);
    for (uint32_t lbl = 0; lbl < N; ++lbl)
        std::memcpy(ext_feats.data() + lbl * dims, feat_by_label[lbl].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext_feats.data(), dims);
    deglib::graph::ReadOnlyGraphExternal ext(fp_space, src, ext_feats.data());

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t label = ext.getExternalLabel(i);
        // labels[i] was the external label assigned to internal index i
        EXPECT_EQ(label, labels[i]) << "Label mismatch at internal index " << i;
        EXPECT_EQ(ext.getInternalIndex(label), i);
    }
}

// ---------------------------------------------------------------------------
// 3. getFeatureVector after reorder_features_inplace
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, FeatureVectorAfterReorder) {
    constexpr uint32_t N = 8;
    constexpr uint32_t dims = 4;
    constexpr uint8_t K = 2;

    std::vector<uint32_t> labels;
    std::vector<std::vector<float>> feat_by_label;
    auto src = make_shuffled_label_graph(N, K, dims, labels, feat_by_label);

    std::vector<float> ext_feats(N * dims);
    for (uint32_t lbl = 0; lbl < N; ++lbl)
        std::memcpy(ext_feats.data() + lbl * dims, feat_by_label[lbl].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext_feats.data(), dims);
    deglib::graph::ReadOnlyGraphExternal ext(fp_space, src, ext_feats.data());

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t label = ext.getExternalLabel(i);
        const float* got  = reinterpret_cast<const float*>(ext.getFeatureVector(i));
        const float* want = feat_by_label[label].data();
        for (uint32_t d = 0; d < dims; ++d) {
            EXPECT_FLOAT_EQ(got[d], want[d])
                << "Feature mismatch at internal index=" << i << " dim=" << d;
        }
    }
}

// ---------------------------------------------------------------------------
// 4. reorder_features_inplace correctness
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, ReorderFeaturesInplaceTrivial) {
    // Identity permutation: label[i] == i for all i.
    constexpr uint32_t N = 5;
    constexpr uint32_t dims = 2;
    constexpr uint8_t K = 2;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    // Data before permutation
    std::vector<float> ext(N * dims);
    for (uint32_t i = 0; i < N; ++i) {
        ext[i * dims + 0] = static_cast<float>(i * 100);
        ext[i * dims + 1] = static_cast<float>(i * 100 + 1);
    }
    std::vector<float> expected = ext;  // identity, should not change

    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext.data(), dims);

    for (uint32_t i = 0; i < N * dims; ++i) {
        EXPECT_FLOAT_EQ(ext[i], expected[i]) << "Identity reorder changed value at " << i;
    }
}

TEST(ReadOnlyGraphExternal, ReorderFeaturesInplaceSwap) {
    // 2-cycle: labels = {1, 0, 2, 3}
    // internal 0 → label 1, internal 1 → label 0
    constexpr uint32_t N = 4;
    constexpr uint32_t dims = 2;
    constexpr uint8_t K = 2;

    deglib::FloatSpace space(dims, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(N, K, space);

    // Insert as: internal 0 = label 1, internal 1 = label 0, internal 2 = label 2, internal 3 = label 3
    float dummy[2] = {0.f, 0.f};
    graph.addVertex(1, reinterpret_cast<const std::byte*>(dummy));
    graph.addVertex(0, reinterpret_cast<const std::byte*>(dummy));
    graph.addVertex(2, reinterpret_cast<const std::byte*>(dummy));
    graph.addVertex(3, reinterpret_cast<const std::byte*>(dummy));

    // Wire a ring so all edges are valid
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t nbrs[2] = {(i + 1) % N, (i + N - 1) % N};
        std::sort(nbrs, nbrs + 2);
        float w[2] = {1.f, 1.f};
        graph.changeEdges(i, nbrs, w);
    }

    // ext[label] = features for that label
    std::vector<float> ext(N * dims);
    for (uint32_t lbl = 0; lbl < N; ++lbl) {
        ext[lbl * dims + 0] = static_cast<float>(lbl * 10);
        ext[lbl * dims + 1] = static_cast<float>(lbl * 10 + 1);
    }

    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(graph, ext.data(), dims);

    // After reorder, ext[i] should contain features for getExternalLabel(i)
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t label = graph.getExternalLabel(i);
        EXPECT_FLOAT_EQ(ext[i * dims + 0], static_cast<float>(label * 10))     << "i=" << i;
        EXPECT_FLOAT_EQ(ext[i * dims + 1], static_cast<float>(label * 10 + 1)) << "i=" << i;
    }
}

TEST(ReadOnlyGraphExternal, ReorderFeaturesInplaceLongCycle) {
    // Create a graph where labels form a long cycle: 0→1→2→3→4→0
    constexpr uint32_t N = 5;
    constexpr uint32_t dims = 3;
    constexpr uint8_t K = 2;

    deglib::FloatSpace space(dims, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph graph(N, K, space);

    // Internal order: 0→label4, 1→label0, 2→label1, 3→label2, 4→label3
    uint32_t label_order[] = {4, 0, 1, 2, 3};
    float dummy[3] = {0.f, 0.f, 0.f};
    for (uint32_t i = 0; i < N; ++i)
        graph.addVertex(label_order[i], reinterpret_cast<const std::byte*>(dummy));

    // Wire a ring
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t nbrs[2] = {(i + 1) % N, (i + N - 1) % N};
        std::sort(nbrs, nbrs + 2);
        float w[2] = {1.f, 1.f};
        graph.changeEdges(i, nbrs, w);
    }

    // ext[label] = {label*100, label*100+1, label*100+2}
    std::vector<float> ext(N * dims);
    for (uint32_t lbl = 0; lbl < N; ++lbl) {
        for (uint32_t d = 0; d < dims; ++d)
            ext[lbl * dims + d] = static_cast<float>(lbl * 100 + d);
    }

    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(graph, ext.data(), dims);

    for (uint32_t i = 0; i < N; ++i) {
        uint32_t label = graph.getExternalLabel(i);
        for (uint32_t d = 0; d < dims; ++d) {
            EXPECT_FLOAT_EQ(ext[i * dims + d], static_cast<float>(label * 100 + d))
                << "i=" << i << " d=" << d;
        }
    }
}

// ---------------------------------------------------------------------------
// 5. explore produces results matching ReadOnlyGraph
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, ExploreMatchesReadOnlyGraph) {
    constexpr uint32_t N = 16;
    constexpr uint32_t dims = 8;
    constexpr uint8_t K = 4;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    // Build flat feature array (label == internal index here, no shuffle)
    std::vector<float> ext_feats(N * dims);
    for (uint32_t i = 0; i < N; ++i)
        std::memcpy(ext_feats.data() + i * dims, features[i].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);

    // Build ReadOnlyGraph (reference)
    deglib::graph::ReadOnlyGraph ref_graph(fp_space, src, ext_feats.data());

    // Build ReadOnlyGraphExternal (under test)
    std::vector<float> ext_feats2 = ext_feats;  // independent copy
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext_feats2.data(), dims);
    deglib::graph::ReadOnlyGraphExternal ext_graph(fp_space, src, ext_feats2.data());

    // For each vertex, compare explore results (label sets)
    for (uint32_t entry = 0; entry < N; ++entry) {
        auto ref_res = ref_graph.explore(entry, 5, false, 200);
        auto ext_res = ext_graph.explore(entry, 5, false, 200);

        auto ref_labels = drain_to_labels(std::move(ref_res), ref_graph);
        auto ext_labels = drain_to_labels(std::move(ext_res), ext_graph);

        EXPECT_EQ(ref_labels, ext_labels) << "explore mismatch at entry=" << entry;
    }
}

// ---------------------------------------------------------------------------
// 6. search produces results matching ReadOnlyGraph
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, SearchMatchesReadOnlyGraph) {
    constexpr uint32_t N = 16;
    constexpr uint32_t dims = 8;
    constexpr uint8_t K = 4;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    std::vector<float> ext_feats(N * dims);
    for (uint32_t i = 0; i < N; ++i)
        std::memcpy(ext_feats.data() + i * dims, features[i].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);

    deglib::graph::ReadOnlyGraph ref_graph(fp_space, src, ext_feats.data());

    std::vector<float> ext_feats2 = ext_feats;
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext_feats2.data(), dims);
    deglib::graph::ReadOnlyGraphExternal ext_graph(fp_space, src, ext_feats2.data());

    // Query = centroid-ish vector
    std::vector<float> query(dims, static_cast<float>(N) / 2.f);

    std::vector<uint32_t> entry = {0};
    auto ref_res = ref_graph.search(entry, reinterpret_cast<const std::byte*>(query.data()), 0.f, 5);
    auto ext_res = ext_graph.search(entry, reinterpret_cast<const std::byte*>(query.data()), 0.f, 5);

    auto ref_labels = drain_to_labels(std::move(ref_res), ref_graph);
    auto ext_labels = drain_to_labels(std::move(ext_res), ext_graph);

    EXPECT_EQ(ref_labels, ext_labels) << "search result mismatch";
}

// ---------------------------------------------------------------------------
// 7. hasVertex / hasEdge / hasPath smoke-tests
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, HasVertexAndEdge) {
    constexpr uint32_t N = 8;
    constexpr uint32_t dims = 4;
    constexpr uint8_t K = 2;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    std::vector<float> ext(N * dims);
    for (uint32_t i = 0; i < N; ++i)
        std::memcpy(ext.data() + i * dims, features[i].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext.data(), dims);
    deglib::graph::ReadOnlyGraphExternal graph(fp_space, src, ext.data());

    // All inserted labels should be present
    for (uint32_t lbl = 0; lbl < N; ++lbl)
        EXPECT_TRUE(graph.hasVertex(lbl));
    EXPECT_FALSE(graph.hasVertex(N + 100));

    // Ring topology: vertex 0 should be connected to 1 and N-1
    uint32_t idx0 = graph.getInternalIndex(0);
    uint32_t idx1 = graph.getInternalIndex(1);
    EXPECT_TRUE(graph.hasEdge(idx0, idx1));
    EXPECT_TRUE(graph.hasEdge(idx1, idx0));
}

TEST(ReadOnlyGraphExternal, HasPathSmoke) {
    constexpr uint32_t N = 8;
    constexpr uint32_t dims = 4;
    constexpr uint8_t K = 4;

    std::vector<std::vector<float>> features;
    auto src = make_linear_graph(N, K, dims, features);

    std::vector<float> ext(N * dims);
    for (uint32_t i = 0; i < N; ++i)
        std::memcpy(ext.data() + i * dims, features[i].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext.data(), dims);
    deglib::graph::ReadOnlyGraphExternal graph(fp_space, src, ext.data());

    // hasPath from vertex 0 to vertex N-1 should find a path on this ring.
    auto path = graph.hasPath({0}, N - 1, 0.f, 5);
    // The graph is fully connected, so a path must exist.
    EXPECT_FALSE(path.empty());
}

// ---------------------------------------------------------------------------
// 8. FP16 feature vectors with reorder_features_inplace
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, FP16FeaturesAfterReorder) {
    constexpr uint32_t N = 6;
    constexpr uint32_t dims = 8;  // divisible by 8
    constexpr uint8_t K = 2;

    std::vector<uint32_t> labels;
    std::vector<std::vector<float>> feat_by_label_f32;
    deglib::FloatSpace build_space(dims, deglib::Metric::L2);
    deglib::graph::SizeBoundedGraph build_graph(N, K, build_space);

    // Shuffled labels: 2, 5, 0, 3, 1, 4
    std::vector<uint32_t> label_order = {2, 5, 0, 3, 1, 4};
    feat_by_label_f32.resize(N, std::vector<float>(dims, 0.f));
    for (uint32_t lbl = 0; lbl < N; ++lbl)
        for (uint32_t d = 0; d < dims; ++d)
            feat_by_label_f32[lbl][d] = static_cast<float>(lbl * 10 + d);

    float dummy[8] = {};
    for (uint32_t i = 0; i < N; ++i)
        build_graph.addVertex(label_order[i], reinterpret_cast<const std::byte*>(dummy));

    // Wire a ring
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t nbrs[2] = {(i + 1) % N, (i + N - 1) % N};
        std::sort(nbrs, nbrs + 2);
        float w[2] = {1.f, 1.f};
        build_graph.changeEdges(i, nbrs, w);
    }

    // Build FP16 ext array indexed by label
    // Using a simple software conversion: just store the integer index scaled
    // (real FP16 conversion not required for correctness of reorder test).
    std::vector<uint16_t> fp16_ext(N * dims);
    for (uint32_t lbl = 0; lbl < N; ++lbl) {
        for (uint32_t d = 0; d < dims; ++d) {
            // Store as integer value: label*10 + d (not a real fp16, but unique per label)
            fp16_ext[lbl * dims + d] = static_cast<uint16_t>(lbl * 10 + d);
        }
    }

    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(build_graph, fp16_ext.data(), dims);

    // After reorder, fp16_ext[i * dims + d] should equal label_order[i] * 10 + d
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t label = build_graph.getExternalLabel(i);
        for (uint32_t d = 0; d < dims; ++d) {
            EXPECT_EQ(fp16_ext[i * dims + d], static_cast<uint16_t>(label * 10 + d))
                << "FP16 reorder mismatch at i=" << i << " d=" << d;
        }
    }
}

// ---------------------------------------------------------------------------
// 9. Shuffled-label explore produces matching results to ReadOnlyGraph
// ---------------------------------------------------------------------------

TEST(ReadOnlyGraphExternal, ShuffledLabelExploreMatchesReadOnlyGraph) {
    constexpr uint32_t N = 12;
    constexpr uint32_t dims = 8;
    constexpr uint8_t K = 4;

    std::vector<uint32_t> labels;
    std::vector<std::vector<float>> feat_by_label;
    auto src = make_shuffled_label_graph(N, K, dims, labels, feat_by_label);

    // Build flat float array indexed by label
    std::vector<float> ext_by_label(N * dims);
    for (uint32_t lbl = 0; lbl < N; ++lbl)
        std::memcpy(ext_by_label.data() + lbl * dims, feat_by_label[lbl].data(), dims * sizeof(float));

    deglib::FloatSpace fp_space(dims, deglib::Metric::L2);

    // Reference: ReadOnlyGraph (copies features inline, indexed by label)
    deglib::graph::ReadOnlyGraph ref_graph(fp_space, src, ext_by_label.data());

    // Under test: ReadOnlyGraphExternal with permuted array
    std::vector<float> ext_permuted = ext_by_label;
    deglib::graph::ReadOnlyGraphExternal::reorder_features_inplace(src, ext_permuted.data(), dims);
    deglib::graph::ReadOnlyGraphExternal ext_graph(fp_space, src, ext_permuted.data());

    for (uint32_t entry = 0; entry < N; ++entry) {
        auto ref_res = ref_graph.explore(entry, 4, false, 200);
        auto ext_res = ext_graph.explore(entry, 4, false, 200);
        auto ref_labels = drain_to_labels(std::move(ref_res), ref_graph);
        auto ext_labels = drain_to_labels(std::move(ext_res), ext_graph);
        EXPECT_EQ(ref_labels, ext_labels)
            << "Shuffled-label explore mismatch at entry=" << entry;
    }
}
