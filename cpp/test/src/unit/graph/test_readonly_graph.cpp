#include "gtest/gtest.h"

#include "deglib/graph.h"
#include "deglib/deglib.h"
#include "deglib/search/reranker.h"
#include "deglib/search/searcher.h"
#include "deglib/transformation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

deglib::graph::SizeBoundedGraph
make_size_bounded_graph(uint32_t n, uint32_t dim, uint8_t k, const std::vector<float>& features, const std::vector<uint32_t>& edges) {
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);
    deglib::graph::SizeBoundedGraph graph(n, k, space);
    std::vector<float> weights(k, 0.0f);

    for (uint32_t i = 0; i < n; ++i) {
        graph.addVertex(i, reinterpret_cast<const std::byte*>(features.data() + size_t(i) * dim));
        graph.changeEdges(i, edges.data() + size_t(i) * k, weights.data());
    }

    return graph;
}

}  // namespace

namespace {

TEST(ReadOnlyGraphTest, BuildsInt8GraphWithNativeL2Metric) {
    constexpr uint32_t n = 64;
    constexpr uint32_t dim = 32;
    constexpr uint8_t k = 4;

    std::vector<int8_t> features(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            features[static_cast<size_t>(i) * dim + j] = static_cast<int8_t>((i + j * 7) % 128);
        }
    }

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Int8_L2);
    auto graph = deglib::create_random_graph(std::as_bytes(std::span(features)).data(), n, k, space, 42);

    EXPECT_EQ(graph.size(), n);
    EXPECT_EQ(graph.getFeatureSpace().metric().value, deglib::distances::MetricType::Int8_L2);

    const auto res = graph.search_ef(std::span<const int8_t>(features.data(), dim), 1, 32);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res.front().getIdentifier(), 0u);
    EXPECT_FLOAT_EQ(res.front().getDistance(), 0.0f);
}

TEST(ReadOnlyGraphTest, BuildsInt8GraphWithNativeInnerProductMetric) {
    constexpr uint32_t n = 64;
    constexpr uint32_t dim = 32;
    constexpr uint8_t k = 4;

    std::vector<int8_t> features(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            features[static_cast<size_t>(i) * dim + j] = static_cast<int8_t>(i == 0 ? 100 : 1);
        }
    }

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::Int8_InnerProduct);
    auto graph = deglib::create_random_graph(std::as_bytes(std::span(features)).data(), n, k, space, 42);

    EXPECT_EQ(graph.size(), n);
    EXPECT_EQ(graph.getFeatureSpace().metric().value, deglib::distances::MetricType::Int8_InnerProduct);

    const auto res = graph.search_ef(std::span<const int8_t>(features.data(), dim), 1, 32);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res.front().getIdentifier(), 0u);
}

TEST(ReadOnlyGraphTest, BuildsFP32GraphWithNativeL2Metric) {
    constexpr uint32_t n = 64;
    constexpr uint32_t dim = 32;
    constexpr uint8_t k = 4;

    std::vector<float> features(static_cast<size_t>(n) * dim);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            features[static_cast<size_t>(i) * dim + j] = static_cast<float>((i + j) % 17);
        }
    }

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);
    auto graph = deglib::create_random_graph(std::as_bytes(std::span(features)).data(), n, k, space, 42);

    EXPECT_EQ(graph.size(), n);
    EXPECT_EQ(graph.getFeatureSpace().metric().value, deglib::distances::MetricType::FP32_L2);

    const auto res = graph.search_ef(std::span<const float>(features.data(), dim), 1, 32);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res.front().getIdentifier(), 0u);
    EXPECT_FLOAT_EQ(res.front().getDistance(), 0.0f);
}

TEST(ReadOnlyGraphTest, BuildsFP32GraphWithNativeInnerProductMetric) {
    constexpr uint32_t n = 64;
    constexpr uint32_t dim = 32;
    constexpr uint8_t k = 4;

    std::vector<float> features(static_cast<size_t>(n) * dim, 1.0f);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            features[static_cast<size_t>(i) * dim + j] = static_cast<float>(1 + ((i + j) % 7));
        }
    }

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_InnerProduct);
    auto graph = deglib::create_random_graph(std::as_bytes(std::span(features)).data(), n, k, space, 42);

    EXPECT_EQ(graph.size(), n);
    EXPECT_EQ(graph.getFeatureSpace().metric().value, deglib::distances::MetricType::FP32_InnerProduct);

    const auto res = graph.search_ef(std::span<const float>(features.data(), dim), 1, 32);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res.front().getIdentifier(), 0u);
}

TEST(ReadOnlyGraphTest, OptimizeTunesPrefetchParameters) {
    constexpr uint32_t n = 32;
    constexpr uint32_t dim = 4;
    constexpr uint8_t k = 4;

    std::vector<float> features(static_cast<size_t>(n) * dim, 0.0f);
    std::vector<uint32_t> edges(static_cast<size_t>(n) * k);
    for (uint32_t i = 0; i < n; ++i) {
        features[static_cast<size_t>(i) * dim + 0] = static_cast<float>(i);
        features[static_cast<size_t>(i) * dim + 1] = static_cast<float>(i % 7);
        features[static_cast<size_t>(i) * dim + 2] = static_cast<float>((i * 3) % 11);
        features[static_cast<size_t>(i) * dim + 3] = static_cast<float>((i * 5) % 13);

        for (uint32_t j = 0; j < k; ++j) {
            edges[static_cast<size_t>(i) * k + j] = (i + j) % n;
        }
    }

    auto source = make_size_bounded_graph(n, dim, k, features, edges);
    source.setEntryVertexIndices({0, 1});

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);
    auto graph = deglib::graph::ReadOnlyGraph::from_graph(source, space);

    graph.optimize(8);

    EXPECT_GE(graph.getPo(), 0);
    EXPECT_GE(graph.getPl(), 1);
    EXPECT_GE(graph.getNl(), 1);

    const auto results = graph.search_ef(std::span<const std::byte>(reinterpret_cast<const std::byte*>(features.data()), dim * sizeof(float)), 5, 10);
    EXPECT_LE(results.size(), 5u);
    EXPECT_GE(results.size(), 1u);
}

TEST(ReadOnlyGraphTest, HasPathAndFromGraphUseInternalGraphInterfaces) {
    constexpr uint32_t n = 16;
    constexpr uint32_t dim = 4;
    constexpr uint8_t k = 4;

    std::vector<float> features(static_cast<size_t>(n) * dim, 0.0f);
    std::vector<uint32_t> edges(static_cast<size_t>(n) * k);
    for (uint32_t i = 0; i < n; ++i) {
        features[static_cast<size_t>(i) * dim] = static_cast<float>(i);
        edges[static_cast<size_t>(i) * k + 0] = i;
        edges[static_cast<size_t>(i) * k + 1] = (i + 1) % n;
        edges[static_cast<size_t>(i) * k + 2] = (i + 2) % n;
        edges[static_cast<size_t>(i) * k + 3] = (i + 3) % n;
    }

    auto source = make_size_bounded_graph(n, dim, k, features, edges);
    source.setEntryVertexIndices({0});

    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);
    auto graph = deglib::graph::ReadOnlyGraph::from_graph(source, space);

    const auto path = graph.hasPath({0}, 10, 0.0f, 5);
    ASSERT_FALSE(path.empty());
    EXPECT_EQ(path.front().getIdentifier(), 10u);

    const auto copy = deglib::graph::ReadOnlyGraph::from_graph(graph, space);
    EXPECT_EQ(copy.size(), graph.size());
    EXPECT_EQ(copy.getEdgesPerVertex(), graph.getEdgesPerVertex());
    EXPECT_TRUE(copy.hasVertex(7));
    EXPECT_TRUE(copy.hasEdge(0, 1));
    EXPECT_EQ(copy.getExternalLabels(), graph.getExternalLabels());
}

TEST(ReadOnlyGraphTest, SerializesStandardGraphFormat) {
    constexpr uint32_t n = 16;
    constexpr uint32_t dim = 4;
    constexpr uint8_t k = 4;

    std::vector<float> features(static_cast<size_t>(n) * dim, 0.0f);
    std::vector<uint32_t> edges(static_cast<size_t>(n) * k);
    for (uint32_t i = 0; i < n; ++i) {
        features[static_cast<size_t>(i) * dim] = static_cast<float>(i);
        for (uint32_t j = 0; j < k; ++j) {
            edges[static_cast<size_t>(i) * k + j] = (i + j) % n;
        }
    }

    auto source = make_size_bounded_graph(n, dim, k, features, edges);
    source.setEntryVertexIndices({3});

    const auto path = std::filesystem::temp_directory_path() / "readonly_layout_graph.bin";
    ASSERT_TRUE(source.saveGraph(path.string().c_str()));

    const auto graph = deglib::graph::ReadOnlyGraph::loadGraph(path.string().c_str());
    EXPECT_EQ(graph.size(), n);
    EXPECT_EQ(graph.getEdgesPerVertex(), k);
    EXPECT_EQ(graph.getEntryVertexIndices(), (std::vector<uint32_t>{3}));
    EXPECT_EQ(graph.getPo(), source.getPo());
    EXPECT_EQ(graph.getPl(), source.getPl());
    EXPECT_EQ(graph.getNl(), source.getNl());
    EXPECT_EQ(graph.getExternalLabels(), source.getExternalLabels());

    for (uint32_t i = 0; i < n; ++i) {
        const auto* graph_features = graph.getFeatureVector(i);
        for (uint32_t j = 0; j < dim; ++j) {
            float value = 0.0f;
            std::memcpy(&value, graph_features + size_t(j) * sizeof(float), sizeof(float));
            EXPECT_FLOAT_EQ(value, features[static_cast<size_t>(i) * dim + j]);
        }

        const auto* neighbors = graph.getNeighborIndices(i);
        for (uint32_t j = 0; j < k; ++j) {
            EXPECT_EQ(neighbors[j], edges[static_cast<size_t>(i) * k + j]);
        }
    }

    std::remove(path.string().c_str());
}

TEST(ReadOnlyGraphTest, OptimizeIsNoopForEmptyGraph) {
    deglib::distances::FloatSpace space(4, deglib::distances::Metric::FP32_L2);
    deglib::graph::ReadOnlyGraph graph(1, 1, space);

    graph.optimize(0);

    EXPECT_EQ(graph.size(), 1u);
}

TEST(ReadOnlyGraphTest, DenseAndSparseLabelLookup) {
    constexpr uint32_t n = 16;
    constexpr uint32_t dim = 4;
    constexpr uint8_t k = 2;
    deglib::distances::FloatSpace space(dim, deglib::distances::Metric::FP32_L2);

    // 1. Dense labels 0..n-1 (exercises O(1) fast path)
    std::vector<float> features(n * dim, 0.0f);
    std::vector<uint32_t> edges(n * k, 0);
    for (uint32_t i = 0; i < n; ++i) {
        features[i * dim] = static_cast<float>(i);
        edges[i * k] = (i + 1) % n;
        edges[i * k + 1] = (i + 2) % n;
    }
    auto sb_dense = make_size_bounded_graph(n, dim, k, features, edges);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_TRUE(sb_dense.hasVertex(i));
        EXPECT_EQ(sb_dense.getInternalIndex(i), i);
    }
    EXPECT_FALSE(sb_dense.hasVertex(n));

    auto ro_dense = deglib::graph::ReadOnlyGraph::from_graph(sb_dense);
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_TRUE(ro_dense.hasVertex(i));
        EXPECT_EQ(ro_dense.getInternalIndex(i), i);
    }
    EXPECT_FALSE(ro_dense.hasVertex(n));

    // 2. Non-identity / sparse labels (exercises label_to_index_ fallback)
    deglib::graph::SizeBoundedGraph sb_sparse(n, k, space);
    std::vector<uint32_t> sparse_labels = {100, 0, 50, 2, 75, 1, 999, 7};
    for (size_t idx = 0; idx < sparse_labels.size(); ++idx) {
        std::array<float, dim> vec{static_cast<float>(idx), 0.0f, 0.0f, 0.0f};
        sb_sparse.addVertex(sparse_labels[idx], reinterpret_cast<const std::byte*>(vec.data()));
    }
    for (uint32_t idx = 0; idx < sparse_labels.size(); ++idx) {
        const uint32_t lbl = sparse_labels[idx];
        EXPECT_TRUE(sb_sparse.hasVertex(lbl));
        EXPECT_EQ(sb_sparse.getInternalIndex(lbl), idx);
    }
    EXPECT_FALSE(sb_sparse.hasVertex(3));

    auto ro_sparse = deglib::graph::ReadOnlyGraph::from_graph(sb_sparse);
    for (uint32_t idx = 0; idx < sparse_labels.size(); ++idx) {
        const uint32_t lbl = sparse_labels[idx];
        EXPECT_TRUE(ro_sparse.hasVertex(lbl));
        EXPECT_EQ(ro_sparse.getInternalIndex(lbl), idx);
    }
    EXPECT_FALSE(ro_sparse.hasVertex(3));
}

}  // namespace
