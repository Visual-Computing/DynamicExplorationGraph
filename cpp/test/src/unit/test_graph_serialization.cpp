#include <deglib/builder.h>
#include <deglib/deglib.h>
#include <deglib/graph.h>
#include <deglib/transformation.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

using namespace deglib;
using namespace deglib::transformation;

TEST(GraphSerializationTest, BuildWithPipelineSaveAndLoadSearch) {
    const uint32_t dim = 32;
    const uint32_t count = 50;
    const uint8_t edges_per_vertex = 4;
    std::mt19937 rnd(42);

    // 1. Zufällige Float-Vektoren erzeugen
    std::vector<float> data(count * dim);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : data) v = dist(rnd);

    // 2. Quantizer trainieren (z. B. Scalar INT8)
    auto sqi8 = std::make_shared<ScalarQuantizerInt8>(dim, 3.0f);
    sqi8->fit(data.data(), count);

    std::vector<std::shared_ptr<FeatureTransformation>> transformations = {sqi8};

    // 3. Mutable Graph mit INT8_L2 FeatureSpace anlegen
    const auto feature_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::Int8_L2);
    auto sizebounded = std::make_unique<deglib::graph::SizeBoundedGraph>(count, edges_per_vertex, feature_space);
    auto deg = DynamicExplorationGraph(std::move(sizebounded), transformations);

    EXPECT_TRUE(deg.hasTransformations());
    EXPECT_EQ(get_output_bytes(deg.getTransformations()), dim * sizeof(int8_t));

    // 4. Builder erstellen und Daten über Floats einspeisen (Ingestion-Kompression an der Eingangstür!)
    auto builder = deglib::create_builder(deg, rnd);
    EXPECT_TRUE(builder.hasTransformations());

    for (uint32_t i = 0; i < count; ++i) {
        std::span<const float> feat(data.data() + i * dim, dim);
        builder.addEntry(i, feat);
    }
    builder.build();
    EXPECT_EQ(deg.size(), count);

    // 5. Query-Suche mit Floats direkt über DEG Facade (ohne Searcher)
    std::span<const float> q0(data.data(), dim);
    auto res_before = deg.search(q0, 5, 0.1f);
    EXPECT_FALSE(res_before.empty());
    // Der Vektor 0 sollte der nächste Nachbar zu sich selbst sein
    bool found_self = false;
    for (const auto& od : res_before) {
        if (od.getIdentifier() == 0) found_self = true;
    }
    EXPECT_TRUE(found_self);

    // 6. Graph in Datei speichern (.deg v2 Container)
    const std::string test_file = "test_graph_container_pipeline.deg";
    deg.saveGraph(test_file);
    EXPECT_TRUE(std::filesystem::exists(test_file));
    // InternalGraph (ReadOnlyGraph) kann die Datei direkt laden, ignoriert den DEGX-Trailer
    auto ro_graph = deglib::graph::ReadOnlyGraph::loadGraph(test_file.c_str());
    EXPECT_EQ(ro_graph.size(), count);

    // 7. Graph aus Container-Datei neu laden
    auto deg_loaded = DynamicExplorationGraph::loadGraph(test_file);
    EXPECT_EQ(deg_loaded.size(), count);
    EXPECT_TRUE(deg_loaded.hasTransformations());
    EXPECT_EQ(deg_loaded.getInputDim(), dim);
    EXPECT_EQ(get_output_bytes(deg_loaded.getTransformations()), dim * sizeof(int8_t));

    // 8. Suche mit dem geladenen Graphen auf Float-Query
    auto res_after = deg_loaded.search(q0, 5, 0.1f);
    EXPECT_FALSE(res_after.empty());
    bool found_self_after = false;
    for (const auto& od : res_after) {
        if (od.getIdentifier() == 0) found_self_after = true;
    }
    EXPECT_TRUE(found_self_after);

    // 9. Batch-Suche über DEG
    std::vector<float> batch_queries(data.begin(), data.begin() + 3 * dim);
    auto batch_results = deg_loaded.search_batch(std::span<const float>(batch_queries), 3, 5, 0.1f);
    EXPECT_EQ(batch_results.size(), 3u);
    for (size_t q = 0; q < 3; ++q) {
        bool self_in_batch = false;
        for (const auto& od : batch_results[q]) {
            if (od.getIdentifier() == q) self_in_batch = true;
        }
        EXPECT_TRUE(self_in_batch);
    }

    // Cleanup
    std::filesystem::remove(test_file);
}

TEST(GraphSerializationTest, BuildWithEvpPipelineSaveAndLoad) {
    const uint32_t dim = 64;
    const uint32_t count = 30;
    const uint8_t edges_per_vertex = 4;
    const uint32_t non_zeros = 16;
    std::mt19937 rnd(123);

    std::vector<float> data(count * dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(rnd);

    auto evp = std::make_shared<EvpQuantizer>(dim, non_zeros);
    evp->fit(data.data(), count);

    std::vector<std::shared_ptr<FeatureTransformation>> transformations = {evp};

    const auto feature_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::EVP_InnerProduct);
    auto sizebounded = std::make_unique<deglib::graph::SizeBoundedGraph>(count, edges_per_vertex, feature_space);
    auto deg = DynamicExplorationGraph(std::move(sizebounded), transformations);

    auto builder = deglib::create_builder(deg, rnd);
    std::vector<uint32_t> initial_labels = {0, 1, 2, 3, 4};
    builder.addEntries(std::span<const uint32_t>(initial_labels), std::span<const float>(data.data(), 5 * dim));
    for (uint32_t i = 5; i < count; ++i) {
        builder.addEntry(i, std::span<const float>(data.data() + i * dim, dim));
    }
    builder.build();
    EXPECT_EQ(deg.size(), count);

    const std::string test_file = "test_graph_evp_container.deg";
    deg.saveGraph(test_file);
    EXPECT_TRUE(std::filesystem::exists(test_file));
    auto ro_graph = deglib::graph::ReadOnlyGraph::loadGraph(test_file.c_str());
    EXPECT_EQ(ro_graph.size(), count);

    auto deg_loaded = DynamicExplorationGraph::loadGraph(test_file);
    EXPECT_TRUE(deg_loaded.hasTransformations());
    EXPECT_EQ(get_output_bytes(deg_loaded.getTransformations()), 2 * (dim / 8));

    std::span<const float> q0(data.data(), dim);
    auto res = deg_loaded.search(q0, 3, 0.1f);
    EXPECT_FALSE(res.empty());

    std::filesystem::remove(test_file);
}
