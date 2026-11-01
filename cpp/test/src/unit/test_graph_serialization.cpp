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

    // 3. Float-Daten durch die Ingestion-Pipeline transformieren
    const size_t out_bytes = get_output_bytes(transformations);
    std::vector<std::byte> quantized_data(count * out_bytes);
    apply_database(transformations, data.data(), quantized_data.data(), count);

    // 4. Graph deterministisch und wohlgeformt mit create_random_graph erstellen
    const auto feature_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::Int8_L2);
    auto deg = deglib::create_random_graph(quantized_data.data(), count, edges_per_vertex, feature_space, 42, transformations);

    EXPECT_TRUE(deg.hasTransformations());
    EXPECT_EQ(get_output_bytes(deg.getTransformations()), dim * sizeof(int8_t));
    EXPECT_EQ(deg.size(), count);

    // 5. Query-Suche vor dem Speichern durchführen
    std::span<const float> q0(data.data(), dim);
    auto res_before = deg.search(q0, 5, 0.1f);
    EXPECT_FALSE(res_before.empty());

    std::vector<float> batch_queries(data.begin(), data.begin() + 3 * dim);
    auto batch_results_before = deg.search_batch(std::span<const float>(batch_queries), 3, 5, 0.1f);
    EXPECT_EQ(batch_results_before.size(), 3u);

    // 6. Graph in Datei speichern (.deg v2 Container)
    const std::string test_file = "test_graph_container_pipeline.deg";
    deg.saveGraph(test_file);
    EXPECT_TRUE(std::filesystem::exists(test_file));

    // InternalGraph (ReadOnlyGraph) kann die Datei direkt laden, ignoriert den DEGX-Trailer
    auto ro_graph = deglib::graph::ReadOnlyGraph::loadGraph(test_file.c_str());
    EXPECT_EQ(ro_graph.size(), count);

    // 7. Graph aus Container-Datei neu laden
    auto deg_loaded = DynamicExplorationGraph::load_readonly(test_file);
    EXPECT_EQ(deg_loaded.size(), count);
    EXPECT_TRUE(deg_loaded.hasTransformations());
    EXPECT_EQ(deg_loaded.getInputDim(), dim);
    EXPECT_EQ(get_output_bytes(deg_loaded.getTransformations()), dim * sizeof(int8_t));

    // 8. Suche nach dem Laden: Ergebnisse müssen exakt mit vor dem Speichern übereinstimmen
    auto res_after = deg_loaded.search(q0, 5, 0.1f);
    EXPECT_EQ(res_before.size(), res_after.size());

    // Sortierte Kopien der Ergebnisse vergleichen (Heap -> aufsteigend sortiert)
    auto sorted_before = res_before;
    sorted_before.sort();
    auto sorted_after = res_after;
    sorted_after.sort();
    for (size_t i = 0; i < sorted_before.size(); ++i) {
        EXPECT_EQ(sorted_before[i].getIdentifier(), sorted_after[i].getIdentifier());
        EXPECT_FLOAT_EQ(sorted_before[i].getDistance(), sorted_after[i].getDistance());
    }

    // 9. Batch-Suche nach dem Laden vergleichen
    auto batch_results_after = deg_loaded.search_batch(std::span<const float>(batch_queries), 3, 5, 0.1f);
    EXPECT_EQ(batch_results_after.size(), 3u);
    for (size_t q = 0; q < 3; ++q) {
        auto q_before = batch_results_before[q];
        q_before.sort();
        auto q_after = batch_results_after[q];
        q_after.sort();
        EXPECT_EQ(q_before.size(), q_after.size());
        for (size_t i = 0; i < q_before.size(); ++i) {
            EXPECT_EQ(q_before[i].getIdentifier(), q_after[i].getIdentifier());
            EXPECT_FLOAT_EQ(q_before[i].getDistance(), q_after[i].getDistance());
        }
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

    const size_t out_bytes = get_output_bytes(transformations);
    std::vector<std::byte> quantized_data(count * out_bytes);
    apply_database(transformations, data.data(), quantized_data.data(), count);

    const auto feature_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::EVP_InnerProduct);
    auto deg = deglib::create_random_graph(quantized_data.data(), count, edges_per_vertex, feature_space, 123, transformations);
    EXPECT_EQ(deg.size(), count);

    std::span<const float> q0(data.data(), dim);
    auto res_before = deg.search(q0, 3, 0.1f);
    EXPECT_FALSE(res_before.empty());

    const std::string test_file = "test_graph_evp_container.deg";
    deg.saveGraph(test_file);
    EXPECT_TRUE(std::filesystem::exists(test_file));
    auto ro_graph = deglib::graph::ReadOnlyGraph::loadGraph(test_file.c_str());
    EXPECT_EQ(ro_graph.size(), count);

    auto deg_loaded = DynamicExplorationGraph::load_readonly(test_file);
    EXPECT_TRUE(deg_loaded.hasTransformations());
    EXPECT_EQ(get_output_bytes(deg_loaded.getTransformations()), 2 * (dim / 8));

    auto res_after = deg_loaded.search(q0, 3, 0.1f);
    EXPECT_EQ(res_before.size(), res_after.size());

    auto sorted_before = res_before;
    sorted_before.sort();
    auto sorted_after = res_after;
    sorted_after.sort();
    for (size_t i = 0; i < sorted_before.size(); ++i) {
        EXPECT_EQ(sorted_before[i].getIdentifier(), sorted_after[i].getIdentifier());
        EXPECT_FLOAT_EQ(sorted_before[i].getDistance(), sorted_after[i].getDistance());
    }

    std::filesystem::remove(test_file);
}
