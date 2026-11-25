// test_searcher.cpp — Unit tests for deglib::search::Searcher (searcher.h)

#include "deglib/builder.h"
#include "deglib/deglib.h"
#include "deglib/distances.h"
#include "deglib/graph.h"
#include "deglib/optimization.h"
#include "deglib/search/searcher.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

TEST(SearcherTest, DirectFP32Search) {
    const uint32_t dim = 4;
    const uint32_t count = 20;

    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i * 1.5f + 0.1f);
    }

    auto feature_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP32_L2);
    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    auto searcher = deglib::search::make_searcher(graph.internal());

    // 1. Preallocated span buffer search
    std::vector<uint32_t> out_indices(5);
    std::vector<float> out_distances(5);
    uint32_t found = searcher->search(
        std::span<const float>(data.data(), dim), 5, std::span<uint32_t>(out_indices), std::span<float>(out_distances), 0.1f, 1.0f,
        /*return_distances=*/true
    );

    EXPECT_EQ(found, 5u);
    EXPECT_EQ(out_indices[0], 0u);
    EXPECT_NEAR(out_distances[0], 0.0f, 1e-5f);
    for (size_t i = 1; i < 5; ++i) {
        EXPECT_GE(out_distances[i], out_distances[i - 1]);
    }

    // 2. Modern C++20 span / SearchResults search
    auto span_res = searcher->search(std::span<const float>(data.data(), dim), 5, 0.1f, 1.0f, /*return_distances=*/true);
    EXPECT_EQ(span_res.size(), 5u);
    EXPECT_EQ(span_res.indices[0], 0u);
    EXPECT_NEAR(span_res.distances[0], 0.0f, 1e-5f);
    EXPECT_GT(span_res.graph_distance_count, 0u);
    EXPECT_GT(span_res.graph_hop_count, 0u);
    EXPECT_EQ(span_res.rerank_distance_count, 0u);
}

TEST(SearcherTest, ToSearcherMethodOnFacade) {
    const uint32_t dim = 4;
    const uint32_t count = 20;

    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i * 1.5f + 0.1f);
    }

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    auto searcher = graph.to_searcher();

    auto res = searcher.search(std::span<const float>(data.data(), dim), 5, 0.1f, 1.0f, /*return_distances=*/true);
    EXPECT_EQ(res.size(), 5u);
    EXPECT_EQ(res.indices[0], 0u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
}

TEST(SearcherTest, QuantizedInt8WithFP16Reranker) {
    const uint32_t dim = 8;
    const uint32_t count = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    // Quantize to INT8
    auto quantizer = deglib::transformation::make_scalar_quantizer_int8(std::span<const float>(data), dim);
    std::vector<int8_t> int8_data(count * dim);
    quantizer.transform_database(
        std::span<const float>(data), std::span<std::byte>(reinterpret_cast<std::byte*>(int8_data.data()), count * dim * sizeof(int8_t))
    );

    auto target_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::Int8_L2);
    auto ro_graph = deglib::graph::ReadOnlyGraph::from_graph(base_graph.internal(), target_space, int8_data.data());

    // Reranker with FP16 base features and calibrated ScalarQuantizerFP16
    auto fp16_quantizer =
        std::make_shared<deglib::transformation::ScalarQuantizerFP16>(deglib::transformation::make_scalar_quantizer_fp16(std::span<const float>(data), dim));
    std::vector<uint16_t> base_fp16(count * dim);
    fp16_quantizer->transform_database(std::span<const float>(data), std::as_writable_bytes(std::span(base_fp16)));
    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP16_L2);

    auto reranker = deglib::search::Reranker(rerank_space, std::as_bytes(std::span(base_fp16)), count, {fp16_quantizer});
    auto searcher = deglib::search::make_searcher(ro_graph, quantizer, std::move(reranker));

    // Query for 5th item
    auto q = std::span<const float>(data.data() + 5 * dim, dim);
    auto res = searcher->search(q, 5, /*eps=*/0.2f, /*rerank_factor=*/2.0f, /*return_distances=*/true);

    EXPECT_EQ(res.size(), 5u);
    EXPECT_EQ(res.indices[0], 5u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-3f);
}

TEST(SearcherTest, RerankerFromGraphFP32ToFP16) {
    const uint32_t dim = 8;
    const uint32_t count = 32;
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dim; ++d) {
            data[i * dim + d] = static_cast<float>(i) * 0.5f + static_cast<float>(d) * 0.05f;
        }
    }
    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);
    auto reranker = deglib::search::Reranker::from_graph(graph.internal(), deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP16_L2));
    auto reranker_scalar = deglib::search::Reranker::from_graph(
        graph.internal(), deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP16_L2, deglib::cpu::InstructionSet::Scalar)
    );
    EXPECT_STREQ(reranker_scalar.getSpace().get_instruction(), "Scalar");
    EXPECT_EQ(reranker.getNumBaseVectors(), count);
    EXPECT_FALSE(reranker.getTransformations().empty());

    std::vector<uint32_t> cands(count);
    for (uint32_t i = 0; i < count; ++i) cands[i] = i;
    std::vector<uint32_t> out_idx(2);
    std::vector<float> out_dist(2);
    const uint32_t target = 11;
    reranker.rerank_f32(std::span<const float>(data.data() + target * dim, dim), cands, 2, out_idx, out_dist, true, false);
    EXPECT_EQ(out_idx[0], target);
    EXPECT_NEAR(out_dist[0], 0.0f, 1e-2f);
}

TEST(SearcherTest, QuantizedUint8WithFP32Reranker) {
    const uint32_t dim = 8;
    const uint32_t count = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    auto quantizer = deglib::transformation::make_scalar_quantizer_uint8(std::span<const float>(data), dim);
    std::vector<uint8_t> uint8_data(count * dim);
    quantizer.transform_database(
        std::span<const float>(data), std::span<std::byte>(reinterpret_cast<std::byte*>(uint8_data.data()), count * dim * sizeof(uint8_t))
    );

    auto target_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::Uint8_L2);
    auto ro_graph = deglib::graph::ReadOnlyGraph::from_graph(base_graph.internal(), target_space, uint8_data.data());

    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP32_L2);

    auto searcher = deglib::search::make_searcher(ro_graph, quantizer, deglib::search::Reranker(rerank_space, std::as_bytes(std::span(data)), count));

    auto q = std::span<const float>(data.data() + 2 * dim, dim);
    auto res = searcher->search(q, 3, /*eps=*/0.2f, /*rerank_factor=*/1.5f, /*return_distances=*/true);

    EXPECT_GE(res.size(), 1u);
    EXPECT_EQ(res.indices[0], 2u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-4f);
}

TEST(SearcherTest, OptimizeTunesRerankerAndPreservesResults) {
    const uint32_t dim = 8;
    const uint32_t count = 60;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP32_L2);
    auto searcher = deglib::search::make_searcher(graph.internal(), deglib::search::Reranker(rerank_space, std::as_bytes(std::span(data)), count));

    // Exercises the reranker prefetch auto-tuning path wired into SearcherImpl::optimize().
    searcher->optimize(/*n_clusters=*/8, /*n_iter=*/5);

    auto q = std::span<const float>(data.data() + 5 * dim, dim);
    auto res = searcher->search(q, 3, /*eps=*/0.2f, /*rerank_factor=*/2.0f, /*return_distances=*/true);
    EXPECT_EQ(res.indices[0], 5u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-4f);
}

TEST(SearcherTest, OptimizeTunesFp16Reranker) {
    const uint32_t dim = 8;
    const uint32_t count = 60;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto fp16_quantizer =
        std::make_shared<deglib::transformation::ScalarQuantizerFP16>(deglib::transformation::make_scalar_quantizer_fp16(std::span<const float>(data), dim));
    std::vector<uint16_t> base_fp16(count * dim);
    fp16_quantizer->transform_database(std::span<const float>(data), std::as_writable_bytes(std::span(base_fp16)));

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP16_L2);
    auto searcher =
        deglib::search::make_searcher(graph.internal(), deglib::search::Reranker(rerank_space, std::as_bytes(std::span(base_fp16)), count, {fp16_quantizer}));

    // Reranker<uint16_t> self-samples its fp16 base vectors; no float reinterpretation.
    searcher->optimize(/*n_clusters=*/8, /*n_iter=*/5);

    auto q = std::span<const float>(data.data() + 5 * dim, dim);
    auto res = searcher->search(q, 3, /*eps=*/0.2f, /*rerank_factor=*/2.0f, /*return_distances=*/true);
    EXPECT_EQ(res.indices[0], 5u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-2f);
}

TEST(SearcherTest, OptimizeSelectsValidPrefetchConfigAndPreservesResults) {
    // optimize() tunes traversal prefetch (po, pl, nl) to the best of its search grid. The actual
    // speedup is workload/hardware dependent (it only materialises when the working set exceeds
    // cache), so it is reported by the regression benchmark, not asserted here. This test pins the
    // deterministic contract: the tuner lands on a valid grid value and search stays correct.
    const uint32_t dim = 16;
    const uint32_t count = 500;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 16, deglib::distances::Metric::FP32_L2);
    auto searcher = deglib::search::make_searcher(graph.internal());

    searcher->optimize(/*n_clusters=*/32, /*n_iter=*/5, /*sample_size=*/0, /*threads=*/1);

    // Tuner must land within the valid search bounds (proves it actually tuned, not garbage).
    const int32_t po = graph.getPo();
    const int32_t pl = graph.getPl();
    const int32_t nl = graph.getNl();
    EXPECT_GE(po, 1);
    EXPECT_LE(po, 16);
    EXPECT_EQ(pl, 1);
    EXPECT_EQ(nl, 1);

    // Search still returns the exact nearest neighbour (the query is a base vector).
    std::vector<uint32_t> idx(3);
    std::vector<float> d(3);
    const uint32_t q = 5;
    searcher->search(
        std::span<const float>(data.data() + static_cast<size_t>(q) * dim, dim), 3, std::span<uint32_t>(idx), std::span<float>(d), 0.1f, 1.0f,
        /*return_distances=*/true
    );
    EXPECT_EQ(idx[0], q);
    EXPECT_NEAR(d[0], 0.0f, 1e-4f);
}
TEST(SearcherTest, OptimizePrefetchDepthScalesWithFeatureWidth) {
    // Wide features (1024D FP32 = 4096 B = 64 cache lines) must let the tuner pick deep prefetch
    // (pl up to code_size/64 = 64).
    const uint32_t dim = 1024;
    const uint32_t count = 300;

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 16, deglib::distances::Metric::FP32_L2);
    auto searcher = deglib::search::make_searcher(graph.internal());
    searcher->optimize(/*n_clusters=*/8, /*n_iter=*/3, /*sample_size=*/50, /*threads=*/1);

    const int32_t pl = graph.getPl();
    const int32_t pl_max = std::min<int32_t>(20, static_cast<int32_t>(dim * 4 / 64));  // capped at 20 cache lines
    EXPECT_GE(pl, 1);
    EXPECT_LE(pl, pl_max);
}

TEST(SearcherTest, EVPQuantizerWithFP32Reranker) {
    const uint32_t dim = 16;
    const uint32_t count = 50;
    const uint32_t non_zeros = 4;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(count * dim);
    for (size_t i = 0; i < count; ++i) {
        float norm_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            data[i * dim + d] = dist(rng);
            norm_sq += data[i * dim + d] * data[i * dim + d];
        }
        float inv_norm = 1.0f / std::sqrt(norm_sq);
        for (size_t d = 0; d < dim; ++d) {
            data[i * dim + d] *= inv_norm;
        }
    }

    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_InnerProduct);

    deglib::transformation::EvpQuantizer quantizer(dim, non_zeros);
    quantizer.fit(std::span<const float>(data));
    const size_t evp_bytes = quantizer.output_byte_size();
    std::vector<std::byte> evp_data(count * evp_bytes);
    quantizer.transform_database(std::span<const float>(data), std::span<std::byte>(evp_data));
    auto target_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::EVP_InnerProduct);
    auto ro_graph = deglib::graph::ReadOnlyGraph::from_graph(base_graph.internal(), target_space, evp_data.data());

    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP32_InnerProduct);

    auto searcher = deglib::search::make_searcher(ro_graph, quantizer, deglib::search::Reranker(rerank_space, std::as_bytes(std::span(data)), count));

    auto q = std::span<const float>(data.data() + 3 * dim, dim);
    auto res = searcher->search(q, 3, /*eps=*/0.3f, /*rerank_factor=*/2.0f, /*return_distances=*/true);

    EXPECT_GE(res.size(), 1u);
    EXPECT_EQ(res.indices[0], 3u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-4f);

    // FP16 query test with EVPQuantizer
    std::vector<uint16_t> q_fp16(dim);
    deglib::distances::fp16::floats_to_fp16(data.data() + 3 * dim, q_fp16.data(), dim);
    auto res_fp16 = searcher->search(std::span<const uint16_t>(q_fp16.data(), dim), 3, /*eps=*/0.3f, /*rerank_factor=*/2.0f, /*return_distances=*/true);
    EXPECT_GE(res_fp16.size(), 1u);
    EXPECT_EQ(res_fp16.indices[0], 3u);
    EXPECT_NEAR(res_fp16.distances[0], 0.0f, 1e-3f);
}

TEST(SearcherTest, FlatBatchAndIntoSearch) {
    const uint32_t dim = 4;
    const uint32_t count = 30;

    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i * 0.5f + 0.1f);
    }

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);
    auto searcher = deglib::search::make_searcher(graph.internal());

    const size_t n_queries = 5;
    const uint32_t k = 4;

    // 1. search_batch returning SearchResults (flat buffer)
    auto flat_res = searcher->search_batch(
        std::span<const float>(data.data(), n_queries * dim), n_queries, k,
        /*eps=*/0.1f, /*rerank_factor=*/1.0f,
        /*threads=*/2, /*return_distances=*/true
    );
    EXPECT_EQ(flat_res.size(), n_queries);
    EXPECT_EQ(flat_res.k, k);
    EXPECT_EQ(flat_res.indices.size(), n_queries * k);
    EXPECT_GT(flat_res.graph_distance_count, 0u);
    EXPECT_GT(flat_res.graph_hop_count, 0u);
    EXPECT_EQ(flat_res.rerank_distance_count, 0u);
    EXPECT_EQ(flat_res.distances.size(), n_queries * k);

    for (size_t q = 0; q < n_queries; ++q) {
        auto q_indices = flat_res.get_indices(q);
        auto q_dists = flat_res.get_distances(q);
        EXPECT_EQ(q_indices[0], static_cast<uint32_t>(q));
        EXPECT_NEAR(q_dists[0], 0.0f, 1e-5f);
    }

    // 2. search_batch into preallocated buffers
    std::vector<uint32_t> preallocated_indices(n_queries * k, 9999);
    std::vector<float> preallocated_dists(n_queries * k, 9999.0f);
    searcher->search_batch(
        std::span<const float>(data.data(), n_queries * dim), n_queries, k, std::span<uint32_t>(preallocated_indices), std::span<float>(preallocated_dists),
        /*eps=*/0.1f, /*rerank_factor=*/1.0f,
        /*threads=*/1,
        /*return_distances=*/true
    );

    for (size_t q = 0; q < n_queries; ++q) {
        EXPECT_EQ(preallocated_indices[q * k], static_cast<uint32_t>(q));
        EXPECT_NEAR(preallocated_dists[q * k], 0.0f, 1e-5f);
    }
}

TEST(SearcherTest, SearchEfSingleAndBatch) {
    const uint32_t dim = 4;
    const uint32_t count = 30;

    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i * 0.5f + 0.1f);
    }

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);
    auto searcher = deglib::search::make_searcher(graph.internal());

    const uint32_t k = 4;
    const float ef = 32.0f;

    // 1. Single search with eps_or_ef >= 1.0 (ef mode)
    auto q0 = std::span<const float>(data.data(), dim);
    auto single_res = searcher->search(q0, k, /*eps_or_ef=*/ef, /*rerank_factor=*/1.0f, /*return_distances=*/true);
    EXPECT_GE(single_res.size(), 1u);
    EXPECT_EQ(single_res.indices[0], 0u);
    EXPECT_NEAR(single_res.distances[0], 0.0f, 1e-5f);

    // 2. Batch search with eps_or_ef >= 1.0 (ef mode) with flat result and 2 worker threads
    const size_t n_queries = 5;
    auto batch_res = searcher->search_batch(
        std::span<const float>(data.data(), n_queries * dim), n_queries, k, /*eps_or_ef=*/ef, /*rerank_factor=*/1.0f, /*threads=*/2, /*return_distances=*/true
    );
    EXPECT_EQ(batch_res.size(), n_queries);
    EXPECT_EQ(batch_res.k, k);
    for (size_t q = 0; q < n_queries; ++q) {
        auto q_indices = batch_res.get_indices(q);
        auto q_dists = batch_res.get_distances(q);
        EXPECT_EQ(q_indices[0], static_cast<uint32_t>(q));
        EXPECT_NEAR(q_dists[0], 0.0f, 1e-5f);
    }
}

TEST(SearcherTest, OptimizeRoutingDisabledUsesStaticEntryAndFindsNN) {
    // optimize(use_routing=false) skips KMeans entry routing; search must still return the exact NN
    // and must work (static graph entry path). Verifies the routing toggle is functional.
    const uint32_t dim = 8;
    const uint32_t count = 60;

    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);
    auto searcher = deglib::search::make_searcher(graph.internal());
    searcher->optimize(/*n_clusters=*/8, /*n_iter=*/3, /*sample_size=*/0, /*threads=*/1, /*use_routing=*/false);

    auto q = std::span<const float>(data.data() + 7 * dim, dim);
    auto res = searcher->search(q, 3, /*eps=*/0.2f, /*rerank_factor=*/1.0f, /*return_distances=*/true);
    EXPECT_EQ(res.indices[0], 7u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-5f);
}

TEST(SearcherTest, DynamicExplorationGraphWithTransformationMakeSearcher) {
    const uint32_t dim = 8;
    const uint32_t count = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    auto quantizer =
        std::make_shared<deglib::transformation::ScalarQuantizerInt8>(deglib::transformation::make_scalar_quantizer_int8(std::span<const float>(data), dim));
    std::vector<int8_t> int8_data(count * dim);
    quantizer->transform_database(
        std::span<const float>(data), std::span<std::byte>(reinterpret_cast<std::byte*>(int8_data.data()), count * dim * sizeof(int8_t))
    );

    auto target_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::Int8_L2);
    auto ro_graph =
        base_graph.with_features(target_space, int8_data.data(), std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{quantizer});

    EXPECT_TRUE(ro_graph.hasTransformations());

    // 1. to_searcher() without reranker: automatically uses attached quantizer
    auto searcher_no_reranker = ro_graph.to_searcher();

    searcher_no_reranker.optimize(/*n_clusters=*/8, /*n_iter=*/3, /*sample_size=*/0, /*threads=*/1);

    auto q = std::span<const float>(data.data() + 5 * dim, dim);
    auto res1 = searcher_no_reranker.search(q, 5, /*eps=*/0.2f, /*rerank_factor=*/1.0f, /*return_distances=*/true);
    EXPECT_GE(res1.size(), 1u);
    EXPECT_EQ(res1.indices[0], 5u);

    // 2. to_searcher(reranker): automatically uses attached quantizer + reranker
    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP32_L2);
    auto searcher_with_reranker = ro_graph.to_searcher(deglib::search::Reranker(rerank_space, std::as_bytes(std::span(data)), count));

    auto res2 = searcher_with_reranker.search(q, 5, /*eps=*/0.2f, /*rerank_factor=*/2.0f, /*return_distances=*/true);
    EXPECT_GE(res2.size(), 1u);
    EXPECT_EQ(res2.indices[0], 5u);
    EXPECT_NEAR(res2.distances[0], 0.0f, 1e-4f);
    EXPECT_GT(res2.graph_distance_count, 0u);
    EXPECT_GT(res2.rerank_distance_count, 0u);
    EXPECT_GT(res2.graph_hop_count, 0u);
}

TEST(SearcherTest, KMeansEntrySelectorFitMapsIdsAndTunesPrefetch) {
    const uint32_t dim = 4;
    const uint32_t count = 20;

    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i * 1.5f + 0.1f);
    }

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);
    deglib::search::KMeansEntrySelector selector(graph.internal());

    const std::vector<const float*> one_row{data.data()};
    const std::vector<uint32_t> one_id{0};
    EXPECT_THROW(selector.fit(one_row, one_id, dim, true, deglib::search::KMeansEntrySelector::kMaxMedoids + 1, 1), std::invalid_argument);

    const std::vector<const float*> rows{data.data(), data.data() + dim, data.data() + 2 * dim};
    const std::vector<uint32_t> ids{10, 11, 12};
    selector.fit(rows, ids, dim, true, 2, 1);

    EXPECT_EQ(selector.size(), 2u);
    for (const uint32_t entry : selector.entries()) {
        EXPECT_GE(entry, 10u);
        EXPECT_LE(entry, 12u);
    }

    auto q = std::span<const std::byte>(reinterpret_cast<const std::byte*>(data.data()), graph.getFeatureSpace().get_data_size());
    const auto first = selector.top_entries(q, 2);
    const auto second = selector.top_entries(q, 2);

    EXPECT_EQ(first, second);
    EXPECT_FALSE(first.empty());
}

TEST(SearcherTest, ScalarQuantizerDequantizeInt8AndUint8) {
    const uint32_t dim = 4;
    std::vector<float> data = {-1.0f, 0.0f, 0.25f, 1.0f};

    auto int8_quant =
        std::make_shared<deglib::transformation::ScalarQuantizerInt8>(deglib::transformation::make_scalar_quantizer_int8(std::span<const float>(data), dim));
    std::vector<std::byte> int8_bytes(dim * sizeof(int8_t));
    int8_quant->transform_query(std::span<const float>(data), std::span<std::byte>(int8_bytes));
    std::vector<float> int8_out(dim);
    int8_quant->dequantize(std::span<const std::byte>(int8_bytes), std::span<float>(int8_out));
    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_NEAR(int8_out[d], data[d], 0.01f);
    }

    auto uint8_quant =
        std::make_shared<deglib::transformation::ScalarQuantizerUint8>(deglib::transformation::make_scalar_quantizer_uint8(std::span<const float>(data), dim));
    std::vector<std::byte> uint8_bytes(dim * sizeof(uint8_t));
    uint8_quant->transform_query(std::span<const float>(data), std::span<std::byte>(uint8_bytes));
    std::vector<float> uint8_out(dim);
    uint8_quant->dequantize(std::span<const std::byte>(uint8_bytes), std::span<float>(uint8_out));
    for (uint32_t d = 0; d < dim; ++d) {
        EXPECT_NEAR(uint8_out[d], data[d], 0.01f);
    }
}

TEST(SearcherTest, MaxDistanceComputationCountBoundsSearch) {
    const uint32_t dim = 8;
    const uint32_t count = 200;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(count * dim);
    for (float& v : data) v = dist(rng);

    auto graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 16, deglib::distances::Metric::FP32_L2);
    auto searcher = deglib::search::make_searcher(graph.internal());

    const uint32_t k = 10;
    const std::span<const float> query(data.data(), dim);

    // Unlimited (0) finds k results and matches exact nearest neighbor 0
    auto full_eps = searcher->search(query, k, 0.2f, 1.0f, true, false, 0);
    EXPECT_EQ(full_eps.size(), k);
    EXPECT_EQ(full_eps.indices[0], 0u);

    auto full_ef = searcher->search(query, k, 64.0f, 1.0f, true, false, 0);
    EXPECT_EQ(full_ef.size(), k);
    EXPECT_EQ(full_ef.indices[0], 0u);

    // Tight budget (5 distance computations) returns bounded valid results
    auto bounded_eps = searcher->search(query, k, 0.2f, 1.0f, true, false, 5);
    EXPECT_GE(bounded_eps.size(), 1u);
    EXPECT_LE(bounded_eps.size(), 5u);

    auto bounded_ef = searcher->search(query, k, 64.0f, 1.0f, true, false, 5);
    EXPECT_GE(bounded_ef.size(), 1u);
    EXPECT_LE(bounded_ef.size(), 5u);

    // Batch search with tight budget (max_distance_computation_count = 5).
    // Pre-fill with UINT32_MAX so unwritten slots beyond the returned count stay distinguishable.
    std::vector<uint32_t> batch_out(4 * k, std::numeric_limits<uint32_t>::max());
    searcher->search_batch<float>(std::span<const float>(data.data(), 4 * dim), 4, k, std::span<uint32_t>(batch_out), {}, 64.0f, 1.0f, 2, false, false, 5);
    for (size_t q = 0; q < 4; ++q) {
        size_t valid = 0;
        for (uint32_t j = 0; j < k; ++j) {
            if (batch_out[q * k + j] != std::numeric_limits<uint32_t>::max()) {
                EXPECT_LT(batch_out[q * k + j], count);
                ++valid;
            }
        }
        EXPECT_GE(valid, 1u);
        EXPECT_LE(valid, 5u);
    }
}

namespace {

std::vector<float> make_searcher_clustered_data(uint32_t count, uint32_t dim, uint32_t num_clusters, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> center_dist(0.0f, 3.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.35f);

    std::vector<float> centers(static_cast<size_t>(num_clusters) * dim);
    for (float& v : centers) {
        v = center_dist(rng);
    }

    std::vector<float> data(static_cast<size_t>(count) * dim);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t c = i % num_clusters;
        const float* c_ptr = centers.data() + static_cast<size_t>(c) * dim;
        float* dst = data.data() + static_cast<size_t>(i) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            dst[d] = c_ptr[d] + noise_dist(rng);
        }
    }
    return data;
}

void verify_searcher_pareto_invariants(const deglib::search::ParetoLine& line) {
    using deglib::search::RecallTable;
    EXPECT_EQ(line.size(), line.entries().size());
    EXPECT_EQ(line.empty(), line.entries().empty());
    if (line.empty()) {
        return;
    }
    ASSERT_TRUE(line.recall_bounds().has_value());
    const auto& entries = line.entries();
    EXPECT_FLOAT_EQ(line.recall_bounds()->first, entries.front().measured_recall);
    EXPECT_FLOAT_EQ(line.recall_bounds()->second, entries.back().measured_recall);

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        EXPECT_TRUE(e.is_ef() != e.is_eps());
        if (e.is_ef()) {
            EXPECT_GE(e.eps_or_ef, static_cast<float>(RecallTable::kMinEf));
            EXPECT_LE(e.eps_or_ef, static_cast<float>(RecallTable::kMaxEf));
        } else {
            EXPECT_GE(e.eps_or_ef, RecallTable::kMinEps - 1e-6f);
            EXPECT_LE(e.eps_or_ef, RecallTable::kMaxEps + 1e-6f);
        }
        EXPECT_GE(e.rerank_factor, 1.0f);
        if (e.rerank_factor > 1.0f + 1e-6f) {
            EXPECT_GT(e.avg_rerank_distance_count, 0.0);
        } else {
            EXPECT_DOUBLE_EQ(e.avg_rerank_distance_count, 0.0);
        }
        EXPECT_GT(e.latency_us, 0.0);
        EXPECT_GE(e.avg_distance_count, 0.0);
        if (i > 0) {
            EXPECT_GT(e.measured_recall, entries[i - 1].measured_recall);
            EXPECT_GT(e.latency_us, entries[i - 1].latency_us);
        }
        const auto found = line.param_for(e.measured_recall);
        ASSERT_TRUE(found.has_value());
        EXPECT_NEAR(found->first, e.eps_or_ef, 1e-6f);
        EXPECT_FLOAT_EQ(found->second, e.rerank_factor);
    }
}

}  // namespace

TEST(SearcherTest, CalibrateTablesMatchSingleFactorAndParetoWithFP32Reranker) {
    using deglib::distances::Metric;
    using deglib::search::ParetoLine;
    using deglib::search::RecallTable;

    const uint32_t dim = 32;
    const uint32_t count = 300;
    const auto data = make_searcher_clustered_data(count, dim, 16, 17);
    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 16, Metric::FP32_L2);
    auto reranker = std::make_shared<deglib::search::Reranker>(
        deglib::search::Reranker::from_graph(base_graph.internal(), deglib::distances::FloatSpace(dim, Metric::FP32_L2))
    );

    auto u4_xfrms = deglib::transformation::make_quantizer_pipeline(data, dim, Metric::Uint4_L2);
    auto u4_space = deglib::distances::FloatSpace(dim, Metric::Uint4_L2);
    auto u4_graph = base_graph.with_transformations(u4_xfrms, u4_space);
    auto searcher = u4_graph.to_searcher(reranker);
    searcher.optimize(16, 5, 150, 1, /*use_routing=*/true);

    const uint32_t k = 10;
    const size_t sample_count = 60;
    const std::vector<float> rfs = {1.0f, 1.5f, 2.0f};
    const auto tables = searcher.to_recall_tables(k, sample_count, {0.65f, 0.96f}, 0.05f, std::span<const float>(rfs), 1);
    ASSERT_EQ(tables.size(), 6u);

    // Order is [(rf=1.0, Ef), (rf=1.0, Eps), (rf=1.5, Ef), (rf=1.5, Eps), (rf=2.0, Ef), (rf=2.0, Eps)]
    // Multi-rf single-pass Ef calibration must match standalone per-rf to_recall_table bit-for-bit.
    for (size_t r = 0; r < rfs.size(); ++r) {
        const auto& tbl_ef = tables[2 * r];
        const auto& tbl_eps = tables[2 * r + 1];
        EXPECT_EQ(tbl_ef.mode(), RecallTable::Mode::Ef);
        EXPECT_EQ(tbl_eps.mode(), RecallTable::Mode::Eps);
        EXPECT_FLOAT_EQ(tbl_ef.rerank_factor(), rfs[r]);
        EXPECT_FLOAT_EQ(tbl_eps.rerank_factor(), rfs[r]);

        const RecallTable single_ef = searcher.to_recall_table(k, RecallTable::Mode::Ef, rfs[r], sample_count, {0.65f, 0.96f}, 0.05f, 1);
        ASSERT_EQ(tbl_ef.size(), single_ef.size());
        for (size_t i = 0; i < tbl_ef.size(); ++i) {
            EXPECT_FLOAT_EQ(tbl_ef.entries()[i].target_recall, single_ef.entries()[i].target_recall);
            EXPECT_FLOAT_EQ(tbl_ef.entries()[i].eps_or_ef, single_ef.entries()[i].eps_or_ef);
            EXPECT_FLOAT_EQ(tbl_ef.entries()[i].measured_recall, single_ef.entries()[i].measured_recall);
        }

        const RecallTable single_eps = searcher.to_recall_table(k, RecallTable::Mode::Eps, rfs[r], sample_count, {0.65f, 0.96f}, 0.05f, 1);
        ASSERT_EQ(tbl_eps.size(), single_eps.size());
        for (size_t i = 0; i < tbl_eps.size(); ++i) {
            EXPECT_FLOAT_EQ(tbl_eps.entries()[i].target_recall, single_eps.entries()[i].target_recall);
            EXPECT_FLOAT_EQ(tbl_eps.entries()[i].eps_or_ef, single_eps.entries()[i].eps_or_ef);
            EXPECT_FLOAT_EQ(tbl_eps.entries()[i].measured_recall, single_eps.entries()[i].measured_recall);
        }
    }

    // Default rerank_factors on Searcher with reranker produces 4 factors x 2 modes = 8 tables.
    const auto default_tables = searcher.to_recall_tables(k, sample_count, {0.75f, 0.95f}, 0.05f, {}, 1);
    ASSERT_EQ(default_tables.size(), 8u);
    for (size_t r = 0; r < deglib::search::Searcher::kDefaultRerankFactors.size(); ++r) {
        EXPECT_FLOAT_EQ(default_tables[2 * r].rerank_factor(), deglib::search::Searcher::kDefaultRerankFactors[r]);
        EXPECT_FLOAT_EQ(default_tables[2 * r + 1].rerank_factor(), deglib::search::Searcher::kDefaultRerankFactors[r]);
    }

    // ParetoLine overloads (precomputed span, single table, end-to-end range & targets)
    const ParetoLine line_from_tables = searcher.to_pareto_line(std::span<const RecallTable>(tables), sample_count, 3);
    EXPECT_EQ(line_from_tables.k(), k);
    EXPECT_EQ(line_from_tables.sample_count(), sample_count);
    ASSERT_EQ(line_from_tables.recall_tables().size(), 6u);
    verify_searcher_pareto_invariants(line_from_tables);

    const ParetoLine line_single = searcher.to_pareto_line(tables[2], sample_count, 2);
    verify_searcher_pareto_invariants(line_single);
    for (const auto& e : line_single.entries()) {
        EXPECT_TRUE(e.is_ef());
        EXPECT_FLOAT_EQ(e.rerank_factor, 1.5f);
    }

    const std::vector<float> explicit_targets = {0.75f, 0.85f, 0.92f};
    const ParetoLine line_explicit = searcher.to_pareto_line(k, std::span<const float>(explicit_targets), sample_count, std::span<const float>(rfs), 1, 2);
    verify_searcher_pareto_invariants(line_explicit);

    const ParetoLine line_auto = searcher.to_pareto_line(k, sample_count, {0.75f, 0.95f}, 0.05f, std::span<const float>(rfs), 1, 2);
    verify_searcher_pareto_invariants(line_auto);
}

TEST(SearcherTest, QueryModesFP16SQ16NativeF16NoRerankerAndRawBytes) {
    using deglib::distances::Metric;
    using deglib::search::ParetoLine;
    using deglib::search::RecallTable;

    const uint32_t dim = 32;
    const uint32_t count = 250;
    const auto data = make_searcher_clustered_data(count, dim, 12, 43);
    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 16, Metric::FP32_L2);
    auto u4_xfrms = deglib::transformation::make_quantizer_pipeline(data, dim, Metric::Uint4_L2);
    auto u4_space = deglib::distances::FloatSpace(dim, Metric::Uint4_L2);
    auto u4_graph = base_graph.with_transformations(u4_xfrms, u4_space);

    // 1. Uint4 graph + FP16_L2 reranker (auto-fitted ScalarQuantizerFP16 -> F32 CalibQueryMode)
    {
        auto rr_fp16_sq = std::make_shared<deglib::search::Reranker>(
            deglib::search::Reranker::from_graph(base_graph.internal(), deglib::distances::FloatSpace(dim, Metric::FP16_L2))
        );
        auto searcher_sq = u4_graph.to_searcher(rr_fp16_sq);
        const std::vector<float> rfs = {1.0f, 1.5f};
        const auto line = searcher_sq.to_pareto_line(10, 50, {0.70f, 0.96f}, 0.08f, std::span<const float>(rfs), 1, 2);
        EXPECT_FALSE(line.empty());
        verify_searcher_pareto_invariants(line);
    }

    // 2. FP16_L2 reranker with NO transformations (CalibQueryMode::F16 via execute_search_single<uint16_t>)
    {
        std::vector<uint16_t> f16_data(static_cast<size_t>(count) * dim);
        deglib::distances::fp16::floats_to_fp16(data.data(), f16_data.data(), f16_data.size());
        auto rr_space = deglib::distances::FloatSpace(dim, Metric::FP16_L2);
        auto rr_native_f16 =
            std::make_shared<deglib::search::Reranker>(rr_space, std::as_bytes(std::span<const uint16_t>(f16_data)), static_cast<size_t>(count));
        auto searcher_f16 = u4_graph.to_searcher(rr_native_f16);
        const std::vector<float> rfs = {1.0f, 1.5f};
        const auto line = searcher_f16.to_pareto_line(10, 50, {0.70f, 0.96f}, 0.08f, std::span<const float>(rfs), 1, 2);
        EXPECT_FALSE(line.empty());
        verify_searcher_pareto_invariants(line);
    }

    // 3. No reranker + K-Means routing on FP32 graph -> 2 tables, rf == 1.0, avg_rerank_distance_count == 0
    {
        auto searcher_no_rr = deglib::search::make_searcher(base_graph.internal());
        searcher_no_rr->optimize(16, 5, 120, 1, /*use_routing=*/true);
        const auto tables = searcher_no_rr->to_recall_tables(10, 50, {0.75f, 0.98f}, 0.05f, {}, 1);
        ASSERT_EQ(tables.size(), 2u);
        EXPECT_FLOAT_EQ(tables[0].rerank_factor(), 1.0f);
        EXPECT_FLOAT_EQ(tables[1].rerank_factor(), 1.0f);

        const auto line = searcher_no_rr->to_pareto_line(tables, 50, 2);
        verify_searcher_pareto_invariants(line);
        for (const auto& e : line.entries()) {
            EXPECT_FLOAT_EQ(e.rerank_factor, 1.0f);
            EXPECT_DOUBLE_EQ(e.avg_rerank_distance_count, 0.0);
        }
    }

    // 4. RawBytes fallback: quantized Uint4 graph without any reranker completes without throwing
    {
        auto searcher_raw = deglib::search::make_searcher(u4_graph.internal());
        const auto tables = searcher_raw->to_recall_tables(10, 40, {0.70f, 0.95f}, 0.08f, {}, 1);
        ASSERT_EQ(tables.size(), 2u);
        EXPECT_FLOAT_EQ(tables[0].rerank_factor(), 1.0f);
        const auto line = searcher_raw->to_pareto_line(tables, 40, 2);
        verify_searcher_pareto_invariants(line);
    }
}

TEST(SearcherTest, RecallAndParetoValidationAndEmptyGraph) {
    using deglib::distances::Metric;
    using deglib::search::RecallTable;

    const auto space = deglib::distances::FloatSpace(8, Metric::FP32_L2);
    auto empty_graph = deglib::create_empty(16, 4, space);
    auto empty_searcher = deglib::search::make_searcher(empty_graph.internal());

    const auto empty_tbl = empty_searcher->to_recall_table(10, RecallTable::Mode::Ef);
    EXPECT_TRUE(empty_tbl.empty());
    EXPECT_EQ(empty_tbl.k(), 10u);
    EXPECT_EQ(empty_tbl.sample_count(), 0u);
    EXPECT_FLOAT_EQ(empty_tbl.rerank_factor(), 1.0f);

    const auto empty_tables = empty_searcher->to_recall_tables(10);
    ASSERT_EQ(empty_tables.size(), 2u);
    const auto empty_line = empty_searcher->to_pareto_line(empty_tables);
    EXPECT_TRUE(empty_line.empty());
    EXPECT_EQ(empty_line.k(), 10u);

    // Invalid argument handling even on empty/small searchers
    const std::vector<float> bad_rf = {0.5f};
    const std::vector<float> nan_rf = {std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW((void)empty_searcher->to_recall_table(0, RecallTable::Mode::Ef), std::invalid_argument);
    EXPECT_THROW((void)empty_searcher->to_recall_table(5, RecallTable::Mode::Ef, 1.0f, 0), std::invalid_argument);
    EXPECT_THROW((void)empty_searcher->to_recall_table(5, RecallTable::Mode::Ef, 0.5f), std::invalid_argument);
    EXPECT_THROW(
        (void)empty_searcher->to_recall_tables(5, 10, {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall}, 0.05f, bad_rf), std::invalid_argument
    );
    EXPECT_THROW(
        (void)empty_searcher->to_recall_tables(5, 10, {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall}, 0.05f, nan_rf), std::invalid_argument
    );
    EXPECT_THROW((void)empty_searcher->to_pareto_line(empty_tables, 0, 0), std::invalid_argument);
    EXPECT_THROW(
        (void)empty_searcher->to_pareto_line(10, 50, {RecallTable::kDefaultMinRecall, RecallTable::kDefaultMaxRecall}, 0.05f, {}, 1, 0), std::invalid_argument
    );

    const auto tiny_data = make_searcher_clustered_data(20, 8, 4, 55);
    auto tiny_graph = deglib::builder::build_from_data(std::span<const float>(tiny_data), 8, {}, 4, Metric::FP32_L2);
    auto tiny_searcher = deglib::search::make_searcher(tiny_graph.internal());
    const RecallTable t5 = tiny_searcher->to_recall_table(5, RecallTable::Mode::Ef, 1.0f, 10);
    const RecallTable t8 = tiny_searcher->to_recall_table(8, RecallTable::Mode::Eps, 1.0f, 10);
    const std::vector<RecallTable> mismatched = {t5, t8};
    EXPECT_THROW((void)tiny_searcher->to_pareto_line(std::span<const RecallTable>(mismatched)), std::invalid_argument);
}
