// test_searcher.cpp — Unit tests for deglib::search::Searcher (searcher.h)

#include "deglib/builder.h"
#include "deglib/distances.h"
#include "deglib/graph.h"
#include "deglib/optimization.h"
#include "deglib/search/searcher.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
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

    // 1. Raw buffer search
    std::vector<uint32_t> out_indices(5);
    std::vector<float> out_distances(5);
    uint32_t found = searcher->search_f32(data.data(), 5, 0.1f, 1.0f, out_indices.data(), out_distances.data());

    EXPECT_EQ(found, 5u);
    EXPECT_EQ(out_indices[0], 0u);
    EXPECT_NEAR(out_distances[0], 0.0f, 1e-5f);
    for (size_t i = 1; i < 5; ++i) {
        EXPECT_GE(out_distances[i], out_distances[i - 1]);
    }

    // 2. Modern C++20 span / SearchResult search
    auto span_res = searcher->search(std::span<const float>(data.data(), dim), 5, 0.1f, 1.0f, /*return_distances=*/true);
    EXPECT_EQ(span_res.size(), 5u);
    EXPECT_EQ(span_res.indices[0], 0u);
    EXPECT_NEAR(span_res.distances[0], 0.0f, 1e-5f);
}

TEST(SearcherTest, QuantizedInt8WithFP16Refiner) {
    const uint32_t dim = 8;
    const uint32_t count = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    // Quantize to INT8
    auto quantizer = deglib::optimization::make_scalar_quantizer_int8(data.data(), count, dim);
    std::vector<int8_t> int8_data(count * dim);
    quantizer.quantize(data.data(), int8_data.data(), count, dim);

    auto target_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::Int8_L2);
    auto ro_graph = deglib::graph::convert_to_readonly_graph(base_graph.internal(), target_space, int8_data.data());

    // Refiner with FP16 base features
    std::vector<uint16_t> base_fp16(count * dim);
    deglib::distances::fp16::floats_to_fp16(data.data(), base_fp16.data(), data.size());
    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP16_L2);

    using RefinerT = deglib::search::ExactRefiner<uint16_t>;

    auto searcher = deglib::search::make_searcher(
        ro_graph,
        quantizer,
        RefinerT(rerank_space, base_fp16.data(), count)
    );

    // Query for 5th item
    auto q = std::span<const float>(data.data() + 5 * dim, dim);
    auto res = searcher->search(q, 5, /*eps=*/0.2f, /*rerank_factor=*/2.0f, /*return_distances=*/true);

    EXPECT_EQ(res.size(), 5u);
    EXPECT_EQ(res.indices[0], 5u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-3f);
}

TEST(SearcherTest, QuantizedUint8WithFP32Refiner) {
    const uint32_t dim = 8;
    const uint32_t count = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    std::vector<float> data(count * dim);
    for (auto& v : data) v = dist(rng);

    auto base_graph = deglib::builder::build_from_data(std::span<const float>(data), dim, {}, 8, deglib::distances::Metric::FP32_L2);

    auto quantizer = deglib::optimization::make_scalar_quantizer_uint8(data.data(), count, dim);
    std::vector<uint8_t> uint8_data(count * dim);
    quantizer.quantize(data.data(), uint8_data.data(), count, dim);

    auto target_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::Uint8_L2);
    auto ro_graph = deglib::graph::convert_to_readonly_graph(base_graph.internal(), target_space, uint8_data.data());

    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP32_L2);

    using RefinerT = deglib::search::ExactRefiner<float>;

    auto searcher = deglib::search::make_searcher(
        ro_graph,
        quantizer,
        RefinerT(rerank_space, data.data(), count)
    );

    auto q = std::span<const float>(data.data() + 2 * dim, dim);
    auto res = searcher->search(q, 3, /*eps=*/0.2f, /*rerank_factor=*/1.5f, /*return_distances=*/true);

    EXPECT_GE(res.size(), 1u);
    EXPECT_EQ(res.indices[0], 2u);
    EXPECT_NEAR(res.distances[0], 0.0f, 1e-4f);
}

TEST(SearcherTest, EVPQuantizerWithFP32Refiner) {
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

    deglib::quantization::evp::EvpQuantizer quantizer(non_zeros);
    auto evp_data = quantizer.quantize(data.data(), count, dim);
    auto target_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::EVP_InnerProduct);
    auto ro_graph = deglib::graph::convert_to_readonly_graph(base_graph.internal(), target_space, evp_data.data());

    auto rerank_space = deglib::distances::FloatSpace(dim, deglib::distances::Metric::FP32_InnerProduct);

    using QuantT = deglib::quantization::evp::EvpQuantizer;
    using RefinerT = deglib::search::ExactRefiner<float>;

    auto searcher = deglib::search::make_searcher(
        ro_graph,
        quantizer,
        RefinerT(rerank_space, data.data(), count)
    );

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

    // 1. search_batch returning SearchResultBatch (flat buffer)
    auto flat_res = searcher->search_batch(
        std::span<const float>(data.data(), n_queries * dim), n_queries, k,
        /*eps=*/0.1f, /*rerank_factor=*/1.0f,
        /*threads=*/2, /*return_distances=*/true
    );
    EXPECT_EQ(flat_res.size(), n_queries);
    EXPECT_EQ(flat_res.k, k);
    EXPECT_EQ(flat_res.indices.size(), n_queries * k);
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
        std::span<const float>(data.data(), n_queries * dim),
        n_queries, k,
        std::span<uint32_t>(preallocated_indices),
        std::span<float>(preallocated_dists),
        /*eps=*/0.1f, /*rerank_factor=*/1.0f,
        /*threads=*/1,
        /*return_distances=*/true
    );

    for (size_t q = 0; q < n_queries; ++q) {
        EXPECT_EQ(preallocated_indices[q * k], static_cast<uint32_t>(q));
        EXPECT_NEAR(preallocated_dists[q * k], 0.0f, 1e-5f);
    }
}

