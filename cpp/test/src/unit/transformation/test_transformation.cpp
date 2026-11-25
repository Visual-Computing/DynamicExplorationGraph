#include <deglib/deglib.h>
#include <deglib/transformation.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <sstream>
#include <vector>

using namespace deglib::transformation;

TEST(TransformTest, TransformationWorkspaceGrowth) {
    TransformationWorkspace ws;

    auto* buf1 = ws.get_buffer_a(128);
    EXPECT_NE(buf1, nullptr);

    auto* buf2 = ws.get_buffer_a(64);
    EXPECT_EQ(buf1, buf2);

    auto* buf3 = ws.get_buffer_a(10000);
    EXPECT_NE(buf3, nullptr);

    auto* buf_b = ws.get_buffer_b(256);
    EXPECT_NE(buf_b, nullptr);
}

TEST(TransformTest, MipsTransformRoundtrip) {
    const uint32_t dim = 16;
    const size_t count = 10;
    std::vector<float> data(count * dim, 1.0f);

    auto mips = std::make_shared<MipsTransform>(dim, 0.0f);
    mips->fit(std::span<const float>(data));
    EXPECT_GT(mips->max_norm(), 0.0f);

    std::vector<std::byte> out_db(count * mips->output_byte_size());
    mips->transform_database(std::span<const float>(data), std::span<std::byte>(out_db));

    const float* out_db_f = reinterpret_cast<const float*>(out_db.data());
    // All transformed vectors must have unit norm
    for (size_t i = 0; i < count; ++i) {
        double sum_sq = 0.0;
        for (uint32_t d = 0; d < dim + 1; ++d) {
            float v = out_db_f[i * (dim + 1) + d];
            sum_sq += v * v;
        }
        EXPECT_NEAR(std::sqrt(sum_sq), 1.0, 1e-4);
    }

    // Serialization & Deserialization
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    mips->serialize(ss);

    auto mips_deser = MipsTransform::deserialize(ss);
    EXPECT_EQ(mips_deser->input_dim(), mips->input_dim());
    EXPECT_EQ(mips_deser->output_dim(), mips->output_dim());
    EXPECT_FLOAT_EQ(mips_deser->max_norm(), mips->max_norm());
}

TEST(TransformTest, EvpQuantizerRoundtrip) {
    const uint32_t dim = 64;
    const size_t count = 20;
    const uint32_t non_zeros = 16;
    std::vector<float> data(count * dim, 0.5f);

    auto evp = std::make_shared<EvpQuantizer>(dim, non_zeros);
    evp->fit(std::span<const float>(data));

    std::vector<std::byte> out_db(count * evp->output_byte_size());
    evp->transform_database(std::span<const float>(data), std::span<std::byte>(out_db));

    // Serialization & Deserialization
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    evp->serialize(ss);

    auto evp_deser = EvpQuantizer::deserialize(ss);
    EXPECT_EQ(evp_deser->input_dim(), evp->input_dim());
    EXPECT_EQ(evp_deser->output_dim(), evp->output_dim());
    EXPECT_EQ(evp_deser->output_byte_size(), evp->output_byte_size());
    EXPECT_EQ(evp_deser->non_zeros(), non_zeros);
    EXPECT_EQ(evp_deser->mean().size(), dim);
}

TEST(TransformTest, ScalarQuantizersRoundtrip) {
    const uint32_t dim = 32;
    const size_t count = 10;
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i) * 0.1f - 5.0f;

    // INT8
    auto sqi8 = std::make_shared<ScalarQuantizerInt8>(dim);
    sqi8->fit(std::span<const float>(data));
    std::stringstream ss_i8(std::ios::in | std::ios::out | std::ios::binary);
    sqi8->serialize(ss_i8);
    auto sqi8_deser = ScalarQuantizerInt8::deserialize(ss_i8);
    EXPECT_FLOAT_EQ(sqi8_deser->abs_max(), sqi8->abs_max());
    EXPECT_FLOAT_EQ(sqi8_deser->scale(), sqi8->scale());

    // UINT8
    auto squ8 = std::make_shared<ScalarQuantizerUint8>(dim);
    squ8->fit(std::span<const float>(data));
    std::stringstream ss_u8(std::ios::in | std::ios::out | std::ios::binary);
    squ8->serialize(ss_u8);
    auto squ8_deser = ScalarQuantizerUint8::deserialize(ss_u8);
    EXPECT_FLOAT_EQ(squ8_deser->min_val(), squ8->min_val());
    EXPECT_FLOAT_EQ(squ8_deser->max_val(), squ8->max_val());
    EXPECT_FLOAT_EQ(squ8_deser->scale(), squ8->scale());
}

TEST(TransformTest, VectorOfTransformsChainingAndSerialization) {
    const uint32_t in_dim = 32;
    const size_t count = 8;
    std::vector<float> data(count * in_dim);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i % 7) - 3.0f;

    // Step 1: MIPS (in_dim -> in_dim + 1 = 33)
    auto mips = std::make_shared<MipsTransform>(in_dim, 0.0f);
    mips->fit(std::span<const float>(data));

    // Step 2: ScalarQuantizerInt8 (33 -> 33)
    auto sqi8 = std::make_shared<ScalarQuantizerInt8>(in_dim + 1);
    std::vector<float> mips_data(count * (in_dim + 1));
    std::vector<std::byte> mips_bytes(mips_data.size() * sizeof(float));
    mips->transform_database(std::span<const float>(data), std::span<std::byte>(mips_bytes));
    std::memcpy(mips_data.data(), mips_bytes.data(), mips_bytes.size());
    sqi8->fit(std::span<const float>(mips_data));

    std::vector<std::shared_ptr<FeatureTransformation>> transformations = {mips, sqi8};

    EXPECT_EQ(transformations.size(), 2u);
    EXPECT_EQ(get_input_dim(transformations), in_dim);
    EXPECT_EQ(get_output_dim(transformations), in_dim + 1);
    EXPECT_EQ(get_output_byte_size(transformations), in_dim + 1);

    // Transform vectors
    std::vector<std::byte> out_db(count * get_output_byte_size(transformations));
    apply_on_database(transformations, std::span<const float>(data), std::span<std::byte>(out_db));

    std::vector<std::byte> out_q(count * get_output_byte_size(transformations));
    apply_on_query(transformations, std::span<const float>(data), std::span<std::byte>(out_q));

    // Serialization & Deserialization
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    serialize_transformations(transformations, ss);

    auto transformations_deser = deserialize_transformations(ss);
    EXPECT_EQ(transformations_deser.size(), 2u);
    EXPECT_EQ(get_input_dim(transformations_deser), in_dim);
    EXPECT_EQ(get_output_dim(transformations_deser), in_dim + 1);
    EXPECT_EQ(get_output_byte_size(transformations_deser), in_dim + 1);

    // Re-apply with deserialized transformations
    std::vector<std::byte> out_db_deser(count * get_output_byte_size(transformations_deser));
    apply_on_database(transformations_deser, std::span<const float>(data), std::span<std::byte>(out_db_deser));
    EXPECT_EQ(out_db, out_db_deser);
}

TEST(TransformTest, GraphAndRerankerChainPrefixTransformations) {
    const uint32_t in_dim = 8;
    const uint32_t count = 16;
    std::vector<float> raw_data(count * in_dim);
    for (size_t i = 0; i < raw_data.size(); ++i) raw_data[i] = static_cast<float>(i % 11) - 5.0f;

    auto mips = std::make_shared<MipsTransform>(in_dim, 0.0f);
    mips->fit(std::span<const float>(raw_data));

    std::vector<float> lifted(count * mips->output_dim());
    std::vector<std::byte> lifted_bytes(lifted.size() * sizeof(float));
    mips->transform_database(std::span<const float>(raw_data), std::span<std::byte>(lifted_bytes));
    std::memcpy(lifted.data(), lifted_bytes.data(), lifted_bytes.size());

    deglib::distances::FloatSpace l2_space(mips->output_dim(), deglib::distances::Metric::FP32_L2);
    auto graph = deglib::create_random_graph(lifted_bytes.data(), count, 4, l2_space, 7, {mips});

    auto sqi8 = std::make_shared<ScalarQuantizerInt8>(mips->output_dim());
    sqi8->fit(std::span<const float>(lifted));

    // Path A: with_transformations preserves each backend and chains [MipsTransform, ScalarQuantizerInt8]
    auto ro = graph.to_readonly().with_transformations({sqi8}, std::nullopt, 1);
    EXPECT_FALSE(ro.isMutable());
    ASSERT_EQ(ro.getTransformations().size(), 2u);
    EXPECT_EQ(ro.getTransformations()[0]->name(), "MipsTransform");
    EXPECT_EQ(ro.getTransformations()[1]->name(), "ScalarQuantizerInt8");

    auto mut = graph.with_transformations({sqi8});
    EXPECT_TRUE(mut.isMutable());
    EXPECT_NE(dynamic_cast<const deglib::graph::SizeBoundedGraph*>(&mut.internal()), nullptr);
    ASSERT_EQ(mut.getTransformations().size(), 2u);
    EXPECT_EQ(mut.getTransformations()[1]->name(), "ScalarQuantizerInt8");

    auto dyn = graph.to_dynamic(64).with_transformations({sqi8});
    EXPECT_TRUE(dyn.isMutable());
    const auto* dyn_internal = dynamic_cast<const deglib::graph::DynamicGraph*>(&dyn.internal());
    ASSERT_NE(dyn_internal, nullptr);
    EXPECT_EQ(dyn_internal->chunk_capacity(), 64u);
    ASSERT_EQ(dyn.getTransformations().size(), 2u);
    EXPECT_EQ(dyn.getTransformations()[1]->name(), "ScalarQuantizerInt8");

    // Path B: with_features resets prefix transformations and preserves backend
    auto ro_reset = graph.to_readonly().with_features(l2_space, lifted.data());
    EXPECT_FALSE(ro_reset.isMutable());
    EXPECT_TRUE(ro_reset.getTransformations().empty());
}
