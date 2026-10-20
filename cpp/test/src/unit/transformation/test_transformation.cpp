#include <deglib/transformation.h>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <sstream>
#include <vector>

using namespace deglib::transformation;

TEST(TransformTest, QueryWorkspaceGrowth) {
    QueryWorkspace ws;

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
    mips->fit(data.data(), count);
    EXPECT_GT(mips->max_norm(), 0.0f);

    std::vector<std::byte> out_db(count * mips->output_bytes());
    mips->transform_database(data.data(), out_db.data(), count);

    const float* out_db_f = reinterpret_cast<const float*>(out_db.data());
    // All transformed vectors must have norm max_norm
    for (size_t i = 0; i < count; ++i) {
        double sum_sq = 0.0;
        for (uint32_t d = 0; d < dim + 1; ++d) {
            float v = out_db_f[i * (dim + 1) + d];
            sum_sq += v * v;
        }
        EXPECT_NEAR(std::sqrt(sum_sq), mips->max_norm(), 1e-4);
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
    evp->fit(data.data(), count);

    std::vector<std::byte> out_db(count * evp->output_bytes());
    evp->transform_database(data.data(), out_db.data(), count);

    // Serialization & Deserialization
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    evp->serialize(ss);

    auto evp_deser = EvpQuantizer::deserialize(ss);
    EXPECT_EQ(evp_deser->input_dim(), evp->input_dim());
    EXPECT_EQ(evp_deser->output_dim(), evp->output_dim());
    EXPECT_EQ(evp_deser->output_bytes(), evp->output_bytes());
    EXPECT_EQ(evp_deser->non_zeros, non_zeros);
    EXPECT_EQ(evp_deser->get_mean().size(), dim);
}

TEST(TransformTest, ScalarQuantizersRoundtrip) {
    const uint32_t dim = 32;
    const size_t count = 10;
    std::vector<float> data(count * dim);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i) * 0.1f - 5.0f;

    // INT8
    auto sqi8 = std::make_shared<ScalarQuantizerInt8>(dim, 10.0f);
    sqi8->fit(data.data(), count);
    std::stringstream ss_i8(std::ios::in | std::ios::out | std::ios::binary);
    sqi8->serialize(ss_i8);
    auto sqi8_deser = ScalarQuantizerInt8::deserialize(ss_i8);
    EXPECT_FLOAT_EQ(sqi8_deser->abs_max, sqi8->abs_max);
    EXPECT_FLOAT_EQ(sqi8_deser->scale, sqi8->scale);

    // UINT8
    auto squ8 = std::make_shared<ScalarQuantizerUint8>(dim, -5.0f, 15.0f);
    squ8->fit(data.data(), count);
    std::stringstream ss_u8(std::ios::in | std::ios::out | std::ios::binary);
    squ8->serialize(ss_u8);
    auto squ8_deser = ScalarQuantizerUint8::deserialize(ss_u8);
    EXPECT_FLOAT_EQ(squ8_deser->min_val, squ8->min_val);
    EXPECT_FLOAT_EQ(squ8_deser->max_val, squ8->max_val);
    EXPECT_FLOAT_EQ(squ8_deser->scale, squ8->scale);
}

TEST(TransformTest, VectorOfTransformsChainingAndSerialization) {
    const uint32_t in_dim = 32;
    const size_t count = 8;
    std::vector<float> data(count * in_dim);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i % 7) - 3.0f;

    // Step 1: MIPS (in_dim -> in_dim + 1 = 33)
    auto mips = std::make_shared<MipsTransform>(in_dim, 0.0f);
    mips->fit(data.data(), count);

    // Step 2: ScalarQuantizerInt8 (33 -> 33)
    auto sqi8 = std::make_shared<ScalarQuantizerInt8>(in_dim + 1, 5.0f);

    std::vector<std::shared_ptr<FeatureTransformation>> transformations = {mips, sqi8};

    EXPECT_EQ(transformations.size(), 2u);
    EXPECT_EQ(get_input_dim(transformations), in_dim);
    EXPECT_EQ(get_output_dim(transformations), in_dim + 1);
    EXPECT_EQ(get_output_bytes(transformations), in_dim + 1);

    // Transform vectors
    std::vector<std::byte> out_db(count * get_output_bytes(transformations));
    apply_database(transformations, data.data(), out_db.data(), count);

    std::vector<std::byte> out_q(count * get_output_bytes(transformations));
    apply_query(transformations, data.data(), out_q.data(), count);

    // Serialization & Deserialization
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    serialize_transformations(transformations, ss);

    auto transformations_deser = deserialize_transformations(ss);
    EXPECT_EQ(transformations_deser.size(), 2u);
    EXPECT_EQ(get_input_dim(transformations_deser), in_dim);
    EXPECT_EQ(get_output_dim(transformations_deser), in_dim + 1);
    EXPECT_EQ(get_output_bytes(transformations_deser), in_dim + 1);

    // Re-apply with deserialized transformations
    std::vector<std::byte> out_db_deser(count * get_output_bytes(transformations_deser));
    apply_database(transformations_deser, data.data(), out_db_deser.data(), count);
    EXPECT_EQ(out_db, out_db_deser);
}
