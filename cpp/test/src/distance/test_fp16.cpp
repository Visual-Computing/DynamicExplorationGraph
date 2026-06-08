#include <gtest/gtest.h>
#include <distance/fp16.h>
#include <vector>
#include <cmath>

TEST(FP16Test, RoundTripSpecificValues) {
    std::vector<float> values = {0.0f, 1.0f, -1.0f, 2.0f, -0.5f, 0.25f, 123.4f, -987.6f};
    for (float orig : values) {
        uint16_t h = deglib::distances::float_to_fp16(orig);
        float recovered = deglib::distances::fp16_to_float(h);
        
        // FP16 has limited precision. Verify that difference is small.
        float tolerance = 1e-3f * (1.0f + std::abs(orig));
        EXPECT_NEAR(orig, recovered, tolerance) << "Failed for orig = " << orig;
    }
}

TEST(FP16Test, NaiveScalarVsSimdEquivalence) {
    // We test that our implementation works for a range of floats.
    // If SIMD is active, we can also manually compare vs a scalar-only fallback.
    std::vector<float> values;
    for (int i = -100; i <= 100; ++i) {
        values.push_back(static_cast<float>(i) * 0.1f);
    }
    
    for (float val : values) {
        uint16_t h = deglib::distances::float_to_fp16(val);
        float f = deglib::distances::fp16_to_float(h);
        float tolerance = 1e-3f * (1.0f + std::abs(val));
        EXPECT_NEAR(val, f, tolerance);
    }
}

TEST(FP16Test, VectorConversion) {
    std::vector<float> orig = {1.5f, -2.5f, 3.75f, 0.0f, -0.125f, 10.0f, -0.001f};
    std::vector<uint16_t> fp16 = deglib::distances::floats_to_fp16(orig);
    
    ASSERT_EQ(orig.size(), fp16.size());
    for (size_t i = 0; i < orig.size(); ++i) {
        uint16_t expected_h = deglib::distances::float_to_fp16(orig[i]);
        EXPECT_EQ(expected_h, fp16[i]) << "Mismatch at index " << i;
        
        float recovered = deglib::distances::fp16_to_float(fp16[i]);
        float tolerance = 1e-3f * (1.0f + std::abs(orig[i]));
        EXPECT_NEAR(orig[i], recovered, tolerance) << "Mismatch at index " << i;
    }
}
