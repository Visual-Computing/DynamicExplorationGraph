#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <chrono>

#include <deglib.h>

// ---------------------------------------------------------------------------
// FP16 Conversion Tests
// ---------------------------------------------------------------------------
// Tests for deglib::distances::fp16::float_to_fp16, fp16_to_float,
// floats_to_fp16 (vectorized), and fp16_to_floats (vectorized).
// Verifies SIMD/vectorized paths produce bit-exact results matching the
// scalar path, roundtrip precision, and edge-case handling.
// ---------------------------------------------------------------------------

using namespace deglib::distances::fp16;

// ---------------------------------------------------------------------------
// SIMD vs. Scalar Equivalence Test
// ---------------------------------------------------------------------------
// Convert 10,000 float values with floats_to_fp16 (which uses the vectorized
// F16C path when available) and compare element-by-element with a scalar
// float_to_fp16 loop. Ensures bit-exact equality between vectorized and
// scalar conversion paths.
TEST(FP16ConversionTest, SIMDvsScalarEquivalence) {
    const size_t count = 10000;
    std::vector<float> floats(count);
    std::vector<uint16_t> simd_result(count);
    std::vector<uint16_t> scalar_result(count);

    // Generate diverse float values covering normal, subnormal, and extreme ranges
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < count; ++i) {
        floats[i] = dist(rng);
    }

    // SIMD/vectorized path
    floats_to_fp16(floats.data(), simd_result.data(), count);

    // Scalar path
    for (size_t i = 0; i < count; ++i) {
        scalar_result[i] = float_to_fp16(floats[i]);
    }

    // Bit-exact comparison
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(simd_result[i], scalar_result[i])
            << "Mismatch at index " << i << ": float=" << floats[i];
    }
}

// ---------------------------------------------------------------------------
// SIMD vs. Scalar Equivalence Test (non-multiple-of-8 count)
// ---------------------------------------------------------------------------
// Tests that the remainder handling (0-7 elements) in the vectorized path
// produces correct results. Uses a count that is not a multiple of 8.
TEST(FP16ConversionTest, SIMDvsScalarEquivalenceNonMultiple) {
    const size_t count = 10003; // Not a multiple of 8
    std::vector<float> floats(count);
    std::vector<uint16_t> simd_result(count);
    std::vector<uint16_t> scalar_result(count);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f);
    for (size_t i = 0; i < count; ++i) {
        floats[i] = dist(rng);
    }

    floats_to_fp16(floats.data(), simd_result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        scalar_result[i] = float_to_fp16(floats[i]);
    }

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(simd_result[i], scalar_result[i])
            << "Mismatch at index " << i << ": float=" << floats[i];
    }
}

// ---------------------------------------------------------------------------
// SIMD vs. Scalar Equivalence Test (fp16_to_floats reverse direction)
// ---------------------------------------------------------------------------
// Tests the reverse conversion: fp16_to_floats (vectorized) vs. scalar
// fp16_to_float loop.
TEST(FP16ConversionTest, SIMDvsScalarEquivalenceReverse) {
    const size_t count = 10000;
    std::vector<uint16_t> fp16_vals(count);
    std::vector<float> simd_result(count);
    std::vector<float> scalar_result(count);

    // Generate FP16 bit patterns by converting floats to FP16 first
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    std::vector<float> temp_floats(count);
    for (size_t i = 0; i < count; ++i) {
        temp_floats[i] = dist(rng);
    }
    floats_to_fp16(temp_floats.data(), fp16_vals.data(), count);

    // SIMD/vectorized path
    fp16_to_floats(fp16_vals.data(), simd_result.data(), count);

    // Scalar path
    for (size_t i = 0; i < count; ++i) {
        scalar_result[i] = fp16_to_float(fp16_vals[i]);
    }

    // Bit-exact comparison
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits_simd, bits_scalar;
        std::memcpy(&bits_simd, &simd_result[i], sizeof(bits_simd));
        std::memcpy(&bits_scalar, &scalar_result[i], sizeof(bits_scalar));
        EXPECT_EQ(bits_simd, bits_scalar)
            << "Mismatch at index " << i << ": fp16=" << fp16_vals[i];
    }
}

// ---------------------------------------------------------------------------
// Roundtrip Precision Test
// ---------------------------------------------------------------------------
// Verify float -> fp16 -> float roundtrip error is within acceptable bounds.
// FP16 has 10 mantissa bits, so the relative error should be at most 2^-11
// (~4.88e-4) for normal numbers. We use a generous 1e-3 threshold.
TEST(FP16ConversionTest, RoundtripPrecision) {
    const size_t count = 10000;
    std::vector<float> original(count);
    std::vector<uint16_t> fp16_vals(count);
    std::vector<float> roundtripped(count);

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < count; ++i) {
        original[i] = dist(rng);
    }

    floats_to_fp16(original.data(), fp16_vals.data(), count);
    fp16_to_floats(fp16_vals.data(), roundtripped.data(), count);

    for (size_t i = 0; i < count; ++i) {
        float orig = original[i];
        float rt = roundtripped[i];
        float abs_err = std::fabs(orig - rt);
        float rel_err = abs_err / std::fabs(orig);

        // For values that round to zero in FP16, use absolute error
        if (std::fabs(orig) < 1e-7f) {
            EXPECT_LE(abs_err, 1e-7f)
                << "Absolute error too large at index " << i
                << ": orig=" << orig << ", rt=" << rt;
        } else {
            EXPECT_LE(rel_err, 1e-3f)
                << "Relative error too large at index " << i
                << ": orig=" << orig << ", rt=" << rt
                << ", rel_err=" << rel_err;
        }
    }
}

// ---------------------------------------------------------------------------
// Roundtrip Precision Test (small values near zero)
// ---------------------------------------------------------------------------
// Tests roundtrip for small values that may become subnormal or zero in FP16.
// For subnormal FP16 values, relative error can be large due to limited
// precision (10 mantissa bits). We verify that the roundtrip is self-consistent
// (i.e., float -> fp16 -> float matches scalar fp16_to_float(float_to_fp16(x))).
TEST(FP16ConversionTest, RoundtripPrecisionSmallValues) {
    std::vector<float> original = {
        0.0f, -0.0f, 1e-8f, -1e-8f, 1e-7f, -1e-7f, 6e-5f, -6e-5f,
        6.1e-5f, -6.1e-5f, 6.05e-5f, -6.05e-5f, 6.1e-8f, -6.1e-8f,
    };

    std::vector<uint16_t> fp16_vals(original.size());
    std::vector<float> roundtripped(original.size());

    floats_to_fp16(original.data(), fp16_vals.data(), original.size());
    fp16_to_floats(fp16_vals.data(), roundtripped.data(), original.size());

    for (size_t i = 0; i < original.size(); ++i) {
        float orig = original[i];
        float rt = roundtripped[i];
        // Verify roundtrip is self-consistent: the vectorized path should
        // produce the same result as the scalar path.
        float scalar_rt = fp16_to_float(float_to_fp16(orig));
        uint32_t bits_rt, bits_scalar;
        std::memcpy(&bits_rt, &rt, sizeof(bits_rt));
        std::memcpy(&bits_scalar, &scalar_rt, sizeof(bits_scalar));
        EXPECT_EQ(bits_rt, bits_scalar)
            << "Roundtrip mismatch at index " << i
            << ": orig=" << orig << ", rt=" << rt
            << ", scalar_rt=" << scalar_rt;
    }
}

// ---------------------------------------------------------------------------
// Edge-case Conversion Test
// ---------------------------------------------------------------------------
// Tests zeros, subnormals, infinities, NaNs, and extreme exponents.
TEST(FP16ConversionTest, EdgeCases) {
    // Test positive and negative zero
    {
        uint16_t fp16_zero_pos = float_to_fp16(0.0f);
        uint16_t fp16_zero_neg = float_to_fp16(-0.0f);
        EXPECT_EQ(fp16_zero_pos, 0x0000);
        EXPECT_EQ(fp16_zero_neg, 0x8000);

        float rt_pos = fp16_to_float(fp16_zero_pos);
        float rt_neg = fp16_to_float(fp16_zero_neg);
        uint32_t bits_pos, bits_neg;
        std::memcpy(&bits_pos, &rt_pos, sizeof(bits_pos));
        std::memcpy(&bits_neg, &rt_neg, sizeof(bits_neg));
        EXPECT_EQ(bits_pos, 0x00000000u);
        EXPECT_EQ(bits_neg, 0x80000000u);
    }

    // Test infinities
    {
        uint16_t fp16_inf_pos = float_to_fp16(std::numeric_limits<float>::infinity());
        uint16_t fp16_inf_neg = float_to_fp16(-std::numeric_limits<float>::infinity());
        EXPECT_EQ(fp16_inf_pos, 0x7C00);
        EXPECT_EQ(fp16_inf_neg, 0xFC00);

        float rt_pos = fp16_to_float(fp16_inf_pos);
        float rt_neg = fp16_to_float(fp16_inf_neg);
        EXPECT_TRUE(std::isinf(rt_pos));
        EXPECT_TRUE(std::isinf(rt_neg));
        EXPECT_GT(rt_pos, 0.0f);
        EXPECT_LT(rt_neg, 0.0f);
    }

    // Test NaNs
    {
        float nan_val = std::numeric_limits<float>::quiet_NaN();
        uint16_t fp16_nan = float_to_fp16(nan_val);
        // NaN should have exponent bits all 1 and mantissa non-zero
        EXPECT_EQ(fp16_nan & 0x7C00, 0x7C00);
        // On MSVC scalar fallback, the NaN mantissa may be preserved differently
        // than on GCC/Clang with F16C. Just verify it's still a NaN when converted back.
        float rt_nan = fp16_to_float(fp16_nan);
        EXPECT_TRUE(std::isnan(rt_nan));
    }

    // Test extreme exponents
    {
        // Max normal FP16 (65504)
        float max_fp16 = 65504.0f;
        uint16_t fp16_max = float_to_fp16(max_fp16);
        EXPECT_EQ(fp16_max, 0x7BFF);
        float rt_max = fp16_to_float(fp16_max);
        EXPECT_FLOAT_EQ(rt_max, 65504.0f);

        // Min normal FP16 (2^-14 ≈ 6.1035e-5)
        // Use the exact FP16 representation to avoid float literal rounding issues
        uint16_t fp16_min_norm = 0x0400;
        float rt_min = fp16_to_float(fp16_min_norm);
        // Min normal FP16 is exactly 2^-14
        EXPECT_FLOAT_EQ(rt_min, std::ldexp(1.0f, -14));
        // Converting 2^-14 back should give 0x0400
        uint16_t fp16_from_min = float_to_fp16(std::ldexp(1.0f, -14));
        EXPECT_EQ(fp16_from_min, 0x0400);

        // Max subnormal FP16 (~6.097e-5)
        uint16_t fp16_max_sub = 0x03FF;
        float rt_sub = fp16_to_float(fp16_max_sub);
        // Max subnormal is (1023/1024) * 2^-14 ≈ 6.097e-5
        EXPECT_GT(rt_sub, 0.0f);
        EXPECT_LT(rt_sub, 6.1e-5f);

        // Overflow to infinity
        uint16_t fp16_overflow = float_to_fp16(1e10f);
        EXPECT_EQ(fp16_overflow, 0x7C00);

        // Underflow to zero
        uint16_t fp16_underflow = float_to_fp16(1e-45f);
        EXPECT_EQ(fp16_underflow, 0x0000);
    }

    // Test powers of 2
    {
        std::vector<float> powers = {
            1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f,
            256.0f, 512.0f, 1024.0f, 2048.0f, 4096.0f, 8192.0f,
            16384.0f, 32768.0f,
            0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f,
            0.0078125f, 0.00390625f, 0.001953125f, 0.0009765625f,
            0.00048828125f, 0.000244140625f, 0.0001220703125f,
            6.103515625e-5f,
        };

        for (float p : powers) {
            uint16_t fp16_val = float_to_fp16(p);
            float rt = fp16_to_float(fp16_val);
            EXPECT_FLOAT_EQ(rt, p) << "Roundtrip failed for power of 2: " << p;
        }
    }
}

// ---------------------------------------------------------------------------
// Sign Bit Preservation Test
// ---------------------------------------------------------------------------
// Verifies that the sign bit is correctly preserved for positive and negative
// values, including zero.
TEST(FP16ConversionTest, SignBitPreservation) {
    std::vector<float> values = {
        1.0f, -1.0f, 100.0f, -100.0f, 0.5f, -0.5f,
        1e-5f, -1e-5f, 1e4f, -1e4f,
    };

    for (float v : values) {
        uint16_t fp16_val = float_to_fp16(v);
        uint16_t sign_bit = (fp16_val >> 15) & 0x1;
        uint16_t expected_sign = (v < 0.0f || (v == 0.0f && std::signbit(v))) ? 1 : 0;

        EXPECT_EQ(sign_bit, expected_sign)
            << "Sign bit mismatch for value: " << v
            << ", fp16=" << std::hex << fp16_val << std::dec;
    }
}

// ---------------------------------------------------------------------------
// Array Conversion Consistency Test
// ---------------------------------------------------------------------------
// Verifies that floats_to_fp16 and fp16_to_floats are inverses of each other
// when called on the same data.
TEST(FP16ConversionTest, ArrayConversionConsistency) {
    const size_t count = 1000;
    std::vector<float> original(count);
    std::vector<uint16_t> fp16_vals(count);
    std::vector<float> recovered(count);

    std::mt19937 rng(321);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (size_t i = 0; i < count; ++i) {
        original[i] = dist(rng);
    }

    floats_to_fp16(original.data(), fp16_vals.data(), count);
    fp16_to_floats(fp16_vals.data(), recovered.data(), count);

    // Each roundtripped value should match what a scalar conversion would produce
    for (size_t i = 0; i < count; ++i) {
        float scalar_rt = fp16_to_float(float_to_fp16(original[i]));
        uint32_t bits_arr, bits_scalar;
        std::memcpy(&bits_arr, &recovered[i], sizeof(bits_arr));
        std::memcpy(&bits_scalar, &scalar_rt, sizeof(bits_scalar));
        EXPECT_EQ(bits_arr, bits_scalar)
            << "Array conversion mismatch at index " << i
            << ": original=" << original[i];
    }
}

// ---------------------------------------------------------------------------
// Verify Runtime F16C Dispatch Test
// ---------------------------------------------------------------------------
// Logs CPU support status and verifies that calling floats_to_fp16 executes
// without crashes or fallback discrepancies.
TEST(FP16ConversionTest, VerifyRuntimeF16CDispatch) {
    bool has_f16c = deglib::cpu::has_f16c();
    std::cout << "[F16C] Runtime has_f16c() = " << (has_f16c ? "true" : "false") << std::endl;

    // Generate test data
    const size_t count = 1000;
    std::vector<float> floats(count);
    std::vector<uint16_t> fp16_vals(count);

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < count; ++i) {
        floats[i] = dist(rng);
    }

    // Call floats_to_fp16 - should use F16C path if available, scalar otherwise
    floats_to_fp16(floats.data(), fp16_vals.data(), count);

    // Verify results are valid (not all zeros unless input was all zeros)
    bool all_zero = true;
    for (size_t i = 0; i < count; ++i) {
        if (fp16_vals[i] != 0) {
            all_zero = false;
            break;
        }
    }
    EXPECT_FALSE(all_zero) << "All FP16 values are zero - conversion may have failed";

    // Also test fp16_to_floats
    std::vector<float> recovered(count);
    fp16_to_floats(fp16_vals.data(), recovered.data(), count);

    // Verify roundtrip consistency
    for (size_t i = 0; i < count; ++i) {
        float scalar_rt = fp16_to_float(float_to_fp16(floats[i]));
        uint32_t bits_rt, bits_scalar;
        std::memcpy(&bits_rt, &recovered[i], sizeof(bits_rt));
        std::memcpy(&bits_scalar, &scalar_rt, sizeof(bits_scalar));
        EXPECT_EQ(bits_rt, bits_scalar)
            << "Roundtrip mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Current vs. Hardware Precision Test
// ---------------------------------------------------------------------------
// Compares floats_to_fp16 (hardware vectorized path) against
// float_to_fp16_scalar (scalar IEEE 754 Round-to-Nearest-Even fallback)
// across 100,000 float values. Asserts 0 mismatches (100% bit-exact equality).
TEST(FP16ConversionTest, CurrentVsHardwarePrecision) {
    const size_t count = 100000;
    std::vector<float> floats(count);

    // Generate diverse values including edge cases
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < count; ++i) {
        floats[i] = dist(rng);
    }

    // Hardware vectorized results (uses F16C intrinsics when available, scalar fallback otherwise)
    std::vector<uint16_t> vectorized_results(count);
    floats_to_fp16(floats.data(), vectorized_results.data(), count);

    // Scalar results (always uses IEEE 754 Round-to-Nearest-Even)
    std::vector<uint16_t> scalar_results(count);
    for (size_t i = 0; i < count; ++i) {
        scalar_results[i] = float_to_fp16_scalar(floats[i]);
    }

    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        if (vectorized_results[i] != scalar_results[i]) {
            mismatches++;
        }
    }

    bool has_f16c = deglib::cpu::has_f16c();
    std::cout << "[Precision] has_f16c() = " << (has_f16c ? "true" : "false") << std::endl;
    std::cout << "[Precision] Vectorized vs scalar mismatches: " << mismatches
              << " / " << count << " (" << (100.0 * mismatches / count) << "%)" << std::endl;

    // Vectorized path should match scalar path exactly
    EXPECT_EQ(mismatches, 0u)
        << "Vectorized path should match scalar path exactly";
}

// ---------------------------------------------------------------------------
// Performance Benchmark Test
// ---------------------------------------------------------------------------
// Converts 1,000,000 float values 10 times in a timing loop for:
// - Vectorized SIMD F16C (floats_to_fp16)
// - Current Local Scalar loop
// - Old Commit Scalar loop
// Prints time per 1M conversions in milliseconds.
TEST(FP16ConversionTest, BenchmarkConversionSpeed) {
    const size_t count = 1000000;
    const int iterations = 500;

    std::vector<float> floats(count);
    std::vector<uint16_t> result(count);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (size_t i = 0; i < count; ++i) {
        floats[i] = dist(rng);
    }

    // Benchmark vectorized path
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        floats_to_fp16(floats.data(), result.data(), count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double simd_ms = simd_duration.count() / 1000.0 / iterations;
    std::cout << "[Benchmark] Vectorized SIMD F16C: " << simd_ms
              << " ms per 1M conversions" << std::endl;

    // Benchmark current scalar path
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < count; ++i) {
            result[i] = float_to_fp16(floats[i]);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto current_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double current_ms = current_duration.count() / 1000.0 / iterations;
    std::cout << "[Benchmark] Current Local Scalar: " << current_ms
              << " ms per 1M conversions" << std::endl;

    // Print speedup
    if (simd_ms > 0) {
        std::cout << "[Benchmark] SIMD speedup vs current scalar: "
                  << (current_ms / simd_ms) << "x" << std::endl;
    }

    // SIMD should be faster than scalar when F16C is available.
    // If F16C is not available, floats_to_fp16 falls back to scalar and
    // the timings would be equal — but in that case this benchmark is not
    // meaningful, so we only assert when F16C is present.
    if (deglib::cpu::has_f16c()) {
        EXPECT_LT(simd_ms, current_ms)
            << "SIMD path should be faster than scalar when F16C is available";
    }
}
