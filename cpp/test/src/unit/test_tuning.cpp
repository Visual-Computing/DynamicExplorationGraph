// test_tuning.cpp — Unit tests for deglib::tuning base-only dataset profiling and target prediction

#include "deglib/deglib.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_random_vectors(size_t n, uint32_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(n * static_cast<size_t>(dim));
    for (float& v : data) {
        v = dist(rng);
    }
    return data;
}

double brute_force_participation_ratio(const std::vector<float>& data, size_t n, uint32_t dim) {
    std::vector<double> mean(dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            mean[j] += static_cast<double>(data[i * dim + j]);
        }
    }
    for (uint32_t j = 0; j < dim; ++j) {
        mean[j] /= static_cast<double>(n);
    }

    std::vector<double> cov(static_cast<size_t>(dim) * dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            const double yj = static_cast<double>(data[i * dim + j]) - mean[j];
            for (uint32_t k = 0; k < dim; ++k) {
                const double yk = static_cast<double>(data[i * dim + k]) - mean[k];
                cov[static_cast<size_t>(j) * dim + k] += yj * yk;
            }
        }
    }

    double tr = 0.0;
    double frob2 = 0.0;
    for (uint32_t j = 0; j < dim; ++j) {
        tr += cov[static_cast<size_t>(j) * dim + j];
        for (uint32_t k = 0; k < dim; ++k) {
            const double s = cov[static_cast<size_t>(j) * dim + k];
            frob2 += s * s;
        }
    }
    return (tr * tr) / (static_cast<double>(dim) * frob2);
}

double brute_force_pp_iqr(const std::vector<float>& data, size_t n, uint32_t dim) {
    std::vector<double> pp;
    pp.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double min1 = (std::numeric_limits<double>::infinity)();
        double min2 = (std::numeric_limits<double>::infinity)();
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double d2 = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                const double diff = static_cast<double>(data[i * dim + d]) - static_cast<double>(data[j * dim + d]);
                d2 += diff * diff;
            }
            if (d2 > static_cast<double>(deglib::tuning::kDuplicateDistSqThreshold)) {
                if (d2 < min1) {
                    min2 = min1;
                    min1 = d2;
                } else if (d2 < min2) {
                    min2 = d2;
                }
            }
        }
        if (std::isfinite(min2) && min1 > 0.0 && min2 > min1) {
            pp.push_back(2.0 / std::log(min2 / min1));
        }
    }
    std::sort(pp.begin(), pp.end());
    auto pct = [&](double q) {
        const double pos = q * static_cast<double>(pp.size() - 1);
        const size_t lo = static_cast<size_t>(pos);
        const size_t hi = (std::min)(lo + 1, pp.size() - 1);
        const double frac = pos - static_cast<double>(lo);
        return pp[lo] + frac * (pp[hi] - pp[lo]);
    };
    return pct(0.75) - pct(0.25);
}

TEST(TuningTest, ParticipationRatioAnalytical) {
    // 1. Rank-1 data along axis 0 -> participation_ratio == 1 / dim.
    {
        constexpr size_t n = 100;
        constexpr uint32_t dim = 16;
        std::vector<float> rank1(n * dim, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            rank1[i * dim] = static_cast<float>(i) * 0.5f - 10.0f;
        }
        const auto profile = deglib::tuning::measure_dataset_profile(rank1, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
        EXPECT_NEAR(profile.participation_ratio, 1.0f / static_cast<float>(dim), 1e-4f);
    }

    // 2. Orthogonal equal-variance rank-k subspace (k = 8, dim = 32) via Walsh characters -> 8 / 32 = 0.25.
    {
        constexpr size_t n = 64;
        constexpr uint32_t dim = 32;
        constexpr uint32_t rank = 8;
        std::vector<float> sub(n * dim, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < rank; ++j) {
                const unsigned bits = static_cast<unsigned>(i) & static_cast<unsigned>(j + 1);
                sub[i * dim + j] = (std::popcount(bits) & 1) ? -1.0f : 1.0f;
            }
        }
        const auto profile = deglib::tuning::measure_dataset_profile(sub, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
        EXPECT_NEAR(profile.participation_ratio, static_cast<float>(rank) / static_cast<float>(dim), 1e-4f);
    }

    // 3. Both n >= dim (200 x 32) and n < dim (40 x 128) match brute-force covariance Frobenius calculation.
    {
        const auto tall = make_random_vectors(200, 32, 11);
        const auto p_tall = deglib::tuning::measure_dataset_profile(tall, 32, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
        const double bf_tall = brute_force_participation_ratio(tall, 200, 32);
        EXPECT_NEAR(p_tall.participation_ratio, static_cast<float>(bf_tall), 1e-4f);

        const auto wide = make_random_vectors(40, 128, 22);
        const auto p_wide = deglib::tuning::measure_dataset_profile(wide, 128, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
        const double bf_wide = brute_force_participation_ratio(wide, 40, 128);
        EXPECT_NEAR(p_wide.participation_ratio, static_cast<float>(bf_wide), 1e-4f);
    }
}

TEST(TuningTest, ScaleInvarianceL2) {
    constexpr size_t n = 250;
    constexpr uint32_t dim = 24;
    const auto base = make_random_vectors(n, dim, 123);

    std::vector<float> tiny(base.size());
    std::vector<float> huge(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        tiny[i] = base[i] * 1e-3f;
        huge[i] = base[i] * 1e+4f;
    }

    const auto p_base = deglib::tuning::predict_optimization_target(base, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
    const auto p_tiny = deglib::tuning::predict_optimization_target(tiny, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
    const auto p_huge = deglib::tuning::predict_optimization_target(huge, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);

    EXPECT_NEAR(p_tiny.profile.participation_ratio, p_base.profile.participation_ratio, 1e-4f);
    EXPECT_NEAR(p_tiny.profile.rel_contrast_1, p_base.profile.rel_contrast_1, 1e-4f);
    EXPECT_NEAR(p_tiny.profile.pp_iqr, p_base.profile.pp_iqr, 1e-2f);
    EXPECT_NEAR(p_tiny.profile.d16_skew, p_base.profile.d16_skew, 1e-4f);
    EXPECT_NEAR(p_tiny.profile.ball_cnt_p90_p10_r15, p_base.profile.ball_cnt_p90_p10_r15, 1e-4f);
    EXPECT_NEAR(p_tiny.profile.hadamard_kurt, p_base.profile.hadamard_kurt, 1e-4f);
    EXPECT_NEAR(p_tiny.lowlid_advantage_pct, p_base.lowlid_advantage_pct, 1e-3f);

    EXPECT_NEAR(p_huge.profile.participation_ratio, p_base.profile.participation_ratio, 1e-4f);
    EXPECT_NEAR(p_huge.profile.rel_contrast_1, p_base.profile.rel_contrast_1, 1e-4f);
    EXPECT_NEAR(p_huge.profile.pp_iqr, p_base.profile.pp_iqr, 1e-2f);
    EXPECT_NEAR(p_huge.profile.d16_skew, p_base.profile.d16_skew, 1e-4f);
    EXPECT_NEAR(p_huge.profile.ball_cnt_p90_p10_r15, p_base.profile.ball_cnt_p90_p10_r15, 1e-4f);
    EXPECT_NEAR(p_huge.profile.hadamard_kurt, p_base.profile.hadamard_kurt, 1e-4f);
    EXPECT_NEAR(p_huge.lowlid_advantage_pct, p_base.lowlid_advantage_pct, 1e-3f);
}

TEST(TuningTest, InnerProductNormalization) {
    constexpr size_t n = 250;
    constexpr uint32_t dim = 16;
    const auto base = make_random_vectors(n, dim, 33);

    std::mt19937 rng(44);
    std::uniform_real_distribution<float> scale_dist(0.1f, 20.0f);
    std::vector<float> scaled = base;
    for (size_t i = 0; i < n; ++i) {
        const float s = scale_dist(rng);
        for (uint32_t j = 0; j < dim; ++j) {
            scaled[i * dim + j] *= s;
        }
    }

    const auto ip_base = deglib::tuning::measure_dataset_profile(base, dim, deglib::distances::Metric::FP32_InnerProduct, 0, 0, 7, 2);
    const auto ip_scaled = deglib::tuning::measure_dataset_profile(scaled, dim, deglib::distances::Metric::FP32_InnerProduct, 0, 0, 7, 2);
    EXPECT_NEAR(ip_scaled.participation_ratio, ip_base.participation_ratio, 1e-5f);
    EXPECT_NEAR(ip_scaled.rel_contrast_1, ip_base.rel_contrast_1, 1e-5f);
    EXPECT_NEAR(ip_scaled.pp_iqr, ip_base.pp_iqr, 1e-2f);
    EXPECT_NEAR(ip_scaled.d16_skew, ip_base.d16_skew, 1e-4f);
    EXPECT_NEAR(ip_scaled.ball_cnt_p90_p10_r15, ip_base.ball_cnt_p90_p10_r15, 1e-4f);
    EXPECT_NEAR(ip_scaled.hadamard_kurt, ip_base.hadamard_kurt, 1e-4f);

    const auto l2_base = deglib::tuning::measure_dataset_profile(base, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
    const auto l2_scaled = deglib::tuning::measure_dataset_profile(scaled, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
    EXPECT_GT(std::abs(l2_scaled.pp_iqr - l2_base.pp_iqr) + std::abs(l2_scaled.rel_contrast_1 - l2_base.rel_contrast_1), 0.01f);
}

TEST(TuningTest, DuplicateResilience) {
    constexpr size_t n = 300;
    constexpr uint32_t dim = 16;
    const auto clean = make_random_vectors(n, dim, 55);

    std::vector<float> with_dups = clean;
    constexpr size_t dup_rows = 15;  // 5% duplicates
    with_dups.insert(with_dups.end(), clean.begin(), clean.begin() + dup_rows * dim);

    const auto p_clean = deglib::tuning::measure_dataset_profile(clean, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);
    const auto p_dups = deglib::tuning::measure_dataset_profile(with_dups, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 2);

    EXPECT_TRUE(std::isfinite(p_dups.participation_ratio));
    EXPECT_TRUE(std::isfinite(p_dups.pp_iqr));
    EXPECT_TRUE(std::isfinite(p_dups.rel_contrast_1));
    EXPECT_NEAR(p_dups.pp_iqr, p_clean.pp_iqr, 10.0f);
    EXPECT_NEAR(p_dups.d16_skew, p_clean.d16_skew, 0.5f);
    EXPECT_NEAR(p_dups.ball_cnt_p90_p10_r15, p_clean.ball_cnt_p90_p10_r15, 1.0f);
    EXPECT_NEAR(p_dups.hadamard_kurt, p_clean.hadamard_kurt, 0.05f);
    EXPECT_NEAR(p_dups.rel_contrast_1, p_clean.rel_contrast_1, 0.05f);
}

TEST(TuningTest, FP16MatchesFP32) {
    constexpr size_t n = 300;
    constexpr uint32_t dim = 32;
    const auto f32 = make_random_vectors(n, dim, 77);
    std::vector<uint16_t> f16(f32.size());
    deglib::distances::fp16::floats_to_fp16(f32.data(), f16.data(), f32.size());

    const auto p32 = deglib::tuning::predict_optimization_target(f32, dim, deglib::distances::Metric::FP32_L2, 200, 150, 7, 2);
    const auto p16 = deglib::tuning::predict_optimization_target(std::span<const uint16_t>(f16), dim, deglib::distances::Metric::FP32_L2, 200, 150, 7, 2);

    EXPECT_EQ(p16.profile.sample_count, p32.profile.sample_count);
    EXPECT_EQ(p16.profile.dim, p32.profile.dim);
    EXPECT_NEAR(p16.profile.participation_ratio, p32.profile.participation_ratio, 0.01f);
    EXPECT_NEAR(p16.profile.rel_contrast_1, p32.profile.rel_contrast_1, 0.01f);
    EXPECT_NEAR(p16.profile.pp_iqr, p32.profile.pp_iqr, 0.25f);
    EXPECT_NEAR(p16.profile.d16_skew, p32.profile.d16_skew, 0.05f);
    EXPECT_NEAR(p16.profile.ball_cnt_p90_p10_r15, p32.profile.ball_cnt_p90_p10_r15, 0.1f);
    EXPECT_NEAR(p16.profile.hadamard_kurt, p32.profile.hadamard_kurt, 0.02f);
    EXPECT_NEAR(p16.lowlid_advantage_pct, p32.lowlid_advantage_pct, 0.5f);
}

TEST(TuningTest, ThreadInvarianceAndDeterminism) {
    constexpr size_t n = 400;
    constexpr uint32_t dim = 48;
    const auto data = make_random_vectors(n, dim, 88);

    const auto r1 = deglib::tuning::predict_optimization_target(data, dim, deglib::distances::Metric::FP32_InnerProduct, 300, 200, 7, 1);
    const auto r4 = deglib::tuning::predict_optimization_target(data, dim, deglib::distances::Metric::FP32_InnerProduct, 300, 200, 7, 4);

    EXPECT_EQ(r1.target, r4.target);
    EXPECT_NEAR(r1.profile.participation_ratio, r4.profile.participation_ratio, 1e-5f);
    EXPECT_NEAR(r1.profile.rel_contrast_1, r4.profile.rel_contrast_1, 1e-5f);
    EXPECT_NEAR(r1.profile.pp_iqr, r4.profile.pp_iqr, 1e-2f);
    EXPECT_NEAR(r1.profile.d16_skew, r4.profile.d16_skew, 1e-4f);
    EXPECT_NEAR(r1.profile.ball_cnt_p90_p10_r15, r4.profile.ball_cnt_p90_p10_r15, 1e-4f);
    EXPECT_NEAR(r1.profile.hadamard_kurt, r4.profile.hadamard_kurt, 1e-4f);
    EXPECT_NEAR(r1.lowlid_advantage_pct, r4.lowlid_advantage_pct, 1e-4f);
}

TEST(TuningTest, EdgeCasesAndValidation) {
    const std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW((void)deglib::tuning::measure_dataset_profile(vec, 0), std::invalid_argument);
    EXPECT_THROW((void)deglib::tuning::measure_dataset_profile(vec, 3), std::invalid_argument);
    const std::vector<uint16_t> vec16 = {0x3c00, 0x3c00, 0x3c00, 0x3c00};
    EXPECT_THROW((void)deglib::tuning::measure_dataset_profile(std::span<const uint16_t>(vec16), 0), std::invalid_argument);
    EXPECT_THROW((void)deglib::tuning::measure_dataset_profile(std::span<const uint16_t>(vec16), 3), std::invalid_argument);

    // Empty (0 rows)
    const std::vector<float> empty;
    const auto p_empty = deglib::tuning::predict_optimization_target(empty, 8);
    EXPECT_EQ(p_empty.target, deglib::builder::OptimizationTarget::StreamingData);
    EXPECT_FLOAT_EQ(p_empty.lowlid_advantage_pct, 0.0f);
    EXPECT_EQ(p_empty.profile.sample_count, 0u);

    // Single vector (1 row)
    const std::vector<float> single(8, 1.0f);
    const auto p_single = deglib::tuning::predict_optimization_target(single, 8);
    EXPECT_EQ(p_single.target, deglib::builder::OptimizationTarget::StreamingData);
    EXPECT_FLOAT_EQ(p_single.lowlid_advantage_pct, 0.0f);
    EXPECT_EQ(p_single.profile.sample_count, 1u);

    // All identical vectors
    const std::vector<float> identical(64 * 8, 3.5f);
    const auto p_ident = deglib::tuning::predict_optimization_target(identical, 8);
    EXPECT_EQ(p_ident.target, deglib::builder::OptimizationTarget::StreamingData);
    EXPECT_FLOAT_EQ(p_ident.lowlid_advantage_pct, 0.0f);
    EXPECT_TRUE(std::isfinite(p_ident.profile.participation_ratio));
    EXPECT_TRUE(std::isfinite(p_ident.profile.pp_iqr));
    EXPECT_TRUE(std::isfinite(p_ident.profile.rel_contrast_1));
    EXPECT_TRUE(std::isfinite(p_ident.profile.d16_skew));
    EXPECT_TRUE(std::isfinite(p_ident.profile.ball_cnt_p90_p10_r15));
    EXPECT_TRUE(std::isfinite(p_ident.profile.hadamard_kurt));
}

TEST(TuningTest, TiledMatchesBruteForceNonAligned) {
    constexpr size_t n = 137;
    constexpr uint32_t dim = 300;
    const auto data = make_random_vectors(n, dim, 99);

    const auto profile = deglib::tuning::measure_dataset_profile(data, dim, deglib::distances::Metric::FP32_L2, 0, 0, 7, 4);
    const double bf_pr = brute_force_participation_ratio(data, n, dim);
    const double bf_pp = brute_force_pp_iqr(data, n, dim);
    EXPECT_NEAR(profile.participation_ratio, static_cast<float>(bf_pr), 1e-4f);
    EXPECT_NEAR(profile.pp_iqr, static_cast<float>(bf_pp), 0.05f);

    const auto sub_1 = deglib::tuning::measure_dataset_profile(data, dim, deglib::distances::Metric::FP32_L2, 0, 50, 7, 1);
    const auto sub_4 = deglib::tuning::measure_dataset_profile(data, dim, deglib::distances::Metric::FP32_L2, 0, 50, 7, 4);
    EXPECT_NEAR(sub_1.participation_ratio, sub_4.participation_ratio, 1e-5f);
    EXPECT_NEAR(sub_1.pp_iqr, sub_4.pp_iqr, 1e-2f);
    EXPECT_NEAR(sub_1.d16_skew, sub_4.d16_skew, 1e-4f);
    EXPECT_NEAR(sub_1.ball_cnt_p90_p10_r15, sub_4.ball_cnt_p90_p10_r15, 1e-4f);
    EXPECT_NEAR(sub_1.hadamard_kurt, sub_4.hadamard_kurt, 1e-4f);
    EXPECT_NEAR(sub_1.rel_contrast_1, sub_4.rel_contrast_1, 1e-5f);
}

TEST(TuningTest, VibeReferenceCalibration) {
    struct VibeRef {
        const char* name;
        deglib::tuning::DatasetProfile profile;
        deglib::builder::OptimizationTarget expected_target;
        float expected_adv_pct;
        deglib::distances::Metric expected_quant;
        float expected_uint7_adv_pct;
    };

    const VibeRef refs[] = {
        {"laion-clip",
         {16000, 512, 0.174291f, 1.907535f, 68.90f, -0.062f, 1.708f, 0.1052f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::LowLID,
         6.70f,
         deglib::distances::Metric::Int8_InnerProduct,
         -2.22f},
        {"yandex",
         {16000, 200, 0.551745f, 1.963387f, 49.41f, -1.655f, 1.013f, -0.0233f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::LowLID,
         9.00f,
         deglib::distances::Metric::Int8_InnerProduct,
         -3.55f},
        {"imagenet-align",
         {16000, 640, 0.214643f, 2.047639f, 54.86f, -0.128f, 1.018f, 0.0378f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::LowLID,
         4.92f,
         deglib::distances::Metric::Int8_InnerProduct,
         -3.79f},
        {"gooaq-distilroberta",
         {16000, 768, 0.287027f, 1.592527f, 56.24f, -0.845f, 1.000f, -0.0272f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::LowLID,
         2.02f,
         deglib::distances::Metric::Uint7_L2,
         1.33f},
        {"yahoo-minilm",
         {16000, 384, 0.390076f, 1.578273f, 67.51f, -0.758f, 1.000f, -0.0395f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::StreamingData,
         -3.97f,
         deglib::distances::Metric::Uint7_L2,
         1.02f},
        {"agnews-mxbai",
         {16000, 1024, 0.121258f, 1.614592f, 65.92f, -0.184f, 1.002f, -0.0025f, deglib::distances::Metric::FP32_L2},
         deglib::builder::OptimizationTarget::StreamingData,
         -3.25f,
         deglib::distances::Metric::Uint7_L2,
         0.0f},
        {"ccnews-nomic",
         {16000, 768, 0.149226f, 1.730994f, 75.12f, -0.924f, 1.002f, -0.0178f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::StreamingData,
         -3.05f,
         deglib::distances::Metric::Uint7_L2,
         0.38f},
        {"landmark-dino",
         {16000, 768, 0.120246f, 1.728638f, 88.61f, -0.451f, 1.001f, -0.0266f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::StreamingData,
         -8.92f,
         deglib::distances::Metric::Uint7_L2,
         0.79f},
        {"arxiv-nomic",
         {16000, 768, 0.081903f, 1.670677f, 72.52f, -0.057f, 1.164f, -0.0239f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::StreamingData,
         -7.53f,
         deglib::distances::Metric::Uint7_L2,
         2.33f},
        {"inaturalist-resnet",
         {16000, 2048, 0.033901f, 2.115867f, 64.16f, 0.063f, 10.651f, -0.1809f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::StreamingData,
         -11.07f,
         deglib::distances::Metric::Uint7_L2,
         12.14f},
        {"celeba-resnet",
         {16000, 2048, 0.010431f, 2.179041f, 82.63f, 0.594f, 22.169f, 0.0753f, deglib::distances::Metric::FP32_InnerProduct},
         deglib::builder::OptimizationTarget::StreamingData,
         -9.89f,
         deglib::distances::Metric::Uint7_L2,
         63.02f},
    };

    std::vector<float> preds;
    std::vector<float> qpreds;
    for (const auto& ref : refs) {
        const auto pred = deglib::tuning::predict_optimization_target(ref.profile);
        EXPECT_EQ(pred.target, ref.expected_target) << ref.name;
        EXPECT_NEAR(pred.lowlid_advantage_pct, ref.expected_adv_pct, 0.25f) << ref.name;
        preds.push_back(pred.lowlid_advantage_pct);

        const auto qpred = deglib::tuning::predict_quantization(ref.profile);
        EXPECT_EQ(qpred.metric, ref.expected_quant) << ref.name;
        EXPECT_NEAR(qpred.uint7_advantage_pct, ref.expected_uint7_adv_pct, 0.20f) << ref.name;
        qpreds.push_back(qpred.uint7_advantage_pct);
    }

    EXPECT_GT(preds[1], preds[0]);   // yandex > laion-clip
    EXPECT_GT(preds[0], preds[2]);   // laion-clip > imagenet-align
    EXPECT_GT(preds[2], preds[3]);   // imagenet-align > gooaq
    EXPECT_GT(preds[3], 0.0f);       // gooaq > 0
    EXPECT_GT(preds[5], preds[9]);   // agnews-mxbai > inaturalist-resnet
    EXPECT_GT(preds[5], preds[10]);  // agnews-mxbai > celeba-resnet

    // Quantization advantage ordering across InnerProduct datasets:
    // celeba > inaturalist > arxiv > gooaq > yahoo > landmark > ccnews > 0 > laion > yandex > imagenet
    EXPECT_GT(qpreds[10], qpreds[9]);
    EXPECT_GT(qpreds[9], qpreds[8]);
    EXPECT_GT(qpreds[8], qpreds[3]);
    EXPECT_GT(qpreds[3], qpreds[4]);
    EXPECT_GT(qpreds[4], qpreds[7]);
    EXPECT_GT(qpreds[7], qpreds[6]);
    EXPECT_GT(qpreds[6], 0.0f);
    EXPECT_GT(0.0f, qpreds[0]);
    EXPECT_GT(qpreds[0], qpreds[1]);
    EXPECT_GT(qpreds[1], qpreds[2]);
}

TEST(TuningTest, QuantizationL2RuleAndInnerProductInvariance) {
    // Even an imagenet-align-shaped geometry falls back to Uint7_L2 when metric is L2.
    const deglib::tuning::DatasetProfile l2_fp32_profile{16000,  640,    0.214643f, 2.047639f,
                                                         54.86f, -0.128f, 1.018f,   0.0378f, deglib::distances::Metric::FP32_L2};
    const auto q_l2_fp32 = deglib::tuning::predict_quantization(l2_fp32_profile);
    EXPECT_EQ(q_l2_fp32.metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_FLOAT_EQ(q_l2_fp32.uint7_advantage_pct, 0.0f);

    const deglib::tuning::DatasetProfile l2_fp16_profile{16000,  640,    0.214643f, 2.047639f,
                                                         54.86f, -0.128f, 1.018f,   0.0378f, deglib::distances::Metric::FP16_L2};
    const auto q_l2_fp16 = deglib::tuning::predict_quantization(l2_fp16_profile);
    EXPECT_EQ(q_l2_fp16.metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_FLOAT_EQ(q_l2_fp16.uint7_advantage_pct, 0.0f);

    // Per-row positive rescaling on InnerProduct vectors preserves quantization prediction.
    constexpr size_t n = 250;
    constexpr uint32_t dim = 16;
    const auto base = make_random_vectors(n, dim, 101);
    std::mt19937 rng(102);
    std::uniform_real_distribution<float> scale_dist(0.1f, 20.0f);
    std::vector<float> scaled = base;
    for (size_t i = 0; i < n; ++i) {
        const float s = scale_dist(rng);
        for (uint32_t j = 0; j < dim; ++j) {
            scaled[i * dim + j] *= s;
        }
    }

    const auto q_base = deglib::tuning::predict_quantization(base, dim, deglib::distances::Metric::FP32_InnerProduct, 0, 0, 7, 2);
    const auto q_scaled = deglib::tuning::predict_quantization(scaled, dim, deglib::distances::Metric::FP32_InnerProduct, 0, 0, 7, 2);
    EXPECT_EQ(q_base.profile.metric, deglib::distances::Metric::FP32_InnerProduct);
    EXPECT_EQ(q_scaled.profile.metric, deglib::distances::Metric::FP32_InnerProduct);
    EXPECT_EQ(q_scaled.metric, q_base.metric);
    EXPECT_NEAR(q_scaled.uint7_advantage_pct, q_base.uint7_advantage_pct, 1e-4f);
}

TEST(TuningTest, QuantizationFP16DeterminismAndProfileReuse) {
    constexpr size_t n = 300;
    constexpr uint32_t dim = 32;
    const auto f32 = make_random_vectors(n, dim, 177);
    std::vector<uint16_t> f16(f32.size());
    deglib::distances::fp16::floats_to_fp16(f32.data(), f16.data(), f32.size());

    const auto q32 = deglib::tuning::predict_quantization(f32, dim, deglib::distances::Metric::FP32_InnerProduct, 200, 150, 7, 2);
    const auto q16 = deglib::tuning::predict_quantization(std::span<const uint16_t>(f16), dim, deglib::distances::Metric::FP16_InnerProduct, 200, 150, 7, 2);
    EXPECT_EQ(q16.profile.metric, deglib::distances::Metric::FP16_InnerProduct);
    EXPECT_EQ(q16.metric, q32.metric);
    EXPECT_NEAR(q16.uint7_advantage_pct, q32.uint7_advantage_pct, 0.2f);

    const auto q_t1 = deglib::tuning::predict_quantization(f32, dim, deglib::distances::Metric::FP32_InnerProduct, 200, 150, 7, 1);
    const auto q_t4 = deglib::tuning::predict_quantization(f32, dim, deglib::distances::Metric::FP32_InnerProduct, 200, 150, 7, 4);
    EXPECT_EQ(q_t1.metric, q_t4.metric);
    EXPECT_NEAR(q_t1.uint7_advantage_pct, q_t4.uint7_advantage_pct, 1e-4f);

    const auto prof = deglib::tuning::measure_dataset_profile(f32, dim, deglib::distances::Metric::FP32_InnerProduct, 200, 150, 7, 2);
    const auto q_reuse = deglib::tuning::predict_quantization(prof);
    EXPECT_EQ(q_reuse.metric, q32.metric);
    EXPECT_FLOAT_EQ(q_reuse.uint7_advantage_pct, q32.uint7_advantage_pct);
}

TEST(TuningTest, QuantizationEdgeCasesAndCustomCalibration) {
    const std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW((void)deglib::tuning::predict_quantization(vec, 0), std::invalid_argument);
    EXPECT_THROW((void)deglib::tuning::predict_quantization(vec, 3), std::invalid_argument);
    const std::vector<uint16_t> vec16 = {0x3c00, 0x3c00, 0x3c00, 0x3c00};
    EXPECT_THROW((void)deglib::tuning::predict_quantization(std::span<const uint16_t>(vec16), 0), std::invalid_argument);
    EXPECT_THROW((void)deglib::tuning::predict_quantization(std::span<const uint16_t>(vec16), 3), std::invalid_argument);

    // Empty (0 rows)
    const std::vector<float> empty;
    const auto q_empty = deglib::tuning::predict_quantization(empty, 8, deglib::distances::Metric::FP32_InnerProduct);
    EXPECT_EQ(q_empty.metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_FLOAT_EQ(q_empty.uint7_advantage_pct, 0.0f);
    EXPECT_EQ(q_empty.profile.metric, deglib::distances::Metric::FP32_InnerProduct);

    // Single row
    const std::vector<float> single(8, 1.0f);
    const auto q_single = deglib::tuning::predict_quantization(single, 8, deglib::distances::Metric::FP32_InnerProduct);
    EXPECT_EQ(q_single.metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_FLOAT_EQ(q_single.uint7_advantage_pct, 0.0f);
    EXPECT_EQ(q_single.profile.metric, deglib::distances::Metric::FP32_InnerProduct);

    // All identical vectors -> participation_ratio == 0 -> safe fallback
    const std::vector<float> identical(64 * 8, 3.5f);
    const auto q_ident = deglib::tuning::predict_quantization(identical, 8, deglib::distances::Metric::FP32_InnerProduct);
    EXPECT_EQ(q_ident.metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_FLOAT_EQ(q_ident.uint7_advantage_pct, 0.0f);

    // Extreme tiny participation_ratio -> clamped finite score, Uint7_L2
    const deglib::tuning::DatasetProfile tiny_pr{16000, 2048, 1e-12f, 1.5f, 60.0f, 0.0f, 1.0f, 0.0f, deglib::distances::Metric::FP32_InnerProduct};
    const auto q_tiny = deglib::tuning::predict_quantization(tiny_pr);
    EXPECT_EQ(q_tiny.metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_TRUE(std::isfinite(q_tiny.uint7_advantage_pct));
    EXPECT_GT(q_tiny.uint7_advantage_pct, 0.0f);

    // NaN / Inf fallback
    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    const float inf_val = std::numeric_limits<float>::infinity();
    const deglib::tuning::DatasetProfile nan_pr{16000, 512, nan_val, 1.9f, 68.0f, 0.0f, 1.0f, 0.0f, deglib::distances::Metric::FP32_InnerProduct};
    const deglib::tuning::DatasetProfile inf_rc{16000, 512, 0.2f, inf_val, 68.0f, 0.0f, 1.0f, 0.0f, deglib::distances::Metric::FP32_InnerProduct};
    EXPECT_EQ(deglib::tuning::predict_quantization(nan_pr).metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_FLOAT_EQ(deglib::tuning::predict_quantization(nan_pr).uint7_advantage_pct, 0.0f);
    EXPECT_EQ(deglib::tuning::predict_quantization(inf_rc).metric, deglib::distances::Metric::Uint7_L2);
    EXPECT_FLOAT_EQ(deglib::tuning::predict_quantization(inf_rc).uint7_advantage_pct, 0.0f);

    // Custom calibration override
    deglib::tuning::QuantizationModelCalibration custom_cal{};
    custom_cal.w_intercept = -0.5;
    custom_cal.w_dim_over_participation = 0.0;
    custom_cal.w_rel_contrast = 0.0;
    const deglib::tuning::DatasetProfile valid_ip{16000, 768, 0.1f, 1.7f, 70.0f, 0.0f, 1.0f, 0.0f, deglib::distances::Metric::FP32_InnerProduct};
    const auto q_custom = deglib::tuning::predict_quantization(valid_ip, custom_cal);
    EXPECT_EQ(q_custom.metric, deglib::distances::Metric::Int8_InnerProduct);
    EXPECT_LT(q_custom.uint7_advantage_pct, -25.0f);
}

}  // namespace
