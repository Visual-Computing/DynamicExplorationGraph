#pragma once

#include "deglib/builder.h"
#include "deglib/concurrent.h"
#include "deglib/distance/fp16.h"
#include "deglib/distances.h"
#include "deglib/transformation/normalize.h"
#include "deglib/utils/random.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace deglib::tuning {

inline constexpr size_t kDefaultSampleCount = 16000;
inline constexpr size_t kDefaultQueryCount = 0;  // 0 = evaluate all sampled vectors
inline constexpr size_t kDefaultRandomPairs = 65536;
inline constexpr uint32_t kDefaultSeed = 7;
inline constexpr float kDuplicateDistSqThreshold = 1e-10f;  // Euclidean d <= 1e-5

/**
 * Base-only geometric profile of a vector dataset.
 *
 * All features are duplicate-resilient, scale-invariant measurements computed
 * on a uniform random subsample of base vectors (L2-normalized for InnerProduct/Cosine):
 * - participation_ratio:   (tr(S))^2 / (dim * ||S||_F^2) of the centered scatter matrix S = Y^T Y.
 * - rel_contrast_1:        lower-quartile relative contrast median(d_rand) / Q1(d_1) against random pairwise distances.
 * - pp_iqr:                interquartile range (Q3 - Q1) of duplicate-free Two-NN local ID estimates 1 / log(d_2 / d_1).
 * - d16_skew:              skewness of 16-NN distances across the sampled base vectors.
 * - ball_cnt_p90_p10_r15:  P90 / max(P10, 1) of neighbor counts within radius 1.5 * median(d_16).
 * - hadamard_kurt:         mean per-coordinate excess kurtosis after randomized Fast Walsh-Hadamard rotation.
 */
struct DatasetProfile {
    uint32_t sample_count = 0;
    uint32_t dim = 0;
    float participation_ratio = 0.0f;
    float rel_contrast_1 = 0.0f;
    float pp_iqr = 0.0f;
    float d16_skew = 0.0f;
    float ball_cnt_p90_p10_r15 = 0.0f;
    float hadamard_kurt = 0.0f;
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2;
};

/**
 * Predicted graph construction optimization target and expected QPS advantage.
 */
struct TargetPrediction {
    deglib::builder::OptimizationTarget target = deglib::builder::OptimizationTarget::StreamingData;
    float lowlid_advantage_pct = 0.0f;
    DatasetProfile profile{};
};

/**
 * Predicted quantizer metric (Uint7_L2 vs Int8_InnerProduct) and expected QPS advantage.
 */
struct QuantizationPrediction {
    deglib::distances::Metric metric = deglib::distances::Metric::Uint7_L2;
    float uint7_advantage_pct = 0.0f;
    DatasetProfile profile{};
};

/**
 * Z-score + symmetric tanh calibration constants predicting log2(QPS_LowLID / QPS_StreamingData).
 *
 * Fitted on 10-seed means across the 13 training VIBE benchmark datasets
 * (sample_count=16000, query_count=0), holding out simplewiki-openai.
 */
struct TargetModelCalibration {
    double mu_d16_skew = -0.4903139398;
    double mu_ball_cnt_p90_p10_r15 = 4.4456391023;
    double mu_pp_iqr = 72.1514323895;
    double mu_hadamard_kurt = -0.0038754936;

    double sd_d16_skew = 0.7063880784;
    double sd_ball_cnt_p90_p10_r15 = 6.5147137624;
    double sd_pp_iqr = 23.8973006334;
    double sd_hadamard_kurt = 0.0715152265;

    double w_intercept = -1.2011258418;
    double w_d16_skew = -1.1787951650;
    double w_ball_cnt_p90_p10_r15 = -0.7266040198;
    double w_pp_iqr = -1.9881177479;
    double w_hadamard_kurt = 1.9634843799;

    double tanh_temp = 2.7;
    double log2_scale = 0.1711850246;
};

inline constexpr TargetModelCalibration kDefaultTargetCalibration{};

/**
 * Calibration constants for the K5-Top raw-vector quantization predictor (deglib.build).
 *
 * Predicts log2(QPS_Uint7_L2 / QPS_Int8_InnerProduct) via z-score + symmetric tanh (T=2.5, lambda=0.05)
 * from 5 features: coord_pos_frac, d1_tail_weight (on S), km16_var_expl (on S[:2000]),
 * log1p_mips_extra_ratio (from raw vector norms), and top32_var_ratio (on centered raw X[:4000]).
 *
 * Verified across 13 InnerProduct VIBE datasets x 10 seeds (sample_count=16000):
 * - 12-dataset fit (excluding arxiv-nomic tie): 12/12 LOO, 13/13 10-seed stable, rho=0.9580, RMSE=0.0525.
 * - 13-dataset evaluation: 12/12 LOO, 13/13 10-seed stable, rho=0.8352, RMSE=0.0552.
 *
 * Use this model in deglib.build(..., compress="8bit") where raw unnormalized vectors x are available.
 * Do NOT use it on already L2-normalized 32-bit graph vectors (||x|| == 1), where it degrades to 9/12 LOO;
 * use NormalizedQuantizationCalibration / predict_quantization_normalized (A1+A2) instead.
 */
struct K5QuantizationCalibration {
    double mu[5] = {0.451962508742, 1.01575590266, 0.056276651577, 0.290925029357, 0.430163613707};
    double sd[5] = {0.10597489595, 0.198663404925, 0.084832199336, 0.408868756066, 0.120893466906};
    double w[6] = {0.069528367635, 0.106903931192, -0.084378382048, 0.236850492395, 0.294950166115, -0.087842345105};
    double tanh_temp = 2.5;
};

inline constexpr K5QuantizationCalibration kDefaultK5QuantizationCalibration{};

/**
 * Calibration constants for the A1+A2 L2-normalization-invariant quantization ensemble (deglib.compress).
 *
 * Averages two 5-feature Ridge models (A1 with lambda=0.02, A2 with lambda=0.005) fitted on all
 * 13 InnerProduct VIBE datasets x 10 seeds (including arxiv-nomic):
 * - A1: mle_lid16_tail, rc_1_32, log1p(km16_ch_index), s_eig_pr_abs, is_cosine
 * - A2: d1_skew, log1p(km16_ch_index), s_top1_var_ratio, s_top10_var_ratio, is_cosine
 *
 * Every feature is 100% invariant to per-vector L2 normalization. `is_cosine` (1.0 if an
 * L2NormalizeTransform was attached to the graph, 0.0 if input vectors were already normalized)
 * is preserved on 32-bit graphs and passed explicitly.
 *
 * Verified in C++ across 13 InnerProduct VIBE datasets x 10 seeds:
 * 12/12 LOO, 13/13 10-seed stable, rho=0.8736, RMSE=0.0660.
 *
 * Use this model in deglib.compress(graph, compress="8bit") when original raw vectors are no longer
 * available and only already-normalized 32-bit graph vectors remain.
 */
struct NormalizedQuantizationCalibration {
    double mu1[5] = {2.094364033259, 1.607992305533, 1.38352234172, 106.583545230518, 0.384615384615};
    double sd1[5] = {0.173713904065, 0.185177378713, 1.462685812064, 55.729987183483, 0.486504255411};
    double w1[6] = {0.058241096438, 0.032304031315, -0.069819157704, 0.155771124155, 0.105924348051, 0.142014595245};

    double mu2[5] = {-0.606555704009, 1.38352234172, 0.061350359796, 0.248731304815, 0.384615384615};
    double sd2[5] = {0.696227543675, 1.462685812064, 0.042617357279, 0.098138232838, 0.486504255411};
    double w2[6] = {0.058241096438, -0.051024756716, 0.1273745551, 0.12126984743, -0.180407863562, 0.102825619064};
};

inline constexpr NormalizedQuantizationCalibration kDefaultNormalizedQuantizationCalibration{};

namespace detail {

inline size_t resolve_thread_count(size_t num_threads) noexcept {
    return (num_threads == 0) ? (std::max<size_t>)(1, std::thread::hardware_concurrency()) : num_threads;
}

inline std::vector<size_t> sample_distinct_indices(size_t total, size_t count, uint32_t seed) {
    if (count == 0 || count >= total) {
        std::vector<size_t> all(total);
        std::iota(all.begin(), all.end(), size_t{0});
        return all;
    }

    // Floyd's O(count) sampling avoids allocating an O(total) permutation buffer.
    std::unordered_set<size_t> selected;
    selected.reserve(count * 2);

    if (total <= static_cast<size_t>((std::numeric_limits<uint32_t>::max)())) {
        std::mt19937 rng(seed);
        for (size_t i = total - count; i < total; ++i) {
            deglib::random::DeterministicUniformIntDistribution<uint32_t> dist(0, static_cast<uint32_t>(i));
            const size_t t = static_cast<size_t>(dist(rng));
            if (!selected.insert(t).second) {
                selected.insert(i);
            }
        }
    } else {
        std::mt19937_64 rng64(seed);
        for (size_t i = total - count; i < total; ++i) {
            const uint64_t range = static_cast<uint64_t>(i) + 1ULL;
            const uint64_t limit = ((std::numeric_limits<uint64_t>::max)() / range) * range;
            uint64_t r = rng64();
            while (r >= limit) {
                r = rng64();
            }
            const size_t t = static_cast<size_t>(r % range);
            if (!selected.insert(t).second) {
                selected.insert(i);
            }
        }
    }

    std::vector<size_t> indices(selected.begin(), selected.end());
    std::sort(indices.begin(), indices.end());
    return indices;
}

inline double exact_median(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    const size_t n = values.size();
    const size_t mid = n / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<ptrdiff_t>(mid), values.end());
    if (n % 2 != 0) {
        return values[mid];
    }
    const double left_max = *std::max_element(values.begin(), values.begin() + static_cast<ptrdiff_t>(mid));
    return 0.5 * (left_max + values[mid]);
}

inline double sorted_percentile(const std::vector<double>& sorted, double q) noexcept {
    if (sorted.empty()) return 0.0;
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = (std::min)(lo + 1, sorted.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

inline constexpr size_t kTileM = 64;
inline constexpr size_t kTileN = 64;
inline constexpr size_t kTileK = 256;

struct TileScratch {
    std::vector<float> a_panel;
    std::vector<float> b_panel;
    std::vector<float> c_tile;

    TileScratch() : a_panel(kTileM * kTileK, 0.0f), b_panel(kTileN * kTileK, 0.0f), c_tile(kTileM * kTileN, 0.0f) {}
};

inline void pack_panel(
    const float* src,
    size_t src_stride,
    size_t row_start,
    size_t row_count,
    size_t col_start,
    size_t col_count,
    size_t row_pad,
    size_t col_pad,
    float* dst
) noexcept {
    const size_t copy_bytes = col_count * sizeof(float);
    for (size_t r = 0; r < row_count; ++r) {
        const float* srow = src + (row_start + r) * src_stride + col_start;
        float* drow = dst + r * col_pad;
        std::memcpy(drow, srow, copy_bytes);
        if (col_count < col_pad) {
            std::fill(drow + col_count, drow + col_pad, 0.0f);
        }
    }
    if (row_count < row_pad) {
        std::fill(dst + row_count * col_pad, dst + row_pad * col_pad, 0.0f);
    }
}

using GemmPanelFn = void (*)(const float* a_panel, size_t bm_pad, const float* b_panel, size_t bn_pad, size_t bk_pad, float* c_tile) noexcept;

inline void gemm_panel_accum_scalar(const float* a_panel, size_t bm_pad, const float* b_panel, size_t bn_pad, size_t bk_pad, float* c_tile) noexcept {
    for (size_t i = 0; i < bm_pad; i += 2) {
        const float* a0 = a_panel + i * bk_pad;
        const float* a1 = a0 + bk_pad;
        float* c0 = c_tile + i * kTileN;
        float* c1 = c0 + kTileN;
        for (size_t j = 0; j < bn_pad; j += 4) {
            const float* b0 = b_panel + j * bk_pad;
            const float* b1 = b0 + bk_pad;
            const float* b2 = b1 + bk_pad;
            const float* b3 = b2 + bk_pad;

            float acc00 = 0.0f;
            float acc01 = 0.0f;
            float acc02 = 0.0f;
            float acc03 = 0.0f;
            float acc10 = 0.0f;
            float acc11 = 0.0f;
            float acc12 = 0.0f;
            float acc13 = 0.0f;

            for (size_t k = 0; k < bk_pad; k += 4) {
                for (size_t u = 0; u < 4; ++u) {
                    const float va0 = a0[k + u];
                    const float va1 = a1[k + u];
                    const float vb0 = b0[k + u];
                    const float vb1 = b1[k + u];
                    const float vb2 = b2[k + u];
                    const float vb3 = b3[k + u];
                    acc00 = std::fma(va0, vb0, acc00);
                    acc01 = std::fma(va0, vb1, acc01);
                    acc02 = std::fma(va0, vb2, acc02);
                    acc03 = std::fma(va0, vb3, acc03);
                    acc10 = std::fma(va1, vb0, acc10);
                    acc11 = std::fma(va1, vb1, acc11);
                    acc12 = std::fma(va1, vb2, acc12);
                    acc13 = std::fma(va1, vb3, acc13);
                }
            }

            c0[j + 0] += acc00;
            c0[j + 1] += acc01;
            c0[j + 2] += acc02;
            c0[j + 3] += acc03;
            c1[j + 0] += acc10;
            c1[j + 1] += acc11;
            c1[j + 2] += acc12;
            c1[j + 3] += acc13;
        }
    }
}

#if defined(DEGLIB_X86)
DEGLIB_TARGET_AVX2 inline void
gemm_panel_accum_avx2(const float* a_panel, size_t bm_pad, const float* b_panel, size_t bn_pad, size_t bk_pad, float* c_tile) noexcept {
    for (size_t i = 0; i < bm_pad; i += 2) {
        const float* a0 = a_panel + i * bk_pad;
        const float* a1 = a0 + bk_pad;
        float* c0 = c_tile + i * kTileN;
        float* c1 = c0 + kTileN;
        for (size_t j = 0; j < bn_pad; j += 4) {
            const float* b0 = b_panel + j * bk_pad;
            const float* b1 = b0 + bk_pad;
            const float* b2 = b1 + bk_pad;
            const float* b3 = b2 + bk_pad;

            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc02 = _mm256_setzero_ps();
            __m256 acc03 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc12 = _mm256_setzero_ps();
            __m256 acc13 = _mm256_setzero_ps();

            for (size_t k = 0; k < bk_pad; k += 8) {
                const __m256 va0 = _mm256_loadu_ps(a0 + k);
                const __m256 va1 = _mm256_loadu_ps(a1 + k);

                const __m256 vb0 = _mm256_loadu_ps(b0 + k);
                acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                acc10 = _mm256_fmadd_ps(va1, vb0, acc10);

                const __m256 vb1 = _mm256_loadu_ps(b1 + k);
                acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                acc11 = _mm256_fmadd_ps(va1, vb1, acc11);

                const __m256 vb2 = _mm256_loadu_ps(b2 + k);
                acc02 = _mm256_fmadd_ps(va0, vb2, acc02);
                acc12 = _mm256_fmadd_ps(va1, vb2, acc12);

                const __m256 vb3 = _mm256_loadu_ps(b3 + k);
                acc03 = _mm256_fmadd_ps(va0, vb3, acc03);
                acc13 = _mm256_fmadd_ps(va1, vb3, acc13);
            }

            c0[j + 0] += deglib::distances::fp32_ip::fp32_hsum256(acc00);
            c0[j + 1] += deglib::distances::fp32_ip::fp32_hsum256(acc01);
            c0[j + 2] += deglib::distances::fp32_ip::fp32_hsum256(acc02);
            c0[j + 3] += deglib::distances::fp32_ip::fp32_hsum256(acc03);
            c1[j + 0] += deglib::distances::fp32_ip::fp32_hsum256(acc10);
            c1[j + 1] += deglib::distances::fp32_ip::fp32_hsum256(acc11);
            c1[j + 2] += deglib::distances::fp32_ip::fp32_hsum256(acc12);
            c1[j + 3] += deglib::distances::fp32_ip::fp32_hsum256(acc13);
        }
    }
}

DEGLIB_TARGET_AVX512 inline void
gemm_panel_accum_avx512(const float* a_panel, size_t bm_pad, const float* b_panel, size_t bn_pad, size_t bk_pad, float* c_tile) noexcept {
    for (size_t i = 0; i < bm_pad; i += 4) {
        const float* a0 = a_panel + i * bk_pad;
        const float* a1 = a0 + bk_pad;
        const float* a2 = a1 + bk_pad;
        const float* a3 = a2 + bk_pad;
        float* c0 = c_tile + i * kTileN;
        float* c1 = c0 + kTileN;
        float* c2 = c1 + kTileN;
        float* c3 = c2 + kTileN;
        for (size_t j = 0; j < bn_pad; j += 4) {
            const float* b0 = b_panel + j * bk_pad;
            const float* b1 = b0 + bk_pad;
            const float* b2 = b1 + bk_pad;
            const float* b3 = b2 + bk_pad;

            __m512 acc00 = _mm512_setzero_ps();
            __m512 acc01 = _mm512_setzero_ps();
            __m512 acc02 = _mm512_setzero_ps();
            __m512 acc03 = _mm512_setzero_ps();
            __m512 acc10 = _mm512_setzero_ps();
            __m512 acc11 = _mm512_setzero_ps();
            __m512 acc12 = _mm512_setzero_ps();
            __m512 acc13 = _mm512_setzero_ps();
            __m512 acc20 = _mm512_setzero_ps();
            __m512 acc21 = _mm512_setzero_ps();
            __m512 acc22 = _mm512_setzero_ps();
            __m512 acc23 = _mm512_setzero_ps();
            __m512 acc30 = _mm512_setzero_ps();
            __m512 acc31 = _mm512_setzero_ps();
            __m512 acc32 = _mm512_setzero_ps();
            __m512 acc33 = _mm512_setzero_ps();

            for (size_t k = 0; k < bk_pad; k += 16) {
                const __m512 va0 = _mm512_loadu_ps(a0 + k);
                const __m512 va1 = _mm512_loadu_ps(a1 + k);
                const __m512 va2 = _mm512_loadu_ps(a2 + k);
                const __m512 va3 = _mm512_loadu_ps(a3 + k);

                const __m512 vb0 = _mm512_loadu_ps(b0 + k);
                acc00 = _mm512_fmadd_ps(va0, vb0, acc00);
                acc10 = _mm512_fmadd_ps(va1, vb0, acc10);
                acc20 = _mm512_fmadd_ps(va2, vb0, acc20);
                acc30 = _mm512_fmadd_ps(va3, vb0, acc30);

                const __m512 vb1 = _mm512_loadu_ps(b1 + k);
                acc01 = _mm512_fmadd_ps(va0, vb1, acc01);
                acc11 = _mm512_fmadd_ps(va1, vb1, acc11);
                acc21 = _mm512_fmadd_ps(va2, vb1, acc21);
                acc31 = _mm512_fmadd_ps(va3, vb1, acc31);

                const __m512 vb2 = _mm512_loadu_ps(b2 + k);
                acc02 = _mm512_fmadd_ps(va0, vb2, acc02);
                acc12 = _mm512_fmadd_ps(va1, vb2, acc12);
                acc22 = _mm512_fmadd_ps(va2, vb2, acc22);
                acc32 = _mm512_fmadd_ps(va3, vb2, acc32);

                const __m512 vb3 = _mm512_loadu_ps(b3 + k);
                acc03 = _mm512_fmadd_ps(va0, vb3, acc03);
                acc13 = _mm512_fmadd_ps(va1, vb3, acc13);
                acc23 = _mm512_fmadd_ps(va2, vb3, acc23);
                acc33 = _mm512_fmadd_ps(va3, vb3, acc33);
            }

            c0[j + 0] += deglib::distances::fp32_ip::fp32_hsum512(acc00);
            c0[j + 1] += deglib::distances::fp32_ip::fp32_hsum512(acc01);
            c0[j + 2] += deglib::distances::fp32_ip::fp32_hsum512(acc02);
            c0[j + 3] += deglib::distances::fp32_ip::fp32_hsum512(acc03);

            c1[j + 0] += deglib::distances::fp32_ip::fp32_hsum512(acc10);
            c1[j + 1] += deglib::distances::fp32_ip::fp32_hsum512(acc11);
            c1[j + 2] += deglib::distances::fp32_ip::fp32_hsum512(acc12);
            c1[j + 3] += deglib::distances::fp32_ip::fp32_hsum512(acc13);

            c2[j + 0] += deglib::distances::fp32_ip::fp32_hsum512(acc20);
            c2[j + 1] += deglib::distances::fp32_ip::fp32_hsum512(acc21);
            c2[j + 2] += deglib::distances::fp32_ip::fp32_hsum512(acc22);
            c2[j + 3] += deglib::distances::fp32_ip::fp32_hsum512(acc23);

            c3[j + 0] += deglib::distances::fp32_ip::fp32_hsum512(acc30);
            c3[j + 1] += deglib::distances::fp32_ip::fp32_hsum512(acc31);
            c3[j + 2] += deglib::distances::fp32_ip::fp32_hsum512(acc32);
            c3[j + 3] += deglib::distances::fp32_ip::fp32_hsum512(acc33);
        }
    }
}
#endif

inline GemmPanelFn resolve_gemm_panel_fn() noexcept {
#if defined(DEGLIB_X86)
    if (deglib::cpu::has_avx512()) {
        return &gemm_panel_accum_avx512;
    }
    if (deglib::cpu::has_avx2()) {
        return &gemm_panel_accum_avx2;
    }
#endif
    return &gemm_panel_accum_scalar;
}

inline float compute_participation_ratio(const float* data, size_t n, uint32_t dim, size_t threads) {
    if (n < 2 || dim == 0) return 0.0f;

    std::vector<double> mean(dim, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const float* row = data + i * dim;
        for (uint32_t j = 0; j < dim; ++j) {
            mean[j] += static_cast<double>(row[j]);
        }
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    for (uint32_t j = 0; j < dim; ++j) {
        mean[j] *= inv_n;
    }

    std::vector<float> centered(n * static_cast<size_t>(dim));
    double tr_s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float* src = data + i * dim;
        float* dst = centered.data() + i * dim;
        for (uint32_t j = 0; j < dim; ++j) {
            const double diff = static_cast<double>(src[j]) - mean[j];
            dst[j] = static_cast<float>(diff);
            tr_s += diff * diff;
        }
    }

    if (tr_s <= 1e-24) return 0.0f;

    // Uniform scaling makes tr(S) == dim, keeping float products around O(1) without affecting (tr)^2 / ||S||_F^2.
    const float alpha = static_cast<float>(std::sqrt(static_cast<double>(dim) / tr_s));
    for (float& v : centered) {
        v *= alpha;
    }

    // Form the smaller Gram matrix (dim x dim when n >= dim, n x n when n < dim) since both share nonzero eigenvalues.
    std::vector<float> transposed;
    const float* gram_ptr = centered.data();
    size_t rows = n;
    size_t cols = dim;
    if (n >= dim) {
        transposed.resize(static_cast<size_t>(dim) * n);
        for (size_t i = 0; i < n; ++i) {
            const float* row = centered.data() + i * dim;
            for (uint32_t j = 0; j < dim; ++j) {
                transposed[static_cast<size_t>(j) * n + i] = row[j];
            }
        }
        gram_ptr = transposed.data();
        rows = dim;
        cols = n;
    }

    const GemmPanelFn gemm_panel_fn = resolve_gemm_panel_fn();
    const size_t num_tiles = (rows + kTileM - 1) / kTileM;
    std::vector<std::pair<size_t, size_t>> tiles;
    tiles.reserve((num_tiles * (num_tiles + 1)) / 2);
    for (size_t bi = 0; bi < num_tiles; ++bi) {
        for (size_t bj = bi; bj < num_tiles; ++bj) {
            tiles.emplace_back(bi, bj);
        }
    }

    const size_t actual_threads = (std::min)(threads, tiles.size());
    std::vector<TileScratch> scratch(actual_threads);
    std::vector<double> thread_tr(actual_threads, 0.0);
    std::vector<double> thread_frob2(actual_threads, 0.0);

    deglib::concurrent::parallel_batch_for(
        size_t{0}, tiles.size(), actual_threads,
        [&](size_t begin, size_t end, size_t tid) {
            TileScratch& sc = scratch[tid];
            double local_tr = 0.0;
            double local_frob2 = 0.0;
            for (size_t t = begin; t < end; ++t) {
                const auto [bi, bj] = tiles[t];
                const size_t i0 = bi * kTileM;
                const size_t bm = (std::min)(kTileM, rows - i0);
                const size_t bm_pad = (bm + 3) & ~size_t{3};
                std::fill_n(sc.c_tile.data(), bm_pad * kTileN, 0.0f);

                if (bi == bj) {
                    for (size_t k0 = 0; k0 < cols; k0 += kTileK) {
                        const size_t bk = (std::min)(kTileK, cols - k0);
                        const size_t bk_pad = (bk + 15) & ~size_t{15};
                        pack_panel(gram_ptr, cols, i0, bm, k0, bk, bm_pad, bk_pad, sc.a_panel.data());
                        gemm_panel_fn(sc.a_panel.data(), bm_pad, sc.a_panel.data(), bm_pad, bk_pad, sc.c_tile.data());
                    }
                    for (size_t r = 0; r < bm; ++r) {
                        const float* c_row = sc.c_tile.data() + r * kTileN;
                        const double diag = static_cast<double>(c_row[r]);
                        local_tr += diag;
                        local_frob2 += diag * diag;
                        for (size_t c = r + 1; c < bm; ++c) {
                            const double v = static_cast<double>(c_row[c]);
                            local_frob2 += 2.0 * v * v;
                        }
                    }
                } else {
                    const size_t j0 = bj * kTileN;
                    const size_t bn = (std::min)(kTileN, rows - j0);
                    const size_t bn_pad = (bn + 3) & ~size_t{3};
                    for (size_t k0 = 0; k0 < cols; k0 += kTileK) {
                        const size_t bk = (std::min)(kTileK, cols - k0);
                        const size_t bk_pad = (bk + 15) & ~size_t{15};
                        pack_panel(gram_ptr, cols, i0, bm, k0, bk, bm_pad, bk_pad, sc.a_panel.data());
                        pack_panel(gram_ptr, cols, j0, bn, k0, bk, bn_pad, bk_pad, sc.b_panel.data());
                        gemm_panel_fn(sc.a_panel.data(), bm_pad, sc.b_panel.data(), bn_pad, bk_pad, sc.c_tile.data());
                    }
                    for (size_t r = 0; r < bm; ++r) {
                        const float* c_row = sc.c_tile.data() + r * kTileN;
                        for (size_t c = 0; c < bn; ++c) {
                            const double v = static_cast<double>(c_row[c]);
                            local_frob2 += 2.0 * v * v;
                        }
                    }
                }
            }
            thread_tr[tid] += local_tr;
            thread_frob2[tid] += local_frob2;
        },
        4
    );

    double tr_scaled = 0.0;
    double frob2 = 0.0;
    for (size_t t = 0; t < actual_threads; ++t) {
        tr_scaled += thread_tr[t];
        frob2 += thread_frob2[t];
    }

    if (frob2 <= 1e-24) return 0.0f;
    const double ratio = (tr_scaled * tr_scaled) / (static_cast<double>(dim) * frob2);
    return static_cast<float>(std::clamp(ratio, 0.0, 1.0));
}

struct KnnContrastStats {
    float pp_iqr = 0.0f;
    float rel_contrast_1 = 0.0f;
    float d16_skew = 0.0f;
    float ball_cnt_p90_p10_r15 = 0.0f;
};

inline float compute_ball_cnt_p90_p10_r15(
    const float* data,
    size_t n,
    uint32_t dim,
    const std::vector<float>& x_norms2,
    double r15,
    size_t threads
) {
    const size_t m = (std::min)(n, size_t{4000});
    if (m < 2 || r15 <= 0.0 || !std::isfinite(r15)) return 1.0f;

    const float r15_sq = static_cast<float>(r15 * r15);
    const GemmPanelFn gemm_panel_fn = resolve_gemm_panel_fn();
    deglib::distances::FloatSpace l2_space(dim, deglib::distances::Metric::FP32_L2);
    const auto dist_func = l2_space.get_dist_func();
    const void* param = l2_space.get_dist_func_param();
    const float tol_factor = (std::max)(2e-3f, static_cast<float>(dim) * 1e-6f);

    const size_t num_q_tiles = (m + kTileM - 1) / kTileM;
    const size_t actual_threads = (std::min)(threads, num_q_tiles);
    std::vector<TileScratch> scratch(actual_threads);
    std::vector<double> counts(m, 0.0);

    deglib::concurrent::parallel_batch_for(
        size_t{0}, num_q_tiles, actual_threads,
        [&](size_t begin, size_t end, size_t tid) {
            TileScratch& sc = scratch[tid];
            for (size_t qt = begin; qt < end; ++qt) {
                const size_t i0 = qt * kTileM;
                const size_t bm = (std::min)(kTileM, m - i0);
                const size_t bm_pad = (bm + 3) & ~size_t{3};
                uint32_t tile_cnt[kTileM] = {};

                for (size_t j0 = 0; j0 < n; j0 += kTileN) {
                    const size_t bn = (std::min)(kTileN, n - j0);
                    const size_t bn_pad = (bn + 3) & ~size_t{3};
                    std::fill_n(sc.c_tile.data(), bm_pad * kTileN, 0.0f);

                    for (size_t k0 = 0; k0 < dim; k0 += kTileK) {
                        const size_t bk = (std::min)(static_cast<size_t>(dim) - k0, kTileK);
                        const size_t bk_pad = (bk + 15) & ~size_t{15};
                        pack_panel(data, dim, i0, bm, k0, bk, bm_pad, bk_pad, sc.a_panel.data());
                        pack_panel(data, dim, j0, bn, k0, bk, bn_pad, bk_pad, sc.b_panel.data());
                        gemm_panel_fn(sc.a_panel.data(), bm_pad, sc.b_panel.data(), bn_pad, bk_pad, sc.c_tile.data());
                    }

                    for (size_t r = 0; r < bm; ++r) {
                        const size_t qi = i0 + r;
                        const float qn = x_norms2[qi];
                        const float* c_row = sc.c_tile.data() + r * kTileN;
                        const float* q_ptr = data + qi * dim;
                        uint32_t local_cnt = tile_cnt[r];
                        for (size_t c = 0; c < bn; ++c) {
                            const size_t db_row = j0 + c;
                            if (db_row == qi) continue;
                            const float norm_sum = qn + x_norms2[db_row];
                            const float d2_approx = norm_sum - 2.0f * c_row[c];
                            const float err_margin = tol_factor * norm_sum;
                            if (d2_approx <= r15_sq + err_margin) {
                                const float d2 = dist_func(q_ptr, data + db_row * dim, param);
                                if (d2 <= r15_sq) {
                                    ++local_cnt;
                                }
                            }
                        }
                        tile_cnt[r] = local_cnt;
                    }
                }
                for (size_t r = 0; r < bm; ++r) {
                    counts[i0 + r] = static_cast<double>(tile_cnt[r]);
                }
            }
        },
        4
    );

    std::sort(counts.begin(), counts.end());
    const double p90 = sorted_percentile(counts, 0.90);
    const double p10 = sorted_percentile(counts, 0.10);
    return static_cast<float>(p90 / (std::max)(p10, 1.0));
}

inline KnnContrastStats compute_knn_and_contrast(const float* data, size_t n, uint32_t dim, size_t query_count, uint32_t seed, size_t threads) {
    if (n < 2 || dim == 0) return {};

    std::vector<float> x_norms2(n);
    for (size_t i = 0; i < n; ++i) {
        const float* row = data + i * dim;
        double norm2 = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            const double v = static_cast<double>(row[d]);
            norm2 += v * v;
        }
        x_norms2[i] = static_cast<float>(norm2);
    }

    const bool all_queries = (query_count == 0 || query_count >= n);
    std::vector<size_t> q_indices;
    std::vector<float> q_buf;
    std::vector<float> q_norms2_buf;
    const float* q_data = data;
    const float* q_norms2 = x_norms2.data();
    size_t q_count = n;

    if (!all_queries) {
        q_indices = sample_distinct_indices(n, query_count, seed + 1);
        q_count = q_indices.size();
        q_buf.resize(q_count * static_cast<size_t>(dim));
        q_norms2_buf.resize(q_count);
        const size_t row_bytes = static_cast<size_t>(dim) * sizeof(float);
        for (size_t i = 0; i < q_count; ++i) {
            const size_t src_row = q_indices[i];
            std::memcpy(q_buf.data() + i * dim, data + src_row * dim, row_bytes);
            q_norms2_buf[i] = x_norms2[src_row];
        }
        q_data = q_buf.data();
        q_norms2 = q_norms2_buf.data();
    }

    constexpr size_t kMaxK = 16;
    const size_t k_target = (std::min)(kMaxK, n - 1);

    const GemmPanelFn gemm_panel_fn = resolve_gemm_panel_fn();
    deglib::distances::FloatSpace l2_space(dim, deglib::distances::Metric::FP32_L2);
    const auto dist_func = l2_space.get_dist_func();
    const void* param = l2_space.get_dist_func_param();
    const float tol_factor = (std::max)(2e-3f, static_cast<float>(dim) * 1e-6f);

    const size_t num_q_tiles = (q_count + kTileM - 1) / kTileM;
    const size_t actual_threads = (std::min)(threads, num_q_tiles);
    std::vector<TileScratch> scratch(actual_threads);
    std::vector<double> d1_out(q_count, -1.0);
    std::vector<double> d16_out(q_count, -1.0);
    std::vector<double> pp_out(q_count, -1.0);

    deglib::concurrent::parallel_batch_for(
        size_t{0}, num_q_tiles, actual_threads,
        [&](size_t begin, size_t end, size_t tid) {
            TileScratch& sc = scratch[tid];
            for (size_t qt = begin; qt < end; ++qt) {
                const size_t i0 = qt * kTileM;
                const size_t bm = (std::min)(kTileM, q_count - i0);
                const size_t bm_pad = (bm + 3) & ~size_t{3};

                float tile_topk[kTileM][kMaxK];
                for (size_t r = 0; r < bm; ++r) {
                    std::fill_n(tile_topk[r], k_target, (std::numeric_limits<float>::infinity)());
                }

                for (size_t j0 = 0; j0 < n; j0 += kTileN) {
                    const size_t bn = (std::min)(kTileN, n - j0);
                    const size_t bn_pad = (bn + 3) & ~size_t{3};
                    std::fill_n(sc.c_tile.data(), bm_pad * kTileN, 0.0f);

                    for (size_t k0 = 0; k0 < dim; k0 += kTileK) {
                        const size_t bk = (std::min)(static_cast<size_t>(dim) - k0, kTileK);
                        const size_t bk_pad = (bk + 15) & ~size_t{15};
                        pack_panel(q_data, dim, i0, bm, k0, bk, bm_pad, bk_pad, sc.a_panel.data());
                        pack_panel(data, dim, j0, bn, k0, bk, bn_pad, bk_pad, sc.b_panel.data());
                        gemm_panel_fn(sc.a_panel.data(), bm_pad, sc.b_panel.data(), bn_pad, bk_pad, sc.c_tile.data());
                    }

                    for (size_t r = 0; r < bm; ++r) {
                        const size_t qi = i0 + r;
                        const size_t q_row = all_queries ? qi : q_indices[qi];
                        const float qn = q_norms2[qi];
                        float* topk = tile_topk[r];
                        float kth = topk[k_target - 1];
                        const float* c_row = sc.c_tile.data() + r * kTileN;
                        const float* q_ptr = q_data + qi * dim;
                        for (size_t c = 0; c < bn; ++c) {
                            const size_t db_row = j0 + c;
                            if (db_row == q_row) continue;
                            const float norm_sum = qn + x_norms2[db_row];
                            const float d2_approx = norm_sum - 2.0f * c_row[c];
                            const float err_margin = tol_factor * norm_sum;
                            if (d2_approx <= kth + err_margin || d2_approx <= kDuplicateDistSqThreshold + err_margin) {
                                const float d2 = dist_func(q_ptr, data + db_row * dim, param);
                                if (d2 > kDuplicateDistSqThreshold && d2 < kth) {
                                    size_t pos = k_target - 1;
                                    while (pos > 0 && d2 < topk[pos - 1]) {
                                        topk[pos] = topk[pos - 1];
                                        --pos;
                                    }
                                    topk[pos] = d2;
                                    kth = topk[k_target - 1];
                                }
                            }
                        }
                    }
                }

                for (size_t r = 0; r < bm; ++r) {
                    const float min1 = tile_topk[r][0];
                    if (std::isfinite(min1)) {
                        d1_out[i0 + r] = std::sqrt(static_cast<double>(min1));
                        if (k_target >= 2) {
                            const float min2 = tile_topk[r][1];
                            if (std::isfinite(min2) && min1 > 0.0f && min2 > min1) {
                                pp_out[i0 + r] = 2.0 / std::log(static_cast<double>(min2) / static_cast<double>(min1));
                            }
                        }
                        // Use the farthest finite neighbor up to k_target as d16.
                        for (size_t k = k_target; k > 0; --k) {
                            if (std::isfinite(tile_topk[r][k - 1])) {
                                d16_out[i0 + r] = std::sqrt(static_cast<double>(tile_topk[r][k - 1]));
                                break;
                            }
                        }
                    }
                }
            }
        },
        4
    );

    std::vector<double> valid_d1;
    valid_d1.reserve(d1_out.size());
    for (double d : d1_out) {
        if (d > 0.0) valid_d1.push_back(d);
    }
    if (valid_d1.size() < 2) return {};

    std::sort(valid_d1.begin(), valid_d1.end());
    const double q25_d1 = sorted_percentile(valid_d1, 0.25);

    std::vector<double> valid_pp;
    valid_pp.reserve(pp_out.size());
    for (double v : pp_out) {
        if (v > 0.0 && std::isfinite(v)) valid_pp.push_back(v);
    }
    double pp_iqr = 0.0;
    if (valid_pp.size() >= 2) {
        std::sort(valid_pp.begin(), valid_pp.end());
        pp_iqr = sorted_percentile(valid_pp, 0.75) - sorted_percentile(valid_pp, 0.25);
    }

    std::vector<double> valid_d16;
    valid_d16.reserve(d16_out.size());
    for (double d : d16_out) {
        if (d > 0.0 && std::isfinite(d)) valid_d16.push_back(d);
    }
    double d16_skew = 0.0;
    double med_d16 = 0.0;
    if (valid_d16.size() >= 2) {
        double mean_d16 = 0.0;
        for (double d : valid_d16) mean_d16 += d;
        mean_d16 /= static_cast<double>(valid_d16.size());
        double var_d16 = 0.0;
        for (double d : valid_d16) {
            const double diff = d - mean_d16;
            var_d16 += diff * diff;
        }
        var_d16 /= static_cast<double>(valid_d16.size());
        const double sd_d16 = std::sqrt(var_d16);
        if (sd_d16 > 1e-12) {
            double m3 = 0.0;
            for (double d : valid_d16) {
                const double z = (d - mean_d16) / sd_d16;
                m3 += z * z * z;
            }
            d16_skew = m3 / static_cast<double>(valid_d16.size());
        }
        med_d16 = exact_median(valid_d16);
    }

    std::mt19937 pair_rng(seed + 100);
    deglib::random::DeterministicUniformIntDistribution<uint32_t> row_dist(0, static_cast<uint32_t>(n - 1));
    std::vector<double> drand;
    drand.reserve(kDefaultRandomPairs);
    for (size_t p = 0; p < kDefaultRandomPairs; ++p) {
        const uint32_t a = row_dist(pair_rng);
        uint32_t b = row_dist(pair_rng);
        if (b == a) b = (b + 1) % static_cast<uint32_t>(n);
        const float d2 = dist_func(data + static_cast<size_t>(a) * dim, data + static_cast<size_t>(b) * dim, param);
        drand.push_back(std::sqrt(static_cast<double>((std::max)(0.0f, d2))));
    }
    const double med_rand = exact_median(drand);
    const double rel_contrast_1 = (q25_d1 > 1e-12) ? (med_rand / q25_d1) : 0.0;
    const float ball_cnt = compute_ball_cnt_p90_p10_r15(data, n, dim, x_norms2, 1.5 * med_d16, threads);

    return KnnContrastStats{
        static_cast<float>(pp_iqr),
        static_cast<float>(rel_contrast_1),
        static_cast<float>(d16_skew),
        ball_cnt,
    };
}

inline void fast_walsh_hadamard_inplace(float* vec, size_t p2) noexcept {
    for (size_t h = 1; h < p2; h <<= 1) {
        for (size_t i = 0; i < p2; i += (h << 1)) {
            for (size_t j = 0; j < h; ++j) {
                const float a = vec[i + j];
                const float b = vec[i + j + h];
                vec[i + j] = a + b;
                vec[i + j + h] = a - b;
            }
        }
    }
}

inline float compute_hadamard_kurt(const float* data, size_t n, uint32_t dim, uint32_t seed, size_t threads) {
    if (n < 2 || dim == 0) return 0.0f;

    size_t p2 = 1;
    while (p2 < static_cast<size_t>(dim)) p2 <<= 1;
    const float inv_sqrt_p2 = 1.0f / std::sqrt(static_cast<float>(p2));

    std::mt19937 sign_rng(seed + 200);
    std::vector<float> signs(dim);
    for (uint32_t j = 0; j < dim; ++j) {
        signs[j] = ((sign_rng() & 1u) == 0u) ? 1.0f : -1.0f;
    }

    const size_t actual_threads = (std::min)(threads, n);
    std::vector<std::vector<double>> thread_sum(actual_threads, std::vector<double>(p2, 0.0));
    std::vector<std::vector<double>> thread_sum2(actual_threads, std::vector<double>(p2, 0.0));
    std::vector<std::vector<float>> thread_buf(actual_threads, std::vector<float>(p2, 0.0f));

    deglib::concurrent::parallel_batch_for(
        size_t{0}, n, actual_threads,
        [&](size_t begin, size_t end, size_t tid) {
            auto& buf = thread_buf[tid];
            auto& s1 = thread_sum[tid];
            auto& s2 = thread_sum2[tid];
            for (size_t i = begin; i < end; ++i) {
                const float* row = data + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    buf[j] = row[j] * signs[j];
                }
                std::fill(buf.begin() + dim, buf.end(), 0.0f);
                fast_walsh_hadamard_inplace(buf.data(), p2);
                for (size_t c = 0; c < p2; ++c) {
                    const double v = static_cast<double>(buf[c] * inv_sqrt_p2);
                    s1[c] += v;
                    s2[c] += v * v;
                }
            }
        },
        16
    );

    const double inv_n = 1.0 / static_cast<double>(n);
    std::vector<double> mean(p2, 0.0);
    std::vector<double> inv_sd(p2, 0.0);
    for (size_t c = 0; c < p2; ++c) {
        double s1 = 0.0;
        double s2 = 0.0;
        for (size_t t = 0; t < actual_threads; ++t) {
            s1 += thread_sum[t][c];
            s2 += thread_sum2[t][c];
        }
        const double m = s1 * inv_n;
        const double var = (std::max)(s2 * inv_n - m * m, 0.0);
        mean[c] = m;
        inv_sd[c] = 1.0 / (std::max)(std::sqrt(var), 1e-8);
    }

    std::vector<std::vector<double>> thread_m4(actual_threads, std::vector<double>(p2, 0.0));
    deglib::concurrent::parallel_batch_for(
        size_t{0}, n, actual_threads,
        [&](size_t begin, size_t end, size_t tid) {
            auto& buf = thread_buf[tid];
            auto& m4 = thread_m4[tid];
            for (size_t i = begin; i < end; ++i) {
                const float* row = data + i * dim;
                for (uint32_t j = 0; j < dim; ++j) {
                    buf[j] = row[j] * signs[j];
                }
                std::fill(buf.begin() + dim, buf.end(), 0.0f);
                fast_walsh_hadamard_inplace(buf.data(), p2);
                for (size_t c = 0; c < p2; ++c) {
                    const double z = (static_cast<double>(buf[c] * inv_sqrt_p2) - mean[c]) * inv_sd[c];
                    const double z2 = z * z;
                    m4[c] += z2 * z2;
                }
            }
        },
        16
    );

    double kurt_sum = 0.0;
    for (size_t c = 0; c < p2; ++c) {
        double m4 = 0.0;
        for (size_t t = 0; t < actual_threads; ++t) {
            m4 += thread_m4[t][c];
        }
        kurt_sum += (m4 * inv_n) - 3.0;
    }
    return static_cast<float>(kurt_sum / static_cast<double>(p2));
}

inline DatasetProfile measure_profile_from_sampled(
    std::vector<float>& sampled,
    uint32_t dim,
    deglib::distances::Metric metric,
    size_t query_count,
    uint32_t seed,
    size_t num_threads
) {
    const size_t n = sampled.size() / dim;
    const size_t threads = resolve_thread_count(num_threads);

    if (metric.get_distance_kind() == deglib::distances::MetricDistanceKind::InnerProduct) {
        if (threads <= 1 || n < 1000) {
            for (size_t i = 0; i < n; ++i) {
                float* row = sampled.data() + i * dim;
                deglib::transformation::detail::normalize_vector_l2(row, row, dim);
            }
        } else {
            deglib::concurrent::parallel_batch_for(
                size_t{0}, n, threads,
                [&](size_t begin, size_t end, size_t /*tid*/) {
                    for (size_t i = begin; i < end; ++i) {
                        float* row = sampled.data() + i * dim;
                        deglib::transformation::detail::normalize_vector_l2(row, row, dim);
                    }
                },
                16
            );
        }
    }

    DatasetProfile profile;
    profile.sample_count = static_cast<uint32_t>(n);
    profile.dim = dim;
    profile.participation_ratio = compute_participation_ratio(sampled.data(), n, dim, threads);
    const KnnContrastStats knn = compute_knn_and_contrast(sampled.data(), n, dim, query_count, seed, threads);
    profile.rel_contrast_1 = knn.rel_contrast_1;
    profile.pp_iqr = knn.pp_iqr;
    profile.d16_skew = knn.d16_skew;
    profile.ball_cnt_p90_p10_r15 = knn.ball_cnt_p90_p10_r15;
    profile.hadamard_kurt = compute_hadamard_kurt(sampled.data(), n, dim, seed, threads);
    profile.metric = metric;
    return profile;
}

inline void jacobi_eigenvalues_sym(std::vector<double>& a, size_t n, std::vector<double>& evals) {
    evals.resize(n);
    for (size_t iter = 0; iter < 60 * n * n; ++iter) {
        size_t p = 0, q = 1;
        double max_off = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const double v = std::abs(a[i * n + j]);
                if (v > max_off) {
                    max_off = v;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < 1e-12) break;
        const double app = a[p * n + p];
        const double aqq = a[q * n + q];
        const double apq = a[p * n + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                      : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
        for (size_t k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            const double akp = a[k * n + p];
            const double akq = a[k * n + q];
            a[k * n + p] = a[p * n + k] = c * akp - s * akq;
            a[k * n + q] = a[q * n + k] = s * akp + c * akq;
        }
    }
    for (size_t i = 0; i < n; ++i) evals[i] = a[i * n + i];
    std::sort(evals.begin(), evals.end(), std::greater<double>());
}

inline void mgs_orthonormalize(std::vector<float>& q, size_t rows, size_t cols) {
    for (size_t j = 0; j < cols; ++j) {
        for (size_t k = 0; k < j; ++k) {
            double dot = 0.0;
            for (size_t i = 0; i < rows; ++i) {
                dot += static_cast<double>(q[i * cols + j]) * static_cast<double>(q[i * cols + k]);
            }
            const float fdot = static_cast<float>(dot);
            for (size_t i = 0; i < rows; ++i) {
                q[i * cols + j] -= fdot * q[i * cols + k];
            }
        }
        double n2 = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            const double v = static_cast<double>(q[i * cols + j]);
            n2 += v * v;
        }
        const float inv = (n2 > 1e-24) ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
        for (size_t i = 0; i < rows; ++i) {
            q[i * cols + j] *= inv;
        }
    }
}

// L2-normalize sampled rows into `S`; optionally count positive raw coordinates and record raw norms.
inline double normalize_rows_l2(
    const float* data,
    size_t n,
    uint32_t dim,
    size_t threads,
    std::vector<float>& S,
    std::vector<float>* out_norms,
    bool count_pos
) {
    S.resize(n * dim);
    if (out_norms) out_norms->resize(n);
    std::vector<uint64_t> pos_counts(threads, 0);

    deglib::concurrent::parallel_batch_for(
        size_t{0}, n, threads,
        [&](size_t begin, size_t end, size_t tid) {
            uint64_t pc = 0;
            for (size_t i = begin; i < end; ++i) {
                const float* src = data + i * dim;
                float* dst = S.data() + i * dim;
                double n2 = 0.0;
                for (uint32_t d = 0; d < dim; ++d) {
                    const float v = src[d];
                    if (count_pos && v > 0.0f) ++pc;
                    n2 += static_cast<double>(v) * static_cast<double>(v);
                }
                const float nm = static_cast<float>(std::sqrt(n2));
                if (out_norms) (*out_norms)[i] = nm;
                const float inv = (nm > 1e-12f) ? (1.0f / nm) : 0.0f;
                for (uint32_t d = 0; d < dim; ++d) dst[d] = src[d] * inv;
            }
            pos_counts[tid] += pc;
        },
        16
    );
    if (!count_pos) return 0.0;
    uint64_t tot = 0;
    for (uint64_t c : pos_counts) tot += c;
    return static_cast<double>(tot) / static_cast<double>(n * static_cast<size_t>(dim));
}

// Blocked AVX2 SGEMM top-k_nn Euclidean distances on L2-normalized S (returns sorted ascending per row).
inline void knn_dists_on_normalized(
    const std::vector<float>& S,
    size_t n,
    uint32_t dim,
    size_t k_nn,
    size_t threads,
    std::vector<float>& out_dists
) {
    const GemmPanelFn gemm_panel_fn = resolve_gemm_panel_fn();
    deglib::distances::FloatSpace l2_space(dim, deglib::distances::Metric::FP32_L2);
    const auto dist_func = l2_space.get_dist_func();
    const void* param = l2_space.get_dist_func_param();
    const float tol_factor = (std::max)(2e-3f, static_cast<float>(dim) * 1e-6f);
    const size_t num_q_tiles = (n + kTileM - 1) / kTileM;
    const size_t actual_threads = (std::min)(threads, num_q_tiles);
    std::vector<TileScratch> scratch(actual_threads);
    out_dists.assign(n * k_nn, (std::numeric_limits<float>::infinity)());

    deglib::concurrent::parallel_batch_for(
        size_t{0}, num_q_tiles, actual_threads,
        [&](size_t begin, size_t end, size_t tid) {
            TileScratch& sc = scratch[tid];
            std::vector<float> row_d2(kTileM * k_nn);
            for (size_t qt = begin; qt < end; ++qt) {
                const size_t i0 = qt * kTileM;
                const size_t bm = (std::min)(kTileM, n - i0);
                const size_t bm_pad = (bm + 3) & ~size_t{3};
                std::fill(row_d2.begin(), row_d2.end(), (std::numeric_limits<float>::infinity)());
                float kth_d2[kTileM];
                std::fill_n(kth_d2, bm, (std::numeric_limits<float>::infinity)());

                for (size_t j0 = 0; j0 < n; j0 += kTileN) {
                    const size_t bn = (std::min)(kTileN, n - j0);
                    const size_t bn_pad = (bn + 3) & ~size_t{3};
                    std::fill_n(sc.c_tile.data(), bm_pad * kTileN, 0.0f);
                    for (size_t k0 = 0; k0 < dim; k0 += kTileK) {
                        const size_t bk = (std::min)(static_cast<size_t>(dim) - k0, kTileK);
                        const size_t bk_pad = (bk + 15) & ~size_t{15};
                        pack_panel(S.data(), dim, i0, bm, k0, bk, bm_pad, bk_pad, sc.a_panel.data());
                        pack_panel(S.data(), dim, j0, bn, k0, bk, bn_pad, bk_pad, sc.b_panel.data());
                        gemm_panel_fn(sc.a_panel.data(), bm_pad, sc.b_panel.data(), bn_pad, bk_pad, sc.c_tile.data());
                    }
                    for (size_t r = 0; r < bm; ++r) {
                        const size_t qi = i0 + r;
                        float* rd = row_d2.data() + r * k_nn;
                        float worst = kth_d2[r];
                        const float* c_row = sc.c_tile.data() + r * kTileN;
                        const float* q_ptr = S.data() + qi * dim;
                        const float err_margin = tol_factor * 2.0f;
                        for (size_t c = 0; c < bn; ++c) {
                            const size_t db_row = j0 + c;
                            if (db_row == qi) continue;
                            const float d2_approx = 2.0f - 2.0f * c_row[c];
                            if (d2_approx <= worst + err_margin) {
                                const float d2 = dist_func(q_ptr, S.data() + db_row * dim, param);
                                if (d2 < worst) {
                                    if (k_nn == 1) {
                                        rd[0] = d2;
                                        worst = d2;
                                    } else {
                                        size_t pos = k_nn - 1;
                                        while (pos > 0 && rd[pos - 1] > d2) {
                                            rd[pos] = rd[pos - 1];
                                            --pos;
                                        }
                                        rd[pos] = d2;
                                        worst = rd[k_nn - 1];
                                    }
                                }
                            }
                        }
                        kth_d2[r] = worst;
                    }
                }
                for (size_t r = 0; r < bm; ++r) {
                    const float* rd = row_d2.data() + r * k_nn;
                    float* dst = out_dists.data() + (i0 + r) * k_nn;
                    for (size_t k = 0; k < k_nn; ++k) dst[k] = std::sqrt((std::max)(0.0f, rd[k]));
                }
            }
        },
        1
    );
}

struct SphericalKm16Stats {
    double var_expl = 0.0;
    double ch_index = 0.0;
};

// Spherical 16-means on S[:min(n, 2000)]; k-means++ initialization keeps cluster metrics stable across seeds.
inline SphericalKm16Stats compute_spherical_km16(const std::vector<float>& S, size_t n, uint32_t dim, uint32_t seed, bool kmeans_pp) {
    const size_t n_km = (std::min)(n, size_t{2000});
    constexpr size_t K_KM = 16;
    if (n_km <= K_KM) return {};

    std::mt19937 km_rng(seed);
    std::vector<float> cents(K_KM * dim, 0.0f);
    if (kmeans_pp) {
        std::uniform_int_distribution<size_t> u0(0, n_km - 1);
        std::copy_n(S.data() + u0(km_rng) * dim, dim, cents.data());
        std::vector<double> min_d2(n_km, 1e30);
        for (size_t c = 1; c < K_KM; ++c) {
            const float* prev_c = cents.data() + (c - 1) * dim;
            double sum_d2 = 0.0;
            for (size_t i = 0; i < n_km; ++i) {
                const float* row = S.data() + i * dim;
                double d2 = 0.0;
                for (uint32_t d = 0; d < dim; ++d) {
                    const double diff = static_cast<double>(row[d]) - static_cast<double>(prev_c[d]);
                    d2 += diff * diff;
                }
                if (d2 < min_d2[i]) min_d2[i] = d2;
                sum_d2 += min_d2[i];
            }
            size_t chosen = 0;
            if (sum_d2 > 1e-18) {
                std::uniform_real_distribution<double> ur(0.0, sum_d2);
                const double target = ur(km_rng);
                double acc = 0.0;
                for (size_t i = 0; i < n_km; ++i) {
                    acc += min_d2[i];
                    if (acc >= target) {
                        chosen = i;
                        break;
                    }
                }
            } else {
                chosen = u0(km_rng);
            }
            std::copy_n(S.data() + chosen * dim, dim, cents.data() + c * dim);
        }
    } else {
        std::vector<size_t> perm_km(n_km);
        std::iota(perm_km.begin(), perm_km.end(), size_t{0});
        for (size_t i = 0; i < K_KM; ++i) {
            std::uniform_int_distribution<size_t> dist(i, n_km - 1);
            std::swap(perm_km[i], perm_km[dist(km_rng)]);
            std::copy_n(S.data() + perm_km[i] * dim, dim, cents.data() + i * dim);
        }
    }

    std::vector<uint32_t> lbl(n_km, 0);
    std::vector<double> new_cents(K_KM * dim, 0.0);
    std::vector<uint32_t> counts(K_KM, 0);
    const int iters = kmeans_pp ? 10 : 8;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < n_km; ++i) {
            const float* row = S.data() + i * dim;
            float best_s = -1e30f;
            uint32_t best_c = 0;
            for (size_t c = 0; c < K_KM; ++c) {
                const float* cp = cents.data() + c * dim;
                float s = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) s += row[d] * cp[d];
                if (s > best_s) {
                    best_s = s;
                    best_c = static_cast<uint32_t>(c);
                }
            }
            lbl[i] = best_c;
        }
        std::fill(new_cents.begin(), new_cents.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t i = 0; i < n_km; ++i) {
            const uint32_t c = lbl[i];
            ++counts[c];
            const float* row = S.data() + i * dim;
            double* dst = new_cents.data() + c * dim;
            for (uint32_t d = 0; d < dim; ++d) dst[d] += static_cast<double>(row[d]);
        }
        for (size_t c = 0; c < K_KM; ++c) {
            if (counts[c] == 0) continue;
            float* cp = cents.data() + c * dim;
            const double* src = new_cents.data() + c * dim;
            double n2 = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                const double v = src[d] / static_cast<double>(counts[c]);
                cp[d] = static_cast<float>(v);
                n2 += v * v;
            }
            const float inv = (n2 > 1e-24) ? static_cast<float>(1.0 / std::sqrt(n2)) : 0.0f;
            for (uint32_t d = 0; d < dim; ++d) cp[d] *= inv;
        }
    }

    std::vector<double> col_mean(dim, 0.0);
    for (size_t i = 0; i < n_km; ++i) {
        const float* row = S.data() + i * dim;
        float best_s = -1e30f;
        uint32_t best_c = 0;
        for (size_t c = 0; c < K_KM; ++c) {
            const float* cp = cents.data() + c * dim;
            float s = 0.0f;
            for (uint32_t d = 0; d < dim; ++d) s += row[d] * cp[d];
            if (s > best_s) {
                best_s = s;
                best_c = static_cast<uint32_t>(c);
            }
        }
        lbl[i] = best_c;
        for (uint32_t d = 0; d < dim; ++d) col_mean[d] += static_cast<double>(row[d]);
    }
    for (uint32_t d = 0; d < dim; ++d) col_mean[d] /= static_cast<double>(n_km);

    double tot_var = 0.0, within = 0.0;
    for (size_t i = 0; i < n_km; ++i) {
        const float* row = S.data() + i * dim;
        const float* cp = cents.data() + lbl[i] * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            const double dt = static_cast<double>(row[d]) - col_mean[d];
            const double dw = static_cast<double>(row[d]) - static_cast<double>(cp[d]);
            tot_var += dt * dt;
            within += dw * dw;
        }
    }
    const double between = (std::max)(0.0, tot_var - within);
    SphericalKm16Stats out;
    out.var_expl = (std::max)(0.0, 1.0 - within / (std::max)(tot_var, 1e-9));
    out.ch_index = (within > 1e-12) ? (between / (K_KM - 1.0)) / (within / (n_km - K_KM)) : 0.0;
    return out;
}

inline void center_submatrix(const float* data, size_t m4, uint32_t dim, std::vector<float>& Xs, double& frob2) {
    Xs.resize(m4 * dim);
    std::vector<double> mean_x(dim, 0.0);
    for (size_t i = 0; i < m4; ++i) {
        const float* src = data + i * dim;
        for (uint32_t d = 0; d < dim; ++d) mean_x[d] += static_cast<double>(src[d]);
    }
    for (uint32_t d = 0; d < dim; ++d) mean_x[d] /= static_cast<double>(m4);
    frob2 = 0.0;
    for (size_t i = 0; i < m4; ++i) {
        const float* src = data + i * dim;
        float* dst = Xs.data() + i * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            const float v = src[d] - static_cast<float>(mean_x[d]);
            dst[d] = v;
            frob2 += static_cast<double>(v) * static_cast<double>(v);
        }
    }
}

inline void subspace_top_ratios(
    const std::vector<float>& Xs,
    size_t m4,
    uint32_t dim,
    double frob2,
    size_t r_sub_req,
    size_t threads,
    uint32_t seed,
    double& out_top1,
    double& out_top10,
    double& out_top32
) {
    const size_t r_sub = (std::min)(static_cast<size_t>(dim), r_sub_req);
    std::vector<float> Q_sub(dim * r_sub);
    std::mt19937 sub_rng(seed + 777);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    for (float& v : Q_sub) v = norm_dist(sub_rng);
    mgs_orthonormalize(Q_sub, dim, r_sub);

    std::vector<float> Z_sub(m4 * r_sub, 0.0f);
    std::vector<std::vector<float>> q_local(threads, std::vector<float>(dim * r_sub, 0.0f));
    for (int it = 0; it < 6; ++it) {
        deglib::concurrent::parallel_batch_for(
            size_t{0}, m4, threads,
            [&](size_t begin, size_t end, size_t /*tid*/) {
                for (size_t i = begin; i < end; ++i) {
                    const float* xrow = Xs.data() + i * dim;
                    float* zrow = Z_sub.data() + i * r_sub;
                    std::fill_n(zrow, r_sub, 0.0f);
                    for (uint32_t d = 0; d < dim; ++d) {
                        const float xv = xrow[d];
                        const float* qrow = Q_sub.data() + d * r_sub;
                        for (size_t c = 0; c < r_sub; ++c) zrow[c] += xv * qrow[c];
                    }
                }
            },
            16
        );
        for (auto& ql : q_local) std::fill(ql.begin(), ql.end(), 0.0f);
        deglib::concurrent::parallel_batch_for(
            size_t{0}, m4, threads,
            [&](size_t begin, size_t end, size_t tid) {
                float* ql = q_local[tid].data();
                for (size_t i = begin; i < end; ++i) {
                    const float* xrow = Xs.data() + i * dim;
                    const float* zrow = Z_sub.data() + i * r_sub;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const float xv = xrow[d];
                        float* dst = ql + d * r_sub;
                        for (size_t c = 0; c < r_sub; ++c) dst[c] += xv * zrow[c];
                    }
                }
            },
            16
        );
        std::fill(Q_sub.begin(), Q_sub.end(), 0.0f);
        for (const auto& ql : q_local) {
            for (size_t idx = 0; idx < Q_sub.size(); ++idx) Q_sub[idx] += ql[idx];
        }
        mgs_orthonormalize(Q_sub, dim, r_sub);
    }
    deglib::concurrent::parallel_batch_for(
        size_t{0}, m4, threads,
        [&](size_t begin, size_t end, size_t /*tid*/) {
            for (size_t i = begin; i < end; ++i) {
                const float* xrow = Xs.data() + i * dim;
                float* zrow = Z_sub.data() + i * r_sub;
                std::fill_n(zrow, r_sub, 0.0f);
                for (uint32_t d = 0; d < dim; ++d) {
                    const float xv = xrow[d];
                    const float* qrow = Q_sub.data() + d * r_sub;
                    for (size_t c = 0; c < r_sub; ++c) zrow[c] += xv * qrow[c];
                }
            }
        },
        16
    );
    std::vector<double> G_sub(r_sub * r_sub, 0.0);
    for (size_t i = 0; i < m4; ++i) {
        const float* zrow = Z_sub.data() + i * r_sub;
        for (size_t a = 0; a < r_sub; ++a) {
            for (size_t b = a; b < r_sub; ++b) {
                G_sub[a * r_sub + b] += static_cast<double>(zrow[a]) * static_cast<double>(zrow[b]);
            }
        }
    }
    for (size_t a = 0; a < r_sub; ++a) {
        for (size_t b = 0; b < a; ++b) G_sub[a * r_sub + b] = G_sub[b * r_sub + a];
    }
    std::vector<double> evals;
    jacobi_eigenvalues_sym(G_sub, r_sub, evals);
    const double inv_f = 1.0 / (std::max)(frob2, 1e-18);
    double s1 = 0.0, s10 = 0.0, s32 = 0.0;
    for (size_t i = 0; i < r_sub; ++i) {
        const double ev = (std::max)(0.0, evals[i]);
        if (i < 1) s1 += ev;
        if (i < 10) s10 += ev;
        if (i < 32) s32 += ev;
    }
    out_top1 = s1 * inv_f;
    out_top10 = s10 * inv_f;
    out_top32 = s32 * inv_f;
}

inline double compute_eig_pr_gram(const std::vector<float>& Xs, size_t m4, uint32_t dim, double frob2, size_t threads) {
    const GemmPanelFn gemm_panel_fn = resolve_gemm_panel_fn();
    const size_t num_q_tiles = (m4 + kTileM - 1) / kTileM;
    const size_t actual_threads = (std::min)(threads, num_q_tiles);
    std::vector<TileScratch> scratch(actual_threads);
    std::vector<double> sum_sq(actual_threads, 0.0);

    deglib::concurrent::parallel_batch_for(
        size_t{0}, num_q_tiles, actual_threads,
        [&](size_t begin, size_t end, size_t tid) {
            TileScratch& sc = scratch[tid];
            double local_sq = 0.0;
            for (size_t qt = begin; qt < end; ++qt) {
                const size_t i0 = qt * kTileM;
                const size_t bm = (std::min)(kTileM, m4 - i0);
                const size_t bm_pad = (bm + 3) & ~size_t{3};
                for (size_t j0 = i0; j0 < m4; j0 += kTileN) {
                    const size_t bn = (std::min)(kTileN, m4 - j0);
                    const size_t bn_pad = (bn + 3) & ~size_t{3};
                    std::fill_n(sc.c_tile.data(), bm_pad * kTileN, 0.0f);
                    for (size_t k0 = 0; k0 < dim; k0 += kTileK) {
                        const size_t bk = (std::min)(static_cast<size_t>(dim) - k0, kTileK);
                        const size_t bk_pad = (bk + 15) & ~size_t{15};
                        pack_panel(Xs.data(), dim, i0, bm, k0, bk, bm_pad, bk_pad, sc.a_panel.data());
                        pack_panel(Xs.data(), dim, j0, bn, k0, bk, bn_pad, bk_pad, sc.b_panel.data());
                        gemm_panel_fn(sc.a_panel.data(), bm_pad, sc.b_panel.data(), bn_pad, bk_pad, sc.c_tile.data());
                    }
                    const double weight = (j0 == i0) ? 1.0 : 2.0;
                    for (size_t r = 0; r < bm; ++r) {
                        const float* c_row = sc.c_tile.data() + r * kTileN;
                        for (size_t c = 0; c < bn; ++c) {
                            const double v = static_cast<double>(c_row[c]);
                            local_sq += weight * v * v;
                        }
                    }
                }
            }
            sum_sq[tid] += local_sq;
        },
        1
    );
    double tot_sq = 0.0;
    for (double s : sum_sq) tot_sq += s;
    return (frob2 * frob2) / (std::max)(tot_sq, 1e-18);
}

inline std::vector<float> sample_fp32_rows(std::span<const float> data, uint32_t dim, size_t sample_count, uint32_t seed) {
    const size_t total_rows = data.size() / dim;
    const size_t n = (sample_count == 0 || sample_count >= total_rows) ? total_rows : sample_count;
    const std::vector<size_t> indices = sample_distinct_indices(total_rows, n, seed);
    std::vector<float> sampled(indices.size() * static_cast<size_t>(dim));
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(float);
    for (size_t i = 0; i < indices.size(); ++i) {
        std::memcpy(sampled.data() + i * dim, data.data() + indices[i] * dim, row_bytes);
    }
    return sampled;
}

inline std::vector<float> sample_fp16_rows(std::span<const uint16_t> data_fp16, uint32_t dim, size_t sample_count, uint32_t seed) {
    const size_t total_rows = data_fp16.size() / dim;
    const size_t n = (sample_count == 0 || sample_count >= total_rows) ? total_rows : sample_count;
    const std::vector<size_t> indices = sample_distinct_indices(total_rows, n, seed);
    std::vector<float> sampled(indices.size() * static_cast<size_t>(dim));
    for (size_t i = 0; i < indices.size(); ++i) {
        deglib::distances::fp16::fp16_to_floats(data_fp16.data() + indices[i] * dim, sampled.data() + i * dim, dim);
    }
    return sampled;
}

inline QuantizationPrediction predict_quantization_k5_from_sampled(
    const std::vector<float>& sampled,
    uint32_t dim,
    deglib::distances::Metric metric,
    uint32_t seed,
    size_t num_threads,
    const K5QuantizationCalibration& cal
) {
    const size_t n = sampled.size() / dim;
    DatasetProfile prof{};
    prof.sample_count = static_cast<uint32_t>(n);
    prof.dim = dim;
    prof.metric = metric;

    if (metric.get_distance_kind() != deglib::distances::MetricDistanceKind::InnerProduct || n < 2) {
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }

    const size_t threads = resolve_thread_count(num_threads);
    std::vector<float> norms;
    std::vector<float> S;
    const double coord_pos_frac = normalize_rows_l2(sampled.data(), n, dim, threads, S, &norms, true);

    double sum_norm = 0.0;
    float max_norm = 0.0f;
    for (float nm : norms) {
        sum_norm += static_cast<double>(nm);
        if (nm > max_norm) max_norm = nm;
    }
    if (max_norm <= 1e-12f) {
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    const double mean_norm = sum_norm / static_cast<double>(n);
    const double M = static_cast<double>(max_norm) * 1.0001;
    double sum_extra = 0.0;
    for (float nm : norms) {
        const double ex2 = (std::max)(0.0, M * M - static_cast<double>(nm) * static_cast<double>(nm));
        sum_extra += std::sqrt(ex2);
    }
    const double mips_extra_ratio = (sum_extra / static_cast<double>(n)) / (std::max)(mean_norm, 1e-9);
    const double log1p_mips_extra_ratio = std::log1p((std::max)(mips_extra_ratio, 0.0));

    std::vector<float> d1_f;
    knn_dists_on_normalized(S, n, dim, 1, threads, d1_f);
    std::vector<double> d1_vals(n);
    double sum_d1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        d1_vals[i] = static_cast<double>(d1_f[i]);
        sum_d1 += d1_vals[i];
    }
    if (sum_d1 <= 1e-12 * static_cast<double>(n)) {
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    std::sort(d1_vals.begin(), d1_vals.end());
    const double d1_p25 = sorted_percentile(d1_vals, 0.25);
    const double d1_p50 = sorted_percentile(d1_vals, 0.50);
    const double d1_p75 = sorted_percentile(d1_vals, 0.75);
    const double d1_p95 = sorted_percentile(d1_vals, 0.95);
    const double d1_tail_weight = (d1_p95 - d1_p50) / (std::max)(d1_p75 - d1_p25, 1e-9);

    const double km16_var_expl = compute_spherical_km16(S, n, dim, seed, false).var_expl;

    const size_t m4 = (std::min)(n, size_t{4000});
    std::vector<float> Xs;
    double frob2 = 0.0;
    center_submatrix(sampled.data(), m4, dim, Xs, frob2);
    if (frob2 <= 1e-12) {
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    double t1_r = 0.0, t10_r = 0.0, top32_var_ratio = 0.0;
    subspace_top_ratios(Xs, m4, dim, frob2, 44, threads, seed, t1_r, t10_r, top32_var_ratio);

    const double raw_f[5] = {coord_pos_frac, d1_tail_weight, km16_var_expl, log1p_mips_extra_ratio, top32_var_ratio};
    double log2_score = cal.w[0];
    for (int i = 0; i < 5; ++i) {
        log2_score += cal.w[i + 1] * std::tanh(((raw_f[i] - cal.mu[i]) / cal.sd[i]) / cal.tanh_temp);
    }
    const float adv_pct = static_cast<float>((std::exp2(log2_score) - 1.0) * 100.0);
    const auto q_metric = (adv_pct >= 0.0f) ? deglib::distances::Metric::Uint7_L2 : deglib::distances::Metric::Int8_InnerProduct;
    return QuantizationPrediction{q_metric, adv_pct, prof};
}

inline QuantizationPrediction predict_quantization_a1a2_from_sampled(
    const std::vector<float>& sampled,
    uint32_t dim,
    bool is_cosine,
    deglib::distances::Metric metric,
    uint32_t seed,
    size_t num_threads,
    const NormalizedQuantizationCalibration& cal
) {
    const size_t n = sampled.size() / dim;
    DatasetProfile prof{};
    prof.sample_count = static_cast<uint32_t>(n);
    prof.dim = dim;
    prof.metric = metric;

    if (metric.get_distance_kind() != deglib::distances::MetricDistanceKind::InnerProduct || n < 33) {
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }

    const size_t threads = resolve_thread_count(num_threads);
    std::vector<float> S;
    normalize_rows_l2(sampled.data(), n, dim, threads, S, nullptr, false);

    std::vector<float> knn_d;
    knn_dists_on_normalized(S, n, dim, 32, threads, knn_d);

    std::vector<double> d1_vals(n);
    std::vector<double> lid16_vals(n);
    double sum_d1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float* row = knn_d.data() + i * 32;
        const double d1 = static_cast<double>(row[0]);
        d1_vals[i] = d1;
        sum_d1 += d1;
        const double d16 = (std::max)(static_cast<double>(row[15]), 1e-12);
        double log_sum = 0.0;
        for (size_t j = 0; j < 15; ++j) {
            const double dj = (std::max)(static_cast<double>(row[j]), 1e-12);
            log_sum += std::log(dj / d16);
        }
        const double mean_log = (std::min)(-1e-6, log_sum / 15.0);
        lid16_vals[i] = -1.0 / mean_log;
    }
    const double mean_d1 = sum_d1 / static_cast<double>(n);
    if (mean_d1 <= 1e-12) {
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    double m2 = 0.0, m3 = 0.0;
    for (double v : d1_vals) {
        const double c = v - mean_d1;
        m2 += c * c;
        m3 += c * c * c;
    }
    m2 /= static_cast<double>(n);
    m3 /= static_cast<double>(n);
    const double d1_skew = (m2 > 1e-18) ? (m3 / std::pow(m2, 1.5)) : 0.0;

    std::sort(lid16_vals.begin(), lid16_vals.end());
    const double mle_lid16_tail = sorted_percentile(lid16_vals, 0.95) / (std::max)(sorted_percentile(lid16_vals, 0.50), 1e-12);

    deglib::distances::FloatSpace l2_space(dim, deglib::distances::Metric::FP32_L2);
    const auto dist_func = l2_space.get_dist_func();
    const void* param = l2_space.get_dist_func_param();
    std::mt19937_64 rng_rand(seed);
    std::uniform_int_distribution<size_t> udist(0, n - 1);
    double sum_drand = 0.0;
    for (size_t p = 0; p < kDefaultRandomPairs; ++p) {
        const size_t a = udist(rng_rand);
        const size_t b = udist(rng_rand);
        const float d2 = dist_func(S.data() + a * dim, S.data() + b * dim, param);
        sum_drand += std::sqrt(static_cast<double>((std::max)(0.0f, d2)));
    }
    const double rc_1_32 = (sum_drand / static_cast<double>(kDefaultRandomPairs)) / (std::max)(mean_d1, 1e-12);

    const double ch = compute_spherical_km16(S, n, dim, seed, true).ch_index;
    const double log1p_ch = std::log1p((std::max)(0.0, ch));

    const size_t m4 = (std::min)(n, size_t{4000});
    std::vector<float> Xs;
    double frob2 = 0.0;
    center_submatrix(S.data(), m4, dim, Xs, frob2);
    if (frob2 <= 1e-12) {
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    const double s_eig_pr_abs = compute_eig_pr_gram(Xs, m4, dim, frob2, threads);
    double s_top1 = 0.0, s_top10 = 0.0, s_top32 = 0.0;
    subspace_top_ratios(Xs, m4, dim, frob2, 16, threads, seed, s_top1, s_top10, s_top32);

    const double cos_f = is_cosine ? 1.0 : 0.0;
    const double f1[5] = {mle_lid16_tail, rc_1_32, log1p_ch, s_eig_pr_abs, cos_f};
    const double f2[5] = {d1_skew, log1p_ch, s_top1, s_top10, cos_f};
    double s1 = cal.w1[0], s2 = cal.w2[0];
    for (int i = 0; i < 5; ++i) {
        s1 += cal.w1[i + 1] * ((f1[i] - cal.mu1[i]) / cal.sd1[i]);
        s2 += cal.w2[i + 1] * ((f2[i] - cal.mu2[i]) / cal.sd2[i]);
    }
    const double log2_score = std::clamp(0.5 * (s1 + s2), -10.0, 10.0);
    const float adv_pct = static_cast<float>((std::exp2(log2_score) - 1.0) * 100.0);
    const auto q_metric = (adv_pct >= 0.0f) ? deglib::distances::Metric::Uint7_L2 : deglib::distances::Metric::Int8_InnerProduct;
    return QuantizationPrediction{q_metric, adv_pct, prof};
}

}  // namespace detail

/**
 * Measure the base-only geometric DatasetProfile from contiguous FP32 feature vectors.
 */
[[nodiscard]] inline DatasetProfile measure_dataset_profile(
    std::span<const float> data,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0
) {
    if (dim == 0) {
        throw std::invalid_argument("measure_dataset_profile: dim must be greater than 0");
    }
    if (data.size() % dim != 0) {
        throw std::invalid_argument("measure_dataset_profile: data size must be an exact multiple of dim");
    }

    const size_t total_rows = data.size() / dim;
    if (total_rows < 2) {
        return DatasetProfile{static_cast<uint32_t>(total_rows), dim, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, metric};
    }

    const size_t n = (sample_count == 0 || sample_count >= total_rows) ? total_rows : sample_count;
    const std::vector<size_t> indices = detail::sample_distinct_indices(total_rows, n, seed);

    std::vector<float> sampled(indices.size() * static_cast<size_t>(dim));
    const size_t row_bytes = static_cast<size_t>(dim) * sizeof(float);
    for (size_t i = 0; i < indices.size(); ++i) {
        std::memcpy(sampled.data() + i * dim, data.data() + indices[i] * dim, row_bytes);
    }

    return detail::measure_profile_from_sampled(sampled, dim, metric, query_count, seed, num_threads);
}

/**
 * Measure the base-only geometric DatasetProfile from contiguous FP16 (uint16_t bit pattern) vectors.
 * Converts only the sampled rows to FP32.
 */
[[nodiscard]] inline DatasetProfile measure_dataset_profile(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0
) {
    if (dim == 0) {
        throw std::invalid_argument("measure_dataset_profile: dim must be greater than 0");
    }
    if (data_fp16.size() % dim != 0) {
        throw std::invalid_argument("measure_dataset_profile: data size must be an exact multiple of dim");
    }

    const size_t total_rows = data_fp16.size() / dim;
    if (total_rows < 2) {
        return DatasetProfile{static_cast<uint32_t>(total_rows), dim, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, metric};
    }

    const size_t n = (sample_count == 0 || sample_count >= total_rows) ? total_rows : sample_count;
    const std::vector<size_t> indices = detail::sample_distinct_indices(total_rows, n, seed);

    std::vector<float> sampled(indices.size() * static_cast<size_t>(dim));
    for (size_t i = 0; i < indices.size(); ++i) {
        deglib::distances::fp16::fp16_to_floats(data_fp16.data() + indices[i] * dim, sampled.data() + i * dim, dim);
    }

    return detail::measure_profile_from_sampled(sampled, dim, metric, query_count, seed, num_threads);
}

/**
 * Predict whether OptimizationTarget::LowLID outperforms OptimizationTarget::StreamingData
 * from a precomputed DatasetProfile.
 */
[[nodiscard]] inline TargetPrediction
predict_optimization_target(const DatasetProfile& profile, const TargetModelCalibration& cal = kDefaultTargetCalibration) noexcept {
    if (profile.sample_count < 2 || profile.pp_iqr <= 0.0f || !std::isfinite(profile.pp_iqr) || !std::isfinite(profile.d16_skew) ||
        !std::isfinite(profile.ball_cnt_p90_p10_r15) || !std::isfinite(profile.hadamard_kurt)) {
        return TargetPrediction{deglib::builder::OptimizationTarget::StreamingData, 0.0f, profile};
    }

    const double z_d16 = (static_cast<double>(profile.d16_skew) - cal.mu_d16_skew) / cal.sd_d16_skew;
    const double z_ball = (static_cast<double>(profile.ball_cnt_p90_p10_r15) - cal.mu_ball_cnt_p90_p10_r15) / cal.sd_ball_cnt_p90_p10_r15;
    const double z_pp = (static_cast<double>(profile.pp_iqr) - cal.mu_pp_iqr) / cal.sd_pp_iqr;
    const double z_had = (static_cast<double>(profile.hadamard_kurt) - cal.mu_hadamard_kurt) / cal.sd_hadamard_kurt;

    const double logit = cal.w_intercept + cal.w_d16_skew * z_d16 + cal.w_ball_cnt_p90_p10_r15 * z_ball + cal.w_pp_iqr * z_pp + cal.w_hadamard_kurt * z_had;
    const double log2_score = cal.log2_scale * std::tanh(logit / cal.tanh_temp);
    const float adv_pct = static_cast<float>((std::exp2(log2_score) - 1.0) * 100.0);
    const auto target = (adv_pct > 0.0f) ? deglib::builder::OptimizationTarget::LowLID : deglib::builder::OptimizationTarget::StreamingData;
    return TargetPrediction{target, adv_pct, profile};
}

/**
 * Measure a DatasetProfile and predict the optimal graph construction target from FP32 vectors.
 */
[[nodiscard]] inline TargetPrediction predict_optimization_target(
    std::span<const float> data,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const TargetModelCalibration& cal = kDefaultTargetCalibration
) {
    const DatasetProfile profile = measure_dataset_profile(data, dim, metric, sample_count, query_count, seed, num_threads);
    return predict_optimization_target(profile, cal);
}

/**
 * Measure a DatasetProfile and predict the optimal graph construction target from FP16 vectors.
 */
[[nodiscard]] inline TargetPrediction predict_optimization_target(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const TargetModelCalibration& cal = kDefaultTargetCalibration
) {
    const DatasetProfile profile = measure_dataset_profile(data_fp16, dim, metric, sample_count, query_count, seed, num_threads);
    return predict_optimization_target(profile, cal);
}

/**
 * Predict the optimal quantizer metric (Uint7_L2 vs Int8_InnerProduct) from raw, unnormalized FP32 vectors
 * using the K5-Top model (intended for `deglib.build(..., compress="8bit")`).
 *
 * Non-InnerProduct (L2) datasets unconditionally select Metric::Uint7_L2.
 * Do NOT pass already L2-normalized 32-bit graph vectors here (K5-Top relies on raw vector norms via
 * `log1p_mips_extra_ratio` and raw covariance via `top32_var_ratio`); use `predict_quantization_normalized`
 * for `deglib.compress(graph, compress="8bit")` instead.
 */
[[nodiscard]] inline QuantizationPrediction predict_quantization(
    std::span<const float> data,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t /*query_count*/ = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const K5QuantizationCalibration& cal = kDefaultK5QuantizationCalibration
) {
    if (dim == 0) {
        throw std::invalid_argument("predict_quantization: dim must be greater than 0");
    }
    if (data.size() % dim != 0) {
        throw std::invalid_argument("predict_quantization: data size must be an exact multiple of dim");
    }
    const size_t total_rows = data.size() / dim;
    if (metric.get_distance_kind() != deglib::distances::MetricDistanceKind::InnerProduct || total_rows < 2) {
        DatasetProfile prof{static_cast<uint32_t>(total_rows), dim, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, metric};
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    const std::vector<float> sampled = detail::sample_fp32_rows(data, dim, sample_count, seed);
    return detail::predict_quantization_k5_from_sampled(sampled, dim, metric, seed, num_threads, cal);
}

/**
 * Predict the optimal quantizer metric (Uint7_L2 vs Int8_InnerProduct) from raw, unnormalized FP16 vectors
 * using the K5-Top model (intended for `deglib.build(..., compress="8bit")`).
 */
[[nodiscard]] inline QuantizationPrediction predict_quantization(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t /*query_count*/ = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const K5QuantizationCalibration& cal = kDefaultK5QuantizationCalibration
) {
    if (dim == 0) {
        throw std::invalid_argument("predict_quantization: dim must be greater than 0");
    }
    if (data_fp16.size() % dim != 0) {
        throw std::invalid_argument("predict_quantization: data size must be an exact multiple of dim");
    }
    const size_t total_rows = data_fp16.size() / dim;
    if (metric.get_distance_kind() != deglib::distances::MetricDistanceKind::InnerProduct || total_rows < 2) {
        DatasetProfile prof{static_cast<uint32_t>(total_rows), dim, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, metric};
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    const std::vector<float> sampled = detail::sample_fp16_rows(data_fp16, dim, sample_count, seed);
    return detail::predict_quantization_k5_from_sampled(sampled, dim, metric, seed, num_threads, cal);
}

/**
 * Predict the optimal quantizer metric (Uint7_L2 vs Int8_InnerProduct) from already L2-normalized FP32 vectors
 * (e.g., extracted from an existing 32-bit graph during `deglib.compress(graph, compress="8bit")`)
 * using the 100% L2-normalization-invariant A1+A2 ensemble.
 *
 * @param is_cosine true if the graph carries an L2NormalizeTransform (original inputs were unnormalized cosine);
 *                  false if inputs were already unit-normalized.
 */
[[nodiscard]] inline QuantizationPrediction predict_quantization_normalized(
    std::span<const float> data,
    uint32_t dim,
    bool is_cosine,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_InnerProduct,
    size_t sample_count = kDefaultSampleCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const NormalizedQuantizationCalibration& cal = kDefaultNormalizedQuantizationCalibration
) {
    if (dim == 0) {
        throw std::invalid_argument("predict_quantization_normalized: dim must be greater than 0");
    }
    if (data.size() % dim != 0) {
        throw std::invalid_argument("predict_quantization_normalized: data size must be an exact multiple of dim");
    }
    const size_t total_rows = data.size() / dim;
    if (metric.get_distance_kind() != deglib::distances::MetricDistanceKind::InnerProduct || total_rows < 33) {
        DatasetProfile prof{static_cast<uint32_t>(total_rows), dim, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, metric};
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    const std::vector<float> sampled = detail::sample_fp32_rows(data, dim, sample_count, seed);
    return detail::predict_quantization_a1a2_from_sampled(sampled, dim, is_cosine, metric, seed, num_threads, cal);
}

/**
 * Predict the optimal quantizer metric (Uint7_L2 vs Int8_InnerProduct) from already L2-normalized FP16 vectors
 * using the 100% L2-normalization-invariant A1+A2 ensemble.
 */
[[nodiscard]] inline QuantizationPrediction predict_quantization_normalized(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    bool is_cosine,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_InnerProduct,
    size_t sample_count = kDefaultSampleCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const NormalizedQuantizationCalibration& cal = kDefaultNormalizedQuantizationCalibration
) {
    if (dim == 0) {
        throw std::invalid_argument("predict_quantization_normalized: dim must be greater than 0");
    }
    if (data_fp16.size() % dim != 0) {
        throw std::invalid_argument("predict_quantization_normalized: data size must be an exact multiple of dim");
    }
    const size_t total_rows = data_fp16.size() / dim;
    if (metric.get_distance_kind() != deglib::distances::MetricDistanceKind::InnerProduct || total_rows < 33) {
        DatasetProfile prof{static_cast<uint32_t>(total_rows), dim, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, metric};
        return QuantizationPrediction{deglib::distances::Metric::Uint7_L2, 0.0f, prof};
    }
    const std::vector<float> sampled = detail::sample_fp16_rows(data_fp16, dim, sample_count, seed);
    return detail::predict_quantization_a1a2_from_sampled(sampled, dim, is_cosine, metric, seed, num_threads, cal);
}

}  // namespace deglib::tuning
