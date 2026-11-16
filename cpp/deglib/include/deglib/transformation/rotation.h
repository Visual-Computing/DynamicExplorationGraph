#pragma once

#include "deglib/concurrent.h"
#include "deglib/config.h"
#include "deglib/distance/fp16.h"
#include "deglib/transformation.h"
#include "deglib/utils/cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#if defined(DEGLIB_X86)
    #include <immintrin.h>
#endif

namespace deglib::transformation {

namespace detail {

struct RotationStats {
    float crest_factor = 0.0f;
    float excess_kurtosis = 0.0f;
    bool recommended = false;
};

/**
 * Evaluates coordinate anisotropy on a strided sample of vectors to decide whether
 * orthogonal rotation is beneficial.
 */
inline RotationStats analyze_rotation_benefit(
    const float* data, size_t total_items, uint32_t dim, float kurtosis_threshold = 2.0f, float crest_threshold = 9.5f
) noexcept {
    RotationStats stats;
    if (total_items == 0 || dim == 0) return stats;

    const size_t num_vectors = total_items / dim;
    const size_t stride = std::max<size_t>(1, num_vectors / 10000);
    size_t sample_vecs = 0;

    double sum = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (size_t v_idx = 0; v_idx < num_vectors; v_idx += stride) {
        const float* vec = data + v_idx * dim;
        for (uint32_t j = 0; j < dim; ++j) {
            const float v = vec[j];
            const float av = std::abs(v);
            if (av > max_abs) max_abs = av;
            const double dv = static_cast<double>(v);
            sum += dv;
            sum_sq += dv * dv;
        }
        ++sample_vecs;
    }

    const size_t n = sample_vecs * dim;
    if (n == 0) return stats;

    const double mean = sum / static_cast<double>(n);
    const double rms_norm_sq = sum_sq / static_cast<double>(sample_vecs);
    const double iso_std = std::sqrt(std::max(1e-24, rms_norm_sq / static_cast<double>(dim)));

    double m2 = 0.0;
    double m4 = 0.0;
    for (size_t v_idx = 0; v_idx < num_vectors; v_idx += stride) {
        const float* vec = data + v_idx * dim;
        for (uint32_t j = 0; j < dim; ++j) {
            const double d = static_cast<double>(vec[j]) - mean;
            const double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
    }
    m2 /= static_cast<double>(n);
    m4 /= static_cast<double>(n);

    stats.crest_factor = static_cast<float>(static_cast<double>(max_abs) / iso_std);
    stats.excess_kurtosis = (m2 > 1e-24) ? static_cast<float>(m4 / (m2 * m2) - 3.0) : 0.0f;
    stats.recommended = (stats.excess_kurtosis > kurtosis_threshold) ||
                        (crest_threshold > 0.0f && stats.crest_factor > crest_threshold);
    return stats;
}

inline void fwht_inplace_unscaled_scalar(float* v, uint32_t n) noexcept {
    for (uint32_t h = 1; h < n; h <<= 1) {
        for (uint32_t i = 0; i < n; i += (h << 1)) {
            float* a = v + i;
            float* b = v + i + h;
            for (uint32_t j = 0; j < h; ++j) {
                const float x = a[j];
                const float y = b[j];
                a[j] = x + y;
                b[j] = x - y;
            }
        }
    }
}

#if defined(DEGLIB_X86)
DEGLIB_TARGET_AVX2 inline void fwht_inplace_unscaled_avx2(float* v, uint32_t n) noexcept {
    const __m256 mask1 = _mm256_castsi256_ps(_mm256_setr_epi32(
        0, int(0x80000000u), 0, int(0x80000000u), 0, int(0x80000000u), 0, int(0x80000000u)
    ));
    const __m256 mask2 = _mm256_castsi256_ps(_mm256_setr_epi32(
        0, 0, int(0x80000000u), int(0x80000000u), 0, 0, int(0x80000000u), int(0x80000000u)
    ));
    const __m256 mask4 = _mm256_castsi256_ps(_mm256_setr_epi32(
        0, 0, 0, 0, int(0x80000000u), int(0x80000000u), int(0x80000000u), int(0x80000000u)
    ));

    for (uint32_t i = 0; i < n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);
        x = _mm256_add_ps(_mm256_xor_ps(x, mask1), _mm256_permute_ps(x, 0xB1));
        x = _mm256_add_ps(_mm256_xor_ps(x, mask2), _mm256_permute_ps(x, 0x4E));
        x = _mm256_add_ps(_mm256_xor_ps(x, mask4), _mm256_permute2f128_ps(x, x, 0x01));
        _mm256_storeu_ps(v + i, x);
    }

    for (uint32_t h = 8; h < n; h <<= 1) {
        for (uint32_t i = 0; i < n; i += (h << 1)) {
            float* a = v + i;
            float* b = v + i + h;
            for (uint32_t j = 0; j < h; j += 8) {
                const __m256 vx = _mm256_loadu_ps(a + j);
                const __m256 vy = _mm256_loadu_ps(b + j);
                _mm256_storeu_ps(a + j, _mm256_add_ps(vx, vy));
                _mm256_storeu_ps(b + j, _mm256_sub_ps(vx, vy));
            }
        }
    }
}
#endif

/**
 * Unscaled in-place Fast Walsh-Hadamard transform on a power-of-two block of size n.
 * Callers fold the 1/sqrt(n) factor into the sign vector to avoid a second pass.
 */
inline void fwht_inplace_unscaled(float* v, uint32_t n) noexcept {
#if defined(DEGLIB_X86)
    if (n >= 8 && deglib::cpu::has_avx2()) {
        fwht_inplace_unscaled_avx2(v, n);
        return;
    }
#endif
    fwht_inplace_unscaled_scalar(v, n);
}

inline void mul_vec_scalar(const float* src, const float* scale, float* dst, uint32_t n) noexcept {
    for (uint32_t j = 0; j < n; ++j) {
        dst[j] = src[j] * scale[j];
    }
}

#if defined(DEGLIB_X86)
DEGLIB_TARGET_AVX2 inline void mul_vec_avx2(const float* src, const float* scale, float* dst, uint32_t n) noexcept {
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8) {
        _mm256_storeu_ps(dst + j, _mm256_mul_ps(_mm256_loadu_ps(src + j), _mm256_loadu_ps(scale + j)));
    }
    for (; j < n; ++j) {
        dst[j] = src[j] * scale[j];
    }
}
#endif

inline void mul_vec(const float* src, const float* scale, float* dst, uint32_t n) noexcept {
#if defined(DEGLIB_X86)
    if (n >= 8 && deglib::cpu::has_avx2()) {
        mul_vec_avx2(src, scale, dst, n);
        return;
    }
#endif
    mul_vec_scalar(src, scale, dst, n);
}

inline std::vector<uint32_t> decompose_pow2_blocks(uint32_t dim) {
    std::vector<uint32_t> blocks;
    uint32_t rem = dim;
    while (rem > 0) {
        uint32_t b = 1;
        while ((b << 1) <= rem) b <<= 1;
        blocks.push_back(b);
        rem -= b;
    }
    return blocks;
}

}  // namespace detail

/**
 * @brief Fast Walsh-Hadamard orthogonal rotation in O(D log D).
 *
 * Uses sign-flipped FWHT for power-of-2 dimensions (1 round) and multi-round
 * sign-flipped block-FWHT + permutation (3 rounds) for arbitrary dimensions.
 */
class FastHadamardRotation : public FeatureTransformation {
  public:
    FastHadamardRotation() = default;
    explicit FastHadamardRotation(uint32_t dim, uint64_t seed = 42, uint32_t rounds = 0)
        : dim_(dim), seed_(seed), rounds_(rounds) {
        if (dim_ > 0) init_plan();
    }

    uint32_t tag() const noexcept override { return TAG_FWHT; }
    std::string_view name() const noexcept override { return "FastHadamardRotation"; }
    uint32_t input_dim() const noexcept override { return dim_; }
    uint32_t output_dim() const noexcept override { return dim_; }
    size_t input_byte_size() const noexcept override { return static_cast<size_t>(dim_) * sizeof(float); }
    size_t output_byte_size() const noexcept override { return static_cast<size_t>(dim_) * sizeof(float); }

    deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const override { return input_metric; }
    bool is_fitted() const noexcept override { return is_fitted_; }

    uint32_t dim() const noexcept { return dim_; }
    uint64_t seed() const noexcept { return seed_; }
    uint32_t rounds() const noexcept { return actual_rounds_; }
    bool is_enabled() const noexcept { return enabled_; }
    void set_enabled(bool e) noexcept { enabled_ = e; }
    float crest_factor() const noexcept { return crest_factor_; }
    float excess_kurtosis() const noexcept { return excess_kurtosis_; }

    void fit(std::span<const float> data) override { fit(data, true); }

    void fit(std::span<const float> data, bool auto_detect) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("FastHadamardRotation: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("FastHadamardRotation: data size must be multiple of dim");
        init_plan();
        const auto stats = detail::analyze_rotation_benefit(data.data(), data.size(), dim_);
        crest_factor_ = stats.crest_factor;
        excess_kurtosis_ = stats.excess_kurtosis;
        enabled_ = auto_detect ? stats.recommended : true;
        is_fitted_ = true;
    }

    /**
     * Fits rotation using only an excess-kurtosis threshold:
     *   kurtosis_threshold < 0: disabled
     *   kurtosis_threshold == 0: always enabled
     *   kurtosis_threshold > 0: enabled iff excess_kurtosis > kurtosis_threshold
     */
    void fit_with_kurtosis(std::span<const float> data, float kurtosis_threshold) {
        if (data.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("FastHadamardRotation: dim must be > 0 prior to fit()");
        if (data.size() % dim_ != 0) throw std::invalid_argument("FastHadamardRotation: data size must be multiple of dim");
        init_plan();
        if (kurtosis_threshold < 0.0f) {
            enabled_ = false;
            is_fitted_ = true;
            return;
        }
        const auto stats = detail::analyze_rotation_benefit(data.data(), data.size(), dim_, kurtosis_threshold, -1.0f);
        crest_factor_ = stats.crest_factor;
        excess_kurtosis_ = stats.excess_kurtosis;
        enabled_ = (kurtosis_threshold == 0.0f) ? true : (excess_kurtosis_ > kurtosis_threshold);
        is_fitted_ = true;
    }

    void transform_one(const float* src, float* dst, std::vector<float>& scratch) const noexcept {
        if (!enabled_) {
            if (dst != src) std::memcpy(dst, src, static_cast<size_t>(dim_) * sizeof(float));
            return;
        }

        detail::mul_vec(src, signs_.data(), dst, dim_);
        for (uint32_t r = 0; r < actual_rounds_; ++r) {
            uint32_t off = 0;
            for (uint32_t b : blocks_) {
                if (b > 1) detail::fwht_inplace_unscaled(dst + off, b);
                off += b;
            }
            if (r + 1 < actual_rounds_) {
                const float* s_next = signs_.data() + static_cast<size_t>(r + 1) * dim_;
                if (blocks_.size() > 1) {
                    scratch.resize(dim_);
                    std::memcpy(scratch.data(), dst, static_cast<size_t>(dim_) * sizeof(float));
                    const uint32_t* p = perms_.data() + static_cast<size_t>(r) * dim_;
                    for (uint32_t j = 0; j < dim_; ++j) {
                        dst[j] = scratch[p[j]] * s_next[j];
                    }
                } else {
                    detail::mul_vec(dst, s_next, dst, dim_);
                }
            }
        }
    }

    void transform_database(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }
    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        transform_vectors(in, out, numThreads);
    }
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override { transform_vectors(in, out, 0); }

    void serialize(std::ostream& os) const override {
        os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        os.write(reinterpret_cast<const char*>(&seed_), sizeof(seed_));
        os.write(reinterpret_cast<const char*>(&rounds_), sizeof(rounds_));
        const uint8_t en = enabled_ ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&en), sizeof(en));
        os.write(reinterpret_cast<const char*>(&crest_factor_), sizeof(crest_factor_));
        os.write(reinterpret_cast<const char*>(&excess_kurtosis_), sizeof(excess_kurtosis_));
    }

    static std::shared_ptr<FastHadamardRotation> deserialize(std::istream& is) {
        uint32_t dim = 0;
        uint64_t seed = 42;
        uint32_t rounds = 0;
        uint8_t en = 1;
        float crest = 0.0f;
        float kurt = 0.0f;
        if (!is.read(reinterpret_cast<char*>(&dim), sizeof(dim))) return nullptr;
        if (!is.read(reinterpret_cast<char*>(&seed), sizeof(seed))) return nullptr;
        if (!is.read(reinterpret_cast<char*>(&rounds), sizeof(rounds))) return nullptr;
        if (!is.read(reinterpret_cast<char*>(&en), sizeof(en))) return nullptr;
        if (!is.read(reinterpret_cast<char*>(&crest), sizeof(crest))) return nullptr;
        if (!is.read(reinterpret_cast<char*>(&kurt), sizeof(kurt))) return nullptr;
        auto obj = std::make_shared<FastHadamardRotation>(dim, seed, rounds);
        obj->enabled_ = (en != 0);
        obj->crest_factor_ = crest;
        obj->excess_kurtosis_ = kurt;
        obj->is_fitted_ = true;
        return obj;
    }

  private:
    void init_plan() {
        blocks_ = detail::decompose_pow2_blocks(dim_);
        actual_rounds_ = (rounds_ > 0) ? rounds_ : (blocks_.size() == 1 ? 1u : 3u);

        std::mt19937_64 rng(seed_);
        signs_.resize(static_cast<size_t>(actual_rounds_) * dim_);
        for (uint32_t r = 0; r < actual_rounds_; ++r) {
            float* s = signs_.data() + static_cast<size_t>(r) * dim_;
            uint32_t off = 0;
            for (uint32_t b : blocks_) {
                const float inv_sqrt = (b > 1) ? (1.0f / std::sqrt(static_cast<float>(b))) : 1.0f;
                for (uint32_t j = 0; j < b; ++j) {
                    s[off + j] = ((rng() & 1ULL) ? 1.0f : -1.0f) * inv_sqrt;
                }
                off += b;
            }
        }

        perms_.clear();
        if (blocks_.size() > 1 && actual_rounds_ > 1) {
            perms_.resize(static_cast<size_t>(actual_rounds_ - 1) * dim_);
            for (uint32_t r = 0; r + 1 < actual_rounds_; ++r) {
                auto* p = perms_.data() + static_cast<size_t>(r) * dim_;
                std::iota(p, p + dim_, 0u);
                std::shuffle(p, p + dim_, rng);
            }
        }
    }

    void transform_vectors(std::span<const float> in, std::span<std::byte> out, size_t numThreads) const {
        if (in.empty()) return;
        if (dim_ == 0) throw std::invalid_argument("FastHadamardRotation: dim must be > 0");
        if (in.size() % dim_ != 0) throw std::invalid_argument("FastHadamardRotation: input size must be multiple of dim");
        const size_t count = in.size() / dim_;
        const size_t req_bytes = count * dim_ * sizeof(float);
        if (out.size() < req_bytes) throw std::invalid_argument("FastHadamardRotation: output buffer too small");

        float* dst = reinterpret_cast<float*>(out.data());
        const float* src = in.data();

        if (!enabled_) {
            if (dst != src) std::memcpy(dst, src, req_bytes);
            return;
        }

        if (numThreads <= 1 || count < 1000) {
            thread_local std::vector<float> scratch;
            for (size_t i = 0; i < count; ++i) {
                transform_one(src + i * dim_, dst + i * dim_, scratch);
            }
        } else {
            deglib::concurrent::parallel_batch_for(
                size_t{0}, count, numThreads,
                [&](size_t begin, size_t end, size_t /*thread_id*/) {
                    thread_local std::vector<float> scratch;
                    for (size_t i = begin; i < end; ++i) {
                        transform_one(src + i * dim_, dst + i * dim_, scratch);
                    }
                },
                4
            );
        }
    }

    uint32_t dim_ = 0;
    uint64_t seed_ = 42;
    uint32_t rounds_ = 0;
    uint32_t actual_rounds_ = 1;
    bool enabled_ = false;
    bool is_fitted_ = false;
    float crest_factor_ = 0.0f;
    float excess_kurtosis_ = 0.0f;
    std::vector<uint32_t> blocks_;
    std::vector<float> signs_;
    std::vector<uint32_t> perms_;
};

}  // namespace deglib::transformation
