/**
 * @file microbench_fp16_l2.cpp
 * @brief Microbenchmark: FP32 L2 vs FP16 L2 distance functions.
 *
 * Measures throughput of:
 *   1. FP32 L2  (L2Float16Ext  / L2Float8Ext)       — float32 input
 *   2. FP16 L2  (FP16L2Ext16   / FP16L2Ext8)        — uint16 input
 *
 * Data: random synthetic vectors, dim=128 (matches llama-dev task2 dataset).
 * The floats_to_fp16() conversion (from task2/mode5.h) is used to generate
 * the FP16 input data from the same float source vectors.
 *
 * Metrics reported per kernel:
 *   - ns/call           (single query vs single database vector)
 *   - MCalls/s          (throughput)
 *   - compare_batch throughput (8 db vectors per call, AVX batch path)
 *
 * Build: added to benchmark/CMakeLists.txt as target microbench_fp16_l2.
 * Run  : microbench_fp16_l2.exe [dim] [num_db] [num_queries] [num_warmup] [num_iters]
 *
 * Usage example:
 *   microbench_fp16_l2.exe 128 10000 1000 5 20
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

#include "distances.h"

// ---------------------------------------------------------------------------
// floats_to_fp16: identical to task2/mode5.h implementation
// ---------------------------------------------------------------------------
static std::vector<uint16_t> floats_to_fp16(const std::vector<float>& v) {
    std::vector<uint16_t> out(v.size());
#if defined(USE_AVX512) || defined(USE_AVX) || defined(USE_SSE)
    size_t i = 0;
    for (; i + 4 <= v.size(); i += 4) {
        __m128 f4 = _mm_loadu_ps(&v[i]);
        __m128i h4 = _mm_cvtps_ph(f4, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h4);
        out[i]   = tmp[0];
        out[i+1] = tmp[1];
        out[i+2] = tmp[2];
        out[i+3] = tmp[3];
    }
    for (; i < v.size(); ++i) {
        __m128 f1 = _mm_set_ss(v[i]);
        __m128i h1 = _mm_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT);
        alignas(16) uint16_t tmp[8];
        _mm_storeu_si128((__m128i*)tmp, h1);
        out[i] = tmp[0];
    }
#else
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        uint16_t sign     = static_cast<uint16_t>((bits >> 16) & 0x8000u);
        int32_t  exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFFu;
        if (exponent <= 0)       { out[i] = sign; }
        else if (exponent >= 31) { out[i] = static_cast<uint16_t>(sign | 0x7C00u); }
        else                     { out[i] = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13)); }
    }
#endif
    return out;
}

// ---------------------------------------------------------------------------
// Timer helpers
// ---------------------------------------------------------------------------
using Clock = std::chrono::high_resolution_clock;
using Ns    = std::chrono::nanoseconds;

static double now_ns() {
    return static_cast<double>(
        std::chrono::duration_cast<Ns>(Clock::now().time_since_epoch()).count());
}

// ---------------------------------------------------------------------------
// Naive reference (for correctness verification)
// ---------------------------------------------------------------------------
static float l2_fp32_naive(const float* a, const float* b, size_t dim) {
    float s = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

static float l2_fp16_naive(const uint16_t* a, const uint16_t* b, size_t dim) {
    float s = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float fa = deglib::distances::fp16l2_to_float(a[i]);
        float fb = deglib::distances::fp16l2_to_float(b[i]);
        float d = fa - fb;
        s += d * d;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------
struct BenchConfig {
    size_t dim        = 128;   // dimension (llama-dev default)
    size_t num_db     = 10000; // database vectors to allocate
    size_t num_q      = 1000;  // query vectors
    int    warmup     = 5;     // warm-up rounds (discarded)
    int    iters      = 20;    // measured rounds
};

// ---------------------------------------------------------------------------
// Generate flat random FP32 data and corresponding FP16 copy
// ---------------------------------------------------------------------------
static void generate_data(
    size_t n, size_t dim,
    std::vector<float>&    fp32_flat,
    std::vector<uint16_t>& fp16_flat,
    unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    fp32_flat.resize(n * dim);
    for (auto& v : fp32_flat) v = dist(rng);

    // Normalize each vector to unit norm (more realistic for ANN workloads)
    for (size_t i = 0; i < n; ++i) {
        float* row = &fp32_flat[i * dim];
        float norm = 0.0f;
        for (size_t j = 0; j < dim; ++j) norm += row[j] * row[j];
        norm = std::sqrt(norm);
        if (norm > 1e-8f)
            for (size_t j = 0; j < dim; ++j) row[j] /= norm;
    }

    // Convert entire flat array to FP16
    std::vector<float> tmp(fp32_flat.begin(), fp32_flat.end());
    fp16_flat = floats_to_fp16(tmp);
}

// ---------------------------------------------------------------------------
// Statistics helper
// ---------------------------------------------------------------------------
struct BenchStats {
    double min_ns;
    double median_ns;
    double mean_ns;
    double stddev_ns;
    double cv_pct;     // coefficient of variation = stddev/mean * 100
};

static BenchStats compute_stats(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / samples.size();
    double sq = 0.0;
    for (double v : samples) { double d = v - mean; sq += d * d; }
    double stddev = std::sqrt(sq / samples.size());
    double median = samples[samples.size() / 2];
    return { samples.front(), median, mean, stddev, (mean > 0.0 ? stddev / mean * 100.0 : 0.0) };
}

// ---------------------------------------------------------------------------
// Benchmark runner: single compare()
// ---------------------------------------------------------------------------
template <typename Comparator, typename VecType>
static BenchStats run_single(
    const std::vector<VecType>& queries_flat,
    const std::vector<VecType>& db_flat,
    size_t dim, size_t num_q, size_t num_db,
    int warmup, int iters,
    volatile float* sink)
{
    const VecType* q_ptr = queries_flat.data();
    const VecType* db_ptr = db_flat.data();
    const double calls_per_run = static_cast<double>(num_q) * static_cast<double>(num_db);

    auto do_run = [&]() {
        for (size_t qi = 0; qi < num_q; ++qi) {
            const VecType* q = q_ptr + qi * dim;
            for (size_t di = 0; di < num_db; ++di) {
                const VecType* d = db_ptr + di * dim;
                float dist = Comparator::compare(q, d, &dim);
                *sink = dist;  // prevent optimization
            }
        }
    };

    for (int i = 0; i < warmup; ++i) do_run();

    std::vector<double> ns_per_call(iters);
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ns();
        do_run();
        ns_per_call[i] = (now_ns() - t0) / calls_per_run;
    }
    return compute_stats(ns_per_call);
}

// ---------------------------------------------------------------------------
// Benchmark runner: compare_batch() — batch of 8 db vectors per call
// ---------------------------------------------------------------------------
template <typename Comparator, typename VecType>
static BenchStats run_batch(
    const std::vector<VecType>& queries_flat,
    const std::vector<VecType>& db_flat,
    size_t dim, size_t num_q, size_t num_db,
    int warmup, int iters,
    volatile float* sink)
{
    const VecType* q_ptr = queries_flat.data();
    const VecType* db_ptr = db_flat.data();
    const double calls_per_run = static_cast<double>(num_q) * static_cast<double>(num_db);

    const size_t batch = 8;
    const size_t full_batches = num_db / batch;
    const size_t remainder    = num_db % batch;

    std::vector<std::vector<const void*>> batch_ptrs(full_batches);
    for (size_t b = 0; b < full_batches; ++b) {
        batch_ptrs[b].resize(batch);
        for (size_t k = 0; k < batch; ++k)
            batch_ptrs[b][k] = db_ptr + (b * batch + k) * dim;
    }
    std::vector<const void*> rem_ptrs(remainder);
    for (size_t k = 0; k < remainder; ++k)
        rem_ptrs[k] = db_ptr + (full_batches * batch + k) * dim;

    std::vector<float> dists(batch);

    auto do_run = [&]() {
        for (size_t qi = 0; qi < num_q; ++qi) {
            const VecType* q = q_ptr + qi * dim;
            for (size_t b = 0; b < full_batches; ++b) {
                Comparator::compare_batch(q, batch_ptrs[b].data(), batch, &dim, dists.data());
                *sink = dists[0];
            }
            for (size_t k = 0; k < remainder; ++k) {
                float d = Comparator::compare(q, rem_ptrs[k], &dim);
                *sink = d;
            }
        }
    };

    for (int i = 0; i < warmup; ++i) do_run();

    std::vector<double> ns_per_call(iters);
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ns();
        do_run();
        ns_per_call[i] = (now_ns() - t0) / calls_per_run;
    }
    return compute_stats(ns_per_call);
}

// ---------------------------------------------------------------------------
// Print SIMD support summary
// ---------------------------------------------------------------------------
static void print_simd_info() {
    std::printf("SIMD support compiled in:");
#if defined(USE_AVX512)
    std::printf(" AVX-512");
#endif
#if defined(USE_AVX)
    std::printf(" AVX");
#endif
#if defined(USE_SSE)
    std::printf(" SSE");
#endif
#if !defined(USE_AVX512) && !defined(USE_AVX) && !defined(USE_SSE)
    std::printf(" none (scalar only)");
#endif
    std::printf("\n");
}

// ---------------------------------------------------------------------------
// Print a benchmark result row with statistics
// ---------------------------------------------------------------------------
static void print_row(const char* label,
                      const BenchStats& s,  // single
                      const BenchStats& b)  // batch-8
{
    // Format: label | single median(CV%) | batch-8 median(CV%)
    std::printf("  %-32s"
                "  single: %6.2f ns [min=%5.2f mean=%5.2f sd=%4.2f CV=%4.1f%%]"
                "  batch8: %6.2f ns [min=%5.2f mean=%5.2f sd=%4.2f CV=%4.1f%%]\n",
                label,
                s.median_ns, s.min_ns, s.mean_ns, s.stddev_ns, s.cv_pct,
                b.median_ns, b.min_ns, b.mean_ns, b.stddev_ns, b.cv_pct);
}

// ---------------------------------------------------------------------------
// Verify correctness: FP16 L2 vs FP32 L2 on a small example
// ---------------------------------------------------------------------------
static bool verify_correctness(size_t dim) {
    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> af(dim), bf(dim);
    for (auto& v : af) v = dist(rng);
    for (auto& v : bf) v = dist(rng);

    // Normalize
    auto normalize = [&](std::vector<float>& v) {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        for (auto& x : v) x /= norm;
    };
    normalize(af);
    normalize(bf);

    auto ah = floats_to_fp16(af);
    auto bh = floats_to_fp16(bf);

    float fp32_ref = l2_fp32_naive(af.data(), bf.data(), dim);
    float fp16_ref = l2_fp16_naive(ah.data(), bh.data(), dim);

    float fp32_simd = deglib::distances::L2Float16Ext::compare(af.data(), bf.data(), &dim);
    float fp16_simd_8  = deglib::distances::FP16L2Ext8::compare(ah.data(), bh.data(), &dim);
    float fp16_simd_16 = deglib::distances::FP16L2Ext16::compare(ah.data(), bh.data(), &dim);

    bool ok = true;
    float tol = 1e-2f;

    auto check = [&](const char* name, float got, float ref) {
        float err = std::fabs(got - ref);
        bool pass = err <= tol * (1.0f + std::fabs(ref));
        std::printf("  %-30s got=%.6f  ref=%.6f  err=%.2e  %s\n",
                    name, got, ref, err, pass ? "OK" : "FAIL");
        if (!pass) ok = false;
    };

    std::printf("Correctness check (dim=%zu):\n", dim);
    check("FP32 L2 (SIMD) vs naive",    fp32_simd,    fp32_ref);
    check("FP16 L2 ext8  vs fp16 naive", fp16_simd_8,  fp16_ref);
    check("FP16 L2 ext16 vs fp16 naive", fp16_simd_16, fp16_ref);
    // FP16 L2 should approximate FP32 L2 (with quantization error)
    float quanterr = std::fabs(fp16_ref - fp32_ref) / (1.0f + std::fabs(fp32_ref));
    std::printf("  FP16 vs FP32 L2 relative quantization error: %.4f %%\n", quanterr * 100.0f);
    std::printf("\n");
    return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    BenchConfig cfg;
    if (argc > 1) cfg.dim      = static_cast<size_t>(std::atoi(argv[1]));
    if (argc > 2) cfg.num_db   = static_cast<size_t>(std::atoi(argv[2]));
    if (argc > 3) cfg.num_q    = static_cast<size_t>(std::atoi(argv[3]));
    if (argc > 4) cfg.warmup   = std::atoi(argv[4]);
    if (argc > 5) cfg.iters    = std::atoi(argv[5]);

    print_simd_info();
    std::printf("Config: dim=%zu  num_db=%zu  num_queries=%zu  warmup=%d  iters=%d\n\n",
                cfg.dim, cfg.num_db, cfg.num_q, cfg.warmup, cfg.iters);

    // Correctness check before benchmarking
    if (!verify_correctness(cfg.dim)) {
        std::fprintf(stderr, "ERROR: correctness check failed!\n");
        return 1;
    }

    // Generate data
    std::vector<float>    db_fp32,  q_fp32;
    std::vector<uint16_t> db_fp16,  q_fp16;
    generate_data(cfg.num_db, cfg.dim, db_fp32, db_fp16, 42);
    generate_data(cfg.num_q,  cfg.dim, q_fp32,  q_fp16, 123);

    volatile float sink = 0.0f;

    std::printf("Benchmark: single compare() — one query vs one db vector\n");
    std::printf("          + compare_batch() — one query vs 8 db vectors (batch path)\n");
    std::printf("%-34s  %-42s  %s\n",
                "Kernel", "Single", "Batch-8");
    std::printf("%s\n", std::string(120, '-').c_str());

    // ----- FP32 L2 -----
    {
        BenchStats s, b;
        if (cfg.dim % 16 == 0) {
            s = run_single<deglib::distances::L2Float16Ext, float>(
                    q_fp32, db_fp32, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::L2Float16Ext, float>(
                    q_fp32, db_fp32, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP32 L2 (ext16, dim%16==0)", s, b);
        } else if (cfg.dim % 8 == 0) {
            s = run_single<deglib::distances::L2Float8Ext, float>(
                    q_fp32, db_fp32, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::L2Float8Ext, float>(
                    q_fp32, db_fp32, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP32 L2 (ext8, dim%8==0)", s, b);
        } else {
            s = run_single<deglib::distances::L2Float, float>(
                    q_fp32, db_fp32, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::L2Float, float>(
                    q_fp32, db_fp32, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP32 L2 (scalar/sse1)", s, b);
        }
    }

    // ----- FP16 L2 -----
    {
        BenchStats s, b;
        if (cfg.dim % 32 == 0) {
            s = run_single<deglib::distances::FP16L2Ext32, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::FP16L2Ext32, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP16 L2 (ext32, dim%32==0)", s, b);
        } else if (cfg.dim % 16 == 0) {
            s = run_single<deglib::distances::FP16L2Ext16, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::FP16L2Ext16, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP16 L2 (ext16, dim%16==0)", s, b);
        } else if (cfg.dim % 8 == 0) {
            s = run_single<deglib::distances::FP16L2Ext8, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::FP16L2Ext8, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP16 L2 (ext8, dim%8==0)", s, b);
        } else {
            s = run_single<deglib::distances::FP16L2Default, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::FP16L2Default, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP16 L2 (scalar)", s, b);
        }
    }

    // ----- Also run FP16 Inner Product for comparison -----
    {
        BenchStats s, b;
        if (cfg.dim % 16 == 0) {
            s = run_single<deglib::distances::FP16InnerProductExt16, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            b = run_batch<deglib::distances::FP16InnerProductExt16, uint16_t>(
                    q_fp16, db_fp16, cfg.dim, cfg.num_q, cfg.num_db,
                    cfg.warmup, cfg.iters, &sink);
            print_row("FP16 IP  (ext16, dim%16==0)", s, b);
        }
    }

    std::printf("\n");
    std::printf("Speedup estimate (FP32 L2 single / FP16 L2 single) at dim=%zu:\n", cfg.dim);
    std::printf("  Memory footprint ratio: FP32=%zu bytes vs FP16=%zu bytes per vector\n",
                cfg.dim * sizeof(float), cfg.dim * sizeof(uint16_t));
    std::printf("  => FP16 vectors are 2x smaller => typically better cache utilization\n");
    std::printf("  => FP16 L2 costs extra cvtph_ps instructions vs FP32 L2\n");

    (void)sink;
    return 0;
}
