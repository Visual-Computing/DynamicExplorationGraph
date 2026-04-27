#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>

static int tests_run = 0;
static int tests_passed = 0;

void check(bool condition, const char* name) {
    ++tests_run;
    if (condition) {
        ++tests_passed;
        std::cout << "  PASS: " << name << "\n";
    } else {
        std::cerr << "  FAIL: " << name << "\n";
    }
}

// --- Naive L2 ---
float l2_naive(const float* a, const float* b, size_t dim) {
    float sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// --- InnerProduct ---
float ip_naive(const float* a, const float* b, size_t dim) {
    float dot = 0;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
}

// --- L2 uint8 ---
float l2_uint8(const uint8_t* a, const uint8_t* b, size_t dim) {
    int64_t sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        sum += d * d;
    }
    return (float)sum;
}

// ===== Tests =====

void test_l2_naive() {
    std::cout << "[L2 Naive]\n";

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    float d = l2_naive(a.data(), b.data(), 3);
    // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
    check(abs(d - 27.0f) < 1e-4f, "l2(1,2,3 vs 4,5,6) = 27");

    std::vector<float> zero(3, 0.0f);
    check(l2_naive(zero.data(), zero.data(), 3) == 0.0f, "l2(zero, zero) = 0");

    std::vector<float> orig = {-1.0f, 0.0f, 7.0f, 100.0f};
    float d2 = l2_naive(orig.data(), orig.data(), 4);
    check(d2 == 0.0f, "l2(same, same) = 0");
}

void test_l2_known_vectors() {
    std::cout << "[L2 Known Vectors]\n";

    // (0,0) vs (3,4) -> 3^2 + 4^2 = 25
    float a[] = {0.0f, 0.0f};
    float b[] = {3.0f, 4.0f};
    check(l2_naive(a, b, 2) == 25.0f, "l2(0,0 vs 3,4) = 25");

    // SIFT-like 128-dim: only 2 dims differ
    std::vector<float> p(128, 0.0f);
    p[0] = 3.0f; p[1] = 4.0f;
    std::vector<float> q(128, 0.0f);
    check(l2_naive(p.data(), q.data(), 128) == 25.0f, "128-dim: only first 2 dims differ -> 25");
}

void test_inner_product() {
    std::cout << "[Inner Product]\n";

    // unit vector dot itself = 1, so 1-1 = 0
    float a[] = {1.0f, 0, 0, 0};
    float b[] = {1.0f, 0, 0, 0};
    check(ip_naive(a, b, 4) == 0.0f, "ip(1,0,0,0 self) = 0");

    // [1,1,1,1] dot [1,1,1,1] = 4, so 1-4 = -3
    float c[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float d[] = {1.0f, 1.0f, 1.0f, 1.0f};
    check(ip_naive(c, d, 4) == -3.0f, "ip([1,1,1,1], [1,1,1,1]) = -3");

    // orthogonal vectors
    float e[] = {1.0f, 0, 0, 0};
    float f[] = {0.0f, 0, 0, 1.0f};
    check(ip_naive(e, f, 4) == 1.0f, "ip(orthogonal) = 1");

    // 128 dim: same vector, all values = 0.5
    std::vector<float> g(128, 0.5f);
    // dot = 128 * 0.25 = 32, so 1 - 32 = -31
    check(ip_naive(g.data(), g.data(), 128) == -31.0f, "ip(128-dim 0.5 self) = -31");
}

void test_l2_uint8() {
    std::cout << "[L2 uint8]\n";

    uint8_t a[128], b[128];
    memset(a, 0, 128);
    memset(b, 0, 128);
    check(l2_uint8(a, b, 128) == 0.0f, "l2(uint8 zeros) = 0");

    // [max, ...] vs [0, ...] per dimension: 255^2 = 65025 each
    memset(a, 255, 128);
    memset(b, 0, 128);
    // 128 * 65025 = 8323200
    float d = l2_uint8(a, b, 128);
    check(abs(d - 128.0f * 65025.0f) < 1000, "l2(uint8 all-255 vs all-0) ~8323200");

    // diff by 1 in each of 10 dims: 10 * 1 = 10
    uint8_t c[128], d2[128];
    memset(c, 0, 128);
    memset(d2, 0, 128);
    for (int i = 0; i < 10; ++i) d2[i] = 1;
    check(l2_uint8(c, d2, 128) == 10.0f, "l2(uint8 diff 1 in 10 dims) = 10");
}

void test_symmetry() {
    std::cout << "[Symmetry Check]\n";

    std::vector<float> a(4, 0.0f);
    a[0] = 3.0f; a[1] = 4.0f;
    std::vector<float> b(4, 0.0f);
    b[2] = 1.0f; b[3] = 2.0f;

    float ab = l2_naive(a.data(), b.data(), 4);
    float ba = l2_naive(b.data(), a.data(), 4);
    check(ab == ba, "l2(a,b) == l2(b,a)");

    float ip_ab = ip_naive(a.data(), b.data(), 4);
    float ip_ba = ip_naive(b.data(), a.data(), 4);
    check(ip_ab == ip_ba, "ip(a,b) == ip(b,a)");
}

void test_edge_cases() {
    std::cout << "[Edge Cases]\n";

    // Zero-size vectors
    float a = 1.0f, b = 2.0f;
    check(l2_naive(&a, &b, 0) == 0.0f, "l2(dim=0) = 0");

    // Very close vectors
    float x1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x2[] = {1.0f, 2.0f, 3.0f, 4.0001f};
    float d = l2_naive(x1, x2, 4);
    check(abs(d - 0.00000001f) < 1e-7f, "l2(tiny diff) ~ 1e-8");
}

int main() {
    std::cout << "=== DEG Distance Unit Tests ===\n\n";

    test_l2_naive();
    test_l2_known_vectors();
    test_inner_product();
    test_l2_uint8();
    test_symmetry();
    test_edge_cases();

    std::cout << "\n=== " << tests_passed << "/" << tests_run << " passed ===\n";
    return tests_passed < tests_run ? 1 : 0;
}
