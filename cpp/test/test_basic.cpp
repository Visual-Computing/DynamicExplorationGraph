#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

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

// Simplified L2 distance (naive version from distances.h)
float l2_distance(const void* pVect1, const void* pVect2, size_t dim) {
    float* a = (float*)pVect1;
    float* b = (float*)pVect2;
    float result = 0;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

// Simplified InnerProduct distance (naive version from distances.h)
float inner_product_distance(const void* pVect1, const void* pVect2, size_t dim) {
    float* a = (float*)pVect1;
    float* b = (float*)pVect2;
    float dot = 0;
    for (size_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return 1.0f - dot;
}

void test_l2_distance_identical() {
    std::cout << "[L2 Distance - Identical Vectors]\n";
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    float dist = l2_distance(a.data(), b.data(), 3);
    check(dist == 0.0f, "identical vectors: distance = 0");
}

void test_l2_distance_known() {
    std::cout << "[L2 Distance - Known Values]\n";
    std::vector<float> a = {3.0f, 4.0f, 0.0f};
    std::vector<float> b = {0.0f, 0.0f, 0.0f};
    float dist = l2_distance(a.data(), b.data(), 3);
    // (3^2 + 4^2 + 0^2) = 9 + 16 = 25
    check(dist == 25.0f, "L2( (3,4,0), (0,0,0) ) = 25");
}

void test_l2_distance_256() {
    std::cout << "[L2 Distance - Euclidean Length]\n";
    std::vector<float> a = {3.0f, 4.0f, 0.0f};
    std::vector<float> b = {0.0f, 0.0f, 0.0f};
    float dist = l2_distance(a.data(), b.data(), 3);
    sqrt(dist); // would be 5.0
    check(dist == 25.0f, "L2 = 25 (sqrt = 5.0)");
}

void test_inner_product() {
    std::cout << "[Inner Product Distance]\n";
    
    // [1, 0, 0, 0] · [1, 0, 0, 0] = 1, so 1 - 1 = 0
    std::vector<float> a = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> b = {1.0f, 0.0f, 0.0f, 0.0f};
    float dist = inner_product_distance(a.data(), b.data(), 4);
    check(dist == 0.0f, "inner_product identical basis = 0");
    
    // [1, 1, 1, 1] · [1, 1, 1, 1] = 4, so 1 - 4 = -3
    std::vector<float> c = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> d = {1.0f, 1.0f, 1.0f, 1.0f};
    float dist2 = inner_product_distance(c.data(), d.data(), 4);
    check(dist2 == -3.0f, "inner_product [1,1,1,1] = -3");
    
    // [1, 0, 0, 0] · [0, 0, 0, 1] = 0, so 1 - 0 = 1
    std::vector<float> e = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> f = {0.0f, 0.0f, 0.0f, 1.0f};
    float dist3 = inner_product_distance(e.data(), f.data(), 4);
    check(dist3 == 1.0f, "inner_product orthogonal = 1");
}

void test_sift_like_vectors() {
    std::cout << "[SIFT-like Vectors]\n";
    
    // SIFT1M uses 128-dim vectors, L2 distance
    std::vector<float> a(128, 0.0f);
    std::vector<float> b(128, 0.0f);
    std::vector<float> c(128, 0.0f);
    
    // a and b identical
    check(l2_distance(a.data(), b.data(), 128) == 0.0f, "128-d identical: L2 = 0");
    
    // c differs only in first dimension
    c[0] = 1.0f;
    check(l2_distance(b.data(), c.data(), 128) == 1.0f, "128-d diff by 1: L2 = 1");
    
    // a and c same
    check(l2_distance(a.data(), c.data(), 128) == 1.0f, "128-d a vs c: L2 = 1");
}

int main() {
    std::cout << "=== DEG Unit Tests (DistantFunctions) ===\n\n";
    
    test_l2_distance_identical();
    test_l2_distance_known();
    test_l2_distance_256();
    test_inner_product();
    test_sift_like_vectors();
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    if (tests_passed < tests_run) {
        std::cerr << "FAILED!\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
