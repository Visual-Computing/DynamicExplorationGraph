#include "evp_bits.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool condition, const char* message) {
    if (condition) {
        printf("  PASS: %s\n", message);
        tests_passed++;
    } else {
        printf("  FAIL: %s\n", message);
        tests_failed++;
    }
}

static void check_float(float actual, float expected, float epsilon, const char* message) {
    if (std::fabs(actual - expected) < epsilon) {
        printf("  PASS: %s (%.6f == %.6f)\n", message, actual, expected);
        tests_passed++;
    } else {
        printf("  FAIL: %s - expected %.6f, got %.6f\n", message, expected, actual);
        tests_failed++;
    }
}

// ============================================================================
// Test 1: test_evp_similarity_simple
// ============================================================================
static void test_evp_similarity_simple() {
    printf("\n--- test_evp_similarity_simple ---\n");

    // 8-dim Vektoren mit eindeutigen Absolutwerten -> keine Ties
    // a: [1, 2, 3, 4, 5, 6, 7, 8], non_zeros=4 -> top 4 by abs = idx 4,5,6,7
    // b: [8, 7, 6, 5, 4, 3, 2, 1], non_zeros=4 -> top 4 by abs = idx 0,1,2,3
    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,  // a
        8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,  // b
    };
    const size_t count = 2;
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    deglib::EvpBitsArray array(data, count, dim, non_zeros);

    // Vector a: top 4 = idx 4,5,6,7 (all >0)
    // ones(a): bits 4,5,6,7 = 0xF0
    // negs(a): all 0 = 0x00
    //
    // Vector b: top 4 = idx 0,1,2,3 (all >0)
    // ones(b): bits 0,1,2,3 = 0x0F
    // negs(b): all 0 = 0x00
    //
    // aa = popcount(0xF0 & 0x0F) = 0
    // bb = popcount(0x00 & 0x00) = 0
    // cc = popcount(0xF0 & 0x00) = 0
    // dd = popcount(0x0F & 0x00) = 0
    // sim = (0+0+16) - (0+0) = 16

    float sim = deglib::evp_similarity(array, 0, 1);
    check_float(sim, 16.0f, 0.01f, "similarity(a, b) == 16.0 (no overlap)");

    // sim(a,a): ones=0xF0, negs=0x00
    // aa = popcount(0xF0) = 4, bb = 0, cc = 0, dd = 0
    // sim = (4+0+16) - 0 = 20
    float sim_aa = deglib::evp_similarity(array, 0, 0);
    check_float(sim_aa, 20.0f, 0.01f, "similarity(a, a) == 20.0");

    // sim(b,b): ones=0x0F, negs=0x00
    // aa = popcount(0x0F) = 4, sim = (4+0+16) - 0 = 20
    float sim_bb = deglib::evp_similarity(array, 1, 1);
    check_float(sim_bb, 20.0f, 0.01f, "similarity(b, b) == 20.0");
}

// ============================================================================
// Test 2: test_evp_bits_conversion
// ============================================================================
static void test_evp_bits_conversion() {
    printf("\n--- test_evp_bits_conversion ---\n");

    float data[] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f,
                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const size_t count = 1;
    const uint32_t dim = 16;
    const uint32_t non_zeros = 8;

    deglib::EvpBitsArray array(data, count, dim, non_zeros);

    check(array.size() == 1, "size == 1");
    check(array.dim() == 16, "dim == 16");
    check(array.non_zeros() == 8, "non_zeros == 8");
    check(array.bytes_per_evp() == 4, "bytes_per_evp == 4 (2 * 16/8)");

    const std::byte* ones = array.ones_ptr(0);
    uint8_t ones_val = static_cast<uint8_t>(*ones);
    check(ones_val == 0x55, "ones mask == 0x55 (bits at 0,2,4,6)");

    const std::byte* negs = array.negs_ptr(0);
    uint8_t negs_val = static_cast<uint8_t>(*negs);
    check(negs_val == 0xAA, "negative_ones mask == 0xAA (bits at 1,3,5,7)");
}

// ============================================================================
// Test 3: test_from_embeddings_consistency
// ============================================================================
static void test_from_embeddings_consistency() {
    printf("\n--- test_from_embeddings_consistency ---\n");

    std::mt19937 rng(42);
    const size_t count = 100;
    const uint32_t dim = 128;
    const uint32_t non_zeros = 32;

    std::vector<float> data(count * dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count * dim; ++i) {
        data[i] = dist(rng);
    }

    deglib::EvpBitsArray array(data.data(), count, dim, non_zeros);

    bool consistent = true;
    for (size_t i = 0; i < 10 && consistent; ++i) {
        const std::byte* ones = array.ones_ptr(static_cast<uint32_t>(i));
        const std::byte* negs = array.negs_ptr(static_cast<uint32_t>(i));

        const uint64_t* o = reinterpret_cast<const uint64_t*>(ones);
        const uint64_t* n = reinterpret_cast<const uint64_t*>(negs);
        size_t mask_bytes = dim / 8;
        size_t num_uint64 = mask_bytes / sizeof(uint64_t);
        for (size_t j = 0; j < num_uint64; ++j) {
            if ((o[j] & n[j]) != 0) {
                consistent = false;
                break;
            }
        }
    }
    check(consistent, "ones und negative_ones haben keine gemeinsamen Bits");
    check(array.bytes_per_evp() == 32, "bytes_per_evp == 32 (2 * 128/8)");
}

// ============================================================================
// Test 4: test_from_embeddings_with_ties
// ============================================================================
static void test_from_embeddings_with_ties() {
    printf("\n--- test_from_embeddings_with_ties ---\n");

    // All values equal -> partial_sort selects the last non_zeros
    float data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const size_t count = 1;
    const uint32_t dim = 8;
    const uint32_t non_zeros = 4;

    deglib::EvpBitsArray array(data, count, dim, non_zeros);

    check(array.size() == 1, "handles all-same values without crash");

    // ones mask: all > 0, top 4 from partial_sort
    // partial_sort with <: the first 4 (smallest) moved forward,
    // the last 4 are the "top 4". Since all equal, the last 4
    // remain in their original order: idx 4,5,6,7
    const std::byte* ones = array.ones_ptr(0);
    uint8_t ones_val = static_cast<uint8_t>(*ones);
    check(ones_val == 0xF0, "ones mask == 0xF0 (last 4 indices after partial_sort)");
}

// ============================================================================
// Test 5: test_invalid_non_zeros
// ============================================================================
static void test_invalid_non_zeros() {
    printf("\n--- test_invalid_non_zeros ---\n");

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const size_t count = 1;
    const uint32_t dim = 8;

    bool caught = false;
    try { deglib::EvpBitsArray array(data, count, dim, dim); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "non_zeros >= dim throws std::invalid_argument");

    caught = false;
    try { deglib::EvpBitsArray array(data, count, dim, dim + 1); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "non_zeros > dim throws std::invalid_argument");
}

// ============================================================================
// Test 6: test_dim_not_divisible_by_8
// ============================================================================
static void test_dim_not_divisible_by_8() {
    printf("\n--- test_dim_not_divisible_by_8 ---\n");

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                    9.0f, 10.0f, 11.0f, 12.0f};
    const size_t count = 1;

    bool caught = false;
    try { deglib::EvpBitsArray array(data, count, 12, 4); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "dim=12 (not divisible by 8) throws std::invalid_argument");

    caught = false;
    try { deglib::EvpBitsArray array(data, count, 7, 3); }
    catch (const std::invalid_argument&) { caught = true; }
    check(caught, "dim=7 (not divisible by 8) throws std::invalid_argument");

    caught = false;
    try { deglib::EvpBitsArray array(data, count, 8, 4); }
    catch (const std::invalid_argument&) { caught = true; }
    check(!caught, "dim=8 (divisible by 8) does NOT throw");
}

// ============================================================================
// Hauptprogramm
// ============================================================================
int main() {
    printf("=== EVP Unit Tests ===\n");

    test_evp_similarity_simple();
    test_evp_bits_conversion();
    test_from_embeddings_consistency();
    test_from_embeddings_with_ties();
    test_invalid_non_zeros();
    test_dim_not_divisible_by_8();

    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
