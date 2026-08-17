#include "deglib/filter.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

TEST(FilterRegression, VaryingFillRatesStressTest) {
    constexpr size_t max_value = 999'999;
    constexpr size_t max_id_count = 1'000'000;
    std::vector<int> valid_ids;

    // Deterministic uniform distribution for reproducible regression checks
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(max_value));
    const size_t num_queries = 100'000;
    std::vector<int> query_ids(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        query_ids[i] = dist(rng);
    }

    std::cout << "\n=== Filter Regression / Stress Test (1M elements) ===\n";

    for (size_t fill_rate = 0; fill_rate <= 1'000'000; fill_rate += 100'000) {
        valid_ids.clear();
        valid_ids.reserve(fill_rate);
        for (size_t i = 0; i < fill_rate; ++i) {
            valid_ids.push_back(static_cast<int>(i));
        }

        // Measure time for Filter creation
        auto start_creation = std::chrono::high_resolution_clock::now();
        deglib::search::Filter filter(valid_ids.data(), valid_ids.size(), max_value, max_id_count);
        auto end_creation = std::chrono::high_resolution_clock::now();
        double creation_ms = std::chrono::duration<double, std::milli>(end_creation - start_creation).count();

        EXPECT_EQ(filter.size(), fill_rate);
        const double expected_rate = static_cast<double>(fill_rate) / static_cast<double>(max_id_count);
        EXPECT_NEAR(filter.get_inclusion_rate(), expected_rate, 1e-6);

        // Measure iteration via for_each_valid_label
        auto start_retrieval = std::chrono::high_resolution_clock::now();
        uint64_t retrieved_ids_count = 0;
        filter.for_each_valid_label([&retrieved_ids_count](uint32_t /*valid_label*/) { retrieved_ids_count++; });
        auto end_retrieval = std::chrono::high_resolution_clock::now();
        double retrieval_ms = std::chrono::duration<double, std::milli>(end_retrieval - start_retrieval).count();

        EXPECT_EQ(retrieved_ids_count, fill_rate);

        // Perform 100k random is_valid checks
        size_t valid_count = 0;
        auto start_is_valid = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_queries; ++i) {
            if (filter.is_valid(query_ids[i])) {
                valid_count++;
            }
        }
        auto end_is_valid = std::chrono::high_resolution_clock::now();
        double is_valid_ms = std::chrono::duration<double, std::milli>(end_is_valid - start_is_valid).count();

        std::cout << std::fixed << std::setprecision(3) << "Fill rate: " << std::setw(7) << fill_rate << " | Creation: " << std::setw(6) << creation_ms << " ms"
                  << " | ForEach: " << std::setw(6) << retrieval_ms << " ms"
                  << " | 100k isValid: " << std::setw(6) << is_valid_ms << " ms"
                  << " | Valid in queries: " << valid_count << "\n";
    }
}
