#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <utility>


namespace deglib::graph {

class Filter {
private:
    std::vector<uint64_t> bitset_; // Bitset as a vector of 64-bit integers
    size_t max_value_;             // Maximum valid labels
    size_t max_label_count_;          // Total possible valid labels
    size_t current_valid_count_;   // Current number of valid labels

    // Helper function to determine the index and bit position in the bitset
    std::pair<size_t, uint64_t> get_bit_position(int label) const {
        size_t index = label / 64;              // Index in the vector
        uint64_t bit = 1ULL << (label % 64);    // Position of the bit within the 64 bits
        return {index, bit};
    }

public:
    // Constructor: valid labels as int-pointer and size, max_value defines the maximum label value, max_label_count is the total possible labels
    Filter(const int* valid_labels, size_t size, size_t max_value, size_t max_label_count)
        : max_value_(max_value), max_label_count_(max_label_count), current_valid_count_(0) {
        size_t bitset_size = (max_value_ / 64) + 1; // Required size of the bitset
        bitset_.resize(bitset_size, 0);

        for (size_t i = 0; i < size; ++i) {
            if (valid_labels[i] >= 0 && static_cast<size_t>(valid_labels[i]) <= max_value_) {
                auto [index, bit] = get_bit_position(valid_labels[i]);
                if ((bitset_[index] & bit) == 0) { // Only count unique valid labels
                    bitset_[index] |= bit;
                    current_valid_count_++;
                }
            }
        }
    }

    // Check if an label is valid
    bool is_valid(int label) const {
        if (label < 0 || static_cast<size_t>(label) > max_value_) {
            return false;
        }
        auto [index, bit] = get_bit_position(label);
        return (bitset_[index] & bit) != 0;
    }

    // Get the number of valid labels
    size_t size() const {
        return current_valid_count_;
    }

    // Apply a function to each valid label
    template <typename FUNC>
    void for_each_valid_label(FUNC&& func) const {
        for (size_t i = 0; i < bitset_.size(); ++i) {
            uint64_t bits = bitset_[i];
            for (int bit_pos = 0; bits != 0; ++bit_pos) {
                if (bits & 1ULL) {
                    func(static_cast<int>(i * 64 + bit_pos));
                }
                bits >>= 1;
            }
        }
    }

    // Get the inclusion rate: ratio of valid labels to max possible labels
    double get_inclusion_rate() const {
        return static_cast<double>(current_valid_count_) / max_label_count_;
    }
};

}  // namespace deglib::graph