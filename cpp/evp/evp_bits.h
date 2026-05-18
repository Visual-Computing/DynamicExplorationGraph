#pragma once

#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include "concurrent.h"

namespace deglib {

/**
 * Liest eine ivecs-Datei (int32-Vektoren).
 * Format: [dim(4B)][data(dim x 4B)] pro Vektor.
 * Analog zu fvecs_read, aber mit int32 statt float.
 *
 * @param fname Dateipfad
 * @param d_out Ausgabe: Dimension
 * @param n_out Ausgabe: Anzahl der Vektoren
 * @return std::unique_ptr<std::byte[]> mit rohen int32-Daten
 */
inline auto ivecs_read(const char* fname, size_t& d_out, size_t& n_out) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{}) {
        std::fprintf(stderr, "error when accessing file %s, size is: %ju message: %s\n",
                     fname, file_size, ec.message().c_str());
        std::abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open()) {
        std::fprintf(stderr, "could not open %s\n", fname);
        std::abort();
    }

    // read dimension header
    uint32_t dims = 0;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    assert((dims > 0 && dims < 1'000'000) && "unreasonable dimension");

    // compute number of rows
    assert(file_size % ((dims + 1) * sizeof(int32_t)) == 0 || !"weird file size");
    size_t n = static_cast<size_t>(file_size) / ((dims + 1) * sizeof(int32_t));
    d_out = dims;
    n_out = n;

    // read data
    auto x = std::make_unique<std::byte[]>(file_size);
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x.get()), static_cast<std::streamsize>(file_size));
    if (!ifstream) assert(ifstream.gcount() == static_cast<std::streamsize>(file_size) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; ++i)
        std::memmove(&x[i * dims * sizeof(int32_t)],
                     &x[sizeof(dims) + i * (dims + 1) * sizeof(int32_t)],
                     dims * sizeof(int32_t));

    ifstream.close();
    return x;
}

/**
 * Speichert alle EvpBits-Vektoren in einem einzigen zusammenhängenden
 * std::vector<std::byte>, analog zum StaticFeatureRepository-Speicherlayout.
 *
 * Layout pro Vektor: [ones (dim/8 bytes)][negative_ones (dim/8 bytes)]
 * Total pro Vektor: 2 * ceil(dim / 8) bytes (KEIN Alignment, KEIN Padding)
 *
 * Die Dimension dim muss durch 8 teilbar sein, sonst wird std::invalid_argument
 * geworfen.
 *
 * Entspricht der Rust-Implementierung: direkte Bit-Operationen (and + popcount),
 * KEINE int8-Konvertierung.
 */
class EvpBitsArray {
public:
    /**
     * Erstellt eine leere EvpBitsArray.
     */
    EvpBitsArray() = default;

    /**
     * Erstellt eine EvpBitsArray und konvertiert float-Daten.
     * @param data Pointer zu [count x dim] float-Daten (row-major)
     * @param count Anzahl der Vektoren
     * @param dim Dimension der Vektoren (MUSS durch 8 teilbar sein!)
     * @param non_zeros Anzahl der nicht-null Elemente pro Vektor
     * @throws std::invalid_argument wenn dim % 8 != 0
     * @throws std::invalid_argument wenn non_zeros >= dim
     */
    EvpBitsArray(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0) {
        from_embeddings(data, count, dim, non_zeros, numThreads);
    }

    // --- Konvertierung von float-Daten ---

    /**
     * Konvertiert float-Daten zu EvpBits.
     * @param data Pointer zu [count x dim] float-Daten (row-major)
     * @param count Anzahl der Vektoren
     * @param dim Dimension (MUSS durch 8 teilbar sein!)
     * @param non_zeros Anzahl der nicht-null Elemente pro Vektor
     * @throws std::invalid_argument wenn dim % 8 != 0
     * @throws std::invalid_argument wenn non_zeros >= dim
     */
    void from_embeddings(const float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t numThreads = 0) {
        if (dim % 8 != 0) {
            throw std::invalid_argument("EvpBitsArray: dim must be divisible by 8, got " + std::to_string(dim));
        }
        if (non_zeros >= dim) {
            throw std::invalid_argument("EvpBitsArray: non_zeros must be < dim, got non_zeros=" +
                                        std::to_string(non_zeros) + " dim=" + std::to_string(dim));
        }

        count_ = count;
        dim_ = dim;
        non_zeros_ = non_zeros;
        bytes_per_evp_ = 2 * ((dim_ + 7) / 8);  // 2 * ceil(dim / 8)

        bits_.resize(count_ * bytes_per_evp_);

        // Pre-allocate buffers once
        abs_vals_buffer_.resize(dim);
        is_top_buffer_.resize(dim);

        // Parallelize conversion with batching.
        // Each thread processes a chunk of vectors sequentially,
        // with buffers copied per-thread (thread-local).
        const size_t chunk_size = 8192;
        const size_t num_chunks = (count_ + chunk_size - 1) / chunk_size;
        deglib::concurrent::parallel_for(
            static_cast<size_t>(0), num_chunks, numThreads,
            [this, data, dim, non_zeros, chunk_size](size_t chunk_id, size_t /*threadId*/) {
                size_t start = chunk_id * chunk_size;
                size_t end = std::min(start + chunk_size, count_);
                // Thread-lokale Puffer (keine Race conditions)
                std::vector<std::pair<float, uint32_t>> abs_vals(dim);
                std::vector<uint8_t> is_top(dim, 0);

                for (size_t i = start; i < end; ++i) {
                    convert_embedding(&data[i * dim], &bits_[i * bytes_per_evp_],
                                      dim, non_zeros, abs_vals, is_top);
                }
            });
    }

    // --- Zugriff ---

    /**
     * Gibt Pointer auf die ones-Bytes des Vektors mit index zurück.
     * ones beginnen bei offset 0 innerhalb des Vektor-Blocks.
     */
    inline const std::byte* ones_ptr(uint32_t index) const {
        return bits_.data() + size_t(index) * bytes_per_evp_;
    }

    /**
     * Gibt Pointer auf die negative_ones-Bytes des Vektors mit index zurück.
     * negative_ones beginnen bei offset dim/8 innerhalb des Vektor-Blocks.
     */
    inline const std::byte* negs_ptr(uint32_t index) const {
        return bits_.data() + size_t(index) * bytes_per_evp_ + dim_ / 8;
    }

    /**
     * Gibt die Dimension zurück.
     */
    uint32_t dim() const { return dim_; }

    /**
     * Gibt die Anzahl der Vektoren zurück.
     */
    size_t size() const { return count_; }

    /**
     * Gibt die Bytes pro EvpBits zurück.
     */
    size_t bytes_per_evp() const { return bytes_per_evp_; }

    /**
     * Gibt die Anzahl der nicht-null Elemente pro Vektor zurück.
     */
    uint32_t non_zeros() const { return non_zeros_; }

    /**
     * Gibt den rohen Byte-Pointer zurück.
     */
    const std::byte* data() const { return bits_.data(); }

private:
    size_t count_ = 0;
    uint32_t dim_ = 0;
    uint32_t non_zeros_ = 0;
    size_t bytes_per_evp_ = 0;  // 2 * ceil(dim / 8)
    std::vector<std::byte> bits_;  // [count][bytes_per_evp]

    // Pre-allocierte Puffer für convert_embedding (vermeidet Allokationen pro Vektor)
    std::vector<std::pair<float, uint32_t>> abs_vals_buffer_;
    std::vector<uint8_t> is_top_buffer_;

    /**
     * Konvertiert einen float-Embedding zu ones/negs Bytes.
     * Schreibt direkt in den bits_ Block.
     * Entspricht Rust: from_embedding mit set_bit auf Bit-Container.
     *
     * Verwendet einen pre-allocierten Puffer für abs_vals und is_top,
     * um Allokationen pro Vektor zu vermeiden.
     */
    void convert_embedding(const float* embedding, std::byte* vertex_dst,
                           uint32_t dim, uint32_t non_zeros,
                           std::vector<std::pair<float, uint32_t>>& abs_vals,
                           std::vector<uint8_t>& is_top) {
        // Step 1: Find the non_zeros indices with largest absolute values
        abs_vals.resize(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            abs_vals[i] = {std::abs(embedding[i]), i};
        }

        // Partition so that the largest `non_zeros` absolute values are in the
        // tail of `abs_vals`. `std::nth_element` is O(n) average and avoids the
        // extra work of sorting the top-k fully.
        std::nth_element(abs_vals.begin(), abs_vals.begin() + (dim - non_zeros), abs_vals.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });

        // Step 2: Set is_top for the top non_zeros indices
        // After nth_element, the largest values are in the last
        // `non_zeros` positions: [dim-non_zeros, dim).
        is_top.assign(dim, 0);
        for (uint32_t j = dim - non_zeros; j < dim; ++j) {
            is_top[abs_vals[j].second] = 1;
        }

        // Step 3: Convert to bits
        const size_t mask_bytes = dim / 8;
        std::byte* ones_dst = vertex_dst;
        std::byte* negs_dst = vertex_dst + mask_bytes;

        // Write ones mask
        for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            int byte_val = 0;
            for (int bit = 0; bit < 8; ++bit) {
                const uint32_t bit_idx = byte_idx * 8 + bit;
                if (bit_idx >= dim) break;
                if (is_top[bit_idx] && embedding[bit_idx] > 0.0f) {
                    byte_val |= (1 << bit);
                }
            }
            ones_dst[byte_idx] = static_cast<std::byte>(byte_val);
        }

        // negative_ones mask schreiben
        for (uint32_t byte_idx = 0; byte_idx < mask_bytes; ++byte_idx) {
            int byte_val = 0;
            for (int bit = 0; bit < 8; ++bit) {
                const uint32_t bit_idx = byte_idx * 8 + bit;
                if (bit_idx >= dim) break;
                if (is_top[bit_idx] && embedding[bit_idx] < 0.0f) {
                    byte_val |= (1 << bit);
                }
            }
            negs_dst[byte_idx] = static_cast<std::byte>(byte_val);
        }
    }
};

}  // namespace deglib

// ============================================================================
// Similarity-Funktionen (freie Funktionen)
// ============================================================================

namespace deglib {

/**
 * Berechnet EvpBits-Similarity zwischen zwei Vektoren.
 *
 * Formel: (aa + bb + dim*2) - (cc + dd)
 *   aa = popcount(ones(a) AND ones(b))
 *   bb = popcount(neg(a) AND neg(b))
 *   cc = popcount(ones(a) AND neg(b))
 *   dd = popcount(ones(b) AND neg(a))
 *
 * Entspricht exakt der Rust-Implementierung in evp.rs::similarity().
 *
 * @param array EvpBitsArray
 * @param a Index des ersten Vektors
 * @param b Index des zweiten Vektors
 * @return Similarity als float
 */
inline float evp_similarity(const EvpBitsArray& array, uint32_t a, uint32_t b) {
    const uint32_t dim = array.dim();
    const size_t mask_bytes = dim / 8;

    const std::byte* ones_a = array.ones_ptr(a);
    const std::byte* negs_a = array.negs_ptr(a);
    const std::byte* ones_b = array.ones_ptr(b);
    const std::byte* negs_b = array.negs_ptr(b);

    uint32_t aa = 0, bb = 0, cc = 0, dd = 0;

    // Process 8 bytes (64 bits) at a time using uint64_t
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= mask_bytes; i += sizeof(uint64_t)) {
        const uint64_t o1 = *reinterpret_cast<const uint64_t*>(&ones_a[i]);
        const uint64_t o2 = *reinterpret_cast<const uint64_t*>(&ones_b[i]);
        const uint64_t n1 = *reinterpret_cast<const uint64_t*>(&negs_a[i]);
        const uint64_t n2 = *reinterpret_cast<const uint64_t*>(&negs_b[i]);
        aa += std::popcount(o1 & o2);
        bb += std::popcount(n1 & n2);
        cc += std::popcount(o1 & n2);
        dd += std::popcount(o2 & n1);
    }

    // Process remaining bytes
    for (; i < mask_bytes; ++i) {
        unsigned int b1 = static_cast<unsigned int>(static_cast<uint8_t>(ones_a[i]));
        unsigned int b2 = static_cast<unsigned int>(static_cast<uint8_t>(ones_b[i]));
        unsigned int n1 = static_cast<unsigned int>(static_cast<uint8_t>(negs_a[i]));
        unsigned int n2 = static_cast<unsigned int>(static_cast<uint8_t>(negs_b[i]));
        aa += std::popcount(b1 & b2);
        bb += std::popcount(n1 & n2);
        cc += std::popcount(b1 & n2);
        dd += std::popcount(b2 & n1);
    }

    return static_cast<float>(aa + bb + dim * 2) - static_cast<float>(cc + dd);
}

/**
 * Berechnet Similarity zwischen zwei raw byte-Pointern.
 * Optimiert für Cache-freundlichen Zugriff.
 *
 * @param ones_a Pointer auf ones-Bytes von a
 * @param negs_a Pointer auf negative_ones-Bytes von a
 * @param ones_b Pointer auf ones-Bytes von b
 * @param negs_b Pointer auf negative_ones-Bytes von b
 * @param dim Dimension
 * @return Similarity als float
 */
inline float evp_similarity_bytes(const std::byte* ones_a, const std::byte* negs_a,
                                   const std::byte* ones_b, const std::byte* negs_b,
                                   uint32_t dim) {
    const size_t mask_bytes = dim / 8;

    uint32_t aa = 0, bb = 0, cc = 0, dd = 0;

    size_t i = 0;
    for (; i + sizeof(uint64_t) <= mask_bytes; i += sizeof(uint64_t)) {
        const uint64_t o1 = *reinterpret_cast<const uint64_t*>(&ones_a[i]);
        const uint64_t o2 = *reinterpret_cast<const uint64_t*>(&ones_b[i]);
        const uint64_t n1 = *reinterpret_cast<const uint64_t*>(&negs_a[i]);
        const uint64_t n2 = *reinterpret_cast<const uint64_t*>(&negs_b[i]);
        aa += std::popcount(o1 & o2);
        bb += std::popcount(n1 & n2);
        cc += std::popcount(o1 & n2);
        dd += std::popcount(o2 & n1);
    }

    for (; i < mask_bytes; ++i) {
        unsigned int b1 = static_cast<unsigned int>(static_cast<uint8_t>(ones_a[i]));
        unsigned int b2 = static_cast<unsigned int>(static_cast<uint8_t>(ones_b[i]));
        unsigned int n1 = static_cast<unsigned int>(static_cast<uint8_t>(negs_a[i]));
        unsigned int n2 = static_cast<unsigned int>(static_cast<uint8_t>(negs_b[i]));
        aa += std::popcount(b1 & b2);
        bb += std::popcount(n1 & n2);
        cc += std::popcount(b1 & n2);
        dd += std::popcount(b2 & n1);
    }

    return static_cast<float>(aa + bb + dim * 2) - static_cast<float>(cc + dd);
}

/**
 * Returns the maximum possible similarity for given dim and non_zeros.
 * Corresponds to the similarity of a vector with itself.
 */
inline float get_max_similarity(uint32_t dim, uint32_t non_zeros) {
    // A vector with itself: all non_zeros bits match in ones and negs
    // aa = non_zeros (all ones bits match)
    // bb = count of negative values (all negs bits match)
    // cc = dd = 0 (no overlaps between ones and negs)
    // max = non_zeros + neg_count + dim*2 - 0
    // But we don't know how many are negative - approximate with non_zeros
    // In practice: similarity(a, a) = non_zeros + neg_count + dim*2
    // Since we don't know neg_count, we use: 2 * non_zeros + dim * 2
    // This is an upper bound
    return static_cast<float>(2 * non_zeros + dim * 2);
}

}  // namespace deglib
