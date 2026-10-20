#pragma once

#include "deglib/distances.h"
#include "deglib/graph/internal_graph.h"
#include "deglib/transformation.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace deglib::serialization {

// 4-byte Magic sentinel placed right after vertex data to identify the trailer
constexpr uint32_t TRAILER_MAGIC = 0x58474544;  // "DEGX" ('D' | 'E'<<8 | 'G'<<16 | 'X'<<24)

// Extension block FourCC tags
constexpr uint32_t TAG_XFRM = 0x4D524658;  // "XFRM"
constexpr uint32_t TAG_PIPE = 0x45504950;  // "PIPE" (legacy alias)
constexpr uint32_t TAG_PREF = 0x46455250;  // "PREF"
constexpr uint32_t TAG_ENTR = 0x52544E45;  // "ENTR"

/**
 * In-memory representation of graph file contents.
 */
struct ParsedGraphData {
    std::optional<deglib::distances::FloatSpace> feature_space;
    uint32_t size = 0;
    uint8_t edges_per_vertex = 0;
    int32_t po = 8;
    int32_t pl = 3;
    int32_t nl = 3;
    std::vector<uint32_t> entry_vertex_indices;
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations;

    // Buffer holding the raw contiguous vertex records:
    // [feature bytes][neighbor indices][weights][label] for vertex 0..size-1
    std::vector<std::byte> storage;
    const std::byte* vertex_bytes = nullptr;
};

/**
 * Universal graph file reader.
 * 1. Reads standard 8-byte graph header (metric_type, dim, size, edges_per_vertex) at offset 0.
 * 2. Reads raw contiguous vertex records directly from offset 8.
 * 3. If file contains trailing bytes after vertices, inspects optional "DEGX" extension trailer
 *    and parses TLV extension blocks (Transforms, Prefetch tuning, Entry vertex indices).
 */
inline ParsedGraphData read_graph_file(const char* path_graph) {
    auto is = std::ifstream(path_graph, std::ios::binary);
    if (!is.is_open()) {
        std::fprintf(stderr, "read_graph_file: could not open %s\n", path_graph);
        perror("");
        abort();
    }

    ParsedGraphData data;

    // 1. Standard 8-byte graph header
    uint8_t metric_type = 0;
    is.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
    uint16_t dim = 0;
    is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    data.feature_space.emplace(dim, static_cast<deglib::distances::Metric>(metric_type));

    is.read(reinterpret_cast<char*>(&data.size), sizeof(data.size));
    is.read(reinterpret_cast<char*>(&data.edges_per_vertex), sizeof(data.edges_per_vertex));

    const size_t feat_size = data.feature_space->get_data_size();
    const size_t neighbor_bytes = size_t(data.edges_per_vertex) * sizeof(uint32_t);
    const size_t weight_bytes = size_t(data.edges_per_vertex) * sizeof(float);
    const size_t byte_size_per_vertex = feat_size + neighbor_bytes + weight_bytes + sizeof(uint32_t);
    const size_t total_vertex_bytes = size_t(data.size) * byte_size_per_vertex;

    // 2. Vertex records
    data.storage.resize(total_vertex_bytes);
    is.read(reinterpret_cast<char*>(data.storage.data()), total_vertex_bytes);
    data.vertex_bytes = data.storage.data();

    data.po = 8;
    data.pl = 3;
    data.nl = 3;
    data.transformations = {};

    // 3. Optional Extension Trailer (DEGX)
    uint32_t magic = 0;
    if (is.read(reinterpret_cast<char*>(&magic), sizeof(magic)) && magic == TRAILER_MAGIC) {
        while (is.good() && !is.eof()) {
            uint32_t tag = 0;
            uint32_t length = 0;
            if (!is.read(reinterpret_cast<char*>(&tag), sizeof(tag))) break;
            if (!is.read(reinterpret_cast<char*>(&length), sizeof(length))) break;

            if (tag == TAG_PREF && length >= 12) {
                is.read(reinterpret_cast<char*>(&data.po), sizeof(data.po));
                is.read(reinterpret_cast<char*>(&data.pl), sizeof(data.pl));
                is.read(reinterpret_cast<char*>(&data.nl), sizeof(data.nl));
                if (length > 12) is.ignore(length - 12);
            } else if (tag == TAG_ENTR && length >= sizeof(uint32_t)) {
                uint32_t count = 0;
                is.read(reinterpret_cast<char*>(&count), sizeof(count));
                if (count > 0 && length >= sizeof(uint32_t) + count * sizeof(uint32_t)) {
                    data.entry_vertex_indices.resize(count);
                    is.read(reinterpret_cast<char*>(data.entry_vertex_indices.data()), count * sizeof(uint32_t));
                }
                const size_t consumed = sizeof(uint32_t) + count * sizeof(uint32_t);
                if (length > consumed) is.ignore(length - consumed);
            } else if ((tag == TAG_XFRM || tag == TAG_PIPE) && length > 0) {
                std::string xfrm_str(length, '\0');
                is.read(xfrm_str.data(), length);
                std::stringstream xfrm_ss(xfrm_str, std::ios::in | std::ios::binary);
                data.transformations = deglib::transformation::deserialize_transformations(xfrm_ss);
            } else {
                // Forward compatibility: skip unknown extension tags
                is.ignore(length);
            }
        }
    }

    return data;
}

/**
 * Universal graph serializer.
 * 1. Writes standard 8-byte graph header (metric_type, dim, size, edges_per_vertex) at offset 0.
 * 2. Writes raw contiguous vertex records.
 * 3. Appends optional "DEGX" extension trailer with TLV blocks (Prefetch, Entry points, Transforms).
 */
inline bool save_graph(
    const char* path,
    const deglib::graph::InternalGraph& graph,
    const std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>* transformations = nullptr
) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream os(path, std::ios::binary | std::ios::out);
    if (!os.is_open()) {
        std::fprintf(stderr, "save_graph: cannot open file %s for writing\n", path);
        return false;
    }

    // 1. Standard 8-byte graph header
    uint8_t metric_type = static_cast<uint8_t>(graph.getFeatureSpace().metric().value);
    uint16_t dim = uint16_t(graph.getFeatureSpace().dim());
    uint32_t graph_size = graph.size();
    uint8_t edges_per_vertex = graph.getEdgesPerVertex();

    os.write(reinterpret_cast<const char*>(&metric_type), sizeof(metric_type));
    os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    os.write(reinterpret_cast<const char*>(&graph_size), sizeof(graph_size));
    os.write(reinterpret_cast<const char*>(&edges_per_vertex), sizeof(edges_per_vertex));

    // 2. Vertex records
    const size_t feat_size = graph.getFeatureSpace().get_data_size();
    const size_t neighbor_bytes = size_t(edges_per_vertex) * sizeof(uint32_t);
    const size_t weight_bytes = size_t(edges_per_vertex) * sizeof(float);
    std::vector<float> dummy_weights(edges_per_vertex, 0.0f);

    for (uint32_t i = 0; i < graph_size; ++i) {
        os.write(reinterpret_cast<const char*>(graph.getFeatureVector(i)), feat_size);
        os.write(reinterpret_cast<const char*>(graph.getNeighborIndices(i)), neighbor_bytes);
        os.write(reinterpret_cast<const char*>(dummy_weights.data()), weight_bytes);
        uint32_t lbl = graph.getExternalLabel(i);
        os.write(reinterpret_cast<const char*>(&lbl), sizeof(uint32_t));
    }

    // 3. Optional Extension Trailer (DEGX)
    const bool has_pref = (graph.getPo() != 8 || graph.getPl() != 3 || graph.getNl() != 3);
    const bool has_entries = !graph.getEntryVertexIndices().empty();
    const bool has_transforms = (transformations != nullptr && !transformations->empty());

    if (has_pref || has_entries || has_transforms) {
        uint32_t magic = TRAILER_MAGIC;
        os.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

        // Prefetch tuning
        if (has_pref) {
            uint32_t tag = TAG_PREF;
            uint32_t len = 12;
            int32_t po = graph.getPo();
            int32_t pl = graph.getPl();
            int32_t nl = graph.getNl();
            os.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
            os.write(reinterpret_cast<const char*>(&len), sizeof(len));
            os.write(reinterpret_cast<const char*>(&po), sizeof(po));
            os.write(reinterpret_cast<const char*>(&pl), sizeof(pl));
            os.write(reinterpret_cast<const char*>(&nl), sizeof(nl));
        }

        // Entry vertex indices
        if (has_entries) {
            const auto& entries = graph.getEntryVertexIndices();
            uint32_t tag = TAG_ENTR;
            uint32_t count = static_cast<uint32_t>(entries.size());
            uint32_t len = sizeof(uint32_t) + count * sizeof(uint32_t);
            os.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
            os.write(reinterpret_cast<const char*>(&len), sizeof(len));
            os.write(reinterpret_cast<const char*>(&count), sizeof(count));
            os.write(reinterpret_cast<const char*>(entries.data()), count * sizeof(uint32_t));
        }

        // Transforms
        if (has_transforms) {
            std::ostringstream xfrm_ss(std::ios::binary);
            deglib::transformation::serialize_transformations(*transformations, xfrm_ss);
            std::string xfrm_str = xfrm_ss.str();
            uint32_t tag = TAG_XFRM;
            uint32_t len = static_cast<uint32_t>(xfrm_str.size());
            os.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
            os.write(reinterpret_cast<const char*>(&len), sizeof(len));
            os.write(xfrm_str.data(), len);
        }
    }

    os.flush();
    return os.good();
}

}  // namespace deglib::serialization
