#pragma once

#include <map>

#include "hdf5_types.h"
#include "hdf5_io.h"
#include "hdf5_superblock.h"
#include "hdf5_heap.h"
#include "hdf5_btree.h"
#include "hdf5_snod.h"
#include "hdf5_ohdr.h"

namespace hdf5_reader {
namespace detail {

// Forward declarations
inline void collect_datasets(
    std::ifstream& f,
    uint64_t btree_abs,
    uint64_t heap_abs,
    uint64_t base,
    const std::string& prefix,
    std::map<std::string, DatasetInfo>& out);

inline void collect_datasets_from_ohdr(
    std::ifstream& f,
    uint64_t ohdr_abs,
    uint64_t base,
    const std::string& prefix,
    std::map<std::string, DatasetInfo>& out)
{
    OhdrInfo ohdr = parse_ohdr(f, ohdr_abs, base);
    if (ohdr.is_group) {
        if (!ohdr.links.empty()) {
            for (const auto& l : ohdr.links) {
                std::string full_name = prefix.empty() ? l.name : (prefix + "/" + l.name);
                collect_datasets_from_ohdr(f, l.obj_abs, base, full_name, out);
            }
        } else if (ohdr.group_btree_abs != UNDEF64 && ohdr.group_heap_abs != UNDEF64) {
            collect_datasets(f, ohdr.group_btree_abs, ohdr.group_heap_abs, base, prefix, out);
        }
    } else if (ohdr.elem_size > 0 && ohdr.dim0 > 0 && ohdr.data_abs > 0) {
        DatasetInfo info;
        info.name = prefix;
        info.file_offset = ohdr.data_abs;
        info.total_bytes = ohdr.data_len;
        info.element_size = ohdr.elem_size;
        info.num_rows = ohdr.dim0;
        info.num_cols = ohdr.dim1;
        out[prefix] = info;
    }
}

// ============================================================================
// Collect all leaf datasets under a group recursively.
// prefix = path prefix, e.g. "" for root, "allknn" for /allknn
// ============================================================================
inline void collect_datasets(
    std::ifstream& f,
    uint64_t btree_abs,
    uint64_t heap_abs,
    uint64_t base,
    const std::string& prefix,
    std::map<std::string, DatasetInfo>& out)
{
    // Collect SNOD addresses from B-tree
    std::vector<uint64_t> snod_addrs;
    traverse_btree(f, btree_abs, base, snod_addrs);

    // Read local heap (string pool)
    std::vector<uint8_t> heap = parse_heap(f, heap_abs, base);

    // Process each SNOD
    for (uint64_t snod_abs : snod_addrs) {
        auto entries = parse_snod(f, snod_abs, base);
        for (const auto& ste : entries) {
            std::string name = heap_str(heap, ste.name_off);
            if (name.empty()) continue;
            std::string full_name = prefix.empty() ? name : (prefix + "/" + name);

            // Fetch metadata and fallback to scratch if ohdr parser didn't mark as group
            try {
                collect_datasets_from_ohdr(f, ste.ohdr_abs, base, full_name, out);
            } catch (const std::exception& ex) {
                // Fallback: use scratch fields from STE entry for groups
                if (ste.cache_type == 1 && ste.scratch_btree != UNDEF64 && ste.scratch_heap != UNDEF64) {
                    collect_datasets(f, ste.scratch_btree, ste.scratch_heap, base, full_name, out);
                }
            }
        }
    }
}

} // namespace detail

// ============================================================================
// Public API
// ============================================================================

/**
 * @brief Scan all datasets in an HDF5 file (recursive group traversal).
 * @return Map from dataset path (e.g. "train", "allknn/indices") → DatasetInfo
 */
inline std::map<std::string, DatasetInfo> scan_datasets(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error(
        "hdf5_reader: cannot open: " + filepath);

    auto sb = detail::parse_superblock(f);
    if (sb.root_btree == detail::UNDEF64 || sb.root_heap == detail::UNDEF64)
        throw std::runtime_error("hdf5_reader: root B-Tree or Heap address is undefined");

    std::map<std::string, DatasetInfo> result;
    detail::collect_datasets(f, sb.root_btree, sb.root_heap, sb.base, "", result);
    return result;
}

/**
 * @brief Print all found datasets (for debugging).
 */
inline void print_datasets(const std::map<std::string, DatasetInfo>& datasets) {
    std::cout << "HDF5 datasets found:\n";
    for (const auto& [name, info] : datasets) {
        std::cout << "  " << name
                  << "  rows=" << info.num_rows
                  << "  cols=" << info.num_cols
                  << "  elem=" << info.element_size << "B"
                  << "  offset=" << info.file_offset
                  << "  bytes=" << info.total_bytes << "\n";
    }
}

} // namespace hdf5_reader
