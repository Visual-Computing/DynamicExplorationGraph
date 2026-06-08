#pragma once

/**
 * @file hdf5_reader.h
 * @brief Pure C++ header-only HDF5 binary reader (no libhdf5).
 *
 * This is a shim that includes the modular sub-headers under evp/hdf5/.
 *
 * Supports:
 *  - HDF5 Superblock version 0
 *  - B-Tree version 1 (leaf + internal nodes) + Symbol Table Nodes (SNOD)
 *  - Groups (Object Header v1 with Symbol Table message, type 17)
 *  - Object Header version 1 (datasets)
 *  - Object Header version 2 (with "OHDR" signature)
 *  - Contiguous data layout (Layout Class 1), uncompressed
 *  - 8-byte offsets and lengths
 *
 * Tested with benchmark-dev-wikipedia-bge-m3-small.h5 (SISAP 2026).
 *
 * File structure of this particular file:
 *   Root Group (Superblock root STE)
 *     ├── allknn  (Group, cache_type=1) → sub-B-tree → SNOD → allknn dataset
 *     ├── itest   (Group, cache_type=1)
 *     ├── otest   (Group, cache_type=1)
 *     └── train   (Dataset, cache_type=0)
 */

#include "hdf5/hdf5_types.h"
#include "hdf5/hdf5_io.h"
#include "hdf5/hdf5_superblock.h"
#include "hdf5/hdf5_heap.h"
#include "hdf5/hdf5_btree.h"
#include "hdf5/hdf5_snod.h"
#include "hdf5/hdf5_ohdr.h"
#include "hdf5/hdf5_scan.h"
#include "hdf5/hdf5_readers.h"
