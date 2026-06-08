# HDF5 Reader Refactoring Plan

## Goal
Split the monolithic `evp/hdf5_reader.h` (629 lines, single header) into a modular
structure under a new `evp/hdf5/` subdirectory. Zero behavioral changes.

## Current Problems
- Single 629-line header mixing IO helpers, format parsers, recursion, and public API.
- `parse_ohdr_v2` broken for `flags=0x20` (timestamps) — message boundaries misaligned.
- All inline in one header, hard to test individual parsers.

## HDF5 Versioning Overview

The HDF5 format has multiple versioned components. Our reader must handle each variant:

### Superblock
- **v0** (only version we support): 8-byte offsets, 8-byte lengths, root STE inline.
- Higher versions (1–9+) exist but are out of scope.

### Object Header (OHDr)
- **v1**: First byte is `1`, followed by `reserved(1)`, `n_msgs(2)`, `obj_ref_count(4)`,
  `header_size(4)`, `reserved(4)`. Messages are fixed 8-byte aligned with
  `type(2) + size(2) + flags(1) + reserved(3)`.
- **v2**: Starts with `"OHDr"` signature, then `version(1)`, `flags(1)`, optional
  timestamps (`flags>>5 & 1`), optional phase-change (`flags>>4 & 1`),
  chunk0 size (variable width per `flags & 0x3`), then messages with
  `type(2) + size(2) + flags(1) + [cre_order(2)]`.

### Data Layout Message (type 8)
- **v1/v2**: `dimensionality(1)`, `layout_class(1)`, `reserved(5)`, `data_addr(8)`.
  - `layout_class=1` → contiguous, address only (no length — compute from dataspace).
- **v3/v4**: `layout_class(1)`, then if `1` (contiguous):
  `data_addr(8)`, `data_len(8)`. Explicit byte length included.

### Dataspace Message (type 1)
- **v1**: `ds_ver(1)`, `rank(1)`, `flags(1)`, `reserved(5)`, then `rank` × `uint64` dims.
  If `flags & 0x01` (H5S_VALID_MAX): additional `rank` × `uint64` max dims follow.
- **v2+**: `ds_ver(1)`, `rank(1)`, `flags(1)`, `type(1)` (H5S_SIMPLE=1, etc.),
  then `rank` × `uint64` dims. If `flags & 0x01`: additional `rank` × `uint64` max dims follow.

### Datatype Message (type 3)
- `class_and_ver(1)`, `bitfields(3)`, `size_in_bytes(4)`.

### Symbol Table Message (type 17)
- `btree_root_rel(8)`, `heap_rel(8)`. Present only on groups.

### B-Tree
- **v1** (only version we support): `TREE` sig, `node_type(1)`, `node_level(1)`,
  `entries_used(2)`, `left_sibling(8)`, `right_sibling(8)`, then
  `key(8), child(8)` × N.

### SNOD (Symbol Table Node)
- `SNOD` sig, `version(1)`, `reserved(1)`, `count(2)`, then `count` × STE entries.
- Each STE: `name_off(8)`, `ohdr_rel(8)`, `cache_type(4)`, `reserved(4)`,
  `scratch_btree(8)`, `scratch_heap(8)`.

### Local Heap
- `HEAP` sig, `version(1)`, `reserved(3)`, `data_size(8)`, `free_list_off(8)`,
  `data_rel(8)`.

---

## SISAP 2026 Small File Structure (benchmark-dev-wikipedia-bge-m3-small.h5)

### Disk Files
```
C:\Data\ANN\sisap2026\small\
  benchmark-dev-wikipedia-bge-m3-small.h5
  allknn.ivecs
  train.fvecs
  train.hvecs
  glove/
```
Note: `itest.ivecs` is NOT on disk — the test `ReadItestKnns_MatchesIvecs` skips.

### Root Level
- **Root B-Tree**: 1 leaf SNOD (NOT 4). HDF5 packs all root-level entries into a
  single SNOD node regardless of how many groups/datasets exist.
- **Root SNOD**: Contains entries for `allknn`, `itest`, `otest`, `train`.
- **Root heap**: Stores group/dataset names as null-terminated strings, accessible
  via `name_off` in SNOD entries. Names are at arbitrary offsets within the heap
  data block — never assume offset 0, and never use `strstr` on raw heap data
  (padding and free-list entries can confuse naive string search).

### Group: `itest` (OHDr v1 ✅)
- Uses OHDr v1 (first byte = `1`) — parsed correctly.
- Sub-B-Tree: 1 leaf SNOD with 3 entries.
- Group heap: contains "knns", "dists", "queries" as null-terminated strings.
- Datasets:
  - `itest/knns`    — 10000 × 1000 int32   (40 MB)
  - `itest/dists`   — 10000 × 1000 float32 (40 MB)
  - `itest/queries` — 10000 × 1024 float32 (40.96 MB)

### Group: `allknn` (OHDr v2 ✅)
- Uses OHDr v2 with `flags=0x20` (timestamps enabled: `flags>>5 & 1`).
- Sub-B-Tree: 1 leaf SNOD with 2 entries (`dists`, `knns`).
- Sub-datasets: `allknn/dists` (200000 × 32 float32, 25600000 bytes), `allknn/knns` (200000 × 32 int32, 25600000 bytes)
- `allknn/knns` data is 1-indexed (indices 1..200000). Verified 100% match against `allknn.ivecs` ground truth.

### Group: `otest` (OHDr v2 ✅)
- Uses OHDr v2 with `flags=0x20` (timestamps) for `dists`/`knns`, and `flags=0x00` for `queries`.
- Sub-datasets:
  - `otest/dists`   — 10000 × 1000 float32 (40000000 bytes)
  - `otest/knns`    — 10000 × 1000 int32  (40000000 bytes)
  - `otest/queries` — 10000 × 1024 fp16   (20480000 bytes)
- No ground truth files on disk. Sanity checks pass: indices in [1, 200000], distances >= 0, query norms ≈ 1.0.

### Dataset: `train` (OHDr v2 ✅)
- Uses OHDr v2 with `flags=0x00` (no timestamps).
- 200000 × 1024 fp16 (409600000 bytes)
- Verified 100% match against `train.hvecs` ground truth (all 204,800,000 FP16 values).

### Large File (benchmark-dev-wikipedia-bge-m3.h5, 14.2 GB)
Same structure as small, but with larger train/allknn datasets:
- `train`: 6350000 × 1024 fp16 (13,004,800,000 bytes) — verified vs train.hvecs ✅
- `allknn/dists`: 6350000 × 32 float32 (812,800,000 bytes)
- `allknn/knns`: 6350000 × 32 int32 (812,800,000 bytes) — verified vs allknn.ivecs ✅
- `itest/*`, `otest/*`: same dimensions as small (10000 queries)

### OHDr v2 Known Issue (Resolved)
OHDr v2 parsing has been fixed. The issues were:
1. Message type field is 1 byte (not 2) in v2.
2. No 8-byte message alignment between messages in v2.
3. 4-byte CRC32 checksum at end of chunk0 must be skipped.
4. Dataspace v2 has a `type` byte after `flags` that must be skipped before reading dimensions.
5. Dataspace `flags & 0x01` (H5S_VALID_MAX) means max dimensions follow current dimensions.
All datasets (train, allknn/*, otest/*) are now correctly discovered with proper dimensions.

### Indexing Note
- SISAP `knns` datasets use **mixed indexing** — do not assume uniform behavior:
  - **1-indexed**: `itest/knns` (int32, indices 1..200000), `allknn/knns` (int32, indices 1..200000),
    `test/knns` in llama-dev.h5 (INT64, indices 1..256921)
  - **0-indexed**: `otest/knns` (int32, indices 0..199999) — the only exception
- `allknn.ivecs` is also 1-indexed. Comparison is direct: `hdf5_data[i] == ivecs_data[i]`
  (no -1 offset needed).
- `itest.ivecs` (if it existed) may also be 1-indexed — the existing test subtracts 1 as a
  precaution, but this depends on how the file was generated.

### SISAP File Examples

#### Small File (benchmark-dev-wikipedia-bge-m3-small.h5)
- SISAP 2026 benchmark dataset.
- Groups: `itest`, `otest`, `allknn`, `train`.
- `allknn/knns` verified 100% match against `allknn.ivecs` ground truth.

#### Large File (benchmark-dev-wikipedia-bge-m3.h5, 14.2 GB)
- Same structure as small, but with larger train/allknn datasets.
- `allknn/knns` verified vs allknn.ivecs ✅

#### llama-dev (llama-dev.h5)
- **Disk File**: `C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5`
- **SISAP-generated file** — same Python conversion pipeline as other SISAP datasets.
- **Root Level**:
  - `train` dataset (FP32, 256921 x 128)
  - `test` group
- **Group `test`**:
  - Contains `queries` (FP32, 1000 x 128), `dists` (FP64 / double, 1000 x 100), and `knns` (INT64, 1000 x 100).
  - Uses HDF5 Object Header Continuation Messages (Type 16) and Link Messages (Type 6).
  - Stale B-Tree/SNOD structures only contained 2 entries (missing `knns`). The Link Messages (Type 6) represent the up-to-date group structure.
  - Reader updated to follow recursive continuation blocks and parse Link Messages directly, bypassing stale B-Tree/SNOD nodes when `links` are found.

---

## Current Structure

    evp/
      hdf5_reader.h              # shim: includes all below (backward compat)
      evp_common.h               # now_ms(), hvecs_read() (legacy .hvecs support)
      deglib_evp_test.cpp        # main benchmark (HDF5 + legacy .hvecs/.ivecs)
      deglib_evp_microbench.cpp  # SIMD microbenchmark
      hdf5/
        structure.md              # this file
        hdf5_types.h              # DatasetInfo, OhdrInfo, SteEntry, Superblock, UNDEF64, SIG_*
        hdf5_io.h                 # r16/r32/r64, frd8/16/32/64, fskip, fseek, heap_str
        hdf5_superblock.h         # parse_superblock() — superblock v0 only
        hdf5_heap.h               # parse_heap()
        hdf5_btree.h              # traverse_btree() — btree v1 only
        hdf5_snod.h               # parse_snod()
        hdf5_ohdr.h               # apply_msg(), parse_ohdr_v1(), parse_ohdr_v2(), parse_ohdr()
        hdf5_scan.h               # collect_datasets(), scan_datasets(), print_datasets() (supports recursive continuation & links)
        hdf5_readers.h            # read_fp16_vectors(), read_int32_flat(), read_matrix_int64(), read_matrix_fp64()
        test/
          test_main.cpp           # gtest entry point
          test_types.cpp          # type struct defaults, constants
          test_io.cpp             # r16/r32/r64 endian, heap_str
          test_structural.cpp     # heap, btree, snod against real file
          test_ohdr.cpp           # OHDr v1/v2 parsing
          test_integration.cpp    # end-to-end scan + readers + ground truth (small)
          test_dataset_dims.cpp   # per-dataset dimension verification (small)
          test_sanity.cpp         # data integrity: indices, distances, norms (small)
          test_integration_large.cpp  # end-to-end scan + readers + ground truth (large)
          test_dataset_dims_large.cpp # per-dataset dimension verification (large)
          test_sanity_large.cpp   # data integrity: indices, distances, norms (large)
          test_llama_dev.cpp      # end-to-end structure and values verification for llama-dev (uses Link & Continuation msgs)

## Dependency Order
    1. hdf5_types.h       (no deps)
    2. hdf5_io.h          (deps: types)
    3. hdf5_superblock.h  (deps: types, io)
    4. hdf5_heap.h        (deps: types, io)
    5. hdf5_btree.h       (deps: types, io)
    6. hdf5_snod.h        (deps: types, io)
    7. hdf5_ohdr.h        (deps: types, io)
    8. hdf5_scan.h        (deps: types, io, superblock, heap, btree, snod, ohdr)
    9. hdf5_readers.h     (deps: types, io)

## Key Rules
- All functions remain inline (header-only, no new compilation units)
- All code stays in namespace hdf5_reader (or hdf5_reader::detail)
- Existing `#include "hdf5_reader.h"` in all .cpp files works unchanged
- Each sub-header includes needed standard headers + prerequisite hdf5/ headers
- Version-specific parsing logic must be clearly separated within each parser
- **No dataset-specific functions in generic headers** — `hdf5_readers.h` only contains
  generic readers (read_fp16_vectors, read_int32_flat). Dataset-specific loading (e.g.
  "load train", "load allknn/knns") belongs in the consumer code (deglib_evp_test.cpp),
  never in the library headers.

## Test Design Notes (Lessons Learned)
- **Never use `strstr` on raw heap data** to verify name presence. Heap data may
  contain padding, free-list entries, or internal structures that break naive
  string search. Always use `heap_str(heap, name_off)` with SNOD entry offsets.
- **Root B-Tree may have only 1 leaf SNOD** even when the root group contains
  many entries. HDF5 packs multiple entries into a single SNOD node. Do not
  assume 1 SNOD leaf per group entry.
- **Root heap does not necessarily start with names at offset 0**. The heap
  data portion (returned by `parse_heap`) is a raw memory block with strings
  at arbitrary offsets. Use SNOD `name_off` values for access.
- **OHDr v2 with timestamps (`flags & 0x20`)** has been fixed. Message type is 1 byte,
  no 8-byte alignment, CRC32 at end of chunk0, and dataspace v2 type byte are all handled.
- **`itest.ivecs` is not on disk** — ground truth comparison test skips.
- **FP16 datasets**: `train` and `otest/queries` have element_size=2, both small and large files.
- **Large file (14.2 GB)**: all 9 datasets verified. train/allknn scaled to 6.35M rows.
- **63 small tests + 33 large tests + 5 llama-dev tests = 101 total (2 skipped, 99 passed, 0 failures).**
- Test include paths: `#include "hdf5_types.h"` (not `hdf5/hdf5_types.h`) when
  `evp/hdf5/` is set as the include directory in CMake (`add_hdf5_test` macro).
- **`heap_str` test vectors must null-terminate** — a `std::vector<uint8_t>`
  initialized from a braced list does not pad with zeros, so `heap_str` will
  read past the intended string into uninitialized memory. Always append `0`
  as the last element.

## Implementation Status
1. Create evp/hdf5/ directory ✅
2. Write each of the 9 files by cutting code from current hdf5_reader.h ✅
3. Write hdf5_reader.h shim that includes all 9 files in order ✅
4. Build all targets to verify zero regressions ✅
5. Add unit tests under evp/hdf5/test/ ✅
6. Fix test assumptions to match actual file structure ✅
7. Fix OHDr v2 + dataspace v2 parsing (type byte, max dims, timestamps) ✅
8. Add large file tests (14.2 GB, 6.35M vectors) ✅
9. 101 tests total: 99 passed, 2 skipped (itest.ivecs missing x2), 0 failed ✅

## Remaining Work
- **Add `itest.ivecs` ground truth file** to enable `ReadItestKnns_MatchesIvecs` test (both small and large).
- **Ground truth for `otest/*`** is not available on disk — sanity checks are the best we can do.
