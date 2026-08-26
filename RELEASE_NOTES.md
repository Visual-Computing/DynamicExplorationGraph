# Release Notes

## deglib v0.2.1

### Highlights
* **New Distance Metrics:**
  * **`FP16_L2` Metric:** Added native IEEE 754 half-precision Euclidean distance (`Metric::FP16_L2`) with AVX-512, AVX2 (F16C/FMA), and Scalar kernels, fully integrated into C++ `FloatSpace` and Python bindings (`deglib.Metric.FP16_L2`).
  * **`Int8_InnerProduct` Metric:** Added signed 8-bit integer inner product (`Metric::Int8_InnerProduct`) with AVX-512, AVX512-VNNI, AVX2, AVX2-VNNI, and Scalar kernels.
  * **`Int8_L2` Metric:** Added signed 8-bit integer L2 distance (`Metric::Int8_L2`) with AVX-512, AVX2, and Scalar kernels, including Python bindings support (`np.int8`).
  * **`Uint8_InnerProduct` Metric:** Added native 8-bit unsigned integer inner product distance (`Metric::Uint8_InnerProduct`) with AVX-512, AVX512-VNNI, AVX2, AVX2-VNNI, and Scalar kernels.
* **Scalar Quantization (`deglib::optimization::quantization`):**
  * Added `SymCalibratorInt8` for symmetric Int8 quantization ($[-127, 127]$).
  * Added `AffineCalibratorUint8` for affine/asymmetric UInt8 quantization ($[0, 255]$ with scale and offset).
  * Python APIs exposed via `deglib.optimization.quantize_sym_int8(data)` and `deglib.optimization.quantize_affine_uint8(data)`.
* **SIMD & VNNI Optimizations:**
  * Added dedicated AVX-VNNI and AVX512-VNNI hardware acceleration for 8-bit integer dot products (`Int8_IP` and `UInt8_IP`).
  * Accelerated 8-bit integer distance computations with optimized SIMD horizontal reductions in AVX-512 (`_mm512_reduce_add_epi32`) and AVX2.
  * Centralized CPU instruction set detection and fallback resolution in `deglib::cpu::resolve_instruction_set()`.
  * Fixed SIMD instruction fallback selection in distance kernels when running on VNNI-capable CPUs.
* **Cross-Platform & Header Consistency Fixes:**
  * Added missing `<stdexcept>` header in `cpu.h` fixing `std::runtime_error` compilation errors on GCC and Clang (Linux / macOS).
  * Added missing `<cstring>` includes across integer distance headers for portable builds.
  * Fixed GCC/Clang inlining error (`target specific option mismatch` in `_mm256_cvtph_ps`) by including `avx2,f16c,fma` in `DEGLIB_TARGET_AVX512` and `DEGLIB_TARGET_AVX512_VNNI` target attributes.
  * Renamed `evp_inner_product.h` to `evp_ip.h` for consistent header naming convention across distance modules.
* **Python Memory Leak Fix:** Resolved memory leak in Python bindings when sequentially loading multiple graph instances from disk.
* **VIBE Benchmark Suite (`examples/vibe`):**
  * Added complete Vector Index Benchmark (VIBE) runner with automated Hugging Face HDF5 dataset downloads, grid evaluations, Pareto frontier plotting, and default FLAS pre-sorting.
  * Extended VIBE DegANN with support for query feature quantization, FP16 reranking, and MRNG edge pruning.
  * Enhanced benchmark visualizer with color-coded targets, $k$-intensity curves, and dashed styling for pruned graphs.
* **Sliding Window Streaming Demo (`examples/sliding_window`):** Added an interactive dynamic graph benchmark showcasing real-time sliding window ingestion (continuous additions and deletions).
* **Build & CI Improvements:**
  * Configured MSVC `/MP` (multi-threaded compilation) and `/bigobj` (extended object section limit) in CMake compilation options.
  * Updated Read the Docs webhook trigger in CI to pass authentication tokens as POST form data.
  * Updated `python/README.md` and removed obsolete memory safety documentation.
---

## deglib v0.2.0

### Overview
deglib v0.2.0 is a major release featuring API consolidation, dynamic graph streaming capabilities, runtime SIMD dispatching (AVX-512 / AVX2), FLAS 1D layout pre-sorting, EVP & FP16 quantization, and bit-exact cross-platform determinism across Windows, Linux, and macOS.

---

### ⚠️ Breaking Changes & Migration Guide

#### 1. Unified Search & Span API (C++)
* **Raw pointer removal:** `search()` and `search_intern()` no longer accept raw `const std::byte*` pointers. Queries are now passed via bounds-checked `std::span<const float>`.
* **Standardized parameter order:**
  ```cpp
  // Old:
  graph.search(query_ptr, eps, k);
  
  // New (consistent across all graph types):
  graph.search(query_span, k, eps, include_entry, filter, max_dist);
  ```
* **Unified internal search:** `explore()` was merged into `search_intern()`.
* **Pathfinding:** `ReadOnlyGraph::hasPath()` now consistently includes the target vertex in the returned path (matching `SizeBoundedGraph`).

#### 2. Namespace & Header Organization (C++)
* All modules are now organized under clean functional namespaces:
  * Distance metrics & spaces: `deglib::distances::*`
  * Graph builders & pruning: `deglib::builder::*`, `deglib::optimization::*`
  * Search & filters: `deglib::search::*`
  * Graph analysis: `deglib::analysis::*`
  * SIMD & CPU detection: `deglib::cpu::*`
* Single top-level include: `#include <deglib/deglib.h>`.

#### 3. Metric Renaming & Identifier Access
* Metrics follow the consistent format `DataType_MetricName` (e.g. `Metric::FP32_L2`, `Metric::FP32_InnerProduct`, `Metric::Uint8_L2`, `Metric::FP16_InnerProduct`, `Metric::EVP_InnerProduct`).
* `ObjectDistance::getInternalIndex()` was renamed to `ObjectDistance::getIdentifier()`.
* `FeatureRepository` and `StaticFeatureRepository` were removed from the core library to maintain a clean graph-only abstraction.

---

### 🚀 Key Features & Algorithms

#### FLAS (Fast Linear Assignment Sorter)
* **1D Layout Pre-Sorting:** FLAS organizes vector datasets into a 1D sequence using a Self-Organizing Map (SOM) grid, moving-average filters, and an allocation-free Jonker-Volgenant linear assignment solver on quantized cost matrices.
* **Performance Impact:** Laying out high-dimensional vectors along a 1D locality curve prior to graph construction significantly improves CPU cache locality during traversal, resulting in faster graph building and higher search QPS.
* **Deterministic Parallelism:** Employs thread-local scratch buffers and atomic position locks with sequential vertex ingestion, guaranteeing deterministic vertex ordering regardless of thread count.
* **Python API:** Available via `deglib.optimization.presort()`, returning the permutation array for custom preprocessing.

#### MIPS (Maximum Inner Product Search) Transformation
* **Space Transformation:** Maps $d$-dimensional vectors into $(d+1)$ dimensions by appending $\sqrt{M^2 - \|x\|^2}$ (where $M = \max_i \|x_i\|$), translating inner product maximization into Euclidean distance minimization.
* **Universal Indexing:** Enables unmodified, high-performance L2 exploration graphs to run exact and approximate inner-product search on unnormalized embeddings (e.g., LLaMA, OpenAI).
* **Python API:** Available via `deglib.optimization.mips_l2_transform()`.

#### Edge Pruning & Graph Analysis
* **4 Consolidated Heuristics:** Unified in `deglib::optimization::pruning`:
  * `prune_worst_edges`: Removes edges with the highest local distance values.
  * `remove_non_mrng_edges`: Enforces the Monotonic Relative Neighborhood Graph (MRNG) criterion.
  * `remove_non_mrng_edges_weight_sorted`: Weight-sorted MRNG pruning.
  * `remove_non_mrng_edges_iterative`: Iterative edge minimization.
* **Topology Analysis (`deglib::analysis`):** Tools for validating graph reachability (detecting isolated islands) and vertex degree distribution statistics.
* **Python API:** Available via `deglib.optimization.prune_worst_edges()`.

#### Quantization: EVP & Native FP16
* **EVP (Exact Vector Projection):** Bit-packed Top-K ternary masks ($2 \times \text{dim}/8$ bytes/vector) with hardware-accelerated symmetric inner product calculation (`Metric::EVP_InnerProduct`).
* **FP16 Inner Product:** Native IEEE 754 half-precision float distance kernel (`Metric::FP16_InnerProduct`) with AVX-512 / AVX2 F16C hardware acceleration and scalar round-to-nearest-even fallback.
* **Safety Fix:** Resolved prototype buffer overflows by ensuring 64-bit aligned SIMD memory access (`_mm_storel_epi64`).

#### DynamicGraph: Chunked Memory Management for Streaming
* **Power-of-Two Allocation:** Memory is structured in chunks of $2^k$ vertices with $O(1)$ bit-shift address resolution (`index >> k`, `index & (2^k - 1)`), eliminating monolithic `realloc` spikes and pointer invalidations.
* **Slot Recycling & Tombstones:** Deleted vertices are marked with `INVALID_LABEL` and recycled via an `IndexPool` ring buffer. Completely empty chunks are automatically freed.
* **Lifecycle Tracking:** `BuilderStatus` tracks ID changes per update (`step_added_ids`, `step_deleted_ids`, `total_added_ids`, `total_deleted_ids`).

#### Dynamic SIMD CPU Dispatching
* **Runtime Feature Detection:** Automatically detects CPU capabilities (AVX-512, AVX2, VNNI, F16C) via cached CPUID checks at startup without requiring separate compile-time binaries.
* **ResidualMode Branch Elimination:** C++20 `if constexpr` avoids scalar tail loops when vector dimensions align with SIMD registers.
* **Batch Distance Kernel (`compare_batch`):** Computes distances from one query to 4–8 database vectors in a single pass by keeping the query pinned in SIMD registers.
* **Python Inspection:** Active SIMD kernels can be queried via `FloatSpace.get_instruction()`.

#### 100% Bit-Exact Cross-Platform Determinism
* **Portable PRNG:** Custom 32-bit Xorshift PRNG and `DeterministicUniformIntDistribution` guarantee identical pseudo-random sequences across Windows, Linux, and macOS (x86_64 & Apple Silicon).
* **Numeric Consistency:** Scalar fallbacks utilize single-rounding `std::fma` to eliminate compiler-specific math drift.
* **Verification:** CI tests validate graph structure and search results against 64-bit FNV-1a checksums.

---

### 🐍 Python API Summary

All core capabilities are exposed via Python with native NumPy array support:

* **High-Level Graph Creation:**
  * `deglib.build_from_data(data, metric, ...)`: Direct graph construction and optimization from 2D NumPy arrays.
  * `deglib.create_empty(...)` & `deglib.create_random_graph(...)`: Graph factory constructors.
* **Serialization:** `deglib.load_readonly_graph(path)` and `deglib.load_dynamic_graph(path)`.
* **Optimization Module:**
  * `deglib.optimization.presort(data, metric, ...)`: FLAS 1D layout sorting.
  * `deglib.optimization.mips_l2_transform(vectors)`: MIPS $(d \rightarrow d+1)$ transformation.
  * `deglib.optimization.prune_worst_edges(graph)`: Graph edge thinning.
* **Hardware & Batch Distances:**
  * `FloatSpace.get_instruction()`: Introspects active SIMD backend (`"AVX512"`, `"AVX2"`, `"Scalar"`).
  * `FloatSpace.compute_distances(query, targets)`: Fast 1-to-N batch distance computation.
* **Predictable Search Output:** Searches return uniform `(num_queries, k)` NumPy arrays with `uint32_max` (IDs) and `NaN`/`inf` (distances) padding for unreachable neighbors.
