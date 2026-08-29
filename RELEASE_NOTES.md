# Release Notes

## deglib v0.2.3

### Overview
deglib v0.2.3 introduces a zero-overhead C++20 templated **`Searcher`** pipeline for inference (`deglib::search::Searcher` / `deglib.search.create_searcher`), eliminating Python bridge and memory allocation overhead in single-query loops by bundling query quantization, graph search, and exact SIMD candidate reranking into a single native invocation. Additionally, it brings official VIBE distance-tolerance recall calculation and clean C++20 `std::span` convenience interfaces.

---

### 🚀 Key Features & Improvements

#### High-Performance Inference Pipeline (`deglib::search::Searcher`)
* **Zero-Overhead End-to-End Querying:** Integrates query quantization (FP32/FP16 $\rightarrow$ INT8/UINT8/EVP), graph traversal on quantized indices, and SIMD candidate refinement against FP16/FP32 base features in a single C++ call.
* **Static Template Specialization (`SearcherImpl<QuantT, RefinerT>`):**
  * Compile-time `if constexpr` branch elimination avoiding virtual method dispatch in the hot search loop.
  * Full support for all quantizers (`NoQuantizer`, `ScalarInt8Quantizer`, `ScalarInt8PerDimQuantizer`, `ScalarUint8Quantizer`, `ScalarUint8PerDimQuantizer`, `EVPQuantizer`).
  * Support for unquantized and quantized candidate refiners (`NoRefiner`, `ExactRefiner<uint16_t>`, `ExactRefiner<float>`).
* **Modern C++20 Interface:**
  * `std::span<const T>` and `SearchResult` helpers for safe, expressive, and allocation-free querying.
  * Direct result count reporting (`uint32_t count`) indicating valid found neighbors.
  * Fast `unsorted=true` mode skipping heap sorting for pure candidate set retrieval.
  * Optional `return_distances=true` flag for returning both neighbor IDs and exact distance values.
* **Python Factory & Bindings (`deglib.search.create_searcher`):**
  * Automatic type deduction from graph and quantizer instances without manual `query_dtype` flags.

#### VIBE Benchmark Enhancements (`examples/vibe`)
* **Official VIBE Distance-Tolerance Recall:** Implemented official ground-truth distance threshold evaluation ($t = \text{gt\_distances}[k-1] + 10^{-3}$) using exact metric math (`euclidean`, `cosine`, `normalized`, `ip`) identical to `vibe/distance.py`.
* **Single-Query Evaluation Loop:** Benchmark runner uses `adapter.query(v, n)` powered by the C++ `Searcher` with `unsorted=True` for peak single-threaded QPS without Python bridge latency.
* **Garbage Collection & Cooldown:** Memory cleanup and settling pause after graph fitting.

---

## deglib v0.2.2
### Overview
deglib v0.2.2 introduces a strictly object-oriented **Fit-then-Quantize** scalar quantization suite (`ScalarQuantizerInt8`, `ScalarQuantizerInt8PerDim`, `ScalarQuantizerUint8`, `ScalarQuantizerUint8PerDim`), `make_scalar_quantizer_*` factory functions, full native FP16 (`uint16_t` / `np.float16`) and FP32 quantization support across all classes, and removes legacy one-shot procedural helpers.

---

### 🚀 Key Features & Improvements

#### State-Aware Scalar Quantization (`deglib::optimization::quantization`)
* **Pure Object-Oriented Design:** Quantizers preserve learned database distribution parameters across database batches and query vectors via `.fit()`, `.quantize()`, `.fit_quantize()`, and `.dequantize()`.
* **Complete Symmetrical Quantizer Suite:**
  * `ScalarQuantizerInt8`: Global symmetric signed Int8 mapping ($[-127, 127]$) for Cosine and Inner Product spaces.
  * `ScalarQuantizerInt8PerDim`: Per-dimension symmetric signed Int8 mapping ($[-127, 127]$).
  * `ScalarQuantizerUint8`: Global affine unsigned 8-bit mapping ($[0, 255]$ with dynamic scale and offset calibration) for Euclidean ($L_2$) search.
  * `ScalarQuantizerUint8PerDim`: Per-dimension affine unsigned 8-bit mapping ($[0, 255]$).
* **Native FP16 Support:** All quantizers natively accept both FP32 (`float`) and IEEE 754 half-precision FP16 (`uint16_t` / `np.float16`) inputs.
* **Factory Functions (`make_scalar_quantizer_*`):**
  * `make_scalar_quantizer_int8(vectors, drop_ratio=0.0)`
  * `make_scalar_quantizer_int8_perdim(vectors, drop_ratio=0.0)`
  * `make_scalar_quantizer_uint8(vectors, drop_ratio=0.0)`
  * `make_scalar_quantizer_uint8_perdim(vectors, drop_ratio=0.0)`
* **Python API Clean-Up:** Removed obsolete procedural batch helpers (`quantize_int8`, `quantize_uint8`, etc.) in favor of the pure `ScalarQuantizer*` classes and `make_scalar_quantizer_*` factory methods in `deglib.optimization`.

---

## deglib v0.2.1

### Overview
deglib v0.2.1 expands metric support with native half-precision and 8-bit integer formats, introduces scalar quantization tools with AVX-VNNI hardware acceleration, fixes a sequential memory leak in Python bindings, and adds the comprehensive VIBE benchmark suite.

---

### 🚀 Key Features & Improvements

#### New Distance Metrics
* **`FP16_L2` Metric:** Native IEEE 754 half-precision Euclidean distance (`Metric::FP16_L2`) with AVX-512, AVX2 (F16C/FMA), and Scalar kernels, fully integrated into C++ `FloatSpace` and Python bindings (`deglib.Metric.FP16_L2`).
* **`Int8_InnerProduct` Metric:** Signed 8-bit integer inner product (`Metric::Int8_InnerProduct`) with AVX-512, AVX512-VNNI, AVX2, AVX2-VNNI, and Scalar kernels.
* **`Int8_L2` Metric:** Signed 8-bit integer Euclidean distance (`Metric::Int8_L2`) with AVX-512, AVX2, and Scalar kernels, including Python bindings support (`np.int8`).
* **`Uint8_InnerProduct` Metric:** 8-bit unsigned integer inner product distance (`Metric::Uint8_InnerProduct`) with AVX-512, AVX512-VNNI, AVX2, AVX2-VNNI, and Scalar kernels.

#### SIMD & Hardware Acceleration (AVX-VNNI)
* **VNNI Acceleration:** Dedicated AVX-VNNI and AVX512-VNNI hardware vector dot product instructions for 8-bit integer inner products (`Int8_IP` and `UInt8_IP`).
* **Fast Horizontal Reductions:** Accelerated integer distance evaluation using optimized SIMD horizontal reductions in AVX-512 (`_mm512_reduce_add_epi32`) and AVX2.
* **Centralized CPU Dispatch:** Streamlined CPU instruction set detection and runtime fallback resolution in `deglib::cpu::resolve_instruction_set()`.

#### Benchmarks & Example Workflows
* **VIBE Benchmark Suite (`examples/vibe`):** Added Vector Index Benchmark runner with automated Hugging Face dataset downloads, grid evaluations, interactive Plotly HTML visualizations with GUI log exploration, query feature quantization, and FP16/FP32 candidate reranking.
* **Sliding Window Streaming Benchmark (`examples/sliding_window`):** Added dynamic streaming experiment reproducing continuous updates against DEG from the CleANN benchmark.

---

### 🛠️ Bug Fixes & Stability

* **Python Memory Leak:** Resolved a memory leak in Python bindings when sequentially loading or converting multiple graph instances from disk.
* **Compiler Compatibility (GCC / Clang):**
  * Added missing `<stdexcept>` header in `cpu.h` fixing `std::runtime_error` compilation failures on Linux and macOS.
  * Added missing `<cstring>` headers across integer distance implementations for standard portability.
  * Fixed GCC/Clang inlining errors (`target specific option mismatch` in `_mm256_cvtph_ps`) by including `avx2,f16c,fma` in `DEGLIB_TARGET_AVX512` target attributes.
* **Header Consistency:** Renamed `evp_inner_product.h` to `evp_ip.h` for uniform naming across all distance headers.
* **Deterministic CI Tests:** Configured fixed seeds across graph integration tests to guarantee 100% test reproducibility across all platforms and Python versions.

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
