# Release Notes

## deglib v0.2.8

### Overview
deglib v0.2.8 fixes result array sizing for single-query searches in Python. When fewer candidates than $k$ are found (such as during intentionally small exploration radii $ef < k$), single-query results are directly trimmed to the exact number of found neighbors without `UINT32_MAX` sentinels or memory allocations. Batch query matrices continue to preserve their regular rectangular $[N, k]$ shape with standard `UINT32_MAX` padding.

### 🚀 Improvements & Fixes
* **Exact Single-Query Result Sizing (`Searcher.search`):**
  * Single queries (1D or single-row 2D) now return arrays sized to the exact count of found neighbors (`found <= k`), eliminating out-of-bounds `UINT32_MAX` (4294967295) indices when $ef < k$.
  * Zero-allocation in-place shape adjustment (`result.resize({found})`) with 0ns overhead for the standard $found = k$ path.
* **Benchmark & Example Cleanup:**
  * Removed obsolete sentinel filter checks (`- {4294967295, -1}`) in `examples/vibe/main.py`.

---

## deglib v0.2.7

### Overview
deglib v0.2.7 is a major architecture, performance, and tooling release (+26,500 lines across 62 commits since v0.2.6). It introduces an end-to-end **Zero-Overhead Query Execution Pipeline** eliminating Python bridge latency and memory allocations, a modular **Feature Transformation & Quantization Engine** with native index serialization (`DEGX` extension container), native **`ScalarQuantizerFP16`** with power-of-two exponent scaling, an **interleaved ReadOnlyGraph layout** for peak cache locality, a completely rewritten **Concrete Search & Reranker Architecture**, zero-copy ground-truth analysis with memory-mapped files, and an interactive **2D & 3D Graph Exploration Tool**.

---

### 🚀 Key Features & Improvements

#### 1. Zero-Overhead Query Execution & Searcher Unification
* **Unified Single & Batch Search (`Searcher.search`):**
  * `Searcher.search()` directly accepts 1D queries (`[D]`), 2D single-vector queries (`[1, D]` / `[D, 1]`), and 2D query batches (`[N, D]`) through a single unified C++ method.
  * Completely eliminated Python-side dimension checks, `.ravel()`, and `np.ascontiguousarray` calls in query loops.
  * C++ employs Zero-Copy C-contiguous buffer verification via NumPy C-API (`py::array::ensure`). If contiguous, memory is directly mapped without allocations or copies.
  * Direct GIL release during batch exploration with dynamic thread pool distribution.
* **Built-in Query Normalization (`normalize_query`):**
  * `Searcher` and `create_searcher(..., normalize_query=True)` support automatic in-place $L_2$ query normalization ($q \leftarrow q / \|q\|_2$) directly in thread-local scratch memory.
  * Queries are normalized once prior to graph traversal and candidate reranking, eliminating duplicate normalizations and Python math overhead.
* **Smart Searcher Optimization (`Searcher::optimize`):**
  * Automatically dequantizes quantized graph features (FP16, INT8, UINT8, EVP) to compute exact K-Means medoid entry points, capped at 512 clusters.
  * Tunes cache line prefetch offsets (`po`) and lengths (`pl`) for candidate reranking on the host CPU.

#### 2. Embedded Feature Transformation & Quantization Engine (`deglib::transformation`)
* **Pipeline Integration & Container Serialization (`DEGX` Trailer):**
  * Graphs can embed a sequence of feature transformations (`setTransformations`, `getTransformations`) that are automatically applied to incoming queries during search.
  * Graph files now support an extensible `DEGX` container trailer with Tag-Length-Value blocks for the transformation pipeline (`TAG_XFRM`), tuned prefetch settings (`TAG_PREF`), and entry points (`TAG_ENTR`), automatically restored on `load_readonly_graph()` and `load_dynamic_graph()`.
  * Transformation stages serialize using FourCC chunk tags (`TAG_SQ8`, `TAG_EVP`, `TAG_MIPS`, `TAG_NORM`, `TAG_FP16`).
* **Standalone `Reranker` & Query Transformation Pipeline:**
  * Introduced standalone candidate reranker (`deglib::search::Reranker` / `deglib.search.Reranker`) with dedicated `FloatSpace`, base vector storage, and prefetch auto-tuning (`optimize()`).
  * `Reranker` and `Searcher` accept query transformation pipelines (`rerank_transformations`), allowing candidates from quantized graphs (e.g. INT8 or EVP) to be scored against custom feature representations (e.g. FP16 base features) without separate manual conversions.
* **`ScalarQuantizerFP16` (New):**
  * High-precision FP32-to-FP16 cast transformation (`make_scalar_quantizer_fp16`).
  * Features power-of-two exponent scaling to map large feature magnitudes safely into the IEEE 754 float16 range without Inf/NaN or numerical underflow.
* **`L2NormalizeTransform` (New):**
  * Standalone serializable transformation stage (`make_l2_normalize_transform`, FourCC `NORM`) providing vector $L_2$ normalization for arbitrary pipeline workflows.
* **Data-Driven EVP Auto-Calibration (`EvpQuantizer`):**
  * `non_zeros=0` (default) evaluates candidate sparsity levels across logarithmic and linear scales during `fit()` and automatically selects the optimal non-zero bit count.
  * Supports per-dimension mean subtraction and SIMD-accelerated popcount distance execution (`Metric.EVP_InnerProduct`).

#### 3. Graph Architecture & Cache Locality
* **Interleaved ReadOnlyGraph Layout:**
  * `ReadOnlyGraph` stores feature data and outgoing edge neighbor lists in an interleaved memory layout, ensuring that traversing a vertex's edges automatically loads its feature into the L1/L2 CPU cache.
* **Topology & Path Reachability:**
  * Added `has_path` reachability validation to C++ `DynamicExplorationGraph` and Python bindings to verify graph connectivity and path reachability between arbitrary vertices.
  * Streaming data optimization: `OptimizationTarget::StreamingData` utilizes early termination during graph restoration to prevent rebuild degradation.
* **MRNG Terminology Alignment:**
  * Renamed legacy RNG methods and heuristics consistently to Monotonic Relative Neighborhood Graph (`MRNG`).

#### 4. Tooling, Visualization & Benchmarks
* **Interactive 2D/3D Graph Explorer (`examples/graph_2d`):**
  * Full interactive visualization tool for exploring DEG topology on synthetic and custom 2D distributions (Uniform, Gaussian, Spiral, Swiss Roll).
  * Real-time greedy search route visualization, MRNG edge violation inspection, and 3D visualization of the lifted MIPS dimension.
* **Zero-Copy Ground Truth & Graph Analysis (`deglib::analysis`):**
  * Ground-truth evaluation in `analyze_graph()` and `calc_graph_quality()` supports memory-mapped files (`mmap`) and strided array views with zero memory copies.
* **Builder Progress Interface:**
  * Replaced the string-based `"progress"` callback in Python with the clean boolean flag `show_progress=True`.
* **Enhanced VIBE Benchmark Suite:**
  * Integrated EVP quantization and FP16 candidate reranking in `examples/vibe`.
  * Automatic ground-truth metric detection and streamlined query evaluation loop.

---

### ⚠️ Breaking Changes & Migration Guide

* **Quantizer `drop_ratio` replaced by `quantile`:**
  All scalar quantizers (`ScalarQuantizerInt8`, `ScalarQuantizerUint8`, `ScalarQuantizer*PerDim`) now take `quantile` (e.g. `1.0` for no clipping, `0.99` for 1% clipping) instead of `drop_ratio`.
* **Transformation Module Imports:**
  Transformations must be imported from `deglib.transformation` (or directly from `deglib`) instead of `deglib.optimization`.
* **Search Parameter Handling:**
  Python `Searcher.search()` delegates directly to C++. Any callers passing unaligned non-array inputs should provide standard NumPy arrays.

---

### 🛠️ Bug Fixes

* **Reranker Memory Safety:** Validated base vector buffer dimensions against graph size and handled zero-candidate sets without out-of-bounds reads.
* **EVP Mean on FP16:** Fixed mean calculation across arbitrary dimensions for half-precision floating-point data.
* **Searcher Bypass:** Bypassed redundant candidate rerank execution when `rerank_factor == 1.0` and no reranker is attached.
* **Compiler & Platform Support:** Added `-mf16c` compilation flag to GCC/Clang AVX2 builds, resolved MSVC generator ABI compatibility, and updated GitHub Actions for Node 24 runtime support.

---

## deglib v0.2.6

### Overview
deglib v0.2.6 adds a second search strategy alongside the existing relative-epsilon search: **`ef` fixed-pool search**, now selectable through a single `eps_or_ef` query parameter. The new `ef` mode supports candidate filtering and a distance-computation budget. `Searcher::optimize()` gained K-Means entry-point selection and prefetch auto-tuning, and a set of graph-layout and SIMD improvements makes both build and search faster.

---

### 🚀 Key Features & Improvements

#### `ef` Fixed-Pool Search (new)
* **Second exploration strategy:** In addition to the existing relative-epsilon (`eps`) search, queries can now run a fixed candidate-pool (`ef`) best-first search.
* **Single `eps_or_ef` parameter:** One argument selects the mode — values $\ge 1.0$ are treated as a fixed pool size (`ef`), values $< 1.0$ run the relative-epsilon search. Applies to the C++ `Searcher` and the Python bindings.
* **Candidate filtering:** `ef` search accepts a filter predicate; filtered candidates guide exploration but never occupy the result set, so the search radius is not prematurely collapsed.
* **Distance-computation budget:** `ef` search can cap the number of distance evaluations per query (`max_distance_computation_count`) for predictable worst-case latency.

#### Searcher Tuning (`Searcher::optimize()`)
* **K-Means entry points:** Clusters graph features to pick better search entry points, improving recall at a given effort.
* **Prefetch auto-tuning:** Automatically tunes memory-prefetch parameters for the target machine.

#### Performance
* **Faster distance kernels:** Vectorized SIMD tail handling across the FP32/FP16/Int8/UInt8 distance headers, plus feature and neighbor prefetching during graph traversal.
* **Cache-friendly graph layout:** `ReadOnlyGraph` now stores features in a contiguous, aligned layout.

---

### ⚠️ Breaking Changes & Migration Guide

* **`Searcher` search parameter:** `search()` / `search_batch()` (C++ and Python) now take `eps_or_ef` in place of `eps`. Existing relative-epsilon callers keep the same behavior by passing a value $< 1.0$.
* **Graph creation & loading API:** `create_empty`, `create_dynamic_empty`, and `create_random_graph` moved to the `deglib` namespace, and `create_builder` / `load_readonly_graph` / `load_dynamic_graph` / `load_mutable_graph` are exposed from `deglib.h`. Update call sites to the new locations.
* **EVP quantization:** `EvpQuantizer` is now a stateful object shared by database and queries; the `quantize_evp_*` free functions were removed. Use the quantizer instance directly.

---

### 🛠️ Bug Fixes

* **Dynamic graph deletions:** Removing vertices from `SizeBoundedGraph` now keeps internal indices contiguous (swap-with-last compaction), fixing sliding-window graph extensions and `ReadOnlyGraph` serialization after deletions.

## deglib v0.2.5

### Overview
deglib v0.2.5 fixes a critical buffer overread bug in the AVX512 and AVX2 batch distance calculation kernels (`compare_batch`) for vector dimensions that require dual-SIMD and residual/tail processing (such as dimension 784 in MNIST / Fashion-MNIST, or other dimensions where $\text{dim} \pmod{32} \neq 0$ or $\text{dim} \pmod{64} \neq 0$).

---

### 🐛 Bug Fixes

* **SIMD Batch Distance Stepping (`compare_batch`):** Fixed chunk step and loop iteration counts in `HasDualSimd` blocks across `fp32_l2.h`, `fp32_ip.h`, `int8_l2.h`, and `uint8_l2.h`:
  * `L2Float_AVX512` / `InnerProductFloat_AVX512`: Dual-SIMD chunk count corrected to $\frac{\text{dim}}{32}$ (processing two 16-float vectors per iteration) instead of $\frac{\text{dim}}{16}$.
  * `L2Float_AVX2` / `InnerProductFloat_AVX2`: Dual-SIMD chunk count corrected to $\frac{\text{dim}}{16}$ (processing two 8-float vectors per iteration) instead of $\frac{\text{dim}}{8}$.
  * `L2Int8_AVX512` / `L2UInt8_AVX512`: Dual-SIMD chunk count corrected to $\frac{\text{dim}}{64}$ (processing two 32-byte vectors per iteration) instead of $\frac{\text{dim}}{32}$.
  * `L2Int8_AVX2` / `L2UInt8_AVX2`: Dual-SIMD chunk count corrected to $\frac{\text{dim}}{32}$ (processing two 16-byte vectors per iteration) instead of $\frac{\text{dim}}{16}$.
  * Prevents reading out-of-bounds memory in the subsequent `HasSimd` and `HasTail` branches, eliminating segmentation faults (Exit Code 139) and NaN distance calculations during graph construction and search on datasets with unaligned dimensions.

---

### 🧪 Testing

* **Unaligned Dimension Test Coverage:**
  * Added parameterized tests in Python (`test_build_unaligned_dimensions`) covering dimensions `48`, `80`, `100`, and `784` across `FP32_L2` and `FP32_InnerProduct`.
  * Expanded C++ distance test suite (`test_fp32_l2`, `test_fp32_inner_product`, `test_int8_l2`, `test_uint8_l2`) with unaligned dimension coverage in `VariousDims` and `MatchesSingleCompare`.

---

## deglib v0.2.4

### Overview
deglib v0.2.4 refactors the query execution pipeline to be fully stateless and thread-safe, eliminates query-time allocations, streamlines candidate reranking, and optimizes FP16 quantization kernels. It updates `Searcher` to accept search parameters (`eps`, `rerank_factor`) directly per query, adds contiguous flat batch search (`SearchResultBatch`), separates inner loop branches for result extraction, and consolidates the Python search interface.

---

### ⚠️ Breaking Changes & Migration Guide

* **Stateless `Searcher` API:** Removed mutable internal search parameters (`search_eps_`, `rerank_factor_`) and their setters/getters (`set_query_arguments`, `set_search_eps`, `set_rerank_factor`). `eps` and `rerank_factor` are now query-level parameters passed directly to `search()` and `search_batch()`.
* **Rerank API Consolidation (`deglib::search::rerank`):** Removed disparate internal helpers (`rerank_into`, etc.) and exposed overloaded `deglib::search::rerank` functions with explicit `return_distances` and `unsorted` boolean flags.
* **Unified Python `Searcher.search`:** Single entry point handling 1D vectors, row-vectors `(1, dim)`, column-vectors `(dim, 1)`, and 2D batch matrices `(N, dim)` with `threads`. Removed redundant `search_batch` from the Python class.

---

### 🚀 Key Features & Improvements

#### Stateless & Thread-Safe Search Execution (`deglib::search`)
* **Reentrant Querying:** Searcher instances are fully thread-safe and reentrant, allowing multiple threads to query the same instance with varying `eps` and `rerank_factor` trade-offs without data races or locks.
* **Branch-Free Result Extraction:** Separated neighbor index unpacking and optional distance copying into distinct, branchless loops (`populate_graph_results` and `drain_heap_into`) to eliminate per-element conditional branches on the hot path.
* **Contiguous Batch Search (`SearchResultBatch`):** Flat 1D vector layout for indices and distances (`n_queries * k`) with zero-overhead `get_indices(q)` and `get_distances(q)` row slicing. Supports zero-allocation execution with user-provided `std::span` buffers.
* **Small Buffer Optimization (SBO):** Uses stack buffers for transformed query bytes and candidate indices for vectors $\le 512$ bytes, avoiding heap allocations during search.

#### Quantization & Kernel Optimizations (`deglib::optimization::quantization`)
* **Chunked SIMD FP16 Quantization:** Processes FP16 (`uint16_t`) vectors in fixed 64-element chunks via `fp16_to_floats`, ensuring complete auto-vectorization (AVX2/F16C/AVX-512) without dimension limits.
* **In-Place EVP Query Quantization:** Direct query quantization into preallocated search buffers without intermediate vector allocations.
* **API Clean-Up:** Encapsulated internal quantizer row transformations (`transform`, `transform_row`, `transform_back`) as private. Removed obsolete methods (`dequantize`, `fit_quantize`, `inv_scale`).

---

## deglib v0.2.3

### Overview
deglib v0.2.3 introduces a zero-overhead C++20 templated **`Searcher`** pipeline for inference (`deglib::search::Searcher` / `deglib.search.create_searcher`), eliminating Python bridge and memory allocation overhead in single-query loops by bundling query quantization, graph search, and exact SIMD candidate reranking into a single native invocation. Additionally, it brings official VIBE distance-tolerance recall calculation and clean C++20 `std::span` convenience interfaces.

---

### 🚀 Key Features & Improvements

#### High-Performance Inference Pipeline (`deglib::search::Searcher`)
* **Zero-Overhead End-to-End Querying:** Integrates query quantization (FP32/FP16 $\rightarrow$ INT8/UINT8/EVP), graph traversal on quantized indices, and SIMD candidate refinement against FP16/FP32 base features in a single C++ call.
* **Static Template Specialization (`SearcherImpl<QuantT, RefinerT>`):**
  * Compile-time `if constexpr` branch elimination avoiding virtual method dispatch in the hot search loop.
  * Full support for all quantizers (`NoQuantizer`, `ScalarQuantizerInt8`, `ScalarQuantizerInt8PerDim`, `ScalarQuantizerUint8`, `ScalarQuantizerUint8PerDim`, `EVPQuantizer`).
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
