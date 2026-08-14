# deglib: C++ Architecture & Namespace Specification

This document specifies the target directory organization, namespace hierarchy, and header structure of the `deglib` C++ header-only library.

---

## Directory & Header Overview

All header files are located in `cpp/deglib/include/deglib/`:

```
deglib/
├── deglib.h                            # Main library umbrella include header
├── graph.h                             # Public DynamicExplorationGraph user facade
├── builder.h                           # EvenRegularGraphBuilder & build status types
├── optimization.h                      # Graph topology optimization algorithms
├── analysis.h                          # Graph diagnostics, quality metrics & reachability
├── distances.h                         # Feature spaces, metric enums & SIMD distance wrappers
├── filter.h                            # ANNS search label filtering
├── search.h                            # ANNS range search algorithms
├── concurrent.h                        # Multi-threading & concurrency utilities
├── config.h                            # Feature flags & compilation configuration
│
├── graph/                              # Internal graph hierarchy (0..N-1 indexed)
│   ├── internal_graph.h                # Abstract base interface for internal graphs
│   ├── mutable_graph.h                 # Abstract base interface for mutable graphs
│   ├── sizebounded_graph.h             # Concrete preallocated size-bounded graph
│   ├── readonly_graph.h                # Concrete compact read-only search graph
│   ├── visited_list_pool.h             # Thread-safe visited list pool for ANNS
│   └── dynamic_graph.h                 # Graph interface declarations
│
├── optimization/                       # Topology optimization kernels
│   ├── pruning.h                       # Edge pruning algorithms
│   ├── flas/                           # Fast Local Anisotropic Search
│   └── quantization/                   # EVP quantization kernels
│
├── distance/                           # SIMD distance computation kernels
│   ├── fp32_l2.h                       # Single-precision L2 distance
│   ├── fp32_ip.h                       # Single-precision Inner Product distance
│   ├── evp_inner_product.h             # EVP Quantized Inner Product distance
│   ├── fp16_ip.h                       # FP16 Half-precision Inner Product distance
│   ├── uint8_l2.h                      # Uint8 L2 distance
│   ├── fp32.h / fp16.h / uint8.h       # Vector type distance helpers
│   └── residual_mode.h                 # Quantization residual modes
│
└── utils/                              # SIMD, Memory & Random utilities
    ├── cpu.h                           # CPU feature detection & InstructionSet enum
    ├── memory.h                        # Aligned memory allocations & prefetching
    └── random.h                        # Randomization helpers
```

---

## Target Namespace Structure & API Reference

```
deglib::                                # Root namespace: DynamicExplorationGraph facade
├── graph::                             # Internal graph hierarchy (InternalGraph, SizeBoundedGraph, ReadOnlyGraph, ...)
├── builder::                           # Graph construction (EvenRegularGraphBuilder, OptimizationTarget, BuilderStatus)
├── search::                            # Search utilities and label filtering (Filter, range search)
├── distances::                         # Vector space & SIMD metrics (FloatSpace, Metric, SIMD kernels)
├── cpu::                               # CPU capabilities & InstructionSet enum (InstructionSet, has_avx2, has_avx512)
├── optimization::                      # Topology refinement (prune_worst_edges, prune_non_mrng_edges)
└── analysis::                          # Graph quality diagnostics & reachability (analyze_graph, GraphStats)
```

---

### 1. Root Namespace: `deglib::`

Contains the primary high-level user facade for ANNS search & exploration.

#### Class `deglib::DynamicExplorationGraph`
End-user facade translating external labels (User IDs) to internal indices (`0..N-1`).
- `explicit DynamicExplorationGraph(deglib::graph::InternalGraph& graph)`
- `uint32_t size() const`: Returns total vertex count.
- `uint8_t getEdgesPerVertex() const`: Returns configured number of edges per vertex.
- `const deglib::distances::FloatSpace& getFeatureSpace() const`: Returns the graph feature space.
- `bool hasVertex(uint32_t external_label) const`: Checks if external label exists.
- `deglib::graph::InternalGraph& internal()`: Direct access to the underlying internal graph.
- `const deglib::graph::InternalGraph& internal() const`
- `bool isMutable() const`: Returns `true` if graph supports vertex/edge modifications.
- `ResultSet search(span<const T> query, uint32_t k, float eps=0.0f, const deglib::search::Filter* filter=nullptr, uint32_t max_dc=0) const`: Approximate nearest neighbor search.
- `ResultSet explore(uint32_t entry_external_label, uint32_t k, uint32_t max_dc=0, float eps=0.0f, bool include_entry=true, const deglib::search::Filter* filter=nullptr) const`: Exploration from entry vertex label.
- `std::vector<uint32_t> getNeighbors(uint32_t external_label) const`: Returns external labels of neighbor vertices.
- `DynamicExplorationGraph to_readonly() const`: Converts graph to a compact read-only instance.
- `bool saveGraph(const std::string& path) const`: Saves the graph to disk if mutable.

---

### 2. Namespace: `deglib::distances::`

Feature space definitions, metric enums, and SIMD distance kernels.

#### Enums & Classes
- `enum class Metric`: Distance metrics (`FP32_L2`, `FP32_InnerProduct`, `Uint8_L2`, `FP16_InnerProduct`, `EVP_InnerProduct`).
- `class FloatSpace`: Feature space definition and SIMD distance calculator.

#### Distance Kernels
- `deglib::distances::fp32_l2`: Single-precision L2 distance routines.
- `deglib::distances::fp32_ip`: FP32 Inner Product distance routines.
- `deglib::distances::evp_ip`: EVP Quantized Inner Product distance routines.
- `deglib::distances::fp16_ip`: FP16 Half-Precision Inner Product distance routines.
- `deglib::distances::uint8_l2`: Uint8 L2 distance routines.

---

### 3. Namespace: `deglib::cpu::`

CPU feature detection and SIMD instruction set definitions.

#### Enums & Functions
- `enum class InstructionSet`: SIMD instruction set enum (`Auto`, `Scalar`, `AVX2`, `AVX512`).
- `bool has_avx2()`: Checks runtime AVX2 availability.
- `bool has_avx512()`: Checks runtime AVX512 availability.

---

### 4. Namespace: `deglib::search::`

Search algorithms and search filtering objects.

#### Classes
- `class Filter`: Boolean label filter for restricting search result sets during search and explore operations.

---

### 5. Namespace: `deglib::graph::`

Low-level internal graph hierarchy operating directly on internal indices (`0..N-1`) for memory bandwidth performance.

#### Classes
- `class InternalGraph`: Abstract base class for internal 0..N-1 indexed graphs.
- `class MutableGraph : public InternalGraph`: Abstract base class for mutable internal graphs.
- `class SizeBoundedGraph : public MutableGraph`: Fixed-capacity, preallocated graph supporting dynamic vertex/edge mutations.
- `class ReadOnlyGraph : public InternalGraph`: Compact read-only graph optimized for search performance.
- `class VisitedListPool`: Thread-safe pool managing visited bitsets during search.

---

### 6. Namespace: `deglib::builder::`

Graph construction and optimization algorithms.

#### Classes & Types
- `enum class OptimizationTarget`: Build optimization strategy (`StreamingData`, `HighLID`, `LowLID`).
- `struct BuilderStatus`: Status object passed to build callbacks (`step`, `added`, `deleted`, `improved`, `tries`).
- `class EvenRegularGraphBuilder`:
  - `EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd, OptimizationTarget optimization_target = OptimizationTarget::LowLID, uint8_t extend_k = 0, float extend_eps = 0.1f, uint8_t improve_k = 0, float improve_eps = 0.001f, uint8_t max_path_length = 5, uint32_t swap_tries = 0, uint32_t additional_swap_tries = 0)`
  - `void addEntry(uint32_t label, std::vector<std::byte> feature)`: Queues entry addition.
  - `void removeEntry(uint32_t label)`: Queues entry removal.
  - `uint32_t getNumNewEntries() const`: Queued addition count.
  - `uint32_t getNumRemoveEntries() const`: Queued removal count.
  - `void setThreadCount(size_t thread_count)`: Sets parallel build thread count.
  - `void setBatchSize(uint32_t tasks_per_batch, uint32_t task_size)`: Sets multithreading batch parameters.
  - `void build(std::function<void(BuilderStatus&)> callback, bool infinite = false)`: Runs graph construction.
  - `void stop()`: Requests build termination.

---

### 7. Namespace: `deglib::optimization::`

Graph topology refinement algorithms.

#### Functions
- `void prune_worst_edges(deglib::graph::MutableGraph& graph, const uint8_t prune_worst, const size_t numThreads = 0)`: Prunes worst edges per vertex.
- `void prune_non_mrng_edges(deglib::graph::MutableGraph& graph, const size_t numThreads = 0)`: Removes non-MRNG edges.

---

### 8. Namespace: `deglib::analysis::`

Graph diagnostic metrics, connectivity validation, and reachability.

#### Struct `GraphStats`
Attributes: `vertex_count`, `edge_count`, `feature_dims`, `edges_per_vertex`, `avg_out_degree`, `min_out_degree`, `max_out_degree`, `avg_in_degree`, `min_in_degree`, `max_in_degree`, `source_vertices`, `search_reachability`, `exploration_reachability`, `memory_bytes`.

#### Functions
- `GraphStats analyze_graph(const deglib::graph::InternalGraph& graph)`
- `float calc_avg_edge_weight(const deglib::graph::MutableGraph& graph, const int scale = 1)`
- `std::vector<float> calc_edge_weight_histogram(const deglib::graph::MutableGraph& graph, const bool sorted, const int scale = 1)`
- `bool check_graph_weights(const deglib::graph::MutableGraph& graph)`
- `bool check_graph_regularity(const deglib::graph::InternalGraph& graph, const uint32_t expected_vertices, const bool check_back_link = false)`
- `bool check_graph_connectivity(const deglib::graph::InternalGraph& graph)`
- `uint32_t calc_non_rng_edges(const deglib::graph::MutableGraph& graph)`
- `float calc_search_reachability(const deglib::graph::InternalGraph& graph)`
- `float calc_exploration_reach(const deglib::graph::InternalGraph& graph)`
