# deglib: Python API Reference

`deglib` provides high-performance Python bindings for the Dynamic Exploration Graph (DEG) library for Approximate Nearest Neighbor Search (ANNS) and graph exploration.

```python
import deglib
```

> **Note:** Signatures and types in this reference are presented in a clean, simplified format for maximum readability.

---

## Modules & Package Structure

| Module | Primary Elements | Purpose |
|---|---|---|
| [`deglib`](#1-root-package-deglib) | `DynamicExplorationGraph`, `build_from_data`, `load_*` | Main user facade for querying, exploration, and graph lifecycle. |
| [`deglib.builder`](#2-module-deglibbuilder) | `GraphBuilder`, `OptimizationTarget`, `BuilderStatus`, `build_from_data` | Incremental vector addition/deletion, parallel batch construction, and optimization. |
| [`deglib.distances`](#3-module-deglibdistances) | `FloatSpace`, `Metric`, `quantize_batch`, `floats_to_fp16`, `fp16_to_floats` | Vector metrics, SIMD feature spaces, batch distance evaluation, and quantization. |
| [`deglib.search`](#4-module-deglibsearch) | `Filter` | Bitset-based label filtering for ANNS search and exploration. |
| [`deglib.optimization`](#5-module-degliboptimization) | `prune_*`, `presort`, `mips_l2_*` | MRNG graph pruning, FLAS 1D dataset presorting, and MIPS L2 transformations. |
| [`deglib.analysis`](#6-module-deglibanalysis) | `analyze_graph`, `check_*`, `calc_*` | Graph validation, connectivity verification, and reachability metrics. |
| [`deglib.cpu`](#7-module-deglibcpu) | `InstructionSet`, `has_avx2`, `has_avx512` | Runtime CPU SIMD capability detection and instruction set enum. |

---

## 1. Root Package: `deglib`

### Top-Level Functions

```python
# Build and optimize a DynamicExplorationGraph directly from a numpy array in one call
deglib.build_from_data(
    data,                                      # 2D numpy array [N, D] of feature vectors
    labels=None,                               # Optional 1D uint32 array [N] of external labels (default: 0..N-1)
    edges_per_vertex=32,                       # Number of outgoing edges per vertex (must be even)
    capacity=-1,                               # Max graph capacity (-1 = data.shape[0])
    metric=Metric.FP32_L2,                     # Distance metric (Metric enum or string e.g. "FP32_L2")
    instruction="Auto",                        # SIMD instruction set ("Auto", "AVX2", "AVX512", "Scalar")
    optimization_target=OptimizationTarget.LowLID, # OptimizationTarget (LowLID, HighLID, StreamingData)
    extend_k=0,                                # Number of neighbors during graph extension (0 = edges_per_vertex)
    extend_eps=0.2,                            # Search expansion epsilon during graph extension
    improve_k=0,                               # Number of neighbors during edge improvement
    improve_eps=0.001,                         # Search expansion epsilon during edge improvement
    max_path_length=5,                         # Maximum edge swaps before canceling an improvement try
    swap_tries=0,                              # Number of improvement attempts per build step
    additional_swap_tries=0,                   # Additional improvement attempts after successful improvement
    thread_count=0,                            # Worker thread count (0 = all available CPU cores)
    seed=None,                                 # Random seed (None = default seed)
    callback=None                              # Callback function(BuilderStatus) or "progress" for CLI bar
) -> DynamicExplorationGraph

# Load saved graphs from disk
deglib.load_readonly_graph(path) -> DynamicExplorationGraph
deglib.load_dynamic_graph(path, chunk_size=1024) -> DynamicExplorationGraph
```

### Class: `deglib.DynamicExplorationGraph`

The main graph facade used to search, explore, and convert exploration graphs. All operations work with user-defined **external object labels** (`uint32`).

```python
class DynamicExplorationGraph:
    # --- Factories & Loading ---
    # Create empty mutable graph with fixed capacity in preallocated memory (SizeBoundedGraph)
    create_empty(capacity, feature_space, edges_per_vertex=32) -> DynamicExplorationGraph

    # Create empty mutable graph with chunk-based memory that grows dynamically (DynamicGraph)
    create_dynamic_empty(feature_space, edges_per_vertex=32, chunk_size=1024) -> DynamicExplorationGraph

    # Create a randomized initial exploration graph from a numpy feature array
    create_random_graph(features, feature_space, edges_per_vertex=32, seed=7) -> DynamicExplorationGraph

    # Load a saved read-only graph file optimized for fast search
    load_readonly_graph(path) -> DynamicExplorationGraph

    # Load a saved graph file as a mutable dynamic chunk-allocated graph
    load_dynamic_graph(path, chunk_size=1024) -> DynamicExplorationGraph

    # --- Search & Querying ---
    # Search nearest neighbors for 1D (single) or 2D (batch) query vectors.
    # query: 1D [D] or 2D [Q, D] numpy array
    # eps: Search expansion factor (higher = more accurate, slower)
    # k: Number of nearest neighbors to return per query
    # filter_labels: Filter object or numpy array of valid external labels
    # max_distance_computation_count: Budget cap for distance calculations (0 = unlimited)
    # threads: Worker thread count for batch queries (0 = all CPU cores)
    # return_distances: If True, returns (indices, distances) tuple; if False, returns only indices
    # unsorted: If True, skips sorting results in ascending distance order
    search(query, eps=0.0, k=10, filter_labels=None, max_distance_computation_count=0, threads=0, return_distances=True, unsorted=False)

    # Explore graph starting from entry vertex label(s) (supports single int or batch array).
    # Returns (indices, distances) if return_distances=True, otherwise indices array.
    explore(entry_label, k, max_distance_computation_count=0, eps=0.0, include_entry=True, threads=1, filter_labels=None, return_distances=True, unsorted=False)

    # Returns list of neighbor external labels connected to a given vertex
    get_neighbors(external_label) -> list[int]

    # --- State & Inspection ---
    size() -> int                              # Number of active vertices
    get_edges_per_vertex() -> int              # Outgoing edges per vertex
    get_feature_space() -> FloatSpace          # Metric and dimensionality
    has_vertex(external_label) -> bool         # True if label exists in graph
    is_mutable() -> bool                       # True if graph can be modified via GraphBuilder

    # --- Conversions & Saving ---
    # Convert to compact, immutable graph (ReadOnlyGraph) for maximum query throughput
    to_readonly(feature_space=None, custom_features=None) -> DynamicExplorationGraph

    # Convert to mutable fixed-capacity graph (SizeBoundedGraph) with optional new metric/features
    to_mutable(feature_space=None, custom_features=None, capacity=0) -> DynamicExplorationGraph

    # Convert to mutable chunk-allocated graph (DynamicGraph) that expands dynamically
    to_dynamic(feature_space=None, custom_features=None, chunk_size=1024) -> DynamicExplorationGraph

    # Save graph topology and features to disk (graph must be mutable)
    save_graph(path)
```

---

## 2. Module: `deglib.builder`

Provides graph construction, incremental updates, and multithreaded edge optimization.

```python
from deglib.builder import GraphBuilder, OptimizationTarget, BuilderStatus, build_from_data

# Optimization strategy based on dataset characteristics
class OptimizationTarget:
    StreamingData  # Optimized for streaming or shifting distributions (single-threaded insertion)
    HighLID        # Optimized for high intrinsic dimensionality (> 15, parallel extension)
    LowLID         # Optimized for low intrinsic dimensionality (<= 15, parallel extension)


# Status object passed to build progress callbacks
class BuilderStatus:
    step: int                 # Number of graph manipulation steps completed
    added: int                # Total number of added vertices
    deleted: int              # Total number of deleted vertices
    improved: int             # Total number of successful edge improvements
    tries: int                # Total number of improvement attempts
    step_added_ids: list[int] # External labels added in the current step
    step_deleted_ids: list[int]# External labels deleted in the current step
    total_added_ids: list[int]# All added external labels across the entire build
    total_deleted_ids: list[int]# All deleted external labels across the entire build


class GraphBuilder:
    def __init__(
        graph,                                 # Mutable DynamicExplorationGraph
        seed=None,                             # Random seed
        optimization_target=OptimizationTarget.LowLID, # Optimization target strategy
        extend_k=0,                            # Neighbors during extension (0 = graph's edges_per_vertex)
        extend_eps=0.1,                        # Epsilon for neighbor search during extension
        improve_k=0,                           # Neighbors during edge improvement
        improve_eps=0.001,                     # Epsilon for neighbor search during improvement
        max_path_length=5,                     # Max edge swaps per improvement attempt
        swap_tries=0,                          # Improvement attempts per build step
        additional_swap_tries=0                # Extra attempts after a successful swap
    )

    # Queue a single vector [D] or batch [N, D] for addition
    add_entry(external_label, feature)

    # Queue an external label for deletion
    remove_entry(external_label)

    # Queue status inspection
    get_num_new_entries() -> int               # Count of pending additions
    get_num_remove_entries() -> int            # Count of pending deletions

    # Multithreading configuration
    set_thread_count(thread_count)             # Set worker thread count (0 = all CPU cores)
    set_batch_size(tasks_per_batch, task_size) # Configure batch granularity
    get_batch_size() -> int                    # Current total batch size (threads * tasks * task_size)

    # Execute build/optimization loop.
    # callback: function(BuilderStatus) or "progress" for console progress bar
    # infinite: If True, runs indefinitely until stop() is called from another thread
    build(callback=None, infinite=False) -> BuilderStatus

    # Request graceful termination of the build loop between steps
    stop()
```

---

## 3. Module: `deglib.distances`

Vector metrics, SIMD feature spaces, batch distance evaluation, and quantization.

```python
from deglib.distances import FloatSpace, Metric, quantize_batch, floats_to_fp16, fp16_to_floats

# Supported distance metrics
class Metric:
    FP32_L2            # Single-precision Euclidean L2 distance (dtype: np.float32)
    FP32_InnerProduct  # Single-precision Inner Product (1 - <a,b>, dtype: np.float32)
    Uint8_L2           # 8-bit unsigned integer L2 distance (dtype: np.uint8)
    FP16_InnerProduct  # Half-precision FP16 Inner Product (dtype: np.uint16)
    EVP_InnerProduct   # Extreme Vector Quantization binary/ternary IP (dtype: np.uint8)

    get_dtype() -> np.dtype # Expected numpy data type for feature vectors


class FloatSpace:
    # Create feature space with dimension, metric, and optional SIMD instruction set
    create(dim, metric, instruction=InstructionSet.Auto) -> FloatSpace

    dim() -> int                               # Dimensionality
    metric() -> Metric                         # Configured metric enum
    get_data_size() -> int                     # Byte size of a single vector
    get_instruction() -> InstructionSet        # Active SIMD instruction set ("AVX512", "AVX2", "Scalar")

    # Compute distance between two individual 1D feature vectors
    compute_distance(vec1, vec2) -> float

    # Compute 1D array of distances between query [D] and a batch of targets [N, D]
    compute_distances(query, targets) -> np.ndarray

    # Multi-threaded exact distance candidate reranking in C++.
    # queries: 2D array [Q, D]
    # candidate_indices: 2D uint32 array [Q, K_cand]
    # base_vectors: 2D array [N, D] (defaults to queries if None)
    # k_top: Number of nearest candidates to return per query (0 = all)
    # return_distances: If True, returns (indices, distances) tuple
    rerank(queries, candidate_indices, base_vectors=None, k_top=0, num_threads=0, return_distances=False, unsorted=False)


# --- Quantization & Type Conversion ---

# Quantize float32 or float16 vectors to byte-packed EVP format using C++ multithreading
quantize_batch(vectors, non_zeros, num_threads=0) -> np.ndarray

# Convert float32 numpy array to uint16-packed FP16 representation
floats_to_fp16(floats) -> np.ndarray

# Convert uint16-packed FP16 array back to float32
fp16_to_floats(fp16_vals) -> np.ndarray
```

---

## 4. Module: `deglib.search`

Bitset-based label filtering for search and exploration.

```python
from deglib.search import Filter

class Filter:
    def __init__(valid_labels, max_value=-1, max_label_count=-1)
    """
    Creates a search label filter restricting results to allowed IDs.

    valid_labels: 1D int32 numpy array containing all allowed external labels.
    max_value: Maximum label value (computed automatically if -1).
    max_label_count: Total dataset size (defaults to graph size).
    """

    # Static factory supporting None, numpy array, or existing Filter
    create_filter(filter_labels, graph_size) -> cpp_filter
```

---

## 5. Module: `deglib.optimization`

Graph topology refinement, FLAS 1D dataset presorting, and MIPS L2 transformations.

```python
from deglib.optimization import (
    prune_non_mrng_edges,
    prune_worst_edges,
    presort,
    mips_l2_transform,
    mips_l2_transform_query,
)

# Remove all edges violating the Monotonic Relative Neighbor Graph (MRNG) rule. Returns removed edge count.
prune_non_mrng_edges(graph, num_threads=0) -> int

# Prune the worst (longest/highest-weight) neighbors per vertex by replacing them with self-loops
prune_worst_edges(graph, prune_worst, num_threads=0)

# 1D dataset pre-sorting using Fast Linear Assignment Sorter (FLAS).
# vectors: 2D float32 array [N, D]
# space_or_metric: FloatSpace or Metric enum (default: Metric.FP32_L2)
# radius_decay: Neighborhood decay factor (default: 0.9)
# threads: Worker threads (0 = all CPU cores)
# callback: function(progress_float) or "progress" for console output
# Returns 1D uint32 permutation array [N] of sorted vector indices.
presort(vectors, space_or_metric=None, radius_decay=0.9, threads=0, callback=None) -> np.ndarray

# Transform database vectors from d dimensions to (d+1) dimensions for MIPS via L2 distance.
# database: 2D float32 array [N, d]
# Returns tuple (transformed_database [N, d+1], max_norm float M).
mips_l2_transform(database) -> tuple[np.ndarray, float]

# Pad query vectors with 0 at dimension (d+1) for MIPS queries against an L2-transformed database.
# queries: 1D [d] or 2D [Q, d] float32 array
# Returns transformed array of shape [d+1] or [Q, d+1].
mips_l2_transform_query(queries) -> np.ndarray
```

---

## 6. Module: `deglib.analysis`

Graph health diagnostics, regularity validation, and reachability metrics.

```python
from deglib.analysis import (
    analyze_graph,
    check_graph_regularity,
    check_graph_weights,
    check_graph_connectivity,
    calc_avg_edge_weight,
    calc_edge_weight_histogram,
    calc_non_rng_edges,
    calc_search_reachability,
    calc_exploration_reach,
)

# Comprehensive graph analysis returning a dictionary of topology and health metrics
analyze_graph(graph) -> dict
# Returns dictionary:
# {
#     "vertex_count": int,              # Total active vertices
#     "edge_count": int,                # Total valid edges
#     "feature_dims": int,              # Vector dimensionality
#     "edges_per_vertex": int,          # Outgoing edge capacity per vertex
#     "avg_out_degree": float,          # Average out-degree
#     "min_out_degree": int,            # Minimum out-degree
#     "max_out_degree": int,            # Maximum out-degree
#     "avg_in_degree": float,           # Average in-degree
#     "min_in_degree": int,             # Minimum in-degree
#     "max_in_degree": int,             # Maximum in-degree
#     "source_vertices": int,           # Vertices with in-degree = 0 (unreachable as targets)
#     "search_reachability": float,     # Ratio of vertices reachable from entry points (0.0 to 1.0)
#     "exploration_reachability": float,# Average exploration reachability across all vertices (0.0 to 1.0)
#     "memory_bytes": int               # Estimated memory footprint in bytes
# }

# Verify that vertex count, ascending edge ordering, and neighbor uniqueness are valid
check_graph_regularity(graph, expected_vertices, check_back_link=False) -> bool

# Verify that stored edge weights match exact feature distance calculations
check_graph_weights(graph) -> bool

# Verify that the graph consists of a single connected component
check_graph_connectivity(graph) -> bool

# Calculate average edge weight across all graph edges
calc_avg_edge_weight(graph, scale=1) -> float

# Calculate 10-bin histogram of graph edge weights
calc_edge_weight_histogram(graph, sort=True, scale=1) -> list[float]

# Count total non-RNG conform edges
calc_non_rng_edges(graph) -> int

# Fraction of vertices reachable from graph entry points via BFS (0.0 to 1.0)
calc_search_reachability(graph) -> float

# Average exploration reachability across all starting vertices (0.0 to 1.0)
calc_exploration_reach(graph) -> float
```

---

## 7. Module: `deglib.cpu`

SIMD capability detection and instruction set enumeration.

```python
from deglib.cpu import InstructionSet, has_avx2, has_avx512

class InstructionSet:
    Auto    # Automatically select highest supported instruction set
    Scalar  # Standard scalar operations
    AVX2    # 256-bit AVX2 + FMA SIMD
    AVX512  # 512-bit AVX-512 SIMD

# Runtime hardware support checks
has_avx2() -> bool    # Returns True if host CPU supports AVX2
has_avx512() -> bool  # Returns True if host CPU supports AVX-512
```
