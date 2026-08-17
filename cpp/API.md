# deglib: C++ API Reference

`deglib` is a high-performance, header-only C++ library for Dynamic Exploration Graphs and Approximate Nearest Neighbor Search (ANNS).

Include the umbrella header to access all functionalities:

```cpp
#include <deglib/deglib.h>
```

> **Note:** Signatures and types in this reference are shown in a simplified format (omitting template boilerplate, internal qualifiers, and operator overloads) to provide a clear and readable overview of available operations.

---

## Namespaces & Header Structure

| Namespace | Header | Purpose |
|---|---|---|
| [`deglib::`](#1-namespace-deglib-user-facade) | `<deglib/graph.h>` | High-level user facade (`DynamicExplorationGraph`) translating external object labels to internal indices. |
| [`deglib::builder::`](#2-namespace-deglibbuilder) | `<deglib/builder.h>` | Graph construction, streaming vector insertion/deletion, and continuous optimization. |
| [`deglib::distances::`](#3-namespace-deglibdistances) | `<deglib/distances.h>` | Metric enums (`Metric`), feature spaces (`FloatSpace`), and distance functions. |
| [`deglib::search::`](#4-namespace-deglibsearch) | `<deglib/search.h>`, `<deglib/filter.h>` | Search label filtering (`Filter`) and exact distance candidate reranking (`rerank`). |
| [`deglib::optimization::`](#5-namespace-degliboptimization) | `<deglib/optimization.h>` | MRNG pruning, FLAS 1D presorting, EVP quantization, and MIPS L2 transformations. |
| [`deglib::analysis::`](#6-namespace-deglibanalysis) | `<deglib/analysis.h>` | Graph health diagnostics, regularity validation, and reachability metrics. |
| [`deglib::graph::`](#7-namespace-deglibgraph-internal-graphs--results) | `<deglib/graph/*.h>` | Low-level internal graph hierarchy (`0..N-1` indices) and search `ResultSet` (max-heap). |
| [`deglib::cpu::`](#8-namespace-deglibcpu) | `<deglib/utils/cpu.h>` | Hardware feature detection (AVX2, AVX512) and `InstructionSet` configuration. |
| [`deglib::memory::`](#9-namespace-deglibmemory) | `<deglib/utils/memory.h>` | CPU cache line prefetching utilities. |
| [`deglib::concurrent::`](#10-namespace-deglibconcurrent) | `<deglib/concurrent.h>` | Parallel execution loops (`parallel_for`, `parallel_batch_for`). |

---

## 1. Namespace: `deglib::` (User Facade)

Header: `<deglib/graph.h>`

The root namespace provides [`DynamicExplorationGraph`](#class-dynamicexplorationgraph), the primary user-facing class that wraps internal graph structures and maps external object labels (`uint32_t`) to internal graph indices.

```cpp
namespace deglib {

class DynamicExplorationGraph {
public:
    // --- Factory Methods ---

    /// Create an empty mutable graph with fixed maximum capacity (SizeBoundedGraph)
    static DynamicExplorationGraph create_empty(
        uint32_t max_vertex_count,
        uint8_t edges_per_vertex,
        FloatSpace feature_space
    );

    /// Create an empty mutable graph with dynamic chunk-allocated growth (DynamicGraph)
    static DynamicExplorationGraph create_dynamic_empty(
        uint8_t edges_per_vertex,
        FloatSpace feature_space,
        uint32_t chunk_size = 1024
    );

    /// Create a randomized exploration graph from raw feature data
    static DynamicExplorationGraph create_random_graph(
        byte* feature_data,
        uint32_t vertex_count,
        uint8_t edges_per_vertex,
        FloatSpace feature_space,
        uint32_t seed = 7
    );

    // --- Search & Exploration ---

    /// Search for the k-nearest neighbors of a query vector
    ResultSet search(
        span<float> query,
        uint32_t k,
        float eps = 0.0f,
        Filter* filter = nullptr,
        uint32_t max_distance_computations = 0
    );

    /// Explore the graph starting from an existing vertex label
    ResultSet explore(
        uint32_t entry_label,
        uint32_t k,
        uint32_t max_distance_computations = 0,
        float eps = 0.0f,
        bool include_entry = true,
        Filter* filter = nullptr
    );

    /// Returns the neighbor labels connected to a given vertex
    vector<uint32_t> getNeighbors(uint32_t label);

    // --- State & Inspection ---

    uint32_t size();                                       ///< Total number of active vertices
    uint8_t getEdgesPerVertex();                           ///< Outgoing edges per vertex
    FloatSpace getFeatureSpace();                          ///< Graph metric and dimensionality
    bool hasVertex(uint32_t label);                        ///< Check if label exists
    bool isMutable();                                      ///< Check if graph allows modifications
    InternalGraph& internal();                             ///< Access underlying internal graph

    // --- Conversions & Persistence ---

    /// Convert to a compact, read-only graph (ReadOnlyGraph) optimized for search
    DynamicExplorationGraph to_readonly();

    /// Convert to a mutable SizeBoundedGraph (fixed capacity)
    DynamicExplorationGraph to_mutable(uint32_t new_max_size = 0);
    DynamicExplorationGraph to_mutable(FloatSpace new_space, void* new_features = nullptr, uint32_t new_max_size = 0);

    /// Convert to a mutable DynamicGraph (chunk-allocated dynamic growth)
    DynamicExplorationGraph to_dynamic(uint32_t chunk_size = 1024);
    DynamicExplorationGraph to_dynamic(FloatSpace new_space, void* new_features = nullptr, uint32_t chunk_size = 1024);

    /// Save graph topology and features to disk (graph must be mutable)
    bool saveGraph(string path);
};

/// Build and optimize a DynamicExplorationGraph directly from contiguous feature data
DynamicExplorationGraph build_from_data(
    span<float> data,
    uint32_t dims,
    span<uint32_t> labels = {},
    uint8_t edges_per_vertex = 32,
    Metric metric = Metric::FP32_L2,
    OptimizationTarget target = OptimizationTarget::LowLID,
    uint8_t extend_k = 64,
    float extend_eps = 0.1f,
    uint8_t improve_k = 0,
    float improve_eps = 0.001f,
    uint8_t max_path_length = 5,
    uint32_t swap_tries = 0,
    uint32_t additional_swap_tries = 0,
    size_t thread_count = 0,
    uint32_t seed = 42,
    function<void(BuilderStatus&)> callback = nullptr
);

} // namespace deglib
```

---

## 2. Namespace: `deglib::builder::`

Header: `<deglib/builder.h>`

Graph construction, streaming vector additions/removals, and continuous edge optimization.

```cpp
namespace deglib::builder {

/// Build strategy based on dataset characteristics
enum OptimizationTarget {
    StreamingData, ///< Dynamic streaming data or shifting distributions
    HighLID,       ///< Datasets with high Local Intrinsic Dimensionality (> 15)
    LowLID         ///< Datasets with low Local Intrinsic Dimensionality (<= 15)
};

/// Build progress and modification counters passed to callback functions
struct BuilderStatus {
    uint64_t step;                        ///< Number of manipulation steps completed
    uint64_t added;                       ///< Total added vertices
    uint64_t deleted;                     ///< Total deleted vertices
    uint64_t improved;                    ///< Total successful edge improvements
    uint64_t tries;                       ///< Total improvement attempts

    vector<uint32_t> step_added_ids;     ///< Labels added in current step
    vector<uint32_t> step_deleted_ids;   ///< Labels deleted in current step
    vector<uint32_t> total_added_ids;    ///< All added labels across the build
    vector<uint32_t> total_deleted_ids;  ///< All deleted labels across the build
};

class EvenRegularGraphBuilder {
public:
    /// Construct builder for a DynamicExplorationGraph facade or MutableGraph
    EvenRegularGraphBuilder(
        DynamicExplorationGraph& graph,
        mt19937& rnd,
        OptimizationTarget target = OptimizationTarget::StreamingData,
        uint8_t extend_k = 0,
        float extend_eps = 0.1f,
        uint8_t improve_k = 0,
        float improve_eps = 0.001f,
        uint8_t max_path_length = 5,
        uint32_t swap_tries = 0,
        uint32_t additional_swap_tries = 0
    );

    // --- Entry Queue Management ---

    /// Queue a new vector insertion
    void addEntry(uint32_t label, span<float> feature);
    void addEntry(uint32_t label, vector<byte> feature);

    /// Queue a vertex label for deletion
    void removeEntry(uint32_t label);

    uint32_t getNumNewEntries();          ///< Count of pending additions
    uint32_t getNumRemoveEntries();       ///< Count of pending deletions

    // --- Execution ---

    void setThreadCount(size_t thread_count);                      ///< Set worker thread count
    void setBatchSize(uint32_t tasks_per_batch, uint32_t task_size); ///< Configure batch granularity

    /// Execute graph construction and optimization loop
    BuilderStatus build(function<void(BuilderStatus&)> callback = nullptr, bool infinite = false);
    BuilderStatus build();

    /// Request termination of the build loop
    void stop();
};

} // namespace deglib::builder
```

---

## 3. Namespace: `deglib::distances::`

Header: `<deglib/distances.h>`

Distance metrics, feature space configurations, and distance evaluation routines.

```cpp
namespace deglib::distances {

/// Supported distance metrics
enum class Metric {
    FP32_L2,           ///< Single-precision Euclidean L2 distance
    FP32_InnerProduct, ///< Single-precision Inner Product distance (1 - <a, b>)
    Uint8_L2,          ///< Unsigned 8-bit integer L2 distance
    FP16_InnerProduct, ///< Half-precision FP16 Inner Product distance
    EVP_InnerProduct   ///< Extreme Vector Quantization (1-bit / ternary) Inner Product
};

/// Represents a vector feature space (dimensionality, metric, byte layout, and SIMD kernel)
class FloatSpace {
public:
    FloatSpace(size_t dim, Metric metric, InstructionSet instruction = InstructionSet::Auto);

    size_t dim();                  ///< Vector dimensionality
    Metric metric();               ///< Configured distance metric
    size_t get_data_size();        ///< Size in bytes of a single vector
    const char* get_instruction(); ///< Active SIMD kernel ("AVX512", "AVX2", "Scalar")
};

// --- Distance Computation Functions ---

/// Compute distance between two feature vectors
float compute_distance(FloatSpace space, void* vec1, void* vec2);

/// Compute distances between queries and targets into a pre-allocated output array
void compute_distances(
    FloatSpace space,
    void* queries,
    size_t num_queries,
    void* targets,
    size_t num_targets,
    float* result_distances
);

/// Compute distances between queries and targets, returning a vector of floats
vector<float> compute_distances(
    FloatSpace space,
    void* queries,
    size_t num_queries,
    void* targets,
    size_t num_targets
);

} // namespace deglib::distances
```

---

## 4. Namespace: `deglib::search::`

Headers: `<deglib/search.h>`, `<deglib/filter.h>`

Search result filtering and post-search candidate reranking.

```cpp
namespace deglib::search {

/// Bitset filter to restrict ANNS search/explore results to allowed external labels
class Filter {
public:
    Filter(int* valid_labels, size_t size, size_t max_value, size_t max_label_count);

    bool is_valid(int label);               ///< Check if label passes filter
    size_t size();                          ///< Number of valid labels
    double get_inclusion_rate();            ///< Ratio of valid labels to total labels
    void for_each_valid_label(auto func);   ///< Iterate over all valid labels
};

/// Multi-threaded exact distance reranking of candidate neighbors for each query
vector<ResultSet> rerank(
    FloatSpace space,
    void* queries,
    size_t num_queries,
    void* base_vectors,
    size_t num_base_vectors,
    uint32_t* base_candidates,
    size_t candidates_per_query,
    size_t k_top = 0,
    size_t num_threads = 0
);

} // namespace deglib::search
```

---

## 5. Namespace: `deglib::optimization::`

Header: `<deglib/optimization.h>`

Graph topology pruning, 1D dataset presorting (FLAS), EVP quantization, and MIPS transformations.

```cpp
namespace deglib::optimization {

// --- Graph Pruning & Edge Optimization ---

/// Prune the worst (longest/highest-weight) neighbors per vertex, replacing them with self-loops
void prune_worst_edges(MutableGraph& graph, uint8_t prune_worst, size_t num_threads = 0);

/// Parallel removal of edges violating the Monotonic Relative Neighbor Graph (MRNG) rule
uint32_t prune_non_mrng_edges(MutableGraph& graph, size_t num_threads = 0);

/// Remove non-MRNG edges using a globally weight-sorted strategy
uint32_t prune_non_mrng_edges_weight_sorted(MutableGraph& graph, size_t num_threads = 0);

/// Iteratively remove non-MRNG edges per-vertex until convergence
uint32_t prune_non_mrng_edges_iterative(MutableGraph& graph, size_t num_threads = 0);

/// Optimize graph edges using continuous EvenRegularGraphBuilder improvement steps
void optimize_edges(MutableGraph& graph, uint8_t k_opt, float eps_opt, uint8_t i_opt, uint32_t iterations);

// --- Dataset Pre-sorting (FLAS) ---

/// 1D dataset pre-sorting for improved cache locality and graph quality.
/// Returns permutation indices in sorted order.
vector<uint32_t> presort(
    float* data,
    size_t count,
    FloatSpace space,
    float radius_decay = 0.9f,
    size_t num_threads = 0,
    function<bool(float)> callback = nullptr
);

// --- Extreme Vector Quantization (EVP) ---

/// Quantize a single vector to packed EVP representation
vector<byte> quantize_evp_single(float* embedding, uint32_t dim, uint32_t non_zeros);
vector<byte> quantize_evp_single(uint16_t* embedding, uint32_t dim, uint32_t non_zeros);

/// Quantize a batch of vectors to packed EVP representation
vector<byte> quantize_evp_batch(float* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t num_threads = 0);
vector<byte> quantize_evp_batch(uint16_t* data, size_t count, uint32_t dim, uint32_t non_zeros, size_t num_threads = 0);

// --- MIPS to L2 Space Transformation ---

/// Transform database vectors from d dimensions to (d+1) dimensions for MIPS via L2 distance. Returns max norm.
float mips_l2_transform(float* input, size_t count, size_t dim, float* output);
pair<vector<float>, float> mips_l2_transform(vector<float> input, size_t count, size_t dim);

/// Pad query vectors with 0 at dimension (d+1) for MIPS queries against an L2-transformed database
void mips_l2_transform_query(float* input, size_t count, size_t dim, float* output);
vector<float> mips_l2_transform_query(vector<float> input, size_t count, size_t dim);

} // namespace deglib::optimization
```

---

## 6. Namespace: `deglib::analysis::`

Header: `<deglib/analysis.h>`

Graph quality analysis, connectivity verification, degree distribution, and reachability diagnostics.

```cpp
namespace deglib::analysis {

/// Comprehensive graph statistics
struct GraphStats {
    uint32_t vertex_count;              ///< Active vertices count
    uint32_t edge_count;                ///< Total valid edges
    uint32_t feature_dims;              ///< Vector dimensionality
    uint8_t edges_per_vertex;           ///< Outgoing edges per vertex
    float avg_out_degree;               ///< Average out-degree
    uint32_t min_out_degree;            ///< Minimum out-degree
    uint32_t max_out_degree;            ///< Maximum out-degree
    float avg_in_degree;                ///< Average in-degree
    uint32_t min_in_degree;             ///< Minimum in-degree
    uint32_t max_in_degree;             ///< Maximum in-degree
    uint32_t source_vertices;           ///< Vertices with 0 in-degree (unreachable as targets)
    float search_reachability;          ///< Ratio of vertices reachable from entry points
    float exploration_reachability;     ///< Average exploration reachability across all vertices
    size_t memory_bytes;                ///< Estimated memory footprint in bytes
};

/// Perform comprehensive graph analysis
GraphStats analyze_graph(InternalGraph& graph);

/// Verify graph regularity (correct vertex count, ascending sorted unique neighbors, no self-loops)
bool check_graph_regularity(InternalGraph& graph, uint32_t expected_vertices, bool check_back_link = false);

/// Verify that cached edge weights match exact feature distance calculations
bool check_graph_weights(MutableGraph& graph);

/// Verify that the graph consists of a single connected component
bool check_graph_connectivity(InternalGraph& graph);

/// Calculate average edge weight across the graph
float calc_avg_edge_weight(MutableGraph& graph, int scale = 1);

/// Calculate 10-bin histogram of graph edge weights
vector<float> calc_edge_weight_histogram(MutableGraph& graph, bool sorted, int scale = 1);

/// Count number of non-RNG conform edges
uint32_t calc_non_rng_edges(MutableGraph& graph);

/// Calculate number of vertices reachable from graph entry points via BFS
uint32_t calc_search_reachability(InternalGraph& graph);

/// Calculate average exploration reachability across all vertices
float calc_exploration_reach(InternalGraph& graph);

} // namespace deglib::analysis
```

---

## 7. Namespace: `deglib::graph::` (Internal Graphs & Results)

Headers: `<deglib/graph/*.h>`

Low-level internal graph representations operating directly on contiguous **internal indices** (`0..N-1`) for maximum traversal speed, along with ANNS search result types.

```cpp
namespace deglib::graph {

/// Single neighbor candidate (identifier and distance)
class ObjectDistance {
public:
    uint32_t getIdentifier(); ///< Vertex label or internal index
    float getDistance();      ///< Computed distance to query
};

/// Max-heap containing the top-k nearest neighbor results
class ResultSet {
public:
    size_t size();
    bool empty();
    ObjectDistance top();                                       ///< Candidate with largest distance in heap
    void pop();                                                 ///< Remove largest candidate
    void emplace(uint32_t identifier, float distance);
    ObjectDistance replace_top(uint32_t identifier, float distance); ///< Replace top candidate
    void reserve(size_t capacity);
    void sort();                                                ///< Sort elements in ascending distance order
};

// --- Graph Interfaces ---

/// Abstract base interface for 0..N-1 indexed graphs
class InternalGraph {
public:
    uint32_t size();
    uint8_t getEdgesPerVertex();
    FloatSpace getFeatureSpace();
    bool hasVertex(uint32_t label);
    uint32_t getExternalLabel(uint32_t internal_index);
    uint32_t getInternalIndex(uint32_t label);
    uint32_t* getNeighborIndices(uint32_t internal_index);
    uint32_t* getEntryVertexIndices();
    bool hasEdge(uint32_t from_index, uint32_t to_index);

    ResultSet search(span<float> query, uint32_t k, float eps = 0.0f, Filter* filter = nullptr, uint32_t max_dc = 0);
    ResultSet explore(uint32_t entry_index, uint32_t k, uint32_t max_dc = 0, float eps = 0.0f, bool include_entry = true, Filter* filter = nullptr);
};

/// Abstract base interface for mutable graphs supporting vertex & edge updates
class MutableGraph : public InternalGraph {
public:
    bool addVertex(uint32_t label, byte* feature_vector);
    bool removeVertex(uint32_t label);
    void changeEdges(uint32_t internal_index, uint32_t* neighbor_indices, float* neighbor_weights);
    float* getNeighborWeights(uint32_t internal_index);
    float getEdgeWeight(uint32_t from_index, uint32_t to_index);
    byte* getFeatureVector(uint32_t internal_index);
    bool saveGraph(const char* file_path);
};

// --- Concrete Graph Implementations ---

/// Mutable graph with fixed maximum capacity and flat preallocated memory arrays
class SizeBoundedGraph : public MutableGraph {
public:
    SizeBoundedGraph(uint32_t max_vertex_count, uint8_t edges_per_vertex, FloatSpace feature_space);

    static SizeBoundedGraph create_random_graph(byte* feature_data, uint32_t vertex_count, uint8_t edges_per_vertex, FloatSpace feature_space, uint32_t seed = 7);
    static SizeBoundedGraph from_graph(InternalGraph& graph, uint32_t new_max_size = 0);
    static SizeBoundedGraph from_graph(InternalGraph& graph, FloatSpace custom_space, void* custom_features = nullptr, uint32_t new_max_size = 0);
    static SizeBoundedGraph load_from_file(const char* file_path, FloatSpace feature_space);
};

/// Mutable graph with chunk-allocated dynamically growing memory
class DynamicGraph : public MutableGraph {
public:
    DynamicGraph(uint8_t edges_per_vertex, FloatSpace feature_space, uint32_t chunk_size = 1024);

    static DynamicGraph from_graph(InternalGraph& graph, uint32_t chunk_size = 1024);
    static DynamicGraph from_graph(InternalGraph& graph, FloatSpace custom_space, void* custom_features = nullptr, uint32_t chunk_size = 1024);
    static DynamicGraph load_from_file(const char* file_path, FloatSpace feature_space, uint32_t chunk_size = 1024);
};

/// Compact immutable graph layout optimized for query serving
class ReadOnlyGraph : public InternalGraph {
public:
    ReadOnlyGraph(uint32_t max_vertex_count, uint8_t edges_per_vertex, FloatSpace feature_space);
    ReadOnlyGraph(uint32_t max_vertex_count, uint8_t edges_per_vertex, FloatSpace feature_space, InternalGraph& graph);

    static ReadOnlyGraph load_from_file(const char* file_path, FloatSpace feature_space);
};

} // namespace deglib::graph
```

---

## 8. Namespace: `deglib::cpu::`

Header: `<deglib/utils/cpu.h>`

Runtime hardware feature detection and SIMD instruction set configuration.

```cpp
namespace deglib::cpu {

enum class InstructionSet {
    Auto,   ///< Automatically select the highest instruction set supported by host CPU
    Scalar, ///< Standard scalar operations
    AVX2,   ///< 256-bit AVX2 + FMA SIMD
    AVX512  ///< 512-bit AVX-512 SIMD
};

/// Runtime AVX2 support check
bool has_avx2();

/// Runtime AVX-512 support check
bool has_avx512();

/// Returns string representation ("Auto", "Scalar", "AVX2", "AVX512")
const char* instruction_set_to_string(InstructionSet inst);

} // namespace deglib::cpu
```

---

## 9. Namespace: `deglib::memory::`

Header: `<deglib/utils/memory.h>`

Cache line prefetching utilities.

```cpp
namespace deglib::memory {

/// Prefetches memory into L1 cache for subsequent distance computations
void prefetch(void* ptr, size_t size = 128);

} // namespace deglib::memory
```

---

## 10. Namespace: `deglib::concurrent::`

Header: `<deglib/concurrent.h>`

High-throughput multithreading loop primitives.

```cpp
namespace deglib::concurrent {

/// Parallel loop using dynamic work stealing across threads
void parallel_for(size_t start, size_t end, size_t num_threads, auto fn);

/// Parallel loop using contiguous chunk partitioning for cache locality
void parallel_batch_for(size_t start, size_t end, size_t num_threads, auto fn);

} // namespace deglib::concurrent
```
