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
| [`deglib::transformation::`](#11-namespace-deglibtransformation) | `<deglib/transformation.h>` | Query and database feature transformation and quantization pipeline. |
| [`deglib::cpu::`](#8-namespace-deglibcpu) | `<deglib/utils/cpu.h>` | Hardware feature detection (AVX2, AVX512) and `InstructionSet` configuration. |
| [`deglib::memory::`](#9-namespace-deglibmemory) | `<deglib/utils/memory.h>` | CPU cache line prefetching utilities. |
| [`deglib::concurrent::`](#10-namespace-deglibconcurrent) | `<deglib/concurrent.h>` | Parallel execution loops (`parallel_for`, `parallel_batch_for`). |

---

## 1. Namespace: `deglib::` (User Facade & Top-Level Functions)

Header: `<deglib/deglib.h>` / `<deglib/graph.h>`

The root namespace provides [`DynamicExplorationGraph`](#class-dynamicexplorationgraph), the primary user-facing class that wraps internal graph structures and maps external object labels (`uint32_t`) to internal graph indices, as well as high-level factory and loading functions.

```cpp
namespace deglib {

// --- Top-Level Graph Creation, Building & Loading Functions ---

/// Build and optimize a DynamicExplorationGraph directly from contiguous feature data in one call
template <typename T = float>
DynamicExplorationGraph build_from_data(
    std::span<const T> data,
    uint32_t dims,
    std::span<const uint32_t> labels = {},
    uint8_t edges_per_vertex = 32,
    Metric metric = Metric::FP32_L2,
    OptimizationTarget target = OptimizationTarget::LowLID,
    uint8_t extend_k = 64,
    float extend_eps = 0.1f,
    uint8_t improve_k = 0,
    float improve_eps = 0.001f,
    uint8_t max_path_length = 5,
    uint32_t improve_tries = 0,
    size_t thread_count = 0,
    uint32_t seed = 42,
    std::function<void(BuilderStatus&)> callback = nullptr
);

/// Create an empty mutable graph with fixed maximum capacity (backed by SizeBoundedGraph)
DynamicExplorationGraph create_empty(
    uint32_t max_vertex_count,
    uint8_t edges_per_vertex,
    const FloatSpace& feature_space
);

/// Create an empty mutable graph with dynamic chunk-allocated growth (backed by DynamicGraph)
DynamicExplorationGraph create_dynamic_empty(
    uint8_t edges_per_vertex,
    const FloatSpace& feature_space,
    uint32_t chunk_size = 1024
);

/// Create a randomized exploration graph from raw feature data
DynamicExplorationGraph create_random_graph(
    const std::byte* feature_data,
    uint32_t vertex_count,
    uint8_t edges_per_vertex,
    const FloatSpace& feature_space,
    uint32_t seed = 7
);

/// Load a saved graph file as an immutable, compact graph (backed by ReadOnlyGraph)
DynamicExplorationGraph load_readonly_graph(const char* path_graph);

/// Load a saved graph file as a dynamic chunk-allocated mutable graph (backed by DynamicGraph)
DynamicExplorationGraph load_dynamic_graph(const char* path_graph, uint32_t chunk_size = 1024);

/// Load a saved graph file as a fixed-capacity mutable graph (backed by SizeBoundedGraph)
DynamicExplorationGraph load_mutable_graph(const char* path_graph, uint32_t new_max_size = 0);

/// Create an EvenRegularGraphBuilder directly for a DynamicExplorationGraph
EvenRegularGraphBuilder create_builder(
    DynamicExplorationGraph& graph,
    std::mt19937& rnd,
    OptimizationTarget optimization_target = OptimizationTarget::LowLID,
    uint8_t extend_k = 0,
    float extend_eps = 0.1f,
    uint8_t improve_k = 0,
    float improve_eps = 0.001f,
    uint8_t max_path_length = 5,
    uint32_t improve_tries = 0
);

class DynamicExplorationGraph {
public:
    // --- Constructors & Ownership ---
    explicit DynamicExplorationGraph(
        std::unique_ptr<deglib::graph::InternalGraph> graph,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    );
    explicit DynamicExplorationGraph(
        deglib::graph::InternalGraph& graph,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    );

    // --- Search (Single Query) ---

    /// Search k-nearest neighbors using dynamic exploration radius (eps) -> returns max-heap ResultSet
    template <typename T = float>
    ResultSet search(
        std::span<const T> query,
        uint32_t k,
        float eps = 0.0f,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0
    ) const;

    /// Search k-nearest neighbors using fixed beam width (ef) -> returns ascending sorted vector
    template <typename T = float>
    std::vector<ObjectDistance> search_ef(
        std::span<const T> query,
        uint32_t k,
        uint32_t ef,
        bool include_entry = true,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0
    ) const;

    // --- Search Batch (Parallel Multithreaded) ---

    /// Batch search with dynamic exploration radius (eps) returning container of ResultSets
    template <typename T = float>
    std::vector<ResultSet> search_batch(
        std::span<const T> queries,
        size_t count,
        uint32_t k,
        float eps = 0.0f,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        size_t threads = 1
    ) const;

    /// High-performance zero-allocation batch search into pre-allocated flat spans (eps)
    template <typename T = float>
    void search_batch(
        std::span<const T> queries,
        size_t count,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        float eps = 0.0f,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        size_t threads = 1,
        bool unsorted = false
    ) const;

    /// High-performance zero-allocation batch search into pre-allocated flat spans (ef beam width)
    template <typename T = float>
    void search_batch_ef(
        std::span<const T> queries,
        size_t count,
        uint32_t k,
        uint32_t ef,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        bool include_entry = true,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        size_t threads = 1,
        bool unsorted = false
    ) const;

    /// Batch search with fixed beam width (ef) returning vector of sorted results
    template <typename T = float>
    std::vector<std::vector<ObjectDistance>> search_batch_ef(
        std::span<const T> queries,
        size_t count,
        uint32_t k,
        uint32_t ef,
        bool include_entry = true,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        size_t threads = 1
    ) const;

    // --- Exploration & Path Traversal ---

    /// Explore starting from an existing external label using dynamic exploration radius (eps)
    ResultSet explore(
        uint32_t entry_external_label,
        uint32_t k,
        uint32_t max_distance_computations = 0,
        float eps = 0.0f,
        bool include_entry = true,
        const Filter* filter = nullptr
    ) const;

    /// Explore starting from an existing external label using fixed beam width (ef)
    std::vector<ObjectDistance> explore_ef(
        uint32_t entry_external_label,
        uint32_t k,
        uint32_t ef,
        uint32_t max_distance_computations = 0,
        bool include_entry = true,
        const Filter* filter = nullptr
    ) const;

    /// Batch exploration into pre-allocated flat spans (eps)
    void explore_batch(
        std::span<const uint32_t> entry_external_labels,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        uint32_t max_distance_computations = 0,
        float eps = 0.0f,
        bool include_entry = true,
        const Filter* filter = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const;

    /// Batch exploration returning flat indices and optional distances (eps)
    std::pair<std::vector<uint32_t>, std::vector<float>> explore_batch(
        std::span<const uint32_t> entry_external_labels,
        uint32_t k,
        uint32_t max_distance_computations = 0,
        float eps = 0.0f,
        bool include_entry = true,
        const Filter* filter = nullptr,
        size_t threads = 1,
        bool return_distances = true,
        bool unsorted = false
    ) const;

    /// Batch exploration into pre-allocated flat spans (ef)
    void explore_batch_ef(
        std::span<const uint32_t> entry_external_labels,
        uint32_t k,
        uint32_t ef,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        uint32_t max_distance_computations = 0,
        bool include_entry = true,
        const Filter* filter = nullptr,
        size_t threads = 1,
        bool unsorted = false
    ) const;

    /// Batch exploration returning flat indices and optional distances (ef)
    std::pair<std::vector<uint32_t>, std::vector<float>> explore_batch_ef(
        std::span<const uint32_t> entry_external_labels,
        uint32_t k,
        uint32_t ef,
        uint32_t max_distance_computations = 0,
        bool include_entry = true,
        const Filter* filter = nullptr,
        size_t threads = 1,
        bool return_distances = true,
        bool unsorted = false
    ) const;

    /// Greedy best-first reachability search towards a stored vertex; backtracks traversal path
    std::vector<ObjectDistance> hasPath(
        const std::vector<uint32_t>& entry_external_labels,
        uint32_t to_external_label,
        float eps = 0.0f,
        uint32_t k = 1
    ) const;

    // --- Neighbor Retrieval ---

    /// Retrieve neighbor external labels for a single vertex
    std::vector<uint32_t> getNeighbors(uint32_t external_label) const;

    /// Batch retrieve neighbor external labels into caller-supplied flat span
    void getNeighbors_batch(
        std::span<const uint32_t> external_labels,
        std::span<uint32_t> out_neighbors,
        size_t threads = 1
    ) const;

    /// Batch retrieve neighbor external labels returning flat vector
    std::vector<uint32_t> getNeighbors_batch(
        std::span<const uint32_t> external_labels,
        size_t threads = 1
    ) const;

    // --- State & Inspection ---

    uint32_t size() const;                                 ///< Total number of active vertices
    uint8_t getEdgesPerVertex() const;                     ///< Outgoing edges per vertex
    const FloatSpace& getFeatureSpace() const;             ///< Graph metric and dimensionality
    bool hasVertex(uint32_t external_label) const;         ///< Check if external label exists
    bool isMutable() const;                                ///< Check if graph allows modifications
    InternalGraph& internal();                             ///< Access underlying internal graph
    const InternalGraph& internal() const;

    // --- Transformations Pipeline ---

    bool hasTransformations() const noexcept;              ///< True if query transformations are attached
    size_t getInputDim() const noexcept;                   ///< Expected raw query dimensionality
    const auto& getTransformations() const noexcept;       ///< Attached transformation stages

    // --- Traversal Prefetch Tuning & Entry Vertices ---

    const std::vector<uint32_t>& getEntryVertexIndices() const;
    void setEntryVertexIndices(std::vector<uint32_t> indices);
    int32_t getPo() const;                                 ///< Feature prefetch offset
    int32_t getPl() const;                                 ///< Feature cache line prefetch count
    int32_t getNl() const;                                 ///< Neighbor prefetch look-ahead
    void setPo(int32_t po);
    void setPl(int32_t pl);
    void setNl(int32_t nl);
    void setPrefetch(int32_t po, int32_t pl, int32_t nl = 3);

    /// Auto-tune traversal prefetch parameters (po, pl, nl) by timing traversal over sampled vertices
    void optimize(size_t sample_count = 50, uint32_t k = 100, uint32_t ef = 200, uint32_t seed = 7) const;

    // --- Conversions & Persistence ---

    /// Convert to a compact, read-only graph (ReadOnlyGraph) optimized for search
    DynamicExplorationGraph to_readonly() const;

    /// Convert to a mutable SizeBoundedGraph (fixed capacity)
    DynamicExplorationGraph to_mutable(uint32_t new_max_size = 0) const;
    DynamicExplorationGraph to_mutable(const FloatSpace& new_space, const void* new_features = nullptr, uint32_t new_max_size = 0) const;

    /// Convert to a mutable DynamicGraph (chunk-allocated dynamic growth)
    DynamicExplorationGraph to_dynamic(uint32_t chunk_size = 1024) const;
    DynamicExplorationGraph to_dynamic(const FloatSpace& new_space, const void* new_features = nullptr, uint32_t chunk_size = 1024) const;

    /// Save graph topology, features, transformations, and prefetch parameters to binary file
    bool saveGraph(const char* path) const;
    bool saveGraph(const std::string& path) const;

    /// Load graph from binary file
    static DynamicExplorationGraph loadGraph(const char* path);
    static DynamicExplorationGraph loadGraph(const std::string& path);
};

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
    /// Construct builder for a MutableGraph
    EvenRegularGraphBuilder(
        deglib::graph::MutableGraph& graph,
        mt19937& rnd,
        OptimizationTarget target = OptimizationTarget::LowLID,
        uint8_t extend_k = 0,
        float extend_eps = 0.1f,
        uint8_t improve_k = 0,
        float improve_eps = 0.001f,
        uint8_t max_path_length = 5,
        uint32_t improve_tries = 0
    );

    /// Construct builder for a DynamicExplorationGraph facade (inherits transformations)
    EvenRegularGraphBuilder(
        DynamicExplorationGraph& graph,
        mt19937& rnd,
        OptimizationTarget target = OptimizationTarget::LowLID,
        uint8_t extend_k = 0,
        float extend_eps = 0.1f,
        uint8_t improve_k = 0,
        float improve_eps = 0.001f,
        uint8_t max_path_length = 5,
        uint32_t improve_tries = 0
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
    Uint8_InnerProduct,///< Unsigned 8-bit integer Inner Product distance
    FP16_L2,           ///< Half-precision FP16 Euclidean L2 distance
    FP16_InnerProduct, ///< Half-precision FP16 Inner Product distance
    EVP_InnerProduct,  ///< Extreme Vector Quantization (1-bit / ternary) Inner Product
    Int8_InnerProduct, ///< Signed 8-bit integer Inner Product distance
    Int8_L2            ///< Signed 8-bit integer L2 distance
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

/// Stateful reranker holding base vectors for repeated exact-distance candidate reranking
template <typename DataT>
class Reranker {
public:
    Reranker(FloatSpace space, const DataT* base_vectors, size_t num_base_vectors);

    FloatSpace getSpace() const;                    ///< Distance metric and dimensionality
    const DataT* getBaseVectors() const;            ///< Held base vectors
    size_t getNumBaseVectors() const;               ///< Number of held base vectors
    int32_t getPo() const;                          ///< Rerank feature prefetch offset (0 = prefetch disabled)
    int32_t getPl() const;                          ///< Cache lines prefetched per feature vector (0 = whole vector)
    void setPo(int32_t po);                         ///< Set the prefetch offset; values <= 0 disable prefetching
    void setPl(int32_t pl);                         ///< Set the prefetch length; values <= 0 prefetch the whole vector
    void setPrefetch(int32_t po, int32_t pl);       ///< Set both prefetch parameters at once

    /// Rerank candidate indices for one query, returning the top-k as a ResultSet (max-heap)
    ResultSet rerank(span<const byte> query, span<const uint32_t> candidate_indices, uint32_t k) const;

    /// Multi-threaded batch reranking of candidate indices for many queries
    vector<ResultSet> rerank(
        const void* queries, size_t num_queries, const uint32_t* base_candidates,
        size_t candidates_per_query, size_t k_top = 0, size_t num_threads = 0
    ) const;

    /// Auto-tune rerank prefetch parameters (po, pl) by timing reranking over sampled base vectors
    void optimize(size_t sample_count = 50, size_t num_candidates = 100, uint32_t k = 10);
};

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

/// Parallel removal of edges violating the Relative Neighborhood Graph (RNG) rule
uint32_t prune_non_rng_edges(MutableGraph& graph, size_t num_threads = 0);

/// Remove non-RNG edges using a globally weight-sorted strategy
uint32_t prune_non_rng_edges_weight_sorted(MutableGraph& graph, size_t num_threads = 0);

/// Iteratively remove non-RNG edges per-vertex until convergence
uint32_t prune_non_rng_edges_iterative(MutableGraph& graph, size_t num_threads = 0);

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

/// EVP quantizer holding the shared non_zeros setting.
/// Construct once and reuse for database and query quantization.
/// Defined in deglib::quantization::evp; the factory make_evp_quantizer lives in deglib::optimization.
class EvpQuantizer {
    explicit EvpQuantizer(uint32_t non_zeros);
    void quantize(float* src, byte* dst, size_t count, uint32_t dim, size_t num_threads = 0);
    void quantize(uint16_t* src, byte* dst, size_t count, uint32_t dim, size_t num_threads = 0);
    vector<byte> quantize(float* src, size_t count, uint32_t dim, size_t num_threads = 0);
    vector<byte> quantize(uint16_t* src, size_t count, uint32_t dim, size_t num_threads = 0);
    // Same four flavors with std::span (count derived from span size, validated).
};
EvpQuantizer make_evp_quantizer(uint32_t non_zeros);

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
    const std::vector<uint32_t>& getEntryVertexIndices() const;
    bool hasEdge(uint32_t from_index, uint32_t to_index);

    ResultSet search(std::span<const float> query, uint32_t k, float eps = 0.0f, const Filter* filter = nullptr, uint32_t max_dc = 0);
    std::vector<ObjectDistance> search_ef(std::span<const float> query, uint32_t k, uint32_t ef, bool include_entry = true, const Filter* filter = nullptr, uint32_t max_dc = 0);
    ResultSet explore(uint32_t entry_index, uint32_t k, uint32_t max_dc = 0, float eps = 0.0f, bool include_entry = true, const Filter* filter = nullptr);
    std::vector<ObjectDistance> explore_ef(uint32_t entry_index, uint32_t k, uint32_t ef, uint32_t max_dc = 0, bool include_entry = true, const Filter* filter = nullptr);
    virtual std::vector<ObjectDistance> hasPath(const std::vector<uint32_t>& entry_indices, uint32_t to_vertex, float eps, uint32_t k) const = 0;

    // --- Traversal Prefetch Tuning ---
    int32_t getPo() const;                               ///< Feature prefetch offset (candidates ahead)
    int32_t getPl() const;                               ///< Feature prefetch look-ahead (cache lines)
    int32_t getNl() const;                               ///< Neighbor prefetch look-ahead
    void setPrefetch(int32_t po, int32_t pl, int32_t nl = 3);
    /// Auto-tune (po, pl, nl) by timing traversal over sampled vertices (self-sampled features).
    void optimize(size_t sample_count = 50, uint32_t k = 100, uint32_t ef = 200, uint32_t seed = 7);
};

/// Abstract base interface for mutable graphs supporting vertex & edge updates
class MutableGraph : public InternalGraph {
public:
    uint32_t addVertex(uint32_t label, byte* feature_vector);
    std::vector<uint32_t> removeVertex(uint32_t label);
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

    static SizeBoundedGraph create_empty(uint32_t max_vertex_count, uint8_t edges_per_vertex, FloatSpace feature_space);
    static SizeBoundedGraph from_graph(InternalGraph& graph, uint32_t new_max_size = 0);
    static SizeBoundedGraph from_graph(InternalGraph& graph, FloatSpace custom_space, void* custom_features = nullptr, uint32_t new_max_size = 0);
    // Loading is done via the free function deglib::graph::load_sizebounded_graph(const char* file_path, uint32_t new_max_size = 0)
};

/// Mutable graph with chunk-allocated dynamically growing memory
class DynamicGraph : public MutableGraph {
public:
    DynamicGraph(uint8_t edges_per_vertex, FloatSpace feature_space, uint32_t chunk_size = 1024);

    static DynamicGraph from_graph(InternalGraph& graph, uint32_t chunk_size = 1024);
    static DynamicGraph from_graph(InternalGraph& graph, FloatSpace custom_space, void* custom_features = nullptr, uint32_t chunk_size = 1024);
    // Loading is done via the free function deglib::graph::load_dynamic_graph(const char* file_path, uint32_t chunk_size = 1024)
};

/// Compact immutable graph layout optimized for query serving
class ReadOnlyGraph : public InternalGraph {
public:
    ReadOnlyGraph(uint32_t max_vertex_count, uint8_t edges_per_vertex, FloatSpace feature_space);
    ReadOnlyGraph(uint32_t max_vertex_count, uint8_t edges_per_vertex, FloatSpace feature_space, InternalGraph& graph);

    // Loading is done via the free function deglib::graph::load_readonly_graph(const char* file_path)
};

} // namespace deglib::graph
```

---

## 8. Namespace: `deglib::cpu::`

Header: `<deglib/utils/cpu.h>`

Runtime hardware feature detection and SIMD instruction set configuration.

```cpp
namespace deglib::cpu {

enum class InstructionSet : uint8_t {
    Auto = 0,        ///< Automatically select the highest instruction set supported by host CPU
    Scalar = 1,      ///< Standard scalar operations
    AVX2 = 2,        ///< 256-bit AVX2 + FMA SIMD
    AVX2_VNNI = 3,   ///< 256-bit AVX2 with VNNI acceleration
    AVX512 = 4,      ///< 512-bit AVX-512 SIMD
    AVX512_VNNI = 5  ///< 512-bit AVX-512 with VNNI acceleration
};

/// Runtime AVX2 support check
bool has_avx2();

/// Runtime AVX-512 support check
bool has_avx512();

/// Returns string representation ("Auto", "Scalar", "AVX2", "AVX2_VNNI", "AVX512", "AVX512_VNNI")
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
void prefetch(const char* ptr, size_t size = 128);

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

---

## 11. Namespace: `deglib::transformation::`

Header: `<deglib/transformation.h>`

Vector transformation pipeline for preprocessing and quantizing query and database vectors transparently.

```cpp
namespace deglib::transformation {

/// Base interface for vector feature transformations and quantization stages
class FeatureTransformation {
public:
    virtual ~FeatureTransformation() = default;
    virtual size_t input_dim() const = 0;
    virtual size_t output_dim() const = 0;
    virtual size_t output_bytes() const = 0;
    virtual TransformationType type() const = 0;
    virtual void transform_database(const float* in, std::byte* out, size_t count) const = 0;
    virtual void transform_query(const float* in, std::byte* out, size_t count) const = 0;
};

/// Workspace buffer for query transformation without heap allocations on search path
class QueryWorkspace {
public:
    std::byte* get_buffer_a(size_t bytes);
    std::byte* get_buffer_b(size_t bytes);
};

// --- Pipeline Functions ---

/// Calculate total output byte size of the final transformation stage
size_t get_output_bytes(const std::vector<std::shared_ptr<FeatureTransformation>>& chain);

/// Apply sequential transformations to database vectors
void apply_database(const std::vector<std::shared_ptr<FeatureTransformation>>& chain, const float* in, std::byte* out, size_t count);

/// Apply sequential transformations to query vectors
void apply_query(const std::vector<std::shared_ptr<FeatureTransformation>>& chain, const float* in, std::byte* out, size_t count);

/// Serialize transformations to binary byte stream
std::vector<std::byte> serialize_transformations(const std::vector<std::shared_ptr<FeatureTransformation>>& chain);

/// Deserialize transformations from binary byte stream
std::vector<std::shared_ptr<FeatureTransformation>> deserialize_transformations(const std::byte* data, size_t size);

} // namespace deglib::transformation
```

