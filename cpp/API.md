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
| [`deglib::optimization::`](#5-namespace-degliboptimization) | `<deglib/optimization.h>` | MRNG pruning and FLAS 1D presorting. |
| [`deglib::analysis::`](#6-namespace-deglibanalysis) | `<deglib/analysis.h>` | Graph health diagnostics, regularity validation, and reachability metrics. |
| [`deglib::graph::`](#7-namespace-deglibgraph-internal-graphs--results) | `<deglib/graph/*.h>` | Low-level internal graph hierarchy (`0..N-1` indices), reusable `SearchWorkspace`, and `ResultSet`. |
| [`deglib::transformation::`](#11-namespace-deglibtransformation) | `<deglib/transformation.h>` | Query and database feature transformation and quantization pipeline. |
| [`deglib::tuning::`](#12-namespace-deglibtuning) | `<deglib/tuning.h>` | Pre-build dataset profiling, `OptimizationTarget` prediction, and quantizer metric prediction. |
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

    /// Search k-nearest neighbors using dynamic exploration radius (eps) -> returns sorted SearchResults (unless unsorted=true)
    template <typename T = float>
    deglib::search::SearchResults search(
        std::span<const T> query,
        uint32_t k,
        float eps = 0.0f,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        bool return_distances = true,
        bool unsorted = false,
        bool pad_missing = false
    ) const;

    /// Search k-nearest neighbors using fixed beam width (ef) -> returns SearchResults
    template <typename T = float>
    deglib::search::SearchResults search_ef(
        std::span<const T> query,
        uint32_t k,
        uint32_t ef,
        bool include_entry = true,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        bool return_distances = true,
        bool pad_missing = false
    ) const;

    // --- Search Batch (Parallel Multithreaded) ---

    /// Batch search with dynamic exploration radius (eps) returning flat SearchResults with telemetry
    template <typename T = float>
    deglib::search::SearchResults search_batch(
        std::span<const T> queries,
        size_t count,
        uint32_t k,
        float eps = 0.0f,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        size_t threads = 1,
        bool return_distances = true,
        bool unsorted = false
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

    /// Batch search with fixed beam width (ef) returning flat SearchResults with telemetry
    template <typename T = float>
    deglib::search::SearchResults search_batch_ef(
        std::span<const T> queries,
        size_t count,
        uint32_t k,
        uint32_t ef,
        bool include_entry = true,
        const Filter* filter = nullptr,
        uint32_t max_distance_computations = 0,
        size_t threads = 1,
        bool return_distances = true,
        bool unsorted = false
    ) const;

    // --- Exploration & Path Traversal ---

    /// Explore starting from an existing external label using dynamic exploration radius (eps)
    deglib::search::SearchResults explore(
        uint32_t entry_external_label,
        uint32_t k,
        uint32_t max_distance_computations = 0,
        float eps = 0.0f,
        bool include_entry = true,
        const Filter* filter = nullptr,
        bool return_distances = true,
        bool unsorted = false
    ) const;

    /// Explore starting from an existing external label using fixed beam width (ef)
    deglib::search::SearchResults explore_ef(
        uint32_t entry_external_label,
        uint32_t k,
        uint32_t ef,
        uint32_t max_distance_computations = 0,
        bool include_entry = true,
        const Filter* filter = nullptr,
        bool return_distances = true
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

    /// Batch exploration returning SearchResults (eps)
    deglib::search::SearchResults explore_batch(
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

    /// Batch exploration returning SearchResults (ef)
    deglib::search::SearchResults explore_batch_ef(
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
        const std::vector<uint32_t>& from_external_labels,
        uint32_t to_external_label,
        float eps = 0.0f,
        uint32_t k = 1
    ) const;

    // --- Neighbor Retrieval ---

    /// Retrieve neighbor external labels for a single vertex
    std::vector<uint32_t> getNeighbors(uint32_t external_label) const;

    /// Batch retrieve neighbor external labels into caller-supplied flat span
    void getNeighbors(
        std::span<const uint32_t> external_labels,
        std::span<uint32_t> out_neighbors,
        size_t threads = 1
    ) const;

    /// Batch retrieve neighbor external labels returning flat vector
    std::vector<uint32_t> getNeighbors(
        std::span<const uint32_t> external_labels,
        size_t threads = 1
    ) const;

    // --- State & Inspection ---

    uint32_t size() const;                                 ///< Total number of active vertices
    uint8_t getEdgesPerVertex() const;                     ///< Outgoing edges per vertex
    const FloatSpace& getFeatureSpace() const;             ///< Graph metric and dimensionality
    bool hasVertex(uint32_t external_label) const;         ///< Check if external label exists
    std::vector<uint32_t> getExternalLabels() const;       ///< All external labels ordered by internal index (0 .. size()-1)
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
    void optimize(size_t sample_count = 300, uint32_t k = 100, uint32_t ef = 200, size_t median_subset = 4096) const;

    /// Calibrate a single-mode RecallTable mapping target recall@k values to ef (in [10, 500], step 1) or eps (in [0.001, 0.20], resolution 0.0001)
    deglib::search::RecallTable to_recall_table(
        uint32_t k,
        deglib::search::RecallTable::Mode mode,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        size_t threads = 0
    ) const;
    deglib::search::RecallTable to_recall_table(
        uint32_t k,
        deglib::search::RecallTable::Mode mode,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        size_t threads = 0
    ) const;

    /// Calibrate both Ef and Eps RecallTables in one pass sharing exact ground truth
    std::vector<deglib::search::RecallTable> to_recall_tables(
        uint32_t k,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        size_t threads = 0
    ) const;
    std::vector<deglib::search::RecallTable> to_recall_tables(
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        size_t threads = 0
    ) const;

    /// Build a unified ParetoLine over real measured RecallTable operating points in (measured_recall, latency_us) space
    deglib::search::ParetoLine to_pareto_line(
        std::span<const deglib::search::RecallTable> recall_tables,
        size_t sample_count = 0,
        size_t rounds = 3
    ) const;
    deglib::search::ParetoLine to_pareto_line(
        const deglib::search::RecallTable& recall_table,
        size_t sample_count = 0,
        size_t rounds = 3
    ) const;
    deglib::search::ParetoLine to_pareto_line(
        uint32_t k,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        size_t threads = 0,
        size_t rounds = 3
    ) const;
    deglib::search::ParetoLine to_pareto_line(
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        size_t threads = 0,
        size_t rounds = 3
    ) const;

    // --- Conversions & Persistence ---
    // Feature Memory Ordering:
    // Any replacement feature buffer (custom_features) must match the graph's internal index layout (row i = vertex i):
    // 1. Standard / Static Graphs (Default): Vertex i is simply the i-th vector given to the builder. Pass the same feature array directly!
    // 2. Custom Labels or Dynamic Graphs with Deletions: Vertices are indexed by their current internal slots.
    //    Use getExternalLabels() to look up which label resides at which slot (e.g. data[labels[i]]).

    /// Pure backend copies (preserve topology, labels, features, and query transformations)
    DynamicExplorationGraph to_readonly() const;
    DynamicExplorationGraph to_mutable(uint32_t new_max_size = 0, size_t num_threads = 0) const;
    DynamicExplorationGraph to_dynamic(uint32_t chunk_size = 1024, size_t num_threads = 0) const;

    /// Backend-preserving content operations (keep ReadOnlyGraph, SizeBoundedGraph capacity, or DynamicGraph chunk_size)
    DynamicExplorationGraph with_metric(
        deglib::distances::FloatSpace target_space,
        size_t num_threads = 0
    ) const;
    DynamicExplorationGraph with_transformations(
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> additional_transformations,
        std::optional<deglib::distances::FloatSpace> target_space = std::nullopt,
        size_t num_threads = 0
    ) const;
    DynamicExplorationGraph with_features(
        deglib::distances::FloatSpace target_space,
        const void* custom_features,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> new_transformations = {},
        size_t num_threads = 0
    ) const;

    /// Extract stored float features directly into an owning Reranker, prepending any attached graph transformations
    deglib::search::Reranker to_reranker(
        deglib::distances::FloatSpace target_space,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
        size_t num_threads = 0
    ) const;

    /// Create a Searcher over this graph using its attached query transformations (graph must outlive Searcher)
    deglib::search::Searcher to_searcher(
        std::shared_ptr<deglib::search::Reranker> reranker = nullptr,
        bool normalize_query = false
    ) const;
    deglib::search::Searcher to_searcher(
        deglib::search::Reranker reranker,
        bool normalize_query = false
    ) const;

    /// Save graph topology, features, transformations, and prefetch parameters to binary file
    bool saveGraph(const char* path) const;
    bool saveGraph(const std::string& path) const;

    /// Load graph from binary file into an immutable read-only graph (ReadOnlyGraph)
    static DynamicExplorationGraph load_readonly(const char* path);
    static DynamicExplorationGraph load_readonly(const std::string& path);

    /// Load graph from binary file into a mutable SizeBoundedGraph
    static DynamicExplorationGraph load_mutable(const char* path, uint32_t new_max_size = 0);
    static DynamicExplorationGraph load_mutable(const std::string& path, uint32_t new_max_size = 0);

    /// Load graph from binary file into a mutable chunk-allocated DynamicGraph
    static DynamicExplorationGraph load_dynamic(const char* path, uint32_t chunk_size = 1024);
    static DynamicExplorationGraph load_dynamic(const std::string& path, uint32_t chunk_size = 1024);
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
    EVP_InnerProduct,  ///< Equi-Voronoi Polytopes (EVP, 1-bit / ternary) Inner Product
    Int8_InnerProduct, ///< Signed 8-bit integer Inner Product distance
    Int8_L2,           ///< Signed 8-bit integer L2 distance
    Uint4_L2,          ///< Unsigned 4-bit integer L2 distance (packed 2 per byte)
    Uint4_InnerProduct,///< Unsigned 4-bit integer Inner Product distance (packed 2 per byte)
    Int4_L2,           ///< Signed 4-bit integer L2 distance (packed 2 per byte)
    Int4_InnerProduct, ///< Signed 4-bit integer Inner Product distance (packed 2 per byte)
    UInt7_L2,          ///< Unsigned 7-bit integer L2 distance in [0, 127]
    UInt7_InnerProduct ///< Unsigned 7-bit integer Inner Product distance in [0, 127]
};

/// Represents a vector feature space (dimensionality, metric, byte layout, and SIMD kernel)
class FloatSpace {
public:
    FloatSpace(size_t dim, Metric metric, InstructionSet instruction = InstructionSet::Auto);

    size_t dim();                  ///< Vector dimensionality
    Metric metric();               ///< Configured distance metric
    size_t get_data_size();        ///< Size in bytes of a single vector
    const char* get_instruction(); ///< Active SIMD kernel ("AVX512_VNNI", "AVX512", "AVX2_VNNI", "AVX2", "Scalar")
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

/// Stateful reranker holding base vectors for repeated exact-distance candidate reranking.
///
/// Candidate Indexing Contract:
/// The Reranker evaluates candidate index `c` directly as row `base_vectors[c]`.
/// - With Searcher (Most Common): Searcher passes graph internal indices (0 .. size()-1) into Reranker,
///   then remaps winning results to external labels.
///   * Standard / Static Graphs: Vertex i is simply the i-th vector given to the builder. Pass the same feature array directly!
///   * Custom Labels or Dynamic Graphs with Deletions: Row i must contain the vector for label `graph.getExternalLabels()[i]`.
/// - Standalone Usage: The row ordering of `base_vectors` matches whatever candidate ID space you pass
///   (e.g. if candidates are external labels 0..N-1, row l contains feature vector l).
class Reranker {
public:
    Reranker(
        FloatSpace space,
        std::span<const std::byte> base_vectors,
        size_t num_base_vectors,
        std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
    );

    const FloatSpace& getSpace() const;             ///< Distance metric and dimensionality
    const std::byte* getBaseVectors() const;        ///< Held base vectors (row c scored for candidate c)
    size_t getNumBaseVectors() const;               ///< Number of held base vectors
    int32_t getPo() const;                          ///< Rerank feature prefetch offset (0 = prefetch disabled)
    int32_t getPl() const;                          ///< Cache lines prefetched per feature vector (0 = whole vector)
    void setPrefetch(int32_t po, int32_t pl);       ///< Set both prefetch parameters at once (values <= 0 disable)

    /// Rerank candidate indices for one query, returning the top-k as a ResultSet (max-heap)
    ResultSet rerank(span<const byte> query, span<const uint32_t> candidate_indices, uint32_t k) const;

    /// Multi-threaded batch reranking of candidate indices for many queries
    vector<ResultSet> rerank(
        const void* queries, size_t num_queries, const uint32_t* base_candidates,
        size_t candidates_per_query, size_t k_top = 0, size_t num_threads = 0
    ) const;

    /// Auto-tune rerank prefetch parameters (po, pl) by timing reranking over sampled base vectors
    void optimize(size_t num_sample_queries = 300, size_t num_candidates = 150, uint32_t k = 100, size_t num_threads = 1);
};

/// Unified single-query and batch search result container with traversal telemetry.
struct SearchResults {
    size_t n_queries = 0;
    uint32_t k = 0;
    bool is_batch = false;
    bool with_distances = false;

    std::vector<uint32_t> indices;
    std::vector<float> distances;

    uint64_t graph_distance_count = 0;
    uint64_t rerank_distance_count = 0;
    uint64_t graph_hop_count = 0;

    size_t size() const noexcept;
    bool empty() const noexcept;
    bool has_distances() const noexcept;
    std::span<const uint32_t> get_indices(size_t q = 0) const noexcept;
    std::span<const float> get_distances(size_t q = 0) const noexcept;
};

/// Single-mode parameter calibration table mapping target recall@k levels along Mode::Ef (in [10, 500], step 1) or Mode::Eps (in [0.001, 0.20], resolution 0.0001), picking the smallest parameter achieving measured_recall >= target_recall.
class RecallTable {
public:
    enum class Mode : uint8_t { Ef, Eps };

    struct Entry {
        float target_recall = 0.0f;
        float eps_or_ef = 0.0f;
        float measured_recall = 0.0f;
    };

    static constexpr uint32_t kMinEf = 10;
    static constexpr uint32_t kMaxEf = 500;
    static constexpr uint32_t kEfResolution = 1;
    static constexpr float kMinEps = 0.001f;
    static constexpr float kMaxEps = 0.20f;
    static constexpr float kEpsResolution = 0.0001f;

    static RecallTable build(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        Mode mode,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        size_t threads = 0
    );
    static RecallTable build(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        Mode mode,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        size_t threads = 0
    );
    static std::vector<RecallTable> build_all(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        size_t threads = 0
    );
    static std::vector<RecallTable> build_all(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        size_t threads = 0
    );

    uint32_t k() const noexcept;
    size_t sample_count() const noexcept;
    Mode mode() const noexcept;
    float rerank_factor() const noexcept;
    bool empty() const noexcept;
    size_t size() const noexcept;
    std::optional<std::pair<float, float>> recall_bounds() const noexcept;
    const std::vector<Entry>& entries() const noexcept;
    std::vector<float> eps_or_ef_values() const;
    std::optional<float> param_for(float target_recall, bool clamp = false) const;
    std::string to_string() const;
};

/// Unified recall-latency Pareto frontier combining one or more RecallTables.
class ParetoLine {
public:
    struct Entry {
        float target_recall = 0.0f;
        float eps_or_ef = 0.0f;  // >= 1.0f -> ef, < 1.0f -> eps
        float rerank_factor = 1.0f;
        float measured_recall = 0.0f;
        double latency_us = 0.0;
        double qps = 0.0;
        double avg_distance_count = 0.0;
        double avg_rerank_distance_count = 0.0;

        bool is_ef() const noexcept;
        bool is_eps() const noexcept;
    };

    static constexpr size_t kDefaultRounds = 3;

    static ParetoLine build(
        const deglib::graph::InternalGraph& graph,
        std::span<const RecallTable> recall_tables,
        size_t sample_count = 0,
        size_t rounds = kDefaultRounds
    );
    static ParetoLine build(
        const deglib::graph::InternalGraph& graph,
        const RecallTable& recall_table,
        size_t sample_count = 0,
        size_t rounds = kDefaultRounds
    );
    static ParetoLine build(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        size_t threads = 0,
        size_t rounds = kDefaultRounds
    );
    static ParetoLine build(
        const deglib::graph::InternalGraph& graph,
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        size_t threads = 0,
        size_t rounds = kDefaultRounds
    );

    uint32_t k() const noexcept;
    size_t sample_count() const noexcept;
    bool empty() const noexcept;
    size_t size() const noexcept;
    std::optional<std::pair<float, float>> recall_bounds() const noexcept;
    const std::vector<Entry>& entries() const noexcept;
    const std::vector<RecallTable>& recall_tables() const noexcept;
    std::vector<float> eps_or_ef_values() const;
    std::vector<std::pair<float, float>> params() const;
    std::optional<std::pair<float, float>> param_for(float target_recall, bool clamp = false) const;
    std::string to_string() const;
};

/// High-performance Searcher coordinating query transformation, graph exploration, and reranking.
class Searcher {
public:
    static constexpr std::array<float, 4> kDefaultRerankFactors = {1.0f, 1.25f, 1.5f, 2.0f};

    bool normalize_query() const noexcept;
    void set_normalize_query(bool val) noexcept;

    void optimize(
        uint32_t n_clusters = 128,
        uint32_t n_iter = 15,
        size_t sample_size = 0,
        size_t threads = 0,
        bool use_routing = true
    );

    /// Calibrate a single-mode RecallTable for this Searcher pipeline
    RecallTable to_recall_table(
        uint32_t k,
        RecallTable::Mode mode,
        float rerank_factor = 1.0f,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        size_t threads = 0
    ) const;
    RecallTable to_recall_table(
        uint32_t k,
        RecallTable::Mode mode,
        std::span<const float> target_recalls,
        float rerank_factor = 1.0f,
        size_t sample_count = 300,
        size_t threads = 0
    ) const;

    /// Calibrate Ef and Eps RecallTables across rerank_factors (defaults to {1.0, 1.25, 1.5, 2.0} when reranking) in a shared single pass
    std::vector<RecallTable> to_recall_tables(
        uint32_t k,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        std::span<const float> rerank_factors = {},
        size_t threads = 0
    ) const;
    std::vector<RecallTable> to_recall_tables(
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        std::span<const float> rerank_factors = {},
        size_t threads = 0
    ) const;

    /// Build an end-to-end recall-latency ParetoLine across (ef, eps) x rerank_factors
    ParetoLine to_pareto_line(
        std::span<const RecallTable> recall_tables,
        size_t sample_count = 0,
        size_t rounds = ParetoLine::kDefaultRounds
    ) const;
    ParetoLine to_pareto_line(
        const RecallTable& recall_table,
        size_t sample_count = 0,
        size_t rounds = ParetoLine::kDefaultRounds
    ) const;
    ParetoLine to_pareto_line(
        uint32_t k,
        size_t sample_count = 300,
        std::pair<float, float> recall_range = {0.80f, 0.999f},
        float recall_step = 0.005f,
        std::span<const float> rerank_factors = {},
        size_t threads = 0,
        size_t rounds = ParetoLine::kDefaultRounds
    ) const;
    ParetoLine to_pareto_line(
        uint32_t k,
        std::span<const float> target_recalls,
        size_t sample_count = 300,
        std::span<const float> rerank_factors = {},
        size_t threads = 0,
        size_t rounds = ParetoLine::kDefaultRounds
    ) const;

    template <typename T>
    uint32_t search(
        std::span<const T> query,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const;

    template <typename T>
    SearchResults search(
        std::span<const T> query,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const;

    template <typename T>
    void search_batch(
        std::span<const T> queries,
        size_t n_queries,
        uint32_t k,
        std::span<uint32_t> out_indices,
        std::span<float> out_distances = {},
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const;

    template <typename T>
    SearchResults search_batch(
        std::span<const T> queries,
        size_t n_queries,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) const;
};

} // namespace deglib::search
```

---

## 5. Namespace: `deglib::optimization::`

Header: `<deglib/optimization.h>`

Graph topology pruning and 1D dataset presorting (FLAS).

```cpp
namespace deglib::optimization {

// --- Graph Pruning & Edge Optimization ---

/// Prune the worst (longest/highest-weight) neighbors per vertex, replacing them with self-loops
void prune_worst_edges(MutableGraph& graph, uint8_t prune_worst, size_t num_threads = 0);

/// Parallel removal of edges violating the Monotonic Relative Neighborhood Graph (MRNG) rule
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
    const float* data,
    size_t count,
    const deglib::distances::FloatSpace& space,
    float radius_decay = 0.9f,
    size_t num_threads = 0,
    function<bool(float)> callback = nullptr
);

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
    float graph_quality = -1.0f;        ///< Ratio of edges matching true NN ground truth (-1.0 if not computed)
    size_t memory_bytes;                ///< Estimated memory footprint in bytes
};

/// Perform comprehensive graph analysis (optionally with ground truth quality calculation)
GraphStats analyze_graph(
    const InternalGraph& graph,
    const uint32_t* base_top_data = nullptr,
    size_t base_top_rows = 0,
    size_t base_top_k = 0,
    size_t sample_size = 0,
    size_t num_threads = 0,
    size_t base_top_stride = 0
);

/// Calculate ratio of outgoing edges matching true nearest-neighbor ground truth
float calc_graph_quality(
    const InternalGraph& graph,
    const uint32_t* base_top_data,
    size_t base_top_rows,
    size_t base_top_k,
    size_t sample_size = 0,
    size_t num_threads = 0,
    size_t base_top_stride = 0
);

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

/// Count number of non-MRNG conform edges
uint32_t calc_non_mrng_edges(MutableGraph& graph);

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

/// Reusable scratch workspace and result container populated by InternalGraph traversals
struct SearchWorkspace {
    ResultSet results;                ///< Max-heap populated by eps search (`used_pool == false`)
    UncheckedSet next_vertices;       ///< Min-heap frontier
    deglib::search::ResultList pool;  ///< Sorted beam populated by ef search (`used_pool == true`)

    uint32_t graph_distance_count = 0;
    uint32_t graph_hop_count = 0;
    bool used_pool = false;

    size_t size() const noexcept;
    size_t result_count(size_t k) const noexcept;
    bool empty() const noexcept;
    uint32_t extract_internal_ids(uint32_t k, uint32_t* out_internal_ids) const noexcept;
    template <typename GraphT>
    uint32_t drain_into(const GraphT& graph, uint32_t k, uint32_t* out_indices, float* out_distances, bool unsorted, bool pad_missing = false) noexcept;
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

    template <typename T>
    void search(SearchWorkspace& ws, std::span<const T> query, uint32_t k, float eps = 0.0f, const Filter* filter = nullptr, uint32_t max_dc = 0) const;
    template <typename T>
    SearchWorkspace search(std::span<const T> query, uint32_t k, float eps = 0.0f, const Filter* filter = nullptr, uint32_t max_dc = 0) const;
    template <typename T>
    void search_ef(SearchWorkspace& ws, std::span<const T> query, uint32_t k, uint32_t ef, bool include_entry = true, const Filter* filter = nullptr, uint32_t max_dc = 0) const;
    template <typename T>
    SearchWorkspace search_ef(std::span<const T> query, uint32_t k, uint32_t ef, bool include_entry = true, const Filter* filter = nullptr, uint32_t max_dc = 0) const;
    void explore(SearchWorkspace& ws, uint32_t entry_index, uint32_t k, uint32_t max_dc = 0, float eps = 0.0f, bool include_entry = true, const Filter* filter = nullptr) const;
    SearchWorkspace explore(uint32_t entry_index, uint32_t k, uint32_t max_dc = 0, float eps = 0.0f, bool include_entry = true, const Filter* filter = nullptr) const;
    void explore_ef(SearchWorkspace& ws, uint32_t entry_index, uint32_t k, uint32_t ef, uint32_t max_dc = 0, bool include_entry = true, const Filter* filter = nullptr) const;
    SearchWorkspace explore_ef(uint32_t entry_index, uint32_t k, uint32_t ef, uint32_t max_dc = 0, bool include_entry = true, const Filter* filter = nullptr) const;
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
    static SizeBoundedGraph from_graph(const InternalGraph& graph, uint32_t new_max_size = 0, size_t num_threads = 0);
    static SizeBoundedGraph from_graph(const InternalGraph& graph, FloatSpace target_space, const void* custom_features, uint32_t new_max_size = 0, size_t num_threads = 0);
    static SizeBoundedGraph from_graph(
        const InternalGraph& graph,
        std::optional<FloatSpace> target_feature_space,
        std::span<const std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
        uint32_t new_max_size = 0,
        size_t num_threads = 0
    );
    // Loading is done via the free function deglib::graph::load_sizebounded_graph(const char* file_path, uint32_t new_max_size = 0)
};

/// Mutable graph with chunk-allocated dynamically growing memory
class DynamicGraph : public MutableGraph {
public:
    DynamicGraph(uint8_t edges_per_vertex, FloatSpace feature_space, uint32_t chunk_size = 1024);

    static DynamicGraph from_graph(const InternalGraph& graph, uint32_t chunk_size = 1024, size_t num_threads = 0);
    static DynamicGraph from_graph(const InternalGraph& graph, FloatSpace target_space, const void* custom_features, uint32_t chunk_size = 1024, size_t num_threads = 0);
    static DynamicGraph from_graph(
        const InternalGraph& graph,
        std::optional<FloatSpace> target_feature_space,
        std::span<const std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
        uint32_t chunk_size = 1024,
        size_t num_threads = 0
    );
    // Loading is done via the free function deglib::graph::load_dynamic_graph(const char* file_path, uint32_t chunk_size = 1024)
};

/// Compact immutable graph layout optimized for query serving
class ReadOnlyGraph : public InternalGraph {
public:
    ReadOnlyGraph(uint32_t count, uint8_t edges_per_vertex, const FloatSpace& space);

    static ReadOnlyGraph from_graph(const InternalGraph& input_graph);
    static ReadOnlyGraph from_graph(const InternalGraph& input_graph, FloatSpace target_space, const void* custom_features);
    static ReadOnlyGraph from_graph(
        const InternalGraph& input_graph,
        std::optional<FloatSpace> target_feature_space,
        std::span<const std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
        size_t num_threads = 0
    );
    static ReadOnlyGraph loadGraph(const char* path_graph);
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

/// Runtime AVX2-VNNI (256-bit) support check
bool has_avx2_vnni();

/// Runtime AVX-512 support check
bool has_avx512();

/// Runtime AVX-512 VNNI support check
bool has_avx512_vnni();

/// Runtime AVX-512 VPOPCNTDQ support check
bool has_avx512_vpopcntdq();

/// Runtime AVX-512 FP16 (native arithmetic) support check
bool has_avx512_fp16();

/// Runtime FP16 support check (F16C half-precision conversion)
bool has_fp16();

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

/// Semantic category of a FeatureTransformation stage
enum class TransformationType : uint8_t { Spatial, Rotation, Quantizer };

/// Base interface for vector feature transformations and quantization stages
class FeatureTransformation {
public:
    virtual ~FeatureTransformation() = default;

    virtual uint32_t tag() const noexcept = 0;
    virtual TransformationType type() const noexcept = 0;
    virtual std::string_view name() const noexcept = 0;
    virtual uint32_t input_dim() const noexcept = 0;
    virtual uint32_t output_dim() const noexcept = 0;
    virtual size_t input_byte_size() const noexcept = 0;
    virtual size_t output_byte_size() const noexcept = 0;

    virtual deglib::distances::Metric get_target_metric(deglib::distances::Metric input_metric) const;
    virtual bool is_fitted() const noexcept;

    virtual void fit(std::span<const float> data);
    virtual void fit(std::span<const uint16_t> data_fp16);

    virtual void transform_database(std::span<const float> in, std::span<std::byte> out) const = 0;
    virtual void transform_database(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const;

    virtual void transform_query(std::span<const float> in, std::span<std::byte> out) const = 0;
    virtual void transform_query(std::span<const uint16_t> in_fp16, std::span<std::byte> out) const;

    virtual void serialize(std::ostream& os) const = 0;
};

// --- Scalar Quantization Classes ---

/// Symmetric signed int8 scalar quantization
class ScalarQuantizerInt8 : public FeatureTransformation {
public:
    ScalarQuantizerInt8(uint32_t dim, float abs_max = 0.0f);
    float abs_max() const noexcept;
    float scale() const noexcept;
    void fit(std::span<const float> data, float quantile = 1.0f);
    void fit(std::span<const uint16_t> data_fp16, float quantile = 1.0f);
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<ScalarQuantizerInt8> deserialize(std::istream& is);
};

/// Per-dimension symmetric signed int8 scalar quantization
class ScalarQuantizerInt8PerDim : public FeatureTransformation {
public:
    explicit ScalarQuantizerInt8PerDim(uint32_t dim);
    std::span<const float> abs_maxs() const noexcept;
    std::span<const float> scales() const noexcept;
    void fit(std::span<const float> data, float quantile = 1.0f);
    void fit(std::span<const uint16_t> data_fp16, float quantile = 1.0f);
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<ScalarQuantizerInt8PerDim> deserialize(std::istream& is);
};

/// Asymmetric unsigned uint8 scalar quantization
class ScalarQuantizerUint8 : public FeatureTransformation {
public:
    ScalarQuantizerUint8(uint32_t dim, float min_val = 0.0f, float max_val = 0.0f);
    float min_val() const noexcept;
    float max_val() const noexcept;
    float scale() const noexcept;
    void fit(std::span<const float> data, float quantile = 1.0f);
    void fit(std::span<const uint16_t> data_fp16, float quantile = 1.0f);
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<ScalarQuantizerUint8> deserialize(std::istream& is);
};

/// Per-dimension asymmetric unsigned uint8 scalar quantization
class ScalarQuantizerUint8PerDim : public FeatureTransformation {
public:
    explicit ScalarQuantizerUint8PerDim(uint32_t dim);
    std::span<const float> mins() const noexcept;
    std::span<const float> maxs() const noexcept;
    std::span<const float> scales() const noexcept;
    void fit(std::span<const float> data, float quantile = 1.0f);
    void fit(std::span<const uint16_t> data_fp16, float quantile = 1.0f);
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<ScalarQuantizerUint8PerDim> deserialize(std::istream& is);
};

/// Asymmetric 4-bit unsigned (uint4, packed 2 per byte), signed (int4, packed 2 per byte),
/// and 7-bit unsigned (uint7 in [0, 127]) scalar quantizers are also provided:
/// - ScalarQuantizerUint4
/// - ScalarQuantizerInt4
/// - ScalarQuantizerUInt7

/// Half-precision IEEE 754 float16 cast transformation
class ScalarQuantizerFP16 : public FeatureTransformation {
public:
    explicit ScalarQuantizerFP16(uint32_t dim = 0);
    uint32_t dim() const noexcept;
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<ScalarQuantizerFP16> deserialize(std::istream& is);
};

/// L2 Normalization transformation (||x||_2 = 1.0)
class L2NormalizeTransform : public FeatureTransformation {
public:
    explicit L2NormalizeTransform(uint32_t dim = 0);
    uint32_t dim() const noexcept;
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<L2NormalizeTransform> deserialize(std::istream& is);
};

/// Fast Walsh-Hadamard orthogonal rotation (optional auto-enable via excess kurtosis)
class FastHadamardRotation : public FeatureTransformation {
public:
    FastHadamardRotation(uint32_t dim, uint64_t seed = 42, bool enabled = true);
    bool is_enabled() const noexcept;
    float excess_kurtosis() const noexcept;
    void fit_with_kurtosis(std::span<const float> data, float threshold = 0.8f);
    void fit(std::span<const float> data) override;
    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t num_threads) const;
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<FastHadamardRotation> deserialize(std::istream& is);
};

// --- Equi-Voronoi Polytopes Quantization (EVP) ---

class EvpQuantizer : public FeatureTransformation {
public:
    explicit EvpQuantizer(uint32_t non_zeros = 0, uint32_t dim = 0);
    uint32_t non_zeros() const noexcept;
    void transform_database(std::span<const float> in, std::span<std::byte> out, size_t num_threads) const;
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<EvpQuantizer> deserialize(std::istream& is);
};

std::shared_ptr<EvpQuantizer> make_evp_quantizer(std::span<const float> data, uint32_t dim, uint32_t non_zeros = 0);

// --- MIPS to L2 Space Transformation ---

class MipsTransform : public FeatureTransformation {
public:
    MipsTransform(uint32_t dim, float max_norm = 0.0f);
    float max_norm() const noexcept;
    void transform_database(std::span<const float> in, std::span<std::byte> out) const override;
    void transform_query(std::span<const float> in, std::span<std::byte> out) const override;
    void serialize(std::ostream& os) const override;
    static std::unique_ptr<MipsTransform> deserialize(std::istream& is);
};

float mips_l2_transform(std::span<const float> input, size_t count, size_t dim, std::span<float> output);
void mips_l2_transform_query(std::span<const float> input, size_t count, size_t dim, std::span<float> output);

// --- Workspace & Pipeline Execution Functions ---

/// Workspace buffer for transformation pipelines without heap allocations on search path
class TransformationWorkspace {
public:
    std::byte* get_buffer_a(size_t bytes);
    std::byte* get_buffer_b(size_t bytes);
    float* get_float_buffer(size_t count);
};

/// Calculate total output byte size of the final transformation stage
size_t get_output_byte_size(std::span<const std::shared_ptr<FeatureTransformation>> transformations) noexcept;
uint32_t get_input_dim(std::span<const std::shared_ptr<FeatureTransformation>> transformations) noexcept;
uint32_t get_output_dim(std::span<const std::shared_ptr<FeatureTransformation>> transformations) noexcept;
deglib::distances::Metric get_target_metric(std::span<const std::shared_ptr<FeatureTransformation>> transformations, deglib::distances::Metric input_metric);

/// Build a quantizer pipeline for any supported quantizer (Uint4, Int4, UInt7, Uint8, Int8, FP16, EVP),
/// with automatic FastHadamardRotation based on excess kurtosis for scalar quantizers
std::vector<std::shared_ptr<FeatureTransformation>> make_quantizer_pipeline(
    std::span<const float> data, uint32_t dim, deglib::distances::MetricDataType data_type,
    float quantile = -1.0f, bool use_rerank = false, size_t num_threads = 0, uint64_t rotation_seed = 42
);
std::vector<std::shared_ptr<FeatureTransformation>> make_quantizer_pipeline(
    std::span<const float> data, uint32_t dim, deglib::distances::Metric metric,
    float quantile = -1.0f, bool use_rerank = false, size_t num_threads = 0, uint64_t rotation_seed = 42
);

/// Apply sequential transformations to database vectors (FP32 or FP16)
void apply_on_database(std::span<const std::shared_ptr<FeatureTransformation>> transformations, std::span<const float> data, std::span<std::byte> out, size_t num_threads = 0);
void apply_on_database(std::span<const std::shared_ptr<FeatureTransformation>> transformations, std::span<const uint16_t> data_fp16, std::span<std::byte> out, size_t num_threads = 0);

/// Apply sequential transformations to query vectors
void apply_on_query(std::span<const std::shared_ptr<FeatureTransformation>> transformations, std::span<const float> query, std::span<std::byte> out);

/// Stream serialization of transformation pipelines
void serialize_transformations(const std::vector<std::shared_ptr<FeatureTransformation>>& transformations, std::ostream& os);
std::vector<std::shared_ptr<FeatureTransformation>> deserialize_transformations(std::istream& is);

} // namespace deglib::transformation
```

---

## 12. Namespace: `deglib::tuning::`

Header: `<deglib/tuning.h>` (included via `<deglib/deglib.h>`)

Pre-build geometric profiling and hyperparameter prediction from raw base vectors (no queries or ground truth required).

```cpp
namespace deglib::tuning {

inline constexpr size_t kDefaultSampleCount = 16000;
inline constexpr size_t kDefaultQueryCount  = 0;
inline constexpr size_t kDefaultRandomPairs = 65536;
inline constexpr uint32_t kDefaultSeed      = 7;

/// Base-only geometric profile of a vector dataset
struct DatasetProfile {
    uint32_t sample_count = 0;
    uint32_t dim = 0;
    float participation_ratio = 0.0f;  // (tr(S))^2 / (dim * ||S||_F^2)
    float rel_contrast_1 = 0.0f;       // median(d_rand) / Q1(d_1)
    float pp_iqr = 0.0f;               // IQR of duplicate-free Two-NN local ID 1 / log(d_2 / d_1)
    float q_mode_u7 = 0.0f;            // mean modal bin fraction under 7-bit min-max quantization (128 levels)
    float q_mode_u4 = 0.0f;            // mean modal bin fraction under 4-bit min-max quantization (16 levels)
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2;
};

/// Predicted optimization target and expected QPS advantage of LowLID over StreamingData
struct TargetPrediction {
    deglib::builder::OptimizationTarget target = deglib::builder::OptimizationTarget::StreamingData;
    float lowlid_advantage_pct = 0.0f;
    DatasetProfile profile{};
};

/// Predicted quantizer metric (Uint7_L2 vs Int8_InnerProduct) and expected QPS advantage
struct QuantizationPrediction {
    deglib::distances::Metric metric = deglib::distances::Metric::Uint7_L2;
    float uint7_advantage_pct = 0.0f;
    DatasetProfile profile{};
};

/// Linear z-score calibration constants predicting log2(QPS_LowLID / QPS_StreamingData)
struct TargetModelCalibration {
    double mu_pp_iqr, mu_q_mode_u7, mu_q_mode_u4;
    double sd_pp_iqr, sd_q_mode_u7, sd_q_mode_u4;
    double w_intercept, w_pp_iqr, w_q_mode_u7, w_q_mode_u4;
};

inline constexpr TargetModelCalibration kDefaultTargetCalibration{};

/// Linear z-score calibration constants predicting log2(QPS_Uint7_L2 / QPS_Int8_InnerProduct) on InnerProduct datasets
struct QuantizationModelCalibration {
    double mu_dim_over_participation, mu_rel_contrast;
    double sd_dim_over_participation, sd_rel_contrast;
    double w_intercept, w_dim_over_participation, w_rel_contrast;
};

inline constexpr QuantizationModelCalibration kDefaultQuantizationCalibration{};

/// Measure DatasetProfile from contiguous FP32 or FP16 (uint16_t) base vectors
DatasetProfile measure_dataset_profile(
    std::span<const float> data,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0
);

DatasetProfile measure_dataset_profile(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0
);

/// Predict OptimizationTarget::LowLID vs StreamingData from a DatasetProfile or directly from base vectors
TargetPrediction predict_optimization_target(
    const DatasetProfile& profile,
    const TargetModelCalibration& cal = kDefaultTargetCalibration
) noexcept;

TargetPrediction predict_optimization_target(
    std::span<const float> data,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const TargetModelCalibration& cal = kDefaultTargetCalibration
);

TargetPrediction predict_optimization_target(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const TargetModelCalibration& cal = kDefaultTargetCalibration
);

/// Predict Metric::Uint7_L2 vs Metric::Int8_InnerProduct from a DatasetProfile or directly from base vectors
/// (Non-InnerProduct / L2 datasets unconditionally select Metric::Uint7_L2)
QuantizationPrediction predict_quantization(
    const DatasetProfile& profile,
    const QuantizationModelCalibration& cal = kDefaultQuantizationCalibration
) noexcept;

QuantizationPrediction predict_quantization(
    std::span<const float> data,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const QuantizationModelCalibration& cal = kDefaultQuantizationCalibration
);

QuantizationPrediction predict_quantization(
    std::span<const uint16_t> data_fp16,
    uint32_t dim,
    deglib::distances::Metric metric = deglib::distances::Metric::FP32_L2,
    size_t sample_count = kDefaultSampleCount,
    size_t query_count = kDefaultQueryCount,
    uint32_t seed = kDefaultSeed,
    size_t num_threads = 0,
    const QuantizationModelCalibration& cal = kDefaultQuantizationCalibration
);

} // namespace deglib::tuning
```

