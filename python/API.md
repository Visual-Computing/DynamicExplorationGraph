# `deglib` Python API Reference

The `deglib` package separates its **High-Level API** (re-exported directly at top-level `deglib`) from **Low-Level / Expert APIs** (organized under submodules `deglib.cpu`, `deglib.distances`, `deglib.graph`, `deglib.builder`, `deglib.search`, `deglib.transformation`, `deglib.optimization`, `deglib.analysis`, `deglib.tuning`).

- **See also:** [Python Metric, Quantization & Pipeline Matrix](../docs/python_metric_quantization_matrix.md) for the full behavior table across `metric`, `compress`, `presort`, `reranker`, and `searcher`.

## API Layers: High-Level vs. Low-Level

| Layer | Namespace | Entry Points |
| :--- | :--- | :--- |
| **High-Level API** | `deglib.*` | `build`, `presort`, `compress`, `reranker`, `reranker_with_data`, `searcher`, `load_graph`, `DynamicExplorationGraph`, `Searcher`, `Reranker`, `SearchResults`, `Metric`, `OptimizationTarget` |
| **Low-Level Graph & I/O** | `deglib.graph.*` | `create_empty`, `create_dynamic_empty`, `create_random_graph`, `load_readonly_graph`, `load_dynamic_graph`, `load_mutable_graph` |
| **Low-Level Incremental Build** | `deglib.builder.*` | `GraphBuilder`, `BuilderStatus`, `ProgressCallback` |
| **Low-Level Search & Calibration** | `deglib.search.*` | `Filter`, `RecallTable`, `ParetoLine`, `rerank` |
| **Transformations & Quantizers** | `deglib.transformation.*` | `FeatureTransformation`, `TransformationType`, `make_quantizer_pipeline`, `make_mips_transform`, `make_l2_normalize_transform`, `make_fast_hadamard_rotation`, `make_evp_quantizer`, `make_scalar_quantizer_*`, `apply_on_database`, `apply_on_query` |
| **Optimization & Pruning** | `deglib.optimization.*` | `prune_non_mrng_edges`, `prune_worst_edges` |
| **Distances & CPU ISA** | `deglib.distances.*`, `deglib.cpu.*` | `floats_to_fp16`, `fp16_to_floats`, `InstructionSet`, `has_avx2`, `has_avx2_vnni`, `has_avx512`, `has_avx512_vnni`, `has_avx512_fp16`, `has_fp16` |
| **Analysis & Tuning** | `deglib.analysis.*`, `deglib.tuning.*` | `analyze_graph`, `GraphStats`, `calc_graph_quality`, `DatasetProfile`, `measure_dataset_profile`, `predict_optimization_target`, `predict_quantization` |

---

## Table of Contents

- [Quickstart & End-to-End Workflows](#quickstart--end-to-end-workflows)
- [1. High-Level Build, Compress & Search (`deglib.build`, `deglib.compress`, `deglib.reranker`, `deglib.searcher`)](#1-high-level-build-compress--search-deglibbuild-deglibcompress-deglibreranker-deglibsearcher)
- [2. Metrics & CPU Instruction Sets (`deglib.distances`, `deglib.cpu`)](#2-metrics--cpu-instruction-sets-deglibdistances-deglibcpu)
- [3. Graphs (`deglib.graph`)](#3-graphs-deglibgraph)
- [4. Incremental Graph Building (`deglib.builder`)](#4-incremental-graph-building-deglibbuilder)
- [5. High-Throughput Search, Filtering & Reranking (`deglib.search`)](#5-high-throughput-search-filtering--reranking-deglibsearch)
- [6. Feature Transformations & Quantization (`deglib.transformation`)](#6-feature-transformations--quantization-deglibtransformation)
- [7. Optimization & Pruning (`deglib.optimization`)](#7-optimization--pruning-degliboptimization)
- [8. Graph Analysis (`deglib.analysis`)](#8-graph-analysis-deglibanalysis)
- [9. Pre-Build Tuning & Prediction (`deglib.tuning`)](#9-pre-build-tuning--prediction-deglibtuning)

---

## Quickstart & End-to-End Workflows

### Simple FP32 Build & Search

```python
import numpy as np
import deglib

data = np.random.randn(10_000, 128).astype(np.float32)
query = np.random.randn(128).astype(np.float32)

# Build optimized graph (optimize=True by default)
graph = deglib.build(data, edges_per_vertex=32, show_progress=True)
searcher = deglib.searcher(graph)

# Fast index-only search (return_distances=False by default on Searcher.search)
res = searcher.search(query, k=10, eps_or_ef=0.1)
indices = res.indices

# Search returning both labels and exact distances plus traversal/rerank telemetry
res = searcher.search(query, k=10, eps_or_ef=0.1, return_distances=True)
indices, distances = res.indices, res.distances
```

### Quantized Search with Self-Extracted FP16 Reranking

```python
import deglib

# 1. Pre-sort with FLAS and build FP32 cosine graph
base_graph = deglib.build(
    data,
    metric="cosine",
    edges_per_vertex=30,
    presort=True,
    optimize=False,
    threads=8,
)

# 2. Extract FP16 reranker from FP32 graph, compress to 8-bit Searcher, and optimize routing
reranker = deglib.reranker(base_graph, compress="16bit", threads=8)
searcher = deglib.searcher(
    base_graph,
    compress="8bit",
    reranker=reranker,
    optimize=True,
    threads=8,
)

indices = searcher.search(queries, k=10, eps_or_ef=64, rerank_factor=2.0, unsorted=True).indices
```

### MIPS (`metric="ip"`) with Compression

```python
import deglib

# Automatically lifts into MIPS(d+1) for presort/build and swaps back to d-dim InnerProduct (or 16bit/8bit/4bit)
ro_graph = deglib.build(
    train_vectors,
    metric="ip",
    compress="16bit",
    edges_per_vertex=32,
    extend_k=64,
    presort=True,
    threads=8,
)
res = ro_graph.search(query, eps_or_ef=0.18, k=10)
indices, distances = res.indices, res.distances
```

---

## 1. High-Level Build, Compress & Search (`deglib.build`, `deglib.compress`, `deglib.reranker`, `deglib.searcher`)

### `deglib.build`

```python
deglib.build(
    data: np.ndarray,
    labels: Iterable[int] | np.ndarray | None = None,
    metric: Literal["l2", "euclidean", "cosine", "normalized", "ip"] | Metric | str = "l2",
    compress: Literal["32bit", "16bit", "8bit", "4bit", "2bit", "auto"] | Metric | str | None = None,
    edges_per_vertex: int = 32,
    optimization_target: OptimizationTarget | str = OptimizationTarget.Auto,
    extend_k: int = 0,
    extend_eps: float = 0.1,
    improve_k: int = 0,
    improve_eps: float = 0.001,
    max_path_length: int = 5,
    improve_tries: int = 0,
    seed: int = 7,
    presort: bool | np.ndarray = False,
    optimize: bool = True,
    threads: int = 0,
    show_progress: bool | None = None,
) -> DynamicExplorationGraph
```

Builds a graph from a 2D NumPy array:
- **`metric`**: High-level spatial modes (`"l2"` / `"euclidean"`, `"cosine"` / `"normalized"`, `"ip"`) or an explicit `Metric`. `"cosine"` attaches `L2NormalizeTransform` if vectors are not unit-norm; `"ip"` builds in `MipsTransform (d+1)` space and swaps stored features back to unnormalized `d`-dim `InnerProduct` (or `compress` target).
- **`compress`**: Optional post-build quantization (`"32bit"`, `"16bit"`, `"8bit"`, `"4bit"`, `"2bit"`, `"auto"`, or an explicit `Metric`). Preserves mutability via `deglib.compress` for `"l2"` / `"cosine"`, while `"ip"` `"16bit"`/`"8bit"` swaps to a compact `ReadOnlyGraph`.
- **`presort`**: `True` runs 1D FLAS pre-sorting via `deglib.presort` and reorders data + labels together; an `np.ndarray` uses precomputed sorted indices; `False` keeps insertion order.
- **`optimize`**: Runs `graph.optimize()` after building when `True` (default).
- **`threads`**: Worker thread count (`0` uses all CPU cores).

### `deglib.presort`, `deglib.compress`, `deglib.reranker`, `deglib.reranker_with_data`, `deglib.searcher`

- `deglib.presort(vectors, metric="l2", radius_decay=0.9, threads=0, callback=None, show_progress=False) -> np.ndarray`: Runs 1D FLAS pre-sorting (handles `"ip"` MIPS lifting automatically).
- `deglib.compress(graph, target, threads=0) -> DynamicExplorationGraph`: Compresses a float graph (`"32bit"`, `"16bit"`, `"8bit"`, `"4bit"`, `"2bit"`, `"auto"`, or `Metric`), preserving the input graph's backend type. For `"8bit"` / `"auto"`, selects `Uint7_L2` on L2 graphs, `deglib.tuning.predict_quantization` (`Uint7_L2` vs. `Int8_InnerProduct`) on unit-sphere `cosine`/`normalized` graphs, and `Int8_InnerProduct` on unnormalized `InnerProduct` graphs. For `"4bit"`, always selects `Uint4_L2` (auto-prepending `MipsTransform` on unnormalized `InnerProduct` graphs).
- `deglib.reranker(graph, compress="16bit", optimize=True, threads=0) -> Reranker`: Extracts a `Reranker` from a float graph's stored vectors.
- `deglib.reranker_with_data(graph, data, metric="16bit", optimize=True, threads=0) -> Reranker`: Builds a `Reranker` from external feature vectors aligned with `graph` labels.
- `deglib.searcher(graph, compress=None, reranker=None, optimize=True, threads=0) -> Searcher`: Wraps `graph` (optionally compressing first) in an optimized `ReadOnlyGraph`-backed `Searcher`.
  - **Memory & Ownership:** When `graph` is already a `ReadOnlyGraph` and `compress=None`, `searcher.graph` shares `graph` zero-copy (`searcher.graph is graph`). When `graph` is mutable or `compress` is set, a new `ReadOnlyGraph` is allocated for `searcher.graph` (drop `graph` if you no longer need the mutable/uncompressed copy in RAM).

---

## 2. Metrics & CPU Instruction Sets (`deglib.distances`, `deglib.cpu`)

`deglib.Metric` directly encapsulates distance function, vector dtype/byte-size calculations, distance evaluation helpers, and optional SIMD `InstructionSet` bindings. `FloatSpace` is internal and not exposed in the public API.

### `Metric` (`IntEnum`)

Supported metric members:
- **FP32**: `Metric.FP32_L2`, `Metric.FP32_InnerProduct` (`np.float32`)
- **FP16**: `Metric.FP16_L2`, `Metric.FP16_InnerProduct` (`np.float16` or raw `np.uint16` bit representation)
- **UINT8**: `Metric.Uint8_L2`, `Metric.Uint8_InnerProduct` (`np.uint8`)
- **INT8**: `Metric.Int8_L2`, `Metric.Int8_InnerProduct` (`np.int8`)
- **UINT7**: `Metric.UInt7_L2`, `Metric.UInt7_InnerProduct` (`np.int8`, values in `[0, 127]`)
- **INT4**: `Metric.Int4_L2`, `Metric.Int4_InnerProduct` (`np.uint8`, packed 2 values/byte)
- **UINT4**: `Metric.Uint4_L2`, `Metric.Uint4_InnerProduct` (`np.uint8`, packed 2 values/byte)
- **EVP**: `Metric.EVP_InnerProduct` (`np.uint8`, packed bitmasks of size `2 * ((dim + 7) // 8)`)

Methods on `Metric` (usable both on `Metric.*` enum members and bound metrics returned by `with_instruction` / `graph.get_metric()`):

- `Metric.resolve(metric: Metric | str | int, distance_kind: str | None = None, instruction: InstructionSet | str | None = None) -> Metric`
- `metric.with_instruction(instruction: InstructionSet | str) -> Metric`: Returns a metric bound to a specific `InstructionSet` (compares `==` to the base `Metric` enum member).
- `metric.get_instruction(dim: int = 16) -> InstructionSet`: Returns the bound or auto-resolved CPU instruction set.
- `metric.get_dtype() -> np.dtype`: Canonical storage dtype for features.
- `metric.get_accepted_dtypes() -> tuple[np.dtype, ...]`: Accepted array dtypes (e.g., `(np.uint16, np.float16)` for FP16).
- `metric.get_data_size(dim: int) -> int`: Stored byte size per database vector of dimension `dim`.
- `metric.get_query_data_size(dim: int) -> int`: Query byte size per vector of dimension `dim`.
- `metric.compute_distance(vec1: np.ndarray, vec2: np.ndarray) -> float`: Compute distance between two 1D vectors.
- `metric.compute_distances(query: np.ndarray, targets: np.ndarray, threads: int = 0) -> np.ndarray`: Compute distances between 1D/2D `query` and 2D `targets` (returns shape `(n_queries, n_targets)`).

### `InstructionSet` (`IntEnum`) & CPU Helpers (`deglib.cpu`)

- Members: `InstructionSet.Auto`, `InstructionSet.Scalar`, `InstructionSet.AVX2`, `InstructionSet.AVX2_VNNI`, `InstructionSet.AVX512`, `InstructionSet.AVX512_VNNI`, `InstructionSet.AVX512_FP16`.
- `InstructionSet.resolve(instruction: InstructionSet | str | int) -> InstructionSet`
- Capability checks: `has_avx2()`, `has_avx2_vnni()`, `has_avx512()`, `has_avx512_vnni()`, `has_avx512_fp16()`, `has_fp16()`

### FP16 Conversion Helpers (`deglib.distances`)

- `floats_to_fp16(floats: np.ndarray) -> np.ndarray`: Converts `float32` array to IEEE-754 half-precision `uint16` buffer.
- `fp16_to_floats(fp16_vals: np.ndarray) -> np.ndarray`: Converts `uint16` FP16 buffer back to `float32`.

---

## 3. Graphs (`deglib.graph`)

All graphs (fixed-capacity mutable `SizeBoundedGraph`, chunked `DynamicGraph`, and compact immutable `ReadOnlyGraph`) are accessed through `DynamicExplorationGraph`.

### Factory & Loader Functions

- `create_empty(capacity: int, dim: int, edges_per_vertex: int = 32, metric: Metric | str = Metric.FP32_L2, transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None) -> DynamicExplorationGraph`
- `create_dynamic_empty(dim: int, edges_per_vertex: int = 32, metric: Metric | str = Metric.FP32_L2, chunk_size: int = 1024, transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None) -> DynamicExplorationGraph`
- `create_random_graph(features: np.ndarray, edges_per_vertex: int = 32, metric: Metric | str = Metric.FP32_L2, seed: int = 7, optimize: bool = True, transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None) -> DynamicExplorationGraph`
- `load_graph(path: Path | str) -> DynamicExplorationGraph`: High-level loader (loads as read-only).
- `load_readonly_graph(path: Path | str) -> DynamicExplorationGraph`
- `load_mutable_graph(path: Path | str, capacity: int = 0) -> DynamicExplorationGraph`
- `load_dynamic_graph(path: Path | str, chunk_size: int = 1024) -> DynamicExplorationGraph`

### `DynamicExplorationGraph` Methods

#### Introspection & Topology
- `size() -> int`
- `get_dim() -> int`
- `get_metric() -> Metric`: Returns metric bound to the graph's `InstructionSet` (use `==` when comparing to `Metric.*`).
- `get_instruction() -> InstructionSet`
- `get_data_size() -> int`
- `get_edges_per_vertex() -> int`
- `is_mutable() -> bool`
- `has_vertex(external_label: int) -> bool`
- `get_labels() -> np.ndarray`: External labels ordered by internal graph slot (`0 .. size()-1`).
- `get_neighbors(external_label: int | np.ndarray | Sequence[int], threads: int = 1) -> list[int] | np.ndarray`
- `get_po() -> int`, `get_pl() -> int`, `get_nl() -> int`
- `has_path(from_external_labels: Sequence[int], to_external_label: int, eps_or_ef: float = 0.0, max_distance_computation_count: int = 0) -> list[ObjectDistance]`
- `save_graph(path: Path | str) -> bool` (requires mutable graph)
- `optimize(sample_count: int = 300, median_subset: int = 4096) -> None`
- `to_recall_table(k: int, mode: Literal["ef", "eps"], sample_count: int = 300, recall_range: tuple[float, float] | None = None, recall_step: float | None = None, target_recalls: Sequence[float] | np.ndarray | None = None, threads: int = 0) -> RecallTable`: Calibrates a single-mode recall@k parameter curve (`ef ∈ [10, 500]` step `1`, or `eps ∈ [0.001, 0.20]` resolution `0.0001`), picking the smallest parameter achieving `measured_recall >= target_recall` using self-sampled queries, exact ground truth, and an approximated brute-force time guard (defaults `recall_range=(0.80, 0.999)`, `recall_step=0.005`).
- `to_recall_tables(k: int, sample_count: int = 300, recall_range: tuple[float, float] | None = None, recall_step: float | None = None, target_recalls: Sequence[float] | np.ndarray | None = None, threads: int = 0) -> list[RecallTable]`: Calibrates both `"ef"` and `"eps"` `RecallTable` curves in one pass sharing exact ground truth.
- `to_pareto_line(k: int | None = None, recall_tables: RecallTable | Sequence[RecallTable] | None = None, sample_count: int | None = None, recall_range: tuple[float, float] | None = None, recall_step: float | None = None, target_recalls: Sequence[float] | np.ndarray | None = None, threads: int = 0, rounds: int = 3) -> ParetoLine`: Unifies measured operating points across `recall_tables` (defaulting to `to_recall_tables(k, ...)`) by single-threaded median latency in `(measured_recall, latency_us)` space, prunes Pareto-dominated points, and exposes `.recall_tables`.

#### Graph Conversion & Searcher Construction
- `to_readonly() -> DynamicExplorationGraph`: Pure backend copy to `ReadOnlyGraph` (preserves features, metric, and `transformations_`).
- `to_mutable(max_size: int = 0, threads: int = 0) -> DynamicExplorationGraph`: Pure backend copy to `SizeBoundedGraph` (preserves features, metric, and `transformations_`; uses `threads` for edge-weight recomputation when converting from a `ReadOnlyGraph`).
- `to_dynamic(chunk_size: int = 1024, threads: int = 0) -> DynamicExplorationGraph`: Pure backend copy to `DynamicGraph` (preserves features, metric, and `transformations_`; uses `threads` for edge-weight recomputation when converting from a `ReadOnlyGraph`).
- `with_metric(metric: Metric | str, threads: int = 0) -> DynamicExplorationGraph`: Re-tags or auto-quantizes stored floating-point features to `metric`, preserving the current backend (`ReadOnlyGraph`, `SizeBoundedGraph`, or `DynamicGraph`). When moving from float to a quantized metric, appends the calibrated pipeline to `transformations_`; when changing between same-dtype metrics (e.g. `FP32_L2` -> `FP32_InnerProduct`), keeps features unchanged and recomputes edge weights if mutable.
- `with_transformations(transformations: FeatureTransformation | Sequence[FeatureTransformation], metric: Metric | str | None = None, threads: int = 0) -> DynamicExplorationGraph`: Self-transforms stored floating-point features (`FP32`/`FP16`) using explicit fitted `transformations` and appends them to `transformations_`, preserving the current backend.
- `with_features(features: np.ndarray, metric: Metric | str | None = None, order: Literal["label", "internal"] = "label", transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None, threads: int = 0) -> DynamicExplorationGraph`: Replaces graph features with an external array (auto-calibrating/quantizing FP32/FP16 arrays when needed and `transformations=None`), resets `transformations_` to the new pipeline, and preserves the current backend. `order` defaults to `"label"` (rows aligned with external labels); pass `"internal"` when rows already follow `graph.get_labels()`.
- `to_reranker(metric: Metric | str, transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None, optimize: bool = True, threads: int = 0) -> Reranker`
  - Extracts and quantizes base vectors directly from an `FP32`/`FP16` graph, chaining `graph.transformations_ + transformations`. For external datasets, construct `deglib.Reranker(metric, base_vectors, transformations=..., row_indices=..., optimize=..., threads=...)` directly.
- `to_searcher(metric: Metric | str | None = None, reranker: Reranker | None = None, normalize_query: bool = False, optimize: bool = True, threads: int = 0) -> Searcher`
  - Converts the graph to a `ReadOnlyGraph` in a single pass (applying/quantizing `metric` directly when `metric` differs from the current metric) and wraps it in a `Searcher`. For custom clustering parameters, pass `optimize=False` and call `searcher.optimize(...)`.

#### Direct Graph Search & Exploration
- `search(query: np.ndarray, eps_or_ef: float = 0.0, k: int = 10, filter_labels: Any = None, max_distance_computation_count: int = 0, threads: int = 0, return_distances: bool = True, unsorted: bool = False) -> SearchResults`
- `explore(entry_external_label: int | np.ndarray | Sequence[int], k: int, max_distance_computation_count: int = 0, eps_or_ef: float = 0.0, include_entry: bool = True, threads: int = 1, filter_labels: Any = None, return_distances: bool = True, unsorted: bool = False) -> SearchResults`

---

## 4. Incremental Graph Building (`deglib.builder`)

### `OptimizationTarget` (`Enum`)
- `OptimizationTarget.StreamingData`
- `OptimizationTarget.LowLID`
- `OptimizationTarget.HighLID`
- `OptimizationTarget.resolve(target: OptimizationTarget | str | int) -> OptimizationTarget`

### `GraphBuilder`

```python
GraphBuilder(
    graph: DynamicExplorationGraph,
    seed: int = 7,
    optimization_target: OptimizationTarget | str = OptimizationTarget.LowLID,
    extend_k: int = 0,
    extend_eps: float = 0.1,
    improve_k: int = 0,
    improve_eps: float = 0.001,
    max_path_length: int = 5,
    improve_tries: int = 0,
    threads: int = 0,
)
```

Methods:
- `add_entry(external_label: int | Iterable[int] | np.ndarray, feature: np.ndarray) -> None`
- `remove_entry(external_label: int | Iterable[int] | np.ndarray) -> None`
- `get_num_new_entries() -> int`
- `get_num_remove_entries() -> int`
- `set_threads(threads: int) -> None` (`0` selects all available CPU cores)
- `set_batch_size(tasks_per_batch: int, task_size: int) -> None`
- `get_batch_size() -> int`
- `build(callback: Callable[[BuilderStatus], None] | None = None, show_progress: bool = False, infinite: bool = False) -> BuilderStatus`
- `stop() -> None`

### `BuilderStatus` & `ProgressCallback`
- `BuilderStatus` attributes: `step`, `added`, `deleted`, `tries`, `improved`, `quality_changed`, `total_added_ids`, `total_deleted_ids`, `step_added_ids`, `step_deleted_ids`.
- `ProgressCallback(num_new_entries: int, num_remove_entries: int, bar_length: int = 60, min_print_interval: float = 0.1)`

---

## 5. High-Throughput Search, Filtering & Reranking (`deglib.search`)

### `SearchResults`

Returned by `DynamicExplorationGraph.search()`, `DynamicExplorationGraph.explore()`, and `Searcher.search()`. Holds C++ result memory exposed to Python as zero-copy NumPy views:

- `indices: np.ndarray` (`uint32`, 1D `(found,)` / `(k,)` or 2D `(N, k)` for batches)
- `distances: np.ndarray | None` (`float32`, same shape when `return_distances=True`, else `None`)
- `graph_distance_count: int` — total graph vertex distance evaluations performed
- `rerank_distance_count: int` — total exact reranker distance evaluations performed (`0` unless `rerank_factor > 1.0` with a reranker)
- `graph_hop_count: int` — total graph vertex expansions (hops) performed
- `__len__() -> int` — number of results (single query) or query count (batch)

### `Searcher`

```python
Searcher(
    graph: DynamicExplorationGraph,
    reranker: Reranker | None = None,
    normalize_query: bool = False,
    optimize: bool = True,
    threads: int = 0,
)
```

`Searcher` attributes & methods:
- `searcher.graph -> DynamicExplorationGraph` (the backing read-only graph)
- `searcher.normalize_query -> bool` (get/set query L2 normalization)
- `searcher.optimize(n_clusters: int = 256, n_iter: int = 20, sample_size: int = 0, threads: int = 0, use_routing: bool = True) -> None`
- `searcher.search(query: np.ndarray, k: int = 10, eps_or_ef: float = 0.1, rerank_factor: float = 1.0, threads: int = 1, return_distances: bool = False, unsorted: bool = False, max_distance_computation_count: int = 0) -> SearchResults`
  - `eps_or_ef < 1.0`: relative distance margin (`eps`); `eps_or_ef >= 1.0`: fixed beam size (`ef`).
  - Defaults `return_distances=False, unsorted=False` (populates 1D `(found,)` or 2D `(N, k)` `.indices`, `.distances=None` unless `return_distances=True`, plus traversal/rerank telemetry).
- `searcher.to_recall_table(k: int, mode: Literal["ef", "eps"], rerank_factor: float = 1.0, sample_count: int = 300, recall_range: tuple[float, float] | None = None, recall_step: float | None = None, target_recalls: Sequence[float] | np.ndarray | None = None, threads: int = 0) -> RecallTable`
- `searcher.to_recall_tables(k: int, sample_count: int = 300, recall_range: tuple[float, float] | None = None, recall_step: float | None = None, target_recalls: Sequence[float] | np.ndarray | None = None, rerank_factors: Sequence[float] | np.ndarray | None = None, threads: int = 0) -> list[RecallTable]`: Calibrates `("ef", "eps") x rerank_factors` in a shared single pass (`rerank_factors` defaults to `[1.0, 1.25, 1.5, 2.0]` when a reranker is attached, else `[1.0]`).
- `searcher.to_pareto_line(k: int | None = None, recall_tables: RecallTable | Sequence[RecallTable] | None = None, sample_count: int | None = None, recall_range: tuple[float, float] | None = None, recall_step: float | None = None, target_recalls: Sequence[float] | np.ndarray | None = None, rerank_factors: Sequence[float] | np.ndarray | None = None, threads: int = 0, rounds: int = 3) -> ParetoLine`

### `Filter`

```python
Filter(valid_labels: np.ndarray, max_value: int = -1, max_label_count: int = -1)
```

### `RecallTable`

- `RecallTable` (returned by `graph.to_recall_table(...)` / `to_recall_tables(...)` and `searcher.to_recall_table(...)` / `to_recall_tables(...)`):
  - Properties: `.k -> int`, `.sample_count -> int`, `.is_ef -> bool`, `.is_eps -> bool`, `.rerank_factor -> float`, `.recall_bounds -> tuple[float, float] | None`, `.entries -> list[tuple[float, float, float]]` (`(target_recall, eps_or_ef, measured_recall)`)
  - Methods: `.eps_or_ef_values() -> list[float]`, `.param_for(target_recall: float, clamp: bool = False) -> float | None`, `len(table)`, `str(table)`

### `ParetoLine`

- `ParetoLine` (returned by `graph.to_pareto_line(...)` and `searcher.to_pareto_line(...)`):
  - Properties: `.k -> int`, `.sample_count -> int`, `.recall_bounds -> tuple[float, float] | None`, `.recall_tables -> list[RecallTable]`, `.entries -> list[tuple[float, float, float, float, float, float]]` (`(target_recall, eps_or_ef, rerank_factor, measured_recall, latency_us, qps)`)
  - Methods: `.eps_or_ef_values() -> list[float]`, `.params() -> list[tuple[float, float]]`, `.param_for(target_recall: float, clamp: bool = False) -> tuple[float, float] | None` (nearest frontier entry by `measured_recall`, no interpolation), `len(line)`, `str(line)`

### `Reranker` & `rerank`

```python
Reranker(
    metric: Metric | str,
    base_vectors: np.ndarray,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
    optimize: bool = False,
    threads: int = 0,
)
```

Methods:
- `get_num_base_vectors() -> int`, `get_po() -> int`, `get_pl() -> int`
- `optimize(sample_count: int = 300, num_candidates: int = 150) -> None`
- `rerank(queries: np.ndarray, candidate_indices: np.ndarray, k_top: int = 0, threads: int = 0, return_distances: bool = False, unsorted: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]`

Functional helper:
```python
rerank(
    metric: Metric | str,
    queries: np.ndarray,
    candidate_indices: np.ndarray,
    base_vectors: np.ndarray | None = None,
    k_top: int = 0,
    threads: int = 0,
    return_distances: bool = False,
    unsorted: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]
```

*(Note: passing `float32` `base_vectors` to `Reranker` / `rerank` with `Metric.FP16_*` automatically calibrates a `ScalarQuantizerFP16` and applies it to queries during `rerank`.)*

---

## 6. Feature Transformations & Quantization (`deglib.transformation`)

All transformations are created pre-fitted and validated via `make_*` factories and share the `FeatureTransformation` interface (`name`, `type` (`TransformationType`: `Spatial`, `Rotation`, `Quantizer`), `input_dim`, `output_dim`, `output_dtype`, `output_byte_size`, `is_fitted`, `transform_database(vectors, threads=0)`, `transform_query(queries)`, `serialize()`). Raw unfitted quantizer constructors and `mips_l2_transform(_query)` are not part of the API.

- `make_mips_transform(vectors: np.ndarray, max_norm: float = 0.0) -> FeatureTransformation` (`MipsTransform` exposing `.transform_database(...)` and `.transform_query(...)`)
- `make_l2_normalize_transform(vectors_or_dim: np.ndarray | int) -> FeatureTransformation`
- `make_fast_hadamard_rotation(vectors: np.ndarray, seed: int = 7, rounds: int = 0, auto_detect: bool = True) -> FeatureTransformation`
- `make_quantizer_pipeline(vectors: np.ndarray, quant: Metric | str, quantile: float = -1.0, use_rerank: bool = False, threads: int = 0, seed: int = 7) -> list[FeatureTransformation]`
- `make_evp_quantizer(vectors: np.ndarray, non_zeros: int = 0, threads: int = 0) -> FeatureTransformation` (`non_zeros=0` auto-calibrates sparsity)
- `make_scalar_quantizer_int8(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation`
- `make_scalar_quantizer_uint8(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation`
- `make_scalar_quantizer_uint7(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation`
- `make_scalar_quantizer_uint4(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation`
- `make_scalar_quantizer_int4(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation`
- `make_scalar_quantizer_fp16(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation`
- `apply_on_database(transformations: FeatureTransformation | Sequence[FeatureTransformation], vectors: np.ndarray, threads: int = 0) -> np.ndarray`
- `apply_on_query(transformations: FeatureTransformation | Sequence[FeatureTransformation], queries: np.ndarray) -> np.ndarray`

---

## 7. Optimization & Pruning (`deglib.optimization`)

- `presort(vectors: np.ndarray, metric: Metric | str = Metric.FP32_L2, radius_decay: float = 0.9, threads: int = 0, callback: Callable[[float], bool | None] | None = None, show_progress: bool = False) -> np.ndarray`: Runs 1D FLAS spatial sorting; returns `uint32` permutation indices. (`callback(progress)` cancels sorting only if it explicitly returns `True`.)
- `prune_non_mrng_edges(graph: DynamicExplorationGraph, threads: int = 0) -> int`
- `prune_worst_edges(graph: DynamicExplorationGraph, prune_worst: int, threads: int = 0) -> None`

---

## 8. Graph Analysis (`deglib.analysis`)

- `analyze_graph(graph: Any, base_top: Any | None = None, sample_size: int = 0, threads: int = 0, print_stats: bool = True) -> GraphStats`
- `calc_graph_quality(graph: Any, base_top: Any, sample_size: int = 0, threads: int = 0) -> float`
- `GraphStats`: summary metrics returned by `analyze_graph`.

---

## 9. Pre-Build Tuning & Prediction (`deglib.tuning`)

Base-only geometric profiling and hyperparameter prediction from `float32`/`float16` feature vectors (no queries or ground truth required). Also re-exported at top level (`deglib.*`).

- `DatasetProfile(sample_count=0, dim=0, participation_ratio=0.0, d1_cv=0.0, rel_contrast_1=0.0, metric=Metric.FP32_L2)`: frozen dataclass storing measured dataset geometry.
- `TargetPrediction(target: OptimizationTarget, lowlid_advantage_pct: float, profile: DatasetProfile)`: predicted graph construction optimization target (`LowLID` vs `StreamingData`).
- `QuantizationPrediction(metric: Metric, uint7_advantage_pct: float, profile: DatasetProfile)`: predicted quantizer metric (`Metric.Uint7_L2` vs `Metric.Int8_InnerProduct`).
- `measure_dataset_profile(data: np.ndarray, metric: Metric | str | int = Metric.FP32_L2, sample_count: int = 16000, query_count: int = 0, seed: int = 7, threads: int = 0) -> DatasetProfile`
- `predict_optimization_target(data_or_profile: np.ndarray | DatasetProfile, metric: Metric | str | int = Metric.FP32_L2, sample_count: int = 16000, query_count: int = 0, seed: int = 7, threads: int = 0) -> TargetPrediction`
- `predict_quantization(data_or_profile: np.ndarray | DatasetProfile, metric: Metric | str | int = Metric.FP32_L2, sample_count: int = 16000, query_count: int = 0, seed: int = 7, threads: int = 0) -> QuantizationPrediction`: predicts whether `Metric.Uint7_L2` or `Metric.Int8_InnerProduct` achieves higher QPS (L2/Euclidean datasets unconditionally select `Metric.Uint7_L2`).
