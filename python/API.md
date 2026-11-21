# `deglib` Python API Reference

Complete reference for the `deglib` Python package. Everything documented below is exported from top-level `deglib` as well as its submodules (`deglib.cpu`, `deglib.distances`, `deglib.graph`, `deglib.builder`, `deglib.search`, `deglib.transformation`, `deglib.optimization`, `deglib.analysis`).

---

## Table of Contents

- [Quickstart & End-to-End Workflows](#quickstart--end-to-end-workflows)
- [1. Top-Level Build & Search (`deglib.build`, `graph.to_searcher`)](#1-top-level-build--search-deglibbuild-graphto_searcher)
- [2. Metrics & CPU Instruction Sets (`deglib.distances`, `deglib.cpu`)](#2-metrics--cpu-instruction-sets-deglibdistances-deglibcpu)
- [3. Graphs (`deglib.graph`)](#3-graphs-deglibgraph)
- [4. Incremental Graph Building (`deglib.builder`)](#4-incremental-graph-building-deglibbuilder)
- [5. High-Throughput Search, Filtering & Reranking (`deglib.search`)](#5-high-throughput-search-filtering--reranking-deglibsearch)
- [6. Feature Transformations & Quantization (`deglib.transformation`)](#6-feature-transformations--quantization-deglibtransformation)
- [7. Optimization & Pruning (`deglib.optimization`)](#7-optimization--pruning-degliboptimization)
- [8. Graph Analysis (`deglib.analysis`)](#8-graph-analysis-deglibanalysis)

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
searcher = graph.to_searcher()

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

# 1. Pre-sort with FLAS and build FP32 graph
base_graph = deglib.build(
    data,
    metric=deglib.Metric.FP32_InnerProduct,
    edges_per_vertex=30,
    presort=True,
    optimize=False,
    threads=8,
)

# 2. Convert into Int8 ReadOnlyGraph and auto-extract/quantize FP16 rerank data
searcher = base_graph.to_searcher(
    metric=deglib.Metric.Int8_InnerProduct,
    rerank_metric=deglib.Metric.FP16_InnerProduct,  # opt-in reranking; None disables reranking
    normalize_query=True,
    optimize=True,
    use_routing=True,
    n_clusters=256,
    threads=8,
)

indices = searcher.search(queries, k=10, eps_or_ef=64, rerank_factor=2.0, unsorted=True).indices
```

### MIPS via $(d + 1)$-Dimensional $L_2$ Lifting & FP16 Swap

```python
import deglib

mips = deglib.make_mips_transform(train_vectors)
lifted = mips.transform_database(train_vectors)

graph = deglib.build(
    lifted,
    edges_per_vertex=32,
    extend_k=64,
    metric=deglib.Metric.FP32_L2.with_instruction("AVX512"),
    presort=True,
    optimize=False,
    threads=8,
)

# Swap features back to original d-dim FP16 InnerProduct while preserving graph labels & edges
ro_graph = graph.to_readonly(
    deglib.Metric.FP16_InnerProduct.with_instruction("AVX512"),
    custom_features=train_vectors,
    threads=8,
)
res = ro_graph.search(query, eps_or_ef=0.18, k=10)
indices, distances = res.indices, res.distances
```

---

## 1. Top-Level Build & Search (`deglib.build`, `graph.to_searcher`)

### `deglib.build`

```python
deglib.build(
    data: np.ndarray,
    labels: Iterable[int] | np.ndarray | None = None,
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    optimization_target: OptimizationTarget | str = OptimizationTarget.LowLID,
    presort: bool | np.ndarray = False,
    presort_decay: float = 0.9,
    optimize: bool = True,
    readonly: bool = False,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
    capacity: int = -1,
    extend_k: int = 0,
    extend_eps: float = 0.1,
    improve_k: int = 0,
    improve_eps: float = 0.001,
    max_path_length: int = 5,
    improve_tries: int = 0,
    seed: int = 7,
    threads: int = 0,
    callback: Callable[[BuilderStatus], None] | None = None,
    show_progress: bool = False,
) -> DynamicExplorationGraph
```

Builds a graph from a 2D NumPy array:
- When `data` is `float32`/`float16` and `metric` is quantized (`Int8_*`, `Uint8_*`, `Int4_*`, `Uint4_*`, `UInt7_*`, `EVP_InnerProduct`, or `FP16_*` from `float32`), `build` automatically constructs a calibrated transformation pipeline (`make_quantizer_pipeline`) unless explicit `transformations` are provided.
- **`presort`**: `True` runs 1D FLAS pre-sorting via `deglib.presort` and reorders data + labels together; an `np.ndarray` uses precomputed sorted indices; `False` keeps insertion order.
- **`optimize`**: Runs `graph.optimize()` after building when `True` (default).
- **`readonly`**: Returns a compact `ReadOnlyGraph` when `True`, or a mutable `SizeBoundedGraph` when `False` (default).
- **`threads`**: Worker thread count (`0` uses all CPU cores).

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
- `load_graph(path: Path | str) -> DynamicExplorationGraph`: Auto-detects format (loads as read-only).
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
- `get_neighbors(external_label: int) -> list[int]`
- `get_neighbors_batch(external_labels: np.ndarray | Sequence[int], threads: int = 1) -> np.ndarray`
- `get_po() -> int`, `get_pl() -> int`, `get_nl() -> int`
- `has_path(from_external_labels: Sequence[int], to_external_label: int, eps_or_ef: float = 0.0, max_distance_computation_count: int = 0) -> list[ObjectDistance]`
- `save_graph(path: Path | str) -> bool` (requires mutable graph)
- `optimize(sample_count: int = 300, median_subset: int = 4096) -> None`

#### Graph Conversion & Searcher Construction
- `to_readonly(metric: Metric | str | None = None, custom_features: np.ndarray | None = None, custom_features_order: Literal["label", "internal"] = "label", transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None, use_rerank: bool = False, optimize: bool = True, threads: int = 0) -> DynamicExplorationGraph`
- `to_mutable(metric: Metric | str | None = None, custom_features: np.ndarray | None = None, custom_features_order: Literal["label", "internal"] = "label", capacity: int = 0, transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None, threads: int = 0) -> DynamicExplorationGraph`
- `to_dynamic(metric: Metric | str | None = None, custom_features: np.ndarray | None = None, custom_features_order: Literal["label", "internal"] = "label", chunk_size: int = 1024, transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None, threads: int = 0) -> DynamicExplorationGraph`
- `to_reranker(metric: Metric | str, data: np.ndarray | None = None, order: Literal["label", "internal"] = "label", transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None, optimize: bool = True, threads: int = 0) -> Reranker`
  - Creates a standalone `Reranker` aligned with this graph's internal vertex order (`data=None` extracts and quantizes vectors directly from an `FP32`/`FP16` graph), allowing one `Reranker` to be shared across multiple quantized `Searcher` instances without duplicating base vector memory.
- `to_searcher(metric: Metric | str | None = None, reranker: Reranker | None = None, rerank_metric: Metric | str | None = None, rerank_data: np.ndarray | None = None, rerank_order: Literal["label", "internal"] = "label", rerank_transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None, normalize_query: bool = False, optimize: bool = True, use_routing: bool = True, n_clusters: int = 256, n_iter: int = 20, sample_size: int = 0, threads: int = 0) -> Searcher`
  - **Self-quantization**: When changing `metric` on an `FP32`/`FP16` graph without `custom_features` or `transformations`, features are quantized directly from stored floating-point graph data.
  - **Opt-in reranking**: `reranker=None` and `rerank_metric=None` (default) disables reranking (passing `rerank_data` without `rerank_metric` raises `ValueError`). Pass `reranker=...` to share a prebuilt `Reranker` (zero-copy), or `rerank_metric=...` (with `rerank_data=None` to extract automatically from an `FP32`/`FP16` source graph).
  - **Ordering**: `custom_features_order` and `rerank_order` default to `"label"` (rows aligned with original dataset IDs / external labels); pass `"internal"` when rows already follow `graph.get_labels()`.

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

### `Searcher` & `create_searcher`

```python
create_searcher(
    graph: DynamicExplorationGraph,
    metric: Metric | str | None = None,
    reranker: Reranker | None = None,
    rerank_metric: Metric | str | None = None,
    rerank_data: np.ndarray | None = None,
    rerank_order: Literal["label", "internal"] = "label",
    rerank_transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
    normalize_query: bool = False,
    optimize: bool = True,
    use_routing: bool = True,
    n_clusters: int = 256,
    n_iter: int = 20,
    sample_size: int = 0,
    threads: int = 0,
) -> Searcher
```

`Searcher` attributes & methods:
- `searcher.graph -> DynamicExplorationGraph` (the backing read-only graph)
- `searcher.normalize_query -> bool` (get/set query L2 normalization)
- `searcher.optimize(n_clusters: int = 256, n_iter: int = 20, sample_size: int = 0, threads: int = 0, use_routing: bool = True) -> None`
- `searcher.search(query: np.ndarray, k: int = 10, eps_or_ef: float = 0.1, rerank_factor: float = 1.0, threads: int = 1, return_distances: bool = False, unsorted: bool = False, max_distance_computation_count: int = 0) -> SearchResults`
  - `eps_or_ef < 1.0`: relative distance margin (`eps`); `eps_or_ef >= 1.0`: fixed beam size (`ef`).
  - Defaults `return_distances=False, unsorted=False` (populates 1D `(found,)` or 2D `(N, k)` `.indices`, `.distances=None` unless `return_distances=True`, plus traversal/rerank telemetry).

### `Filter`

```python
Filter(valid_labels: np.ndarray, max_value: int = -1, max_label_count: int = -1)
```

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

All transformations are created pre-fitted and validated via `make_*` factories and share the `FeatureTransformation` interface (`input_dim`, `output_dim`, `output_dtype`, `output_byte_size`, `is_fitted`, `transform_database(vectors, threads=0)`, `transform_query(queries)`, `serialize()`). Raw unfitted quantizer constructors and `mips_l2_transform(_query)` are not part of the API.

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
