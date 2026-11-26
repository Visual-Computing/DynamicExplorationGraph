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
| **Analysis & Tuning** | `deglib.analysis.*`, `deglib.tuning.*` | `analyze_graph`, `GraphStats`, `calc_graph_quality`, `DatasetProfile`, `measure_dataset_profile`, `predict_optimization_target`, `predict_quantization`, `predict_quantization_normalized` |

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

# Build graph (metric="l2" by default)
graph = deglib.build(data)

# Option A: Search directly on the graph
res = graph.search(query, k=10, eps_or_ef=0.1)
indices = res.indices

# Option B: Wrap in a Searcher for extra search optimizations (see `deglib.searcher` below)
searcher = deglib.searcher(graph)
res = searcher.search(query, k=10, eps_or_ef=0.1)
indices = res.indices
```

### Quantized Search with Self-Extracted FP16 Reranking

```python
import deglib

# 1. Build FP32 cosine graph
base_graph = deglib.build(data, metric="cosine")

# 2. Compress to 8-bit and auto-extract an FP16 reranker
searcher = deglib.searcher(base_graph, compress="8bit", reranker=True)

# 3. Fetch 2 * k candidates from graph and re-score with reranker
indices = searcher.search(queries, k=10, eps_or_ef=64, rerank_factor=2.0).indices
```

### MIPS (`metric="ip"`) with Compression

```python
import deglib

# Lifts into MIPS(d+1) for build, then compresses back to d-dim FP16 IP
ip_graph = deglib.build(train_vectors, metric="ip", compress="16bit")
res = ip_graph.search(query, k=10, eps_or_ef=0.18)
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
    presort: bool | np.ndarray | None = None,
    optimize: bool = True,
    threads: int = 0,
    show_progress: bool | None = None,
) -> DynamicExplorationGraph
```

Builds and optionally optimizes/compresses a `DynamicExplorationGraph` from a 2D NumPy feature matrix.

**Parameters:**
- **`data`**: 2D feature matrix of shape `(N, dim)`. Supports `float32` and `float16` for high-level metrics and `compress`, or pre-quantized dtypes (e.g. `int8`, `uint8`) when an explicit quantized `Metric` is passed.
- **`labels`**: Optional 1D sequence or `np.ndarray` of external `uint32` vertex IDs of length `N`. Defaults to `0 .. N - 1`.
- **`metric`**: High-level spatial mode or an explicit `Metric` / metric string (e.g. `Metric.Int8_L2`, `"Uint7_L2"`):
  - `"l2"` / `"euclidean"`: Euclidean L2 distance (works on both normalized and unnormalized data).
  - `"cosine"`: Angle/direction similarity on unnormalized vectors; attaches `L2NormalizeTransform` if not already unit-norm so DB/queries are normalized and `8bit`/`auto` prediction (`Uint7_L2` vs. `Int8_InnerProduct`) works accurately.
  - `"normalized"`: Use when input vectors are **already** L2-normalized (`||x||_2 == 1`); avoids `MIPS(d + 1)` build overhead and enables `8bit`/`auto` prediction (whereas `"ip"` would lift into `MIPS(d + 1)` and force `Int8_InnerProduct`).
  - `"ip"` / `"innerproduct"` / `"inner_product"` / `"mips"`: Norm-sensitive Maximum Inner Product Search on unnormalized vectors; builds in `MipsTransform (d + 1) + L2` space at the `compress` bit-width and swaps stored features back to unnormalized `d`-dim `InnerProduct`.
- **`compress`**: Optional target precision/metric (`"32bit"` / `"fp32"` / `"float32"`, `"16bit"` / `"fp16"` / `"float16"`, `"8bit"`, `"4bit"`, `"2bit"` / `"evp"`, `"auto"`, or an explicit `Metric` / metric string). Builds the graph directly in the compressed format (saving build RAM) while preserving mutability (`SizeBoundedGraph`). Use `deglib.compress(graph, ...)` instead when you want to build in `FP32`/`FP16` first and compress afterwards.
- **`edges_per_vertex`**: Number of outgoing neighbors per vertex (`k_graph`).
- **`optimization_target`**: Graph construction strategy (`OptimizationTarget.Auto`, `OptimizationTarget.LowLID`, `OptimizationTarget.HighLID`, `OptimizationTarget.StreamingData`, or string equivalents). `Auto` predicts `LowLID` vs. `StreamingData` from dataset geometry.
- **`extend_k`**: Candidate pool size during neighbor extension (`0` uses the builder default `2 * edges_per_vertex`).
- **`extend_eps`**: Exploration tolerance `eps` during neighbor extension.
- **`improve_k`**: Candidate pool size during edge improvement (`0` defaults to `edges_per_vertex` in `GraphBuilder`).
- **`improve_eps`**: Exploration tolerance `eps` during edge improvement.
- **`max_path_length`**: Maximum path length considered during RNG edge pruning/improvement.
- **`improve_tries`**: Number of improvement passes per step.
- **`seed`**: Random seed used during graph construction.
- **`presort`**: `None` (default) enables 1D FLAS pre-sorting automatically for `float32`/`float16` inputs (and skips it for pre-quantized inputs); `True` forces pre-sorting (reordering `data` and `labels` together); a 1D `uint32` `np.ndarray` applies precomputed sorted indices; `False` keeps insertion order.
- **`optimize`**: Runs `graph.optimize()` (prefetch & entry-point tuning) after building when `True`.
- **`threads`**: Worker thread count (`0` uses all available CPU cores).
- **`show_progress`**: Whether to print a live progress bar on stdout (`None` defaults to `sys.stdout.isatty()`).

**Returns:** Mutable `SizeBoundedGraph` (`DynamicExplorationGraph`) with the requested `metric` and `compress` target.

---

### `deglib.presort`

```python
deglib.presort(
    vectors: np.ndarray,
    metric: Metric | str = Metric.FP32_L2,
    radius_decay: float = 0.9,
    threads: int = 0,
    callback: Callable[[float], bool | None] | None = None,
    show_progress: bool = False,
) -> np.ndarray
```

Runs 1D FLAS spatial pre-sorting on dataset vectors (automatically lifting `"ip"` into MIPS space).

**Parameters:**
- **`vectors`**: 2D `float32` or `float16` NumPy array of shape `(N, dim)`.
- **`metric`**: High-level metric string (`"l2"`, `"euclidean"`, `"cosine"`, `"normalized"`, `"ip"`, `"mips"`, etc.) or explicit FP32/FP16 `Metric`.
- **`radius_decay`**: Neighborhood radius decay rate per FLAS iteration in `(0, 1)`.
- **`threads`**: Worker thread count (`0` uses all CPU cores).
- **`callback`**: Optional progress callback receiving progress in `[0.0, 1.0]`; returning `True` cancels sorting.
- **`show_progress`**: If `True`, prints a live progress bar to stdout.

**Returns:** 1D `uint32` `np.ndarray` of permutation indices of length `N`.

---

### `deglib.compress`

```python
deglib.compress(
    graph: DynamicExplorationGraph,
    target: Literal["32bit", "16bit", "8bit", "4bit", "2bit", "auto"] | Metric | str,
    threads: int = 0,
) -> DynamicExplorationGraph
```

Compresses an uncompressed (`FP32` or `FP16`) graph to the requested bit-width or `Metric`, preserving topology and external labels.

**Parameters:**
- **`graph`**: Source uncompressed `FP32` or `FP16` `DynamicExplorationGraph`.
- **`target`**: Target precision or metric:
  - `"32bit"` / `"fp32"` / `"float32"`: `FP32_L2` or `FP32_InnerProduct` (`FP32` graphs only; no-op).
  - `"16bit"` / `"fp16"` / `"float16"`: `FP16_L2` or `FP16_InnerProduct`.
  - `"8bit"` / `"auto"`: `Uint7_L2` on L2 graphs, `deglib.tuning.predict_quantization` (`Uint7_L2` vs. `Int8_InnerProduct`) on unit-sphere `cosine`/`normalized` graphs, and `Int8_InnerProduct` on unnormalized `InnerProduct` graphs.
  - `"4bit"`: `Uint4_L2` on L2 and `cosine`/`normalized` graphs; `Int4_InnerProduct` on unnormalized `InnerProduct` graphs (no `d+1` `MipsTransform` overhead).
  - `"2bit"` / `"evp"`: `EVP_InnerProduct`.
  - Any explicit `Metric` or metric name string (e.g. `Metric.Int8_L2`, `"Uint8_L2"`).
- **`threads`**: Worker thread count (`0` uses all CPU cores).

**Returns:** Compressed `DynamicExplorationGraph`.

---

### `deglib.reranker`

```python
deglib.reranker(
    graph: DynamicExplorationGraph,
    compress: Literal["16bit", "8bit"] | Metric | str | None = "16bit",
    optimize: bool = True,
    threads: int = 0,
) -> Reranker
```

Extracts a standalone `Reranker` directly from an uncompressed (`FP32` or `FP16`) `graph`'s stored vectors (raises `ValueError` if `graph` is already quantized; use `deglib.reranker_with_data` instead).

**Parameters:**
- **`graph`**: Uncompressed (`FP32` or `FP16`) `DynamicExplorationGraph` whose stored vectors and labels are extracted.
- **`compress`**: Target reranker precision (`"16bit"`, `"8bit"` / `"auto"`, `"32bit"` / `None`, `"int8"`, `"uint8"`, `"uint7"`, or an explicit `Metric`).
- **`optimize`**: If `True`, runs `reranker.optimize()` to tune candidate prefetching.
- **`threads`**: Worker thread count (`0` uses all CPU cores).

**Returns:** `Reranker` aligned with `graph.get_labels()`.

---

### `deglib.reranker_with_data`

```python
deglib.reranker_with_data(
    graph: DynamicExplorationGraph,
    data: np.ndarray,
    labels: Sequence[int] | np.ndarray | None = None,
    compress: Literal["16bit", "8bit"] | Metric | str | None = "16bit",
    metric: Literal["l2", "cosine", "normalized", "ip"] | Metric | str | None = None,
    optimize: bool = True,
    threads: int = 0,
) -> Reranker
```

Creates a `Reranker` from external `data` aligned with `graph` (supports quantized and presorted graphs while preserving any `L2NormalizeTransform` / `MipsTransform` prefix attached to `graph`).

**Parameters:**
- **`graph`**: Target `DynamicExplorationGraph` whose internal vertex order (`graph.get_labels()`) the reranker aligns to.
- **`data`**: 2D `float32` or `float16` feature matrix.
- **`labels`**: Optional external labels corresponding to rows of `data`. Defaults to `0 .. len(data) - 1`.
- **`compress`**: Target reranker precision (`"16bit"`, `"8bit"` / `"auto"`, `"32bit"`, `None`, or explicit `Metric`).
- **`metric`**: Optional distance kind override (`"l2"` / `"euclidean"`, `"cosine"`, `"normalized"`, `"ip"` / `"innerproduct"` / `"mips"`, or explicit `Metric`). Defaults to `graph.get_metric()`.
- **`optimize`**: If `True`, runs `reranker.optimize()` to tune candidate prefetching.
- **`threads`**: Worker thread count (`0` uses all CPU cores).

**Returns:** `Reranker` aligned with `graph.get_labels()`.

---

### `deglib.searcher`

```python
deglib.searcher(
    graph: DynamicExplorationGraph,
    compress: Literal["32bit", "16bit", "8bit", "4bit", "2bit", "auto"] | Metric | str | None = None,
    reranker: Reranker | bool | Metric | str | None = None,
    optimize: bool = True,
    threads: int = 0,
) -> Searcher
```

Wraps `graph` (optionally compressing first via `deglib.compress`) in an optimized `ReadOnlyGraph`-backed `Searcher`. Compared to `graph.search(...)`, `Searcher` adds a compact `ReadOnlyGraph` layout and, when `optimize=True`, tunes K-Means entry-point routing and memory prefetching for both graph traversal and reranking.

> **Memory & Ownership:** When `graph` is already a `ReadOnlyGraph` and `compress=None`, `searcher.graph` shares `graph` zero-copy (`searcher.graph is graph`). When `graph` is mutable or `compress` is set, a new `ReadOnlyGraph` is allocated for `searcher.graph` (drop `graph` if you no longer need the mutable/uncompressed copy in RAM).

**Parameters:**
- **`graph`**: `DynamicExplorationGraph` to search.
- **`compress`**: Optional compression target (`"32bit"`, `"16bit"`, `"8bit"`, `"4bit"`, `"2bit"`, `"auto"`, `Metric`, or `None` to use `graph` as-is).
- **`reranker`**: Optional reranker for two-stage candidate re-scoring:
  - `None` / `False`: No reranking.
  - `Reranker`: Prebuilt `Reranker` (from `deglib.reranker` or `deglib.reranker_with_data`).
  - `True`: Automatically extracts a `"16bit"` reranker from `graph` before compression (requires an uncompressed `FP32` or `FP16` `graph`).
  - `str` or `Metric` (e.g. `"8bit"`, `"16bit"`, `"32bit"`, `Metric.Int8_L2`): Automatically builds a reranker with that precision from `graph` via `deglib.reranker` before compression.
- **`optimize`**: If `True`, runs `searcher.optimize(threads=threads)` (K-Means routing entry points and prefetch tuning).
- **`threads`**: Worker thread count for compression and optimization (`0` uses all CPU cores).

**Returns:** `Searcher` ready for `searcher.search(...)`.

---

## 2. Metrics & CPU Instruction Sets (`deglib.distances`, `deglib.cpu`)

### Choosing High-Level Metric Strings (`"l2"`, `"cosine"`, `"normalized"`, `"ip"`)

| Metric String | Input Data | Behavior & When to Choose |
|---|---|---|
| `"l2"` / `"euclidean"` | Unnormalized or normalized | Euclidean distance (`FP32_L2` / `FP16_L2`). |
| `"cosine"` | Unnormalized | Angle/direction similarity. Attaches `L2NormalizeTransform` if vectors are not already unit-norm, ensuring DB/queries are normalized and `8bit`/`auto` prediction (`Uint7_L2` vs. `Int8_InnerProduct`) works accurately. |
| `"normalized"` | Already unit-norm (`\|\|x\|\|_2 == 1`) | Use instead of `"ip"` when vectors are already L2-normalized: avoids unnecessary `MIPS(d + 1)` build overhead and enables `8bit`/`auto` prediction. |
| `"ip"` / `"innerproduct"` / `"mips"` | Unnormalized | Norm-sensitive Maximum Inner Product Search. Builds in `MipsTransform (d + 1) + L2` for strong topology, then swaps back to `d`-dim `InnerProduct`. |

`deglib.Metric` encapsulates distance functions, vector dtype/byte-size calculations, distance evaluation helpers, and optional SIMD `InstructionSet` bindings. `FloatSpace` is internal and not exposed in the public API.

### `Metric` (`IntEnum`)

Supported metric members:
- **FP32**: `Metric.FP32_L2`, `Metric.FP32_InnerProduct` (`np.float32`)
- **FP16**: `Metric.FP16_L2`, `Metric.FP16_InnerProduct` (`np.float16` or raw `np.uint16` bit representation)
- **UINT8**: `Metric.Uint8_L2`, `Metric.Uint8_InnerProduct` (`np.uint8`)
- **INT8**: `Metric.Int8_L2`, `Metric.Int8_InnerProduct` (`np.int8`)
- **UINT7**: `Metric.Uint7_L2`, `Metric.Uint7_InnerProduct` (`np.int8`, values in `[0, 127]`)
- **INT4**: `Metric.Int4_L2`, `Metric.Int4_InnerProduct` (`np.uint8`, packed 2 values/byte)
- **UINT4**: `Metric.Uint4_L2`, `Metric.Uint4_InnerProduct` (`np.uint8`, packed 2 values/byte)
- **EVP**: `Metric.EVP_InnerProduct` (`np.uint8`, packed bitmasks of size `2 * ((dim + 7) // 8)`)

#### `Metric.resolve`

```python
Metric.resolve(
    metric: Metric | str | int,
    distance_kind: str | None = None,
    instruction: InstructionSet | str | None = None,
) -> Metric
```

Resolves a `Metric` from an enum member, integer value, or case-insensitive name/parts (e.g. `Metric.resolve("int8", "ip", instruction="AVX2")`).

#### `Metric` Instance Methods

```python
metric.with_instruction(instruction: InstructionSet | str) -> Metric
metric.get_instruction(dim: int = 16) -> InstructionSet
metric.get_dtype() -> np.dtype
metric.get_accepted_dtypes() -> tuple[np.dtype, ...]
metric.get_data_size(dim: int) -> int
metric.get_query_data_size(dim: int) -> int
metric.compute_distance(vec1: np.ndarray, vec2: np.ndarray) -> float
metric.compute_distances(query: np.ndarray, targets: np.ndarray, threads: int = 0) -> np.ndarray
```

- **`with_instruction`**: Returns a `Metric` handle bound to a specific `InstructionSet` (compares `==` to the base `Metric` enum member).
- **`get_instruction`**: Returns the bound or auto-resolved CPU `InstructionSet` for dimension `dim`.
- **`get_dtype`**: Canonical NumPy storage dtype for stored features.
- **`get_accepted_dtypes`**: All directly accepted NumPy dtypes (e.g. `(np.uint16, np.float16)` for FP16).
- **`get_data_size` / `get_query_data_size`**: Byte size of one stored/query vector of dimension `dim`.
- **`compute_distance`**: Computes distance between two 1D vectors (`vec1`, `vec2`).
- **`compute_distances`**: Computes pairwise distances between 1D/2D `query` and 2D `targets` (returns `(n_queries, n_targets)`).

---

### `InstructionSet` (`IntEnum`) & CPU Helpers (`deglib.cpu`)

Members: `InstructionSet.Auto`, `InstructionSet.Scalar`, `InstructionSet.AVX2`, `InstructionSet.AVX2_VNNI`, `InstructionSet.AVX512`, `InstructionSet.AVX512_VNNI`.

```python
InstructionSet.resolve(instruction: InstructionSet | str | int) -> InstructionSet

has_avx2() -> bool
has_avx2_vnni() -> bool
has_avx512() -> bool
has_avx512_vnni() -> bool
has_avx512_fp16() -> bool
has_fp16() -> bool
```

---

### FP16 Conversion Helpers (`deglib.distances`)

```python
floats_to_fp16(floats: np.ndarray) -> np.ndarray
fp16_to_floats(fp16_vals: np.ndarray) -> np.ndarray
```

- **`floats_to_fp16`**: Converts a `float32` array to IEEE-754 half-precision `uint16` bit-patterns.
- **`fp16_to_floats`**: Converts a `uint16` FP16 bit-pattern array back to `float32`.

---

## 3. Graphs (`deglib.graph`)

All graphs (fixed-capacity mutable `SizeBoundedGraph`, chunked `DynamicGraph`, and compact immutable `ReadOnlyGraph`) are accessed through `DynamicExplorationGraph`.

### Factory & Loader Functions

```python
create_empty(
    capacity: int,
    dim: int,
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
) -> DynamicExplorationGraph

create_dynamic_empty(
    dim: int,
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    chunk_size: int = 1024,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
) -> DynamicExplorationGraph

create_random_graph(
    features: np.ndarray,
    edges_per_vertex: int = 32,
    metric: Metric | str = Metric.FP32_L2,
    seed: int = 7,
    optimize: bool = True,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
) -> DynamicExplorationGraph

load_graph(path: Path | str) -> DynamicExplorationGraph
load_readonly_graph(path: Path | str) -> DynamicExplorationGraph
load_mutable_graph(path: Path | str, capacity: int = 0) -> DynamicExplorationGraph
load_dynamic_graph(path: Path | str, chunk_size: int = 1024) -> DynamicExplorationGraph
```

- **`create_empty`**: Creates an empty fixed-capacity mutable `SizeBoundedGraph`.
- **`create_dynamic_empty`**: Creates an empty chunk-allocated mutable `DynamicGraph`.
- **`create_random_graph`**: Creates a random `k`-regular graph directly from a 2D `features` array (running `graph.optimize()` when `optimize=True`).
- **`load_graph` / `load_readonly_graph`**: Loads a `.deg` graph from disk as an immutable `ReadOnlyGraph`.
- **`load_mutable_graph`**: Loads a `.deg` graph as a fixed-capacity `SizeBoundedGraph` (`capacity=0` uses the saved vertex count).
- **`load_dynamic_graph`**: Loads a `.deg` graph as a chunked `DynamicGraph`.

---

### `DynamicExplorationGraph` Methods

#### Introspection, Persistence & Calibration

```python
graph.size() -> int
graph.get_dim() -> int
graph.get_metric() -> Metric
graph.get_instruction() -> InstructionSet
graph.get_data_size() -> int
graph.get_edges_per_vertex() -> int
graph.is_mutable() -> bool
graph.has_vertex(external_label: int) -> bool
graph.get_labels() -> np.ndarray
graph.get_neighbors(external_label: int | np.ndarray | Sequence[int], threads: int = 1) -> list[int] | np.ndarray
graph.get_po() -> int
graph.get_pl() -> int
graph.get_nl() -> int
graph.has_transformations() -> bool
graph.get_transformations() -> list[FeatureTransformation]
graph.get_entry_vertex_indices() -> list[int]
graph.set_entry_vertex_indices(indices: Sequence[int] | np.ndarray) -> None
graph.save_graph(path: Path | str) -> None
graph.optimize(sample_count: int = 300, median_subset: int = 4096) -> None
```

- **`get_metric()`**: Returns the active `Metric` bound to the graph's `InstructionSet` (use `==` when comparing to `Metric.*`).
- **`get_labels()`**: 1D `uint32` array of external labels ordered by internal vertex slot (`0 .. size() - 1`).
- **`get_neighbors(external_label, threads=1)`**: Returns `list[int]` for a single label, or a 2D `uint32` array of shape `(len(external_label), edges_per_vertex)` for a batch.
- **`get_po()`, `get_pl()`, `get_nl()`**: Traversal prefetch offset, feature prefetch look-ahead (cache lines, `0` = whole vector), and neighbor prefetch look-ahead.
- **`save_graph(path)`**: Saves topology, features, transformations, and prefetch config to `.deg` format (auto-converts read-only graphs if needed).
- **`optimize(sample_count=300, median_subset=4096)`**: Auto-tunes traversal prefetch parameters (`po`, `pl`, `nl`) and the graph-median entry point.

```python
graph.has_path(
    from_external_labels: int | np.ndarray | Sequence[int],
    to_external_label: int,
    eps: float = 0.0,
    k: int = 1,
    return_distances: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray
```

Checks reachability via greedy traversal from `from_external_labels` to `to_external_label`.

```python
graph.to_recall_table(
    k: int,
    mode: Literal["ef", "eps"],
    sample_count: int = 300,
    recall_range: tuple[float, float] | None = None,
    recall_step: float | None = None,
    target_recalls: Sequence[float] | np.ndarray | None = None,
    threads: int = 0,
) -> RecallTable

graph.to_recall_tables(
    k: int,
    sample_count: int = 300,
    recall_range: tuple[float, float] | None = None,
    recall_step: float | None = None,
    target_recalls: Sequence[float] | np.ndarray | None = None,
    threads: int = 0,
) -> list[RecallTable]

graph.to_pareto_line(
    k: int | None = None,
    recall_tables: RecallTable | Sequence[RecallTable] | None = None,
    sample_count: int | None = None,
    recall_range: tuple[float, float] | None = None,
    recall_step: float | None = None,
    target_recalls: Sequence[float] | np.ndarray | None = None,
    threads: int = 0,
    rounds: int = 3,
) -> ParetoLine
```

- **`to_recall_table`**: Calibrates a single-mode recall@k parameter curve (`ef ∈ [10, 500]` step `1`, or `eps ∈ [0.001, 0.20]` resolution `0.0001`), picking the smallest parameter achieving `measured_recall >= target_recall` using self-sampled queries and exact ground truth (defaults `recall_range=(0.80, 0.999)`, `recall_step=0.005`).
- **`to_recall_tables`**: Calibrates both `"ef"` and `"eps"` `RecallTable` curves in one pass sharing exact ground truth.
- **`to_pareto_line`**: Unifies measured operating points across `recall_tables` (defaulting to `to_recall_tables(k, ...)`) by single-threaded median latency in `(measured_recall, latency_us)` space, prunes Pareto-dominated points, and exposes `.recall_tables`.

---

#### Graph Conversion & Searcher Construction

```python
graph.to_readonly() -> DynamicExplorationGraph
graph.to_mutable(max_size: int = 0, threads: int = 0) -> DynamicExplorationGraph
graph.to_dynamic(chunk_size: int = 1024, threads: int = 0) -> DynamicExplorationGraph

graph.with_metric(metric: Metric | str, threads: int = 0) -> DynamicExplorationGraph

graph.with_transformations(
    transformations: FeatureTransformation | Sequence[FeatureTransformation],
    metric: Metric | str | None = None,
    threads: int = 0,
) -> DynamicExplorationGraph

graph.with_features(
    features: np.ndarray,
    metric: Metric | str | None = None,
    order: Literal["label", "internal"] = "label",
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
    threads: int = 0,
) -> DynamicExplorationGraph

graph.to_reranker(
    metric: Metric | str,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
    optimize: bool = True,
    threads: int = 0,
) -> Reranker

graph.to_searcher(
    metric: Metric | str | None = None,
    reranker: Reranker | None = None,
    normalize_query: bool = False,
    optimize: bool = True,
    threads: int = 0,
) -> Searcher
```

- **`to_readonly` / `to_mutable` / `to_dynamic`**: Pure backend copies preserving features, metric, and attached transformations (`threads` is used for parallel edge-weight recomputation when converting from `ReadOnlyGraph`).
- **`with_metric`**: Re-tags or auto-quantizes stored floating-point features to `metric`.
- **`with_transformations`**: Self-transforms stored `FP32`/`FP16` features using explicit fitted `transformations` and appends them to the graph's query pipeline.
- **`with_features`**: Replaces graph features with an external array (auto-calibrating/quantizing `FP32`/`FP16` arrays when `transformations=None`) and resets the query pipeline. `order="label"` indexes rows by external labels; `order="internal"` expects rows aligned with `graph.get_labels()`.
- **`to_reranker`**: Extracts and quantizes base vectors directly from an `FP32`/`FP16` graph, chaining the graph's prefix transformations with `transformations`.
- **`to_searcher`**: Converts the graph to a `ReadOnlyGraph` in a single pass (applying/quantizing `metric` directly when `metric` differs) and wraps it in a `Searcher`.

---

#### Direct Graph Search & Exploration

```python
graph.search(
    query: np.ndarray,
    eps_or_ef: float = 0.0,
    k: int = 10,
    filter_labels: Any = None,
    max_distance_computation_count: int = 0,
    threads: int = 0,
    return_distances: bool = True,
    unsorted: bool = False,
) -> SearchResults

graph.explore(
    entry_external_label: int | np.ndarray | Sequence[int],
    k: int,
    max_distance_computation_count: int = 0,
    eps_or_ef: float = 0.0,
    include_entry: bool = True,
    threads: int = 1,
    filter_labels: Any = None,
    return_distances: bool = True,
    unsorted: bool = False,
) -> SearchResults
```

- **`query`**: 1D query vector or 2D query batch `(N, dim)`.
- **`eps_or_ef`**: Search budget (`< 1.0`: relative distance margin `eps`; `>= 1.0`: fixed beam size `ef`).
- **`k`**: Number of nearest neighbors to return.
- **`filter_labels`**: Optional `Filter` instance, 1D array, or sequence of allowed external labels.
- **`max_distance_computation_count`**: Optional hard cap on distance evaluations (`0` = unlimited).
- **`threads`**: Worker thread count for batch queries (`0` uses all CPU cores).
- **`return_distances`**: If `False`, `.distances` on `SearchResults` is `None`.
- **`unsorted`**: If `True`, skips final sorting of returned candidates.
- **`entry_external_label` / `include_entry`** (`explore` only): Starting external vertex label(s) and whether to include the entry vertex in the results.

---

## 4. Incremental Graph Building (`deglib.builder`)

### `OptimizationTarget` (`IntEnum`)

- `OptimizationTarget.Auto`: Predicts `LowLID` vs. `StreamingData` from dataset geometry (resolved to `LowLID` in `GraphBuilder` when passed directly).
- `OptimizationTarget.StreamingData`: Sequential single-threaded insertion/deletion for shifting data streams.
- `OptimizationTarget.HighLID`: Parallel batch extension replacing worst neighbor edges (for LID > 15).
- `OptimizationTarget.LowLID`: Default parallel batch extension minimizing metric distortion (for LID <= 15).
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

**Parameters:**
- **`graph`**: Mutable `DynamicExplorationGraph` (`SizeBoundedGraph` or `DynamicGraph`).
- **`seed`**: Random seed for graph construction.
- **`optimization_target`**: Construction strategy (`LowLID`, `HighLID`, `StreamingData`, or `Auto`).
- **`extend_k`**: Candidate pool size during neighbor extension (`<= edges_per_vertex` defaults to `2 * edges_per_vertex`).
- **`extend_eps`**: Exploration tolerance `eps` during neighbor extension.
- **`improve_k`**: Candidate pool size during edge improvement (`0` defaults to `edges_per_vertex`).
- **`improve_eps`**: Exploration tolerance `eps` during edge improvement.
- **`max_path_length`**: Maximum path length for RNG edge pruning/improvement.
- **`improve_tries`**: Number of improvement passes per step.
- **`threads`**: Worker thread count (`0` uses default/all cores).

**Methods:**

```python
builder.add_entry(external_label: int | Iterable[int] | np.ndarray, feature: np.ndarray) -> None
builder.remove_entry(external_label: int | Iterable[int] | np.ndarray) -> None
builder.get_num_new_entries() -> int
builder.get_num_remove_entries() -> int
builder.set_threads(threads: int) -> None
builder.set_batch_size(tasks_per_batch: int, task_size: int) -> None
builder.get_batch_size() -> int
builder.build(
    callback: Callable[[BuilderStatus], None] | None = None,
    show_progress: bool = False,
    infinite: bool = False,
) -> BuilderStatus
builder.stop() -> None
```

### `BuilderStatus` & `ProgressCallback`

- **`BuilderStatus` attributes**: `step`, `added`, `deleted`, `improved`, `tries`, `step_added_ids`, `step_deleted_ids`, `total_added_ids`, `total_deleted_ids`.
- **`ProgressCallback`**:

```python
ProgressCallback(
    num_new_entries: int,
    num_remove_entries: int,
    bar_length: int = 60,
    min_print_interval: float = 0.1,
)
```

---

## 5. High-Throughput Search, Filtering & Reranking (`deglib.search`)

### `SearchResults`

Returned by `DynamicExplorationGraph.search()`, `DynamicExplorationGraph.explore()`, and `Searcher.search()`. Holds C++ result memory exposed to Python as zero-copy NumPy views:

- **`indices: np.ndarray`**: `uint32`, 1D `(found,)` / `(k,)` or 2D `(N, k)` for batches.
- **`distances: np.ndarray | None`**: `float32`, same shape when `return_distances=True`, else `None`.
- **`graph_distance_count: int`**: Total graph vertex distance evaluations performed.
- **`rerank_distance_count: int`**: Total exact reranker distance evaluations performed (`0` unless `rerank_factor > 1.0` with a reranker).
- **`graph_hop_count: int`**: Total graph vertex expansions (hops) performed.
- **`__len__() -> int`**: Number of results (single query) or query count (batch).

### `Searcher`

```python
Searcher(
    graph: DynamicExplorationGraph,
    reranker: Reranker | None = None,
    normalize_query: bool = False,
    optimize: bool = False,
    threads: int = 0,
)
```

**Parameters:**
- **`graph`**: Backing `DynamicExplorationGraph` (typically a `ReadOnlyGraph`).
- **`reranker`**: Optional `Reranker` for two-stage candidate re-scoring.
- **`normalize_query`**: If `True`, L2-normalizes `float32` queries before graph traversal.
- **`optimize`**: If `True`, calls `self.optimize(threads=threads)` after construction.
- **`threads`**: Worker thread count used when `optimize=True` (`0` uses all CPU cores).

**Attributes & Methods:**

```python
searcher.graph -> DynamicExplorationGraph
searcher.normalize_query -> bool  # get/set

searcher.optimize(
    n_clusters: int = 256,
    n_iter: int = 20,
    sample_size: int = 0,
    threads: int = 0,
    use_routing: bool = True,
) -> None

searcher.search(
    query: np.ndarray,
    k: int = 10,
    eps_or_ef: float = 0.1,
    rerank_factor: float = 1.0,
    threads: int = 1,
    return_distances: bool = False,
    unsorted: bool = False,
    max_distance_computation_count: int = 0,
) -> SearchResults

searcher.to_recall_table(
    k: int,
    mode: Literal["ef", "eps"],
    rerank_factor: float = 1.0,
    sample_count: int = 300,
    recall_range: tuple[float, float] | None = None,
    recall_step: float | None = None,
    target_recalls: Sequence[float] | np.ndarray | None = None,
    threads: int = 0,
) -> RecallTable

searcher.to_recall_tables(
    k: int,
    sample_count: int = 300,
    recall_range: tuple[float, float] | None = None,
    recall_step: float | None = None,
    target_recalls: Sequence[float] | np.ndarray | None = None,
    rerank_factors: Sequence[float] | np.ndarray | None = None,
    threads: int = 0,
) -> list[RecallTable]

searcher.to_pareto_line(
    k: int | None = None,
    recall_tables: RecallTable | Sequence[RecallTable] | None = None,
    sample_count: int | None = None,
    recall_range: tuple[float, float] | None = None,
    recall_step: float | None = None,
    target_recalls: Sequence[float] | np.ndarray | None = None,
    rerank_factors: Sequence[float] | np.ndarray | None = None,
    threads: int = 0,
    rounds: int = 3,
) -> ParetoLine
```

- **`search(...)`**: `eps_or_ef < 1.0` uses relative distance margin `eps`; `eps_or_ef >= 1.0` uses fixed beam size `ef`. Defaults `return_distances=False, unsorted=False`.
- **`to_recall_tables(...)`**: Calibrates `("ef", "eps") x rerank_factors` in a shared single pass (`rerank_factors` defaults to `[1.0, 1.25, 1.5, 2.0]` when a reranker is attached, else `[1.0]`).

### `Filter`

```python
Filter(valid_labels: np.ndarray, max_value: int = -1, max_label_count: int = -1)
```

Bitset filter restricting graph search and exploration to a subset of external labels.

### `RecallTable`

Returned by `graph.to_recall_table(...)` / `to_recall_tables(...)` and `searcher.to_recall_table(...)` / `to_recall_tables(...)`:

```python
table.k -> int
table.sample_count -> int
table.is_ef -> bool
table.is_eps -> bool
table.rerank_factor -> float
table.recall_bounds -> tuple[float, float] | None
table.entries -> list[tuple[float, float, float]]  # (target_recall, eps_or_ef, measured_recall)

table.eps_or_ef_values() -> list[float]
table.param_for(target_recall: float, clamp: bool = False) -> float | None
len(table) -> int
str(table) -> str
```

### `ParetoLine`

Returned by `graph.to_pareto_line(...)` and `searcher.to_pareto_line(...)`:

```python
line.k -> int
line.sample_count -> int
line.recall_bounds -> tuple[float, float] | None
line.recall_tables -> list[RecallTable]
line.entries -> list[tuple[float, float, float, float, float, float]]  # (target_recall, eps_or_ef, rerank_factor, measured_recall, latency_us, qps)

line.eps_or_ef_values() -> list[float]
line.params() -> list[tuple[float, float]]
line.param_for(target_recall: float, clamp: bool = False) -> tuple[float, float] | None  # nearest frontier entry by measured_recall, no interpolation
len(line) -> int
str(line) -> str
```

### `Reranker` & `rerank`

```python
Reranker(
    metric: Metric | str,
    base_vectors: np.ndarray,
    transformations: FeatureTransformation | Sequence[FeatureTransformation] | None = None,
    optimize: bool = False,
    threads: int = 0,
    row_indices: np.ndarray | None = None,
)
```

**Parameters:**
- **`metric`**: Distance metric used to score candidates (e.g. `Metric.FP16_L2`, `Metric.FP32_InnerProduct`).
- **`base_vectors`**: 2D array of base vectors. If `float32` or `float16`, automatically converted/quantized to `metric` using `transformations` (or an auto-calibrated quantizer).
- **`transformations`**: Optional fitted `FeatureTransformation` or sequence applied to queries (and `float32`/`float16` `base_vectors` when needed).
- **`optimize`**: If `True`, runs `self.optimize()` immediately after construction.
- **`threads`**: Worker thread count for quantization (`0` uses all CPU cores).
- **`row_indices`**: Optional 1D `uint32` index mapping (e.g. `graph.get_labels()`) when `base_vectors` is indexed by external labels.

**Methods:**

```python
reranker.get_num_base_vectors() -> int
reranker.get_instruction() -> InstructionSet
reranker.get_po() -> int
reranker.get_pl() -> int
reranker.optimize(sample_count: int = 300, num_candidates: int = 150) -> None
reranker.rerank(
    queries: np.ndarray,
    candidate_indices: np.ndarray,
    k_top: int = 0,
    threads: int = 0,
    return_distances: bool = False,
    unsorted: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]
```

**Functional helper:**

```python
deglib.search.rerank(
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

All transformations are created pre-fitted and validated via `make_*` factories and share the `FeatureTransformation` interface:
- **Properties**: `name`, `type` (`TransformationType`: `Spatial`, `Rotation`, `Quantizer`), `input_dim`, `output_dim`, `output_dtype`, `output_byte_size`, `is_fitted`
- **Methods**: `transform_database(vectors, threads=0)`, `transform_query(queries)`, `serialize()`

```python
make_mips_transform(vectors: np.ndarray, max_norm: float = 0.0) -> FeatureTransformation
make_l2_normalize_transform(vectors_or_dim: np.ndarray | int) -> FeatureTransformation
make_fast_hadamard_rotation(
    vectors: np.ndarray,
    seed: int = 7,
    rounds: int = 0,
    auto_detect: bool = True,
) -> FeatureTransformation
make_quantizer_pipeline(
    vectors: np.ndarray,
    quant: Metric | str,
    quantile: float = -1.0,
    use_rerank: bool = False,
    threads: int = 0,
    seed: int = 7,
) -> list[FeatureTransformation]
make_evp_quantizer(vectors: np.ndarray, non_zeros: int = 0, threads: int = 0) -> FeatureTransformation
make_scalar_quantizer_int8(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation
make_scalar_quantizer_uint8(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation
make_scalar_quantizer_uint7(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation
make_scalar_quantizer_uint4(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation
make_scalar_quantizer_int4(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation
make_scalar_quantizer_fp16(vectors: np.ndarray, quantile: float = -1.0) -> FeatureTransformation

apply_on_database(
    transformations: FeatureTransformation | Sequence[FeatureTransformation],
    vectors: np.ndarray,
    threads: int = 0,
) -> np.ndarray
apply_on_query(
    transformations: FeatureTransformation | Sequence[FeatureTransformation],
    queries: np.ndarray,
) -> np.ndarray
```

- **`make_mips_transform`**: Spherical MIPS-to-L2 transform (`d -> d + 1`), appending `sqrt(M^2 - ||x||^2)` on database vectors and `0.0` on query vectors (`max_norm <= 0` calibrates `M` as max norm).
- **`make_l2_normalize_transform`**: Unit L2 normalization (`x -> x / ||x||_2`).
- **`make_fast_hadamard_rotation`**: Randomized Fast Walsh-Hadamard rotation to spread outlier coordinates before quantization.
- **`make_quantizer_pipeline`**: Builds a calibrated pipeline (optional Hadamard rotation + quantizer) for `quant` (`Metric` or string such as `"uint7"`, `"int8"`, `"uint4"`, `"fp16"`, `"evp"`).
- **`make_evp_quantizer`**: Calibrates `EvpQuantizer` (`non_zeros=0` auto-calibrates sparsity).

---

## 7. Optimization & Pruning (`deglib.optimization`)

```python
presort(
    vectors: np.ndarray,
    metric: Metric | str = Metric.FP32_L2,
    radius_decay: float = 0.9,
    threads: int = 0,
    callback: Callable[[float], bool | None] | None = None,
    show_progress: bool = False,
) -> np.ndarray

prune_non_mrng_edges(graph: DynamicExplorationGraph, threads: int = 0) -> int
prune_worst_edges(graph: DynamicExplorationGraph, prune_worst: int, threads: int = 0) -> None
```

- **`presort`**: Runs 1D FLAS spatial sorting; returns `uint32` permutation indices (`callback(progress)` cancels sorting if it returns `True`).
- **`prune_non_mrng_edges`**: Prunes edges violating the Monotonic Relative Neighborhood Graph (MRNG) condition in-place; returns the number of modified vertices.
- **`prune_worst_edges`**: Replaces the `prune_worst` longest outgoing edges of each vertex with self-loops in-place.

---

## 8. Graph Analysis (`deglib.analysis`)

```python
analyze_graph(
    graph: Any,
    base_top: Any | None = None,
    sample_size: int = 0,
    threads: int = 0,
    print_stats: bool = True,
) -> GraphStats

calc_graph_quality(
    graph: Any,
    base_top: Any,
    sample_size: int = 0,
    threads: int = 0,
) -> float

calc_avg_edge_weight(graph: Any, scale: int = 1) -> float
calc_edge_weight_histogram(graph: Any, sort: bool, scale: int = 1) -> list[float]
check_graph_weights(graph: Any) -> bool
check_graph_regularity(graph: Any, expected_vertices: int, check_back_link: bool = False) -> bool
check_graph_connectivity(graph: Any) -> bool
calc_non_mrng_edges(graph: Any) -> int
calc_search_reachability(graph: Any) -> float
calc_exploration_reach(graph: Any) -> float
```

- **`GraphStats`**: Summary metrics returned by `analyze_graph` (supports attribute, dict-like indexing, `.keys()`, `.to_dict()`, and `str(stats)`).

---

## 9. Pre-Build Tuning & Prediction (`deglib.tuning`)

Base-only geometric profiling and hyperparameter prediction from `float32`/`float16` feature vectors (no queries or ground truth required).

### Data Classes

```python
@dataclass(frozen=True)
class DatasetProfile:
    sample_count: int = 0
    dim: int = 0
    participation_ratio: float = 0.0
    rel_contrast_1: float = 0.0
    pp_iqr: float = 0.0
    d16_skew: float = 0.0
    ball_cnt_p90_p10_r15: float = 0.0
    hadamard_kurt: float = 0.0
    metric: Metric = Metric.FP32_L2

@dataclass(frozen=True)
class TargetPrediction:
    target: OptimizationTarget
    lowlid_advantage_pct: float
    profile: DatasetProfile

@dataclass(frozen=True)
class QuantizationPrediction:
    metric: Metric
    uint7_advantage_pct: float
    profile: DatasetProfile
```

### Functions

```python
measure_dataset_profile(
    data: np.ndarray,
    metric: Metric | str | int = Metric.FP32_L2,
    sample_count: int = 16000,
    query_count: int = 0,
    seed: int = 7,
    threads: int = 0,
) -> DatasetProfile

predict_optimization_target(
    data_or_profile: np.ndarray | DatasetProfile,
    metric: Metric | str | int = Metric.FP32_L2,
    sample_count: int = 16000,
    query_count: int = 0,
    seed: int = 7,
    threads: int = 0,
) -> TargetPrediction

predict_quantization(
    data_or_graph: np.ndarray | DynamicExplorationGraph,
    metric: Metric | str | int = Metric.FP32_L2,
    sample_count: int = 16000,
    query_count: int = 0,
    seed: int = 7,
    threads: int = 0,
) -> QuantizationPrediction

predict_quantization_normalized(
    data: np.ndarray,
    is_cosine: bool,
    metric: Metric | str | int = Metric.FP32_InnerProduct,
    sample_count: int = 16000,
    seed: int = 7,
    threads: int = 0,
) -> QuantizationPrediction
```

- **`measure_dataset_profile`**: Computes base-only geometric properties used for pre-build tuning.
- **`predict_optimization_target`**: Predicts whether `OptimizationTarget.LowLID` outperforms `OptimizationTarget.StreamingData`.
- **`predict_quantization`**: Predicts whether `Metric.Uint7_L2` or `Metric.Int8_InnerProduct` achieves higher QPS (L2/Euclidean datasets unconditionally select `Metric.Uint7_L2`; accepts either a 2D array or a `DynamicExplorationGraph`).
- **`predict_quantization_normalized`**: Predicts `Metric.Uint7_L2` vs. `Metric.Int8_InnerProduct` from already L2-normalized vectors (`is_cosine=True` when `L2NormalizeTransform` was attached).
