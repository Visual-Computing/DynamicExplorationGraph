# deglib: Python Bindings for the Dynamic Exploration Graph

Python bindings for the high-performance C++ Dynamic Exploration Graph (DEG) library, enabling approximate nearest neighbor search (ANNS) and graph exploration with state-of-the-art recall vs. QPS trade-offs.

---

## Table of Contents

- [Installation](#installation)
  - [From PyPI](#from-pypi)
  - [Build from Source](#build-from-source)
- [Quickstart & Examples](#quickstart--examples)
  - [Basic Usage](#basic-usage)
  - [Graph Types & Lifecycles](#graph-types--lifecycles)
  - [Saving and Loading Graphs](#saving-and-loading-graphs)
  - [Incremental / Streaming Graph Construction](#incremental--streaming-graph-construction)
  - [Filtered Search & Candidate Reranking](#filtered-search--candidate-reranking)
  - [Exploratory Search & Graph Navigation](#exploratory-search--graph-navigation)
  - [Graph Optimization, Quantization & Reranking Pipeline](#graph-optimization-quantization--reranking-pipeline)
- [Concepts & Parameters](#concepts--parameters)
  - [OptimizationTarget](#optimizationtarget)
  - [Search Parameter `eps`](#search-parameter-eps)
  - [Supported Metrics & Data Types](#supported-metrics--data-types)
- [Example Projects](#example-projects)
- [API Reference](#api-reference)
- [Development Setup](#development-setup)
---

## Installation

### From PyPI

```bash
pip install deglib
```

### Build from Source

To build and install the package directly from the repository:

```bash
cd python/
python setup.py copy_build_files
pip install .
```

---

## Quickstart & Examples

### Basic Usage

```python
import numpy as np
import deglib

num_samples, dims = 10_000, 128

# 1. Create random feature dataset and query vector
data = np.random.random((num_samples, dims)).astype(np.float32)
query = np.random.random(dims).astype(np.float32)

# 2. Build index directly from data (multithreaded by default)
graph = deglib.builder.build_from_data(data, edges_per_vertex=32, callback="progress")

# 3. Query top-k nearest neighbors
indices, distances = graph.search(query, k=10, eps=0.1)

print("Nearest neighbors:", indices)
print("Distances:", distances)
```

---

### Graph Types & Lifecycles

All search graphs are represented by `DynamicExplorationGraph`, backed by one of three internal graph engines:

1. **Fixed-Capacity Mutable (`SizeBoundedGraph`)**: Memory is preallocated for a fixed maximum capacity. Fast and memory-efficient for static/batch datasets.
   ```python
   space = deglib.FloatSpace.create(dims=128, metric=deglib.Metric.FP32_L2)
   graph = deglib.create_empty(capacity=10_000, feature_space=space, edges_per_vertex=32)
   ```

2. **Chunk-Allocated Mutable (`DynamicGraph`)**: Dynamically allocates memory in chunks (e.g. 1024 vertices per chunk). Ideal for streaming datasets where total capacity is unknown.
   ```python
   graph = deglib.create_dynamic_empty(feature_space=space, edges_per_vertex=32, chunk_size=1024)
   ```

3. **Read-Only Deployment (`ReadOnlyGraph`)**: Stripped of mutation structures for minimal memory footprint and maximum query throughput.
   ```python
   readonly_graph = graph.to_readonly()
   ```

---

### Saving and Loading Graphs

```python
# Save mutable graph to disk
graph.save_graph("index.deg")

# Load as compact, optimized read-only graph for production search
readonly_graph = deglib.load_readonly_graph("index.deg")

# Load as dynamic graph supporting further insertions and deletions
dynamic_graph = deglib.load_dynamic_graph("index.deg")

# Load as fixed-capacity mutable graph
mutable_graph = deglib.load_mutable_graph("index.deg")
```

---

### Incremental / Streaming Graph Construction

For continuous additions and removals, use `GraphBuilder`:

```python
import numpy as np
import deglib
from deglib.builder import GraphBuilder, OptimizationTarget
from deglib.distances import FloatSpace, Metric

dims = 128
max_capacity = 10_000
edges_per_vertex = 32

# 1. Create feature space and empty mutable graph
space = FloatSpace.create(dims, metric=Metric.FP32_L2)
graph = deglib.create_empty(max_capacity, space, edges_per_vertex)

# 2. Initialize builder
builder = GraphBuilder(graph, optimization_target=OptimizationTarget.StreamingData, seed=42)

# 3. Add entries (individually or in batches)
labels = np.arange(100, dtype=np.uint32)
features = np.random.random((100, dims)).astype(np.float32)
builder.add_entry(labels, features)

# 4. Remove entries by external label
builder.remove_entry(42)

# 5. Build / optimize
builder.build()
```

---

### Filtered Search & Candidate Reranking

```python
from deglib.search import Filter, rerank

# Search only within allowed external labels
allowed_ids = np.array([1, 5, 10, 42, 99], dtype=np.int32)
search_filter = Filter(allowed_ids)

indices, distances = graph.search(query, k=5, eps=0.1, filter_labels=search_filter)

# Exact distance reranking across candidates
queries = np.random.random((10, dims)).astype(np.float32)
candidates = np.random.randint(0, 1000, size=(10, 50), dtype=np.uint32)
base_vectors = np.random.random((1000, dims)).astype(np.float32)

top_indices, top_distances = rerank(
    space=graph.get_feature_space(),
    queries=queries,
    candidate_indices=candidates,
    base_vectors=base_vectors,
    k_top=10,
    return_distances=True,
)
```

---

### Exploratory Search & Graph Navigation

DEG supports exploratory search directly from existing vertex labels:

```python
# Explore graph starting from entry vertex label 105
explored_labels, distances = graph.explore(entry_external_label=105, k=10, eps=0.1, include_entry=False)
```

---

### Graph Optimization, Quantization & Reranking Pipeline

A complete end-to-end pipeline demonstrating **FLAS pre-sorting**, **multithreaded graph building**, **RNG edge pruning**, **Int8 quantization**, **ReadOnlyGraph conversion**, and **exact FP32 candidate reranking**:

```python
import numpy as np
import deglib
from deglib.optimization import presort, prune_non_rng_edges, quantize_int8
from deglib.search import rerank

num_vectors, dims = 10_000, 128
data = np.random.randn(num_vectors, dims).astype(np.float32)
query = np.random.randn(1, dims).astype(np.float32)

# 1. Pre-sort vectors using FLAS for improved memory locality and index construction speed
perm = presort(data, metric=deglib.Metric.FP32_InnerProduct, callback="progress")
sorted_data = data[perm]

# 2. Build exploration graph on unquantized/original data
graph = deglib.builder.build_from_data(
    sorted_data,
    metric=deglib.Metric.FP32_InnerProduct,
    edges_per_vertex=32,
    callback="progress",
)

# 3. Prune redundant non-RNG edges to optimize graph topology
pruned_count = prune_non_rng_edges(graph)
print(f"Pruned {pruned_count} redundant edges.")

# 4. Quantize dataset to Int8 for memory reduction and ultra-fast quantized graph search
quant_int8_data = quantize_int8(sorted_data)
int8_space = deglib.FloatSpace.create(dims, deglib.Metric.Int8_InnerProduct)

# 5. Convert mutable graph to a compact ReadOnlyGraph equipped with the quantized features
readonly_graph = graph.to_readonly(feature_space=int8_space, custom_features=quant_int8_data)

# 6. Search candidates on the quantized ReadOnlyGraph (fetch 2x candidates for reranking)
quant_query = quantize_int8(query)
candidate_indices = readonly_graph.search(quant_query, k=20, eps=0.1, return_distances=False)

# 7. Exact distance reranking of top candidates on original float32 data
fp32_space = deglib.FloatSpace.create(dims, deglib.Metric.FP32_InnerProduct)
final_top_indices, distances = rerank(
    space=fp32_space,
    queries=query,
    candidate_indices=candidate_indices,
    base_vectors=sorted_data,
    k_top=10,
    return_distances=True,
)

print("Top-10 nearest neighbor indices:", final_top_indices[0])
print("Top-10 exact distances:", distances[0])
```

---

## Concepts & Parameters

### `OptimizationTarget`
Controls the topology optimization strategy:
- `OptimizationTarget.LowLID`: Default for datasets with low local intrinsic dimensionality (supports multithreaded building).
- `OptimizationTarget.HighLID`: Optimized for datasets with high local intrinsic dimensionality (supports multithreaded building).
- `OptimizationTarget.StreamingData`: Optimized for continuous dynamic additions and deletions.

### Search Parameter `eps`
- The epsilon parameter expands the search priority queue during graph exploration.
- Small values (e.g. `eps=0.001` or `eps=0.01`): Faster query execution.
- Higher values (e.g. `eps=0.1` to `eps=0.3`): Higher recall rate.

### Supported Metrics & Data Types
- `Metric.FP32_L2`: Euclidean distance (`np.float32`)
- `Metric.FP32_InnerProduct`: Inner product / cosine distance (`np.float32`)
- `Metric.Uint8_L2`: 8-bit unsigned integer Euclidean distance (`np.uint8`)
- `Metric.Uint8_InnerProduct`: 8-bit unsigned integer inner product (`np.uint8`)
- `Metric.FP16_L2`: 16-bit half-precision Euclidean distance (`np.uint16`)
- `Metric.FP16_InnerProduct`: 16-bit half-precision inner product (`np.uint16`)
- `Metric.EVP_InnerProduct`: Quantized Extreme Value Property bit-packed vectors (`np.uint8`)
- `Metric.Int8_InnerProduct`: 8-bit signed integer inner product (`np.int8`)
- `Metric.Int8_L2`: 8-bit signed integer Euclidean distance (`np.int8`)

---

## API Reference

For a complete overview of all Python modules, classes, and function signatures, see [API.md](API.md) or the [Official Documentation](https://dynamic-exploration-graph.readthedocs.io/).

---

## Example Projects

Ready-to-run example scripts with visual progress and evaluation are located in the [examples/](../examples/) directory:

- [examples/knng/](../examples/knng/): k-NN graph construction benchmark using EVP quantization and FP16 reranking (SISAP 2026 Challenge Task 1).
- [examples/mips/](../examples/mips/): Maximum Inner Product Search (MIPS) benchmark using $(d+1)$-dimensional $L_2$ transformation, FLAS pre-sorting, and SIMD FP16 inner products (SISAP 2026 Challenge Task 2).
- [examples/static_data/](../examples/static_data/): Static dataset indexing and ANNS benchmark.
- [examples/vibe/](../examples/vibe/): VIBE benchmark for modern embedding datasets with scalar quantization and interactive Plotly visualizations.
- [examples/dynamic_data/](../examples/dynamic_data/): Dynamic streaming additions and deletions.
- [examples/sliding_window/](../examples/sliding_window/): Sliding window continuous update benchmark against CleANN.


---

## Development Setup

If you want to contribute, modify C++ bindings, run tests, or build release packages:

```bash
# 1. Setup virtual environment and dependencies (using uv)
cd python/
uv venv
uv pip install setuptools==83.0.0 pybind11==3.0.4 build==1.5.0 wheel==0.48.0
uv run python setup.py copy_build_files

# 2. Install in editable mode
uv pip install -e . --no-build-isolation --verbose

# 3. Run test suite
uv run pytest

# 4. Format code
uv run ruff format .

# 5. Build distribution packages (sdist and binary wheels)
uv run python -m build
```