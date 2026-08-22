# deglib: Python Bindings for the Dynamic Exploration Graph

Python bindings for the high-performance C++ Dynamic Exploration Graph (DEG) library, enabling approximate nearest neighbor search (ANNS) and graph exploration with state-of-the-art recall vs. QPS trade-offs.

---

## Table of Contents

- [Installation](#installation)
  - [From PyPI](#from-pypi)
  - [Development & Compiling from Source](#development--compiling-from-source)
  - [Running Tests](#running-tests)
  - [Building Packages](#building-packages)
- [Quickstart & Examples](#quickstart--examples)
  - [Basic Usage](#basic-usage)
  - [Graph Types & Lifecycles](#graph-types--lifecycles)
  - [Saving and Loading Graphs](#saving-and-loading-graphs)
  - [Incremental / Streaming Graph Construction](#incremental--streaming-graph-construction)
  - [Filtered Search & Candidate Reranking](#filtered-search--candidate-reranking)
  - [Exploratory Search & Graph Navigation](#exploratory-search--graph-navigation)
  - [Graph Optimization & Pruning](#graph-optimization--pruning)
- [Concepts & Parameters](#concepts--parameters)
  - [OptimizationTarget](#optimizationtarget)
  - [Search Parameter `eps`](#search-parameter-eps)
  - [Supported Metrics & Data Types](#supported-metrics--data-types)
- [Example Projects](#example-projects)
- [API Reference](#api-reference)

---

## Installation

### From PyPI

```bash
pip install deglib
```

### Development & Compiling from Source

We recommend using [`uv`](https://docs.astral.sh/uv/) for fast virtual environment and dependency management.

**1. Create and Activate Virtual Environment**
```bash
cd python/
uv venv
```

**2. Install Build Dependencies & Copy Core C++ Files**
```bash
uv pip install setuptools==83.0.0 pybind11==3.0.4 build==1.5.0 wheel==0.48.0
uv run python setup.py copy_build_files
```

**3. Install in Editable Mode**
```bash
uv pip install -e . --no-build-isolation --verbose
```

### Running Tests

Run the test suite using `pytest`:

```bash
uv run pytest
```

### Code Formatting & Linting

Format Python files using `ruff`:

```bash
uv run ruff format .
```

### Building Packages

Build source distributions (`sdist`) and binary wheels:

```bash
uv run python -m build
```

To build manylinux/musllinux wheels for Linux distribution via PyPI (ensure `copy_build_files` has been executed first):
```bash
uv run python setup.py copy_build_files
cibuildwheel --archs auto64 --output-dir dist
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
    return_distances=True
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

### Graph Optimization & Pruning

```python
from deglib.optimization import prune_non_rng_edges, presort

# 1. 1D pre-sorting of vectors using FLAS for improved memory locality and build speed
perm = presort(data, metric=deglib.Metric.FP32_L2, callback="progress")
sorted_data = data[perm]

# 2. Remove redundant non-RNG edges after graph construction
removed_edges = prune_non_rng_edges(graph)
print(f"Removed {removed_edges} non-RNG edges.")
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
- `Metric.FP16_InnerProduct`: 16-bit half-precision inner product (`np.uint16`)
- `Metric.EVP_InnerProduct`: Quantized Extreme Value Property bit-packed vectors (`np.uint8`)

---

## API Reference

For a complete overview of all Python modules, classes, and function signatures, see [API.md](API.md) or the [Official Documentation](https://dynamic-exploration-graph.readthedocs.io/).

---

## Example Projects

Ready-to-run example scripts with visual progress and evaluation are located in the [examples/](../examples/) directory:

- [examples/knng/](../examples/knng/): k-NN graph construction and evaluation.
- [examples/dynamic_data/](../examples/dynamic_data/): Dynamic streaming additions and deletions.
- [examples/static_data/](../examples/static_data/): Static dataset indexing and ANNS benchmark.
- [examples/mips/](../examples/mips/): Maximum Inner Product Search (MIPS).
