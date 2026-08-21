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
  - [Saving and Loading Graphs](#saving-and-loading-graphs)
  - [Incremental / Streaming Graph Construction](#incremental--streaming-graph-construction)
  - [Exploratory Search & Graph Navigation](#exploratory-search--graph-navigation)
  - [Memory Safety & Referencing C++ Buffers](#memory-safety--referencing-c-buffers)
- [Concepts & Parameters](#concepts--parameters)
  - [Internal Index vs. External Label](#internal-index-vs-external-label)
  - [OptimizationTarget](#optimizationtarget)
  - [Search Parameter `eps`](#search-parameter-eps)
- [Example Projects](#example-projects)
- [Publishing a New Version](#publishing-a-new-version)

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

# 2. Build index directly from data
graph = deglib.builder.build_from_data(data, edges_per_vertex=32, callback="progress")

# 3. Query top-k nearest neighbors
indices, distances = graph.search(query, k=10, eps=0.1)

print("Nearest neighbors:", indices)
print("Distances:", distances)
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
graph = deglib.DynamicExplorationGraph.create_empty(max_capacity, space, edges_per_vertex)

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

### Exploratory Search & Graph Navigation

DEG supports exploratory search directly from existing vertex labels:

```python
# Explore graph starting from entry vertex label 105
explored_labels, distances = graph.explore(entry_label=105, k=10, eps=0.1, include_entry=False)
```

---

### Memory Safety & Referencing C++ Buffers

When fetching feature vectors from a graph:

```python
# Feature vector references internal C++ memory owned by graph
feature_vector = graph.get_feature_vector(42)

# If the graph might be deleted or modified, create an explicit copy:
safe_vector = graph.get_feature_vector(42, copy=True)
```

---

## Concepts & Parameters

### Internal Index vs. External Label
- **`internal_index`**: Dense index in range `0..size-1` used internally for high-performance memory indexing.
- **`external_label`**: User-defined unique identifier (`uint32`) assigned during `add_entry(label, feature)`. Search and exploration results map back to external labels.

### `OptimizationTarget`
Controls the topology optimization strategy:
- `OptimizationTarget.StreamingData`: Default for continuous dynamic additions and deletions.
- `OptimizationTarget.LowLID`: Optimized for datasets with low local intrinsic dimensionality (supports multithreaded building).
- `OptimizationTarget.HighLID`: Optimized for datasets with high local intrinsic dimensionality (supports multithreaded building).

### Search Parameter `eps`
- The epsilon parameter expands the search priority queue during graph exploration.
- Small values (e.g. `eps=0.001` or `eps=0.01`): Faster query execution.
- Higher values (e.g. `eps=0.1` to `eps=0.3`): Higher recall rate.

---

## API Reference

For a complete overview of all Python modules, classes, and function signatures, see [API.md](API.md).

---

## Example Projects

Ready-to-run example scripts with visual progress and evaluation are located in the [examples/](../examples/) directory:

- [examples/knng/](../examples/knng/): k-NN graph construction and evaluation.
- [examples/dynamic_data/](../examples/dynamic_data/): Dynamic streaming additions and deletions.
- [examples/static_data/](../examples/static_data/): Static dataset indexing and ANNS benchmark.
- [examples/mips/](../examples/mips/): Maximum Inner Product Search (MIPS).

---

## Publishing a New Version

1. Ensure all updates are fetched:
   ```bash
   git checkout main && git pull
   ```
2. Update version in `python/src/deglib/__init__.py`.
3. Tag and push:
   ```bash
   git add -A
   git commit -m "vX.Y.Z"
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push && git push origin --tags
   ```
