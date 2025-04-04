# deglib: Python bindings for the Dynamic Exploration Graph

Python bindings for the C++ library Dynamic Exploration Graph (DEG) and its predecessor continuous refining Exploration Graph (crEG).

## Table of Contents
- [Installation](#installation)
- [Examples](#examples)
- [Naming](#naming)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)

## Installation

### Using pip
```shell
pip install deglib
```
This will install a source package, that needs to compile the C++ code in order to create an optimized version for your system.

### Compiling from Source

**Create Virtual Environment**
```shell
# create virtualenv with virtualenvwrapper or venv
mkvirtualenv deglib
# or
python -m venv /path/to/deglib_env && . /path/to/deglib_env/bin/activate
```

**Get the Source**
```shell
# clone git repository
git clone https://github.com/Visual-Computing/DynamicExplorationGraph.git
cd DynamicExplorationGraph/python
```

**Install the Package from Source**
```shell
pip install setuptools pybind11 build
python setup.py copy_build_files  # copy c++ library to ./lib/
pip install .
```
This will compile the C++ code and install deglib into your virtual environment, so it may take a while.

**Testing**

To execute all tests.
```shell
pytest
```

**Building Packages**

Build packages (sdist and wheels):
```shell
python -m build
```

Note: If you want to publish linux wheels to pypi you have to convert
the wheel to musllinux-/manylinux-wheels.
This can be easily done using `cibuildwheel` (if docker is installed):

```shell
cibuildwheel --archs auto64 --output-dir dist
```

## Examples
### Loading Data
To load a dataset formatted like the [TexMex-Datasets](http://corpus-texmex.irisa.fr/):
```python
import deglib
import numpy as np

dataset: np.ndarray = deglib.repository.fvecs_read("path/to/data.fvecs")
num_samples, dims = dataset.shape
```
The dataset is a numpy array with shape (N, D), where N is the number of feature
vectors and D is the number of dimensions of each feature vector.

### Building a Graph

```python
graph = deglib.builder.build_from_data(dataset, edges_per_vertex=30, callback="progress")
graph.save_graph("/path/to/graph.deg")
rd_graph = deglib.graph.load_readonly_graph("/path/to/graph.deg")
```

*Note: Threaded building is not supported for lid == LID.Unknown (the default). Use `lid=deglib.builder.LID.High` or `lid=deglib.builder.LID.Low` in `build_from_data()` for multithreaded building*

### Searching the Graph
```python
# query can have shape (D,) or (Q, D), where
# D is the dimensionality of the dataset and
# Q is the number of queries.
query = np.random.random((dims,)).astype(np.float32)
result, dists = graph.search(query, eps=0.1, k=10)  # get 10 nearest features to query
print('best dataset index:', result[0])
best_match = dataset[result[0]]
```

For more examples see [tests](tests).

### Referencing C++ memory
Consider the following example:
```python
feature_vector = graph.get_feature_vector(42)
del graph
print(feature_vector)
```
This will crash as `feature_vector` is holding a reference to memory that is owned by `graph`. This can lead to undefined behaviour (most likely segmentation fault).
Be careful to keep objects in memory that are referenced. If you need it use the `copy=True` option:

```python
feature_vector = graph.get_feature_vector(10, copy=True)
del graph
print(feature_vector)  # no problem
```

Copying feature vectors will be slower.

## Naming
### Vertex = Feature Vector
Each vertex in the graph corresponds to a feature vector of the dataset.

### Internal Index vs External Label
There are two kinds of indices used in a graph: `internal_index` and `external_label`. Both are integers and specify
a vertex in a graph.

Internal Indices are dense, which means that every `internal_index < len(graph)` can be used.
For example: If you add 100 vertices and remove the vertex with internal_index 42, the last vertex in the graph will
be moved to index 42.

In contrast, external label is a user defined identifier for each added vertex
(see `builder.add_entry(external_label, feature_vector)`). Adding or Removing vertices to the graph will keep the
connection between external labels and associated feature vector.

When you create the external labels by starting with `0` and increasing it for each entry by `1` and don't remove
elements from the graph, external labels and internal indices are equal.

```python
# as long as no elements are removed
# external labels and internal indices are equal
for i, vec in enumerate(data):
    builder.add_entry(i, vec)
```

### Eps
The eps-search-parameter controls how many nodes are checked during search.
Lower eps values like 0.001 are faster but less accurate.
Higher eps values like 0.1 are slower but more accurate. Should always be greater 0.

### LID.Unknown vs LID.High or LID.Low
The crEG paper introduces an additional parameter, *LIDType*, to determine whether a dataset exhibits high complexity and Local Intrinsic Dimensionality (LID) or if it is relatively low. In contrast, the DEG paper presents a new algorithm that does not rely on this information. Consequently, DEG defaults to *LID.Unknown*. However, if the LID is known, utilizing it can be beneficial, as multi-threaded graph construction is only possible with these parameters.




## Limitations
- The python wrapper at the moment only supports `float32` and `uint8` feature vectors.
- Threaded building is not supported for `lid=LID.Unknown`. Use `LID.High` or `LID.Low` instead.

## Troubleshooting

### BuildError: `pybind11/typing.h:104:58: error: ‘copy_n’ is not a member of ‘std’`

This is a pybind11 bug, that occurs when compiling it with gcc-14. Change the pybind version to 2.12.

## How to publish a new version
- Run `git checkout main` and `git pull` to be sure, all updates are fetched
- Edit version number in `python/src/deglib/__init__.py` to `x.y.z`
- Run `git add -A`, `git commit -m 'vx.y.z'` and `git tag -a vx.y.z -m 'vx.y.z'`
- Run `git push` and `git push origin --tags`
