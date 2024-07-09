# deglib: Python bindings for the Dynamic Exploration Graph

Python bindings for the C++ library Dynamic Exploration Graph used in the paper:
[Fast Approximate Nearest Neighbor Search with a Dynamic Exploration Graph using Continuous Refinement](https://arxiv.org/abs/2307.10479)

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
mkdir install_dir && cd install_dir
# TODO: "-b feat/python_bindings" not necessary after merge
git clone -b feat/python_bindings git@github.com:Visual-Computing/DynamicExplorationGraph.git
cd DynamicExplorationGraph
```

**Install the Package from Source**
```shell
cd python
pip install setuptools pybind11 build
python3 setup.py copy_build_files  # copy c++ library to ./lib/
pip install .
```
This will compile the C++ code and install deglib into your virtual environment, so it may take a while.

**Building Packages**

Build packages (sdist and wheels):
```shell
python3 -m build
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
import deglib

graph = deglib.builder.EvenRegularGraphBuilder.build_from_data(dataset, edges_per_vertex=32)
graph.save_graph("/path/to/graph.deg")
rd_graph = deglib.graph.load_readonly_graph("/path/to/graph.deg")
```

### Searching the Graph
```python
query = np.random.random((dims,)).astype(np.float32)
result = graph.search(query, eps=0.1, k=10)  # get 10 nearest features to query
for r in result:
    print(r.get_internal_index(), r.get_distance())
```

### Referencing C++ memory
Consider the following example:
```python
feature_vector = graph.get_feature_vector(42)
del graph
print(feature_vector)
```
This will crash as `feature_vector` is holding a reference to memory that is owned by `graph`. This can lead to segmentation faults.
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
TODO

### Relative Neighborhood Graph / RNG-conform
TODO

## Limitations
- The python wrapper at the moment only supports `float32` feature vectors.
- The elements of a `ResultSet` are not sorted by distance.

## Troubleshooting

### BuildError: `pybind11/typing.h:104:58: error: ‘copy_n’ is not a member of ‘std’`

This is a pybind11 bug, that occurs when compiling it with gcc-14. Change the pybind version to 2.12.
