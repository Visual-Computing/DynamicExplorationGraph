# deglib: Python bindings for the Dynamic Exploration Graph

Python bindings for the C++ library Dynamic Exploration Graph used in the paper:
[Fast Approximate Nearest Neighbor Search with a Dynamic Exploration Graph using Continuous Refinement](https://arxiv.org/abs/2307.10479)

## Installation
TODO

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
git clone -b feat/python_bindings --recurse-submodules git@github.com:Visual-Computing/DynamicExplorationGraph.git
cd DynamicExplorationGraph
```

**Install the Package (easy)**
```shell
cd cpp/python-bindings
pip install .
```
This will compile the C++ code and install deglib into your virtual environment, so it may take a while.

**Install the Package (manually for development)**
```shell
cd cpp

# compile deglib C++ sources, which will also create the python bindings
# shared-object-file (see build/python-bindings/deglib_cpp.cpython-3*.so
mkdir -p build
cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
(cd build; make -j $(nproc))  # if you encounter errors, see Troubleshooting

# link deglib_cpp in your virtual environment
cd python-bindings
./scripts/link-lib.sh  # create symlink in virtual env (unix only)

# install python dependencies
pip install -r requirements.txt
```
This will compile the C++ code and link that compiled shared-object-file into your python environment.
The python-deglib library will only be available when executing python3 from the `python-bindings` directory.

### Troubleshooting

`pybind11/typing.h:104:58: error: ‘copy_n’ is not a member of ‘std’`

This is a pybind11 bug, that occurs when compiling it with gcc-14. Change the pybind version to 2.12:
```shell
cd cpp/python-bindings/external/pybind11
git checkout v2.12
```

## Examples
### Loading Data
To load a dataset formatted like the [TexMex-Datasets](http://corpus-texmex.irisa.fr/):
```python
import deglib
import numpy as np

dataset: np.ndarray = deglib.repository.fvecs_read("path/to/data.fvecs")
num_samples, dims = dataset.shape
print(num_samples, dims)
```
The dataset is a numpy array with shape (N, D), where N is the number of feature
vectors and D is the number of dimensions of each feature vector.

### Building a Graph

```python
import deglib.graph

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
TODO

## Naming
- Vertex = Feature Vector

### Internal Index vs External Label
- internal index is dense (no holes)
- external label is user defined identifier
- only matters when removing elements

## Limitations
- only float spaces
- ResultSet ordering

## TODO
- documentation
  - readme
- setup
  - make pypi package
  - continuous integration
- check:
  - internal_index, external_label
  - always float feature vectors (is ok)
- Questions:
  - What is RNG conform? -> Relative Neighborhood Graph
  - builder options (eps, k)
- add License
- remove test functions from deglib_cpp.cpp
- pybind11 v2.12 branch
- use nanobind

- Python Packaging
  - Try packaging with cmake: https://github.com/pybind/cmake_example/tree/master, pybind-example: https://github.com/pybind/pybind11/blob/master/pyproject.toml, scikit-examples: https://github.com/scikit-build/scikit-build-sample-projects/tree/main/projects/hello-cpp
    - Add option to set instruction set to avx2 (for wheels)
    - try with build directory; otherwise move setup.py/pyproject into cpp/
  - nanobind
  - create GitHub workflow for wheels
    - build wheel with avx2
    - warning for users who could use avx512 (+ install instructions)
    - error for users who don't support avx2 (+ install instructions)
  - 3 libs in a package for different cpu instructions (maybe ignore)