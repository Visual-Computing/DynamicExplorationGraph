# deglib: Python bindings for the Dynamic Exploration Graph

Python bindings for the C++ library Dynamic Exploration Graph used in the paper:
[Fast Approximate Nearest Neighbor Search with a Dynamic Exploration Graph using Continuous Refinement](https://arxiv.org/abs/2307.10479)

## Installation
TODO

### Using pip

### Compiling from source

## How to use
TODO

### Referencing c++ memory
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
