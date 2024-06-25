# deglib: Python bindings for the Dynamic Exploration Graph

Python bindings for the c++ library Dynamic Exploration Graph used in the paper:
[Fast Approximate Nearest Neighbor Search with a Dynamic Exploration Graph using Continuous Refinement](https://arxiv.org/abs/2307.10479)

## Installation
TODO

### Using pip

### Compiling from source

## How to use
TODO

## Referencing c++ memory
TODO

## Internal Index vs External Label
- internal index is dense (no holes)
- external label is just going upwards
- only matters when removing elements

## TODO
- complete api
  - write some tests
  - add __repr__ functions
  - remove tqdm
- documentation
  - code comments
  - readme
- setup
  - make pypi/conda package
  - continuous integration
- copy argument for functions with numpy arrays
- check:
  - internal_index, external_label
  - always float feature vectors
- Questions:
  - What is RNG conform? -> Relative Neighborhood Graph
  - builder options (eps, k)
