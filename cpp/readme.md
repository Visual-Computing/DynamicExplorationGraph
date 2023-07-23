# deglib: C++ library of the Dynamic Exploration Graph

Header only C++ library of the Dynamic Exploration Graph (DEG) used in the paper:
[Fast Approximate Nearest Neighbor Search with a Dynamic Exploration Graph using Continuous Refinement](https://arxiv.org/abs/2307.10479)

## How to use

### Prepare the data

Download and extract the data set files from the main [readme](../readme.md) file.

### Compile

After cloning the git repository, rename `cmake-variants.sample.yaml` to `cmake-variants.yaml` and change the `DATA_PATH` variable inside of the file to represent a directory where the dataset is located.

```
git clone --recurse-submodules https://github.com/Visual-Computing/DynamicExplorationGraph.git
cd cpp/
mkdir build/ && cd build/
cmake -DCMAKE_BUIKD_TYPE=Release ..
make -j
```

### Reproduce our results

Modify and run the `/benchmark/src/deglib_build_benchmark.cpp` file to create a new graph. Existing graphs can be tested with `/benchmark/src/deglib_anns_benchmark.cpp` and `/benchmark/src/deglib_explore_benchmark.cpp`.

Parameters:

|  Dataset  |  d  | k_ext | eps_ext | k_opt | eps_opt | i_opt |
|:---------:|:---:|:------:|:------:|:-----:|:-------:|:-----:|
| Audio     | 20  |  40   |   0.3   |  20   |  0.001  |   5   |
| Enron     | 30  |  60   |   0.3   |  30   |  0.001  |   5   |
| SIFT1M    | 30  |  60   |   0.2   |  30   |  0.001  |   5   |
| GloVe-100 | 30  |  30   |   0.2   |  30   |  0.001  |   5   |

## Pre-build Dynamic Exploration Graphs

Here we provide pre-built Dynamic Exploration Graphs used in our papar's experiments.

|  Dataset  |  DEG  |
|:---------:|:---:|
| Audio     | [audio_192D_L2_DEG20.deg](https://static.visual-computing.com/paper/DEG/audio_192D_L2_DEG20.deg.gz) |
| Enron     | [enron_1369D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/enron_1369D_L2_DEG30.deg.gz) |
| SIFT1M    | [sift_128D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/sift_128D_L2_DEG30.deg.gz) |
| GloVe-100 | [glove_100D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/glove_100D_L2_DEG30.deg.gz) |

