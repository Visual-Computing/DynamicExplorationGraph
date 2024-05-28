# deglib: C++ library of the Dynamic Exploration Graph

Header only C++ library of the Dynamic Exploration Graph (DEG) used in the paper:
[Fast Approximate Nearest Neighbor Search with a Dynamic Exploration Graph using Continuous Refinement](https://arxiv.org/abs/2307.10479)

## How to use

### Prepare the data

Download and extract the data set files from the main [readme](../readme.md) file.

### Prerequisites

+ GCC 10.0+ with OpenMP
+ CMake 3.19+

IMPORTANT NOTE: this code uses AVX-256 instructions for fast distance computation, so your machine will need to support AVX-256 instructions, this can be checked using cat /proc/cpuinfo | grep avx2.

### Compile

1. Install Dependencies:
```
$ sudo apt-get install gcc-10 g++-10 cmake libboost-dev libgoogle-perftools-dev
```

On older systems setup gcc10 as the default compiler version:
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 1000 --slave /usr/bin/g++ g++ /usr/bin/g++-10
sudo update-alternatives --config gcc
```

2. Compile deglib

After cloning the git repository, rename `CMakePresets.json.sample` to `CMakePresets.json` and change the `DATA_PATH` variable inside of the file to represent a directory where the dataset is located.

```
git clone --recurse-submodules https://github.com/Visual-Computing/DynamicExplorationGraph.git
cd DynamicExplorationGraph/cpp/
mkdir build/ && cd build/
cmake -DCMAKE_BUILD_TYPE=Release --preset default ..
make -j
```

### Reproduce our results

To create a new graph, modify and run the `/benchmark/src/deglib_build_benchmark.cpp` file. Existing graphs can be tested with `/benchmark/src/deglib_anns_benchmark.cpp` and `/benchmark/src/deglib_explore_benchmark.cpp`.

Parameters:

|  Dataset  |  d  | k_ext | eps_ext | k_opt | eps_opt | i_opt |
|:---------:|:---:|:-----:|:-------:|:-----:|:-------:|:-----:|
| Audio     | 20  |  40   |   0.3   |  20   |  0.001  |   5   |
| Enron     | 30  |  60   |   0.3   |  30   |  0.001  |   5   |
| SIFT1M    | 30  |  60   |   0.2   |  30   |  0.001  |   5   |
| GloVe-100 | 30  |  30   |   0.2   |  30   |  0.001  |   5   |

## Pre-build Dynamic Exploration Graphs

The provided Dynamic Exploration Graphs are used in the experiments section of our paper.

|  Dataset  |  DEG  |
|:---------:|:---:|
| Audio     | [audio_192D_L2_DEG20.deg](https://static.visual-computing.com/paper/DEG/audio_192D_L2_DEG20.deg.gz) |
| Enron     | [enron_1369D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/enron_1369D_L2_DEG30.deg.gz) |
| SIFT1M    | [sift_128D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/sift_128D_L2_DEG30.deg.gz) |
| GloVe-100 | [glove_100D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/glove_100D_L2_DEG30.deg.gz) |

