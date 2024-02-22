# crEG: C++ library of the continuous refining Exploration Graph 

Header only C++ library of the continuous refining Exploration Graph.

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
After cloning the git repository, rename `cmake-variants.sample.yaml` to `cmake-variants.yaml` and change the `DATA_PATH` variable inside of the file to represent a directory where the dataset is located.

```
git clone --recurse-submodules https://github.com/Visual-Computing/DynamicExplorationGraph.git ExplorationGraph
cd ExplorationGraph/cpp/
mkdir build/ && cd build/
cmake -DCMAKE_BUIKD_TYPE=Release ..
make -j
```

### Reproduce our results

To create a new graph, modify and run the `/benchmark/src/deglib_build_benchmark.cpp` file. Existing graphs can be tested with `/benchmark/src/deglib_anns_benchmark.cpp` and `/benchmark/src/deglib_explore_benchmark.cpp`.

Construction Parameters:

|  Dataset  |  d  | k_ext | eps_ext | scheme |
|:---------:|:---:|:------:|:------:|:------:|
| Audio     | 20  |  40   |   0.1   |    D   |
| Deep1M    | 30  |  60   |   0.1   |    D   |
| SIFT1M    | 30  |  60   |   0.1   |    D   |
| GloVe-100 | 30  |  60   |   0.1   |    C   |

Refinement Parameters:

|  Dataset  | k_opt | eps_opt | i_opt | iterations |
|:---------:|:-----:|:-------:|:-----:|:----------:|
| Audio     |  20   |  0.001  |   5   |    20,000  |
| Deep1M    |  30   |  0.001  |   5   |   400,000  |
| SIFT1M    |  30   |  0.001  |   5   |   200,000  |
| GloVe-100 |  30   |  0.001  |   5   |   400,000  |


## Pre-build continuous refining Exploration Graph 

The provided continuous refining Exploration Graph are used in the experiments section of our paper.

|  Dataset  |  EG  |  crEG  |
|:---------:|:---:|
| Audio     | [audio_192D_L2_EG20.deg](https://static.visual-computing.com/paper/DEG/audio_192D_L2_EG20.deg.gz) | [audio_192D_L2_crEG20.deg](https://static.visual-computing.com/paper/DEG/audio_192D_L2_crEG20.deg.gz) |
| Deep1M    | [deep1m_96D_L2_EG30.deg](https://static.visual-computing.com/paper/DEG/deep1m_96D_L2_EG30.deg.gz) | [deep1m_96D_L2_crEG30.deg](https://static.visual-computing.com/paper/DEG/deep1m_96D_L2_crEG30.deg.gz) |
| SIFT1M    | [sift_128D_L2_EG30.deg](https://static.visual-computing.com/paper/DEG/sift_128D_L2_EG30.deg.gz) | [sift_128D_L2_crEG30.deg](https://static.visual-computing.com/paper/DEG/sift_128D_L2_crEG30.deg.gz) |
| GloVe-100 | [glove_100D_L2_EG30.deg](https://static.visual-computing.com/paper/DEG/glove_100D_L2_EG30.deg.gz) | [glove_100D_L2_crEG30.deg](https://static.visual-computing.com/paper/DEG/glove_100D_L2_crEG30.deg.gz) |

