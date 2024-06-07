# crEG: An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval

The continuous refining Exploration Graph (crEG) is a graph-based approximate nearest neighbor search (ANNS) algorithm. It can index static data set using an incremental extension, a continuous edge optimization. The resulting graph is highly efficient in regards to the queries per seconds in relation to the received recall rate. crEG delivers state-of-the-art performance for indexed and unindexed queries (query is not part of the index). 

## Benchmark datasets incl. exploration queries

| Data set  | Download                 | dimension | nb base vectors | nb query vectors | original website                                               |
|-----------|--------------------------|-----------|-----------------|------------------|----------------------------------------------------------------|
| Audio    |[audio.tar.gz](https://static.visual-computing.com/paper/DEG/audio.tar.gz)| 192       | 53,387       | 200           | [original website](https://www.cs.princeton.edu/cass/)             |           |
| Deep1M    |[deep1m.tar.gz](https://static.visual-computing.com/paper/DEG/deep1m.tar.gz)| 96       | 1,000,000       | 10,000           | [original website](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)             |
| SIFT1M    |[sift.tar.gz](https://static.visual-computing.com/paper/DEG/sift.tar.gz)| 128       | 1,000,000       | 10,000           | [original website](http://corpus-texmex.irisa.fr/)             |
| GloVe-100 | [glove-100.tar.gz](https://static.visual-computing.com/paper/DEG/glove-100.tar.gz) | 100       | 1,183,514       | 10,000           | [original website](https://nlp.stanford.edu/projects/glove/)   |

## Performance

For the four data set above crEG delievered the best ANNS and exploration efficiency amoung all the tested graph-based algorithms.

***NOTE:** All experiments where conduced single threaded on a Ryzen 2700x CPU, operating at a constant core clock speed of 4GHz, and 64GB of DDR4 memory running at 2133MHz.

**Approximate Nearest Neighbor Search**
![ANNS](figures/anns_qps_vs_recall.jpg)

**Exploratory Search (indexed queries and ideal start seed of for the graph search)**
![Exploration](figures/exploration_qps_vs_recall.jpg)

**Statistics of the graphs used in the experiments**
![Exploration](figures/indexing_stats.jpg)


## Reproduction

In order to reproduce our results please checkout the `\cpp\` directory and its readme file for more information about the parameters settings of the graphs.

## Pre-build continuous refining Exploration Graph 

The provided continuous refining Exploration Graph are used in the experiments section of our paper.

|  Dataset  |  EG  |  crEG  |
|:---------:|:----:|:------:|
| Audio     | [audio_192D_L2_EG20.zip](https://static.visual-computing.com/paper/DEG/audio_192D_L2_EG20.zip) | [audio_192D_L2_crEG20.zip](https://static.visual-computing.com/paper/DEG/audio_192D_L2_crEG20.zip) |
| Deep1M    | [deep1m_96D_L2_EG30.zip](https://static.visual-computing.com/paper/DEG/deep1m_96D_L2_EG30.zip) | [deep1m_96D_L2_crEG30.zip](https://static.visual-computing.com/paper/DEG/deep1m_96D_L2_crEG30.zip) |
| SIFT1M    | [sift_128D_L2_EG30.zip](https://static.visual-computing.com/paper/DEG/sift_128D_L2_EG30.zip) | [sift_128D_L2_crEG30.zip](https://static.visual-computing.com/paper/DEG/sift_128D_L2_crEG30.zip) |
| GloVe-100 | [glove_100D_L2_EG30.zip](https://static.visual-computing.com/paper/DEG/glove_100D_L2_EG30.zip) | [glove_100D_L2_crEG30.zip](https://static.visual-computing.com/paper/DEG/glove_100D_L2_crEG30.zip) |

## Reference

Please cite our work in your publications if it helps your research:

```
@inproceedings{crEG,
  author = {Hezel, Nico and Barthel, Kai Uwe and Schall, Konstantin and Jung, Klaus},
  title = {An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval},
  year = 2024,
  booktitle = {Proc. International Conference on Multimedia Retrieval (ICMR’24)},
  ublisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  booktitle = {Proceedings of the 2024 International Conference on Multimedia Retrieval},
  location = {Phuket, Thailand},
  series = {ICMR '24}
}
```

