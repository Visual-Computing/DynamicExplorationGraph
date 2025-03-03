# DEG: Fast Approximate Nearest Neighbor Search

The Dynamic Exploration Graph (DEG) is a graph-based algorithm for approximate nearest neighbor search (ANNS). It indexes both static and dynamic datasets using three algorithms: incremental extension, continuous edge optimization, and vertex deletion. The resulting graph demonstrates high efficiency in terms of queries per second relative to the achieved recall rate. DEG provides state-of-the-art performance for both indexed and unindexed queries (where the query is not part of the index).

## Usage
For a short introduction on how to use deglib for vector search, see our [Python Examples](python/README.md#examples).

## Release

- [2025/01/09] The latest iteration of DEG uses a more efficient way of removing and adding a vertex. More details can be found in our new paper [Dynamic Exploration Graph: A Novel Approach for Efficient Nearest Neighbor Search in Evolving Multimedia Datasets](https://link.springer.com/chapter/10.1007/978-981-96-2054-8_25).
- [2024/05/01] Our paper [An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval](https://doi.org/10.1145/3652583.3658117) is accepted by ICMR2024 as **oral presentation**
- [2023/12/02] The new continuous refining Exploration Graph (crEG) containing a more efficient and thread-safe way to extend DEG. Currently found in the [crEG branch](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/crEG) of this repository.
- [2023/07/19] First version of Dynamic Exploration Graph is out! For more details please refere to our paper: 
[Fast Approximate nearest neighbor search with the Dynamic Exploration Graph using continuous refinement](https://arxiv.org/abs/2307.10479)

## Reproduction

The following files contain the datasets used in our paper. Including exploration queries and ground truth data.

| Data set  | Download                                                                           | dimension | nb base vectors | nb query vectors | original website                                               |
|-----------|------------------------------------------------------------------------------------|-----------|-----------------|------------------|----------------------------------------------------------------|
| Audio    | [audio.tar.gz](https://static.visual-computing.com/paper/DEG/audio.tar.gz)         | 192       | 53,387       | 200           | [original website](https://www.cs.princeton.edu/cass/)             |
| Enron    | [enron.tar.gz](https://static.visual-computing.com/paper/DEG/enron.tar.gz)         | 1369      | 94,987       | 200           | [original website](https://www.cs.cmu.edu/~enron/)             |
| SIFT1M    | [sift.tar.gz](https://static.visual-computing.com/paper/DEG/sift.tar.gz)           | 128       | 1,000,000       | 10,000           | [original website](http://corpus-texmex.irisa.fr/)             |
| DEEP1M    | [deep1m.tar.gz](https://static.visual-computing.com/paper/DEG/deep1m.tar.gz)     | 96        | 1,000,000       | 10,000           | [original website](https://github.com/facebookresearch/ppuda)             |
| GloVe-100 | [glove-100.tar.gz](https://static.visual-computing.com/paper/DEG/glove-100.tar.gz) | 100       | 1,183,514       | 10,000           | [original website](https://nlp.stanford.edu/projects/glove/)   |

In order to reproduce the results in our older papers, please visit the corresponding github branch.
- [main branch](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/main) for "Dynamic Exploration Graph: A Novel Approach for Efficient Nearest Neighbor Search in Evolving Multimedia Datasets"
- [crEG branch](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/crEG) for "An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval"
- [arxiv branch](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/arxiv) for "Fast Approximate nearest neighbor search with the Dynamic Exploration Graph using continuous refinement"




## Performance

***NOTE:** All experiments where conduced single threaded on a Ryzen 2700x CPU, operating at a constant core clock speed of 4GHz, and 64GB of DDR4 memory running at 2133MHz.

**Approximate Nearest Neighbor Search**
![ANNS](figures/anns_qps_vs_recall.jpg)

**Exploratory Search (indexed queries)**
![Exploration](figures/exploration_qps_vs_recall.jpg)

## Reference

Please cite our work in your publications if it helps your research:

Dynamic Exploration Graph
```
@article{Hezel2025,
  author = {Hezel, Nico and Barthel, Uwe Kai and Schilling, Bruno and Schall, Konstantin and Jung, Klaus},
  title = {Dynamic Exploration Graph: A Novel Approach for Efficient Nearest Neighbor Search in Evolving Multimedia Datasets},
  booktitle={MultiMedia Modeling},
  publisher={Springer Nature},
  pages={333--347},
  isbn={978-981-96-2054-8},
  year = 2025
}
```

continuous refining Exploration Graph
```
@article{Hezel2024,
  author = {Hezel, Nico and Barthel, Uwe Kai and Schall, Konstantin and Jung, Klaus},
  title = {An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval},
  booktitle = {Proceedings of the 2024 International Conference on Multimedia Retrieval},
  year = {2024},
  isbn = {9798400706196},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3652583.3658117},
  pages = {657–665},
  series = {ICMR '24}
}
```
