# DEG: Fast Approximate nearest neighbor search with the Dynamic Exploration Graph using continuous refinement

The Dynamic Exploration Graph (DEG) is a graph-based approximate nearest neighbor search (ANNS) algorithm. It can index static and dynamic data set using an incremental extension, a continuous edge optimization and a deletion algorithm. The resulting graph is highly efficient in regards to the queries per seconds in relation to the received recall rate. DEG delivers state-of-the-art performance for indexed and unindexed queries (query is not part of the index). For more details please refere to our paper:
[Fast Approximate nearest neighbor search with the Dynamic Exploration Graph using continuous refinement](https://arxiv.org/abs/2307.10479)

## Benchmark datasets incl. exploration queries

| Data set  | Download                 | dimension | nb base vectors | nb query vectors | original website                                               |
|-----------|--------------------------|-----------|-----------------|------------------|----------------------------------------------------------------|
| Audio    |[audio.tar.gz](http://static.visual-computing.com/paper/DEG/audio.tar.gz)| 192       | 53,387       | 200           | [original website](https://www.cs.princeton.edu/cass/)             |
| Enron    |[enron.tar.gz](http://static.visual-computing.com/paper/DEG/enron.tar.gz)| 1369       | 94,987       | 200           | [original website](https://www.cs.cmu.edu/~enron/)             |
| SIFT1M    |[sift.tar.gz](http://static.visual-computing.com/paper/DEG/sift.tar.gz)| 128       | 1,000,000       | 10,000           | [original website](http://corpus-texmex.irisa.fr/)             |
| GloVe-100 | [glove-100.tar.gz](http://static.visual-computing.com/paper/DEG/glove-100.tar.gz) | 100       | 1,183,514       | 10,000           | [original website](https://nlp.stanford.edu/projects/glove/)   |

## Performance

In order to reproduce our results please checkout the `\cpp\` directory and its readme file. For the four data set above DEG delievered the best ANNS and exploration efficiency amoung all the tested graph-based algorithms.

***NOTE:** All experiments where conduced single threaded on a Ryzen 2700x CPU, operating at a constant core clock speed of 4GHz, and 64GB of DDR4 memory running at 2133MHz.

**Approximate Nearest Neighbor Search**
![ANNS](figures/anns_qps_vs_recall.jpg)

**Exploratory Search (indexed queries)**
![Exploration](figures/exploration_qps_vs_recall.jpg)

## Reference

Please cite our work in your publications if it helps your research:

```
@article{Hezel2023,
  author = {Hezel, Nico and Barthel, Uwe Kai and Schall, Konstantin and Jung, Klaus},
  ee = {https://arxiv.org/abs/2307.10479},
  journal = {CoRR},
  title = {Fast Approximate nearest neighbor search with the Dynamic Exploration Graph using continuous refinement.},
  volume = {abs/2307.10479},
  year = 2023
}
```

