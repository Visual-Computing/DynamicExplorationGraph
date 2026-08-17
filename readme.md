# Dynamic Exploration Graph (DEG)

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](cpp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](python/)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-green.svg)](https://dynamic-exploration-graph.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

The **Dynamic Exploration Graph (DEG)** is a high-throughput, graph-based algorithm for Approximate Nearest Neighbor Search (ANNS) and exploratory search in high-dimensional vector spaces.

DEG efficiently handles both static and dynamic streaming datasets via three continuous algorithms:
1. **Incremental Extension**: Fast insertion of new vertices into the graph topology.
2. **Continuous Edge Optimization**: Constant refinement and local edge swaps to maintain optimal search properties.
3. **Vertex Deletion**: Dynamic removal of elements while preserving graph connectivity and regular degree.

---

## Repository Structure

```
DynamicExplorationGraph/
├── cpp/          # High-performance C++20 Header-Only library, CMake Presets, Tests & Benchmarks
├── python/       # Python Bindings (deglib), Pytest Suite & Wheel Build Configuration
├── examples/     # Ready-to-run Python examples (knng, dynamic_data, static_data, mips)
├── java/         # Java implementation & Benchmarks
└── docs/         # Sphinx / ReadTheDocs Documentation
```

---

## Quickstart

### Python

Install `deglib` via pip:

```bash
pip install deglib
```

Build an index and query nearest neighbors:

```python
import numpy as np
import deglib

N_SAMPLES, DIMS = 10_000, 128

# Generate example dataset and query
data = np.random.random((N_SAMPLES, DIMS)).astype(np.float32)
query = np.random.random(DIMS).astype(np.float32)

# Build index directly from data
graph = deglib.builder.build_from_data(data)

# Query top-k nearest neighbors
indices, distances = graph.search(query, k=16, eps=0.1)

print("Nearest neighbor indices:", indices)
print("Distances:", distances)
```

For more Python examples, check the [examples/](examples/) directory or read the [Official Documentation](https://dynamic-exploration-graph.readthedocs.io/en/latest/tutorials/quickstart.html).

---

### C++20

`deglib` is a header-only C++20 library with zero mandatory runtime dependencies.

```cpp
#include <deglib/deglib.h>
#include <iostream>
#include <vector>
#include <random>

int main() {
    const uint32_t num_vectors = 1000;
    const uint32_t dims = 128;

    // 1. Generate example feature dataset
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> dataset(num_vectors * dims);
    for (auto& val : dataset) val = dist(rng);

    // 2. Build graph index directly from data
    auto graph = deglib::build_from_data(
        std::span<const float>(dataset), 
        dims, 
        /*labels=*/{}, 
        /*edges_per_vertex=*/32, 
        deglib::distances::Metric::FP32_L2
    );

    // 3. Query k-nearest neighbors
    std::vector<float> query(dims);
    for (auto& val : query) val = dist(rng);

    auto results = graph.search(std::span<const float>(query), /*k=*/10, /*eps=*/0.1f);

    std::cout << "Top nearest neighbors:\n";
    for (const auto& match : results) {
        std::cout << "  Label: " << match.getIdentifier()
                  << " | Distance: " << match.getDistance() << "\n";
    }
}
```

For full C++ build instructions, CMake presets, and architecture details, refer to the [cpp/ README](cpp/readme.md).

---

## Publications & Releases

- **[2025/01/09]** Our paper [Dynamic Exploration Graph: A Novel Approach for Efficient Nearest Neighbor Search in Evolving Multimedia Datasets](https://link.springer.com/chapter/10.1007/978-981-96-2054-8_25) was presented at MMM 2025.
- **[2024/05/01]** Our paper [An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval](https://doi.org/10.1145/3652583.3658117) was presented at ACM ICMR 2024 (archived in [crEG branch](https://github.com/Visual-Computing/DynamicExplorationGraph/tree/crEG)).
- **[2023/07/19]** First version of DEG released! See our preprint [Fast Approximate nearest neighbor search with the Dynamic Exploration Graph using continuous refinement](https://arxiv.org/abs/2307.10479).

---

## Datasets

The following standard datasets are used for benchmarking and evaluation:

| Dataset   | Archive Name       | Dimension | Base Vectors | Query Vectors | Reference |
|-----------|--------------------|-----------|--------------|---------------|-----------|
| Audio     | `audio.tar.gz`     | 192       | 53,387       | 200           | [Princeton CASS](https://www.cs.princeton.edu/cass/) |
| Enron     | `enron.tar.gz`     | 1369      | 94,987       | 200           | [CMU Enron](https://www.cs.cmu.edu/~enron/) |
| SIFT1M    | `sift.tar.gz`      | 128       | 1,000,000    | 10,000        | [Texmex](http://corpus-texmex.irisa.fr/) |
| DEEP1M    | `deep1m.tar.gz`    | 96        | 1,000,000    | 10,000        | [PPUDA](https://github.com/facebookresearch/ppuda) |
| GloVe-100 | `glove-100.tar.gz` | 100       | 1,183,514    | 10,000        | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |

> [!NOTE]
> When executing the benchmarks in [cpp/](cpp/) or the [Python examples](examples/), datasets are automatically downloaded and prepared in the configured data directory on first run.

### Pre-built Graphs

| Dataset   | DEG Graph File |
|-----------|----------------|
| SIFT1M    | [sift_128D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/sift_128D_L2_DEG30.zip) |
| DEEP1M    | [deep1m_96D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/deep1m_96D_L2_DEG30.zip) |
| GloVe-100 | [glove_100D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/glove_100D_L2_DEG30.zip) |

---

## Performance

> [!NOTE]
> Experiments were conducted single-threaded on an AMD Ryzen 2700X CPU (4 GHz core clock, 64 GB DDR4 RAM @ 2133 MHz).

**Approximate Nearest Neighbor Search (ANNS)**  
![ANNS](figures/anns_qps_vs_recall.jpg)

**Exploratory Search (Indexed Queries)**  
![Exploration](figures/exploration_qps_vs_recall.jpg)

---

## Citation

If DEG or crEG helps your research, please cite our publications:

**Dynamic Exploration Graph (MMM 2025)**:
```bibtex
@article{Hezel2025,
  author    = {Hezel, Nico and Barthel, Uwe Kai and Schilling, Bruno and Schall, Konstantin and Jung, Klaus},
  title     = {Dynamic Exploration Graph: A Novel Approach for Efficient Nearest Neighbor Search in Evolving Multimedia Datasets},
  booktitle = {MultiMedia Modeling},
  publisher = {Springer Nature},
  pages     = {333--347},
  isbn      = {978-981-96-2054-8},
  year      = {2025}
}
```

**Continuous Refining Exploration Graph (ACM ICMR 2024)**:
```bibtex
@inproceedings{Hezel2024,
  author    = {Hezel, Nico and Barthel, Uwe Kai and Schall, Konstantin and Jung, Klaus},
  title     = {An Exploration Graph with Continuous Refinement for Efficient Multimedia Retrieval},
  booktitle = {Proceedings of the 2024 International Conference on Multimedia Retrieval},
  publisher = {Association for Computing Machinery},
  pages     = {657--665},
  isbn      = {9798400706196},
  doi       = {10.1145/3652583.3658117},
  series    = {ICMR '24},
  year      = {2024}
}
```
