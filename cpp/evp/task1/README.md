# EVP Benchmark Task 1: Modularized Benchmark Modes

This directory contains the modularized implementation of the benchmark modes originally consolidated in `deglib_evp_test.cpp`. These modularized modes strictly accept HDF5 input datasets (`.h5` or `.hdf5` extension) containing the `train` features and optionally the `allknn/knns` ground-truth.

Every mode supports two mutually exclusive output behaviours, controlled via CLI flags:

* **Recall mode** (default): loads the `allknn/knns` ground truth and prints `Recall@K` after the search.
* **Save mode** (`--no-recall --output <path>`): skips ground-truth loading, runs the search once, and writes the retrieved neighbor indices to a binary **ivecs** file (one row per query; each row: `uint32_t d` followed by `d × uint32_t` label values).

---

## Benchmark Modes Overview

### 1. `modi1.h` (FP16 Build, FP16 Explore)
* **Metric**: Graph uses `Metric::FP16InnerProduct`, exploration uses `Metric::FP16InnerProduct`.
* **Behavior**: Builds the Size-Bounded Graph using FP16 features directly and explores it via FP16 Inner Product similarity. Acts as a high-quality FP16 baseline.

### 2. `modi2.h` (EVP Linear Search)
* **Metric**: Exact comparison using `EvpBitsSimilarity::compare`.
* **Behavior**: Exact all-pairs linear search baseline. Quantizes all vectors to EVP bits and performs exact brute-force search. No graph structure is built.

### 3. `modi3.h` (EVP Build, EVP Explore)
* **Metric**: Graph uses `Metric::EvpBits`, exploration uses `Metric::EvpBits`.
* **Behavior**: Builds the graph with EVP-quantized features and explores it directly using EVP bits similarity. No reranking is performed.

### 4. `modi4.h` (EVP Build, EVP Explore, FP16 Rerank)
* **Metric**: Graph uses `Metric::EvpBits`, exploration uses `Metric::EvpBits`.
* **Behavior**: Quantizes FP16 features to EVP-bits to construct a Size-Bounded Graph. During search, it explores the EVP graph to retrieve candidates and then reranks the candidates using exact FP16 Inner Product distances.

### 5. `modi5.h` (EVP Build, FP16 External Search)
* **Metric**: Graph uses `Metric::EvpBits`, search uses `Metric::FP16InnerProduct`.
* **Behavior**: Builds the graph with EVP features. Then, copies the FP16 data and permutes it in-place using the graph's internal indexing order. A `ReadOnlyGraphExternal` is constructed that directly references the FP16 array without copies, allowing direct FP16 search over the graph topology.

### 6. `modi6.h` (EVP Build, FP16 Asymmetric Search)
* **Metric**: Graph uses `Metric::EvpBits`, search uses `Metric::FP16EvpAsymmetric`.
* **Behavior**: Builds the graph using EVP. Converts it to a `ReadOnlyGraph` with asymmetric float space. Performs search where the query is in FP16 representation and the database is represented using EVP bits in the graph.

### 7. `modi7.h` (EVP Build, FP16 Asymmetric Search, FP16 Rerank)
* **Metric**: Graph uses `Metric::EvpBits`, search uses `Metric::FP16EvpAsymmetric`, rerank uses `Metric::FP16InnerProduct`.
* **Behavior**: Same asymmetric search configuration as `modi6`, but with an added candidate reranking phase using exact FP16 Inner Product distances.

---

## Compilation

The modes are built into a single executable `deglib_evp_task1`.

Configure and build:
```powershell
cmake --preset "Visual Studio Community 2022"
cmake --build --preset "Visual Studio Community 2022 release" --target deglib_evp_task1
```

---

## Execution and CLI Options

Run the binary by providing the input H5 file path, the target mode (`modi1` - `modi7`), and optional arguments:
```powershell
./build/Visual Studio Community 2022/evp/Release/deglib_evp_task1.exe <hdf5_file_path> <mode> [options...]
```

### Options
| Option | Default | Description |
|---|---|---|
| `--threads <n>` | `6` | Number of parallel threads. |
| `--non-zeros <n>` | `512` | Non-zero coordinates kept during EVP quantization. |
| `--k-top <n>` | `15` | Number of nearest neighbors to retrieve and evaluate. |
| `--k-graph <n>` | `32` | Graph degree / edges per vertex in the SizeBoundedGraph. |
| `--k-ext <n>` | `32` | Graph builder degree during regular extension. |
| `--eps-ext <f>` | `0.001` | Graph builder entry search expansion coefficient. |
| `--max-dist <n>` | `200` | Maximum number of distance computations per query during graph exploration/search. Higher values improve recall at the cost of speed. |
| `--evpK <n>` | `0` (auto) | Overrides the internal candidate list size for graph search. Defaults: `50` for asymmetric-rerank (modi7), `max(k_top, max_dist)` for EVP-rerank (modi4). |
| `--no-recall` | — | Skip ground-truth loading and recall computation. Must be combined with `--output`. |
| `--output <path>` | — | Write retrieved Top-K labels to a binary **ivecs** file (one row per query). |

### Examples

```powershell
# Recall mode with default max_dist=200
.\deglib_evp_task1.exe dataset.h5 evp

# Recall mode with a specific max_dist to trade speed for accuracy
.\deglib_evp_task1.exe dataset.h5 evp --max-dist 400

# Save mode: run once, write results to disk (no recall computed)
.\deglib_evp_task1.exe dataset.h5 evp --no-recall --output results.ivecs --max-dist 200
```
