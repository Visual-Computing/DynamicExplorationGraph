# DEG C++ Benchmark Suite

This directory contains the native C++ benchmark tools used to reproduce the experimental results and performance figures in the **Dynamic Exploration Graph (DEG)** paper.

---

## Overview of Benchmark Tools

| Executable | Purpose | Paper Reference |
| :--- | :--- | :--- |
| **`bench_static_data`** | Full static graph construction (AddAll), graph topology analysis, ANNS recall vs. QPS evaluation, and exploratory search across standard benchmarks. | *Section: Static Graph Evaluation & ANNS / Exploration Search* |
| **`bench_dynamic_data`** | Evaluates 3 dynamic streaming scenarios with continuous insertions and deletions (`AddHalf`, `AddHalfRemoveAndAddOneAtATime`, `AddAllRemoveHalf`). | *Section: Dynamic Graph Evaluation & Streaming Updates* |
| **`bench_edge_optimization`** | Evaluates edge swapping and continuous graph topology optimization on regular graphs over iterations. | *Section: Edge Optimization & Local Intrinsic Dimensionality (LID)* |
| **`bench_flas_presort`** | Evaluates the impact of FLAS (Fast Linear Assignment Sorter) 1D pre-sorting on build speed, cache locality, and query QPS. | *Section: Layout Optimization & FLAS Pre-Sorting* |

---

## Supported Datasets

The benchmark tools support standard ANN benchmark datasets with automated downloading and caching:

| Dataset | Dimension | Base Vectors | Queries | Metric | Target LID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SIFT1M** | 128 | 1,000,000 | 10,000 | `FP32_L2` | LowLID |
| **DEEP1M** | 96 | 1,000,000 | 10,000 | `FP32_L2` | LowLID |
| **GloVe-100** | 100 | 1,183,514 | 10,000 | `FP32_L2` | HighLID |
| **Audio** | 192 | 53,387 | 200 | `FP32_L2` | LowLID ($K=20$) |
| **Enron** | 1,369 | 94,987 | 200 | `FP32_L2` | LowLID ($K=30$) |

---

## Building the Benchmarks

The benchmark executables are built as part of the C++ release configuration.

```bash
cd cpp/

# 1. Configure with your platform preset (Windows, Linux, or macOS)
cmake --preset windows-msvc-avx2  # or linux-gcc-avx2 / macos-clang-avx2

# 2. Build Release binaries
cmake --build --preset windows-msvc-avx2-release
```

The compiled binaries will be located in:
* **Windows:** `cpp/build/windows-msvc-avx2/bench/Release/`
* **Linux:** `cpp/build/linux-gcc-avx2/bench/`
* **macOS:** `cpp/build/macos-clang-avx2/bench/`

---

## Usage Guide

### Dataset Directory
By default, benchmarks look for or download datasets to `./data` or the path specified by the `DEG_DATA_PATH` environment variable. You can also supply the `--data-path` argument explicitly:

```bash
export DEG_DATA_PATH="/path/to/data"
```

---

### 1. `bench_static_data`
Builds a complete exploration graph from scratch (or loads a cached `.deg` file) and evaluates ANNS Recall vs. QPS across multiple $\epsilon$ exploration margins.

```bash
# Run on SIFT1M (default)
./bench_static_data sift1m

# Run on DEEP1M with custom thread count and AVX-512
./bench_static_data deep1m --threads 16 --instruction avx512

# Run across all datasets sequentially
./bench_static_data all --data-path /data/ann
```

**Options:**
* `[dataset]`: `sift1m`, `deep1m`, `glove`, `audio`, `enron`, or `all`
* `--data-path, -d <path>`: Root directory for datasets and graphs.
* `--force-rebuild`: Forces graph rebuilding even if a cached `.deg` exists.
* `--instruction <inst>`: SIMD instruction set (`auto`, `avx512`, `avx2`, `scalar`).
* `--threads <count>`: Thread count for graph construction.

---

### 2. `bench_dynamic_data`
Simulates streaming workloads and incremental graph maintenance:
1. **`AddHalf`**: Builds initial graph with 50% of the dataset, then evaluates search.
2. **`AddHalfRemoveAndAddOneAtATime`**: Simulates a sliding window by alternating removing an existing node and inserting a new node one-by-one.
3. **`AddAllRemoveHalf`**: Inserts all items, then deletes 50% of the nodes, testing graph decay resistance and vertex recycling.

```bash
./bench_dynamic_data sift1m --data-path /data/ann
```

---

### 3. `bench_edge_optimization`
Evaluates edge refinement and quality improvement on regular graphs over iterations:

```bash
./bench_edge_optimization audio --threads 8
```

**Options:**
* `--log-after <iters>`: Number of edge-swapping iterations between benchmark checkpoints (default: `100,000`).
* `--max-iterations <iters>`: Maximum total swap iterations (default: `1,000,000`).

---

### 4. `bench_flas_presort`
Compares graph build time, cache locality, and resulting ANNS search performance between:
1. Raw / unorganized input ordering.
2. FLAS (Fast Linear Assignment Sorter) 1D pre-sorted ordering.

```bash
./bench_flas_presort sift1m --decay 0.9 --threads 12
```

---

## Log Output & Graph Cache

All benchmark tools generate structured log files and graph caches in:
```
<DEG_DATA_PATH>/<dataset>/deg/
├── <dim>D_<metric>_K<k>_AddK<k_ext>Eps<eps>_<lid>.deg        # Graph binary
└── <dim>D_<metric>_K<k>_AddK<k_ext>Eps<eps>_<lid>.deg.log    # Detailed QPS, Recall & Topology logs
```
