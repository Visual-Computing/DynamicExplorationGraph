# Maximum Inner Product Search (MIPS) Example

This example demonstrates how to build and query a **Maximum Inner Product Search (MIPS)** graph using **Dynamic Exploration Graph (DEG)** in Python, based on Task 2 (`mode5_flas`) from the **SISAP 2026 Indexing Challenge**.

## Problem & Algorithmic Approach

Standard graph-based nearest neighbor search algorithms (e.g. DEG, HNSW) are optimized for metric spaces such as Euclidean distance ($L_2$). For Maximum Inner Product Search (MIPS), the goal is to find database vectors $x_i$ maximizing $\langle q, x_i \rangle$ for a query vector $q$.

We use a 5-step pipeline that reduces MIPS to $L_2$ graph search while retaining high query efficiency:

1. **$(d+1)$-Dimensional $L_2$ Transformation**:
   Each database vector $x_i \in \mathbb{R}^d$ is mapped to $x'_i = [x_i, \sqrt{M^2 - \|x_i\|^2}] \in \mathbb{R}^{d+1}$ where $M^2 = \max_j \|x_j\|^2$. 
   Because all transformed vectors $x'_i$ have equal norm $\|x'_i\| = M$, minimizing $\|q' - x'_i\|^2$ under $L_2$ distance (with $q' = [q, 0]$) is strictly equivalent to maximizing the inner product $\langle q, x_i \rangle$.

2. **1D FLAS Pre-Sorting**:
   The transformed vectors are pre-sorted using Fast Linear Assignment Sorting (`deglib.optimization.presort`) with `Metric.FP32_L2` to optimize memory layout and cache locality during graph construction.

3. **Graph Construction in FP32 $L_2$ Space**:
   A `DynamicExplorationGraph` is constructed using `GraphBuilder` in $(d+1)$-dimensional `Metric.FP32_L2` space ($K_{\text{graph}} = 32, K_{\text{ext}} = 64$).

4. **FP16 Feature Swapping & `ReadOnlyGraph`**:
   The original $d$-dimensional FP32 database vectors are converted to 16-bit half-precision floats (`deglib.floats_to_fp16`). The graph topology built in step 3 is converted to a `ReadOnlyGraph` with `Metric.FP16_InnerProduct` space by passing the FP16 feature buffer (`graph.to_readonly(feature_space=..., custom_features=...)`).

5. **SIMD FP16 Inner Product Search**:
   Query vectors are converted to FP16 (`deglib.floats_to_fp16`) and searched on the `ReadOnlyGraph` using fast SIMD FP16 inner product distance routines ($\varepsilon_{\text{search}} = 0.18$, max distance evaluation budget sweep: `6000, 6500, 7000, 7500, 8000, 9000`).

---

## Dataset

This example benchmarks on the **SISAP 2026 `llama-dev`** dataset (Llama embeddings):
- **Repository**: [`SISAP-Challenges/SISAP2026`](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026)
- **Files**: `llama-dev/llama-dev.h5` and `config.json`
- **Data Structure**:
  - `train`: Floating-point database vectors ($N$ vectors, $d$ dimensions)
  - `test/queries`: Query vectors
  - `test/knns`: Precomputed ground-truth top-$K$ nearest neighbor indices

The script automatically downloads `llama-dev.h5` into `~/.cache/deg_datasets/llama-dev/` on the first run if not already present.

---

## Setup & Running with `uv`

### 1. Build Library & Synchronize Environment

```bash
cd examples/mips/
uv sync
```

> **Note:** `uv sync` automatically compiles the latest `deglib` C++ pybind11 extension from `../../python` into the local environment.

If C++ code or pybind11 bindings were modified, force rebuilding the extension:
```bash
uv sync --reinstall-package deglib
```

### 2. Run MIPS Benchmark & Plot Trade-off Curve

```bash
uv run main.py
```

The script displays an interactive Matplotlib trade-off plot (Recall vs Search Time in ms) with `max_dist` data point annotations.

---

## Command-Line Options

```
usage: main.py [-h] [--dataset DATASET] [--k-graph K_GRAPH] [--k-ext K_EXT]
               [--eps-ext EPS_EXT] [--eps-search EPS_SEARCH] [--no-flas]
               [--flas-decay FLAS_DECAY] [--prune-worst PRUNE_WORST]
               [--threads THREADS] [--output-plot OUTPUT_PLOT] [--no-show]

options:
  --dataset DATASET               Path to HDF5 dataset file or 'llama-dev' (default: llama-dev)
  --k-graph K_GRAPH               Graph degree per vertex (default: 32)
  --k-ext K_EXT                   Builder search size parameter (default: 64)
  --eps-ext EPS_EXT               Builder search expansion factor (default: 0.2)
  --eps-search EPS_SEARCH         Epsilon search factor (default: 0.18)
  --no-flas                       Disable FLAS 1D pre-sorting
  --flas-decay FLAS_DECAY         FLAS neighborhood radius decay rate (default: 0.9)
  --prune-worst PRUNE_WORST       Number of worst neighbors to replace with self-loops (default: 0)
  --threads THREADS               Number of parallel threads (default: 8)
  --output-plot OUTPUT_PLOT       Path to save trade-off curve PNG
  --no-show                       Disable GUI plot display
```
