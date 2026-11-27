# Maximum Inner Product Search (MIPS) Example

This example demonstrates how to build and query a **Maximum Inner Product Search (MIPS)** graph using **Dynamic Exploration Graph (DEG)** in Python, based on Task 2 (`mode5_flas`) from the **SISAP 2026 Indexing Challenge**.

> [!NOTE]
> All runtime and throughput results published in our papers were evaluated using the native C++ implementation (`cpp/`). Due to language bindings and runtime dynamics, Python executions may have a small overhead.

## Problem & Algorithmic Approach

Standard graph-based nearest neighbor search algorithms (e.g. DEG, HNSW) are optimized for metric spaces such as Euclidean distance ($L_2$). For Maximum Inner Product Search (MIPS), the goal is to find database vectors $x_i$ maximizing $\langle q, x_i \rangle$ for a query vector $q$.

Using the high-level `deglib` API (`deglib.build(..., metric="ip")` and `deglib.searcher(..., compress="16bit")`), the entire MIPS pipeline is handled automatically:

1. **$(d+1)$-Dimensional $L_2$ Transformation & 1D FLAS Pre-Sorting**:
   `deglib.build(train_vectors, metric="ip", presort=True, ...)` automatically lifts each database vector $x_i \in \mathbb{R}^d$ into $x'_i = [x_i, \sqrt{M^2 - \|x_i\|^2}] \in \mathbb{R}^{d+1}$ where $M^2 = \max_j \|x_j\|^2$, applies 1D FLAS pre-sorting, and builds the graph in `FP32_L2` space ($K_{\text{graph}} = 32, K_{\text{ext}} = 64$).

2. **Automatic Feature Swap Back to $d$-Dim Inner Product**:
   After graph construction, `deglib.build` automatically swaps the stored features back to the original $d$-dimensional `FP32_InnerProduct` representation.

3. **Compression & Optimized `Searcher`**:
   `deglib.searcher(graph, compress="16bit", optimize=True)` compresses the graph to `FP16_InnerProduct`, wraps it in a compact `ReadOnlyGraph` with K-Means routing entry points and prefetch tuning, and sweeps over max distance evaluation budgets (`6000, 6500, 7000, 7500, 8000, 9000`) at $\varepsilon_{\text{search}} = 0.18$.

---

## Dataset

This example benchmarks on the **SISAP 2026 `llama-dev`** dataset (Llama embeddings):
- **Repository**: [`SISAP-Challenges/SISAP2026`](https://huggingface.co/datasets/SISAP-Challenges/SISAP2026)
- **Files**: `llama-dev/llama-dev.h5`
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

The script displays an interactive Matplotlib trade-off plot (Recall vs. QPS) with `max_dist` data point annotations.

---

## Command-Line Options

```
usage: main.py [-h] [--dataset DATASET] [--k-graph K_GRAPH] [--k-ext K_EXT]
               [--eps-ext EPS_EXT] [--eps-search EPS_SEARCH]
               [--compress {32bit,16bit,8bit,4bit,2bit,auto}]
               [--flas | --no-flas] [--build-threads BUILD_THREADS]
               [--search-threads SEARCH_THREADS] [--opt-target OPT_TARGET]
               [--num-runs NUM_RUNS] [--max-dist MAX_DIST]
               [--output-plot OUTPUT_PLOT] [--no-show]

options:
  --dataset DATASET               Path to HDF5 dataset file or 'llama-dev' (default: llama-dev)
  --k-graph K_GRAPH               Graph degree per vertex (default: 32)
  --k-ext K_EXT                   Builder search size parameter (default: 64)
  --eps-ext EPS_EXT               Builder search expansion factor (default: 0.001)
  --eps-search EPS_SEARCH         Epsilon search factor (default: 0.18)
  --compress {32bit,16bit,8bit,4bit,2bit,auto}
                                  Target precision for MIPS graph (default: 16bit)
  --flas, --no-flas               Enable or disable FLAS 1D pre-sorting (default: enabled)
  --build-threads BUILD_THREADS   Number of threads for graph building (default: 1)
  --search-threads SEARCH_THREADS Number of parallel threads for query search (default: 8)
  --opt-target OPT_TARGET         Optimization target: Auto, StreamingData, LowLID, HighLID (default: LowLID)
  --num-runs NUM_RUNS             Number of search repetitions for averaging (default: 100)
  --max-dist MAX_DIST             Comma-separated list of max distance budgets (default: 6000,6500,7000,7500,8000,9000)
  --output-plot OUTPUT_PLOT       Path to save trade-off curve PNG
  --no-show                       Disable GUI plot display
```
