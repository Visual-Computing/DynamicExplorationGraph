# DEG Sliding Window Benchmark (CleANN Paper Baseline Reproduction)

This example reproduces the **Sliding Window Batched Update** baseline experiment evaluated against the Dynamic Exploration Graph (DEG) in the **CleANN** paper:
> **"CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search"**  
> *Ziyu Zhang, Yuanhao Wei, Joshua Engels, Julian Shun (MIT CSAIL)*  
> [arXiv:2507.19802v1](https://arxiv.org/abs/2507.19802v1)

---

## Paper Benchmark Mapping

| Benchmark Aspect | CleANN Paper Specification | Reproduction in this Example |
|---|---|---|
| **Section** | Section 6.1 (*Setup*), Section 6.2.3 (*Sequential Baseline Comparison*) | Implemented in [`main.py`](./main.py) |
| **Dataset** | **RedCaps-512** (512D CLIP image-text embeddings with distribution drift) | Downloaded automatically from Zenodo: [`redcaps-512-angular.hdf5`](https://zenodo.org/records/13137120) |
| **Window Size ($N$)** | $500{,}000$ active vectors | `--window-size 500000` (default) |
| **Streaming Updates** | 100 rounds of $5{,}000$ inserts & $5{,}000$ deletes (1% of window) | `--rounds 100 --batch-size 5000` |
| **DEG Deletion Strategy** | Dynamic on-the-fly edge repair upon `remove_entry` | Performed automatically by `deglib.builder.GraphBuilder` |
| **Query Evaluation** | 800 test queries, $k = 10$, $\varepsilon = 0.03$ | `--num-queries 800 --search-k 10 --search-eps 0.03` |
| **Execution Mode** | Single-threaded execution (1 thread) | `builder.set_thread_count(1)` & `threads=1` |
| **Target Figures** | **Figure 20** (Recall 10@10) & **Figure 21** (Single-thread Throughput) | Saved to `sliding_window_benchmark.png` |

---

## How It Works

1. **Initial Index Build:**
   * Loads the first $N = 500{,}000$ vectors from the RedCaps dataset.
   * Populates a mutable `DynamicExplorationGraph` using `OptimizationTarget.StreamingData`.
2. **100 Sliding Window Update Rounds:**
   * **Insert Batch:** Queues $M = 5{,}000$ new incoming vectors via `builder.add_entry()`.
   * **Delete Batch:** Queues $M = 5{,}000$ oldest active vectors via `builder.remove_entry()`.
   * **Build / Repair:** Calls `builder.build()`. DEG automatically heals and updates local graph connectivity around deleted vertices.
   * **Ground Truth Computation:** Computes exact brute-force Top-10 nearest neighbors for the 800 test queries over the currently active $500{,}000$ vectors.
   * **Query Benchmark:** Executes DEG search with exploration $\varepsilon = 0.03$, measuring Recall@10, single-thread Search QPS/latency, and Update throughput (updates/ms).
3. **Visual Summary:**
   * Generates a dual-panel plot reproducing Figure 20 (Recall@10 over rounds) and Figure 21 (Throughput).

---

## Quick Start

### 1. Environment Setup

```bash
cd examples/sliding_window
uv sync
```

### 2. Run Full Paper Benchmark (500k Vectors, 100 Rounds, RedCaps)

```bash
uv run main.py --dataset redcaps --window-size 500000 --rounds 100 --search-eps 0.03 --search-k 10
```
*(On first execution, `redcaps-512-angular.hdf5` will be downloaded automatically from Zenodo to your cache directory).*

### 3. Run a Fast Test Run (Smaller Window / Fewer Rounds)

```bash
uv run main.py --dataset redcaps --window-size 50000 --rounds 10
```

---

## CLI Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | `str` | `redcaps` | Benchmark dataset (`redcaps`, `sift1m`, `glove`, `deep1m`, or `custom`). |
| `--window-size` | `int` | `500000` | Number of active vectors maintained in the sliding window ($N$). |
| `--batch-size` | `int` | `5000` | Number of vectors inserted & deleted per round (CleANN paper: 5,000). |
| `--rounds` | `int` | `100` | Number of sliding window update rounds (paper: 100). |
| `--search-eps` | `float` | `0.03` | DEG search exploration $\varepsilon$ (paper: 0.03). |
| `--search-k` | `int` | `10` | Top-$k$ nearest neighbors to evaluate (paper: 10). |
| `--num-queries` | `int` | `800` | Number of test queries to evaluate per round. |
| `--edges-per-vertex` | `int` | `32` | DEG graph out-degree bound ($k$). |
| `--no-show` | `flag` | `False` | Disable interactive GUI plot display. |
| `--save-plot` | `str` | `sliding_window_benchmark.png` | Output path for result curves. |

---

## References

* **CleANN Paper:** Zhang et al., *"CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search"*, arXiv:2507.19802v1, 2025.
* **RedCaps Dataset on Zenodo:** [https://zenodo.org/records/13137120](https://zenodo.org/records/13137120)
* **CleANN Code Repository:** [https://github.com/SylviaZiyuZhang/CleANN](https://github.com/SylviaZiyuZhang/CleANN)
