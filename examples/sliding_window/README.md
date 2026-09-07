# DEG Sliding Window Benchmark (CleANN Paper Reproduction)

This benchmark reproduces the **Sliding Window Batched Update** baseline experiment from the **CleANN** paper:
> **"CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search"**  
> *Ziyu Zhang, Yuanhao Wei, Joshua Engels, Julian Shun (MIT CSAIL)* — [arXiv:2507.19802v1](https://arxiv.org/abs/2507.19802v1)

The evaluation runs on **RedCaps-512** (512D OpenAI CLIP image-text embeddings). RedCaps is particularly challenging because it exhibits **real-world distribution drift** over time as vectors are streamed in chronological order, combined with **out-of-distribution (OOD) search queries**.

---

## Quick Start

All benchmark parameters default to the exact CleANN paper setup (**RedCaps-512**, $N=500{,}000$, 100 rounds, 5,000 updates/round, $k=10$, $\varepsilon=0.03$, single-threaded):

```bash
cd examples/sliding_window
uv sync
uv run main.py
```
*(On the first run, `redcaps-512-angular.hdf5` is downloaded automatically from Zenodo).*

### Fast Quick-Check (Small Window / 10 Rounds)

```bash
uv run main.py --window-size 50000 --batch-size 500 --rounds 10
```

---

## Benchmark Setup & Paper Mapping

| Parameter / Aspect | CleANN Paper Value | Default in `main.py` | Description |
|---|---|---|---|
| **Dataset** | **RedCaps-512** (512D) | `redcaps` | Chronological CLIP embeddings with natural distribution drift |
| **Window Size ($N$)** | $500{,}000$ vectors | `500000` | Active vectors maintained in the sliding window |
| **Batch Size ($M$)** | $5{,}000$ (1% of window) | `5000` | Replaced per round (5k inserts + 5k deletes) |
| **Rounds** | 100 rounds | `100` | Total sliding-window iterations (1M vectors streamed in total) |
| **Search $k$ & $\varepsilon$** | $k=10$, $\varepsilon=0.03$ | `10`, `0.03` | Top-10 search exploration parameter |
| **Query Set** | 800 test queries | `800` | OOD queries evaluated against exact brute-force Top-10 ground truth |
| **Threads** | 1 thread (single-thread) | `1` | Single-threaded evaluation for direct baseline comparison |
| **Reproduced Figures** | Fig. 20 (Recall 10@10), Fig. 21 (Throughput) | Saved to `sliding_window_benchmark.png` |

---

## Output Metrics

Each round prints the core evaluation metrics:
* **`Recall@10`**: Exact top-10 accuracy over the active 500k sliding window.
* **`QPS`**: Single-threaded search throughput (queries / second).
* **`Search/ms`**: Search operations per millisecond ($\text{QPS} / 1000$).
* **`Update/ms`**: Update throughput in vector replacements per millisecond ($\text{batch\_size} / \text{update\_time\_ms}$).

---

## Optional CLI Overrides

| Argument | Type | Default | Description |
|---|---|---|---|
| `--window-size` | `int` | `500000` | Size of sliding window ($N$). |
| `--batch-size` | `int` | `5000` | Inserts and deletes per round ($M$). |
| `--rounds` | `int` | `100` | Number of update rounds. |
| `--search-eps` | `float` | `0.03` | Search exploration factor $\varepsilon$. |
| `--search-k` | `int` | `10` | Top-$k$ nearest neighbors to query. |
| `--num-queries` | `int` | `800` | Query count per round. |
| `--edges-per-vertex` | `int` | `32` | DEG graph degree bound ($k$). |
| `--no-show` | `flag` | `False` | Skip GUI plot display. |
| `--save-plot` | `str` | `sliding_window_benchmark.png` | Path to save plot artifact. |

---

## References

* **CleANN Paper:** [arXiv:2507.19802v1](https://arxiv.org/abs/2507.19802v1)
* **RedCaps Dataset:** [Zenodo Record 13137120](https://zenodo.org/records/13137120)
* **CleANN Reference Implementation:** [github.com/SylviaZiyuZhang/CleANN](https://github.com/SylviaZiyuZhang/CleANN)
