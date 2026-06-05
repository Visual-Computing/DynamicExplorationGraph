# EVP Benchmark Task 2: Approximate MIPS on LLM Attention Workloads

This directory implements graph-based approximate Maximum Inner Product Search (MIPS)
benchmarks for LLM attention key/value cache workloads, using the SISAP 2026
llama-dev dataset.

The executable `deglib_evp_task2` accepts a single HDF5 input file
(only `llama-dev.h5` is supported for task2) containing `train`, `test/queries`,
and `test/knns` datasets.

---

## Benchmark Modes

### mode1 (FP32 Build + FP32 Search — Baseline)

Builds a SizeBoundedGraph using `Metric::InnerProduct` (FP32) and searches
with exact FP32 inner-product similarity. This is the highest-recall baseline.

Optional FLAS pre-sort can be enabled via `--flas` (see below).

### mode2 (FP16 Build + FP16 Search)

Converts FP32 training vectors to FP16, builds the graph with
`Metric::FP16InnerProduct`, and searches with FP16 inner-product similarity.
Lower memory footprint at modest recall degradation.

### mode3 (FP32 Build + FP16 Search)

Builds the graph using exact FP32 features, then converts to a ReadOnlyGraph
with FP16 features for search. Combines FP32-quality graph topology with
FP16-speed search.

### mode4 (FP32 Build + Asymmetric FP16×EVP Search)

Builds the graph with FP32 features, quantizes the database to EVP bits,
then performs asymmetric search (FP16 query × EVP database) via
`Metric::FP16EvpAsymmetric`. Requires `--non-zeros` for EVP sparsity control.

All modes support `--flas` to enable FLAS pre-sort of training vectors
before graph construction (off by default).

---

## FLAS Pre-Sort

FLAS (Fast Linear Assignment Sorting) reorders the training vectors along a
1D line so that similar vectors end up at adjacent grid positions. This
arrangement can improve graph construction quality and search performance.

FLAS is **off by default**. Enable it with `--flas` and tune with:

| Flag | Default | Description |
|---|---|---|
| `--flas` | off | Enable FLAS pre-sort |
| `--flas-metric <l2\|ip>` | `l2` | Distance metric for FLAS sorting: L2 Euclidean or Inner Product |
| `--flas-radius-decay <f>` | `0.93` | Swap radius decay factor per iteration (lower = faster convergence) |

FLAS is supported by all modes but is most effective with FP32 modes (mode1/3).

---

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--threads <n>` | 8 | Worker threads for query exploration |
| `--build-threads <n>` | 1 | Threads for graph construction |
| `--k-top <n>` | 30 | Number of nearest neighbors to retrieve |
| `--k-graph <n>` | 30 | Graph degree (edges per vertex) |
| `--k-ext <n>` | 30 | Search size during graph construction |
| `--eps-ext <f>` | 0.001 | Search expansion during construction |
| `--eps-search <f>` | 0.3 | Search expansion during querying |
| `--max-dist <list>` | 20000 | Max distance computations per query (can be comma-separated list) |
| `--non-zeros <n>` | 64 | EVP quantization sparsity (mode4 only) |
| `--no-recall` | — | Skip ground-truth loading + recall computation |
| `--output <path>` | — | Write results to a binary ivecs file |
| `--graph <path>` | — | Save/load a pre-built graph file |
| `--prune-worst <n>` | 0 | Replace n worst edges per vertex with self-loops |
| `--num-runs <n>` | 1 | Repeat query exploration and average results |
| `--opt-target <str>` | LowLID | Builder optimization target (LowLID, HighLID, etc.) |

---

## Examples

```powershell
# Baseline FP32 mode (no FLAS)
.\deglib_evp_task2.exe "C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5" mode1 --max-dist 5000

# FP32 mode with FLAS (default L2 metric)
.\deglib_evp_task2.exe "C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5" mode1 --max-dist 5000 --flas

# FLAS with Inner Product metric and faster decay
.\deglib_evp_task2.exe "C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5" mode1 --flas --flas-metric ip --flas-radius-decay 0.8

# FP16 build + search with FLAS
.\deglib_evp_task2.exe "C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5" mode2 --flas

# FP32 build + asymmetric EVP search
.\deglib_evp_task2.exe "C:\Data\ANN\sisap2026\llama-dev\llama-dev.h5" mode4 --non-zeros 128
```

---

## Building

```powershell
cmake --preset windows-msvc
cmake --build --preset windows-msvc-release --target deglib_evp_task2
```
