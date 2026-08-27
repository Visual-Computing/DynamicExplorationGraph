# VIBE ANNS Benchmark Reproduction with DEG

This example reproduces the Approximate Nearest Neighbor Search (ANNS) experiments from the **[VIBE (Vector Index Benchmark for Embeddings)](https://vector-index-bench.github.io/)** project using the Dynamic Exploration Graph (DEG).

VIBE provides realistic embedding datasets covering in-distribution and out-of-distribution (OOD) search across text, vision, and multi-modal models with exact pre-computed Top-100 ground truth neighbors.

## Supported VIBE Datasets

| Dataset Key | VIBE Name | Type | Size ($N$) | Dimension ($D$) | Metric |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `agnews-mxbai` | AGNews-mxbai | In-Distribution | 120,000 | 1,024 | L2 (Euclidean) |
| `arxiv-nomic` | ArXiv-nomic | In-Distribution | 2,000,000 | 768 | Inner Product |
| `landmark-dino` | Landmark-dino | In-Distribution | 760,757 | 768 | Cosine |
| `msmarco-qwen` | MSMARCO-qwen | In-Distribution | 8,841,823 | 1,024 | Inner Product |
| `gooaq-distilroberta`| GooAQ-distilroberta | In-Distribution | 1,471,375 | 768 | Inner Product |
| `laion-clip` | LAION-clip | Out-of-Distribution | 1,000,000 | 512 | Inner Product |
| `imagenet-align` | ImageNet-align | Out-of-Distribution | 1,281,167 | 640 | Inner Product |
| `imagenet-clip` | ImageNet-clip | In-Distribution | 1,281,167 | 512 | Inner Product |
| `yandex` | Yandex-200 | Out-of-Distribution | 1,000,000 | 200 | Cosine |
| `yahoo-minilm` | Yahoo-minilm | In-Distribution | 677,305 | 384 | Inner Product |

---

## Quick Start

### 1. Environment Setup

```bash
cd examples/vibe
uv sync
```

### 2. Run Benchmark

Datasets are automatically downloaded on demand directly from the official [VIBE Hugging Face repository](https://huggingface.co/datasets/vector-index-bench/vibe) and stored in `D:/Data/DEG` (or `~/.cache/deg` / `VIBE_CACHE_DIR`).

```bash
# Run benchmark on LAION-clip (generates and opens interactive HTML plot)
uv run main.py --dataset laion-clip

# Run on Yahoo-MiniLM without opening browser popup
uv run main.py --dataset yahoo-minilm --no-show
```

### 3. Interactive Plot Viewer / Log Explorer

You can explore existing benchmark log files and generate standalone interactive Plotly charts at any time:

```bash
# Open interactive GUI Log Explorer (browse dataset logs & view plots)
uv run plot_from_log.py

# Render and open interactive plot directly for a specific dataset
uv run plot_from_log.py --dataset laion-clip

# Render and open interactive plot directly from an explicit log file
uv run plot_from_log.py --log D:/Data/DEG/laion-clip/deg-fp32/laion-clip_benchmark.log
```

---

## Interactive Plot Features

The generated benchmark visualizations (`*_anns_benchmark.html`) provide:
- **Grouped Interactive Legend**: Filter curves by **Optimization Target** (`LowLID`, `StreamingData`, `HighLID`), **Pruning Status** (`Unpruned`, `MRNG Pruned`), **Graph Degree $K$** ($16, 24, 30, 40, 48$), and **Rerank Factors** ($1.0\times, 1.2\times, 1.5\times, 2.0\times$).
- **Locked Plot Axes**: Filtering curves turns elements on and off smoothly without jumpy axis rescaling.
- **Mouse Navigation**: Stufenloser **Mouse-Wheel Zoom** and click-and-drag **Panning**. Double-click resets view.
- **Detailed Tooltips**: Hover over data points to inspect exact Recall@100, QPS, search $\varepsilon$, graph parameters, and rerank configuration.

---

## Configuration & Command-Line Options

### Configuration via `config.yml`
All benchmark search parameters, graph degrees, optimization targets, pruning options, search $\varepsilon$ ranges, and reranking factors are configured centrally in [`config.yml`](file:///C:/Lang/cpp/DynamicExplorationGraph/examples/vibe/config.yml):
- **Graph parameters**: `k`, `opt_target`, `prune_non_rng`, `threads`, `query_dtype`.
- **Query parameters**: `search_eps`, `rerank_size_factor`.

### `main.py`
- `--dataset`, `-d`: Dataset name (e.g. `laion-clip`, `arxiv-nomic`, `agnews-mxbai`, etc.).
- `--cache-dir`, `-c`: Custom directory for datasets and graphs (default: `D:/Data/DEG` or `~/.cache/deg`).
- `--build-threads`, `-t`: Number of CPU threads for graph building (default: `cpu_count // 2`).
- `--query-dtype`: Override query and feature storage precision (`float32`, `int8`).
- `--cpu`, `--cpu-affinity`: Pin the benchmark process to specific CPU core ID(s).
- `--no-show`: Do not open browser window after benchmark completes.

### `plot_from_log.py`
- *(No arguments)*: Launches the dark-themed **GUI Log Explorer** to browse logs across all VIBE dataset folders.
- `--dataset`, `-d`: Dataset name to resolve standard log path.
- `--log`, `-l`: Explicit path to a `*.log` benchmark file.
- `--output`, `-o`: Output HTML file path.
- `--no-open`: Do not automatically open the generated HTML in the default web browser.
