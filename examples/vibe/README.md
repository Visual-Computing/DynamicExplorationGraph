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

## Quick Start

### 1. Environment Setup

```bash
cd examples/vibe
uv sync
```

### 2. Run Benchmark

Datasets are automatically downloaded on demand directly from the official [VIBE Hugging Face repository](https://huggingface.co/datasets/vector-index-bench/vibe) and stored in `D:/Data/VIBE` (or `~/.cache/vibe` / `VIBE_CACHE_DIR`).

```bash
# Run on AGNews-mxbai (compact dataset ~120k vectors)
uv run main.py --dataset agnews-mxbai

# Run on Yahoo-MiniLM (677k vectors)
uv run main.py --dataset yahoo-minilm

# Run without GUI plot popups
uv run main.py --dataset arxiv-nomic --no-show
```

### 3. Command-Line Options

```bash
uv run main.py --help
```

- `--dataset`, `-d`: Dataset name to benchmark (e.g. `agnews-mxbai`, `arxiv-nomic`, `landmark-dino`, `msmarco-qwen`, `gooaq-distilroberta`, `laion-clip`, `imagenet-align`, `imagenet-clip`, `yandex`, `yahoo-minilm`).
- `--cache-dir`, `-c`: Custom directory for dataset files and saved `.deg` graphs.
- `--build-threads`, `-t`: Number of CPU threads for graph building (default: half of CPU cores, `threads // 2`).
- `--k`: Graph degree $k$ (out-degree per vertex).
- `--extend-k`: Exploration width during graph building.
- `--eps`: Build $\varepsilon$ parameter.
- `--anns-k`: Number of nearest neighbors to evaluate (default: 100).
- `--rebuild-graph`: Force graph re-construction even if a cached graph file exists.
- `--no-show`: Do not open interactive matplotlib plot window.
