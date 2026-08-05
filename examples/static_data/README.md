# DEG Paper Reproduction Benchmark

This example project demonstrates how to download paper datasets from `readme.md`, build a DEG graph with preset parameters, run search queries, measure Recall vs. QPS (Queries Per Second), and display the results plot interactively.

## Available Datasets

- `audio` (192D, 53k base vectors, L2 distance)
- `enron` (1369D, 94k base vectors, L2 distance)
- `sift1m` (128D, 1M base vectors, L2 distance)
- `deep1m` (96D, 1M base vectors, L2 distance)
- `glove-100` (100D, 1.18M base vectors, Angular / InnerProduct distance)

## Prerequisites & UV Setup

1. **Install `uv`** (if not already installed):
   - **Linux/macOS**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - **Windows (PowerShell)**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - **Pip**: `pip install uv`

2. **Sync Dependencies**:
   Inside this directory, initialize the environment and install dependencies (including building local `deglib` bindings):
   ```bash
   uv sync
   ```

## Running the Benchmark

`uv run` automatically uses the managed virtual environment.

### 1. Run Full Benchmark (e.g. SIFT1M)

```bash
uv run main.py --dataset sift1m
```

### 2. Fast Test Run (e.g. Audio dataset with 1,000 vectors)

```bash
uv run main.py --dataset audio --max-base-vecs 1000
```

## Options

- `--dataset`: Dataset name (`sift1m`, `audio`, `enron`, `deep1m`, `glove-100`). Default: `sift1m`.
- `--cache-dir`: Persistent cache folder for datasets. Default: `~/.cache/deg_datasets`.
- `--output-plot`: Optional path to also save the plot to a PNG image (e.g. `--output-plot recall_vs_qps.png`).
- `--no-show`: Disable opening the interactive plot window (useful for headless CI environments).
- `--max-base-vecs`: Limit base vectors for fast debugging / testing.
