# DEG Paper Reproduction Benchmark

This example project demonstrates how to download paper datasets from `readme.md`, build a DEG graph with preset parameters, run search queries, measure Recall vs. QPS (Queries Per Second), and display the results plot interactively.

> [!NOTE]
> All runtime and throughput results published in our papers were evaluated using the native C++ implementation (`cpp/`). Due to language bindings and runtime dynamics, Python executions may have a small overhead.

## Available Datasets

- `audio` (192D, 53k base vectors, L2 distance)
- `enron` (1369D, 94k base vectors, L2 distance)
- `sift1m` (128D, 1M base vectors, L2 distance)
- `deep1m` (96D, 1M base vectors, L2 distance)
- `glove-100` (100D, 1.18M base vectors, Angular / InnerProduct distance)

## Prerequisites: Building the Python Library (`deglib`)

The `deglib` Python library contains C++ pybind11 bindings (`deglib_cpp`). 

Since `pyproject.toml` references `deglib` locally (`path = "../../python"`), running `uv sync` automatically copies the C++ sources from `cpp/`, invokes CMake, and compiles/installs the latest `deglib` C++ bindings into the local environment:

```bash
# Install dependencies & compile latest deglib C++ bindings
uv sync

# If C++ code or pybind11 bindings were modified, force rebuilding the extension:
uv sync --reinstall-package deglib
```

> **Note:** A C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS) and CMake must be available in your system `PATH`. `setup.py` handles copying C++ files, running CMake, and compiling the extension during the `uv sync` step.

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
