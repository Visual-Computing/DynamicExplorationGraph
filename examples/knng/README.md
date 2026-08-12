# k-Nearest Neighbor Graph (k-NNG) Construction Example

This example demonstrates how to build a **k-Nearest Neighbor Graph (k-NNG)** using **Dynamic Exploration Graph (DEG)** based on Task 1 from the **SISAP 2026 Indexing Challenge**.

## Problem & Approach

Given $N$ high-dimensional vectors, the goal of k-NNG construction (self-join) is to compute the top-$K$ ($K=15$) nearest neighbors for every vector in the dataset.

The approach demonstrates **Mode 4 (`evp-rerank`)**, which achieves state-of-the-art trade-offs between construction speed and neighbor recall ($\ge 88\%$):

1. **EVP Quantization**: Feature vectors are converted to compact sparse EVP-bit representations using the C++ `deglib_cpp.quantize_batch` function (`--non-zeros 512`).
2. **DEG Construction**: A dynamic exploration graph is constructed using DEG's `GraphBuilder` with the `EVP_InnerProduct` metric for fast quantized distance computation.
3. **Graph Exploration**: Exploration for vertex $i$ walks the DEG graph neighborhood using fast EVP bit-level inner product distances to collect candidates (`evpK = 50`).
4. **FP16 Candidate Reranking**: Exact inner-product distances are computed using `deglib_cpp.floats_to_fp16` and `deglib_cpp.fp16_to_floats` for candidate sets to produce final $k$-nearest neighbor edges.

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

## Running the Benchmark with `uv`

### 1. Run Full Benchmark (200K Wikipedia BGE-M3 vectors)

```bash
uv run main.py
```

*The script automatically downloads `benchmark-dev-wikipedia-bge-m3-small.h5` from Hugging Face Hub if not cached locally.*


## Command-Line Options

- `--dataset`: Path to HDF5 dataset file or `"small"` (default: downloads/uses Wikipedia BGE-M3 200K dataset).
- `--max-vecs`: Limit vector count for fast verification.
- `--non-zeros`: Number of non-zero active components in EVP quantization 
- `--k-graph`: Graph degree per vertex 
- `--max-dist`: Maximum distance calculation budget per query 
- `--evpK`: Number of candidate vertices retrieved before FP16 reranking 
- `--threads`: Number of parallel execution threads
- `--output-plot`: Save execution time breakdown chart to a file.
