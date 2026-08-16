# DEG Examples

This directory contains standalone Python projects demonstrating various features and benchmark workflows of the Dynamic Exploration Graph (DEG).

Each directory inside `examples/` is a self-contained [`uv`](https://docs.astral.sh/uv/) project with its own `pyproject.toml`.

## Prerequisites & UV Setup

We use [`uv`](https://docs.astral.sh/uv/) for fast Python environment and dependency management.

### 1. Install `uv`

- **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Via Pip**:
  ```bash
  pip install uv
  ```

### 2. Environment Setup

Navigating into any example directory and running `uv sync` or `uv run` will automatically set up a isolated Python virtual environment, build the local `deglib` C++ bindings, and install all required dependencies:

```bash
cd examples/knng
uv sync

# If C++ code or pybind11 bindings were modified, force rebuilding the extension:
uv sync --reinstall-package deglib
```

## Projects

- [`knng`](./knng/): k-Nearest Neighbor Graph (k-NNG) construction benchmark using EVP quantization and FP16 reranking (SISAP 2026 Challenge Task 1).
- [`mips`](./mips/): Maximum Inner Product Search (MIPS) benchmark using $(d+1)$-dimensional $L_2$ transformation, FLAS pre-sorting, and SIMD FP16 inner products (SISAP 2026 Challenge Task 2).
- [`static_data`](./static_data/): DEG paper search benchmark reproduction (Recall vs. QPS) on static datasets (`sift1m`, `deep1m`, `glove-100`, `audio`, `enron`).
- [`dynamic_data`](./dynamic_data/): DEG dynamic data streaming benchmark (`AddHalf`, `AddHalfRemoveAndAddOneAtATime`, `AddAllRemoveHalf`).
