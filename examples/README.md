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
cd examples/paper_reproduction
uv sync
```

## Projects

- [`paper_reproduction`](./paper_reproduction/): Reproduce DEG paper search benchmarks (Recall vs QPS) on datasets mentioned in the root `readme.md` (`Audio`, `Enron`, `SIFT1M`, `DEEP1M`, `GloVe-100`).
