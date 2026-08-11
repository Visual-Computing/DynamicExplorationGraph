# DEG Dynamic Data Benchmark (Python)

Python mirror of `cpp/bench/src/bench_dynamic_data.cpp`.

Builds and evaluates DEG graphs under three dynamic data streaming conditions and measures ANNS recall vs. QPS performance against the **half-dataset ground truth** (first half of base vectors).

## Tested Data Stream Patterns

| Stream Type | Description |
|---|---|
| `AddHalf` | Insert only the first half of base vectors |
| `AddHalfRemoveAndAddOneAtATime` | Interleaved insertion and deletion operations |
| `AddAllRemoveHalf` | Insert all vectors, then remove the second half |

All three use `OptimizationTarget.StreamingData`.

## Usage

```bash
uv run python main.py [dataset] [options]
```

### Datasets
- `sift1m` — SIFT1M (1M vectors, 128D, default)
- `deep1m` — DEEP1M (1M vectors, 96D)
- `glove` / `glove-100` — GloVe (1.18M vectors, 100D)
- `audio` — Audio (53.3k vectors, 192D)
- `enron` — Enron (94.9k vectors, 1369D)
- `all` — Run all datasets sequentially

### Options

| Option | Description |
|---|---|
| `--graph-dir <path>` | Directory to save/load graph files. Filenames follow C++ naming: `{dims}D_K{k}_{stream_type}.deg` |
| `--force-rebuild` | Rebuild graphs even if files already exist |
| `--instruction <inst>` | Distance instruction set: `auto`, `avx512`, `avx2`, `scalar` |
| `--threads <n>` | Build threads (default: 1) |
| `--cache-dir <path>` | Dataset cache directory (default: `~/.cache/deg_datasets`) |
| `--no-show` | Do not display interactive plots |
| `--max-base-vecs <n>` | Limit base vectors for quick testing |

### Examples

```bash
# Quick test with audio dataset
uv run python main.py audio --max-base-vecs 5000 --no-show

# Full SIFT1M benchmark, saving graphs to disk
uv run python main.py sift1m --graph-dir /data/deg_graphs/sift1m/dynamic

# All datasets
uv run python main.py all --graph-dir /data/deg_graphs --threads 4
```

## Prerequisites: Building the Python Library (`deglib`)

The `deglib` Python library contains C++ pybind11 bindings (`deglib_cpp`). 

Since `pyproject.toml` references `deglib` locally (`path = "../../python"`), running `uv sync` automatically copies the C++ sources from `cpp/`, invokes CMake, and compiles/installs the latest `deglib` C++ bindings into the local environment:

```bash
# Install dependencies & compile latest deglib C++ bindings
uv sync
```

> **Note:** A C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS) and CMake must be available in your system `PATH`. `setup.py` handles copying C++ files, running CMake, and compiling the extension during the `uv sync` step.

## Key Differences from `static_data`

- Iterates over 3 `DataStreamType`s instead of a single `AddAll`
- Uses `StreamingData` optimization target for all datasets
- ANNS test uses **half-dataset ground truth** (queries evaluated against first `base_count/2` vectors)
- No exploration test
- Graph files named `{dims}D_K{k}_{stream_type}.deg` (C++ compatible)
