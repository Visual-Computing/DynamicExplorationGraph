## Rules

* Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
* Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
* Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
* Keep components modular and concerns clearly separated.
* Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
* Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
* Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
* Use modern cpp 20 approaches

- **Debug Mode**: Agents must **never** use debug mode. This means no debug output, debug logging, or debug builds.
- **Security**: Agents must ensure that no security-critical changes are made without sufficient verification.
- **Test Coverage**: Any new functionality must be covered by unit tests.
- **Code Quality**: Code must adhere to existing project style and guidelines.

## Project Overview

This repository contains implementations in C++, Java, and Python for a Dynamic Exploration Graph algorithm.

### Directory Structure

```
DynamicExplorationGraph/
├── cpp/          # C++ Header-only library + Tests + Benchmarks
├── python/       # Python Bindings, Tests, setup.py
├── examples/     # Example projects (knng, dynamic_data, static_data)
├── java/         # Java implementation
├── docs/         # Documentation
└── readme.md     # Main README
```

## Essential Commands

### Compile & Test C++

The `CMakePresets.json` file is located in the `cpp/` directory:

```bash
cd cpp/

# 1. Configure (choose preset matching your OS: windows-msvc-avx2, linux-gcc-avx2, macos-clang-avx2)
cmake --preset windows-msvc-avx2

# Fast test-only config (skips heavy benchmark suites):
cmake --preset windows-msvc-avx2 -DENABLE_BENCHMARKS=OFF

# 2. Build:
# Build specific targets (fast, avoids compiling all benchmarks and tests):
cmake --build --preset windows-msvc-avx2-release --target <target_name>
# e.g.: cmake --build --preset windows-msvc-avx2-release --target test_readonly_graph test_dynamic_graph

# Or build everything (default):
cmake --build --preset windows-msvc-avx2-release

# 3. Test:
# All tests:
ctest --preset windows-msvc-avx2-release --output-on-failure
# Filter specific tests:
ctest --preset windows-msvc-avx2-release -R <regex> --output-on-failure
```

### Python Environment & Command Execution

> **Important for Agents**: Python commands are executed consistently via `uv` (`uv run ...`, `uv pip ...`). Every `run_command` spawns a new non-interactive subshell; never invoke `python` or `pip` directly globally, always use `uv`.

### Build & Install Python Bindings

```bash
cd python/
uv venv
uv pip install setuptools==83.0.0 pybind11==3.0.4 build==1.5.0 wheel==0.48.0
uv pip install -e . --no-build-isolation --verbose
```

### Reinstall Python Bindings from Scratch (Clean Slate)

```bash
cd python/
rm -rf .venv build/ dist/ *.egg-info
uv venv
uv pip install --upgrade pip
uv pip install setuptools==83.0.0 pybind11==3.0.4 build==1.5.0 wheel==0.48.0
uv pip install -e . --no-build-isolation --verbose
uv run python -c "import deglib; print(deglib.__version__)"
```

### Run Python Tests

```bash
cd python/
uv run pytest
```

### Code Formatting (C++ & Python)

```bash
# Format C++ files
clang-format -i $(git ls-files '*.cpp' '*.h')

# Format Python files
cd python/
uv run ruff format .
```

### Run Examples with UV

```bash
cd examples/knng/
uv sync
uv run main.py --dataset small --no-show
```

### Common Errors & Solutions

| Error | Solution |
|---|---|
| `CMakePresets.json not found` | `CMakePresets.json` is located in `cpp/` – run `cd cpp/` prior to the command |
| `'copy_n' is not a member of 'std'` | pybind11 version issue: `uv pip install pybind11==3.0.4` |
| `ModuleNotFoundError: No module named 'deglib'` | `uv pip install -e . --no-build-isolation` |
| `.venv` not found | Run `uv venv` **inside** `python/` |
| `uv: command not found` | Install via `pip install uv` or the official installer |
| `Python was not found` (Windows) | Always execute commands via `uv run ...` or `uv pip ...` |

### Debug Mode

**Prohibited.** Always use Release configuration for all builds.
