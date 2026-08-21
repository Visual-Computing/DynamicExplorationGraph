# Dynamic Exploration Graph (deglib) Documentation

This directory contains the Sphinx documentation source files for `deglib`.

## Prerequisites & Setup

The documentation dependencies (including `sphinx`, themes, and the local `deglib` package) are managed via [`uv`](https://docs.astral.sh/uv/).

To initialize the environment and install dependencies:

```bash
uv sync
```

## Building Documentation

### Build HTML

To build the HTML documentation:

```bash
uv run sphinx-build -b html . _build/html
```

Once built, open `_build/html/index.html` in your browser.

### Build Markdown

To build Markdown documentation:

```bash
uv run sphinx-build -b markdown . _build/markdown
```

### Clean Build Directory

To remove previous build artifacts:

```bash
# PowerShell
Remove-Item -Recurse -Force _build

# Bash
rm -rf _build
```
