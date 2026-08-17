# deglib: C++ Library for Dynamic Exploration Graphs

`deglib` is a high-performance, header-only C++20 library implementing the **Dynamic Exploration Graph (DEG)** and its continuous refinement algorithms for Approximate Nearest Neighbor Search (ANNS).

It supports both static and dynamic streaming datasets through incremental extension, continuous edge optimization, and vertex deletion with state-of-the-art recall vs. QPS trade-offs.

---

## Key Features

- **Header-Only C++20**: Zero mandatory runtime dependencies for the core library.
- **Multiple Graph Topologies**:
  - `deglib::graph::SizeBoundedGraph`: Preallocated fixed-capacity graph for maximum build throughput.
  - `deglib::graph::DynamicGraph`: Chunk-allocated dynamic graph supporting arbitrary growth and shrinkage.
  - `deglib::graph::ReadOnlyGraph`: Compact, memory-efficient graph optimized for read-only ANNS search queries.
- **Multi-Threaded Builder**: `deglib::builder::EvenRegularGraphBuilder` with lock-free batch scheduling and customizable optimization profiles (`StreamingData`, `LowLID`, `HighLID`).
- **Hardware-Accelerated SIMD**:
  - Hand-optimized AVX-512 and AVX2 vector kernels with automatic runtime/compiler dispatch and Scalar fallback.
  - Metrics: `FP32_L2`, `FP32_InnerProduct`, `Uint8_L2`, `FP16_InnerProduct`, and quantized `EVP_InnerProduct`.
- **Graph Optimization & Diagnostics**:
  - Topology pruning (`prune_worst_edges`, `prune_non_mrng_edges`).
  - Analysis suite (`analyze_graph`, connectivity validation, exploration reachability).
- **Label Filtering**: Metadata and boolean ID filtering during search via `deglib::search::Filter`.

---

## Quickstart (C++20)

Below is an example demonstrating how to build an index from data and query nearest neighbors in a few lines of C++20 using `deglib`:

```cpp
#include <deglib/deglib.h>
#include <iostream>
#include <vector>
#include <random>

int main() {
    const uint32_t num_vectors = 1000;
    const uint32_t dims = 128;

    // 1. Generate example feature dataset
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> dataset(num_vectors * dims);
    for (auto& val : dataset) val = dist(rng);

    // 2. Build graph index directly from data (high-level facade)
    auto graph = deglib::build_from_data(
        std::span<const float>(dataset), 
        dims, 
        /*labels=*/{}, 
        /*edges_per_vertex=*/32, 
        deglib::distances::Metric::FP32_L2
    );

    // 3. Query k-nearest neighbors
    std::vector<float> query(dims);
    for (auto& val : query) val = dist(rng);

    const uint32_t k = 10;
    const float eps = 0.1f;
    auto results = graph.search(std::span<const float>(query), k, eps);

    std::cout << "Top " << k << " nearest neighbors:" << std::endl;
    for (const auto& match : results) {
        std::cout << "  Label ID: " << match.getIdentifier()
                  << " | Distance: " << match.getDistance() << std::endl;
    }

    return 0;
}
```

### Dynamic / Incremental Construction

For streaming workloads with dynamic insertions and deletions, use `DynamicExplorationGraph` together with `EvenRegularGraphBuilder`:

```cpp
// 1. Create a mutable high-level graph
auto feature_space = deglib::distances::FloatSpace(dims, deglib::distances::Metric::FP32_L2);
auto graph = deglib::DynamicExplorationGraph::create_empty(num_vectors, /*edges_per_vertex=*/32, feature_space);

// 2. Initialize builder directly with the DynamicExplorationGraph facade
std::mt19937 rng(42);
auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rng);

// 3. Add feature vectors using typed std::span
for (uint32_t i = 0; i < num_vectors; ++i) {
    std::span<const float> vec(&dataset[i * dims], dims);
    builder.addEntry(i, vec);
}
builder.build();

// 4. Query
auto results = graph.search(std::span<const float>(query), /*k=*/10, /*eps=*/0.1f);
```

---

## Prerequisites

- **Compiler**: Modern C++20 compliant compiler:
  - **Windows**: Visual Studio 2022 (MSVC v143+)
  - **Linux**: GCC 11+ or Clang 13+
  - **macOS**: AppleClang 13+
- **Build System**: CMake 3.19+
- **CPU**: x86_64 CPU with AVX2 or AVX-512 support recommended (Scalar fallback available).

---

## Building, Testing & Benchmarking

The project uses standard **CMake Presets** for all platforms and configurations.

### 1. Configure

Select the preset matching your operating system and CPU target:

```bash
# Windows (MSVC + AVX2)
cmake --preset windows-msvc-avx2

# Linux (GCC + AVX2)
cmake --preset linux-gcc-avx2

# macOS (Clang + AVX2)
cmake --preset macos-clang-avx2
```

### 2. Build

Build the project in **Release** configuration:

```bash
# Windows
cmake --build --preset windows-msvc-avx2-release

# Linux
cmake --build --preset linux-gcc-avx2-release

# macOS
cmake --build --preset macos-clang-avx2-release
```

### 3. Run Tests

All unit, integration, and regression tests are executed via `ctest`:

```bash
# Windows
ctest --preset windows-msvc-avx2-release --output-on-failure

# Linux
ctest --preset linux-gcc-avx2-release --output-on-failure

# macOS
ctest --preset macos-clang-avx2-release --output-on-failure
```

### 4. Code Formatting

Format all C++ source files using `clang-format` according to the repository's root `.clang-format` rules:

```bash
# Format all C++ files
clang-format -i $(git ls-files '*.cpp' '*.h')
```

### 5. Run Benchmarks

Benchmark executables are generated in the build directory (`build/<preset>/bench/`):

- `bench_static_data`: Evaluates static dataset graph construction and search throughput/recall.
- `bench_dynamic_data`: Evaluates dynamic vertex additions and deletions under streaming conditions.
- `bench_edge_optimization`: Evaluates continuous edge refinement algorithms and query performance.

#### Dataset Acquisition & Directory Structure

Benchmarks automatically download the required `.tar.gz` archive, extract it, and generate any required ground truth files in the data root directory on first run.

When prepared, the directory hierarchy under `<data-root>` is structured as follows:

```text
<data-root>/
└── sift1m/
    ├── sift1m.tar.gz                   # Downloaded archive
    └── sift1m/                         # Extracted dataset files
        ├── sift1m_base.fvecs           # Base feature vectors
        ├── sift1m_query.fvecs          # Query vectors
        ├── sift1m_explore_query.fvecs  # Exploration queries
        ├── sift1m_explore_entry_vertex.ivecs
        ├── sift1m_explore_groundtruth_top1000.ivecs
        └── sift1m_groundtruth_top100_nb1000000.ivecs
```

#### Specifying the Dataset Directory

You can configure the dataset path using any of the following methods (evaluated in order):
- **Command-line flag**: `--data-path <path>` or `-d <path>`
- **Environment variable**: `DEG_DATA_PATH=/path/to/data`
- **Default fallback**: `./data` in the current working directory

#### Benchmark Execution Examples

- **Linux / macOS**:
  ```bash
  # Run static benchmark on SIFT1M with custom data path
  ./build/linux-gcc-avx2/bench/bench_static_data sift1m --data-path ./data

  # Run dynamic streaming benchmark on Audio dataset
  ./build/linux-gcc-avx2/bench/bench_dynamic_data audio
  ```

- **Windows (PowerShell)**:
  ```powershell
  # Run static benchmark on SIFT1M with custom data path
  .\build\windows-msvc-avx2\bench\Release\bench_static_data.exe sift1m --data-path ./data

  # Run dynamic streaming benchmark on Audio dataset
  .\build\windows-msvc-avx2\bench\Release\bench_dynamic_data.exe audio
  ```

---

## Architecture & API Reference

For a complete breakdown of namespaces, class hierarchies, and header layouts, refer to [ARCHITECTURE.md](ARCHITECTURE.md):

- `deglib::`: High-level user facade (`DynamicExplorationGraph`).
- `deglib::graph::`: Low-level graph representations (`SizeBoundedGraph`, `DynamicGraph`, `ReadOnlyGraph`, `InternalGraph`).
- `deglib::builder::`: Construction and multithreaded builders (`EvenRegularGraphBuilder`).
- `deglib::distances::`: Feature spaces and SIMD metrics (`FloatSpace`, `Metric`).
- `deglib::optimization::`: Graph topology refinement and edge pruning.
- `deglib::analysis::`: Graph health metrics, degree statistics, and reachability.

---

## Datasets

The following standard datasets are supported for benchmarking and reproducing paper results (automatically downloaded and preprocessed by the benchmark executables):

| Dataset   | Archive Name       | Dimension | Base Vectors | Query Vectors | Reference |
|-----------|--------------------|-----------|--------------|---------------|-----------|
| Audio     | `audio.tar.gz`     | 192       | 53,387       | 200           | [Princeton CASS](https://www.cs.princeton.edu/cass/) |
| Enron     | `enron.tar.gz`     | 1369      | 94,987       | 200           | [CMU Enron](https://www.cs.cmu.edu/~enron/) |
| SIFT1M    | `sift.tar.gz`      | 128       | 1,000,000    | 10,000        | [Texmex](http://corpus-texmex.irisa.fr/) |
| DEEP1M    | `deep1m.tar.gz`    | 96        | 1,000,000    | 10,000        | [PPUDA](https://github.com/facebookresearch/ppuda) |
| GloVe-100 | `glove-100.tar.gz` | 100       | 1,183,514    | 10,000        | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |

### Pre-built Graphs

| Dataset   | DEG Graph File |
|-----------|----------------|
| SIFT1M    | [sift_128D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/sift_128D_L2_DEG30.zip) |
| DEEP1M    | [deep1m_96D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/deep1m_96D_L2_DEG30.zip) |
| GloVe-100 | [glove_100D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/glove_100D_L2_DEG30.zip) |
