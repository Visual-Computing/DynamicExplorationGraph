# Umsetzungsplan Phase 7: Performance-Benchmark & Regressionstest (Audio-Dataset)

Auf Basis deines Feedbacks werden alle unnötigen Microbenchmarks gestrichen. Stattdessen richten wir einen **allgemeinen Performance- & Regressionstest-Benchmark** ein.

Dieser Benchmark lädt ein echtes Benchmark-Dataset (z. B. `audio.tar.gz` von `https://static.visual-computing.com/paper/DEG/audio.tar.gz`), baut den Graphen auf, führt ANN-Suchabfragen durch und misst Recall sowie Throughput (QPS / Latenz), um **Performance-Degradationen zuverlässig zu erkennen**.

---

## Betroffene Komponenten & Dateien in Phase 7

### 1. Benchmark-Executable & Dataset Setup
- **[NEW] `cpp/benchmark/src/deglib_audio_benchmark.cpp`**:
  - Lädt den `audio`-Datensatz (`audio_base.fvecs`, `audio_query.fvecs`, `audio_groundtruth.ivecs`).
  - Baut den DEG-Graphen (`EvenRegularGraphBuilder`) auf.
  - Führt Suchabfragen mit verschiedenen Epsilon-Werten durch.
  - Berechnet Exaktheit (Recall @ K), Durchsatz (QPS) und Bauzeit.
- **[MODIFY] [CMakeLists.txt](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/benchmark/CMakeLists.txt)**:
  - Registrierung des Benchmark-Targets `run_audio_benchmark`.

---

## Verifikationsplan für Phase 7

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target run_audio_benchmark
cd bin/Release
run_audio_benchmark.exe
```
Erfolgs-Kriterium: **Graph wird aufgebaut, Suchabfragen werden ausgeführt und die Performance-Metriken (QPS, Recall, Bauzeit) werden sauber ausgegeben.**
