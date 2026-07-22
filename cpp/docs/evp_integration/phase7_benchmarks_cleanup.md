# Umsetzungsplan Phase 7: Benchmarks & Finaler Cleanup (TDD & Verifikation)

Dieser Detailplan beschreibt Phase 7: Die Überholung der Benchmark-Executables für EVP/FP16 und die finale Bereinigung des Repositories.

---

## Betroffene Komponenten & Dateien in Phase 7

### 1. Benchmark-Infrastruktur & Microbenchmarks

#### [MODIFY] [CMakeLists.txt](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/benchmark/CMakeLists.txt)
- Einbinden der neuen Benchmark-Executables bei `ENABLE_BENCHMARKS=ON`.

#### [NEW] `cpp/benchmark/src/microbench_fp16_l2.cpp`
- Performance-Messung für FP16 L2 Distanzen vs. FP32 SIMD.

#### [NEW] [deglib_build_and_test.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/benchmark/src/deglib_build_and_test.cpp)
- Integrierter Benchmark-Runner für deglib mit EVP-Quantisierung und Asymmetrischen Distanzen.

---

## Verifikationsplan für Phase 7

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target run_evp microbench_fp16_l2
cd bin/Release
microbench_fp16_l2.exe
```
Erfolgs-Kriterium: **Sämtliche Unit-Tests aller Phasen bestehen und Benchmarks kompilieren sauber.**
