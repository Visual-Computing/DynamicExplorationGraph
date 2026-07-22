# Master-Integrationsplan: EVP Branch Integration in Main (TDD)

Dieser Gesamtplan regelt die geordnete, schrittweise Übernahme der Funktionalitäten des `origin/evp` Branches in den `main` Branch.

> [!IMPORTANT]
> **Test-Driven Development (TDD) ist verpflichtend!** Jede Phase bringt zuerst die entsprechenden Unit-Tests ein und stellt sicher, dass alle Tests auf Windows mit MSVC (Visual Studio 2022) 100% grün bestehen.
> Der SISAP-Ordner und HDF5-Reader werden vollständig ausgeschlossen.
> Detaillierter Code-Audit aller Header: Siehe [Geordneter Diff für `deglib/include/`](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/deglib_include_diff.md).

---

## Übersicht der Phasen & Status

### [Phase 1: Test-Infrastruktur & CPU Feature Detection](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase1_test_infrastructure.md) (✅ ABGESCHLOSSEN - Siehe [Ergebnisse Phase 1](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase1_results.md))
- CMake-Testumgebung mit GoogleTest & MSVC AVX2/AVX512/SSE4.2 CPU Detection.
- Executable: `test_cpu_features.exe` (100% PASSED).

### [Phase 2: FP16 & Basis-Distanzfunktionen TDD](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase2_fp16_and_distances.md) (✅ ABGESCHLOSSEN - Siehe [Ergebnisse Phase 2](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase2_results.md))
- Tests: `test_fp16`, `test_fp32_l2`, `test_fp32_inner_product`, `test_fp16_inner_product`, `test_uint8_l2`, `test_config_cascade`, `test_distances`.
- Executables: 7/7 PASSED (100% PASSED).

### [Phase 3: EVP Quantisierung & Asymmetrische Distanzen TDD](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase3_evp_quantization.md) (✅ ABGESCHLOSSEN - Siehe [Ergebnisse Phase 3](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase3_results.md))
- Tests: `test_evp_quantize`, `test_evp_inner_product`, `test_fp16_evp_asym_inner_product`.
- Executables: 3/3 PASSED (36/36 Tests 100% PASSED).

### [Phase 4: ReadonlyGraphExternal & Graphenstrukturen TDD](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase4_graph_structures.md) (✅ ABGESCHLOSSEN - Siehe [Ergebnisse Phase 4](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase4_results.md))
- Tests: `test_readonly_graph_external`, `test_sizebounded_graph`.
- Executables: 2/2 PASSED (37/37 Tests 100% PASSED).

### [Phase 5: Core Deglib Integration (Builder, Search, Repository) TDD](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase5_core_deglib_integration.md) (✅ ABGESCHLOSSEN - Siehe [Ergebnisse Phase 5](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase5_results.md))
- Tests: `test_builder`, `test_search`, `test_filter`, `test_repository`, `test_visited_list_pool`.
- Executables: 5/5 PASSED (78/78 Tests 100% PASSED, StreamingData/SimpleSwap ausgeschlossen).

### [Phase 6: Standalone FLAS Utility TDD](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase6_flas_utility.md) (✅ ABGESCHLOSSEN - Siehe [Ergebnisse Phase 6](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase6_results.md))
- Subdir: `cpp/deglib/include/flas/` (`fast_linear_assignment_sorter.hpp`, `junker_volgenant_solver.hpp`, `map_field.hpp`).
- Executable: `test_flas.exe` (6/6 Tests 100% PASSED).

### [Phase 7: Performance-Benchmark & Regressionstest](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase7_benchmarks_cleanup.md) (✅ ABGESCHLOSSEN - Siehe [Ergebnisse Phase 7](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/docs/evp_integration/phase7_results.md))
- Audio Benchmark Executable: `run_audio_benchmark.exe` (Build Throughput ~33k vec/s, Search QPS ~7.2k queries/s).

---

## Fazit & Verifikation der Gesamtintegration

Sämtliche 19 Unit-Test Executables sowie der Performance-Benchmark wurden auf Windows MSVC x64 C++20 kompiliert und erfolgreich verifiziert. Alle 7 Phasen der EVP-Integration sind vollständig und fehlerfrei abgeschlossen.
