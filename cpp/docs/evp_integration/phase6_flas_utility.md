# Umsetzungsplan Phase 6: Standalone FLAS Utility (Optional, TDD)

Dieser Detailplan beschreibt Phase 6 (Optional): Die Extraktion des **Fast Linear Assignment Sorters (FLAS)** als sauberes, unabhängiges Utility-Modul in `deglib` (ohne jeglichen SISAP- oder HDF5-Code).

---

## Betroffene Komponenten & Dateien in Phase 6

### TDD Step 1: FLAS Unit-Tests etablieren

#### [NEW] `test_flas.cpp`
- Testet den Jonker-Volgenant Solver und den Fast Linear Assignment Sorter auf Feature-Vektoren.

---

### TDD Step 2: Modul-Implementierung

#### [NEW] `deglib/include/flas/fast_linear_assignment_sorter.hpp`
- Standalone Algorithmus zur Vorsortierung von Vektor-Features für verbesserte Speicherlokalität.

#### [NEW] `deglib/include/flas/junker_volgenant_solver.hpp`
- Jonker-Volgenant Zuweisungs-Solver.

---

## Verifikationsplan für Phase 6

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target test_flas
cd bin/Release
test_flas.exe
```
Erfolgs-Kriterium: **FLAS Unit-Test läuft erfolgreich.**
