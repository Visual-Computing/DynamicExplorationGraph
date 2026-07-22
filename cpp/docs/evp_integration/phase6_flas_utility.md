# Umsetzungsplan Phase 6: Standalone FLAS Utility (TDD)

In Phase 6 wird der **Fast Linear Assignment Sorter (FLAS)** als sauberes, unabhängiges Modul in `deglib` integriert – ohne jegliche Abhängigkeiten zu SISAP oder HDF5.

---

## Speicherort & Struktur für FLAS

Die Dateien werden sauber unter `deglib/include/flas/` abgelegt:

### 1. Header-Dateien (Modul)
- **[NEW] [fast_linear_assignment_sorter.hpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/flas/fast_linear_assignment_sorter.hpp)**: Algorithmus zur Umordnung/Reorganisation von Vektor-Features für optimale Cache-Lokalität.
- **[NEW] [junker_volgenant_solver.hpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/flas/junker_volgenant_solver.hpp)**: Jonker-Volgenant Zuweisungs-Solver.
- **[NEW] [map_field.hpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/flas/map_field.hpp)**: Hilfsstruktur für Map-Felder.

### 2. Unit-Test (TDD)
- **[NEW] `cpp/test/src/flas/test_flas.cpp`**: Testet die Korrektheit des Jonker-Volgenant Solvers und des FLAS-Sorters.

---

## Verifikationsplan für Phase 6

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target test_flas
cd bin/Release
test_flas.exe
```
Erfolgs-Kriterium: **`test_flas.exe` schlägt nicht fehl und verifiziert die Zuweisungs-Logik.**
