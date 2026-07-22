# Umsetzungsplan Phase 1: Test-Infrastruktur, CPU Feature Detection & Build System (TDD-Basis)

Dieser Detailplan beschreibt Phase 1 unter strikter Anwendung von **Test-Driven Development (TDD)**:
Jede Funktionalität – einschließlich der automatischen CPU-Feature-Erkennung (AVX2/AVX512/SSE4.2/F16C) und der Compiler-Flags – wird durch dedizierte Unit-Tests abgedeckt.

---

## Betroffene Komponenten & Dateien in Phase 1

### TDD Step 1: Tests für CPU-Features & Build-Konfiguration etablieren

#### [NEW] `test_cpu_features.cpp` (in `cpp/test/src/test_cpu_features.cpp`)
- **Ziel**: Überprüft zur Laufzeit die durch CMake erkannten und gesetzten CPU-Instruktionssätze.
- **Inhalt des Tests**:
  - Prüft CPUID-Befehlsausführung unter Windows/MSVC (`__cpuid`, `__cpuidex`).
  - Verifiziert, dass compilierte AVX2/SSE4.2/AVX512-Makros (z. B. `__AVX2__`, `__AVX512F__`, `__F16C__`) mit den tatsächlichen CPU-Fähigkeiten der ausführenden Hardware übereinstimmen.
  - Prüft, dass SIMD-Grundbefehle ohne Faults ausgeführt werden können.

#### [NEW] [test_main.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/test_main.cpp)
- Minimaler Google Test Runner Entry-Point.

#### [NEW] [test_helpers.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_helpers.h)
- Hilfsfunktionen für Test-Setup, Zufallsdaten und Epsilon-Toleranz-Prüfungen.

---

### TDD Step 2: Implementierung der CMake CPU-Detection & Build Target Setup

#### [NEW] [CMakeLists.txt](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/CMakeLists.txt)
- Erstellung der CMake Test-Konfiguration.
- Hinzufügen des Makros `add_deglib_test(name source)` und des neuen Test-Targets `test_cpu_features`.

#### [MODIFY] [DetectCPUFeatures.cmake](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/cmake_modules/DetectCPUFeatures.cmake)
- Einführung des Makros `check_cpu_feature(CODE FLAGS VAR)` mit `check_cxx_source_runs`.
- Hinzufügen von AVX512F (`AVX512F_MIN_PROG`), AVX2, SSE4.2 Proben.

#### [MODIFY] [CMakeLists.txt](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/CMakeLists.txt)
- Erstellung des `compile-options` INTERFACE Targets.
- Trennung und Anwendung der MSVC Compiler Flags (`/arch:AVX2`, `/arch:AVX512`, `/arch:SSE2`, `/EHsc`, `/W4`, `/O2`).

---

## Verifikationsplan für Phase 1

### Automated Test Steps (MSVC C++20 / Windows)
1. **CMake Re-Configuration**:
   ```cmd
   cd c:\Lang\cpp\DynamicExplorationGraph\cpp
   mkdir build
   cd build
   cmake -G "Visual Studio 17 2022" -A x64 ..
   ```
2. **Kompilierung & Ausführung des CPU-Feature Unit-Tests**:
   ```cmd
   cmake --build . --config Release --target test_cpu_features
   cd bin/Release
   test_cpu_features.exe
   ```
Erfolgs-Kriterium: **`test_cpu_features.exe` schlägt nicht fehl und verifiziert die korrekten MSVC Architecture Flags.**
