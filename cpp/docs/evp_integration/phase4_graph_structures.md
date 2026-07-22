# Umsetzungsplan Phase 4: Readonly Graph External & Graphenstrukturen (TDD)

Dieser Detailplan beschreibt Phase 4: Die Integration der erweiterten Graphenstruktur `readonly_graph_external.h`, welche es erlaubt, Vektor-Features (z. B. FP16 oder EVP-quantisiert) außerhalb der Hauptgraphenstruktur im Speicher zu halten.

---

## Betroffene Komponenten & Dateien in Phase 4

### TDD Step 1: Unit-Tests übernehmen

#### [NEW] [test_readonly_graph_external.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/graph/test_readonly_graph_external.cpp)
- Testet Laden, Speichern, Nachbarzugriffe und externe Feature-Speicheranbindung für `ReadonlyGraphExternal`.

#### [NEW] [test_sizebounded_graph.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/graph/test_sizebounded_graph.cpp)
- Testet dynamisches Größenwachstum, Neighbor-Pruning und Speicher-Management des `SizeboundedGraph`.

---

### TDD Step 2: Implementierung & Anpassungen

#### [NEW] [readonly_graph_external.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/graph/readonly_graph_external.h)
- Neue Graphen-Klasse mit Entkopplung von Graph-Adjazenzlisten und Feature-Arrays.

#### [MODIFY] [readonly_graph.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/graph/readonly_graph.h)
- Anpassungen an In-Memory Layouts und Kompatibilität mit externen Repräsentationen.

#### [MODIFY] [sizebounded_graph.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/graph/sizebounded_graph.h)
- Optimiertes Neighbor-Pruning, Entfall überflüssiger Approximation.

---

## Verifikationsplan für Phase 4

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target test_readonly_graph_external test_sizebounded_graph
cd bin/Release
test_readonly_graph_external.exe
test_sizebounded_graph.exe
```
Erfolgs-Kriterium: **Graphen-Tests bestehen 100% ohne Speicherlecks oder Abstürze.**
