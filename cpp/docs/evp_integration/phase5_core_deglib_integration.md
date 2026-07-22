# Umsetzungsplan Phase 5: Core Deglib Integration & Cleanup (TDD)

Dieser Detailplan beschreibt Phase 5: Das Zusammenführen der neuen Distanzen & Graphenstrukturen in die Kern-Komponenten von `deglib` (`builder.h`, `search.h`, `repository.h`, `analysis.h`) sowie die Bereinigung des veralteten `include_new/`-Verzeichnisses.

---

## Betroffene Komponenten & Dateien in Phase 5

### TDD Step 1: Core Test-Suite überarbeiten & aktualisieren

#### [MODIFY] [test_builder.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/test_builder.cpp)
- Testet Graphenaufbau, Swap-Test-Optimierungen und Multi-Threading Graph Extension.

#### [MODIFY] [test_search.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/test_search.cpp)
- Testet Graph-Suche mit variablen Recall-Zielen und Batched Distanz-Evaluierungen.

#### [MODIFY] [test_filter.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/test_filter.cpp) & [test_repository.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/test_repository.cpp)
- Testet Filter-Operationen und Repository Feature-Speicherung.

#### [MODIFY] [test_visited_list_pool.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/test_visited_list_pool.cpp)
- Testet concurrent VisitedListPool Zuweisungen.

---

### TDD Step 2: Core Header Updates & Redundanz-Löschung

#### [MODIFY] [builder.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/builder.h)
- Integration von Batched Distance Distanzberechnungen während des Graph-Builds.

#### [MODIFY] [search.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/search.h) & [analysis.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/analysis.h)
- Unterstützung für externe Feature-Arrays und schnellere Reachability-Berechnung.

#### [DELETE] `cpp/deglib/include_new/*`
- Das veraltete und unvollständige Duplikat-Verzeichnis `include_new` wird vollständig aus dem Repository gelöscht.

---

## Verifikationsplan für Phase 5

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target test_builder test_search test_filter test_repository test_visited_list_pool
cd bin/Release
test_builder.exe
test_search.exe
test_filter.exe
test_repository.exe
test_visited_list_pool.exe
```
Erfolgs-Kriterium: **Gesamte Core Deglib Test-Suite grün.**
