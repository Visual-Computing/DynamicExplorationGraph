# Detaillierter Umsetzungsplan Phase 5: Core Deglib Integration & Auswahlanalyse

Dieses Dokument beschreibt alle Änderungen in den Kerndateien von `deglib` zwischen `main` und `origin/evp` im Detail. Da du **nicht alle** Änderungen aus `origin/evp` übernehmen möchtest, dient dieser Plan als Entscheidungsgrundlage für jede einzelne Funktion/Änderung.

---

## Detail-Analyse aller Änderungen in den Core-Headern

### 1. [analysis.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/analysis.h)
- **Änderung A: Fix für MRNG-Check bei InnerProduct Metric (`5817cfd`)**:
  - *Problem in main*: Die Abstands-Prüfung für die MRNG-Regel (Minimum Relative Neighborhood Graph) nutzte für InnerProduct falsche Größer/Kleiner-Vergleiche, da InnerProduct umgekehrt zu L2-Distanzen skaliert.
  - *Empfehlung*: **ÜBERNEHMEN** (Fixes ein fundamentales Verhalten bei InnerProduct Graph-Analysen).
- **Änderung B: Leere Graphen bei Konnektivitätstests überspringen (`23596fc`)**:
  - Verhinderung von Abstürzen (Divide-by-Zero / NullPointer) wenn `graph.size() == 0`.
  - *Empfehlung*: **ÜBERNEHMEN** (Sicherheits-Fix).
- **Änderung C: Parallelisierung mit Batching (`8921f80`)**:
  - Umstellung von `compute_search_reachability` auf das in Phase 1-4 eingeführte `parallel_for` mit Batch-Größen.
  - *Empfehlung*: **ÜBERNEHMEN** (Performance).

---

### 2. [repository.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/repository.h)
- **Änderung A: Unterstützung für FP16 & extern gelagerte Features**:
  - `getFeature()` und `getFeatureVector()` Template-Methoden zur Unterstützung von `ReadonlyGraphExternal` und quantisierten Datentypen.
  - *Empfehlung*: **ÜBERNEHMEN** (Notwendig für FP16 / EVP Kompatibilität).

---

### 3. [config.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/config.h)
- **Änderungen**:
  - Hinzufügen von Batched Distance Makros / Flag-Definitionen.
  - *Empfehlung*: **ÜBERNEHMEN** (Notwendig für SIMD Distanzen).

---

### 4. [builder.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/builder.h) (Großer Umbau)
Hier befinden sich viele unterschiedliche Features, die wir **einzeln auswählen** können:

- **Änderung 4.1: Multi-Threaded Edge Case & Race Condition Fixes (`28c4d9c`)**:
  - Behebt seltene Deadlocks / Concurrent Modification Bugs in `restoreGraph` und `extendGraph` bei hoher Thread-Anzahl.
  - *Empfehlung*: **ÜBERNEHMEN** (Kritisch für Stabilität).

- **Änderung 4.2: Batched Distances beim Graphenaufbau (`742cac4`, `8921f80`)**:
  - Berechnet Distanzen bei `extendGraph` und `restoreGraph` in Batches (SIMD-Optimierung).
  - *Empfehlung*: **ÜBERNEHMEN** (Beschleunigt Graphenaufbau um 2-4x).

- **Änderung 4.3: Path Search Heuristik für `restoreGraph` (`135df0e`, `3438a97`, `37df2f8`)**:
  - Ersetzt den alten Flood-Fill Algorithmus in `restoreGraph` durch eine Pfad-Suche Heuristik. Macht `restoreGraph` deterministisch.
  - *Empfehlung*: **DISPUTED / USER DECISION** (Sehr gut für Stabilität, ändert aber die Graphen-Struktur leicht im Vergleich zu altem main).

- **Änderung 4.4: StreamingData Algorithmus Schema A, B, C, D (`481b321`, `388005c`)**:
  - Experimenteller StreamingData-Builder für schrittweisen Daten-Ingest.
  - *Empfehlung*: **OPTIONAL / WEGLASSEN** (Falls du nur Kern-Builder brauchst).

- **Änderung 4.5: Simple Edge Swap Tests & Optimization (`7e2faa8`, `b8a96c1`)**:
  - Code für simple Edge Swaps zur Recall-Schätzung während des Builds.
  - *Empfehlung*: **OPTIONAL / WEGLASSEN** (War hauptsächlich für Benchmarks gedacht).

---

### 5. Bereinigung von `cpp/deglib/include_new/`
- Das Verzeichnis `include_new` ist im Branch `evp` vollständig gelöscht worden, da es sich um eine alte experimentelle Kopie handelte.
- *Empfehlung*: **LÖSCHEN** (Hält das Repository sauber).

---

## TDD Verifikationsplan für Phase 5

Die Core Test-Executables werden nacheinander eingebunden und ausgeführt:
- `test_repository.exe`
- `test_filter.exe`
- `test_visited_list_pool.exe`
- `test_search.exe`
- `test_builder.exe`
