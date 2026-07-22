# Detaillierter Vergleich: Code im `origin/evp` Branch vs. `main` Branch (`deglib/include/`)

Dieser Bericht listet **exakt alle Dateien, Methoden, Code-Zeilen und Features** auf, die im `origin/evp` Branch existieren, aber im originalen `main` Branch fehlen bzw. abweichen.

Ziel ist es zu prüfen, ob noch weitere Funktionen aus `origin/evp` in `main` übernommen werden sollen.

---

## 📋 1. Gesamt-Übersicht aller 7 geänderten Header-Dateien

In `deglib/include/` unterscheiden sich exakt **7 Dateien** zwischen `origin/main` und `origin/evp`:

| Datei | Zeilendiff in EVP | Im EVP-Branch enthaltene Features | Status in Main / Vorschlag |
|---|---|---|---|
| **`deglib/include/builder.h`** | +1.489 / -1.489 | 1. **StreamingData Algorithmen** (`SchemeA`, `SchemeB`, `SchemeC`, `SchemeD`) in `extendGraphUnknownLID`<br>2. **Simple Edge Swaps** (`use_simple_edge_swaps_`) für Ablations-Tests<br>3. **Parallel Task Batching** (`extend_thread_task_size`) | **1 & 2**: Auf Wunsch weggelassen.<br>**3**: In Main übernommen. |
| **`deglib/include/analysis.h`** | +300 / -300 | 1. **MRNG Vergleiche für InnerProduct**: Korrektur der Distanzordnung bei Skalarprodukt-Graphen (`checkRNG`)<br>2. **Histograms & Graph Components**: Analysefunktionen für Graphzusammenhang | **1 & 2**: In Main übernommen. |
| **`deglib/include/repository.h`** | +40 / -10 | 1. **`ivecs_read()`**: Liest 32-Bit Integer `.ivecs` Groundtruth-Dateien<br>2. **`inline` Modifier**: Vermeidung von ODR-Violations beim Mehrfach-Include | **1 & 2**: In Main übernommen. |
| **`deglib/include/concurrent.h`** | +6 / -0 | **5-Parameter `parallel_for`**: Unterstützt `batchSize` Parameter für schnellere Thread-Batches | **In Main übernommen** (mit 4-Arg Rückwärtskompatibilität). |
| **`deglib/include/config.h`** | +89 / -12 | Parameter-Parsing für `extend_thread_task_size` und `improve_thread_task_size` | **In Main übernommen**. |
| **`deglib/include/deglib.h`** | +10 / -0 | Include-Einträge für `evp_quantize.h` und `fast_linear_assignment_sorter.h` | **In Main übernommen**. |
| **`deglib/include/visited_list_pool.h`** | +60 / -60 | Reine Clang-Formatierung (Whitespace / Einrückungen) | **In Main übernommen**. |

---

## 🔍 2. Detaillierter Zeilen- & Methoden-Diff pro Datei

### A. `deglib/include/builder.h`
Der EVP-Branch erweitert `EvenRegularGraphBuilder` um zwei große Forschungs-Features:

#### 1. StreamingData Algorithmen (Schemas A, B, C, D)
- **Code im EVP-Branch**:
  ```cpp
  enum OptimizationTarget {
      StreamingData_SchemeA,  // Abgleich mit naheliegendsten Nachbarn
      StreamingData_SchemeB,  // Kürzeste Kante entfernen
      StreamingData_SchemeC,  // Schlechte Kanten des Nachbarn entfernen (Default)
      StreamingData_SchemeD,  // Minimierung der Distortion
      HighLID,
      LowLID
  };
  ```
- **Funktion `extendGraphUnknownLID(const BuilderAddTask& add_task)`**:
  - ~180 Zeilen Code im EVP-Branch für schrittweisen Single-Threaded Aufbau ohne LID-Kenntnis.
- **Entscheidung**: *Vom User als nicht benötigt / "weglassen" markiert.*

#### 2. Simple Edge Swaps
- **Code im EVP-Branch**:
  - `use_simple_edge_swaps_` Bool-Flag im Konstruktor.
  - Tausch-Schleifen zur nachträglichen Optimierung in der Graphbau-Restaurierungsphase.
- **Entscheidung**: *Vom User als nicht benötigt / "weglassen" markiert.*

---

### B. `deglib/include/analysis.h`
- **Code im EVP-Branch**:
  - `checkRNG()`: Verwendet bei `Metric::InnerProduct` umgekehrte Relationsoperatoren (`>` statt `<`), da beim Skalarprodukt höhere Werte eine höhere Ähnlichkeit bedeuten.
  - `calc_edge_weight_histogram()`, `count_graph_components()`, `compute_in_degrees()`, `check_graph_bidirectional()`, `count_search_reachable()`.
- **Status in Main**: Alle Korrekturen wurden übernommen.

---

### C. `deglib/include/repository.h`
- **Code im EVP-Branch**:
  - Neue Funktion `inline auto ivecs_read(const char* fname, size_t& d_out, size_t& n_out)` liest Integer-Matrizen (Groundtruth-Nachbarn).
  - Alle Funktionen wurden als `inline` deklariert.
- **Status in Main**: Wurde vollständig übernommen.

---

### D. Aus `cpp/sisap/flas/` nach `deglib/include/flas/` überführt (Phase 6)
- **Code im EVP-Branch**: Befand sich ursprünglich im vom User abgelehnten `sisap/` Ordner.
- **Vom EVP-Branch übernommen**:
  - `deglib/include/flas/fast_linear_assignment_sorter.h` (454 Zeilen)
  - `deglib/include/flas/junker_volgenant_solver.h` (245 Zeilen)
  - `deglib/include/flas/map_field.h` (33 Zeilen)
- **Status in Main**: Erfolgreich als sauberes Modul unter `deglib/include/flas/` integriert.

---

## ❓ 3. Zusammenfassende Prüffrage an den User

Es gibt im `origin/evp` Branch im Verzeichnis `deglib/include/` **KEINE weiteren unübernommenen Funktionen oder Dateien** mehr, außer:
1. `StreamingData_SchemeA..D` in `builder.h` (wolltest du weglassen)
2. `SimpleSwap` Logik in `builder.h` (wolltest du weglassen)

Möchtest du bezüglich `deglib/include/` noch irgendetwas aus dem `origin/evp` Branch übernehmen oder ist `deglib/include/` damit zu 100% fertig gestellt?
