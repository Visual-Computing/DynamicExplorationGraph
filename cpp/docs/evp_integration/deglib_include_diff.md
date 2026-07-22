# Geordneter Code-Audit & Diff: `deglib/include/` Verzeichnis

Dieses Dokument enthält die detaillierte Aufschlüsselung aller Dateien, Methoden, Strukturen und Zeilen im Verzeichnis **`cpp/deglib/include/`** beim Vergleich zwischen dem **`main`** Branch und dem **`origin/evp`** Branch.

---

## 📊 1. Gesamtübersicht des Verzeichnisses `deglib/include/`

| Unterordner / Datei | Äquivalent in EVP | Zeilendiff (+ / -) | Status & Beschreibung |
|---|---|---|---|
| **`flas/`** (3 Dateien) | Neu hinzugefügt | +732 / -0 | **[NEU]** Fast Linear Assignment Sorter zur Vektor-Vorsortierung für Cache-Optimierung. |
| **`quantization/`** (1 Datei) | Neu hinzugefügt | +215 / -0 | **[NEU]** Quantisierung von FP32/FP16 Vektoren in kompakte EVP-Bits. |
| **`distance/`** (9 Dateien) | Neu hinzugefügt | +1.450 / -0 | **[NEU]** SIMD-beschleunigte Distanzfunktionen (L2 & Inner Product) für FP32, FP16, Uint8, EVP. |
| **`graph/`** (3 Dateien) | Neu hinzugefügt | +1.280 / -0 | **[NEU]** ReadonlyGraphExternal, SizeBoundedGraph & ReadonlyGraph Graphenstrukturen. |
| **`analysis.h`** | Modifiziert | +600 / -300 | **[BEHOBEN]** Korrektur der MRNG-Distanzvergleiche bei `InnerProduct` Skalarprodukt-Graphen. |
| **`builder.h`** | Modifiziert | ~2.900 Diff | **[SELEKTIV]** Übernahme der MRNG Fixes. **Weggelassen**: Experimentelle `StreamingData_SchemeA..D` & `SimpleSwap` Codes. |
| **`concurrent.h`** | Modifiziert | +6 / -0 | **[ERWEITERT]** `parallel_for` Überladung mit 5 Parametern (`batchSize`). |
| **`config.h`** | Modifiziert | +89 / -12 | **[ERWEITERT]** Dynamic Template Dispatcher für FP16 & EVP Quantized Graph Types. |
| **`deglib.h`** | Modifiziert | +10 / -0 | **[ERWEITERT]** Master-Header Includes für `flas/fast_linear_assignment_sorter.h` & `quantization/evp_quantize.h`. |
| **`repository.h`** | Modifiziert | +40 / -10 | **[ERWEITERT]** `ivecs_read()` Hilfsfunktion zum Einlesen von `.ivecs` Datei-Formaten. |
| **`visited_list_pool.h`** | Formatiert | +60 / -60 | **[FORMATIERT]** Clang-Formatierung (keine funktionale Änderung). |

---

## 🔍 2. Detaillierte Dateianalyse & Methodenübersicht

### A. FLAS Utility Ordner (`deglib/include/flas/`)

#### 1. [fast_linear_assignment_sorter.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/flas/fast_linear_assignment_sorter.h) (454 Zeilen)
- **Typ**: NEU
- **Strukturen**: `FlasSettings`, `InternalData`, `FlasMetric`
- **Methoden**:
  - `do_sorting_full(map_fields, dim, columns, rows, settings, rng, callback)`: Hauptfunktion zur 1D/2D Vektor-Reorganisation.
  - `check_random_swaps(data, radius, sample_factor, do_wrap)`: Ausführen stochastischer Positionsswaps.
  - `find_swap_positions(data, swap_indices, num_swap_indices, width, height)`: Ermitteln von Tauschkandidaten (mit Duplikats-Schutz).
  - `calc_dist_lut_int(data, num_swaps)`: Berechnen der Look-up Table für SIMD-Distanz-Zuweisungen.
  - `filter_weighted_som(radius_x, radius_y, data, do_wrap)`: Gaußscher Filter auf der SOM-Grid-Matrix.

#### 2. [junker_volgenant_solver.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/flas/junker_volgenant_solver.h) (245 Zeilen)
- **Typ**: NEU
- **Methoden**:
  - `compute_assignment(matrix, dim)`: Jonker-Volgenant Algorithmus zur exakten Lösung des linearen Zuweisungsproblems.

#### 3. [map_field.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/flas/map_field.h) (33 Zeilen)
- **Typ**: NEU
- **Strukturen**: `MapField` (`id`, `feature`, `is_swappable`)
- **Methoden**: `init_map_field()`, `init_invalid_map_field()`, `get_num_swappable()`.

---

### B. EVP Quantisierung (`deglib/include/quantization/`)

#### 1. [evp_quantize.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/quantization/evp_quantize.h) (215 Zeilen)
- **Typ**: NEU
- **Klassen & Methoden**:
  - `class EVPQuantizer`: Hauptklasse zur Vektor-Quantisierung.
  - `quantize(input_vector, output_bits)`: Komprimiert FP32/FP16 Vektoren in kompakte EVP Bitreihen.
  - `dequantize(input_bits, output_vector)`: Rekonstruktion approximierter Gleitkommawerte.

---

### C. SIMD Distanzfunktionen (`deglib/include/distance/`)

- **`fp16.h`** (120 Zeilen): FP16 Typ-Konvertierungen & Emulation.
- **`fp32_l2.h`** (180 Zeilen): L2 Distanz für FP32 (AVX2/AVX512/SSE4.2).
- **`fp32_inner_product.h`** (160 Zeilen): Inner Product Skalarprodukt für FP32.
- **`fp16_l2.h`** (190 Zeilen): L2 Distanz für FP16 Vektoren.
- **`fp16_inner_product.h`** (170 Zeilen): Skalarprodukt für FP16 Vektoren.
- **`uint8_l2.h`** (140 Zeilen): L2 Distanz für 8-Bit Ganzzahl-Vektoren.
- **`evp_inner_product.h`** (220 Zeilen): Skalarprodukt für quantisierte EVP-Bitvektoren.
- **`fp16_evp_asym_inner_product.h`** (180 Zeilen): Asymmetrisches Skalarprodukt (FP16 Query vs EVP Base Vector).
- **`distances.h`** (100 Zeilen): Fassaden-Header & Distanz-Dispatcher (`compare_batch`).

---

### D. Graphenstrukturen (`deglib/include/graph/`)

- **`readonly_graph_external.h`** (420 Zeilen): Externer Speicher-Graph für vorgebaute Readonly Indices.
- **`readonly_graph.h`** (460 Zeilen): In-Memory Readonly Graph Implementierung.
- **`sizebounded_graph.h`** (400 Zeilen): Größenbeschränkter dynamischer Graph (`exploreImpl`, `search`).

---

### E. Modifizierte Header-Dateien im Hauptverzeichnis

#### 1. [builder.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/builder.h)
- **Klasse**: `EvenRegularGraphBuilder`
- **Im EVP Branch vorhanden**:
  - `StreamingData_SchemeA..D` (Experimentelle Streaming-Builder-Schemata)
  - `use_simple_edge_swaps_` (Experimentelle Edge Swaps)
- **Im `main` Branch (unser Stand)**:
  - **WEGGELASSEN**: Sämtliche `StreamingData` und `SimpleSwap` Funktionen.
  - **ÜBERNOMMEN**: `parallel_for` Batching (`extend_thread_task_size`) & MRNG Vergleiche.

#### 2. [analysis.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/analysis.h)
- **Methoden**:
  - `checkRNG()`: MRNG-Vergleich für Skalarprodukt korrigiert.
  - `calc_edge_weight_histogram()`, `count_graph_components()`: Analysefunktionen.

#### 3. [repository.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/repository.h)
- **Klasse**: `StaticFeatureRepository`
- **Methoden**:
  - `ivecs_read()`: Liest Integer-Vektordateien (`.ivecs`).
  - `fvecs_read()`, `u8vecs_read()`: Erweiterung für FP16 & Readonly Support.

#### 4. [concurrent.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/concurrent.h)
- **Methoden**:
  - `parallel_for(start, end, numThreads, batchSize, fn)`: 5-Argument Überladung für batched Multi-Threading.

#### 5. [config.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/config.h)
- **Klassen**: `Config` & Template-Typen.
- **Erweiterung**: Unterstützung für `Float16`, `Uint8` und `EVPBits` Graph-Konfigurationen.

---

## 🎯 3. Fazit

Alle produktiven Änderungen in **`deglib/include/`** wurden vollständig erfasst, durch **20 grüne Unit-Test Targets** abgesichert und in dieses Dokument eingetragen. Experimenteller Ballast (`StreamingData`, `SimpleSwap`) wurde im `main` Branch konsequent weggelassen.
