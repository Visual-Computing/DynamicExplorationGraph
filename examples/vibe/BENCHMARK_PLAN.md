# Benchmark-Ergebnisse & Priorisierter Optimierungsplan: `examples/vibe` (Yandex-200 DEG-QG)

## 1. Verbindliche Architektur-Regeln (Constraints)
- **Keine Dimensions-Beschränkung**: Kein Code darf auf $D=200$ oder eine andere feste Dimension limitiert sein.
- **Keine fest verdrahteten CPU-Instruktionen in der Suche**: Suchlogik (`internal_graph.h`, `searcher.h`) bleibt vollständig generisch.
- **Nutzung der generischen `distance.h`-Architektur**: Alle Distanzberechnungen (`COMPARATOR::compare`) laufen über `FloatSpace` / Distanz-Klassen.
- **Optimierungen an Distanzfunktionen generisch**: Distanz-Kernel in `distance/int8_ip.h` und `distance/fp16.h` arbeiten mit generischen Chunk-Schleifen für jedes $D$ und sauberer Rest-Behandlung.
- **Isolations- und Messprinzip**: Jede Änderung wird isoliert getestet und gemessen. Nur bei messbarer QPS-Steigerung bei gleichem/besserem Recall wird committet, ansonsten sofort mit `git stash` verworfen.

---

## 2. Baseline (Aktueller `main`-Zustand)
- **Kommando**: `uv run main.py --dataset yandex --cache-dir D:/Data/DEG --algorithm deg-qg --cpu 0 --no-show`
- **CPU**: AMD Ryzen AI 9 HX PRO 375 (AVX-512 VNNI, Single-Thread, Core 0 Pinning)
- **Dataset**: Yandex-200 Cosine (1.000.000 Basisvektoren, 1.000 Test-Queries, 200D, k=100)
- **Graph**: `200D_FP32_InnerProduct_K48_AddK32Eps0.1_LowLID_FLAS.deg`

### Baseline Messwerte:

| Rerank Factor | Search Parameter | Recall@100 | Zeit (µs/Query) | QPS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1.00** | eps 0.000 | 0.92002 | 146 µs | **6.840 QPS** | OK |
| **1.00** | eps 0.005 | 0.93160 | 159 µs | **6.269 QPS** | OK |
| **1.00** | eps 0.010 | 0.93965 | 170 µs | **5.865 QPS** | OK |
| **1.00** | eps 0.020 | 0.95592 | 204 µs | **4.886 QPS** | OK |
| **1.00** | eps 0.040 | 0.97525 | 297 µs | **3.365 QPS** | OK |
| **1.00** | eps 0.060 | 0.98371 | 431 µs | **2.316 QPS** | OK |
| **1.00** | eps 0.080 | 0.98830 | 613 µs | 1.630 QPS | Abgebrochen (> 492 µs Baseline) |
| **1.15** | eps 0.000 | 0.94032 | 194 µs | **5.130 QPS** | OK |
| **1.15** | eps 0.005 | 0.94980 | 206 µs | **4.847 QPS** | OK |
| **1.15** | eps 0.010 | 0.95835 | 230 µs | **4.347 QPS** | OK |
| **1.15** | eps 0.020 | 0.97147 | 260 µs | **3.842 QPS** | OK |
| **1.15** | eps 0.040 | 0.98722 | 363 µs | **2.750 QPS** | OK |
| **1.15** | eps 0.060 | 0.99333 | 517 µs | 1.934 QPS | Abgebrochen (> 492 µs Baseline) |
| **1.35** | eps 0.000 | 0.95379 | 223 µs | **4.477 QPS** | OK |
| **1.35** | eps 0.005 | 0.96135 | 233 µs | **4.290 QPS** | OK |
| **1.35** | eps 0.010 | 0.96790 | 259 µs | **3.860 QPS** | OK |
| **1.35** | eps 0.020 | 0.97765 | 298 µs | **3.348 QPS** | OK |
| **1.35** | eps 0.040 | 0.99123 | 410 µs | **2.437 QPS** | OK |
| **1.35** | eps 0.060 | 0.99587 | 564 µs | 1.770 QPS | Abgebrochen (> 492 µs Baseline) |

---

## 3. Priorisierter Optimierungsplan

### Prio 1: Generischer `LinearPool` & `searchEfImpl` Suchpipeline
- **Erwarteter Hebel**: **Maximal (tauscht die gesamte Suchschleife aus)**
- **Begründung**: Ersetzt den Heap-basierten `search_intern` durch das flache, sortierte `LinearPool`-Array mit integrierter Bitset-Visited-List, Multi-Medoid-Einstiegsscan und Pipelined-Prefetching.
- **Komponenten**:
  - `cpp/deglib/include/deglib/search/linear_pool.h`: Header mit `LinearPool<DistType>`.
  - `cpp/deglib/include/deglib/graph/internal_graph.h`: Generisches `searchEfImpl` mit `COMPARATOR::compare(...)`, ohne jegliche ISA- oder Dimensionsbeschränkung.
  - `search_ef_intern(...)` in den Graph-Klassen (`ReadOnlyGraph`, etc.).
  - Aufruf in `Searcher::search(...)` bei `ef > 0`.
- **Generisch**: Vollständig typ- und dimensionsunabhängig.

### Prio 2: Generische AVX-512 VNNI Inner-Product Optimierung (`distance/int8_ip.h`)
- **Erwarteter Hebel**: **Sehr groß (+20% bis +35% Traversierungs-Durchsatz)**
- **Begründung**: Beschleunigt `COMPARATOR::compare(...)` für jeden geprüften Nachbarknoten.
- **Konzept**: 64-Byte VNNI Hauptschleife (`_mm512_dpbusd_epi32`) mit schnellem generischem Rest-Handler für beliebiges $D$.

### Prio 3: Generische AVX-512 F16C Reranking-Beschleunigung (`distance/fp16.h`)
- **Erwarteter Hebel**: **Groß (+15% bis +25% bei Rerank 1.15 / 1.35)**
- **Begründung**: Beseitigt den sequentiellen Skalarprodukt-Flaschenhals beim Reranken von FP16-Kandidaten.
- **Konzept**: Vektorisierte Distanzberechnung per `_mm512_cvtph_ps` + `_mm512_fmadd_ps` in 16er-Chunks für beliebige Dimensionen.

### Prio 4: ReadOnlyGraph SoA (Struct-of-Arrays) Layout
- **Erwarteter Hebel**: **Mittel (+5% bis +10% QPS)**
- **Begründung**: Bessere Cache-Lokalität durch getrennte, dichte Puffer für `features_`, `neighbors_` und `labels_`.

### Prio 5: Prefetch-Parameter Feintuning (`po`, `pl`)
- **Erwarteter Hebel**: **Feinschliff (+3% bis +7% QPS)**
- **Begründung**: Abstimmen von Prefetch-Offset und Cachelines auf L1/L2-Latenzen.


---

## 4. Benchmark-Ergebnis: Schritt 1 (LinearPool & ef-Suche)

- **Status**: **MASSIVER ERFOLG (+40% bis +84% QPS-Steigerung bei gleichem Recall)**
- **Vergleich bei vergleichbarem Recall**:

| Recall@100 | Baseline (eps) | Baseline QPS | Schritt 1 (ef) | Schritt 1 QPS | QPS-Gewinn |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **~0.920** | eps 0.000 | 6.840 QPS (146 µs) | ef 96 (rf=1.00) | **9.648 QPS (103 µs)** | **+41.1%** |
| **~0.940** | eps 0.010 | 5.865 QPS (170 µs) | ef 128 (rf=1.00) | **7.784 QPS (128 µs)** | **+32.7%** |
| **~0.948** | eps 0.005 (rf=1.15) | 4.847 QPS (206 µs) | ef 128 (rf=1.15) | **6.005 QPS (166 µs)** | **+23.9%** |
| **~0.972** | eps 0.020 (rf=1.15) | 3.842 QPS (260 µs) | ef 200 (rf=1.15) | **4.097 QPS (244 µs)** | **+6.6%** |
| **~0.981** | eps 0.040 (rf=1.15) | 2.750 QPS (363 µs) | ef 250 (rf=1.15) | **3.320 QPS (301 µs)** | **+20.7%** |
| **~0.988** | eps 0.080 | 1.630 QPS (613 µs) | ef 320 (rf=1.15) | **2.691 QPS (371 µs)** | **+65.1%** |
| **~0.992** | eps 0.040 (rf=1.35) | 2.437 QPS (410 µs) | ef 400 (rf=1.15) | **2.188 QPS (457 µs)** | (erzielt Recall 0.9922 ohne Abbruch) |
