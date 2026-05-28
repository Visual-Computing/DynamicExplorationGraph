# Benchmark-Vergleich: Batch-Berechnung vs. Sequentiell + Prefetching

Dieses Dokument vergleicht die Leistungsergebnisse der EVP Benchmark-Task 1 unter Verwendung von zwei verschiedenen Implementierungsstrategien in der Dynamic Exploration Graph (DEG) C++ Bibliothek:

1. **Batch-Berechnung (`compare_batch`)**: Die Distanzberechnungen für alle qualifizierenden Nachbarn werden gebündelt und in einem optimierten SIMD-Block berechnet.
2. **Sequentiell + Prefetching**: Die Nachbardaten werden sequentiell geladen und vorab in den Cache geholt (`prefetch`), während die Distanzen nacheinander mit dem regulären `COMPARATOR` ermittelt werden.

---

## 📊 1. Benchmark-Tabelle: Batch-Berechnung (`compare_batch`)

*Dieser Durchlauf nutzt die parallele SIMD-Batch-Vektorisierung für Nachbarschafts-Scans.*

| Pos | Methode | Settings | Load Time | Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | **Overall Time** | Recall | Ideal RAM |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **deglib FP16 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0.5 s | 0.0 s | 17.5 s | 0.0 s | 1.1 s | 0.0 s | **19.1 s** | 0.8289 | 460MB |
| 2 | **evp linear search (cpp)** | — | 0.6 s | 0.8 s | 0.0 s | 0.0 s | 209.3 s | 0.0 s | **211.0 s** | 0.7084 | 102MB |
| 3 | **deglib+evp Build&Explore (cpp)** | `M=32`, `MaxDist=200` | 0.5 s | 0.7 s | 4.9 s | 0.0 s | 0.8 s | 0.0 s | **6.9 s** | 0.6701 | 102MB |
| 4 | **deglib+evp Build&Explore+FP16 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.4 s | 0.7 s | 4.9 s | 0.0 s | 1.6 s | 4.1 s | **11.7 s** | 0.8206 | 512MB |
| 5 | **deglib+evp build+FP16 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.3 s | 0.7 s | 5.3 s | 0.1 s | 4.2 s | 0.0 s | **10.6 s** | 0.8248 | 512MB |
| 6 | **deglib+evp build+Asym FP16&EVP Explore (cpp)** | `M=32`, `MaxDist=200` | 0.4 s | 0.8 s | 5.0 s | 0.0 s | 1.4 s | 0.0 s | **7.6 s** | 0.7252 | 512MB |
| 7 | **deglib+evp build+Asym FP16&EVP Explore+FP Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=50` | 0.4 s | 0.8 s | 5.8 s | 0.0 s | 1.5 s | 1.1 s | **8.6 s** | 0.8247 | 512MB |

---

## 📊 2. Benchmark-Tabelle: Sequentiell + Prefetching

*Dieser Durchlauf nutzt die sequentielle Distanzberechnung mit optimiertem Prefetching der Feature-Vektoren.*

| Pos | Methode | Settings | Load Time | Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | **Overall Time** | Recall | Ideal RAM |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **deglib FP16 Build&Explore (cpp)** | `M=32`, `MaxDist=100` | 0.5 s | 0.0 s | 18.5 s | 0.0 s | 1.2 s | 0.0 s | **20.2 s** | 0.8287 | 460MB |
| 2 | **evp linear search (cpp)** | — | 0.6 s | 0.8 s | 0.0 s | 0.0 s | 209.3 s | 0.0 s | **211.0 s** | 0.7084 | 102MB |
| 3 | **deglib+evp Build&Explore (cpp)** | `M=32`, `MaxDist=200` | 0.4 s | 0.8 s | 4.9 s | 0.0 s | 0.9 s | 0.0 s | **7.0 s** | 0.6699 | 102MB |
| 4 | **deglib+evp Build&Explore+FP16 Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=200` | 0.4 s | 0.8 s | 4.8 s | 0.0 s | 1.3 s | 4.1 s | **11.4 s** | 0.8206 | 512MB |
| 5 | **deglib+evp build+FP16 Explore (cpp)** | `M=32`, `MaxDist=200` | 0.3 s | 0.8 s | 4.7 s | 0.1 s | 4.4 s | 0.0 s | **10.3 s** | 0.8250 | 512MB |
| 6 | **deglib+evp build+Asym FP16&EVP Explore (cpp)** | `M=32`, `MaxDist=200` | 0.4 s | 0.8 s | 5.0 s | 0.1 s | 1.3 s | 0.0 s | **7.6 s** | 0.7253 | 512MB |
| 7 | **deglib+evp build+Asym FP16&EVP Explore+FP Rerank (cpp)** | `M=32`, `MaxDist=200`, `evpK=50` | 0.4 s | 0.8 s | 5.1 s | 0.0 s | 1.5 s | 1.1 s | **9.0 s** | 0.8250 | 512MB |

---

## 📈 3. Direkter Vergleich

### Vergleich der Gesamtzeiten (**Overall Time**)

| Pos | Methode | Batch-Berechnung | Sequentiell + Prefetch | Differenz |
| :--- | :--- | :---: | :---: | :---: |
| 1 | **deglib FP16 Build&Explore** | 19.1 s | 20.2 s | **+1.1 s** (langsamer) |
| 3 | **deglib+evp Build&Explore** | 6.9 s | 7.0 s | **+0.1 s** (nahezu identisch) |
| 4 | **deglib+evp Build&Explore+FP16 Rerank** | 11.7 s | 11.4 s | **-0.3 s** (schneller) |
| 5 | **deglib+evp build+FP16 Explore** | 10.6 s | 10.3 s | **-0.3 s** (schneller) |
| 6 | **deglib+evp build+Asym FP16&EVP Explore** | 7.6 s | 7.6 s | **0.0 s** (identisch) |
| 7 | **deglib+evp build+Asym FP16&EVP Explore+FP Rerank** | 8.6 s | 9.0 s | **+0.4 s** (etwas langsamer) |

### Vergleich der reinen Suchzeiten (**Explore Time**)

Da die Build- und HDF5-Ladezeiten systembedingten Schwankungen unterliegen, ist der Vergleich der **reinen Suchzeit** (Explore Time) am aussagekräftigsten für die Algorithmenleistung:

| Pos | Methode | Explore Time (Batch) | Explore Time (Seq + Prefetch) | Differenz |
| :--- | :--- | :---: | :---: | :---: |
| 1 | **FP16 Explore** | 1.1 s (1097.7 ms) | 1.2 s (1184.1 ms) | **+86.4 ms** (Batch schneller) |
| 3 | **EVP Explore** | 0.8 s (820.6 ms) | 0.9 s (870.9 ms) | **+50.3 ms** (Batch schneller) |
| 4 | **EVP Explore (vor Rerank)** | 1.6 s (1605.8 ms) | 1.3 s (1333.4 ms) | **-272.4 ms** (Seq+Prefetch schneller! 🚀) |
| 5 | **FP16 External Explore** | 4.2 s (4222.9 ms) | 4.4 s (4353.1 ms) | **+130.2 ms** (Batch schneller) |
| 6 | **Asym FP16&EVP Explore** | 1.4 s (1401.2 ms) | 1.3 s (1310.2 ms) | **-91.0 ms** (Seq+Prefetch schneller! 🚀) |
| 7 | **Asym Explore (vor Rerank)** | 1.5 s (1518.7 ms) | 1.5 s (1500.4 ms) | **-18.3 ms** (nahezu identisch) |

---

## 💡 Fazit & Erkenntnisse

> [!TIP]
> **Für komplexe Topologien (wie EVP-Quantisierung mit Reranking):**
> Der sequentielle Ansatz kombiniert mit aggressivem Prefetching der Nachbar-Feature-Vektoren (`Seq + Prefetch`) übertrifft die Batch-Distanzberechnung in Szenarien mit komplexen Pfaden (wie in **Pos 4** und **Pos 6**). Hier profitiert der Algorithmus stark von der geringen Latenz durch präventives Caching, was die reinen Suchzeiten um bis zu **~270 ms** senkt.

> [!NOTE]
> **Für lineare/homogene Vektoren (wie FP16):**
> Die Batch-Distanzberechnung (`compare_batch`) behält bei homogenen und breiten Vektoren (z. B. reinem FP16-Routing) einen minimalen Performance-Vorteil von ca. **50 ms bis 130 ms**, da der Compiler die SIMD-Register hier optimaler auslasten kann, wenn Distanzen in zusammenhängenden Blöcken berechnet werden.
