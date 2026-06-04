# mode1 (FP32 build + FP32 search) — Results for 80% recall

## Konfiguration

| Parameter | Wert |
|---|---|
| Mode | mode1 (FP32 build + FP32 search) |
| Metric | ip |
| Dataset | llama-dev (256921 vectors, 128 dims, k_top=30) |
| threads | 6 |
| search | 10x Wiederholung, Durchschnitt in ms |

## Getestete Kombinationen

### k_graph=30, k_ext=60, eps_ext=0.001

**Graph-Datei:** `C:\Data\ANN\sisap2026\llama-dev\llama-dev_k30_kext60_eps0.001_ip.deg`

| eps_search | max_dist | recall | search time (ms) |
|---|---|---|---|
| 0.1 | 10000 | 15.17% | 13.4 |
| 0.1 | 30000 | 15.17% | 11.8 |
| 0.2 | 10000 | 35.32% | 37.3 |
| 0.2 | 30000 | 35.43% | 38.6 |
| 0.3 | 10000 | 54.52% | 91.8 |
| 0.3 | 30000 | 59.37% | 152.6 |
| 0.5 | 10000 | 64.32% | 142.7 |
| 0.5 | 30000 | 79.62% | 431.5 |

**Schnellste Kombination über 80%:** Nicht erreicht (max 79.62%)

### k_graph=30, k_ext=60, eps_ext=0.01

**Graph-Datei:** `C:\Data\ANN\sisap2026\llama-dev\llama-dev_k30_kext60_eps0.01_ip.deg`

| eps_search | max_dist | recall | search time (ms) |
|---|---|---|---|
| 0.3 | 20000 | 59.11% | 150.6 |
| 0.3 | 25000 | 59.64% | 157.8 |
| 0.3 | 30000 | 59.90% | 164.3 |
| 0.5 | 20000 | 74.22% | 299.2 |
| 0.5 | 25000 | 76.74% | 369.8 |
| 0.5 | 30000 | 78.44% | 446.3 |

**Schnellste Kombination über 80%:** Nicht erreicht (max 78.44%)

### k_graph=30, k_ext=60, eps_ext=0.05

**Graph-Datei:** `C:\Data\ANN\sisap2026\llama-dev\llama-dev_k30_kext60_eps0.05_ip.deg`

| eps_search | max_dist | recall | search time (ms) |
|---|---|---|---|
| 0.3 | 20000 | 60.26% | 135.9 |
| 0.3 | 30000 | 60.99% | 157.8 |
| 0.5 | 20000 | 74.99% | 299.7 |
| 0.5 | 30000 | 79.06% | 436.7 |

**Schnellste Kombination über 80%:** Nicht erreicht (max 79.06%)

### k_graph=48, k_ext=96, eps_ext=0.001

**Graph-Datei:** `C:\Data\ANN\sisap2026\llama-dev\llama-dev_k48_kext96_eps0.001_ip.deg`

| eps_search | max_dist | recall | search time (ms) |
|---|---|---|---|
| 0.3 | 20000 | 80.96% | 220.8 |
| 0.3 | 25000 | 83.29% | 256.5 |
| 0.5 | 20000 | 84.85% | 272.0 |
| 0.5 | 25000 | 88.09% | 339.6 |

**Schnellste Kombination über 80%:** eps_search=0.3, max_dist=20000 → 80.96% in 220.8ms

**Bester bisheriger Gesamt-Sieger:** eps_search=0.3, max_dist=20000, eps_ext=0.001, k_graph=48 → 80.96% in 220.8ms

---

# FP16 Exploration (Modes 2 & 3) vs FP32 Baseline (Mode 1)

## Konfiguration

- **Dataset:** llama-dev (256921 vectors, 128 dims, k_top=30)
- **Parameter:** k_graph=30, k_ext=30, eps_ext=0.001, max_dist=20000, eps_search=0.30

## Ergebnisse

| Mode | Beschreibung | Build Threads | Search Threads | Recall@30 | Search Time (ms) |
|---|---|---|---|---|---|
| **Mode 1** | FP32 Build + FP32 Search | 6 | 6 | ~76.58% | 200.8 |
| **Mode 2** | FP16 Build + FP16 Search | 6 | 6 | ~72.24% | 165.2 |
| **Mode 3** | FP32 Build + FP16 Search | 6 | 6 | ~69.50% | 146.6 |
| **Mode 1** | FP32 Build + FP32 Search | 1 (Deterministisch) | 6 | 76.76% | 190.5 |
| **Mode 2** | FP16 Build + FP16 Search | 1 (Deterministisch) | 6 | 76.41% | 160.7 |
| **Mode 3** | FP32 Build + FP16 Search | 1 (Deterministisch) | 6 | 76.75% | 154.9 |

---

# OptimizationTarget Vergleich — Mode 1 (FP32 Build + FP32 Search)

## Konfiguration

- **Dataset:** llama-dev (256921 vectors, 128 dims, k_top=30)
- **Parameter:** k_graph=30, k_ext=30, eps_ext=0.001, build-threads=1, max_dist=20000, eps_search=0.30
- **Suchen:** 10 Wiederholungen, Durchschnitt

## Ergebnisse

| OptimizationTarget | Build Time (s) | Recall@30 | Search Time (ms) |
|---|---|---|---|
| **LowLID** | 18.5 | 76.76% | 147.8 |
| **HighLID** | 15.0 | 72.86% | 144.1 |
| **StreamingData_SchemeA** | 70.1 | 69.91% | 150.4 |
| **StreamingData_SchemeB** | 73.1 | 64.47% | 149.2 |
| **StreamingData_SchemeC** | 63.5 | 77.07% | 158.0 |
| **StreamingData_SchemeD** | 71.3 | 80.82% | 157.5 |
| **SchemeA** | 20.4 | 72.17% | 194.6 |
| **SchemeB** | 16.7 | 54.18% | 145.1 |

---

# OptimizationTarget Vergleich — Mode 1 (k_ext=60, eps_ext=0.001)

## Konfiguration

- **Dataset:** llama-dev (256921 vectors, 128 dims, k_top=30)
- **Parameter:** k_graph=30, k_ext=60, eps_ext=0.001, build-threads=1, max_dist=20000, eps_search=0.30
- **Suchen:** 10 Wiederholungen, Durchschnitt

## Ergebnisse

| OptimizationTarget | Build Time (s) | Recall@30 | Search Time (ms) |
|---|---|---|---|
| **LowLID** | 46.9 | 79.46% | 147.4 |
| **HighLID** | 36.1 | 76.29% | 148.5 |
| **StreamingData_SchemeA** | 160.9 | 75.03% | 145.0 |
| **StreamingData_SchemeB** | 175.5 | 70.41% | 147.2 |
| **StreamingData_SchemeC** | 155.3 | 79.91% | 152.3 |
| **StreamingData_SchemeD** | 165.0 | **84.40%** | 152.5 |
| **SchemeA** | 47.1 | 79.46% | 147.5 |
| **SchemeB** | 41.1 | 60.96% | 148.4 |

---

# OptimizationTarget Vergleich — Mode 2 (k_ext=60, eps_ext=0.001, FP16 Build + FP16 Search)

## Konfiguration

- **Dataset:** llama-dev (256921 vectors, 128 dims, k_top=30)
- **Parameter:** k_graph=30, k_ext=60, eps_ext=0.001, build-threads=1, max_dist=20000, eps_search=0.30
- **Suchen:** 10 Wiederholungen, Durchschnitt

## Ergebnisse

| OptimizationTarget | Build Time (s) | Recall@30 | Search Time (ms) |
|---|---|---|---|
| **LowLID** | 47.0 | 78.11% | 128.9 |
| **HighLID** | 36.2 | 74.75% | 129.0 |
| **StreamingData_SchemeA** | 161.1 | 73.57% | 128.9 |
| **StreamingData_SchemeB** | 176.5 | 68.07% | 128.3 |
| **StreamingData_SchemeC** | 155.3 | 78.56% | 130.6 |
| **StreamingData_SchemeD** | 166.1 | **82.84%** | 131.2 |
| **SchemeA** | 47.2 | 78.11% | 129.0 |
| **SchemeB** | 41.2 | 59.88% | 128.5 |

---

# OptimizationTarget Vergleich — Mode 3 (k_ext=60, eps_ext=0.001, FP32 Build + FP16 Search)

## Konfiguration

- **Dataset:** llama-dev (256921 vectors, 128 dims, k_top=30)
- **Parameter:** k_graph=30, k_ext=60, eps_ext=0.001, build-threads=1, max_dist=20000, eps_search=0.30
- **Suchen:** 10 Wiederholungen, Durchschnitt

## Ergebnisse

| OptimizationTarget | Build Time (s) | Recall@30 | Search Time (ms) |
|---|---|---|---|
| **LowLID** | 47.0 | 79.46% | 129.0 |
| **HighLID** | 36.1 | 76.29% | 128.5 |
| **StreamingData_SchemeA** | 160.8 | 75.03% | 128.6 |
| **StreamingData_SchemeB** | 175.3 | 70.41% | 128.8 |
| **StreamingData_SchemeC** | 154.8 | 79.91% | 130.6 |
| **StreamingData_SchemeD** | 165.1 | **84.40%** | 130.5 |
| **SchemeA** | 47.1 | 79.46% | 129.6 |
| **SchemeB** | 41.6 | 60.96% | 129.5 |
