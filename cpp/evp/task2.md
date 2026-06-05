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

---

# OptimizationTarget Vergleich — Mode 3 + FLAS (FP32 Build + FP16 Search, FLAS Pre-Sort)

## Konfiguration

- **Dataset:** llama-dev (256921 vectors, 128 dims, k_top=30)
- **Parameter:** k_graph=30, k_ext=60, eps_ext=0.001, build-threads=1, max_dist=20000, eps_search=0.30
- **FLAS:** L2, R=0.93, alle weiteren Parameter default
- **Suchen:** 10 Wiederholungen, Durchschnitt

## Ergebnisse

| OptimizationTarget | Build Time (s) | Recall@30 | Search Time (ms) | Δ Recall vs ohne FLAS |
|---|---|---|---|---|
| **LowLID** | 68.7 | 77.87% | 118.8 | −1.59pp |
| **HighLID** | 56.3 | 55.33% | 79.5 | −20.96pp |
| **StreamingData_SchemeA** | 68.7 | 59.70% | 118.8 | −15.33pp |
| **StreamingData_SchemeB** | 77.1 | 50.79% | 106.9 | −19.62pp |
| **StreamingData_SchemeC** | 79.9 | 79.33% | 128.7 | −0.58pp |
| **StreamingData_SchemeD** | 89.8 | 82.10% | 148.0 | −2.30pp |
| **SchemeA** | 68.5 | 50.11% | 120.0 | −29.35pp |
| **SchemeB** | 32.4 | 33.60% | 79.5 | −27.36pp |


# FLAS Pre-Sort Vergleich — Mode 1 (FP32 Build + FP32 Search)

## Konfiguration

- **Dataset:** llama-dev (256921 vectors, 128 dims, k_top=30)

- **Graphbau:** mode1, k_graph=30, k_ext=30, eps_ext=0.001, eps_search=0.30, max_dist=20000, build-threads=1, search-threads=8, num-runs=1, opt_target=LowLID, prune_worst=0
- **FLAS:** metric/radius_decay pro Zeile (siehe Tabelle), sonst alles default (`do_wrap=false, initial_radius_factor=0.5, num_filters=1, radius_end=1.0, weight_swappable=1.0, weight_non_swappable=100.0, weight_hole=0.01, sample_factor=1.0, max_swap_positions=9, optimize_narrow_grids=1, columns=1, rows=count, seed=42`)

## Ergebnisse

| FLAS Config | FLAS Time (s) | Build Time (s) | Search Time (ms) | Recall@30 | Δ Recall |
|---|---|---|---|---|---|
| **Ohne FLAS** (Baseline) | — | 24.5 | 217.7 | 76.76% | — |
| **FLAS L2, R=0.93 (Default)** | 57.7 | **19.9** | **145.7** | **79.57%** | **+2.81pp** |
| **FLAS L2, R=0.80** | 18.3 | 21.1 | 102.0 | 56.77% | −19.99pp |
| **FLAS L2, R=0.99** | 322.9 | 23.8 | 101.9 | 59.01% | −17.75pp |
| **FLAS IP, R=0.93** | 59.4 | 21.0 | 133.1 | 74.89% | −1.87pp |
| **FLAS IP, R=0.99** | 362.1 | 21.4 | 70.5 | 38.48% | −38.28pp |
| **FLAS L2, R=0.93, num_filters=3** | 73.1 | 16.8 | 145.7 | 59.00% | −17.76pp |
| **FLAS L2, R=0.93, max_swap_positions=18** | 66.7 | 19.7 | 137.0 | 77.89% | −1.68pp |
| **FLAS L2, R=0.93, sample_factor=2** | 81.6 | 20.9 | 99.9 | 53.12% | −23.45pp |
| **FLAS L2, R=0.93, optimize_narrow_grids=0** | 53.6 | 21.2 | 137.7 | 71.37% | −8.20pp |

**Fazit:** Die Default-Einstellungen aus `default_flas_settings()` sind für diesen Workload optimal. Alle getesteten Abweichungen (`num_filters>1`, `max_swap_positions>9`, `sample_factor>1`, `optimize_narrow_grids=0`) verschlechtern das Ergebnis.