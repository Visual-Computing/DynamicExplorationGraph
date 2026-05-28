# Benchmark Results — Large Dataset (wikipedia-bge-m3, 6.35M x 1024 fp16)

Dataset: `C:\Data\ANN\sisap2026\large\benchmark-dev-wikipedia-bge-m3.h5`
Parameters: `k_graph=32`, `k_ext=32`, `eps_ext=0.001`, `k_top=15`, `threads=6`

| Modus | Methode | Quant Time | Build Time | Convert Time | Explore Time | Rerank Time | Total Time | Recall@15 |
|:---:|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | EVP build+EVP explore | 22.76 s | 278.13 s | — | 46.77 s | — | 347.67 s | 0.6278 |
| 4 | EVP build+EVP explore+FP16 rerank (evpK=200) | 23.27 s | 318.91 s | — | 617.22 s | 121.79 s | 1081.19 s | 0.7343 |
| 5 | EVP build+FP16 external search | 22.79 s | 266.86 s | 3.92 s | 647.31 s | — | 940.87 s | 0.7391 |
| 6 | EVP build+asym FP16&EVP explore | 22.72 s | 282.88 s | 2.15 s | 54.52 s | — | 362.26 s | 0.6695 |
| 7 | EVP build+asym explore+FP16 rerank (evpK=50) | 22.75 s | 284.85 s | 2.43 s | 58.71 s | 30.00 s | 398.74 s | 0.7382 |

**Notes:**
- Total Time includes Load time (~21.3 s), Quantize, Build, Convert, Explore, and Rerank phases.
- Modes 3, 5, 6 have no rerank phase.
- Best recall: **Mode 5** (0.7391) and **Mode 7** (0.7382).
- Fastest explore: **Mode 3** (46.77 s) and **Mode 6** (54.52 s).
- Mode 4 (evpK=200) has the longest explore phase due to large candidate list.
