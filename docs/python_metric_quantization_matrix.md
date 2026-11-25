# Python High-Level API: Metric, Quantization & Pipeline Matrix

This reference documents the exact internal behavior of `deglib.presort`, `deglib.build`, `deglib.compress`, `deglib.searcher`, `deglib.reranker`, and `deglib.reranker_with_data` for every high-level `metric` and `compress` option, along with the underlying rules and thresholds.

---

## 1. `deglib.presort` & `deglib.build` by High-Level Metric

> **Rule:** `deglib.build(...)` returns a mutable `SizeBoundedGraph` when `compress=None` (or `"32bit"` for `"ip"`), so `save_graph()` and in-place operations work directly, and returns a compressed `ReadOnlyGraph` when `compress` is set. Use `deglib.GraphBuilder` and `deglib.create_dynamic_empty` for dynamically growing graphs.

| `metric` | `deglib.presort(data, metric=...)` | `deglib.build` (Graph Construction) | `deglib.build` Post-Build Finalization (`compress=None`) | Rationale |
|---|---|---|---|---|
| `"l2"` / `"euclidean"` | `FP32_L2` (`float32`) / `FP16_L2` (`float16`) directly on `data` (`dim = d`) | `FP32_L2` / `FP16_L2` directly on `data` (`dim = d`, no transforms) | Mutable `SizeBoundedGraph` (`dim = d`) | True metric; no spatial transformation needed |
| `"cosine"` / `"normalized"` | `FP32_L2` / `FP16_L2` directly on `data` (`dim = d`) | `FP32_InnerProduct` / `FP16_InnerProduct`; attaches `L2NormalizeTransform` if `not is_l2_normalized(data)` (`dim = d`) | Mutable `SizeBoundedGraph` (`dim = d`, keeps `[L2NormalizeTransform]` if attached) | Automatically normalizes DB and query vectors when input is not unit-norm |
| `"ip"` / `"innerproduct"` / `"inner_product"` / `"mips"` | `make_mips_transform(data)` → FLAS in `FP32_L2` (`dim = d + 1`) | `make_mips_transform(data)` → build in `FP32_L2` / `FP16_L2` (`dim = d + 1`) | `.with_features(work_data, metric=FP32_InnerProduct / FP16_InnerProduct, order="internal")` (**`dim = d`**, **no** transforms, mutable `SizeBoundedGraph`) | `MIPS(d+1)` yields stronger graph topology; swapping back to unnormalized `d`-dim `InnerProduct` preserves norms, removes `d+1` from search (`128 B` alignment / `AVX512_VNNI`), and lets `searcher(graph)` / `reranker(graph)` run without external `data` |
| Explicit `Metric` (e.g. `Metric.Int8_L2`, `Metric.EVP_InnerProduct`) | `FP32`/`FP16` L2 or IP matching `metric` (`presort=True` disallowed on pre-quantized inputs) | Quantizes `float32`/`float16` input before `add_entry` via `make_quantizer_pipeline(...)`; or ingests pre-quantized arrays directly | Mutable `SizeBoundedGraph` (or `ReadOnlyGraph` if `compress` is set) | Direct control without high-level spatial resolution |

---

## 2. `deglib.build(..., compress=...)` – Target Quantization

| `metric` | `compress` | Output Metric | Attached Query Transformations | `dim` |
|---|---|---|---|---:|
| `"l2"` | `None` / `"32bit"` | `FP32_L2` (`FP16_L2` for `float16`) | `[]` | `d` |
| `"l2"` | `"16bit"` | `FP16_L2` | `[ScalarQuantizerFP16]` *(or `[]` for `float16` input)* | `d` |
| `"l2"` | `"8bit"` / `"auto"` | `Uint7_L2` | `[(FWHT), ScalarQuantizerUint7]` | `d` |
| `"l2"` | `"4bit"` | `Uint4_L2` | `[(FWHT), ScalarQuantizerUint4]` | `d` |
| `"l2"` | `"2bit"` | `EVP_InnerProduct` | `[EvpQuantizer]` | `d` |
| `"cosine"` | `None` / `"32bit"` | `FP32_InnerProduct` (`FP16_InnerProduct` for `float16`) | `[L2NormalizeTransform]` *(if input unnormalized)* | `d` |
| `"cosine"` | `"16bit"` | `FP16_InnerProduct` | `[L2Norm?, ScalarQuantizerFP16]` | `d` |
| `"cosine"` | `"8bit"` / `"auto"` | `predict_quantization(...)` (`Int8_InnerProduct` vs. `Uint7_L2`) | `[L2Norm?, (FWHT), ScalarQuantizer*]` | `d` |
| `"cosine"` | `"4bit"` | `Int4_InnerProduct` | `[L2NormalizeTransform?, (FWHT), ScalarQuantizerInt4]` | `d` |
| `"cosine"` | `"2bit"` | `EVP_InnerProduct` | `[L2Norm?, EvpQuantizer]` | `d` |
| **`"ip"`** | `None` / `"32bit"` | `FP32_InnerProduct` (`FP16_InnerProduct` for `float16`) | `[]` *(unnormalized)* | `d` |
| **`"ip"`** | `"16bit"` | `FP16_InnerProduct` | `[ScalarQuantizerFP16]` *(or `[]` for `float16` input)* | `d` |
| **`"ip"`** | `"8bit"` / `"auto"` | `Int8_InnerProduct` | `[FastHadamardRotation, ScalarQuantizerInt8]` *(unnormalized, `AVX512_VNNI`)* | `d` |
| **`"ip"`** | `"4bit"` | `Int4_InnerProduct` | `[L2NormalizeTransform, FastHadamardRotation, ScalarQuantizerInt4]` | `d` |
| **`"ip"`** | `"2bit"` | `EVP_InnerProduct` | `[MipsTransform, EvpQuantizer]` | `d + 1` |
| **`"ip"`** | Explicit `L2` metric (e.g. `Metric.Uint7_L2`, `Metric.Int8_L2`) | Target `L2` metric | `[MipsTransform, (FWHT), ScalarQuantizer*]` | `d + 1` |

---

## 3. `deglib.compress(graph, target)` & `deglib.searcher(graph, compress=...)`

When `compress` is provided, `deglib.searcher(graph, compress=...)` calls `deglib.compress(graph, compress)` and then `search_graph.to_searcher(reranker=reranker, optimize=optimize, threads=threads)`.

### 3.1 Source Graph is `L2` (`FP32_L2` / `FP16_L2`)

| `target` / `compress` | Target Metric | Transformations | Notes |
|---|---|---|---|
| `None` / `"32bit"` | `FP32_L2` | existing | Returns `graph` unchanged if already `FP32_L2` (raises `ValueError` if `graph` is `FP16_L2`) |
| `"16bit"` | `FP16_L2` | `[ScalarQuantizerFP16]` | Near-lossless, half the feature RAM |
| `"8bit"` / `"auto"` | `Uint7_L2` | `[(FWHT), ScalarQuantizerUint7]` | `FWHT` auto-selected by kurtosis |
| `"4bit"` | `Uint4_L2` | `[(FWHT), ScalarQuantizerUint4]` | Recommended with `FP16_L2` reranker |
| `"2bit"` | `EVP_InnerProduct` | `[EvpQuantizer]` | Intended for unit-norm / rerank pipelines |

### 3.2 Source Graph is `InnerProduct` (`FP32_InnerProduct` / `FP16_InnerProduct`)

| `target` / `compress` | Target Metric | Transformations | Notes |
|---|---|---|---|
| `None` / `"32bit"` | `FP32_InnerProduct` | existing | Returns `graph` unchanged if already `FP32_InnerProduct` (raises `ValueError` if `graph` is `FP16_InnerProduct`) |
| `"16bit"` | `FP16_InnerProduct` | `[ScalarQuantizerFP16]` | Native MIPS without reranker |
| `"8bit"` / `"auto"` | `Int8_InnerProduct` | `[(FWHT), ScalarQuantizerInt8]` | `AVX512_VNNI`; for `≥ 80%` Recall@30 on MIPS pair with `FP16_IP` reranker (`rf ≈ 2.0`) + `optimize(n_clusters=256, use_routing=True)` |
| `"4bit"` | `Int4_InnerProduct` | `[L2NormalizeTransform, (FWHT), ScalarQuantizerInt4]` *(if no `L2Norm` prefix exists yet)* | Calibrated via `normalize=True` on L2-normalized vectors; lossy on true MIPS with wide norm spread |
| `"2bit"` | `EVP_InnerProduct` | `[EvpQuantizer]` | High-compression 2-bit filtering + reranking |
| Explicit `L2` `Metric` (`Metric.Uint7_L2`, `Metric.Int8_L2`, `Metric.Uint4_L2`, `Metric.Uint8_L2`) | Target `L2` metric | `[MipsTransform?, (FWHT), ScalarQuantizer*]` *(auto-prepends `MipsTransform` via `mips=True` only when stored graph vectors are not unit-norm and have no prefix)* | Enables `L2` quantizers directly on `InnerProduct` graphs without adding `d+1` overhead on unit-norm `cosine`/`normalized` graphs |

---

## 4. `deglib.reranker` & `deglib.reranker_with_data`

| Function | Precondition | Behavior |
|---|---|---|
| `deglib.reranker(graph, compress="16bit")` | `graph` is uncompressed (`FP32_*` or `FP16_*`) | Extracts base vectors directly from `graph` (in internal `graph.get_labels()` order). Distance kind (`L2` vs. `InnerProduct`) is inferred from `graph.get_metric()`. |
| `deglib.reranker_with_data(graph, data, labels=..., compress="16bit", metric=...)` | Original `float32`/`float16` `data` available (works even when `graph` is already quantized to `8bit`/`4bit`/`2bit`) | Maps `graph.get_labels()` to rows in `data` via `labels` without copying `data`. Preserves any `L2NormalizeTransform` / `MipsTransform` prefix on `graph`. `metric="ip"` selects `InnerProduct` unless a `MipsTransform` prefix is present. |

---

## 5. Automatic `FastHadamardRotation` (`FWHT`) & Calibration

In `cpp/deglib/include/deglib/transformation.h` (`make_quantizer_pipeline` / `default_rotation_kurtosis_threshold`):

- **Quantile calibration:** Default is always `quantile = -1.0` (fast min/max calibration, no slow quantile search).
- **Kurtosis threshold for automatic `FWHT`:**

| `MetricDataType` | Threshold (`use_rerank=False`) | Threshold (`use_rerank=True`) |
|---|---:|---:|
| `Uint4`, `Int4` | `0.8` | `0.8` |
| `Uint7` | `0.8` | `5.0` |
| `Uint8` | `1.85` | `5.0` |
| `Int8` | `1.85` | `25.0` |
| `FP16`, `EVP` | disabled (`-1.0`) | disabled (`-1.0`) |

*(Note: `deglib.build(..., metric="ip", compress="8bit"|"4bit")` always applies `make_fast_hadamard_rotation` explicitly.)*

---

## 6. Operational Rules

1. **Why not store `L2Norm(d)` features on `IP` graphs?**  
   `L2Norm(d)` discards vector norms (`||x|| = 1`). On true MIPS datasets (e.g. `yi`, `llama`, `llama-dev` with norms `~15` to `~33`), that turns MIPS into Cosine and prevents `deglib.reranker(graph)` from recovering true inner-product scores without external original data.
2. **`save_graph()` on `ReadOnlyGraph`:**  
   `.deg` files store edge weights and require a mutable graph. `DynamicExplorationGraph.save_graph()` automatically converts a `ReadOnlyGraph` via `self.to_mutable().save_graph(path)`.
3. **`.deg` serialization:**  
   All `FeatureTransformation` tags (`TAG_MIPS`, `TAG_NORM`, `TAG_FWHT`, `TAG_SQ16`, `TAG_SQI8`, `TAG_SQU8`, `TAG_SQU7`, `TAG_SQI4`, `TAG_SQU4`, `TAG_EVPQ`) are serialized and deserialized across save/load roundtrips.
4. **Best MIPS search configuration (`k=30`, `llama-dev`):**  
   `Searcher(Int8_InnerProduct, FWHT)` + `FP16_InnerProduct` reranker with `searcher.optimize(n_clusters=256, use_routing=True)` and `rerank_factor=2.0`.
