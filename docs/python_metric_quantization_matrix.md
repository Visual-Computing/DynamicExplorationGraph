# Python High-Level API: Metric, Quantization & Pipeline Matrix

This reference documents the exact internal behavior of the high-level Python API (`deglib.presort`, `deglib.build`, `deglib.compress`, `deglib.searcher`, `deglib.reranker`, `deglib.reranker_with_data`, `deglib.load_graph`) for every `metric` and `compress` option, along with the exact source locations and edge-case rules.

**Implementation map:**
- Spatial metric resolution & prefix transforms (`is_l2_normalized`, `resolve_spatial_metric`): `python/src/deglib/transformation.py`
- 1D FLAS pre-sorting (`presort`): `python/src/deglib/optimization.py`
- High-level build & post-build finalization (`build`): `python/src/deglib/builder.py`
- Compression & metric conversion (`compress`, `_prepare_metric_transform`, `load_readonly_graph`): `python/src/deglib/graph.py`
- Reranker & searcher wrappers (`reranker`, `reranker_with_data`, `searcher`): `python/src/deglib/search.py`
- Graph-based quantizer calibration (`make_quantizer_pipeline_from_graph`, `fit_mips_transform_from_graph`): `python/src/deg_cpp/bindings_transformation.cpp`
- Quantizer pipeline & FWHT kurtosis thresholds (`make_quantizer_pipeline`): `cpp/deglib/include/deglib/transformation.h`

---

## 1. `deglib.presort` & `deglib.build` by High-Level Metric

> **Mutability rule:** `deglib.compress(graph, target)` always preserves the backend type of `graph` (`SizeBoundedGraph` stays `SizeBoundedGraph`, `DynamicGraph` stays `DynamicGraph`, `ReadOnlyGraph` stays `ReadOnlyGraph`). `deglib.build(...)` always returns a mutable `SizeBoundedGraph` regardless of `metric` or `compress`. For dynamically growing streaming graphs, use `deglib.builder.GraphBuilder` with `deglib.graph.create_dynamic_empty`.

| `metric` (aliases) | `deglib.presort(data, metric=...)` | `deglib.build` (Graph Construction) | `deglib.build` Post-Build Finalization (`compress=None`) | Rationale |
|---|---|---|---|---|
| `"l2"` / `"euclidean"` | `FP32_L2` (`float32`) / `FP16_L2` (`float16`) directly on `data` (`dim = d`) | Quantizes `data` before `add_entry` when `compress` is set, and builds directly in the target space (`dim = d`) | Mutable `SizeBoundedGraph` (`dim = d`, keeps `[(FWHT), Quantizer]` if `compress` is set) | True metric; building directly in compressed form saves build RAM |
| `"cosine"` / `"normalized"` | `FP32_L2` / `FP16_L2` directly on `data` (`dim = d`) | Attaches `L2NormalizeTransform` if `not is_l2_normalized(data)` (`dim = d`); `presort=None`/`True` sorts on normalized vectors; if `compress` is set, quantizes normalized vectors before `add_entry` and builds directly in target space | Mutable `SizeBoundedGraph` (`dim = d`, keeps `[L2NormalizeTransform?, (FWHT), Quantizer]`) | Automatically normalizes DB and query vectors and preserves `L2NormalizeTransform` for `deglib.compress(..., "8bit")` auto-prediction |
| `"ip"` / `"innerproduct"` / `"inner_product"` / `"mips"` | `make_mips_transform(data)` → FLAS in `FP32_L2` (`dim = d + 1`), or `FP32_InnerProduct` / `FP16_InnerProduct` (`dim = d`) if `is_l2_normalized(data)` | If `is_l2_normalized(data)`: behaves like `"normalized"` (`FP32_InnerProduct` / `FP16_InnerProduct`, no `MipsTransform`). Otherwise: `make_mips_transform(data)` → quantizes to target bit-width `L2` (`FP32_L2`, `FP16_L2`, `Uint7_L2` for `"8bit"`/`"auto"`, `Uint4_L2` for `"4bit"`/`"2bit"`) before `add_entry` and builds in `MIPS(d + 1) + L2` | `.with_features(work_data, metric=..., order="internal")` (**`dim = d`**, mutable `SizeBoundedGraph`; or keeps `L2 (d+1)` if `compress` is an explicit `L2` `Metric`) | `MIPS(d+1)` yields stronger graph topology for unnormalized IP vectors; when inputs are already unit-norm (`is_l2_normalized(data)`), auto-switching to `"normalized"` avoids `MIPS(d+1)` overhead and enables `predict_quantization_normalized` |
| Explicit `Metric` (e.g. `Metric.Int8_L2`, `Metric.EVP_InnerProduct`) | `FP32`/`FP16` L2 or IP matching `metric` (`presort=True` raises `ValueError` on pre-quantized inputs) | Quantizes `float32`/`float16` input before `add_entry` via `make_quantizer_pipeline(...)`; or ingests pre-quantized arrays directly | Mutable `SizeBoundedGraph` (or `ReadOnlyGraph` if `compress` is set) | Direct control without high-level spatial resolution |

### Unit-Norm Detection Tolerances
- **`is_l2_normalized(data)`** (`transformation.py` for `"cosine"` / `"normalized"`): checks `| ||x||_2 - 1 | <= tol` with `tol = 1e-2` (`float16`) and `tol = 1e-3` (`float32`). Empty arrays (`n == 0`) are treated as normalized.
- **`fit_mips_transform_from_graph(internal, ...)`** (`bindings_transformation.cpp` for `InnerProduct -> L2` conversion): computes `||x||^2 = 1 - ip_dist(x, x)` via `FloatSpace(FP16/FP32_InnerProduct)` across all `n` vectors and skips `MipsTransform` when `|min_norm_sq - 1| <= tol` and `|max_norm_sq - 1| <= tol` (`tol = 1e-2` for `FP16`, `1e-3` for `FP32`).

---

## 2. `deglib.build(..., compress=...)` – Direct Compressed Build

`deglib.build` quantizes **before** `add_entry` so the graph is constructed directly in the compressed format (for unnormalized `metric="ip"`, it builds in `MIPS(d + 1) + L2` at the corresponding bit-width before swapping back to `d`-dim `InnerProduct`; if `is_l2_normalized(data)` is true, `metric="ip"` automatically behaves like `"normalized"`). For post-build compression of an `FP32`/`FP16` graph, use `deglib.compress(graph, target)` or `deglib.searcher(graph, compress=target)`.

| `metric` | `compress` | Output Metric | Attached Query Transformations | `dim` | Backend |
|---|---|---|---|---:|---|
| `"l2"` | `None` / `"32bit"` | `FP32_L2` (`FP16_L2` for `float16`) | `[]` | `d` | `SizeBoundedGraph` |
| `"l2"` | `"16bit"` | `FP16_L2` | `[ScalarQuantizerFP16]` *(or `[]` if input is `float16`)* | `d` | `SizeBoundedGraph` |
| `"l2"` | `"8bit"` / `"auto"` | `Uint7_L2` | `[(FWHT), ScalarQuantizerUint7]` | `d` | `SizeBoundedGraph` |
| `"l2"` | `"4bit"` | `Uint4_L2` | `[(FWHT), ScalarQuantizerUint4]` | `d` | `SizeBoundedGraph` |
| `"l2"` | `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[EvpQuantizer]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `None` / `"32bit"` | `FP32_InnerProduct` (`FP16_InnerProduct` for `float16`) | `[L2NormalizeTransform]` *(if input unnormalized)* | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"16bit"` | `FP16_InnerProduct` | `[L2Norm?, ScalarQuantizerFP16]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"8bit"` / `"auto"` | `predict_quantization(data)` (**K5-Top**: `Int8_InnerProduct` vs. `Uint7_L2`) | `[L2Norm?, (FWHT), ScalarQuantizer*]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"4bit"` | `Uint4_L2` | `[L2NormalizeTransform?, (FWHT), ScalarQuantizerUint4]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[L2Norm?, EvpQuantizer]` | `d` | `SizeBoundedGraph` |
| **`"ip"`** | `None` / `"32bit"` | `FP32_InnerProduct` (`FP16_InnerProduct` for `float16`) | `[]` *(unnormalized)* | `d` | `SizeBoundedGraph` |
| **`"ip"`** | `"16bit"` | `FP16_InnerProduct` | `[]` *(unnormalized)* | `d` | `SizeBoundedGraph` |
| **`"ip"`** | `"8bit"` / `"auto"` | `Int8_InnerProduct` (`AVX512_VNNI`) | `[FastHadamardRotation, ScalarQuantizerInt8]` *(unnormalized)* | `d` | `SizeBoundedGraph` |
| **`"ip"`** | `"4bit"` | `Int4_InnerProduct` | `[(FWHT), ScalarQuantizerInt4]` *(unnormalized)* | `d` | `SizeBoundedGraph` |
| **`"ip"`** | `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[EvpQuantizer]` *(unnormalized)* | `d` | `SizeBoundedGraph` |
| **`"ip"`** | Explicit `L2` metric (e.g. `Metric.Uint7_L2`, `Metric.Int8_L2`) | Target `L2` metric | `[MipsTransform, (FWHT), ScalarQuantizer*]` | `d + 1` | `SizeBoundedGraph` |

### Empty & Pre-Quantized Edge Cases in `deglib.build`
- **`n_samples == 0`**: returns an empty `SizeBoundedGraph` (`dim = d`, `metric = FP32_*` or `FP16_*` for `float16`); `compress` is skipped (calling `deglib.compress` directly on an empty graph raises `ValueError`).
- **Pre-quantized input** (e.g. `int8`, `uint8`): `compress` must be `None`, `presort=None` (default) skips FLAS pre-sorting, `presort=True` raises `ValueError` (precomputed `sorted_indices` array is still allowed).

---

## 3. `deglib.compress(graph, target)` & `deglib.searcher(graph, compress=...)`

When `compress` is provided, `deglib.searcher(graph, compress=...)` calls `deglib.compress(graph, compress)` and then `search_graph.to_searcher(reranker=reranker, optimize=optimize, threads=threads)`.
- **ISA rule:** `compress` does **not** pin the base `FP32` graph's resolved ISA (e.g. `AVX512`) onto `target_metric`, so `Int8_InnerProduct` / `Uint8_InnerProduct` automatically select `AVX512_VNNI` / `AVX2_VNNI` when supported.
- **Invalid inputs:** calling `compress` (or `to_searcher(metric=...)` with a different dtype) on an already-quantized graph raises `ValueError`; `"32bit"` on an `FP16_*` graph raises `ValueError`; compressing an empty graph raises `ValueError`.

### 3.1 Source Graph is `L2` (`FP32_L2` / `FP16_L2`)

| `target` / `compress` | Target Metric | Transformations | Notes |
|---|---|---|---|
| `None` / `"32bit"` | `FP32_L2` | existing | Returns `graph` unchanged if already `FP32_L2` (raises `ValueError` if `graph` is `FP16_L2`) |
| `"16bit"` | `FP16_L2` | `[ScalarQuantizerFP16]` | Near-lossless, half the feature RAM |
| `"8bit"` / `"auto"` | `Uint7_L2` | `[(FWHT), ScalarQuantizerUint7]` | `FWHT` auto-selected by kurtosis |
| `"4bit"` | `Uint4_L2` | `[(FWHT), ScalarQuantizerUint4]` | Recommended with `FP16_L2` reranker |
| `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[EvpQuantizer]` | Intended for unit-norm / rerank pipelines |

### 3.2 Source Graph is `InnerProduct` (`FP32_InnerProduct` / `FP16_InnerProduct`)

| `target` / `compress` | Target Metric | Transformations | Notes |
|---|---|---|---|
| `None` / `"32bit"` | `FP32_InnerProduct` | existing | Returns `graph` unchanged if already `FP32_InnerProduct` (raises `ValueError` if `graph` is `FP16_InnerProduct`) |
| `"16bit"` | `FP16_InnerProduct` | `[ScalarQuantizerFP16]` | Native MIPS without reranker |
| `"8bit"` / `"auto"` | `Uint7_L2` or `Int8_InnerProduct` | `[(FWHT), ScalarQuantizer*]` | Uses `predict_quantization(graph)` (**A1+A2** via `predict_quantization_normalized(..., is_cosine=has_l2n)`) on unit-sphere `cosine`/`normalized` graphs, and `Int8_InnerProduct` on unnormalized `InnerProduct` (`ip`) graphs |
| `"4bit"` | `Uint4_L2` or `Int4_InnerProduct` | `[(FWHT), ScalarQuantizer*]` | Uses `Uint4_L2` on unit-sphere `cosine`/`normalized` graphs (no `d+1` needed), and **`Int4_InnerProduct`** (`d`-dim, no `MipsTransform`) on unnormalized `InnerProduct` (`ip`) graphs |
| `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[EvpQuantizer]` | High-compression 2-bit filtering + reranking (`d`-dim, uniform without `MipsTransform` across all metrics) |
| Explicit `L2` `Metric` (`Metric.Uint7_L2`, `Metric.Int8_L2`, `Metric.Uint4_L2`, `Metric.Uint8_L2`) | Target `L2` metric | `[MipsTransform?, (FWHT), ScalarQuantizer*]` *(auto-prepends `MipsTransform` via `mips=True` only when stored graph vectors are not unit-norm and have no prefix)* | Enables `L2` quantizers directly on `InnerProduct` graphs without adding `d+1` overhead on unit-norm `cosine`/`normalized` graphs |

### 3.3 8-Bit Quantizer Predictor Rules (`K5-Top` vs. `A1+A2`)

For `compress="8bit"` / `"auto"`, the target quantizer (`Uint7_L2` vs. `Int8_InnerProduct`) is chosen by:
1. **`"l2"` / `"euclidean"`:** Always **`Uint7_L2`** (no predictor call).
2. **`"ip"` (unnormalized `InnerProduct`):** Always **`Int8_InnerProduct`** (no predictor call).
3. **`"cosine"` / `"normalized"` (unit-sphere `InnerProduct`):**
   - **`K5-Top`** (`predict_quantization(data, ...)`): Used in **`deglib.build`** and **`deglib.reranker_with_data`** when raw unnormalized `data` is available (relies on raw norms via `log1p_mips_extra_ratio` and raw covariance via `top32_var_ratio`).
   - **`A1+A2`** (`predict_quantization(graph, ...)` / `predict_quantization_normalized(data, is_cosine, ...)`): Used in **`deglib.compress`**, **`deglib.searcher(..., compress="8bit")`**, and **`deglib.reranker(..., compress="8bit")`** when only already-normalized 32-bit graph vectors remain (100% L2-normalization-invariant; `is_cosine` is preserved from `L2NormalizeTransform`).

### 3.4 Why Unnormalized `IP` Uses `Int4_InnerProduct` & `EVP_InnerProduct` Without `MipsTransform` (`d + 1`)

Empirical benchmark on both VIBE unnormalized IP datasets (`yi` 187,843 × 128D and `llama` 256,921 × 128D, `128D_FP32_InnerProduct_K16_AddK32Eps0.1_LowLID_FLAS.deg`, `k=10`, `threads=1`, `16-bit` reranker):

| Metric | `MipsTransform` (`d+1`) | `yi` (`ef=200, rr=2.0`) Recall / QPS | `llama` (`ef=200, rr=2.0`) Recall / QPS | Notes |
|---|---|---|---|---|
| `Int4_InnerProduct` | No (`128D`) | `0.3119` / `11,657` | **`0.1348`** / `9,774` | **Best 4-bit IP choice** — beats `Uint4_L2 + MIPS` in recall (+2.2x on `llama`) and QPS, and avoids `d+1` alignment overhead |
| `Int4_InnerProduct` | Yes (`129D`) | `0.3160` / `10,875` | `0.0928` / `9,780` | Worse on `llama` and slower due to `d+1` |
| `Uint4_L2` | Yes (`129D`) | `0.2615` / `11,086` | `0.0605` / `10,647` | Lower recall than `Int4_InnerProduct` (and collapses to ~28–60 QPS under `eps=0.1`) |
| `Uint4_L2` / `Int4_L2` | No (`128D`) | `0.0415` / `0.0349` | `0.0000` / `0.0000` | L2 without `MipsTransform` cannot order unnormalized IP norms |
| `Int4_L2` | Yes (`129D`) | `0.2271` / `9,605` | `0.0645` / `9,470` | Strictly worse than `Uint4_L2 + MIPS` and `Int4_InnerProduct` |
| `Uint4_InnerProduct` | Yes / No | `0.0184` / `0.0004` | `0.0039` / `0.0001` | Unsigned 4-bit IP does not preserve signed inner-product ordering |
| `EVP_InnerProduct` | No (`128D`) | `0.0346` / `13,875` | `0.0003` / `13,061` | Kept `d`-dim (without `MipsTransform`) for uniformity with `Int4_InnerProduct` and `Int8_InnerProduct` |
| `EVP_InnerProduct` | Yes (`129D`) | `0.0485` / `12,262` | `0.0012` / `12,319` | Negligible recall on unnormalized IP, lower QPS |

**Rules derived from this benchmark:**
- Never use `Uint4_InnerProduct` for unnormalized IP.
- Never use `Uint4_L2` or `Int4_L2` on unnormalized IP without `MipsTransform` (`d+1`).
- For `compress="4bit"` on unnormalized IP (`deglib.build(..., metric="ip")` and `deglib.compress` / `deglib.searcher`), always use **`Int4_InnerProduct`** in **`d` dimensions** (no `MipsTransform`).
- For `compress="2bit"` / `"evp"` on `metric="ip"`, always swap back to **`EVP_InnerProduct`** in **`d` dimensions** (no `MipsTransform`) for uniform `d`-dim behavior across all `InnerProduct` compressions (`32bit`, `16bit`, `8bit`, `4bit`, `2bit`).

---

## 4. `deglib.reranker` & `deglib.reranker_with_data`

| Function | Precondition | Behavior & Edge Cases |
|---|---|---|
| `deglib.reranker(graph, compress="16bit")` | `graph` is uncompressed (`FP32_*` or `FP16_*`); raises `ValueError` if `graph` is quantized | Extracts base vectors directly from `graph` (in internal `graph.get_labels()` order). Distance kind (`L2` vs. `InnerProduct`) is inferred from `graph.get_metric()`. Supports `compress` in `{None, "32bit", "16bit", "8bit"/"auto", "int8", "uint8", "uint7", Metric}`. For `"8bit"`/`"auto"`: selects `Uint7_L2` (`L2`), `Int8_InnerProduct` (unnormalized `IP`), or `predict_quantization(graph)` (**A1+A2** on unit-sphere `cosine`/`normalized` graphs). Preserves `AVX512_VNNI` on `Int8_InnerProduct`. |
| `deglib.reranker_with_data(graph, data, labels=..., compress="16bit", metric=...)` | Original `float32`/`float16` `data` available (works on both float and quantized graphs) | Maps `graph.get_labels()` to rows in `data` via `labels` (`0..N-1` fast path, sorted `searchsorted`, or dictionary lookup; raises `KeyError` on missing label) without copying `data`. Preserves any `L2NormalizeTransform` / `MipsTransform` prefix on `graph`. `metric="ip"` selects `InnerProduct` unless a `MipsTransform` prefix is present. For `compress="8bit"`/`"auto"`: selects `Uint7_L2` (`L2`), `Int8_InnerProduct` (unnormalized `IP`), or `predict_quantization(data)` (**K5-Top** on `cosine`/`normalized` `data`). |

- **Shared-prefix deduplication (`Searcher::split_shared_prefix_transforms`):** When `searcher` and `reranker` share a leading spatial prefix (`L2NormalizeTransform` or `MipsTransform`), `Searcher` runs the prefix **once** per query and forwards the transformed `float32` query to both the quantized search graph and `reranker_->rerank_f32(..., reranker_transform_offset_)`.

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

*(Note: `deglib.build(..., metric="ip", compress="8bit")` always applies `make_fast_hadamard_rotation` explicitly.)*

---

## 6. Operational & Serialization Rules

1. **Why not store `L2Norm(d)` features on `IP` graphs?**  
   `L2Norm(d)` discards vector norms (`||x|| = 1`). On true MIPS datasets (e.g. `yi`, `llama`, `llama-dev` with norms `~15` to `~33`), that turns MIPS into Cosine and prevents `deglib.reranker(graph)` from recovering true inner-product scores without external original data.
2. **`save_graph()` on `ReadOnlyGraph`:**  
   `.deg` files store edge weights and require a mutable graph. `DynamicExplorationGraph.save_graph()` automatically converts a `ReadOnlyGraph` via `self.to_mutable().save_graph(path)`.
3. **`.deg` serialization:**  
   All `FeatureTransformation` tags (`TAG_MIPS`, `TAG_NORM`, `TAG_FWHT`, `TAG_SQ16`, `TAG_SQI8`, `TAG_SQU8`, `TAG_SQU7`, `TAG_SQI4`, `TAG_SQU4`, `TAG_EVPQ`) are serialized and deserialized across save/load roundtrips.
4. **`deglib.load_graph(path)`:**  
   High-level loader returning an immutable `ReadOnlyGraph` (`deglib.graph.load_readonly_graph`). Use `deglib.graph.load_dynamic_graph` or `deglib.graph.load_mutable_graph` when you need a mutable graph backend.
