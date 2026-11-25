# Python High-Level API: Metric, Quantization & Pipeline Matrix

This reference documents the exact internal behavior of the high-level Python API (`deglib.presort`, `deglib.build`, `deglib.compress`, `deglib.searcher`, `deglib.reranker`, `deglib.reranker_with_data`, `deglib.load_graph`) for every `metric` and `compress` option, along with the exact source locations and edge-case rules.

**Implementation map:**
- Spatial metric resolution & prefix transforms (`is_l2_normalized`, `resolve_spatial_metric`): `python/src/deglib/transformation.py`
- 1D FLAS pre-sorting (`presort`): `python/src/deglib/optimization.py`
- High-level build & post-build finalization (`build`): `python/src/deglib/builder.py`
- Compression & metric conversion (`compress`, `_prepare_metric_transform`, `load_graph`): `python/src/deglib/graph.py`
- Reranker & searcher wrappers (`reranker`, `reranker_with_data`, `searcher`): `python/src/deglib/search.py`
- Graph-based quantizer calibration (`make_quantizer_pipeline_from_graph`, `fit_mips_transform_from_graph`): `python/src/deg_cpp/bindings_transformation.cpp`
- Quantizer pipeline & FWHT kurtosis thresholds (`make_quantizer_pipeline`): `cpp/deglib/include/deglib/transformation.h`

---

## 1. `deglib.presort` & `deglib.build` by High-Level Metric

> **Mutability rule:** `deglib.compress(graph, target)` always preserves the backend type of `graph` (mutable stays mutable, `ReadOnlyGraph` stays `ReadOnlyGraph`). `deglib.build(...)` returns a mutable `SizeBoundedGraph` for `"l2"` / `"cosine"` and for `"ip"` with `compress=None`/`"32bit"`/`"2bit"`/explicit `L2`, and returns a `ReadOnlyGraph` for `"ip"` with `compress="16bit"`/`"8bit"`/`"4bit"`. For dynamically growing streaming graphs, use `deglib.builder.GraphBuilder` with `deglib.graph.create_dynamic_empty`.

| `metric` (aliases) | `deglib.presort(data, metric=...)` | `deglib.build` (Graph Construction) | `deglib.build` Post-Build Finalization (`compress=None`) | Rationale |
|---|---|---|---|---|
| `"l2"` / `"euclidean"` | `FP32_L2` (`float32`) / `FP16_L2` (`float16`) directly on `data` (`dim = d`) | `FP32_L2` / `FP16_L2` directly on `data` (`dim = d`, no transforms) | Mutable `SizeBoundedGraph` (`dim = d`) | True metric; no spatial transformation needed |
| `"cosine"` / `"normalized"` | `FP32_L2` / `FP16_L2` directly on `data` (`dim = d`) | `FP32_InnerProduct` / `FP16_InnerProduct`; attaches `L2NormalizeTransform` if `not is_l2_normalized(data)` (`dim = d`). `deglib.build(..., presort=True)` sorts on the **normalized** vectors. | Mutable `SizeBoundedGraph` (`dim = d`, keeps `[L2NormalizeTransform]` if attached) | Automatically normalizes DB and query vectors when input is not unit-norm |
| `"ip"` / `"innerproduct"` / `"inner_product"` / `"mips"` | `make_mips_transform(data)` → FLAS in `FP32_L2` (`dim = d + 1`) | `make_mips_transform(data)` → build in `FP32_L2` / `FP16_L2` (`dim = d + 1`) | `.with_features(work_data, metric=FP32_InnerProduct / FP16_InnerProduct, order="internal")` (**`dim = d`**, **no** transforms, mutable `SizeBoundedGraph`) | `MIPS(d+1)` yields stronger graph topology; swapping back to unnormalized `d`-dim `InnerProduct` preserves norms, removes `d+1` from search (`128 B` alignment / `AVX512_VNNI`), and lets `searcher(graph)` / `reranker(graph)` run without external `data` |
| Explicit `Metric` (e.g. `Metric.Int8_L2`, `Metric.EVP_InnerProduct`) | `FP32`/`FP16` L2 or IP matching `metric` (`presort=True` raises `ValueError` on pre-quantized inputs) | Quantizes `float32`/`float16` input before `add_entry` via `make_quantizer_pipeline(...)`; or ingests pre-quantized arrays directly | Mutable `SizeBoundedGraph` (or `ReadOnlyGraph` if `compress` is set) | Direct control without high-level spatial resolution |

### Unit-Norm Detection Tolerances
- **`is_l2_normalized(data)`** (`transformation.py` for `"cosine"` / `"normalized"`): checks `| ||x||_2 - 1 | <= tol` with `tol = 1e-2` (`float16`) and `tol = 1e-3` (`float32`). Empty arrays (`n == 0`) are treated as normalized.
- **`fit_mips_transform_from_graph(internal, ...)`** (`bindings_transformation.cpp` for `InnerProduct -> L2` conversion): computes `||x||^2 = 1 - ip_dist(x, x)` via `FloatSpace(FP16/FP32_InnerProduct)` across all `n` vectors and skips `MipsTransform` when `|min_norm_sq - 1| <= tol` and `|max_norm_sq - 1| <= tol` (`tol = 1e-2` for `FP16`, `1e-3` for `FP32`).

---

## 2. `deglib.build(..., compress=...)` – Target Quantization

| `metric` | `compress` | Output Metric | Attached Query Transformations | `dim` | Backend |
|---|---|---|---|---:|---|
| `"l2"` | `None` / `"32bit"` | `FP32_L2` (`FP16_L2` for `float16`) | `[]` | `d` | `SizeBoundedGraph` |
| `"l2"` | `"16bit"` | `FP16_L2` | `[ScalarQuantizerFP16]` *(or `[]` if input is `float16`)* | `d` | `SizeBoundedGraph` |
| `"l2"` | `"8bit"` / `"auto"` | `Uint7_L2` | `[(FWHT), ScalarQuantizerUint7]` | `d` | `SizeBoundedGraph` |
| `"l2"` | `"4bit"` | `Uint4_L2` | `[(FWHT), ScalarQuantizerUint4]` | `d` | `SizeBoundedGraph` |
| `"l2"` | `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[EvpQuantizer]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `None` / `"32bit"` | `FP32_InnerProduct` (`FP16_InnerProduct` for `float16`) | `[L2NormalizeTransform]` *(if input unnormalized)* | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"16bit"` | `FP16_InnerProduct` | `[L2Norm?, ScalarQuantizerFP16]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"8bit"` / `"auto"` | `predict_quantization(...)` (`Int8_InnerProduct` vs. `Uint7_L2`) | `[L2Norm?, (FWHT), ScalarQuantizer*]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"4bit"` | `Int4_InnerProduct` | `[L2NormalizeTransform?, (FWHT), ScalarQuantizerInt4]` | `d` | `SizeBoundedGraph` |
| `"cosine"` | `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[L2Norm?, EvpQuantizer]` | `d` | `SizeBoundedGraph` |
| **`"ip"`** | `None` / `"32bit"` | `FP32_InnerProduct` (`FP16_InnerProduct` for `float16`) | `[]` *(unnormalized)* | `d` | `SizeBoundedGraph` |
| **`"ip"`** | `"16bit"` | `FP16_InnerProduct` | `[]` *(unnormalized)* | `d` | `ReadOnlyGraph` |
| **`"ip"`** | `"8bit"` / `"auto"` | `Int8_InnerProduct` (`AVX512_VNNI`) | `[FastHadamardRotation, ScalarQuantizerInt8]` *(unnormalized)* | `d` | `ReadOnlyGraph` |
| **`"ip"`** | `"4bit"` | `Int4_InnerProduct` | `[L2NormalizeTransform, FastHadamardRotation, ScalarQuantizerInt4]` | `d` | `ReadOnlyGraph` |
| **`"ip"`** | `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[MipsTransform, EvpQuantizer]` | `d + 1` | `SizeBoundedGraph` |
| **`"ip"`** | Explicit `L2` metric (e.g. `Metric.Uint7_L2`, `Metric.Int8_L2`) | Target `L2` metric | `[MipsTransform, (FWHT), ScalarQuantizer*]` | `d + 1` | `SizeBoundedGraph` |

### Empty & Pre-Quantized Edge Cases in `deglib.build`
- **`n_samples == 0`**: returns an empty `SizeBoundedGraph` (`dim = d`, `metric = FP32_*` or `FP16_*` for `float16`); `compress` is skipped (calling `deglib.compress` directly on an empty graph raises `ValueError`).
- **Pre-quantized input** (e.g. `int8`, `uint8`): `compress` must be `None`, `presort=True` raises `ValueError` (precomputed `sorted_indices` array is still allowed).

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
| `"8bit"` / `"auto"` | `Int8_InnerProduct` | `[(FWHT), ScalarQuantizerInt8]` | `AVX512_VNNI`; for `≥ 80%` Recall@30 on MIPS pair with `FP16_IP` reranker (`rf ≈ 2.0`) + `optimize(n_clusters=256, use_routing=True)` |
| `"4bit"` | `Int4_InnerProduct` | `[L2NormalizeTransform, (FWHT), ScalarQuantizerInt4]` *(if no `L2Norm` prefix exists yet)* | Calibrated via `normalize=True` on L2-normalized vectors; lossy on true MIPS with wide norm spread |
| `"2bit"` / `"evp"` | `EVP_InnerProduct` | `[EvpQuantizer]` | High-compression 2-bit filtering + reranking |
| Explicit `L2` `Metric` (`Metric.Uint7_L2`, `Metric.Int8_L2`, `Metric.Uint4_L2`, `Metric.Uint8_L2`) | Target `L2` metric | `[MipsTransform?, (FWHT), ScalarQuantizer*]` *(auto-prepends `MipsTransform` via `mips=True` only when stored graph vectors are not unit-norm and have no prefix)* | Enables `L2` quantizers directly on `InnerProduct` graphs without adding `d+1` overhead on unit-norm `cosine`/`normalized` graphs |

---

## 4. `deglib.reranker` & `deglib.reranker_with_data`

| Function | Precondition | Behavior & Edge Cases |
|---|---|---|
| `deglib.reranker(graph, compress="16bit")` | `graph` is uncompressed (`FP32_*` or `FP16_*`); raises `ValueError` if `graph` is quantized | Extracts base vectors directly from `graph` (in internal `graph.get_labels()` order). Distance kind (`L2` vs. `InnerProduct`) is inferred from `graph.get_metric()`. Supports `compress` in `{None, "32bit", "16bit", "8bit"/"int8", "uint8", "uint7", Metric}`. Preserves `AVX512_VNNI` on `Int8_InnerProduct`. |
| `deglib.reranker_with_data(graph, data, labels=..., compress="16bit", metric=...)` | Original `float32`/`float16` `data` available (works on both float and quantized graphs) | Maps `graph.get_labels()` to rows in `data` via `labels` (`0..N-1` fast path, sorted `searchsorted`, or dictionary lookup; raises `KeyError` on missing label) without copying `data`. Preserves any `L2NormalizeTransform` / `MipsTransform` prefix on `graph`. `metric="ip"` selects `InnerProduct` unless a `MipsTransform` prefix is present. |

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

*(Note: `deglib.build(..., metric="ip", compress="8bit"|"4bit")` always applies `make_fast_hadamard_rotation` explicitly.)*

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
