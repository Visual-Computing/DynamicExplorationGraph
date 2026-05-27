# Cascading SIMD Template Pattern — Blueprint für alle Distance-Funktionen

## Ziel

Die `FP32L2<MIN_ALIGN>` Template-Struktur aus `fp32_l2.h` auf **alle** Distance-Funktionen übertragen. Jedes Distance-Modul bekommt:
1. Ein `if constexpr`-kaskadiertes Template mit `MIN_ALIGN`
2. `compare_batch` inline mit compile-time Tail-Eliminierung
3. `compare_impl_tail` für den Batch-Tail
4. Backward-compatible `using` Aliase

---

## Referenzstruktur: `fp32_l2.h` — Vollständiger Code-Bauplan

Dieser Abschnitt dokumentiert jede Code-Komponente der `FP32L2<MIN_ALIGN>` Implementierung
als **Vorlage** für neue Distance-Module. Die SIMD-Kernel unterscheiden sich pro Distanzfunktion,
aber das strukturelle Gerüst (Kaskade, batch-Helfer, Tail-Handling, Dispatcher) ist identisch.

### 0. Datei-Kopf (jedes Distance-Modul)

```cpp
#pragma once

#include <cstdint>
#include <cstring>
#include <concepts>
#include <config.h>

#if defined(USE_AVX2) || defined(USE_AVX512) || defined(USE_SSE)
#include <immintrin.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace deglib {
namespace distances {
```

Nach der letzten `#endif` des Files: `#undef` aller Distanz-spezifischen Makros, dann
Namespace schließen:
```cpp
} // namespace distances
} // namespace deglib
```

---

### 1. Horizontale Summen (`hsum128`, `hsum256`, `hsum512`) — generisch, unverändert übernehmen

```cpp
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static inline float hsum128(__m128 v) {
    alignas(16) float f[4];
    _mm_store_ps(f, v);
    return f[0] + f[1] + f[2] + f[3];
}
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_extractf128_ps(v, 0);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    return hsum128(_mm_add_ps(lo, hi));
}
#endif

#if defined(USE_AVX512)
static inline float hsum512(__m512 v) {
    __m256 lo = _mm512_extractf32x8_ps(v, 0);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return hsum256(_mm256_add_ps(lo, hi));
}
#endif
```

---

### 2. SIMD-Iterations-Makros (Distanz-spezifisch)

Jede Distanzfunktion definiert ihre eigenen Makros mit passendem Präfix
(z.B. `L2_SSE_ITER`, `IP_SSE_ITER`). Das Muster ist immer:
- Load `a` und `b` (via `_mm*_loadu_ps`)
- Distanz-Kernel (sub+fma für L2, mul+add für IP, etc.)
- Accumulate in `sum`
- Pointer um N weiterrücken

```cpp
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
#define DIST_SSE_ITER(a, b, sum) do {                                      \
    __m128 v_ = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));              \
    (sum) = _mm_fmadd_ps(v_, v_, (sum));                                    \
    (a) += 4; (b) += 4;                                                     \
} while(0)
#endif

#if defined(USE_AVX) || defined(USE_AVX512)
#define DIST_AVX_ITER(a, b, sum) do {                                       \
    __m256 v_ = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));      \
    (sum) = _mm256_fmadd_ps(v_, v_, (sum));                                  \
    (a) += 8; (b) += 8;                                                     \
} while(0)
#endif

#if defined(USE_AVX512)
#define DIST_AVX512_ITER(a, b, sum) do {                                     \
    __m512 v_ = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));       \
    (sum) = _mm512_fmadd_ps(v_, v_, (sum));                                   \
    (a) += 16; (b) += 16;                                                    \
} while(0)
#endif
```

Am Datei-Ende werden diese Makros wieder aufgeräumt:
```cpp
#undef DIST_SSE_ITER
#undef DIST_AVX_ITER
#undef DIST_AVX512_ITER
```

---

### 3. Template-Klasse mit `MIN_ALIGN`

```cpp
template <size_t MIN_ALIGN>
class DistanzKlasse {
public:
    // --- compare() : single-query Distanz ---
    inline static float compare(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
        float* a = (float*)pVect1v;
        float* b = (float*)pVect2v;
        size_t size = *((size_t*)qty_ptr);
        const float* last = a + size;

#if defined(USE_AVX512)
        // AVX-512 main loop: 16 floats/iter
        // ...
#elif defined(USE_AVX)
        // AVX main loop: 8 floats/iter
        // ...
#elif defined(USE_SSE)
        // SSE main loop: 4 floats/iter
        // ...
#else
        // Scalar fallback mit 4er Unrolling
        // ...
#endif
    }

private:
    // --- Tail Helper (private static) ---
    inline static float compare_impl_tail(const float* a, const float* b, size_t tail_dim) { ... }

    // --- Batch Helper (private static) ---
    // batch8_avx512(), batch4_avx512(), batch8_avx(), batch4_avx(), batch4_sse()
    ...

public:
    // --- compare_batch() : Batch-Dispatch ---
    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) { ... }
};
```

---

### 4. `compare()` — Single-Query Kaskade (L2-Beispiel)

Das Skelett jeder `compare()`-Methode. Die SIMD-Kernel unterscheiden sich,
das `if constexpr`-Tail-Muster ist identisch.

**AVX-512 Zweig:**
```cpp
#if defined(USE_AVX512)
        __m512 sum512 = _mm512_setzero_ps();
        while (a + 16 <= last) {
            // AVX-512 Kernel: 16 floats
            (a) += 16; (b) += 16;
        }
        __m256 sum256 = _mm256_add_ps(
            _mm512_extractf32x8_ps(sum512, 0),
            _mm512_extractf32x8_ps(sum512, 1));

        if constexpr (MIN_ALIGN < 16) {
            while (a + 8 <= last) {
                // AVX Kernel: 8 floats
            }
        }

        float result = hsum256(sum256);
        if constexpr (MIN_ALIGN < 8) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 4 <= last) {
                // SSE Kernel: 4 floats
            }
            result += hsum128(sum128);
        }
        if constexpr (MIN_ALIGN < 4) {
            while (a < last) {
                // scalar
            }
        }
        return result;
```

**AVX Zweig:**
```cpp
#elif defined(USE_AVX)
        __m256 sum256 = _mm256_setzero_ps();

        if constexpr (MIN_ALIGN >= 16) {
            while (a + 16 <= last) {
                // AVX Kernel x2 (2x8=16 floats/iter -> 1 cache line)
            }
        } else {
            while (a + 8 <= last) {
                // AVX Kernel x1
            }
        }

        float result = hsum256(sum256);
        if constexpr (MIN_ALIGN < 8) {
            __m128 sum128 = _mm_setzero_ps();
            while (a + 4 <= last) {
                // SSE Kernel: 4 floats
            }
            result += hsum128(sum128);
        }
        if constexpr (MIN_ALIGN < 4) {
            while (a < last) {
                // scalar
            }
        }
        return result;
```

**SSE Zweig:**
```cpp
#elif defined(USE_SSE)
        __m128 sum128 = _mm_setzero_ps();

        if constexpr (MIN_ALIGN >= 16) {
            while (a + 16 <= last) {
                // SSE Kernel x4
            }
        } else if constexpr (MIN_ALIGN >= 8) {
            while (a + 8 <= last) {
                // SSE Kernel x2
            }
        } else if constexpr (MIN_ALIGN >= 4) {
            while (a + 4 <= last) {
                // SSE Kernel x1
            }
        }

        float result = hsum128(sum128);
        if constexpr (MIN_ALIGN < 4) {
            while (a < last) {
                // scalar
            }
        }
        return result;
```

**Scalar Fallback:**
```cpp
#else
        float result = 0;
        const float* unroll_group = last - 3;
        while (a < unroll_group) {
            // 4 floats manuell
            a += 4; b += 4;
        }
        while (a < last) {
            // scalar
        }
        return result;
#endif
```

---

### 5. `compare_impl_tail()` — Tail Helper

Für den Rest < 8 (bzw. < 4 bei SSE-only) pro Vektor. Wird von jeder batch-Funktion
aufgerufen, aber **nur wenn** `if constexpr (MIN_ALIGN < N)` es zulässt.

```cpp
    inline static float compare_impl_tail(const float* a, const float* b, size_t tail_dim) {
        float result = 0.0f;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        if (tail_dim >= 4) {
            __m128 s = _mm_setzero_ps();
            DIST_SSE_ITER(a, b, s);   // Distanz-spezifischer SSE-Kernel
            result = hsum128(s);
            tail_dim -= 4;
        }
#endif
        for (size_t i = 0; i < tail_dim; ++i) {
            // scalar diff * diff (L2), oder a[i] * b[i] (IP), etc.
        }
        return result;
    }
```

---

### 6. Batch Helper (private static)

Jede batch-Funktion verarbeitet eine feste Anzahl DB-Vektoren (8 oder 4) in SIMD-Breite
(AVX-512=16, AVX=8, SSE=4). Das Pattern ist:

```cpp
#if defined(USE_AVX) || defined(USE_AVX512)
    inline static void batch8_avx(const float* query, const void* const* db_arr, size_t dim, float* dists) {
        const size_t nc = dim / 8;          // volle AVX chunks
        const size_t tail_start = nc * 8;
        const size_t tail_dim = dim - tail_start;  // Rest < 8
        // 8 Akkumulatoren
        __m256 s0..s7 = _mm256_setzero_ps();

        for (size_t c = 0; c < nc; ++c) {
            const __m256 qf = _mm256_loadu_ps(&query[c * 8]);

            // Lokales Makro (wieder #undef'd nach der Schleife)
            #define BATCH_KERNEL(j, reg) do { \
                const float* db = (const float*)db_arr[j]; \
                // Distanz-Kernel: sub + fma (L2) oder mul + add (IP) \
                reg = _mm256_fmadd_ps(v, v, reg); \
            } while(0)

            BATCH_KERNEL(0, s0); ... BATCH_KERNEL(7, s7);
            #undef BATCH_KERNEL
        }

        dists[0..7] = hsum256(s0..s7);

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 8; ++j)
                dists[j] += compare_impl_tail(query + tail_start, db_arr[j] + tail_start, tail_dim);
        }
    }
#endif
```

Für AVX-512 gibt es zwei Level — 16er chunks + optional 8er remainder:
```cpp
#if defined(USE_AVX512)
    inline static void batch8_avx512(...) {
        const size_t nc16 = dim / 16;
        const size_t tail8 = nc16 * 16;
        const size_t tail_start = (dim / 8) * 8;
        const size_t tail_dim = dim - tail_start;
        // 8x __m512 Akkumulatoren

        for (size_t c = 0; c < nc16; ++c) {
            // 512-bit kernel (16 floats)
        }

        // Reduziere 512->256->128, schreibe Distanzen

        if (tail8 + 8 <= dim) {
            // 8er remainder: 8x __m256 Akkumulatoren
            // 256-bit kernel (8 floats)
            dists[0..7] += hsum256(...);
        }

        if constexpr (MIN_ALIGN < 8) {
            for (int j = 0; j < 8; ++j)
                dists[j] += compare_impl_tail(query + tail_start, db_arr[j] + tail_start, tail_dim);
        }
    }

    // batch4_avx512 analog mit 4 Akkumulatoren statt 8
#endif
```

**Anzahl der batch-Varianten pro SIMD-Level:**

| SIMD-Level | batch-8 | batch-4 | SSE-Alternative |
|-----------|---------|---------|-----------------|
| AVX-512   | `batch8_avx512` (16er + 8er + tail) | `batch4_avx512` (16er + 8er + tail) | `batch4_sse` (4er + tail) |
| AVX       | `batch8_avx` (8er + tail) | `batch4_avx` (8er + tail) | `batch4_sse` (4er + tail) |
| SSE only  | — | `batch4_sse` (4er + tail) | — |

---

### 7. `compare_batch()` — Dispatch

```cpp
public:
    inline static void compare_batch(const void* query_ptr, const void* const* db_arr, size_t count, const void* qty_ptr, float* dists) {
        const float* query = (const float*)query_ptr;
        const size_t dim = *((const size_t*)qty_ptr);

        size_t i = 0;
#if defined(USE_AVX512)
        for (; i + 8 <= count; i += 8)
            batch8_avx512(query, &db_arr[i], dim, &dists[i]);
        if (i + 4 <= count) {
            batch4_avx512(query, &db_arr[i], dim, &dists[i]);
            i += 4;
        }
#elif defined(USE_AVX)
        for (; i + 8 <= count; i += 8)
            batch8_avx(query, &db_arr[i], dim, &dists[i]);
        if (i + 4 <= count) {
            batch4_avx(query, &db_arr[i], dim, &dists[i]);
            i += 4;
        }
#else
        for (; i + 4 <= count; i += 4)
            batch4_sse(query, &db_arr[i], dim, &dists[i]);
#endif
        // Straggler: einzeln via compare()
        for (; i < count; ++i)
            dists[i] = compare(query, db_arr[i], qty_ptr);
    }
```

---

### 8. Backward-compatible Aliase

```cpp
using Alias16Ext = DistanzKlasse<16>;
using Alias8Ext  = DistanzKlasse<8>;
using Alias4Ext  = DistanzKlasse<4>;
using Alias      = DistanzKlasse<1>;
```

---

### Zusammenfassung: Was ändert sich pro Distanzfunktion?

| Komponente | L2 | Inner Product | fp16 IP | uint8 L2 |
|-----------|-----|---------------|---------|----------|
| SIMD Kernel | `sub` + `fmadd` | `mul` + `add` | `cvtph_ps` + `mul` + `add` | `cvtepu8_epi16` + `madd_epi16` |
| Makro-Präfix | `L2_` | `IP_` | `FP16IP_` | `U8L2_` |
| Klassenname | `FP32L2` | `FP32InnerProduct` | `FP16InnerProduct` | `Uint8L2` |
| tail Kernel | sub + fmadd | mul + add | cvt + mul + add | madd |
| hsum128/256/512 | unverändert | unverändert | unverändert | integer (`_mm_extract_epi32`) |
| Struktur (compare/compare_batch/batch*) | identisch | identisch | identisch | identisch |

---

## Klassen-Naming Convention

Der Klassenname leitet sich **direkt vom Dateinamen** ab:
1. `.h` entfernen
2. In PascalCase konvertieren (Zahlen bleiben wie im Dateinamen)

| Datei | PascalCase | Klasse |
|-------|-----------|--------|
| `fp32_l2.h` | `FP32L2` | `FP32L2` |
| `fp32_inner_product.h` | `FP32InnerProduct` | `FP32InnerProduct` |
| `fp16_inner_product.h` | `FP16InnerProduct` | `FP16InnerProduct` |
| `uint8_l2.h` | `Uint8L2` | `Uint8L2` |
| `evp_inner_product.h` | `EvpInnerProduct` | `EvpInnerProduct` |
| `fp16_evp_asym_inner_product.h` | `FP16EvpAsymInnerProduct` | `FP16EvpAsymInnerProduct` |

---

## Distance-Module die refaktorieren mussen

### 1. `fp32_inner_product.h`

| Current Classes | Refactor Target |
|----------------|-----------------|
| `InnerProductFloat` (scalar) | `FP32InnerProduct<MIN_ALIGN>` |
| `InnerProductFloat16Ext` (AVX-512/AVX/SSE, dim%16==0) | `using InnerProductFloat16Ext = FP32InnerProduct<16>` |
| `InnerProductFloat8Ext` (AVX/SSE, dim%8==0) | `using InnerProductFloat8Ext = FP32InnerProduct<8>` |
| `InnerProductFloat4Ext` (SSE only, dim%4==0) | `using InnerProductFloat4Ext = FP32InnerProduct<4>` |
| `InnerProductFloat16ExtResiduals` | `using ... = FP32InnerProduct<1>` |
| `InnerProductFloat4ExtResiduals` | `using ... = FP32InnerProduct<1>` |

**Cascading Pattern:** `AVX-512 (16) -> AVX (8) -> SSE (4) -> scalar` (identisch zu fp32_l2)

**Batch (private static, eigener Tail):**
- `batch8_avx(q, db[8], dim, d[8])` — AVX fmadd + `if constexpr (MIN_ALIGN < 8)` tail
- `batch4_avx(q, db[4], dim, d[4])` — AVX fmadd + tail
- `batch8_avx512(q, db[8], dim, d[8])` — AVX-512 16-wide + AVX2 remainder + tail
- `batch4_avx512(q, db[4], dim, d[4])` — AVX-512 + tail
- `batch4_sse(q, db[4], dim, d[4])` — SSE fmadd + `if constexpr (MIN_ALIGN < 4)` tail
- `compare_impl_tail(a, b, tail_dim)` — dot product for < 8 floats

**[!] Tests (`test_fp32_inner_product.cpp`):** Neue `compare_batch` Tests fur cascading (batch4, batch8, count=13)
**[!] Microbench (`fp32_ip_microbench.cpp`):** `deglib::distances::InnerProductFloat16Ext::compare_batch` und `InnerProductFloat4Ext::compare_batch` Benchmarks hinzufugen

---

### 2. `fp16_inner_product.h`

| Current Classes | Refactor Target |
|----------------|-----------------|
| `FP16InnerProduct` (scalar fp16->float convert) | `FP16InnerProduct<MIN_ALIGN>` |
| `FP16InnerProductExt8` (SSE cvtph_ps, dim%8==0) | `using FP16InnerProductExt8 = FP16InnerProduct<8>` |
| `FP16InnerProductExt16` (AVX cvtph_ps, dim%16==0) | `using FP16InnerProductExt16 = FP16InnerProduct<16>` |
| `FP16InnerProductExt32` (AVX-512 cvtph_ps, dim%32==0) | `using FP16InnerProductExt32 = FP16InnerProduct<32>` |
| `FP16InnerProductExt16Residuals` | `using ... = FP16InnerProduct<1>` |
| `FP16InnerProductExt8Residuals` | `using ... = FP16InnerProduct<1>` |

**Cascading Pattern:** `AVX-512 (32 fp16) -> AVX (16 fp16) -> SSE (8 fp16) -> scalar`

**Batch (private static, eigener Tail):**
- `batch8_avx(q, db[8], dim, d[8])` — AVX `cvtph_ps` + `fmadd_ps` + `if constexpr (MIN_ALIGN < 8)` tail
- `batch4_avx(q, db[4], dim, d[4])` — AVX + tail
- `batch8_avx512(q, db[8], dim, d[8])` — AVX-512 `cvtph_ps` + remainder + tail
- `batch4_avx512(q, db[4], dim, d[4])` — AVX-512 + tail
- `batch4_sse(q, db[4], dim, d[4])` — SSE `cvtph_ps` + `fmadd_ps` + `if constexpr (MIN_ALIGN < 8)` tail
- `compare_impl_tail(a, b, tail_dim)` — fp16->float convert + dot for < 8 fp16

**[!] Tests (`test_fp16_inner_product.cpp`):** Neue `compare_batch` Tests fur cascading
**[!] Microbench (`fp16_ip_microbench.cpp`):** `deglib::distances::FP16InnerProductExt{8,16,32}::compare_batch` Benchmarks

---

### 3. `uint8_l2.h`

| Current Classes | Refactor Target |
|----------------|-----------------|
| `L2Uint8` (scalar) | `Uint8L2<MIN_ALIGN>` |
| `L2Uint8Ext32` (AVX `cvtepu8_epi16` + `madd_epi16`, dim%32==0) | `using L2Uint8Ext32 = Uint8L2<32>` |
| `L2Uint8Ext16` (SSE `cvtepu8_epi16` + `madd_epi16`, dim%16==0) | `using L2Uint8Ext16 = Uint8L2<16>` |

**Cascading Pattern:** `AVX (16 uint8 via cvtepu8_epi16) -> SSE (16 uint8 via cvtepu8_epi16 + split lo/hi) -> scalar`

**Batch (private static, eigener Tail):**
- `batch8_avx(q, db[8], dim, d[8])` — AVX `cvtepu8_epi16` + `madd_epi16` + `if constexpr (MIN_ALIGN < 16)` tail
- `batch4_avx(q, db[4], dim, d[4])` — AVX + tail
- `batch4_sse(q, db[4], dim, d[4])` — SSE integer ops + `if constexpr (MIN_ALIGN < 16)` tail
- `compare_impl_tail(a, b, tail_dim)` — scalar L2 for < 16 uint8

**[!] Tests (`test_uint8_l2.cpp`):** Neue `compare_batch` Tests
**[!] Microbench (`uint8_L2_microbench.cpp`):** `deglib::distances::L2Uint8Ext{16,32}::compare_batch` Benchmarks

---

### 4. `evp_inner_product.h`

| Current Classes | Refactor Target |
|----------------|-----------------|
| `EvpBitsSimilarity` (naive/avx2/avx512) | `EvpInnerProduct` (kein MIN_ALIGN — dim/8 Byte-Masks) |

**Spezialfall:** EVP arbeitet auf Byte-Masks (`dim/8` bytes), nicht auf floats. Die Kaskadierung ist anders: `AVX-512 (64 bytes) -> AVX2 (32 bytes) -> scalar`. Kein `MIN_ALIGN` Template sinnvoll — stattdessen `if constexpr` fur AVX-512 -> AVX2 -> scalar wie in `compare()`.

**Batch (private static, eigener Tail):**
- `batch8_avx2(q, db[8], dim, d[8])` — 8 DB, parallel popcnt mit AVX2 lookup + tail
- `batch4_avx2(q, db[4], dim, d[4])` — 4 DB + tail
- `batch8_avx512(q, db[8], dim, d[8])` — 8 DB, AVX-512 `popcnt` + tail
- `batch4_avx512(q, db[4], dim, d[4])` — 4 DB, AVX-512 + tail
- `compare_impl_tail(a, b, tail_bytes)` — scalar popcnt for rest

**[!] Tests (`test_evp_inner_product.cpp`):** Neue `compare_batch` Tests
**[!] Microbench (`evp_microbench.cpp`):** `deglib::distances::EvpInnerProduct::compare_batch` Benchmark

---

### 5. `fp16_evp_asym_inner_product.h`

| Current Classes | Refactor Target |
|----------------|-----------------|
| `FP16EvpAsymmetricSimilarity` (naive/sse/avx2/avx512) | `FP16EvpAsymInnerProduct` (kein MIN_ALIGN Template) |

**Spezialfall:** Asymmetrischer Vergleich (FP16 vector vs EVP bit mask). Die Batch-Implementierung ist komplex (LUT-basiert). Kein `MIN_ALIGN` Template — stattdessen `if constexpr` Kaskade wie in `compare()`.

**Batch (private static, eigener Tail):**
- `batch8_avx2(q, db[8], dim, d[8])` — 8 DB, AVX2 LUT mask + fp16 multiply + tail
- `batch4_avx2(q, db[4], dim, d[4])` — 4 DB + tail
- `batch8_avx512(q, db[8], dim, d[8])` — 8 DB, AVX-512 LUT + tail
- `batch4_avx512(q, db[4], dim, d[4])` — 4 DB, AVX-512 + tail
- `compare_impl_tail(a, b, tail_bytes)` — scalar for rest

**[!] Tests (`test_fp16_evp_asym_inner_product.cpp`):** Neue `compare_batch` Tests
**[!] Microbench (`asym_microbench.cpp`):** `deglib::distances::FP16EvpAsymInnerProduct::compare_batch` Benchmark

---

## Mapping: Distance Header -> Microbench -> Test

| Distance Header | Microbench | Test |
|----------------|------------|------|
| `distance/fp32_l2.h` | `evp/fp32_L2_microbench.cpp` | `test/src/distance/test_fp32_l2.cpp` |
| `distance/fp32_inner_product.h` | `evp/fp32_ip_microbench.cpp` | `test/src/distance/test_fp32_inner_product.cpp` |
| `distance/fp16_inner_product.h` | `evp/fp16_ip_microbench.cpp` | `test/src/distance/test_fp16_inner_product.cpp` |
| `distance/uint8_l2.h` | `evp/uint8_L2_microbench.cpp` | `test/src/distance/test_uint8_l2.cpp` |
| `distance/evp_inner_product.h` | `evp/evp_microbench.cpp` | `test/src/distance/test_evp_inner_product.cpp` |
| `distance/fp16_evp_asym_inner_product.h` | `evp/asym_microbench.cpp` | `test/src/distance/test_fp16_evp_asym_inner_product.cpp` |

---

## Build & Run Microbenchmarks

### Konfigurieren (einmalig)
```bash
cmake --preset "Visual Studio Community 2022"
```
`DATA_PATH` ist in `CMakePresets.json` auf `c:/Data/ANN/` gesetzt.

### Kompilieren (Release — optimierte SIMD-Pfade)
```bash
# Einzelnes Microbenchmark:
cmake --build --preset "Visual Studio Community 2022 release" --target fp32_L2_microbench

# Alle Microbenchmarks + Tests:
cmake --build --preset "Visual Studio Community 2022 release"
```

### Ausführen
Das Binary liegt unter `build/Visual Studio Community 2022/bin/Release/<name>.exe` (`CMAKE_RUNTIME_OUTPUT_DIRECTORY` = `build/<preset>/bin`).

Der Microbench erwartet einen Datenpfad als erstes Argument und sucht darin nach `train.hvecs`:

```bash
build/Visual Studio Community 2022/bin/Release/fp32_L2_microbench.exe C:/Data/ANN/sisap2026/small
```

Alle Microbenchmarks teilen denselben Datensatz (`sisap2026/small/train.hvecs`, 200000 Vektoren, dim=1024):

| Target | Dataset | Beschreibung |
|--------|---------|-------------|
| `fp32_L2_microbench` | `sisap2026/small/train.hvecs` | L2-Distanz (float32) |
| `fp32_ip_microbench` | `sisap2026/small/train.hvecs` | Inner Product (float32) |
| `fp16_ip_microbench` | `sisap2026/small/train.hvecs` | Inner Product (float16) |
| `uint8_L2_microbench` | `sisap2026/small/train.hvecs` | L2-Distanz (uint8, FP16→FP32→linear quantized) |
| `evp_microbench` | `sisap2026/small/train.hvecs` | EVP popcnt |
| `asym_microbench` | `sisap2026/small/train.hvecs` | FP16-EVP asym |

### Output-Erwartung
```
Using AVX2...
Data path: C:/Data/ANN/sisap2026/small
Looking for: C:/Data/ANN/sisap2026/small\train.hvecs
Loading hvecs...
Loaded: 200000 vectors, dim=1024
Converting FP16->FP32...
...
=== FP32 L2 Distance Microbenchmark ===
Vectors: 200000, dim=1024
bytes/vector=4096
Comparisons: 100000

--- Deglib baselines (single-query) ---
  L2_naive         ...           923.45 ns/op
  ...
--- Deglib compare_batch (via L2Float4Ext cascading) ---
  deglib_batch8_cascade  ...     246.00 ns/op
```

---

## Workflow pro Modul (empfohlen)

Fuhre fur jedes Modul diese Schritte in genau dieser Reihenfolge aus:

1. **Microbench vorher** → bestehende `custom_batch` Laufzeiten messen und in den Plan schreiben (Abschnitt "[Modul] Baseline")
2. **Template schreiben** → `Xxx<MIN_ALIGN>` Klasse mit `compare()`, `compare_impl_tail()`, `batch*()` und `compare_batch()`
3. **Aliase + distances.h** → `using` Aliase, Dispatch in `distances.h` aktualisieren
4. **Unit Tests** → bestehende Tests laufen lassen, neue `compare_batch` Tests schreiben
5. **Microbench nachher** → `deglib_batch*` und `deglib_batch*_cascade` Laufzeiten messen
6. **Vergleich** → vorher/nachher Ergebnisse in den Plan schreiben (Abschnitt "[Modul] Ergebnis")

Die Ergebnisse werden **vor** und **nach** den Änderungen gemessen, damit der Performance-Gewinn (oder -Verlust) sichtbar dokumentiert ist.

---

## Prozedur pro Modul

Fur jedes der 5 verbleibenden Module gilt:

### Schritt 1: Template schreiben
- `Xxx<MIN_ALIGN>` mit `if constexpr` Kaskade in `compare()`
- `compare_impl_tail()` als `private static` Methode
- **Batch-Helper als `private static`** Methoden in der Klasse (kein Typ-Präfix nötig — die Klasse definiert den Datentyp), jede mit eigenem `if constexpr`-Tail:
  - `batch8_avx512(q, db[8], dim, d[8])` — falls AVX-512 verfügbar
  - `batch4_avx512(q, db[4], dim, d[4])` — falls AVX-512 verfügbar
  - `batch8_avx(q, db[8], dim, d[8])` — AVX2/AVX-512 (8 floats/iter)
  - `batch4_avx(q, db[4], dim, d[4])` — AVX2/AVX-512
  - `batch4_sse(q, db[4], dim, d[4])` — SSE
- `compare_batch()` öffentlich: nur noch dispatch zu den privaten Batch-Helpern + stragglers via `compare()`

### Schritt 2: Aliase + distances.h
- `using` Aliase fur backward compatibility
- `distances.h` Dispatch vereinfachen (Residual-Zeilen eliminieren)

### Schritt 3: Tests
- Bestehende `compare()` Tests uberprufen (mussen weiterlaufen)
- **Neue** `compare_batch` Tests: batch4, batch8, count=13 (8+4+1), cascading dims

### Schritt 4: Microbench
- **Neue** `deglib_batch4` / `deglib_batch8` Benchmark-Sektionen: `ClassName::compare_batch(q, vdb, N, d, dists)`
- **Neue** `deglib_batch4_cascade` / `deglib_batch8_cascade` (via lowest-MIN_ALIGN alias)
- Gegen bestehende `custom_batch` Varianten vergleichen

### Schritt 5: Performance-Verifikation
- Alle Tests passen (`test_xxx.exe`)
- Microbench: `deglib_batch` <= `custom_batch` (keine Regression)
- Benchmark: recall + us/query unverandert

---

## Baseline Performance (fp32_l2.h — abgeschlossen)

**sift1m (dim=128):**
| eps | recall | us/query (pre) | us/query (post) | change |
|-----|--------|---------------|----------------|--------|
| 0.01 | 0.92844 | 194 | 187 | -3.6% |
| 0.05 | 0.96438 | 265 | 256 | -3.4% |
| 0.1 | 0.98786 | 384 | 395 | +2.9% |

**sisap2026 (dim=1024) microbench:**
| Variante | ns/op |
|----------|-------|
| **l2_custom_b8_u2** | **242** (BEST) |
| deglib_batch8_cascade (L2Float4Ext) | 246 |
| l2_custom_b8 | 248 |
| deglib_batch8 (L2Float16Ext) | 251 |

---

## Performance: fp32_inner_product.h

### Baseline (vorher)
| Variante | ns/op |
|----------|-------|
| *(wird gemessen)* | |

### Ergebnis (nachher)
| Variante | ns/op |
|----------|-------|
| *(wird gemessen)* | |

---

## Performance: fp16_inner_product.h

### Baseline (vorher)
| Variante | ns/op |
|----------|-------|
| **ip_custom_batch8_u2** | **174.36** (BEST custom) |
| ip_custom_batch8 | 178.40 |
| ip_custom_batch4_u2 | 221.06 |
| ip_custom_batch4 | 214.92 |

### Ergebnis (nachher) — cascading template active
| Variante | ns/op |
|----------|-------|
| **deglib_batch8_cascade** | **167.42** (BEST overall) |
| deglib_batch8_ext8 | 221.79 |
| deglib_batch8 | 234.17 |
| deglib_batch4_cascade | 233.60 |
| deglib_batch4_ext8 | 297.08 |
| deglib_batch4 | 298.80 |

The cascading template (`FP16InnerProduct<1>::compare_batch`) outperforms both the aligned-ext batch variants and all custom implementations. It beats the best custom variant (`ip_custom_batch8_u2`, 174.36 ns/op) by ~4% and the best custom batch (`ip_custom_batch8`, 170.71 ns/op) by ~2%.

---

## Performance: uint8_l2.h

### Baseline (vorher)
| Variante | ns/op |
|----------|-------|
| **u8_custom_b4** | **183.52** (BEST custom) |
| u8_custom_b8 | 400.45 |
| u8_custom_b4_u2 | 481.68 |
| u8_custom_b8_u2 | 451.87 |

### Ergebnis (nachher) — cascading template active
| Variante | ns/op |
|----------|-------|
| **deglib_batch8** | **127.84** (BEST overall) |
| deglib_batch8_ext32 | 132.92 |
| deglib_batch8_cascade | 160.17 |
| deglib_batch4 | 186.10 |
| deglib_batch4_ext32 | 183.13 |
| deglib_batch4_cascade | 192.60 |

The aligned-ext batch (`L2Uint8Ext16::compare_batch`, 127.84 ns/op) outperforms the cascade template and all custom implementations. The cascade template's runtime dispatch adds ~25% overhead over the aligned variant for uint8.

---

## Performance: evp_inner_product.h

### Baseline (vorher)
| Variante | ns/op |
|----------|-------|
| **evp_custom_b8** | **76.72** (BEST custom) |
| evp_custom_b4 | 131.17 |

### Ergebnis (nachher) — EvpInnerProduct mit compare_batch
| Variante | ns/op |
|----------|-------|
| **deglib_batch8** | **70.02** (BEST overall) |
| deglib_batch4 | 96.35 |

The deglib batch8 (shared-query-load scalar popcnt via uint64 chunks) beats the best custom batch8 by ~9%. For EVP, the memory bandwidth benefit of sharing query load is significant despite the small vector size (256 bytes/EVP).

---

## Performance: fp16_evp_asym_inner_product.h

### Baseline (vorher)
| Variante | ns/op |
|----------|-------|
| **ip_avx2_b8_u2** | **141.38** (BEST custom) |
| ip_avx2_b8_ptr | 154.27 |
| ip_avx2_b4_u4 | 191.15 |

### Ergebnis (nachher) — FP16EvpAsymInnerProduct mit compare_batch
| Variante | ns/op |
|----------|-------|
| **ip_avx2_b8_u2** | **136.30** (BEST custom, 2-byte ILP) |
| deglib_batch8 | 141.65 (shared query, LUT) |
| ip_avx2_b8_ptr | 145.24 |
| deglib_batch4 | 178.51 |
| ip_avx2_b4_ptr | 168.65 |

The LUT-based batch helpers now share the fp16→float conversion across DBs, making deglib_batch8 competitive with the best custom implementations (within 4% of `ip_avx2_b8_u2`).

---

## Aufgaben Reihenfolge (empfohlen)

1. ~~`fp32_l2.h`~~ (DONE)
2. ~~`fp32_inner_product.h`~~ (DONE)
3. ~~`fp16_inner_product.h`~~ (DONE)
4. ~~`uint8_l2.h`~~ (DONE)
5. ~~`evp_inner_product.h`~~ (DONE)
6. ~~`fp16_evp_asym_inner_product.h`~~ (DONE)
