# Umsetzungsplan Phase 2: FP16 & Basis-Distanzfunktionen (TDD)

Dieser Detailplan fokussiert sich auf Phase 2: Die Test-getriebene Einführung von FP16-Datentyp-Unterstützung und modularisierten Distanzberechnungsfunktionen (FP32, FP16, UInt8) in `deglib`.

---

## Betroffene Komponenten & Dateien in Phase 2

### TDD Step 1: Unit-Tests übernehmen (Test-First)
Bevor Distanz-Code angepasst wird, werden die spezifischen Unit-Tests aus dem Branch `evp` übernommen:

#### [NEW] [test_fp16.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_fp16.cpp)
- Testet Konvertierung zwischen float32 und float16 (`deglib::f16`), F16C / AVX2 Konvertierungshandler.

#### [NEW] [test_fp32_l2.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_fp32_l2.cpp)
- Testet exakte L2-Distanzberechnung für FP32-Vektoren (skalar, SSE, AVX2, AVX512, Batched).

#### [NEW] [test_fp32_inner_product.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_fp32_inner_product.cpp)
- Testet Inner-Product (Skalarprodukt) Distanzberechnung für FP32.

#### [NEW] [test_fp16_inner_product.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_fp16_inner_product.cpp)
- Testet Inner-Product Distanzen auf FP16 Vektordaten.

#### [NEW] [test_uint8_l2.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_uint8_l2.cpp)
- Testet L2-Distanzen auf vorzeichenlosen 8-Bit Ganzzahldaten (`uint8_t`).

#### [NEW] [test_config_cascade.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_config_cascade.cpp)
- Testet automatische Kaskadierung / Modus-Auswahl von SIMD-Implementierungen.

---

### TDD Step 2: Implementierung & Refaktorisierung

#### [NEW] [fp16.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/fp16.h)
- Stellt `deglib::f16` bereit inkl. Hardware-beschleunigter float32 <-> float16 Konvertierung.

#### [NEW] [fp32_l2.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/fp32_l2.h) & [fp32_inner_product.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/fp32_inner_product.h)
- Entkoppelte, hochoptimierte FP32 Distanzfunktionen mit SSE, AVX2 und AVX512 SIMD Intrinsics.

#### [NEW] [fp16_l2.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/fp16_l2.h) & [fp16_inner_product.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/fp16_inner_product.h)
- FP16 Distanzfunktionen mit F16C / AVX2 / AVX512 Unterstüzung.

#### [NEW] [uint8_l2.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/uint8_l2.h)
- UInt8 L2-Distanzfunktionen.

#### [MODIFY] [distances.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distances.h)
- Umstellung auf modular eingebundene Distanz-Header.

---

## Verifikationsplan für Phase 2

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target test_fp16 test_fp32_l2 test_fp32_inner_product test_fp16_inner_product test_uint8_l2 test_config_cascade test_distances
cd bin/Release
test_fp16.exe
test_fp32_l2.exe
test_fp32_inner_product.exe
test_fp16_inner_product.exe
test_uint8_l2.exe
test_config_cascade.exe
test_distances.exe
```
Erfolgs-Kriterium: **100% aller Distanz-Tests bestanden.**
