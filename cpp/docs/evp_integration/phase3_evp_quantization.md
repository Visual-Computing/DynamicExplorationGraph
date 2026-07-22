# Umsetzungsplan Phase 3: EVP Quantisierung & Asymmetrische EVP-Distanzen (TDD)

Dieser Detailplan beschreibt Phase 3: Die Test-getriebene Integration der **Extreme Vector Precision (EVP)** Quantisierung und asymmetrischen Distanzberechnungen.

---

## Betroffene Komponenten & Dateien in Phase 3

### TDD Step 1: Unit-Tests etablieren

#### [NEW] [test_evp_quantize.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/quantization/test_evp_quantize.cpp)
- Überprüft die Quantisierung von FP32-Vektoren in das EVP-Format (Bitrepräsentation, Skalierungsfaktoren, Min/Max Ränge, Decodierungsgenauigkeit).

#### [NEW] [test_evp_inner_product.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_evp_inner_product.cpp)
- Überprüft die mathematische Exaktheit und Korrektheit der EVP Inner Product Distanzen.

#### [NEW] [test_fp16_evp_asym_inner_product.cpp](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/test/src/distance/test_fp16_evp_asym_inner_product.cpp)
- Testet asymmetrische Distanzberechnungen, bei denen ein Vektor in FP16 und der andere als EVP-quantisierter Vektor vorliegt.

---

### TDD Step 2: Implementierung

#### [NEW] [evp_quantize.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/quantization/evp_quantize.h)
- Implementiert die EVP-Quantisierungsroutine (Transformation, Quantisierungs-Bits, Bounded Range Skalierung).

#### [NEW] [evp_inner_product.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/evp_inner_product.h)
- Symmetrische EVP Distanzfunktionen für quantisierte Daten.

#### [NEW] [fp16_evp_asym_inner_product.h](file:///c:/Lang/cpp/DynamicExplorationGraph/cpp/deglib/include/distance/fp16_evp_asym_inner_product.h)
- Asymmetrische FP16 vs. EVP-quantisierte Skalarprodukt-Distanzen inkl. SIMD-Optimierungen.

---

## Verifikationsplan für Phase 3

```cmd
cd c:\Lang\cpp\DynamicExplorationGraph\cpp\build
cmake --build . --config Release --target test_evp_quantize test_evp_inner_product test_fp16_evp_asym_inner_product
cd bin/Release
test_evp_quantize.exe
test_evp_inner_product.exe
test_fp16_evp_asym_inner_product.exe
```
Erfolgs-Kriterium: **Alle Quantisierungs- und EVP-Distanztests grün.**
