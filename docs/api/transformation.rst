Transformation & Quantization
=============================

Feature transformations (MIPS-to-L2), vector quantization (INT8, UINT8, INT4, UINT4, UINT7, FP16, EVP), and transformation pipelines.

Overview
--------

The ``deglib.transformation`` module provides tools for:
- Mapping Maximum Inner Product Search (MIPS) to Euclidean (L2) distance search.
- Fitting and applying stateful vector quantizers to compress feature vectors into INT8, UINT8, or bit-packed EVP formats.
- Chaining multiple transformations into pipelines that can be embedded into read-only graph indices.

Feature Transformations
-----------------------

.. autoclass:: deglib.transformation.FeatureTransformation
   :members:

Transform & Quantizer Factories
-------------------------------

All transformations are created pre-fitted/validated via ``make_*`` factory functions:

.. autofunction:: deglib.transformation.make_mips_transform

.. autofunction:: deglib.transformation.make_l2_normalize_transform

.. autofunction:: deglib.transformation.make_fast_hadamard_rotation

.. autofunction:: deglib.transformation.make_quantizer_pipeline

.. autofunction:: deglib.transformation.make_evp_quantizer

   .. rubric:: EVP Notes

   **Equi-Voronoi Polytopes (EVP)** quantizes high-dimensional float32 or float16 vectors into ternary
   representations :math:`\{-1, 0, +1\}^d`. Each vector is stored in two compact bitmasks
   (each :math:`\lceil d / 8 \rceil` bytes): one mask for :math:`+1` bits, and one for :math:`-1` bits, totaling :math:`2 \times \lceil d / 8 \rceil` bytes.
   Arbitrary dimensions (not divisible by 8) are supported via high-bit zero padding in the tail byte.
   Distances are computed using symmetric bitwise popcount operations (`Metric.EVP_InnerProduct`).

   - **``non_zeros = 0`` (Default / Recommended)**: Enables automatic data-driven calibration on the input vectors.
     The factory samples 2,000 base vectors and 100 queries, computes exact top-100 ground truth,
     and selects the ``non_zeros`` setting maximizing top-100 overlap.
   - **``non_zeros > 0``**: Manually sets the exact number of non-zero bits retained per vector (must be :math:`< d`).

.. autofunction:: deglib.transformation.make_scalar_quantizer_int8

.. autofunction:: deglib.transformation.make_scalar_quantizer_int8_perdim

.. autofunction:: deglib.transformation.make_scalar_quantizer_uint8

.. autofunction:: deglib.transformation.make_scalar_quantizer_uint8_perdim

.. autofunction:: deglib.transformation.make_scalar_quantizer_uint4

.. autofunction:: deglib.transformation.make_scalar_quantizer_int4

.. autofunction:: deglib.transformation.make_scalar_quantizer_uint7

.. autofunction:: deglib.transformation.make_scalar_quantizer_fp16

Pipeline Application
--------------------

.. autofunction:: deglib.transformation.apply_on_database

.. autofunction:: deglib.transformation.apply_on_query

Example Usage
-------------

Scalar INT8 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import deglib
   import numpy as np

   # Generate float32 database vectors
   dim = 128
   data = np.random.randn(10000, dim).astype(np.float32)

   # Fit and quantize with a scalar INT8 quantizer
   quantizer = deglib.transformation.make_scalar_quantizer_int8(data)
   int8_data = quantizer.transform_database(data)

   # Build graph with quantized features (or let deglib.build quantize float32 automatically)
   graph = deglib.build(int8_data, metric=deglib.Metric.Int8_L2)

   # Quantize query vector at search time
   query = np.random.randn(dim).astype(np.float32)
   q_int8 = quantizer.transform_query(query)
   indices, distances = graph.search(q_int8, k=10, eps_or_ef=0.1)

EVP Bit-Packing with Auto-Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import deglib
   import numpy as np

   dim = 512
   data = np.random.randn(10000, dim).astype(np.float32)

   # Auto-calibrates optimal non_zeros directly on data
   quantizer = deglib.transformation.make_evp_quantizer(data)
   print(f"Calibrated optimal non_zeros: {quantizer.non_zeros}")

   # Quantize base data into packed bitmasks (2 * ((dim + 7) // 8) bytes per vector)
   evp_data = quantizer.transform_database(data)

   # Build graph with EVP metric
   graph = deglib.build(evp_data, metric=deglib.Metric.EVP_InnerProduct)


