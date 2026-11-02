Transformation & Quantization
=============================

Feature transformations (MIPS-to-L2), vector quantization (INT8, UINT8, EVP), and transformation pipelines.

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

.. autoclass:: deglib.transformation.MipsTransform
   :members:

.. autofunction:: deglib.transformation.mips_l2_transform

.. autofunction:: deglib.transformation.mips_l2_transform_query

Quantizers
----------

.. autoclass:: deglib.transformation.EvpQuantizer
   :members:

   .. rubric:: Notes

   **Equi-Voronoi Polytopes (EVP)** quantizes high-dimensional float32 or float16 vectors into ternary
   representations :math:`\{-1, 0, +1\}^d`. Each vector is stored in two compact bitmasks
   (each :math:`\lceil d / 8 \rceil` bytes): one mask for :math:`+1` bits, and one for :math:`-1` bits, totaling :math:`2 \times \lceil d / 8 \rceil` bytes.
   Arbitrary dimensions (not divisible by 8) are supported via high-bit zero padding in the tail byte.
   Distances are computed using symmetric bitwise popcount operations (`Metric.EVP_InnerProduct`).

   **Non-Zeros Parameter & Auto-Calibration:**
   The parameter ``non_zeros`` determines how many extreme dimensions (highest magnitude after mean-centering)
   are retained as :math:`\pm 1` bits per vector, with the remaining set to zero.

   - **``non_zeros = 0`` (Default / Recommended)**: Enables automatic data-driven calibration during ``fit()``.
     The quantizer samples 2,000 base vectors and 100 queries from the training data, computes exact top-100 ground truth,
     and evaluates candidate sparsity levels across both logarithmic and linear scales. It selects the ``non_zeros``
     setting that maximizes top-100 overlap (typically under 2 seconds).
     This automatically adapts to both sparse activation spaces (such as CNN ReLU features where only 1-5% non-zeros is optimal)
     and dense embedding spaces (such as Transformer embeddings where ~50% non-zeros is optimal).
   - **``non_zeros > 0``**: Manually sets the exact number of non-zero bits retained per vector (must be :math:`< d`).

.. autoclass:: deglib.transformation.ScalarQuantizerInt8
   :members:

.. autoclass:: deglib.transformation.ScalarQuantizerInt8PerDim
   :members:

.. autoclass:: deglib.transformation.ScalarQuantizerUint8
   :members:

.. autoclass:: deglib.transformation.ScalarQuantizerUint8PerDim
   :members:

Quantizer Factories
-------------------

.. autofunction:: deglib.transformation.make_evp_quantizer

.. autofunction:: deglib.transformation.make_scalar_quantizer_int8

.. autofunction:: deglib.transformation.make_scalar_quantizer_int8_perdim

.. autofunction:: deglib.transformation.make_scalar_quantizer_uint8

.. autofunction:: deglib.transformation.make_scalar_quantizer_uint8_perdim

Pipeline Serialization & Application
------------------------------------

.. autofunction:: deglib.transformation.serialize_transformations

.. autofunction:: deglib.transformation.deserialize_transformations

.. autofunction:: deglib.transformation.apply_database

.. autofunction:: deglib.transformation.apply_query

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

   # Fit a scalar INT8 quantizer
   quantizer = deglib.transformation.make_scalar_quantizer_int8(data)
   int8_data = quantizer.quantize(data)

   # Build graph with quantized features
   graph = deglib.builder.build_from_data(int8_data, metric=deglib.Metric.Int8_L2)

   # Quantize query vector at search time
   query = np.random.randn(dim).astype(np.float32)
   q_int8 = quantizer.quantize(query)
   indices, distances = graph.search(q_int8, k=10, eps_or_ef=0.1)

EVP Bit-Packing with Auto-Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import deglib
   import numpy as np

   dim = 512
   data = np.random.randn(10000, dim).astype(np.float32)

   # dim=dim, non_zeros=0 enables data-driven auto-calibration on fit()
   quantizer = deglib.transformation.make_evp_quantizer(dim=dim, non_zeros=0)
   quantizer.fit(data)
   print(f"Calibrated optimal non_zeros: {quantizer.non_zeros}")

   # Quantize base data into packed bitmasks (2 * ((dim + 7) // 8) bytes per vector)
   evp_data = quantizer.quantize(data)

   # Build graph with EVP metric
   graph = deglib.builder.build_from_data(evp_data, metric=deglib.Metric.EVP_InnerProduct)


