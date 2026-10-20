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
