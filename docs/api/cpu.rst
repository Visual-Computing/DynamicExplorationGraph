CPU & Hardware Acceleration
============================

Hardware capability checks and instruction set configuration for vector calculations.

Overview
--------

``deglib`` automatically optimizes distance computations based on available CPU extensions.
The :class:`~deglib.cpu.InstructionSet` enum allows specifying or inspecting the instruction set
used by spaces during distance calculations.

By default, ``InstructionSet.Auto`` detects the best available instruction set on the host CPU.

InstructionSet
--------------

.. autoclass:: deglib.cpu.InstructionSet
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Hardware Detection
------------------

.. autofunction:: deglib.cpu.has_avx2

.. autofunction:: deglib.cpu.has_avx512

Example Usage
-------------

.. code-block:: python

   import deglib

   # Check host CPU capabilities
   print("AVX2 supported:", deglib.cpu.has_avx2())
   print("AVX-512 supported:", deglib.cpu.has_avx512())

   # Create a FloatSpace with automatic CPU instruction set selection
   space = deglib.FloatSpace.create(
       dim=128,
       metric=deglib.Metric.FP32_L2,
       instruction=deglib.cpu.InstructionSet.Auto,
   )
   print("Active instruction set:", space.get_instruction())

