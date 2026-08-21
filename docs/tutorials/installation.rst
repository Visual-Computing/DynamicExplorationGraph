Installation
============

Requirements
------------

- **Python**: 3.10 or newer
- **Operating Systems**: Linux (x86_64), Windows (x86_64), macOS (Apple Silicon / ARM64)
- **Dependencies**: `numpy`

Standard Installation
---------------------

The easiest and recommended way to install `deglib` is from `PyPI <https://pypi.org/project/deglib/>`_:

Using pip:

.. code-block:: bash

   pip install deglib

Using uv:

.. code-block:: bash

   uv add deglib

Building from Source
--------------------

If you want the latest development version or need to build directly on your machine:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Visual-Computing/DynamicExplorationGraph.git
      cd DynamicExplorationGraph/python/

2. Install pinned build dependencies and copy required files:

   .. code-block:: bash

      pip install setuptools==83.0.0 pybind11==3.0.4 build==1.5.0 wheel==0.48.0
      python setup.py copy_build_files

   Or using ``uv``:

   .. code-block:: bash

      uv pip install setuptools==83.0.0 pybind11==3.0.4 build==1.5.0 wheel==0.48.0
      uv run python setup.py copy_build_files


3. Build and install:

   .. code-block:: bash

      pip install -e . --no-build-isolation

   Or with ``uv``:

   .. code-block:: bash

      uv pip install -e . --no-build-isolation

Verifying Installation
----------------------

Verify that `deglib` is installed correctly and check the active version:

.. code-block:: python

   import deglib
   print(f"deglib version: {deglib.__version__}")
   print(f"AVX2 support: {deglib.cpu.has_avx2()}")


