Installation
============

Via pip
-------
Install from `PyPI <https://pypi.org/project/deglib/>`_. This is best for most users.

.. code-block:: sh

    pip install deglib

From Source
-----------
You can also compile `deglib` yourself.

This can be useful if you want to:

* make use of AVX512 instructions
* have the newest version
* develop new features

.. code-block:: sh

    # get the source
    git clone https://github.com/Visual-Computing/DynamicExplorationGraph.git
    cd DynamicExplorationGraph/python/

    # create virtualenv
    python -m venv venv && . venv/bin/activate

    # install build dependencies
    pip install setuptools>=77.0 pybind11 build
    python setup.py copy_build_files  # copy c++ library to ./lib/

    # install
    pip install .
