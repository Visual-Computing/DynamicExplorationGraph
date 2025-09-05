Python Installation
===================

Via pip
-------
Install from `pypi <https://pypi.org/project/deglib/>`_.

.. code-block:: sh

    pip install deglib

From Source
-----------
.. code-block:: sh

    # get the source
    git clone https://github.com/Visual-Computing/DynamicExplorationGraph.git
    cd DynamicExplorationGraph/python/

    # create venv
    python -m venv venv && . venv/bin/activate

    # install build dependencies
    pip install setuptools>=77.0 pybind11 build
    python setup.py copy_build_files  # copy c++ library to ./lib/

    # install
    pip install .
