=============================================================================================
cosmulator: accelerating cosmological inference with JAX machine-learning emulators on GPU
=============================================================================================

:Author: Dily Duan Yi Ong
:License: MIT
:Homepage: https://github.com/DilyOng/cosmulator
:PyPI: https://pypi.org/project/cosmulator/

.. image:: https://img.shields.io/pypi/v/cosmulator.svg
   :target: https://pypi.org/project/cosmulator/
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/cosmulator.svg
   :target: https://pypi.org/project/cosmulator/
   :alt: Python versions

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/DilyOng/cosmulator/blob/master/LICENSE
   :alt: License: MIT

.. image:: https://img.shields.io/conda/vn/conda-forge/cosmulator.svg
   :target: https://anaconda.org/conda-forge/cosmulator
   :alt: conda-forge

``cosmulator`` trains JAX normalising-flow emulators of nested-sampling
posteriors on GPU, so a distribution can be resampled instantly for fast
marginalisation and Bayesian evidence, and manages the resulting emulators
(training, validation, storage and serving) across a grid of cosmological
models and surveys.

Developed under a UKRI-funded high performance computing (DiRAC) project.

Installation
------------

.. code:: bash

    pip install cosmulator            # inspect and use emulators (numpy only)
    pip install "cosmulator[train]"   # adds the JAX training stack

Requires Python 3.11 or later.
