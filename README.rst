=============================================================================================
cosmulator: accelerating cosmological inference with JAX machine-learning emulators on GPU
=============================================================================================

:Author: Dily Duan Yi Ong
:License: MIT
:Homepage: https://github.com/DilyOng/cosmulator

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
