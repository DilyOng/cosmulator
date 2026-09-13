==========================================================================
cosmulator: emulators of expensive posterior distributions
==========================================================================

:cosmulator: Train, validate and serve machine-learning emulators of posteriors
:Author: Dily Duan Yi Ong
:Version: 0.0.1
:Homepage: https://github.com/DilyOng/cosmulator
:License: MIT

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/DilyOng/cosmulator/blob/master/LICENSE
   :alt: License information

Nested sampling and MCMC produce posterior samples at great computational
expense: a single cosmological run can consume thousands of CPU hours, and a
systematic study needs one run per model and dataset combination. Once those
samples exist, however, the distribution they describe can be learned by a
normalising flow and resampled instantly thereafter.

``cosmulator`` is the infrastructure around that idea. It is not a normalising
flow implementation; it trains, validates, stores and serves them at scale.

Status
------

Early development. The API is not yet stable and no release has been made.

Design
------

Using an emulator is cheap; training one is not. The package is split along
that line:

- the core install requires only ``numpy`` and ``pyyaml``, so that stored
  emulators can be inspected and the test suite run on any machine;
- the training stack (JAX and friends) lives behind an optional extra and is
  imported lazily, never at module scope.

This is enforced by the test suite rather than by convention.

Installation
------------

.. code:: bash

    pip install cosmulator            # core: inspect and use emulators
    pip install "cosmulator[train]"   # adds the JAX training stack

Requires Python 3.12 or later, a floor set by the training stack (see
``pyproject.toml``).

Contributing
------------

Issues and pull requests are welcome via the `GitHub repository
<https://github.com/DilyOng/cosmulator>`__. See ``.github/CONTRIBUTING.md``.
