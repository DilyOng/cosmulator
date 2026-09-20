==========================================================================
cosmulator: emulators of expensive posterior distributions
==========================================================================

:Author: Dily Duan Yi Ong
:License: MIT
:Homepage: https://github.com/DilyOng/cosmulator

Nested sampling produces cosmological posteriors at great expense — thousands
of CPU hours per model–dataset run. ``cosmulator`` learns a normalising flow
from those samples so the posterior can be resampled instantly, and manages
the resulting emulators — training, validation, storage and serving — across a
grid of runs.

Developed under a DiRAC seedcorn allocation.

Installation
------------

.. code:: bash

    pip install cosmulator            # inspect and use emulators (numpy only)
    pip install "cosmulator[train]"   # adds the JAX training stack

Requires Python 3.12 or later.
