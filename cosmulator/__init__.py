"""cosmulator: emulators of expensive posterior distributions.

Nested sampling and MCMC produce posterior samples at great computational
expense. ``cosmulator`` trains normalising-flow emulators of those posteriors so
that the distribution can be resampled instantly afterwards, and manages the
resulting emulators as durable, shareable artefacts.

The package is deliberately split so that using an emulator is cheap while
training one is not: the training stack is an optional extra, and nothing in the
core import path requires it.
"""

from cosmulator._version import __version__  # noqa: F401
