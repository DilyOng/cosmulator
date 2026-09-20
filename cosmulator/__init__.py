"""cosmulator: emulators of expensive posterior distributions.

Nested sampling and MCMC produce posterior samples at great computational
expense. ``cosmulator`` trains normalising-flow emulators of those posteriors so
that the distribution can be resampled instantly afterwards, and manages the
resulting emulators as durable, shareable artefacts.

The package is deliberately split so that using an emulator is cheap while
training one is not: the training stack is an optional extra, and nothing in the
core import path requires it.
"""

from cosmulator._version import __version__
from cosmulator.diagnostics import (
    effective_sample_size,
    forward_kl,
    moment_error,
    self_consistency,
)
from cosmulator.fetch import EmulatorArchive, fetch
from cosmulator.manifest import (
    Hyperparameters,
    Manifest,
    Provenance,
    Quality,
    Training,
)
from cosmulator.parameters import (
    BASE,
    COSMO_SAMPLED,
    cosmological_parameters,
    parameter_names,
)
from cosmulator.store import Emulator, EmulatorStore
from cosmulator.validation import (
    certify,
    marginal_gof,
    mmd,
    out_of_bounds,
)

__all__ = [
    "__version__",
    "BASE",
    "COSMO_SAMPLED",
    "Emulator",
    "EmulatorArchive",
    "EmulatorStore",
    "Hyperparameters",
    "Manifest",
    "Provenance",
    "Quality",
    "Training",
    "certify",
    "cosmological_parameters",
    "effective_sample_size",
    "fetch",
    "forward_kl",
    "marginal_gof",
    "mmd",
    "moment_error",
    "out_of_bounds",
    "parameter_names",
    "self_consistency",
]
