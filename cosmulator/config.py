"""Where cosmulator looks for things on the current machine.

Paths differ between a laptop, a shared cluster and a continuous-integration
runner, and a library that hardcodes any of them can only ever run in one
place. Every location is therefore resolved here, from the environment, with a
documented default.

No credential is read here or anywhere else in the package. Tokens are passed
explicitly to the one class that needs them; see :mod:`cosmulator.publish`.
"""

import os
from pathlib import Path

#: Environment variable naming the local emulator store.
STORE_ENV = "COSMULATOR_STORE"

#: Environment variable naming the download cache.
CACHE_ENV = "COSMULATOR_CACHE"

#: Environment variable naming the root of a local chain grid.
CHAINS_ENV = "COSMULATOR_CHAINS"


def store_root():
    """Return the directory holding locally trained emulators.

    Returns
    -------
    pathlib.Path
        ``$COSMULATOR_STORE`` if set, otherwise ``~/.cosmulator/store``.
    """
    return Path(os.environ.get(STORE_ENV, Path.home() / ".cosmulator" / "store"))


def cache_root():
    """Return the directory downloaded emulators are cached in.

    Downloads are cached so that repeated use of an emulator costs one network
    round trip rather than one per call.

    Returns
    -------
    pathlib.Path
        ``$COSMULATOR_CACHE`` if set, otherwise ``~/.cosmulator/cache``.
    """
    return Path(os.environ.get(CACHE_ENV, Path.home() / ".cosmulator" / "cache"))


def chains_root():
    """Return the root of a local grid of posterior chains, if configured.

    Only training needs this; consuming a published emulator does not, which is
    why there is no default. A machine that has never trained anything should
    not be asked to invent a plausible path.

    Returns
    -------
    pathlib.Path or None
        ``$COSMULATOR_CHAINS`` if set, otherwise None.
    """
    value = os.environ.get(CHAINS_ENV)
    return Path(value) if value else None
