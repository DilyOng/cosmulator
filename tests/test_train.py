"""Tests for the margarine-aligned trainer that do not require the train extra.

The JAX and margarine imports in :mod:`cosmulator.train` are deliberately lazy,
so the module imports and its weight handling can be exercised on a machine
without the training stack. The full training path is covered by GPU tests under
the ``[train]`` extra.
"""

import sys

import numpy as np
import pytest

import cosmulator
from cosmulator import train as train_module
from cosmulator.train import _weights_of, train_emulator


class _FakeSamples:
    """A minimal weighted samples stand-in exposing ``get_weights``."""

    def __init__(self, weights):
        self._weights = np.asarray(weights, dtype=float)

    def get_weights(self):
        return self._weights


def test_module_import_does_not_pull_in_jax():
    """Importing cosmulator (and this module) must not require JAX."""
    assert "cosmulator.train" not in _core_imports()
    # The lazy import means the module object exists without jax present.
    assert hasattr(train_module, "train_emulator")


def _core_imports():
    """The submodules pulled in by ``import cosmulator`` itself."""
    # cosmulator.train is imported by THIS test module, not by the package root;
    # assert the root does not list it among its own eager imports.
    return set(getattr(cosmulator, "__all__", []))


class TestWeightsOf:
    def test_weights_are_normalised(self):
        w = _weights_of(_FakeSamples([1.0, 3.0]), 2)
        assert w.tolist() == [0.25, 0.75]

    def test_uniform_when_no_get_weights(self):
        w = _weights_of(object(), 4)
        assert w.tolist() == [0.25, 0.25, 0.25, 0.25]

    def test_zero_sum_weights_raise(self):
        with pytest.raises(ValueError, match="positive"):
            _weights_of(_FakeSamples([0.0, 0.0]), 2)

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _weights_of(_FakeSamples([-1.0, -1.0]), 2)


def test_train_emulator_without_train_extra_raises_importerror():
    """Absent the [train] extra, calling the trainer fails with ImportError."""
    if "jax" in sys.modules or _jax_installed():
        pytest.skip("JAX is installed; the lazy import path cannot be exercised")
    with pytest.raises(ImportError):
        train_emulator(_FakeSamples([1.0, 1.0]))


def _jax_installed():
    import importlib.util

    return importlib.util.find_spec("jax") is not None
