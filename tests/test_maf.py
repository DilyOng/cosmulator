"""Tests for the MAF emulator trainer that do not need the train extra.

flowjax/JAX are imported lazily inside ``train_maf_emulator``, so the module
imports and its weighted standardiser can be exercised without the training stack.
The full GPU training path is covered separately under the ``[train]`` extra.
"""

import importlib.util

import numpy as np
import pytest

from cosmulator import maf as maf_module
from cosmulator.maf import _weighted_standardiser, train_maf_emulator


class _FakeSamples:
    def __init__(self, weights):
        self._weights = np.asarray(weights, dtype=float)

    def get_weights(self):
        return self._weights


def test_module_imports_without_flowjax():
    assert hasattr(maf_module, "train_maf_emulator")


class TestWeightedStandardiser:
    def test_recovers_weighted_mean_and_std(self):
        rng = np.random.default_rng(0)
        theta = rng.normal(size=(50000, 2)) * [2.0, 0.5] + [1.0, -3.0]
        w = np.full(len(theta), 1.0 / len(theta))
        mean, std = _weighted_standardiser(theta, w)
        assert np.allclose(mean, [1.0, -3.0], atol=0.05)
        assert np.allclose(std, [2.0, 0.5], atol=0.05)

    def test_standardising_gives_unit_variance(self):
        rng = np.random.default_rng(1)
        theta = rng.normal(size=(40000, 3)) * [5.0, 0.1, 20.0]
        w = np.full(len(theta), 1.0 / len(theta))
        mean, std = _weighted_standardiser(theta, w)
        z = (theta - mean) / std
        assert np.allclose(z.std(axis=0), 1.0, atol=0.02)

    def test_weights_are_honoured(self):
        # Two clusters; up-weighting one shifts the weighted mean toward it.
        theta = np.vstack([np.zeros((1000, 1)), 10 * np.ones((1000, 1))])
        w = np.concatenate([np.full(1000, 9.0), np.full(1000, 1.0)])
        w = w / w.sum()
        mean, _ = _weighted_standardiser(theta, w)
        assert mean[0] == pytest.approx(1.0, abs=0.01)  # 0.9*0 + 0.1*10


def test_training_without_extra_raises_importerror():
    if importlib.util.find_spec("flowjax") is not None:
        pytest.skip("flowjax installed; cannot exercise the missing-extra path")
    with pytest.raises(ImportError):
        train_maf_emulator(_FakeSamples([1.0, 1.0]))
