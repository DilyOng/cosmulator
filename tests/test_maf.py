"""Tests for the MAF emulator trainer that do not need the train extra.

flowjax/JAX are imported lazily inside ``train_maf_emulator``, so the module
imports and its weighted standardiser can be exercised without the training stack.
The full GPU training path is covered separately under the ``[train]`` extra.
"""

import importlib.util

import numpy as np
import pytest

from cosmulator import maf as maf_module
from cosmulator.maf import (
    _from_unbounded,
    _railing_bounds,
    _to_unbounded,
    _weight_stratified_split,
    _weighted_cov,
    _weighted_quantile,
    _weighted_standardiser,
    train_maf_emulator,
)


class TestBijector:
    # two-sided, lower-only, upper-only, unbounded
    BOUNDS = np.array([[0.0, 100.0], [0.0, np.inf], [-np.inf, 1.0],
                       [-np.inf, np.inf]])

    def test_roundtrip_is_exact(self):
        rng = np.random.default_rng(0)
        x = np.column_stack([rng.uniform(1, 99, 500), rng.uniform(0.1, 5, 500),
                             rng.uniform(-5, 0.9, 500), rng.normal(size=500)])
        y = _to_unbounded(x, self.BOUNDS)
        assert np.isfinite(y).all()
        assert np.allclose(_from_unbounded(y, self.BOUNDS), x, atol=1e-9)

    def test_inverse_always_lands_in_bounds(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 5, size=(2000, 4))
        x = _from_unbounded(y, self.BOUNDS)
        assert x[:, 0].min() > 0 and x[:, 0].max() < 100   # two-sided
        assert x[:, 1].min() > 0                            # lower bound
        assert x[:, 2].max() < 1                            # upper bound


class TestSelectiveRailing:
    def test_weighted_quantile_matches_unweighted_median(self):
        x = np.linspace(0, 10, 1001)
        w = np.ones_like(x)
        assert _weighted_quantile(x, w, 0.5) == pytest.approx(5.0, abs=0.05)

    def test_only_railing_side_is_kept(self):
        rng = np.random.default_rng(0)
        # col 0 rails against lower wall 0 (exponential pile-up), col 1 sits in the
        # interior of [0, 100] far from either wall.
        rail = rng.exponential(0.3, size=20000)
        interior = 50 + rng.normal(0, 3, size=20000)
        theta = np.column_stack([rail, interior])
        w = np.full(len(theta), 1.0 / len(theta))
        bounds = np.array([[0.0, 5.0], [0.0, 100.0]])
        eff, railing = _railing_bounds(theta, w, bounds, margin_sigma=1.0)
        assert railing == [0]                       # only col 0 flagged
        assert np.isfinite(eff[0, 0]) and eff[0, 0] == 0.0   # lower wall kept
        assert not np.isfinite(eff[0, 1])           # far upper wall dropped
        assert not np.isfinite(eff[1]).any()        # interior column untouched


class TestWhitening:
    def test_cov_diagonal_matches_standardiser_variance(self):
        rng = np.random.default_rng(0)
        theta = rng.normal(size=(40000, 3)) * [2.0, 0.5, 5.0] + [1.0, -3.0, 10.0]
        w = np.full(len(theta), 1.0 / len(theta))
        mean, std = _weighted_standardiser(theta, w)
        cov = _weighted_cov(theta, w, mean)
        assert np.allclose(np.sqrt(np.diag(cov)), std, rtol=1e-6)

    def test_cholesky_whitening_gives_identity_covariance(self):
        rng = np.random.default_rng(1)
        # correlated Gaussian: whitening should decorrelate to identity cov.
        A = np.array([[3.0, 0.0], [2.0, 1.0]])
        theta = rng.normal(size=(60000, 2)) @ A.T + [5.0, -2.0]
        w = np.full(len(theta), 1.0 / len(theta))
        mean = w @ theta
        cov = _weighted_cov(theta, w, mean)
        L = np.linalg.cholesky(cov)
        z = (theta - mean) @ np.linalg.inv(L).T
        zc = _weighted_cov(z, w, w @ z)
        assert np.allclose(zc, np.eye(2), atol=0.02)


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


class TestWeightStratifiedSplit:
    def _skewed(self, n=5000, seed=0):
        rng = np.random.default_rng(seed)
        return np.abs(rng.normal(size=n)) + 1e-3

    def test_fractions_and_partition(self):
        w = self._skewed()
        tr, va, te = _weight_stratified_split(w, 0.1, 0.15)
        n = len(w)
        assert len(te) / n == pytest.approx(0.15, abs=0.01)
        assert len(va) / n == pytest.approx(0.10, abs=0.01)
        # disjoint and complete
        assert len(set(tr) | set(va) | set(te)) == n
        assert len(tr) + len(va) + len(te) == n

    def test_splits_are_weight_representative(self):
        # Each split's mean weight should track the overall mean, unlike a random
        # split that could hand one split most of the mass.
        w = self._skewed()
        tr, va, te = _weight_stratified_split(w, 0.1, 0.15)
        for idx in (tr, va, te):
            assert w[idx].mean() == pytest.approx(w.mean(), rel=0.1)

    def test_no_test_split_when_zero(self):
        w = self._skewed()
        tr, va, te = _weight_stratified_split(w, 0.1, 0.0)
        assert te.size == 0
        assert len(tr) + len(va) == len(w)


def test_training_without_extra_raises_importerror():
    if importlib.util.find_spec("flowjax") is not None:
        pytest.skip("flowjax installed; cannot exercise the missing-extra path")
    with pytest.raises(ImportError):
        train_maf_emulator(_FakeSamples([1.0, 1.0]))
