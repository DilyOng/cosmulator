"""Tests for post-hoc affine recalibration of emulator samples."""

import numpy as np
import pytest

from cosmulator import moment_error, recalibrate


class TestRecalibrate:
    def test_over_dispersion_is_removed(self):
        """A uniformly 10% too-broad emulator becomes exact after recalibration."""
        rng = np.random.default_rng(0)
        target = rng.normal(size=(20000, 3)) * [1.0, 2.0, 0.5]
        emulated = target * 1.10  # 10% over-dispersed
        before = moment_error(target, emulated)["mean_abs_width_err_pct"]
        fixed = recalibrate(emulated, target)
        after = moment_error(target, fixed)["mean_abs_width_err_pct"]
        assert before > 9.0
        assert after < 0.5

    def test_bias_is_removed(self):
        """A shifted emulator is re-centred on the target."""
        rng = np.random.default_rng(1)
        target = rng.normal(size=(20000, 2))
        emulated = target + [3.0, -1.0]
        fixed = recalibrate(emulated, target)
        assert np.allclose(fixed.mean(axis=0), target.mean(axis=0), atol=0.05)

    def test_correlations_are_matched(self):
        """The corrected covariance matches the target's, off-diagonals included."""
        rng = np.random.default_rng(2)
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        target = rng.multivariate_normal([0, 0], cov, size=40000)
        emulated = rng.normal(size=(40000, 2)) * 1.3  # broad and uncorrelated
        fixed = recalibrate(emulated, target)
        assert np.allclose(np.cov(fixed, rowvar=False), cov, atol=0.05)

    def test_target_weights_are_honoured(self):
        """Recalibration matches the WEIGHTED target moments, not the raw ones."""
        rng = np.random.default_rng(3)
        target = rng.normal(size=(20000, 2))
        # Up-weight a shifted, broadened subpopulation so weighted moments differ.
        target[:5000] = target[:5000] * 2.0 + 5.0
        weights = np.ones(20000)
        weights[:5000] = 6.0
        w = weights / weights.sum()
        mu_w = w @ target
        emulated = rng.normal(size=(20000, 2))
        fixed = recalibrate(emulated, target, target_weights=weights)
        assert np.allclose(fixed.mean(axis=0), mu_w, atol=0.1)

    def test_dimension_mismatch_is_reported(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            recalibrate(np.zeros((10, 3)), np.zeros((10, 2)))

    def test_near_degenerate_target_still_factorises(self):
        """A near-singular target covariance (w0-wa-like) does not crash."""
        rng = np.random.default_rng(4)
        base = rng.normal(size=(20000, 1))
        target = np.hstack([base, base + 1e-4 * rng.normal(size=(20000, 1))])
        emulated = rng.normal(size=(20000, 2))
        fixed = recalibrate(emulated, target)
        assert np.all(np.isfinite(fixed))
