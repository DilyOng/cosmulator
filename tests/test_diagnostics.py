"""Tests for the emulator fidelity diagnostics.

Where possible these use distributions whose answer is known analytically, so
that a metric is checked against truth rather than against its own past output.
"""

import numpy as np
import pytest

from cosmulator.diagnostics import (
    effective_sample_size,
    forward_kl,
    moment_error,
    self_consistency,
)


class TestSelfConsistency:
    """dlogp vanishes for a perfect emulator and is signed for a poor one."""

    def test_identical_densities_give_zero(self):
        """When the two expectations agree, the metric is exactly zero."""
        logq = np.full(1000, -3.2)
        result = self_consistency(logq, logq)
        assert result["dlogp"] == pytest.approx(0.0)
        assert result["abs_dlogp"] == pytest.approx(0.0)

    def test_over_dispersion_is_negative(self):
        """An emulator spread too widely scores below zero.

        This is the sign convention that matters: over-dispersion is the
        failure mode that makes an emulator look harmlessly conservative.
        """
        at_target = np.full(500, -2.0)
        at_emulated = np.full(500, -3.0)
        assert self_consistency(at_target, at_emulated)["dlogp"] < 0

    def test_under_dispersion_is_positive(self):
        """An emulator that is too sharp scores above zero."""
        at_target = np.full(500, -3.0)
        at_emulated = np.full(500, -2.0)
        assert self_consistency(at_target, at_emulated)["dlogp"] > 0

    def test_weights_are_honoured(self):
        """Weighting the target changes the answer, as it must for NS chains."""
        at_target = np.array([-1.0, -5.0])
        at_emulated = np.array([-3.0, -3.0])
        uniform = self_consistency(at_target, at_emulated)["dlogp"]
        skewed = self_consistency(at_target, at_emulated, weights=[0.99, 0.01])
        assert uniform != pytest.approx(skewed["dlogp"])

    def test_infinities_are_dropped_not_propagated(self):
        """A flow may legitimately return -inf far into the tails."""
        at_target = np.array([-2.0, -np.inf, -2.0])
        result = self_consistency(at_target, np.full(3, -2.0))
        assert np.isfinite(result["dlogp"])
        assert result["dlogp"] == pytest.approx(0.0)

    def test_all_infinite_target_raises(self):
        """If nothing is finite the metric is undefined and says so."""
        with pytest.raises(ValueError, match="no finite log densities"):
            self_consistency(np.full(3, -np.inf), np.full(3, -1.0))

    def test_expectations_are_reported_for_diagnosis(self):
        """The two halves are returned so a surprise can be traced."""
        result = self_consistency(np.full(10, -2.0), np.full(10, -3.0))
        assert result["E_P_logq"] == pytest.approx(-2.0)
        assert result["E_q_logq"] == pytest.approx(-3.0)


class TestForwardKL:
    """D_KL(P||q): a proper divergence that over-dispersion cannot flatter."""

    @staticmethod
    def _gaussian_logpdf(x, scale=1.0):
        """Log density of an isotropic zero-mean Gaussian at ``x``."""
        d = x.shape[1]
        return -0.5 * np.sum(x ** 2, axis=1) / scale ** 2 - 0.5 * d * np.log(
            2.0 * np.pi * scale ** 2
        )

    def test_perfect_emulator_is_near_zero(self):
        """A flow equal to the target has zero divergence up to sampling noise."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20000, 3))
        result = forward_kl(self._gaussian_logpdf(x), x)
        assert result["forward_kl"] == pytest.approx(0.0, abs=0.05)

    def test_over_dispersion_is_positive(self):
        """Where self-consistency goes negative, forward KL stays positive.

        This is the whole point: over-dispersion is a real error, and a proper
        divergence must report it as such rather than be lowered by it.
        """
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50000, 2))
        over_broad = self._gaussian_logpdf(x, scale=1.5)
        assert forward_kl(over_broad, x)["forward_kl"] > 0

    def test_matches_the_gaussian_analytic_value(self):
        """For known Gaussians the answer is checked against the closed form.

        D_KL(N(0,I) || N(0,s^2 I)) = d/2 (1/s^2 - 1) + d ln s.
        """
        rng = np.random.default_rng(2)
        d, s = 2, 1.5
        x = rng.normal(size=(100000, d))
        analytic = 0.5 * d * (1.0 / s ** 2 - 1.0) + d * np.log(s)
        result = forward_kl(self._gaussian_logpdf(x, scale=s), x)
        assert result["forward_kl"] == pytest.approx(analytic, abs=0.02)

    def test_missing_mass_is_penalised_not_dropped(self):
        """A flow that assigns zero density to real posterior mass scores worse.

        Dropping the -inf points, as the symmetric self-consistency does, would
        reward a mode-collapsed flow; here it must be penalised instead.
        """
        rng = np.random.default_rng(3)
        x = rng.normal(size=(1000, 2))
        logq = self._gaussian_logpdf(x)
        good = forward_kl(logq, x)["forward_kl"]
        collapsed = logq.copy()
        collapsed[:100] = -np.inf  # emulator misses 10% of the mass
        bad = forward_kl(collapsed, x)["forward_kl"]
        assert np.isfinite(bad)
        assert bad > good + 100

    def test_length_mismatch_is_reported(self):
        """Densities and samples must describe the same points."""
        with pytest.raises(ValueError, match="expected"):
            forward_kl(np.zeros(5), np.zeros((10, 2)))


class TestMomentError:
    """Per-parameter bias and width, against a known Gaussian."""

    def test_a_perfect_emulator_scores_zero(self):
        """Identical sample sets give no bias and no width error."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20000, 3))
        result = moment_error(x, x)
        assert result["mean_abs_bias_sigma"] == pytest.approx(0.0, abs=1e-12)
        assert result["mean_abs_width_err_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_a_known_shift_is_recovered_in_sigma_units(self):
        """Shifting by 0.5 sigma is reported as 0.5."""
        rng = np.random.default_rng(1)
        target = rng.normal(loc=0.0, scale=2.0, size=(50000, 1))
        emulated = target + 1.0  # 1.0 == 0.5 sigma when sigma == 2
        result = moment_error(target, emulated)
        assert result["mean_abs_bias_sigma"] == pytest.approx(0.5, abs=0.02)

    def test_a_known_over_dispersion_is_recovered_as_a_percentage(self):
        """A flow 20% too wide reports +20%."""
        rng = np.random.default_rng(2)
        target = rng.normal(size=(50000, 1))
        result = moment_error(target, target * 1.2)
        assert result["width_error_pct"][0] == pytest.approx(20.0, abs=0.5)

    def test_results_are_reported_per_parameter(self):
        """A single mean hides which parameter is wrong."""
        rng = np.random.default_rng(3)
        target = rng.normal(size=(20000, 3))
        emulated = target.copy()
        emulated[:, 1] *= 1.5
        result = moment_error(target, emulated)
        assert len(result["width_error_pct"]) == 3
        assert abs(result["width_error_pct"][1]) > abs(result["width_error_pct"][0])

    def test_sample_counts_need_not_match(self):
        """An emulator can be sampled as often as you like."""
        rng = np.random.default_rng(4)
        target = rng.normal(size=(1000, 2))
        emulated = rng.normal(size=(7777, 2))
        assert "mean_abs_bias_sigma" in moment_error(target, emulated)

    def test_dimension_mismatch_is_reported(self):
        """Comparing different parameter counts is a bug, not a score."""
        with pytest.raises(ValueError, match="dimension mismatch"):
            moment_error(np.zeros((10, 3)), np.zeros((10, 2)))

    def test_zero_spread_target_is_reported(self):
        """A constant parameter cannot normalise a bias."""
        with pytest.raises(ValueError, match="zero spread"):
            moment_error(np.ones((10, 1)), np.ones((10, 1)))


class TestEffectiveSampleSize:
    """Kish's ESS, the number that decides if a chain is worth training on."""

    def test_uniform_weights_give_the_full_count(self):
        """Equal weights mean every sample counts."""
        assert effective_sample_size(np.ones(500)) == pytest.approx(500.0)

    def test_scale_invariance(self):
        """ESS depends on the shape of the weights, not their normalisation."""
        w = np.array([3.0, 1.0, 1.0, 5.0])
        assert effective_sample_size(w) == pytest.approx(effective_sample_size(w * 1e6))

    def test_one_dominant_weight_collapses_to_one(self):
        """A chain dominated by a single point carries one sample of value."""
        w = np.array([1.0] + [1e-12] * 999)
        assert effective_sample_size(w) == pytest.approx(1.0, abs=0.01)

    def test_zero_weights_are_ignored(self):
        """NS chains are full of exactly-zero-weight prior-tail points."""
        assert effective_sample_size([1.0, 1.0, 0.0, 0.0]) == pytest.approx(2.0)

    def test_empty_or_all_zero_is_zero_not_nan(self):
        """A degenerate chain reports no effective samples rather than NaN."""
        assert effective_sample_size([]) == 0.0
        assert effective_sample_size([0.0, 0.0]) == 0.0


def test_diagnostics_do_not_require_jax():
    """Scoring an emulator must work without the training stack."""
    import sys

    assert "jax" not in sys.modules
    assert "margarine" not in sys.modules
