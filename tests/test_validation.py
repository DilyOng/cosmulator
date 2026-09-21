"""Tests for the validation suite -- i.e. tests that test the tester.

A validator is only trustworthy if it PASSES cases known to be good and FAILS
cases known to be bad, so most of these feed distributions with a known
ground-truth relationship and assert the verdict. Pure NumPy, no GPU, no flowjax.
"""

import numpy as np
import pytest

from cosmulator.validation import (
    certify,
    marginal_gof,
    mmd,
    out_of_bounds,
)


def _gaussian(n, d, seed, mean=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, scale, size=(n, d))


class TestOutOfBounds:
    def test_exact_fraction(self):
        samples = np.zeros((100, 1))
        samples[:10] = -1.0  # ten below the lower bound
        bounds = np.array([[0.0, np.inf]])
        assert out_of_bounds(samples, bounds) == pytest.approx(0.10)

    def test_all_inside_is_zero(self):
        samples = _gaussian(500, 2, 0)
        bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        assert out_of_bounds(samples, bounds) == 0.0


class TestMarginalGof:
    def test_identical_is_near_zero(self):
        a = _gaussian(4000, 3, 1)
        b = _gaussian(4000, 3, 2)
        res = marginal_gof(a, b)
        assert res["max_wasserstein_sigma"] < 0.05

    def test_shift_is_detected(self):
        a = _gaussian(4000, 1, 1)
        b = _gaussian(4000, 1, 2, mean=1.0)  # one sigma shift
        res = marginal_gof(a, b)
        assert res["max_wasserstein_sigma"] > 0.5

    def test_is_weight_aware(self):
        # Target is bimodal, but almost all weight sits on the mode near 0.
        pts = np.concatenate([np.zeros(500), 10.0 * np.ones(500)])[:, None]
        w = np.concatenate([np.full(500, 0.99 / 500), np.full(500, 0.01 / 500)])
        emulated = _gaussian(2000, 1, 3, mean=0.0, scale=0.05)
        # If weights are honoured the target marginal is concentrated near 0, so
        # the emulator near 0 matches well. If weights were ignored, the target
        # would look bimodal and the distance would be large.
        res = marginal_gof(pts, emulated, weights=w)
        assert res["max_wasserstein_sigma"] < 0.2


class TestMmd:
    def test_identical_not_significant(self):
        a = _gaussian(3000, 3, 10)
        b = _gaussian(3000, 3, 11)
        res = mmd(a, b, seed=0)
        assert res["pvalue"] > 0.05

    def test_covariance_difference_detected(self):
        # Same 1D marginals, different correlation: MMD must catch what moments
        # and per-margin tests cannot.
        rng = np.random.default_rng(20)
        n = 4000
        z = rng.normal(size=(n, 2))
        rho = 0.95
        corr = np.column_stack([z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]])
        decorr = corr.copy()
        decorr[:, 1] = rng.permutation(decorr[:, 1])  # destroy correlation only
        res = mmd(corr, decorr, seed=0)
        assert res["pvalue"] < 0.01

    def test_weighted_target_has_no_pvalue(self):
        # No calibrated null exists yet for a weighted target vs unweighted
        # emulator, so the statistic is reported but the p-value is withheld.
        rng = np.random.default_rng(5)
        target = rng.normal(size=(3000, 2))
        w = np.abs(rng.normal(size=3000)) + 0.01  # non-uniform
        emulated = rng.normal(size=(3000, 2))
        res = mmd(target, emulated, weights=w, seed=0)
        assert res["pvalue"] is None
        assert 0.0 <= res["retained_target_mass"] <= 1.0

    def test_weighted_statistic_detects_decorrelation(self):
        # Even without a p-value, the mmd2 statistic is a valid discrepancy: a
        # decorrelated weighted target must score higher than a matched one.
        rng = np.random.default_rng(21)
        n = 4000
        z = rng.normal(size=(n, 2))
        rho = 0.95
        target = np.column_stack(
            [z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]])
        w = np.abs(rng.normal(size=n)) + 0.01
        matched = target.copy()
        decorr = target.copy()
        decorr[:, 1] = rng.permutation(decorr[:, 1])
        s_matched = mmd(target, matched, weights=w, seed=0)["mmd2"]
        s_decorr = mmd(target, decorr, weights=w, seed=0)["mmd2"]
        assert s_decorr > s_matched

    def test_reports_retained_mass_when_truncated(self):
        rng = np.random.default_rng(6)
        target = rng.normal(size=(3000, 2))
        emulated = rng.normal(size=(3000, 2))
        res = mmd(target, emulated, max_points=500, seed=0)
        assert res["n_target"] == 500
        assert res["retained_target_mass"] < 1.0


class TestCertifyVerdicts:
    def test_pass_on_matching(self):
        # Large N: sample-std noise ~1/sqrt(2N) must sit well under the 1% width
        # tolerance, so a genuine match passes regardless of seed.
        target = _gaussian(40000, 3, 1)
        emulated = _gaussian(40000, 3, 2)
        report = certify(target, emulated, seed=0)
        assert report["verdict"] == "pass"
        assert report["reasons"] == []

    def test_fail_on_overdispersion(self):
        target = _gaussian(4000, 3, 1)
        emulated = _gaussian(4000, 3, 2, scale=1.10)  # 10% too wide
        report = certify(target, emulated, seed=0)
        assert report["verdict"] == "fail"
        assert any("width" in r for r in report["reasons"])

    def test_fail_on_bias(self):
        target = _gaussian(4000, 3, 1)
        emulated = _gaussian(4000, 3, 2, mean=0.3)  # shifted 0.3 sigma
        report = certify(target, emulated, seed=0)
        assert report["verdict"] == "fail"
        assert any("bias" in r for r in report["reasons"])

    def test_warn_on_decorrelation(self):
        # Right marginals, wrong correlation: passes moments/margins, MMD warns.
        rng = np.random.default_rng(20)
        n = 4000
        z = rng.normal(size=(n, 2))
        rho = 0.95
        target = np.column_stack(
            [z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]])
        emulated = target.copy()
        emulated[:, 1] = rng.permutation(emulated[:, 1])
        report = certify(target, emulated, seed=0)
        assert report["verdict"] == "warn"
        assert any("MMD" in r for r in report["reasons"])

    def test_fail_on_out_of_bounds(self):
        target = np.abs(_gaussian(4000, 1, 1)) + 0.5  # target well inside (>0)
        emulated = _gaussian(4000, 1, 2, mean=0.6)    # leaks below zero
        bounds = np.array([[0.0, np.inf]])
        report = certify(target, emulated, bounds=bounds, seed=0)
        assert report["verdict"] == "fail"
        assert any("out-of-bounds" in r for r in report["reasons"])

    def test_insufficient_on_low_ess(self):
        target = _gaussian(4000, 2, 1)
        emulated = _gaussian(4000, 2, 2)
        # Heavily skewed weights -> tiny effective sample size (~50 dominant).
        w = np.full(4000, 1e-6)
        w[:50] = 1.0
        report = certify(target, emulated, weights=w, min_ess=200, seed=0)
        assert report["verdict"] == "insufficient"
        assert report["ess"] < 200

    def test_insufficient_gate_precedes_metric_errors(self):
        # A near-constant parameter would make moment_error raise; the ESS gate
        # must fire first and return "insufficient" rather than crashing.
        rng = np.random.default_rng(8)
        target = np.zeros((4000, 2))
        target[:, 0] = rng.normal(size=4000)  # column 1 has zero spread
        emulated = rng.normal(size=(4000, 2))
        w = np.full(4000, 1e-6)
        w[:20] = 1.0  # ESS ~ 20
        report = certify(target, emulated, weights=w, min_ess=200, seed=0)
        assert report["verdict"] == "insufficient"

    def test_per_parameter_table_is_labelled(self):
        target = _gaussian(4000, 3, 1)
        emulated = _gaussian(4000, 3, 2)
        report = certify(target, emulated, parameters=["a", "b", "c"], seed=0)
        assert [r["parameter"] for r in report["per_parameter"]] == ["a", "b", "c"]
        assert "worst_width" in report and "worst_shape" in report

    def test_single_parameter_width_gate_catches_hidden_bad_param(self):
        # One parameter badly over-dispersed, but the model-mean width stays under
        # tolerance: the per-parameter cap must fail it and name the parameter.
        rng = np.random.default_rng(9)
        d = 8
        target = rng.normal(size=(5000, d))
        emulated = target.copy()
        emulated[:, 3] *= 1.04  # ~4% too wide on parameter index 3 only
        names = [f"x{i}" for i in range(d)]
        report = certify(target, emulated, parameters=names, seed=0)
        assert report["moments"]["mean_abs_width_err_pct"] < 1.0  # mean looks fine
        assert report["verdict"] == "fail"
        assert report["worst_width"]["parameter"] == "x3"
        assert any("x3" in r and "single-parameter" in r for r in report["reasons"])

    def test_weighted_match_does_not_spuriously_warn(self):
        # Weighted target -> MMD has no p-value -> the joint test cannot warn.
        # A well-matched weighted case must not warn.
        rng = np.random.default_rng(7)
        target = rng.normal(size=(10000, 3))
        w = np.abs(rng.normal(size=10000)) + 0.01  # non-uniform, position-independent
        emulated = rng.normal(size=(40000, 3))
        report = certify(target, emulated, weights=w, seed=0)
        assert report["mmd"]["pvalue"] is None
        assert report["verdict"] != "warn"
