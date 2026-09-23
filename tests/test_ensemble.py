"""Tests for the ensemble deliverable that do not need the train extra.

The mixture log_prob / sampling paths need JAX+flowjax and are exercised on GPU;
here we cover the JAX-free structure: the D_KL prior-box guards and the
equal-weight bookkeeping.
"""
import numpy as np
import pytest

from cosmulator.ensemble import EnsembleEmulator, _member_valid

# a prior box for three bounded parameters
BOX = np.array([[0.0, 1.0], [-2.0, 2.0], [10.0, 100.0]])


class TestMemberValid:
    def test_accepts_in_box_member(self):
        rng = np.random.default_rng(0)
        g = np.column_stack([rng.uniform(0.1, 0.9, 5000),
                             rng.uniform(-1.5, 1.5, 5000),
                             rng.uniform(20, 90, 5000)])
        assert _member_valid(g, BOX)

    def test_accepts_small_wall_leakage(self):
        # a healthy flow (no bijector) may leak a little mass just past a wall
        rng = np.random.default_rng(1)
        g = np.column_stack([rng.normal(0.02, 0.05, 5000),   # ~a few % below 0
                             rng.uniform(-1.5, 1.5, 5000),
                             rng.uniform(20, 90, 5000)])
        assert _member_valid(g, BOX)

    def test_rejects_nonfinite(self):
        g = np.column_stack([np.full(100, 0.5), np.zeros(100), np.full(100, 50.0)])
        g[0, 0] = np.inf
        assert not _member_valid(g, BOX)

    def test_rejects_gross_blowup(self):
        rng = np.random.default_rng(2)
        g = np.column_stack([rng.uniform(0.1, 0.9, 5000),
                             rng.uniform(-1.5, 1.5, 5000),
                             rng.uniform(20, 90, 5000)])
        g[0, 2] = 1e10                          # astronomical excursion
        assert not _member_valid(g, BOX)

    def test_rejects_mostly_out_of_box(self):
        rng = np.random.default_rng(3)
        g = np.column_stack([rng.uniform(2.0, 5.0, 5000),    # all well above hi=1
                             rng.uniform(-1.5, 1.5, 5000),
                             rng.uniform(20, 90, 5000)])
        assert not _member_valid(g, BOX)

    def test_truth_independent_ignores_scale_vs_any_reference(self):
        # a valid member is accepted regardless of how its spread compares to a
        # (here absent) posterior -- the gate never sees the truth
        rng = np.random.default_rng(4)
        g = np.column_stack([rng.uniform(0.4, 0.6, 5000),    # narrow but in-box
                             rng.uniform(-0.1, 0.1, 5000),
                             rng.uniform(49, 51, 5000)])
        assert _member_valid(g, BOX)


def _empty(bounds=None):
    return EnsembleEmulator(members=[], parameters=["a", "b"],
                            arch={"flow_layers": 4, "nn_width": 32, "nn_depth": 1,
                                  "spline": False}, bounds=bounds)


def test_module_imports_without_flowjax():
    e = _empty()
    assert e.K == 0
    assert e.parameters == ["a", "b"]


def test_dkl_requires_bounds():
    with pytest.raises(ValueError, match="bounds"):
        _empty(bounds=None).D_KL()


def test_dkl_requires_finite_box():
    # a one-sided (improper uniform) prior has no finite log-volume
    with pytest.raises(ValueError, match="finite"):
        _empty(bounds=np.array([[0.0, 1.0], [0.0, np.inf]])).D_KL()


def test_bounds_stored_as_array():
    e = _empty(bounds=[[0.0, 1.0], [-2.0, 2.0]])
    assert isinstance(e.bounds, np.ndarray)
    assert e.bounds.shape == (2, 2)
