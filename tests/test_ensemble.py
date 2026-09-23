"""Tests for the ensemble deliverable that do not need the train extra.

The mixture log_prob / sampling paths need JAX+flowjax and are exercised on GPU;
here we cover the JAX-free structure: the D_KL prior-box guards and the
equal-weight bookkeeping.
"""
import numpy as np
import pytest

from cosmulator.ensemble import EnsembleEmulator, _member_healthy


class TestMemberHealth:
    REF = np.array([2.0, 0.5, 20.0])

    def test_accepts_well_scaled_finite_member(self):
        rng = np.random.default_rng(0)
        g = rng.normal(size=(5000, 3)) * self.REF
        assert _member_healthy(g, self.REF)

    def test_rejects_nonfinite(self):
        g = np.ones((100, 3))
        g[0, 0] = np.inf
        assert not _member_healthy(g, self.REF)

    def test_rejects_blown_up_scale(self):
        rng = np.random.default_rng(1)
        g = rng.normal(size=(5000, 3)) * self.REF
        g[:, 2] *= 100.0                       # one parameter diverges
        assert not _member_healthy(g, self.REF, std_factor=5.0)

    def test_rejects_collapsed_scale(self):
        g = np.zeros((5000, 3)) + [2.0, 0.5, 20.0]   # zero spread
        assert not _member_healthy(g, self.REF)


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
