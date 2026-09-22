"""Tests for the ensemble deliverable that do not need the train extra.

The mixture log_prob / sampling paths need JAX+flowjax and are exercised on GPU;
here we cover the JAX-free structure: the D_KL prior-box guards and the
equal-weight bookkeeping.
"""
import numpy as np
import pytest

from cosmulator.ensemble import EnsembleEmulator


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
