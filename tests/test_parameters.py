"""Tests for model-aware cosmological-parameter selection."""

import pytest

from cosmulator import BASE, COSMO_SAMPLED, cosmological_parameters, parameter_names


class _FakeMultiIndex:
    """Minimal stand-in for a pandas MultiIndex of (name, label, ...) tuples.

    anesthetic stores columns this way; the real thing needs pandas, which the
    core package does not depend on, so the level-0 access is exercised here
    against a tiny fake instead.
    """

    def __init__(self, names):
        self._names = list(names)

    def get_level_values(self, level):
        assert level == 0
        return list(self._names)


class _FakeSamples:
    """Stand-in for an anesthetic Samples object: it exposes ``columns``."""

    def __init__(self, columns):
        self.columns = columns


class TestParameterNames:
    def test_plain_sequence_is_returned_as_a_list(self):
        assert parameter_names(["logA", "ns", "H0"]) == ["logA", "ns", "H0"]

    def test_multiindex_first_level_is_extracted(self):
        cols = _FakeMultiIndex(["logA", "ns", "H0", "ombh2", "omch2", "tau"])
        assert parameter_names(cols) == ["logA", "ns", "H0", "ombh2", "omch2", "tau"]

    def test_object_with_columns_is_unwrapped(self):
        s = _FakeSamples(_FakeMultiIndex(["logA", "H0"]))
        assert parameter_names(s) == ["logA", "H0"]

    def test_tuples_are_flattened_to_their_first_entry(self):
        assert parameter_names([("logA", "$A$"), ("H0", "$H_0$")]) == ["logA", "H0"]


class TestCosmologicalParameters:
    def test_lcdm_returns_the_six_base_parameters(self):
        cols = BASE + ["As", "omegam", "sigma8", "A_planck", "logL", "weights"]
        assert cosmological_parameters(cols) == BASE

    def test_extensions_are_included_in_canonical_order(self):
        # walcdm: base + w0 + wa, given in a scrambled column order.
        cols = ["wa", "H0", "w0", "logA", "tau", "ns", "ombh2", "omch2", "As"]
        assert cosmological_parameters(cols) == [
            "logA", "ns", "H0", "ombh2", "omch2", "tau", "w0", "wa",
        ]

    def test_derived_and_nuisance_columns_are_excluded(self):
        cols = BASE + ["omk", "As", "sigma8", "S8", "A_cib_217", "calib_100T"]
        result = cosmological_parameters(cols)
        assert result == BASE + ["omk"]
        assert "As" not in result and "A_cib_217" not in result

    def test_ordering_follows_the_superset_not_the_columns(self):
        cols = ["tau", "omch2", "ombh2", "H0", "ns", "logA"]  # reversed
        assert cosmological_parameters(cols) == BASE

    def test_only_known_sampled_parameters_are_returned(self):
        result = cosmological_parameters(BASE + ["mnu", "omk", "nrun"])
        assert set(result) <= set(COSMO_SAMPLED)

    def test_missing_base_parameter_raises(self):
        cols = ["logA", "ns", "H0", "ombh2", "omch2"]  # no tau
        with pytest.raises(ValueError, match="tau"):
            cosmological_parameters(cols)

    def test_accepts_a_samples_like_object(self):
        s = _FakeSamples(_FakeMultiIndex(BASE + ["w", "wa"]))
        assert cosmological_parameters(s) == BASE + ["w", "wa"]
