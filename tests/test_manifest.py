"""Tests for the on-disk emulator manifest."""

import json

import pytest

from cosmulator.manifest import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    Hyperparameters,
    Manifest,
    Provenance,
    Quality,
    Training,
)


def make_manifest(**overrides):
    """Build a representative manifest, overriding selected fields."""
    kwargs = dict(
        name="lcdm/planck_2018_plik/seed0",
        estimator="RealNVP",
        model_file="model.marg",
        provenance=Provenance(
            source="ns_lcdm_planck_2018_plik.csv",
            parameters=["logA", "ns", "H0", "ombh2", "omch2", "tau"],
            n_samples=106937,
            n_effective=12880,
        ),
        hyperparameters=Hyperparameters(
            seed=0,
            extra={"hidden_size": 400, "num_layers": 8, "lr": 3e-05},
        ),
        training=Training(seconds=4606.3, backend="gpu", load_seconds=1.0),
        quality=Quality(metrics={"dlogp": -0.0196, "abs_dlogp": 0.0196}),
    )
    kwargs.update(overrides)
    return Manifest(**kwargs)


class TestRoundTrip:
    """A manifest survives being written and read back."""

    def test_dict_round_trip_is_lossless(self):
        """to_dict followed by from_dict reproduces the original."""
        original = make_manifest()
        assert Manifest.from_dict(original.to_dict()) == original

    def test_file_round_trip_is_lossless(self, tmp_path):
        """write followed by read reproduces the original."""
        original = make_manifest()
        original.write(tmp_path)
        assert Manifest.read(tmp_path) == original

    def test_write_creates_the_directory(self, tmp_path):
        """Writing into a missing directory creates it."""
        target = tmp_path / "a" / "b" / "c"
        path = make_manifest().write(target)
        assert path == target / MANIFEST_NAME
        assert path.is_file()

    def test_written_file_is_readable_json(self, tmp_path):
        """The manifest is plain JSON, inspectable without this package."""
        make_manifest().write(tmp_path)
        data = json.loads((tmp_path / MANIFEST_NAME).read_text())
        assert data["estimator"] == "RealNVP"
        assert data["provenance"]["n_effective"] == 12880
        assert data["hyperparameters"]["seed"] == 0


class TestSchemaVersioning:
    """The schema version guards against silent misreads."""

    def test_new_manifests_carry_the_current_version(self):
        """A freshly built manifest records the current schema version."""
        assert make_manifest().schema_version == SCHEMA_VERSION

    def test_a_newer_schema_is_refused(self, tmp_path):
        """A manifest from a future version raises rather than being misread."""
        data = make_manifest().to_dict()
        data["schema_version"] = SCHEMA_VERSION + 1
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(data))
        with pytest.raises(ValueError, match="newer than this version"):
            Manifest.read(tmp_path)

    def test_an_older_schema_is_accepted(self, tmp_path):
        """Manifests from older versions remain readable."""
        data = make_manifest().to_dict()
        data["schema_version"] = 0
        (tmp_path / MANIFEST_NAME).write_text(json.dumps(data))
        assert Manifest.read(tmp_path).schema_version == 0


class TestFailureModes:
    """Malformed or absent manifests fail clearly."""

    def test_missing_manifest_names_the_directory(self, tmp_path):
        """The error says where it looked."""
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            Manifest.read(tmp_path)

    def test_missing_required_field_is_reported(self):
        """A truncated manifest names the fields it lacks."""
        data = make_manifest().to_dict()
        del data["estimator"]
        with pytest.raises(ValueError, match="missing required fields.*estimator"):
            Manifest.from_dict(data)

    def test_quality_is_optional(self):
        """An unvalidated emulator has a manifest but no metrics."""
        data = make_manifest().to_dict()
        del data["quality"]
        assert Manifest.from_dict(data).quality.metrics == {}


class TestQuality:
    """Quality metrics are free-form but conveniently accessible."""

    def test_get_returns_a_known_metric(self):
        """A recorded metric is returned."""
        assert make_manifest().quality.get("dlogp") == pytest.approx(-0.0196)

    def test_get_returns_the_default_for_an_unknown_metric(self):
        """An absent metric yields the supplied default rather than raising."""
        assert make_manifest().quality.get("not_measured", "n/a") == "n/a"


def test_manifest_module_does_not_require_jax():
    """Reading manifests must not pull in the optional training stack."""
    import sys

    assert "jax" not in sys.modules
    assert "margarine" not in sys.modules
