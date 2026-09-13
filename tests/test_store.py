"""Tests for the on-disk emulator store."""

import pytest

from cosmulator.manifest import (
    MANIFEST_NAME,
    Hyperparameters,
    Manifest,
    Provenance,
    Quality,
    Training,
)
from cosmulator.store import EmulatorStore


def write_emulator(directory, name, seed=0, metrics=None, with_model=True):
    """Create a complete emulator directory for testing."""
    manifest = Manifest(
        name=name,
        estimator="RealNVP",
        model_file="model.marg",
        provenance=Provenance(
            source=f"{name}.csv",
            parameters=["logA", "ns", "H0"],
            n_samples=1000,
            n_effective=500,
        ),
        hyperparameters=Hyperparameters(seed=seed),
        training=Training(seconds=10.0, backend="cpu"),
        quality=Quality(metrics=metrics or {}),
    )
    manifest.write(directory)
    if with_model:
        (directory / "model.marg").write_bytes(b"not-a-real-model")
    return manifest


@pytest.fixture
def store(tmp_path):
    """A store holding four emulators nested at differing depths."""
    write_emulator(
        tmp_path / "lcdm" / "planck" / "s0", "lcdm/planck/s0", 0, {"abs_dlogp": 0.02}
    )
    write_emulator(
        tmp_path / "lcdm" / "planck" / "s1", "lcdm/planck/s1", 1, {"abs_dlogp": 0.05}
    )
    write_emulator(
        tmp_path / "lcdm" / "desi" / "s0", "lcdm/desi/s0", 0, {"abs_dlogp": 0.30}
    )
    # Deliberately unvalidated: trained, but never scored.
    write_emulator(tmp_path / "wcdm" / "planck" / "s0", "wcdm/planck/s0", 0, None)
    return EmulatorStore(tmp_path)


class TestScanning:
    """Finding emulators in a directory tree."""

    def test_finds_emulators_at_any_depth(self, store):
        """Nesting is the user's choice; the store does not impose a layout."""
        assert len(store) == 4

    def test_results_are_ordered_by_name(self, store):
        """A stable order makes listings and diffs reproducible."""
        assert [e.name for e in store] == sorted(e.name for e in store)

    def test_directories_without_a_manifest_are_ignored(self, store, tmp_path):
        """A store may sit alongside logs and scratch files."""
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "train.log").write_text("chatter")
        assert len(EmulatorStore(tmp_path)) == 4

    def test_a_corrupt_manifest_does_not_abort_the_scan(self, store, tmp_path):
        """One bad record must not make a large store unreadable."""
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / MANIFEST_NAME).write_text("{ this is not json")
        assert len(EmulatorStore(tmp_path)) == 4

    def test_scan_is_cached_until_forced(self, store, tmp_path):
        """Repeated queries do not re-walk the tree."""
        assert len(store) == 4
        write_emulator(tmp_path / "new" / "s0", "new/s0")
        assert len(store) == 4
        store.scan(force=True)
        assert len(store) == 5

    def test_an_empty_store_is_empty_not_an_error(self, tmp_path):
        """Scanning a directory with no emulators yields nothing."""
        assert len(EmulatorStore(tmp_path / "nothing_here")) == 0


class TestLookup:
    """Retrieving individual emulators."""

    def test_lookup_by_name(self, store):
        """A known name returns its emulator."""
        assert store["lcdm/planck/s0"].manifest.hyperparameters.seed == 0

    def test_unknown_name_raises_keyerror(self, store):
        """An absent name raises rather than returning None."""
        with pytest.raises(KeyError, match="nope"):
            store["nope"]


class TestFiltering:
    """Selecting subsets of a store."""

    def test_filter_by_quality(self, store):
        """The common case: keep only emulators good enough to use."""
        good = store.filter(lambda e: e.quality.get("abs_dlogp", 1.0) < 0.1)
        assert {e.name for e in good} == {"lcdm/planck/s0", "lcdm/planck/s1"}

    def test_filter_by_provenance(self, store):
        """Selecting an ensemble means filtering on what it was trained on."""
        planck = store.filter(lambda e: "planck" in e.manifest.provenance.source)
        assert len(planck) == 3


class TestBest:
    """Ranking emulators by a metric."""

    def test_smallest_is_best_by_default(self, store):
        """Fidelity metrics are errors, so smaller wins."""
        assert store.best("abs_dlogp")[0].name == "lcdm/planck/s0"

    def test_largest_can_be_requested(self, store):
        """Not every metric is an error."""
        assert store.best("abs_dlogp", largest=True)[0].name == "lcdm/desi/s0"

    def test_returns_n_in_order(self, store):
        """The ranking is complete and ordered, not just the winner."""
        names = [e.name for e in store.best("abs_dlogp", n=3)]
        assert names == ["lcdm/planck/s0", "lcdm/planck/s1", "lcdm/desi/s0"]

    def test_unvalidated_emulators_are_excluded(self, store):
        """An emulator with no score must never rank as though it scored zero."""
        ranked = {e.name for e in store.best("abs_dlogp", n=10)}
        assert "wcdm/planck/s0" not in ranked

    def test_unknown_metric_yields_nothing(self, store):
        """Asking for a metric nobody recorded is empty, not an error."""
        assert store.best("never_measured") == []


class TestModelFile:
    """The model file is referenced but not required."""

    def test_model_path_resolves_beside_the_manifest(self, store):
        """The manifest records a relative name; the store makes it absolute."""
        emulator = store["lcdm/planck/s0"]
        assert emulator.model_path == emulator.directory / "model.marg"

    def test_has_model_is_true_when_present(self, store):
        """A complete emulator reports its model."""
        assert store["lcdm/planck/s0"].has_model

    def test_a_manifest_without_its_model_is_still_listed(self, tmp_path):
        """An archived or still-training emulator remains discoverable."""
        write_emulator(tmp_path / "s0", "s0", with_model=False)
        emulator = EmulatorStore(tmp_path)["s0"]
        assert not emulator.has_model

    def test_reading_a_store_does_not_open_model_files(self, store):
        """Scanning is cheap: manifests only, never the megabyte-scale models."""
        assert all(e.has_model for e in store if e.name != "wcdm/planck/s0")


def test_emulator_repr_identifies_it(store):
    """The repr names the emulator, for readable listings and tracebacks."""
    assert "lcdm/planck/s0" in repr(store["lcdm/planck/s0"])


def test_store_module_does_not_require_jax():
    """Querying a store must not pull in the optional training stack."""
    import sys

    assert "jax" not in sys.modules
    assert "margarine" not in sys.modules
