"""Tests for uploading emulators to Zenodo.

No test contacts Zenodo and no test handles a real credential. A stub session
records every request, which is what lets the tests assert where the token
travels rather than merely that a call succeeded.
"""

import ast
import json
from pathlib import Path

import pytest

from cosmulator.manifest import (
    Hyperparameters,
    Manifest,
    Provenance,
    Quality,
    Training,
)
from cosmulator.publish import (
    DEPOSIT_URL,
    SANDBOX_DEPOSIT_URL,
    EmulatorPublisher,
)
from cosmulator.store import EmulatorStore

FAKE_TOKEN = "not-a-real-token-0000"


class StubResponse:
    """A minimal stand-in for a requests response."""

    def __init__(self, payload=None, status=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status

    def json(self):
        """Return the decoded body."""
        return self._payload

    def raise_for_status(self):
        """Raise if the status indicates failure."""
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


class StubSession:
    """Records every request so the tests can inspect them."""

    def __init__(self, payloads=None):
        self.payloads = payloads or {}
        self.requests = []

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        """Record the request and return a canned response."""
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "timeout": timeout,
                "params": kwargs.get("params"),
                "json": kwargs.get("json"),
            }
        )
        for fragment, payload in self.payloads.items():
            if fragment in url:
                return StubResponse(payload)
        return StubResponse(
            {"id": 12345, "links": {"bucket": "https://bucket.test/abc"}}
        )


def write_emulator(directory, name, seed, metrics=None, with_model=True):
    """Create a complete emulator directory."""
    Manifest(
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
        training=Training(seconds=10.0, backend="gpu"),
        quality=Quality(metrics=metrics if metrics is not None else {}),
    ).write(directory)
    if with_model:
        (directory / "model.marg").write_bytes(b"model-bytes")


@pytest.fixture
def ensemble(tmp_path):
    """A three-member ensemble on disk."""
    for seed in (0, 1, 2):
        write_emulator(
            tmp_path / f"seed{seed}",
            f"lcdm/planck/s{seed}",
            seed,
            {"abs_dlogp": 0.01 * (seed + 1)},
        )
    return list(EmulatorStore(tmp_path))


class TestSafetyDefaults:
    """Publishing is irreversible, so the safe target is the default."""

    def test_sandbox_is_the_default(self):
        """Omitting the argument must not reach the live archive."""
        publisher = EmulatorPublisher(FAKE_TOKEN)
        assert publisher.sandbox is True
        assert publisher.base_url == SANDBOX_DEPOSIT_URL

    def test_production_must_be_requested_explicitly(self):
        """Reaching the permanent archive requires saying so."""
        publisher = EmulatorPublisher(FAKE_TOKEN, sandbox=False)
        assert publisher.base_url == DEPOSIT_URL

    def test_repr_announces_production_loudly(self):
        """A publisher pointed at the live archive should be obvious."""
        assert "PRODUCTION" in repr(EmulatorPublisher(FAKE_TOKEN, sandbox=False))
        assert "sandbox" in repr(EmulatorPublisher(FAKE_TOKEN))

    def test_repr_never_reveals_the_token(self):
        """A repr often ends up in logs and tracebacks."""
        assert FAKE_TOKEN not in repr(EmulatorPublisher(FAKE_TOKEN, sandbox=False))

    def test_a_missing_token_fails_immediately(self):
        """Failing at construction beats failing halfway through an upload."""
        with pytest.raises(ValueError, match="access token is required"):
            EmulatorPublisher("")


class TestTokenHandling:
    """Where the token travels is the whole point of this module."""

    def test_token_is_sent_as_an_authorization_header(self, ensemble):
        """A header is scrubbed by default; a query parameter is not."""
        session = StubSession()
        publisher = EmulatorPublisher(FAKE_TOKEN, session=session)
        publisher.create_deposit()
        assert session.requests[0]["headers"]["Authorization"] == f"Bearer {FAKE_TOKEN}"

    def test_token_never_appears_in_a_url(self, ensemble):
        """This is the exact failure that leaked a live token once before."""
        session = StubSession()
        publisher = EmulatorPublisher(FAKE_TOKEN, session=session)
        publisher.create_deposit()
        publisher.publish(1)
        publisher.new_version(1)
        publisher.find_deposits("anything")
        for request in session.requests:
            assert FAKE_TOKEN not in request["url"]
            assert FAKE_TOKEN not in json.dumps(request["params"])

    def test_every_request_sets_a_timeout(self):
        """An untimed request wedges a batch job rather than failing it."""
        session = StubSession()
        publisher = EmulatorPublisher(FAKE_TOKEN, session=session)
        publisher.create_deposit()
        publisher.publish(1)
        assert all(r["timeout"] is not None for r in session.requests)


class TestMetadata:
    """What a prospective user sees before downloading."""

    def test_title_matches_what_fetch_searches_for(self, ensemble):
        """Publishing and downloading must agree on the naming convention."""
        from cosmulator.fetch import deposit_title

        publisher = EmulatorPublisher(FAKE_TOKEN, session=StubSession())
        metadata = publisher.describe("lcdm", "planck", [e.manifest for e in ensemble])
        assert metadata["metadata"]["title"] == deposit_title("lcdm", "planck")

    def test_description_records_the_ensemble_size_and_seeds(self, ensemble):
        """An ensemble is only meaningful if its membership is stated."""
        publisher = EmulatorPublisher(FAKE_TOKEN, session=StubSession())
        text = publisher.describe("lcdm", "planck", [e.manifest for e in ensemble])[
            "metadata"
        ]["description"]
        assert "3 independently seeded members" in text
        assert "[0, 1, 2]" in text

    def test_description_reports_the_quality_range(self, ensemble):
        """A user should be able to judge the ensemble without downloading it."""
        publisher = EmulatorPublisher(FAKE_TOKEN, session=StubSession())
        text = publisher.describe("lcdm", "planck", [e.manifest for e in ensemble])[
            "metadata"
        ]["description"]
        assert "0.0100" in text and "0.0300" in text

    def test_unscored_ensembles_say_so_rather_than_implying_quality(self, tmp_path):
        """Silence about quality must not read as good quality."""
        write_emulator(tmp_path / "seed0", "lcdm/planck/s0", 0, metrics={})
        emulators = list(EmulatorStore(tmp_path))
        publisher = EmulatorPublisher(FAKE_TOKEN, session=StubSession())
        text = publisher.describe("lcdm", "planck", [e.manifest for e in emulators])[
            "metadata"
        ]["description"]
        assert "have not been scored" in text


class TestUpload:
    """Putting an ensemble into a deposit."""

    def test_members_are_namespaced_by_seed(self, ensemble):
        """One deposit holds a whole ensemble, so names must not collide."""
        session = StubSession()
        publisher = EmulatorPublisher(FAKE_TOKEN, session=session)
        deposit = {"links": {"bucket": "https://bucket.test/abc"}}
        publisher.upload_emulator(deposit, ensemble[1])
        uploaded = [r["url"] for r in session.requests]
        assert any("seed1/emulator.json" in u for u in uploaded)
        assert any("seed1/model.marg" in u for u in uploaded)

    def test_a_manifest_without_its_model_is_refused(self, tmp_path):
        """Publishing a manifest with no model would create a broken record."""
        write_emulator(tmp_path / "seed0", "lcdm/p/s0", 0, with_model=False)
        emulator = list(EmulatorStore(tmp_path))[0]
        publisher = EmulatorPublisher(FAKE_TOKEN, session=StubSession())
        with pytest.raises(FileNotFoundError, match="no model file"):
            publisher.upload_emulator(
                {"links": {"bucket": "https://bucket.test/abc"}}, emulator
            )

    def test_a_deposit_without_a_bucket_is_reported(self, ensemble):
        """A malformed deposit should fail clearly, not with a KeyError deep in."""
        publisher = EmulatorPublisher(FAKE_TOKEN, session=StubSession())
        with pytest.raises(KeyError, match="no upload bucket"):
            publisher.upload_emulator({}, ensemble[0])


class TestPublishEnsemble:
    """The end-to-end convenience path."""

    def test_dry_run_contacts_nothing(self, ensemble):
        """Checking what would be published must be free of side effects."""
        session = StubSession()
        publisher = EmulatorPublisher(FAKE_TOKEN, session=session)
        result = publisher.publish_ensemble("lcdm", "planck", ensemble, dry_run=True)
        assert session.requests == []
        assert result["metadata"]["title"].startswith("cosmulator:")

    def test_an_empty_ensemble_is_refused(self):
        """Publishing nothing would create an empty permanent record."""
        publisher = EmulatorPublisher(FAKE_TOKEN, session=StubSession())
        with pytest.raises(ValueError, match="no emulators to publish"):
            publisher.publish_ensemble("lcdm", "planck", [])


class TestSeparationFromFetch:
    """The consuming half must remain free of publishing code."""

    def test_publish_is_not_imported_by_fetch(self):
        """Asserted from both sides, so neither direction can drift."""
        tree = ast.parse(Path("cosmulator/fetch.py").read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any("publish" in name for name in imported)

    def test_publish_is_not_exported_from_the_package_namespace(self):
        """Importing cosmulator must not put a publisher within easy reach."""
        import cosmulator

        assert "EmulatorPublisher" not in cosmulator.__all__
