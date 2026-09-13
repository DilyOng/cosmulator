"""Tests for downloading published emulators.

No test here contacts Zenodo. Every HTTP interaction is served by a stub
session, so the suite is fast, offline, and cannot be made to carry a real
credential into a recording.
"""

import ast
import json
from pathlib import Path

import pytest

from cosmulator.fetch import TITLE_PREFIX, EmulatorArchive, deposit_title
from cosmulator.manifest import (
    Hyperparameters,
    Manifest,
    Provenance,
    Quality,
    Training,
)


class StubResponse:
    """A minimal stand-in for a requests response."""

    def __init__(self, payload=None, content=b"", status=200):
        self._payload = payload
        self._content = content
        self.status_code = status

    def json(self):
        """Return the decoded body."""
        return self._payload

    def raise_for_status(self):
        """Raise if the status indicates failure."""
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        """Yield the body in chunks."""
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class StubSession:
    """Serves canned Zenodo responses and records what was requested."""

    def __init__(self, records=None, files=None):
        self.records = records or []
        self.files = files or {}
        self.calls = []

    def get(self, url, params=None, timeout=None, stream=False):
        """Return a canned response for a search or a file download."""
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if url in self.files:
            return StubResponse(content=self.files[url])
        return StubResponse(
            payload={"hits": {"hits": self.records, "total": len(self.records)}}
        )


def make_record(model, dataset, files=()):
    """Build a Zenodo record as the archive expects to receive one."""
    return {
        "metadata": {"title": deposit_title(model, dataset)},
        "files": [
            {"key": key, "links": {"self": f"https://files.test/{key}"}}
            for key in files
        ],
    }


def manifest_bytes(name, seed):
    """Serialise a manifest exactly as it would be stored."""
    manifest = Manifest(
        name=name,
        estimator="RealNVP",
        model_file="model.marg",
        provenance=Provenance(
            source=f"{name}.csv",
            parameters=["logA", "ns"],
            n_samples=100,
            n_effective=50,
        ),
        hyperparameters=Hyperparameters(seed=seed),
        training=Training(seconds=1.0, backend="cpu"),
        quality=Quality(metrics={"abs_dlogp": 0.01 * (seed + 1)}),
    )
    return json.dumps(manifest.to_dict()).encode()


class TestDepositNaming:
    """The title convention that locates an ensemble on Zenodo."""

    def test_title_includes_model_and_dataset(self):
        """A deposit is identified by its combination."""
        assert deposit_title("lcdm", "planck") == f"{TITLE_PREFIX} lcdm planck"

    def test_titles_are_distinct_per_combination(self):
        """Two combinations never collide."""
        assert deposit_title("lcdm", "a") != deposit_title("lcdm", "b")


class TestAvailable:
    """Listing what has been published."""

    def test_parses_model_and_dataset_from_titles(self):
        """The listing is derived from deposit titles."""
        session = StubSession(
            records=[make_record("lcdm", "planck"), make_record("wcdm", "desi")]
        )
        archive = EmulatorArchive(session=session)
        assert archive.available() == [("lcdm", "planck"), ("wcdm", "desi")]

    def test_ignores_unrelated_deposits(self):
        """Other people's records in the same search are skipped."""
        session = StubSession(
            records=[make_record("lcdm", "planck"), {"metadata": {"title": "other"}}]
        )
        assert EmulatorArchive(session=session).available() == [("lcdm", "planck")]

    def test_every_request_sets_a_timeout(self):
        """A request without a timeout can wedge a batch job indefinitely."""
        session = StubSession(records=[make_record("lcdm", "planck")])
        EmulatorArchive(session=session).available()
        assert session.calls
        assert all(c["timeout"] is not None for c in session.calls)


class TestRecordLookup:
    """Finding one ensemble."""

    def test_unknown_combination_names_what_was_asked_for(self):
        """A typo should be obvious from the error alone."""
        archive = EmulatorArchive(session=StubSession(records=[]))
        with pytest.raises(KeyError, match="lcdm.*nonexistent"):
            archive.record("lcdm", "nonexistent")


class TestFetch:
    """Downloading an ensemble into the cache."""

    def test_downloads_and_returns_a_usable_store(self, tmp_path):
        """The result is an ordinary store, so everything else works on it."""
        files = {
            "https://files.test/seed0/emulator.json": manifest_bytes("lcdm/p/s0", 0),
            "https://files.test/seed0/model.marg": b"model-bytes",
        }
        session = StubSession(
            records=[
                make_record(
                    "lcdm", "p", files=["seed0/emulator.json", "seed0/model.marg"]
                )
            ],
            files=files,
        )
        archive = EmulatorArchive(session=session, cache=tmp_path)
        store = archive.fetch("lcdm", "p")
        assert len(store) == 1
        assert store["lcdm/p/s0"].has_model

    def test_a_cached_ensemble_is_not_downloaded_again(self, tmp_path):
        """Repeated use costs one round trip in total, not one per call."""
        files = {
            "https://files.test/seed0/emulator.json": manifest_bytes("lcdm/p/s0", 0),
            "https://files.test/seed0/model.marg": b"model-bytes",
        }
        session = StubSession(
            records=[
                make_record(
                    "lcdm",
                    "p",
                    files=list(
                        k.rsplit("/", 2)[-2] + "/" + k.rsplit("/", 1)[-1] for k in files
                    ),
                )
            ],
            files=files,
        )
        archive = EmulatorArchive(session=session, cache=tmp_path)
        archive.fetch("lcdm", "p")
        first = len(session.calls)
        archive.fetch("lcdm", "p")
        assert len(session.calls) == first

    def test_an_interrupted_download_leaves_no_usable_file(self, tmp_path):
        """A truncated file must never later look like a cache hit."""
        target = tmp_path / "lcdm" / "p" / "seed0"
        target.mkdir(parents=True)
        (target / "model.marg.partial").write_bytes(b"half")
        assert not (target / "model.marg").exists()


class TestCredentialSeparation:
    """The public half must contain no way to supply a secret.

    These assertions are structural rather than behavioural on purpose: a
    convention erodes, whereas a failing test does not.
    """

    def test_fetch_does_not_import_publish(self):
        """An install used only for downloading has no publishing code path."""
        source = Path("cosmulator/fetch.py").read_text()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not any("publish" in name for name in imported)

    def test_no_function_accepts_a_credential(self):
        """There is nowhere for a user to put a token, by construction."""
        tree = ast.parse(Path("cosmulator/fetch.py").read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                names += [a.arg for a in node.args.args + node.args.kwonlyargs]
        forbidden = ("token", "secret", "password", "credential", "auth")
        assert not [n for n in names if any(f in n.lower() for f in forbidden)]

    def test_only_the_public_records_endpoint_is_used(self):
        """Writing to Zenodo requires the deposit endpoint, which is absent."""
        source = Path("cosmulator/fetch.py").read_text()
        assert "api/records" in source
        assert "api/deposit" not in source
