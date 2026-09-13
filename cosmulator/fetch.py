"""Download published emulators from Zenodo.

This is the half of the package a user touches. It reads Zenodo's public
records API, which needs no authentication, and there is deliberately nowhere
in this module to supply a credential: no class here accepts a token, and
nothing here imports :mod:`cosmulator.publish`.

That separation is the point. An installation used only for consuming
emulators contains no code path that wants a secret, so there is nothing for a
misconfigured script or a recorded test to leak.

Emulators are published one deposit per (model, dataset), each holding the
whole ensemble of independently seeded flows for that combination. Chains and
emulators live in separate deposits because they have different lifecycles: a
chain is computed once, while an emulator is retrained whenever the
architecture or the hyperparameters improve.
"""

import shutil
from pathlib import Path
from urllib.parse import quote

import requests

from cosmulator.config import cache_root
from cosmulator.manifest import MANIFEST_NAME
from cosmulator.store import EmulatorStore

#: Zenodo's public records endpoint. No authentication, read only.
RECORDS_URL = "https://zenodo.org/api/records"

#: Sandbox equivalent, for testing against a disposable Zenodo.
SANDBOX_RECORDS_URL = "https://sandbox.zenodo.org/api/records"

#: Prefix identifying this package's deposits in a Zenodo search.
TITLE_PREFIX = "cosmulator:"

#: Seconds before a request to Zenodo is abandoned.
#:
#: Every call sets this. A request without a timeout can hang indefinitely,
#: which in a batch job means a silently wedged task rather than a failure.
TIMEOUT = 60


def deposit_title(model, dataset):
    """Return the Zenodo title identifying one emulator ensemble.

    Parameters
    ----------
    model : str
        Model name, for example ``'lcdm'``.
    dataset : str
        Dataset identifier, for example ``'planck_2018_plik'``.

    Returns
    -------
    str
        The deposit title, of the form ``'cosmulator: <model> <dataset>'``.
    """
    return f"{TITLE_PREFIX} {model} {dataset}"


class EmulatorArchive:
    """The published collection of emulators on Zenodo.

    Notes
    -----
    This class has no token parameter and cannot write to Zenodo. Publishing
    lives in :mod:`cosmulator.publish`, which is never imported from here.

    Parameters
    ----------
    sandbox : bool, optional
        Query Zenodo's sandbox rather than the live archive.
    cache : str or pathlib.Path, optional
        Directory to cache downloads in. Defaults to
        :func:`cosmulator.config.cache_root`.
    session : requests.Session, optional
        HTTP session to use. Supplying one allows connection reuse across many
        downloads, and lets tests substitute a transport.
    """

    def __init__(self, sandbox=False, cache=None, session=None):
        self.records_url = SANDBOX_RECORDS_URL if sandbox else RECORDS_URL
        self.cache = Path(cache) if cache is not None else cache_root()
        self._session = session or requests.Session()

    def _search(self, query, size=25):
        """Run one page of a Zenodo record search.

        Parameters
        ----------
        query : str
            Zenodo query string.
        size : int, optional
            Results per page. Zenodo caps this at 25.

        Returns
        -------
        dict
            The decoded response body.

        Raises
        ------
        requests.HTTPError
            If Zenodo returns an error status.
        """
        response = self._session.get(
            self.records_url,
            params={"q": query, "size": size},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def available(self):
        """List every (model, dataset) pair published in the archive.

        Returns
        -------
        list of tuple
            Sorted ``(model, dataset)`` pairs.

        Notes
        -----
        A network failure raises rather than returning an empty list. Silently
        reporting "nothing is published" when the network is down is the more
        dangerous outcome, because it looks like a valid answer.
        """
        pairs = set()
        page, total = 1, None
        while True:
            response = self._session.get(
                self.records_url,
                params={"q": f'title:"{TITLE_PREFIX}"', "size": 25, "page": page},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            hits = data.get("hits", {}).get("hits", [])
            if not hits:
                break
            if total is None:
                total = data.get("hits", {}).get("total", 0)
            for hit in hits:
                title = hit.get("metadata", {}).get("title", "")
                if title.startswith(TITLE_PREFIX):
                    parts = title[len(TITLE_PREFIX) :].strip().split(maxsplit=1)
                    if len(parts) == 2:
                        pairs.add((parts[0], parts[1]))
            if total is not None and page * 25 >= total:
                break
            page += 1
        return sorted(pairs)

    def record(self, model, dataset):
        """Return the Zenodo record for one emulator ensemble.

        Parameters
        ----------
        model : str
            Model name.
        dataset : str
            Dataset identifier.

        Returns
        -------
        dict
            The Zenodo record.

        Raises
        ------
        KeyError
            If nothing is published for that combination. The error names the
            combination, so a typo is obvious.
        """
        title = deposit_title(model, dataset)
        data = self._search(f'title:"{quote(title, safe=": ")}"', size=1)
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            raise KeyError(f"no published emulators for ({model!r}, {dataset!r})")
        return hits[0]

    def fetch(self, model, dataset, force=False):
        """Download one emulator ensemble into the local cache.

        A combination already present in the cache is not downloaded again
        unless ``force`` is set, so repeated use costs one network round trip
        in total rather than one per call.

        Parameters
        ----------
        model : str
            Model name.
        dataset : str
            Dataset identifier.
        force : bool, optional
            Re-download even if the cache already holds the ensemble.

        Returns
        -------
        EmulatorStore
            A store over the cached ensemble. Everything that works on a
            locally trained store works here unchanged.
        """
        target = self.cache / model / dataset
        if target.is_dir() and not force and any(target.rglob(MANIFEST_NAME)):
            return EmulatorStore(target)

        if force and target.is_dir():
            shutil.rmtree(target)

        record = self.record(model, dataset)
        for entry in record.get("files", []):
            name = entry.get("key")
            url = entry.get("links", {}).get("self")
            if not name or not url:
                continue
            # Zenodo stores each ensemble member's files under a flat naming
            # scheme; the leading component is the emulator's own directory.
            destination = target / Path(name).parent
            destination.mkdir(parents=True, exist_ok=True)
            self._download_file(url, destination / Path(name).name)
        return EmulatorStore(target)

    def _download_file(self, url, path):
        """Stream one file to disk.

        Streamed rather than buffered because a model file may be tens of
        megabytes and a user may fetch many of them.

        Parameters
        ----------
        url : str
            Direct download URL.
        path : pathlib.Path
            Destination.
        """
        with self._session.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            # Write to a temporary name first so that an interrupted download
            # cannot leave a truncated file that later looks like a cache hit.
            partial = path.with_suffix(path.suffix + ".partial")
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
            partial.replace(path)


def fetch(model, dataset, seed=None, sandbox=False, cache=None):
    """Download one emulator, or one ensemble, ready to use.

    This is the package's front door for a user who wants to sample a posterior
    without the chain that produced it.

    Parameters
    ----------
    model : str
        Model name, for example ``'lcdm'``.
    dataset : str
        Dataset identifier, for example ``'planck_2018_plik'``.
    seed : int, optional
        Return the single ensemble member with this seed. If omitted the whole
        ensemble is returned as a store.
    sandbox : bool, optional
        Use Zenodo's sandbox.
    cache : str or pathlib.Path, optional
        Override the download cache location.

    Returns
    -------
    Emulator or EmulatorStore
        One emulator when ``seed`` is given, otherwise the whole ensemble.

    Raises
    ------
    KeyError
        If the combination is not published, or the ensemble has no member
        with the requested seed.
    """
    archive = EmulatorArchive(sandbox=sandbox, cache=cache)
    store = archive.fetch(model, dataset)
    if seed is None:
        return store
    for emulator in store:
        if emulator.manifest.hyperparameters.seed == seed:
            return emulator
    available = sorted(e.manifest.hyperparameters.seed for e in store)
    raise KeyError(
        f"no emulator with seed {seed} for ({model!r}, {dataset!r}); "
        f"available seeds: {available}"
    )


__all__ = ["EmulatorArchive", "deposit_title", "fetch"]
