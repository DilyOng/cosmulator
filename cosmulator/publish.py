"""Upload trained emulators to Zenodo.

This is the credentialed half of the package and the only module that knows
Zenodo's deposit API exists. Nothing in :mod:`cosmulator.fetch` imports it, so
an installation used purely for consuming emulators contains no code path that
wants a secret.

Two decisions here are deliberate departures from the pattern this package
otherwise follows from ``unimpeded``.

The token is sent in an ``Authorization`` header, never as a query parameter.
A token in a URL is recorded by web-server logs, proxies, ``Referer`` headers
and any HTTP test recording, which is precisely how a live Zenodo token once
reached a public repository and stayed there for five months. A token in a
header is what every scrubbing tool inspects by default.

The sandbox is the default. Publishing is irreversible: a published deposit
cannot be deleted, and its DOI is permanent. Defaulting to the live archive
would mean a single forgotten argument produces a permanent public record, so
reaching production requires saying so explicitly.
"""

from pathlib import Path

import requests

from cosmulator.manifest import MANIFEST_NAME

#: Zenodo's deposit endpoints. Writing requires authentication.
DEPOSIT_URL = "https://zenodo.org/api/deposit/depositions"

#: Sandbox equivalent: a disposable Zenodo whose records carry no weight.
SANDBOX_DEPOSIT_URL = "https://sandbox.zenodo.org/api/deposit/depositions"

#: Seconds before a request to Zenodo is abandoned.
TIMEOUT = 120

#: Prefix identifying this package's deposits, matching :mod:`cosmulator.fetch`.
TITLE_PREFIX = "cosmulator:"


class EmulatorPublisher:
    """Create, populate and publish emulator deposits on Zenodo.

    Parameters
    ----------
    token : str
        A Zenodo personal access token with the ``deposit:write`` and
        ``deposit:actions`` scopes. Passed explicitly: this package never reads
        a credential from the environment, so that the one place a token
        enters the process is visible in the caller's own code.
    sandbox : bool, optional
        Use Zenodo's sandbox. Defaults to True, because publishing to the live
        archive is irreversible and should require an explicit choice.

    Examples
    --------
    >>> publisher = EmulatorPublisher(token, sandbox=True)   # doctest: +SKIP
    >>> deposit = publisher.create_deposit()                 # doctest: +SKIP
    """

    def __init__(self, token, sandbox=True, session=None):
        if not token:
            raise ValueError("a Zenodo access token is required to publish")
        self.sandbox = bool(sandbox)
        self.base_url = SANDBOX_DEPOSIT_URL if self.sandbox else DEPOSIT_URL
        self._token = token
        self._session = session or requests.Session()

    @property
    def _headers(self):
        """dict: Authorization header carrying the token.

        The token travels here rather than in the query string so that it is
        not captured by server logs, proxies or HTTP recordings.
        """
        return {"Authorization": f"Bearer {self._token}"}

    def _request(self, method, url, **kwargs):
        """Issue one authenticated request.

        Every call routes through here so that the token is attached in exactly
        one place and every request carries a timeout.

        Parameters
        ----------
        method : str
            HTTP method.
        url : str
            Target URL.
        **kwargs
            Passed to the underlying session.

        Returns
        -------
        requests.Response
            The response, already checked for an error status.
        """
        headers = dict(kwargs.pop("headers", {}))
        headers.update(self._headers)
        response = self._session.request(
            method, url, headers=headers, timeout=TIMEOUT, **kwargs
        )
        response.raise_for_status()
        return response

    def create_deposit(self):
        """Create a new empty deposit.

        Returns
        -------
        dict
            The new deposit, including its ``id`` and upload ``links``.
        """
        return self._request("POST", self.base_url, json={}).json()

    def describe(self, model, dataset, manifests):
        """Build the Zenodo metadata for one emulator ensemble.

        The description records what a prospective user needs in order to judge
        the ensemble without downloading it: how many members, what they were
        trained on, and how good they are.

        Parameters
        ----------
        model : str
            Model name.
        dataset : str
            Dataset identifier.
        manifests : list of Manifest
            The ensemble members being published.

        Returns
        -------
        dict
            Metadata in Zenodo's expected shape.
        """
        seeds = sorted(m.hyperparameters.seed for m in manifests)
        scores = [
            m.quality.get("abs_dlogp")
            for m in manifests
            if m.quality.get("abs_dlogp") is not None
        ]
        parameters = manifests[0].provenance.parameters if manifests else []
        quality = (
            f"Self-consistency |dlogp| ranges from {min(scores):.4f} to "
            f"{max(scores):.4f} across the ensemble. "
            if scores
            else "Ensemble members have not been scored. "
        )
        description = (
            f"Normalising-flow emulators of the {model} posterior for "
            f"{dataset}, trained on nested-sampling chains. "
            f"The ensemble holds {len(manifests)} independently seeded members "
            f"(seeds {seeds}), each emulating the marginal posterior over "
            f"{', '.join(parameters)}. {quality}"
            "Sampling an emulator reproduces the posterior without rerunning "
            "the inference that produced it."
        )
        return {
            "metadata": {
                "title": f"{TITLE_PREFIX} {model} {dataset}",
                "upload_type": "dataset",
                "description": description,
                "keywords": ["emulator", "normalising flow", model, dataset],
            }
        }

    def update_metadata(self, deposit_id, metadata):
        """Attach metadata to an existing deposit.

        Parameters
        ----------
        deposit_id : int
            The deposit to update.
        metadata : dict
            As produced by :meth:`describe`.

        Returns
        -------
        dict
            The updated deposit.
        """
        url = f"{self.base_url}/{deposit_id}"
        return self._request("PUT", url, json=metadata).json()

    def upload_file(self, deposit, path, name=None):
        """Upload one file into a deposit.

        Parameters
        ----------
        deposit : dict
            A deposit as returned by :meth:`create_deposit`.
        path : str or pathlib.Path
            Local file to upload.
        name : str, optional
            Name to store it under. Defaults to the file's own name. Ensemble
            members pass a path-like name so that members stay distinguishable
            within a single deposit.

        Returns
        -------
        dict
            Zenodo's description of the stored file.

        Raises
        ------
        KeyError
            If the deposit has no upload bucket.
        """
        path = Path(path)
        bucket = deposit.get("links", {}).get("bucket")
        if not bucket:
            raise KeyError("deposit has no upload bucket; was it created correctly?")
        with open(path, "rb") as handle:
            response = self._request(
                "PUT", f"{bucket}/{name or path.name}", data=handle
            )
        return response.json()

    def upload_emulator(self, deposit, emulator):
        """Upload one ensemble member, model and manifest together.

        Files are namespaced by the member's seed so that a single deposit can
        hold a whole ensemble without collisions, and so that a downloader can
        reconstruct the directory layout.

        Parameters
        ----------
        deposit : dict
            The target deposit.
        emulator : Emulator
            The ensemble member to upload.

        Returns
        -------
        list of dict
            Zenodo's description of each stored file.

        Raises
        ------
        FileNotFoundError
            If the emulator's model file is absent.
        """
        if not emulator.has_model:
            raise FileNotFoundError(
                f"{emulator.name} has a manifest but no model file at "
                f"{emulator.model_path}"
            )
        prefix = f"seed{emulator.manifest.hyperparameters.seed}"
        stored = [
            self.upload_file(
                deposit,
                emulator.directory / MANIFEST_NAME,
                name=f"{prefix}/{MANIFEST_NAME}",
            ),
            self.upload_file(
                deposit,
                emulator.model_path,
                name=f"{prefix}/{emulator.manifest.model_file}",
            ),
        ]
        return stored

    def publish(self, deposit_id):
        """Publish a deposit, minting its DOI.

        Publishing is irreversible. A published deposit cannot be deleted and
        its DOI is permanent, which is why this class defaults to the sandbox.

        Parameters
        ----------
        deposit_id : int
            The deposit to publish.

        Returns
        -------
        dict
            The published deposit, including ``doi`` and ``conceptdoi``.
        """
        url = f"{self.base_url}/{deposit_id}/actions/publish"
        return self._request("POST", url).json()

    def new_version(self, deposit_id):
        """Open a new version of an already published deposit.

        Retraining an ensemble produces a new version rather than a new
        deposit, so that the concept DOI continues to resolve to the latest
        emulators while each version stays individually citable.

        Parameters
        ----------
        deposit_id : int
            An already published deposit.

        Returns
        -------
        dict
            The draft deposit for the new version.
        """
        url = f"{self.base_url}/{deposit_id}/actions/newversion"
        return self._request("POST", url).json()

    def find_deposits(self, title):
        """Return the caller's own deposits matching a title.

        Parameters
        ----------
        title : str
            Title to search for.

        Returns
        -------
        list of dict
            Matching deposits belonging to the authenticated user.
        """
        response = self._request(
            "GET", self.base_url, params={"q": f'title:"{title}"', "size": 25}
        )
        return response.json()

    def discard_draft(self, deposit_id):
        """Delete an unpublished deposit.

        Only drafts can be removed; a published deposit is permanent. This
        exists so that an interrupted upload does not leave debris behind.

        Parameters
        ----------
        deposit_id : int
            The draft to delete.

        Returns
        -------
        bool
            True if the draft was deleted.
        """
        self._request("DELETE", f"{self.base_url}/{deposit_id}")
        return True

    def publish_ensemble(self, model, dataset, emulators, dry_run=False):
        """Create, populate and publish a deposit for one ensemble.

        Parameters
        ----------
        model : str
            Model name.
        dataset : str
            Dataset identifier.
        emulators : list of Emulator
            The ensemble members to publish.
        dry_run : bool, optional
            Build the metadata and validate the inputs without contacting
            Zenodo. Returns the metadata that would have been sent.

        Returns
        -------
        dict
            The published deposit, or the prepared metadata when ``dry_run``.

        Raises
        ------
        ValueError
            If no emulators were supplied.
        """
        emulators = list(emulators)
        if not emulators:
            raise ValueError(f"no emulators to publish for ({model!r}, {dataset!r})")

        metadata = self.describe(model, dataset, [e.manifest for e in emulators])
        if dry_run:
            return metadata

        deposit = self.create_deposit()
        for emulator in emulators:
            self.upload_emulator(deposit, emulator)
        self.update_metadata(deposit["id"], metadata)
        return self.publish(deposit["id"])

    def __repr__(self):
        """Return a description that never reveals the token."""
        where = "sandbox" if self.sandbox else "PRODUCTION"
        return f"<EmulatorPublisher {where}>"


__all__ = ["EmulatorPublisher"]
