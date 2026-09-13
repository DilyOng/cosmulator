"""A searchable library of trained emulators on disk.

A production run of this package produces thousands of emulators: one for every
combination of model, dataset and random seed. That collection is only useful
if it can be interrogated, so the store treats a directory tree as a database
whose records are :class:`~cosmulator.manifest.Manifest` files.

Scanning and querying read manifests only. A store of ten thousand emulators
can therefore be filtered on a laptop with no JAX installed and no model file
touched; the cost is a few hundred bytes read per emulator rather than tens of
megabytes.
"""

from pathlib import Path

from cosmulator.manifest import MANIFEST_NAME, Manifest


class Emulator:
    """One trained emulator on disk.

    Wraps a manifest with the location it was found in, so that the model file
    can be resolved when it is eventually needed.

    Parameters
    ----------
    directory : str or pathlib.Path
        The emulator's own directory, containing its manifest and model file.
    manifest : Manifest
        The manifest read from that directory.
    """

    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.manifest = manifest

    @classmethod
    def read(cls, directory):
        """Load an emulator's manifest from its directory.

        The model file is not read. This is the cheap operation that makes a
        large store searchable.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory containing the manifest.

        Returns
        -------
        Emulator
            The emulator described there.
        """
        directory = Path(directory)
        return cls(directory, Manifest.read(directory))

    @property
    def name(self):
        """str: The emulator's identifier within its store."""
        return self.manifest.name

    @property
    def model_path(self):
        """pathlib.Path: Absolute path to the model file.

        The file is not required to exist; a manifest may describe an emulator
        whose model has been archived or has yet to finish training. Use
        :attr:`has_model` to check.
        """
        return self.directory / self.manifest.model_file

    @property
    def has_model(self):
        """bool: Whether the model file is actually present."""
        return self.model_path.is_file()

    @property
    def quality(self):
        """Quality: The emulator's recorded quality metrics."""
        return self.manifest.quality

    def __repr__(self):
        """Return a short, unambiguous description."""
        return f"<Emulator {self.name!r} at {self.directory}>"


class EmulatorStore:
    """A directory tree of trained emulators.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to search. Emulators may be nested to any depth beneath it,
        which lets a store be organised by model and dataset without the store
        itself imposing a layout.

    Examples
    --------
    >>> store = EmulatorStore("emulators")           # doctest: +SKIP
    >>> len(store)                                   # doctest: +SKIP
    4000
    >>> good = store.filter(lambda e: e.quality.get("abs_dlogp", 1) < 0.05)
    ... # doctest: +SKIP
    """

    def __init__(self, root):
        self.root = Path(root)
        self._emulators = None

    def scan(self, force=False):
        """Find every emulator beneath the root.

        Directories containing a manifest are treated as emulators; everything
        else is ignored, so a store may sit alongside logs and scratch files.
        A manifest that cannot be read is skipped rather than aborting the
        scan, because one corrupt record should not make the rest of a large
        store unreadable.

        Parameters
        ----------
        force : bool, optional
            Rescan even if a previous scan is cached.

        Returns
        -------
        list of Emulator
            Every emulator found, ordered by name.
        """
        if self._emulators is not None and not force:
            return self._emulators

        found = []
        for manifest_path in sorted(self.root.rglob(MANIFEST_NAME)):
            try:
                found.append(Emulator.read(manifest_path.parent))
            except (ValueError, OSError):
                continue
        self._emulators = sorted(found, key=lambda e: e.name)
        return self._emulators

    def filter(self, predicate):
        """Select emulators matching a predicate.

        Parameters
        ----------
        predicate : callable
            Called with each :class:`Emulator`; those for which it returns a
            true value are kept.

        Returns
        -------
        list of Emulator
            The matching emulators, in name order.
        """
        return [e for e in self.scan() if predicate(e)]

    def best(self, metric, n=1, largest=False):
        """Return the emulators scoring best on a quality metric.

        Emulators lacking the metric are excluded rather than sorted to one
        end, so an unvalidated emulator is never mistaken for a good one.

        Parameters
        ----------
        metric : str
            Name of the metric, as recorded in the manifest's quality mapping.
        n : int, optional
            How many to return.
        largest : bool, optional
            Return the highest scores instead of the lowest. Most fidelity
            metrics are errors, where smaller is better, so the default is to
            return the smallest.

        Returns
        -------
        list of Emulator
            Up to ``n`` emulators, best first.
        """
        scored = [(e.quality.get(metric), e) for e in self.scan()]
        scored = [(v, e) for v, e in scored if v is not None]
        scored.sort(key=lambda pair: pair[0], reverse=largest)
        return [e for _, e in scored[:n]]

    def __len__(self):
        """Return the number of emulators in the store."""
        return len(self.scan())

    def __iter__(self):
        """Iterate over the emulators in name order."""
        return iter(self.scan())

    def __getitem__(self, name):
        """Return one emulator by name.

        Parameters
        ----------
        name : str
            The emulator's manifest name.

        Returns
        -------
        Emulator
            The matching emulator.

        Raises
        ------
        KeyError
            If no emulator in the store has that name.
        """
        for emulator in self.scan():
            if emulator.name == name:
                return emulator
        raise KeyError(name)

    def __repr__(self):
        """Return a short, unambiguous description."""
        count = len(self._emulators) if self._emulators is not None else "unscanned"
        return f"<EmulatorStore {self.root} ({count})>"
