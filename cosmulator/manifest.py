"""The on-disk description of a single trained emulator.

An emulator is expensive to produce and cheap to describe. Training one of the
flows in this package takes GPU-hours; the record of what was trained, from
what, with which hyperparameters, and how well it turned out, is a few hundred
bytes of JSON. Keeping those two things separate is the central design decision
of the storage layer.

The consequence is that a library of thousands of emulators can be scanned,
filtered and compared without loading a single model, and therefore without
requiring JAX. Only sampling from an emulator needs the training stack.

Nothing in this module imports the optional training dependencies, and
``tests/test_packaging.py`` enforces that.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

#: Version of the manifest layout written by this package.
#:
#: Bumped whenever the schema changes in a way that older readers cannot
#: interpret. :func:`Manifest.from_dict` refuses anything newer than it knows
#: about, so a future artefact fails loudly rather than being silently
#: misread.
SCHEMA_VERSION = 1

#: Name of the manifest file inside an emulator directory.
MANIFEST_NAME = "emulator.json"


@dataclass(frozen=True)
class Provenance:
    """Where the training data came from.

    Parameters
    ----------
    source : str
        Identifier for the posterior samples the emulator was trained on. A
        path, a URL or a database key; the package does not interpret it.
    parameters : list of str
        Names of the parameters the flow was trained on, in order. This is the
        input dimension of the emulator and the column order its samples are
        returned in.
    n_samples : int
        Number of samples in the source chain before any resampling.
    n_effective : int
        Effective sample size after weighting. Nested sampling chains carry
        many near-zero-weight points, so this is typically far smaller than
        ``n_samples`` and is the number the flow actually learns from.
    """

    source: str
    parameters: list
    n_samples: int
    n_effective: int


@dataclass(frozen=True)
class Hyperparameters:
    """The architecture and optimiser settings used for training.

    Stored separately from timing and quality so that a search over
    architectures can be reconstructed from a directory of emulators without
    re-reading training logs.

    Parameters
    ----------
    seed : int
        Random seed. Recorded because an ensemble of independently seeded
        flows is the package's unit of epistemic uncertainty; without the seed
        an ensemble member cannot be reproduced or identified.
    extra : dict, optional
        Estimator-specific settings. Keeping these in a free-form mapping means
        a new flow architecture can be stored without a schema change.
    """

    seed: int
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Training:
    """What happened when the emulator was trained.

    Parameters
    ----------
    seconds : float
        Wall-clock training time, excluding data loading. This is the quantity
        that compute allocations are budgeted in, so it is recorded for every
        emulator rather than inferred from logs.
    backend : str
        The JAX backend used, typically ``'gpu'`` or ``'cpu'``. Timings are not
        comparable across backends.
    load_seconds : float, optional
        Time spent reading and resampling the source chain.
    """

    seconds: float
    backend: str
    load_seconds: float = 0.0


@dataclass(frozen=True)
class Quality:
    """How faithfully the emulator reproduces its target.

    Parameters
    ----------
    metrics : dict
        Named quality metrics. Free-form because the right measure of emulator
        fidelity is an open question and should not be frozen into the schema.
    """

    metrics: dict = field(default_factory=dict)

    def get(self, name, default=None):
        """Return one metric by name.

        Parameters
        ----------
        name : str
            Metric name.
        default : object, optional
            Returned when the metric is absent.

        Returns
        -------
        object
            The metric value, or ``default``.
        """
        return self.metrics.get(name, default)


@dataclass(frozen=True)
class Manifest:
    """The complete description of one trained emulator.

    Parameters
    ----------
    name : str
        Identifier for this emulator, unique within its store.
    estimator : str
        The kind of density estimator, for example ``'RealNVP'``.
    model_file : str
        Name of the model file, relative to the manifest's own directory.
    provenance : Provenance
        What it was trained on.
    hyperparameters : Hyperparameters
        How it was configured.
    training : Training
        What training cost.
    quality : Quality, optional
        How good it is. Absent until the emulator has been validated.
    schema_version : int, optional
        Layout version; defaults to the current :data:`SCHEMA_VERSION`.
    """

    name: str
    estimator: str
    model_file: str
    provenance: Provenance
    hyperparameters: Hyperparameters
    training: Training
    quality: Quality = field(default_factory=Quality)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self):
        """Return the manifest as a JSON-serialisable dictionary.

        Returns
        -------
        dict
            Nested plain-Python representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Rebuild a manifest from its dictionary form.

        Parameters
        ----------
        data : dict
            As produced by :meth:`to_dict`.

        Returns
        -------
        Manifest
            The reconstructed manifest.

        Raises
        ------
        ValueError
            If the manifest was written by a newer version of the schema than
            this package understands, or if a required field is missing.
        """
        version = data.get("schema_version", 0)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema version {version} is newer than this version "
                f"of cosmulator understands ({SCHEMA_VERSION}); upgrade the "
                f"package to read it"
            )
        known = {f.name for f in fields(cls)}
        missing = known - {"quality", "schema_version"} - set(data)
        if missing:
            raise ValueError(f"manifest is missing required fields: {sorted(missing)}")
        return cls(
            name=data["name"],
            estimator=data["estimator"],
            model_file=data["model_file"],
            provenance=Provenance(**data["provenance"]),
            hyperparameters=Hyperparameters(**data["hyperparameters"]),
            training=Training(**data["training"]),
            quality=Quality(**data.get("quality", {})),
            schema_version=version,
        )

    def write(self, directory):
        """Write the manifest into an emulator directory.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory to write :data:`MANIFEST_NAME` into. Created if needed.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False))
        return path

    @classmethod
    def read(cls, directory):
        """Read the manifest from an emulator directory.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory containing :data:`MANIFEST_NAME`.

        Returns
        -------
        Manifest
            The manifest found there.

        Raises
        ------
        FileNotFoundError
            If the directory contains no manifest.
        """
        path = Path(directory) / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"no {MANIFEST_NAME} in {directory}")
        return cls.from_dict(json.loads(path.read_text()))
