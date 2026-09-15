"""Model-aware selection of the sampled cosmological parameters of a chain.

An emulator should be trained on the parameters a chain actually sampled, and
only those. A nested-sampling chain carries far more columns than that: derived
quantities (``As``, ``omegam``, ``sigma8``, ...), instrument and foreground
nuisance parameters, and per-sample bookkeeping (``logL``, ``weights``). Training
a flow on all of them would waste capacity on quantities that are deterministic
functions of the others or irrelevant to cosmology, and would change dimension
from one chain to the next in ways that are easy not to notice.

The grid spans many models -- LCDM plus one- or two-parameter extensions (dark
energy, curvature, neutrinos, tensors, running, lensing amplitude) -- so the set
of sampled parameters differs per chain. This module resolves that set from the
chain's own columns, intersected with a curated superset of *sampled* cosmological
parameters, so the same code trains the right flow for every model without a
per-model configuration table.

This module operates on column names alone and needs neither JAX nor margarine.
"""

# Superset of SAMPLED cosmological parameters across all grid models. Derived
# parameters (As, omegam, sigma8, S8, ...) and nuisance parameters are excluded
# by omission: the selection is the intersection of this list with a chain's
# columns, so a parameter absent here is never trained on.
COSMO_SAMPLED = [
    "logA", "ns", "H0", "ombh2", "omch2", "tau",   # base LCDM
    "w", "w0", "wa",                                # dark energy
    "mnu", "nnu", "meffsterile",                    # neutrinos
    "omk",                                          # curvature
    "r",                                            # tensors
    "nrun",                                         # running
    "Alens",                                        # lensing amplitude
]

# The six parameters present in every grid model. Their absence signals a naming
# mismatch (an alias not in COSMO_SAMPLED), which would otherwise yield a flow of
# the wrong dimension trained on the wrong columns -- so it is raised, not ignored.
BASE = ["logA", "ns", "H0", "ombh2", "omch2", "tau"]


def parameter_names(samples):
    """Return the flat list of column names of a samples object or column index.

    Parameters
    ----------
    samples : anesthetic.Samples or pandas.DataFrame or column index or sequence
        Either an object exposing a ``columns`` attribute (an anesthetic
        ``Samples`` or a ``DataFrame``), a pandas column ``Index`` /
        ``MultiIndex``, or a plain sequence of names. Anesthetic stores columns
        as a ``MultiIndex`` whose first level holds the parameter name; that
        level is what is returned.

    Returns
    -------
    list of str
        The parameter names, one per column, with any label/weight levels of a
        ``MultiIndex`` flattened away.
    """
    columns = getattr(samples, "columns", samples)
    get_level = getattr(columns, "get_level_values", None)
    if get_level is not None:
        return list(get_level(0))
    return [c[0] if isinstance(c, tuple) else c for c in columns]


def cosmological_parameters(samples):
    """The sampled cosmological parameters present in a chain, in canonical order.

    Parameters
    ----------
    samples : anesthetic.Samples or pandas.DataFrame or column index or sequence
        The chain, or anything :func:`parameter_names` accepts.

    Returns
    -------
    list of str
        The members of :data:`COSMO_SAMPLED` that appear in the chain, in the
        order they appear in :data:`COSMO_SAMPLED` (so the ordering is stable
        across chains and independent of column order).

    Raises
    ------
    ValueError
        If any of the six :data:`BASE` parameters is absent. Every grid model
        samples those, so a missing one means the chain names a parameter by an
        alias this module does not know -- which would silently train a flow of
        the wrong dimension. Failing here turns that into an immediate, legible
        error.
    """
    names = parameter_names(samples)
    present = set(names)
    missing = [p for p in BASE if p not in present]
    if missing:
        raise ValueError(
            f"chain is missing base cosmological parameters {missing} "
            f"(a naming alias not in COSMO_SAMPLED?); first columns seen: "
            f"{names[:15]}"
        )
    return [p for p in COSMO_SAMPLED if p in present]
