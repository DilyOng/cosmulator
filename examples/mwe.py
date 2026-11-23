#!/usr/bin/env python3
r"""Minimum working example: train a cosmulator emulator on a synthetic posterior.

This script needs no cluster, no downloaded chain and no private data. It builds a
*synthetic* weighted posterior with a known, deliberately non-Gaussian shape -- a
curved :math:`w_0`--:math:`w_a` "banana", the degeneracy a MAF exists to capture --
then trains a normalising-flow emulator of it and certifies the result, end to end,
on a laptop CPU in a couple of minutes.

It is written to be read as documentation as much as run: it shows the smallest
real call sequence a new user needs, and the exact object a chain has to look like.

What it demonstrates
--------------------
1. The target. A weighted sample set standing in for a nested-sampling posterior:
   draws from a broad proposal, carrying importance weights ``p_target / q_proposal``
   -- so the *weighted* training path is exercised, exactly as for a real chain.
2. ``train_maf_emulator`` -- train one spline-MAF and read its certification verdict.
3. ``EnsembleEmulator.train`` -- the production deliverable (a ``K``-member mixture),
   and ``.certify`` against the same target. (Skip with ``--single``.)

The emulator is trained on whatever parameters you name; here the chain carries two
columns, ``w0`` and ``wa``. A chain only has to support ``chain[params].to_numpy()``
and ``chain.get_weights()`` -- an :class:`anesthetic.Samples` does, and so does the
minimal :class:`_Chain` shim below, which keeps this example free of a heavy
dependency the package itself does not require.

Usage
-----
::

    pip install "cosmulator[train,hpo]"     # JAX stack (+ Optuna, unused here)
    python examples/mwe.py                  # single flow + K=4 ensemble
    python examples/mwe.py --single --plot  # one flow only, and save a corner PNG

Requires the optional ``[train]`` extra (JAX, flowjax, equinox, optax). Without it,
the training call raises an informative ``ImportError`` -- the data-generation and
chain plumbing above it still run, so the example also doubles as a plumbing check.
"""

import argparse

import numpy as np

from cosmulator.maf import train_maf_emulator


def _normal_logpdf(x, mean, std):
    """Log density of a 1-D Gaussian, elementwise."""
    z = (x - mean) / std
    return -0.5 * z * z - np.log(std) - 0.5 * np.log(2.0 * np.pi)


class _Chain:
    """Minimal stand-in for an :class:`anesthetic.Samples`.

    The trainer only ever asks a chain for two things: the parameter columns, via
    ``chain[names].to_numpy()``, and the posterior weights, via ``get_weights()``.
    This duck-typed shim provides exactly those, so the example needs no anesthetic
    (which cosmulator does not depend on). A real analysis passes an
    ``anesthetic.Samples`` here instead -- nothing else changes.
    """

    def __init__(self, data, columns, weights):
        self._data = np.asarray(data, dtype=np.float64)   # shape (n, d)
        self._columns = list(columns)
        self._weights = np.asarray(weights, dtype=np.float64)

    def __getitem__(self, names):
        idx = [self._columns.index(n) for n in names]
        return _Columns(self._data[:, idx])

    def get_weights(self):
        return self._weights


class _Columns:
    """What ``chain[names]`` returns: a view exposing ``.to_numpy()``."""

    def __init__(self, array):
        self._array = array

    def to_numpy(self):
        return self._array


def make_banana(n=20000, seed=0):
    r"""A synthetic weighted :math:`w_0`--:math:`w_a` posterior.

    The target is a curved, correlated density
    ``p(w0) * p(wa | w0)`` with
    ``w0 ~ N(-1, 0.30)`` and ``wa | w0 ~ N(mu(w0), 0.40)``,
    where ``mu(w0) = -3.2 (w0 + 1) + 0.8 (w0 + 1)^2`` bends the ridge into a banana.

    It is delivered as *weighted* samples (importance weights against a broad
    Gaussian proposal), mimicking a nested-sampling posterior whose points carry
    unequal weights -- the setting the trainer is built for.

    Returns
    -------
    chain : _Chain
        Columns ``("w0", "wa")`` and importance weights.
    bounds : ndarray, shape (2, 2)
        A prior box enclosing the mass, for the out-of-bounds gate and the
        ensemble's deployed truncation.
    """
    rng = np.random.default_rng(seed)

    # Broad proposal we can draw from cheaply, deliberately wider than the target.
    w0 = rng.normal(-1.0, 0.60, n)
    wa = rng.normal(0.0, 1.50, n)

    # Target density, known in closed form -> exact importance weights.
    mu = -3.2 * (w0 + 1.0) + 0.8 * (w0 + 1.0) ** 2
    log_p = _normal_logpdf(w0, -1.0, 0.30) + _normal_logpdf(wa, mu, 0.40)
    log_q = _normal_logpdf(w0, -1.0, 0.60) + _normal_logpdf(wa, 0.0, 1.50)
    log_w = log_p - log_q
    weights = np.exp(log_w - log_w.max())

    theta = np.column_stack([w0, wa])
    core = weights > weights.max() * 1e-4           # drop negligible-weight tails
    lo = theta[core].min(axis=0) - 0.3
    hi = theta[core].max(axis=0) + 0.3
    bounds = np.column_stack([lo, hi])

    ess = weights.sum() ** 2 / np.sum(weights ** 2)
    print(f"[target]  n={n}  effective sample size ~ {ess:,.0f}  "
          f"(weighted, non-Gaussian banana)")
    return _Chain(theta, ("w0", "wa"), weights), bounds


def report_verdict(tag, cert):
    """Print the headline line of a certification report.

    Keys mirror :func:`cosmulator.validation.certify`: the mean marginal-width
    error and mean bias live under ``moments``, the worst 1-D Wasserstein distance
    under ``marginals``, and ``reasons`` explains a non-pass verdict.
    """
    if cert is None:
        print(f"[{tag}]  certification skipped")
        return
    moments = cert.get("moments", {})
    marginals = cert.get("marginals", {})
    extra = []
    width = moments.get("mean_abs_width_err_pct")
    bias = moments.get("mean_abs_bias_sigma")
    wass = marginals.get("max_wasserstein_sigma")
    if width is not None:
        extra.append(f"mean width err {width:.3f}%")
    if bias is not None:
        extra.append(f"bias {bias:.3f} sigma")
    if wass is not None:
        extra.append(f"worst Wasserstein {wass:.3f} sigma")
    print(f"[{tag}]  verdict = {cert.get('verdict', '?').upper()}"
          + ("   " + ", ".join(extra) if extra else ""))
    reasons = cert.get("reasons")
    if reasons:
        print(f"[{tag}]  reasons: " + "; ".join(reasons))


def maybe_plot(chain, phys, path):
    """Scatter the target against emulator draws, if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot]  matplotlib not installed; skipping figure")
        return
    theta = chain[("w0", "wa")].to_numpy()
    w = chain.get_weights()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.scatter(theta[:, 0], theta[:, 1], s=4, alpha=0.15,
               c=w / w.max(), cmap="Greys", label="target (weighted)")
    ax.scatter(phys[:, 0], phys[:, 1], s=2, alpha=0.08,
               color="#0072B2", label="emulator draws")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$w_a$")
    ax.set_title("cosmulator MWE: synthetic banana vs emulator")
    ax.legend(loc="upper right", markerscale=3, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[plot]  wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", action="store_true",
                        help="train one flow only (skip the ensemble)")
    parser.add_argument("--epochs", type=int, default=400,
                        help="max training epochs (default 400)")
    parser.add_argument("--n", type=int, default=20000,
                        help="number of target samples (default 20000)")
    parser.add_argument("--plot", action="store_true",
                        help="save a target-vs-emulator scatter PNG")
    args = parser.parse_args()

    chain, bounds = make_banana(n=args.n)
    params = ["w0", "wa"]

    # --- (1) a single spline-MAF, certified against the target ------------------
    print("\n[single] training one spline-MAF ...")
    result = train_maf_emulator(
        chain,
        parameters=params,
        spline=True,          # the banana is strongly curved: use the spline MAF
        flow_layers=6,
        nn_width=32,
        nn_depth=2,
        epochs=args.epochs,
        patience=60,
        bounds=bounds,
        certify=True,
    )
    report_verdict("single", result["certification"])

    if args.plot:
        import jax
        flow, mean, std = result["flow"], result["mean"], result["std"]
        draws = np.asarray(flow.sample(jax.random.key(0), (20000,))) * std + mean
        maybe_plot(chain, draws, "examples/mwe_corner.png")

    # --- (2) the production K-member ensemble (the real deliverable) -------------
    if not args.single:
        from cosmulator.ensemble import EnsembleEmulator

        print("\n[ensemble] training a K=4 mixture (production object) ...")
        ens = EnsembleEmulator.train(
            chain,
            parameters=params,
            K=4,
            bounds=bounds,
            spline=True,
            flow_layers=6,
            nn_width=32,
            nn_depth=2,
            epochs=args.epochs,
            patience=60,
        )
        print(f"[ensemble] K={ens.K} members "
              f"({ens.invalid_runs} invalid runs replaced)")
        report_verdict("ensemble", ens.certify(chain))

    print("\nDone. In a real analysis, replace make_banana() with an "
          "anesthetic.Samples chain read from disk.")


if __name__ == "__main__":
    main()
