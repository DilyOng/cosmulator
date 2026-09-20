"""Certify that a trained emulator matches its target posterior.

Where :mod:`cosmulator.diagnostics` provides individual measures, this module
turns them into a **decision**: given samples from a target posterior ``P`` (with
nested-sampling weights) and samples from an emulator ``q``, it runs a battery of
checks and returns a verdict of ``"pass"``, ``"warn"``, ``"insufficient"`` or
``"fail"``, with the numbers behind it.

Two ideas shape the design.

*Certification is tolerance-based, not a hypothesis test.* Any consistent
two-sample test rejects an emulator that is not *exactly* the target once it is
given enough samples, and no emulator is ever exactly the target. So the pass/fail
decision is made against **tolerances** on interpretable discrepancies (marginal
width and bias, out-of-bounds mass), chosen for the downstream science. The joint
two-sample test (MMD) is reported and used to *warn*, not to hard-fail, precisely
because its p-value is guaranteed to shrink with sample size.

*The target is a weighted empirical posterior, not the exact one.* ``P`` is the
weighted output of one nested-sampling run, with an effective sample size (ESS)
far below its row count. Below a minimum ESS no test has the power to certify
anything, so the verdict is ``"insufficient"`` rather than a false ``"pass"``.

Everything here is pure NumPy: no JAX, no classifier library. Given two arrays of
samples it runs on a laptop, so a stored emulator can be re-certified cheaply.
"""

import numpy as np

from cosmulator.diagnostics import (
    _normalised_weights,
    effective_sample_size,
    moment_error,
)

# Default acceptance tolerances. These are scientific choices, deliberately
# surfaced here to be overridden per grid: what counts as "close enough" depends
# on the downstream use of the posterior, not on the statistics.
DEFAULT_TOLERANCES = {
    "width_pct": 1.0,        # mean |sigma_q - sigma_P| / sigma_P, per cent
    "bias_sigma": 0.1,       # mean |mean_q - mean_P| / sigma_P
    "wasserstein_sigma": 0.05,  # worst-parameter 1D Wasserstein, in target sigma
    "oob_frac": 0.001,       # fraction of emulator mass outside the prior box
    "mmd_pvalue": 0.01,      # below this the joint test flags a difference (warn)
}

# Below this effective sample size the target is too poorly resolved to certify.
# A heuristic, not a theorem: it is the ESS at which the two-sample tests here
# start to lose the power to detect a discrepancy at the tolerances above.
DEFAULT_MIN_ESS = 200.0


def _weighted_cdf(samples, weights, grid):
    """Evaluate the weighted empirical CDF of ``samples`` on ``grid``."""
    order = np.argsort(samples)
    s = samples[order]
    cum = np.cumsum(weights[order])
    # Fraction of weight at or below each grid point.
    idx = np.searchsorted(s, grid, side="right")
    return np.concatenate([[0.0], cum])[idx]


def _w1_ks(a, wa, b, wb):
    """1D Wasserstein-1 distance and KS statistic between two weighted samples."""
    grid = np.sort(np.concatenate([a, b]))
    fa = _weighted_cdf(a, wa, grid)
    fb = _weighted_cdf(b, wb, grid)
    ks = float(np.max(np.abs(fa - fb)))
    # Wasserstein-1 = integral of |F_a - F_b| dx (area between the two CDFs).
    dx = np.diff(grid)
    w1 = float(np.sum(np.abs(fa[:-1] - fb[:-1]) * dx))
    return w1, ks


def marginal_gof(target_samples, emulated_samples, weights=None):
    """Per-parameter 1D goodness-of-fit beyond the first two moments.

    :func:`cosmulator.diagnostics.moment_error` checks mean and variance; a flow
    can match both while missing skewness, a heavy tail or a hard cutoff. The
    weighted 1D Wasserstein-1 distance and Kolmogorov-Smirnov statistic compare
    the *whole* marginal shape. Wasserstein is reported in units of the target's
    own standard deviation, so a single tolerance applies across parameters.

    Parameters
    ----------
    target_samples : array_like, shape (n, d)
        Samples from the target distribution.
    emulated_samples : array_like, shape (m, d)
        Samples from the emulator.
    weights : array_like, optional
        Weights for the target samples.

    Returns
    -------
    dict
        Per-parameter ``wasserstein_sigma`` and ``ks`` arrays, and the worst
        (maximum) of each across parameters.
    """
    target = np.atleast_2d(np.asarray(target_samples, dtype=float))
    emulated = np.atleast_2d(np.asarray(emulated_samples, dtype=float))
    if target.shape[1] != emulated.shape[1]:
        raise ValueError(
            f"dimension mismatch: target has {target.shape[1]} parameters, "
            f"emulated has {emulated.shape[1]}"
        )
    d = target.shape[1]
    wt = _normalised_weights(weights, target.shape[0])
    we = np.full(emulated.shape[0], 1.0 / emulated.shape[0])

    target_mean = wt @ target
    target_std = np.sqrt(wt @ (target - target_mean) ** 2)
    scale = np.maximum(np.abs(target_mean), 1.0)
    if not np.all(target_std > 1e-12 * scale):
        raise ValueError("a target parameter has zero spread")

    w1_sigma, ks = np.empty(d), np.empty(d)
    for j in range(d):
        w1, k = _w1_ks(target[:, j], wt, emulated[:, j], we)
        w1_sigma[j] = w1 / target_std[j]
        ks[j] = k
    return {
        "wasserstein_sigma": w1_sigma.tolist(),
        "ks": ks.tolist(),
        "max_wasserstein_sigma": float(np.max(w1_sigma)),
        "max_ks": float(np.max(ks)),
    }


def out_of_bounds(emulated_samples, bounds):
    """Fraction of emulator samples falling outside the prior box.

    A normalising flow has infinite support, but a posterior lives inside its
    hard prior boundaries (a density for a positive quantity must not leak below
    zero). This reports the fraction of emulator draws with any coordinate
    outside ``bounds``; a nonzero value is unphysical mass the flow invented.

    Parameters
    ----------
    emulated_samples : array_like, shape (m, d)
        Samples from the emulator.
    bounds : array_like, shape (d, 2)
        Per-parameter ``(lower, upper)`` prior limits. Use ``-inf``/``inf`` for
        an unbounded parameter.

    Returns
    -------
    float
        Fraction of samples outside the box, in ``[0, 1]``.
    """
    emulated = np.atleast_2d(np.asarray(emulated_samples, dtype=float))
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (emulated.shape[1], 2):
        raise ValueError(
            f"bounds have shape {bounds.shape}, expected ({emulated.shape[1]}, 2)"
        )
    lower, upper = bounds[:, 0], bounds[:, 1]
    inside = np.all((emulated >= lower) & (emulated <= upper), axis=1)
    return float(1.0 - inside.mean())


def _mmd2(kernels, a, b):
    """Unbiased weighted MMD^2 summed over kernels, for weight vectors ``a``, ``b``.

    ``a`` and ``b`` are length-``N`` weight vectors over the *pooled* point set
    (each nonzero only on its own group), so a permutation is just a rewrite of
    ``a`` and ``b`` and never recomputes a kernel. The kernels have their diagonal
    zeroed, so ``a.K.a`` is the off-diagonal sum ``sum_{i!=j} a_i a_j k_ij`` whose
    weights total ``1 - ||a||^2`` rather than 1; dividing by ``1 - ||a||^2`` makes
    it an unbiased estimate of ``E[k(X,X')]`` and, crucially, removes the
    dependence on within-group weight concentration (``||a||^2 = 1/ESS``) that
    would otherwise differ between the skewed target and the uniform emulator and
    corrupt the permutation null. The cross-term weights already sum to 1.
    """
    na = 1.0 - float(a @ a)
    nb = 1.0 - float(b @ b)
    if na <= 0 or nb <= 0:  # a group with a single effective point
        return float("nan")
    total = 0.0
    for k in kernels:
        total += a @ (k @ a) / na + b @ (k @ b) / nb - 2.0 * a @ (k @ b)
    return float(total)


def mmd(target_samples, emulated_samples, weights=None, *,
        max_points=1000, n_permutations=200, seed=0):
    """Maximum Mean Discrepancy joint two-sample test (multi-scale, weighted).

    MMD measures how far apart two sample clouds are across all smooth features
    at once, so unlike per-parameter checks it sees correlation and joint-shape
    errors. It needs only samples, handles the target weights natively (no
    resampling, which would create duplicate points a test could exploit), and
    is deterministic given the seed. A characteristic (RBF) kernel is used at
    several bandwidths around the median heuristic, since a single bandwidth can
    be blind to structure at other scales.

    Both sample sets are standardised by the target's weighted moments first, so
    the isotropic kernel sees comparable scales per parameter. The ``mmd2``
    statistic is always returned as a discrepancy score. A **p-value** is returned
    **only when the target weights are uniform**, via a label-permutation null,
    which is valid there because the pooled points are exchangeable under H0
    (q == P). For a *weighted* target -- the usual nested-sampling case -- neither
    a label permutation (points are not exchangeable, weighted vs unweighted) nor a
    naive sign-flip wild bootstrap gives a calibrated null (both were found to be
    strongly conservative in calibration experiments, because the statistic is not
    degenerate under H0 for skewed weights and the RBF kernel's constant offset is
    not removed). So ``pvalue`` is ``None`` for weighted targets; a validated
    weighted null (a properly centred multiplier or spectral construction) is
    deferred rather than shipped uncalibrated.

    Parameters
    ----------
    target_samples : array_like, shape (n, d)
        Samples from the target distribution.
    emulated_samples : array_like, shape (m, d)
        Samples from the emulator.
    weights : array_like, optional
        Weights for the target samples.
    max_points : int
        Cap on samples per set to keep the kernel matrices affordable. The target
        is truncated to its highest-weight points (mass-preserving); raise it if
        the target's effective sample size exceeds it, or the joint test sees only
        the posterior peak.
    n_permutations : int
        Number of label permutations for the (uniform-weight) null. ``0`` skips
        the p-value.
    seed : int
        Seed for subsampling and permutations.

    Returns
    -------
    dict
        ``mmd2`` (the observed statistic), ``pvalue`` (``None`` for weighted
        targets or when ``n_permutations`` is 0), ``n_target``/``n_emulated``
        actually used, and ``retained_target_mass`` (the target posterior mass
        kept after truncation to ``max_points``; below 1 the joint test sees only
        the highest-weight region).
    """
    rng = np.random.default_rng(seed)
    target = np.atleast_2d(np.asarray(target_samples, dtype=float))
    emulated = np.atleast_2d(np.asarray(emulated_samples, dtype=float))
    if target.shape[1] != emulated.shape[1]:
        raise ValueError("dimension mismatch between target and emulated samples")
    wt = _normalised_weights(weights, target.shape[0])
    # Whether the target weights are (essentially) uniform decides both how to
    # subsample it and whether the permutation null is valid (see below).
    uniform = np.allclose(wt, wt[0])

    # Standardise by the target's weighted moments.
    mean = wt @ target
    std = np.sqrt(wt @ (target - mean) ** 2)
    std = np.where(std > 0, std, 1.0)
    x = (target - mean) / std
    y = (emulated - mean) / std

    def subsample(a, w, weighted):
        if a.shape[0] <= max_points:
            return a, w / w.sum(), 1.0
        if weighted:
            # Keep the highest-weight points: a mass-preserving truncation (the
            # trainer's tail-filter idea). Uniform subsampling of a skewed chain
            # is unbiased only in expectation and usually discards the few points
            # carrying the mass. ``retained`` reports the target mass kept.
            idx = np.argsort(w)[-max_points:]
        else:
            # Uniform weights: an i.i.d. sample, so a random subset is
            # representative and reproducible via the seed (argsort would pick an
            # arbitrary order-dependent subset).
            idx = rng.choice(a.shape[0], size=max_points, replace=False)
        retained = float(w[idx].sum() / w.sum())
        return a[idx], w[idx] / w[idx].sum(), retained

    x, wx, retained_mass = subsample(x, wt, weighted=not uniform)
    y, wy, _ = subsample(y, np.full(y.shape[0], 1.0), weighted=False)
    nx, ny = x.shape[0], y.shape[0]

    # Pool the two sets and build each RBF kernel matrix ONCE. Permutations then
    # only reshuffle the weight vectors, never recompute a kernel.
    pooled = np.vstack([x, y])
    d2 = (np.sum(pooled * pooled, 1)[:, None] + np.sum(pooled * pooled, 1)[None, :]
          - 2.0 * pooled @ pooled.T)
    d2 = np.maximum(d2, 0.0)
    median = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    kernels = [np.exp(-(1.0 / (s * median)) * d2) for s in (0.5, 1.0, 2.0)]
    # Zero the diagonal for the UNBIASED MMD^2. The biased estimator keeps the
    # self-terms sum(w_i^2) k(x_i,x_i) = sum(w_i^2) = 1/ESS, which is large for the
    # skewed target and small for the uniform emulator; permutations mix the
    # weights and change that term, giving the observed statistic a structural
    # boost the null lacks and hence spuriously tiny p-values. The groups are
    # disjoint, so the cross-term a.K.b never touches the diagonal.
    for k in kernels:
        np.fill_diagonal(k, 0.0)

    # Observed statistic: group A = target, group B = emulator.
    a = np.concatenate([wx, np.zeros(ny)])
    b = np.concatenate([np.zeros(nx), wy])
    obs = _mmd2(kernels, a, b)
    base = {"mmd2": obs, "n_target": int(nx), "n_emulated": int(ny),
            "retained_target_mass": retained_mass}

    # A calibrated p-value needs the pooled points to be EXCHANGEABLE under H0,
    # which holds when both samples are i.i.d. draws from the same distribution --
    # i.e. uniform target weights. When the target is a *weighted* nested-sampling
    # measure and the emulator is unweighted i.i.d. draws, the two are not
    # exchangeable, and a naive sign-flip wild bootstrap does not fix it (the
    # statistic is non-degenerate under H0 for skewed weights, so it comes out
    # badly miscalibrated). So for a weighted target we report the statistic but no
    # p-value, pending a properly centred weighted null.
    if n_permutations <= 0 or not uniform:
        return {**base, "pvalue": None}

    wpool = np.concatenate([wx, wy])
    ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(nx + ny)
        ia, ib = perm[:nx], perm[nx:]
        pa = np.zeros(nx + ny)
        pb = np.zeros(nx + ny)
        pa[ia] = wpool[ia] / wpool[ia].sum()
        pb[ib] = wpool[ib] / wpool[ib].sum()
        if _mmd2(kernels, pa, pb) >= obs:
            ge += 1
    pvalue = (ge + 1) / (n_permutations + 1)
    return {**base, "pvalue": float(pvalue)}


def certify(target_samples, emulated_samples, weights=None, *,
            bounds=None, tolerances=None, min_ess=DEFAULT_MIN_ESS, seed=0):
    """Run the full validation battery and return a tolerance-based verdict.

    Parameters
    ----------
    target_samples : array_like, shape (n, d)
        Samples from the target posterior. Ideally a held-out split the emulator
        did not train on, so this measures generalisation, not memorisation.
    emulated_samples : array_like, shape (m, d)
        Samples drawn from the trained emulator.
    weights : array_like, optional
        Weights for the target samples (nested-sampling posterior weights).
    bounds : array_like, shape (d, 2), optional
        Prior limits per parameter; enables the out-of-bounds gate.
    tolerances : dict, optional
        Overrides for :data:`DEFAULT_TOLERANCES`.
    min_ess : float
        Below this target ESS the verdict is ``"insufficient"``.
    seed : int
        Seed for the MMD subsampling and permutation null.

    Returns
    -------
    dict
        ``verdict`` (``"pass"``/``"warn"``/``"insufficient"``/``"fail"``),
        ``reasons`` (list of strings), ``ess``, and every metric computed
        (``moments``, ``marginals``, ``mmd``, and ``oob_frac`` when ``bounds``
        given), so a verdict can be audited without recomputing.

    Notes
    -----
    The decision order is: insufficient ESS first (cannot certify, and computed
    before the other metrics since a degenerate target can make them raise); then
    hard fails on the interpretable tolerances and unphysical out-of-bounds mass;
    then a warn if the joint MMD test still detects a difference the marginals
    passed; otherwise pass. The MMD p-value never hard-fails, because it is
    guaranteed to shrink as sample size grows even for a good emulator, and it only
    warns when a valid p-value exists (uniform-weight targets). For a weighted
    target MMD is reported as a discrepancy but does not yet drive the verdict,
    pending a validated weighted null.
    """
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)

    n = np.atleast_2d(np.asarray(target_samples, dtype=float)).shape[0]
    ess = effective_sample_size(weights) if weights is not None else float(n)

    # ESS gate FIRST: a degenerate low-ESS target may make the metrics themselves
    # ill-defined (a near-constant parameter makes moment_error raise), so certify
    # must return "insufficient" before computing anything.
    report = {"ess": ess, "tolerances": tol}
    if ess < min_ess:
        report["verdict"] = "insufficient"
        report["reasons"] = [
            f"target ESS {ess:.0f} < {min_ess:.0f}: too poorly resolved to certify"
        ]
        return report

    report["moments"] = moment_error(target_samples, emulated_samples, weights)
    report["marginals"] = marginal_gof(target_samples, emulated_samples, weights)
    report["mmd"] = mmd(target_samples, emulated_samples, weights, seed=seed)
    if bounds is not None:
        report["oob_frac"] = out_of_bounds(emulated_samples, bounds)

    reasons = []
    width = report["moments"]["mean_abs_width_err_pct"]
    bias = report["moments"]["mean_abs_bias_sigma"]
    wass = report["marginals"]["max_wasserstein_sigma"]
    if width > tol["width_pct"]:
        reasons.append(f"width error {width:.2f}% > {tol['width_pct']}%")
    if bias > tol["bias_sigma"]:
        reasons.append(f"bias {bias:.3f} sigma > {tol['bias_sigma']}")
    if wass > tol["wasserstein_sigma"]:
        reasons.append(
            f"marginal Wasserstein {wass:.3f} sigma > {tol['wasserstein_sigma']}"
        )
    if "oob_frac" in report and report["oob_frac"] > tol["oob_frac"]:
        reasons.append(
            f"out-of-bounds mass {report['oob_frac']:.4f} > {tol['oob_frac']}"
        )

    if reasons:
        report["verdict"] = "fail"
        report["reasons"] = reasons
        return report

    p = report["mmd"]["pvalue"]
    if p is not None and p < tol["mmd_pvalue"]:
        report["verdict"] = "warn"
        report["reasons"] = [
            f"passes marginal tolerances but MMD detects a joint difference "
            f"(p={p:.3g}); inspect correlations"
        ]
        return report

    report["verdict"] = "pass"
    report["reasons"] = []
    return report
