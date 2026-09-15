"""Measures of how faithfully an emulator reproduces its target.

A trained flow always produces samples. Whether those samples are the posterior
is a separate question, and one that is easy not to ask: an emulator that is
subtly over-dispersed looks entirely healthy until its output is used for
inference. These diagnostics exist so that every emulator carries evidence of
its own fidelity, and so that a bad one can be identified automatically rather
than by eye.

The functions here operate on plain arrays of samples. They neither train nor
load models, and so require no JAX; given samples from an emulator and samples
from its target, any of them can be computed on a laptop.
"""

import numpy as np


def _normalised_weights(weights, size):
    """Return finite, normalised weights, defaulting to uniform.

    Parameters
    ----------
    weights : array_like or None
        Per-sample weights, or None for uniform weighting.
    size : int
        Number of samples the weights must cover.

    Returns
    -------
    numpy.ndarray
        Weights summing to one.

    Raises
    ------
    ValueError
        If the weights do not match the samples, or sum to zero.
    """
    if weights is None:
        return np.full(size, 1.0 / size)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (size,):
        raise ValueError(f"weights have shape {weights.shape}, expected ({size},)")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must be finite and sum to a positive value")
    return weights / total


def self_consistency(log_prob_at_target, log_prob_at_emulated, weights=None):
    r"""Compare the emulator's density at its own samples and at the target's.

    For a flow :math:`q` approximating a posterior :math:`P`, the quantity

    .. math::
        \Delta\log p = E_q[\log q] - E_P[\log q]

    vanishes when the two distributions agree. It is cheap, needs only the
    flow's own density, and is signed: a negative value indicates the emulator
    spreads probability more widely than the target, which is the failure mode
    that flatters an emulator by making it look conservative.

    Parameters
    ----------
    log_prob_at_target : array_like
        The emulator's log density evaluated at samples from the target.
    log_prob_at_emulated : array_like
        The emulator's log density evaluated at its own samples.
    weights : array_like, optional
        Weights for the target samples. Nested sampling output is weighted, and
        ignoring that biases the comparison.

    Returns
    -------
    dict
        ``dlogp`` and its absolute value, plus the two expectations that formed
        it, so that a surprising result can be diagnosed without recomputing.

    Notes
    -----
    Non-finite densities are dropped before averaging. A flow evaluated far
    into the tails can legitimately return ``-inf``, and discarding those
    points is preferable to propagating a NaN through the whole metric.
    """
    at_target = np.asarray(log_prob_at_target, dtype=float)
    at_emulated = np.asarray(log_prob_at_emulated, dtype=float)

    w = _normalised_weights(weights, at_target.size)
    finite = np.isfinite(at_target)
    if not finite.any():
        raise ValueError("no finite log densities at the target samples")
    e_target = float(np.sum(w[finite] * at_target[finite]) / np.sum(w[finite]))

    finite_emulated = np.isfinite(at_emulated)
    if not finite_emulated.any():
        raise ValueError("no finite log densities at the emulated samples")
    e_emulated = float(np.mean(at_emulated[finite_emulated]))

    dlogp = e_emulated - e_target
    return {
        "dlogp": dlogp,
        "abs_dlogp": abs(dlogp),
        "E_q_logq": e_emulated,
        "E_P_logq": e_target,
    }


def forward_kl(log_prob_at_target, target_samples, weights=None):
    r"""Forward Kullback--Leibler divergence from the target to the emulator.

    For a target posterior :math:`P` and an emulator :math:`q`,

    .. math::
        D_{\mathrm{KL}}(P\,\|\,q) = -H(P) - E_P[\log q]
                                  = H(P, q) - H(P),

    reported in nats. Unlike :func:`self_consistency`, this is a genuine
    divergence: it is non-negative, zero only when the emulator equals the
    target, and it *cannot* be made small by over-dispersion. That makes it the
    metric to minimise when tuning an emulator; the signed self-consistency
    :math:`\Delta\log p` can be driven towards zero by an over-broad flow, and
    so rewards the very failure it is meant to catch.

    The cross-entropy :math:`H(P, q) = -E_P[\log q]` is computed exactly from
    the flow's own density at the target samples. The entropy :math:`H(P)` is
    not available for a nuisance-marginalised posterior, so it is estimated by
    the Gaussian entropy of the target's covariance,
    :math:`\tfrac12\ln[(2\pi e)^d \det\Sigma]`. Because a Gaussian has the
    largest entropy of any distribution with a given covariance, this
    *overestimates* :math:`H(P)`, so **when the emulator has finite density at
    every target sample** the returned value is a **lower bound** on the true
    forward KL --- tight when the target is close to Gaussian, as marginal
    cosmological posteriors usually are. Where the emulator's density underflows
    at some target samples those points are floored (see Notes) rather than
    dropped, and the fraction of target weight affected is returned as
    ``missed_mass_frac`` so the failure is visible without distorting the score.

    Parameters
    ----------
    log_prob_at_target : array_like, shape (n,)
        The emulator's log density evaluated at the ``n`` target samples.
    target_samples : array_like, shape (n, d)
        The target samples themselves, used for the covariance in :math:`H(P)`.
    weights : array_like, optional
        Weights for the target samples. Nested sampling output is weighted, and
        ignoring that biases both the cross-entropy and the covariance.

    Returns
    -------
    dict
        ``forward_kl`` (the lower bound, in nats), the ``cross_entropy`` and
        ``entropy_gaussian`` terms that formed it, and ``missed_mass_frac`` --
        the fraction of target weight at which the emulator's density
        underflowed -- so a value can be diagnosed without recomputing.

    Raises
    ------
    ValueError
        If the densities and samples disagree in length, the target covariance
        is not positive definite, or no target sample has finite density.

    Notes
    -----
    Non-finite densities are **floored, not dropped, and not given an unphysical
    penalty**. A normalising flow with a Gaussian base has infinite support and
    never assigns true-zero density, so a ``-inf`` at a target sample is
    numerical underflow in the far tail, not a genuinely missed mode. Dropping
    such points, as :func:`self_consistency` does for its symmetric quantity,
    would flatter a mode-collapsed flow; but assigning a huge fixed penalty
    (e.g. ``-1e6``) is equally wrong -- a fraction of a percent of tail weight
    then contributes hundreds of nats and swamps the real signal, leaving the
    score flat and useless for optimisation. Instead each underflowed density is
    floored at the least-likely *finite* log density in the batch. That is
    bounded, so a benign tail underflow (tiny weight) is negligible, while a
    genuinely missed mode (high weight underflowing) is still penalised in
    proportion to its weight -- and ``missed_mass_frac`` reports the fraction
    regardless, so mode collapse is never hidden.
    """
    at_target = np.asarray(log_prob_at_target, dtype=float)
    samples = np.atleast_2d(np.asarray(target_samples, dtype=float))
    n, d = samples.shape
    if at_target.shape != (n,):
        raise ValueError(
            f"log densities have shape {at_target.shape}, expected ({n},) "
            f"to match the target samples"
        )

    w = _normalised_weights(weights, n)
    # Floor, rather than drop, a non-finite log density, so a genuinely missed mode
    # is still penalised in proportion to its weight, but a flow with infinite
    # support (whose -inf is mere numerical underflow in the far tail) is not
    # assigned an unphysical, signal-swamping penalty. The floor is the least-likely
    # *finite* density the emulator can evaluate here; a benign tail underflow then
    # carries negligible weight, while a high-weight miss carries a large one. The
    # fraction of target weight that underflowed is returned as ``missed_mass_frac``.
    finite = np.isfinite(at_target)
    if not finite.any():
        raise ValueError("no finite log densities at the target samples")
    floor = float(at_target[finite].min())
    floored = np.where(finite, at_target, floor)
    missed_mass = float(w[~finite].sum())
    cross_entropy = -float(np.sum(w * floored))

    mean = w @ samples
    centred = samples - mean
    cov = (centred * w[:, None]).T @ centred
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("target covariance is not positive definite")
    entropy_gaussian = 0.5 * (d * (1.0 + np.log(2.0 * np.pi)) + logdet)

    return {
        "forward_kl": float(cross_entropy - entropy_gaussian),
        "cross_entropy": float(cross_entropy),
        "entropy_gaussian": float(entropy_gaussian),
        "missed_mass_frac": missed_mass,
    }


def moment_error(target_samples, emulated_samples, weights=None):
    """Compare the first two moments of the target and the emulator.

    Reported per parameter and averaged. Where
    :func:`self_consistency` gives one number for the whole distribution, this
    says which parameters are wrong and in what way, which is what a user needs
    in order to act on a poor score.

    Parameters
    ----------
    target_samples : array_like, shape (n, d)
        Samples from the target distribution.
    emulated_samples : array_like, shape (m, d)
        Samples drawn from the emulator. Need not match ``n``.
    weights : array_like, optional
        Weights for the target samples.

    Returns
    -------
    dict
        Per-parameter ``bias_sigma`` and ``width_error_pct`` arrays, and their
        mean absolute values. Bias is expressed in units of the target's own
        standard deviation so that parameters with different scales can be
        compared and a single tolerance applied across them.

    Raises
    ------
    ValueError
        If the two sample sets have different dimensionality, or a target
        parameter has zero spread.
    """
    target = np.atleast_2d(np.asarray(target_samples, dtype=float))
    emulated = np.atleast_2d(np.asarray(emulated_samples, dtype=float))
    if target.shape[1] != emulated.shape[1]:
        raise ValueError(
            f"dimension mismatch: target has {target.shape[1]} parameters, "
            f"emulated has {emulated.shape[1]}"
        )

    w = _normalised_weights(weights, target.shape[0])
    target_mean = w @ target
    target_std = np.sqrt(w @ (target - target_mean) ** 2)
    # A constant parameter yields a standard deviation of order machine epsilon
    # rather than exactly zero, so an absolute test against zero passes and the
    # bias below divides by ~1e-16. Compare against the parameter's own scale.
    scale = np.maximum(np.abs(target_mean), 1.0)
    if not np.all(target_std > 1e-12 * scale):
        raise ValueError(
            "a target parameter has zero spread; cannot express bias in "
            "units of its standard deviation"
        )

    bias = (emulated.mean(axis=0) - target_mean) / target_std
    width = 100.0 * (emulated.std(axis=0) - target_std) / target_std
    return {
        "bias_sigma": bias.tolist(),
        "width_error_pct": width.tolist(),
        "mean_abs_bias_sigma": float(np.mean(np.abs(bias))),
        "mean_abs_width_err_pct": float(np.mean(np.abs(width))),
    }


def effective_sample_size(weights):
    r"""Return the effective number of independent samples in a weighted set.

    Nested sampling chains carry many points of negligible weight, so the
    nominal length of a chain badly overstates how much the flow can learn
    from it. This is the number that matters when deciding whether a chain is
    worth training on.

    Parameters
    ----------
    weights : array_like
        Per-sample weights. Need not be normalised.

    Returns
    -------
    float
        Kish's effective sample size, :math:`(\sum w)^2 / \sum w^2`.
    """
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / np.square(w).sum())
