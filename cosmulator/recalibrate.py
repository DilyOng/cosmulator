"""Post-hoc affine recalibration of emulator samples to the target's moments.

A trained flow can reproduce the *shape* of a posterior -- its degeneracies and
non-Gaussian curvature -- while being systematically too broad (over-dispersed) or
slightly shifted. Retraining to remove a few percent of width is expensive and not
always effective; a cheap, exact alternative is to correct the emulator's output
*after* sampling with a single affine map that matches its first two moments to the
target's.

For emulator samples with mean :math:`\\mu_q` and covariance :math:`\\Sigma_q` and a
target with mean :math:`\\mu_P` and covariance :math:`\\Sigma_P`,

.. math::
    x' = \\mu_P + L_P L_q^{-1} (x - \\mu_q),

where :math:`\\Sigma = L L^\\top` are Cholesky factors. The corrected samples have
mean :math:`\\mu_P` and covariance :math:`\\Sigma_P` *exactly*, so the marginal width
and bias errors become zero by construction, at no inference cost. The map is linear,
so it preserves the emulator's copula (the shape of the degeneracies); it corrects
only the global location, scale and linear correlations. Where over-dispersion is
uniform -- the common case for these flows -- that is the whole error.

This module operates on plain arrays and needs neither JAX nor margarine.
"""

import numpy as np

from cosmulator.diagnostics import _normalised_weights


def _cholesky_psd(cov, floor_rel=1e-8):
    """Cholesky factor of a covariance, with a tiny jitter for conditioning.

    Parameters
    ----------
    cov : numpy.ndarray, shape (d, d)
        A symmetric positive (semi-)definite covariance.
    floor_rel : float
        Jitter added to the diagonal, relative to the mean variance, so a
        numerically semi-definite covariance (a near-degenerate posterior such as
        w0-wa) still factorises.

    Returns
    -------
    numpy.ndarray, shape (d, d)
        The lower-triangular Cholesky factor.
    """
    d = cov.shape[0]
    jitter = float(np.trace(cov) / d) * floor_rel
    return np.linalg.cholesky(cov + np.eye(d) * jitter)


def recalibrate(emulated_samples, target_samples, target_weights=None):
    r"""Affine-correct emulator samples to match the target's mean and covariance.

    Parameters
    ----------
    emulated_samples : array_like, shape (m, d)
        Samples drawn from the emulator.
    target_samples : array_like, shape (n, d)
        Samples from the target distribution, used only for their (weighted) mean
        and covariance. Need not match ``m``.
    target_weights : array_like, optional
        Weights for the target samples. Nested-sampling output is weighted, and
        ignoring that would recalibrate to the wrong covariance.

    Returns
    -------
    numpy.ndarray, shape (m, d)
        The emulated samples after the affine correction. Their sample mean and
        covariance equal the target's (weighted) mean and covariance up to Monte
        Carlo error.

    Raises
    ------
    ValueError
        If the two sample sets differ in dimensionality, or either covariance is
        not positive definite even after jitter.

    Notes
    -----
    The correction is ``x' = mu_P + (x - mu_q) @ (L_P L_q^{-1}).T`` for row-vector
    batches, with ``Sigma_P = L_P L_P^T`` (target) and ``Sigma_q = L_q L_q^T``
    (emulator). Being linear it preserves the emulator's copula and only fixes the
    location, scale and linear correlations -- it cannot repair a genuinely wrong
    *shape*, so the corrected corner plot should still be inspected.
    """
    emulated = np.atleast_2d(np.asarray(emulated_samples, dtype=float))
    target = np.atleast_2d(np.asarray(target_samples, dtype=float))
    if emulated.shape[1] != target.shape[1]:
        raise ValueError(
            f"dimension mismatch: emulated has {emulated.shape[1]} parameters, "
            f"target has {target.shape[1]}"
        )

    w = _normalised_weights(target_weights, target.shape[0])
    mu_p = w @ target
    dt = target - mu_p
    cov_p = (w[:, None] * dt).T @ dt

    mu_q = emulated.mean(axis=0)
    cov_q = np.cov(emulated, rowvar=False)
    cov_q = np.atleast_2d(cov_q)

    l_p = _cholesky_psd(cov_p)
    l_q = _cholesky_psd(cov_q)
    transform = l_p @ np.linalg.inv(l_q)
    return mu_p + (emulated - mu_q) @ transform.T
