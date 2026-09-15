"""Train a normalising-flow emulator of a posterior, the margarine way.

This follows the workflow of Bevins et al. (the margarine papers, arXiv:2205.12841
and 2207.11457) and Harry Bevins' guidance for the JAX version of margarine:

* a **RealNVP** flow (margarine's recommended estimator in the JAX release; MAFs
  are unreliable there),
* trained on the **sampled cosmological parameters only** (not nuisance
  parameters), selected with :func:`cosmulator.parameters.cosmological_parameters`,
* on the **weighted** nested-sampling samples directly -- margarine's loss is
  :math:`-\\sum_j w_j \\log q(\\theta_j)`, with the weights taken straight from the
  chain. The posterior is **not** resampled to equal weight first; doing so
  discards ~90% of the samples and the fine weight structure, and is a deviation
  from the published method,
* with **Adam** (margarine builds ``optax.adam`` internally; only the learning
  rate is exposed).

Training requires the optional ``[train]`` extra (JAX, margarine, flax, optax);
this module is therefore not imported by the package root, so ``import
cosmulator`` stays cheap and JAX-free.
"""

import numpy as np

from cosmulator.parameters import cosmological_parameters


def _weights_of(samples, n):
    """Return normalised sample weights, defaulting to uniform.

    Parameters
    ----------
    samples : object
        A samples object; if it exposes ``get_weights`` (as an anesthetic
        ``Samples`` does) those weights are used, otherwise uniform weights.
    n : int
        Number of samples, used to build uniform weights as a fallback.

    Returns
    -------
    numpy.ndarray
        Weights of length ``n`` summing to one.
    """
    getter = getattr(samples, "get_weights", None)
    if getter is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(getter(), dtype=np.float64)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("sample weights must be finite and sum to a positive value")
    return w / total


def train_emulator(
    samples,
    parameters=None,
    *,
    hidden_size=128,
    num_layers=2,
    num_coupling_layers=4,
    learning_rate=1e-3,
    epochs=2000,
    patience=100,
    batch_size=1024,
    mass_fraction=0.9999,
    seed=0,
):
    """Train a RealNVP emulator on the weighted cosmological posterior of a chain.

    Parameters
    ----------
    samples : anesthetic.Samples
        The chain. Its ``get_weights`` supplies the posterior weights that enter
        the loss; its columns supply the parameter set.
    parameters : sequence of str, optional
        The parameters to train on. Defaults to the sampled cosmological
        parameters detected by
        :func:`cosmulator.parameters.cosmological_parameters`, which is the
        margarine-aligned choice (cosmological parameters only, no nuisance).
    hidden_size, num_layers, num_coupling_layers : int
        RealNVP architecture: width of each coupling network, number of layers in
        each, and number of coupling layers. Defaults are margarine's.
    learning_rate : float
        Adam step size. Default ``1e-3`` matches margarine's / Bevins' default.
    epochs : int
        Maximum training epochs; margarine stops early on the validation loss.
    patience : int
        Early-stopping patience, in epochs, on the validation loss.
    batch_size : int
        Minibatch size, capped at the number of samples.
    mass_fraction : float
        Keep the highest-weight samples that together carry this fraction of the
        posterior mass, dropping the negligible-weight tail. Nested-sampling output
        includes extreme low-weight prior-phase points; kept in, they set
        margarine's gaussianisation range and drive the loss to NaN (the JAX
        RealNVP, unlike the TF MAF, does not mask tiny weights). Default keeps
        99.99% of the mass. Pass ``1.0`` to keep every sample.
    seed : int
        Seed for the JAX PRNG (data split and initialisation).

    Returns
    -------
    tuple
        ``(flow, parameters)`` -- the trained ``margarine`` ``RealNVP`` and the
        list of parameter names it was trained on, in order.

    Notes
    -----
    Imports margarine and JAX lazily, so the failure mode when the ``[train]``
    extra is absent is an informative ``ImportError`` at call time rather than at
    package import.
    """
    import jax
    from margarine.estimators.realnvp import RealNVP

    if parameters is None:
        parameters = cosmological_parameters(samples)
    parameters = list(parameters)
    dim = len(parameters)

    theta = np.asarray(samples[parameters].to_numpy(), dtype=np.float64)
    weights = _weights_of(samples, len(theta))

    # Weighted training -- no equal-weight resampling -- but on the mass-carrying
    # samples only: drop the negligible-weight tail (see mass_fraction) so extreme
    # prior-phase outliers do not break the gaussianisation. margarine renormalises
    # the kept weights internally and minimises -sum_j w_j log q(theta_j).
    if mass_fraction < 1.0:
        order = np.argsort(weights)[::-1]
        cutoff = int(np.searchsorted(np.cumsum(weights[order]), mass_fraction)) + 1
        keep = np.sort(order[:cutoff])
        theta, weights = theta[keep], weights[keep]
    theta_j = jax.numpy.asarray(theta)
    weights_j = jax.numpy.asarray(weights)

    flow = RealNVP(
        theta_j,
        weights=weights_j,
        in_size=dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_coupling_layers=num_coupling_layers,
    )
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    flow.train(
        subkey,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=patience,
        batch_size=min(batch_size, len(theta)),
    )
    return flow, parameters
