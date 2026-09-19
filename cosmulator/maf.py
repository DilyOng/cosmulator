r"""Train a Masked Autoregressive Flow emulator of a posterior, in pure JAX.

This is the emulator that works. It reproduces cosmological nested-sampling
posteriors to ~1% marginal width on GPU, and captures non-Gaussian degeneracies
(e.g. the curved :math:`w_0`--:math:`w_a` banana) that affine coupling flows
(RealNVP) cannot. It is built directly on `flowjax <https://github.com/danielward27/flowjax>`_
-- a **MAF**, trained by **weighted maximum likelihood**, on the sampled
cosmological parameters, with honest per-parameter standardisation. It deliberately
does **not** depend on margarine: margarine's RealNVP pipeline over-disperses (its
prior-bound "gaussianisation" compresses the data relative to the base, inflating
the marginal widths to ~5--11%), which is what this module was written to escape.

The recipe mirrors the older TensorFlow-MAF emulator that produced faithful fits,
reimplemented in JAX so it runs on GPU with a single change of substrate.

Training requires the optional ``[train]`` extra (JAX, flowjax, equinox, optax);
this module is not imported by the package root, so ``import cosmulator`` stays
cheap and JAX-free.
"""

import numpy as np

from cosmulator.parameters import cosmological_parameters


def _weighted_standardiser(theta, weights):
    """Return the weighted per-parameter mean and standard deviation.

    Parameters
    ----------
    theta : numpy.ndarray, shape (n, d)
        Samples.
    weights : numpy.ndarray, shape (n,)
        Normalised weights.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean, std)``, each shape ``(d,)``. Standardising as
        ``(theta - mean) / std`` maps the posterior to zero mean and unit
        marginal variance -- the honest replacement for margarine's over-wide
        min/max gaussianisation, which compresses the data and over-disperses.
    """
    mean = weights @ theta
    std = np.sqrt(weights @ (theta - mean) ** 2)
    return mean, std


def train_maf_emulator(
    samples,
    parameters=None,
    *,
    flow_layers=8,
    nn_width=50,
    nn_depth=2,
    learning_rate=1e-3,
    batch_size=1024,
    epochs=1500,
    patience=150,
    mass_fraction=0.9999,
    spline=False,
    seed=0,
):
    """Train a MAF (or spline-MAF) emulator on a weighted cosmological posterior.

    Parameters
    ----------
    samples : anesthetic.Samples
        The chain. Its ``get_weights`` supplies the posterior weights and its
        columns the parameters.
    parameters : sequence of str, optional
        Parameters to train on; defaults to the sampled cosmological parameters
        (:func:`cosmulator.parameters.cosmological_parameters`).
    flow_layers : int
        Number of autoregressive layers in the flow.
    nn_width, nn_depth : int
        Width and depth of the conditioner network in each layer.
    learning_rate : float
        Adam step size (with global-norm gradient clipping at 1.0).
    batch_size : int
        Minibatch size, capped at the number of training samples.
    epochs : int
        Maximum epochs; training stops early on the held-out weighted NLL.
    patience : int
        Early-stopping patience in epochs.
    mass_fraction : float
        Keep the highest-weight samples carrying this fraction of the posterior
        mass, dropping the negligible-weight tail. Pass ``1.0`` to keep all.
    spline : bool
        Use rational-quadratic-spline transformers (a Neural Spline Flow) instead
        of affine ones. Marginally sharper, at more parameters.
    seed : int
        Seed for the JAX PRNG.

    Returns
    -------
    dict
        ``flow`` (the trained flowjax distribution, in standardised space),
        ``parameters`` (list, in order), ``mean`` and ``std`` (the standardiser,
        each an array of length ``d``), and ``best_val_nll``. Draw physical
        samples with ``flow.sample(key, (n,)) * std + mean``; evaluate density at
        physical ``x`` with ``flow.log_prob((x - mean) / std)``.

    Notes
    -----
    Imports flowjax and JAX lazily, so the failure mode without the ``[train]``
    extra is an informative ``ImportError`` at call time. flowjax requires
    new-style typed PRNG keys (``jax.random.key``).
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from flowjax.bijections import RationalQuadraticSpline
    from flowjax.distributions import Normal
    from flowjax.flows import masked_autoregressive_flow

    if parameters is None:
        parameters = cosmological_parameters(samples)
    parameters = list(parameters)
    d = len(parameters)

    theta = np.asarray(samples[parameters].to_numpy(), dtype=np.float64)
    w = np.asarray(samples.get_weights(), dtype=np.float64)
    w = w / w.sum()
    if mass_fraction < 1.0:
        order = np.argsort(w)[::-1]
        cutoff = int(np.searchsorted(np.cumsum(w[order]), mass_fraction)) + 1
        keep = np.sort(order[:cutoff])
        theta, w = theta[keep], w[keep] / w[keep].sum()

    mean, std = _weighted_standardiser(theta, w)
    z = jnp.asarray((theta - mean) / std)
    wj = jnp.asarray(w)

    key = jax.random.key(seed)
    key, fkey = jax.random.split(key)
    transformer = RationalQuadraticSpline(knots=8, interval=4.0) if spline else None
    flow = masked_autoregressive_flow(
        fkey, base_dist=Normal(jnp.zeros(d)), transformer=transformer,
        flow_layers=flow_layers, nn_width=nn_width, nn_depth=nn_depth,
    )

    key, sk = jax.random.split(key)
    perm = jax.random.permutation(sk, len(z))
    n_val = max(1, int(0.1 * len(z)))
    z_val, w_val = z[perm[:n_val]], wj[perm[:n_val]]
    z_tr, w_tr = z[perm[n_val:]], wj[perm[n_val:]]

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    opt_state = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    def weighted_nll(fl, x, wt):
        return -jnp.sum(wt * fl.log_prob(x)) / jnp.sum(wt)

    @eqx.filter_jit
    def step(fl, ost, x, wt):
        loss, grads = eqx.filter_value_and_grad(weighted_nll)(fl, x, wt)
        updates, ost = opt.update(grads, ost, eqx.filter(fl, eqx.is_inexact_array))
        return eqx.apply_updates(fl, updates), ost, loss

    bs = min(batch_size, len(z_tr))
    best_val, best_flow, bad = np.inf, flow, 0
    for _ in range(epochs):
        key, sk = jax.random.split(key)
        order = jax.random.permutation(sk, len(z_tr))
        for i in range(0, len(z_tr), bs):
            b = order[i:i + bs]
            flow, opt_state, _ = step(flow, opt_state, z_tr[b], w_tr[b])
        val = float(weighted_nll(flow, z_val, w_val))
        if np.isfinite(val) and val < best_val:
            best_val, best_flow, bad = val, flow, 0
        else:
            bad += 1
            if bad >= patience:
                break

    return {
        "flow": best_flow,
        "parameters": parameters,
        "mean": np.asarray(mean),
        "std": np.asarray(std),
        "best_val_nll": best_val,
    }
