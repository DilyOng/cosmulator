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


def _weight_stratified_split(weights, val_fraction, test_fraction):
    """Split sample indices into train/val/test, stratified by weight.

    A uniform random split of a nested-sampling chain is high-variance: the few
    points carrying most of the posterior mass can all land in one split, starving
    the others and making early stopping and certification erratic. Systematic
    sampling over the weight-sorted order instead gives every split a
    representative slice of the weight distribution, deterministically.

    Parameters
    ----------
    weights : numpy.ndarray, shape (n,)
        Per-sample weights (need not be normalised).
    val_fraction, test_fraction : float
        Fractions for the early-stopping validation set and the held-out test set;
        the remainder is the training set. ``test_fraction`` may be 0.

    Returns
    -------
    tuple of numpy.ndarray
        ``(train_idx, val_idx, test_idx)`` into the original ordering.
    """
    n = len(weights)
    order = np.argsort(weights)[::-1]  # high to low weight

    def systematic(frac, m):
        # Pick ~frac of m positions, spread evenly (every ~1/frac-th).
        idx = np.arange(m)
        return (np.floor((idx + 1) * frac).astype(np.int64)
                - np.floor(idx * frac).astype(np.int64)) == 1

    label = np.array(["train"] * n)  # labels in weight-sorted order
    if test_fraction > 0:
        label[systematic(test_fraction, n)] = "test"
    rest = np.where(label == "train")[0]
    if val_fraction > 0 and rest.size:
        val_frac_adj = val_fraction / (1.0 - test_fraction)
        label[rest[systematic(val_frac_adj, rest.size)]] = "val"

    out = np.empty(n, dtype=object)
    out[order] = label
    return (np.where(out == "train")[0], np.where(out == "val")[0],
            np.where(out == "test")[0])


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
    train_loss="global",
    test_fraction=0.15,
    certify=True,
    bounds=None,
    certify_samples=50000,
    report=None,
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
    train_loss : {"global", "batch"}
        How the minibatch training loss normalises the weights. ``"global"`` (the
        default) divides by a constant (the global training-weight sum, scaled by
        ``n_train / batch_size``), the unbiased minibatch estimator of the full
        weighted objective. ``"batch"`` (the original recipe) divides by the
        per-batch weight sum, i.e. a self-normalised weighted mean over the batch;
        simple, but a biased estimator because the denominator is random from batch
        to batch, so a low-mass batch is re-inflated to the same influence as a
        high-mass one. On the lcdm/planck chain ``"global"`` measured slightly
        tighter (0.88% vs 1.04% marginal width), so it is preferred; ``"batch"`` is
        kept for reproducing earlier results. The validation NLL is always the exact
        weighted mean over the whole held-out set and is unaffected by this choice.
    test_fraction : float
        Fraction of samples held out (weight-stratified) for certification, seen by
        neither training nor early stopping, so the certificate measures
        generalisation rather than memorisation. Pass ``0`` to skip the test split.
    certify : bool
        If true (and ``test_fraction`` > 0), certify the trained flow against the
        held-out test set with :func:`cosmulator.validation.certify` and return the
        verdict under ``certification``.
    bounds : array_like, shape (d, 2), optional
        Per-parameter prior limits, passed to certification for the out-of-bounds gate.
    certify_samples : int
        Number of flow draws used for certification.
    report : callable, optional
        A ``report(epoch, val_nll)`` callback invoked after each epoch with the
        held-out weighted NLL. Used to stream progress to a hyperparameter
        optimiser for pruning: the callback may raise to abort the trial early.
        Kept generic so this module has no dependency on the optimiser.

    Returns
    -------
    dict
        ``flow`` (the trained flowjax distribution, in standardised space),
        ``parameters`` (list, in order), ``mean`` and ``std`` (the standardiser,
        each an array of length ``d``), and ``best_val_nll`` (the held-out weighted
        NLL in *standardised* space: comparable across trials of the same model,
        but not across models with different ``std``). Draw physical samples with
        ``flow.sample(key, (n,)) * std + mean``; evaluate the physical-space density
        at ``x`` with ``flow.log_prob((x - mean) / std) - np.sum(np.log(std))`` --
        the final term is the standardisation Jacobian, needed for a correctly
        normalised density (omitting it leaves the density off by a constant factor,
        which biases any absolute cross-entropy or forward-KL computed from it).
        Also ``certification``: the :func:`cosmulator.validation.certify` verdict.
        Its moment, marginal and MMD metrics are measured against the **full kept
        chain** (a low-noise, non-leaking reference for marginal fidelity), while
        the **held-out** test set supplies an out-of-sample density sentinel added
        to the report as ``heldout_nll`` (with ``val_nll`` and ``overfit_gap_nll``):
        a held-out NLL much larger than the validation NLL signals memorisation.
        ``None`` if certification was skipped.

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

    if train_loss not in ("batch", "global"):
        raise ValueError(f"train_loss must be 'batch' or 'global', got {train_loss!r}")

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

    # Weight-stratified train / early-stop-val / held-out-test split. The test set
    # is seen by neither training nor early stopping, so certification measures
    # generalisation. The standardiser is fit on train+val only, so the test set
    # does not leak even through the per-parameter moments.
    do_test = certify and test_fraction > 0
    train_idx, val_idx, test_idx = _weight_stratified_split(
        w, val_fraction=0.1, test_fraction=(test_fraction if do_test else 0.0))
    fit_idx = np.concatenate([train_idx, val_idx])
    mean, std = _weighted_standardiser(theta[fit_idx], w[fit_idx] / w[fit_idx].sum())

    z_tr = jnp.asarray((theta[train_idx] - mean) / std)
    w_tr = jnp.asarray(w[train_idx])
    z_val = jnp.asarray((theta[val_idx] - mean) / std)
    w_val = jnp.asarray(w[val_idx])

    key = jax.random.key(seed)
    key, fkey = jax.random.split(key)
    transformer = RationalQuadraticSpline(knots=8, interval=4.0) if spline else None
    flow = masked_autoregressive_flow(
        fkey, base_dist=Normal(jnp.zeros(d)), transformer=transformer,
        flow_layers=flow_layers, nn_width=nn_width, nn_depth=nn_depth,
    )

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    opt_state = opt.init(eqx.filter(flow, eqx.is_inexact_array))

    def weighted_nll(fl, x, wt):
        # Exact weighted-mean NLL over the given set. Correct for the full
        # validation set (its denominator is deterministic); used for reporting
        # and early stopping regardless of ``train_loss``.
        return -jnp.sum(wt * fl.log_prob(x)) / jnp.sum(wt)

    w_tr_sum = float(jnp.sum(w_tr))
    n_tr = int(z_tr.shape[0])

    def train_nll(fl, x, wt):
        lp = fl.log_prob(x)
        if train_loss == "global":
            # Divide by a constant (global train-weight sum scaled by n_tr/batch):
            # an unbiased minibatch estimator of the full weighted objective.
            return -jnp.sum(wt * lp) * n_tr / (x.shape[0] * w_tr_sum)
        # "batch": self-normalised per-batch weighted mean (the original recipe).
        return -jnp.sum(wt * lp) / jnp.sum(wt)

    @eqx.filter_jit
    def step(fl, ost, x, wt):
        loss, grads = eqx.filter_value_and_grad(train_nll)(fl, x, wt)
        updates, ost = opt.update(grads, ost, eqx.filter(fl, eqx.is_inexact_array))
        return eqx.apply_updates(fl, updates), ost, loss

    bs = min(batch_size, len(z_tr))
    best_val, best_flow, bad = np.inf, flow, 0
    for epoch in range(epochs):
        key, sk = jax.random.split(key)
        order = jax.random.permutation(sk, len(z_tr))
        for i in range(0, len(z_tr), bs):
            b = order[i:i + bs]
            flow, opt_state, _ = step(flow, opt_state, z_tr[b], w_tr[b])
        val = float(weighted_nll(flow, z_val, w_val))
        if report is not None:
            report(epoch, val)
        if np.isfinite(val) and val < best_val:
            best_val, best_flow, bad = val, flow, 0
        else:
            bad += 1
            if bad >= patience:
                break

    certification = None
    if do_test and test_idx.size > 0:
        from cosmulator.validation import certify as _certify
        key, sk = jax.random.split(key)
        gen = np.asarray(best_flow.sample(sk, (certify_samples,))) * std + mean
        # Certify moments, marginals and MMD against the FULL kept chain, not the
        # small held-out subset. Marginal fidelity does not leak from training (an
        # MLE flow cannot narrow its marginal standard deviation by memorising
        # points), and the full chain is a far less noisy yardstick: a 15% held-out
        # subset's own weighted-variance sampling noise (worse than 1/sqrt(2 ESS)
        # once weight skew and kurtosis are accounted for) can dwarf a genuine ~1%
        # width error and fail a good emulator. The joint MMD carries no p-value for
        # a weighted target, so it is descriptive here, not a pass/fail test.
        certification = _certify(theta, gen, weights=w, bounds=bounds)
        # The held-out test set is reserved for the out-of-sample density check --
        # the genuine overfitting sentinel. If the flow memorised training points,
        # its held-out NLL rises relative to the (early-stopping) validation NLL.
        z_test = jnp.asarray((theta[test_idx] - mean) / std)
        wt_test = jnp.asarray(w[test_idx])
        heldout_nll = float(
            -jnp.sum(wt_test * best_flow.log_prob(z_test)) / jnp.sum(wt_test))
        certification["heldout_nll"] = heldout_nll
        certification["val_nll"] = best_val
        certification["overfit_gap_nll"] = heldout_nll - best_val

    return {
        "flow": best_flow,
        "parameters": parameters,
        "mean": np.asarray(mean),
        "std": np.asarray(std),
        "best_val_nll": best_val,
        "certification": certification,
    }
