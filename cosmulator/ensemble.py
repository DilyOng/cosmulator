r"""Deployment ensemble of MAF emulators -- the production deliverable.

A single flow trained on a weighted posterior lands in a different shallow
optimum every seed (GPU training is non-deterministic and the weighted-NLL loss
is near-degenerate), so its certified per-parameter width swings by ~2-3
percentage points run to run. Averaging ``K`` independently seeded flows as an
equal-weight mixture cancels those idiosyncratic marginal errors: on
``planck_2018_plik`` a ``K=8`` mixture certifies *below* the best single seed
(walcdm 0.83% vs best 1.33%, nlcdm 0.61% vs best 1.20%), passing < 1% on every
model tried, and it comes with margarine-style error bars for free (the spread
across members).

The ensemble is a genuine density model, not a bag of samples: with
:math:`q_k` the ``k``-th member's density in physical parameter space (its own
standardiser Jacobian folded in),

.. math::
    q_\mathrm{ens}(\theta) = \frac1K \sum_k q_k(\theta), \qquad
    \log q_\mathrm{ens}(\theta) = \operatorname{logsumexp}_k \log q_k(\theta)
    - \log K,

so :meth:`EnsembleEmulator.log_prob` is exact and enables the marginal
information gain :meth:`EnsembleEmulator.D_KL` and downstream importance
sampling. Averaging member log-probs instead would be wrong by Jensen.

Training needs the ``[train]`` extra (JAX, flowjax, equinox); this module keeps
those imports lazy so ``import cosmulator`` stays cheap.
"""
import json
import os

import numpy as np

from cosmulator.parameters import cosmological_parameters

_ARCH_KEYS = ("flow_layers", "nn_width", "nn_depth", "spline")


def _enable_x64():
    import jax
    jax.config.update("jax_enable_x64", True)


def _member_valid(gen, bounds, expand=100.0, oob_frac_max=0.5):
    """True if a member's draws are numerically valid -- a TRUTH-INDEPENDENT gate.

    A run is rejected only for failing to instantiate a usable model, never for
    disagreeing with the target posterior (which would leak the answer into the
    certification). The criteria use only a-priori information -- finiteness and
    the known prior box:

    * any non-finite draw -> invalid;
    * any draw beyond the prior box expanded by ``expand`` box-widths (a gross
      blow-up, e.g. 1e10 on a parameter bounded at 0.1) -> invalid;
    * more than ``oob_frac_max`` of draws outside the prior box -> invalid.

    The thresholds are deliberately loose: a healthy flow (trained in physical
    space without hard walls) may leak a little mass just past a wall, which is
    fine; only a numerically failed sampler trips this gate. ``bounds`` is the
    prior box ``(d, 2)`` with ``+/-inf`` for unbounded sides.
    """
    gen = np.asarray(gen, dtype=np.float64)
    if not np.isfinite(gen).all():
        return False
    bounds = np.asarray(bounds, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    fin = np.isfinite(lo) & np.isfinite(hi)
    if fin.any():
        j = np.where(fin)[0]
        width = hi[j] - lo[j]
        g = gen[:, j]
        if np.any(g < lo[j] - expand * width) or np.any(g > hi[j] + expand * width):
            return False
        oob = (g < lo[j]) | (g > hi[j])
        if oob.any(axis=1).mean() > oob_frac_max:
            return False
    return True


def _build_flow_template(key, d, arch):
    """A zero-content flow of the right architecture to deserialise leaves into."""
    import jax.numpy as jnp
    from flowjax.bijections import RationalQuadraticSpline
    from flowjax.distributions import Normal
    from flowjax.flows import masked_autoregressive_flow

    transformer = (RationalQuadraticSpline(knots=8, interval=4.0)
                   if arch.get("spline") else None)
    return masked_autoregressive_flow(
        key, base_dist=Normal(jnp.zeros(d)), transformer=transformer,
        flow_layers=arch["flow_layers"], nn_width=arch["nn_width"],
        nn_depth=arch["nn_depth"])


class _Member:
    """One trained flow plus the affine map from physical theta to flow space.

    ``z = (theta - mean) @ Winv.T`` where ``Winv`` is ``diag(1/std)`` for the
    diagonal standardiser or ``L^{-1}`` for a Cholesky whitener. The log-Jacobian
    ``log|det dz/dtheta| = -log|det W|`` turns the flow's density in ``z`` into the
    density in physical ``theta``.
    """

    def __init__(self, flow, mean, std, whiten_L=None):
        import jax.numpy as jnp
        self.flow = flow
        self.mean = jnp.asarray(np.asarray(mean, dtype=np.float64))
        if whiten_L is not None:
            W = np.asarray(whiten_L, dtype=np.float64)
            self.W = jnp.asarray(W)
            self.Winv = jnp.asarray(np.linalg.inv(W))
            self._log_det_W = float(np.sum(np.log(np.abs(np.diag(W)))))
            self._diag = False
        else:
            std = np.asarray(std, dtype=np.float64)
            self.W = jnp.asarray(std)                 # diagonal stored as a vector
            self.Winv = jnp.asarray(1.0 / std)
            self._log_det_W = float(np.sum(np.log(np.abs(std))))
            self._diag = True

    def _to_z(self, theta):
        if self._diag:
            return (theta - self.mean) * self.Winv
        return (theta - self.mean) @ self.Winv.T

    def _from_z(self, z):
        if self._diag:
            return z * self.W + self.mean
        return z @ self.W.T + self.mean

    def log_prob(self, theta):
        # log q_k(theta) = log p_z(z) - log|det W|   (change of variables)
        return self.flow.log_prob(self._to_z(theta)) - self._log_det_W

    def sample(self, key, n):
        return self._from_z(self.flow.sample(key, (n,)))


class EnsembleEmulator:
    """Equal-weight mixture of ``K`` MAF flows over the same parameters."""

    def __init__(self, members, parameters, arch, bounds=None):
        self.members = list(members)
        self.parameters = list(parameters)
        self.arch = dict(arch)
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        self.invalid_runs = 0

    @property
    def K(self):
        return len(self.members)

    # ------------------------------------------------------------------ build
    @classmethod
    def train(cls, samples, parameters=None, K=8, seeds=None, bounds=None,
              max_retries=None, report=None, **train_kwargs):
        """Train ``K`` validity-screened members from a fixed recipe.

        Each member is trained with ``certify=False`` and a distinct seed. When
        ``bounds`` are given, every trained member is checked by the
        truth-independent :func:`_member_valid` gate (finiteness + prior box); an
        invalid run (a numerically failed MAF sampler) is discarded and replaced by
        a fresh seed, so the ensemble is ``K`` valid members BY CONSTRUCTION rather
        than by post-hoc pruning against the target. The number of invalid runs is
        recorded as ``self.invalid_runs`` (report it -- it is part of the honest
        methodology). ``train_kwargs`` pass through to
        :func:`cosmulator.maf.train_maf_emulator`.
        """
        from cosmulator.maf import train_maf_emulator

        if parameters is None:
            parameters = cosmological_parameters(samples)
        parameters = list(parameters)
        base_seeds = list(range(K)) if seeds is None else list(seeds)
        if max_retries is None:
            max_retries = 2 * K
        # a seed pool: the K requested seeds, then extras to replace invalid runs
        pool = base_seeds + list(range(max(base_seeds) + 1,
                                       max(base_seeds) + 1 + max_retries))
        arch = {k: train_kwargs.get(k) for k in _ARCH_KEYS if k in train_kwargs}
        bnds = None if bounds is None else np.asarray(bounds, dtype=np.float64)

        members, invalid = [], 0
        for s in pool:
            if len(members) >= K:
                break
            out = train_maf_emulator(samples, parameters=parameters, seed=s,
                                     certify=False, bounds=bounds, **train_kwargs)
            m = _Member(out["flow"], out["mean"], out["std"], out.get("whiten_L"))
            ok = True
            if bnds is not None:
                import jax
                # 50k probes the ~4.6-sigma tails where affine-inverse blow-ups
                # live; 20k was too small to catch the stochastic failures.
                g = np.asarray(m.sample(jax.random.key(90000 + s), 50000))
                ok = _member_valid(g, bnds)
            if ok:
                members.append(m)
            else:
                invalid += 1
            if report is not None:
                report(s, out["best_val_nll"], ok)
        ens = cls(members, parameters, arch, bounds=bounds)
        ens.invalid_runs = invalid
        return ens

    # ----------------------------------------------------------------- density
    def sample(self, n, seed=0):
        """Draw ``n`` samples from the mixture (pick a member uniformly, then draw)."""
        _enable_x64()
        import jax

        key = jax.random.key(seed)
        counts = np.random.default_rng(seed).multinomial(
            n, np.full(self.K, 1.0 / self.K))
        out = []
        for m, c in zip(self.members, counts):
            if c == 0:
                continue
            key, sk = jax.random.split(key)
            out.append(np.asarray(m.sample(sk, int(c))))
        return np.concatenate(out, axis=0)

    def log_prob(self, theta):
        """Exact mixture log-density ``logsumexp_k(log q_k) - log K`` in physical space."""
        _enable_x64()
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp

        theta = jnp.asarray(np.asarray(theta, dtype=np.float64))
        lps = jnp.stack([m.log_prob(theta) for m in self.members], axis=0)
        return np.asarray(logsumexp(lps, axis=0) - np.log(self.K))

    # -------------------------------------------------------------- diagnostics
    def certify(self, samples, weights=None, n=None, seed=0, **certify_kwargs):
        """Certify pooled mixture samples against the (weighted) reference chain."""
        from cosmulator.validation import certify as _certify

        theta = np.asarray(samples[self.parameters].to_numpy(), dtype=np.float64)
        if weights is None:
            weights = np.asarray(samples.get_weights(), dtype=np.float64)
        weights = weights / weights.sum()
        n = n or 50000 * self.K
        gen = self.sample(n, seed=seed)
        bounds = certify_kwargs.pop("bounds", self.bounds)
        return _certify(theta, gen, weights=weights, parameters=self.parameters,
                        bounds=bounds, **certify_kwargs)

    def D_KL(self, bounds=None, n=200000, seed=0):
        r"""Marginal information gain ``D_KL(q || pi)`` for a uniform prior box.

        With ``pi`` uniform on the prior box of (log-)volume ``log V``,
        ``D_KL = E_{theta~q}[log q(theta) - log pi] = E_q[log q] + log V``,
        estimated by Monte Carlo over the ensemble's own draws using the exact
        mixture ``log_prob`` (unbiased given exact sampling). This is the *marginal*
        cosmological information gain -- what anesthetic's nested-sampling ``D_KL``
        (the joint gain over all sampled dimensions, nuisances included) cannot give.
        """
        bounds = self.bounds if bounds is None else np.asarray(bounds, dtype=np.float64)
        if bounds is None:
            raise ValueError("D_KL needs the prior box `bounds` (d, 2).")
        lo, hi = bounds[:, 0], bounds[:, 1]
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise ValueError("uniform-prior D_KL needs finite two-sided bounds.")
        log_V = float(np.sum(np.log(hi - lo)))
        theta = self.sample(n, seed=seed)
        log_q = self.log_prob(theta)
        return float(np.mean(log_q) + log_V)

    def drop_invalid(self, bounds=None, n=20000, seed=0):
        """Post-hoc safety net: drop numerically invalid members (truth-independent).

        Applies the same a-priori :func:`_member_valid` gate as training (finiteness
        + prior box), so it never references the target posterior. Prefer building a
        valid ensemble at train time (:meth:`train` retries invalid seeds); this is
        for cleaning ensembles saved before that existed. Returns ``(kept, dropped)``
        and mutates in place.
        """
        _enable_x64()
        import jax

        bounds = self.bounds if bounds is None else np.asarray(bounds, dtype=np.float64)
        if bounds is None:
            raise ValueError("drop_invalid needs the prior box `bounds` (d, 2).")
        key = jax.random.key(seed)
        keep = []
        for m in self.members:
            key, sk = jax.random.split(key)
            g = np.asarray(m.sample(sk, n))
            if _member_valid(g, bounds):
                keep.append(m)
        dropped = len(self.members) - len(keep)
        self.members = keep
        return len(keep), dropped

    def knn_kl(self, samples, weights=None, k=5, n_true=20000, n_emu=20000, seed=0):
        """k-NN KL divergence between the true samples and the emulator's samples.

        The headline emulator-accuracy number: ``D_KL(true || emulated)`` (and the
        reverse) estimated directly in cosmological-parameter space from equal-weight
        draws of each, with no nuisances and no prior term (see
        :func:`cosmulator.diagnostics.knn_kl_divergence`). Near zero means the
        emulator matches the true marginal posterior.
        """
        from cosmulator.diagnostics import (equal_weight_resample,
                                            knn_kl_divergence)

        theta = np.asarray(samples[self.parameters].to_numpy(), dtype=np.float64)
        if weights is None:
            weights = np.asarray(samples.get_weights(), dtype=np.float64)
        true = equal_weight_resample(theta, weights, size=n_true, seed=seed)
        emu = self.sample(n_emu, seed=seed + 1)
        return {"kl_true_emu": knn_kl_divergence(true, emu, k=k),
                "kl_emu_true": knn_kl_divergence(emu, true, k=k)}

    # ------------------------------------------------------------- persistence
    def save(self, directory):
        """Serialise the ensemble to ``directory`` (one .eqx per member + metadata)."""
        import equinox as eqx

        os.makedirs(directory, exist_ok=True)
        meta = {"parameters": self.parameters, "arch": self.arch, "K": self.K,
                "bounds": None if self.bounds is None else self.bounds.tolist()}
        maps = {}
        for i, m in enumerate(self.members):
            eqx.tree_serialise_leaves(os.path.join(directory, f"member_{i}.eqx"),
                                      m.flow)
            maps[f"mean_{i}"] = np.asarray(m.mean)
            maps[f"W_{i}"] = np.asarray(m.W)
            maps[f"diag_{i}"] = np.array(m._diag)
        np.savez(os.path.join(directory, "maps.npz"), **maps)
        with open(os.path.join(directory, "ensemble.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory):
        """Load an ensemble saved by :meth:`save`."""
        _enable_x64()                       # members are trained/saved in float64
        import equinox as eqx
        import jax

        with open(os.path.join(directory, "ensemble.json")) as f:
            meta = json.load(f)
        maps = np.load(os.path.join(directory, "maps.npz"))
        params, arch, K = meta["parameters"], meta["arch"], meta["K"]
        d = len(params)
        members = []
        for i in range(K):
            tmpl = _build_flow_template(jax.random.key(0), d, arch)
            flow = eqx.tree_deserialise_leaves(
                os.path.join(directory, f"member_{i}.eqx"), tmpl)
            mean, W = maps[f"mean_{i}"], maps[f"W_{i}"]
            if bool(maps[f"diag_{i}"]):
                members.append(_Member(flow, mean, W))
            else:
                members.append(_Member(flow, mean, None, whiten_L=W))
        bounds = meta.get("bounds")
        return cls(members, params, arch,
                   bounds=None if bounds is None else np.asarray(bounds))
