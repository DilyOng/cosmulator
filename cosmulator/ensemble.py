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

_ARCH_KEYS = ("flow_layers", "nn_width", "nn_depth", "spline", "nn_activation")


def _enable_x64():
    import jax

    jax.config.update("jax_enable_x64", True)


def _member_valid(gen, bounds, expand=100.0, oob_frac_max=0.5):
    """Return whether a member's draws are numerically valid (TRUTH-INDEPENDENT).

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
    """Build a zero-content flow of the right architecture to deserialise into."""
    import jax
    import jax.numpy as jnp
    from flowjax.bijections import RationalQuadraticSpline
    from flowjax.distributions import Normal
    from flowjax.flows import masked_autoregressive_flow

    transformer = (
        RationalQuadraticSpline(knots=8, interval=4.0) if arch.get("spline") else None
    )
    # The activation is a static (non-array) field, so it is NOT stored in the
    # serialised leaves -- it must be rebuilt here from the manifest or the
    # deserialised weights silently load into the wrong network. Mirror the
    # string->callable map used at train time (maf.py); default relu for
    # pre-fix ensembles whose arch predates nn_activation being recorded.
    act = arch.get("nn_activation", "relu")
    if isinstance(act, str):
        act = {
            "relu": jax.nn.relu,
            "tanh": jax.nn.tanh,
            "silu": jax.nn.silu,
            "gelu": jax.nn.gelu,
        }[act]
    return masked_autoregressive_flow(
        key,
        base_dist=Normal(jnp.zeros(d)),
        transformer=transformer,
        flow_layers=arch["flow_layers"],
        nn_width=arch["nn_width"],
        nn_depth=arch["nn_depth"],
        nn_activation=act,
    )


def _member_base_u(member, theta):
    """Push physical ``theta`` to a member's standard-normal base space.

    Each member is an invertible flow with a standard-normal base, so
    ``u = T(theta) = bijection.inverse(standardise(theta))`` is distributed as
    :math:`\\mathcal N(0, I_d)` when ``theta`` is drawn from that member. This is
    the forward map used by the analytic base-radius goodness-of-fit test.
    """
    import jax
    import jax.numpy as jnp

    try:
        from flowjax.wrappers import unwrap
    except Exception:  # pragma: no cover - older flowjax without wrappers

        def unwrap(x):
            return x

    z = member._to_z(jnp.asarray(np.asarray(theta, dtype=np.float64)))
    bij = unwrap(member.flow.bijection)
    u = jax.vmap(bij.inverse)(z)
    base = member.flow.base_dist
    try:  # standardise by the base loc/scale if it is not already standard
        u = (u - base.loc) / base.scale
    except Exception:  # pragma: no cover - base already standard normal
        pass
    return np.asarray(u, dtype=np.float64)


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
            self.W = jnp.asarray(std)  # diagonal stored as a vector
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
        """Number of flows in the ensemble."""
        return len(self.members)

    # ------------------------------------------------------------------ build
    @classmethod
    def train(
        cls,
        samples,
        parameters=None,
        K=8,
        seeds=None,
        bounds=None,
        max_retries=None,
        report=None,
        **train_kwargs,
    ):
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
        pool = base_seeds + list(
            range(max(base_seeds) + 1, max(base_seeds) + 1 + max_retries)
        )
        arch = {k: train_kwargs.get(k) for k in _ARCH_KEYS if k in train_kwargs}
        # keep the activation JSON-serialisable: store its name, not the callable
        if callable(arch.get("nn_activation")):
            arch["nn_activation"] = getattr(arch["nn_activation"], "__name__", "relu")
        bnds = None if bounds is None else np.asarray(bounds, dtype=np.float64)

        members, invalid = [], 0
        for s in pool:
            if len(members) >= K:
                break
            out = train_maf_emulator(
                samples,
                parameters=parameters,
                seed=s,
                certify=False,
                bounds=bounds,
                **train_kwargs,
            )
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
    def _sample_raw(self, n, seed=0):
        """Draw ``n`` raw mixture samples (may leak past the prior box)."""
        _enable_x64()
        import jax

        key = jax.random.key(seed)
        counts = np.random.default_rng(seed).multinomial(
            n, np.full(self.K, 1.0 / self.K)
        )
        out = []
        for m, c in zip(self.members, counts):
            if c == 0:
                continue
            key, sk = jax.random.split(key)
            out.append(np.asarray(m.sample(sk, int(c))))
        return np.concatenate(out, axis=0)

    def sample(self, n, seed=0, clip=None):
        """Draw ``n`` samples from the mixture (pick a member uniformly, then draw).

        The flow is a smooth, unbounded density, so it leaks a little mass past
        hard prior walls. When the prior box ``bounds`` is known, draws are
        rejection-truncated to it by default -- i.e. sampled from the flow
        truncated to the prior support and renormalised -- so the deployed
        emulator respects the prior. Pass ``clip=False`` for raw (unbounded) draws.
        """
        clip = (self.bounds is not None) if clip is None else clip
        if not clip or self.bounds is None:
            return self._sample_raw(n, seed)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        out, got, s = [], 0, seed
        while got < n:
            g = self._sample_raw(int(max(n - got, 1) * 1.3) + 32, s)
            s += 1
            g = g[~((g < lo) | (g > hi)).any(axis=1)]
            if len(g):
                out.append(g)
                got += len(g)
        return np.concatenate(out, axis=0)[:n]

    def log_prob(self, theta):
        """Exact mixture log-density ``logsumexp_k(log q_k) - log K`` (physical)."""
        _enable_x64()
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp

        theta = jnp.asarray(np.asarray(theta, dtype=np.float64))
        lps = jnp.stack([m.log_prob(theta) for m in self.members], axis=0)
        return np.asarray(logsumexp(lps, axis=0) - np.log(self.K))

    # -------------------------------------------------------------- diagnostics
    def certify(self, samples, weights=None, n=None, seed=0, **certify_kwargs):
        """Certify the DEPLOYED (prior-truncated) mixture against the reference chain.

        The raw out-of-box fraction of the unbounded flow is reported as
        ``raw_oob_frac`` -- a model diagnostic -- but it does not gate the verdict,
        because the deployed sampler truncates to the prior box. Width, KL and
        Wasserstein are therefore judged on the in-box product the user receives.
        """
        from cosmulator.validation import certify as _certify

        theta = np.asarray(samples[self.parameters].to_numpy(), dtype=np.float64)
        if weights is None:
            weights = np.asarray(samples.get_weights(), dtype=np.float64)
        weights = weights / weights.sum()
        n = n or 50000 * self.K
        bounds = certify_kwargs.pop("bounds", self.bounds)
        gen = self._sample_raw(n, seed=seed)
        raw_oob = None
        if bounds is not None:
            b = np.asarray(bounds, dtype=np.float64)
            inbox = ~((gen < b[:, 0]) | (gen > b[:, 1])).any(axis=1)
            raw_oob = float(1.0 - inbox.mean())
            gen = gen[inbox]  # certify the truncated product actually deployed
        res = _certify(
            theta,
            gen,
            weights=weights,
            parameters=self.parameters,
            bounds=bounds,
            **certify_kwargs,
        )
        res["raw_oob_frac"] = raw_oob

        # Density-level KS-on-logP: needs the flow density, so it is computed here
        # rather than in validation.certify (which sees only samples). Reported
        # alongside MMD, and used as a soft effect-size gate (never a p-value):
        # a pass is downgraded to warn if D exceeds the ks_logp tolerance.
        from cosmulator.diagnostics import ks_logp as _ks_logp
        from cosmulator.validation import DEFAULT_TOLERANCES

        res["ks_logp"] = _ks_logp(self.log_prob(theta), self.log_prob(gen),
                                  weights=weights)
        ks_tol = dict(DEFAULT_TOLERANCES)
        ks_tol.update(certify_kwargs.get("tolerances") or {})
        if res["ks_logp"]["ks"] > ks_tol["ks_logp"] and res.get("verdict") in ("pass", "warn"):
            if res["verdict"] == "pass":
                res["verdict"] = "warn"
                res["reasons"] = res.get("reasons", [])
            res["reasons"].append(
                f"passes marginal tolerances but KS-on-logP D="
                f"{res['ks_logp']['ks']:.3f} > {ks_tol['ks_logp']} "
                f"(density-level mismatch; dlogp={res['ks_logp']['dlogp']:+.2f} nats)"
            )
        return res

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

    def nuisance_free_logL(self, theta, log_evidence, bounds=None):
        r"""Nuisance-free (nuisance-marginalised) log-likelihood at ``theta``.

        Converts the emulated nuisance-marginalised posterior into the
        nuisance-free likelihood of Bevins et al. (2022),
        ``L(theta) = P(theta) * Z / pi(theta)``, where ``P`` is the marginal
        posterior density (this emulator's own :meth:`log_prob`), ``Z`` the
        evidence of the parent run, and ``pi`` the marginal prior. Because the
        emulator is trained only on the sampled cosmological parameters, which
        carry a uniform box prior, the marginal prior is the constant
        ``pi = 1/V`` with ``log V = sum_i log(hi_i - lo_i)``, so

            log L(theta) = log q(theta) + log Z + log V.

        The untruncated (physical) :meth:`log_prob` is used, which is the
        correctly normalised density for a likelihood. This assumes a uniform,
        separable marginal prior on the sampled parameters, and is not valid for
        a non-uniform marginal prior (for example a derived parameter).

        Parameters
        ----------
        theta : array_like, shape (n, d)
            Points at which to evaluate the likelihood, in the sampled-parameter
            order of :attr:`parameters`.
        log_evidence : float
            The natural-log evidence ``log Z`` of the parent nested-sampling run
            (for example from the ``unimpeded`` manifest).
        bounds : array_like, shape (d, 2), optional
            Prior box, defaulting to the stored :attr:`bounds`.

        Returns
        -------
        ndarray, shape (n,)
            The nuisance-free ``log L(theta)`` with its absolute normalisation
            (the evidence and prior-volume factors included), so it can be
            combined across experiments and compared between models.
        """
        bounds = self.bounds if bounds is None else np.asarray(bounds, dtype=np.float64)
        if bounds is None:
            raise ValueError("nuisance_free_logL needs the prior box `bounds` (d, 2).")
        lo, hi = bounds[:, 0], bounds[:, 1]
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise ValueError(
                "nuisance_free_logL needs finite two-sided bounds "
                "(uniform marginal prior)."
            )
        log_V = float(np.sum(np.log(hi - lo)))
        return self.log_prob(theta) + float(log_evidence) + log_V

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
        from cosmulator.diagnostics import equal_weight_resample, knn_kl_divergence

        theta = np.asarray(samples[self.parameters].to_numpy(), dtype=np.float64)
        if weights is None:
            weights = np.asarray(samples.get_weights(), dtype=np.float64)
        true = equal_weight_resample(theta, weights, size=n_true, seed=seed)
        emu = self.sample(n_emu, seed=seed + 1)
        return {
            "kl_true_emu": knn_kl_divergence(true, emu, k=k),
            "kl_emu_true": knn_kl_divergence(emu, true, k=k),
        }

    def ks_logp(self, samples, weights=None, n_emu=20000, seed=0):
        r"""KS test on the emulator log-density -- the recommended certification.

        Evaluates this ensemble's exact mixture :meth:`log_prob` at the true
        samples and at the emulator's own draws, then runs the weighted
        two-sample Kolmogorov--Smirnov test of
        :func:`cosmulator.diagnostics.ks_logp` on those two one-dimensional
        :math:`\log q` distributions. Being one-dimensional and built on the
        analytic flow density, it certifies the emulator reliably even on weakly
        constrained posteriors with few samples, where :meth:`knn_kl` (a 6-D
        k-NN density estimate) degrades for reasons unrelated to the emulator.

        Returns the dict from :func:`cosmulator.diagnostics.ks_logp`: ``ks``
        (0 = perfect), ``pvalue``, ``n_eff`` and the signed ``dlogp``.
        """
        from cosmulator.diagnostics import ks_logp as _ks_logp

        theta = np.asarray(samples[self.parameters].to_numpy(), dtype=np.float64)
        if weights is None:
            weights = np.asarray(samples.get_weights(), dtype=np.float64)
        z_target = self.log_prob(theta)
        emu = self.sample(n_emu, seed=seed + 1)
        z_emu = self.log_prob(emu)
        return _ks_logp(z_target, z_emu, weights=weights)

    def base_radius_gof(self, samples, weights=None):
        r"""Analytic one-sample goodness-of-fit via the base-space radius, per member.

        Each member :math:`q_k` is an invertible flow with a standard-normal
        base, so if :math:`\theta \sim q_k` then :math:`r^2 = \lVert
        T_k(\theta)\rVert^2 \sim \chi^2_d` *exactly*. Pushing the posterior chain
        through each member's forward map and testing the squared radius against
        the analytic :math:`\chi^2_d` CDF
        (:func:`cosmulator.diagnostics.ks_chi2_base`) is a genuine one-sample KS
        test: the reference is closed-form, so no Monte-Carlo reference draws are
        involved (unlike :meth:`ks_logp`, whose density ordering is estimated from
        emulator draws). The signed weighted mean :math:`\langle r^2\rangle`
        relative to :math:`d` reads directly as over-concentration
        (:math:`\langle r^2\rangle > d` = the member is too narrow).

        The mixture itself has no single forward map, so the test is per member;
        the spread of the member statistics is itself informative (a wide spread
        means the seeds land in very different optima). The null floor is
        distribution-free, hence identical across members and computed once.

        Returns
        -------
        dict
            ``ks_per_member`` and ``mean_r2_per_member`` (length ``K`` arrays),
            ``ks_median``/``ks_max``, ``mean_r2_median``, the shared ``floor``,
            ``ks_over_floor_median``, ``dof`` and ``n_members``.
        """
        from cosmulator.diagnostics import ks_chi2_base, ks_one_sample_floor

        theta = np.asarray(samples[self.parameters].to_numpy(), dtype=np.float64)
        if weights is None:
            weights = np.asarray(samples.get_weights(), dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        d = len(self.parameters)
        floor = ks_one_sample_floor(weights)  # distribution-free: compute once
        ks, mean_r2 = [], []
        for m in self.members:
            u = _member_base_u(m, theta)
            r2 = np.sum(u * u, axis=1)
            res = ks_chi2_base(r2, d, weights=weights, floor=floor)
            ks.append(res["ks"])
            mean_r2.append(res["mean_r2"])
        ks = np.asarray(ks)
        mean_r2 = np.asarray(mean_r2)
        return {
            "ks_per_member": ks,
            "mean_r2_per_member": mean_r2,
            "ks_median": float(np.median(ks)),
            "ks_max": float(ks.max()),
            "mean_r2_median": float(np.median(mean_r2)),
            "floor": float(floor),
            "ks_over_floor_median": (
                float(np.median(ks) / floor) if floor > 0 else float("inf")
            ),
            "dof": d,
            "n_members": len(self.members),
        }

    # ------------------------------------------------------------- persistence
    def save(self, directory):
        """Serialise the ensemble to ``directory`` (one .eqx per member + metadata)."""
        import equinox as eqx

        os.makedirs(directory, exist_ok=True)
        meta = {
            "parameters": self.parameters,
            "arch": self.arch,
            "K": self.K,
            "bounds": None if self.bounds is None else self.bounds.tolist(),
        }
        maps = {}
        for i, m in enumerate(self.members):
            eqx.tree_serialise_leaves(
                os.path.join(directory, f"member_{i}.eqx"), m.flow
            )
            maps[f"mean_{i}"] = np.asarray(m.mean)
            maps[f"W_{i}"] = np.asarray(m.W)
            maps[f"diag_{i}"] = np.array(m._diag)
        np.savez(os.path.join(directory, "maps.npz"), **maps)
        with open(os.path.join(directory, "ensemble.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory):
        """Load an ensemble saved by :meth:`save`."""
        _enable_x64()  # members are trained/saved in float64
        import equinox as eqx
        import jax

        with open(os.path.join(directory, "ensemble.json")) as f:
            meta = json.load(f)
        maps = np.load(os.path.join(directory, "maps.npz"))
        params, arch, K = meta["parameters"], meta["arch"], meta["K"]
        # Ensembles saved before nn_activation was persisted have it recorded in
        # a sidecar nn_activation.json (written by the migration) instead of the
        # arch, so the original ensemble.json is never rewritten. Honour it here.
        if "nn_activation" not in arch:
            sidecar = os.path.join(directory, "nn_activation.json")
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    arch = {**arch, "nn_activation": json.load(f)["nn_activation"]}
        d = len(params)
        members = []
        for i in range(K):
            tmpl = _build_flow_template(jax.random.key(0), d, arch)
            flow = eqx.tree_deserialise_leaves(
                os.path.join(directory, f"member_{i}.eqx"), tmpl
            )
            mean, W = maps[f"mean_{i}"], maps[f"W_{i}"]
            if bool(maps[f"diag_{i}"]):
                members.append(_Member(flow, mean, W))
            else:
                members.append(_Member(flow, mean, None, whiten_L=W))
        bounds = meta.get("bounds")
        return cls(
            members, params, arch, bounds=None if bounds is None else np.asarray(bounds)
        )
