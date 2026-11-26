#!/usr/bin/env python3
r"""Reproduce thesis Fig. 8.19: emulator -> likelihood -> PolyChord, then overlay.

LCDM under Planck 2018 (CamSpec), the six cosmological parameters
``logA, ns, H0, ombh2, omch2, tau``. We take the trained cosmulator emulator,
turn its density into the nuisance-free likelihood, re-run nested sampling with
PolyChord in those six parameters alone, and corner-overlay the result (orange)
against the full nested-sampling posterior (blue).

The parameter count is NOT special: ``d = len(ens.parameters)``, so the identical
code does 2, 6 or 8 parameters -- it is whatever the emulator was trained on. The
sibling ``mwe.py`` uses a 2-D synthetic banana only so it runs with no data.

Run on a machine with pypolychord + the emulator + the reference chain (e.g. CSD3):
    mpirun -np <ntasks> python mwe_cosmulator_ns.py
"""
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # cosmulator flows are float64

from cosmulator.ensemble import EnsembleEmulator

HOME = os.path.expanduser("~")
RDS = "/home/dlo26/rds/rds-dirac-dp192-63QXlf5HuFo/dlo26"

# ---- what identifies this (model, dataset) case -----------------------------
EMU_DIR = f"{HOME}/nfl_demo/emulators/case1_lcdm_planck_CamSpec"   # trained emulator
CHAIN_ORIG = (f"{RDS}/grid/ns/lcdm/planck_2018_CamSpec/"
              "planck_2018_CamSpec_polychord_raw/planck_2018_CamSpec")  # full NS (blue)
LOG_Z = -5323.098561536487        # parent run's log-evidence (unimpeded manifest)
OUT_DIR = os.environ.get("OUT_DIR", f"{HOME}/nfl_demo/chains_case1")
FILE_ROOT = "nfl_lcdm_planck_CamSpec"
TITLE = r"$\Lambda$CDM\,$\cdot$\,Planck 2018 (CamSpec)"


# =============================================================================
# STEP 1 -- the emulator (6 cosmological parameters)
# =============================================================================
ens = EnsembleEmulator.load(EMU_DIR)
params = list(ens.parameters)                      # [logA, ns, H0, ombh2, omch2, tau]
bounds = np.asarray(ens.bounds, dtype=float)
lo, hi = bounds[:, 0], bounds[:, 1]
d = len(params)
print(f"[emulator] loaded K={ens.K}, d={d} params {params}")

# No data to hand? swap STEP 1 for the 2-D runnable toy from mwe.py:
#     from mwe import make_banana
#     chain, bounds = make_banana(n=20000); params = ["w0", "wa"]; LOG_Z = 0.0
#     ens = EnsembleEmulator.train(chain, parameters=params, K=4, bounds=bounds,
#                                  spline=True, flow_layers=6, nn_width=32,
#                                  nn_depth=2, epochs=400, patience=60)


# =============================================================================
# STEP 2 -- density -> likelihood (nuisance-free conversion, JIT-compiled)
# =============================================================================
# log L(theta) = log q(theta) + log Z + log V   (Bevins et al. 2022). This is
# exactly ens.nuisance_free_logL, but we JIT-compile the mixture log-density ONCE
# so PolyChord's millions of per-point calls are compiled forward passes. Calling
# the un-jitted method per point makes XLA compile eagerly and can exhaust memory
# on a shared node -- compile once instead.
import jax.numpy as jnp
from jax.scipy.special import logsumexp

_members = ens.members
_logK = float(jnp.log(ens.K))
_log_V = float(np.sum(np.log(hi - lo)))      # flat box: log prior-volume


@jax.jit
def _logq(theta1):                            # (1, d) -> scalar mixture log-density
    lps = jnp.stack([m.log_prob(theta1) for m in _members], axis=0)
    return logsumexp(lps, axis=0)[0] - _logK


_logq(jnp.asarray((lo + hi) / 2.0)[None, :])  # trigger compilation once, up front


def loglikelihood(theta):
    logL = float(_logq(jnp.asarray(theta, dtype=float)[None, :])) + LOG_Z + _log_V
    return logL, []          # PolyChord needs a (logL, derived-list) TUPLE


def prior(hypercube):
    return (lo + np.asarray(hypercube) * (hi - lo)).tolist()   # unit cube -> box, as a LIST


# =============================================================================
# STEP 3 -- re-run nested sampling with PolyChord
# =============================================================================
import pypolychord
from pypolychord.settings import PolyChordSettings

os.makedirs(OUT_DIR, exist_ok=True)
settings = PolyChordSettings(d, 0)        # nDims=d, nDerived=0
settings.base_dir = OUT_DIR
settings.file_root = FILE_ROOT
settings.nlive = int(os.environ.get("NLIVE", "1000"))            # thesis: 1000
settings.num_repeats = 2 * d
settings.nprior = 10 * settings.nlive
settings.precision_criterion = float(os.environ.get("PRECISION", "1e-10"))  # thesis: 1e-10
settings.read_resume = False

print(f"[polychord] sampling d={d} ...")
out = pypolychord.run_polychord(loglikelihood, d, 0, settings, prior)
print(f"[result] recovered log Z = {out.logZ:.3f} +/- {out.logZerr:.3f} (parent {LOG_Z:.3f})")


# =============================================================================
# STEP 4 -- the Fig. 8.19 overlay: full NS (blue) vs emulated NS (orange)
# =============================================================================
if os.environ.get("SKIP_PLOT") == "1":     # set for a quick sampler-only smoke test
    raise SystemExit(0)
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from anesthetic import read_chains, make_2d_axes

LABELS = {"logA": r"$\ln(10^{10}A_\mathrm{s})$", "ns": r"$n_\mathrm{s}$", "H0": r"$H_0$",
          "ombh2": r"$\Omega_\mathrm{b}h^2$", "omch2": r"$\Omega_\mathrm{c}h^2$",
          "tau": r"$\tau$"}
C_ORIG, C_NEW, AFILL = "C0", "C1", 0.30

orig = read_chains(CHAIN_ORIG)                                   # full nested sampling
new = read_chains(os.path.join(OUT_DIR, FILE_ROOT))             # emulated NS re-run
cols = list(new.drop_labels().columns)
new = new.rename(columns={cols[i]: params[i] for i in range(d)}, level=0)

fig, axes = make_2d_axes(params, labels=LABELS, upper=True, figsize=(5.8, 5.8))
kde = dict(kinds=dict(lower="kde_2d", diagonal="kde_1d"), alpha=AFILL)   # lower triangle
sc = dict(kinds=dict(upper="scatter_2d"), alpha=0.25, ms=1.0, rasterized=True)  # upper
orig.plot_2d(axes, color=C_ORIG, **kde)
new.plot_2d(axes, color=C_NEW, **kde)
orig.plot_2d(axes, color=C_ORIG, **sc)
new.plot_2d(axes, color=C_NEW, **sc)

handles = [Patch(facecolor=C_ORIG, alpha=AFILL, label="Full nested sampling (unimpeded)"),
           Patch(facecolor=C_NEW, alpha=AFILL, label="Emulated nuisance-free likelihood")]
fig.legend(handles=handles, title=TITLE, loc="upper right", frameon=True)
fig.savefig(os.path.join(OUT_DIR, "corner_case1.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "corner_case1.png"), dpi=220, bbox_inches="tight")
print(f"[plot] wrote {OUT_DIR}/corner_case1.pdf and .png")
