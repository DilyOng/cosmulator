"""Backfill nn_activation for legacy ensembles WITHOUT touching ensemble.json.

For each ensemble.json whose arch lacks nn_activation, recover the activation
from the colocated summary.json (hpo_best_params / best_params) or the best
completed trial in trials.csv, and write it to a NEW sidecar file
``nn_activation.json`` in the same directory. The original ensemble.json and
the weight files are never modified. load() reads the sidecar when the arch
lacks the key. Idempotent; atomic writes; skips anything already covered.

Usage: python migrate_activation.py <root> [--apply]
"""
import csv, glob, json, os, sys, tempfile

ROOT = sys.argv[1]
APPLY = "--apply" in sys.argv
VALID = {"relu", "tanh", "silu", "gelu"}
SIDECAR = "nn_activation.json"


def from_summary(d):
    for key in ("hpo_best_params", "best_params"):
        act = (d.get(key) or {}).get("nn_activation")
        if act in VALID:
            return act
    return None


def from_trials(path, best_number=None):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("state") != "COMPLETE":
                continue
            act = r.get("params_nn_activation")
            try:
                val = float(r.get("value", "nan"))
            except ValueError:
                continue
            if act in VALID and val == val:
                rows.append((int(r["number"]), val, act))
    if not rows:
        return None
    if best_number is not None:
        for n, _, act in rows:
            if n == best_number:
                return act
    return min(rows, key=lambda t: t[1])[2]


def write_atomic(path, obj):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


fs = sorted(glob.glob(os.path.join(ROOT, "**", "ensemble.json"), recursive=True))
already = wrote = unrec = 0
by_src = {"summary": 0, "trials": 0}
rec_act, unrec_paths = {}, []
for f in fs:
    d = os.path.dirname(f)
    arch = json.load(open(f)).get("arch", {})
    if arch.get("nn_activation") in VALID or os.path.exists(os.path.join(d, SIDECAR)):
        already += 1
        continue
    act, src = None, None
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        act = from_summary(s)
        if act:
            src = "summary"
    if act is None:
        best_n = json.load(open(sp)).get("hpo_best_trial_number") if os.path.exists(sp) else None
        act = from_trials(os.path.join(d, "trials.csv"), best_n)
        if act:
            src = "trials"
    if act is None:
        unrec += 1
        unrec_paths.append(os.path.relpath(f, ROOT))
        continue
    by_src[src] += 1
    rec_act[act] = rec_act.get(act, 0) + 1
    if APPLY:
        write_atomic(
            os.path.join(d, SIDECAR),
            {
                "nn_activation": act,
                "_source": f"backfilled from {src} ({'hpo_best_params' if src=='summary' else 'best trial'})",
                "_note": "legacy activation backfill; ensemble.json is unchanged",
            },
        )
    wrote += 1

print(f"{'APPLIED' if APPLY else 'DRY-RUN'}  root={ROOT}")
print(f"  ensemble.json found      : {len(fs)}")
print(f"  already covered          : {already}")
print(f"  sidecars {'written' if APPLY else 'to write'}     : {wrote}  (source {by_src})")
print(f"  recovered activations    : {rec_act}")
print(f"  UNRECOVERABLE (left as-is): {unrec}")
