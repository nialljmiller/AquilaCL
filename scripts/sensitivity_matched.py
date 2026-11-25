"""Sensitivity-setting probe within one behaviour at a time (full 200-sample windows, birds with both settings,
grouped by bird), on stored h and on handcrafted features. If the setting stays readable inside a behaviour,
something besides behaviour carries it.
Usage: python scripts/sensitivity_matched.py --run <run> --windows <windows> --out <new json>"""
import argparse, json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aquilacl.evaluate import _feature_matrix, load_run

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--windows", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"{a.out} exists")
meta, H = load_run(a.run, a.windows)
F = _feature_matrix(meta)
rng = np.random.default_rng(0)
k = meta["sensitivity_k"].to_numpy()
full = meta["acc_samples"].to_numpy() == 200
lab = lambda c: meta[c].to_numpy() == 1 if c in meta else np.zeros(len(meta), bool)
subsets = {"all_full": np.ones(len(meta), bool), "flapping": lab("label_flapping"),
           "circling": lab("label_circling"), "gliding": lab("label_gliding"),
           "climbing_soar": lab("label_climbing_soar"),
           "vedba_0.05_0.15": (meta["vedba"].to_numpy() > 0.05) & (meta["vedba"].to_numpy() < 0.15)}
out = {}
for name, sel in subsets.items():
    rows = []
    for b, idx in meta[full & sel].groupby("bird").indices.items():
        ii = np.flatnonzero(full & sel)[idx]
        if min((k[ii] == 0).sum(), (k[ii] == 1).sum()) < 50:
            continue
        for c in (0, 1):
            r = ii[k[ii] == c]
            rows.append(rng.choice(r, min(1000, len(r)), replace=False))
    if not rows:
        out[name] = {"birds": 0}
        continue
    r = np.concatenate(rows)
    y, g = k[r], meta["bird"].to_numpy()[r]
    nb = len(np.unique(g))
    res = {"birds": int(nb), "n": int(len(r))}
    if nb >= 2:
        for xn, X in (("h", H[r].astype(np.float32)), ("features", F[r])):
            pred = np.empty(len(r), dtype=y.dtype)
            for tr, te in GroupKFold(min(5, nb)).split(X, y, g):
                pred[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr]).predict(X[te])
            res[xn] = float(balanced_accuracy_score(y, pred))
    out[name] = res
    print(name, res, flush=True)
a.out.write_text(json.dumps(out, indent=1))
