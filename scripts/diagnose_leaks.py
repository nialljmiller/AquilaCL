"""Diagnose the two nuisance leaks found in stage 4 (sensitivity setting, short vs continuous ACC window).

For a sample of windows per bird it computes a few interpretable device statistics and asks how well each
one alone (and all together) separates the nuisance on held-out birds. It also reports where the settings
occur (per bird, per date) and the raw count range per setting, to tell resolution from clipping from timing.

Usage: python scripts/diagnose_leaks.py --data <data dir> --run <run dir> [--per-bird 3000]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--per-bird", type=int, default=3000)
args = ap.parse_args()
rng = np.random.default_rng(0)
W = args.data / "windows_10s"
hold = set(json.loads((args.run / "holdout.json").read_text()))

rows = []
for d in sorted(p for p in W.iterdir() if p.is_dir()):
    m = pd.read_parquet(d / "meta.parquet")
    f = pd.read_parquet(d / "features.parquet", columns=["vedba"])
    acc = np.load(d / "acc.npy", mmap_mode="r")
    am = np.load(d / "acc_mask.npy", mmap_mode="r")
    i = np.sort(rng.choice(len(m), min(args.per_bird, len(m)), replace=False))
    a = acc[i].astype(np.float32)
    mk = am[i]
    n = mk.sum(1)
    absmax = np.where(mk[..., None], np.abs(a), 0).max(axis=(1, 2))
    # quantisation step: smallest non-zero spacing of distinct values below 0.5 g (float16 exact there)
    step = np.full(len(i), np.nan)
    for j in range(len(i)):
        v = a[j][mk[j]].ravel()
        v = np.unique(v[np.abs(v) < 0.5])
        if len(v) > 3:
            dv = np.diff(v)
            dv = dv[dv > 1e-6]
            if len(dv):
                step[j] = np.percentile(dv, 10)
    # sample-to-sample noise (median abs second difference), robust to motion
    d2 = np.abs(np.diff(a, 2, axis=1))
    m2 = mk[:, 2:] & mk[:, 1:-1] & mk[:, :-2]
    noise = np.array([np.median(d2[j][m2[j]]) if m2[j].any() else np.nan for j in range(len(i))])
    # first present slot (position of a short burst inside the 200-slot window)
    first = np.where(mk.any(1), mk.argmax(1), -1)
    rows.append(pd.DataFrame({
        "bird": d.name, "tag": m["tag"].to_numpy()[i], "t_start": m["t_start"].to_numpy()[i],
        "k": m["sensitivity_k"].to_numpy()[i], "acc_samples": n, "absmax": absmax, "step": step,
        "noise": noise, "first_slot": first, "vedba": f["vedba"].to_numpy()[i],
        "segment_windows": m["segment_windows"].to_numpy()[i],
    }))
df = pd.concat(rows, ignore_index=True)
df["holdout"] = df["bird"].isin(hold)
df["short"] = df["acc_samples"] < 100
out = {}

# where the sensitivity settings occur
pb = df.groupby("bird").agg(frac_k1=("k", "mean"), frac_short=("short", "mean"), holdout=("holdout", "first"))
out["birds_with_k1"] = int((pb["frac_k1"] > 0).sum())
out["birds_all_k1"] = int((pb["frac_k1"] == 1).sum())
out["birds_mixed_k"] = int(((pb["frac_k1"] > 0) & (pb["frac_k1"] < 1)).sum())
out["holdout_frac_k1"] = pb[pb["holdout"]]["frac_k1"].round(3).to_dict()
out["holdout_frac_short"] = pb[pb["holdout"]]["frac_short"].round(3).to_dict()
out["birds_with_short"] = int((pb["frac_short"] > 0).sum())
out["birds_mixed_short"] = int(((pb["frac_short"] > 0) & (pb["frac_short"] < 1)).sum())
df["month"] = pd.to_datetime(df["t_start"]).dt.to_period("M").astype(str)
out["k1_by_month"] = df.groupby("month")["k"].mean().round(3).to_dict()
out["short_by_month"] = df.groupby("month")["short"].mean().round(3).to_dict()

# statistic summaries per setting
for col in ["absmax", "step", "noise", "vedba", "acc_samples"]:
    out[f"{col}_by_k"] = df.groupby("k")[col].describe(percentiles=[0.01, 0.5, 0.99]).round(5).to_dict("index")
out["frac_absmax_ge_1.9_by_k"] = df.groupby("k")["absmax"].apply(lambda s: float((s >= 1.9).mean())).to_dict()
out["short_first_slot"] = df[df["short"]]["first_slot"].describe().round(1).to_dict()
out["short_segment_windows"] = df[df["short"]]["segment_windows"].describe().round(1).to_dict()
out["short_acc_samples_counts"] = df[df["short"]]["acc_samples"].value_counts().head(10).to_dict()

# how well single interpretable statistics separate the nuisance on held-out birds (train on others)
def probe(feats, target):
    tr, te = ~df["holdout"], df["holdout"]
    X = df[feats].fillna(-1).to_numpy()
    y = df[target].astype(int).to_numpy()
    if len(np.unique(y[te])) < 2:
        return None
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[tr], y[tr])
    return round(float(balanced_accuracy_score(y[te], m.predict(X[te]))), 3)

df["log_step"] = np.log2(df["step"])
df["log_noise"] = np.log10(df["noise"] + 1e-5)
for feats in (["log_step"], ["absmax"], ["log_noise"], ["log_step", "absmax", "log_noise"]):
    out[f"k_from_{'+'.join(feats)}"] = probe(feats, "k")
# same, restricted to moving windows (VeDBA > 0.1 g), where resolution matters less
mov = df["vedba"] > 0.1
sub = df
df = sub[mov]
out["moving_k_from_log_step"] = probe(["log_step"], "k")
out["moving_k_from_absmax"] = probe(["absmax"], "k")
df = sub
print(json.dumps(out, indent=1, default=str))
