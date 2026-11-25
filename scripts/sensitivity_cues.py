"""What still tells the two sensitivity settings apart after the common format (step and clip removed).

For full windows (200 samples) of birds that use both settings, per setting:
- still windows (VeDBA < 0.02): noise floor = SD of the first difference per axis, after the common format;
- every full window: mean |first difference| at continuous-mode burst junctions (sample index multiple of 24)
  against all other positions, i.e. whether joined 24-sample bursts leave a step at the joins;
- calendar month of the windows.
Usage: python scripts/sensitivity_cues.py --windows <windows dir> --format <format.json> --out <new json>
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--windows", type=Path, required=True)
ap.add_argument("--format", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--per-bird", type=int, default=3000)
a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"{a.out} exists")
fmt = json.loads(a.format.read_text())
step, clip = fmt["step_g"], fmt["clip_g"]
rng = np.random.default_rng(0)
rows = {0: [], 1: []}
for d in sorted(p for p in a.windows.iterdir() if p.is_dir()):
    m = pd.read_parquet(d / "meta.parquet")
    if len(m) == 0:
        continue
    f = pd.read_parquet(d / "features.parquet")
    full = (m["acc_samples"].to_numpy() == 200)
    k = m["sensitivity_k"].to_numpy()
    if len(np.unique(k[full])) < 2:
        continue
    acc = np.load(d / "acc.npy", mmap_mode="r")
    for c in (0, 1):
        idx = np.flatnonzero(full & (k == c))
        idx = np.sort(rng.choice(idx, min(a.per_bird, len(idx)), replace=False))
        x = np.clip(np.round(np.asarray(acc[idx], np.float32) / step) * step, -clip, clip)
        dx = np.abs(np.diff(x, axis=1))  # (n, 199, 3); diff j is between samples j and j+1
        junction = np.zeros(199, bool)
        junction[23::24] = True
        still = f["vedba"].to_numpy()[idx] < 0.02
        rows[c].append(pd.DataFrame({
            "bird": d.name, "still": still,
            "noise_sd": np.diff(x, axis=1).std(axis=1).mean(axis=1) / np.sqrt(2),
            "jump_junction": dx[:, junction].mean(axis=(1, 2)), "jump_other": dx[:, ~junction].mean(axis=(1, 2)),
            "distinct_levels": np.array([len(np.unique(xi[:, 2])) for xi in x]),
            "month": m["t_start"].iloc[idx].dt.month.to_numpy(),
        }))
out = {}
for c in (0, 1):
    df = pd.concat(rows[c])
    s = df[df.still]
    out[f"k{c}"] = {
        "windows": int(len(df)), "still_windows": int(len(s)), "birds": int(df.bird.nunique()),
        "still_noise_sd_g_median": float(s.noise_sd.median()), "still_noise_sd_g_q25_q75": s.noise_sd.quantile([.25, .75]).tolist(),
        "still_distinct_levels_z_median": float(s.distinct_levels.median()),
        "jump_junction_over_other_median": float((df.jump_junction / df.jump_other.clip(lower=1e-9)).median()),
        "still_jump_junction_over_other_median": float((s.jump_junction / s.jump_other.clip(lower=1e-9)).median()),
        "month_share": df.month.value_counts(normalize=True).sort_index().round(3).to_dict(),
    }
n0 = pd.concat(rows[0]).query("still").groupby("bird").noise_sd.median()
n1 = pd.concat(rows[1]).query("still").groupby("bird").noise_sd.median()
both = n0.index.intersection(n1.index)
out["paired_still_noise_ratio_k1_over_k0"] = {"birds": int(len(both)), "median": float((n1[both] / n0[both]).median()) if len(both) else None}
a.out.write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
