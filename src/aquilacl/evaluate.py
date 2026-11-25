"""Stage 4 evaluation of a trained run.

1. Hour-level probe suite (identical to the one applied to the first-draft embeddings): mean h per bird-hour.
2. Window-level probes on held-out birds only: linear models fitted on training-bird windows, scored on the
   birds that never entered pretraining. Behaviour targets should score high; device targets low.
3. Cluster stability: k-means on a bird-balanced sample of h for a range of k; adjusted Rand index between
   refits on bird-bootstrap samples (compare with the handcrafted baseline, ARI 0.35-0.44).
4. Invariance tests on held-out birds (needs the checkpoint): re-encode windows after changing only the
   recording format and count how often the k-means state stays the same.
   - burst: a continuous 10 s window versus a 24- or 40-sample span of it (what a short bout records);
   - sensitivity: a 1024 counts/g window re-gridded to 512 counts/g resolution, and a 512 counts/g window
     clipped to the narrower 1024 counts/g range.
   Reference: how often the state stays the same between consecutive windows of one segment.
   A probe can always read recording length from the mask; what matters is whether states change.

Usage: python -m aquilacl.evaluate --run <run dir> --windows <windows dir> --events <data>/events
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dataset, probes

WINDOW_BEHAVIOUR_REG = ["vedba", "speed_mean", "climb_mean", "turn_abs_mean", "flap_band_frac", "sun_elev_deg"]
WINDOW_LABELS = ["label_flapping", "label_perched", "label_circling", "label_gliding", "label_climbing_soar", "label_night"]
WINDOW_NUISANCE = ["sensitivity_k", "decoding", "short_acc"]


def load_run(run: Path, windows: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metas, H = [], []
    for d in sorted(p for p in windows.iterdir() if p.is_dir()):
        e = np.load(run / "embeddings" / f"{d.name}.npz")
        m = pd.concat([pd.read_parquet(d / "meta.parquet"), pd.read_parquet(d / "features.parquet")], axis=1)
        if len(m) != len(e["h"]):
            raise ValueError(f"{d.name}: {len(m)} windows but {len(e['h'])} embeddings")
        metas.append(m)
        H.append(e["h"].astype(np.float32))
    meta = pd.concat(metas, ignore_index=True)
    meta["short_acc"] = meta["acc_samples"] < 100
    return meta, np.concatenate(H)


def sample_per_bird(meta, rows, n, rng):
    out = []
    for b, idx in meta[rows].groupby("bird").indices.items():
        ii = np.flatnonzero(rows)[idx]
        out.append(rng.choice(ii, min(n, len(ii)), replace=False))
    return np.concatenate(out)


def holdout_window_probes(meta, H, hold, rng, n_train=2000, n_test=5000) -> dict:
    is_hold = meta["bird"].isin(hold).to_numpy()
    tr = sample_per_bird(meta, ~is_hold, n_train, rng)
    te = sample_per_bird(meta, is_hold, n_test, rng)
    res = {}

    def fit(y, kind):
        ytr, yte = y[tr], y[te]
        ok_tr, ok_te = pd.notna(ytr), pd.notna(yte)
        if kind == "reg":
            ok_tr &= np.isfinite(ytr.astype(float))
            ok_te &= np.isfinite(yte.astype(float))
        if ok_te.sum() < 50 or len(pd.unique(ytr[ok_tr])) < 2:
            return {"skipped": "too few"}
        if kind == "reg":
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(H[tr][ok_tr], ytr[ok_tr].astype(float))
            return {"r2": float(r2_score(yte[ok_te].astype(float), m.predict(H[te][ok_te]))), "n_test": int(ok_te.sum())}
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(H[tr][ok_tr], ytr[ok_tr].astype(str))
        if len(pd.unique(yte[ok_te])) < 2:
            return {"skipped": "one class in held-out birds"}
        return {"balanced_accuracy": float(balanced_accuracy_score(yte[ok_te].astype(str), m.predict(H[te][ok_te]))),
                "chance": 1 / len(pd.unique(ytr[ok_tr])), "n_test": int(ok_te.sum())}

    for t in WINDOW_BEHAVIOUR_REG:
        y = meta[t].to_numpy(dtype=float)
        if t == "vedba":
            y = np.log10(np.clip(y, 1e-3, None))
        res[t] = fit(y, "reg")
    for t in WINDOW_LABELS:
        res[t] = fit(meta[t].to_numpy(), "class")
    for t in WINDOW_NUISANCE:
        res[t + " (nuisance)"] = fit(meta[t].astype(str).to_numpy(), "class")
    return res


def cluster_stability(meta, H, rng, ks=(6, 8, 10, 12, 16, 20), per_bird=1000, n_boot=5) -> dict:
    birds = meta["bird"].to_numpy()
    allrows = np.ones(len(meta), bool)
    ev = sample_per_bird(meta, allrows, 300, rng)
    sc = StandardScaler().fit(H[ev])
    out = {}
    ub = np.unique(birds)
    for k in ks:
        ref = None
        aris = []
        for b in range(n_boot + 1):
            pick = ub if b == 0 else rng.choice(ub, len(ub), replace=True)
            idx = np.concatenate([rng.choice(np.flatnonzero(birds == x), min(per_bird, int((birds == x).sum())), replace=False) for x in pick])
            km = KMeans(k, n_init=3, random_state=b).fit(sc.transform(H[idx]))
            lab = km.predict(sc.transform(H[ev]))
            if ref is None:
                ref = lab
            else:
                aris.append(float(adjusted_rand_score(ref, lab)))
        out[str(k)] = {"ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris))}
    return out


def _latest_model(run: Path, dev: str):
    import torch

    from .model import Encoder
    from .windows import GPS_CHANNELS

    ck = sorted(run.glob("ckpt_*.pt"))[-1]
    model = Encoder(len(GPS_CHANNELS)).to(dev)
    model.load_state_dict(torch.load(ck, map_location=dev)["model"])
    return model


def invariance_tests(run, windows, meta, H, hold, rng, dev, k=12, n=20000, range_1024_g=2.0, step_512_g=1 / 512) -> dict:
    """Same-state rates after changing only the recording format of held-out windows (see module docstring).

    Format changes are applied in the sensor frame, then the run's own preparation (common format, rotation)
    is applied exactly as in training, so the test is valid for runs with and without the format fix.
    """
    from .train import encode

    cfg = json.loads((run / "config.json").read_text())
    step_g, clip_g = float(cfg.get("step_g", 0)), float(cfg.get("clip_g", 0))
    model = _latest_model(run, dev)
    is_hold = meta["bird"].isin(hold).to_numpy()
    tr = sample_per_bird(meta, ~is_hold, 1000, rng)
    sc = StandardScaler().fit(H[tr])
    km = KMeans(k, n_init=3, random_state=0).fit(sc.transform(H[tr]))

    data = dataset.load(windows, birds=sorted(hold), rotate=False)
    m = data.meta
    R = {r.bird: np.asarray(r.R, np.float32).reshape(3, 3) for r in data.orientation.itertuples()}
    Rrow = np.stack([R[b] for b in m["bird"]])

    def states(sel, acc, am):
        a = dataset.common_format(acc, am, step_g, clip_g).astype(np.float32)
        a = np.einsum("ntj,nij->nti", a, Rrow[sel]) * am[..., None]
        h = encode(model, a, am, data.gps[sel], data.gps_mask[sel], dev)[0]
        return km.predict(sc.transform(h))

    out = {"k": k}
    allrows = np.arange(len(m))
    st = states(allrows, data.acc, data.acc_mask)
    seg, bird = m["segment"].to_numpy(), m["bird"].to_numpy()
    nxt = (seg[1:] == seg[:-1]) & (bird[1:] == bird[:-1])
    out["same_state_next_window"] = float((st[1:][nxt] == st[:-1][nxt]).mean())

    # a long window versus one 40- or 64-sample burst of it, moved to the window start (as real short bouts)
    long_ = np.flatnonzero(m["acc_samples"].to_numpy() >= 180)
    sel = np.sort(rng.choice(long_, min(n, len(long_)), replace=False))
    L = rng.choice([40, 64], len(sel))
    start = (rng.random(len(sel)) * (200 - L)).astype(int)
    t = np.arange(200)[None]
    src = np.minimum(start[:, None] + t, 199)
    am = np.take_along_axis(data.acc_mask[sel], src, 1) & (t < L[:, None])
    a = np.take_along_axis(data.acc[sel], src[..., None], 1) * am[..., None]
    out["burst_same_state"] = float((st[sel] == states(sel, a, am)).mean())
    out["burst_n"] = int(len(sel))

    kk = m["sensitivity_k"].to_numpy()
    swaps = [
        ("k1_regridded_to_512", kk == 1, lambda a: np.round(a.astype(np.float32) / step_512_g) * step_512_g),
        ("k0_clipped_to_1024_range", kk == 0, lambda a: np.clip(a.astype(np.float32), -range_1024_g, range_1024_g)),
    ]
    for name, rows, fn in swaps:
        rows = np.flatnonzero(rows)
        if len(rows) < 50:
            out[name + "_same_state"] = None
            continue
        sel = np.sort(rng.choice(rows, min(n, len(rows)), replace=False))
        am = data.acc_mask[sel]
        out[name + "_same_state"] = float((st[sel] == states(sel, fn(data.acc[sel]) * am[..., None], am)).mean())
        out[name + "_n"] = int(len(sel))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--range-1024-g", type=float, default=2.0, help="+-g range of the 1024 counts/g setting")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    meta, H = load_run(args.run, args.windows)
    hold = json.loads((args.run / "holdout.json").read_text())

    # 1. hour level, same suite as the first-draft model
    meta["hour"] = meta["t_start"].dt.floor("h")
    hcols = [f"h_{i}" for i in range(H.shape[1])]
    hdf = pd.DataFrame(H, columns=hcols)
    hdf["bird"], hdf["hour"] = meta["bird"].to_numpy(), meta["hour"].to_numpy()
    hour_h = hdf.groupby(["bird", "hour"])[hcols].mean().reset_index()
    tgt = probes.hour_targets(args.windows)
    df = hour_h.merge(tgt, on=["bird", "hour"], how="inner")
    mag = {p.stem for p in (args.events / "magnetometer").glob("*.parquet")}
    res = {"hour_level": probes.hour_suite(df, df[hcols].to_numpy(float), mag), "hours": int(len(df))}
    # 2. window level on held-out birds
    res["window_level_holdout"] = holdout_window_probes(meta, H, hold, rng)
    res["holdout_birds"] = hold
    # 3. cluster stability
    res["cluster_stability_h"] = cluster_stability(meta, H, rng)
    # 4. invariance to recording format
    res["invariance"] = invariance_tests(args.run, args.windows, meta, H, hold, rng, args.device, range_1024_g=args.range_1024_g)
    (args.run / "evaluation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
