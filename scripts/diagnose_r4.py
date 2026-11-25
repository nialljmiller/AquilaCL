"""Diagnose a trained run (r4): what the sensitivity probe reads, and where bird identity comes from.

Part 0  Reproduction check. Windows are loaded in the sensor frame without the common format; the model
        input is rebuilt here exactly as dataset.load does it (common_format(step_g, clip_g) per sensor axis,
        then the run's own rotation R from orientation.json, mask; GPS as dataset.load gives it, with the run's
        masked channels hidden). A few hundred windows are encoded with it and compared with the stored
        embeddings (made on the GPU under bf16 autocast, saved as float16). A CPU bf16 encoding gives the noise
        floor; two deliberately wrong pipelines (no common format, no rotation) show that the check can fail.
Part 1  Sensitivity setting: counterfactual amplitude test. Full 1024 counts/g (k=1) windows are re-quantised
        to the 512 counts/g step with a random grid phase per window and axis, a' = round((a+p)/s0)*s0 - p,
        then prepared as above. Controls: k=0 windows re-gridded at their own step with a random phase, and
        k=0 windows clipped to +-2.0 g. Metrics: cosine distance to the unchanged window (also relative to the
        distance between a window and the same bird's next window), k-means state changes, and the share of
        counterfactual k=1 windows a sensitivity probe trained on real h (grouped by bird, out of fold) now
        calls k=0. High flip rate: the probe reads amplitude resolution; about 0: it reads recording mode.
        Matched probes: exactly full windows, those that are also still, and h re-encoded without GPS.
Part 2  Bird identity on held-out birds (fit first half of each record, test second half): by behaviour,
        under input perturbations of the test windows, per-bird residual yaw and tilt in the model frame, and
        handcrafted features with posture recomputed in the model frame.

Every encoding compared with another is made here on the same device, so numerical noise cancels.

Usage: python scripts/diagnose_r4.py --run <run dir> --windows <windows dir> --out <new .json> [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aquilacl import dataset
from aquilacl.baseline import FEATURES
from aquilacl.evaluate import _feature_matrix, _latest_model, load_run, sample_per_bird
from aquilacl.train import encode

S0_NOMINAL = 1 / 512  # g per count at 512 counts/g
CLIP_K1_G = 2.0  # range of the 1024 counts/g setting used for the k0 -> k1-like control
NEXT_MAX_S = 60.0


# --------------------------------------------------------------------------- small numeric helpers
def requantize(a: np.ndarray, step, phase) -> np.ndarray:
    """Round ``a`` onto the grid {k*step - phase}; ``step`` and ``phase`` broadcast against ``a``."""
    a = np.asarray(a, np.float32)
    return (np.round((a + phase) / step) * step - phase).astype(np.float32)


def masked_mean(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(N, T, C) mean over present samples; NaN for windows without samples."""
    n = m.sum(1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (a * m[..., None]).sum(1) / n, np.nan)


def axial_angle(xy: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal direction of (N, T, 2) horizontal dynamic acceleration (present samples, window mean removed).

    Returns the axial angle in degrees in [0, 180) from +x towards +y, and the eigenvalue ratio l1/l2
    (near 1: no preferred direction, the angle means little).
    """
    mf = m.astype(np.float32)[..., None]
    n = np.maximum(mf.sum(1), 1)
    d = (xy - (xy * mf).sum(1, keepdims=True) / n[:, None]) * mf
    cxx, cyy = (d[..., 0] ** 2).sum(1) / n[:, 0], (d[..., 1] ** 2).sum(1) / n[:, 0]
    cxy = (d[..., 0] * d[..., 1]).sum(1) / n[:, 0]
    theta = np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy)) % 180.0
    r = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy**2)
    l1, l2 = (cxx + cyy) / 2 + r, (cxx + cyy) / 2 - r
    return theta, l1 / np.maximum(l2, 1e-12)


def axial_mean(theta_deg: np.ndarray) -> tuple[float, float, float]:
    """Circular mean (deg, mod 180), circular standard deviation (deg) and resultant length of axial angles."""
    z = np.exp(2j * np.radians(np.asarray(theta_deg, float))).mean()
    R = float(abs(z))
    return float(np.degrees(np.angle(z)) / 2 % 180), float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12))) / 2)), R


def rot_z(deg: np.ndarray) -> np.ndarray:
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    R = np.zeros((len(t), 3, 3), np.float32)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    return R


def rot_random(n: int, max_deg: float, rng) -> np.ndarray:
    """Rotations about random axes by angles uniform in +-max_deg (as train.random_rotation)."""
    k = rng.normal(size=(n, 3))
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    ang = np.radians(rng.uniform(-max_deg, max_deg, n))
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(0, 2, 1)
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return (np.eye(3) + s * K + (1 - c) * K @ K).astype(np.float32)


def cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def observed_step(a: np.ndarray, m: np.ndarray, lim: float = 0.25) -> np.ndarray:
    """Per-axis quantisation step: per window the median spacing of distinct present values below ``lim`` g
    (float16 resolution there is <= 2^-13 g), then the median over windows. NaN where nothing qualifies."""
    est = np.full((len(a), 3), np.nan)
    for j in range(len(a)):
        for ax in range(3):
            v = np.unique(a[j, m[j], ax])
            v = v[np.abs(v) < lim]
            dv = np.diff(v)
            dv = dv[dv > 1e-6]
            if len(dv) >= 3:
                est[j, ax] = np.median(dv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an axis without values below lim gives NaN
        return np.nanmedian(est, axis=0)


def clean(o):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def dist_summary(d: np.ndarray) -> dict:
    return {"median": float(np.median(d)), "mean": float(np.mean(d)), "p90": float(np.percentile(d, 90)), "n": int(len(d))}


def logreg(max_iter=2000):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, class_weight="balanced"))


def grouped_probe(X, y, groups):
    """Balanced accuracy of a grouped (by bird) cross-validated logistic probe, and bird -> fold model that
    never saw that bird."""
    n_groups = len(np.unique(groups))
    pred = np.empty(len(y), dtype=y.dtype)
    models = {}
    for tr, te in GroupKFold(min(5, n_groups)).split(X, y, groups):
        mdl = logreg().fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
        models.update({b: mdl for b in np.unique(groups[te])})
    return float(balanced_accuracy_score(y, pred)), models


def predict_oof(models, X, groups):
    out = np.empty(len(X), dtype=int)
    for b in np.unique(groups):
        i = groups == b
        out[i] = models[b].predict(X[i])
    return out


def id_score(y, pred, rng, n_boot=2000) -> dict:
    """Balanced accuracy as the mean per-bird recall, with a 95 % bootstrap interval over birds (probe fixed)."""
    birds = np.unique(y)
    rec = np.array([(pred[y == b] == b).mean() for b in birds])
    boots = rec[rng.integers(0, len(rec), (n_boot, len(rec)))].mean(1)
    return {"balanced_accuracy": float(rec.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_birds": int(len(birds)), "n": int(len(y)), "chance": 1 / len(birds),
            "per_bird_recall": dict(zip(birds, rec.round(4))), "per_bird_n": {b: int((y == b).sum()) for b in birds}}


def drop_ci(y, pred0, pred1, rng, n_boot=2000) -> list[float]:
    birds = np.unique(y)
    d = np.array([(pred0[y == b] == b).mean() - (pred1[y == b] == b).mean() for b in birds])
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


# --------------------------------------------------------------------------- model input, rebuilt
class Pipeline:
    """Sensor-frame windows (as read from acc.npy) -> model input -> h, exactly as dataset.load + train.embed."""

    def __init__(self, windows: Path, data, R: dict, step_g: float, clip_g: float, model, dev: str):
        self.data, self.R, self.step_g, self.clip_g, self.model, self.dev = data, R, step_g, clip_g, model, dev
        m = data.meta
        self.bird = m["bird"].to_numpy()
        self.pos = m.groupby("bird").cumcount().to_numpy()
        self.raw = {b: np.load(windows / b / "acc.npy", mmap_mode="r") for b in np.unique(self.bird)}
        self.raw_dtype = {b: str(a.dtype) for b, a in self.raw.items()}
        for b, a in self.raw.items():
            if len(a) != int((self.bird == b).sum()):
                raise ValueError(f"{b}: acc.npy rows differ from loaded windows")

    def sensor(self, rows) -> np.ndarray:
        """Sensor-frame acceleration in g in its stored precision (as float32, which is exact)."""
        out = np.empty((len(rows), *self.data.acc.shape[1:]), np.float32)
        b = self.bird[rows]
        for x in np.unique(b):
            i = np.flatnonzero(b == x)
            out[i] = self.raw[x][self.pos[rows[i]]].astype(np.float32)
        return out

    def model_input(self, rows, a, am, fmt=True, rotate=True) -> np.ndarray:
        if fmt and (self.step_g > 0 or self.clip_g > 0):
            x = dataset.common_format(a, am, self.step_g, self.clip_g)
        else:
            x = a.astype(np.float16)
        if not rotate:
            return x
        out = np.empty_like(x)
        b = self.bird[rows]
        for k in np.unique(b):
            i = np.flatnonzero(b == k)
            R = self.R[k]
            out[i] = (x[i].astype(np.float32) @ R.T).astype(np.float16) * am[i][:, :, None]
        return out

    def h(self, rows, a=None, am=None, no_gps=False, post=None, fmt=True, rotate=True, bf16=False):
        """h for loaded rows; ``a`` replaces the sensor-frame input, ``post(x, am)`` changes the model-frame input."""
        am = self.data.acc_mask[rows] if am is None else am
        a = self.sensor(rows) if a is None else a
        x = self.model_input(rows, a, am, fmt, rotate)
        if post is not None:
            x = post(x, am)
        g, gm = self.data.gps[rows], self.data.gps_mask[rows]
        if no_gps:
            g, gm = np.zeros_like(g), np.zeros_like(gm)
        if not bf16:
            return encode(self.model, x, am, g, gm, self.dev)[0], x
        self.model.eval()
        H = []
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            for s in range(0, len(x), 4096):
                sl = slice(s, s + 4096)
                h, _ = self.model(torch.as_tensor(x[sl]).float(), torch.as_tensor(am[sl]), torch.as_tensor(g[sl]).float(),
                                  torch.as_tensor(gm[sl]))
                H.append(h.float().numpy())
        return np.concatenate(H), x


def rotate_post(Rw: np.ndarray):
    def f(x, am):
        return (np.einsum("ntj,nij->nti", x.astype(np.float32), Rw)).astype(np.float16) * am[:, :, None]
    return f


# --------------------------------------------------------------------------- parts
def part0_reproduction(P, H_stored_rows, rows) -> dict:
    h_cpu, _ = P.h(rows)
    h_bf, _ = P.h(rows, bf16=True)
    h_nofmt, _ = P.h(rows, fmt=False)
    h_norot, _ = P.h(rows, rotate=False)
    hmax = float(np.abs(H_stored_rows).max())

    def cmp(a, b):
        d = np.abs(a - b)
        cs = 1 - cos_dist(a, b)
        return {"max_abs": float(d.max()), "median_abs": float(np.median(d)), "max_abs_over_max_h": float(d.max() / hmax),
                "min_cos": float(cs.min()), "median_cos": float(np.median(cs))}

    out = {"n": int(len(rows)), "max_abs_h": hmax,
           "stored_vs_cpu_fp32": cmp(H_stored_rows, h_cpu),
           "stored_vs_cpu_bf16": cmp(H_stored_rows, h_bf),
           "cpu_bf16_vs_cpu_fp32 (noise floor)": cmp(h_bf, h_cpu),
           "fp16_storage_resolution": hmax * 2.0**-11,
           "wrong_pipeline_no_common_format_vs_stored": cmp(H_stored_rows, h_nofmt),
           "wrong_pipeline_no_rotation_vs_stored": cmp(H_stored_rows, h_norot)}
    tol = 3 * max(out["cpu_bf16_vs_cpu_fp32 (noise floor)"]["max_abs"], out["fp16_storage_resolution"])
    got = out["stored_vs_cpu_fp32"]["max_abs"]
    out["tolerance_max_abs"] = tol
    out["pass"] = bool(got <= tol)
    # the rotation changes the input by up to the mounting tilt, so a wrong rotation must show; the common format
    # changes it by at most half a grid step, which a well-trained model may legitimately ignore (reported only)
    out["discriminates"] = bool(out["wrong_pipeline_no_rotation_vs_stored"]["max_abs"] > tol)
    return out


def part1_sensitivity(P, meta, H, l_of_g, feats, km, sc, args, rng) -> dict:
    data = P.data
    lm = data.meta
    k_all = meta["sensitivity_k"].to_numpy()
    res = {"s0_nominal_g": S0_NOMINAL, "clip_k1_g": CLIP_K1_G,
           "phase": "uniform in [0, s0) per window and sensor axis; same phase draw for nominal and observed step"}

    # per bird, per axis observed steps (sensor frame, as stored), k0 and k1
    steps = {}
    both_birds = sorted({b for b, g in meta[meta["acc_samples"] >= 190].groupby("bird")["sensitivity_k"] if g.nunique() > 1})
    for b in both_birds:
        lr = np.flatnonzero(lm["bird"].to_numpy() == b)
        kk = lm["sensitivity_k"].to_numpy()[lr]
        full = lm["acc_samples"].to_numpy()[lr] >= 190
        steps[b] = {}
        for c in (0, 1):
            r = lr[full & (kk == c)]
            r = np.sort(rng.choice(r, min(300, len(r)), replace=False))
            steps[b][c] = observed_step(P.sensor(r), data.acc_mask[r]) if len(r) else np.full(3, np.nan)
    res["observed_step_g"] = {b: {f"k{c}": v.round(7) for c, v in s.items()} for b, s in steps.items()}
    s0_obs = {b: np.where(np.isfinite(s[0]), s[0], S0_NOMINAL) for b, s in steps.items()}
    res["observed_step_note"] = ("per bird and sensor axis, median over <=300 full windows of the median spacing of distinct "
                                 "values below 0.25 g; where no k0 estimate exists the nominal 1/512 g is used")
    res["observed_k0_step_over_nominal"] = dist_summary(np.concatenate([v / S0_NOMINAL for v in s0_obs.values()])) if s0_obs else None

    # reference: distance between a window and the same bird's next window (<= 60 s later), CPU encodings
    t = ((lm["t_start"] - lm["t_start"].min()) / pd.Timedelta(seconds=1)).to_numpy()
    lb = lm["bird"].to_numpy()
    nxt_ok = np.r_[(lb[1:] == lb[:-1]) & (np.diff(t) > 0) & (np.diff(t) <= NEXT_MAX_S), False]

    still = ((meta["label_perched"] == 1) | (meta["vedba"] < 0.02)).to_numpy()
    for name, full in (("ge190", meta["acc_samples"].to_numpy() >= 190), ("eq200", meta["acc_samples"].to_numpy() == 200)):
        r = {}
        rows = []
        for b in both_birds:
            ii = np.flatnonzero(full & (meta["bird"].to_numpy() == b))
            kk = k_all[ii]
            if len(np.unique(kk)) < 2:
                continue
            for c in (0, 1):
                x = ii[kk == c]
                rows.append(rng.choice(x, min(args.per_class, len(x)), replace=False))
        if not rows:
            res[name] = {"skipped": "no birds with both settings"}
            continue
        g_rows = np.sort(np.concatenate(rows))
        y, grp = k_all[g_rows].astype(int), meta["bird"].to_numpy()[g_rows]
        r["birds"] = int(len(np.unique(grp)))
        r["n_k0"], r["n_k1"] = int((y == 0).sum()), int((y == 1).sum())
        r["probe_stored_h_balanced_accuracy"], models = grouped_probe(H[g_rows], y, grp)
        r["probe_features_balanced_accuracy"], _ = grouped_probe(feats[g_rows], y, grp)

        sel = {c: np.sort(rng.choice(g_rows[y == c], min(args.n, int((y == c).sum())), replace=False)) for c in (0, 1)}
        base, d_next = {}, {}
        for c in (0, 1):
            lr = l_of_g[sel[c]]
            a = P.sensor(lr)
            am = data.acc_mask[lr]
            gb = meta["bird"].to_numpy()[sel[c]]
            h0, x0 = P.h(lr, a, am)
            base[c] = (lr, a, am, gb, h0, x0)
            pred0 = predict_oof(models, h0, gb)
            r[f"k{c}_original"] = {"n": int(len(lr)), "frac_predicted_k0": float((pred0 == 0).mean()),
                                   "n_present_samples_median": float(np.median(am.sum(1)))}
            # next-window reference
            has = nxt_ok[lr]
            hn, _ = P.h(lr[has] + 1)
            d_next[c] = cos_dist(h0[has], hn)
            r[f"k{c}_next_window_cos_dist"] = dist_summary(d_next[c])
            r[f"k{c}_next_window_state_change"] = float((km.predict(sc.transform(h0[has])) != km.predict(sc.transform(hn))).mean())
        ref = float(np.median(np.r_[d_next[0], d_next[1]]))
        r["next_window_cos_dist_median_pooled"] = float(ref)

        def evaluate_cf(c, label, a_cf):
            lr, a, am, gb, h0, x0 = base[c]
            h1, x1 = P.h(lr, a_cf, am)
            d = cos_dist(h0, h1)
            pres = am[..., None].repeat(3, -1)
            pred1 = predict_oof(models, h1, gb)
            r[label] = {"cos_dist": dist_summary(d), "cos_dist_over_next_window_median": float(np.median(d) / ref),
                        "state_change": float((km.predict(sc.transform(h0)) != km.predict(sc.transform(h1))).mean()),
                        "frac_predicted_k0": float((pred1 == 0).mean()),
                        "frac_model_input_samples_changed": float((x0 != x1)[pres].mean()),
                        "mean_abs_model_input_change_g": float(np.abs(x0.astype(np.float32) - x1.astype(np.float32))[pres].mean())}

        for c in (0, 1):
            lr, a, am, gb, h0, x0 = base[c]
            phase01 = rng.random((len(lr), 1, 3)).astype(np.float32)
            s_obs = np.stack([s0_obs[b] for b in gb])[:, None, :].astype(np.float32)
            tag = "k1_to_k0" if c == 1 else "k0_regrid_control"
            evaluate_cf(c, f"{tag}_nominal_step", requantize(a, S0_NOMINAL, phase01 * S0_NOMINAL) * am[..., None])
            evaluate_cf(c, f"{tag}_observed_step", requantize(a, s_obs, phase01 * s_obs) * am[..., None])
        lr, a, am = base[0][:3]
        pres = am[..., None].repeat(3, -1)
        r["k0_clip_2g_frac_raw_samples_beyond"] = float((np.abs(a) > CLIP_K1_G)[pres].mean())
        evaluate_cf(0, "k0_clip_2g", np.clip(a, -CLIP_K1_G, CLIP_K1_G))
        r["flip_rate_k1_to_k0_nominal"] = r["k1_to_k0_nominal_step"]["frac_predicted_k0"] - r["k1_original"]["frac_predicted_k0"]
        r["flip_rate_note"] = "share of counterfactual k1 windows predicted k0 minus share of unchanged k1 windows predicted k0 (same out-of-fold probe, CPU encodings)"
        res[name] = r

    # matched probes: exactly full windows; also still; h without GPS
    res["matched"] = {}
    for name, keep in (("eq200", meta["acc_samples"].to_numpy() == 200),
                       ("eq200_still", (meta["acc_samples"].to_numpy() == 200) & still)):
        rows = []
        for b in both_birds:
            ii = np.flatnonzero(keep & (meta["bird"].to_numpy() == b))
            kk = k_all[ii]
            if len(np.unique(kk)) < 2 or min((kk == 0).sum(), (kk == 1).sum()) < 20:
                continue
            for c in (0, 1):
                x = ii[kk == c]
                rows.append(rng.choice(x, min(args.per_class, len(x)), replace=False))
        if not rows:
            res["matched"][name] = {"skipped": "fewer than 20 windows per setting in every bird"}
            continue
        g_rows = np.sort(np.concatenate(rows))
        y, grp = k_all[g_rows].astype(int), meta["bird"].to_numpy()[g_rows]
        h_nogps, _ = P.h(l_of_g[g_rows], no_gps=True)
        res["matched"][name] = {
            "birds": int(len(np.unique(grp))), "n_k0": int((y == 0).sum()), "n_k1": int((y == 1).sum()),
            "h": grouped_probe(H[g_rows], y, grp)[0],
            "features": grouped_probe(feats[g_rows], y, grp)[0],
            "h_without_gps": grouped_probe(h_nogps, y, grp)[0],
        }
    return res


def part2_identity(P, meta, H, l_of_g, feats, hold, args, rng) -> dict:
    data = P.data
    y_all = meta["bird"].to_numpy()
    tst = meta["t_start"].to_numpy()
    subsets = {
        "still": ((meta["label_perched"] == 1) | (meta["vedba"] < 0.02)).to_numpy(),
        "night_still": ((meta["sun_elev_deg"] < -6) & (meta["vedba"] < 0.02)).to_numpy(),
        "flapping": (meta["label_flapping"] == 1).to_numpy(),
        "soaring_gliding": ((meta["label_circling"] == 1) | (meta["label_gliding"] == 1) | (meta["label_climbing_soar"] == 1)).to_numpy(),
    }
    first_half, second_half = np.zeros(len(meta), bool), np.zeros(len(meta), bool)
    tr, te, te_full = [], [], []
    for b in hold:
        idx = np.flatnonzero(y_all == b)
        if len(idx) < 100:
            continue
        idx = idx[np.argsort(tst[idx], kind="stable")]
        half = len(idx) // 2
        first_half[idx[:half]], second_half[idx[half:]] = True, True
        tr.append(rng.choice(idx[:half], min(args.per_bird, half), replace=False))
        te.append(rng.choice(idx[half:], min(args.per_bird, len(idx) - half), replace=False))
        te_full.append(rng.choice(idx[half:], min(args.per_bird_subset, len(idx) - half), replace=False))
    tr, te, te_full = (np.sort(np.concatenate(x)) for x in (tr, te, te_full))
    res = {"split": "per bird sorted by t_start; probe on first half, test on second half",
           "n_train": int(len(tr)), "n_test": int(len(te))}
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(H[tr], y_all[tr])
    res["stored_h"] = id_score(y_all[te], probe.predict(H[te]), rng)

    # by behaviour: probe trained on all, tested per subset (full second half, capped per bird); within subset
    res["by_behaviour"] = {}
    for name, s in subsets.items():
        r = {}
        tt = te_full[s[te_full]]
        cnt = pd.Series(y_all[tt]).value_counts()
        tt = tt[np.isin(y_all[tt], cnt.index[cnt >= args.min_subset])]
        r["train_all_test_subset"] = id_score(y_all[tt], probe.predict(H[tt]), rng) if len(tt) else None
        a, b_ = [], []
        for b in np.unique(y_all[tr]):
            i1 = np.flatnonzero(first_half & s & (y_all == b))
            i2 = np.flatnonzero(second_half & s & (y_all == b))
            if min(len(i1), len(i2)) < args.min_subset:
                continue
            a.append(rng.choice(i1, min(args.per_bird, len(i1)), replace=False))
            b_.append(rng.choice(i2, min(args.per_bird, len(i2)), replace=False))
        if len(a) >= 2:
            a, b_ = np.concatenate(a), np.concatenate(b_)
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(H[a], y_all[a])
            r["train_subset_test_subset"] = id_score(y_all[b_], m.predict(H[b_]), rng)
        else:
            r["train_subset_test_subset"] = None
        res["by_behaviour"][name] = r

    # perturbations of the test windows (probe fixed, trained on unperturbed stored h)
    lte = l_of_g[te]
    yte = y_all[te]
    a = P.sensor(lte)
    am = data.acc_mask[lte]
    pres = am[..., None]
    h0, _ = P.h(lte, a, am)
    p0 = probe.predict(h0)
    res["cpu_unperturbed"] = id_score(yte, p0, rng)
    n = len(lte)
    perturb = {
        "gps_removed": dict(no_gps=True),
        "acc_removed": dict(a=np.zeros_like(a), am=np.zeros_like(am)),
        "yaw_uniform_180deg": dict(post=rotate_post(rot_z(rng.uniform(-180, 180, n)))),
        "yaw_uniform_30deg": dict(post=rotate_post(rot_z(rng.uniform(-30, 30, n)))),
        "offset_uniform_0.02g_per_axis": dict(a=(a + rng.uniform(-0.02, 0.02, (n, 1, 3)).astype(np.float32)) * pres),
        "gain_uniform_3pct_per_axis": dict(a=a * rng.uniform(0.97, 1.03, (n, 1, 3)).astype(np.float32)),
        "rotation_3d_uniform_15deg": dict(post=rotate_post(rot_random(n, 15.0, rng))),
        "noise_gauss_0.01g": dict(a=(a + rng.normal(0, 0.01, a.shape).astype(np.float32)) * pres),
    }
    res["perturbation_frames"] = ("offset, gain and noise in the sensor frame before the common format (re-quantised like "
                                  "real data); yaw and 3-D rotation in the model frame after R; all drawn per window "
                                  "(and per axis) uniformly in the stated +- range; noise Gaussian with sd 0.01 g")
    res["perturbed"] = {}
    for name, kw in perturb.items():
        kw = dict(kw)
        if "a" not in kw:
            kw["a"] = a
        h1, _ = P.h(lte, **kw)
        p1 = probe.predict(h1)
        s = id_score(yte, p1, rng)
        s["drop"] = res["cpu_unperturbed"]["balanced_accuracy"] - s["balanced_accuracy"]
        s["drop_ci95"] = drop_ci(yte, p0, p1, rng)
        s["cos_dist_to_unperturbed"] = dist_summary(cos_dist(h0, h1))
        res["perturbed"][name] = s

    # residual yaw from flapping and residual tilt from still windows, in the model frame
    res["yaw_flapping"], res["static_still"] = {}, {}
    means = []
    lm_bird = data.meta["bird"].to_numpy()
    lflap = (data.meta["label_flapping"] == 1).to_numpy()
    lstill = ((data.meta["label_perched"] == 1) | (data.meta["vedba"] < 0.02)).to_numpy()
    n_acc = data.acc_mask.sum(1)
    for b in hold:
        r = np.flatnonzero((lm_bird == b) & lflap & (n_acc >= 40))
        if len(r) >= 20:
            r = np.sort(rng.choice(r, min(args.n_geom, len(r)), replace=False))
            amr = data.acc_mask[r]
            x = P.model_input(r, P.sensor(r), amr).astype(np.float32)
            th, ratio = axial_angle(x[..., :2], amr)
            mu, sd, R = axial_mean(th)
            good = ratio >= 2
            mu2, sd2, _ = axial_mean(th[good]) if good.sum() >= 10 else (np.nan, np.nan, np.nan)
            res["yaw_flapping"][b] = {"n": int(len(r)), "axial_mean_deg": mu, "axial_sd_deg": sd, "resultant": R,
                                      "eig_ratio_median": float(np.median(ratio)), "n_ratio_ge2": int(good.sum()),
                                      "axial_mean_deg_ratio_ge2": mu2, "axial_sd_deg_ratio_ge2": sd2}
            means.append(mu)
        r = np.flatnonzero((lm_bird == b) & lstill & (n_acc >= 20))
        if len(r) >= 20:
            r = np.sort(rng.choice(r, min(args.n_geom, len(r)), replace=False))
            amr = data.acc_mask[r]
            v = np.median(masked_mean(P.model_input(r, P.sensor(r), amr).astype(np.float32), amr), axis=0)
            nv = float(np.linalg.norm(v))
            res["static_still"][b] = {"n": int(len(r)), "median_vector_g": v.round(5), "norm_g": nv,
                                      "tilt_from_z_deg": float(np.degrees(np.arccos(np.clip(v[2] / nv, -1, 1)))),
                                      "tilt_azimuth_deg": float(np.degrees(np.arctan2(v[1], v[0])))}
    if means:
        mu, sd, R = axial_mean(np.array(means))
        res["yaw_across_birds"] = {"axial_mean_deg": mu, "axial_sd_deg": sd, "resultant": R, "n_birds": len(means)}
    tilts = [v["tilt_from_z_deg"] for v in res["static_still"].values()]
    if tilts:
        res["tilt_across_birds"] = {"median_deg": float(np.median(tilts)), "max_deg": float(np.max(tilts)), "n_birds": len(tilts)}

    # handcrafted features with posture in the sensor frame (stored, and recomputed from the window mean)
    # versus posture from the window mean in the model frame
    cols = [c for c in FEATURES if c in meta]
    ip, ir = cols.index("pitch_deg"), cols.index("roll_deg")
    rows = np.r_[tr, te]
    lr = l_of_g[rows]
    am_r = data.acc_mask[lr]
    a_r = P.sensor(lr)

    def posture(x):
        s = masked_mean(x.astype(np.float32), am_r)
        return np.degrees(np.arctan2(s[:, 0], np.hypot(s[:, 1], s[:, 2]))), np.degrees(np.arctan2(s[:, 1], s[:, 2]))

    def with_posture(pitch, roll):
        F = feats[rows].copy()
        F[:, ip] = np.where(np.isfinite(pitch), pitch, np.nanmedian(pitch))
        F[:, ir] = np.where(np.isfinite(roll), roll, np.nanmedian(roll))
        return F

    variants = {"features_stored": feats[rows],
                "features_sensor_frame_mean_posture": with_posture(*posture(a_r)),
                "features_model_frame_posture": with_posture(*posture(P.model_input(lr, a_r, am_r))),
                "features_no_posture": np.delete(feats[rows], [ip, ir], axis=1)}
    res["features_identity"] = {}
    ntr = len(tr)
    for name, F in variants.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(F[:ntr], y_all[tr])
        res["features_identity"][name] = id_score(y_all[te], m.predict(F[ntr:]), rng)
    return res


# --------------------------------------------------------------------------- main
def summary(out: dict) -> str:
    p0 = out["part0_reproduction"]
    L = [f"REPRODUCTION {'PASS' if p0['pass'] else 'FAIL'}: stored vs CPU max|dh| {p0['stored_vs_cpu_fp32']['max_abs']:.4g} "
         f"(tolerance {p0['tolerance_max_abs']:.4g}, |h|max {p0['max_abs_h']:.3g}), min cos {p0['stored_vs_cpu_fp32']['min_cos']:.5f}; "
         f"wrong rotation {'clearly worse' if p0['discriminates'] else 'NOT clearly worse (check cannot discriminate)'}: "
         f"no format {p0['wrong_pipeline_no_common_format_vs_stored']['max_abs']:.4g}, "
         f"no rotation {p0['wrong_pipeline_no_rotation_vs_stored']['max_abs']:.4g}"]
    p1 = out.get("part1_sensitivity", {})
    for name in ("ge190", "eq200"):
        r = p1.get(name)
        if not r or "skipped" in r:
            continue
        L.append(f"[sensitivity {name}] birds {r['birds']}; probe stored h {r['probe_stored_h_balanced_accuracy']:.4f}, "
                 f"features {r['probe_features_balanced_accuracy']:.4f}; k1 unchanged predicted k0 {r['k1_original']['frac_predicted_k0']:.3f}")
        for lab in ("k1_to_k0_nominal_step", "k1_to_k0_observed_step", "k0_regrid_control_nominal_step",
                    "k0_regrid_control_observed_step", "k0_clip_2g"):
            x = r[lab]
            L.append(f"   {lab}: cos dist med {x['cos_dist']['median']:.4g} ({x['cos_dist_over_next_window_median']:.3f} x next window), "
                     f"state change {x['state_change']:.3f}, predicted k0 {x['frac_predicted_k0']:.3f}, "
                     f"input samples changed {x['frac_model_input_samples_changed']:.3f}")
    for name, r in p1.get("matched", {}).items():
        if "skipped" not in r:
            L.append(f"[matched {name}] birds {r['birds']} n {r['n_k0']}/{r['n_k1']}: h {r['h']:.4f}, features {r['features']:.4f}, "
                     f"h without GPS {r['h_without_gps']:.4f}")
    p2 = out.get("part2_identity")
    if p2:
        def f(s):
            return "n/a" if s is None else f"{s['balanced_accuracy']:.3f} [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] ({s['n_birds']} birds)"
        L.append(f"[identity] stored h {f(p2['stored_h'])}; CPU {f(p2['cpu_unperturbed'])}; chance {p2['stored_h']['chance']:.3f}")
        for k, v in p2["by_behaviour"].items():
            L.append(f"   {k}: train all {f(v['train_all_test_subset'])}; within {f(v['train_subset_test_subset'])}")
        for k, v in p2["perturbed"].items():
            L.append(f"   perturbed {k}: {v['balanced_accuracy']:.3f}, drop {v['drop']:+.3f} [{v['drop_ci95'][0]:+.3f}, {v['drop_ci95'][1]:+.3f}]")
        if "yaw_across_birds" in p2:
            y = p2["yaw_across_birds"]
            L.append(f"   residual yaw (flapping) across birds: axial sd {y['axial_sd_deg']:.1f} deg over {y['n_birds']} birds")
        if "tilt_across_birds" in p2:
            t = p2["tilt_across_birds"]
            L.append(f"   residual tilt (still) from +z: median {t['median_deg']:.2f} deg, max {t['max_deg']:.2f} deg")
        for k, v in p2["features_identity"].items():
            L.append(f"   {k}: {f(v)}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new JSON file (refused if it exists)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=20000, help="windows per counterfactual set")
    ap.add_argument("--n-check", type=int, default=500, help="windows for the reproduction check")
    ap.add_argument("--per-class", type=int, default=1500, help="sensitivity probe windows per bird and setting")
    ap.add_argument("--per-bird", type=int, default=2000, help="identity probe windows per bird and half")
    ap.add_argument("--per-bird-subset", type=int, default=20000, help="second-half windows per bird for behaviour subsets")
    ap.add_argument("--min-subset", type=int, default=20, help="minimum windows per bird in a behaviour subset")
    ap.add_argument("--n-geom", type=int, default=5000, help="windows per bird for yaw and tilt")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    partial = args.out.with_name(args.out.name + ".partial")
    for p in (args.out, partial):
        if p.exists():
            raise SystemExit(f"{p} exists: give a new --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    rng = np.random.default_rng(args.seed)

    cfg = json.loads((args.run / "config.json").read_text())
    step_g, clip_g = float(cfg.get("step_g", 0)), float(cfg.get("clip_g", 0))
    mask_ch = tuple(cfg.get("mask_gps_channels", []))
    hold = json.loads((args.run / "holdout.json").read_text())
    R = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3) for o in json.loads((args.run / "orientation.json").read_text())}
    meta, H = load_run(args.run, args.windows)
    feats = _feature_matrix(meta)
    out = {"run": str(args.run), "windows": str(args.windows), "checkpoint": sorted(args.run.glob("ckpt_*.pt"))[-1].name,
           "step_g": step_g, "clip_g": clip_g, "mask_gps_channels": list(mask_ch), "seed": args.seed, "args": vars(args)}

    # birds needed: those with both settings in full windows, and the held-out birds
    full = meta[meta["acc_samples"] >= 190]
    both = sorted(b for b, g in full.groupby("bird")["sensitivity_k"] if g.nunique() > 1)
    need = sorted(set(both) | set(hold))
    out["birds_both_settings"], out["birds_loaded"] = len(both), len(need)
    print(f"loading {len(need)} birds ({len(both)} with both settings, {len(hold)} held out)", flush=True)
    data = dataset.load(args.windows, birds=need, step_g=0.0, clip_g=0.0, rotate=False, mask_gps_channels=mask_ch)
    model = _latest_model(args.run, args.device)
    P = Pipeline(args.windows, data, R, step_g, clip_g, model, args.device)
    out["raw_acc_dtype"] = sorted(set(P.raw_dtype.values()))
    rec = {o.bird: np.asarray(o.R, np.float32) for o in data.orientation.itertuples()}
    out["orientation_recomputed_vs_stored_max_abs"] = float(max(np.abs(rec[b].reshape(3, 3) - R[b]).max() for b in rec))

    # loaded rows <-> load_run rows (same bird, same position), checked on t_start
    l_of_g = np.full(len(meta), -1, int)
    gb, lb = meta["bird"].to_numpy(), data.meta["bird"].to_numpy()
    for b in need:
        gi, li = np.flatnonzero(gb == b), np.flatnonzero(lb == b)
        if len(gi) != len(li) or not (meta["t_start"].to_numpy()[gi] == data.meta["t_start"].to_numpy()[li]).all():
            raise ValueError(f"{b}: loaded windows do not match the run's embeddings")
        l_of_g[gi] = li

    def save():
        partial.write_text(json.dumps(clean(out), indent=1))

    loaded_g = np.flatnonzero(l_of_g >= 0)
    chk = np.sort(rng.choice(loaded_g, min(args.n_check, len(loaded_g)), replace=False))
    out["part0_reproduction"] = part0_reproduction(P, H[chk], l_of_g[chk])
    print(summary(out), flush=True)
    save()

    is_hold = meta["bird"].isin(hold).to_numpy()
    km_rows = sample_per_bird(meta, ~is_hold, 1000, rng)
    sc = StandardScaler().fit(H[km_rows])
    km = KMeans(args.k, n_init=3, random_state=0).fit(sc.transform(H[km_rows]))
    out["part1_sensitivity"] = part1_sensitivity(P, meta, H, l_of_g, feats, km, sc, args, rng) if both else {"skipped": "no birds with both settings"}
    save()
    print("part 1 done", flush=True)
    out["part2_identity"] = part2_identity(P, meta, H, l_of_g, feats, hold, args, rng)
    save()
    os.replace(partial, args.out)
    print("\n== summary\n" + summary(out))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
