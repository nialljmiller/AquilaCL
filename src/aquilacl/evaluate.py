"""Stage 4 evaluation of a trained run.

1. Hour-level probe suite (identical to the one applied to the first-draft embeddings): mean h per bird-hour.
2. Window-level probes on held-out birds only: linear models fitted on training-bird windows, scored on the
   birds that never entered pretraining. Behaviour targets should score high; device targets low.
3. Cluster stability: k-means on a bird-balanced sample of h for a range of k; adjusted Rand index between
   refits on bird-bootstrap samples (compare with the handcrafted baseline, ARI 0.35-0.44).
4. Invariance tests on held-out birds (needs the checkpoint): re-encode windows after changing only the
   recording format and count how often the k-means state stays the same.
   - burst: a full 10 s window versus a span of it as long as a real non-full window, moved to the window start;
   - sensitivity: a 1024 counts/g window re-gridded to 512 counts/g resolution, and a 512 counts/g window
     clipped to the narrower 1024 counts/g range.
   - GPS removed from daytime windows that have both sensors (which states acceleration alone can assign).
   Reference: how often the state stays the same between consecutive windows of one segment.
   A probe can always read recording length from the mask; what matters is whether states change.
   Once a run uses the common format, the two sensitivity swaps are identical by construction (score 1.0);
   for such runs the evidence is the full-window probe in (5).
5. Nuisance probes next to the same probes on handcrafted features: sensitivity setting on full windows
   (birds with both settings) and bird identity on held-out birds (later half of each record). The feature
   floor uses posture (pitch, roll) in the run's gravity-aligned frame (orientation.json, as in states.py);
   the sensor-frame values, which carry the tag mounting angle, are kept under *_features_sensor_frame. Bird
   identity is also given with 95 % intervals from a bootstrap over held-out birds (per-bird recall of the
   single fit, birds resampled with replacement) and as the paired difference h - features.

6. r7 probes (key r7_probes): information that rotation or yaw augmentation could remove, which the probes in (2)
   cannot see.
   - Window level on held-out birds, fitted on training-bird windows (as in (2)), separately in each subset of
     windows: all, full (acc_samples >= 190), short, k=0, k=1. Both the training and the test sample are drawn
     inside the subset. Targets: the probes of (2); pitch_rot_deg and roll_rot_deg (posture in the run's
     gravity-aligned frame, states.rotated_posture with orientation.json); dom_freq_hz_flapping (dominant
     wingbeat frequency, label_flapping == 1 full windows only); dyn_horizontal_share (share of the dynamic
     acceleration variance on the rotated x and y axes, (var_x + var_y) / (var_x + var_y + var_z)); turn_sign
     (turn_net_deg > 0 vs < 0, k=1 windows with GPS).
   - sex_flight (needs --reference): bird-level sex probe over ALL birds with known sex (the encoder never saw
     sex), on mean L2-normalised h within flight windows, with a ground-window control. Flight: the flight state
     of a K=2 states.parquet in --states2 (the state with the higher median speed_mean), else label-based
     (speed_mean > 5 m/s or label_flapping == 1; ground = label_perched == 1). Classifier, grouped CV, inner C
     selection and the between-bird permutation null are demography.run_classifiers. Tested (p values).
   The section uses its own generator (seed, 7), so its numbers are the same in a full run and with --only r7.
   The invariance section (4) also reports its same-state rates per subset (keys *_by_subset, new keys only).

Usage: python -m aquilacl.evaluate --run <run dir> --windows <windows dir> --events <data>/events
       [--only hour window cluster nuisance invariance r7] [--reference CSV] [--states2 DIR]
       [--n-perm 200] [--n-repeats 5] [--workers 1]
--only runs the named sections and writes evaluation_<section>.json for each (evaluation.json is untouched).
The sections then draw from a fresh random generator, so sampled rows differ from those of a full run.
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
SECTIONS = {"hour": "hour_level", "window": "window_level_holdout", "cluster": "cluster_stability_h",
            "nuisance": "nuisance_vs_features", "invariance": "invariance",
            "r7": "r7_probes"}  # --only name: result key; r7 last, so earlier sections draw as before
FULL_SAMPLES = 190  # full 10 s window
SUBSETS = ("all", "full", "short", "k0", "k1")
FLIGHT_SPEED_MS = 5.0  # label-based flight for the sex probe (as demography.FLAP_SPEED_MS)
SEX_MIN_WINDOWS = 50  # windows a bird needs in a mode (flight / ground) to enter the sex probe


def load_run(run: Path, windows: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metas, H = [], []
    for d in sorted(p for p in windows.iterdir() if p.is_dir()):
        if len(pd.read_parquet(d / "meta.parquet", columns=["bird"])) == 0:
            continue  # bird without kept windows (e.g. all calibrations unreliable): never embedded
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


def fit_probe(H, y, tr, te, kind) -> dict:
    """Linear probe fitted on rows ``tr`` and scored on rows ``te`` (``kind`` 'reg': ridge R2; else balanced accuracy)."""
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


def holdout_window_probes(meta, H, hold, rng, n_train=2000, n_test=5000) -> dict:
    is_hold = meta["bird"].isin(hold).to_numpy()
    tr = sample_per_bird(meta, ~is_hold, n_train, rng)
    te = sample_per_bird(meta, is_hold, n_test, rng)
    res = {}
    fit = lambda y, kind: fit_probe(H, y, tr, te, kind)

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


def invariance_tests(run, windows, meta, H, hold, rng, dev, k=12, n=20000) -> dict:
    """Same-state rates after changing only the recording format of held-out windows (see module docstring).

    Format changes are applied in the sensor frame, then the run's own preparation (common format, rotation)
    is applied exactly as in training, so the test is valid for runs with and without the format fix.
    """
    from .train import encode

    cfg = json.loads((run / "config.json").read_text())
    step_g, clip_g = float(cfg.get("step_g", 0)), float(cfg.get("clip_g", 0))
    fmt_path = windows / "format.json"
    fmt = json.loads(fmt_path.read_text()) if fmt_path.exists() else {"step_g": 1 / 512, "clip_g": 1.9}
    model = _latest_model(run, dev)
    is_hold = meta["bird"].isin(hold).to_numpy()
    tr = sample_per_bird(meta, ~is_hold, 1000, rng)
    sc = StandardScaler().fit(H[tr])
    km = KMeans(k, n_init=3, random_state=0).fit(sc.transform(H[tr]))

    data = dataset.load(windows, birds=sorted(hold), rotate=False, mask_gps_channels=tuple(cfg.get("mask_gps_channels", [])))
    m = data.meta
    R = {r.bird: np.asarray(r.R, np.float32).reshape(3, 3) for r in data.orientation.itertuples()}
    if (run / "orientation.json").exists():  # use the rotations the run was trained with
        R.update({o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3) for o in json.loads((run / "orientation.json").read_text())})
    Rrow = np.stack([R[b] for b in m["bird"]])

    def states(sel, acc, am, gm=None):
        a = dataset.common_format(acc, am, step_g, clip_g).astype(np.float32)
        a = np.einsum("ntj,nij->nti", a, Rrow[sel]) * am[..., None]
        gm = data.gps_mask[sel] if gm is None else gm
        h = encode(model, a, am, data.gps[sel] * gm, gm, dev)[0]
        return km.predict(sc.transform(h))

    out = {"k": k}
    allrows = np.arange(len(m))
    st = states(allrows, data.acc, data.acc_mask)
    seg, bird = m["segment"].to_numpy(), m["bird"].to_numpy()
    nxt = (seg[1:] == seg[:-1]) & (bird[1:] == bird[:-1])
    out["same_state_next_window"] = float((st[1:][nxt] == st[:-1][nxt]).mean())
    sub = subset_masks(m)  # both windows of a pair in the subset
    out["same_state_next_window_by_subset"] = rate_by_subset(
        (st[1:] == st[:-1])[nxt], {s: (v[1:] & v[:-1])[nxt] for s, v in sub.items()})

    # a full window versus a span as long as a real non-full window, moved to the window start
    n_acc = meta["acc_samples"].to_numpy()
    short_lengths = n_acc[(n_acc < 190) & (n_acc >= 24)]
    if len(short_lengths) == 0:
        short_lengths = np.array([40])  # no non-full windows in this data set (synthetic tests)
    long_ = np.flatnonzero(m["acc_samples"].to_numpy() >= 190)
    sel = np.sort(rng.choice(long_, min(n, len(long_)), replace=False))
    L = rng.choice(short_lengths, len(sel))
    start = (rng.random(len(sel)) * (200 - L)).astype(int)
    t = np.arange(200)[None]
    src = np.minimum(start[:, None] + t, 199)
    am = np.take_along_axis(data.acc_mask[sel], src, 1) & (t < L[:, None])
    a = np.take_along_axis(data.acc[sel], src[..., None], 1) * am[..., None]
    same = st[sel] == states(sel, a, am)
    out["burst_same_state"] = float(same.mean())
    out["burst_n"] = int(len(sel))
    out["burst_same_state_by_subset"] = rate_by_subset(same, {s: v[sel] for s, v in sub.items()})  # full only

    # GPS removed from daytime windows that have both sensors: which states can acceleration alone assign?
    both = np.flatnonzero((m["gps_fixes"].to_numpy() > 0) & (m["sun_elev_deg"].to_numpy() > 0))
    if len(both) >= 50:
        sel = np.sort(rng.choice(both, min(n, len(both)), replace=False))
        s1 = states(sel, data.acc[sel], data.acc_mask[sel], np.zeros_like(data.gps_mask[sel]))
        same = st[sel] == s1
        out["gps_removed_same_state"] = float(same.mean())
        out["gps_removed_same_state_by_state"] = {int(c): round(float(same[st[sel] == c].mean()), 3)
                                                  for c in np.unique(st[sel]) if (st[sel] == c).sum() >= 50}
        out["gps_removed_n"] = int(len(sel))
        out["gps_removed_same_state_by_subset"] = rate_by_subset(same, {s: v[sel] for s, v in sub.items()})

    kk = m["sensitivity_k"].to_numpy()
    swaps = [
        ("k1_regridded_to_512", kk == 1, lambda a: np.round(a.astype(np.float32) / fmt["step_g"]) * fmt["step_g"]),
        ("k0_clipped_to_1024_range", kk == 0, lambda a: np.clip(a.astype(np.float32), -fmt["clip_g"], fmt["clip_g"])),
    ]
    for name, rows, fn in swaps:
        rows = np.flatnonzero(rows)
        if step_g > 0 or clip_g > 0:  # identical inputs by construction: see nuisance_vs_features instead
            out[name + "_same_state"] = "not applicable (run uses the common format)"
            continue
        if len(rows) < 50:
            out[name + "_same_state"] = None
            continue
        sel = np.sort(rng.choice(rows, min(n, len(rows)), replace=False))
        am = data.acc_mask[sel]
        same = st[sel] == states(sel, fn(data.acc[sel]) * am[..., None], am)
        out[name + "_same_state"] = float(same.mean())
        out[name + "_n"] = int(len(sel))
        out[name + "_same_state_by_subset"] = rate_by_subset(same, {s: v[sel] for s, v in sub.items()})
    return out


def subset_masks(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row masks of the window subsets in SUBSETS: all, full (acc_samples >= 190), short, k0, k1."""
    full = meta["acc_samples"].to_numpy() >= FULL_SAMPLES
    k = meta["sensitivity_k"].to_numpy()
    return {"all": np.ones(len(meta), bool), "full": full, "short": ~full, "k0": k == 0, "k1": k == 1}


def rate_by_subset(same: np.ndarray, masks: dict[str, np.ndarray], min_n: int = 20) -> dict:
    """Share of True in ``same`` within each mask, with its n (None when n < min_n)."""
    return {k: {"same_state": float(same[v].mean()) if v.sum() >= min_n else None, "n": int(v.sum())}
            for k, v in masks.items()}


def _feature_matrix(meta, posture: np.ndarray | None = None):
    """Handcrafted-feature floor; ``posture`` (N, 2) replaces the sensor-frame pitch_deg, roll_deg."""
    from .baseline import FEATURES

    if posture is not None:
        meta = meta.assign(pitch_deg=posture[:, 0], roll_deg=posture[:, 1])
    X = meta[[c for c in FEATURES if c in meta]].to_numpy(float)
    miss = ~np.isfinite(X)
    X = np.where(miss, np.nanmedian(X, axis=0), X)
    return np.c_[X, ~np.isfinite(meta["speed_mean"].to_numpy(float)) if "speed_mean" in meta else miss.any(1)]  # + "no GPS"


def rotated_posture_rows(run: Path, windows: Path, meta: pd.DataFrame, rows: np.ndarray) -> np.ndarray | None:
    """(N, 2) pitch, roll in the run's gravity-aligned frame for ``rows`` of ``meta`` (NaN elsewhere).

    Uses states.rotated_posture with the rotations in <run>/orientation.json on the sensor-frame acc.npy of
    each bird; ``meta`` must hold each bird's windows contiguously and in file order (as load_run returns them).
    None if the run has no orientation.json.
    """
    path = run / "orientation.json"
    if not path.exists():
        return None
    from .states import rotated_posture

    R = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3) for o in json.loads(path.read_text())}
    birds = meta["bird"].to_numpy()
    within = meta.groupby("bird", sort=False).cumcount().to_numpy()
    rows = np.unique(rows)
    out = np.full((len(meta), 2), np.nan)
    for b in pd.unique(birds[rows]):
        if b not in R:
            raise ValueError(f"no rotation in {path} for {b}")
        r = rows[birds[rows] == b]
        acc = np.load(windows / str(b) / "acc.npy", mmap_mode="r")
        am = np.load(windows / str(b) / "acc_mask.npy", mmap_mode="r")
        out[r] = rotated_posture(acc[within[r]], am[within[r]], R[b])
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 200, seed: int = 0) -> np.ndarray:
    """95 % percentile intervals of the mean over the last axis, resampling its entries with replacement.

    ``values`` (n,) or (sets, n), e.g. per-bird recall (whose mean is the balanced accuracy). Every row uses the
    same resamples, so a row holding a difference of two others gives the paired interval. Returns (sets, 2).
    """
    v = np.atleast_2d(np.asarray(values, float))
    idx = np.random.default_rng(seed).integers(0, v.shape[1], (n_boot, v.shape[1]))
    return np.percentile(v[:, idx].mean(-1), [2.5, 97.5], axis=1).T


def per_class_recall(y: np.ndarray, pred: np.ndarray, classes) -> np.ndarray:
    return np.array([float((pred[y == c] == c).mean()) for c in classes])


def nuisance_vs_features(meta, H, hold, rng, per_class=1500, posture_fn=None, n_boot=200, boot_seed=0) -> dict:
    """Nuisance probes on h next to the same probes on handcrafted features (the level behaviour alone explains).

    sensitivity_full_windows: full windows of birds that use both settings, grouped by bird. bird_id_holdout: held-out birds, fit on the first half of each bird's time span, test on
    the second half. A value on h well above the feature value means the representation carries the nuisance
    beyond behaviour.

    ``posture_fn(rows)`` returns gravity-aligned (pitch, roll) for those rows (see rotated_posture_rows) or
    None; when given, the feature floor uses it and the sensor-frame floor is reported as *_sensor_frame. The
    bird-identity bootstrap resamples held-out birds (``n_boot`` times, own generator ``boot_seed``, so ``rng``
    is drawn from exactly as before).
    """
    from sklearn.model_selection import GroupKFold

    out = {}
    # sensitivity setting: full windows only (the setting follows burst type, which also sets window fullness),
    # birds that use both settings, grouped cross-validation by bird
    full = meta["acc_samples"].to_numpy() >= 190
    rows = []
    for b, idx in meta[full].groupby("bird").indices.items():
        ii = np.flatnonzero(full)[idx]
        kk = meta["sensitivity_k"].to_numpy()[ii]
        if len(np.unique(kk)) < 2:
            continue
        for c in np.unique(kk):
            r = ii[kk == c]
            rows.append(rng.choice(r, min(per_class, len(r)), replace=False))
    n_birds = len({meta["bird"].iat[r[0]] for r in rows})
    sens_rows = np.concatenate(rows) if n_birds >= 2 else np.array([], int)
    # bird identity among held-out birds, later half of each bird's record (rows drawn before any fit, as before)
    tr, te = [], []
    for b in hold:
        idx = np.flatnonzero(meta["bird"].to_numpy() == b)
        if len(idx) < 100:
            continue
        idx = idx[np.argsort(meta["t_start"].to_numpy()[idx], kind="stable")]
        half = len(idx) // 2
        tr.append(rng.choice(idx[:half], min(2000, half), replace=False))
        te.append(rng.choice(idx[half:], min(2000, len(idx) - half), replace=False))
    tr, te = np.concatenate(tr), np.concatenate(te)

    feats_sensor = _feature_matrix(meta)
    posture = posture_fn(np.concatenate([sens_rows, tr, te])) if posture_fn is not None else None
    sets = [("h", H), ("features", feats_sensor if posture is None else _feature_matrix(meta, posture))]
    if posture is not None:
        sets.append(("features_sensor_frame", feats_sensor))
    out["features_posture_frame"] = "sensor" if posture is None else "gravity_aligned"

    if n_birds >= 2:
        r = sens_rows
        y = meta["sensitivity_k"].to_numpy()[r]
        g = meta["bird"].to_numpy()[r]
        for name, F in sets:
            X = F[r]
            pred = np.empty(len(r), dtype=y.dtype)
            for ftr, fte in GroupKFold(min(5, n_birds)).split(X, y, g):
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X[ftr], y[ftr])
                pred[fte] = mdl.predict(X[fte])
            out[f"sensitivity_full_windows_{name}"] = float(balanced_accuracy_score(y, pred))
    out["sensitivity_full_windows_birds"] = int(n_birds)

    y = meta["bird"].to_numpy()
    classes = sorted(np.unique(y[te]))
    recall = {}
    for name, X in sets:
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X[tr], y[tr])
        pred = mdl.predict(X[te])
        out[f"bird_id_holdout_{name}"] = float(balanced_accuracy_score(y[te], pred))
        recall[name] = per_class_recall(y[te], pred, classes)
        out[f"bird_id_holdout_{name}_recall_by_bird"] = {str(c): round(float(v), 4) for c, v in zip(classes, recall[name])}
    names = [n for n, _ in sets]
    diffs = [f"h_minus_{n}" for n in names if n != "h"]
    rows_ = [recall[n] for n in names] + [recall["h"] - recall[d[len("h_minus_"):]] for d in diffs]
    ci = bootstrap_mean_ci(np.stack(rows_), n_boot, boot_seed)
    for name, c in zip(names + diffs, ci):
        out[f"bird_id_holdout_{name}_ci95"] = [float(c[0]), float(c[1])]
    for d in diffs:
        out[f"bird_id_holdout_{d}"] = float(out["bird_id_holdout_h"] - out[f"bird_id_holdout_{d[len('h_minus_'):]}"])
    out["bird_id_bootstrap"] = {"resamples": int(n_boot), "birds": int(len(classes)), "unit": "held-out bird"}
    out["bird_id_chance"] = 1 / len(np.unique(y[tr]))
    return out


# ---------------------------------------------------------------- r7 probes (see module docstring, section 6)

def dyn_horizontal_share(acc: np.ndarray, mask: np.ndarray, R: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """(var_x + var_y) / (var_x + var_y + var_z) of the dynamic acceleration rotated by R (sensor -> gravity on +z).

    Dynamic = acceleration minus features.static_component (the static part states.rotated_posture averages);
    variances over present samples only. NaN for windows with < 2 present samples or no dynamic variance.
    """
    from .features import static_component

    out = np.full(len(acc), np.nan)
    for i in range(0, len(acc), chunk):
        a = np.asarray(acc[i:i + chunk], np.float32)
        m = np.asarray(mask[i:i + chunk], bool)
        d = (a - static_component(a, m)) @ np.asarray(R, np.float32).T
        w = m[:, :, None].astype(np.float32)
        n = m.sum(1)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = (d * w).sum(1) / n
            var = (((d - mu[:, None]) ** 2) * w).sum(1) / n
            tot = var.sum(1)
            share = (var[:, 0] + var[:, 1]) / tot
        out[i:i + chunk] = np.where((n[:, 0] >= 2) & (tot > 0), share, np.nan)
    return out


def rotated_window_targets(run: Path, windows: Path, meta: pd.DataFrame, rows: np.ndarray) -> np.ndarray | None:
    """(N, 3) pitch_rot_deg, roll_rot_deg, dyn_horizontal_share in the run's gravity-aligned frame for ``rows``.

    NaN elsewhere. Same frame and row mapping as rotated_posture_rows (orientation.json, birds contiguous and in
    file order in ``meta``). None if the run has no orientation.json.
    """
    path = run / "orientation.json"
    if not path.exists():
        return None
    from .states import rotated_posture

    R = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3) for o in json.loads(path.read_text())}
    birds = meta["bird"].to_numpy()
    within = meta.groupby("bird", sort=False).cumcount().to_numpy()
    rows = np.unique(rows)
    out = np.full((len(meta), 3), np.nan)
    for b in pd.unique(birds[rows]):
        if b not in R:
            raise ValueError(f"no rotation in {path} for {b}")
        r = rows[birds[rows] == b]
        acc = np.load(windows / str(b) / "acc.npy", mmap_mode="r")
        am = np.load(windows / str(b) / "acc_mask.npy", mmap_mode="r")
        a, m = np.asarray(acc[within[r]]), np.asarray(am[within[r]])
        out[r, :2] = rotated_posture(a, m, R[b])
        out[r, 2] = dyn_horizontal_share(a, m, R[b])
    return out


R7_ROTATED = ["pitch_rot_deg", "roll_rot_deg", "dyn_horizontal_share"]


def _col(meta, c, dtype=float):
    return meta[c].to_numpy(dtype) if c in meta else None


def r7_window_probes(meta, H, hold, rng, targets_fn=None, n_train=1000, n_test=2000) -> dict:
    """Window-level probes on held-out birds in each subset of SUBSETS (fit and test inside the subset).

    ``targets_fn(rows)`` returns rotated_window_targets for those rows (or None). Domains: rotated targets and the
    probes of holdout_window_probes use every window of the subset; dom_freq_hz_flapping uses label_flapping == 1
    full windows; turn_sign uses k=1 windows with GPS and turn_net_deg != 0.
    """
    is_hold = meta["bird"].isin(hold).to_numpy()
    sub = subset_masks(meta)
    full = sub["full"]
    k = meta["sensitivity_k"].to_numpy()
    flap = _col(meta, "label_flapping")
    dom = _col(meta, "dom_freq_hz")
    turn = _col(meta, "turn_net_deg")
    gps = meta["gps_fixes"].to_numpy() > 0 if "gps_fixes" in meta else np.zeros(len(meta), bool)
    domains = {"any": np.ones(len(meta), bool)}
    if flap is not None and dom is not None:
        domains["flap_full"] = (flap == 1) & full & np.isfinite(dom)
    if turn is not None:
        domains["turn"] = (k == 1) & gps & np.isfinite(turn) & (turn != 0)

    def draw(rows):
        if not (rows & ~is_hold).any() or not (rows & is_hold).any():
            return None
        return sample_per_bird(meta, rows & ~is_hold, n_train, rng), sample_per_bird(meta, rows & is_hold, n_test, rng)

    samples = {(s, d): draw(sub[s] & dm) for s in SUBSETS for d, dm in domains.items()}
    rot = None
    if targets_fn is not None:
        any_rows = [x for (s, d), v in samples.items() if d == "any" and v is not None for x in v]
        rot = targets_fn(np.concatenate(any_rows)) if any_rows else None

    out = {}
    for s in SUBSETS:
        res = {}
        any_s = samples[(s, "any")]
        fit = (lambda y, kind: fit_probe(H, y, *any_s, kind)) if any_s is not None else (lambda y, kind: {"skipped": "no windows"})
        for t in WINDOW_BEHAVIOUR_REG:
            if t not in meta:
                res[t] = {"skipped": "column missing"}
                continue
            y = meta[t].to_numpy(dtype=float)
            res[t] = fit(np.log10(np.clip(y, 1e-3, None)) if t == "vedba" else y, "reg")
        for t in WINDOW_LABELS:
            res[t] = fit(meta[t].to_numpy(), "class") if t in meta else {"skipped": "column missing"}
        for t in WINDOW_NUISANCE:
            res[t + " (nuisance)"] = fit(meta[t].astype(str).to_numpy(), "class") if t in meta else {"skipped": "column missing"}
        for j, t in enumerate(R7_ROTATED):
            res[t] = fit(rot[:, j], "reg") if rot is not None else {"skipped": "no orientation.json"}
        # wingbeat frequency in flapping full windows
        if s == "short":
            res["dom_freq_hz_flapping"] = {"skipped": "not applicable (full windows only)"}
        elif "flap_full" not in domains:
            res["dom_freq_hz_flapping"] = {"skipped": "column missing"}
        elif samples[(s, "flap_full")] is None:
            res["dom_freq_hz_flapping"] = {"skipped": "no windows"}
        else:
            res["dom_freq_hz_flapping"] = fit_probe(H, dom, *samples[(s, "flap_full")], "reg")
        # turn direction in continuous mode with GPS
        if s == "k0":
            res["turn_sign"] = {"skipped": "not applicable (k=1 windows only)"}
        elif "turn" not in domains:
            res["turn_sign"] = {"skipped": "column missing"}
        elif samples[(s, "turn")] is None:
            res["turn_sign"] = {"skipped": "no windows"}
        else:
            res["turn_sign"] = fit_probe(H, np.where(turn > 0, "positive", "negative"), *samples[(s, "turn")], "class")
        out[s] = res
    return out


def _bird_mean_h(H: np.ndarray, codes: np.ndarray, G: int, chunk: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Mean L2-normalised h per code 0..G-1 (code < 0 ignored) and the row count, in chunks of rows."""
    S, n = np.zeros((G, H.shape[1])), np.zeros(G, int)
    for i in range(0, len(H), chunk):
        c = codes[i:i + chunk]
        ok = c >= 0
        if not ok.any():
            continue
        h = H[i:i + chunk][ok].astype(np.float64)
        h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
        np.add.at(S, c[ok], h)
        n += np.bincount(c[ok], minlength=G)
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / n[:, None], n


def sex_flight_probe(meta, H, ref_csv: Path, states2: Path | None = None, state_col: str = "state_hmm",
                     n_perm: int = 200, n_repeats: int = 5, seed: int = 0, workers: int = 1,
                     min_windows: int = SEX_MIN_WINDOWS, run_name: str = "run", exclude=()) -> dict:
    """Bird-level sex probe (female = 1) on mean h in flight windows, with a ground-window control.

    All birds with known sex (the encoder never saw sex labels); pre-fledge windows of tagged-young birds are
    dropped first (as in demography). Uses demography.make_spec / run_classifiers: L2 logistic regression with
    inner grouped C selection, outer StratifiedGroupKFold by bird, ``n_repeats`` repeats, labels permuted
    between birds (``n_perm``) for p values of AUC and balanced accuracy.
    """
    from . import demography, reference

    birds = meta["bird"].to_numpy().astype(str)
    ub = pd.unique(birds)
    ref = reference.load_birds(ref_csv, list(ub)).set_index("bird")
    speed = _col(meta, "speed_mean")
    speed = np.full(len(meta), np.nan) if speed is None else speed
    keep = np.ones(len(meta), bool)
    n_pre = 0
    for b in ub:  # pre-fledge windows of tagged-young birds
        if pd.isna(ref.loc[b, "tagged_young"]) or pd.isna(ref.loc[b, "hatch_year"]):
            continue
        idx = np.flatnonzero(birds == b)
        t = meta["t_start"].iloc[idx]
        fl = reference.fledge_from_speed(t, speed[idx], ref.loc[b, "hatch_year"])
        age = reference.age_on({"tagged_young": True, "hatch_year": ref.loc[b, "hatch_year"]}, t, fl["fledge"])
        pre = (age["age_class"] == "pre_fledge").to_numpy()
        keep[idx[pre]] = False
        n_pre += int(pre.sum())

    out = {"unit": "bird", "target": "sex (female = 1)", "test": "tested: labels permuted between birds",
           "encoder_saw_sex": False, "min_windows_per_bird": int(min_windows), "pre_fledge_windows_dropped": n_pre}
    if states2 is not None:
        import pyarrow.parquet as pq

        cols = pq.read_schema(states2 / "states.parquet").names
        col = state_col if state_col in cols else "state"
        st = pd.read_parquet(states2 / "states.parquet", columns=["bird", "row", col])
        st["bird"] = st["bird"].astype(str)
        within = meta.groupby("bird", sort=False).cumcount().to_numpy()
        key = pd.MultiIndex.from_arrays([birds, within])
        pos = key.get_indexer(pd.MultiIndex.from_arrays([st["bird"].to_numpy(), st["row"].to_numpy()]))
        state = np.full(len(meta), -1)
        state[pos[pos >= 0]] = st[col].to_numpy()[pos >= 0]
        K = int(st[col].max()) + 1
        if K != 2:
            return {**out, "skipped": f"--states2 has K={K}, need K=2 (ground, flight)"}
        med = [np.nanmedian(np.where(state == c, speed, np.nan)) if np.isfinite(speed[state == c]).any() else -np.inf
               for c in range(2)]
        fs = int(np.argmax(med))
        flight, ground = state == fs, state == 1 - fs
        out.update(flight_definition=f"K=2 states ({states2.name}, column {col}): state {fs}, the one with the higher "
                   f"median speed_mean", flight_state=fs, state_rows_unmatched=int((pos < 0).sum()))
    else:
        flap, perched = _col(meta, "label_flapping"), _col(meta, "label_perched")
        flap = np.zeros(len(meta)) if flap is None else flap
        flight = (np.nan_to_num(speed, nan=-1) > FLIGHT_SPEED_MS) | (flap == 1)
        ground = (np.zeros(len(meta)) if perched is None else perched) == 1
        ground &= ~flight
        out["flight_definition"] = f"label-based: speed_mean > {FLIGHT_SPEED_MS} m/s or label_flapping == 1; ground: label_perched == 1"

    sex = ref["sex"].map({"f": 1, "m": 0})
    y = sex.dropna().astype(int)
    y = y[~y.index.isin(list(exclude))]  # locked test birds of a scorecard split: labels never looked at
    out.update(birds_excluded=int(len(exclude)))
    out.update(birds_known_sex=int(len(y)), birds_f=int((y == 1).sum()), birds_m=int((y == 0).sum()))
    bcode = pd.Index(ub).get_indexer(birds)
    specs, info = {}, {}
    for mode, sel in (("flight", flight), ("ground", ground)):
        M, n = _bird_mean_h(H, np.where(sel & keep, bcode, -1), len(ub))
        ok = (n >= min_windows) & pd.Index(ub).isin(y.index)
        F = pd.DataFrame(M[ok], index=pd.Index(ub[ok], name="bird"))
        info[mode] = {"birds_too_few_windows": int((pd.Index(ub).isin(y.index) & ~ok).sum()),
                      "windows_median_per_bird": float(np.median(n[ok])) if ok.any() else None}
        sp = demography.make_spec(F, y, "bird", run_name, "sex", f"mean_h_{mode}")
        if sp is None:
            info[mode]["skipped"] = "fewer than 2 birds of a sex"
        else:
            specs[mode] = sp
    if specs:
        clf, _ = demography.run_classifiers(specs, n_repeats, {"bird": n_perm}, seed, workers)
        for mode, row in zip(specs, clf.itertuples()):
            info[mode].update(n_birds=int(row.n_birds), n_f=int(row.n_pos_birds), n_m=int(row.n_neg_birds),
                              auc=float(row.auc_mean), auc_sd_over_repeats=float(row.auc_sd),
                              balanced_accuracy=float(row.bacc_mean), p_auc=float(row.p_auc), p_bacc=float(row.p_bacc),
                              null_auc_mean=float(row.null_auc_mean), null_auc_q95=float(row.null_auc_q95),
                              n_perm=int(row.n_perm), n_repeats=int(row.n_repeats))
    out.update(info)
    return out


def r7_probes(run: Path, windows: Path, meta, H, hold, rng, ref_csv=None, states2=None, state_col="state_hmm",
              n_perm=200, n_repeats=5, seed=0, workers=1, n_train=1000, n_test=2000) -> dict:
    out = {"subsets": {"all": "every window", "full": f"acc_samples >= {FULL_SAMPLES}",
                       "short": f"acc_samples < {FULL_SAMPLES}", "k0": "sensitivity_k == 0 (scheduled bursts)",
                       "k1": "sensitivity_k == 1 (continuous mode)"},
           "window_n_train_per_bird": n_train, "window_n_test_per_bird": n_test,
           "window_holdout": r7_window_probes(meta, H, hold, rng, lambda rows: rotated_window_targets(run, windows, meta, rows),
                                              n_train, n_test)}
    if ref_csv is None:
        out["sex_flight"] = {"skipped": "no --reference"}
    else:
        out["sex_flight"] = sex_flight_probe(meta, H, Path(ref_csv), None if states2 is None else Path(states2), state_col,
                                             n_perm, n_repeats, seed, workers, run_name=run.name,
                                             exclude=json.loads((run / "locked.json").read_text())
                                             if (run / "locked.json").exists() else ())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), default=None,
                    help="run only these sections; each writes evaluation_<section>.json instead of evaluation.json")
    ap.add_argument("--reference", type=Path, default=None, help="r7: reference CSV (sex); without it the sex probe is skipped")
    ap.add_argument("--states2", type=Path, default=None, help="r7: K=2 states dir (states.parquet) for flight windows")
    ap.add_argument("--state-col", default="state_hmm", help="r7: state column in --states2 (falls back to 'state')")
    ap.add_argument("--n-perm", type=int, default=200, help="r7: between-bird permutations of the sex probe")
    ap.add_argument("--n-repeats", type=int, default=5, help="r7: CV repeats of the sex probe")
    ap.add_argument("--workers", type=int, default=1, help="r7: joblib workers of the sex probe")
    ap.add_argument("--r7-n-train", type=int, default=1000, help="r7: training windows per bird and subset")
    ap.add_argument("--r7-n-test", type=int, default=2000, help="r7: held-out windows per bird and subset")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    meta, H = load_run(args.run, args.windows)
    hold = json.loads((args.run / "holdout.json").read_text())
    todo = list(SECTIONS) if args.only is None else [s for s in SECTIONS if s in args.only]

    res = {}
    for sec in todo:
        part = {}
        if sec == "hour":  # 1. hour level, same suite as the first-draft model
            meta["hour"] = meta["t_start"].dt.floor("h")
            hcols = [f"h_{i}" for i in range(H.shape[1])]
            hdf = pd.DataFrame(H, columns=hcols)
            hdf["bird"], hdf["hour"] = meta["bird"].to_numpy(), meta["hour"].to_numpy()
            hour_h = hdf.groupby(["bird", "hour"])[hcols].mean().reset_index()
            tgt = probes.hour_targets(args.windows)
            df = hour_h.merge(tgt, on=["bird", "hour"], how="inner")
            mag = {p.stem for p in (args.events / "magnetometer").glob("*.parquet")}
            part = {"hour_level": probes.hour_suite(df, df[hcols].to_numpy(float), mag), "hours": int(len(df))}
        elif sec == "window":  # 2. window level on held-out birds
            part = {"window_level_holdout": holdout_window_probes(meta, H, hold, rng), "holdout_birds": hold}
        elif sec == "cluster":  # 3. cluster stability
            part = {"cluster_stability_h": cluster_stability(meta, H, rng)}
        elif sec == "nuisance":  # 4. nuisance probes against the handcrafted-feature level
            part = {"nuisance_vs_features": nuisance_vs_features(
                meta, H, hold, rng, posture_fn=lambda rows: rotated_posture_rows(args.run, args.windows, meta, rows))}
        elif sec == "invariance":  # 5. invariance to recording format
            part = {"invariance": invariance_tests(args.run, args.windows, meta, H, hold, rng, args.device)}
        elif sec == "r7":  # 6. probes for information rotation/yaw invariance could remove; own generator
            part = {"r7_probes": r7_probes(args.run, args.windows, meta, H, hold, np.random.default_rng([args.seed, 7]),
                                           args.reference, args.states2, args.state_col, args.n_perm, args.n_repeats,
                                           args.seed, args.workers, args.r7_n_train, args.r7_n_test)}
        if args.only is not None:
            part.setdefault("holdout_birds", hold)
            (args.run / f"evaluation_{sec}.json").write_text(json.dumps(part, indent=1))
        res.update(part)
    if args.only is None:
        (args.run / "evaluation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
