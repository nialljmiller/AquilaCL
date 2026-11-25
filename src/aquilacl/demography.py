"""Bird-level demography from the learned latent space: sex, tagged-young status, wingbeat control, age development.

The unit of analysis is the BIRD. Cross-validation folds are grouped by bird (StratifiedGroupKFold), permutation
tests shuffle labels between birds, and bird-month samples are aggregated back to one prediction per bird before
scoring. Nothing is tested at window level.

Pass 1 (one worker per bird; meta/features are read once, then h and states of every run): pre-fledge windows of
tagged-young birds are dropped before anything else (only counted). h is L2-normalised per window (as in
aquilacl.states). Per bird and per bird-month (year-month with >= MONTH_MIN_WINDOWS kept windows): mean h, mean h
of windows with / without GPS, mean h within each state, state budget (share per state among windows with a state),
handcrafted-feature means (aquilacl.baseline.FEATURES with the run's gravity-aligned pitch/roll from
states.parquet, plus the share of windows without GPS speed, as in evaluate._feature_matrix), wingbeat medians,
age-month aggregates.

Outputs in --out (bird-keyed tables are runtime outputs only):
- birds.csv: per bird reference fields (no comment text), nuisance variables, fledge estimate, window counts.
- confounds.csv (a): each demographic (sex f vs m; tagged_young vs not stated; migrant vs not stated) against
  cohort, tag_model, tag_firmware, deploy month, has_magnetometer (if a column exists), track length, continuous-
  mode share, GPS share. Categorical: Fisher exact (2x2) or permutation chi-square; continuous: permutation of the
  difference in means, with AUC (Mann-Whitney) as effect size.
- features_<run>_{bird,month}.parquet (b).
- classifiers.csv, nested.csv, classifiers_<level>.png (c).
- wingbeat.csv, wingbeat_birds.csv, wingbeat_<run>.png (d).
- age_months_<run>.csv, age_same_months_<run>.csv, age_same_months_tests.csv, age_<run>.png (e).
- summary.md (f), metadata.json.

Classifier design (c): L2 logistic regression with class_weight='balanced' on median-imputed, standardised
features; C chosen inside each training fold from C_GRID by grouped inner CV (log loss). Outer CV:
StratifiedGroupKFold by bird, n_repeats repeats with different seeds. Metrics: bird-level ROC AUC and balanced
accuracy (bird-month predictions are averaged per bird first). Null: labels permuted between birds, folds
recomputed on the permuted labels, ONE repeat per permutation; the observed statistic is the mean over n_repeats,
so the null is wider than the observed statistic's sampling distribution and the p-value is conservative.
Nested comparison (bird level only): paired dAUC = AUC(h + nuisance) - AUC(nuisance) on identical folds; null by
Freedman-Lane: h is regressed on the nuisance features (least squares with intercept, all birds), and the residual
rows are permuted between birds and added back to the fitted part, with nuisance and labels fixed. This keeps the
part of h that the nuisance variables explain and breaks only what h carries beyond them.

Usage: python -m aquilacl.demography --run R1 --run R2 --states S1 --states S2 --windows W --reference CSV
       --out NEW [--n-perm 2000] [--n-perm-month 200] [--n-repeats 20] [--seed 0] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy import sparse, stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import reference
from .baseline import FEATURES as HC_FEATURES

FULL_SAMPLES = 190  # wingbeat: full 10 s windows only (frequency resolution depends on run length)
MONTH_MIN_WINDOWS = 200
MIN_BIRD_WINDOWS = 100
MIN_MODE_WINDOWS = 50  # within-mode control: windows of the mode a bird needs
WINGBEAT_MIN_WINDOWS = 20
AGE_MIN_WINDOWS = 200  # y2+ centroid and same-calendar-month periods
SAME_MONTHS = (7, 8, 9)
FLAP_SPEED_MS = 5.0
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
N_SPLITS, INNER_SPLITS = 5, 3
EXACT_SIGNFLIP_MAX = 16
HC_COLS = list(HC_FEATURES) + ["no_gps"]
AGE_HC = ["vedba", "flap_band_frac", "speed_mean", "climb_mean", "turn_abs_mean"]
FEATURE_SETS = ["nuisance", "handcrafted", "state_budget", "mean_h", "mean_h_in_states", "h_plus_nuisance",
                "mean_h_gps_only", "mean_h_nogps_only"]
TARGETS = {"sex": "female vs male", "young_tagged_vs_not_stated": "young-tagged vs age not stated (descriptive)"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- small helpers

def group_mean(codes: np.ndarray, G: int, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean of the rows of X per code 0..G-1 (code < 0 ignored); also the row count per code."""
    X = np.asarray(X, np.float64)
    X = X[:, None] if X.ndim == 1 else X
    ok = codes >= 0
    c = codes[ok]
    S = sparse.csr_matrix((np.ones(len(c)), (c, np.arange(len(c)))), shape=(G, len(c)))
    fin = np.isfinite(X[ok])
    num = S @ np.where(fin, X[ok], 0.0)
    den = S @ fin.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1), np.nan), np.bincount(c, minlength=G)


def budget(codes: np.ndarray, state: np.ndarray, G: int, K: int) -> np.ndarray:
    """(G, K) share of each state among windows with a state (state >= 0), per code."""
    ok = (codes >= 0) & (state >= 0)
    C = np.bincount(codes[ok] * K + state[ok], minlength=G * K).reshape(G, K).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return C / C.sum(1, keepdims=True)


def median_by(codes: np.ndarray, G: int, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med, n = np.full(G, np.nan), np.zeros(G, int)
    s = pd.Series(v).where(codes >= 0)
    ok = s.notna().to_numpy()
    if ok.any():
        g = s[ok].groupby(codes[ok])
        m, c = g.median(), g.size()
        med[m.index.to_numpy()], n[c.index.to_numpy()] = m.to_numpy(), c.to_numpy()
    return med, n


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))) if sp > 0 else np.nan


def _permuted(labels: np.ndarray, n_perm: int, rng, strata: np.ndarray | None = None) -> np.ndarray:
    if strata is None:
        return np.stack([rng.permutation(labels) for _ in range(n_perm)])
    out = np.tile(labels, (n_perm, 1))
    for s in np.unique(strata):
        ix = np.flatnonzero(strata == s)
        for p in range(n_perm):
            out[p, ix] = labels[rng.permutation(ix)]
    return out


def perm_mean_diff(v: np.ndarray, lab: np.ndarray, n_perm: int, rng, strata=None) -> tuple[float, float]:
    """Difference in means (lab 1 minus lab 0) and its two-sided permutation p (labels shuffled between birds)."""
    v, lab = np.asarray(v, float), np.asarray(lab, int)
    obs = v[lab == 1].mean() - v[lab == 0].mean()
    L = _permuted(lab, n_perm, rng, strata).astype(float)
    d = (L @ v) / L.sum(1) - ((1 - L) @ v) / (1 - L).sum(1)
    return float(obs), float((1 + (np.abs(d) >= abs(obs) - 1e-12).sum()) / (1 + n_perm))


def perm_chi2(cat: np.ndarray, lab: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    """Chi-square of a category x group table and its permutation p (group labels shuffled between birds)."""
    cc = pd.Categorical(cat).codes
    C, G = cc.max() + 1, 2

    def chi(lb):
        t = np.bincount(cc * G + lb, minlength=C * G).reshape(C, G).astype(float)
        e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(np.where(e > 0, (t - e) ** 2 / e, 0))

    obs = chi(lab)
    null = np.array([chi(rng.permutation(lab)) for _ in range(n_perm)])
    return float(obs), float((1 + (null >= obs - 1e-12).sum()) / (1 + n_perm))


def sign_flip(d: np.ndarray, n_perm: int, rng) -> float:
    """Two-sided paired sign-flip p for the mean of d; exact enumeration for n <= EXACT_SIGNFLIP_MAX."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return np.nan
    if n <= EXACT_SIGNFLIP_MAX:
        S = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        return float((np.abs(S @ d) >= abs(d.sum()) - 1e-12).mean())
    S = rng.choice([-1.0, 1.0], size=(n_perm, n))
    return float((1 + (np.abs(S @ d) >= abs(d.sum()) - 1e-12).sum()) / (1 + n_perm))


def sign_test(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    nz = d[d != 0]
    return float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else np.nan


# ---------------------------------------------------------------- pass 1: per-bird aggregation

def _read(path: Path, want: list[str]) -> pd.DataFrame:
    have = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in want if c in have])


def bird_pass(d: Path, info: dict, runs: list[dict], st_by_run: dict) -> dict | None:
    """All aggregates of one bird (see module docstring). ``st_by_run``: run name -> this bird's states rows."""
    have = pq.read_schema(d / "meta.parquet").names
    mag = [c for c in have if "magnet" in c.lower()]
    m = _read(d / "meta.parquet", ["t_start", "sensitivity_k", "acc_samples", "gps_fixes"] + mag)
    n_all = len(m)
    if n_all == 0:
        return None
    f = _read(d / "features.parquet", list(HC_FEATURES) + ["dom_freq_hz", "label_flapping"])
    t = pd.to_datetime(m["t_start"], utc=True).reset_index(drop=True)
    speed = f["speed_mean"].to_numpy(float) if "speed_mean" in f else np.full(n_all, np.nan)
    gps = (m["gps_fixes"].to_numpy() > 0) | np.isfinite(speed)
    cont = m["sensitivity_k"].to_numpy() == 1
    res = {"bird": d.name, "n_all": n_all, "first": t.min(), "last": t.max(),
           "has_magnetometer": bool((m[mag].fillna(0).to_numpy() > 0).any()) if mag else None,
           "n_pre_fledge": 0, "fledge": None}
    keep = np.ones(n_all, bool)
    age = None
    if info.get("young"):
        fl = reference.fledge_from_speed(t, speed, info["hatch_year"])
        res["fledge"] = fl
        age = reference.age_on({"tagged_young": True, "hatch_year": info["hatch_year"]}, t, fl["fledge"])
        keep = (age["age_class"] != "pre_fledge").to_numpy()
        res["n_pre_fledge"] = int((~keep).sum())
        age = age[keep].reset_index(drop=True)
    kept_rows = np.flatnonzero(keep)
    t, speed, gps, cont = t[keep].reset_index(drop=True), speed[keep], gps[keep], cont[keep]
    f = f.iloc[kept_rows].reset_index(drop=True)
    acc = m["acc_samples"].to_numpy()[keep]
    sens = m["sensitivity_k"].to_numpy()[keep]
    n = len(t)
    res.update(n_kept=n, cont_share=float(cont.mean()) if n else np.nan, gps_share=float(gps.mean()) if n else np.nan)
    ym = t.dt.strftime("%Y-%m").to_numpy()
    um, mc = np.unique(ym, return_inverse=True)
    cnt = np.bincount(mc, minlength=len(um))
    good = np.flatnonzero(cnt >= MONTH_MIN_WINDOWS)
    remap = np.full(len(um), -1)
    remap[good] = np.arange(len(good))
    mcode = remap[mc] if n else np.zeros(0, int)
    M = len(good)
    res["months"] = um[good].tolist()
    res["month_n"] = cnt[good]
    res["month_cont"] = group_mean(mcode, M, cont)[0][:, 0]
    res["month_gps"] = group_mean(mcode, M, gps)[0][:, 0]
    res["month_cal"] = np.array([int(s[5:]) for s in um[good]], int)
    full = acc >= FULL_SAMPLES
    dom = f["dom_freq_hz"].to_numpy(float) if "dom_freq_hz" in f else np.full(n, np.nan)
    dom_full = np.where(full, dom, np.nan)
    flap_lab = (f["label_flapping"].to_numpy(float) == 1) if "label_flapping" in f else np.zeros(n, bool)
    res["wb_label"] = [median_by(np.where(flap_lab & sel, 0, -1), 1, dom_full)
                       for sel in (np.ones(n, bool), sens == 0, sens == 1)]
    zeros = np.zeros(n, int)
    if age is not None:
        acls = age["age_class"].to_numpy()
        amon = age["age_months"].to_numpy()
        later = np.isin(acls, ["y2", "y3", "y4plus"])
        cal = t.dt.month.to_numpy()
        in_same = np.isin(cal, SAME_MONTHS)
        period = np.where(in_same & (acls == "juvenile_y1"), 0, np.where(in_same & (acls == "y2"), 1, -1))
        ua, acode = np.unique(amon, return_inverse=True)
    res["runs"] = {}
    for r in runs:
        h = np.load(r["run"] / "embeddings" / f"{d.name}.npz")["h"]
        if len(h) != n_all:
            raise ValueError(f"{d.name}: {n_all} windows but {len(h)} embeddings in {r['name']}")
        h = h[keep].astype(np.float32)
        h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
        K = r["K"]
        st = st_by_run[r["name"]]
        state = np.full(n_all, -1)
        pitch, roll = np.full(n_all, np.nan), np.full(n_all, np.nan)
        if st is not None and len(st):
            rows = st["row"].to_numpy()
            if rows.max() >= n_all or rows.min() < 0:
                raise ValueError(f"{d.name}: states row outside 0..{n_all - 1} in {r['name']}")
            state[rows] = st[r["state_col"]].to_numpy()
            pitch[rows], roll[rows] = st["pitch_rot_deg"].to_numpy(), st["roll_rot_deg"].to_numpy()
        state, pitch, roll = state[keep], pitch[keep], roll[keep]
        hc = f.reindex(columns=list(HC_FEATURES)).assign(pitch_deg=pitch, roll_deg=roll)
        hc["no_gps"] = (~np.isfinite(speed)).astype(float)
        hcX = hc[HC_COLS].to_numpy(float)
        o = {"mean_h": group_mean(zeros, 1, h)[0][0], "budget": budget(zeros, state, 1, K)[0],
             "hc": group_mean(zeros, 1, hcX)[0][0]}
        for name, sel in (("gps", gps), ("nogps", ~gps)):
            mh, c = group_mean(np.where(sel, 0, -1), 1, h)
            o[f"mean_h_{name}"], o[f"n_{name}"] = mh[0], int(c[0])
        o["h_state"] = group_mean(state, K, h)[0]  # (K, D)
        o["m_mean_h"] = group_mean(mcode, M, h)[0]
        o["m_budget"] = budget(mcode, state, M, K)
        o["m_hc"] = group_mean(mcode, M, hcX)[0]
        ms = np.where((mcode >= 0) & (state >= 0), mcode * K + state, -1)
        o["m_h_state"] = group_mean(ms, M * K, h)[0].reshape(M, K, -1)
        o["wb_state"] = [median_by(np.where(sel, state, -1), K, dom_full)
                         for sel in (np.ones(n, bool), sens == 0, sens == 1)]
        o["state_vedba"] = median_by(state, K, np.where(acc >= 40, f["vedba"].to_numpy(float), np.nan))[0] \
            if "vedba" in f else np.full(K, np.nan)
        o["state_speed"] = median_by(state, K, speed)[0]
        if age is not None:
            A = len(ua)
            mh, an = group_mean(acode, A, h)
            cen, cn = group_mean(np.where(later, 0, -1), 1, h)
            cen = cen[0] if cn[0] >= AGE_MIN_WINDOWS else np.full(h.shape[1], np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                dist = 1 - (mh @ cen) / (np.linalg.norm(mh, axis=1) * np.linalg.norm(cen))
            o["age"] = {"age_month": ua, "n": an, "budget": budget(acode, state, A, K), "dist": dist,
                        "centroid_n": int(cn[0])}
            pm, pn = group_mean(period, 2, hc[AGE_HC].to_numpy(float))
            o["same"] = {"n": pn, "budget": budget(period, state, 2, K), "hc": pm}
        res["runs"][r["name"]] = o
    return res


# ---------------------------------------------------------------- classifiers

def _prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))


def _lr(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def _min_class_groups(y, g) -> int:
    _, ix = np.unique(g, return_index=True)
    yb = y[ix]
    return int(min((yb == 0).sum(), (yb == 1).sum()))


def grouped_folds(y: np.ndarray, g: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold by bird; n_splits is capped at the number of birds in the smaller class (>= 2 needed)."""
    k = min(n_splits, _min_class_groups(y, g))
    if k < 2:
        return []
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g))


def choose_c(X, y, g, seed) -> float:
    folds = grouped_folds(y, g, INNER_SPLITS, seed) if len(C_GRID) > 1 else []
    if not folds:
        return C_GRID[len(C_GRID) // 2]
    loss = np.zeros(len(C_GRID))
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        a, b = _prep(X[tr], X[te])
        for i, C in enumerate(C_GRID):
            loss[i] += log_loss(y[te], _lr(C).fit(a, y[tr]).predict_proba(b)[:, 1], labels=[0, 1])
    return C_GRID[int(np.argmin(loss))]


def cv_once(X: np.ndarray, codes: np.ndarray, ybird: np.ndarray, seed: int) -> tuple[float, float]:
    """One repeat of grouped, stratified CV; bird-level (AUC, balanced accuracy)."""
    y = ybird[codes]
    p = np.full(len(y), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*least populated class.*")
        folds = grouped_folds(y, codes, N_SPLITS, seed)
        if not folds:
            return np.nan, np.nan
        for tr, te in folds:
            C = choose_c(X[tr], y[tr], codes[tr], seed + 1)
            a, b = _prep(X[tr], X[te])
            p[te] = _lr(C).fit(a, y[tr]).predict_proba(b)[:, 1]
    nb = len(ybird)
    cnt = np.bincount(codes, minlength=nb)
    pb = np.bincount(codes, weights=p, minlength=nb) / np.maximum(cnt, 1)
    ok = cnt > 0
    return float(roc_auc_score(ybird[ok], pb[ok])), float(balanced_accuracy_score(ybird[ok], pb[ok] >= 0.5))


def _block(spec: dict, jobs: list) -> list:
    out = []
    with threadpool_limits(1):
        for key, kind, i, seed, perm in jobs:
            if kind == "nested_null":  # Freedman-Lane: fitted part on nuisance + permuted residuals
                Xh = spec["h_fit"] + spec["h_res"][perm]
                Xn = spec["X"][:, spec["n_h"]:]
                a = cv_once(np.c_[Xh, Xn], spec["codes"], spec["ybird"], seed)
                b = cv_once(Xn, spec["codes"], spec["ybird"], seed)
                out.append((key, kind, i, a[0] - b[0], a[1] - b[1]))
            else:
                yb = spec["ybird"] if perm is None else spec["ybird"][perm]
                out.append((key, kind, i, *cv_once(spec["X"], spec["codes"], yb, seed)))
    return out


def run_classifiers(specs: dict, n_repeats: int, n_perm: dict, seed: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """specs: key -> {X, codes, ybird, level, run, target, set, n_h?}. Returns (classifiers, nested) tables."""
    rng = np.random.default_rng(seed)
    blocks = []
    rep_seeds = [seed * 7919 + r for r in range(n_repeats)]
    for key, sp in specs.items():
        nb = len(sp["ybird"])
        jobs = [(key, "obs", r, s, None) for r, s in enumerate(rep_seeds)]
        P = n_perm[sp["level"]]
        jobs += [(key, "null", i, 10_000_000 + seed * 100_003 + i, rng.permutation(nb)) for i in range(P)]
        if sp.get("n_h") and sp["level"] == "bird":
            jobs += [(key, "nested_null", i, 20_000_000 + seed * 100_003 + i, rng.permutation(nb)) for i in range(P)]
        size = 25  # CV runs per joblib task
        blocks += [(sp, jobs[j:j + size]) for j in range(0, len(jobs), size)]
    log(f"classifiers: {len(specs)} analyses, {sum(len(b[1]) for b in blocks):,} CV runs in {len(blocks)} blocks")
    res = Parallel(n_jobs=workers)(delayed(_block)(sp, jb) for sp, jb in blocks)
    R = pd.DataFrame([x for r in res for x in r], columns=["key", "kind", "i", "auc", "bacc"])
    rows = []
    for key, sp in specs.items():
        o = R[(R.key == key) & (R.kind == "obs")]
        nl = R[(R.key == key) & (R.kind == "null")]
        ok = nl.dropna()
        row = {"level": sp["level"], "run": sp["run"], "target": sp["target"], "feature_set": sp["set"],
               "n_birds": len(sp["ybird"]), "n_pos_birds": int(sp["ybird"].sum()),
               "n_neg_birds": int((1 - sp["ybird"]).sum()), "n_samples": len(sp["codes"]),
               "n_features": sp["X"].shape[1], "auc_mean": o.auc.mean(), "auc_sd": o.auc.std(),
               "bacc_mean": o.bacc.mean(), "bacc_sd": o.bacc.std(), "n_repeats": len(o), "n_perm": len(ok),
               "null_auc_mean": ok.auc.mean(), "null_auc_q95": ok.auc.quantile(0.95) if len(ok) else np.nan,
               "p_auc": (1 + (ok.auc >= o.auc.mean() - 1e-12).sum()) / (1 + len(ok)) if len(ok) else np.nan,
               "p_bacc": (1 + (ok.bacc >= o.bacc.mean() - 1e-12).sum()) / (1 + len(ok)) if len(ok) else np.nan}
        rows.append(row)
    clf = pd.DataFrame(rows)
    nest = []
    for key, sp in specs.items():
        if sp["set"] != "h_plus_nuisance":
            continue
        base = [k for k, s in specs.items() if s["set"] == "nuisance" and s["level"] == sp["level"]
                and s["run"] == sp["run"] and s["target"] == sp["target"]]
        if not base:
            continue
        a = R[(R.key == key) & (R.kind == "obs")].set_index("i")
        b = R[(R.key == base[0]) & (R.kind == "obs")].set_index("i")
        dA, dB = (a.auc - b.auc).dropna(), (a.bacc - b.bacc).dropna()
        nl = R[(R.key == key) & (R.kind == "nested_null")].dropna()
        nest.append({"level": sp["level"], "run": sp["run"], "target": sp["target"],
                     "delta_auc_mean": dA.mean(), "delta_auc_sd": dA.std(),
                     "repeats_delta_auc_positive": int((dA > 0).sum()), "n_repeats": len(dA),
                     "delta_bacc_mean": dB.mean(), "n_perm": len(nl),
                     "p_delta_auc": (1 + (nl.auc >= dA.mean() - 1e-12).sum()) / (1 + len(nl)) if len(nl) else np.nan,
                     "null": "residuals of h on nuisance permuted between birds (Freedman-Lane)" if len(nl) else
                     "not tested (descriptive)"})
    return clf, pd.DataFrame(nest)


# ---------------------------------------------------------------- assembly

def nuisance_bird(B: pd.DataFrame) -> pd.DataFrame:
    dm = B["deploy_month"].astype(float)
    X = pd.DataFrame({"cohort_2025": (B["cohort"] == "2025").astype(float), "cont_share": B["cont_share"],
                      "gps_share": B["gps_share"], "track_days": B["track_days"],
                      "deploy_month_sin": np.sin(2 * np.pi * dm / 12), "deploy_month_cos": np.cos(2 * np.pi * dm / 12)},
                     index=B.index)
    tm = pd.get_dummies(B["tag_model"].fillna("missing"), prefix="tag_model", dtype=float)
    return pd.concat([X, tm], axis=1) if tm.shape[1] > 1 else X


def feature_tables(P: list[dict], B: pd.DataFrame, run: str, K: int) -> tuple[dict, dict]:
    """Bird-level and bird-month feature sets (DataFrames indexed by bird / (bird, month))."""
    birds = [p["bird"] for p in P]
    o = {p["bird"]: p["runs"][run] for p in P}
    D = len(next(iter(o.values()))["mean_h"])
    hcols = [f"h{i}" for i in range(D)]
    scols = [f"state{k}" for k in range(K)]
    hscols = [f"s{k}_h{i}" for k in range(K) for i in range(D)]

    def frame(key, cols, idx=birds):
        return pd.DataFrame(np.stack([np.ravel(o[b][key]) for b in idx]), index=pd.Index(idx, name="bird"), columns=cols)

    nb = nuisance_bird(B.loc[birds])
    bird = {"nuisance": nb, "handcrafted": frame("hc", HC_COLS), "state_budget": frame("budget", scols),
            "mean_h": frame("mean_h", hcols), "mean_h_in_states": frame("h_state", hscols)}
    bird["h_plus_nuisance"] = pd.concat([bird["mean_h"], nb], axis=1)
    for mode in ("gps", "nogps"):
        ok = [b for b in birds if o[b][f"n_{mode}"] >= MIN_MODE_WINDOWS]
        bird[f"mean_h_{mode}_only"] = frame(f"mean_h_{mode}", hcols, ok)
    # bird-month
    idx = pd.MultiIndex.from_tuples([(p["bird"], ym) for p in P for ym in p["months"]], names=["bird", "month"])

    def mframe(key, cols):
        arrs = [o[p["bird"]][key].reshape(len(p["months"]), -1) for p in P if len(p["months"])]
        return pd.DataFrame(np.concatenate(arrs) if arrs else np.zeros((0, len(cols))), index=idx, columns=cols)

    if len(idx):
        cal = np.concatenate([p["month_cal"] for p in P if len(p["months"])])
        mn = nb.loc[idx.get_level_values("bird")].set_axis(idx, axis=0).assign(
            cont_share=np.concatenate([p["month_cont"] for p in P if len(p["months"])]),
            gps_share=np.concatenate([p["month_gps"] for p in P if len(p["months"])]),
            month_sin=np.sin(2 * np.pi * cal / 12), month_cos=np.cos(2 * np.pi * cal / 12))
        month = {"nuisance": mn, "handcrafted": mframe("m_hc", HC_COLS), "state_budget": mframe("m_budget", scols),
                 "mean_h": mframe("m_mean_h", hcols), "mean_h_in_states": mframe("m_h_state", hscols)}
        month["h_plus_nuisance"] = pd.concat([month["mean_h"], mn], axis=1)
    else:
        month = {}
    return bird, month


def labels(B: pd.DataFrame) -> dict[str, pd.Series]:
    return {"sex": B["sex"].map({"f": 1, "m": 0}).dropna().astype(int),
            "young_tagged_vs_not_stated": B["tagged_young"].notna().astype(int)}


def make_spec(F: pd.DataFrame, y: pd.Series, level: str, run: str, target: str, fset: str, n_h: int = 0) -> dict | None:
    birds_of_rows = F.index.get_level_values("bird")
    F = F[birds_of_rows.isin(y.index)]
    F = F[F.notna().any(axis=1)]
    if len(F) == 0:
        return None
    rb = F.index.get_level_values("bird")
    ub = pd.Index(sorted(set(rb)))
    yb = y.loc[ub].to_numpy(int)
    if min(yb.sum(), (1 - yb).sum()) < 2:
        return None
    sp = {"X": F.to_numpy(np.float64), "codes": ub.get_indexer(rb), "ybird": yb, "level": level, "run": run,
          "target": target, "set": fset, "n_h": n_h, "birds": ub}
    if n_h:  # residuals of h on nuisance (median-imputed, with intercept) for the Freedman-Lane nested null
        X = sp["X"]
        Xh, Xn = X[:, :n_h], X[:, n_h:]
        Xh = np.where(np.isfinite(Xh), Xh, np.nanmedian(Xh, axis=0))
        Xn = np.where(np.isfinite(Xn), Xn, np.nan_to_num(np.nanmedian(Xn, axis=0)))
        Xn1 = np.c_[np.ones(len(Xn)), Xn]
        sp["h_fit"] = Xn1 @ np.linalg.lstsq(Xn1, Xh, rcond=None)[0]
        sp["h_res"] = Xh - sp["h_fit"]
    return sp


# ---------------------------------------------------------------- confounds, wingbeat, age

def confounds(B: pd.DataFrame, n_perm: int, rng) -> pd.DataFrame:
    demos = {"sex": B["sex"].map({"f": 1, "m": 0}),
             "tagged_young": B["tagged_young"].notna().astype(float),
             "migrant": B["migrant"].notna().astype(float)}
    names = {"sex": ("m", "f"), "tagged_young": ("not stated", "young"), "migrant": ("not stated", "migrant")}
    cats = ["cohort", "tag_model", "tag_firmware", "deploy_month"] + (["has_magnetometer"] if B["has_magnetometer"].notna().any() else [])
    conts = ["track_days", "cont_share", "gps_share"]
    rows = []
    for dname, lab in demos.items():
        lab = lab.dropna().astype(int)
        base = {"demographic": dname, "n_birds_with_label": len(lab), "n_group0": int((lab == 0).sum()),
                "n_group1": int((lab == 1).sum()), "group0": names[dname][0], "group1": names[dname][1]}
        if min(base["n_group0"], base["n_group1"]) < 2:
            rows.append({**base, "nuisance": "(all)", "test": "not tested: a group has < 2 birds"})
            continue
        for c in cats:
            v = B.loc[lab.index, c].astype(object).where(B.loc[lab.index, c].notna(), "missing").astype(str)
            tab = pd.crosstab(v, lab)
            desc = "; ".join(f"{k}: {names[dname][0]} {int(r.get(0, 0))} / {names[dname][1]} {int(r.get(1, 0))}"
                             for k, r in tab.iterrows())
            if tab.shape[0] < 2:
                rows.append({**base, "nuisance": c, "kind": "categorical", "table": desc,
                             "test": "not tested: one category"})
            elif tab.shape == (2, 2):
                odds, p = stats.fisher_exact(tab.to_numpy())
                rows.append({**base, "nuisance": c, "kind": "categorical", "table": desc, "test": "Fisher exact",
                             "statistic": odds, "p_value": p})
            else:
                chi, p = perm_chi2(v.to_numpy(), lab.to_numpy(), n_perm, rng)
                rows.append({**base, "nuisance": c, "kind": "categorical", "table": desc,
                             "test": f"permutation chi-square ({n_perm})", "statistic": chi, "p_value": p})
        for c in conts:
            v = B.loc[lab.index, c].astype(float)
            ok = v.notna()
            if ok.sum() < 4:
                continue
            d, p = perm_mean_diff(v[ok].to_numpy(), lab[ok].to_numpy(), n_perm, rng)
            a, b = v[ok][lab[ok] == 1], v[ok][lab[ok] == 0]
            auc = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) if len(a) and len(b) else np.nan
            rows.append({**base, "nuisance": c, "kind": "continuous",
                         "table": f"mean {names[dname][0]} {b.mean():.3g} / {names[dname][1]} {a.mean():.3g}",
                         "test": f"permutation of mean difference ({n_perm})", "statistic": d, "effect_auc": auc,
                         "p_value": p})
    return pd.DataFrame(rows)


def flapping_state(states_dir: Path, P: list[dict], run: str, K: int) -> tuple[int | None, str]:
    rule = f"state with the highest median VeDBA among states with median GPS speed > {FLAP_SPEED_MS} m/s"
    prof = states_dir / "state_profiles.csv"
    if prof.exists():
        p = pd.read_csv(prof, index_col=0)
        if {"vedba_median", "speed_mean_median"} <= set(p.columns):
            c = p[p["speed_mean_median"] > FLAP_SPEED_MS]
            if len(c):
                return int(c["vedba_median"].idxmax()), rule + " (state_profiles.csv, all windows)"
    v = np.nanmedian(np.stack([p["runs"][run]["state_vedba"] for p in P]), axis=0)
    s = np.nanmedian(np.stack([p["runs"][run]["state_speed"] for p in P]), axis=0)
    c = np.flatnonzero(s > FLAP_SPEED_MS)
    if not len(c):
        return None, rule + " (no state qualifies)"
    return int(c[np.nanargmax(v[c])]), rule + " (median over birds of per-bird medians; state_profiles.csv absent)"


def wingbeat(P, B, runs, flap, n_perm, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel_names = ["all full windows", "full windows, sensitivity_k 0", "full windows, sensitivity_k 1"]
    per_bird, rows = [], []
    sets = [("label_flapping==1", None, lambda p, j: p["wb_label"][j])]
    for r in runs:
        if flap[r["name"]][0] is not None:
            s = flap[r["name"]][0]
            sets.append((f"flapping state {s}", r["name"],
                         lambda p, j, rn=r["name"], s=s: (p["runs"][rn]["wb_state"][j][0][s:s + 1],
                                                          p["runs"][rn]["wb_state"][j][1][s:s + 1])))
    for name, run, get in sets:
        for j, sn in enumerate(sel_names):
            vals = pd.Series({p["bird"]: get(p, j)[0][0] for p in P if get(p, j)[1][0] >= WINGBEAT_MIN_WINDOWS})
            ns = pd.Series({p["bird"]: int(get(p, j)[1][0]) for p in P})
            sx = B.loc[vals.index, "sex"]
            ok = sx.isin(["m", "f"])
            v, lab = vals[ok].to_numpy(float), (sx[ok] == "f").astype(int).to_numpy()
            for b in vals.index:
                per_bird.append({"selection": name, "subset": sn, "run": run or "(run-independent)", "bird": b,
                                 "sex": B.loc[b, "sex"], "cohort": B.loc[b, "cohort"], "median_dom_freq_hz": vals[b],
                                 "windows": ns[b]})
            row = {"selection": name, "run": run or "(run-independent)", "subset": sn,
                   "min_windows_per_bird": WINGBEAT_MIN_WINDOWS, "n_f": int(lab.sum()), "n_m": int((1 - lab).sum())}
            if min(row["n_f"], row["n_m"]) >= 2:
                d, p = perm_mean_diff(v, lab, n_perm, rng)
                coh = B.loc[vals[ok].index, "cohort"].to_numpy()
                _, pc = perm_mean_diff(v, lab, n_perm, rng, strata=coh)
                row.update(mean_f=v[lab == 1].mean(), mean_m=v[lab == 0].mean(), median_f=np.median(v[lab == 1]),
                           median_m=np.median(v[lab == 0]), diff_f_minus_m=d, hedges_g=hedges_g(v[lab == 1], v[lab == 0]),
                           p_perm=p, p_perm_within_cohort=pc, n_perm=n_perm)
            else:
                row["note"] = "not tested: fewer than 2 birds of a sex"
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_bird)


def age_tables(P, B, run, K, n_perm, rng) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    months, same, tests = [], [], []
    for p in P:
        o = p["runs"][run]
        if "age" not in o:
            continue
        a = o["age"]
        for i, am in enumerate(a["age_month"]):
            months.append({"bird": p["bird"], "age_month": int(am), "windows": int(a["n"][i]),
                           **{f"state{k}": a["budget"][i, k] for k in range(K)}, "dist_to_own_y2plus": a["dist"][i],
                           "y2plus_windows": a["centroid_n"]})
        s = o["same"]
        if (s["n"] >= AGE_MIN_WINDOWS).all():
            row = {"bird": p["bird"], "windows_y1": int(s["n"][0]), "windows_y2": int(s["n"][1])}
            for k in range(K):
                row[f"state{k}_y1"], row[f"state{k}_y2"] = s["budget"][0, k], s["budget"][1, k]
            for j, c in enumerate(AGE_HC):
                row[f"{c}_y1"], row[f"{c}_y2"] = s["hc"][0, j], s["hc"][1, j]
            same.append(row)
    S = pd.DataFrame(same)
    for v in [f"state{k}" for k in range(K)] + AGE_HC:
        if not len(S):
            break
        d = (S[f"{v}_y2"] - S[f"{v}_y1"]).dropna().to_numpy()
        tests.append({"run": run, "variable": v, "n_birds": len(d), "mean_y1": S[f"{v}_y1"].mean(),
                      "mean_y2": S[f"{v}_y2"].mean(), "mean_diff_y2_minus_y1": d.mean() if len(d) else np.nan,
                      "birds_increase": int((d > 0).sum()), "p_sign_test": sign_test(d),
                      "p_sign_flip": sign_flip(d, n_perm, rng),
                      "sign_flip": "exact" if len(d) <= EXACT_SIGNFLIP_MAX else f"{n_perm} random"})
    return pd.DataFrame(months), S, tests


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_classifiers(clf: pd.DataFrame, level: str, path: Path) -> None:
    plt = _plt()
    sub = clf[clf.level == level]
    panels = sub[["run", "target"]].drop_duplicates().to_numpy().tolist()
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True, squeeze=False)
    for ax, (run, tgt) in zip(axes[0], panels):
        s = sub[(sub.run == run) & (sub.target == tgt)].set_index("feature_set").reindex(FEATURE_SETS).dropna(how="all")
        y = np.arange(len(s))
        ax.errorbar(s["auc_mean"], y, xerr=s["auc_sd"], fmt="o", color="#2a6f97", ms=5, lw=1, label="AUC (mean, sd over repeats)")
        ax.scatter(s["null_auc_q95"], y, marker="|", s=120, color="#888888", label="null 95th percentile")
        ax.axvline(0.5, color="#bbbbbb", lw=0.8)
        ax.set_yticks(y, s.index)
        ax.set_xlim(0.2, 1.0)
        ax.set_title(f"{run}\n{TARGETS[tgt]}", fontsize=8)
        ax.set_xlabel("bird-level ROC AUC")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_wingbeat(wb_birds: pd.DataFrame, run: str, path: Path) -> None:
    plt = _plt()
    sub = wb_birds[wb_birds.subset == "all full windows"]
    sub = sub[(sub.run == run) | (sub.run == "(run-independent)")]
    sels = sub.selection.unique().tolist()
    if not sels:
        return
    fig, axes = plt.subplots(1, len(sels), figsize=(3.2 * len(sels), 3.4), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, sel in zip(axes[0], sels):
        s = sub[sub.selection == sel]
        for i, (sx, col) in enumerate((("f", "tab:red"), ("m", "tab:blue"))):
            v = s[s.sex == sx].median_dom_freq_hz.to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=col, alpha=0.8, label=f"{sx} (n={len(v)})")
            if len(v):
                ax.hlines(np.median(v), i - 0.25, i + 0.25, color="black", lw=1.5)
        ax.set_xticks([0, 1], ["female", "male"])
        ax.set_title(sel, fontsize=8)
        ax.set_ylabel("per-bird median dominant frequency (Hz)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_age(am: pd.DataFrame, K: int, path: Path) -> None:
    if not len(am):
        return
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for b, g in am.groupby("bird"):
        g = g[g.windows >= MONTH_MIN_WINDOWS]
        a1.plot(g.age_month, g.dist_to_own_y2plus, color="#2a6f97", lw=0.8, alpha=0.6)
    a1.set_xlabel("months since assumed hatch date")
    a1.set_ylabel("1 - cos(month mean h, own y2+ centroid)")
    a1.set_title("distance to the bird's own later centroid (one line per bird)", fontsize=8)
    g = am[am.windows >= MONTH_MIN_WINDOWS].groupby("age_month")[[f"state{k}" for k in range(K)]].mean()
    for k in range(K):
        a2.plot(g.index, g[f"state{k}"], lw=1.5, label=f"state {k}")
    a2.set_xlabel("months since assumed hatch date")
    a2.set_ylabel("bird-averaged state share")
    a2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- summary

def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_summary(out: Path, meta: dict, B: pd.DataFrame, conf: pd.DataFrame, clf: pd.DataFrame, nest: pd.DataFrame,
                  wb: pd.DataFrame, age_tests: pd.DataFrame, age_info: dict) -> None:
    L = ["# Demography from the learned latent space", ""]
    a = meta["assumptions"]
    L += ["All tests are at bird level: folds are grouped by bird, permutations shuffle labels between birds, and "
          "bird-month predictions are averaged per bird before scoring. Window-level significance is never used.", "",
          "## Data and assumptions", "",
          f"- Runs: {', '.join(meta['runs'])}. Each number below names its run; 'run-independent' numbers use only "
          "windows, features and reference data.",
          f"- Birds with windows: {len(B)}. Sex: {int((B.sex == 'f').sum())} f, {int((B.sex == 'm').sum())} m, "
          f"{int(B.sex.isna().sum())} missing (of which {meta['reference']['sex_conflicts']} conflicting between "
          "deployments).",
          f"- Tagged young (comment terms {', '.join(a['young_terms'])}): {int(B.tagged_young.notna().sum())} birds; "
          "all others are 'age not stated' (not 'adult'). Hits per term (reference rows): "
          f"{meta['reference']['young_term_hits_rows']}. Young deployments before 1 May (hatch_year rule doubtful): "
          f"{meta['reference']['young_deployed_jan_apr']}.",
          f"- Migrant mentioned: {int(B.migrant.notna().sum())} birds; 'non-migrant' mentions (not counted as "
          f"migrant): {meta['reference']['non_migrant_mentions_rows']} rows.",
          f"- Hatch date: {a['hatch_date']}. Default fledge: {a['fledge_default']}. Data-driven fledge: "
          f"{a['fledge_data_rule']}. Fledge source: {age_info['fledge_sources']}; detected on the bird's first data "
          f"day (upper bound only): {age_info['fledge_first_day']}.",
          f"- Pre-fledge windows removed from every analysis: {age_info['pre_fledge']:,}.",
          f"- Cohort: year of the bird's first window <= 2024 -> '2024', else '2025'. Bird-level samples need >= "
          f"{MIN_BIRD_WINDOWS} kept windows; bird-month samples >= {MONTH_MIN_WINDOWS}.", ""]
    L += ["## (a) Confounds (tested, bird level)", "",
          "| demographic | nuisance | test | summary | p |", "|---|---|---|---|---|"]
    for _, r in conf.iterrows():
        L.append(f"| {r['demographic']} | {r.get('nuisance', '')} | {r.get('test', '')} | {r.get('table', '')} | "
                 f"{_fmt(r.get('p_value'))} |")
    L += ["", "## (c) Classifiers (tested, bird level)", "",
          f"Logistic regression, C from {list(C_GRID)} by inner grouped CV, {meta['n_repeats']} repeats of grouped "
          f"stratified {N_SPLITS}-fold CV. p: labels permuted between birds ({meta['n_perm']} permutations at bird "
          f"level, {meta['n_perm_month']} at bird-month level), one CV repeat per permutation, so p is conservative. "
          "'young_tagged_vs_not_stated' is not young vs adult; it is descriptive, because the nuisance floor is expected "
          "to be high (DESIGN: the 2024 cohort is mostly nestlings). The handcrafted set includes the share of "
          "windows without GPS speed (a nuisance), as in evaluate._feature_matrix. The GPS-only / no-GPS control is "
          "also a continuous-mode vs burst-mode split (RESULTS_stage4_r4-r6 §1).", "",
          "| level | run | target | feature set | birds (pos/neg) | AUC mean (sd) | bal. acc. | null AUC q95 | p AUC | p bal. acc. |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in clf.iterrows():
        L.append(f"| {r.level} | {r.run} | {r.target} | {r.feature_set} | {r.n_birds} ({r.n_pos_birds}/{r.n_neg_birds}) "
                 f"| {_fmt(r.auc_mean)} ({_fmt(r.auc_sd)}) | {_fmt(r.bacc_mean)} | {_fmt(r.null_auc_q95)} | "
                 f"{_fmt(r.p_auc, 4)} | {_fmt(r.p_bacc, 4)} |")
    L += ["", "Does mean h add beyond the nuisance floor? dAUC = AUC(h + nuisance) - AUC(nuisance), same folds.", "",
          "| level | run | target | dAUC mean (sd) | repeats with dAUC > 0 | p | null |", "|---|---|---|---|---|---|---|"]
    for _, r in nest.iterrows():
        L.append(f"| {r.level} | {r.run} | {r.target} | {_fmt(r.delta_auc_mean)} ({_fmt(r.delta_auc_sd)}) | "
                 f"{r.repeats_delta_auc_positive}/{r.n_repeats} | {_fmt(r.p_delta_auc, 4)} | {r.null} |")
    L += ["", "## (d) Positive control: wingbeat frequency, female vs male (tested, bird level)", "",
          f"Per-bird median dom_freq_hz of full windows (acc_samples >= {FULL_SAMPLES}), birds with >= "
          f"{WINGBEAT_MIN_WINDOWS} such windows. Flapping state rule: {age_info['flap_rule']}. Expected: females "
          "(larger) lower. p_perm: sex permuted between birds; p within cohort: permuted within cohort.", "",
          "| selection | run | subset | n f / m | mean f | mean m | f - m (Hz) | Hedges g | p | p within cohort |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in wb.iterrows():
        L.append(f"| {r.selection} | {r.run} | {r.subset} | {r.n_f} / {r.n_m} | {_fmt(r.get('mean_f'))} | "
                 f"{_fmt(r.get('mean_m'))} | {_fmt(r.get('diff_f_minus_m'))} | {_fmt(r.get('hedges_g'), 2)} | "
                 f"{_fmt(r.get('p_perm'), 4)} | {_fmt(r.get('p_perm_within_cohort'), 4)} |")
    L += ["", "## (e) Age development of tagged-young birds", "",
          "Descriptive: age_months_<run>.csv and age_<run>.png (state budget and distance of each month's mean h to "
          f"the bird's own y2+ centroid; a centroid needs >= {AGE_MIN_WINDOWS} y2+ windows). Birds with a y2+ "
          f"centroid: {age_info['centroid_birds']}.", "",
          f"Tested: same calendar months ({', '.join(map(str, SAME_MONTHS))}) in year 1 (juvenile_y1) vs year 2 (y2), "
          f"birds with >= {AGE_MIN_WINDOWS} windows in both, paired across birds (sign test and sign-flip "
          f"permutation). Birds with both periods: {age_info['same_birds']}.", "",
          "| run | variable | birds | mean y1 | mean y2 | mean diff (y2 - y1) | birds increasing | p sign test | p sign flip |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in age_tests.iterrows():
        L.append(f"| {r.run} | {r.variable} | {r.n_birds} | {_fmt(r.mean_y1)} | {_fmt(r.mean_y2)} | "
                 f"{_fmt(r.mean_diff_y2_minus_y1)} | {r.birds_increase} | {_fmt(r.p_sign_test, 4)} | "
                 f"{_fmt(r.p_sign_flip, 4)} ({r.sign_flip}) |")
    L += ["", "No p-value above is corrected for multiple comparisons.", ""]
    (out / "summary.md").write_text("\n".join(L))


# ---------------------------------------------------------------- pipeline

def run(runs: list[Path], states: list[Path], windows: Path, ref_csv: Path, out: Path, n_perm: int = 2000,
        n_perm_month: int = 200, n_repeats: int = 20, seed: int = 0, workers: int = 1, state_col: str = "state_hmm") -> dict:
    t0 = time.time()
    if len(runs) != len(states):
        raise SystemExit("give one --states per --run, in the same order")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"run directory names must differ: {names}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    rng = np.random.default_rng(seed)
    dirs = [d for d in sorted(p for p in windows.iterdir() if p.is_dir())
            if (d / "meta.parquet").exists() and pq.ParquetFile(d / "meta.parquet").metadata.num_rows > 0]
    birds = [d.name for d in dirs]
    ref = reference.load_birds(ref_csv, birds).set_index("bird")
    log(f"{len(birds)} birds with windows; reference: {ref.attrs}")
    rinfo = []
    st_split = {}
    for r, s in zip(runs, states):
        st = pd.read_parquet(s / "states.parquet", columns=["bird", "row", state_col, "pitch_rot_deg", "roll_rot_deg"])
        st["bird"] = st["bird"].astype(str)
        K = int(st[state_col].max()) + 1
        rinfo.append({"name": r.name, "run": r, "states": s, "K": K, "state_col": state_col})
        st_split[r.name] = dict(tuple(st.groupby("bird", sort=False)))
        extra = set(st_split[r.name]) - set(birds)
        if extra:
            raise ValueError(f"{s}: states.parquet has {len(extra)} birds without windows")
        log(f"{r.name}: K={K}, {len(st):,} state rows")
        del st
    info = {b: {"young": pd.notna(ref.loc[b, "tagged_young"]) and pd.notna(ref.loc[b, "hatch_year"]),
                "hatch_year": ref.loc[b, "hatch_year"]} for b in birds}
    log("pass 1: per-bird aggregation")
    res = Parallel(n_jobs=workers)(
        delayed(bird_pass)(d, info[d.name], rinfo, {r["name"]: st_split[r["name"]].pop(d.name, None) for r in rinfo})
        for d in dirs)
    del st_split
    P = [p for p in res if p is not None]
    log(f"pass 1 done ({time.time() - t0:.0f} s)")

    # bird table
    B = ref.loc[[p["bird"] for p in P]].copy()
    B["n_windows"] = [p["n_all"] for p in P]
    B["n_pre_fledge"] = [p["n_pre_fledge"] for p in P]
    B["n_kept"] = [p["n_kept"] for p in P]
    first = pd.Series([p["first"] for p in P], index=B.index)
    last = pd.Series([p["last"] for p in P], index=B.index)
    B["cohort"] = np.where(first.dt.year <= 2024, "2024", "2025")
    B["track_days"] = (last - first).dt.total_seconds() / 86400.0
    B["deploy_month"] = pd.to_datetime(B["deploy_on"], utc=True).dt.month.fillna(first.dt.month)
    B["cont_share"] = [p["cont_share"] for p in P]
    B["gps_share"] = [p["gps_share"] for p in P]
    B["has_magnetometer"] = [p["has_magnetometer"] for p in P]
    for k in ("fledge", "fledge_source", "fledge_is_first_data_day"):
        B[k] = [p["fledge"][k] if p["fledge"] else None for p in P]
    out.mkdir(parents=True, exist_ok=True)
    B.reset_index().to_csv(out / "birds.csv", index=False)

    log("(a) confounds")
    conf = confounds(B, n_perm, rng)
    conf.to_csv(out / "confounds.csv", index=False)

    log("(b) feature tables, (c) classifiers")
    Y = labels(B)
    usable = B.index[B["n_kept"] >= MIN_BIRD_WINDOWS]
    specs = {}
    for r in rinfo:
        bird_sets, month_sets = feature_tables(P, B, r["name"], r["K"])
        pd.concat({k: v for k, v in bird_sets.items()}, axis=1).to_parquet(out / f"features_{r['name']}_bird.parquet")
        if month_sets:
            pd.concat({k: v for k, v in month_sets.items()}, axis=1).to_parquet(out / f"features_{r['name']}_month.parquet")
        D = bird_sets["mean_h"].shape[1]
        for tgt, y in Y.items():
            yb = y[y.index.isin(usable)]
            for level, sets in (("bird", bird_sets), ("bird_month", month_sets)):
                for fs, F in sets.items():
                    sp = make_spec(F, yb, level, r["name"], tgt, fs, n_h=D if fs == "h_plus_nuisance" else 0)
                    if sp is not None:
                        specs[f"{level}|{r['name']}|{tgt}|{fs}"] = sp
    clf, nest = run_classifiers(specs, n_repeats, {"bird": n_perm, "bird_month": n_perm_month}, seed, workers)
    clf.to_csv(out / "classifiers.csv", index=False)
    nest.to_csv(out / "nested.csv", index=False)
    for level in ("bird", "bird_month"):
        fig_classifiers(clf, level, out / f"classifiers_{level}.png")

    log("(d) wingbeat")
    flap = {r["name"]: flapping_state(r["states"], P, r["name"], r["K"]) for r in rinfo}
    wb, wb_birds = wingbeat(P, B, rinfo, flap, n_perm, rng)
    wb.to_csv(out / "wingbeat.csv", index=False)
    wb_birds.to_csv(out / "wingbeat_birds.csv", index=False)
    for r in rinfo:
        fig_wingbeat(wb_birds, r["name"], out / f"wingbeat_{r['name']}.png")

    log("(e) age development")
    tests, centroid_birds, same_birds = [], {}, {}
    for r in rinfo:
        am, S, t = age_tables(P, B, r["name"], r["K"], n_perm, rng)
        am.to_csv(out / f"age_months_{r['name']}.csv", index=False)
        S.to_csv(out / f"age_same_months_{r['name']}.csv", index=False)
        tests += t
        centroid_birds[r["name"]] = int(am.groupby("bird")["y2plus_windows"].first().ge(AGE_MIN_WINDOWS).sum()) if len(am) else 0
        same_birds[r["name"]] = len(S)
        fig_age(am, r["K"], out / f"age_{r['name']}.png")
    age_tests = pd.DataFrame(tests, columns=["run", "variable", "n_birds", "mean_y1", "mean_y2", "mean_diff_y2_minus_y1",
                                             "birds_increase", "p_sign_test", "p_sign_flip", "sign_flip"])
    age_tests.to_csv(out / "age_same_months_tests.csv", index=False)

    young = B[B["tagged_young"].notna()]
    age_info = {"fledge_sources": young["fledge_source"].value_counts().to_dict(),
                "fledge_first_day": int(young["fledge_is_first_data_day"].fillna(False).astype(bool).sum()),
                "pre_fledge": int(B["n_pre_fledge"].sum()), "centroid_birds": centroid_birds, "same_birds": same_birds,
                "flap_rule": "; ".join(f"{k}: state {v[0]} ({v[1]})" for k, v in flap.items())}
    meta = {"runs": names, "states": [str(s) for s in states], "windows": str(windows), "reference_csv": str(ref_csv),
            "state_col": state_col, "n_perm": n_perm, "n_perm_month": n_perm_month, "n_repeats": n_repeats,
            "seed": seed, "workers": workers, "assumptions": reference.assumptions(),
            "reference": {k: v for k, v in ref.attrs.items()}, "age": age_info,
            "thresholds": {"MIN_BIRD_WINDOWS": MIN_BIRD_WINDOWS, "MONTH_MIN_WINDOWS": MONTH_MIN_WINDOWS,
                           "MIN_MODE_WINDOWS": MIN_MODE_WINDOWS, "WINGBEAT_MIN_WINDOWS": WINGBEAT_MIN_WINDOWS,
                           "AGE_MIN_WINDOWS": AGE_MIN_WINDOWS, "FULL_SAMPLES": FULL_SAMPLES,
                           "SAME_MONTHS": list(SAME_MONTHS), "C_GRID": list(C_GRID), "N_SPLITS": N_SPLITS,
                           "INNER_SPLITS": INNER_SPLITS},
            "null_design": "labels permuted between birds; folds recomputed per permutation; one CV repeat per "
                           "permutation vs the mean of n_repeats observed (conservative)",
            "has_magnetometer_column": bool(B["has_magnetometer"].notna().any()),
            "birds": len(B), "birds_usable": int(len(usable)), "runtime_s": None}
    write_summary(out, meta, B, conf, clf, nest, wb, age_tests, age_info)
    meta["runtime_s"] = round(time.time() - t0, 1)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    log(f"done ({meta['runtime_s']} s)")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--states", type=Path, action="append", required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-perm-month", type=int, default=200)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--state-col", default="state_hmm", choices=["state_hmm", "state"])
    a = ap.parse_args(argv)
    run(a.run, a.states, a.windows, a.reference, a.out, a.n_perm, a.n_perm_month, a.n_repeats, a.seed, a.workers,
        a.state_col)


if __name__ == "__main__":
    main()
