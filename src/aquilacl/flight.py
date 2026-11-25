"""Flight as a continuum (window scale) and flight bouts (minute scale) on a frozen SimCLR embedding.

Background: at the 10-s window scale, flight windows do not split into stable sub-states (the sub-state rule fails
at every k), so flight is treated here as a continuum in h (part A), and structure is sought at the scale of
continuous-mode bouts lasting minutes (part B).

Inputs per run (same order on the command line): the run directory (embeddings/<bird>.npz key 'h'), the K=2 states
directory (state 1 = flight; checked from its state_profiles.csv: state 1 must have the higher median speed) and the
K=6 states directory (state_hmm, described only by its own state_profiles.csv, never by a hard-coded name: runs
number states differently). Rows are joined on (bird, row), row being the index within the bird's meta.parquet.

Part A (window scale, flight windows = K=2 ``state_hmm`` == 1 by default, as in budgets.py)
- h is L2-normalised. PCA is fitted on a bird-balanced sample (states.balanced_sample: up to ``per_bird`` windows
  per bird, months equally weighted within a bird). The sign of each PC is fixed so that its strongest correlate
  (largest |Spearman rho| on the pooled sample) is positive.
- Per PC and per subset (all flight windows, scheduled mode k=0 only, continuous mode k=1 only; recording mode
  decides which windows have GPS, so pooled numbers mix two populations): Spearman rho with speed_mean, climb_mean,
  turn_abs_mean, vedba, flap_band_frac, dom_freq_hz (NaN when acc_samples < 40, as in states.profiles),
  local_solar_hour, hours_from_noon, month, sensitivity_k, acc_samples, each on its pairwise-complete windows of a
  bird-balanced sample of that subset, with n; variance explained WITHIN the subset's sample; eta^2 by bird and by
  month; R^2 of the PC on sin/cos of hour and of month (circular-linear). Axis label = the three strongest
  correlates of that subset with their n (the same PC can mean different things in k0 and k1). All descriptive.
- Flight maps (descriptive): PC1/PC2 grid; median of each feature per cell (cells with >= MIN_CELL windows);
  bird-weighted density (each bird's density sums to 1, then averaged over birds); female minus male density and
  month minus all-months density, per recording mode. Speed vs climb map (windows with GPS) coloured by median PC.
- Sex test (TESTED): bird value = mean of the bird's month means (months with >= MIN_BM_WINDOWS windows; birds with
  >= MIN_BIRD_WINDOWS windows in the subset). Strata = deployment year (reference deploy_on) x age group (tagged
  young / not stated); statistic = weighted mean of within-stratum female-minus-male differences, two-sided p with
  sex labels permuted between birds within strata (NaN when no stratum holds both sexes); raw difference, Hedges' g,
  n_f, n_m, birds in informative strata, and the AUC with which the strata alone separate the sexes (>= 0.9: not
  testable in this design). Variants: all windows (primary) and full windows only (acc_samples >= FULL_SAMPLES,
  sensitivity). Holm over all runs x subsets x PCs per variant. p at the floor prints as '<= 1/(n_perm+1)'.
- With two runs: Spearman between the PCs of the two runs on a bird-balanced sample of windows that are flight in
  both (descriptive).

Part B (bout scale)
- Canonical bout (bout_ids; imported by other modules): a maximal run of a bird's sensitivity_k == 1 windows in
  time order whose consecutive k=1 starts are at most BOUT_STEP_S = 30 s apart; windows of another mode in between do
  not split it (see bout_ids for the measured evidence). 30 s is the trough of the k=1 step histogram on windows_v4.
  Segments of windows.py break at 1.5 s acceleration gaps and k_in_segment has holes where short windows were
  dropped, so the time rule, not segment, joins windows. Bouts >= MIN_BOUT_S = 60 s are kept (duration = last start
  - first start + last window length acc_samples / 20 Hz; continuous recording is capped near 900 s, so duration is
  partly a recording setting). bird_coverage.csv gives per-bird counts under this rule and the first run's rule.
- Descriptors per bout: duration, windows, GPS share, flight share (K=2), mean speed, net climb = sum(climb_mean x
  n_speed) (GPS seconds of the window, not 10 s) and its first-half / second-half split, climb rate per minute,
  gross turning rate = sum(turn_abs_mean x n_turn) per minute (includes heading noise; not called circling), net
  circles = |sum turn_net_deg| / 360 and net circles per minute (the "circling" word uses this), share of each K=6
  state (NaN for a bout without K=6 states), state changes per minute, glide ratio = sum(speed_mean x n_speed) /
  sum(-climb_mean x n_speed) over descending
  windows (climb_mean < -GLIDE_MIN_SINK_MS; NaN below GLIDE_MIN_LOSS_M of height lost; ground speed, so wind is included), flapping share
  (label_flapping, known values), mean flap_band_frac, mean and sd of PC1..5 of part A, h spread
  (1 - |mean unit h|), start month and local solar hour.
- Clustering (as stages 5a/5a-2): CLUSTER_COLS standardised (NaN -> median), PCA (95 %), diagonal GMM with k_over
  components on a bird-capped fit sample (states.balanced_sample with ``bout_per_bird``), Ward merge to K =
  1..max_k; bird-bootstrap Jaccard (per-type lists in bout_k_table.csv are in Ward order; the chosen K's list in
  the final numbering is jaccard_chosen_by_type in flight.json) on all bouts (``n_boot`` refits on birds resampled with replacement); presence
  (substates.presence: >= 5 birds with >= 20 bouts of the type, both cohorts). Rule fixed in advance: the largest K
  with every type's mean Jaccard >= 0.75 and presence; otherwise K = 1 (bouts not split). The per-K table is always
  written. Types are numbered by ascending median speed.
- Per type: medians, state shares, transition counts, circular mean start hour, month shares, and bird-level
  female/male and young-tagged/not-stated shares (per-bird share of bouts in the type, birds with >=
  MIN_BIRD_BOUTS bouts; permutation p between birds within strata, TESTED but exploratory, no correction). Words used for a type
  come only from fixed numeric rules (WORD_RULES).

Outputs: --out/<run name>/ (csv, png, json) per run, cross_run_pc_spearman.csv, summary.md, metadata.json. Tables
may be keyed by bird; no position, no exact timestamp is written.

Usage: python -m aquilacl.flight --run R [--run R2] --windows W --states2 S2 [--states2 ...] --states6 S6
       [--states6 ...] --out NEW [--reference CSV] [--state-col state_hmm] [--n-boot 100] [--n-perm 2000]
       [--workers N] [--seed 0] [--umap]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import binned_statistic_2d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import reference
from .demography import hedges_g
from .states import (JACCARD_MIN, SPECTRUM_SAMPLES, balanced_sample, fit_gmm, jaccard_best, joint, onehot,
                     posterior, refit_labels, ward_merge)
from .substates import presence

DEFAULT_REFERENCE = Path("/project/ai4wy-eap/nmille39/aquilacl/source/movebank_2026-11-24/"
                         "LifeTrack Golden Eagle Bighorn Basin WY-reference-data(1).csv")
WINDOW_S = 10.0
FS = 20.0
BOUT_STEP_S = 30.0  # trough of the k=1 step histogram (mode 22-25 s, tail to 29 s, trough 29-31 s; see bout_ids)
LEGACY_BOUT_STEP_S = 25.0  # rule of the first run (any other-mode window split a bout); counts only, for reconciliation
MIN_BOUT_S = 60.0
FULL_SAMPLES = 190  # full-window sensitivity: acc_samples >= this
MIN_BM_WINDOWS = 10  # a bird-month enters a bird's mean with >= this many windows
GLIDE_MIN_LOSS_M = 20.0
GLIDE_MIN_SINK_MS = 0.5  # a window counts as descending (glide ratio) when climb_mean < -0.5 m/s
N_PC = 5
MIN_CELL = 20
MIN_BIRD_WINDOWS = 50
MIN_MONTH_WINDOWS = 50
MIN_BIRD_BOUTS = 5
N_BINS = 50
META = ["t_start", "segment", "k_in_segment", "sensitivity_k", "acc_samples", "gps_fixes", "local_solar_hour"]
FEATS = ["vedba", "flap_band_frac", "dom_freq_hz", "speed_mean", "climb_mean", "turn_abs_mean", "turn_net_deg",
         "path_m", "n_speed", "n_turn", "label_flapping"]
CORR_VARS = ["speed_mean", "climb_mean", "turn_abs_mean", "vedba", "flap_band_frac", "dom_freq_hz",
             "local_solar_hour", "hours_from_noon", "month", "sensitivity_k", "acc_samples"]
MAP_VARS = ["speed_mean", "climb_mean", "turn_abs_mean", "vedba", "flap_band_frac", "dom_freq_hz",
            "local_solar_hour", "sensitivity_k", "acc_samples"]
SUBSETS = {"all": None, "k0": 0, "k1": 1}
CLUSTER_BASE = ["log_duration", "speed_mean", "climb_rate_m_min", "circles_per_min", "flap_share",
                "state_changes_per_min", "h_spread"]
# words for bout types, from fixed numeric thresholds on the type medians (decided before any result was seen)
WORD_RULES = {"climb_rate_m_min": 10.0,   # |median climb rate| above this (m/min): net climbing / descending
              "half_climb_m": 20.0,       # first half above +this and second half below -this: climb then descent
              "state_climb_ms": 0.5,      # a K=6 state with median climb above +this is "climbing", below -this "descending"
              "circles_per_min": 1.0,     # median circles per minute at or above this: circling
              "flap_share": 0.3}          # median flapping share at or above this: flapping-rich


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _read(path: Path, want: list[str]) -> pd.DataFrame:
    have = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in want if c in have])


# ---------------------------------------------------------------- loading

def bout_ids(t_s: np.ndarray, cont: np.ndarray, max_step: float = BOUT_STEP_S) -> np.ndarray:
    """Canonical continuous-mode bout id per window of ONE bird in time order (-1 for windows not in continuous mode).

    Only continuous windows (``cont``) define bouts: a new bout starts at a continuous window whose previous
    CONTINUOUS window starts more than ``max_step`` seconds earlier. Windows of another mode lying between two
    continuous windows <= ``max_step`` apart do not split the bout (they keep id -1). Measured on windows_v4 (30 s):
    20,417 k=0 windows sit inside such runs; 97 % are 40-sample bursts, 3 % have GPS; k=1 steps (10.5-120 s) that
    contain one k=0 window have median 24.8 s, steps without one 24.0 s, and 10-s cadence resumes after them. They
    are not scheduled bursts: per bird a median 3 % start within 3 s of the bird's scheduled clock (chance 2 %),
    against 87 % of k=0 burst starts outside bouts. They are a recording-format interruption, not an end of
    continuous mode. Other modules (budgets, anomaly) import this function; the signature is fixed."""
    t_s, cont = np.asarray(t_s, float), np.asarray(cont, bool)
    out = np.full(len(t_s), -1, np.int64)
    idx = np.flatnonzero(cont)
    if len(idx):
        new = np.r_[True, np.diff(t_s[idx]) > max_step]
        out[idx] = np.cumsum(new) - 1
    return out


def _legacy_bout_ids(t_s: np.ndarray, cont: np.ndarray, max_step: float = LEGACY_BOUT_STEP_S) -> np.ndarray:
    """The first run's rule (any window of another mode splits a bout); only for the reconciliation counts."""
    t_s, cont = np.asarray(t_s, float), np.asarray(cont, bool)
    new = cont.copy()
    if len(t_s) > 1:
        new[1:] &= ~cont[:-1] | (np.diff(t_s) > max_step)
    return np.where(cont, np.cumsum(new) - 1, -1)


def bout_durations(t_s: np.ndarray, ids: np.ndarray, acc_samples: np.ndarray | None = None) -> np.ndarray:
    """Duration per bout id 0..n-1: last start - first start + length of the last window (acc_samples / FS, at
    least 1 s; WINDOW_S when acc_samples is not given). t_s in time order."""
    ids = np.asarray(ids)
    m = ids >= 0
    if not m.any():
        return np.zeros(0)
    b, t = ids[m], np.asarray(t_s, float)[m]
    last_len = (np.maximum(np.asarray(acc_samples, float)[m] / FS, 1.0) if acc_samples is not None
                else np.full(len(t), WINDOW_S))
    n = b.max() + 1
    first = np.full(n, np.inf)
    np.minimum.at(first, b, t)
    lastt = np.full(n, -np.inf)
    np.maximum.at(lastt, b, t)
    end = np.zeros(n)
    is_last = t == lastt[b]
    end[b[is_last]] = t[is_last] + last_len[is_last]
    return end - first


def load_bird(d: Path, run: Path, s2: tuple | None, s6: tuple | None):
    """Flight windows and continuous-mode windows of one bird with features, K=2/K=6 states, bout id and unit h."""
    m = _read(d / "meta.parquet", META)
    n = len(m)
    if n == 0:
        return None
    f = _read(d / "features.parquet", FEATS)
    for c in FEATS:
        if c not in f:
            f[c] = np.nan
    h = np.load(run / "embeddings" / f"{d.name}.npz")["h"]
    if len(h) != n or len(f) != n:
        raise ValueError(f"a bird directory has {n} windows, {len(f)} feature rows, {len(h)} embeddings")
    flight = np.zeros(n, bool)
    st6 = np.full(n, -1, np.int16)
    for s, name in ((s2, "K=2"), (s6, "K=6")):
        if s is not None and len(s[0]) and (s[0].max() >= n or s[0].min() < 0):
            raise ValueError(f"a bird has {name} states rows outside its meta.parquet")
    if s2 is not None:
        flight[s2[0][s2[1] == 1]] = True
    if s6 is not None:
        st6[s6[0]] = s6[1]
    df = pd.concat([m.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    df["row"] = np.arange(n)
    df["flight"], df["state6"] = flight, st6
    df["bout_local"] = -1
    order = np.argsort(df["t_start"].to_numpy(), kind="stable")
    t_s = (df["t_start"] - df["t_start"].min()).dt.total_seconds().to_numpy()
    cont = df["sensitivity_k"].to_numpy() == 1
    ts_o, c_o, acc_o = t_s[order], cont[order], df["acc_samples"].to_numpy(float)[order]
    ids = bout_ids(ts_o, c_o)
    df.loc[order, "bout_local"] = ids
    df["t_s"] = t_s
    dur, leg = bout_durations(ts_o, ids, acc_o), _legacy_bout_ids(ts_o, c_o)
    dleg = bout_durations(ts_o, leg)  # the first run measured duration as last start - first start + 10 s
    cover = {"bird": d.name, "windows": n, "k1_windows": int(cont.sum()), "flight_windows": int(flight.sum()),
             "in_states2": s2 is not None, "in_states6": s6 is not None,
             "bouts_all": int(len(dur)), "bouts_ge_60s": int((dur >= MIN_BOUT_S).sum()),
             "legacy_bouts_all": int(len(dleg)), "legacy_bouts_ge_60s": int((dleg >= MIN_BOUT_S).sum()),
             "k0_inside_bouts": int(_inside(ts_o, c_o, ids))}
    keep = (flight | cont)
    H = np.asarray(h[keep], np.float32)
    H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    out = df[keep].reset_index(drop=True)
    out.insert(0, "bird", d.name)
    return out, H, int(m["t_start"].min().year), n, cover


def _inside(t_s: np.ndarray, cont: np.ndarray, ids: np.ndarray) -> int:
    """Number of non-continuous windows lying between two continuous windows of the same bout (time order)."""
    idx = np.flatnonzero(cont)
    if len(idx) < 2:
        return 0
    pos = np.searchsorted(idx, np.flatnonzero(~cont))
    ok = (pos > 0) & (pos < len(idx))
    pos = pos[ok]
    return int((ids[idx[pos - 1]] == ids[idx[pos]]).sum())


def _by_bird(path: Path, col: str) -> dict:
    st = pd.read_parquet(path / "states.parquet", columns=["bird", "row", col])
    return {b: (g["row"].to_numpy(), g[col].to_numpy()) for b, g in st.groupby("bird")}


def check_flight_state(states2: Path) -> dict:
    """State 1 of the K=2 partition must be the faster one (median GPS speed); fail loudly otherwise."""
    p = pd.read_csv(states2 / "state_profiles.csv", index_col=0)
    sp = p["speed_mean_median"]
    if len(p) != 2 or not sp.loc[1] > sp.loc[0]:
        raise SystemExit(f"{states2}: expected K=2 with state 1 faster than state 0, got speed medians {sp.tolist()}")
    return {"speed_median_state0": float(sp.loc[0]), "speed_median_state1": float(sp.loc[1])}


def state_descriptions(states6: Path) -> pd.DataFrame:
    p = pd.read_csv(states6 / "state_profiles.csv", index_col=0)
    cols = ["vedba_median", "speed_mean_median", "climb_mean_median", "turn_abs_mean_median"]
    return p[[c for c in cols if c in p]].rename(columns=lambda c: c.replace("_median", "").replace("_mean", ""))


def deploy_cohort(birdinfo: pd.DataFrame, first_year: dict) -> tuple[dict, int]:
    """Cohort = year of the bird's earliest reference deploy_on; the first-window year only where deploy_on is
    missing (returned count)."""
    dep = birdinfo["deploy_on"] if "deploy_on" in birdinfo else pd.Series(dtype="datetime64[ns, UTC]")
    out, fallback = {}, 0
    for b, y in first_year.items():
        v = dep.get(b, pd.NaT)
        if pd.notna(v):
            out[b] = str(pd.Timestamp(v).year)
        else:
            out[b] = str(y)
            fallback += 1
    return out, fallback


def load_run(run: Path, windows: Path, states2: Path, states6: Path, state_col: str, workers: int,
             birdinfo: pd.DataFrame):
    s2, s6 = _by_bird(states2, state_col), _by_bird(states6, "state_hmm")
    dirs = sorted(p for p in windows.iterdir() if p.is_dir())
    res = Parallel(n_jobs=workers)(delayed(load_bird)(d, run, s2.get(d.name), s6.get(d.name)) for d in dirs)
    n_empty = sum(r is None for r in res)
    res = [r for r in res if r is not None]
    D = pd.concat([r[0] for r in res], ignore_index=True)
    H = np.concatenate([r[1] for r in res])
    cover = pd.DataFrame([r[4] for r in res])
    first_year = {r[4]["bird"]: r[2] for r in res}
    cohort, fallback = deploy_cohort(birdinfo, first_year)
    D["cohort"] = D["bird"].map(cohort)
    cover["cohort"] = cover["bird"].map(cohort)
    D["month"] = D["t_start"].dt.month
    lsh = D["local_solar_hour"]
    D["hours_from_noon"] = np.abs(lsh % 24.0 - 12.0)
    D.loc[D["acc_samples"] < SPECTRUM_SAMPLES, ["flap_band_frac", "dom_freq_hz"]] = np.nan
    counts = {"windows_total": int(sum(r[3] for r in res)), "birds": int(len(res)),
              "bird_dirs_without_windows": int(n_empty),
              "birds_missing_from_states2": int((~cover["in_states2"]).sum()),
              "birds_missing_from_states6": int((~cover["in_states6"]).sum()),
              "birds_without_flight_windows": int((cover["flight_windows"] == 0).sum()),
              "birds_with_k1_windows": int((cover["k1_windows"] > 0).sum()),
              "cohort_from_first_window_year": int(fallback),
              "cohort_birds": {k: int(v) for k, v in cover["cohort"].value_counts().sort_index().items()}}
    log(f"{run.name}: {counts['birds']} birds with windows ({n_empty} bird directories without windows); missing from "
        f"K=2 states {counts['birds_missing_from_states2']}, from K=6 states {counts['birds_missing_from_states6']}; "
        f"{counts['birds_without_flight_windows']} birds without K=2 flight windows; cohort from first-window year "
        f"(no deploy_on) for {fallback} birds")
    return D, H, counts, cover


# ---------------------------------------------------------------- statistics helpers

def eta2(z: np.ndarray, groups) -> float:
    """Share of the variance of z between groups (NaN rows ignored)."""
    z = np.asarray(z, float)
    g = pd.Series(np.asarray(groups)).to_numpy()
    ok = np.isfinite(z) & pd.notna(g)
    z, codes = z[ok], pd.factorize(g[ok])[0]
    if len(z) < 2:
        return np.nan
    n = np.bincount(codes)
    mu = np.bincount(codes, weights=z) / n
    tot = ((z - z.mean()) ** 2).sum()
    return float((n * (mu - z.mean()) ** 2).sum() / tot) if tot > 0 else np.nan


def circ_r2(z: np.ndarray, x: np.ndarray, period: float) -> float:
    """R^2 of z regressed on sin and cos of x (period ``period``)."""
    z, x = np.asarray(z, float), np.asarray(x, float)
    ok = np.isfinite(z) & np.isfinite(x)
    if ok.sum() < 3:
        return np.nan
    a = 2 * np.pi * x[ok] / period
    A = np.column_stack([np.ones(ok.sum()), np.sin(a), np.cos(a)])
    beta, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    r = z[ok] - A @ beta
    tot = ((z[ok] - z[ok].mean()) ** 2).sum()
    return float(1 - (r ** 2).sum() / tot) if tot > 0 else np.nan


def spearman(x: np.ndarray, z: np.ndarray) -> tuple[float, int]:
    x, z = np.asarray(x, float), np.asarray(z, float)
    ok = np.isfinite(x) & np.isfinite(z)
    n = int(ok.sum())
    if n < 3 or np.ptp(x[ok]) == 0 or np.ptp(z[ok]) == 0:
        return np.nan, n
    return float(stats.spearmanr(x[ok], z[ok])[0]), n


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok])]
    adj = np.minimum(1, np.maximum.accumulate(p[o] * (len(o) - np.arange(len(o)))))
    out[o] = adj
    return out


def _subset_mask(F: pd.DataFrame, sub: str) -> np.ndarray:
    k = SUBSETS[sub]
    return np.ones(len(F), bool) if k is None else (F["sensitivity_k"].to_numpy() == k)


# ---------------------------------------------------------------- part A

def fit_axes(F: pd.DataFrame, HF: np.ndarray, per_bird: int, rng, seed: int):
    """PCA on a bird-balanced sample; scores of all flight windows with signs fixed by the strongest correlate."""
    fit = balanced_sample(F, np.ones(len(F), bool), per_bird, rng)
    pca = PCA(n_components=min(N_PC, HF.shape[1], len(fit)), random_state=seed).fit(HF[fit])
    Z = pca.transform(HF).astype(np.float32)
    signs = []
    for i in range(Z.shape[1]):
        rhos = [spearman(F[v].to_numpy()[fit], Z[fit, i])[0] for v in CORR_VARS]
        j = int(np.nanargmax(np.abs(rhos))) if np.isfinite(rhos).any() else 0
        s = -1.0 if np.isfinite(rhos[j]) and rhos[j] < 0 else 1.0
        Z[:, i] *= s
        signs.append(s)
    pca.components_[: len(signs)] *= np.asarray(signs)[:, None]
    return pca, Z, fit


def axis_tables(F: pd.DataFrame, Z: np.ndarray, HF: np.ndarray, pca: PCA, per_bird: int, rng
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlates (with n) and summaries of each PC per subset. variance_explained is the PC's share of the total
    variance of unit h WITHIN the subset's bird-balanced sample (the pooled fit value is variance_explained_fit)."""
    corr, summ = [], []
    for sub in SUBSETS:
        mask = _subset_mask(F, sub)
        if mask.sum() < 3:
            continue
        smp = balanced_sample(F, mask, per_bird, rng)
        total_var = float(np.var(HF[smp].astype(np.float64), axis=0).sum())
        for i in range(Z.shape[1]):
            z = Z[smp, i]
            rows = []
            for v in CORR_VARS:
                r, n = spearman(F[v].to_numpy()[smp], z)
                rows.append({"subset": sub, "pc": i + 1, "variable": v, "spearman": r, "n": n})
            corr += rows
            top = sorted((r for r in rows if np.isfinite(r["spearman"]) and r["n"] >= 30),
                         key=lambda r: -abs(r["spearman"]))[:3]
            summ.append({"subset": sub, "pc": i + 1,
                         "variance_explained": float(np.var(z.astype(np.float64)) / total_var) if total_var > 0 else np.nan,
                         "variance_explained_fit": float(pca.explained_variance_ratio_[i]),
                         "windows_sample": int(len(smp)), "birds": int(F["bird"].iloc[smp].nunique()),
                         "eta2_bird": eta2(z, F["bird"].to_numpy()[smp]),
                         "eta2_month": eta2(z, F["month"].to_numpy()[smp]),
                         "circ_r2_hour": circ_r2(z, F["local_solar_hour"].to_numpy()[smp], 24.0),
                         "circ_r2_month": circ_r2(z, F["month"].to_numpy()[smp] - 0.5, 12.0),
                         "label": ", ".join(f"{r['variable']} {r['spearman']:+.2f} (n={r['n']:,})" for r in top)})
    return pd.DataFrame(corr), pd.DataFrame(summ)


def strat_perm_diff(v: np.ndarray, lab: np.ndarray, strata, n_perm: int, rng) -> dict:
    """Group 1 minus group 0 within strata: weighted mean of the per-stratum differences of means (weights
    n1 n0 / (n1 + n0)) over the strata that hold both groups, with a two-sided permutation p (labels permuted between
    birds within strata). Birds in strata holding one group only do not enter. NaN when no stratum holds both."""
    v, lab = np.asarray(v, float), np.asarray(lab, int)
    codes = pd.factorize(pd.Series(np.asarray(strata, object)).astype(str))[0]
    inf = [c for c in np.unique(codes) if 0 < lab[codes == c].sum() < (codes == c).sum()]
    res = {"diff_strat": np.nan, "p_perm": np.nan, "n_informative": 0, "strata_informative": len(inf)}
    if not inf:
        return res
    keep = np.isin(codes, inf)
    v, lab, codes = v[keep], lab[keep], codes[keep]
    res["n_informative"] = int(keep.sum())
    groups = [np.flatnonzero(codes == c) for c in inf]

    def stat(L):
        tot, wsum = 0.0, 0.0
        for ix in groups:
            L1 = L[:, ix].astype(float)
            n1 = L1.sum(1)
            n0 = len(ix) - n1
            d = L1 @ v[ix] / n1 - (1 - L1) @ v[ix] / n0
            w = n1 * n0 / len(ix)
            tot, wsum = tot + w * d, wsum + w
        return tot / wsum

    obs = float(stat(lab[None, :])[0])
    L = np.tile(lab, (n_perm, 1))
    for ix in groups:
        L[:, ix] = lab[ix][np.argsort(rng.random((n_perm, len(ix))), axis=1)]
    null = stat(L)
    res["diff_strat"] = obs
    res["p_perm"] = float((1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (1 + n_perm))
    return res


def auc(score: np.ndarray, lab: np.ndarray) -> float:
    """Area under the ROC curve of ``score`` for lab == 1 (ties count half)."""
    score, lab = np.asarray(score, float), np.asarray(lab, int)
    n1, n0 = int(lab.sum()), int((1 - lab).sum())
    if not n1 or not n0:
        return np.nan
    r = stats.rankdata(score)
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def nuisance_auc(lab: np.ndarray, strata) -> float:
    """How well the strata alone separate the groups: AUC of each bird's stratum share of group 1 (in sample)."""
    st = pd.Series(np.asarray(strata, object)).astype(str)
    share = pd.Series(np.asarray(lab, float)).groupby(st.to_numpy()).transform("mean").to_numpy()
    a = auc(share, lab)
    return float(max(a, 1 - a)) if np.isfinite(a) else np.nan


def bird_month_means(F: pd.DataFrame, Z: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Per bird: mean over its bird-months (>= MIN_BM_WINDOWS windows each) of the month-mean PC scores; birds with
    >= MIN_BIRD_WINDOWS windows in the mask and at least one such month."""
    pcs = [f"pc{i + 1}" for i in range(Z.shape[1])]
    df = pd.DataFrame(Z[mask], columns=pcs)
    df["bird"] = F["bird"].to_numpy()[mask]
    df["ym"] = (F["t_start"].dt.year * 100 + F["t_start"].dt.month).to_numpy()[mask]
    g = df.groupby(["bird", "ym"])
    cell = g[pcs].mean()
    cell["n"] = g.size()
    tot = df.groupby("bird").size()
    cell = cell[cell["n"] >= MIN_BM_WINDOWS]
    bm = cell.groupby(level="bird")[pcs].mean()
    bm["months"] = cell.groupby(level="bird").size()
    bm["windows"] = tot.reindex(bm.index)
    return bm[bm["windows"] >= MIN_BIRD_WINDOWS]


SEX_VARIANTS = {"all": None, "full": FULL_SAMPLES}  # all windows; full windows only (acc_samples >= FULL_SAMPLES)


def sex_tests(F: pd.DataFrame, Z: np.ndarray, birdinfo: pd.DataFrame, n_perm: int, rng,
              labels: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Female minus male along each PC, per window variant and recording-mode subset. Bird value = mean of its
    bird-month means. Strata = deployment year x age group (tagged young / not stated); labels are permuted between
    birds within strata (strat_perm_diff). Also the raw difference, Hedges g of the bird values and the AUC with
    which the strata alone separate the sexes (>= 0.9: not testable in this design). ``labels``: (subset, pc) ->
    the axis label of that subset (its own strongest correlates)."""
    sex = birdinfo["sex"]
    young = birdinfo["tagged_young"].map(lambda x: "young" if x == True else "not stated")  # noqa: E712
    cohort = F.groupby("bird")["cohort"].first()
    means, rows = [], []
    acc = F["acc_samples"].to_numpy(float)
    for var, thr in SEX_VARIANTS.items():
        for sub in SUBSETS:
            mask = _subset_mask(F, sub) & (acc >= thr if thr is not None else True)
            bm = bird_month_means(F, Z, mask)
            bm["variant"], bm["subset"] = var, sub
            bm["sex"] = sex.reindex(bm.index).to_numpy()
            bm["stratum"] = (cohort.reindex(bm.index).astype(str) + "|" + young.reindex(bm.index).astype(str)).to_numpy()
            means.append(bm.rename_axis("bird").reset_index())
            known = bm[bm["sex"].isin(["f", "m"])]
            lab = (known["sex"] == "f").to_numpy().astype(int)
            strata = known["stratum"].to_numpy()
            for i in range(Z.shape[1]):
                v = known[f"pc{i + 1}"].to_numpy(float)
                row = {"variant": var, "subset": sub, "pc": i + 1, "n_f": int(lab.sum()), "n_m": int((1 - lab).sum()),
                       "axis_label": (labels or {}).get((sub, i + 1), "")}
                if row["n_f"] >= 3 and row["n_m"] >= 3:
                    r = strat_perm_diff(v, lab, strata, n_perm, rng)
                    row.update(diff_f_minus_m=float(v[lab == 1].mean() - v[lab == 0].mean()),
                               hedges_g=hedges_g(v[lab == 1], v[lab == 0]), nuisance_auc=nuisance_auc(lab, strata), **r)
                    row["testable"] = bool(row["nuisance_auc"] < 0.9 and np.isfinite(r["p_perm"]))
                else:
                    row.update(testable=False, n_informative=0)
                rows.append(row)
    return pd.DataFrame(rows), pd.concat(means, ignore_index=True) if means else pd.DataFrame()


def holm_family(tests: list[pd.DataFrame]) -> None:
    """Holm in place over all runs per variant: the primary family is variant 'all' (every run x subset x PC
    row); variant 'full' is a sensitivity family of its own. Untestable rows (NaN p or nuisance AUC >= 0.9) are
    outside the family."""
    allt = pd.concat([t.assign(_run=i, _row=np.arange(len(t))) for i, t in enumerate(tests)], ignore_index=True)
    if "p_perm" not in allt:
        allt["p_perm"] = np.nan
    allt["p_holm"] = np.nan
    for var in allt["variant"].unique():
        sel = (allt["variant"] == var) & allt["testable"].fillna(False).astype(bool)
        allt.loc[sel, "p_holm"] = holm(allt.loc[sel, "p_perm"].to_numpy())
        allt.loc[allt["variant"] == var, "holm_family_size"] = int(sel.sum())
    for i, t in enumerate(tests):
        a = allt[allt["_run"] == i].sort_values("_row")
        t["p_holm"] = a["p_holm"].to_numpy()
        t["holm_family_size"] = a["holm_family_size"].to_numpy()


def _grid(F: pd.DataFrame, Z: np.ndarray, fit: np.ndarray):
    lo = np.nanpercentile(Z[fit, :2], 0.5, axis=0)
    hi = np.nanpercentile(Z[fit, :2], 99.5, axis=0)
    return np.linspace(lo[0], hi[0], N_BINS + 1), np.linspace(lo[1], hi[1], N_BINS + 1)


def bird_density(x: np.ndarray, y: np.ndarray, groups: np.ndarray, ex: np.ndarray, ey: np.ndarray, min_n: int):
    """Per-group 2-D density (each group's cells sum to 1 over in-grid windows); groups with < min_n windows dropped."""
    ix, iy = np.digitize(x, ex) - 1, np.digitize(y, ey) - 1
    nb = len(ex) - 1
    ok = (ix >= 0) & (ix < nb) & (iy >= 0) & (iy < nb) & pd.notna(groups)
    codes, uniq = pd.factorize(np.asarray(groups)[ok])
    C = np.bincount(codes * nb * nb + ix[ok] * nb + iy[ok], minlength=len(uniq) * nb * nb).reshape(len(uniq), nb, nb)
    n = C.sum((1, 2))
    keep = n >= min_n
    return C[keep] / n[keep, None, None], np.asarray(uniq)[keep]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _diverging(ax, D, ex, ey, title, plt):
    v = np.nanmax(np.abs(D)) if np.isfinite(D).any() and np.nanmax(np.abs(D)) > 0 else 1.0
    im = ax.pcolormesh(ex, ey, D.T, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)


def flight_maps(F: pd.DataFrame, Z: np.ndarray, fit: np.ndarray, sex: pd.Series, out: Path) -> dict:
    """Feature-median maps, bird-weighted density, sex and month density differences (all descriptive)."""
    plt = _plt()
    ex, ey = _grid(F, Z, fit)
    x, y = Z[:, 0], Z[:, 1]
    info = {"grid_pc1": [float(ex[0]), float(ex[-1])], "grid_pc2": [float(ey[0]), float(ey[-1])], "bins": N_BINS}
    birds = F["bird"].to_numpy()
    for sub in SUBSETS:
        m = _subset_mask(F, sub)
        if m.sum() < MIN_CELL:
            continue
        fig, axs = plt.subplots(2, 5, figsize=(20, 7.5))
        dens, _ = bird_density(x[m], y[m], birds[m], ex, ey, MIN_BIRD_WINDOWS)
        if len(dens):
            im = axs[0, 0].pcolormesh(ex, ey, np.log10(dens.mean(0).T + 1e-6), cmap="viridis")
            plt.colorbar(im, ax=axs[0, 0], fraction=0.046)
        axs[0, 0].set_title(f"bird-weighted density (log10), {len(dens)} birds", fontsize=8)
        cnt = binned_statistic_2d(x[m], y[m], None, "count", bins=[ex, ey]).statistic
        for ax, v in zip(axs.ravel()[1:], MAP_VARS):
            val = F[v].to_numpy(float)[m]
            ok = np.isfinite(val)
            if ok.sum() == 0:
                ax.set_axis_off()
                continue
            stat = "mean" if v == "sensitivity_k" else "median"
            S = binned_statistic_2d(x[m][ok], y[m][ok], val[ok], stat, bins=[ex, ey]).statistic
            n_ok = binned_statistic_2d(x[m][ok], y[m][ok], None, "count", bins=[ex, ey]).statistic
            S[(n_ok < MIN_CELL) | (cnt < MIN_CELL)] = np.nan
            im = ax.pcolormesh(ex, ey, S.T, cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{stat} {v} (cells >= {MIN_CELL} windows)", fontsize=8)
        for ax in axs.ravel():
            ax.set_xlabel("PC1", fontsize=7)
            ax.set_ylabel("PC2", fontsize=7)
        fig.suptitle(f"Flight map in PC1/PC2, {sub} flight windows (n={int(m.sum()):,})")
        fig.tight_layout()
        fig.savefig(out / f"flightmap_pc_{sub}.png", dpi=110)
        plt.close(fig)

    # sex difference per subset
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, sub in zip(axs, SUBSETS):
        m = _subset_mask(F, sub)
        dens, ub = bird_density(x[m], y[m], birds[m], ex, ey, MIN_BIRD_WINDOWS)
        sx = sex.reindex(ub).to_numpy()
        nf, nm = int((sx == "f").sum()), int((sx == "m").sum())
        if nf and nm:
            _diverging(ax, dens[sx == "f"].mean(0) - dens[sx == "m"].mean(0), ex, ey,
                       f"{sub}: female minus male density ({nf} f, {nm} m birds)", plt)
        else:
            ax.set_title(f"{sub}: not enough birds of both sexes", fontsize=8)
        info[f"sex_birds_{sub}"] = {"f": nf, "m": nm}
    fig.tight_layout()
    fig.savefig(out / "density_sex.png", dpi=110)
    plt.close(fig)

    # month difference per recording mode
    for sub in ("k0", "k1"):
        m = _subset_mask(F, sub)
        all_d, _ = bird_density(x[m], y[m], birds[m], ex, ey, MIN_BIRD_WINDOWS)
        if not len(all_d):
            continue
        base = all_d.mean(0)
        bm = pd.Series(birds[m]) + "|" + F["month"].to_numpy()[m].astype(str)
        dens, keys = bird_density(x[m], y[m], bm.to_numpy(), ex, ey, MIN_MONTH_WINDOWS)
        months = np.array([int(k.rsplit("|", 1)[1]) for k in keys], int)
        fig, axs = plt.subplots(3, 4, figsize=(17, 11))
        nb = {}
        for mo, ax in zip(range(1, 13), axs.ravel()):
            sel = months == mo
            nb[mo] = int(sel.sum())
            if sel.sum() >= 3:
                _diverging(ax, dens[sel].mean(0) - base, ex, ey, f"month {mo} minus all ({sel.sum()} birds)", plt)
            else:
                ax.set_title(f"month {mo}: {sel.sum()} birds (< 3, not shown)", fontsize=8)
        fig.suptitle(f"Bird-weighted density difference by month, {sub} flight windows (PC1 x, PC2 y)")
        fig.tight_layout()
        fig.savefig(out / f"density_month_{sub}.png", dpi=110)
        plt.close(fig)
        info[f"month_birds_{sub}"] = nb

    # speed vs climb, coloured by median PC
    sp, cl = F["speed_mean"].to_numpy(float), F["climb_mean"].to_numpy(float)
    ok = np.isfinite(sp) & np.isfinite(cl)
    if ok.sum() >= MIN_CELL:
        es, ec = np.linspace(0, 30, N_BINS + 1), np.linspace(-5, 5, N_BINS + 1)
        fig, axs = plt.subplots(1, 4, figsize=(19, 4.5))
        dens, ub = bird_density(sp[ok], cl[ok], birds[ok], es, ec, MIN_BIRD_WINDOWS)
        if len(dens):
            im = axs[0].pcolormesh(es, ec, np.log10(dens.mean(0).T + 1e-6), cmap="viridis")
            plt.colorbar(im, ax=axs[0], fraction=0.046)
        axs[0].set_title(f"bird-weighted density (log10), {len(dens)} birds", fontsize=8)
        cnt = binned_statistic_2d(sp[ok], cl[ok], None, "count", bins=[es, ec]).statistic
        for i, ax in enumerate(axs[1:]):
            S = binned_statistic_2d(sp[ok], cl[ok], Z[ok, i], "median", bins=[es, ec]).statistic
            S[cnt < MIN_CELL] = np.nan
            im = ax.pcolormesh(es, ec, S.T, cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"median PC{i + 1}", fontsize=8)
        for ax in axs:
            ax.set_xlabel("GPS speed (m/s)", fontsize=7)
            ax.set_ylabel("climb (m/s)", fontsize=7)
        fig.suptitle(f"Flight windows with GPS (n={int(ok.sum()):,})")
        fig.tight_layout()
        fig.savefig(out / "flightmap_speed_climb.png", dpi=110)
        plt.close(fig)
    return info


def umap_figure(F: pd.DataFrame, HF: np.ndarray, per_bird: int, rng, seed: int, out: Path) -> str:
    try:
        import umap
    except ImportError:
        return "umap-learn not installed; no UMAP figure"
    smp = balanced_sample(F, np.ones(len(F), bool), per_bird, rng)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(HF[smp])
    plt = _plt()
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.5))
    for ax, v in zip(axs, ["speed_mean", "climb_mean", "vedba", "sensitivity_k"]):
        c = F[v].to_numpy(float)[smp]
        ok = np.isfinite(c)
        ax.scatter(xy[~ok, 0], xy[~ok, 1], s=1, c="0.85", rasterized=True)
        lo, hi = (np.nanpercentile(c, [2, 98]) if ok.any() else (0, 1))
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], s=1, c=c[ok], cmap="viridis", vmin=lo, vmax=hi, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"{v} (grey: unknown)", fontsize=8)
    fig.suptitle(f"UMAP of unit h, bird-balanced sample of {len(smp):,} flight windows (figure only)")
    fig.tight_layout()
    fig.savefig(out / "umap_flight.png", dpi=110)
    plt.close(fig)
    return f"UMAP on {len(smp)} windows"


# ---------------------------------------------------------------- part B

def bout_table(B: pd.DataFrame, Zb: np.ndarray, Hb: np.ndarray, K6: int) -> pd.DataFrame:
    """Descriptors of every bout (rows of B carry ``bout``; Zb PC scores and Hb unit h in the same order)."""
    B = B.assign(_i=np.arange(len(B))).sort_values(["bout", "t_s"], kind="stable")
    i = B["_i"].to_numpy()
    Zb, Hb = Zb[i], Hb[i]
    g = B.groupby("bout", sort=True)
    wlen = (np.maximum(B["acc_samples"].to_numpy(float) / FS, 1.0) if "acc_samples" in B
            else np.full(len(B), WINDOW_S))
    t_end = pd.Series(B["t_s"].to_numpy() + wlen, index=B.index)
    last = B["t_s"] == g["t_s"].transform("max")
    dur = t_end.where(last).groupby(B["bout"]).max() - g["t_s"].min()
    t0 = g["t_s"].transform("min")
    mid = t0 + dur.reindex(B["bout"]).to_numpy() / 2
    # GPS seconds per window: n_speed fixes at 1 Hz (a 64-sample tail with 3 fixes covers 3 s, not 10 s)
    gps_s = B["n_speed"].astype(float).where(B["n_speed"] > 0)
    climb_m = B["climb_mean"] * gps_s
    first = (B["t_s"] + wlen / 2) < mid
    desc = (B["climb_mean"] < -GLIDE_MIN_SINK_MS) & B["speed_mean"].notna()
    turn_abs = B["turn_abs_mean"] * B["n_turn"]
    T = pd.DataFrame({
        "bird": g["bird"].first(), "cohort": g["cohort"].first(), "t_start": g["t_start"].min(),
        "windows": g.size(), "duration_s": dur,
        "gps_share": g["n_speed"].apply(lambda s: float((s > 0).mean())),
        "flight_share": g["flight"].mean(), "speed_mean": g["speed_mean"].mean(),
        "net_climb_m": climb_m.groupby(B["bout"]).sum(min_count=1),
        "climb_first_half_m": climb_m.where(first).groupby(B["bout"]).sum(min_count=1),
        "climb_second_half_m": climb_m.where(~first).groupby(B["bout"]).sum(min_count=1),
        "turn_abs_total_deg": turn_abs.groupby(B["bout"]).sum(min_count=1),
        "turn_net_deg": B["turn_net_deg"].groupby(B["bout"]).sum(min_count=1),
        "flap_share": g["label_flapping"].mean(), "flap_band_frac_mean": g["flap_band_frac"].mean(),
        "start_hour": g["local_solar_hour"].first(), "month": g["month"].first(),
    })
    dist = (B["speed_mean"] * gps_s).where(desc).groupby(B["bout"]).sum(min_count=1)
    loss = (-climb_m).where(desc).groupby(B["bout"]).sum(min_count=1)
    T["height_lost_m"] = loss
    T["glide_ratio"] = (dist / loss).where(loss >= GLIDE_MIN_LOSS_M)
    minutes = T["duration_s"] / 60.0
    T["log_duration"] = np.log10(T["duration_s"])
    T["climb_rate_m_min"] = T["net_climb_m"] / minutes
    # gross absolute turning includes GPS heading noise (5.6 deg/s of jitter on straight flight reads as 0.84
    # "circles"/min), so it is reported only as a turning rate; circles are net turning (direction changes cancel)
    T["turn_gross_deg_per_min"] = T["turn_abs_total_deg"] / minutes
    T["circles_net"] = T["turn_net_deg"].abs() / 360.0
    T["circles_per_min"] = T["circles_net"] / minutes
    st = B["state6"].to_numpy()
    has = st >= 0  # windows without a K=6 state do not count in the shares
    sh = pd.crosstab(B["bout"].to_numpy()[has], st[has], normalize="index").reindex(index=T.index, columns=range(K6))
    for s in range(K6):
        T[f"share_state_{s}"] = sh[s].reindex(T.index).to_numpy()  # NaN for a bout without any K=6 state
    b = B["bout"].to_numpy()
    same = (b[1:] == b[:-1]) & (st[1:] >= 0) & (st[:-1] >= 0)
    change = same & (st[1:] != st[:-1])
    T["state_changes"] = pd.Series(change.astype(int), index=b[1:]).groupby(level=0).sum().reindex(T.index).fillna(0)
    T["state_changes_per_min"] = T["state_changes"] / minutes
    for k in range(Zb.shape[1]):
        zs = pd.Series(Zb[:, k]).groupby(b)
        T[f"pc{k + 1}_mean"], T[f"pc{k + 1}_sd"] = zs.mean().to_numpy(), zs.std(ddof=0).to_numpy()
    codes = pd.factorize(b, sort=True)[0]
    Hm = np.zeros((codes.max() + 1, Hb.shape[1]))
    np.add.at(Hm, codes, Hb)
    Hm /= np.bincount(codes)[:, None]
    T["h_spread"] = 1.0 - np.linalg.norm(Hm, axis=1)
    return T


def transition_counts(B: pd.DataFrame, bout_type: pd.Series, K6: int) -> pd.DataFrame:
    """Counts of K=6 state changes (from != to) of consecutive windows within bouts, per bout type."""
    B = B.sort_values(["bout", "t_s"], kind="stable")
    b, st = B["bout"].to_numpy(), B["state6"].to_numpy()
    ok = (b[1:] == b[:-1]) & (st[1:] >= 0) & (st[:-1] >= 0) & (st[1:] != st[:-1])
    t = bout_type.reindex(b[1:][ok]).to_numpy()
    df = pd.DataFrame({"type": t, "from": st[:-1][ok], "to": st[1:][ok]})
    return df.groupby(["type", "from", "to"]).size().rename("count").reset_index()


def cluster_bouts(T: pd.DataFrame, cols: list[str], per_bird: int, k_over: int, max_k: int, n_boot: int, seed: int,
                  workers: int) -> tuple[np.ndarray, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    X0 = T[cols].astype(float)
    X0 = X0.fillna(X0.median()).fillna(0.0).to_numpy()
    Xs = StandardScaler().fit_transform(X0)
    fit_idx = balanced_sample(T.reset_index(drop=True), np.ones(len(T), bool), per_bird, rng)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=seed).fit(Xs[fit_idx])
    X = pca.transform(Xs).astype(np.float64)
    k_over = int(max(2, min(k_over, len(fit_idx) // 10)))
    ks = list(range(1, min(max_k, k_over) + 1))
    gmm = fit_gmm(X[fit_idx], k_over, seed)
    merges = ward_merge(gmm.means_, ks)
    post = posterior(joint(X, gmm.weights_, gmm.means_, gmm.covariances_))
    ref = {K: (post @ onehot(merges[K], K)).argmax(1) for K in ks}
    birds = T["bird"].to_numpy()
    ub = np.unique(birds)
    by_bird = {b: fit_idx[birds[fit_idx] == b] for b in ub}
    picks = [rng.choice(ub, len(ub), replace=True) for _ in range(n_boot)]
    boots = Parallel(n_jobs=workers)(
        delayed(refit_labels)(X[np.concatenate([by_bird[b] for b in pk])], X, k_over, ks, seed + 1 + i)
        for i, pk in enumerate(picks))
    cohort_of = T.groupby("bird")["cohort"].first()
    rows = []
    for K in ks:
        jac = np.mean([jaccard_best(ref[K], bt[K], K) for bt in boots], axis=0) if boots else np.ones(K)
        pres = presence(birds, ref[K], K, cohort_of)
        rows.append({"K": K, "jaccard_min": float(jac.min()), "jaccard": jac.round(4).tolist(),
                     "presence_birds": pres["birds"], "cohort_ok": pres["cohort_ok"], "presence_ok": pres["ok"],
                     "rule_ok": bool(jac.min() >= JACCARD_MIN and pres["ok"])})
    table = pd.DataFrame(rows)
    good = [r["K"] for r in rows if r["rule_ok"] and r["K"] > 1]
    K = max(good) if good else 1
    lab = ref[K]
    med = pd.Series(T["speed_mean"].to_numpy()).groupby(lab).median().reindex(range(K)).fillna(np.inf).to_numpy()
    rank = np.argsort(np.argsort(med, kind="stable"))
    jk = table.loc[table["K"] == K, "jaccard"].iat[0]
    info = {"chosen_k": int(K), "rule_met_k_gt_1": bool(good), "k_over": k_over, "ks": ks,
            "cluster_columns": cols, "pca_components": int(pca.n_components_), "fit_sample": int(len(fit_idx)),
            "bouts": int(len(T)), "n_boot": n_boot, "seed": seed,
            "jaccard_chosen_by_type": [jk[i] for i in np.argsort(rank)],
            "rule": (f"largest K >= 2 with every bout type's mean bird-bootstrap Jaccard >= {JACCARD_MIN} and presence "
                     "(>= 5 birds with >= 20 bouts of the type, both cohorts); otherwise K = 1 (not split). Fixed "
                     "before results were seen.")}
    return rank[lab], table, info


def _share_test(T: pd.DataFrame, birdinfo: pd.DataFrame, col: str, positive, n_perm: int, rng, K: int) -> list[dict]:
    """Per type: per-bird share of bouts in the type, mean in group 1 vs group 0, permutation p between birds within
    strata (sex: deployment year x age group; young-tagged: deployment year)."""
    counts = pd.crosstab(T["bird"], T["bout_type"]).reindex(columns=range(K), fill_value=0)
    counts = counts[counts.sum(axis=1) >= MIN_BIRD_BOUTS]
    share = counts.div(counts.sum(axis=1), axis=0)
    grp = birdinfo[col].reindex(share.index)
    cohort = T.groupby("bird")["cohort"].first().reindex(share.index).astype(str)
    young = birdinfo["tagged_young"].reindex(share.index).map(lambda x: "young" if x == True else "not stated")  # noqa: E712
    if positive == "f":
        known = grp.isin(["f", "m"])
        lab = (grp[known] == "f").astype(int).to_numpy()
        strata = (cohort + "|" + young)[known].to_numpy()
    else:
        known = pd.Series(True, index=share.index)
        lab = (grp == True).astype(int).to_numpy()  # noqa: E712  (True vs NaN = not stated)
        strata = cohort.to_numpy()
    share = share[known.to_numpy()]
    out = []
    for t in range(K):
        v = share[t].to_numpy(float)
        r = {"type": t, "group": col, "n_1": int(lab.sum()), "n_0": int((1 - lab).sum())}
        r["mean_share_1"] = float(v[lab == 1].mean()) if r["n_1"] else np.nan
        r["mean_share_0"] = float(v[lab == 0].mean()) if r["n_0"] else np.nan
        if K > 1 and r["n_1"] >= 3 and r["n_0"] >= 3:
            st = strat_perm_diff(v, lab, strata, n_perm, rng)
            r.update(diff=float(v[lab == 1].mean() - v[lab == 0].mean()), diff_strat=st["diff_strat"],
                     p_perm=st["p_perm"], n_informative=st["n_informative"], nuisance_auc=nuisance_auc(lab, strata),
                     hedges_g=hedges_g(v[lab == 1], v[lab == 0]))
        out.append(r)
    return out


def circ_mean_hour(h) -> float:
    a = np.deg2rad(pd.Series(h).dropna().to_numpy(float) * 15.0)
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 15.0 % 24) if len(a) else np.nan


def type_words(row: pd.Series, trans: pd.DataFrame, sdesc: pd.DataFrame) -> str:
    """Words for a bout type from WORD_RULES only; numbers are always given next to them."""
    w = []
    cr = row["climb_rate_m_min"]
    if cr > WORD_RULES["climb_rate_m_min"]:
        w.append("net climbing")
    elif cr < -WORD_RULES["climb_rate_m_min"]:
        w.append("net descending")
    else:
        w.append("about level")
    up = set(sdesc.index[sdesc.get("climb", pd.Series(dtype=float)) > WORD_RULES["state_climb_ms"]])
    down = set(sdesc.index[sdesc.get("climb", pd.Series(dtype=float)) < -WORD_RULES["state_climb_ms"]])
    n_ud = int(trans[trans["from"].isin(up) & trans["to"].isin(down)]["count"].sum())
    n_du = int(trans[trans["from"].isin(down) & trans["to"].isin(up)]["count"].sum())
    if (row["climb_first_half_m"] > WORD_RULES["half_climb_m"] and row["climb_second_half_m"] < -WORD_RULES["half_climb_m"]
            and n_ud > n_du):
        w.append("climb then descent (climbing-state -> descending-state changes outnumber the reverse)")
    if row["circles_per_min"] >= WORD_RULES["circles_per_min"]:
        w.append("circling")
    if row["flap_share"] >= WORD_RULES["flap_share"]:
        w.append("flapping-rich")
    return "; ".join(w)


def bout_figure(types: pd.DataFrame, K6: int, out: Path) -> None:
    plt = _plt()
    K = len(types)
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.6 * K + 2.5))
    left = np.zeros(K)
    cmap = plt.get_cmap("tab10")
    for s in range(K6):
        v = types[f"share_state_{s}"].to_numpy()
        axs[0].barh(types.index, v, left=left, color=cmap(s), edgecolor="white", linewidth=1, label=f"state {s}")
        left += v
    axs[0].set_xlabel("median-bout share of K=6 states (mean over bouts)")
    axs[0].set_ylabel("bout type")
    axs[0].legend(fontsize=7, ncol=3, frameon=False)
    cols = ["speed_mean", "climb_rate_m_min", "circles_per_min", "flap_share", "duration_s"]
    M = types[cols].to_numpy(float)
    Mz = (M - np.nanmean(M, 0)) / np.where(np.nanstd(M, 0) > 0, np.nanstd(M, 0), 1)
    im = axs[1].imshow(Mz, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for (r, c), val in np.ndenumerate(M):
        axs[1].text(c, r, f"{val:.2g}", ha="center", va="center", fontsize=7)
    axs[1].set_xticks(range(len(cols)), cols, rotation=30, fontsize=7)
    axs[1].set_yticks(range(K))
    plt.colorbar(im, ax=axs[1], fraction=0.046, label="z across types")
    axs[1].set_title("type medians (text) and their z-score (colour)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "bout_types.png", dpi=110)
    plt.close(fig)


def part_b(D: pd.DataFrame, Zall: np.ndarray, H: np.ndarray, sdesc: pd.DataFrame, birdinfo: pd.DataFrame, out: Path,
           a) -> dict:
    cont = (D["sensitivity_k"].to_numpy() == 1) & (D["bout_local"].to_numpy() >= 0)
    B = D[cont].copy()
    B["bout"] = pd.factorize(B["bird"] + "|" + B["bout_local"].astype(str))[0]
    K6 = int(max(len(sdesc), D["state6"].max() + 1))
    T = bout_table(B, Zall[cont], H[cont], K6)
    n_all = len(T)
    T = T[T["duration_s"] >= MIN_BOUT_S]
    info = {"bouts_all": int(n_all), "bouts_ge_60s": int(len(T)), "birds_with_bouts": int(T["bird"].nunique()),
            "bout_definition": f"flight.bout_ids: sensitivity_k == 1 windows with consecutive k=1 starts <= "
                               f"{BOUT_STEP_S:g} s apart (windows of another mode in between do not split a bout); "
                               f"duration = last start - first start + last window length (acc_samples / {FS:g} Hz); "
                               f"kept if duration >= {MIN_BOUT_S:g} s",
            "duration_s_quantiles": T["duration_s"].quantile([0.1, 0.5, 0.9]).round(1).tolist() if len(T) else []}
    if len(T) < 30 or T["bird"].nunique() < 3:
        info["note"] = "too few bouts to cluster"
        T.drop(columns="t_start").to_csv(out / "bouts.csv")
        return info
    cols = CLUSTER_BASE + [f"share_state_{s}" for s in range(K6)] + \
        [f"pc{k}_{s}" for k in range(1, Zall.shape[1] + 1) for s in ("mean", "sd")]
    lab, ktab, cinfo = cluster_bouts(T, cols, a.bout_per_bird, a.bout_k_over, a.bout_max_k, a.n_boot, a.seed,
                                     a.workers)
    T["bout_type"] = lab
    K = cinfo["chosen_k"]
    info.update(cinfo)
    ktab.to_csv(out / "bout_k_table.csv", index=False)
    T.drop(columns="t_start").rename_axis("bout").to_csv(out / "bouts.csv")
    trans = transition_counts(B[B["bout"].isin(T.index)], T["bout_type"], K6)
    trans.to_csv(out / "bout_type_transitions.csv", index=False)
    desc = ["duration_s", "windows", "gps_share", "flight_share", "speed_mean", "net_climb_m", "climb_first_half_m",
            "climb_second_half_m", "climb_rate_m_min", "turn_gross_deg_per_min", "circles_net", "circles_per_min",
            "state_changes_per_min", "glide_ratio", "height_lost_m", "flap_share", "flap_band_frac_mean", "h_spread"] + \
        [f"pc{k}_mean" for k in range(1, Zall.shape[1] + 1)]
    g = T.groupby("bout_type")
    types = g[desc].median()
    for s in range(K6):
        types[f"share_state_{s}"] = g[f"share_state_{s}"].mean()
    types["bouts"], types["birds"] = g.size(), g["bird"].nunique()
    types["start_hour_circmean"] = g["start_hour"].apply(circ_mean_hour)
    types["glide_ratio_n"] = g["glide_ratio"].count()
    yt = T["bird"].map(birdinfo["tagged_young"]) == True  # noqa: E712
    types["bout_share_young_tagged"] = yt.groupby(T["bout_type"]).mean()
    for mo in range(1, 13):
        types[f"month_share_{mo}"] = (T["month"] == mo).groupby(T["bout_type"]).mean()
    types["words"] = [type_words(types.loc[t], trans[trans["type"] == t], sdesc) for t in types.index]
    types.to_csv(out / "bout_types.csv")
    rng = np.random.default_rng(a.seed + 7)
    tests = []
    if birdinfo["sex"].notna().any():
        tests += _share_test(T, birdinfo, "sex", "f", a.n_perm, rng, K)
    tests += _share_test(T, birdinfo, "tagged_young", True, a.n_perm, rng, K)
    pd.DataFrame(tests).to_csv(out / "bout_type_groups.csv", index=False)
    bout_figure(types, K6, out)
    info["types"] = types.reset_index().to_dict(orient="records")
    info["group_tests"] = tests
    info["k_table"] = ktab.to_dict(orient="records")
    return info


# ---------------------------------------------------------------- pipeline

def run_one(run: Path, states2: Path, states6: Path, windows: Path, birdinfo: pd.DataFrame, out: Path, a):
    rng = np.random.default_rng(a.seed)
    flight_check = check_flight_state(states2)
    sdesc = state_descriptions(states6)
    sdesc.to_csv(out / "k6_state_profiles_used.csv")
    log(f"{run.name}: loading")
    D, H, counts, cover = load_run(run, windows, states2, states6, a.state_col, a.workers, birdinfo)
    cover.to_csv(out / "bird_coverage.csv", index=False)  # bird-keyed counts (windows, states, bouts both rules)
    counts["bout_counts"] = bout_count_summary(cover)
    fl = D["flight"].to_numpy()
    F, HF = D[fl].reset_index(drop=True), H[fl]
    log(f"{run.name}: {len(F):,} flight windows of {F['bird'].nunique()} birds; {int((D['sensitivity_k'] == 1).sum()):,} "
        "continuous-mode windows")
    pca, Z, fit = fit_axes(F, HF, a.per_bird, rng, a.seed)
    corr, summ = axis_tables(F, Z, HF, pca, a.per_bird, rng)
    corr.to_csv(out / "axis_correlations.csv", index=False)
    summ.to_csv(out / "axis_summary.csv", index=False)
    sex = birdinfo["sex"]
    labels = {(r["subset"], r["pc"]): r["label"] for r in summ.to_dict(orient="records")}
    tests, bmeans = sex_tests(F, Z, birdinfo, a.n_perm, rng, labels)
    bmeans.to_csv(out / "bird_pc_means.csv", index=False)  # sex_tests_axes.csv is written after Holm over runs
    maps = flight_maps(F, Z, fit, sex, out)
    umap_note = umap_figure(F, HF, a.umap_per_bird, rng, a.seed, out) if a.umap else "not requested"
    Zall = pca.transform(H).astype(np.float32)  # PC signs are folded into pca.components_ by fit_axes
    log(f"{run.name}: bouts")
    binfo = part_b(D, Zall, H, sdesc, birdinfo, out, a)
    res = {"run": run.name, "states2": str(states2), "states6": str(states6), "flight_check": flight_check,
           **counts, "flight_windows": int(len(F)), "flight_windows_k0": int((F["sensitivity_k"] == 0).sum()),
           "flight_windows_k1": int((F["sensitivity_k"] == 1).sum()), "flight_birds": int(F["bird"].nunique()),
           "pca_fit_sample": int(len(fit)), "variance_explained": pca.explained_variance_ratio_.round(4).tolist(),
           "axis_summary": summ.to_dict(orient="records"), "sex_tests": tests,
           "maps": maps, "umap": umap_note, "bouts": binfo, "k6_states": sdesc.round(3).reset_index().to_dict("records")}
    keys = F[["bird", "row"]].copy()
    return res, keys, Z


def bout_count_summary(cover: pd.DataFrame) -> dict:
    """Bout counts and birds under the canonical rule and the first run's rule (counts only)."""
    def q(x):
        return [float(v) for v in np.quantile(x, [0, 0.5, 1])] if len(x) else []
    k1 = cover[cover["k1_windows"] > 0]
    return {"birds_with_k1_windows": int(len(k1)),
            "canonical": {"bouts": int(cover["bouts_all"].sum()), "bouts_ge_60s": int(cover["bouts_ge_60s"].sum()),
                          "birds_ge_60s": int((cover["bouts_ge_60s"] > 0).sum()),
                          "per_bird_ge_60s_min_median_max": q(k1["bouts_ge_60s"])},
            "legacy": {"bouts": int(cover["legacy_bouts_all"].sum()), "bouts_ge_60s": int(cover["legacy_bouts_ge_60s"].sum()),
                       "birds_ge_60s": int((cover["legacy_bouts_ge_60s"] > 0).sum()),
                       "per_bird_ge_60s_min_median_max": q(k1["legacy_bouts_ge_60s"])},
            "other_mode_windows_inside_bouts": int(cover["k0_inside_bouts"].sum())}


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def cross_run(keys: list, Zs: list, names: list, rng, n_max: int = 200_000) -> pd.DataFrame:
    """Spearman between the PCs of two runs on windows that are flight in both; bird-balanced sample (at most
    n_max / birds windows per bird)."""
    a = keys[0].assign(i=np.arange(len(keys[0])))
    b = keys[1].assign(j=np.arange(len(keys[1])))
    m = a.merge(b, on=["bird", "row"])
    if len(m):
        cap = max(1, n_max // m["bird"].nunique())
        pick = [g.index.to_numpy() if len(g) <= cap else rng.choice(g.index.to_numpy(), cap, replace=False)
                for _, g in m.groupby("bird")]
        m = m.loc[np.sort(np.concatenate(pick))]
    rows = []
    for p in range(Zs[0].shape[1]):
        for q in range(Zs[1].shape[1]):
            r, n = spearman(Zs[0][m["i"].to_numpy(), p], Zs[1][m["j"].to_numpy(), q])
            rows.append({f"pc_{names[0]}": p + 1, f"pc_{names[1]}": q + 1, "spearman": r, "n": n})
    return pd.DataFrame(rows)


def _f(x, nd=2):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _fp(p, n_perm: int) -> str:
    """Permutation p; at the floor 1/(n_perm + 1) it is printed as '<= 1/(n_perm+1)'."""
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "NA"
    return f"<= 1/{n_perm + 1}" if p <= 1.0 / (n_perm + 1) + 1e-12 else f"{p:.4f}"


def write_summary(out: Path, results: list[dict], cross: pd.DataFrame | None, meta: dict) -> None:
    n_perm = int(meta.get("args", {}).get("n_perm", 2000))
    L = ["# Flight continuum and flight bouts", "",
         "Descriptive unless marked TESTED. Every test uses the bird as the unit (bird values; labels permuted between "
         "birds). Recording mode matters: scheduled-mode (k=0) flight windows have almost no GPS and continuous-mode "
         "(k=1) windows almost always have it, so every axis is also reported per mode. In k=0, correlates with "
         "speed and climb come from the few k=0 windows that have GPS (a selected subset; see n).", ""]
    if meta.get("reference_note"):
        L += [f"Reference data: {meta['reference_note']}", ""]
    for r in results:
        bc = r.get("bout_counts", {})
        L += [f"## Run {r['run']}", "",
              f"- Birds: {r['birds']} with windows ({r.get('bird_dirs_without_windows', 0)} bird directories without "
              f"windows); missing from the K=2 states {r.get('birds_missing_from_states2', 0)}, from the K=6 states "
              f"{r.get('birds_missing_from_states6', 0)}; {r.get('birds_without_flight_windows', 0)} birds have no K=2 "
              f"flight window (K=2 column {meta.get('args', {}).get('state_col', 'state_hmm')}). Cohort = deployment "
              f"year (reference deploy_on): {r.get('cohort_birds', {})}; first-window year used for "
              f"{r.get('cohort_from_first_window_year', 0)} birds without deploy_on. Bird-keyed counts: bird_coverage.csv.",
              f"- Flight windows (K=2 state 1): {r['flight_windows']:,} of {r['windows_total']:,} windows, "
              f"{r['flight_birds']} birds ({r['flight_windows_k0']:,} scheduled mode, {r['flight_windows_k1']:,} continuous mode).",
              f"- PCA of unit h on a bird-balanced sample of {r['pca_fit_sample']:,} flight windows. Variance explained "
              f"in the fit sample by PC1-PC{len(r['variance_explained'])}: " + ", ".join(f"{v:.3f}" for v in r["variance_explained"]) + ".",
              "", "### A. Axes of flight (descriptive)", "",
              "var. expl. is the PC's share of the variance of unit h within that subset's bird-balanced sample.", "",
              "| subset | PC | var. expl. | eta2 bird | eta2 month | R2 hour (circ.) | strongest correlates in this subset (Spearman, n windows) |",
              "|---|---|---|---|---|---|---|"]
        for x in r["axis_summary"]:
            L.append(f"| {x['subset']} | {x['pc']} | {_f(x['variance_explained'], 3)} | {_f(x['eta2_bird'])} | "
                     f"{_f(x['eta2_month'])} | {_f(x['circ_r2_hour'])} | {x['label']} |")
        L += ["", "An axis whose strongest correlates are sensitivity_k or acc_samples is a recording-format axis, not "
              "a flight axis. The same PC can have different correlates in k0 and k1: read each test row with its own "
              "subset's correlates. eta2 bird is the share of the axis variance that lies between birds.", "",
              "### A. Sex along each axis (TESTED)", "",
              f"Bird value = mean of the bird's month means (months with >= {MIN_BM_WINDOWS} windows; birds with >= "
              f"{MIN_BIRD_WINDOWS} windows in the subset). Statistic = weighted mean of the female-minus-male differences "
              "within strata deployment year x age group (tagged young / not stated); sex labels permuted between birds "
              "within strata; birds in strata holding one sex do not enter (n inf.). raw f - m and Hedges g are on all "
              "birds with known sex. Nuisance AUC = how well the strata alone separate the sexes (>= 0.9: not testable "
              f"in this design). p at the permutation floor is printed as '<= 1/{n_perm + 1}'. Holm family: all runs x "
              "subsets x PCs of variant 'all' (primary); variant 'full' (windows with acc_samples >= "
              f"{FULL_SAMPLES} only) is a sensitivity family with its own Holm.", "",
              "| variant | subset | PC | axis in this subset | n f | n m | n inf. | nuisance AUC | raw f - m | strat. f - m | "
              "Hedges g | p | p Holm (family size) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in r["sex_tests"]:
            ph, pp = t.get("p_holm"), t.get("p_perm")
            if pp is not None and np.isfinite(pp) and not t.get("testable"):
                pstr, hol = "not testable in this design", "-"
            else:
                pstr = _fp(pp, n_perm)
                hol = f"{_f(ph, 4)} ({int(t['holm_family_size'])})" if ph is not None and np.isfinite(ph) else "NA"
            L.append(f"| {t['variant']} | {t['subset']} | {t['pc']} | {t.get('axis_label', '')} | {t['n_f']} | {t['n_m']} | "
                     f"{t.get('n_informative', 0)} | {_f(t.get('nuisance_auc'))} | {_f(t.get('diff_f_minus_m'), 3)} | "
                     f"{_f(t.get('diff_strat'), 3)} | {_f(t.get('hedges_g'))} | {pstr} | {hol} |")
        bo = r["bouts"]
        L += ["", "### B. Continuous-mode bouts", "", f"- Definition (canonical; other modules import flight.bout_ids): {bo['bout_definition']}.",
              f"- Bouts: {bo['bouts_all']:,} in total, {bo['bouts_ge_60s']:,} of at least {MIN_BOUT_S:g} s from "
              f"{bo['birds_with_bouts']} birds. Duration 10/50/90 % quantiles (s): {bo['duration_s_quantiles']}. "
              "Continuous recording is capped near 900 s, so duration is partly a recording setting."]
        if bc:
            c, g = bc["canonical"], bc["legacy"]
            L += [f"- {bc['birds_with_k1_windows']} of {r['birds']} birds have continuous-mode (k=1) ACC windows; birds "
                  "without them record only scheduled ACC bursts, so bouts counted from raw 1 Hz GPS (stage 6) include "
                  "birds that can have no bout here. Per-bird counts under both rules are in bird_coverage.csv.",
                  f"- Canonical rule: {c['bouts']:,} bouts, {c['bouts_ge_60s']:,} >= 60 s in {c['birds_ge_60s']} birds "
                  f"(per bird min/median/max {c['per_bird_ge_60s_min_median_max']}). First run's rule (25 s, any "
                  f"other-mode window splits): {g['bouts']:,} bouts, {g['bouts_ge_60s']:,} >= 60 s in {g['birds_ge_60s']} "
                  f"birds (per bird {g['per_bird_ge_60s_min_median_max']}). Other-mode windows inside canonical bouts: "
                  f"{bc['other_mode_windows_inside_bouts']:,}."]
        if "chosen_k" in bo:
            L += [f"- Bout types: rule ({bo['rule']}) gives K = {bo['chosen_k']}"
                  + (" (no K >= 2 met the rule: bouts are not split into stable types)." if not bo["rule_met_k_gt_1"] else ".")
                  + f" Bootstrap refits: {bo['n_boot']}.",
                  "", "| K | min Jaccard | presence ok | rule ok |", "|---|---|---|---|"]
            for k in bo["k_table"]:
                L.append(f"| {k['K']} | {_f(k['jaccard_min'], 3)} | {k['presence_ok']} | {k['rule_ok']} |")
            L += ["", "K=6 states used for shares and transitions (from this run's state_profiles.csv; the K=6 "
                  "partition is itself a fallback choice that did not meet the stability rule):", ""]
            for x in r["k6_states"]:
                L.append(f"- state {x['state']}: " + ", ".join(f"{k} {_f(v, 2)}" for k, v in x.items() if k != "state"))
            L += ["", "Bout types (medians over bouts; words only from fixed numeric rules; circles are NET turning / 360, "
                  "so opposite-direction turns cancel; gross absolute turning is given only as a rate in deg/min because "
                  "GPS heading noise inflates it):", ""]
            for t in bo["types"]:
                top = sorted(((t[f'share_state_{x}'], x) for x in range(len(r["k6_states"])) if f"share_state_{x}" in t
                              and t[f'share_state_{x}'] is not None and np.isfinite(t[f'share_state_{x}'])), reverse=True)[:2]
                L.append(f"- Type {t['bout_type']} ({t['bouts']} bouts, {t['birds']} birds): {t['words']}. Duration "
                         f"{_f(t['duration_s'], 0)} s, speed {_f(t['speed_mean'], 1)} m/s, net climb "
                         f"{_f(t['net_climb_m'], 0)} m (first half {_f(t['climb_first_half_m'], 0)}, second half "
                         f"{_f(t['climb_second_half_m'], 0)}), {_f(t['circles_per_min'], 2)} net circles/min, gross "
                         f"turning {_f(t['turn_gross_deg_per_min'], 0)} deg/min, glide ratio "
                         f"{_f(t['glide_ratio'], 1)} (n={t['glide_ratio_n']}), flapping share {_f(t['flap_share'], 2)}, "
                         f"main states " + ", ".join(f"{x} ({_f(v, 2)})" for v, x in top) +
                         f", mean start hour {_f(t['start_hour_circmean'], 1)} solar, share of bouts by young-tagged "
                         f"birds {_f(t['bout_share_young_tagged'], 2)}.")
            if bo["group_tests"]:
                L += ["", "Bird-level shares of bout types (TESTED per type, exploratory, no correction; permutation within "
                      "deployment year x age group for sex, within deployment year for young-tagged):", "",
                      "| type | group | n 1 | n 0 | mean share 1 | mean share 0 | nuisance AUC | p |", "|---|---|---|---|---|---|---|---|"]
                for t in bo["group_tests"]:
                    g1 = "female vs male" if t["group"] == "sex" else "young-tagged vs not stated"
                    L.append(f"| {t['type']} | {g1} | {t['n_1']} | {t['n_0']} | {_f(t['mean_share_1'], 3)} | "
                             f"{_f(t['mean_share_0'], 3)} | {_f(t.get('nuisance_auc'))} | {_fp(t.get('p_perm'), n_perm)} |")
        elif "note" in bo:
            L.append(f"- {bo['note']}.")
        L += ["", "Height and distance per window use the window's GPS seconds (climb_mean x n_speed, speed_mean x "
              "n_speed), not 10 s. Glide ratio is ground distance over height lost in descending windows (climb < "
              "-0.5 m/s; ground speed, so wind is included); it is NaN below 20 m of height lost.", ""]
    if cross is not None and len(cross):
        L += ["## Axes compared between runs (descriptive)", "",
              "Spearman between the PCs of the two runs on windows that are flight in both, bird-balanced sample "
              "(cross_run_pc_spearman.csv). Largest |rho| per PC of the first run:", ""]
        c0, c1 = cross.columns[0], cross.columns[1]
        for p, g in cross.groupby(c0):
            best = g.loc[g["spearman"].abs().idxmax()]
            L.append(f"- PC{p}: PC{int(best[c1])} ({_f(best['spearman'])}, n={int(best['n']):,})")
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--states2", type=Path, action="append", required=True)
    ap.add_argument("--states6", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--state-col", default="state_hmm",
                    help="K=2 column defining flight (state_hmm, as budgets.py, or state)")
    ap.add_argument("--per-bird", type=int, default=3000)
    ap.add_argument("--umap", action="store_true")
    ap.add_argument("--umap-per-bird", type=int, default=300)
    ap.add_argument("--bout-per-bird", type=int, default=300)
    ap.add_argument("--bout-k-over", type=int, default=20)
    ap.add_argument("--bout-max-k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if not (len(a.run) == len(a.states2) == len(a.states6)):
        raise SystemExit("--run, --states2 and --states6 must be given the same number of times, in the same order")
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty; give a new directory")
    names = [r.name for r in a.run]
    if len(set(names)) != len(names):
        raise SystemExit("run directories must have different names")
    birds = sorted(p.name for p in a.windows.iterdir()
                   if p.is_dir() and (p / "meta.parquet").exists() and pq.read_metadata(p / "meta.parquet").num_rows > 0)
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "word_rules": WORD_RULES,
            "constants": {"BOUT_STEP_S": BOUT_STEP_S, "MIN_BOUT_S": MIN_BOUT_S, "GLIDE_MIN_LOSS_M": GLIDE_MIN_LOSS_M,
                          "GLIDE_MIN_SINK_MS": GLIDE_MIN_SINK_MS,
                          "MIN_CELL": MIN_CELL, "MIN_BIRD_WINDOWS": MIN_BIRD_WINDOWS, "MIN_BIRD_BOUTS": MIN_BIRD_BOUTS}}
    if a.reference.exists():
        birdinfo = reference.load_birds(a.reference, birds).set_index("bird")[["sex", "tagged_young", "deploy_on"]]
        dy = birdinfo["deploy_on"].dt.year.value_counts().sort_index()
        meta["reference_note"] = (f"{len(birds)} birds with windows: sex f {int((birdinfo['sex'] == 'f').sum())}, m "
                                  f"{int((birdinfo['sex'] == 'm').sum())}, missing {int(birdinfo['sex'].isna().sum())}; "
                                  f"young-tagged {int(birdinfo['tagged_young'].notna().sum())}; deployment year "
                                  + ", ".join(f"{int(y)}: {n}" for y, n in dy.items())
                                  + f", missing {int(birdinfo['deploy_on'].isna().sum())}")
    else:
        birdinfo = pd.DataFrame({"sex": np.nan, "tagged_young": np.nan, "deploy_on": pd.NaT},
                                index=pd.Index(birds, name="bird"))
        meta["reference_note"] = "reference CSV not found; sex and young-tagged outputs are empty"
    a.out.mkdir(parents=True, exist_ok=True)
    results, keys, Zs = [], [], []
    for run, s2, s6 in zip(a.run, a.states2, a.states6):
        o = a.out / run.name
        o.mkdir()
        r, k, z = run_one(run, s2, s6, a.windows, birdinfo, o, a)
        results.append(r)
        keys.append(k)
        Zs.append(z)
    holm_family([r["sex_tests"] for r in results])
    for r in results:
        o = a.out / r["run"]
        r["sex_tests"].to_csv(o / "sex_tests_axes.csv", index=False)
        r["sex_tests"] = r["sex_tests"].to_dict(orient="records")
        (o / "flight.json").write_text(json.dumps(r, indent=1, default=_json))
    cross = None
    if len(results) == 2:
        cross = cross_run(keys, Zs, names, np.random.default_rng(a.seed))
        cross.to_csv(a.out / "cross_run_pc_spearman.csv", index=False)
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=1, default=_json))
    write_summary(a.out, results, cross, meta)
    log("done")


if __name__ == "__main__":
    main()
