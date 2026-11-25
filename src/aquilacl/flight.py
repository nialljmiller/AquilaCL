"""Flight as a continuum (window scale) and flight bouts (minute scale) on a frozen SimCLR embedding.

Background: at the 10-s window scale, flight windows do not split into stable sub-states (the sub-state rule fails
at every k), so flight is treated here as a continuum in h (part A), and structure is sought at the scale of
continuous-mode bouts lasting minutes (part B).

Inputs per run (same order on the command line): the run directory (embeddings/<bird>.npz key 'h'), the K=2 states
directory (state 1 = flight; checked from its state_profiles.csv: state 1 must have the higher median speed) and the
K=6 states directory (state_hmm, described only by its own state_profiles.csv, never by a hard-coded name: runs
number states differently). Rows are joined on (bird, row), row being the index within the bird's meta.parquet.

Part A (window scale, flight windows = K=2 ``state`` == 1 by default)
- h is L2-normalised. PCA is fitted on a bird-balanced sample (states.balanced_sample: up to ``per_bird`` windows
  per bird, months equally weighted within a bird). The sign of each PC is fixed so that its strongest correlate
  (largest |Spearman rho| on the pooled sample) is positive.
- Per PC and per subset (all flight windows, scheduled mode k=0 only, continuous mode k=1 only; recording mode
  decides which windows have GPS, so pooled numbers mix two populations): Spearman rho with speed_mean, climb_mean,
  turn_abs_mean, vedba, flap_band_frac, dom_freq_hz (NaN when acc_samples < 40, as in states.profiles),
  local_solar_hour, hours_from_noon, month, sensitivity_k, acc_samples, each on its pairwise-complete windows of a
  bird-balanced sample of that subset, with n; eta^2 by bird and by month; R^2 of the PC on sin/cos of hour and of
  month (circular-linear). Axis label = the three strongest correlates. All descriptive.
- Flight maps (descriptive): PC1/PC2 grid; median of each feature per cell (cells with >= MIN_CELL windows);
  bird-weighted density (each bird's density sums to 1, then averaged over birds); female minus male density and
  month minus all-months density, per recording mode. Speed vs climb map (windows with GPS) coloured by median PC.
- Sex test (TESTED): per bird the mean PC score (birds with >= MIN_BIRD_WINDOWS flight windows in the subset);
  difference female minus male with a two-sided permutation p (sex labels permuted between birds), Hedges' g,
  n_f, n_m; Holm correction over all PC x subset tests of a run.
- With two runs: Spearman between the PCs of the two runs on windows that are flight in both (descriptive).

Part B (bout scale)
- A continuous-mode bout is a maximal run of a bird's sensitivity_k == 1 windows in time order with consecutive
  window starts at most BOUT_STEP_S = 25 s apart (10-s window + gap <= 15 s) and no window of another mode in
  between. Segments of windows.py break at 1.5 s acceleration gaps and k_in_segment has holes where short windows
  were dropped, so the time rule, not segment, joins windows. Bouts >= MIN_BOUT_S = 60 s are kept
  (duration = last start - first start + 10 s; continuous recording is capped near 900 s, so duration is partly a
  recording setting).
- Descriptors per bout: duration, windows, GPS share, flight share (K=2), mean speed, net climb = sum(climb_mean x
  10 s) and its first-half / second-half split, climb rate per minute, total absolute turning = sum(turn_abs_mean x
  n_turn), gross circles = total / 360, net circles = |sum turn_net_deg| / 360, circles per minute, share of each
  K=6 state, state changes per minute, glide ratio = sum(speed_mean x 10) / sum(-climb_mean x 10) over descending
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
  MIN_BIRD_BOUTS bouts; permutation p between birds, TESTED but exploratory, no correction). Words used for a type
  come only from fixed numeric rules (WORD_RULES).

Outputs: --out/<run name>/ (csv, png, json) per run, cross_run_pc_spearman.csv, summary.md, metadata.json. Tables
may be keyed by bird; no position, no exact timestamp is written.

Usage: python -m aquilacl.flight --run R [--run R2] --windows W --states2 S2 [--states2 ...] --states6 S6
       [--states6 ...] --out NEW [--reference CSV] [--n-boot 20] [--n-perm 2000] [--workers N] [--seed 0] [--umap]
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
from .demography import hedges_g, perm_mean_diff
from .states import (JACCARD_MIN, SPECTRUM_SAMPLES, balanced_sample, fit_gmm, jaccard_best, joint, onehot,
                     posterior, refit_labels, ward_merge)
from .substates import presence

DEFAULT_REFERENCE = Path("/project/ai4wy-eap/nmille39/aquilacl/source/movebank_2026-11-24/"
                         "LifeTrack Golden Eagle Bighorn Basin WY-reference-data(1).csv")
WINDOW_S = 10.0
BOUT_STEP_S = 25.0
MIN_BOUT_S = 60.0
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
    """Bout id per window of ONE bird in time order (-1 for windows not in continuous mode).

    A new bout starts at a continuous window whose predecessor is not continuous or starts more than ``max_step``
    seconds earlier."""
    t_s, cont = np.asarray(t_s, float), np.asarray(cont, bool)
    new = cont.copy()
    if len(t_s) > 1:
        new[1:] &= ~cont[:-1] | (np.diff(t_s) > max_step)
    ids = np.cumsum(new) - 1
    return np.where(cont, ids, -1)


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
    df.loc[order, "bout_local"] = bout_ids(t_s[order], (df["sensitivity_k"].to_numpy() == 1)[order])
    df["t_s"] = t_s
    keep = (flight | (df["sensitivity_k"].to_numpy() == 1))
    H = np.asarray(h[keep], np.float32)
    H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    out = df[keep].reset_index(drop=True)
    out.insert(0, "bird", d.name)
    return out, H, int(m["t_start"].min().year), n


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


def load_run(run: Path, windows: Path, states2: Path, states6: Path, state_col: str, workers: int):
    s2, s6 = _by_bird(states2, state_col), _by_bird(states6, "state_hmm")
    dirs = sorted(p for p in windows.iterdir() if p.is_dir())
    res = Parallel(n_jobs=workers)(delayed(load_bird)(d, run, s2.get(d.name), s6.get(d.name)) for d in dirs)
    res = [r for r in res if r is not None]
    D = pd.concat([r[0] for r in res], ignore_index=True)
    H = np.concatenate([r[1] for r in res])
    first_year = {r[0]["bird"].iat[0]: r[2] for r in res if len(r[0])}
    D["cohort"] = D["bird"].map(lambda b: "2024" if first_year[b] <= 2024 else "2025")
    D["month"] = D["t_start"].dt.month
    lsh = D["local_solar_hour"]
    D["hours_from_noon"] = np.abs(lsh % 24.0 - 12.0)
    D.loc[D["acc_samples"] < SPECTRUM_SAMPLES, ["flap_band_frac", "dom_freq_hz"]] = np.nan
    return D, H, {"windows_total": int(sum(r[3] for r in res)), "birds": int(len(res))}


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


def axis_tables(F: pd.DataFrame, Z: np.ndarray, pca: PCA, per_bird: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr, summ = [], []
    for sub in SUBSETS:
        mask = _subset_mask(F, sub)
        if mask.sum() < 3:
            continue
        smp = balanced_sample(F, mask, per_bird, rng)
        for i in range(Z.shape[1]):
            z = Z[smp, i]
            rows = []
            for v in CORR_VARS:
                r, n = spearman(F[v].to_numpy()[smp], z)
                rows.append({"subset": sub, "pc": i + 1, "variable": v, "spearman": r, "n": n})
            corr += rows
            top = sorted((r for r in rows if np.isfinite(r["spearman"]) and r["n"] >= 30),
                         key=lambda r: -abs(r["spearman"]))[:3]
            summ.append({"subset": sub, "pc": i + 1, "variance_explained": float(pca.explained_variance_ratio_[i]),
                         "windows_sample": int(len(smp)), "birds": int(F["bird"].iloc[smp].nunique()),
                         "eta2_bird": eta2(z, F["bird"].to_numpy()[smp]),
                         "eta2_month": eta2(z, F["month"].to_numpy()[smp]),
                         "circ_r2_hour": circ_r2(z, F["local_solar_hour"].to_numpy()[smp], 24.0),
                         "circ_r2_month": circ_r2(z, F["month"].to_numpy()[smp] - 0.5, 12.0),
                         "label": ", ".join(f"{r['variable']} {r['spearman']:+.2f}" for r in top)})
    return pd.DataFrame(corr), pd.DataFrame(summ)


def sex_tests(F: pd.DataFrame, Z: np.ndarray, sex: pd.Series, n_perm: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bird mean PC scores per subset and female-minus-male permutation tests between birds."""
    means, rows = [], []
    for sub in SUBSETS:
        mask = _subset_mask(F, sub)
        g = pd.DataFrame(Z[mask], columns=[f"pc{i + 1}" for i in range(Z.shape[1])]).groupby(F["bird"].to_numpy()[mask])
        bm = g.mean()
        bm["windows"] = g.size()
        bm = bm[bm["windows"] >= MIN_BIRD_WINDOWS]
        bm["subset"], bm["sex"] = sub, sex.reindex(bm.index).to_numpy()
        means.append(bm.rename_axis("bird").reset_index())
        known = bm[bm["sex"].isin(["f", "m"])]
        lab = (known["sex"] == "f").to_numpy().astype(int)
        for i in range(Z.shape[1]):
            v = known[f"pc{i + 1}"].to_numpy(float)
            row = {"subset": sub, "pc": i + 1, "n_f": int(lab.sum()), "n_m": int((1 - lab).sum())}
            if row["n_f"] >= 3 and row["n_m"] >= 3:
                d, p = perm_mean_diff(v, lab, n_perm, rng)
                row.update(diff_f_minus_m=d, hedges_g=hedges_g(v[lab == 1], v[lab == 0]), p_perm=p)
            rows.append(row)
    t = pd.DataFrame(rows)
    if "p_perm" in t:
        t["p_holm"] = holm(t["p_perm"].to_numpy())
    return t, pd.concat(means, ignore_index=True) if means else pd.DataFrame()


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
    t0, t1 = g["t_s"].transform("min"), g["t_s"].transform("max")
    mid = (t0 + t1 + WINDOW_S) / 2
    climb_m = B["climb_mean"] * WINDOW_S
    first = (B["t_s"] + WINDOW_S / 2) < mid
    desc = (B["climb_mean"] < -GLIDE_MIN_SINK_MS) & B["speed_mean"].notna()
    turn_abs = B["turn_abs_mean"] * B["n_turn"]
    T = pd.DataFrame({
        "bird": g["bird"].first(), "cohort": g["cohort"].first(), "t_start": g["t_start"].min(),
        "windows": g.size(), "duration_s": g["t_s"].max() - g["t_s"].min() + WINDOW_S,
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
    dist = (B["speed_mean"] * WINDOW_S).where(desc).groupby(B["bout"]).sum(min_count=1)
    loss = (-climb_m).where(desc).groupby(B["bout"]).sum(min_count=1)
    T["height_lost_m"] = loss
    T["glide_ratio"] = (dist / loss).where(loss >= GLIDE_MIN_LOSS_M)
    minutes = T["duration_s"] / 60.0
    T["log_duration"] = np.log10(T["duration_s"])
    T["climb_rate_m_min"] = T["net_climb_m"] / minutes
    T["circles_gross"] = T["turn_abs_total_deg"] / 360.0
    T["circles_net"] = T["turn_net_deg"].abs() / 360.0
    T["circles_per_min"] = T["circles_gross"] / minutes
    st = B["state6"].to_numpy()
    has = st >= 0  # windows without a K=6 state do not count in the shares
    sh = pd.crosstab(B["bout"].to_numpy()[has], st[has], normalize="index").reindex(index=T.index, columns=range(K6))
    for s in range(K6):
        T[f"share_state_{s}"] = sh[s].reindex(T.index).fillna(0.0).to_numpy()
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
    """Per type: per-bird share of bouts in the type, mean in group 1 vs group 0, permutation p between birds."""
    counts = pd.crosstab(T["bird"], T["bout_type"]).reindex(columns=range(K), fill_value=0)
    counts = counts[counts.sum(axis=1) >= MIN_BIRD_BOUTS]
    share = counts.div(counts.sum(axis=1), axis=0)
    grp = birdinfo[col].reindex(share.index)
    if positive == "f":
        known = grp.isin(["f", "m"])
        lab = (grp[known] == "f").astype(int).to_numpy()
    else:
        known = pd.Series(True, index=share.index)
        lab = (grp == True).astype(int).to_numpy()  # noqa: E712  (True vs NaN = not stated)
    share = share[known.to_numpy()]
    out = []
    for t in range(K):
        v = share[t].to_numpy(float)
        r = {"type": t, "group": col, "n_1": int(lab.sum()), "n_0": int((1 - lab).sum())}
        r["mean_share_1"] = float(v[lab == 1].mean()) if r["n_1"] else np.nan
        r["mean_share_0"] = float(v[lab == 0].mean()) if r["n_0"] else np.nan
        if K > 1 and r["n_1"] >= 3 and r["n_0"] >= 3:
            r["diff"], r["p_perm"] = perm_mean_diff(v, lab, n_perm, rng)
            r["hedges_g"] = hedges_g(v[lab == 1], v[lab == 0])
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
            "bout_definition": f"sensitivity_k == 1 windows, consecutive starts <= {BOUT_STEP_S} s apart, no other "
                               f"mode between; kept if duration >= {MIN_BOUT_S} s",
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
            "climb_second_half_m", "climb_rate_m_min", "circles_gross", "circles_net", "circles_per_min",
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
    D, H, counts = load_run(run, windows, states2, states6, a.state_col, a.workers)
    fl = D["flight"].to_numpy()
    F, HF = D[fl].reset_index(drop=True), H[fl]
    log(f"{run.name}: {len(F):,} flight windows of {F['bird'].nunique()} birds; {int((D['sensitivity_k'] == 1).sum()):,} "
        "continuous-mode windows")
    pca, Z, fit = fit_axes(F, HF, a.per_bird, rng, a.seed)
    corr, summ = axis_tables(F, Z, pca, a.per_bird, rng)
    corr.to_csv(out / "axis_correlations.csv", index=False)
    summ.to_csv(out / "axis_summary.csv", index=False)
    sex = birdinfo["sex"]
    tests, bmeans = sex_tests(F, Z, sex, a.n_perm, rng)
    tests.to_csv(out / "sex_tests_axes.csv", index=False)
    bmeans.to_csv(out / "bird_pc_means.csv", index=False)
    maps = flight_maps(F, Z, fit, sex, out)
    umap_note = umap_figure(F, HF, a.umap_per_bird, rng, a.seed, out) if a.umap else "not requested"
    Zall = pca.transform(H).astype(np.float32)  # PC signs are folded into pca.components_ by fit_axes
    log(f"{run.name}: bouts")
    binfo = part_b(D, Zall, H, sdesc, birdinfo, out, a)
    res = {"run": run.name, "states2": str(states2), "states6": str(states6), "flight_check": flight_check,
           **counts, "flight_windows": int(len(F)), "flight_windows_k0": int((F["sensitivity_k"] == 0).sum()),
           "flight_windows_k1": int((F["sensitivity_k"] == 1).sum()), "flight_birds": int(F["bird"].nunique()),
           "pca_fit_sample": int(len(fit)), "variance_explained": pca.explained_variance_ratio_.round(4).tolist(),
           "axis_summary": summ.to_dict(orient="records"), "sex_tests": tests.to_dict(orient="records"),
           "maps": maps, "umap": umap_note, "bouts": binfo, "k6_states": sdesc.round(3).reset_index().to_dict("records")}
    (out / "flight.json").write_text(json.dumps(res, indent=1, default=_json))
    keys = F[["bird", "row"]].copy()
    return res, keys, Z


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
    a = keys[0].assign(i=np.arange(len(keys[0])))
    b = keys[1].assign(j=np.arange(len(keys[1])))
    m = a.merge(b, on=["bird", "row"])
    if len(m) > n_max:
        m = m.iloc[np.sort(rng.choice(len(m), n_max, replace=False))]
    rows = []
    for p in range(Zs[0].shape[1]):
        for q in range(Zs[1].shape[1]):
            r, n = spearman(Zs[0][m["i"].to_numpy(), p], Zs[1][m["j"].to_numpy(), q])
            rows.append({f"pc_{names[0]}": p + 1, f"pc_{names[1]}": q + 1, "spearman": r, "n": n})
    return pd.DataFrame(rows)


def _f(x, nd=2):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_summary(out: Path, results: list[dict], cross: pd.DataFrame | None, meta: dict) -> None:
    L = ["# Flight continuum and flight bouts", "",
         "Descriptive unless marked TESTED. Every test uses the bird as the unit (bird means; labels permuted between "
         "birds). Recording mode matters: scheduled-mode (k=0) flight windows have almost no GPS and continuous-mode "
         "(k=1) windows almost always have it, so every axis is also reported per mode.", ""]
    if meta.get("reference_note"):
        L += [f"Reference data: {meta['reference_note']}", ""]
    for r in results:
        L += [f"## Run {r['run']}", "",
              f"- Flight windows (K=2 state 1): {r['flight_windows']:,} of {r['windows_total']:,} windows, "
              f"{r['flight_birds']} birds ({r['flight_windows_k0']:,} scheduled mode, {r['flight_windows_k1']:,} continuous mode).",
              f"- PCA of unit h on a bird-balanced sample of {r['pca_fit_sample']:,} flight windows. Variance explained "
              f"by PC1-PC{len(r['variance_explained'])}: " + ", ".join(f"{v:.3f}" for v in r["variance_explained"]) + ".",
              "", "### A. Axes of flight (descriptive)", "",
              "| subset | PC | var. expl. | eta2 bird | eta2 month | R2 hour (circ.) | strongest correlates (Spearman) |",
              "|---|---|---|---|---|---|---|"]
        for s in r["axis_summary"]:
            L.append(f"| {s['subset']} | {s['pc']} | {_f(s['variance_explained'], 3)} | {_f(s['eta2_bird'])} | "
                     f"{_f(s['eta2_month'])} | {_f(s['circ_r2_hour'])} | {s['label']} |")
        L += ["", "An axis whose strongest correlates are sensitivity_k or acc_samples is a recording-format axis, not "
              "a flight axis. eta2 bird is the share of the axis variance that lies between birds.", "",
              "### A. Sex along each axis (TESTED, bird means, permutation between birds, Holm over the rows)", "",
              "| subset | PC | n f | n m | f - m | Hedges g | p | p Holm |", "|---|---|---|---|---|---|---|---|"]
        for t in r["sex_tests"]:
            L.append(f"| {t['subset']} | {t['pc']} | {t['n_f']} | {t['n_m']} | {_f(t.get('diff_f_minus_m'), 3)} | "
                     f"{_f(t.get('hedges_g'))} | {_f(t.get('p_perm'), 4)} | {_f(t.get('p_holm'), 4)} |")
        bo = r["bouts"]
        L += ["", "### B. Continuous-mode bouts", "", f"- Definition: {bo['bout_definition']}.",
              f"- Bouts: {bo['bouts_all']:,} in total, {bo['bouts_ge_60s']:,} of at least 60 s from "
              f"{bo['birds_with_bouts']} birds. Duration 10/50/90 % quantiles (s): {bo['duration_s_quantiles']}. "
              "Continuous recording is capped near 900 s, so duration is partly a recording setting."]
        if "chosen_k" in bo:
            L += [f"- Bout types: rule ({bo['rule']}) gives K = {bo['chosen_k']}"
                  + (" (no K >= 2 met the rule: bouts are not split into stable types)." if not bo["rule_met_k_gt_1"] else "."),
                  "", "| K | min Jaccard | presence ok | rule ok |", "|---|---|---|---|"]
            for k in bo["k_table"]:
                L.append(f"| {k['K']} | {_f(k['jaccard_min'], 3)} | {k['presence_ok']} | {k['rule_ok']} |")
            L += ["", "K=6 states used for shares and transitions (from this run's state_profiles.csv; the K=6 "
                  "partition is itself a fallback choice that did not meet the stability rule):", ""]
            for s in r["k6_states"]:
                L.append(f"- state {s['state']}: " + ", ".join(f"{k} {_f(v, 2)}" for k, v in s.items() if k != "state"))
            L += ["", "Bout types (medians over bouts; words only from fixed numeric rules):", ""]
            for t in bo["types"]:
                top = sorted(((t[f'share_state_{s}'], s) for s in range(len(r["k6_states"])) if f"share_state_{s}" in t),
                             reverse=True)[:2]
                L.append(f"- Type {t['bout_type']} ({t['bouts']} bouts, {t['birds']} birds): {t['words']}. Duration "
                         f"{_f(t['duration_s'], 0)} s, speed {_f(t['speed_mean'], 1)} m/s, net climb "
                         f"{_f(t['net_climb_m'], 0)} m (first half {_f(t['climb_first_half_m'], 0)}, second half "
                         f"{_f(t['climb_second_half_m'], 0)}), {_f(t['circles_per_min'], 2)} circles/min, glide ratio "
                         f"{_f(t['glide_ratio'], 1)} (n={t['glide_ratio_n']}), flapping share {_f(t['flap_share'], 2)}, "
                         f"main states " + ", ".join(f"{s} ({_f(v, 2)})" for v, s in top) +
                         f", mean start hour {_f(t['start_hour_circmean'], 1)} solar, share of bouts by young-tagged "
                         f"birds {_f(t['bout_share_young_tagged'], 2)}.")
            if bo["group_tests"]:
                L += ["", "Bird-level shares of bout types (TESTED per type, exploratory, no correction):", "",
                      "| type | group | n 1 | n 0 | mean share 1 | mean share 0 | p |", "|---|---|---|---|---|---|---|"]
                for t in bo["group_tests"]:
                    g1 = "female vs male" if t["group"] == "sex" else "young-tagged vs not stated"
                    L.append(f"| {t['type']} | {g1} | {t['n_1']} | {t['n_0']} | {_f(t['mean_share_1'], 3)} | "
                             f"{_f(t['mean_share_0'], 3)} | {_f(t.get('p_perm'), 4)} |")
        elif "note" in bo:
            L.append(f"- {bo['note']}.")
        L += ["", "Glide ratio is ground distance over height lost in descending windows (climb < -0.5 m/s; ground speed, so wind is "
              "included); it is NaN below 20 m of height lost.", ""]
    if cross is not None and len(cross):
        L += ["## Axes compared between runs (descriptive)", "",
              "Spearman between the PCs of the two runs on windows that are flight in both "
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
    ap.add_argument("--state-col", default="state", help="K=2 column defining flight (state or state_hmm)")
    ap.add_argument("--per-bird", type=int, default=3000)
    ap.add_argument("--umap", action="store_true")
    ap.add_argument("--umap-per-bird", type=int, default=300)
    ap.add_argument("--bout-per-bird", type=int, default=300)
    ap.add_argument("--bout-k-over", type=int, default=20)
    ap.add_argument("--bout-max-k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=20)
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
    birds = sorted(p.name for p in a.windows.iterdir() if p.is_dir())
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "word_rules": WORD_RULES,
            "constants": {"BOUT_STEP_S": BOUT_STEP_S, "MIN_BOUT_S": MIN_BOUT_S, "GLIDE_MIN_LOSS_M": GLIDE_MIN_LOSS_M,
                          "GLIDE_MIN_SINK_MS": GLIDE_MIN_SINK_MS,
                          "MIN_CELL": MIN_CELL, "MIN_BIRD_WINDOWS": MIN_BIRD_WINDOWS, "MIN_BIRD_BOUTS": MIN_BIRD_BOUTS}}
    if a.reference.exists():
        birdinfo = reference.load_birds(a.reference, birds).set_index("bird")[["sex", "tagged_young"]]
        meta["reference_note"] = (f"sex f {int((birdinfo['sex'] == 'f').sum())}, m {int((birdinfo['sex'] == 'm').sum())}, "
                                  f"missing {int(birdinfo['sex'].isna().sum())}; young-tagged "
                                  f"{int(birdinfo['tagged_young'].notna().sum())}")
    else:
        birdinfo = pd.DataFrame({"sex": np.nan, "tagged_young": np.nan}, index=pd.Index(birds, name="bird"))
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
    cross = None
    if len(results) == 2:
        cross = cross_run(keys, Zs, names, np.random.default_rng(a.seed))
        cross.to_csv(a.out / "cross_run_pc_spearman.csv", index=False)
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=1, default=_json))
    write_summary(a.out, results, cross, meta)
    log("done")


if __name__ == "__main__":
    main()
