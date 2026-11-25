"""Stage 7: anomalies at two levels (unusual window for this bird, unusual bird among birds), validated with injected
faults. Answers PLAN_2026-11-25 step 7 and review_methods_2026-11 §5.

    python -m aquilacl.anomaly --run <run> [--run <run2>] --windows <dir> --states2 <K=2 states dir per run> \
        --out <new dir> [--reference <csv>] [--k 20] [--n-inject 2000] [--workers N] [--seed 0] [--device cpu]

Level 1 (window within its bird). h is L2-normalised. A window's raw score is the distance to its k-th nearest
neighbour in a reference drawn from the SAME bird, the same recording-format stratum and the same K=2 state, within
about +-45 days (absolute time, not wrapped across years: queries are grouped in 15-day chunks and each chunk's
reference is the windows within 45 days of the chunk centre), capped at ``--ref-cap`` windows. Windows of the
query's own segment and own local day are never neighbours (consecutive windows of a continuous segment are near
copies of each other). With fewer than ``--min-ref`` candidates the reference widens to +-90 days, then to the whole
record; below that the window is unscored (counted per fallback level). The distance is divided by the median
k-th-neighbour distance of reference windows among themselves (local density), then turned into a robust z
(median / 1.4826 MAD) within bird x stratum x state, so strata are comparable. Flagged windows are those above the
99th percentile of z over ALL scored windows (one pooled threshold; a per-stratum threshold would make the format
check below circular). A variant that scores more than 5 points fewer of any format factor's windows than
`format` is disqualified (l1_coverage_<run>.csv). Episodes: runs of >= 3 flagged windows with consecutive k_in_segment in one segment
(possible almost only in continuous k=1 segments; scheduled bursts are mostly one window per segment). Bird-days:
share of a bird-day's windows above the 99th percentile of z of their own bird x stratum x state (expected 1%).

Format neutrality is a real test because the threshold is pooled: the flagged set's share of each format factor
(short window, continuous mode, GPS, single-window segment, segment edge, 40- and 64-sample windows) is compared
with its share among scored windows. Several stratifications (VARIANTS) are scored in the same pass and the one
with the smallest worst format ratio is used for episodes, bird-days and injection; all are reported
(l1_variants_<run>.csv). This is the documented iteration of the stratification.

Level 2 (bird among birds). Per run, a month-balanced sample of each bird's windows per base format stratum x K=2
state is projected on 20 principal components of the bird-balanced pool. Per bird and cell: unbiased MMD^2 with an
RBF kernel (median-heuristic bandwidth) between the bird and a leave-one-bird-out pool with every other bird
weighted equally. Null: random pool samples of the bird's size against the same reference. Identity is strong, so
nearly every bird differs from the pool; birds are ranked by a robust z of MMD^2 across birds within the cell, and
the bird score is the mean of its cell z. Month change: MMD^2 between consecutive available months of a bird and
cell, minus the within-month floor (MMD^2 between odd and even local days of each month); robust z across all
month pairs of the cell. Bird-level nuisance relations: Spearman rho, p from permuting values between birds.

Validation (descriptive). Real windows are rebuilt from the sensor frame exactly as dataset.load does (common
format of the run's config.json, the run's stored rotation R from orientation.json, mask; GPS scaled with the run's
masked channels hidden) and re-encoded with the run's latest checkpoint (train.encode). Faults: one sensor axis
stuck at its first value; a 90 deg mounting slip about a random horizontal axis of the gravity-aligned frame (and a
90 deg yaw slip, which r6 was trained to ignore); amplitude halved; GPS speed set to 0 on flapping windows with GPS;
a window of another bird (same base stratum, same K=2 state of the first run) scored in the host's record, once
encoded with its own bird's R (transplant_own_R: identity) and once with the host's R (transplant_host_R: a record
mix-up, which also carries the mounting difference). Each injected window keeps its host window's bird, stratum, K=2 state, segment and
day. Recall is the share above the
pooled 99th-percentile threshold; the false-positive rate is that of the re-encoded untouched copies.

Privacy: latitude and longitude are never read. Blind pages show relative seconds and the month only, and no
bird id; their key is a separate CSV. Bird ids are never printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from . import dataset  # noqa: E402
from .windows import GPS_CHANNELS  # noqa: E402

FULL_SAMPLES = 190
CHUNK_DAYS = 15
SEASON_DAYS = (45, 90, None)  # reference half-width per fallback level; None = whole record
NIGHT_SUN = -6.0
MIN_Z_GROUP = 30  # scored windows needed in a bird x stratum x state for median/MAD
SCALE_N = 200  # reference windows used for the local-density scale
FLAG_Q = 99.0
MIN_EPISODE = 3
MIN_BIRD_DAY = 20
L2_PCS = 20
L2_CAP = 80  # windows per bird x cell x year-month kept for level 2
L2_BIRD_N = 300  # windows per bird and cell compared with the pool
L2_MIN_BIRD = 50
L2_MIN_MONTH = 40
NEUTRAL = 1.5  # format ratios within [1/NEUTRAL, NEUTRAL] count as roughly neutral
META_COLS = ["t_start", "segment", "k_in_segment", "segment_windows", "sensitivity_k", "acc_samples", "gps_fixes",
             "sun_elev_deg", "local_solar_hour"]
FEAT_COLS = ["label_flapping", "speed_mean"]
VARIANTS = ("format", "format_seglen", "format_seglen_accbin")
BASE_NAMES = {c: f"{'full' if c & 4 else 'short'}_k{(c >> 1) & 1}_{'gps' if c & 1 else 'nogps'}" for c in range(8)}
FAULTS = ("stuck_axis", "tilt90", "yaw90", "amplitude_half", "gps_speed_zero", "transplant_own_R", "transplant_host_R")
MAX_COVERAGE_DROP = 0.05  # a variant may score at most 5 points fewer of any format factor's windows than "format"
SPEED_CH = ("ground-speed", "speed_track_ms", "step_m")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


# ---------------------------------------------------------------- strata


def local_day(t: pd.Series, solar_hour: np.ndarray) -> np.ndarray:
    """Integer local solar day: UTC time shifted by (local solar hour - UTC hour), wrapped to +-12 h."""
    t = pd.to_datetime(t, utc=True)
    utc_h = (t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600).to_numpy(float)
    off = (np.asarray(solar_hour, float) - utc_h + 12) % 24 - 12
    off = np.nan_to_num(off)
    loc = t + pd.to_timedelta(off, unit="h")
    return ((loc - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(days=1)).to_numpy(np.int64)


def base_stratum(acc_samples, k, gps_fixes) -> np.ndarray:
    """0..7 = full * 4 + continuous (k=1) * 2 + GPS present."""
    return ((np.asarray(acc_samples) >= FULL_SAMPLES).astype(int) * 4 + (np.asarray(k) == 1).astype(int) * 2
            + (np.asarray(gps_fixes) > 0).astype(int))


def strata(f: pd.DataFrame, variant: str, flight: np.ndarray) -> np.ndarray:
    """Stratum code per window for a VARIANTS entry, always including the K=2 state (``flight``)."""
    code = base_stratum(f["acc_samples"], f["sensitivity_k"], f["gps_fixes"]).astype(np.int64)
    if "seglen" in variant:
        code = code * 3 + np.digitize(f["segment_windows"].to_numpy(), [2, 10])
    if "accbin" in variant:
        a = f["acc_samples"].to_numpy()
        code = code * 4 + np.where(a >= FULL_SAMPLES, 0, np.where(a == 40, 1, np.where(a == 64, 2, 3)))
    return code * 2 + np.asarray(flight, bool).astype(np.int64)


# ---------------------------------------------------------------- level 1 kernels


def kth_dist(Q, R, qday, rday, qseg, rseg, k: int, block: int = 4096) -> np.ndarray:
    """Distance (L2-normalised vectors) from each row of Q to its k-th nearest row of R, excluding rows of R with the
    same day or the same segment as the query; NaN where fewer than k remain."""
    out = np.full(len(Q), np.nan, np.float32)
    for s in range(0, len(Q), block):
        sl = slice(s, s + block)
        sim = Q[sl] @ R.T
        bad = (qday[sl, None] == rday[None]) | (qseg[sl, None] == rseg[None])
        sim[bad] = -np.inf
        n_ok = (~bad).sum(1)
        kk = min(k, R.shape[0])
        top = -np.partition(-sim, kk - 1, axis=1)[:, kk - 1]
        d = np.sqrt(np.clip(2 - 2 * top, 0, None))
        d[n_ok < k] = np.nan
        out[sl] = d
    return out


def score_groups(H, day, seg, key, k, cap, min_ref, seed, extra=None):
    """Density-normalised k-NN score of every window within its key, by 15-day chunk (see module docstring).

    ``extra`` = (Hq, idx) scores further vectors Hq as if they were the windows ``idx`` (same key, day, segment,
    reference). Returns (score, level, extra_score): level is the fallback (0: +-45 d, 1: +-90 d, 2: whole record,
    -1: unscored)."""
    n = len(H)
    score, level = np.full(n, np.nan, np.float32), np.full(n, -1, np.int8)
    ex_score = np.full(0 if extra is None else len(extra[1]), np.nan, np.float32)
    ex_idx = np.zeros(0, int) if extra is None else np.asarray(extra[1])
    chunk = np.floor_divide(day, CHUNK_DAYS)
    for g in np.unique(key):
        ig = np.flatnonzero(key == g)
        for c in np.unique(chunk[ig]):
            q = ig[chunk[ig] == c]
            centre = c * CHUNK_DAYS + CHUNK_DAYS / 2
            for lev, half in enumerate(SEASON_DAYS):
                cand = ig if half is None else ig[np.abs(day[ig] - centre) <= half]
                if len(cand) >= min_ref:
                    break
            else:
                continue
            rng = np.random.default_rng(seed_of(seed, g, c))
            ref = np.sort(rng.choice(cand, min(cap, len(cand)), replace=False))
            sc_rows = ref if len(ref) <= SCALE_N else np.sort(rng.choice(ref, SCALE_N, replace=False))
            scale = np.nanmedian(kth_dist(H[sc_rows], H[ref], day[sc_rows], day[ref], seg[sc_rows], seg[ref], k))
            if not np.isfinite(scale) or scale <= 0:
                continue
            score[q] = kth_dist(H[q], H[ref], day[q], day[ref], seg[q], seg[ref], k) / scale
            level[q] = lev
            j = np.flatnonzero(np.isin(ex_idx, q))
            if len(j):
                hi = ex_idx[j]
                ex_score[j] = kth_dist(extra[0][j], H[ref], day[hi], day[ref], seg[hi], seg[ref], k) / scale
    return score, level, ex_score


def robust_z(x: np.ndarray, groups: np.ndarray, min_n: int = MIN_Z_GROUP):
    """(x - median) / (1.4826 MAD) within groups; NaN in groups with < min_n finite values or MAD 0.
    Returns z and {group: (median, scale)}."""
    z = np.full(len(x), np.nan, np.float32)
    par = {}
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        v = x[i]
        ok = np.isfinite(v)
        if ok.sum() < min_n:
            continue
        med = float(np.median(v[ok]))
        s = 1.4826 * float(np.median(np.abs(v[ok] - med)))
        if s <= 0:
            continue
        par[g] = (med, s)
        z[i] = (v - med) / s
    return z, par


def apply_z(x: np.ndarray, groups: np.ndarray, par: dict) -> np.ndarray:
    return np.array([(v - par[g][0]) / par[g][1] if g in par else np.nan for v, g in zip(x, groups)], np.float32)


def group_quantile_flag(z: np.ndarray, groups: np.ndarray, q: float = FLAG_Q) -> np.ndarray:
    out = np.zeros(len(z), bool)
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        ok = np.isfinite(z[i])
        if ok.sum():
            out[i] = z[i] > np.percentile(z[i][ok], q)
    return out


def flag_runs(group: np.ndarray, kin: np.ndarray, flag: np.ndarray, min_len: int = MIN_EPISODE) -> list[tuple[int, int]]:
    """(first, last) positions of runs of flagged rows with the same group and k stepping by exactly 1 (rows must be
    sorted by group, then k); only runs of at least ``min_len``."""
    n = len(flag)
    if n == 0:
        return []
    cont = np.r_[False, (group[1:] == group[:-1]) & (np.diff(kin) == 1) & flag[1:] & flag[:-1]]
    out, start = [], None
    for i in range(n):
        if flag[i] and not cont[i]:
            start = i
        if flag[i] and (i == n - 1 or not cont[i + 1]):
            if start is not None and i - start + 1 >= min_len:
                out.append((start, i))
            start = None
    return out


FACTORS = {  # name: (mask function on the scored frame, is a format factor)
    f"short window (acc_samples < {FULL_SAMPLES})": (lambda d: d["acc_samples"] < FULL_SAMPLES, True),
    "40-sample window": (lambda d: d["acc_samples"] == 40, True),
    "64-sample window": (lambda d: d["acc_samples"] == 64, True),
    "continuous mode (k=1)": (lambda d: d["sensitivity_k"] == 1, True),
    "GPS present": (lambda d: d["gps_fixes"] > 0, True),
    "single-window segment": (lambda d: d["segment_windows"] <= 1, True),
    "segment edge (first/last of >1)": (lambda d: (d["segment_windows"] > 1) & ((d["k_in_segment"] == 0) | (
        d["k_in_segment"] == d["segment_windows"] - 1)), True),
    "K=2 flight state": (lambda d: d["flight"], False),
    "night (sun < -6 deg)": (lambda d: d["sun_elev_deg"] < NIGHT_SUN, False),
}


def factor_table(d: pd.DataFrame, flag: np.ndarray) -> pd.DataFrame:
    """Share of flagged windows with each factor against the share among scored windows (ratio 1 = neutral)."""
    fl = d[flag]
    facs = {n: f for n, (f, _) in FACTORS.items()}
    facs.update({f"month {m}": (lambda x, m=i + 1: x["month"] == m) for i, m in enumerate(MONTHS)})
    rows = []
    for name, fn in facs.items():
        m_all, m_fl = np.asarray(fn(d), bool), np.asarray(fn(fl), bool)
        base = float(m_all.mean()) if len(d) else np.nan
        share = float(m_fl.mean()) if len(fl) else np.nan
        rows.append({"factor": name, "format_factor": FACTORS.get(name, (None, False))[1], "n_scored": len(d),
                     "n_flagged": len(fl), "n_flagged_with": int(m_fl.sum()), "share_flagged": share,
                     "share_scored": base, "ratio": share / base if base > 0 else np.nan,
                     "birds_flagged_with": int(fl.loc[m_fl, "bird"].nunique()) if len(fl) else 0})
    return pd.DataFrame(rows)


def coverage_table(d: pd.DataFrame, Z: dict) -> pd.DataFrame:
    """Share of each format factor's windows that received a z, per variant (a variant must not look neutral by
    leaving the awkward windows unscored)."""
    rows = []
    for name, (fn, is_fmt) in FACTORS.items():
        if not is_fmt:
            continue
        m = np.asarray(fn(d), bool)
        rows.append({"factor": name, "n_windows": int(m.sum()),
                     **{v: float(np.isfinite(z[m]).mean()) if m.any() else np.nan for v, z in Z.items()}})
    return pd.DataFrame(rows)


def worst_format_ratio(ft: pd.DataFrame, min_base: float = 1e-3) -> float:
    """max |log ratio| over format factors present in at least ``min_base`` of scored windows, as a ratio >= 1."""
    r = ft[ft["format_factor"] & (ft["share_scored"] >= min_base)]["ratio"].astype(float)
    r = r[np.isfinite(r)]
    if not len(r):
        return np.nan
    return float(np.exp(np.abs(np.log(np.maximum(r, 1e-6))).max()))


# ---------------------------------------------------------------- level 2 kernels


def sqdist(X, Y):
    return np.maximum((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None] - 2 * X @ Y.T, 0)


def median_gamma(X: np.ndarray, rng, n: int = 2000) -> float:
    s = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    d = sqdist(s, s)[np.triu_indices(len(s), 1)]
    return 1.0 / max(float(np.median(d)), 1e-12)


def mmd2(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD^2 with an RBF kernel exp(-gamma |x-y|^2)."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return np.nan
    kxx, kyy, kxy = np.exp(-gamma * sqdist(X, X)), np.exp(-gamma * sqdist(Y, Y)), np.exp(-gamma * sqdist(X, Y))
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                 - 2 * kxy.mean())


def pool_draw(birds: np.ndarray, exclude, n: int, rng) -> np.ndarray:
    """n rows of the pool, every bird except ``exclude`` weighted equally."""
    ok = birds != exclude
    if not ok.any():
        return np.zeros(0, int)
    _, inv, cnt = np.unique(birds[ok], return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    idx = np.flatnonzero(ok)
    return rng.choice(idx, min(n, len(idx)), replace=False, p=w / w.sum())


def bird_vs_pool(X: np.ndarray, birds: np.ndarray, b, gamma: float, n_null: int, rng, n: int = L2_BIRD_N) -> dict:
    """MMD^2 of bird b's rows against a leave-one-out, bird-balanced pool sample (2n), with a null from n-row pool
    samples against the same reference."""
    own = np.flatnonzero(birds == b)
    if len(own) < L2_MIN_BIRD:
        return {}
    own = rng.choice(own, min(n, len(own)), replace=False)
    ref = pool_draw(birds, b, 2 * len(own), rng)
    rest = np.setdiff1d(np.flatnonzero(birds != b), ref)
    if len(ref) < L2_MIN_BIRD or len(rest) < len(own):
        return {}
    obs = mmd2(X[own], X[ref], gamma)
    null = np.array([mmd2(X[pool_draw(np.where(np.isin(np.arange(len(birds)), rest), birds, b), b, len(own), rng)],
                          X[ref], gamma) for _ in range(n_null)]) if n_null else np.zeros(0)
    out = {"n_bird": len(own), "n_ref": len(ref), "mmd2": obs}
    if n_null:
        sd = float(null.std(ddof=1)) if n_null > 1 else np.nan
        out.update(null_mean=float(null.mean()), null_sd=sd, p_null=float((1 + (null >= obs).sum()) / (n_null + 1)),
                   std_effect=(obs - null.mean()) / sd if sd and sd > 0 else np.nan)
    return out


def month_change(X: np.ndarray, ym: np.ndarray, day: np.ndarray, gamma: float, min_n: int = L2_MIN_MONTH) -> list[dict]:
    """MMD^2 between consecutive available months minus the mean within-month floor (odd vs even local days)."""
    months = [m for m in sorted(set(ym)) if (ym == m).sum() >= min_n]

    def floor(m):
        i = ym == m
        a, b = X[i & (day % 2 == 0)], X[i & (day % 2 == 1)]
        return mmd2(a, b, gamma) if min(len(a), len(b)) >= 5 else np.nan

    fl = {m: floor(m) for m in months}
    out = []
    for a, b in zip(months[:-1], months[1:]):
        d = mmd2(X[ym == a], X[ym == b], gamma)
        f = np.nanmean([fl[a], fl[b]]) if np.isfinite([fl[a], fl[b]]).any() else np.nan
        ya, ma = map(int, a.split("-"))
        yb, mb = map(int, b.split("-"))
        out.append({"month_from": a, "month_to": b, "gap_months": (yb - ya) * 12 + mb - ma, "mmd2": d, "floor": f,
                    "excess": d - f if np.isfinite(f) else np.nan})
    return out


def perm_spearman(x: np.ndarray, y: np.ndarray, n_perm: int, rng) -> tuple[float, float, int]:
    """Spearman rho between bird-level x and y, p from permuting y between birds (two-sided)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, int(len(x))
    rho = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    return float(rho), float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (n_perm + 1)), int(len(x))


# ---------------------------------------------------------------- model input and faults


def acc_input(a: np.ndarray, am: np.ndarray, R: np.ndarray, step_g: float, clip_g: float) -> np.ndarray:
    """Sensor-frame acceleration (N, T, 3) -> model input, as dataset.load: common format per sensor axis, then the
    per-row rotation R (N, 3, 3), times the mask."""
    x = dataset.common_format(a, am, step_g, clip_g) if (step_g > 0 or clip_g > 0) else a.astype(np.float16)
    out = np.empty_like(x)
    Rs, inv = np.unique(np.asarray(R, np.float32).reshape(len(R), 9), axis=0, return_inverse=True)
    for j, Rj in enumerate(Rs):  # same matmul as dataset.load (bit-identical float16 result)
        i = np.flatnonzero(inv.ravel() == j)
        out[i] = (x[i].astype(np.float32) @ Rj.reshape(3, 3).T).astype(np.float16) * am[i][:, :, None]
    return out


def gps_input(graw: np.ndarray, gm: np.ndarray, mask_channels=()) -> tuple[np.ndarray, np.ndarray]:
    """Raw gps.npy rows -> model input, as dataset.load (fixed scale, log1p accuracy, hidden channels, clip, mask)."""
    scale = np.array([dataset.GPS_SCALE[c] for c in GPS_CHANNELS], np.float32)
    g = np.asarray(graw, np.float32) / scale
    hi = GPS_CHANNELS.index("eobs:horizontal-accuracy-estimate")
    g[:, :, hi] = np.log1p(np.asarray(graw, np.float32)[:, :, hi])
    gm = np.asarray(gm, bool).copy()
    for c in mask_channels:
        gm[:, :, GPS_CHANNELS.index(c)] = False
    return (np.clip(g, -dataset.GPS_CLIP, dataset.GPS_CLIP) * gm).astype(np.float16), gm


def rot_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def fault_stuck(a: np.ndarray, am: np.ndarray, rng) -> np.ndarray:
    """One random sensor axis held at its first present value."""
    a = a.copy()
    for i in range(len(a)):
        p = np.flatnonzero(am[i])
        if len(p):
            ax = rng.integers(3)
            a[i, p, ax] = a[i, p[0], ax]
    return a


def fault_rotate(a: np.ndarray, R: np.ndarray, rng, deg: float = 90.0, yaw: bool = False) -> np.ndarray:
    """Mounting slip in the sensor frame that equals a rotation Q of the gravity-aligned frame (about a random
    horizontal axis, or about the vertical with ``yaw``): a' = R^T Q R a, so the model sees Q R a."""
    out = np.empty_like(a, dtype=np.float32)
    for i in range(len(a)):
        if yaw:
            Q = rot_axis([0, 0, 1], deg)
        else:
            t = rng.uniform(0, 2 * np.pi)
            Q = rot_axis([np.cos(t), np.sin(t), 0], deg)
        out[i] = a[i].astype(np.float32) @ (R[i].T @ Q @ R[i]).T
    return out


def fault_gps_zero(graw: np.ndarray, gm: np.ndarray) -> np.ndarray:
    g = np.asarray(graw, np.float32).copy()
    for c in SPEED_CH:
        j = GPS_CHANNELS.index(c)
        g[:, :, j] = np.where(gm[:, :, j], 0.0, g[:, :, j])
    return g


# ---------------------------------------------------------------- loading


def _cols(path: Path, want: list[str]) -> pd.DataFrame:
    have = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in want if c in have])


def load_bird_frame(d: Path) -> pd.DataFrame | None:
    m = _cols(d / "meta.parquet", META_COLS)
    if len(m) == 0:
        return None
    f = _cols(d / "features.parquet", FEAT_COLS)
    df = pd.concat([m.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    for c in FEAT_COLS:
        if c not in df:
            df[c] = np.nan
    if "segment_windows" not in df:
        df["segment_windows"] = df.groupby("segment")["segment"].transform("size")
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True)
    df["bird"] = d.name
    df["row"] = np.arange(len(df))
    df["day"] = local_day(df["t_start"], df["local_solar_hour"].to_numpy())
    df["month"] = df["t_start"].dt.month
    df["ym"] = df["t_start"].dt.strftime("%Y-%m")
    df["base"] = base_stratum(df["acc_samples"], df["sensitivity_k"], df["gps_fixes"])
    return df


def flight_state(st: pd.DataFrame, speed: pd.Series) -> int:
    """The K=2 state with the higher median GPS speed (not assumed to be state 1)."""
    med = pd.Series(speed.to_numpy(), index=st.index).groupby(st.to_numpy()).median()
    return int(med.idxmax())


def norm_rows(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, np.float32)
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- per-bird pass


def bird_pass(f: pd.DataFrame, runs: list[dict], k: int, cap: int, min_ref: int, seed: int, extra: dict) -> dict:
    """Level-1 scores of one bird for every run and variant, extra (injected) queries, level-2 samples."""
    b = f["bird"].iat[0]
    day, seg = f["day"].to_numpy(), f["segment"].to_numpy()
    res = {"bird": b, "runs": {}}
    for r in runs:
        H = norm_rows(np.load(Path(r["path"]) / "embeddings" / f"{b}.npz")["h"])
        if len(H) != len(f):
            raise ValueError("embedding rows differ from meta rows")
        fl = f[f"flight_{r['name']}"].to_numpy()
        ex = extra.get(r["name"])
        rr = {"z": {}, "level": {}, "flag_bird": {}, "extra_z": {}}
        for v in VARIANTS:
            key = strata(f, v, fl)
            sc, lev, exs = score_groups(H, day, seg, key, k, cap, min_ref, seed_of(seed, b, r["name"], v),
                                        None if ex is None else (ex["H"], ex["idx"]))
            z, par = robust_z(sc, key)
            rr["z"][v], rr["level"][v] = z, lev
            rr["flag_bird"][v] = group_quantile_flag(z, key)
            if ex is not None:
                rr["extra_z"][v] = apply_z(exs, key[ex["idx"]], par)
        # level-2 sample: up to L2_CAP windows per base stratum x state x year-month
        rng = np.random.default_rng(seed_of(seed, b, r["name"], "l2"))
        cell = f["base"].to_numpy() * 2 + fl.astype(int)
        grp = pd.Series(np.arange(len(f))).groupby([cell, f["ym"].to_numpy()]).indices
        pick = np.sort(np.concatenate([rng.choice(ix, min(L2_CAP, len(ix)), replace=False) for ix in grp.values()]))
        rr["l2_rows"], rr["l2_h"] = pick, H[pick].astype(np.float16)
        res["runs"][r["name"]] = rr
    return res


# ---------------------------------------------------------------- injection


class Encoder:
    """The run's model and input settings (config.json, orientation.json, latest checkpoint)."""

    def __init__(self, run: Path, device: str):
        from .evaluate import _latest_model

        cfg = json.loads((run / "config.json").read_text())
        self.step_g, self.clip_g = float(cfg.get("step_g", 0) or 0), float(cfg.get("clip_g", 0) or 0)
        self.mask_ch = tuple(cfg.get("mask_gps_channels", []))
        self.R = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3)
                  for o in json.loads((run / "orientation.json").read_text())}
        self.model, self.dev = _latest_model(run, device), device

    def h(self, a, am, R, graw, gm) -> np.ndarray:
        from .train import encode

        x = acc_input(a, am, R, self.step_g, self.clip_g)
        g, gmm = gps_input(graw, gm, self.mask_ch)
        return norm_rows(encode(self.model, x, am, g, gmm, self.dev)[0])


def read_rows(windows: Path, birds: np.ndarray, rows: np.ndarray):
    """Sensor-frame acc, acc mask, raw GPS and GPS mask of (bird, row) pairs, in order."""
    a = np.zeros((len(rows), 200, 3), np.float32)
    am = np.zeros((len(rows), 200), bool)
    g = np.zeros((len(rows), 10, len(GPS_CHANNELS)), np.float32)
    gm = np.zeros(g.shape, bool)
    for b in np.unique(birds):
        i = np.flatnonzero(birds == b)
        d = windows / str(b)
        r = rows[i]
        a[i] = np.load(d / "acc.npy", mmap_mode="r")[r]
        am[i] = np.load(d / "acc_mask.npy", mmap_mode="r")[r]
        g[i] = np.load(d / "gps.npy", mmap_mode="r")[r]
        gm[i] = np.load(d / "gps_mask.npy", mmap_mode="r")[r]
    return a, am, g, gm


def pick_sources(F: pd.DataFrame, n: int, rng, state_col: str) -> dict:
    """Bird-balanced source windows (positions in F): acc faults, GPS failure (flapping with GPS), transplant
    donor/host pairs (different birds, same base stratum and same K=2 state in ``state_col``)."""
    def balanced(mask):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return np.zeros(0, int)
        _, inv, cnt = np.unique(F["bird"].to_numpy()[idx], return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]
        return np.sort(rng.choice(idx, min(n, len(idx)), replace=False, p=w / w.sum()))

    acc_src = balanced(F["acc_samples"].to_numpy() >= 20)
    gps_src = balanced((F["label_flapping"].to_numpy() == 1) & (F["gps_fixes"].to_numpy() > 0))
    donors = balanced(F["acc_samples"].to_numpy() >= 20)
    hosts = np.full(len(donors), -1)
    match = F["base"].to_numpy() * 2 + F[state_col].to_numpy().astype(int)
    by_match = {c: np.flatnonzero(match == c) for c in np.unique(match)}
    bird = F["bird"].to_numpy()
    for j, d in enumerate(donors):
        pool = by_match[match[d]]
        for _ in range(20):
            h = rng.choice(pool)
            if bird[h] != bird[d]:
                hosts[j] = h
                break
    ok = hosts >= 0
    return {"acc": acc_src, "gps": gps_src, "donor": donors[ok], "host": hosts[ok]}


def build_injections(F: pd.DataFrame, windows: Path, run: Path, src: dict, device: str, rng) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Encode clean and faulty versions. Returns (query table: fault, host position in F, source position), H, and
    the reproduction check against stored embeddings."""
    enc = Encoder(run, device)
    bird = F["bird"].to_numpy()
    rows = F["row"].to_numpy()
    tables, Hs = [], []

    def add(fault, host, source, H):
        tables.append(pd.DataFrame({"fault": fault, "host": host, "source": source}))
        Hs.append(H)

    s = src["acc"]
    a, am, g, gm = read_rows(windows, bird[s], rows[s])
    R = np.stack([enc.R[b] for b in bird[s]])
    h_clean = enc.h(a, am, R, g, gm)
    add("clean", s, s, h_clean)
    add("stuck_axis", s, s, enc.h(fault_stuck(a, am, rng), am, R, g, gm))
    add("tilt90", s, s, enc.h(fault_rotate(a, R, rng), am, R, g, gm))
    add("yaw90", s, s, enc.h(fault_rotate(a, R, rng, yaw=True), am, R, g, gm))
    add("amplitude_half", s, s, enc.h(a * 0.5, am, R, g, gm))
    stored = np.zeros_like(h_clean)
    for b in np.unique(bird[s]):
        i = np.flatnonzero(bird[s] == b)
        stored[i] = norm_rows(np.load(run / "embeddings" / f"{b}.npz")["h"][rows[s][i]])
    cos = (stored * h_clean).sum(1)
    repro = {"n": int(len(s)), "min_cosine_stored_vs_reencoded": float(cos.min()),
             "median_cosine_stored_vs_reencoded": float(np.median(cos)), "device": device}
    s = src["gps"]
    if len(s):
        a, am, g, gm = read_rows(windows, bird[s], rows[s])
        R = np.stack([enc.R[b] for b in bird[s]])
        add("clean_gps_source", s, s, enc.h(a, am, R, g, gm))
        add("gps_speed_zero", s, s, enc.h(a, am, R, fault_gps_zero(g, gm), gm))
    d, h = src["donor"], src["host"]
    if len(d):
        a, am, g, gm = read_rows(windows, bird[d], rows[d])
        add("transplant_own_R", h, d, enc.h(a, am, np.stack([enc.R[b] for b in bird[d]]), g, gm))
        add("transplant_host_R", h, d, enc.h(a, am, np.stack([enc.R[b] for b in bird[h]]), g, gm))
    return pd.concat(tables, ignore_index=True), np.concatenate(Hs), repro


def recall_table(q: pd.DataFrame, thr: float, F: pd.DataFrame, rng, n_boot: int = 1000) -> pd.DataFrame:
    """Share of scored injected windows above thr per fault, 95% bootstrap interval over host birds; the clean
    flag rate of the same source windows for comparison."""
    rows = []
    hb = F["bird"].to_numpy()[q["host"].to_numpy()]
    q = q.assign(host_bird=hb, hit=q["z"] > thr, scored=np.isfinite(q["z"]))
    clean = q[q["fault"] == "clean"].set_index("source")["hit"]
    clean_g = q[q["fault"] == "clean_gps_source"].set_index("source")["hit"]
    for fault, g in q.groupby("fault", sort=False):
        g = g[g["scored"]]
        if not len(g):
            continue
        per = g.groupby("host_bird")["hit"].agg(["sum", "size"])
        bi = rng.integers(0, len(per), (n_boot, len(per)))
        boot = per["sum"].to_numpy()[bi].sum(1) / per["size"].to_numpy()[bi].sum(1)
        base = clean_g if fault in ("gps_speed_zero", "clean_gps_source") else clean
        same = base.reindex(g["source"]).dropna() if not fault.startswith("transplant") else pd.Series(dtype=float)
        rows.append({"fault": fault, "n_injected": int((q["fault"] == fault).sum()), "n_scored": len(g),
                     "n_birds": int(g["host_bird"].nunique()), "flag_rate": float(g["hit"].mean()),
                     "ci95_lo": float(np.percentile(boot, 2.5)), "ci95_hi": float(np.percentile(boot, 97.5)),
                     "clean_flag_rate_same_sources": float(same.mean()) if len(same) else np.nan,
                     "role": "false-positive rate" if fault.startswith("clean") else "recall"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def fig_factors(var_tables: dict, run: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    names = [n for n in FACTORS]
    y = np.arange(len(names))
    for j, (v, ft) in enumerate(var_tables.items()):
        r = ft.set_index("factor").reindex(names)["ratio"].to_numpy(float)
        ax.barh(y + (j - 1) * 0.27, r, height=0.27, label=v)
    ax.axvline(1, color="k", lw=0.8)
    for x in (1 / NEUTRAL, NEUTRAL):
        ax.axvline(x, color="grey", ls=":", lw=0.8)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("share among flagged / share among scored (1 = neutral)")
    ax.set_title(f"{run}: flagged windows by factor, per stratification", fontsize=10)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_recall(rec: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    runs = list(dict.fromkeys(rec["run"]))
    faults = list(dict.fromkeys(rec["fault"]))
    x = np.arange(len(faults))
    w = 0.8 / max(len(runs), 1)
    for j, r in enumerate(runs):
        g = rec[rec["run"] == r].set_index("fault").reindex(faults)
        ax.bar(x + j * w, g["flag_rate"], w, yerr=[g["flag_rate"] - g["ci95_lo"], g["ci95_hi"] - g["flag_rate"]],
               label=r, capsize=2)
    ax.axhline(0.01, color="grey", ls=":", lw=0.8)
    ax.set_xticks(x + w * (len(runs) - 1) / 2, faults, rotation=20, fontsize=8)
    ax.set_ylabel("share above the pooled 99th percentile")
    ax.set_title("Injected faults: recall (clean rows: false-positive rate)", fontsize=10)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_l2(birds: pd.DataFrame, run: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, c, lab in ((axes[0], "cont_share", "share of continuous-mode windows"),
                       (axes[1], "track_days", "track days")):
        ax.scatter(birds[c], birds[f"l2_score_{run}"], s=12)
        ax.set_xlabel(lab)
        ax.set_ylabel("level-2 bird score (mean robust z)")
    fig.suptitle(f"{run}: unusual birds against recording nuisance (one dot per bird)", fontsize=10)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def blind_page(episodes: list[dict], path: Path) -> plt.Figure:
    """One page of blind episode plots. Each entry: rank, month, state, n_flagged, t_rel (s, per window), acc
    (list of (200, 3) arrays, NaN where absent), gps_speed / gps_climb (list of (10,) arrays), flagged (bools).
    No bird id, date or score is drawn."""
    fig, axes = plt.subplots(len(episodes), 2, figsize=(14, 2.6 * len(episodes) + 0.4), squeeze=False,
                             layout="constrained")
    for (ax_a, ax_g), e in zip(axes, episodes):
        for t0, a, fl in zip(e["t_rel"], e["acc"], e["flagged"]):
            t = t0 + np.arange(a.shape[0]) / 20.0
            for j, (c, lab) in enumerate(zip(("tab:red", "tab:green", "tab:blue"), ("x", "y", "z"))):
                ax_a.plot(t, a[:, j], color=c, lw=0.5, label=lab if t0 == e["t_rel"][0] else None)
            if fl:
                ax_a.axvspan(t0, t0 + 10, color="gold", alpha=0.25, lw=0)
        for t0, sp, cl in zip(e["t_rel"], e["gps_speed"], e["gps_climb"]):
            t = t0 + np.arange(len(sp))
            ax_g.plot(t, sp, "k.-", ms=2, lw=0.6)
            ax_g.plot(t, cl, ".-", color="tab:purple", ms=2, lw=0.6)
        ax_a.set_title(f"Episode {e['rank']}: month {e['month']}, K=2 state {e['state']}, "
                       f"{e['n_flagged']} flagged windows (shaded)", fontsize=8)
        ax_a.set_ylabel("acc (g), gravity-aligned")
        ax_g.set_ylabel("GPS speed (black), climb (purple), m/s", fontsize=7)
        ax_a.legend(fontsize=6, ncol=3, loc="upper right")
    axes[-1, 0].set_xlabel("seconds from first shown window")
    axes[-1, 1].set_xlabel("seconds from first shown window")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return fig


def episode_payload(F: pd.DataFrame, ep: pd.Series, windows: Path, R: np.ndarray, rank: int, pad: int = 2) -> dict:
    b, sg = ep["bird"], ep["segment"]
    seg = F[(F["bird"] == b) & (F["segment"] == sg)].sort_values("k_in_segment")
    seg = seg[(seg["k_in_segment"] >= ep["k_first"] - pad) & (seg["k_in_segment"] <= ep["k_last"] + pad)]
    a, am, g, gm = read_rows(windows, seg["bird"].to_numpy(), seg["row"].to_numpy())
    a = a @ R.T
    a[~am] = np.nan
    si, ci = GPS_CHANNELS.index("ground-speed"), GPS_CHANNELS.index("climb_ms")
    t0 = seg["t_start"].iat[0]
    kk = seg["k_in_segment"].to_numpy()
    return {"rank": rank, "month": MONTHS[int(ep["month"]) - 1], "state": ep["state_name"],
            "n_flagged": int(ep["n_windows"]), "t_rel": (seg["t_start"] - t0).dt.total_seconds().tolist(),
            "acc": list(a), "gps_speed": [np.where(gm[i, :, si], g[i, :, si], np.nan) for i in range(len(seg))],
            "gps_climb": [np.where(gm[i, :, ci], g[i, :, ci], np.nan) for i in range(len(seg))],
            "flagged": list((kk >= ep["k_first"]) & (kk <= ep["k_last"]))}


# ---------------------------------------------------------------- driver


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}g}"


def run(runs: list[Path], states2: list[Path], windows: Path, out: Path, reference_csv: Path | None = None, k: int = 20,
        n_inject: int = 2000, workers: int = 1, seed: int = 0, device: str = "cpu", ref_cap: int = 2000,
        min_ref: int = 200, n_null: int = 50, n_perm: int = 2000, top_episodes: int = 20) -> dict:
    t0 = time.time()
    if len(runs) != len(states2):
        raise SystemExit("give one --states2 per --run, in the same order")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"run directory names must differ: {names}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dirs = [d for d in sorted(p for p in windows.iterdir() if p.is_dir())
            if (d / "meta.parquet").exists() and pq.ParquetFile(d / "meta.parquet").metadata.num_rows > 0]
    log(f"{len(dirs)} birds with windows")
    F = pd.concat(Parallel(n_jobs=workers)(delayed(load_bird_frame)(d) for d in dirs), ignore_index=True)
    meta = {"runs": names, "states2": [str(s) for s in states2], "windows": str(windows), "k": k, "ref_cap": ref_cap,
            "min_ref": min_ref, "n_inject": n_inject, "seed": seed, "device": device, "n_null": n_null,
            "n_perm": n_perm, "variants": list(VARIANTS), "chunk_days": CHUNK_DAYS, "season_days": list(SEASON_DAYS),
            "flight_state": {}, "n_windows": len(F), "n_birds": int(F["bird"].nunique())}
    rinfo = []
    for r, s in zip(runs, states2):
        st = pd.read_parquet(s / "states.parquet", columns=["bird", "row", "state_hmm"])
        st["bird"] = st["bird"].astype(str)
        st = F[["bird", "row"]].merge(st, on=["bird", "row"], how="left")
        if st["state_hmm"].isna().any():
            raise ValueError(f"{s}: {int(st['state_hmm'].isna().sum())} windows without a K=2 state")
        if st["state_hmm"].nunique() != 2:
            raise ValueError(f"{s}: expected K=2 states")
        fs = flight_state(st["state_hmm"], F["speed_mean"])
        F[f"flight_{r.name}"] = (st["state_hmm"].to_numpy() == fs)
        meta["flight_state"][r.name] = fs
        rinfo.append({"name": r.name, "path": str(r)})
    log(f"{len(F):,} windows; flight state per run: {meta['flight_state']}")

    # injection: encode before the scoring pass so every query is scored with the same references
    src = pick_sources(F, n_inject, rng, f"flight_{rinfo[0]['name']}")
    meta["injection_sources"] = {k_: int(len(v)) for k_, v in src.items()}
    inj, repro = {}, {}
    import torch

    torch.set_num_threads(max(workers, 1))  # BLAS env limits (slurm script) are for the joblib workers
    for r in runs:
        log(f"{r.name}: encoding injected windows")
        q, Hq, repro[r.name] = build_injections(F, windows, r, src, device, rng)
        inj[r.name] = (q, Hq)
    (out / "inject_reproduction.json").write_text(json.dumps(repro, indent=1))

    log("level 1 scoring pass")
    starts = F.groupby("bird", sort=False).indices
    jobs = []
    for b, ix in starts.items():
        ex = {}
        for name, (q, Hq) in inj.items():
            j = np.flatnonzero(np.isin(q["host"].to_numpy(), ix))
            if len(j):
                ex[name] = {"H": Hq[j], "idx": q["host"].to_numpy()[j] - ix[0], "qpos": j}
        jobs.append((ix, ex))
    res = Parallel(n_jobs=workers)(delayed(bird_pass)(F.iloc[ix].reset_index(drop=True), rinfo, k, ref_cap, min_ref,
                                                      seed, ex) for ix, ex in jobs)
    log(f"level 1 pass done ({time.time() - t0:.0f} s)")

    summary = {"runs": {}}
    nuis_scores = {}
    birds_tab = bird_table(F, reference_csv)
    for r in rinfo:
        name = r["name"]
        rs = [x["runs"][name] for x in res]
        Z = {v: np.concatenate([x["z"][v] for x in rs]) for v in VARIANTS}
        LEV = {v: np.concatenate([x["level"][v] for x in rs]) for v in VARIANTS}
        FB = {v: np.concatenate([x["flag_bird"][v] for x in rs]) for v in VARIANTS}
        D = F.assign(flight=F[f"flight_{name}"])
        vrows, vtabs = [], {}
        for v in VARIANTS:
            ok = np.isfinite(Z[v])
            thr = float(np.percentile(Z[v][ok], FLAG_Q))
            ft = factor_table(D[ok], Z[v][ok] > thr)
            vtabs[v] = ft
            lev_names = {0: "n_ref_pm45d", 1: "n_ref_pm90d", 2: "n_ref_whole_record", -1: "n_no_reference"}
            vrows.append({"variant": v, "threshold_z": thr, "n_scored": int(ok.sum()), "n_unscored": int((~ok).sum()),
                          **{nm: int((LEV[v] == lv).sum()) for lv, nm in lev_names.items()},
                          "n_unscored_small_group": int((~ok & (LEV[v] >= 0)).sum()),
                          "worst_format_ratio": worst_format_ratio(ft)})
        var = pd.DataFrame(vrows)
        cov = coverage_table(D, Z)
        cov.to_csv(out / f"l1_coverage_{name}.csv", index=False)
        drop = cov.set_index("factor")
        drop = np.nan_to_num(drop[VARIANTS[0]].to_numpy()[:, None] - drop[list(VARIANTS)].to_numpy()).max(0)
        var["max_coverage_drop_vs_format"] = drop
        var["qualified"] = drop <= MAX_COVERAGE_DROP
        cand = var["worst_format_ratio"].where(var["qualified"]).fillna(np.inf)
        chosen = str(var.loc[cand.idxmin(), "variant"])
        var["chosen"] = var["variant"] == chosen
        var.to_csv(out / f"l1_variants_{name}.csv", index=False)
        pd.concat({v: t for v, t in vtabs.items()}, names=["variant"]).reset_index(level=0).to_csv(
            out / f"l1_factors_{name}.csv", index=False)
        fig_factors(vtabs, name, out / f"l1_factors_{name}.png")
        z, thr = Z[chosen], float(var.loc[var["chosen"], "threshold_z"].iat[0])
        ok = np.isfinite(z)
        flag = ok & (z > thr)
        D = D.assign(z=z, flag=flag, flag_bird=FB[chosen])
        D["state_name"] = np.where(D["flight"], "flight", "ground")
        D["stratum"] = [BASE_NAMES[c] for c in D["base"].to_numpy()]
        unsc = D.assign(unscored=~ok).groupby(["stratum", "state_name"])["unscored"].agg(["sum", "size"])
        unsc.rename(columns={"sum": "n_unscored", "size": "n_windows"}).to_csv(out / f"l1_unscored_{name}.csv")
        top = D[ok].nlargest(2000, "z")
        top.assign(date=top["t_start"].dt.strftime("%Y-%m-%d"))[
            ["bird", "row", "date", "month", "stratum", "state_name", "segment", "k_in_segment", "segment_windows",
             "acc_samples", "z"]].to_csv(out / f"l1_top_windows_{name}.csv", index=False)
        # episodes
        S = D.sort_values(["bird", "segment", "k_in_segment"])
        grp = pd.factorize(S["bird"].astype(str) + "|" + S["segment"].astype(str))[0]
        runs_ = flag_runs(grp, S["k_in_segment"].to_numpy(), S["flag"].to_numpy())
        eps = []
        for i, j in runs_:
            e = S.iloc[i:j + 1]
            eps.append({"bird": e["bird"].iat[0], "segment": e["segment"].iat[0], "k_first": int(e["k_in_segment"].iat[0]),
                        "k_last": int(e["k_in_segment"].iat[-1]), "n_windows": len(e), "mean_z": float(e["z"].mean()),
                        "max_z": float(e["z"].max()), "date": e["t_start"].iat[0].strftime("%Y-%m-%d"),
                        "month": int(e["month"].iat[0]), "state_name": e["state_name"].mode().iat[0],
                        "continuous_share": float((e["sensitivity_k"] == 1).mean()),
                        "gps_share": float((e["gps_fixes"] > 0).mean()),
                        "short_share": float((e["acc_samples"] < FULL_SAMPLES).mean())})
        E = pd.DataFrame(eps, columns=["bird", "segment", "k_first", "k_last", "n_windows", "mean_z", "max_z", "date",
                                       "month", "state_name", "continuous_share", "gps_share", "short_share"])
        E = E.sort_values("mean_z", ascending=False).reset_index(drop=True)
        E.to_csv(out / f"l1_episodes_{name}.csv", index=False)
        # bird-days
        bd = D[ok].groupby(["bird", "day"]).agg(n=("z", "size"), n_flag=("flag_bird", "sum"),
                                                 date=("t_start", lambda t: t.iat[0].strftime("%Y-%m-%d")),
                                                 continuous_share=("sensitivity_k", lambda s: float((s == 1).mean())))
        bd = bd[bd["n"] >= MIN_BIRD_DAY].reset_index().drop(columns="day")
        bd["share"] = bd["n_flag"] / bd["n"]
        bd["excess"] = bd["share"] / ((100 - FLAG_Q) / 100)
        bd["p_binom_descriptive"] = stats.binom.sf(bd["n_flag"] - 1, bd["n"], (100 - FLAG_Q) / 100)
        bd.sort_values("p_binom_descriptive").to_csv(out / f"l1_bird_days_{name}.csv", index=False)
        # blind pages (top episodes, at most 3 per bird)
        tops = E.groupby("bird", sort=False).head(3).head(top_episodes).reset_index(drop=True)
        Rmap = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3)
                for o in json.loads((Path(r["path"]) / "orientation.json").read_text())}
        key = []
        pays = []
        for i, ep in tops.iterrows():
            pays.append(episode_payload(F, ep, windows, Rmap[ep["bird"]], i + 1))
            key.append({"episode": i + 1, **ep.to_dict()})
        for p in range(0, len(pays), 4):
            blind_page(pays[p:p + 4], out / f"blind_{name}_p{p // 4 + 1}.png")
        pd.DataFrame(key, columns=["episode", *E.columns]).to_csv(out / f"blind_key_{name}.csv", index=False)
        # injection recall
        q, _ = inj[name]
        qz = np.full(len(q), np.nan, np.float32)
        for x, (ix, ex) in zip(rs, jobs):
            if name in ex:
                qz[ex[name]["qpos"]] = x["extra_z"][chosen]
        q = q.assign(z=qz)
        rec = recall_table(q, thr, F, rng).assign(run=name, variant=chosen, threshold_z=thr)
        rec.to_csv(out / f"inject_{name}.csv", index=False)
        # level 2
        l2b, l2m = level2(F, D, rs, starts, name, rng, n_null, workers, seed)
        l2b.to_csv(out / f"l2_birds_{name}.csv", index=False)
        l2m.to_csv(out / f"l2_months_{name}.csv", index=False)
        bscore = l2b.groupby("bird")["z_across_birds"].mean()
        mx = l2m.groupby("bird")["z_change"].max() if len(l2m) else pd.Series(dtype=float)
        birds_tab[f"l1_flag_rate_{name}"] = D[ok].groupby("bird")["flag"].mean().reindex(birds_tab.index)
        birds_tab[f"l2_score_{name}"] = bscore.reindex(birds_tab.index)
        birds_tab[f"l2_max_change_{name}"] = mx.reindex(birds_tab.index)
        fig_l2(birds_tab, name, out / f"l2_scores_{name}.png")
        nuis_scores[name] = [f"l1_flag_rate_{name}", f"l2_score_{name}", f"l2_max_change_{name}"]
        ftc = vtabs[chosen]
        summary["runs"][name] = {
            "variant": chosen, "threshold_z": thr, "variants": var.to_dict("records"),
            "factors": ftc[["factor", "ratio", "n_flagged_with", "birds_flagged_with"]].to_dict("records"),
            "n_episodes": len(E), "episodes_continuous_share": float(E["continuous_share"].mean()) if len(E) else None,
            "bird_days": len(bd), "bird_days_p_lt_0.001": int((bd["p_binom_descriptive"] < 1e-3).sum()),
            "recall": rec.to_dict("records"), "l2_birds": int(l2b["bird"].nunique()) if len(l2b) else 0,
            "l2_month_pairs": len(l2m)}
        log(f"{name}: variant {chosen}, worst format ratio {var.loc[var['chosen'], 'worst_format_ratio'].iat[0]:.2f}, "
            f"{len(E)} episodes")
    rec_all = pd.concat([pd.read_csv(out / f"inject_{n}.csv") for n in names], ignore_index=True)
    fig_recall(rec_all, out / "inject_recall.png")
    birds_tab.reset_index().to_csv(out / "birds.csv", index=False)
    nu = nuisance_table(birds_tab, nuis_scores, n_perm, rng)
    nu.to_csv(out / "nuisance.csv", index=False)
    meta["runtime_s"] = round(time.time() - t0, 1)
    meta["summary"] = summary
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    write_summary(out, meta, nu, repro)
    log(f"done ({meta['runtime_s']} s)")
    return meta


def level2(F, D, rs, starts, name, rng, n_null, workers, seed):
    """Bird-vs-pool and month-change tables for one run (see module docstring)."""
    pos = np.concatenate([starts[x_b][x["l2_rows"]] for x_b, x in zip(starts, rs)])
    H = np.concatenate([x["l2_h"] for x in rs]).astype(np.float32)
    S = D.iloc[pos][["bird", "base", "flight", "ym", "day"]].reset_index(drop=True)
    S["cell"] = [f"{BASE_NAMES[b]}|{'flight' if f else 'ground'}" for b, f in zip(S["base"], S["flight"])]
    # PCA on a bird-balanced pool
    fit = np.concatenate([rng.choice(ix, min(2000, len(ix)), replace=False) for ix in S.groupby("bird").indices.values()])
    pca = PCA(L2_PCS, random_state=seed).fit(H[fit])
    X = pca.transform(H).astype(np.float32)
    gamma = median_gamma(X[fit], rng)
    rows, mrows = [], []
    for cell, g in S.groupby("cell"):
        ix = g.index.to_numpy()
        birds = g["bird"].to_numpy()
        ub = [b for b in np.unique(birds) if (birds == b).sum() >= L2_MIN_BIRD]
        if len(ub) < 5:
            continue
        res = Parallel(n_jobs=workers)(delayed(bird_vs_pool)(X[ix], birds, b, gamma, n_null,
                                                             np.random.default_rng(seed_of(seed, name, cell, b))) for b in ub)
        for b, r_ in zip(ub, res):
            if r_:
                rows.append({"bird": b, "cell": cell, **r_})
        for b in ub:
            i = ix[birds == b]
            for m in month_change(X[i], S["ym"].to_numpy()[i], S["day"].to_numpy()[i], gamma):
                mrows.append({"bird": b, "cell": cell, **m})
    B = pd.DataFrame(rows)
    if len(B):
        B["z_across_birds"] = B.groupby("cell")["mmd2"].transform(
            lambda v: (v - v.median()) / max(1.4826 * (v - v.median()).abs().median(), 1e-12))
    M = pd.DataFrame(mrows)
    if len(M):
        M["z_change"] = M.groupby("cell")["excess"].transform(
            lambda v: (v - v.median()) / max(1.4826 * (v - v.median()).abs().median(), 1e-12))
    return B, M


def bird_table(F: pd.DataFrame, reference_csv: Path | None) -> pd.DataFrame:
    g = F.groupby("bird")
    first, last = g["t_start"].min(), g["t_start"].max()
    B = pd.DataFrame({"n_windows": g.size(), "track_days": (last - first).dt.total_seconds() / 86400,
                      "cohort_2025": (first.dt.year > 2024).astype(float),
                      "cont_share": g["sensitivity_k"].apply(lambda s: float((s == 1).mean())),
                      "gps_share": g["gps_fixes"].apply(lambda s: float((s > 0).mean())),
                      "short_share": g["acc_samples"].apply(lambda s: float((s < FULL_SAMPLES).mean()))})
    if reference_csv is not None:
        from . import reference

        ref = reference.load_birds(reference_csv, list(B.index)).set_index("bird")
        B["sex_f"] = ref["sex"].map({"f": 1.0, "m": 0.0}).reindex(B.index)
        B["migrant"] = ref["migrant"].notna().astype(float).reindex(B.index)
        B["tagged_young"] = ref["tagged_young"].notna().astype(float).reindex(B.index)
    B.index.name = "bird"
    return B


NUISANCE = ["cohort_2025", "track_days", "n_windows", "cont_share", "gps_share", "short_share", "sex_f", "migrant",
            "tagged_young"]


def nuisance_table(B: pd.DataFrame, scores: dict, n_perm: int, rng) -> pd.DataFrame:
    rows = []
    for run_, cols in scores.items():
        for c in cols:
            for nu in NUISANCE:
                if nu not in B:
                    continue
                rho, p, n = perm_spearman(B[c].to_numpy(float), B[nu].to_numpy(float), n_perm, rng)
                rows.append({"run": run_, "score": c, "nuisance": nu, "spearman_rho": rho, "p_perm_birds": p,
                             "n_birds": n})
    return pd.DataFrame(rows)


def write_summary(out: Path, meta: dict, nu: pd.DataFrame, repro: dict) -> None:
    L = ["# Anomalies at two levels (stage 7)", "",
         "What this is: a search for unusual behaviour, first as unusual windows of a bird compared with the same "
         "bird in the same season, recording format and coarse state (ground or flight), then as unusual birds "
         "compared with all others. Labels: [descriptive] = counted, no test; [tested] = permutation test with the "
         "bird as the unit.", ""]
    for name, s in meta["summary"]["runs"].items():
        L += [f"## {name}", "",
              f"- Stratification used: `{s['variant']}` (among variants not disqualified for coverage, the one with the smallest worst format ratio; all variants "
              f"in l1_variants_{name}.csv). Pooled 99th-percentile threshold: robust z > {_fmt(s['threshold_z'])}."]
        for v in s["variants"]:
            L.append(f"  - `{v['variant']}`: {v['n_scored']:,} windows scored, {v['n_unscored']:,} unscored ({v['n_no_reference']:,} without a "
                     f"reference, {v['n_unscored_small_group']:,} in groups under {MIN_Z_GROUP} windows), worst "
                     f"format ratio {_fmt(v['worst_format_ratio'])}, largest drop in format-factor coverage against `format` "
                     f"{_fmt(v['max_coverage_drop_vs_format'])}{'' if v['qualified'] else ' (disqualified)'} "
                     "[descriptive]")
        L.append("- Flagged windows by factor (share among flagged / share among scored; 1 = no over-representation) "
                 "[descriptive]:")
        for f in s["factors"]:
            if not f["factor"].startswith("month"):
                L.append(f"  - {f['factor']}: ratio {_fmt(f['ratio'])} ({f['n_flagged_with']} windows, "
                         f"{f['birds_flagged_with']} birds)")
        neutral = all(1 / NEUTRAL <= f["ratio"] <= NEUTRAL for f in s["factors"]
                      if f["factor"] in FACTORS and FACTORS[f["factor"]][1] and np.isfinite(f["ratio"]))
        L.append(f"- Format neutrality (every format ratio within {1 / NEUTRAL:.2f}-{NEUTRAL}): "
                 f"{'met' if neutral else 'NOT met; read the flags as partly recording artefacts'} (factors without "
                 "flagged or scored windows are left out).")
        L.append(f"- Episodes (>= {MIN_EPISODE} consecutive flagged windows in one segment): {s['n_episodes']}; mean "
                 f"continuous-mode share {_fmt(s['episodes_continuous_share'])}. Episodes need consecutive windows, so "
                 "they come almost only from continuous recording; bird-days cover scheduled bursts [descriptive].")
        L.append(f"- Bird-days with >= {MIN_BIRD_DAY} windows: {s['bird_days']}; with a binomial tail p < 0.001 for "
                 f"the share above the bird's own 99th percentile: {s['bird_days_p_lt_0.001']} (descriptive screen: "
                 "windows of one day are not independent).")
        L.append("- Injected faults, share above the threshold (95% interval from resampling birds) [descriptive]:")
        for r in s["recall"]:
            L.append(f"  - {r['fault']} ({r['role']}): {_fmt(r['flag_rate'])} [{_fmt(r['ci95_lo'])}, "
                     f"{_fmt(r['ci95_hi'])}], {r['n_scored']} windows, {r['n_birds']} birds")
        rp = repro.get(name, {})
        L.append(f"- Re-encoding check: cosine between stored and re-encoded clean h, minimum "
                 f"{_fmt(rp.get('min_cosine_stored_vs_reencoded'), 4)}, median "
                 f"{_fmt(rp.get('median_cosine_stored_vs_reencoded'), 4)} ({rp.get('n')} windows).")
        L.append("- Transplant rows: `transplant_own_R` is another bird's window as that bird's model input, scored "
                 "in the host's record (the identity question: r4 keeps identity, r6 reduces it); "
                 "`transplant_host_R` is a record mix-up and also contains the mounting difference between tags.")
        L.append(f"- Level 2{' (primary run for level 2: identity kept)' if 'r4' in name else ''}: {s['l2_birds']} birds compared with the pool; {s['l2_month_pairs']} consecutive "
                 "month pairs scored. Every bird is expected to differ from the pool (identity); the ranking, not "
                 "the p-value, is the result (l2_birds, l2_months).")
        L.append("")
    L += ["## Bird-level scores against nuisance [tested: Spearman, permutation between birds]", "",
          "Level 2 is read primarily from the run that keeps identity (r4); the identity-reduced run is shown "
          "for comparison.", "", "| run | score | nuisance | rho | p | birds |", "|---|---|---|---|---|---|"]
    for _, r in nu.iterrows():
        L.append(f"| {r['run']} | {r['score']} | {r['nuisance']} | {_fmt(r['spearman_rho'])} | "
                 f"{_fmt(r['p_perm_birds'])} | {r['n_birds']} |")
    L += ["", "## Files", "",
          "- l1_variants_*, l1_factors_* (csv, png): stratification variants and flagged-set factors.",
          "- l1_top_windows_*, l1_episodes_*, l1_bird_days_*, l1_unscored_*: window, episode and bird-day results.",
          "- blind_*_p*.png: top episodes without bird id or date (relative seconds, month); blind_key_*.csv links "
          "them to birds.", "- inject_*.csv, inject_recall.png, inject_reproduction.json: fault validation.",
          "- l2_birds_*, l2_months_*, l2_scores_*.png, birds.csv, nuisance.csv: level 2 and nuisance relations.",
          "- metadata.json: settings and counts.", "",
          "Not verified here: whether any flag is a real behaviour; that needs the blind pages to be read."]
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--states2", type=Path, action="append", required=True, help="K=2 states dir, one per --run")
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None, help="Movebank reference CSV (sex, migrant, young)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-inject", type=int, default=2000)
    ap.add_argument("--ref-cap", type=int, default=2000)
    ap.add_argument("--min-ref", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    run(a.run, a.states2, a.windows, a.out, a.reference, a.k, a.n_inject, a.workers, a.seed, a.device, a.ref_cap,
        a.min_ref, a.n_null, a.n_perm)


if __name__ == "__main__":
    main()
