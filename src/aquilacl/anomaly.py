"""Stage 7, version 2: two separate products, sensor/recording quality control (QC) and behavioural anomalies.
Answers PLAN_2026-11-25 step 7 and review_methods_2026-11 §5, after anomaly_v1 showed that h-distance cannot see
sensor faults (h was trained to ignore rotation, scale, noise, crops and missing sensors) and that short windows
and the recording mode drove the window ranking.

    python -m aquilacl.anomaly --windows <dir> --out <new dir> [--run <run> --states2 <K=2 dir> --states6 <K=6 dir>]...
        [--reference <csv>] [--n-inject 2000] [--workers N] [--seed 0] [--parts qc behaviour] [--birds ...]

PART 1. Sensor/recording QC (deterministic; no model). Per window, from acc.npy / gps.npy in the sensor frame:
- stuck_axis: one axis constant (SD < STUCK_SD_G) while another varies (SD >= VARY_SD_G); windows with >= 20 samples.
- static_norm: still window (VeDBA < STILL_VEDBA) whose mean acceleration norm is off 1 g by > NORM_TOL_G.
- tilt_night: still night window whose gravity direction is > TILT_WIN_DEG from the roost direction of the
  previous ROOST_BEFORE qualifying nights (the bird's whole-record roost direction while fewer than 3 exist).
- yaw_flap: flapping window (rule of features.reference_labels) with >= 100 samples whose principal horizontal
  dynamic-acceleration axis (plane perpendicular to the bird's roost gravity direction; axial, 0-90 deg) is
  > YAW_WIN_DEG from the axial mean of the bird's flapping windows of the previous YAW_REF_DAYS days. A yaw slip
  (rotation about gravity) leaves every gravity check unchanged; this axis is the only acc quantity that sees it.
- clipping: >= CLIP_SHARE of samples within 1.5 steps of the bird x sensitivity x axis extreme (rails at >= 1.4 g).
- gps_flap_zero: flapping window with >= 5 speed fixes whose ground speed never exceeds 0.5 m/s.
- gps_impossible: ground speed > 50 m/s, climb > 20 or < -50 m/s, a 1-s speed change > 25 m/s, or ground speed
  and track speed differing by > 15 m/s.
Per night: roost change = angle between the median gravity direction of the next ROOST_AFTER qualifying nights
(this one included) and that of the previous ROOST_BEFORE; events = runs of nights above ROOST_CP_DEG (30 deg; chosen
after seeing the clean rate on the real data, so slip recall is not out of sample). A 180 deg yaw slip is invisible
to every check (gravity unchanged, flapping axis axial).
Per day: still static-norm median off 1 g by > DAY_NORM_TOL_G; flapping-axis axial mean of the day > YAW_DAY_DEG
from the previous YAW_REF_DAYS days; share of flapping-with-GPS windows at zero speed >= 0.5.
Validation: the v1 faults (stuck axis, 90 deg tilt about a horizontal axis of the roost frame, 90 deg yaw about the
roost gravity direction, amplitude halved, GPS speed zeroed on flapping windows with GPS, another bird's window) plus
saturation (x2.5, clipped at the bird's rails) and a GPS speed spike (one fix at 80 m/s), applied to raw windows of
the host bird and run through the same code; recall per check among ELIGIBLE windows, eligible share, and the
clean false-positive rate. Persistent slips: from a random onset, all later still-night windows are tilted by
15/30/90 deg (or all later flapping windows yawed); detected if the night/day statistic exceeds its threshold
within -4..+4 nights/days of the onset (SLIP_DEG: 15-45 deg in 5 deg steps and 90 deg); the clean rate is the same
test at the same positions without a slip. Injections that set exactly the thresholded quantity (stuck axis,
amplitude, saturation, GPS zero / spike) only show that the code works.
Timelines use days relative to the bird's first window (no dates).

PART 2. Behavioural anomalies (h of each --run). Full windows only (acc_samples >= 190) at the window, bout and
day levels; short windows are not reported at any of them.
- Window level: k-NN distance (k-th neighbour, L2-normalised h) within the same bird, recording mode
  (sensitivity_k 0/1) and K=2 state, reference within +-45 days (15-day query chunks; fallback +-90 d, whole record;
  same segment and same local day never neighbours), divided by the reference's median k-NN distance, robust z
  within bird x mode x state. Primary flags: 99th percentile of z per mode x state, pooled over birds (mode and
  state neutral by construction; birds can differ in rate). "Flags" at every behavioural level are the top 1% ranked
  for inspection, not detections (no window-level injection exists). One pooled threshold and the per-bird top 1% are
  reported as well. The remaining format factors (segment edge, 190-199 samples, partial GPS, single-window
  segment) are the real neutrality test.
- Bout level (continuous mode): bouts as flight.bout_ids (k=1 windows <= 25 s apart, >= 60 s) with >= 6 full
  windows; length bins 6-15 / 16-45 / 46-84 / >= 85 full windows (the last = at the ~900 s recording cap). Content
  score: k-NN distance of the bout's mean h to the bird's other bouts of the same bin on other days within +-45 d
  (fallbacks as above), robust z within bird x bin. Sequence score: max of (order: k-NN distance of the
  half-difference vector mean(second half) - mean(first half); smoothness: low mean cosine between consecutive full
  windows), each robust z within bird x bin. Primary flags: 99th percentile per bin, pooled over birds.
- Day level (scheduled mode): bird-days with >= DAY_M full k=0 windows over >= DAY_MIN_HOURS distinct local solar
  hours (partial days are counted, not scored). Day score = DAY_K-th smallest energy distance (DAY_PCS principal
  components of h; PCA on a bird-balanced sample of full k=0 windows) between the day and each other eligible day of
  the bird within +-45 d; each comparison draws DAY_M windows per side WITHOUT replacement from the hours both days
  have (hour h gives at most min(nQ_h, nY_h) to each side), so both sides are single days of equal size and hour
  composition with no repeated window (with replacement, days with few windows per hour scored high).
  Robust z within bird; one pooled 99th-percentile threshold.
- Validation with h-level injections: a bout replaced by another bird's bout of the same length bin whose K=6
  state shares differ from the host bout (L1 >= 0.5: diff_mix), differ from every bout of the host in that bin
  within +-45 d (novel_mix), or match the host bout (L1 <= 0.2: same_mix, the identity control); a bout's windows
  reversed or shuffled (sequence score); a day replaced by the same bird's eligible day >= 120 days away, or by
  another bird's eligible day within +-15 days of the year. Recall at the primary threshold of the host's stratum,
  AUC against the clean scores of the same hosts, 95% intervals from resampling host birds.
- Format neutrality: share of each factor among flagged / among scored units; met when every format factor
  present in >= 0.1% of scored units has a ratio within 0.67-1.5. It concerns scored units only: short windows are
  excluded before scoring, and the summary states the share of windows, bird-days and birds with no score.

Level 2 (bird among birds; every window format, each in its own cell): cells = window format (full / 40-sample /
64-sample / other short) x mode x K=2 state x season (DJF, MAM, JJA, SON). Each bird with >= L2_N windows in a cell
contributes exactly L2_N (equal size); unbiased MMD^2 (RBF, median heuristic, L2_PCS principal components) between
the bird and 2 L2_N rows of the other birds of the cell, each bird weighted equally; only cells with >= L2_MIN_BIRDS
(20) birds; within-cell rank of the bird as a fraction (rank - 0.5) / n. Bird score = median of its within-cell
ranks (>= L2_MIN_CELLS cells). Month change: MMD^2 between consecutive calendar months (L2_MONTH_N rows each; pairs
across a gap are listed, not scored) minus the within-month floor (odd vs even local days, L2_MONTH_N / 2 rows
each); bird summary = mean z of its changes. Nuisance [descriptive screen]: Spearman rho and partial Spearman (ranks
residualised on track_days, n_windows, deployment year and deployment month by least squares; p from permuting
residuals between birds), Holm over the table, floor p printed as "<= 1/(n_perm+1)". Cohort = real deployment year
(earliest deploy-on of the reference data). The level-1 flag rate is not a nuisance score: with per-bird references
it carries no between-bird information by design.

Privacy: latitude and longitude are never read. Outputs carry relative days and months, never dates, clock times
or solar hours. Bird ids are never printed to stdout; blind pages carry no bird id (their key is a separate CSV).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
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

from . import features  # noqa: E402
from .flight import BOUT_STEP_S, MIN_BOUT_S, bout_ids  # noqa: E402
from .windows import GPS_CHANNELS  # noqa: E402

FULL_SAMPLES = 190
NIGHT_SUN = -6.0
FALLBACK_OFFSET_H = -7.0  # as budgets.py: used only for a bird with no local_solar_hour at all
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
NEUTRAL = 1.5  # format ratios within [1/NEUTRAL, NEUTRAL] count as neutral
FLAG_Q = 99.0
MIN_Z_GROUP = 30
META_COLS = ["t_start", "segment", "k_in_segment", "segment_windows", "sensitivity_k", "acc_samples", "gps_fixes",
             "sun_elev_deg", "local_solar_hour"]
FEAT_COLS = ["label_flapping", "speed_mean", "vedba"]

# ---- QC thresholds (physical or from the clean distributions measured on 16+15 birds, see the module report)
MIN_ACC = 20
STILL_VEDBA = 0.03
STUCK_SD_G = 1e-4  # an axis this flat is constant (the step is about 0.001-0.002 g; clean data: never)
VARY_SD_G = 2e-4
NORM_TOL_G = 0.1  # clean still windows: 0.1-99.9 % of |norm| within 0.955-1.08 g
DAY_NORM_TOL_G = 0.03  # per-day median; clean per-bird MAD about 0.001 g
TILT_WIN_DEG = 60.0
ROOST_BEFORE, ROOST_AFTER, ROOST_MIN_WIN = 7, 5, 5
ROOST_CP_DEG = 30.0  # chosen AFTER seeing the real clean rate (15 deg: 9%) and the injected slips; not pre-set
FLAP_VEDBA, FLAP_BAND = 0.25, 0.35  # features.reference_labels defaults
YAW_MIN_SAMPLES, YAW_MIN_ANISO = 100, 0.3
YAW_WIN_DEG, YAW_DAY_DEG, YAW_REF_DAYS, YAW_MIN_REF, YAW_DAY_MIN = 30.0, 15.0, 30, 20, 10
CLIP_RAIL_G, CLIP_SHARE = 1.4, 0.05
GPS_ZERO_MS, GPS_MIN_FIX = 0.5, 5
GPS_MAX_SPEED, GPS_MAX_CLIMB, GPS_MIN_CLIMB, GPS_MAX_DSPEED, GPS_MAX_MISMATCH = 50.0, 20.0, -50.0, 25.0, 15.0
DAY_ZERO_SHARE, DAY_ZERO_MIN = 0.5, 5
SLIP_DEG = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 90.0)  # recall curve over 20-45 deg (I24)
SLIP_WIN = 4
QC_CHECKS = ("stuck_axis", "static_norm", "tilt_night", "yaw_flap", "clipping", "gps_flap_zero", "gps_impossible")
QC_FAULTS = ("clean", "stuck_axis", "tilt90", "yaw90", "amplitude_half", "saturate", "transplant",
             "clean_still_night", "tilt90_still_night", "clean_flapping", "yaw90_flapping", "clean_gps_source",
             "gps_speed_zero", "clean_gps_any", "gps_spike")
CHECK_FOR_FAULT = {"stuck_axis": "stuck_axis", "tilt90": "tilt_night", "yaw90": "yaw_flap",
                   "tilt90_still_night": "tilt_night", "yaw90_flapping": "yaw_flap",
                   "amplitude_half": "static_norm", "saturate": "clipping", "gps_speed_zero": "gps_flap_zero",
                   "gps_spike": "gps_impossible", "transplant": "tilt_night"}

# ---- behaviour
CHUNK_DAYS = 15
SEASON_DAYS = (45, 90, None)
SCALE_N = 200
BOUT_MIN_FULL = 6
BOUT_BINS = (6, 16, 46, 85)  # 6-15, 16-45, 46-84, >= 85 (at the recording cap) full windows
BOUT_CAP_FULL = 85
BOUT_K, BOUT_MIN_REF = 5, 20
DAY_M, DAY_MIN_HOURS, DAY_K, DAY_PCS, DAY_MIN_PAIRS = 20, 12, 5, 10, 10
MIX_DIFF, MIX_SAME = 0.5, 0.2
OTHER_SEASON_DAYS, OTHER_BIRD_DOY = 120, 15

# ---- level 2
L2_PCS, L2_N, L2_MIN_CELLS, L2_MONTH_N, L2_MIN_BIRDS = 20, 200, 3, 40, 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((unit(u) * unit(v)).sum(-1), -1, 1)))


def rot_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    k = unit(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def plane_basis(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to g (from the sensor axis least aligned with g)."""
    g = unit(g)
    e = np.eye(3)[int(np.argmin(np.abs(g)))]
    e1 = unit(e - (e @ g) * g)
    return e1, np.cross(g, e1)


def horizontal_axis(g: np.ndarray, rng) -> np.ndarray:
    e1, e2 = plane_basis(g)
    t = rng.uniform(0, 2 * np.pi)
    return np.cos(t) * e1 + np.sin(t) * e2


def solar_offset(t: pd.Series, solar_hour: np.ndarray) -> tuple[np.ndarray, int]:
    """Local mean solar time minus UTC (hours in [-12, 12)) per window; missing local_solar_hour filled from the
    previous / next window in time (as budgets.solar_offset_h), else FALLBACK_OFFSET_H. Returns (offset, n missing)."""
    t = pd.to_datetime(pd.Series(t), utc=True).reset_index(drop=True)
    utc_h = (t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600).to_numpy(float)
    off = (np.asarray(solar_hour, float) - utc_h + 12) % 24 - 12
    n_nan = int(np.isnan(off).sum())
    if n_nan:
        order = np.argsort(t.to_numpy(), kind="stable")
        s = pd.Series(off[order])
        filled = s.ffill().bfill().to_numpy() if s.notna().any() else np.full(len(s), FALLBACK_OFFSET_H)
        off = np.empty_like(off)
        off[order] = filled
    return off, n_nan


def local_days(t: pd.Series, solar_hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer local solar day and night (noon to noon) from UTC time and local solar hour (gaps filled, see
    solar_offset)."""
    t = pd.to_datetime(pd.Series(t), utc=True).reset_index(drop=True)
    off, _ = solar_offset(t, solar_hour)
    loc = (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy() / 86400 + off / 24
    return np.floor(loc).astype(np.int64), np.floor(loc + 0.5).astype(np.int64)


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


def stratum_flags(z: np.ndarray, strata: np.ndarray, q: float = FLAG_Q) -> tuple[np.ndarray, dict]:
    """z above the q-th percentile of its stratum (pooled over birds); returns flags and {stratum: threshold}."""
    z = np.asarray(z, float)
    thr = {}
    for s in np.unique(strata):
        v = z[(strata == s) & np.isfinite(z)]
        if len(v):
            thr[s] = float(np.percentile(v, q))
    t = np.array([thr.get(s, np.inf) for s in strata])
    return np.nan_to_num(z, nan=-np.inf) > t, thr


def norm_rows(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, np.float32)
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)


def _cols(path: Path, want: list[str]) -> pd.DataFrame:
    have = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[c for c in want if c in have])


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}g}"


# ================================================================ factor tables (shared)


def factor_table(d: pd.DataFrame, flag: np.ndarray, factors: dict) -> pd.DataFrame:
    """Share of flagged units with each factor against the share among scored units (ratio 1 = neutral).
    ``factors``: name -> (mask function on d, is a format factor)."""
    flag = np.asarray(flag, bool)
    fl = d[flag]
    facs = dict(factors)
    if "month" in d:
        facs.update({f"month {m}": ((lambda x, m=i + 1: x["month"] == m), False) for i, m in enumerate(MONTHS)})
    rows = []
    for name, (fn, is_fmt) in facs.items():
        m_all, m_fl = np.asarray(fn(d), bool), np.asarray(fn(fl), bool)
        base = float(m_all.mean()) if len(d) else np.nan
        share = float(m_fl.mean()) if len(fl) else np.nan
        rows.append({"factor": name, "format_factor": bool(is_fmt), "n_scored": len(d), "n_flagged": len(fl),
                     "n_flagged_with": int(m_fl.sum()), "share_flagged": share, "share_scored": base,
                     "ratio": share / base if base > 0 else np.nan,
                     "birds_flagged_with": int(fl.loc[m_fl, "bird"].nunique()) if len(fl) else 0})
    return pd.DataFrame(rows)


def worst_format_ratio(ft: pd.DataFrame, min_base: float = 1e-3) -> float:
    """max |log ratio| over format factors present in at least ``min_base`` of scored units, as a ratio >= 1."""
    r = ft[ft["format_factor"] & (ft["share_scored"] >= min_base)]["ratio"].astype(float)
    r = r[np.isfinite(r)]
    if not len(r):
        return np.nan
    return float(np.exp(np.abs(np.log(np.maximum(r, 1e-6))).max()))


def boot_rate(hit: np.ndarray, birds: np.ndarray, rng, n_boot: int = 1000) -> tuple[float, float, float]:
    """Rate of ``hit`` with a 95% interval from resampling birds."""
    hit = np.asarray(hit, float)
    if not len(hit):
        return np.nan, np.nan, np.nan
    per = pd.DataFrame({"b": birds, "h": hit}).groupby("b")["h"].agg(["sum", "size"])
    bi = rng.integers(0, len(per), (n_boot, len(per)))
    bt = per["sum"].to_numpy()[bi].sum(1) / per["size"].to_numpy()[bi].sum(1)
    return float(hit.mean()), float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = stats.rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ================================================================ PART 1: QC


def window_stats(a: np.ndarray, am: np.ndarray, g: np.ndarray, gm: np.ndarray) -> dict:
    """Per-window QC quantities of sensor-frame acc (N, T, 3) in g and raw GPS (N, S, C)."""
    a = np.asarray(a, np.float32)
    n = am.sum(1)
    mw = am[:, :, None]
    mean = (a * mw).sum(1) / np.maximum(n, 1)[:, None]
    sd = np.sqrt((((a - mean[:, None]) ** 2) * mw).sum(1) / np.maximum(n - 1, 1)[:, None])
    f = features.acc_features(a, am)
    vedba, fb = f["vedba"].to_numpy(), f["flap_band_frac"].to_numpy()
    flap = (vedba > FLAP_VEDBA) & (fb > FLAP_BAND) & (n >= MIN_ACC)
    C = np.full((len(a), 3, 3), np.nan, np.float32)
    big = flap & (n >= YAW_MIN_SAMPLES)
    if big.any():
        dyn = (a[big] - mean[big][:, None]) * mw[big]
        C[big] = np.einsum("nti,ntj->nij", dyn, dyn) / n[big][:, None, None]
    sp = GPS_CHANNELS.index("ground-speed")
    tr, cl = GPS_CHANNELS.index("speed_track_ms"), GPS_CHANNELS.index("climb_ms")
    gs = np.where(gm[:, :, sp], g[:, :, sp], np.nan)
    st = np.where(gm[:, :, tr], g[:, :, tr], np.nan)
    cb = np.where(gm[:, :, cl], g[:, :, cl], np.nan)
    n_sp = np.isfinite(gs).sum(1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gs_max = np.nanmax(gs, 1)
        bad = ((gs_max > GPS_MAX_SPEED) | (np.nanmax(cb, 1) > GPS_MAX_CLIMB) | (np.nanmin(cb, 1) < GPS_MIN_CLIMB)
               | (np.nanmax(np.abs(np.diff(gs, axis=1)), 1) > GPS_MAX_DSPEED)
               | (np.nanmax(np.abs(gs - st), 1) > GPS_MAX_MISMATCH))
    return {"n": n, "mean": mean, "sd": sd, "vedba": vedba, "static_norm": f["static_norm"].to_numpy(),
            "flap": flap, "C": C, "n_speed": n_sp, "gs_max": gs_max, "gps_bad": bad & (n_sp >= 1)}


def rail_values(a: np.ndarray, am: np.ndarray, kk: np.ndarray) -> dict:
    """{sensitivity_k: (3, 2) array of (min, max) per axis}; an extreme counts as a rail only at >= CLIP_RAIL_G."""
    out = {}
    for k in np.unique(kk):
        s = kk == k
        m = am[s][:, :, None]
        lo = np.where(m, a[s], np.inf).min((0, 1))
        hi = np.where(m, a[s], -np.inf).max((0, 1))
        out[int(k)] = np.c_[np.where(lo <= -CLIP_RAIL_G, lo, np.nan), np.where(hi >= CLIP_RAIL_G, hi, np.nan)]
    return out


def clip_share(a: np.ndarray, am: np.ndarray, kk: np.ndarray, rails: dict, tol: float = 0.0035) -> np.ndarray:
    out = np.zeros(len(a), np.float32)
    for k, r in rails.items():
        s = np.flatnonzero(kk == k)
        if not len(s):
            continue
        x = np.asarray(a[s], np.float32)
        on = np.zeros(x.shape, bool)
        for ax in range(3):
            for j in range(2):
                if np.isfinite(r[ax, j]):
                    on[:, :, ax] |= np.abs(x[:, :, ax] - r[ax, j]) <= tol
        out[s] = (on.any(2) & am[s]).sum(1) / np.maximum(am[s].sum(1), 1)
    return out


def axial_angle(C: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal axis angle (rad, in [0, pi)) and anisotropy of the dynamic covariance projected on (e1, e2)."""
    cxx = np.einsum("i,nij,j->n", e1, C, e1)
    cyy = np.einsum("i,nij,j->n", e2, C, e2)
    cxy = np.einsum("i,nij,j->n", e1, C, e2)
    two = np.arctan2(2 * cxy, cxx - cyy)
    aniso = np.hypot(2 * cxy, cxx - cyy) / np.maximum(cxx + cyy, 1e-12)
    return np.mod(two / 2, np.pi), aniso


def axial_dev_deg(phi: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = np.abs(np.mod(phi - ref + np.pi / 2, np.pi) - np.pi / 2)
    return np.degrees(d)


def roost_direction(mean: np.ndarray, still: np.ndarray, night: np.ndarray) -> np.ndarray:
    """Median unit gravity direction of still night windows (still windows if none; NaN if none)."""
    for sel in (still & night, still):
        if sel.sum() >= 5:
            return unit(np.median(unit(mean[sel]), 0))
    return np.full(3, np.nan)


def nightly(u: np.ndarray, night_id: np.ndarray, min_win: int = ROOST_MIN_WIN) -> tuple[np.ndarray, np.ndarray]:
    """Qualifying nights (>= min_win windows) and their median unit direction."""
    if not len(u):
        return np.zeros(0, np.int64), np.zeros((0, 3))
    df = pd.DataFrame(unit(u), columns=list("xyz"))
    df["n"] = night_id
    cnt = df.groupby("n").size()
    med = df.groupby("n")[list("xyz")].median()[cnt >= min_win]
    return med.index.to_numpy(np.int64), unit(med.to_numpy())


def roost_change(U: np.ndarray, before: int = ROOST_BEFORE, after: int = ROOST_AFTER) -> tuple[np.ndarray, np.ndarray]:
    """Per qualifying night i: angle between the median direction of nights i..i+after-1 and of the previous
    ``before`` nights (NaN where incomplete); and the previous-nights reference direction (bird median while < 3)."""
    n = len(U)
    cp = np.full(n, np.nan)
    ref = np.full((n, 3), np.nan)
    whole = unit(np.median(U, 0)) if n else np.full(3, np.nan)
    for i in range(n):
        ref[i] = unit(np.median(U[max(0, i - before):i], 0)) if i >= 3 else whole
        if i >= before and i + after <= n:
            cp[i] = angle_deg(np.median(U[i - before:i], 0), np.median(U[i:i + after], 0))
    return cp, ref


def runs_above(x: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """(first, last) index of runs where x > thr (NaN breaks nothing and counts as below)."""
    hi = np.nan_to_num(x, nan=-np.inf) > thr
    out, start = [], None
    for i, h in enumerate(hi):
        if h and start is None:
            start = i
        if start is not None and (not h or i == len(hi) - 1):
            out.append((start, i if h else i - 1))
            start = None
    return out


def yaw_reference(phi: np.ndarray, day: np.ndarray, query_day: np.ndarray, ref_days: int = YAW_REF_DAYS,
                  min_ref: int = YAW_MIN_REF) -> np.ndarray:
    """Axial mean angle of ``phi`` over days [d - ref_days, d - 1] per query day (whole record minus day d if
    fewer than min_ref); NaN without enough windows."""
    out = np.full(len(query_day), np.nan)
    if not len(phi):
        return out
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    ud = np.unique(day)
    C = pd.Series(c).groupby(day).sum().reindex(ud).to_numpy()
    S = pd.Series(s).groupby(day).sum().reindex(ud).to_numpy()
    N = pd.Series(np.ones(len(day))).groupby(day).sum().reindex(ud).to_numpy()
    cC, cS, cN = np.r_[0, np.cumsum(C)], np.r_[0, np.cumsum(S)], np.r_[0, np.cumsum(N)]
    for j, d in enumerate(np.unique(query_day)):
        lo, hi = np.searchsorted(ud, d - ref_days), np.searchsorted(ud, d)
        sc, ss, sn = cC[hi] - cC[lo], cS[hi] - cS[lo], cN[hi] - cN[lo]
        if sn < min_ref:
            own = np.searchsorted(ud, d)
            on = own < len(ud) and ud[own] == d
            sc, ss = cC[-1] - (C[own] if on else 0), cS[-1] - (S[own] if on else 0)
            sn = cN[-1] - (N[own] if on else 0)
        if sn >= min_ref:
            out[query_day == d] = np.mod(np.arctan2(ss, sc) / 2, np.pi)
    return out


def qc_flags(S: dict, clip: np.ndarray, night: np.ndarray, night_ref: np.ndarray, phi: np.ndarray,
             aniso: np.ndarray, yaw_ref: np.ndarray) -> pd.DataFrame:
    """Eligibility ('elig_<check>') and flag per check from window_stats output and the bird's references."""
    n, sd = S["n"], np.sort(S["sd"], 1)
    ok = n >= MIN_ACC
    still = ok & (S["vedba"] < STILL_VEDBA)
    out = {}
    out["elig_stuck_axis"] = ok
    out["stuck_axis"] = ok & (sd[:, 0] < STUCK_SD_G) & (sd[:, 1] >= VARY_SD_G)
    out["elig_static_norm"] = still
    out["static_norm"] = still & (np.abs(S["static_norm"] - 1) > NORM_TOL_G)
    has_ref = np.isfinite(night_ref).all(1)
    out["elig_tilt_night"] = still & night & has_ref
    ang = np.where(has_ref, angle_deg(S["mean"], np.nan_to_num(night_ref)), np.nan)
    out["tilt_night"] = out["elig_tilt_night"] & (ang > TILT_WIN_DEG)
    ey = S["flap"] & (n >= YAW_MIN_SAMPLES) & (aniso > YAW_MIN_ANISO) & np.isfinite(yaw_ref) & np.isfinite(phi)
    out["elig_yaw_flap"] = ey
    out["yaw_flap"] = ey & (axial_dev_deg(np.nan_to_num(phi), np.nan_to_num(yaw_ref)) > YAW_WIN_DEG)
    out["elig_clipping"] = ok
    out["clipping"] = ok & (clip >= CLIP_SHARE)
    eg = S["flap"] & (S["n_speed"] >= GPS_MIN_FIX)
    out["elig_gps_flap_zero"] = eg
    out["gps_flap_zero"] = eg & (S["gs_max"] < GPS_ZERO_MS)
    out["elig_gps_impossible"] = S["n_speed"] >= 1
    out["gps_impossible"] = S["gps_bad"]
    return pd.DataFrame(out)


# ---------------------------------------------------------------- QC faults on raw windows


def fault_stuck(a: np.ndarray, am: np.ndarray, rng) -> np.ndarray:
    """One random sensor axis held at its first present value."""
    a = np.array(a, np.float32)
    for i in range(len(a)):
        p = np.flatnonzero(am[i])
        if len(p):
            ax = rng.integers(3)
            a[i, p, ax] = a[i, p[0], ax]
    return a


def fault_rotate(a: np.ndarray, g_roost: np.ndarray, rng, deg: float = 90.0, yaw: bool = False) -> np.ndarray:
    """Mounting slip: rotation of the sensor frame about the roost gravity direction (yaw) or about a random axis
    perpendicular to it (tilt)."""
    Q = rot_axis(g_roost, deg) if yaw else rot_axis(horizontal_axis(g_roost, rng), deg)
    return np.asarray(a, np.float32) @ Q.T.astype(np.float32)


def fault_saturate(a: np.ndarray, am: np.ndarray, kk: np.ndarray, rails: dict, gain: float = 2.5) -> np.ndarray:
    a = np.asarray(a, np.float32) * gain
    for k, r in rails.items():
        s = kk == k
        lo = np.where(np.isfinite(r[:, 0]), r[:, 0], -np.inf)
        hi = np.where(np.isfinite(r[:, 1]), r[:, 1], np.inf)
        a[s] = np.clip(a[s], lo, hi)
    return a * am[:, :, None]


def fault_gps_zero(graw: np.ndarray, gm: np.ndarray) -> np.ndarray:
    g = np.asarray(graw, np.float32).copy()
    for c in ("ground-speed", "speed_track_ms", "step_m"):
        j = GPS_CHANNELS.index(c)
        g[:, :, j] = np.where(gm[:, :, j], 0.0, g[:, :, j])
    return g


def fault_gps_spike(graw: np.ndarray, gm: np.ndarray, rng, speed: float = 80.0) -> np.ndarray:
    g = np.asarray(graw, np.float32).copy()
    j = GPS_CHANNELS.index("ground-speed")
    for i in range(len(g)):
        p = np.flatnonzero(gm[i, :, j])
        if len(p):
            g[i, rng.choice(p), j] = speed
    return g


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


def _refs_at(rows: np.ndarray, R: dict) -> tuple[np.ndarray, np.ndarray]:
    """Night reference direction and yaw reference angle of the bird's windows ``rows``."""
    return R["night_ref_row"][rows], R["yaw_ref_row"][rows]


def qc_eval(a, am, g, gm, kk, night, rows, R: dict) -> pd.DataFrame:
    """Window stats -> flags with the bird's references R at the host rows."""
    S = window_stats(a, am, g, gm)
    phi, an = axial_angle(np.nan_to_num(S["C"]), *R["basis"]) if np.isfinite(R["g_roost"]).all() else (
        np.full(len(a), np.nan), np.zeros(len(a)))
    phi = np.where(np.isfinite(S["C"][:, 0, 0]), phi, np.nan)
    nref, yref = _refs_at(rows, R)
    return qc_flags(S, clip_share(a, am, kk, R["rails"]), night, nref, phi, an, yref)


def qc_bird(d: Path, f: pd.DataFrame, inject: dict, seed: int, chunk: int = 20000) -> dict:
    """QC of one bird: per-window flags, per-day table, per-night roost series, flapping axes, injected faults
    (``inject``: fault -> host positions in f) and the references needed for transplant hosts."""
    n = len(f)
    kk = f["sensitivity_k"].to_numpy()
    night = f["sun_elev_deg"].to_numpy() < NIGHT_SUN
    A = np.load(d / "acc.npy", mmap_mode="r")
    AM = np.load(d / "acc_mask.npy")
    G = np.load(d / "gps.npy", mmap_mode="r")
    GM = np.load(d / "gps_mask.npy", mmap_mode="r")
    parts = []
    rails = {}
    for s in range(0, n, chunk):  # first pass: rails (extremes per sensitivity) without holding all of acc in float32
        a = np.asarray(A[s:s + chunk], np.float32)
        for k, r in rail_values(a, AM[s:s + chunk], kk[s:s + chunk]).items():
            rails[k] = r if k not in rails else np.c_[np.fmin(rails[k][:, 0], r[:, 0]), np.fmax(rails[k][:, 1], r[:, 1])]
    for s in range(0, n, chunk):
        a = np.asarray(A[s:s + chunk], np.float32)
        am = AM[s:s + chunk]
        S = window_stats(a, am, np.asarray(G[s:s + chunk]), np.asarray(GM[s:s + chunk]))
        S["clip"] = clip_share(a, am, kk[s:s + chunk], rails)
        parts.append(S)
    S = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    still = (S["n"] >= MIN_ACC) & (S["vedba"] < STILL_VEDBA)
    g_roost = roost_direction(S["mean"], still, night)
    basis = plane_basis(g_roost) if np.isfinite(g_roost).all() else (np.full(3, np.nan), np.full(3, np.nan))
    # nights
    sn = still & night
    nid = f["night"].to_numpy()
    nights, U = nightly(S["mean"][sn], nid[sn])
    cp, nref = roost_change(U) if len(U) else (np.zeros(0), np.zeros((0, 3)))
    pos = np.searchsorted(nights, nid)
    night_ref_row = np.full((n, 3), np.nan)
    if len(nights):
        # reference of a window's night: previous nights of its own night if qualifying, else of the next qualifying
        night_ref_row = nref[np.minimum(pos, len(nights) - 1)]
        night_ref_row[pos >= len(nights)] = nref[-1] if len(nref) else np.nan
    # flapping axes
    day = f["day"].to_numpy()
    if np.isfinite(g_roost).all():
        phi, an = axial_angle(np.nan_to_num(S["C"]), *basis)
        phi = np.where(np.isfinite(S["C"][:, 0, 0]), phi, np.nan)
    else:
        phi, an = np.full(n, np.nan), np.zeros(n)
    ey = S["flap"] & (S["n"] >= YAW_MIN_SAMPLES) & (an > YAW_MIN_ANISO) & np.isfinite(phi)
    yaw_ref_row = yaw_reference(phi[ey], day[ey], day)
    R = {"g_roost": g_roost, "basis": basis, "rails": rails, "night_ref_row": night_ref_row,
         "yaw_ref_row": yaw_ref_row}
    FL = qc_flags(S, S["clip"], night, night_ref_row, phi, an, yaw_ref_row)
    # per day
    D = pd.DataFrame({"day": day, "still": still, "norm": np.where(still, S["static_norm"], np.nan)})
    for c in QC_CHECKS:
        D[c] = FL[c].to_numpy()
        D[f"elig_{c}"] = FL[f"elig_{c}"].to_numpy()
    days = D.groupby("day").agg(n_windows=("still", "size"), n_still=("still", "sum"),
                                still_norm_median=("norm", "median"),
                                **{f"n_{c}": (c, "sum") for c in QC_CHECKS},
                                **{f"n_elig_{c}": (f"elig_{c}", "sum") for c in QC_CHECKS})
    dd = days.index.to_numpy()
    yd = yaw_reference(phi[ey], day[ey], dd)
    ym = pd.DataFrame({"d": day[ey], "c": np.cos(2 * phi[ey]), "s": np.sin(2 * phi[ey])}).groupby("d").agg(
        c=("c", "sum"), s=("s", "sum"), n=("c", "size")).reindex(dd)
    yaw_day = np.mod(np.arctan2(ym["s"].to_numpy(), ym["c"].to_numpy()) / 2, np.pi)
    days["yaw_day_dev_deg"] = np.where((ym["n"].to_numpy() >= YAW_DAY_MIN) & np.isfinite(yd),
                                       axial_dev_deg(np.nan_to_num(yaw_day), np.nan_to_num(yd)), np.nan)
    days["roost_change_deg"] = pd.Series(cp, index=nights).reindex(dd).to_numpy()  # night ending on this day
    days["day_static_norm"] = (days["n_still"] >= 10) & ((days["still_norm_median"] - 1).abs() > DAY_NORM_TOL_G)
    days["day_yaw"] = days["yaw_day_dev_deg"] > YAW_DAY_DEG
    days["day_gps_flap_zero"] = (days["n_elig_gps_flap_zero"] >= DAY_ZERO_MIN) & (
        days["n_gps_flap_zero"] >= DAY_ZERO_SHARE * days["n_elig_gps_flap_zero"])
    days["day_stuck"] = (days["n_elig_stuck_axis"] >= 10) & (days["n_stuck_axis"] >= 0.5 * days["n_elig_stuck_axis"])
    days = days.reset_index()
    # injected faults (host bird's own windows)
    rng = np.random.default_rng(seed)
    inj = []
    for fault, ix in inject.items():
        if not len(ix):
            continue
        a, am, g, gm = read_rows(d.parent, np.array([d.name] * len(ix)), f["row"].to_numpy()[ix])
        k_, nt = kk[ix], night[ix]
        if fault == "stuck_axis":
            a = fault_stuck(a, am, rng)
        elif fault in ("tilt90", "yaw90", "tilt90_still_night", "yaw90_flapping"):
            if not np.isfinite(g_roost).all():
                continue
            a = np.stack([fault_rotate(a[i:i + 1], g_roost, rng, yaw=fault.startswith("yaw"))[0]
                          for i in range(len(a))])
        elif fault == "amplitude_half":
            a = a * 0.5
        elif fault == "saturate":
            a = fault_saturate(a, am, k_, rails)
        elif fault == "gps_speed_zero":
            g = fault_gps_zero(g, gm)
        elif fault == "gps_spike":
            g = fault_gps_spike(g, gm, rng)
        r = qc_eval(a, am, g, gm, k_, nt, ix, R)
        inj.append(r.assign(fault=fault, host=ix))
    phi_e = pd.DataFrame({"day": day[ey], "phi": phi[ey]})
    return {"flags": FL, "days": days, "nights": pd.DataFrame({"night": nights, "roost_change_deg": cp}),
            "U": U, "phi": phi_e, "R": R, "inject": pd.concat(inj, ignore_index=True) if inj else pd.DataFrame(),
            "still_dirs": (S["mean"][sn], nid[sn])}


def slip_injection(U_dirs: tuple, rng, n_per_bird: int = 10) -> list[dict]:
    """Persistent tilt: from a random onset night, all later still-night windows rotated by deg about a random
    horizontal axis; the night statistic is recomputed. Returns per onset and angle: detected (any night within
    onset +-SLIP_WIN above ROOST_CP_DEG) and the same test on the clean series."""
    mean, nid = U_dirs
    nights, U = nightly(mean, nid)
    if len(U) < ROOST_BEFORE + ROOST_AFTER + 2 * SLIP_WIN + 5:
        return []
    cp0, _ = roost_change(U)
    g = unit(np.median(U, 0))
    out = []
    lo, hi = ROOST_BEFORE + SLIP_WIN, len(U) - ROOST_AFTER - SLIP_WIN
    for on in rng.choice(np.arange(lo, hi), min(n_per_bird, hi - lo), replace=False):
        w = slice(on - SLIP_WIN, on + SLIP_WIN + 1)
        clean = bool((np.nan_to_num(cp0[w], nan=-1) > ROOST_CP_DEG).any())
        ax = horizontal_axis(g, rng)
        for deg in SLIP_DEG:
            Q = rot_axis(ax, deg)
            later = nid >= nights[on]
            m2 = np.where(later[:, None], unit(mean) @ Q.T, unit(mean))
            n2, U2 = nightly(m2, nid)
            cp, _ = roost_change(U2)
            out.append({"kind": "tilt", "deg": deg, "detected": bool((np.nan_to_num(cp[w], nan=-1) > ROOST_CP_DEG).any()),
                        "clean_flag": clean})
    return out


def yaw_slip_injection(phi: pd.DataFrame, rng, n_per_bird: int = 10) -> list[dict]:
    """Persistent yaw from a random onset day (all later flapping axes turned by deg); day statistic as qc_bird."""
    if len(phi) < 200:
        return []

    def day_dev(p):
        dd = np.unique(p["day"])
        ref = yaw_reference(p["phi"].to_numpy(), p["day"].to_numpy(), dd)
        m = p.assign(c=np.cos(2 * p["phi"]), s=np.sin(2 * p["phi"])).groupby("day").agg(c=("c", "sum"), s=("s", "sum"),
                                                                                        n=("c", "size")).reindex(dd)
        ang = np.mod(np.arctan2(m["s"], m["c"]) / 2, np.pi).to_numpy()
        dev = np.where((m["n"].to_numpy() >= YAW_DAY_MIN) & np.isfinite(ref), axial_dev_deg(ang, np.nan_to_num(ref)), np.nan)
        return dd, dev

    dd, dev0 = day_dev(phi)
    ok = np.flatnonzero(np.isfinite(dev0))
    if len(ok) < 2 * SLIP_WIN + 10:
        return []
    out = []
    for j in rng.choice(ok[SLIP_WIN:-SLIP_WIN], min(n_per_bird, len(ok) - 2 * SLIP_WIN), replace=False):
        onset = dd[j]
        w = (dd >= onset - SLIP_WIN) & (dd <= onset + SLIP_WIN)
        clean = bool((np.nan_to_num(dev0[w], nan=-1) > YAW_DAY_DEG).any())
        for deg in SLIP_DEG:
            p = phi.assign(phi=np.where(phi["day"] >= onset, np.mod(phi["phi"] + np.radians(deg), np.pi), phi["phi"]))
            _, dev = day_dev(p)
            out.append({"kind": "yaw", "deg": deg, "detected": bool((np.nan_to_num(dev[w], nan=-1) > YAW_DAY_DEG).any()),
                        "clean_flag": clean})
    return out


def pick_qc_sources(F: pd.DataFrame, n: int, rng) -> dict:
    """Bird-balanced source positions in F per QC fault (GPS faults on windows with GPS)."""
    def balanced(mask):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return np.zeros(0, int)
        _, inv, cnt = np.unique(F["bird"].to_numpy()[idx], return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]
        return np.sort(rng.choice(idx, min(n, len(idx)), replace=False, p=w / w.sum()))

    acc = balanced(F["acc_samples"].to_numpy() >= MIN_ACC)
    flap_gps = balanced((F["label_flapping"].to_numpy() == 1) & (F["gps_fixes"].to_numpy() >= GPS_MIN_FIX))
    any_gps = balanced(F["gps_fixes"].to_numpy() >= 3)
    src = {f: acc for f in ("clean", "stuck_axis", "tilt90", "yaw90", "amplitude_half", "saturate")}
    src.update(clean_gps_source=flap_gps, gps_speed_zero=flap_gps, clean_gps_any=any_gps, gps_spike=any_gps)
    # eligibility-targeted sources (from the stored features), so the gravity and flap-axis checks are validated on
    # hundreds of eligible windows rather than the few a random draw gives
    n_ok = F["acc_samples"].to_numpy()
    still_night = balanced((n_ok >= MIN_ACC) & (F["vedba"].to_numpy() < STILL_VEDBA)
                           & (F["sun_elev_deg"].to_numpy() < NIGHT_SUN))
    flapping = balanced((n_ok >= YAW_MIN_SAMPLES) & (F["label_flapping"].to_numpy() == 1))
    src.update(clean_still_night=still_night, tilt90_still_night=still_night, clean_flapping=flapping,
               yaw90_flapping=flapping)
    donors = balanced(F["acc_samples"].to_numpy() >= MIN_ACC)
    bird = F["bird"].to_numpy()
    hosts = np.array([rng.choice(acc) for _ in donors]) if len(acc) else np.zeros(0, int)
    keep = bird[hosts] != bird[donors] if len(hosts) else np.zeros(0, bool)
    src["transplant_donor"], src["transplant"] = donors[keep], hosts[keep]
    return src


def qc_recall(I: pd.DataFrame, F: pd.DataFrame, rng) -> pd.DataFrame:
    """Per fault x check: eligible share, flag rate among eligible (recall / false-positive rate) with a 95%
    interval over host birds, and the 'any check' rate."""
    rows = []
    hb = F["bird"].to_numpy()[I["host"].to_numpy()]
    I = I.assign(host_bird=hb, any_check=I[list(QC_CHECKS)].any(axis=1))
    for fault in QC_FAULTS:
        g = I[I["fault"] == fault]
        if not len(g):
            continue
        for c in (*QC_CHECKS, "any_check"):
            el = np.ones(len(g), bool) if c == "any_check" else g[f"elig_{c}"].to_numpy()
            r, lo, hi = boot_rate(g[c].to_numpy()[el], g["host_bird"].to_numpy()[el], rng) if el.any() else (np.nan,) * 3
            rows.append({"fault": fault, "check": c, "n_injected": len(g), "n_birds": int(g["host_bird"].nunique()),
                         "eligible_share": float(el.mean()), "n_eligible": int(el.sum()), "flag_rate_eligible": r,
                         "ci95_lo": lo, "ci95_hi": hi, "flag_rate_all": float(g[c].mean()),
                         "target": CHECK_FOR_FAULT.get(fault) == c,
                         "role": "false-positive rate" if fault.startswith("clean") else "recall"})
    return pd.DataFrame(rows)


def fig_qc_timeline(days: pd.DataFrame, nights: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True, layout="constrained")
    ax[0].plot(days["rel_day"], days["still_norm_median"], ".", ms=2)
    ax[0].axhspan(1 - DAY_NORM_TOL_G, 1 + DAY_NORM_TOL_G, color="0.9")
    ax[0].set_ylabel("still |a| median (g)")
    ax[1].plot(nights["rel_day"], nights["roost_change_deg"], ".", ms=2)
    ax[1].axhline(ROOST_CP_DEG, color="r", lw=0.8, ls=":")
    ax[1].set_ylabel("roost change (deg)")
    ax[2].plot(days["rel_day"], days["yaw_day_dev_deg"], ".", ms=2)
    ax[2].axhline(YAW_DAY_DEG, color="r", lw=0.8, ls=":")
    ax[2].set_ylabel("flap-axis change (deg)")
    for c in QC_CHECKS:
        v = days[f"n_{c}"].to_numpy()
        if v.any():
            ax[3].plot(days["rel_day"], v, ".", ms=3, label=c)
    ax[3].set_ylabel("flagged windows / day")
    ax[3].set_xlabel("days since the bird's first window")
    if ax[3].get_legend_handles_labels()[0]:
        ax[3].legend(fontsize=6, ncol=4)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run_qc(F: pd.DataFrame, dirs: dict, out: Path, n_inject: int, workers: int, seed: int, rng) -> dict:
    src = pick_qc_sources(F, n_inject, rng)
    starts = F.groupby("bird", sort=False).indices
    jobs = []
    for b, ix in starts.items():
        lo = ix[0]
        inj = {f: src[f][np.isin(src[f], ix)] - lo for f in QC_FAULTS if f in src and f != "transplant"}
        jobs.append((b, ix, inj))
    res = Parallel(n_jobs=workers)(delayed(qc_bird)(dirs[b], F.iloc[ix].reset_index(drop=True), inj,
                                                    seed_of(seed, b, "qc")) for b, ix, inj in jobs)
    log(f"QC pass done for {len(res)} birds")
    I = [r["inject"].assign(host=r["inject"]["host"] + ix[0]) for r, (_, ix, _) in zip(res, jobs) if len(r["inject"])]
    # transplant: another bird's raw window evaluated with the host's references
    d_, h_ = src["transplant_donor"], src["transplant"]
    if len(d_):
        a, am, g, gm = read_rows(next(iter(dirs.values())).parent, F["bird"].to_numpy()[d_], F["row"].to_numpy()[d_])
        bix = {b: j for j, (b, _, _) in enumerate(jobs)}
        hb = F["bird"].to_numpy()[h_]
        for b in np.unique(hb):
            i = np.flatnonzero(hb == b)
            R = res[bix[b]]["R"]
            lo = starts[b][0]
            r = qc_eval(a[i], am[i], g[i], gm[i], F["sensitivity_k"].to_numpy()[h_[i]],
                        F["sun_elev_deg"].to_numpy()[h_[i]] < NIGHT_SUN, h_[i] - lo, R)
            I.append(r.assign(fault="transplant", host=h_[i]))
    I = pd.concat(I, ignore_index=True)
    rec = qc_recall(I, F, rng)
    rec.to_csv(out / "qc_inject.csv", index=False)
    # persistent slips
    sl = []
    for b, r in zip(starts, res):
        sr = np.random.default_rng(seed_of(seed, b, "slip"))
        sl += [dict(x, bird=b) for x in slip_injection(r["still_dirs"], sr)]
        sl += [dict(x, bird=b) for x in yaw_slip_injection(r["phi"], sr)]
    SL = pd.DataFrame(sl, columns=["kind", "deg", "detected", "clean_flag", "bird"])
    slip = []
    for (kind, deg), g in SL.groupby(["kind", "deg"]):
        r_, lo_, hi_ = boot_rate(g["detected"].to_numpy(), g["bird"].to_numpy(), rng)
        c_, clo, chi = boot_rate(g["clean_flag"].to_numpy(), g["bird"].to_numpy(), rng)
        slip.append({"kind": kind, "deg": deg, "n_onsets": len(g), "n_birds": int(g["bird"].nunique()),
                     "recall": r_, "ci95_lo": lo_, "ci95_hi": hi_, "clean_rate_same_positions": c_,
                     "clean_ci95_lo": clo, "clean_ci95_hi": chi})
    slip = pd.DataFrame(slip)
    slip.to_csv(out / "qc_slip_inject.csv", index=False)
    # per-window flags (flagged windows only), per-day table, events, per-bird counts, timelines
    first_day = F.groupby("bird")["day"].transform("min").to_numpy()
    wf, dtab, ntab, ev, brow = [], [], [], [], []
    tl = out / "qc_timelines"
    tl.mkdir(exist_ok=True)
    for (b, ix, _), r in zip(jobs, res):
        FL = r["flags"]
        anyf = FL[list(QC_CHECKS)].any(axis=1).to_numpy()
        fd = first_day[ix[0]]
        if anyf.any():
            w = FL.loc[anyf, list(QC_CHECKS)].assign(bird=b, row=F["row"].to_numpy()[ix][anyf],
                                                     rel_day=F["day"].to_numpy()[ix][anyf] - fd,
                                                     month=F["month"].to_numpy()[ix][anyf])
            wf.append(w)
        dy = r["days"].assign(bird=b, rel_day=lambda x: x["day"] - fd)
        mo = F.iloc[ix].groupby("day")["month"].first()
        dy["month"] = mo.reindex(dy["day"]).to_numpy()
        nt = r["nights"].assign(bird=b, rel_day=lambda x: x["night"] - fd)
        events = runs_above(nt["roost_change_deg"].to_numpy(), ROOST_CP_DEG)
        for i, j in events:
            seg = nt.iloc[i:j + 1]
            ev.append({"bird": b, "kind": "roost_change", "rel_day_first": int(seg["rel_day"].iat[0]),
                       "rel_day_last": int(seg["rel_day"].iat[-1]), "n_nights": len(seg),
                       "max_deg": float(seg["roost_change_deg"].max())})
        for i, j in runs_above(dy["yaw_day_dev_deg"].to_numpy(), YAW_DAY_DEG):
            seg = dy.iloc[i:j + 1]
            ev.append({"bird": b, "kind": "flap_axis_change", "rel_day_first": int(seg["rel_day"].iat[0]),
                       "rel_day_last": int(seg["rel_day"].iat[-1]), "n_nights": len(seg),
                       "max_deg": float(seg["yaw_day_dev_deg"].max())})
        span_y = max((F["day"].to_numpy()[ix].max() - fd + 1) / 365.25, 1 / 365.25)
        row = {"bird": b, "n_windows": len(ix), "track_years": span_y, "n_nights_qualifying": int(nt["roost_change_deg"].notna().sum()),
               "roost_events": len(events), "roost_events_per_year": len(events) / span_y,
               "roost_nights_above": int((nt["roost_change_deg"] > ROOST_CP_DEG).sum()),
               "g_roost_found": bool(np.isfinite(r["R"]["g_roost"]).all())}
        for c in QC_CHECKS:
            row[f"n_{c}"] = int(FL[c].sum())
            row[f"n_elig_{c}"] = int(FL[f"elig_{c}"].sum())
            row[f"rate_{c}"] = float(FL[c].sum() / max(FL[f"elig_{c}"].sum(), 1))
        for c in ("day_static_norm", "day_yaw", "day_gps_flap_zero", "day_stuck"):
            row[f"days_{c}"] = int(dy[c].sum())
        brow.append(row)
        dtab.append(dy.drop(columns="day"))
        ntab.append(nt.drop(columns="night"))
    Wf = pd.concat(wf, ignore_index=True) if wf else pd.DataFrame(columns=["bird", "row", "rel_day", "month", *QC_CHECKS])
    Wf.to_parquet(out / "qc_windows_flagged.parquet", index=False)
    Dt = pd.concat(dtab, ignore_index=True)
    Dt.to_csv(out / "qc_bird_days.csv", index=False)
    Nt = pd.concat(ntab, ignore_index=True)
    Nt.to_csv(out / "qc_nights.csv", index=False)
    Ev = pd.DataFrame(ev, columns=["bird", "kind", "rel_day_first", "rel_day_last", "n_nights", "max_deg"])
    Ev.to_csv(out / "qc_events.csv", index=False)
    Bq = pd.DataFrame(brow)
    Bq.to_csv(out / "qc_birds.csv", index=False)
    Parallel(n_jobs=workers)(delayed(fig_qc_timeline)(Dt[Dt["bird"] == b], Nt[Nt["bird"] == b], tl / f"{b}.png",
                                                      "QC timeline (days relative to the bird's first window)")
                             for b in starts)
    tot = {c: int(Bq[f"n_{c}"].sum()) for c in QC_CHECKS}
    elig = {c: int(Bq[f"n_elig_{c}"].sum()) for c in QC_CHECKS}
    return {"recall": rec.to_dict("records"), "slip": slip.to_dict("records"), "flag_counts": tot,
            "eligible_counts": elig, "birds_with_flags": {c: int((Bq[f"n_{c}"] > 0).sum()) for c in QC_CHECKS},
            "roost_events": int((Ev["kind"] == "roost_change").sum()),
            "flap_axis_events": int((Ev["kind"] == "flap_axis_change").sum()),
            "roost_events_per_year_median": float(Bq["roost_events_per_year"].median()),
            "days": {c: int(Dt[c].sum()) for c in ("day_static_norm", "day_yaw", "day_gps_flap_zero", "day_stuck")},
            "n_birds": len(Bq), "n_bird_days": len(Dt)}


# ================================================================ PART 2: behaviour


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


def score_groups(H, day, seg, key, k, cap, min_ref, seed):
    """Density-normalised k-NN score of every window within its key, by 15-day chunk (module docstring).
    Returns (score, level): level 0 = +-45 d, 1 = +-90 d, 2 = whole record, -1 = unscored."""
    n = len(H)
    score, level = np.full(n, np.nan, np.float32), np.full(n, -1, np.int8)
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
    return score, level


def bout_vectors(Hseq: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean h, half-difference (second half minus first half) and smoothness (mean cosine of consecutive rows) of
    a bout's full windows in time order."""
    n = len(Hseq)
    h = n // 2
    s = float((Hseq[1:] * Hseq[:-1]).sum(1).mean()) if n > 1 else np.nan
    return Hseq.mean(0), Hseq[n - h:].mean(0) - Hseq[:h].mean(0), s


def bout_bin(n_full: np.ndarray) -> np.ndarray:
    return np.digitize(n_full, BOUT_BINS) - 1  # -1 below the minimum


def knn_ref_dist(X: np.ndarray, day: np.ndarray, grp: np.ndarray, qX: np.ndarray, qday: np.ndarray, qgrp: np.ndarray,
                 k: int = BOUT_K, min_ref: int = BOUT_MIN_REF, exclude_day: np.ndarray | None = None):
    """k-th nearest Euclidean distance of each query to the rows of the same group on other days, within +-45 d
    (fallback +-90 d, whole record). Returns distance and fallback level (-1 none)."""
    d_out = np.full(len(qX), np.nan)
    lev_out = np.full(len(qX), -1, np.int8)
    for i in range(len(qX)):
        ok = (grp == qgrp[i]) & (day != qday[i])
        if exclude_day is not None:
            ok &= day != exclude_day[i]
        for lev, half in enumerate(SEASON_DAYS):
            cand = ok if half is None else ok & (np.abs(day - qday[i]) <= half)
            if cand.sum() >= min_ref:
                break
        else:
            continue
        dist = np.linalg.norm(X[cand] - qX[i], axis=1)
        d_out[i] = np.partition(dist, k - 1)[k - 1]
        lev_out[i] = lev
    return d_out, lev_out


def score_bouts(T: pd.DataFrame, M: np.ndarray, Dv: np.ndarray, Sm: np.ndarray, extra: dict | None = None):
    """Content, order and smoothness z of one bird's bouts (T: day, bin) and of extra queries scored as host bouts
    (extra: M, D, S, host index into T)."""
    day, grp = T["day"].to_numpy(), T["bin"].to_numpy()
    dc, lev = knn_ref_dist(M, day, grp, M, day, grp)
    do, _ = knn_ref_dist(Dv, day, grp, Dv, day, grp)
    zc, pc = robust_z(dc, grp)
    zo, po = robust_z(do, grp)
    zs, ps = robust_z(-Sm, grp)
    res = {"content_z": zc, "order_z": zo, "smooth_z": zs, "level": lev}
    if extra is not None and len(extra["host"]):
        h = extra["host"]
        ec, _ = knn_ref_dist(M, day, grp, extra["M"], day[h], grp[h])
        eo, _ = knn_ref_dist(Dv, day, grp, extra["D"], day[h], grp[h])
        res["extra"] = {"content_z": apply_z(ec, grp[h], pc), "order_z": apply_z(eo, grp[h], po),
                        "smooth_z": apply_z(-extra["S"], grp[h], ps)}
    return res


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """2 E|X-Y| - E|X-X'| - E|Y-Y'| with within-sample means over distinct pairs."""
    def pd_(A, B):
        return np.sqrt(np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T, 0))

    dxx, dyy = pd_(X, X), pd_(Y, Y)
    n, m = len(X), len(Y)
    return float(2 * pd_(X, Y).mean() - dxx.sum() / (n * (n - 1)) - dyy.sum() / (m * (m - 1)))


def day_pair_ed(QX: np.ndarray, QH: np.ndarray, Y: np.ndarray, YH: np.ndarray, rng, m: int = DAY_M) -> float:
    """Energy distance between two bird-days on their common hours, both sides sampled WITHOUT replacement: hour h
    can give at most min(nQ_h, nY_h) windows to each side; m slots are drawn without replacement from those
    per-hour capacities, and each side then takes that many distinct windows of each hour. Both samples have the
    same size and hour composition and no repeated window, so a day with few windows per hour (schedule change,
    low battery, missed bursts) is not made to look concentrated by duplicates. NaN if the capacities sum to < m."""
    common = np.intersect1d(np.unique(QH), np.unique(YH))
    q_by = {h: np.flatnonzero(QH == h) for h in common}
    y_by = {h: np.flatnonzero(YH == h) for h in common}
    cap = np.array([min(len(q_by[h]), len(y_by[h])) for h in common], int)
    if cap.sum() < m:
        return np.nan
    slots = np.repeat(np.arange(len(common)), cap)
    cnt = np.bincount(rng.choice(slots, m, replace=False), minlength=len(common))
    qi = np.concatenate([rng.choice(q_by[h], c, replace=False) for h, c in zip(common, cnt) if c])
    yi = np.concatenate([rng.choice(y_by[h], c, replace=False) for h, c in zip(common, cnt) if c])
    return energy_distance(QX[qi], Y[yi])


def eligible_days(day: np.ndarray, hour: np.ndarray) -> np.ndarray:
    """Days with >= DAY_M windows covering >= DAY_MIN_HOURS distinct local hours (partial days are not scored:
    a few hours of one day are autocorrelated and score high whatever the behaviour)."""
    if not len(day):
        return np.zeros(0, np.int64)
    g = pd.DataFrame({"d": day, "h": hour}).groupby("d")["h"].agg(["size", "nunique"])
    return g.index[(g["size"] >= DAY_M) & (g["nunique"] >= DAY_MIN_HOURS)].to_numpy()


def score_day(X: np.ndarray, day: np.ndarray, hour: np.ndarray, q_rows: np.ndarray, host_day: int, rng,
              exclude_day: int | None = None, qX: np.ndarray | None = None, q_hour: np.ndarray | None = None,
              days: np.ndarray | None = None):
    """Day score: the DAY_K-th smallest energy distance (day_pair_ed) between the query day and each other day of
    the bird among ``days`` (default: eligible_days) within +-45 d (fallback +-90 d, whole record; >= DAY_MIN_PAIRS days with a
    distance needed). Both sides of every comparison are single days of the same size and hour composition, so
    within-day autocorrelation and the day's coverage do not drive the score. Query = rows of X, or qX/q_hour.
    Returns (score, level): level -1 = too few comparable days."""
    QX = X[q_rows] if qX is None else qX
    QH = hour[q_rows] if q_hour is None else q_hour
    if len(QX) < DAY_M:
        return np.nan, -1
    days = eligible_days(day, hour) if days is None else np.asarray(days)
    days = days[days != host_day]
    if exclude_day is not None:
        days = days[days != exclude_day]
    dist = {}
    for lev, half in enumerate(SEASON_DAYS):
        cand = days if half is None else days[np.abs(days - host_day) <= half]
        for d_ in cand:
            if d_ not in dist:
                r = day == d_
                dist[d_] = day_pair_ed(QX, QH, X[r], hour[r], rng)
        v = np.array([dist[d_] for d_ in cand])
        v = v[np.isfinite(v)]
        if len(v) >= DAY_MIN_PAIRS:
            return float(np.partition(v, DAY_K - 1)[DAY_K - 1]), lev
    return np.nan, -1


def _edge(d: pd.DataFrame) -> pd.Series:
    return (d["segment_windows"] > 1) & ((d["k_in_segment"] == 0) | (d["k_in_segment"] == d["segment_windows"] - 1))


WINDOW_FACTORS = {
    "continuous mode (k=1)": (lambda d: d["sensitivity_k"] == 1, True),
    "GPS present": (lambda d: d["gps_fixes"] > 0, True),
    "partial GPS (1-9 fixes)": (lambda d: (d["gps_fixes"] > 0) & (d["gps_fixes"] < 10), True),
    "single-window segment": (lambda d: d["segment_windows"] <= 1, True),
    "segment edge, scheduled (first/last of >1, k=0)": (lambda d: (d["sensitivity_k"] == 0) & _edge(d), True),
    "segment edge, continuous (first/last of >1, k=1; take-off/landing)": (lambda d: (d["sensitivity_k"] == 1)
                                                                          & _edge(d), True),
    "190-199 samples": (lambda d: d["acc_samples"] < 200, True),
    "K=2 flight state": (lambda d: d["flight"], False),
    "night (sun < -6 deg)": (lambda d: d["sun_elev_deg"] < NIGHT_SUN, False),
}
BOUT_FACTORS = {
    "6-15 full windows": (lambda d: d["bin"] == 0, True),
    "16-45 full windows": (lambda d: d["bin"] == 1, True),
    "46-84 full windows": (lambda d: d["bin"] == 2, True),
    f"at the recording cap (>= {BOUT_CAP_FULL} full windows)": (lambda d: d["n_full"] >= BOUT_CAP_FULL, True),
    "bout with short windows": (lambda d: d["n_short"] > 0, True),
    "GPS on < 90% of full windows": (lambda d: d["gps_share"] < 0.9, True),
    "starts at night": (lambda d: d["night_start"], False),
}
DAY_FACTORS = {
    f"{DAY_M}-39 full windows": (lambda d: d["n_full"] < 40, True),
    "40-79 full windows": (lambda d: (d["n_full"] >= 40) & (d["n_full"] < 80), True),
    ">= 80 full windows": (lambda d: d["n_full"] >= 80, True),
    f"{DAY_MIN_HOURS}-14 distinct hours": (lambda d: d["n_hours"] <= 14, True),
    "day with continuous-mode bouts": (lambda d: d["n_k1"] > 0, True),
    "night share > 0.5": (lambda d: d["night_share"] > 0.5, False),
}


def bird_bouts(f: pd.DataFrame) -> pd.DataFrame:
    """Bouts of one bird (flight.bout_ids) with >= 60 s and >= BOUT_MIN_FULL full windows: table with the rows
    (positions in f) of the full windows in time order."""
    t_s = (f["t_start"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    order = np.argsort(t_s, kind="stable")
    bid = np.full(len(f), -1)
    bid[order] = bout_ids(t_s[order], (f["sensitivity_k"].to_numpy() == 1)[order], BOUT_STEP_S)
    df = pd.DataFrame({"bout": bid, "t": t_s, "full": f["acc_samples"].to_numpy() >= FULL_SAMPLES,
                       "gps": f["gps_fixes"].to_numpy() > 0, "pos": np.arange(len(f))})
    df = df[df["bout"] >= 0].sort_values(["bout", "t"], kind="stable")
    rows = []
    for b, g in df.groupby("bout", sort=True):
        dur = g["t"].iat[-1] - g["t"].iat[0] + 10.0
        full = g.loc[g["full"], "pos"].to_numpy()
        if dur < MIN_BOUT_S or len(full) < BOUT_MIN_FULL:
            continue
        rows.append({"bout": int(b), "rows": full, "n_full": len(full), "n_short": int((~g["full"]).sum()),
                     "gps_share": float(g.loc[g["full"], "gps"].mean()), "duration_s": float(dur)})
    T = pd.DataFrame(rows, columns=["bout", "rows", "n_full", "n_short", "gps_share", "duration_s"])
    if len(T):
        first = np.array([r[0] for r in T["rows"]])
        T["day"] = f["day"].to_numpy()[first]
        T["month"] = f["month"].to_numpy()[first]
        T["night_start"] = f["sun_elev_deg"].to_numpy()[first] < NIGHT_SUN
        T["bin"] = bout_bin(T["n_full"].to_numpy())
    return T


def behaviour_bird(b: str, f: pd.DataFrame, emb: Path, flight: np.ndarray, pca_day: tuple, k: int, cap: int,
                   min_ref: int, seed: int, bout_extra: dict, day_extra: list) -> dict:
    """Window, bout and day scores of one bird for one run, its injected queries and its level-2 samples."""
    H = norm_rows(np.load(emb / f"{b}.npz")["h"])
    if len(H) != len(f):
        raise ValueError("embedding rows differ from meta rows")
    full = f["acc_samples"].to_numpy() >= FULL_SAMPLES
    kk = f["sensitivity_k"].to_numpy()
    day, seg = f["day"].to_numpy(), f["segment"].to_numpy()
    # window level
    iw = np.flatnonzero(full)
    key = kk[iw] * 2 + flight[iw].astype(int)
    sc, lev = score_groups(H[iw], day[iw], seg[iw], key, k, cap, min_ref, seed_of(seed, b, "w"))
    z, _ = robust_z(sc, key)
    win = pd.DataFrame({"pos": iw, "z": z, "level": lev, "key": key})
    # bouts
    T = bird_bouts(f)
    bres = None
    if len(T):
        vec = [bout_vectors(H[r]) for r in T["rows"]]
        M, Dv, Sm = np.stack([v[0] for v in vec]), np.stack([v[1] for v in vec]), np.array([v[2] for v in vec])
        ex = None
        if bout_extra:
            ev = [bout_vectors(s) for s in bout_extra["seq"]]
            ex = {"M": np.stack([v[0] for v in ev]), "D": np.stack([v[1] for v in ev]), "S": np.array([v[2] for v in ev]),
                  "host": np.asarray(bout_extra["host"])}
        bres = score_bouts(T, M, Dv, Sm, ex)
    # days (scheduled, full)
    i0 = np.flatnonzero(full & (kk == 0) & np.isfinite(f["local_solar_hour"].to_numpy()))
    hour = np.floor(f["local_solar_hour"].to_numpy()[i0]).astype(int) % 24
    X = ((H[i0] - pca_day[0]) @ pca_day[1].T).astype(np.float32)
    dd = day[i0]
    rng = np.random.default_rng(seed_of(seed, b, "day"))
    drows = []
    elig = eligible_days(dd, hour)
    k1_days = set(day[kk == 1])
    night_w = f["sun_elev_deg"].to_numpy()[i0] < NIGHT_SUN
    cnt = pd.Series(dd).value_counts()
    n_partial = int(((cnt >= DAY_M) & ~cnt.index.isin(elig)).sum())
    for d_ in np.sort(elig):
        q = np.flatnonzero(dd == d_)
        s, lv = score_day(X, dd, hour, q, d_, rng, days=elig)
        drows.append({"day": int(d_), "n_full": len(q), "n_hours": int(np.unique(hour[q]).size),
                      "night_share": float(night_w[q].mean()), "n_k1": int(d_ in k1_days), "score": s, "level": lv})
    Dd = pd.DataFrame(drows, columns=["day", "n_full", "n_hours", "night_share", "n_k1", "score", "level"])
    ex_day = []
    for e in day_extra:  # {"host_day", "X" (PCA rows), "hour", "exclude_day", "kind"}
        s, _ = score_day(X, dd, hour, np.zeros(0, int), e["host_day"], rng, e.get("exclude_day"), e["X"], e["hour"],
                         days=elig)
        ex_day.append(s)
    # level-2 sample (bird level, so every format is used, each in its own cell): exactly L2_N windows per cell
    # (window format x mode x state) x season, and up to 4 L2_MONTH_N per (cell, year-month)
    l2 = {}
    srng = np.random.default_rng(seed_of(seed, b, "l2"))
    cell = l2_cell(f["acc_samples"].to_numpy(), kk, flight)
    seas = f["season"].to_numpy()
    for c in np.unique(cell):
        for s_ in range(4):
            ix = np.flatnonzero((cell == c) & (seas == s_))
            if len(ix) >= L2_N:
                l2[(int(c), s_)] = H[np.sort(srng.choice(ix, L2_N, replace=False))].astype(np.float16)
    ym = f["ym"].to_numpy()
    l2m = {}
    for c in np.unique(cell):
        for m_ in np.unique(ym[cell == c]):
            ix = np.flatnonzero((cell == c) & (ym == m_))
            if len(ix) >= L2_MONTH_N:
                pick = np.sort(srng.choice(ix, min(len(ix), 4 * L2_MONTH_N), replace=False))
                l2m[(int(c), m_)] = (H[pick].astype(np.float16), day[pick])
    return {"win": win, "bouts": T, "bres": bres, "days": Dd, "ex_day": ex_day, "l2": l2, "l2m": l2m,
            "n_partial_days": n_partial}


ACC_CLASSES = ("full", "40-sample", "64-sample", "other short")


def l2_cell(acc_samples: np.ndarray, k: np.ndarray, flight: np.ndarray) -> np.ndarray:
    a = np.asarray(acc_samples)
    cls = np.where(a >= FULL_SAMPLES, 0, np.where(a == 40, 1, np.where(a == 64, 2, 3)))
    return (cls * 2 + (np.asarray(k) == 1)) * 2 + np.asarray(flight, bool)


def l2_cell_name(c: int) -> str:
    return f"{ACC_CLASSES[c // 4]}|k{(c // 2) % 2}|{'flight' if c % 2 else 'ground'}"


def pick_bout_injections(BT: pd.DataFrame, H: dict, n: int, rng) -> pd.DataFrame:
    """Host bouts (bird-balanced) and their injected sequences: transplants from another bird's bout of the same
    length bin whose K=6 state shares differ from the host bout (L1 >= MIX_DIFF: diff_mix), differ from EVERY bout of
    the host bird in the same bin within +-45 d (novel_mix: a kind of bout this bird does not show then) or match
    the host bout (L1 <= MIX_SAME: same_mix, the identity control); reverse; shuffle."""
    if not len(BT):
        return pd.DataFrame(columns=["kind", "bird", "host", "seq"])
    _, inv, cnt = np.unique(BT["bird"].to_numpy(), return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    hosts = rng.choice(len(BT), min(n, len(BT)), replace=False, p=w / w.sum())
    shares = np.stack(BT["shares"].to_numpy())
    bird, bins, days = BT["bird"].to_numpy(), BT["bin"].to_numpy(), BT["day"].to_numpy()
    rows = []
    for j in hosts:
        hb, hr = bird[j], BT["rows"].iat[j]
        seq = H[hb][hr]
        rows.append({"kind": "reverse", "bird": hb, "host": int(BT["local"].iat[j]), "seq": seq[::-1]})
        rows.append({"kind": "shuffle", "bird": hb, "host": int(BT["local"].iat[j]), "seq": seq[rng.permutation(len(seq))]})
        l1 = np.abs(shares - shares[j]).sum(1)
        own = np.flatnonzero((bird == hb) & (bins == bins[j]) & (np.abs(days - days[j]) <= SEASON_DAYS[0]))
        own = own[own != j]
        other = np.flatnonzero((bird != hb) & (bins == bins[j]))
        novel = np.zeros(len(BT), bool)
        if len(own) and len(other):
            mind = np.abs(shares[other][:, None, :] - shares[own][None]).sum(2).min(1)
            novel[other[mind >= MIX_DIFF]] = True
        for kind, sel in (("transplant_diff_mix", l1 >= MIX_DIFF), ("transplant_novel_mix", novel),
                          ("transplant_same_mix", l1 <= MIX_SAME)):
            cand = np.flatnonzero(sel & (bird != hb) & (bins == bins[j]))
            if not len(cand):
                continue
            dj = rng.choice(cand)
            ds = H[bird[dj]][BT["rows"].iat[dj]]
            if len(ds) > len(seq):
                st = rng.integers(0, len(ds) - len(seq) + 1)
                ds = ds[st:st + len(seq)]
            rows.append({"kind": kind, "bird": hb, "host": int(BT["local"].iat[j]), "seq": ds})
    return pd.DataFrame(rows)


def pick_day_injections(DT: pd.DataFrame, F: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Host bird-days (>= DAY_M full k=0 windows; bird-balanced) and donors: same bird >= OTHER_SEASON_DAYS away;
    another bird within +-OTHER_BIRD_DOY days of the year."""
    if not len(DT):
        return pd.DataFrame(columns=["kind", "bird", "host_day", "donor_bird", "donor_day"])
    _, inv, cnt = np.unique(DT["bird"].to_numpy(), return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    hosts = rng.choice(len(DT), min(n, len(DT)), replace=False, p=w / w.sum())
    bird, day, doy = DT["bird"].to_numpy(), DT["day"].to_numpy(), DT["doy"].to_numpy()
    rows = []
    for j in hosts:
        same = np.flatnonzero((bird == bird[j]) & (np.abs(day - day[j]) >= OTHER_SEASON_DAYS))
        if len(same):
            dj = rng.choice(same)
            rows.append({"kind": "other_season", "bird": bird[j], "host_day": day[j], "donor_bird": bird[dj],
                         "donor_day": day[dj]})
        other = np.flatnonzero((bird != bird[j]) & (np.abs((doy - doy[j] + 182) % 365 - 182) <= OTHER_BIRD_DOY))
        if len(other):
            dj = rng.choice(other)
            rows.append({"kind": "other_bird", "bird": bird[j], "host_day": day[j], "donor_bird": bird[dj],
                         "donor_day": day[dj]})
    return pd.DataFrame(rows)


def mix_shares(st6: np.ndarray, rows, K: int) -> np.ndarray:
    s = st6[rows]
    s = s[s >= 0]
    return np.bincount(s, minlength=K) / max(len(s), 1)


def flight_state(st: pd.Series, speed: pd.Series) -> int:
    """The K=2 state with the higher median GPS speed (not assumed to be state 1)."""
    med = pd.Series(speed.to_numpy(), index=st.index).groupby(st.to_numpy()).median()
    return int(med.idxmax())


def _states(F: pd.DataFrame, s: Path, k_expected: int | None) -> np.ndarray:
    st = pd.read_parquet(s / "states.parquet", columns=["bird", "row", "state_hmm"])
    st["bird"] = st["bird"].astype(str)
    st = F[["bird", "row"]].merge(st, on=["bird", "row"], how="left")
    if k_expected == 2:
        if st["state_hmm"].isna().any():
            raise ValueError(f"{s}: {int(st['state_hmm'].isna().sum())} windows without a K=2 state")
        if st["state_hmm"].nunique() != 2:
            raise ValueError(f"{s}: expected K=2 states")
    return st["state_hmm"].fillna(-1).to_numpy().astype(int)


def rate_rows(level: str, kind: str, z: np.ndarray, birds: np.ndarray, thr, clean: np.ndarray, rng,
              score: str) -> dict:
    """Recall / false-positive rate at ``thr`` (scalar or one threshold per row), AUC against ``clean``."""
    z = np.asarray(z, float)
    ok = np.isfinite(z)
    thr_a = np.broadcast_to(np.asarray(thr, float), z.shape)
    r, lo, hi = boot_rate(z[ok] > thr_a[ok], birds[ok], rng) if ok.any() else (np.nan,) * 3
    return {"level": level, "injection": kind, "score": score, "n_injected": len(z), "n_scored": int(ok.sum()),
            "n_birds": int(pd.unique(birds[ok]).size), "flag_rate": r, "ci95_lo": lo, "ci95_hi": hi,
            "auc_vs_clean_hosts": auc(z, clean), "threshold_z": float(thr) if np.ndim(thr) == 0 else "per stratum",
            "role": "false-positive rate" if kind == "clean" else "recall"}


def coverage(F: pd.DataFrame, W: pd.DataFrame, B: pd.DataFrame, Dd: pd.DataFrame | None) -> dict:
    """What gets no behavioural score (I25): windows without a window-level score (short windows are excluded by
    design, full windows can lack a reference), and bird-days and birds with no score at any level (no scored
    window, no scored bout starting that day, no day score)."""
    n = len(F)
    short = int((F["acc_samples"].to_numpy() < FULL_SAMPLES).sum())
    okw = np.isfinite(W["z"].to_numpy())
    keys = [W.loc[okw, ["bird", "day"]]]
    if len(B):
        keys.append(B.loc[np.isfinite(B["content_z"]) | np.isfinite(B["seq_z"]), ["bird", "day"]])
    if Dd is not None:
        keys.append(Dd.loc[np.isfinite(Dd["z"]), ["bird", "day"]])
    scored = pd.concat(keys).drop_duplicates()
    all_days = F[["bird", "day"]].drop_duplicates()
    n_days_scored = len(all_days.merge(scored, on=["bird", "day"]))
    n_day_level = int(np.isfinite(Dd["z"]).sum()) if Dd is not None else 0
    birds_any = set(scored["bird"])
    return {"n_windows": n, "n_short": short, "share_short": short / max(n, 1),
            "n_full_unscored": int((~okw).sum()),
            "share_windows_no_window_score": (n - int(okw.sum())) / max(n, 1),
            "n_bird_days": len(all_days), "n_bird_days_no_score": len(all_days) - n_days_scored,
            "share_bird_days_no_score": (len(all_days) - n_days_scored) / max(len(all_days), 1),
            "share_bird_days_no_day_score": (len(all_days) - n_day_level) / max(len(all_days), 1),
            "n_birds": int(F["bird"].nunique()), "n_birds_no_score": int(F["bird"].nunique() - len(birds_any)),
            "n_birds_no_window_score": int(F["bird"].nunique() - W.loc[okw, "bird"].nunique()),
            "n_birds_no_bout_score": int(F["bird"].nunique() - (B.loc[np.isfinite(B["content_z"]), "bird"].nunique()
                                                                if len(B) else 0)),
            "n_birds_no_day_score": int(F["bird"].nunique() - (Dd.loc[np.isfinite(Dd["z"]), "bird"].nunique()
                                                               if Dd is not None else 0))}


def run_behaviour(F: pd.DataFrame, run: Path, s2: Path, s6: Path, out: Path, k: int, cap: int, min_ref: int,
                  n_inject: int, workers: int, seed: int, rng, top_bouts: int, windows: Path) -> dict:
    name = run.name
    emb = run / "embeddings"
    st2 = _states(F, s2, 2)
    fs = flight_state(pd.Series(st2), F["speed_mean"])
    flight = st2 == fs
    st6 = _states(F, s6, None)
    K6 = int(st6.max()) + 1
    starts = F.groupby("bird", sort=False).indices
    # all h of the run (float16) for injections and the day PCA
    H = {b: norm_rows(np.load(emb / f"{b}.npz")["h"]).astype(np.float16) for b in starts}
    full = F["acc_samples"].to_numpy() >= FULL_SAMPLES
    k0 = full & (F["sensitivity_k"].to_numpy() == 0)
    fit = []
    for b, ix in starts.items():
        i = np.flatnonzero(k0[ix])
        if len(i):
            fit.append(H[b][rng.choice(i, min(2000, len(i)), replace=False)].astype(np.float32))
    pca = PCA(DAY_PCS, random_state=seed).fit(np.concatenate(fit))
    pca_day = (pca.mean_.astype(np.float32), pca.components_.astype(np.float32))
    # bouts of every bird (metadata only) for injections
    bts = []
    for b, ix in starts.items():
        T = bird_bouts(F.iloc[ix].reset_index(drop=True))
        if len(T):
            T = T.assign(bird=b, local=np.arange(len(T)))
            T["shares"] = [mix_shares(st6[ix], r, K6) for r in T["rows"]]
            bts.append(T)
    BT = pd.concat(bts, ignore_index=True) if bts else pd.DataFrame()
    BI = pick_bout_injections(BT, {b: H[b].astype(np.float32) for b in BT["bird"].unique()} if len(BT) else {},
                              n_inject, rng)
    # day injections
    dts = []
    for b, ix in starts.items():
        sh = F["local_solar_hour"].to_numpy()
        i = ix[k0[ix] & np.isfinite(sh[ix])]
        dd = np.sort(eligible_days(F["day"].to_numpy()[i], np.floor(sh[i]).astype(int) % 24))
        if len(dd):
            doy = F.iloc[i].groupby("day")["t_start"].first().dt.dayofyear.reindex(dd).to_numpy()
            dts.append(pd.DataFrame({"bird": b, "day": dd, "doy": doy}))
    DT = pd.concat(dts, ignore_index=True) if dts else pd.DataFrame(columns=["bird", "day", "doy"])
    DI = pick_day_injections(DT, F, n_inject, rng)
    day_extra = {b: [] for b in starts}
    for _, e in DI.iterrows():
        ix = starts[e["donor_bird"]]
        i = ix[k0[ix] & (F["day"].to_numpy()[ix] == e["donor_day"]) & np.isfinite(F["local_solar_hour"].to_numpy()[ix])]
        Xd = ((H[e["donor_bird"]][i - ix[0]].astype(np.float32) - pca_day[0]) @ pca_day[1].T)
        hr = np.floor(F["local_solar_hour"].to_numpy()[i]).astype(int) % 24
        day_extra[e["bird"]].append({"host_day": e["host_day"], "X": Xd, "hour": hr, "kind": e["kind"],
                                     "exclude_day": e["donor_day"] if e["kind"] == "other_season" else None})
    bout_extra = {}
    for b, g in BI.groupby("bird"):
        bout_extra[b] = {"seq": list(g["seq"]), "host": g["host"].to_numpy(), "kind": g["kind"].to_numpy()}
    del H
    log(f"{name}: scoring pass ({len(BT)} bouts, {len(DT)} bird-days, {len(BI)} bout and {len(DI)} day injections)")
    res = Parallel(n_jobs=workers)(delayed(behaviour_bird)(
        b, F.iloc[ix].reset_index(drop=True), emb, flight[ix], pca_day, k, cap, min_ref, seed_of(seed, name),
        {k_: v for k_, v in bout_extra.get(b, {}).items() if k_ != "kind"}, day_extra[b]) for b, ix in starts.items())
    log(f"{name}: scoring pass done")
    first_day = F.groupby("bird")["day"].min()
    summ = {"flight_state": fs, "K6": K6}
    # ---- window level
    W = []
    for (b, ix), r in zip(starts.items(), res):
        w = r["win"]
        W.append(F.iloc[ix[w["pos"].to_numpy()]][["bird", "row", "day", "month", "sensitivity_k", "gps_fixes",
                                                     "segment_windows", "k_in_segment", "acc_samples",
                                                     "sun_elev_deg"]].assign(
            z=w["z"].to_numpy(), level=w["level"].to_numpy(), flight=flight[ix[w["pos"].to_numpy()]],
            key=w["key"].to_numpy()))
    W = pd.concat(W, ignore_index=True)
    W["rel_day"] = W["day"] - first_day.reindex(W["bird"]).to_numpy()
    ok = np.isfinite(W["z"].to_numpy())
    thr_w = float(np.percentile(W["z"].to_numpy()[ok], FLAG_Q))
    Ws = W[ok].reset_index(drop=True)
    pooled = Ws["z"].to_numpy() > thr_w
    strat, thr_strat = stratum_flags(Ws["z"].to_numpy(), Ws["key"].to_numpy())
    grp = Ws["bird"].astype(str).to_numpy() + "|" + Ws["key"].astype(str).to_numpy()
    pergrp = group_quantile_flag(Ws["z"].to_numpy(), grp)
    ft_s = factor_table(Ws, strat, WINDOW_FACTORS).assign(threshold="per mode x state, pooled over birds (primary)")
    ft_p = factor_table(Ws, pooled, WINDOW_FACTORS).assign(threshold="one pooled threshold")
    ft_g = factor_table(Ws, pergrp, WINDOW_FACTORS).assign(threshold="per bird x mode x state")
    pd.concat([ft_s, ft_p, ft_g]).to_csv(out / f"beh_window_factors_{name}.csv", index=False)
    top = Ws.assign(top1pct=strat).nlargest(2000, "z")
    top[["bird", "row", "rel_day", "month", "sensitivity_k", "flight", "z", "top1pct"]].to_csv(
        out / f"beh_top_windows_{name}.csv", index=False)
    lev_counts = {int(l_): int((W["level"] == l_).sum()) for l_ in (-1, 0, 1, 2)}
    cols = ["factor", "format_factor", "share_scored", "ratio", "n_flagged_with", "birds_flagged_with"]
    summ["window"] = {"n_full": len(W), "n_scored": int(ok.sum()), "levels": lev_counts, "threshold_z": thr_w,
                      "thresholds_stratum": {f"k{k_ // 2}|{'flight' if k_ % 2 else 'ground'}": v
                                             for k_, v in thr_strat.items()},
                      "n_flagged": int(strat.sum()), "birds_flagged": int(Ws.loc[strat, "bird"].nunique()),
                      "worst_ratio_stratum": worst_format_ratio(ft_s), "worst_ratio_pooled": worst_format_ratio(ft_p),
                      "worst_ratio_per_group": worst_format_ratio(ft_g),
                      "factors_stratum": ft_s[cols].to_dict("records"), "factors_pooled": ft_p[cols].to_dict("records"),
                      "factors_per_group": ft_g[cols].to_dict("records")}
    # ---- bouts
    B, EXB = [], []
    for (b, ix), r in zip(starts.items(), res):
        T = r["bouts"]
        if not len(T):
            continue
        br = r["bres"]
        t = T.drop(columns="rows").assign(bird=b, content_z=br["content_z"], order_z=br["order_z"],
                                          smooth_z=br["smooth_z"], level=br["level"],
                                          first_row=[F["row"].to_numpy()[ix[rr[0]]] for rr in T["rows"]],
                                          last_row=[F["row"].to_numpy()[ix[rr[-1]]] for rr in T["rows"]])
        B.append(t)
        if "extra" in br:
            ex = bout_extra[b]
            EXB.append(pd.DataFrame({"bird": b, "kind": ex["kind"], "host": ex["host"], **br["extra"]}))
    B = pd.concat(B, ignore_index=True) if B else pd.DataFrame()
    if len(B):
        B["seq_z"] = np.fmax(B["order_z"], B["smooth_z"])
        B["rel_day"] = B["day"] - first_day.reindex(B["bird"]).to_numpy()
        # primary flags: 99th percentile per length bin, pooled over birds (bin-neutral by construction);
        # one pooled threshold reported for comparison
        B["top1pct_content"], thr_c = stratum_flags(B["content_z"].to_numpy(), B["bin"].to_numpy())
        B["top1pct_seq"], thr_s = stratum_flags(B["seq_z"].to_numpy(), B["bin"].to_numpy())
        thr_o = stratum_flags(B["order_z"].to_numpy(), B["bin"].to_numpy())[1]
        thr_m = stratum_flags(B["smooth_z"].to_numpy(), B["bin"].to_numpy())[1]
        B.drop(columns=["day"]).to_csv(out / f"beh_bouts_{name}.csv", index=False)
        okc, oks = np.isfinite(B["content_z"]), np.isfinite(B["seq_z"])
        ft = []
        for sc, fl, ok_ in (("content", "top1pct_content", okc), ("sequence", "top1pct_seq", oks)):
            zc = B[f"{'seq' if sc == 'sequence' else sc}_z"].to_numpy()
            ft.append(factor_table(B[ok_], B.loc[ok_, fl], BOUT_FACTORS).assign(score=sc, threshold="per bin (primary)"))
            ft.append(factor_table(B[ok_], zc[ok_] > np.nanpercentile(zc, FLAG_Q), BOUT_FACTORS).assign(
                score=sc, threshold="one pooled threshold"))
        FT = pd.concat(ft)
        FT.to_csv(out / f"beh_bout_factors_{name}.csv", index=False)
        cols = ["factor", "format_factor", "share_scored", "ratio", "n_flagged_with"]
        pick = lambda sc, th: FT[(FT["score"] == sc) & (FT["threshold"] == th)]  # noqa: E731
        summ["bout"] = {"n_bouts": len(B), "n_scored_content": int(okc.sum()), "n_scored_seq": int(oks.sum()),
                        "n_birds": int(B["bird"].nunique()), "n_by_bin": B["bin"].value_counts().sort_index().tolist(),
                        "n_flag_content": int(B["top1pct_content"].sum()), "n_flag_seq": int(B["top1pct_seq"].sum()),
                        **{f"worst_ratio_{sc}_{t}": worst_format_ratio(pick(sc, th)) for sc in ("content", "sequence")
                           for t, th in (("primary", "per bin (primary)"), ("pooled", "one pooled threshold"))},
                        "factors_content": pick("content", "per bin (primary)")[cols].to_dict("records"),
                        "factors_seq": pick("sequence", "per bin (primary)")[cols].to_dict("records"),
                        "factors_content_pooled": pick("content", "one pooled threshold")[cols].to_dict("records"),
                        "factors_seq_pooled": pick("sequence", "one pooled threshold")[cols].to_dict("records")}
    # ---- days
    Dl = []
    for (b, ix), r in zip(starts.items(), res):
        if len(r["days"]):
            Dl.append(r["days"].assign(bird=b))
    Dd = pd.concat(Dl, ignore_index=True) if Dl else pd.DataFrame()
    if len(Dd):
        Dd["z"], dpar = robust_z(Dd["score"].to_numpy(), Dd["bird"].to_numpy())
        Dd["rel_day"] = Dd["day"] - first_day.reindex(Dd["bird"]).to_numpy()
        mo = F.groupby(["bird", "day"])["month"].first()
        Dd["month"] = mo.reindex(pd.MultiIndex.from_arrays([Dd["bird"], Dd["day"]])).to_numpy()
        okd = np.isfinite(Dd["z"])
        thr_d = float(np.nanpercentile(Dd["z"], FLAG_Q))
        Dd["top1pct"] = Dd["z"] > thr_d
        Dd.drop(columns=["day"]).to_csv(out / f"beh_days_{name}.csv", index=False)
        ftd = factor_table(Dd[okd], Dd.loc[okd, "top1pct"], DAY_FACTORS)
        ftd.to_csv(out / f"beh_day_factors_{name}.csv", index=False)
        summ["day"] = {"n_days": len(Dd), "n_scored": int(okd.sum()), "n_birds": int(Dd["bird"].nunique()),
                       "n_too_few_days": int((Dd["level"] == -1).sum()), "n_flagged": int(Dd["top1pct"].sum()),
                       "n_partial_days_not_scored": int(sum(r["n_partial_days"] for r in res)),
                       "threshold_z": thr_d, "worst_ratio": worst_format_ratio(ftd),
                       "factors": ftd[["factor", "format_factor", "share_scored", "ratio",
                                       "n_flagged_with"]].to_dict("records")}
    summ["coverage"] = coverage(F, W, B, Dd if len(Dd) else None)
    # ---- injections
    rows = []
    if len(B) and EXB:
        E = pd.concat(EXB, ignore_index=True)
        E["seq_z"] = np.fmax(E["order_z"], E["smooth_z"])
        Bk = B.set_index(["bird", B.groupby("bird").cumcount()])
        clean_c = Bk["content_z"].reindex(pd.MultiIndex.from_arrays([E["bird"], E["host"]])).to_numpy()
        clean_s = Bk["seq_z"].reindex(pd.MultiIndex.from_arrays([E["bird"], E["host"]])).to_numpy()
        hb = E[["bird", "host"]].drop_duplicates()
        cc = Bk["content_z"].reindex(pd.MultiIndex.from_arrays([hb["bird"], hb["host"]])).to_numpy()
        cs = Bk["seq_z"].reindex(pd.MultiIndex.from_arrays([hb["bird"], hb["host"]])).to_numpy()
        hbin_c = Bk["bin"].reindex(pd.MultiIndex.from_arrays([hb["bird"], hb["host"]])).to_numpy()
        tb = lambda thr, bins: np.array([thr.get(x, np.inf) for x in bins])  # noqa: E731
        rows.append(rate_rows("bout", "clean", cc, hb["bird"].to_numpy(), tb(thr_c, hbin_c), cc, rng, "content"))
        rows.append(rate_rows("bout", "clean", cs, hb["bird"].to_numpy(), tb(thr_s, hbin_c), cs, rng, "sequence"))
        hbin = Bk["bin"].reindex(pd.MultiIndex.from_arrays([E["bird"], E["host"]])).to_numpy()
        for kind, g in E.groupby("kind"):
            m = (E["kind"] == kind).to_numpy()
            hbk = hbin[m]
            if kind.startswith("transplant"):
                rows.append(rate_rows("bout", kind, g["content_z"].to_numpy(), g["bird"].to_numpy(), tb(thr_c, hbk),
                                      clean_c[m], rng, "content"))
            else:
                rows.append(rate_rows("bout", kind, g["seq_z"].to_numpy(), g["bird"].to_numpy(), tb(thr_s, hbk),
                                      clean_s[m], rng, "sequence"))
                for bn in range(len(BOUT_BINS)):
                    mb = hbk == bn
                    if mb.any():
                        rows.append(rate_rows("bout", f"{kind} (bin {bn}: {BOUT_BINS[bn]}+ full windows)",
                                              g["seq_z"].to_numpy()[mb], g["bird"].to_numpy()[mb], tb(thr_s, hbk[mb]),
                                              clean_s[m][mb], rng, "sequence"))
                for sc, th in (("order_z", thr_o), ("smooth_z", thr_m)):
                    rows.append(rate_rows("bout", kind, g[sc].to_numpy(), g["bird"].to_numpy(), tb(th, hbk),
                                          Bk[sc].reindex(pd.MultiIndex.from_arrays([g["bird"], g["host"]])).to_numpy(),
                                          rng, sc))
    if len(Dd) and len(DI):
        zmap = Dd.set_index(["bird", "day"])
        ez, kinds, eb, clean = [], [], [], []
        for (b, ix), r in zip(starts.items(), res):
            for e, s in zip(day_extra[b], r["ex_day"]):
                m_, s_ = dpar.get(b, (np.nan, np.nan))
                ez.append((s - m_) / s_ if s_ and s_ > 0 else np.nan)
                kinds.append(e["kind"])
                eb.append(b)
                clean.append(zmap["z"].get((b, e["host_day"]), np.nan))
        ez, kinds, eb, clean = map(np.asarray, (ez, kinds, eb, clean))
        hd = DI[["bird", "host_day"]].drop_duplicates()
        cz = np.array([zmap["z"].get((b, d), np.nan) for b, d in zip(hd["bird"], hd["host_day"])])
        rows.append(rate_rows("day", "clean", cz, hd["bird"].to_numpy(), thr_d, cz, rng, "energy distance"))
        for kind in np.unique(kinds):
            m = kinds == kind
            rows.append(rate_rows("day", kind, ez[m], eb[m], thr_d, clean[m], rng, "energy distance"))
    INJ = pd.DataFrame(rows)
    INJ.assign(run=name).to_csv(out / f"beh_inject_{name}.csv", index=False)
    summ["inject"] = INJ.to_dict("records")
    # ---- blind pages: top content bouts (at most 3 per bird)
    if len(B) and top_bouts:
        tops = B[B["top1pct_content"]].sort_values("content_z", ascending=False).groupby("bird", sort=False).head(3)
        tops = tops.head(top_bouts).reset_index(drop=True)
        Rmap = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3)
                for o in json.loads((run / "orientation.json").read_text())} if (run / "orientation.json").exists() else {}
        pays, key = [], []
        for i, t in tops.iterrows():
            ix = starts[t["bird"]]
            T = res[list(starts).index(t["bird"])]["bouts"]
            rr = T.loc[T["bout"] == t["bout"], "rows"].iat[0]
            pays.append(bout_payload(F.iloc[ix[rr]], windows, Rmap.get(t["bird"], np.eye(3, dtype=np.float32)), i + 1,
                                     int(t["month"])))
            key.append({"page_entry": i + 1, "bird": t["bird"], "bout": int(t["bout"]), "rel_day": int(t["rel_day"]),
                        "month": int(t["month"]), "content_z": float(t["content_z"]), "n_full": int(t["n_full"])})
        for p in range(0, len(pays), 4):
            blind_page(pays[p:p + 4], out / f"blind_bouts_{name}_p{p // 4 + 1}.png")
        pd.DataFrame(key).to_csv(out / f"blind_key_{name}.csv", index=False)
    # ---- level 2
    l2b, l2s, l2m = level2([r["l2"] for r in res], [r["l2m"] for r in res], list(starts), name, rng, workers, seed)
    l2b.to_csv(out / f"l2_cells_{name}.csv", index=False)
    l2s.to_csv(out / f"l2_birds_{name}.csv", index=False)
    l2m.to_csv(out / f"l2_months_{name}.csv", index=False)
    summ["l2"] = {"n_cells_scored": len(l2b), "n_cells": int(l2b["cell"].nunique()) if len(l2b) else 0,
                  "n_birds": int(l2b["bird"].nunique()) if len(l2b) else 0,
                  "n_birds_scored": int(l2s["l2_score"].notna().sum()) if len(l2s) else 0,
                  "n_month_pairs": len(l2m),
                  "n_month_pairs_consecutive": int((l2m["gap_months"] == 1).sum()) if len(l2m) else 0}
    return {"summary": summ, "l2_scores": l2s, "l2_months": l2m}


# ---------------------------------------------------------------- blind pages


def blind_page(entries: list[dict], path: Path) -> plt.Figure:
    """One page of blind plots. Each entry: title, t_rel (s, per window), acc (list of (200, 3) arrays, NaN where
    absent), gps_speed / gps_climb (list of (10,) arrays). No bird id, date or score is drawn."""
    fig, axes = plt.subplots(len(entries), 2, figsize=(14, 2.6 * len(entries) + 0.4), squeeze=False,
                             layout="constrained")
    for (ax_a, ax_g), e in zip(axes, entries):
        for t0, a in zip(e["t_rel"], e["acc"]):
            t = t0 + np.arange(a.shape[0]) / 20.0
            for j, (c, lab) in enumerate(zip(("tab:red", "tab:green", "tab:blue"), ("x", "y", "z"))):
                ax_a.plot(t, a[:, j], color=c, lw=0.4, label=lab if t0 == e["t_rel"][0] else None)
        for t0, sp, cl in zip(e["t_rel"], e["gps_speed"], e["gps_climb"]):
            t = t0 + np.arange(len(sp))
            ax_g.plot(t, sp, "k.-", ms=2, lw=0.6)
            ax_g.plot(t, cl, ".-", color="tab:purple", ms=2, lw=0.6)
        ax_a.set_title(e["title"], fontsize=8)
        ax_a.set_ylabel("acc (g), gravity-aligned")
        ax_g.set_ylabel("GPS speed (black), climb (purple), m/s", fontsize=7)
        ax_a.legend(fontsize=6, ncol=3, loc="upper right")
    axes[-1, 0].set_xlabel("seconds from the bout's first full window")
    axes[-1, 1].set_xlabel("seconds from the bout's first full window")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return fig


def bout_payload(rows: pd.DataFrame, windows: Path, R: np.ndarray, rank: int, month: int) -> dict:
    a, am, g, gm = read_rows(windows, rows["bird"].to_numpy(), rows["row"].to_numpy())
    a = a @ R.T
    a[~am] = np.nan
    si, ci = GPS_CHANNELS.index("ground-speed"), GPS_CHANNELS.index("climb_ms")
    t0 = rows["t_start"].iat[0]
    return {"title": f"Bout {rank}: month {MONTHS[month - 1]}, {len(rows)} full windows",
            "t_rel": (rows["t_start"] - t0).dt.total_seconds().tolist(), "acc": list(a),
            "gps_speed": [np.where(gm[i, :, si], g[i, :, si], np.nan) for i in range(len(rows))],
            "gps_climb": [np.where(gm[i, :, ci], g[i, :, ci], np.nan) for i in range(len(rows))]}


# ================================================================ level 2


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


def cell_mmd(X: dict, gamma: float, rng) -> dict:
    """{bird: MMD^2 of the bird's rows against 2 L2_N rows of the other birds, each bird weighted equally}."""
    birds = list(X)
    out = {}
    for b in birds:
        others = [o for o in birds if o != b]
        if not others:
            continue
        per = np.bincount(rng.integers(0, len(others), 2 * L2_N), minlength=len(others))
        ref = np.concatenate([X[o][rng.choice(len(X[o]), c, replace=False)] for o, c in zip(others, per) if c])
        out[b] = mmd2(X[b], ref, gamma)
    return out


L2_CELL_COLS = ["bird", "cell", "mmd2", "rank_in_cell", "z_across_birds", "n_birds_cell"]
L2_BIRD_COLS = ["bird", "l2_score", "n_cells"]


def l2_bird_scores(jobs: list, res: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell table and bird scores from per-cell {bird: MMD^2}. Within each cell (>= L2_MIN_BIRDS birds) a bird's
    position is its rank among the cell's birds as a fraction in (0, 1) ((rank - 0.5) / n), which has the same null
    distribution in every cell whatever its size (a robust z in a thin cell does not). Bird score = median of its
    within-cell ranks over its cells (NaN with fewer than L2_MIN_CELLS cells), so no single cell decides it.
    z_across_birds is kept for reading only."""
    rows = []
    for (c, _), r in zip(jobs, res):
        b_ = list(r)
        v = np.array([r[b] for b in b_], float)
        ok = np.isfinite(v)
        if ok.sum() < L2_MIN_BIRDS:
            continue
        rk = np.full(len(v), np.nan)
        rk[ok] = (stats.rankdata(v[ok]) - 0.5) / ok.sum()
        med = np.median(v[ok])
        sc = max(1.4826 * np.median(np.abs(v[ok] - med)), 1e-12)
        for b, m, q in zip(b_, v, rk):
            rows.append({"bird": b, "cell": f"{l2_cell_name(c[0])}|{SEASONS[c[1]]}", "mmd2": m, "rank_in_cell": q,
                         "z_across_birds": (m - med) / sc, "n_birds_cell": int(ok.sum())})
    Bc = pd.DataFrame(rows, columns=L2_CELL_COLS)
    srows = []
    for b, g in Bc.groupby("bird"):
        q = g["rank_in_cell"].to_numpy(float)
        q = q[np.isfinite(q)]
        srows.append({"bird": b, "l2_score": float(np.median(q)) if len(q) >= L2_MIN_CELLS else np.nan,
                      "n_cells": len(q)})
    return Bc, pd.DataFrame(srows, columns=L2_BIRD_COLS)


def level2(samples: list[dict], msamples: list[dict], birds: list[str], name: str, rng, workers: int, seed: int):
    """Bird x cell MMD^2 z across birds, bird scores (fixed number of cells and all cells), month changes."""
    cells = sorted({c for s in samples for c in s})
    if not cells:
        return pd.DataFrame(columns=L2_CELL_COLS), pd.DataFrame(columns=L2_BIRD_COLS), pd.DataFrame()
    allx = np.concatenate([s[c] for s in samples for c in s]).astype(np.float32)
    fit = allx[rng.choice(len(allx), min(50000, len(allx)), replace=False)]
    pca = PCA(L2_PCS, random_state=seed).fit(fit)
    gamma = median_gamma(pca.transform(fit), rng)
    jobs = []
    for c in cells:
        X = {b: pca.transform(s[c].astype(np.float32)) for b, s in zip(birds, samples) if c in s}
        if len(X) >= L2_MIN_BIRDS:
            jobs.append((c, X))
    res = Parallel(n_jobs=workers)(delayed(cell_mmd)(X, gamma, np.random.default_rng(seed_of(seed, name, c)))
                                   for c, X in jobs)
    Bc, Bs = l2_bird_scores(jobs, res)
    # month change (equal sizes)
    mrows = []
    for b, ms in zip(birds, msamples):
        r_ = np.random.default_rng(seed_of(seed, name, b, "month"))
        by_cell = {}
        for (c, m), v in ms.items():
            by_cell.setdefault(c, []).append((m, v))
        for c, lst in by_cell.items():
            lst.sort()
            prep = {}
            for m, (h, d) in lst:
                X = pca.transform(h.astype(np.float32))
                prep[m] = (X, d)
            months = [m for m, _ in lst]
            for a, b_ in zip(months[:-1], months[1:]):
                Xa, da = prep[a]
                Xb, db = prep[b_]
                d_ = mmd2(Xa[r_.choice(len(Xa), L2_MONTH_N, replace=False)], Xb[r_.choice(len(Xb), L2_MONTH_N, replace=False)], gamma)
                fl = []
                for X, dd in (prep[a], prep[b_]):
                    o, e = X[dd % 2 == 1], X[dd % 2 == 0]
                    if min(len(o), len(e)) >= L2_MONTH_N // 2:
                        fl.append(mmd2(o[r_.choice(len(o), L2_MONTH_N // 2, replace=False)],
                                       e[r_.choice(len(e), L2_MONTH_N // 2, replace=False)], gamma))
                f_ = float(np.mean(fl)) if fl else np.nan
                ya, ma = map(int, a.split("-"))
                yb, mb = map(int, b_.split("-"))
                mrows.append({"bird": b, "cell": l2_cell_name(c), "month_from_m": ma,
                              "gap_months": (yb - ya) * 12 + mb - ma, "mmd2": d_, "floor": f_,
                              "excess": d_ - f_ if np.isfinite(f_) else np.nan})
    M = pd.DataFrame(mrows)
    if len(M):
        # only consecutive calendar months (gap 1) are changes; pairs across a gap are kept in the table, unscored
        M["z_change"] = np.nan
        c1 = M["gap_months"] == 1
        M.loc[c1, "z_change"] = M[c1].groupby("cell")["excess"].transform(
            lambda v: (v - v.median()) / max(1.4826 * (v - v.median()).abs().median(), 1e-12))
    return Bc, Bs, M


# ================================================================ bird table and nuisance


def bird_table(F: pd.DataFrame, reference_csv: Path | None) -> pd.DataFrame:
    g = F.groupby("bird")
    first, last = g["t_start"].min(), g["t_start"].max()
    B = pd.DataFrame({"n_windows": g.size(), "track_days": (last - first).dt.total_seconds() / 86400,
                      "lsh_missing_share": g["lsh_filled"].mean() if "lsh_filled" in F else np.nan,
                      "cont_share": g["sensitivity_k"].apply(lambda s: float((s == 1).mean())),
                      "gps_share": g["gps_fixes"].apply(lambda s: float((s > 0).mean())),
                      "short_share": g["acc_samples"].apply(lambda s: float((s < FULL_SAMPLES).mean()))})
    if reference_csv is not None:
        from . import reference

        ref = reference.load_birds(reference_csv, list(B.index)).set_index("bird")
        # cohort = the real deployment year (earliest deployment's deploy-on), not the first window's year
        dep = pd.to_datetime(ref["deploy_on"], utc=True).reindex(B.index)
        B["deploy_year"] = dep.dt.year.astype(float)
        B["deploy_month"] = dep.dt.month.astype(float)
        B["sex_f"] = ref["sex"].map({"f": 1.0, "m": 0.0}).reindex(B.index)
        B["migrant"] = ref["migrant"].notna().astype(float).reindex(B.index)
        B["tagged_young"] = ref["tagged_young"].notna().astype(float).reindex(B.index)
    B.index.name = "bird"
    return B


NUISANCE = ["deploy_year", "deploy_month", "track_days", "n_windows", "cont_share", "gps_share", "short_share",
            "sex_f", "migrant", "tagged_young"]
CONTROLS = ["track_days", "n_windows", "deploy_year", "deploy_month"]  # rank controls of the partial version


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p over the finite entries (NaN stays NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, np.maximum.accumulate(p[o] * (len(o) - np.arange(len(o)))))
    out[o] = adj
    return out


def fmt_p(p, n_perm: int) -> str:
    """p for printing: '<= 1/(n_perm+1)' at the permutation floor (M8), 'n/a' when not computed."""
    if p is None or not np.isfinite(p):
        return "n/a"
    floor = 1.0 / (n_perm + 1)
    return f"<= {floor:.2g}" if p <= floor * (1 + 1e-9) else f"{p:.3g}"


def perm_spearman(x: np.ndarray, y: np.ndarray, n_perm: int, rng) -> tuple[float, float, int]:
    """Spearman rho between bird-level x and y, p from permuting y between birds (two-sided)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, int(len(x))
    rho = stats.spearmanr(x, y)[0]
    null = np.array([stats.spearmanr(x, rng.permutation(y))[0] for _ in range(n_perm)])
    return float(rho), float((1 + (np.abs(null) >= abs(rho) - 1e-12).sum()) / (n_perm + 1)), int(len(x))


def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray, n_perm: int, rng) -> tuple[float, float, int]:
    """Spearman partial correlation of x and y given the columns of Z: ranks of x and y residualised on the ranks
    of Z (least squares with intercept), Pearson r of the residuals; p from permuting y's residuals between birds."""
    Z = np.atleast_2d(np.asarray(Z, float).T).T if np.ndim(Z) == 1 else np.asarray(Z, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    if ok.sum() < 6 or np.unique(y[ok]).size < 2:
        return np.nan, np.nan, int(ok.sum())
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    A = np.c_[np.ones(ok.sum()), np.column_stack([stats.rankdata(c) for c in Z[ok].T])]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, np.nan, int(ok.sum())
    r = float(np.corrcoef(ex, ey)[0, 1])
    null = np.array([np.corrcoef(ex, rng.permutation(ey))[0, 1] for _ in range(n_perm)])
    return r, float((1 + (np.abs(null) >= abs(r) - 1e-12).sum()) / (n_perm + 1)), int(ok.sum())


DESIGN_AUC = 0.9  # shared rule: a binary nuisance separated this well by deployment year or month is not testable


def design_auc(B: pd.DataFrame, nu: str) -> float:
    """For a binary nuisance: the largest separation (AUC folded to >= 0.5) of its two groups by deploy_year or
    deploy_month; NaN for non-binary nuisances or without deployment data."""
    if nu not in B or nu in ("deploy_year", "deploy_month"):
        return np.nan
    y = B[nu].to_numpy(float)
    if set(np.unique(y[np.isfinite(y)])) != {0.0, 1.0}:
        return np.nan
    best = np.nan
    for c in ("deploy_year", "deploy_month"):
        if c in B:
            x = B[c].to_numpy(float)
            a = auc(x[y == 1], x[y == 0])
            if np.isfinite(a):
                best = np.nanmax([best, max(a, 1 - a)])
    return float(best)


def nuisance_table(B: pd.DataFrame, scores: dict, n_perm: int, rng) -> pd.DataFrame:
    rows = []
    for run_, cols in scores.items():
        for c in cols:
            if c not in B:
                continue
            for nu in NUISANCE:
                if nu not in B:
                    continue
                rho, p, n = perm_spearman(B[c].to_numpy(float), B[nu].to_numpy(float), n_perm, rng)
                ctrl = [x for x in CONTROLS if x != nu and x in B and B[x].notna().any()]
                pr, pp, pn = partial_spearman(B[c].to_numpy(float), B[nu].to_numpy(float), B[ctrl].to_numpy(float),
                                              n_perm, rng)
                rows.append({"run": run_, "score": c, "nuisance": nu, "spearman_rho": rho, "p_perm_birds": p,
                             "n_birds": n, "controls": "+".join(ctrl), "partial_rho": pr, "p_partial": pp,
                             "n_birds_partial": pn, "design_auc": design_auc(B, nu)})
    T = pd.DataFrame(rows)
    if len(T):  # family = every testable test of the column in this table
        T["testable"] = ~(T["design_auc"] >= DESIGN_AUC)
        for c, h in (("p_perm_birds", "p_holm"), ("p_partial", "p_partial_holm")):
            T[h] = holm(np.where(T["testable"], T[c].to_numpy(float), np.nan))
        T["n_perm"] = n_perm
    return T


# ================================================================ driver


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
    off, n_nan = solar_offset(df["t_start"], df["local_solar_hour"].to_numpy())
    t = df["t_start"]
    utc_h = (t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600).to_numpy(float)
    df["lsh_filled"] = ~np.isfinite(df["local_solar_hour"].to_numpy(float))
    df["local_solar_hour"] = np.where(df["lsh_filled"], (utc_h + off) % 24, df["local_solar_hour"].to_numpy(float))
    df["day"], df["night"] = local_days(t, df["local_solar_hour"].to_numpy())
    df["month"] = df["t_start"].dt.month
    df["season"] = (df["month"] % 12) // 3
    df["ym"] = df["t_start"].dt.strftime("%Y-%m")
    return df


def run(windows: Path, out: Path, runs: list[Path] = (), states2: list[Path] = (), states6: list[Path] = (),
        reference_csv: Path | None = None, k: int = 20, n_inject: int = 2000, workers: int = 1, seed: int = 0,
        ref_cap: int = 2000, min_ref: int = 200, n_perm: int = 2000, top_bouts: int = 16, birds: list[str] | None = None,
        parts: tuple = ("qc", "behaviour")) -> dict:
    t0 = time.time()
    runs, states2, states6 = list(runs), list(states2), list(states6)
    if not (len(runs) == len(states2) == len(states6)):
        raise SystemExit("give one --states2 and one --states6 per --run, in the same order")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise SystemExit(f"run directory names must differ: {names}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dirs = [d for d in sorted(p for p in windows.iterdir() if p.is_dir())
            if (d / "meta.parquet").exists() and pq.ParquetFile(d / "meta.parquet").metadata.num_rows > 0
            and (birds is None or d.name in birds)]
    log(f"{len(dirs)} birds with windows")
    F = pd.concat(Parallel(n_jobs=workers)(delayed(load_bird_frame)(d) for d in dirs), ignore_index=True)
    meta = {"runs": names, "states2": [str(s) for s in states2], "states6": [str(s) for s in states6],
            "windows": str(windows), "k": k, "ref_cap": ref_cap, "min_ref": min_ref, "n_inject": n_inject,
            "seed": seed, "n_perm": n_perm, "parts": list(parts), "n_windows": len(F),
            "n_birds": int(F["bird"].nunique()), "subset_birds": birds is not None,
            "local_solar_hour_missing": {"n_windows": int(F["lsh_filled"].sum()),
                                         "share": float(F["lsh_filled"].mean()),
                                         "n_birds": int(F.loc[F["lsh_filled"], "bird"].nunique())},
            "qc_thresholds": {k_: v for k_, v in globals().items() if k_.isupper() and isinstance(v, (int, float))}}
    if "qc" in parts:
        log("QC")
        meta["qc"] = run_qc(F, {d.name: d for d in dirs}, out, n_inject, workers, seed, rng)
        log(f"QC done ({time.time() - t0:.0f} s)")
    birds_tab = bird_table(F, reference_csv)
    scores, meta["behaviour"] = {}, {}
    if "behaviour" in parts:
        for r, s2, s6 in zip(runs, states2, states6):
            log(f"behaviour {r.name}")
            res = run_behaviour(F, r, s2, s6, out, k, ref_cap, min_ref, n_inject, workers, seed, rng, top_bouts,
                                windows)
            meta["behaviour"][r.name] = res["summary"]
            l2s = res["l2_scores"].set_index("bird")
            birds_tab[f"l2_score_{r.name}"] = l2s["l2_score"].reindex(birds_tab.index) if len(l2s) else np.nan
            mc = res["l2_months"]
            birds_tab[f"l2_mean_change_{r.name}"] = (mc.groupby("bird")["z_change"].mean().reindex(birds_tab.index)
                                                    if len(mc) else np.nan)
            scores[r.name] = [f"l2_score_{r.name}", f"l2_mean_change_{r.name}"]
    birds_tab.reset_index().to_csv(out / "birds.csv", index=False)
    nu = nuisance_table(birds_tab, scores, n_perm, rng)
    nu.to_csv(out / "nuisance.csv", index=False)
    meta["runtime_s"] = round(time.time() - t0, 1)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    write_summary(out, meta, nu)
    log(f"done ({meta['runtime_s']} s)")
    return meta


def _neutral(factors: list[dict], min_base: float = 1e-3) -> bool:
    """Every format factor present in >= min_base of the scored units has a ratio within [1/NEUTRAL, NEUTRAL]."""
    r = [f["ratio"] for f in factors if f["format_factor"] and np.isfinite(f["ratio"])
         and f.get("share_scored", 1) >= min_base]
    return all(1 / NEUTRAL <= x <= NEUTRAL for x in r)


def _verdict(factors: list[dict]) -> str:
    """'met'/'NOT met' and the ratios (format factors marked *; under 0.1% of units marked 'rare')."""
    parts = [f"{f['factor']}{'*' if f['format_factor'] else ''} {_fmt(f['ratio'])}"
             f"{' (rare)' if f.get('share_scored', 1) < 1e-3 else ''}" for f in factors
             if not f["factor"].startswith("month")]
    return (f"format neutrality among scored units {'met' if _neutral(factors) else 'NOT met'} (units excluded "
            "before scoring are not tested by this); " + "; ".join(parts))


def _thr(d: dict) -> str:
    return ", ".join(f"{k} {_fmt(v)}" for k, v in d.items())


def write_summary(out: Path, meta: dict, nu: pd.DataFrame) -> None:
    L = ["# Sensor QC and behavioural anomalies (stage 7, version 2)", "",
         "Two separate products. Part 1 checks the sensors and the recording with fixed rules on the raw data (no "
         "model). Part 2 looks for unusual behaviour with h, using only full windows (>= 190 samples) and comparing "
         "like with like (same bird, recording mode, coarse state, season). Labels: [descriptive] = counted; "
         "[tested] = permutation test with the bird as the unit.", "",
         "Behavioural 'flags' are the top 1% of each score ranked for inspection (a fixed share by construction), "
         "not detections: there is no window-level injection, so window ranks have no recall. Bout and day ranks "
         "have recall only for the h-level injections listed below.", ""]
    ls = meta.get("local_solar_hour_missing")
    if ls:
        L += [f"- local_solar_hour missing on {ls['n_windows']:,} windows ({_fmt(100 * ls['share'])}%, {ls['n_birds']} "
              "birds); their solar offset is filled from the previous/next window in time (as budgets.py) [descriptive].",
              ""]
    if meta.get("subset_birds"):
        L += ["**Subset run (not all birds): numbers are a check, not results.**", ""]
    q = meta.get("qc")
    if q:
        L += ["## Part 1: sensor and recording QC", "",
              f"- {q['n_birds']} birds, {q['n_bird_days']} bird-days. Flagged windows per check (eligible windows; "
              "birds with any flag) [descriptive]:"]
        for c in QC_CHECKS:
            L.append(f"  - {c}: {q['flag_counts'][c]:,} of {q['eligible_counts'][c]:,} ({q['birds_with_flags'][c]} birds)")
        L.append("- Flag rate among eligible windows of the real data (an upper bound of the false-positive rate, "
                 "since some flags may be real faults): " + "; ".join(
                     f"{c} {_fmt(q['flag_counts'][c] / max(q['eligible_counts'][c], 1))}" for c in QC_CHECKS))
        L.append(f"- Roost-direction change events (> {ROOST_CP_DEG:g} deg between the {ROOST_AFTER} following and "
                 f"{ROOST_BEFORE} previous nights): {q['roost_events']} (median per bird-year "
                 f"{_fmt(q['roost_events_per_year_median'])}); flapping-axis change events (> {YAW_DAY_DEG:g} deg): "
                 f"{q['flap_axis_events']} [descriptive].")
        L.append(f"- Flagged bird-days: {q['days']} [descriptive].")
        L.append("- Injected faults on raw windows (recall among eligible windows of the check meant for the fault; "
                 "eligible share; 95% interval over birds) [descriptive]:")
        for r in q["recall"]:
            if r["target"] or r["fault"].startswith("clean") or r["check"] == "any_check":
                L.append(f"  - {r['fault']} / {r['check']} ({r['role']}): {_fmt(r['flag_rate_eligible'])} "
                         f"[{_fmt(r['ci95_lo'])}, {_fmt(r['ci95_hi'])}]; eligible share {_fmt(r['eligible_share'])} of "
                         f"{r['n_injected']} injected windows ({r['n_eligible']} eligible); {r['n_birds']} birds")
        L.append("- Caveat: for stuck_axis, amplitude_half, saturate, gps_speed_zero and gps_spike the injection sets "
                 "exactly the quantity the check thresholds, so those recalls show the code works, not that real "
                 "faults of that kind would look like that.")
        L.append(f"- Persistent slips (detected within +-{SLIP_WIN} nights/days of the onset; clean rate = same test, "
                 f"no slip). The roost threshold ROOST_CP_DEG = {ROOST_CP_DEG:g} deg was chosen after seeing the clean "
                 "rate on the real data (15 deg gave 9%) and the injected slips, so the recall below is not an "
                 "out-of-sample estimate. A yaw slip of 180 deg (rotation about gravity) is invisible to every check: "
                 "gravity is unchanged and the flapping axis is axial (defined modulo 180 deg).")
        rs = [r for r in q["slip"] if 20 <= r["deg"] <= 45]
        for kind in sorted({r["kind"] for r in rs}):
            k_ = [r for r in rs if r["kind"] == kind]
            L.append(f"  - {kind}, recall over 20-45 deg: " + ", ".join(f"{r['deg']:g} deg {_fmt(r['recall'])}"
                                                                         for r in k_))
        for r in q["slip"]:
            L.append(f"  - {r['kind']} {r['deg']:g} deg: recall {_fmt(r['recall'])} [{_fmt(r['ci95_lo'])}, "
                     f"{_fmt(r['ci95_hi'])}], clean rate {_fmt(r['clean_rate_same_positions'])}, {r['n_onsets']} onsets, "
                     f"{r['n_birds']} birds")
        L.append("")
    for name, s in meta.get("behaviour", {}).items():
        w = s["window"]
        L += [f"## Part 2: behaviour, run {name}", "",
              f"- Window level (full windows only): {w['n_scored']:,} of {w['n_full']:,} scored (reference levels "
              f"{w['levels']}; -1 = unscored). Top 1% ranked for inspection: 99th percentile per mode x K=2 state, pooled over birds "
              f"({w['n_flagged']:,} windows, {w['birds_flagged']} birds; thresholds {_thr(w['thresholds_stratum'])}); "
              "mode and state are neutral by construction, the other format factors are a real test [descriptive]:",
              f"  - primary: {_verdict(w['factors_stratum'])}",
              f"  - one pooled threshold (z > {_fmt(w['threshold_z'])}): {_verdict(w['factors_pooled'])}",
              f"  - top 1% per bird x mode x state: {_verdict(w['factors_per_group'])}"]
        if "bout" in s:
            b = s["bout"]
            L.append(f"- Bout level (continuous mode): {b['n_bouts']:,} bouts of {b['n_birds']} birds (by length bin "
                     f"{b['n_by_bin']}); content scored {b['n_scored_content']:,}, sequence scored "
                     f"{b['n_scored_seq']:,}. Top 1% ranked for inspection: 99th percentile per length bin, pooled over birds "
                     f"({b['n_flag_content']} content, {b['n_flag_seq']} sequence); bins neutral by construction "
                     "[descriptive]:")
            L.append(f"  - content, primary: {_verdict(b['factors_content'])}")
            L.append(f"  - content, one pooled threshold: {_verdict(b['factors_content_pooled'])}")
            L.append(f"  - sequence, primary: {_verdict(b['factors_seq'])}")
            L.append(f"  - sequence, one pooled threshold: {_verdict(b['factors_seq_pooled'])}")
        if "day" in s:
            d = s["day"]
            L.append(f"- Day level (scheduled mode, full windows): {d['n_scored']:,} of {d['n_days']:,} bird-days with "
                     f">= {DAY_M} full windows over >= {DAY_MIN_HOURS} local hours scored ({d['n_too_few_days']} with fewer than "
                     f"{DAY_MIN_PAIRS} comparable days; {d['n_partial_days_not_scored']:,} days with >= {DAY_M} windows "
                     f"but fewer hours not scored), "
                     f"{d['n_birds']} birds; top 1% ranked for inspection above one pooled threshold z > {_fmt(d['threshold_z'])}; "
                     "pairs of days are compared on equal-size samples drawn without replacement within common hours "
                     "[descriptive]: "
                     f"{_verdict(d['factors'])}")
        cv = s.get("coverage")
        if cv:
            L.append(f"- Not scored [descriptive]: {_fmt(100 * cv['share_windows_no_window_score'])}% of all "
                     f"{cv['n_windows']:,} windows have no window-level score ({cv['n_short']:,} short windows, "
                     f"{_fmt(100 * cv['share_short'])}%, excluded by design; {cv['n_full_unscored']:,} full windows "
                     f"without a reference); {cv['n_bird_days_no_score']:,} of {cv['n_bird_days']:,} bird-days "
                     f"({_fmt(100 * cv['share_bird_days_no_score'])}%) have no score at any level and "
                     f"{_fmt(100 * cv['share_bird_days_no_day_score'])}% no day-level score; birds without a score: "
                     f"any level {cv['n_birds_no_score']}, window {cv['n_birds_no_window_score']}, bout "
                     f"{cv['n_birds_no_bout_score']}, day {cv['n_birds_no_day_score']} of {cv['n_birds']}. Format "
                     "neutrality below concerns scored units only.")
        L.append("- Injections (share above the primary threshold of the host's stratum; AUC against the clean "
                 "scores of the same hosts; 95% interval over birds) [descriptive]:")
        for r in s.get("inject", []):
            L.append(f"  - {r['level']} / {r['injection']} / {r['score']} ({r['role']}): {_fmt(r['flag_rate'])} "
                     f"[{_fmt(r['ci95_lo'])}, {_fmt(r['ci95_hi'])}], AUC {_fmt(r['auc_vs_clean_hosts'])}, "
                     f"{r['n_scored']} scored, {r['n_birds']} birds")
        l2 = s["l2"]
        L.append(f"- Level 2: {l2['n_cells_scored']} bird x cell comparisons in {l2['n_cells']} cells with >= "
                 f"{L2_MIN_BIRDS} birds (window format x mode x state x season, {L2_N} windows each), {l2['n_birds']} "
                 f"birds; bird score = median within-cell rank (0-1) over its cells, {l2['n_birds_scored']} birds "
                 f"with >= {L2_MIN_CELLS} cells; {l2['n_month_pairs']} month pairs, of which "
                 f"{l2['n_month_pairs_consecutive']} consecutive months enter the change score [descriptive].")
        L.append("")
    if len(nu):
        n_perm = int(meta.get("n_perm", 2000))
        L += ["## Bird-level scores against nuisance [descriptive screen; permutation p between birds, Holm-adjusted]",
              "", f"{int(nu['p_holm'].notna().sum())} Spearman and {int(nu['p_partial_holm'].notna().sum())} partial "
              f"testable tests; Holm is applied over all testable tests of each p column in this table. Partial: ranks residualised on "
              f"the controls ({', '.join(c for c in CONTROLS if nu['controls'].str.contains(c).any())}; the nuisance "
              "itself is left out of its own controls; deploy_month enters as a linear rank) by least squares; Pearson r of residuals; p "
              f"from permuting residuals between birds. p at the permutation floor prints as <= 1/({n_perm}+1); a Holm "
              "value computed from a floor p is an upper bound. "
              "deploy_year is the real deployment year (earliest deploy-on), used as the cohort. A binary nuisance "
              f"whose groups deployment year or month separates with AUC >= {DESIGN_AUC:g} is labelled not testable "
              "in this design.", "",
              "| run | score | nuisance | rho | p | p Holm | partial rho | p | p Holm | birds |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in nu.iterrows():
            if not r["testable"]:
                L.append(f"| {r['run']} | {r['score']} | {r['nuisance']} | {_fmt(r['spearman_rho'])} | not testable in "
                         f"this design (deployment year/month separate the groups, AUC {_fmt(r['design_auc'])}) | | "
                         f"{_fmt(r['partial_rho'])} | | | {r['n_birds']} |")
                continue
            L.append(f"| {r['run']} | {r['score']} | {r['nuisance']} | {_fmt(r['spearman_rho'])} | "
                     f"{fmt_p(r['p_perm_birds'], n_perm)} | {fmt_p(r['p_holm'], n_perm)} | {_fmt(r['partial_rho'])} | "
                     f"{fmt_p(r['p_partial'], n_perm)} | {fmt_p(r['p_partial_holm'], n_perm)} | {r['n_birds']} |")
    L += ["", "## Files", "",
          "- qc_inject.csv, qc_slip_inject.csv: QC validation; qc_windows_flagged.parquet, qc_bird_days.csv, "
          "qc_nights.csv, qc_events.csv, qc_birds.csv, qc_timelines/<bird>.png: QC results (relative days).",
          "- beh_window_factors_*, beh_top_windows_*: window level; beh_bouts_*, beh_bout_factors_*: bouts; "
          "beh_days_*, beh_day_factors_*: bird-days; beh_inject_*: validation; blind_bouts_*_p*.png with "
          "blind_key_*.csv: top bouts without bird id or date.",
          "- l2_cells_*, l2_birds_*, l2_months_*, birds.csv, nuisance.csv: level 2.", "- metadata.json: settings and counts.",
          "", "Not verified here: whether any flag is a real fault or behaviour; that needs the timelines and blind "
          "pages to be read."]
    (out / "summary.md").write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", default=[])
    ap.add_argument("--states2", type=Path, action="append", default=[], help="K=2 states dir, one per --run")
    ap.add_argument("--states6", type=Path, action="append", default=[], help="K=6 states dir, one per --run")
    ap.add_argument("--reference", type=Path, default=None, help="Movebank reference CSV (sex, migrant, young)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-inject", type=int, default=2000)
    ap.add_argument("--ref-cap", type=int, default=2000)
    ap.add_argument("--min-ref", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", nargs="+", default=["qc", "behaviour"], choices=["qc", "behaviour"])
    ap.add_argument("--birds", nargs="*", default=None, help="subset of birds (checks only)")
    a = ap.parse_args(argv)
    run(a.windows, a.out, a.run, a.states2, a.states6, a.reference, a.k, a.n_inject, a.workers, a.seed, a.ref_cap,
        a.min_ref, a.n_perm, birds=a.birds, parts=tuple(a.parts))


if __name__ == "__main__":
    main()
