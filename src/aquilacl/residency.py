"""Movement strategy of each bird from its GPS track, and behaviour of migrants vs residents in the basin.

Privacy: positions exist only in memory. Every output holds distances (km, rounded to 1 km), day-of-year or days
since deployment, and classes. No latitude, longitude, centroid or study-area centre is written or logged.

Part 1. Tracks (one worker per bird, raw GPS parquet)
- Fixes: finite, not (0, 0), latitude/longitude in range, at or after the reference deploy-on date (if given).
- Daily position: median latitude and longitude of the fixes of each UTC day. Bird median: median of all its fixes.
  Study-area centre: median of the per-bird medians. Deployment centroid: median of the daily positions of the first
  FIRST_DAYS days. Day 0 = first day with a daily position ("days since deployment").
- Per day: d0 = great-circle distance (km) to the deployment centroid, NSD = d0^2 (km^2), north0 = northward
  displacement from the centroid (km), dc and northc the same from the study-area centre.

Part 1a. NSD models (Bunnefeld et al. 2011), fitted by least squares on NSD (scaled by its maximum) over days
0..FIT_DAYS (first annual cycle plus 2 months; one double sigmoid cannot fit two years of migration):
  resident  NSD = c                                         (1 parameter)
  nomad     NSD = b t                                       (1)
  disperser NSD = D / (1 + exp((th - t) / ph))              (3)
  migrant   NSD = D / (1 + exp((th1 - t) / ph1)) - D / (1 + exp((th1 + stay - t) / ph2))   (5)
  AIC = n log(RSS / n) + 2 (k + 1); sigmoid models are fitted from a grid of starts with bounds
  (ph in [PHI_MIN, PHI_MAX] days, midpoints inside the fitted days). Daily NSD is autocorrelated, so AIC alone
  prefers the sigmoid models for any foray. Acceptance rules (a rejected model gives way to the next by AIC):
  disperser and migrant need sqrt(D) >= MIN_DISP_KM; migrant needs stay >= MIN_STAY_DAYS and the return midpoint
  inside the fitted days; nomad needs sqrt(b T) >= MIN_DISP_KM. Censoring: 'resident' needs a track that covers
  spring (daily positions after day FIRST_DAYS in >= 2 of Mar, Apr, May), and 'disperser' needs >= SETTLE_DAYS
  of track after the departure midpoint (otherwise a migrant that has not yet returned); else 'undetermined'.
  Departure / return day-of-year = the dates of the two sigmoid midpoints (half of the NSD change, not onsets).
Part 1b. Rule classes (whole track): a migration is an excursion that leaves MIN_DISP_KM of the deployment centroid,
  reaches >= MIGRANT_NORTH_KM north of it, and is back within MIN_DISP_KM within RETURN_WITHIN_DAYS of leaving
  (departure / return = first day beyond / back within MIN_DISP_KM). Classes in order:
  insufficient (span < min_days or fewer than min_days/2 daily positions); migrant (>= 1 migration);
  undetermined (far north without return and the track ends < RETURN_WITHIN_DAYS after leaving);
  resident (30-day running median of d0 never > MIN_DISP_KM; needs spring coverage, else undetermined);
  disperser (away > MIN_DISP_KM for the last FINAL_DAYS and continuously for >= SETTLE_DAYS; away for less ->
  undetermined); nomad (everything else: sustained excursions that end back near the deployment area).
  The rule class is the primary 'strategy'; 'consensus' is the class where both methods agree.

Part 2. Outputs in --out: residency.csv (per bird), agreement.csv, residency_summary.md, nsd_birds.png,
nsd_by_strategy.png, behaviour_tests.csv, behaviour_birds.csv, classifiers.csv, nested.csv, metadata.json.

Part 3. Behaviour comparison, migrant (1) vs resident (0), bird level. Birds: cohort 2024 (year of the first
window <= 2024), windows in COMPARE_MONTHS (Nov-Mar) on UTC days on which the bird's daily position is within
BASIN_KM of the study-area centre (nearest daily position within 2 days). Groupings: rule classes (primary) and
consensus classes (sensitivity), each for non-young birds ('adult': tagged_young not stated; primary) and all ages.
(a) TESTED: share of K=2 flight and of each K=6 state among scheduled (sensitivity_k == 0) windows with a state
    (continuous mode is triggered by fast movement and would inflate flight); birds need >= MIN_SHARE_WINDOWS.
    DESCRIPTIVE: the same shares over all windows.
(b) TESTED: per-bird medians of speed (m/s), turn radius (m; speed / radians(turn_abs_mean), turn > 0) and climb
    (m/s) in K=2 flight and in each K=6 flight state; GPS is almost only in continuous windows, so these come
    from all windows with GPS (mostly sensitivity_k == 1); birds need >= MIN_STATE_WINDOWS.
    Tests: difference in means with labels permuted between birds (aquilacl.demography.perm_mean_diff), also
    within sex strata (missing sex = own stratum); Hedges g; Holm adjustment within each family and grouping.
    A side needs >= MIN_GROUP_BIRDS birds, else 'not tested'.
(c) TESTED: grouped-CV logistic classifier (aquilacl.demography.run_classifiers: StratifiedGroupKFold by bird,
    labels permuted between birds, Freedman-Lane nested null for h beyond nuisance) on the mean L2-normalised h of
    the run in K=2 flight, in K=2 ground, and both; all windows (recording-mode mix enters through cont_share).
    Nuisance floor: track days, deploy month (sin, cos), continuous-mode share (all windows of the bird); cohort
    is constant within cohort 2024 and is added only if it varies. Each h set has its own nuisance spec on the
    same birds (run tag '<run>:flight' etc.), so the nested comparison uses identical folds.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from sklearn.metrics import cohen_kappa_score

from . import demography as dm
from . import reference

R_EARTH = 6371.0
KM_PER_DEG = np.pi / 180 * R_EARTH
FIRST_DAYS = 30
FIT_DAYS = 425
MIN_DISP_KM = 200.0
MIGRANT_NORTH_KM = 1000.0
RETURN_WITHIN_DAYS = 365
MIN_STAY_DAYS = 30
SETTLE_DAYS = 240
FINAL_DAYS = 60
ROLL_DAYS, ROLL_MIN = 30, 10
PHI_MIN, PHI_MAX = 1.0, 60.0
SPRING_MONTHS = (3, 4, 5)
SUMMER_MONTHS, WINTER_MONTHS = (6, 7, 8), (12, 1, 2)
COMPARE_MONTHS = (11, 12, 1, 2, 3)
BASIN_KM = 200.0
BASIN_FILL_DAYS = 2
MIN_SHARE_WINDOWS = 100
MIN_STATE_WINDOWS = 20
MIN_GROUP_BIRDS = 4
MODELS = ("resident", "nomad", "disperser", "migrant")
N_PAR = {"resident": 1, "nomad": 1, "disperser": 3, "migrant": 5}
STATES2 = {0: "ground", 1: "flight"}
STATES6 = {0: "rest", 1: "perched_active", 2: "thermal_circling", 3: "fast_glide", 4: "cruising", 5: "flapping"}
FLIGHT6 = (2, 3, 4, 5)  # flight states of states_r6_yaw60_bpb4 (brief numbering; check against state_profiles.csv)
GPS_COLS = ["timestamp", "location-long", "location-lat"]
log = dm.log


# ---------------------------------------------------------------- geometry and daily positions

def hav_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def clean_fixes(g: pd.DataFrame, deploy_on=None) -> pd.DataFrame:
    """Valid fixes (finite, in range, not (0, 0), at or after deploy_on) with a UTC 'date' column."""
    g = g.rename(columns={"location-lat": "lat", "location-long": "lon"})
    g = g.assign(timestamp=pd.to_datetime(g["timestamp"], utc=True)).dropna(subset=["timestamp", "lat", "lon"])
    ok = (g.lat.abs() <= 90) & (g.lon.abs() <= 180) & ((g.lat.abs() > 1e-6) | (g.lon.abs() > 1e-6))
    if deploy_on is not None and pd.notna(deploy_on):
        ok &= g.timestamp >= pd.Timestamp(deploy_on)
    g = g[ok]
    return g.assign(date=g.timestamp.dt.floor("D"))


def daily_positions(g: pd.DataFrame) -> pd.DataFrame:
    """Median latitude/longitude per UTC day (input from clean_fixes); in memory only."""
    d = g.groupby("date")[["lat", "lon"]].median()
    d["n"] = g.groupby("date").size()
    return d.reset_index()


def read_bird(path: Path, deploy_on) -> dict:
    g = clean_fixes(pd.read_parquet(path, columns=GPS_COLS), deploy_on)
    med = (float(g.lat.median()), float(g.lon.median())) if len(g) else None
    return {"bird": path.stem, "daily": daily_positions(g) if len(g) else None, "median": med}


def track(daily: pd.DataFrame, centre: tuple[float, float]) -> pd.DataFrame:
    """Per day: days since deployment, day of year, month, distances (km) and NSD (km^2). No positions returned."""
    day = ((daily["date"] - daily["date"].iloc[0]) / pd.Timedelta(days=1)).round().astype(int).to_numpy()
    first = daily[day < FIRST_DAYS]
    c0 = (first.lat.median(), first.lon.median())
    d0 = hav_km(c0[0], c0[1], daily.lat.to_numpy(), daily.lon.to_numpy())
    dc = hav_km(centre[0], centre[1], daily.lat.to_numpy(), daily.lon.to_numpy())
    return pd.DataFrame({"day": day, "epoch_day": ((daily["date"] - pd.Timestamp("1970-01-01", tz="UTC")) //
                                                   pd.Timedelta(days=1)).to_numpy(),
                         "doy": daily["date"].dt.dayofyear.to_numpy(), "month": daily["date"].dt.month.to_numpy(),
                         "year": daily["date"].dt.year.to_numpy(), "d0_km": d0, "nsd_km2": d0 ** 2,
                         "north0_km": (daily.lat.to_numpy() - c0[0]) * KM_PER_DEG, "dc_km": dc,
                         "northc_km": (daily.lat.to_numpy() - centre[0]) * KM_PER_DEG})


def covers_spring(tr: pd.DataFrame) -> bool:
    return len(set(tr.loc[tr.day >= FIRST_DAYS, "month"]) & set(SPRING_MONTHS)) >= 2


def _doy(tr: pd.DataFrame, day: float) -> float:
    """Day of year of (first date + day), via the first row's epoch day."""
    if not np.isfinite(day):
        return np.nan
    e = int(tr.epoch_day.iat[0] - tr.day.iat[0] + round(day))
    return float((pd.Timestamp("1970-01-01", tz="UTC") + pd.Timedelta(days=e)).dayofyear)


# ---------------------------------------------------------------- NSD models

def _sig(t, th, ph):
    return 1.0 / (1.0 + np.exp(np.clip((th - t) / ph, -60, 60)))


def curve(model: str, p, t):
    t = np.asarray(t, float)
    if model == "resident":
        return np.full(len(t), p[0])
    if model == "nomad":
        return p[0] * t
    if model == "disperser":
        return p[0] * _sig(t, p[1], p[2])
    return p[0] * (_sig(t, p[1], p[2]) - _sig(t, p[1] + p[3], p[4]))


def _aic(rss: float, n: int, k: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + 2 * (k + 1)


def fit_models(t: np.ndarray, nsd: np.ndarray) -> dict:
    """Least-squares fit of the four NSD models; params in km^2 / days. Returns model -> {p, rss, aic}."""
    t, y0 = np.asarray(t, float), np.asarray(nsd, float)
    s = max(float(np.max(y0)), 1e-9)
    y, n, T = y0 / s, len(t), float(t.max())
    out = {}
    c = y.mean()
    out["resident"] = {"p": np.array([c * s]), "rss": float(((y - c) ** 2).sum())}
    b = float((t * y).sum() / max((t * t).sum(), 1e-12))
    out["nomad"] = {"p": np.array([b * s]), "rss": float(((y - b * t) ** 2).sum())}
    qs = np.quantile(t, np.linspace(0.05, 0.85, 9))
    best = (np.inf, None)
    for th in qs:
        for ph in (3.0, 10.0):
            r = least_squares(lambda p: curve("disperser", p, t) - y, [1.0, th, ph],
                              bounds=([0, 0, PHI_MIN], [4, T, PHI_MAX]))
            if 2 * r.cost < best[0]:
                best = (2 * r.cost, r.x)
    out["disperser"] = {"p": best[1] * [s, 1, 1], "rss": best[0]}
    best = (np.inf, None)
    for th in qs:
        for stay in (45.0, 90.0, 150.0, 210.0):
            if th + stay > T:
                continue
            for ph in (3.0, 10.0):
                r = least_squares(lambda p: curve("migrant", p, t) - y, [1.0, th, ph, stay, ph],
                                  bounds=([0, 0, PHI_MIN, 1, PHI_MIN], [4, T, PHI_MAX, T, PHI_MAX]))
                if 2 * r.cost < best[0]:
                    best = (2 * r.cost, r.x)
    if best[1] is None:
        out["migrant"] = {"p": np.full(5, np.nan), "rss": np.inf}
    else:
        out["migrant"] = {"p": best[1] * [s, 1, 1, 1, 1], "rss": best[0]}
    for m in MODELS:
        out[m]["aic"] = _aic(out[m]["rss"], n, N_PAR[m]) if np.isfinite(out[m]["rss"]) else np.inf
    return out


def accept(model: str, p, T: float) -> bool:
    if model == "resident":
        return True
    if model == "nomad":
        return p[0] > 0 and np.sqrt(p[0] * T) >= MIN_DISP_KM
    if model == "disperser":
        return np.sqrt(p[0]) >= MIN_DISP_KM and p[1] < T
    return (np.isfinite(p).all() and np.sqrt(p[0]) >= MIN_DISP_KM and p[3] >= MIN_STAY_DAYS
            and p[1] + p[3] <= T)


def nsd_classify(tr: pd.DataFrame, min_days: int) -> dict:
    span = int(tr.day.max()) if len(tr) else 0
    res = {"nsd_class": "insufficient", "nsd_note": "", "nsd_best_aic": None, "nsd_delta_km": np.nan,
           "dep_doy_nsd": np.nan, "ret_doy_nsd": np.nan, "fit": None,
           **{f"daic_{m}": np.nan for m in MODELS}}
    if span < min_days or len(tr) < min_days / 2:
        res["nsd_note"] = "short track"
        return res
    w = tr[tr.day <= FIT_DAYS]
    t, y = w.day.to_numpy(float), w.nsd_km2.to_numpy(float)
    T = float(t.max())
    fits = fit_models(t, y)
    aics = {m: fits[m]["aic"] for m in MODELS}
    amin = min(aics.values())
    res.update({f"daic_{m}": round(aics[m] - amin, 1) for m in MODELS})
    order = sorted(MODELS, key=lambda m: aics[m])
    res["nsd_best_aic"] = order[0]
    chosen = next(m for m in order if accept(m, fits[m]["p"], T))
    p = fits[chosen]["p"]
    res["fit"] = {"model": chosen, "p": p.tolist(), "t_max": T}
    cls, note = chosen, "" if chosen == order[0] else f"best AIC {order[0]} rejected"
    if chosen in ("disperser", "migrant"):
        res["nsd_delta_km"] = float(np.sqrt(p[0]))
    if chosen == "migrant":
        res["dep_doy_nsd"], res["ret_doy_nsd"] = _doy(tr, p[1]), _doy(tr, p[1] + p[3])
    if chosen == "resident" and not covers_spring(w):
        cls, note = "undetermined", "no spring coverage"
    if chosen == "disperser" and span - p[1] < SETTLE_DAYS:
        cls, note = "undetermined", "away at track end; too short to tell settling from a seasonal stay"
    res["nsd_class"], res["nsd_note"] = cls, note.strip("; ")
    return res


# ---------------------------------------------------------------- rule classes

def rule_classify(tr: pd.DataFrame, min_days: int) -> dict:
    span = int(tr.day.max()) if len(tr) else 0
    res = {"rule_class": "insufficient", "rule_note": "", "n_migrations": 0,
           **{f"{k}_doy_rule_y{i}": np.nan for i in (1, 2) for k in ("dep", "ret")}}
    if span < min_days or len(tr) < min_days / 2:
        res["rule_note"] = "short track"
        return res
    day, d0, n0 = tr.day.to_numpy(), tr.d0_km.to_numpy(), tr.north0_km.to_numpy()
    near, far = d0 < MIN_DISP_KM, n0 >= MIGRANT_NORTH_KM
    events, far_open, cur = [], None, -1
    while True:
        fi = np.flatnonzero(far & (day > cur))
        if not len(fi):
            break
        f = fi[0]
        before = np.flatnonzero(near[:f])
        dep = before[-1] + 1 if len(before) else 0
        after = np.flatnonzero(near[f:])
        if not len(after):
            far_open = dep
            break
        ret = f + after[0]
        if day[ret] - day[dep] <= RETURN_WITHIN_DAYS:
            events.append((dep, ret))
        cur = day[ret]
    res["n_migrations"] = len(events)
    for i, (a, b) in enumerate(events[:2], 1):
        res[f"dep_doy_rule_y{i}"], res[f"ret_doy_rule_y{i}"] = float(tr.doy.iat[a]), float(tr.doy.iat[b])
    full = pd.Series(d0, index=day).reindex(np.arange(span + 1))
    roll = full.rolling(ROLL_DAYS, min_periods=ROLL_MIN, center=True).median()
    last = d0[day > span - FINAL_DAYS]
    back = np.flatnonzero(near)
    away_since = day[back[-1]] + 1 if len(back) else 0
    if events:
        cls, note = "migrant", ""
    elif far_open is not None and span - day[far_open] < RETURN_WITHIN_DAYS:
        cls, note = "undetermined", "far north without return; track ends before a return is possible"
    elif not (roll > MIN_DISP_KM).any():
        cls, note = ("resident", "") if covers_spring(tr) else ("undetermined", "no spring coverage")
    elif len(last) and np.median(last) > MIN_DISP_KM:
        cls, note = ("disperser", "") if span - away_since >= SETTLE_DAYS else \
            ("undetermined", "away at track end; too short to tell settling from a seasonal stay")
    else:
        cls, note = "nomad", ""
    res["rule_class"], res["rule_note"] = cls, note
    return res


def distances(tr: pd.DataFrame) -> dict:
    def med(months):
        v = tr.loc[tr.month.isin(months), "dc_km"]
        return float(v.median()) if len(v) else np.nan
    return {"track_days": int(tr.day.max()), "n_days": len(tr), "covers_spring": bool(covers_spring(tr)),
            "max_dist_km": float(tr.dc_km.max()), "max_north_km": float(tr.northc_km.max()),
            "max_dist_deploy_km": float(tr.d0_km.max()), "max_north_deploy_km": float(tr.north0_km.max()),
            "summer_dist_km": med(SUMMER_MONTHS), "winter_dist_km": med(WINTER_MONTHS),
            "basin_days_share": float((tr.dc_km <= BASIN_KM).mean())}


def classify_bird(tr: pd.DataFrame, min_days: int) -> dict:
    n = nsd_classify(tr, min_days)
    r = rule_classify(tr, min_days)
    return {**distances(tr), **{k: v for k, v in n.items() if k != "fit"}, **r, "_fit": n["fit"]}


# ---------------------------------------------------------------- behaviour pass

def _read(path: Path, want: list[str]) -> pd.DataFrame:
    have = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in want if c in have])


def _states(rows, n: int, col: str) -> np.ndarray:
    s = np.full(n, -1)
    if rows is not None and len(rows):
        r = rows["row"].to_numpy()
        if r.min() < 0 or r.max() >= n:
            raise ValueError("states row index outside the bird's windows")
        s[r] = rows[col].to_numpy()
    return s


def behaviour_bird(d: Path, run: Path, st2, st6, basin_days: np.ndarray, state_col: str, months=COMPARE_MONTHS) -> dict:
    """Bird aggregates for part 3 (see module docstring)."""
    m = _read(d / "meta.parquet", ["t_start", "sensitivity_k"])
    n = len(m)
    t = pd.to_datetime(m["t_start"], utc=True).reset_index(drop=True)
    k = m["sensitivity_k"].to_numpy()
    res = {"bird": d.name, "first": t.min(), "last": t.max(), "n_all": n,
           "cont_share": float((k == 1).mean()) if n else np.nan}
    eday = ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(days=1)).to_numpy()
    sel = t.dt.month.isin(months).to_numpy() & np.isin(eday, basin_days)
    res["n_sel"] = int(sel.sum())
    s2, s6 = _states(st2, n, state_col), _states(st6, n, state_col)
    k0 = sel & (k == 0)
    for tag, s, names in (("k2", s2, STATES2), ("k6", s6, STATES6)):
        for mode, msk in (("sched", k0), ("all", sel)):
            ok = msk & (s >= 0)
            res[f"n_{tag}_{mode}"] = int(ok.sum())
            for c, name in names.items():
                if tag == "k2" and c == 0:
                    continue
                res[f"share_{tag}_{name}_{mode}"] = float((s[ok] == c).mean()) if ok.any() else np.nan
    f = _read(d / "features.parquet", ["speed_mean", "turn_abs_mean", "climb_mean"]).reindex(
        columns=["speed_mean", "turn_abs_mean", "climb_mean"])
    sp, tu, cl = (f[c].to_numpy(float) for c in f.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        rad = np.where(tu > 0, sp / np.radians(tu), np.nan)
    groups = {"k2_flight": s2 == 1, **{f"k6_{STATES6[c]}": s6 == c for c in FLIGHT6}}
    for g, msk in groups.items():
        w = sel & msk & np.isfinite(sp)
        res[f"kin_n_{g}"] = int(w.sum())
        res[f"kin_k1share_{g}"] = float((k[w] == 1).mean()) if w.any() else np.nan
        for v, a in (("speed_ms", sp), ("radius_m", rad), ("climb_ms", cl)):
            x = a[w]
            x = x[np.isfinite(x)]
            res[f"{v}_{g}"] = float(np.median(x)) if len(x) >= MIN_STATE_WINDOWS else np.nan
    h = np.load(run / "embeddings" / f"{d.name}.npz")["h"]
    if len(h) != n:
        raise ValueError("embedding rows differ from the windows of one bird")
    h = h.astype(np.float32)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    for c, name in STATES2.items():
        w = sel & (s2 == c)
        res[f"n_h_{name}"] = int(w.sum())
        res[f"h_{name}"] = h[w].mean(0) if w.sum() >= MIN_STATE_WINDOWS else np.full(h.shape[1], np.nan)
    return res


# ---------------------------------------------------------------- comparison

def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok])]
    adj = np.minimum(1, np.maximum.accumulate((len(o) - np.arange(len(o))) * p[o]))
    out[o] = adj
    return out


def compare_table(V: pd.DataFrame, lab: pd.Series, sex: pd.Series, family: str, grouping: str, n_perm: int, rng,
               tested: bool = True) -> list[dict]:
    rows = []
    for c in V.columns:
        v = V[c].astype(float)
        ok = v.notna() & lab.notna()
        x, y = v[ok].to_numpy(), lab[ok].astype(int).to_numpy()
        a, b = x[y == 1], x[y == 0]
        row = {"grouping": grouping, "family": family, "variable": c, "n_migrant": len(a), "n_resident": len(b),
               "mean_migrant": a.mean() if len(a) else np.nan, "mean_resident": b.mean() if len(b) else np.nan,
               "median_migrant": np.median(a) if len(a) else np.nan,
               "median_resident": np.median(b) if len(b) else np.nan,
               "diff_mean": a.mean() - b.mean() if len(a) and len(b) else np.nan,
               "hedges_g": dm.hedges_g(a, b) if len(a) and len(b) else np.nan}
        if not tested:
            row["test"] = "descriptive (not tested)"
        elif min(len(a), len(b)) < MIN_GROUP_BIRDS:
            row["test"] = f"not tested: a group has < {MIN_GROUP_BIRDS} birds"
        else:
            d, p = dm.perm_mean_diff(x, y, n_perm, rng)
            strata = pd.Categorical(sex[ok].fillna("missing")).codes
            _, ps = dm.perm_mean_diff(x, y, n_perm, rng, strata)
            row.update({"test": f"bird-label permutation ({n_perm})", "diff_mean": d, "p_perm": p, "p_perm_sex_strata": ps})
        rows.append(row)
    return rows


def nuisance(B: pd.DataFrame) -> pd.DataFrame:
    dmn = B["deploy_month"].astype(float)
    X = pd.DataFrame({"track_days": B["track_days_windows"].astype(float),
                      "deploy_month_sin": np.sin(2 * np.pi * dmn / 12), "deploy_month_cos": np.cos(2 * np.pi * dmn / 12),
                      "cont_share": B["cont_share"].astype(float)}, index=B.index)
    if B["cohort"].nunique() > 1:
        X["cohort_2025"] = (B["cohort"] == "2025").astype(float)
    return X


def h_frame(P: dict, birds, key: str) -> pd.DataFrame:
    H = np.stack([P[b][f"h_{key}"] for b in birds])
    return pd.DataFrame(H, index=pd.Index(birds, name="bird"), columns=[f"{key}_h{i}" for i in range(H.shape[1])])


# ---------------------------------------------------------------- figures

def _plt():
    return dm._plt()


def fig_nsd_birds(R: pd.DataFrame, tracks: dict, path: Path) -> None:
    plt = _plt()
    birds = [b for b in R.index if b in tracks]
    if not birds:
        return
    nc = 8
    nr = int(np.ceil(len(birds) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc, 1.9 * nr), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, b in zip(axes.ravel(), birds):
        tr, fit = tracks[b]
        ax.set_visible(True)
        ax.plot(tr.day, tr.nsd_km2, ".", ms=1.2, color="#555555")
        if fit is not None:
            tt = np.linspace(0, fit["t_max"], 300)
            ax.plot(tt, curve(fit["model"], np.array(fit["p"]), tt), color="#c0392b", lw=1)
        ax.set_title(f"{b}: {R.at[b, 'strategy']} / {R.at[b, 'strategy_nsd']}", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.yaxis.get_offset_text().set_fontsize(5)
    fig.supxlabel("days since deployment")
    fig.supylabel("NSD (km$^2$)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_nsd_by_strategy(R: pd.DataFrame, tracks: dict, path: Path) -> None:
    plt = _plt()
    classes = [c for c in ("resident", "migrant", "disperser", "nomad", "undetermined", "insufficient")
               if (R.strategy == c).any()]
    if not classes:
        return
    fig, axes = plt.subplots(1, len(classes), figsize=(3.4 * len(classes), 3.2), squeeze=False, sharey=True)
    for ax, c in zip(axes[0], classes):
        bs = [b for b in R.index[R.strategy == c] if b in tracks]
        for b in bs:
            tr = tracks[b][0]
            ax.plot(tr.day, tr.nsd_km2, lw=0.6, alpha=0.6, color="#c0392b" if R.at[b, "young"] else "#2a6f97")
        ax.set_title(f"{c} (rule), n={len(bs)}\nblue: age not stated, red: tagged young", fontsize=7)
        ax.set_xlabel("days since deployment")
    axes[0][0].set_ylabel("NSD (km$^2$)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- summary

def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def _md(df: pd.DataFrame) -> list[str]:
    if not len(df):
        return ["(none)"]
    cols = [str(c) for c in df.columns]
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(_f(v) if isinstance(v, float) else str(v) for v in r.to_numpy()) + " |")
    return L


def write_summary(out: Path, R: pd.DataFrame, agree: pd.DataFrame, kappa: dict, groups: pd.DataFrame,
                  T: pd.DataFrame, clf: pd.DataFrame, nest: pd.DataFrame, meta: dict) -> None:
    L = ["# Residency from GPS tracks, and migrants vs residents", "",
         "No coordinates are stored: distances in km (rounded), day-of-year, classes. Thresholds and model "
         "definitions: module docstring of aquilacl.residency and metadata.json.", "",
         "## 1. Strategy per bird (descriptive)", "",
         "Rule classes (primary) by age group (young = tagged as young; other = age not stated):", ""]
    R2 = R.assign(age=np.where(R.young, "young", "other"))
    L += _md(pd.crosstab(R2.strategy, R2.age).reset_index())
    L += ["", "NSD-model classes (Bunnefeld et al. 2011, AIC with acceptance rules) by age group:", ""]
    L += _md(pd.crosstab(R2.strategy_nsd, R2.age).reset_index())
    L += ["", "### Agreement between the two methods (rows: rule, columns: NSD; descriptive)", ""]
    L += _md(pd.crosstab(R.strategy, R.strategy_nsd).reset_index())
    L += ["", f"Birds classified by both (no insufficient/undetermined): {kappa['n']}; agreement "
          f"{_f(kappa['agree'])}; Cohen's kappa {_f(kappa['kappa'])}.", "",
          "### Agreement with the reference 'migrant' flag (True or not stated; descriptive)", ""]
    L += _md(agree)
    mig = R[R.strategy == "migrant"]
    L += ["", "### Migration timing and ranges of rule migrants (descriptive)", "",
          f"- Rule migrants: {len(mig)} ({int(mig.young.sum())} tagged young).",
          f"- Departure day-of-year (rule, first migration): median {_f(mig.dep_doy_rule_y1.median())}, "
          f"range {_f(mig.dep_doy_rule_y1.min())}-{_f(mig.dep_doy_rule_y1.max())}; return: median "
          f"{_f(mig.ret_doy_rule_y1.median())}, range {_f(mig.ret_doy_rule_y1.min())}-{_f(mig.ret_doy_rule_y1.max())}.",
          f"- Sigmoid midpoints (NSD migrant fits, {int(R.dep_doy_nsd.notna().sum())} birds): departure median "
          f"{_f(R.dep_doy_nsd.median())}, return median {_f(R.ret_doy_nsd.median())} (midpoints, not onsets).",
          "", "Median distances (km) from the study-area centre per rule class:", ""]
    L += _md(R.groupby("strategy")[["max_dist_km", "max_north_km", "summer_dist_km", "winter_dist_km"]].median()
             .round(0).reset_index())
    L += ["", "## 2. Behaviour: migrants vs residents, cohort 2024, Nov-Mar, days in the basin", "",
          f"Windows on UTC days with the daily position within {BASIN_KM:.0f} km of the study-area centre. "
          "Groupings: 'rule' (primary) and 'consensus' (both methods agree; sensitivity); 'adult' = age not "
          "stated (primary), 'all' = including tagged-young birds.", ""]
    L += _md(groups)
    for fam, title in (("a_shares_scheduled", "(a) TESTED: state shares among scheduled (k=0) windows"),
                       ("a_shares_all_modes", "(a') DESCRIPTIVE: state shares among all windows (trigger-biased)"),
                       ("b_kinematics", "(b) TESTED: flight kinematics (per-bird medians; windows with GPS, "
                                        "mostly continuous mode)")):
        S = T[T.family == fam]
        if not len(S):
            continue
        cols = [c for c in ["grouping", "variable", "n_migrant", "n_resident", "median_migrant", "median_resident",
                            "diff_mean", "hedges_g", "p_perm", "p_perm_sex_strata", "p_holm", "test"] if c in S]
        L += ["", f"### {title}", ""] + _md(S[cols])
    L += ["", "### (c) TESTED: classifier on mean h in K=2 flight / ground (bird-level AUC, grouped CV, "
          "bird-label permutation)", ""]
    if len(clf):
        L += _md(clf[["run", "target", "feature_set", "n_pos_birds", "n_neg_birds", "auc_mean", "auc_sd",
                      "null_auc_q95", "p_auc"]])
        L += ["", "h beyond the nuisance floor (Freedman-Lane nested null):", ""]
        L += _md(nest[["run", "target", "delta_auc_mean", "delta_auc_sd", "n_perm", "p_delta_auc"]]) if len(nest) \
            else ["(none)"]
    else:
        L += ["not tested: too few birds per group"]
    L += ["", "## Caveats", "",
          "- Shares are window shares of scheduled bursts (k=0), not time budgets; kinematics come from continuous "
          "mode, which starts on fast movement.",
          "- Migrant-flagged birds were mostly deployed in October; deploy month is in the nuisance floor, and a "
          "high floor AUC is the expected confound.",
          "- p-values are uncorrected unless in p_holm (Holm within family and grouping). Female birds fly faster "
          "and turn wider: p_perm_sex_strata permutes labels within sex.",
          f"- Runtime {meta.get('runtime_s')} s."]
    (out / "residency_summary.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def run(gps: Path, windows: Path, ref_csv: Path, states2: Path, states6: Path, run_dir: Path, out: Path,
        min_days: int = 120, n_perm: int = 2000, n_repeats: int = 20, seed: int = 0, workers: int = 1,
        state_col: str = "state_hmm") -> dict:
    t0 = time.time()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    rng = np.random.default_rng(seed)
    files = sorted(gps.glob("*.parquet"))
    wdirs = {d.name: d for d in sorted(p for p in windows.iterdir() if p.is_dir())
             if (d / "meta.parquet").exists() and pq.ParquetFile(d / "meta.parquet").metadata.num_rows > 0}
    birds = sorted({f.stem for f in files} | set(wdirs))
    ref = reference.load_birds(ref_csv, birds).set_index("bird")
    log(f"{len(files)} GPS files, {len(wdirs)} birds with windows")

    # part 1: tracks (positions in memory only)
    G = Parallel(n_jobs=workers)(delayed(read_bird)(f, ref.at[f.stem, "deploy_on"]) for f in files)
    G = [g for g in G if g["daily"] is not None and len(g["daily"])]
    centre = tuple(np.median(np.array([g["median"] for g in G]), axis=0))
    tracks, rows = {}, []
    for g in G:
        tr = track(g["daily"], centre)
        c = classify_bird(tr, min_days)
        tracks[g["bird"]] = (tr, c.pop("_fit"))
        rows.append({"bird": g["bird"], "gps_first_year": int(tr.year.iat[0]), **c})
    del G
    R = pd.DataFrame(rows).set_index("bird")
    R["young"] = ref.loc[R.index, "tagged_young"].notna().to_numpy()
    R["migrant_flag"] = ref.loc[R.index, "migrant"].notna().to_numpy()
    R["sex"] = ref.loc[R.index, "sex"].to_numpy()
    R["strategy"] = R["rule_class"]
    R["strategy_nsd"] = R["nsd_class"]
    R["consensus"] = np.where(R.strategy == R.strategy_nsd, R.strategy, "disagree")
    log(f"part 1: {len(R)} birds classified ({time.time() - t0:.0f} s)")

    both = ~R.strategy.isin(["insufficient", "undetermined"]) & ~R.strategy_nsd.isin(["insufficient", "undetermined"])
    kappa = {"n": int(both.sum()), "agree": float((R.strategy[both] == R.strategy_nsd[both]).mean()) if both.any() else np.nan,
             "kappa": float(cohen_kappa_score(R.strategy[both], R.strategy_nsd[both])) if both.sum() > 1 else np.nan}
    ag = []
    for meth, col in (("rule", "strategy"), ("nsd", "strategy_nsd"), ("consensus", "consensus")):
        for age, sub in (("other", R[~R.young]), ("young", R[R.young])):
            for flag, s in (("flagged migrant", sub[sub.migrant_flag]), ("not flagged", sub[~sub.migrant_flag])):
                vc = s[col].value_counts()
                ag.append({"method": meth, "age": age, "flag": flag, "birds": len(s),
                           **{c: int(vc.get(c, 0)) for c in ("migrant", "resident", "disperser", "nomad",
                                                             "undetermined", "insufficient", "disagree")}})
    agree = pd.DataFrame(ag)
    agree = agree.loc[:, (agree != 0).any(axis=0) | agree.columns.isin(["birds"])]
    wfirst = {b: pd.read_parquet(d / "meta.parquet", columns=["t_start"])["t_start"].min() for b, d in wdirs.items()}
    R["cohort"] = [("2024" if pd.Timestamp(wfirst[b]).year <= 2024 else "2025") if b in wfirst else
                   ("2024" if R.at[b, "gps_first_year"] <= 2024 else "2025") for b in R.index]
    R["cohort_source"] = ["windows" if b in wfirst else "gps" for b in R.index]
    km = [c for c in R if c.endswith("_km")]
    R[km] = R[km].round(0)
    doy = [c for c in R if "_doy_" in c]
    R[doy] = R[doy].round(0)
    R["basin_days_share"] = R["basin_days_share"].round(3)
    cols = ["young", "migrant_flag", "sex", "cohort", "cohort_source", "strategy", "strategy_nsd", "consensus",
            "rule_note", "nsd_note", "track_days", "n_days", "covers_spring", "max_dist_km", "max_north_km",
            "max_dist_deploy_km", "max_north_deploy_km", "summer_dist_km", "winter_dist_km", "basin_days_share",
            "n_migrations", "dep_doy_rule_y1", "ret_doy_rule_y1", "dep_doy_rule_y2", "ret_doy_rule_y2",
            "nsd_best_aic", "nsd_delta_km", "dep_doy_nsd", "ret_doy_nsd"] + [f"daic_{m}" for m in MODELS]
    out.mkdir(parents=True, exist_ok=True)  # part 1 outputs first, so they survive a failure in part 3
    R[cols].reset_index().to_csv(out / "residency.csv", index=False)
    agree.to_csv(out / "agreement.csv", index=False)
    fig_nsd_birds(R, tracks, out / "nsd_birds.png")
    fig_nsd_by_strategy(R, tracks, out / "nsd_by_strategy.png")

    # part 3: behaviour
    basin = {b: tracks[b][0] for b in tracks}
    basin_days = {}
    for b, tr in basin.items():
        s = pd.Series(tr.dc_km.to_numpy(), index=tr.epoch_day.to_numpy())
        full = s.reindex(np.arange(s.index.min(), s.index.max() + 1))
        full = full.ffill(limit=BASIN_FILL_DAYS).combine_first(full.bfill(limit=BASIN_FILL_DAYS))
        basin_days[b] = full.index[(full <= BASIN_KM).to_numpy()].to_numpy()
    st = {}
    for name, sdir, K in (("k2", states2, 2), ("k6", states6, 6)):
        S = pd.read_parquet(sdir / "states.parquet", columns=["bird", "row", state_col])
        S["bird"] = S["bird"].astype(str)
        if int(S[state_col].max()) + 1 != K:
            raise ValueError(f"{sdir}: expected K={K}, found {int(S[state_col].max()) + 1}")
        st[name] = dict(tuple(S.groupby("bird", sort=False)))
        del S
    wb = [b for b in wdirs if b in basin_days]
    res = Parallel(n_jobs=workers)(
        delayed(behaviour_bird)(wdirs[b], run_dir, st["k2"].pop(b, None), st["k6"].pop(b, None), basin_days[b],
                                state_col) for b in wb)
    del st
    P = {p["bird"]: p for p in res}
    log(f"part 3 pass: {len(P)} birds ({time.time() - t0:.0f} s)")
    B = R.loc[[b for b in R.index if b in P]].copy()
    B["cont_share"] = [P[b]["cont_share"] for b in B.index]
    B["track_days_windows"] = [(P[b]["last"] - P[b]["first"]).total_seconds() / 86400 for b in B.index]
    dep = pd.to_datetime(ref.loc[B.index, "deploy_on"], utc=True).dt.month
    B["deploy_month"] = dep.fillna(pd.Series({b: P[b]["first"].month for b in B.index})).to_numpy()
    scal = [k for k in next(iter(P.values())) if not k.startswith("h_") and k not in ("bird", "first", "last")]
    BB = pd.DataFrame([{k: P[b][k] for k in scal} | {"bird": b} for b in B.index]).set_index("bird")

    groupings = {}
    for gname, col in (("rule", "strategy"), ("consensus", "consensus")):
        for age in ("adult", "all"):
            sub = B[(B.cohort == "2024") & B[col].isin(["migrant", "resident"])]
            if age == "adult":
                sub = sub[~sub.young]
            groupings[f"{gname}|{age}"] = (sub[col] == "migrant").astype(int)
    grp_rows, T = [], []
    share_s = [c for c in BB if c.startswith("share_") and c.endswith("_sched")]
    share_a = [c for c in BB if c.startswith("share_") and c.endswith("_all")]
    kin = [c for c in BB if c.split("_")[0] in ("speed", "radius", "climb")]
    for gk, lab in groupings.items():
        X = BB.loc[lab.index]
        n_s = X["n_k2_sched"]
        grp_rows.append({"grouping": gk, "migrants": int(lab.sum()), "residents": int((1 - lab).sum()),
                         "migrants_f/m/missing": "/".join(str(int(((B.loc[lab.index[lab == 1], "sex"].fillna("x")) == s).sum()))
                                                          for s in ("f", "m", "x")),
                         "residents_f/m/missing": "/".join(str(int(((B.loc[lab.index[lab == 0], "sex"].fillna("x")) == s).sum()))
                                                           for s in ("f", "m", "x")),
                         "birds_with_basin_windows": int((X["n_sel"] > 0).sum())})
        sex = B.loc[lab.index, "sex"]
        VS = X[share_s].where(n_s >= MIN_SHARE_WINDOWS, axis=0)
        VA = X[share_a].where(X["n_k2_all"] >= MIN_SHARE_WINDOWS, axis=0)
        for fam, V, tested in (("a_shares_scheduled", VS, True), ("a_shares_all_modes", VA, False),
                               ("b_kinematics", X[kin], True)):
            rows_ = compare_table(V, lab, sex, fam, gk, n_perm, rng, tested)
            if tested:
                ph = holm(np.array([r.get("p_perm", np.nan) for r in rows_]))
                for r, p in zip(rows_, ph):
                    r["p_holm"] = p
            T += rows_
    T = pd.DataFrame(T)
    groups = pd.DataFrame(grp_rows)
    BB.join(B[["strategy", "consensus", "young", "sex", "cohort", "deploy_month", "track_days_windows"]]
            ).reset_index().to_csv(out / "behaviour_birds.csv", index=False)
    T.to_csv(out / "behaviour_tests.csv", index=False)

    # (c) classifiers
    specs = {}
    rn = run_dir.name
    for gk, lab in groupings.items():
        if min(lab.sum(), (1 - lab).sum()) < MIN_GROUP_BIRDS:
            continue
        for tag, keys in (("flight", ["flight"]), ("ground", ["ground"]), ("flight_ground", ["flight", "ground"])):
            H = pd.concat([h_frame(P, lab.index, k) for k in keys], axis=1)
            ok = H.notna().all(axis=1)
            y = lab[ok]
            if min(y.sum(), (1 - y).sum()) < MIN_GROUP_BIRDS:
                continue
            Nu = nuisance(B.loc[y.index])
            Hy = H.loc[y.index]
            for fs, F, nh in (("nuisance", Nu, 0), ("mean_h", Hy, 0), ("h_plus_nuisance", pd.concat([Hy, Nu], axis=1), Hy.shape[1])):
                sp = dm.make_spec(F, y, "bird", f"{rn}:{tag}", f"migrant_vs_resident|{gk}", fs, n_h=nh)
                if sp is not None:
                    specs[f"{gk}|{tag}|{fs}"] = sp
    if specs:
        clf, nest = dm.run_classifiers(specs, n_repeats, {"bird": n_perm}, seed, workers)
    else:
        clf, nest = pd.DataFrame(), pd.DataFrame()
    log(f"part 3 tests done ({time.time() - t0:.0f} s)")

    clf.to_csv(out / "classifiers.csv", index=False)
    nest.to_csv(out / "nested.csv", index=False)
    meta = {"gps": str(gps), "windows": str(windows), "reference_csv": str(ref_csv), "states2": str(states2),
            "states6": str(states6), "run": str(run_dir), "state_col": state_col, "min_days": min_days,
            "n_perm": n_perm, "n_repeats": n_repeats, "seed": seed, "workers": workers,
            "birds_gps": len(R), "birds_windows": len(P),
            "birds_without_deploy_on": int(ref.loc[R.index, "deploy_on"].isna().sum()),
            "thresholds": {k: globals()[k] for k in ("FIRST_DAYS", "FIT_DAYS", "MIN_DISP_KM", "MIGRANT_NORTH_KM",
                                                     "RETURN_WITHIN_DAYS", "MIN_STAY_DAYS", "SETTLE_DAYS",
                                                     "FINAL_DAYS", "ROLL_DAYS", "ROLL_MIN", "PHI_MIN", "PHI_MAX",
                                                     "BASIN_KM", "BASIN_FILL_DAYS", "MIN_SHARE_WINDOWS",
                                                     "MIN_STATE_WINDOWS", "MIN_GROUP_BIRDS")},
            "months": {"spring": SPRING_MONTHS, "summer": SUMMER_MONTHS, "winter": WINTER_MONTHS,
                       "compare": COMPARE_MONTHS},
            "states6_names": STATES6, "flight6": FLIGHT6, "kappa": kappa, "runtime_s": None}
    meta["runtime_s"] = round(time.time() - t0, 1)
    write_summary(out, R.assign(young=R.young.astype(bool)), agree, kappa, groups, T, clf, nest, meta)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    log(f"done ({meta['runtime_s']} s)")
    return meta


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gps", type=Path, required=True, help="raw GPS directory (<bird>.parquet)")
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--states2", type=Path, required=True, help="K=2 states directory (0 ground, 1 flight)")
    ap.add_argument("--states6", type=Path, required=True, help="K=6 states directory of the same run")
    ap.add_argument("--run", type=Path, required=True, help="run directory with embeddings/<bird>.npz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-days", type=int, default=120)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--state-col", default="state_hmm", choices=["state_hmm", "state"])
    a = ap.parse_args(argv)
    run(a.gps, a.windows, a.reference, a.states2, a.states6, a.run, a.out, a.min_days, a.n_perm, a.n_repeats,
        a.seed, a.workers, a.state_col)


if __name__ == "__main__":
    main()
