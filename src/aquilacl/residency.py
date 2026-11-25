"""Movement strategy of each bird from its GPS track, and behaviour of migrants vs residents in the basin.

Privacy: positions exist only in memory. Per-bird outputs hold distances from the study-area centre (km, rounded to
KM_ROUND = 50 km; no northing, no second origin except the 30-day running-median and NSD plateau distances), day of
year or days since deployment, and classes. No latitude, longitude, centroid or study-area centre is written or
logged.

Part 1. Tracks (one worker per bird, raw GPS parquet)
- Fixes: finite, not (0, 0), latitude/longitude in range, at or after the reference deploy-on date (if given) and at
  or before the latest deploy-off date when every deployment of the bird is closed.
- Daily position: median of the hourly median latitude/longitude of each UTC day (a 1 Hz bout counts as one hour).
  Bird median: median of all its fixes. Stationary end: a trailing run of >= STILL_DAYS days within STILL_KM of the
  last day's position with daily maximum ground speed < STILL_MS is cut off (possible mortality or tag loss) and
  its length kept (end_still_days).
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
  inside the fitted days; nomad needs sqrt(b T) >= MIN_DISP_KM. Censoring: 'disperser' needs >= SETTLE_DAYS of
  track after the departure midpoint (otherwise a migrant that has not yet returned), else 'undetermined'.
  Departure / return day-of-year = the dates of the two sigmoid midpoints (half of the NSD change, not onsets).
Part 1b. Rule classes (primary 'strategy'; whole track; d0 / north0 from the deployment centroid, 30-day running
  median 'roll' of d0 with >= ROLL_MIN positions). A far-north excursion starts on the first day >= MIGRANT_NORTH_KM
  north; it is a migration if the bird comes back below RETURN_NORTH_KM north within RETURN_WITHIN_DAYS of leaving
  (departure / return = first day beyond / back below RETURN_NORTH_KM north, NaN when the previous daily position
  is more than MAX_GAP_DAYS earlier; the return is not required to reach the deployment centroid, because birds
  trapped in October on passage were deployed off their winter range). NSD sigmoid midpoints are NaN likewise when
  they fall in a gap > MAX_GAP_DAYS. Threshold sensitivity: the classes are recomputed at SENS_NORTH_KM from the
  deployment centroid and from the study-area centre (threshold_sensitivity.csv). In order:
  insufficient      span < min_days or fewer than min_days/2 daily positions;
  migrant           >= 1 migration;
  migrant_departed  a far-north excursion that is still open when the track ends (< RETURN_WITHIN_DAYS after leaving;
                    an open excursion longer than that is 'disperser');
  resident_full     roll never > MIN_DISP_KM and span >= FULL_YEAR_DAYS;
  present_partial_year  roll never > MIN_DISP_KM, span < FULL_YEAR_DAYS: present summer-autumn, annual cycle
                    unobserved (not a resident in any comparison);
  juvenile_dispersal  tagged-young bird whose roll exceeds MIN_DISP_KM (or that leaves far north without return);
  disperser         roll > MIN_DISP_KM over the last FINAL_DAYS and continuously away for >= SETTLE_DAYS;
  away_at_end       away over the last FINAL_DAYS but for < SETTLE_DAYS (settling and a seasonal stay not separable);
  nomad             everything else: sustained excursions beyond MIN_DISP_KM that end back near the deployment area.
  Coarse classes for agreement and for part 3: migrant = migrant + migrant_departed; resident = resident_full (NSD:
  resident over >= FULL_YEAR_DAYS, a shorter NSD resident is present_partial_year); disperser = disperser +
  juvenile_dispersal (+ NSD disperser); nomad; other classes are not compared. 'consensus' = the coarse class where
  both methods agree.

Part 2. Outputs in --out: residency.csv (per bird), agreement.csv, threshold_sensitivity.csv, residency_summary.md,
nsd_birds.png,
nsd_by_strategy.png, behaviour_groups.csv, behaviour_tests.csv, behaviour_birds.csv (one row per bird and selection),
classifiers.csv, nested.csv, metadata.json.

Part 3. Behaviour comparison, coarse migrant (1) vs coarse resident (0), bird level. Window days are UTC days
(t_start floored in UTC); a window is in an area on a day if the nearest daily position within BASIN_FILL_DAYS days is
inside it. Selections: 'basin' = within BASIN_KM of the study-area centre; 'winter_range' = within BASIN_KM of the
bird's own Nov-Mar median daily position (EXTRA: migrants and residents are then compared in different places).
Variants (cohort = year of the reference deploy-on date, earliest deployment; first GPS-fix year if missing):
  c2024_novmar_basin       cohort 2024, COMPARE_MONTHS (Nov-Mar), basin days (the original comparison);
  all_novmar_basin         all cohorts, Nov-Mar, basin days; permutation within cohort;
  all_allmonths_basin      all cohorts, all months, basin days; permutation within cohort;
  all_novmar_winter_range  all cohorts, Nov-Mar, winter-range days (extra, labelled); permutation within cohort.
Each variant has groupings 'rule' (primary; residents = resident_full), 'rule_partial' (sensitivity; residents =
resident_full + present_partial_year) and 'consensus', each for non-young birds ('adult': tagged young not stated)
and all ages. Group sizes (and the cohorts holding both groups) are in behaviour_groups.csv.
Testability: the nuisance floor (track days, deploy month, continuous-mode share, cohort) of every grouping is fitted
first (classifier on the birds with selected windows); at AUC >= NUISANCE_AUC_MAX the grouping is "not testable in
this design": its (a)/(b) rows keep counts and effect sizes, p-values are NaN.
(a) TESTED: share of K=2 flight among scheduled windows with a state: sensitivity_k == 0, first window of its burst
    (k_in_segment == 0), sun above the horizon (continuous mode is triggered by fast movement and would inflate
    flight); birds need >= MIN_SHARE_WINDOWS. DESCRIPTIVE: the K=6 shares of the same windows, and all shares over
    all windows.
(b) TESTED: per-bird medians of speed (m/s), turn radius (m; speed / radians(turn_abs_mean), turn > 0) and climb
    (m/s) in K=2 flight and in each K=6 flight state; GPS is almost only in continuous windows, so these come
    from all windows with GPS (mostly sensitivity_k == 1); birds need >= MIN_STATE_WINDOWS.
    Tests: difference in means with labels permuted between birds (cohort 2024 variant:
    aquilacl.demography.perm_mean_diff, p_perm_sex_strata within sex). All-cohort variants use only cohorts that hold
    both groups; the statistic is the weighted within-cohort difference (perm_strat_diff), p_perm permutes within
    cohort and p_perm_sex_strata within cohort x sex (missing sex = own stratum; the within-stratum statistic; NaN
    when no stratum holds both groups). Hedges g (pooled); Holm adjustment of p_perm over the declared family of each
    grouping (K=2 share + all kinematics variables; untestable rows count towards the family size). A side needs
    >= MIN_GROUP_BIRDS birds with a value (in cohorts holding both groups), else 'not tested'.
(c) TESTED where testable (rule groupings with >= MIN_CLF_BIRDS per class): grouped-CV logistic classifier (aquilacl.demography.run_classifiers: StratifiedGroupKFold
    by bird, labels permuted between birds (not within cohort), Freedman-Lane nested null for h beyond nuisance) on
    the mean L2-normalised h of the run in K=2 flight, in K=2 ground, and both; all selected windows. Nuisance floor:
    track days, deploy month (sin, cos), continuous-mode share (all windows of the bird), and cohort (one-hot
    deployment year) when it varies.
    Each h set has its own nuisance spec on the same birds (run tag '<run>:flight' etc.), so the nested comparison
    uses identical folds.
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
MAX_GAP_DAYS = 7  # a migration date is NaN when the crossing lies in a GPS gap longer than this
STILL_DAYS, STILL_KM, STILL_MS = 14, 1.0, 2.0  # stationary track end (possible mortality or tag loss)
KM_ROUND = 50  # per-bird distances in residency.csv are rounded to this
SENS_NORTH_KM = (750.0, 1000.0, 1250.0)  # threshold sensitivity of the far-north rule
NUISANCE_AUC_MAX = 0.9  # nuisance floor at or above this: the grouping is "not testable in this design"
MIN_CLF_BIRDS = 8  # birds per class for the h classifier
MIN_DISP_KM = 200.0
MIGRANT_NORTH_KM = 1000.0
RETURN_NORTH_KM = 500.0
RETURN_WITHIN_DAYS = 365
FULL_YEAR_DAYS = 330
MIN_STAY_DAYS = 30
SETTLE_DAYS = 240
FINAL_DAYS = 60
ROLL_DAYS, ROLL_MIN = 30, 10
PHI_MIN, PHI_MAX = 1.0, 60.0
SPRING_MONTHS = (3, 4, 5)
SUMMER_MONTHS, WINTER_MONTHS = (6, 7, 8), (12, 1, 2)
COMPARE_MONTHS = (11, 12, 1, 2, 3)
ALL_MONTHS = tuple(range(1, 13))
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
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
GPS_COLS = ["timestamp", "location-long", "location-lat", "ground-speed"]
log = dm.log


# ---------------------------------------------------------------- geometry and daily positions

def hav_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def clean_fixes(g: pd.DataFrame, deploy_on=None, deploy_off=None) -> pd.DataFrame:
    """Valid fixes (finite, in range, not (0, 0), within [deploy_on, deploy_off]) with a UTC 'date' column."""
    g = g.rename(columns={"location-lat": "lat", "location-long": "lon", "ground-speed": "speed"})
    g = g.assign(timestamp=pd.to_datetime(g["timestamp"], utc=True)).dropna(subset=["timestamp", "lat", "lon"])
    ok = (g.lat.abs() <= 90) & (g.lon.abs() <= 180) & ((g.lat.abs() > 1e-6) | (g.lon.abs() > 1e-6))
    if deploy_on is not None and pd.notna(deploy_on):
        ok &= g.timestamp >= pd.Timestamp(deploy_on)
    if deploy_off is not None and pd.notna(deploy_off):
        ok &= g.timestamp <= pd.Timestamp(deploy_off)
    g = g[ok]
    return g.assign(date=g.timestamp.dt.floor("D"))


def daily_positions(g: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day: median of the hourly median latitude/longitude (so a 1 Hz bout counts as one hour, not as
    thousands of fixes), number of fixes, and the day's maximum ground speed (m/s; NaN without speeds). In memory only."""
    hr = g.timestamp.dt.floor("h")
    h = g.groupby(["date", hr])[["lat", "lon"]].median()
    d = h.groupby(level="date").median()
    d["n"] = g.groupby("date").size()
    d["vmax"] = g.groupby("date")["speed"].max() if "speed" in g else np.nan
    return d.reset_index()


def stationary_tail(daily: pd.DataFrame) -> int:
    """Calendar days of the trailing run of days whose position is within STILL_KM of the last day's and whose
    maximum ground speed is below STILL_MS (days without a speed do not count as still)."""
    if len(daily) < 2:
        return 0
    dist = hav_km(daily.lat.iat[-1], daily.lon.iat[-1], daily.lat.to_numpy(), daily.lon.to_numpy())
    still = (dist < STILL_KM) & (daily["vmax"].to_numpy(float) < STILL_MS)
    n = len(still) - (np.flatnonzero(~still)[-1] + 1) if (~still).any() else len(still)
    return int((daily.date.iat[-1] - daily.date.iat[-n]).days + 1) if n else 0


def read_bird(path: Path, deploy_on, deploy_off=None) -> dict:
    """Daily positions of one bird (in memory). A stationary end of >= STILL_DAYS is cut off and its length kept."""
    have = pq.read_schema(path).names
    g = clean_fixes(pd.read_parquet(path, columns=[c for c in GPS_COLS if c in have]), deploy_on, deploy_off)
    med = (float(g.lat.median()), float(g.lon.median())) if len(g) else None
    daily = daily_positions(g) if len(g) else None
    still = stationary_tail(daily) if daily is not None else 0
    if still >= STILL_DAYS:  # keep up to the first stationary day (a wholly stationary track keeps one day)
        daily = daily[daily.date <= daily.date.iat[-1] - pd.Timedelta(days=still - 1)].reset_index(drop=True)
    return {"bird": path.stem, "daily": daily, "median": med, "end_still_days": still}


def deploy_off_dates(ref_csv: Path, birds) -> pd.Series:
    """Latest deploy-off per bird, only when every deployment of the bird has one (an open deployment: NaT).
    reference.load_birds does not return deploy_off, so the column is read here with its column names."""
    head = pd.read_csv(ref_csv, nrows=0).columns
    idc = next((c for c in reference.COLUMNS["id"] if c in head), None)
    offc = next((c for c in reference.COLUMNS["deploy_off"] if c in head), None)
    out = pd.Series(pd.NaT, index=pd.Index(list(birds), name="bird"), dtype="datetime64[ns, UTC]")
    if idc is None or offc is None:
        return out
    r = pd.read_csv(ref_csv, usecols=[idc, offc], dtype=str, keep_default_na=False).drop_duplicates()
    r["bird"] = reference.bird_id(r[idc])
    r["off"] = pd.to_datetime(r[offc].str.strip(), errors="coerce", format="mixed", utc=True)
    g = r.groupby("bird")["off"].agg(lambda s: s.max() if s.notna().all() else pd.NaT)
    return out.fillna(g.reindex(out.index))


def track(daily: pd.DataFrame, centre: tuple[float, float]) -> pd.DataFrame:
    """Per day: days since deployment, day of year, month, distances (km) and NSD (km^2). No positions returned.

    dw_km: distance to the bird's own winter-range centre (median Nov-Mar daily position; NaN without such days)."""
    day = ((daily["date"] - daily["date"].iloc[0]) / pd.Timedelta(days=1)).round().astype(int).to_numpy()
    first = daily[day < FIRST_DAYS]
    c0 = (first.lat.median(), first.lon.median())
    lat, lon = daily.lat.to_numpy(), daily.lon.to_numpy()
    d0 = hav_km(c0[0], c0[1], lat, lon)
    dc = hav_km(centre[0], centre[1], lat, lon)
    win = daily["date"].dt.month.isin(COMPARE_MONTHS).to_numpy()
    dw = hav_km(np.median(lat[win]), np.median(lon[win]), lat, lon) if win.any() else np.full(len(lat), np.nan)
    return pd.DataFrame({"day": day, "epoch_day": epoch_days(daily["date"]),
                         "doy": daily["date"].dt.dayofyear.to_numpy(), "month": daily["date"].dt.month.to_numpy(),
                         "year": daily["date"].dt.year.to_numpy(), "d0_km": d0, "nsd_km2": d0 ** 2,
                         "north0_km": (lat - c0[0]) * KM_PER_DEG, "dc_km": dc, "dw_km": dw,
                         "northc_km": (lat - centre[0]) * KM_PER_DEG})


def epoch_days(t) -> np.ndarray:
    """UTC day number (days since 1970-01-01) of timestamps; tz-naive input is taken as UTC."""
    t = pd.to_datetime(pd.Series(t), utc=True)
    return ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(days=1)).to_numpy(np.int64)


def area_days(epoch_day: np.ndarray, dist_km: np.ndarray, max_km: float = BASIN_KM,
              fill: int = BASIN_FILL_DAYS) -> np.ndarray:
    """Epoch days (sorted) on which the nearest daily position within `fill` days is within max_km.

    Days with a position use it; days without one use the nearest position (ties: the earlier day)."""
    e = np.asarray(epoch_day, np.int64)
    dist = np.asarray(dist_km, float)
    ok = np.isfinite(dist)
    e, dist = e[ok], dist[ok]
    if not len(e):
        return np.array([], np.int64)
    o = np.argsort(e)
    e, dist = e[o], dist[o]
    allday = np.arange(e[0] - fill, e[-1] + fill + 1)
    j = np.searchsorted(e, allday)
    lo, hi = np.clip(j - 1, 0, len(e) - 1), np.clip(j, 0, len(e) - 1)
    dlo, dhi = np.abs(allday - e[lo]), np.abs(e[hi] - allday)
    near = np.where(dhi < dlo, hi, lo)
    gap = np.minimum(dlo, dhi)
    return allday[(gap <= fill) & (dist[near] <= max_km)]


def covers_spring(tr: pd.DataFrame) -> bool:
    return len(set(tr.loc[tr.day >= FIRST_DAYS, "month"]) & set(SPRING_MONTHS)) >= 2


def _doy(tr: pd.DataFrame, day: float) -> float:
    """Day of year of (first date + day), via the first row's epoch day."""
    if not np.isfinite(day):
        return np.nan
    e = int(tr.epoch_day.iat[0] - tr.day.iat[0] + round(day))
    return float((pd.Timestamp("1970-01-01", tz="UTC") + pd.Timedelta(days=e)).dayofyear)


def _in_gap(day: np.ndarray, x: float) -> float:
    """Length (days) of the interval between the observed days that bracket day x (0 when x is an observed day)."""
    j = int(np.searchsorted(day, x))
    if j < len(day) and day[j] == x:
        return 0.0
    if j == 0 or j == len(day):
        return np.inf
    return float(day[j] - day[j - 1])


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
        res["dep_doy_nsd"], res["ret_doy_nsd"] = (_doy(tr, x) if _in_gap(tr.day.to_numpy(), x) <= MAX_GAP_DAYS
                                                  else np.nan for x in (p[1], p[1] + p[3]))
    if chosen == "disperser" and span - p[1] < SETTLE_DAYS:
        cls, note = "undetermined", "away at track end; too short to tell settling from a seasonal stay"
    if chosen == "resident" and span < FULL_YEAR_DAYS:  # same definition as the rule class
        cls, note = "present_partial_year", PARTIAL_LABEL
    res["nsd_class"], res["nsd_note"] = cls, note.strip("; ")
    return res


# ---------------------------------------------------------------- rule classes

def _crossing_ok(day: np.ndarray, i: int) -> bool:
    """True when the crossing into observed day index i is dated: i > 0 and the gap from the previous observed day
    is at most MAX_GAP_DAYS (otherwise the first fix after a gap would stand in for the date)."""
    return i > 0 and day[i] - day[i - 1] <= MAX_GAP_DAYS


ORIGINS = {"deploy": ("d0_km", "north0_km"), "centre": ("dc_km", "northc_km")}


def rule_classify(tr: pd.DataFrame, min_days: int, young: bool = False, north_km: float = MIGRANT_NORTH_KM,
                  origin: str = "deploy") -> dict:
    """Rule class of one track (module docstring, Part 1b). north_km / origin: threshold sensitivity (the return
    threshold RETURN_NORTH_KM and MIN_DISP_KM stay fixed)."""
    span = int(tr.day.max()) if len(tr) else 0
    res = {"rule_class": "insufficient", "rule_note": "", "n_migrations": 0, "max_roll_d0_km": np.nan,
           **{f"{k}_doy_rule_y{i}": np.nan for i in (1, 2) for k in ("dep", "ret")}}
    if span < min_days or len(tr) < min_days / 2:
        res["rule_note"] = "short track"
        return res
    dcol, ncol = ORIGINS[origin]
    day, d0, n0 = tr.day.to_numpy(), tr[dcol].to_numpy(), tr[ncol].to_numpy()
    far, south = n0 >= north_km, n0 < RETURN_NORTH_KM
    events, far_open, cur = [], None, -1
    while True:
        fi = np.flatnonzero(far & (day > cur))
        if not len(fi):
            break
        f = fi[0]
        before = np.flatnonzero(south[:f])
        dep = before[-1] + 1 if len(before) else 0
        after = np.flatnonzero(south[f:])
        if not len(after):
            far_open = dep
            break
        ret = f + after[0]
        if day[ret] - day[dep] <= RETURN_WITHIN_DAYS:
            events.append((dep, ret))
        cur = day[ret]
    res["n_migrations"] = len(events)
    for i, (a, b) in enumerate(events[:2], 1):
        res[f"dep_doy_rule_y{i}"] = float(tr.doy.iat[a]) if _crossing_ok(day, a) else np.nan
        res[f"ret_doy_rule_y{i}"] = float(tr.doy.iat[b]) if _crossing_ok(day, b) else np.nan
    full = pd.Series(d0, index=day).reindex(np.arange(span + 1))
    roll = full.rolling(ROLL_DAYS, min_periods=ROLL_MIN, center=True).median()
    res["max_roll_d0_km"] = float(roll.max()) if roll.notna().any() else np.nan
    stays = not (roll > MIN_DISP_KM).any()
    last = d0[day > span - FINAL_DAYS]
    back = np.flatnonzero(d0 < MIN_DISP_KM)
    away_since = day[back[-1]] + 1 if len(back) else 0
    if events:
        cls, note = "migrant", ""
    elif far_open is not None and not young:
        cls, note = (("migrant_departed", "far north at track end") if span - day[far_open] < RETURN_WITHIN_DAYS
                     else ("disperser", "far north for over a year"))
    elif stays:
        cls, note = ("resident_full", "") if span >= FULL_YEAR_DAYS else ("present_partial_year", PARTIAL_LABEL)
    elif young:
        cls, note = "juvenile_dispersal", ""
    elif len(last) and np.median(last) > MIN_DISP_KM:
        cls, note = ("disperser", "") if span - away_since >= SETTLE_DAYS else \
            ("away_at_end", "away at track end; too short to tell settling from a seasonal stay")
    else:
        cls, note = "nomad", ""
    res["rule_class"], res["rule_note"] = cls, note
    return res


PARTIAL_LABEL = "present summer-autumn, annual cycle unobserved (track < FULL_YEAR_DAYS)"


COARSE = {"migrant": "migrant", "migrant_departed": "migrant", "resident": "resident", "resident_full": "resident",
          "disperser": "disperser", "juvenile_dispersal": "disperser", "nomad": "nomad"}
RULE_CLASSES = ("migrant", "migrant_departed", "resident_full", "present_partial_year", "juvenile_dispersal",
                "disperser", "away_at_end", "nomad", "insufficient")


def coarse(cls) -> str:
    """Coarse class shared by both methods ('other' = not compared: insufficient, undetermined, away_at_end)."""
    return COARSE.get(cls, "other")


def distances(tr: pd.DataFrame) -> dict:
    def med(months, col="dc_km"):
        v = tr.loc[tr.month.isin(months), col]
        return float(v.median()) if len(v) else np.nan
    return {"track_days": int(tr.day.max()), "n_days": len(tr), "covers_spring": bool(covers_spring(tr)),
            "max_dist_km": float(tr.dc_km.max()), "max_north_km": float(tr.northc_km.max()),
            "max_dist_deploy_km": float(tr.d0_km.max()), "max_north_deploy_km": float(tr.north0_km.max()),
            "summer_dist_km": med(SUMMER_MONTHS), "winter_dist_km": med(COMPARE_MONTHS),
            "winter_dist_deploy_km": med(COMPARE_MONTHS, "d0_km"),  # summary class medians only, not per bird
            "winter_days": int(tr.month.isin(COMPARE_MONTHS).sum()),
            "basin_days_share": float((tr.dc_km <= BASIN_KM).mean()),
            "winter_basin_share": float((tr.loc[tr.month.isin(COMPARE_MONTHS), "dc_km"] <= BASIN_KM).mean())
            if tr.month.isin(COMPARE_MONTHS).any() else np.nan}


def classify_bird(tr: pd.DataFrame, min_days: int, young: bool = False) -> dict:
    n = nsd_classify(tr, min_days)
    r = rule_classify(tr, min_days, young)
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


SELECTIONS = {"novmar_basin": ("dc_km", COMPARE_MONTHS), "allmonths_basin": ("dc_km", ALL_MONTHS),
              "novmar_winter_range": ("dw_km", COMPARE_MONTHS)}
VARIANTS = {"c2024_novmar_basin": ("novmar_basin", "2024"), "all_novmar_basin": ("novmar_basin", None),
            "all_allmonths_basin": ("allmonths_basin", None), "all_novmar_winter_range": ("novmar_winter_range", None)}
EXTRA_VARIANTS = ("all_novmar_winter_range",)


def window_selection(t_start, sel_days: dict) -> dict:
    """Boolean window masks per selection: month in the selection's months and UTC day in its area days."""
    t = pd.to_datetime(pd.Series(t_start), utc=True).reset_index(drop=True)
    eday, month = epoch_days(t), t.dt.month.to_numpy()
    return {name: np.isin(month, months) & np.isin(eday, days) for name, (days, months) in sel_days.items()}


def _aggregates(sel, eday, month, k, sched, s2, s6, sp, rad, cl, h) -> dict:
    """sched: scheduled-burst windows used for shares (k == 0, first window of its burst, sun above the horizon)."""
    res = {"n_sel": int(sel.sum()), "n_sel_days": int(len(np.unique(eday[sel])))}
    res.update({f"n_sel_month{mo:02d}": int((sel & (month == mo)).sum()) for mo in range(1, 13)})
    k0 = sel & sched
    for tag, s, names in (("k2", s2, STATES2), ("k6", s6, STATES6)):
        for mode, msk in (("sched", k0), ("all", sel)):
            ok = msk & (s >= 0)
            res[f"n_{tag}_{mode}"] = int(ok.sum())
            for c, name in names.items():
                if tag == "k2" and c == 0:
                    continue
                res[f"share_{tag}_{name}_{mode}"] = float((s[ok] == c).mean()) if ok.any() else np.nan
    groups = {"k2_flight": s2 == 1, **{f"k6_{STATES6[c]}": s6 == c for c in FLIGHT6}}
    for g, msk in groups.items():
        w = sel & msk & np.isfinite(sp)
        res[f"kin_n_{g}"] = int(w.sum())
        res[f"kin_k1share_{g}"] = float((k[w] == 1).mean()) if w.any() else np.nan
        for v, a in (("speed_ms", sp), ("radius_m", rad), ("climb_ms", cl)):
            x = a[w]
            x = x[np.isfinite(x)]
            res[f"{v}_{g}"] = float(np.median(x)) if len(x) >= MIN_STATE_WINDOWS else np.nan
    for c, name in STATES2.items():
        w = sel & (s2 == c)
        res[f"n_h_{name}"] = int(w.sum())
        res[f"h_{name}"] = h[w].mean(0) if w.sum() >= MIN_STATE_WINDOWS else np.full(h.shape[1], np.nan)
    return res


def behaviour_bird(d: Path, run: Path, st2, st6, sel_days: dict, state_col: str) -> dict:
    """Bird aggregates for part 3 per selection (sel_days: name -> (area epoch days, months))."""
    m = _read(d / "meta.parquet", ["t_start", "sensitivity_k", "k_in_segment", "sun_elev_deg"])
    n = len(m)
    t = pd.to_datetime(m["t_start"], utc=True).reset_index(drop=True)
    k = m["sensitivity_k"].to_numpy()
    first = m["k_in_segment"].to_numpy() == 0 if "k_in_segment" in m else np.ones(n, bool)
    sun = m["sun_elev_deg"].to_numpy(float) > 0 if "sun_elev_deg" in m else np.ones(n, bool)
    sched = (k == 0) & first & sun
    res = {"bird": d.name, "first": t.min(), "last": t.max(), "n_all": n,
           "cont_share": float((k == 1).mean()) if n else np.nan}
    s2, s6 = _states(st2, n, state_col), _states(st6, n, state_col)
    f = _read(d / "features.parquet", ["speed_mean", "turn_abs_mean", "climb_mean"]).reindex(
        columns=["speed_mean", "turn_abs_mean", "climb_mean"])
    sp, tu, cl = (f[c].to_numpy(float) for c in f.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        rad = np.where(tu > 0, sp / np.radians(tu), np.nan)
    h = np.load(run / "embeddings" / f"{d.name}.npz")["h"]
    if len(h) != n:
        raise ValueError("embedding rows differ from the windows of one bird")
    h = h.astype(np.float32)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    masks = window_selection(t, sel_days)
    eday = epoch_days(t)
    month = t.dt.month.to_numpy()
    res["sel"] = {name: _aggregates(msk, eday, month, k, sched, s2, s6, sp, rad, cl, h) for name, msk in masks.items()}
    return res


# ---------------------------------------------------------------- comparison

def holm(p: np.ndarray, m: int | None = None) -> np.ndarray:
    """Holm adjustment over a declared family of m tests (default len(p)); rows that could not be tested (NaN)
    still count towards m, so the family does not shrink to the testable rows."""
    p = np.asarray(p, float)
    m = len(p) if m is None else int(m)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok])]
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(len(o))) * p[o]))
    out[o] = adj
    return out


def _strata_codes(*cols: pd.Series) -> np.ndarray:
    key = pd.concat([c.astype(object).fillna("missing").astype(str) for c in cols], axis=1).agg("|".join, axis=1)
    return pd.Categorical(key).codes


def informative_strata(y: np.ndarray, strata: np.ndarray | None) -> tuple[int, int]:
    """(strata holding both groups, birds in them); without strata the whole sample is one stratum."""
    strata = np.zeros(len(y), int) if strata is None else np.asarray(strata)
    n_s = n_b = 0
    for s in np.unique(strata):
        ys = y[strata == s]
        if 0 < ys.sum() < len(ys):
            n_s, n_b = n_s + 1, n_b + len(ys)
    return n_s, n_b


def perm_strat_diff(x: np.ndarray, y: np.ndarray, strata: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    """Within-stratum difference in means (migrant minus resident), averaged over strata holding both groups with
    weights n1 n0 / (n1 + n0), and its two-sided p with labels permuted between birds within strata. With one
    stratum this is the plain difference in means. No stratum holding both groups: (nan, nan), not the
    permutation floor."""
    x, y, strata = np.asarray(x, float), np.asarray(y, int), np.asarray(strata)
    if informative_strata(y, strata)[0] == 0:
        return np.nan, np.nan

    def stat(L):  # L: (P, n) labels
        num = np.zeros(len(L))
        den = 0.0
        for s in np.unique(strata):
            ix = np.flatnonzero(strata == s)
            n1 = int(y[ix].sum())
            n0 = len(ix) - n1
            if not n1 or not n0:
                continue
            Ls = L[:, ix].astype(float)
            w = n1 * n0 / (n1 + n0)
            num += w * ((Ls @ x[ix]) / n1 - ((1 - Ls) @ x[ix]) / n0)
            den += w
        return num / den if den else np.full(len(L), np.nan)

    obs = float(stat(y[None, :])[0])
    if not np.isfinite(obs):
        return np.nan, np.nan
    L = np.tile(y, (n_perm, 1))
    for s in np.unique(strata):
        ix = np.flatnonzero(strata == s)
        for i in range(n_perm):
            L[i, ix] = y[ix][rng.permutation(len(ix))]
    d = stat(L)
    return obs, float((1 + (np.abs(d) >= abs(obs) - 1e-12).sum()) / (1 + n_perm))


def compare_table(V: pd.DataFrame, lab: pd.Series, sex: pd.Series, family: str, grouping: str, n_perm: int, rng,
                  tested: bool = True, cohort: pd.Series | None = None) -> list[dict]:
    """Migrant vs resident per variable. With cohort given, only birds of cohorts holding both groups count (n_*,
    means, g, the >= MIN_GROUP_BIRDS gate and the tests); the tested diff_mean is then the weighted within-cohort
    difference (perm_strat_diff) with p_perm permuting within cohort, and p_perm_sex_strata within cohort x sex
    (same statistic over cohort x sex strata). hedges_g is over the pooled birds of those cohorts. n_*_all_cohorts:
    birds with a value before the restriction."""
    rows = []
    for c in V.columns:
        v = V[c].astype(float)
        ok = v.notna() & lab.notna()
        n_all = (int((lab[ok] == 1).sum()), int((lab[ok] == 0).sum()))
        if cohort is not None:  # only cohorts holding both groups enter the effect and the test
            both = lab[ok].groupby(cohort[ok]).transform(lambda q: 0 < q.sum() < len(q))
            ok = ok & both.reindex(ok.index, fill_value=False).astype(bool)
        x, y = v[ok].to_numpy(), lab[ok].astype(int).to_numpy()
        a, b = x[y == 1], x[y == 0]
        st1 = _strata_codes(cohort[ok]) if cohort is not None else None
        st2 = _strata_codes(*([cohort[ok]] if cohort is not None else []), sex[ok])
        row = {"grouping": grouping, "family": family, "variable": c, "n_migrant": len(a), "n_resident": len(b),
               "n_migrant_all_cohorts": n_all[0], "n_resident_all_cohorts": n_all[1],
               "mean_migrant": a.mean() if len(a) else np.nan, "mean_resident": b.mean() if len(b) else np.nan,
               "median_migrant": np.median(a) if len(a) else np.nan,
               "median_resident": np.median(b) if len(b) else np.nan,
               "diff_mean": a.mean() - b.mean() if len(a) and len(b) else np.nan,
               "hedges_g": dm.hedges_g(a, b) if len(a) and len(b) else np.nan}
        row["informative_strata"], row["birds_in_informative_strata"] = informative_strata(y, st1)
        row["informative_sex_strata"], row["birds_in_informative_sex_strata"] = informative_strata(y, st2)
        if not tested:
            row["test"] = "descriptive (not tested)"
        elif min(len(a), len(b)) < MIN_GROUP_BIRDS:
            row["test"] = f"not tested: a group has < {MIN_GROUP_BIRDS} birds"
        else:
            if st1 is None:
                d, p = dm.perm_mean_diff(x, y, n_perm, rng)
            else:  # cohort-stratified statistic: cohort levels do not enter the difference
                d, p = perm_strat_diff(x, y, st1, n_perm, rng)
            # sex (x cohort) strata: within-stratum statistic; NaN when no stratum holds both groups
            _, ps = perm_strat_diff(x, y, st2, n_perm, rng)
            strat = "within cohort" if cohort is not None else "unstratified"
            row.update({"test": f"bird-label permutation ({n_perm}, {strat})", "diff_mean": d, "p_perm": p,
                        "p_perm_sex_strata": ps})
        rows.append(row)
    return rows


def nuisance(B: pd.DataFrame) -> pd.DataFrame:
    dmn = B["deploy_month"].astype(float)
    X = pd.DataFrame({"track_days": B["track_days_windows"].astype(float),
                      "deploy_month_sin": np.sin(2 * np.pi * dmn / 12), "deploy_month_cos": np.cos(2 * np.pi * dmn / 12),
                      "cont_share": B["cont_share"].astype(float)}, index=B.index)
    levels = sorted(B["cohort"].astype(str).unique())
    for c in levels[1:]:  # deployment year, one-hot with the first level dropped
        X[f"cohort_{c}"] = (B["cohort"].astype(str) == c).astype(float)
    return X


def h_frame(P: dict, birds, key: str, sel: str) -> pd.DataFrame:
    H = np.stack([P[b]["sel"][sel][f"h_{key}"] for b in birds])
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
    classes = [c for c in RULE_CLASSES if (R.strategy == c).any()]
    if not classes:
        return
    fig, axes = plt.subplots(1, len(classes), figsize=(3.0 * len(classes), 3.2), squeeze=False, sharey=True)
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


def _pfmt(df: pd.DataFrame, n_perm) -> pd.DataFrame:
    """p columns as text; a p at the permutation floor prints as '<= 1/(n+1)' (n: the row's or the given n_perm)."""
    df = df.copy()
    for c in [c for c in df if c.startswith("p_")]:
        n = df["n_perm"] if "n_perm" in df else pd.Series(n_perm, index=df.index)
        df[c] = [("n/a" if not np.isfinite(p) else f"<= 1/{int(k) + 1}" if p <= 1 / (k + 1) + 1e-12 else _f(float(p)))
                 for p, k in zip(df[c].astype(float), n.astype(float))]
    return df


def write_summary(out: Path, R: pd.DataFrame, agree: pd.DataFrame, kappa: dict, sens: pd.DataFrame,
                  groups: pd.DataFrame, T: pd.DataFrame, clf: pd.DataFrame, nest: pd.DataFrame, meta: dict) -> None:
    n_perm = meta["n_perm"]
    L = ["# Residency from GPS tracks, and migrants vs residents", "",
         f"No coordinates are stored: per-bird distances in residency.csv are rounded to {KM_ROUND} km and only "
         "distances (no northing) are written; day-of-year and classes. Thresholds and class definitions: module "
         "docstring of aquilacl.residency and metadata.json. Cohort = year of the reference deploy-on date.", "",
         "## 1. Strategy per bird (descriptive)", "",
         f"Rule classes (primary) by cohort, age group (young = tagged as young; other = age not stated) and the "
         f"reference 'migrant' flag. resident_full = never beyond {MIN_DISP_KM:.0f} km (30-day running median) over "
         f">= {FULL_YEAR_DAYS} days; present_partial_year = the same on a shorter track: {PARTIAL_LABEL}; it is not "
         "counted as resident in any test. Tracks are cut at deploy-off (birds with a closed deployment: "
         f"{meta['birds_with_deploy_off']}) and before a stationary end of >= {STILL_DAYS} days within "
         f"{STILL_KM:.0f} km and below {STILL_MS:.0f} m/s (cut: {meta['birds_stationary_end_cut']} birds).", ""]
    R2 = R.assign(group=R.cohort.astype(str) + "|" + np.where(R.young, "young", "other") + "|" +
                  np.where(R.migrant_flag, "flagged", "not flagged"))
    ct = pd.crosstab(R2.strategy, R2.group).reindex([c for c in RULE_CLASSES if c in set(R2.strategy)])
    ct.loc["total"] = ct.sum()
    L += _md(ct.reset_index())
    L += ["", "NSD-model classes (Bunnefeld et al. 2011, AIC with acceptance rules; 'resident' needs a full year "
          "like the rule class) by age group:", ""]
    L += _md(pd.crosstab(R2.strategy_nsd, np.where(R.young, "young", "other")).reset_index())
    L += ["", "### Agreement between the two methods on coarse classes (rows: rule, columns: NSD; descriptive)", "",
          "Coarse classes: migrant = migrant + migrant_departed; resident = resident_full (NSD: resident over a full "
          "year); disperser = disperser + juvenile_dispersal (NSD: disperser); nomad; other = insufficient, "
          "undetermined, away_at_end, present_partial_year (not compared).", ""]
    L += _md(pd.crosstab(R.strategy_coarse, R.strategy_nsd_coarse).reset_index())
    L += ["", f"Birds with a compared class in both methods: {kappa['n']}; agreement {_f(kappa['agree'])}; "
          f"Cohen's kappa {_f(kappa['kappa'])}.", "",
          "### Agreement with the reference 'migrant' flag (True or not stated; descriptive)", ""]
    L += _md(agree)
    L += ["", "### Threshold sensitivity of the rule classes (descriptive; threshold_sensitivity.csv)", "",
          f"Far-north threshold at {', '.join(f'{x:.0f}' for x in SENS_NORTH_KM)} km from the deployment centroid "
          f"('deploy', the primary rule) and from the study-area centre ('centre'); the return threshold "
          f"({RETURN_NORTH_KM:.0f} km north) and the displacement threshold ({MIN_DISP_KM:.0f} km) are fixed. "
          "n_max_north_beyond = birds whose maximum daily northward displacement reaches the threshold (no return "
          "needed), against rule_migrants = 'migrant' + 'migrant_departed' (rule classes).", ""]
    if len(sens):
        sh = sens.assign(rule_migrants=sens.migrant + sens.migrant_departed)
        keep = ["origin", "north_km", "age", "flag", "birds", "n_max_north_beyond", "rule_migrants", "migrant",
                "migrant_departed", "resident_full", "present_partial_year", "nomad", "disperser",
                "juvenile_dispersal", "away_at_end", "insufficient"]
        L += _md(sh[keep])
    mig = R[R.strategy == "migrant"]
    L += ["", "### Migration timing and ranges (descriptive)", "",
          f"- Rule migrants with a return: {len(mig)}; migrant_departed (far north at track end): "
          f"{int((R.strategy == 'migrant_departed').sum())}; tagged young among both: "
          f"{int(R.young[R.strategy_coarse == 'migrant'].sum())}.",
          f"- Dates are NaN when the crossing falls in a GPS gap of more than {MAX_GAP_DAYS} days (first migration: "
          f"departure dated for {int(mig.dep_doy_rule_y1.notna().sum())} of {len(mig)}, return for "
          f"{int(mig.ret_doy_rule_y1.notna().sum())}).",
          f"- Departure day-of-year (rule, first migration; first day beyond {RETURN_NORTH_KM:.0f} km north): median "
          f"{_f(mig.dep_doy_rule_y1.median())}, range {_f(mig.dep_doy_rule_y1.min())}-{_f(mig.dep_doy_rule_y1.max())}; "
          f"return: median {_f(mig.ret_doy_rule_y1.median())}, range {_f(mig.ret_doy_rule_y1.min())}-"
          f"{_f(mig.ret_doy_rule_y1.max())}.",
          f"- Sigmoid midpoints (NSD migrant fits with dated midpoints, {int(R.dep_doy_nsd.notna().sum())} birds): "
          f"departure median {_f(R.dep_doy_nsd.median())}, return median {_f(R.ret_doy_nsd.median())} (midpoints, "
          "not onsets).", "",
          f"Median per-bird values per rule class (km rounded to {KM_ROUND}; classes with < 3 birds not shown, so no "
          "single bird's values appear). The winter columns use the same days, each bird's Nov-Mar daily positions: "
          "winter_dist_km = class median of each bird's median Nov-Mar distance from the study-area centre; "
          "winter_dist_deploy_km = the same from the bird's own deployment centroid; winter_basin_share = class "
          f"median of each bird's share of Nov-Mar days within {BASIN_KM:.0f} km of the study-area centre; "
          "winter_days = class median of Nov-Mar days; n_winter = birds with any Nov-Mar day. A low "
          "winter_basin_share with a winter_dist_deploy_km well below winter_dist_km means the class winters near "
          "where its birds were deployed but outside the basin radius around the study-area centre.", ""]
    cols_ = ["max_dist_km", "summer_dist_km", "winter_dist_km", "winter_dist_deploy_km"]
    med = (R.groupby("strategy")[cols_].median() / KM_ROUND).round() * KM_ROUND
    med["winter_days"] = R.groupby("strategy").winter_days.median()
    med["winter_basin_share"] = R.groupby("strategy").winter_basin_share.median().round(2)
    med["n"] = R.groupby("strategy").size()
    med["n_winter"] = R[R.winter_days > 0].groupby("strategy").size().reindex(med.index).fillna(0).astype(int)
    med = med[med.n >= 3]
    L += _md(med.reset_index())

    # ---- section 2
    L += ["", "## 2. Behaviour: coarse migrants vs residents", "",
          f"Shared rule: a grouping whose nuisance floor alone separates the groups (bird-level grouped-CV AUC of "
          f"track days, deploy month, continuous-mode share and cohort >= {NUISANCE_AUC_MAX}) is NOT TESTABLE IN THIS "
          "DESIGN; its rows keep effect sizes and bird counts but no p-values. Residents: 'rule' = resident_full only "
          "(primary); 'rule_partial' adds present_partial_year (sensitivity); 'consensus' = rule and NSD agree. "
          "Variants (module docstring, Part 3): c2024_novmar_basin, all_novmar_basin, all_allmonths_basin "
          "(cohort = deployment year as permutation stratum and nuisance), all_novmar_winter_range (EXTRA: different "
          "places). 'adult' = age not stated, 'all' = including tagged young.", "",
          "### Nuisance floor per grouping (birds with selected windows)", ""]
    gshow = groups[["grouping", "migrants", "migrants_with_windows", "residents", "residents_with_windows",
                    "cohorts_with_both_groups", "nuisance_auc"]].copy()
    gshow["status"] = ["n/a: < %d birds in a group" % MIN_GROUP_BIRDS if not np.isfinite(a) else
                       ("NOT TESTABLE in this design" if a >= NUISANCE_AUC_MAX else "testable")
                       for a in gshow.nuisance_auc.astype(float)]
    L += _md(gshow)
    testable = set(gshow.grouping[gshow.status == "testable"])
    L += ["", f"Testable groupings: {len(testable)} of {len(gshow)}.", "", "### Birds per group", ""]
    L += _md(groups.drop(columns=["nuisance_auc"]))
    show = ["grouping", "variable", "n_migrant", "n_resident", "median_migrant", "median_resident", "diff_mean",
            "hedges_g", "p_perm", "p_perm_sex_strata", "p_holm", "informative_strata", "informative_sex_strata"]
    for fam, title in (("a_shares_scheduled", "(a) K=2 flight share among scheduled windows (first window of each "
                        "burst, sun above the horizon)"),
                       ("b_kinematics", "(b) flight kinematics (per-bird medians; windows with GPS, mostly "
                                        "continuous mode)")):
        S = T[T.family == fam] if len(T) else T
        if not len(S):
            continue
        big = S[(S.n_migrant >= MIN_GROUP_BIRDS) & (S.n_resident >= MIN_GROUP_BIRDS)]
        tst, des = big[big.grouping.isin(testable)], big[~big.grouping.isin(testable)]
        L += ["", f"### {title}: TESTED (testable groupings only)", "",
              f"Holm over the declared family of each grouping (the K=2 share and the {len(set(T[T.family == 'b_kinematics'].variable))} "
              "kinematics variables, testable or not). All rows, with bird counts, in behaviour_tests.csv.", ""]
        L += _md(_pfmt(tst[[c for c in show if c in tst]], n_perm)) if len(tst) else ["(none)"]
        L += ["", f"### {title}: DESCRIPTIVE (not testable in this design; effect sizes only)", ""]
        L += _md(des[[c for c in show if c in des and not c.startswith("p_")]]) if len(des) else ["(none)"]
    L += ["", "K=6 scheduled shares (a_shares_scheduled_k6) and shares over all windows (a_shares_all_modes) are "
          "descriptive only (behaviour_tests.csv).", "",
          f"### (c) Classifier on mean h in K=2 flight / ground (rule groupings, >= {MIN_CLF_BIRDS} birds per class; "
          "bird-level AUC, grouped CV, bird-label permutation)", ""]
    hc = clf[~clf.run.str.endswith(":floor")] if len(clf) else clf
    if len(hc):
        L += _md(_pfmt(hc[["run", "target", "feature_set", "n_pos_birds", "n_neg_birds", "auc_mean", "auc_sd",
                           "null_auc_q95", "n_perm", "p_auc", "status"]], n_perm))
        L += ["", "h beyond the nuisance floor (Freedman-Lane nested null; descriptive where not testable):", ""]
        L += _md(_pfmt(nest[["run", "target", "delta_auc_mean", "delta_auc_sd", "n_perm", "p_delta_auc", "status"]],
                       n_perm)) if len(nest) else ["(none)"]
    else:
        L += [f"not run: no rule grouping has >= {MIN_CLF_BIRDS} birds per class with mean h"]
    L += ["", "## Caveats", "",
          "- Shares are window shares of scheduled bursts, not time budgets; one window per burst and daylight only, "
          "but not matched for hour or month between groups (month mix in behaviour_groups.csv). Burst lengths differ "
          "between tags (many scheduled bursts give short windows), so the share mixes window lengths. Kinematics "
          "come from continuous mode, which starts on fast movement.",
          "- Migrants spend few of their days within the basin radius; basin variants compare the migrants' rare "
          "basin days (often passage) with the residents' usual days. The winter-range variant compares birds in "
          "different places.",
          "- Migrant-flagged birds were mostly deployed in October; class, deploy month and cohort are inseparable, "
          "which is what the nuisance floor measures. The classifier null permutes labels between all birds.",
          f"- p-values at the permutation floor print as '<= 1/(n+1)'. p_perm_sex_strata is NaN when no "
          "(cohort x) sex stratum holds both groups (informative_sex_strata = 0).",
          "- Classes depend on thresholds (threshold sensitivity table above).",
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
    ref["deploy_off"] = deploy_off_dates(ref_csv, ref.index)
    log(f"{len(files)} GPS files, {len(wdirs)} birds with windows")

    # part 1: tracks (positions in memory only)
    tracks, R = classify_all(files, ref, min_days, workers)
    log(f"part 1: {len(R)} birds classified ({time.time() - t0:.0f} s)")
    kappa, agree = agreement(R)
    sens = threshold_table(tracks, R, min_days)
    R["cohort"], R["cohort_source"] = cohorts(R, ref)
    km = [c for c in R if c.endswith("_km")]
    R[km] = (R[km] / KM_ROUND).round(0) * KM_ROUND
    doy = [c for c in R if "_doy_" in c]
    R[doy] = R[doy].round(0)
    R[["basin_days_share", "winter_basin_share"]] = R[["basin_days_share", "winter_basin_share"]].round(3)
    cols = ["young", "migrant_flag", "sex", "cohort", "cohort_source", "strategy", "strategy_coarse", "strategy_nsd",
            "strategy_nsd_coarse", "consensus", "rule_note", "nsd_note", "deploy_off_known", "end_still_days",
            "track_days", "n_days", "covers_spring", "max_dist_km", "max_roll_d0_km", "summer_dist_km",
            "winter_dist_km", "winter_days", "basin_days_share", "winter_basin_share", "n_migrations",
            "dep_doy_rule_y1", "ret_doy_rule_y1", "dep_doy_rule_y2", "ret_doy_rule_y2", "nsd_best_aic",
            "nsd_delta_km", "dep_doy_nsd", "ret_doy_nsd"] + [f"daic_{m}" for m in MODELS]
    out.mkdir(parents=True, exist_ok=True)  # part 1 outputs first, so they survive a failure in part 3
    R[cols].reset_index().to_csv(out / "residency.csv", index=False)
    agree.to_csv(out / "agreement.csv", index=False)
    sens.to_csv(out / "threshold_sensitivity.csv", index=False)
    fig_nsd_birds(R, tracks, out / "nsd_birds.png")
    fig_nsd_by_strategy(R, tracks, out / "nsd_by_strategy.png")

    # part 3: behaviour
    sel_days = {b: {name: (area_days(tr.epoch_day.to_numpy(), tr[col].to_numpy()), months)
                    for name, (col, months) in SELECTIONS.items()} for b, (tr, _) in tracks.items()}
    del tracks
    st = {}
    for name, sdir, K in (("k2", states2, 2), ("k6", states6, 6)):
        S = pd.read_parquet(sdir / "states.parquet", columns=["bird", "row", state_col])
        S["bird"] = S["bird"].astype(str)
        if int(S[state_col].max()) + 1 != K:
            raise ValueError(f"{sdir}: expected K={K}, found {int(S[state_col].max()) + 1}")
        st[name] = dict(tuple(S.groupby("bird", sort=False)))
        del S
    wb = [b for b in wdirs if b in sel_days]
    res = Parallel(n_jobs=workers)(
        delayed(behaviour_bird)(wdirs[b], run_dir, st["k2"].pop(b, None), st["k6"].pop(b, None), sel_days[b],
                                state_col) for b in wb)
    del st
    P = {p["bird"]: p for p in res}
    log(f"part 3 pass: {len(P)} birds ({time.time() - t0:.0f} s)")
    B = R.loc[[b for b in R.index if b in P]].copy()
    B["cont_share"] = [P[b]["cont_share"] for b in B.index]
    B["track_days_windows"] = [(P[b]["last"] - P[b]["first"]).total_seconds() / 86400 for b in B.index]
    dep = pd.to_datetime(ref.loc[B.index, "deploy_on"], utc=True).dt.month
    B["deploy_month"] = dep.fillna(pd.Series({b: P[b]["first"].month for b in B.index})).to_numpy()
    BB = {}
    for sel in SELECTIONS:
        scal = [k for k in next(iter(P.values()))["sel"][sel] if not k.startswith("h_")]
        BB[sel] = pd.DataFrame([{k: P[b]["sel"][sel][k] for k in scal} | {"bird": b} for b in B.index]).set_index("bird")
    groupings = make_groupings(B)

    # (c) first: nuisance floor of every grouping (birds with selected windows) decides testability; h classifiers
    # for the rule groupings
    specs = {}
    rn = run_dir.name
    for gk, (sel, lab) in groupings.items():
        y = lab[BB[sel].loc[lab.index, "n_sel"] > 0]
        if min(y.sum(), (1 - y).sum()) >= MIN_GROUP_BIRDS:
            sp = dm.make_spec(nuisance(B.loc[y.index]), y, "bird", f"{rn}:floor", f"migrant_vs_resident|{gk}",
                              "nuisance")
            if sp is not None:
                specs[f"{gk}|floor|nuisance"] = sp
        if "|rule|" not in gk:
            continue
        for tag, keys in (("flight", ["flight"]), ("ground", ["ground"]), ("flight_ground", ["flight", "ground"])):
            H = pd.concat([h_frame(P, lab.index, k, sel) for k in keys], axis=1)
            yh = lab[H.notna().all(axis=1)]
            if min(yh.sum(), (1 - yh).sum()) < MIN_CLF_BIRDS:
                continue
            Nu, Hy = nuisance(B.loc[yh.index]), H.loc[yh.index]
            for fs, F, nh in (("nuisance", Nu, 0), ("mean_h", Hy, 0),
                              ("h_plus_nuisance", pd.concat([Hy, Nu], axis=1), Hy.shape[1])):
                sp = dm.make_spec(F, yh, "bird", f"{rn}:{tag}", f"migrant_vs_resident|{gk}", fs, n_h=nh)
                if sp is not None:
                    specs[f"{gk}|{tag}|{fs}"] = sp
    if specs:
        clf, nest = dm.run_classifiers(specs, n_repeats, {"bird": n_perm}, seed, workers)
        clf, nest = testability(clf, nest)
    else:
        clf, nest = pd.DataFrame(), pd.DataFrame()
    floor = {} if not len(clf) else {
        r.target.split("|", 1)[1]: r.auc_mean for r in clf[clf.run.str.endswith(":floor")].itertuples()}
    log(f"part 3 classifiers done ({time.time() - t0:.0f} s)")

    # (a), (b) comparisons; a grouping whose nuisance floor reaches NUISANCE_AUC_MAX is not testable
    grp_rows, T = [], []
    for gk, (sel, lab) in groupings.items():
        X = BB[sel].loc[lab.index]
        sex, coh = B.loc[lab.index, "sex"], B.loc[lab.index, "cohort"]
        multi = coh.nunique() > 1
        grp_rows.append(group_row(gk, sel, lab, X, sex, coh) | {"nuisance_auc": floor.get(gk, np.nan)})
        k2s = ["share_k2_flight_sched"]
        k6s = [c for c in X if c.startswith("share_k6_") and c.endswith("_sched")]
        share_a = [c for c in X if c.startswith("share_") and c.endswith("_all")]
        kin = [c for c in X if c.split("_")[0] in ("speed", "radius", "climb")]
        VS = X[k2s + k6s].where(X["n_k2_sched"] >= MIN_SHARE_WINDOWS, axis=0)
        VA = X[share_a].where(X["n_k2_all"] >= MIN_SHARE_WINDOWS, axis=0)
        rows_g = []
        for fam, V, tested in (("a_shares_scheduled", VS[k2s], True), ("a_shares_scheduled_k6", VS[k6s], False),
                               ("a_shares_all_modes", VA, False), ("b_kinematics", X[kin], True)):
            rows_g += compare_table(V, lab, sex, fam, gk, n_perm, rng, tested, coh if multi else None)
        declared = [r for r in rows_g if r["family"] in TESTED_FAMILIES]
        ph = holm(np.array([r.get("p_perm", np.nan) for r in declared]), len(declared))
        for r, p in zip(declared, ph):
            r["p_holm"] = p
        fa = floor.get(gk, np.nan)
        for r in rows_g:
            r["nuisance_auc"] = fa
            if r["family"] in TESTED_FAMILIES and np.isfinite(fa) and fa >= NUISANCE_AUC_MAX:
                r["test"] = f"not testable in this design (nuisance floor AUC {fa:.2f})"
                r.update({"p_perm": np.nan, "p_perm_sex_strata": np.nan, "p_holm": np.nan})
        T += rows_g
    T = pd.DataFrame(T)
    groups = pd.DataFrame(grp_rows)
    groups.to_csv(out / "behaviour_groups.csv", index=False)
    info = B[["strategy", "strategy_coarse", "consensus", "young", "sex", "cohort", "deploy_month",
              "track_days_windows", "cont_share"]]
    pd.concat([BB[sel].drop(columns=["cont_share"], errors="ignore").join(info).assign(selection=sel)
               for sel in SELECTIONS]).reset_index().to_csv(out / "behaviour_birds.csv", index=False)
    T.to_csv(out / "behaviour_tests.csv", index=False)
    clf.to_csv(out / "classifiers.csv", index=False)
    nest.to_csv(out / "nested.csv", index=False)
    log(f"part 3 tests done ({time.time() - t0:.0f} s)")
    meta = {"gps": str(gps), "windows": str(windows), "reference_csv": str(ref_csv), "states2": str(states2),
            "states6": str(states6), "run": str(run_dir), "state_col": state_col, "min_days": min_days,
            "n_perm": n_perm, "n_repeats": n_repeats, "seed": seed, "workers": workers,
            "birds_gps": len(R), "birds_windows": len(P),
            "birds_without_deploy_on": int(ref.loc[R.index, "deploy_on"].isna().sum()),
            "birds_with_deploy_off": int(R.deploy_off_known.sum()),
            "birds_stationary_end_cut": int((R.end_still_days >= STILL_DAYS).sum()),
            "thresholds": {k: globals()[k] for k in (
                "FIRST_DAYS", "FIT_DAYS", "MIN_DISP_KM", "MIGRANT_NORTH_KM", "RETURN_NORTH_KM", "RETURN_WITHIN_DAYS",
                "FULL_YEAR_DAYS", "MIN_STAY_DAYS", "SETTLE_DAYS", "FINAL_DAYS", "ROLL_DAYS", "ROLL_MIN", "PHI_MIN",
                "PHI_MAX", "BASIN_KM", "BASIN_FILL_DAYS", "MIN_SHARE_WINDOWS", "MIN_STATE_WINDOWS",
                "MIN_GROUP_BIRDS", "MIN_CLF_BIRDS", "MAX_GAP_DAYS", "STILL_DAYS", "STILL_KM", "STILL_MS", "KM_ROUND",
                "NUISANCE_AUC_MAX")},
            "sensitivity_north_km": SENS_NORTH_KM,
            "months": {"spring": SPRING_MONTHS, "summer": SUMMER_MONTHS, "winter": COMPARE_MONTHS,
                       "compare": COMPARE_MONTHS},
            "selections": {k: [v[0], list(v[1])] for k, v in SELECTIONS.items()},
            "variants": {k: list(v) for k, v in VARIANTS.items()}, "extra_variants": list(EXTRA_VARIANTS),
            "methods": METHODS, "tested_families": TESTED_FAMILIES,
            "coarse": COARSE, "states6_names": STATES6, "flight6": FLIGHT6, "kappa": kappa, "runtime_s": None}
    meta["runtime_s"] = round(time.time() - t0, 1)
    write_summary(out, R.assign(young=R.young.astype(bool)), agree, kappa, sens, groups, T, clf, nest, meta)
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    log(f"done ({meta['runtime_s']} s)")
    return meta


TESTED_FAMILIES = ("a_shares_scheduled", "b_kinematics")
METHODS = {"rule": ("strategy_coarse", ("resident_full",)),  # primary: residents observed over a full year
           "rule_partial": ("strategy_coarse", ("resident_full", "present_partial_year")),  # sensitivity
           "consensus": ("consensus", ("resident_full",))}


def cohorts(R: pd.DataFrame, ref: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Cohort = year of the reference deploy-on (earliest deployment); first GPS-fix year when deploy-on is missing."""
    dy = pd.to_datetime(ref["deploy_on"].reindex(R.index), utc=True)
    coh = [str(int(dy[b].year)) if pd.notna(dy[b]) else str(int(R.at[b, "gps_first_year"])) for b in R.index]
    return coh, ["deploy_on" if pd.notna(dy[b]) else "first_gps_fix" for b in R.index]


def testability(clf: pd.DataFrame, nest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label classifier and nested rows by the nuisance-only AUC of the same run and target."""
    nu = clf[clf.feature_set == "nuisance"].set_index(["run", "target"]).auc_mean

    def lab(df):
        a = [nu.get((r, t), np.nan) for r, t in zip(df.run, df.target)]
        s = ["not testable in this design (nuisance floor AUC >= %.1f)" % NUISANCE_AUC_MAX
             if np.isfinite(x) and x >= NUISANCE_AUC_MAX else ("tested" if np.isfinite(x) else "no nuisance floor")
             for x in a]
        return df.assign(nuisance_auc=a, status=s)
    return lab(clf), (lab(nest) if len(nest) else nest)


def classify_all(files: list[Path], ref: pd.DataFrame, min_days: int, workers: int) -> tuple[dict, pd.DataFrame]:
    """Part 1 for all GPS files: tracks {bird: (track, nsd fit)} and the per-bird class table (no positions)."""
    off = ref["deploy_off"] if "deploy_off" in ref else pd.Series(pd.NaT, index=ref.index)
    G = Parallel(n_jobs=workers)(delayed(read_bird)(f, ref.at[f.stem, "deploy_on"], off.get(f.stem)) for f in files)
    G = [g for g in G if g["daily"] is not None and len(g["daily"])]
    centre = tuple(np.median(np.array([g["median"] for g in G]), axis=0))
    young = ref["tagged_young"].notna()
    tracks, rows = {}, []
    for g in G:
        tr = track(g["daily"], centre)
        c = classify_bird(tr, min_days, bool(young.get(g["bird"], False)))
        if g["end_still_days"] >= STILL_DAYS:
            c["rule_note"] = "; ".join(x for x in (c["rule_note"], f"stationary end of {g['end_still_days']} days "
                                                   "cut (possible mortality or tag loss)") if x)
        tracks[g["bird"]] = (tr, c.pop("_fit"))
        rows.append({"bird": g["bird"], "gps_first_year": int(tr.year.iat[0]), "end_still_days": g["end_still_days"],
                     **c})
    del G
    R = pd.DataFrame(rows).set_index("bird")
    R["deploy_off_known"] = off.reindex(R.index).notna().to_numpy()
    R["young"] = young.reindex(R.index).fillna(False).astype(bool).to_numpy()
    R["migrant_flag"] = ref.loc[R.index, "migrant"].notna().to_numpy()
    R["sex"] = ref.loc[R.index, "sex"].to_numpy()
    R["strategy"] = R["rule_class"]
    R["strategy_nsd"] = R["nsd_class"]
    R["strategy_coarse"] = R.strategy.map(coarse)
    R["strategy_nsd_coarse"] = R.strategy_nsd.map(coarse)
    R["consensus"] = np.where(R.strategy_coarse == R.strategy_nsd_coarse, R.strategy_coarse, "disagree")
    return tracks, R


def threshold_table(tracks: dict, R: pd.DataFrame, min_days: int) -> pd.DataFrame:
    """Rule-class counts at each far-north threshold (SENS_NORTH_KM) from both origins (deployment centroid,
    study-area centre), by age group and 'migrant' flag; RETURN_NORTH_KM and MIN_DISP_KM stay fixed.
    n_max_north_beyond: birds whose maximum daily northward displacement reaches the threshold (no return needed)."""
    rows = []
    for origin, (_, ncol) in ORIGINS.items():
        for nk in SENS_NORTH_KM:
            cls = pd.Series({b: rule_classify(tr, min_days, bool(R.at[b, "young"]), nk, origin)["rule_class"]
                             for b, (tr, _) in tracks.items()})
            beyond = pd.Series({b: bool((tr[ncol] >= nk).any()) for b, (tr, _) in tracks.items()})
            for age, flag in (("other", True), ("other", False), ("young", False), ("young", True)):
                ix = R.index[(R.young == (age == "young")) & (R.migrant_flag == flag)]
                if not len(ix):
                    continue
                vc = cls.reindex(ix).value_counts()
                rows.append({"origin": origin, "north_km": nk, "return_km": RETURN_NORTH_KM, "age": age,
                             "flag": "flagged migrant" if flag else "not flagged", "birds": len(ix),
                             "n_max_north_beyond": int(beyond.reindex(ix).sum()),
                             **{c: int(vc.get(c, 0)) for c in RULE_CLASSES}})
    return pd.DataFrame(rows)


def agreement(R: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Kappa between the coarse classes of both methods, and class counts per age group and 'migrant' flag."""
    both = (R.strategy_coarse != "other") & (R.strategy_nsd_coarse != "other")
    kappa = {"n": int(both.sum()),
             "agree": float((R.strategy_coarse[both] == R.strategy_nsd_coarse[both]).mean()) if both.any() else np.nan,
             "kappa": float(cohen_kappa_score(R.strategy_coarse[both], R.strategy_nsd_coarse[both]))
             if both.sum() > 1 and R.strategy_coarse[both].nunique() + R.strategy_nsd_coarse[both].nunique() > 2
             else np.nan}
    ag = []
    for meth, col in (("rule", "strategy"), ("rule_coarse", "strategy_coarse"), ("nsd", "strategy_nsd"),
                      ("nsd_coarse", "strategy_nsd_coarse"), ("consensus", "consensus")):
        for age, sub in (("other", R[~R.young]), ("young", R[R.young])):
            for flag, s in (("flagged migrant", sub[sub.migrant_flag]), ("not flagged", sub[~sub.migrant_flag])):
                ag.append({"method": meth, "age": age, "flag": flag, "birds": len(s), **s[col].value_counts().to_dict()})
    agree = pd.DataFrame(ag).fillna(0)
    cls = [c for c in agree if c not in ("method", "age", "flag", "birds")]
    agree[cls] = agree[cls].astype(int)
    agree = agree[agree.birds > 0]
    return kappa, agree


def make_groupings(B: pd.DataFrame) -> dict:
    """'<variant>|<method>|<age>' -> (selection, 0/1 labels: 1 = coarse migrant, 0 = resident as the method defines
    it: 'rule' resident_full only (primary), 'rule_partial' resident_full + present_partial_year (sensitivity),
    'consensus' resident_full that the NSD models also call resident)."""
    out = {}
    for variant, (sel, cohort) in VARIANTS.items():
        for method, (col, res_classes) in METHODS.items():
            for age in ("adult", "all"):
                mig = B[col] == "migrant"
                res = B.strategy.isin(res_classes) & (B[col] != "disagree")
                sub = B[mig | res]
                if cohort is not None:
                    sub = sub[sub.cohort == cohort]
                if age == "adult":
                    sub = sub[~sub.young]
                out[f"{variant}|{method}|{age}"] = (sel, (sub[col] == "migrant").astype(int))
    return out


def group_row(gk: str, sel: str, lab: pd.Series, X: pd.DataFrame, sex: pd.Series, coh: pd.Series) -> dict:
    """Bird counts of one grouping: per group, with selected windows, with enough scheduled windows, by sex."""
    row = {"grouping": gk, "selection": sel}
    for g, v in (("migrants", 1), ("residents", 0)):
        ix = lab.index[lab == v]
        row[g] = len(ix)
        row[f"{g}_with_windows"] = int((X.loc[ix, "n_sel"] > 0).sum())
        dd = X.loc[ix, "n_sel_days"][X.loc[ix, "n_sel"] > 0]
        row[f"{g}_median_days"] = float(dd.median()) if len(dd) else np.nan
        M = X.loc[ix][X.loc[ix, "n_sel"] > 0][[f"n_sel_month{mo:02d}" for mo in range(1, 13)]].to_numpy(float)
        mix = (M / M.sum(1, keepdims=True)).mean(0) if len(M) else np.zeros(12)
        row[f"{g}_month_mix"] = " ".join(f"{MONTH_ABBR[i]} {mix[i]:.2f}" for i in range(12) if mix[i] >= 0.005)
        row[f"{g}_with_{MIN_SHARE_WINDOWS}_sched"] = int((X.loc[ix, "n_k2_sched"] >= MIN_SHARE_WINDOWS).sum())
        row[f"{g}_f/m/missing"] = "/".join(str(int((sex[ix].fillna("x") == s).sum())) for s in ("f", "m", "x"))
    have = X["n_sel"] > 0
    row["cohorts_with_both_groups"] = ",".join(sorted(str(c) for c in coh[have].unique()
                                                      if 0 < lab[have][coh[have] == c].sum() < (coh[have] == c).sum()))
    return row


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
