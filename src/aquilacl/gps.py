"""GPS fixes: quality filter, continuous runs and per-fix kinematics.

All rates use the elapsed time between the two fixes involved. The first draft
divided altitude differences by the median interval of all sensor events in an
hour, which inflated descent rates about five-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_R = 6_371_000.0

LON, LAT = "location-long", "location-lat"
ALT, SPEED, HEADING = "height-above-ellipsoid", "ground-speed", "heading"
HACC = "eobs:horizontal-accuracy-estimate"

# Kinematics plausibility limits (see ``kinematics``).
MIN_TURN_STEP_M = 5.0  # below 5 m the bearing is dominated by position noise (turn rates > 90 deg/s at 0.5-2 m steps)
MIN_TURN_DT_S = 1.0  # sub-second fix intervals give turn rates > 180 deg/s from a single bearing change
MAX_TRACK_SPEED_MS = 70.0  # the tag's own ground-speed never exceeds 69.6 m/s; faster position steps are jumps
MAX_SPEED_DIFF_MS = 10.0  # position-derived speed this far from the tag's ground-speed indicates a position error
MAX_CLIMB_MS = 20.0  # sustained climb above 20 m/s is beyond any soaring or flapping climb: altitude error
MIN_CLIMB_MS = -70.0  # dives may be fast, but not faster than the horizontal speed limit


def good_fixes(df: pd.DataFrame, max_hacc_m: float = 25.0, min_sats: int = 4) -> pd.DataFrame:
    """Keep fixes with a position, a valid status and acceptable accuracy."""
    ok = df[LON].between(-180, 180) & df[LAT].between(-90, 90)
    if "eobs:status" in df:
        ok &= df["eobs:status"].fillna("A").isin(["A"])
    if HACC in df:
        ok &= df[HACC].fillna(0) <= max_hacc_m
    if "gps:satellite-count" in df:
        ok &= df["gps:satellite-count"].fillna(min_sats) >= min_sats
    return df[ok]


def runs(t: pd.Series, max_gap_s: float) -> np.ndarray:
    """Integer run id: a new run starts whenever the gap exceeds ``max_gap_s``."""
    dt = t.diff().dt.total_seconds().to_numpy()
    return np.cumsum(~(dt <= max_gap_s))  # NaN at start counts as a break


def kinematics(df: pd.DataFrame, max_gap_s: float = 3.0) -> pd.DataFrame:
    """Per-fix motion quantities within continuous runs (dense GPS).

    Adds: run (id), dt_s, step_m, east_m/north_m (local tangent plane per
    run), speed_track_ms, climb_ms, bearing_deg, turn_rate_dps, and
    cum_turn_deg (signed cumulative turning within the run).
    Rates are NaN for the first fix of each run. Duplicate timestamps keep the
    first fix. A step whose speed exceeds MAX_TRACK_SPEED_MS or differs from the
    tag's ground-speed by more than MAX_SPEED_DIFF_MS is a position outlier
    (``pos_outlier``): its step, speed, climb and bearing are NaN, so turn rates
    using it are NaN too. Climb outside [MIN_CLIMB_MS, MAX_CLIMB_MS] is NaN.
    Turn rate is NaN when either step is under MIN_TURN_STEP_M or dt < MIN_TURN_DT_S.
    """
    d = df.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="first").reset_index(drop=True).copy()
    d["run"] = runs(d["timestamp"], max_gap_s)
    # Seconds since the first fix, independent of the datetime resolution.
    t = ((d["timestamp"] - d["timestamp"].iloc[0]) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    lon, lat = np.radians(d[LON].to_numpy()), np.radians(d[LAT].to_numpy())
    run = d["run"].to_numpy()

    lat0 = pd.Series(lat).groupby(run).transform("mean").to_numpy()
    lon0 = pd.Series(lon).groupby(run).transform("mean").to_numpy()
    east = EARTH_R * (lon - lon0) * np.cos(lat0)
    north = EARTH_R * (lat - lat0)
    d["east_m"], d["north_m"] = east, north

    same = np.r_[False, run[1:] == run[:-1]]
    dt = np.r_[np.nan, np.diff(t)]
    de, dn = np.r_[np.nan, np.diff(east)], np.r_[np.nan, np.diff(north)]
    dz = np.r_[np.nan, np.diff(d[ALT].to_numpy(dtype=float))] if ALT in d else np.full(len(d), np.nan)
    dt[~same] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.where(same, np.hypot(de, dn), np.nan)
        speed = np.where(same & (dt > 0), step / dt, np.nan)
        gs = d[SPEED].to_numpy(dtype=float) if SPEED in d else np.full(len(d), np.nan)
        bad = (speed > MAX_TRACK_SPEED_MS) | (np.abs(speed - gs) > MAX_SPEED_DIFF_MS)  # NaN compares False
        climb = np.where(same & (dt > 0) & ~bad, dz / dt, np.nan)
        climb[(climb > MAX_CLIMB_MS) | (climb < MIN_CLIMB_MS)] = np.nan
        d["dt_s"] = dt
        d["pos_outlier"] = bad
        d["step_m"] = np.where(bad, np.nan, step)
        d["speed_track_ms"] = np.where(bad, np.nan, speed)
        d["climb_ms"] = climb
        bearing = np.degrees(np.arctan2(de, dn)) % 360
        bearing[~same | bad | ~(step >= MIN_TURN_STEP_M)] = np.nan
        d["bearing_deg"] = bearing
        dturn = (np.r_[np.nan, np.diff(bearing)] + 180) % 360 - 180
        prev_same = np.r_[False, same[1:] & same[:-1]]
        dturn[~prev_same] = np.nan
        d["turn_rate_dps"] = np.where(dt >= MIN_TURN_DT_S, dturn / dt, np.nan)
    d["cum_turn_deg"] = pd.Series(np.nan_to_num(dturn)).groupby(run).cumsum().to_numpy()
    return d


def run_summary(k: pd.DataFrame) -> pd.DataFrame:
    """One row per continuous run: duration, distance, climb, circling."""
    g = k.groupby("run")
    out = pd.DataFrame(
        {
            "start": g["timestamp"].first(),
            "end": g["timestamp"].last(),
            "n_fixes": g.size(),
            "path_m": g["step_m"].sum(),
            "median_speed_ms": g["speed_track_ms"].median(),
            "mean_climb_ms": g["climb_ms"].mean(),
            "net_turn_deg": g["cum_turn_deg"].last(),
            "abs_turn_deg": g["turn_rate_dps"].apply(lambda s: np.nansum(np.abs(s))),
        }
    )
    out["duration_s"] = (out["end"] - out["start"]).dt.total_seconds()
    disp = g.apply(lambda x: np.hypot(x["east_m"].iloc[-1] - x["east_m"].iloc[0], x["north_m"].iloc[-1] - x["north_m"].iloc[0]))
    out["straightness"] = disp / out["path_m"].replace(0, np.nan)
    return out
