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
    Rates are NaN for the first fix of each run.
    """
    d = df.sort_values("timestamp").reset_index(drop=True).copy()
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
        step = np.hypot(de, dn)
        d["dt_s"] = dt
        d["step_m"] = np.where(same, step, np.nan)
        d["speed_track_ms"] = np.where(same & (dt > 0), step / dt, np.nan)
        d["climb_ms"] = np.where(same & (dt > 0), dz / dt, np.nan)
        bearing = np.degrees(np.arctan2(de, dn)) % 360
        bearing[~same | (step < 0.5)] = np.nan
        d["bearing_deg"] = bearing
        dturn = (np.r_[np.nan, np.diff(bearing)] + 180) % 360 - 180
        prev_same = np.r_[False, same[1:] & same[:-1]]
        dturn[~prev_same] = np.nan
        d["turn_rate_dps"] = np.where(dt > 0, dturn / dt, np.nan)
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
