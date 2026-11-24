"""Sun elevation from time and position (NOAA approximation, about 0.5 degree accuracy).

Used instead of UTC hour for anything about time of day: at the study site 06-11 UTC is local night.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sun_elevation_deg(t: pd.Series | pd.DatetimeIndex, lon_deg, lat_deg) -> np.ndarray:
    """Sun elevation in degrees for UTC timestamps ``t`` at ``lon_deg``/``lat_deg`` (east, north positive)."""
    t = pd.DatetimeIndex(pd.to_datetime(t, utc=True))
    jd = (t - pd.Timestamp("2000-01-01 12:00", tz="UTC")) / pd.Timedelta(days=1)
    jd = np.asarray(jd, dtype=float)
    n = jd / 36525.0  # Julian centuries since J2000
    L0 = np.radians((280.46646 + n * (36000.76983 + n * 0.0003032)) % 360)
    M = np.radians(357.52911 + n * (35999.05029 - 0.0001537 * n))
    e = 0.016708634 - n * (0.000042037 + 0.0000001267 * n)
    C = np.radians(
        np.sin(M) * (1.914602 - n * (0.004817 + 0.000014 * n))
        + np.sin(2 * M) * (0.019993 - 0.000101 * n)
        + np.sin(3 * M) * 0.000289
    )
    lam = L0 + C - np.radians(0.00569 + 0.00478 * np.sin(np.radians(125.04 - 1934.136 * n)))
    eps = np.radians(23.0 + (26.0 + (21.448 - n * (46.815 + n * (0.00059 - n * 0.001813))) / 60) / 60)
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    y = np.tan(eps / 2) ** 2
    eqt = 4 * np.degrees(
        y * np.sin(2 * L0)
        - 2 * e * np.sin(M)
        + 4 * e * y * np.sin(M) * np.cos(2 * L0)
        - 0.5 * y * y * np.sin(4 * L0)
        - 1.25 * e * e * np.sin(2 * M)
    )  # minutes
    minutes_utc = np.asarray((t - t.normalize()) / pd.Timedelta(minutes=1), dtype=float)
    true_solar_min = (minutes_utc + eqt + 4 * np.asarray(lon_deg, dtype=float)) % 1440
    hour_angle = np.radians(true_solar_min / 4 - 180)
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    cos_z = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return 90 - np.degrees(np.arccos(np.clip(cos_z, -1, 1)))


def local_solar_hour(t, lon_deg) -> np.ndarray:
    """Mean local solar time in hours (UTC + longitude / 15)."""
    t = pd.DatetimeIndex(pd.to_datetime(t, utc=True))
    h = np.asarray((t - t.normalize()) / pd.Timedelta(hours=1), dtype=float)
    return (h + np.asarray(lon_deg, dtype=float) / 15.0) % 24
