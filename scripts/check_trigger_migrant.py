"""Two checks on the raw GPS, reported as aggregates only (no coordinates in stdout or in the json).

1. What starts continuous mode. Continuous bouts are runs of GPS fixes 1-2 s apart (at least 10 fixes).
   For each bout onset: ground speed of the last scheduled fix before the bout (if within 15 min), speed of the
   first bout fixes, hour (UTC), battery voltage; compared with all scheduled fixes. Also: after a scheduled fix
   with speed > 5 m/s, how often does a bout start within 15 min, against scheduled fixes with speed < 1 m/s.
2. What 'migrant' in the reference notes means. Per bird: maximum distance (km) from the study-area centre
   (median of all birds' fixes) and maximum northward displacement (km), the month in which the bird is furthest
   north, and the share of fixes > 500 km from the centre; compared between migrant-flagged, young-tagged and
   other birds.

Usage: python scripts/check_trigger_migrant.py --gps <raw/gps dir> --reference <csv> --out <new json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquilacl import reference

ap = argparse.ArgumentParser()
ap.add_argument("--gps", type=Path, required=True)
ap.add_argument("--reference", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--min-bout", type=int, default=10, help="fixes 1-2 s apart that make a continuous bout")
a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"{a.out} exists")

COLS = ["timestamp", "location-long", "location-lat", "ground-speed", "eobs:battery-voltage"]
R_EARTH = 6371.0


def hav_km(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(h))


files = sorted(a.gps.glob("*.parquet"))
# study-area centre: median of per-bird median positions (kept in memory only)
meds = []
for f in files:
    g = pd.read_parquet(f, columns=["location-long", "location-lat"]).dropna()
    g = g[(g["location-lat"].abs() > 1e-6) | (g["location-long"].abs() > 1e-6)]
    if len(g):
        meds.append(g.median().to_numpy())
c_lon, c_lat = np.median(np.array(meds), axis=0)

onsets, sched_all, per_bird = [], [], []
for f in files:
    bird = f.stem
    g = pd.read_parquet(f, columns=COLS).dropna(subset=["timestamp"]).sort_values("timestamp")
    g = g.drop_duplicates("timestamp")
    t = g["timestamp"].astype("int64").to_numpy() // 1000  # s
    v = g["ground-speed"].to_numpy(float)
    batt = g["eobs:battery-voltage"].to_numpy(float)
    dt = np.diff(t)
    fast = np.r_[False, dt <= 2]  # fix i is within 2 s of fix i-1
    # runs of 'fast' links
    run_id = np.cumsum(~fast)
    sizes = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    in_bout = sizes >= a.min_bout
    start = in_bout & np.r_[True, run_id[1:] != run_id[:-1]]  # first fix of each bout
    sched = ~in_bout
    sched_all.append(pd.DataFrame({"v": v[sched], "batt": batt[sched], "hour": (t[sched] // 3600) % 24}))
    for i in np.flatnonzero(start):
        j = i - 1
        prev_v = v[j] if j >= 0 and sched[j] and t[i] - t[j] <= 900 else np.nan
        first_v = np.nanmedian(v[i:i + 5])
        n = int(sizes[i])
        end = i + n - 1
        nxt = end + 1
        onsets.append({"prev_sched_v": prev_v, "first5_v": first_v, "hour": int((t[i] // 3600) % 24),
                       "batt": batt[i], "bout_s": int(t[end] - t[i]), "last5_v": np.nanmedian(v[max(i, end - 4):end + 1]),
                       "next_sched_v": v[nxt] if nxt < len(v) and t[nxt] - t[end] <= 900 else np.nan})
    # trigger sensitivity: scheduled fixes followed by a bout start within 15 min
    st = t[start]
    si = np.flatnonzero(sched)
    k = np.searchsorted(st, t[si], side="right")
    has = ((k < len(st)) & ((st[np.minimum(k, len(st) - 1)] - t[si]) <= 900)) if len(st) else np.zeros(len(si), bool)
    per_bird.append({"bird": bird, "sched_fast_then_bout": float(has[v[si] > 5].mean()) if (v[si] > 5).any() else np.nan,
                     "sched_slow_then_bout": float(has[v[si] < 1].mean()) if (v[si] < 1).any() else np.nan,
                     "n_bouts": int(start.sum())})
    # migration geometry
    ll = g[["location-lat", "location-long"]].to_numpy(float)
    ok = np.isfinite(ll).all(1) & ((np.abs(ll[:, 0]) > 1e-6) | (np.abs(ll[:, 1]) > 1e-6))
    if ok.sum():
        d = hav_km(c_lat, c_lon, ll[ok, 0], ll[ok, 1])
        north = (ll[ok, 0] - c_lat) * (np.pi / 180) * R_EARTH
        months = pd.to_datetime(g["timestamp"].to_numpy()[ok]).month
        per_bird[-1].update({"max_dist_km": float(d.max()), "max_north_km": float(north.max()),
                             "min_north_km": float(north.min()), "share_gt500km": float((d > 500).mean()),
                             "month_furthest_north": int(months[np.argmax(north)]), "days": float((t[-1] - t[0]) / 86400)})

on = pd.DataFrame(onsets)
sc = pd.concat(sched_all)
pb = pd.DataFrame(per_bird)
ref = reference.load_birds(a.reference, pb["bird"].tolist()).set_index("bird")
pb["group"] = np.where(ref.loc[pb.bird, "migrant"].eq(True).to_numpy(), "migrant",
                       np.where(ref.loc[pb.bird, "tagged_young"].eq(True).to_numpy(), "young_tagged", "other"))


def q(s):
    s = pd.Series(s).dropna()
    return {"n": int(len(s)), "q10": float(s.quantile(.1)) if len(s) else None, "median": float(s.median()) if len(s) else None,
            "q90": float(s.quantile(.9)) if len(s) else None}


out = {
    "trigger": {
        "bouts": int(len(on)), "birds_with_bouts": int((pb.n_bouts > 0).sum()),
        "prev_scheduled_fix_speed_ms": q(on.prev_sched_v), "all_scheduled_fix_speed_ms": q(sc.v),
        "share_prev_sched_gt5ms": float((on.prev_sched_v > 5).mean()), "share_all_sched_gt5ms": float((sc.v > 5).mean()),
        "first5_fix_speed_ms": q(on.first5_v), "share_first5_gt5ms": float((on.first5_v > 5).mean()),
        "last5_fix_speed_ms": q(on.last5_v), "next_scheduled_after_bout_speed_ms": q(on.next_sched_v),
        "bout_duration_s": q(on.bout_s),
        "onset_hour_utc_share": on.hour.value_counts(normalize=True).sort_index().round(3).to_dict(),
        "sched_hour_utc_share": sc.hour.value_counts(normalize=True).sort_index().round(3).to_dict(),
        "battery_v_onset": q(on.batt), "battery_v_scheduled": q(sc.batt),
        "per_bird_p_bout_within_15min_after_fast_sched": q(pb.sched_fast_then_bout),
        "per_bird_p_bout_within_15min_after_slow_sched": q(pb.sched_slow_then_bout),
    },
    "migrant": {grp: {"birds": int(len(d)), "max_dist_km": q(d.max_dist_km), "max_north_km": q(d.max_north_km),
                      "share_fixes_gt500km": q(d.share_gt500km), "birds_max_north_gt1000km": int((d.max_north_km > 1000).sum()),
                      "month_furthest_north": d.month_furthest_north.value_counts().sort_index().to_dict(),
                      "track_days": q(d.days)}
                for grp, d in pb.groupby("group")},
}
a.out.write_text(json.dumps(out, indent=1, default=str))
pb.to_csv(a.out.with_suffix(".per_bird.csv"), index=False)  # per-bird distances only, no coordinates
print(json.dumps(out, indent=1, default=str))
