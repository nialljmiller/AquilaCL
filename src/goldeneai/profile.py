"""Stage 2: sampling-schedule profile, accelerometer calibration and per-observation features.

Reads ``<data>/events`` (from ``goldeneai.ingest``) and writes, under ``<data>/profile``:

    schedule.csv             per bird and UTC month: bursts by length, burst interval, GPS cadence, 1 Hz runs
    calibration.csv          per bird/tag: fitted offset and gain, model, condition, posture spread, and the
                             published tag-generation slope for comparison
    acc/<bird>.parquet       one row per acceleration burst: time, length, nearest GPS fix, sun elevation,
                             stillness flag and calibrated burst features
    gps/<bird>.parquet       quality-filtered fixes with kinematics (rates from actual fix times) and sun elevation
    gps_runs/<bird>.parquet  one row per continuous dense-GPS run

Sun elevation uses the nearest fix within 12 h (position only needs to be approximate). A burst counts as still when its largest per-axis standard deviation is small AND either the sun is below
-6 degrees or the nearest GPS fix within ``--still-gps-s`` has ground speed below 0.5 m/s. Steady banked soaring
also gives low-variation bursts, so low variation alone is not enough.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import acc, gps, solar

ACC_RAW = "eobs:accelerations-raw"
TAG = "tag-local-identifier"


def published_slope(tag_id: str) -> float:
    """g per count from the moveACC tag-generation rule (cross-check only)."""
    try:
        t = int(tag_id)
    except (TypeError, ValueError):
        return float("nan")
    if t <= 2241:
        return float("nan")  # Gen 1: depends on the sensitivity setting
    return 0.0022 if t <= 4117 else 1 / 512


def _attach_gps(t: pd.Series, fixes: pd.DataFrame, tol_s: float) -> pd.DataFrame:
    """Nearest good GPS fix (time, lon, lat, ground speed) within ``tol_s`` of each time in ``t``."""
    left = pd.DataFrame({"timestamp": t.to_numpy()}).reset_index(names="row")
    if fixes.empty:
        return pd.DataFrame(index=left["row"], columns=["gps_dt_s", gps.LON, gps.LAT, gps.SPEED], dtype=float)
    right = fixes[["timestamp", gps.LON, gps.LAT, gps.SPEED]].copy()
    right["gps_time"] = right["timestamp"]
    m = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tol_s),
    ).set_index("row").sort_index()
    m["gps_dt_s"] = (m["timestamp"] - m["gps_time"]).dt.total_seconds()
    return m[["gps_dt_s", gps.LON, gps.LAT, gps.SPEED]]


def profile_bird(events: Path, bird: str, out: Path, still_sd_counts: float, still_gps_s: float, max_gap_s: float):
    """Process one bird; returns (schedule rows, calibration rows)."""
    g_path, a_path = events / "gps" / f"{bird}.parquet", events / "acceleration" / f"{bird}.parquet"
    fixes_all = pq.read_table(g_path).to_pandas() if g_path.exists() else pd.DataFrame(columns=["timestamp"])
    fixes = gps.good_fixes(fixes_all) if len(fixes_all) else fixes_all
    sched: list[dict] = []
    cal_rows: list[dict] = []

    # GPS kinematics and dense runs
    if len(fixes):
        kin = gps.kinematics(fixes, max_gap_s=max_gap_s)
        kin["sun_elev_deg"] = solar.sun_elevation_deg(kin["timestamp"], kin[gps.LON], kin[gps.LAT])
        runs = gps.run_summary(kin)
        (out / "gps").mkdir(parents=True, exist_ok=True)
        (out / "gps_runs").mkdir(parents=True, exist_ok=True)
        kin.to_parquet(out / "gps" / f"{bird}.parquet")
        runs.to_parquet(out / "gps_runs" / f"{bird}.parquet")
        kin["month"] = kin["timestamp"].dt.strftime("%Y-%m")
        runs["month"] = runs["start"].dt.strftime("%Y-%m")
        dt_all = fixes_all["timestamp"].diff().dt.total_seconds()
        fixes_all = fixes_all.assign(month=fixes_all["timestamp"].dt.strftime("%Y-%m"), dt=dt_all)
        for month, f in fixes_all.groupby("month"):
            k = kin[kin["month"] == month]
            r = runs[runs["month"] == month]
            sched.append(
                {
                    "bird": bird,
                    "month": month,
                    "stream": "gps",
                    "rows": len(f),
                    "good_fixes": len(k),
                    "frac_dt_le_1_5s": float((f["dt"] <= 1.5).mean()),
                    "frac_dt_1_5_to_30s": float(f["dt"].between(1.5, 30, inclusive="right").mean()),
                    "frac_dt_30_to_400s": float(f["dt"].between(30, 400, inclusive="right").mean()),
                    "frac_dt_gt_400s": float((f["dt"] > 400).mean()),
                    "median_dt_s": float(f["dt"].median()),
                    "dense_runs_ge_30s": int((r["duration_s"] >= 30).sum()),
                    "dense_run_seconds": float(r.loc[r["n_fixes"] > 1, "duration_s"].sum()),
                }
            )
    else:
        kin = pd.DataFrame()

    # Acceleration bursts
    if a_path.exists():
        a = pq.read_table(a_path).to_pandas()
        flat = list(a[ACC_RAW])
        n, mean, sd = acc.burst_stats(flat)
        b = pd.DataFrame(
            {
                "timestamp": a["timestamp"].to_numpy(),
                TAG: a[TAG].to_numpy(),
                "decoding": a["data-decoding-software"].to_numpy(),
                "n_samples": n,
                "fs_hz": a["eobs:acceleration-sampling-frequency-per-axis"].to_numpy(),
                "mean_x": mean[:, 0],
                "mean_y": mean[:, 1],
                "mean_z": mean[:, 2],
                "sd_max_counts": sd,
            }
        )
        near = _attach_gps(b["timestamp"], fixes, tol_s=600)
        b = pd.concat([b, near.reset_index(drop=True)], axis=1)
        # Sun elevation only needs an approximate position: take the nearest fix within 12 h
        # (tags may not record GPS at night, when most still bursts occur).
        pos = _attach_gps(b["timestamp"], fixes, tol_s=12 * 3600).reset_index(drop=True)
        b["pos_dt_s"] = pos["gps_dt_s"]
        b["sun_elev_deg"] = solar.sun_elevation_deg(b["timestamp"], pos[gps.LON], pos[gps.LAT])
        slow = (b["gps_dt_s"].abs() <= still_gps_s) & (b[gps.SPEED] < 0.5)
        b["still"] = (b["sd_max_counts"] < still_sd_counts) & ((b["sun_elev_deg"] < -6) | slow)

        feats = []
        for tag, idx in b.groupby(TAG).groups.items():
            sel = b.loc[idx]
            cal = acc.fit_calibration(sel[["mean_x", "mean_y", "mean_z"]].to_numpy(), still=sel["still"].to_numpy())
            slope = published_slope(tag)
            cal_rows.append(
                {
                    "bird": bird,
                    "tag": tag,
                    "bursts": len(sel),
                    "still_bursts": int(sel["still"].sum()),
                    "model": cal.model,
                    "offset_x": cal.offset[0],
                    "offset_y": cal.offset[1],
                    "offset_z": cal.offset[2],
                    "gain_x": cal.gain[0],
                    "gain_y": cal.gain[1],
                    "gain_z": cal.gain[2],
                    "condition": cal.condition,
                    "posture_spread_deg": cal.posture_spread_deg,
                    "residual_g": cal.residual_g,
                    "published_counts_per_g": 1 / slope if slope == slope else np.nan,
                    "fitted_vs_published": cal.gain.mean() * slope if slope == slope else np.nan,
                }
            )
            for i in idx:
                f = acc.burst_features(cal.apply(acc.decode(flat[i])), fs=float(b.at[i, "fs_hz"] or 20.0))
                f["row"] = i
                feats.append(f)
        b = b.join(pd.DataFrame(feats).set_index("row").drop(columns=["n_samples"]))
        (out / "acc").mkdir(parents=True, exist_ok=True)
        b.to_parquet(out / "acc" / f"{bird}.parquet")

        b["month"] = b["timestamp"].dt.strftime("%Y-%m")
        for month, m in b.groupby("month"):
            lens = m["n_samples"].value_counts()
            sched.append(
                {
                    "bird": bird,
                    "month": month,
                    "stream": "acceleration",
                    "rows": len(m),
                    "bursts_24": int(lens.get(24, 0)),
                    "bursts_40": int(lens.get(40, 0)),
                    "bursts_264": int(lens.get(264, 0)),
                    "bursts_other": int(len(m) - lens.reindex([24, 40, 264]).fillna(0).sum()),
                    "median_interval_s": float(m["timestamp"].diff().dt.total_seconds().median()),
                    "frac_with_gps_600s": float(m["gps_dt_s"].notna().mean()),
                    "frac_still": float(m["still"].mean()),
                }
            )

    for sensor in ("magnetometer", "orientation"):
        p = events / sensor / f"{bird}.parquet"
        if p.exists():
            t = pq.read_table(p, columns=["timestamp"]).to_pandas()
            for month, c in t.groupby(t["timestamp"].dt.strftime("%Y-%m")).size().items():
                sched.append({"bird": bird, "month": month, "stream": sensor, "rows": int(c)})
    return sched, cal_rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path, help="directory that holds events/ from goldeneai.ingest")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--still-sd-counts", type=float, default=8.0)
    ap.add_argument("--still-gps-s", type=float, default=120.0)
    ap.add_argument("--max-gap-s", type=float, default=3.0, help="largest gap inside a dense GPS run")
    args = ap.parse_args(argv)
    events, out = args.data / "events", args.data / "profile"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; use a new data directory or move it aside")
    birds = sorted({p.stem for s in ("gps", "acceleration") for p in (events / s).glob("*.parquet")})
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [
            pool.submit(profile_bird, events, b, out, args.still_sd_counts, args.still_gps_s, args.max_gap_s)
            for b in birds
        ]
        results = [j.result() for j in jobs]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for s, _ in results for r in s]).to_csv(out / "schedule.csv", index=False)
    cal = pd.DataFrame([r for _, c in results for r in c])
    cal.to_csv(out / "calibration.csv", index=False)
    print(f"profiled {len(birds)} birds -> {out}")
    if len(cal):
        print(cal["model"].value_counts().to_string())
        print(cal[["still_bursts", "condition", "posture_spread_deg", "residual_g", "fitted_vs_published"]].describe().to_string())


if __name__ == "__main__":
    main()
