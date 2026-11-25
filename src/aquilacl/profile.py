"""Stage 2: sampling-schedule profile, accelerometer calibration and per-observation features.

Reads ``<data>/events`` (from ``aquilacl.ingest``) and writes, under ``<data>/profile``:

    schedule.csv             per bird and UTC month: bursts by length, burst interval, GPS cadence, 1 Hz runs
    calibration.csv          per bird/tag/sensitivity setting k: fitted offset and gain, model, condition, posture spread,
                             reliability flags, and the published tag-generation slope for comparison
    format.json              common acceleration format for training (step_g, clip_g) and the per-period values behind it
    acc/<bird>.parquet       one row per acceleration burst: time, length, nearest GPS fix, sun elevation,
                             stillness flag and calibrated burst features
    gps/<bird>.parquet       quality-filtered fixes with kinematics (rates from actual fix times) and sun elevation
    gps_runs/<bird>.parquet  one row per continuous dense-GPS run

Sun elevation uses the nearest fix within 12 h (position only needs to be approximate). A burst is "at rest" when
the sun is below -6 degrees or the nearest GPS fix within ``--still-gps-s`` has ground speed below 0.5 m/s. Steady
banked soaring also gives low-variation bursts, so low variation alone is not enough.

Sensitivity: a burst at rest is "quiet" when its largest per-axis standard deviation is below 1% of its raw
mean-vector length (scale-free, so it works before the setting is known). Each (tag, burst length) group gets
k = round(log2(median quiet length / 512)); groups with fewer than 20 quiet bursts take the k most groups of that
burst length have across all birds (``acc.group_sensitivity``). A burst is "still" (used for calibration) when it is
at rest and its largest per-axis standard deviation is below ``--still-sd-counts`` / 512 g, i.e. that many counts at
512 counts/g and twice as many at 1024 counts/g.

The run makes two passes over the bursts: the first collects the quiet-burst statistics of every group of every
bird (for the cross-bird majority), the second assigns k, calibrates and computes the features.
"""

from __future__ import annotations

import argparse
import json
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
    left = pd.DataFrame({"timestamp": pd.Series(t.to_numpy()).astype("datetime64[ns]")}).reset_index(names="row")
    if fixes.empty:
        return pd.DataFrame(index=left["row"], columns=["gps_dt_s", gps.LON, gps.LAT, gps.SPEED], dtype=float)
    right = fixes[["timestamp", gps.LON, gps.LAT, gps.SPEED]].copy()
    right["timestamp"] = right["timestamp"].astype("datetime64[ns]")
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


def _fixes(events: Path, bird: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All GPS rows and the quality-filtered fixes of one bird."""
    g_path = events / "gps" / f"{bird}.parquet"
    fixes_all = pq.read_table(g_path).to_pandas() if g_path.exists() else pd.DataFrame(columns=["timestamp"])
    return fixes_all, (gps.good_fixes(fixes_all) if len(fixes_all) else fixes_all)


def _bursts(a: pd.DataFrame, fixes: pd.DataFrame, still_gps_s: float) -> tuple[pd.DataFrame, list, pd.Series]:
    """Per-burst table (stats, nearest GPS, sun elevation, at-rest and quiet flags) and the raw bursts."""
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
            "norm_counts": np.linalg.norm(mean - acc.RAW_CENTRE, axis=1),
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
    rest = (b["sun_elev_deg"] < -6) | slow
    b["quiet"] = rest & (b["sd_max_counts"] < acc.QUIET_REL_SD * b["norm_counts"])
    return b, flat, rest


def _group_stats(b: pd.DataFrame) -> pd.DataFrame:
    """Quiet-burst count and median quiet mean-vector length per (tag, burst length)."""
    q = b["norm_counts"].where(b["quiet"])
    g = pd.DataFrame({TAG: b[TAG], "n_samples": b["n_samples"], "quiet": b["quiet"], "q": q})
    return (
        g.groupby([TAG, "n_samples"])
        .agg(n_bursts=("quiet", "size"), n_quiet=("quiet", "sum"), median_norm=("q", "median"))
        .reset_index()
        .rename(columns={TAG: "tag"})
    )


def quiet_groups(events: Path, bird: str, still_gps_s: float) -> pd.DataFrame:
    """First pass: quiet-burst statistics per (tag, burst length) of one bird (reads only)."""
    a_path = events / "acceleration" / f"{bird}.parquet"
    if not a_path.exists():
        return pd.DataFrame(columns=["bird", "tag", "n_samples", "n_bursts", "n_quiet", "median_norm"])
    a = pq.read_table(a_path, columns=["timestamp", TAG, ACC_RAW, "data-decoding-software",
                                       "eobs:acceleration-sampling-frequency-per-axis"]).to_pandas()
    b, _, _ = _bursts(a, _fixes(events, bird)[1], still_gps_s)
    return _group_stats(b).assign(bird=bird)


def profile_bird(
    events: Path,
    bird: str,
    out: Path,
    still_sd_counts: float,
    still_gps_s: float,
    max_gap_s: float,
    length_k: dict[int, int] | None = None,
):
    """Process one bird; returns (schedule rows, calibration rows).

    ``length_k`` is the majority k per burst length over all birds (from the first pass); groups of
    this bird with too few quiet bursts take it. When None, the majority over this bird's own groups is used.
    """
    a_path = events / "acceleration" / f"{bird}.parquet"
    fixes_all, fixes = _fixes(events, bird)
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
        b, flat, rest = _bursts(a, fixes, still_gps_s)
        grp = acc.group_sensitivity(_group_stats(b), length_k).set_index(["tag", "n_samples"])
        key = pd.MultiIndex.from_arrays([b[TAG], b["n_samples"]])
        b["sensitivity_k"] = grp["k"].reindex(key).to_numpy().astype(int)
        b["k_source"] = grp["k_source"].reindex(key).to_numpy()
        # Still (for calibration): the --still-sd-counts threshold holds at 512 counts/g and scales with the setting.
        still_sd_g = still_sd_counts / acc.BASE_COUNTS_PER_G
        b["still"] = rest & (b["sd_max_counts"] < still_sd_g * acc.BASE_COUNTS_PER_G * 2.0 ** b["sensitivity_k"])

        feats = []
        means = b[["mean_x", "mean_y", "mean_z"]].to_numpy()
        for tag, rows in b.groupby(TAG).groups.items():
            rows = np.asarray(rows)
            ks = b.loc[rows, "sensitivity_k"].to_numpy()
            cals = acc.calibrate_tag(means[rows], b.loc[rows, "still"].to_numpy(), ks)
            for k, cal in cals.items():
                idx = rows[ks == k]
                sel = b.loc[idx]
                g = grp.loc[tag]
                g = g[g["k"] == k]
                flags = acc.calibration_flags(cal)
                slope = published_slope(tag) / 2**k
                cal_rows.append(
                    {
                        "bird": bird,
                        "tag": tag,
                        "sensitivity_k": int(k),
                        "first": sel["timestamp"].min(),
                        "last": sel["timestamp"].max(),
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
                        "median_still_g": cal.median_g,
                        "reliable": not flags,
                        "flags": ";".join(flags),
                        "n_groups": len(g),
                        "burst_lengths": ";".join(str(int(n)) for n in g.index),
                        "k_source": ";".join(sorted(set(g["k_source"]))),
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


def common_format(cal: pd.DataFrame, clip_margin_g: float = 0.05, outlier_sd: float = 3.0) -> dict:
    """Common acceleration format that hides the sensitivity setting, from the reliable calibrations.

    ``step_g``: the coarsest calibrated step among k=0 periods, max over periods and axes of 1/gain.
    ``clip_g``: the narrowest saturation level among k=1 periods, min over periods and axes of
    min(offset, 4095 - offset) / gain (``saturation_g`` per period), minus ``clip_margin_g``. Only periods with
    ``reliable`` true count; ``<name>_periods`` lists each one's value and ``<name>_set_by`` the deciding one.

    A single doubtful fit should not set the format for every bird (a full fit with a 15% gain error once gave a
    clip of 1.52 g against 1.90-1.96 g for most periods). Periods more than ``outlier_sd`` robust standard
    deviations (1.4826 x MAD) beyond the median in the deciding direction are left out of the extreme and listed
    in ``<name>_excluded``; their own data keep their own step or rail.
    """
    ax = ["x", "y", "z"]
    ok = cal[cal["reliable"].astype(bool)]
    out: dict = {"clip_margin_g": clip_margin_g}
    for name, key, k, value, pick in (
        ("step_g", "step_g", 0, lambda r: max(1 / r[f"gain_{a}"] for a in ax), max),
        ("clip_g", "saturation_g", 1,
         lambda r: min(min(r[f"offset_{a}"], acc.RAW_MAX - r[f"offset_{a}"]) / r[f"gain_{a}"] for a in ax), min),
    ):
        periods = [
            {"bird": r["bird"], "tag": str(r["tag"]), "sensitivity_k": k, "model": r["model"], key: float(value(r))}
            for _, r in ok[ok["sensitivity_k"] == k].iterrows()
        ]
        v = np.array([d[key] for d in periods])
        excluded = []
        if len(v) >= 5:
            med = np.median(v)
            sd = 1.4826 * np.median(np.abs(v - med))
            far = (v > med + outlier_sd * sd) if pick is max else (v < med - outlier_sd * sd)
            excluded = [d for d, f in zip(periods, far) if f]
            periods_used = [d for d, f in zip(periods, far) if not f]
        else:
            periods_used = periods
        best = pick(periods_used, key=lambda d: d[key]) if periods_used else None
        out[f"{name}_excluded"] = excluded
        out[name] = None if best is None else best[key] - (clip_margin_g if name == "clip_g" else 0.0)
        out[f"{name}_set_by"] = best
        out[f"{name}_periods"] = periods
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path, help="directory that holds events/ from aquilacl.ingest")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--still-sd-counts", type=float, default=8.0, help="still threshold in counts at 512 counts/g")
    ap.add_argument("--still-gps-s", type=float, default=120.0)
    ap.add_argument("--max-gap-s", type=float, default=3.0, help="largest gap inside a dense GPS run")
    ap.add_argument("--out", type=Path, help="output directory (default <data>/profile)")
    args = ap.parse_args(argv)
    events, out = args.data / "events", args.out or args.data / "profile"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; use a new data directory or move it aside")
    birds = sorted({p.stem for s in ("gps", "acceleration") for p in (events / s).glob("*.parquet")})
    with ProcessPoolExecutor(args.workers) as pool:
        groups = pd.concat([j.result() for j in [pool.submit(quiet_groups, events, b, args.still_gps_s) for b in birds]])
        groups = acc.group_sensitivity(groups)
        length_k = acc.length_majority(groups)
        jobs = [
            pool.submit(profile_bird, events, b, out, args.still_sd_counts, args.still_gps_s, args.max_gap_s, length_k)
            for b in birds
        ]
        results = [j.result() for j in jobs]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r for s, _ in results for r in s]).to_csv(out / "schedule.csv", index=False)
    cal = pd.DataFrame([r for _, c in results for r in c])
    cal.to_csv(out / "calibration.csv", index=False)
    fmt = common_format(cal) if len(cal) else {}
    (out / "format.json").write_text(json.dumps(fmt, indent=1))
    print(f"profiled {len(birds)} birds -> {out}")
    print(f"majority k per burst length: {length_k}")
    if len(groups):
        print(groups.groupby(["n_samples", "k", "k_source"]).size().to_string())
    if len(cal):
        print(cal["model"].value_counts().to_string())
        print(f"unreliable periods: {int((~cal['reliable']).sum())} of {len(cal)}")
        print(cal[["still_bursts", "condition", "posture_spread_deg", "residual_g", "fitted_vs_published"]].describe().to_string())
        print(f"step_g {fmt['step_g']}  clip_g {fmt['clip_g']}")


if __name__ == "__main__":
    main()
