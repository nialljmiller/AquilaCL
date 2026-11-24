"""Stage 3a: cut the co-recorded streams into fixed 10 s windows.

A window starts at the beginning of a continuous accelerometer segment (bursts separated by at most
``max_acc_gap_s``) and successive windows follow every ``window_s`` seconds until the segment ends.
A short recording bout therefore gives one window; a long continuous segment gives a sequence.

Per window we store
    acc       (200, 3) float16   calibrated acceleration in g, placed by sample time at 20 Hz
    acc_mask  (200,)   bool      which samples exist
    gps       (10, C)  float32   per-second GPS kinematics (see GPS_CHANNELS); NaN replaced by 0
    gps_mask  (10, C)  bool      which values exist
and one metadata row (bird, tag, time, segment, position in segment, sun elevation, ...).

Absolute position and absolute time are kept in the metadata for analysis only; they are not model inputs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import acc, gps, solar

FS = 20.0
GPS_CHANNELS = ["ground-speed", "speed_track_ms", "climb_ms", "turn_rate_dps", "step_m", gps.HACC]
ACC_RAW = "eobs:accelerations-raw"
TAG = "tag-local-identifier"


def load_calibration(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, dtype={"tag": str})
    return c.set_index(["bird", "tag", "sensitivity_k"])


def segment_ids(t_start_s: np.ndarray, t_end_s: np.ndarray, max_gap_s: float) -> np.ndarray:
    """A new segment starts when a burst begins more than ``max_gap_s`` after the previous one ended."""
    gap = t_start_s[1:] - t_end_s[:-1]
    return np.cumsum(np.r_[True, gap > max_gap_s])


def build_bird(
    bird: str,
    events: Path,
    profile: Path,
    cal: pd.DataFrame,
    out: Path,
    window_s: float = 10.0,
    max_acc_gap_s: float = 1.5,
    min_acc_s: float = 1.2,
) -> dict:
    """Build all windows of one bird and write them to ``out/<bird>/``."""
    a = pq.read_table(events / "acceleration" / f"{bird}.parquet", columns=["timestamp", TAG, ACC_RAW, "data-decoding-software"]).to_pandas()
    prof = pq.read_table(profile / "acc" / f"{bird}.parquet", columns=["timestamp", "sensitivity_k", "sun_elev_deg"]).to_pandas()
    if len(a) != len(prof) or not (a["timestamp"].to_numpy() == prof["timestamp"].to_numpy()).all():
        raise ValueError(f"{bird}: acceleration events and profile rows are not aligned")
    a["sensitivity_k"] = prof["sensitivity_k"].to_numpy()
    a["sun_elev_deg"] = prof["sun_elev_deg"].to_numpy()

    epoch = a["timestamp"].iloc[0]
    ts = ((a["timestamp"] - epoch) / pd.Timedelta(seconds=1)).to_numpy()
    n_samp = np.array([len(x) // 3 for x in a[ACC_RAW]])
    te = ts + n_samp / FS
    seg = segment_ids(ts, te, max_acc_gap_s)

    # Window grid: every segment is tiled from its start in steps of window_s.
    seg_start = pd.Series(ts).groupby(seg).transform("min").to_numpy()
    seg_end = pd.Series(te).groupby(seg).transform("max").to_numpy()
    seg_ids, first_idx = np.unique(seg, return_index=True)
    n_k = np.maximum(1, np.ceil((seg_end[first_idx] - seg_start[first_idx]) / window_s).astype(int))
    base = np.r_[0, np.cumsum(n_k)[:-1]]  # index of each segment's first window
    n_win = int(n_k.sum())
    w_seg = np.repeat(seg_ids, n_k)
    w_k = np.concatenate([np.arange(n) for n in n_k])
    w_t0 = np.repeat(seg_start[first_idx], n_k) + w_k * window_s
    seg_pos = np.searchsorted(seg_ids, seg)  # segment index per burst

    n_slots = int(round(window_s * FS))
    acc_arr = np.zeros((n_win, n_slots, 3), dtype=np.float16)
    acc_mask = np.zeros((n_win, n_slots), dtype=bool)
    burst_window = np.empty(len(a), dtype=np.int64)  # window holding each burst's first sample
    calib_cache: dict = {}
    for i, (raw, tag, kk) in enumerate(zip(a[ACC_RAW], a[TAG], a["sensitivity_k"])):
        key = (bird, str(tag), int(kk))
        if key not in calib_cache:
            r = cal.loc[key]
            calib_cache[key] = acc.Calibration(
                r[["offset_x", "offset_y", "offset_z"]].to_numpy(float),
                r[["gain_x", "gain_y", "gain_z"]].to_numpy(float),
                r["model"], int(r["still_bursts"]), r["posture_spread_deg"], r["condition"], r["residual_g"],
            )
        g = calib_cache[key].apply(acc.decode(raw))
        p = seg_pos[i]
        # integer sample index from the segment start avoids rounding collisions
        idx = int(round((ts[i] - seg_start[first_idx[p]]) * FS)) + np.arange(len(g))
        k = np.minimum(idx // n_slots, n_k[p] - 1)
        slot = np.minimum(idx - k * n_slots, n_slots - 1)
        w = base[p] + k
        acc_arr[w, slot] = g
        acc_mask[w, slot] = True
        burst_window[i] = w[0]

    keep = acc_mask.sum(axis=1) >= int(min_acc_s * FS)

    # GPS: per-second slots inside each window
    fixes = pq.read_table(profile / "gps" / f"{bird}.parquet").to_pandas()
    gps_arr = np.zeros((n_win, int(window_s), len(GPS_CHANNELS)), dtype=np.float32)
    gps_mask = np.zeros_like(gps_arr, dtype=bool)
    if len(fixes):
        tf = ((fixes["timestamp"] - epoch) / pd.Timedelta(seconds=1)).to_numpy()
        order = np.argsort(w_t0)
        pos = np.searchsorted(w_t0[order], tf, side="right") - 1
        valid = pos >= 0
        widx = np.where(valid, order[np.clip(pos, 0, None)], -1)
        slot = np.floor(tf - w_t0[np.clip(widx, 0, None)]).astype(int)
        ok = valid & (slot >= 0) & (slot < int(window_s))
        vals = fixes[GPS_CHANNELS].to_numpy(dtype=np.float32)
        fin = np.isfinite(vals)
        gps_arr[widx[ok], slot[ok]] = np.nan_to_num(vals[ok])
        gps_mask[widx[ok], slot[ok]] = fin[ok]
    # burst whose data starts each window (for tag, sensitivity, decoding, sun elevation)
    first_burst = np.full(n_win, -1)
    first_burst[burst_window[::-1]] = np.arange(len(a))[::-1]
    empty = first_burst < 0
    first_burst[empty] = np.searchsorted(burst_window, np.flatnonzero(empty), side="right") - 1

    t0 = epoch + pd.to_timedelta(w_t0, unit="s")
    meta = pd.DataFrame(
        {
            "bird": bird,
            "tag": a[TAG].to_numpy()[first_burst].astype(str),
            "t_start": t0,
            "segment": w_seg,
            "k_in_segment": w_k,
            "segment_windows": pd.Series(w_k).groupby(w_seg).transform("size").to_numpy(),
            "sensitivity_k": a["sensitivity_k"].to_numpy()[first_burst],
            "decoding": a["data-decoding-software"].to_numpy()[first_burst],
            "sun_elev_deg": a["sun_elev_deg"].to_numpy()[first_burst],
            "acc_samples": acc_mask.sum(axis=1),
            "gps_fixes": gps_mask[:, :, 0].sum(axis=1) if len(GPS_CHANNELS) else 0,
        }
    )
    if len(fixes):
        # nearest fix within 12 h, for local solar time only
        near = pd.merge_asof(
            pd.DataFrame({"t": pd.Series(t0).astype("datetime64[ns]")}).sort_values("t"),
            pd.DataFrame(
                {"t": fixes["timestamp"].astype("datetime64[ns]"), "lon": fixes[gps.LON], "lat": fixes[gps.LAT]}
            ).sort_values("t"),
            on="t", direction="nearest", tolerance=pd.Timedelta(hours=12),
        )
        meta["local_solar_hour"] = solar.local_solar_hour(near["t"], near["lon"].to_numpy())
    else:
        meta["local_solar_hour"] = np.nan
    meta["gps_fixes"] = gps_mask[:, :, 0].sum(axis=1)

    meta, acc_arr, acc_mask, gps_arr, gps_mask = (
        meta[keep].reset_index(drop=True), acc_arr[keep], acc_mask[keep], gps_arr[keep], gps_mask[keep]
    )
    from .features import acc_features, gps_features, reference_labels

    feats = pd.concat([acc_features(acc_arr, acc_mask), gps_features(gps_arr, gps_mask)], axis=1)
    labels = reference_labels(feats, meta["sun_elev_deg"].to_numpy())
    d = out / bird
    d.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(d / "meta.parquet")
    pd.concat([feats, labels.add_prefix("label_")], axis=1).to_parquet(d / "features.parquet")
    np.save(d / "acc.npy", acc_arr)
    np.save(d / "acc_mask.npy", acc_mask)
    np.save(d / "gps.npy", gps_arr)
    np.save(d / "gps_mask.npy", gps_mask)
    summary = {"bird": bird, "windows": len(meta), "dropped_short": int((~keep).sum()),
               "with_gps": int((meta["gps_fixes"] > 0).sum())}
    summary.update({c: int(labels[c].sum()) for c in labels})
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data", type=Path, help="directory holding events/")
    ap.add_argument("--profile", type=Path, required=True, help="stage-2 output directory (profile_v2)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    cal = load_calibration(args.profile / "calibration.csv")
    birds = sorted(p.stem for p in (args.profile / "acc").glob("*.parquet"))
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(build_bird, birds, [args.data / "events"] * len(birds), [args.profile] * len(birds),
                            [cal] * len(birds), [args.out] * len(birds), [args.window_s] * len(birds)))
    r = pd.DataFrame(res)
    r.to_csv(args.out / "windows_summary.csv", index=False)
    print(r.sum(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
