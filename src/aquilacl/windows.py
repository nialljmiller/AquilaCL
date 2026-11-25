"""Stage 3a: cut the co-recorded streams into fixed 10 s windows.

A window starts at the beginning of a continuous accelerometer segment (bursts separated by at most
``max_acc_gap_s``) and successive windows follow every ``window_s`` seconds until the segment ends.
A short recording bout therefore gives one window; a long continuous segment gives a sequence.

Per window we store
    acc       (200, 3) float16   calibrated acceleration in g, placed by sample time at 20 Hz
                                 (consecutive bursts placed contiguously, see ``place_bursts``)
    acc_mask  (200,)   bool      which samples exist
    gps       (10, C)  float32   per-second GPS kinematics (see GPS_CHANNELS); NaN replaced by 0
    gps_mask  (10, C)  bool      which values exist
and one metadata row (bird, tag, time, segment, position in segment, sun elevation, ...).

Absolute position and absolute time are kept in the metadata for analysis only; they are not model inputs.
"""

from __future__ import annotations

import argparse
import shutil
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
SNAP_SLOTS = 2  # timestamp jitter of consecutive continuous-mode bursts (1.165-1.25 s for 1.2 s of data) is under 2 samples


def load_calibration(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, dtype={"tag": str})
    return c.set_index(["bird", "tag", "sensitivity_k"])


def segment_ids(t_start_s: np.ndarray, t_end_s: np.ndarray, max_gap_s: float) -> np.ndarray:
    """A new segment starts when a burst begins more than ``max_gap_s`` after the previous one ended."""
    gap = t_start_s[1:] - t_end_s[:-1]
    return np.cumsum(np.r_[True, gap > max_gap_s])


def place_bursts(start_idx: np.ndarray, n_samp: np.ndarray, seg: np.ndarray, snap: int = SNAP_SLOTS):
    """Sample index of each burst within its segment, never overwriting earlier samples.

    ``start_idx`` is the burst start rounded to the sample grid from the segment start. A burst starting within
    ``snap`` slots of the previous burst's placed end starts exactly at that end (continuous recording; larger timing differences stay as real gaps);
    otherwise it is placed by time. Samples that would land on already placed samples are dropped from the head
    of the later burst. Returns (placed start, leading samples dropped, snapped flag), all per burst.
    """
    start = np.asarray(start_idx, dtype=np.int64).copy()
    drop = np.zeros(len(start), dtype=np.int64)
    snapped = np.zeros(len(start), dtype=bool)
    end = 0
    for i in range(len(start)):
        if i > 0 and seg[i] == seg[i - 1]:
            if start[i] != end and abs(start[i] - end) <= snap:
                start[i], snapped[i] = end, True
            drop[i] = min(max(end - start[i], 0), n_samp[i])
            end = max(end, start[i] + n_samp[i])
        else:
            end = start[i] + n_samp[i]
    return start, drop, snapped


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

    seg_ids, first_idx, seg_pos = np.unique(seg, return_index=True, return_inverse=True)
    seg_start = ts[first_idx]  # bursts are in time order
    r0 = np.round((ts - seg_start[seg_pos]) * FS).astype(np.int64)
    start, drop, snapped = place_bursts(r0, n_samp, seg)

    # Window grid: every segment is tiled from its start in steps of window_s, covering all placed samples.
    n_slots = int(round(window_s * FS))
    seg_len = pd.Series(start + n_samp).groupby(seg_pos).max().to_numpy()
    n_k = np.maximum(1, -(-seg_len // n_slots))
    base = np.r_[0, np.cumsum(n_k)[:-1]]  # index of each segment's first window
    n_win = int(n_k.sum())
    w_seg = np.repeat(seg_ids, n_k)
    w_k = np.concatenate([np.arange(n) for n in n_k])
    w_t0 = np.repeat(seg_start, n_k) + w_k * window_s

    acc_arr = np.zeros((n_win, n_slots, 3), dtype=np.float16)
    acc_mask = np.zeros((n_win, n_slots), dtype=bool)
    burst_window = np.empty(len(a), dtype=np.int64)  # window holding each burst's first placed sample
    calib_cache: dict = {}
    unreliable_bursts = 0
    for i, (raw, tag, kk) in enumerate(zip(a[ACC_RAW], a[TAG], a["sensitivity_k"])):
        p = seg_pos[i]
        burst_window[i] = base[p] + min(start[i] + drop[i], seg_len[p] - 1) // n_slots
        if drop[i] == n_samp[i]:
            continue
        key = (bird, str(tag), int(kk))
        if key not in calib_cache:
            r = cal.loc[key]
            # bursts of a calibration flagged unreliable by the profile stage are left out, not silently used
            calib_cache[key] = None if not bool(r.get("reliable", True)) else acc.Calibration(
                r[["offset_x", "offset_y", "offset_z"]].to_numpy(float),
                r[["gain_x", "gain_y", "gain_z"]].to_numpy(float),
                r["model"], int(r["still_bursts"]), r["posture_spread_deg"], r["condition"], r["residual_g"],
            )
        if calib_cache[key] is None:
            unreliable_bursts += 1
            continue
        g = calib_cache[key].apply(acc.decode(raw))[drop[i]:]
        idx = start[i] + drop[i] + np.arange(len(g))
        w, slot = base[p] + idx // n_slots, idx % n_slots
        acc_arr[w, slot] = g
        acc_mask[w, slot] = True

    keep = acc_mask.sum(axis=1) >= int(min_acc_s * FS)

    # GPS: per-second slots inside each window
    gps_file = profile / "gps" / f"{bird}.parquet"  # absent when a bird has no good fixes
    fixes = pq.read_table(gps_file).to_pandas() if gps_file.exists() else pd.DataFrame()
    gps_arr = np.zeros((n_win, int(window_s), len(GPS_CHANNELS)), dtype=np.float32)
    gps_mask = np.zeros_like(gps_arr, dtype=bool)
    if len(fixes):
        tf = ((fixes["timestamp"] - epoch) / pd.Timedelta(seconds=1)).to_numpy()
        order = np.argsort(w_t0)
        pos = np.searchsorted(w_t0[order], tf, side="right") - 1
        valid = pos >= 0
        widx = np.where(valid, order[np.clip(pos, 0, None)], -1)
        off = tf - w_t0[np.clip(widx, 0, None)]
        slot = np.floor(off).astype(int)
        ok = np.flatnonzero(valid & (slot >= 0) & (slot < int(window_s)))
        # one fix per (window, second): the one closest to the slot start
        key = widx[ok] * int(window_s) + slot[ok]
        o = np.lexsort((off[ok], key))
        if len(ok):
            ok = ok[o][np.r_[True, key[o][1:] != key[o][:-1]]]
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
               "bursts_unreliable_calibration": unreliable_bursts,
               "with_gps": int((meta["gps_fixes"] > 0).sum()),
               "bursts_snapped": int(snapped.sum()), "samples_dropped_overlap": int(drop.sum())}
    summary.update({c: int(np.nansum(labels[c])) for c in labels})
    summary.update({c + "_known": int(labels[c].notna().sum()) for c in labels})
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
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.profile / "format.json").exists():  # common acceleration format, read by training from here
        shutil.copyfile(args.profile / "format.json", args.out / "format.json")
    birds = sorted(p.stem for p in (args.profile / "acc").glob("*.parquet"))
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(build_bird, birds, [args.data / "events"] * len(birds), [args.profile] * len(birds),
                            [cal] * len(birds), [args.out] * len(birds), [args.window_s] * len(birds)))
    r = pd.DataFrame(res)
    r.to_csv(args.out / "windows_summary.csv", index=False)
    print(r.sum(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
