"""Time budgets (share of TIME per behaviour) from the scheduled, behaviour-independent recording stream.

Why not window counts. Continuous mode (sensitivity_k == 1, 1 Hz GPS) is switched on when the bird moves fast
(bout onsets median 11.6 m/s, RESULTS_stage6 §2) and then produces one 10-s window every 10 s for up to ~900 s.
A flight therefore yields up to ~90 windows while a scheduled burst yields one, so the share of WINDOWS in a state
over-weights flight by an amount that depends on the trigger, the season and the bird. Continuous-mode windows are
excluded from the scheduled sample for that reason; they are used only (a) as direct evidence of flight for bout
time that the schedule does not cover (below) and (b) for the composition of flight, conditional on being in a
triggered bout.

Scheduled samples. Per bird, windows with sensitivity_k == 0 sorted by t_start; a window starts a new burst when it
begins more than BURST_GAP_S (10 s + 1 s tolerance) after the previous k == 0 window (a 264-sample burst gives two
windows exactly 10 s apart). The first window of each burst is one scheduled sample. The gap rule is used instead of
k_in_segment == 0 because windows.py merges a k == 0 burst into a continuous segment when it is within 1.5 s of
it and drops short windows after numbering; the disagreement with k_in_segment == 0 is counted.

Schedule. Start-to-start gaps between scheduled samples, rounded to 60 s; the interval is the mode. It is estimated
per bird (nominal), per bird x year-month (schedule changes) and per bird x year-month x day/night (sun elevation
> 0 at the sample; tags may use a different night schedule). Each sample uses the finest level with
>= MIN_GAPS gaps. Shares of gaps at the interval, shorter, at a multiple (missed bursts) and otherwise longer are
reported in schedule.csv.

Weights (time represented). Scheduled sample i covers [t_i, min(t_{i+1}, t_i + CAP_FACTOR x interval_i)), so a
data gap adds at most half an interval to the sample before it. Coverage = covered / elapsed time.

Scheduled bursts during continuous bouts. Bouts are runs of k == 1 windows less than BOUT_JOIN_S apart, from the
first t_start to the last t_start + its recorded seconds. The code measures whether the schedule continues inside
bouts: per bird-month the clock phase (mode of t mod interval, 5 s bins) gives the expected slots inside each bout;
a slot is found when a scheduled sample starts within SLOT_TOL_S of it. A missing slot means the burst was either
suppressed by the firmware or absorbed into the continuous segment (then its window carries k == 1); the metadata
cannot separate the two, and both are treated alike. Also reported: the share of slots whose covering k == 1
window is K=2 flight (direct evidence for imputing flight).
Two variants are always computed:
- 'plain': scheduled coverage only (bouts ignored).
- 'imputed': a scheduled sample's coverage also ends at the start of the next bout that begins after it, and
  bout time not covered by any scheduled sample is IMPUTED AS FLIGHT ('flight_imputed', its own category in every
  table). If the schedule continues inside bouts, the scheduled samples there are classified by their own state
  and the imputed time is only the part of the bout before the first internal sample.
The headline variant is chosen PER BIRD by --bout-imputation (auto: 'imputed' if the bird's share of found slots
is < SUPPRESSION_FOUND_MAX, else 'plain'; birds with < MIN_SLOTS_BIRD slots follow the pooled share; on/off force
one variant for all birds). Both variants for all birds are reported alongside (d_flight_plain / d_flight_imp).

Behaviour per scheduled sample: K=2 state (state_hmm of --states2), K=6 state (state_hmm of --states6) and the
ground sub-state (substates.parquet label s.j where its GMM 'state' is 0 and the K=2 state is ground; otherwise
'ground_unassigned'). Scheduled k == 0 windows rarely have GPS, so K=6 flight types on them rest on acceleration
alone; flight composition is therefore also given from continuous bouts (K=6 among K=2-flight k == 1 windows,
weighted by recorded seconds = acc_samples / 20) and that composition is CONDITIONAL ON BEING IN A TRIGGERED BOUT.

Days and hours are LOCAL MEAN SOLAR time (offset = local_solar_hour - UTC hour per window; missing values are filled
from the previous/next window, else FALLBACK_OFFSET_H). Covered intervals are split at solar-hour boundaries, so
day and hour budgets add up exactly. Daily hours are 24 x (state seconds / classified covered seconds) of the day;
raw covered hours are kept too. A bird-day enters aggregates if coverage >= --min-coverage (default 0.5;
sensitivity 0.3 / 0.7 reported). Pre-fledge windows of tagged-young birds are dropped before anything else
(as in demography.py) and counted. Cohort: year of the bird's first window <= 2024 -> '2024', else '2025'.

Aggregation (descriptive): per bird and cell (month, group, ...) the mean over its qualifying days (>= MIN_DAYS_CELL);
then the bird-weighted mean over birds with a bootstrap-over-birds 95 % CI. Tests (bird is the unit): per calendar
month the difference in bird means of daily flight hours (bird-months with >= MIN_TEST_DAYS qualifying days, months
with >= MIN_GROUP_BIRDS birds per group); combined statistic = weighted mean of the monthly differences, weight
n1 n0 / (n1 + n0). One permutation of the labels between birds is applied to all months at once (a bird keeps one
label in every month), so the dependence between months of the same bird is kept in the null.

Outputs (--out, new or empty; bird-keyed tables hold no coordinates, offsets or UTC times): birds.csv,
schedule.csv, recording_check.json, birddays.csv, budgets_by_group.csv, diel.csv, window_vs_time.csv,
flight_composition.csv, tests.csv, tests_by_month.csv, coverage_sensitivity.csv, *.png, summary.md, metadata.json.

Usage: python -m aquilacl.budgets --windows W --states2 S2 --states6 S6 --substates SUB --reference CSV --out NEW
       [--min-coverage 0.5] [--n-boot 1000] [--n-perm 5000] [--workers N] [--bout-imputation auto|on|off]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed

from . import reference

WINDOW_S, FS = 10.0, 20.0
BURST_GAP_S = 11.0
BOUT_JOIN_S = 30.0
CAP_FACTOR = 1.5
ROUND_S = 60.0
MIN_GAPS = 20
SLOT_TOL_S = 15.0
PHASE_BIN_S = 5.0
SUPPRESSION_FOUND_MAX = 0.5
FALLBACK_OFFSET_H = -7.0
MIN_DAYS_CELL = 3
MIN_TEST_DAYS = 5
MIN_GROUP_BIRDS = 3
DIEL_MIN_S = 3600.0
COMP_MIN_S = 600.0
MIN_SLOTS_BIRD = 20
SENS_COVERAGE = (0.3, 0.7)
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}
K2_NAMES = {0: "ground", 1: "flight"}
K6_NAMES = {0: "rest", 1: "perched-active", 2: "thermal circling", 3: "fast descending glide", 4: "cruising",
            5: "flapping"}  # stage-6 labels of states_r6_yaw60_bpb4; other state dirs need their own names
SUB_NAMES = {"0.0": "very still (GPS absent, sensitivity flag)", "0.1": "still, rolled", "0.2": "low activity",
             "0.3": "active on the ground"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#888888"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- recording structure (pure functions)

def to_seconds(t) -> np.ndarray:
    t = pd.to_datetime(pd.Series(t), utc=True)
    return ((t - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)


def scheduled_mask(t: np.ndarray, k: np.ndarray) -> np.ndarray:
    """First window of each k == 0 burst (t sorted ascending)."""
    idx = np.flatnonzero(np.asarray(k) == 0)
    m = np.zeros(len(t), bool)
    if len(idx):
        gap = np.diff(t[idx], prepend=-np.inf)
        m[idx[gap > BURST_GAP_S]] = True
    return m


def solar_offset_h(t: np.ndarray, lsh: np.ndarray) -> tuple[np.ndarray, bool]:
    """Local mean solar time minus UTC, hours in [-12, 12); gaps filled from neighbours (t sorted)."""
    off = pd.Series((np.asarray(lsh, float) - (t % 86400) / 3600 + 12) % 24 - 12)
    if off.notna().any():
        return off.ffill().bfill().to_numpy(), False
    return np.full(len(t), FALLBACK_OFFSET_H), True


def interval_mode(gaps) -> tuple[float, int]:
    g = np.asarray(gaps, float)
    g = g[np.isfinite(g) & (g >= ROUND_S / 2)]
    if not len(g):
        return np.nan, 0
    v, c = np.unique(np.round(g / ROUND_S) * ROUND_S, return_counts=True)
    return float(v[np.argmax(c)]), len(g)


def gap_shares(g: np.ndarray, I: float) -> dict:
    g = g[np.isfinite(g)]
    if not len(g) or not np.isfinite(I):
        return {"share_at_interval": np.nan, "share_shorter": np.nan, "share_multiple": np.nan, "share_other_longer": np.nan}
    at = np.abs(g - I) <= ROUND_S / 2
    short = g < I - ROUND_S / 2
    mult = (g > I + ROUND_S / 2) & (np.abs(g - np.round(g / I) * I) <= ROUND_S / 2)
    return {"share_at_interval": at.mean(), "share_shorter": short.mean(), "share_multiple": mult.mean(),
            "share_other_longer": (~(at | short | mult)).mean()}


def schedule(ts: np.ndarray, ym: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, float, pd.DataFrame, dict]:
    """Interval per scheduled sample (finest level with >= MIN_GAPS gaps), nominal interval, table, phases."""
    gaps = np.r_[np.diff(ts), np.nan]
    nominal, n = interval_mode(gaps)
    I = np.full(len(ts), nominal)
    rows = [{"year_month": "all", "period": "all", "n_gaps": n, "interval_s": nominal, **gap_shares(gaps, nominal)}]
    month_I = {}
    for m in np.unique(ym):
        sel = ym == m
        Im, nm = interval_mode(gaps[sel])
        rows.append({"year_month": m, "period": "all", "n_gaps": nm, "interval_s": Im, **gap_shares(gaps[sel], Im)})
        if nm >= MIN_GAPS:
            I[sel] = Im
        month_I[m] = Im if nm >= MIN_GAPS else nominal
        for dn, name in ((True, "day"), (False, "night")):
            s2 = sel & (day == dn)
            Id, nd = interval_mode(gaps[s2])
            rows.append({"year_month": m, "period": name, "n_gaps": nd, "interval_s": Id, **gap_shares(gaps[s2], Id)})
            if nd >= MIN_GAPS:
                I[s2] = Id
    I = np.where(np.isfinite(I), I, 300.0)
    phases = {m: clock_phase(ts[ym == m], month_I[m]) for m in month_I}
    return I, nominal, pd.DataFrame(rows), {m: (month_I[m], phases[m]) for m in month_I}


def clock_phase(ts: np.ndarray, I: float) -> float:
    if not len(ts) or not np.isfinite(I) or I <= 0:
        return np.nan
    ph = (np.round((ts % I) / PHASE_BIN_S) * PHASE_BIN_S) % I
    v, c = np.unique(ph, return_counts=True)
    return float(v[np.argmax(c)])


def bouts_from_windows(t: np.ndarray, k: np.ndarray, acc_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs of k == 1 windows (t sorted) less than BOUT_JOIN_S apart: (start, end, bout id per window or -1)."""
    idx = np.flatnonzero(np.asarray(k) == 1)
    bid = np.full(len(t), -1)
    if not len(idx):
        return np.zeros(0), np.zeros(0), bid
    new = np.diff(t[idx], prepend=-np.inf) >= BOUT_JOIN_S
    b = np.cumsum(new) - 1
    bid[idx] = b
    start = t[idx][new]
    last = np.r_[np.flatnonzero(new)[1:] - 1, len(idx) - 1]
    end = t[idx][last] + np.maximum(np.asarray(acc_samples)[idx][last] / FS, 1.0)
    end = np.maximum(end, start + 1.0)
    return start, end, bid


def coverage_ends(ts: np.ndarray, I: np.ndarray, bout_start: np.ndarray, impute: bool) -> np.ndarray:
    e = np.minimum(np.r_[ts[1:], np.inf], ts + CAP_FACTOR * I)
    if impute and len(bout_start):
        j = np.searchsorted(bout_start, ts, side="right")
        nb = np.where(j < len(bout_start), bout_start[np.minimum(j, len(bout_start) - 1)], np.inf)
        e = np.minimum(e, nb)
    return e


def uncovered_pieces(s: np.ndarray, e: np.ndarray, bs: np.ndarray, be: np.ndarray):
    """Parts of each bout [bs, be) not covered by the sorted, disjoint intervals [s, e): (start, end, bout)."""
    ps, pe, pb = [], [], []
    for j, (a, b) in enumerate(zip(bs, be)):
        cur = a
        for i in range(np.searchsorted(e, a, "right"), np.searchsorted(s, b, "left")):
            if s[i] > cur:
                ps.append(cur), pe.append(min(s[i], b)), pb.append(j)
            cur = max(cur, e[i])
            if cur >= b:
                break
        if cur < b:
            ps.append(cur), pe.append(b), pb.append(j)
    return np.array(ps, float), np.array(pe, float), np.array(pb, int)


def split_bins(start: np.ndarray, end: np.ndarray, bin_s: float):
    """Split [start, end) at multiples of bin_s: (source index, bin index, seconds), seconds > 0 only."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    b0 = np.floor(start / bin_s).astype(np.int64)
    b1 = np.floor(np.nextafter(end, -np.inf) / bin_s).astype(np.int64)
    n = np.where(end > start, b1 - b0 + 1, 0)
    src = np.repeat(np.arange(len(start)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    b = b0[src] + k
    sec = np.minimum(end[src], (b + 1) * bin_s) - np.maximum(start[src], b * bin_s)
    ok = sec > 0
    return src[ok], b[ok], sec[ok]


def hourly(start_loc: np.ndarray, end_loc: np.ndarray, cats: dict[str, np.ndarray], n_start: np.ndarray) -> pd.DataFrame:
    """Seconds per solar hour bin and category system (columns '<system>_<label>'); n_sched counts sample starts."""
    src, b, sec = split_bins(start_loc, end_loc, 3600.0)
    parts = []
    for name, lab in cats.items():
        t = pd.DataFrame({"hb": b, "c": np.asarray(lab, object)[src], "s": sec}).groupby(["hb", "c"])["s"].sum().unstack(fill_value=0.0)
        t.columns = [f"{name}_{c}" for c in t.columns]
        parts.append(t)
    H = pd.concat(parts, axis=1).fillna(0.0) if parts and len(b) else pd.DataFrame(index=pd.Index([], name="hb"))
    hb0 = np.floor(start_loc[n_start] / 3600.0).astype(np.int64)
    H["n_sched"] = pd.Series(hb0).value_counts().reindex(H.index).fillna(0).to_numpy()
    H.index.name = "hb"
    return H


# ---------------------------------------------------------------- per-bird pass

def _read(path: Path, want: list[str]) -> pd.DataFrame:
    have = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in want if c in have])


def _state_arr(st: pd.DataFrame | None, n: int, col: str) -> np.ndarray:
    a = np.full(n, -1)
    if st is not None and len(st):
        r = st["row"].to_numpy()
        if r.min() < 0 or r.max() >= n:
            raise ValueError("state row outside the bird's windows")
        a[r] = st[col].to_numpy()
    return a


def bird_pass(d: Path, info: dict, st2, st6, sub, K6: int) -> dict | None:
    m = _read(d / "meta.parquet", ["t_start", "sensitivity_k", "acc_samples", "gps_fixes", "local_solar_hour",
                                   "sun_elev_deg", "k_in_segment", "segment"])
    n_all = len(m)
    if n_all == 0:
        return None
    f = _read(d / "features.parquet", ["speed_mean"])
    t_all = to_seconds(m["t_start"])
    s2, s2g = _state_arr(st2, n_all, "state_hmm"), _state_arr(st2, n_all, "state")
    s6 = _state_arr(st6, n_all, "state_hmm")
    sub_lab = np.full(n_all, None, object)
    sub_state = np.full(n_all, -1)
    if sub is not None and len(sub):
        sub_lab[sub["row"].to_numpy()] = sub["label"].astype(str).to_numpy()
        sub_state[sub["row"].to_numpy()] = sub["state"].to_numpy()
    res = {"bird": d.name, "n_windows": n_all, "first_year": int(pd.Timestamp(t_all.min(), unit="s").year),
           "n_pre_fledge": 0, "fledge": None}
    keep = np.ones(n_all, bool)
    if info.get("young"):
        speed = f["speed_mean"].to_numpy(float) if "speed_mean" in f else np.full(n_all, np.nan)
        fl = reference.fledge_from_speed(pd.Series(pd.to_datetime(t_all, unit="s", utc=True)), speed, info["hatch_year"])
        res["fledge"] = fl["fledge"]
        keep = t_all >= (fl["fledge"] - pd.Timestamp(0, tz="UTC")).total_seconds()
        res["n_pre_fledge"] = int((~keep).sum())
    order = np.flatnonzero(keep)[np.argsort(t_all[keep], kind="stable")]
    t = t_all[order]
    k = m["sensitivity_k"].to_numpy()[order]
    acc = m["acc_samples"].to_numpy()[order]
    kin = m["k_in_segment"].to_numpy()[order]
    seg = m["segment"].to_numpy()[order]
    sun = m["sun_elev_deg"].to_numpy(float)[order]
    c2, c2g, c6, slab, sst = s2[order], s2g[order], s6[order], sub_lab[order], sub_state[order]
    res["n_kept"] = len(t)
    if not len(t):
        return res
    off, fallback = solar_offset_h(t, m["local_solar_hour"].to_numpy(float)[order])
    res["offset_fallback"] = fallback

    # labels of every kept window
    lab2 = np.array([K2_NAMES.get(v, "unknown") for v in c2], object)
    lab6 = np.where(c6 >= 0, c6.astype(str), "unknown").astype(object)
    labs = np.where(lab2 == "ground", np.where((sst == 0) & (slab != None), slab, "ground_unassigned"), lab2)  # noqa: E711

    sched = scheduled_mask(t, k)
    si = np.flatnonzero(sched)
    ts = t[si]
    k0 = k == 0
    seg_has_k1 = pd.Series(k == 1).groupby(seg).transform("any").to_numpy()
    bs, be, bid = bouts_from_windows(t, k, acc)
    jb = np.searchsorted(bs, ts, "right") - 1
    in_bout = (jb >= 0) & (ts < be[np.maximum(jb, 0)]) if len(bs) else np.zeros(len(ts), bool)
    res["check"] = {"k0_windows": int(k0.sum()), "scheduled": len(si),
                    "k0_first_by_segment": int((k0 & (kin == 0)).sum()),
                    "sched_with_k_in_segment_gt0": int((kin[si] > 0).sum()),
                    "k0_kin0_not_sched": int((k0 & (kin == 0) & ~sched).sum()),
                    "sched_in_segment_with_k1": int(seg_has_k1[si].sum()),
                    "sched_inside_bout_span": int(in_bout.sum()),
                    "sched_with_gps": int((m["gps_fixes"].to_numpy()[order][si] > 0).sum()),
                    "sched_k2_state_vs_hmm_agree": int((c2[si] == c2g[si]).sum()),
                    "sched_k2_known": int((c2[si] >= 0).sum()),
                    "bouts": len(bs), "bout_seconds": float((be - bs).sum())}
    res["counts"] = {sysn: {"all": pd.Series(L).value_counts().to_dict(), "sched": pd.Series(L[si]).value_counts().to_dict()}
                     for sysn, L in (("k2", lab2), ("k6", lab6), ("sub", labs))}
    if not len(si):
        return res
    ym = pd.to_datetime(ts, unit="s").strftime("%Y-%m").to_numpy()
    I, nominal, sch, month_phase = schedule(ts, ym, sun[si] > 0)
    res["nominal_interval_s"] = nominal
    res["schedule"] = sch.assign(bird=d.name)

    # slots of the clock inside bouts
    n_slot = n_found = n_flight = n_win = 0
    bym = pd.to_datetime(bs, unit="s").strftime("%Y-%m").to_numpy() if len(bs) else np.zeros(0, object)
    k1 = np.flatnonzero(k == 1)
    for a, b, mm in zip(bs, be, bym):
        Im, ph = month_phase.get(mm, (nominal, clock_phase(ts, nominal)))
        if not (np.isfinite(Im) and np.isfinite(ph)):
            continue
        q0, q1 = np.ceil((a + SLOT_TOL_S - ph) / Im), np.floor((b - SLOT_TOL_S - ph) / Im)
        if q1 < q0:
            continue
        slots = ph + Im * np.arange(q0, q1 + 1)
        j = np.searchsorted(ts, slots)
        near = np.minimum(np.abs(ts[np.clip(j, 0, len(ts) - 1)] - slots), np.abs(ts[np.clip(j - 1, 0, len(ts) - 1)] - slots))
        n_slot += len(slots)
        n_found += int((near <= SLOT_TOL_S).sum())
        w = np.searchsorted(t[k1], slots, "right") - 1
        okw = (w >= 0) & (slots < t[k1][np.maximum(w, 0)] + WINDOW_S)
        n_win += int(okw.sum())
        n_flight += int((c2[k1][w[okw]] == 1).sum())
    res["slots"] = {"slots": n_slot, "found": n_found, "slot_in_k1_window": n_win, "slot_window_flight": n_flight}

    # timelines; behaviour of each scheduled sample
    off_s = off[si] * 3600.0
    boff = off[np.searchsorted(t, bs)] * 3600.0 if len(bs) else np.zeros(0)
    res["hourly"], res["span"] = {}, {}
    plain_e = coverage_ends(ts, I, bs, impute=False)
    for variant in ("plain", "imputed"):
        e = coverage_ends(ts, I, bs, impute=variant == "imputed")
        if variant == "imputed":
            ps, pe, pb = uncovered_pieces(ts, e, bs, be)
        else:
            ps, pe, pb = np.zeros(0), np.zeros(0), np.zeros(0, int)
        start_loc = np.r_[ts + off_s, ps + boff[pb] if len(pb) else np.zeros(0)]
        end_loc = np.r_[e + off_s, pe + boff[pb] if len(pb) else np.zeros(0)]
        imp = np.full(len(pb), "flight_imputed", object)
        cats = {"k2": np.r_[lab2[si], imp], "k6": np.r_[lab6[si], imp], "sub": np.r_[labs[si], imp]}
        is_start = np.r_[np.ones(len(ts), bool), np.zeros(len(pb), bool)]
        res["hourly"][variant] = hourly(start_loc, end_loc, cats, is_start)
        span = max(e.max(), pe.max() if len(pe) else -np.inf) - ts[0]
        res["span"][variant] = {"span_s": float(span), "sched_s": float((e - ts).sum()), "imputed_s": float((pe - ps).sum())}
    # K2 x K6 time of scheduled samples (plain weights) and bout flight composition
    w = plain_e - ts
    res["k2k6_sched"] = pd.DataFrame({"k2": lab2[si], "k6": lab6[si], "s": w}).groupby(["k2", "k6"])["s"].sum()
    fl = (k == 1) & (c2 == 1)
    res["bout_comp"] = pd.DataFrame({"month": pd.to_datetime(t[fl], unit="s").month, "k6": lab6[fl], "s": acc[fl] / FS}) \
        .groupby(["month", "k6"])["s"].sum()
    res["bout_k2"] = pd.Series(lab2[k == 1]).value_counts().to_dict()
    return res


# ---------------------------------------------------------------- aggregation helpers

def boot_ci(V: np.ndarray, n_boot: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bird-weighted mean (rows = birds) and percentile 95 % CI from resampling birds."""
    V = np.asarray(V, float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(V, 0)
        if len(V) < 2 or n_boot <= 0:
            return mean, np.full(V.shape[1], np.nan), np.full(V.shape[1], np.nan)
        B = np.nanmean(V[rng.integers(0, len(V), (n_boot, len(V)))], 1)
        return mean, np.nanpercentile(B, 2.5, 0), np.nanpercentile(B, 97.5, 0)


def day_vars(D: pd.DataFrame) -> list[str]:
    return [c for c in D.columns if c.startswith("d_")] + ["coverage", "coverage_day", "coverage_night"]


def aggregate(D: pd.DataFrame, group_col: str | None, n_boot: int, rng, by_month: bool = True) -> pd.DataFrame:
    vars_ = day_vars(D)
    rows = []
    gcol = group_col or "_all"
    X = D.assign(_all="all")
    X = X[X[gcol].notna()]
    for mon in ([0] + sorted(X["month"].unique().tolist()) if by_month else [0]):
        Y = X if mon == 0 else X[X["month"] == mon]
        g = Y.groupby([gcol, "bird"])
        M, n = g[vars_].mean(), g.size()
        M, n = M[n >= MIN_DAYS_CELL], n[n >= MIN_DAYS_CELL]
        for grp in M.index.get_level_values(0).unique():
            V = M.loc[grp]
            mean, lo, hi = boot_ci(V.to_numpy(), n_boot, rng)
            for v, a, b, c in zip(vars_, mean, lo, hi):
                rows.append({"grouping": group_col or "all", "group": grp, "month": int(mon), "variable": v,
                             "n_birds": len(V), "n_bird_days": int(n.loc[grp].sum()), "mean": a, "ci_lo": b, "ci_hi": c})
    return pd.DataFrame(rows)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))) if sp > 0 else np.nan


def permute_labels(lab: np.ndarray, n_perm: int, rng, strata: np.ndarray | None = None) -> np.ndarray:
    out = np.tile(lab, (n_perm, 1))
    groups = [np.arange(len(lab))] if strata is None else [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for ix in groups:
        for p in range(n_perm):
            out[p, ix] = lab[rng.permutation(ix)]
    return out


def month_combined_test(V: pd.DataFrame, lab: pd.Series, n_perm: int, rng, strata: pd.Series | None = None):
    """V: birds x months (NaN = no qualifying bird-month); lab: bird -> 0/1. Difference = group 1 minus group 0.

    One permutation of labels between birds applies to all months. Returns (summary dict, per-month DataFrame).
    """
    birds = V.index[V.index.isin(lab.dropna().index)]
    V = V.loc[birds]
    L = lab.loc[birds].astype(int).to_numpy()
    P = V.notna().to_numpy(float)
    X = np.nan_to_num(V.to_numpy(float))

    def stats_(Lm):  # Lm: (p, B)
        n1, n0 = Lm @ P, (1 - Lm) @ P
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (Lm @ X) / n1 - ((1 - Lm) @ X) / n0
            w = np.where((n1 >= MIN_GROUP_BIRDS) & (n0 >= MIN_GROUP_BIRDS), n1 * n0 / (n1 + n0), 0.0)
            comb = np.nansum(np.where(w > 0, w * d, 0), 1) / w.sum(axis=1)
        return d, w, comb

    d, w, comb = stats_(L[None, :].astype(float))
    st = None if strata is None else strata.loc[birds].to_numpy()
    Lp = permute_labels(L, n_perm, rng, st).astype(float)
    dn, wn, cn = stats_(Lp)
    ok = np.isfinite(cn)
    p = float((1 + (np.abs(cn[ok]) >= abs(comb[0]) - 1e-12).sum()) / (1 + ok.sum())) if np.isfinite(comb[0]) else np.nan
    rows = []
    for j, mon in enumerate(V.columns):
        a = V.iloc[:, j][L == 1].dropna().to_numpy()
        b = V.iloc[:, j][L == 0].dropna().to_numpy()
        okj = wn[:, j] > 0
        pj = float((1 + (np.abs(dn[okj, j]) >= abs(d[0, j]) - 1e-12).sum()) / (1 + okj.sum())) if w[0, j] > 0 else np.nan
        rows.append({"month": int(mon), "n1": len(a), "n0": len(b), "mean1": a.mean() if len(a) else np.nan,
                     "mean0": b.mean() if len(b) else np.nan, "diff": d[0, j], "hedges_g": hedges_g(a, b),
                     "in_combined": bool(w[0, j] > 0), "weight": w[0, j], "p_month_descriptive": pj})
    per = pd.DataFrame(rows)
    used = per[per.in_combined]
    summ = {"n_birds_1": int((L == 1).sum()), "n_birds_0": int((L == 0).sum()), "months_used": int(len(used)),
            "combined_diff_h": float(comb[0]), "median_monthly_hedges_g": float(used.hedges_g.median()) if len(used) else np.nan,
            "p_perm": p, "n_perm": int(ok.sum())}
    return summ, per


def run_tests(D: pd.DataFrame, B: pd.DataFrame, var: str, n_perm: int, rng) -> tuple[list[dict], list[pd.DataFrame]]:
    bm = D.groupby(["bird", "month"])[var].agg(["mean", "size"])
    V = bm["mean"].where(bm["size"] >= MIN_TEST_DAYS).unstack()
    c24 = B.index[B.cohort == "2024"]
    specs = [
        ("sex", "female minus male", B.sex.map({"f": 1, "m": 0}), None, None),
        ("sex_within_cohort", "female minus male, labels permuted within cohort", B.sex.map({"f": 1, "m": 0}), B.cohort, None),
        ("migrant_cohort2024", "migrant-flagged minus not flagged, cohort 2024", B.migrant_flag.astype(float), None, c24),
        ("young_cohort2024", "young-tagged minus not stated, cohort 2024 (pre-fledge days excluded)",
         B.young_flag.astype(float), None, c24),
    ]
    summ, per = [], []
    for name, desc, lab, strata, subset in specs:
        lab = lab.dropna()
        if subset is not None:
            lab = lab[lab.index.isin(subset)]
        Vs = V[V.index.isin(lab.index)]
        if lab.nunique() < 2 or not len(Vs):
            continue
        s, p = month_combined_test(Vs, lab, n_perm, rng, strata)
        summ.append({"test": name, "description": desc, "variable": var, **s})
        per.append(p.assign(test=name, variable=var))
    return summ, per


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                         "font.size": 8})
    return plt


def _line(ax, A: pd.DataFrame, var: str, label: str, color: str, x="month"):
    s = A[A.variable == var].sort_values(x)
    s = s[s[x] > 0] if x == "month" else s
    if not len(s):
        return
    ax.plot(s[x], s["mean"], "-o", color=color, lw=2, ms=4, label=label)
    ax.fill_between(s[x], s["ci_lo"], s["ci_hi"], color=color, alpha=0.18, lw=0)


def figures(out: Path, agg: pd.DataFrame, diel: pd.DataFrame, wvt: pd.DataFrame, D: pd.DataFrame, min_cov: float,
            comp: pd.DataFrame, K6: int, sub_labels: list[str]) -> None:
    plt = _plt()
    A = agg[agg.grouping == "all"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    _line(ax, A, "d_flight", "flight (headline, per-bird variant)", COLORS[0])
    _line(ax, A, "d_ground", "ground", COLORS[1])
    s = A[(A.variable == "d_flight_plain") & (A.month > 0)].sort_values("month")
    ax.plot(s.month, s["mean"], "--", color=COLORS[0], lw=1, label="flight, plain variant for all birds")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("calendar month")
    ax.set_ylabel("hours per day (bird mean, 95% CI over birds)")
    ax.legend(frameon=False)
    fig.tight_layout(), fig.savefig(out / "monthly_budget.png", dpi=130), plt.close(fig)

    for sysn, labs, names, fname in (("k6", [str(i) for i in range(K6)] + ["flight_imputed"],
                                      {**{str(k): v for k, v in K6_NAMES.items()}, "flight_imputed": "flight imputed (bout)"},
                                      "monthly_k6.png"),
                                     ("sub", sub_labels + ["ground_unassigned", "flight", "flight_imputed"],
                                      {**SUB_NAMES, "flight_imputed": "flight imputed (bout)"}, "monthly_substates.png")):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        bottom = np.zeros(12)
        for i, lb in enumerate(labs):
            s = A[(A.variable == f"d_{sysn}_{lb}") & (A.month > 0)].set_index("month")["mean"].reindex(range(1, 13)).fillna(0).to_numpy()
            ax.bar(range(1, 13), s, bottom=bottom, color=COLORS[i % len(COLORS)], edgecolor="white", lw=1,
                   label=f"{lb} {names.get(lb, '')}".strip())
            bottom += s
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("calendar month")
        ax.set_ylabel("hours per day (bird means)")
        ax.legend(frameon=False, fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout(), fig.savefig(out / fname, dpi=130), plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, sea in enumerate(["DJF", "MAM", "JJA", "SON"]):
        _line(ax, diel[diel.season == sea].rename(columns={"hour": "h"}), "share_flight", sea, COLORS[i], x="h")
    ax.set_xlabel("local mean solar hour")
    ax.set_ylabel("share of covered time in flight")
    ax.legend(frameon=False)
    fig.tight_layout(), fig.savefig(out / "diel.png", dpi=130), plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, (gcol, title) in zip(axes.ravel(), (("sex", "sex"), ("cohort", "cohort"), ("migrant_c2024", "migrant flag, cohort 2024"),
                                                ("young_c2024", "young-tagged, cohort 2024"))):
        G = agg[agg.grouping == gcol]
        for i, g in enumerate(sorted(G.group.unique(), key=str)):
            _line(ax, G[G.group == g], "d_flight", f"{g} (n birds max {G[G.group == g].n_birds.max()})", COLORS[i])
        ax.set_title(title)
        ax.set_xticks(range(1, 13))
        ax.legend(frameon=False, fontsize=6)
    axes[0, 0].set_ylabel("flight hours per day")
    axes[1, 0].set_ylabel("flight hours per day")
    fig.tight_layout(), fig.savefig(out / "groups_flight.png", dpi=130), plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 3, 3]})
    for ax, sysn in zip(axes, ("k2", "k6", "sub")):
        W = wvt[wvt.system == sysn]
        x = np.arange(len(W))
        for i, (col, lab) in enumerate((("window_share_all", "window count, all windows"),
                                        ("window_share_scheduled", "window count, scheduled only"),
                                        ("time_share", "time share"))):
            ax.bar(x + (i - 1) * 0.27, W[col], width=0.25, color=COLORS[i], label=lab)
        ax.set_xticks(x, W.state, rotation=60, fontsize=6)
        ax.set_title(sysn)
    axes[0].set_ylabel("share (bird mean)")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(), fig.savefig(out / "window_vs_time.png", dpi=130), plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(D["coverage"].clip(0, 1), bins=40, color=COLORS[0])
    for thr in (0.3, min_cov, 0.7):
        ax.axvline(thr, color="#333333", lw=1, ls="-" if thr == min_cov else ":")
    ax.set_xlabel("bird-day coverage (share of 24 h)")
    ax.set_ylabel("bird-days")
    fig.tight_layout(), fig.savefig(out / "coverage.png", dpi=130), plt.close(fig)

    C = comp[comp.month == 0]
    if len(C):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(K6)
        for i, src in enumerate(("continuous bouts (conditional on a triggered bout)", "scheduled flight samples (ACC-only)")):
            s = C[C.source == src].set_index("k6")["mean"].reindex([str(j) for j in range(K6)])
            ax.bar(x + (i - 0.5) * 0.38, s.to_numpy(), width=0.36, color=COLORS[i], label=src)
        ax.set_xticks(x, [K6_NAMES.get(j, str(j)) for j in range(K6)], rotation=30, fontsize=7)
        ax.set_ylabel("share of flight time (bird mean)")
        ax.legend(frameon=False, fontsize=6)
        fig.tight_layout(), fig.savefig(out / "flight_composition.png", dpi=130), plt.close(fig)


# ---------------------------------------------------------------- main run

def day_table(P: list[dict], variant: dict[str, str] | str) -> pd.DataFrame:
    """Bird-day rows from the per-bird hourly tables: raw covered hours (h_*), 24-h scaled hours (d_*), coverage
    (also coverage_day = solar hours 06-18 covered / 12 h, coverage_night = the rest). ``variant``: one variant name
    or bird -> variant."""
    parts = []
    for p in P:
        v = variant if isinstance(variant, str) else variant[p["bird"]]
        H = p.get("hourly", {}).get(v)
        if H is None or not len(H):
            continue
        Dd = H.groupby(H.index // 24).sum()
        k2 = H[[c for c in H.columns if c.startswith("k2_")]].sum(axis=1)
        dayh = ((H.index % 24) >= 6) & ((H.index % 24) < 18)
        Dd["_cov_day"] = k2[dayh].groupby(H.index[dayh] // 24).sum().reindex(Dd.index).fillna(0) / (12 * 3600)
        Dd["_cov_night"] = k2[~dayh].groupby(H.index[~dayh] // 24).sum().reindex(Dd.index).fillna(0) / (12 * 3600)
        Dd.index.name = "dayn"
        parts.append(Dd.reset_index().assign(bird=p["bird"], variant=v))
    D = pd.concat(parts, ignore_index=True).fillna(0.0)
    sec = {c: D[c] for c in D.columns if c[:3] in ("k2_", "k6_") or c.startswith("sub_")}
    out = pd.DataFrame({"bird": D["bird"], "dayn": D["dayn"].astype(int), "variant": D["variant"],
                        "n_sched": D["n_sched"].astype(int)})
    k2cols = [c for c in sec if c.startswith("k2_")]
    tot = sum(sec[c] for c in k2cols)
    imp = sec.get("k2_flight_imputed", 0.0)
    out["coverage"] = tot / 86400.0
    out["coverage_sched"] = (tot - imp) / 86400.0
    out["coverage_day"], out["coverage_night"] = D["_cov_day"], D["_cov_night"]
    for c, v in sec.items():
        out["h_" + c] = v / 3600.0
    for sysn in ("k2", "k6", "sub"):
        cols = [c for c in sec if c.startswith(sysn + "_") and c != f"{sysn}_unknown"]
        cls = sum(sec[c] for c in cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            for c in cols:
                out["d_" + c] = np.where(cls > 0, 24.0 * sec[c] / cls, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cls2 = tot - sec.get("k2_unknown", 0.0)
        out["d_flight"] = np.where(cls2 > 0, 24.0 * (sec.get("k2_flight", 0.0) + imp) / cls2, np.nan)
        out["d_ground"] = np.where(cls2 > 0, 24.0 * sec.get("k2_ground", 0.0) / cls2, np.nan)
    date = pd.to_datetime(out["dayn"] * 86400, unit="s")
    out.insert(1, "solar_date", date.dt.strftime("%Y-%m-%d"))
    out["month"] = date.dt.month
    out["year"] = date.dt.year
    return out


def run(windows: Path, states2: Path, states6: Path, substates: Path, ref_csv: Path, out: Path, min_coverage: float = 0.5,
        n_boot: int = 1000, n_perm: int = 5000, workers: int = 1, bout_imputation: str = "auto", seed: int = 0) -> dict:
    t0 = time.time()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    rng = np.random.default_rng(seed)
    dirs = [d for d in sorted(p for p in windows.iterdir() if p.is_dir())
            if (d / "meta.parquet").exists() and pq.ParquetFile(d / "meta.parquet").metadata.num_rows > 0]
    birds = [d.name for d in dirs]
    ref = reference.load_birds(ref_csv, birds).set_index("bird")
    log(f"{len(birds)} birds with windows")

    def split(path, cols):
        st = pd.read_parquet(path, columns=cols)
        st["bird"] = st["bird"].astype(str)
        return dict(tuple(st.groupby("bird", sort=False))), st

    s2, st = split(states2 / "states.parquet", ["bird", "row", "state", "state_hmm"])
    s6, st6 = split(states6 / "states.parquet", ["bird", "row", "state_hmm"])
    K6 = int(st6["state_hmm"].max()) + 1
    sb, stb = split(substates / "substates.parquet", ["bird", "row", "state", "label"])
    sub_labels = sorted(stb.loc[stb["state"] == 0, "label"].astype(str).unique().tolist())
    del st, st6, stb
    info = {b: {"young": pd.notna(ref.loc[b, "tagged_young"]) and pd.notna(ref.loc[b, "hatch_year"]),
                "hatch_year": ref.loc[b, "hatch_year"]} for b in birds}
    log(f"pass over birds (K6={K6}, ground sub-states {sub_labels})")
    res = Parallel(n_jobs=workers)(delayed(bird_pass)(d, info[d.name], s2.pop(d.name, None), s6.pop(d.name, None),
                                                      sb.pop(d.name, None), K6) for d in dirs)
    P = [p for p in res if p is not None and p.get("n_kept", 0) > 0 and "hourly" in p]
    log(f"pass done ({time.time() - t0:.0f} s); {len(P)} birds with scheduled samples")
    out.mkdir(parents=True, exist_ok=True)

    # recording check and imputation decision
    chk = pd.DataFrame([p["check"] for p in P], index=[p["bird"] for p in P])
    sl = pd.DataFrame([p["slots"] for p in P], index=[p["bird"] for p in P])
    found_share = sl.found.sum() / max(sl.slots.sum(), 1)
    per_bird_found = (sl.found / sl.slots.where(sl.slots > 0))
    suppressed = found_share < SUPPRESSION_FOUND_MAX
    # per bird: its own found share decides (auto); birds with < MIN_SLOTS_BIRD slots follow the pooled decision
    own = per_bird_found.where(sl.slots >= MIN_SLOTS_BIRD)
    bird_supp = own.lt(SUPPRESSION_FOUND_MAX).where(own.notna(), suppressed)
    if bout_imputation != "auto":
        bird_supp[:] = bout_imputation == "on"
    vmap = {b: "imputed" if bird_supp.get(b, suppressed) else "plain" for b in [p["bird"] for p in P]}
    variant = "per bird" if len(set(vmap.values())) > 1 else next(iter(vmap.values()))
    bout_k2 = pd.DataFrame([p["bout_k2"] for p in P]).sum()
    check = {
        "scheduled_samples": int(chk.scheduled.sum()), "k0_windows": int(chk.k0_windows.sum()),
        "scheduled_with_k_in_segment_gt0": int(chk.sched_with_k_in_segment_gt0.sum()),
        "k0_k_in_segment0_not_scheduled": int(chk.k0_kin0_not_sched.sum()),
        "scheduled_in_segment_with_k1_windows": int(chk.sched_in_segment_with_k1.sum()),
        "scheduled_inside_bout_span": int(chk.sched_inside_bout_span.sum()),
        "scheduled_with_gps_share": float(chk.sched_with_gps.sum() / max(chk.scheduled.sum(), 1)),
        "k2_state_hmm_agrees_with_state_on_scheduled": float(chk.sched_k2_state_vs_hmm_agree.sum() / max(chk.sched_k2_known.sum(), 1)),
        "bouts": int(chk.bouts.sum()), "bout_hours": float(chk.bout_seconds.sum() / 3600),
        "clock_slots_inside_bouts": int(sl.slots.sum()), "slots_with_scheduled_sample": int(sl.found.sum()),
        "share_slots_found_pooled": float(found_share), "share_slots_found_per_bird_median": float(per_bird_found.median()),
        "slots_in_k1_window": int(sl.slot_in_k1_window.sum()),
        "share_slot_windows_flight": float(sl.slot_window_flight.sum() / max(sl.slot_in_k1_window.sum(), 1)),
        "k1_windows_k2_share": {k: float(v / bout_k2.sum()) for k, v in bout_k2.items()},
        "schedule_suppressed_in_bouts": bool(suppressed), "bout_imputation_mode": bout_imputation, "headline_variant": variant,
        "birds_imputed": int(sum(v == "imputed" for v in vmap.values())),
        "birds_plain": int(sum(v == "plain" for v in vmap.values())),
        "birds_decided_by_pooled_rule": int(own.isna().sum()),
    }
    (out / "recording_check.json").write_text(json.dumps(check, indent=1))
    log(f"recording check: {check['share_slots_found_pooled']:.3f} of clock slots inside bouts have a scheduled sample; "
        f"{check['birds_imputed']} birds use the imputed variant, {check['birds_plain']} the plain variant")

    sch = pd.concat([p["schedule"] for p in P if "schedule" in p], ignore_index=True)
    nominal = pd.Series({p["bird"]: p.get("nominal_interval_s", np.nan) for p in P})
    sch["nominal_interval_s"] = sch.bird.map(nominal)
    sch["differs_from_nominal"] = (sch.interval_s != sch.nominal_interval_s) & (sch.n_gaps >= MIN_GAPS)
    sch.to_csv(out / "schedule.csv", index=False)

    # bird table
    B = ref.loc[[p["bird"] for p in P]].copy()
    B["cohort"] = ["2024" if p["first_year"] <= 2024 else "2025" for p in P]
    B["migrant_flag"] = B.migrant.eq(True).astype(int)
    B["young_flag"] = B.tagged_young.eq(True).astype(int)
    B["fledge"] = [p["fledge"] for p in P]
    B["n_windows"] = [p["n_windows"] for p in P]
    B["n_pre_fledge"] = [p["n_pre_fledge"] for p in P]
    B["n_scheduled"] = chk.scheduled.reindex(B.index).to_numpy()
    B["nominal_interval_s"] = nominal.reindex(B.index).to_numpy()
    B["offset_fallback"] = [p.get("offset_fallback") for p in P]
    for v in ("plain", "imputed"):
        sp = pd.DataFrame([p["span"][v] for p in P], index=B.index)
        B[f"share_covered_scheduled_{v}"] = sp.sched_s / sp.span_s
        B[f"share_imputed_{v}"] = sp.imputed_s / sp.span_s
        B[f"share_gap_{v}"] = 1 - (sp.sched_s + sp.imputed_s) / sp.span_s
    B["share_slots_found"] = per_bird_found.reindex(B.index).to_numpy()
    B["clock_slots_in_bouts"] = sl.slots.reindex(B.index).to_numpy()
    B["variant"] = [vmap[b] for b in B.index]
    sel = {b: vmap[b] for b in B.index}
    for c in ("share_covered_scheduled", "share_imputed", "share_gap"):
        B[f"{c}_headline"] = [B.loc[b, f"{c}_{sel[b]}"] for b in B.index]
    mdiff = sch[(sch.period == "all") & (sch.year_month != "all")].groupby("bird").differs_from_nominal.sum()
    B["months_schedule_differs"] = mdiff.reindex(B.index).fillna(0).astype(int).to_numpy()
    B.drop(columns=[c for c in B.columns if "comment" in c], errors="ignore").reset_index().to_csv(out / "birds.csv", index=False)

    # bird-day table (headline variant, plus the other variant's flight and coverage)
    D = day_table(P, vmap)
    idx = pd.MultiIndex.from_frame(D[["bird", "dayn"]])
    for v, tag in (("plain", "plain"), ("imputed", "imp")):
        X = day_table(P, v).set_index(["bird", "dayn"])
        D[f"d_flight_{tag}"] = X["d_flight"].reindex(idx).to_numpy()
        D[f"coverage_{tag}"] = X["coverage"].reindex(idx).to_numpy()
    imp_cols = [c for c in D.columns if c.endswith("_flight_imputed")]
    D[imp_cols] = D[imp_cols].fillna(0.0)
    for c in ("sex", "cohort", "migrant_flag", "young_flag"):
        D[c] = D.bird.map(B[c])
    D["migrant_all"] = np.where(D.migrant_flag == 1, "migrant", "not_flagged")
    D["migrant_c2024"] = np.where(D.cohort == "2024", np.where(D.migrant_flag == 1, "migrant", "not_flagged"), None)
    D["young_c2024"] = np.where(D.cohort == "2024", np.where(D.young_flag == 1, "young_tagged", "not_stated"), None)
    D["age_class"] = "not_stated"
    for b in B.index[B.young_flag == 1]:
        sel = (D.bird == b).to_numpy()
        a = reference.age_on(B.loc[b], pd.to_datetime(D.loc[sel, "dayn"] * 86400 + 43200, unit="s", utc=True), B.loc[b, "fledge"])
        D.loc[sel, "age_class"] = a["age_class"].to_numpy()
    n_prefledge_days = int((D.age_class == "pre_fledge").sum())
    D = D[D.age_class != "pre_fledge"].reset_index(drop=True)
    D.drop(columns=["dayn"]).to_csv(out / "birddays.csv", index=False)
    Q = D[D.coverage >= min_coverage].copy()
    log(f"{len(D):,} bird-days, {len(Q):,} with coverage >= {min_coverage}")

    # aggregates
    agg = pd.concat([aggregate(Q, g, n_boot, rng) for g in
                     (None, "sex", "age_class", "migrant_all", "migrant_c2024", "young_c2024", "cohort")], ignore_index=True)
    agg.to_csv(out / "budgets_by_group.csv", index=False)

    # diel (qualifying days only)
    drows = []
    qd = set(zip(Q.bird, Q.dayn))
    hparts = []
    for p in P:
        H = p["hourly"][vmap[p["bird"]]]
        dd = H.index // 24
        keep = np.array([(p["bird"], x) in qd for x in dd])
        Hk = H[keep].copy()
        Hk["hour"] = Hk.index % 24
        Hk["season"] = pd.to_datetime((Hk.index // 24) * 86400, unit="s").month.map(SEASONS)
        hparts.append(Hk.assign(bird=p["bird"]))
    Hh = pd.concat(hparts).fillna(0.0)
    k2c = [c for c in Hh.columns if c.startswith("k2_") and c != "k2_unknown"]
    k6c = [c for c in Hh.columns if c.startswith("k6_") and c != "k6_unknown"]
    for sea in ("all", "DJF", "MAM", "JJA", "SON"):
        X = Hh if sea == "all" else Hh[Hh.season == sea]
        G = X.groupby(["bird", "hour"])[k2c + k6c].sum()
        cls2, cls6 = G[k2c].sum(axis=1), G[k6c].sum(axis=1)
        S = pd.DataFrame({"share_flight": (G.get("k2_flight", 0) + G.get("k2_flight_imputed", 0)) / cls2})
        for c in k6c:
            S["share_" + c] = G[c] / cls6
        S = S[cls2 >= DIEL_MIN_S]
        for h in range(24):
            if h not in S.index.get_level_values(1):
                continue
            V = S.xs(h, level=1)
            mean, lo, hi = boot_ci(V.to_numpy(), n_boot, rng)
            for v, a, b_, c_ in zip(S.columns, mean, lo, hi):
                drows.append({"season": sea, "hour": h, "variable": v, "n_birds": len(V), "mean": a, "ci_lo": b_, "ci_hi": c_})
    diel = pd.DataFrame(drows)
    diel.to_csv(out / "diel.csv", index=False)

    # window-count share vs time share, both over all kept (non-pre-fledge) data of each bird (no coverage rule);
    # 'flight_total' = flight + flight_imputed (bout windows count as 'flight' in the window shares)
    tot = pd.DataFrame({p["bird"]: p["hourly"][vmap[p["bird"]]].sum() for p in P}).T.fillna(0.0)
    wrows = []
    for sysn in ("k2", "k6", "sub"):
        cnt_all = pd.DataFrame({p["bird"]: p["counts"][sysn]["all"] for p in P}).T.fillna(0).drop(columns="unknown", errors="ignore")
        cnt_s = pd.DataFrame({p["bird"]: p["counts"][sysn]["sched"] for p in P}).T.fillna(0).drop(columns="unknown", errors="ignore")
        tc = tot[[c for c in tot.columns if c.startswith(sysn + "_") and c != f"{sysn}_unknown"]]
        tc.columns = [c[len(sysn) + 1:] for c in tc.columns]
        states = sorted(set(cnt_all.columns) | set(tc.columns))
        sh = {name: (X.reindex(columns=states, fill_value=0).div(X.reindex(columns=states, fill_value=0).sum(axis=1), axis=0))
              for name, X in (("window_share_all", cnt_all), ("window_share_scheduled", cnt_s), ("time_share", tc))}
        if "flight" in states:
            states = states + ["flight_total"]
            sh = {k: v.assign(flight_total=v["flight"] + v.get("flight_imputed", 0.0)) for k, v in sh.items()}
        stats_ = {}
        for name, S in sh.items():
            S = S[S.sum(axis=1) > 0]
            stats_[name] = (boot_ci(S.to_numpy(), n_boot, rng), len(S))
        for i, s in enumerate(states):
            r = {"system": sysn, "state": s}
            for name, ((mean, lo, hi), nb) in stats_.items():
                r.update({name: mean[i], f"{name}_lo": lo[i], f"{name}_hi": hi[i], f"{name}_n_birds": nb})
            r["ratio_window_all_to_time"] = r["window_share_all"] / r["time_share"] if r["time_share"] > 0 else np.nan
            wrows.append(r)
    wvt = pd.DataFrame(wrows)
    wvt.to_csv(out / "window_vs_time.csv", index=False)

    # flight composition: continuous bouts (conditional) and scheduled flight samples
    crows = []
    def sched_flight(p):
        x = p["k2k6_sched"]
        if "flight" not in x.index.get_level_values(0):
            return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["month", "k6"]))
        x = x.xs("flight", level="k2")
        return pd.Series(x.to_numpy(), index=pd.MultiIndex.from_arrays([np.zeros(len(x), int), x.index], names=["month", "k6"]))

    for src, getter in (("continuous bouts (conditional on a triggered bout)", lambda p: p["bout_comp"]),
                        ("scheduled flight samples (ACC-only)", sched_flight)):
        S = pd.concat({p["bird"]: getter(p) for p in P}, names=["bird"])
        if not len(S):
            continue
        S = S.unstack("k6", fill_value=0.0).reindex(columns=[str(i) for i in range(K6)], fill_value=0.0)
        for mon in [0] + sorted(m for m in S.index.get_level_values("month").unique() if m > 0):
            X = S.groupby(level="bird").sum() if mon == 0 else S.xs(mon, level="month")
            X = X[X.sum(axis=1) >= COMP_MIN_S]
            if not len(X):
                continue
            mean, lo, hi = boot_ci(X.div(X.sum(axis=1), axis=0).to_numpy(), n_boot, rng)
            for j, c in enumerate(X.columns):
                crows.append({"source": src, "month": int(mon), "k6": c, "name": K6_NAMES.get(int(c), c), "n_birds": len(X),
                              "flight_hours": float(X[c].sum() / 3600), "mean": mean[j], "ci_lo": lo[j], "ci_hi": hi[j]})
    comp = pd.DataFrame(crows)
    comp.to_csv(out / "flight_composition.csv", index=False)
    k2k6 = pd.concat({p["bird"]: p["k2k6_sched"] for p in P}).groupby(level=[1, 2]).sum().unstack(fill_value=0) / 3600

    # tests and coverage sensitivity
    log("tests")
    tsum, tper, srows = [], [], []
    for thr in sorted({*SENS_COVERAGE, min_coverage}):
        Qt = D[D.coverage >= thr]
        npm = n_perm if thr == min_coverage else min(n_perm, 2000)
        s_, p_ = run_tests(Qt, B, "d_flight", npm, rng)
        tsum += [dict(r, min_coverage=thr, main=thr == min_coverage) for r in s_]
        tper += [x.assign(min_coverage=thr) for x in p_]
        g = Qt.groupby(["month", "bird"]).d_flight.agg(["mean", "size"])
        g = g[g["size"] >= MIN_DAYS_CELL]
        for mon in range(1, 13):
            if mon in g.index.get_level_values(0):
                x = g.xs(mon)
                srows.append({"min_coverage": thr, "month": mon, "n_birds": len(x), "n_bird_days": int(x["size"].sum()),
                              "mean_flight_h": x["mean"].mean()})
        srows.append({"min_coverage": thr, "month": 0, "n_birds": Qt.bird.nunique(), "n_bird_days": len(Qt),
                      "mean_flight_h": Qt.groupby("bird").d_flight.mean().mean()})
    tests = pd.DataFrame(tsum)
    tests.to_csv(out / "tests.csv", index=False)
    if tper:
        pd.concat(tper, ignore_index=True).to_csv(out / "tests_by_month.csv", index=False)
    sens = pd.DataFrame(srows)
    sens.to_csv(out / "coverage_sensitivity.csv", index=False)

    log("figures")
    figures(out, agg, diel, wvt, Q, min_coverage, comp, K6, sub_labels)
    meta = {"windows": str(windows), "states2": str(states2), "states6": str(states6), "substates": str(substates),
            "min_coverage": min_coverage, "n_boot": n_boot, "n_perm": n_perm, "seed": seed, "headline_variant": variant,
            "constants": {k: globals()[k] for k in ("BURST_GAP_S", "BOUT_JOIN_S", "CAP_FACTOR", "ROUND_S", "MIN_GAPS",
                                                    "SLOT_TOL_S", "SUPPRESSION_FOUND_MAX", "MIN_DAYS_CELL", "MIN_TEST_DAYS",
                                                    "MIN_GROUP_BIRDS", "DIEL_MIN_S", "COMP_MIN_S")},
            "birds": len(B), "bird_days": len(D), "bird_days_qualifying": len(Q), "pre_fledge_days_dropped": n_prefledge_days,
            "pre_fledge_windows_dropped": int(B.n_pre_fledge.sum()), "seconds": round(time.time() - t0)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    write_summary(out, meta, check, B, sch, D, Q, agg, diel, wvt, comp, tests, sens, k2k6, K6, sub_labels)
    log(f"done ({time.time() - t0:.0f} s)")
    return meta


# ---------------------------------------------------------------- summary

def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_summary(out, meta, check, B, sch, D, Q, agg, diel, wvt, comp, tests, sens, k2k6, K6, sub_labels) -> None:
    A = agg[(agg.grouping == "all")]
    L = ["# Time budgets from the scheduled recording stream", "",
         "All numbers below are generated by `aquilacl.budgets`. Everything is DESCRIPTIVE except the section 'Tests'.", "",
         "## Why time budgets and not window shares", "",
         "Continuous mode (1 Hz GPS) starts when the bird moves fast and then yields one window every 10 s, while a scheduled "
         "burst yields one window per interval. Window shares therefore over-count flight. Here each scheduled sample "
         "represents the time until the next scheduled sample (at most 1.5 x the local schedule interval); continuous "
         "windows are not samples.", "",
         "## Recording structure (measured)", "",
         f"- Scheduled samples: {check['scheduled_samples']:,} (first window of each k=0 burst) from "
         f"{check['k0_windows']:,} k=0 windows. {check['scheduled_with_k_in_segment_gt0']:,} scheduled samples have "
         f"k_in_segment > 0; {check['k0_k_in_segment0_not_scheduled']:,} k=0 windows with k_in_segment = 0 are not "
         "first of a burst.",
         f"- Share of scheduled samples with any GPS fix: {_f(check['scheduled_with_gps_share'], 3)}.",
         f"- Nominal interval per bird: " + ", ".join(f"{int(k)} s: {n} birds" for k, n in
                                                     B.nominal_interval_s.value_counts().sort_index().items()) + ".",
         f"- Bird-months whose interval differs from the bird's nominal interval: "
         f"{int(sch[(sch.period == 'all') & (sch.year_month != 'all')].differs_from_nominal.sum())}; "
         f"bird-months where day and night intervals differ: {_daynight_diff(sch)}.",
         f"- Gaps at the interval (per-bird median share): {_f(sch[sch.year_month == 'all'].share_at_interval.median(), 3)}; "
         f"at a multiple (missed bursts): {_f(sch[sch.year_month == 'all'].share_multiple.median(), 3)}; shorter: "
         f"{_f(sch[sch.year_month == 'all'].share_shorter.median(), 3)}.",
         f"- Continuous bouts: {check['bouts']:,}, {check['bout_hours']:,.0f} h. Of {check['clock_slots_inside_bouts']:,} "
         f"expected clock slots inside bouts, {check['slots_with_scheduled_sample']:,} "
         f"({_f(check['share_slots_found_pooled'], 3)} pooled; per-bird median {_f(check['share_slots_found_per_bird_median'], 3)}) "
         "have a scheduled sample. A missing slot means the burst was suppressed or absorbed into the continuous segment; "
         "the metadata cannot tell which.",
         f"- The k=1 window covering a slot is K=2 flight in {_f(check['share_slot_windows_flight'], 3)} of "
         f"{check['slots_in_k1_window']:,} slots; all k=1 windows: K=2 shares {json.dumps({k: round(x, 4) for k, x in check['k1_windows_k2_share'].items()})}.",
         f"- K=2 state_hmm agrees with the GMM state on {_f(check['k2_state_hmm_agrees_with_state_on_scheduled'], 4)} of scheduled samples.",
         f"- **Headline variant per bird** (mode {check['bout_imputation_mode']}): {check['birds_imputed']} birds 'imputed' "
         f"(bout time not covered by a scheduled sample is IMPUTED AS FLIGHT, category 'flight_imputed'), "
         f"{check['birds_plain']} birds 'plain' (no imputation). A bird's own share of found slots decides (< "
         f"{SUPPRESSION_FOUND_MAX} -> imputed); {check['birds_decided_by_pooled_rule']} birds with < {MIN_SLOTS_BIRD} "
         f"slots follow the pooled share, which judges the schedule "
         f"{'SUPPRESSED' if check['schedule_suppressed_in_bouts'] else 'continuing'} inside bouts.",
         f"- Elapsed time per bird (first to last sample), median shares: covered by scheduled samples "
         f"{_f(B['share_covered_scheduled_headline'].median(), 3)}, imputed from bouts "
         f"{_f(B['share_imputed_headline'].median(), 3)}, gap {_f(B['share_gap_headline'].median(), 3)}.", "",
         "## Bird-days and coverage", "",
         f"- {meta['birds']} birds, {meta['bird_days']:,} bird-days (local mean solar days); {meta['pre_fledge_days_dropped']} "
         f"pre-fledge bird-days and {meta['pre_fledge_windows_dropped']:,} pre-fledge windows dropped.",
         f"- Rule: a bird-day counts if covered time >= {meta['min_coverage']:.0%} of 24 h: {meta['bird_days_qualifying']:,} "
         f"bird-days from {Q.bird.nunique()} birds. Median coverage of all bird-days {_f(D.coverage.median(), 3)}.",
         "- Daily hours = 24 x (state time / classified covered time) of the day. If coverage is uneven within a day "
         "(e.g. longer night intervals), the scaling assumes the uncovered time has the same composition. Median "
         f"coverage of qualifying days: solar 06-18 h {_f(Q.coverage_day.median(), 3)}, 18-06 h {_f(Q.coverage_night.median(), 3)}.", "",
         "| min coverage | bird-days | birds | mean daily flight h (bird mean, all months) |", "|---|---|---|---|"]
    for _, r in sens[sens.month == 0].iterrows():
        L.append(f"| {r.min_coverage:.1f} | {int(r.n_bird_days):,} | {int(r.n_birds)} | {_f(r.mean_flight_h)} |")
    L += ["", "## Daily budget by month (all birds; bird mean, 95% CI over birds)", "",
          "| month | birds | flight h (headline) | ground h | flight h, plain for all birds | flight h, imputed for all birds | imputed flight h |", "|---|---|---|---|---|---|---|"]
    for mon in range(0, 13):
        g = A[A.month == mon].set_index("variable")
        if "d_flight" not in g.index:
            continue
        r = g.loc["d_flight"]
        L.append(f"| {'all' if mon == 0 else mon} | {int(r.n_birds)} | {_f(r['mean'])} [{_f(r.ci_lo)}, {_f(r.ci_hi)}] | "
                 f"{_f(g.loc['d_ground', 'mean'])} | {_f(g.loc['d_flight_plain', 'mean'])} | {_f(g.loc['d_flight_imp', 'mean'])} | "
                 f"{_f(g.loc['d_k2_flight_imputed', 'mean']) if 'd_k2_flight_imputed' in g.index else '0'} |")
    g = A[A.month == 0].set_index("variable")
    L += ["", "K=6 hours per day (all months, bird mean): " + "; ".join(
        f"{K6_NAMES.get(int(c[5:]), c[5:]) if c[5:].isdigit() else c[5:]} {_f(g.loc[c, 'mean'])}"
        for c in g.index if c.startswith("d_k6_")) + ".",
          "Ground sub-states, hours per day: " + "; ".join(f"{c[6:]} ({SUB_NAMES.get(c[6:], '')}) {_f(g.loc[c, 'mean'])}"
                                                           for c in g.index if c.startswith("d_sub_")) + ".",
          "K=6 flight types on scheduled samples are inferred mostly without GPS (acceleration only).", "",
          "## Diel budget (local mean solar hour, all seasons)", ""]
    dl = diel[(diel.season == "all") & (diel.variable == "share_flight")].sort_values("hour")
    L.append("Flight share of covered time by hour: " + ", ".join(f"{int(r.hour)}h {_f(r['mean'])}" for _, r in dl.iterrows()) + ".")
    L += ["", "## Groups (daily flight hours, all months pooled per bird; descriptive)", "",
          "| grouping | group | birds | flight h [95% CI] |", "|---|---|---|---|"]
    for gcol in ("sex", "age_class", "cohort", "migrant_all", "migrant_c2024", "young_c2024"):
        for _, r in agg[(agg.grouping == gcol) & (agg.month == 0) & (agg.variable == "d_flight")].iterrows():
            L.append(f"| {gcol} | {r.group} | {int(r.n_birds)} | {_f(r['mean'])} [{_f(r.ci_lo)}, {_f(r.ci_hi)}] |")
    L += ["", "Pooling months mixes seasons: groups differ in which months they were tracked (e.g. migrant-flagged birds "
          "are mostly deployed in October). Use the monthly rows in budgets_by_group.csv.", "",
          "## Window-count share vs time share (old bias)", "",
          "| system | state | window share (all) | window share (scheduled) | time share | ratio all/time |", "|---|---|---|---|---|---|"]
    for _, r in wvt.iterrows():
        L.append(f"| {r.system} | {r.state} | {_f(r.window_share_all, 3)} | {_f(r.window_share_scheduled, 3)} | "
                 f"{_f(r.time_share, 3)} | {_f(r.ratio_window_all_to_time)} |")
    L += ["", "## Flight composition", "",
          "Continuous-bout composition is CONDITIONAL ON BEING IN A TRIGGERED BOUT (fast movement); it is not the "
          "composition of all flight time.", "", "| source | state | birds | share [95% CI] |", "|---|---|---|---|"]
    for _, r in comp[comp.month == 0].iterrows():
        L.append(f"| {r.source} | {r['name']} | {int(r.n_birds)} | {_f(r['mean'], 3)} [{_f(r.ci_lo, 3)}, {_f(r.ci_hi, 3)}] |")
    L += ["", "Scheduled time (hours, plain weights) by K=2 (rows) x K=6 (columns): consistency check.", "",
          k2k6.round(1).to_markdown() if _has_tabulate() else k2k6.round(1).to_string(), "",
          "## Tests (bird is the unit; daily flight hours)", "",
          "Per month: difference of bird means (bird-months with >= 5 qualifying days; months with >= 3 birds per group). "
          "Combined: weighted mean of monthly differences; p from permuting labels between birds (one permutation for all months).", "",
          "| test | min coverage | birds (1 / 0) | months | combined diff (h) | median monthly Hedges g | p | permutations |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        L.append(f"| {r.description} | {r.min_coverage:.1f}{' (main)' if r.main else ''} | {r.n_birds_1} / {r.n_birds_0} | "
                 f"{r.months_used} | {_f(r.combined_diff_h)} | {_f(r.median_monthly_hedges_g)} | {_f(r.p_perm, 4)} | {r.n_perm} |")
    L += ["", "## Caveats", "",
          "- The trigger rule is not known from the firmware; it is inferred from speeds at bout onset.",
          "- Imputing uncovered bout time as flight relies on continuous mode being flight (see the K=2 share of k=1 windows above).",
          "- The 1.5 x interval cap leaves long data gaps uncovered; coverage and the day rule control for that, and "
          "the 30% / 70% sensitivity is reported.",
          "- Migrant and young-tagged tests are within cohort 2024 only; residency is confounded with deployment month, "
          "and young-tagged birds are juveniles while the comparison birds are of unknown age.",
          "- Sex differences may reflect breeding roles, ages and the months each bird was tracked.",
          "- State names are the stage-6 labels of the K=6 run; ground sub-states come from substates_r6_k2.", ""]
    (out / "summary.md").write_text("\n".join(L))


def _daynight_diff(sch: pd.DataFrame) -> int:
    x = sch[(sch.period != "all") & (sch.n_gaps >= MIN_GAPS)].pivot_table(index=["bird", "year_month"], columns="period",
                                                                           values="interval_s")
    if not {"day", "night"} <= set(x.columns):
        return 0
    x = x.dropna()
    return int((x["day"] != x["night"]).sum())


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--states2", type=Path, required=True, help="K=2 states directory (states.parquet)")
    ap.add_argument("--states6", type=Path, required=True, help="K=6 states directory (states.parquet)")
    ap.add_argument("--substates", type=Path, required=True, help="substates directory (substates.parquet)")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--bout-imputation", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.windows, a.states2, a.states6, a.substates, a.reference, a.out, a.min_coverage, a.n_boot, a.n_perm, a.workers,
        a.bout_imputation, a.seed)


if __name__ == "__main__":
    main()
