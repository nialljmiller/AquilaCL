"""Time budgets (share of TIME per behaviour) from the scheduled, behaviour-independent recording stream.

Why not window counts. Continuous mode (sensitivity_k == 1, 1 Hz GPS) is switched on when the bird moves fast
(bout onsets median 11.6 m/s, RESULTS_stage6 §2) and then produces one 10-s window every 10 s for up to ~900 s.
A flight therefore yields up to ~90 windows while a scheduled burst yields one, so the share of WINDOWS in a state
over-weights flight by an amount that depends on the trigger, the season and the bird. Continuous-mode windows are
excluded from the scheduled sample for that reason; they are used only (a) as direct evidence of flight for bout
time that the schedule does not cover (below) and (b) for the composition of flight, conditional on being in a
triggered bout.

Scheduled samples. Per bird, windows with sensitivity_k == 0 sorted by t_start, EXCLUDING k == 0 windows that lie
inside a continuous bout span (recording-format interruptions of continuous mode, almost never on the bird's clock;
counted); a window starts a new burst when it
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

Scheduled bursts during continuous bouts. Bouts are the shared definition of aquilacl.flight.bout_ids (runs of
k == 1 windows in time order with a bounded start-to-start step; windows of another mode in between do not split a
bout; see flight.py); a bout runs from the first t_start to the last t_start + its recorded seconds; all bouts count (no minimum duration). The code measures whether the schedule continues inside
bouts: per bird-month the clock phase (mode of t mod interval, 5 s bins) gives the expected slots inside each bout;
a slot is found when any k == 0 burst (including the excluded ones inside bouts) starts within SLOT_TOL_S of it. A missing slot means the burst was either
suppressed by the firmware or absorbed into the continuous segment (then its window carries k == 1); the metadata
cannot separate the two, and both are treated alike.
Two variants are always computed:
- 'plain': scheduled coverage only (bouts ignored).
- 'bouts': ALL bout time is CLASSIFIED BY THE STATE OF ITS OWN k == 1 WINDOWS (each bout window represents
  [its t_start, the next bout window's t_start), the last one until the bout end). These are observed states, not
  imputed flight: bout windows that are K=2 ground count as ground. A scheduled sample's coverage ends at the start
  of the next bout and is clipped to time outside bouts, so k == 0 samples that fall inside a bout span (on or off
  the clock) cover no bout time; their number and the bout time the plain variant gives them are reported. Bout time
  keeps its own categories ('<system>_bout_<label>') and is added to the scheduled categories for daily hours.
The headline variant is chosen PER BIRD by --bout-imputation (auto: 'bouts' if the bird's share of found slots
is < SUPPRESSION_FOUND_MAX, else 'plain'; birds with < MIN_SLOTS_BIRD slots follow the pooled share; on/off force
one variant for all birds). Both variants for all birds are reported alongside (d_flight_plain / d_flight_bouts).

Behaviour per scheduled sample: K=2 state (state_hmm of --states2), K=6 state (state_hmm of --states6) and the
ground sub-state (substates.parquet label s.j where its GMM 'state' is 0 and the K=2 state is ground; otherwise
'ground_unassigned'). Scheduled k == 0 windows rarely have GPS, so K=6 flight types on them rest on acceleration
alone; flight composition is therefore also given from continuous bouts (K=6 among K=2-flight k == 1 windows,
weighted by recorded seconds = acc_samples / 20) and that composition is CONDITIONAL ON BEING IN A TRIGGERED BOUT.

Days and hours are LOCAL MEAN SOLAR time (offset = local_solar_hour - UTC hour per window; missing values are filled
from the previous/next window, else FALLBACK_OFFSET_H). Covered intervals are split at solar-hour boundaries, so
day and hour budgets add up exactly.

Daily hours are STRATIFIED BY SOLAR HOUR. Coverage is very uneven within the day: most tags record no ACC from about
20:00 to 05:00 local solar time (a fixed clock window, measured in budgets_r6), and the night is almost all ground.
Scaling a day's covered time to 24 h (the first estimator, kept as 'scaled_24h' for comparison) therefore inflates
daily flight hours. Here, for each bird-day and each of the 24 solar-hour bins, the state shares of the covered time
in the bin are used when the bin has >= MIN_BIN_S classified scheduled seconds ('direct' bin); daily hours = sum
over the 24 bins of (share x 1 h). Bout time is observed and enters a bin as itself (by state); the scheduled
share applies to the rest of that hour (so a bin holding only bout time is not scaled up to a full hour).
The diel profile uses the same bin hours (direct bins of qualifying days), so its sum over hours matches the daily
flight hours up to the different bird sets per hour. A bin that is not direct is filled, in this order, from (1) the bird's own bird-month mean
share for that hour (seconds-weighted over its direct bins), (2) the bird's own all-month mean for that hour,
(3) the POPULATION mean for that calendar month and hour (bird-weighted mean of the birds' own month-hour shares,
>= POP_MIN_BIRDS birds), then the population all-month mean for that hour. Level (3) is needed because no data
exist for the night hours of most birds; with --fill own, days that need it do not qualify. A bird-day qualifies
if >= --min-direct-bins (default 12) of its 24 bins are direct and every other bin can be filled (sensitivity:
8 / 14 direct bins, and own-data filling only). Hours filled at each level are kept per bird-day. Raw covered
hours (h_*) and the coverage share are kept too. Pre-fledge windows of tagged-young birds are dropped before
anything else (as in demography.py) and counted (windows and the solar days that hold them). Cohort = deployment
year: the year of the bird's earliest reference deploy_on (the first-window year only when deploy_on is missing;
counted).

Aggregation (descriptive): per bird and cell (month, group, ...) the mean over its qualifying days (>= MIN_DAYS_CELL);
then the bird-weighted mean over birds with a bootstrap-over-birds 95 % CI. The all-months headline is MONTH-BALANCED:
the mean of the 12 monthly bird means (bootstrap over birds); the day-weighted per-bird mean over all months is kept,
labelled, because it over-weights the months in which most birds were tracked. Monthly curves are also given for
birds whose age is not stated (no young-tagged birds), split by the reference migrant flag (a flag, not a track-derived
residency class; track-derived classes are deferred to residency.py and are not imported here).

Tests (bird is the unit): per calendar month the difference in bird means of daily flight hours (bird-months with
>= MIN_TEST_DAYS qualifying days, months with >= MIN_GROUP_BIRDS birds per group); combined statistic = weighted mean
of the monthly differences, weight n1 n0 / (n1 + n0). One permutation of the labels between birds is applied to all
months at once (a bird keeps one label in every month), so the dependence between months of the same bird is kept in
the null; stratified permutation keeps labels within deployment year, and p is NaN when no stratum holds both labels.
Each comparison gets a NUISANCE AUC (rank AUC of the label on deploy day-of-year, deployment year and nominal
interval, the largest of the three, max(AUC, 1 - AUC)); a comparison with nuisance AUC >= NUISANCE_AUC_MAX is
'not testable in this design' and is reported descriptively only (difference, Hedges g, bird counts; no p). Holm
correction runs over the testable comparisons of the main day rule. p at the permutation floor prints as
'<= 1/(n+1)'. In cohort 2024 the migrant flag and the young-tagged flag are (nearly) complementary, so they are ONE
contrast (migrant-flagged, deployed Sep-Oct, vs young-tagged, deployed Jun-Jul): age, residency and deployment month
cannot be separated.

K=6 flight states are reported only as a DESCRIPTIVE ACC-STATE MIX of flight time, under neutral names: scheduled
samples almost never have GPS, and state 5 (named 'flapping' in stage 6) holds about a fifth of flight time against
about 2 % flapping in published accounts, so the K=6 flight states are not behaviours.

Outputs (--out, new or empty; bird-keyed tables hold no coordinates, offsets or UTC times): birds.csv,
schedule.csv, recording_check.json, birddays.csv, budgets_by_group.csv, diel.csv, window_vs_time.csv,
flight_composition.csv, tests.csv, tests_by_month.csv, coverage_sensitivity.csv, scaled_24h_comparison.csv,
*.png, summary.md, metadata.json.

Usage: python -m aquilacl.budgets --windows W --states2 S2 --states6 S6 --substates SUB --reference CSV --out NEW
       [--min-direct-bins 12] [--fill population|own] [--n-boot 1000] [--n-perm 5000] [--workers N]
       [--bout-imputation auto|on|off]   (on = 'bouts' variant for every bird, off = 'plain')
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
from .flight import bout_ids

WINDOW_S, FS = 10.0, 20.0
BURST_GAP_S = 11.0
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
DIEL_MIN_BINS = 3          # direct bins per bird and solar hour for the diel profile
COMP_MIN_S = 600.0
MIN_SLOTS_BIRD = 20
MIN_BIN_S = 600.0          # classified seconds for a solar-hour bin to count as directly covered
MIN_DIRECT_BINS = 12       # default day rule: >= 12 of 24 bins direct, the rest fillable
POP_MIN_BIRDS = 3          # population fill cell needs this many birds with direct coverage
SENS_RULES = ((8, "population"), (14, "population"), (12, "own"))
OLD_MIN_COVERAGE = 0.5     # first estimator's day rule (scaled_24h side table)
SYSTEMS = ("k2", "k6", "sub")
BOUT = "bout_"             # prefix of bout-time categories: '<system>_bout_<label>'
STILL_VEDBA_G = 0.02       # 'still' window (stage-4 rule, scripts/diagnose_r4.py): label_perched == 1 or vedba < 0.02 g
SHORT_ACC = 190            # windows with fewer ACC samples are short (40-sample bouts, 64-sample tails, short runs)
AFTER_BOUT_S = 60.0        # a scheduled sample starting within this many seconds after a bout end is 'right after a bout'
NUISANCE_AUC_MAX = 0.9     # a comparison whose nuisance AUC reaches this is 'not testable in this design'
NON_MIGRATION_MONTHS = (12, 1, 2, 6, 7, 8)  # months without spring (Mar-May) or autumn (Sep-Nov) passage (assumed)
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}
K2_NAMES = {0: "ground", 1: "flight"}
# K=6 names: ground states keep their stage-6 names; flight states get NEUTRAL names (ACC-state mix, not behaviours).
# Stage-6 names of the flight states of states_r6_yaw60_bpb4 (other state dirs need their own): 2 'thermal circling',
# 3 'fast descending glide', 4 'cruising', 5 'flapping' (21 % of flight time here vs about 2 % published: not flapping).
K6_NAMES = {0: "rest", 1: "perched-active", 2: "flight ACC-state 2", 3: "flight ACC-state 3", 4: "flight ACC-state 4",
            5: "flight ACC-state 5"}
K6_STAGE6 = {2: "thermal circling", 3: "fast descending glide", 4: "cruising", 5: "flapping"}
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
    """Continuous bouts of one bird (t sorted) by the canonical rule aquilacl.flight.bout_ids on k == 1 windows
    (only k == 1 windows define bouts; other-mode windows in between do not split them): (start, end, bout index per
    window or -1), bouts sorted by start; end = max over the bout's windows of t_start + recorded seconds (>= 1 s)."""
    t, acc_samples = np.asarray(t, float), np.asarray(acc_samples, float)
    bid = np.full(len(t), -1)
    raw = np.asarray(bout_ids(t, np.asarray(k) == 1))
    idx = np.flatnonzero(raw >= 0)
    if not len(idx):
        return np.zeros(0), np.zeros(0), bid
    _, inv = np.unique(raw[idx], return_inverse=True)
    nb = inv.max() + 1
    start = np.full(nb, np.inf)
    end = np.full(nb, -np.inf)
    np.minimum.at(start, inv, t[idx])
    np.maximum.at(end, inv, t[idx] + np.maximum(acc_samples[idx] / FS, 1.0))
    order = np.argsort(start, kind="stable")
    rank = np.empty(nb, int)
    rank[order] = np.arange(nb)
    bid[idx] = rank[inv]
    start, end = start[order], np.maximum(end[order], start[order] + 1.0)
    return start, end, bid


def bout_spans(t: np.ndarray, bid: np.ndarray, be: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time each bout window represents: [its t_start, the next window of the same bout) and, for the last window
    of a bout, [t_start, bout end). Returns (start, end, window index), sorted and disjoint (bouts do not overlap)."""
    idx = np.flatnonzero(np.asarray(bid) >= 0)
    if not len(idx):
        return np.zeros(0), np.zeros(0), idx
    b = bid[idx]
    same = np.r_[b[1:] == b[:-1], False]
    nxt = np.r_[t[idx][1:], np.inf]
    return t[idx], np.where(same, nxt, be[b]), idx


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


def timeline(ts: np.ndarray, I: np.ndarray, bs: np.ndarray, be: np.ndarray, variant: str):
    """Covered intervals of one bird. 'plain': each scheduled sample covers [t_i, coverage end). 'bouts': the bout
    spans [bs, be) are covered by the bouts themselves (typed later by their own windows), and scheduled coverage is
    cut at the next bout start and clipped to time outside bouts, so a k == 0 sample inside a bout span covers no bout
    time. Returns (sample start, sample end, sample index) of the scheduled pieces; bouts are not included."""
    e = coverage_ends(ts, I, bs, impute=variant == "bouts")
    if variant != "bouts" or not len(bs):
        return ts, e, np.arange(len(ts))
    return uncovered_pieces(bs, be, ts, e)


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


# ---------------------------------------------------------------- solar-hour stratified daily budgets

def is_bout(col: str) -> bool:
    return f"_{BOUT}" in col


def _direct(H: pd.DataFrame, cols: list[str]):
    """Per bin: scheduled seconds per category (bout columns zeroed), classified scheduled seconds, bout seconds per
    category (zero in scheduled columns) and the direct mask. Bout time is observed and counts as itself; the scheduled
    share applies to the rest of the hour (3600 - bout s). A bin is direct if its scheduled seconds reach MIN_BIN_S,
    or the whole rest of the hour (a bin filled by a bout needs no share)."""
    X = H.reindex(columns=cols, fill_value=0.0).to_numpy(float, copy=True)
    bm = np.array([is_bout(c) for c in cols], bool).reshape(-1)
    Xb = np.where(bm[None, :], X, 0.0)
    X[:, bm] = 0.0
    cls = X.sum(1)
    rest = np.maximum(3600.0 - Xb.sum(1), 0.0)
    return X, cls, Xb, cls >= np.minimum(MIN_BIN_S, rest - 1.0)


def _cell_shares(X: np.ndarray, cls: np.ndarray, ok: np.ndarray, keys: list[np.ndarray], cols: list[str]) -> pd.DataFrame | None:
    """Seconds-weighted shares of the direct bins pooled per key (index = keys, columns = cols)."""
    if not ok.any():
        return None
    g = pd.DataFrame(X[ok], columns=cols).assign(_c=cls[ok]).groupby([k[ok] for k in keys]).sum()
    return g[cols].div(g["_c"], axis=0)


def _cal_month(dayn: np.ndarray) -> np.ndarray:
    return pd.to_datetime(np.asarray(dayn, np.int64) * 86400, unit="s").month.to_numpy()


def population_cells(Hs: list[pd.DataFrame], cats: dict[str, list[str]]) -> dict:
    """Per system: (calendar month x hour table, hour table) of bird-weighted mean shares over birds with direct
    coverage there (each bird's share pooled over its own direct bins); cells with < POP_MIN_BIRDS birds dropped."""
    out = {}
    for s, cols in cats.items():
        mh, hh = [], []
        for H in Hs:
            if not len(H):
                continue
            hb = H.index.to_numpy(np.int64)
            X, cls, _, ok = _direct(H, cols)
            ok = ok & (cls > 0)
            a = _cell_shares(X, cls, ok, [_cal_month(hb // 24), hb % 24], cols)
            b = _cell_shares(X, cls, ok, [hb % 24], cols)
            if a is not None:
                mh.append(a), hh.append(b)

        def bmean(L):
            if not L:
                return None
            A = pd.concat(L)
            g = A.groupby(level=list(range(A.index.nlevels)))
            n = g.size()
            return g.mean()[n >= POP_MIN_BIRDS]

        out[s] = (bmean(mh), bmean(hh))
    return out


def stratify(H: pd.DataFrame, cats: dict[str, list[str]], pop: dict | None = None) -> pd.DataFrame:
    """Daily hours per category from solar-hour strata for one bird (H: hourly() table, index = absolute local hour bin).

    For each day with any covered bin and each of the 24 hours: the bin's own state shares of scheduled time if direct,
    else filled from the bird-month mean for that hour (1), the bird's all-month mean for that hour (2), the population
    month-hour mean, else the population hour mean (3, only if ``pop`` is given). Bin hours = bout seconds / 3600
    (per bout category) + (1 - bout seconds / 3600) x share; daily hours = sum over the 24 bins (NaN if any bin stays
    unfilled). Fill counts (hours) come from the K=2 system; d_flight_from_population / d_flight_from_bird_allmonth
    are the flight hours that the population fill / the bird's all-month hour mean contribute.
    """
    hb = H.index.to_numpy(np.int64)
    day, hour = hb // 24, hb % 24
    days = np.unique(day)
    di = np.searchsorted(days, day)
    dates = pd.to_datetime(days * 86400, unit="s")
    ym_day = (dates.year * 12 + dates.month - 1).to_numpy()
    cm_day = dates.month.to_numpy()
    res = {"dayn": days}
    for s, cols in cats.items():
        if not cols:
            continue
        X, cls, Xb, ok = _direct(H, cols)
        S = np.full((len(days), 24, len(cols)), np.nan)
        L = np.full((len(days), 24), -1)
        IMP = np.zeros((len(days), 24, len(cols)))
        IMP[di, hour] = Xb
        with np.errstate(invalid="ignore", divide="ignore"):
            S[di[ok], hour[ok]] = np.where(cls[ok, None] > 0, X[ok] / cls[ok, None], 0.0)
        L[di[ok], hour[ok]] = 0
        ok = ok & (cls > 0)  # fill tables come from bins with scheduled time
        P = pop.get(s, (None, None)) if pop else (None, None)
        levels = ((1, _cell_shares(X, cls, ok, [ym_day[di], hour], cols), lambda d, h: [ym_day[d], h]),
                  (2, _cell_shares(X, cls, ok, [hour], cols), lambda d, h: [h]),
                  (3, P[0], lambda d, h: [cm_day[d], h]),
                  (3, P[1], lambda d, h: [h]))
        for lev, table, keyf in levels:
            d_, h_ = np.nonzero(L < 0)
            if not len(d_):
                break
            if table is None or not len(table):
                continue
            k = keyf(d_, h_)
            idx = pd.MultiIndex.from_arrays(k) if len(k) > 1 else pd.Index(k[0])
            V = table.reindex(index=idx, columns=cols).to_numpy(float)
            g = np.isfinite(V).all(1)
            S[d_[g], h_[g]], L[d_[g], h_[g]] = V[g], lev
        unf = (L < 0).sum(1)
        wrest = 1.0 - IMP.sum(-1) / 3600.0
        B = wrest[..., None] * S + IMP / 3600.0  # S of bout columns is 0 where known, NaN where unfilled
        Hd = np.nansum(B, axis=1)
        Hd[unf > 0] = np.nan
        for j, c in enumerate(cols):
            res["d_" + c] = Hd[:, j]
        if s == "k2":
            res["n_direct"] = (L == 0).sum(1)
            res["direct_mask"] = ["".join("x" if x == 0 else "-" for x in row) for row in L]  # solar hours 0..23
            res["h_fill_bird_month"], res["h_fill_bird"] = (L == 1).sum(1), (L == 2).sum(1)
            res["h_fill_population"], res["h_unfilled"] = (L == 3).sum(1), unf
            fj = [j for j, c in enumerate(cols) if c == "k2_flight"]
            res["d_flight_from_population"] = np.where(L == 3, wrest * S[..., fj].sum(-1), 0.0).sum(1)
            res["d_flight_from_bird_allmonth"] = np.where(L == 2, wrest * S[..., fj].sum(-1), 0.0).sum(1)
    return pd.DataFrame(res)


def bin_hours(H: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Hours per category in each bin (bout time as itself + the scheduled share for the rest of the hour) and the
    direct mask; rows of direct bins sum to 1."""
    X, cls, Xb, ok = _direct(H, cols)
    with np.errstate(invalid="ignore", divide="ignore"):
        B = (1.0 - Xb.sum(1) / 3600.0)[:, None] * np.where(cls[:, None] > 0, X / cls[:, None], 0.0)
    return B + Xb / 3600.0, ok


def qualifies(D: pd.DataFrame, min_direct: int, fill: str) -> pd.Series:
    """Day rule: >= min_direct direct bins and every other bin filled (fill='own': without the population level)."""
    q = (D["n_direct"] >= min_direct) & (D["h_unfilled"] == 0) & D["d_flight"].notna()
    if fill == "own":
        q &= D["h_fill_population"] == 0
    return q


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
    f = _read(d / "features.parquet", ["speed_mean", "vedba", "label_perched"])
    t_all = to_seconds(m["t_start"])
    s2, s2g = _state_arr(st2, n_all, "state_hmm"), _state_arr(st2, n_all, "state")
    s6 = _state_arr(st6, n_all, "state_hmm")
    sub_lab = np.full(n_all, None, object)
    sub_state = np.full(n_all, -1)
    if sub is not None and len(sub):
        sub_lab[sub["row"].to_numpy()] = sub["label"].astype(str).to_numpy()
        sub_state[sub["row"].to_numpy()] = sub["state"].to_numpy()
    res = {"bird": d.name, "n_windows": n_all, "first_year": int(pd.Timestamp(t_all.min(), unit="s").year),
           "n_pre_fledge": 0, "n_pre_fledge_days": 0, "fledge": None}
    # continuous-mode windows on ALL windows of the bird (I2 recount): full windows, 'still' windows, K=2 ground
    k_all, acc_all = m["sensitivity_k"].to_numpy(), m["acc_samples"].to_numpy()
    vedba = f["vedba"].to_numpy(float) if "vedba" in f else np.full(n_all, np.nan)
    perched = f["label_perched"].to_numpy(float) if "label_perched" in f else np.full(n_all, np.nan)
    still = (perched == 1) | (vedba < STILL_VEDBA_G)
    k1, full = k_all == 1, acc_all == 200
    res["k1_count"] = {"k1_windows": int(k1.sum()), "k1_full": int((k1 & full).sum()), "k1_full_still": int((k1 & full & still).sum()),
                       "k1_still": int((k1 & still).sum()), "k1_k2_known": int((k1 & (s2 >= 0)).sum()),
                       "k1_k2_ground": int((k1 & (s2 == 0)).sum()), "k1_full_k2_ground": int((k1 & full & (s2 == 0)).sum())}
    keep = np.ones(n_all, bool)
    if info.get("young"):
        speed = f["speed_mean"].to_numpy(float) if "speed_mean" in f else np.full(n_all, np.nan)
        fl = reference.fledge_from_speed(pd.Series(pd.to_datetime(t_all, unit="s", utc=True)), speed, info["hatch_year"])
        res["fledge"] = fl["fledge"]
        keep = t_all >= (fl["fledge"] - pd.Timestamp(0, tz="UTC")).total_seconds()
        res["n_pre_fledge"] = int((~keep).sum())
        if (~keep).any():  # solar days holding pre-fledge windows
            o = np.argsort(t_all[~keep], kind="stable")
            tp = t_all[~keep][o]
            offp, _ = solar_offset_h(tp, m["local_solar_hour"].to_numpy(float)[~keep][o])
            res["n_pre_fledge_days"] = int(len(np.unique(np.floor((tp + offp * 3600) / 86400))))
    order = np.flatnonzero(keep)[np.argsort(t_all[keep], kind="stable")]
    t = t_all[order]
    k = k_all[order]
    acc = acc_all[order]
    kin = m["k_in_segment"].to_numpy()[order]
    seg = m["segment"].to_numpy()[order]
    sun = m["sun_elev_deg"].to_numpy(float)[order]
    c2, c2g, c6, slab, sst = s2[order], s2g[order], s6[order], sub_lab[order], sub_state[order]
    res["n_kept"] = len(t)
    if not len(t):
        res["drop_reason"] = "no windows after the pre-fledge cut"
        return res
    off, fallback = solar_offset_h(t, m["local_solar_hour"].to_numpy(float)[order])
    res["offset_fallback"] = fallback

    # labels of every kept window
    lab2 = np.array([K2_NAMES.get(v, "unknown") for v in c2], object)
    lab6 = np.where(c6 >= 0, c6.astype(str), "unknown").astype(object)
    labs = np.where(lab2 == "ground", np.where((sst == 0) & (slab != None), slab, "ground_unassigned"), lab2)  # noqa: E711

    k0 = k == 0
    bs, be, bid = bouts_from_windows(t, k, acc)
    # k == 0 windows inside a bout span are recording-format interruptions of continuous mode, not scheduled bursts
    # (flight.bout_ids: about 3 % start on the bird's clock, near chance): they are excluded from the scheduled samples
    jw = np.searchsorted(bs, t, "right") - 1
    k0_in_bout = k0 & (jw >= 0) & (t < be[np.maximum(jw, 0)]) if len(bs) else np.zeros(len(t), bool)
    burst_start_all = scheduled_mask(t, k)
    sched = scheduled_mask(t, np.where(k0_in_bout, -1, k))
    si = np.flatnonzero(sched)
    ts = t[si]
    ts_slot = t[burst_start_all]  # every k == 0 burst start (also inside bouts), for the clock-slot measurement
    seg_has_k1 = pd.Series(k == 1).groupby(seg).transform("any").to_numpy()
    jb = np.searchsorted(bs, ts, "right") - 1
    in_bout = (jb >= 0) & (ts < be[np.maximum(jb, 0)]) if len(bs) else np.zeros(len(ts), bool)
    # I9: scheduled samples that are short windows and/or start right after a bout end
    short = acc[si] < SHORT_ACC
    after = (jb >= 0) & ~in_bout & (ts - be[np.maximum(jb, 0)] <= AFTER_BOUT_S) if len(bs) else np.zeros(len(ts), bool)
    res["check"] = {"k0_windows": int(k0.sum()), "scheduled": len(si),
                    "k0_windows_inside_bouts_excluded": int(k0_in_bout.sum()),
                    "k0_burst_starts_inside_bouts_excluded": int((burst_start_all & k0_in_bout).sum()),
                    "k0_first_by_segment": int((k0 & (kin == 0)).sum()),
                    "sched_with_k_in_segment_gt0": int((kin[si] > 0).sum()),
                    "k0_kin0_not_sched": int((k0 & (kin == 0) & ~sched).sum()),
                    "sched_in_segment_with_k1": int(seg_has_k1[si].sum()),
                    "sched_inside_bout_span": int(in_bout.sum()),
                    "sched_with_gps": int((m["gps_fixes"].to_numpy()[order][si] > 0).sum()),
                    "sched_k2_state_vs_hmm_agree": int((c2[si] == c2g[si]).sum()),
                    "sched_k2_known": int((c2[si] >= 0).sum()),
                    "sched_short": int(short.sum()), "sched_after_bout": int(after.sum()),
                    "sched_short_and_after_bout": int((short & after).sum()),
                    "sched_short_flight": int((short & (c2[si] == 1)).sum()),
                    "sched_after_bout_flight": int((after & (c2[si] == 1)).sum()),
                    "sched_other_flight": int((~short & ~after & (c2[si] == 1)).sum()),
                    "bouts": len(bs), "bout_seconds": float((be - bs).sum())}
    res["counts"] = {sysn: {"all": pd.Series(L).value_counts().to_dict(), "sched": pd.Series(L[si]).value_counts().to_dict()}
                     for sysn, L in (("k2", lab2), ("k6", lab6), ("sub", labs))}
    if not len(si):
        res["drop_reason"] = "no scheduled samples"
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
        j = np.searchsorted(ts_slot, slots)
        near = np.minimum(np.abs(ts_slot[np.clip(j, 0, len(ts_slot) - 1)] - slots),
                          np.abs(ts_slot[np.clip(j - 1, 0, len(ts_slot) - 1)] - slots))
        n_slot += len(slots)
        n_found += int((near <= SLOT_TOL_S).sum())
        w = np.searchsorted(t[k1], slots, "right") - 1
        okw = (w >= 0) & (slots < t[k1][np.maximum(w, 0)] + WINDOW_S)
        n_win += int(okw.sum())
        n_flight += int((c2[k1][w[okw]] == 1).sum())
    res["slots"] = {"slots": n_slot, "found": n_found, "slot_in_k1_window": n_win, "slot_window_flight": n_flight}

    # timelines; behaviour of each scheduled sample; bout time typed by its own windows
    off_s = off[si] * 3600.0
    ws, we, wi = bout_spans(t, bid, be)
    res["hourly"], res["span"] = {}, {}
    plain_e = coverage_ends(ts, I, bs, impute=False)
    for variant in ("plain", "bouts"):
        qs, qe, qi = timeline(ts, I, bs, be, variant)  # scheduled pieces and the sample each belongs to
        if variant == "bouts":
            ps, pe, pw = ws, we, wi  # whole bout spans, each piece typed by its own k == 1 window
        else:
            ps, pe, pw = np.zeros(0), np.zeros(0), np.zeros(0, int)
        # zero-length entries at each sample start only count samples per hour (n_sched); they carry no time
        start_loc = np.r_[ts + off_s, qs + off_s[qi], ps + off[pw] * 3600.0]
        end_loc = np.r_[ts + off_s, qe + off_s[qi], pe + off[pw] * 3600.0]
        cats = {name: np.r_[lab[si], lab[si][qi], np.array([BOUT + str(x) for x in lab[pw]], object)]
                for name, lab in (("k2", lab2), ("k6", lab6), ("sub", labs))}
        is_start = np.r_[np.ones(len(ts), bool), np.zeros(len(qi) + len(pw), bool)]
        res["hourly"][variant] = hourly(start_loc, end_loc, cats, is_start)
        span = max(qe.max() if len(qe) else -np.inf, pe.max() if len(pe) else -np.inf) - ts[0]
        res["span"][variant] = {"span_s": float(span), "sched_s": float((qe - qs).sum()), "bout_s": float((pe - ps).sum()),
                                "bout_ground_s": float((pe - ps)[c2[pw] == 0].sum()),
                                "bout_unknown_s": float((pe - ps)[c2[pw] < 0].sum())}

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
    return [c for c in D.columns if c.startswith(("d_", "t_"))] + ["coverage", "coverage_day", "coverage_night", "n_direct",
                                                            "h_filled", "h_fill_population"]


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


def month_balanced(D: pd.DataFrame, var: str, n_boot: int, rng, birds=None) -> dict:
    """Month-balanced mean: per calendar month the bird mean of bird-month means (bird-months with >= MIN_DAYS_CELL
    days), then the plain mean over the months with data; 95 % CI from resampling birds (rows of the bird x month
    matrix). ``birds``: restrict to these birds."""
    X = D if birds is None else D[D.bird.isin(birds)]
    g = X.groupby(["bird", "month"])[var].agg(["mean", "size"])
    V = g["mean"].where(g["size"] >= MIN_DAYS_CELL).unstack().dropna(how="all")
    out = {"n_birds": int(len(V)), "months": int(V.notna().any().sum()), "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    if not len(V):
        return out
    M = V.to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["mean"] = float(np.nanmean(np.nanmean(M, 0)))
        if len(M) >= 2 and n_boot > 0:
            bs = np.array([np.nanmean(np.nanmean(M[rng.integers(0, len(M), len(M))], 0)) for _ in range(n_boot)])
            out["ci_lo"], out["ci_hi"] = float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    return out


def rank_auc(x: np.ndarray, lab: np.ndarray) -> float:
    """max(AUC, 1 - AUC) of a numeric variable for a 0/1 label (ties count half); NaN if a group is empty."""
    x, lab = np.asarray(x, float), np.asarray(lab, int)
    ok = np.isfinite(x)
    x, lab = x[ok], lab[ok]
    n1, n0 = int((lab == 1).sum()), int((lab == 0).sum())
    if not n1 or not n0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    auc = (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def nuisance_auc(B: pd.DataFrame, lab: pd.Series) -> tuple[float, str]:
    """Largest rank AUC of the label on the bird-level nuisance variables (deploy day-of-year, deployment year, nominal
    interval) and which variable gives it."""
    X = B.loc[lab.index]
    dep = pd.to_datetime(X["deploy_on"], utc=True, errors="coerce")
    nv = {"deploy_day_of_year": dep.dt.dayofyear.to_numpy(float), "deployment_year": pd.to_numeric(X["cohort"], errors="coerce").to_numpy(float),
          "nominal_interval_s": X["nominal_interval_s"].to_numpy(float)}
    best, name = np.nan, ""
    for k, v in nv.items():
        a = rank_auc(v, lab.to_numpy())
        if np.isfinite(a) and not (a <= best):
            best, name = a, k
    return best, name


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p values (NaN entries ignored and kept NaN)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return out
    o = ok[np.argsort(p[ok], kind="stable")]
    adj = np.maximum.accumulate((len(o) - np.arange(len(o))) * p[o])
    out[o] = np.minimum(adj, 1.0)
    return out


def fmt_p(p: float, n_perm: int, nd: int = 4) -> str:
    """p value text; at the permutation floor 1/(n+1) it prints as '<= 1/(n+1)'."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if n_perm and p <= 1.0 / (n_perm + 1) + 1e-12:
        return f"<= 1/{int(n_perm) + 1}"
    return f"{p:.{nd}f}"


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
    if st is None:
        n_inf = len(L) if 0 < L.sum() < len(L) else 0
    else:  # birds in strata that hold both labels: only they can change label under the stratified permutation
        both = pd.Series(L).groupby(st).agg(lambda x: 0 < x.sum() < len(x))
        n_inf = int(pd.Series(st).isin(both[both].index).sum())
    Lp = permute_labels(L, n_perm, rng, st).astype(float)
    dn, wn, cn = stats_(Lp)
    ok = np.isfinite(cn)
    p = float((1 + (np.abs(cn[ok]) >= abs(comb[0]) - 1e-12).sum()) / (1 + ok.sum())) \
        if np.isfinite(comb[0]) and n_inf > 0 and ok.any() else np.nan
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
            "p_perm": p, "n_perm": int(ok.sum()), "n_birds_informative_strata": int(n_inf)}
    return summ, per


def run_tests(D: pd.DataFrame, B: pd.DataFrame, var: str, n_perm: int, rng) -> tuple[list[dict], list[pd.DataFrame]]:
    """Bird-level comparisons of daily flight hours. Sex: primary with labels permuted within deployment year, plus
    the unstratified version. Cohort 2024: migrant-flagged vs young-tagged is ONE contrast (age, residency and
    deployment month confounded), all months and the non-migration months only. Every row carries the nuisance AUC
    and 'testable' (nuisance AUC < NUISANCE_AUC_MAX)."""
    bm = D.groupby(["bird", "month"])[var].agg(["mean", "size"])
    V = bm["mean"].where(bm["size"] >= MIN_TEST_DAYS).unstack()
    c24 = B.index[B.cohort == "2024"]
    mig, yng = B.migrant_flag == 1, B.young_flag == 1
    contrast = B.migrant_flag.where(B.index.isin(c24) & (mig ^ yng)).astype(float)  # 1 migrant, 0 young; others NaN
    sexlab = B.sex.map({"f": 1, "m": 0})
    nonmig = [c for c in V.columns if int(c) in NON_MIGRATION_MONTHS]
    dm = pd.to_datetime(B["deploy_on"], utc=True, errors="coerce").dt.month

    def months_of(sel):
        v = sorted(dm[sel & contrast.notna()].dropna().astype(int).unique())
        return "/".join(str(x) for x in v) or "n/a"

    specs = [
        ("sex_within_deploy_year", "female minus male, labels permuted within deployment year (primary)", sexlab, B.cohort, None),
        ("sex", "female minus male, labels permuted across all birds", sexlab, None, None),
        ("migrant_vs_young_c2024", f"cohort 2024: migrant-flagged (deploy months {months_of(mig)}) minus young-tagged "
         f"(deploy months {months_of(yng)}); ONE confounded contrast (age, residency, deployment month)", contrast, None, None),
        ("migrant_vs_young_c2024_nonmigration", "same contrast, non-migration months only "
         f"({', '.join(str(m) for m in NON_MIGRATION_MONTHS)})", contrast, None, nonmig),
    ]
    summ, per = [], []
    for name, desc, lab, strata, months in specs:
        lab = lab.dropna()
        Vs = V[V.index.isin(lab.index)]
        if months is not None:
            Vs = Vs[months]
        if lab.nunique() < 2 or not len(Vs) or not Vs.shape[1]:
            continue
        s, p = month_combined_test(Vs, lab, n_perm, rng, strata)
        auc, aucv = nuisance_auc(B, lab.loc[lab.index.isin(Vs.dropna(how="all").index)])
        confounded = bool(np.isfinite(auc) and auc >= NUISANCE_AUC_MAX)
        testable = bool(np.isfinite(auc) and not confounded and np.isfinite(s["p_perm"]))
        status = ("not testable in this design (descriptive)" if confounded else "tested" if testable
                  else "no test (too few birds per month or no informative stratum)")
        s["p_perm_confounded_do_not_report"] = s["p_perm"] if confounded else np.nan
        if confounded:  # no p is reported for a comparison the design cannot test
            s["p_perm"] = np.nan
        summ.append({"test": name, "description": desc, "variable": var, **s, "nuisance_auc": auc, "nuisance_variable": aucv,
                     "testable": testable, "status": status})
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


def figures(out: Path, agg: pd.DataFrame, diel: pd.DataFrame, wvt: pd.DataFrame, D: pd.DataFrame, min_direct: int,
            comp: pd.DataFrame, K6: int, sub_labels: list[str]) -> None:
    plt = _plt()
    A = agg[agg.grouping == "all"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    _line(ax, A, "d_flight", "flight incl. bout time (headline, per-bird variant)", COLORS[0])
    s = A[(A.variable == "d_flight_plain") & (A.month > 0)].sort_values("month")
    ax.plot(s.month, s["mean"], "--", color=COLORS[0], lw=1, label="flight, plain variant for all birds")
    s = A[(A.variable == "d_flight_scaled_24h") & (A.month > 0)].sort_values("month")
    ax.plot(s.month, s["mean"], ":", color=COLORS[1], lw=1.5, label="flight, first estimator (scaled_24h), same days")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("calendar month")
    ax.set_ylabel("hours per day (bird mean, 95% CI over birds)")
    ax.legend(frameon=False)
    fig.tight_layout(), fig.savefig(out / "monthly_budget.png", dpi=130), plt.close(fig)

    # stacked hours: ground states (scheduled + bout time) and flight total; K=6 flight states are not shown as hours
    ground6 = [str(i) for i in range(K6) if i in (0, 1)]
    for sysn, labs, names, fname in (("k6", ground6 + ["flight"], {**{str(k): v for k, v in K6_NAMES.items()},
                                                                    "flight": "flight total (all ACC-states)"}, "monthly_k6.png"),
                                     ("sub", sub_labels + ["ground_unassigned", "flight"],
                                      {**SUB_NAMES, "flight": "flight total"}, "monthly_substates.png")):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        bottom = np.zeros(12)
        for i, lb in enumerate(labs):
            var = "d_flight" if lb == "flight" else f"t_{sysn}_{lb}"
            s = A[(A.variable == var) & (A.month > 0)].set_index("month")["mean"].reindex(range(1, 13)).fillna(0).to_numpy()
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
    ax.set_ylabel("flight hours per hour (direct bins)")
    ax.legend(frameon=False)
    fig.tight_layout(), fig.savefig(out / "diel.png", dpi=130), plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, (gcol, title) in zip(axes.ravel(), (("sex", "sex"), ("cohort", "deployment year"),
                                                ("contrast_c2024", "cohort 2024: migrant-flagged vs young-tagged (confounded)"),
                                                ("not_stated_by_flag", "age not stated, by migrant flag"))):
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
                                        ("time_share_scaled_24h", "time share, scaled_24h (first estimator)"),
                                        ("time_share", "time share, solar-hour stratified"))):
            ax.bar(x + (i - 1.5) * 0.2, W[col], width=0.19, color=COLORS[i], label=lab)
        ax.set_xticks(x, W.state, rotation=60, fontsize=6)
        ax.set_title(sysn)
    axes[0].set_ylabel("share (bird mean)")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(), fig.savefig(out / "window_vs_time.png", dpi=130), plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(D["n_direct"], bins=np.arange(26) - 0.5, color=COLORS[0])
    for thr in sorted({min_direct, *(r[0] for r in SENS_RULES)}):
        ax.axvline(thr - 0.5, color="#333333", lw=1, ls="-" if thr == min_direct else ":")
    ax.set_xlabel("directly covered solar-hour bins per bird-day")
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
        ax.set_title("descriptive ACC-state mix of flight (K=6 clusters, not behaviours)", fontsize=7)
        ax.legend(frameon=False, fontsize=6)
        fig.tight_layout(), fig.savefig(out / "flight_composition.png", dpi=130), plt.close(fig)


# ---------------------------------------------------------------- main run

def _vof(variant: dict[str, str] | str):
    return (lambda p: variant) if isinstance(variant, str) else (lambda p: variant[p["bird"]])


def category_columns(P: list[dict]) -> dict[str, list[str]]:
    cols = set()
    for p in P:
        for H in p.get("hourly", {}).values():
            cols |= set(H.columns)
    return {s: sorted(c for c in cols if c.startswith(s + "_") and not c.endswith("_unknown")) for s in SYSTEMS}


def day_table(P: list[dict], variant: dict[str, str] | str, cats: dict[str, list[str]], pop: dict | None) -> pd.DataFrame:
    """Bird-day rows from the per-bird hourly tables: solar-hour stratified daily hours (d_*; see stratify), fill
    counts, raw covered hours (h_*), the first estimator 24 x state / covered time (s24_*, d_flight_scaled_24h), and
    coverage (also coverage_day = solar hours 06-18 covered / 12 h, coverage_night = the rest). Totals per state
    t_<system>_<label> = scheduled-share hours + bout hours of that label; d_flight / d_ground = K=2 totals; d_bout =
    bout hours (all K=2 labels). ``variant``: one
    variant name or bird -> variant; ``pop``: population_cells() of the same variant choice (None = own data only)."""
    parts, strat = [], []
    vof = _vof(variant)
    for p in P:
        v = vof(p)
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
        strat.append(stratify(H, cats, pop).assign(bird=p["bird"]))
    D = pd.concat(parts, ignore_index=True).fillna(0.0)
    sec = {c: D[c] for c in D.columns if c[:3] in ("k2_", "k6_") or c.startswith("sub_")}
    out = pd.DataFrame({"bird": D["bird"], "dayn": D["dayn"].astype(int), "variant": D["variant"],
                        "n_sched": D["n_sched"].astype(int)})
    k2cols = [c for c in sec if c.startswith("k2_")]
    tot = sum(sec[c] for c in k2cols)
    imp = sum((sec[c] for c in k2cols if is_bout(c)), 0.0)
    out["coverage"] = tot / 86400.0
    out["coverage_sched"] = (tot - imp) / 86400.0
    out["coverage_day"], out["coverage_night"] = D["_cov_day"], D["_cov_night"]
    for c, v in sec.items():
        out["h_" + c] = v / 3600.0
    for sysn in SYSTEMS:  # first estimator (scaled_24h), kept for comparison
        cols = [c for c in sec if c.startswith(sysn + "_") and not c.endswith("_unknown")]
        cls = sum(sec[c] for c in cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            for c in cols:
                out["s24_" + c] = np.where(cls > 0, 24.0 * sec[c] / cls, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cls2 = tot - sec.get("k2_unknown", 0.0) - sec.get(f"k2_{BOUT}unknown", 0.0)
        fl24 = sec.get("k2_flight", 0.0) + sec.get(f"k2_{BOUT}flight", 0.0)
        out["d_flight_scaled_24h"] = np.where(cls2 > 0, 24.0 * fl24 / cls2, np.nan)
    St = pd.concat(strat, ignore_index=True)
    out = out.merge(St, on=["bird", "dayn"], how="left", validate="one_to_one")
    for sysn in SYSTEMS:
        dc = [c for c in out.columns if c.startswith(f"d_{sysn}_")]
        known = out[[c for c in dc if not is_bout(c)]].notna().any(axis=1) if dc else pd.Series(False, index=out.index)
        for c in dc:
            if is_bout(c):
                out.loc[known & out[c].isna(), c] = 0.0
        for lb in sorted({c[len(f"d_{sysn}_"):].replace(BOUT, "", 1) for c in dc}):
            parts_ = [out[c] for c in (f"d_{sysn}_{lb}", f"d_{sysn}_{BOUT}{lb}") if c in out]
            out[f"t_{sysn}_{lb}"] = sum(parts_[1:], parts_[0]).where(known)
    out = out.copy()  # de-fragment after adding the per-state totals
    out["d_flight"] = out.get("t_k2_flight", np.nan)
    out["d_ground"] = out.get("t_k2_ground", np.nan)
    bcols = [c for c in out.columns if c.startswith(f"d_k2_{BOUT}")]
    out["d_bout"] = out[bcols].sum(axis=1, min_count=1) if bcols else 0.0
    out["d_bout_ground"] = out.get(f"d_k2_{BOUT}ground", 0.0)
    out["h_filled"] = out[["h_fill_bird_month", "h_fill_bird", "h_fill_population"]].sum(axis=1)
    date = pd.to_datetime(out["dayn"] * 86400, unit="s")
    out.insert(1, "solar_date", date.dt.strftime("%Y-%m-%d"))
    out["month"] = date.dt.month
    out["year"] = date.dt.year
    return out


def sens_rows(Qt: pd.DataFrame, tags: dict) -> list[dict]:
    """Per month (0 = all months pooled per bird, day-weighted) the bird mean of daily flight hours over birds with
    >= MIN_DAYS_CELL qualifying days in the cell (the same bird filter as budgets_by_group.csv)."""
    cols = ["d_flight", "d_flight_scaled_24h", "h_filled", "h_fill_population"]
    rows = []
    for mon in range(0, 13):
        Z = Qt if mon == 0 else Qt[Qt.month == mon]
        g = Z.groupby("bird")[cols].mean()
        n = Z.groupby("bird").size()
        x = g[n >= MIN_DAYS_CELL]
        if not len(x):
            continue
        rows.append({**tags, "month": mon, "n_birds": len(x), "n_bird_days": int(n[n >= MIN_DAYS_CELL].sum()),
                     "mean_flight_h": x.d_flight.mean(), "mean_flight_h_scaled_24h": x.d_flight_scaled_24h.mean(),
                     "mean_hours_filled": x.h_filled.mean(), "mean_hours_filled_population": x.h_fill_population.mean()})
    return rows


def run(windows: Path, states2: Path, states6: Path, substates: Path, ref_csv: Path, out: Path,
        min_direct_bins: int = MIN_DIRECT_BINS, fill: str = "population", n_boot: int = 1000, n_perm: int = 5000,
        workers: int = 1, bout_imputation: str = "auto", seed: int = 0) -> dict:
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
    dropped = pd.Series([p.get("drop_reason", "no windows") if p is not None else "no windows"
                         for p in res if p is None or not (p.get("n_kept", 0) > 0 and "hourly" in p)], dtype=object)
    k1c = pd.DataFrame([p["k1_count"] for p in res if p is not None and "k1_count" in p]).sum()
    log(f"pass done ({time.time() - t0:.0f} s); {len(P)} birds with scheduled samples, {len(dropped)} dropped")
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
    vmap = {b: "bouts" if bird_supp.get(b, suppressed) else "plain" for b in [p["bird"] for p in P]}
    variant = "per bird" if len(set(vmap.values())) > 1 else next(iter(vmap.values()))
    bout_k2 = pd.DataFrame([p["bout_k2"] for p in P]).sum()
    check = {
        "scheduled_samples": int(chk.scheduled.sum()), "k0_windows": int(chk.k0_windows.sum()),
        "scheduled_with_k_in_segment_gt0": int(chk.sched_with_k_in_segment_gt0.sum()),
        "k0_k_in_segment0_not_scheduled": int(chk.k0_kin0_not_sched.sum()),
        "scheduled_in_segment_with_k1_windows": int(chk.sched_in_segment_with_k1.sum()),
        "scheduled_inside_bout_span": int(chk.sched_inside_bout_span.sum()),
        "k0_windows_inside_bouts_excluded": int(chk.k0_windows_inside_bouts_excluded.sum()),
        "k0_burst_starts_inside_bouts_excluded": int(chk.k0_burst_starts_inside_bouts_excluded.sum()),
        "scheduled_with_gps_share": float(chk.sched_with_gps.sum() / max(chk.scheduled.sum(), 1)),
        "k2_state_hmm_agrees_with_state_on_scheduled": float(chk.sched_k2_state_vs_hmm_agree.sum() / max(chk.sched_k2_known.sum(), 1)),
        "bouts": int(chk.bouts.sum()), "bout_hours": float(chk.bout_seconds.sum() / 3600),
        "clock_slots_inside_bouts": int(sl.slots.sum()), "slots_with_scheduled_sample": int(sl.found.sum()),
        "share_slots_found_pooled": float(found_share), "share_slots_found_per_bird_median": float(per_bird_found.median()),
        "slots_in_k1_window": int(sl.slot_in_k1_window.sum()),
        "share_slot_windows_flight": float(sl.slot_window_flight.sum() / max(sl.slot_in_k1_window.sum(), 1)),
        "k1_windows_k2_share": {k: float(v / bout_k2.sum()) for k, v in bout_k2.items()},
        # I2: continuous-mode windows of ALL windows (all birds with windows, pre-fledge included); still = stage-4 rule
        "k1_recount": {k: int(v) for k, v in k1c.items()},
        # I9: scheduled samples that are short windows (acc_samples < SHORT_ACC) and/or start <= AFTER_BOUT_S after a bout
        "scheduled_short": int(chk.sched_short.sum()), "scheduled_after_bout": int(chk.sched_after_bout.sum()),
        "scheduled_short_and_after_bout": int(chk.sched_short_and_after_bout.sum()),
        "share_flight_scheduled_short": float(chk.sched_short_flight.sum() / max(chk.sched_short.sum(), 1)),
        "share_flight_scheduled_after_bout": float(chk.sched_after_bout_flight.sum() / max(chk.sched_after_bout.sum(), 1)),
        "share_flight_scheduled_other": float(chk.sched_other_flight.sum() / max(
            chk.scheduled.sum() - chk.sched_short.sum() - chk.sched_after_bout.sum() + chk.sched_short_and_after_bout.sum(), 1)),
        "schedule_suppressed_in_bouts": bool(suppressed), "bout_imputation_mode": bout_imputation, "headline_variant": variant,
        "birds_bouts": int(sum(v == "bouts" for v in vmap.values())),
        "birds_plain": int(sum(v == "plain" for v in vmap.values())),
        "birds_decided_by_pooled_rule": int(own.isna().sum()),
        "birds_with_windows": len(birds), "birds_used": len(P), "birds_dropped": {str(k): int(v) for k, v in dropped.value_counts().items()},
    }
    (out / "recording_check.json").write_text(json.dumps(check, indent=1))
    log(f"recording check: {check['share_slots_found_pooled']:.3f} of clock slots inside bouts have a scheduled sample; "
        f"{check['birds_bouts']} birds use the bouts variant, {check['birds_plain']} the plain variant")

    sch = pd.concat([p["schedule"] for p in P if "schedule" in p], ignore_index=True)
    nominal = pd.Series({p["bird"]: p.get("nominal_interval_s", np.nan) for p in P})
    sch["nominal_interval_s"] = sch.bird.map(nominal)
    sch["differs_from_nominal"] = (sch.interval_s != sch.nominal_interval_s) & (sch.n_gaps >= MIN_GAPS)
    sch.to_csv(out / "schedule.csv", index=False)

    # bird table
    B = ref.loc[[p["bird"] for p in P]].copy()
    dep_year = pd.to_datetime(B["deploy_on"], utc=True, errors="coerce").dt.year
    B["cohort_from_first_window"] = dep_year.isna().to_numpy()
    B["cohort"] = [str(int(y)) if pd.notna(y) else str(p["first_year"]) for y, p in zip(dep_year, P)]  # deployment year
    B["migrant_flag"] = B.migrant.eq(True).astype(int)
    B["young_flag"] = B.tagged_young.eq(True).astype(int)
    B["fledge"] = [p["fledge"] for p in P]
    B["n_windows"] = [p["n_windows"] for p in P]
    B["n_pre_fledge"] = [p["n_pre_fledge"] for p in P]
    B["n_pre_fledge_days"] = [p["n_pre_fledge_days"] for p in P]
    B["n_scheduled"] = chk.scheduled.reindex(B.index).to_numpy()
    B["nominal_interval_s"] = nominal.reindex(B.index).to_numpy()
    B["offset_fallback"] = [p.get("offset_fallback") for p in P]
    for v in ("plain", "bouts"):
        sp = pd.DataFrame([p["span"][v] for p in P], index=B.index)
        B[f"share_covered_scheduled_{v}"] = sp.sched_s / sp.span_s
        B[f"share_bout_{v}"] = sp.bout_s / sp.span_s
        B[f"share_gap_{v}"] = 1 - (sp.sched_s + sp.bout_s) / sp.span_s
        if v == "bouts":
            B["bout_hours_typed"], B["bout_hours_ground"] = sp.bout_s / 3600, sp.bout_ground_s / 3600
            B["bout_hours_unknown_state"] = sp.bout_unknown_s / 3600
    B["share_slots_found"] = per_bird_found.reindex(B.index).to_numpy()
    B["clock_slots_in_bouts"] = sl.slots.reindex(B.index).to_numpy()
    B["variant"] = [vmap[b] for b in B.index]
    sel = {b: vmap[b] for b in B.index}
    for c in ("share_covered_scheduled", "share_bout", "share_gap"):
        B[f"{c}_headline"] = [B.loc[b, f"{c}_{sel[b]}"] for b in B.index]
    mdiff = sch[(sch.period == "all") & (sch.year_month != "all")].groupby("bird").differs_from_nominal.sum()
    B["months_schedule_differs"] = mdiff.reindex(B.index).fillna(0).astype(int).to_numpy()
    B.drop(columns=[c for c in B.columns if "comment" in c], errors="ignore").reset_index().to_csv(out / "birds.csv", index=False)

    # bird-day table (headline variant, plus the other variant's flight and coverage); population fill cells first
    cats = category_columns(P)
    pops = {v: population_cells([p["hourly"][_vof(v)(p)] for p in P], cats) for v in ("plain", "bouts")}
    pop_head = population_cells([p["hourly"][vmap[p["bird"]]] for p in P], cats)
    D = day_table(P, vmap, cats, pop_head)
    idx = pd.MultiIndex.from_frame(D[["bird", "dayn"]])
    for v, tag in (("plain", "plain"), ("bouts", "bouts")):
        X = day_table(P, v, cats, pops[v]).set_index(["bird", "dayn"])
        D[f"d_flight_{tag}"] = X["d_flight"].reindex(idx).to_numpy()
        D[f"coverage_{tag}"] = X["coverage"].reindex(idx).to_numpy()
    for c in ("sex", "cohort", "migrant_flag", "young_flag"):
        D[c] = D.bird.map(B[c])
    D["migrant_all"] = np.where(D.migrant_flag == 1, "migrant", "not_flagged")
    # cohort 2024: migrant-flagged vs young-tagged is one confounded contrast; birds with both or neither flag are left out
    D["contrast_c2024"] = np.where((D.cohort == "2024") & ((D.migrant_flag == 1) ^ (D.young_flag == 1)),
                                   np.where(D.migrant_flag == 1, "migrant_flagged", "young_tagged"), None)
    D["not_stated_by_flag"] = np.where(D.young_flag == 0, np.where(D.migrant_flag == 1, "migrant_flagged", "not_flagged"), None)
    D["age_class"] = "not_stated"
    for b in B.index[B.young_flag == 1]:
        sel = (D.bird == b).to_numpy()
        a = reference.age_on(B.loc[b], pd.to_datetime(D.loc[sel, "dayn"] * 86400 + 43200, unit="s", utc=True), B.loc[b, "fledge"])
        D.loc[sel, "age_class"] = a["age_class"].to_numpy()
    n_prefledge_days = int((D.age_class == "pre_fledge").sum())  # boundary days only (windows dropped before)
    D = D[D.age_class != "pre_fledge"].reset_index(drop=True)
    D["qualifies"] = qualifies(D, min_direct_bins, fill)
    D.drop(columns=["dayn"]).to_csv(out / "birddays.csv", index=False)
    Q = D[D.qualifies].copy()
    rule = f">= {min_direct_bins} direct solar-hour bins, fill '{fill}'"
    log(f"{len(D):,} bird-days, {len(Q):,} qualifying ({rule})")

    # aggregates
    agg = pd.concat([aggregate(Q, g, n_boot, rng) for g in
                     (None, "sex", "age_class", "migrant_all", "contrast_c2024", "not_stated_by_flag", "cohort")], ignore_index=True)
    agg.to_csv(out / "budgets_by_group.csv", index=False)
    # month-balanced all-months values (mean of the monthly bird means) for daily flight hours
    notstated = B.index[B.young_flag == 0]
    mb_sets = [("all birds", None), ("age not stated", notstated),
               ("age not stated, migrant-flagged", B.index[(B.young_flag == 0) & (B.migrant_flag == 1)]),
               ("age not stated, not flagged", B.index[(B.young_flag == 0) & (B.migrant_flag == 0)])]
    mbal = pd.DataFrame([{"bird_set": nm, "variable": v, **month_balanced(Q, v, n_boot, rng, bs_)}
                         for nm, bs_ in mb_sets for v in ("d_flight", "d_bout")])
    mbal.to_csv(out / "month_balanced.csv", index=False)

    # diel (qualifying days only): per bird and solar hour, the mean over its DIRECT bins of the bin's hours per state
    # (bout time as itself, the scheduled share for the rest of the hour; same rule as the daily hours), so the sum
    # over hours is comparable with the daily flight hours; >= DIEL_MIN_BINS direct bins per bird-hour
    drows = []
    qd = set(zip(Q.bird, Q.dayn))
    hparts = []
    for p in P:
        H = p["hourly"][vmap[p["bird"]]]
        keep = np.array([(p["bird"], x) in qd for x in H.index // 24], bool)
        if not keep.any():
            continue
        Hk = H[keep]
        Bk2, ok2 = bin_hours(Hk, cats["k2"])
        Bk6, ok6 = bin_hours(Hk, cats["k6"])
        fl = [j for j, c in enumerate(cats["k2"]) if c in ("k2_flight", f"k2_{BOUT}flight")]
        X = pd.DataFrame(np.where(ok6[:, None], Bk6, np.nan), columns=["share_" + c for c in cats["k6"]])
        X.insert(0, "share_flight", np.where(ok2, Bk2[:, fl].sum(1), np.nan))
        X["hour"] = Hk.index % 24
        X["season"] = pd.Series(_cal_month(Hk.index // 24)).map(SEASONS).to_numpy()
        hparts.append(X.assign(bird=p["bird"]))
    Hh = pd.concat(hparts, ignore_index=True)
    vcols = [c for c in Hh.columns if c.startswith("share_")]
    for sea in ("all", "DJF", "MAM", "JJA", "SON"):
        X = Hh if sea == "all" else Hh[Hh.season == sea]
        g = X.groupby(["bird", "hour"])
        S = g[vcols].mean()
        S = S[g["share_flight"].count() >= DIEL_MIN_BINS]
        for h in range(24):
            if h not in S.index.get_level_values(1):
                continue
            V = S.xs(h, level=1)
            mean, lo, hi = boot_ci(V.to_numpy(), n_boot, rng)
            for v, a_, b_, c_ in zip(S.columns, mean, lo, hi):
                drows.append({"season": sea, "hour": h, "variable": v, "n_birds": len(V), "mean": a_, "ci_lo": b_, "ci_hi": c_})
    diel = pd.DataFrame(drows)
    diel.to_csv(out / "diel.csv", index=False)

    # window-count share vs time share. Window shares and 'time_share_covered' (pooled covered time, the first run's
    # time share) use all kept (non-pre-fledge) data of each bird; 'time_share' (stratified daily hours / 24) and
    # 'time_share_scaled_24h' (first estimator) are bird means over qualifying days (birds with >= MIN_DAYS_CELL).
    # Time shares count bout time under its state (bout_<label> folded into <label>), as window shares count bout windows.
    tot = pd.DataFrame({p["bird"]: p["hourly"][vmap[p["bird"]]].sum() for p in P}).T.fillna(0.0)
    qn = Q.groupby("bird").size()
    Qb = Q[Q.bird.isin(qn.index[qn >= MIN_DAYS_CELL])]

    def fold(X: pd.DataFrame) -> pd.DataFrame:
        return X.T.groupby(lambda c: c.replace(BOUT, "", 1)).sum().T

    def day_share(prefix: str, sysn: str) -> pd.DataFrame:
        cols = [c for c in Qb.columns if c.startswith(f"{prefix}{sysn}_") and not c.endswith("_unknown")]
        X = Qb.groupby("bird")[cols].mean() / 24.0
        X.columns = [c[len(prefix) + len(sysn) + 1:] for c in cols]
        return fold(X)

    wrows = []
    for sysn in ("k2", "k6", "sub"):
        cnt_all = pd.DataFrame({p["bird"]: p["counts"][sysn]["all"] for p in P}).T.fillna(0).drop(columns="unknown", errors="ignore")
        cnt_s = pd.DataFrame({p["bird"]: p["counts"][sysn]["sched"] for p in P}).T.fillna(0).drop(columns="unknown", errors="ignore")
        tc = tot[[c for c in tot.columns if c.startswith(sysn + "_") and not c.endswith("_unknown")]]
        tc.columns = [c[len(sysn) + 1:] for c in tc.columns]
        tc = fold(tc)
        states = sorted(set(cnt_all.columns) | set(tc.columns))
        sh = {name: (X.reindex(columns=states, fill_value=0).div(X.reindex(columns=states, fill_value=0).sum(axis=1), axis=0))
              for name, X in (("window_share_all", cnt_all), ("window_share_scheduled", cnt_s), ("time_share", day_share("t_", sysn)),
                              ("time_share_scaled_24h", day_share("s24_", sysn)), ("time_share_covered", tc))}
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

    # tests and day-rule sensitivity (headline rule first)
    log("tests")
    tsum, tper, srows = [], [], []
    rules = [(min_direct_bins, fill)] + [r for r in SENS_RULES if r != (min_direct_bins, fill)]
    for nb_, fl_ in rules:
        main_ = (nb_, fl_) == (min_direct_bins, fill)
        Qt = D[qualifies(D, nb_, fl_)]
        npm = n_perm if main_ else min(n_perm, 2000)
        s_, p_ = run_tests(Qt, B, "d_flight", npm, rng)
        tsum += [dict(r, min_direct_bins=nb_, fill=fl_, main=main_) for r in s_]
        tper += [x.assign(min_direct_bins=nb_, fill=fl_) for x in p_]
        srows += sens_rows(Qt, dict(min_direct_bins=nb_, fill=fl_, main=main_, bird_set="all qualifying birds"))
    # I7: own-data filling qualifies a different bird set; the population fill on exactly those birds
    qo = D[qualifies(D, min_direct_bins, "own")].groupby("bird").size()
    own_birds = qo.index[qo >= MIN_DAYS_CELL]
    Qp = D[qualifies(D, min_direct_bins, "population") & D.bird.isin(own_birds)]
    srows += sens_rows(Qp, dict(min_direct_bins=min_direct_bins, fill="population", main=False, bird_set="own-fill bird set"))
    srows += sens_rows(D[qualifies(D, min_direct_bins, "own") & D.bird.isin(own_birds)],
                       dict(min_direct_bins=min_direct_bins, fill="own", main=False, bird_set="own-fill bird set"))
    own_set = {"n_birds": len(own_birds), **{f"deployment_year_{k}": int(v) for k, v in B.loc[own_birds, "cohort"].value_counts().items()},
               "young_tagged": int(B.loc[own_birds, "young_flag"].sum()), "migrant_flagged": int(B.loc[own_birds, "migrant_flag"].sum())}
    tests = pd.DataFrame(tsum)
    if len(tests):  # Holm over the testable comparisons of the main day rule
        fam = tests.main & tests.testable
        tests["p_holm"] = np.nan
        tests.loc[fam, "p_holm"] = holm(tests.loc[fam, "p_perm"].to_numpy())
    tests.to_csv(out / "tests.csv", index=False)
    if tper:
        pd.concat(tper, ignore_index=True).to_csv(out / "tests_by_month.csv", index=False)
    sens = pd.DataFrame(srows)
    sens.to_csv(out / "coverage_sensitivity.csv", index=False)

    # side table: first estimator (scaled_24h) vs stratified, under the first run's coverage rule and the new rule
    crow = []
    for rname, sel in ((f"first run: coverage >= {OLD_MIN_COVERAGE}", D.coverage >= OLD_MIN_COVERAGE), (f"new: {rule}", D.qualifies)):
        X = D[sel]
        for est, col in (("scaled_24h", "d_flight_scaled_24h"), ("stratified", "d_flight")):
            Y = X[X[col].notna()]
            for mon in range(0, 13):
                Z = Y if mon == 0 else Y[Y.month == mon]
                g = Z.groupby("bird")[col].agg(["mean", "size"])
                g = g[g["size"] >= MIN_DAYS_CELL]
                if len(g):
                    crow.append({"day_rule": rname, "estimator": est, "month": mon, "n_birds": len(g),
                                 "n_bird_days": int(g["size"].sum()), "mean_flight_h": g["mean"].mean()})
    cmp24 = pd.DataFrame(crow)
    cmp24.to_csv(out / "scaled_24h_comparison.csv", index=False)
    dl_all = diel[(diel.season == "all") & (diel.variable == "share_flight")]
    head = agg[(agg.grouping == "all") & (agg.month == 0) & (agg.variable == "d_flight")]
    diel_check = {"diel_flight_share_sum_h": float(dl_all["mean"].sum()), "diel_hours_with_data": int(len(dl_all)),
                  "diel_min_birds_per_hour": int(dl_all.n_birds.min()) if len(dl_all) else 0,
                  "mean_daily_flight_h_stratified": float(head["mean"].iloc[0]) if len(head) else np.nan,
                  "mean_daily_flight_h_scaled_24h_same_days": float(agg[(agg.grouping == "all") & (agg.month == 0) &
                                                                        (agg.variable == "d_flight_scaled_24h")]["mean"].iloc[0])
                  if len(head) else np.nan}

    log("figures")
    figures(out, agg, diel, wvt, D, min_direct_bins, comp, K6, sub_labels)
    meta = {"windows": str(windows), "states2": str(states2), "states6": str(states6), "substates": str(substates),
            "estimator": "solar-hour stratified", "min_direct_bins": min_direct_bins, "fill": fill, "day_rule": rule,
            "n_boot": n_boot, "n_perm": n_perm, "seed": seed, "headline_variant": variant, "diel_check": diel_check,
            "bout_definition": "aquilacl.flight.bout_ids on k == 1 windows (shared with flight.py), all durations",
            "constants": {k: globals()[k] for k in ("BURST_GAP_S", "CAP_FACTOR", "ROUND_S", "MIN_GAPS",
                                                    "SLOT_TOL_S", "SUPPRESSION_FOUND_MAX", "MIN_DAYS_CELL", "MIN_TEST_DAYS",
                                                    "MIN_GROUP_BIRDS", "DIEL_MIN_BINS", "COMP_MIN_S", "MIN_BIN_S",
                                                    "POP_MIN_BIRDS", "OLD_MIN_COVERAGE", "STILL_VEDBA_G", "SHORT_ACC",
                                                    "AFTER_BOUT_S", "NUISANCE_AUC_MAX", "NON_MIGRATION_MONTHS")},
            "birds": len(B), "bird_days": len(D), "bird_days_qualifying": len(Q),
            "pre_fledge_boundary_days_dropped": n_prefledge_days,
            "pre_fledge_solar_days_with_windows": int(B.n_pre_fledge_days.sum()),
            "pre_fledge_windows_dropped": int(B.n_pre_fledge.sum()),
            "cohort_from_first_window_year": int(B.cohort_from_first_window.sum()),
            "birds_by_deployment_year": {str(k): int(v) for k, v in B.cohort.value_counts().sort_index().items()},
            "sex_counts": {str(k): int(v) for k, v in B.sex.fillna("missing").value_counts().items()},
            "own_fill_bird_set": own_set,
            "month_balanced_flight_h": mbal[(mbal.bird_set == "all birds") & (mbal.variable == "d_flight")].iloc[0].to_dict()
            if len(mbal) else {}, "seconds": round(time.time() - t0)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    write_summary(out, meta, check, B, sch, D, Q, agg, diel, wvt, comp, tests, sens, k2k6, K6, sub_labels, cmp24, mbal)
    log(f"done ({time.time() - t0:.0f} s)")
    return meta


# ---------------------------------------------------------------- summary

def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_summary(out, meta, check, B, sch, D, Q, agg, diel, wvt, comp, tests, sens, k2k6, K6, sub_labels, cmp24,
                  mbal=None) -> None:
    A = agg[(agg.grouping == "all")]
    k1 = check.get("k1_recount", {})
    drops = check.get("birds_dropped", {})
    L = ["# Time budgets from the scheduled recording stream", "",
         "All numbers below are generated by `aquilacl.budgets`. Everything is DESCRIPTIVE except rows marked 'tested' "
         "in the section 'Tests'.", "",
         "## Why time budgets and not window shares", "",
         "Continuous mode (1 Hz GPS) starts when the bird moves fast and then yields one window every 10 s, while a scheduled "
         "burst yields one window per interval. Window shares therefore over-count flight. Here each scheduled sample "
         "represents the time until the next scheduled sample (at most 1.5 x the local schedule interval); bout time not "
         "covered by the schedule is classified by the state of its own continuous-mode windows (not imputed as flight).", "",
         "Daily hours are stratified by local solar hour: per bird-day, the state shares of the covered time in each of "
         "the 24 hour bins, summed over the bins (x 1 h). Bins without direct coverage are filled from the bird's own "
         "bird-month mean for that hour, then its all-month mean for that hour, then the population mean (calendar month x "
         "hour, then hour). The first run scaled the day's covered time to 24 h ('scaled_24h'); because most tags do not "
         "record at night and the night is almost all ground, that inflated daily flight hours. It is kept for comparison.", "",
         "## Birds", "",
         f"- {check.get('birds_with_windows', 'n/a')} birds with windows; {meta['birds']} used; dropped: "
         + (", ".join(f"{v} ({k})" for k, v in drops.items()) if drops else "none") + ".",
         "- Deployment year (cohort; year of the earliest reference deploy_on): "
         + ", ".join(f"{k}: {v} birds" for k, v in meta["birds_by_deployment_year"].items())
         + f"; {meta['cohort_from_first_window_year']} birds without deploy_on use their first-window year.",
         "- Sex among used birds: " + ", ".join(f"{k} {v}" for k, v in meta["sex_counts"].items()) + ".",
         f"- Young-tagged: {int(B.young_flag.sum())} birds; migrant-flagged: {int(B.migrant_flag.sum())}; both flags: "
         f"{int(((B.young_flag == 1) & (B.migrant_flag == 1)).sum())}. Deployment year 2024: "
         f"{int(((B.cohort == '2024') & (B.migrant_flag == 1)).sum())} migrant-flagged, "
         f"{int(((B.cohort == '2024') & (B.young_flag == 1)).sum())} young-tagged, "
         f"{int(((B.cohort == '2024') & (B.migrant_flag == 0) & (B.young_flag == 0)).sum())} with neither flag.",
         "- Deploy months (birds): " + "; ".join(f"{nm} {_month_counts(B, sel)}" for nm, sel in (
             ("migrant-flagged", B.migrant_flag == 1), ("young-tagged", B.young_flag == 1),
             ("neither flag", (B.migrant_flag == 0) & (B.young_flag == 0)))) + ".",
         f"- Pre-fledge data of young-tagged birds dropped before anything else: {meta['pre_fledge_windows_dropped']:,} windows "
         f"on {meta['pre_fledge_solar_days_with_windows']:,} solar days (bird-days); {meta['pre_fledge_boundary_days_dropped']} "
         "further boundary days (fledge day) dropped from the day table.", "",
         "## Recording structure (measured)", "",
         f"- Scheduled samples: {check['scheduled_samples']:,} (first window of each k=0 burst) from "
         f"{check['k0_windows']:,} k=0 windows. {check['scheduled_with_k_in_segment_gt0']:,} scheduled samples have "
         f"k_in_segment > 0; {check['k0_k_in_segment0_not_scheduled']:,} k=0 windows with k_in_segment = 0 are not "
         "first of a burst.",
         f"- Share of scheduled samples with any GPS fix: {_f(check['scheduled_with_gps_share'], 3)}.",
         f"- k=0 windows inside bout spans: {check['k0_windows_inside_bouts_excluded']:,} "
         f"({check['k0_burst_starts_inside_bouts_excluded']:,} burst starts). They are recording-format interruptions of "
         "continuous mode (almost never on the bird's clock; see flight.bout_ids) and are EXCLUDED from the scheduled "
         f"samples; scheduled samples left inside bout spans: {check['scheduled_inside_bout_span']:,}.",
         f"- Scheduled samples that are short windows (acc_samples < {SHORT_ACC}): {check['scheduled_short']:,}; starting "
         f"<= {AFTER_BOUT_S:.0f} s after a bout end: {check['scheduled_after_bout']:,}; both: "
         f"{check['scheduled_short_and_after_bout']:,}. K=2 flight share: short {_f(check['share_flight_scheduled_short'], 3)}, "
         f"after a bout {_f(check['share_flight_scheduled_after_bout'], 3)}, all other scheduled samples "
         f"{_f(check['share_flight_scheduled_other'], 3)}. Scheduled bursts are short by design on most tags, so "
         "'short' does not mark continuous-run tails; the after-bout samples are the ones that may be tails.",
         f"- Nominal interval per bird: " + ", ".join(f"{int(k)} s: {n} birds" for k, n in
                                                     B.nominal_interval_s.value_counts().sort_index().items()) + ".",
         f"- Bird-months whose interval differs from the bird's nominal interval: "
         f"{int(sch[(sch.period == 'all') & (sch.year_month != 'all')].differs_from_nominal.sum())}; "
         f"bird-months where day and night intervals differ: {_daynight_diff(sch)}.",
         f"- Gaps at the interval (per-bird median share): {_f(sch[sch.year_month == 'all'].share_at_interval.median(), 3)}; "
         f"at a multiple (missed bursts): {_f(sch[sch.year_month == 'all'].share_multiple.median(), 3)}; shorter: "
         f"{_f(sch[sch.year_month == 'all'].share_shorter.median(), 3)}.",
         f"- Continuous bouts ({meta['bout_definition']}): {check['bouts']:,}, {check['bout_hours']:,.0f} h. Of "
         f"{check['clock_slots_inside_bouts']:,} expected clock slots inside bouts, {check['slots_with_scheduled_sample']:,} "
         f"({_f(check['share_slots_found_pooled'], 3)} pooled; per-bird median {_f(check['share_slots_found_per_bird_median'], 3)}) "
         "have a scheduled sample. A missing slot means the burst was suppressed or absorbed into the continuous segment; "
         "the metadata cannot tell which.",
         f"- Continuous-mode (k=1) windows, all birds with windows, all data: {k1.get('k1_windows', 0):,}; full (200 samples) "
         f"{k1.get('k1_full', 0):,}; 'still' (stage-4 rule: label_perched = 1 or VeDBA < {STILL_VEDBA_G} g) among full "
         f"{k1.get('k1_full_still', 0):,}, among all {k1.get('k1_still', 0):,}; K=2 ground {k1.get('k1_k2_ground', 0):,} of "
         f"{k1.get('k1_k2_known', 0):,} with a state ({_f(k1.get('k1_k2_ground', 0) / max(k1.get('k1_k2_known', 0), 1), 4)}); "
         f"full and K=2 ground {k1.get('k1_full_k2_ground', 0):,}. (The earlier '25 of 91,913' came from a window subsample "
         "of the stage-4 diagnosis, scripts/diagnose_r4.py, not from all windows; quote these counts instead.)",
         f"- The k=1 window covering a clock slot is K=2 flight in {_f(check['share_slot_windows_flight'], 3)} of "
         f"{check['slots_in_k1_window']:,} slots.",
         f"- K=2 state_hmm agrees with the GMM state on {_f(check['k2_state_hmm_agrees_with_state_on_scheduled'], 4)} of scheduled samples.",
         f"- **Headline variant per bird** (mode {check['bout_imputation_mode']}): {check['birds_bouts']} birds 'bouts' "
         f"(bout time not covered by a scheduled sample is classified by its own k=1 windows' states, categories 'bout_*'), "
         f"{check['birds_plain']} birds 'plain' (bouts ignored). A bird's own share of found slots decides (< "
         f"{SUPPRESSION_FOUND_MAX} -> bouts); {check['birds_decided_by_pooled_rule']} birds with < {MIN_SLOTS_BIRD} "
         f"slots follow the pooled share, which judges the schedule "
         f"{'SUPPRESSED' if check['schedule_suppressed_in_bouts'] else 'continuing'} inside bouts.",
         f"- Bout time (bouts variant, all typed by the bout's own windows): {_f(B.bout_hours_typed.sum(), 0)} h, of which K=2 ground "
         f"{_f(B.bout_hours_ground.sum(), 0)} h and without a state {_f(B.bout_hours_unknown_state.sum(), 0)} h.",
         f"- Elapsed time per bird (first to last sample), median shares: covered by scheduled samples "
         f"{_f(B['share_covered_scheduled_headline'].median(), 3)}, bout time {_f(B['share_bout_headline'].median(), 3)}, "
         f"gap {_f(B['share_gap_headline'].median(), 3)}.", "",
         "## Bird-days, coverage and filling", "",
         f"- {meta['birds']} birds, {meta['bird_days']:,} bird-days (local mean solar days).",
         f"- Night recording: median coverage of all bird-days: solar 06-18 h {_f(D.coverage_day.median(), 3)}, 18-06 h "
         f"{_f(D.coverage_night.median(), 3)}. A solar-hour bin is direct if it has >= {MIN_BIN_S:.0f} s of classified "
         "covered time. Direct bins per bird-day: "
         + ", ".join(f"{h}h {_f(x, 3)}" for h, x in _direct_by_hour(D).items()) + " (share of bird-days with the hour direct).",
         f"- Rule: a bird-day qualifies if >= {meta['min_direct_bins']} of its 24 solar-hour bins are direct and every other "
         f"bin can be filled (fill '{meta['fill']}'): {meta['bird_days_qualifying']:,} bird-days from {Q.bird.nunique()} birds.",
         f"- Hours per qualifying day (median / mean): direct {_f(Q.n_direct.median(), 0)} / {_f(Q.n_direct.mean(), 1)}; "
         f"filled from the bird-month {_f(Q.h_fill_bird_month.mean(), 2)} (mean); from the bird's all-month hour mean "
         f"{_f(Q.h_fill_bird.mean(), 2)}; from the POPULATION {_f(Q.h_fill_population.median(), 0)} / "
         f"{_f(Q.h_fill_population.mean(), 2)}. Flight hours contributed by the population fill: mean "
         f"{_f(Q.d_flight_from_population.mean(), 3)} h per day (max {_f(Q.d_flight_from_population.max(), 3)}); by the "
         f"bird's all-month hour mean: {_f(Q.d_flight_from_bird_allmonth.mean(), 3)} h.",
         f"- Check against the diel profile: sum over hours of the bird-mean flight share = "
         f"{_f(meta['diel_check']['diel_flight_share_sum_h'])} h ({meta['diel_check']['diel_hours_with_data']} hours with data, "
         f"at least {meta['diel_check']['diel_min_birds_per_hour']} birds per hour; night hours come from fewer birds, so the "
         f"sum mixes bird sets).", "",
         "Day-rule sensitivity. Mean daily flight h = bird mean of per-bird means over all qualifying days (day-weighted "
         f"within a bird), birds with >= {MIN_DAYS_CELL} qualifying days (the same filter as the 'all months, day-weighted' "
         "row below). The own-fill rule qualifies a DIFFERENT bird set; the 'own-fill bird set' rows compare the two fills "
         "on the same birds.", "",
         "| min direct bins | fill | bird set | bird-days | birds | mean daily flight h | same days, scaled_24h | mean hours filled | of which population |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sens[sens.month == 0].iterrows():
        L.append(f"| {int(r.min_direct_bins)}{' (main)' if r.main else ''} | {r.fill} | {r.bird_set} | {int(r.n_bird_days):,} | "
                 f"{int(r.n_birds)} | {_f(r.mean_flight_h)} | {_f(r.mean_flight_h_scaled_24h)} | {_f(r.mean_hours_filled)} | "
                 f"{_f(r.mean_hours_filled_population)} |")
    os_ = meta.get("own_fill_bird_set", {})
    L += ["", "Own-fill bird set: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in os_.items()) + ".", ""]
    # I8: recording window of migrant-flagged birds in May-August (far north and west: solar time shifts)
    L += ["Direct solar-hour bins in May-August, share of bird-days (I8: a tag clock fixed in UTC shifts in solar time "
          "when a bird is far north-west; hours not recorded there are filled from the bird-month, then the bird's "
          "all-month hour mean, which for migrants is mostly winter in the basin):", ""]
    for nm, sel in (("migrant-flagged", D.migrant_flag == 1), ("not flagged", D.migrant_flag == 0)):
        X = D[sel & D.month.between(5, 8)]
        dh = _direct_by_hour(X)
        if dh:
            qx = X[X.qualifies]
            L.append(f"- {nm} ({X.bird.nunique()} birds, {len(X):,} bird-days): " + " ".join(f"{h}h {_f(v, 2)}" for h, v in dh.items())
                     + f". Qualifying days: hours filled from the bird's all-month mean {_f(qx.h_fill_bird.mean(), 2)} / day, "
                     f"flight from that fill {_f(qx.d_flight_from_bird_allmonth.mean(), 3)} h / day.")
    L += ["", "First estimator vs stratified (bird mean of daily flight hours, all months; birds with >= "
          f"{MIN_DAYS_CELL} days; full monthly rows in scaled_24h_comparison.csv):", "",
          "| day rule | estimator | birds | bird-days | flight h |", "|---|---|---|---|---|"]
    for _, r in cmp24[cmp24.month == 0].iterrows():
        L.append(f"| {r.day_rule} | {r.estimator} | {int(r.n_birds)} | {int(r.n_bird_days):,} | {_f(r.mean_flight_h)} |")
    # headline
    L += ["", "## Daily flight hours: headline", ""]
    if mbal is not None and len(mbal):
        L += ["MONTH-BALANCED = mean of the 12 monthly bird means (each month counts once; CI from resampling birds). "
              "DAY-WEIGHTED = per-bird mean over all its qualifying days, then the bird mean; it over-weights the months "
              "in which most birds were tracked (October-November).", "",
              "| bird set | variable | birds | months | month-balanced h [95% CI] |", "|---|---|---|---|---|"]
        for _, r in mbal.iterrows():
            L.append(f"| {r.bird_set} | {'flight incl. bout time' if r.variable == 'd_flight' else 'bout time (all K=2 states)'} | "
                     f"{int(r.n_birds)} | {int(r.months)} | {_f(r['mean'])} [{_f(r.ci_lo)}, {_f(r.ci_hi)}] |")
        L.append("")
    L += ["## Daily budget by month (all birds; bird mean, 95% CI over birds)", "",
          "Bird sets differ between months (birds per month in the table). Row 'all' is DAY-WEIGHTED (see the headline).", "",
          "| month | birds | flight h (headline) | ground h | flight h, plain for all birds | flight h, bouts for all birds | "
          "bout time h (of which ground) | flight h, scaled_24h (first estimator) | hours filled (population) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for mon in range(0, 13):
        g = A[A.month == mon].set_index("variable")
        if "d_flight" not in g.index:
            continue
        r = g.loc["d_flight"]
        gv = lambda v: g.loc[v, "mean"] if v in g.index else np.nan  # noqa: E731
        L.append(f"| {'all (day-weighted)' if mon == 0 else mon} | {int(r.n_birds)} | {_f(r['mean'])} [{_f(r.ci_lo)}, {_f(r.ci_hi)}] | "
                 f"{_f(gv('d_ground'))} | {_f(gv('d_flight_plain'))} | {_f(gv('d_flight_bouts'))} | "
                 f"{_f(gv('d_bout'))} ({_f(gv('d_bout_ground'))}) | {_f(gv('d_flight_scaled_24h'))} | {_f(gv('h_filled'), 1)} "
                 f"({_f(gv('h_fill_population'), 1)}) |")
    # I3: age-not-stated birds, split by the migrant flag
    G = agg[(agg.variable == "d_flight") & (((agg.grouping == "age_class") & (agg.group == "not_stated")) |
                                            (agg.grouping == "not_stated_by_flag"))]
    if len(G):
        L += ["", "Monthly daily flight hours for birds whose AGE IS NOT STATED (young-tagged birds excluded), and split by "
              "the reference migrant flag (a flag, not a track-derived residency class; track-derived classes are not "
              "used here). Cells: mean [95% CI] (birds).", "",
              "| month | age not stated | not stated, migrant-flagged | not stated, not flagged |", "|---|---|---|---|"]
        cell = lambda gr, gp, m: G[(G.grouping == gr) & (G.group == gp) & (G.month == m)]  # noqa: E731
        for mon in range(0, 13):
            cs = [cell("age_class", "not_stated", mon), cell("not_stated_by_flag", "migrant_flagged", mon),
                  cell("not_stated_by_flag", "not_flagged", mon)]
            if not any(len(c) for c in cs):
                continue
            L.append(f"| {'all (day-weighted)' if mon == 0 else mon} | " + " | ".join(
                f"{_f(c['mean'].iloc[0])} [{_f(c.ci_lo.iloc[0])}, {_f(c.ci_hi.iloc[0])}] ({int(c.n_birds.iloc[0])})" if len(c) else "n/a"
                for c in cs) + " |")
    g = A[A.month == 0].set_index("variable")
    gt = lambda v: _f(g.loc[v, "mean"]) if v in g.index else "n/a"  # noqa: E731
    L += ["", "Hours per day, all months (day-weighted bird means): flight total " + gt("d_flight") + "; "
          + "; ".join(f"{K6_NAMES[i]} {gt(f't_k6_{i}')}" for i in (0, 1) if f"t_k6_{i}" in g.index) + ".",
          "Ground sub-states, hours per day: " + "; ".join(f"{c[6:]} ({SUB_NAMES.get(c[6:], '')}) {_f(g.loc[c, 'mean'])}"
                                                           for c in g.index if c.startswith("t_sub_") and c[6:] in SUB_NAMES) + ".",
          "K=6 FLIGHT states are not reported as behaviour hours. Scheduled samples have GPS in "
          f"{_f(check['scheduled_with_gps_share'], 3)} of cases, so their flight states rest on acceleration alone, and "
          f"state 5 (named '{K6_STAGE6.get(5, '')}' in stage 6) holds {_comp_share(comp, '5')} of flight "
          "against about 2 % flapping in published accounts. The flight composition section gives the K=6 mix only as a descriptive "
          "ACC-state mix. Ground-state hours at night come mostly from the POPULATION fill (few birds record at night).", "",
          "## Diel budget (local mean solar hour, all seasons)", ""]
    dl = diel[(diel.season == "all") & (diel.variable == "share_flight")].sort_values("hour")
    L.append("Flight share of each solar hour, direct bins of qualifying days (birds): " + ", ".join(f"{int(r.hour)}h {_f(r['mean'])} ({int(r.n_birds)})"
                                                                         for _, r in dl.iterrows()) + ".")
    L += ["", "## Groups (daily flight hours, all months pooled per bird, day-weighted; descriptive)", "",
          "| grouping | group | birds | flight h [95% CI] |", "|---|---|---|---|"]
    for gcol in ("sex", "age_class", "cohort", "migrant_all", "contrast_c2024", "not_stated_by_flag"):
        for _, r in agg[(agg.grouping == gcol) & (agg.month == 0) & (agg.variable == "d_flight")].iterrows():
            L.append(f"| {'deployment year' if gcol == 'cohort' else gcol} | {r.group} | {int(r.n_birds)} | "
                     f"{_f(r['mean'])} [{_f(r.ci_lo)}, {_f(r.ci_hi)}] |")
    L += ["", "Pooling months mixes seasons: groups differ in which months they were tracked and deployed (deploy months "
          f"above). Use the monthly rows in budgets_by_group.csv.", "",
          "## Window-count share vs time share (old bias)", "",
          "Time share = stratified daily hours / 24 (qualifying days; bout time counted under its state); scaled_24h = first "
          "estimator on the same days; covered = pooled covered time over all data (the first run's 'time share', biased "
          "towards daytime). K=6 flight states are ACC clusters, not behaviours.", "",
          "| system | state | window share (all) | window share (scheduled) | time share | time share, scaled_24h | "
          "time share, covered | ratio all/time |", "|---|---|---|---|---|---|---|---|"]
    for _, r in wvt.iterrows():
        nm = K6_NAMES.get(int(r.state), r.state) if r.system == "k6" and str(r.state).isdigit() else r.state
        L.append(f"| {r.system} | {nm} | {_f(r.window_share_all, 3)} | {_f(r.window_share_scheduled, 3)} | "
                 f"{_f(r.time_share, 3)} | {_f(r.time_share_scaled_24h, 3)} | {_f(r.time_share_covered, 3)} | "
                 f"{_f(r.ratio_window_all_to_time)} |")
    L += ["", "## Flight composition: descriptive ACC-state mix (not behaviours)", "",
          "Shares of K=6 states within K=2 flight. The K=6 flight states are acceleration clusters; they are not validated "
          f"as behaviours (state 5 was named '{K6_STAGE6.get(5, '')}' in stage 6 but is far more common than flapping in "
          "published accounts; scheduled samples almost never have GPS). Continuous-bout composition is CONDITIONAL ON "
          "BEING IN A TRIGGERED BOUT (fast movement).", "",
          "| source | state | birds | share [95% CI] |", "|---|---|---|---|"]
    for _, r in comp[comp.month == 0].iterrows():
        L.append(f"| {r.source} | {r['name']} | {int(r.n_birds)} | {_f(r['mean'], 3)} [{_f(r.ci_lo, 3)}, {_f(r.ci_hi, 3)}] |")
    L += ["", "Scheduled time (hours, plain weights) by K=2 (rows) x K=6 (columns): consistency check.", "",
          k2k6.round(1).to_markdown() if _has_tabulate() else k2k6.round(1).to_string(), "",
          "## Tests (bird is the unit; daily flight hours)", "",
          f"Per month: difference of bird means (bird-months with >= {MIN_TEST_DAYS} qualifying days; months with >= "
          f"{MIN_GROUP_BIRDS} birds per group). Combined: weighted mean of monthly differences; p from permuting labels "
          "between birds (one permutation for all months; 'within deployment year' keeps labels inside each year; p is n/a "
          "when no stratum holds both labels). Nuisance AUC: largest rank AUC of the label on deploy day-of-year, "
          f"deployment year and nominal interval; at >= {NUISANCE_AUC_MAX} the comparison is NOT TESTABLE IN THIS DESIGN "
          "and only its difference, Hedges g and bird counts are given. Holm runs over the testable comparisons of the main "
          "day rule.", "",
          "| comparison | day rule | birds (1 / 0) | months | combined diff (h) | median monthly Hedges g | nuisance AUC (variable) "
          "| status | p | p Holm | permutations |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        pt = fmt_p(r.p_perm, r.n_perm) if r.testable else "not given" if r.status.startswith("not testable") else "n/a"
        ph = _f(r.p_holm, 4) if r.testable and r.main and "p_holm" in r else "-"
        L.append(f"| {r.description} | {int(r.min_direct_bins)}, {r.fill}{' (main)' if r.main else ''} | {r.n_birds_1} / {r.n_birds_0} | "
                 f"{r.months_used} | {_f(r.combined_diff_h)} | {_f(r.median_monthly_hedges_g)} | {_f(r.nuisance_auc)} "
                 f"({r.nuisance_variable}) | {r.status} | {pt} | {ph} | {r.n_perm} |")
    L += ["", "The cohort-2024 contrast is ONE comparison: migrant-flagged birds and young-tagged birds (tagged as "
          "nestlings) were deployed in different months (row text), so age, residency and deployment month cannot be "
          "separated; birds with neither flag are left out. A residency comparison on track-derived classes (same deployment year, "
          "non-migration months) is deferred to residency.py and is not imported here. Read the sex comparison only "
          "through its Holm-corrected p.", "",
          "## Caveats", "",
          "- The trigger rule is not known from the firmware; it is inferred from speeds at bout onset.",
          "- Bout time is typed by the K=2 / K=6 / sub-state of its own k=1 windows; bout windows without a state leave "
          "their time to the scheduled share of the hour.",
          "- Most tags do not record from about 20:00 to 05:00 local solar time, so night bins of most birds are filled "
          "from the population (birds that record at night). Night flight is about zero where it is recorded, so flight "
          "hours barely depend on this; ground sub-state and K=6 ground hours at night do, and they are not "
          "group-comparable. With own-data filling only, a different, smaller bird set qualifies (table above).",
          "- Fill shares come from scheduled time only (bout time is not part of them), so a filled DAYTIME bin "
          f"carries slightly less bout flight than a recorded one; daytime (05-19 h) bins filled per qualifying day: "
          f"{_f(_daytime_filled(Q), 2)} (mean).",
          "- Migrant-flagged birds in summer record at earlier solar hours (far north-west); their unrecorded evening hours "
          "are filled from their own data (see the May-August direct-bin table).",
          "- Stratification removes the day/night coverage bias within a day; day length itself does not bias daily hours "
          f"(each hour bin counts 1 h). Monthly bird sets differ ({_nb_range(A)} birds per month); the age-not-stated table "
          "removes the young-tagged birds, who enter in their hatch summer.",
          "- The 1.5 x interval cap leaves long data gaps uncovered; those bins are filled or the day does not qualify; "
          "the 8 / 14 direct-bin and own-fill sensitivity is reported.",
          f"- Non-migration months for the contrast ({', '.join(str(m) for m in NON_MIGRATION_MONTHS)}) are an assumption, "
          "not measured per bird.",
          "- Sex differences may reflect breeding roles, ages and the months each bird was tracked.",
          "- Ground state names are the stage-6 labels of the K=6 run; flight states carry neutral names; ground sub-states "
          "come from substates_r6_k2.", ""]
    (out / "summary.md").write_text("\n".join(L))


def _nb_range(A: pd.DataFrame) -> str:
    x = A[(A.variable == "d_flight") & (A.month > 0)].n_birds
    return f"{int(x.min())}-{int(x.max())}" if len(x) else "n/a"


def _comp_share(comp: pd.DataFrame, k6: str) -> str:
    x = comp[(comp.month == 0) & (comp.k6.astype(str) == k6)] if len(comp) else comp
    return "; ".join(f"{r['mean']:.2f} of {r.source}" for _, r in x.iterrows()) or "n/a"


def _month_counts(B: pd.DataFrame, sel) -> str:
    m = pd.to_datetime(B.loc[sel, "deploy_on"], utc=True, errors="coerce").dt.month.dropna().astype(int).value_counts().sort_index()
    return "(" + ", ".join(f"month {k}: {v}" for k, v in m.items()) + ")" if len(m) else "(n/a)"


def _direct_by_hour(D: pd.DataFrame) -> dict:
    """Share of bird-days with each solar hour directly covered (from the 24-character direct_mask)."""
    M = np.array([[c == "x" for c in m] for m in D["direct_mask"].astype(str) if len(m) == 24], bool)
    return {h: float(M[:, h].mean()) for h in range(24)} if len(M) else {}


def _daytime_filled(Q: pd.DataFrame) -> float:
    M = np.array([[c == "x" for c in m] for m in Q["direct_mask"].astype(str) if len(m) == 24], bool)
    return float((~M[:, 5:20]).sum(1).mean()) if len(M) else np.nan


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
    ap.add_argument("--min-direct-bins", type=int, default=MIN_DIRECT_BINS,
                    help="bird-day rule: directly covered solar-hour bins (of 24) needed")
    ap.add_argument("--fill", choices=["population", "own"], default="population",
                    help="headline filling of uncovered hour bins: own data, then population (default) or own data only")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--bout-imputation", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.windows, a.states2, a.states6, a.substates, a.reference, a.out, a.min_direct_bins, a.fill, a.n_boot, a.n_perm,
        a.workers, a.bout_imputation, a.seed)


if __name__ == "__main__":
    main()
