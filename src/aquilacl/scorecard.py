"""Goal scorecard (docs/SCORECARD_spec.md, v2 of 2026-11-27): every design is scored on one fixed list of rows.

    python -m aquilacl.scorecard --run RUN_DIR [--run ...] --windows W --split SPLIT --reference REF --states2 STATES_DIR
        --residency RES_DIR --out OUT_DIR [--rows G1 G2 ...] [--final] [--workers N] [--device cuda|cpu] [--n-perm 200]

Outputs. One run: OUT_DIR/scorecard.json, OUT_DIR/scorecard.md and OUT_DIR/boot/*.parquet (per-row test scores for
the paired bird bootstrap of scripts/compare_scorecards.py). Several runs: the same under OUT_DIR/<run name>/. OUT_DIR
must be new or empty. The handcrafted references and nuisance floors do not depend on the run; they are computed once
per row and stored under OUT_DIR/../scorecard_reference/<key>/<row>.json (created exclusively, never overwritten;
reused when present), so every design shares the same constants. <key> hashes the split, the data paths, the frozen
injection list, the mode (search or final) and the sample sizes below.

Birds. split.json {train, search, locked}. Search mode (default): locked birds are never loaded. Window rows (G2, G6,
G7*, G9, G9b, G10) train their probe on training birds and test on the search birds; bird and bird-day rows use every
non-locked bird with windows (repeated grouped CV). --final: locked birds replace search birds as the test set of the
window rows; bird rows fit on the non-locked birds and score the locked birds once (direction only, no CV); G8b and
G8c are not scored in final mode.

Fixed draws: every random draw of row Gx uses numpy.random.default_rng([20261127, ROW_IDS[Gx], ...]), drawn from
metadata only, so the model, the reference, the floor and every design see the same windows, folds and resamples.
ROW_IDS numbers the rows in the order of the spec's goal table (G1 = 1, G1b = 2, ...).

Linear models: logistic regression (class_weight balanced) or ridge, median-imputed and standardised, C = alpha = 1.0.
Bird-level CV is a copy of demography.cv_once with C fixed at 1.0 (demography.run_classifiers selects C by inner CV and
cannot permute within strata); folds come from demography.grouped_folds, permutations from demography._permuted.
Permutation p values (n_perm, default 200) are computed for the repeated-CV rows (G1, G1b, G3, G8); window rows report
no p (their uncertainty is the paired bird bootstrap in compare_scorecards). G4/G4b use demography.sign_flip.

Representations. h = embeddings/<bird>.npz 'h' (L2-normalised per window before any bird / day / month mean, as in
demography and states). Time/position-informed designs (design.json uses_time) are scored on 'h_notime' in the rows
G2, G3, G4, G4b, G5, G7c, G8, G8b when that key exists, else those rows carry "time/position-informed: not
comparable". Environment designs (uses_env) carry "environment-informed: not comparable" on the same rows plus G9.
G6 and G9b re-embed windows with the GPS mask zeroed: latest checkpoint loaded as evaluate._latest_model, acceleration
prepared as in dataset.load (common_format with config.json step_g/clip_g in the sensor frame, then the rotation R of
the run's orientation.json, which includes the yaw for derotate_yaw runs), GPS values and mask all zero. Level-2
runs (design.json level 2, parent_run): G6/G9b re-embed EVERY window with the parent's checkpoint (GPS zeroed) and
pass the parent h streams through seq.contextual_embed; G7 makes each injection in the parent h stream of the host
bird (a crop of +-max_len windows around the bout or day; donor day times shifted to the host day), re-runs the crop
through seq.contextual_embed (notime=True on the time/position rows) and uses the same crop without injection as the
clean query. Such rows are labelled in their flags.
Handcrafted reference (sensor frame): evaluate._feature_matrix (baseline.FEATURES + no-GPS flag) unless the row says
otherwise; G6 and G9b use the acceleration-only columns ACC_FEATURES; G7 uses the standardised feature matrix,
L2-normalised per window, in place of h. Own-frame references (descriptive in the spec) are not computed.

Row choices where the spec leaves room (also listed in scorecard.json "choices"):
- G1: a mode counts for a bird when it has >= MODE_MIN flight windows; bird needs sex and >= G1_MIN_FLIGHT flight
  windows. Turn radius = speed_mean / radians(turn_abs_mean) (turn > 0), as residency.py. G1b uses >= G1_MIN_FLIGHT
  flight windows in that mode; its scheduled-mode reference has no kinematic medians (no GPS in k=0).
- Scheduled domain (G2, G3, G4, G4b, G5, G7c incl. its day PCA and eligible days): k=0 windows with acc_samples >=
  SCHED_MIN (40), not only full windows: scheduled bursts are mostly 40 or 64 samples by design (spec note 2026-11-27).
  Continuous-mode / full-window rows (G6, G9b, G10, flapping) keep FULL (190).
- G2: scheduled domain, sun > 0, measured local_solar_hour; up to 2000 windows per bird on both sides. Floor:
  gps_fixes, static_norm, acc_samples (window length).
- G3 floor: daily window count, share of 40-sample, 64-sample and >= 190-sample windows, continuous-mode share,
  days since deployment, deployment year, young-tagged, movement class, daily mean static_norm.
- G3: local solar day (anomaly.local_days); bird weights 1/days; null = an independent random rotation of each bird's
  day-of-year angle (labels cannot be swapped between birds with different numbers of days).
- G4: per bird, AUC that same-month-different-year pairs are closer (cosine distance) than different-year pairs >= 3
  calendar months apart; birds need >= 1 pair of each. G4b (descriptive): per bird and year pair with >= 2 shared
  calendar months, mean pairwise cosine between the month-matched year differences (consistency of a year offset).
- G5: year 1 / year 2 = reference.age_on classes juvenile_y1 / y2 (pre-fledge windows dropped); direction =
  logistic regression y2 vs y1 on the other young birds' bird-months (leave one bird out); per bird the share of
  shared calendar months whose y2 month scores higher.
- G6: domain k=1, acc_samples >= 190, gps_fixes >= 8; up to 2000 training and 3000 test windows per bird; a class
  qualifies on the scored test rows (>= 30 positives from >= 3 test birds). Secondary row: the VeDBA-based
  features.parquet labels on the same rows.
- G7: frozen list (injections.parquet beside split.json). The scorecard refuses to run without it and never creates
  it: scripts/scorecard_precount.py draws a candidate, its --freeze mode copies it (exclusive create). Hosts drawn
  only among scorable units (bouts: >= BOUT_MIN_REF same-bin reference bouts and >= MIN_Z_GROUP per bird x bin;
  days: clean score defined, >= MIN_Z_GROUP scorable days per bird, a donor comparable with >= DAY_MIN_PAIRS days)
  from the test set (search, or locked with --final), donors from that set plus the training birds. Bout mixes are
  shares of the 7 GPS-only labels plus an "unlabelled" share among the bout's windows with >= 8 fixes (labels
  overlap, so a mix does not sum to 1; anomaly's L1 thresholds 0.5 / 0.2 are kept). Kinds: novel_mix (anomaly's
  "novel": differs by L1 >= 0.5 from every host bout of the bin within +-45 d), same_mix (L1 <= 0.2), shuffle,
  other_season (a day >= 120 d away in circular day of year; windows of two days are compared within the same
  local hour x length class 40 / 64 / >= 190). Scoring = anomaly's bout content / sequence z and day energy-distance z, on the
  host birds only; AUC of injected against the clean scores of the same hosts. Days (eligible days, day
  donors, day PCA) use the scheduled domain (k=0, >= SCHED_MIN samples), not anomaly's full-window days.
- G8: positives = residency strategy migrant or migrant_departed; negatives = every other strategy except
  'insufficient'; bird vector = mean h over flight windows in Nov-Mar; >= G1_MIN_FLIGHT such windows.
- G9b: full windows, up to 1000 per bird and mode; reference = ACC_FEATURES (the no-GPS flag would give the mode).
- G10: targets speed_mean, climb_mean, log10 vedba, fore-aft tilt = roll_rot_deg of --states2 (states_r6_k2, r6
  roost frame, in which roll_rot_deg is the fore-aft tilt; one constant target for every design), dom_freq_hz in
  label_flapping == 1 full windows. The reference drops the target's own column.
- G8c: states.py procedure with reduced sizes (G8C) and K = 2..10; sub-states per coarse K=2 state for k = 2..6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from . import anomaly, demography, evaluate, reference, states
from .windows import GPS_CHANNELS

SPEC = "docs/SCORECARD_spec.md v2.1 (2026-11-27)"
SEED0 = 20261127
ROW_IDS = {"G1": 1, "G1b": 2, "G2": 3, "G3": 4, "G4": 5, "G4b": 6, "G5": 7, "G6": 8, "G7a": 9, "G7c": 10, "G7d": 11,
           "G7b": 12, "G8": 13, "G8b": 14, "G8c": 15, "G9": 16, "G9b": 17, "G10": 18}
ROWS = {  # id: goal, kind, metric, better, minimum effect
    "G1": ("Sex", "S", "AUC", "up", 0.03),
    "G1b": ("Sex by mode", "D", "AUC (continuous; scheduled)", "up", None),
    "G2": ("Daily rhythm", "S", "circular R2 of local solar hour", "up", 0.02),
    "G3": ("Seasonal pattern", "S", "circular R2 of day of year", "up", 0.02),
    "G4": ("Seasonal repeatability across years", "D/S", "mean per-bird AUC", "up", 0.03),
    "G4b": ("Year-to-year difference", "D", "mean cosine of month-matched year differences", "up", None),
    "G5": ("Lifetime (young-tagged)", "D", "mean per-bird pairwise AUC (year 2 older)", "up", None),
    "G6": ("Behaviour from acceleration alone", "S", "mean one-vs-rest AUC", "up", 0.03),
    "G7a": ("Odd bouts", "S", "AUC(novel mix) - AUC(same mix)", "up", 0.03),
    "G7c": ("Odd days", "S", "AUC", "up", 0.03),
    "G7d": ("Odd order", "D", "AUC", "up", None),
    "G7b": ("Identity control", "D", "AUC (same mix)", "down", None),
    "G8": ("Movement-strategy groups", "D/S", "AUC", "up", 0.03),
    "G8b": ("Unsupervised bird groups", "D", "max over k of min bootstrap Jaccard", "up", None),
    "G8c": ("Behaviour types", "D", "largest stable K", "up", None),
    "G9": ("Individual identity", "S", "balanced accuracy", "down", 0.02),
    "G9b": ("Recording mode from acceleration", "D", "balanced accuracy", "down", None),
    "G10": ("Kinematic guard rails", "guard", "R2 per target", "up", None),
}
GUARD_TOL = 0.02
TIME_ROWS = ("G2", "G3", "G4", "G4b", "G5", "G7c", "G8", "G8b")  # spec list; G4b added (same bird-month table as G4)
ENV_ROWS = TIME_ROWS + ("G9",)
WINDOW_ROWS = ("G2", "G6", "G7a", "G7b", "G7c", "G7d", "G9", "G9b", "G10")
FULL = 190  # full 10 s window: continuous-mode / full-window rows (G6, G9b, G10, flapping)
SCHED_MIN = 40  # scheduled-domain rows (G2, G3, G4, G4b, G5, G7c): k=0 windows with >= 40 samples (spec note 2026-11-27)
MODE_MIN, G1_MIN_FLIGHT = 50, 200
G3_MIN_WINDOWS, G4_MIN_WINDOWS, G4_MIN_BIRDS, G3_MONTHS = 50, 300, 15, 9
G8_FLOOR_MAX = 0.9
WINTER_MONTHS = (11, 12, 1, 2, 3)
MIGRANT_CLASSES = ("migrant", "migrant_departed")
G6_MIN_FIXES, G6_MIN_POS, G6_MIN_BIRDS = 8, 30, 3
CAPS = {"G2": 2000, "G6_train": 2000, "G6_test": 3000, "G9b": 1000, "G10_train": 1000, "G10_test": 2000,
        "G7_bout_hosts": 600, "G7_day_hosts": 600, "G7_pca_per_bird": 2000}
G8C = {"per_bird": 1000, "eval_per_bird": 300, "k_over": 20, "n_boot": 10, "ks": list(range(2, 11)),
       "sub_k_over": 10, "sub_ks": list(range(2, 7)), "sub_per_bird": 500}
N_FOLDS, N_REPEATS, N_PERM, N_BOOT_JACCARD = 5, 5, 200, 200
GPS_LABELS = ["perched", "circling", "climbing_straight", "gliding", "steep_descent", "take_off", "landing"]
ACC_FEATURES = ["vedba", "vedba_sd", "odba", "static_norm", "pitch_deg", "roll_deg", "flap_band_frac", "dom_freq_hz"]
VEDBA_LABELS = ["label_perched", "label_circling", "label_gliding", "label_climbing_soar", "label_flapping"]
FOREAFT_COL = "roll_rot_deg"
G10_TARGETS = {"speed": "speed_mean", "climb": "climb_mean", "vedba": "vedba", "fore_aft_tilt": None,
               "wingbeat_freq": "dom_freq_hz"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rng_of(row: str, *extra: int) -> np.random.Generator:
    return np.random.default_rng([SEED0, ROW_IDS[row], *extra])


def variant_of(name: str) -> str:
    return re.sub(r"_s\d+$", "", name)


# ================================================================ statistics shared with compare_scorecards

def weighted_auc(y, s, w=None) -> float:
    """ROC AUC (ties count half) with row weights (bootstrap multiplicities)."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok = np.isfinite(s) & np.isfinite(y) & (w > 0)
    y, s, w = y[ok], s[ok], w[ok]
    if not len(y):
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, w * (y == 1), len(u))
    wn = np.bincount(inv, w * (y == 0), len(u))
    tot = wp.sum() * wn.sum()
    return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / tot) if tot > 0 else np.nan


def r2_stats(Y: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> list[float]:
    """[sum w, sum w |P - Y|^2, sum w Y (per column), sum w |Y|^2]: R2 of any union of row sets is exact from these."""
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    P = np.asarray(P, float).reshape(len(P), -1)
    w = np.ones(len(Y)) if w is None else np.asarray(w, float)
    return [float(w.sum()), float((w * ((P - Y) ** 2).sum(1)).sum()), *map(float, (w[:, None] * Y).sum(0)),
            float((w * (Y ** 2).sum(1)).sum())]


def r2_from_stats(S: np.ndarray, mult: np.ndarray | None = None) -> float:
    """R2 = 1 - SSE / SST from per-bird r2_stats rows (bird multiplicities ``mult``)."""
    S = np.atleast_2d(np.asarray(S, float))
    m = np.ones(len(S)) if mult is None else np.asarray(mult, float)
    T = (m[:, None] * S).sum(0)
    sw, sse, sy, syy = T[0], T[1], T[2:-1], T[-1]
    sst = syy - (sy ** 2).sum() / sw if sw > 0 else np.nan
    return float(1 - sse / sst) if sw > 0 and sst > 0 else np.nan


def circular_r2(theta: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> float:
    """Spec: y = (cos, sin) of the angle, R2 = 1 - sum w |P - y|^2 / sum w |y - weighted mean y|^2."""
    return r2_from_stats(np.array([r2_stats(angle_xy(theta), P, w)]))


def angle_xy(theta) -> np.ndarray:
    t = np.asarray(theta, float)
    return np.c_[np.cos(t), np.sin(t)]


_BOOT_CACHE: dict = {}


def boot_birds(boot: dict) -> list[str]:
    b = boot["birds"]
    return sorted(b) if isinstance(b, dict) else sorted(b)


def boot_value(boot: dict, base: Path, mult: dict[str, float]) -> float:
    """Statistic of a row on a bird resample; ``mult`` maps bird -> multiplicity (missing = 0)."""
    t = boot["type"]
    if t in ("r2_birds", "auc_birds", "mean_birds", "bacc_birds"):
        bs = sorted(boot["birds"])
        m = np.array([mult.get(b, 0.0) for b in bs])
        V = np.array([boot["birds"][b] for b in bs], float)
        if t == "r2_birds":
            return r2_from_stats(V, m)
        if t == "auc_birds":
            return weighted_auc(V[:, 0], V[:, 1], m)
        if t == "mean_birds":
            ok = np.isfinite(V) & (m > 0)
            return float((V[ok] * m[ok]).sum() / m[ok].sum()) if ok.any() else np.nan
        n0, c0, n1, c1 = (m[:, None] * V).sum(0)
        return float((c0 / n0 + c1 / n1) / 2) if n0 > 0 and n1 > 0 else np.nan
    if t == "auc_rows":
        key = str(base / boot["file"])
        if key not in _BOOT_CACHE:
            D_ = pd.read_parquet(base / boot["file"])
            codes, ub = pd.factorize(D_["bird"])
            _BOOT_CACHE[key] = (D_, codes, list(ub))
        D, codes, ub = _BOOT_CACHE[key]
        w = np.array([mult.get(b, 0.0) for b in ub])[codes]
        vals = {}
        for part in boot["combine"]:
            m = (D["part"] == part).to_numpy()
            vals[part] = weighted_auc(D["y"].to_numpy()[m], D["s"].to_numpy()[m], w[m])
        if boot.get("combine_mode") == "mean":
            v = np.array(list(vals.values()), float)
            return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        return float(sum(boot["combine"][p] * vals[p] for p in vals))
    raise ValueError(f"unknown bootstrap type {t}")


# ================================================================ data

@dataclass
class Ctx:
    windows: Path
    split: dict
    final: bool
    F: pd.DataFrame
    birds: list
    nonlocked: list
    locked: list
    train: list
    test: list
    ref: pd.DataFrame
    strategy: pd.Series
    flight: np.ndarray
    foreaft: np.ndarray
    keep: np.ndarray
    age: np.ndarray
    workers: int = 1
    device: str = "cpu"
    n_perm: int = N_PERM
    inj: pd.DataFrame | None = None
    info: dict = field(default_factory=dict)
    _memo: dict = field(default_factory=dict)

    @property
    def bcode(self) -> np.ndarray:
        if "bcode" not in self._memo:
            self._memo["bcode"] = pd.Index(self.birds).get_indexer(self.F["bird"].to_numpy())
        return self._memo["bcode"]

    def is_in(self, birds) -> np.ndarray:
        return np.isin(self.F["bird"].to_numpy(), list(birds))

    def memo(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]


def _bird_frame(d: Path) -> pd.DataFrame | None:
    f = anomaly.load_bird_frame(d)
    if f is None:
        return None
    feats = pd.read_parquet(d / "features.parquet")
    for c in feats.columns:
        f[c] = feats[c].to_numpy()
    return f


def load_frame(windows: Path, birds, workers: int = 1, features: bool = True) -> pd.DataFrame:
    """Windows of ``birds`` (sorted), file row order, local solar day/month from anomaly.load_bird_frame."""
    dirs = [windows / b for b in sorted(birds) if (windows / b / "meta.parquet").exists()]
    fn = _bird_frame if features else anomaly.load_bird_frame
    parts = [p for p in Parallel(n_jobs=workers)(delayed(fn)(d) for d in dirs) if p is not None]
    F = pd.concat(parts, ignore_index=True)
    F["bird"] = F["bird"].astype(str)
    loc = pd.to_datetime(F["day"].to_numpy(), unit="D")  # local solar day (not UTC) for months and years
    F["month"], F["year"] = loc.month.to_numpy(), loc.year.to_numpy()
    return F


def flight_and_foreaft(F: pd.DataFrame, states2: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fixed flight mask (K=2 state_hmm with the higher median speed_mean) and the fixed fore-aft tilt target."""
    st = pd.read_parquet(states2 / "states.parquet", columns=["bird", "row", "state_hmm", FOREAFT_COL])
    st["bird"] = st["bird"].astype(str)
    st = st[st["bird"].isin(set(F["bird"]))]
    key = pd.MultiIndex.from_arrays([F["bird"].to_numpy(), F["row"].to_numpy()])
    pos = key.get_indexer(pd.MultiIndex.from_arrays([st["bird"].to_numpy(), st["row"].to_numpy()]))
    ok = pos >= 0
    state = np.full(len(F), -1)
    state[pos[ok]] = st["state_hmm"].to_numpy()[ok]
    fa = np.full(len(F), np.nan)
    fa[pos[ok]] = st[FOREAFT_COL].to_numpy()[ok]
    ks = sorted(set(state[state >= 0]))
    if len(ks) != 2:
        raise SystemExit(f"{states2}: need K=2 states, found {ks}")
    sp = F["speed_mean"].to_numpy(float)
    med = [np.nanmedian(sp[(state == k) & np.isfinite(sp)]) if ((state == k) & np.isfinite(sp)).any() else -np.inf
           for k in ks]
    fs = ks[int(np.argmax(med))]
    return state == fs, fa, {"flight_state": int(fs), "windows_without_state": int((state < 0).sum()),
                             "states_dir": str(states2)}


def age_classes(F: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    """reference.age_on class per window (data-driven fledge); 'not_stated' for birds not tagged young."""
    out = np.full(len(F), "not_stated", dtype=object)
    for b, ix in F.groupby("bird", sort=False).indices.items():
        if b not in ref.index or pd.isna(ref.loc[b, "tagged_young"]) or pd.isna(ref.loc[b, "hatch_year"]):
            continue
        t = F["t_start"].iloc[ix]
        fl = reference.fledge_from_speed(t, F["speed_mean"].to_numpy(float)[ix], ref.loc[b, "hatch_year"])
        out[ix] = reference.age_on({"tagged_young": True, "hatch_year": ref.loc[b, "hatch_year"]}, t,
                                   fl["fledge"])["age_class"].to_numpy()
    return out


def birds_with_windows(windows: Path, birds) -> list[str]:
    import pyarrow.parquet as pq

    return sorted(b for b in birds if (windows / b / "meta.parquet").exists()
                  and pq.ParquetFile(windows / b / "meta.parquet").metadata.num_rows > 0)


def build_context(windows: Path, split_path: Path, ref_csv: Path, states2: Path, residency: Path, final: bool,
                  workers: int = 1, device: str = "cpu", n_perm: int = N_PERM) -> Ctx:
    split = json.loads(split_path.read_text())
    locked = birds_with_windows(windows, split["locked"])
    train = birds_with_windows(windows, split["train"])
    search = birds_with_windows(windows, split["search"])
    nonlocked = sorted(set(train) | set(search))
    birds = sorted(nonlocked + locked) if final else nonlocked  # locked birds are never loaded in search mode
    log(f"loading {len(birds)} birds ({'final' if final else 'search'} mode)")
    F = load_frame(windows, birds, workers)
    birds = sorted(F["bird"].unique())
    ref = reference.load_birds(ref_csv, birds).set_index("bird")
    first = F.groupby("bird")["t_start"].min()
    ref["deploy_year"] = ref["deploy_on"].dt.year.fillna(first.dt.year).astype(int)
    ref["deploy_month"] = ref["deploy_on"].dt.month.fillna(first.dt.month).astype(int)
    ref["young"] = ref["tagged_young"].notna().astype(float)
    rs = pd.read_csv(residency / "residency.csv", dtype={"bird": str}).set_index("bird")
    strategy = rs["strategy"].reindex(birds)
    flight, fa, finfo = flight_and_foreaft(F, states2)
    age = age_classes(F, ref)
    inj_path = split_path.parent / "injections.parquet"
    ctx = Ctx(windows, split, final, F, birds, [b for b in nonlocked if b in birds],
              [b for b in locked if b in birds] if final else [], [b for b in train if b in birds],
              [b for b in (locked if final else search) if b in birds], ref, strategy, flight, fa, age != "pre_fledge",
              age, workers, device, n_perm)
    ctx.info = {"flight": finfo, "pre_fledge_windows_dropped": int((age == "pre_fledge").sum()),
                "birds_loaded": len(birds), "train_birds": len(ctx.train), "test_birds": len(ctx.test),
                "nonlocked_birds": len(ctx.nonlocked), "locked_birds_scored": len(ctx.locked),
                "injections": str(inj_path)}
    ctx._memo["inj_path"] = inj_path
    return ctx


def feature_matrix(F: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    from .baseline import FEATURES

    cols = [c for c in FEATURES if c in F] + ["no_gps"]
    return evaluate._feature_matrix(F).astype(np.float32), cols


# ================================================================ representations

class Rep:
    """One representation of every window of ctx.F: h of a run, or handcrafted features."""

    def __init__(self, name: str, X: np.ndarray, unit: bool, cols: list | None = None, nogps=None,
                 note: str | None = None):
        self.name, self.X, self.unit, self.cols, self._nogps, self.note = name, X, unit, cols, nogps, note
        self.is_features = cols is not None
        self._unit_rows = None
        self.l2 = None  # level-2 runs: {"run", "parent_H", "notime"} (G7 injections in the parent stream)

    def window(self, rows, drop: str | None = None) -> np.ndarray:
        X = np.asarray(self.X[rows], np.float32)
        if self.is_features and drop in self.cols:
            X = np.delete(X, self.cols.index(drop), axis=1)
        return X

    def nogps(self, rows) -> np.ndarray:
        return self._nogps(rows)

    def unit_rows(self, ctx: Ctx) -> np.ndarray:
        """Per-window unit vectors (h L2-normalised; features standardised on training birds, then L2-normalised)."""
        if self._unit_rows is None:
            X = np.asarray(self.X, np.float32)
            if self.is_features:
                tr = ctx.is_in(ctx.train)
                mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
                X = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1)).astype(np.float32)
            self._unit_rows = anomaly.norm_rows(X)
        return self._unit_rows


def group_mean(X: np.ndarray, codes: np.ndarray, G: int, unit: bool, chunk: int = 400_000):
    """NaN-aware mean of rows of X per code 0..G-1 (code < 0 ignored); rows L2-normalised first if ``unit``."""
    D = X.shape[1]
    S, N = np.zeros((G, D)), np.zeros((G, D))
    for i in range(0, len(X), chunk):
        c = codes[i:i + chunk]
        ok = c >= 0
        if not ok.any():
            continue
        x = np.asarray(X[i:i + chunk][ok], np.float64)
        if unit:
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        fin = np.isfinite(x)
        M = sparse.csr_matrix((np.ones(ok.sum()), (c[ok], np.arange(ok.sum()))), shape=(G, ok.sum()))
        S += M @ np.where(fin, x, 0.0)
        N += M @ fin.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(N > 0, S / np.maximum(N, 1), np.nan), np.bincount(codes[codes >= 0], minlength=G)


# ================================================================ models

def _quiet():
    w = warnings.catch_warnings()
    w.__enter__()
    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*least populated class.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return w


def ridge_predict(Xtr, Ytr, Xte, w=None) -> np.ndarray:
    a, b = demography._prep(Xtr, Xte)
    return Ridge(alpha=1.0).fit(a, Ytr, sample_weight=w).predict(b)


def logit_fit(Xtr, ytr, Xte):
    a, b = demography._prep(Xtr, Xte)
    m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(a, ytr)
    return m, b


def cv_oof(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Copy of demography.cv_once with C fixed at 1.0 (no inner C selection); returns out-of-fold P(y = 1)."""
    p = np.full(len(y), np.nan)
    folds = demography.grouped_folds(y, np.arange(len(y)), N_FOLDS, seed)
    for tr, te in folds:
        a, b = demography._prep(X[tr], X[te])
        p[te] = demography._lr(1.0).fit(a, y[tr]).predict_proba(b)[:, 1]
    return p


def bird_cv(ctx: Ctx, X: pd.DataFrame, y: pd.Series, row: str, strata: pd.Series | None = None,
            n_perm: int | None = None) -> dict:
    """Bird-level AUC: repeated grouped CV over non-locked birds with a between-bird permutation null; with --final,
    fit on non-locked birds and score the locked birds (direction only)."""
    n_perm = ctx.n_perm if n_perm is None else n_perm
    y = y.dropna().astype(int)
    X = X[X.notna().any(axis=1)]
    w = _quiet()
    try:
        if ctx.final:
            fit = sorted(set(X.index) & set(y.index) & set(ctx.nonlocked))
            te = sorted(set(X.index) & set(y.index) & set(ctx.locked))
            yt = y.loc[te].to_numpy()
            if len(te) < 2 or len(set(yt)) < 2 or len(set(y.loc[fit])) < 2:
                return {"skipped": "too few labelled birds"}
            a, b = demography._prep(X.loc[fit].to_numpy(float), X.loc[te].to_numpy(float))
            p = demography._lr(1.0).fit(a, y.loc[fit].to_numpy()).predict_proba(b)[:, 1]
            return {"score": float(roc_auc_score(yt, p)), "n_birds": len(te), "n_fit_birds": len(fit),
                    "n_pos": int(yt.sum()), "n_neg": int((1 - yt).sum()), "n": len(te),
                    "test": "direction only (final: fit on non-locked birds, scored once on locked birds)",
                    "boot": {"type": "auc_birds", "birds": {b_: [int(v), float(q)] for b_, v, q in zip(te, yt, p)}}}
        bs = sorted(set(X.index) & set(y.index) & set(ctx.nonlocked))
        Xa, ya = X.loc[bs].to_numpy(float), y.loc[bs].to_numpy()
        if min(ya.sum(), (1 - ya).sum()) < 2:
            return {"skipped": "fewer than 2 birds in a class"}
        rng = rng_of(row)
        seeds = rng.integers(0, 2**31 - 1, N_REPEATS)
        P = np.stack([cv_oof(Xa, ya, int(s)) for s in seeds])
        aucs = np.array([roc_auc_score(ya, p) for p in P])
        st = None if strata is None else strata.reindex(bs).to_numpy()
        null = []
        if n_perm:
            perms = demography._permuted(ya, n_perm, rng, st)
            pseeds = rng.integers(0, 2**31 - 1, n_perm)
            null = np.array([roc_auc_score(yp, cv_oof(Xa, yp, int(s))) for yp, s in zip(perms, pseeds)])
        pm = P.mean(0)
        out = {"score": float(aucs.mean()), "sd_over_repeats": float(aucs.std(ddof=1)), "n_birds": len(bs),
               "n_pos": int(ya.sum()), "n_neg": int((1 - ya).sum()), "n": len(bs), "n_repeats": N_REPEATS,
               "boot": {"type": "auc_birds", "birds": {b_: [int(v), float(q)] for b_, v, q in zip(bs, ya, pm)}}}
        if n_perm:
            out.update(p_perm=float((1 + (null >= aucs.mean() - 1e-12).sum()) / (1 + n_perm)), n_perm=int(n_perm),
                       null_auc_mean=float(null.mean()), null_auc_q95=float(np.quantile(null, 0.95)),
                       test="tested: labels permuted between birds" + ("" if strata is None else
                                                                        " within deployment year"))
        return out
    finally:
        w.__exit__(None, None, None)


def sample_rows(ctx: Ctx, key: str, mask: np.ndarray, birds, n: int, row: str, sub: int = 0) -> np.ndarray:
    """Up to n rows per bird (evaluate.sample_per_bird) of mask & birds; fixed generator per row and draw."""
    def draw():
        m = mask & ctx.is_in(birds)
        if not m.any():
            return np.zeros(0, int)
        return np.sort(evaluate.sample_per_bird(ctx.F, m, n, rng_of(row, sub)))
    return ctx.memo(("sample", key), draw)


def per_bird_r2(birds: np.ndarray, Y, P, w=None) -> dict:
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    P = np.asarray(P, float).reshape(len(P), -1)
    w = np.ones(len(Y)) if w is None else np.asarray(w, float)
    return {str(b): r2_stats(Y[birds == b], P[birds == b], w[birds == b]) for b in np.unique(birds)}


def window_r2(ctx: Ctx, rep_rows_fn, Y, tr, te, w_te=None) -> dict:
    Xtr, Xte = rep_rows_fn(tr), rep_rows_fn(te)
    ok_tr = np.isfinite(Y[tr]).all(1) if Y.ndim == 2 else np.isfinite(Y[tr])
    ok_te = np.isfinite(Y[te]).all(1) if Y.ndim == 2 else np.isfinite(Y[te])
    if ok_tr.sum() < 20 or ok_te.sum() < 20:
        return {"skipped": "too few rows"}
    P = ridge_predict(Xtr[ok_tr], Y[tr][ok_tr], Xte[ok_te])
    b = ctx.F["bird"].to_numpy()[te][ok_te]
    st = per_bird_r2(b, Y[te][ok_te], P)
    return {"score": r2_from_stats(np.array(list(st.values()))), "n_birds": int(len(st)), "n": int(ok_te.sum()),
            "n_train": int(ok_tr.sum()), "boot": {"type": "r2_birds", "birds": st}}


# ================================================================ bird rows (G1, G1b, G8, G8b)

def bird_counts(ctx: Ctx, sel: np.ndarray) -> np.ndarray:
    return np.bincount(ctx.bcode[sel], minlength=len(ctx.birds))


def g1_selection(ctx: Ctx) -> dict:
    F = ctx.F
    k = F["sensitivity_k"].to_numpy()
    sel = ctx.flight & ctx.keep
    n0, n1 = bird_counts(ctx, sel & (k == 0)), bird_counts(ctx, sel & (k == 1))
    sex = ctx.ref["sex"].reindex(ctx.birds)
    has = sex.notna().to_numpy()
    B = np.array(ctx.birds)
    return {"sel": sel, "n0": n0, "n1": n1, "G1": list(B[has & (n0 + n1 >= G1_MIN_FLIGHT)]),
            "G1b_continuous": list(B[has & (n1 >= G1_MIN_FLIGHT)]), "G1b_scheduled": list(B[has & (n0 >= G1_MIN_FLIGHT)])}


def g1_vectors(ctx: Ctx, rep: Rep) -> dict:
    s = g1_selection(ctx)
    k = ctx.F["sensitivity_k"].to_numpy()
    G = len(ctx.birds)
    M0, _ = group_mean(rep.X, np.where(s["sel"] & (k == 0), ctx.bcode, -1), G, rep.unit)
    M1, _ = group_mean(rep.X, np.where(s["sel"] & (k == 1), ctx.bcode, -1), G, rep.unit)
    u0, u1 = (s["n0"] >= MODE_MIN)[:, None], (s["n1"] >= MODE_MIN)[:, None]
    both = np.where(u0 & u1, (M0 + M1) / 2, np.where(u1, M1, np.where(u0, M0, np.where(
        (s["n1"] >= s["n0"])[:, None], M1, M0))))
    idx = pd.Index(ctx.birds, name="bird")
    out = {"G1": pd.DataFrame(both, index=idx), "continuous": pd.DataFrame(M1, index=idx),
           "scheduled": pd.DataFrame(M0, index=idx)}
    if rep.is_features:  # stage-6 signal: bird medians in continuous-mode flight windows
        sp = ctx.F["speed_mean"].to_numpy(float)
        tr = ctx.F["turn_abs_mean"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            rad = np.where(tr > 0, sp / np.radians(tr), np.nan)
        codes = np.where(s["sel"] & (k == 1), ctx.bcode, -1)
        kin = pd.DataFrame({f"median_{n}": demography.median_by(codes, G, v)[0]
                            for n, v in (("speed", sp), ("turn_radius", rad), ("turn_rate", tr))}, index=idx)
        out["G1"] = pd.concat([out["G1"].add_prefix("f"), kin], axis=1)
        out["continuous"] = pd.concat([out["continuous"].add_prefix("f"), kin], axis=1)
    return out


def bird_nuisance(ctx: Ctx, extra: bool = False) -> pd.DataFrame:
    F = ctx.F
    g = F.groupby("bird")
    r = ctx.ref.reindex(ctx.birds)
    dm = r["deploy_month"].astype(float)
    X = pd.DataFrame({"cont_share": g["sensitivity_k"].apply(lambda s: float((s == 1).mean())),
                      "gps_share": g["gps_fixes"].apply(lambda s: float((s > 0).mean())),
                      "track_days": (g["t_start"].max() - g["t_start"].min()).dt.total_seconds() / 86400}
                     ).reindex(ctx.birds)
    X["deploy_month_sin"], X["deploy_month_cos"] = np.sin(2 * np.pi * dm / 12).to_numpy(), np.cos(2 * np.pi * dm / 12).to_numpy()
    yrs = pd.get_dummies(r["deploy_year"].astype(str), prefix="deploy_year", dtype=float)
    X = pd.concat([X, yrs.iloc[:, 1:]], axis=1)
    if extra:
        X["young"] = r["young"].to_numpy()
        winter = F["month"].isin(WINTER_MONTHS).to_numpy()
        X["winter_windows"] = bird_counts(ctx, winter)
    X.index.name = "bird"
    return X


def sex_labels(ctx: Ctx) -> pd.Series:
    return ctx.ref["sex"].reindex(ctx.birds).map({"f": 1, "m": 0})


def row_G1(ctx, rep, cache):
    v = cache.setdefault(("g1v", rep.name), g1_vectors(ctx, rep))
    s = g1_selection(ctx)
    return bird_cv(ctx, v["G1"].loc[s["G1"]], sex_labels(ctx), "G1")


def row_G1b(ctx, rep, cache):
    v = cache.setdefault(("g1v", rep.name), g1_vectors(ctx, rep))
    s = g1_selection(ctx)
    out = {}
    for mode in ("continuous", "scheduled"):
        out[mode] = bird_cv(ctx, v[mode].loc[s[f"G1b_{mode}"]], sex_labels(ctx), "G1b")
    sc = [out[m].get("score", np.nan) for m in ("continuous", "scheduled")]
    return {"score": sc, "n_birds": [out[m].get("n_birds") for m in out], "parts": out}


def floor_G1(ctx, row):
    s = g1_selection(ctx)
    X = bird_nuisance(ctx)
    if row == "G1":
        return bird_cv(ctx, X.loc[s["G1"]], sex_labels(ctx), "G1")
    out = {m: bird_cv(ctx, X.loc[s[f"G1b_{m}"]], sex_labels(ctx), "G1b") for m in ("continuous", "scheduled")}
    return {"score": [out[m].get("score", np.nan) for m in out], "parts": out}


def g8_selection(ctx: Ctx) -> tuple[np.ndarray, list, pd.Series]:
    sel = ctx.flight & ctx.keep & ctx.F["month"].isin(WINTER_MONTHS).to_numpy()
    n = bird_counts(ctx, sel)
    st = ctx.strategy
    y = pd.Series(np.where(st.isin(MIGRANT_CLASSES), 1.0, np.where(st.notna() & (st != "insufficient"), 0.0, np.nan)),
                  index=ctx.birds)
    birds = [b for b, c in zip(ctx.birds, n) if c >= G1_MIN_FLIGHT and np.isfinite(y[b])]
    return sel, birds, y


def row_G8(ctx, rep, cache):
    sel, birds, y = g8_selection(ctx)
    M, _ = group_mean(rep.X, np.where(sel, ctx.bcode, -1), len(ctx.birds), rep.unit)
    X = pd.DataFrame(M, index=pd.Index(ctx.birds, name="bird")).loc[birds]
    return bird_cv(ctx, X, y, "G8", strata=ctx.ref["deploy_year"])


def floor_G8(ctx, row):
    sel, birds, y = g8_selection(ctx)
    return bird_cv(ctx, bird_nuisance(ctx, extra=True).loc[birds], y, "G8", strata=ctx.ref["deploy_year"])


def kmeans_stability(X: np.ndarray, row: str, ks=(2, 3, 4, 5), n_boot: int = N_BOOT_JACCARD) -> dict:
    X = np.nan_to_num((X - np.nanmean(X, 0)) / np.where(np.nanstd(X, 0) > 0, np.nanstd(X, 0), 1))
    rng = rng_of(row)
    per_k = {}
    for k in ks:
        if len(X) < 2 * k:
            continue
        ref = KMeans(k, n_init=10, random_state=0).fit_predict(X)
        jac = []
        for b in range(n_boot):
            idx = rng.integers(0, len(X), len(X))
            lab = KMeans(k, n_init=3, random_state=b).fit(X[idx]).predict(X)
            u = np.unique(idx)
            jac.append(states.jaccard_best(ref[u], lab[u], k))
        j = np.mean(jac, 0)
        per_k[str(k)] = {"jaccard": j.round(4).tolist(), "jaccard_min": float(j.min()),
                         "sizes": np.bincount(ref, minlength=k).tolist()}
    if not per_k:
        return {"skipped": "too few birds"}
    best = max(per_k, key=lambda k: per_k[k]["jaccard_min"])
    return {"score": per_k[best]["jaccard_min"], "best_k": int(best), "stable": per_k[best]["jaccard_min"] >= 0.75,
            "per_k": per_k, "n_birds": len(X), "n": len(X), "n_boot": n_boot,
            "rule": "k with the highest minimum bootstrap Jaccard; stable when >= 0.75 (stage-6 rule)"}


def row_G8b(ctx, rep, cache):
    if ctx.final:
        return {"skipped": "not scored in final mode"}
    v = cache.setdefault(("g1v", rep.name), g1_vectors(ctx, rep))
    s = g1_selection(ctx)
    birds = [b for b, n0, n1 in zip(ctx.birds, s["n0"], s["n1"]) if n0 + n1 >= G1_MIN_FLIGHT]
    return kmeans_stability(v["G1"].loc[birds].to_numpy(float), "G8b")


# ================================================================ G2, G3, G4, G4b, G5

def day_mask(ctx: Ctx) -> np.ndarray:
    F = ctx.F
    return ((F["sensitivity_k"].to_numpy() == 0) & (F["acc_samples"].to_numpy() >= SCHED_MIN)
            & (F["sun_elev_deg"].to_numpy() > 0) & ~F["lsh_filled"].to_numpy())  # measured local solar hour only


def row_G2(ctx, rep, cache):
    dom = day_mask(ctx)
    tr = sample_rows(ctx, "G2tr", dom, ctx.train, CAPS["G2"], "G2", 0)
    te = sample_rows(ctx, "G2te", dom, ctx.test, CAPS["G2"], "G2", 1)
    Y = angle_xy(2 * np.pi * ctx.F["local_solar_hour"].to_numpy(float) / 24)
    return window_r2(ctx, lambda r: rep.window(r), Y, tr, te)


def floor_G2(ctx, row):
    cols = ["gps_fixes", "static_norm", "acc_samples"]
    X = ctx.F[cols].to_numpy(np.float32)
    return row_G2(ctx, Rep("floor", X, False, cols=cols), {})


def bird_days(ctx: Ctx) -> dict:
    def make():
        F = ctx.F
        dom = day_mask(ctx)
        key = pd.MultiIndex.from_arrays([F["bird"].to_numpy(), F["day"].to_numpy()])
        codes_all, uniq = pd.factorize(key)
        n = np.bincount(codes_all[dom], minlength=len(uniq))
        good = n >= G3_MIN_WINDOWS
        remap = np.full(len(uniq), -1)
        remap[good] = np.arange(good.sum())
        codes = np.where(dom, remap[codes_all], -1)
        T = pd.DataFrame({"bird": uniq.get_level_values(0)[good], "day": uniq.get_level_values(1)[good],
                          "n": n[good]})
        T["doy"] = pd.to_datetime(T["day"].to_numpy(), unit="D").dayofyear.to_numpy()
        T["month"] = pd.to_datetime(T["day"].to_numpy(), unit="D").month.to_numpy()
        # nuisance (daily recording format etc.)
        allc = remap[codes_all]
        ok = allc >= 0
        cnt = np.bincount(allc[ok], minlength=len(T))
        T["n_windows_day"] = cnt
        n_acc = F["acc_samples"].to_numpy()[ok]
        for name, m in (("share_40", n_acc == 40), ("share_64", n_acc == 64), ("share_full", n_acc >= FULL)):
            T[name] = np.bincount(allc[ok], weights=m, minlength=len(T)) / cnt
        T["cont_share"] = np.bincount(allc[ok], weights=(F["sensitivity_k"].to_numpy()[ok] == 1), minlength=len(T)) / cnt
        sn = F["static_norm"].to_numpy(float)
        T["static_norm_mean"] = group_mean(sn[:, None], codes, len(T), False)[0][:, 0]
        dep = ctx.ref["deploy_on"].reindex(T["bird"]).to_numpy()
        first_day = F.groupby("bird")["day"].min().reindex(T["bird"]).to_numpy()
        dep_day = np.array([(d - pd.Timestamp(0, tz="UTC")).days if pd.notna(d) else fd for d, fd in zip(dep, first_day)])
        T["days_since_deploy"] = T["day"].to_numpy() - dep_day
        return {"codes": codes, "T": T}
    return ctx.memo("bird_days", make)


def g3_nuisance(ctx: Ctx, T: pd.DataFrame) -> np.ndarray:
    r = ctx.ref.reindex(T["bird"])
    yrs = pd.get_dummies(r["deploy_year"].astype(str).to_numpy(), dtype=float).iloc[:, 1:]
    strat = pd.get_dummies(ctx.strategy.reindex(T["bird"]).fillna("none").to_numpy(), dtype=float).iloc[:, 1:]
    base = T[["n_windows_day", "share_40", "share_64", "share_full", "cont_share", "days_since_deploy", "static_norm_mean"]].to_numpy(float)
    return np.c_[base, yrs.to_numpy(), r["young"].to_numpy(float), strat.to_numpy()]


def grouped_r2_cv(ctx: Ctx, X: np.ndarray, theta: np.ndarray, birds: np.ndarray, row: str, n_perm: int) -> dict:
    """Bird-day circular R2: repeated grouped CV over non-locked birds, bird weights 1/rows; rotation null."""
    w_all = 1.0 / pd.Series(birds).map(pd.Series(birds).value_counts()).to_numpy(float)
    Y = angle_xy(theta)
    wq = _quiet()
    try:
        if ctx.final:
            tr, te = np.isin(birds, ctx.nonlocked), np.isin(birds, ctx.locked)
            if tr.sum() < 20 or te.sum() < 1:
                return {"skipped": "too few bird-days"}
            P = ridge_predict(X[tr], Y[tr], X[te], w_all[tr])
            st = per_bird_r2(birds[te], Y[te], P, w_all[te])
            return {"score": r2_from_stats(np.array(list(st.values()))), "n_birds": len(st), "n": int(te.sum()),
                    "test": "final: fit on non-locked birds, scored once on locked birds",
                    "boot": {"type": "r2_birds", "birds": st}}
        keep = np.isin(birds, ctx.nonlocked)
        X, Y, birds, w, theta = X[keep], Y[keep], birds[keep], w_all[keep], theta[keep]
        ub = np.array(sorted(set(birds)))
        if len(ub) < N_FOLDS:
            return {"skipped": "too few birds"}
        code = pd.Index(ub).get_indexer(birds)
        rng = rng_of(row)
        folds = []
        for _ in range(N_REPEATS):
            f = np.empty(len(ub), int)
            for i, ch in enumerate(np.array_split(rng.permutation(len(ub)), N_FOLDS)):
                f[ch] = i
            folds.append(f[code])

        def oof(Yt, fold):
            P = np.zeros_like(Yt)
            for i in range(N_FOLDS):
                te = fold == i
                P[te] = ridge_predict(X[~te], Yt[~te], X[te], w[~te])
            return P

        Ps = [oof(Y, f) for f in folds]
        r2s = np.array([circular_r2(theta, P, w) for P in Ps])
        st = per_bird_r2(birds, Y, np.mean(Ps, 0), w)
        out = {"score": float(r2s.mean()), "sd_over_repeats": float(r2s.std(ddof=1)), "n_birds": len(ub),
               "n": int(len(birds)), "n_repeats": N_REPEATS, "boot": {"type": "r2_birds", "birds": st},
               "boot_note": "per-bird statistics of the out-of-fold predictions averaged over repeats"}
        if n_perm:
            null = []
            for _ in range(n_perm):
                th = theta + rng.uniform(0, 2 * np.pi, len(ub))[code]
                null.append(circular_r2(th, oof(angle_xy(th), folds[0]), w))
            null = np.array(null)
            out.update(p_perm=float((1 + (null >= r2s.mean() - 1e-12).sum()) / (1 + n_perm)), n_perm=int(n_perm),
                       null_mean=float(null.mean()),
                       test="tested: each bird's day-of-year angle rotated by its own random offset")
        return out
    finally:
        wq.__exit__(None, None, None)


def row_G3(ctx, rep, cache, X=None, n_perm=None):
    bd = bird_days(ctx)
    T = bd["T"]
    if X is None:
        X = group_mean(rep.X, bd["codes"], len(T), rep.unit)[0]
    res = grouped_r2_cv(ctx, X, 2 * np.pi * T["doy"].to_numpy(float) / 365.25, T["bird"].to_numpy(),
                        "G3", ctx.n_perm if n_perm is None else n_perm)
    months = T.groupby("bird")["month"].nunique()
    sc = set(ctx.locked if ctx.final else ctx.nonlocked)
    res["birds_with_9_calendar_months"] = int((months[months.index.isin(sc)] >= G3_MONTHS).sum())
    return res


def floor_G3(ctx, row):
    bd = bird_days(ctx)
    return row_G3(ctx, None, {}, X=g3_nuisance(ctx, bd["T"]), n_perm=0)


def bird_months(ctx: Ctx, by: str = "year") -> dict:
    """Bird-month units of scheduled daytime full windows (>= G4_MIN_WINDOWS): by calendar year or by age year."""
    def make():
        F = ctx.F
        dom = day_mask(ctx) & ctx.keep
        if by == "year":
            yr = F["year"].to_numpy()
        else:
            yr = np.select([ctx.age == "juvenile_y1", ctx.age == "y2"], [1, 2], 0)
            dom &= yr > 0
        key = pd.MultiIndex.from_arrays([F["bird"].to_numpy(), yr, F["month"].to_numpy()])
        ca, uniq = pd.factorize(key)
        n = np.bincount(ca[dom], minlength=len(uniq))
        good = n >= G4_MIN_WINDOWS
        remap = np.full(len(uniq), -1)
        remap[good] = np.arange(good.sum())
        T = pd.DataFrame({"bird": uniq.get_level_values(0)[good], "year": uniq.get_level_values(1)[good],
                          "month": uniq.get_level_values(2)[good], "n": n[good]})
        return {"codes": np.where(dom, remap[ca], -1), "T": T}
    return ctx.memo(("bird_months", by), make)


def _std(M: np.ndarray) -> np.ndarray:
    mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
    return np.nan_to_num((M - mu) / np.where(sd > 0, sd, 1))


def _cosdist(A, B):
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return 1 - A @ B.T


def g4_pairs(T: pd.DataFrame, b: str):
    t = T[T["bird"] == b]
    i = t.index.to_numpy()
    y, m = t["year"].to_numpy(), t["month"].to_numpy()
    dy = y[:, None] != y[None]
    dm = np.abs(m[:, None] - m[None])
    dm = np.minimum(dm, 12 - dm)
    up = np.triu(np.ones((len(t), len(t)), bool), 1)
    return i, dy & (dm == 0) & up, dy & (dm >= 3) & up


def g4_qualifying(T: pd.DataFrame, birds) -> list[str]:
    out = []
    for b in sorted(set(T["bird"]) & set(birds)):
        _, pos, neg = g4_pairs(T, b)
        if pos.any() and neg.any():
            out.append(b)
    return out


def row_G4(ctx, rep, cache):
    bm = bird_months(ctx)
    T = bm["T"]
    Z = _std(group_mean(rep.X, bm["codes"], len(T), rep.unit)[0])
    birds = g4_qualifying(T, ctx.locked if ctx.final else ctx.nonlocked)
    per = {}
    for b in birds:
        i, pos, neg = g4_pairs(T, b)
        D = _cosdist(Z[i], Z[i])
        per[b] = weighted_auc(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], -np.r_[D[pos], D[neg]])
    if not per:
        return {"skipped": "no bird qualifies", "n_birds": 0}
    v = np.array(list(per.values()))
    p = demography.sign_flip(v - 0.5, 10_000, rng_of("G4"))
    return {"score": float(v.mean()), "n_birds": len(v), "n": int(len(T)), "p_sign_flip": p,
            "test": "tested: per-bird AUC - 0.5, sign-flip over birds", "boot": {"type": "mean_birds", "birds": per}}


def row_G4b(ctx, rep, cache):
    bm = bird_months(ctx)
    T = bm["T"]
    Z = _std(group_mean(rep.X, bm["codes"], len(T), rep.unit)[0])
    per = {}
    for b in sorted(set(T["bird"]) & set(ctx.locked if ctx.final else ctx.nonlocked)):
        t = T[T["bird"] == b]
        vals = []
        ys = sorted(set(t["year"]))
        for a in range(len(ys)):
            for c in range(a + 1, len(ys)):
                ta = t[t["year"] == ys[a]].set_index("month")
                tc = t[t["year"] == ys[c]].set_index("month")
                sh = sorted(set(ta.index) & set(tc.index))
                if len(sh) < 2:
                    continue
                ia = t.index[(t["year"] == ys[a]).to_numpy() & t["month"].isin(sh).to_numpy()]
                ic = t.index[(t["year"] == ys[c]).to_numpy() & t["month"].isin(sh).to_numpy()]
                ia = sorted(ia, key=lambda i_: T.at[i_, "month"])
                ic = sorted(ic, key=lambda i_: T.at[i_, "month"])
                d = Z[ic] - Z[ia]
                C = 1 - _cosdist(d, d)
                vals.append(float(C[np.triu_indices(len(sh), 1)].mean()))
        if vals:
            per[b] = float(np.mean(vals))
    if not per:
        return {"skipped": "no bird has two years with >= 2 shared months", "n_birds": 0}
    v = np.array(list(per.values()))
    return {"score": float(v.mean()), "n_birds": len(v), "p_sign_flip": demography.sign_flip(v, 10_000, rng_of("G4b")),
            "test": "descriptive (calendar year confounded with tag age and deployment year)",
            "boot": {"type": "mean_birds", "birds": per}}


def row_G5(ctx, rep, cache):
    bm = bird_months(ctx, by="age")
    T = bm["T"]
    if not len(T):
        return {"skipped": "no young-bird months", "n_birds": 0}
    Z = _std(group_mean(rep.X, bm["codes"], len(T), rep.unit)[0])
    pool = ctx.locked if ctx.final else ctx.nonlocked
    qual = []
    for b in sorted(set(T["bird"]) & set(pool)):
        t = T[T["bird"] == b]
        if set(t.loc[t["year"] == 1, "month"]) & set(t.loc[t["year"] == 2, "month"]):
            qual.append(b)
    per = {}
    wq = _quiet()
    try:
        for b in qual:
            trm = (T["bird"] != b).to_numpy() & (np.isin(T["bird"], ctx.nonlocked) if ctx.final else True)
            if len(set(T.loc[trm, "year"])) < 2 or T.loc[trm, "bird"].nunique() < 2:
                continue
            te = (T["bird"] == b).to_numpy()
            m, Xb = logit_fit(Z[trm], T.loc[trm, "year"].to_numpy(), Z[te])
            s = pd.Series(m.decision_function(Xb), index=pd.MultiIndex.from_frame(T.loc[te, ["year", "month"]]))
            sh = sorted(set(T.loc[te & (T["year"] == 1).to_numpy(), "month"]) & set(T.loc[te & (T["year"] == 2).to_numpy(), "month"]))
            d = np.array([s[(2, m_)] - s[(1, m_)] for m_ in sh])
            per[b] = float(((d > 0) + 0.5 * (d == 0)).mean())
    finally:
        wq.__exit__(None, None, None)
    if not per:
        return {"skipped": "no young bird with the same months in year 1 and year 2", "n_birds": 0}
    v = np.array(list(per.values()))
    return {"score": float(v.mean()), "n_birds": len(v), "n": int(len(T)),
            "test": "descriptive (confounded with calendar year and tag age)", "boot": {"type": "mean_birds", "birds": per}}


# ================================================================ G6, G9b: acceleration alone

def gps_labels(g: np.ndarray, gm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 7) GPS-only labels (GPS_LABELS, spec table) and the >= 8-valid-fix flag, from raw gps.npy / gps_mask.npy."""
    i_sp, i_cl, i_tr = (GPS_CHANNELS.index(c) for c in ("ground-speed", "climb_ms", "turn_rate_dps"))
    g = np.asarray(g, np.float64)
    gm = np.asarray(gm, bool)
    sp = np.where(gm[..., i_sp], g[..., i_sp], np.nan)
    cl = np.where(gm[..., i_cl], g[..., i_cl], np.nan)
    tr = np.where(gm[..., i_tr], g[..., i_tr], np.nan)
    v = gm[..., i_sp]
    n = v.sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med, mx = np.nanmedian(sp, 1), np.nanmax(sp, 1)
        mcl, mincl = np.nanmean(cl, 1), np.nanmin(cl, 1)
        net = np.abs(np.nansum(tr, 1))
        c, rc = np.cumsum(v, 1), np.cumsum(v[:, ::-1], 1)[:, ::-1]
        first = np.nanmedian(np.where(v & (c <= 3), sp, np.nan), 1)
        last = np.nanmedian(np.where(v & (rc <= 3), sp, np.nan), 1)
    ok = n >= G6_MIN_FIXES
    L = np.stack([med < 1, (med > 4) & (net >= 180), (med > 4) & (mcl > 0.5) & (net < 90),
                  (med > 8) & (mcl < -0.5) & (net < 45), (mincl < -5) & (mx > 15), (first < 2) & (last > 5),
                  (first > 5) & (last < 2)], 1) & ok[:, None]
    return L, ok


def read_rows(windows: Path, F: pd.DataFrame, rows: np.ndarray, names: tuple[str, ...]) -> dict:
    """Arrays ``names`` (acc, acc_mask, gps, gps_mask) for global rows of F, in the order of ``rows``."""
    rows = np.asarray(rows)
    out = {}
    birds, within = F["bird"].to_numpy()[rows], F["row"].to_numpy()[rows]
    for b in pd.unique(birds):
        m = np.flatnonzero(birds == b)
        w = within[m]
        o = np.argsort(w)
        for nm in names:
            a = np.load(windows / str(b) / f"{nm}.npy", mmap_mode="r")
            x = np.asarray(a[w[o]])
            if nm not in out:
                out[nm] = np.zeros((len(rows),) + x.shape[1:], x.dtype)
            out[nm][m[o]] = x
    return out


def gps_label_rows(windows: Path, F: pd.DataFrame, rows) -> tuple[np.ndarray, np.ndarray]:
    if not len(rows):
        return np.zeros((0, len(GPS_LABELS)), bool), np.zeros(0, bool)
    d = read_rows(windows, F, rows, ("gps", "gps_mask"))
    return gps_labels(d["gps"], d["gps_mask"])


def encode_rows(run: Path, windows: Path, F: pd.DataFrame, rows: np.ndarray, device: str, keep_gps: bool = False,
                chunk: int = 50_000) -> np.ndarray:
    """h of windows ``rows`` re-encoded with the run's latest checkpoint, prepared as dataset.load prepares training
    input (common format, rotation from orientation.json); GPS values and mask zeroed unless ``keep_gps``."""
    from . import dataset
    from .train import encode

    cfg = json.loads((run / "config.json").read_text())
    step_g, clip_g = float(cfg.get("step_g", 0) or 0), float(cfg.get("clip_g", 0) or 0)
    opath = run / "orientation.json"
    if not opath.exists():
        raise SystemExit(f"{opath} missing: the training frame cannot be reproduced")
    orient = json.loads(opath.read_text())
    if cfg.get("derotate_yaw") and not all("yaw_deg" in o for o in orient):
        raise SystemExit(f"{opath}: derotate_yaw run without yaw in orientation.json")
    R = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3) for o in orient}
    mask_ch = list(cfg.get("mask_gps_channels", []))
    model = evaluate._latest_model(run, device)
    scale = np.array([dataset.GPS_SCALE[c] for c in GPS_CHANNELS], np.float32)
    rows = np.asarray(rows)
    out = None
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        d = read_rows(windows, F, r, ("acc", "acc_mask", "gps", "gps_mask") if keep_gps else ("acc", "acc_mask"))
        acc, am = d["acc"], d["acc_mask"]
        if step_g > 0 or clip_g > 0:
            acc = dataset.common_format(acc, am, step_g, clip_g)
        Rr = np.stack([R[b] for b in F["bird"].to_numpy()[r]])
        a = np.einsum("ntj,nij->nti", acc.astype(np.float32), Rr).astype(np.float16) * am[:, :, None]
        if keep_gps:
            graw, gm = d["gps"], d["gps_mask"].copy()
            g = graw / scale
            ih = GPS_CHANNELS.index("eobs:horizontal-accuracy-estimate")
            g[:, :, ih] = np.log1p(graw[:, :, ih])
            for c in mask_ch:
                gm[:, :, GPS_CHANNELS.index(c)] = False
            g = (np.clip(g, -dataset.GPS_CLIP, dataset.GPS_CLIP) * gm).astype(np.float16)
        else:
            g = np.zeros((len(r), 10, len(GPS_CHANNELS)), np.float16)
            gm = np.zeros(g.shape, bool)
        h = encode(model, a, am, g, gm, device)[0]
        if out is None:
            out = np.zeros((len(rows), h.shape[1]), np.float32)
        out[s:s + len(r)] = h
    return out if out is not None else np.zeros((0, 0), np.float32)


def g6_rows(ctx: Ctx) -> dict:
    def make():
        F = ctx.F
        dom = ((F["sensitivity_k"].to_numpy() == 1) & (F["acc_samples"].to_numpy() >= FULL)
               & (F["gps_fixes"].to_numpy() >= G6_MIN_FIXES))
        tr = sample_rows(ctx, "G6tr", dom, ctx.train, CAPS["G6_train"], "G6", 0)
        te = sample_rows(ctx, "G6te", dom, ctx.test, CAPS["G6_test"], "G6", 1)
        Ltr, oktr = gps_label_rows(ctx.windows, F, tr)
        Lte, okte = gps_label_rows(ctx.windows, F, te)
        tr, Ltr, te, Lte = tr[oktr], Ltr[oktr], te[okte], Lte[okte]
        bte = F["bird"].to_numpy()[te]
        qual, counts = [], {}
        for j, lab in enumerate(GPS_LABELS):
            pos_birds = pd.Series(bte[Lte[:, j]]).value_counts()
            counts[lab] = {"test_positives": int(Lte[:, j].sum()), "test_birds_with_positives": int(len(pos_birds)),
                           "train_positives": int(Ltr[:, j].sum())}
            if Lte[:, j].sum() >= G6_MIN_POS and len(pos_birds) >= G6_MIN_BIRDS:
                qual.append(lab)
        return {"tr": tr, "te": te, "Ltr": Ltr, "Lte": Lte, "qualifying": qual, "counts": counts}
    return ctx.memo("g6rows", make)


def _auc_parts_file(out_dir: Path | None, row: str, rep_name: str, parts: dict) -> dict | None:
    """Write per-test-row (bird, part, y, s) for the paired bootstrap; returns the relative file name."""
    if out_dir is None:
        return None
    (out_dir / "boot").mkdir(parents=True, exist_ok=True)
    name = f"boot/{row}_{rep_name}.parquet"
    pd.concat([pd.DataFrame({"bird": b, "part": p, "y": y.astype(np.int8), "s": s.astype(np.float32)})
               for p, (b, y, s) in parts.items()], ignore_index=True).to_parquet(out_dir / name, index=False)
    return name


def row_G6(ctx, rep, cache, out_dir=None):
    g = g6_rows(ctx)
    tr, te = g["tr"], g["te"]
    if not len(tr) or not len(te):
        return {"skipped": "no windows"}
    Xtr, Xte = rep.nogps(tr), rep.nogps(te)
    bte = ctx.F["bird"].to_numpy()[te]
    per, parts = {}, {}
    wq = _quiet()
    try:
        for j, lab in enumerate(GPS_LABELS):
            if g["Ltr"][:, j].sum() < 2 or (~g["Ltr"][:, j]).sum() < 2 or len(set(g["Lte"][:, j])) < 2:
                per[lab] = None
                continue
            m, b = logit_fit(Xtr, g["Ltr"][:, j].astype(int), Xte)
            s = m.decision_function(b)
            per[lab] = float(roc_auc_score(g["Lte"][:, j], s))
            if lab in g["qualifying"]:
                parts[lab] = (bte, g["Lte"][:, j], s)
        sec = {}
        for lab in VEDBA_LABELS:
            if lab not in ctx.F:
                continue
            ytr, yte = ctx.F[lab].to_numpy(float)[tr], ctx.F[lab].to_numpy(float)[te]
            a, c = np.isfinite(ytr), np.isfinite(yte)
            if len(set(ytr[a])) < 2 or len(set(yte[c])) < 2:
                sec[lab] = None
                continue
            m, b = logit_fit(Xtr[a], ytr[a].astype(int), Xte[c])
            sec[lab] = float(roc_auc_score(yte[c], m.decision_function(b)))
    finally:
        wq.__exit__(None, None, None)
    q = [per[l_] for l_ in g["qualifying"] if per.get(l_) is not None]
    fn = _auc_parts_file(out_dir, "G6", rep.name, parts)
    out = {"score": float(np.mean(q)) if q else np.nan, "per_class": per, "qualifying": g["qualifying"],
           "class_counts": g["counts"], "n_birds": int(len(set(bte))), "n": int(len(te)), "n_train": int(len(tr)),
           "secondary_vedba_labels": {"note": "features.parquet labels use VeDBA (partly circular)", "auc": sec}}
    if fn:
        out["boot"] = {"type": "auc_rows", "file": fn, "birds": sorted(set(bte)),
                       "combine": {p: 1.0 / max(len(parts), 1) for p in parts}, "combine_mode": "mean"}
    return out


def row_G9b(ctx, rep, cache):
    F = ctx.F
    full = F["acc_samples"].to_numpy() >= FULL
    k = F["sensitivity_k"].to_numpy()
    tr = np.r_[sample_rows(ctx, "G9b_tr0", full & (k == 0), ctx.train, CAPS["G9b"], "G9b", 0),
               sample_rows(ctx, "G9b_tr1", full & (k == 1), ctx.train, CAPS["G9b"], "G9b", 1)]
    te = np.r_[sample_rows(ctx, "G9b_te0", full & (k == 0), ctx.test, CAPS["G9b"], "G9b", 2),
               sample_rows(ctx, "G9b_te1", full & (k == 1), ctx.test, CAPS["G9b"], "G9b", 3)]
    if len(set(k[tr])) < 2 or len(set(k[te])) < 2:
        return {"skipped": "one recording mode only"}
    wq = _quiet()
    try:
        m, b = logit_fit(rep.nogps(tr), k[tr], rep.nogps(te))
        pred = m.predict(b)
    finally:
        wq.__exit__(None, None, None)
    bte, yte = F["bird"].to_numpy()[te], k[te]
    per = {str(bb): [int(((bte == bb) & (yte == 0)).sum()), int(((bte == bb) & (yte == 0) & (pred == 0)).sum()),
                     int(((bte == bb) & (yte == 1)).sum()), int(((bte == bb) & (yte == 1) & (pred == 1)).sum())]
           for bb in np.unique(bte)}
    boot = {"type": "bacc_birds", "birds": per}
    return {"score": boot_value(boot, Path("."), {b_: 1.0 for b_ in per}), "n_birds": len(per), "n": int(len(te)),
            "boot": boot}


# ================================================================ G7

def bout_shares(L: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Shares of the GPS-only labels and of 'unlabelled' among a bout's windows with >= 8 valid fixes."""
    if not ok.any():
        return np.full(L.shape[1] + 1, np.nan)
    Lk = L[ok]
    return np.r_[Lk.mean(0), (~Lk.any(1)).mean()]


def length_class(n_acc) -> np.ndarray:
    """0: 40-63 samples, 1: 64-189, 2: full (>= FULL)."""
    n_acc = np.asarray(n_acc)
    return np.where(n_acc >= FULL, 2, np.where(n_acc >= 64, 1, 0))


def sched_rows(f: pd.DataFrame) -> np.ndarray:
    """Positions of scheduled-domain windows with a measured local solar hour (G7c days)."""
    return np.flatnonzero((f["sensitivity_k"].to_numpy() == 0) & (f["acc_samples"].to_numpy() >= SCHED_MIN)
                          & ~f["lsh_filled"].to_numpy())


def day_strata(f: pd.DataFrame, rows) -> np.ndarray:
    """Local solar hour x length class: windows of two days are compared only within the same stratum."""
    rows = np.asarray(rows, int)
    hr = np.floor(f["local_solar_hour"].to_numpy()[rows]).astype(int) % 24
    return hr * 3 + length_class(f["acc_samples"].to_numpy()[rows])


def day_table(f: pd.DataFrame) -> dict:
    """Eligible days of one bird (anomaly.eligible_days on hours) and their stratum count matrix."""
    i0 = sched_rows(f)
    dd = f["day"].to_numpy()[i0]
    hr = np.floor(f["local_solar_hour"].to_numpy()[i0]).astype(int) % 24
    st = day_strata(f, i0)
    el = np.sort(anomaly.eligible_days(dd, hr))
    C = np.zeros((len(el), 72), int)
    pos = {d: i for i, d in enumerate(el)}
    for d, s in zip(dd, st):
        if d in pos:
            C[pos[d], s] += 1
    return {"days": el, "C": C}


def day_scorable(C: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """cap[i, j] = windows two days can each give in common strata; a day's clean score is finite when >= DAY_MIN_PAIRS
    other days have cap >= DAY_M; the bird's robust z needs >= MIN_Z_GROUP finite days."""
    cap = np.minimum(C[:, None, :], C[None]).sum(2) if len(C) else np.zeros((0, 0), int)
    ok = cap >= anomaly.DAY_M
    np.fill_diagonal(ok, False)
    fin = ok.sum(1) >= anomaly.DAY_MIN_PAIRS
    return ok, fin, bool(fin.sum() >= anomaly.MIN_Z_GROUP)


def bout_scorable(T: pd.DataFrame) -> np.ndarray:
    """Bout content / sequence z are finite when the bout has >= BOUT_MIN_REF same-bin bouts of the bird on other
    days (whole-record fallback) and its bin group has >= MIN_Z_GROUP such bouts (anomaly.score_bouts)."""
    if not len(T):
        return np.zeros(0, bool)
    day, b = T["day"].to_numpy(), T["bin"].to_numpy()
    refs = np.array([((b == b[i]) & (day != day[i])).sum() for i in range(len(T))])
    fin = refs >= anomaly.BOUT_MIN_REF
    grp = pd.Series(fin).groupby(b).transform("sum").to_numpy()
    return fin & (grp >= anomaly.MIN_Z_GROUP) & (b >= 0)


def circular_day_gap(a, b) -> np.ndarray:
    g = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 365.25
    return np.minimum(g, 365.25 - g)


def make_injections(windows: Path, split: dict, workers: int = 1) -> tuple[pd.DataFrame, dict]:
    """G7 host/donor lists (model-free; see module docstring) and their counts. Hosts: each test set (search, locked)
    in turn, bird-balanced, drawn only among scorable units; donors from that set plus the training birds."""
    train = birds_with_windows(windows, split["train"])
    sets = {"search": birds_with_windows(windows, split["search"]), "locked": birds_with_windows(windows, split["locked"])}
    allb = sorted(set(train) | set(sets["search"]) | set(sets["locked"]))
    log(f"injections: reading {len(allb)} birds")
    frames = {f["bird"].iat[0]: f for f in Parallel(n_jobs=workers)(
        delayed(anomaly.load_bird_frame)(windows / b) for b in allb) if f is not None}

    def bird_tables(b):
        f = frames[b]
        T = anomaly.bird_bouts(f)
        if len(T):
            rows = np.concatenate(list(T["rows"]))
            g = read_rows(windows, f.assign(bird=b), rows, ("gps", "gps_mask"))
            L, ok = gps_labels(g["gps"], g["gps_mask"])
            ends = np.cumsum([len(r) for r in T["rows"]])
            T["shares"] = [bout_shares(L[e - len(r):e], ok[e - len(r):e]) for r, e in zip(T["rows"], ends)]
            T["bird"] = b
            T["scorable"] = bout_scorable(T)
        return T, day_table(f)

    tabs = dict(zip(allb, Parallel(n_jobs=workers)(delayed(bird_tables)(b) for b in allb)))
    rows, stats = [], {}
    rb, rd = rng_of("G7a"), rng_of("G7c")
    for set_name in ("search", "locked"):
        hosts, pool = sets[set_name], sorted(set(train) | set(sets[set_name]))
        BT = pd.concat([tabs[b][0] for b in pool if len(tabs[b][0])], ignore_index=True) if pool else pd.DataFrame()
        if len(BT):
            BT = BT[np.isfinite(np.stack(BT["shares"].to_numpy())).all(1)].reset_index(drop=True)
        hb = BT["bird"].isin(hosts).to_numpy() if len(BT) else np.zeros(0, bool)
        st = {"bout_hosts_listed": int(hb.sum()), "bout_hosts_scorable": int((hb & BT["scorable"].to_numpy()).sum())
              if len(BT) else 0}
        if len(BT):
            rows += _pick_bouts(BT, hosts, CAPS["G7_bout_hosts"], rb, set_name)
        r_d, st_d = _pick_days({b: tabs[b][1] for b in hosts}, frames, CAPS["G7_day_hosts"], rd, set_name)
        rows += r_d
        stats[set_name] = {**st, **st_d}
    cols = ["set", "kind", "bird", "bin", "host_rows", "donor_bird", "donor_rows", "host_day", "donor_day"]
    df = pd.DataFrame(rows, columns=cols)
    for (s, k), g in df.groupby(["set", "kind"]):
        stats[s][k] = {"injections": int(len(g)), "host_birds": int(g["bird"].nunique()),
                       "donor_birds": int(g["donor_bird"].nunique()),
                       "donors": int(len({(d, tuple(r)) for d, r in zip(g["donor_bird"], g["donor_rows"])}))}
    return df, stats


def _pick_bouts(BT: pd.DataFrame, hosts, n: int, rng, set_name: str) -> list[dict]:
    """anomaly.pick_bout_injections restricted to scorable host bouts of the host birds and to the kinds G7 uses,
    returning (bird, rows). The novel-mix test uses the host bird's bouts of the bin within +-45 d, the window of the
    scoring's first level (anomaly.SEASON_DAYS[0]); when fewer than BOUT_MIN_REF such bouts exist the scoring falls
    back to +-90 d / the whole record, while novelty stays defined on +-45 d (as anomaly v3)."""
    hix = np.flatnonzero(BT["bird"].isin(hosts).to_numpy() & BT["scorable"].to_numpy())
    if not len(hix):
        return []
    _, inv, cnt = np.unique(BT["bird"].to_numpy()[hix], return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    pick = rng.choice(hix, min(n, len(hix)), replace=False, p=w / w.sum())
    shares = np.stack(BT["shares"].to_numpy())
    bird, bins, days = BT["bird"].to_numpy(), BT["bin"].to_numpy(), BT["day"].to_numpy()
    out = []
    for j in pick:
        hb, hr = bird[j], np.asarray(BT["rows"].iat[j])
        base = {"set": set_name, "bird": hb, "bin": int(bins[j]), "host_rows": hr.tolist(), "host_day": int(days[j]),
                "donor_day": -1}
        out.append({**base, "kind": "shuffle", "donor_bird": hb, "donor_rows": hr[rng.permutation(len(hr))].tolist()})
        l1 = np.abs(shares - shares[j]).sum(1)
        own = np.flatnonzero((bird == hb) & (bins == bins[j]) & (np.abs(days - days[j]) <= anomaly.SEASON_DAYS[0]))
        own = own[own != j]
        other = np.flatnonzero((bird != hb) & (bins == bins[j]))
        novel = np.zeros(len(BT), bool)
        if len(own) and len(other):
            mind = np.abs(shares[other][:, None, :] - shares[own][None]).sum(2).min(1)
            novel[other[mind >= anomaly.MIX_DIFF]] = True
        for kind, sel in (("novel_mix", novel), ("same_mix", l1 <= anomaly.MIX_SAME)):
            cand = np.flatnonzero(sel & (bird != hb) & (bins == bins[j]))
            if not len(cand):
                continue
            dj = rng.choice(cand)
            dr = np.asarray(BT["rows"].iat[dj])
            if len(dr) > len(hr):
                st = rng.integers(0, len(dr) - len(hr) + 1)
                dr = dr[st:st + len(hr)]
            out.append({**base, "kind": kind, "donor_bird": bird[dj], "donor_rows": dr.tolist()})
    return out


def _pick_days(tabs: dict, frames: dict, n: int, rng, set_name: str) -> tuple[list[dict], dict]:
    """anomaly.pick_day_injections, other-season kind only: donor = a day of the same bird >= OTHER_SEASON_DAYS away
    in calendar (circular) day of year. Hosts are drawn only among scorable days (clean score finite, the bird's z
    defined, and >= 1 donor whose content can be compared with >= DAY_MIN_PAIRS other days by hour x length class)."""
    cand = []  # (bird, host index, donor indices)
    listed = 0
    for b, t in tabs.items():
        el = t["days"]
        listed += len(el)
        if not len(el):
            continue
        ok, fin, bird_ok = day_scorable(t["C"])
        if not bird_ok:
            continue
        for i in np.flatnonzero(fin):
            gap = circular_day_gap(el, el[i])
            ex = np.ones(len(el), bool)
            ex[i] = False
            don = [j for j in np.flatnonzero(gap >= anomaly.OTHER_SEASON_DAYS)
                   if (ok[j] & ex & (np.arange(len(el)) != j)).sum() >= anomaly.DAY_MIN_PAIRS]
            if don:
                cand.append((b, i, np.array(don)))
    stats = {"day_hosts_listed": int(listed), "day_hosts_scorable": len(cand)}
    if not cand:
        return [], stats
    birds = np.array([c[0] for c in cand])
    _, inv, cnt = np.unique(birds, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    out = []
    for j in rng.choice(len(cand), min(n, len(cand)), replace=False, p=w / w.sum()):
        b, i, don = cand[j]
        el = tabs[b]["days"]
        dj = int(rng.choice(don))
        f = frames[b]
        r = sched_rows(f)
        r = r[f["day"].to_numpy()[r] == el[dj]]
        out.append({"set": set_name, "kind": "other_season", "bird": b, "bin": -1, "host_rows": [],
                    "donor_bird": b, "donor_rows": r.tolist(), "host_day": int(el[i]), "donor_day": int(el[dj])})
    return out, stats


def write_injections(path: Path, df: pd.DataFrame) -> None:
    """Create ``path`` exclusively (FileExistsError if it exists: the list is frozen)."""
    import io

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    with open(path, "xb") as fh:
        fh.write(buf.getvalue())


def load_injections(path: Path) -> tuple[pd.DataFrame, dict]:
    """The frozen list; the scorecard refuses to run without it (it is created only by
    scripts/scorecard_precount.py --freeze, never here)."""
    if not path.exists():
        raise SystemExit(f"{path} missing: draw it with scripts/scorecard_precount.py and freeze it with --freeze")
    df = pd.read_parquet(path)
    info = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "counts": {f"{s}|{k}": int(v) for (s, k), v in df.groupby(["set", "kind"]).size().items()}}
    return df, info


def _g7_bird(Fb: pd.DataFrame, Hb: np.ndarray, bouts: list, days: list, pca: tuple, seed: list) -> dict:
    """Clean and injected bout / day z of one host bird (anomaly v3 scoring, host bird only)."""
    out = {"bout": [], "day": []}
    T = anomaly.bird_bouts(Fb)
    if len(T) and bouts:
        first = {int(r[0]): i for i, r in enumerate(T["rows"])}
        vec = [anomaly.bout_vectors(Hb[r]) for r in T["rows"]]
        M, Dv, Sm = np.stack([v[0] for v in vec]), np.stack([v[1] for v in vec]), np.array([v[2] for v in vec])
        keep = [e for e in bouts if e["host_first"] in first]
        if keep:
            # injected queries, then (level-2 runs) the same crops without injection as the clean host queries
            qs = [e["seq"] for e in keep] + [e["clean_seq"] for e in keep if "clean_seq" in e]
            ev = [anomaly.bout_vectors(q) for q in qs]
            host = np.array([first[e["host_first"]] for e in keep])
            hq = np.r_[host, host[[i for i, e in enumerate(keep) if "clean_seq" in e]]]
            ex = {"M": np.stack([v[0] for v in ev]), "D": np.stack([v[1] for v in ev]),
                  "S": np.array([v[2] for v in ev]), "host": hq}
            br = anomaly.score_bouts(T, M, Dv, Sm, ex)
            seq_c = np.fmax(br["order_z"], br["smooth_z"])
            cz_x = br["extra"]["content_z"]
            seq_x = np.fmax(br["extra"]["order_z"], br["extra"]["smooth_z"])
            n = len(keep)
            ci = {i: n + j for j, i in enumerate(i_ for i_, e in enumerate(keep) if "clean_seq" in e)}
            for i, (e, h) in enumerate(zip(keep, host)):
                if e["kind"] == "shuffle":
                    clean = seq_x[ci[i]] if i in ci else seq_c[h]
                    out["bout"].append((e["kind"], float(seq_x[i]), float(clean)))
                else:
                    clean = cz_x[ci[i]] if i in ci else br["content_z"][h]
                    out["bout"].append((e["kind"], float(cz_x[i]), float(clean)))
    if days:
        i0 = sched_rows(Fb)
        hr = np.floor(Fb["local_solar_hour"].to_numpy()[i0]).astype(int) % 24
        hour = day_strata(Fb, i0)  # windows are matched by hour x length class in every day comparison
        X = ((Hb[i0] - pca[0]) @ pca[1].T).astype(np.float32)
        dd = Fb["day"].to_numpy()[i0]
        elig = np.sort(anomaly.eligible_days(dd, hr))
        rng = np.random.default_rng(seed)
        sc = np.array([anomaly.score_day(X, dd, hour, np.flatnonzero(dd == d_), d_, rng, days=elig)[0] for d_ in elig])
        z, par = anomaly.robust_z(sc, np.zeros(len(sc), int))
        zmap = dict(zip(elig.tolist(), z.tolist()))
        for e in days:
            s, _ = anomaly.score_day(X, dd, hour, np.zeros(0, int), e["host_day"], rng, e["donor_day"],
                                     ((e["seq"] - pca[0]) @ pca[1].T).astype(np.float32), e["hour"], days=elig)
            zi = (s - par[0][0]) / par[0][1] if 0 in par else np.nan
            clean = zmap.get(e["host_day"], np.nan)
            if "clean_seq" in e:  # level-2: the same crop without injection
                s0, _ = anomaly.score_day(X, dd, hour, np.zeros(0, int), e["host_day"], rng, None,
                                          ((e["clean_seq"] - pca[0]) @ pca[1].T).astype(np.float32),
                                          e["clean_hour"], days=elig)
                clean = (s0 - par[0][0]) / par[0][1] if 0 in par else np.nan
            out["day"].append(("other_season", float(zi), float(clean)))
    return out


def seq_meta(F: pd.DataFrame) -> pd.DataFrame:
    """seq.META_COLS as the level-2 model reads them from meta.parquet (local_solar_hour unfilled)."""
    lsh = np.where(F["lsh_filled"].to_numpy(), np.nan, F["local_solar_hour"].to_numpy(float))
    return pd.DataFrame({"t_start": F["t_start"].to_numpy(), "sensitivity_k": F["sensitivity_k"].to_numpy(),
                         "acc_samples": F["acc_samples"].to_numpy(), "local_solar_hour": lsh})


def l2_embed(run: Path, pieces: dict, device: str, notime: bool) -> dict:
    """seq.contextual_embed of {key: (parent h, meta)}; each key is its own stream."""
    from . import seq

    if not pieces:
        return {}
    return seq.contextual_embed(run, {k: v[0] for k, v in pieces.items()}, {k: v[1] for k, v in pieces.items()},
                                device, notime)


def l2_all(ctx: Ctx, run: Path, Hp: np.ndarray, notime: bool = False) -> np.ndarray:
    """Contextual h of every window of ctx.F from parent h ``Hp`` (one stream per bird)."""
    starts = ctx.F.groupby("bird", sort=False).indices
    res = l2_embed(run, {b: (Hp[ix], seq_meta(ctx.F.iloc[ix])) for b, ix in starts.items()}, ctx.device, notime)
    out = None
    for b, ix in starts.items():
        if out is None:
            out = np.zeros((len(ctx.F), res[b].shape[1]), np.float32)
        out[ix] = res[b]
    return out


def level2_g7_seqs(ctx: Ctx, rep: Rep, inj: pd.DataFrame, part: str) -> dict:
    """Level-2 runs: injections made in the PARENT h stream of the host bird (a crop of +-max_len windows around
    the bout or day), re-run through seq.contextual_embed; the same crop without injection gives the clean query.
    Returns {injection index: {seq, clean_seq[, hour, clean_hour]}} (unit rows)."""
    l2 = rep.l2
    L = int(json.loads((l2["run"] / "config.json").read_text()).get("max_len", 512))
    F, Hp = ctx.F, l2["parent_H"]
    starts = F.groupby("bird", sort=False).indices
    pieces, query = {}, {}
    for e in inj.itertuples():
        ix = starts[e.bird]
        Fb = F.iloc[ix]
        hp = np.asarray(Hp[ix], np.float32)
        mb = seq_meta(Fb)
        if part == "bouts":
            hr = np.asarray(e.host_rows, int)
            lo, hi = max(0, hr[0] - L), min(len(ix), hr[-1] + 1 + L)
            ck = f"{e.bird}|clean|{hr[0]}"
            if ck not in pieces:
                pieces[ck] = (hp[lo:hi], mb.iloc[lo:hi].reset_index(drop=True))
            dsrc = np.asarray(Hp[starts[e.donor_bird][np.asarray(e.donor_rows, int)]], np.float32)
            h = hp[lo:hi].copy()
            m = mb.iloc[lo:hi].reset_index(drop=True)
            q = hr - lo
            nd = min(len(dsrc), len(q))
            h[q[:nd]] = dsrc[:nd]
            drop = q[nd:]
            keep_ = np.setdiff1d(np.arange(len(h)), drop)
            qpos = np.searchsorted(keep_, q[:nd])
            ik = f"{e.bird}|inj|{e.Index}"
            pieces[ik] = (h[keep_], m.iloc[keep_].reset_index(drop=True))
            query[e.Index] = (ik, qpos, ck, q, None, None)
        else:
            day = Fb["day"].to_numpy()
            hd = np.flatnonzero(day == e.host_day)
            lo, hi = max(0, hd[0] - L), min(len(ix), hd[-1] + 1 + L)
            ixd = starts[e.donor_bird]
            Fd = F.iloc[ixd]
            dd = np.flatnonzero(Fd["day"].to_numpy() == e.donor_day)
            md = seq_meta(Fd.iloc[dd])
            md["t_start"] = md["t_start"] + pd.Timedelta(days=int(e.host_day) - int(e.donor_day))
            before, after = np.arange(lo, hd[0]), np.arange(hd[-1] + 1, hi)
            h = np.concatenate([hp[before], np.asarray(Hp[ixd[dd]], np.float32), hp[after]])
            m = pd.concat([mb.iloc[before], md, mb.iloc[after]], ignore_index=True)
            order = np.argsort(m["t_start"].to_numpy(), kind="stable")
            h, m = h[order], m.iloc[order].reset_index(drop=True)
            sel0 = np.isin(dd, sched_rows(Fd))
            inv = np.argsort(order)
            qpos = inv[len(before) + np.flatnonzero(sel0)]
            hsel = np.isin(hd, sched_rows(Fb))
            ik, ck = f"{e.bird}|inj|{e.Index}", f"{e.bird}|clean|{e.Index}"
            pieces[ik] = (h, m)
            pieces[ck] = (hp[lo:hi], mb.iloc[lo:hi].reset_index(drop=True))
            query[e.Index] = (ik, qpos, ck, hd[hsel] - lo, day_strata(Fd, dd[sel0]), day_strata(Fb, hd[hsel]))
    res = l2_embed(l2["run"], pieces, ctx.device, l2["notime"])
    out = {}
    for i, (ik, qp, ck, cq, hr_i, hr_c) in query.items():
        o = {"seq": anomaly.norm_rows(res[ik][qp]), "clean_seq": anomaly.norm_rows(res[ck][cq])}
        if hr_i is not None:
            o["hour"], o["clean_hour"] = hr_i, hr_c
        out[i] = o
    return out


def g7_score(ctx: Ctx, rep: Rep, part: str) -> dict:
    """part 'bouts' (G7a, G7b, G7d) or 'days' (G7c): per injection (host bird, kind, injected z, clean host z)."""
    if ctx.inj is None:
        return {"skipped": "no injection list"}
    U = rep.unit_rows(ctx)
    F = ctx.F
    starts = F.groupby("bird", sort=False).indices
    inj = ctx.inj[ctx.inj["set"] == ("locked" if ctx.final else "search")]
    inj = inj[inj["kind"].isin(["other_season"] if part == "days" else ["novel_mix", "same_mix", "shuffle"])]
    inj = inj[inj["bird"].isin(ctx.test) & inj["donor_bird"].isin(starts.keys())]
    pca = None
    if part == "days":
        sched = (F["acc_samples"].to_numpy() >= SCHED_MIN) & (F["sensitivity_k"].to_numpy() == 0)
        fit = sample_rows(ctx, "G7pca", sched, ctx.train, CAPS["G7_pca_per_bird"], "G7c", 1)
        p = PCA(min(anomaly.DAY_PCS, U.shape[1], len(fit)), random_state=0).fit(U[fit])
        pca = (p.mean_.astype(np.float32), p.components_.astype(np.float32))
    if getattr(rep, "l2", None) and rep.l2.get("uses_env"):
        raise NotImplementedError("G7 for level-2 runs with environment inputs is not implemented (I8): the injected "
                                  "crop would need the donor's env features")
    l2 = level2_g7_seqs(ctx, rep, inj, part) if getattr(rep, "l2", None) else None
    jobs = []
    for bi, b in enumerate(sorted(set(inj["bird"]))):
        ix = starts[b]
        g = inj[inj["bird"] == b]
        bouts, days = [], []
        for e in g.itertuples():
            if l2 is not None:
                q = l2[e.Index]
                if part == "bouts":
                    bouts.append({"kind": e.kind, "host_first": int(e.host_rows[0]), "seq": q["seq"],
                                  "clean_seq": q["clean_seq"]})
                else:
                    days.append({"host_day": int(e.host_day), "donor_day": int(e.donor_day), **q})
                continue
            seq = U[starts[e.donor_bird][np.asarray(e.donor_rows, int)]]
            if part == "bouts":
                bouts.append({"kind": e.kind, "host_first": int(e.host_rows[0]), "seq": seq})
            else:
                dr = starts[e.donor_bird][np.asarray(e.donor_rows, int)]
                days.append({"host_day": int(e.host_day), "donor_day": int(e.donor_day), "seq": seq,
                             "hour": day_strata(F, dr)})
        jobs.append(delayed(_g7_bird)(F.iloc[ix].reset_index(drop=True), U[ix], bouts, days, pca,
                                      [SEED0, ROW_IDS["G7c"], bi]))
    res = Parallel(n_jobs=ctx.workers)(jobs)
    recs = [(b, *r) for b, rr in zip(sorted(set(inj["bird"])), res) for r in rr["bout" if part == "bouts" else "day"]]
    return {"records": pd.DataFrame(recs, columns=["bird", "kind", "z", "clean"]), "n_injections": int(len(inj))}


def _g7_entry(ctx, rep, cache, out_dir, part, kinds: dict, row: str):
    key = ("g7", rep.name, part)
    if key not in cache:
        cache[key] = g7_score(ctx, rep, part)
    r = cache[key]
    if "skipped" in r:
        return r
    R = r["records"]
    parts, vals = {}, {}
    for kind in kinds:
        g = R[R["kind"] == kind]
        g = g[np.isfinite(g["z"]) & np.isfinite(g["clean"])]
        b = np.r_[g["bird"].to_numpy(), g["bird"].to_numpy()]
        y = np.r_[np.ones(len(g)), np.zeros(len(g))]
        s = np.r_[g["z"].to_numpy(), g["clean"].to_numpy()]
        parts[kind] = (b, y, s)
        vals[kind] = {"auc": weighted_auc(y, s), "n_scored": int(len(g)), "n_birds": int(g["bird"].nunique())}
    score = float(sum(w * vals[k]["auc"] for k, w in kinds.items()))
    fn = _auc_parts_file(out_dir, row, rep.name, parts)
    out = {"score": score, "per_kind": vals, "n_birds": int(R["bird"].nunique()),
           "n": int(sum(v["n_scored"] for v in vals.values())), "n_injections_listed": r["n_injections"]}
    if fn:
        out["boot"] = {"type": "auc_rows", "file": fn, "birds": sorted(set(R["bird"])), "combine": kinds}
    return out


def row_G7a(ctx, rep, cache, out_dir=None):
    return _g7_entry(ctx, rep, cache, out_dir, "bouts", {"novel_mix": 1.0, "same_mix": -1.0}, "G7a")


def row_G7b(ctx, rep, cache, out_dir=None):
    return _g7_entry(ctx, rep, cache, out_dir, "bouts", {"same_mix": 1.0}, "G7b")


def row_G7d(ctx, rep, cache, out_dir=None):
    return _g7_entry(ctx, rep, cache, out_dir, "bouts", {"shuffle": 1.0}, "G7d")


def row_G7c(ctx, rep, cache, out_dir=None):
    return _g7_entry(ctx, rep, cache, out_dir, "days", {"other_season": 1.0}, "G7c")


# ================================================================ G8c, G9, G10

def _presence(birds, lab, K, cohort_of) -> dict:
    counts = pd.crosstab(birds, lab).reindex(columns=range(K), fill_value=0)
    present = counts >= states.PRESENCE_MIN_WINDOWS
    n = present.sum(axis=0)
    coh = [set(cohort_of[present.index[present[s]]]) >= {"2024", "2025"} for s in range(K)]
    return {"birds": n.astype(int).tolist(), "ok": bool((n >= states.PRESENCE_MIN_BIRDS).all() and all(coh))}


def stability_scan(X_fit, X_ev, birds_fit, birds_ev, X_all_fn, birds_all, cohort_of, k_over, ks, n_boot, seed,
                   workers) -> dict:
    """states.py rule on a sample: GMM over-clustering, Ward merge, bird-bootstrap Jaccard, presence."""
    gmm = states.fit_gmm(X_fit, k_over, seed)
    merges = states.ward_merge(gmm.means_, ks)
    post = states.posterior(states.joint(X_ev, gmm.weights_, gmm.means_, gmm.covariances_))
    ref = {K: (post @ states.onehot(merges[K], K)).argmax(1) for K in ks}
    rng = np.random.default_rng(seed)
    ub = np.unique(birds_fit)
    by = {b: np.flatnonzero(birds_fit == b) for b in ub}
    picks = [rng.choice(ub, len(ub), replace=True) for _ in range(n_boot)]
    boots = Parallel(n_jobs=workers)(delayed(states.refit_labels)(X_fit[np.concatenate([by[b] for b in pk])], X_ev,
                                                                  k_over, ks, seed + 1 + i) for i, pk in enumerate(picks))
    labs_all = {K: [] for K in ks}
    for Xc in X_all_fn():
        pa = states.posterior(states.joint(Xc, gmm.weights_, gmm.means_, gmm.covariances_))
        for K in ks:
            labs_all[K].append((pa @ states.onehot(merges[K], K)).argmax(1))
    per_k = {}
    for K in ks:
        jac = np.mean([states.jaccard_best(ref[K], bt[K], K) for bt in boots], axis=0)
        pres = _presence(birds_all, np.concatenate(labs_all[K]), K, cohort_of)
        per_k[str(K)] = {"jaccard": jac.round(4).tolist(), "jaccard_min": float(jac.min()), "presence": pres}
    good = [K for K in ks if per_k[str(K)]["jaccard_min"] >= states.JACCARD_MIN and per_k[str(K)]["presence"]["ok"]]
    return {"largest_stable_k": max(good) if good else None, "per_k": per_k, "ref": ref, "gmm": gmm, "merges": merges}


def row_G8c(ctx, rep, cache):
    if ctx.final:
        return {"skipped": "not scored in final mode"}
    if rep.is_features:
        return {"skipped": "no handcrafted reference in the spec"}
    F = ctx.F
    U = rep.unit_rows(ctx)
    sc = ctx.is_in(ctx.nonlocked)
    rng = rng_of("G8c")
    fit_idx = states.balanced_sample(F, sc, G8C["per_bird"], rng)
    ev_idx = states.balanced_sample(F, sc, G8C["eval_per_bird"], rng)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=0).fit(U[fit_idx])
    Xf, Xe = pca.transform(U[fit_idx]), pca.transform(U[ev_idx])
    birds = F["bird"].to_numpy()
    rows_all = np.flatnonzero(sc)
    cohort_of = pd.Series(np.where(F.groupby("bird")["t_start"].min().dt.year <= 2024, "2024", "2025"),
                          index=F.groupby("bird")["t_start"].min().index)

    def chunks(rows):
        for i in range(0, len(rows), 500_000):
            yield pca.transform(U[rows[i:i + 500_000]])

    main = stability_scan(Xf, Xe, birds[fit_idx], birds[ev_idx], lambda: chunks(rows_all), birds[rows_all],
                          cohort_of, G8C["k_over"], G8C["ks"], G8C["n_boot"], 0, ctx.workers)
    # sub-states within each coarse K=2 state of this fit
    m2 = main["merges"][2]
    g = main["gmm"]

    def coarse(X):
        return (states.posterior(states.joint(X, g.weights_, g.means_, g.covariances_)) @ states.onehot(m2, 2)).argmax(1)

    cf, ce = coarse(Xf), coarse(Xe)
    ca = np.concatenate([coarse(Xc) for Xc in chunks(rows_all)])
    sp = F["speed_mean"].to_numpy(float)[rows_all]
    sub = {}
    for c in (0, 1):
        f_, e_ = np.flatnonzero(cf == c), np.flatnonzero(ce == c)
        f_ = np.concatenate([rng.choice(f_[birds[fit_idx][f_] == b], min(G8C["sub_per_bird"], int((birds[fit_idx][f_] == b).sum())),
                                        replace=False) for b in np.unique(birds[fit_idx][f_])]) if len(f_) else f_
        ra = rows_all[ca == c]
        if len(f_) < 50 * G8C["sub_k_over"] or len(e_) < 50:
            sub[str(c)] = {"skipped": "too few windows"}
            continue
        s = stability_scan(Xf[f_], Xe[e_], birds[fit_idx][f_], birds[ev_idx][e_], lambda: chunks(ra), birds[ra],
                           cohort_of, G8C["sub_k_over"], G8C["sub_ks"], G8C["n_boot"], 1 + c, ctx.workers)
        sub[str(c)] = {"median_speed": float(np.nanmedian(sp[ca == c])) if np.isfinite(sp[ca == c]).any() else None,
                       "largest_stable_k_sub": s["largest_stable_k"],
                       "per_k": s["per_k"]}
    return {"score": main["largest_stable_k"], "per_k": main["per_k"], "substates": sub, "n_birds": len(ctx.nonlocked),
            "n": int(len(fit_idx)), "sizes": G8C, "pca_components": int(pca.n_components_),
            "rule": f"largest K with every state's mean bootstrap Jaccard >= {states.JACCARD_MIN} and presence "
                    f"(>= {states.PRESENCE_MIN_BIRDS} birds with >= {states.PRESENCE_MIN_WINDOWS} windows, both cohorts)"}


def row_G9(ctx, rep, cache):
    out = evaluate.nuisance_vs_features(ctx.F, rep.X, ctx.test, rng_of("G9"), posture_fn=None)
    rec = out["bird_id_holdout_h_recall_by_bird"]
    return {"score": out["bird_id_holdout_h"], "n_birds": len(rec), "chance": out["bird_id_chance"],
            "per_bird_recall": rec, "sensitivity_full_windows": out.get("sensitivity_full_windows_h"),
            "boot": {"type": "mean_birds", "birds": rec},
            "protocol": "evaluate.nuisance_vs_features: fit on the first half of each test bird's time span, test on the second"}


def row_G10(ctx, rep, cache):
    F = ctx.F
    anyr = np.ones(len(F), bool)
    flap = (F["label_flapping"].to_numpy(float) == 1) & (F["acc_samples"].to_numpy() >= FULL)
    tr = sample_rows(ctx, "G10tr", anyr, ctx.train, CAPS["G10_train"], "G10", 0)
    te = sample_rows(ctx, "G10te", anyr, ctx.test, CAPS["G10_test"], "G10", 1)
    trf = sample_rows(ctx, "G10trf", flap, ctx.train, CAPS["G10_train"], "G10", 2)
    tef = sample_rows(ctx, "G10tef", flap, ctx.test, CAPS["G10_test"], "G10", 3)
    out = {}
    for name, col in G10_TARGETS.items():
        if name == "fore_aft_tilt":
            y = ctx.foreaft
        elif name == "vedba":
            y = np.log10(np.clip(F["vedba"].to_numpy(float), 1e-3, None))
        else:
            y = F[col].to_numpy(float)
        a, b = (trf, tef) if name == "wingbeat_freq" else (tr, te)
        out[name] = window_r2(ctx, lambda r, c=col: rep.window(r, drop=c), y, a, b)
    return {"score": {k: v.get("score", np.nan) for k, v in out.items()}, "per_target": out,
            "n_birds": int(len(set(F["bird"].to_numpy()[te]))), "n": int(len(te)),
            "tolerance": f"must not drop by more than {GUARD_TOL} absolute",
            "fore_aft_target": f"{FOREAFT_COL} of --states2 (r6 roost frame), constant for every design"}


ROW_FN = {"G1": row_G1, "G1b": row_G1b, "G2": row_G2, "G3": row_G3, "G4": row_G4, "G4b": row_G4b, "G5": row_G5,
          "G6": row_G6, "G7a": row_G7a, "G7c": row_G7c, "G7d": row_G7d, "G7b": row_G7b, "G8": row_G8, "G8b": row_G8b,
          "G8c": row_G8c, "G9": row_G9, "G9b": row_G9b, "G10": row_G10}
FLOOR_FN = {"G1": floor_G1, "G1b": floor_G1, "G2": floor_G2, "G3": floor_G3, "G8": floor_G8}
OUT_DIR_ROWS = ("G6", "G7a", "G7b", "G7c", "G7d")  # rows writing per-test-row bootstrap files


def call_row(row, ctx, rep, cache, out_dir=None) -> dict:
    t0 = time.time()
    try:
        fn = ROW_FN[row]
        res = fn(ctx, rep, cache, out_dir) if row in OUT_DIR_ROWS else fn(ctx, rep, cache)
    except Exception as e:  # one failing row must not lose the others; the error is recorded
        import traceback

        res = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=5)}
        log(f"{row} ({rep.name}): ERROR {type(e).__name__}")
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ================================================================ reference cache

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _strip(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("boot", "traceback")}


def row_constants() -> dict:
    """Every module-level constant that defines a row (upper-case names), for the reference-cache key."""
    return {k: v for k, v in sorted(globals().items()) if k.isupper() and k not in ("ROWS", "ROW_FN", "FLOOR_FN")
            and isinstance(v, (int, float, str, tuple, list, dict))}


def reference_key(ctx: Ctx, args) -> str:
    inj = ctx._memo.get("inj_info", {})
    blob = json.dumps({"spec": SPEC, "windows": str(args.windows), "split": ctx.split, "final": ctx.final,
                       "states2": str(args.states2), "residency": str(args.residency), "reference": str(args.reference),
                       "injections": inj.get("sha256"), "n_perm": ctx.n_perm, "constants": row_constants()},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def reference_rows(ctx: Ctx, rows: list[str], cache_dir: Path) -> dict:
    """Handcrafted reference (sensor frame) and nuisance floor per row, computed once and stored (exclusive create)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    Fs, cols = feature_matrix(ctx.F)
    Facc = ctx.F[ACC_FEATURES].to_numpy(np.float32)
    frep = Rep("features", Fs, False, cols=cols, nogps=lambda r: Facc[r])
    out, cache = {}, {}
    for row in rows:
        p = cache_dir / f"{row}.json"
        if p.exists():
            for _ in range(60):  # another job may be writing it
                try:
                    out[row] = json.loads(p.read_text())
                    break
                except json.JSONDecodeError:
                    time.sleep(10)
            if row in out:
                continue
        log(f"reference {row}")
        ref = call_row(row, ctx, frep, cache)
        if "error" in ref:
            raise SystemExit(f"reference {row} failed ({ref['error']}); nothing cached")
        try:
            flo = FLOOR_FN[row](ctx, row) if row in FLOOR_FN else None
        except Exception as e:
            raise SystemExit(f"nuisance floor {row} failed ({type(e).__name__}: {e}); nothing cached")
        ent = {"reference": _jsonable(_strip(ref)), "floor": None if flo is None else _jsonable(_strip(flo)),
               "computed": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            with open(p, "x") as fh:
                fh.write(json.dumps(ent, indent=1))
        except FileExistsError:
            ent = json.loads(p.read_text())
        out[row] = ent
    return out


# ================================================================ scoring one run

def read_design(run: Path) -> dict:
    d = {"uses_time": False, "uses_env": False, "level": 1, "parent_run": None}
    p = run / "design.json"
    if p.exists():
        d.update(json.loads(p.read_text()))
    return d


def resolve_parent(run: Path, parent) -> Path:
    p = Path(parent)
    return p if p.is_absolute() else (run.parent / p)


def load_h(run: Path, ctx: Ctx, key: str = "h") -> np.ndarray | None:
    starts = ctx.F.groupby("bird", sort=False).indices
    out = None
    for b, ix in starts.items():
        e = np.load(run / "embeddings" / f"{b}.npz")
        if key not in e.files:
            return None
        h = e[key]
        if len(h) != len(ix):
            raise ValueError(f"{run.name}: embedding rows differ from meta rows")
        if out is None:
            out = np.zeros((len(ctx.F), h.shape[1]), np.float16)
        out[ix] = h
    return out


def contamination(run: Path, ctx: Ctx) -> str | None:
    hold = set(json.loads((run / "holdout.json").read_text())) if (run / "holdout.json").exists() else set()
    locked = set(json.loads((run / "locked.json").read_text())) if (run / "locked.json").exists() else set()
    if ctx.final:
        return "contaminated: run trained on locked birds" if set(ctx.test) - hold - locked else None
    return "contaminated: run trained on search birds" if set(ctx.test) - hold else None


def score_run(ctx: Ctx, run: Path, rows: list[str], refs: dict, out_dir: Path) -> dict:
    t0 = time.time()
    design = read_design(run)
    enc_run = resolve_parent(run, design["parent_run"]) if int(design.get("level", 1)) == 2 and design.get("parent_run") else run
    log(f"run {run.name}: loading h")
    H = load_h(run, ctx)
    if H is None:
        raise SystemExit(f"{run}: embeddings without key 'h'")
    Hnt = load_h(run, ctx, "h_notime") if design.get("uses_time") else None
    enc_cache = {}
    level2 = enc_run != run

    def nogps(r):
        if level2:  # parent re-embedded with GPS zeroed for every window, then the level-2 model (whole streams)
            if "all" not in enc_cache:
                log(f"{run.name}: level-2 GPS-zeroed re-embedding of all windows")
                Hz = encode_rows(enc_run, ctx.windows, ctx.F, np.arange(len(ctx.F)), ctx.device)
                enc_cache["all"] = l2_all(ctx, run, Hz)
            return enc_cache["all"][r]
        key = r.tobytes()
        if key not in enc_cache:
            enc_cache[key] = encode_rows(enc_run, ctx.windows, ctx.F, r, ctx.device)
        return enc_cache[key]

    rep = Rep("h", H, True, nogps=nogps)
    rep_t = Rep("h_notime", Hnt, True, nogps=nogps) if Hnt is not None else rep
    if level2:
        Hp = load_h(enc_run, ctx)
        rep.l2 = {"run": run, "parent_H": Hp, "notime": False, "uses_env": bool(design.get("uses_env"))}
        if rep_t is not rep:
            rep_t.l2 = {**rep.l2, "notime": True}
    contam = contamination(run, ctx)
    cache = {}
    entries = {}
    for row in rows:
        r_rep = rep_t if (row in TIME_ROWS and design.get("uses_time")) else rep
        log(f"{run.name} {row}")
        res = call_row(row, ctx, r_rep, cache, out_dir)
        entries[row] = assemble(row, res, refs.get(row, {}), design, contam, r_rep,
                                enc_run if row in ("G6", "G9b") else None, run)
        if level2 and row.startswith("G7"):
            entries[row]["flags"].append("injected into the parent h stream, re-run through the level-2 model")
    md = {"spec": SPEC, "split_sha256": hashlib.sha256(json.dumps(ctx.split, sort_keys=True).encode()).hexdigest(),
          "run": run.name, "run_dir": str(run), "variant": variant_of(run.name),
          "final": ctx.final, "design": design, "row_ids": ROW_IDS, "seed": f"numpy default_rng([{SEED0}, row id, ...])",
          "birds": ctx.info, "caps": CAPS, "g8c_sizes": G8C, "n_perm": ctx.n_perm, "n_folds": N_FOLDS,
          "n_repeats": N_REPEATS, "injections": ctx._memo.get("inj_info"),
          "reference_cache": str(ctx._memo.get("ref_dir")), "contamination": contam,
          "choices": __doc__.split("Row choices where the spec leaves room")[1].strip(),
          "rows": entries, "runtime_s": round(time.time() - t0, 1)}
    return _jsonable(md)


def assemble(row, res, ref, design, contam, rep, enc_run, run) -> dict:
    goal, kind, metric, better, mine = ROWS[row]
    ent = {"goal": goal, "kind": kind, "metric": metric, "better": better, "min_effect": mine,
           "model": res.get("score"), "n_birds": res.get("n_birds"), "n": res.get("n"), "flags": []}
    r = ref.get("reference") or {}
    f = ref.get("floor")
    ent["reference"] = r.get("score")
    ent["floor"] = f.get("score") if f else None
    if row == "G4":
        nq = r.get("n_birds") or 0
        ent["kind"] = "S" if nq >= G4_MIN_BIRDS else "D"
        ent["kind_rule"] = f"S if >= {G4_MIN_BIRDS} birds qualify ({nq})"
    if row == "G8":
        fl = ent["floor"]
        ent["kind"] = "S" if fl is not None and fl < G8_FLOOR_MAX else "D"
        ent["kind_rule"] = f"S if the nuisance floor < {G8_FLOOR_MAX}"
        if fl is not None and fl >= G8_FLOOR_MAX:
            ent["flags"].append("not testable in this design (nuisance floor >= 0.9)")
    if "error" in res:
        ent["flags"].append("error")
    if "skipped" in res:
        ent["flags"].append(f"skipped: {res['skipped']}")
    if row in WINDOW_ROWS and contam:
        ent["flags"].append(contam)
    if row in TIME_ROWS and design.get("uses_time"):
        ent["flags"].append("scored on h_notime" if rep.name == "h_notime" else "time/position-informed: not comparable")
    if row in ENV_ROWS and design.get("uses_env"):
        ent["flags"].append("environment-informed: not comparable")
    if enc_run is not None:
        ent["reembedded_with"] = enc_run.name
        if enc_run != run:
            ent["flags"].append(f"parent run {enc_run.name} re-embedded with GPS zeroed, then the level-2 model")
    ent["suspended"] = False
    if mine is not None and isinstance(ent["model"], (int, float)) and isinstance(ent["floor"], (int, float)):
        if np.isfinite(ent["model"]) and ent["model"] - ent["floor"] < mine:
            ent["suspended"] = True
            ent["flags"].append("explained by nuisance")
    ent["details"] = _strip(res)
    if "boot" in res:
        ent["boot"] = res["boot"]
    if row == "G10" and "per_target" in res:
        ent["reference"] = {k: v.get("score") for k, v in (r.get("per_target") or {}).items()}
        ent["boot"] = {k: v["boot"] for k, v in res["per_target"].items() if "boot" in v}
        ent["details"]["per_target"] = {k: _strip(v) for k, v in res["per_target"].items()}
    return ent


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, dict):
        return "; ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "; ".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_md(path: Path, card: dict) -> None:
    L = [f"# Scorecard: {card['run']}", "", f"Spec: {card['spec']}. Mode: {'final (locked birds)' if card['final'] else 'search'}. "
         f"Test birds: {card['birds'].get('test_birds')}; training birds: {card['birds'].get('train_birds')}; "
         f"non-locked birds: {card['birds'].get('nonlocked_birds')}.", "",
         "| Id | Goal | Kind | Metric (better) | Model | Reference (sensor frame) | Nuisance floor | Birds | n | Flags |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for row, e in card["rows"].items():
        arrow = "higher" if e["better"] == "up" else "lower"
        L.append(f"| {row} | {e['goal']} | {e['kind']} | {e['metric']} ({arrow}) | {_fmt(e['model'])} | "
                 f"{_fmt(e['reference'])} | {_fmt(e['floor'])} | {_fmt(e['n_birds'])} | {_fmt(e['n'])} | "
                 f"{'; '.join(e['flags']) or ''} |")
    L += ["", "Rows S decide between designs (scripts/compare_scorecards.py); D are descriptive; G10 is a guard rail "
          f"(no target may drop by more than {GUARD_TOL}). Details, sample sizes and tests are in scorecard.json."]
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--states2", type=Path, required=True)
    ap.add_argument("--residency", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", nargs="+", choices=list(ROWS), default=None)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-perm", type=int, default=N_PERM, help="between-bird permutations (spec: 200)")
    a = ap.parse_args(argv)
    t0 = time.time()
    names = [r.name for r in a.run]
    if len(set(names)) != len(names):
        raise SystemExit(f"run names must differ: {names}")
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty; give a new directory")
    rows = [r for r in ROWS if a.rows is None or r in a.rows]
    if not (a.split.parent / "injections.parquet").exists():
        load_injections(a.split.parent / "injections.parquet")  # SystemExit: the frozen list is required
    ctx = build_context(a.windows, a.split, a.reference, a.states2, a.residency, a.final, a.workers, a.device, a.n_perm)
    # always (refuses without the frozen list): its hash is part of the reference-cache key
    ctx.inj, ctx._memo["inj_info"] = load_injections(ctx._memo["inj_path"])
    ref_dir = a.out.resolve().parent / "scorecard_reference" / f"{'final' if a.final else 'search'}_{reference_key(ctx, a)}"
    ctx._memo["ref_dir"] = ref_dir
    refs = reference_rows(ctx, rows, ref_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    for run in a.run:
        od = a.out if len(a.run) == 1 else a.out / run.name
        od.mkdir(parents=True, exist_ok=True)
        card = score_run(ctx, run, rows, refs, od)
        card["total_runtime_s"] = round(time.time() - t0, 1)
        (od / "scorecard.json").write_text(json.dumps(card, indent=1))
        write_md(od / "scorecard.md", card)
        log(f"{run.name}: written ({card['runtime_s']} s)")
    log(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
