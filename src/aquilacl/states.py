"""Stage 5a: behaviour states on a frozen SimCLR embedding, chosen by stability, smoothed and characterised.

Method
1. Prepare. ``h`` is L2-normalised and projected on the principal components that explain 95 % of the variance.
   PCA, the mixture and all refits use a bird-balanced fit sample: up to ``per_bird`` windows per bird, drawn
   with weight 1/(windows in that bird-month), so months are about equally represented within a bird.
2. Over-cluster and merge. A diagonal-covariance Gaussian mixture with ``k_over`` components is merged by Ward
   linkage on the component means (PCA space) to every K in ``k_min..k_max`` (capped at ``k_over``). A window's
   state posterior is the sum of the posteriors of the state's components; ``state`` is its argmax and
   ``p_state`` its maximum.
3. Choose K. For each K:
   (a) bird-bootstrap stability: ``n_boot`` refits on birds resampled with replacement (each drawn bird brings
       its fit-sample windows), merged to K, assigned on a fixed evaluation sample (``eval_per_bird`` windows per
       bird); per reference state, the Jaccard index with its best-matching bootstrap state (Hennig 2007),
       averaged over refits;
   (b) held-out birds: a fit without the birds in ``holdout.json`` labels the held-out birds' evaluation windows;
       ARI against the full fit;
   (c) presence (full-fit ``state`` of all windows): a bird has a state if it holds >= PRESENCE_MIN_WINDOWS of
       its windows; every state needs
       >= PRESENCE_MIN_BIRDS such birds and at least one in each cohort (year of the bird's first window <= 2024
       -> "2024", else "2025");
   (d) BIC of a K-component diagonal mixture fitted directly on the fit sample (information only).
   Rule: the largest K at which every state has mean Jaccard >= JACCARD_MIN and presence holds. If no K meets
   it, the K with the highest minimum Jaccard among K with presence (else among all K) is used and
   ``rule_met`` is false in state_selection.json. ``--k`` overrides the choice (all K are still reported).
4. Temporal smoothing. HMM over chains of windows of one bird and segment with ``k_in_segment`` stepping by
   exactly 1 (a gap breaks the chain). Emission of state s: log sum_{c in s} (w_c / W_s) N_c(x), i.e. the
   state's own sub-mixture without the state prior W_s. Deviation from "log-sum-exp without weights": the
   unweighted sum adds about log(number of components) to states built from many components, which is not
   p(x | state); the within-state renormalised weights give p(x | state) and still exclude the state frequency
   (DESIGN §6). Transitions: counts of consecutive ``state`` pairs within chains plus 1, row-normalised; start
   probabilities: ``state`` frequencies. Viterbi per chain (baseline.viterbi); single-window chains keep
   ``state``.
5. Characterise. Profiles are grouped by ``state_hmm``.

Fixed choices: PCA is fitted once and kept for all refits (it is a linear rotation of the embedding fitted on
the balanced sample; refitting it per bootstrap would add cost, not information). States are numbered by
ascending median vedba of their GMM ``state`` windows with at least 40 samples, so 0 is the quietest (profiles
are grouped by ``state_hmm``, so their vedba medians need not be strictly increasing).

Outputs in ``--out`` (the interface read by stage 5b, aquilacl.analysis)
- states.parquet: one row per window. bird, row (index within the bird's meta.parquet), t_start, segment,
  k_in_segment, pitch_rot_deg, roll_rot_deg (posture in the gravity-aligned frame),
  state (int, merged GMM), state_hmm (int), p_state (max state posterior of the GMM),
  pca_1, pca_2, pca_3.
- state_profiles.csv: one row per state_hmm (index ``state``). windows, birds, birds_2024, birds_2025; for
  vedba, odba, flap_band_frac, dom_freq_hz (windows with >= 40 samples only), pitch_rot_deg, roll_rot_deg
  (posture from the window's static acceleration rotated by the bird's R; z up), speed_mean, climb_mean,
  turn_abs_mean, sun_elev_deg, local_solar_hour, month: ``<v>_q25``, ``<v>_median``, ``<v>_q75``; gps_share;
  ``frac_<label>`` and ``n_<label>`` (known values only); ``share_sens_<k>``; share_full (acc_samples >= 190);
  share_2024; dwell_mean_windows, dwell_median_windows, dwell_p90_windows (runs of equal state_hmm within a
  chain; runs at chain ends are truncated); night_share (sun < -6 deg). Artifact checks: top_bird,
  max_bird_share, max_tag_share, sens_ratio_max (state share of a setting / its overall share, maximum over
  settings), nonfull_ratio (non-full share / overall non-full share), flight_like, and flag_bird, flag_tag,
  flag_sensitivity, flag_nonfull, flag_night_flight (thresholds in ARTIFACT, also in state_selection.json);
  jaccard and presence_birds of the GMM state with the same number (selection step 3).
  local_solar_hour and month quantiles are linear, not circular: use states.parquet for diel/seasonal patterns.
- state_selection.json: per K jaccard (per state; for the chosen K in the final state numbering, for other K in
  Ward order), jaccard_min, heldout_ari, presence (birds per state,
  cohort_ok, ok), bic; chosen_k, rule, rule_met, k_override; pca_components, pca_variance; k_over; sample sizes.
- transitions.csv: HMM transition matrix (rows from, columns to state).
- model.npz: pca_mean, pca_components, gmm_weights, gmm_means, gmm_var, merge (state of each component),
  transmat, startprob. :func:`assign` labels new embeddings with it.

Usage: python -m aquilacl.states --run <run> --windows <windows> --out <new dir> [--k K] [--k-over 40]
       [--seed 0] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from . import features
from .baseline import viterbi

JACCARD_MIN = 0.75
PRESENCE_MIN_BIRDS = 5
PRESENCE_MIN_WINDOWS = 20
FULL_SAMPLES = 190
SPECTRUM_SAMPLES = 40
ARTIFACT = {"bird_share": 0.5, "tag_share": 0.5, "sens_ratio": 2.0, "nonfull_ratio": 2.0,
            "flight_speed_ms": 5.0, "flight_vedba_g": 0.2, "night_sun_deg": -6.0, "night_flight_share": 0.1}
META_COLS = ["bird", "tag", "t_start", "segment", "k_in_segment", "sensitivity_k", "sun_elev_deg", "acc_samples",
             "gps_fixes", "local_solar_hour"]
ACC_VARS = ["vedba", "odba", "flap_band_frac", "dom_freq_hz"]
PROFILE_VARS = ACC_VARS + ["pitch_rot_deg", "roll_rot_deg", "speed_mean", "climb_mean", "turn_abs_mean",
                           "sun_elev_deg", "local_solar_hour", "month"]
N_INIT = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- loading

def rotated_posture(acc: np.ndarray, mask: np.ndarray, R: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """(pitch, roll) in degrees of the window-mean static acceleration rotated by R (sensor -> gravity on +z)."""
    out = np.full((len(acc), 2), np.nan)
    for i in range(0, len(acc), chunk):
        a = np.asarray(acc[i:i + chunk], np.float32)
        m = np.asarray(mask[i:i + chunk])
        s = features._masked_mean(features.static_component(a, m), m[:, :, None].astype(np.float32), 1) @ R.T
        out[i:i + chunk, 0] = np.degrees(np.arctan2(s[:, 0], np.hypot(s[:, 1], s[:, 2])))
        out[i:i + chunk, 1] = np.degrees(np.arctan2(s[:, 1], s[:, 2]))
    return out


def load_bird(d: Path, run: Path, R: np.ndarray):
    m = pd.read_parquet(d / "meta.parquet")
    if len(m) == 0:
        return None  # bird without windows: no other file is read
    h = np.load(run / "embeddings" / f"{d.name}.npz")["h"].astype(np.float32)
    if len(h) != len(m):
        raise ValueError(f"{d.name}: {len(m)} windows but {len(h)} embeddings")
    df = pd.concat([m[META_COLS].reset_index(drop=True), pd.read_parquet(d / "features.parquet").reset_index(drop=True)], axis=1)
    df["row"] = np.arange(len(df))
    df[["pitch_rot_deg", "roll_rot_deg"]] = rotated_posture(np.load(d / "acc.npy", mmap_mode="r"),
                                                            np.load(d / "acc_mask.npy", mmap_mode="r"), R)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    return df, h


def load(run: Path, windows: Path, workers: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """All windows with features, rotated posture, month and cohort; L2-normalised h in the same order."""
    R = {o["bird"]: np.asarray(o["R"], np.float32).reshape(3, 3) for o in json.loads((run / "orientation.json").read_text())}
    dirs = sorted(p for p in windows.iterdir() if p.is_dir())
    missing = [d.name for d in dirs if d.name not in R and len(pd.read_parquet(d / "meta.parquet", columns=["bird"]))]
    if missing:
        raise ValueError(f"no rotation in orientation.json for {missing}")
    res = Parallel(n_jobs=workers)(delayed(load_bird)(d, run, R.get(d.name)) for d in dirs)
    res = [r for r in res if r is not None]
    meta = pd.concat([r[0] for r in res], ignore_index=True)
    H = np.concatenate([r[1] for r in res])
    meta["month"] = meta["t_start"].dt.month
    first = meta.groupby("bird")["t_start"].transform("min")
    meta["cohort"] = np.where(first.dt.year <= 2024, "2024", "2025")
    return meta, H


def balanced_sample(meta: pd.DataFrame, rows: np.ndarray, n: int, rng) -> np.ndarray:
    """Up to n windows per bird, calendar months (year-month) equally weighted within the bird."""
    out = []
    sub = meta.loc[rows, ["bird"]].assign(ym=meta.loc[rows, "t_start"].dt.to_period("M").astype(str))
    for _, g in sub.groupby("bird"):
        w = 1.0 / g.groupby("ym")["ym"].transform("size").to_numpy()
        out.append(rng.choice(g.index.to_numpy(), min(n, len(g)), replace=False, p=w / w.sum()))
    return np.sort(np.concatenate(out))


# ---------------------------------------------------------------- mixture and merging

def log_density(X: np.ndarray, means: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log N(x; mean_c, diag var_c) for every window and component, (n, k)."""
    prec = 1.0 / var
    q = (X**2) @ prec.T - 2 * X @ (means * prec).T + (means**2 * prec).sum(1)
    return -0.5 * (X.shape[1] * np.log(2 * np.pi) + np.log(var).sum(1) + q)


def fit_gmm(X: np.ndarray, k: int, seed: int) -> GaussianMixture:
    with threadpool_limits(1):
        return GaussianMixture(k, covariance_type="diag", n_init=N_INIT, max_iter=300, random_state=seed).fit(X)


def ward_merge(means: np.ndarray, ks) -> dict[int, np.ndarray]:
    """State of each component for every K, replaying the first k_over-K Ward merges."""
    n = len(means)
    Z = linkage(means, "ward")
    out = {}
    for K in ks:
        parent = np.arange(2 * n - 1)
        for i in range(n - K):
            parent[int(Z[i, 0])] = parent[int(Z[i, 1])] = n + i
        root = np.arange(n)
        while (parent[root] != root).any():
            root = parent[root]
        lab = np.unique(root, return_inverse=True)[1]
        assert lab.max() + 1 == K
        out[K] = lab
    return out


def onehot(lab: np.ndarray, K: int) -> np.ndarray:
    return np.eye(K)[lab]


def joint(X, weights, means, var) -> np.ndarray:
    """log w_c + log N_c(x), float32."""
    return (log_density(X, means, var) + np.log(weights)).astype(np.float32)


def posterior(lj: np.ndarray) -> np.ndarray:
    return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))


def emissions(lj: np.ndarray, merge: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """log p(x | state) = log sum_{c in s} (w_c / W_s) N_c(x); the state prior W_s is left out."""
    K = merge.max() + 1
    W = np.bincount(merge, weights=weights, minlength=K)
    return np.stack([logsumexp(lj[:, merge == s], axis=1) - np.log(W[s]) for s in range(K)], axis=1)


def refit_labels(X, ev, k_over, ks, seed) -> dict[int, np.ndarray]:
    g = fit_gmm(X, k_over, seed)
    post = posterior(joint(ev, g.weights_, g.means_, g.covariances_))
    return {K: (post @ onehot(lab, K)).argmax(1) for K, lab in ward_merge(g.means_, ks).items()}


def jaccard_best(a: np.ndarray, b: np.ndarray, K: int) -> np.ndarray:
    """For each state of a, the Jaccard index with its best-matching state of b."""
    C = np.bincount(a * K + b, minlength=K * K).reshape(K, K).astype(float)
    union = C.sum(1)[:, None] + C.sum(0)[None, :] - C
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(C / union).max(1)


def bic_k(X, K, seed) -> float:
    with threadpool_limits(1):
        return float(GaussianMixture(K, covariance_type="diag", max_iter=300, random_state=seed).fit(X).bic(X))


# ---------------------------------------------------------------- HMM

def chains(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Window order (bird, segment, k_in_segment) and a chain-start flag in that order."""
    order = np.lexsort((meta["k_in_segment"].to_numpy(), meta["segment"].to_numpy(), meta["bird"].to_numpy()))
    b, s, k = (meta[c].to_numpy()[order] for c in ("bird", "segment", "k_in_segment"))
    start = np.ones(len(order), bool)
    start[1:] = (b[1:] != b[:-1]) | (s[1:] != s[:-1]) | (k[1:] != k[:-1] + 1)
    return order, start


def transitions(state: np.ndarray, start: np.ndarray, K: int) -> np.ndarray:
    """Counts of consecutive states within chains (+1), row-normalised. ``state`` in chain order."""
    nxt = ~start[1:]
    C = np.bincount(state[:-1][nxt] * K + state[1:][nxt], minlength=K * K).reshape(K, K) + 1.0
    return C / C.sum(1, keepdims=True)


def _viterbi_block(log_emit, starts, ends, log_T, log_pi):
    return [viterbi(log_emit[s:e], log_T, log_pi) for s, e in zip(starts, ends)]


def smooth(log_emit: np.ndarray, state: np.ndarray, start: np.ndarray, T: np.ndarray, pi: np.ndarray,
           workers: int = 1) -> np.ndarray:
    """Viterbi per chain (arrays in chain order); single-window chains keep ``state``."""
    out = state.copy()
    s = np.flatnonzero(start)
    e = np.r_[s[1:], len(start)]
    multi = (e - s) > 1
    s, e = s[multi], e[multi]
    log_T, log_pi = np.log(T), np.log(pi)
    blocks = np.array_split(np.arange(len(s)), max(1, min(len(s), 4 * workers)))
    res = Parallel(n_jobs=workers)(
        delayed(_viterbi_block)(log_emit[s[ii[0]]:e[ii[-1]]], s[ii] - s[ii[0]], e[ii] - s[ii[0]], log_T, log_pi)
        for ii in blocks if len(ii))
    for ii, paths in zip((ii for ii in blocks if len(ii)), res):
        for j, p in zip(ii, paths):
            out[s[j]:e[j]] = p
    return out


# ---------------------------------------------------------------- characterisation

def profiles(df: pd.DataFrame, start: np.ndarray, order: np.ndarray, K: int) -> pd.DataFrame:
    """One row per state_hmm 0..K-1 (see module docstring). ``start``/``order`` from :func:`chains`."""
    df = df.copy()
    df.loc[df["acc_samples"] < SPECTRUM_SAMPLES, ACC_VARS] = np.nan
    df["full"] = df["acc_samples"] >= FULL_SAMPLES
    df["has_gps"] = df["gps_fixes"] > 0
    df["night"] = df["sun_elev_deg"] < ARTIFACT["night_sun_deg"]
    g = df.groupby("state_hmm")
    p = pd.DataFrame({"windows": g.size(), "birds": g["bird"].nunique()})
    for c in ("2024", "2025"):
        p[f"birds_{c}"] = df[df["cohort"] == c].groupby("state_hmm")["bird"].nunique()
    for v in PROFILE_VARS:
        q = g[v].quantile([0.25, 0.5, 0.75]).unstack()
        p[f"{v}_q25"], p[f"{v}_median"], p[f"{v}_q75"] = q[0.25], q[0.5], q[0.75]
    p["gps_share"] = g["has_gps"].mean()
    for c in (c for c in df if c.startswith("label_")):
        p[c.replace("label_", "frac_")] = g[c].mean()  # NaN (unknown) values are skipped
        p[c.replace("label_", "n_")] = g[c].count()
    base_sens = df["sensitivity_k"].value_counts(normalize=True)
    sens = pd.crosstab(df["state_hmm"], df["sensitivity_k"], normalize="index")
    for k in sens:
        p[f"share_sens_{k}"] = sens[k]
    p["share_full"] = g["full"].mean()
    p["share_2024"] = g["cohort"].apply(lambda s: (s == "2024").mean())
    # dwell: runs of one state_hmm within a chain
    st = df["state_hmm"].to_numpy()[order]
    new = start.copy()
    new[1:] |= st[1:] != st[:-1]
    rs = np.flatnonzero(new)
    runs = pd.DataFrame({"state": st[rs], "len": np.diff(np.r_[rs, len(st)])}).groupby("state")["len"]
    p["dwell_mean_windows"], p["dwell_median_windows"] = runs.mean(), runs.median()
    p["dwell_p90_windows"] = runs.quantile(0.9)
    p["night_share"] = g["night"].mean()
    # artifact checks
    bird_share = pd.crosstab(df["state_hmm"], df["bird"], normalize="index")
    p["top_bird"], p["max_bird_share"] = bird_share.idxmax(axis=1), bird_share.max(axis=1)
    p["max_tag_share"] = pd.crosstab(df["state_hmm"], df["bird"] + "/" + df["tag"].astype(str), normalize="index").max(axis=1)
    p["sens_ratio_max"] = (sens / base_sens[sens.columns]).max(axis=1)
    p["nonfull_ratio"] = (1 - p["share_full"]) / max(1 - df["full"].mean(), 1e-12)
    p["flight_like"] = (p["speed_mean_median"] > ARTIFACT["flight_speed_ms"]) | (p["vedba_median"] > ARTIFACT["flight_vedba_g"])
    p["flag_bird"] = p["max_bird_share"] > ARTIFACT["bird_share"]
    p["flag_tag"] = p["max_tag_share"] > ARTIFACT["tag_share"]
    p["flag_sensitivity"] = p["sens_ratio_max"] > ARTIFACT["sens_ratio"]
    p["flag_nonfull"] = p["nonfull_ratio"] > ARTIFACT["nonfull_ratio"]
    p["flag_night_flight"] = p["flight_like"] & (p["night_share"] > ARTIFACT["night_flight_share"])
    p = p.reindex(range(K))  # a state emptied by smoothing keeps its row
    p.index.name = "state"
    return p.fillna({"windows": 0, "birds": 0, "birds_2024": 0, "birds_2025": 0})


# ---------------------------------------------------------------- model use

def assign(model: Path, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """State, p_state and log p(x | state) of new embeddings h (N, 128) with a saved model.npz."""
    m = np.load(model)
    h = h.astype(np.float32)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    X = (h - m["pca_mean"]) @ m["pca_components"].T
    lj = joint(X, m["gmm_weights"], m["gmm_means"], m["gmm_var"])
    sp = posterior(lj) @ onehot(m["merge"], int(m["merge"].max()) + 1)
    return sp.argmax(1), sp.max(1), emissions(lj, m["merge"], m["gmm_weights"])


# ---------------------------------------------------------------- pipeline

def run(run_dir: Path, windows: Path, out: Path, k: int | None = None, k_over: int = 40, k_min: int = 6,
        k_max: int = 30, per_bird: int = 3000, eval_per_bird: int = 500, n_boot: int = 20, seed: int = 0,
        workers: int = 1) -> dict:
    if k is not None and not 1 <= k <= k_over:
        raise SystemExit(f"--k must be between 1 and k_over={k_over}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    rng = np.random.default_rng(seed)
    log("loading")
    meta, H = load(run_dir, windows, workers)
    birds = meta["bird"].to_numpy()
    allrows = np.ones(len(meta), bool)
    fit_idx = balanced_sample(meta, allrows, per_bird, rng)
    ev_idx = balanced_sample(meta, allrows, eval_per_bird, rng)
    log(f"{len(meta):,} windows, {meta['bird'].nunique()} birds, fit sample {len(fit_idx):,}")

    pca = PCA(n_components=0.95, svd_solver="full", random_state=seed).fit(H[fit_idx])
    X = pca.transform(H).astype(np.float32)
    del H
    ks = [K for K in range(k_min, k_max + 1) if K <= k_over]
    if k is not None and k not in ks:
        ks = sorted(ks + [k])

    log(f"PCA {pca.n_components_} components; full GMM k_over={k_over}")
    gmm = fit_gmm(X[fit_idx], k_over, seed)
    merges = ward_merge(gmm.means_, ks)
    lj = np.concatenate([joint(X[i:i + 500_000], gmm.weights_, gmm.means_, gmm.covariances_)
                         for i in range(0, len(X), 500_000)])
    post = posterior(lj)
    ref = {K: (post[ev_idx] @ onehot(merges[K], K)).argmax(1) for K in ks}

    log(f"{n_boot} bird bootstraps, held-out fit, BIC")
    ub = np.unique(birds)
    by_bird = {b: fit_idx[birds[fit_idx] == b] for b in ub}
    picks = [rng.choice(ub, len(ub), replace=True) for _ in range(n_boot)]
    hold = sorted(set(json.loads((run_dir / "holdout.json").read_text())) & set(ub))
    not_hold = fit_idx[~np.isin(birds[fit_idx], hold)]
    jobs = [delayed(refit_labels)(X[np.concatenate([by_bird[b] for b in pk])], X[ev_idx], k_over, ks, seed + 1 + i)
            for i, pk in enumerate(picks)]
    jobs.append(delayed(refit_labels)(X[not_hold], X[ev_idx], k_over, ks, seed + 1 + n_boot))
    jobs += [delayed(bic_k)(X[fit_idx], K, seed) for K in ks]
    res = Parallel(n_jobs=workers)(jobs)
    boots, held, bics = res[:n_boot], res[n_boot], res[n_boot + 1:]

    ev_hold = np.isin(birds[ev_idx], hold)
    cohort_of = meta.groupby("bird")["cohort"].first()
    per_k = {}
    for K, bic in zip(ks, bics):
        jac = np.mean([jaccard_best(ref[K], bt[K], K) for bt in boots], axis=0)
        lab = (post @ onehot(merges[K], K)).argmax(1)
        counts = pd.crosstab(birds, lab).reindex(columns=range(K), fill_value=0)
        present = counts >= PRESENCE_MIN_WINDOWS
        n_birds = present.sum(axis=0)
        cohort_ok = [set(cohort_of[present.index[present[s]]]) >= {"2024", "2025"} for s in range(K)]
        per_k[K] = {
            "jaccard": jac.round(4).tolist(), "jaccard_min": float(jac.min()),
            "heldout_ari": float(adjusted_rand_score(ref[K][ev_hold], held[K][ev_hold])) if ev_hold.any() else None,
            "presence": {"birds": n_birds.astype(int).tolist(), "cohort_ok": cohort_ok,
                         "ok": bool((n_birds >= PRESENCE_MIN_BIRDS).all() and all(cohort_ok))},
            "bic": bic,
        }
    good = [K for K in ks if per_k[K]["jaccard_min"] >= JACCARD_MIN and per_k[K]["presence"]["ok"]]
    pool = [K for K in ks if per_k[K]["presence"]["ok"]] or ks
    chosen_rule = max(good) if good else max(pool, key=lambda K: per_k[K]["jaccard_min"])
    K = k if k is not None else chosen_rule
    log(f"K by rule {chosen_rule} (rule met: {bool(good)}); using K={K}")

    # final labels; states renumbered by ascending median vedba of full-recording windows
    sp = post @ onehot(merges[K], K)
    state = sp.argmax(1)
    v = meta["vedba"].where(meta["acc_samples"] >= SPECTRUM_SAMPLES)
    rank = np.argsort(np.argsort(v.groupby(state).median().reindex(range(K)).fillna(np.inf).to_numpy(), kind="stable"))
    merge = rank[merges[K]]
    state = rank[state]
    inv = np.argsort(rank)  # old id of each new state: per-state lists of the chosen K follow the new numbering
    pk = per_k[K]
    pk["jaccard"] = [pk["jaccard"][i] for i in inv]
    pk["presence"]["birds"] = [pk["presence"]["birds"][i] for i in inv]
    pk["presence"]["cohort_ok"] = [pk["presence"]["cohort_ok"][i] for i in inv]
    p_state = sp.max(1)
    order, start = chains(meta)
    T = transitions(state[order], start, K)
    pi = np.bincount(state, minlength=K) / len(state) + 1e-12
    log_emit = emissions(lj[order], merge, gmm.weights_)
    log("HMM smoothing")
    state_hmm = np.empty_like(state)
    state_hmm[order] = smooth(log_emit, state[order], start, T, pi, workers)
    meta["state"], meta["state_hmm"] = state, state_hmm

    log("profiles and outputs")
    out.mkdir(parents=True, exist_ok=True)
    st = meta[["bird", "row", "t_start", "segment", "k_in_segment", "pitch_rot_deg", "roll_rot_deg"]].copy()
    st["state"], st["state_hmm"], st["p_state"] = state.astype(np.int16), state_hmm.astype(np.int16), p_state.astype(np.float32)
    for i in range(min(3, X.shape[1])):
        st[f"pca_{i + 1}"] = X[:, i]
    st.to_parquet(out / "states.parquet", index=False)
    prof = profiles(meta, start, order, K)
    prof["jaccard"], prof["presence_birds"] = pk["jaccard"], pk["presence"]["birds"]
    prof.to_csv(out / "state_profiles.csv")
    pd.DataFrame(T, index=pd.Index(range(K), name="from"), columns=range(K)).to_csv(out / "transitions.csv")
    np.savez(out / "model.npz", pca_mean=pca.mean_.astype(np.float32), pca_components=pca.components_.astype(np.float32),
             gmm_weights=gmm.weights_, gmm_means=gmm.means_, gmm_var=gmm.covariances_, merge=merge,
             transmat=T, startprob=pi)
    summary = {
        "chosen_k": int(K), "k_by_rule": int(chosen_rule), "k_override": k, "rule_met": bool(good),
        "rule": (f"largest K with every state's mean bootstrap Jaccard >= {JACCARD_MIN} and presence (>= "
                 f"{PRESENCE_MIN_BIRDS} birds with >= {PRESENCE_MIN_WINDOWS} windows, both cohorts); if none, the K "
                 "with the highest minimum Jaccard among K with presence (else all K)"),
        "per_k": {str(K_): v_ for K_, v_ in per_k.items()},
        "k_over": k_over, "pca_components": int(pca.n_components_),
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "windows": int(len(meta)), "birds": int(len(ub)), "fit_sample": int(len(fit_idx)),
        "eval_sample": int(len(ev_idx)), "n_boot": n_boot, "holdout_birds": hold, "seed": seed,
        "hmm_vs_gmm_agreement": float((state == state_hmm).mean()),
        "transition_self_prob": np.diag(T).round(4).tolist(), "artifact_thresholds": ARTIFACT,
    }
    (out / "state_selection.json").write_text(json.dumps(summary, indent=1))
    log("done")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=None, help="override the selected number of states")
    ap.add_argument("--k-over", type=int, default=40)
    ap.add_argument("--k-min", type=int, default=6)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--per-bird", type=int, default=3000)
    ap.add_argument("--eval-per-bird", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    s = run(a.run, a.windows, a.out, a.k, a.k_over, a.k_min, a.k_max, a.per_bird, a.eval_per_bird, a.n_boot,
            a.seed, a.workers)
    print(json.dumps({k: v for k, v in s.items() if k != "per_k"}, indent=1))


if __name__ == "__main__":
    main()
