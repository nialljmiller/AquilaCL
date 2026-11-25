"""Stage 5a-2: sub-states within each coarse state, chosen by a rule fixed in advance; coarse-state confusion.

Input: a stage 5a output directory (aquilacl.states) and the run and windows it was made from.

Sub-states (per coarse state s, using the GMM label ``state`` of states.parquet)
1. Windows of s only. Bird-balanced fit and evaluation samples as in stage 5a (states.balanced_sample: up to
   ``per_bird`` / ``eval_per_bird`` windows per bird, months equally weighted within a bird), drawn from a random
   stream of their own (seed, 1, s).
2. PCA refitted within s on the L2-normalised h of the fit sample (95 % of the variance).
3. The stage 5a procedure inside s: diagonal GMM with ``k_over`` components (default 20), Ward merge of the
   component means to every k_sub in 1..max_sub; bird-bootstrap Jaccard per sub-state (``n_boot`` refits on
   birds resampled with replacement, evaluated on the fixed evaluation sample, Hennig 2007); held-out-bird ARI
   (fit without the birds of holdout.json); presence on all windows of s (>= PRESENCE_MIN_BIRDS birds with
   >= PRESENCE_MIN_WINDOWS windows of the sub-state, birds of both cohorts).
4. Rule (fixed before any result was seen; written to substate_selection.json): the largest k_sub such that every
   sub-state has mean bootstrap Jaccard >= JACCARD_MIN and presence holds; otherwise k_sub = 1 (s is not split).
   k_sub = 1 always meets it (Jaccard trivially 1). Held-out ARI is reported and not used by the rule.
5. All windows of s get the argmax sub-state of the full fit. Sub-states are numbered 0..k_sub-1 within s by
   ascending median vedba (windows with >= 40 samples), as states are in stage 5a; label "s.j", so an unsplit
   state is "s.0". No HMM smoothing is applied to sub-states.

Caveat: sub-state stability is conditional on the coarse assignment of the full stage 5a fit. The coarse
instability (bootstrap Jaccard of the coarse states) is not carried into the sub-state numbers.

Coarse diagnostics (chosen K of stage 5a): the stage 5a fit and evaluation samples and bootstrap bird draws are
rebuilt from its seed (same random sequence as states.run, sample sizes checked against state_selection.json);
X = (h - pca_mean) @ pca_components.T from model.npz; each bootstrap refit is merged to K and labels the
evaluation sample. Every bootstrap cluster is mapped to the reference state it overlaps most (many-to-one, so a
bootstrap that merges two states and splits a third is shown as it is). coarse_confusion.csv row s, column t:
mean over bootstraps of the share of reference-state-s windows that fall in bootstrap clusters mapped to t.
jaccard_recomputed is the stage 5a per-state Jaccard recomputed here (a check; EM may differ slightly from the
stage 5a run), jaccard_stage5a the stored value.

Outputs in ``--out`` (must be new or empty)
- substates.parquet: bird, row, state, substate, label; one row per window in states.parquet order.
- substate_selection.json: rule and parameters; per state per k_sub: jaccard (per sub-state; chosen k_sub in
  final numbering, other k_sub in Ward order), jaccard_min, heldout_ari (null for k_sub = 1 or when the held-out reference
  labels have one class), presence, ok;
  chosen k_sub; coarse diagnostics.
- substate_profiles.csv: one row per sub-state (label, state, substate): the stage 5a profile columns
  (states.profiles; dwell = runs of one sub-state within a chain of consecutive windows of a segment),
  dwell_median_s (dwell_median_windows x the median window step within chains, from the data), month_share_<m>,
  share_of_state, jaccard, presence_birds.
- coarse_confusion.csv: K x K matrix (see above) with jaccard_recomputed and jaccard_stage5a.
- substates.md: plain-language description of each sub-state by its feature medians.

Usage: python -m aquilacl.substates --states <stage 5a dir> --run <run> --windows <windows> --out <new dir>
       [--max-sub 6] [--n-boot 20] [--k-over 20] [--seed 0] [--workers N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from . import states
from .states import (JACCARD_MIN, PRESENCE_MIN_BIRDS, PRESENCE_MIN_WINDOWS, SPECTRUM_SAMPLES, balanced_sample,
                     fit_gmm, jaccard_best, joint, log, onehot, posterior, refit_labels, ward_merge)

RULE = (f"per coarse state: the largest k_sub in 1..max_sub such that every sub-state has mean bird-bootstrap "
        f"Jaccard >= {JACCARD_MIN} and presence (>= {PRESENCE_MIN_BIRDS} birds with >= {PRESENCE_MIN_WINDOWS} "
        "windows of the sub-state, birds of both cohorts); otherwise k_sub = 1 (not split). Held-out-bird ARI is "
        "reported, not used. Fixed before results were seen.")
CAVEAT = ("Sub-state Jaccard, ARI and presence are conditional on the coarse assignment of the full stage 5a fit; "
          "the coarse-state instability is not included in them, so they overstate end-to-end stability.")
DESCRIBE = [("vedba", "VeDBA", "g", 3), ("flap_band_frac", "flap band fraction", "", 2),
            ("dom_freq_hz", "dominant frequency", "Hz", 2), ("pitch_rot_deg", "pitch", "deg", 0),
            ("roll_rot_deg", "roll", "deg", 0), ("speed_mean", "GPS speed", "m/s", 1),
            ("climb_mean", "climb", "m/s", 2), ("turn_abs_mean", "turn", "deg/s", 1)]
FLAGS = ["flag_bird", "flag_tag", "flag_sensitivity", "flag_nonfull", "flag_night_flight"]
CLIMB_WORD = 0.5  # m/s: |median climb| above this is called climbing / descending in substates.md
REL_IQR = 0.5     # a sub-state median beyond the state median +- REL_IQR x state IQR is called higher / lower


# ---------------------------------------------------------------- helpers

def presence(birds: np.ndarray, lab: np.ndarray, K: int, cohort_of: pd.Series) -> dict:
    """Stage 5a presence: birds with >= PRESENCE_MIN_WINDOWS windows per state, and both cohorts among them."""
    counts = pd.crosstab(birds, lab).reindex(columns=range(K), fill_value=0)
    present = counts >= PRESENCE_MIN_WINDOWS
    n_birds = present.sum(axis=0)
    cohort_ok = [set(cohort_of[present.index[present[s]]]) >= {"2024", "2025"} for s in range(K)]
    return {"birds": n_birds.astype(int).tolist(), "cohort_ok": cohort_ok,
            "ok": bool((n_birds >= PRESENCE_MIN_BIRDS).all() and all(cohort_ok))}


def confusion(ref: np.ndarray, boot: np.ndarray, K: int) -> np.ndarray:
    """Share of each reference state in bootstrap clusters mapped (by largest overlap) to each reference state."""
    C = np.bincount(ref * K + boot, minlength=K * K).reshape(K, K).astype(float)
    to = C.argmax(0)  # reference state of each bootstrap cluster
    M = np.zeros((K, K))
    for c in range(K):
        M[:, to[c]] += C[:, c]
    return M / np.maximum(M.sum(1, keepdims=True), 1)


def _fit(X, k_over, seed):
    return fit_gmm(X, k_over, seed)


# ---------------------------------------------------------------- coarse diagnostics

def coarse_jobs(meta, H, coarse, sel, model, per_bird, eval_per_bird):
    """Rebuild the stage 5a samples and bootstrap draws; return refit jobs and the context to evaluate them."""
    seed, K, k_over, n_boot = sel["seed"], int(sel["chosen_k"]), int(sel["k_over"]), int(sel["n_boot"])
    rng = np.random.default_rng(seed)
    allrows = np.ones(len(meta), bool)
    fit_idx = balanced_sample(meta, allrows, per_bird, rng)
    ev_idx = balanced_sample(meta, allrows, eval_per_bird, rng)
    if len(fit_idx) != sel["fit_sample"] or len(ev_idx) != sel["eval_sample"]:
        raise SystemExit(f"stage 5a samples not reproduced (fit {len(fit_idx)} vs {sel['fit_sample']}, eval "
                         f"{len(ev_idx)} vs {sel['eval_sample']}): pass the --coarse-per-bird/--coarse-eval-per-bird "
                         "used for stage 5a")
    birds = meta["bird"].to_numpy()
    ub = np.unique(birds)
    picks = [rng.choice(ub, len(ub), replace=True) for _ in range(n_boot)]
    proj = lambda idx: ((H[idx] - model["pca_mean"]) @ model["pca_components"].T).astype(np.float32)
    Xf, Xe = proj(fit_idx), proj(ev_idx)
    pos = {b: np.flatnonzero(birds[fit_idx] == b) for b in ub}
    jobs = [delayed(refit_labels)(Xf[np.concatenate([pos[b] for b in pk])], Xe, k_over, [K], seed + 1 + i)
            for i, pk in enumerate(picks)]
    return jobs, {"K": K, "ref": coarse[ev_idx], "n_boot": n_boot, "k_over": k_over, "seed": seed}


def coarse_result(ctx, boots, sel) -> tuple[pd.DataFrame, dict]:
    K, ref = ctx["K"], ctx["ref"]
    Ms = np.stack([confusion(ref, b[K], K) for b in boots])
    jac = np.mean([jaccard_best(ref, b[K], K) for b in boots], axis=0)
    M = Ms.mean(0)
    df = pd.DataFrame(M.round(4), index=pd.Index(range(K), name="state"), columns=[str(t) for t in range(K)])
    df["jaccard_recomputed"] = jac.round(4)
    df["jaccard_stage5a"] = sel["per_k"][str(K)]["jaccard"]
    pairs = sorted(({"pair": [s, t], "share_s_to_t": round(float(M[s, t]), 4), "share_t_to_s": round(float(M[t, s]), 4),
                     "sum": round(float(M[s, t] + M[t, s]), 4)} for s in range(K) for t in range(s + 1, K)),
                   key=lambda d: -d["sum"])
    info = {"k": K, "n_boot": ctx["n_boot"], "k_over": ctx["k_over"], "seed": ctx["seed"],
            "jaccard_recomputed": jac.round(4).tolist(), "jaccard_stage5a": sel["per_k"][str(K)]["jaccard"],
            "stay_share": np.diag(M).round(4).tolist(), "pairs_by_exchange": pairs,
            "method": "bootstrap clusters mapped many-to-one to the reference state of largest overlap; "
                      "share of reference-state windows per mapped state, mean over bootstraps"}
    return df, info


# ---------------------------------------------------------------- sub-states

def sub_jobs(meta, H, coarse, s, max_sub, k_over, per_bird, eval_per_bird, n_boot, seed, hold_all):
    mask = coarse == s
    rng = np.random.default_rng([seed, 1, int(s)])
    fit = balanced_sample(meta, mask, per_bird, rng)
    ev = balanced_sample(meta, mask, eval_per_bird, rng)
    rows = np.flatnonzero(mask)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=seed).fit(H[fit])
    Xs = pca.transform(H[rows]).astype(np.float32)
    fpos, epos = np.searchsorted(rows, fit), np.searchsorted(rows, ev)
    Xf, Xe = Xs[fpos], Xs[epos]
    birds = meta["bird"].to_numpy()
    bf = birds[fit]
    ub = np.unique(bf)
    picks = [rng.choice(ub, len(ub), replace=True) for _ in range(n_boot)]
    hold = sorted(set(hold_all) & set(ub))
    ko = min(k_over, len(fit))
    ks = [k for k in range(1, max_sub + 1) if k <= ko]
    pos = {b: np.flatnonzero(bf == b) for b in ub}
    jobs = [delayed(_fit)(Xf, ko, seed)]
    jobs += [delayed(refit_labels)(Xf[np.concatenate([pos[b] for b in pk])], Xe, ko, ks, seed + 1 + i)
             for i, pk in enumerate(picks)]
    jobs.append(delayed(refit_labels)(Xf[~np.isin(bf, hold)], Xe, ko, ks, seed + 1 + n_boot))
    ctx = {"s": int(s), "rows": rows, "Xs": Xs, "epos": epos, "ks": ks, "hold": hold, "k_over": ko,
           "ev_hold": np.isin(birds[ev], hold), "n_boot": n_boot,
           "info": {"windows": int(len(rows)), "birds": int(len(np.unique(birds[rows]))), "fit_sample": int(len(fit)),
                    "eval_sample": int(len(ev)), "pca_components": int(pca.n_components_),
                    "pca_variance": float(pca.explained_variance_ratio_.sum()), "k_over": ko, "holdout_birds": hold}}
    return jobs, ctx


def sub_result(ctx, res, meta, cohort_of) -> tuple[np.ndarray, dict]:
    """Sub-state of every window of the state (rows order) and its selection record."""
    g, boots, held = res[0], res[1:1 + ctx["n_boot"]], res[1 + ctx["n_boot"]]
    ks, Xs, rows = ctx["ks"], ctx["Xs"], ctx["rows"]
    merges = ward_merge(g.means_, ks)
    post = np.concatenate([posterior(joint(Xs[i:i + 500_000], g.weights_, g.means_, g.covariances_))
                           for i in range(0, len(Xs), 500_000)])
    ref = {k: (post[ctx["epos"]] @ onehot(merges[k], k)).argmax(1) for k in ks}
    birds = meta["bird"].to_numpy()[rows]
    per_k, labs = {}, {}
    for k in ks:
        jac = np.mean([jaccard_best(ref[k], b[k], k) for b in boots], axis=0)
        labs[k] = (post @ onehot(merges[k], k)).argmax(1)
        eh = ctx["ev_hold"]
        ok_ari = k > 1 and len(np.unique(ref[k][eh])) >= 2  # ARI of a single-class reference is degenerate
        ari = float(adjusted_rand_score(ref[k][eh], held[k][eh])) if ok_ari else None
        pres = presence(birds, labs[k], k, cohort_of)
        per_k[k] = {"jaccard": jac.round(4).tolist(), "jaccard_min": float(jac.min()), "heldout_ari": ari,
                    "presence": pres, "ok": bool(jac.min() >= JACCARD_MIN and pres["ok"])}
    chosen = max(k for k in ks if per_k[k]["ok"]) if any(per_k[k]["ok"] for k in ks) else 1
    lab = labs[chosen]
    v = meta["vedba"].where(meta["acc_samples"] >= SPECTRUM_SAMPLES).to_numpy()[rows]
    med = pd.Series(v).groupby(lab).median().reindex(range(chosen)).fillna(np.inf).to_numpy()
    rank = np.argsort(np.argsort(med, kind="stable"))
    inv = np.argsort(rank)
    pk = per_k[chosen]
    pk["jaccard"] = [pk["jaccard"][i] for i in inv]
    for c in ("birds", "cohort_ok"):
        pk["presence"][c] = [pk["presence"][c][i] for i in inv]
    rec = dict(ctx["info"], chosen_k_sub=int(chosen), per_k_sub={str(k): per_k[k] for k in ks})
    return rank[lab], rec


# ---------------------------------------------------------------- characterisation

def window_step_s(meta, order, start) -> float:
    t = meta["t_start"].to_numpy()[order]
    d = (np.diff(t)[~start[1:]] / np.timedelta64(1, "s")).astype(float)
    return float(np.median(d)) if len(d) else float("nan")


def build_profiles(meta, gid, n, order, start, step_s) -> pd.DataFrame:
    df = meta.assign(state_hmm=gid)
    p = states.profiles(df, start, order, n)
    p["dwell_median_s"] = p["dwell_median_windows"] * step_s
    ms = pd.crosstab(gid, meta["month"], normalize="index").reindex(index=range(n), columns=range(1, 13), fill_value=0)
    for m in range(1, 13):
        p[f"month_share_{m:02d}"] = ms[m].to_numpy()
    return p


def _fmt(x, nd):
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def describe(row, base) -> str:
    """One paragraph: medians, then differences from the whole coarse state."""
    parts = [f"{_fmt(row[f'{v}_median'], nd)}{(' ' + u) if u else ''} {name}" for v, name, u, nd in DESCRIBE]
    months = row[[f"month_share_{m:02d}" for m in range(1, 13)]].sort_values(ascending=False)
    top = ", ".join(f"{int(k[-2:])}: {v:.0%}" for k, v in months.head(3).items() if v > 0)
    txt = (f"{row['share_of_state']:.1%} of the state's windows, {int(row['birds'])} birds. Medians: "
           + "; ".join(parts) + f". Night share {row['night_share']:.1%}. Most frequent months {top}.")
    rel = []
    for v, name, _, _ in DESCRIBE:
        iqr = base[f"{v}_q75"] - base[f"{v}_q25"]
        d = row[f"{v}_median"] - base[f"{v}_median"]
        if pd.notna(d) and pd.notna(iqr) and iqr > 0 and abs(d) > REL_IQR * iqr:
            rel.append(f"{'higher' if d > 0 else 'lower'} {name}")
    if pd.notna(base["night_share"]) and abs(row["night_share"] - base["night_share"]) > 0.1:
        rel.append(f"{'more' if row['night_share'] > base['night_share'] else 'less'} night")
    c = row["climb_mean_median"]
    if pd.notna(c) and abs(c) > CLIMB_WORD:
        rel.append("climbing" if c > 0 else "descending")
    flags = [f for f in FLAGS if bool(row.get(f, False)) and str(row.get(f)) != "nan"]
    txt += (f" GPS present in {row['gps_share']:.0%} of windows (speed, climb and turn come from these only); "
            f"artifact flags raised: {', '.join(flags) if flags else 'none'}.")
    txt += " Compared with the whole state: " + (", ".join(rel) if rel else "no median differs by more than "
                                                 f"{REL_IQR} x the state's interquartile range") + "."
    return txt


def write_md(path, sel_out, prof, coarse_prof, cinfo) -> None:
    L = ["# Sub-states", "", f"Rule (fixed in advance): {RULE}", "", f"Caveat: {CAVEAT}", "",
         f"Differences are called higher / lower when the sub-state median lies more than {REL_IQR} x the state's "
         f"interquartile range from the state median; climbing / descending when |median climb| > {CLIMB_WORD} m/s. "
         "No behaviour names are given; all numbers are medians of the windows of the sub-state "
         "(acceleration variables over windows with >= 40 samples, GPS variables over windows with GPS).", ""]
    for s, rec in sel_out["states"].items():
        s = int(s)
        L += [f"## State {s}: k_sub = {rec['chosen_k_sub']}", "", "| k_sub | min Jaccard | held-out ARI | presence | meets rule |",
              "|---|---|---|---|---|"]
        for k, r in rec["per_k_sub"].items():
            ari = "-" if r["heldout_ari"] is None else f"{r['heldout_ari']:.3f}"
            L.append(f"| {k} | {r['jaccard_min']:.3f} | {ari} | {'yes' if r['presence']['ok'] else 'no'} "
                     f"| {'yes' if r['ok'] else 'no'} |")
        base = coarse_prof.loc[s]
        L += ["", "Whole state: " + "; ".join(f"{_fmt(base[f'{v}_median'], nd)}{(' ' + u) if u else ''} {name}"
                                              for v, name, u, nd in DESCRIBE) + f"; night share {base['night_share']:.1%}.", ""]
        if rec["chosen_k_sub"] == 1:
            L += ["Not split: no k_sub >= 2 meets the rule.", ""]
            continue
        for _, row in prof[prof["state"] == s].iterrows():
            L += [f"- **{row['label']}** (mean Jaccard {row['jaccard']:.3f}): " + describe(row, base)]
        L.append("")
    K = cinfo["k"]
    L += ["## Coarse-state exchange across bootstraps", "",
          f"K = {K}, {cinfo['n_boot']} bird bootstraps. Share of a state's evaluation windows that stay in it: "
          + ", ".join(f"{s}: {v:.2f}" for s, v in enumerate(cinfo["stay_share"])) + ".",
          "Recomputed Jaccard: " + ", ".join(f"{s}: {v:.3f}" for s, v in enumerate(cinfo["jaccard_recomputed"]))
          + " (stage 5a: " + ", ".join(f"{v:.3f}" for v in cinfo["jaccard_stage5a"]) + ").", "",
          "Pairs with the largest exchange (share s to t, t to s):", ""]
    L += [f"- {p['pair'][0]} and {p['pair'][1]}: {p['share_s_to_t']:.3f}, {p['share_t_to_s']:.3f}"
          for p in cinfo["pairs_by_exchange"][:5]]
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- pipeline

def run(states_dir: Path, run_dir: Path, windows: Path, out: Path, max_sub: int = 6, k_over: int = 20,
        per_bird: int = 3000, eval_per_bird: int = 500, n_boot: int = 20, seed: int = 0, workers: int = 1,
        coarse_per_bird: int = 3000, coarse_eval_per_bird: int = 500) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; give a new directory")
    if max_sub < 1:
        raise SystemExit("--max-sub must be >= 1")
    sel = json.loads((states_dir / "state_selection.json").read_text())
    st = pd.read_parquet(states_dir / "states.parquet", columns=["bird", "row", "state"])
    model = np.load(states_dir / "model.npz")
    log("loading")
    meta, H = states.load(run_dir, windows, workers)
    j = meta[["bird", "row"]].merge(st, on=["bird", "row"], how="left")
    if len(st) != len(meta) or j["state"].isna().any():
        raise SystemExit("states.parquet does not cover the windows of --windows one-to-one")
    coarse = j["state"].to_numpy().astype(int)
    K = int(sel["chosen_k"])
    cohort_of = meta.groupby("bird")["cohort"].first()
    hold_all = json.loads((run_dir / "holdout.json").read_text())

    cjobs, cctx = coarse_jobs(meta, H, coarse, sel, model, coarse_per_bird, coarse_eval_per_bird)
    sjobs, sctx = [], []
    for s in range(K):
        jb, cx = sub_jobs(meta, H, coarse, s, max_sub, k_over, per_bird, eval_per_bird, n_boot, seed, hold_all)
        sjobs.append(jb)
        sctx.append(cx)
    log(f"{len(meta):,} windows; {len(cjobs)} coarse refits and {sum(map(len, sjobs))} sub-state fits")
    res = Parallel(n_jobs=workers)(cjobs + [j_ for jb in sjobs for j_ in jb])
    del H
    cdf, cinfo = coarse_result(cctx, res[:len(cjobs)], sel)

    i = len(cjobs)
    sub = np.zeros(len(meta), int)
    recs = {}
    for cx, jb in zip(sctx, sjobs):
        lab, recs[str(cx["s"])] = sub_result(cx, res[i:i + len(jb)], meta, cohort_of)
        sub[cx["rows"]] = lab
        i += len(jb)
    log("profiles and outputs")
    nsub = np.array([recs[str(s)]["chosen_k_sub"] for s in range(K)])
    offs = np.r_[0, np.cumsum(nsub)[:-1]]
    gid = offs[coarse] + sub
    n = int(nsub.sum())
    order, start = states.chains(meta)
    step = window_step_s(meta, order, start)
    prof = build_profiles(meta, gid, n, order, start, step)
    g_state = np.repeat(np.arange(K), nsub)
    g_sub = np.concatenate([np.arange(k) for k in nsub])
    prof.insert(0, "label", [f"{a}.{b}" for a, b in zip(g_state, g_sub)])
    prof.insert(1, "state", g_state)
    prof.insert(2, "substate", g_sub)
    state_n = np.bincount(coarse, minlength=K)
    prof.insert(4, "share_of_state", prof["windows"].to_numpy() / state_n[g_state])
    prof["jaccard"] = [recs[str(a)]["per_k_sub"][str(nsub[a])]["jaccard"][b] for a, b in zip(g_state, g_sub)]
    prof["presence_birds"] = [recs[str(a)]["per_k_sub"][str(nsub[a])]["presence"]["birds"][b] for a, b in zip(g_state, g_sub)]
    prof = prof.reset_index(drop=True)
    coarse_prof = build_profiles(meta, coarse, K, order, start, step)

    out.mkdir(parents=True, exist_ok=True)
    lab_df = meta[["bird", "row"]].assign(state=coarse.astype(np.int16), substate=sub.astype(np.int16),
                                          label=[f"{a}.{b}" for a, b in zip(coarse, sub)])
    st[["bird", "row"]].merge(lab_df, on=["bird", "row"], how="left").to_parquet(out / "substates.parquet", index=False)
    prof.to_csv(out / "substate_profiles.csv", index=False)
    cdf.to_csv(out / "coarse_confusion.csv")
    summary = {
        "rule": RULE, "caveat": CAVEAT,
        "parameters": {"jaccard_min": JACCARD_MIN, "presence_min_birds": PRESENCE_MIN_BIRDS,
                       "presence_min_windows": PRESENCE_MIN_WINDOWS, "presence_cohorts": ["2024", "2025"],
                       "max_sub": max_sub, "k_over": k_over, "per_bird": per_bird, "eval_per_bird": eval_per_bird,
                       "n_boot": n_boot, "seed": seed, "fallback": "k_sub = 1 (state not split)",
                       "heldout_ari_in_rule": False, "k_sub_1": "Jaccard trivially 1, held-out ARI undefined (null); ARI also null when held-out reference labels have one class",
                       "numbering": "sub-states 0..k_sub-1 by ascending median vedba within the state; unsplit = s.0",
                       "pca": "refitted within each state on its fit sample, 95 % variance"},
        "states_dir": str(states_dir), "coarse_k": K, "windows": int(len(meta)), "window_step_s": step,
        "chosen": {s: r["chosen_k_sub"] for s, r in recs.items()}, "states": recs, "coarse": cinfo,
    }
    (out / "substate_selection.json").write_text(json.dumps(summary, indent=1, default=float))
    write_md(out / "substates.md", summary, prof, coarse_prof, cinfo)
    log("done")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", type=Path, required=True, help="stage 5a output directory")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-sub", type=int, default=6)
    ap.add_argument("--k-over", type=int, default=20, help="GMM components within a state before Ward merging")
    ap.add_argument("--per-bird", type=int, default=3000)
    ap.add_argument("--eval-per-bird", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--coarse-per-bird", type=int, default=3000, help="per_bird used for stage 5a")
    ap.add_argument("--coarse-eval-per-bird", type=int, default=500, help="eval_per_bird used for stage 5a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    s = run(a.states, a.run, a.windows, a.out, a.max_sub, a.k_over, a.per_bird, a.eval_per_bird, a.n_boot, a.seed,
            a.workers, a.coarse_per_bird, a.coarse_eval_per_bird)
    print(json.dumps({"chosen": s["chosen"], "coarse_pairs": s["coarse"]["pairs_by_exchange"][:5]}, indent=1))


if __name__ == "__main__":
    main()
