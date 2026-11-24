"""Stage 3c: handcrafted-feature baseline for state discovery.

Gaussian mixture on transformed window features; the number of states is chosen by BIC on a bird-balanced
sample. Temporal smoothing uses the mixture as emission model and a transition matrix estimated from
consecutive windows inside continuous segments (Viterbi per segment). Stability: refit on bird-bootstrap
samples and compare assignments (adjusted Rand index) on a fixed evaluation sample.

This is the bar the SimCLR representation has to clear.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

FEATURES = ["vedba", "flap_band_frac", "pitch_deg", "roll_deg", "speed_mean", "climb_mean", "turn_abs_mean"]


def load(windows: Path) -> pd.DataFrame:
    frames = []
    for d in sorted(p for p in windows.iterdir() if p.is_dir()):
        m = pd.read_parquet(d / "meta.parquet")
        f = pd.read_parquet(d / "features.parquet")
        frames.append(pd.concat([m, f], axis=1))
    return pd.concat(frames, ignore_index=True)


def design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Transformed feature matrix and the rows usable for clustering (GPS and >= 2 s of acceleration)."""
    X = pd.DataFrame(
        {
            "log_vedba": np.log10(df["vedba"].clip(lower=1e-3)),
            "flap_band_frac": df["flap_band_frac"],
            "pitch_deg": df["pitch_deg"],
            "abs_roll_deg": df["roll_deg"].abs(),
            "log_speed": np.log10(df["speed_mean"].clip(lower=0) + 0.5),
            "climb_mean": df["climb_mean"].clip(-10, 10),
            "log_turn": np.log10(df["turn_abs_mean"].clip(lower=0) + 1),
        }
    )
    ok = X.notna().all(axis=1) & (df["n_speed"] >= 5) & (df["acc_samples"] >= 40)
    return X.to_numpy(float), ok.to_numpy()


def bird_balanced_sample(birds: np.ndarray, rows: np.ndarray, per_bird: int, rng) -> np.ndarray:
    out = []
    for b in np.unique(birds[rows]):
        idx = np.flatnonzero(rows & (birds == b))
        out.append(rng.choice(idx, min(per_bird, len(idx)), replace=False))
    return np.concatenate(out)


def transitions(post: np.ndarray, segment_key: np.ndarray) -> np.ndarray:
    """Expected transition counts between consecutive windows of the same segment, row-normalised."""
    same = segment_key[1:] == segment_key[:-1]
    T = post[:-1][same].T @ post[1:][same] + 1e-3
    return T / T.sum(axis=1, keepdims=True)


def viterbi(log_emit: np.ndarray, log_T: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    n, k = log_emit.shape
    delta = log_pi + log_emit[0]
    back = np.zeros((n, k), dtype=np.int32)
    for t in range(1, n):
        cand = delta[:, None] + log_T
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(k)] + log_emit[t]
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def run(windows: Path, out: Path, k_range=range(4, 17), per_bird: int = 3000, n_boot: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    df = load(windows)
    X, ok = design(df)
    birds = df["bird"].to_numpy()
    fit_idx = bird_balanced_sample(birds, ok, per_bird, rng)
    mu, sd = X[fit_idx].mean(axis=0), X[fit_idx].std(axis=0)
    Z = (X - mu) / sd

    bic = {}
    models = {}
    for k in k_range:
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed).fit(Z[fit_idx])
        bic[k] = float(g.bic(Z[fit_idx]))
        models[k] = g
    k_best = min(bic, key=bic.get)
    gmm = models[k_best]

    rows = np.flatnonzero(ok)
    log_emit = np.full((len(df), k_best), np.nan)
    log_emit[rows] = gmm._estimate_weighted_log_prob(Z[rows])
    post = np.exp(log_emit[rows] - logsumexp(log_emit[rows], axis=1, keepdims=True))
    df["state_gmm"] = -1
    df.loc[rows, "state_gmm"] = post.argmax(axis=1)

    seg_key = (df["bird"] + "/" + df["segment"].astype(str)).to_numpy()
    T = transitions(post, seg_key[rows])
    log_T, log_pi = np.log(T), np.log(gmm.weights_)
    df["state_hmm"] = -1
    sub = df.loc[rows, ["bird", "segment"]].copy()
    sub["i"] = np.arange(len(rows))
    for _, g in sub.groupby(["bird", "segment"], sort=False):
        ii = g["i"].to_numpy()
        df.loc[rows[ii], "state_hmm"] = viterbi(log_emit[rows[ii]], log_T, log_pi)

    # Stability: refit on bird bootstraps; compare on a fixed evaluation sample.
    eval_idx = bird_balanced_sample(birds, ok, 500, rng)
    ref = gmm.predict(Z[eval_idx])
    aris = []
    ub = np.unique(birds[ok])
    for b in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        idx = np.concatenate([rng.choice(np.flatnonzero(ok & (birds == x)), min(per_bird, int((ok & (birds == x)).sum())), replace=False) for x in pick])
        gb = GaussianMixture(k_best, covariance_type="full", n_init=3, random_state=seed + 1 + b).fit(Z[idx])
        aris.append(float(adjusted_rand_score(ref, gb.predict(Z[eval_idx]))))

    # State profiles
    lab = [c for c in df if c.startswith("label_")]
    first = df.groupby("bird")["t_start"].transform("min")
    df["cohort"] = np.where(first.dt.year <= 2024, "2024", "2025")
    prof = df[df["state_hmm"] >= 0].groupby("state_hmm").agg(
        windows=("vedba", "size"),
        birds=("bird", "nunique"),
        birds_2024=("bird", lambda s: s[df.loc[s.index, "cohort"] == "2024"].nunique()),
        birds_2025=("bird", lambda s: s[df.loc[s.index, "cohort"] == "2025"].nunique()),
        vedba=("vedba", "median"),
        flap_band_frac=("flap_band_frac", "median"),
        dom_freq_hz=("dom_freq_hz", "median"),
        pitch_deg=("pitch_deg", "median"),
        speed=("speed_mean", "median"),
        climb=("climb_mean", "median"),
        turn=("turn_abs_mean", "median"),
        sun_elev=("sun_elev_deg", "median"),
        **{c.replace("label_", "frac_"): (c, "mean") for c in lab},
    )
    out.mkdir(parents=True, exist_ok=True)
    prof.to_csv(out / "state_profiles.csv")
    df[["bird", "t_start", "segment", "k_in_segment", "state_gmm", "state_hmm"]].to_parquet(out / "states.parquet")
    summary = {
        "windows_total": int(len(df)),
        "windows_clustered": int(ok.sum()),
        "fit_sample": int(len(fit_idx)),
        "bic": bic,
        "k_best": int(k_best),
        "bootstrap_ari": aris,
        "hmm_vs_gmm_agreement": float((df.loc[rows, "state_hmm"] == df.loc[rows, "state_gmm"]).mean()),
        "transition_self_prob": np.diag(T).round(3).tolist(),
        "features": FEATURES,
        "feature_mean": mu.tolist(),
        "feature_sd": sd.tolist(),
    }
    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("windows", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-bird", type=int, default=3000)
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--boot", type=int, default=5)
    args = ap.parse_args(argv)
    s = run(args.windows, args.out, range(args.kmin, args.kmax + 1), args.per_bird, args.boot)
    print(json.dumps({k: v for k, v in s.items() if k not in ("feature_mean", "feature_sd")}, indent=1))


if __name__ == "__main__":
    main()
