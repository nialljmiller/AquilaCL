"""Linear probes with whole birds held out (GroupKFold by bird).

A probe asks how linearly recoverable a quantity is from a representation. Behaviour quantities should
be recoverable; device quantities (sensor fitted or not, decoding version, sampling) should not.

Run as a module to probe the first-draft embeddings at hour level::

    python -m aquilacl.probes old --embeddings logs/main/00030000_embeddings.csv --windows <windows dir> \
        --events <data>/events --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, n_splits: int = 5, seed: int = 0) -> dict:
    """Held-out-bird score: balanced accuracy (classification) or R^2 (regression).

    Also returns the per-bird scores, so the spread across birds is visible, and the chance level.
    """
    ok = np.isfinite(X).all(axis=1) & (pd.notna(y))
    X, y, groups = X[ok], np.asarray(y)[ok], np.asarray(groups)[ok]
    if kind == "class":
        classes = pd.unique(y)
        if len(classes) < 2:
            return {"n": int(len(y)), "skipped": "one class"}
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        return {"n": int(len(y)), "skipped": "fewer than 3 birds"}
    pred = np.empty(len(y), dtype=object if kind == "class" else float)
    for tr, te in GroupKFold(n_splits=min(n_splits, n_groups)).split(X, y, groups):
        if kind == "class":
            if len(pd.unique(y[tr])) < 2:
                pred[te] = y[tr][0]
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        else:
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    res: dict = {"n": int(len(y)), "birds": int(n_groups)}
    per_bird = []
    for b in np.unique(groups):
        s = groups == b
        if kind == "class":
            if len(pd.unique(y[s])) >= 2:
                per_bird.append(balanced_accuracy_score(y[s], pred[s].astype(y.dtype)))
        elif s.sum() > 2 and np.var(y[s]) > 0:
            per_bird.append(r2_score(y[s], pred[s]))
    if kind == "class":
        res["balanced_accuracy"] = float(balanced_accuracy_score(y, pred.astype(y.dtype)))
        res["chance"] = float(1 / len(pd.unique(y)))
    else:
        res["r2"] = float(r2_score(y, pred))
    if per_bird:
        res["per_bird_median"] = float(np.median(per_bird))
        res["per_bird_min"] = float(np.min(per_bird))
        res["per_bird_n"] = len(per_bird)
    return res


def hour_targets(windows: Path) -> pd.DataFrame:
    """Aggregate 10 s window features and labels to (bird, UTC hour)."""
    frames = []
    for d in sorted(p for p in windows.iterdir() if p.is_dir()):
        m = pd.read_parquet(d / "meta.parquet", columns=["bird", "t_start", "decoding", "gps_fixes", "sun_elev_deg"])
        f = pd.read_parquet(d / "features.parquet")
        x = pd.concat([m, f], axis=1)
        x["hour"] = x["t_start"].dt.floor("h")
        lab = [c for c in x if c.startswith("label_")]
        agg = x.groupby(["bird", "hour"]).agg(
            n_windows=("vedba", "size"),
            vedba=("vedba", "mean"),
            speed=("speed_mean", "mean"),
            climb=("climb_mean", "mean"),
            turn=("turn_abs_mean", "mean"),
            sun_elev=("sun_elev_deg", "mean"),
            gps_windows=("gps_fixes", lambda s: float((s > 0).mean())),
            decoding=("decoding", lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan),
            **{c.replace("label_", "frac_"): (c, "mean") for c in lab},
        )
        frames.append(agg.reset_index())
    return pd.concat(frames, ignore_index=True)


HOUR_BEHAVIOUR = ("vedba", "speed", "climb", "turn", "frac_flapping", "frac_perched", "frac_circling",
                  "frac_gliding", "frac_climbing_soar", "frac_night", "sun_elev")


def hour_suite(df: pd.DataFrame, X: np.ndarray, mag_birds: set) -> dict:
    """The hour-level probe suite used for both the first-draft and the new embeddings."""
    df = df.copy()
    first = df.groupby("bird")["hour"].transform("min")
    df["cohort"] = np.where(first.dt.year <= 2024, "2024", "2025")
    df["has_magnetometer"] = df["bird"].isin(list(mag_birds))
    df["any_gps"] = df["gps_windows"] > 0
    groups = df["bird"].to_numpy()
    r = {}
    c24 = (df["cohort"] == "2024").to_numpy()
    r["has_magnetometer_within_2024"] = probe(X[c24], df.loc[c24, "has_magnetometer"].to_numpy(), groups[c24], "class")
    r["decoding_version"] = probe(X, df["decoding"].astype(str).to_numpy(), groups, "class")
    r["any_gps_in_hour"] = probe(X, df["any_gps"].to_numpy(), groups, "class")
    for t in HOUR_BEHAVIOUR:
        r[t] = probe(X, df[t].to_numpy(dtype=float), groups, "reg")
    r["cohort (descriptive)"] = probe(X, df["cohort"].to_numpy(), groups, "class")
    return r


def old_embeddings(args) -> None:
    emb = pd.read_csv(args.embeddings)
    emb["hour"] = pd.to_datetime(emb["window_start"], utc=True).dt.tz_localize(None)
    emb["bird"] = emb["bird_id"].astype(str)
    tgt = hour_targets(args.windows)
    df = emb.merge(tgt, on=["bird", "hour"], how="inner")
    mag_birds = {p.stem for p in (args.events / "magnetometer").glob("*.parquet")}
    results = {"rows_matched": int(len(df)), "rows_embedding": int(len(emb)), "birds": int(df["bird"].nunique())}
    for space in ("z", "h"):
        X = df[[c for c in df if c.startswith(f"{space}_")]].to_numpy(dtype=float)
        results[space] = hour_suite(df, X, mag_birds)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probes_old_step30k.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("old", help="probe the first-draft hourly embeddings")
    o.add_argument("--embeddings", type=Path, required=True)
    o.add_argument("--windows", type=Path, required=True)
    o.add_argument("--events", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "old":
        old_embeddings(args)


if __name__ == "__main__":
    main()
