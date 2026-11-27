"""Scorecard bird split (docs/SCORECARD_spec.md): locked test birds, search birds, training birds.

Eligible for locking: birds with windows, minus every bird listed in --exclude files (earlier held-out sets that
informed decisions). Eligible birds are sorted by stratum (sex, deployment year, young-tagged, migrant flag), then
by sha256(salt + bird) within the stratum; every k-th bird (k = eligible / n_locked, offset floor(k / 2)) is locked.
Search birds = the --search file (the r7 holdout). Training = all other birds with windows.

    python scripts/make_split.py --windows WINDOWS --reference REF.csv --search runs/r7_base_s0/holdout.json \
        --exclude runs/r6_yaw60_bpb4/holdout.json runs/r7_base_s0/holdout.json --out DIR

Writes DIR/split.json ({train, search, locked, method}) and DIR/split_summary.json (counts per stratum, no ids).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from aquilacl import reference

SALT = "aquilacl_locked_2026-11"


def stratum_table(birds: list[str], ref: pd.DataFrame) -> pd.DataFrame:
    r = ref.set_index("bird").reindex(birds)
    return pd.DataFrame({
        "bird": birds,
        "sex": r["sex"].fillna("unknown").to_numpy(),
        "deploy_year": r["deploy_on"].dt.year.fillna(0).astype(int).to_numpy(),
        "young": r["tagged_young"].fillna(False).astype(bool).to_numpy(),
        "migrant": r["migrant"].fillna(False).astype(bool).to_numpy(),
    })


def systematic_lock(T: pd.DataFrame, n_locked: int, salt: str = SALT) -> list[str]:
    """Every k-th bird of the stratum-ordered list (proportional allocation; strata of any size)."""
    T = T.assign(h=[hashlib.sha256((salt + b).encode()).hexdigest() for b in T["bird"]])
    T = T.sort_values(["sex", "deploy_year", "young", "migrant", "h"]).reset_index(drop=True)
    k = len(T) / n_locked
    idx = sorted({int(k // 2 + i * k) for i in range(n_locked)})
    return sorted(T["bird"].iloc[idx])


def make_split(birds: list[str], ref: pd.DataFrame, search: list[str], exclude: set[str], n_locked: int = 20) -> dict:
    T = stratum_table(birds, ref)
    elig = T[~T["bird"].isin(exclude | set(search))]
    if len(elig) < n_locked:
        raise SystemExit(f"only {len(elig)} eligible birds for {n_locked} locked")
    locked = systematic_lock(elig, n_locked)
    train = sorted(set(birds) - set(locked) - set(search))
    return {"train": train, "search": sorted(search), "locked": locked,
            "method": f"systematic stratified (sex, deploy_year, young, migrant) by sha256('{SALT}' + bird); "
                      f"eligible {len(elig)} of {len(birds)} birds"}


def summary(split: dict, T: pd.DataFrame) -> dict:
    out = {}
    for name in ("train", "search", "locked"):
        t = T[T["bird"].isin(split[name])]
        out[name] = {"birds": int(len(t)), "sex": t["sex"].value_counts().to_dict(),
                     "deploy_year": {str(k): int(v) for k, v in t["deploy_year"].value_counts().items()},
                     "young": int(t["young"].sum()), "migrant_flag": int(t["migrant"].sum())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--search", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--n-locked", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if (a.out / "split.json").exists():
        raise SystemExit(f"{a.out / 'split.json'} exists; the split is drawn once")
    birds = sorted(p.name for p in a.windows.iterdir() if (p / "meta.parquet").exists())
    ref = reference.load_birds(a.reference, birds)
    search = json.loads(a.search.read_text())
    exclude = set().union(*(json.loads(p.read_text()) for p in a.exclude)) if a.exclude else set()
    split = make_split(birds, ref, search, exclude, a.n_locked)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "split.json").write_text(json.dumps(split, indent=1))
    s = summary(split, stratum_table(birds, ref))
    (a.out / "split_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
