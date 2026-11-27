"""Model-free counts the scorecard spec requires before it is frozen (docs/SCORECARD_spec.md v2), and the frozen G7
injection list.

    python scripts/scorecard_precount.py --windows W --split SPLIT --reference REF --states2 STATES_DIR \
        --residency RES_DIR --out NEW_DIR [--workers N] [--n-perm 200]
    python scripts/scorecard_precount.py --freeze NEW_DIR/injections_candidate.parquet --split SPLIT

Mode 1 writes NEW_DIR/precount.json (runtime output; per-bird counts are keyed by bird id and never printed) and the
drawn G7 list as NEW_DIR/injections_candidate.parquet, and prints its counts (per set and kind: hosts listed, hosts
scorable, injections, donors, birds). Mode 2 (--freeze) copies a candidate to <split dir>/injections.parquet with an
exclusive create (refused if it exists); aquilacl.scorecard refuses to run until that file exists. Search mode
only: locked birds are not loaded except for drawing their (model-free) injection hosts. CPU only.
- G6: GPS-only label positives on the scorecard's own test rows (total, per search bird, birds with positives),
  the same over every domain window of the search birds, search birds with continuous-mode windows and with domain
  windows, and the classes that qualify (>= 30 test positives from >= 3 search birds).
- G8: nuisance-floor AUC (deployment year, continuous share, GPS share, track days, deploy month, young-tagged,
  winter-window count; permutation within deployment year) and the resulting kind (S if < 0.9).
- G4 / G5: qualifying birds; G3: birds with >= 9 calendar months of qualifying bird-days.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquilacl import scorecard as sc


def g6_counts(ctx: sc.Ctx) -> dict:
    g = sc.g6_rows(ctx)
    F = ctx.F
    bte = F["bird"].to_numpy()[g["te"]]
    k1 = F["sensitivity_k"].to_numpy() == 1
    dom = k1 & (F["acc_samples"].to_numpy() >= sc.FULL) & (F["gps_fixes"].to_numpy() >= sc.G6_MIN_FIXES)
    test = ctx.is_in(ctx.test)
    rows_all = np.flatnonzero(dom & test)
    L_all, ok_all = sc.gps_label_rows(ctx.windows, F, rows_all)
    b_all = F["bird"].to_numpy()[rows_all]
    labels = {}
    for j, lab in enumerate(sc.GPS_LABELS):
        labels[lab] = {**g["counts"][lab],
                       "test_positives_by_bird": pd.Series(bte[g["Lte"][:, j]]).value_counts().astype(int).to_dict(),
                       "all_domain_positives": int(L_all[:, j].sum()),
                       "all_domain_positives_by_bird": pd.Series(b_all[L_all[:, j]]).value_counts().astype(int).to_dict(),
                       "qualifies": lab in g["qualifying"]}
    return {"labels": labels, "qualifying": g["qualifying"], "test_windows": int(len(g["te"])),
            "train_windows": int(len(g["tr"])), "all_domain_windows_search": int(ok_all.sum()),
            "search_birds": len(ctx.test),
            "search_birds_with_continuous_windows": int(pd.unique(F["bird"].to_numpy()[k1 & test]).size),
            "search_birds_with_domain_windows": int(pd.unique(F["bird"].to_numpy()[dom & test]).size),
            "rule": f">= {sc.G6_MIN_POS} test positives from >= {sc.G6_MIN_BIRDS} search birds"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("windows", "split", "reference", "states2", "residency", "out"):
        ap.add_argument(f"--{a}", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-perm", type=int, default=sc.N_PERM)
    ap.add_argument("--freeze", type=Path, default=None, help="candidate parquet to freeze as <split dir>/injections.parquet")
    a = ap.parse_args(argv)
    if a.freeze is not None:
        if a.split is None:
            raise SystemExit("--freeze needs --split")
        dst = a.split.parent / "injections.parquet"
        try:
            with open(dst, "xb") as fh:
                fh.write(a.freeze.read_bytes())
        except FileExistsError:
            raise SystemExit(f"{dst} exists: the injection list is frozen and is never overwritten")
        print(json.dumps(sc.load_injections(dst)[1], indent=1))
        return
    missing = [k for k in ("windows", "split", "reference", "states2", "residency", "out") if getattr(a, k) is None]
    if missing:
        raise SystemExit(f"missing {missing}")
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty; give a new directory")
    ctx = sc.build_context(a.windows, a.split, a.reference, a.states2, a.residency, False, a.workers, "cpu", a.n_perm)
    out = {"spec": sc.SPEC, "birds": ctx.info, "row_ids": sc.ROW_IDS}
    sc.log("G6 counts")
    out["G6"] = g6_counts(ctx)
    sc.log("G8 floor")
    fl = sc.floor_G8(ctx, "G8")
    out["G8"] = {"floor": sc._jsonable(sc._strip(fl)), "kind": "S" if fl.get("score", 1.0) < sc.G8_FLOOR_MAX else "D",
                 "rule": f"S if the floor AUC < {sc.G8_FLOOR_MAX}"}
    bm = sc.bird_months(ctx)
    q4 = sc.g4_qualifying(bm["T"], ctx.nonlocked)
    out["G4"] = {"birds_qualifying": len(q4), "kind": "S" if len(q4) >= sc.G4_MIN_BIRDS else "D",
                 "bird_months": int(len(bm["T"]))}
    T5 = sc.bird_months(ctx, by="age")["T"]
    q5 = [b for b in sorted(set(T5["bird"])) if set(T5.loc[(T5["bird"] == b) & (T5["year"] == 1), "month"])
          & set(T5.loc[(T5["bird"] == b) & (T5["year"] == 2), "month"])]
    out["G5"] = {"birds_qualifying": len(q5), "young_birds_with_months": int(T5["bird"].nunique())}
    T3 = sc.bird_days(ctx)["T"]
    m3 = T3.groupby("bird")["month"].nunique()
    out["G3"] = {"bird_days": int(len(T3)), "birds": int(T3["bird"].nunique()),
                 "birds_with_9_calendar_months": int((m3 >= sc.G3_MONTHS).sum())}
    sc.log("injections")
    inj, stats = sc.make_injections(a.windows, ctx.split, a.workers)
    a.out.mkdir(parents=True, exist_ok=True)
    sc.write_injections(a.out / "injections_candidate.parquet", inj)
    out["injections_candidate"] = {"path": str(a.out / "injections_candidate.parquet"), "counts": stats}
    (a.out / "precount.json").write_text(json.dumps(sc._jsonable(out), indent=1))
    summary = {"G6_qualifying": out["G6"]["qualifying"], "G6_test_windows": out["G6"]["test_windows"],
               "G6_search_birds_with_continuous_windows": out["G6"]["search_birds_with_continuous_windows"],
               "G8_floor_auc": out["G8"]["floor"].get("score"), "G8_kind": out["G8"]["kind"],
               "G4_birds": out["G4"]["birds_qualifying"], "G5_birds": out["G5"]["birds_qualifying"],
               "G3_birds_9_months": out["G3"]["birds_with_9_calendar_months"], "injections_candidate": stats}
    print(json.dumps(sc._jsonable(summary), indent=1))  # counts only, no bird ids


if __name__ == "__main__":
    main()
