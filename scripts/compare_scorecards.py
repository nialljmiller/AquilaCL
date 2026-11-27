"""Selection rule of docs/SCORECARD_spec.md (v2): compare designs (variants) against a baseline from their scorecards.

    python scripts/compare_scorecards.py OUT_DIR [OUT_DIR ...] --baseline VARIANT [--min-seeds 5] [--n-boot 1000]
        [--json FILE]

Each OUT_DIR holds scorecard.json (or subdirectories that do, as written by aquilacl.scorecard with several --run).
Variant = run name minus a trailing _s<n>; seeds are paired by <n> (a run without _s<n> is seed "0").

Per selection row (kind S in the scorecards; G4 and G8 only when they qualify) a difference is real when all hold:
1. the mean paired difference (better direction) exceeds 2 x the sd of the paired per-seed differences;
2. it exceeds the row's minimum effect;
3. the 95 % interval of a paired bootstrap over birds excludes 0: n_boot resamples of the birds scored in every
   scorecard compared (sorted ids, generator default_rng([20261127, row id])), the same resamples for both designs;
   per resample the statistic of each seed is recomputed from the stored per-bird / per-row inputs and the
   difference of the means over seeds is taken.
Rows flagged not comparable, contaminated, skipped or in error in any scorecard compared give "n/a".
Suspension: a design whose mean (model - nuisance floor) on a row is below the minimum effect ("explained by
nuisance") cannot be better on that row, but it can be worse when the other design is not suspended and the
difference passes the rule; a blocked verdict is reported as "suspended" (a tie).
Guard rails (G10): no target's mean over seeds may drop by more than 0.02 absolute; a target with a non-finite mean
or a bad flag is "n/a", and a variant with any n/a guard target cannot be overall "better".
Overall: "incomplete" when any selection row carries an error flag in either design; else "better" when >= 1 selection
row is really better, none really worse and every guard target passes; "worse" for the mirror case; otherwise "tie".
With fewer than --min-seeds paired seeds (spec: 5) verdicts are marked provisional. All scorecards must share the
spec string, mode (search / final), injection-list sha256, split and reference-cache name; otherwise nothing is
compared.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from aquilacl import scorecard as sc

BAD = ("not comparable", "contaminated", "skipped", "error")


def load_cards(dirs: list[Path]) -> list[dict]:
    cards = []
    for d in dirs:
        files = [d / "scorecard.json"] if (d / "scorecard.json").exists() else sorted(d.glob("*/scorecard.json"))
        if not files:
            raise SystemExit(f"{d}: no scorecard.json")
        for f in files:
            c = json.loads(f.read_text())
            c["_base"] = f.parent
            m = re.search(r"_s(\d+)$", c["run"])
            c["_seed"] = m.group(1) if m else "0"
            c["_variant"] = sc.variant_of(c["run"])
            cards.append(c)
    return cards


def _num(v):
    return float(v) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else np.nan


def boot_diff(cv: list[dict], cb: list[dict], row: str, sign: float, n_boot: int, target: str | None = None):
    """95 % interval of sign * (mean over seeds V - mean over seeds B) over identical bird resamples."""
    def boot_of(c):
        b = c["rows"][row].get("boot")
        return (b or {}).get(target) if target else b

    boots = [boot_of(c) for c in cv + cb]
    if any(b is None for b in boots):
        return None, 0
    birds = sorted(set.intersection(*[set(sc.boot_birds(b)) for b in boots]))
    if len(birds) < 2:
        return None, len(birds)
    rng = np.random.default_rng([sc.SEED0, sc.ROW_IDS[row]])
    idx = rng.integers(0, len(birds), (n_boot, len(birds)))
    d = np.empty(n_boot)
    for i, r in enumerate(idx):
        cnt = np.bincount(r, minlength=len(birds))
        mult = {b: float(k) for b, k in zip(birds, cnt) if k}
        v = [sc.boot_value(boot_of(c), c["_base"], mult) for c in cv]
        w = [sc.boot_value(boot_of(c), c["_base"], mult) for c in cb]
        d[i] = sign * (np.nanmean(v) - np.nanmean(w))
    d = d[np.isfinite(d)]
    return (np.percentile(d, [2.5, 97.5]).tolist() if len(d) else None), len(birds)


def _suspended(cs: list[dict], row: str, mine) -> bool:
    fl = [_num(c["rows"][row].get("floor")) for c in cs]
    if mine is None or not np.isfinite(fl).all():
        return False
    return bool(np.mean([_num(c["rows"][row]["model"]) for c in cs]) - np.mean(fl) < mine)


def row_verdict(V: dict, B: dict, row: str, n_boot: int) -> dict:
    seeds = sorted(set(V) & set(B))
    cv, cb = [V[s] for s in seeds], [B[s] for s in seeds]
    ent = cb[0]["rows"][row]
    better, mine = ent["better"], ent["min_effect"]
    sign = 1.0 if better == "up" else -1.0
    flags = [f for c in cv + cb for f in c["rows"][row]["flags"]]
    out = {"row": row, "seeds": len(seeds), "min_effect": mine,
           "error": any(f == "error" for f in flags)}
    bad = [f for f in flags if any(k in f for k in BAD)]
    if bad:
        return {**out, "verdict": "n/a", "reason": sorted(set(bad))[:3]}
    sv, sb = _suspended(cv, row, mine), _suspended(cb, row, mine)
    a = np.array([_num(c["rows"][row]["model"]) for c in cv])
    b = np.array([_num(c["rows"][row]["model"]) for c in cb])
    d = sign * (a - b)
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
    ci, nb = boot_diff(cv, cb, row, sign, n_boot)
    c1 = np.isfinite(sd) and abs(m) > 2 * sd
    c2 = mine is not None and abs(m) > mine
    c3 = ci is not None and (ci[0] > 0 or ci[1] < 0) and np.sign(ci[0]) == np.sign(m)
    verdict = ("better" if m > 0 else "worse") if (c1 and c2 and c3) else "tie"
    reason = None
    if verdict == "better" and sv:
        verdict, reason = "suspended", "variant explained by nuisance: cannot be better"
    elif verdict == "worse" and sb:
        verdict, reason = "suspended", "baseline explained by nuisance: cannot be better"
    return {**out, "variant_mean": float(np.mean(a)), "baseline_mean": float(np.mean(b)), "mean_diff_better": m,
            "paired_sd": sd, "boot_ci95": ci, "boot_birds": nb, "suspended": {"variant": sv, "baseline": sb},
            "rule": {"2sd": bool(c1), "min_effect": bool(c2), "bootstrap": bool(c3)}, "verdict": verdict,
            **({"reason": reason} if reason else {})}


def guard(V: dict, B: dict) -> dict:
    seeds = sorted(set(V) & set(B))
    out = {}
    for t in sc.G10_TARGETS:
        def vals(D):
            return [_num((D[s]["rows"].get("G10", {}).get("model") or {}).get(t)) for s in seeds]
        flags = [f for D in (V, B) for s in seeds for f in D[s]["rows"].get("G10", {}).get("flags", [])]
        va, vb = vals(V), vals(B)
        if any(any(k in f for k in BAD) for f in flags) or not (np.isfinite(va).all() and np.isfinite(vb).all()):
            out[t] = {"status": "n/a", "variant_drops": False, "baseline_drops": False}
            continue
        a, b = float(np.mean(va)), float(np.mean(vb))
        out[t] = {"status": "ok", "variant": a, "baseline": b, "diff": a - b, "variant_drops": bool(a - b < -sc.GUARD_TOL),
                  "baseline_drops": bool(b - a < -sc.GUARD_TOL)}
    return out


SAME_KEYS = ("spec", "final", "split_sha256")


def check_same(cards: list[dict]) -> None:
    """Refuse to compare scorecards made under different specs, modes, injection lists, splits or reference caches."""
    def key(c):
        return (*(c.get(k) for k in SAME_KEYS), (c.get("injections") or {}).get("sha256"),
                Path(str(c.get("reference_cache"))).name)
    keys = {key(c) for c in cards}
    if len(keys) > 1:
        raise SystemExit("scorecards differ in spec / mode / split / injection list / reference cache: "
                         f"{sorted(map(str, keys))}")


def compare(cards: list[dict], baseline: str, min_seeds: int = 5, n_boot: int = 1000) -> dict:
    check_same(cards)
    by = {}
    for c in cards:
        by.setdefault(c["_variant"], {})[c["_seed"]] = c
    if baseline not in by:
        raise SystemExit(f"baseline {baseline} not among {sorted(by)}")
    B = by[baseline]
    res = {}
    for v, V in sorted(by.items()):
        if v == baseline:
            continue
        seeds = sorted(set(V) & set(B))
        if not seeds:
            res[v] = {"overall": "n/a (no paired seeds)"}
            continue
        ref = B[seeds[0]]["rows"]
        sel = [r for r in sc.ROWS if r in ref and r in V[seeds[0]]["rows"]
               and all(B[s]["rows"][r]["kind"] == "S" and V[s]["rows"][r]["kind"] == "S" for s in seeds)]
        rows = [row_verdict(V, B, r, n_boot) for r in sel]
        g = guard(V, B) if "G10" in ref else {}
        vb = [r["row"] for r in rows if r["verdict"] == "better"]
        vw = [r["row"] for r in rows if r["verdict"] == "worse"]
        v_drop = [t for t, x in g.items() if x["variant_drops"]]
        b_drop = [t for t, x in g.items() if x["baseline_drops"]]
        g_na = [t for t, x in g.items() if x["status"] == "n/a"]
        if any(r["error"] for r in rows):
            overall = "incomplete"
        elif vb and not vw and not v_drop and not g_na and g:
            overall = "better"
        elif vw and not vb and not b_drop:
            overall = "worse"
        else:
            overall = "tie"
        res[v] = {"seeds": len(seeds), "provisional": len(seeds) < min_seeds, "rows": rows, "guard": g,
                  "really_better": vb, "really_worse": vw, "guard_fails": v_drop, "guard_na": g_na,
                  "overall": overall}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--min-seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    res = compare(load_cards(a.dirs), a.baseline, a.min_seeds, a.n_boot)
    for v, r in res.items():
        print(f"\n== {v} vs {a.baseline}: {r['overall']}" + (" (provisional: fewer than "
                                                               f"{a.min_seeds} paired seeds)" if r.get("provisional") else ""))
        for x in r.get("rows", []):
            extra = (f"diff {x['mean_diff_better']:+.4f} (sd {x['paired_sd']:.4f}, CI {x['boot_ci95']})"
                     if "mean_diff_better" in x else str(x.get("reason", "")))
            print(f"  {x['row']:5s} {x['verdict']:9s} {extra}")
        for t, gx in r.get("guard", {}).items():
            print(f"  G10 {t}: " + ("n/a" if gx["status"] == "n/a" else
                                    f"diff {gx['diff']:+.4f}{'  GUARD FAIL' if gx['variant_drops'] else ''}"))
    print("\nSuspended rows: a design explained by nuisance on a row cannot be better on it, but can be worse.")
    if a.json:
        with open(a.json, "x") as fh:  # never overwrite
            fh.write(json.dumps(sc._jsonable(res), indent=1))


if __name__ == "__main__":
    main()
