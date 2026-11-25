"""Compare evaluation metrics of several runs, grouped into variants over seeds, against a baseline variant.

Each run directory holds evaluation.json and/or evaluation_<section>.json (aquilacl.evaluate). The files are
merged per run: evaluation.json first, then the section files in name order; a key present in both takes the
section file's value (a later --only rerun). Every numeric leaf is one metric, named by its path ("a.b.c"); lists
of two numbers (95 % intervals) become ".lo"/".hi". Bird-keyed entries (*recall_by_bird*, holdout_birds) are left
out, so the tables carry no bird ids.

Variant and seed come from the run directory name: "<variant>_s<seed>" (e.g. r7_full_s0, r7_full_s1 -> variant
r7_full, seeds 0 and 1); a name without that suffix is a single-run variant. Per variant and metric: mean and sd
over the runs that have the metric (sd with ddof 1; none for one run). Difference to the baseline variant:
- paired: when both variants have the metric for >= 2 common seeds, the mean and sd of the per-seed differences;
- unpaired otherwise: difference of the means, with the pooled sd sqrt((sd_v^2 + sd_b^2) / 2) when both have
  >= 2 runs, else no sd.
Direction: lower is better for nuisance and identity metrics (keys containing "(nuisance)", "bird_id_holdout",
"sensitivity_full_windows"); counts, chance levels, p values, null values and sds have no direction; every other
metric is higher-is-better (probe R2 / balanced accuracy / AUC, same-state rates, ARI). A metric is flagged
WORSE when the variant is worse than the baseline by more than 2 of the sds above; without an sd it is not
flagged (the difference is still shown). Descriptive: no test is made here.

Usage: python scripts/compare_runs.py RUN_DIR [RUN_DIR ...] --baseline VARIANT --out PREFIX
Writes PREFIX.md and PREFIX.csv (refuses to overwrite either).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED_RE = re.compile(r"^(?P<variant>.+)_s(?P<seed>\d+)$")
EXCLUDE = ("recall_by_bird", "holdout_birds")
LOWER = ("(nuisance)", "bird_id_holdout", "sensitivity_full_windows")
NEUTRAL_NAMES = {"n", "k", "n_test", "chance", "resamples", "birds", "hours", "flight_state", "bird_id_chance"}
NEUTRAL_PREFIX = ("n_", "p_", "null_", "birds_", "windows_", "window_n_", "min_", "pre_fledge", "state_rows")
NEUTRAL_SUFFIX = ("_n", "_sd", "_birds", "_sd_over_repeats", "_chance", "_resamples")


def flatten(d, prefix="") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if any(e in str(k) for e in EXCLUDE):
            continue
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, bool) or v is None or isinstance(v, str):
            continue
        elif isinstance(v, (int, float)):
            if np.isfinite(v):
                out[key] = float(v)
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            out[key + ".lo"], out[key + ".hi"] = float(v[0]), float(v[1])
    return out


def load_metrics(run: Path) -> dict[str, float]:
    files = ([run / "evaluation.json"] if (run / "evaluation.json").exists() else []) + sorted(run.glob("evaluation_*.json"))
    out = {}
    for f in files:
        out.update(flatten(json.loads(f.read_text())))
    return out


def direction(metric: str) -> str:
    name = metric.split(".")[-1]
    if name in {"lo", "hi"}:
        name = metric.split(".")[-2]
    if name in NEUTRAL_NAMES or name.startswith(NEUTRAL_PREFIX) or name.endswith(NEUTRAL_SUFFIX):
        return "none"
    return "lower" if any(x in metric for x in LOWER) else "higher"


def variant_of(run: Path) -> tuple[str, str | None]:
    m = SEED_RE.match(run.name)
    return (m["variant"], m["seed"]) if m else (run.name, None)


def _sd(x) -> float:
    return float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan


def compare(runs: list[Path], baseline: str) -> pd.DataFrame:
    """Long table: one row per metric and variant (see module docstring)."""
    recs = []
    seen = set()
    for r in runs:
        v, s = variant_of(r)
        if (v, s) in seen:
            raise SystemExit(f"two runs for variant {v} seed {s}")
        seen.add((v, s))
        for m, x in load_metrics(r).items():
            recs.append({"variant": v, "seed": s if s is not None else r.name, "paired_seed": s, "metric": m, "value": x})
    L = pd.DataFrame(recs, columns=["variant", "seed", "paired_seed", "metric", "value"])
    variants = list(dict.fromkeys(variant_of(r)[0] for r in runs))
    if baseline not in variants:
        raise SystemExit(f"baseline variant {baseline!r} not among {variants}")
    rows = []
    for metric, g in L.groupby("metric", sort=True):
        d = direction(metric)
        base = g[g.variant == baseline]
        for v in variants:
            gv = g[g.variant == v]
            row = {"metric": metric, "direction": d, "variant": v, "n_runs": len(gv),
                   "mean": gv.value.mean() if len(gv) else np.nan, "sd": _sd(gv.value.to_numpy())}
            if v != baseline and len(gv) and len(base):
                a = gv.dropna(subset=["paired_seed"]).set_index("paired_seed").value
                b = base.dropna(subset=["paired_seed"]).set_index("paired_seed").value
                common = a.index.intersection(b.index)
                if len(common) >= 2:
                    diff = (a[common] - b[common]).to_numpy()
                    row.update(diff=diff.mean(), diff_sd=_sd(diff), n_pairs=len(common), sd_rule="paired")
                else:
                    sv, sb = _sd(gv.value.to_numpy()), _sd(base.value.to_numpy())
                    row.update(diff=row["mean"] - base.value.mean(), diff_sd=np.sqrt((sv ** 2 + sb ** 2) / 2),
                               n_pairs=len(common), sd_rule="pooled" if np.isfinite(sv) and np.isfinite(sb) else "none")
                worse = {"higher": -row["diff"], "lower": row["diff"]}.get(d, np.nan)
                row["worse_by"] = worse
                row["worse_2sd"] = bool(np.isfinite(worse) and np.isfinite(row["diff_sd"])
                                        and worse > 2 * row["diff_sd"] + 1e-12)
            rows.append(row)
    cols = ["metric", "direction", "variant", "n_runs", "mean", "sd", "diff", "diff_sd", "n_pairs", "sd_rule",
            "worse_by", "worse_2sd"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _f(x) -> str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4g}"


def markdown(T: pd.DataFrame, runs: list[Path], baseline: str) -> str:
    variants = list(dict.fromkeys(T.variant))
    by_v = {v: [r.name for r in runs if variant_of(r)[0] == v] for v in variants}
    lines = ["# Run comparison", "",
             f"Baseline variant: **{baseline}**. Values: mean ± sd over seeds (n runs); Δ = variant − baseline "
             "(paired over common seeds when >= 2 pair, else difference of means). **WORSE** = worse than the "
             "baseline by more than 2 sd of the difference (paired sd, else pooled seed sd; no flag without an sd). "
             "Direction: ↑ higher is better, ↓ lower is better, · none. Descriptive, no test.", "", "Runs per variant:", ""]
    lines += [f"- {v}: {', '.join(by_v[v])}" for v in variants]
    flagged = T[T.worse_2sd.map(lambda x: pd.notna(x) and bool(x))]
    lines += ["", f"## Flagged ({len(flagged)})", ""]
    if len(flagged):
        lines += ["| metric | variant | Δ | sd of Δ | rule |", "|---|---|---|---|---|"]
        lines += [f"| {r.metric} | {r.variant} | {_f(r.diff)} | {_f(r.diff_sd)} | {r.sd_rule} |" for r in flagged.itertuples()]
    else:
        lines.append("None.")
    lines += ["", "## All metrics", "", "| metric | dir | " + " | ".join(variants) + " |",
              "|---|---|" + "---|" * len(variants)]
    arrow = {"higher": "↑", "lower": "↓", "none": "·"}
    for metric, g in T.groupby("metric", sort=True):
        g = g.set_index("variant")
        cells = []
        for v in variants:
            r = g.loc[v]
            if r.n_runs == 0:
                cells.append("—")
                continue
            c = f"{_f(r['mean'])}" + (f" ± {_f(r.sd)}" if np.isfinite(r.sd) else "") + f" ({int(r.n_runs)})"
            if v != baseline and pd.notna(r["diff"]):
                c += f"; Δ {r['diff']:+.4g}"
                if pd.notna(r.worse_2sd) and bool(r.worse_2sd):
                    c += " **WORSE**"
            cells.append(c)
        lines.append(f"| {metric} | {arrow[g['direction'].iat[0]]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--baseline", required=True, help="baseline variant name (run name without _s<seed>)")
    ap.add_argument("--out", required=True, type=Path, help="output prefix: writes PREFIX.md and PREFIX.csv")
    args = ap.parse_args(argv)
    md, csv = args.out.with_name(args.out.name + ".md"), args.out.with_name(args.out.name + ".csv")
    for p in (md, csv):
        if p.exists():
            raise SystemExit(f"{p} exists; give a new --out prefix")
    missing = [str(r) for r in args.runs if not any(r.glob("evaluation*.json"))]
    if missing:
        print(f"warning: no evaluation*.json in {len(missing)} run dir(s): {', '.join(missing)}", file=sys.stderr)
    T = compare(args.runs, args.baseline)
    md.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(csv, index=False)
    md.write_text(markdown(T, args.runs, args.baseline))
    print(f"{T.metric.nunique()} metrics, {T.variant.nunique()} variants; flagged {int(T.worse_2sd.map(lambda x: pd.notna(x) and bool(x)).sum())}; "
          f"wrote {md} and {csv}")
    return T


if __name__ == "__main__":
    main()
