"""Stage 5b: analysis suite and figures for the behaviour states of stage 5a.

Answers every question asked by the first-draft plots (docs/analysis_inventory.md, modules M1-M8) with the states
discovered on the SimCLR embedding, in time, state and feature space only. Latitude and longitude are never read.

    python -m aquilacl.analysis --states <5a dir> --windows <windows dir> --run <run dir> \
        --reference <Movebank reference-data CSV> --out <new dir>

Writes PNG figures, CSV tables and index.md (every output with the question it answers, the first-draft plot it
replaces, and the notes of this run). Comparisons between birds are made on per-bird values (each bird weighs the
same); temporal curves are bird-weighted and carry a coverage panel.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FS = 20.0  # Hz, acceleration sampling rate
WINDOW_S = 10.0  # s per window
FULL_SAMPLES = 190  # a window with >= 190 of 200 samples is "full" (same cut as evaluate.py)
META_COLS = ["t_start", "segment", "k_in_segment", "sensitivity_k", "sun_elev_deg", "acc_samples", "gps_fixes",
             "local_solar_hour"]
FEATURES = {  # column: axis label with unit
    "vedba": "VeDBA (g)", "odba": "ODBA (g)", "flap_band_frac": "flap-band power fraction",
    "dom_freq_hz": "dominant frequency (Hz)", "pitch_rot_deg": "pitch, gravity-aligned frame (deg)",
    "roll_rot_deg": "roll, gravity-aligned frame (deg)",
    "speed_mean": "ground speed (m/s)", "climb_mean": "climb rate (m/s)", "turn_abs_mean": "|turn rate| (deg/s)",
}
# Exact local solar hour or sun elevation together with the time would give the position (nest areas), so the
# window-level outputs carry only coarse bins: whole hours and 5-degree sun-elevation steps.
CATALOGUE_COLS = ["bird", "t_start", "state_gmm", "state_hmm", "p_state", "vedba", "flap_band_frac", "pitch_rot_deg",
                  "speed_mean", "climb_mean", "turn_abs_mean", "solar_hour_bin", "sun_elev_bin", "sensitivity_k",
                  "acc_samples", "gps_fixes"]
MIN_BIN_WINDOWS = 10  # a bird enters a time bin's bird-weighted share with at least this many windows
MIN_BIRD_WINDOWS = 100  # a bird enters per-bird comparisons with at least this many windows
STABLE_JACCARD = 0.75  # Hennig 2007
REF_FIELDS = {  # reference-data field: accepted column names (first present wins). Only these columns are read.
    "id": ["individual-local-identifier", "animal-id"],
    "sex": ["animal-sex", "sex"],
    "life_stage": ["animal-life-stage", "life-stage"],
    "deploy_on": ["deploy-on-date", "deploy-on-timestamp"],
}
SEX_COLOURS = {"m": "tab:blue", "f": "tab:red"}
MISSING = {"", "u", "unknown", "nan", "none", "na", "n/a"}

# First-draft outputs (docs/analysis_inventory.md §1) -> the outputs of this stage that answer the same question.
FIRST_DRAFT = [
    ("loss_curve.png, training_curves.png", "did training converge?", "m1_loss.png, m1_cosine.png"),
    ("{step}_embeddings.png, global_pca.png, global_umap.png, {step}_{PCA,tSNE,UMAP}_*.png",
     "latent structure and what organises it", "m2_embedding_map.png, m1_probes.png, m1_probes.csv"),
    ("cluster_umap.png", "number and size of states", "m3_selection.png, m3_jaccard.png, m2_embedding_map.png"),
    ("cluster_profiles_{movement,activity,temporal}.png", "meaning of states",
     "m4_state_profile.png, m4_state_features.png, m4_state_cards_*.png, m4_state_profile.csv"),
    ("cluster_heatmap.png", "is a state a bird (or tag) identity?", "m6_bird_budgets.png, m4_state_format.png"),
    ("bird_timelines.png", "state sequences in time", "m6_calendar.png, m5_transitions.png, m5_dwell.png"),
    ("bird_profiles.png", "per-bird state budget", "m6_bird_budgets.png, m6_bird_budgets.csv"),
    ("pop_diversity.png, age_diversity.png", "how variable are birds?", "m6_bird_diversity.png, m6_bird_summary.csv"),
    ("pop_sex.png, sex_umap.png", "do sexes separate in latent space?", "m2_embedding_map.png (sex panel)"),
    ("anomaly.png", "unusual windows", "m7_outliers.png, m7_outliers.csv, m7_rare_states.png, m7_outlier_factors.csv"),
    ("temporal_doy.png, behaviour_seasonal.png", "seasonal change",
     "m5_seasonal.png, m5_seasonal_by_mode.png, m5_seasonal_by_mode.csv"),
    ("temporal_diel.png, behaviour_diel.png", "diel rhythm", "m5_diel_solar_hour.png, m5_diel_sun_elevation.png"),
    ("per_bird/bird_<id>.png", "single-bird deep dive", "m6_calendar.png, m6_bird_budgets.png (one row per bird)"),
    ("sex_features.png, sex_cluster_usage.png", "do sexes differ in state use?", "m6_group_sex.png, m6_group_tests.csv"),
    ("sex_seasonal.png, sex_diel.png", "sex x season, sex x diel", "m6_sex_diel.png, m6_sex_seasonal.png"),
    ("behaviour_umap.png, behaviour_cluster_matrix.png, behaviour_feature_validation.png",
     "states vs behaviour labels", "m4_state_labels.png, m4_state_label_confusion.csv"),
    ("behaviour_per_bird.png, behaviour_timelines.png", "behaviour budget per bird and in time",
     "m6_bird_budgets.png, m6_calendar.png"),
    ("age_umap.png, age_features.png, age_clusters.png, age_heatmap.png", "do life stages differ?",
     "m6_group_life_stage.png, m6_group_tests.csv (only where the reference data give a life stage)"),
    ("age_trajectory.png, age_seasonal.png, age_diel.png", "do young birds converge on adults?",
     "m6_development.png (only where life stage and deployment date are given)"),
    ("age_features_window.png", "window features by age", "m6_group_life_stage.png (bird level)"),
    ("window_catalogue_{h}h.csv, bird_catalogue_enhanced.csv", "per-window and per-bird tables",
     "m8_window_catalogue.csv.gz, m6_bird_summary.csv"),
    ("gp_features*.csv", "GP hyperparameters per window", "not covered: unused by any first-draft analysis and "
     "invalid (inventory §2); no GP features exist in this pipeline"),
]
# With age from aquilacl.reference, the age rows of FIRST_DRAFT point to these outputs instead.
AGE_FIRST_DRAFT = {
    "age_umap.png, age_features.png, age_clusters.png, age_heatmap.png":
        "m6_group_tagged_young.png, m6_group_tests.csv (descriptive: tagged young vs age not stated)",
    "age_trajectory.png, age_seasonal.png, age_diel.png":
        "m6_development.png, m6_development.csv, m6_age_y1_vs_y2.png, m6_age_y1_vs_y2.csv, "
        "m6_age_y1_vs_y2_summary.csv",
    "age_features_window.png": "m6_group_tagged_young.png (bird level)",
}
DAYS_PER_MONTH = 30.44


class Outputs:
    """Registry of written outputs; index.md is generated from it, so it lists only files that exist."""

    def __init__(self, root: Path):
        self.root, self.items, self.notes = root, [], []

    def fig(self, fig, name: str, module: str, question: str) -> None:
        fig.savefig(self.root / name, dpi=110, bbox_inches="tight")
        plt.close(fig)
        self.items.append((name, module, question))

    def table(self, df: pd.DataFrame, name: str, module: str, question: str, **kw) -> None:
        df.to_csv(self.root / name, **kw)
        self.items.append((name, module, question))

    def note(self, text: str) -> None:
        self.notes.append(text)
        print("note:", text)

    def answered(self, cell: str) -> str:
        """Mark each output named in a FIRST_DRAFT cell that this run did not write."""
        if cell.startswith("not covered"):
            return cell
        names = [n for n, _, _ in self.items]
        parts = []
        for tok in cell.split(", "):
            name = tok.split(" (")[0]
            ok = any(fnmatch.fnmatch(n, name) for n in names)
            parts.append(tok if ok else f"{tok} [not produced in this run, see notes]")
        return ", ".join(parts)

    def write_index(self, header: list[str], first_draft: list | None = None) -> None:
        lines = ["# aquilacl stage 5b analysis", "", *header, "", "## Outputs", "",
                 "| File | Module | Question |", "|---|---|---|"]
        lines += [f"| [{n}]({n}) | {m} | {q} |" for n, m, q in self.items]
        lines += ["", "## First-draft plots and where their questions are answered", "",
                  "| First-draft output | Question | Answered by |", "|---|---|---|"]
        lines += [f"| {a} | {q} | {self.answered(b)} |" for a, q, b in (first_draft or FIRST_DRAFT)]
        lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        (self.root / "index.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- loading


def bird_dirs(windows: Path) -> list[Path]:
    """Bird directories with at least one window (birds with 0 windows have empty files)."""
    return [d for d in sorted(p for p in windows.iterdir() if p.is_dir())
            if (d / "meta.parquet").exists() and pq.ParquetFile(d / "meta.parquet").metadata.num_rows > 0]


def _read(path: Path, want: list[str]) -> pd.DataFrame:
    have = set(pq.read_schema(path).names)
    df = pd.read_parquet(path, columns=[c for c in want if c in have])
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def circular_mean_hour(h: pd.Series) -> float:
    """Mean of a clock time (h), so that 23 h and 1 h average to 0 h, not 12 h."""
    a = np.deg2rad(h.dropna().to_numpy(float) * 15.0)
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 15.0 % 24) if len(a) else np.nan


def load_windows(states_dir: Path, windows: Path, out: Outputs, state_col: str = "state_hmm") -> pd.DataFrame:
    """One row per window with state, metadata, features and reference labels (no positions).

    ``state`` is the analysis state (``state_col`` of states.parquet); the unsmoothed GMM state is ``state_gmm``.
    """
    sp = states_dir / "states.parquet"
    st = _read(sp, ["bird", "row", "state", "state_hmm", "p_state", "pitch_rot_deg", "roll_rot_deg"])
    st["bird"] = st["bird"].astype(str)
    by_bird = dict(tuple(st.groupby("bird", sort=False)))
    frames = []
    for d in bird_dirs(windows):
        m = _read(d / "meta.parquet", META_COLS)
        fcols = [c for c in FEATURES if not c.endswith("_rot_deg")] + [c for c in pq.read_schema(d / "features.parquet").names if c.startswith("label_")]
        m = pd.concat([m, _read(d / "features.parquet", fcols)], axis=1)
        s = by_bird.pop(d.name, None)
        if s is None:
            out.note(f"bird {d.name}: {len(m)} windows but no rows in states.parquet; left out")
            continue
        if s["row"].max() >= len(m) or s["row"].min() < 0:
            raise ValueError(f"{d.name}: states.parquet row {s['row'].max()} outside 0..{len(m) - 1}")
        m["row"] = np.arange(len(m))
        m["n_meta"] = len(m)
        m["bird"] = d.name
        frames.append(m.merge(s.drop(columns="bird"), on="row", how="inner", validate="1:1"))
        if len(frames[-1]) < len(m):
            out.note(f"bird {d.name}: {len(m) - len(frames[-1])} of {len(m)} windows have no row in states.parquet")
    if by_bird:
        raise ValueError(f"states.parquet has birds without windows: {sorted(by_bird)[:5]}")
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df = df.rename(columns={"state": "state_gmm"})
    df["state"] = df["state_gmm" if state_col == "state" else state_col]
    df = df[df["state"].notna()].reset_index(drop=True)
    if len(df) < n0:
        out.note(f"{n0 - len(df)} windows without a state were left out")
    df["bird"] = df["bird"].astype("category")
    df["t_start"] = pd.to_datetime(df["t_start"], utc=True)
    df["year"] = df["t_start"].dt.year
    df["doy"] = df["t_start"].dt.dayofyear
    df["month"] = df["t_start"].dt.strftime("%Y-%m")
    df["regime"] = pd.Categorical(np.where(df["acc_samples"] >= FULL_SAMPLES, "full", "non-full"))
    df["solar_hour_bin"] = np.floor(df["local_solar_hour"])  # coarse bins only in window-level outputs
    df["sun_elev_bin"] = np.floor(df["sun_elev_deg"] / 5) * 5
    for c in ("state", "state_gmm", "state_hmm"):
        if c in df:
            df[c] = df[c].astype(int)
    return df


def load_reference(path: Path, birds: list[str], out: Outputs, age_source: bool = False
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-bird sex, life stage and deployment date from the reference CSV, never invented.

    Only the columns in REF_FIELDS are read. Several deployments of one animal: a field with conflicting values is
    reported as "conflict" and treated as missing. Returns (per-bird table, coverage table).
    """
    head = pd.read_csv(path, nrows=0).columns
    cols = {f: next((c for c in names if c in head), None) for f, names in REF_FIELDS.items()}
    if cols["id"] is None:
        raise ValueError(f"reference CSV has none of the id columns {REF_FIELDS['id']}")
    ref = pd.read_csv(path, usecols=[c for c in cols.values() if c], dtype=str, keep_default_na=False)
    ref = ref.rename(columns={c: f for f, c in cols.items() if c})
    # Movebank exports may write the animal as "<ring id>; <tag id>": the bird id is the part before ";"
    ref["id"] = ref["id"].str.split(";").str[0].str.strip()
    meta = pd.DataFrame({"bird": birds})
    rows = []
    for f in ("sex", "life_stage", "deploy_on"):
        if cols[f] is None:
            meta[f] = np.nan
            rows.append({"field": f, "column": "(absent)", "birds": len(birds), "matched": 0, "with_value": 0,
                         "missing": len(birds), "conflict": 0})
            continue
        v = ref[["id", f]].copy()
        v[f] = v[f].str.strip()
        if f != "deploy_on":
            v[f] = v[f].str.lower()
        v.loc[v[f].str.lower().isin(MISSING), f] = None
        if f == "sex":
            v[f] = v[f].replace({"male": "m", "female": "f"})
        v = v.dropna()
        if f == "deploy_on":  # earliest deployment, not a conflict
            vals = pd.to_datetime(v[f], errors="coerce", format="mixed").groupby(v["id"]).min()
        else:
            vals = v.groupby("id")[f].agg(lambda s: s.iat[0] if s.nunique() == 1 else "conflict")
        meta[f] = meta["bird"].map(vals)
        matched = meta["bird"].isin(set(ref["id"]))
        conflict = int((meta[f] == "conflict").sum())
        meta.loc[meta[f] == "conflict", f] = np.nan
        rows.append({"field": f, "column": cols[f], "birds": len(birds), "matched": int(matched.sum()),
                     "with_value": int(meta[f].notna().sum()), "missing": int(meta[f].isna().sum()) - conflict,
                     "conflict": conflict})
    cov = pd.DataFrame(rows)
    unmatched = [b for b in birds if b not in set(ref["id"])]
    if unmatched:
        out.note(f"{len(unmatched)} of {len(birds)} birds have no row in the reference CSV (matched on "
                 f"'{cols['id']}'); their metadata are reported as missing")
    if cols["life_stage"] is None and not age_source:
        out.note("the reference CSV has no life-stage column; life stage is not inferred from comments, dates or "
                 "movement, so no life-stage comparison is made")
    return meta, cov


AGE_FIELDS = ["bird", "sex", "tagged_young", "hatch_year", "deploy_on", "migrant"]  # the reference columns used here


def load_age(path: Path, birds: list[str], out: Outputs):
    """Per-bird age fields from ``aquilacl.reference`` (load_birds / age_on), or (None, None) when that module
    cannot be imported (then age is not used, as before it existed). Only AGE_FIELDS are kept; free-text fields
    such as ``young_term`` and the tag columns are never carried into outputs."""
    try:
        reference = importlib.import_module(".reference", __package__ or "aquilacl")
    except ImportError as e:
        out.note(f"aquilacl.reference could not be imported ({type(e).__name__}); age is not used and the "
                 "life-stage outputs fall back to the reference CSV's life-stage column")
        return None, None
    ref = reference.load_birds(path, birds)
    ref = ref.assign(bird=ref["bird"].astype(str)).drop_duplicates("bird").set_index("bird", drop=False)
    ref = ref[[c for c in AGE_FIELDS if c in ref]].reindex(birds)
    ref["bird"] = ref.index
    return reference, ref


def flag_label(v, yes: str, no: str) -> str:
    """True / False / missing -> yes / no / "not stated" (NaN is never read as True)."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "not stated"
    if isinstance(v, (bool, np.bool_, int, np.integer, float)) and v in (True, 1):
        return yes
    if isinstance(v, (bool, np.bool_, int, np.integer, float)) and v in (False, 0):
        return no
    return "not stated"


def call_age_on(reference, row: pd.Series, ts: pd.Series, fledge=None) -> tuple[np.ndarray, np.ndarray]:
    """reference.age_on as (age_days, age_class); accepts a returned tuple or a DataFrame with those columns."""
    r = reference.age_on(row, ts) if fledge is None else reference.age_on(row, ts, fledge=fledge)
    d, c = (r["age_days"], r["age_class"]) if isinstance(r, pd.DataFrame) else r
    return np.asarray(d, float), np.asarray(c, object)


def window_age(df: pd.DataFrame, ref: pd.DataFrame, reference) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Age in days and age class of every window, from reference.age_on per bird (tz-aware UTC timestamps).

    When the module has ``fledge_from_speed``, each tagged-young bird's fledge date comes from it (on the bird's
    window times and speed_mean) and is passed to age_on. Returns (age_days, age_class, per-bird fledge table)."""
    age = np.full(len(df), np.nan)
    cls = np.full(len(df), "not_stated", dtype=object)
    fl = []
    use_speed = hasattr(reference, "fledge_from_speed") and "speed_mean" in df
    for b, ix in df.groupby("bird", observed=True).indices.items():
        if str(b) not in ref.index:
            continue
        row = ref.loc[str(b)]
        ts = df["t_start"].iloc[ix].reset_index(drop=True)
        fledge = None
        if use_speed and flag_label(row.get("tagged_young"), "y", "n") == "y" and pd.notna(row.get("hatch_year")):
            r = reference.fledge_from_speed(ts, df["speed_mean"].iloc[ix].to_numpy(float), row["hatch_year"])
            fledge = r["fledge"]
            fl.append({"bird": str(b), "fledge": fledge, "fledge_source": r.get("fledge_source"),
                       "fledge_is_first_data_day": r.get("fledge_is_first_data_day")})
        age[ix], cls[ix] = call_age_on(reference, row, ts, fledge)
    return age, cls, pd.DataFrame(fl, columns=["bird", "fledge", "fledge_source", "fledge_is_first_data_day"])


def fledge_age_days(reference, row: pd.Series, first_window: pd.Timestamp) -> float:
    """Age (days) at which reference.age_on changes from pre_fledge to juvenile_y1 for this bird.

    age_on is evaluated daily over two years from 1 January of the hatch year (or from 400 days before the
    bird's first window when no hatch year is given). NaN when no pre_fledge -> juvenile_y1 change is seen."""
    hy = row.get("hatch_year")
    start = (pd.Timestamp(year=int(hy), month=1, day=1, tz="UTC") if pd.notna(hy)
             else (first_window - pd.Timedelta(days=400)).floor("D"))
    d, c = call_age_on(reference, row, pd.Series(pd.date_range(start, periods=731, freq="D")))
    juv = (c == "juvenile_y1") & np.isfinite(d)
    if not (c == "pre_fledge").any() or not juv.any():
        return np.nan
    return float(d[juv].min())


def load_h(run: Path, picks: pd.DataFrame) -> np.ndarray:
    """Embeddings h for the (bird, row) pairs in ``picks`` (row = position in the bird's meta)."""
    parts = []
    for b, g in picks.groupby("bird", observed=True, sort=False):
        h = np.load(run / "embeddings" / f"{b}.npz")["h"]
        parts.append((g.index.to_numpy(), h[g["row"].to_numpy()].astype(np.float32)))
    out = np.zeros((len(picks), parts[0][1].shape[1]), np.float32)
    pos = {ix: i for i, ix in enumerate(picks.index)}
    for ix, h in parts:
        out[[pos[i] for i in ix]] = h
    return out


def check_embeddings(run: Path, df: pd.DataFrame) -> None:
    """h rows are aligned to meta by position (as in evaluate.load_run); the counts must match."""
    n = df.groupby("bird", observed=True)["n_meta"].first()
    for b, nb in n.items():
        with np.load(run / "embeddings" / f"{b}.npz") as e:
            if len(e["h"]) != nb:
                raise ValueError(f"{b}: {len(e['h'])} embeddings but {nb} windows")


def load_traces(windows: Path, picks: pd.DataFrame) -> list[np.ndarray]:
    """(200, 3) acceleration in g for each picked window, NaN where no sample exists."""
    res = {}
    for b, g in picks.groupby("bird", observed=True, sort=False):
        a = np.load(windows / str(b) / "acc.npy", mmap_mode="r")
        m = np.load(windows / str(b) / "acc_mask.npy", mmap_mode="r")
        for ix, r in zip(g.index, g["row"]):
            x = np.asarray(a[r], np.float32).copy()
            x[~np.asarray(m[r])] = np.nan
            res[ix] = x
    return [res[i] for i in picks.index]


def load_selection(states_dir: Path) -> tuple[pd.DataFrame, int | None, pd.Series | None]:
    """Adapter for stage 5a state_selection.json (the only place its schema is assumed).

    5a writes {"chosen_k": K, "rule_met": bool, "per_k": {"<K>": {"jaccard": [per state], "jaccard_min",
    "heldout_ari", "presence": {"birds": [...], "cohort_ok": [...], "ok"}, "bic"}}}; the chosen K's Jaccard list
    follows the final state numbering of the unsmoothed GMM ``state``. Also accepted: "per_k" as a list of records
    with "k", "jaccard" as a {state: value} mapping. Returns (one row per K with scalar metrics, chosen K, Jaccard
    per state).
    """
    sel = json.loads((states_dir / "state_selection.json").read_text())
    chosen = next((sel[k] for k in ("chosen_k", "k_chosen", "k") if k in sel), None)
    per = sel.get("per_k", {})
    recs = per if isinstance(per, list) else [{"k": k, **v} for k, v in per.items()]
    rows, jac = [], None
    for r in recs:
        row = {"k": int(r["k"]), **{k: v for k, v in r.items()
                                    if k != "k" and isinstance(v, (int, float)) and not isinstance(v, bool)}}
        pres = r.get("presence")
        if isinstance(pres, dict):  # 5a: {"birds": [per state], "cohort_ok": [...], "ok": bool}
            row["presence_ok"] = float(bool(pres.get("ok")))
            if pres.get("birds"):
                row["presence_min_birds"] = float(min(pres["birds"]))
        rows.append(row)
        if chosen is not None and int(r["k"]) == int(chosen):
            jac = next((v for k, v in r.items() if "jaccard" in k and isinstance(v, (dict, list))), None)
    jac = sel.get("jaccard", jac)
    tab = pd.DataFrame(rows).astype(float).sort_values("k") if rows else pd.DataFrame()
    if isinstance(jac, list):
        jac = dict(enumerate(jac))
    jac = pd.Series({int(k): float(v) for k, v in jac.items()}).sort_index() if jac else None
    return tab, (int(chosen) if chosen is not None else None), jac


def read_loss(path: Path) -> pd.DataFrame:
    """loss.csv is appended across resumes: drop repeated headers, keep the last row per step."""
    df = pd.read_csv(path, dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["step"])
    return df.drop_duplicates("step", keep="last").sort_values("step")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = v
    return out


# ---------------------------------------------------------------- helpers


def state_colours(states) -> dict:
    states = sorted(states)
    cmap = plt.get_cmap("tab10" if len(states) <= 10 else "tab20" if len(states) <= 20 else "turbo")
    cols = cmap.colors if hasattr(cmap, "colors") else cmap(np.linspace(0, 1, len(states)))
    return {s: cols[i % len(cols)] for i, s in enumerate(states)}


def bird_share(df: pd.DataFrame, bin_col: str, states: list, col: str = "state") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Bird-weighted state share per bin: per-bird shares averaged over birds with >= MIN_BIN_WINDOWS windows."""
    c = df.groupby([bin_col, "bird", col], observed=True).size().unstack(col, fill_value=0)
    c = c.reindex(columns=states, fill_value=0)
    tot = c.sum(axis=1)
    sh = c.div(tot, axis=0)[tot >= MIN_BIN_WINDOWS]
    return sh.groupby(level=0).mean(), sh.groupby(level=0).size(), df.groupby(bin_col, observed=True).size()


def per_bird_budget(df: pd.DataFrame, states: list) -> pd.DataFrame:
    c = df.groupby(["bird", "state"], observed=True).size().unstack("state", fill_value=0).reindex(columns=states, fill_value=0)
    return c.div(c.sum(axis=1), axis=0).assign(windows=c.sum(axis=1))


def plot_trace(ax, a: np.ndarray, title: str, small: bool = True) -> None:
    t = np.arange(len(a)) / FS
    for i, (lab, c) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"))):
        ax.plot(t, a[:, i], lw=0.6, color=c, label=lab)
    ax.plot(t, np.linalg.norm(a, axis=1), lw=1.0, color="k", label="|a|")
    ax.set_xlim(0, len(a) / FS)  # non-full windows keep the full time axis
    ax.set_title(title, fontsize=7 if small else 9)
    ax.tick_params(labelsize=6)


def share_grid(df, bin_col, width, group_col, xlabel, title, states, colours, xlim=None):
    """Bird-weighted stacked state share per bin (top) with coverage (bottom), one column per group."""
    groups = sorted(df[group_col].dropna().unique())
    fig, ax = plt.subplots(2, max(len(groups), 1), figsize=(4.2 * max(len(groups), 1), 5.5), squeeze=False,
                           sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for j, g in enumerate(groups):
        sub = df[df[group_col] == g]
        share, nb, nw = bird_share(sub, bin_col, states)
        bottom = np.zeros(len(share))
        for s in states:
            ax[0, j].bar(share.index, share[s], width=width, bottom=bottom, color=colours[s], align="edge",
                         label=f"state {s}", lw=0)
            bottom += share[s].to_numpy()
        ax[0, j].set_title(f"{g} ({sub['bird'].nunique()} birds)", fontsize=9)
        ax[0, j].set_ylim(0, 1)
        ax[1, j].bar(nw.index, nw.to_numpy(), width=width, align="edge", color="0.6", lw=0)
        tw = ax[1, j].twinx()
        tw.plot(nb.index + width / 2, nb.to_numpy(), "k.-", ms=3, lw=0.8)
        tw.set_ylim(0, max(nb.max() if len(nb) else 1, 1) * 1.1)
        ax[1, j].set_xlabel(xlabel)
        if j == len(groups) - 1:
            tw.set_ylabel("birds in share (line)")
        if xlim:
            ax[1, j].set_xlim(*xlim)
    ax[0, 0].set_ylabel("bird-weighted state share")
    ax[1, 0].set_ylabel("windows (bars)")
    ax[0, -1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(title, fontsize=10)
    return fig


# ---------------------------------------------------------------- M1 training diagnostics


def fig_loss(loss: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(loss["step"], loss["loss"], lw=0.4, color="0.6", label="per logged step")
    w = max(len(loss) // 100, 1)
    ax.plot(loss["step"], loss["loss"].rolling(w, min_periods=1).mean(), color="k", label=f"rolling mean ({w})")
    ax.set_xlabel("training step")
    ax.set_ylabel("NT-Xent loss")
    if "lr" in loss:
        tw = ax.twinx()
        tw.plot(loss["step"], loss["lr"], color="tab:orange", lw=0.8)
        tw.set_ylabel("learning rate", color="tab:orange")
    ax.legend(fontsize=7)
    return fig


def fig_cosine(loss: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for c, col in (("pos_cos", "tab:green"), ("neg_cos", "tab:red")):
        if c in loss:
            ax.plot(loss["step"], loss[c], lw=0.6, color=col, label=c.replace("_", " ") + "ine similarity")
    ax.set_xlabel("training step")
    ax.set_ylabel("mean cosine similarity")
    ax.legend(fontsize=7)
    return fig


def fig_probes(probes: pd.DataFrame):
    p = probes[probes["section"] == "window_level_holdout"]
    p = p[p["metric"].isin(["r2", "balanced_accuracy"])]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(p) + 1.2))
    ax.barh(p["target"] + " (" + p["metric"] + ")", p["value"], color="tab:blue")
    ax.axvline(0.5, color="0.5", lw=0.6, ls=":")
    ax.set_xlabel("held-out-bird score (R² or balanced accuracy)")
    ax.set_xlim(min(0, p["value"].min() if len(p) else 0), 1)
    return fig


# ---------------------------------------------------------------- M2 global structure


def embedding_sample(df: pd.DataFrame, per_bird: int, total: int, rng) -> pd.DataFrame:
    per_bird = min(per_bird, max(total // max(df["bird"].nunique(), 1), 1))
    idx = [rng.choice(ix, min(per_bird, len(ix)), replace=False) for ix in df.groupby("bird", observed=True).indices.values()]
    return df.iloc[np.sort(np.concatenate(idx))]


def project_2d(H: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    X = StandardScaler().fit_transform(H)
    try:
        import umap  # optional

        p = dict(method="umap", n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=seed, input="standardised h")
        return umap.UMAP(**{k: v for k, v in p.items() if k not in ("method", "input")}).fit_transform(X), p
    except ImportError:
        return PCA(2, random_state=seed).fit_transform(X), dict(method="pca (umap-learn not installed)", input="standardised h")


def fig_embedding_map(sample: pd.DataFrame, xy: np.ndarray, colours: dict, method: str):
    panels = [("state", "state"), ("log10 VeDBA (g)", np.log10(sample["vedba"].clip(lower=1e-3))),
              ("ground speed (m/s)", sample["speed_mean"]), ("local solar hour (h)", sample["local_solar_hour"]),
              ("day of year", sample["doy"]), ("sensitivity setting k", "sensitivity_k"),
              ("recording regime", "regime"), ("sex (reference data)", "sex")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for ax, (title, v) in zip(axes.ravel(), panels):
        if isinstance(v, str):
            cats = sample[v]
            for i, c in enumerate(sorted(cats.dropna().unique()) + ([None] if cats.isna().any() else [])):
                m = cats.isna().to_numpy() if c is None else (cats == c).to_numpy()
                col = ("0.8" if c is None else colours[c] if v == "state" else SEX_COLOURS.get(c)
                       or plt.get_cmap("tab10")(i))
                ax.scatter(xy[m, 0], xy[m, 1], s=1, color=col, label=f"{'missing' if c is None else c} ({m.sum()})",
                           rasterized=True)
            ax.legend(fontsize=6, markerscale=6, loc="best")
        else:
            v = pd.to_numeric(v, errors="coerce").to_numpy(float)
            ok = np.isfinite(v)
            ax.scatter(xy[~ok, 0], xy[~ok, 1], s=1, color="0.85", rasterized=True)
            sc = ax.scatter(xy[ok, 0], xy[ok, 1], s=1, c=v[ok], cmap="twilight" if "hour" in title else "viridis",
                            rasterized=True, vmin=np.nanpercentile(v, 1) if ok.any() else None,
                            vmax=np.nanpercentile(v, 99) if ok.any() else None)
            fig.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{method} of h for display only, {len(sample)} windows, {sample['bird'].nunique()} birds "
                 "(grey = missing)", fontsize=10)
    return fig


# ---------------------------------------------------------------- M3 state selection


def fig_selection(sel: pd.DataFrame, chosen):
    metrics = [c for c in sel.columns if c != "k"]
    fig, ax = plt.subplots(1, max(len(metrics), 1), figsize=(3.3 * max(len(metrics), 1), 3), squeeze=False)
    for a, m in zip(ax[0], metrics):
        a.plot(sel["k"], sel[m], "o-", color="k", ms=3)
        if chosen is not None:
            a.axvline(chosen, color="tab:red", lw=0.8, ls="--")
        a.set_xlabel("number of states K")
        a.set_title(m.replace("_", " "), fontsize=9)
    return fig


def fig_jaccard(jac: pd.Series, colours: dict):
    fig, ax = plt.subplots(figsize=(0.45 * len(jac) + 2, 3))
    ax.bar([str(s) for s in jac.index], jac.to_numpy(), color=[colours.get(s, "0.5") for s in jac.index])
    ax.axhline(STABLE_JACCARD, color="k", lw=0.8, ls="--")
    ax.set_xlabel("state")
    ax.set_ylabel("bootstrap Jaccard (chosen K)")
    ax.set_ylim(0, 1)
    return fig


# ---------------------------------------------------------------- M4 state cards


def state_profile_table(df: pd.DataFrame, states: list) -> pd.DataFrame:
    g = df.groupby("state")
    feats = [f for f in FEATURES if f in df]
    q = g[feats].quantile([0.25, 0.5, 0.75]).unstack()
    q.columns = [f"{f}_{ {0.25: 'q25', 0.5: 'median', 0.75: 'q75'}[p]}" for f, p in q.columns]
    budget = per_bird_budget(df, states)
    t = pd.DataFrame({"windows": g.size(), "share_windows": g.size() / len(df),
                      "birds_with_state": (budget[states] > 0).sum(),
                      "birds_over_1pct": (budget[states] >= 0.01).sum(),
                      "share_full_windows": g["regime"].apply(lambda s: (s == "full").mean()),
                      "share_with_gps": g["gps_fixes"].apply(lambda s: (s > 0).mean()),
                      "mean_local_solar_hour": g["local_solar_hour"].apply(circular_mean_hour)})
    for k in sorted(df["sensitivity_k"].dropna().unique()):
        t[f"share_sensitivity_k{int(k)}"] = g["sensitivity_k"].apply(lambda s, k=k: (s == k).mean())
    t = t.join(q).reindex(states)
    t.index.name = "state"
    return t


def fig_state_profile(prof: pd.DataFrame):
    cols = [c for c in prof.columns if c.endswith("_median")]
    v = prof[cols].astype(float)
    z = (v - v.median()) / (v.quantile(0.75) - v.quantile(0.25)).replace(0, 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 0.45 * len(prof) + 1.5))
    im = ax.imshow(z.to_numpy().clip(-3, 3), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for (i, j), x in np.ndenumerate(v.to_numpy()):
        ax.text(j, i, f"{x:.3g}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), [FEATURES[c[:-7]] for c in cols], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(prof)), [f"state {s}" for s in prof.index])
    fig.colorbar(im, ax=ax, label="median relative to other states (IQR units)")
    ax.set_title("state medians (numbers) and their contrast across states (colour)", fontsize=9)
    return fig


def fig_state_features(df: pd.DataFrame, states: list, colours: dict):
    feats = [f for f in FEATURES if f in df]
    fig, axes = plt.subplots(int(np.ceil(len(feats) / 3)), 3, figsize=(14, 3 * np.ceil(len(feats) / 3)), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        data = [df.loc[df["state"] == s, f].dropna().to_numpy() for s in states]
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], showfliers=False, patch_artist=True,
                        whis=(5, 95))
        for patch, s in zip(bp["boxes"], states):
            patch.set_facecolor(colours[s])
        ax.set_xticks(range(1, len(states) + 1), [str(s) for s in states], fontsize=7)
        ax.set_ylabel(FEATURES[f], fontsize=8)
        if f in ("vedba", "odba", "turn_abs_mean"):
            ax.set_yscale("symlog", linthresh=0.01)
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("window features per state (box 25-75 %, whiskers 5-95 %); x = state", fontsize=10)
    return fig


def label_confusion(df: pd.DataFrame, states: list) -> pd.DataFrame:
    rows = []
    for lab in sorted(c for c in df.columns if c.startswith("label_")):
        for s in states:
            v = df.loc[df["state"] == s, lab]
            known = v.notna()
            rows.append({"state": s, "label": lab[6:], "windows": len(v), "known": int(known.sum()),
                         "true": int((v[known] == 1).sum()),
                         "frac_true_of_known": float((v[known] == 1).mean()) if known.any() else np.nan})
    return pd.DataFrame(rows)


def fig_state_labels(conf: pd.DataFrame):
    m = conf.pivot(index="state", columns="label", values="frac_true_of_known")
    fig, ax = plt.subplots(figsize=(1.0 * m.shape[1] + 2.5, 0.45 * m.shape[0] + 1.5))
    im = ax.imshow(m.to_numpy(float), cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for (i, j), x in np.ndenumerate(m.to_numpy(float)):
        ax.text(j, i, "n/a" if np.isnan(x) else f"{x:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=30, ha="right")
    ax.set_yticks(range(m.shape[0]), [f"state {s}" for s in m.index])
    fig.colorbar(im, ax=ax, label="fraction with label, of windows where the label is known")
    ax.set_title("states x rule-based reference labels (labels are not exclusive)", fontsize=9)
    return fig


def fig_state_format(prof: pd.DataFrame, df: pd.DataFrame, colours: dict):
    cols = ["share_full_windows", "share_with_gps"] + [c for c in prof.columns if c.startswith("share_sensitivity")]
    pop = {"share_full_windows": (df["regime"] == "full").mean(), "share_with_gps": (df["gps_fixes"] > 0).mean()}
    pop.update({c: (df["sensitivity_k"] == int(c.rsplit("k", 1)[1])).mean() for c in cols if c.startswith("share_sens")})
    fig, axes = plt.subplots(1, len(cols) + 1, figsize=(3.2 * (len(cols) + 1), 3.2))
    x = [str(s) for s in prof.index]
    for ax, c in zip(axes, cols):
        ax.bar(x, prof[c], color=[colours[s] for s in prof.index])
        ax.axhline(pop[c], color="k", ls="--", lw=0.8, label="all windows")
        ax.set_ylim(0, 1)
        ax.set_title(c.replace("share_", "share ").replace("_", " "), fontsize=9)
        ax.set_xlabel("state")
    axes[0].legend(fontsize=7)
    nb = df["bird"].nunique()
    axes[-1].bar(x, prof["birds_over_1pct"] / nb, color=[colours[s] for s in prof.index])
    axes[-1].set_ylim(0, 1)
    axes[-1].set_title(f"share of {nb} birds with >= 1 % of windows", fontsize=9)
    axes[-1].set_xlabel("state")
    return fig


def pick_examples(df: pd.DataFrame, mask: np.ndarray, n: int, rng) -> pd.DataFrame:
    """Up to n windows from distinct birds where possible, among the confident half (p_state >= median).

    p_state is the GMM posterior of the unsmoothed state, so only windows whose GMM state equals the card's
    state are used when there are enough of them."""
    sub = df[mask]
    if "state_gmm" in sub:
        agree = sub[sub["state_gmm"] == sub["state"]]
        sub = agree if len(agree) >= n else sub
    if "p_state" in sub and sub["p_state"].notna().any():
        sub = sub[sub["p_state"] >= sub["p_state"].median()]
    sub = sub.iloc[rng.permutation(len(sub))]
    first = sub.drop_duplicates("bird")
    return pd.concat([first, sub.drop(first.index)]).head(n)


def fig_state_card(df, windows, s, prof, colours, rng):
    ex = pick_examples(df, (df["state"] == s).to_numpy(), 8, rng)
    traces = load_traces(windows, ex[["bird", "row"]])
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5), sharey=True, squeeze=False, layout="constrained")
    for ax, a, (_, r) in zip(axes.ravel(), traces, ex.iterrows()):
        plot_trace(ax, a, f"{r['bird']} {r['t_start']:%Y-%m-%d %H:%M} UTC\nk={r['sensitivity_k']} "
                          f"samples={r['acc_samples']} gps={r['gps_fixes']}")
    for ax in axes.ravel()[len(traces):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, ncol=4)
    for ax in axes[:, 0]:
        ax.set_ylabel("acceleration (g)")
    for ax in axes[-1]:
        ax.set_xlabel("time in window (s)")
    p = prof.loc[s]
    fig.suptitle(f"state {s}: {int(p['windows'])} windows ({p['share_windows']:.1%}), used by "
                 f"{int(p['birds_over_1pct'])} birds (>= 1 %); median VeDBA {p.get('vedba_median', np.nan):.3g} g, "
                 f"speed {p.get('speed_mean_median', np.nan):.3g} m/s", fontsize=10, color=colours[s])
    return fig


# ---------------------------------------------------------------- M5 temporal


def runs_table(df: pd.DataFrame, col: str = "state_hmm"):
    """Runs of one state over consecutive windows of a continuous segment; censored = touches a segment end."""
    d = df.sort_values(["bird", "segment", "k_in_segment"])[["bird", "segment", "k_in_segment", col]]
    b, seg, k, s = (d[c].to_numpy() for c in ("bird", "segment", "k_in_segment", col))
    b = pd.Categorical(b).codes
    same_seg = np.r_[False, (b[1:] == b[:-1]) & (seg[1:] == seg[:-1]) & (k[1:] == k[:-1] + 1)]
    new_run = ~same_seg | np.r_[True, s[1:] != s[:-1]]
    rid = np.cumsum(new_run)
    g = pd.DataFrame({"run": rid, "state": s, "seg_start": ~same_seg,
                      "seg_end": ~np.r_[same_seg[1:], False]}).groupby("run")
    r = pd.DataFrame({"state": g["state"].first(), "windows": g.size(),
                      "censored": g["seg_start"].first() | g["seg_end"].last()})
    r["duration_s"] = r["windows"] * WINDOW_S
    return r, same_seg, s


def transition_matrix(same_seg: np.ndarray, s: np.ndarray, states: list) -> pd.DataFrame:
    i = np.flatnonzero(same_seg)
    c = pd.crosstab(pd.Categorical(s[i - 1], states), pd.Categorical(s[i], states), dropna=False)
    c.index.name, c.columns.name = "from", "to"
    return c


def fig_transitions(counts: pd.DataFrame):
    p = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(0.5 * len(p) + 2.5, 0.5 * len(p) + 2))
    im = ax.imshow(p.to_numpy(float), cmap="magma_r", vmin=0, vmax=1)
    for (i, j), x in np.ndenumerate(p.to_numpy(float)):
        if np.isfinite(x) and x >= 0.01:
            ax.text(j, i, f"{x:.2f}", ha="center", va="center", fontsize=6, color="w" if x > 0.5 else "k")
    ax.set_xticks(range(len(p)), p.columns)
    ax.set_yticks(range(len(p)), p.index)
    ax.set_xlabel("state in next 10 s window")
    ax.set_ylabel("state in window")
    fig.colorbar(im, ax=ax, label="P(next | current), consecutive windows within segments")
    return fig


def fig_dwell(runs: pd.DataFrame, states: list, colours: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, cens in zip(axes, (False, True)):
        for s in states:
            d = np.sort(runs.loc[(runs["state"] == s) & (runs["censored"] == cens), "duration_s"].to_numpy())
            if len(d):
                ax.step(d, 1 - np.arange(len(d)) / len(d), where="post", color=colours[s], label=f"{s} ({len(d)})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("run duration (s)")
        ax.set_title("complete runs" if not cens else "censored at a segment end (lower bound)", fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6, title="state (runs)", title_fontsize=6)
    axes[0].set_ylabel("P(duration >= x)")
    return fig


MODE_K = {1: "k=1", 0: "k=0"}  # sensitivity setting = recording mode (k=1: continuous, always with GPS)


def seasonal_by_mode(df: pd.DataFrame, states: list) -> tuple[pd.DataFrame, dict]:
    """Bird-weighted state share per calendar month (UTC, years pooled), raw and stratified by recording mode
    (sensitivity_k 1 vs 0), by GPS present/absent and by both; plus a mode-adjusted share.

    Mode-adjusted: within each month, the k=0 and k=1 shares are combined with fixed weights, the bird-weighted
    overall mode mix (mean over birds of the bird's share of k=1 windows, among its k=0/k=1 windows). A month
    without a share for both modes gets NaN, never a fill. Returns (long table, mode weights)."""
    d = df.assign(cal_month=df["t_start"].dt.month,
                  mode=df["sensitivity_k"].map(MODE_K),
                  gps=np.where(df["gps_fixes"] > 0, "GPS", "no GPS"))
    d["mode_gps"] = d["mode"] + ", " + d["gps"]
    rows = []

    def add(strat, stratum, sub):
        share, nb, nw = bird_share(sub, "cal_month", states)
        t = share.add_prefix("share_state_")
        t["n_birds"], t["n_windows"] = nb, nw.reindex(t.index)
        rows.append(t.reset_index().rename(columns={"cal_month": "month"}).assign(stratification=strat,
                                                                                    stratum=stratum))
        return share, nb

    add("all", "all windows", d)
    per = {}
    for col in ("mode", "gps", "mode_gps"):
        for v in sorted(d[col].dropna().unique()):
            per[v] = add(col, v, d[d[col] == v])
    km = d[d["mode"].notna()]
    k1 = km.groupby("bird", observed=True)["mode"].apply(lambda s: (s == "k=1").mean())
    w = {"k=1": float(k1.mean()) if len(k1) else np.nan}
    w["k=0"] = 1 - w["k=1"]
    if "k=0" in per and "k=1" in per:
        s0, s1 = per["k=0"][0], per["k=1"][0]
        months = s0.index.union(s1.index)
        adj = s0.reindex(months) * w["k=0"] + s1.reindex(months) * w["k=1"]  # NaN where a mode is missing
        t = adj.add_prefix("share_state_")
        t["n_birds"] = np.minimum(per["k=0"][1].reindex(months), per["k=1"][1].reindex(months))
        t["n_windows"] = km.groupby("cal_month").size().reindex(months)
        rows.append(t.rename_axis("month").reset_index().assign(stratification="mode_adjusted",
                                                                 stratum="mode-adjusted"))
    tab = pd.concat(rows, ignore_index=True)
    first = ["stratification", "stratum", "month", "n_birds", "n_windows"]
    return tab[first + [c for c in tab.columns if c not in first]], w


def seasonal_mode_contrast(tab: pd.DataFrame, states: list) -> pd.DataFrame:
    """Per state: mean monthly share in the months where k=1 has a share minus the mean in the other months, for
    all windows and within k=0 only. A contrast that keeps its sign within k=0 is not only the recording schedule."""
    k1 = tab.loc[(tab["stratum"] == "k=1") & tab.filter(like="share_state_").notna().any(axis=1), "month"]
    k1 = set(k1.astype(int))
    rows = []
    for s in states:
        r = {"state": s, "k1_months": " ".join(str(m) for m in sorted(k1))}
        for name, stratum in (("all", "all windows"), ("within_k0", "k=0")):
            t = tab[tab["stratum"] == stratum].dropna(subset=[f"share_state_{s}"])
            a, b = t[t["month"].isin(k1)], t[~t["month"].isin(k1)]
            r[f"contrast_{name}"] = (a[f"share_state_{s}"].mean() - b[f"share_state_{s}"].mean()
                                     if len(a) and len(b) else np.nan)
            r[f"months_in_{name}"], r[f"months_out_{name}"] = len(a), len(b)
        r["same_sign_within_k0"] = (bool(np.sign(r["contrast_all"]) == np.sign(r["contrast_within_k0"]))
                                    if np.isfinite(r["contrast_all"]) and np.isfinite(r["contrast_within_k0"])
                                    else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def seasonal_mode_note(tab: pd.DataFrame, df: pd.DataFrame, states: list, w: dict,
                       contrast: pd.DataFrame | None = None) -> list[str]:
    """Seasonal range (max - min over months with a share) per state, raw vs within one mode vs mode-adjusted."""
    def rng_of(stratum, s):
        v = tab.loc[tab["stratum"] == stratum, f"share_state_{s}"].dropna()
        return (f"{v.max() - v.min():.2f} ({len(v)} mo)") if len(v) else "n/a"

    cols = ["all windows", "k=0", "k=1", "mode-adjusted"]
    parts = [f"state {s}: " + ", ".join(f"{c} {rng_of(c, s)}" for c in cols) for s in states]
    notes = ["m5_seasonal_by_mode: seasonal range of the bird-weighted monthly state share (max - min over the "
             "calendar months with a share; months in brackets), raw / within k=0 only / within k=1 only / "
             f"mode-adjusted (fixed mix k=1 {w.get('k=1', np.nan):.2f}). A seasonal difference that also appears "
             "within k=0 (which covers the whole year) is not explained by the recording schedule alone; one seen "
             "only in the raw shares can be the April-August concentration of k=1. " + "; ".join(parts)]
    if contrast is not None and contrast["contrast_all"].notna().any():
        c = contrast
        notes.append(
            f"m5_seasonal_by_mode_contrast: mean monthly share in the months where k=1 has a share "
            f"({c['k1_months'].iat[0]}) minus the other months, all windows vs within k=0 only; a contrast that keeps "
            "its sign within k=0 is not only the recording schedule. " + "; ".join(
                f"state {r.state}: all {r.contrast_all:+.3f}, within k=0 {r.contrast_within_k0:+.3f} ("
                + ("same sign" if r.same_sign_within_k0 is True else "sign differs" if r.same_sign_within_k0 is False
                   else "n/a") + ")" for r in c.itertuples()))
    elif contrast is not None:
        notes.append("m5_seasonal_by_mode_contrast: no contrast (k=1 has a share in no month or in every month)")
    adj = tab[tab["stratum"] == "mode-adjusted"]
    nan_m = sorted(adj.loc[adj.filter(like="share_state_").isna().all(axis=1), "month"].astype(int).tolist())
    all_m = sorted(tab.loc[tab["stratum"] == "all windows", "month"].astype(int).tolist())
    missing = [m for m in all_m if m not in set(adj.loc[adj.filter(like="share_state_").notna().any(axis=1),
                                                         "month"].astype(int))]
    if missing or nan_m:
        notes.append(f"m5_seasonal_by_mode: no mode-adjusted share in months {sorted(set(missing) | set(nan_m))} "
                     "(one mode has no bird with enough windows there); left empty, not filled")
    k0g = df[(df["sensitivity_k"] == 0) & (df["gps_fixes"] > 0)]
    if len(k0g):
        short = float((k0g["regime"] == "non-full").mean())
        notes.append(f"m5_seasonal_by_mode: {len(k0g)} k=0 windows have GPS; {short:.1%} of them are non-full "
                     "(short) windows" + (", so the 'k=0, GPS' stratum is mostly short windows" if short > 0.5 else ""))
    return notes


def fig_seasonal_by_mode(tab: pd.DataFrame, states: list, colours: dict):
    """One panel per stratum: monthly bird-weighted state shares (lines) and birds per month (grey bars)."""
    panels = list(dict.fromkeys(tab["stratum"]))
    panels = [p for p in ["all windows", "mode-adjusted"] if p in panels] + \
             [p for p in panels if p not in ("all windows", "mode-adjusted")]
    nc = 4
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.3 * nr), squeeze=False, sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), panels):
        t = tab[tab["stratum"] == p].set_index("month").reindex(range(1, 13)).reset_index()  # gaps break lines
        tw = ax.twinx()
        tw.bar(t["month"], t["n_birds"].fillna(0), color="0.88", width=0.8, zorder=0)
        tw.set_ylim(0, max(tab["n_birds"].max() if tab["n_birds"].notna().any() else 1, 1) * 3)
        tw.tick_params(labelsize=6, colors="0.5")
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)
        for s in states:
            ax.plot(t["month"], t[f"share_state_{s}"], "o-", ms=2.5, lw=1, color=colours[s], label=f"state {s}")
        ax.set_title(p, fontsize=9)
        ax.set_xticks(range(1, 13))
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(0, 1)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("bird-weighted state share")
    for ax in axes[-1]:
        ax.set_xlabel("calendar month (UTC, years pooled)")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("monthly state use by recording mode (k=1 continuous, k=0 burst) and GPS; grey bars = birds in "
                 "the share (right axis); mode-adjusted = fixed overall mode mix within each month", fontsize=9)
    return fig


# ---------------------------------------------------------------- M6 individuals


def fig_bird_budgets(budget: pd.DataFrame, states: list):
    b = budget.sort_values(by=states, ascending=False)
    b = b.loc[b[states].idxmax(axis=1).map(states.index).sort_values(kind="stable").index]
    fig, ax = plt.subplots(figsize=(0.55 * len(states) + 4, 0.17 * len(b) + 1.5))
    im = ax.imshow(b[states].to_numpy(float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(states)), [str(s) for s in states])
    ax.set_yticks(range(len(b)), [f"{i} (n={int(n)})" for i, n in zip(b.index, b["windows"])], fontsize=5)
    ax.set_xlabel("state")
    fig.colorbar(im, ax=ax, label="share of the bird's windows")
    return fig


def fig_calendar(df: pd.DataFrame, colours: dict):
    """Daily dominant state per bird (UTC dates); light grey = no windows that day within the bird's span."""
    day = df["t_start"].dt.floor("D").dt.tz_localize(None)
    dom = df.assign(day=day).groupby(["bird", "day", "state"], observed=True).size().unstack("state").idxmax(axis=1)
    days = pd.date_range(day.min(), day.max(), freq="D")
    birds = list(dom.index.get_level_values(0).unique())
    img = np.ones((len(birds), len(days), 3))
    di = {d: i for i, d in enumerate(days)}
    for r, b in enumerate(birds):
        s = dom.loc[b]
        lo, hi = di[s.index.min()], di[s.index.max()]
        img[r, lo:hi + 1] = 0.88
        for d, st in s.items():
            img[r, di[d]] = matplotlib.colors.to_rgb(colours[st])
    fig, ax = plt.subplots(figsize=(14, 0.16 * len(birds) + 1.8))
    ax.imshow(img, aspect="auto", interpolation="nearest",
              extent=(matplotlib.dates.date2num(days[0]), matplotlib.dates.date2num(days[-1] + pd.Timedelta(days=1)),
                      len(birds) - 0.5, -0.5))
    ax.xaxis_date()
    ax.set_yticks(range(len(birds)), birds, fontsize=5)
    ax.set_xlabel("date (UTC)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[s]) for s in sorted(colours)]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.88"))
    ax.legend(handles, [f"state {s}" for s in sorted(colours)] + ["no windows (gap)"], fontsize=6,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("dominant state per bird and day", fontsize=9)
    return fig


def bird_summary(df: pd.DataFrame, budget: pd.DataFrame, states: list, meta: pd.DataFrame, rng) -> pd.DataFrame:
    """Per-bird coverage and budget diversity (entropy on an equal-size subsample, so coverage does not drive it)."""
    big = budget["windows"] >= MIN_BIRD_WINDOWS  # smaller birds get NaN, so one tiny bird does not set the size
    n_eq = int(min(budget.loc[big, "windows"].min(), 2000)) if big.any() else 0
    ent = {}
    for b, ix in df.groupby("bird", observed=True).indices.items():
        if len(ix) < max(n_eq, MIN_BIRD_WINDOWS):
            continue
        s = df["state"].to_numpy()[rng.choice(ix, n_eq, replace=False)]
        p = np.bincount(pd.Categorical(s, states).codes, minlength=len(states)) / n_eq
        ent[b] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    g = df.groupby("bird", observed=True)
    t = pd.DataFrame({"windows": g.size(), "first": g["t_start"].min(), "last": g["t_start"].max(),
                      "days_with_windows": g["t_start"].apply(lambda s: s.dt.floor("D").nunique()),
                      "share_full_windows": g["regime"].apply(lambda s: (s == "full").mean()),
                      "median_vedba_g": g["vedba"].median(), "state_entropy_bits": pd.Series(ent),
                      "entropy_subsample": n_eq})
    t["span_days"] = (t["last"] - t["first"]).dt.total_seconds() / 86400
    t = t.join(budget[states].add_prefix("share_state_")).join(meta.set_index("bird")[["sex", "life_stage"]])
    t.index.name = "bird"
    return t


def fig_bird_diversity(summ: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    for sex, c in SEX_COLOURS.items():
        m = summ["sex"] == sex
        ax.scatter(summ.loc[m, "windows"], summ.loc[m, "state_entropy_bits"], s=12, color=c, label=f"{sex} ({m.sum()})")
    m = ~summ["sex"].isin(["m", "f"])
    ax.scatter(summ.loc[m, "windows"], summ.loc[m, "state_entropy_bits"], s=12, color="0.6", label=f"sex missing ({m.sum()})")
    ax.set_xscale("log")
    ax.set_xlabel("windows per bird")
    ax.set_ylabel(f"state-budget entropy (bits, {int(summ['entropy_subsample'].iat[0])} windows per bird)")
    ax.legend(fontsize=7)
    return fig


def permutation_test(budget: pd.DataFrame, groups: pd.Series, states: list, n_perm: int, rng) -> pd.DataFrame:
    """Between-group sum of squares of per-bird state shares, per state and summed over states; labels permuted."""
    X = budget[states].to_numpy(float)
    codes = pd.Categorical(groups).codes
    G = codes.max() + 1

    def stat(c):  # c: (P, n) group codes -> (P, S) between-group SS
        oh = (c[:, :, None] == np.arange(G)).astype(float)  # P n G
        n = oh.sum(1)  # P G
        mean = np.einsum("png,ns->pgs", oh, X) / n[:, :, None]
        return (n[:, :, None] * (mean - X.mean(0)) ** 2).sum(1)

    obs = stat(codes[None])[0]
    perm = np.stack([rng.permutation(codes) for _ in range(n_perm)])
    null = np.concatenate([stat(perm[i:i + 1000]) for i in range(0, n_perm, 1000)])
    p_state = (1 + (null >= obs - 1e-12).sum(0)) / (1 + n_perm)
    p_all = (1 + (null.sum(1) >= obs.sum() - 1e-12).sum()) / (1 + n_perm)
    rows = [{"state": "all (sum)", "p_value": p_all}] + [{"state": s, "p_value": p} for s, p in zip(states, p_state)]
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        for r in rows[1:]:
            r[f"mean_share_{g}"] = float(X[m, states.index(r["state"])].mean())
    return pd.DataFrame(rows)


def group_tests(budget, meta, states, n_perm, rng, out: Outputs, colours, fields=("sex", "life_stage"),
                descriptive: dict | None = None):
    """Bird-level permutation tests per field; ``descriptive`` fields get their caveat in the status column."""
    rows = []
    descriptive = descriptive or {}
    b = budget[budget["windows"] >= MIN_BIRD_WINDOWS].join(meta.set_index("bird"))
    for field in fields:
        g = b[field].dropna()
        counts = g.value_counts()
        keep = counts[counts >= 2].index
        base = {"field": field, "birds_with_value": int(len(g)), "birds_missing": int(b[field].isna().sum()),
                "groups": "; ".join(f"{k}: {v}" for k, v in counts.items())}
        if len(keep) < 2:
            rows.append({**base, "state": "", "p_value": np.nan,
                         "status": "not tested: fewer than 2 groups with >= 2 birds"})
            out.note(f"{field}: not tested and m6_group_{field}.png not produced (groups with >= {MIN_BIRD_WINDOWS} "
                     f"windows per bird: {base['groups'] or 'none'})")
            continue
        sub = b[b[field].isin(keep)]
        t = permutation_test(sub, sub[field], states, n_perm, rng)
        status = f"tested, {len(sub)} birds, {n_perm} permutations"
        if field in descriptive:
            status = f"{status}; descriptive only: {descriptive[field]}"
        rows += [{**base, **r, "status": status} for r in t.to_dict("records")]
        q = (f"does per-bird state use differ by {field.replace('_', ' ')}? (bird level, n per group in legend)"
             + (f"; descriptive only: {descriptive[field]}" if field in descriptive else ""))
        out.fig(fig_group(sub, field, states, colours), f"m6_group_{field}.png", "M6", q)
    return pd.DataFrame(rows)


def development_table(df: pd.DataFrame, ref: pd.DataFrame, reference, states: list, out: Outputs,
                      fledges: pd.DataFrame | None = None):
    """Windows of tagged-young birds outside pre_fledge, with months since fledging (or months of age when the
    fledging age cannot be read from reference.age_on). Returns (windows with ``dev_month``, x-axis label).

    Fledging: the per-bird fledge dates of window_age when the reference module supplied them; otherwise the age
    at which age_on first gives juvenile_y1 (fledge_age_days)."""
    young = set(ref.index[ref["tagged_young"].eq(True)])
    sub = df[df["bird"].astype(str).isin(young) & (df["age_class"] != "pre_fledge") & df["age_days"].notna()].copy()
    if not len(sub):
        return sub, None
    if fledges is not None and len(fledges) and fledges["fledge"].notna().all() \
            and set(sub["bird"].astype(str)) <= set(fledges["bird"]):
        f = pd.to_datetime(sub["bird"].astype(str).map(fledges.set_index("bird")["fledge"]), utc=True)
        src = fledges["fledge_source"].value_counts().to_dict()
        out.note(f"m6_development: fledge dates from aquilacl.reference.fledge_from_speed, by source {src}; "
                 f"{int(fledges['fledge_is_first_data_day'].eq(True).sum())} of them are the "
                 "bird's first day of data (an upper bound)")
        sub["dev_month"] = np.floor((sub["t_start"] - f).dt.total_seconds() / 86400 / DAYS_PER_MONTH)
        sub = sub[sub["dev_month"] >= 0]
        sub["group"] = "tagged young"
        return sub, "months since fledging"
    first = sub.groupby("bird", observed=True)["t_start"].min()
    fl = pd.Series({str(b): fledge_age_days(reference, ref.loc[str(b)], t) for b, t in first.items()})
    if fl.notna().any():
        if fl.isna().any():
            out.note(f"m6_development: {int(fl.isna().sum())} of {len(fl)} tagged-young birds left out: "
                     "reference.age_on shows no pre_fledge -> juvenile_y1 change for them, so no fledging age")
        sub = sub[sub["bird"].astype(str).isin(fl.index[fl.notna()])]
        x = (sub["age_days"] - sub["bird"].astype(str).map(fl).astype(float)) / DAYS_PER_MONTH
        label = "months since fledging"
    else:
        out.note("m6_development: reference.age_on shows no pre_fledge -> juvenile_y1 change for any tagged-young "
                 "bird, so the x axis is months of age (from age_days), not months since fledging")
        x = sub["age_days"] / DAYS_PER_MONTH
        label = "months of age"
    sub["dev_month"] = np.floor(x)
    sub = sub[sub["dev_month"] >= 0]
    sub["group"] = "tagged young"
    return sub, label


def y1_vs_y2(df: pd.DataFrame, ref: pd.DataFrame, states: list, n_perm: int, rng):
    """Same calendar month, first year (juvenile_y1) vs second year (y2), per tagged-young bird.

    A (bird, month) pair needs >= MIN_BIN_WINDOWS windows in each year. Differences (y2 - y1) are averaged within
    each bird first, so each bird counts once in the sign-flip test. The k=1 and GPS shares of each side are
    reported, because the recording mode can differ between the two years. Returns (pairs, summary)."""
    young = set(ref.index[ref["tagged_young"].eq(True)])
    sub = df[df["bird"].astype(str).isin(young) & df["age_class"].isin(["juvenile_y1", "y2"])]
    sub = sub.assign(cal_month=sub["t_start"].dt.month, year_of_life=sub["age_class"].map(
        {"juvenile_y1": "y1", "y2": "y2"}), k1=(sub["sensitivity_k"] == 1).astype(float),
        gps=(sub["gps_fixes"] > 0).astype(float), bird=sub["bird"].astype(str))
    keys = ["bird", "cal_month", "year_of_life"]
    c = sub.groupby(keys + ["state"]).size().unstack("state", fill_value=0).reindex(columns=states, fill_value=0)
    n = c.sum(axis=1)
    t = c.div(n, axis=0).add_prefix("share_state_")
    t["windows"] = n
    t["k1_share"] = sub.groupby(keys)["k1"].mean()
    t["gps_share"] = sub.groupby(keys)["gps"].mean()
    t = t[t["windows"] >= MIN_BIN_WINDOWS].unstack("year_of_life")
    if not len(t) or "y1" not in t.columns.get_level_values(1) or "y2" not in t.columns.get_level_values(1):
        return pd.DataFrame(), pd.DataFrame()
    t = t.dropna(subset=[(f"share_state_{states[0]}", "y1"), (f"share_state_{states[0]}", "y2")])
    if not len(t):
        return pd.DataFrame(), pd.DataFrame()
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    pairs = t.reset_index().rename(columns={"cal_month": "month"})
    for s in states:
        pairs[f"diff_state_{s}"] = pairs[f"share_state_{s}_y2"] - pairs[f"share_state_{s}_y1"]
    per_bird = pairs.groupby("bird")[[c for c in pairs if c.startswith(("share_state_", "diff_state_"))]].mean()
    D = per_bird[[f"diff_state_{s}" for s in states]].to_numpy(float)
    nb = len(D)
    if 2 ** nb <= n_perm:  # few birds: all sign patterns (exact)
        signs = 1.0 - 2.0 * ((np.arange(2 ** nb)[:, None] >> np.arange(nb)) & 1)
        n_perm, how = len(signs), "all sign patterns (exact)"
    else:
        signs, how = rng.choice([-1.0, 1.0], size=(n_perm, nb)), "random sign flips"
    null = np.abs(signs @ D) / nb
    obs = np.abs(D.mean(0))
    hits = (null >= obs - 1e-12).sum(0)
    p = hits / n_perm if how.endswith("(exact)") else (1 + hits) / (1 + n_perm)  # exact set includes the observed
    summ = pd.DataFrame({"state": states, "birds": nb, "bird_months": len(pairs),
                         "mean_share_y1": [per_bird[f"share_state_{s}_y1"].mean() for s in states],
                         "mean_share_y2": [per_bird[f"share_state_{s}_y2"].mean() for s in states],
                         "mean_diff_y2_minus_y1": D.mean(0), "p_sign_flip": p,
                         "mean_k1_share_y1": pairs["k1_share_y1"].mean(), "mean_k1_share_y2": pairs["k1_share_y2"].mean(),
                         "mean_gps_share_y1": pairs["gps_share_y1"].mean(),
                         "mean_gps_share_y2": pairs["gps_share_y2"].mean(),
                         "status": f"{nb} birds, {n_perm} {how}, bird-level (pairs averaged within bird)"})
    return pairs, summ


def fig_y1_vs_y2(pairs: pd.DataFrame, summ: pd.DataFrame, states: list, colours: dict):
    pb = pairs.groupby("bird")[[c for c in pairs if c.startswith("share_state_")]].mean()
    fig, axes = plt.subplots(1, len(states), figsize=(1.9 * len(states) + 1, 3.4), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, s, (_, r) in zip(axes[0], states, summ.iterrows()):
        for _, b in pb.iterrows():
            ax.plot([0, 1], [b[f"share_state_{s}_y1"], b[f"share_state_{s}_y2"]], "-o", ms=3, lw=0.8,
                    color=colours[s], alpha=0.7)
        ax.set_xticks([0, 1], ["year 1", "year 2"], fontsize=7)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"state {s}\np = {r['p_sign_flip']:.3g}", fontsize=8)
    axes[0, 0].set_ylabel("share in matched calendar months")
    fig.suptitle(f"tagged-young birds, same calendar months in year 1 and year 2\n({int(summ['birds'].iat[0])} birds, "
                 f"{int(summ['bird_months'].iat[0])} bird-months; one line per bird)", fontsize=9)
    return fig


def fig_group(sub: pd.DataFrame, field: str, states: list, colours: dict):
    groups = sorted(sub[field].unique())
    fig, ax = plt.subplots(figsize=(0.9 * len(states) + 2, 3.8))
    w = 0.8 / len(groups)
    rng = np.random.default_rng(0)
    for j, g in enumerate(groups):
        v = sub[sub[field] == g]
        for i, s in enumerate(states):
            x = i + (j - (len(groups) - 1) / 2) * w
            ax.scatter(x + rng.uniform(-w / 4, w / 4, len(v)), v[s], s=8, color=plt.get_cmap("tab10")(j),
                       label=f"{g} ({len(v)} birds)" if i == 0 else None)
            ax.hlines(v[s].median(), x - w / 2.5, x + w / 2.5, color="k", lw=1.2)
    ax.set_xticks(range(len(states)), [str(s) for s in states])
    ax.set_xlabel("state")
    ax.set_ylabel("share of the bird's windows")
    ax.legend(fontsize=7, title=field.replace("_", " "))
    return fig


# ---------------------------------------------------------------- M7 rare behaviour


def knn_scores(df: pd.DataFrame, run: Path, rng, per_group=300, min_group=30, k=10) -> pd.DataFrame:
    """kNN distance in standardised h within bird x month x recording regime, relative to the group median.

    Scores a sample of up to ``per_group`` windows of every group with >= ``min_group`` windows (empty frame when
    no group is large enough)."""
    grp = df.groupby(["bird", "month", "regime"], observed=True).indices
    picks = [rng.choice(ix, min(per_group, len(ix)), replace=False) for ix in grp.values() if len(ix) >= min_group]
    if not picks:
        return pd.DataFrame()
    sample = df.iloc[np.sort(np.concatenate(picks))]
    X = StandardScaler().fit_transform(load_h(run, sample[["bird", "row"]]))
    score = np.full(len(sample), np.nan)
    pos = pd.Series(np.arange(len(sample)), index=sample.index)
    for _, g in sample.groupby(["bird", "month", "regime"], observed=True):
        i = pos[g.index].to_numpy()
        d = NearestNeighbors(n_neighbors=min(k + 1, len(i))).fit(X[i]).kneighbors(X[i])[0][:, -1]
        score[i] = d / np.median(d)
    return sample.assign(knn_score=score)


def top_outliers(scored: pd.DataFrame, top=24, per_bird=3) -> pd.DataFrame:
    """The gallery: highest kNN scores, at most ``per_bird`` per bird."""
    if not len(scored):
        return pd.DataFrame()
    sample = scored.sort_values("knn_score", ascending=False)
    sample = sample.groupby("bird", observed=True).head(per_bird).head(top)
    cols = ["bird", "row", "t_start", "month", "regime", "state", "knn_score", "acc_samples", "sensitivity_k",
            "gps_fixes", "vedba", "speed_mean", "solar_hour_bin"]
    return sample[[c for c in cols if c in sample]]


def knn_outliers(df: pd.DataFrame, run: Path, rng, per_group=300, min_group=30, k=10, top=24, per_bird=3):
    """kNN distance in standardised h within bird x month x recording regime, relative to the group median."""
    return top_outliers(knn_scores(df, run, rng, per_group, min_group, k), top, per_bird)


OUTLIER_FACTORS = {  # factor: window mask
    f"short window (acc_samples < {FULL_SAMPLES})": lambda d: d["acc_samples"] < FULL_SAMPLES,
    "sensitivity k=1 (continuous mode)": lambda d: d["sensitivity_k"] == 1,
    "GPS present (gps_fixes > 0)": lambda d: d["gps_fixes"] > 0,
}


UNRELIABLE_FACTOR = "bird had unreliable-calibration bursts left out (windows_summary.csv)"


def unreliable_calibration_birds(windows: Path) -> set | None:
    """Birds with bursts_unreliable_calibration > 0 in the windows directory's windows_summary.csv (written by
    windows.py); None when the file or column is absent."""
    p = windows / "windows_summary.csv"
    if not p.exists():
        return None
    w = pd.read_csv(p, dtype={"bird": str})
    if "bird" not in w or "bursts_unreliable_calibration" not in w:
        return None
    return set(w.loc[pd.to_numeric(w["bursts_unreliable_calibration"], errors="coerce").fillna(0) > 0, "bird"])


def outlier_factors(scored: pd.DataFrame, gallery: pd.DataFrame, top_frac: float = 0.01,
                    unreliable_birds: set | None = None) -> pd.DataFrame:
    """Share of outlier windows with each recording factor against its share among all scored windows.

    ratio = share among outliers / share among scored windows (1 = not over-represented). Two outlier sets: the
    top ``top_frac`` of scored windows by kNN score, and the gallery of m7_outliers.csv. With
    ``unreliable_birds``, a bird-level factor: the window's bird had bursts of an unreliable calibration left
    out (those bursts never become windows, so a window-level or period-level flag cannot exist)."""
    factors = dict(OUTLIER_FACTORS)
    if unreliable_birds is not None:
        factors[UNRELIABLE_FACTOR] = lambda d: d["bird"].astype(str).isin(unreliable_birds)
    n_top = max(int(np.ceil(top_frac * scored["knn_score"].notna().sum())), 1)
    sets = {f"top {top_frac:.0%} of scored windows": scored.nlargest(n_top, "knn_score"),
            "gallery (m7_outliers.csv)": gallery}
    rows = []
    for name, s in sets.items():
        for f, fn in factors.items():
            base = float(fn(scored).mean())
            share = float(fn(s).mean()) if len(s) else np.nan
            rows.append({"outlier_set": name, "factor": f, "n_outliers": len(s), "n_outliers_with_factor":
                         int(fn(s).sum()) if len(s) else 0, "share_outliers": share, "n_scored": len(scored),
                         "share_scored": base, "ratio": share / base if base > 0 else np.nan})
    return pd.DataFrame(rows)


def fig_gallery(windows: Path, rows: pd.DataFrame, title: str, extra: str = ""):
    traces = load_traces(windows, rows[["bird", "row"]])
    nc = 6
    nr = max(int(np.ceil(len(traces) / nc)), 1)
    fig, axes = plt.subplots(nr, nc, figsize=(18, 2.8 * nr + 0.6), squeeze=False, sharey=True, layout="constrained")
    for ax, a, (_, r) in zip(axes.ravel(), traces, rows.iterrows()):
        t = f"{r['bird']} {r['t_start']:%Y-%m-%d %H:%M}\nstate {r['state']} k={r['sensitivity_k']} n={r['acc_samples']}"
        plot_trace(ax, a, t + (f" {extra}={r[extra]:.2f}" if extra else ""))
    for ax in axes.ravel()[len(traces):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, ncol=4)
    for ax in axes[:, 0]:
        ax.set_ylabel("acceleration (g)")
    for ax in axes[-1]:
        ax.set_xlabel("time in window (s)")
    fig.suptitle(title, fontsize=10)
    return fig


# ---------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states", type=Path, required=True, help="stage 5a output directory")
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True, help="Movebank reference-data CSV")
    ap.add_argument("--out", type=Path, required=True, help="new or empty directory")
    ap.add_argument("--state-col", choices=["state_hmm", "state"], default="state_hmm",
                    help="analysis state: HMM-smoothed (default; 5a profiles use it) or unsmoothed GMM state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-per-bird", type=int, default=500)
    ap.add_argument("--map-total", type=int, default=40000)
    ap.add_argument("--permutations", type=int, default=9999)
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; give a new directory")
    args.out.mkdir(parents=True, exist_ok=True)
    out = Outputs(args.out)
    rng = np.random.default_rng(args.seed)

    df = load_windows(args.states, args.windows, out, args.state_col)
    check_embeddings(args.run, df)
    states = sorted(set(df["state_gmm"]) | set(df.get("state_hmm", df["state"])))
    colours = state_colours(states)
    birds = sorted(str(b) for b in df["bird"].unique())
    reference, ref = load_age(args.reference, birds, out)
    age = ref is not None
    meta, cov = load_reference(args.reference, birds, out, age_source=age)
    df["sex"] = df["bird"].astype(str).map(meta.set_index("bird")["sex"])
    if age:
        df["age_days"], df["age_class"], fledges = window_age(df, ref, reference)
        gmeta = pd.DataFrame({"bird": birds, "tagged_young": [
            flag_label(v, "tagged young", "not young") for v in ref["tagged_young"]]})
        gmeta["tagged_young"] = gmeta["tagged_young"].replace({"not stated": "age not stated"})
        if "migrant" in ref:
            gmeta["migrant"] = [flag_label(v, "migrant", "not migrant") for v in ref["migrant"]]
        cov = pd.concat([cov, pd.DataFrame([
            {"field": f, "column": "aquilacl.reference.load_birds", "birds": len(birds), "matched": np.nan,
             "with_value": int(ref[f].eq(True).sum() if f in ("tagged_young", "migrant") else ref[f].notna().sum()),
             "missing": int(len(birds) - (ref[f].eq(True).sum() if f in ("tagged_young", "migrant")
                                          else ref[f].notna().sum())), "conflict": 0}
            for f in ("tagged_young", "hatch_year", "migrant") if f in ref])], ignore_index=True)
        cov["matched"] = cov["matched"].astype("Int64")

    # M1
    loss = read_loss(args.run / "loss.csv")
    out.fig(fig_loss(loss), "m1_loss.png", "M1", "did training converge? (loss and learning rate)")
    out.fig(fig_cosine(loss), "m1_cosine.png", "M1", "are positives pulled together and negatives apart?")
    ev = flatten(json.loads((args.run / "evaluation.json").read_text()))
    evt = pd.DataFrame({"key": list(ev), "value": list(ev.values())})
    parts = evt["key"].str.rsplit(".", n=1, expand=True)
    evt["metric"] = parts[1]
    evt["section"] = parts[0].str.split(".").str[0]
    evt["target"] = parts[0].str.split(".", n=1).str[1].fillna("")
    probes = evt[evt["section"].isin(["hour_level", "window_level_holdout"])]
    out.table(probes[["section", "target", "metric", "value"]], "m1_probes.csv", "M1",
              "what does h encode? linear probes (hour level and held-out birds)", index=False)
    out.table(evt[evt["section"].isin(["invariance", "nuisance_vs_features", "cluster_stability_h"])]
              [["section", "target", "metric", "value"]], "m1_invariance.csv", "M1",
              "is h invariant to recording format (burst crop, GPS removal, sensitivity) and stable?", index=False)
    if (probes["section"] == "window_level_holdout").any():
        out.fig(fig_probes(probes), "m1_probes.png", "M1", "how well do held-out birds' behaviour targets and "
                "nuisances decode from h? (nuisance ideal = chance)")

    # M2
    sample = embedding_sample(df, args.map_per_bird, args.map_total, rng)
    xy, params = project_2d(load_h(args.run, sample[["bird", "row"]]), args.seed)
    params.update(per_bird_cap=min(args.map_per_bird, max(args.map_total // len(birds), 1)), total_cap=args.map_total,
                  windows=len(sample), birds=int(sample["bird"].nunique()), seed=args.seed)
    out.table(pd.DataFrame([params]), "m2_map_params.csv", "M2", "parameters of the display projection", index=False)
    out.fig(fig_embedding_map(sample, xy, colours, params["method"].split()[0].upper()), "m2_embedding_map.png", "M2",
            "how is h organised by state, activity, speed, time of day, season, sensitivity, recording regime and "
            "sex? (display only, no analysis is done on it)")

    # M3
    sel, chosen, jac = load_selection(args.states)
    if len(sel):
        out.table(sel, "m3_selection.csv", "M3", "stability, held-out ARI, presence and BIC per K", index=False)
        out.fig(fig_selection(sel, chosen), "m3_selection.png", "M3",
                f"how many states? (chosen K = {chosen}, dashed)")
    else:
        out.note("state_selection.json has no per-K entries in the expected form; m3_selection not produced")
    if jac is not None:
        out.table(jac.rename("jaccard").rename_axis("state").reset_index(), "m3_jaccard.csv", "M3",
                  "is each state reproducible under bootstrap?", index=False)
        out.fig(fig_jaccard(jac, colours), "m3_jaccard.png", "M3",
                f"is each state reproducible? (dashed: Jaccard {STABLE_JACCARD})")
    else:
        out.note("no per-state Jaccard in state_selection.json; stability of individual states not shown, and the "
                 "rare-state gallery uses the smallest states without a stability filter")
    for name in ("state_profiles.csv", "transitions.csv"):
        if (args.states / name).exists():
            shutil.copyfile(args.states / name, args.out / f"m4_5a_{name}")
            out.items.append((f"m4_5a_{name}", "M4/M5", f"stage 5a {name}, copied unchanged"))
    sel_raw = json.loads((args.states / "state_selection.json").read_text())
    if sel_raw.get("rule_met") is False:
        out.note(f"stage 5a: no K met the selection rule; K = {chosen} is the fallback ({sel_raw.get('rule', '')})")
    if (args.states / "state_profiles.csv").exists():
        p5 = pd.read_csv(args.states / "state_profiles.csv", index_col=0)
        for f in (c for c in p5 if c.startswith("flag_")):
            hit = p5.index[p5[f].astype(str).str.lower() == "true"].tolist()
            if hit:
                out.note(f"stage 5a artifact check {f} is set for state(s) {hit} (state_hmm numbering)")

    # M4
    prof = state_profile_table(df, states)
    out.table(prof, "m4_state_profile.csv", "M4", "feature medians/IQR, coverage and format shares per state")
    out.fig(fig_state_profile(prof), "m4_state_profile.png", "M4", "what does each state look like in features?")
    out.fig(fig_state_features(df, states, colours), "m4_state_features.png", "M4",
            "distribution of each feature per state")
    conf = label_confusion(df, states)
    out.table(conf, "m4_state_label_confusion.csv", "M4", "states x reference labels (counts over known labels)", index=False)
    if len(conf):
        out.fig(fig_state_labels(conf), "m4_state_labels.png", "M4", "which reference behaviours fall in each state?")
    out.fig(fig_state_format(prof, df, colours), "m4_state_format.png", "M4",
            "is a state a recording format (sensitivity, full/non-full, GPS) or a few birds rather than a behaviour?")
    for s in states:
        if not (df["state"] == s).any():
            out.note(f"state {s} has no windows in the analysis column ({args.state_col}); no card")
            continue
        out.fig(fig_state_card(df, args.windows, s, prof, colours, rng), f"m4_state_cards_{s}.png", "M4",
                f"state {s}: example acceleration traces (8 windows, distinct birds where possible)")

    # M5
    nsolar = int(df["local_solar_hour"].isna().sum())
    df["hour_bin"] = np.floor(df["local_solar_hour"])
    out.fig(share_grid(df.dropna(subset=["hour_bin"]), "hour_bin", 1, "year", "local solar hour (h)",
                       f"diel state use per year ({nsolar} windows without solar time left out)", states, colours, (0, 24)),
            "m5_diel_solar_hour.png", "M5", "diel rhythm of state use in local solar time, per year, with coverage")
    df["sun_bin"] = np.floor(df["sun_elev_deg"] / 5) * 5
    out.fig(share_grid(df.dropna(subset=["sun_bin"]), "sun_bin", 5, "year", "sun elevation (deg)",
                       "state use vs sun elevation per year", states, colours),
            "m5_diel_sun_elevation.png", "M5", "state use vs sun elevation (day/twilight/night), per year, with coverage")
    df["doy_bin"] = (df["doy"] - 1) // 7 * 7 + 1
    out.fig(share_grid(df, "doy_bin", 7, "year", "day of year (7-day bins, UTC)", "seasonal state use per year",
                       states, colours, (1, 367)),
            "m5_seasonal.png", "M5", "seasonal change of state use, per year, with coverage")
    smode, mode_w = seasonal_by_mode(df, states)
    out.table(smode, "m5_seasonal_by_mode.csv", "M5", "monthly bird-weighted state shares (years pooled): all windows, "
              "by recording mode (k=1/k=0), by GPS present/absent, by both, and mode-adjusted", index=False)
    out.fig(fig_seasonal_by_mode(smode, states, colours), "m5_seasonal_by_mode.png", "M5",
            "which seasonal differences remain within one recording mode? (k=1 continuous recording is concentrated "
            "in April-August)")
    contrast = seasonal_mode_contrast(smode, states)
    out.table(contrast, "m5_seasonal_by_mode_contrast.csv", "M5", "per state: share in the months with k=1 data "
              "minus the other months, all windows vs within k=0 only (does the seasonal difference survive within "
              "one mode?)", index=False)
    for n in seasonal_mode_note(smode, df, states, mode_w, contrast):
        out.note(n)
    col = args.state_col
    runs, same_seg, seq = runs_table(df, "state")
    counts = transition_matrix(same_seg, seq, states)
    out.table(counts, "m5_transitions.csv", "M5", f"counts of {col} transitions between consecutive windows")
    out.fig(fig_transitions(counts), "m5_transitions.png", "M5", f"which states follow which ({col})?")
    dw = runs.groupby(["state", "censored"])["duration_s"].describe(percentiles=[0.5, 0.9]).reset_index()
    out.table(dw, "m5_dwell.csv", "M5", f"run durations per state ({col}), complete vs censored", index=False)
    out.fig(fig_dwell(runs, states, colours), "m5_dwell.png", "M5", f"how long does a bird stay in a state ({col})?")

    # M6
    budget = per_bird_budget(df, states)
    out.table(budget, "m6_bird_budgets.csv", "M6", "per-bird state budget")
    out.fig(fig_bird_budgets(budget, states), "m6_bird_budgets.png", "M6",
            "how does state use differ between birds; is a state confined to some birds?")
    out.fig(fig_calendar(df, colours), "m6_calendar.png", "M6", "state sequences per bird in calendar time, gaps marked")
    summ = bird_summary(df, budget, states, meta, rng)
    if age:
        summ = summ.join(ref[[c for c in ("tagged_young", "hatch_year", "migrant") if c in ref]])
    out.table(summ, "m6_bird_summary.csv", "M6", "per-bird coverage, diversity and metadata")
    out.fig(fig_bird_diversity(summ), "m6_bird_diversity.png", "M6",
            "how variable are birds' budgets, independent of coverage?")
    out.table(cov, "m6_metadata_coverage.csv", "M6", "which birds have sex, life stage and deployment date "
              "in the reference data (missing and conflicting values are counted, never filled)", index=False)
    if age:
        fields, desc = ["sex", "tagged_young"], {
            "tagged_young": "'age not stated' is not 'adult'; it mixes adults with birds whose age is unrecorded"}
        if "migrant" in gmeta and gmeta["migrant"].eq("migrant").any():
            fields.append("migrant")
            if not gmeta["migrant"].eq("not migrant").any():
                desc["migrant"] = "'not stated' is not 'resident'"
        elif "migrant" in gmeta:
            out.note("migrant: no bird has the migrant flag in aquilacl.reference; no migrant test")
        gm = meta.drop(columns=["life_stage"]).merge(gmeta, on="bird", how="left")
        gt = group_tests(budget, gm, states, args.permutations, rng, out, colours, fields, desc)
        q = ("do state budgets differ by sex, tagged young vs age not stated (descriptive) or migrant flag? "
             "(bird-level permutation test; not adjusted for cohort, coverage or tracking duration)")
    else:
        gt = group_tests(budget, meta, states, args.permutations, rng, out, colours)
        q = ("do state budgets differ by sex or life stage? (bird-level permutation test; not adjusted for cohort, "
             "coverage or tracking duration)")
    out.table(gt, "m6_group_tests.csv", "M6", q, index=False)
    if df["sex"].notna().any():
        out.fig(share_grid(df.dropna(subset=["hour_bin", "sex"]), "hour_bin", 1, "sex", "local solar hour (h)",
                           "diel state use by sex (all years)", states, colours, (0, 24)),
                "m6_sex_diel.png", "M6", "does diel state use differ by sex? (bird-weighted, with coverage)")
        out.fig(share_grid(df.dropna(subset=["sex"]), "doy_bin", 7, "sex", "day of year (7-day bins, UTC)",
                           "seasonal state use by sex (years pooled)", states, colours, (1, 367)),
                "m6_sex_seasonal.png", "M6", "does seasonal state use differ by sex? (years pooled, with coverage)")
    m = meta.set_index("bird")
    if age:
        dev, xlabel = development_table(df, ref, reference, states, out, fledges)
        if len(dev):
            share, nb, nw = bird_share(dev, "dev_month", states)
            dt = share.add_prefix("share_state_").assign(n_birds=nb, n_windows=nw.reindex(share.index))
            dt.index.name = xlabel.replace(" ", "_")
            out.table(dt, "m6_development.csv", "M6", f"tagged-young birds: bird-weighted state share by {xlabel} "
                      "(pre_fledge windows left out), birds and windows per month")
            out.fig(share_grid(dev, "dev_month", 1, "group", xlabel, f"tagged-young birds: state use by {xlabel} "
                               "(pre_fledge left out; line = birds in the share)", states, colours),
                    "m6_development.png", "M6", f"how does state use of tagged-young birds change with {xlabel}?")
        else:
            out.note("m6_development not produced: no tagged-young bird has windows after pre_fledge")
        pairs, ysum = y1_vs_y2(df, ref, states, args.permutations, rng)
        if len(pairs):
            out.table(pairs, "m6_age_y1_vs_y2.csv", "M6", "tagged-young birds: state shares in the same calendar "
                      "month in year 1 and year 2, with each side's k=1 and GPS share", index=False)
            out.table(ysum, "m6_age_y1_vs_y2_summary.csv", "M6", "year 2 minus year 1, paired within bird and "
                      "calendar month, bird-level sign-flip test", index=False)
            out.fig(fig_y1_vs_y2(pairs, ysum, states, colours), "m6_age_y1_vs_y2.png", "M6",
                    "does a tagged-young bird use states differently in its second year than in the same months "
                    "of its first year?")
        else:
            out.note(f"m6_age_y1_vs_y2 not produced: no tagged-young bird has >= {MIN_BIN_WINDOWS} windows in the "
                     "same calendar month in both year 1 and year 2")
    elif m["life_stage"].notna().any() and m["deploy_on"].notna().any():
        dep = df["bird"].astype(str).map(m["deploy_on"])
        df["life_stage"] = df["bird"].astype(str).map(m["life_stage"])
        df["months_bin"] = np.floor((df["t_start"].dt.tz_localize(None) - dep).dt.days / 30.44)
        out.fig(share_grid(df.dropna(subset=["months_bin", "life_stage"]), "months_bin", 1, "life_stage",
                           "months since deployment", "state use vs time since deployment by life stage at tagging",
                           states, colours),
                "m6_development.png", "M6", "does state use of birds tagged young change towards that of older birds?")
    else:
        out.note("m6_development.png not produced: no bird has both a life stage and a deployment date")

    # M7
    share = prof["share_windows"]
    cand = share[jac.reindex(share.index) >= STABLE_JACCARD] if jac is not None else share
    rare = cand.sort_values().index[:3].tolist()
    if jac is not None and not rare:
        rare = share.sort_values().index[:3].tolist()
        out.note(f"no state has Jaccard >= {STABLE_JACCARD}; m7_rare_states shows the smallest states unfiltered")
        jac = None
    ex = pd.concat([pick_examples(df, (df["state"] == s).to_numpy(), 6, rng) for s in rare]) if rare else pd.DataFrame()
    if len(ex):
        out.fig(fig_gallery(args.windows, ex, f"smallest {'stable ' if jac is not None else ''}states: {rare} "
                            f"(shares {', '.join(f'{share[s]:.2%}' for s in rare)})"),
                "m7_rare_states.png", "M7", "what do the rarest reproducible states look like?")
    scored = knn_scores(df, args.run, rng)
    outl = top_outliers(scored)
    if len(outl):
        unrel = unreliable_calibration_birds(args.windows)
        out.table(outlier_factors(scored, outl, unreliable_birds=unrel), "m7_outlier_factors.csv", "M7",
                  "are outliers over-represented among short windows, k=1 (continuous mode) or GPS windows? "
                  "(ratio = share among outliers / share among scored windows)", index=False)
        out.note("m7_outlier_factors: the baseline is the scored sample (up to 300 windows per bird x month x "
                 "regime), not all windows. Scores are relative to the median of the window's own bird x month x "
                 "regime group, so short windows are compared with short windows and their ratio is partly fixed "
                 "by that design. Calibration: windows.py leaves out every burst of a calibration marked "
                 "unreliable, so no window comes from an unreliable period and a period-level flag cannot exist; "
                 + ("the bird-level factor (bird had such bursts left out, which can leave holes in its windows) "
                    f"comes from windows_summary.csv ({len(unrel)} birds)" if unrel is not None else
                    "windows_summary.csv with bursts_unreliable_calibration is not in the windows directory, so "
                    "no calibration factor is reported"))
        out.table(outl, "m7_outliers.csv", "M7", "most isolated windows in h within bird x month x regime "
                  "(format columns included to separate recording artefacts from behaviour)", index=False)
        out.fig(fig_gallery(args.windows, outl, "kNN-distance outliers (score = 10th-neighbour distance / group "
                            "median)", "knn_score"),
                "m7_outliers.png", "M7", "what do unusual windows look like (behaviour or recording artefact)?")

    # M8
    cat = df[[c for c in CATALOGUE_COLS if c in df]].rename(columns={"state_gmm": "state"})
    out.table(cat, "m8_window_catalogue.csv.gz", "M8", "one row per window: bird, time, state, key features",
              index=False, chunksize=500_000, compression="gzip")

    header = [f"- states: `{args.states}`", f"- windows: `{args.windows}`", f"- run: `{args.run}`",
              f"- reference: `{args.reference}` (only the id, sex, life-stage and deployment-date columns are read"
              + ("; age, tagged-young, hatch-year and migrant fields come from aquilacl.reference)" if age else ")"),
              f"- {len(df)} windows, {len(birds)} birds, {len(states)} states; seed {args.seed}",
              f"- analysis state = `{args.state_col}` of states.parquet (transitions, dwell, budgets, cards); "
              "per-state Jaccard (M3) refers to the unsmoothed GMM `state` with the same number",
              "- dates and day of year are UTC; local solar hour comes from stage 3 (nearest fix within 12 h)",
              f"- full window = >= {FULL_SAMPLES} of 200 acceleration samples; windows are {WINDOW_S:.0f} s",
              f"- temporal shares are bird-weighted: each bird with >= {MIN_BIN_WINDOWS} windows in a bin counts once",
              f"- per-bird comparisons use birds with >= {MIN_BIRD_WINDOWS} windows",
              "- no latitude, longitude or maps are read or shown"]
    first_draft = [(a, q, AGE_FIRST_DRAFT.get(a, b) if age else b) for a, q, b in FIRST_DRAFT]
    out.write_index(header, first_draft)
    print(f"wrote {len(out.items)} outputs to {args.out}")


if __name__ == "__main__":
    main()
