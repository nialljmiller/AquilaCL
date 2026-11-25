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
    "dom_freq_hz": "dominant frequency (Hz)", "pitch_deg": "pitch (deg)", "roll_deg": "roll (deg)",
    "speed_mean": "ground speed (m/s)", "climb_mean": "climb rate (m/s)", "turn_abs_mean": "|turn rate| (deg/s)",
}
CATALOGUE_COLS = ["bird", "t_start", "state_gmm", "state_hmm", "p_state", "vedba", "flap_band_frac", "pitch_deg",
                  "speed_mean", "climb_mean", "turn_abs_mean", "sun_elev_deg", "local_solar_hour", "sensitivity_k",
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
    ("anomaly.png", "unusual windows", "m7_outliers.png, m7_outliers.csv, m7_rare_states.png"),
    ("temporal_doy.png, behaviour_seasonal.png", "seasonal change", "m5_seasonal.png"),
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

    def write_index(self, header: list[str]) -> None:
        lines = ["# aquilacl stage 5b analysis", "", *header, "", "## Outputs", "",
                 "| File | Module | Question |", "|---|---|---|"]
        lines += [f"| [{n}]({n}) | {m} | {q} |" for n, m, q in self.items]
        lines += ["", "## First-draft plots and where their questions are answered", "",
                  "| First-draft output | Question | Answered by |", "|---|---|---|"]
        lines += [f"| {a} | {q} | {self.answered(b)} |" for a, q, b in FIRST_DRAFT]
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


def load_windows(states_dir: Path, windows: Path, out: Outputs, state_col: str = "state_hmm") -> pd.DataFrame:
    """One row per window with state, metadata, features and reference labels (no positions).

    ``state`` is the analysis state (``state_col`` of states.parquet); the unsmoothed GMM state is ``state_gmm``.
    """
    sp = states_dir / "states.parquet"
    st = _read(sp, ["bird", "row", "state", "state_hmm", "p_state"])
    st["bird"] = st["bird"].astype(str)
    by_bird = dict(tuple(st.groupby("bird", sort=False)))
    frames = []
    for d in bird_dirs(windows):
        m = _read(d / "meta.parquet", META_COLS)
        fcols = list(FEATURES) + [c for c in pq.read_schema(d / "features.parquet").names if c.startswith("label_")]
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
    for c in ("state", "state_gmm", "state_hmm"):
        if c in df:
            df[c] = df[c].astype(int)
    return df


def load_reference(path: Path, birds: list[str], out: Outputs) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    ref["id"] = ref["id"].str.strip()
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
    if cols["life_stage"] is None:
        out.note("the reference CSV has no life-stage column; life stage is not inferred from comments, dates or "
                 "movement, so no life-stage comparison is made")
    return meta, cov


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
                      "median_local_solar_hour": g["local_solar_hour"].median()})
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
    """Up to n windows from distinct birds where possible, among the confident half (p_state >= median)."""
    sub = df[mask]
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


def group_tests(budget, meta, states, n_perm, rng, out: Outputs, colours):
    rows = []
    b = budget[budget["windows"] >= MIN_BIRD_WINDOWS].join(meta.set_index("bird"))
    for field in ("sex", "life_stage"):
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
        rows += [{**base, **r, "status": f"tested, {len(sub)} birds, {n_perm} permutations"} for r in t.to_dict("records")]
        out.fig(fig_group(sub, field, states, colours), f"m6_group_{field}.png", "M6",
                f"does per-bird state use differ by {field.replace('_', ' ')}? (bird level, n per group in legend)")
    return pd.DataFrame(rows)


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


def knn_outliers(df: pd.DataFrame, run: Path, rng, per_group=300, min_group=30, k=10, top=24, per_bird=3):
    """kNN distance in standardised h within bird x month x recording regime, relative to the group median."""
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
    sample = sample.assign(knn_score=score).sort_values("knn_score", ascending=False)
    sample = sample.groupby("bird", observed=True).head(per_bird).head(top)
    cols = ["bird", "row", "t_start", "month", "regime", "state", "knn_score", "acc_samples", "sensitivity_k",
            "gps_fixes", "vedba", "speed_mean", "local_solar_hour"]
    return sample[[c for c in cols if c in sample]]


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
    meta, cov = load_reference(args.reference, birds, out)
    df["sex"] = df["bird"].astype(str).map(meta.set_index("bird")["sex"])

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
    out.table(summ, "m6_bird_summary.csv", "M6", "per-bird coverage, diversity and metadata")
    out.fig(fig_bird_diversity(summ), "m6_bird_diversity.png", "M6",
            "how variable are birds' budgets, independent of coverage?")
    out.table(cov, "m6_metadata_coverage.csv", "M6", "which birds have sex, life stage and deployment date "
              "in the reference data (missing and conflicting values are counted, never filled)", index=False)
    out.table(group_tests(budget, meta, states, args.permutations, rng, out, colours), "m6_group_tests.csv", "M6",
              "do state budgets differ by sex or life stage? (bird-level permutation test; not adjusted for cohort, "
              "coverage or tracking duration)", index=False)
    if df["sex"].notna().any():
        out.fig(share_grid(df.dropna(subset=["hour_bin", "sex"]), "hour_bin", 1, "sex", "local solar hour (h)",
                           "diel state use by sex (all years)", states, colours, (0, 24)),
                "m6_sex_diel.png", "M6", "does diel state use differ by sex? (bird-weighted, with coverage)")
        out.fig(share_grid(df.dropna(subset=["sex"]), "doy_bin", 7, "sex", "day of year (7-day bins, UTC)",
                           "seasonal state use by sex (years pooled)", states, colours, (1, 367)),
                "m6_sex_seasonal.png", "M6", "does seasonal state use differ by sex? (years pooled, with coverage)")
    m = meta.set_index("bird")
    if m["life_stage"].notna().any() and m["deploy_on"].notna().any():
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
    outl = knn_outliers(df, args.run, rng)
    if len(outl):
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
              f"- reference: `{args.reference}` (only the id, sex, life-stage and deployment-date columns are read)",
              f"- {len(df)} windows, {len(birds)} birds, {len(states)} states; seed {args.seed}",
              f"- analysis state = `{args.state_col}` of states.parquet (transitions, dwell, budgets, cards); "
              "per-state Jaccard (M3) refers to the unsmoothed GMM `state` with the same number",
              "- dates and day of year are UTC; local solar hour comes from stage 3 (nearest fix within 12 h)",
              f"- full window = >= {FULL_SAMPLES} of 200 acceleration samples; windows are {WINDOW_S:.0f} s",
              f"- temporal shares are bird-weighted: each bird with >= {MIN_BIN_WINDOWS} windows in a bin counts once",
              f"- per-bird comparisons use birds with >= {MIN_BIRD_WINDOWS} windows",
              "- no latitude, longitude or maps are read or shown"]
    out.write_index(header)
    print(f"wrote {len(out.items)} outputs to {args.out}")


if __name__ == "__main__":
    main()
