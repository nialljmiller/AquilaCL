"""Per-bird reference data (sex, tagged-young status, deployment, tag) and an age model for tagged-young birds.

Source: the Movebank reference-data CSV. Only the columns in COLUMNS are read. Comments (which contain colour-band
text) are searched for fixed whole-word terms and never written anywhere; only the matched, normalised term is
kept.

Rules
- Bird id: 'animal-id' may be written "<ring id>; <tag id>"; the bird id is the part before ';', stripped.
- Several rows per bird (re-deployments): exact duplicate rows are dropped; ``n_deployments`` counts the rest.
  Sex values that disagree between rows give NaN (counted in ``attrs['sex_conflicts']``). tag_model and
  tag_firmware: the only non-empty value, or 'conflict' if several distinct ones occur.
- tagged_young: True if any 'animal-comments' or 'deployment-comments' of the bird contains one of YOUNG_TERMS
  (case-insensitive, whole word). Otherwise NaN ("not stated"); never False or "adult" by default.
  young_term: the normalised matched term (first in YOUNG_TERMS order); hatch_year: deploy-on year of the earliest
  matching deployment, as given (literal rule; ``attrs['young_deployed_jan_apr']`` counts deployments before the
  assumed hatch date, where the rule is doubtful).
- migrant: True if a comment contains the whole word 'migrant' not preceded by 'non-'/'non '; NaN otherwise.
  Mentions of 'non-migrant' are counted (``attrs['non_migrant_mentions']``), not turned into False.

Age model (assumptions, also returned by :func:`assumptions`)
- Hatch date = HATCH_MONTH_DAY of hatch_year; default fledge date = FLEDGE_MONTH_DAY of hatch_year.
- Data-driven fledge (:func:`fledge_from_speed`): first UTC day after the hatch date with >= FLEDGE_MIN_WINDOWS
  windows of GPS speed > FLEDGE_SPEED_MS, if that day is before FLEDGE_DEADLINE_MONTH_DAY; else the default.
  For birds tagged after fledging the detected day is often the first day of data (an upper bound).
- Age classes: 'pre_fledge' (before fledge), 'juvenile_y1' (fledge to the next hatch date), 'y2', 'y3', 'y4plus'
  (each later year starts on the hatch date); 'not_stated' for birds without tagged_young.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HATCH_MONTH_DAY = (5, 1)  # assumed hatch date: 1 May of hatch_year
FLEDGE_MONTH_DAY = (7, 15)  # default fledge date: 15 July of hatch_year
FLEDGE_DEADLINE_MONTH_DAY = (9, 30)  # a data-driven fledge must fall before 30 September
FLEDGE_MIN_WINDOWS = 30
FLEDGE_SPEED_MS = 5.0
DAYS_PER_MONTH = 30.4375

COLUMNS = {  # field: accepted column names (first present wins)
    "id": ["animal-id", "individual-local-identifier"],
    "tag_id": ["tag-id", "tag-local-identifier"],
    "deploy_on": ["deploy-on-date", "deploy-on-timestamp"],
    "deploy_off": ["deploy-off-date", "deploy-off-timestamp"],
    "sex": ["animal-sex", "sex"],
    "life_stage": ["animal-life-stage"],
    "animal_comments": ["animal-comments"],
    "deployment_comments": ["deployment-comments"],
    "tag_model": ["tag-model"],
    "tag_firmware": ["tag-firmware"],
}
COMMENT_FIELDS = ["animal_comments", "deployment_comments"]
MISSING = {"", "u", "unknown", "nan", "none", "na", "n/a"}
# normalised term: whole-word pattern (case-insensitive). Order = priority when several match.
YOUNG_TERMS = {
    "nestling": r"nestlings?",
    "fledgling": r"fledglings?",
    "chick": r"chicks?",
    "hatch_year": r"hatch[- ]?years?|HY",
    "juvenile": r"juveniles?",
}
_YOUNG_RE = {k: re.compile(rf"(?<![A-Za-z0-9])(?:{v})(?![A-Za-z0-9])", re.IGNORECASE) for k, v in YOUNG_TERMS.items()}
_MIGRANT_RE = re.compile(r"(?<![A-Za-z0-9])migrants?(?![A-Za-z0-9])", re.IGNORECASE)
_NON_MIGRANT_RE = re.compile(r"(?<![A-Za-z0-9])non[- ]?migrants?(?![A-Za-z0-9])", re.IGNORECASE)
AGE_CLASSES = ["pre_fledge", "juvenile_y1", "y2", "y3", "y4plus", "not_stated"]


def assumptions() -> dict:
    return {"hatch_date": f"{HATCH_MONTH_DAY[1]} {_month(HATCH_MONTH_DAY[0])} of hatch_year",
            "fledge_default": f"{FLEDGE_MONTH_DAY[1]} {_month(FLEDGE_MONTH_DAY[0])} of hatch_year",
            "fledge_data_rule": (f"first UTC day after the hatch date with >= {FLEDGE_MIN_WINDOWS} windows of GPS "
                                 f"speed_mean > {FLEDGE_SPEED_MS} m/s, used if before {FLEDGE_DEADLINE_MONTH_DAY[1]} "
                                 f"{_month(FLEDGE_DEADLINE_MONTH_DAY[0])} of hatch_year, else the default"),
            "hatch_year_rule": "deploy-on year of the earliest deployment whose comments contain a young term",
            "young_terms": list(YOUNG_TERMS),
            "age_classes": "pre_fledge < fledge <= juvenile_y1 < hatch date of hatch_year+1 <= y2 < +2 <= y3 < +3 <= y4plus"}


def _month(m: int) -> str:
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]


def young_term(text: str) -> str | None:
    """The normalised young term found in ``text`` (first in YOUNG_TERMS order), or None."""
    for k, rx in _YOUNG_RE.items():
        if rx.search(text or ""):
            return k
    return None


def is_migrant(text: str) -> bool:
    """'migrant' as a whole word, not as part of 'non-migrant' / 'non migrant' / 'nonmigrant'."""
    return bool(_MIGRANT_RE.search(_NON_MIGRANT_RE.sub(" ", text or "")))


def bird_id(animal_id: pd.Series) -> pd.Series:
    return animal_id.astype(str).str.split(";").str[0].str.strip()


def _single(s: pd.Series):
    v = s[s.notna()].unique()
    return v[0] if len(v) == 1 else ("conflict" if len(v) else np.nan)


def load_birds(path: Path, birds: list[str]) -> pd.DataFrame:
    """One row per bird in ``birds`` (see module docstring). Counts go to ``attrs``."""
    head = pd.read_csv(path, nrows=0).columns
    cols = {f: next((c for c in names if c in head), None) for f, names in COLUMNS.items()}
    if cols["id"] is None:
        raise ValueError(f"reference CSV has none of the id columns {COLUMNS['id']}")
    ref = pd.read_csv(path, usecols=[c for c in cols.values() if c], dtype=str, keep_default_na=False)
    ref = ref.rename(columns={c: f for f, c in cols.items() if c})
    for f in cols:
        if f not in ref:
            ref[f] = ""
    n_rows = len(ref)
    ref = ref.drop_duplicates().reset_index(drop=True)
    ref["bird"] = bird_id(ref["id"])
    for f in ("sex", "tag_model", "tag_firmware", "life_stage"):
        v = ref[f].str.strip()
        ref[f] = v.where(~v.str.lower().isin(MISSING))
    ref["sex"] = ref["sex"].str.lower().replace({"male": "m", "female": "f"})
    ref.loc[~ref["sex"].isin(["m", "f"]) & ref["sex"].notna(), "sex"] = np.nan
    for f in ("deploy_on", "deploy_off"):
        ref[f] = pd.to_datetime(ref[f].str.strip(), errors="coerce", format="mixed", utc=True)

    hits = {k: 0 for k in YOUNG_TERMS}
    col_hits = {c: 0 for c in COMMENT_FIELDS}
    terms, migr, nonmigr = [], [], 0
    for _, r in ref.iterrows():
        t = None
        for c in COMMENT_FIELDS:
            tc = young_term(r[c])
            if tc is not None:
                col_hits[c] += 1
                t = t or tc
        if t is not None:
            hits[t] += 1
        terms.append(t)
        text = " ".join(r[c] for c in COMMENT_FIELDS)
        migr.append(is_migrant(text))
        nonmigr += bool(_NON_MIGRANT_RE.search(text))
    ref["young_term"] = terms
    ref["migrant"] = migr

    rows = []
    sex_conflicts = 0
    for b in birds:
        g = ref[ref["bird"] == b]
        row = {"bird": b, "n_deployments": len(g)}
        sx = g["sex"].dropna().unique()
        sex_conflicts += len(sx) > 1
        row["sex"] = sx[0] if len(sx) == 1 else np.nan
        row["deploy_on"] = g["deploy_on"].min() if len(g) else pd.NaT
        y = g[g["young_term"].notna()].sort_values("deploy_on")
        row["tagged_young"] = True if len(y) else np.nan
        row["young_term"] = y["young_term"].iat[0] if len(y) else np.nan
        row["hatch_year"] = float(y["deploy_on"].iat[0].year) if len(y) and pd.notna(y["deploy_on"].iat[0]) else np.nan
        row["young_deploy_on"] = y["deploy_on"].iat[0] if len(y) else pd.NaT
        row["migrant"] = True if g["migrant"].any() else np.nan
        row["tag_model"] = _single(g["tag_model"])
        row["tag_firmware"] = _single(g["tag_firmware"])
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("tagged_young", "migrant"):
        out[c] = out[c].astype(object)
    yd = out["young_deploy_on"]
    out.attrs = {
        "reference_rows": n_rows, "reference_rows_after_dedupe": len(ref), "reference_birds": int(ref["bird"].nunique()),
        "birds_requested": len(birds), "birds_unmatched": int((out["n_deployments"] == 0).sum()),
        "sex_conflicts": int(sex_conflicts), "sex_m": int((out["sex"] == "m").sum()),
        "sex_f": int((out["sex"] == "f").sum()), "sex_missing": int(out["sex"].isna().sum()),
        "tagged_young": int(out["tagged_young"].notna().sum()),
        "young_term_hits_rows": hits, "young_term_hits_by_column_rows": col_hits,
        "young_deployed_jan_apr": int(((yd.dt.month < HATCH_MONTH_DAY[0]) & yd.notna()).sum()),
        "migrant": int(out["migrant"].notna().sum()), "non_migrant_mentions_rows": int(nonmigr),
        "life_stage_non_empty_rows": int(ref["life_stage"].notna().sum()),
        "columns_used": {f: c for f, c in cols.items() if c},
    }
    return out.drop(columns="young_deploy_on")


def _date(year, md) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=md[0], day=md[1], tz="UTC")


def fledge_from_speed(t_start, speed_mean, hatch_year) -> dict:
    """Data-driven fledge date of a tagged-young bird from per-window (t_start, speed_mean). See module docstring."""
    default = _date(hatch_year, FLEDGE_MONTH_DAY)
    t = pd.to_datetime(pd.Series(t_start), utc=True).reset_index(drop=True)
    sp = pd.Series(np.asarray(speed_mean, float))
    hatch, deadline = _date(hatch_year, HATCH_MONTH_DAY), _date(hatch_year, FLEDGE_DEADLINE_MONTH_DAY)
    res = {"fledge": default, "fledge_source": "default", "fledge_is_first_data_day": False}
    after = t >= hatch
    if not after.any():
        return res
    day = t.dt.floor("D")
    fast = (sp > FLEDGE_SPEED_MS) & after
    n = fast.groupby(day).sum()
    ok = n[n >= FLEDGE_MIN_WINDOWS]
    if len(ok) and ok.index[0] < deadline:
        res["fledge"] = ok.index[0]
        res["fledge_source"] = "data"
        res["fledge_is_first_data_day"] = bool(ok.index[0] == day[after].min())
    return res


def age_on(bird_row, timestamps, fledge: pd.Timestamp | None = None) -> pd.DataFrame:
    """Age in days since the assumed hatch date, months of age and age class for each timestamp.

    ``bird_row`` needs tagged_young and hatch_year (a row of :func:`load_birds`). ``fledge`` overrides the default
    fledge date (e.g. from :func:`fledge_from_speed`).
    """
    t = pd.to_datetime(pd.Series(timestamps), utc=True).reset_index(drop=True)
    ty = bird_row.get("tagged_young")
    young = bool(pd.notna(ty) and ty) and pd.notna(bird_row.get("hatch_year"))
    if not young:
        return pd.DataFrame({"age_days": np.full(len(t), np.nan), "age_months": np.full(len(t), np.nan),
                             "age_class": np.full(len(t), "not_stated", dtype=object)})
    hy = int(bird_row["hatch_year"])
    hatch = _date(hy, HATCH_MONTH_DAY)
    fl = fledge if fledge is not None else _date(hy, FLEDGE_MONTH_DAY)
    age = ((t - hatch).dt.total_seconds() / 86400.0).to_numpy()
    cls = np.full(len(t), "y4plus", dtype=object)
    for yrs, name in ((3, "y3"), (2, "y2"), (1, "juvenile_y1")):
        cls[(t < _date(hy + yrs, HATCH_MONTH_DAY)).to_numpy()] = name
    cls[(t < fl).to_numpy()] = "pre_fledge"
    return pd.DataFrame({"age_days": age, "age_months": np.floor(age / DAYS_PER_MONTH), "age_class": cls})
