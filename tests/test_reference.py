import numpy as np
import pandas as pd
import pytest

from aquilacl import reference

COLS = ["tag-id", "animal-id", "deploy-on-date", "deploy-off-date", "animal-comments", "animal-sex",
        "animal-life-stage", "deployment-comments", "tag-model", "tag-firmware"]


def _csv(tmp_path, rows):
    p = tmp_path / "ref (1).csv"
    pd.DataFrame(rows, columns=COLS).to_csv(p, index=False)
    return p


ROWS = [  # synthetic ids only
    ["1", "B01; 1", "2024-06-10 00:00:00.000", "", "", "m", "", "Tagged as a nestling. Band text XY", "M1", "fw1"],
    ["1", "B01; 1", "2024-06-10 00:00:00.000", "", "", "m", "", "Tagged as a nestling. Band text XY", "M1", "fw1"],
    ["2", "B01; 2", "2025-03-01 00:00:00.000", "", "", "male", "", "", "M1", "fw2"],
    ["3", "B02; 3", "2024-08-01 00:00:00.000", "", "migrant", "f", "", "HY bird", "M1", ""],
    ["4", "B03; 4", "2025-07-01 00:00:00.000", "", "non-migrant", "m", "", "harness fitted; hyperactive", "M2", "fw1"],
    ["5", "B03; 5", "2025-09-01 00:00:00.000", "", "", "f", "", "", "", ""],
    ["6", "B04", "2025-06-01 00:00:00.000", "", "Chick", "", "", "", "M2", "fw1"],
    ["7", "B05; 7", "2025-01-15 00:00:00.000", "", "adult female", "f", "", "", "M2", "fw1"],
]


def test_load_birds(tmp_path):
    out = reference.load_birds(_csv(tmp_path, ROWS), ["B01", "B02", "B03", "B04", "B05", "B06"]).set_index("bird")
    exp = {  # bird: sex, tagged_young, young_term, hatch_year, n_deployments, migrant, tag_firmware
        "B01": ("m", True, "nestling", 2024.0, 2, None, "conflict"),
        "B02": ("f", True, "hatch_year", 2024.0, 1, True, None),
        "B03": (None, None, None, None, 2, None, "fw1"),
        "B04": (None, True, "chick", 2025.0, 1, None, "fw1"),
        "B05": ("f", None, None, None, 1, None, "fw1"),
        "B06": (None, None, None, None, 0, None, None),
    }
    for b, (sex, young, term, hy, nd, migr, fw) in exp.items():
        r = out.loc[b]
        assert (r["sex"] == sex) if sex else pd.isna(r["sex"]), b
        assert (r["tagged_young"] is True) if young else pd.isna(r["tagged_young"]), b  # never False
        assert (r["young_term"] == term) if term else pd.isna(r["young_term"]), b
        assert (r["hatch_year"] == hy) if hy else pd.isna(r["hatch_year"]), b
        assert r["n_deployments"] == nd, b
        assert (r["migrant"] is True) if migr else pd.isna(r["migrant"]), b
        assert (r["tag_firmware"] == fw) if fw else pd.isna(r["tag_firmware"]), b
    assert out.loc["B01", "deploy_on"] == pd.Timestamp("2024-06-10", tz="UTC")  # earliest deployment
    a = out.attrs
    assert a["sex_conflicts"] == 1 and a["birds_unmatched"] == 1 and a["reference_rows_after_dedupe"] == 7
    assert a["non_migrant_mentions_rows"] == 1 and a["tagged_young"] == 3
    assert a["young_term_hits_by_column_rows"] == {"animal_comments": 1, "deployment_comments": 2}
    text = out.astype(str).to_string()
    for s in ("Band text", "harness", "adult", "Chick", "HY bird"):  # comment text is never emitted
        assert s not in text


@pytest.mark.parametrize("text, term", [
    ("Tagged as a nestling.", "nestling"), ("HY", "hatch_year"), ("Chick", "chick"), ("hy", "hatch_year"),
    ("hatch-year female", "hatch_year"), ("Hatch year", "hatch_year"), ("two juveniles", "juvenile"),
    ("FLEDGLING", "fledgling"), ("nestlings, chicks", "nestling"), ("hyperactive", None), ("harness", None),
    ("why", None), ("Thy", None), ("chickadee", None), ("adult", None), ("", None), (None, None),
])
def test_young_term(text, term):
    assert reference.young_term(text) == term


@pytest.mark.parametrize("text, hit", [("migrant", True), ("Migrants", True), ("non-migrant", False),
                                       ("non migrant", False), ("immigrant", False), ("", False)])
def test_migrant(text, hit):
    assert reference.is_migrant(text) == hit


@pytest.mark.parametrize("t, cls", [
    ("2024-07-01", "pre_fledge"), ("2024-07-16", "juvenile_y1"), ("2025-04-30 23:59", "juvenile_y1"),
    ("2025-05-01", "y2"), ("2026-04-30", "y2"), ("2026-06-01", "y3"), ("2027-05-01", "y4plus"), ("2030-01-01", "y4plus"),
])
def test_age_classes(t, cls):
    row = pd.Series({"tagged_young": True, "hatch_year": 2024.0})
    a = reference.age_on(row, [pd.Timestamp(t, tz="UTC")])
    assert a["age_class"].iat[0] == cls


def test_age_days_override_and_not_stated():
    row = {"tagged_young": True, "hatch_year": 2024}
    a = reference.age_on(row, [pd.Timestamp("2024-05-02", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")],
                         fledge=pd.Timestamp("2024-06-20", tz="UTC"))
    assert a["age_days"].iat[0] == 1 and a["age_months"].iat[0] == 0
    assert a["age_class"].tolist() == ["pre_fledge", "juvenile_y1"]
    n = reference.age_on({"tagged_young": np.nan, "hatch_year": np.nan}, [pd.Timestamp("2024-07-01", tz="UTC")])
    assert n["age_class"].iat[0] == "not_stated" and np.isnan(n["age_days"].iat[0])


def _days(spec):
    """spec: list of (date, n windows, speed)."""
    t, s = [], []
    for d, n, v in spec:
        t += list(pd.Timestamp(d, tz="UTC") + pd.to_timedelta(np.arange(n) * 10, unit="s"))
        s += [v] * n
    return t, s


@pytest.mark.parametrize("spec, source, date, first", [
    ([("2024-06-01", 50, 0.1), ("2024-07-01", 20, 6.0), ("2024-07-03", 40, 6.0)], "data", "2024-07-03", False),
    ([("2024-04-20", 40, 8.0), ("2024-06-01", 50, 0.1)], "default", "2024-07-15", False),  # before hatch: ignored
    ([("2024-06-01", 50, 0.1), ("2024-10-02", 40, 8.0)], "default", "2024-07-15", False),  # after the deadline
    ([("2024-08-10", 40, 8.0)], "data", "2024-08-10", True),  # tagged after fledging: first data day
])
def test_fledge_from_speed(spec, source, date, first):
    t, s = _days(spec)
    r = reference.fledge_from_speed(t, s, 2024)
    assert r["fledge_source"] == source and r["fledge"] == pd.Timestamp(date, tz="UTC")
    assert r["fledge_is_first_data_day"] == first
