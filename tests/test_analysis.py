import json
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

from aquilacl import analysis

STATES = {  # state: (vedba g, speed m/s, flap_band_frac)
    0: (0.02, 0.1, 0.1),
    1: (0.5, 12.0, 0.7),
    2: (0.08, 9.0, 0.2),
}
BIRDS = [f"B{i}" for i in range(6)]


def _inputs(tmp_path, n_seg=30, seg_len=20, seed=0):
    rng = np.random.default_rng(seed)
    win, run, sd = tmp_path / "windows", tmp_path / "run", tmp_path / "states"
    (run / "embeddings").mkdir(parents=True)
    sd.mkdir()
    rows = []
    for b, bird in enumerate(BIRDS):
        meta, feats, acc, h = [], [], [], []
        for s in range(n_seg):
            t0 = pd.Timestamp("2024-07-01", tz="UTC") + pd.Timedelta(days=13 * s + b, hours=int(rng.integers(24)))
            seq = rng.integers(3, size=3)  # state changes within a segment, so some runs are not censored
            for k in range(seg_len):
                st = int(seq[3 * k // seg_len])
                v, sp, fb = STATES[st]
                meta.append(dict(bird=bird, tag=f"T{b}", t_start=t0 + pd.Timedelta(seconds=10 * k), segment=s,
                                 k_in_segment=k, sensitivity_k=s % 2, sun_elev_deg=float(rng.uniform(-30, 60)),
                                 acc_samples=200 if k else 40, gps_fixes=int(rng.integers(0, 3)),
                                 local_solar_hour=np.nan if s == 0 else float(rng.uniform(0, 24))))
                feats.append(dict(vedba=v * rng.lognormal(0, 0.1), odba=1.5 * v, flap_band_frac=fb, dom_freq_hz=2.5,
                                  pitch_deg=0.0, roll_deg=5.0, speed_mean=sp, climb_mean=0.1, turn_abs_mean=3.0,
                                  label_perched=float(st == 0), label_flapping=float(st == 1),
                                  label_circling=np.nan if k % 3 else float(st == 2)))
                acc.append(rng.normal(0, v, (200, 3)) + [0, 0, 1])
                h.append(np.eye(16)[st] * 3 + rng.normal(0, 0.3, 16))
                rows.append(dict(bird=bird, row=len(meta) - 1, t_start=meta[-1]["t_start"], segment=s, state=st,
                                 state_hmm=st if k % 7 else (st + 1) % 3, p_state=float(rng.uniform(0.5, 1)),
                                 pitch_rot_deg=float(rng.normal(0, 5)), roll_rot_deg=float(rng.normal(0, 5))))
        d = win / bird
        d.mkdir(parents=True)
        pd.DataFrame(meta).to_parquet(d / "meta.parquet")
        pd.DataFrame(feats).to_parquet(d / "features.parquet")
        np.save(d / "acc.npy", np.asarray(acc, np.float16))
        m = np.ones((len(acc), 200), bool)
        m[[x["acc_samples"] < 200 for x in meta], 40:] = False
        np.save(d / "acc_mask.npy", m)
        np.savez(run / "embeddings" / f"{bird}.npz", h=np.asarray(h, np.float16), z=np.zeros((len(h), 4), np.float16),
                 row=np.arange(len(h)))
    (win / "B_empty").mkdir()  # bird with 0 windows: empty files
    pd.DataFrame({"bird": pd.Series(dtype=str)}).to_parquet(win / "B_empty" / "meta.parquet")
    pd.DataFrame({"vedba": pd.Series(dtype=float)}).to_parquet(win / "B_empty" / "features.parquet")

    st = pd.DataFrame(rows)
    for i in (1, 2, 3):
        st[f"pca_{i}"] = rng.normal(size=len(st))
    st.to_parquet(sd / "states.parquet")
    pd.DataFrame({"state": [0, 1, 2], "vedba_median": [0.02, 0.5, 0.08], "flag_nonfull": [False, True, False]}
                 ).to_csv(sd / "state_profiles.csv", index=False)
    pd.DataFrame(np.full((3, 3), 1 / 3), index=pd.Index(range(3), name="from")).to_csv(sd / "transitions.csv")
    per_k = {str(k): {"jaccard": [0.9] * k, "jaccard_min": 0.9, "heldout_ari": None if k == 4 else 0.8,
                      "presence": {"birds": [6] * k, "cohort_ok": [True] * k, "ok": True}, "bic": 1000.0 - k}
             for k in (2, 3, 4)}
    per_k["3"]["jaccard"] = [0.95, 0.9, 0.6]  # as stage 5a writes it: list in final state numbering
    (sd / "state_selection.json").write_text(json.dumps({"chosen_k": 3, "rule_met": False, "rule": "r", "per_k": per_k}))

    log = "step,loss,lr,pos_cos,neg_cos,sec\n"
    body = "".join(f"{s},{4 - s / 100},{1e-3},{0.5 + s / 400},{0.1},{s}\n" for s in range(0, 100, 10))
    (run / "loss.csv").write_text(log + body + log + body)  # appended after a resume
    ev = {"window_level_holdout": {"vedba": {"r2": 0.9}, "label_perched": {"balanced_accuracy": 0.95, "chance": 0.5},
                                   "sensitivity_k (nuisance)": {"balanced_accuracy": 0.99}},
          "hour_level": {"speed": {"r2": 0.8, "birds": 6}},
          "invariance": {"burst_same_state": 0.7, "burst_n": 100}}
    (run / "evaluation.json").write_text(json.dumps(ev))

    ref = pd.DataFrame({  # B4: two deployments with conflicting sex; B5 absent; X9 not in the windows
        "tag-id": [1, 2, 3, 4, 5, 6, 7, 8],
        "animal-id": ["B0", "B1", "B1", "B2", "B3", "B4", "B4", "X9"],
        "animal-sex": ["m", "f", "u", "m", "f", "m", "f", "f"],
        "animal-life-stage": ["adult", "", "", "juvenile", "", "", "", ""],
        "deploy-on-date": ["2024-06-12 17:50:00.000", "2024-06-20", "2025-06-20", "", "2024-10-01", "", "", ""],
        "deploy-on-latitude": [43.1] * 8,
        "animal-comments": ["SECRET nest site text"] * 8,
    })
    ref.to_csv(tmp_path / "reference.csv", index=False)
    return ["--states", str(sd), "--windows", str(win), "--run", str(run), "--reference", str(tmp_path / "reference.csv"),
            "--permutations", "99"]


def _no_reference(monkeypatch):
    """Force the fallback: aquilacl.reference cannot be imported (it may or may not exist in the tree)."""
    monkeypatch.setitem(sys.modules, "aquilacl.reference", None)


def test_analysis_end_to_end(tmp_path, monkeypatch):
    _no_reference(monkeypatch)
    args = _inputs(tmp_path)
    out = tmp_path / "out"
    analysis.main(args + ["--out", str(out)])
    index = (out / "index.md").read_text()
    listed = re.findall(r"\]\(([^)]+)\)", index)
    assert len(listed) >= 25
    for name in listed:
        assert (out / name).stat().st_size > 0, name
    for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"):
        assert f"| {m} |" in index
    # first-draft map marks outputs this run could not write: only the life-stage test (1 bird per group here)
    missing = re.findall(r"(m\d_\w+\.\w+)[^,|]*\[not produced in this run", index)
    assert missing == ["m6_group_life_stage.png"] * 2
    assert "life_stage: not tested and m6_group_life_stage.png not produced" in index
    assert {f"m4_state_cards_{s}.png" for s in STATES} | {"m7_outliers.png", "m7_rare_states.png"} <= set(listed)

    # metadata: reported, never invented
    cov = pd.read_csv(out / "m6_metadata_coverage.csv").set_index("field")
    assert cov.loc["sex", ["matched", "with_value", "conflict", "missing"]].tolist() == [5, 4, 1, 1]
    assert cov.loc["life_stage", "with_value"] == 2 and cov.loc["deploy_on", "with_value"] == 3
    summ = pd.read_csv(out / "m6_bird_summary.csv").set_index("bird")
    assert summ.loc[["B4", "B5"], "sex"].isna().all() and summ.loc["B1", "sex"] == "f"
    tests = pd.read_csv(out / "m6_group_tests.csv")
    assert tests.loc[tests["field"] == "sex", "status"].str.startswith("tested").all()
    assert tests.loc[tests["field"] == "life_stage", "status"].str.startswith("not tested").all()
    assert "reference CSV" in index  # unmatched bird reported in the notes
    assert "flag_nonfull is set for state(s) [1]" in index and "no K met the selection rule" in index
    assert pd.read_csv(out / "m3_jaccard.csv")["jaccard"].tolist() == [0.95, 0.9, 0.6]

    # privacy: no positions or comment text in any output
    cat = pd.read_csv(out / "m8_window_catalogue.csv.gz", nrows=5)
    assert not any("lat" in c or "lon" in c for c in cat.columns)
    for p in list(out.glob("*.csv")) + [out / "m8_window_catalogue.csv.gz"]:  # exact solar time/elevation + time = position
        cols = pd.read_csv(p, nrows=1).columns
        assert "local_solar_hour" not in cols and "sun_elev_deg" not in cols, p.name
    for p in out.glob("*.csv"):
        assert "SECRET" not in p.read_text() and "43.1" not in p.read_text(), p.name
    cat = pd.read_csv(out / "m8_window_catalogue.csv.gz")
    assert len(cat) == 6 * 30 * 20 and (cat["state"] != cat["state_hmm"]).any()  # GMM state and HMM state kept apart

    # transitions computed within segments only: 6 birds x 30 segments x 7 steps
    assert pd.read_csv(out / "m5_transitions.csv", index_col=0).to_numpy().sum() == 6 * 30 * 19

    # season vs recording mode and outlier factors
    sm = pd.read_csv(out / "m5_seasonal_by_mode.csv")
    assert {"all windows", "k=0", "k=1", "GPS", "no GPS", "mode-adjusted"} <= set(sm["stratum"])
    assert "m5_seasonal_by_mode: seasonal range" in index
    of = pd.read_csv(out / "m7_outlier_factors.csv")
    assert len(of) == 6 and of.loc[~of["factor"].str.startswith("short"), "ratio"].notna().all()
    # non-full groups here have < 30 windows, so no short window is scored: ratio undefined, not 0
    assert of.loc[of["factor"].str.startswith("short"), "ratio"].isna().all()
    assert "windows_summary.csv with bursts_unreliable_calibration is not in the windows directory" in index
    ct = pd.read_csv(out / "m5_seasonal_by_mode_contrast.csv")
    assert len(ct) == 3 and "m5_seasonal_by_mode_contrast" in index
    assert "aquilacl.reference could not be imported" in index

    with pytest.raises(SystemExit):
        analysis.main(args + ["--out", str(out)])


def test_state_without_windows_in_analysis_column(tmp_path, monkeypatch):
    _no_reference(monkeypatch)
    args = _inputs(tmp_path)
    sp = tmp_path / "states" / "states.parquet"
    st = pd.read_parquet(sp)
    st.loc[st["state_hmm"] == 2, "state_hmm"] = 0  # smoothing emptied state 2
    st.to_parquet(sp)
    analysis.main(args + ["--out", str(tmp_path / "out2")])
    assert "state 2 has no windows" in (tmp_path / "out2" / "index.md").read_text()


def test_reference_id_with_tag_suffix(tmp_path):
    """Movebank writes animal-id as "<ring id>; <tag id>"; birds must still match, and two deployments of one
    animal with the same sex are not a conflict."""
    from aquilacl import analysis

    p = tmp_path / "ref.csv"
    p.write_text("tag-id,animal-id,animal-sex,deploy-on-date\n"
                 "1,0001-00001; 1,m,2024-06-01\n2,0001-00001; 2,m,2025-06-01\n3,0001-00002; 3,f,2024-07-01\n")

    class _Out:
        notes = []

        def note(self, s):
            self.notes.append(s)

    meta, cov = analysis.load_reference(p, ["0001-00001", "0001-00002", "0001-00003"], _Out())
    sex = meta.set_index("bird")["sex"]
    assert sex.iloc[:2].tolist() == ["m", "f"] and pd.isna(sex.iloc[2])
    assert int(cov.set_index("field").loc["sex", "matched"]) == 2
    assert str(meta.set_index("bird")["deploy_on"].iloc[0].date()) == "2024-06-01"


YOUNG = {"B0", "B1", "B2"}  # tagged young, hatched 2024-05-01, fledged at 70 days; B3-B5: age not stated
FLEDGE_DAYS = 70


def _fake_reference(style="tuple"):
    """The interface given to stage 5b: age_on -> (age_days, age_class). style "frame": age_on returns a DataFrame,
    takes ``fledge`` and the module has fledge_from_speed (the form the real module took)."""
    mod = types.ModuleType("aquilacl.reference")

    def load_birds(path, birds):
        return pd.DataFrame({
            "bird": birds, "sex": ["m", "f"] * (len(birds) // 2) + ["m"] * (len(birds) % 2),
            "tagged_young": [True if b in YOUNG else np.nan for b in birds],
            "young_term": ["MARKERTERM" if b in YOUNG else np.nan for b in birds],
            "hatch_year": [2024.0 if b in YOUNG else np.nan for b in birds],
            "deploy_on": pd.Timestamp("2024-06-20", tz="UTC"), "n_deployments": 1,
            "migrant": [True if b in ("B0", "B3") else np.nan for b in birds],
            "tag_model": "MARKERTAG", "tag_firmware": "MARKERTAG"})

    def age_on(row, ts, fledge=None):
        ts = pd.to_datetime(pd.Series(ts), utc=True)
        if row["tagged_young"] is not True:
            d, cls = np.full(len(ts), np.nan), np.full(len(ts), "not_stated", dtype=object)
        else:
            d = ((ts - pd.Timestamp("2024-05-01", tz="UTC")).dt.total_seconds() / 86400).to_numpy()
            fl = FLEDGE_DAYS if fledge is None else (fledge - pd.Timestamp("2024-05-01", tz="UTC")).days
            cls = np.select([d < fl, d < 365, d < 730, d < 1095], ["pre_fledge", "juvenile_y1", "y2", "y3"],
                            "y4plus").astype(object)
        return pd.DataFrame({"age_days": d, "age_class": cls}) if style == "frame" else (d, cls)

    def fledge_from_speed(t_start, speed_mean, hatch_year):
        return {"fledge": pd.Timestamp("2024-05-01", tz="UTC") + pd.Timedelta(days=FLEDGE_DAYS),
                "fledge_source": "data", "fledge_is_first_data_day": False}

    mod.load_birds, mod.age_on = load_birds, age_on
    if style == "frame":
        mod.fledge_from_speed = fledge_from_speed
    return mod


@pytest.mark.parametrize("style", ["tuple", "frame"])
def test_age_outputs(tmp_path, monkeypatch, style):
    monkeypatch.setitem(sys.modules, "aquilacl.reference", _fake_reference(style))
    args = _inputs(tmp_path)
    out = tmp_path / "out"
    analysis.main(args + ["--out", str(out)])
    index = (out / "index.md").read_text()
    listed = set(re.findall(r"\]\(([^)]+)\)", index))
    assert {"m6_group_tagged_young.png", "m6_group_migrant.png", "m6_development.png", "m6_development.csv",
            "m6_age_y1_vs_y2.csv", "m6_age_y1_vs_y2_summary.csv", "m6_age_y1_vs_y2.png"} <= listed
    assert "m6_group_life_stage.png" not in listed and "not produced in this run" not in index
    assert "no life-stage column" not in index

    tests = pd.read_csv(out / "m6_group_tests.csv")
    assert set(tests["field"]) == {"sex", "tagged_young", "migrant"}
    ty = tests[tests["field"] == "tagged_young"]
    assert ty["status"].str.contains("descriptive only").all() and "age not stated: 3" in ty["groups"].iat[0]
    assert tests.loc[tests["field"] == "migrant", "groups"].iat[0] in ("not stated: 4; migrant: 2",
                                                                      "migrant: 2; not stated: 4")

    dev = pd.read_csv(out / "m6_development.csv", index_col=0)
    assert dev.index.name == "months_since_fledging" and dev.index.min() >= 0
    assert dev["n_birds"].max() <= len(YOUNG)
    # first window 2024-07-01 is 61 days old (pre_fledge, left out); month 0 starts at fledging (day 70)
    assert np.allclose(dev.filter(like="share_state_").sum(axis=1), 1)

    pairs = pd.read_csv(out / "m6_age_y1_vs_y2.csv")
    assert set(pairs["bird"]) <= YOUNG and len(set(pairs["bird"])) == 3 and set(pairs["month"]) == {7}
    assert {"k1_share_y1", "k1_share_y2", "gps_share_y1", "gps_share_y2"} <= set(pairs.columns)
    summ = pd.read_csv(out / "m6_age_y1_vs_y2_summary.csv")
    assert (summ["birds"] == 3).all() and summ["p_sign_flip"].between(0.25, 1).all()  # 3 birds: 8 exact patterns
    assert summ["status"].str.contains("8 all sign patterns").all()

    cov = pd.read_csv(out / "m6_metadata_coverage.csv").set_index("field")
    assert cov.loc["tagged_young", "with_value"] == 3 and cov.loc["sex", "matched"] == 5
    bs = pd.read_csv(out / "m6_bird_summary.csv").set_index("bird")
    assert bs.loc["B0", "tagged_young"] and pd.isna(bs.loc["B3", "tagged_young"])
    # free-text and tag fields of the reference module never reach an output
    for p in list(out.glob("*.csv")) + [out / "index.md"]:
        assert "MARKERTERM" not in p.read_text() and "MARKERTAG" not in p.read_text(), p.name


def test_flag_label_never_reads_nan_as_true():
    lab = [analysis.flag_label(v, "y", "n") for v in (True, np.True_, 1.0, False, 0, np.nan, None, pd.NA, "true")]
    assert lab == ["y", "y", "y", "n", "n", "not stated", "not stated", "not stated", "not stated"]


def test_seasonal_by_mode_adjusted():
    """k=1 only in May-July: the mode-adjusted share exists only there and equals the fixed-mix combination."""
    rows = []
    for b in range(4):
        for month in range(1, 13):
            for i in range(40):
                k1 = month in (5, 6, 7) and i < 20
                # state depends on mode only: k=1 -> state 1, k=0 -> state 0 (in May-July 1/4 of k=0 is state 2)
                st = 1 if k1 else (2 if month in (5, 6, 7) and i % 4 == 0 else 0)
                rows.append(dict(bird=f"B{b}", t_start=pd.Timestamp(2025, month, 1 + i % 28, tz="UTC"),
                                 sensitivity_k=int(k1), gps_fixes=10 if k1 else int(i % 10 == 0),
                                 acc_samples=200 if k1 or i % 10 else 40, state=st))
    df = pd.DataFrame(rows)
    df["bird"] = df["bird"].astype("category")
    df["regime"] = np.where(df["acc_samples"] >= analysis.FULL_SAMPLES, "full", "non-full")
    tab, w = analysis.seasonal_by_mode(df, [0, 1, 2])
    assert w["k=1"] == pytest.approx(3 * 20 / (12 * 40))
    adj = tab[tab["stratum"] == "mode-adjusted"].set_index("month")
    have = adj["share_state_0"].dropna().index.tolist()
    assert have == [5, 6, 7]
    k0 = tab[tab["stratum"] == "k=0"].set_index("month")
    assert sorted(k0.index) == list(range(1, 13))
    # in May: k=0 share of state 0 = 15/20 (i%4==0 -> state 2), k=1 share of state 1 = 1
    assert adj.loc[5, "share_state_0"] == pytest.approx(w["k=0"] * 0.75)
    assert adj.loc[5, "share_state_1"] == pytest.approx(w["k=1"])
    raw = tab[tab["stratum"] == "all windows"].set_index("month")
    assert raw.loc[5, "share_state_1"] == pytest.approx(0.5)  # raw share mixes in the recording schedule
    ct = analysis.seasonal_mode_contrast(tab, [0, 1, 2]).set_index("state")
    assert ct["k1_months"].iat[0] == "5 6 7"
    # state 1 only in k=1: raw contrast 0.5 - 0, within k=0 0 - 0 -> the raw difference is the recording schedule
    assert ct.loc[1, "contrast_all"] == pytest.approx(0.5) and ct.loc[1, "contrast_within_k0"] == pytest.approx(0)
    # state 2 is seasonal within k=0 too (1/4 of k=0 windows in May-July): survives
    assert ct.loc[2, "contrast_within_k0"] == pytest.approx(0.25) and ct.loc[2, "same_sign_within_k0"]
    notes = analysis.seasonal_mode_note(tab, df, [0, 1, 2], w, ct.reset_index())
    assert any("no mode-adjusted share in months [1, 2, 3, 4, 8, 9, 10, 11, 12]" in n for n in notes)
    assert any("k=0 windows have GPS" in n for n in notes)
    assert any("state 2: all +0.125, within k=0 +0.250 (same sign)" in n for n in notes)


def test_outlier_factors_ratio():
    scored = pd.DataFrame({"bird": ["A"] * 190 + ["C"] * 10,
                           "knn_score": np.arange(200, dtype=float), "acc_samples": [40] * 20 + [200] * 180,
                           "sensitivity_k": [0] * 100 + [1] * 100, "gps_fixes": [0] * 150 + [5] * 50})
    gallery = scored.nlargest(10, "knn_score")
    t = analysis.outlier_factors(scored, gallery, top_frac=0.05, unreliable_birds={"C"})
    t = t.set_index(["outlier_set", "factor"])
    top = t.xs("top 5% of scored windows")
    assert top["n_outliers"].iat[0] == 10
    assert top.loc["sensitivity k=1 (continuous mode)", "ratio"] == pytest.approx(1 / 0.5)
    assert top.loc["GPS present (gps_fixes > 0)", "ratio"] == pytest.approx(1 / 0.25)
    assert top.loc["short window (acc_samples < 190)", "ratio"] == 0
    assert top.loc[analysis.UNRELIABLE_FACTOR, "ratio"] == pytest.approx(1 / 0.05)  # all top 10 from bird C
    assert analysis.UNRELIABLE_FACTOR not in analysis.outlier_factors(scored, gallery)["factor"].tolist()


def test_with_real_reference_module(tmp_path):
    """Runs with the repository's aquilacl.reference if it exists (no young term in the fixture comments)."""
    pytest.importorskip("aquilacl.reference")
    args = _inputs(tmp_path)
    analysis.main(args + ["--out", str(tmp_path / "out")])
    index = (tmp_path / "out" / "index.md").read_text()
    assert "aquilacl.reference could not be imported" not in index
    assert "tagged_young: not tested" in index
    for p in (tmp_path / "out").glob("*.csv"):
        assert "SECRET" not in p.read_text(), p.name
