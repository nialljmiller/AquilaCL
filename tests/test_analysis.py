import json
import re

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
                                 state_hmm=st if k % 7 else (st + 1) % 3, p_state=float(rng.uniform(0.5, 1))))
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


def test_analysis_end_to_end(tmp_path):
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
    for p in out.glob("*.csv"):
        assert "SECRET" not in p.read_text() and "43.1" not in p.read_text(), p.name
    cat = pd.read_csv(out / "m8_window_catalogue.csv.gz")
    assert len(cat) == 6 * 30 * 20 and (cat["state"] != cat["state_hmm"]).any()  # GMM state and HMM state kept apart

    # transitions computed within segments only: 6 birds x 30 segments x 7 steps
    assert pd.read_csv(out / "m5_transitions.csv", index_col=0).to_numpy().sum() == 6 * 30 * 19

    with pytest.raises(SystemExit):
        analysis.main(args + ["--out", str(out)])
