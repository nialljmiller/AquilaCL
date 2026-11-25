import json

import numpy as np
import pandas as pd
import pytest

from aquilacl import demography as dm

D, K = 8, 3
FEATS = ["vedba", "flap_band_frac", "pitch_deg", "roll_deg", "speed_mean", "climb_mean", "turn_abs_mean", "dom_freq_hz"]


# ---------------------------------------------------------------- grouped CV and permutation nulls

@pytest.mark.parametrize("n_birds, rows_per_bird, n_splits", [(20, 1, 5), (23, 7, 5), (9, 3, 3), (6, 5, 5)])
def test_grouped_folds_keep_birds_apart(n_birds, rows_per_bird, n_splits):
    rng = np.random.default_rng(n_birds)
    codes = np.repeat(np.arange(n_birds), rows_per_bird)
    yb = np.r_[np.zeros(n_birds // 2), np.ones(n_birds - n_birds // 2)].astype(int)
    folds = dm.grouped_folds(yb[codes], codes, n_splits, seed=int(rng.integers(100)))
    assert len(folds) >= 2
    seen = []
    for tr, te in folds:
        assert not set(codes[tr]) & set(codes[te])
        seen += sorted(set(codes[te]))
    assert sorted(seen) == list(range(n_birds))  # each bird tested exactly once


def _spec(X, codes, yb, level="bird"):
    return {"X": X, "codes": codes, "ybird": yb, "level": level, "run": "r", "target": "sex", "set": "mean_h", "n_h": 0}


def test_classifier_planted_signal_small_p():
    rng = np.random.default_rng(0)
    yb = np.r_[np.zeros(15), np.ones(15)].astype(int)
    codes = np.repeat(np.arange(30), 4)  # bird-month style: 4 rows per bird
    X = rng.normal(0, 1, (len(codes), 6)) + 1.2 * yb[codes][:, None]
    clf, _ = dm.run_classifiers({"k": _spec(X, codes, yb, "bird_month")}, n_repeats=3,
                                n_perm={"bird_month": 49}, seed=0, workers=1)
    r = clf.iloc[0]
    assert r.auc_mean > 0.9 and r.p_auc == pytest.approx(1 / 50) and r.n_perm == 49


def test_classifier_null_p_roughly_uniform(monkeypatch):
    monkeypatch.setattr(dm, "C_GRID", (0.1,))  # one C: no inner CV, keeps the test fast
    rng = np.random.default_rng(1)
    ps = []
    for i in range(30):
        yb = rng.permutation(np.r_[np.zeros(12), np.ones(12)].astype(int))
        X = rng.normal(0, 1, (24, 4))
        clf, _ = dm.run_classifiers({"k": _spec(X, np.arange(24), yb)}, 1, {"bird": 19}, seed=i, workers=1)
        ps.append(clf["p_auc"].iat[0])
    ps = np.array(ps)
    assert (ps <= 0.05).mean() <= 0.2 and 0.3 < ps.mean() < 0.7


def test_perm_mean_diff_uniform_and_planted():
    rng = np.random.default_rng(2)
    lab = np.r_[np.zeros(10), np.ones(10)].astype(int)
    p_null = [dm.perm_mean_diff(rng.normal(size=20), lab, 199, rng)[1] for _ in range(200)]
    assert abs(np.mean(np.array(p_null) <= 0.1) - 0.1) < 0.07
    d, p = dm.perm_mean_diff(rng.normal(size=20) + 2 * lab, lab, 199, rng)
    assert d > 1 and p == pytest.approx(1 / 200)
    strata = np.tile([0, 1], 10)
    assert dm.perm_mean_diff(rng.normal(size=20) + 2 * lab, lab, 199, rng, strata)[1] < 0.05


def test_nested_null_keeps_h_nuisance_link(monkeypatch):
    """y depends only on nuisance; h is a noisy copy of nuisance -> dAUC must not look significant."""
    monkeypatch.setattr(dm, "C_GRID", (0.1,))
    rng = np.random.default_rng(3)
    n = 30
    nu = rng.normal(size=(n, 2))
    yb = (nu[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    h = nu @ rng.normal(size=(2, 10)) + 0.3 * rng.normal(size=(n, 10))
    F = pd.DataFrame(np.c_[h, nu], index=pd.Index([f"B{i:02d}" for i in range(n)], name="bird"))
    y = pd.Series(yb, index=F.index)
    specs = {k: dm.make_spec(X, y, "bird", "r", "sex", fs, n_h=nh) for k, (X, fs, nh) in
             {"a": (F, "h_plus_nuisance", 10), "b": (F.iloc[:, 10:], "nuisance", 0)}.items()}
    assert np.allclose(specs["a"]["h_fit"] + specs["a"]["h_res"], h)
    _, nest = dm.run_classifiers(specs, 3, {"bird": 39}, seed=0, workers=1)
    assert nest["p_delta_auc"].iat[0] > 0.1


@pytest.mark.parametrize("d, p", [([1, 1, 1, 1, 1], 2 / 32), ([1, -1], 1.0), ([], np.nan)])
def test_sign_flip_exact(d, p):
    got = dm.sign_flip(np.array(d, float), 10, np.random.default_rng(0))
    assert (np.isnan(got) and np.isnan(p)) or got == pytest.approx(p)


# ---------------------------------------------------------------- end to end

def _bird(win, run_dirs, states_dirs, b, sex, young, rng, centres):
    """Windows Jun 2024 .. Sep 2025 (~210 per month); state 2 = flapping (fast, high VeDBA)."""
    months = pd.period_range("2024-06", "2025-09", freq="M")
    t = np.concatenate([(m.start_time.tz_localize("UTC") + pd.Timedelta(days=2) +
                         pd.to_timedelta(np.arange(210) * 600, unit="s")).to_numpy() for m in months])
    t = pd.to_datetime(t, utc=True)
    n = len(t)
    state = rng.integers(0, K, n)
    if young:  # grounded until 3 July 2024, then 40 fast windows that day
        pre = t < pd.Timestamp("2024-07-03", tz="UTC")
        state[pre] = 0
        day = np.flatnonzero((t >= pd.Timestamp("2024-07-03", tz="UTC")) & (t < pd.Timestamp("2024-07-04", tz="UTC")))
        state[day[:40]] = 2
    speed = np.where(state == 0, 0.1, np.where(state == 1, 8.0, 11.0)) + rng.normal(0, 0.3, n)
    speed[rng.random(n) < 0.2] = np.nan  # windows without GPS
    flap = state == 2
    dom = np.where(flap, (2.3 if sex == "f" else 2.7) + rng.normal(0, 0.05, n), rng.uniform(0.5, 5, n))
    meta = pd.DataFrame({"bird": b, "tag": "T", "t_start": t, "segment": np.arange(n), "k_in_segment": 0,
                         "sensitivity_k": (rng.random(n) < 0.3).astype(int), "acc_samples": 200,
                         "gps_fixes": np.where(np.isnan(speed), 0, 10)})
    feats = pd.DataFrame({"vedba": np.where(flap, 0.3, 0.05) + rng.normal(0, 0.01, n), "flap_band_frac": rng.random(n),
                          "pitch_deg": rng.normal(0, 5, n), "roll_deg": rng.normal(0, 5, n), "speed_mean": speed,
                          "climb_mean": rng.normal(0, 1, n), "turn_abs_mean": rng.uniform(0, 20, n), "dom_freq_hz": dom,
                          "label_flapping": flap.astype(float)})
    d = win / b
    d.mkdir(parents=True)
    meta.to_parquet(d / "meta.parquet")
    feats.to_parquet(d / "features.parquet")
    offset = (0.6 if sex == "f" else -0.6) * np.eye(D)[0]
    for run, sdir in zip(run_dirs, states_dirs):
        h = centres[state] + offset + rng.normal(0, 0.3, (n, D))
        np.savez(run / "embeddings" / f"{b}.npz", h=h.astype(np.float16))
        sdir.setdefault("rows", []).append(pd.DataFrame({"bird": b, "row": np.arange(n), "state": state, "state_hmm": state,
                                                         "pitch_rot_deg": rng.normal(0, 5, n), "roll_rot_deg": rng.normal(0, 5, n)}))


def _synthetic(tmp_path):
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 1, (K, D))
    win = tmp_path / "windows"
    runs = [tmp_path / "runs" / "rA", tmp_path / "runs" / "rB"]
    sdirs = [tmp_path / "states_A", tmp_path / "states_B"]
    for r in runs:
        (r / "embeddings").mkdir(parents=True)
    acc = [{}, {}]
    birds = [(f"B{i:02d}", "f" if i % 2 else "m", i < 4) for i in range(12)]
    for b, sex, young in birds:
        _bird(win, runs, acc, b, sex, young, rng, centres)
    (win / "Bempty").mkdir()
    pd.DataFrame(columns=["bird", "t_start"]).to_parquet(win / "Bempty" / "meta.parquet")
    for s, a in zip(sdirs, acc):
        s.mkdir()
        pd.concat(a["rows"]).to_parquet(s / "states.parquet")
    pd.DataFrame({"state": range(K), "vedba_median": [0.05, 0.05, 0.3], "speed_mean_median": [0.1, 8.0, 11.0]}
                 ).set_index("state").to_csv(sdirs[0] / "state_profiles.csv")  # second run: fallback rule
    rows = [[str(i), f"{b}; {100 + i}", "2024-06-05 00:00:00.000", "", "", sex, "",
             "Tagged as a nestling. Band XY" if young else "", "M1" if i < 6 else "M2", "fw1"]
            for i, (b, sex, young) in enumerate(birds)]
    ref = tmp_path / "reference data (1).csv"
    pd.DataFrame(rows, columns=["tag-id", "animal-id", "deploy-on-date", "deploy-off-date", "animal-comments",
                                "animal-sex", "animal-life-stage", "deployment-comments", "tag-model", "tag-firmware"]
                 ).to_csv(ref, index=False)
    return runs, sdirs, win, ref


def test_end_to_end_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "C_GRID", (0.1,))  # inner C selection is covered by test_classifier_planted_signal_small_p
    runs, sdirs, win, ref = _synthetic(tmp_path)
    out = tmp_path / "out"
    args = ["--run", str(runs[0]), "--run", str(runs[1]), "--states", str(sdirs[0]), "--states", str(sdirs[1]),
            "--windows", str(win), "--reference", str(ref), "--out", str(out), "--n-perm", "19",
            "--n-perm-month", "5", "--n-repeats", "2", "--workers", "1"]
    dm.main(args)
    for f in ["birds.csv", "confounds.csv", "classifiers.csv", "nested.csv", "wingbeat.csv", "wingbeat_birds.csv",
              "age_same_months_tests.csv", "summary.md", "metadata.json", "classifiers_bird.png",
              "features_rA_bird.parquet", "features_rB_month.parquet", "age_months_rA.csv", "age_rA.png",
              "wingbeat_rB.png"]:
        assert (out / f).exists(), f
    B = pd.read_csv(out / "birds.csv").set_index("bird")
    assert len(B) == 12 and B["tagged_young"].notna().sum() == 4
    young = B[B["tagged_young"].notna()]
    assert (young["fledge_source"] == "data").all() and (young["n_pre_fledge"] > 0).all()
    assert (B.loc[B["tagged_young"].isna(), "n_pre_fledge"] == 0).all()
    clf = pd.read_csv(out / "classifiers.csv")
    assert set(clf["level"]) == {"bird", "bird_month"} and set(clf["target"]) == {"sex", "young_tagged_vs_not_stated"}
    assert set(clf["feature_set"]) == set(dm.FEATURE_SETS)
    mh = clf[(clf.level == "bird") & (clf.target == "sex") & (clf.feature_set == "mean_h")]
    assert (mh["auc_mean"] > 0.9).all() and (mh["n_birds"] == 12).all()  # planted sex offset in h
    nest = pd.read_csv(out / "nested.csv")  # 2 runs x 2 targets x 2 levels; permutation null at bird level only
    assert len(nest) == 8 and (nest.loc[nest.level == "bird", "n_perm"] == 19).all()
    assert (nest.loc[nest.level == "bird_month", "n_perm"] == 0).all()
    wb = pd.read_csv(out / "wingbeat.csv")
    allw = wb[wb.subset == "all full windows"]
    assert len(allw) == 3 and (allw["diff_f_minus_m"] < -0.3).all()  # label + flapping state of each run
    assert allw["selection"].str.contains("flapping state 2").sum() == 2
    t = pd.read_csv(out / "age_same_months_tests.csv")
    assert set(t["run"]) == {"rA", "rB"} and (t["n_birds"] == 4).all()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["assumptions"]["hatch_date"].startswith("1 May") and meta["reference"]["tagged_young"] == 4
    text = (out / "summary.md").read_text() + B.astype(str).to_string()
    assert "Band XY" not in text and "15 Jul" in text
    with pytest.raises(SystemExit):
        dm.main(args)  # refuses a non-empty output directory
