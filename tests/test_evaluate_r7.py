"""r7 probes in aquilacl.evaluate and scripts/compare_runs.py on tiny synthetic run directories."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquilacl import evaluate, states

FS = 20
T = np.arange(200) / FS


def _rot(ax, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _make(tmp_path, n_birds=16, n_win=90, seed=0, sex_signal=True):
    """Windows + run dir. Gravity-frame pitch, roll and horizontal dynamic share are planted through acc.npy
    (with a per-bird mounting tilt) and written into h, as are wingbeat frequency, turn sign and (flight only) sex."""
    rng = np.random.default_rng(seed)
    win, run = tmp_path / "win", tmp_path / "run"
    (run / "embeddings").mkdir(parents=True)
    orient, truth, ref = [], {}, []
    for b in range(n_birds):
        name = f"X{b:02d}"
        Rm = _rot("x", rng.uniform(-25, 25)) @ _rot("y", rng.uniform(-25, 25))  # sensor = Rm @ gravity frame
        orient.append({"bird": name, "R": Rm.T.ravel().tolist()})
        sex = b % 2
        ref.append({"animal-id": name, "animal-sex": "f" if sex else "m", "deploy-on-date": "2025-01-01",
                    "animal-comments": "", "deployment-comments": ""})
        p, r, q = rng.uniform(-30, 30, n_win), rng.uniform(-30, 30, n_win), rng.uniform(0.1, 0.9, n_win)
        k = (np.arange(n_win) % 2).astype(int)
        L = np.where(np.arange(n_win) % 3 == 0, 40, 200)
        flap = (rng.random(n_win) < 0.5).astype(float)
        dom = rng.uniform(2, 4, n_win)
        turn = rng.normal(0, 90, n_win)
        speed = np.where(flap == 1, 12.0, rng.choice([1.0, 9.0], n_win))
        perched = ((flap == 0) & (speed < 5)).astype(float)
        acc, am = np.zeros((n_win, 200, 3), np.float16), np.zeros((n_win, 200), bool)
        for i in range(n_win):
            pr, rr = np.radians(p[i]), np.radians(r[i])
            g = np.array([np.sin(pr), np.cos(pr) * np.sin(rr), np.cos(pr) * np.cos(rr)])
            wave = 0.3 * np.sin(2 * np.pi * 3 * T + rng.uniform(0, 6))  # 3 Hz: averages out over the 1 s static mean
            dyn = wave[:, None] * np.array([np.sqrt(q[i] / 2), np.sqrt(q[i] / 2), np.sqrt(1 - q[i])])
            acc[i] = ((g + dyn) @ Rm.T).astype(np.float16)
            am[i, :L[i]] = True
        fly = (speed > 5) | (flap == 1)
        h = np.c_[p / 30, r / 30, (q - 0.5) * 4, np.where(flap == 1, dom - 3, 0) * 2, np.sign(turn) * k,
                  (sex - 0.5) * fly * (1.0 if sex_signal else 0.0), np.ones(n_win)]
        h = np.c_[h + rng.normal(0, 0.15, h.shape), rng.normal(0, 0.3, (n_win, 3))].astype(np.float32)
        d = win / name
        d.mkdir(parents=True)
        np.save(d / "acc.npy", acc)
        np.save(d / "acc_mask.npy", am)
        t0 = pd.Timestamp("2025-03-01", tz="UTC")
        pd.DataFrame({"bird": name, "tag": "1", "t_start": [t0 + pd.Timedelta(minutes=5 * i) for i in range(n_win)],
                      "segment": np.arange(n_win), "k_in_segment": 0, "segment_windows": 1, "sensitivity_k": k,
                      "decoding": "21", "sun_elev_deg": 30.0, "acc_samples": L, "gps_fixes": np.where(k == 1, 10, 0)}
                     ).to_parquet(d / "meta.parquet")
        pd.DataFrame({"vedba": 0.1 + flap, "speed_mean": speed, "climb_mean": 0.0, "turn_abs_mean": np.abs(turn),
                      "turn_net_deg": np.where(k == 1, turn, np.nan), "flap_band_frac": flap, "dom_freq_hz": dom,
                      "label_flapping": flap, "label_perched": perched, "label_circling": 0.0, "label_gliding": 0.0,
                      "label_climbing_soar": 0.0, "label_night": 0.0}).to_parquet(d / "features.parquet")
        np.savez(run / "embeddings" / f"{name}.npz", h=h)
        truth[name] = (p, r, q)
    (run / "orientation.json").write_text(json.dumps(orient))
    (run / "holdout.json").write_text(json.dumps([f"X{b:02d}" for b in range(n_birds - 6, n_birds)]))
    pd.DataFrame(ref).to_csv(tmp_path / "ref.csv", index=False)
    (tmp_path / "events").mkdir()
    return win, run, truth


def test_rotated_targets_equal_planted_values(tmp_path):
    win, run, truth = _make(tmp_path, n_birds=3, n_win=30)
    meta, H = evaluate.load_run(run, win)
    rows = np.arange(0, len(meta), 2)
    out = evaluate.rotated_window_targets(run, win, meta, rows)
    assert np.isnan(out[1::2]).all()
    tp = np.concatenate([np.c_[truth[b]] for b in sorted(truth)])
    full = rows[meta["acc_samples"].to_numpy()[rows] >= 190]
    np.testing.assert_allclose(out[full, :2], tp[full, :2], atol=0.6)  # degrees (float16 storage)
    np.testing.assert_allclose(out[rows, :2], tp[rows, :2], atol=3.0)  # 2 s windows: static part less exact
    np.testing.assert_allclose(out[rows, 2], tp[rows, 2], atol=0.03)
    # the sensor frame differs (mounting tilt): a frame bug would not pass
    sensor = states.rotated_posture(np.load(win / "X00" / "acc.npy"), np.load(win / "X00" / "acc_mask.npy"), np.eye(3))
    assert np.abs(sensor - tp[:30, :2]).max() > 5


def test_r7_window_probes_recover_planted_signals_per_subset(tmp_path):
    win, run, _ = _make(tmp_path)
    meta, H = evaluate.load_run(run, win)
    hold = json.loads((run / "holdout.json").read_text())
    res = evaluate.r7_probes(run, win, meta, H, hold, np.random.default_rng(0), n_train=500, n_test=500)
    wh = res["window_holdout"]
    assert set(wh) == set(evaluate.SUBSETS)
    for s in evaluate.SUBSETS:
        for t in ("pitch_rot_deg", "roll_rot_deg", "dyn_horizontal_share"):
            assert wh[s][t]["r2"] > 0.8, (s, t, wh[s][t])
    for s in ("all", "full", "k0", "k1"):
        assert wh[s]["dom_freq_hz_flapping"]["r2"] > 0.8, (s, wh[s]["dom_freq_hz_flapping"])
    assert "not applicable" in wh["short"]["dom_freq_hz_flapping"]["skipped"]
    for s in ("all", "full", "short", "k1"):
        assert wh[s]["turn_sign"]["balanced_accuracy"] > 0.9, (s, wh[s]["turn_sign"])
    assert "not applicable" in wh["k0"]["turn_sign"]["skipped"]
    assert "skipped" in wh["k0"]["sensitivity_k (nuisance)"]  # one class inside a k subset
    assert res["sex_flight"] == {"skipped": "no --reference"}


def test_sex_probe_planted_and_null(tmp_path):
    win, run, _ = _make(tmp_path, n_birds=20, n_win=60)
    meta, H = evaluate.load_run(run, win)
    res = evaluate.sex_flight_probe(meta, H, tmp_path / "ref.csv", n_perm=49, n_repeats=2, seed=0, min_windows=10)
    assert res["birds_known_sex"] == 20 and res["flight"]["n_birds"] == 20
    assert res["flight"]["auc"] > 0.9 and res["flight"]["p_auc"] <= 0.05
    assert res["ground"]["p_auc"] > 0.05  # sex was planted in flight windows only
    # null: sex labels unrelated to h; p should spread over (0, 1]
    ps = []
    for s in range(8):
        sub = tmp_path / f"null{s}"
        sub.mkdir()
        w, r, _ = _make(sub, n_birds=16, n_win=30, seed=100 + s, sex_signal=False)
        m, h = evaluate.load_run(r, w)
        ps.append(evaluate.sex_flight_probe(m, h, sub / "ref.csv", n_perm=19, n_repeats=1, seed=s, min_windows=5)["flight"]["p_auc"])
    ps = np.array(ps)
    assert 0.25 < ps.mean() < 0.75 and (ps <= 0.05).sum() <= 2, ps


def test_sex_probe_uses_k2_states_and_picks_fast_state(tmp_path):
    win, run, _ = _make(tmp_path, n_birds=12, n_win=40)
    meta, H = evaluate.load_run(run, win)
    fly = (meta["speed_mean"].to_numpy() > 5) | (meta["label_flapping"].to_numpy() == 1)
    st = pd.DataFrame({"bird": meta["bird"], "row": meta.groupby("bird").cumcount(), "state_hmm": np.where(fly, 0, 1)})
    (tmp_path / "st").mkdir()
    st.to_parquet(tmp_path / "st" / "states.parquet")
    res = evaluate.sex_flight_probe(meta, H, tmp_path / "ref.csv", tmp_path / "st", n_perm=19, n_repeats=1, min_windows=5)
    assert res["flight_state"] == 0 and res["state_rows_unmatched"] == 0
    assert res["flight"]["auc"] > 0.9


def test_main_only_r7_writes_section_file(tmp_path):
    win, run, _ = _make(tmp_path, n_birds=12, n_win=90)
    evaluate.main(["--run", str(run), "--windows", str(win), "--events", str(tmp_path / "events"), "--only", "r7",
                   "--reference", str(tmp_path / "ref.csv"), "--n-perm", "9", "--n-repeats", "1",
                   "--r7-n-train", "200", "--r7-n-test", "200"])
    assert not (run / "evaluation.json").exists()
    r7 = json.loads((run / "evaluation_r7.json").read_text())["r7_probes"]
    assert r7["window_holdout"]["full"]["pitch_rot_deg"]["r2"] > 0.5
    assert r7["sex_flight"]["flight"]["n_perm"] == 9


def _load_compare():
    spec = importlib.util.spec_from_file_location("compare_runs", Path(__file__).parents[1] / "scripts" / "compare_runs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_runs_aggregates_seeds_and_flags(tmp_path):
    cr = _load_compare()
    vals = {"base": [0.80, 0.82, 0.81], "r7_full": [0.70, 0.71, 0.69], "r7_lite": [0.81, 0.80, 0.83]}
    runs = []
    for v, xs in vals.items():
        for s, x in enumerate(xs):
            d = tmp_path / f"{v}_s{s}"
            d.mkdir()
            ev = {"window_level_holdout": {"label_flapping": {"balanced_accuracy": x, "n_test": 100}},
                  "nuisance_vs_features": {"bird_id_holdout_h": 1 - x, "bird_id_holdout_h_ci95": [0.1, 0.3],
                                           "bird_id_holdout_h_recall_by_bird": {"X00": 0.5}},
                  "holdout_birds": ["X00"]}
            if not (v == "r7_lite" and s == 2):  # a run without the r7 section
                (d / "evaluation_r7.json").write_text(json.dumps({"r7_probes": {"sex_flight": {"flight": {"auc": x}}}}))
            (d / "evaluation.json").write_text(json.dumps(ev))
            runs.append(d)
    single = tmp_path / "r4_v4"
    single.mkdir()
    (single / "evaluation.json").write_text(json.dumps({"window_level_holdout": {"label_flapping": {"balanced_accuracy": 0.9}}}))
    out = tmp_path / "cmp" / "table"
    T = cr.main([str(r) for r in runs + [single]] + ["--baseline", "base", "--out", str(out)])
    assert (tmp_path / "cmp" / "table.md").exists() and (tmp_path / "cmp" / "table.csv").exists()
    assert not T.metric.str.contains("recall_by_bird|holdout_birds").any()
    row = lambda m, v: T[(T.metric == m) & (T.variant == v)].iloc[0]
    ba = "window_level_holdout.label_flapping.balanced_accuracy"
    r = row(ba, "r7_full")
    assert r.n_runs == 3 and abs(r["mean"] - 0.70) < 1e-9 and abs(r.sd - 0.01) < 1e-9
    assert r.sd_rule == "paired" and abs(r["diff"] + 0.11) < 1e-9 and r.worse_2sd
    assert not row(ba, "r7_lite").worse_2sd
    idn = row("nuisance_vs_features.bird_id_holdout_h", "r7_full")
    assert idn.direction == "lower" and idn.worse_2sd  # identity rose (1 - x): worse for a lower-is-better metric
    assert row("window_level_holdout.label_flapping.n_test", "r7_full").direction == "none"
    sx = row("r7_probes.sex_flight.flight.auc", "r7_lite")
    assert sx.n_runs == 2 and sx.n_pairs == 2
    s4 = row(ba, "r4_v4")
    assert s4.n_runs == 1 and np.isnan(s4.sd) and s4.sd_rule == "none" and not s4.worse_2sd
    assert "bird_id_holdout_h_ci95.lo" in " ".join(T.metric)
    with pytest.raises(SystemExit):  # refuses to overwrite
        cr.main([str(r) for r in runs] + ["--baseline", "base", "--out", str(out)])
