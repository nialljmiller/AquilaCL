"""Scorecard (aquilacl.scorecard, scripts/compare_scorecards.py, scripts/scorecard_precount.py) on small synthetic data."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from aquilacl import scorecard as sc
from aquilacl.windows import GPS_CHANNELS


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- statistics

@pytest.mark.parametrize("case", ["perfect", "mean", "noisy_weighted"])
def test_circular_r2(case):
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 400)
    Y = sc.angle_xy(th)
    w = rng.uniform(0.5, 2, 400) if case == "noisy_weighted" else None
    P = {"perfect": Y, "mean": np.tile(np.average(Y, 0, w), (400, 1)),
         "noisy_weighted": Y + rng.normal(0, 0.3, Y.shape)}[case]
    r2 = sc.circular_r2(th, P, w)
    ww = np.ones(400) if w is None else w
    direct = 1 - (ww * ((P - Y) ** 2).sum(1)).sum() / (ww * ((Y - np.average(Y, 0, ww)) ** 2).sum(1)).sum()
    assert r2 == pytest.approx(direct)
    assert r2 == pytest.approx({"perfect": 1.0, "mean": 0.0}.get(case, r2))
    # per-bird sufficient statistics recombine exactly, and multiplicity 2 equals duplicating a bird
    birds = np.repeat(np.arange(4), 100)
    S = np.array([sc.r2_stats(Y[birds == b], P[birds == b], ww[birds == b]) for b in range(4)])
    assert sc.r2_from_stats(S) == pytest.approx(direct)
    dup = np.r_[np.arange(400), np.flatnonzero(birds == 0)]
    assert sc.r2_from_stats(S, np.array([2, 1, 1, 1])) == pytest.approx(
        sc.r2_from_stats(np.array([sc.r2_stats(Y[dup], P[dup], ww[dup])])))


def test_weighted_auc_matches_sklearn_and_duplication():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(y + rng.normal(0, 1, 300), 1)  # ties
    assert sc.weighted_auc(y, s) == pytest.approx(roc_auc_score(y, s))
    w = rng.integers(0, 3, 300)
    rep = np.repeat(np.arange(300), w)
    assert sc.weighted_auc(y, s, w) == pytest.approx(roc_auc_score(y[rep], s[rep]))


def _fix(speed, climb=0.0, turn=0.0, n=10):
    g = np.zeros((1, 10, len(GPS_CHANNELS)), np.float32)
    gm = np.zeros(g.shape, bool)
    g[0, :, GPS_CHANNELS.index("ground-speed")] = speed
    g[0, :, GPS_CHANNELS.index("climb_ms")] = climb
    g[0, :, GPS_CHANNELS.index("turn_rate_dps")] = turn
    gm[0, :n] = True
    return g, gm


@pytest.mark.parametrize("speed,climb,turn,n,expect", [
    (0.5, 0, 0, 10, {"perched"}),
    (6.0, 0, 20, 10, {"circling"}),  # 10 fixes x 20 deg = 200 deg
    (6.0, 1.0, 0, 10, {"climbing_straight"}),
    (10.0, -1.0, 0, 10, {"gliding"}),
    (16.0, -6.0, 0, 10, {"gliding", "steep_descent"}),  # labels overlap
    (np.r_[[1.0] * 5, [8.0] * 5], 0, 0, 10, {"take_off"}),
    (np.r_[[8.0] * 5, [1.0] * 5], 0, 0, 10, {"landing"}),
    (0.5, 0, 0, 7, set()),  # fewer than 8 valid fixes
])
def test_gps_labels(speed, climb, turn, n, expect):
    L, ok = sc.gps_labels(*_fix(speed, climb, turn, n))
    assert {lab for lab, v in zip(sc.GPS_LABELS, L[0]) if v} == expect
    assert ok[0] == (n >= 8)


@pytest.mark.parametrize("k,n_acc,sun,filled,inside", [
    (0, 40, 30.0, False, True),    # 40-sample scheduled burst belongs to the scheduled domain (spec note 2026-11-27)
    (0, 64, 30.0, False, True),
    (0, 200, 30.0, False, True),
    (0, 31, 30.0, False, False),   # below SCHED_MIN
    (0, 40, -5.0, False, False),   # night
    (1, 200, 30.0, False, False),  # continuous mode
    (0, 40, 30.0, True, False),    # local solar hour filled, not measured
])
def test_scheduled_domain(k, n_acc, sun, filled, inside):
    F = pd.DataFrame({"sensitivity_k": [k], "acc_samples": [n_acc], "sun_elev_deg": [sun], "lsh_filled": [filled]})
    assert bool(sc.day_mask(type("C", (), {"F": F})())[0]) == inside


# ---------------------------------------------------------------- selection rule

def _cards(tmp, delta, noise=0.005, cand_flags=(), floors=(None, None), guards=(0.8, 0.8), g_flags=()):
    """5 seeds of a baseline and a candidate with one selection row (G1, per-bird bootstrap inputs) and G10."""
    rng = np.random.default_rng(2)
    birds = [f"x{i}" for i in range(12)]
    out = []
    for s in range(5):
        base = rng.normal(0, 0.002, 12) + 0.7
        seed_noise = rng.normal(0, noise)  # seed-to-seed variation of the candidate
        for j, (name, d) in enumerate((("base", 0.0), ("cand", delta + seed_noise))):
            per = {b: float(v + d) for b, v in zip(birds, base)}
            rows = {"G1": {"kind": "S", "better": "up", "min_effect": 0.03, "model": float(np.mean(list(per.values()))),
                           "floor": floors[j], "flags": list(cand_flags) if j else [],
                           "boot": {"type": "mean_birds", "birds": per}},
                    "G10": {"kind": "guard", "better": "up", "min_effect": None, "flags": list(g_flags) if j else [],
                            "model": {t: guards[j] for t in sc.G10_TARGETS}}}
            out.append({"run": f"{name}_s{s}", "rows": rows, "spec": sc.SPEC, "final": False, "split_sha256": "a",
                        "injections": {"sha256": "b"}, "reference_cache": "/x/search_c", "_base": tmp,
                        "_seed": str(s), "_variant": name})
    return out


@pytest.mark.parametrize("kw,row_verdict,overall", [
    ({"delta": 0.10}, "better", "better"),
    ({"delta": -0.10}, "worse", "worse"),
    ({"delta": 0.10, "noise": 0.20}, "tie", "tie"),          # seed noise: mean difference < 2 paired sd
    ({"delta": 0.01, "noise": 0.0005}, "tie", "tie"),        # below the 0.03 minimum effect
    ({"delta": 0.10, "cand_flags": ["contaminated: run trained on search birds"]}, "n/a", "tie"),
    ({"delta": 0.10, "floors": (0.70, 0.79)}, "suspended", "tie"),  # suspended candidate cannot be better (I4)
    ({"delta": -0.10, "floors": (0.55, 0.59)}, "worse", "worse"),   # ... but can be worse (I4)
    ({"delta": 0.10, "guards": (0.8, 0.75)}, "better", "tie"),      # guard rail drops > 0.02
    ({"delta": 0.10, "guards": (0.8, float("nan"))}, "better", "tie"),  # guard n/a blocks better (I3)
    ({"delta": 0.10, "g_flags": ["error"]}, "better", "tie"),            # guard with a bad flag (I3)
    ({"delta": 0.10, "cand_flags": ["error"]}, "n/a", "incomplete"),    # error on a selection row (I6)
])
def test_selection_rule(tmp_path, kw, row_verdict, overall):
    cs = _script("compare_scorecards")
    res = cs.compare(_cards(tmp_path, **kw), "base", n_boot=200)["cand"]
    assert res["rows"][0]["verdict"] == row_verdict
    assert res["overall"] == overall and not res["provisional"]


@pytest.mark.parametrize("key,value", [("spec", "other"), ("final", True), ("split_sha256", "z"),
                                       ("injections", {"sha256": "z"}), ("reference_cache", "/x/search_d")])
def test_compare_refuses_mixed_scorecards(tmp_path, key, value):
    cs = _script("compare_scorecards")
    cards = _cards(tmp_path, 0.1)
    cards[3][key] = value
    with pytest.raises(SystemExit):
        cs.compare(cards, "base", n_boot=10)


# ---------------------------------------------------------------- injections: circular gap, freezing

@pytest.mark.parametrize("a,b,gap", [(0, 200, 165.25), (0, 365, 0.25), (10, 375, 0.25), (0, 120, 120.0),
                                      (0, 250, 115.25), (100, 100 + 2 * 365.25, 0.0)])
def test_circular_day_gap(a, b, gap):
    assert sc.circular_day_gap(a, b) == pytest.approx(gap)
    assert (sc.circular_day_gap(a, b) >= 120) == (gap >= 120)


def test_injections_frozen(tmp_path):
    p = tmp_path / "injections.parquet"
    with pytest.raises(SystemExit):
        sc.load_injections(p)  # the scorecard refuses without the frozen list and never creates it
    df = pd.DataFrame({"set": ["search"], "kind": ["shuffle"], "bird": ["x"], "bin": [0], "host_rows": [[1, 2]],
                       "donor_bird": ["x"], "donor_rows": [[2, 1]], "host_day": [3], "donor_day": [-1]})
    cand = tmp_path / "cand.parquet"
    sc.write_injections(cand, df)
    (tmp_path / "split.json").write_text("{}")
    pc = _script("scorecard_precount")
    pc.main(["--freeze", str(cand), "--split", str(tmp_path / "split.json")])
    before = p.read_bytes()
    with pytest.raises(SystemExit):
        pc.main(["--freeze", str(cand), "--split", str(tmp_path / "split.json")])
    with pytest.raises(FileExistsError):
        sc.write_injections(p, df.assign(bird="y"))
    got, info = sc.load_injections(p)
    assert p.read_bytes() == before and got["bird"].tolist() == ["x"]


def test_reference_key_and_failed_reference(world, small, monkeypatch, tmp_path):
    ctx = sc.build_context(world["W"], world["split"], world["ref"], world["states2"], world["res"], False)
    args = type("A", (), {"windows": world["W"], "states2": world["states2"], "residency": world["res"],
                          "reference": world["ref"]})()
    k0 = sc.reference_key(ctx, args)
    monkeypatch.setattr(sc, "SCHED_MIN", 64)
    assert sc.reference_key(ctx, args) != k0  # every row constant is in the key (I1)
    monkeypatch.setitem(sc.ROW_FN, "G2", lambda *a: 1 / 0)
    with pytest.raises(SystemExit):
        sc.reference_rows(ctx, ["G2"], tmp_path / "ref")
    assert not list((tmp_path / "ref").glob("*.json"))  # a failed reference row caches nothing (I2)


# ---------------------------------------------------------------- synthetic data set and end-to-end run

N_DAYS, DAY_STEP = 125, 3
BIRDS = [f"b{i:02d}" for i in range(11)]
TRAIN, SEARCH, LOCKED = BIRDS[:6], BIRDS[6:9], BIRDS[9:]


def _bird(i, rng):
    meta = []
    seg = 0
    t0 = pd.Timestamp("2024-05-01")
    for d in range(N_DAYS):
        day = t0 + pd.Timedelta(days=d * DAY_STEP)
        for j in range(36):  # scheduled bursts every 40 min, local solar hour = UTC - 7
            lsh = j * 40 / 60
            meta.append((day + pd.Timedelta(hours=lsh + 7), seg, 0, 1, 0, 30.0 if 6 <= lsh <= 18 else -20.0, 200, 0, lsh))
            seg += 1
        for k in range(12):  # one continuous-mode bout at local noon, 10 s steps
            lsh = 12 + k * 10 / 3600
            meta.append((day + pd.Timedelta(hours=lsh + 7), seg, k, 12, 1, 50.0, 200, 10, lsh))
        seg += 1
    n = len(meta)
    M = pd.DataFrame(meta, columns=["t_start", "segment", "k_in_segment", "segment_windows", "sensitivity_k",
                                    "sun_elev_deg", "acc_samples", "gps_fixes", "local_solar_hour"])
    M.insert(0, "tag", f"t{i}")
    M.insert(0, "bird", BIRDS[i])
    M["decoding"] = "ok"
    k1 = M["sensitivity_k"].to_numpy() == 1
    t = np.arange(200) / 20
    A = np.tile(np.array([0.1, 0.05, 0.99], np.float32), (n, 200, 1)) + rng.normal(0, 0.01, (n, 200, 3))
    A[k1, :, 1] += 0.4 * np.sin(2 * np.pi * (3 + 0.1 * (i % 2)) * t)
    G = np.zeros((n, 10, len(GPS_CHANNELS)), np.float32)
    GM = np.zeros(G.shape, bool)
    day_idx = np.repeat(np.arange(N_DAYS), 48)
    circ = ((day_idx + i) % 2 == 0)
    sp = 10 + (i % 2) + rng.normal(0, 1, (n, 10))
    if i < 2:  # two training birds also perch during some bouts: a mix the search birds never show (novel_mix)
        sp[np.repeat(np.arange(N_DAYS) % 3 == 0, 48)] = 0.5
    G[k1, :, GPS_CHANNELS.index("ground-speed")] = sp[k1]
    G[k1, :, GPS_CHANNELS.index("speed_track_ms")] = sp[k1]
    G[k1, :, GPS_CHANNELS.index("climb_ms")] = np.where(circ[k1, None], 1.0, -1.0)
    G[k1, :, GPS_CHANNELS.index("turn_rate_dps")] = np.where(circ[k1, None], 25.0, 0.0)
    G[k1, :, GPS_CHANNELS.index("step_m")] = sp[k1]
    G[k1, :, GPS_CHANNELS.index("eobs:horizontal-accuracy-estimate")] = 5.0
    GM[k1] = True
    dyn = A - A.mean(1, keepdims=True)
    vedba = np.linalg.norm(dyn, axis=2).mean(1)
    f = pd.DataFrame({"vedba": vedba, "vedba_sd": vedba * 0.3, "odba": vedba * 1.4, "static_norm": 1.0 + rng.normal(0, 0.01, n),
                      "pitch_deg": rng.normal(5, 2, n), "roll_deg": rng.normal(3, 2, n),
                      "flap_band_frac": np.where(k1, 0.6, 0.1), "dom_freq_hz": np.where(k1, 3.0 + 0.1 * (i % 2), 1.0),
                      "speed_mean": np.where(k1, sp.mean(1), np.nan), "speed_max": np.where(k1, sp.max(1), np.nan),
                      "climb_mean": np.where(k1, np.where(circ, 1.0, -1.0), np.nan), "climb_min": np.where(k1, -1.0, np.nan),
                      "turn_abs_mean": np.where(k1, np.where(circ, 25.0, 1.0), np.nan),
                      "turn_net_deg": np.where(k1, np.where(circ, 250.0, 5.0), np.nan),
                      "path_m": np.where(k1, sp.sum(1), np.nan), "hacc_mean": np.where(k1, 5.0, np.nan),
                      "n_speed": np.where(k1, 10, 0), "n_turn": np.where(k1, 9, 0),
                      "label_night": (M["sun_elev_deg"] < -6).astype(float), "label_perched": (~k1).astype(float),
                      "label_flapping": k1.astype(float), "label_circling": (k1 & circ).astype(float),
                      "label_gliding": (k1 & ~circ).astype(float), "label_climbing_soar": (k1 & circ).astype(float)})
    o = np.argsort(M["t_start"].to_numpy(), kind="stable")  # meta.parquet is in time order
    M, f = M.iloc[o].reset_index(drop=True), f.iloc[o].reset_index(drop=True)
    return M, A[o].astype(np.float16), np.ones((n, 200), bool), G[o], GM[o], f


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("sc")
    W = root / "windows"
    rng = np.random.default_rng(3)
    run = root / "runs" / "rA_s0"
    (run / "embeddings").mkdir(parents=True)
    states, orient = [], []
    for i, b in enumerate(BIRDS):
        M, A, AM, G, GM, f = _bird(i, rng)
        d = W / b
        d.mkdir(parents=True)
        M.to_parquet(d / "meta.parquet", index=False)
        f.to_parquet(d / "features.parquet", index=False)
        for nm, x in (("acc", A), ("acc_mask", AM), ("gps", G), ("gps_mask", GM)):
            np.save(d / f"{nm}.npy", x)
        k1 = M["sensitivity_k"].to_numpy() == 1
        h = rng.normal(0, 1, (len(M), 128)) + 2 * k1[:, None] * np.sin(np.arange(128))[None] + 0.5 * (i % 2)
        np.savez(run / "embeddings" / f"{b}.npz", h=h.astype(np.float16))
        states.append(pd.DataFrame({"bird": b, "row": np.arange(len(M)), "state_hmm": (k1 | (rng.random(len(M)) < 0.2)).astype(int),
                                    "roll_rot_deg": rng.normal(-60, 5, len(M))}))
        orient.append({"bird": b, "R": np.eye(3).ravel().tolist()})
    st = root / "states_k2"
    st.mkdir()
    pd.concat(states).to_parquet(st / "states.parquet", index=False)
    import torch

    from aquilacl.model import Encoder
    torch.manual_seed(0)
    torch.save({"model": Encoder(len(GPS_CHANNELS)).state_dict()}, run / "ckpt_0000001.pt")
    (run / "config.json").write_text(json.dumps({"step_g": 0.002, "clip_g": 1.8, "mask_gps_channels": [
        "eobs:horizontal-accuracy-estimate"], "derotate_yaw": False}))
    (run / "orientation.json").write_text(json.dumps(orient))
    (run / "holdout.json").write_text(json.dumps(SEARCH))
    runB = root / "runs" / "rB_s0"  # time + env informed, trained on one search bird
    runB.mkdir()
    for p in run.iterdir():
        if p.is_file():
            (runB / p.name).write_bytes(p.read_bytes())
    (runB / "embeddings").symlink_to(run / "embeddings")
    (runB / "holdout.json").write_text(json.dumps(SEARCH[:2]))
    (runB / "design.json").write_text(json.dumps({"uses_time": True, "uses_env": True, "level": 1, "parent_run": None}))
    sp = root / "split"
    sp.mkdir()
    (sp / "split.json").write_text(json.dumps({"train": TRAIN, "search": SEARCH, "locked": LOCKED}))
    ref = pd.DataFrame({"animal-id": BIRDS, "deploy-on-date": ["2024-05-01"] * len(BIRDS),
                        "animal-sex": ["f" if i % 2 else "m" for i in range(len(BIRDS))],
                        "animal-comments": ["nestling" if i == 1 else "" for i in range(len(BIRDS))],
                        "deployment-comments": [""] * len(BIRDS), "tag-model": ["m1"] * len(BIRDS)})
    ref.to_csv(root / "ref.csv", index=False)
    res = root / "residency"
    res.mkdir()
    pd.DataFrame({"bird": BIRDS, "strategy": ["migrant" if i % 2 else "nomad" for i in range(len(BIRDS))]}).to_csv(
        res / "residency.csv", index=False)
    return {"root": root, "W": W, "run": run, "runB": runB, "split": sp / "split.json", "ref": root / "ref.csv",
            "states2": st, "res": res}


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setitem(sc.CAPS, "G2", 100)
    for k, v in (("G6_train", 40), ("G6_test", 60), ("G9b", 40), ("G10_train", 60), ("G10_test", 80),
                 ("G7_bout_hosts", 30), ("G7_day_hosts", 20), ("G7_pca_per_bird", 200)):
        monkeypatch.setitem(sc.CAPS, k, v)
    monkeypatch.setattr(sc, "G8C", {"per_bird": 150, "eval_per_bird": 60, "k_over": 6, "n_boot": 2, "ks": [2, 3, 4],
                                    "sub_k_over": 3, "sub_ks": [2], "sub_per_bird": 60})
    monkeypatch.setattr(sc, "G3_MIN_WINDOWS", 10)
    monkeypatch.setattr(sc, "G4_MIN_WINDOWS", 50)
    monkeypatch.setattr(sc, "G1_MIN_FLIGHT", 20)
    monkeypatch.setattr(sc, "N_BOOT_JACCARD", 5)
    monkeypatch.setattr(sc, "G6_MIN_POS", 10)


def _args(w, out, *extra, runs=("run", "runB")):
    a = []
    for r in runs:
        a += ["--run", str(w[r])]
    return a + ["--windows", str(w["W"]), "--split", str(w["split"]), "--reference", str(w["ref"]), "--states2",
                str(w["states2"]), "--residency", str(w["res"]), "--out", str(out), "--n-perm", "5", "--workers", "1",
                *extra]


def _no_locked_ids(paths):
    for p in paths:
        for f in [p] if p.is_file() else [x for x in p.rglob("*") if x.is_file()]:
            txt = f.read_text(errors="ignore") if f.suffix in (".json", ".md") else \
                pd.read_parquet(f).astype(str).to_csv() if f.suffix == ".parquet" else ""
            assert not any(b in txt for b in LOCKED), f"locked bird id in {f.name}"


def _precount_and_freeze(w):
    inj = w["split"].parent / "injections.parquet"
    if inj.exists():
        return inj
    pc = _script("scorecard_precount")
    pc_out = w["root"] / "precount"
    pc.main(["--windows", str(w["W"]), "--split", str(w["split"]), "--reference", str(w["ref"]), "--states2",
             str(w["states2"]), "--residency", str(w["res"]), "--out", str(pc_out), "--n-perm", "5"])
    pc.main(["--freeze", str(pc_out / "injections_candidate.parquet"), "--split", str(w["split"])])
    return inj


def test_end_to_end_cli(world, small):
    out = world["root"] / "scores" / "cardA"
    if not (world["split"].parent / "injections.parquet").exists():
        with pytest.raises(SystemExit):
            sc.main(_args(world, out))  # refuses without the frozen list
    inj = _precount_and_freeze(world)
    pre = json.loads((world["root"] / "precount" / "precount.json").read_text())
    D = pd.read_parquet(inj)
    assert set(D["set"]) <= {"search", "locked"} and not D.empty
    d = D[D["kind"] == "other_season"]
    assert len(d) and (sc.circular_day_gap(d["host_day"], d["donor_day"]) >= 120).all()  # C1
    st = pre["injections_candidate"]["counts"]["search"]
    assert st["day_hosts_scorable"] <= st["day_hosts_listed"] and st["bout_hosts_scorable"] <= st["bout_hosts_listed"]
    sc.main(_args(world, out))
    A = json.loads((out / "rA_s0" / "scorecard.json").read_text())
    B = json.loads((out / "rB_s0" / "scorecard.json").read_text())
    assert set(A["rows"]) == set(sc.ROWS) and (out / "rA_s0" / "scorecard.md").exists()
    errs = {r: e["details"].get("error") for r, e in A["rows"].items() if "error" in e["flags"]}
    assert not errs, errs
    for r in ("G1", "G2", "G4", "G6", "G9", "G10", "G7a", "G7b", "G7c", "G7d", "G8", "G8b"):
        assert A["rows"][r]["model"] is not None, r
    assert A["rows"]["G9"]["n_birds"] == len(SEARCH)
    # flags: time / environment / contamination on run B only, on the rows the spec names
    assert not any("not comparable" in f or "contaminated" in f for e in A["rows"].values() for f in e["flags"])
    assert "time/position-informed: not comparable" in B["rows"]["G2"]["flags"]
    assert "environment-informed: not comparable" in B["rows"]["G9"]["flags"]
    assert not any("not comparable" in f for f in B["rows"]["G1"]["flags"])
    assert "contaminated: run trained on search birds" in B["rows"]["G6"]["flags"]
    assert not any("contaminated" in f for f in B["rows"]["G1"]["flags"])
    # locked birds never appear in a score, a bootstrap file or the reference cache
    ref_dir = out.parent / "scorecard_reference"
    _no_locked_ids([out, ref_dir])
    # references are reused, never recomputed or overwritten; a non-empty output directory is refused
    files = {p: p.stat().st_mtime_ns for p in ref_dir.rglob("*.json")}
    out2 = world["root"] / "scores" / "cardA2"
    sc.main(_args(world, out2, "--rows", "G1", "G2", runs=("run",)))
    assert {p: p.stat().st_mtime_ns for p in ref_dir.rglob("*.json")} == files
    A2 = json.loads((out2 / "scorecard.json").read_text())
    assert A2["rows"]["G2"]["model"] == pytest.approx(A["rows"]["G2"]["model"])
    with pytest.raises(SystemExit):
        sc.main(_args(world, out2, runs=("run",)))
    # compare: the same run as two "variants" ties on every row
    cs = _script("compare_scorecards")
    cards = cs.load_cards([out / "rA_s0", out2])
    for c, v in zip(cards, ("x", "y")):
        c["_variant"] = v
    assert cs.compare(cards, "x", n_boot=50)["y"]["overall"] == "tie"


def test_final_mode_scores_locked_birds(world, small):
    _precount_and_freeze(world)
    out = world["root"] / "scores" / "final"
    sc.main(_args(world, out, "--final", "--rows", "G1", "G2", "G9", runs=("run",)))
    c = json.loads((out / "scorecard.json").read_text())
    assert c["final"] and c["rows"]["G9"]["n_birds"] == len(LOCKED)
    assert sorted(c["rows"]["G2"]["boot"]["birds"]) == LOCKED
    assert "contaminated: run trained on locked birds" in c["rows"]["G2"]["flags"]  # no locked.json in the run


def test_encode_rows_reproduces_training_input(world):
    """GPS kept: encode_rows must give the h of dataset.load + train.encode (same frame and format handling)."""
    from aquilacl import dataset, evaluate
    from aquilacl.train import encode

    run = world["run"]
    cfg = json.loads((run / "config.json").read_text())
    birds = SEARCH[:2]
    data = dataset.load(world["W"], birds=birds, step_g=cfg["step_g"], clip_g=cfg["clip_g"],
                        mask_gps_channels=tuple(cfg["mask_gps_channels"]))
    orient = run / "orientation.json"
    saved = orient.read_bytes()
    try:
        orient.write_text(json.dumps(data.orientation[["bird", "R"]].to_dict("records")))
        F = sc.load_frame(world["W"], birds)
        rows = np.r_[np.arange(0, 40), np.arange(len(F) - 40, len(F))]
        model = evaluate._latest_model(run, "cpu")
        ref = encode(model, data.acc[rows], data.acc_mask[rows], data.gps[rows], data.gps_mask[rows], "cpu")[0]
        got = sc.encode_rows(run, world["W"], F, rows, "cpu", keep_gps=True)
        assert np.allclose(got, ref, atol=1e-4)
        z = sc.encode_rows(run, world["W"], F, rows, "cpu")
        assert not np.allclose(z[40:], ref[40:], atol=1e-3)  # GPS removed changes continuous-mode windows
    finally:
        orient.write_bytes(saved)


def test_level2_paths_with_identity_context(world, small, monkeypatch):
    """Level-2 wiring (GPS-zeroed parent re-embedding -> seq.contextual_embed; G7 injections in the parent stream),
    with contextual_embed replaced by the identity so the expected values are known."""
    from aquilacl import seq

    calls = []

    def ident(run_dir, h_by_bird, meta_by_bird, device="cpu", notime=False):
        calls.append((len(h_by_bird), notime))
        for m in meta_by_bird.values():
            assert m["t_start"].is_monotonic_increasing and list(m.columns) == seq.META_COLS
        return {k: np.asarray(v, np.float32) for k, v in h_by_bird.items()}

    monkeypatch.setattr(seq, "contextual_embed", ident)
    _precount_and_freeze(world)
    # cheap stand-in for the GPS-zeroed re-embedding (the real one is tested in test_encode_rows_...)
    monkeypatch.setattr(sc, "encode_rows", lambda run, W, F, rows, dev, **k: np.c_[
        F[sc.ACC_FEATURES].to_numpy(np.float32)[rows], np.zeros((len(rows), 120), np.float32)])
    l2 = world["root"] / "runs" / "rL2_s0"
    l2.mkdir()
    (l2 / "embeddings").symlink_to(world["run"] / "embeddings")
    (l2 / "holdout.json").write_text(json.dumps(SEARCH))
    (l2 / "config.json").write_text(json.dumps({"max_len": 64}))
    (l2 / "design.json").write_text(json.dumps({"uses_time": False, "uses_env": False, "level": 2,
                                                "parent_run": str(world["run"])}))
    out = world["root"] / "scores" / "l2"
    sc.main(_args(world, out, "--rows", "G6", "G7a", "G7c", "G7d", runs=("run",)))
    out2 = world["root"] / "scores" / "l2b"
    sc.main([x if x != str(world["run"]) else str(l2)
             for x in _args(world, out2, "--rows", "G6", "G7a", "G7c", "G7d", runs=("run",))])
    A = json.loads((out / "scorecard.json").read_text())["rows"]
    L = json.loads((out2 / "scorecard.json").read_text())["rows"]
    assert calls and any("parent run" in f for f in L["G6"]["flags"])
    assert L["G6"]["model"] == pytest.approx(A["G6"]["model"], abs=1e-3)  # identity context: same as the parent
    assert L["G7d"]["model"] == pytest.approx(A["G7d"]["model"], abs=0.05)  # clean queries come from crops
    assert L["G7c"]["model"] is not None and L["G7a"]["model"] is not None
