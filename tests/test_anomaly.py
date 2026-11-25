import json
import re

import numpy as np
import pandas as pd
import pytest

from aquilacl import anomaly as an
from aquilacl.windows import GPS_CHANNELS

# ---------------------------------------------------------------- synthetic data

N_BIRDS, N_DAYS, DIM = 6, 150, 32
G_DIR = an.unit(np.array([0.1, 0.05, 0.99]))
PLANT_BOUT = ("B0", 40)  # bird, day index of the bout whose h is far from everything
PLANT_DAY = ("B1", 60)  # bird, day index whose scheduled windows look like flight
PLANT_STUCK = ("B2", 50)  # bird, day index whose scheduled windows have a stuck x axis
PLANT_SLIP = ("B3", 100, 40.0)  # bird, first day index of a 40 deg tag slip


def _q(a, step=0.002):
    return (np.round(a / step) * step).astype(np.float16)


def _bird(i, rng, vecs):
    """Scheduled hourly full windows (still; night = local hours 22-4) and one continuous bout per day (12-20 full
    windows, 90% flapping with GPS, then a 64-sample tail); h from fixed state vectors plus noise."""
    b = f"B{i}"
    t0 = pd.Timestamp("2025-03-01", tz="UTC")
    meta, acc, am, gps, gm, h, st2, st6 = [], [], [], [], [], [], [], []
    seg = 0
    e1 = an.plane_basis(G_DIR)[0]
    for d in range(N_DAYS):
        Q = np.eye(3)
        if b == PLANT_SLIP[0] and d >= PLANT_SLIP[1]:
            Q = an.rot_axis(an.horizontal_axis(G_DIR, np.random.default_rng(99)), PLANT_SLIP[2])
        for hr in range(24):
            t = t0 + pd.Timedelta(days=d, hours=hr + 7)  # local solar hour = UTC - 7
            night = hr >= 22 or hr <= 4
            a = G_DIR + rng.normal(0, 0.003, (200, 3))
            if b == PLANT_STUCK[0] and d == PLANT_STUCK[1]:
                a[:, 0] = a[0, 0]
            acc.append(_q(a @ Q.T))
            am.append(np.ones(200, bool))
            gps.append(np.zeros((10, len(GPS_CHANNELS)), np.float32))
            gm.append(np.zeros((10, len(GPS_CHANNELS)), bool))
            meta.append({"t_start": t, "segment": seg, "k_in_segment": 0, "segment_windows": 1, "sensitivity_k": 0,
                         "acc_samples": 200, "gps_fixes": 0, "sun_elev_deg": -20.0 if night else 30.0,
                         "local_solar_hour": float(hr), "label_flapping": 0.0, "speed_mean": np.nan, "vedba": 0.01})
            v = vecs["rest"] if night else vecs["perch"]
            if b == PLANT_DAY[0] and d == PLANT_DAY[1]:
                v = vecs["flight2"]
            h.append(v + vecs[b] + rng.normal(0, 0.15, DIM))
            st2.append(1)
            st6.append(0 if night else 1)
            seg += 1
        n = 12 + (d * 7 + i) % 9
        mix = 2 + (d + i) % 2  # K=6 state of the bout: 2 or 3
        drift = rng.normal(0, 1, DIM)
        for k in range(n + 1):
            t = t0 + pd.Timedelta(days=d, hours=14 + 7, seconds=10 * k)
            tt = np.arange(200) / 20
            flap = k % 10 != 9
            a = G_DIR + rng.normal(0, 0.02, (200, 3))
            if flap:
                a += 0.6 * np.sin(2 * np.pi * 2.5 * tt)[:, None] * G_DIR + 0.25 * np.sin(2 * np.pi * 2.5 * tt + 0.5)[:, None] * e1
            ns = 64 if k == n else 200
            m = np.zeros(200, bool)
            m[:ns] = True
            a[~m] = 0
            Q = np.eye(3)
            if b == PLANT_SLIP[0] and d >= PLANT_SLIP[1]:
                Q = an.rot_axis(an.horizontal_axis(G_DIR, np.random.default_rng(99)), PLANT_SLIP[2])
            acc.append(_q(a @ Q.T))
            am.append(m)
            g = np.zeros((10, len(GPS_CHANNELS)), np.float32)
            g[:, 0] = rng.normal(12, 1.5, 10)
            g[:, 1] = g[:, 0] + rng.normal(0, 0.3, 10)
            g[:, 2] = rng.normal(0, 1, 10)
            g[:, 4] = g[:, 0]
            g[:, 5] = 5
            gps.append(g)
            gm.append(np.ones_like(g, bool))
            meta.append({"t_start": t, "segment": seg, "k_in_segment": k, "segment_windows": n + 1, "sensitivity_k": 1,
                         "acc_samples": ns, "gps_fixes": 10, "sun_elev_deg": 40.0, "local_solar_hour": 14.0,
                         "label_flapping": float(flap), "speed_mean": float(g[:, 0].mean()), "vedba": 0.4 if flap else 0.03})
            v = vecs[f"flight{mix}"] + (k / n - 0.5) * vecs["drift"] * 0.8 + 0.1 * drift
            if b == PLANT_BOUT[0] and d == PLANT_BOUT[1]:
                v = vecs["far"]
            h.append(v + vecs[b] + rng.normal(0, 0.1, DIM))
            st2.append(0)
            st6.append(mix)
        seg += 1
    m = pd.DataFrame(meta)
    return b, m, np.stack(acc), np.stack(am), np.stack(gps), np.stack(gm), an.norm_rows(np.stack(h)), st2, st6


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    rng = np.random.default_rng(0)
    vecs = {k: rng.normal(0, 1, DIM) for k in ("rest", "perch", "flight2", "flight3", "drift")}
    vecs["far"] = -3 * vecs["flight2"]
    for i in range(N_BIRDS):
        vecs[f"B{i}"] = rng.normal(0, 0.3, DIM)
    windows, run, s2, s6 = root / "windows", root / "runs" / "rtest", root / "st2", root / "st6"
    for p in (windows, run / "embeddings", s2, s6):
        p.mkdir(parents=True)
    S2, S6 = [], []
    for i in range(N_BIRDS):
        b, m, a, am, g, gm, h, st2, st6 = _bird(i, rng, vecs)
        d = windows / b
        d.mkdir()
        feats = ["label_flapping", "speed_mean", "vedba"]
        m.drop(columns=feats).assign(bird=b).to_parquet(d / "meta.parquet")
        m[feats].to_parquet(d / "features.parquet")
        np.save(d / "acc.npy", a)
        np.save(d / "acc_mask.npy", am)
        np.save(d / "gps.npy", g)
        np.save(d / "gps_mask.npy", gm)
        np.savez(run / "embeddings" / f"{b}.npz", h=h.astype(np.float16))
        S2.append(pd.DataFrame({"bird": b, "row": np.arange(len(m)), "state_hmm": st2}))
        S6.append(pd.DataFrame({"bird": b, "row": np.arange(len(m)), "state_hmm": st6}))
    pd.concat(S2).to_parquet(s2 / "states.parquet")
    pd.concat(S6).to_parquet(s6 / "states.parquet")
    (run / "orientation.json").write_text(json.dumps([{"bird": f"B{i}", "R": np.eye(3).ravel().tolist()}
                                                      for i in range(N_BIRDS)]))
    return {"windows": windows, "run": run, "s2": s2, "s6": s6, "root": root}


# ---------------------------------------------------------------- QC kernels


def _windows(rng, n=40, flap=False, gps=False):
    tt = np.arange(200) / 20
    a = G_DIR + rng.normal(0, 0.003 if not flap else 0.02, (n, 200, 3))
    if flap:
        e1 = an.plane_basis(G_DIR)[0]
        a = a + 0.6 * np.sin(2 * np.pi * 2.5 * tt)[None, :, None] * G_DIR + \
            0.25 * np.sin(2 * np.pi * 2.5 * tt + 0.5)[None, :, None] * e1
    a = _q(a).astype(np.float32)
    am = np.ones((n, 200), bool)
    g = np.zeros((n, 10, len(GPS_CHANNELS)), np.float32)
    gm = np.zeros(g.shape, bool)
    if gps:
        g[:, :, 0] = rng.normal(12, 1, (n, 10))
        g[:, :, 1] = g[:, :, 0]
        gm[:] = True
    return a, am, g, gm


def _refs(n):
    return {"g_roost": G_DIR, "basis": an.plane_basis(G_DIR), "rails": {0: np.full((3, 2), np.nan)},
            "night_ref_row": np.repeat(G_DIR[None], n, 0), "yaw_ref_row": np.zeros(n)}


@pytest.mark.parametrize("fault, check, flap, night", [
    ("stuck_axis", "stuck_axis", False, False),
    ("amplitude_half", "static_norm", False, False),
    ("tilt90", "tilt_night", False, True),
    ("yaw90", "yaw_flap", True, False),
    ("gps_speed_zero", "gps_flap_zero", True, False),
    ("gps_spike", "gps_impossible", True, False),
])
def test_qc_check_sees_its_fault_and_not_clean(fault, check, flap, night):
    rng = np.random.default_rng(1)
    a, am, g, gm = _windows(rng, flap=flap, gps=flap)
    n = len(a)
    kk, nt, rows = np.zeros(n, int), np.full(n, night), np.arange(n)
    R = _refs(n)
    e1 = R["basis"][0]
    # reference flapping axis: angle of e1 in the basis = 0
    clean = an.qc_eval(a, am, g, gm, kk, nt, rows, R)
    assert clean[f"elig_{check}"].mean() > 0.9, check
    assert not clean[list(an.QC_CHECKS)].any().any()
    if fault == "stuck_axis":
        a2 = an.fault_stuck(a, am, rng)
    elif fault == "amplitude_half":
        a2 = a * 0.5
    elif fault in ("tilt90", "yaw90"):
        a2 = an.fault_rotate(a, G_DIR, rng, yaw=fault == "yaw90")
    else:
        a2 = a
    g2 = an.fault_gps_zero(g, gm) if fault == "gps_speed_zero" else an.fault_gps_spike(g, gm, rng) if fault == "gps_spike" else g
    bad = an.qc_eval(a2, am, g2, gm, kk, nt, rows, R)
    assert bad.loc[bad[f"elig_{check}"], check].mean() > 0.95, (fault, check)
    assert e1 @ G_DIR == pytest.approx(0, abs=1e-9)


def test_saturation_and_rails():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1.0, (50, 200, 3)).astype(np.float32)
    a[0, :5, 0] = 1.9  # the tag's rail
    am = np.ones((50, 200), bool)
    kk = np.zeros(50, int)
    rails = an.rail_values(a, am, kk)
    assert rails[0][0, 1] == pytest.approx(max(1.9, a[:, :, 0].max()))
    sat = an.fault_saturate(a, am, kk, rails)
    assert (an.clip_share(sat, am, kk, rails) >= an.CLIP_SHARE).mean() > (an.clip_share(a, am, kk, rails) >= an.CLIP_SHARE).mean()


def test_roost_change_finds_slip_and_slip_injection():
    rng = np.random.default_rng(3)
    nights = np.arange(80)
    U = np.stack([an.unit(G_DIR + rng.normal(0, 0.02, 3)) for _ in nights])
    cp0, _ = an.roost_change(U)
    assert an.runs_above(cp0, an.ROOST_CP_DEG) == []
    Q = an.rot_axis(an.horizontal_axis(G_DIR, rng), 40)
    U2 = U.copy()
    U2[40:] = U2[40:] @ Q.T
    cp, ref = an.roost_change(U2)
    ev = an.runs_above(cp, an.ROOST_CP_DEG)
    assert len(ev) == 1 and ev[0][0] <= 40 <= ev[0][1] + 2
    assert np.isfinite(ref).all()
    mean = np.repeat(U, 6, 0) + rng.normal(0, 0.01, (480, 3))
    res = an.slip_injection((mean, np.repeat(nights, 6)), rng, n_per_bird=6)
    r = pd.DataFrame(res)
    assert r.loc[r["deg"] == 90, "detected"].all() and not r["clean_flag"].any()


def test_yaw_reference_and_axial_deviation():
    rng = np.random.default_rng(4)
    day = np.repeat(np.arange(60), 20)
    phi = np.mod(0.3 + rng.normal(0, 0.05, len(day)), np.pi)
    ref = an.yaw_reference(phi, day, np.array([0, 30, 59]))
    assert np.allclose(ref, 0.3, atol=0.03)  # day 0: whole record minus day 0
    assert an.axial_dev_deg(np.array([0.3 + np.pi / 2]), np.array([0.3]))[0] == pytest.approx(90)
    assert an.axial_dev_deg(np.array([np.pi - 0.01]), np.array([0.01]))[0] == pytest.approx(np.degrees(0.02))
    out = pd.DataFrame(an.yaw_slip_injection(pd.DataFrame({"day": day, "phi": phi}), rng, n_per_bird=5))
    assert out.loc[out["deg"] == 90, "detected"].all() and not out["clean_flag"].any()


# ---------------------------------------------------------------- behaviour kernels


def test_robust_z_and_knn_exclusions():
    rng = np.random.default_rng(5)
    g = np.r_[np.zeros(2000), np.ones(2000)].astype(int)
    x = np.r_[rng.normal(1, 0.1, 2000), rng.normal(5, 2.0, 2000)]
    x[[5, 2005]] = [3.0, 45.0]
    z, par = an.robust_z(x, g)
    thr = np.percentile(z, 99)
    assert z[5] > thr and z[2005] > thr
    assert np.allclose(an.apply_z(x[[5, 2005]], g[[5, 2005]], par), z[[5, 2005]])
    H = an.norm_rows(rng.normal(size=(600, 16)))
    day, seg = np.repeat(np.arange(60), 10), np.repeat(np.arange(120), 5)
    assert (an.kth_dist(H[:3], H, day[:3], day, seg[:3], seg, k=1) > 1e-3).all()
    sc, lev = an.score_groups(H, day, seg, np.repeat([0, 1], 300), k=5, cap=200, min_ref=50, seed=0)
    assert np.isfinite(sc).all() and set(np.unique(lev)) <= {0, 1, 2}


def test_bout_vectors_order_and_smoothness():
    rng = np.random.default_rng(6)
    t = np.linspace(0, 1, 20)[:, None]
    seq = an.norm_rows(np.c_[np.cos(t), np.sin(t), np.zeros((20, 6))] + rng.normal(0, 0.01, (20, 8)))
    m, d, s = an.bout_vectors(seq)
    m2, d2, s2 = an.bout_vectors(seq[::-1])
    assert np.allclose(m, m2, atol=1e-6) and np.allclose(d, -d2, atol=1e-6) and s == pytest.approx(s2)
    rough = an.norm_rows(rng.normal(size=(20, 8)))
    assert an.bout_vectors(rough)[2] < s
    assert list(an.bout_bin(np.array([5, 6, 15, 16, 45, 46, 84, 85, 89]))) == [-1, 0, 0, 1, 1, 2, 2, 3, 3]


def test_energy_distance_and_day_pairs():
    rng = np.random.default_rng(7)
    X, Y = rng.normal(size=(200, 4)), rng.normal(size=(300, 4))
    assert abs(an.energy_distance(X, Y)) < 0.05
    assert an.energy_distance(X + 1, Y) > 0.5
    # hour composition: the query day's hour-3 windows sit at +2, its hour-7 windows at 0; the other day matches
    # only when hours are paired one to one
    QH = np.r_[np.full(30, 3), np.full(30, 7)]
    QX = np.r_[np.full((30, 2), 2.0), np.zeros((30, 2))] + rng.normal(0, 0.1, (60, 2))
    YH = np.r_[np.full(30, 3), np.full(90, 7), np.full(20, 12)]
    Y = np.r_[np.full((30, 2), 2.0), np.zeros((90, 2)), np.full((20, 2), 9.0)] + rng.normal(0, 0.1, (140, 2))
    same = np.mean([an.day_pair_ed(QX, QH, Y, YH, rng) for _ in range(20)])
    Y2 = Y.copy()
    Y2[YH == 3] = 0.0  # the other day's hour-3 windows look like hour 7
    assert abs(same) < 0.25 and np.mean([an.day_pair_ed(QX, QH, Y2, YH, rng) for _ in range(20)]) > 1.0
    assert np.isnan(an.day_pair_ed(QX, QH, Y[YH == 12], YH[YH == 12], rng))
    day = np.repeat(np.arange(30), 24)
    hour = np.tile(np.arange(24), 30)
    Z = rng.normal(size=(720, 3))
    Z[day == 15] += 3
    sc = [an.score_day(Z, day, hour, np.flatnonzero(day == d), d, rng)[0] for d in range(30)]
    assert int(np.nanargmax(sc)) == 15


def test_partial_spearman_removes_confounder_and_mmd():
    rng = np.random.default_rng(8)
    c = rng.normal(size=80)
    x, y = c + rng.normal(0, 0.3, 80), c + rng.normal(0, 0.3, 80)
    rho, p, _ = an.perm_spearman(x, y, 199, rng)
    pr, pp, n = an.partial_spearman(x, y, c[:, None], 199, rng)
    assert rho > 0.7 and p <= 0.01 and abs(pr) < 0.3 and n == 80
    X, Y = rng.normal(size=(200, 5)), rng.normal(size=(300, 5))
    gam = an.median_gamma(np.r_[X, Y], rng)
    assert abs(an.mmd2(X, Y, gam)) < 0.01 and an.mmd2(X + 1.0, Y, gam) > 0.05


def test_factor_table_ratios():
    d = pd.DataFrame({"bird": np.repeat(["a", "b"], 50), "x": np.r_[np.zeros(80), np.ones(20)].astype(bool)})
    flag = np.zeros(100, bool)
    flag[80:90] = flag[:10] = True
    ft = an.factor_table(d, flag, {"x": (lambda q: q["x"], True)}).set_index("factor")
    assert ft.loc["x", "ratio"] == pytest.approx(0.5 / 0.2)
    assert an.worst_format_ratio(ft.reset_index()) == pytest.approx(2.5)


def test_blind_page_has_no_date_or_bird(tmp_path):
    rng = np.random.default_rng(9)
    e = {"title": "Bout 1: month Mar, 3 full windows", "t_rel": [0.0, 10.0, 20.0],
         "acc": [rng.normal(size=(200, 3)) for _ in range(3)], "gps_speed": [rng.normal(size=10)] * 3,
         "gps_climb": [rng.normal(size=10)] * 3}
    fig = an.blind_page([e], tmp_path / "p.png")
    texts = " ".join(t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text")))
    assert (tmp_path / "p.png").exists() and "Bout 1" in texts
    assert not re.search(r"\d{4}-\d{2}|\d{1,2}:\d{2}|B\d", texts)


# ---------------------------------------------------------------- end to end


def test_end_to_end_synthetic(synth, monkeypatch):
    monkeypatch.setattr(an, "L2_MIN_BIRDS", 5)  # 6 synthetic birds; the real rule is 20
    out = synth["root"] / "out"
    meta = an.run(synth["windows"], out, [synth["run"]], [synth["s2"]], [synth["s6"]], k=5, n_inject=60, workers=1,
                  seed=0, ref_cap=300, min_ref=50, n_perm=49, top_bouts=4)
    s = meta["behaviour"]["rtest"]
    assert s["flight_state"] == 0  # derived from speed, not assumed to be 1
    for f in ("summary.md", "metadata.json", "qc_inject.csv", "qc_slip_inject.csv", "qc_bird_days.csv",
              "qc_events.csv", "qc_birds.csv", "qc_windows_flagged.parquet", "beh_window_factors_rtest.csv",
              "beh_bouts_rtest.csv", "beh_days_rtest.csv", "beh_inject_rtest.csv", "l2_birds_rtest.csv",
              "nuisance.csv", "birds.csv", "blind_key_rtest.csv", "qc_timelines/B0.png"):
        assert (out / f).exists(), f
    # QC: planted stuck day and slip are found; injected faults are seen by their checks; clean rarely flagged
    W = pd.read_parquet(out / "qc_windows_flagged.parquet")
    stuck = W[W["stuck_axis"]]
    assert set(stuck["bird"]) == {PLANT_STUCK[0]} and (stuck["rel_day"] == PLANT_STUCK[1]).all() and len(stuck) == 24
    ev = pd.read_csv(out / "qc_events.csv")
    rc = ev[ev["kind"] == "roost_change"]
    assert set(rc["bird"]) == {PLANT_SLIP[0]}
    assert ((rc["rel_day_first"] <= PLANT_SLIP[1] + 1) & (rc["rel_day_last"] >= PLANT_SLIP[1] - 5)).any()
    rec = pd.read_csv(out / "qc_inject.csv")
    tgt = rec[rec["target"] & (rec["fault"] != "transplant")].set_index("fault")
    for f in ("stuck_axis", "amplitude_half", "gps_speed_zero", "gps_spike"):
        assert tgt.loc[f, "flag_rate_eligible"] > 0.9, f
    cl = rec[(rec["fault"] == "clean") & (rec["check"] == "any_check")]
    assert cl["flag_rate_eligible"].iat[0] < 0.1  # the clean sample includes the planted stuck day and slip
    # behaviour: planted bout and planted day rank first in their birds; no short window is scored
    B = pd.read_csv(out / "beh_bouts_rtest.csv")
    top = B.loc[B["content_z"].idxmax()]
    assert top["bird"] == PLANT_BOUT[0] and top["rel_day"] == PLANT_BOUT[1]
    D = pd.read_csv(out / "beh_days_rtest.csv")
    top = D.loc[D["z"].idxmax()]
    assert top["bird"] == PLANT_DAY[0] and top["rel_day"] == PLANT_DAY[1]
    top_w = pd.read_csv(out / "beh_top_windows_rtest.csv")
    rows = {b: pd.read_parquet(synth["windows"] / b / "meta.parquet")["acc_samples"].to_numpy() for b in top_w["bird"].unique()}
    assert all(rows[b][r] >= an.FULL_SAMPLES for b, r in zip(top_w["bird"], top_w["row"]))
    inj = pd.read_csv(out / "beh_inject_rtest.csv").set_index(["injection", "score"])
    assert inj.loc[("transplant_diff_mix", "content"), "flag_rate"] > 0.5
    assert inj.loc[("shuffle", "sequence"), "auc_vs_clean_hosts"] > 0.8
    assert inj.loc[("other_bird", "energy distance"), "auc_vs_clean_hosts"] > 0.8
    nu = pd.read_csv(out / "nuisance.csv")
    assert {"partial_rho", "p_partial", "p_holm", "p_partial_holm"} <= set(nu.columns)
    assert not nu["score"].str.contains("l1_").any() and nu["score"].str.startswith("l2_score_").any()
    assert (nu["p_holm"].dropna() >= nu["p_perm_birds"].dropna()).all()
    assert pd.read_csv(out / "l2_birds_rtest.csv")["l2_score"].between(0, 1).all()
    assert "top1pct_content" in B and "top1pct" in D
    cv = s["coverage"]
    assert cv["n_short"] > 0 and 0 < cv["share_windows_no_window_score"] < 1 and cv["n_birds"] == N_BIRDS
    txt = (out / "summary.md").read_text()
    for phrase in ("ranked for inspection", "Not scored", "chosen after seeing", "180 deg", "recall over 20-45 deg",
                   "Holm", "local_solar_hour missing"):
        assert phrase in txt, phrase
    assert meta["local_solar_hour_missing"]["n_windows"] == 0
    # privacy: no position, date, clock or solar column in any table
    for p in list(out.glob("*.csv")) + list(out.glob("*.parquet")):
        cols = pd.read_parquet(p).columns if p.suffix == ".parquet" else pd.read_csv(p, nrows=0).columns
        assert not {c for c in cols if re.search(r"lat|lon|solar|date|^t_start$|time", c.lower())}, p.name
    with pytest.raises(SystemExit):
        an.run(synth["windows"], out, [synth["run"]], [synth["s2"]], [synth["s6"]])


# ---------------------------------------------------------------- round-3 regressions (review 2026-11-25)


def _old_day_pair_ed(QX, QH, Y, YH, rng, m=an.DAY_M):
    """The pre-fix pairing (reference windows drawn WITH replacement), kept here only as the test's control."""
    common = np.intersect1d(np.unique(QH), np.unique(YH))
    use = np.flatnonzero(np.isin(QH, common))
    if len(use) < m:
        return np.nan
    qi = rng.choice(use, m, replace=False)
    yi = np.array([rng.choice(np.flatnonzero(YH == h)) for h in QH[qi]])
    return an.energy_distance(QX[qi], Y[yi])


def _schedule_change_share(seed):
    """iid behaviour; days 0-59 have 12 windows per hour (300 s schedule), days 60-119 4 per hour (900 s). Returns the
    share of the top-10% day scores that fall in the 900 s period (0.5 = no schedule effect)."""
    rng = np.random.default_rng(seed)
    day, hour = [], []
    for d in range(120):
        per = 12 if d < 60 else 4
        day += [d] * (24 * per)
        hour += list(np.repeat(np.arange(24), per))
    day, hour = np.array(day), np.array(hour)
    X = rng.normal(size=(len(day), 10))
    sc = np.array([an.score_day(X, day, hour, np.flatnonzero(day == d), d, rng)[0] for d in range(120)])
    top = sc > np.percentile(sc, 90)
    return float(top[60:].sum() / top.sum())


def test_day_score_not_driven_by_schedule_change(monkeypatch):
    """C4: with reference windows drawn with replacement, days of the sparser schedule fill the top of the ranking;
    drawn without replacement (min(nQ, nY) per common hour) they do not. Measured here over 6 seeds: old 0.76,
    new 0.49. Residual (documented, not fixed): at 1 window per hour the query sample is nearly fixed and sparse
    days still dominate."""
    seeds = (0, 1, 2)
    new = np.mean([_schedule_change_share(s) for s in seeds])
    monkeypatch.setattr(an, "day_pair_ed", _old_day_pair_ed)
    old = np.mean([_schedule_change_share(s) for s in seeds])
    assert 0.3 <= new <= 0.7, new
    assert old - new >= 0.15, (old, new)


def test_day_pair_no_repeats_and_equal_hours(monkeypatch):
    rng = np.random.default_rng(12)
    QH = np.repeat(np.arange(24), 12)
    YH = np.repeat(np.arange(24), 1)  # one window per hour on the other day
    QX, Y = rng.normal(size=(len(QH), 3)), rng.normal(size=(len(YH), 3))
    seen = {}
    monkeypatch.setattr(an, "energy_distance", lambda A, B: seen.update(A=A, B=B) or 0.0)
    an.day_pair_ed(QX, QH, Y, YH, rng)
    A, B = seen["A"], seen["B"]
    assert len(A) == len(B) == an.DAY_M
    assert len(np.unique(B, axis=0)) == len(B) and len(np.unique(A, axis=0)) == len(A)
    hq = np.sort([QH[np.flatnonzero((QX == a).all(1))[0]] for a in A])
    hy = np.sort([YH[np.flatnonzero((Y == b).all(1))[0]] for b in B])
    assert (hq == hy).all()
    assert np.isnan(an.day_pair_ed(QX, QH, Y[:19], YH[:19], rng))  # capacity 19 < DAY_M


def test_solar_hour_gaps_are_filled_from_neighbours():
    """I26: a window with no local_solar_hour keeps its neighbours' offset instead of offset 0 (another day)."""
    t = pd.Series(pd.to_datetime(["2025-05-01 05:00", "2025-05-01 06:00", "2025-05-01 06:30"], utc=True))
    lsh = np.array([22.0, np.nan, 23.5])  # offset -7 h
    off, n_nan = an.solar_offset(t, lsh)
    assert n_nan == 1 and np.allclose(off, -7.0)
    day, _ = an.local_days(t, lsh)
    assert len(set(day)) == 1
    off2, _ = an.solar_offset(t, np.full(3, np.nan))
    assert np.allclose(off2, an.FALLBACK_OFFSET_H)


def test_level2_ranks_min_birds_and_empty(monkeypatch):
    """I22 / M6: cells need >= L2_MIN_BIRDS birds; within-cell ranks; median over cells; no KeyError when empty."""
    rng = np.random.default_rng(13)
    birds = [f"B{i}" for i in range(25)]
    r_big = {b: float(rng.normal()) for b in birds}
    r_big["B0"] = 50.0  # one extreme cell
    jobs = [((0, 0), None), ((1, 0), None), ((2, 1), None), ((3, 2), None), ((4, 3), None)]
    res = [r_big] + [{b: float(rng.normal()) for b in birds} for _ in range(3)] + [{b: 1.0 for b in birds[:10]}]
    Bc, Bs = an.l2_bird_scores(jobs, res)
    assert Bc["n_birds_cell"].min() >= an.L2_MIN_BIRDS and Bc["cell"].nunique() == 4
    assert Bc["rank_in_cell"].between(0, 1).all()
    b0 = Bs.set_index("bird").loc["B0"]
    assert b0["n_cells"] == 4 and b0["l2_score"] < 0.99  # the extreme cell alone does not decide the score
    Bc0, Bs0 = an.l2_bird_scores([((0, 0), None)], [{b: 1.0 for b in birds[:5]}])
    assert len(Bc0) == 0 and list(Bs0.columns) == an.L2_BIRD_COLS
    # level2 end to end with no cell of >= L2_MIN_BIRDS birds, and a month gap
    samples = [{(0, 0): rng.normal(size=(an.L2_N, 32)).astype(np.float16)} for _ in range(6)]
    ms = [{(0, m): (rng.normal(size=(4 * an.L2_MONTH_N, 32)).astype(np.float16), np.arange(4 * an.L2_MONTH_N) % 30)
           for m in ("2025-01", "2025-02", "2025-04")} for _ in range(6)]
    Bc, Bs, M = an.level2(samples, ms, [f"B{i}" for i in range(6)], "t", rng, 1, 0)
    assert len(Bc) == 0 and len(Bs) == 0
    assert M.loc[M["gap_months"] > 1, "z_change"].isna().all() and M.loc[M["gap_months"] == 1, "z_change"].notna().all()


def test_holm_floor_format_and_deploy_year(tmp_path):
    """I27 / M8 / C5."""
    assert np.allclose(an.holm(np.array([0.01, 0.04, np.nan, 0.03])), [0.03, 0.06, np.nan, 0.06], equal_nan=True)
    assert an.fmt_p(1 / 2001, 2000) == "<= 0.0005" and an.fmt_p(0.03, 2000) == "0.03" and an.fmt_p(np.nan, 9) == "n/a"
    # cohort = real deployment year: a 2026 deployment is not merged with 2025
    F = pd.DataFrame({"bird": ["B0"] * 3 + ["B1"] * 3,
                      "t_start": pd.to_datetime(["2025-06-02", "2025-07-01", "2025-08-01",
                                                 "2026-06-20", "2026-07-01", "2026-08-01"], utc=True),
                      "sensitivity_k": 0, "gps_fixes": 0, "acc_samples": 200, "lsh_filled": [False, True, False] * 2})
    ref = tmp_path / "ref.csv"
    pd.DataFrame({"animal-id": ["B0", "B1"], "deploy-on-date": ["2025-06-01", "2026-06-15"], "animal-sex": ["f", "m"],
                  "animal-comments": ["", ""], "deployment-comments": ["", ""]}).to_csv(ref, index=False)
    B = an.bird_table(F, ref)
    assert B.loc["B0", "deploy_year"] == 2025 and B.loc["B1", "deploy_year"] == 2026
    assert B.loc["B1", "deploy_month"] == 6 and "cohort_2025" not in B
    assert B["lsh_missing_share"].round(3).tolist() == [0.333, 0.333]
    # shared rule: a binary nuisance separated by deployment month (AUC >= 0.9) is not testable
    Bd = pd.DataFrame({"deploy_year": [2024.0] * 10, "deploy_month": [10.0] * 5 + [6.0] * 5,
                       "migrant": [1.0] * 5 + [0.0] * 5, "sex_f": [1.0, 0.0] * 5})
    assert an.design_auc(Bd, "migrant") == 1.0 and an.design_auc(Bd, "sex_f") < 0.9
    assert np.isnan(an.design_auc(Bd, "deploy_month"))


def test_yaw_180_is_invisible_and_slip_angles():
    """I24: documents the blind spot (axial statistic, gravity unchanged) and the 20-45 deg recall grid."""
    rng = np.random.default_rng(14)
    a, am, g, gm = _windows(rng, flap=True)
    R = _refs(len(a))
    a2 = an.fault_rotate(a, G_DIR, rng, deg=180.0, yaw=True)
    fl = an.qc_eval(a2, am, g, gm, np.zeros(len(a), int), np.zeros(len(a), bool), np.arange(len(a)), R)
    assert fl["elig_yaw_flap"].mean() > 0.9 and not fl["yaw_flap"].any()
    assert {20.0, 25.0, 30.0, 35.0, 40.0, 45.0} <= set(an.SLIP_DEG)
