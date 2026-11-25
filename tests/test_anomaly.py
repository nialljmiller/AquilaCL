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
    YH = np.r_[np.full(10, 3), np.full(90, 7), np.full(20, 12)]
    Y = np.r_[np.full((10, 2), 2.0), np.zeros((90, 2)), np.full((20, 2), 9.0)] + rng.normal(0, 0.1, (120, 2))
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


def test_end_to_end_synthetic(synth):
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
    assert {"partial_rho", "p_partial"} <= set(nu.columns) and not nu["score"].str.contains("l1_").any()
    # privacy: no position, date, clock or solar column in any table
    for p in list(out.glob("*.csv")) + list(out.glob("*.parquet")):
        cols = pd.read_parquet(p).columns if p.suffix == ".parquet" else pd.read_csv(p, nrows=0).columns
        assert not {c for c in cols if re.search(r"lat|lon|solar|date|^t_start$|time", c.lower())}, p.name
    with pytest.raises(SystemExit):
        an.run(synth["windows"], out, [synth["run"]], [synth["s2"]], [synth["s6"]])
