import json
import re

import numpy as np
import pandas as pd
import pytest
import torch

from aquilacl import anomaly as an
from aquilacl import dataset
from aquilacl.model import Encoder
from aquilacl.windows import GPS_CHANNELS


# ---------------------------------------------------------------- synthetic data

PLANTED = [1, 2, 3, 4]


def _bird_windows(rng, n_seg=60, start="2025-03-01"):
    """Scheduled single bursts (short or full, no GPS) and continuous 6-window segments with GPS."""
    rows, acc, am, gps, gm = [], [], [], [], []
    t0 = pd.Timestamp(start, tz="UTC")
    for s in range(n_seg):
        t = t0 + pd.Timedelta(hours=float(s * 23.3 % 1400))
        cont = s % 3 == 0
        nwin = 6 if cont else 1
        for k in range(nwin):
            n = 200 if cont or s % 2 else 40
            a = np.zeros((200, 3), np.float32)
            a[:n] = rng.normal(0, 0.3 if cont else 0.05, (n, 3)) + np.array([0.1, 0.0, 0.98])
            m = np.zeros(200, bool)
            m[:n] = True
            g = np.zeros((10, len(GPS_CHANNELS)), np.float32)
            gmask = np.zeros_like(g, bool)
            if cont:
                g[:, 0] = g[:, 1] = rng.normal(12, 2, 10)
                g[:, 2] = rng.normal(0, 1, 10)
                g[:, 4] = g[:, 0]
                g[:, 5] = 5
                gmask[:] = True
            rows.append({"t_start": t + pd.Timedelta(seconds=10 * k), "segment": s, "k_in_segment": k,
                         "segment_windows": nwin, "sensitivity_k": int(cont), "acc_samples": n,
                         "gps_fixes": 10 if cont else 0, "sun_elev_deg": -10.0 if s % 4 == 1 else 20.0,
                         "local_solar_hour": float((t.hour - 7) % 24), "speed_mean": float(g[:, 0].mean()) if cont else np.nan,
                         "label_flapping": 1.0 if cont and k % 2 else 0.0, "vedba": 0.3 if cont else 0.01})
            acc.append(a)
            am.append(m)
            gps.append(g)
            gm.append(gmask)
    return pd.DataFrame(rows), np.stack(acc), np.stack(am), np.stack(gps), np.stack(gm)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """A windows dir (6 birds), a run with a random encoder whose stored h is made by the documented pipeline, and
    K=2 states (state 0 = flight, to check that the flight state is derived, not assumed)."""
    root = tmp_path_factory.mktemp("synth")
    rng = np.random.default_rng(0)
    windows, run, st_dir = root / "windows", root / "runs" / "rtest", root / "states2"
    for p in (windows, run / "embeddings", st_dir):
        p.mkdir(parents=True)
    torch.manual_seed(0)
    model = Encoder(len(GPS_CHANNELS))
    torch.save({"model": model.state_dict()}, run / "ckpt_000001.pt")
    (run / "config.json").write_text(json.dumps({"step_g": 0.002, "clip_g": 1.8, "mask_gps_channels":
                                                 ["eobs:horizontal-accuracy-estimate"]}))
    orient, st = [], []
    enc_rows = {}
    for i in range(6):
        b = f"B{i}"
        m, a, am, g, gm = _bird_windows(rng)
        d = windows / b
        d.mkdir()
        m.drop(columns=["speed_mean", "label_flapping", "vedba"]).assign(bird=b).to_parquet(d / "meta.parquet")
        m[["speed_mean", "label_flapping", "vedba"]].to_parquet(d / "features.parquet")
        np.save(d / "acc.npy", a.astype(np.float16))
        np.save(d / "acc_mask.npy", am)
        np.save(d / "gps.npy", g)
        np.save(d / "gps_mask.npy", gm)
        R = an.rot_axis([1, 0.5, 0], 5 + i).astype(np.float32)
        orient.append({"bird": b, "R": R.ravel().tolist()})
        enc_rows[b] = (a.astype(np.float16).astype(np.float32), am, g, gm, R)
        st.append(pd.DataFrame({"bird": b, "row": np.arange(len(m)), "state_hmm": 1 - m["sensitivity_k"].to_numpy()}))
    (run / "orientation.json").write_text(json.dumps(orient))
    pd.concat(st).to_parquet(st_dir / "states.parquet")
    enc = an.Encoder(run, "cpu")
    far = an.norm_rows(np.random.default_rng(9).normal(size=(1, 128)))
    for b, (a, am, g, gm, R) in enc_rows.items():
        h = enc.h(a, am, np.repeat(R[None], len(a), 0), g, gm)
        if b == "B0":
            h[PLANTED] = far  # positive control: 4 consecutive windows of continuous segment 0, far from everything
        np.savez(run / "embeddings" / f"{b}.npz", h=h.astype(np.float16))
    return {"windows": windows, "run": run, "states": st_dir, "root": root}


# ---------------------------------------------------------------- level 1 kernels


def test_robust_z_makes_strata_comparable_and_pooled_flags_both():
    rng = np.random.default_rng(1)
    g = np.r_[np.zeros(2000), np.ones(2000)].astype(int)
    x = np.r_[rng.normal(1, 0.1, 2000), rng.normal(5, 2.0, 2000)]
    x[[5, 2005]] = [3.0, 45.0]  # one planted outlier per stratum (20 robust SD)
    z, par = an.robust_z(x, g)
    thr = np.percentile(z, 99)
    assert z[5] > thr and z[2005] > thr
    flagged = z > thr
    assert 0.6 < flagged[g == 0].sum() / flagged[g == 1].sum() < 1.6  # raw x would flag only stratum 1
    assert np.allclose(an.apply_z(x[[5, 2005]], g[[5, 2005]], par), z[[5, 2005]])
    assert an.group_quantile_flag(z, g).sum() == pytest.approx(40, abs=4)


@pytest.mark.parametrize("group, k, flag, expected", [
    ([0] * 6, [0, 1, 2, 3, 4, 5], [1, 1, 1, 0, 1, 1], [(0, 2)]),
    ([0] * 6, [0, 1, 2, 4, 5, 6], [1, 1, 1, 1, 1, 1], [(0, 2), (3, 5)]),  # gap in k_in_segment breaks a run
    ([0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 2, 3], [1, 1, 1, 1, 1, 1], [(2, 5)]),  # segment change breaks a run
    ([0] * 3, [0, 1, 2], [1, 0, 1], []),
])
def test_flag_runs(group, k, flag, expected):
    assert an.flag_runs(np.array(group), np.array(k), np.array(flag, bool)) == expected


def test_knn_excludes_same_segment_and_day_and_extra_queries_match_real_scores():
    rng = np.random.default_rng(2)
    H = an.norm_rows(rng.normal(size=(600, 16)))
    day = np.repeat(np.arange(60), 10)
    seg = np.repeat(np.arange(120), 5)
    Q = H[:3]
    d_ex = an.kth_dist(Q, H, day[:3], day, seg[:3], seg, k=1)
    assert (d_ex > 1e-3).all()  # own row (same segment and day) never matches
    assert np.allclose(an.kth_dist(Q, H, day[:3] + 1000, day, seg[:3] + 1000, seg, k=1), 0, atol=1e-3)
    key = np.repeat([0, 1], 300)
    idx = np.array([3, 350, 599])
    sc, lev, ex = an.score_groups(H, day, seg, key, k=5, cap=200, min_ref=50, seed=0, extra=(H[idx], idx))
    assert np.isfinite(sc).all() and set(np.unique(lev)) <= {0, 1, 2}
    assert np.allclose(ex, sc[idx], rtol=1e-5)
    sc2, lev2, _ = an.score_groups(H, day, seg, key, k=5, cap=200, min_ref=10_000, seed=0)
    assert np.isnan(sc2).all() and (lev2 == -1).all()


def test_factor_table_ratios():
    d = pd.DataFrame({"bird": np.repeat(["a", "b"], 50), "acc_samples": np.r_[np.full(80, 200), np.full(20, 40)],
                      "sensitivity_k": 0, "gps_fixes": 0, "segment_windows": 1, "k_in_segment": 0,
                      "flight": False, "sun_elev_deg": 10.0, "month": 3})
    flag = np.zeros(100, bool)
    flag[80:90] = True
    flag[:10] = True
    ft = an.factor_table(d, flag).set_index("factor")
    assert ft.loc["short window (acc_samples < 190)", "ratio"] == pytest.approx(0.5 / 0.2)
    assert an.worst_format_ratio(ft.reset_index()) == pytest.approx(2.5)


# ---------------------------------------------------------------- level 2 kernels


def test_mmd_and_permutation_spearman_null_and_planted():
    rng = np.random.default_rng(3)
    X, Y = rng.normal(size=(200, 5)), rng.normal(size=(300, 5))
    gam = an.median_gamma(np.r_[X, Y], rng)
    assert abs(an.mmd2(X, Y, gam)) < 0.01
    assert an.mmd2(X + 1.0, Y, gam) > 0.05
    birds = np.repeat(np.arange(10), 60)
    P = rng.normal(size=(600, 5))
    P[birds == 3] += 1.5
    r3 = an.bird_vs_pool(P, birds, 3, gam, 20, rng, n=60)
    r5 = an.bird_vs_pool(P, birds, 5, gam, 20, rng, n=60)
    assert r3["mmd2"] > r5["mmd2"] and r3["p_null"] < 0.1
    x = rng.normal(size=40)
    rho, p, n = an.perm_spearman(x, x + rng.normal(0, 0.3, 40), 199, rng)
    assert rho > 0.8 and p <= 0.01 and n == 40
    ps = [an.perm_spearman(rng.normal(size=30), rng.normal(size=30), 99, rng)[1] for _ in range(60)]
    assert 0.3 < np.mean(ps) < 0.7
    ym = np.repeat(["2025-01", "2025-02", "2025-03"], 100)
    day = np.tile(np.arange(100), 3)
    Z = rng.normal(size=(300, 5))
    Z[200:] += 2
    ch = pd.DataFrame(an.month_change(Z, ym, day, gam))
    assert ch["excess"].iat[1] > 10 * abs(ch["excess"].iat[0])


# ---------------------------------------------------------------- faults and model input


def test_faults_and_input_pipeline_match_dataset_load(synth):
    rng = np.random.default_rng(4)
    R = np.stack([an.rot_axis([1, 0, 0], 20)] * 5).astype(np.float32)
    a = (rng.normal(0, 0.2, (5, 200, 3)) + R[0].T @ [0, 0, 1]).astype(np.float32)  # gravity on +z after R
    am = np.ones((5, 200), bool)
    am[0, 50:] = False
    st = an.fault_stuck(a, am, rng)
    assert all((np.ptp(st[i][am[i]], axis=0) == 0).sum() >= 1 for i in range(5))
    for yaw in (False, True):
        r = an.fault_rotate(a, R, rng, yaw=yaw)
        assert np.allclose(np.linalg.norm(r, axis=2), np.linalg.norm(a, axis=2), atol=1e-4)
        gz_before = (a.mean(1) @ R[0].T)[:, 2]
        gz_after = (r.mean(1) @ R[0].T)[:, 2]
        assert np.allclose(gz_after, gz_before, atol=1e-3) if yaw else (np.abs(gz_after) < 0.3).all()
    g = np.ones((2, 10, len(GPS_CHANNELS)), np.float32)
    gm = np.ones_like(g, bool)
    gm[0, :5] = False
    gz = an.fault_gps_zero(g, gm)
    sp = GPS_CHANNELS.index("ground-speed")
    assert (gz[1, :, sp] == 0).all() and (gz[0, :5, sp] == 1).all() and (gz[:, :, 2] == 1).all()
    # the rebuilt input equals dataset.load for the same rotation
    data = dataset.load(synth["windows"], birds=["B0"], step_g=0.002, clip_g=1.8,
                        mask_gps_channels=("eobs:horizontal-accuracy-estimate",))
    Rb = np.asarray(data.orientation["R"].iat[0], np.float32).reshape(3, 3)
    a0, am0, g0, gm0 = an.read_rows(synth["windows"], np.array(["B0"] * 20), np.arange(20))
    x = an.acc_input(a0, am0, np.repeat(Rb[None], 20, 0), 0.002, 1.8)
    gi, gmi = an.gps_input(g0, gm0, ("eobs:horizontal-accuracy-estimate",))
    assert np.array_equal(x, data.acc[:20]) and np.array_equal(gi, data.gps[:20]) and np.array_equal(gmi, data.gps_mask[:20])


def test_blind_page_has_no_date_or_bird(tmp_path):
    rng = np.random.default_rng(5)
    e = {"rank": 1, "month": "Mar", "state": "flight", "n_flagged": 3, "t_rel": [0.0, 10.0, 20.0],
         "acc": [rng.normal(size=(200, 3)) for _ in range(3)], "gps_speed": [rng.normal(size=10)] * 3,
         "gps_climb": [rng.normal(size=10)] * 3, "flagged": [False, True, True]}
    fig = an.blind_page([e], tmp_path / "p.png")
    texts = " ".join(t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text")))
    assert (tmp_path / "p.png").exists() and "Episode 1" in texts
    assert not re.search(r"\d{4}-\d{2}|\d{1,2}:\d{2}|B\d", texts)


# ---------------------------------------------------------------- end to end


def test_end_to_end_synthetic(synth):
    out = synth["root"] / "out"
    meta = an.run([synth["run"]], [synth["states"]], synth["windows"], out, k=5, n_inject=40, workers=1, seed=0,
                  ref_cap=300, min_ref=20, n_null=5, n_perm=49, top_episodes=4)
    assert meta["flight_state"]["rtest"] == 0  # derived from speed, not assumed to be 1
    for f in ("summary.md", "metadata.json", "l1_variants_rtest.csv", "l1_factors_rtest.png", "inject_rtest.csv",
              "inject_recall.png", "l2_birds_rtest.csv", "nuisance.csv", "birds.csv", "l1_bird_days_rtest.csv"):
        assert (out / f).exists(), f
    rec = pd.read_csv(out / "inject_rtest.csv").set_index("fault")
    assert {"clean", "stuck_axis", "tilt90", "amplitude_half", "gps_speed_zero", "transplant_own_R",
            "transplant_host_R"} <= set(rec.index)
    rp = json.loads((out / "inject_reproduction.json").read_text())["rtest"]
    assert rp["median_cosine_stored_vs_reencoded"] > 0.999
    var = pd.read_csv(out / "l1_variants_rtest.csv")
    assert var["chosen"].sum() == 1 and set(var["variant"]) == set(an.VARIANTS)
    assert (out / "l1_coverage_rtest.csv").exists() and var.loc[var["chosen"], "qualified"].all()
    top = pd.read_csv(out / "l1_top_windows_rtest.csv")
    assert set(PLANTED) <= set(top.loc[top["bird"] == "B0"].head(len(PLANTED) + 4)["row"])
    ep = pd.read_csv(out / "l1_episodes_rtest.csv")
    assert ((ep["bird"] == "B0") & (ep["segment"] == 0) & (ep["n_windows"] >= 3)).any()
    key = pd.read_csv(out / "blind_key_rtest.csv")
    assert len(key) >= 1 and (out / "blind_rtest_p1.png").exists()
    for p in out.glob("*.csv"):  # never a position column
        assert not {c for c in pd.read_csv(p, nrows=0).columns if re.search(r"lat|lon|solar", c.lower())}, p.name
    with pytest.raises(SystemExit):
        an.run([synth["run"]], [synth["states"]], synth["windows"], out)
