import json

import numpy as np
import pandas as pd
import pytest

from aquilacl import flight

N_BIRDS = 10
N_BOUTS = 44  # per bird, half of each planted type: >= 20 per type for presence
K6 = 4  # synthetic K=6 stand-in: 0 ground, 1 climbing circling, 2 descending glide, 3 flapping


@pytest.mark.parametrize("t, cont, want", [
    ([0, 10, 35, 61, 71], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1]),   # a 25 s step joins, 26 s splits
    ([0, 10, 20, 30], [1, 1, 0, 1], [0, 0, -1, 1]),            # a window of another mode splits
    ([0, 10], [0, 0], [-1, -1]),
])
def test_bout_ids(t, cont, want):
    assert flight.bout_ids(np.array(t, float), np.array(cont, bool)).tolist() == want


def test_small_statistics():
    z = np.r_[np.zeros(5), np.ones(5)]
    assert flight.eta2(z, ["a"] * 5 + ["b"] * 5) == pytest.approx(1.0)
    hour = np.linspace(0, 24, 49)[:-1]
    assert flight.circ_r2(np.cos(2 * np.pi * hour / 24), hour, 24.0) == pytest.approx(1.0)
    assert flight.holm(np.array([0.01, 0.04, np.nan, 0.03])).round(3).tolist()[:2] == [0.03, 0.06]


def test_bout_descriptors_by_hand():
    n = 8  # 80 s: climb +2 m/s for 4 windows, then -1 m/s at 15 m/s for 4 windows
    B = pd.DataFrame({"bout": 0, "bird": "B0", "cohort": "2024", "t_start": pd.Timestamp("2024-06-01", tz="UTC"),
                      "t_s": 10.0 * np.arange(n), "n_speed": 10, "flight": True,
                      "speed_mean": [8.0] * 4 + [15.0] * 4, "climb_mean": [2.0] * 4 + [-1.0] * 4,
                      "turn_abs_mean": [36.0] * 4 + [0.0] * 4, "n_turn": 10, "turn_net_deg": [360.0] * 4 + [0.0] * 4,
                      "label_flapping": [0.0] * 7 + [np.nan], "flap_band_frac": 0.1, "local_solar_hour": 11.0,
                      "month": 6, "state6": [1] * 4 + [2] * 4})
    Z = np.zeros((n, 2))
    H = np.tile(np.eye(1, 4), (n, 1))
    T = flight.bout_table(B, Z, H, K6).iloc[0]
    assert T["duration_s"] == 80 and T["net_climb_m"] == pytest.approx(40.0)
    assert T["climb_first_half_m"] == pytest.approx(80.0) and T["climb_second_half_m"] == pytest.approx(-40.0)
    assert T["circles_gross"] == pytest.approx(4.0) and T["circles_net"] == pytest.approx(4.0)
    assert T["glide_ratio"] == pytest.approx(15.0)  # 600 m forward over 40 m lost
    assert T["share_state_1"] == 0.5 and T["state_changes"] == 1 and T["flap_share"] == 0.0
    assert T["h_spread"] == pytest.approx(0.0)


def test_fit_axes_recovers_planted_axis_with_positive_sign():
    rng = np.random.default_rng(1)
    n = 3000
    F = pd.DataFrame({"bird": np.repeat([f"B{i}" for i in range(6)], n // 6),
                      "t_start": pd.Timestamp("2024-05-01", tz="UTC") + pd.to_timedelta(np.arange(n) * 600, unit="s"),
                      "speed_mean": rng.uniform(5, 20, n)})
    for v in flight.CORR_VARS:
        if v not in F:
            F[v] = rng.normal(size=n)
    u, base = rng.normal(0, 1, 32), rng.normal(0, 3, 32)
    H = base - 0.2 * np.outer(F["speed_mean"], u) + rng.normal(0, 0.3, (n, 32))  # planted with a negative sign
    H /= np.linalg.norm(H, axis=1, keepdims=True)
    pca, Z, fit = flight.fit_axes(F, H.astype(np.float32), 400, rng, 0)
    rho = flight.spearman(F["speed_mean"].to_numpy(), Z[:, 0])[0]
    assert rho > 0.9 and np.allclose(pca.transform(H)[:, :5], Z, atol=1e-4)


def _bird(win, runs, b, rng, s2, s6):
    year = 2024 if b < N_BIRDS // 2 else 2025
    t0 = pd.Timestamp(f"{year}-03-01", tz="UTC")
    rows = []
    t = 0.0
    for i in range(60):  # scheduled bursts, 300 s apart; a quarter in flight (no GPS)
        fl = rng.random() < 0.25
        rows.append(dict(t=t, k=0, beh=3 if fl else 0, speed=np.nan, climb=np.nan, turn=np.nan))
        t += 300
    for j in range(N_BOUTS):  # continuous bouts of two planted types
        typ = j % 2
        n = int(rng.integers(8, 20))
        for w in range(n):
            if typ == 0:
                climbing = w < n // 2
                rows.append(dict(t=t, k=1, beh=1 if climbing else 2, speed=10.0 if climbing else 18.0,
                                 climb=2.0 if climbing else -2.0, turn=30.0 if climbing else 2.0))
            else:
                rows.append(dict(t=t, k=1, beh=3, speed=12.0, climb=0.0, turn=5.0))
            t += 10
        t += 600
    m = pd.DataFrame(rows)
    n = len(m)
    sp = m["speed"].to_numpy() + rng.normal(0, 0.5, n)
    meta = pd.DataFrame(dict(bird=f"B{b}", tag=f"T{b}", t_start=t0 + pd.to_timedelta(m["t"], unit="s"),
                             segment=np.arange(n), k_in_segment=0, segment_windows=1, sensitivity_k=m["k"],
                             decoding="x", sun_elev_deg=20.0, acc_samples=np.where(m["k"] == 1, 200, 24),
                             gps_fixes=np.where(m["k"] == 1, 10, 0), local_solar_hour=rng.uniform(6, 18, n)))
    feats = pd.DataFrame(dict(vedba=np.where(m["beh"] == 3, 0.3, 0.07) + rng.normal(0, 0.01, n),
                              flap_band_frac=np.where(m["beh"] == 3, 0.8, 0.1), dom_freq_hz=3.0,
                              speed_mean=sp, climb_mean=m["climb"] + rng.normal(0, 0.2, n),
                              turn_abs_mean=m["turn"], turn_net_deg=m["turn"] * 10, path_m=sp * 10,
                              n_speed=np.where(m["k"] == 1, 10, 0), n_turn=np.where(m["k"] == 1, 10, 0),
                              label_flapping=np.where(m["k"] == 1, (m["beh"] == 3).astype(float), np.nan)))
    d = win / f"B{b}"
    d.mkdir(parents=True)
    meta.to_parquet(d / "meta.parquet")
    feats.to_parquet(d / "features.parquet")
    latent_speed = np.where(np.isfinite(m["speed"]), m["speed"], 12.0)
    for r, run in enumerate(runs):
        c = np.random.default_rng(r).normal(0, 1, (K6, 64))
        u = np.random.default_rng(10 + r).normal(0, 1, 64)
        h = c[m["beh"]] + 0.3 * np.outer(latent_speed - 12, u) + rng.normal(0, 0.3, (n, 64))
        np.savez(run / "embeddings" / f"B{b}.npz", h=h.astype(np.float16))
    fl = (m["beh"] > 0).astype(int).to_numpy()
    for s in s2:
        s.append(pd.DataFrame(dict(bird=f"B{b}", row=np.arange(n), state=fl, state_hmm=fl)))
    for s in s6:
        s.append(pd.DataFrame(dict(bird=f"B{b}", row=np.arange(n), state=m["beh"], state_hmm=m["beh"])))


def _synthetic(tmp_path):
    rng = np.random.default_rng(0)
    win = tmp_path / "win"
    runs = [tmp_path / "rA", tmp_path / "rB"]
    for r in runs:
        (r / "embeddings").mkdir(parents=True)
    s2rows, s6rows = [[], []], [[], []]
    for b in range(N_BIRDS):
        _bird(win, runs, b, rng, s2rows, s6rows)
    s2dirs, s6dirs = [tmp_path / "k2A", tmp_path / "k2B"], [tmp_path / "k6A", tmp_path / "k6B"]
    for i in range(2):
        s2dirs[i].mkdir()
        s6dirs[i].mkdir()
        pd.concat(s2rows[i]).to_parquet(s2dirs[i] / "states.parquet")
        pd.concat(s6rows[i]).to_parquet(s6dirs[i] / "states.parquet")
        pd.DataFrame({"state": [0, 1], "speed_mean_median": [0.1, 12.0]}).set_index("state").to_csv(
            s2dirs[i] / "state_profiles.csv")
        pd.DataFrame({"state": range(K6), "vedba_median": [0.01, 0.07, 0.07, 0.3],
                      "speed_mean_median": [0.0, 10.0, 18.0, 12.0], "climb_mean_median": [0.0, 2.0, -2.0, 0.0],
                      "turn_abs_mean_median": [0.0, 30.0, 2.0, 5.0]}).set_index("state").to_csv(
            s6dirs[i] / "state_profiles.csv")
    rows = [[str(i), f"B{b}; {100 + i}", "2024-06-05 00:00:00.000", "", "", "f" if b % 2 else "m", "",
             "Tagged as a nestling" if b in (1, 2) else "", "M1", "fw1"] for i, b in enumerate(range(N_BIRDS))]
    ref = tmp_path / "reference data (1).csv"
    pd.DataFrame(rows, columns=["tag-id", "animal-id", "deploy-on-date", "deploy-off-date", "animal-comments",
                                "animal-sex", "animal-life-stage", "deployment-comments", "tag-model", "tag-firmware"]
                 ).to_csv(ref, index=False)
    return runs, s2dirs, s6dirs, win, ref


def _args(runs, s2, s6, win, ref, out):
    a = ["--windows", str(win), "--out", str(out), "--reference", str(ref), "--n-boot", "4", "--n-perm", "99",
         "--per-bird", "200", "--bout-max-k", "4"]
    for r, x, y in zip(runs, s2, s6):
        a += ["--run", str(r), "--states2", str(x), "--states6", str(y)]
    return a


def test_end_to_end_cli(tmp_path):
    runs, s2, s6, win, ref = _synthetic(tmp_path)
    out = tmp_path / "out"
    flight.main(_args(runs, s2, s6, win, ref, out))
    for f in ["summary.md", "metadata.json", "cross_run_pc_spearman.csv"]:
        assert (out / f).exists()
    for r in ("rA", "rB"):
        o = out / r
        for f in ["axis_summary.csv", "axis_correlations.csv", "sex_tests_axes.csv", "bird_pc_means.csv",
                  "flightmap_pc_all.png", "flightmap_pc_k1.png", "density_sex.png", "density_month_k1.png",
                  "flightmap_speed_climb.png", "bouts.csv", "bout_k_table.csv", "bout_types.csv",
                  "bout_type_groups.csv", "bout_type_transitions.csv", "bout_types.png", "flight.json"]:
            assert (o / f).exists(), f
        # planted kinematic axes: speed loads on PC1 or PC2; the strongest correlate of each is positive (sign rule)
        c = pd.read_csv(o / "axis_correlations.csv")
        top = c[(c["subset"] == "all") & (c["pc"] <= 2)].pivot(index="variable", columns="pc", values="spearman")
        assert top.loc["speed_mean"].abs().max() > 0.2
        assert all(top[p].loc[top[p].abs().idxmax()] > 0 for p in (1, 2))
        assert top.loc["sensitivity_k"].abs().max() < 0.2
        t = pd.read_csv(o / "sex_tests_axes.csv")
        k1 = t[t["subset"] == "k1"]  # k0: < MIN_BIRD_WINDOWS flight windows per bird, so no test (NaN)
        assert k1["p_perm"].between(0, 1).all() and (k1["n_f"] == 5).all() and t.loc[t["subset"] == "k0", "p_perm"].isna().all()
        bouts = pd.read_csv(o / "bouts.csv")
        assert len(bouts) == N_BIRDS * N_BOUTS and (bouts["duration_s"] >= 60).all()
        k = pd.read_csv(o / "bout_k_table.csv")
        assert k["K"].min() == 1 and k["jaccard_min"].between(0, 1).all()
        info = json.loads((o / "flight.json").read_text())["bouts"]
        assert info["chosen_k"] >= 1
        # no positions and no exact timestamps in any table
        for f in o.glob("*.csv"):
            cols = {x.lower() for x in pd.read_csv(f, nrows=0).columns}
            assert not cols & {"lat", "lon", "location-lat", "location-long", "t_start", "timestamp"}, f
    s = (out / "summary.md").read_text()
    assert "Run rA" in s and "TESTED" in s


def test_planted_bout_types_and_words(tmp_path):
    runs, s2, s6, win, ref = _synthetic(tmp_path)
    out = tmp_path / "out"
    flight.main(_args(runs[:1], s2[:1], s6[:1], win, ref, out))
    types = pd.read_csv(out / "rA" / "bout_types.csv")
    info = json.loads((out / "rA" / "flight.json").read_text())["bouts"]
    assert info["chosen_k"] == 2  # two planted bout types, both stable and present in both cohorts
    w = types.sort_values("speed_mean")["words"].tolist()  # flapping bouts (12 m/s) are slower than climb-glide bouts
    assert "flapping-rich" in w[0] and "climb then descent" in w[1] and "circling" not in w[0]


def test_refusals(tmp_path):
    runs, s2, s6, win, ref = _synthetic(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        flight.main(_args(runs, s2, s6, win, ref, out))
    with pytest.raises(SystemExit, match="same number"):
        flight.main(_args(runs, s2, s6, win, ref, tmp_path / "o2")[:-2])
    pd.DataFrame({"state": [0, 1], "speed_mean_median": [12.0, 0.1]}).set_index("state").to_csv(
        s2[0] / "state_profiles.csv")
    with pytest.raises(SystemExit, match="faster"):
        flight.check_flight_state(s2[0])
