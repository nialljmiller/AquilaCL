import json

import numpy as np
import pandas as pd
import pytest

from aquilacl import demography as dm
from aquilacl import residency as rs

LAT0, LON0 = 44.4371, -108.2913  # planted, distinctive; must never appear in any output
DEG_KM = rs.KM_PER_DEG


def _path(kind: str, days: int, rng, start="2024-10-15", offset_north=0.0):
    """Daily (lat, lon) offsets in km north/east for a planted strategy, plus dates."""
    t = np.arange(days, dtype=float)
    north = np.zeros(days)
    east = np.zeros(days)
    dates = pd.date_range(start, periods=days, freq="D", tz="UTC")
    if kind == "migrant":  # north 1500 km from ~1 Apr to ~1 Oct, every year
        doy = dates.dayofyear.to_numpy()
        north = 1500 * (rs._sig(doy, 95, 3) - rs._sig(doy, 275, 3))
    elif kind == "disperser":  # moves 500 km east around day 150 and stays
        east = 500 * rs._sig(t, 150, 4)
    elif kind == "nomad":  # 400 km excursion east for 90 days, then back
        east = 400 * (rs._sig(t, 120, 4) - rs._sig(t, 210, 4))
    elif kind == "late_leaver":  # leaves 400 km east 100 days before the track ends
        east = 400 * rs._sig(t, days - 100, 4)
    elif kind == "passage_migrant":  # deployed on passage; winters 350 km east of it, summers 1500 km north
        doy = dates.dayofyear.to_numpy()
        east = 350 * rs._sig(t, 40, 3)
        north = 1500 * (rs._sig(doy, 95, 3) - rs._sig(doy, 275, 3)) * rs._sig(t, 40, 3)
    north = north + rng.normal(0, 5, days)
    east = east + rng.normal(0, 5, days)
    return dates, north, east


def _track_from(dates, north, east):
    lat = LAT0 + north / DEG_KM
    lon = LON0 + east / (DEG_KM * np.cos(np.radians(LAT0)))
    daily = pd.DataFrame({"date": dates, "lat": lat, "lon": lon, "n": 4})
    return rs.track(daily, (LAT0, LON0))


def test_haversine_and_daily_positions():
    assert rs.hav_km(LAT0, LON0, LAT0 + 1, LON0) == pytest.approx(DEG_KM, rel=1e-6)
    ts = pd.to_datetime(["2024-10-14 12:00", "2024-10-15 01:00", "2024-10-15 02:00", "2024-10-15 03:00",
                         "2024-10-16 05:00", "2024-10-16 06:00"])
    g = pd.DataFrame({"timestamp": ts, "location-long": [LON0, LON0, LON0 + 0.2, 0.0, LON0, np.nan],
                      "location-lat": [LAT0, LAT0, LAT0 + 0.2, 0.0, LAT0 + 0.1, LAT0]})
    d = rs.daily_positions(rs.clean_fixes(g, pd.Timestamp("2024-10-15", tz="UTC")))
    assert len(d) == 2 and d["n"].tolist() == [2, 1]  # pre-deploy, (0, 0) and NaN fixes dropped
    assert d.lat.iat[0] == pytest.approx(LAT0 + 0.1)


@pytest.mark.parametrize("model, p", [("resident", [1e4]), ("nomad", [800.0]),
                                      ("disperser", [500.0 ** 2, 150, 4]),
                                      ("migrant", [1500.0 ** 2, 160, 3, 180, 3])])
def test_nsd_fit_recovers_model(model, p):
    rng = np.random.default_rng(0)
    t = np.arange(400, dtype=float)
    y = np.maximum(rs.curve(model, np.array(p), t) + rng.normal(0, 2e3, len(t)), 0)
    fits = rs.fit_models(t, y)
    best = min(rs.MODELS, key=lambda m: fits[m]["aic"])
    assert best == model
    if model == "migrant":
        q = fits["migrant"]["p"]
        assert abs(q[1] - 160) < 5 and abs(q[1] + q[3] - 340) < 5


@pytest.mark.parametrize("kind, days, young, rule, nsd", [
    ("resident", 500, False, "resident_full", "resident"),
    ("migrant", 760, False, "migrant", "migrant"),  # two years: NSD fit on the first cycle only
    ("disperser", 600, False, "disperser", "disperser"),
    ("nomad", 500, False, "nomad", None),  # Bunnefeld has no 'return from a short excursion' class: not checked
    ("resident", 90, False, "insufficient", "insufficient"),
    ("resident", 130, False, "resident_partial", "resident"),  # Oct-Feb only: no spring coverage needed any more
    ("late_leaver", 500, False, "away_at_end", None),
    ("disperser", 600, True, "juvenile_dispersal", "disperser"),
    ("nomad", 500, True, "juvenile_dispersal", None),
    ("resident", 400, True, "resident_full", "resident"),
    # regression: the old rule needed the return to reach the deployment centroid and called this a disperser
    ("passage_migrant", 760, False, "migrant", None),
])
def test_classify_planted_tracks(kind, days, young, rule, nsd):
    dates, n, e = _path(kind, days, np.random.default_rng(1))
    tr = _track_from(dates, n, e)
    c = rs.classify_bird(tr, 120, young)
    assert c["rule_class"] == rule
    if nsd:
        assert c["nsd_class"] == nsd
    if kind == "migrant":
        assert c["n_migrations"] == 2
        for y in (1, 2):  # rule: first day beyond / back within 200 km (planted transitions near doy 95 and 275)
            assert 85 <= c[f"dep_doy_rule_y{y}"] <= 100 and 268 <= c[f"ret_doy_rule_y{y}"] <= 285
        assert abs(c["dep_doy_nsd"] - 95) <= 5 and abs(c["ret_doy_nsd"] - 275) <= 5
        assert c["max_north_deploy_km"] > 1400


def test_far_without_return_is_migrant_departed():
    dates, n, e = _path("migrant", 250, np.random.default_rng(2))  # ends in June, north
    c = rs.classify_bird(_track_from(dates, n, e), 120)
    assert c["rule_class"] == "migrant_departed" and c["nsd_class"] == "undetermined"
    assert rs.coarse(c["rule_class"]) == "migrant" and rs.coarse("away_at_end") == "other"


def test_window_days_match_daily_positions():
    """Guard (no key bug was found in the first run): raw GPS timestamps are tz-naive datetime64[ms] UTC and
    window t_start is datetime64[ns]; both map to the same UTC day, and the +-2-day fill is the nearest position."""
    gps_ts = pd.Series(pd.to_datetime(["2024-12-01 23:50", "2024-12-02 00:10", "2024-12-08 12:00"])).astype(
        "datetime64[ms]")
    g = pd.DataFrame({"timestamp": gps_ts, "location-lat": [LAT0, LAT0 + 5, LAT0], "location-long": LON0})
    tr = rs.track(rs.daily_positions(rs.clean_fixes(g)), (LAT0, LON0))
    assert tr.dc_km.round().tolist() == [0, round(5 * DEG_KM), 0]
    days = rs.area_days(tr.epoch_day.to_numpy(), tr.dc_km.to_numpy())
    e0 = int(tr.epoch_day.iat[0])
    # offsets from Dec 1: Dec 1 (0) in; Dec 2 (1) out, its own position; Dec 3-4 nearest Dec 2 (out); Dec 5 (4) is
    # 3 days from both positions, not filled; Dec 6-7 (5, 6) nearest Dec 8 (in); Dec 8 (7) in; Dec 9-10 and
    # Nov 29-30 filled
    assert sorted(days - e0) == [-2, -1, 0, 5, 6, 7, 8, 9]
    t_start = pd.Series(pd.to_datetime(["2024-12-01 00:30", "2024-12-02 00:30", "2024-12-06 23:59",
                                        "2024-12-05 12:00"])).astype("datetime64[ns]")
    m = rs.window_selection(t_start, {"w": (days, rs.COMPARE_MONTHS), "summer": (days, (6, 7, 8))})
    assert m["w"].tolist() == [True, False, True, False] and not m["summer"].any()
    tz = rs.window_selection(t_start.dt.tz_localize("UTC"), {"w": (days, rs.COMPARE_MONTHS)})
    assert tz["w"].tolist() == m["w"].tolist()


def test_cohort_strata():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    coh = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])  # cohort 0 all migrants, cohort 2 all residents
    assert rs.informative_strata(y, coh) == (1, 4) and rs.informative_strata(y, None) == (1, 10)
    lab = pd.Series(y, index=[f"b{i}" for i in range(10)])
    V = pd.DataFrame({"v": np.r_[np.full(4, 1.0), np.zeros(6)]}, index=lab.index)
    r = rs.compare_table(V, lab, pd.Series("f", index=lab.index), "f", "g", 99, np.random.default_rng(0), True,
                         pd.Series(coh.astype(str), index=lab.index))[0]
    # only cohort 1 holds both groups: 2 migrants, 2 residents there, so the test is not run although 4 migrants exist
    assert r["n_migrant_all_cohorts"] == 4 and r["n_migrant"] == 2 and r["n_resident"] == 2
    assert r["informative_strata"] == 1 and r["test"].startswith("not tested")
    y2 = np.r_[np.ones(6), np.zeros(6), np.ones(2), np.zeros(4)].astype(int)
    coh2 = np.r_[np.zeros(12), np.ones(6)].astype(int).astype(str)
    lab2 = pd.Series(y2, index=[f"c{i}" for i in range(18)])
    v2 = np.where(y2 == 1, 1.0, 0.0) + np.r_[np.zeros(12), np.full(6, 5.0)]  # cohort 1 sits 5 higher
    r2 = rs.compare_table(pd.DataFrame({"v": v2}, index=lab2.index), lab2, pd.Series("f", index=lab2.index), "f",
                          "g", 199, np.random.default_rng(0), True, pd.Series(coh2, index=lab2.index))[0]
    # unequal group shares per cohort: a pooled mean difference would mix in the cohort level (0.25 here); the
    # within-cohort statistic recovers the planted difference of 1
    assert r2["informative_strata"] == 2 and "within cohort" in r2["test"] and r2["p_perm"] < 0.05
    assert r2["diff_mean"] == pytest.approx(1.0)
    d, p = rs.perm_strat_diff(v2, y2, np.zeros(18, int), 99, np.random.default_rng(0))
    assert d == pytest.approx(v2[y2 == 1].mean() - v2[y2 == 0].mean())


def test_holm():
    assert np.allclose(rs.holm(np.array([0.01, 0.04, np.nan, 0.03])), [0.03, 0.06, np.nan, 0.06], equal_nan=True)


# ---------------------------------------------------------------- end to end

D = 8


def _synthetic(tmp_path):
    rng = np.random.default_rng(0)
    gps, win, run = tmp_path / "gps", tmp_path / "windows", tmp_path / "runs" / "r6x"
    s2, s6 = tmp_path / "states_k2", tmp_path / "states_k6"
    for p in (gps, win, run / "embeddings", s2, s6):
        p.mkdir(parents=True)
    kinds = ["migrant"] * 6 + ["resident"] * 6 + ["disperser"] * 2
    rows2, rows6, ref = [], [], []
    for i, kind in enumerate(kinds):
        b = f"B{i:02d}"
        young = kind == "disperser"
        dates, n, e = _path(kind, 430, rng, start="2024-06-20" if young else "2024-10-15")
        ts = pd.DatetimeIndex(np.concatenate([dates + pd.Timedelta(hours=h) for h in (2, 8, 14, 20)]))
        ts = ts.tz_convert(None) if ts.tz is not None else ts
        nn, ee = np.tile(n, 4), np.tile(e, 4)
        g = pd.DataFrame({"timestamp": ts.astype("datetime64[ms]"),
                          "location-long": LON0 + ee / (DEG_KM * np.cos(np.radians(LAT0))),
                          "location-lat": LAT0 + nn / DEG_KM, "ground-speed": 1.0})
        g.sort_values("timestamp").to_parquet(gps / f"{b}.parquet")
        # windows: 12 per day on every 3rd day
        wt = pd.DatetimeIndex(np.concatenate([dates[::3] + pd.Timedelta(minutes=30 * k) for k in range(12)]))
        wt = (wt.tz_localize("UTC") if wt.tz is None else wt).sort_values()
        N = len(wt)
        mig = kind == "migrant"
        k = (rng.random(N) < 0.3).astype(int)
        p_fly = np.where(k == 1, 0.95, 0.45 if mig else 0.2)
        st2 = (rng.random(N) < p_fly).astype(int)
        st6 = np.where(st2 == 1, rng.integers(2, 6, N), rng.integers(0, 2, N))
        speed = np.where(st2 == 1, (12.0 if mig else 10.0) + rng.normal(0, 0.5, N), np.nan)
        speed[(k == 0)] = np.nan
        meta = pd.DataFrame({"bird": b, "tag": "T", "t_start": wt, "segment": np.arange(N), "k_in_segment": 0,
                             "sensitivity_k": k, "acc_samples": 200, "gps_fixes": np.where(np.isnan(speed), 0, 10)})
        feats = pd.DataFrame({"speed_mean": speed, "turn_abs_mean": rng.uniform(2, 20, N),
                              "climb_mean": rng.normal(0, 1, N)})
        (win / b).mkdir()
        meta.to_parquet(win / b / "meta.parquet")
        feats.to_parquet(win / b / "features.parquet")
        h = rng.normal(0, 1, (N, D)) + (1.0 if mig else -1.0) * np.eye(D)[0] + st2[:, None] * np.eye(D)[1]
        np.savez(run / "embeddings" / f"{b}.npz", h=h.astype(np.float16))
        for rows, s in ((rows2, st2), (rows6, st6)):
            rows.append(pd.DataFrame({"bird": b, "row": np.arange(N), "state": s, "state_hmm": s}))
        comment = "Tagged as a nestling. Band XY" if young else ("migrant golden eagle" if i < 4 else "")
        ref.append([str(i), f"{b}; {100 + i}", str(dates[0].date()) + " 00:00:00.000", "", "", "f" if i % 2 else "m",
                    "", comment, "M1", "fw1"])
    pd.concat(rows2).to_parquet(s2 / "states.parquet")
    pd.concat(rows6).to_parquet(s6 / "states.parquet")
    refp = tmp_path / "reference data (1).csv"
    pd.DataFrame(ref, columns=["tag-id", "animal-id", "deploy-on-date", "deploy-off-date", "animal-comments",
                               "animal-sex", "animal-life-stage", "deployment-comments", "tag-model", "tag-firmware"]
                 ).to_csv(refp, index=False)
    return gps, win, refp, s2, s6, run


def test_end_to_end_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "C_GRID", (0.1,))
    gps, win, refp, s2, s6, run = _synthetic(tmp_path)
    out = tmp_path / "out"
    args = ["--gps", str(gps), "--windows", str(win), "--reference", str(refp), "--states2", str(s2),
            "--states6", str(s6), "--run", str(run), "--out", str(out), "--n-perm", "19", "--n-repeats", "2",
            "--workers", "1"]
    rs.main(args)
    for f in ["residency.csv", "agreement.csv", "residency_summary.md", "nsd_birds.png", "nsd_by_strategy.png",
              "behaviour_groups.csv", "behaviour_tests.csv", "behaviour_birds.csv", "classifiers.csv", "nested.csv", "metadata.json"]:
        assert (out / f).exists(), f
    R = pd.read_csv(out / "residency.csv").set_index("bird")
    assert (R.loc[[f"B{i:02d}" for i in range(6)], "strategy"] == "migrant").all()
    assert (R.loc[[f"B{i:02d}" for i in range(6, 12)], "strategy"] == "resident_full").all()
    assert (R.loc[["B12", "B13"], "strategy"] == "juvenile_dispersal").all() and R.loc[["B12", "B13"], "young"].all()
    assert (R.loc[[f"B{i:02d}" for i in range(6)], "winter_basin_share"] > 0.98).all()
    G = pd.read_csv(out / "behaviour_groups.csv").set_index("grouping")
    assert set(G.index) == {f"{v}|{m}|{a}" for v in rs.VARIANTS for m in ("rule", "consensus") for a in ("adult", "all")}
    g = G.loc["c2024_novmar_basin|rule|adult"]
    assert g.migrants == 6 and g.residents == 6 and g.migrants_with_windows == 6 and g.residents_with_windows == 6
    assert g.migrants_median_days > 30 and "Dec" in g.migrants_month_mix and "Jul" not in g.migrants_month_mix
    assert R["migrant_flag"].sum() == 4
    km = [c for c in R if c.endswith("_km")]
    assert (R[km].dropna() == R[km].dropna().round(0)).all().all()  # whole km
    assert not [c for c in R if "lat" in c.lower() or "lon" in c.lower()]
    T = pd.read_csv(out / "behaviour_tests.csv")
    a = T[(T.grouping == "c2024_novmar_basin|rule|adult") & (T.family == "a_shares_scheduled")
          & (T.variable == "share_k2_flight_sched")].iloc[0]
    assert a.n_migrant == 6 and a.n_resident == 6 and a.diff_mean > 0.15 and a.p_perm < 0.1
    sp = T[(T.grouping == "c2024_novmar_basin|rule|adult") & (T.variable == "speed_ms_k2_flight")].iloc[0]
    assert sp.diff_mean > 1.5
    assert (T[T.family == "a_shares_all_modes"].test == "descriptive (not tested)").all()
    clf = pd.read_csv(out / "classifiers.csv")
    mh = clf[(clf.target == "migrant_vs_resident|c2024_novmar_basin|rule|adult") & (clf.feature_set == "mean_h")]
    assert len(mh) == 3 and (mh.auc_mean > 0.9).all()  # planted offset in h
    nest = pd.read_csv(out / "nested.csv")
    assert set(nest.run) == {"r6x:flight", "r6x:ground", "r6x:flight_ground"} and (nest.n_perm == 19).all()
    # no planted coordinate or comment text anywhere in the text outputs
    text = "".join((out / f).read_text() for f in ["residency.csv", "agreement.csv", "residency_summary.md",
                                                  "behaviour_tests.csv", "behaviour_birds.csv", "behaviour_groups.csv",
                                                  "metadata.json"])
    for s in ("44.43", "108.29", "Band XY"):
        assert s not in text
    assert json.loads((out / "metadata.json").read_text())["thresholds"]["MIN_DISP_KM"] == 200
    with pytest.raises(SystemExit):
        rs.main(args)
