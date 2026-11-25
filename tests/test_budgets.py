import json

import numpy as np
import pandas as pd
import pytest

from aquilacl import budgets as bu


# ---------------------------------------------------------------- recording structure

def test_scheduled_mask_first_window_of_each_burst():
    t = np.array([0.0, 10.0, 300.0, 600.0, 610.0, 620.0, 900.0, 905.0])
    k = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    assert bu.scheduled_mask(t, k).tolist() == [True, False, True, True, False, False, False, True]


@pytest.mark.parametrize("gaps, expect", [([300] * 30 + [600] * 5 + [310, 295], 300.0), ([900] * 10 + [1800] * 3, 900.0),
                                           ([5, 10], np.nan)])
def test_interval_mode(gaps, expect):
    I, _ = bu.interval_mode(gaps)
    assert (np.isnan(I) and np.isnan(expect)) or I == expect


def test_night_schedule_gets_its_own_interval_and_cap():
    day = np.arange(0, 12 * 3600, 300.0)
    night = np.arange(12 * 3600, 24 * 3600, 3600.0)
    ts = np.r_[day, night, day + 86400, night + 86400]
    is_day = np.r_[np.ones(len(day), bool), np.zeros(len(night), bool)]
    is_day = np.r_[is_day, is_day]
    I, nominal, tab, _ = bu.schedule(ts, np.full(len(ts), "2024-08"), is_day)
    assert nominal == 300 and set(I[~is_day]) == {3600.0} and set(I[is_day]) == {300.0}
    e = bu.coverage_ends(ts, I, np.zeros(0), impute=False)
    assert (e - ts)[:-1].sum() == pytest.approx(ts[-1] - ts[0])  # no time lost at night
    r = tab[tab.period == "night"].iloc[0]
    assert r.interval_s == 3600 and r.n_gaps >= bu.MIN_GAPS


def _timeline(ts, bs, be, impute, I=300.0):
    ts, bs, be = (np.asarray(x, float) for x in (ts, bs, be))
    e = bu.coverage_ends(ts, np.full(len(ts), I), bs, impute)
    ps, pe, _ = bu.uncovered_pieces(ts, e, bs, be) if impute else (np.zeros(0), np.zeros(0), None)
    return e, ps, pe


@pytest.mark.parametrize("case, ts, impute, sched_s, imputed_s", [
    # 300-s clock, bout 1000-1900 with the slots at 1200/1500/1800 suppressed
    ("suppressed, imputed", [0, 300, 600, 900, 2100, 2400], True, 900 + 100 + 300 + 450, 900),
    ("suppressed, plain", [0, 300, 600, 900, 2100, 2400], False, 900 + 450 + 300 + 450, 0),
    # schedule continues inside the bout: only bout start -> first internal sample is imputed
    ("continuing, imputed", [0, 300, 600, 900, 1200, 1500, 1800, 2100], True, 900 + 100 + 900 + 450, 200),
])
def test_weights_cap_and_imputation(case, ts, impute, sched_s, imputed_s):
    e, ps, pe = _timeline(ts, [1000.0], [1900.0], impute)
    assert (e - np.asarray(ts, float)).sum() == pytest.approx(sched_s), case
    assert (pe - ps).sum() == pytest.approx(imputed_s), case
    # no double counting: intervals are disjoint
    iv = sorted(list(zip(ts, e)) + list(zip(ps, pe)))
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(iv, iv[1:])), case


def test_bouts_from_windows():
    t = np.array([0, 100, 110, 120, 400, 700, 710], float)
    k = np.array([0, 1, 1, 1, 0, 1, 1])
    acc = np.full(7, 200)
    s, e, bid = bu.bouts_from_windows(t, k, acc)
    assert s.tolist() == [100, 700] and e.tolist() == [130, 720] and bid.tolist() == [-1, 0, 0, 0, -1, 1, 1]


def test_split_bins_and_solar_day_across_utc_midnight():
    src, b, sec = bu.split_bins(np.array([3000.0, 0.0]), np.array([8000.0, 10.0]), 3600.0)
    assert sec.sum() == pytest.approx(5010) and b.tolist() == [0, 1, 2, 0]
    # 23:00-01:00 UTC at a -7 h offset is 16:00-18:00 local: one solar day, not two
    t = np.array([86400 - 3600.0, 86400 + 3600.0])
    off, fb = bu.solar_offset_h(t, np.array([16.0, np.nan]))
    assert not fb and off.tolist() == [-7.0, -7.0]
    src, b, _ = bu.split_bins(t[:1] + off[0] * 3600, t[1:] + off[1] * 3600, 86400.0)
    assert len(set(b)) == 1


# ---------------------------------------------------------------- solar-hour stratified daily budgets

def _H(days, hours, flight_share, cover=3600.0):
    """Hourly table (index = absolute local hour bin) with K=2 seconds; flight_share: hour -> share."""
    rows = [(d * 24 + h, cover * flight_share.get(h, 0.0), cover * (1 - flight_share.get(h, 0.0)))
            for d in days for h in hours]
    H = pd.DataFrame(rows, columns=["hb", "k2_flight", "k2_ground"]).set_index("hb")
    return H.assign(n_sched=12)


JULY = 19905  # 2024-07-01 as days since 1970-01-01
CATS = {"k2": ["k2_flight", "k2_ground"]}


def test_stratified_vs_scaled_with_night_off_window():
    day_hours = range(5, 20)  # tag records 05-20 local solar only
    Ha = _H([JULY, JULY + 1], day_hours, {h: 0.5 for h in (10, 11, 12, 13)})
    night = [_H([JULY, JULY + 1], range(24), {}) for _ in range(3)]  # birds that record at night: all ground
    pop = bu.population_cells([Ha] + night, CATS)
    S = bu.stratify(Ha, CATS, pop)
    assert np.allclose(S.d_k2_flight, 2.0) and np.allclose(S.d_k2_flight + S.d_k2_ground, 24)
    assert (S.n_direct == 15).all() and (S.h_fill_population == 9).all() and (S.h_unfilled == 0).all()
    assert np.allclose(S.d_flight_from_population, 0.0)
    scaled = 24 * Ha.k2_flight.sum() / (Ha.k2_flight.sum() + Ha.k2_ground.sum())
    assert scaled == pytest.approx(2.0 * 24 / 15)  # the first estimator inflates flight by 24 / covered hours
    own = bu.stratify(Ha, CATS, None)  # own data only: the night cannot be filled
    assert (own.h_unfilled == 9).all() and own.d_k2_flight.isna().all()
    # fewer than POP_MIN_BIRDS night recorders: no population cell for the night
    assert bu.stratify(Ha, CATS, bu.population_cells([Ha] + night[:2], CATS)).h_unfilled.eq(9).all()


def test_fill_order_bird_month_then_bird_then_unqualified():
    fs = {12: 0.5}
    H = pd.concat([_H([JULY, JULY + 1], range(24), fs),
                   _H([JULY + 2], [h for h in range(24) if h != 12], fs),        # July, hour 12 missing -> bird-month
                   _H([JULY + 31], [h for h in range(24) if h != 12], fs),       # August, no August hour 12 -> bird
                   _H([JULY + 32], range(24), fs, cover=300.0)])                  # all bins below MIN_BIN_S -> filled
    S = bu.stratify(H, CATS, None).set_index("dayn")
    assert S.loc[JULY + 2, "h_fill_bird_month"] == 1 and S.loc[JULY + 2, "h_fill_bird"] == 0
    assert S.loc[JULY + 31, "h_fill_bird"] == 1 and S.loc[JULY + 31, "h_fill_bird_month"] == 0
    assert np.allclose(S.d_k2_flight, 0.5)
    assert S.loc[JULY + 32, "n_direct"] == 0 and S.loc[JULY + 32, "direct_mask"] == "-" * 24
    D = S.reset_index().assign(d_flight=S.d_k2_flight.to_numpy())
    assert bu.qualifies(D, 12, "population").tolist() == [True, True, True, True, False]


# ---------------------------------------------------------------- tests between birds

def test_month_combined_test_null_and_planted():
    rng = np.random.default_rng(0)
    birds = [f"B{i:02d}" for i in range(16)]
    lab = pd.Series(np.r_[np.zeros(8), np.ones(8)], index=birds)
    ps = []
    for _ in range(60):
        V = pd.DataFrame(rng.normal(size=(16, 4)), index=birds, columns=[1, 2, 3, 4])
        V.iloc[rng.random((16, 4)) < 0.2] = np.nan
        ps.append(bu.month_combined_test(V, lab, 99, rng)[0]["p_perm"])
    ps = np.array(ps)
    assert (ps <= 0.05).mean() <= 0.15 and 0.3 < ps.mean() < 0.7
    bird_effect = rng.normal(size=16)[:, None]  # same bird in every month
    V = pd.DataFrame(bird_effect + rng.normal(0, 0.3, (16, 4)) + 2 * lab.to_numpy()[:, None], index=birds, columns=[1, 2, 3, 4])
    s, per = bu.month_combined_test(V, lab, 199, rng)
    assert s["combined_diff_h"] > 1 and s["p_perm"] < 0.01 and per.in_combined.all()


# ---------------------------------------------------------------- end to end on synthetic data

def _write_bird(root, name, rng, flight_bias, days=7, night_off=False):
    t0 = pd.Timestamp("2024-08-01 07:00", tz="UTC")  # local solar midnight at a -7 h offset
    bouts = [(d * 86400 + h * 3600 + 120, 600) for d in range(days) for h in (18, 21)]  # 11:00 and 14:00 local
    rows = []
    for seg, (s, n) in enumerate(bouts):
        rows += [(s + 10 * j, 1, 200, 10_000 + seg, j) for j in range(n // 10)]
    for i, t in enumerate(np.arange(0, days * 86400, 300.0)):
        if night_off and not 5 <= (t / 3600) % 24 < 20:  # tag records 05-20 local only
            continue
        if not any(s <= t < s + n for s, n in bouts):  # schedule suppressed during bouts
            rows.append((t, 0, 40, i, 0))
    rows.sort()
    ts = np.array([r[0] for r in rows])
    n = len(ts)
    k = np.array([r[1] for r in rows])
    hour_local = ((ts / 3600) % 24)
    meta = pd.DataFrame({"bird": name, "tag": "T", "t_start": t0 + pd.to_timedelta(ts, unit="s"), "segment": [r[3] for r in rows],
                         "k_in_segment": [r[4] for r in rows], "segment_windows": 1, "sensitivity_k": k, "decoding": "x",
                         "sun_elev_deg": np.where((hour_local > 6) & (hour_local < 19), 20.0, -20.0),
                         "acc_samples": [r[2] for r in rows], "gps_fixes": np.where(k == 1, 10, 0),
                         "local_solar_hour": hour_local})
    d = root / "windows" / name
    d.mkdir(parents=True)
    meta.to_parquet(d / "meta.parquet")
    pd.DataFrame({"speed_mean": np.where(k == 1, 12.0, np.nan)}).to_parquet(d / "features.parquet")
    day = (hour_local > 8) & (hour_local < 17)
    s2 = np.where(k == 1, 1, (day & (rng.random(n) < flight_bias)).astype(int))
    s6 = np.where(s2 == 1, rng.choice([2, 3, 4, 5], n), rng.choice([0, 1], n))
    sub = np.where(s2 == 0, rng.integers(0, 4, n), 0)
    return (pd.DataFrame({"bird": name, "row": np.arange(n), "state": s2, "state_hmm": s2}),
            pd.DataFrame({"bird": name, "row": np.arange(n), "state": s6, "state_hmm": s6}),
            pd.DataFrame({"bird": name, "row": np.arange(n), "state": s2, "substate": sub,
                          "label": [f"{a}.{b}" for a, b in zip(s2, sub)]}))


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("bud")
    rng = np.random.default_rng(1)
    names = [f"bird{i}" for i in range(8)]
    parts = [_write_bird(root, b, rng, 0.2 + 0.3 * (i % 2), night_off=i % 4 in (1, 2)) for i, b in enumerate(names)]
    for j, d in enumerate(("s2", "s6", "sub")):
        (root / d).mkdir()
        pd.concat([p[j] for p in parts]).to_parquet(root / d / ("substates.parquet" if d == "sub" else "states.parquet"))
    pd.DataFrame({"animal-id": names, "animal-sex": ["f", "m"] * 4, "deploy-on-date": "2024-07-01",
                  "animal-comments": ["migrant", "nestling", "", "", "", "", "", ""], "deployment-comments": ""}).to_csv(root / "ref.csv", index=False)
    out = root / "out"
    bu.run(root / "windows", root / "s2", root / "s6", root / "sub", root / "ref.csv", out, n_boot=20, n_perm=49, workers=1)
    return out


def test_end_to_end_outputs(synthetic_run):
    out = synthetic_run
    for f in ("birds.csv", "schedule.csv", "birddays.csv", "budgets_by_group.csv", "diel.csv", "window_vs_time.csv",
              "flight_composition.csv", "tests.csv", "coverage_sensitivity.csv", "summary.md", "monthly_budget.png"):
        assert (out / f).exists(), f
    chk = json.loads((out / "recording_check.json").read_text())
    assert chk["share_slots_found_pooled"] < 0.2 and chk["headline_variant"] == "imputed" and chk["birds_imputed"] == 8
    assert chk["share_slot_windows_flight"] == 1.0
    D = pd.read_csv(out / "birddays.csv")
    Q = D[D.qualifies]
    assert len(Q) and np.allclose(Q.d_flight + Q.d_ground, 24)
    full = Q[Q.coverage > 0.9]
    # every bout adds 600 s of imputed flight: 2 bouts per day
    assert len(full) and np.allclose(full.h_k2_flight_imputed, 2 * 600 / 3600, atol=0.05)
    # full-day recorders: stratified = scaled; night-off birds: night filled from the population (no flight there),
    # and the first estimator inflates flight (about 24 / 15 covered hours; the 21:00 bout lies in the night)
    assert np.allclose(full.d_flight, full.d_flight_scaled_24h, rtol=0.02)
    off = Q[Q.bird.isin(["bird1", "bird2", "bird5", "bird6"])]
    assert len(off) and (off.h_fill_population == 9).all() and np.allclose(off.d_flight_from_population, 0)
    assert np.allclose(off.d_flight_scaled_24h / off.d_flight, 24 / 15, rtol=0.03)
    W = pd.read_csv(out / "window_vs_time.csv").set_index(["system", "state"])
    assert W.loc[("k2", "flight"), "window_share_all"] > W.loc[("k2", "flight_total"), "time_share"]
    tests = pd.read_csv(out / "tests.csv")
    main = tests[tests.main & (tests.test == "sex")].iloc[0]
    assert main.months_used == 1 and main.n_birds_1 == 4 and np.isfinite(main.p_perm)
    sens = pd.read_csv(out / "coverage_sensitivity.csv")
    rules = set(zip(sens.min_direct_bins, sens.fill))
    assert rules == {(12, "population"), (8, "population"), (14, "population"), (12, "own")}
    m0 = sens[sens.month == 0].set_index(["min_direct_bins", "fill"])
    assert m0.loc[(12, "own"), "n_birds"] == 4  # night-off birds need the population fill
    cmp24 = pd.read_csv(out / "scaled_24h_comparison.csv")
    assert set(cmp24.estimator) == {"scaled_24h", "stratified"}
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["diel_check"]["diel_hours_with_data"] == 24


def test_outputs_hold_no_positions_or_utc_times(synthetic_run):
    for f in synthetic_run.glob("*.csv"):
        cols = {c.lower() for c in pd.read_csv(f, nrows=1).columns}
        assert not cols & {"t_start", "timestamp", "lat", "lon", "location-lat", "location-long", "offset", "offset_h"}, f
        assert not any("utc" in c or "lat" == c[:3] for c in cols), f
