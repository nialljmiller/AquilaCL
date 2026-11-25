import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aquilacl import acc, gps, profile

GAIN, OFFSET = 512.0, np.array([2040.0, 2055.0, 2035.0])


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


# (bird, tag, burst length, counts per g, quiet night bursts per day, true k, k source, calibration model)
# 24- and 40-sample bursts interleave within the same nights at different settings; bird B2's 40-sample
# group has too few quiet bursts, so it takes the k of the 40-sample groups of other birds (B1) and its
# calibration is scaled from the tag's k=1 fit.
GROUPS = [
    ("B1", "5001", 24, 1024.0, 6, 1, "measured", None),
    ("B1", "5001", 40, 512.0, 6, 0, "measured", None),
    ("B2", "5002", 24, 1024.0, 6, 1, "measured", None),
    ("B2", "5002", 40, 512.0, 1, 0, "length_majority", "scaled_from_k1"),
]


def test_sensitivity_per_burst_type(tmp_path):
    rng = np.random.default_rng(5)
    ev, days = tmp_path / "events", pd.date_range("2025-05-01", periods=10, freq="D")
    rows, fixes = [], []
    for bird, tag, n, cpg, per_night, *_ in GROUPS:
        for d in days:
            for h in range(5, 5 + per_night):  # local night (UTC-6), still, varied posture
                th, ph = np.radians(rng.uniform(0, 90)), rng.uniform(0, 2 * np.pi)
                gv = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
                c = gv * cpg + OFFSET + rng.normal(0, 1.5, (n, 3))
                rows.append((bird, tag, d + pd.Timedelta(hours=h, minutes=n), c))
            for h in range(16, 20):  # daytime flapping
                t = np.arange(n) / 20
                gv = np.c_[np.zeros(n), np.zeros(n), 1 + 0.6 * np.sin(2 * np.pi * 2.5 * t)]
                rows.append((bird, tag, d + pd.Timedelta(hours=h, minutes=n), gv * cpg + OFFSET))
    for bird in {g[0] for g in GROUPS}:
        f = pd.DataFrame([{"timestamp": d + pd.Timedelta(hours=h), gps.LON: -108.0, gps.LAT: 44.0, gps.SPEED: 12.0,
                           gps.ALT: 1500.0} for d in days for h in range(14, 26)])
        f["eobs:status"], f[gps.HACC], f["gps:satellite-count"] = "A", 3.0, 10.0
        _write(ev / "gps" / f"{bird}.parquet", f)
        r = sorted((x for x in rows if x[0] == bird), key=lambda x: x[2])
        _write(ev / "acceleration" / f"{bird}.parquet", pd.DataFrame({
            "timestamp": [x[2] for x in r], "tag-local-identifier": [x[1] for x in r],
            "data-decoding-software": "21", "eobs:acceleration-sampling-frequency-per-axis": 20.0,
            "eobs:accelerations-raw": [np.clip(x[3].round(), 0, 4095).astype("int16").ravel() for x in r]}))

    profile.main([str(tmp_path), "--workers", "1"])
    prof = tmp_path / "profile"
    cal = pd.read_csv(prof / "calibration.csv", dtype={"tag": str}).set_index(["bird", "tag", "sensitivity_k"])
    for bird, tag, n, cpg, _, k, source, model in GROUPS:
        b = pd.read_parquet(prof / "acc" / f"{bird}.parquet")
        b = b[b["n_samples"] == n]
        assert (b["sensitivity_k"] == k).all() and (b["k_source"] == source).all()
        assert np.abs(b.loc[b["still"], "static_norm"] - 1).max() < 0.02  # calibrated still |g|
        c = cal.loc[(bird, tag, k)]
        assert c["reliable"] and abs(c["median_still_g"] - 1) < 0.02
        if model:
            assert c["model"] == model
            np.testing.assert_allclose(c[["gain_x", "gain_y", "gain_z"]].to_numpy(float),
                                       cal.loc[(bird, tag, 1), ["gain_x", "gain_y", "gain_z"]].to_numpy(float) / 2)

    # format.json: recomputed from calibration.csv, and close to the generating values
    fmt = json.loads((prof / "format.json").read_text())
    gain = cal[["gain_x", "gain_y", "gain_z"]].to_numpy(float)
    off = cal[["offset_x", "offset_y", "offset_z"]].to_numpy(float)
    k = cal.index.get_level_values("sensitivity_k").to_numpy()
    step = (1 / gain[k == 0]).max()
    clip = (np.minimum(off, 4095 - off) / gain)[k == 1].min() - 0.05
    np.testing.assert_allclose([fmt["step_g"], fmt["clip_g"]], [step, clip], rtol=1e-12)
    np.testing.assert_allclose([fmt["step_g"], fmt["clip_g"]], [1 / 512, OFFSET.min() / 1024 - 0.05], rtol=0.02)
    assert len(fmt["step_g_periods"]) == 2 and len(fmt["clip_g_periods"]) == 2


def test_profile_bird_end_to_end(tmp_path):
    """Synthetic bird: GPS only in daytime; still bursts at night; flapping bursts by day."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-05-01", periods=10, freq="D", tz=None)
    fixes, bursts = [], []
    for d in days:
        for h in range(14, 26):  # 14-02 UTC = local daytime in Wyoming (UTC-6)
            t = d + pd.Timedelta(hours=h)
            fixes.append({"timestamp": t, gps.LON: -108.0, gps.LAT: 44.0, gps.SPEED: 12.0, gps.ALT: 1500.0})
        for h in range(5, 11):  # local night: still, slightly varying posture
            tilt = np.radians(rng.uniform(0, 25))
            g = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
            c = (g * GAIN + OFFSET + rng.normal(0, 1.5, (24, 3))).round().astype("int16")
            bursts.append({"timestamp": d + pd.Timedelta(hours=h), "raw": c.ravel()})
        for h in range(16, 22):  # daytime flapping at 2.5 Hz
            t = np.arange(264) / 20
            g = np.c_[np.zeros(264), np.zeros(264), 1 + 0.6 * np.sin(2 * np.pi * 2.5 * t)]
            c = (g * GAIN + OFFSET).round().astype("int16")
            bursts.append({"timestamp": d + pd.Timedelta(hours=h, minutes=1), "raw": c.ravel()})
    ev = tmp_path / "events"
    f = pd.DataFrame(fixes)
    f["eobs:status"], f[gps.HACC], f["gps:satellite-count"] = "A", 3.0, 10.0
    _write(ev / "gps" / "B1.parquet", f)
    b = pd.DataFrame(bursts)
    acc_df = pd.DataFrame(
        {
            "timestamp": b["timestamp"],
            "tag-local-identifier": "15000",
            "data-decoding-software": "21",
            "eobs:acceleration-sampling-frequency-per-axis": 20.0,
            "eobs:accelerations-raw": b["raw"],
        }
    )
    _write(ev / "acceleration" / "B1.parquet", acc_df)

    sched, cal = profile.profile_bird(ev, "B1", tmp_path / "profile", 8.0, 120.0, 3.0)
    c = cal[0]
    assert len(cal) == 1 and c["sensitivity_k"] == 0
    assert c["still_bursts"] == 60  # all night bursts, no flight bursts
    assert abs(np.mean([c["gain_x"], c["gain_y"], c["gain_z"]]) / GAIN - 1) < 0.06
    out = pd.read_parquet(tmp_path / "profile" / "acc" / "B1.parquet")
    day = out[~out["still"]]
    assert np.allclose(day["dom_freq_hz"], 2.5, atol=0.1)
    assert (out.loc[out["still"], "sun_elev_deg"] < -6).all()
    assert {r["stream"] for r in sched} == {"gps", "acceleration"}


def test_unreliable_fit_is_replaced_by_scaled_donor():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    good = 2048 + 512 * u[:100]  # k=0: clean still bursts in many postures
    bad = 2048 + 1024 * u[100:] * rng.uniform(0.6, 1.4, (100, 1))  # k=1: 60 "still" bursts, norms all over
    means = np.r_[good, bad]
    k = np.r_[np.zeros(100, int), np.ones(100, int)]
    cals = acc.calibrate_tag(means, np.ones(200, bool), k)
    assert not acc.calibration_flags(cals[0])
    assert cals[1].model == "scaled_from_k0"
    np.testing.assert_allclose(cals[1].gain, 2 * cals[0].gain)
