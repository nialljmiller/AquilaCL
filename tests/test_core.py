from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquilacl import acc, gps, solar


def _postures(rng, n, max_tilt_deg):
    """Unit gravity vectors within ``max_tilt_deg`` of the tag's +z axis."""
    tilt = np.radians(rng.uniform(0, max_tilt_deg, n))
    az = rng.uniform(0, 2 * np.pi, n)
    return np.c_[np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)]


OFFSET, GAIN = np.array([2030.0, 2071.0, 2002.0]), np.array([1010.0, 985.0, 1032.0])


@pytest.mark.parametrize(
    "max_tilt, banked_frac, use_mask, gain_rtol, expect_model",
    [
        (180, 0.0, False, 0.01, "full"),  # postures over the whole sphere: full fit
        # Upright perching only: offset and gain along z are confounded, so the fit falls back to
        # a reduced model; about 4% gain error is the identifiability limit, not a fitting fault.
        (30, 0.0, False, 0.06, "shared_gain"),
        (30, 0.3, True, 0.06, "shared_gain"),  # banked-soaring bursts excluded by the stillness mask
    ],
)
def test_calibration(max_tilt, banked_frac, use_mask, gain_rtol, expect_model):
    rng = np.random.default_rng(1)
    n = 600
    g = _postures(rng, n, max_tilt)
    banked = rng.random(n) < banked_frac
    g[banked] *= 1 / np.cos(np.radians(30))  # steady 30-degree bank: 1.15 g, low variation
    means = g * GAIN + OFFSET + rng.normal(0, 2, size=(n, 3))
    cal = acc.fit_calibration(means, still=~banked if use_mask else None)
    if expect_model:
        assert cal.model == expect_model
    np.testing.assert_allclose(cal.gain.mean(), GAIN.mean(), rtol=gain_rtol)
    assert cal.residual_g < 0.02


def test_calibration_without_mask_is_biased_by_banking():
    rng = np.random.default_rng(2)
    g = _postures(rng, 600, 30)
    banked = rng.random(600) < 0.3
    g[banked] *= 1 / np.cos(np.radians(30))
    masked = acc.fit_calibration(g * GAIN + OFFSET, still=~banked)
    unmasked = acc.fit_calibration(g * GAIN + OFFSET)
    assert unmasked.residual_g > 3 * masked.residual_g


@pytest.mark.parametrize(
    "freq, amp, expect_freq",
    [(3.0, 0.5, 3.0), (0.0, 0.0, None)],  # flapping-like oscillation, then still
)
def test_burst_features(freq, amp, expect_freq):
    fs, n = 20.0, 264
    t = np.arange(n) / fs
    g = np.tile([0.0, 0.0, 1.0], (n, 1))
    g[:, 2] += amp * np.sin(2 * np.pi * freq * t)
    f = acc.burst_features(g, fs)
    assert f["static_norm"] == pytest.approx(1.0, abs=0.05)
    if expect_freq is None:
        assert f["vedba"] < 1e-6
    else:
        assert f["dom_freq_hz"] == pytest.approx(expect_freq, abs=0.1)
        assert f["vedba"] == pytest.approx(2 * amp / np.pi, rel=0.15)


def test_sun_elevation():
    # Wyoming, 2025-06-21: local solar noon about 19:05 UTC, midnight about 07:05 UTC
    t = pd.to_datetime(["2025-06-21 19:05", "2025-06-21 07:05"], utc=True)
    el = solar.sun_elevation_deg(t, [-108.0, -108.0], [44.0, 44.0])
    assert el[0] == pytest.approx(90 - 44 + 23.44, abs=1.0)
    assert el[1] < -20


def test_decode_interleaved():
    flat = np.arange(12)
    np.testing.assert_array_equal(acc.decode(flat)[1], [3, 4, 5])


def _thermal(n=120, radius=60.0, v=10.0, climb=2.0, t0="2025-05-01 18:00:00"):
    """1 Hz track circling counter-clockwise while climbing."""
    t = np.arange(n)
    ang = v / radius * t
    east, north = radius * np.cos(ang), radius * np.sin(ang)
    lat = 44.0 + north / gps.EARTH_R * 180 / np.pi
    lon = -108.0 + east / (gps.EARTH_R * np.cos(np.radians(44.0))) * 180 / np.pi
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp(t0) + pd.to_timedelta(t, "s"),
            gps.LON: lon,
            gps.LAT: lat,
            gps.ALT: 1500 + climb * t,
        }
    )


def test_kinematics_thermal_and_gap():
    a = _thermal()
    b = _thermal(n=30, t0="2025-05-01 19:00:00")  # second run after a 1 h gap
    k = gps.kinematics(pd.concat([a, b]))
    assert k["run"].nunique() == 2
    first = k[k.run == k.run.min()]
    assert first["climb_ms"].median() == pytest.approx(2.0, rel=0.01)
    assert first["speed_track_ms"].median() == pytest.approx(10.0, rel=0.02)
    # counter-clockwise circling: turn rate = v / r in degrees per second
    assert first["turn_rate_dps"].median() == pytest.approx(-np.degrees(10 / 60), rel=0.05)
    s = gps.run_summary(k)
    assert abs(s["net_turn_deg"].iloc[0]) > 360 * 3
    assert np.isnan(k[k.run == k.run.max()]["climb_ms"].iloc[0])  # no rate across the gap


@pytest.mark.parametrize(
    "edit, nan_cols, expect_outlier",
    [
        ("jump", ["speed_track_ms", "step_m", "climb_ms"], True),  # 500 m position jump: > 70 m/s
        ("speed_mismatch", ["speed_track_ms", "step_m"], True),  # track 10 m/s vs ground-speed 25 m/s
        ("climb", ["climb_ms"], False),  # +60 m altitude step: climb > 20 m/s
        ("half_second", ["turn_rate_dps"], False),  # fix 0.5 s after the previous one
        ("duplicate", [], False),  # repeated timestamp: first kept, never dt = 0
        ("slow", ["turn_rate_dps"], False),  # 1 m/s: steps under 5 m give no turn rate
    ],
)
def test_kinematics_outliers(edit, nan_cols, expect_outlier):
    a = _thermal(n=20)
    a[gps.SPEED] = 10.0
    i = 10
    if edit == "jump":
        a.loc[i, gps.LAT] += 500 / gps.EARTH_R * 180 / np.pi
    elif edit == "speed_mismatch":
        a[gps.SPEED] = 25.0
    elif edit == "climb":
        a.loc[i:, gps.ALT] += 60
    elif edit == "half_second":
        a.loc[i, "timestamp"] -= pd.Timedelta(seconds=0.5)
    elif edit == "duplicate":
        a = pd.concat([a, a.iloc[[i]].assign(**{gps.LAT: 45.0})]).reset_index(drop=True)
    elif edit == "slow":
        a = _thermal(n=20, v=1.0, radius=6.0)
    k = gps.kinematics(a)
    assert k["timestamp"].is_unique and (k["dt_s"].dropna() > 0).all()
    assert k[gps.LAT].max() < 44.5  # duplicate kept the first fix
    row = k.iloc[i] if edit != "slow" else k.iloc[1:]
    for c in nan_cols:
        assert np.isnan(np.asarray(row[c], dtype=float)).all(), c
    assert bool(np.any(k["pos_outlier"])) == expect_outlier
    assert (k["speed_track_ms"].dropna() <= gps.MAX_TRACK_SPEED_MS).all()
    assert k["climb_ms"].dropna().between(gps.MIN_CLIMB_MS, gps.MAX_CLIMB_MS).all()



def test_normalise_bird_and_compressed_ingest(tmp_path):
    """New-style "<animal>; <tag>" IDs are cut to the animal; a .zst export is read directly."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aquilacl import ingest

    col = pa.array(["0749-00032; 18261", "0799-02952", None])
    assert ingest.normalise_bird(col).to_pylist() == ["0749-00032", "0799-02952", None]
    src = Path(__file__).parents[2] / "reference" / "GoldenEAI_sample.csv"
    if not src.exists():
        pytest.skip("sample CSV not available")
    text = src.read_text().replace(",0799-02952,", ",0799-02952; 14845,")
    zst = tmp_path / "export.csv.zst"
    with pa.output_stream(str(zst), compression="zstd") as f:
        f.write(text.encode())
    counts = ingest.stream_split(zst, tmp_path / "out")
    assert sum(counts.values()) == 200
    birds = {b for _, b in counts}
    assert "0799-02952" in birds and not any(";" in b for b in birds)
