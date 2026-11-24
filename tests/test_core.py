import numpy as np
import pandas as pd
import pytest

from goldeneai import acc, gps


def _random_postures(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_calibration_recovers_offset_and_gain():
    rng = np.random.default_rng(1)
    offset, gain = np.array([2030.0, 2071.0, 2002.0]), np.array([1010.0, 985.0, 1032.0])
    bursts = []
    for g in _random_postures(rng, 400):
        counts = g * gain + offset + rng.normal(0, 2, size=(40, 3))
        bursts.append(counts)
    for _ in range(100):  # moving bursts must be ignored
        bursts.append(offset + rng.normal(0, 300, size=(40, 3)))
    cal = acc.fit_calibration(bursts)
    assert cal.fitted
    np.testing.assert_allclose(cal.offset, offset, atol=3)
    np.testing.assert_allclose(cal.gain, gain, rtol=0.01)
    assert cal.residual_g < 0.01


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
