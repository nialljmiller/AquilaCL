import numpy as np
import pandas as pd
import pytest

from aquilacl import features, gps, profile, windows
from test_profile import _write, OFFSET  # noqa: F401  (shared synthetic helpers)
from test_core import _thermal

GAIN = 512.0


def _burst(g):
    return (g * GAIN + OFFSET).round().astype("int16").ravel()


def test_build_bird_windows_and_labels(tmp_path):
    rng = np.random.default_rng(5)
    t0 = pd.Timestamp("2025-05-01 19:00:00")  # local midday in Wyoming
    rows = []
    # 60 s continuous segment: 50 back-to-back 1.2 s bursts of flapping at 2.5 Hz
    for i in range(50):
        t = np.arange(24) / 20 + i * 1.2
        g = np.c_[np.zeros(24), np.zeros(24), 1 + 0.6 * np.sin(2 * np.pi * 2.5 * t)]
        rows.append((t0 + pd.Timedelta(seconds=i * 1.2), _burst(g)))
    # night bouts: a 2.4 s still bout at local 01:00 on several nights (enough still bursts to calibrate)
    for d in range(10):
        tn = pd.Timestamp("2025-05-02 07:00:00") + pd.Timedelta(days=d)
        for j in range(2):
            g = np.tile([0.0, 0.0, 1.0], (24, 1)) + rng.normal(0, 0.002, (24, 3))
            rows.append((tn + pd.Timedelta(seconds=1.2 * j), _burst(g)))
        for j in range(8):  # extra still bursts the same night, spaced out
            rows.append((tn + pd.Timedelta(minutes=10 * (j + 1)), _burst(np.tile([0.0, 0.0, 1.0], (24, 1)))))
    ev = tmp_path / "events"
    acc_df = pd.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "tag-local-identifier": "15000",
            "data-decoding-software": "21",
            "eobs:acceleration-sampling-frequency-per-axis": 20.0,
            "eobs:accelerations-raw": [r[1] for r in rows],
        }
    ).sort_values("timestamp")
    _write(ev / "acceleration" / "B1.parquet", acc_df)
    th = _thermal(n=60, t0=str(t0))  # 1 Hz circling climb, 10 m/s, during the segment
    th[gps.SPEED] = 10.0
    th["eobs:status"], th[gps.HACC], th["gps:satellite-count"] = "A", 3.0, 10.0
    nights_t = [pd.Timestamp("2025-05-02 06:59:00") + pd.Timedelta(days=d) for d in range(10)]
    night = pd.DataFrame({"timestamp": nights_t, gps.LON: -108.0, gps.LAT: 44.0, gps.ALT: 1500.0, gps.SPEED: 0.0,
                          "eobs:status": "A", gps.HACC: 3.0, "gps:satellite-count": 10.0})
    _write(ev / "gps" / "B1.parquet", pd.concat([th, night]))

    prof = tmp_path / "profile"
    sched, cal = profile.profile_bird(ev, "B1", prof, 8.0, 120.0, 3.0)
    pd.DataFrame(cal).to_csv(prof / "calibration.csv", index=False)
    calib = windows.load_calibration(prof / "calibration.csv")
    s = windows.build_bird("B1", ev, prof, calib, tmp_path / "win")
    assert s["samples_dropped_overlap"] == 0 and s["bursts_snapped"] == 0  # exact 1.2 s bursts are already contiguous

    meta = pd.read_parquet(tmp_path / "win" / "B1" / "meta.parquet")
    f = pd.read_parquet(tmp_path / "win" / "B1" / "features.parquet")
    acc = np.load(tmp_path / "win" / "B1" / "acc.npy")
    seg = meta["segment_windows"] == 6
    assert seg.sum() == 6  # 60 s segment -> six 10 s windows
    assert (meta.loc[seg, "acc_samples"] == 200).all()
    day = f[seg.to_numpy()]
    assert np.allclose(day["dom_freq_hz"], 2.5, atol=0.15)
    assert (day["label_flapping"] == 1).all() and (day["label_circling"] == 0).all()  # flapping is not soaring
    assert (day["turn_abs_mean"] > 6).all() and (day["climb_mean"] > 1.5).all()
    nights = f[(meta["sun_elev_deg"] < -6).to_numpy()]
    assert len(nights) >= 10 and (nights["label_night"] == 1).all() and (nights["label_flapping"] != 1).all()
    m = np.load(tmp_path / "win" / "B1" / "acc_mask.npy")[~seg.to_numpy()]
    still = acc[~seg.to_numpy()].astype(float)
    mean_vec = (still * m[:, :, None]).sum(axis=1) / m.sum(axis=1, keepdims=True)
    assert abs(np.linalg.norm(mean_vec, axis=1).mean() - 1) < 0.05  # calibrated still windows read 1 g


@pytest.mark.parametrize(
    "gap_s, jitter_s, snaps, dropped_each, hole_each",
    [
        (1.2, 0.0, False, 0, 0),  # exact continuous mode
        (1.2, 0.035, True, 0, 0),  # jittered starts (1.13-1.27 s apart): snapped, no holes, no collisions
        (0.8, 0.0, False, 8, 0),  # real overlap: the 8 overlapping samples of each later burst are dropped
        (1.7, 0.0, False, 0, 10),  # real 0.5 s gap: placed by time, the hole stays
    ],
)
def test_place_bursts_contiguous(gap_s, jitter_s, snaps, dropped_each, hole_each):
    n = 30
    ts = np.arange(n) * gap_s + np.random.default_rng(3).uniform(-jitter_s, jitter_s, n)
    ts -= ts[0]
    n_samp = np.full(n, 24)
    r = np.round(ts * windows.FS).astype(int)
    start, drop, snapped = windows.place_bursts(r, n_samp, np.zeros(n, int))
    filled = np.zeros(int((start + n_samp).max()), int)
    for s0, d, k in zip(start, drop, n_samp):
        filled[s0 + d:s0 + k] += 1
    assert filled.max() == 1  # never overwritten
    assert (filled == 0).sum() == hole_each * (n - 1)
    assert drop[1:].tolist() == [dropped_each] * (n - 1)
    assert snapped.any() == snaps
    assert (np.abs(start - r) <= windows.SNAP_SLOTS).all()  # every burst stays near its timestamp


@pytest.mark.parametrize("n_present", [200, 64, 40])
def test_features_do_not_depend_on_window_fullness(n_present):
    rng = np.random.default_rng(4)
    t = np.arange(200) / windows.FS
    g = np.c_[np.zeros(200), np.zeros(200), 1 + 0.033 * t] + rng.normal(0, 0.05, (200, 3))  # slow drift + noise
    full = features.acc_features(g[None], np.ones((1, 200), bool)).iloc[0]
    m = np.zeros((1, 200), bool)
    m[0, 50:50 + n_present] = True
    part = features.acc_features(g[None], m).iloc[0]
    assert part["vedba"] == pytest.approx(full["vedba"], rel=0.15)
    assert part["static_norm"] == pytest.approx(full["static_norm"], abs=0.05)
    assert np.isfinite(part["flap_band_frac"])  # spectrum from >= MIN_SPECTRUM_SAMPLES contiguous samples


def test_labels_unknown_without_inputs():
    nan = np.nan
    f = pd.DataFrame({"vedba": [0.02, 0.02, 0.5], "flap_band_frac": [0.1, nan, 0.6], "speed_mean": [0.2, nan, nan],
                      "turn_abs_mean": [nan, nan, nan], "climb_mean": [0.0, nan, nan], "n_speed": [10, 0, 0], "n_turn": [0, 0, 0]})
    lab = features.reference_labels(f, np.array([10.0, 10.0, nan]))
    assert lab.loc[0, ["perched", "flapping", "night", "climbing_soar"]].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert lab.loc[0, ["circling", "gliding"]].isna().all()  # no turn rates
    assert lab.loc[1, ["perched", "flapping", "circling", "gliding", "climbing_soar"]].isna().all()  # no GPS, no spectrum
    assert lab.loc[2, "flapping"] == 1.0 and np.isnan(lab.loc[2, "night"]) and np.isnan(lab.loc[2, "perched"])
