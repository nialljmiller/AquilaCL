import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aquilacl import gps, profile

GAIN, OFFSET = 1000.0, np.array([2040.0, 2055.0, 2035.0])


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


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
    assert c["still_bursts"] == 60  # all night bursts, no flight bursts
    assert abs(np.mean([c["gain_x"], c["gain_y"], c["gain_z"]]) / GAIN - 1) < 0.06
    out = pd.read_parquet(tmp_path / "profile" / "acc" / "B1.parquet")
    day = out[~out["still"]]
    assert np.allclose(day["dom_freq_hz"], 2.5, atol=0.1)
    assert (out.loc[out["still"], "sun_elev_deg"] < -6).all()
    assert {r["stream"] for r in sched} == {"gps", "acceleration"}
