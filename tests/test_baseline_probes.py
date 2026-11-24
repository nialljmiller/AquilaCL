import json

import numpy as np
import pandas as pd

from aquilacl import baseline, probes

# three behaviours: perched, flapping flight, circling soar
STATES = {
    "perched": dict(vedba=0.03, flap_band_frac=0.1, pitch_deg=10, roll_deg=5, speed_mean=0.1, climb_mean=0.0, turn_abs_mean=0.5),
    "flapping": dict(vedba=0.6, flap_band_frac=0.7, pitch_deg=0, roll_deg=5, speed_mean=12, climb_mean=0.5, turn_abs_mean=2),
    "circling": dict(vedba=0.08, flap_band_frac=0.2, pitch_deg=-5, roll_deg=25, speed_mean=10, climb_mean=2.0, turn_abs_mean=12),
}


def _synthetic_windows(tmp_path, n_birds=8, n_seg=40, seg_len=10, seed=0):
    rng = np.random.default_rng(seed)
    names = list(STATES)
    for b in range(n_birds):
        rows, meta = [], []
        for s in range(n_seg):
            st = names[rng.integers(3)]
            for k in range(seg_len):
                f = {c: v * rng.lognormal(0, 0.15) if c in ("vedba", "speed_mean", "turn_abs_mean") else v + rng.normal(0, 0.05 * (abs(v) + 1))
                     for c, v in STATES[st].items()}
                f.update(n_speed=10, n_turn=9, dom_freq_hz=2.5, label_flapping=st == "flapping", label_perched=st == "perched")
                rows.append(f)
                meta.append(dict(bird=f"B{b}", t_start=pd.Timestamp("2025-05-01") + pd.Timedelta(minutes=10 * s, seconds=10 * k),
                                 segment=s, k_in_segment=k, acc_samples=200, sun_elev_deg=30.0, truth=st))
        d = tmp_path / "win" / f"B{b}"
        d.mkdir(parents=True)
        pd.DataFrame(meta).to_parquet(d / "meta.parquet")
        pd.DataFrame(rows).to_parquet(d / "features.parquet")
    return tmp_path / "win"


def test_baseline_recovers_states(tmp_path):
    win = _synthetic_windows(tmp_path)
    s = baseline.run(win, tmp_path / "out", k_range=range(2, 6), per_bird=400, n_boot=2)
    assert s["k_best"] == 3
    st = pd.read_parquet(tmp_path / "out" / "states.parquet")
    truth = pd.concat(pd.read_parquet(p / "meta.parquet") for p in sorted(win.iterdir()))["truth"].to_numpy()
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(truth, st["state_hmm"]) > 0.95
    assert min(s["bootstrap_ari"]) > 0.9
    assert json.loads((tmp_path / "out" / "baseline_summary.json").read_text())["k_best"] == 3


def test_probe_held_out_birds():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(10), 100)
    y = rng.random(1000) < 0.5
    X = np.c_[y + rng.normal(0, 0.3, 1000), rng.normal(size=1000)]
    r = probes.probe(X, y, groups, "class")
    assert r["balanced_accuracy"] > 0.9 and r["birds"] == 10
    r0 = probes.probe(rng.normal(size=(1000, 2)), y, groups, "class")
    assert abs(r0["balanced_accuracy"] - 0.5) < 0.1  # nothing to find -> chance
    yr = X[:, 0] * 2 + rng.normal(0, 0.1, 1000)
    assert probes.probe(X, yr, groups, "reg")["r2"] > 0.9
