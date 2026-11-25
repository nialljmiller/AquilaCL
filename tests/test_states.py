import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from aquilacl import dataset, states

N_BEHAV, N_BIRDS = 4, 10
FEATS = ["vedba", "vedba_sd", "odba", "static_norm", "pitch_deg", "roll_deg", "flap_band_frac", "dom_freq_hz",
         "speed_mean", "climb_mean", "turn_abs_mean", "n_speed", "n_turn"]
LABELS = ["label_flapping", "label_perched", "label_night"]


def _write_bird(d, meta, feats, acc, h, run):
    d.mkdir(parents=True)
    meta.to_parquet(d / "meta.parquet")
    feats.to_parquet(d / "features.parquet")
    np.save(d / "acc.npy", acc)
    np.save(d / "acc_mask.npy", np.ones(acc.shape[:2], bool))
    np.save(d / "gps.npy", np.zeros((len(meta), 10, 6), np.float16))
    np.save(d / "gps_mask.npy", np.zeros((len(meta), 10, 6), bool))
    if len(meta):
        np.savez(run / "embeddings" / f"{d.name}.npz", h=h)


def _synthetic(tmp_path, seed=0):
    """4 behaviours in a 128-d embedding, a random offset per bird, segments of 1-8 windows; one empty bird."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1, (N_BEHAV, 128))
    win, run = tmp_path / "win", tmp_path / "run"
    (run / "embeddings").mkdir(parents=True)
    truth, orient = [], []
    for b in range(N_BIRDS):
        offset = rng.normal(0, 0.15, 128)
        year = 2024 if b < N_BIRDS // 2 else 2025
        rows = []
        for s in range(60):
            beh = int(rng.integers(N_BEHAV))
            for k in range(int(rng.integers(1, 9))):
                rows.append(dict(segment=s, k_in_segment=k, beh=beh,
                                 t_start=pd.Timestamp(f"{year}-0{1 + s % 6}-01", tz="UTC") + pd.Timedelta(seconds=10 * (60 * s + k))))
        m = pd.DataFrame(rows)
        n = len(m)
        h = (centres[m["beh"]] + offset + rng.normal(0, 0.1, (n, 128))).astype(np.float16)
        meta = pd.DataFrame(dict(bird=f"B{b}", tag=f"T{b}", t_start=m["t_start"], segment=m["segment"],
                                 k_in_segment=m["k_in_segment"], segment_windows=m.groupby("segment")["k_in_segment"].transform("size"),
                                 sensitivity_k=b % 2, decoding="v1", sun_elev_deg=rng.uniform(-30, 60, n),
                                 acc_samples=np.where(rng.random(n) < 0.8, 200, 40), gps_fixes=rng.integers(0, 10, n),
                                 local_solar_hour=rng.uniform(0, 24, n)))
        feats = pd.DataFrame({c: rng.random(n) + m["beh"] for c in FEATS})
        for c in LABELS:
            feats[c] = np.where(rng.random(n) < 0.2, np.nan, (m["beh"] == 1).astype(float))
        v = np.array([0.3, -0.2 + 0.05 * b, -0.9])  # tilted, upside-down tag: gravity direction in the sensor frame
        v /= np.linalg.norm(v)
        acc = np.broadcast_to(v, (n, 200, 3)).astype(np.float16)
        _write_bird(win / f"B{b}", meta, feats, acc, h, run)
        truth.append(m["beh"].to_numpy())
        orient.append({"bird": f"B{b}", "R": dataset.rotation_to_z(v).ravel().tolist()})
    _write_bird(win / "Bempty", pd.DataFrame(columns=meta.columns), pd.DataFrame(columns=feats.columns),
                np.zeros((0, 200, 3), np.float16), None, run)
    (run / "orientation.json").write_text(json.dumps(orient))
    (run / "holdout.json").write_text(json.dumps(["B0", "B9"]))
    return run, win, np.concatenate(truth)


def test_pipeline_recovers_behaviours(tmp_path):
    run, win, truth = _synthetic(tmp_path)
    out = tmp_path / "out"
    s = states.run(run, win, out, k_over=8, k_min=2, k_max=8, per_bird=300, eval_per_bird=150, n_boot=20)
    assert s["chosen_k"] == N_BEHAV and s["rule_met"]
    st = pd.read_parquet(out / "states.parquet")
    assert list(st.columns) == ["bird", "row", "t_start", "segment", "k_in_segment", "pitch_rot_deg", "roll_rot_deg", "state", "state_hmm", "p_state",
                                "pca_1", "pca_2", "pca_3"]
    assert "Bempty" not in set(st["bird"]) and len(st) == len(truth)
    for col in ("state", "state_hmm"):
        assert adjusted_rand_score(truth, st[col]) > 0.9
    prof = pd.read_csv(out / "state_profiles.csv", index_col="state")
    assert len(prof) == N_BEHAV
    for c in ["windows", "birds", "birds_2024", "birds_2025", "vedba_median", "pitch_rot_deg_median", "roll_rot_deg_q75",
              "month_median", "gps_share", "frac_flapping", "share_sens_0", "share_full", "share_2024",
              "dwell_mean_windows", "max_bird_share", "max_tag_share", "sens_ratio_max", "nonfull_ratio",
              "flag_night_flight", "jaccard", "presence_birds"]:
        assert c in prof, c
    assert prof["vedba_median"].is_monotonic_increasing  # states numbered by activity
    assert np.allclose(prof[["pitch_rot_deg_median", "roll_rot_deg_median"]], 0, atol=1)  # rotated posture
    assert pd.read_csv(out / "transitions.csv", index_col=0).shape == (N_BEHAV, N_BEHAV)
    sel = json.loads((out / "state_selection.json").read_text())
    assert set(sel["per_k"]) == {str(k) for k in range(2, 9)} and sel["per_k"]["4"]["heldout_ari"] > 0.9
    h = np.load(run / "embeddings" / "B3.npz")["h"]
    lab, _, _ = states.assign(out / "model.npz", h)
    assert (lab == st.loc[st["bird"] == "B3", "state"].to_numpy()).all()
    with pytest.raises(SystemExit):
        states.run(run, win, out)


@pytest.mark.parametrize("emit_state, k, expect", [
    # long chain with an isolated flip -> smoothed
    ([0, 0, 0, 1, 0, 0], [0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]),
    # the flip sits alone after a k_in_segment gap -> single-window chain keeps its label
    ([0, 0, 0, 1, 0, 0], [0, 1, 2, 4, 6, 7], [0, 0, 0, 1, 0, 0]),
    # a sustained change is kept
    ([0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]),
])
def test_hmm_smoothing(emit_state, k, expect):
    e = np.array(emit_state)
    log_emit = np.where(np.eye(2)[e] > 0, 0.0, -2.0)  # mild evidence for the emitted state
    meta = pd.DataFrame({"bird": "B", "segment": 0, "k_in_segment": k})
    order, start = states.chains(meta)
    T = np.array([[0.98, 0.02], [0.02, 0.98]])
    out = states.smooth(log_emit[order], e[order], start, T, np.array([0.5, 0.5]))
    assert out.tolist() == expect


def test_ward_merge_gives_exact_k():
    means = np.random.default_rng(0).normal(size=(12, 5))
    for K, lab in states.ward_merge(means, range(1, 13)).items():
        assert len(np.unique(lab)) == K
