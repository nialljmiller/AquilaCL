import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from aquilacl import dataset, states, substates

N_BIRDS = 10
FEATS = ["vedba", "vedba_sd", "odba", "static_norm", "pitch_deg", "roll_deg", "flap_band_frac", "dom_freq_hz",
         "speed_mean", "climb_mean", "turn_abs_mean", "n_speed", "n_turn"]


def _synthetic(tmp_path, seed=0):
    """Behaviours 0 and 1 are a planted split of one group (close centres); 2 and 3 are single isotropic blobs.

    With K=3 forced in stage 5a, 0 and 1 form one coarse state and 2, 3 one state each."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0, 1, (3, 128))
    d = rng.normal(0, 1, 128)
    d *= 1.5 / np.linalg.norm(d)
    centres = np.stack([c[0] + d, c[0] - d, c[1], c[2]])
    win, run = tmp_path / "win", tmp_path / "run"
    (run / "embeddings").mkdir(parents=True)
    truth, orient = [], []
    for b in range(N_BIRDS):
        offset = rng.normal(0, 0.02, 128)
        year = 2024 if b < N_BIRDS // 2 else 2025
        rows = []
        for s in range(80):
            beh = int(rng.integers(4))
            for k in range(int(rng.integers(1, 9))):
                rows.append(dict(segment=s, k_in_segment=k, beh=beh,
                                 t_start=pd.Timestamp(f"{year}-0{1 + s % 6}-01", tz="UTC") + pd.Timedelta(seconds=10 * (60 * s + k))))
        m = pd.DataFrame(rows)
        n = len(m)
        h = (centres[m["beh"]] + offset + rng.normal(0, 0.1, (n, 128))).astype(np.float16)
        meta = pd.DataFrame(dict(bird=f"B{b}", tag=f"T{b}", t_start=m["t_start"], segment=m["segment"],
                                 k_in_segment=m["k_in_segment"], sensitivity_k=b % 2, sun_elev_deg=rng.uniform(-30, 60, n),
                                 acc_samples=np.where(rng.random(n) < 0.8, 200, 40), gps_fixes=rng.integers(0, 10, n),
                                 local_solar_hour=rng.uniform(0, 24, n)))
        feats = pd.DataFrame({f: rng.random(n) + m["beh"] for f in FEATS})
        feats["label_flapping"] = np.where(rng.random(n) < 0.2, np.nan, (m["beh"] == 1).astype(float))
        v = np.array([0.3, -0.2 + 0.05 * b, -0.9])
        v /= np.linalg.norm(v)
        d_ = win / f"B{b}"
        d_.mkdir(parents=True)
        meta.to_parquet(d_ / "meta.parquet")
        feats.to_parquet(d_ / "features.parquet")
        np.save(d_ / "acc.npy", np.broadcast_to(v, (n, 200, 3)).astype(np.float16))
        np.save(d_ / "acc_mask.npy", np.ones((n, 200), bool))
        np.savez(run / "embeddings" / f"B{b}.npz", h=h)
        truth.append(m["beh"].to_numpy())
        orient.append({"bird": f"B{b}", "R": dataset.rotation_to_z(v).ravel().tolist()})
    (run / "orientation.json").write_text(json.dumps(orient))
    (run / "holdout.json").write_text(json.dumps(["B0", "B9"]))
    return run, win, np.concatenate(truth)


def test_rule_splits_only_the_planted_state(tmp_path):
    run, win, truth = _synthetic(tmp_path)
    sdir, out = tmp_path / "states", tmp_path / "sub"
    states.run(run, win, sdir, k=3, k_over=8, k_min=2, k_max=4, per_bird=300, eval_per_bird=150, n_boot=4)
    coarse = pd.read_parquet(sdir / "states.parquet")["state"].to_numpy()
    merged = np.where(truth == 1, 0, truth)
    assert adjusted_rand_score(merged, coarse) > 0.95  # stage 5a merged the planted pair into one state
    s_a = int(pd.Series(coarse[truth <= 1]).mode()[0])

    s = substates.run(sdir, run, win, out, max_sub=3, k_over=6, per_bird=300, eval_per_bird=150, n_boot=10,
                      coarse_per_bird=300, coarse_eval_per_bird=150)
    assert s["chosen"] == {str(i): (2 if i == s_a else 1) for i in range(3)}
    rec = s["states"][str(s_a)]["per_k_sub"]
    assert min(rec["2"]["jaccard"]) >= 0.75 and rec["2"]["presence"]["ok"] and rec["2"]["heldout_ari"] > 0.9
    for i in set(range(3)) - {s_a}:
        assert not any(r["ok"] for k, r in s["states"][str(i)]["per_k_sub"].items() if k != "1")
    assert rec["1"]["heldout_ari"] is None and "Fixed before results" in s["rule"]

    sub = pd.read_parquet(out / "substates.parquet")
    assert list(sub.columns) == ["bird", "row", "state", "substate", "label"] and len(sub) == len(truth)
    a = sub["state"].to_numpy() == s_a
    assert adjusted_rand_score(truth[a], sub["substate"].to_numpy()[a]) > 0.9
    assert set(sub["label"]) == {f"{s_a}.0", f"{s_a}.1"} | {f"{i}.0" for i in range(3) if i != s_a}

    prof = pd.read_csv(out / "substate_profiles.csv")
    assert len(prof) == 4 and list(prof.columns[:5]) == ["label", "state", "substate", "windows", "share_of_state"]
    ref_cols = pd.read_csv(sdir / "state_profiles.csv", index_col="state").columns.drop(["jaccard", "presence_birds"])
    for col in list(ref_cols) + ["dwell_median_s", "month_share_01", "jaccard", "presence_birds"]:
        assert col in prof, col
    assert np.allclose(prof["dwell_median_s"], prof["dwell_median_windows"] * 10)

    conf = pd.read_csv(out / "coarse_confusion.csv", index_col="state")
    assert conf.shape == (3, 5) and np.allclose(conf[["0", "1", "2"]].sum(axis=1), 1)
    assert (np.diag(conf[["0", "1", "2"]]) > 0.95).all()
    sel = json.loads((out / "substate_selection.json").read_text())
    assert sel["parameters"]["jaccard_min"] == 0.75 and sel["coarse"]["k"] == 3
    assert "State" in (out / "substates.md").read_text()


def test_confusion_maps_many_to_one():
    ref = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    boot = np.array([0, 0, 0, 0, 1, 1, 2, 2])  # 0 and 1 merged, 2 split
    M = substates.confusion(ref, boot, 3)
    assert M.tolist() == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]


def test_refuses_non_empty_out(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "x").write_text("")
    with pytest.raises(SystemExit, match="not empty"):
        substates.run(tmp_path / "missing", tmp_path / "missing", tmp_path / "missing", out)
