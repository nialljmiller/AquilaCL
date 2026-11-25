import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aquilacl import dataset, train
from aquilacl.model import Encoder
from aquilacl.windows import GPS_CHANNELS

_spec = importlib.util.spec_from_file_location("diagnose_r4", Path(__file__).parents[1] / "scripts" / "diagnose_r4.py")
dg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dg)


def _axial_diff(a, b):
    d = np.abs((np.asarray(a) - b) % 180)
    return np.minimum(d, 180 - d)


@pytest.mark.parametrize("s0", [1 / 512, 0.0021])
def test_requantize_lands_on_shifted_grid(s0):
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (50, 200, 3)).astype(np.float32)
    phase = rng.random((50, 1, 3)).astype(np.float32) * s0
    q = dg.requantize(a, s0, phase)
    k = (q.astype(np.float64) + phase) / s0
    np.testing.assert_allclose(k, np.round(k), atol=1e-3)
    assert np.abs(q - a).max() <= s0 / 2 + 1e-6
    # a pure grid shift moves values, a zero shift on an existing grid does not
    on_grid = np.round(a / s0) * s0
    np.testing.assert_allclose(dg.requantize(on_grid, s0, 0.0), on_grid, atol=1e-6)


@pytest.mark.parametrize("theta", [0.0, 2.0, 45.0, 90.0, 135.0, 178.0, 179.5])
def test_axial_angle_recovers_direction_mod_180(theta):
    rng = np.random.default_rng(1)
    t = np.arange(200) / 20
    u = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    amp = 0.5 * np.sin(2 * np.pi * 2.5 * t)[None, :, None]
    xy = amp * u + rng.normal(0, 0.02, (20, 200, 2)) + 0.3  # offset is removed as the window mean
    m = np.ones((20, 200), bool)
    m[:, 150:] = False
    xy[:, 150:] = 0.0  # absent slots must not count
    th, ratio = dg.axial_angle(xy.astype(np.float32), m)
    assert _axial_diff(th, theta).max() < 1.0
    assert (ratio > 50).all()
    mu, sd, R = dg.axial_mean(th)
    assert _axial_diff(mu, theta) < 0.5 and sd < 1.0 and R > 0.99
    mu, _, _ = dg.axial_mean(np.array([178.0, 2.0]))  # axial mean across the 0/180 wrap
    assert _axial_diff(mu, 0.0) < 1e-6


def _windows(tmp_path, n_birds=6, n_win=240, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(200) / 20
    for b in range(n_birds):
        acc, am, gps, gm, meta, feat = [], [], [], [], [], []
        yaw = np.radians(rng.uniform(0, 180))
        tilt = np.radians(rng.uniform(30, 50))  # mounting tilt, so the rotation visibly changes the model input
        Rb = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        for w in range(n_win):
            beh = ["still", "flap", "night", "soar"][(w // 10) % 4]
            k = int((w // 5) % 2)
            g = np.tile([0.1, 0.0, 1.0], (200, 1)) + rng.normal(0, 0.01, (200, 3))
            if beh == "flap":
                s = 0.5 * np.sin(2 * np.pi * 2.5 * t + rng.uniform(0, 6))
                g[:, 0] += s * np.cos(yaw)
                g[:, 1] += s * np.sin(yaw)
            g = g @ Rb.T
            step = 1 / 1024 if k == 1 else 1 / 512
            g = np.round(g / step) * step
            mk = np.ones(200, bool)
            if k == 1 and w % 3 == 0:
                mk[100] = False  # snap gap
            gg = np.zeros((10, len(GPS_CHANNELS)), np.float32)
            gg[:, 0] = 12.0 if beh in ("flap", "soar") else 0.1
            acc.append((g * mk[:, None]).astype(np.float16))
            am.append(mk)
            gps.append(gg)
            gm.append(np.full((10, len(GPS_CHANNELS)), beh != "night"))
            nan = w % 17 == 0
            meta.append(dict(bird=f"B{b}", tag="1", t_start=pd.Timestamp("2025-06-01") + pd.Timedelta(seconds=10 * w),
                             segment=w // 40, sensitivity_k=k, decoding="21", sun_elev_deg=-20.0 if beh == "night" else 30.0,
                             acc_samples=int(mk.sum()), gps_fixes=0 if beh == "night" else 10))
            lab = lambda c: np.nan if nan else float(c)
            feat.append(dict(vedba=0.4 if beh == "flap" else 0.005, flap_band_frac=0.8 if beh == "flap" else 0.1,
                             pitch_deg=rng.normal(5, 1), roll_deg=rng.normal(0, 1), speed_mean=gg[0, 0], climb_mean=0.0,
                             turn_abs_mean=1.0, label_flapping=lab(beh == "flap"), label_perched=lab(beh == "still"),
                             label_circling=lab(beh == "soar"), label_gliding=lab(False), label_climbing_soar=lab(False),
                             label_night=lab(beh == "night")))
        d = tmp_path / "win" / f"B{b}"
        d.mkdir(parents=True)
        np.save(d / "acc.npy", np.stack(acc)); np.save(d / "acc_mask.npy", np.stack(am))
        np.save(d / "gps.npy", np.stack(gps)); np.save(d / "gps_mask.npy", np.stack(gm))
        pd.DataFrame(meta).to_parquet(d / "meta.parquet"); pd.DataFrame(feat).to_parquet(d / "features.parquet")
    return tmp_path / "win"


def test_diagnose_end_to_end_reproduces_stored_embeddings(tmp_path):
    """Random-init model, embeddings written by train.embed on CPU: the rebuilt pipeline must match them."""
    win = _windows(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    step_g, clip_g, mask = 0.002116, 1.774, ("eobs:horizontal-accuracy-estimate",)
    data = dataset.load(win, step_g=step_g, clip_g=clip_g, mask_gps_channels=mask)
    (run / "orientation.json").write_text(json.dumps(data.orientation.to_dict("records")))
    (run / "holdout.json").write_text(json.dumps(["B3", "B4", "B5"]))
    (run / "config.json").write_text(json.dumps({"step_g": step_g, "clip_g": clip_g, "mask_gps_channels": list(mask)}))
    torch.manual_seed(0)
    model = Encoder(len(GPS_CHANNELS))
    torch.save({"step": 1, "model": model.state_dict()}, run / "ckpt_0000001.pt")
    train.embed(model, data, "cpu", run / "embeddings")
    out = tmp_path / "diag" / "r.json"
    argv = ["--run", str(run), "--windows", str(win), "--out", str(out), "--n", "60", "--n-check", "60", "--per-class", "60",
            "--per-bird", "60", "--per-bird-subset", "100", "--n-geom", "60", "--k", "3"]
    dg.main(argv)
    r = json.loads(out.read_text())
    p0 = r["part0_reproduction"]
    assert p0["pass"]
    # stored h made on CPU here: the rebuilt pipeline matches to float16 storage, a wrong rotation does not
    # (the "discriminates" flag uses the bf16 tolerance meant for GPU-made embeddings, too wide for a random model)
    got = p0["stored_vs_cpu_fp32"]["max_abs"]
    assert got <= p0["fp16_storage_resolution"] * 2
    assert p0["wrong_pipeline_no_rotation_vs_stored"]["max_abs"] > 10 * got
    s = r["part1_sensitivity"]["ge190"]
    assert s["birds"] == 6 and 0 <= s["k1_to_k0_nominal_step"]["frac_predicted_k0"] <= 1
    assert s["k0_clip_2g_frac_raw_samples_beyond"] == 0
    assert set(r["part2_identity"]["perturbed"]) >= {"gps_removed", "acc_removed", "yaw_uniform_180deg"}
    assert len(r["part2_identity"]["yaw_flapping"]) == 3
    assert not out.with_name(out.name + ".partial").exists()
    with pytest.raises(SystemExit):
        dg.main(argv)  # refuses to overwrite
