import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from aquilacl import dataset, evaluate, train
from aquilacl.model import Encoder, nt_xent
from aquilacl.windows import GPS_CHANNELS


def _naive_nt_xent(z0, z1, tau):
    z = F.normalize(torch.cat([z0, z1]), dim=1)
    B = len(z0)
    loss = 0.0
    for i in range(2 * B):
        j = (i + B) % (2 * B)
        sims = torch.stack([z[i] @ z[k] / tau for k in range(2 * B) if k != i])
        pos = z[i] @ z[j] / tau
        loss += -(pos - torch.logsumexp(sims, 0))
    return loss / (2 * B)


def test_nt_xent_matches_naive_and_mask_removes_negatives():
    torch.manual_seed(0)
    z0, z1 = torch.randn(6, 4), torch.randn(6, 4)
    assert torch.allclose(nt_xent(z0, z1, 0.5), _naive_nt_xent(z0, z1, 0.5), atol=1e-5)
    m = torch.zeros(6, 6, dtype=torch.bool)
    m[0, 1] = m[1, 0] = True
    assert nt_xent(z0, z1, 0.5, m) < nt_xent(z0, z1, 0.5) + 1e-6  # fewer negatives cannot raise this loss much


def test_rotation_to_z():
    v = np.array([0.3, -0.5, 0.8])
    R = dataset.rotation_to_z(v)
    np.testing.assert_allclose(R @ (v / np.linalg.norm(v)), [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


def _synthetic(tmp_path, n_birds=10, n_seg=30, seg_len=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(200) / 20
    for b in range(n_birds):
        year = 2024 if b < n_birds // 2 else 2025
        tilt = np.radians(rng.uniform(-20, 20))  # mounting angle differs per bird
        Rb = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        acc, am, gps, gm, meta, feat = [], [], [], [], [], []
        for s in range(n_seg):
            beh = ["still", "flap", "night"][s % 3]
            for k in range(seg_len):
                g = np.tile([0.0, 0.0, 1.0], (200, 1)) + rng.normal(0, 0.005, (200, 3))
                if beh == "flap":
                    g[:, 2] += 0.6 * np.sin(2 * np.pi * 2.6 * t + rng.uniform(0, 6))
                acc.append((g @ Rb.T).astype(np.float16))
                am.append(np.ones(200, bool))
                gg = np.zeros((10, len(GPS_CHANNELS)), np.float32)
                gg[:, 0] = 12.0 if beh == "flap" else 0.1
                gps.append(gg)
                gm.append(np.full((10, len(GPS_CHANNELS)), beh != "night"))
                ts = pd.Timestamp(f"{year}-07-01") + pd.Timedelta(hours=s, seconds=10 * k)
                meta.append(dict(bird=f"B{b}", tag="1", t_start=ts, segment=s, k_in_segment=k, segment_windows=seg_len,
                                 sensitivity_k=0, decoding="21", sun_elev_deg=-20.0 if beh == "night" else 30.0,
                                 acc_samples=200, gps_fixes=0 if beh == "night" else 10))
                feat.append(dict(vedba=0.4 if beh == "flap" else 0.005, speed_mean=gg[0, 0], climb_mean=0.0, turn_abs_mean=1.0,
                                 flap_band_frac=0.8 if beh == "flap" else 0.1, label_flapping=beh == "flap",
                                 label_perched=beh == "still", label_circling=False, label_gliding=False,
                                 label_climbing_soar=False, label_night=beh == "night"))
        d = tmp_path / "win" / f"B{b}"
        d.mkdir(parents=True)
        np.save(d / "acc.npy", np.stack(acc)); np.save(d / "acc_mask.npy", np.stack(am))
        np.save(d / "gps.npy", np.stack(gps)); np.save(d / "gps_mask.npy", np.stack(gm))
        pd.DataFrame(meta).to_parquet(d / "meta.parquet"); pd.DataFrame(feat).to_parquet(d / "features.parquet")
    (tmp_path / "events" / "magnetometer").mkdir(parents=True)
    return tmp_path / "win"


def test_orientation_is_corrected(tmp_path):
    win = _synthetic(tmp_path, n_birds=4)
    data = dataset.load(win)
    night = data.meta["sun_elev_deg"].to_numpy() < -6
    g = data.acc[night].astype(float).mean(axis=1)
    np.testing.assert_allclose(g[:, 2], 1.0, atol=0.02)  # gravity on +z for every bird after rotation
    assert data.orientation["tilt_corrected_deg"].notna().all()


def test_train_embed_evaluate_smoke(tmp_path):
    win = _synthetic(tmp_path)
    run = tmp_path / "run"
    train.main(["--windows", str(win), "--run", str(run), "--steps", "300", "--batch", "64", "--birds-per-batch", "4",
                "--warmup", "20", "--save-every", "150", "--holdout-per-cohort", "2", "--device", "cpu",
                "--p-burst-crop", "0.5", "--step-g", "0.001953125", "--clip-g", "2.0"])
    loss = pd.read_csv(run / "loss.csv")
    assert loss["loss"].iloc[-1] < loss["loss"].iloc[0]
    assert len(list((run / "embeddings").glob("*.npz"))) == 10
    # resume does not truncate the log and does not retrain past --steps
    train.main(["--windows", str(win), "--run", str(run), "--steps", "300", "--batch", "64", "--birds-per-batch", "4",
                "--holdout-per-cohort", "2", "--device", "cpu"])
    assert len(pd.read_csv(run / "loss.csv")) == len(loss)
    evaluate.main(["--run", str(run), "--windows", str(win), "--events", str(tmp_path / "events")])
    ev = json.loads((run / "evaluation.json").read_text())
    assert ev["window_level_holdout"]["label_flapping"]["balanced_accuracy"] > 0.9
    assert len(ev["holdout_birds"]) == 4
    inv = ev["invariance"]
    assert 0 <= inv["burst_same_state"] <= 1 and inv["k1_regridded_to_512_same_state"] is None


def test_burst_crop_moves_span_to_start_and_common_format():
    torch.manual_seed(0)
    acc = torch.arange(200, dtype=torch.float32)[None, :, None].repeat(4, 1, 3)
    mask = torch.ones(4, 200, dtype=torch.bool)
    mask[3, 40:] = False  # already a short bout: left alone
    a, m = train.burst_crop(acc, mask, (40,), p=1.0)
    assert m[:3].sum(1).tolist() == [40, 40, 40] and m[:3, :40].all()
    assert torch.equal(m[3], mask[3]) and torch.equal(a[3], acc[3])
    d = a[:3, 1:40, 0] - a[:3, :39, 0]
    assert torch.all(d == 1)  # contiguous samples, in order
    # common format: 1024 counts/g data regridded to 1/512 g and clipped to +-2 g
    x = np.array([[[0.0015, 2.5, -3.0]]], np.float32)
    y = dataset.common_format(x, np.ones((1, 1), bool), 1 / 512, 2.0).astype(np.float32)
    np.testing.assert_allclose(y[0, 0], [0.001953125, 2.0, -2.0], atol=1e-4)


def test_roost_gravity_falls_back_to_quiet_daytime_windows():
    acc = np.tile(np.array([0.0, 0.6, 0.8], np.float16), (30, 200, 1))
    mask = np.ones((30, 200), bool)
    sun = np.full(30, 20.0)  # no night windows at all
    vedba = np.full(30, 0.005)
    v, n = dataset.roost_gravity(acc, mask, sun, vedba, speed=np.full(30, np.nan))
    np.testing.assert_allclose(v, [0.0, 0.6, 0.8], atol=1e-3)
    assert n == 30
    v, _ = dataset.roost_gravity(acc, mask, sun, vedba, speed=np.full(30, 3.0))  # moving: not usable
    assert v is None
