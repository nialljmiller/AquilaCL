import pytest
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


def _synthetic(tmp_path, n_birds=10, n_seg=30, seg_len=6, seed=0, flap_yaw=None):
    """``flap_yaw`` (per-bird degrees, or None): flapping also oscillates horizontally along that axis (in the
    gravity-aligned frame, before the mounting tilt), so the mounting yaw can be estimated."""
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
                    ph = rng.uniform(0, 6)
                    g[:, 2] += 0.6 * np.sin(2 * np.pi * 2.6 * t + ph)
                    if flap_yaw is not None:
                        ang = np.radians(flap_yaw[b])
                        surge = 0.3 * np.sin(2 * np.pi * 2.6 * t + ph + 1.0)
                        g[:, 0] += surge * np.cos(ang)
                        g[:, 1] += surge * np.sin(ang)
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
    # resubmitting with different settings is refused
    with pytest.raises(SystemExit):
        train.main(["--windows", str(win), "--run", str(run), "--steps", "300", "--tau", "0.1", "--device", "cpu",
                    "--step-g", "0.001953125", "--clip-g", "2.0"])
    loss = pd.read_csv(run / "loss.csv")
    assert loss["loss"].iloc[-1] < loss["loss"].iloc[0]
    assert len(list((run / "embeddings").glob("*.npz"))) == 10
    # resume does not truncate the log and does not retrain past --steps
    train.main(["--windows", str(win), "--run", str(run), "--steps", "300", "--batch", "64", "--birds-per-batch", "4",
                "--holdout-per-cohort", "2", "--device", "cpu", "--p-burst-crop", "0.5", "--step-g", "0.001953125",
                "--clip-g", "2.0", "--warmup", "20"])
    assert len(pd.read_csv(run / "loss.csv")) == len(loss)
    evaluate.main(["--run", str(run), "--windows", str(win), "--events", str(tmp_path / "events")])
    ev = json.loads((run / "evaluation.json").read_text())
    assert ev["window_level_holdout"]["label_flapping"]["balanced_accuracy"] > 0.9
    assert len(ev["holdout_birds"]) == 4
    inv = ev["invariance"]
    assert 0 <= inv["burst_same_state"] <= 1 and inv["k1_regridded_to_512_same_state"] is None or isinstance(inv["k1_regridded_to_512_same_state"], str)


def test_burst_crop_moves_span_to_start_and_common_format():
    torch.manual_seed(0)
    acc = torch.arange(200, dtype=torch.float32)[None, :, None].repeat(4, 1, 3)
    mask = torch.ones(4, 200, dtype=torch.bool)
    mask[3, 40:] = False  # already a short bout: left alone
    a, m = train.burst_crop(acc, mask, torch.tensor([40]), p=1.0)
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


# ---------------------------------------------------------------- identity-reduction options
from types import SimpleNamespace

from aquilacl.model import BirdHead, grad_reverse


def _reference_augment(acc, amask, gps, gmask, cfg, short_lengths):
    """Copy of train.augment before --axis-scale/--offset-g/--yaw-deg existed (r4)."""
    B = acc.shape[0]
    dev = acc.device
    amask = train.contiguous_crop(amask, cfg.min_acc_samples, cfg.p_crop)
    acc, amask = train.burst_crop(acc, amask, short_lengths, cfg.p_burst_crop)
    R = train.random_rotation(B, cfg.rot_deg, dev)
    scale = torch.empty(B, 1, 1, device=dev).uniform_(1 - cfg.scale, 1 + cfg.scale)
    acc = (acc.float() @ R.transpose(1, 2)) * scale + cfg.noise_g * torch.randn_like(acc, dtype=torch.float32)
    acc = acc * amask[..., None]
    sec = gmask.any(-1)
    sec = sec & (torch.rand_like(sec, dtype=torch.float32) > cfg.p_gps_drop)
    sec = train.contiguous_crop(sec, 3, cfg.p_crop)
    gmask = gmask & sec[..., None]
    gps = (gps.float() + cfg.gps_noise * torch.randn_like(gps, dtype=torch.float32)) * gmask
    has_a, has_g = amask.any(1), gmask.flatten(1).any(1)
    r = torch.rand(B, device=dev)
    drop_g = has_a & has_g & (r < cfg.p_drop_gps)
    drop_a = has_a & has_g & (r >= cfg.p_drop_gps) & (r < cfg.p_drop_gps + cfg.p_drop_acc)
    amask = amask & ~drop_a[:, None]
    acc = acc * amask[..., None]
    gmask = gmask & ~drop_g[:, None, None]
    gps = gps * gmask
    return acc, amask, gps, gmask


def _aug_inputs(B=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    acc = torch.randn(B, 200, 3, generator=g).half()
    am = torch.ones(B, 200, dtype=torch.bool)
    am[: B // 2, 60:] = False
    gps = torch.randn(B, 10, len(GPS_CHANNELS), generator=g).half()
    gm = torch.rand(B, 10, len(GPS_CHANNELS), generator=g) > 0.3
    return acc, am, gps, gm, torch.tensor([40, 64, 120])


def _cfg(**kw):
    d = vars(train.get_args(["--windows", "w", "--run", "r"]))
    d.update(kw)
    return SimpleNamespace(**d)


def test_augment_unchanged_when_new_options_off():
    inp = _aug_inputs()
    for cfg in (_cfg(), SimpleNamespace(**{k: v for k, v in vars(_cfg()).items()
                                          if k not in ("axis_scale", "offset_g", "yaw_deg", "yaw_flip_p")})):
        torch.manual_seed(123)
        ref = _reference_augment(*inp[:4], cfg, inp[4])
        ref_next = torch.rand(1)
        torch.manual_seed(123)
        new = train.augment(*inp[:4], cfg, inp[4])
        for a, b in zip(ref, new):
            assert torch.equal(a, b)
        assert torch.equal(torch.rand(1), ref_next)  # no extra random numbers drawn


def test_yaw_axis_scale_offset():
    torch.manual_seed(0)
    Rz = train.yaw_rotation(500, 30.0, "cpu")
    x = torch.randn(500, 7, 3)
    y = x @ Rz.transpose(1, 2)
    assert torch.equal(y[..., 2], x[..., 2])  # z (gravity) component unchanged
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    ang = torch.rad2deg(torch.atan2(Rz[:, 1, 0], Rz[:, 0, 0]))
    assert ang.abs().max() <= 30.0 + 1e-4 and ang.abs().max() > 25.0
    # per-axis gain / offset bounds: constant input, no other augmentation
    B = 400
    acc = torch.ones(B, 200, 3).half()
    am = torch.ones(B, 200, dtype=torch.bool)
    gps = torch.zeros(B, 10, len(GPS_CHANNELS)).half()
    gm = torch.zeros(B, 10, len(GPS_CHANNELS), dtype=torch.bool)
    quiet = dict(rot_deg=0.0, scale=0.0, noise_g=0.0, p_crop=0.0, p_burst_crop=0.0, p_drop_gps=0.0, p_drop_acc=0.0)
    a = train.augment(acc, am, gps, gm, _cfg(axis_scale=0.2, **quiet), torch.tensor([40]))[0]
    per_axis = a[:, 0, :]
    assert torch.all(per_axis >= 0.8 - 1e-6) and torch.all(per_axis <= 1.2 + 1e-6)
    assert torch.all(a == a[:, :1])  # one gain per window and axis, constant in time
    assert (per_axis[:, 0] != per_axis[:, 1]).all()  # axes independent
    a = train.augment(acc, am, gps, gm, _cfg(offset_g=0.05, **quiet), torch.tensor([40]))[0]
    off = a[:, 0, :] - 1.0
    assert off.abs().max() <= 0.05 + 1e-6 and off.abs().max() > 0.04 and torch.all(a == a[:, :1])
    am[:, 100:] = False  # offset is masked like everything else
    a = train.augment(acc, am, gps, gm, _cfg(offset_g=0.05, **quiet), torch.tensor([40]))[0]
    assert torch.all(a[:, 100:] == 0)


def test_grad_reverse_and_bird_head():
    x = torch.randn(5, 4, requires_grad=True)
    (grad_reverse(x, 0.3) * torch.arange(4.0)).sum().backward()
    assert torch.allclose(x.grad, -0.3 * torch.arange(4.0).expand(5, 4))
    assert torch.equal(grad_reverse(x, 0.3), x)
    assert train.adv_coeff(0, 100) == 0.0 and abs(train.adv_coeff(100, 100) - 0.9999092) < 1e-6
    assert train.adv_coeff(10**6, 100) == train.adv_coeff(100, 100) and train.adv_coeff(5, 0) > 0.999
    assert BirdHead(128, 7, 32)(torch.randn(3, 128)).shape == (3, 7)


def test_bird_adv_training_and_resume(tmp_path):
    win = _synthetic(tmp_path)
    common = ["--windows", str(win), "--batch", "64", "--birds-per-batch", "4", "--warmup", "20",
              "--holdout-per-cohort", "2", "--device", "cpu", "--step-g", "0.001953125", "--clip-g", "2.0"]
    run = tmp_path / "adv"
    train.main(common + ["--run", str(run), "--steps", "100", "--save-every", "50", "--bird-adv", "0.5",
                         "--bird-adv-warmup", "100", "--yaw-deg", "20", "--axis-scale", "0.05", "--offset-g", "0.02"])
    adv = pd.read_csv(run / "adv.csv")
    assert list(adv.columns) == ["step", "adv_loss", "adv_acc", "coeff"] and len(adv) == 1
    assert list(pd.read_csv(run / "loss.csv").columns) == ["step", "loss", "lr", "pos_cos", "neg_cos", "sec"]
    ck = torch.load(sorted(run.glob("ckpt_*.pt"))[-1])
    assert ck["bird_head"]["net.2.weight"].shape[0] == 6  # 10 birds - 4 held out
    train.main(common + ["--run", str(run), "--steps", "200", "--save-every", "50", "--bird-adv", "0.5",
                         "--bird-adv-warmup", "100", "--yaw-deg", "20", "--axis-scale", "0.05", "--offset-g", "0.02"])
    assert pd.read_csv(run / "adv.csv")["step"].tolist() == [100, 200]  # resumed from step 100 with the head
    # adversary off: no head in the checkpoint, no adv.csv
    off = tmp_path / "off"
    train.main(common + ["--run", str(off), "--steps", "10", "--save-every", "10"])
    ck_off = torch.load(off / "ckpt_0000010.pt")
    assert "bird_head" not in ck_off and not (off / "adv.csv").exists()
    n_enc = len(list(Encoder(len(GPS_CHANNELS)).parameters()))
    assert len(ck_off["opt"]["param_groups"][0]["params"]) == n_enc  # optimizer holds the encoder only
    assert len(ck["opt"]["param_groups"][0]["params"]) == n_enc + 4  # + head (2 linear layers)
    # evaluation of one section only; feature floor in the gravity-aligned frame
    evaluate.main(["--run", str(run), "--windows", str(win), "--events", str(tmp_path / "events"), "--only", "nuisance"])
    assert not (run / "evaluation.json").exists()
    nv = json.loads((run / "evaluation_nuisance.json").read_text())["nuisance_vs_features"]
    assert nv["features_posture_frame"] == "gravity_aligned"
    for k in ("h", "features", "features_sensor_frame", "h_minus_features"):
        lo, hi = nv[f"bird_id_holdout_{k}_ci95"]
        assert -1.0 <= lo <= hi <= 1.0
    rec = nv["bird_id_holdout_h_recall_by_bird"]
    assert abs(np.mean(list(rec.values())) - nv["bird_id_holdout_h"]) < 1e-3  # balanced accuracy = mean recall
    assert len(nv["bird_id_holdout_h_recall_by_bird"]) == 4
    meta, _ = evaluate.load_run(run, win)
    rows = np.flatnonzero(meta["vedba"].to_numpy() < 0.01)
    post = evaluate.rotated_posture_rows(run, win, meta, rows)
    assert np.isnan(post[np.setdiff1d(np.arange(len(meta)), rows)]).all()
    assert np.abs(post[rows]).max() < 2.0  # still windows: level after rotation
    from aquilacl import states
    b = meta["bird"].to_numpy()[rows]
    sensor = np.concatenate([states.rotated_posture(np.load(win / x / "acc.npy"), np.load(win / x / "acc_mask.npy"),
                                                    np.eye(3, dtype=np.float32)) for x in sorted(set(b))])
    assert np.abs(sensor[:, 1]).max() > 5.0  # sensor frame carries the synthetic mounting tilt


def test_old_config_without_new_keys_resumes(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    cfg = train.get_args(["--windows", str(tmp_path), "--run", str(run), "--step-g", "0", "--clip-g", "0"])
    train.resolve_format(cfg)
    stored = {k: (v if isinstance(v, (int, float, list, bool)) else str(v)) for k, v in vars(cfg).items()}
    for k in ("bird_adv", "bird_adv_warmup", "bird_adv_hidden", "bird_adv_mode", "axis_scale", "offset_g", "yaw_deg",
              "derotate_yaw", "yaw_flip_p", "p_neighbour", "mask_recon", "mask_ratio", "mask_span", "holdout_salt",
              "holdout_exclude"):
        stored.pop(k)
    (run / "config.json").write_text(json.dumps(stored))
    (run / "ckpt_0000001.pt").write_bytes(b"")
    train.record_config(cfg)  # defaults for keys the old run did not have: accepted
    assert "bird_adv" not in json.loads((run / "config.json").read_text())  # stored config left as it was
    cfg.bird_adv = 0.5
    with pytest.raises(SystemExit):
        train.record_config(cfg)


def test_sampler_four_birds_and_neg_mask():
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(6), [50, 400, 300, 1000, 30, 700])
    w = rng.uniform(0.1, 1, len(codes))
    s = train.BirdBatchSampler(codes, w, 1024, 4, 0)
    for _ in range(20):
        i = s()
        assert len(i) == 1024
        b, n = np.unique(codes[i], return_counts=True)
        assert len(b) == 4 and np.all(n == 256)
        assert all(len(np.unique(codes[i[k * 256:(k + 1) * 256]])) == 1 for k in range(4))
    # fewer training birds than birds_per_batch: drawn with replacement, batch size kept
    s2 = train.BirdBatchSampler(codes[codes < 3], w[codes < 3], 1024, 4, 0)
    i = s2()
    assert len(i) == 1024 and set(np.unique(codes[i])) <= {0, 1, 2}
    # bird 0 has 50 windows for 256 draws: duplicates occur and are masked as negatives (same bird, dt 0)
    tsec = torch.from_numpy(np.arange(len(codes)) * 10.0)
    bc = torch.from_numpy(codes)
    it = torch.from_numpy(np.concatenate([s2() for _ in range(1)]))
    same = (bc[it][:, None] == bc[it][None]) & ((tsec[it][:, None] - tsec[it][None]).abs() <= 1800.0)
    dup = (it[:, None] == it[None]) & ~torch.eye(len(it), dtype=torch.bool)
    assert dup.any() and torch.all(same[dup])
    z0, z1 = torch.randn(len(it), 8), torch.randn(len(it), 8)
    assert torch.isfinite(nt_xent(z0, z1, 0.2, same))


def test_bird_bootstrap_ci_on_toy_confusion():
    y = np.repeat(["a", "b", "c", "d"], 10)
    pred = y.copy()
    pred[:3] = "b"  # a: 7/10
    pred[10:20] = "c"  # b: 0/10
    rec = evaluate.per_class_recall(y, pred, sorted(set(y)))
    np.testing.assert_allclose(rec, [0.7, 0.0, 1.0, 1.0])
    from sklearn.metrics import balanced_accuracy_score
    assert abs(rec.mean() - balanced_accuracy_score(y, pred)) < 1e-12
    ci = evaluate.bootstrap_mean_ci(np.stack([rec, rec - 0.1, rec - (rec - 0.1)]), 500, 0)
    assert ci.shape == (3, 2)
    assert 0.0 <= ci[0, 0] < rec.mean() < ci[0, 1] <= 1.0
    np.testing.assert_allclose(ci[1], ci[0] - 0.1)  # same resamples for every row
    np.testing.assert_allclose(ci[2], [0.1, 0.1])  # paired constant difference has zero width
    np.testing.assert_allclose(evaluate.bootstrap_mean_ci(np.full(12, 0.5)), [[0.5, 0.5]])


def test_confusion_adversary_gradient_routing():
    """Confusion mode: the head learns only from its own cross-entropy on detached h; the encoder side receives
    only the push towards a uniform output, whose minimum is log(n_birds) (bounded, unlike gradient reversal)."""
    from aquilacl.model import BirdHead

    torch.manual_seed(0)
    head = BirdHead(8, 5, 16)
    h = torch.randn(32, 8, requires_grad=True)
    y = torch.randint(0, 5, (32,))
    adv = torch.nn.functional.cross_entropy(head(h.detach()), y)
    frozen = {k: v.detach() for k, v in head.named_parameters()}
    conf = -torch.func.functional_call(head, frozen, (h,)).log_softmax(1).mean(1).mean()
    (adv + conf).backward()
    g_head = {k: v.grad.clone() for k, v in head.named_parameters()}
    head.zero_grad()
    adv2 = torch.nn.functional.cross_entropy(head(h.detach()), y)
    adv2.backward()
    for k, v in head.named_parameters():
        assert torch.allclose(g_head[k], v.grad)  # head gradient comes from adv alone
    h2 = h.detach().clone().requires_grad_(True)
    conf2 = -torch.func.functional_call(head, frozen, (h2,)).log_softmax(1).mean(1).mean()
    conf2.backward()
    assert torch.allclose(h.grad, h2.grad)  # h gradient comes from the confusion term alone
    assert conf.item() >= np.log(5) - 1e-6


# ---------------------------------------------------------------- r7 options (yaw de-rotation, neighbour view,
# masked reconstruction, new held-out set); all off by default
import hashlib

from aquilacl.model import AccDecoder


def _reference_augment_r6(acc, amask, gps, gmask, cfg, short_lengths):
    """Copy of train.augment as used for r6 (with --yaw-deg/--axis-scale/--offset-g, before --yaw-flip-p)."""
    B = acc.shape[0]
    dev = acc.device
    amask = train.contiguous_crop(amask, cfg.min_acc_samples, cfg.p_crop)
    acc, amask = train.burst_crop(acc, amask, short_lengths, cfg.p_burst_crop)
    R = train.random_rotation(B, cfg.rot_deg, dev)
    scale = torch.empty(B, 1, 1, device=dev).uniform_(1 - cfg.scale, 1 + cfg.scale)
    yaw_deg, axis_scale, offset_g = cfg.yaw_deg, cfg.axis_scale, cfg.offset_g
    Rz = train.yaw_rotation(B, yaw_deg, dev) if yaw_deg > 0 else None
    gain = torch.empty(B, 1, 3, device=dev).uniform_(1 - axis_scale, 1 + axis_scale) if axis_scale > 0 else None
    offset = torch.empty(B, 1, 3, device=dev).uniform_(-offset_g, offset_g) if offset_g > 0 else None
    a = acc.float()
    if Rz is not None:
        a = a @ Rz.transpose(1, 2)
    a = (a @ R.transpose(1, 2)) * scale
    if gain is not None:
        a = a * gain
    if offset is not None:
        a = a + offset
    acc = a + cfg.noise_g * torch.randn_like(acc, dtype=torch.float32)
    acc = acc * amask[..., None]
    sec = gmask.any(-1)
    sec = sec & (torch.rand_like(sec, dtype=torch.float32) > cfg.p_gps_drop)
    sec = train.contiguous_crop(sec, 3, cfg.p_crop)
    gmask = gmask & sec[..., None]
    gps = (gps.float() + cfg.gps_noise * torch.randn_like(gps, dtype=torch.float32)) * gmask
    has_a, has_g = amask.any(1), gmask.flatten(1).any(1)
    r = torch.rand(B, device=dev)
    drop_g = has_a & has_g & (r < cfg.p_drop_gps)
    drop_a = has_a & has_g & (r >= cfg.p_drop_gps) & (r < cfg.p_drop_gps + cfg.p_drop_acc)
    amask = amask & ~drop_a[:, None]
    acc = acc * amask[..., None]
    gmask = gmask & ~drop_g[:, None, None]
    gps = gps * gmask
    return acc, amask, gps, gmask


def _old_holdout_birds(birds, cohort, per_cohort=6, salt="aquilacl"):
    out = []
    for c in sorted(set(cohort.values())):
        bs = sorted((b for b in birds if cohort[b] == c), key=lambda b: hashlib.sha256((salt + b).encode()).hexdigest())
        out += bs[:per_cohort]
    return sorted(out)


@pytest.mark.parametrize("kw", [dict(), dict(yaw_deg=60.0), dict(yaw_deg=60.0, axis_scale=0.05, offset_g=0.02)])
def test_augment_r6_unchanged_without_flip(kw):
    inp = _aug_inputs(seed=3)
    cfg = _cfg(**kw)
    torch.manual_seed(7)
    ref = _reference_augment_r6(*inp[:4], cfg, inp[4])
    ref_next = torch.rand(1)
    torch.manual_seed(7)
    new = train.augment(*inp[:4], cfg, inp[4])
    for a, b in zip(ref, new):
        assert torch.equal(a, b)
    assert torch.equal(torch.rand(1), ref_next)


def test_r7_defaults():
    cfg = _cfg()
    assert (cfg.derotate_yaw, cfg.yaw_flip_p, cfg.p_neighbour, cfg.mask_recon) == (False, 0.0, 0.0, 0.0)
    assert (cfg.holdout_salt, cfg.holdout_exclude, cfg.mask_ratio, cfg.mask_span) == ("aquilacl", None, 0.3, 10)


def test_yaw_flip():
    torch.manual_seed(0)
    R = train.flip_rotation(20000, 0.5, "cpu")
    flipped = R[:, 0, 0] < 0
    assert abs(flipped.float().mean().item() - 0.5) < 0.02
    assert torch.all(R[flipped, 1, 1] == -1) and torch.all(R[:, 2, 2] == 1) and torch.all(R[~flipped, 0, 0] == 1)
    assert torch.equal(train.flip_rotation(50, 0.0, "cpu"), torch.eye(3).expand(50, 3, 3))
    acc = torch.tensor([0.3, -0.2, 1.0]).expand(64, 200, 3).half()
    am = torch.ones(64, 200, dtype=torch.bool)
    gps = torch.zeros(64, 10, len(GPS_CHANNELS)).half()
    gm = torch.zeros(64, 10, len(GPS_CHANNELS), dtype=torch.bool)
    quiet = dict(rot_deg=0.0, scale=0.0, noise_g=0.0, p_crop=0.0, p_burst_crop=0.0, p_drop_gps=0.0, p_drop_acc=0.0)
    a = train.augment(acc, am, gps, gm, _cfg(yaw_flip_p=1.0, **quiet), torch.tensor([40]))[0]
    expect = torch.tensor([-0.3, 0.2, 1.0]).half().float()
    assert torch.allclose(a, expect.expand_as(a), atol=1e-6)


def test_yaw_estimator_recovers_planted_yaw():
    rng = np.random.default_rng(1)
    t = np.arange(200) / 20
    for planted in (-70.0, 25.0, 100.0, 170.0):
        ang = np.radians(planted)
        surge = 0.3 * np.sin(2 * np.pi * 2.6 * t[None] + rng.uniform(0, 6, (80, 1)))
        acc = np.zeros((80, 200, 3), np.float32)
        acc[..., 0], acc[..., 1] = surge * np.cos(ang), surge * np.sin(ang)
        acc[..., 2] = 1.0 + 0.5 * np.sin(2 * np.pi * 2.6 * t)[None]
        acc += rng.normal(0, 0.02, acc.shape)
        mask = np.ones((80, 200), bool)
        mask[:10, 30:] = False  # 30 present samples: too short to qualify
        y = dataset.estimate_yaw(acc, mask)
        assert y["yaw_source"] == "flapping" and y["yaw_n"] == 70 and y["yaw_resultant"] > 0.99
        d = (y["yaw_deg"] - planted) % 180.0
        assert min(d, 180 - d) < 1.0 and -90 < y["yaw_deg"] <= 90
        R = dataset.rotation_z(-y["yaw_deg"])
        b = acc @ R.T
        np.testing.assert_allclose(b[..., 2], acc[..., 2], atol=1e-6)  # z untouched
        th, _ = dataset.axial_angle(b[..., :2], mask)
        assert np.all(np.minimum(th, 180 - th)[10:] < 3.0)  # flapping axis now on +-x
        assert dataset.estimate_yaw(acc[:55], mask[:55])["yaw_source"] == "too_few_flapping"  # 45 qualify


def test_derotate_yaw_in_load(tmp_path):
    yaws = [-60.0, 35.0, 80.0, 10.0]
    win = _synthetic(tmp_path, n_birds=4, flap_yaw=yaws)
    base = dataset.load(win)
    assert "yaw_deg" not in base.orientation  # off: orientation as before
    data = dataset.load(win, derotate_yaw=True)
    o = data.orientation.set_index("bird")
    assert (o["yaw_source"] == "flapping").all() and (o["yaw_n"] == 60).all()
    for b, planted in zip(o.index, yaws):
        d = (o.loc[b, "yaw_deg"] - planted) % 180
        assert min(d, 180 - d) < 2.0
        Rroost = np.asarray(base.orientation.set_index("bird").loc[b, "R"]).reshape(3, 3)
        R = np.asarray(o.loc[b, "R"]).reshape(3, 3)
        np.testing.assert_allclose(R, dataset.rotation_z(-o.loc[b, "yaw_deg"]) @ Rroost, atol=1e-5)
    flap = (data.meta["label_flapping"] == 1).to_numpy()
    th, _ = dataset.axial_angle(data.acc[flap][..., :2].astype(np.float32), data.acc_mask[flap])
    assert np.median(np.minimum(th, 180 - th)) < 3.0  # flapping axis on +-x for every bird
    night = data.meta["sun_elev_deg"].to_numpy() < -6
    np.testing.assert_allclose(data.acc[night].astype(float).mean(axis=1)[:, 2], 1.0, atol=0.02)  # gravity still +z
    # without a horizontal flapping axis no window qualifies: no yaw correction, flagged
    win2 = _synthetic(tmp_path / "plain", n_birds=2)
    d2 = dataset.load(win2, derotate_yaw=True)
    b2 = dataset.load(win2)
    assert (d2.orientation["yaw_source"] == "too_few_flapping").all() and (d2.orientation["yaw_deg"] == 0).all()
    assert np.array_equal(d2.acc, b2.acc)


def test_derotate_yaw_reference_and_fallback(tmp_path):
    # axes at +88 and -88 deg are 4 deg apart: corrections must differ by about 4 deg, not 176
    win = _synthetic(tmp_path, n_birds=3, flap_yaw=[88.0, -88.0, 0.0])
    # third bird: too few flapping windows (keep only 2 flap segments) -> population axis
    d = win / "B2"
    m = pd.read_parquet(d / "meta.parquet")
    f = pd.read_parquet(d / "features.parquet")
    keep = ~(f["label_flapping"].to_numpy() & (m["segment"].to_numpy() >= 6))
    for name in ("acc", "acc_mask", "gps", "gps_mask"):
        np.save(d / f"{name}.npy", np.load(d / f"{name}.npy")[keep])
    m[keep].reset_index(drop=True).to_parquet(d / "meta.parquet")
    f[keep].reset_index(drop=True).to_parquet(d / "features.parquet")
    o = dataset.load(win, derotate_yaw=True).orientation.set_index("bird")
    assert o.loc["B2", "yaw_source"] == "population_axis" and o.loc["B2", "yaw_n"] == 12
    assert abs(abs(o.loc["B0", "yaw_deg"] - o.loc["B1", "yaw_deg"]) - 4.0) < 2.0
    ref = o["yaw_ref_deg"].iloc[0]
    assert (o["yaw_ref_deg"] == ref).all() and min(abs(ref - 90), abs(ref + 90)) < 2.0
    assert o.loc["B2", "yaw_deg"] == ref


def test_neighbour_index():
    t = pd.Timestamp("2025-01-01")
    s = lambda x: pd.Timedelta(seconds=x)
    meta = pd.DataFrame([
        dict(bird="A", segment=0, sensitivity_k=1, k_in_segment=0, t_start=t),           # 0: next 1
        dict(bird="A", segment=0, sensitivity_k=1, k_in_segment=1, t_start=t + s(10)),   # 1: prev 0, next 2
        dict(bird="A", segment=0, sensitivity_k=1, k_in_segment=2, t_start=t + s(20)),   # 2: prev 1 (3 is > 30 s)
        dict(bird="A", segment=0, sensitivity_k=1, k_in_segment=3, t_start=t + s(60)),   # 3: none (gap 40 s)
        dict(bird="B", segment=0, sensitivity_k=1, k_in_segment=1, t_start=t + s(10)),   # 4: other bird: none
        dict(bird="A", segment=1, sensitivity_k=0, k_in_segment=0, t_start=t + s(30)),   # 5: next 7 (other mode 6 not)
        dict(bird="A", segment=1, sensitivity_k=1, k_in_segment=1, t_start=t + s(40)),   # 6: other mode: none
        dict(bird="A", segment=1, sensitivity_k=0, k_in_segment=1, t_start=t + s(40)),   # 7: prev 5
    ], index=[10, 11, 12, 13, 14, 15, 16, 17])  # positions are used, not the index
    nbr = train.neighbour_index(meta)
    assert nbr.tolist() == [[1, 1], [0, 2], [1, 1], [3, 3], [4, 4], [7, 7], [6, 6], [5, 5]]
    torch.manual_seed(0)
    i = torch.arange(8).repeat(500)
    j = train.pick_neighbour(i, torch.from_numpy(nbr), 0.5)
    assert all(int(jj) in set(nbr[int(ii)]) | {int(ii)} for ii, jj in zip(i[:200], j[:200]))
    moved = (j != i)[i == 1].float().mean().item()
    assert 0.4 < moved < 0.6  # row 1 has two distinct neighbours: moved with probability p
    assert torch.equal(train.pick_neighbour(i, torch.from_numpy(nbr), 0.0), i)


def test_mask_spans_and_recon_loss():
    torch.manual_seed(0)
    am = torch.ones(64, 200, dtype=torch.bool)
    am[:16, 120:] = False
    am[16:24, 15:] = False  # 15 present samples < 2 * span: left alone
    hide = train.mask_spans(am, 0.3, 10)
    assert not (hide & ~am).any() and not hide[16:24].any()
    frac = hide[24:].sum(1).float() / am[24:].sum(1)
    assert 0.15 < frac.mean().item() <= 0.3 + 1e-6
    frac16 = hide[:16].sum(1).float() / 120
    assert 0.15 < frac16.mean().item() <= 0.31
    target = torch.randn(64, 200, 3)
    pred = target.clone()
    pred[~hide] += 5.0  # wrong everywhere except at hidden positions: loss 0
    assert train.recon_loss(pred, target, hide).item() == 0.0
    pred2 = target.clone()
    pred2[hide] += 1.0
    assert abs(train.recon_loss(pred2, target, hide).item() - 1.0) < 1e-5
    assert train.recon_loss(pred2, target, torch.zeros_like(hide)).item() == 0.0  # nothing hidden: 0, no NaN


def test_encoder_interface_and_recon_gradients():
    torch.manual_seed(0)
    enc = Encoder(len(GPS_CHANNELS)).eval()
    acc, am, gps, gm, _ = _aug_inputs(B=8)
    acc = acc.float()
    gps = gps.float()
    h, z = enc(acc, am, gps, gm)
    h2, z2, y = enc(acc, am, gps, gm, return_acc_seq=True)
    assert torch.equal(h, h2) and torch.equal(z, z2) and y.shape == (8, 128, 50)
    assert torch.equal(enc.acc(acc, am), enc.acc.pool(enc.acc.seq_features(acc, am), am))
    enc.train()
    dec = AccDecoder(128)
    hide = train.mask_spans(am, 0.3, 10)
    _, _, y = enc(acc * ~hide[..., None], am & ~hide, gps, gm, return_acc_seq=True)
    pred = dec(y, 200)
    assert pred.shape == (8, 200, 3)
    train.recon_loss(pred, acc, hide).backward()
    assert enc.acc.net[0].c1.weight.grad.abs().sum() > 0 and dec.up.weight.grad.abs().sum() > 0


def test_holdout_salt_and_exclude():
    birds = [f"bird{i:03d}" for i in range(40)]
    cohort = {b: ("2024" if i < 25 else "2025") for i, b in enumerate(birds)}
    old = _old_holdout_birds(birds, cohort, 6)
    assert dataset.holdout_birds(birds, cohort, 6) == old
    assert dataset.holdout_birds(birds, cohort, 6, salt="aquilacl", exclude=None) == old
    new = dataset.holdout_birds(birds, cohort, 6, salt="r7", exclude=old)
    assert not set(new) & set(old) and len(new) == 12
    assert sum(cohort[b] == "2024" for b in new) == 6 and sum(cohort[b] == "2025" for b in new) == 6
    assert new == dataset.holdout_birds(birds, cohort, 6, salt="r7", exclude=set(old))  # deterministic
    assert dataset.holdout_birds(birds, cohort, 6, salt="r7") != old


def test_r7_training_all_options_and_resume(tmp_path):
    win = _synthetic(tmp_path, flap_yaw=[(37 * b) % 180 - 90 for b in range(10)])
    old = dataset.holdout_birds([f"B{b}" for b in range(10)], {f"B{b}": ("2024" if b < 5 else "2025") for b in range(10)}, 2)
    ex = tmp_path / "old_holdout.json"
    ex.write_text(json.dumps(old))
    opts = ["--windows", str(win), "--batch", "64", "--birds-per-batch", "4", "--warmup", "5", "--holdout-per-cohort", "2",
            "--device", "cpu", "--step-g", "0.001953125", "--clip-g", "2.0", "--derotate-yaw", "--yaw-deg", "15",
            "--yaw-flip-p", "0.5", "--p-neighbour", "0.5", "--mask-recon", "1.0", "--holdout-salt", "r7",
            "--holdout-exclude", str(ex), "--run", str(tmp_path / "r7"), "--save-every", "10"]
    run = tmp_path / "r7"
    train.main(opts + ["--steps", "10"])
    hold = json.loads((run / "holdout.json").read_text())
    assert len(hold) == 4 and not set(hold) & set(old)
    orient = json.loads((run / "orientation.json").read_text())
    assert all(o["yaw_source"] == "flapping" for o in orient)
    assert (run / "recon.csv").read_text().splitlines()[0] == "step,recon_loss,masked_frac"
    ck = torch.load(run / "ckpt_0000010.pt")
    n_enc = len(list(Encoder(len(GPS_CHANNELS)).parameters()))
    assert "acc_decoder" in ck and len(ck["opt"]["param_groups"][0]["params"]) == n_enc + 4
    train.main(opts + ["--steps", "20"])  # resume: decoder and optimizer state loaded, continues to 20
    ck2 = torch.load(run / "ckpt_0000020.pt")
    assert ck2["step"] == 20 and not torch.equal(ck2["acc_decoder"]["up.weight"], ck["acc_decoder"]["up.weight"])
    assert len(list((run / "embeddings").glob("*.npz"))) == 10
    with pytest.raises(SystemExit):  # resubmission without the options is refused
        train.main(opts[:16] + ["--run", str(run), "--steps", "20"])
    # all r7 options off: no decoder, no recon.csv, no yaw columns
    off = tmp_path / "off"
    train.main(opts[:16] + ["--run", str(off), "--steps", "5", "--save-every", "5"])
    assert "acc_decoder" not in torch.load(off / "ckpt_0000005.pt") and not (off / "recon.csv").exists()
    assert "yaw_source" not in json.loads((off / "orientation.json").read_text())[0]


def test_recon_logged_every_100_steps(tmp_path):
    win = _synthetic(tmp_path, n_birds=4)
    run = tmp_path / "r"
    train.main(["--windows", str(win), "--run", str(run), "--batch", "32", "--birds-per-batch", "2", "--warmup", "5",
                "--holdout-per-cohort", "1", "--device", "cpu", "--step-g", "0", "--clip-g", "0", "--steps", "100",
                "--save-every", "100", "--mask-recon", "0.5", "--p-neighbour", "1.0"])
    rec = pd.read_csv(run / "recon.csv")
    assert rec["step"].tolist() == [100] and np.isfinite(rec["recon_loss"]).all() and 0.1 < rec["masked_frac"].iloc[0] < 0.35
