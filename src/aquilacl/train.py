"""Stage 4: SimCLR training on 10 s windows, then deterministic embedding of every window.

Usage:
    python -m aquilacl.train --windows <data>/windows_10s --run <runs>/<name> [options]

The run directory receives config.json, holdout.json, orientation.csv, loss.csv (appended, never truncated),
checkpoints, and embeddings/<bird>.npz (h and z for every window, in window order). Re-running with the same
--run resumes from the latest checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import dataset
from .model import Encoder, nt_xent
from .windows import GPS_CHANNELS


# --------------------------------------------------------------------------- augmentation (batched, on device)
def random_rotation(n: int, max_deg: float, device) -> torch.Tensor:
    axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=1)
    ang = torch.deg2rad(torch.empty(n, device=device).uniform_(-max_deg, max_deg))
    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(1, 2)
    eye = torch.eye(3, device=device).expand(n, 3, 3)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return eye + s * K + (1 - c) * K @ K


def contiguous_crop(mask: torch.Tensor, min_len: int, p: float) -> torch.Tensor:
    """With probability p keep one random contiguous span (>= min_len present samples) of each row."""
    B, T = mask.shape
    pos = torch.arange(T, device=mask.device).expand(B, T)
    n = mask.sum(1)
    first = torch.where(mask.any(1), mask.float().argmax(1), torch.zeros_like(n))
    L = (min_len + torch.rand(B, device=mask.device) * (n - min_len).clamp(min=0)).long().clamp(max=T)
    start = first + (torch.rand(B, device=mask.device) * (n - L).clamp(min=0)).long()
    keep = (pos >= start[:, None]) & (pos < (start + L)[:, None])
    do = (torch.rand(B, device=mask.device) < p) & (n > min_len)
    return torch.where(do[:, None], mask & keep, mask)


def burst_crop(acc: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor, p: float):
    """With probability p turn a window into what a short recording bout looks like.

    Windows that are not full (a single 40-sample bout, the 64-sample tail of a 264-sample burst, a short run
    of continuous-mode bursts) hold their samples from the window start, while continuous segments fill all
    200 slots; a probe tells them apart at 0.998. Cutting one view of a full window to a length drawn from
    ``lengths`` (the empirical lengths of non-full windows) and moving it to the window start teaches the
    encoder that both describe the same behaviour, so states are not split by recording schedule.
    """
    B, T = mask.shape
    dev = mask.device
    n = mask.sum(1)
    L = lengths.to(dev)[torch.randint(len(lengths), (B,), device=dev)]
    do = (torch.rand(B, device=dev) < p) & (n > L)
    first = torch.where(mask.any(1), mask.float().argmax(1), torch.zeros_like(n))
    start = first + (torch.rand(B, device=dev) * (n - L).clamp(min=0)).long()
    start = torch.where(do, start, torch.zeros_like(start))
    t = torch.arange(T, device=dev)[None]
    src = (start[:, None] + t).clamp(max=T - 1)
    shifted_m = mask.gather(1, src) & (t < L[:, None])
    shifted_a = acc.gather(1, src[..., None].expand(-1, -1, acc.shape[-1]))
    return torch.where(do[:, None, None], shifted_a, acc), torch.where(do[:, None], shifted_m, mask)


def augment(acc, amask, gps, gmask, cfg, short_lengths: torch.Tensor) -> tuple:
    B = acc.shape[0]
    dev = acc.device
    amask = contiguous_crop(amask, cfg.min_acc_samples, cfg.p_crop)
    acc, amask = burst_crop(acc, amask, short_lengths, cfg.p_burst_crop)
    R = random_rotation(B, cfg.rot_deg, dev)
    scale = torch.empty(B, 1, 1, device=dev).uniform_(1 - cfg.scale, 1 + cfg.scale)
    acc = (acc.float() @ R.transpose(1, 2)) * scale + cfg.noise_g * torch.randn_like(acc, dtype=torch.float32)
    acc = acc * amask[..., None]
    # GPS: drop seconds, then optionally keep a contiguous span
    sec = gmask.any(-1)
    sec = sec & (torch.rand_like(sec, dtype=torch.float32) > cfg.p_gps_drop)
    sec = contiguous_crop(sec, 3, cfg.p_crop)
    gmask = gmask & sec[..., None]
    gps = (gps.float() + cfg.gps_noise * torch.randn_like(gps, dtype=torch.float32)) * gmask
    # whole-modality dropout, never both
    has_a, has_g = amask.any(1), gmask.flatten(1).any(1)
    r = torch.rand(B, device=dev)
    drop_g = has_a & has_g & (r < cfg.p_drop_gps)
    drop_a = has_a & has_g & (r >= cfg.p_drop_gps) & (r < cfg.p_drop_gps + cfg.p_drop_acc)
    amask = amask & ~drop_a[:, None]
    acc = acc * amask[..., None]
    gmask = gmask & ~drop_g[:, None, None]
    gps = gps * gmask
    return acc, amask, gps, gmask


# --------------------------------------------------------------------------- sampling
class BirdBatchSampler:
    """Batches of ``birds_per_batch`` birds x (batch / birds_per_batch) windows, movement-weighted within bird."""

    def __init__(self, bird_codes: np.ndarray, weights: np.ndarray, batch: int, birds_per_batch: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.batch, self.k = batch, birds_per_batch
        self.idx, self.p = [], []
        for b in np.unique(bird_codes):
            i = np.flatnonzero(bird_codes == b)
            w = weights[i]
            self.idx.append(i)
            self.p.append(w / w.sum())

    def __call__(self) -> np.ndarray:
        birds = self.rng.choice(len(self.idx), self.k, replace=len(self.idx) < self.k)
        per = self.batch // self.k
        return np.concatenate([self.rng.choice(self.idx[b], per, p=self.p[b]) for b in birds])


# --------------------------------------------------------------------------- main
def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--birds-per-batch", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--movement-power", type=float, default=0.5, help="window weight = clip(VeDBA)^power")
    ap.add_argument("--neg-mask-s", type=float, default=1800.0, help="windows of the same bird within this many seconds are not negatives")
    ap.add_argument("--min-acc-samples", type=int, default=24)
    ap.add_argument("--p-crop", type=float, default=0.8)
    ap.add_argument("--p-burst-crop", type=float, default=0.3, help="probability a view is cut to the length of a real short window")
    ap.add_argument("--step-g", default="auto", help="common acceleration step in g; 'auto' = <windows>/format.json, 0 = off")
    ap.add_argument("--clip-g", default="auto", help="common acceleration range +-g; 'auto' = <windows>/format.json, 0 = off")
    ap.add_argument("--mask-gps-channels", nargs="*", default=["eobs:horizontal-accuracy-estimate"],
                    help="GPS channels hidden from the model (fix quality is not behaviour)")
    ap.add_argument("--rot-deg", type=float, default=15.0)
    ap.add_argument("--scale", type=float, default=0.1)
    ap.add_argument("--noise-g", type=float, default=0.01)
    ap.add_argument("--p-gps-drop", type=float, default=0.2)
    ap.add_argument("--gps-noise", type=float, default=0.02)
    ap.add_argument("--p-drop-gps", type=float, default=0.25)
    ap.add_argument("--p-drop-acc", type=float, default=0.15)
    ap.add_argument("--holdout-per-cohort", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--embed-only", action="store_true", help="skip training; embed with the latest checkpoint")
    return ap.parse_args(argv)


def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def encode(model, acc, amask, gps, gmask, dev, batch: int = 8192):
    """Deterministic h and z for prepared numpy arrays (no augmentation)."""
    model.eval()
    H, Z = [], []
    for s in range(0, len(acc), batch):
        sl = slice(s, s + batch)
        am = torch.as_tensor(np.asarray(amask[sl])).to(dev)
        a = torch.as_tensor(np.asarray(acc[sl])).to(dev).float()
        g = torch.as_tensor(np.asarray(gps[sl])).to(dev).float()
        gm = torch.as_tensor(np.asarray(gmask[sl])).to(dev)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            h, z = model(a, am, g, gm)
        H.append(h.float().cpu().numpy())
        Z.append(z.float().cpu().numpy())
    model.train()
    return np.concatenate(H), np.concatenate(Z)


def embed(model, data, dev, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    H, Z = encode(model, data.acc, data.acc_mask, data.gps, data.gps_mask, dev)
    for b, idx in data.meta.groupby("bird").indices.items():
        np.savez(out / f"{b}.npz", h=H[idx].astype(np.float16), z=Z[idx].astype(np.float16), row=idx)


RUN_CONTROL = {"run", "steps", "embed_only", "device", "save_every"}  # may differ between resubmissions


def resolve_format(cfg) -> None:
    """Replace 'auto' step/clip by the values the windows stage recorded in <windows>/format.json."""
    fmt_path = cfg.windows / "format.json"
    for key in ("step_g", "clip_g"):
        v = getattr(cfg, key)
        if v == "auto":
            if not fmt_path.exists():
                raise SystemExit(f"{fmt_path} missing: pass --{key.replace('_', '-')} explicitly (0 = off)")
            v = json.loads(fmt_path.read_text())[key]
            if v is None:
                raise SystemExit(f"{fmt_path} has no {key} (no reliable calibration for one setting): pass it explicitly")
        setattr(cfg, key, float(v))


def record_config(cfg) -> None:
    """Write config.json for a new run; for an existing run refuse settings that differ from the stored ones."""
    cur = {k: (v if isinstance(v, (int, float, list, bool)) else str(v)) for k, v in vars(cfg).items()}
    path = cfg.run / "config.json"
    if path.exists() and any(cfg.run.glob("ckpt_*.pt")):
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), v) for k, v in cur.items() if k not in RUN_CONTROL and str(old.get(k)) != str(v)}
        if diff:
            raise SystemExit(f"{cfg.run} was trained with other settings (stored, given): {diff}")
        return
    path.write_text(json.dumps(cur, indent=1))


def main(argv=None):
    cfg = get_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.run.mkdir(parents=True, exist_ok=True)
    resolve_format(cfg)
    if cfg.embed_only and not any(cfg.run.glob("ckpt_*.pt")):
        raise SystemExit(f"--embed-only but no checkpoint in {cfg.run}")
    record_config(cfg)
    dev = cfg.device

    data = dataset.load(cfg.windows, step_g=cfg.step_g, clip_g=cfg.clip_g, mask_gps_channels=tuple(cfg.mask_gps_channels))
    data.orientation.drop(columns="R").to_csv(cfg.run / "orientation.csv", index=False)
    (cfg.run / "orientation.json").write_text(json.dumps(data.orientation.to_dict("records")))
    birds = sorted(data.meta["bird"].unique())
    cohort = data.meta.groupby("bird")["cohort"].first().to_dict()
    hold = dataset.holdout_birds(birds, cohort, cfg.holdout_per_cohort)
    (cfg.run / "holdout.json").write_text(json.dumps(hold, indent=1))
    train_rows = ~data.meta["bird"].isin(hold).to_numpy()

    # whole training set on the device
    tr = np.flatnonzero(train_rows)
    A = torch.from_numpy(data.acc[tr]).to(dev)
    AM = torch.from_numpy(data.acc_mask[tr]).to(dev)
    G = torch.from_numpy(data.gps[tr]).to(dev)
    GM = torch.from_numpy(data.gps_mask[tr]).to(dev)
    meta_tr = data.meta.iloc[tr]
    bird_code = pd.factorize(meta_tr["bird"])[0]
    t0 = meta_tr["t_start"].min()
    tsec = torch.from_numpy(((meta_tr["t_start"] - t0) / pd.Timedelta(seconds=1)).to_numpy(np.float64)).to(dev)
    bcode_t = torch.from_numpy(bird_code).to(dev)
    n_acc = meta_tr["acc_samples"].to_numpy()
    short = n_acc[(n_acc < 190) & (n_acc >= cfg.min_acc_samples)]
    short_len = torch.from_numpy(np.random.default_rng(cfg.seed).choice(short, min(len(short), 100000), replace=False)
                                 if len(short) else np.array([cfg.min_acc_samples])).long().to(dev)
    w = np.clip(meta_tr["vedba"].fillna(0.01).to_numpy(), 0.01, 2.0) ** cfg.movement_power
    sampler = BirdBatchSampler(bird_code, w, cfg.batch, cfg.birds_per_batch, cfg.seed)

    model = Encoder(len(GPS_CHANNELS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    ckpts = sorted(cfg.run.glob("ckpt_*.pt"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        sampler.rng = np.random.default_rng(cfg.seed + step)
        print(f"resumed from {ckpts[-1].name} at step {step}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"windows {len(data.meta):,} train {len(tr):,} holdout birds {len(hold)} params {n_params:,} device {dev}")

    log = open(cfg.run / "loss.csv", "a")
    if log.tell() == 0:
        log.write("step,loss,lr,pos_cos,neg_cos,sec\n")
    t0 = time.time()
    while step < cfg.steps and not cfg.embed_only:
        i = torch.from_numpy(sampler()).to(dev)
        a, am, g, gm = A[i], AM[i], G[i], GM[i]
        v0 = augment(a, am, g, gm, cfg, short_len)
        v1 = augment(a, am, g, gm, cfg, short_len)
        same = (bcode_t[i][:, None] == bcode_t[i][None]) & ((tsec[i][:, None] - tsec[i][None]).abs() <= cfg.neg_mask_s)
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, cfg)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            _, z0 = model(*v0)
            _, z1 = model(*v1)
        with torch.autocast(dev.split(":")[0], enabled=False):
            loss = nt_xent(z0.float(), z1.float(), cfg.tau, same)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                zn0 = torch.nn.functional.normalize(z0.float(), dim=1)
                zn1 = torch.nn.functional.normalize(z1.float(), dim=1)
                pos = (zn0 * zn1).sum(1).mean().item()
                neg = (zn0 @ zn1.T).mean().item()
            log.write(f"{step},{loss.item():.5f},{lr_at(step, cfg):.3e},{pos:.4f},{neg:.4f},{time.time() - t0:.1f}\n")
            log.flush()
        if step % cfg.save_every == 0 or step == cfg.steps:
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, cfg.run / f"ckpt_{step:07d}.pt")
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    log.close()
    del A, AM, G, GM
    embed(model, data, dev, cfg.run / "embeddings")
    print("embeddings written")


if __name__ == "__main__":
    main()
