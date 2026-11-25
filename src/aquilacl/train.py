"""Stage 4: SimCLR training on 10 s windows, then deterministic embedding of every window.

Usage:
    python -m aquilacl.train --windows <data>/windows_10s --run <runs>/<name> [options]

The run directory receives config.json, holdout.json, orientation.csv, loss.csv (appended, never truncated),
checkpoints, and embeddings/<bird>.npz (h and z for every window, in window order). Re-running with the same
--run resumes from the latest checkpoint.

Identity-reduction options (all off by default; r4 and earlier runs are unchanged): --bird-adv W adds a
gradient-reversed bird classifier on h (DANN; loss = nt_xent + W * CE, reversal coefficient ramped over
--bird-adv-warmup steps; adversary loss/accuracy in adv.csv); --axis-scale, --offset-g and --yaw-deg add per-axis
gain, per-axis offset and rotation about gravity to the augmentation.

r7 options (all off by default; r4/r6 settings train exactly as before):
- --derotate-yaw: per-bird mounting yaw from flapping windows, rotated onto +x after the roost rotation
  (dataset.load; applied to training and held-out birds alike, it uses only the ACC-derived flapping label).
  orientation.json/csv record yaw_deg, yaw_n, yaw_resultant, yaw_ref_deg, yaw_source; R there is the combined
  rotation. Birds with < 50 qualifying flapping windows get the population axis (yaw_source population_axis).
  The 180 deg ambiguity (forward/backward) remains: --yaw-flip-p rotates a view by 180 deg about z with that
  probability; --yaw-deg still adds a small jitter.
- --p-neighbour P: with probability P the second view comes from a temporally adjacent window of the same bird
  (same segment and recording mode, k_in_segment +-1, start gap <= 30 s; windows without one use themselves).
- --mask-recon W: masked-reconstruction auxiliary on ACC (spans of --mask-span samples covering about
  --mask-ratio of the present samples of the first view are zeroed and marked missing; a light decoder on the
  ACC branch's per-timestep features predicts them; total loss = nt_xent + W * MSE; recon.csv).
- --holdout-salt / --holdout-exclude: another held-out set, never containing the listed birds.
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
from .model import AccDecoder, BirdHead, Encoder, grad_reverse, nt_xent
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


def yaw_rotation(n: int, max_deg: float, device) -> torch.Tensor:
    """(n, 3, 3) rotations about +z by angles uniform in +-max_deg (mounting yaw, which roost alignment keeps)."""
    ang = torch.deg2rad(torch.empty(n, device=device).uniform_(-max_deg, max_deg))
    c, s = torch.cos(ang), torch.sin(ang)
    R = torch.zeros(n, 3, 3, device=device)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    return R


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


def flip_rotation(n: int, p: float, device) -> torch.Tensor:
    """(n, 3, 3): with probability p a 180 deg rotation about +z (x, y -> -x, -y), else the identity."""
    d = torch.ones(n, 3, device=device)
    d[:, :2] = torch.where(torch.rand(n, device=device)[:, None] < p, -1.0, 1.0)
    return torch.diag_embed(d)


def mask_spans(amask: torch.Tensor, ratio: float, span: int) -> torch.Tensor:
    """Spans to hide for masked reconstruction: (B, T) bool, True where a present sample is masked.

    Each row with at least 2 * span present samples gets round(ratio * n_present / span) (at least 1) spans of
    ``span`` samples, starting uniformly between its first and last present sample, so that about ``ratio`` of
    its present samples are hidden (less where spans overlap). Rows with fewer present samples are left alone.
    """
    B, T = amask.shape
    dev = amask.device
    n = amask.sum(1)
    first = torch.where(amask.any(1), amask.float().argmax(1), torch.zeros_like(n))
    last = T - 1 - amask.flip(1).float().argmax(1)
    k = torch.clamp(torch.round(ratio * n.float() / span), min=1).long()
    kmax = max(1, int(math.ceil(ratio * T / span)))
    room = (last - first + 1 - span).clamp(min=0)
    start = first[:, None] + (torch.rand(B, kmax, device=dev) * (room[:, None] + 1).float()).long().clamp(max=T)
    use = torch.arange(kmax, device=dev)[None] < k[:, None]
    pos = torch.arange(T, device=dev)[None, None]
    cover = ((pos >= start[..., None]) & (pos < start[..., None] + span) & use[..., None]).any(1)
    return cover & amask & (n >= 2 * span)[:, None]


def recon_loss(pred: torch.Tensor, target: torch.Tensor, hide: torch.Tensor) -> torch.Tensor:
    """MSE (float32) over the hidden present samples and 3 axes; 0 when nothing is hidden."""
    wm = hide[..., None].float()
    return (((pred.float() - target.float()) ** 2) * wm).sum() / (3 * wm.sum()).clamp(min=1)


def neighbour_index(meta: pd.DataFrame, max_gap_s: float = 30.0) -> np.ndarray:
    """(N, 2) row positions of the previous and next window of each row in ``meta`` (positional order).

    A neighbour is the window with k_in_segment -+ 1 of the same bird, segment and recording mode
    (sensitivity_k) whose start is at most ``max_gap_s`` away. A missing neighbour is replaced by the other one,
    and a window with neither uses itself in both columns.
    """
    key = ["bird", "segment", "sensitivity_k"]
    d = meta[key + ["k_in_segment", "t_start"]].reset_index(drop=True)
    d["pos"] = np.arange(len(d))
    lookup = d.drop_duplicates(key + ["k_in_segment"])
    out = []
    for dk in (-1, 1):
        q = d[key + ["k_in_segment", "t_start"]].assign(k_in_segment=d["k_in_segment"] + dk)
        r = q.merge(lookup[key + ["k_in_segment", "t_start", "pos"]], on=key + ["k_in_segment"], how="left",
                    suffixes=("", "_n"), validate="many_to_one")
        ok = r["pos"].notna() & ((r["t_start_n"] - r["t_start"]).abs() <= pd.Timedelta(seconds=max_gap_s))
        out.append(np.where(ok, r["pos"].fillna(-1), -1).astype(np.int64))
    prev, nxt = out
    self_ = np.arange(len(d))
    p = np.where(prev >= 0, prev, np.where(nxt >= 0, nxt, self_))
    n = np.where(nxt >= 0, nxt, np.where(prev >= 0, prev, self_))
    return np.stack([p, n], axis=1)


def pick_neighbour(i: torch.Tensor, nbr: torch.Tensor, p: float) -> torch.Tensor:
    """Rows for the second view: with probability p a random neighbour (column 0 or 1 of ``nbr``) of row i."""
    col = torch.randint(0, 2, i.shape, device=i.device)
    use = torch.rand(i.shape, device=i.device) < p
    return torch.where(use, nbr[i, col], i)


def augment(acc, amask, gps, gmask, cfg, short_lengths: torch.Tensor) -> tuple:
    B = acc.shape[0]
    dev = acc.device
    amask = contiguous_crop(amask, cfg.min_acc_samples, cfg.p_crop)
    acc, amask = burst_crop(acc, amask, short_lengths, cfg.p_burst_crop)
    R = random_rotation(B, cfg.rot_deg, dev)
    scale = torch.empty(B, 1, 1, device=dev).uniform_(1 - cfg.scale, 1 + cfg.scale)
    # optional extras (off = no random numbers drawn, so runs without them are unchanged)
    yaw_deg, axis_scale, offset_g = getattr(cfg, "yaw_deg", 0.0), getattr(cfg, "axis_scale", 0.0), getattr(cfg, "offset_g", 0.0)
    Rz = yaw_rotation(B, yaw_deg, dev) if yaw_deg > 0 else None
    gain = torch.empty(B, 1, 3, device=dev).uniform_(1 - axis_scale, 1 + axis_scale) if axis_scale > 0 else None
    offset = torch.empty(B, 1, 3, device=dev).uniform_(-offset_g, offset_g) if offset_g > 0 else None
    flip_p = getattr(cfg, "yaw_flip_p", 0.0)
    Rf = flip_rotation(B, flip_p, dev) if flip_p > 0 else None
    a = acc.float()
    if Rf is not None:
        a = a @ Rf.transpose(1, 2)  # forward/backward ambiguity of the de-rotated yaw (about +z, commutes with Rz)
    if Rz is not None:
        a = a @ Rz.transpose(1, 2)  # yaw about gravity (+z) first, in the roost-aligned frame
    a = (a @ R.transpose(1, 2)) * scale
    if gain is not None:
        a = a * gain
    if offset is not None:
        a = a + offset
    acc = a + cfg.noise_g * torch.randn_like(acc, dtype=torch.float32)
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
def parser() -> argparse.ArgumentParser:
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
    # identity-reduction options, all off by default
    ap.add_argument("--axis-scale", type=float, default=0.0, help="per-axis gain jitter uniform in 1 +- x (on top of --scale)")
    ap.add_argument("--offset-g", type=float, default=0.0, help="per-axis constant offset uniform in +- x g per view")
    ap.add_argument("--yaw-deg", type=float, default=0.0, help="extra rotation about +z (gravity) uniform in +- x degrees")
    ap.add_argument("--bird-adv", type=float, default=0.0,
                    help="weight of a gradient-reversed bird classifier on h (0 = off)")
    ap.add_argument("--bird-adv-warmup", type=int, default=10000,
                    help="steps over which the reversal coefficient ramps 0 -> 1 as 2/(1+exp(-10p))-1")
    ap.add_argument("--bird-adv-hidden", type=int, default=128)
    ap.add_argument("--bird-adv-mode", choices=["confusion", "reverse"], default="confusion",
                    help="confusion: head trained on detached h, encoder pushes the head's output towards uniform "
                         "(bounded, stable); reverse: gradient reversal of the head's cross-entropy (DANN; diverged "
                         "in tests, the encoder makes the head confidently wrong and h grows)")
    # r7 options, all off by default
    ap.add_argument("--derotate-yaw", action="store_true",
                    help="rotate each bird's flapping axis (mounting yaw, mod 180 deg) onto +x after the roost rotation")
    ap.add_argument("--yaw-flip-p", type=float, default=0.0, help="probability of a 180 deg rotation about +z per view")
    ap.add_argument("--p-neighbour", type=float, default=0.0,
                    help="probability that the second view is an adjacent window of the same continuous segment")
    ap.add_argument("--mask-recon", type=float, default=0.0, help="weight of the masked ACC reconstruction loss (0 = off)")
    ap.add_argument("--mask-ratio", type=float, default=0.3, help="share of present ACC samples hidden for reconstruction")
    ap.add_argument("--mask-span", type=int, default=10, help="length in samples of each hidden span")
    ap.add_argument("--holdout-salt", default="aquilacl", help="salt of the held-out bird hash ('aquilacl' = earlier runs)")
    ap.add_argument("--holdout-exclude", type=Path, default=None,
                    help="JSON list of bird ids that must not be held out (e.g. an earlier run's holdout.json)")
    return ap


def get_args(argv=None):
    return parser().parse_args(argv)


def adv_coeff(step: int, warmup: int) -> float:
    """DANN reversal schedule: 2 / (1 + exp(-10 p)) - 1 with p = step / warmup, capped at 1."""
    p = min(step / warmup, 1.0) if warmup > 0 else 1.0
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


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
        ap = parser()
        # a key missing from an older config.json (option added later) matches when the given value is its default
        diff = {k: (old.get(k), v) for k, v in cur.items() if k not in RUN_CONTROL
                and not (k not in old and str(v) == str(ap.get_default(k))) and str(old.get(k)) != str(v)}
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

    data = dataset.load(cfg.windows, step_g=cfg.step_g, clip_g=cfg.clip_g, mask_gps_channels=tuple(cfg.mask_gps_channels),
                        derotate_yaw=cfg.derotate_yaw)
    data.orientation.drop(columns="R").to_csv(cfg.run / "orientation.csv", index=False)
    (cfg.run / "orientation.json").write_text(json.dumps(data.orientation.to_dict("records")))
    birds = sorted(data.meta["bird"].unique())
    cohort = data.meta.groupby("bird")["cohort"].first().to_dict()
    exclude = json.loads(cfg.holdout_exclude.read_text()) if cfg.holdout_exclude is not None else None
    hold = dataset.holdout_birds(birds, cohort, cfg.holdout_per_cohort, salt=cfg.holdout_salt, exclude=exclude)
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
    nbr = None
    if cfg.p_neighbour > 0:
        nbr = torch.from_numpy(neighbour_index(meta_tr)).to(dev)
        n_with = int((nbr[:, 0] != torch.arange(len(nbr), device=dev)).sum())
        print(f"neighbour index: {n_with:,} of {len(nbr):,} training windows have an adjacent window")

    model = Encoder(len(GPS_CHANNELS)).to(dev)
    params = list(model.parameters())
    head = None
    if cfg.bird_adv > 0:  # adversary over the training birds only (held-out birds never enter)
        head = BirdHead(model.fuse[-1].out_features, int(bird_code.max()) + 1, cfg.bird_adv_hidden).to(dev)
        params += list(head.parameters())
    decoder = None
    if cfg.mask_recon > 0:  # decoder from the ACC branch's per-timestep features (2 * width channels)
        decoder = AccDecoder(model.acc.out.in_features // 2).to(dev)
        params += list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    ckpts = sorted(cfg.run.glob("ckpt_*.pt"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=dev)
        model.load_state_dict(ck["model"])
        if head is not None:
            head.load_state_dict(ck["bird_head"])
        if decoder is not None:
            decoder.load_state_dict(ck["acc_decoder"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        sampler.rng = np.random.default_rng(cfg.seed + step)
        print(f"resumed from {ckpts[-1].name} at step {step}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"windows {len(data.meta):,} train {len(tr):,} holdout birds {len(hold)} params {n_params:,} device {dev}")

    log = open(cfg.run / "loss.csv", "a")
    if log.tell() == 0:
        log.write("step,loss,lr,pos_cos,neg_cos,sec\n")
    adv_log = None
    if head is not None:
        adv_log = open(cfg.run / "adv.csv", "a")
        if adv_log.tell() == 0:
            adv_log.write("step,adv_loss,adv_acc,coeff\n")
    rec_log = None
    if decoder is not None:
        rec_log = open(cfg.run / "recon.csv", "a")
        if rec_log.tell() == 0:
            rec_log.write("step,recon_loss,masked_frac\n")
    t0 = time.time()
    while step < cfg.steps and not cfg.embed_only:
        i = torch.from_numpy(sampler()).to(dev)
        a, am, g, gm = A[i], AM[i], G[i], GM[i]
        v0 = augment(a, am, g, gm, cfg, short_len)
        if nbr is not None:  # second view from an adjacent window of the same segment (same bird: same neg mask)
            j = pick_neighbour(i, nbr, cfg.p_neighbour)
            v1 = augment(A[j], AM[j], G[j], GM[j], cfg, short_len)
        else:
            v1 = augment(a, am, g, gm, cfg, short_len)
        if decoder is not None:  # hide spans of the first view; the decoder predicts them from the same pass
            hide = mask_spans(v0[1], cfg.mask_ratio, cfg.mask_span)
            target = v0[0]
            v0 = (v0[0] * ~hide[..., None], v0[1] & ~hide, v0[2], v0[3])
        same = (bcode_t[i][:, None] == bcode_t[i][None]) & ((tsec[i][:, None] - tsec[i][None]).abs() <= cfg.neg_mask_s)
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, cfg)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            if decoder is not None:
                h0, z0, y0 = model(*v0, return_acc_seq=True)
            else:
                h0, z0 = model(*v0)
            h1, z1 = model(*v1)
        with torch.autocast(dev.split(":")[0], enabled=False):
            loss = nt_xent(z0.float(), z1.float(), cfg.tau, same)
            con_loss = loss  # loss.csv keeps the contrastive term alone, comparable with runs without adversary
            if decoder is not None:
                recon = recon_loss(decoder(y0.float(), target.shape[1]), target, hide)
                loss = loss + cfg.mask_recon * recon
            if head is not None:
                coeff = adv_coeff(step, cfg.bird_adv_warmup)
                hb = torch.cat([h0, h1]).float()
                yb = bcode_t[i].repeat(2)
                if cfg.bird_adv_mode == "reverse":
                    logits = head(grad_reverse(hb, coeff))
                    adv_loss = torch.nn.functional.cross_entropy(logits, yb)
                    loss = loss + cfg.bird_adv * adv_loss
                else:  # confusion: the head learns on detached h; the encoder pushes a frozen copy towards uniform
                    logits = head(hb.detach())
                    adv_loss = torch.nn.functional.cross_entropy(logits, yb)
                    frozen = {k: v.detach() for k, v in head.named_parameters()}
                    logp = torch.func.functional_call(head, frozen, (hb,)).log_softmax(1)
                    conf_loss = -logp.mean(1).mean()  # cross-entropy to the uniform distribution, minimum log(n_birds)
                    loss = loss + adv_loss + cfg.bird_adv * coeff * conf_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if head is not None:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if decoder is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 100 == 0:
            with torch.no_grad():
                zn0 = torch.nn.functional.normalize(z0.float(), dim=1)
                zn1 = torch.nn.functional.normalize(z1.float(), dim=1)
                pos = (zn0 * zn1).sum(1).mean().item()
                neg = (zn0 @ zn1.T).mean().item()
            log.write(f"{step},{con_loss.item():.5f},{lr_at(step, cfg):.3e},{pos:.4f},{neg:.4f},{time.time() - t0:.1f}\n")
            log.flush()
            if adv_log is not None:
                acc_adv = (logits.argmax(1) == yb).float().mean().item()
                adv_log.write(f"{step},{adv_loss.item():.5f},{acc_adv:.4f},{coeff:.4f}\n")
                adv_log.flush()
            if rec_log is not None:
                frac = (hide.sum() / (hide.sum() + v0[1].sum()).clamp(min=1)).item()  # hidden / present before hiding
                rec_log.write(f"{step},{recon.item():.6f},{frac:.4f}\n")
                rec_log.flush()
        if step % cfg.save_every == 0 or step == cfg.steps:
            ck = {"step": step, "model": model.state_dict(), "opt": opt.state_dict()}
            if head is not None:
                ck["bird_head"] = head.state_dict()
            if decoder is not None:
                ck["acc_decoder"] = decoder.state_dict()
            torch.save(ck, cfg.run / f"ckpt_{step:07d}.pt")
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    log.close()
    if adv_log is not None:
        adv_log.close()
    if rec_log is not None:
        rec_log.close()
    del A, AM, G, GM
    embed(model, data, dev, cfg.run / "embeddings")
    print("embeddings written")


if __name__ == "__main__":
    main()
