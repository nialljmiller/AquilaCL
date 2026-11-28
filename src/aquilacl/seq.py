"""Level-2 model: a small transformer over the time-ordered clip embeddings h of a level-1 run.

Usage:
    python -m aquilacl.seq --parent <runs>/<level-1 run> --windows <data>/windows_v4 --split <split.json> \
        --run <runs>/<new run> [options]

Why: at the 10 s clip scale flight is a continuum; behaviour types, daily rhythm, seasonal pattern and odd days
live at the scale of minutes to a day. This model reads the frozen level-1 h of each bird in time order and learns
context by masked token modelling.

Input. ``<parent>/embeddings/<bird>.npz`` key 'h' (N x D, rows aligned with ``<windows>/<bird>/meta.parquet``,
which is in time order). Only meta columns t_start, sensitivity_k, acc_samples and local_solar_hour are read.
Birds with empty meta are skipped. The parent must be a level-1 run without time input (design.json absent, or
level 1 and uses_time false) whose holdout.json equals split['search'] (and locked.json, when present,
split['locked']); otherwise the run is refused, because a parent trained on search or locked birds would
contaminate level 2.

Tokens. token = Linear(standardised h) + recording-format embedding (4 formats: scheduled/continuous x short/full
window, full = acc_samples >= FULL_SAMPLES) [+ Linear(time features)] [+ Linear(env features)]. Masked tokens
replace the h part by a learned mask vector and keep the format, time and env parts. Attention uses rotary
position encoding (RoPE) only: attention scores depend on the offset between two tokens, never on a token's
index, so no absolute position enters (tested: left-padding a crop does not change its outputs).

Sequences.
- Stream (all windows of one bird in time order, scheduled and continuous, across days and nights). Training draws
  fixed-length random crops (--max-len tokens) of the train birds' streams (bird chosen uniformly). Window
  embeddings 'h' come from overlapping tiles of the stream (length L = --max-len, stride L/2) whose first start is
  offset by a number drawn from (seed, crc32 of the bird id), never from time; each window takes the output of the
  tile whose centre is nearest. Deviation from day-bounded sequences (reason): the median bird-day has 164 windows,
  so a 512-token crop of a day is usually the whole day and starts at the first recording of the day (the tags
  record from about 05:00 local time, on a fixed clock); with scheduled bursts on a 300 s clock the distance from the
  crop's left edge would then give the hour even with relative-only position encoding. Stream tiles have edges at
  arbitrary times of day, so 'h' carries no absolute position. The bird-day is the pooling unit (day_embeddings),
  not an attention boundary.
- Bouts (flight.bout_ids with its defaults: continuous-mode windows only). Training mixes in random crops of bouts
  (--p-bout of the sequences; bout chosen with probability proportional to its length, bouts of >= 2 windows).
  bout_embeddings come from a separate pass that runs each bout as its own sequence (bouts longer than L, up to
  2,021 windows on windows_v4, are cut into near-equal consecutive chunks) and mean-pools it. The bout edge (bout
  onset) is visible to the model; it is a behaviour event (speed trigger), not a clock position.
- Every window is in its bird's stream, so every window gets a transformer output.

Time encoding (--time-encoding):
- none (default): no time input at all (t_start is used only to order and group, and for bout_ids/local days).
- relative: per token log1p(gap to the previous token of the sequence in s) / log1p(86400); first token 0.
- absolute: sin/cos of local solar hour and of local day of year (comparison twin only).
With relative or absolute, the time features of a sequence are zeroed with probability --time-dropout during
training, and 'h_notime' is written: the same tiles with the time features zeroed (the in-distribution "no time"
case). design.json uses_time = (time-encoding != none).

Environment (--env DIR, default off): DIR/<bird>.parquet with one row per meta row; every numeric column is
standardised with train-bird mean and sd (NaN -> 0) and projected. Without --env nothing is read from any env
location. design.json uses_env = bool(--env) or the parent's uses_env.

Objective: span masking of --mask-min..--mask-max of each sequence's tokens (spans of --mask-span tokens, at least
one token masked and one kept), predicting the standardised frozen input h (train-bird mean/sd) of the masked
tokens (smooth-L1, or 1 - cosine). Optional --contrast W adds NT-Xent between the mean-pooled outputs of two
overlapping crops of the same sequence. Off by default. WARNING: the two crops come from the same bird and usually
the same day, so this term rewards what a bird's windows share; the r7 lesson (neighbour positives) was that
same-bird positives raise bird identity (G9).

Training uses split['train'] birds only (crops, standardisation stats); search and locked birds are embedded only.

Outputs in --run (a new directory, or one this module created; re-running resumes from the latest checkpoint):
config.json, design.json {uses_time, uses_env, level: 2, parent_run}, seq_info.json (counts, tiling), holdout.json
and locked.json (from the split), orientation.json/csv (copied from the parent, whose frame h is in), loss.csv (appended), ckpt_<step>.pt, embeddings/<bird>.npz (h [, h_notime], row;
float16; output dim --out-dim, default the parent's D), day_embeddings/<bird>.npz (day = local solar day number
from anomaly.local_days, n, h [, h_notime]: mean of the day's window h), bout_embeddings/<bird>.npz (bout, first_row,
n, h [, h_notime]).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anomaly import local_days, solar_offset
from .flight import bout_ids
from .model import nt_xent

FULL_SAMPLES = 190
META_COLS = ["t_start", "sensitivity_k", "acc_samples", "local_solar_hour"]
N_TIME = {"none": 0, "relative": 1, "absolute": 4}
RUN_CONTROL = {"run", "steps", "embed_only", "device", "save_every", "log_every", "infer_batch", "embed_input",
               "embed_out"}


# --------------------------------------------------------------------------- model
def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary encoding of x (B, H, L, dh) at integer positions pos (B, L); dh even."""
    dh = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh))
    ang = pos[:, None, :, None].float() * inv  # (B, 1, L, dh/2)
    c, s = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x, pos, keymask):
        B, L, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keymask[:, None, None, :],
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.mlp(self.n2(x)))


class SeqModel(nn.Module):
    def __init__(self, d_in: int, out_dim: int, d_model: int = 256, layers: int = 4, heads: int = 4,
                 ff: int = 1024, dropout: float = 0.1, n_time: int = 0, n_env: int = 0):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        self.mask_tok = nn.Parameter(0.02 * torch.randn(d_model))
        self.fmt = nn.Embedding(4, d_model)
        self.time = nn.Linear(n_time, d_model) if n_time else None
        self.env = nn.Linear(n_env, d_model) if n_env else None
        self.blocks = nn.ModuleList(Block(d_model, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)
        self.pred = nn.Sequential(nn.Linear(out_dim, d_model), nn.GELU(), nn.Linear(d_model, d_in))

    def forward(self, x, fmt, valid, masked=None, tfeat=None, efeat=None):
        """x (B, L, d_in) standardised h; fmt (B, L) long; valid (B, L) bool; masked (B, L) bool or None.
        Returns the contextual embedding (B, L, out_dim)."""
        e = self.inp(x)
        if masked is not None:
            e = torch.where(masked[..., None], self.mask_tok.to(e.dtype), e)
        e = e + self.fmt(fmt)
        if self.time is not None and tfeat is not None:
            e = e + self.time(tfeat)
        if self.env is not None and efeat is not None:
            e = e + self.env(efeat)
        pos = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        for blk in self.blocks:
            e = blk(e, pos, valid)
        return self.out(self.norm(e))


# --------------------------------------------------------------------------- data
def window_format(k: np.ndarray, acc_samples: np.ndarray) -> np.ndarray:
    """0 scheduled short, 1 scheduled full, 2 continuous short, 3 continuous full."""
    return (2 * (np.asarray(k) == 1) + (np.asarray(acc_samples) >= FULL_SAMPLES)).astype(np.int64)


def absolute_features(t: pd.Series, solar_hour: np.ndarray) -> np.ndarray:
    """(N, 4) sin/cos of local solar hour and local day of year (absolute mode only)."""
    t = pd.to_datetime(pd.Series(t)).reset_index(drop=True)
    off, _ = solar_offset(t, solar_hour)
    loc = t.dt.tz_localize(None) if t.dt.tz is not None else t
    loc = loc + pd.to_timedelta(off, unit="h")
    hr = (loc.dt.hour + loc.dt.minute / 60 + loc.dt.second / 3600).to_numpy(float)
    doy = loc.dt.dayofyear.to_numpy(float) - 1 + hr / 24
    a, b = 2 * np.pi * hr / 24, 2 * np.pi * doy / 365.25
    return np.stack([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], 1).astype(np.float32)


def load_streams(parent: Path, windows: Path, time_encoding: str, env: Path | None = None) -> dict:
    """Streams of every bird with windows, h from ``parent``/embeddings. Reads env only when ``env`` is given."""
    H, M = {}, {}
    for d in sorted(p for p in windows.iterdir() if p.is_dir()):
        m = pd.read_parquet(d / "meta.parquet", columns=META_COLS)
        if len(m):
            M[d.name], H[d.name] = m, np.load(parent / "embeddings" / f"{d.name}.npz")["h"]
    return build_streams(H, M, time_encoding, env)


def build_streams(h_by_bird: dict, meta_by_bird: dict, time_encoding: str, env: Path | None = None) -> dict:
    """Concatenated per-bird streams (birds sorted, rows in meta order; birds with empty meta skipped)."""
    birds, offs, H, FMT, T, ABS, E, DAY, BOUTS = [], [0], [], [], [], [], [], [], []
    env_cols = None
    for b in sorted(meta_by_bird):
        m = meta_by_bird[b]
        if len(m) == 0:
            continue
        if not m["t_start"].is_monotonic_increasing:
            raise SystemExit(f"{b}: meta not in time order")
        h = np.asarray(h_by_bird[b], np.float32)
        if len(h) != len(m):
            raise SystemExit(f"{b}: {len(m)} windows but {len(h)} input embeddings")
        ts = pd.to_datetime(m["t_start"]).reset_index(drop=True)
        tsec = ((ts - pd.Timestamp("1970-01-01", tz=ts.dt.tz)) / pd.Timedelta(seconds=1)).to_numpy(np.float64)
        k = m["sensitivity_k"].to_numpy()
        birds.append(b)
        offs.append(offs[-1] + len(m))
        H.append(h)
        FMT.append(window_format(k, m["acc_samples"].to_numpy()))
        T.append(tsec)
        DAY.append(local_days(ts, m["local_solar_hour"].to_numpy(float))[0])
        BOUTS.append(bout_ids(tsec - tsec[0], k == 1))
        if time_encoding == "absolute":
            ABS.append(absolute_features(ts, m["local_solar_hour"].to_numpy(float)))
        if env is not None:
            e = pd.read_parquet(Path(env) / f"{b}.parquet")
            if len(e) != len(m):
                raise SystemExit(f"env {b}: {len(e)} rows but {len(m)} windows")
            e = e.select_dtypes("number")
            if env_cols is None:
                env_cols = list(e.columns)
            elif list(e.columns) != env_cols:
                raise SystemExit(f"env {b}: columns differ from the first bird's")
            E.append(e.to_numpy(np.float32))
    return {"birds": birds, "offs": np.array(offs), "H": np.concatenate(H), "fmt": np.concatenate(FMT),
            "t": np.concatenate(T), "day": DAY, "bout": BOUTS,
            "abs": np.concatenate(ABS) if ABS else None, "env": np.concatenate(E) if E else None, "env_cols": env_cols}


def bout_lists(S: dict, birds: set | None = None) -> list[tuple[int, np.ndarray]]:
    """(bird index, global row indices) of every bout, optionally of the given birds only."""
    out = []
    for i, b in enumerate(S["birds"]):
        if birds is not None and b not in birds:
            continue
        ids = S["bout"][i]
        rows = np.flatnonzero(ids >= 0)
        if not len(rows):
            continue
        cut = np.flatnonzero(np.diff(ids[rows])) + 1
        out += [(i, S["offs"][i] + r) for r in np.split(rows, cut)]
    return out


def stream_tiles(n: int, L: int, offset: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Tiles [s, e) of a stream of n windows (length L, stride L//2, first start offset - stride, clipped to
    [0, n)) and the tile each window takes its output from (nearest centre; it always lies inside that tile)."""
    S = max(1, L // 2)
    s0 = offset % S - S
    starts = [s0]
    while starts[-1] + L < n:
        starts.append(starts[-1] + S)
    tiles = [(max(0, s), min(n, s + L)) for s in starts]
    i = np.arange(n)
    which = np.clip(np.round((i - s0 - L / 2) / S), 0, len(starts) - 1).astype(np.int64)
    return tiles, which


def tile_offset(seed: int, bird: str, L: int) -> int:
    return int(np.random.default_rng([seed, zlib.crc32(bird.encode())]).integers(0, max(1, L // 2)))


# --------------------------------------------------------------------------- batches
class Batcher:
    """Builds model inputs on the device from (B, L) global row indices (-1 = padding)."""

    def __init__(self, S: dict, mean: np.ndarray, sd: np.ndarray, time_encoding: str, env_mean=None, env_sd=None,
                 dev="cpu"):
        self.dev, self.te = dev, time_encoding
        self.H = torch.from_numpy(((S["H"] - mean) / sd).astype(np.float32)).to(dev)
        self.fmt = torch.from_numpy(S["fmt"]).to(dev)
        self.t = torch.from_numpy(S["t"]).to(dev) if time_encoding == "relative" else None
        self.abs = torch.from_numpy(S["abs"]).to(dev) if time_encoding == "absolute" else None
        self.env = None
        if S["env"] is not None:
            e = (S["env"] - env_mean) / env_sd
            self.env = torch.from_numpy(np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)).to(dev)

    def __call__(self, idx: torch.Tensor, notime: torch.Tensor | None = None):
        valid = idx >= 0
        s = idx.clamp(min=0)
        vf = valid[..., None].float()
        x = self.H[s] * vf
        fmt = self.fmt[s] * valid
        tf = None
        if self.te == "relative":
            t = self.t[s]
            gap = torch.zeros_like(t)
            gap[:, 1:] = torch.where(valid[:, 1:] & valid[:, :-1], (t[:, 1:] - t[:, :-1]).clamp(min=0), 0.0)
            tf = (torch.log1p(gap) / math.log1p(86400.0)).float()[..., None] * vf
        elif self.te == "absolute":
            tf = self.abs[s] * vf
        if tf is not None and notime is not None:
            tf = tf * (~notime)[:, None, None].float()
        ef = self.env[s] * vf if self.env is not None else None
        return x, fmt, valid, tf, ef


def pad_lists(lists: list[np.ndarray], L: int) -> np.ndarray:
    out = np.full((len(lists), L), -1, np.int64)
    for j, r in enumerate(lists):
        out[j, :len(r)] = r
    return out


def span_mask(valid: np.ndarray, rng, lo: float, hi: float, span: int) -> np.ndarray:
    """(B, L) bool: spans covering about U(lo, hi) of each row's valid (left-aligned) tokens; >= 1 masked and,
    when the row has >= 2 tokens, >= 1 kept."""
    B, L = valid.shape
    out = np.zeros((B, L), bool)
    for j in range(B):
        n = int(valid[j].sum())
        if n == 0:
            continue
        target = max(1, int(round(rng.uniform(lo, hi) * n)))
        sp = min(span, n)
        for s in rng.integers(0, n - sp + 1, int(math.ceil(target / sp))):
            out[j, s:s + sp] = True
        if n >= 2 and out[j, :n].all():
            out[j, rng.integers(0, n)] = False
    return out


class CropSampler:
    """Random training crops: stream crops of train birds (bird uniform) and bout crops (bout prob ~ length)."""

    def __init__(self, S: dict, train: set, L: int, p_bout: float, seed: int):
        self.L, self.p_bout = L, p_bout
        self.rng = np.random.default_rng(seed)
        self.streams = [(S["offs"][i], S["offs"][i + 1] - S["offs"][i]) for i, b in enumerate(S["birds"]) if b in train]
        self.bouts = [r for _, r in bout_lists(S, train) if len(r) >= 2]
        w = np.array([len(r) for r in self.bouts], float)
        self.bp = w / w.sum() if len(w) else None
        if not self.streams:
            raise SystemExit("no training birds with windows")

    def crop(self):
        """One crop (global rows) and a function giving an overlapping second crop of the same sequence."""
        L, rng = self.L, self.rng
        if self.bp is not None and rng.random() < self.p_bout:
            rows = self.bouts[rng.choice(len(self.bouts), p=self.bp)]
            o, n = None, len(rows)
        else:
            o, n = self.streams[rng.integers(len(self.streams))]
        ln = min(L, n)
        start = int(rng.integers(0, n - ln + 1))

        def take(st):
            st = int(np.clip(st, 0, n - ln))
            return rows[st:st + ln] if o is None else np.arange(o + st, o + st + ln)
        return take(start), lambda: take(start + int(rng.integers(-(ln // 2), ln // 2 + 1)))


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def run_lists(model, batcher, lists: list[np.ndarray], L: int, dev: str, notime: bool, batch: int) -> list[np.ndarray]:
    """Model outputs (float32) for each list of global rows (each of length <= L)."""
    model.eval()
    out = []
    for s in range(0, len(lists), batch):
        chunk = lists[s:s + batch]
        idx = torch.from_numpy(pad_lists(chunk, L)).to(dev)
        nt = torch.full((len(chunk),), notime, dtype=torch.bool, device=dev)
        x, fmt, valid, tf, ef = batcher(idx, nt)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            y = model(x, fmt, valid, None, tf, ef)
        y = y.float().cpu().numpy()
        out += [y[j, :len(r)] for j, r in enumerate(chunk)]
    model.train()
    return out


def window_h(model, batcher, S: dict, i: int, L: int, seed: int, notime: bool, dev: str, batch: int) -> np.ndarray:
    """Contextual h (float32, n x out) of bird i of S from its stream tiles (nearest-centre tile per window)."""
    o, n = S["offs"][i], S["offs"][i + 1] - S["offs"][i]
    tiles, which = stream_tiles(n, L, tile_offset(seed, S["birds"][i], L))
    ys = run_lists(model, batcher, [np.arange(o + s, o + e) for s, e in tiles], L, dev, notime, batch)
    h = np.empty((n, ys[0].shape[1]), np.float32)
    for t_i, (s, e) in enumerate(tiles):
        pos = np.flatnonzero(which[s:e] == t_i)
        h[s + pos] = ys[t_i][pos]
    return h


def embed_all(model, batcher, S: dict, cfg, uses_time: bool, out_dim: int, run: Path) -> dict:
    L, dev = cfg.max_len, cfg.device
    for sub in ("embeddings", "day_embeddings", "bout_embeddings"):
        (run / sub).mkdir(exist_ok=True)
    modes = [False, True] if uses_time else [False]
    info = {"tiles": 0, "bout_chunks": 0}
    all_bouts = bout_lists(S)
    for i, b in enumerate(S["birds"]):
        o, n = S["offs"][i], S["offs"][i + 1] - S["offs"][i]
        info["tiles"] += len(stream_tiles(n, L, tile_offset(cfg.seed, b, L))[0])
        win, day, bout = {"row": np.arange(n)}, {}, {}
        bl = [r for j, r in all_bouts if j == i]
        chunks, owner = [], []
        for q, r in enumerate(bl):
            for c in np.array_split(r, int(math.ceil(len(r) / L))):
                chunks.append(c)
                owner.append(q)
        info["bout_chunks"] += len(chunks)
        owner = np.array(owner, np.int64)
        d_key = S["day"][i]
        ud, d_inv = np.unique(d_key, return_inverse=True)
        day["day"], day["n"] = ud, np.bincount(d_inv)
        bout["bout"] = S["bout"][i][[r[0] - o for r in bl]] if bl else np.zeros(0, np.int64)
        bout["first_row"] = np.array([r[0] - o for r in bl], np.int64)
        bout["n"] = np.array([len(r) for r in bl], np.int64)
        for nt in modes:
            key = "h_notime" if nt else "h"
            h = window_h(model, batcher, S, i, L, cfg.seed, nt, dev, cfg.infer_batch)
            win[key] = h.astype(np.float16)
            sums = np.zeros((len(ud), out_dim))
            np.add.at(sums, d_inv, h)
            day[key] = (sums / day["n"][:, None]).astype(np.float16)
            if chunks:
                yb = run_lists(model, batcher, chunks, L, dev, nt, cfg.infer_batch)
                sb = np.zeros((len(bl), out_dim))
                np.add.at(sb, owner, np.stack([y.sum(0) for y in yb]))
                bout[key] = (sb / bout["n"][:, None]).astype(np.float16)
            else:
                bout[key] = np.zeros((0, out_dim), np.float16)
        np.savez(run / "embeddings" / f"{b}.npz", **win)
        np.savez(run / "day_embeddings" / f"{b}.npz", **day)
        np.savez(run / "bout_embeddings" / f"{b}.npz", **bout)
    return info


def contextual_embed(run_dir, h_by_bird: dict[str, np.ndarray], meta_by_bird: dict[str, pd.DataFrame],
                     device: str = "cpu", notime: bool = False) -> dict[str, np.ndarray]:
    """Contextual h (float32, N x out_dim per bird) of a trained seq run for arbitrary input parent-h arrays.

    For scorecard rows that change the input (GPS-masked h, injected bouts or days). Uses the run's latest
    checkpoint and config.json: the same standardisation (train-bird mean/sd stored in the checkpoint), stream
    tiling (offset from the run's seed and the bird id), format embeddings and time features as the run's own
    embedding pass, which it reproduces exactly for the same input. ``meta_by_bird[b]`` needs META_COLS, rows
    aligned with ``h_by_bird[b]``, in time order. ``notime=True`` gives the h_notime path (time features zeroed;
    identical to h for a run without time encoding). A run with --env reads env features from its configured env
    directory. No settings check against a parent; nothing is written into run_dir.
    """
    run_dir = Path(run_dir)
    ckpts = sorted(run_dir.glob("ckpt_*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoint in {run_dir}")
    c = json.loads((run_dir / "config.json").read_text())
    ck = torch.load(ckpts[-1], map_location="cpu")  # standardisation tensors are read with .numpy()
    env = c.get("env") if c.get("env") not in (None, "None") else None
    te = c["time_encoding"]
    S = build_streams(h_by_bird, meta_by_bird, te, env)
    mean, sd = ck["mean"].numpy(), ck["sd"].numpy()
    em = ck["env_mean"].numpy() if "env_mean" in ck else None
    es = ck["env_sd"].numpy() if "env_sd" in ck else None
    if env is not None and em is None:
        raise SystemExit(f"{ckpts[-1]}: env run without stored env standardisation")
    d_in = S["H"].shape[1]
    if d_in != len(mean):
        raise SystemExit(f"input D {d_in} differs from the run's {len(mean)}")
    model = SeqModel(d_in, c["out_dim"] or d_in, c["d_model"], c["layers"], c["heads"], c["ff"], c["dropout"],
                     N_TIME[te], 0 if S["env"] is None else S["env"].shape[1]).to(device)
    model.load_state_dict(ck["model"])
    batcher = Batcher(S, mean, sd, te, em, es, device)
    nt = notime and te != "none"
    return {b: window_h(model, batcher, S, i, c["max_len"], c["seed"], nt, device, c["infer_batch"])
            for i, b in enumerate(S["birds"])}


def embed_input(cfg) -> None:
    """--embed-input: contextual h of <embed-input>/embeddings/<bird>.npz 'h' with the trained --run, written to
    <embed-out>/embeddings/<bird>.npz (h [, h_notime], row; float16). Meta from --windows."""
    out = cfg.embed_out
    if out is None:
        raise SystemExit("--embed-input needs --embed-out")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: refusing")
    H, M = {}, {}
    for f in sorted((cfg.embed_input / "embeddings").glob("*.npz")):
        m = pd.read_parquet(cfg.windows / f.stem / "meta.parquet", columns=META_COLS)
        if len(m):
            H[f.stem], M[f.stem] = np.load(f)["h"], m
    uses_time = json.loads((cfg.run / "config.json").read_text())["time_encoding"] != "none"
    res = {"h": contextual_embed(cfg.run, H, M, cfg.device)}
    if uses_time:
        res["h_notime"] = contextual_embed(cfg.run, H, M, cfg.device, notime=True)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    for b in res["h"]:
        np.savez(out / "embeddings" / f"{b}.npz", row=np.arange(len(res["h"][b])),
                 **{k: v[b].astype(np.float16) for k, v in res.items()})
    print(f"embedded {len(res['h'])} birds into {out}")


# --------------------------------------------------------------------------- main
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parent", type=Path, default=None, help="level-1 run directory (embeddings/<bird>.npz 'h'); "
                    "required except with --embed-input")
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=None, help="scorecard split.json {train, search, locked}; "
                    "required except with --embed-input")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--time-encoding", choices=list(N_TIME), default="none")
    ap.add_argument("--time-dropout", type=float, default=0.5, help="share of training sequences with time zeroed")
    ap.add_argument("--env", type=Path, default=None, help="directory of per-bird env feature parquets (off)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--p-bout", type=float, default=0.3, help="share of training sequences that are bout crops")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out-dim", type=int, default=0, help="output embedding size (0 = the parent's D)")
    ap.add_argument("--mask-min", type=float, default=0.15)
    ap.add_argument("--mask-max", type=float, default=0.30)
    ap.add_argument("--mask-span", type=int, default=3)
    ap.add_argument("--loss", choices=["smoothl1", "cosine"], default="smoothl1")
    ap.add_argument("--contrast", type=float, default=0.0, help="weight of the two-crop NT-Xent term (0 = off; "
                    "raises bird identity, see the module docstring)")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64, help="sequences per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--infer-batch", type=int, default=128, help="sequences per inference batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--embed-only", action="store_true", help="skip training; embed with the latest checkpoint")
    ap.add_argument("--embed-input", type=Path, default=None,
                    help="embed only: contextual h of DIR/embeddings/<bird>.npz 'h' with the trained --run (meta from "
                         "--windows); written to --embed-out, nothing written into --run")
    ap.add_argument("--embed-out", type=Path, default=None, help="output directory for --embed-input (new or empty)")
    return ap


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def record_config(cfg) -> None:
    cur = {k: (v if isinstance(v, (int, float, list, bool)) or v is None else str(v)) for k, v in vars(cfg).items()}
    path = cfg.run / "config.json"
    if path.exists() and any(cfg.run.glob("ckpt_*.pt")):
        old = json.loads(path.read_text())
        diff = {k: (old.get(k), v) for k, v in cur.items() if k not in RUN_CONTROL and str(old.get(k)) != str(v)}
        if diff:
            raise SystemExit(f"{cfg.run} was trained with other settings (stored, given): {diff}")
        return
    path.write_text(json.dumps(cur, indent=1))


def check_parent(cfg, sp: dict) -> bool:
    """Refuse unsuitable parents; returns the parent's uses_env."""
    if cfg.run.resolve() == cfg.parent.resolve():
        raise SystemExit("--run must differ from --parent")
    des = json.loads((cfg.parent / "design.json").read_text()) if (cfg.parent / "design.json").exists() else {}
    if des.get("level", 1) != 1 or des.get("uses_time", False):
        raise SystemExit(f"{cfg.parent}: parent must be a level-1 run without time input ({des})")
    hp = cfg.parent / "holdout.json"
    if not hp.exists() or set(json.loads(hp.read_text())) != set(sp["search"]):
        raise SystemExit(f"{hp}: parent holdout differs from split['search'] (parent may have trained on search birds)")
    lp = cfg.parent / "locked.json"
    if lp.exists() and set(json.loads(lp.read_text())) != set(sp["locked"]):
        raise SystemExit(f"{lp}: parent locked set differs from split['locked']")
    if not lp.exists():
        print(f"warning: {cfg.parent} has no locked.json; cannot confirm the parent never trained on locked birds")
    return bool(des.get("uses_env", False))


def main(argv=None):
    cfg = parser().parse_args(argv)
    if cfg.embed_input is not None:
        return embed_input(cfg)
    if cfg.parent is None or cfg.split is None:
        raise SystemExit("--parent and --split are required for training")
    sp = json.loads(cfg.split.read_text())
    train_b, search, locked = set(sp["train"]), sorted(sp["search"]), sorted(sp["locked"])
    if train_b & (set(search) | set(locked)):
        raise SystemExit(f"{cfg.split}: train overlaps search or locked")
    parent_env = check_parent(cfg, sp)
    if cfg.run.exists() and any(cfg.run.iterdir()) and not (cfg.run / "config.json").exists():
        raise SystemExit(f"{cfg.run} exists, is not empty and is not a seq run: refusing")
    if cfg.embed_only and not any(cfg.run.glob("ckpt_*.pt")):
        raise SystemExit(f"--embed-only but no checkpoint in {cfg.run}")
    if cfg.contrast > 0:
        print("WARNING: --contrast uses two crops of the same bird sequence as positives; same-bird positives raised "
              "bird identity in r7 (neighbour positives).")
    cfg.run.mkdir(parents=True, exist_ok=True)
    record_config(cfg)
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    uses_time = cfg.time_encoding != "none"
    (cfg.run / "design.json").write_text(json.dumps({"uses_time": uses_time, "uses_env": cfg.env is not None or parent_env,
                                                     "level": 2, "parent_run": os.path.abspath(cfg.parent)}, indent=1))
    (cfg.run / "holdout.json").write_text(json.dumps(search, indent=1))
    (cfg.run / "locked.json").write_text(json.dumps(locked, indent=1))
    for f in ("orientation.json", "orientation.csv"):  # level-2 h lives in the parent's frame
        if (cfg.parent / f).exists():
            shutil.copyfile(cfg.parent / f, cfg.run / f)

    S = load_streams(cfg.parent, cfg.windows, cfg.time_encoding, cfg.env)
    tr_rows = np.concatenate([np.arange(S["offs"][i], S["offs"][i + 1]) for i, b in enumerate(S["birds"]) if b in train_b])
    d_in = S["H"].shape[1]
    out_dim = cfg.out_dim or d_in
    mean = S["H"][tr_rows].mean(0, dtype=np.float64).astype(np.float32)
    sd = np.maximum(S["H"][tr_rows].std(0, dtype=np.float64), 1e-6).astype(np.float32)
    env_mean = env_sd = None
    if S["env"] is not None:
        with warnings.catch_warnings():  # an all-NaN column gets mean 0, sd 1
            warnings.simplefilter("ignore", RuntimeWarning)
            env_mean = np.nan_to_num(np.nanmean(S["env"][tr_rows], 0)).astype(np.float32)
            env_sd = np.nan_to_num(np.nanstd(S["env"][tr_rows], 0), nan=1.0).astype(np.float32)
        env_sd = np.where(env_sd > 1e-6, env_sd, 1.0).astype(np.float32)
        (cfg.run / "env_columns.json").write_text(json.dumps(S["env_cols"]))
    batcher = Batcher(S, mean, sd, cfg.time_encoding, env_mean, env_sd, dev)
    model = SeqModel(d_in, out_dim, cfg.d_model, cfg.layers, cfg.heads, cfg.ff, cfg.dropout, N_TIME[cfg.time_encoding],
                     0 if S["env"] is None else S["env"].shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    step = 0
    ckpts = sorted(cfg.run.glob("ckpt_*.pt"))
    if ckpts:
        ck = torch.load(ckpts[-1], map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from {ckpts[-1].name} at step {step}")
    sampler = CropSampler(S, train_b, cfg.max_len, cfg.p_bout, cfg.seed + step)
    torch.manual_seed(cfg.seed + step)
    days = np.concatenate([np.bincount(np.unique(d, return_inverse=True)[1]) for d in S["day"]])
    info = {"birds": len(S["birds"]), "train_birds": int(sum(b in train_b for b in S["birds"])),
            "windows": int(len(S["H"])), "train_windows": int(len(tr_rows)), "d_in": int(d_in), "out_dim": int(out_dim),
            "max_len": cfg.max_len, "stride": max(1, cfg.max_len // 2), "train_bouts": len(sampler.bouts),
            "bird_days": int(len(days)), "bird_days_longer_than_max_len": int((days > cfg.max_len).sum()),
            "windows_in_days_longer_than_max_len": int(days[days > cfg.max_len].sum()),
            "env_columns": S["env_cols"],
            "params": int(sum(p.numel() for p in model.parameters()))}
    print(json.dumps(info))

    log = open(cfg.run / "loss.csv", "a")
    if log.tell() == 0:
        log.write("step,loss,mtm,contrast,lr,masked_frac,sec\n")
    t0 = time.time()
    while step < cfg.steps and not cfg.embed_only:
        crops = [sampler.crop() for _ in range(cfg.batch)]
        idx_np = pad_lists([c[0] for c in crops], cfg.max_len)
        m_np = span_mask(idx_np >= 0, sampler.rng, cfg.mask_min, cfg.mask_max, cfg.mask_span)
        idx = torch.from_numpy(idx_np).to(dev)
        masked = torch.from_numpy(m_np).to(dev)
        nt = (torch.rand(cfg.batch, device=dev) < cfg.time_dropout) if uses_time else None
        x, fmt, valid, tf, ef = batcher(idx, nt)
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, cfg)
        with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            y = model(x, fmt, valid, masked, tf, ef)
            pred = model.pred(y)
            if cfg.contrast > 0:
                idx2 = torch.from_numpy(pad_lists([c[1]() for c in crops], cfg.max_len)).to(dev)
                x2, f2, v2, t2, e2 = batcher(idx2, nt)
                m2 = torch.from_numpy(span_mask(idx2.cpu().numpy() >= 0, sampler.rng, cfg.mask_min, cfg.mask_max,
                                                cfg.mask_span)).to(dev)
                y2 = model(x2, f2, v2, m2, t2, e2)
        with torch.autocast(dev.split(":")[0], enabled=False):
            sel = masked & valid
            p, tgt = pred.float()[sel], x[sel]
            mtm = (F.smooth_l1_loss(p, tgt) if cfg.loss == "smoothl1"
                   else (1 - F.cosine_similarity(p, tgt, dim=1)).mean())
            loss, con = mtm, torch.zeros((), device=dev)
            if cfg.contrast > 0:
                pool = lambda yy, vv: (yy.float() * vv[..., None]).sum(1) / vv.sum(1, keepdim=True)  # noqa: E731
                con = nt_xent(pool(y, valid), pool(y2, v2), cfg.tau)
                loss = loss + cfg.contrast * con
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % cfg.log_every == 0:
            frac = (sel.sum() / valid.sum()).item()
            log.write(f"{step},{loss.item():.5f},{mtm.item():.5f},{con.item():.5f},{lr_at(step, cfg):.3e},{frac:.4f},"
                      f"{time.time() - t0:.1f}\n")
            log.flush()
        if step % cfg.save_every == 0 or step == cfg.steps:
            torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                        "mean": torch.from_numpy(mean), "sd": torch.from_numpy(sd),
                        **({} if env_mean is None else {"env_mean": torch.from_numpy(env_mean),
                                                        "env_sd": torch.from_numpy(env_sd)})}, cfg.run / f"ckpt_{step:07d}.pt")
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f} s)", flush=True)
    log.close()
    info.update(embed_all(model, batcher, S, cfg, uses_time, out_dim, cfg.run))
    info["trained_steps"] = step
    (cfg.run / "seq_info.json").write_text(json.dumps(info, indent=1))
    print("embeddings written")


if __name__ == "__main__":
    main()
