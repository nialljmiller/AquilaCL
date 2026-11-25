"""Two-branch encoder for 10 s windows.

acceleration (200 x 3 at 20 Hz, + mask) -> 1-D residual CNN -> masked mean/max pooling
GPS (10 s x C kinematics, + mask)       -> per-second linear embedding + small transformer -> masked mean
A sensor that is absent from the window is replaced by a learned "missing" vector, so the model never sees
zero-filled inputs posing as measurements.
fusion MLP -> h (representation); projection MLP -> z (used only by the contrastive loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2)
        self.b1 = nn.BatchNorm1d(c_out)
        self.c2 = nn.Conv1d(c_out, c_out, k, padding=k // 2)
        self.b2 = nn.BatchNorm1d(c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1, stride=stride) if (c_in != c_out or stride != 1) else nn.Identity()

    def forward(self, x):
        y = F.gelu(self.b1(self.c1(x)))
        return F.gelu(self.b2(self.c2(y)) + self.skip(x))


class AccEncoder(nn.Module):
    def __init__(self, width: int = 64, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            ResBlock(4, width), ResBlock(width, width), ResBlock(width, 2 * width, stride=2),
            ResBlock(2 * width, 2 * width), ResBlock(2 * width, 2 * width, stride=2),
        )
        self.out = nn.Linear(4 * width, out)

    def forward(self, acc, mask):  # acc (B, T, 3), mask (B, T)
        x = torch.cat([acc, mask[..., None].to(acc.dtype)], dim=-1).transpose(1, 2)
        y = self.net(x)  # (B, C, T/4)
        m = F.max_pool1d(mask[:, None].float(), 4, 4, ceil_mode=True)[:, :, : y.shape[-1]]
        n = m.sum(-1).clamp(min=1)
        mean = (y * m).sum(-1) / n
        mx = y.masked_fill(m == 0, -1e4).amax(-1)
        return self.out(torch.cat([mean, mx], dim=-1))


class GpsEncoder(nn.Module):
    def __init__(self, channels: int, d: int = 64, layers: int = 2, out: int = 64, seconds: int = 10):
        super().__init__()
        self.inp = nn.Linear(2 * channels, d)
        self.pos = nn.Parameter(torch.zeros(seconds, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, out)

    def forward(self, gps, mask):  # (B, S, C), (B, S, C)
        present = mask.any(-1)  # (B, S)
        x = self.inp(torch.cat([gps, mask.to(gps.dtype)], dim=-1)) + self.pos
        safe = present.clone()
        safe[~present.any(-1), 0] = True  # avoid all-masked rows in attention
        y = self.tf(x, src_key_padding_mask=~safe)
        p = present[..., None].to(y.dtype)
        return self.out((y * p).sum(1) / p.sum(1).clamp(min=1))


class Encoder(nn.Module):
    def __init__(self, gps_channels: int, acc_dim: int = 128, gps_dim: int = 64, h_dim: int = 128, z_dim: int = 64):
        super().__init__()
        self.acc = AccEncoder(out=acc_dim)
        self.gps = GpsEncoder(gps_channels, out=gps_dim)
        self.acc_missing = nn.Parameter(torch.zeros(acc_dim))
        self.gps_missing = nn.Parameter(torch.zeros(gps_dim))
        self.fuse = nn.Sequential(nn.Linear(acc_dim + gps_dim, 256), nn.GELU(), nn.Linear(256, h_dim))
        self.proj = nn.Sequential(nn.Linear(h_dim, h_dim), nn.BatchNorm1d(h_dim), nn.GELU(), nn.Linear(h_dim, z_dim))

    def forward(self, acc, acc_mask, gps, gps_mask):
        has_acc = acc_mask.any(-1, keepdim=True)
        has_gps = gps_mask.flatten(1).any(-1, keepdim=True)
        a = torch.where(has_acc, self.acc(acc, acc_mask), self.acc_missing.to(acc.dtype))
        g = torch.where(has_gps, self.gps(gps, gps_mask), self.gps_missing.to(gps.dtype))
        h = self.fuse(torch.cat([a, g], dim=-1))
        return h, self.proj(h)


def nt_xent(z0: torch.Tensor, z1: torch.Tensor, tau: float, extra_mask: torch.Tensor | None = None) -> torch.Tensor:
    """NT-Xent over 2B views, fully vectorised.

    ``extra_mask`` (B, B) bool marks pairs (i, j) that must NOT be used as negatives (likely the same
    behaviour: same bird and segment within a few windows); it is applied to all four view combinations.
    """
    B = z0.shape[0]
    z = F.normalize(torch.cat([z0, z1]).float(), dim=1)  # call outside autocast: similarities in float32
    sim = (z @ z.T).float() / tau
    eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    if extra_mask is not None:
        m = extra_mask & ~torch.eye(B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(m.repeat(2, 2), float("-inf"))
    target = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, target)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coeff * g, None


def grad_reverse(x: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    """Identity in the forward pass; multiplies the gradient by -coeff in the backward pass (DANN)."""
    return _GradReverse.apply(x, coeff)


class BirdHead(nn.Module):
    """Adversary: MLP on h predicting which training bird a window comes from."""

    def __init__(self, h_dim: int, n_birds: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h_dim, hidden), nn.GELU(), nn.Linear(hidden, n_birds))

    def forward(self, h):
        return self.net(h)
