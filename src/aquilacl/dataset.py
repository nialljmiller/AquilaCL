"""Stage 4 data: load all 10 s windows into memory, normalise orientation and GPS scales, split birds.

Orientation: the tag sits at a different angle on every bird, which leaked into the baseline's posture
features. For each bird we take the mean gravity direction of quiet night windows (roosting) and rotate that
bird's acceleration so this direction points along +z. Rotation about the vertical (mounting yaw) cannot be
recovered this way; small random rotations during training cover the remainder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .windows import GPS_CHANNELS

# Fixed (not per-window) scaling of GPS channels, so absolute speed, climb and turn rate stay visible.
GPS_SCALE = {
    "ground-speed": 10.0,
    "speed_track_ms": 10.0,
    "climb_ms": 2.0,
    "turn_rate_dps": 30.0,
    "step_m": 10.0,
    "eobs:horizontal-accuracy-estimate": 10.0,
}
GPS_CLIP = 5.0


def rotation_to_z(v: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector direction ``v`` onto +z (Rodrigues)."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(v @ z)
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    k = np.cross(v, z)
    s = np.linalg.norm(k)
    k /= s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def roost_gravity(acc: np.ndarray, mask: np.ndarray, sun: np.ndarray, vedba: np.ndarray, max_vedba: float = 0.02):
    """Mean gravity direction over quiet night windows; None if there are fewer than 20."""
    sel = (sun < -6) & (vedba < max_vedba) & (mask.sum(axis=1) >= 20)
    if sel.sum() < 20:
        return None, int(sel.sum())
    a = acc[sel].astype(np.float32)
    m = mask[sel][:, :, None]
    mean = (a * m).sum(axis=1) / m.sum(axis=1)
    u = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    return np.median(u, axis=0), int(sel.sum())


def holdout_birds(birds: list[str], cohort: dict[str, str], per_cohort: int = 6, salt: str = "aquilacl") -> list[str]:
    """Deterministic held-out birds: the ``per_cohort`` lowest hashes within each cohort."""
    out = []
    for c in sorted(set(cohort.values())):
        bs = sorted((b for b in birds if cohort[b] == c), key=lambda b: hashlib.sha256((salt + b).encode()).hexdigest())
        out += bs[:per_cohort]
    return sorted(out)


@dataclass
class WindowData:
    acc: np.ndarray  # (N, 200, 3) float16, oriented, g
    acc_mask: np.ndarray  # (N, 200) bool
    gps: np.ndarray  # (N, 10, C) float16, scaled and clipped
    gps_mask: np.ndarray  # (N, 10, C) bool
    meta: pd.DataFrame  # one row per window (+ features and labels)
    orientation: pd.DataFrame  # per bird: roost direction, windows used, tilt corrected


def load(windows: Path, birds: list[str] | None = None) -> WindowData:
    dirs = sorted(p for p in windows.iterdir() if p.is_dir())
    if birds is not None:
        dirs = [d for d in dirs if d.name in set(birds)]
    accs, amasks, gpss, gmasks, metas, orient = [], [], [], [], [], []
    scale = np.array([GPS_SCALE[c] for c in GPS_CHANNELS], dtype=np.float32)
    for d in dirs:
        m = pd.read_parquet(d / "meta.parquet")
        f = pd.read_parquet(d / "features.parquet")
        acc = np.load(d / "acc.npy")
        am = np.load(d / "acc_mask.npy")
        v, n_used = roost_gravity(acc, am, m["sun_elev_deg"].to_numpy(), f["vedba"].to_numpy())
        if v is not None:
            R = rotation_to_z(v).astype(np.float32)
            acc = (acc.astype(np.float32) @ R.T).astype(np.float16) * am[:, :, None]
            tilt = float(np.degrees(np.arccos(np.clip(v[2] / np.linalg.norm(v), -1, 1))))
        else:
            tilt = float("nan")
        orient.append({"bird": d.name, "roost_windows": n_used, "tilt_corrected_deg": tilt})
        g = np.load(d / "gps.npy") / scale
        g[:, :, GPS_CHANNELS.index("eobs:horizontal-accuracy-estimate")] = np.log1p(
            np.load(d / "gps.npy")[:, :, GPS_CHANNELS.index("eobs:horizontal-accuracy-estimate")]
        )
        gm = np.load(d / "gps_mask.npy")
        g = np.clip(g, -GPS_CLIP, GPS_CLIP) * gm
        accs.append(acc.astype(np.float16))
        amasks.append(am)
        gpss.append(g.astype(np.float16))
        gmasks.append(gm)
        metas.append(pd.concat([m, f], axis=1))
    meta = pd.concat(metas, ignore_index=True)
    first = meta.groupby("bird")["t_start"].transform("min")
    meta["cohort"] = np.where(first.dt.year <= 2024, "2024", "2025")
    return WindowData(
        np.concatenate(accs), np.concatenate(amasks), np.concatenate(gpss), np.concatenate(gmasks), meta, pd.DataFrame(orient)
    )
