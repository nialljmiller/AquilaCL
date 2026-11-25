"""Stage 4 data: load all 10 s windows into memory, normalise orientation and GPS scales, split birds.

Orientation: the tag sits at a different angle on every bird, which leaked into the baseline's posture
features. For each bird we take the mean gravity direction of quiet night windows (roosting) and rotate that
bird's acceleration so this direction points along +z. Rotation about the vertical (mounting yaw) cannot be
recovered this way; small random rotations during training cover the remainder.

Optional yaw de-rotation (``derotate_yaw=True``; off by default, so earlier runs are unchanged): after the roost
rotation, each bird's mounting yaw is estimated from its flapping windows as the axial angle (mod 180 deg) of the
principal horizontal direction of dynamic acceleration (window mean removed, present samples only), averaged
over windows whose horizontal eigenvalue ratio is >= 2, and the bird is rotated about +z so that this axis lies
on +x. It uses no labels other than the ACC-derived ``label_flapping`` (itself computed from VeDBA and the
wingbeat band of the same acceleration), so it is input preprocessing like the roost rotation and is applied to
held-out birds in the same way. Forward and backward along the axis cannot be told apart (180 deg ambiguity).
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


def roost_gravity(acc: np.ndarray, mask: np.ndarray, sun: np.ndarray, vedba: np.ndarray, max_vedba: float = 0.02,
                  speed: np.ndarray | None = None):
    """Mean gravity direction over quiet night windows.

    If there are fewer than 20, quiet windows at any time of day are used instead (and GPS speed below 0.5 m/s
    where a fix exists). Returns (None, n) if there are still fewer than 20; the caller then uses the population
    median direction, so that no bird is left in its raw mounting frame.
    """
    quiet = (vedba < max_vedba) & (mask.sum(axis=1) >= 20)
    sel = quiet & (sun < -6)
    if sel.sum() < 20:
        sel = quiet & (~(speed >= 0.5) if speed is not None else True)
    if sel.sum() < 20:
        return None, int(sel.sum())
    a = acc[sel].astype(np.float32)
    m = mask[sel][:, :, None]
    mean = (a * m).sum(axis=1) / m.sum(axis=1)
    u = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    return np.median(u, axis=0), int(sel.sum())


YAW_MIN_SAMPLES = 40  # present samples a flapping window needs (as scripts/diagnose_r4.py)
YAW_MIN_RATIO = 2.0  # horizontal eigenvalue ratio l1/l2 a window needs
YAW_MIN_WINDOWS = 50  # qualifying windows a bird needs for a yaw estimate


def axial_angle(xy: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal direction of (N, T, 2) horizontal dynamic acceleration (present samples, window mean removed).

    Returns the axial angle in degrees in [0, 180) from +x towards +y, and the eigenvalue ratio l1/l2.
    """
    mf = m.astype(np.float32)[..., None]
    n = np.maximum(mf.sum(1), 1)
    d = (xy - (xy * mf).sum(1, keepdims=True) / n[:, None]) * mf
    cxx, cyy = (d[..., 0] ** 2).sum(1) / n[:, 0], (d[..., 1] ** 2).sum(1) / n[:, 0]
    cxy = (d[..., 0] * d[..., 1]).sum(1) / n[:, 0]
    theta = np.degrees(0.5 * np.arctan2(2 * cxy, cxx - cyy)) % 180.0
    r = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy**2)
    l1, l2 = (cxx + cyy) / 2 + r, (cxx + cyy) / 2 - r
    return theta, l1 / np.maximum(l2, 1e-12)


def axial_mean(theta_deg: np.ndarray) -> tuple[float, float]:
    """Circular mean of axial angles (deg, in (-90, 90]) and resultant length."""
    z = np.exp(2j * np.radians(np.asarray(theta_deg, float))).mean()
    mu = float(np.degrees(np.angle(z)) / 2)
    return (mu + 180.0 if mu <= -90.0 else mu), float(abs(z))


def rotation_z(deg: float) -> np.ndarray:
    """Rotation about +z by ``deg`` (counter-clockwise from +x towards +y)."""
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def estimate_yaw(acc: np.ndarray, mask: np.ndarray, min_windows: int = YAW_MIN_WINDOWS) -> dict:
    """Mounting yaw from flapping windows already rotated so that gravity is +z.

    ``acc`` (N, T, 3) and ``mask`` (N, T) hold the bird's flapping windows. Windows with >= YAW_MIN_SAMPLES
    present samples and eigenvalue ratio >= YAW_MIN_RATIO qualify; with at least ``min_windows`` of them the
    result is the axial mean angle (deg in (-90, 90], from +x towards +y), its resultant length and the count;
    otherwise ``yaw_deg`` is NaN and ``yaw_source`` is 'too_few_flapping'.
    """
    keep = mask.sum(1) >= YAW_MIN_SAMPLES
    n = 0
    if keep.any():
        th, ratio = axial_angle(np.asarray(acc[keep][..., :2], np.float32), mask[keep])
        good = ratio >= YAW_MIN_RATIO
        n = int(good.sum())
    if n < min_windows:
        return {"yaw_deg": float("nan"), "yaw_n": n, "yaw_resultant": float("nan"), "yaw_source": "too_few_flapping"}
    mu, res = axial_mean(th[good])
    return {"yaw_deg": mu, "yaw_n": n, "yaw_resultant": res, "yaw_source": "flapping"}


def common_format(acc: np.ndarray, mask: np.ndarray, step_g: float, clip_g: float) -> np.ndarray:
    """Put acceleration (sensor frame, g) on one grid and one range, whatever the tag's sensitivity setting.

    Tags record at 512 or 1024 counts per g. After calibration both are in g, but the 1024 setting has half
    the step (about 0.001 g against 0.002 g) and clips at about +-2.1 g instead of +-4 g; a probe reads the
    setting from the step alone with balanced accuracy 0.995. Rounding to the coarser step and clipping to the
    narrower range, per sensor axis and before any rotation, removes both. 0 disables either part.
    """
    a = acc.astype(np.float32)
    if step_g > 0:
        a = np.round(a / step_g) * step_g
    if clip_g > 0:
        a = np.clip(a, -clip_g, clip_g)
    return (a * mask[:, :, None]).astype(np.float16)


def holdout_birds(birds: list[str], cohort: dict[str, str], per_cohort: int = 6, salt: str = "aquilacl",
                  exclude=None) -> list[str]:
    """Deterministic held-out birds: the ``per_cohort`` lowest hashes of salt + bird within each cohort.

    Birds in ``exclude`` are never chosen (e.g. birds already used as held-out birds for model selection); they
    stay available for training. The defaults give the list of every run before this option existed.
    """
    ex = set(exclude or ())
    out = []
    for c in sorted(set(cohort.values())):
        bs = sorted((b for b in birds if cohort[b] == c and b not in ex), key=lambda b: hashlib.sha256((salt + b).encode()).hexdigest())
        out += bs[:per_cohort]
    return sorted(out)


def derotation_yaw(orient: list[dict], roost_v: list, accs: list, amasks: list, flaps: list) -> None:
    """Per-bird yaw (in place in ``orient``): yaw_deg, yaw_n, yaw_resultant, yaw_source, yaw_ref_deg.

    Each bird's axial estimate (mod 180) is represented within +-90 deg of the population axis ``yaw_ref_deg``
    (axial mean of the birds' own estimates), so that birds whose axes differ by a few degrees are not rotated
    in opposite senses (which would put forward on +x for some birds and on -x for others). Birds without their
    own estimate get the population axis (yaw_source 'population_axis'), as birds without a roost direction get
    the population median; if no bird has an estimate, no yaw correction is made (yaw_deg 0).
    """
    for o, v, a, m, fl in zip(orient, roost_v, accs, amasks, flaps):
        R0 = rotation_to_z(v).astype(np.float32)
        o.update(estimate_yaw((a[fl].astype(np.float32) @ R0.T) * m[fl][:, :, None], m[fl]))
    own = [o["yaw_deg"] for o in orient if o["yaw_source"] == "flapping"]
    ref = axial_mean(np.array(own))[0] if own else float("nan")
    for o in orient:
        o["yaw_ref_deg"] = ref
        if not own:
            o["yaw_deg"] = 0.0
        elif o["yaw_source"] == "flapping":
            o["yaw_deg"] = float(ref + ((o["yaw_deg"] - ref + 90.0) % 180.0 - 90.0))
        else:
            o["yaw_deg"], o["yaw_source"] = ref, "population_axis"


@dataclass
class WindowData:
    acc: np.ndarray  # (N, 200, 3) float16, oriented, g
    acc_mask: np.ndarray  # (N, 200) bool
    gps: np.ndarray  # (N, 10, C) float16, scaled and clipped
    gps_mask: np.ndarray  # (N, 10, C) bool
    meta: pd.DataFrame  # one row per window (+ features and labels)
    orientation: pd.DataFrame  # per bird: roost direction, windows used, tilt corrected


def load(windows: Path, birds: list[str] | None = None, step_g: float = 0.0, clip_g: float = 0.0,
         rotate: bool = True, mask_gps_channels: tuple[str, ...] = (), derotate_yaw: bool = False) -> WindowData:
    """Load windows.

    ``step_g``/``clip_g`` apply :func:`common_format`. ``rotate=False`` keeps the sensor frame (the per-bird
    rotation is still estimated and returned in ``orientation``). ``mask_gps_channels`` are hidden from the
    model (mask False, value 0), e.g. the horizontal-accuracy estimate, which describes fix quality, not behaviour.

    Birds without enough quiet windows get the median roost direction of the other birds (``source`` column).

    ``derotate_yaw`` adds the yaw rotation described in the module docstring after the roost rotation; the
    stored ``R`` is then the combined rotation (yaw @ roost), and orientation gains yaw_deg (the axis angle in
    the roost frame that was rotated onto +x), yaw_n (qualifying windows), yaw_resultant, yaw_ref_deg and
    yaw_source ('flapping'; 'population_axis' for birds with fewer than YAW_MIN_WINDOWS qualifying windows; see
    :func:`derotation_yaw`). Off by default: R and the columns are as before.
    """
    dirs = sorted(p for p in windows.iterdir() if p.is_dir())
    if birds is not None:
        dirs = [d for d in dirs if d.name in set(birds)]
    accs, amasks, gpss, gmasks, metas, orient, flaps = [], [], [], [], [], [], []
    scale = np.array([GPS_SCALE[c] for c in GPS_CHANNELS], dtype=np.float32)
    for d in dirs:
        m = pd.read_parquet(d / "meta.parquet")
        if len(m) == 0:
            continue  # bird without kept windows
        f = pd.read_parquet(d / "features.parquet")
        acc = np.load(d / "acc.npy")
        am = np.load(d / "acc_mask.npy")
        night_n = int(((m["sun_elev_deg"].to_numpy() < -6) & (f["vedba"].to_numpy() < 0.02) & (am.sum(1) >= 20)).sum())
        v, n_used = roost_gravity(acc, am, m["sun_elev_deg"].to_numpy(), f["vedba"].to_numpy(),
                                  speed=f["speed_mean"].to_numpy() if "speed_mean" in f else None)
        if step_g > 0 or clip_g > 0:
            acc = common_format(acc, am, step_g, clip_g)
        source = "none" if v is None else ("night" if night_n >= 20 else "day_still")
        orient.append({"bird": d.name, "roost_windows": n_used, "source": source, "v": v})
        g = np.load(d / "gps.npy") / scale
        g[:, :, GPS_CHANNELS.index("eobs:horizontal-accuracy-estimate")] = np.log1p(
            np.load(d / "gps.npy")[:, :, GPS_CHANNELS.index("eobs:horizontal-accuracy-estimate")]
        )
        gm = np.load(d / "gps_mask.npy")
        for c in mask_gps_channels:
            gm[:, :, GPS_CHANNELS.index(c)] = False
        g = np.clip(g, -GPS_CLIP, GPS_CLIP) * gm
        accs.append(acc.astype(np.float16))
        amasks.append(am)
        gpss.append(g.astype(np.float16))
        gmasks.append(gm)
        metas.append(pd.concat([m, f], axis=1))
        if derotate_yaw:
            flaps.append(np.flatnonzero(f["label_flapping"].to_numpy(dtype=float) == 1) if "label_flapping" in f
                         else np.zeros(0, int))
    # orientation: per-bird roost direction, or the median of the others for birds without one
    have = [o["v"] / np.linalg.norm(o["v"]) for o in orient if o["v"] is not None]
    median_v = np.median(np.stack(have), axis=0) if have else np.array([0.0, 0.0, 1.0])
    roost_v = []
    for o in orient:
        v = o.pop("v")
        if v is None:
            v, o["source"] = median_v, "population_median"
        roost_v.append(v)
    if derotate_yaw:
        derotation_yaw(orient, roost_v, accs, amasks, flaps)
    for o, i in zip(orient, range(len(accs))):
        v = roost_v[i]
        R = rotation_to_z(v).astype(np.float32)
        if derotate_yaw:  # rotate the axis at yaw_deg (roost frame) onto +x
            R = (rotation_z(-o["yaw_deg"]) @ rotation_to_z(v)).astype(np.float32)
        if rotate:
            accs[i] = (accs[i].astype(np.float32) @ R.T).astype(np.float16) * amasks[i][:, :, None]
        o["tilt_corrected_deg"] = float(np.degrees(np.arccos(np.clip(v[2] / np.linalg.norm(v), -1, 1))))
        o["R"] = R.ravel().tolist()
    meta = pd.concat(metas, ignore_index=True)
    first = meta.groupby("bird")["t_start"].transform("min")
    meta["cohort"] = np.where(first.dt.year <= 2024, "2024", "2025")
    return WindowData(
        np.concatenate(accs), np.concatenate(amasks), np.concatenate(gpss), np.concatenate(gmasks), meta, pd.DataFrame(orient)
    )
