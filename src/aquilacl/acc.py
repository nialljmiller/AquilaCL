"""Acceleration bursts: decoding, per-tag calibration and burst features.

e-obs tags store each burst as interleaved raw counts (X Y Z X Y Z ...), here at
20 Hz per axis, in bursts of 24, 40 or 264 samples per axis. The counts are not
calibrated, so each tag gets an offset and a gain per axis, fitted from bursts
in which the bird is still. For a still bird the only acceleration is gravity,
so the calibrated mean vector should have length 1 g whatever the posture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

RAW_CENTRE = 2048.0  # 12-bit digitiser mid-point, used only as a starting value


def decode(flat: np.ndarray, n_axes: int = 3) -> np.ndarray:
    """Interleaved counts -> array of shape (samples, axes)."""
    flat = np.asarray(flat, dtype=np.float32)
    n = len(flat) // n_axes
    return flat[: n * n_axes].reshape(n, n_axes)


@dataclass
class Calibration:
    offset: np.ndarray  # counts, shape (3,)
    gain: np.ndarray  # counts per g, shape (3,)
    model: str  # "full" (3 offsets, 3 gains), "shared_gain" (3 offsets, 1 gain) or "fixed_offset" (1 gain)
    n_static: int  # bursts used for the fit
    posture_spread_deg: float  # spread of the still-burst gravity directions
    condition: float  # condition number of the fit Jacobian (large = poorly constrained)
    residual_g: float  # median | |a| - 1 g | over the still bursts after calibration

    def apply(self, counts: np.ndarray) -> np.ndarray:
        return (counts - self.offset) / self.gain


def _unpack(model: str, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model == "full":
        return p[:3], np.abs(p[3:])
    if model == "shared_gain":
        return p[:3], np.full(3, abs(p[3]))
    return np.full(3, RAW_CENTRE), np.full(3, abs(p[0]))


def _fit(model: str, m: np.ndarray, gain0: float) -> tuple[np.ndarray, float]:
    x0 = {
        "full": np.r_[np.full(3, RAW_CENTRE), np.full(3, gain0)],
        "shared_gain": np.r_[np.full(3, RAW_CENTRE), gain0],
        "fixed_offset": np.r_[gain0],
    }[model]

    def resid(p):
        o, g = _unpack(model, p)
        return np.linalg.norm((m - o) / g, axis=1) - 1.0

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.05)
    sv = np.linalg.svd(sol.jac, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return sol.x, cond


def fit_calibration(
    means: np.ndarray,
    still: np.ndarray | None = None,
    min_static: int = 50,
    max_static: int = 20000,
    max_condition: float = 200.0,
    seed: int = 0,
) -> Calibration:
    """Fit offset and gain per axis from the mean vectors of still bursts.

    ``means`` holds the per-burst mean counts (N, 3). ``still`` marks the
    bursts to use; the caller decides stillness (low within-burst variation
    AND sun below the horizon or GPS speed near zero), because steady banked
    soaring also gives low-variation bursts, at more than 1 g.

    The full 6-parameter fit needs gravity seen from several directions. A
    backpack tag on a perched eagle mostly sits upright, which leaves offset
    and gain along that axis confounded. When the Jacobian condition number
    exceeds ``max_condition`` the fit falls back to a shared gain, then to a
    fixed 2048 offset with one gain.
    """
    m = np.asarray(means, dtype=np.float64).reshape(-1, 3)
    if still is not None:
        m = m[np.asarray(still, dtype=bool)]
    m = m[np.isfinite(m).all(axis=1)]
    if len(m) > max_static:
        m = m[np.random.default_rng(seed).choice(len(m), max_static, replace=False)]
    spread = float("nan")
    if len(m) >= 3:
        u = (m - RAW_CENTRE) / np.linalg.norm(m - RAW_CENTRE, axis=1, keepdims=True)
        spread = float(np.degrees(np.arccos(np.clip(np.linalg.norm(u.mean(axis=0)), -1, 1))))
    gain0 = float(np.median(np.linalg.norm(m - RAW_CENTRE, axis=1))) if len(m) else 1000.0

    models = ["full", "shared_gain", "fixed_offset"] if len(m) >= min_static else ["fixed_offset"]
    for model in models:
        if len(m) == 0:
            p, cond = np.r_[gain0], float("nan")
            break
        p, cond = _fit(model, m, gain0)
        if cond <= max_condition:
            break
    o, g = _unpack(model, p)
    resid = float(np.median(np.abs(np.linalg.norm((m - o) / g, axis=1) - 1.0))) if len(m) else float("nan")
    return Calibration(o, g, model, len(m), spread, cond, resid)


def sensitivity_regime(
    t: "pd.Series",
    mean_counts: np.ndarray,
    still: np.ndarray,
    base_counts_per_g: float = 512.0,
    min_still_per_day: int = 5,
) -> np.ndarray:
    """Sensitivity setting per burst, as k in base * 2**k counts per g.

    Some tags were reconfigured for weeks or months (here: about 1024 instead of
    512 counts per g). A still burst must read 1 g, so the daily median length of
    the still-burst mean vectors identifies the setting of each day. Days
    without enough still bursts take the setting of the nearest earlier day
    (or the next later one at the start of a track).
    """
    import pandas as pd

    norm = np.linalg.norm(np.asarray(mean_counts, dtype=float) - RAW_CENTRE, axis=1)
    day = pd.to_datetime(t).dt.floor("D")
    df = pd.DataFrame({"day": day.to_numpy(), "norm": norm, "still": np.asarray(still, dtype=bool)})
    daily = df[df["still"]].groupby("day")["norm"].agg(["median", "size"])
    daily = daily[daily["size"] >= min_still_per_day]
    k_day = np.round(np.log2(daily["median"] / base_counts_per_g)).astype(int)
    days = pd.Series(k_day, index=daily.index).reindex(sorted(df["day"].unique()))
    days = days.ffill().bfill().fillna(0).astype(int)
    return days.reindex(df["day"]).to_numpy()


def burst_stats(flat_bursts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-burst sample count, mean counts (N, 3) and largest per-axis std (N,)."""
    n = np.empty(len(flat_bursts), dtype=np.int32)
    mean = np.full((len(flat_bursts), 3), np.nan)
    sd = np.full(len(flat_bursts), np.nan)
    for i, f in enumerate(flat_bursts):
        b = decode(f)
        n[i] = len(b)
        if len(b):
            mean[i] = b.mean(axis=0)
            sd[i] = b.std(axis=0).max()
    return n, mean, sd


def burst_features(g: np.ndarray, fs: float = 20.0, static_window_s: float = 1.0) -> dict[str, float]:
    """Summary features of one calibrated burst (samples, 3) in g.

    Static (posture) acceleration is a centred running mean of
    ``static_window_s``; dynamic acceleration is the remainder. Short bursts
    (fewer samples than the window) use the burst mean as the static part.
    The dominant frequency is only reported for bursts of at least 64 samples;
    shorter bursts give too coarse a spectrum.
    """
    n = len(g)
    w = int(round(static_window_s * fs))
    if n >= 2 * w and w > 1:
        kernel = np.ones(w) / w
        pad = np.pad(g, ((w // 2, w - 1 - w // 2), (0, 0)), mode="edge")
        static = np.column_stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(3)])
    else:
        static = np.broadcast_to(g.mean(axis=0), g.shape)
    dyn = g - static
    s = static.mean(axis=0)
    out = {
        "n_samples": float(n),
        "duration_s": n / fs,
        "vedba": float(np.linalg.norm(dyn, axis=1).mean()),
        "odba": float(np.abs(dyn).sum(axis=1).mean()),
        "static_x": float(s[0]),
        "static_y": float(s[1]),
        "static_z": float(s[2]),
        "static_norm": float(np.linalg.norm(s)),
        # Tag-frame angles; the sign convention depends on how the tag sits.
        "pitch_deg": float(np.degrees(np.arctan2(s[0], np.hypot(s[1], s[2])))),
        "roll_deg": float(np.degrees(np.arctan2(s[1], s[2]))),
        "std_x": float(g[:, 0].std()),
        "std_y": float(g[:, 1].std()),
        "std_z": float(g[:, 2].std()),
        "dom_freq_hz": np.nan,
        "dom_power_frac": np.nan,
    }
    if n >= 64:
        axis = int(np.argmax(dyn.var(axis=0)))
        x = dyn[:, axis] * np.hanning(n)
        p = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(n, 1 / fs)
        band = f > 0.5
        if p[band].sum() > 0:
            k = np.argmax(p[band])
            out["dom_freq_hz"] = float(f[band][k])
            out["dom_power_frac"] = float(p[band][k] / p[band].sum())
    return out
