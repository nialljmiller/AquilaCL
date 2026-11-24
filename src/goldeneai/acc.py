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
    n_static: int  # bursts used for the fit
    residual_g: float  # median | |a| - 1 g | over those bursts after calibration
    fitted: bool  # False if too few static bursts; then a default was used

    def apply(self, counts: np.ndarray) -> np.ndarray:
        return (counts - self.offset) / self.gain


def fit_calibration(
    bursts: list[np.ndarray],
    max_std_counts: float = 8.0,
    min_static: int = 50,
    max_static: int = 20000,
    seed: int = 0,
) -> Calibration:
    """Fit offset and gain per axis from bursts with almost no movement.

    ``bursts`` are (samples, 3) count arrays from one tag. A burst counts as
    static when the standard deviation of every axis is below
    ``max_std_counts``. The fit minimises (|(m - offset) / gain| - 1)^2 over
    the static burst means m. It needs static bursts in several postures,
    which perching and roosting provide.
    """
    means, stds = [], []
    for b in bursts:
        if len(b) >= 8:
            means.append(b.mean(axis=0))
            stds.append(b.std(axis=0).max())
    means_a = np.array(means, dtype=np.float64).reshape(-1, 3)
    static = means_a[np.array(stds) < max_std_counts] if len(stds) else means_a
    if len(static) > max_static:
        static = static[np.random.default_rng(seed).choice(len(static), max_static, replace=False)]

    default_gain = np.median(np.linalg.norm(static - RAW_CENTRE, axis=1)) if len(static) else 1000.0
    x0 = np.r_[np.full(3, RAW_CENTRE), np.full(3, default_gain)]
    if len(static) < min_static:
        cal = Calibration(x0[:3], x0[3:], len(static), np.nan, fitted=False)
        cal.residual_g = _residual(cal, static)
        return cal

    def resid(p):
        return np.linalg.norm((static - p[:3]) / p[3:], axis=1) - 1.0

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=0.05)
    cal = Calibration(sol.x[:3], np.abs(sol.x[3:]), len(static), np.nan, fitted=True)
    cal.residual_g = _residual(cal, static)
    return cal


def _residual(cal: Calibration, static: np.ndarray) -> float:
    if len(static) == 0:
        return float("nan")
    return float(np.median(np.abs(np.linalg.norm(cal.apply(static), axis=1) - 1.0)))


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
