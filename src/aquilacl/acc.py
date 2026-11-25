"""Acceleration bursts: decoding, per-tag calibration and burst features.

e-obs tags store each burst as interleaved raw counts (X Y Z X Y Z ...), here at
20 Hz per axis, in bursts of 24, 40 or 264 samples per axis. The counts are not
calibrated, so each tag gets an offset and a gain per axis, fitted from bursts
in which the bird is still. For a still bird the only acceleration is gravity,
so the calibrated mean vector should have length 1 g whatever the posture.

The tags run at one of two sensitivity settings, 512 * 2**k counts per g (k = 0 or 1 here).
The setting belongs to the burst type (per tag and burst length), not to the day, so k is
assigned per (tag, burst length) group from its quiet bursts (``group_sensitivity``) and
each tag is calibrated once per k (``calibrate_tag``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

RAW_CENTRE = 2048.0  # 12-bit digitiser mid-point, used only as a starting value
RAW_MAX = 4095.0  # largest 12-bit count (saturation)
BASE_COUNTS_PER_G = 512.0  # sensitivity settings are BASE_COUNTS_PER_G * 2**k counts per g
QUIET_REL_SD = 0.01  # quiet burst: largest per-axis sd below 1% of the burst's mean-vector length
MIN_QUIET = 20  # quiet bursts a (tag, burst length) group needs to measure its own k


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
    median_g: float = float("nan")  # median |a| of the still bursts after calibration

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
    n, spread, resid, med = _summary(m, o, g)
    return Calibration(o, g, model, n, spread, cond, resid, med)


def _summary(m: np.ndarray, o: np.ndarray, g: np.ndarray) -> tuple[int, float, float, float]:
    """Still-burst count, posture spread (deg), median | |a| - 1 g | and median |a| under (o, g)."""
    spread = resid = med = float("nan")
    if len(m) >= 3:
        u = (m - RAW_CENTRE) / np.linalg.norm(m - RAW_CENTRE, axis=1, keepdims=True)
        spread = float(np.degrees(np.arccos(np.clip(np.linalg.norm(u.mean(axis=0)), -1, 1))))
    if len(m):
        a = np.linalg.norm((m - o) / g, axis=1)
        resid, med = float(np.median(np.abs(a - 1.0))), float(np.median(a))
    return len(m), spread, resid, med


def length_majority(groups: "pd.DataFrame") -> dict[int, int]:
    """Most common k per burst length over the measured groups (ties go to the smaller k)."""
    m = groups[groups["k_source"] == "measured"]
    return {int(n): int(s.value_counts().sort_index().idxmax()) for n, s in m.groupby("n_samples")["k"]}


def group_sensitivity(
    groups: "pd.DataFrame", length_k: dict[int, int] | None = None, min_quiet: int = MIN_QUIET
) -> "pd.DataFrame":
    """Sensitivity setting k (``BASE_COUNTS_PER_G * 2**k`` counts per g) per burst group.

    ``groups`` has one row per (tag, burst length) with ``n_samples``, ``n_quiet`` and
    ``median_norm`` (median raw mean-vector length of the quiet bursts, in counts). The setting
    belongs to the burst type, not the day, so one k per group suffices; a quiet bird reads
    1 g, so k = round(log2(median_norm / BASE_COUNTS_PER_G)) (``k_source`` "measured").
    Groups with fewer than ``min_quiet`` quiet bursts take the majority k of the measured
    groups of the same burst length (``length_k``, default: from ``groups`` itself;
    "length_majority"); if no measured group has that length, k = 0 ("unknown").
    Returns ``groups`` with columns ``k`` and ``k_source`` added.
    """
    g = groups.copy()
    measured = (g["n_quiet"] >= min_quiet) & (g["median_norm"] > 0)
    g["k"] = 0
    g.loc[measured, "k"] = np.round(np.log2(g.loc[measured, "median_norm"] / BASE_COUNTS_PER_G)).astype(int)
    g["k_source"] = np.where(measured, "measured", "unknown")
    if length_k is None:
        length_k = length_majority(g)
    fb = ~measured & g["n_samples"].isin(list(length_k))
    g.loc[fb, "k"] = g.loc[fb, "n_samples"].map(length_k)
    g.loc[fb, "k_source"] = "length_majority"
    g["k"] = g["k"].astype(int)
    return g


def calibrate_tag(
    means: np.ndarray, still: np.ndarray, k: np.ndarray, min_static: int = 50
) -> dict[int, Calibration]:
    """One calibration per sensitivity setting k of one tag, from the still bursts of that k.

    A setting with at least ``min_static`` still bursts gets its own fit (``fit_calibration``).
    A setting with fewer is derived from the same tag's reliable setting with the most still
    bursts: same offset, gain scaled by 2**(k - k_donor), model "scaled_from_k<donor>". Without a
    reliable donor it gets the fixed-offset fit of its few bursts (flagged "few_still").
    """
    means, still, k = np.asarray(means, float), np.asarray(still, bool), np.asarray(k)
    cals: dict[int, Calibration] = {}
    for kk in np.unique(k):
        sel = still & (k == kk)
        if sel.sum() >= min_static:
            cals[int(kk)] = fit_calibration(means[sel], min_static=min_static)
    donors = [d for d, c in cals.items() if not calibration_flags(c, min_static)]
    for kk in np.unique(k):
        kk = int(kk)
        if kk in cals:
            continue
        m = means[still & (k == kk)]
        if donors:
            d = max(donors, key=lambda d: cals[d].n_static)
            c = cals[d]
            g = c.gain * 2.0 ** (kk - d)
            n, spread, resid, med = _summary(m, c.offset, g)
            cals[kk] = Calibration(c.offset, g, f"scaled_from_k{d}", n, spread, c.condition, resid, med)
        else:
            cals[kk] = fit_calibration(m, min_static=min_static)
    return cals


def calibration_flags(cal: Calibration, min_static: int = 50, max_residual_g: float = 0.05,
                      norm_range: tuple[float, float] = (0.95, 1.05), min_check: int = 5) -> list[str]:
    """Reasons a calibration is unreliable (empty list = reliable).

    "few_still": a fitted (not scaled) calibration from fewer than ``min_static`` still bursts;
    "unchecked": fewer than ``min_check`` still bursts to check it on;
    "residual": median | |a| - 1 g | above ``max_residual_g``;
    "norm": median still |a| outside ``norm_range`` (a gain error above 5%).
    Windows leave out the bursts of flagged calibrations, so these limits trade data for accuracy.
    """
    f = []
    if not cal.model.startswith("scaled") and cal.n_static < min_static:
        f.append("few_still")
    if cal.n_static < min_check:
        f.append("unchecked")
    else:
        if not cal.residual_g <= max_residual_g:
            f.append("residual")
        if not norm_range[0] <= cal.median_g <= norm_range[1]:
            f.append("norm")
    return f


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
