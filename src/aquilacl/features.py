"""Stage 3b: handcrafted window features and rule-based reference labels.

The reference labels are used ONLY to evaluate representations (probes) and to describe states. They are
never training targets. Thresholds follow the literature review (docs/literature.md); every threshold is an
argument of ``reference_labels`` so it is recorded with the run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .windows import FS, GPS_CHANNELS

FLAP_BAND = (1.5, 4.0)  # Hz; golden eagle wingbeat about 2.2-2.8 Hz


def _masked_mean(x: np.ndarray, m: np.ndarray, axis: int) -> np.ndarray:
    n = m.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (x * m).sum(axis=axis) / n, np.nan)


def acc_features(acc: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Features of (N, T, 3) acceleration in g with (N, T) sample mask."""
    a = acc.astype(np.float32)
    m = mask.astype(np.float32)
    static = _masked_mean(a, m[:, :, None], axis=1)  # (N, 3) window-mean posture
    dyn = (a - static[:, None, :]) * m[:, :, None]
    vedba_t = np.linalg.norm(dyn, axis=2)
    out = {
        "vedba": _masked_mean(vedba_t, m, 1),
        "vedba_sd": np.sqrt(np.maximum(_masked_mean(vedba_t**2, m, 1) - _masked_mean(vedba_t, m, 1) ** 2, 0)),
        "odba": _masked_mean(np.abs(dyn).sum(axis=2), m, 1),
        "static_norm": np.linalg.norm(static, axis=1),
        "pitch_deg": np.degrees(np.arctan2(static[:, 0], np.hypot(static[:, 1], static[:, 2]))),
        "roll_deg": np.degrees(np.arctan2(static[:, 1], static[:, 2])),
    }
    # Spectrum of the dynamic component on the axis with most variance; zero-filled gaps.
    var = (dyn**2).sum(axis=1)
    ax = np.argmax(var, axis=1)
    x = dyn[np.arange(len(dyn)), :, ax]
    spec = np.abs(np.fft.rfft(x, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], 1 / FS)
    above = f > 0.5
    band = (f >= FLAP_BAND[0]) & (f <= FLAP_BAND[1])
    tot = spec[:, above].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["flap_band_frac"] = np.where(tot > 0, spec[:, band].sum(axis=1) / tot, np.nan)
        out["dom_freq_hz"] = np.where(tot > 0, f[above][np.argmax(spec[:, above], axis=1)], np.nan)
    short = m.sum(axis=1) < 40  # under 2 s of samples: spectrum too coarse
    out["flap_band_frac"][short] = np.nan
    out["dom_freq_hz"][short] = np.nan
    return pd.DataFrame(out)


def gps_features(g: np.ndarray, gm: np.ndarray) -> pd.DataFrame:
    """Features of (N, S, C) per-second GPS kinematics with mask."""
    c = {name: i for i, name in enumerate(GPS_CHANNELS)}

    def ch(name):
        i = c[name]
        return np.where(gm[:, :, i], g[:, :, i], np.nan)

    speed = ch("ground-speed")
    climb = ch("climb_ms")
    turn = ch("turn_rate_dps")
    step = ch("step_m")
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows give NaN, as intended
        out = {
            "speed_mean": np.nanmean(speed, axis=1),
            "speed_max": np.nanmax(speed, axis=1),
            "climb_mean": np.nanmean(climb, axis=1),
            "climb_min": np.nanmin(climb, axis=1),
            "turn_abs_mean": np.nanmean(np.abs(turn), axis=1),
            "turn_net_deg": np.nansum(turn, axis=1),  # dt = 1 s
            "path_m": np.nansum(step, axis=1),
            "hacc_mean": np.nanmean(ch(GPS_CHANNELS[-1]), axis=1),
            "n_speed": np.isfinite(speed).sum(axis=1),
            "n_turn": np.isfinite(turn).sum(axis=1),
        }
    return pd.DataFrame(out)


def reference_labels(
    f: pd.DataFrame,
    sun_elev_deg: np.ndarray,
    perch_speed: float = 1.0,
    perch_vedba: float = 0.08,
    flap_vedba: float = 0.25,
    flap_band: float = 0.35,
    fly_speed: float = 4.0,
    circle_turn_dps: float = 6.0,
    soar_vedba: float = 0.2,
    glide_climb: float = -0.5,
    glide_turn_dps: float = 3.0,
    min_fixes: int = 5,
) -> pd.DataFrame:
    """Boolean evaluation labels; a window can carry several (they are not exclusive)."""
    gps_ok = f["n_speed"] >= min_fixes
    turn_ok = f["n_turn"] >= min_fixes
    lab = pd.DataFrame(index=f.index)
    lab["night"] = sun_elev_deg < -6
    lab["perched"] = gps_ok & (f["speed_mean"] < perch_speed) & (f["vedba"] < perch_vedba)
    lab["flapping"] = (f["vedba"] > flap_vedba) & (f["flap_band_frac"] > flap_band)
    flying = gps_ok & (f["speed_mean"] > fly_speed)
    lab["circling"] = flying & turn_ok & (f["turn_abs_mean"] > circle_turn_dps) & (f["vedba"] < soar_vedba)
    lab["gliding"] = (
        flying & turn_ok & (f["turn_abs_mean"] < glide_turn_dps) & (f["climb_mean"] < glide_climb) & (f["vedba"] < soar_vedba)
    )
    lab["climbing_soar"] = flying & (f["climb_mean"] > 0.5) & (f["vedba"] < soar_vedba)
    return lab
