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
STATIC_S = 1.0  # running-mean length for the static (gravity) part, as acc.burst_features
MIN_SPECTRUM_SAMPLES = 40  # 2 s of contiguous data: >= 4 wingbeats, 0.5 Hz resolution, 5 bins in FLAP_BAND
CHUNK = 20_000  # windows per block, bounds memory


def _masked_mean(x: np.ndarray, m: np.ndarray, axis: int) -> np.ndarray:
    n = m.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (x * m).sum(axis=axis) / n, np.nan)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of the contiguous run of present samples holding each position."""
    n, t = mask.shape
    ar = np.broadcast_to(np.arange(t), (n, t))
    first = mask & ~np.c_[np.zeros((n, 1), bool), mask[:, :-1]]
    last = mask & ~np.c_[mask[:, 1:], np.zeros((n, 1), bool)]
    start = np.maximum.accumulate(np.where(first, ar, 0), axis=1)
    end = np.minimum.accumulate(np.where(last, ar + 1, t)[:, ::-1], axis=1)[:, ::-1]
    return start, end


def static_component(a: np.ndarray, mask: np.ndarray, fs: float = FS, static_s: float = STATIC_S) -> np.ndarray:
    """Centred running mean of ``static_s`` within each contiguous run, edge-padded (as acc.burst_features).

    Runs shorter than two averaging lengths use the run mean. Absent samples give 0.
    """
    w = int(round(static_s * fs))
    s, e = _runs(mask)
    am = a * mask[:, :, None]
    c = np.concatenate([np.zeros_like(am[:, :1]), np.cumsum(am, axis=1)], axis=1)  # (N, T+1, 3)
    t = np.arange(a.shape[1])
    lo, hi = t - w // 2, t - w // 2 + w
    in_lo, in_hi = np.clip(lo, s, e), np.clip(hi, s, e)
    take = lambda idx: np.take_along_axis(c, idx[:, :, None], axis=1)
    edge = lambda idx: np.take_along_axis(a, np.clip(idx, 0, a.shape[1] - 1)[:, :, None], axis=1)
    run = (take(e) - take(s)) / np.maximum(e - s, 1)[:, :, None]
    roll = (take(in_hi) - take(in_lo) + (in_lo - lo)[:, :, None] * edge(s) + (hi - in_hi)[:, :, None] * edge(e - 1)) / w
    return np.where(((e - s) >= 2 * w)[:, :, None], roll, run) * mask[:, :, None]


def _spectrum(dyn: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flap-band fraction and dominant frequency of the longest contiguous run (Hann over the run, zero-padded)."""
    n, t = mask.shape
    s, e = _runs(mask)
    length = np.where(mask, e - s, 0)
    best = np.argmax(length, axis=1)
    s0, L = s[np.arange(n), best], length[np.arange(n), best]
    k = np.arange(t)
    src = np.clip(s0[:, None] + k, 0, t - 1)
    seg = np.take_along_axis(dyn, src[:, :, None], axis=1)
    seg_m = (k < L[:, None])[:, :, None]
    ax = np.argmax(((seg * seg_m) ** 2).sum(axis=1), axis=1)
    x = seg[np.arange(n), :, ax]
    with np.errstate(invalid="ignore", divide="ignore"):
        hann = np.where(k < L[:, None], 0.5 - 0.5 * np.cos(2 * np.pi * k / np.maximum(L[:, None] - 1, 1)), 0.0)
    spec = np.abs(np.fft.rfft(x * hann, axis=1)) ** 2
    f = np.fft.rfftfreq(t, 1 / FS)
    above = f > 0.5
    band = (f >= FLAP_BAND[0]) & (f <= FLAP_BAND[1])
    tot = spec[:, above].sum(axis=1)
    ok = (tot > 0) & (L >= MIN_SPECTRUM_SAMPLES)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(ok, spec[:, band].sum(axis=1) / tot, np.nan)
        dom = np.where(ok, f[above][np.argmax(spec[:, above], axis=1)], np.nan)
    return frac, dom


def acc_features(acc: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Features of (N, T, 3) acceleration in g with (N, T) sample mask.

    The static part is a 1 s running mean within contiguous present samples, so features do not depend on how
    full the window is. The spectrum uses only the longest contiguous run and needs MIN_SPECTRUM_SAMPLES.
    """
    if len(acc) > CHUNK:
        return pd.concat([acc_features(acc[i:i + CHUNK], mask[i:i + CHUNK]) for i in range(0, len(acc), CHUNK)],
                         ignore_index=True)
    a = acc.astype(np.float32)
    m = mask.astype(np.float32)
    st = static_component(a, mask)
    static = _masked_mean(st, m[:, :, None], axis=1)  # (N, 3) window-mean posture
    dyn = (a - st) * m[:, :, None]
    vedba_t = np.linalg.norm(dyn, axis=2)
    out = {
        "vedba": _masked_mean(vedba_t, m, 1),
        "vedba_sd": np.sqrt(np.maximum(_masked_mean(vedba_t**2, m, 1) - _masked_mean(vedba_t, m, 1) ** 2, 0)),
        "odba": _masked_mean(np.abs(dyn).sum(axis=2), m, 1),
        "static_norm": np.linalg.norm(static, axis=1),
        "pitch_deg": np.degrees(np.arctan2(static[:, 0], np.hypot(static[:, 1], static[:, 2]))),
        "roll_deg": np.degrees(np.arctan2(static[:, 1], static[:, 2])),
    }
    out["flap_band_frac"], out["dom_freq_hz"] = _spectrum(dyn, mask)
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
    """Evaluation labels as float: 1.0 true, 0.0 false, NaN unknown (any input missing).

    A window can carry several labels (they are not exclusive).
    """

    def known(*cols):
        ok = np.ones(len(f), bool)
        for c in cols:
            ok &= np.isfinite(f[c].to_numpy(dtype=float))
        return ok

    def label(cond, ok):
        return pd.Series(np.where(ok, np.asarray(cond, dtype=float), np.nan), index=f.index)

    gps_ok = (f["n_speed"] >= min_fixes).to_numpy() & known("speed_mean")
    turn_ok = (f["n_turn"] >= min_fixes).to_numpy() & known("turn_abs_mean")
    sun = np.asarray(sun_elev_deg, dtype=float)
    lab = pd.DataFrame(index=f.index)
    lab["night"] = label(sun < -6, np.isfinite(sun))
    lab["perched"] = label((f["speed_mean"] < perch_speed) & (f["vedba"] < perch_vedba), gps_ok & known("vedba"))
    lab["flapping"] = label((f["vedba"] > flap_vedba) & (f["flap_band_frac"] > flap_band), known("vedba", "flap_band_frac"))
    flying = f["speed_mean"] > fly_speed
    soar = gps_ok & known("vedba")
    lab["circling"] = label(flying & (f["turn_abs_mean"] > circle_turn_dps) & (f["vedba"] < soar_vedba), soar & turn_ok)
    lab["gliding"] = label(
        flying & (f["turn_abs_mean"] < glide_turn_dps) & (f["climb_mean"] < glide_climb) & (f["vedba"] < soar_vedba),
        soar & turn_ok & known("climb_mean"),
    )
    lab["climbing_soar"] = label(flying & (f["climb_mean"] > 0.5) & (f["vedba"] < soar_vedba), soar & known("climb_mean"))
    return lab
