"""Environmental covariates per window: ERA5 surface fields and Copernicus GLO-30 terrain.

Sources (both anonymous, public)
- ERA5 from the ARCO-ERA5 Zarr store on Google Cloud (``ARCO``): hourly, 0.25 deg, one chunk = one whole-globe
  hour of one variable (721 x 1440 float32, blosc-lz4). Latitude runs 90 -> -90, longitude 0 -> 359.75, the
  time index is hours since 1900-01-01 00 UTC (time[i] == i). Final ERA5 up to the store's ``valid_time_stop``,
  preliminary ERA5T after it, up to ``valid_time_stop_era5t``.
- Copernicus DEM GLO-30 on AWS (``DEM_BASE``): 1 x 1 deg Cloud-Optimised GeoTIFF tiles named by their
  south-west corner; heights are orthometric (EGM2008 geoid). Above 50 deg the longitude spacing is coarser, so
  each tile's own transform is used and pixels are never assumed square.

Privacy. Only whole-globe ERA5 chunks are requested (they carry no information about the birds) and DEM tiles of
the 1-degree cells touched by fixes (approved by the owner). The fix bounding box is computed on the cluster,
stored in <env>/era5/box.json and printed only at 5-degree rounding. Outputs never contain latitude, longitude or
timestamps.

download
  ERA5 variables ``VARS`` cropped in memory to the fix bounding box (rounded outward to whole degrees) and
  stored as one HDF5 file per variable-month, <env>/era5/<var>/<var>_<YYYY-MM>.h5 (datasets data (time, lat,
  lon) float32, hour (hours since 1900), lat, lon; netCDF-like layout written with h5py, not a netCDF library).
  GLO-30 tiles of the 1-degree cells within ``TILE_BUFFER_M`` of any fix -> <env>/dem/<name>.tif; tiles that do
  not exist (sea; HTTP 403/404) get a <name>.absent marker. Resumable and idempotent: a complete file is never
  overwritten or re-fetched; data are written to a uniquely named ``.partial`` file and renamed only when the
  final name does not exist. A variable-month is written only when every hour was fetched and decoded; any
  network error leaves it absent for the next run. Hours are requested only up to the store's ERA5T end, so a
  missing chunk (404) is an error, not NaN.

join
  Per bird and window of windows_v4/<bird>/meta.parquet (rows 1:1, same order):
  position = mean of the bird's good raw GPS fixes inside [t_start, t_start + window_s) (``pos_source`` 1), else
  the raw fix nearest to the window mid-time within +-``NEAR_S`` (2), else none (0, all features NaN);
  ``pos_dt_s`` = |mid-time - fix time| of that fix (0 for own fixes). ERA5 bilinear in space, linear in time at the
  window mid-time. Terrain from GLO-30: bilinear elevation; gradient by central differences over +-``GRAD_M``
  east/north; slope; aspect (azimuth of the downslope direction, clockwise from north; NaN where the gradient is
  below ``FLAT_GRAD``); local relief = max - min elevation within ``RELIEF_M`` (``RELIEF_STEP_M`` grid);
  height above ground = GPS height above ellipsoid - DEM elevation; orographic uplift w_o = u100 dz/dx + v100
  dz/dy (positive when the wind blows towards higher ground). Convective velocity scale (Deardorff 1970)
  w* = (g / THETA_REF * H / (rho c_p) * blh)^(1/3) with H = -ishf (ERA5 fluxes are positive downward, so surface
  heating is negative ishf) and w* = 0 when H <= 0; rho from the ISA troposphere at the DEM elevation (sea level
  where the DEM is absent).
  Geoid: no geoid grid is available offline in the environment, so hag_m is NOT geoid-corrected
  (README.json "geoid_corrected": false); its bias is minus the EGM2008 undulation N (h_ellipsoid = H + N).
  Output <out>/<bird>.parquet + README.json (units, formulas, constants, flag bits, coverage). <out> must be new
  or empty.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com/"
VARS = {
    "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind", "v100": "100m_v_component_of_wind",
    "blh": "boundary_layer_height", "ishf": "instantaneous_surface_sensible_heat_flux",
}
NLAT, NLON, RES = 721, 1440, 0.25
EPOCH = np.datetime64("1900-01-01T00:00:00", "ns")
TILE_BUFFER_M = 1500.0  # DEM tiles are fetched for cells within this distance of a fix (covers GRAD_M and RELIEF_M)
NEAR_S = 1800.0
GRAD_M, RELIEF_M, RELIEF_STEP_M, FLAT_GRAD = 90.0, 1000.0, 100.0, 0.005
G, THETA_REF, CP = 9.81, 300.0, 1004.7
R_EARTH = 6_371_000.0
DEM_NODATA_BELOW = -1000.0

FLAGS = {"no_position": 1, "era5_no_time": 2, "era5_outside_box": 4, "era5_nan": 8, "dem_missing": 16,
         "no_height": 32}


# --------------------------------------------------------------------------------------------- grid helpers
def box_from_fixes(lat: np.ndarray, lon: np.ndarray) -> dict:
    """Bounding box of the fixes rounded outward to whole degrees (lon in [-180, 180])."""
    return {"lat0": math.floor(np.min(lat)), "lat1": math.ceil(np.max(lat)),
            "lon0": math.floor(np.min(lon)), "lon1": math.ceil(np.max(lon))}


def box5(box: dict) -> str:
    """The only printable form of the box: 5-degree rounding outward."""
    f, c = (lambda x: 5 * math.floor(x / 5)), (lambda x: 5 * math.ceil(x / 5))
    return f"lat {f(box['lat0'])}..{c(box['lat1'])}, lon {f(box['lon0'])}..{c(box['lon1'])} (5-degree rounding)"


def box_indices(box: dict):
    """Global row/column indices of the box and its coordinates (lat descending, lon ascending from lon0)."""
    r0, r1 = round((90 - box["lat1"]) / RES), round((90 - box["lat0"]) / RES)
    rows = np.arange(r0, r1 + 1)
    n_lon = round((box["lon1"] - box["lon0"]) / RES) + 1
    cols = (round((box["lon0"] % 360) / RES) + np.arange(n_lon)) % NLON
    return rows, cols, 90 - rows * RES, box["lon0"] + np.arange(n_lon) * RES


def crop(full: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.take(full[rows[0]:rows[-1] + 1], cols, axis=1)


def hours_since_1900(t) -> np.ndarray:
    t = np.asarray(t, dtype="datetime64[ns]")
    return (t - EPOCH).astype(np.int64) / 3.6e12


def month_hours(month: str) -> tuple[int, int]:
    """[first, end) hour indices of a calendar month 'YYYY-MM'."""
    p = pd.Period(month, "M")
    a = int(hours_since_1900([p.start_time.to_datetime64()])[0])
    b = int(hours_since_1900([(p + 1).start_time.to_datetime64()])[0])
    return a, b


def tile_name(lat_floor: int, lon_floor: int) -> str:
    ns, ew = ("N" if lat_floor >= 0 else "S"), ("E" if lon_floor >= 0 else "W")
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def tiles_for_fixes(lat: np.ndarray, lon: np.ndarray, buffer_m: float = TILE_BUFFER_M) -> list[tuple[int, int]]:
    dlat = np.degrees(buffer_m / R_EARTH)
    dlon = dlat / np.maximum(np.cos(np.radians(lat)), 0.05)
    cells = set()
    for sy in (-1, 0, 1):
        for sx in (-1, 0, 1):
            la, lo = lat + sy * dlat, (lon + sx * dlon + 180) % 360 - 180
            cells |= set(zip(np.floor(la).astype(int).tolist(), np.floor(lo).astype(int).tolist()))
    return sorted(cells)


# ----------------------------------------------------------------------------------------------- network
class NotFound(Exception):
    pass


class RateLimit:
    """Shared byte-rate cap across threads (MB/s; 0 = none)."""

    def __init__(self, mbps: float):
        self.rate, self.lock, self.next = mbps * 1e6, threading.Lock(), time.monotonic()

    def take(self, n: int) -> None:
        if self.rate <= 0:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + n / self.rate
        if start > now:
            time.sleep(start - now)


class Fetcher:
    """requests-based GET with retries; 404 (and 403 from the DEM bucket) -> NotFound; other failures, including
    403 from Google Cloud (rate limiting), are retried and raise after ``retries``."""

    def __init__(self, mbps: float = 0.0, retries: int = 5, timeout: float = 120.0):
        self.limit, self.retries, self.timeout, self.local = RateLimit(mbps), retries, timeout, threading.local()
        self.bytes, self.calls, self._lock = 0, 0, threading.Lock()

    def _session(self):
        import requests

        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        return self.local.s

    def __call__(self, url: str) -> bytes:
        err = None
        for attempt in range(self.retries):
            try:
                with self._session().get(url, stream=True, timeout=self.timeout) as r:
                    if r.status_code == 404 or (r.status_code == 403 and url.startswith(DEM_BASE)):
                        raise NotFound(url)
                    r.raise_for_status()
                    buf = bytearray()
                    for part in r.iter_content(1 << 20):
                        self.limit.take(len(part))
                        buf += part
                    n = r.headers.get("Content-Length")
                    if n is not None and int(n) != len(buf):
                        raise OSError(f"short read {len(buf)} of {n}")
                with self._lock:
                    self.bytes += len(buf)
                    self.calls += 1
                return bytes(buf)
            except NotFound:
                raise
            except Exception as e:  # noqa: BLE001 - network errors of any kind are retried, then raised
                err = e
                time.sleep(min(60, 2 ** attempt))
        raise OSError(f"failed after {self.retries} attempts: {url}: {err}")


_BLOSC = None


def blosc_decode(raw: bytes, n: int = NLAT * NLON, dtype: str = "<f4") -> np.ndarray:
    """Decode one blosc chunk with the libblosc bundled in rasterio's wheel (thread-safe ``_ctx`` call)."""
    global _BLOSC
    if _BLOSC is None:
        import rasterio

        lib = ctypes.CDLL(sorted(glob.glob(os.path.dirname(rasterio.__file__) + ".libs/libblosc-*.so*"))[0])
        lib.blosc_decompress_ctx.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _BLOSC = lib
    out = np.empty(n, dtype)
    got = _BLOSC.blosc_decompress_ctx(raw, out.ctypes.data, out.nbytes, 1)
    if got != out.nbytes:
        raise ValueError(f"blosc decoded {got} of {out.nbytes} bytes")
    return out


def store_attrs(fetch) -> dict:
    return json.loads(fetch(ARCO + ".zattrs"))


# ------------------------------------------------------------------------------------------- file writing
def _publish(final: Path, write) -> bool:
    """write(partial_path) into a unique .partial, then rename to ``final`` only if it does not exist yet."""
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(f"{final.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial")
    write(part)
    if final.exists():  # another run finished first: keep its file, leave ours as .partial
        return False
    os.rename(part, final)
    return True


def write_month(final: Path, data: np.ndarray, hour0: int, lat: np.ndarray, lon: np.ndarray, attrs: dict) -> bool:
    import h5py

    def w(p):
        with h5py.File(p, "w") as f:
            d = f.create_dataset("data", data=data.astype(np.float32), chunks=(1,) + data.shape[1:],
                                 compression="gzip", compression_opts=1, shuffle=True)
            f.create_dataset("hour", data=hour0 + np.arange(len(data), dtype=np.int64))
            f.create_dataset("lat", data=np.asarray(lat, np.float64))
            f.create_dataset("lon", data=np.asarray(lon, np.float64))
            for k in ("hour", "lat", "lon"):
                f[k].make_scale(k)
            for i, k in enumerate(("hour", "lat", "lon")):
                d.dims[i].attach_scale(f[k])
            f["hour"].attrs["units"] = "hours since 1900-01-01 00:00:00 UTC"
            for k, v in attrs.items():
                f.attrs[k] = v
    return _publish(final, w)


# ---------------------------------------------------------------------------------------------- download
def download_month(var: str, month: str, box: dict, era5_dir: Path, fetch, decode, streams: int,
                   first_hour: int, last_hour: int, final_last_hour: int) -> dict:
    """One variable-month: skip if present, else fetch every hour (``streams`` threads), crop, write."""
    final = era5_dir / var / f"{var}_{month}.h5"
    if final.exists():
        return {"var": var, "month": month, "status": "skip"}
    a, b = month_hours(month)
    a, b = max(a, first_hour), min(b, last_hour + 1)
    if b <= a:
        return {"var": var, "month": month, "status": "empty"}
    rows, cols, lat, lon = box_indices(box)
    data = np.empty((b - a, len(rows), len(cols)), np.float32)

    def one(k):
        data[k] = crop(decode(fetch(f"{ARCO}{VARS[var]}/{a + k}.0.0")).reshape(NLAT, NLON), rows, cols)

    with ThreadPoolExecutor(streams) as ex:
        list(ex.map(one, range(b - a)))  # any failure propagates: nothing is written for this month
    nan_hours = int(np.isnan(data).all(axis=(1, 2)).sum())
    attrs = {"variable": VARS[var], "short_name": var, "source": ARCO, "hours": b - a, "all_nan_hours": nan_hours,
             "era5t_hours": int(max(0, b - max(a, final_last_hour + 1))), "final_last_hour": final_last_hour,
             "downloaded_utc": pd.Timestamp.now("UTC").isoformat()}
    wrote = write_month(final, data, a, lat, lon, attrs)
    return {"var": var, "month": month, "status": "written" if wrote else "raced", "hours": b - a,
            "all_nan_hours": nan_hours}


def download_tile(cell: tuple[int, int], dem_dir: Path, fetch) -> str:
    name = tile_name(*cell)
    final, absent = dem_dir / f"{name}.tif", dem_dir / f"{name}.absent"
    if final.exists() or absent.exists():
        return "skip"
    try:
        raw = fetch(f"{DEM_BASE}{name}/{name}.tif")
    except NotFound:
        _publish(absent, lambda p: p.write_text("no such tile in the bucket (HTTP 403/404)\n"))
        return "absent"
    _publish(final, lambda p: p.write_bytes(raw))
    return "written"


def _try_tile(cell, dem_dir, fetch) -> str:
    try:
        return download_tile(cell, dem_dir, fetch)
    except Exception:  # noqa: BLE001 - counted as failed; the next run retries it
        return "failed"


def load_fixes(path: Path) -> pd.DataFrame:
    """Good raw fixes of one bird: t (datetime64[ns], UTC), lat, lon, hae; sorted, unique times."""
    import pyarrow.parquet as pq

    from . import gps

    cols = ["timestamp", gps.LON, gps.LAT, gps.ALT, gps.HACC, "eobs:status", "gps:satellite-count"]
    have = set(pq.read_schema(path).names)
    d = gps.good_fixes(pq.read_table(path, columns=[c for c in cols if c in have]).to_pandas())
    d = d[d[gps.LAT].abs() > 0.01]
    out = pd.DataFrame({"t": d["timestamp"].astype("datetime64[ns]").to_numpy(), "lat": d[gps.LAT].to_numpy(float),
                        "lon": d[gps.LON].to_numpy(float),
                        "hae": d[gps.ALT].to_numpy(float) if gps.ALT in d else np.nan})
    return out.sort_values("t", kind="stable").drop_duplicates("t").reset_index(drop=True)


def _all_fix_coords(raw_gps: Path):
    lat, lon = [], []
    for f in sorted(raw_gps.glob("*.parquet")):
        d = load_fixes(f)
        lat.append(d["lat"].to_numpy())
        lon.append(d["lon"].to_numpy())
    return np.concatenate(lat), np.concatenate(lon)


def download(raw_gps: Path, env: Path, what: list[str], variables: list[str], start: str, end: str,
             streams: int, max_mbps: float, dem_max: int | None, fetch=None, decode=None) -> dict:
    fetch = fetch or Fetcher(max_mbps)
    decode = decode or blosc_decode
    t0 = time.time()
    lat, lon = _all_fix_coords(raw_gps)
    out: dict = {}
    if "era5" in what:
        era5 = env / "era5"
        bfile = era5 / "box.json"
        if bfile.exists():
            box = json.loads(bfile.read_text())
        else:
            box = box_from_fixes(lat, lon)
            _publish(bfile, lambda p: p.write_text(json.dumps(box)))
        sa = store_attrs(fetch)
        final_last = int(hours_since_1900([np.datetime64(sa["valid_time_stop"])])[0]) + 23
        store_last = int(hours_since_1900([np.datetime64(sa["valid_time_stop_era5t"])])[0]) - 1
        first = int(hours_since_1900([np.datetime64(start)])[0])
        last = min(int(hours_since_1900([np.datetime64(end)])[0]) + 23, store_last)
        rows, cols, _, _ = box_indices(box)
        print(f"ERA5 box {box5(box)}; grid {len(rows)} x {len(cols)}; hours {first}..{last}; "
              f"final ERA5 to {sa['valid_time_stop']}, ERA5T to {sa['valid_time_stop_era5t']}", flush=True)
        months = [str(p) for p in pd.period_range(pd.Period(start, "M"), pd.Period(end, "M"), freq="M")]
        res = []
        for v in variables:
            for m in months:
                b0 = getattr(fetch, "bytes", 0)
                tm = time.time()
                try:
                    r = download_month(v, m, box, era5, fetch, decode, streams, first, last, final_last)
                except Exception as e:  # noqa: BLE001 - one failed month must not stop the others; rerun picks it up
                    r = {"var": v, "month": m, "status": "failed", "error": str(e)[:300]}
                dt = time.time() - tm
                r["mb"] = round((getattr(fetch, "bytes", 0) - b0) / 1e6, 1)
                r["mb_per_s"] = round(r["mb"] / dt, 1) if dt > 0 and r["mb"] else 0.0
                print(json.dumps(r), flush=True)
                res.append(r)
        out["era5"] = res
    if "dem" in what:
        cells = tiles_for_fixes(lat, lon)
        if dem_max is not None:  # test mode: a spread of tiles including the highest-latitude one
            idx = np.unique(np.linspace(len(cells) - 1, 0, dem_max).round().astype(int))
            cells = [cells[i] for i in idx]
        print(f"DEM tiles: {len(cells)}", flush=True)
        b0 = getattr(fetch, "bytes", 0)
        with ThreadPoolExecutor(streams) as ex:
            st = list(ex.map(lambda c: _try_tile(c, env / "dem", fetch), cells))
        out["dem"] = {s: st.count(s) for s in set(st)}
        out["dem"]["mb"] = round((getattr(fetch, "bytes", 0) - b0) / 1e6, 1)
        print(json.dumps({"dem": out["dem"]}), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["mb_total"] = round(getattr(fetch, "bytes", 0) / 1e6, 1)
    out["failed"] = sum(r["status"] == "failed" for r in out.get("era5", [])) + out.get("dem", {}).get("failed", 0)
    print(json.dumps({"seconds": out["seconds"], "mb_total": out["mb_total"], "failed": out["failed"]}), flush=True)
    return out


# --------------------------------------------------------------------------------------------- join: DEM
def rasterio_reader(path: Path):
    import rasterio

    with rasterio.open(path) as r:
        a = r.read(1).astype(np.float32)
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
        t = r.transform
        return a, (t.a, t.b, t.c, t.d, t.e, t.f)


class DEM:
    """Bilinear sampling across GLO-30 tiles (each tile's own affine transform, pixel centres at +0.5)."""

    def __init__(self, dem_dir: Path, reader=rasterio_reader, max_tiles: int = 12):
        self.dir, self.reader, self.cache, self.max = Path(dem_dir), reader, OrderedDict(), max_tiles

    def _tile(self, cell):
        if cell in self.cache:
            self.cache.move_to_end(cell)
        else:
            p = self.dir / f"{tile_name(*cell)}.tif"
            self.cache[cell] = self.reader(p) if p.exists() else None
            if len(self.cache) > self.max:  # bounded memory: least recently used tile is dropped from RAM
                self.cache.popitem(last=False)
        return self.cache[cell]

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        shape, lat, lon = lat.shape, lat.ravel(), lon.ravel()
        out = np.full(lat.shape, np.nan)
        ok = np.isfinite(lat) & np.isfinite(lon)
        ky, kx = np.floor(np.where(ok, lat, 0)).astype(int), np.floor(np.where(ok, lon, 0)).astype(int)
        key = ky * 1000 + kx
        for k in np.unique(key[ok]):
            sel = np.flatnonzero(ok & (key == k))
            cell = (int(ky[sel[0]]), int(kx[sel[0]]))
            t = self._tile(cell)
            if t is None:
                continue
            a, (ta, tb, tc, td, te, tf) = t
            col = (lon[sel] - tc) / ta - 0.5  # no rotation in GLO-30 (tb = td = 0)
            row = (lat[sel] - tf) / te - 0.5
            h, w = a.shape
            c0 = np.clip(np.floor(col).astype(int), 0, w - 2)
            r0 = np.clip(np.floor(row).astype(int), 0, h - 2)
            fx, fy = np.clip(col - c0, 0, 1), np.clip(row - r0, 0, 1)
            out[sel] = ((a[r0, c0] * (1 - fx) + a[r0, c0 + 1] * fx) * (1 - fy)
                        + (a[r0 + 1, c0] * (1 - fx) + a[r0 + 1, c0 + 1] * fx) * fy)
        out[out < DEM_NODATA_BELOW] = np.nan
        return out.reshape(shape)


def _offset(lat, lon, east_m, north_m):
    return lat + np.degrees(north_m / R_EARTH), lon + np.degrees(east_m / (R_EARTH * np.cos(np.radians(lat))))


def terrain(dem: DEM, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    """elev_m, dzdx, dzdy (east, north; m/m), slope_deg, aspect_deg, relief_m at each point.

    Points are processed cell by cell so the tiles in use (a cell and its neighbours) stay in the DEM cache."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    key = np.floor(lat).astype(np.int64) * 1000 + np.floor(lon).astype(np.int64)
    parts = [_terrain(dem, lat[i], lon[i]).set_index(i) for i in
             (np.flatnonzero(key == k) for k in np.unique(key))]
    return pd.concat(parts).sort_index() if parts else _terrain(dem, lat, lon)


def _terrain(dem: DEM, lat: np.ndarray, lon: np.ndarray) -> pd.DataFrame:
    z = dem.sample(lat, lon)
    zs = [dem.sample(*_offset(lat, lon, e, n)) for e, n in ((GRAD_M, 0), (-GRAD_M, 0), (0, GRAD_M), (0, -GRAD_M))]
    gx, gy = (zs[0] - zs[1]) / (2 * GRAD_M), (zs[2] - zs[3]) / (2 * GRAD_M)
    g = np.hypot(gx, gy)
    aspect = np.where(g >= FLAT_GRAD, np.degrees(np.arctan2(-gx, -gy)) % 360, np.nan)
    s = np.arange(-RELIEF_M, RELIEF_M + 1e-9, RELIEF_STEP_M)
    ee, nn = np.meshgrid(s, s)
    disc = np.hypot(ee, nn) <= RELIEF_M + 1e-9
    relief = np.full(len(lat), np.nan)
    for i0 in range(0, len(lat), 20000):  # bounded memory: 20000 points x ~317 offsets
        la, lo = lat[i0:i0 + 20000, None], lon[i0:i0 + 20000, None]
        zz = dem.sample(*_offset(la, lo, ee[disc][None], nn[disc][None]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (absent tiles) give NaN
            r = np.nanmax(zz, axis=1) - np.nanmin(zz, axis=1)
        r[np.isnan(zz).mean(axis=1) > 0.1] = np.nan
        relief[i0:i0 + 20000] = r
    return pd.DataFrame({"elev_m": z, "dzdx": gx, "dzdy": gy, "slope_deg": np.degrees(np.arctan(g)),
                         "aspect_deg": aspect, "relief_m": relief})


# ------------------------------------------------------------------------------------------ join: positions
def window_positions(meta: pd.DataFrame, fixes: pd.DataFrame, window_s: float = 10.0,
                     near_s: float = NEAR_S) -> pd.DataFrame:
    """Per meta row (same order): lat, lon, hae, mid_hour (hours since 1900), pos_source, pos_dt_s."""
    t0 = meta["t_start"].astype("datetime64[ns]").to_numpy().astype(np.int64)
    ws = int(round(window_s * 1e9))
    mid = t0 + ws // 2
    n = len(meta)
    lat, lon, hae = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    src, dts = np.zeros(n, np.int8), np.full(n, np.nan)
    if len(fixes):
        ft = fixes["t"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        fla, flo, fh = (fixes[c].to_numpy(float) for c in ("lat", "lon", "hae"))
        lo_i, hi_i = np.searchsorted(ft, t0, "left"), np.searchsorted(ft, t0 + ws, "left")
        cnt = hi_i - lo_i
        cs = lambda x: np.r_[0.0, np.cumsum(np.nan_to_num(x))]  # noqa: E731
        hfin = np.r_[0, np.cumsum(np.isfinite(fh))]
        own = cnt > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            for arr, v in ((lat, fla), (lon, flo)):
                c = cs(v)
                arr[own] = (c[hi_i] - c[lo_i])[own] / cnt[own]
            ch, nh = cs(fh), hfin[hi_i] - hfin[lo_i]
            hae[own] = np.where(nh[own] > 0, (ch[hi_i] - ch[lo_i])[own] / nh[own], np.nan)
        src[own], dts[own] = 1, 0.0
        j = np.searchsorted(ft, mid)
        jl, jr = np.clip(j - 1, 0, len(ft) - 1), np.clip(j, 0, len(ft) - 1)
        dl, dr = np.abs(mid - ft[jl]), np.abs(ft[jr] - mid)
        jb = np.where(dr < dl, jr, jl)
        db = np.minimum(dl, dr) / 1e9
        near = ~own & (db <= near_s)
        lat[near], lon[near], hae[near] = fla[jb[near]], flo[jb[near]], fh[jb[near]]
        src[near], dts[near] = 2, db[near]
    return pd.DataFrame({"lat": lat, "lon": lon, "hae": hae, "mid_hour": hours_since_1900(mid.astype("datetime64[ns]")),
                         "pos_source": src, "pos_dt_s": dts.astype(np.float32)})


# ------------------------------------------------------------------------------------------- join: ERA5
def _read_month(path: Path, first_only: bool = False):
    import h5py

    with h5py.File(path, "r") as f:
        d = f["data"][:1] if first_only else f["data"][:]
        return d, int(f["hour"][0]), f["lat"][:], f["lon"][:], int(f.attrs.get("final_last_hour", -1))


def era5_sample(era5_dir: Path, var: str, hour: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Bilinear in space and linear in time; returns (values, flags). Month files are read once each."""
    n = len(hour)
    val, flag = np.full(n, np.nan), np.zeros(n, np.int32)
    ok = np.isfinite(hour) & np.isfinite(lat) & np.isfinite(lon)
    h0 = np.floor(np.where(ok, hour, 0)).astype(np.int64)
    files = {p.stem.split("_")[-1]: p for p in sorted((era5_dir / var).glob(f"{var}_*.h5"))}
    done = np.zeros(n, bool)
    for m in sorted(files):
        a, b = month_hours(m)
        sel = np.flatnonzero(ok & (h0 >= a) & (h0 < b))
        if not len(sel):
            continue
        d, hour0, flat, flon, _ = _read_month(files[m])
        nxt = str(pd.Period(m, "M") + 1)
        if nxt in files:
            d = np.concatenate([d, _read_month(files[nxt], first_only=True)[0]])
        it = h0[sel] - hour0
        ft = hour[sel] - h0[sel]
        tin = (it >= 0) & (it + 1 < len(d))
        fy = (flat[0] - lat[sel]) / RES
        fx = ((lon[sel] - flon[0]) % 360) / RES
        y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
        sin = (y0 >= 0) & (y0 + 1 < len(flat)) & (x0 >= 0) & (x0 + 1 < len(flon))
        flag[sel[~tin]] |= FLAGS["era5_no_time"]
        flag[sel[tin & ~sin]] |= FLAGS["era5_outside_box"]
        g = tin & sin
        s, it, y0, x0 = sel[g], it[g], y0[g], x0[g]
        wy, wx, wt = (fy[g] - y0), (fx[g] - x0), ft[g]

        def plane(k):
            return ((d[k, y0, x0] * (1 - wx) + d[k, y0, x0 + 1] * wx) * (1 - wy)
                    + (d[k, y0 + 1, x0] * (1 - wx) + d[k, y0 + 1, x0 + 1] * wx) * wy)
        val[s] = plane(it) * (1 - wt) + plane(it + 1) * wt
        done[sel] = True
    flag[ok & ~done] |= FLAGS["era5_no_time"]
    flag[ok & done & np.isnan(val) & (flag == 0)] |= FLAGS["era5_nan"]
    return val, flag


def final_last_hour(era5_dir: Path) -> int:
    for p in sorted(era5_dir.glob("*/*.h5")):
        return _read_month(p, first_only=True)[4]
    return -1


# --------------------------------------------------------------------------------------------- join: physics
def air_density(z_m: np.ndarray) -> np.ndarray:
    """ISA troposphere density (kg m-3) at elevation z (m); sea level where z is NaN."""
    z = np.nan_to_num(np.asarray(z_m, float), nan=0.0)
    return 1.225 * np.clip(1 - 2.25577e-5 * z, 0.1, None) ** 4.2559


def uplift(u: np.ndarray, v: np.ndarray, dzdx: np.ndarray, dzdy: np.ndarray) -> np.ndarray:
    """Orographic uplift w_o = V . grad(z) (m/s): positive when the wind blows towards higher ground."""
    return np.asarray(u) * np.asarray(dzdx) + np.asarray(v) * np.asarray(dzdy)


def wstar(ishf: np.ndarray, blh: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Deardorff convective velocity scale (m/s) from ERA5 ishf (W m-2, positive downward) and blh (m)."""
    h_up = -np.asarray(ishf, float)
    with np.errstate(invalid="ignore"):
        w = np.cbrt(G / THETA_REF * np.clip(h_up, 0, None) / (rho * CP) * np.asarray(blh, float))
    return np.where(np.isfinite(h_up) & np.isfinite(blh), w, np.nan)


# ------------------------------------------------------------------------------------------------- join
OUT_COLS = ["pos_source", "pos_dt_s", "u10_ms", "v10_ms", "u100_ms", "v100_ms", "ws10_ms", "ws100_ms", "blh_m",
            "shf_up_wm2", "wstar_ms", "elev_m", "slope_deg", "aspect_deg", "relief_m", "hag_m", "wo_ms", "era5t",
            "env_flag"]
UNITS = {"pos_source": "0 none, 1 mean of the window's own fixes, 2 nearest fix within +-30 min",
         "pos_dt_s": "s, |window mid-time - time of the fix used| (0 for own fixes)",
         "u10_ms": "m/s eastward 10 m", "v10_ms": "m/s northward 10 m", "u100_ms": "m/s eastward 100 m",
         "v100_ms": "m/s northward 100 m", "ws10_ms": "m/s", "ws100_ms": "m/s", "blh_m": "m",
         "shf_up_wm2": "W m-2 upward-positive surface sensible heat flux (= -ERA5 ishf)",
         "wstar_ms": "m/s Deardorff convective velocity scale", "elev_m": "m GLO-30 (EGM2008 orthometric)",
         "slope_deg": "deg", "aspect_deg": "deg clockwise from north, downslope azimuth; NaN where flat",
         "relief_m": "m, max - min elevation within 1000 m", "hag_m": "m, GPS height above ellipsoid - elev_m "
         "(NOT geoid-corrected)", "wo_ms": "m/s orographic uplift u100 dz/dx + v100 dz/dy",
         "era5t": "bool, window time after the final-ERA5 end (preliminary ERA5T values)",
         "env_flag": "int bitmask, see flags"}


def _bird_positions_terrain(args):
    bird, windows, raw_gps, dem_dir, window_s, reader = args
    import pyarrow.parquet as pq

    meta = pq.read_table(windows / bird / "meta.parquet", columns=["t_start", "gps_fixes"]).to_pandas()
    f = raw_gps / f"{bird}.parquet"
    fixes = load_fixes(f) if f.exists() else pd.DataFrame(columns=["t", "lat", "lon", "hae"])
    pos = window_positions(meta, fixes, window_s)
    has = pos["pos_source"].to_numpy() > 0
    ter = pd.DataFrame(np.nan, index=pos.index, columns=["elev_m", "dzdx", "dzdy", "slope_deg", "aspect_deg",
                                                         "relief_m"])
    if has.any():
        u = pos.loc[has, ["lat", "lon"]].drop_duplicates()
        t = terrain(DEM(dem_dir, reader), u["lat"].to_numpy(), u["lon"].to_numpy())
        t.index = pd.MultiIndex.from_frame(u)
        key = pd.MultiIndex.from_frame(pos.loc[has, ["lat", "lon"]])
        ter.loc[has] = t.reindex(key).to_numpy()
    own_vs_meta = float(((pos["pos_source"] == 1).to_numpy() == (meta["gps_fixes"].to_numpy() > 0)).mean())
    return bird, pd.concat([pos, ter], axis=1), own_vs_meta


def join(windows: Path, raw_gps: Path, env: Path, out: Path, birds: list[str] | None = None, workers: int = 1,
         window_s: float = 10.0, reader=rasterio_reader) -> dict:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    birds = birds or sorted(p.parent.name for p in windows.glob("*/meta.parquet"))
    jobs = [(b, windows, raw_gps, env / "dem", window_s, reader) for b in birds]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_bird_positions_terrain, jobs))
    else:
        res = [_bird_positions_terrain(j) for j in jobs]
    sizes = [len(r[1]) for r in res]
    allp = pd.concat([r[1] for r in res], ignore_index=True)
    era = {}
    for v in VARS:
        era[v] = era5_sample(env / "era5", v, allp["mid_hour"].to_numpy(), allp["lat"].to_numpy(),
                             allp["lon"].to_numpy())
        print(f"ERA5 {v}: sampled", flush=True)
    flh = final_last_hour(env / "era5")
    flag = np.zeros(len(allp), np.int32)
    for v in VARS:
        flag |= era[v][1]
    src = allp["pos_source"].to_numpy()
    flag[src == 0] = FLAGS["no_position"]
    flag[(src > 0) & np.isnan(allp["elev_m"].to_numpy())] |= FLAGS["dem_missing"]
    flag[(src > 0) & np.isnan(allp["hae"].to_numpy())] |= FLAGS["no_height"]
    u10, v10, u100, v100 = (era[k][0] for k in ("u10", "v10", "u100", "v100"))
    feat = pd.DataFrame({
        "pos_source": src, "pos_dt_s": allp["pos_dt_s"].to_numpy(),
        "u10_ms": u10, "v10_ms": v10, "u100_ms": u100, "v100_ms": v100,
        "ws10_ms": np.hypot(u10, v10), "ws100_ms": np.hypot(u100, v100), "blh_m": era["blh"][0],
        "shf_up_wm2": -era["ishf"][0],
        "wstar_ms": wstar(era["ishf"][0], era["blh"][0], air_density(allp["elev_m"].to_numpy())),
        "elev_m": allp["elev_m"], "slope_deg": allp["slope_deg"], "aspect_deg": allp["aspect_deg"],
        "relief_m": allp["relief_m"], "hag_m": allp["hae"] - allp["elev_m"],
        "wo_ms": uplift(u100, v100, allp["dzdx"].to_numpy(), allp["dzdy"].to_numpy()),
        "era5t": (src > 0) & (allp["mid_hour"].to_numpy() > flh) if flh >= 0 else False,
        "env_flag": flag,
    })
    for c in OUT_COLS:
        if c not in ("pos_source", "era5t", "env_flag"):
            feat[c] = feat[c].astype(np.float32)
    cov, i0 = {}, 0
    for (bird, _, own_vs), n in zip(res, sizes):
        part = feat.iloc[i0:i0 + n][OUT_COLS].reset_index(drop=True)
        i0 += n
        part.to_parquet(out / f"{bird}.parquet")
        cov[bird] = {"windows": n, "own_fix_matches_meta_gps_fixes": round(own_vs, 4)}
    readme = {
        "rows": "1:1 with windows_v4/<bird>/meta.parquet, same order; no latitude, longitude or time columns",
        "columns": UNITS, "flags": FLAGS,
        "formulas": {
            "wstar_ms": "w* = (g/theta_ref * H/(rho*c_p) * blh)^(1/3), H = -ishf (ERA5 positive downward), "
                        "w* = 0 if H <= 0", "rho": "ISA: 1.225*(1 - 2.25577e-5 z)^4.2559, z = elev_m (0 if NaN)",
            "wo_ms": "u100*dz/dx + v100*dz/dy, gradient by central differences over +-90 m",
            "interpolation": "ERA5 bilinear in space (0.25 deg), linear in time at t_start + window_s/2",
            "hag_m": "height-above-ellipsoid (mean of the fixes used) - GLO-30 elevation"},
        "constants": {"g": G, "theta_ref_K": THETA_REF, "c_p": CP, "grad_m": GRAD_M, "relief_m": RELIEF_M,
                      "relief_step_m": RELIEF_STEP_M, "flat_gradient": FLAT_GRAD, "near_s": NEAR_S,
                      "window_s": window_s},
        "geoid_corrected": False,
        "geoid_note": "GLO-30 heights are EGM2008 orthometric; GPS heights are ellipsoidal. No geoid grid is "
                      "available offline in the environment, so hag_m = true height above ground + N (EGM2008 "
                      "undulation at the point; N is negative over most of North America).",
        "era5_final_last_hour_since_1900": flh,
        "sources": {"era5": ARCO, "dem": DEM_BASE},
        "coverage": {"windows": int(len(feat)), "birds": len(birds),
                     "pos_source_counts": {int(k): int(v) for k, v in feat["pos_source"].value_counts().items()},
                     "non_nan_fraction": {c: round(float(feat[c].notna().mean()), 4) for c in OUT_COLS
                                          if feat[c].dtype.kind == "f"},
                     "flag_counts": {k: int(((flag & b) > 0).sum()) for k, b in FLAGS.items()},
                     "era5t_windows": int(np.sum(feat["era5t"])), "per_bird": cov},
    }
    (out / "README.json").write_text(json.dumps(readme, indent=1))
    return readme


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="ERA5 variable-months and GLO-30 tiles (resumable)")
    d.add_argument("--raw-gps", type=Path, required=True)
    d.add_argument("--env", type=Path, required=True, help="env root; writes era5/ and dem/ under it")
    d.add_argument("--what", default="era5,dem")
    d.add_argument("--vars", default=",".join(VARS))
    d.add_argument("--start", default="2024-06-01")
    d.add_argument("--end", default="2026-11-20", help="last day (inclusive)")
    d.add_argument("--streams", type=int, default=8)
    d.add_argument("--max-mbps", type=float, default=0.0, help="shared rate cap in MB/s (0 = none)")
    d.add_argument("--dem-max", type=int, default=None, help="test only: fetch this many tiles")
    j = sub.add_parser("join", help="per-window features aligned with meta.parquet")
    j.add_argument("--windows", type=Path, required=True)
    j.add_argument("--raw-gps", type=Path, required=True)
    j.add_argument("--env", type=Path, required=True)
    j.add_argument("--out", type=Path, required=True)
    j.add_argument("--birds", default=None, help="comma-separated subset")
    j.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    j.add_argument("--window-s", type=float, default=10.0)
    a = ap.parse_args(argv)
    if a.cmd == "download":
        r = download(a.raw_gps, a.env, a.what.split(","), a.vars.split(","), a.start, a.end, a.streams, a.max_mbps,
                     a.dem_max)
        if r["failed"]:
            raise SystemExit(f"{r['failed']} variable-months/tiles failed; rerun to fetch them")
    else:
        r = join(a.windows, a.raw_gps, a.env, a.out, a.birds.split(",") if a.birds else None, a.workers,
                 a.window_s)
        print(json.dumps({k: v for k, v in r["coverage"].items() if k != "per_bird"}, indent=1))


if __name__ == "__main__":
    main()
