"""Synthetic tests for aquilacl.env (no network, no real coordinates: grids are placed in the open South Pacific
or on the Greenwich meridian at the equator)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from aquilacl import env

LAT0, LON0 = -36.5, -151.5  # synthetic open-ocean point; tiles cover lat -37..-36, lon -152..-150


def _plane_tiles(k=0.05):
    """Two 1-degree tiles with uneven pixel spacing (lat 1/600 deg, lon 1/300 deg) holding z = 1000 + k * east_m."""
    tiles = {}
    for lon_floor in (-152, -151):
        h, w = 600, 300
        lat_c = -36 - (np.arange(h) + 0.5) / h
        lon_c = lon_floor + (np.arange(w) + 0.5) / w
        east = np.radians(lon_c - LON0) * env.R_EARTH * math.cos(math.radians(LAT0))
        a = np.repeat((1000 + k * east)[None, :], h, axis=0).astype(np.float32)
        tiles[env.tile_name(-37, lon_floor)] = (a, (1 / w, 0.0, float(lon_floor), 0.0, -1 / h, -36.0))
    return tiles


def _dem(tmp_path, tiles):
    d = tmp_path / "dem"
    d.mkdir(parents=True)
    for name in tiles:
        (d / f"{name}.tif").write_bytes(b"")
    return d, (lambda p: tiles[p.stem])


def _write_months(era5, var, field, months_hours, lat, lon):
    for m, (a, b) in months_hours.items():
        hrs = np.arange(a, b)
        data = np.stack([field(h, lat[:, None], lon[None, :]) for h in hrs]).astype(np.float32)
        env.write_month(era5 / var / f"{var}_{m}.h5", data, a, lat, lon, {"final_last_hour": a + 1})


def test_grid_indexing_crop_wrap_and_names():
    box = {"lat0": 0, "lat1": 1, "lon0": -1, "lon1": 1}
    rows, cols, lat, lon = env.box_indices(box)
    assert lat.tolist() == [1.0, 0.75, 0.5, 0.25, 0.0]  # ERA5 latitude descends
    assert lon.tolist() == list(np.arange(-1, 1.001, 0.25))
    assert cols.tolist() == [1436, 1437, 1438, 1439, 0, 1, 2, 3, 4]  # wraps across 0 deg (index 0 = 0 E)
    full = np.arange(env.NLAT * env.NLON, dtype=np.float64).reshape(env.NLAT, env.NLON)
    c = env.crop(full, rows, cols)
    assert c.shape == (5, 9) and c[0, 0] == full[356, 1436] and c[-1, 4] == full[360, 0]
    assert env.hours_since_1900([np.datetime64("1900-01-02T00:00")])[0] == 24
    assert env.month_hours("1900-02") == (744, 744 + 28 * 24)
    assert env.tile_name(-37, -152) == "Copernicus_DSM_COG_10_S37_00_W152_00_DEM"
    assert env.tile_name(5, 7) == "Copernicus_DSM_COG_10_N05_00_E007_00_DEM"
    cells = env.tiles_for_fixes(np.array([LAT0]), np.array([-151.9999]))
    assert (-37, -153) in cells and (-37, -152) in cells  # buffer reaches the neighbouring cell
    assert "lat -40..-35" in env.box5({"lat0": -37, "lat1": -36, "lon0": -152, "lon1": -150})


def test_download_resume_partial_rename_no_overwrite(tmp_path):
    box = {"lat0": 0, "lat1": 1, "lon0": -1, "lon1": 1}
    a, _ = env.month_hours("2025-02")
    calls, fail = [], {"hour": a + 3}

    def fetch(url):
        h = int(url.rsplit("/", 1)[1].split(".")[0])
        calls.append(h)
        if h == fail["hour"]:
            raise OSError("transient")
        return str(h).encode()

    decode = lambda raw: np.full(env.NLAT * env.NLON, float(raw.decode()), np.float32)  # noqa: E731
    era5 = tmp_path / "era5"
    (era5 / "u10").mkdir(parents=True)
    foreign = era5 / "u10" / "u10_2025-02.h5.999.deadbeef.partial"
    foreign.write_bytes(b"stale")
    args = ("u10", "2025-02", box, era5, fetch, decode, 2, a, a + 5, a)
    with pytest.raises(OSError):
        env.download_month(*args)
    final = era5 / "u10" / "u10_2025-02.h5"
    assert not final.exists() and sorted(p.name for p in (era5 / "u10").iterdir()) == [foreign.name]
    fail["hour"] = None
    r = env.download_month(*args)
    assert r["status"] == "written" and r["hours"] == 6 and final.exists()
    d, hour0, lat, lon, flh = env._read_month(final)
    assert hour0 == a and d.shape == (6, 5, 9) and np.all(d[:, 0, 0] == np.arange(a, a + 6)) and flh == a
    assert foreign.read_bytes() == b"stale"  # never touched
    calls.clear()
    assert env.download_month(*args)["status"] == "skip" and calls == []
    # DEM: missing tile -> .absent marker, existing tile untouched, reruns fetch nothing
    dem = tmp_path / "dem"

    def dem_fetch(url):
        calls.append(url)
        if "S37" in url:
            raise env.NotFound(url)
        return b"tif-bytes"

    assert env.download_tile((-37, -152), dem, dem_fetch) == "absent"
    assert env.download_tile((5, 7), dem, dem_fetch) == "written"
    assert (dem / f"{env.tile_name(5, 7)}.tif").read_bytes() == b"tif-bytes"
    calls.clear()
    assert env.download_tile((-37, -152), dem, dem_fetch) == "skip" == env.download_tile((5, 7), dem, dem_fetch)
    assert calls == [] and not list(dem.glob("*.partial"))

    def down(url):
        raise OSError("outage")

    assert env._try_tile((6, 7), dem, down) == "failed" and not list(dem.glob(f"{env.tile_name(6, 7)}*"))


def test_era5_bilinear_linear_time_across_month_boundary(tmp_path):
    lat, lon = 0.5 - np.arange(5) * 0.25, -1 + np.arange(9) * 0.25
    ja, jb = env.month_hours("2025-01")
    field = lambda h, la, lo: 2 * la + 3 * lo + 0.5 * (h - jb)  # noqa: E731 - linear in all three
    _write_months(tmp_path / "era5", "u10", field, {"2025-01": (jb - 3, jb), "2025-02": (jb, jb + 3)}, lat, lon)
    hour = np.array([jb - 0.5, jb - 2.25, jb + 1.5, jb - 0.5, jb + 10.0])
    plat = np.array([0.1, -0.3, 0.37, 2.0, 0.1])
    plon = np.array([0.6, -0.9, 0.11, 0.0, 0.6])
    v, f = env.era5_sample(tmp_path / "era5", "u10", hour, plat, plon)
    np.testing.assert_allclose(v[:3], field(hour[:3], plat[:3], plon[:3]), atol=1e-5)
    assert np.isnan(v[3]) and f[3] == env.FLAGS["era5_outside_box"]
    assert np.isnan(v[4]) and f[4] == env.FLAGS["era5_no_time"]


def test_wstar_and_density():
    w = env.wstar(np.array([-200.0, 50.0, np.nan]), np.array([1000.0, 1000.0, 1000.0]), np.array([1.0, 1.0, 1.0]))
    assert w[0] == pytest.approx((9.81 / 300.0 * 200.0 / 1004.7 * 1000.0) ** (1 / 3))
    assert w[1] == 0.0 and np.isnan(w[2])  # downward flux (stable) -> 0; missing -> NaN
    rho = env.air_density(np.array([0.0, 1500.0, np.nan]))
    assert rho[0] == pytest.approx(1.225) and 1.04 < rho[1] < 1.07 and rho[2] == pytest.approx(1.225)


def test_terrain_plane_slope_aspect_relief_and_uplift_sign(tmp_path):
    k = 0.05
    d, reader = _dem(tmp_path, _plane_tiles(k))
    dem = env.DEM(d, reader)
    lat = np.array([LAT0, LAT0, LAT0 + 0.2])
    lon = np.array([LON0, -150.5, -149.5])  # second point in the other tile, third in an absent tile
    t = env.terrain(dem, lat, lon)
    assert t["elev_m"][0] == pytest.approx(1000, abs=1e-3)
    assert t["dzdx"][0] == pytest.approx(k, rel=1e-3) and abs(t["dzdy"][0]) < 1e-6
    assert t["dzdx"][1] == pytest.approx(k, rel=1e-3)
    assert t["slope_deg"][0] == pytest.approx(math.degrees(math.atan(k)), rel=1e-3)
    assert t["aspect_deg"][0] == pytest.approx(270.0, abs=0.01)  # rises to the east -> faces west
    assert t["relief_m"][0] == pytest.approx(2 * env.RELIEF_M * k, rel=1e-3)
    assert t[["elev_m", "slope_deg", "relief_m"]].iloc[2].isna().all()
    gx, gy = t["dzdx"][0], t["dzdy"][0]
    wo = env.uplift(np.array([5.0, -5.0, 0.0, 0.0]), np.array([0.0, 0.0, 3.0, -3.0]), gx, gy)
    assert wo[0] == pytest.approx(5 * k, rel=1e-3) and wo[1] == pytest.approx(-5 * k, rel=1e-3)  # up- / downslope
    assert abs(wo[2]) < 1e-5 and abs(wo[3]) < 1e-5  # wind along the contours: no uplift
    assert env.uplift(0.0, 4.0, 0.0, 0.1) == pytest.approx(0.4)  # southerly wind onto a north-rising slope


def test_window_positions_alignment_and_sources():
    base = pd.Timestamp("2025-03-01 12:00:00")
    fixes = pd.DataFrame({"t": pd.to_datetime([base + pd.Timedelta(seconds=s) for s in (2, 7, 600, 9000)]),
                          "lat": [LAT0, LAT0 + 0.002, LAT0 + 0.01, LAT0 + 0.5],
                          "lon": [LON0, LON0 + 0.004, LON0 + 0.01, LON0 + 0.5],
                          "hae": [1500.0, 1510.0, np.nan, 900.0]})
    t = [base + pd.Timedelta(minutes=8), base, base + pd.Timedelta(hours=1, minutes=30), base + pd.Timedelta(seconds=8)]
    meta = pd.DataFrame({"t_start": pd.to_datetime(t)})  # deliberately not time-sorted
    p = env.window_positions(meta, fixes, 10.0)
    assert len(p) == len(meta)
    assert p["pos_source"].tolist() == [2, 1, 0, 2]
    assert p["lat"][1] == pytest.approx(LAT0 + 0.001) and p["hae"][1] == pytest.approx(1505.0)
    assert p["lat"][3] == pytest.approx(LAT0 + 0.002) and p["pos_dt_s"][3] == pytest.approx(6)  # [8, 18) s: none
    assert p["lat"][0] == pytest.approx(LAT0 + 0.01) and p["pos_dt_s"][0] == pytest.approx(600 - 485)
    assert np.isnan(p["hae"][0]) and np.isnan(p["lat"][2])
    assert p["mid_hour"][1] == pytest.approx(env.hours_since_1900([np.datetime64(base)])[0] + 5 / 3600)


def test_join_end_to_end_no_coordinates(tmp_path):
    bird = "synthetic_a"
    base = pd.Timestamp("2025-01-31 23:20:00")
    t_start = [base + pd.Timedelta(seconds=10 * i) for i in (0, 1)] + [base + pd.Timedelta(hours=5)]
    win = tmp_path / "windows" / bird
    win.mkdir(parents=True)
    pd.DataFrame({"bird": bird, "t_start": pd.to_datetime(t_start), "gps_fixes": [1, 0, 0]}).to_parquet(
        win / "meta.parquet")
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"timestamp": pd.to_datetime([base + pd.Timedelta(seconds=3)]).astype("datetime64[ms]"),
                  "location-long": [LON0], "location-lat": [LAT0], "height-above-ellipsoid": [1300.0],
                  "eobs:horizontal-accuracy-estimate": [5.0], "eobs:status": ["A"],
                  "gps:satellite-count": [8.0]}).to_parquet(raw / f"{bird}.parquet")
    root = tmp_path / "env"
    _, reader = _dem(root, _plane_tiles())
    lat, lon = -36.0 - np.arange(5) * 0.25, -152.0 + np.arange(9) * 0.25
    ja, jb = env.month_hours("2025-01")
    vals = {"u10": 3.0, "v10": 4.0, "u100": 5.0, "v100": 0.0, "blh": 1000.0, "ishf": -200.0}
    for v, c in vals.items():
        _write_months(root / "era5", v, lambda h, la, lo, c=c: c + 0 * la * lo,
                      {"2025-01": (jb - 2, jb), "2025-02": (jb, jb + 2)}, lat, lon)
    out = tmp_path / "features_v1"
    readme = env.join(tmp_path / "windows", raw, root, out, workers=1, reader=reader)
    f = pd.read_parquet(out / f"{bird}.parquet")
    assert len(f) == 3 and list(f.columns) == env.OUT_COLS
    bad = [c for c in f.columns if any(s in c.lower() for s in ("lat", "lon", "time", "timestamp", "t_start"))]
    assert bad == []
    assert f["pos_source"].tolist() == [1, 2, 0] and f["env_flag"][2] == env.FLAGS["no_position"]
    assert f["ws10_ms"][0] == pytest.approx(5.0) and f["wo_ms"][0] == pytest.approx(5 * 0.05, rel=1e-3)
    assert f["hag_m"][0] == pytest.approx(300.0, abs=0.01) and f["env_flag"][0] == 0
    assert f["shf_up_wm2"][0] == pytest.approx(200.0) and f["wstar_ms"][0] > 0
    assert bool(f["era5t"][0]) and f.iloc[2, 2:17].isna().all()
    for key in ("columns", "flags", "formulas", "constants", "coverage", "geoid_corrected"):
        assert key in readme
    txt = (out / "README.json").read_text()
    assert str(LAT0) not in txt and str(LON0) not in txt
    assert json.loads(txt)["coverage"]["per_bird"][bird]["own_fix_matches_meta_gps_fixes"] == 1.0
    with pytest.raises(SystemExit):
        env.join(tmp_path / "windows", raw, root, out, workers=1, reader=reader)
