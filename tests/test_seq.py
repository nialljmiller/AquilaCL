import json

import numpy as np
import pandas as pd
import pytest
import torch

from aquilacl import seq

D = 8
TINY = ["--d-model", "16", "--layers", "1", "--heads", "2", "--ff", "32", "--max-len", "32", "--batch", "4",
        "--steps", "3", "--save-every", "3", "--log-every", "1", "--warmup", "1", "--device", "cpu",
        "--infer-batch", "8", "--dropout", "0"]


def _meta(n_days=3, shift_h=0.0, seed=0):
    """Scheduled windows every 900 s from 05:00 to 19:45 local plus one 20-window continuous bout per day."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day0 = pd.Timestamp("2025-06-01") + pd.Timedelta(days=d) + pd.Timedelta(hours=12)  # 05:00 local = 12:00 UTC
        for s in range(60):
            rows.append((day0 + pd.Timedelta(seconds=900 * s), 0, int(rng.choice([40, 200]))))
        b0 = day0 + pd.Timedelta(hours=int(rng.integers(2, 10)), seconds=450)
        rows += [(b0 + pd.Timedelta(seconds=10 * j), 1, 200) for j in range(20)]
    m = pd.DataFrame(rows, columns=["t_start", "sensitivity_k", "acc_samples"]).sort_values("t_start", kind="stable")
    utc_h = m["t_start"].dt.hour + m["t_start"].dt.minute / 60 + m["t_start"].dt.second / 3600
    m["local_solar_hour"] = (utc_h - 7.0) % 24
    m["t_start"] = m["t_start"] + pd.Timedelta(hours=shift_h)
    m["local_solar_hour"] = (m["local_solar_hour"] + shift_h) % 24
    return m.reset_index(drop=True)


def _setup(tmp_path, shift_h=0.0, nan_birds=(), name="w"):
    birds = [f"b{i:02d}" for i in range(6)]
    split = {"train": birds[:4], "search": birds[4:5], "locked": birds[5:], "method": "test"}
    win, par = tmp_path / name, tmp_path / "parent"
    (par / "embeddings").mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(birds):
        m = _meta(shift_h=shift_h, seed=i)
        (win / b).mkdir(parents=True, exist_ok=True)
        m.to_parquet(win / b / "meta.parquet")
        h = np.random.default_rng(100 + i).normal(size=(len(m), D)) + (m["sensitivity_k"].to_numpy()[:, None])
        if b in nan_birds:
            h[:] = np.nan
        np.savez(par / "embeddings" / f"{b}.npz", h=h.astype(np.float16))
    (win / "b99").mkdir(exist_ok=True)  # a bird without windows: skipped
    _meta().iloc[:0].to_parquet(win / "b99" / "meta.parquet")
    (par / "holdout.json").write_text(json.dumps(split["search"]))
    (par / "orientation.json").write_text("[]")
    (par / "locked.json").write_text(json.dumps(split["locked"]))
    sp = tmp_path / "split.json"
    sp.write_text(json.dumps(split))
    return win, par, sp, birds


def _run(win, par, sp, run, *extra):
    seq.main(["--parent", str(par), "--windows", str(win), "--split", str(sp), "--run", str(run), *TINY, *extra])


def test_outputs_shapes_alignment_and_split(tmp_path):
    win, par, sp, birds = _setup(tmp_path)
    run = tmp_path / "run"
    _run(win, par, sp, run)
    assert json.loads((run / "design.json").read_text()) == {"uses_time": False, "uses_env": False, "level": 2,
                                                             "parent_run": str(par)}
    s = json.loads(sp.read_text())
    assert json.loads((run / "holdout.json").read_text()) == s["search"]
    assert json.loads((run / "locked.json").read_text()) == s["locked"]
    assert not (run / "embeddings" / "b99.npz").exists()
    assert (run / "orientation.json").read_text() == "[]"
    for b in birds:
        m = pd.read_parquet(win / b / "meta.parquet")
        e = np.load(run / "embeddings" / f"{b}.npz")
        assert e["h"].shape == (len(m), D) and "h_notime" not in e.files and np.isfinite(e["h"]).all()
        np.testing.assert_array_equal(e["row"], np.arange(len(m)))
        dy = np.load(run / "day_embeddings" / f"{b}.npz")
        assert dy["n"].sum() == len(m) and dy["h"].shape == (len(dy["day"]), D) and len(dy["day"]) == 3
        bt = np.load(run / "bout_embeddings" / f"{b}.npz")
        assert bt["n"].sum() == (m["sensitivity_k"] == 1).sum() and len(bt["bout"]) == 3
    assert len(pd.read_csv(run / "loss.csv")) == 3


def test_stream_tiles_cover_every_window_inside_its_tile():
    for n in (1, 5, 31, 32, 33, 100, 257):
        for off in (0, 7, 15):
            tiles, which = seq.stream_tiles(n, 32, off)
            for i in range(n):
                s, e = tiles[which[i]]
                assert s <= i < e
                # a window is never near a tile edge unless it is near the stream edge (nearest-centre tile)
                assert min(i - s, e - 1 - i) >= min(7, i, n - 1 - i)


def test_rope_model_has_no_absolute_position():
    torch.manual_seed(0)
    m = seq.SeqModel(D, D, 16, 2, 2, 32, 0.0).eval()
    x, fmt = torch.randn(1, 10, D), torch.randint(0, 4, (1, 10))
    y = m(x, fmt, torch.ones(1, 10, dtype=torch.bool))
    p = 7  # left-pad with masked tokens: real tokens move to positions 7..16
    xp = torch.cat([torch.randn(1, p, D), x], 1)
    fp = torch.cat([torch.zeros(1, p, dtype=torch.long), fmt], 1)
    vp = torch.cat([torch.zeros(1, p, dtype=torch.bool), torch.ones(1, 10, dtype=torch.bool)], 1)
    yp = m(xp, fp, vp)
    torch.testing.assert_close(yp[:, p:], y, atol=1e-5, rtol=1e-4)


def test_no_time_output_invariant_to_clock_shift(tmp_path):
    w0, par, sp, birds = _setup(tmp_path)
    w1, _, _, _ = _setup(tmp_path, shift_h=6.0, name="w_shift")  # same parent h; 6 h moves the local-day partition
    _run(w0, par, sp, tmp_path / "r0")
    _run(w1, par, sp, tmp_path / "r1")
    for b in birds:
        np.testing.assert_array_equal(np.load(tmp_path / "r0/embeddings" / f"{b}.npz")["h"],
                                      np.load(tmp_path / "r1/embeddings" / f"{b}.npz")["h"])
    d0 = np.load(tmp_path / "r0/day_embeddings/b00.npz")["n"]
    d1 = np.load(tmp_path / "r1/day_embeddings/b00.npz")["n"]
    assert not np.array_equal(d0, d1)  # the day partition did change


def test_time_modes_write_h_notime_and_it_ignores_the_clock(tmp_path):
    win, par, sp, birds = _setup(tmp_path)
    _run(win, par, sp, tmp_path / "rel", "--time-encoding", "relative")
    assert json.loads((tmp_path / "rel/design.json").read_text())["uses_time"] is True
    e = np.load(tmp_path / "rel/embeddings/b00.npz")
    assert "h_notime" in e.files and e["h_notime"].shape == e["h"].shape
    assert "h_notime" in np.load(tmp_path / "rel/day_embeddings/b00.npz").files
    assert "h_notime" in np.load(tmp_path / "rel/bout_embeddings/b00.npz").files
    run = tmp_path / "abs"
    _run(win, par, sp, run, "--time-encoding", "absolute")
    before = {b: np.load(run / "embeddings" / f"{b}.npz") for b in birds}
    before = {b: (v["h"].copy(), v["h_notime"].copy()) for b, v in before.items()}
    _setup(tmp_path, shift_h=6.0)  # same windows path, clock shifted; embed again with the same checkpoint
    _run(win, par, sp, run, "--time-encoding", "absolute", "--embed-only")
    changed = False
    for b in birds:
        after = np.load(run / "embeddings" / f"{b}.npz")
        np.testing.assert_array_equal(after["h_notime"], before[b][1])
        changed |= not np.array_equal(after["h"], before[b][0])
    assert changed  # h itself does use the clock


def test_env_off_reads_nothing_env_on_is_used(tmp_path, monkeypatch):
    win, par, sp, birds = _setup(tmp_path)
    env = tmp_path / "env"
    env.mkdir()
    for b in birds:
        n = len(pd.read_parquet(win / b / "meta.parquet"))
        pd.DataFrame({"wind": np.random.default_rng(0).normal(size=n), "blh": np.nan}).to_parquet(env / f"{b}.parquet")
    read = []
    orig = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: (read.append(str(p)), orig(p, *a, **k))[1])
    _run(win, par, sp, tmp_path / "noenv")
    assert read and not any(str(env) in p for p in read)
    assert json.loads((tmp_path / "noenv/design.json").read_text())["uses_env"] is False
    _run(win, par, sp, tmp_path / "withenv", "--env", str(env))
    assert sum(str(env) in p for p in read) == len(birds)
    assert json.loads((tmp_path / "withenv/design.json").read_text())["uses_env"] is True
    assert json.loads((tmp_path / "withenv/env_columns.json").read_text()) == ["wind", "blh"]
    assert np.isfinite(pd.read_csv(tmp_path / "withenv/loss.csv")["loss"]).all()
    H = {b: np.load(par / "embeddings" / f"{b}.npz")["h"] for b in birds[:1]}
    out = seq.contextual_embed(tmp_path / "withenv", H, {b: orig(win / b / "meta.parquet") for b in H})
    np.testing.assert_array_equal(out["b00"].astype(np.float16), np.load(tmp_path / "withenv/embeddings/b00.npz")["h"])


def test_trains_on_train_birds_only(tmp_path):
    win, par, sp, birds = _setup(tmp_path, nan_birds=("b04", "b05"))  # search and locked h are NaN
    run = tmp_path / "run"
    _run(win, par, sp, run, "--steps", "5", "--p-bout", "0.5", "--contrast", "0.1")
    assert np.isfinite(pd.read_csv(run / "loss.csv")["loss"]).all()  # NaN birds never entered a training crop
    assert np.isfinite(np.load(run / "embeddings/b00.npz")["h"].astype(float)).all()
    assert json.loads((run / "seq_info.json").read_text())["train_birds"] == 4


def test_resume_and_config_refusal(tmp_path):
    win, par, sp, _ = _setup(tmp_path)
    run = tmp_path / "run"
    _run(win, par, sp, run, "--steps", "2", "--save-every", "2")
    _run(win, par, sp, run, "--steps", "4", "--save-every", "2")
    assert pd.read_csv(run / "loss.csv")["step"].tolist() == [1, 2, 3, 4]
    assert sorted(p.name for p in run.glob("ckpt_*.pt")) == ["ckpt_0000002.pt", "ckpt_0000004.pt"]
    with pytest.raises(SystemExit, match="other settings"):
        _run(win, par, sp, run, "--steps", "6", "--lr", "1e-2")


def test_parent_checks(tmp_path):
    win, par, sp, _ = _setup(tmp_path)
    with pytest.raises(SystemExit, match="differ from --parent"):
        _run(win, par, sp, par)
    (par / "holdout.json").write_text(json.dumps(["b00"]))
    with pytest.raises(SystemExit, match="parent holdout differs"):
        _run(win, par, sp, tmp_path / "run")
    (par / "holdout.json").write_text(json.dumps(["b04"]))
    (par / "design.json").write_text(json.dumps({"uses_time": True, "uses_env": False, "level": 2, "parent_run": None}))
    with pytest.raises(SystemExit, match="level-1"):
        _run(win, par, sp, tmp_path / "run")


def test_contextual_embed_reproduces_the_run_and_embed_input_cli(tmp_path):
    win, par, sp, birds = _setup(tmp_path)
    run = tmp_path / "run"
    _run(win, par, sp, run, "--time-encoding", "relative")
    files = sorted(p.relative_to(run) for p in run.rglob("*"))
    H = {b: np.load(par / "embeddings" / f"{b}.npz")["h"] for b in birds}
    M = {b: pd.read_parquet(win / b / "meta.parquet") for b in birds}
    for nt, key in ((False, "h"), (True, "h_notime")):
        out = seq.contextual_embed(run, H, M, notime=nt)
        for b in birds:
            np.testing.assert_array_equal(out[b].astype(np.float16), np.load(run / "embeddings" / f"{b}.npz")[key])
    out_dir = tmp_path / "ctx"
    cli = ["--windows", str(win), "--run", str(run), "--embed-input", str(par), "--embed-out", str(out_dir), "--device", "cpu"]
    seq.main(cli)
    e = np.load(out_dir / "embeddings/b03.npz")
    np.testing.assert_array_equal(e["h_notime"], np.load(run / "embeddings/b03.npz")["h_notime"])
    assert sorted(p.relative_to(run) for p in run.rglob("*")) == files  # nothing written into the run
    with pytest.raises(SystemExit, match="not empty"):
        seq.main(cli)
