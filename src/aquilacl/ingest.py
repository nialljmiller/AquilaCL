"""Convert the Movebank CSV export into sorted per-bird, per-sensor Parquet files.

The export (one 18 GB CSV) is read once, in streamed blocks. Rows are split by
``sensor-type`` and by bird, and only the columns that the pipeline uses are
kept. Packed burst columns (acceleration, magnetometer, quaternion) are parsed
into integer lists, so later stages never parse text again.

Layout written under ``out``::

    raw/<sensor>/<bird>.parquet      unsorted, appended during the stream
    events/<sensor>/<bird>.parquet   sorted by time, duplicate event ids removed

The ``raw`` stage refuses to run into a non-empty directory, so a rerun can never
double-count events (the first-draft catalogue builder appended).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

BIRD = "individual-local-identifier"
SENSOR = "sensor-type"

COMMON = {
    "event-id": pa.int64(),
    "timestamp": pa.timestamp("ms"),
    BIRD: pa.string(),
    "tag-local-identifier": pa.string(),
    "data-decoding-software": pa.string(),
}

# Columns kept per sensor stream, with their parsed types. Packed burst
# columns are read as strings and parsed into lists below.
STREAMS: dict[str, dict[str, pa.DataType]] = {
    "gps": {
        "location-long": pa.float64(),
        "location-lat": pa.float64(),
        "ground-speed": pa.float32(),
        "heading": pa.float32(),
        "height-above-ellipsoid": pa.float32(),
        "eobs:horizontal-accuracy-estimate": pa.float32(),
        "eobs:speed-accuracy-estimate": pa.float32(),
        "gps:dop": pa.float32(),
        "gps:satellite-count": pa.float32(),
        "eobs:status": pa.string(),
        "eobs:type-of-fix": pa.float32(),
        "eobs:used-time-to-get-fix": pa.float32(),
        "eobs:battery-voltage": pa.float32(),
        "eobs:temperature": pa.float32(),
        "mag:magnetic-field-raw-x": pa.float32(),
        "mag:magnetic-field-raw-y": pa.float32(),
        "mag:magnetic-field-raw-z": pa.float32(),
        "orientation:quaternion-raw-w": pa.float32(),
        "orientation:quaternion-raw-x": pa.float32(),
        "orientation:quaternion-raw-y": pa.float32(),
        "orientation:quaternion-raw-z": pa.float32(),
    },
    "acceleration": {
        "eobs:start-timestamp": pa.timestamp("ms"),
        "eobs:acceleration-axes": pa.string(),
        "eobs:acceleration-sampling-frequency-per-axis": pa.float32(),
        "eobs:accelerations-raw": pa.string(),
    },
    "magnetometer": {
        "external-temperature": pa.float32(),
        "mag:magnetic-field-sampling-frequency-per-axis": pa.float32(),
        "mag:magnetic-fields-raw": pa.string(),
    },
    "orientation": {
        "orientation:quaternions-sampling-frequency": pa.float32(),
        "orientation:quaternions-raw": pa.string(),
    },
}

PACKED = {
    "eobs:accelerations-raw": pa.int16(),
    "mag:magnetic-fields-raw": pa.int32(),
    "orientation:quaternions-raw": pa.int32(),
}


def parse_packed(col: pa.ChunkedArray | pa.Array, value_type: pa.DataType) -> pa.Array:
    """Turn whitespace-separated integer text into a list<int> column."""
    trimmed = pc.utf8_trim_whitespace(col)
    parts = pc.split_pattern_regex(trimmed, r"\s+")
    return pc.cast(parts, pa.list_(value_type))


def normalise_bird(col):
    """Newer Movebank exports write the animal as "<animal>; <tag>"; keep the animal part."""
    first = pc.list_element(pc.split_pattern(col, ";"), 0)
    return pc.utf8_trim_whitespace(first)


def _column_types() -> dict[str, pa.DataType]:
    types = dict(COMMON)
    types[SENSOR] = pa.string()
    for cols in STREAMS.values():
        types.update(cols)
    return types


def stream_split(csv_path: Path, out: Path, block_mb: int = 256) -> dict[tuple[str, str], int]:
    """Stream the CSV once and append rows to raw/<sensor>/<bird>.parquet."""
    raw = out / "raw"
    if raw.exists() and any(raw.rglob("*.parquet")):
        raise FileExistsError(f"{raw} already holds parquet files; use a new output directory")
    types = _column_types()
    # .zst/.gz/.bz2 inputs are decompressed on the fly (detected from the extension)
    reader = pacsv.open_csv(
        pa.input_stream(str(csv_path)),
        read_options=pacsv.ReadOptions(block_size=block_mb << 20),
        convert_options=pacsv.ConvertOptions(
            column_types=types,
            include_columns=list(types),
            include_missing_columns=True,
            strings_can_be_null=True,
        ),
    )
    writers: dict[tuple[str, str], pq.ParquetWriter] = {}
    counts: dict[tuple[str, str], int] = {}
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            table = table.set_column(table.schema.get_field_index(BIRD), BIRD, normalise_bird(table[BIRD]))
            for sensor, cols in STREAMS.items():
                part = table.filter(pc.equal(table[SENSOR], sensor))
                if part.num_rows == 0:
                    continue
                part = part.select(list(COMMON) + list(cols))
                for name, vtype in PACKED.items():
                    if name in cols:
                        i = part.schema.get_field_index(name)
                        part = part.set_column(i, name, parse_packed(part[name], vtype))
                for bird in pc.unique(part[BIRD]).to_pylist():
                    rows = part.filter(pc.equal(part[BIRD], bird)) if bird is not None else part.filter(pc.is_null(part[BIRD]))
                    key = (sensor, bird or "UNKNOWN")
                    if key not in writers:
                        path = raw / sensor / f"{key[1]}.parquet"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        writers[key] = pq.ParquetWriter(path, rows.schema, compression="zstd")
                    writers[key].write_table(rows)
                    counts[key] = counts.get(key, 0) + rows.num_rows
    finally:
        for w in writers.values():
            w.close()
    return counts


def finalize_one(src: Path, dst: Path) -> tuple[str, int, int]:
    """Sort one raw file by time and drop repeated event ids."""
    table = pq.read_table(src)
    n_in = table.num_rows
    table = table.sort_by([("timestamp", "ascending"), ("event-id", "ascending")])
    ids = table["event-id"].to_numpy()
    if len(ids):
        import numpy as np

        _, first = np.unique(ids, return_index=True)
        keep = np.zeros(len(ids), dtype=bool)
        keep[first] = True
        table = table.filter(pa.array(keep))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression="zstd")
    return str(dst), n_in, table.num_rows


def finalize(out: Path, workers: int = 8) -> list[tuple[str, int, int]]:
    srcs = sorted((out / "raw").rglob("*.parquet"))
    dsts = [out / "events" / s.relative_to(out / "raw") for s in srcs]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(finalize_one, srcs, dsts))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=Path, help="Movebank CSV export")
    ap.add_argument("out", type=Path, help="new output directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block-mb", type=int, default=256)
    ap.add_argument(
        "--expect",
        nargs="*",
        default=[],
        metavar="SENSOR=ROWS",
        help="known row counts (e.g. gps=14876799, total=..., birds=62); exit non-zero on mismatch",
    )
    args = ap.parse_args(argv)
    counts = stream_split(args.csv, args.out, args.block_mb)
    print(f"split {sum(counts.values()):,} rows into {len(counts)} files")
    observed = {"total": sum(counts.values()), "birds": len({b for _, b in counts})}
    for (sensor, _), n in counts.items():
        observed[sensor] = observed.get(sensor, 0) + n
    bad = []
    for item in args.expect:
        key, want = item.split("=")
        got = observed.get(key, 0)
        print(f"check {key}: expected {int(want):,}, got {got:,}")
        if got != int(want):
            bad.append(key)
    if bad:
        raise SystemExit(f"row counts differ from the known values for: {', '.join(bad)}")
    for path, n_in, n_out in finalize(args.out, args.workers):
        if n_in != n_out:
            print(f"{path}: removed {n_in - n_out} repeated event ids")
    print(f"sorted files under {args.out / 'events'}")


if __name__ == "__main__":
    main()
