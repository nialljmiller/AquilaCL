# AquilaCL

Self-supervised contrastive learning (SimCLR) of golden eagle (*Aquila chrysaetos*) movement and behaviour from
e-obs GPS–accelerometer biologging data. The goal is to capture the distinct movements and behaviours in the
tracking data and characterise them through the learned latent space.

Status: redesign in progress. See [docs/DESIGN.md](docs/DESIGN.md) for the problems found in the first draft,
the new method, and the run plan.

## Layout

- `src/aquilacl/ingest.py` – Movebank CSV → sorted per-bird, per-sensor Parquet (burst columns parsed; exact-count check)
- `src/aquilacl/acc.py` – accelerometer burst decoding, per-tag calibration from still bursts, burst features
- `src/aquilacl/gps.py` – GPS quality filter, continuous runs, kinematics from actual fix times
- `src/aquilacl/solar.py` – sun elevation and local solar time
- `src/aquilacl/profile.py` – stage 2: sampling-schedule profile, calibration table, per-observation features
- `slurm/` – job scripts for the UWyo ARCC AI4WY cluster
- `docs/` – design, literature review, inventory of the first-draft analyses

## Data

No tracking data, derived tables or model checkpoints are stored in this repository: the tracks include
nest-area locations. The `.gitignore` excludes data formats.

## Tests

```sh
pip install -e ".[dev]"
pytest -q
```
