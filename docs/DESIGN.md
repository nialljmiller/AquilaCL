# AquilaCL: design

Goal (Nill, 2026-11-24): capture all distinct movements and behaviours of the tagged golden eagles and
characterise them through a SimCLR latent space.

## 1. What is wrong with the first draft (evidence)

| Problem | Evidence | Consequence |
|---|---|---|
| The latent space mostly encodes tag cohort and sampling regime | k=8 cluster C6 = all 19 birds deployed Jun–Oct 2025 (no magnetometer, few empty hours); C1 = the 2024 cohort. Projection z separates zero-GPS from ≥3-GPS hours at 96% balanced accuracy on held-out birds. | Clusters describe data regime, not behaviour. |
| 36% of windows contain no measurement | 51,247 of 144,219 hours have no GPS and the `eobs:activity` channel is empty everywhere, so these rows are all (0,0,0,0,0,1,0). | The model learns sequence length and missingness. |
| The richest behaviour signal is unused | 13.2 M acceleration bursts (20 Hz XYZ, 1.2–13.2 s) are never decoded. | Flapping, soaring, gliding and perching are hard to separate from GPS alone. |
| Non-GPS rows are fed as positions | Acceleration, magnetometer and orientation rows (about 63% of rows) have no position; the encoder sets lon/lat to 0 and then differences them, so each switch between a GPS row and a non-GPS row is a jump of about 108° in longitude. | The position-change channels mainly record which sensor wrote each row. |
| The analyses cannot show the result they claim | Diel plots use UTC, and their "night" shading covers local daytime; the behaviour "validation" plots the same features the rules use; the age script crashes on two plots and the error is swallowed, leaving stale step-20k PNGs; the per-bird behaviour bars stack wrongly. Full list: research/analysis_inventory.md. | The analysis suite has to be rebuilt as well as the model. |
| The unit is a UTC clock hour | Eagle behaviours last seconds to minutes. 06–11 UTC is 23:00–05:00 local (night), so "hour of day" plots are in the wrong time frame. | Mixed behaviours per window; misleading diel plots. |
| Behaviour labels are hourly rules | Vertical rate divided by the median interval of all events (5× too steep); 69% of hours "unknown". | Labels are not usable as reference. |
| Age labels are a movement heuristic | 27 "adult" labels come from movement extent; the field-recorded fledgling is labelled adult; the adult group is almost exactly the 2025 cohort. | Age comparisons are confounded with cohort. |
| Engineering | 15+ min of CSV indexing per run; Python loop over B=2048 in the loss mask every step; inference uses augmented views (not reproducible); `loss.csv` truncated on resume; DataParallel for a 1 M-parameter model. | Slow, irreproducible, risky to rerun. |

## 2. New pipeline

1. **Ingest** (`aquilacl.ingest`): stream the 18 GB CSV once into sorted per-bird, per-sensor Parquet with burst
   columns parsed to integer arrays. Refuses to write into an existing output.
2. **Calibrate and profile** (`aquilacl.acc`, `aquilacl.gps`): per-tag accelerometer offset/gain fitted from still
   bursts (|g| = 1); GPS quality filter; continuous GPS runs with climb rate and turn rate from actual fix times;
   a sampling-schedule profile per tag and month (burst lengths and intervals, GPS cadence). The schedule profile
   decides the window length below.
3. **Observation windows** replace clock hours. A window is a short span (candidate 2–10 min, chosen from the
   schedule profile) that contains at least one measurement. Its contents are a set of *tokens*, each with its
   time offset and modality:
   - accelerometer burst token: calibrated (samples × 3) burst, encoded by a small 1-D CNN;
   - GPS fix token: speed, climb rate, turn rate, step length, accuracy, time since previous fix.
   No absolute position, absolute time, tag id or magnetometer/orientation input (these define cohort), so the
   encoder cannot use them. Empty spans produce no window.
4. **Encoder**: modality embeddings + continuous time encoding + a small transformer with padding masks;
   output h (representation) and z (projection for the loss).
5. **SimCLR training**, changed to target behaviour rather than data regime:
   - positives: two time-crops of the same span, plus thinning of tokens (simulates sparser schedules) and
     whole-modality dropout (accelerometer-only vs GPS-only views must agree; makes the 2024 and 2025 cohorts
     comparable);
   - accelerometer augmentations: small 3-D rotation (harness shift), gain scaling, noise;
   - bird-balanced batches; negatives from the same bird within ±30 min are masked (likely the same behaviour);
   - vectorised NT-Xent, bf16, one H200; deterministic, augmentation-free inference.
6. **Evaluation fixed before training**: ~12 birds held out from pretraining, stratified by cohort.
   - nuisance probes (should be low): bird id, cohort, decoding-software version, GPS density;
   - physical probes (should be high): VeDBA, wingbeat frequency, climb rate, turn rate, speed;
   - baseline: handcrafted features + GMM/HMM on the same windows; the SimCLR space must beat it to be kept.
7. **State discovery and characterisation**: clustering in h (GMM with BIC and HDBSCAN, bootstrap stability),
   HMM over each bird's window sequence for persistent states and transitions; each state described by its
   physical signature, solar elevation (not UTC), season, migration context, sex, cohort, and field-recorded age
   (nestling/fledgling at tagging; others "age unknown", not "adult").

## 2b. Analysis suite

Modules M0–M8 in research/analysis_inventory.md §4 keep every question the old plots asked (latent overview,
state discovery, state characterisation, diel and seasonal patterns, individual/sex/age, anomalies, training
diagnostics). They are reimplemented on the new tables. Statistics are bird-level or use mixed models with bird
as a random effect; time of day is local solar time; every threshold lives in the run config.


## 2c. Changes from the literature review (research/literature.md)

- **Unit and sampling.** One observation = one accelerometer burst with the GPS fixes around it, or a 30–120 s
  stretch of 1 Hz GPS. 1 Hz runs and 5-minute fixes are not mixed in one encoder input. Training windows are
  sampled in proportion to movement (Yuan et al. 2024: without this, a pretext task stayed at chance).
- **No per-window z-scoring.** Absolute speed, climb rate, VeDBA, pitch/roll and wingbeat band power carry the
  behaviour. All accelerometer inputs use a common 24-sample (1.2 s) crop so that burst length (a schedule
  property) cannot leak into the model. Dominant frequency is only used on 264-sample bursts; short bursts use
  1.5–4 Hz band power (golden eagle wingbeat about 2.2–2.8 Hz).
- **Calibration.** Gain and offset are fitted per tag from still bursts. The published tag-ID rule would give
  1/512 g per count for all these tags, which does not fit the sample (still bursts about 1,000 counts from 2048).
  The fit decides; the rule is the cross-check. Each bird's frame is rotated so its median roosting gravity vector
  points up.
- **Negatives.** Batches built from a few birds with many windows each; batch size 256–2048 and τ 0.07–0.5 swept;
  a false-negative-tolerant loss variant tested (Qian et al. 2022: >128 negatives hurt SimCLR on activity data).
- **Augmentations.** Noise, ±10% scaling, ±10–15° rotation, crops, cutout, token thinning, modality dropout. No full
  rotations, axis swaps, shuffling, strong time-warp or reversal (they destroy pitch, wingbeat and climb direction).
- **Acceptance test (fixed before training).** Linear probes on held-out birds.
  - Device nuisances should be near chance: magnetometer vs no magnetometer *within the 2024 cohort* (five
    2024 birds have none), decoding-software version within a bird, burst length, GPS cadence.
  - Behaviour should be high: rule-derived flapping, circling, gliding, perched and night labels (used only for
    evaluation), VeDBA, wingbeat band power, climb and turn rate.
  - Cohort is NOT a pass/fail criterion: here cohort is entangled with age and season (2025 = mostly territorial
    adults seen Oct–Mar; 2024 = mostly nestlings), so hiding it would erase real age and season behaviour. Cohort
    is reported descriptively in calendar months where both cohorts have data.
  - The same probes are run on the old step-30k embeddings, so "improved" is measured against numbers.
  - Baseline: handcrafted features + HMM (momentuHMM-style). Diel results indexed by sun elevation; night is the
    roosting control.
- **States.** HDBSCAN/GMM in the embedding itself, not in UMAP. Kept only if stable across bird-level bootstrap and
  seeds, present in ≥5 birds and in more than one cohort.

## 3. Compute (AI4WY, checked 2026-11-24 08:35 MST)

gp-1: 12 nodes, 1× H200, 72 CPU, 572 GB; 11 idle. gp-2: 24 nodes, 2× H200, 144 CPU, 1.1 TB; 5 idle, most
busy with 24 h array jobs. Nodes are aarch64. Max wall time 24 h. `sbatch --test-only` confirmed that gp-1 accepts
CPU-only jobs and single-GPU jobs. The model needs one GPU; gp-1 is used throughout.

All outputs go to the new directory `/project/ai4wy-eap/nmille39/aquilacl/`. The existing checkout,
`logs/main`, `golden_eai_outputs` and the `ai4wy` conda env are not modified.

## 4. Run plan on AI4WY

All jobs: account ai4wy-eap, partition gp-1, outputs only under /project/ai4wy-eap/nmille39/aquilacl/.

| Stage | What | Request | Status |
|---|---|---|---|
| 0 | New env `aquilacl`, cloned from `ai4wy`, + pyarrow; run tests | login node, install only | awaiting approval |
| 1 | Ingest CSV -> Parquet with exact-count check (`slurm/01_ingest.sh`) | 32 CPU, 256 GB, 6 h, no GPU | awaiting approval |
| 2 | Schedule profile, calibration, burst features, GPS kinematics (`slurm/02_profile.sh`) | 36 CPU, 300 GB, 8 h, no GPU | awaiting approval |
| 3 | Windows/tokens, handcrafted baseline + HMM, probes on old embeddings | CPU jobs | to be written after stage 2 results |
| 4 | SimCLR training + sweeps/ablations | 1× H200 per run, ≤24 h with resume | to be written; separate approval |
| 5 | Analysis suite M0–M8 | CPU (+1 GPU for UMAP if needed) | to be written |
