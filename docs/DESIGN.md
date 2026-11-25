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
   - bird-balanced batches; negatives from the same bird within ±30 min are masked (likely the same behaviour;
     implemented from run r4 on; r1–r3 masked only ±3 windows of the same segment);
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
  behaviour. Planned: a common crop so that burst length (a schedule property) cannot leak into the model.
  Implemented instead (§6): burst-crop views drawn from the real lengths of non-full windows. Dominant frequency is only used on 264-sample bursts; short bursts use
  1.5–4 Hz band power (golden eagle wingbeat about 2.2–2.8 Hz).
- **Calibration.** Gain and offset are fitted per tag from still bursts. The published tag-ID rule would give
  1/512 g per count for all these tags, which does not fit the sample (still bursts about 1,000 counts from 2048).
  The fit decides; the rule is the cross-check. (Superseded, §6: the setting depends on burst type.) Each bird's
  frame is rotated so its median roosting gravity vector points up (§6: fallback for birds without roost windows).
- **Negatives.** Batches built from a few birds with many windows each. Planned but not done: a batch-size sweep
  and a false-negative-tolerant loss (Qian et al. 2022). Done: τ 0.1 and 0.2, with and without modality dropout
  (r1–r3).
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

## 5. Stage 2 results (profile_v2, 2026-11-24)

- **Sampling.** GPS is recorded at 1 Hz in runs: median run 9 s, upper quartile ≥152 s, 10% reach the 899 s
  (15 min) cap; 4,051 h of 1 Hz GPS in total. No GPS runs start between local solar 21:00 and 03:00 (tags do not
  record GPS at night). Accelerometry totals 6,056 h: 60% of it in continuous 5–60 min segments (16,436 segments),
  the rest in 2 s and 13.2 s bursts. 92% of accelerometer bursts lie within 2 s of a GPS fix, so the two streams
  are recorded together.
- **Calibration.** After splitting each tag into sensitivity periods, 88 tag-periods with ≥50 still bursts give
  gains of 0.94–1.11 × the published 512 counts/g (median 1.01) and a median still-burst error of 0.0085 g.
  33 of 62 birds had periods at double sensitivity (~1024 counts/g), mostly April–September 2025. Two periods are
  unreliable (one with 18 still bursts; one isolated single day). **Wrong — see §6.**
- **Consequence for the window.** The natural unit is a 10 s co-recorded window: 10 GPS fixes at 1 Hz plus 200
  accelerometer samples × 3 axes at 20 Hz. Short recording bouts give one window each; long continuous segments
  give consecutive windows, which also provide sequences for temporal models. Roughly 2.5 M windows (estimate).

## 6. Corrections after the stage 4 reviews (2026-11-24)

Three independent reviews (code, data, method) found the following. Earlier outputs are kept unchanged.

- **Sensitivity setting is set by burst type, not by day.** Read from the raw counts of all 26.7 M bursts:
  24-sample bursts (continuous mode) are recorded at about 1024 counts/g; 40- and 264-sample bursts at about
  512 counts/g (within-tag ratio 2.001 ± 0.050 over 62 tags; no mixed group). The per-day rule of profile_v2 put
  73.4% of bursts on the wrong setting, so most continuous-mode data read about 2 g at rest. profile_v2,
  windows_10s, the stage 3 baseline and runs r1–r3 are therefore wrong for anything that depends on amplitude.
  Now (profile_v3): k per (bird, tag, burst length) from the median raw norm of quiet bursts; calibration per
  (tag, k); a setting with few still bursts or an unreliable fit takes the same tag's reliable calibration with
  the gain scaled by 2^Δk; remaining unreliable calibrations are flagged and their bursts left out of windows.
- **Sensitivity leak.** The two settings differ in step (about 0.002 vs 0.001 g) and range (±4 vs about ±1.9 g).
  `format.json` (profile stage) gives the coarsest step and the narrowest range minus 0.05 g; training rounds and
  clips all acceleration to these in the sensor frame, before rotation.
- **Short-window leak.** Non-full windows are 40-sample bouts, 64-sample tails of 264-sample bursts and short runs
  of continuous-mode bursts, all starting at slot 0. Training cuts one view of full windows to these empirical
  lengths and moves it to the window start (probability 0.3).
- **Window placement.** Consecutive 24-sample bursts are placed contiguously (start within 2 slots of the previous
  end); before, 26% of windows had one-slot holes that revealed continuous recording. Overlapping samples are
  dropped, not overwritten.
- **Features and labels.** The static component is a 1 s running mean per contiguous run, so features no longer
  depend on window fullness. Reference labels are NaN where their inputs are missing (before: "not perched" etc.
  for windows without GPS, which let probes score by reading GPS presence).
- **GPS.** Turn rate is masked for steps < 5 m or intervals < 1 s; fixes with track speed > 70 m/s or > 10 m/s
  from ground speed are masked; climb outside +20/−70 m/s is masked. The horizontal-accuracy estimate is hidden
  from the model (fix quality, not behaviour).
- **Orientation.** Seven birds had too few roost windows and were left unrotated (≈116° from everyone else).
  Now quiet daytime windows are used, else the population-median direction.
- **Training.** NT-Xent similarities in float32; same-bird negatives within 30 min masked; resubmission with
  different settings is refused.
- **Evaluation.** Invariance tests (same k-means state after a burst crop, after removing GPS); sensitivity probe
  on full windows of birds with both settings and a bird-identity probe, each next to the same probe on
  handcrafted features. A nuisance probe on the mask (recording length) is expected to stay high; the burst
  invariance test is the criterion.
- **Baseline HMM.** Emissions now use p(x | state) without the mixture weights.
