# r7: per-bird yaw correction, neighbour positives, masked reconstruction (2026-11-25)

Data: windows_v4, 92 birds with windows. Code: commit ab3362e.

Full comparison table on AI4WY: `results_2026-11/v4/compare_r7.md` and `.csv`, made by `scripts/compare_runs.py` over 466 metrics.

## What was tested

r7 put four changes to the training recipe of the current model (r6_yaw60_bpb4) into one recipe, called r7_full:

1. **Per-bird yaw correction** (`--derotate-yaw`). Each bird's tag is turned about the vertical so that its flapping axis lies on +x.
   - The axis is the axial mean of the bird's own flapping windows.
   - 83 birds have enough flapping windows (at least 50 qualifying) for their own estimate. The other 9 birds get the population axis.
   - The ±60° random yaw augmentation shrinks to ±15°, plus a random 180° turn with probability 0.5 (`--yaw-flip-p 0.5`), because the flapping axis gives a direction but not which end is forward.
2. **Neighbour positives** (`--p-neighbour 0.5`). Half of the time, the second view of a window is the next or previous window in the same continuous-mode stretch, instead of an augmented copy of itself.
3. **Masked reconstruction** (`--mask-recon 5`). A small decoder reconstructs hidden 10-sample spans (about 30% of each window) from the encoder's sequence features.
   - The weight 5 was not tuned. Reconstruction was about 4% of the loss at the start.
   - Its error fell from 0.045 to 0.019 (r7_full_s0).
4. **Fresh held-out birds.** There is a new salt, and the 12 held-out birds of r6 are excluded, so no test bird was used to choose between earlier models. Batches hold 4 birds each, as in r6.

**Runs:**

| Recipe | Seeds | What it is |
|---|---|---|
| r7_base | 3 | The r6 recipe (`--yaw-deg 60`) on the new held-out birds |
| r7_full | 3 | Changes 1–4 |
| r7_noderot | 1 | Leaves out change 1 |
| r7_noneigh | 1 | Leaves out change 2 |
| r7_nomask | 1 | Leaves out change 3 |

- Jobs 62280–62288, 60,000 steps each, all COMPLETED.
- Evaluation ran in the same job and added two kinds of check:
  - probes for the turn-sensitive quantities (fore-aft and side-to-side tilt, horizontal share of dynamic acceleration, turn direction), split by window type;
  - a bird-level sex probe on mean h in flight and on the ground, using the K=2 states of r6.

**Not added:** extra sensors. The tags have no barometer, and magnetometer and orientation data exist for only 35 of the 94 birds.

## How to read the table

The table flags 24 metrics as "WORSE". Most of them are not losses.

1. **Tilt columns swap names in the corrected frame.** Posture targets are computed in each run's own frame.
   - Without the yaw correction (r7_base, r7_noderot), the flapping axis is the sensor y axis. The column `roll_rot_deg` then holds the fore-aft tilt (about −65° in flight, about 0° perched).
   - With the correction, the flapping axis is x, so the fore-aft tilt is in `pitch_rot_deg`.
   - Compared by physical meaning, fore-aft tilt is read equally well:

     | Windows | r7_base (`roll_rot_deg` R²) | r7_full (`pitch_rot_deg` R²) |
     |---|---|---|
     | All | 0.944 | 0.931 |
     | Full | 0.898 | 0.975 |
     | Short | 0.948 | 0.902 |

   - Side-to-side tilt is near 0 in both.
   - The five `roll_rot_deg` flags are this relabelling.
2. **The handcrafted-feature identity baseline also depends on the frame.** Handcrafted features identify held-out birds at 0.259 in the roost frame but 0.178 after yaw correction, so the correction removes some mounting identity from the raw data. For that reason `bird_id_holdout_h_minus_features` is not comparable across frames. Raw h and the sensor-frame difference are.
3. **Some gains are trained in directly.** The next-window and burst state-consistency gains (next window same state 0.60 → 0.77) are what neighbour positives optimise. r7_noneigh, the one variant without them, does not gain. These are not evidence of better behaviour states.
4. **`sex_flight.ground.auc` is a negative control.** It is 0.51 for base and 0.47 for full; both are chance.

## Results (held-out birds; mean ± sd over 3 seeds; single runs for ablations)

| Measure | r7_base | r7_full | noderot | noneigh | nomask |
|---|---|---|---|---|---|
| Bird identity from h, held-out birds (lower is better; chance 0.083) | 0.306 ± 0.003 | 0.368 ± 0.004 | 0.385 | 0.302 | 0.371 |
| Sex from mean h in flight, bird-level AUC | 0.761 ± 0.013 | 0.741 ± 0.021 | 0.773 | 0.750 | 0.749 |
| Horizontal share of dynamic acceleration, full windows R² | 0.328 ± 0.012 | 0.628 ± 0.010 | 0.595 | 0.490 | 0.581 |
| Perched vs not, full windows, balanced accuracy | 0.870 ± 0.032 | 0.814 ± 0.032 | 0.944 | 0.888 | 0.777 |
| Turn direction, short windows, balanced accuracy | 0.882 ± 0.004 | 0.814 ± 0.021 | 0.763 | 0.882 | 0.782 |
| Night fraction of the hour, R² | 0.367 | 0.321 | 0.370 | 0.308 | 0.312 |
| Speed, climb, VeDBA, wingbeat band, R² (all windows) | 0.96 / 0.93 / 0.97 / 0.32 | 0.96 / 0.93 / 0.98 / 0.32 | same | same | same |

- **Neighbour positives raise bird identity by 6–8 points.** Every run with them is at 0.37–0.39. The run without them is at 0.30, equal to the baseline. Consecutive windows always come from the same bird, so treating them as alike teaches the model what each bird's windows share. They also lower turn-direction accuracy on short windows.
- **Masked reconstruction has no measurable effect** (r7_full against r7_nomask).
- **Yaw correction:**
  - lowers handcrafted-feature identity in the data (0.259 → 0.178);
  - raises the horizontal-share probe;
  - lowers how well time of day can be read (night fraction R² −0.05, only in runs with the correction).
  - It does not change identity in h when neighbours are absent (noneigh 0.302 against base 0.306).
- **Sex in flight is flat across all recipes** (0.74–0.77, with 0.02–0.04 sd over CV repeats).
- **Cluster stability** of h at K=6–20 moves in both directions with no consistent pattern.

## Decision

- **r7_full is not better than r7_base on the measures the project is about** (less bird identity, more sex signal in flight). Identity is worse and sex is unchanged. r7_full is therefore not adopted.
- **r7_base is the r6 recipe**, and its identity result replicates on 12 fresh held-out birds over 3 seeds: 0.306 ± 0.003, against 0.313 for r6 on the old held-out birds. The current model, **r6_yaw60_bpb4, stays the model**, and the stage-6 and stage-7 downstream outputs built on it stand. They were not rerun, because a rerun on r7_base would repeat the same recipe.
- **r7_noneigh** (yaw correction + flip + masking, no neighbours) matches the baseline on identity and sex and raises the horizontal-share probe. It is a single seed, picked after looking at 466 metrics, so it is not adopted on this evidence.

## Naming correction for earlier documents

- In the roost frame used by r4 and r6, the flapping axis is the sensor y axis. The quantity named `roll_rot_deg` in `states.parquet` and in stage 4–7 outputs is therefore the **fore-aft (pitch) tilt**: about −65° in flight, about 0° perched.
- The quantity named `pitch_rot_deg` is the **side-to-side (roll) tilt**.
- Notes are added to RESULTS_stage6_demography.md and RESULTS_stage7.md. Column names in code and outputs are unchanged.
