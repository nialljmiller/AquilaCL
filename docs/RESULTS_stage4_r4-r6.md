# Stage 4/5: run r4, the sensitivity test, and the bird-identity runs r5–r6

All runs below:
- use windows_v4 (profile_v4, 5,505,206 windows, 92 birds) and the common amplitude format (step 0.002116 g, clip ±1.774 g);
- share the same 12 held-out birds and seed, 60,000 steps, and the same evaluation code;
- differ from r4 only in the settings named. The bird-identity numbers come from `evaluate --only nuisance` (commit 1c5843d or later), run on every model with the same code and rows.

## 1. Sensitivity setting: what the 0.9999 probe measures

Scripts: `scripts/diagnose_r4.py` (job 62082), `scripts/sensitivity_cues.py` (62099), `scripts/sensitivity_matched.py` (62104).

- **Counterfactual.** Windows of r4 at the 1024 counts/g setting were re-quantised to the 512 counts/g step, with a random grid phase, and encoded again.
  - Median change of h: < 0.001 × the distance to the next window.
  - Cluster (K=12) changes: ≤ 0.9%, no more than the control (k=0 windows re-gridded at their own step).
  - The sensitivity classifier on h changed its prediction for **0%** of them.
  - So the representation does not use the step or the range.
- **Recording mode.** In full 200-sample windows of the 32 birds with both settings:
  - 99.99% of k=1 windows have GPS; 0.0005% of k=0 windows do. The 1024 setting is used only in continuous mode, which records GPS every second.
  - Only 25 of 91,913 k=1 full windows are still.
  - Within flapping windows alone, the setting is still readable: 0.9998 from h, 0.9999 from the handcrafted features (which include a no-GPS flag).
- **Conclusion.** The probe reads the recording mode (GPS present, continuous ACC), not amplitude. This cannot be removed without removing GPS.
- **Not explained.** With GPS removed from the input, h still reads the setting at 0.93. This is unexplained; candidates are behaviour mix and continuous-mode ACC.

## 2. Where bird identity comes from (r4; 12 held-out birds; fit on first half of each record, test on second half)

Identity from h on r4 is 0.503 (95% interval over birds 0.394–0.616); chance is 0.083.

| Subset | Identity |
|---|---|
| Still | 0.29 |
| Night still | 0.18 |
| Flapping | 0.50 (probe trained within the subset: 0.65) |
| Soaring / gliding | 0.61 (within the subset: 0.69) |

| Perturbation of test inputs | Drop in identity |
|---|---|
| ACC removed | 0.40 |
| Yaw ±180° | 0.32 |
| GPS removed | 0.12 |
| Yaw ±30° | 0.11 |
| 3-D rotation ±15° | 0.02 |
| Per-axis gain ±3% | 0.00 |
| Per-axis offset ±0.02 g | 0.00 |

- Residual mounting yaw across birds (from the flapping axis) has a standard deviation of 15.8°. The roost alignment cannot correct yaw.
- The handcrafted floor of 0.35 is posture-driven: 0.14 without posture, 0.375 with gravity-aligned posture.

## 3. Runs

Identity is shown as the value with its 95% interval over birds; Δ is the paired difference from r4 over birds.

| Run | Change from r4 | Identity | Δ vs r4 | Birds lower | h − features |
|---|---|---|---|---|---|
| r4_v4 | – | 0.491 [0.376, 0.593] | – | – | +0.115 [+0.032, +0.212] |
| r5_bpb4 | 4 birds per batch | 0.433 [0.322, 0.535] | −0.058 [−0.077, −0.038] | 12/12 | +0.057 |
| r5_conf | bird-confusion adversary 0.3 | 0.492 [0.376, 0.596] | +0.001 [−0.011, +0.014] | 7/12 | +0.117 |
| r5_yaw30 | yaw ±30° | 0.354 [0.249, 0.464] | −0.137 [−0.195, −0.082] | 12/12 | −0.021 |
| r6_yaw30_bpb4 | yaw ±30°, 4 birds per batch | 0.335 [0.227, 0.445] | −0.156 [−0.227, −0.091] | 11/12 | −0.041 |
| **r6_yaw60_bpb4** | yaw ±60°, 4 birds per batch | **0.313 [0.219, 0.413]** | **−0.178 [−0.251, −0.117]** | 12/12 | −0.063 |

- **Behaviour probes.** All runs are within ±0.01 of r4 on the window-level probes (VeDBA, speed, climb, turn, flapping, perched, circling, gliding, climbing-soar, night) and the hour-level probes. Two exceptions:
  - r5_conf hour climb is −0.05;
  - r6_yaw60_bpb4 window turn is 0.913 against 0.919.
- **The adversary did not work.** It reached chance on the training birds (accuracy 0.028; chance 0.0125), but identity on held-out birds did not change.

## 4. States (stage 5; K chosen by bird-bootstrap Jaccard, held-out-bird ARI and presence)

| Run | Chosen K | Min Jaccard | Held-out ARI | Rule met |
|---|---|---|---|---|
| r4 | 6 | 0.735 | 0.710 | no |
| r5_yaw30 | 6 | 0.535 | 0.498 | no |
| r6_yaw30_bpb4 | 6 | 0.499 | 0.480 | no |
| r6_yaw60_bpb4 | 6 | 0.620 | 0.663 | no |

Every run gives the same six states. Values for r6_yaw60_bpb4 are medians:

| State | Description | VeDBA (g) | Speed (m/s) | Climb (m/s) | Turn (°/s) | Share of windows |
|---|---|---|---|---|---|---|
| 0 | Rest | 0.005 | 0.02 | 0.00 | – | 26.6% |
| 1 | Perched, active | 0.032 | 0.04 | 0.00 | 7.7 | 15.7% |
| 2 | Thermal circling | 0.068 | 10.3 | +1.43 | 19.9 | 13.3% |
| 3 | Fast descending glide | 0.090 | 18.3 | −1.90 | 1.8 | 13.4% |
| 4 | Cruising flight | 0.090 | 11.0 | +0.40 | 5.6 | 20.4% |
| 5 | Flapping | 0.283 | 11.3 | +0.16 | 10.2 | 10.4% |

State numbers are those of r6_yaw60_bpb4; other runs number the same states differently. HMM and GMM labels agree on 98% of windows.

- **Stability rule.** No run meets it. Reducing identity lowered state stability relative to r4.
- **Sex.** The reference CSV now matches 92 of 92 birds (fix a171225): 48 m, 41 f, 3 missing. Sex shows no difference in any state share (permutation p 0.14–0.89).
- **Life stage.** Not tested: the reference data contain none.

## 5. Choice

r6_yaw60_bpb4 has the lowest bird identity, and its identity is no longer above the handcrafted floor. It keeps r4's behaviour probes, and among the identity-reduced runs it has the most stable states. It replaces r4 as the current model.

Outputs on AI4WY:
- `runs/r6_yaw60_bpb4`
- `results_2026-11/v4/{states,analysis}_r6_yaw60_bpb4`
