# Stage 4, run r1_tau0.2 (2026-11-24 export; AI4WY job 61985)

> **Superseded (2026-11-24):** run on profile_v2, which had the wrong sensitivity setting for 73% of
> accelerometer bursts (DESIGN.md §6).

Training: 60,000 steps, batch 1024 (16 birds × 64 windows), τ 0.2, 4,306,574 training windows, 12 held-out
birds, 754,496 parameters, 39 min on one GH200. Final loss 3.80; positive-pair cosine ≈ 0.86, negatives ≈ 0.

## Hour level, same probe suite as the first-draft model (held-out birds by GroupKFold)

| Probe | old z | old h | new h |
|---|---|---|---|
| Magnetometer fitted, within 2024 birds (BA; want ≈0.5) | 0.736 | 0.742 | 0.695 |
| Decoding version (BA; chance 0.2) | 0.323 | 0.420 | 0.447 |
| Any GPS in hour (BA; GPS is now an input, so not informative) | 0.906 | 0.960 | 0.994 |
| VeDBA (R²) | 0.327 | 0.529 | **0.949** |
| Speed (R²) | 0.496 | 0.772 | **0.884** |
| Climb rate (R²) | 0.107 | 0.138 | **0.458** |
| Turn rate (R²) | 0.260 | 0.273 | 0.302 |
| Fraction flapping (R²) | 0.163 | 0.291 | **0.787** |
| Fraction perched (R²) | 0.195 | 0.412 | 0.474 |
| Fraction circling (R²) | 0.346 | 0.609 | **0.842** |
| Fraction gliding (R²) | 0.290 | 0.532 | **0.732** |
| Fraction climbing soar (R²) | 0.341 | 0.616 | **0.825** |
| Fraction night (R²) | 0.074 | 0.125 | 0.302 |
| Sun elevation (R²) | 0.158 | 0.232 | 0.369 |
| Cohort (descriptive, BA) | 0.777 | 0.858 | 0.755 |

Old-model numbers are its step-30k embeddings probed against targets from the new export (142,174 matched hours);
new-model numbers use 303,334 hours.

## Window level, 12 birds never seen in pretraining (linear probes fitted on training birds)

Behaviour: VeDBA R² 0.972, speed 0.943, climb 0.884, turn 0.866, flap-band fraction 0.500, sun elevation 0.342;
balanced accuracy for flapping 0.892, perched 0.991, circling 0.962, gliding 0.975, climbing soar 0.960, night 0.863.

Device nuisances (want chance): **sensitivity setting 0.998** (chance 0.5), **short vs full acceleration
window 0.998** (chance 0.5), decoding version 0.514 (chance 0.2).

## Cluster stability (k-means on h, bird-bootstrap refits, adjusted Rand index)

k = 6: 0.729, 8: 0.789, 10: 0.860, 12: 0.877, 16: 0.730, 20: 0.792 (handcrafted baseline: 0.45–0.63).

## Open problems
1. The sensitivity setting is almost perfectly decodable. Calibration converts both settings to g, so the model
   must be reading quantisation step (1/512 vs 1/1024 g) or clipping range. Fix: re-quantisation/dither
   augmentation and a common clipping range; verify with the same probe.
2. Window length (short bouts vs continuous segments) is decodable. It is a sampling-schedule property. Fix
   options: evaluate on length-matched crops; stronger crop augmentation; report states within each regime.
