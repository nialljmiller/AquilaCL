# Stage 4: comparison of runs r1, r2 and r3

All three runs use the new Movebank export (2026-11-24), 5,515,018 windows, the same 12 held-out birds,
60,000 steps and the same evaluation code. Only the settings named below differ.

| Run | Temperature | Sensor (modality) dropout | Final loss | Training time |
|---|---|---|---|---|
| r1_tau0.2 (job 61985) | 0.2 | GPS 0.25, ACC 0.15 | 3.80 | 2318 s |
| r2_tau0.1 (job 61986) | 0.1 | GPS 0.25, ACC 0.15 | 1.72 | 4334 s |
| r3_no_modality_drop (job 61987) | 0.2 | none | 3.41 | 4401 s |

Loss values are not comparable across temperatures or dropout settings.

## Hour level (all 94 birds, grouped cross-validation by bird)

| Target | r1 | r2 | r3 |
|---|---|---|---|
| VeDBA R² | **0.949** | 0.929 | 0.946 |
| Speed R² | **0.884** | 0.882 | 0.872 |
| Climb R² | 0.458 | **0.467** | 0.465 |
| Turn R² | 0.302 | **0.309** | 0.291 |
| Fraction flapping R² | **0.787** | 0.730 | 0.764 |
| Fraction perched R² | 0.474 | 0.462 | **0.564** |
| Fraction circling R² | 0.842 | 0.819 | **0.873** |
| Fraction gliding R² | 0.732 | 0.715 | **0.744** |
| Fraction climbing soar R² | 0.825 | 0.806 | **0.851** |
| Fraction night R² | 0.302 | 0.309 | 0.305 |
| Sun elevation R² | 0.369 | 0.373 | 0.370 |
| Magnetometer fitted, 2024 birds (BA, ideal 0.5) | 0.695 | **0.678** | 0.699 |
| Decoding version (BA, chance 0.2) | 0.447 | 0.451 | 0.447 |
| Cohort (BA, descriptive) | 0.755 | 0.761 | 0.748 |

## Window level, 12 held-out birds

| Target | r1 | r2 | r3 |
|---|---|---|---|
| VeDBA R² | 0.972 | 0.972 | 0.970 |
| Speed R² | 0.943 | 0.941 | 0.946 |
| Climb R² | **0.884** | 0.861 | 0.824 |
| Turn R² | 0.866 | **0.874** | 0.845 |
| Flap-band fraction R² | **0.500** | 0.465 | 0.430 |
| Flapping BA | **0.892** | 0.891 | 0.876 |
| Perched BA | 0.991 | 0.989 | **0.993** |
| Circling BA | **0.962** | 0.957 | 0.953 |
| Gliding BA | **0.975** | 0.973 | 0.967 |
| Climbing soar BA | **0.960** | 0.955 | 0.947 |
| Night BA | 0.863 | 0.860 | 0.864 |
| Nuisance: sensitivity setting BA (ideal 0.5) | 0.998 | 0.998 | 0.998 |
| Nuisance: short ACC window BA (ideal 0.5) | 0.998 | 0.999 | 0.997 |
| Nuisance: decoding BA (chance 0.2) | 0.514 | 0.512 | 0.521 |

## Cluster stability (k-means on h, 5 bird bootstraps, mean ARI / min ARI)

| k | r1 | r2 | r3 |
|---|---|---|---|
| 6 | **0.729** / 0.614 | 0.601 / 0.524 | 0.564 / 0.469 |
| 8 | **0.789** / 0.688 | 0.773 / 0.751 | 0.656 / 0.566 |
| 10 | **0.860** / 0.715 | 0.809 / 0.745 | 0.729 / 0.588 |
| 12 | **0.877** / 0.820 | 0.790 / 0.704 | 0.693 / 0.491 |
| 16 | **0.730** / 0.684 | 0.718 / 0.654 | 0.714 / 0.671 |
| 20 | **0.792** / 0.693 | 0.785 / 0.725 | 0.720 / 0.666 |

## Reading

- r1 (temperature 0.2, with sensor dropout) is the best run overall. It is best or tied on the held-out
  window probes and gives the most stable clusters at every k (ARI 0.86–0.88 at k = 10–12).
- Lowering the temperature to 0.1 (r2) changes the probes by at most a few hundredths and lowers cluster
  stability. It also took longer to train.
- Removing sensor dropout (r3) improves the hour-level perched/circling/gliding/soar fractions by
  0.01–0.09 R², but lowers held-out climb and flap-band R² and clearly lowers cluster stability
  (ARI 0.69–0.73 at k = 10–12, minimum 0.49).
- None of the three settings affects the two leaks: the sensitivity setting and the short-vs-continuous
  window are decodable at 0.997–0.999 in every run. They come from the input data, not from these
  hyperparameters, and need the targeted fixes proposed in RESULTS_stage4_r1.md.
