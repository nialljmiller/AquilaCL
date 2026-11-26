# Stage 6: states from K=2, recording trigger, residency, sex and age (2026-11-25)

Data: windows_v4, 92 birds. Models: r4_v4 (identity kept) and r6_yaw60_bpb4 (identity reduced).
All demographic tests use the bird as the unit: folds are grouped by bird and permutations shuffle labels between birds.

## 1. Number of states

`states.py` previously searched K from 6 upwards (`k_min=6`). With `K_MIN=2` (jobs 62225 and 62226):
- **Both runs choose K=2, and the stability rule is met** (mean bootstrap Jaccard 1.00, held-out ARI 1.00). The two states are ground (VeDBA 0.008 g, no flight speed) and flight (VeDBA 0.09 g, 11.9 m/s).
- **Every K from 3 to 10 fails the rule.** The minimum Jaccard is 0.25–0.74 (r6 best: K=3 at 0.675; r4 best: K=6 at 0.735).

Sub-states (job 62227, r6, rule fixed in advance):
- **Ground splits into 4 stable sub-states** (Jaccard 0.78–0.91):
  - 0.0: very still, GPS absent, 25% at night, sensitivity flag raised, so it may follow recording mode;
  - 0.1: still, tilted −33° fore-aft (column `roll_rot_deg`, which is fore-aft tilt in this frame; see RESULTS_r7.md);
  - 0.2: low activity, VeDBA 0.02 g;
  - 0.3: active on the ground, VeDBA 0.13 g.
- **Flight does not split stably at any k_sub.**

Reading: at a 10-s scale, flight forms a continuum in the representation (speed, climb and turn vary smoothly) and not separate clusters. The six-state solution describes that continuum usefully, but it is not a set of stable discrete classes. Flight types are better described by their continuous coordinates or at a longer time scale.

## 2. What starts continuous recording (`scripts/check_trigger_migrant.py`, `trigger_migrant_bout60.json`)

Bouts of 1 Hz GPS lasting at least 60 s: 53,584 bouts in 80 birds.
- At the first fixes of a bout the bird already moves at a median 11.6 m/s (above 5 m/s in 91% of bouts).
- Bouts last a median of 802 s, capped near 900 s.
- Onsets fall mostly between 17 and 22 UTC.
- After a scheduled fix faster than 5 m/s, a bout follows within 15 min 29% of the time (per-bird median), against 13% after a fix slower than 1 m/s.

This is consistent with a speed or movement trigger; the firmware rule itself is unknown. Continuous mode is therefore conditioned on flight. The scheduled bursts, every 5 min for 81 birds, are the behaviour-independent sample.

## 3. Meaning of 'migrant'

Twenty birds carry the 'migrant' flag. Distances are measured from the study-area centre; no coordinates are stored or reported.
- 15 of the 20 travel more than 1000 km north: median maximum 2,074 km north, 2,876 km from the centre.
- They are furthest north in May–June, and a median 64% of their fixes are more than 500 km away.

This agrees with the owner's description: migrants spend summer far north, towards Alaska, and winter in the Bighorn Basin. However, 11 of 55 birds with neither a migrant flag nor a young tag also go more than 1000 km north. Residency should therefore be derived from the tracks, not taken from the flag. No young-tagged bird goes more than 1000 km north.

Every migrant-flagged bird is in the 2024 cohort, and most were deployed in October; residency is confounded with both.

## 4. Sex (`demography_r4_r6_k6`, job 62233; per-state follow-up run interactively)

All values are bird-level AUCs: 86 birds (41 f, 45 m), repeated grouped CV, permutation p.

| Feature set | r4 | r6 |
|---|---|---|
| Nuisance floor (cohort, continuous-mode share, GPS share, track days, deploy month) | 0.48 | 0.48 |
| Handcrafted means | 0.42 | 0.42 |
| Mean h | 0.68 (p 0.003) | 0.77 (p 0.0005) |
| Mean h within each of 6 states | 0.84 (p 0.0005) | 0.87 (p 0.0005) |
| h beyond nuisance (Freedman–Lane), ΔAUC | +0.24 (p 0.0005) | +0.29 (p 0.0005) |

**Sex is read only from flight, not from rest.** AUC from mean h within one state (5 CV repeats, 100 permutations):

| State | r4 | r6 |
|---|---|---|
| Rest | 0.44 | 0.41 |
| Perched / perched-active | 0.45 / 0.51 | 0.52 |
| Thermal circling / soaring | 0.85 (p 0.01) | 0.65 (p 0.02) |
| Fast glide | 0.78 (p 0.01) | 0.86 (p 0.01) |
| Cruising | – | 0.80 (p 0.01) |
| Flapping | 0.60 (p 0.07) | 0.56 (p 0.18) |

Harness fit or posture at rest would show at rest. It does not, so the signal comes from how the birds fly.

**Direct check against the wing-loading prediction.** Females are larger and have higher wing loading, so they should fly faster and turn more widely. The table gives bird-level medians in the r6 states (32 f, 29 m; 5000 permutations).

| State | Variable | f | m | f − m | p |
|---|---|---|---|---|---|
| Thermal circling | Speed (m/s) | 10.53 | 10.07 | +4.5% | 0.0004 |
| Thermal circling | Turn radius (m) | 29.9 | 28.2 | +6.0% | 0.027 |
| Thermal circling | Climb (m/s) | 1.43 | 1.40 | +1.9% | 0.66 |
| Cruising | Speed (m/s) | 11.48 | 10.71 | +7.2% | 0.0012 |
| Cruising | Turn radius (m) | 118.5 | 100.5 | +17.9% | 0.0002 |
| Fast glide | Speed (m/s) | 18.61 | 18.26 | +1.9% | 0.26 |
| Fast glide | Turn rate (°/s) | 1.73 | 1.85 | −6.7% | 0.010 |

The direction matches the prediction (review_ecology_2026-11.md §1).

Caveats:
- 12 tests, uncorrected. The speed and radius results at p ≤ 0.0012 survive a Bonferroni correction.
- All ages are pooled.
- Sex differs in track length (f 302 vs m 208 days, p 0.05).

**Wingbeat frequency** (positive control): no sex difference (f − m −0.005 Hz, p 0.90). The literature predicts only about 1%, so this control has no power at n = 64 birds and is not decisive.

## 5. Age

**Young-tagged vs age not stated (19 vs 73).** The nuisance floor alone reaches AUC 0.97, because 17 of the 19 young-tagged birds are in the 2024 cohort and all were deployed in June–July (12 in June, 7 in July). h adds nothing beyond it (ΔAUC −0.04, p 0.43). Age at tagging therefore cannot be separated from cohort and deployment timing in this design.

**Within the same birds, July–September of year 1 vs year 2 (6 birds, r4 states, paired):**

| Measure | Year 1 | Year 2 | Birds increasing | p (sign flip) |
|---|---|---|---|---|
| Perched | 0.33 | 0.17 | 1/6 | 0.06 |
| Low activity | 0.16 | 0.08 | 1/6 | 0.06 |
| Soaring | 0.18 | 0.38 | 5/6 | 0.09 |
| Fast glide | 0.04 | 0.12 | 5/6 | 0.06 |
| Speed (m/s, 4 birds) | 9.7 | 12.9 | – | 0.13 |

The direction is consistent with flight development (less perching, more soaring and gliding, faster flight), but six birds give no significant result.

## 6. Season and outliers (analysis_r6_k2, job 62228)

**Season.** The flight-share seasonal range is 0.59 over all windows, but only 0.13 within scheduled-mode windows and 0.18 after mode adjustment. Most of the raw seasonal swing is the recording schedule; about 0.13–0.18 remains as a real seasonal change.

**Outliers.**
- 90% of the top 1% of outlier windows are short windows (ratio 1.58).
- Only 2% are in continuous mode (ratio 0.06) and 5% have GPS (ratio 0.16).
- The current outlier list is therefore mostly a recording-format artefact. Two-level anomaly scoring (review_methods §5) is needed before outliers are interpreted.

## Next

1. Time budgets from the scheduled stream only.
2. Describe flight as a continuum, and at a longer time scale (minutes), in place of discrete classes.
3. Derive residency from the tracks, and test migrants against residents within cohort.
4. Two-level anomaly scoring.
5. Re-run the sex flight test within age class and within cohort.

## 7. Sex in flight, stratified by cohort and age group (run 2026-11-25, interactive; r6 K=6 states)

Each value is the female − male difference in bird-level medians. The overall p comes from permuting sex within cohort, and separately within age group (young-tagged vs not stated).

| State | Variable | All birds | p within cohort | p within age group | Cohort 2024 | Cohort 2025 | Age not stated | Young-tagged |
|---|---|---|---|---|---|---|---|---|
| Thermal circling | Speed | +4.5% | 0.0010 | 0.0008 | +4.3% (p 0.015) | +4.7% (p 0.014) | +3.7% (p 0.006) | +6.9% (p 0.032) |
| Thermal circling | Radius | +6.0% | 0.030 | 0.027 | +7.0% (0.09) | +4.0% (0.26) | +3.8% (0.19) | +9.7% (0.11) |
| Cruising | Speed | +7.2% | 0.0012 | 0.0012 | +8.1% (0.002) | +5.9% (0.10) | +6.5% (0.009) | +8.4% (0.08) |
| Cruising | Radius | +17.9% | 0.0004 | 0.0004 | +26.6% (0.001) | +9.4% (0.15) | +16.0% (0.003) | +32.3% (0.008) |

- The direction is the same in every subgroup (females faster, wider turns).
- Subgroup sizes: cohort 2024 f18/m12, cohort 2025 f14/m17, not stated f24/m25, young-tagged f8/m4.
- The effect is not explained by cohort or by age at tagging.
