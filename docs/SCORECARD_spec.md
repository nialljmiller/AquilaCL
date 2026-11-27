# Goal scorecard: specification (v2, 2026-11-27; revised after the supervisor review; frozen before any model is scored)

## Why this exists

Nill's goal is to characterise golden eagle behaviour and demography from this data:
- sex;
- daily, seasonal, year-to-year and lifetime patterns;
- behaviour types;
- odd behaviour;
- groups.

Until r7, each model round was judged on a different set of mostly 10-second checks. From now on every candidate design is scored on this one fixed list. The design search then aims at the weakest goals.

The metrics, masks, labels and thresholds below are fixed **before** any model is scored. Changing them after seeing scores needs a dated note in this file saying what changed and why.

## Bird split (fixed once; drawn by code, ids stored only on the cluster)

| Set | Birds | Use |
|---|---|---|
| Locked test | 20 | Never in encoder training, never scored during the search. Used once, at the end, for the finalists. |
| Search | 12 | The r7 held-out birds (`runs/r7_base_s0/holdout.json`). Never in encoder training. Window-level probes, identity and behaviour identification are scored on them during the search. |
| Training | the remaining 60 | Encoder training. |

**How the locked birds are drawn:**
- Only birds with windows are eligible, excluding both earlier held-out sets (r6: 12 birds, r7: 12 birds), because both informed decisions.
- The draw is stratified by sex × deployment year × young-tagged × migrant flag, with salt `aquilacl_locked_2026-11`.
- Method: eligible birds are sorted by stratum key, then by sha256(salt + bird) within the stratum. Every k-th bird of that list (k = eligible / 20, starting at offset ⌊k/2⌋) is locked. This allocates birds proportionally and handles strata of one or two birds without special cases (`scripts/make_split.py`).
- **What the locked set can confirm:**
  - window rows (G2, G6, G7, G9, G10): confirmatory;
  - bird-level rows (G1, G8): direction only. 20 birds give an AUC standard error of about 0.12.
- The split file (`split.json`) lives under `results_2026-11/v4/scorecard_split/`. No bird ids are committed.

**Consequences:**
- No existing checkpoint can be scored on the locked or search sets for identity or window probes, because all of them trained on some of those birds.
- The baseline is therefore the r6 recipe retrained on the training set.
- The baseline gets **5 seeds**.
- r7_noneigh (yaw correction + flip + masking, no neighbour positives) also gets 5 seeds on the same split. It was picked on the search birds after 466 metrics, so its search-set scores are mildly optimistic.
- Bird-level label goals (sex, groups) are fair on any checkpoint, because the encoder never sees labels. For those, existing checkpoints may be scored on all 72 non-locked birds.

## Shared rules

- **The bird is the unit.** Folds are grouped by bird and permutations move labels between birds. Every score reports bird counts.
- **Three columns per goal:**
  - model score;
  - **handcrafted reference**: the same protocol on `features.parquet` columns, in the run's own frame where posture is involved, and in the sensor frame as well;
  - **nuisance floor**: the same protocol on recording-format and deployment variables only.

  Designs are ranked on the **raw model score**.
  - The handcrafted reference is computed **once, in the sensor frame**, and is the same constant for every design. Own-frame values are descriptive only, because they change with the design's frame.
  - A selection row is suspended for a design when the model score minus the nuisance floor is below that row's minimum effect. The row is then reported as "explained by nuisance".
- **Repeated grouped CV:** 5 folds × 5 repeats; linear models (logistic or ridge, standardised, C/alpha fixed at 1.0); permutation p with 200 permutations at bird level.
- **Seed noise:** each design is trained with 5 seeds; see the selection rule.
- **Which birds:**
  - Window rows (G2, G6, G9, G9b, G10): the probe is trained on training birds and tested on the 12 search birds.
  - Bird and bird-day rows (G1, G3, G4, G5, G8, G8b): repeated grouped CV over the 72 non-locked birds. Each bird counts equally: per-bird sampling caps, and per-bird weights for bird-day rows.
- **Fixed sampling seeds:** every random draw (windows per bird, injection hosts and donors, CV folds, bootstraps) uses `numpy.random.default_rng([20261127, row_number])`, the same for every design.
- **Flight mask (fixed for every design):** the flight state of `states_r6_k2` (K=2 ground/flight from r6, column `state_hmm`; the flight state is the one with the higher median speed).
  - It is not recomputed per design, so all designs are scored on the same windows.
  - It comes from r6's h, which had GPS input and trained on some search and locked birds, so it carries some recording-mode structure. That does not bias comparisons between designs, and G1b separates the modes.
- **Environment inputs** (wind, terrain): any design using them has a twin without them, on the same split and seeds.
  - The rows listed under "Time and position information" below are scored **only on the environment-free twin**. For the environment design they are printed, marked "environment-informed: not comparable".
  - The reason: wind, boundary-layer height, heat flux and terrain follow daily and seasonal cycles and encode location, so they would raise these scores for trivial reasons.
- **Time and position information (any design):** some inputs or positional encodings are derived from clock time, date, sun, geographic position, or a window's index within a day or bout. Level-2 sequence models are the main case: scheduled bursts sit on a fixed clock, so position within a day sequence gives the hour.
  - For such designs, G2, G3, G4, G5, G7 (other-season part), G8 and G8b are scored on embeddings computed without that information: single windows, or random crops with no absolute position.
  - Otherwise these rows are printed as "time/position-informed: not comparable".
  - The environment rule applies to the same rows: G2, G3, G4, G5, G7 (other season), G8, G8b and G9.

## Goals

Selection rows (S) decide between designs. Descriptive rows (D) are reported but not used for selection (low power or not testable).

| Id | Goal | Kind | Unit and data | Metric (better) | Handcrafted reference | Nuisance floor |
|---|---|---|---|---|---|---|
| G1 | Sex | S | Bird: equal-weight mean of the per-mode (scheduled, continuous) mean h over flight-mask windows; a bird with flight windows in one mode only uses that mode; birds with sex and ≥ 200 flight windows, 72 non-locked birds; pre-fledge windows of young birds dropped | AUC (↑) | mean features over flight windows, plus bird medians of speed, turn radius and turn rate in continuous-mode flight windows (the stage-6 signal) | deployment year, continuous-mode share, GPS share, track days, deploy month |
| G1b | Sex by mode | D | as G1, separately for continuous-mode and scheduled windows | AUC | same | same |
| G2 | Daily rhythm | S | Window: scheduled-mode (k=0) full windows, sun elevation > 0 (daytime only, so night posture does not give the hour away); up to 2000 per bird | circular R² of local solar hour (↑) | features | gps_fixes, static_norm |
| G3 | Seasonal pattern | S | Bird-day: mean h of scheduled-mode daytime full windows; days with ≥ 50 such windows; birds weighted equally; report birds with ≥ 9 calendar months | circular R² of day of year, folds by bird (↑) | daily feature means | daily recording format (window count, short-window share, continuous-mode share), days since deployment, deployment year, young-tagged, movement class, daily mean static_norm |
| G4 | Seasonal repeatability across years | D (S if ≥ 15 birds qualify) | Bird-month: mean h of scheduled daytime full windows, ≥ 300 windows per bird-month, standardised over all bird-months, cosine distance; birds with the same calendar month in ≥ 2 years | Per bird, AUC that same-month-different-year pairs are closer than different-year pairs ≥ 3 months apart; mean over birds; sign-flip test (↑) | features | — |
| G4b | Year-to-year difference | D | Bird-month as G4 | Within bird, does calendar year separate same-month pairs? Descriptive only: calendar year is confounded with tag age and deployment year | features | — |
| G5 | Lifetime (young-tagged birds) | D | Bird-month as G4; young-tagged birds with the same months in year 1 and year 2 | Leave-one-bird-out direction, pairwise AUC that the year-2 month scores "older" (↑); report n birds. Confounded with calendar year (17 of 19 young birds are 2024) and tag age | features | — |
| G6 | Behaviour identification from acceleration alone | S | Window: continuous-mode full windows with ≥ 8 GPS fixes. Labels from GPS only (below). h is recomputed with the GPS mask zeroed. Probe trained on training birds, tested on search birds | mean one-vs-rest AUC over qualifying classes (↑); per class reported. A class qualifies with ≥ 30 test positives in total, coming from ≥ 3 search birds | acceleration-only features: vedba, vedba_sd, odba, static_norm, pitch/roll, flap_band_frac, dom_freq_hz | — |
| G7a | Odd bouts | S | anomaly v3 bout injection, with one change: the host's and donor's "mix" is defined model-free, from shares of the G6 GPS-only labels in the bout (not from K=6 states). Host and donor lists are drawn once and frozen (`scorecard_split/injections.parquet`) and reused for every design | AUC of unseen-mix bouts minus AUC of same-mix bouts, the identity-corrected score (↑) | features through the same protocol | — |
| G7c | Odd days | S | anomaly v3 day injection: a day replaced by the same bird's day from another season; frozen list | AUC (↑) | same | — |
| G7d | Odd order | D | shuffled bout order (it rises mechanically with any sequence model or neighbour training) | AUC | same | — |
| G7b | Identity control | D | same-mix bout from another bird | AUC (↓; detection here means the score reacts to identity) | same | — |
| G8 | Movement-strategy groups | D, or S if the nuisance floor < 0.9 | Bird: track-derived class from `residency_r6_v3b` (migrant or migrant-departed vs every other class with sufficient track); mean h over flight-mask windows outside Apr–Oct | AUC, sex-style CV, permutation within deployment year | same | as G1 plus young-tagged and winter-window count. The floor is evaluated once, before any model is scored. Floor ≥ 0.9 means "not testable in this design". |
| G8b | Unsupervised bird groups | D | Bird: G1's bird vector, standardised; k-means k = 2..5 | the k with the highest minimum bootstrap Jaccard (200 bootstraps), with that Jaccard; stable when ≥ 0.75 (stage-6 rule) | same | — |
| G8c | Behaviour types | D | Window: k-means / GMM on h of the 72 non-locked birds, K = 2..10, the stage-6 stability rule (states.py); ground sub-states as in substates.py | largest K meeting the rule, and stable sub-states per coarse state | — | — |
| G9 | Individual identity | S | Window: 12-way bird classification on the search birds (existing evaluate.py protocol: fit on the first half of each bird's time span, test on the second) | balanced accuracy `bird_id_holdout_h` (↓); per-class recall reported | features (sensor frame) | — |
| G9b | Recording mode from acceleration alone | D | Window: full windows, GPS mask zeroed; continuous vs scheduled | balanced accuracy (↓) | features | — |
| G10 | Kinematic guard rails | guard | Window, search birds: speed, climb, VeDBA, fore-aft tilt (physical meaning, whichever column), wingbeat frequency in flapping windows | R² (must not drop by more than 0.02 absolute) | features | — |

**Circular R²** (G2, G3): the target angle θ becomes y = (cos θ, sin θ); a ridge model predicts ŷ (2 columns). R² = 1 − Σ‖ŷ − y‖² / Σ‖y − ȳ‖², over test rows (bird-weighted for G3).

**GPS-only behaviour labels for G6** (from per-second `gps.npy` channels; speed = `ground-speed`, climb = `climb_ms`, net heading change = |Σ `turn_rate_dps`| over valid fixes; at least 8 valid fixes):

| Label | Rule |
|---|---|
| perched | median ground speed < 1 m/s |
| circling | median speed > 4 m/s and \|net heading change\| ≥ 180° in the window |
| climbing straight | median speed > 4, mean climb > 0.5 m/s, \|net heading change\| < 90° |
| gliding | median speed > 8, mean climb < −0.5 m/s, \|net heading change\| < 45° |
| steep descent | min climb < −5 m/s and max speed > 15 m/s |
| take-off | median speed of the first 3 fixes < 2 m/s and of the last 3 fixes > 5 m/s |
| landing | the reverse of take-off |

- Labels may overlap. Each label is scored one-vs-rest (class-weighted logistic), and the G6 score averages the AUC over the labels that qualify.
- Perched is rare in continuous mode (1,550 of 2,041,063 full windows are still), and take-off is rare (bouts start at a median 11.6 m/s). Before the spec is frozen, the qualifying classes and the search birds with continuous windows are counted. The class list is then written here with a dated note.
- The existing `features.parquet` labels use VeDBA (acceleration). They are reported in G6 as a secondary row only, because predicting them from acceleration is partly circular.

## Selection rule

A difference on a row is **real** only when all three hold (5 seeds per design):
1. the mean paired difference exceeds 2 × the paired seed sd;
2. it exceeds the row's minimum effect, fixed now:

   | Rows | Minimum effect |
   |---|---|
   | AUC rows (G1, G6, G7a, G7c, G8) | 0.03 |
   | R² rows (G2, G3) | 0.02 |
   | G4 AUC | 0.03 |
   | G9 balanced accuracy | 0.02 |

3. the 95% interval of a paired bootstrap over birds (1000 resamples of the scored birds, the same resamples for both designs) excludes 0.

Design A beats design B when at least one selection row (G1, G2, G3, G6, G7a, G7c, G9; G4 and G8 if they qualify) is really better, and no selection row is really worse. In addition, no guard rail may drop by more than its absolute tolerance. Otherwise the result is a tie, which is reported as such.

The locked test birds are scored only for the final two or three designs.

## Order of work

1. Implement `scorecard.py` and the split, then run it on r6_yaw60_bpb4 and the r7 runs. Bird-level rows on 72 birds; window rows marked as contaminated where the run trained on search birds.
2. Retrain the r6 recipe and r7_noneigh on the new split (3 seeds each) and score them. This is the baseline and the per-row seed noise.
3. Level-2 model (sequence of window embeddings over bouts and days) and level-3 bird-month summaries, scored on the same card.
4. Environment inputs (wind, terrain) on level 2 only, with twins.

## Change log

- 2026-11-27 v2, after the supervisor review (`docs/review_fable_scorecard_2026-11-27.md`):
  - time/position rule;
  - model-free injection mixes and frozen lists;
  - G7 split into rows;
  - 5 seeds, minimum effects and bird bootstrap;
  - raw-score ranking with a constant sensor-frame reference;
  - stronger G1 reference;
  - G3 nuisance floor;
  - locked set confirmatory for window rows only;
  - which birds each row uses;
  - G6 definitions;
  - G4 specified and G4b added;
  - G8 floor;
  - G8b criterion and G8c added;
  - G9 metric named;
  - circular R² defined;
  - fixed seeds.
