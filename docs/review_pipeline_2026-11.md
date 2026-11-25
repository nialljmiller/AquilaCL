# Independent review of the AquilaCL pipeline (2026-11-25)

Scope: repository HEAD `c9b4935` plus the working tree of 2026-11-25. The working tree contained an uncommitted
`src/aquilacl/analysis.py` and an untracked `src/aquilacl/substates.py`; other agents were editing both, so they
are cited by function name, not by line. `reference.py` and `demography.py` did not exist when this review was
written. Read: README.md, docs/DESIGN.md, docs/RESULTS_stage3.md, docs/RESULTS_stage4_r1.md,
docs/RESULTS_stage4_r1-r3.md, docs/RESULTS_stage4_r4-r6.md, all of `src/aquilacl/*.py`, all of `scripts/*.py`
(diagnose_r4.py by structure and docstring). No data were read: they are on AI4WY. Every statement about the data
comes from the results documents. Statements marked **verify** are inferences that one query on the data would
settle.

Goal under review (owner): "use this data and deep learning to characterise the behaviours of golden eagles and
their demographics". That covers male vs female, young vs old, unusual individuals and windows, common behaviours,
and seasonal behaviours.

## 0. Verdict in brief

The data engineering is now careful and largely correct. Ingest, calibration and the leak diagnostics are better
than most published biologging pipelines. The representation and state results, however, do not yet support any
of the owner's stated goals:

1. **Window counts are not time budgets.** Recording mode is tied to behaviour, so every "common behaviour" and
   "seasonal" share mixes behaviour with duty cycle (C1).
2. **The identity reduction (r5/r6) may remove the demographic signal.** The behaviour probes that "show no loss"
   cannot see what a ±60° yaw removes, by construction (C2).
3. **Age, cohort, tag model and observation season are one variable**, and the sex null result has no stated power
   (C3).
4. **The "six states" are set by the lower bound of the K search**, not found (C4).
5. **The evaluation labels are thresholds on the model's own inputs.** High probe scores show that information is
   retained. They do not show that the states are behaviours (I1).

## 1. What is sound (keep it)

- **Ingest** (`ingest.py`): refuses to append into an existing output (l.113), checks row counts against known
  totals (l.198–209), and removes duplicate event ids after sorting.
- **Sensitivity diagnosis.**
  - Identifying the setting per burst type rather than per day (DESIGN §6) was a real catch, and the evidence
    behind it is quantitative: within-tag ratio 2.001 ± 0.050 over 62 tags.
  - `acc.group_sensitivity` and `acc.calibrate_tag` (donor scaling by 2^Δk, flags) are sound.
  - The common format (`dataset.common_format`) is a clean way to hide step and range.
- **The re-quantisation counterfactual** (diagnose_r4 Part 1) is an exemplary test. It changes only the
  amplitude grid, with random phase, and shows 0% flips of the sensitivity classifier. It settles that the 0.9999
  probe reads recording mode, not amplitude.
- **Leak-aware feature engineering.**
  - Static component per contiguous run (`features.static_component`).
  - NaN labels where inputs are missing (`features.reference_labels`).
  - Turn-rate masking for short steps and intervals (`gps.kinematics`).
  - Contiguous placement of continuous-mode bursts (`windows.place_bursts`).
- **Nuisance probes reported against a handcrafted floor**, with bird-level bootstrap intervals and paired
  differences (`evaluate.nuisance_vs_features`). This is the correct framing.
- **Statistical unit and time of day.** Solar time and sun elevation are used instead of UTC; bird-level
  statistics are used (`analysis.permutation_test` on per-bird budgets).
- **Hygiene.**
  - A config guard on resume (`train.record_config`).
  - Deterministic, augmentation-free embedding (`train.encode`).
  - Outputs are never overwritten.
  - Superseded results are marked as such in their documents.
- **Privacy.** Lat/lon are never model inputs, and the window-level outputs bin solar hour and sun elevation.

## 2. Critical issues

### C1. Recording mode is behaviour-dependent: window shares are not time budgets (critical)

**Evidence**
- RESULTS_stage4_r4-r6 §1, full 200-sample windows of the 32 birds that use both settings:
  - 99.99% of k=1 (continuous-mode) windows have GPS;
  - only **25 of 91,913** k=1 windows are still;
  - 0.0005% of k=0 windows have GPS.
- DESIGN §5:
  - 1 Hz GPS runs have a median length of 9 s, and 10% reach the 899 s cap;
  - 60% of accelerometer time is in continuous 5–60 min segments.
- Taken together, continuous ACC plus 1 Hz GPS is recorded almost only when the bird flies. This looks like a
  flight-triggered or flight-scheduled mode. Whether it is triggered, and by what, is **verify / owner question
  Q1**.

**Why it breaks the goals**
- `windows.build_bird` tiles each continuous segment into consecutive 10 s windows. A 30 min flight segment gives
  180 windows; a scheduled 2 s burst on a perch gives 1.
- `analysis.per_bird_budget` and `analysis.bird_share` count windows. "Common behaviours" (state shares, e.g.
  "Rest 26.6%, Flapping 10.4%" in RESULTS r4–r6 §4) are therefore shares of *recorded windows*, not of time.
- Seasonal curves (`m5_seasonal.png`) inherit any change in the trigger or schedule across the year. The DESIGN
  §5 note "double sensitivity mostly April–September 2025" comes from the superseded per-day rule and is only a
  hint. The `month_share` output of `scripts/sensitivity_cues.py` (job 62099) is the check.
- e-obs tags also commonly throttle sampling on battery voltage, i.e. sunshine (**owner question Q1**). That would
  couple the amount of data to weather, and weather to soaring.
- A related point (**verify**): the GPS-dependent labels (perched, circling, gliding, climbing_soar;
  `features.py:170` `gps_ok` needs ≥5 fixes) probably exist almost only in continuous-mode windows. The same would
  then hold for the speed and climb medians in the state table: in sparse-mode windows those columns are NaN, so
  the state descriptions would be descriptions of continuous-mode data. Check: cross-tabulate
  `label_perched.notna()` against `sensitivity_k` and full vs non-full. `windows_summary.csv` already holds the
  `*_known` counts per bird.
- **The fix below has its own catch.** The scheduled sample proposed as the time-budget base probably carries
  almost no GPS: about 0% of full k=0 windows (the 264-sample bursts) have GPS. Budgets from that sample would
  then depend on the encoder assigning states from acceleration alone. States that are separated mainly by GPS
  (thermal circling vs cruising vs fast descending glide) may not be separable there at all.
  `evaluate.invariance_tests` already measures this as `gps_removed_same_state_by_state`.

**Fix**
1. Obtain the full schedule and trigger configuration per tag and period (Q1).
2. Define the **time-budget sample** as the regularly scheduled bursts only (e.g. the fixed-interval 40- or
   264-sample bursts), one window per burst. Use continuous segments only for within-bout structure (dwell,
   transitions, wingbeat).
3. If trigger rules exist, weight each window by the inverse of its sampling intensity per bird-day-mode.
4. Stratify every seasonal and demographic comparison by mode, or include mode as a covariate. Report coverage per
   mode next to every share.
5. Correct scheduled-sample budgets for ACC-only misassignment. Estimate an ACC-only confusion matrix on
   continuous-mode windows with GPS removed (as in `gps_removed_same_state_by_state`), then invert it or report
   state shares as ranges. Merge states that ACC alone cannot separate for budget purposes.

**Effort**: 0.5 day of code (a `regime`/`scheduled` flag in `windows.py` meta and weights in `analysis.py`) once
the schedule is known. Getting the schedule from the owner and the e-obs configuration files is the bottleneck.

### C2. Identity reduction conflicts with the demographic goal, and the probes cannot detect what it removes (critical)

**Evidence**
- r6_yaw60_bpb4 was chosen because bird identity fell from 0.491 to 0.313 while "all runs are within ±0.01 of r4
  on the window-level probes" (RESULTS r4–r6 §3, §5).
- The window probe set is `evaluate.py:47–48`: VeDBA, speed, climb, turn, flap_band_frac, sun elevation, and six
  labels.
  - VeDBA is the norm of the dynamic vector, so it is **rotation-invariant**.
  - Speed, climb and turn are GPS channels and are **untouched by any ACC rotation**.
  - The labels are thresholds on those same quantities.
  - Only `flap_band_frac` (max-variance axis) is partly axis-dependent, and it was already the weakest target
    (R² 0.43–0.50 in r1–r3).
- A yaw rotation of ±60° about gravity mixes surge with sway and pitch with roll. That is the information that
  separates flight posture, bank angle in circling, and flapping kinematics. The probe set contains **no target
  that a yaw rotation can change**, so "no behaviour lost" was close to guaranteed before the runs started.
- The binary labels also sit at ceiling (BA 0.95–0.99), where a loss of detail cannot show.
- The yaw range is about 4× the nuisance it was meant to cover: residual mounting-yaw SD across birds is 15.8°
  (r4–r6 §2).
- Bird identity is not purely a nuisance. Females are larger than males (reverse size dimorphism), so wing loading
  and wingbeat frequency are expected to differ by sex, and flight skill changes with age. Below the 8-feature
  handcrafted floor (h − features = −0.063), h carries *less* individual information than eight summary numbers.
  That is consistent with invariance; it is equally consistent with lost behavioural detail.
- The bird-confusion adversary changed nothing on held-out birds (+0.001). Fewer birds per batch (r5_bpb4) changes
  the negatives, not the input. Both are cheaper identity levers than yaw, but neither has been tested for
  demographic signal.

**Fix**
1. Add rotation-sensitive probes:
   - gravity-frame pitch and roll in flight windows (`pitch_rot_deg`, `roll_rot_deg` already exist in
     states.parquet);
   - longitudinal/lateral dynamic-acceleration ratio;
   - `dom_freq_hz` in flapping windows;
   - signed roll during circling (bank angle).
   Compute these targets only **after** a per-bird yaw de-rotation (item 2). Otherwise they carry each bird's
   residual mounting yaw (SD 15.8°), and a higher score for r4 than for r6 could just mean that r4 reads mounting
   identity. Add tilt magnitude (angle from vertical), which yaw does not change, as a control target.
2. Replace the ±60° yaw augmentation with a **deterministic per-bird yaw de-rotation** from the flapping axis.
   `axial_angle` and `rot_z` already exist in `scripts/diagnose_r4.py`. Then keep a small (±15°) yaw augmentation.
3. Before choosing a model, run a **bird-level sex probe** on r4 and r6: per-bird mean h, leave-one-bird-out
   logistic regression, permutation null. Run an **age probe** where age is known. Choose the model that keeps
   behaviour *and* demographic decodability while dropping only device and mounting identity.
4. Report identity *within behaviour subsets*, matched on behaviour mix, rather than overall. Overall identity also
   tracks each bird's behaviour budget.

**Effort**: probes 0.5 day of code and 1 CPU-hour; de-rotation 1 day plus one training run (~1.5 h on one H200).

### C3. Age, cohort, tag model and observation season are confounded; the sex result has no stated power (critical for demographics)

**Evidence**
- DESIGN §2c: "2025 = mostly territorial adults seen Oct–Mar; 2024 = mostly nestlings".
- Cohort also differs in tag hardware (magnetometer; DESIGN §1) and possibly in schedule.
- Cohort is computed from the year of a bird's first window (`dataset.py:159–160`, `states.py:149–150`,
  `baseline.py:130–131`). Deployment date is not used.
- Life stage "not tested: the reference data contain none" (r4–r6 §4). Yet field records exist:
  `docs/analysis_inventory.md` counts "nestling" in 16 comments, "fledgling" in 1 and "migrant" in 21, and DESIGN
  §2.7 plans "field-recorded age (nestling/fledgling at tagging)". `analysis.load_reference` correctly refuses to
  parse comments, so the result is "no age analysis" rather than "no age data".
- Sex: "no difference in any state share (permutation p 0.14–0.89)", 48 m / 41 f.
  - The test is `analysis.permutation_test` on six coarse state shares, unadjusted for cohort, age, month
    coverage, mode or breeding status (the output says so).
  - No effect sizes, intervals or power are given. With about 89 birds and between-bird SDs of state shares that
    are likely large (coverage differs in season and mode), only large differences are detectable. "No difference"
    is not supported; "no large difference in these six coarse shares" is.
- Survivorship: nestlings that die or lose tags drop out. Curves against "months since deployment"
  (`m6_development.png`) are then biased toward survivors, and each later bin has fewer, different birds.
- Within the 2024 cohort, age increases with calendar date, so age and season are collinear unless several hatch
  years overlap (the inventory reported the same collinearity for the first draft).

**Fix**
1. An owner-curated per-bird table:
   - age class or hatch year at tagging, with the method used;
   - sex, with the method used (genetic or morphometric);
   - breeding or territorial status per season;
   - track end cause.
2. Compare age classes only within calendar months where both classes have data. Use mixed models for per-bird
   shares or features, for example:
   - `share ~ age_class + sex + month(cyclic) + mode + (1|bird)`;
   - window-level `dom_freq_hz ~ sex + age + (1|bird)` in flapping windows.
3. Report effect sizes with bird-bootstrap 95% intervals, and a minimum detectable effect for each comparison.
4. **Positive control.** Test the predicted sex difference in wingbeat frequency and glide speed directly on
   handcrafted features. If the pipeline cannot see a size-driven biomechanical difference, it cannot answer the
   sex question from the latent space either.
5. Model survivorship explicitly: report n birds per bin, and run sensitivity analyses restricted to birds tracked
   for at least N months.

**Effort**: analysis code 1–2 days. The age table depends on the owner.

### C4. "Every run gives the same six states" is set by the K search bound (critical for "common behaviours")

**Evidence**
- `states.run(k_min=6)` (`states.py:343`, CLI default l.471; `slurm/05_states.sh` does not override it) searches
  K = 6…30.
- The rule "largest K with every Jaccard ≥ 0.75" was **not met in any run** (r4–r6 §4: minimum Jaccard
  0.50–0.74).
- The fallback (`states.py:425–426`) takes the K with the highest minimum Jaccard. Minimum Jaccard generally falls
  as K grows, so the fallback is biased toward the smallest K allowed, i.e. 6. All four runs returned exactly 6.
- "The same six states" across runs is asserted from the feature medians. No cross-run assignment agreement
  (matched states, ARI on common windows) is reported.
- The states are also only partly stable by the pipeline's own criterion: r6's minimum Jaccard is 0.62, and a
  Jaccard < 0.75 means the state is not reproducible under Hennig's guideline.

**Fix**
1. Rerun stage 5a with `--k-min 2`. Report the Jaccard curve across K.
2. Report cross-run agreement: Hungarian-matched states, and ARI on the same evaluation windows for r4 vs r6 and
   for seeds.
3. If no K ≥ 3 is stable, say so, and present the states as a partition of a continuum. The structure in h may be
   graded (speed × VeDBA), not discrete. The handcrafted baseline's "BIC kept decreasing up to 16" (RESULTS_stage3)
   points the same way.
4. Run the **handcrafted baseline through the identical states.py procedure** (standardised features in place of
   h) so that the "SimCLR must beat the baseline" acceptance test of DESIGN §2.6 is actually run on v4 data.
   `baseline.py` uses a different procedure (full-covariance GMM, BIC choice, 5 bootstraps, ARI) and was only run
   on the superseded profile_v2 data.

**Effort**: about 2–4 CPU-hours per run of states.py, plus 0.5 day for the matching code.

## 3. Important issues

### I1. Evaluation labels are functions of the model inputs: the probes measure retention, not behaviour (important)

- `features.reference_labels` (l.142–185) defines perched, circling, gliding and climbing_soar as thresholds on
  `speed_mean` (the `ground-speed` input channel), `turn_abs_mean` (`turn_rate_dps`), `climb_mean` (`climb_ms`)
  and `vedba` (ACC). These per-second values are given to the GPS encoder almost verbatim (`windows.GPS_CHANNELS`,
  `dataset.GPS_SCALE`).
- A linear probe that recovers a threshold of an input shows only that h keeps that input. It would score high for
  any information-preserving encoder.
- The state descriptions ("Thermal circling: turn 19.9°/s, climb +1.43 m/s") are medians of the same inputs. They
  describe clusters; they do not validate them.
- The thresholds themselves have not been validated:
  - `flap_vedba = 0.25` g and `soar_vedba = 0.2` g;
  - `circle_turn_dps = 6`;
  - the climb thresholds ±0.5 m/s. `climb_mean` over 10 s is essentially (z_end − z_start)/Δt, and GPS
    height-above-ellipsoid noise of a few metres gives about 0.3–0.8 m/s of noise, comparable to the threshold.
    Verify by computing the SD of `climb_mean` in windows with speed < 1 m/s.
- **Fix**:
  1. Calibrate "high" with two reference encoders on the same probes: a **randomly initialised encoder** and
     **raw flattened inputs**. If a random encoder also scores BA 0.95, the probe says nothing about SimCLR.
  2. Obtain independent ground truth (§6.1).
  3. Re-derive label thresholds from a small expert-annotated set.
- **Effort**: random-encoder and raw baselines 0.5 day plus 1 CPU-hour.

### I2. The 12 held-out birds are no longer a test set; training variance is unmeasured (important)

- The same 12 birds (`dataset.holdout_birds`, 6 per first-window-year cohort) were used to choose among r4, r5×3
  and r6×2, and to define the identity and stability numbers that drove the choice. Any number reported for r6 on
  these birds is a validation number and is optimistically biased.
- Each run has one seed. The bird-bootstrap intervals in r4–r6 §3 cover bird sampling, not training variance.
  r5_conf's Δ = +0.001 and the ±0.01 behaviour differences are below any plausible seed-to-seed variation.
- 12 birds give wide intervals (identity 0.219–0.413). With 6 per cohort, a cohort-specific statement rests on 6
  birds.
- **Fix**:
  1. Freeze a second, untouched set of birds now, stratified by sex, age class and cohort, and do not look at it
     until the final model.
  2. Alternatively, use 5-fold bird cross-validation of the whole training pipeline for the final model: 5 runs of
     ~1.5 h each.
  3. Use 3 seeds for any run that is compared to another.
- **Effort**: 5–8 GPU-hours, plus 0.5 day of code for the fold option.

### I3. The identity probe confounds identity with season (important)

- `evaluate.nuisance_vs_features` fits on the first half of each held-out bird's record and tests on the second
  half (`evaluate.py:318–326`). The behaviour mix changes between halves (season, age, breeding), so the score
  mixes identity with temporal drift. The direction of the bias depends on how much behaviour changes.
- **Fix**: interleave by day (alternate days for fit and test), report both splits, and match the behaviour mix
  between fit and test (stratify by coarse state).
- **Effort**: 1–2 hours.

### I4. The HMM is nominal (important)

- Transitions are counted from GMM argmax labels plus 1 (`states.transitions`, l.246–250, used at l.424). They are
  not fitted by EM, and emissions are fixed.
- Chains exist only inside continuous ACC segments (`states.chains` breaks at any gap), i.e. only in the
  flight-biased continuous mode (C1). Sparse-mode windows are single-window chains and keep the GMM state.
- HMM–GMM agreement is 98% (r4–r6 §4), so the smoothing changes 2% of labels.
- Dwell times (`profiles`, dwell_*) are truncated by segment length, so they describe recording bouts rather than
  behaviour bouts.
- **Fix**:
  1. Either call this "label smoothing" and drop HMM claims, or fit a proper HMM or HSMM (e.g. with explicit
     duration distributions) on chains of length ≥ N, with EM and per-bird random effects or covariates (solar
     elevation, season).
  2. Report dwell with right-censoring (Kaplan–Meier), since segment ends censor bouts.
- **Effort**: 1–2 days.

### I5. Ten-second windows cannot represent minute-scale behaviour; GPS outside windows is discarded (important)

- At State 2's median turn rate of 19.9°/s, one circle takes about 18 s, so a 10 s window holds about half a
  circle. It cannot tell a single turn from sustained thermalling, nor measure climb per circle.
- Thermal climbs last minutes, inter-thermal glides 1–5 min, and migration or dispersal bouts hours to days.
- Only windows anchored on ACC bursts exist (`windows.build_bird`). GPS fixes that do not fall inside an ACC window
  are dropped (l.146–163). This includes the sparse fixes (`frac_dt_30_to_400s`, `frac_dt_gt_400s` in
  schedule.csv) that carry daily movement, range use and migration.
- `profile_bird` writes `gps_runs/<bird>.parquet` (`profile.py:160–164`; duration, path, net turn, climb,
  straightness per run), but **nothing downstream reads it** (grep).
- Seasonal behaviours such as migration, dispersal and breeding-season central-place foraging are therefore
  invisible to the model by design.
- **Multi-scale fix**:
  - **Level 1 (s):** keep the 10 s encoder (h).
  - **Level 2 (bout, 1–15 min):**
    - a sequence encoder (small transformer or GRU) over consecutive h within a continuous segment, plus the
      1 Hz run features from `gps_runs`;
    - trained with a temporal-contrastive objective (TS2Vec- or TNC-style; literature.md already lists both),
      using sub-sequence crops of the same bout as positives;
    - outputs a bout embedding used to define thermal climb, inter-thermal glide, sustained flapping and perch
      bouts;
    - mask only same-bird pairs that are close in time (they are likely the same behaviour, i.e. false
      negatives), and keep same-bird negatives from other bouts in every batch, as bpb4 does. Those negatives
      stop the objective from being solved by recognising the bird; r5_bpb4, with more same-bird negatives,
      gave lower identity.
  - **Level 3 (day):**
    - per bird-day metrics from *all* good fixes: daily distance, maximum displacement from the bird's own
      seasonal centroid (computed internally, never exported), hours aloft, altitude range;
    - plus level-1/2 state budgets from the scheduled sample (C1);
    - an HMM over daily steps (resident / exploratory / migratory) gives the seasonal and dispersal axis that the
      owner's goal needs.
- **Effort**:
  - Level 3: 2–3 days, CPU only. Do it first, because it answers the seasonal questions most directly.
  - Level 2: 3–5 days plus GPU runs.

### I6. "Unusual individuals" is not addressed; window outliers are within-bird only (important)

- `analysis.knn_outliers` scores the kNN distance in standardised h within bird × month × regime, relative to the
  group median.
  - It finds windows unusual *for that bird in that month*. Nothing measures whether a *bird* is unusual relative
    to the population.
  - It samples at most 300 windows per group, so rare events in large groups are mostly not scored.
- Identity reduction (C2) further removes the between-bird differences that "unusual individuals" would need.
- **Fix**:
  1. Bird-level anomaly: per bird-month distributions (state budget from the scheduled sample, level-3 daily
     metrics). Compute a distance to the population (e.g. MMD or energy distance on h, Mahalanobis distance on
     budgets) with a leave-one-bird-out reference, matched on month and mode, and a permutation null.
  2. Window-level: score all windows (approximate kNN, e.g. FAISS) rather than a 300-window sample, and report
     outliers by state. Many top outliers will be sensor faults; flag those by `acc_samples`, clipping and
     `sensitivity_k`.
- **Effort**: 1–2 days.

### I7. Seasonal curves are composition-confounded (important)

- `analysis.bird_share` averages per-bird shares over the birds present in each 7-day bin. Birds enter and leave
  the record (deployment, death, tag failure, cohort: adults only seen Oct–Mar). A change across bins can
  therefore be a change of *which* birds are present.
- The coverage panel shows the problem but does not remove it.
- **Fix**: within-bird deviations (each bird's share minus its own mean) or a mixed model with a bird random
  effect and a cyclic month term, fitted on the scheduled sample (C1). Report per-month n birds and intervals.
- **Effort**: 1 day.

### I8. Recording regime can enter the encoder through BatchNorm over padded samples (minor to important, verify)

- `model.AccEncoder` convolves zero-filled absent samples (with a mask channel) through BatchNorm (`model.py:17–28`,
  l.40–47). Batch statistics then depend on the share of absent samples, and conv receptive fields at run edges
  see the zero padding.
- This is a possible route, not a demonstrated one. The mask is itself a model input, so the 0.997–0.999
  short/full probe score says nothing about BatchNorm; `evaluate.py` says as much in its docstring.
- **Check**: the existing `burst_same_state` invariance test, compared between the current encoder and one with
  masked normalisation.
- **Fix if the check shows an effect**: masked normalisation (GroupNorm or LayerNorm with masked statistics), or
  encode only present runs (pad per run).
- **Effort**: 0.5 day plus one run.

### I9. The presence rule is weak (important)

- A state counts as present in a bird with ≥ 20 windows (`states.PRESENCE_MIN_WINDOWS`, l.91), i.e. 200 s of data
  out of about 60,000 windows per bird on average.
- With 5.5 M windows, almost any cluster passes. The rule does not protect against states driven by one or two
  birds.
- The artifact check `max_bird_share > 0.5` (l.94) is also lenient.
- **Fix**: require the state to make up ≥ x% (e.g. 0.5%) of each of ≥ 5 birds' scheduled-sample windows, and cap
  the top bird share at about 0.2.
- **Effort**: 1 hour.

### I10. Silent data-format assumptions (important until verified)

- `windows.py` hard-codes `FS = 20.0` (l.31) and decodes every burst as 3 interleaved axes
  (`acc.decode(n_axes=3)`, `windows.py:92`).
- Neither `eobs:acceleration-axes` nor `eobs:acceleration-sampling-frequency-per-axis` is checked anywhere:
  - the axes column is ingested and never read;
  - the frequency is read only in `profile.py` for burst features.
- A tag configured at a different rate, or with fewer axes, would be placed on the wrong time grid or decoded as
  scrambled XYZ, with no error.
- **Fix**: assert a single (axes, rate) per (tag, burst length) in `profile.py` and refuse other values in
  `windows.py`. One `value_counts` on the data verifies the current exports.
- **Effort**: 1 hour.

### I11. Privacy of window-level outputs (important, verify)

- The catalogue (`analysis.CATALOGUE_COLS`) carries exact `t_start` (UTC) with `solar_hour_bin` (1 h) and
  `sun_elev_bin` (5°). Individually these are coarse.
- Combined with exact times across many windows of one bird, the moments at which `solar_hour_bin` changes give
  the bird's longitude to minutes of time, i.e. about 0.25° per minute. With the `sun_elev_bin` boundary crossings
  they give latitude, so roosts and nests could be localised to a few kilometres. This is an inference; verify
  with a small reconstruction attempt on one bird.
- Also, `meta.parquet` keeps exact `local_solar_hour` and `sun_elev_deg`. That is fine on the cluster; it must not
  be published.
- **Fix**: in published tables, round `t_start` to the day (or the hour, with solar bins removed), or drop the
  solar bins when exact time is present. Publish lat/lon-derived metrics only as relative displacements. Agree the
  resolution with the data owner and the Movebank study's permissions (Q7).
- **Effort**: 2 hours.

### I12. Baseline acceptance test not run on current data (important)

- DESIGN §2.6: "the SimCLR space must beat it to be kept". The only baseline result (RESULTS_stage3) is on
  profile_v2, which is superseded.
- No v4 comparison of h against the handcrafted features exists for state stability, held-out ARI, or the label
  confusion matrix.
- See the fix under C4 (item 4).
- **Effort**: one states.py run on features.

## 4. Minor issues

1. **`profile.py:246`**: `float(b.at[i, "fs_hz"] or 20.0)`. NaN is truthy, so the `or 20.0` fallback can never
   take effect. A NaN rate would reach `acc.burst_features`, where `int(round(static_window_s * fs))` raises
   `ValueError: cannot convert float NaN to integer` (checked in Python 3) and stops stage 2. Stage 2 finished, so
   this export had no NaN rates. The fallback is dead code, not a silent error.
2. **`evaluate.py:95–98, 109`**: `astype(str)` turns a NaN nuisance value (e.g. missing `decoding`) into the class
   "nan", which the probe can then predict.
3. **`train.py:322–330`**: resume restores model, optimiser and sampler RNG, but not the torch or CUDA RNG used by
   `augment`. A resumed run is not identical to an uninterrupted one.
4. **Provenance.** `config.json` records arguments only, not the git commit, the input checksums (windows
   directory, `format.json`) or package versions. `pyproject.toml` leaves all dependencies unpinned (only
   `torch>=2.4`). Record `git rev-parse HEAD`, `pip freeze` and a hash of `windows_summary.csv` in each run
   directory.
5. **Bird counts differ across documents**: 62 (DESIGN §5, stage 3), 94 (r1–r3 hour level), 92 (v4). State which
   export is canonical and record its checksum and row counts in the results.
6. **Cohort is defined from the first window's year**, not from the deployment date (`dataset.py:159–160` and the
   copies in `states.py` and `baseline.py`). A bird deployed in December 2024 whose first kept window is in
   January 2025 becomes "2025". Use the reference deployment date.
7. **`windows.py:166`** relies on NumPy "last write wins" for repeated fancy indices. That is true in practice for
   1-D assignment but is not documented as guaranteed. `np.minimum.at` would be explicit.
8. **`baseline.transitions`** (`baseline.py:60–64`) treats consecutive *usable* rows as adjacent even when a
   dropped window lies between them.
9. **`gps.good_fixes`** lets fixes without an accuracy estimate pass (`fillna(0)`, l.34). Count how many there
   are.
10. **`cluster_stability` in evaluate.py** uses 5 bootstraps. That is too few for the differences discussed in
    RESULTS r1–r3 ("within the noise of 5 bootstraps", as that document itself says).

## 5. Do the results documents support their claims?

| Claim (document) | Supported? | Why |
|---|---|---|
| Sensitivity probe reads recording mode, not amplitude (r4–r6 §1) | **Yes** | Counterfactual with 0% flips, matched control |
| "h still reads the setting at 0.93 without GPS" is unexplained (r4–r6 §1) | Honestly reported | Likely cause: C1 (continuous mode = flight, so the behaviour mix differs by setting); I8 is a possible further route (unverified) |
| Identity reduced "without losing behaviour" (r4–r6 §3, §5) | **No** | The probes are invariant to the manipulation (C2); one seed; selection on the test birds (I2) |
| Adversary did not work (r4–r6 §3) | Yes | Clear train/held-out contrast |
| Same six states in every run (r4–r6 §4) | **No** | K is at the search bound (C4); no cross-run matching |
| No sex difference in state shares (r4–r6 §4) | **Only as "no large difference in 6 coarse shares"** | No effect size or power; confounded (C3); budget is window-weighted (C1) |
| State shares (26.6% rest, …) describe behaviour | **No** | Window counts, not time (C1) |
| r1 best of r1–r3 (r1-r3) | Superseded; differences within noise | As stated in the document |

## 6. What is missing for a publishable characterisation

1. **Independent ground truth.** Any of the following would do:
   - video or field observations synchronised to tags (nest cameras, observer logs);
   - known events such as nest attendance during incubation, derivable internally from the nest location
     without publishing it;
   - an expert blind annotation of about 500–1,000 stratified windows (ACC trace plus GPS track snippet), with
     inter-rater agreement;
   - published labelled accelerometer data from golden eagles or other large soaring raptors, if they can be
     obtained.

   Without one of these, states are clusters with descriptive names.
2. **Uncertainty everywhere.** Bird-bootstrap intervals on every share, effect and probe; seeds; and
   state-assignment uncertainty (`p_state` exists and is not propagated).
3. **Proper statistical models for demographic and seasonal effects.** Mixed models with bird random effects,
   cohort, mode, month and age covariates (C3, I7). Pre-register the comparisons in a short analysis plan before
   unblinding the frozen test birds (I2).
4. **Biomechanical positive controls.** Wingbeat frequency vs sex, glide polar (sink vs speed) vs sex and age.
   These show the pipeline can detect known differences.
5. **Reproducibility package.** Pinned environment, git hash per run, config and data checksums (§4.4); a
   one-command rerun from the windows to the figures; a released derived dataset without positions (I11).
6. **Data permissions and privacy.** Movebank study terms; nest-location sensitivity (persecution, wind-energy
   siting); an agreed publication resolution (Q7).
7. **A baseline comparison on current data** (I12). Reviewers will ask why deep learning is needed if handcrafted
   features plus a GMM give the same states.

## 7. Next 10 actions (prioritised)

| # | Action | Addresses | Effort |
|---|---|---|---|
| 1 | Ask the owner for the tag schedules and triggers (Q1), the curated age, sex and status table (Q2–Q4), and any ground truth (Q5) | C1, C3, §6.1 | Owner time |
| 2 | Add a `scheduled` flag (regular bursts) to window meta. Recompute budgets, seasonal curves and the sex test on the scheduled sample, mode-stratified, with bird-bootstrap CIs and effect sizes. Correct for ACC-only misassignment with the GPS-removed confusion matrix. First check how GPS-dependent labels split by mode | C1, C3, I7 | 1–1.5 days + CPU |
| 3 | Rerun stage 5a with `--k-min 2` on r4 and r6, and on the handcrafted features through the same procedure. Add cross-run Hungarian matching and ARI | C4, I12 | 0.5 day + ~10 CPU-h |
| 4 | Add rotation-sensitive probes, a random-encoder baseline, a raw-input baseline, and a bird-level sex probe. Run them on r4, r5_bpb4 and r6 | C2, I1 | 1 day + 2 CPU-h |
| 5 | Freeze a second untouched test set of birds, stratified by sex, age and cohort. Adopt 3 seeds for compared runs | I2 | 1 hour (then GPU) |
| 6 | Positive control: wingbeat frequency in flapping windows and glide speed/sink vs sex with a mixed model on handcrafted features | C3, §6.4 | 0.5 day |
| 7 | Daily-level metrics from all good fixes, plus a daily-step HMM for resident, exploratory and migratory phases (level 3) | I5, seasonal goal | 2–3 days CPU |
| 8 | Deterministic per-bird yaw de-rotation from the flapping axis; retrain with ±15° yaw and masked normalisation; 3 seeds | C2, I8 | 1.5 days + ~5 GPU-h |
| 9 | Expert blind annotation protocol for about 500 stratified windows, and label-threshold recalibration (incl. climb noise check) | I1, §6.1 | 2 days setup + annotator time |
| 10 | Provenance (git hash, pip freeze, checksums), fs/axes assertions, privacy rounding of published tables | §4, I10, I11 | 0.5 day |

The level-2 bout encoder (I5) and a proper HMM/HSMM (I4) come after these. They are worth doing only once the
scheduled-sample budgets and the demographic signal check (actions 2, 4 and 6) show what the representation has to
carry.

## 8. Questions only the owner can answer

1. **Q1, schedules and triggers.** Which recording schedules were programmed per tag and period? Is continuous ACC
   with 1 Hz GPS triggered (by ACC activity, speed, altitude, geofence or battery voltage) or time-scheduled? Did
   schedules change by season or by remote reprogramming?
2. **Q2, age.** For each bird, what was the age class or hatch year at tagging (nestling, fledgling, juvenile,
   subadult, adult by plumage), and how was it determined? Are the 21 "migrant" comments trapped migrants of known
   age class?
3. **Q3, sex.** How was sex determined (genetic, morphometric, behavioural)? Are any of the 3 missing birds
   resolvable?
4. **Q4, status and track ends.** What was the breeding or territorial status per bird-season? Why did each track
   end (death, tag failure, harness loss, still running)? Are deaths with known cause available?
5. **Q5, ground truth.** Are there nest-camera records, field observation logs, prey-delivery records, or any
   behaviour annotations synchronised with tag time?
6. **Q6, hardware.** What harness type and attachment (backpack or leg-loop) was used? Were any tags reused across
   birds? Were tag models or firmware changed between cohorts?
7. **Q7, publication limits.** What spatial and temporal resolution may be published for these birds (Movebank
   study terms, agency permits, nest-site sensitivity)?
8. **Q8, individual differences.** Are consistent individual differences in behaviour part of the scientific
   question, or a nuisance to remove? This decides whether identity reduction (r5/r6) is wanted at all.
9. **Q9, priority.** Which of the goals (sex, age, unusual birds, common behaviours, seasonal behaviours) matters
   most for the first paper? The design choices in C1, C2 and I5 differ by goal.
