# Literature review for the redesign of the golden eagle SimCLR pipeline

Written 2026-11-24. Method research only. No data were read and no remote machine was contacted.
Web searches used generic topics only (no bird IDs, coordinates or project names).

## Evidence labels

Every factual claim carries one of these labels:

- **[FT]**: full text, or primary source code, read during this review.
- **[AB]**: abstract, search-engine snippet, repository record or secondary description only. The number is probably right, but the methods section was not read.
- **[RC]**: recalled from prior knowledge and not checked during this review. Verify before citing.
- **[OURS]**: a recommendation derived in this review from the evidence above. It is not a published result.

---

## 1. e-obs acceleration data

### 1.1 Raw count format and conversion to g

**Primary source.** The `TransformRawACC` function of the R package moveACC by Anne Scharf (Movebank / MPI) was read directly from
https://gitlab.com/anneks/moveACC/-/raw/master/R/TransformRawACC.R [FT]. It states the e-obs defaults, documented "as of August 2018":

- The formula is `g = (raw - 2048) * slope`. For m/s², multiply by 9.81. The 2048 offset is the midpoint of a 12-bit range.
- The default slope depends on the tag generation, which is identified by the tag ID (tag serial number):

| Tag generation | Tag ID | Default slope (g per count) | Note |
|---|---|---|---|
| Gen 1 | ≤ 2241 | 0.0027 ("low" sensitivity) or 0.001 ("high") | sensitivity set at deployment |
| Gen 2 | 2242–4117 | 0.0022 | sensitivity fixed to low |
| Gen 3 | ≥ 4118 | 1/512 = 0.001953 | sensitivity fixed to low |

- "The orientation of the Y-axis in generation 1 is different to generation 2 and 3." The direction is not stated; the e-obs manual is the reference.
- The package also accepts per-tag, per-axis calibration in the form `nx, ny, nz` (intercepts) and `cx, cy, cz` (slopes). Its example values are intercepts of 2039–2050 and slopes of 0.00194–0.00196.

**Action [OURS]**
- Map each tag ID in the Movebank export (the `tag-local-identifier` column) to its generation before converting. This mapping has not been checked against this dataset. Check every tag, because the generation decides both the slope and the Y-axis direction.
- A tag with a 1/512 slope reads about 512 counts per g. A perched bird should then show a static magnitude of about 512 counts around 2048.

### 1.2 Per-tag calibration from static periods

**Published guidance**
- Garde et al. 2022, *Methods Ecol Evol* 13:813–825, https://doi.org/10.1111/2041-210X.13804 [AB]:
  - Bench tests show that individual axes need a two-level (offset and gain) correction. Without it, DBA was off by up to 5%.
  - Tag position changed DBA by about 9% in pigeons and 13% in kittiwakes.
  - The authors propose a field calibration before deployment.
- GGIR auto-calibration (human wrist accelerometers), https://wadpac.github.io/GGIR/articles/chapter3_QualityAssessment.html [AB]:
  - Offset and gain are fitted from non-movement periods so that the static vector magnitude equals 1 g.
  - The method needs enough variety of orientation across those periods.

**Recommendation [OURS]**
1. Identify static bursts. Night bursts, with the sun below the horizon, are the best source.
   - Static means a per-axis SD below about 2–3 counts, or below about 0.005 g after the default conversion.
2. Fit the offset per tag and per axis. The constraint is that the static magnitude equals 1 g.
3. Fit the gain only if the static bursts cover enough orientations, for example by checking that the eigenvalues of the static-vector covariance are not degenerate.
   - A perched eagle mostly sits upright, so a gain fit will often be ill-conditioned.
   - In that case, keep the default slope and fit only the offsets.
4. Report the residual |‖a‖−1 g| per tag as a quality flag. Use it as a QC table and never as a model input.

### 1.3 Axis orientation and attachment

- **Axis convention.** In e-obs tags on griffon vultures, x is medial–lateral (sway), y is anterior–posterior (surge) and z is dorso-ventral (heave). Source: Nathan et al. 2012, *J Exp Biol* 215:986–996, https://doi.org/10.1242/jeb.058602 [AB].
- **Golden eagle backpack study.** Sur et al. 2017 used the same names (x sway, y surge, z heave) on a CTT backpack tag [FT]. In that study the sway axis was "the most responsive to changes in eagle flight behavior".
- **No standard orientation.** The Movebank Acceleration Viewer manual says axis orientation "depends on how the tag was attached to the animal", https://www.movebank.org/node/26108 [FT].
- **Leg-loop harnesses on golden eagles.** Nourani et al. 2024 used e-obs tags on leg-loop harnesses, *eLife* 13:RP98818, https://doi.org/10.7554/eLife.98818 (full text via PMC11390109) [FT].
  - [OURS, inference] A leg-loop tag sits on the lower back or rump and can rotate more than a backpack.
  - Its pitch reference therefore differs between birds.

**Recommendation [OURS]**
- Normalise orientation per bird (and per deployment).
  - Take the median static gravity vector during night roosting as the bird's "upright" reference.
  - Rotate all bursts so that this vector maps to +z.
  - Compute pitch and roll relative to it.
- Reason: mounting differences between birds (harness type, tag tilt) are a plausible cause of the bird-identity clustering found in the current latent space.
- Keep the harness type as metadata for checks. It must never be a model input.

### 1.4 Derived metrics

**Definitions**
- **Static/dynamic split.** Static = running mean (low-pass) per axis. Dynamic = raw − static.
- **ODBA** = Σ|dynamic| over the three axes. ODBA is from Wilson et al. 2006, *J Anim Ecol* 75:1081, https://doi.org/10.1111/j.1365-2656.2006.01127.x [RC].
- **VeDBA** = ‖dynamic‖. VeDBA is from Qasem et al. 2012, *PLoS ONE* 7:e31187, https://doi.org/10.1371/journal.pone.0031187 [RC].
- **Pitch and roll**, from the static components:
  - pitch = atan2(surge, √(sway² + heave²))
  - roll = atan2(sway, heave)
  - Signs depend on the axis convention above.
- **Movebank burst ODBA.** The Movebank viewer computes ODBA per burst from deviations about the per-axis burst mean [FT].

**Running-mean window length**
- Shepard et al. 2008, "Derivation of body motion via appropriate smoothing of acceleration data", *Aquat Biol* 4:235–241, https://doi.org/10.3354/ab00104 [AB]:
  - ODBA varies with the smoothing window.
  - Use a running mean of at least 3 s for animals whose stroke period is up to 3 s.
  - Otherwise, use at least one full stroke cycle.
- Sur et al. 2017 used a 1 s running mean on 140 Hz golden eagle data [FT].
- Williams et al. 2020 used mean VeDBA per 10 s window on condors [FT].

**How this collides with the data [OURS]**
- At 20 Hz, the three burst lengths are 24, 40 and 264 samples per axis, which is 1.2 s, 2.0 s and 13.2 s.
- A 3 s running mean cannot be applied inside the 1.2 s and 2 s bursts.
- **Encoder inputs:** compute every ACC feature on the common 24-sample crop by one method, with the per-crop mean as the static component (as Movebank does per burst). Burst length then cannot leak through the features.
  - The burst then holds only about 3 (1.2 s) or 5 (2.0 s) wingbeat cycles. A per-burst mean is acceptable for flapping, where the average over whole cycles cancels.
  - It is biased when the bird changes posture within the burst, for example a stoop pull-out or a landing.
- **Characterisation and evaluation only:** on 264-sample bursts, also compute a centred 3 s running mean (61 samples). Never feed it to the encoder.
- Document the per-crop mean as a forced deviation from Shepard et al. 2008.

### 1.5 Wingbeat frequency

**Published values**

| Species | Wingbeat frequency | Source |
|---|---|---|
| Golden eagle | 2.2–2.8 Hz across six eagles (40 Hz ACC); flapping 2–38% of flight time | Gupta, Bewley et al., arXiv 2412.00231, https://arxiv.org/abs/2412.00231 [AB] |
| Golden eagle (one adult female) | about 2.8 Hz, the upper edge of a power-law region in the acceleration spectrum | Laurent et al. 2021, arXiv 2102.12374, https://arxiv.org/abs/2102.12374 (published in *PNAS*, "Turbulence explains the accelerations of an eagle in natural flight") [AB] |
| Andean condor | about 2.7 Hz; flapping about 1% of flight time, more than 70% of it at take-off | Williams et al. 2020, *PNAS* 117:17884, https://doi.org/10.1073/pnas.1907360117 [FT] |

**Frequency resolution of the bursts [OURS]**

| Burst | Duration | FFT bin width (1/T) |
|---|---|---|
| 24 samples | 1.2 s | 0.83 Hz |
| 40 samples | 2.0 s | 0.50 Hz |
| 264 samples | 13.2 s | 0.076 Hz |

The Nyquist frequency is 10 Hz for all three.

- A dominant-frequency (wingbeat) estimate is meaningful only on the 264-sample bursts. Use it for characterisation only: as an encoder token, its presence alone would reveal burst length.
- For short bursts, use band power in about 1.5–4 Hz divided by total dynamic power, plus the heave-axis autocorrelation at lag 7–9 samples (0.35–0.45 s).
- Zero-padding does not add real resolution.
- **Flapping rule [OURS].** Flag a burst as flapping when VeDBA is high and more than about 40–50% of the dynamic heave power falls in 1.5–4 Hz. Calibrate the threshold on the bimodal VeDBA histogram of in-flight bursts.

---

## 2. Classifying flight modes of large soaring raptors

### 2.1 What acceleration data can and cannot separate

- **Sur et al. 2017**, *PLoS ONE* 12:e0174785, https://doi.org/10.1371/journal.pone.0174785 [FT]:
  - Data: one trained golden eagle, CTT backpack, about 140 Hz ACC, labelled from video.
  - Features: 28 in total. Per axis: mean, min, max, SD, skewness, kurtosis, trend and dominant power spectrum. Plus the three inter-axis correlations and ODBA. The static component came from a 1 s running mean.
  - Simple ethogram (flapping / sitting / soaring): RF 92.3%, KNN 86.6%.
  - Complex ethogram (banking vs straight split at about 20° tilt): KNN 91.2%, RF 61.6%.
  - RF kept its accuracy on the simple ethogram down to **10 Hz**, while KNN needed at least 20 Hz.
  - "Soaring" in this paper lumps gliding, thermal circling, wing tucks and single wingbeats.
- **Williams et al. 2015**, "Can accelerometry be used to distinguish between flight types in soaring birds?", *Anim Biotelemetry* 3:45, https://doi.org/10.1186/s40317-015-0077-0 [AB]:
  - KNN on ACC alone had low accuracy for passive flight types (thermal soaring, slope soaring, gliding).
  - The labels came from altitude change plus magnetometer heading.
- **Williams et al. 2020 (condors)** [FT]:
  - Flight types were classified from rate of altitude change (barometric pressure), heading change (magnetometer) and absence of flapping (ACC).
  - Thermal soaring = circling while climbing. Slope soaring = climbing while tacking along a slope. Gliding = descending.
- **Duriez et al. 2014**, *PLoS ONE* 9:e84887, https://doi.org/10.1371/journal.pone.0084887 [FT]:
  - ODBA "detected all flapping bouts".
  - Birds were classed as soaring "as soon as birds started to fly in circles and have a positive vertical speed", and as gliding when "the flight path was rectilinear and altitude was decreasing".
- **Halsey et al. 2009**, *J Field Ornithol* 80:171, "Recording raptor behavior on the wing via accelerometry" [AB, seen in a reference list]: griffon vultures rarely flap except at take-off and landing.

**Conclusion [OURS]**
- Acceleration separates three things reliably: perched vs flying, flapping vs passive flight, and posture (pitch and roll).
- The **kind** of passive flight (thermal circling, slope/orographic soaring, linear gliding) needs 1 Hz GPS, or magnetometer heading, or both.
- The encoder must therefore see both modalities when both exist.
- Batch-level ACC features alone will not separate soaring from gliding.

### 2.2 Thermal circling and gliding from 1 Hz GPS

**Published building blocks**
- **Flying vs stationary.** Instantaneous speed > 4 m/s marks flight (Nathan et al. 2012, as used by Khosravifard et al. 2018) [FT].
  - Khosravifard et al. 2018, *Int J Environ Res*, https://doi.org/10.1007/s41742-018-0093-z [FT].
  - That study classified griffon flight types from high-resolution GPS. Precision: gliding 88%, linear soaring 83%, thermal soaring 75%, slope soaring 53%, spiral gliding 34%. Overall accuracy 70%.
  - Linear soaring required straight flight of at least 350 m.
- **Full circle within 30 s.** Sherub et al. 2016, *Biol Lett* 12:20160432, https://doi.org/10.1098/rsbl.2016.0432 [FT].
  - e-obs tags recorded 1 Hz bursts of 10 min.
  - Airspeed was estimated over 30 s intervals, and only sequences in which the bird made a full circle within 30 s were used.
  - Himalayan vultures increased circle radius by about 12.5% per 1000 m of altitude.
- **Constant-turn loops for wind estimation.** Weinzierl et al. 2016, *Ecol Evol* 6:8706, https://doi.org/10.1002/ece3.2585 [AB].
  - Wind is estimated from short circling segments of 18 s (19 GPS fixes) with a constant turn angle along a closed loop.
  - The method is implemented in the R package `move`.
- **Circling plus climbing = soaring.** Duriez et al. 2014 [FT], as quoted in 2.1.

**Rule constructed here [OURS]: no source gave a single published 1 Hz threshold set**
1. **Quality filter.** Drop fixes with high DOP or poor horizontal accuracy, using the dataset's own distribution, for example HDOP > 5 or `eobs:horizontal-accuracy-estimate` > 10–20 m.
   - Smooth height-above-ellipsoid with a 5–10 s centred filter before differencing. Single-fix vertical noise is several metres.
2. **Flight.** Ground speed > 4 m/s.
3. **Circling.** Sum the unwrapped heading change over a sliding window of 20–40 s; |Σ Δheading| ≥ 360°.
   - Use the GPS `heading` field, or the bearing between successive fixes.
   - Also record turn rate (deg/s), circle radius (≈ v/ω) and turn direction.
4. **Thermal soaring** = circling and smoothed vertical speed > 0 (or > +0.2 m/s to suppress noise).
   - **Circling glide or descent** = circling with vertical speed < 0.
5. **Linear gliding** = not circling and vertical speed < 0.
6. **Linear (straight) soaring or orographic lift** = not circling, vertical speed ≥ 0, and no flapping in the acceleration data.
   - Separate orographic lift from linear thermal streets with terrain: wind-facing slope and a terrain-ruggedness index or distance to ridge line (see 2.3).
7. **Flapping** comes from the acceleration data (1.5), not from GPS.

**How to use these rules [OURS]**
- Use them as **evaluation probes and interpretation labels**. Do not use them as training targets: the current problem is that the latent space was organised around hand rules on hourly summaries.

### 2.3 Orographic vs thermal soaring (golden eagles)

- **Katzner et al. 2015**, *J R Soc Interface* 12:20150530, https://doi.org/10.1098/rsif.2015.0530 [AB]:
  - Data: more than 22,000 GPS points at 30 s intervals from 32 eagles on spring migration.
  - Subsidised flight time: thermal soaring about 42%, gliding between thermals about 45%, orographic soaring about 13%.
  - Orographic soaring was more common in the morning and evening, earlier in the season, and with stronger crosswinds and tailwinds.
  - The exact classification algorithm was not read: the full text was blocked.
- **Lanzone et al. 2012**, *Biol Lett* 8:710, https://doi.org/10.1098/rsbl.2012.0359 [AB]: with stronger wind, golden eagles switch from thermal to orographic soaring.
- **Duerr et al. 2012**, *PLoS ONE* 7:e35548, https://doi.org/10.1371/journal.pone.0035548 [RC]: thermal soaring is more efficient than orographic soaring for directed migration.
- **Bohrer et al. 2012**, *Ecol Lett* 15:96–103 [AB, seen in a reference list]: contrasting migration strategies of golden eagles and turkey vultures in terms of updraft velocity components.
- **State-space model (Pirotta, Katzner et al. 2018)**, https://pubs.usgs.gov/publication/70202319 [AB]:
  - A Bayesian state-space model on 3-D GPS plus topography found five states: perching, gliding, orographic soaring and two variants of thermal soaring.
  - This is a useful precedent for an HMM-type layer on top of the embeddings.
- **Recommendation [OURS].** Annotate fixes post hoc with wind (ERA5 via Movebank Env-DATA), slope aspect and terrain ruggedness. Then test whether the "linear soaring" clusters line up with wind-facing slopes. These are characterisation covariates, not encoder inputs.

### 2.4 Stoops, dives and hunting: a gap

- No usable published thresholds were found for golden eagle stoops or hunting from e-obs data during this review.
- **Candidate signature [OURS, unverified]:**
  - Steep negative vertical speed, roughly < −5 to −10 m/s, sustained over several 1 Hz fixes.
  - Ground speed that rises while vertical speed falls.
  - Then a high-g pull-out in the acceleration data (|a| well above 1 g) followed by a ground contact (perched).
- Hunting and foraging are more likely to show up as **sequence patterns** than as single windows: low, slow quartering flight, a stoop, then a long perch (possible prey handling). This argues for the sequence model in Section 4.

---

## 3. Self-supervised and contrastive learning for accelerometry and biologging

### 3.1 Benchmarks and foundation models

- **BEBE**: Hoffman et al. 2024, *Movement Ecology* 12, https://doi.org/10.1186/s40462-024-00511-8 (code at https://github.com/earthspecies/BEBE) [FT].
  - Nine datasets, 1654 h, 149 individuals.
  - **Splits are by individual.** Five folds; the test set "comprises data from a set of individuals distinct to those in the train set". Metrics are macro-F1 per test individual, then averaged.
  - Deep networks (CNN, CRNN) beat classical feature-based models on all nine datasets.
  - `harnet` (Yuan et al.), frozen, with a small head, was best on five of nine datasets with the full training data, and best on all nine when training data were reduced by about four times.
  - The SSL hypothesis was only "partly" confirmed.
  - The low-pass/high-pass cutoff was a tuned hyperparameter, chosen from {0, 0.1, 0.4, 1.6, 6.4} Hz.
- **Yuan et al. 2024**, *npj Digit Med* 7:91, https://doi.org/10.1038/s41746-024-01062-3 (arXiv 2206.02909) [FT, read via ar5iv]:
  - Pretext tasks: multi-task arrow of time, permutation and time warping. "There was no clear best performing configuration", so all three were used.
  - Input: 10 s windows at 30 Hz.
  - **Windows were sampled in proportion to their standard deviation.** Without this, the arrow-of-time task "stayed at the random chance level" and permutation accuracy fell by about 10 percentage points.
  - Random axis swaps and rotations were added as augmentations, to cover device-orientation differences.
  - No comparison with SimCLR was made.
- **Mao et al. 2025**, *Sci Rep*, https://doi.org/10.1038/s41598-025-27736-w [AB]:
  - Cross-species contrastive pre-training of a PatchTST encoder on animal accelerometry, using a time–frequency consistency objective, then fine-tuning on one species.

**Implications [OURS]**
- Yuan's std-weighted sampling directly addresses the 36% of windows that contain no measurement.
  - Remove no-data windows completely.
  - Down-weight near-static windows in the loss sampler; do not drop them.
- Pretrained `harnet` weights (30 Hz, 10 s windows, wrist) are a possible baseline. They need 300 samples after resampling, so only the 264-sample (13.2 s) bursts, from one sampling regime, can feed them. That baseline would be cohort-restricted and cannot use the 24-sample crop.
- Report BEBE-style per-individual held-out scores.

### 3.2 Contrastive time-series methods

| Method | Key idea | Reference |
|---|---|---|
| SimCLR | Two augmented views, NT-Xent, in-batch negatives | Chen et al. 2020, arXiv 2002.05709 [RC] |
| CPC | Predict future latents with an InfoNCE loss | van den Oord et al. 2018, arXiv 1807.03748 [RC] |
| TS-TCC | Weak and strong augmentations; temporal (cross-view future prediction) and contextual contrast | Eldele et al. 2021, IJCAI, arXiv 2106.14112 [AB] |
| TS2Vec | Hierarchical contrast at timestamp and instance level over random crops; no hand-tuned augmentations | Yue et al. 2022, AAAI, arXiv 2106.10466 [AB] |
| TNC | Temporal neighbourhood as positives, with a PU-learning correction for distant samples | Tonekaboni et al. 2021, ICLR, arXiv 2106.00750 [RC] |
| NNCLR | Nearest-neighbour positives from a support set | cited in Qian et al. 2022 [FT] |

### 3.3 Which augmentations help or hurt for accelerometry

- **Qian, Tian & Miao 2022**, KDD, arXiv 2202.05998 (library CL-HAR) [FT, read via ar5iv]:
  - "No unique augmentation transformation that consistently performs better than others."
  - For SimCLR on UCIHAR: noise and jitter+scaling were best (93.6 and 93.5); **rotation was the worst** (54.1); shuffling also hurt (74.9).
  - CNN backbones beat LSTM and transformer backbones on these small HAR datasets ("convolutional neural networks are more robust").
  - **For SimCLR, increasing negatives from 16 to 128 helped, and more than 128 hurt.** The authors attribute this to same-class samples in the batch being treated as negatives (false negatives).
  - Semantically consistent positives, such as NNCLR's nearest-neighbour positives, gave the top UCIHAR result.
- **Tang et al. 2020**, arXiv 2011.11542 [AB]:
  - SimCLR for HAR improved with fine-tuning and **random rotation**.
  - Augmentation order and type mattered strongly; 64 pairs were tested.
- **Um et al. 2017**, arXiv 1706.00527 [RC]: rotation, permutation and time-warping helped supervised CNNs on wearable Parkinson's data.
- **Cheng et al. 2020** (biosignals), arXiv 2007.04871 [FT]: for EEG, temporal cutout was most effective (+10.9%), followed by temporal delay (+6.1%) and signal mixing (+5.3%).

**Resolution for this domain [OURS]**

Keep or add these augmentations:
- **Jitter**: Gaussian noise at about 1–3% of the signal SD.
- **Amplitude scaling**: 0.9–1.1. This stands in for calibration error, in line with the up-to-5% DBA error in Garde et al.
- **Small rotation**: ±10–15° about a random axis. This stands in for harness shift.
- **Temporal crop and offset**: random sub-windows within a segment.
- **Cutout or masking** of short spans.
- **Modality dropout**: drop the ACC or GPS branch entirely in one view.

Avoid these augmentations:
- **Full SO(3) rotation and axis swap.** They destroy pitch and roll, which separate perching from flight and banking from straight flight. Yuan's axis swap was meant to cover wrist devices from different manufacturers, a nuisance that does not exist here after per-bird orientation normalisation.
- **Permutation or shuffle of segments.** This breaks wingbeat periodicity, which is the flapping cue, and it hurt in Qian et al.
- **Strong time-warp.** Warping 20 Hz data by more than ±10% moves wingbeat frequency into another bird's or behaviour's range.
- **Time reversal as an invariance.** Climbing vs descending matters, so the direction of time carries meaning. Arrow of time is fine as a *pretext prediction task*, but not as a view augmentation.
- **Per-window z-scoring.** It removes absolute scale: a perched window and a flapping window become indistinguishable in variance. This is a likely contributor to the current failure.

### 3.4 Nuisance factors: cohort, sampling cadence, device and individual

**What the evidence says**
- **Cheng et al. 2020**, "Subject-aware contrastive learning for biosignals", arXiv 2007.04871 [FT]:
  - A **subject-specific contrastive loss draws negatives only from the same subject**, so the encoder cannot solve the task by recognising the subject.
  - An **adversarial subject classifier** is trained on frozen-encoder outputs. The encoder is then penalised, with weight λ, to confuse it. No gradient reversal is used.
  - The invariance penalty helped when subjects were few (< 64 for EEG). A weaker λ (0.01 or 0.001) was better for ECG, where some subject information is needed.
- **Domain-adversarial training** (gradient reversal): Ganin et al. 2016, *JMLR*, arXiv 1505.07818 [RC].
- **False negatives**:
  - Qian et al. 2022 [FT]: more than 128 negatives degraded SimCLR.
  - Debiased contrastive loss: Chuang et al. 2020, arXiv 2007.00224 [RC].

**Diagnosis of the current model [OURS]**
- In-batch negatives drawn from 62 birds and several sampling regimes are easiest to separate by cohort and identity cues, for example:
  - which channels are zero-filled;
  - GPS cadence;
  - burst length;
  - mounting angle.
- With batch 2048 and τ = 0.07, the loss rewards exactly those cues.
- Same-behaviour windows from different birds become false negatives at high rates. Perching dominates any raptor's day.
- This is a risk consistent with the observed latent structure, not a proven cause. The Qian result is from small HAR datasets.

**Recommendations [OURS]**
1. **Remove cadence and presence cues from the input.**
   - Crop every ACC burst to a common 24-sample (1.2 s) core for the encoder. Use longer bursts only to provide more crops.
   - Never pass presence flags or zero-filled channels.
   - Keep 1 Hz GPS segments and 5-min fixes in **separate tiers** (see 3.5), not one encoder.
2. **Batch composition.**
   - Build each batch from **a few birds**, with many windows per bird, so that negatives are mostly same-bird and same-cohort. This is Cheng's subject-specific negatives.
   - Alternatively, mask cross-bird negatives in NT-Xent.
3. **Batch size and temperature.**
   - Treat batch size and τ as hyperparameters. Test 256–512 with τ ∈ {0.1, 0.2, 0.5} as well as the current 2048 / 0.07.
   - Consider a debiased loss or nearest-neighbour positives (NNCLR) to reduce false negatives.
4. **Invariance penalties.**
   - Add an adversarial head for bird ID, tag cohort and sampling regime, using Cheng's frozen-classifier penalty or gradient reversal.
   - Start with a small λ (0.01–0.1). Age and sex are partly confounded with bird ID, and some individual style is biological.
5. **Diagnostic probes (the main success metric).**
   - Fit linear probes on frozen embeddings to predict bird ID, cohort, GPS cadence, burst length and harness type. These should sit **near chance**.
   - Fit probes for rule-derived flapping, circling, gliding, perched and night labels. These should be **high**.
   - Use held-out-bird folds, as in BEBE.

### 3.5 Irregular and multi-modal sampling

- **mTAN**: Shukla & Marlin 2021, ICLR, arXiv 2101.10318 [AB].
  - Continuous-time attention: learned time embeddings act as keys, and reference time points act as queries.
  - Handles sparse, irregular multivariate series without zero-filling.
  - Code: https://github.com/reml-lab/mTAN.
- **SeFT** (Horn et al. 2020, arXiv 1909.12064) and **Raindrop** (Zhang et al. 2022) [RC]: set-function and graph encoders that operate on observed (time, channel, value) tokens only.
- **Masked-token / patch transformers** (PatchTST: Nie et al. 2023, arXiv 2211.14730 [RC]):
  - Tokenise patches, and mask patches for reconstruction or as a contrastive view.
  - Missing data become absent tokens rather than zeros.

**Architecture recommendation [OURS]**
- **Unit of analysis.** The unit is the **observation event**, not the clock hour:
  - one ACC burst, plus the GPS fixes within ±30–60 s;
  - or one 1 Hz GPS segment of 30–120 s.
- **Tier A (fine scale, about 1 min).** Two encoders, one per modality:
  - an ACC burst encoder: 1-D CNN on 3 × 24 samples in g after orientation normalisation, plus static pitch/roll and VeDBA as scalar tokens;
  - a 1 Hz GPS segment encoder: CNN or transformer on speed, vertical speed, turn rate and cumulative heading change. Use relative quantities only, never absolute lon/lat or altitude.
  - The two are fused by a token-set transformer with time-delta encodings.
  - Absent modalities are absent tokens, trained with modality dropout.
  - Magnetometer and quaternion data (33 birds, until April 2025) are optional tokens. Because their presence marks a cohort, use them in a separate evaluation, or only as an auxiliary teacher (see recommendation 9). Never use them as a default input.
- **Tier B (bout and day scale).** A sequence model over Tier-A embeddings plus coarse 5-min fixes (step length, net displacement), with time encoded as solar elevation (Section 4.5).
  - This tier carries migration, roosting, foraging bouts and long-distance travel.
  - Hourly UTC windows misalign with local solar time. At about 104–111° W, solar noon is about 18:55–19:25 UTC.

---

## 4. State discovery and characterisation from embeddings

### 4.1 Clustering choices

- **HDBSCAN** (Campello et al. 2013; McInnes et al. 2017, *JOSS*, https://doi.org/10.21105/joss.00205 [RC]):
  - density-based;
  - finds a variable number of clusters and labels noise;
  - suits behaviour data with one huge mode (perching) and rare modes (stoops).
- **B-SOiD** (Hsu & Yttri 2021, *Nat Commun* 12:5188, https://doi.org/10.1038/s41467-021-25420-x [AB]) is a behaviour-discovery precedent:
  - UMAP to a low dimension, then HDBSCAN;
  - then a classifier trained on the cluster labels, so that new data skip re-clustering.
- **k-means / GMM.**
  - k-means assumes spherical clusters of similar size and will split perching into many pieces.
  - A GMM with BIC gives soft assignments and works well as HMM emissions.
- **Unsupervised ethogram precedents in biologging**:
  - Sakamoto et al. 2009, *PLoS ONE* 4:e5379, k-means ethograms from bird acceleration, https://doi.org/10.1371/journal.pone.0005379 [RC];
  - Chimienti et al. 2016, *Ecol Evol* 6:727, https://doi.org/10.1002/ece3.1914 [RC].
- **Recommendation [OURS]**
  - Cluster **in the embedding space or a PCA of it**, not in a 2-D UMAP. Use UMAP only for display. UMAP distorts density, which can manufacture HDBSCAN clusters.
  - Run HDBSCAN for discovery and a GMM over a range of k for a parametric comparison.
  - Over-cluster, then merge clusters that are indistinguishable on interpretable features (VeDBA, wingbeat power, vertical speed, turn rate, pitch).

### 4.2 Choosing the number of states and testing stability

- **Pohle et al. 2017**, *JABES* 22:270–293, https://doi.org/10.1007/s13253-017-0283-8 [AB]:
  - AIC and BIC tend to **overestimate** the number of HMM states on real movement data, because extra states soak up misspecification.
  - Their pragmatic procedure: fit a range of K, inspect the state-dependent distributions, and choose the smallest K that is biologically interpretable.
- **Hennig 2007**, *CSDA* 52:258–271 (R `fpc::clusterboot`) [AB]:
  - cluster-wise bootstrap Jaccard stability;
  - mean Jaccard ≥ 0.75 indicates a valid cluster, ≥ 0.85 a highly stable one, ≤ 0.5 a dissolved one (thresholds from the package documentation).
- **Consensus clustering** (Monti et al. 2003, *Mach Learn* 52:91 [RC]).
- **Recommendation [OURS]**
  - Bootstrap **by bird**: resample birds, not windows, then re-cluster.
  - Also re-train the encoder with 3–5 seeds.
  - Keep only clusters with Jaccard ≥ 0.75 across both kinds of perturbation.
  - Additionally require that each retained cluster is present in at least about 5 birds and in more than one cohort. A cluster found in one bird or one cohort is a nuisance cluster by definition.

### 4.3 Temporal states: HMMs on embeddings vs handcrafted baselines

- **moveHMM**: Michelot et al. 2016, *Methods Ecol Evol* 7:1308, https://doi.org/10.1111/2041-210X.12578 [RC; DOI not confirmed].
- **momentuHMM**: McClintock & Michelot 2018, *Methods Ecol Evol* 9:1518–1530, https://doi.org/10.1111/2041-210X.12995 [AB].
  - Multiple data streams, covariates on transitions, and multiple imputation for irregular fixes.
- **Leos-Barajas et al. 2017**, *Methods Ecol Evol* 8:161–173, https://doi.org/10.1111/2041-210X.12657 [FT, arXiv 1602.06466]:
  - Eagle case study: one Verreaux's eagle, 20 Hz ACC, maximum MSA per 1 s, two states (low activity vs high activity).
  - Wind speed as a covariate on the transitions was favoured by BIC.
  - Time of day was deliberately left out in favour of environmental drivers.
  - Hierarchical HMMs are discussed as a way to model composite behaviours.
- **Recommendation [OURS]**
  - **Baseline (mandatory).** Fit momentuHMM to handcrafted features on the same events:
    - VeDBA;
    - wingbeat band power;
    - pitch;
    - 1 Hz vertical speed;
    - turn rate;
    - step length / ground speed.
    Use K = 3–7 chosen by the Pohle procedure.
  - **Embedding model.** Fit an HMM with GMM emissions on PCA-reduced embeddings, ordered in time within each bird.
  - **Comparison.** Report agreement between the two (adjusted Rand index, normalised mutual information) and their state-wise feature profiles.
  - The SSL pipeline earns its place only if it finds **stable states the baseline misses**, such as stoops or hunting-sequence motifs, without losing the baseline's states.
  - **Hierarchical HMM [OURS].** Use a hierarchical HMM, or HMM-on-Tier-B, for the two time scales: fine flight modes nested in daily or bout-level modes (resident, foraging, travelling or migrating, roosting).

### 4.4 Transitions and characterisation

- Report for each state:
  - dwell-time distributions;
  - the transition matrix;
  - feature profiles (VeDBA, wingbeat power, pitch, vertical speed, turn rate, altitude above ground);
  - solar-elevation profiles;
  - seasonal profiles;
  - age-class profiles;
  - wind and terrain covariates.
- Expected flight sequences, which the transition matrix can check:
  - thermal soaring ↔ gliding cycles;
  - perch → flapping take-off → soaring.
- Rare states (stoop-like, sustained flapping) should be inspected by plotting raw bursts and 1 Hz tracks for a sample of members.

### 4.5 Diel analysis in solar time

- **Recommendation [OURS]**
  - Express time as **solar elevation**, or as hours relative to local sunrise and sunset (for example, R `suncalc` or Python `astral`), not as UTC hour.
  - Night windows (sun below −6°, civil twilight) are a built-in **roosting control**. Any "flight" cluster with substantial night membership indicates a data or nuisance artefact.
  - Expect thermal-soaring states to concentrate around solar noon, and orographic lift to shift towards morning and evening (Katzner et al. 2015 [AB]).

---

## 5. Golden eagle ecology relevant to interpretation

- **Flight development of juveniles**
  - Hemery et al. 2023, *J Avian Biol* 2023(7-8):e03111, https://doi.org/10.1111/jav.03111 (data at https://doi.org/10.5281/zenodo.7802538) [AB]:
    - 84 GPS-tagged juveniles in France.
    - The post-fledging dependence period averaged 177.9 ± 62.2 days.
    - Flight skills improved rapidly during the first about 60 days after leaving the nest, then plateaued.
    - Departure timing was not limited by flight skill.
  - Nourani et al. 2024, *eLife* [FT]: 55 juveniles in the Alps; landscape flyability increased with age over the first 3 years after emigration; uplift use improved with experience.
  - Expectation [OURS]:
    - Juvenile embeddings should drift over the first months: more flapping, less efficient climbs, smaller or less regular circles.
    - Validate age with **known hatch or tagging dates**, not the current movement heuristic. The heuristic is circular if age is then "discovered" in the latent space.
- **Western North America**
  - Murphy et al. 2017, *J Raptor Res* 51:216, https://doi.org/10.3356/JRR-16-80.1 [AB]:
    - 66 juveniles in the Colorado Plateau and Southern Rockies; all dispersed before 1 year of age.
    - About two-thirds dispersed short distances.
  - Poessel et al. 2022, *Ecosphere* 13:e4072, https://doi.org/10.1002/ecs2.4072 [AB]:
    - Of more than 500 tracked eagles, about 160 made non-routine movements longer than 300 km.
    - These movements fit prospecting and nomadism rather than migration.
    - They were made more often by young birds, outside winter and in north–south directions.
  - Implication [OURS]: "migration / long-distance travel" in this population will include non-seasonal nomadic trips. Define it from Tier-B displacement (net km per day), not from the calendar.
- **Migration speed**
  - Rus et al. 2017 (Miller, Katzner co-authors), *Auk* 134:485–497, https://doi.org/10.1642/AUK-16-147.1 [AB]:
    - Eastern North American migrants travelled 139.8 ± 82.2 km/day in spring and 99.1 ± 60.0 km/day in autumn.
    - Hourly speeds were about 25.6 and 22.2 km/h.
    - Younger birds were faster per hour.
    - Solar radiation was the main driver in spring.
  - Usable bounds for a travel state [OURS]: sustained directed displacement of more than about 50–100 km/day.
- **Diel activity**
  - Golden eagles are diurnal soaring birds. Uplift is solar-driven (thermals) or wind-driven (orographic).
  - Night = roosting, apart from GPS noise. This is a general principle supported by the solar-radiation dependence above [AB]; no eagle-specific night-flight study was read.
  - Expect flight to start after sunrise and to peak late morning to afternoon.
- **Flapping is rare**
  - Golden eagles flapped for 2–38% of flight time (Gupta et al. [AB]).
  - Condors flapped for about 1%, concentrated at take-off (Williams 2020 [FT]).
  - Flapping-related states should therefore be small and linked to take-off, landing, low-altitude flight and weak-uplift conditions.
  - A large "flapping" cluster would be suspicious.

---

## 6. Ranked top 10 design recommendations

**1. Replace hourly UTC windows with observation-event units, and drop empty windows.**
- Evidence:
  - 36% of the current windows contain no measurement, so they are identical inputs that the contrastive loss must still separate.
  - Yuan et al. 2024 [FT]: without std-weighted sampling, a pretext task stayed at chance.
  - BEBE [FT] and Sur et al. [FT] work at seconds-scale windows.
- Action:
  - Units are ACC bursts with nearby fixes, or 1 Hz GPS segments of 30–120 s.
  - Day-scale structure moves to a second tier.
  - Sample training windows with probability increasing with signal SD; keep a floor for perching.

**2. Convert acceleration to g and normalise orientation per bird.**
- Evidence:
  - moveACC defaults [FT]: g = (raw − 2048) × slope, with slope 0.0027/0.001 (Gen 1), 0.0022 (Gen 2), 1/512 (Gen 3), and a Gen-1 Y-axis flip.
  - Garde et al. 2022 [AB]: placement changes DBA by 9–13%.
  - Leg-loop vs backpack attachment differs between studies [FT].
- Action:
  - Map tag ID to generation.
  - Fit per-tag offsets from static night bursts.
  - Rotate each bird's data to its median roosting gravity vector.
  - Expected effect: less bird-identity signal in the input.

**3. Feed physically meaningful features alongside raw signals, and never z-score per window.**
- Evidence:
  - VeDBA/ODBA, pitch and wingbeat-band power separate perch, flapping and passive flight (Sur et al. [FT], Williams 2020 [FT], Duriez [FT]).
  - Per-window z-scoring erases the amplitude difference these rely on.
- Action:
  - Use global, per-channel robust scaling (median/IQR over the whole training set).
  - Add VeDBA, static pitch/roll and 1.5–4 Hz band power as tokens.
  - Compute all encoder features on the common 24-sample crop with the per-crop mean as the static component. Dominant frequency (264-sample bursts only, 0.076 Hz resolution) and 3 s filters are for characterisation only.

**4. Make the encoder see both acceleration and 1 Hz GPS, with absent modalities as absent tokens.**
- Evidence:
  - Acceleration alone fails to separate thermal soaring, slope soaring and gliding (Williams et al. 2015 [AB]).
  - Circling and climb rate do separate them (Duriez [FT], Williams 2020 [FT], Sherub [FT]).
  - mTAN and masked-token transformers handle irregular sampling without zero-filling (Shukla & Marlin 2021 [AB]).
- Action:
  - A two-branch encoder (ACC burst CNN + GPS-segment encoder) fused by a token-set transformer with time-delta encodings.
  - Modality dropout during training.
  - No zero-filling and no presence flags.

**5. Remove sampling-regime cues from the input.**
- Evidence:
  - The current latent space separates tag cohort and sampling regime.
  - Burst length (24/40/264), GPS cadence (1 s vs 5 min) and magnetometer availability (33 birds, until April 2025) all mark cohort.
- Action:
  - Crop every burst to a common 24-sample core for the encoder.
  - Use 1 Hz segments only in Tier A and 5-min fixes only in Tier B.
  - Exclude magnetometer and quaternion data from the default model.

**6. Change what counts as a negative: bird-grouped batches, smaller effective negative sets, and a softer temperature.**
- Evidence:
  - Cheng et al. 2020 [FT]: same-subject negatives plus an adversarial subject penalty reduced subject effects when subjects were few. There are 62 birds here.
  - Qian et al. 2022 [FT]: more than 128 negatives degraded SimCLR through same-class false negatives.
- Action:
  - Batches of a few birds × many windows, or masking of cross-bird negatives.
  - Sweep batch 256–2048 and τ 0.07–0.5.
  - Try NNCLR or debiased NT-Xent.

**7. Use domain-appropriate augmentations.**
- Evidence:
  - Qian et al. 2022 [FT]: rotation and shuffling were the worst augmentations for SimCLR on UCIHAR; jitter and scaling were the best.
  - Tang et al. 2020 [AB]: rotation helped in other settings.
  - Cheng et al. [FT]: temporal cutout and delay were most effective.
  - Pitch, roll, wingbeat frequency and the direction of vertical movement carry behaviour.
- Action:
  - Use jitter, ±10% scaling, ±10–15° rotation, crops and offsets, cutout and modality dropout.
  - Do not use full rotation, axis swaps, shuffling, strong time-warp or time reversal as invariances.

**8. Make nuisance-probe and held-out-bird evaluation the acceptance test.**
- Evidence:
  - BEBE [FT] evaluates on individuals unseen in training and averages metrics per individual.
  - Cheng et al. [FT] measure subject invariance directly.
- Action:
  - Linear probes for bird ID, cohort, cadence, burst length and harness should be near chance on held-out data.
  - Probes for rule-derived flapping, circling, gliding, perched and night labels (Section 2.2) should be high.
  - Rule labels are used only for evaluation, never for training.

**9. Calibrate the interpretation against a handcrafted-feature HMM baseline, and use solar time.**
- Evidence:
  - momentuHMM [AB] and Leos-Barajas et al. 2017 [FT] (eagle accelerometer HMM).
  - Pirotta/Katzner state-space model [AB]: five states, two thermal variants.
  - Pohle et al. 2017 [AB]: information criteria overestimate K.
- Action:
  - Fit a momentuHMM on VeDBA, wingbeat power, pitch, vertical speed, turn rate and speed.
  - Fit an HMM on the embeddings.
  - Compare the two with ARI and NMI and with state profiles.
  - Index all diel results by solar elevation and use night as the roosting control.
  - The magnetometer subset (33 birds) can validate circling, or act as a teacher, offline.

**10. Discover states with HDBSCAN or GMM in embedding space, and keep only stable, multi-bird, multi-cohort clusters.**
- Evidence:
  - B-SOiD [AB] (UMAP + HDBSCAN + classifier).
  - Hennig 2007 [AB]: Jaccard ≥ 0.75 indicates a stable cluster.
  - Pohle et al. [AB]: pick the smallest interpretable K.
- Action:
  - Bootstrap over birds and encoder seeds.
  - Require clusters to be present in at least 5 birds and in more than one cohort.
  - Validate age effects with known hatch dates, not the movement heuristic (Hemery et al. 2023 [AB]: a skill plateau about 60 days after leaving the nest).
  - Define migration and nomadism from net daily displacement (Poessel et al. 2022 [AB]; Rus et al. 2017 [AB]: about 100–140 km/day on migration).

---

## 7. Gaps and items to verify

- **e-obs details**
  - The e-obs manual was not obtained. Two things remain unknown: the direction of the Gen-1 Y-axis flip, and whether post-2018 tags changed the defaults.
  - Check tag IDs against the generation ranges in moveACC.
- **Unread classification algorithms**
  - Katzner et al. 2015: full text blocked.
  - Pirotta et al. 2018 state-space model: full text blocked.
  - Their thermal and orographic thresholds were therefore not read.
- **Fluhr and Poessel on flight-mode classification:** no Fluhr flight-classification paper was located, and no Poessel flight-mode paper was read. Poessel appears only in Section 5. Both are uncovered.
- **No published golden eagle signatures** were found for stoops or hunting from e-obs acceleration plus GPS.
- **Unverified items**
  - Citations marked [RC] (SimCLR, CPC, TNC, Um 2017, DANN, debiased contrastive, PatchTST, SeFT/Raindrop, HDBSCAN JOSS, Monti, moveHMM DOI, Sakamoto, Chimienti, Wilson 2006, Qasem 2012, Duerr 2012) were not re-verified.
  - Several facts in [AB] items come from snippets.
