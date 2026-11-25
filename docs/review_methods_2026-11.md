# Methods review for the next stage (sex, age, anomalies, seasons, sub-states)

Written 2026-11-25. Method research only. No project data were read, and no remote machine was contacted.
Web searches used generic topics only (no bird IDs, tag IDs, coordinates or project names).

This file extends `docs/literature.md` (2026-11-24) and does not repeat it. Sources already covered there are referred to by section rather than described again:

- BEBE, Yuan et al. 2024, Mao et al. 2025: literature.md §3.1.
- Qian et al. 2022 and Tang et al. 2020 on augmentations: §3.3.
- Cheng et al. 2020 (subject-aware contrastive learning) and DANN: §3.4.
- Hennig 2007, Pohle et al. 2017, momentuHMM and Leos-Barajas et al. 2017: §4.2–4.3.
- Hemery et al. 2023, Nourani et al. 2024, Poessel et al. 2022: §5.

Some things are already done in the project and are not recommended again: bird-grouped batches (4 birds per batch), yaw augmentation, the bird-bootstrap Jaccard rule, held-out-bird probes, and masking of same-bird negatives within 30 min. See DESIGN.md §2c and §6, and RESULTS_stage4_r4-r6.md.

## Evidence labels (same scheme as literature.md)

- **[FT]**: full text, or the results tables, read during this review.
- **[AB]**: abstract, repository record, press release or search-engine snippet only.
- **[RC]**: recalled and not checked during this review.
- **[OURS]**: an inference or recommendation of this review. It is not a published result.

Effort is given as **S** (hours, on existing outputs), **M** (one to three days, CPU or a single GPU run) or **L** (a week or more, several GPU runs). Risk means the chance that the action costs its effort without changing a conclusion.

---

## 1. Self-supervised learning for animal-borne accelerometry: is SimCLR still the right objective?

### What the evidence says

- **Masked reconstruction beat SimCLR with frozen encoders on human data.**
  - Source: Ek, Presotto, Civitarese, Portet, Lalanda & Bettini 2024, arXiv 2404.15331 [FT, results tables read].
  - Setup: six HAR datasets, waist accelerometer and gyroscope, 2.56 s windows (128 samples at 50 Hz), leave-one-dataset-out with subject-wise splits.
  - Results with the encoder frozen (mean macro-F1):

    | Method | iSPL encoder | HART encoder |
    |---|---|---|
    | MAE | 80.9 | 82.6 |
    | data2vec | 66.9 | 73.8 |
    | SimCLR | 63.0 | 50.2 |
    | Supervised | 65.7 | 75.0 |

  - With fine-tuning, all methods came within about 2 points of each other.
  - SimCLR in that study used random 3-D rotation, which literature.md §3.3 flags as harmful for posture-dependent behaviours. Part of SimCLR's deficit may therefore come from its augmentation choice.
- **Hybrid objectives were best in the newest benchmark.**
  - Source: BenchHAR, Cai et al. 2026, arXiv 2605.08296 [AB].
  - Eight SSL methods were compared, and the method names were not visible.
  - Combined reconstruction and contrastive pre-training generalised best.
  - CNN encoders gave the most generalisable representations.
  - Existing SSL methods "struggle to generalize" to unseen target distributions.
- **Frozen SimCLR can collapse on new data.**
  - Source: Fortes Rey, Nshimyimana & Lukowicz 2023, arXiv 2307.01168 [FT].
  - SimCLR, CPC, multi-task SSL and reconstruction were compared.
  - Frozen SimCLR on one dataset fell to macro-F1 4.4, against a supervised baseline of 46.2. The authors attribute this to augmentation and body-position mismatch.
  - Fine-tuning the encoder improved all the other pretext tasks.
- **Large-scale foundation models use masked reconstruction.**
  - Narayanswamy et al. 2024, "Scaling Wearable Foundation Models" (LSM), arXiv 2410.13638 [AB]: 40 M hours from 165 k people; scaling laws were shown for imputation, interpolation and extrapolation objectives.
  - The MAE-versus-SimCLR comparison inside that paper was not read.
- **No multi-species bio-logging foundation model was found.**
  - BEBE [literature.md §3.1] remains the only animal benchmark with by-individual splits.
  - Mao et al. 2025 is the only cross-species contrastive pre-training found.
- **Comparisons not found.** No head-to-head comparison of SimCLR against TS2Vec, BYOL or VICReg on cross-subject accelerometry was found.
  - The TS-TCC paper (Eldele et al. 2021, arXiv 2106.14112 [AB]) reports TS-TCC above SimCLR and CPC above SimCLR on general time series, with linear evaluation. Its claim that "temporal features matter" supports a temporal objective.
- **Window length.** Nothing found changes literature.md §1.4 and §3.1.
  - The human benchmarks use 2.5–10 s windows.
  - 10 s at 20 Hz is within the range of Yuan et al. (10 s) and BEBE.
- **Evaluation protocol.** Wilson, Schoeman, Bradley & Clemente 2025, *J Anim Ecol* 94:1322–1334, https://doi.org/10.1111/1365-2656.70054 [FT]:
  - 79% of 119 animal accelerometer ML studies did not validate well enough to detect overfitting.
  - Recommendations: leave-individual-out testing, more than one test individual, no tuning on the test set, several metrics with ranges across folds, and a randomised-label sanity check.
  - The project's held-out-bird protocol already meets this. The randomised-label check is not yet done.

### Interpretation for this project [OURS]

- The evidence that masked reconstruction is better comes from human data, with frozen linear probes against **labels**. The project's success criteria are different: physical probes, low identity, and stable states.
- The behaviour probes are already high (window turn 0.913–0.919, and the others within ±0.01 of r4). The weak point is **state stability**, not probe accuracy.
- An objective that keeps more per-window detail (reconstruction) could raise identity as well as behaviour content, because posture and wingbeat detail carry individual signal (RESULTS §2).
- A switch to MAE is therefore not justified by the literature for this goal. A hybrid objective is worth one controlled run.

### Recommendations for this project

1. **Add a masked-reconstruction auxiliary loss to the current SimCLR model (hybrid, as in BenchHAR). Do not replace SimCLR.**
   - Mask 30–50% of accelerometer patches and GPS tokens.
   - Compare against r6_yaw60_bpb4 on four things: the behaviour probes, held-out identity, minimum bird-bootstrap Jaccard and held-out ARI.
   - Keep the new model only if Jaccard or ARI rises and identity does not rise.
   - Effort M (one GPU run). Risk medium: identity may increase.
2. **Run the randomised-label sanity check on the probes, as Wilson et al. 2025 recommend.**
   - Shuffle the probe labels across birds and confirm that the probes fall to chance.
   - This is a cheap protection for the later sex and age claims.
   - Effort S. Risk low.
3. **Do not invest in TS2Vec, BYOL, VICReg or a foundation-model port now.**
   - There is no animal evidence that they beat SimCLR. The bottleneck is stability and identity, not the choice of objective.
   - Revisit only if recommendation 1 shows that the objective matters.
   - Effort none. Risk: missing a gain (low).

---

## 2. Removing individual identity, and when to keep it

### What the evidence says

- **The adversary result in r5_conf has a known explanation.**
  - Source: Elazar & Goldberg 2018, EMNLP, arXiv 1808.06640 [AB].
  - The adversary reached chance on the training data, yet a post-hoc classifier on the same representations recovered the attribute "substantially" above chance.
  - Run r5_conf showed the same pattern: adversary accuracy 0.028 on training birds (chance 0.0125), and held-out identity unchanged (RESULTS §3).
  - [OURS] There is a second reason specific to this project. The adversary learns to confuse the **80 training-bird classes**, while the held-out probe is a new 12-way problem. Confusing a fixed set of classes does not remove the general directions that separate individuals.
- **Algorithms for domain generalisation rarely beat plain training.**
  - Source: Gulrajani & Lopez-Paz 2020/2021, "In Search of Lost Domain Generalization", arXiv 2007.01434 (ICLR 2021) [AB].
  - Carefully tuned ERM matched every domain-generalisation algorithm in DomainBed, and none beat it by more than about 1 point.
  - Model selection without access to the target domain is the main difficulty.
  - Rosenfeld, Ravikumar & Risteski 2021, "The Risks of Invariant Risk Minimization", arXiv 2010.05761 (ICLR 2021) [AB]: IRM can lock onto non-invariant features and often does no better than ERM.
- **Non-adversarial alternatives exist.**
  - Moyer et al. 2018, "Invariant Representations without Adversarial Training", NeurIPS 31, arXiv 1805.09458 [AB]: a variational mutual-information penalty with no min-max game.
  - Linear concept erasure:
    - INLP, Ravfogel et al. 2020, ACL, arXiv 2004.07667 [AB].
    - LEACE, Belrose et al. 2023, arXiv 2306.03819 [AB]: closed-form, "provably prevents all linear classifiers from detecting a concept" with minimal change.
- **Speaker verification has a mature recipe for the same problem.**
  - Nuisance attribute projection (NAP) and within-class covariance normalisation (WCCN), with LDA before WCCN, project out the directions of largest within-class nuisance variability and are fitted on training speakers.
  - Source: Dehak et al. 2011, IEEE TASLP, https://sls.csail.mit.edu/publications/2011/Dehak_IEEE_May2011.pdf [AB].
  - [OURS] Here the roles are reversed: the nuisance is the **between-bird** variability within a behaviour state.
- **Orientation invariance (HAR).** Common approaches are orientation-invariant scalar features (magnitude, heuristic features), mapping to a reference frame, and quaternion-based features.
  - Barua et al. 2024, PMC10874570; a 2022 study, PMC9313140 [AB].
  - None of them compares the approaches on a common benchmark.
  - The project already uses a gravity-aligned frame plus yaw augmentation. Residual mounting yaw has an SD of 15.8°, and the posture-driven handcrafted identity floor is 0.35 (RESULTS §2).
- **Per-subject normalisation.** No source was found that tests per-subject feature standardisation for HAR generalisation.
  - The HAR literature mostly moves the other way, towards personalisation with a few target-subject samples (PMC 30222588; arXiv 2006.07472 [AB]).
- **Individual signal can be biologically real.**
  - Harel, Horvitz & Nathan 2016, *Sci Rep* 6:27865, https://doi.org/10.1038/srep27865 [FT], griffon vultures, 8 juveniles and 9 adults, GPS with ACC tags. Juveniles compared with adults:

    | Measure | Juveniles | Adults |
    |---|---|---|
    | Thermal climb rate | 1.26 m/s | 1.6 m/s |
    | Circling radius | 31.8 m | 35.9 m |
    | Flapping proportion | 0.05 | 0.02 |
    | In-flight ODBA | 1.34 | 0.99 |

  - Hatzl et al. 2026 (golden eagles, Alps; supplementary data https://doi.org/10.5281/zenodo.19002502, preprint https://doi.org/10.1101/2025.04.21.649390) [AB]: the sexes differed **only in time spent flapping**, despite strong size dimorphism.
  - Flack et al. 2018, *Science* 360:911–914, https://doi.org/10.1126/science.aap7781 [AB]: 27 juvenile white storks. Followers flapped considerably more than leaders. Snippets state that within-individual flapping differences were stable; that point was not read in the paper itself.
  - Cecere et al. 2020, *Movement Ecology*, https://doi.org/10.1186/s40462-020-00206-w [AB, keywords only]: consistent inter-individual differences in foraging tactics of lesser kestrels, measured with GPS and accelerometers.

### Interpretation for this project [OURS]

- **Sex and age are properties of the bird.** Any method that makes held-out birds indistinguishable will also erase bird-level sex and age differences. The next goals (Q4) therefore **conflict** with further identity removal.
- **Identity is already near the floor.** Run r6_yaw60_bpb4 has identity 0.313 against a handcrafted floor of 0.35. Pushing it lower risks removing posture and flight style, which are the signals that differ by age and sex in the studies above.
- **The remaining question is mounting versus biology.** If identity comes from the mounting, it should change when a bird is re-tagged or when the harness settles. If it comes from flight style, it should persist.

### Recommendations for this project

1. **Use the representation in two ways.**
   - **State discovery**: optionally apply a post-hoc linear erasure of between-bird directions (step 2).
   - **Demographics, anomalies and seasonal work**: use the un-erased h, always with bird-level splits.
   - Record this split in DESIGN.md as a decision for Nill.
   - Effort S. Risk low.
2. **Post-hoc linear erasure for state discovery.**
   - Fit on **training birds only**, within each coarse state: the between-bird scatter of state-conditional bird means (an LDA/NAP-style projection), or LEACE with bird as the concept.
   - Project h and re-run the held-out identity probe and the stage-5 stability on **held-out birds**.
   - Run INLP or LEACE on a 12–80-class concept with care, because its generalisation to new birds is the same open question as for the adversary.
   - Effort S–M (CPU). Risk medium: it may not transfer, exactly like the adversary.
3. **Mounting-versus-biology test.**
   - Train the identity probe on season or year A and test it on season or year B of the same birds. Compare with a probe trained and tested within one season.
   - If any bird was re-tagged or had its harness changed, test across that event; a large drop there means mounting.
   - Also test identity **within the night-still subset**, where flight style cannot contribute (0.18 on r4; not yet measured on r6_yaw60_bpb4). That value approximates the mounting-only component.
   - Effort S. Risk low.
4. **Stop adversarial work.** Elazar & Goldberg and DomainBed predict the r5_conf result. Do not tune λ further.
   - Effort none. Risk none.

---

## 3. Discovering behavioural states without labels

### What the evidence says

- **Prediction strength** is a second stability criterion next to Hennig's Jaccard (literature.md §4.2).
  - Source: Tibshirani & Walther 2005, *J Comput Graph Stat* 14:511–528; R `fpc::prediction.strength` documentation [FT, documentation read].
  - Split the data in halves, cluster both, classify one half to the other's clusters, and take the **minimum over clusters** of the share of co-clustered pairs that are kept.
  - Choose the largest k with a value above 0.8 (or 0.9).
  - [OURS] The minimum over clusters, like the minimum Jaccard, exposes the single weakest boundary.
- **Duration-aware state models.**
  - Hidden semi-Markov models give each state an explicit dwell-time distribution, and have been applied to accelerometry (PMC6326431, PMC8032701; arXiv 2006.09061) [AB].
  - Keypoint-MoSeq (Weinreb et al. 2024, *Nat Methods*, https://doi.org/10.1038/s41592-024-02318-2) [AB]:
    - A "stickiness" hyperparameter sets the syllable time scale.
    - Without it, noise produced implausibly short syllables.
    - Validation used expert labels, accelerometry, neural data and comparison with B-SOiD, VAME and MotionMapper.
- **Multi-scale (hierarchical) HMMs** model fine behaviour nested in coarse behaviour, with data at different resolutions.
  - Source: Adam, Griffiths, Leos-Barajas et al. 2019, *Methods Ecol Evol* 10:1536–1550, record at https://eprints.whiterose.ac.uk/148171 [AB]. Case studies were cod and a horn shark with acceleration data.
- **Validation without ground truth in soaring birds.**
  - Unsupervised HMMs on albatross accelerometer data alone were about as accurate as models that also used magnetometer data, according to a *Movement Ecology* 2021 paper, https://doi.org/10.1186/s40462-021-00243-z [AB, snippet only; authors not read].
  - Clay et al. 2020, *J Anim Ecol*, NORA record https://nora.nerc.ac.uk/id/eprint/525481 [AB]: an HMM on wandering albatross tracks with three states; wind and sex covariates on the transitions.
- **Clusters are not behaviours.** Unsupervised states capture data structure, not necessarily biologically meaningful behaviour (McClintock et al., arXiv 2002.10497 [AB]).
  - This supports the project's rule-label probes and the night-roost control as validation.

### Interpretation for this project [OURS]

- The HMM and GMM labels agree on 98% of windows, so the current HMM adds almost no temporal smoothing. The states are still window classes, not bouts.
- The minimum Jaccard (0.62) is a single number. It is not reported which boundary fails.
  - Likely candidates are "cruising" against "fast descending glide", and "rest" against "perched active". Both are continua in speed, climb or VeDBA.
  - If one boundary is the problem, merging or re-defining that pair may meet the rule at K = 5 without losing real structure.
- Finer sub-states (for example, thermal-circling variants by wind or climb rate, or flapping at take-off versus sustained flapping) are better found **within** a stable coarse state than by raising global K.

### Recommendations for this project

1. **Report the bootstrap Jaccard per cluster and per pair of clusters** (the co-assignment confusion), not only the minimum.
   - Name the boundary that fails.
   - Test K = 5 with that pair merged against the 0.75 rule.
   - Effort S. Risk low.
2. **Add prediction strength with bird-level halves.** Split the **birds** into halves, not the windows, so it is an individual-level replication test. Report it next to Jaccard.
   - Effort S. Risk low.
3. **Replace the HMM with a sticky HMM or an HSMM with explicit dwell times on the 10 s window sequence of each continuous segment.**
   - Choose stickiness, or a minimum dwell time, from the physics: about 2–3 windows for glides and ≥ 3 windows for thermal circling (one circle takes about 20–40 s).
   - Report dwell-time distributions and state stability after smoothing.
   - Effort M (CPU). Risk medium: it may smooth over real short events such as take-off flapping.
4. **Hierarchical sub-state discovery.** Within each stable coarse state:
   - re-run PCA, GMM and Ward on h restricted to that state;
   - apply the same bird-bootstrap and prediction-strength rules;
   - characterise the sub-states only by variables *not* used to define the parent state, for example wind (ERA5), terrain and solar elevation for thermal sub-states.
   - Effort M. Risk medium.
5. **Interpret states at the bout scale.** Fit a 2-level model (bout-level states over the window-level states, as in Adam et al. 2019) only after recommendation 3. Expected bout states are travel, local foraging and roosting.
   - Effort L. Risk medium.

---

## 4. Inferring sex and age from movement

### What the evidence says

- **No published raptor study was found that classifies sex or age from movement or accelerometry with a reported individual-level accuracy.** Published work reports *group differences*, not classifiers:
  - **Age.**
    - Harel et al. 2016 [FT]: see §2. The effects were moderate, with 8 + 9 birds.
    - Brønnvik & Flack 2025, *Curr Biol* 35:3736–3744, https://doi.org/10.1016/j.cub.2025.06.044 [AB]:
      - 151 white storks tracked over life.
      - Juveniles performed better under supportive conditions; adults performed better under challenging conditions and spent more energy in flight. Snippets add that older birds flapped more.
      - The direction of age effects on flapping is therefore **not** universal and depends on conditions.
    - Hatzl et al. 2026 [AB]: in juvenile golden eagles, flight efficiency (soaring relative to flapping) predicted earlier emigration, but developmental *rates* did not.
  - **Sex.**
    - Hatzl et al. 2026 [AB]: in post-fledging juvenile golden eagles, the sexes differ only in flapping time.
    - García-Jiménez, Pérez-García & Margalida 2018, *BMC Ecol* 18:39, https://doi.org/10.1186/s12898-018-0195-7 [FT]:
      - 19 bearded vultures; the sex effect was "mild", appeared only in cumulative daily distance, and was tested only in territorial birds.
      - Territorial status mattered more than sex or breeding.
    - Poessel et al. 2016 (golden eagles, California), https://pubs.usgs.gov/publication/70170070 [AB]: space use increased with age, was season-dependent, and was "not sex-dependent".
    - Clay et al. 2020 [AB]: wandering albatross sexes respond differently to wind. That species is strongly dimorphic, and the sexes forage in different areas.
- **The only sex classifier found is weak and probably not split by bird.**
  - Fayzur 2018, arXiv 1808.08613 [AB]: shearwater GPS trajectories, F1 about 0.77, 5-fold CV.
  - Whether the folds were split by bird is not stated.
- **Breeding status has working methods.**
  - nestR: Picardi et al. 2020, *Movement Ecology* 8, https://pubs.usgs.gov/publication/70211689 [AB].
  - Nest-site detection positive predictive value was 87–100% and sensitivity 88–92%; attempt fate was correct for 86–100% of attempts.
  - The method uses recursion and persistence of revisits to one site, so it is a GPS-recurse method and not an accelerometer method.
- **Unit of analysis.**
  - Wilson et al. 2025 [FT]: windows from one individual are not independent, and random within-individual splits inflate accuracy.
  - With a bird-level label, the effective sample size is the number of birds (89 with sex; about 17–20 young-tagged), not the 5.5 M windows.
- **Project-specific confounds** [OURS, from DESIGN.md §2c and RESULTS §1]:
  - **Cohort and season.** The 2024 cohort is mostly nestlings; the 2025 cohort is mostly territorial adults seen Oct–Mar. "Young-tagged" is therefore close to "2024 cohort", with that cohort's recording schedule and seasons.
  - **Age is not fixed over time.** A bird tagged as a nestling in 2024 is a subadult in 2026, so "young-tagged" is a time-varying covariate (time since tagging).
  - **Recording mode is informative.** Continuous mode is almost never still and almost always has GPS: 25 still windows out of 91,913 full k=1 windows, and 99.99% of them have GPS. "Not still" includes perched-active windows, so this is not the same as "flight". Any feature based on window shares depends on when continuous mode was on, which may differ by cohort and season.
  - **Territorial and breeding status** is a likely stronger driver than sex (as in the bearded vultures) and is not recorded.

### Interpretation for this project [OURS]

- Expect **sex to be weak or null.** The earlier permutation test on state shares gave p = 0.14–0.89. The literature reports at most mild sex effects in monomorphic-behaviour raptors, concentrated in flapping.
- **The age contrast is young-tagged against age not stated, not young against adult** (DESIGN.md §2). The not-stated group is unlabelled and may contain young birds, so this is a positive-unlabelled problem. Report results as "young-tagged vs not stated". Treat not-stated birds that score as young as candidates to check against field records, not as classifier errors. Expect the AUC to understate the true age separability.
- Expect **age (young-tagged versus not stated) to be detectable**, but largely through cohort, season and recording mode unless these are controlled.
- With about 17–20 young birds, a leave-birds-out AUC will have a 95% interval roughly ±0.15 wide. That is a rough expectation, not computed on the data.

### Recommendations for this project

1. **Bird-level design.**
   - Aggregate per bird and month (or bird and season): state shares within a fixed recording mode, and the means and covariances of h within each state.
   - Fit a regularised logistic regression or a mixed model, with **bird-grouped** cross-validation (leave-k-birds-out, stratified by cohort).
   - Test against a **bird-level permutation null**: shuffle sex or age labels across birds, keeping each bird's windows together, 1,000 times.
   - Report AUC with a bird-bootstrap interval.
   - Effort S–M. Risk low (a null is still an answer).
2. **Matched comparisons for age.**
   - Restrict to calendar months, recording modes and burst types that both groups have.
   - Use time since tagging (or estimated age in months) as the covariate, not a fixed label.
   - Run the same model with **cohort** as the target. If cohort is predicted as well as age, the age result is not separable from cohort.
   - Effort M. Risk medium: there may be too little overlap.
3. **Pre-register the sex hypothesis around flapping** (Hatzl et al.; their result is from post-fledging juveniles, so its transfer to adults is untested): flapping share, flapping VeDBA and wingbeat band power within the flapping state.
   - Treat everything else as exploratory, with correction for multiple tests.
   - Effort S. Risk low.
4. **Within-state latent comparisons over state shares.**
   - For each stable state, test whether the bird-mean h differs by group. Use a bird-level permutation test on the distance between group centroids, or a PERMANOVA on bird means.
   - This uses flight *style*, which is where Harel et al. found the age effects (climb rate, circle radius, flapping).
   - Effort S. Risk low.
5. **Breeding status as a confound or target.** If the field data contain nest locations or territory status, compute nestR-style revisitation per bird-season from GPS and add it as a covariate. Without it, sex effects in the breeding season are uninterpretable.
   - Effort M. Risk: the data may not exist.

---

## 5. Anomalous individuals and windows

### What the evidence says

- **Mahalanobis distance on self-supervised features is a strong unsupervised outlier score.**
  - Source: SSD, Sehwag, Chiang & Mittal 2021, ICLR, arXiv 2103.12051 [AB].
  - The score is the distance to the nearest cluster mean in the SSL feature space.
  - It used only unlabelled in-distribution data, beat most unsupervised detectors, and was on par with supervised ones on image benchmarks.
- **Rare-behaviour detection in a bio-logging application.**
  - Source: Tanigaki et al. 2024, *PNAS Nexus* 3:pgad447, https://doi.org/10.1093/pnasnexus/pgad447 [FT].
  - An isolation forest on window features (mean, variance, crest, RMS; 1 s accelerometer windows) was distilled into an on-tag decision tree. It triggered video on 18 streaked shearwaters.
  - Validation was by **manual review of what the triggered videos showed**, which found new behaviours (head-shaking after take-off, pre-dive dips).
  - The authors note that unsupervised flags do not guarantee novelty.
- **Mortality and carcass detection.**
  - Source: Rast et al. 2024, *J Appl Ecol* 61:2936–2945, https://doi.org/10.1111/1365-2664.14810 [AB, press release and snippets].
  - An SVM classified accelerometer behaviour (trained on about 15,000 labelled points from 2 zoo and 27 wild vultures). Feeding behaviour was combined with GPS by spatial clustering.
  - Field teams checked more than 500 suspected carcasses and more than 1,300 other clusters; the reported precision was about 0.92.
  - The precision and recall figures for the behaviour classifier (0.95 and 0.89) come from snippets and were not verified.
- **A ready-made validation set.** The literature.md night-roost control (any flight at night is an artefact) also serves here: anomalies that concentrate at night or in one recording mode are data problems, not behaviour.

### Interpretation for this project [OURS]

- Held-out identity is 31%. An outlier score computed against the training population will therefore rank **whole new birds** as unusual. A per-bird reference fixes this, but then misses birds that are unusual throughout.
- Two questions need two scores:
  - **Unusual window for this bird**: the distance to the bird's own state-conditional distribution.
  - **Unusual bird**: the distance between the bird's per-state mean embedding (or state shares) and the population distribution of bird means, with leave-one-bird-out.
- Tag or harness problems (for example, a slipped harness) are expected to show up as anomalies, with a step change in gravity direction. This is useful for quality control, but it must be separated from behaviour.

### Recommendations for this project

1. **Window-level score.**
   - Use the Mahalanobis distance to the nearest state component, with each bird's own windows from other days as the reference. Use a robust covariance per state, shrunk towards the population.
   - Flag the top 0.1%.
   - Then check whether the flags concentrate in a bird, a recording mode, a burst type, the night or a single day. Concentration in any of these means an artefact.
   - Effort S–M. Risk medium: many artefacts.
2. **Bird-level score.**
   - Compare each bird's vector of per-state mean h and state shares (within a common recording mode) with the leave-one-out population.
   - Report the top 5 birds with the features that drive the score.
   - Effort S. Risk low.
3. **Validation.**
   - Plot raw bursts and 1 Hz tracks for a stratified sample of the top flags and label them blind (as Tanigaki et al. did with video).
   - Inject synthetic anomalies into held-out windows and measure recall: a rotated gravity vector (harness slip), a flat-line or stuck axis, an impossible speed, a time-reversed climb.
   - If field records of deaths, tag loss or recaptures exist, test whether the score rises in the days before. Do not search for them in this review; it is Nill's decision whether to request them.
   - Effort M. Risk low.
4. **Mortality and tag-off flags are simple rules, not ML.**
   - Flag zero VeDBA and a constant gravity vector for more than 24 h during daylight.
   - Keep these separate from the behaviour-anomaly score.
   - Effort S. Risk low.

---

## 6. Seasonal and longitudinal behaviour with changing duty cycles

### What the evidence says

- **The sampling interval biases movement metrics and inferred states.**
  - Source: Schoombie, Wilson, Ropert-Coudert, Dilley & Ryan 2024, *Movement Ecology* 12:59, https://doi.org/10.1186/s40462-024-00499-1 [FT].
  - Short intervals (≤ 6 s) overestimated distance by about 4% for straight paths and about 9% for sinuous paths.
  - Intervals over 60 s underestimated it by about 10–11% for straight paths and about 38–40% for sinuous paths.
  - Turning-angle error rose sharply between 1 and 10 s.
  - Inferred HMM states were unstable at under about 5 min and at hourly intervals.
  - The authors recommend bias-aware comparison across sampling rates, point (instantaneous) speed at 1 Hz, or subsampling to the coarsest common rate.
- **Hierarchical GAMs for smooth seasonal trends with individual variation.**
  - Source: Pedersen, Miller, Simpson & Ross 2019, *PeerJ* 7:e6876, https://doi.org/10.7717/peerj.6876 [AB].
  - A global smooth plus group-level smooths shrunk towards it, fitted in `mgcv`.
  - [RC] A cyclic cubic spline (`bs = "cc"`) for day of year is standard in `mgcv`. It was not checked in this paper.
- **Irregular sampling in state models.**
  - momentuHMM handles irregular data by filtering or imputing to a regular grid first (McClintock & Michelot 2018, arXiv 1710.03786 [AB]).
  - Continuous-time and integrated HMMs handle irregular sampling directly but are much slower (arXiv 1808.01755; Grainger PhD thesis, Sheffield) [AB].

### Interpretation for this project [OURS]

- **This is the largest unaddressed bias for Q4 and Q6.**
  - Windows exist only when the tag records. Continuous mode (1 Hz GPS, 24-sample bursts) is almost never still. Whether it is switched on by activity or by a timetable is not known; that it is switched on mainly in flight is [OURS, unverified], inferred from co-occurrence.
  - A "share of windows in state s per month" is therefore a share of *recorded* time, not of the bird's time.
  - Any change of duty cycle by season, tag or cohort shows up as a seasonal or demographic effect.
- If the burst-mode schedule (40- and 264-sample bursts) runs at fixed intervals regardless of behaviour, it is the closest thing in this dataset to a **systematic time sample**, and time budgets should come from it. [OURS, unverified: regularity and behaviour-independence of the burst intervals have not been checked.]
- The 10 s windows from continuous mode are best used for **conditional** questions: given that the bird is in flight, which flight mode is it in, and how does that change with season?

### Recommendations for this project

1. **Time-weighted budgets from the fixed-interval schedule only.**
   - **Precondition (check first).** Use the stage-2 schedule profile (per tag and month) to confirm that burst intervals are regular and do not depend on behaviour. Compare inter-burst interval distributions for still against moving bursts and for day against night, and check whether continuous-mode starts follow a timetable or an activity trigger. If they are activity-triggered, the weighting must model the trigger instead of assuming a time sample.
   - For each bird and day, use only windows from the regular burst schedule. Weight each window by the time interval it represents (the inter-burst interval).
   - Compute state budgets as hours per day. Report them by solar elevation and by day of year.
   - Use continuous-mode windows only for within-flight conditional shares.
   - Effort M. Risk low. **This should precede any seasonal, sex or age claim.**
2. **A model that sees the sampling.** Fit an HGAM per state:
   - response: the state indicator, or hours per day;
   - a cyclic day-of-year smooth;
   - a solar-elevation smooth;
   - bird-level random smooths;
   - recording mode and burst type as fixed factors;
   - an offset for log recorded time.
   - Seasonal effects that disappear when recording mode enters the model were sampling effects.
   - Effort M. Risk low.
3. **Subsample to a common schedule for a robustness check.** Thin every bird-month to the sparsest schedule shared across birds (Schoombie et al.) and re-compute the seasonal curves. Curves that change were schedule effects.
   - Effort S. Risk low.
4. **Longitudinal change within birds.** For birds with more than 12 months of data, fit within-bird trends of state-conditional h (for example, the climb-rate direction within thermal circling) against time since tagging. This is the within-individual design that separates age from cohort.
   - Effort M. Risk medium: few birds have long records.
5. **Season-conditioned anomalies.** For Q5, use reference distributions from the same bird and the same season or month, so that a seasonal shift is not flagged as an anomaly.
   - Effort S. Risk low.

---

## 7. Top 8 actions across all questions (priority order)

| # | Action | Question | Effort | Risk | Why first |
|---|---|---|---|---|---|
| 1 | First check that the burst schedule is regular and independent of behaviour; then compute time-weighted state budgets from that schedule only, and use continuous mode only for conditional shares (§6 rec. 1). | Q6, Q4 | M | Low | Every seasonal, sex and age result is otherwise confounded with recording mode: continuous mode is almost never still (25 of 91,913 windows) and almost always has GPS. |
| 2 | Bird-level demographic design: aggregate per bird and month within the same recording mode, bird-grouped CV, bird-level permutation null, cohort as a negative-control target, time since tagging as the age covariate (§4 recs. 1–2). | Q4 | S–M | Low | 89 and about 20 birds are the true sample sizes. Without a cohort control, an age result cannot be interpreted. |
| 3 | Keep identity in h for demographics and anomalies; erase only for state discovery. Stop adversarial work (§2 recs. 1 and 4). | Q2, Q4 | S | Low | Sex and age are bird properties. Identity is already at the handcrafted floor, and the adversary failure matches Elazar & Goldberg. |
| 4 | Per-cluster and per-pair bootstrap Jaccard, plus prediction strength with bird halves; test a K = 5 merge of the failing pair (§3 recs. 1–2). | Q3 | S | Low | Shows whether the stability failure is one boundary between continua or general. |
| 5 | Mounting-versus-biology identity tests: across seasons or years, across re-tag events, night-still subset (§2 rec. 3). | Q2 | S | Low | Decides whether the residual 31% should be removed (mounting) or kept (flight style). |
| 6 | Sticky HMM or HSMM with physically chosen minimum dwell times, then within-state sub-state discovery with the same stability rules (§3 recs. 3–4). | Q3 | M | Medium | The current HMM agrees with the GMM on 98% of windows, so it adds nothing. Sub-states belong within stable coarse states. |
| 7 | Two-level anomaly scores (window within bird and season; bird against leave-one-out population), with artefact concentration checks, blind visual review and synthetic injections (§5 recs. 1–3). | Q5 | S–M | Medium | With 31% identity, a population-level score flags new birds rather than unusual events. |
| 8 | One hybrid SimCLR + masked-reconstruction run, compared on probes, identity, Jaccard and ARI; plus a randomised-label probe sanity check (§1 recs. 1–2). | Q1 | M | Medium | The only literature-backed change to the objective. Kept only if stability rises without more identity. |

None of these actions has been started. Each needs Nill's approval.

---

## 8. References

Labels as above. "Opened" means that the URL given was fetched during this review.

**Self-supervised learning and evaluation**
- Cai Y, Feng R, Yu A, Guo B, Hong Z (2026). BenchHAR: Benchmarking self-supervised learning for generalizable sensor-based activity recognition. arXiv 2605.08296. https://arxiv.org/abs/2605.08296 [AB]
- Ek S, Presotto R, Civitarese G, Portet F, Lalanda P, Bettini C (2024). Comparing self-supervised learning techniques for wearable human activity recognition. arXiv 2404.15331. https://arxiv.org/html/2404.15331v1 [FT]
- Fortes Rey V, Nshimyimana D, Lukowicz P (2023). Don't freeze: finetune encoders for better self-supervised HAR. arXiv 2307.01168. https://arxiv.org/html/2307.01168v1 [FT]
- Haresamudram H, Essa I, Plötz T (2022). Assessing the state of self-supervised human activity recognition using wearables. *Proc ACM IMWUT* 6(3):116. https://arxiv.org/abs/2202.12938 [AB; the conclusions were not visible in the abstract]
- Narayanswamy G et al. (2024). Scaling wearable foundation models. arXiv 2410.13638. https://arxiv.org/abs/2410.13638 [AB]
- Wilson O, Schoeman D, Bradley A, Clemente C (2025). Practical guidelines for validation of supervised machine learning models in accelerometer-based animal behaviour classification. *J Anim Ecol* 94(7):1322–1334. https://doi.org/10.1111/1365-2656.70054 (PMC12214441) [FT, first 100 k characters]

**Invariance and identity**
- Belrose N, Schneider-Joseph D, Ravfogel S, Cotterell R, Raff E, Biderman S (2023). LEACE: Perfect linear concept erasure in closed form. arXiv 2306.03819. https://arxiv.org/abs/2306.03819 [AB]
- Dehak N et al. (2011). *IEEE TASLP* (the title "Front-end factor analysis for speaker verification" is [RC]). https://sls.csail.mit.edu/publications/2011/Dehak_IEEE_May2011.pdf [AB, snippet; WCCN, LDA and NAP comparison]
- Elazar Y, Goldberg Y (2018). Adversarial removal of demographic attributes from text data. EMNLP 2018, pp. 11–21. https://arxiv.org/abs/1808.06640 [AB]
- Gulrajani I, Lopez-Paz D (2021). In search of lost domain generalization. ICLR 2021. https://arxiv.org/abs/2007.01434 [AB]
- Moyer D, Gao S, Brekelmans R, Galstyan A, Ver Steeg G (2018). Invariant representations without adversarial training. NeurIPS 31. https://arxiv.org/abs/1805.09458 [AB, search snippet]
- Ravfogel S, Elazar Y, Gonen H, Twiton M, Goldberg Y (2020). Null it out: guarding protected attributes by iterative nullspace projection. ACL 2020. https://arxiv.org/abs/2004.07667 [AB, search snippet; the author list is RC]
- Rosenfeld E, Ravikumar P, Risteski A (2021). The risks of invariant risk minimization. ICLR 2021. https://arxiv.org/abs/2010.05761 [AB, search snippet]

**Biology of individual, age and sex differences**
- Brønnvik H, Flack A (2025). Experience and motivation shape the flight performance of white storks migrating long-distance. *Curr Biol* 35(15):3736–3744.e2. https://doi.org/10.1016/j.cub.2025.06.044 [AB, repository record]
- Cecere JG, De Pascalis F, Imperio S, Ménard D, Catoni C, Griggio M, Rubolini D (2020). Inter-individual differences in foraging tactics of a colonial raptor: consistency, weather effects, and fitness correlates. *Movement Ecology*. https://doi.org/10.1186/s40462-020-00206-w [AB, PDF metadata and keywords only; the body was not readable]
- Clay TA et al. (2020). Sex-specific effects of wind on the flight decisions of a sexually dimorphic soaring bird. *J Anim Ecol*. https://nora.nerc.ac.uk/id/eprint/525481 [AB]
- Flack A, Nagy M, Fiedler W, Couzin ID, Wikelski M (2018). From local collective behavior to global migratory patterns in white storks. *Science* 360:911–914. https://doi.org/10.1126/science.aap7781 [AB]
- García-Jiménez R, Pérez-García JM, Margalida A (2018). Drivers of daily movement patterns affecting an endangered vulture flight activity. *BMC Ecol* 18:39. https://doi.org/10.1186/s12898-018-0195-7 [FT]
- Harel R, Horvitz N, Nathan R (2016). Adult vultures outperform juveniles in challenging thermal soaring conditions. *Sci Rep* 6:27865. https://doi.org/10.1038/srep27865 [FT]
- Hatzl JS, Tschumi M, Bassi E, Cruickshank SS, Jenny D, Pellissier L, Kamran S, Scharf AK, Sumasgutner P, Grüebler MU (2026). Supplementary material for "The consequences of food and flight efficiency for the timing of natal dispersal in golden eagles". Zenodo. https://doi.org/10.5281/zenodo.19002502 [AB; the journal version was not found; possibly related preprint: https://doi.org/10.1101/2025.04.21.649390, not opened]
- Poessel SA et al. (2016). Age- and season-specific variation in local and long-distance movement behavior of golden eagles. *Eur J Wildl Res*. https://pubs.usgs.gov/publication/70170070 [AB]

**Demographic and breeding inference**
- Fayzur D (2018). Ensemble learning applied to classify GPS trajectories of birds into male or female. arXiv 1808.08613. https://arxiv.org/abs/1808.08613 [AB]
- Picardi S et al. (2020). Analysis of movement recursions to detect reproductive events and estimate their fate in central place foragers. *Movement Ecology* 8. https://pubs.usgs.gov/publication/70211689 [AB]

**State discovery**
- Adam T, Griffiths CA, Leos-Barajas V et al. (2019). Joint modelling of multi-scale animal movement data using hierarchical hidden Markov models. *Methods Ecol Evol* 10(9):1536–1550. https://eprints.whiterose.ac.uk/148171 [AB; the PDF was not readable]
- Tibshirani R, Walther G (2005). Cluster validation by prediction strength. *J Comput Graph Stat* 14:511–528. Documentation: https://search.r-project.org/CRAN/refmans/fpc/html/prediction.strength.html [FT of the documentation, not of the paper]
- Weinreb C et al. (2024). Keypoint-MoSeq: parsing behavior by linking point tracking to pose dynamics. *Nat Methods*. https://doi.org/10.1038/s41592-024-02318-2 [AB]
- HSMMs for accelerometry: https://ar5iv.labs.arxiv.org/html/2006.09061; PMC6326431; PMC8032701 [AB, snippets]
- Unsupervised HMMs on albatross accelerometry: https://doi.org/10.1186/s40462-021-00243-z [AB, snippet; authors not read]

**Anomalies**
- Rast W, Portas R, Shatumbu GI, Berger A, Cloete C, Curk T, Götz T, Aschenborn O, Melzheimer J (2024). Death detector: using vultures as sentinels to detect carcasses by combining bio-logging and machine learning. *J Appl Ecol* 61(12):2936–2945. https://doi.org/10.1111/1365-2664.14810 [AB, press release https://www.eurekalert.org/news-releases/1065419 opened]
- Sehwag V, Chiang M, Mittal P (2021). SSD: A unified framework for self-supervised outlier detection. ICLR 2021. https://arxiv.org/abs/2103.12051 [AB]
- Tanigaki K, Otsuka R, Li A, Hatano Y, Wei Y, Koyama S, Yoda K, Maekawa T (2024). Automatic recording of rare behaviors of wild animals using video bio-loggers with on-board light-weight outlier detector. *PNAS Nexus* 3(1):pgad447. https://doi.org/10.1093/pnasnexus/pgad447 [FT]

**Sampling and seasonal models**
- McClintock BT, Michelot T (2018). momentuHMM. arXiv 1710.03786. https://arxiv.org/abs/1710.03786 [AB]
- Pedersen EJ, Miller DL, Simpson GL, Ross N (2019). Hierarchical generalized additive models in ecology: an introduction with mgcv. *PeerJ* 7:e6876. https://doi.org/10.7717/peerj.6876 [AB; PeerJ and PMC pages blocked]
- Schoombie S, Wilson RP, Ropert-Coudert Y, Dilley BJ, Ryan PG (2024). The efficiency of detecting seabird behaviour from movement patterns: the effect of sampling frequency on inferring movement metrics in Procellariiformes. *Movement Ecology* 12:59. https://doi.org/10.1186/s40462-024-00499-1 [FT, first 100 k characters]

## 9. Gaps and items not verified

- **No raptor sex or age classifier** with an individual-level accuracy was found. The expected accuracies in §4 are reasoning, not published numbers.
- **No animal-data comparison** of SimCLR against masked or hybrid SSL was found. All method-ranking evidence is from human HAR.
- **Springer pages were blocked:**
  - Cecere 2020: the repeatability values were not read.
  - The albatross HMM paper of 2021: authors and accuracy were not read.
  - A cinereous-vulture sex and age study: not cited.
- **Other unread items:**
  - The Vaadia et al. 2025 griffon accelerometer paper (*J Appl Ecol*, https://doi.org/10.1111/1365-2664.70180): only the DOI was readable, so it is not used.
  - The Rast et al. precision and recall figures: from snippets only.
- **Recalled and not checked:** the `mgcv` cyclic spline (`bs = "cc"`) detail.
