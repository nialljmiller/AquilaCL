# Golden Eagle analysis inventory (reference/ code, read 2026-11-24)

Source: `/Users/njm/golden_eagle/reference/` (read only). Line numbers refer to those files.
**AE** = `analyse_eagle_embeddings.py`, **AA** = `analyse_eagle_age.py`, **PS** = `plot_eagle_simclr.py`,
**ES** = `eagle_simclr.py`, **WC** = `build_window_catalogue.py`, **GP** = `build_gp_features.py`, **BC** = `build_bird_catalogue.py`.
"Viewed" = the saved PNG in `reference/logs/main/` was opened and checked against the code.

Facts checked on disk: `GeldenEAI_meta.csv` has 69 birds (31 m, 30 f, 8 NaN sex); deploy months June 12, July 13,
Sept 6, Oct 32; "nestling" in 16 comments, "fledgling" in 1, "migrant" in 21. `GoldenEAI_sample.csv` (200 random
rows): 36 % `gps`, 37 % `acceleration`, 15 % `orientation`, 12 % `magnetometer`; 66.5 % of rows lack lon/lat; only
`gps` rows carry lon/lat/speed/heading/height; `eobs:activity` empty in all 200 rows. `eagle_SimCLR_6526.log`:
144,219 windows, resumed at step 20,000, last step 31,890, loss flat ~3.8 (`loss.csv` 3.54–4.45; chance for
B=2048 is ln 4095 = 8.3). Saved analysis plots: step 30,000, **z-space**, k=8, UMAP subsample 20,000 (from titles),
except `age_diversity.png` and `age_heatmap.png`, titled **step 20,000** = stale.

## 0. Cross-cutting problems (X1–X14)

- **X1 Non-GPS sensor rows enter everything.** ES sets missing lon/lat/speed/heading/alt to 0 (ES 132–145), then
  differences lon/lat (ES 148–149): every GPS↔non-GPS transition is a jump of ~108° lon / ~44° lat; in the sample
  66.5 % of rows have no position. WC counts all rows as `n_fixes` and computes cadence over all rows (WC 125–145).
- **X2 Embeddings are stochastic.** `__getitem__` always augments (ES 223–227); `infer` embeds augmented `view0`
  (ES 614–617): noise, crop, adjacent swaps and a zeroed channel (each p=0.5) are in the analysed vectors.
- **X3 Per-window z-scoring removes absolute level** of speed, altitude, displacement (ES 156–168); no Δt channel.
  Interpretations of clusters by speed/height level are not supported by the representation.
- **X4 Default space z** (AE 1791, AA 1032): projection output trained with cosine similarity, analysed with
  Euclidean k-means/PCA/UMAP/IsolationForest without L2 normalisation.
- **X5 UTC hour as time of day** (WC 121, AE 186, AA 273). Site is UTC−6/−7; "night" shading 19–24 and 0–5 UTC
  (AE 1025–1026) covers local daytime. No solar time or day length.
- **X6 Pseudo-replication.** ~144 k windows from 63 birds compared as independent; no per-bird weighting, no tests.
- **X7 Bird-level values broadcast to windows** (AE 221–236, 586–597): violins/colours use constant per-bird means.
- **X8 Composition confound** in mean-PC-per-bin curves (AE 942, 994; AA 666, 712): which birds/years/cohorts are
  tracked in each bin; DOY bins pool 2024–2026. UTC hours 4–10 give flat PC values in `temporal_diel.png`
  (viewed), consistent with near-identical night windows (cause unverified).
- **X9 k-means labels arbitrary and refit per run** (AE 335, AA 334): stale step-20k heatmap has the dominant
  cluster as C2, step-30k plots have it as C1.
- **X10 Exceptions swallowed** in `run_plots` (AE 1760, AA 1012) plus `warnings.filterwarnings("ignore")`
  (AE 91, AA 66, ES 65, GP 56, PS 29; WC 26 RuntimeWarning): failed plots leave stale PNGs silently.
- **X11 Colour maps.** 63 birds in `tab20` (AE 494, ES 459): colours repeat ~3×. Season colours differ between
  global plots (Set2) and per-bird plots (SEASON_COLOURS, AE 122).
- **X12 Paths.** `age_classification_scores.csv` goes to logdir not outdir (AA 250); launch scripts hardcode
  relative paths and activate conda twice (launch_* 27–32); PS docstring hardcodes a cluster path (PS 9).
- **X13 Thresholds unvalidated** (WC 188, 221–222, 252–253, 291–293, 416–418; AE 1428–1464; AA 104/1037).
- **X14 Mortality / stationary tags not flagged.** Several birds have bird-mean speed 0.1–0.3 m/s and 0 active
  windows (`sex_features.png`, `age_features.png`, viewed).

## 1. Every output

| Output | Producer | Question | Inputs | Method / parameters | Bugs, confounds, questionable choices |
|---|---|---|---|---|---|
| `metadata_with_jd.csv` | BC::process_metadata 45–53 | metadata + JD | meta CSV | `<col>_jd` for every column whose name contains "time"/"date" (33–42) | name heuristic; zone of deploy dates assumed UTC |
| `per_bird_csv/<id>.csv` | BC::split_main_csv_by_bird_fixed_schema 56–118 | one file per bird | raw CSV | 200 k chunks, append | rerun appends duplicates (105–111); all sensor types kept (X1) |
| `bird_catalogue.csv` | BC::summarize_bird_csv 139–234; build_catalogue 237–265 | bird-level summary | per-bird CSV, meta | duration, timestep stats, path km, speed, heading change, height | timestep stats over all sensor rows (165–171); heading change across rows incl. NaN (219–226) |
| `window_catalogue_{h}h.csv` | WC::process_bird 340–377; extract_window_features 113–333 | hand-crafted window features | per-bird CSV | UTC floor, groups ≥2 rows | see §2; whole file loaded despite chunking (351–362); `pending`, `numeric` unused (348, 392); StochiStats columns only if installed (31–42) |
| `bird_catalogue_enhanced.csv` | WC::summarize_bird_from_windows 384–430; merge 518–525 | bird summaries for colouring | windows + bird_catalogue | means/stds, diel amplitude, regime fractions | see §2; UTC diel (X5); DOY range ignores multi-year (421–424) |
| `gp_features_rank####.csv`, `gp_features.csv`, `window_catalogue_with_gp.csv` | GP::process_bird_gp 321–421; fit_gp_batch 186–258; main/_flush 428–577 | GP hyperparameters per window | per-bird CSV, window catalogue | 4 kernels × 3 channels, Adam lr 0.1, 150 it, batch 256 (512 in launch) | see §2; **not used by any analysis**; shard append misaligns columns (571–576); rerun duplicates |
| `{step}_embeddings.csv` | ES::Trainer.infer 597–639 | latent vectors | per-bird CSV | h (256) + z (64), batch 512 | X1, X2, X3; windows ≥3 rows of any sensor (129, 217) |
| `{step}_embeddings.png` (10k, 30k viewed) | ES::plot_embeddings 447–476 | training-time look at h | h | PCA + UMAP(15, 0.1, seed 42) on 10 k random windows | unseeded `np.random.choice` (643); tab20 (X11); UMAP = many tiny islands (near-duplicate vectors likely, unverified) |
| `loss.csv`, `hyperparameters.txt`, `{step}_checkpoint.pt` | ES::Trainer 529–591; main 709–712 | training record | – | loss per step | `loss.csv` truncated at every start incl. resume (530–531): pre-20k history lost; hyperparameters overwritten |
| `loss_curve.png` (viewed) | AE::plot_loss 435–457 | did training converge? | loss.csv | rolling mean len/100 | only 20k–31.9k; flat ~3.8; no LR/alignment/uniformity/probes |
| `global_pca.png` (viewed) | AE::plot_global_pca 462–467 → _global_2d_grid 480–527 | latent structure | z, meta | PCA on 20 k subsample (seed 42); panels bird, DOY, UTC hour, season, bird mean speed, bird tortuosity | X4, X5, X7, X11; `vmin or percentile` ignores vmin=0 (409) |
| `global_umap.png` (viewed) | AE::plot_global_umap 472–477 | same, nonlinear | z | UMAP n_neighbors 30, min_dist 0.05, seed 42 | as above; dozens of isolated islands |
| `cluster_umap.png` (viewed) | AE::plot_cluster_umap 532–566 | number/size of states | z, k-means | KMeans k=8, n_init 10, seed 42, full N | no model selection/stability; C1 78,219 (54 %), C6 30,051; X4, X9 |
| `cluster_profiles_{movement,activity,temporal}.png` (viewed) | AE::plot_cluster_profiles 571–633 | meaning of clusters | labels + bird-level means + DOY/UTC hour | violins per cluster | X7; violin positions misalign if a cluster has ≤1 value (618–619); X5 |
| `cluster_heatmap.png` (viewed) | AE::plot_cluster_heatmap 638–675 | cluster = bird identity? | labels | per-bird fractions, annotate ≥0.30 | C1 vs C6 split follows bird groups (0749-/0829-047xx mostly C6); tag/cadence cause possible, unverified |
| `bird_timelines.png` (viewed) | AE::plot_bird_timelines 680–724 | state sequences | labels | top 16 birds by windows | relative time axis; gaps unmarked |
| `bird_profiles.png` (viewed) | AE::plot_bird_profiles 729–766 | per-bird state budget | labels | stacked fractions, top 16 | window counts depend on cadence |
| `pop_diversity.png` (viewed) | AE::plot_pop_diversity 771–821 | bird variability | z | mean of per-dim std | confounded with track length/coverage; X4 |
| `pop_sex.png` (viewed) | AE::plot_pop_sex 826–861 | sex separation | UMAP, sex | m/f highlight | duplicate of `sex_umap.png`; no test |
| `anomaly.png` (viewed) | AE::plot_anomaly 866–920 | unusual windows/birds | z full N | IsolationForest contamination 0.05, seed 42; red if > median+1 sd | 5 % outliers by construction; outliers sit in the dense central cluster (viewed); no inspection |
| `temporal_doy.png` (viewed) | AE::plot_temporal_doy 925–974 | seasonal drift | z full N | separate full-N PCA (937); 10-day bins | X8; PCA differs from ctx.pca2 |
| `temporal_diel.png` (viewed) | AE::plot_temporal_diel 979–1029 | diel rhythm | z | mean PC per UTC hour | X5 (night shading wrong), X8 |
| `per_bird/bird_<id>.png` ×16 (one viewed) | AE::plot_individual_bird 1034–1162 | single-bird deep dive | UMAP, labels | top N (CLI 16; function default 12) | only subsampled windows shown (1069); "DOY 1–365" box meaningless for multi-year (1149–1158) |
| `sex_umap.png` (viewed) | AE::plot_sex_study A 1205–1225 | sex in latent space | UMAP, sex | m 8,422 / f 11,153 / unknown 425 windows | duplicate of pop_sex |
| `sex_features.png` (viewed) | AE::plot_sex_study B 1228–1278 | sex movement differences | bird-level features | one point per bird | no test; duration differs by sex (cohort confound); X14 |
| `sex_cluster_usage.png` (viewed) | AE C 1281–1305 | sex state use | labels | pooled windows | X6 |
| `sex_seasonal.png` (viewed) | AE D 1308–1339 | sex × season | z, third PCA fit (1310) | 10-day bins | X8 |
| `sex_diel.png` (viewed) | AE E 1342–1369 | sex × diel | z | UTC hour means | X5, X8 |
| `behaviour/behaviour_umap.png` (viewed) | AE::plot_behaviour_inference A 1524–1550; `_classify_window` 1400–1466 | where rule behaviours sit in latent space | window catalogue | stationary speed<1 & frac_low>0.7; diving vdesc<−3 & speed_max>15; thermal frac_climb>0.35 & hconc<0.55 & eff<0.5; gliding frac_desc>0.4 & frac_straight>0.35 & speed>4; flying speed>7 & hconc>0.65 & eff>0.55; hunting 1.5<speed<10 & tort>2.5 & frac_sharp>0.15 | 69 % "unknown" (13,881/20,000); "diving" 1,378 driven by vrate bug (§2); device heading, not track bearing; whether any rows fail the left join (1486) is unverified |
| `behaviour/behaviour_per_bird.png` (viewed) | AE B 1553–1579 | behaviour budget per bird | as above | top 20 birds | **stacking bug**: heights in %, `bottom += vals` in fractions (1569–1572); plot shows only "unknown" |
| `behaviour/behaviour_timelines.png` (viewed) | AE C 1582–1605 | when behaviours occur | as above | top 12 | dominated by "unknown" |
| `behaviour/behaviour_feature_validation.png` (viewed) | AE D 1608–1646 | "validate" rules | features defining the labels | violins | **circular**; axes show vrate to −80,000 m/s and tortuosity 30,000 |
| `behaviour/behaviour_seasonal.png`, `behaviour_diel.png` (viewed) | AE E 1649–1680 | seasonal/diel budget | as above | 10-day / UTC-hour bins | X5; UTC 4–10 all "unknown"; alternating even/odd UTC hours (duty-cycle cause unverified) |
| `behaviour/behaviour_cluster_matrix.png` (viewed) | AE F 1683–1706 | clusters vs behaviours | labels + rule labels | row-normalised | rule labels unvalidated; assumes merge kept row order/count (1684) |
| `age_classification_scores.csv` | AA::load_data 250–256 | inspect age split | catalogue | territorial_score table | in logdir (X12) |
| age labels for all AA plots | AA::classify_age 103–195 | juvenile vs adult | comments, lon/lat extent, duration | "nestling" → young; duration<30 d → uncertain; else 1/(√(Δlon²+Δlat²)/√days) ≥ 0.75 → adult | docstring (7–13) describes a different rule; degrees as distance; **circular**; "fledgling" not matched; hatch year = deploy year even for movement-inferred young (181–185); current CSV 34 young / 27 adult / 2 uncertain |
| `age_umap.png` (viewed) | AA::plot_age_umap 393–448 | age in latent space; development | UMAP 30/0.05, age_months | age_months = (window − 1 July deploy year)/30.44, ≥0 (290–303) | wrong for non-nestling "young"; assumed hatch date |
| `age_features.png` (viewed) | AA::plot_age_features 451–511 | movement by age | bird-level features | one point per bird | circular; duration young ≈440 d vs adult ≈150 d = cohort, not age |
| `age_clusters.png` (viewed) | AA::plot_age_clusters 514–570 | state use by age | labels | pooled | X6, X9 |
| `age_trajectory.png` (viewed) | AA::plot_age_trajectory 573–646 | do juveniles converge on adults? | full PCA on z | bins 0–3/3–6/6–12/12–18/18+ months | bin means within ±0.1 of a ±0.6 spread; age collinear with DOY in one cohort |
| `age_diel.png` (viewed) | AA::plot_age_diel 649–690 | diel by age | full PCA | UTC means | X5, X8 |
| `age_seasonal.png` (viewed) | AA::plot_age_seasonal 693–732 | season by age | full PCA | 10-day bins | X8; adults tracked only from Sept/Oct |
| `age_diversity.png` (viewed, **stale step 20k**) | AA::plot_age_diversity 735–766 | variability by age | z | mean per-dim std | current code KeyError at 762 (`AGE_COLOURS["unknown"]` missing, 86–90); swallowed (X10) |
| `age_heatmap.png` (viewed, **stale step 20k**) | AA::plot_age_heatmap 769–819 | bird × cluster by age | labels | young/adult/"unknown" order | "uncertain" birds dropped (784); KeyError at 815; stale file has 25 young vs 34 now |
| `age_features_window.png` (viewed) | AA::plot_age_features_win 822–879 | window features by age | window catalogue (inner join) | violins | X6; climb/descent to ±80,000 m/s |
| `age_behaviour.png` (viewed) | AA::plot_age_behaviour 882–978 | behaviour by age; juvenile year | copied classifier 900–927 | pooled; 10-day bins young only | duplicated code; inner join here vs left join in AE (differs only if keys fail to match, unverified) |
| `training_curves.png`, `{step}_{PCA,tSNE,UMAP}_{hs,zs}.png`, `{step}_*_from3d.png` | PS::plot_training_curves 210–268; run_reductions 308–344; plot_precomputed_3d 347–375 | old diagnostic viewer | `output_info.csv`/`loss.txt`, `*_hs.npy`, `hs_labels.csv`, `*_hzs_3D.csv` | PCA; tSNE 5 k (unseeded 331); UMAP full N | **dead**: ES writes none of these; `_add_bird_colour` uses `label % n_birds` (198–201) = arbitrary colours |
| sampled CSV | csv_sample.py::sample_csv_streaming 9–70 | sample for inspection | big CSV | reservoir, first 5 kept, seed 42 | fine |

## 2. Derived features and errors

**WC::extract_window_features** (w = all rows in the UTC-floored window)
- Time: `window_hour_utc`, `window_doy` (121–122); `n_fixes` = all rows (126, X1); `n_valid_timestamps`.
- Cadence (133–145): duration = tmax−tmin; `timestep_{median,max,mean,std}_s`; burstiness = (σ−μ)/(σ+μ);
  `frac_large_gaps` = mean(dt > 2·median). Error: mixes GPS fixes with ACC/MAG/orientation events.
- Location (165–172): mean/std/range of lon, lat (degrees); `bbox_area_deg2` = lon_range·lat_range
  (**degrees² used as area**); `radius_of_gyration_km` = √mean(haversine to arithmetic-mean centroid)².
- Path (175–197): `net_displacement_km` first→last fix; `step_km` haversine; `total_path_km` = Σ;
  `path_efficiency` = net/total; step mean/median/std/max/q75/q95; `frac_zero_steps` (step < 10 m, near GPS noise);
  `step_accel_mean/max` = |Δstep_km| (no dt, not an acceleration). Steps span any gap.
- Speed (210–237): stats of device `ground-speed`; `frac_low_speed` <1 m/s, `frac_high_speed` >10 m/s.
  StochiStats (223–231): Stetson K with fake errors 0.1, Eta with t = index, MAD, IQR, skew, kurtosis.
- Heading (240–269): circular mean; circular std √(−2 ln R) (inf at R=0); `turn_*` = |wrapped Δ device heading|
  (not track bearing; any gap; noise when stationary); `frac_sharp_turns` >90°, `frac_straight` <15°;
  `heading_concentration` = R; `tortuosity` = total/net (redundant with 1/efficiency; only computed inside heading
  block, 259; explodes as net→0, 30,000 seen).
- Altitude (272–309): stats of height above **ellipsoid**; `vrate` = Δalt / **`timestep_median_s` of all rows**
  (283–286), not the Δt of each altitude pair → ±80,000 m/s in plots; `vrate_{mean,std,max_climb,max_descent}`,
  `frac_climbing` >0.1, `frac_descending` <−0.1, `frac_level` inherit it. StochiStats `alt_eta`, `alt_mad`.
- Quality/device (312–331): dop, satellites, horizontal accuracy, device/external temperature, battery; never used
  to filter. No filtering on `sensor-type`, `import-marked-outlier`, `visible` anywhere.

**WC::summarize_bird_from_windows**: `bird_<f>_mean/_std` for 22 features (395–403), unweighted; diel amplitude
and peak hour by **UTC hour** (406–412); stationary/active/fast window fractions at speed_mean <0.5, ≥2, ≥8
(416–418); DOY range/first/last (wrong for multi-year); `bird_total_path_km` = Σ window paths.

**GP** (channels speed/altitude/activity × RBF, Matérn 5/2, periodic, RBF×periodic): windows with ≥3 rows and
span ≥60 s (282–295); t normalised to [0,1] by first–last row (295); y z-scored per window (308–314). Columns
`gp_<ch>_<kernel>_{lengthscale,outputscale,noise,lml}` (+ `period`; quasiperiodic `rbf_lengthscale,
per_lengthscale, period`). Errors: (a) lengthscale/period in units of each window's own span, not comparable;
(b) a 1-hour window cannot measure a diurnal period although the docstring says so (38–41); (c) z-scoring fixes
outputscale+noise ≈ 1; (d) `lml` is per-point normalised; (e) any exception → NaN batch silently (247–258);
(f) fixed 150 Adam steps, no convergence check, no seed; (g) batching by identical N → many size-1 batches;
(h) activity channel empty in sample; (i) `_flush` appends frames with differing columns under the first header
(571–576); (j) LOCAL_RANK fallback `rank % world_size` (80) is wrong on multiple nodes.

**ES input channels**: speed, Δlon, Δlat, activity, sin/cos heading, altitude; NaN→0 before Δ (X1); Δlon in
degrees without cos lat; z-scored per window except sin/cos (X3).

## 3. CLI options and config knobs

- **BC**: `main_csv`, `meta_csv`; `--outdir` (golden_eai_outputs), `--chunksize` (200000).
- **WC**: `--per_bird_dir`, `--bird_catalogue`, `--outdir`, `--window_hours` (1), `--chunksize` (200000).
  Hidden: speed 1/10 m/s, turn 90°/15°, vrate ±0.1, zero step 0.01 km, regimes 0.5/2/8 m/s, StochiStats errors 0.1/1.0.
- **GP**: `--per_bird_dir`, `--window_catalogue`, `--outdir`, `--window_hours` (1), `--n_iter` (150), `--batch_size`
  (256), `--merge`; env RANK/SLURM_PROCID, WORLD_SIZE/SLURM_NTASKS, LOCAL_RANK. Hidden: lr 0.1, clip 2.0,
  min span 60 s, min 3 points, flush every 5000 rows.
- **ES**: `--per_bird_dir`, `--logdir` (logs/simclr), `--train`, `--milestone` (0), `--batch_size` (2048), `--lr`
  (3e-4), `--tau` (0.07), `--hidden_dim` (128), `--n_layers` (4), `--out_dim` (64), `--drop_prob` (0.2),
  `--train_steps` (200000; launch 2,000,000), `--save_every` (5000; launch 1000), `--infer_every` (10000),
  `--num_workers` (8), `--window_hours` (1). Hidden: BACKBONE gru, INPUT_DIM 7, Augmenter(time_jitter 0.05,
  crop_min 0.6, channel_drop_p 0.2 unused, noise 0.05, p_each 0.5), cosine eta_min 1e-6, clip 1.0, infer batch 512,
  plot subsample 10 000, UMAP 15/0.1.
- **PS**: `--logdir` (required), `--catalogue`, `--outdir`, `--step`, `--no_umap`. Hidden: tSNE n 5000.
- **AE**: `--logdir` (required), `--catalogue`, `--window_catalogue`, `--outdir` (logdir/analysis), `--step`,
  `--n_clusters` (8), `--n_plot_birds` (16), `--umap_sample` (20000), `--space` z|h (z), `--skip`, `--only` (IDs:
  loss, global_pca, global_umap, cluster_umap, cluster_profiles, cluster_heatmap, bird_timelines, bird_profiles,
  pop_diversity, pop_sex, anomaly, temporal_doy, temporal_diel, individual_bird, sex_study, behaviour). Hidden:
  thread caps 16 (75–78), UMAP 30/0.05, IsolationForest 0.05, DOY bin 10, behaviour thresholds, top 20/12 birds,
  annotation cut-offs 0.30/0.15; `hdbscan` imported, never used (111–115).
- **AA**: `--logdir`, `--catalogue` (required), `--outdir` (logdir/age_analysis), `--step`, `--space`, `--n_clusters`
  (8), `--umap_sample` (20000), `--window_catalogue`, `--territorial_threshold` (0.75), `--skip`, `--only`.
  Hidden: short track 30 d, hatch 1 July, age bins 0/3/6/12/18 months.
- **csv_sample**: `input_csv`, `output_csv`, `--rows` (200), `--keep-first` (5), `--seed` (42).
- **Launch scripts**: SLURM ai4wy, 1 node, 2× GH200, 64 CPUs, 1 day; `MILESTONE=20000`.

## 4. Proposed clean analysis suite (every current question kept)

Principles: one package and one YAML config saved with each run; run id, step, space and seed in every output's
metadata; no blanket exception catching (fail loudly and write a run report listing skipped items); deterministic
embeddings; bird-level statistics or mixed models with bird as a random effect, with uncertainty; all thresholds
in config; one shared join, one behaviour classifier, one colour scheme scaling to 63+ birds.

- **M0 Data preparation** (new inputs: sensor-type filter, GPS quality filter, local solar time). GPS-only fix
  table (drop `import-marked-outlier`, filter DOP/accuracy/satellites) plus a separate ACC table. Per-fix Δt,
  step distance and bearing from positions; vertical rate per altitude pair; local solar time, sunrise/sunset,
  day length, local date, year, cohort, days since deployment; mortality/tag-off detection. Optional new input:
  DEM for height above ground. Windows in local solar time with a minimum number of GPS fixes and a coverage
  fraction; one shared window index for catalogue, GP and SimCLR.
- **M1 Training diagnostics** (loss_curve, embeddings png). Full loss history (append, never truncate), LR,
  alignment/uniformity, embedding rank and per-dimension variance (collapse check), kNN/linear probes for speed
  level, altitude, solar hour and bird. Requires deterministic inference without augmentation.
- **M2 Latent-space overview** (global_pca/umap, PS). PCA explained variance; UMAP with documented parameters on
  a per-bird-capped subsample; colouring by window-level features, solar hour, DOY and cohort; variance
  partitioning and probes (bird vs time vs movement). Run in h (default) and in L2-normalised z.
- **M3 State discovery** (cluster_umap). Choose k by silhouette/BIC and bootstrap stability (ARI) over a range;
  GMM/HDBSCAN alternatives; match labels across runs (Hungarian) so cluster numbers stay stable; check for
  duplicate or degenerate windows before clustering.
- **M4 State characterisation** (cluster_profiles, behaviour_*). Corrected window features per state, example
  tracks, transition matrices and dwell times. Behaviour labels from an independent source: accelerometer-based
  classification (ACC bursts exist in the data) or manually annotated windows, then a states × labels confusion
  matrix. Rule labels kept only as a baseline, evaluated against the independent labels.
- **M5 Temporal: diel and seasonal** (temporal_*, behaviour_seasonal/diel, *_diel, *_seasonal). State use and
  latent position vs local solar hour (and relative to sunrise/sunset) and vs DOY, per bird then pooled, or
  GAM/mixed model with cyclic smooths and a bird random effect; years shown separately; a coverage panel (birds
  and windows per bin) under each curve.
- **M6 Individual, sex, age** (cluster_heatmap, bird_*, per_bird, pop_*, sex_*, age_*). Per-bird budgets and
  calendar timelines with gaps marked; variability normalised for coverage (bootstrap with equal windows per
  bird and season); sex and age tests at bird level (permutation tests or mixed models) adjusted for cohort,
  coverage and duration. Age from metadata only (nestling/fledgling comment or June–July nest deployment →
  hatch year; others unknown unless plumage age is supplied). Territorial score reported as an outcome, never
  as the label. Juvenile development modelled on age with DOY as a covariate. New inputs: confirmed age or
  hatch year, cohort, mortality flags.
- **M7 Anomaly** (anomaly.png). kNN-distance or IsolationForest scores without a fixed contamination rate,
  computed within bird and season; top-N windows shown as raw tracks; data-quality anomalies (GPS errors, gaps,
  mortality) separated from behavioural ones.
- **M8 Feature pipelines** (window catalogue, GP). Rebuilt on M0 tables. GP on real time in seconds with bounded
  priors, raw-scale amplitude, and multi-day windows if diel periodicity is the target; schema fixed before
  writing; seeds; convergence flags. These features serve as probe targets in M1–M4.
