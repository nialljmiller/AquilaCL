# Fable supervisor review, 2026-11-25

REVIEW of budgets.py, flight.py, residency.py, anomaly.py (+ tests, slurm 08-11) and results (budgets_r6, budgets_r6_v2, residency_r6, residency_r6_v2, flight_r6_r4, anomaly_v1). Repo /Users/njm/golden_eagle/repo.

How it was done: I read budgets.py in full and all result sections myself; flight, residency and anomaly were each read in full by a forked reviewer, and I cross-checked their result claims against the summaries. Tests run: test_budgets 15 passed, test_flight 9 passed, test_residency 21 passed, test_anomaly 16 passed. Nothing in the repo was edited. "Run" = reproduced with a synthetic snippet; "read" = from code reading or arithmetic on the summaries; nothing was checked on the real data. anomaly_v2 has no results yet; findings on it are about the code.

OVERALL: the descriptive parts (time-budget estimator v2, residency classes, K=1 bout result) are close to reportable. None of the group comparisons (migrant vs resident, young vs other, most sex results) is reportable as tested yet, because of design confounds, not code bugs. Two code bugs give wrong numbers silently (residency p floor, anomaly day score).

== CRITICAL ==

C1. Budgets: the "migrant" test and the "young" test in cohort 2024 are one contrast (read, from the group table). Cohort 2024 has 35 birds: 20 migrant-flagged, 15 not flagged; 14 young-tagged, 21 not stated. If the flags are disjoint (inferred from the counts, not checked), the 15 "not flagged" are 14 young-tagged + 1, and the 21 "not stated" are the 20 migrants + 1. So +0.43 h (p 0.011) and -0.37 h (p 0.034) are the same comparison, "October-deployed migrants vs juveniles tagged in the nest", with age, residency and deploy month inseparable. The test also pools all 12 months, including migration months, where migrants fly more by definition. Fix: report it as one confounded contrast, not two findings; any residency test must use track-derived classes, same deployment year, non-migration months.

C2. Residency section 2 (behaviour, migrants vs residents) is not testable in this design (read). Nuisance floor AUC is 1.00 in every basin variant; h beyond nuisance is <= 0 everywhere (delta AUC -0.09 to +0.02, p 0.31-0.99). The cohort-2024 comparison has 0 adult residents. All 9 adult residents are `resident_partial`, cohort "2025", median 23 Nov-Mar days, all in November; migrants' "basin days" in the all-months variant are 43% October and 19% April, i.e. passage. The speed result (12.6 vs 10.7 m/s, p 0.0025) is therefore expected by construction (class defined by long movement; continuous mode is speed-triggered). Fix: label section 2 "not testable"; drop "TESTED" from those tables; lead with the nuisance AUC.

C3. residency.py:545,554 `perm_strat_diff` returns p = 1/(n_perm+1) = 0.0005 when no cohort x sex stratum holds both groups (run: migrants all female, residents all male gives (nan, 0.0005)). Several printed `p_perm_sex_strata` values are exactly 0.0005, one beside p_perm 0.0295; whether those are this bug is unverified. Fix: return NaN when the statistic is not finite; report birds in informative strata.

C4. anomaly.py:1046-1049 `day_pair_ed` samples the reference day with replacement, so the energy distance grows when the reference has few windows per hour (run: iid behaviour, 12 then 2 windows/hour; every top-10% day fell in the sparse period, 3 of 3 repeats). Schedule changes (300 s to 900 s), low battery and missed bursts will be flagged as unusual days. This affects the running anomaly_v2. Fix: sample without replacement, min(nQ, nY) per common hour; rerun the day level.

C5. Cohort label merges deployment years: budgets.py:952 and residency.py:800 set "2024" if the first window year is <= 2024, else "2025", so any 2026 deployment is "2025" (read; whether 2026 deployments exist is unverified, but residents with only Jul-Nov data suggest so). "Permuted within cohort" then does not compare same-year birds. Fix: use the real deployment year as stratum everywhere.

== IMPORTANT ==

Budgets (v2 estimator; all read)
I1. Bout time is imputed as 100% flight (budgets.py:35-38, 553-556), but 4.57% of k=1 windows are K=2 ground (1.9% at clock slots). The states of the k=1 windows are already loaded. Fix: classify bout time by its own windows' K=2/K=6 state instead of imputing; this also types the 0.45 h/day (22% of flight) that is now untyped "flight_imputed".
I2. The brief's fact "25 of 91,913 k=1 full windows are still" does not match the results: bouts total 8,661 h (about 3.1 M windows), flight.py counts 2,868,664 continuous flight windows, and 4.57% of k=1 windows (about 137,000) are K=2 ground. 91,913 is probably a subsample. Fix: recount before either number is quoted.
I3. The seasonal curve mixes bird sets (32-80 birds per month). The July minimum (1.16 h vs 3.75 h in May) coincides with newly fledged young entering (14-19 young-tagged, deployed June-July) and migrants being far north. Fix: give the monthly curve for age-not-stated birds only, and per residency class.
I4. The "all months" headline 2.06 h is day-weighted per bird and over-weights October-November (80 and 75 birds). The mean of the 12 monthly means is 2.20 h. Fix: report the month-balanced value.
I5. Multiple testing: 3 distinct main tests, uncorrected. With Holm: migrant 0.034, sex 0.063, young 0.068. Sex moved from p 0.033 (v1) to 0.063 (v2) with the estimator change, so it is not a result.
I6. "Flapping" K=6 state is 0.44 h of 2.05 h flight (21%; 28% of scheduled flight samples), against about 2% published (PLAN correction 3), and scheduled samples have GPS in 0.9% of cases. flight.py results agree: state 5 is 26% of bout windows while the handcrafted flapping label is 6%. Fix: do not report K=6 flight-type hours as behaviour times; rename state 5.
I7. Night fill: under own-data filling, cohort 2024 has 9 young-tagged and 1 other bird and 0 migrants, so the population night fill (5.2 h/day) comes mostly from juveniles and cohort-2025 birds. Flight is unaffected (0.028 h/day); ground sub-state and K=6 ground hours are not group-comparable. The "12, own" row (20 birds, 1.55 h) is a different bird set, not a sensitivity test. Fix: compare own vs population fill on the same 20 birds.
I8. Unverified risk: if the tag's off window is fixed in UTC, it shifts by up to about 3 solar hours for migrants far north-west, and their evening daylight hours would be filled from the bird's all-month mean (winter, in the basin). Fix: print the direct-bin mask by solar hour for migrants in May-August.
I9. Scheduled samples are not filtered by window length (budgets.py:147-154). If 64-sample tails of continuous runs carry k=0, they enter as "scheduled" right after flight. 3.0% of gaps are shorter than the interval. Fix: report the share of scheduled samples with acc_samples < 190 and within 60 s after a bout end, and their state mix.

Flight
I10. The only Holm-significant sex result (PC1 in scheduled mode, g 0.80, Holm p 0.0225 r6 / 0.0075 r4) is on an axis that in that subset correlates with VeDBA -0.49, not with turn/climb as in continuous mode, where sex is not significant (Holm 0.11). Correlates are printed without n (flight.py:302, 846-848), and "var. expl." is the pooled value repeated for every subset. Fix: reword; print n; do not call it turning or climbing.
I11. Sex test has no confound control (flight.py:314-337): window-weighted bird means, labels permuted across all birds, although `perm_mean_diff` accepts strata. Fix: mean of bird-month means; permute within deployment year and age group; repeat with acc_samples >= 190.
I12. Bout climb and height lost assume 10 s of GPS per window (flight.py:517, 526-537; run: a 64-sample tail with 3 fixes at -3 m/s counts as -30 m, true about -9 m). Fix: use climb_mean x n_speed.
I13. "Circling" for all 46,939 bouts comes from gross absolute turning (flight.py:520, 541-543; run: straight flight with 5.6 deg/s jitter gives 0.84 circles/min gross, 0 net). Fix: use net circles or drop the word.
I14. Bird counts disagree: 46,939 bouts >= 60 s in 65 birds here, 53,584 in 80 birds in stage 6 section 2; 43,053 of 89,992 bouts are under 60 s. One cause is confirmed by running: a k=0 window inside a bout splits it (flight.py:133-135). Budgets uses a different bout definition (61,067 bouts, 30 s join). Fix: one bout definition for both modules; report bouts per bird under both.
I15. Birds missing from a states file are dropped silently (flight.py:201): r6 has 89 flight birds, r4 has 91. flight.py uses GMM `state` for K=2 (flight.py:913) while budgets uses `state_hmm`. Fix: log the drops; use one column.

Residency
I16. v2's 12 residents come from a relaxed definition (residency.py:367-368, "no spring coverage needed"); v1 had 3. Every adult resident has a track under 330 days. Fix: `resident_full` as the primary class; rename `resident_partial` "present summer-autumn, annual cycle unobserved".
I17. No deploy-off or mortality filter (residency.py:140-148; run: a stationary 200-day track is `resident_partial`). Fix: cut at deploy_off; flag tracks ending with no displacement and no flight.
I18. Migration dates across GPS gaps are the first fix after the gap (residency.py:342-350; run). This likely explains v2 return day-of-year 18-349 against 276-315 in v1. Fix: NaN when the gap at the crossing exceeds about 7 days.
I19. Scheduled shares use all k=0 windows (residency.py:445), including second windows of long bursts and short windows, with no hour or month matching; fast-glide shares of 0.001-0.006 are a handful of windows. Holm shrinks to testable rows (residency.py:498), so the kinematics p_holm equals p_perm (1 of 15 variables testable). Fix: first window per burst, daylight, K=2 only; Holm over the declared family.
I20. Thresholds decide classes: 1000 km north / 500 km return / 200 km (run: a bird summering 900 km north is "nomad" by rule and "migrant" by NSD). NSD needs only a 200 km, 30-day foray, which makes 7 young birds "migrants"; kappa is 0.50. Rule finds 13 unflagged migrants against the verified 11 of 55 (different origin: deployment centroid vs study centre). In the distance table, `resident_partial` has median winter distance 365 km but winter-basin share 1, and returning migrants have winter-basin share 0.01, which contradicts "migrants winter in the basin" for that class; neither is explained. Fix: class counts at 750/1000/1250 km from both origins; explain the winter columns.

Anomaly
I21. v1 results are not usable and should be marked superseded: injected-fault recall 1-6% at a 0.7-0.8% clean rate (r6 yaw90 0.0070, below the clean rate 0.0081); short windows are 81% (r4) and 94% (r6) of flags; `l2_score` correlates with track_days/n_windows at rho -0.36 to -0.40 and `l2_max_change` at +0.56; 1005-1074 of 15,289 bird-days pass a p < 0.001 screen against about 15 expected.
I22. v2 level 2 (anomaly.py:1680-1698): robust z in cells of 5 birds has a null 99th percentile of 10.7 (run), and the bird score averages 3 random cells, so a thin cell decides the rank. Fix: >= 20 birds per cell, within-cell ranks, median over shared cells.
I23. v2 flags are a fixed top 1% (FLAG_Q=99), with no window-level injection, so no recall exists for window flags. Fix: call them "top 1% ranked for inspection".
I24. v2 QC recall is partly tautological (injections set exactly the thresholded quantity), and ROOST_CP_DEG=30 (anomaly.py:125) was chosen after seeing the clean rate and the injected slips; a 30 degree slip is detected about 70% of the time, 15 degrees never (run). A 180 degree yaw is invisible to the axial statistic (run). Fix: disclose; give recall over 20-45 degrees.
I25. v2 "format neutrality met" is by construction: short windows (about 40% of scored windows in v1) are excluded from level 1. Fix: state the share of windows, birds and bird-days that get no behavioural score.
I26. anomaly.py:202 `np.nan_to_num` gives windows with missing local_solar_hour an offset of 0, moving them to another solar day (run); budgets.py fills from neighbours. Frequency in the data is unverified. Fix: same fill as budgets; count the NaNs.
I27. Nuisance table: about 108 uncorrected p-values, floor 1/2001 printed as 0.0005; partial version controls only track length and window count. Fix: Holm, "<= 0.0005", add deployment year and month, label descriptive.

== MINOR ==
M1. Budgets: "14 pre-fledge bird-days" against 139,823 pre-fledge windows; the windows are removed before days are formed, so the day count is only boundary days. Report the real number.
M2. Budgets: 89 birds (81/7/1 by interval) against 92 with windows (81/10/1); the 3 dropped birds are not named as a count in the summary. Flight prints "f 41, m 50, missing 3" (94), stage 6 has 45 males, the brief 48.
M3. Budgets: two "mean daily flight h" values (2.03 in the sensitivity table, 2.06 headline) from different bird filters; label them.
M4. Flight: Holm is per run (15 tests), 30 were run; n_boot = 20 for bout stability; cross-run comparison is window-weighted; flight.py:548 fills missing state shares with 0.
M5. Residency: daily position is the median of all fixes, so one 1 Hz bout outvotes a day of scheduled fixes (run); classifier runs with 4 birds per class.
M6. Anomaly: level2 raises KeyError when no cell has 5 birds (anomaly.py:1692-1694); month pairs with gaps > 1 enter the change score (1714-1733).
M7. Slurm: 08, 09 request 4 h and 11 requests 3 h against the 2 h target; all refuse a non-empty output directory and require their variables.
M8. p values at the permutation floor should be printed as "<= 1/(n_perm+1)" in all modules.

== PRIVACY ==
P1 (important). tests/test_residency.py:10 hard-codes a latitude/longitude pair that lies inside the study basin. The comment says it is planted and synthetic (I confirmed the line; I cannot confirm it is not a real site). Fix: use a point far from the region.
P2 (important). residency.csv gives per bird, at 1 km, both distance and northing from the study centre plus departure/return day-of-year; distance + northing fixes a position up to an east-west mirror. Fix: round to 50 km, keep one of the two.
P3 (minor). nsd_birds.png, qc_timelines/<bird>.png and bird-keyed CSVs carry bird ids; acceptable as runtime output, must not be committed.
No coordinates, UTC times next to positions or bird ids were found in stdout logs, summary.md files or module code. budgets birds.csv holds deploy and fledge dates but no positions; reference.py reads no coordinate columns.

== WHAT IS SOUND ==
- Budgets v2 estimator: scheduled-sample weighting, bout handling and solar-hour stratification are correct as read; hours add to 24 (15.48+6.46+0.36+0.08+0.72+0.44+0.45 = 23.99), bout hours match (8,661 h / 19,312 bird-days = 0.45 h), diel sum 1.99 h vs 2.06 h. The v1 value 3.13 h was inflated by scaling to 24 h and is correctly superseded; flight hours are stable across the 8/12/14-bin rules (2.01-2.04 h). The measured suppression of the schedule inside bouts (2.5% of slots found) justifies treating bout time separately.
- Every test in all four modules uses the bird as the unit with label permutation between birds; one permutation is applied across months in budgets.
- Flight: K=1 bout result under a rule fixed in advance; per-mode reporting; PC1-4 agree across runs (0.99/0.99/0.95/0.88); Holm arithmetic correct.
- Residency part 1: haversine and northing correct including the longitude wrap (run); counts reconcile to 94 GPS birds; v2 finds 15 of 20 flagged migrants, matching the verified fact.
- Anomaly v2 design: QC separated from behaviour; references exclude the scored unit; scoring within bird, mode and state; no positions read.

== MUST-FIX BEFORE RESULTS ARE REPORTED (ranked) ==
1. Withdraw residency section 2 as tested (C2) and report the budgets migrant/young tests as one confounded contrast (C1).
2. Fix the residency p floor bug (C3) and regenerate `p_perm_sex_strata`.
3. Fix the anomaly day-score sampling (C4) and rerun the day level; treat the running anomaly_v2 day results as invalid until then.
4. Use the real deployment year as cohort in budgets and residency (C5).
5. Mark budgets_r6 (v1), residency_r6 (v1) and anomaly_v1 as superseded (I21).
6. Budgets: classify bout time by observed state (I1); month-balanced headline and age-not-stated seasonal curve (I3, I4); Holm on the tests (I5); stop reporting K=6 "flapping" hours (I6).
7. Reconcile the counts: 91,913 vs about 3.0 M k=1 windows (I2); 65 vs 80 birds with bouts and one bout definition (I14); 89/91/92/94 birds and the male count (I15, M2).
8. Flight: reword the PC1 scheduled-mode sex result, add stratified permutation (I10, I11); fix climb and "circling" (I12, I13).
9. Residency part 1: deploy-off filter, gap-aware dates, `resident_partial` label, threshold sensitivity (I16-I18, I20).
10. Anomaly v2: level-2 cell size (I22), solar-hour NaN (I26), wording of flags, recall and neutrality (I23-I25), nuisance table (I27).
11. Privacy: replace the test coordinates and coarsen residency.csv (P1, P2).

Reportable now as descriptive, with the stated caveats: daily flight about 2.0-2.2 h (plain 1.82 h, with bouts 2.06 h), the diel profile, the window-share vs time-share bias table, the K=1 bout result, the cross-run axis agreement, and the residency class counts against the migrant flag.