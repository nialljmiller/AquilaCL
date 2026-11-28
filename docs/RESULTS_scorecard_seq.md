# Scorecard: level-2 sequence model sc_seq vs sc_base (2026-11-28)

Spec `docs/SCORECARD_spec.md` v2.1, same split and frozen injection list as `RESULTS_scorecard_baseline.md`.

**Design.** sc_seq_s<n> is a masked-token transformer over the time-ordered h stream of parent run sc_base_s<n>. It uses 512-window crops, rotary position only, no time encoding and no environment input. Five seeds were trained (jobs 62637–62641).

**Scoring.**
- First scoring (62642–62646, `scorecard/sc_seq_s<n>`): rows G6, G7a–d and G9b failed. `seq.contextual_embed` loaded the checkpoint on the GPU and then read it with `.numpy()`. Fixed in a0ef903.
- Full rescoring (62654–62658, `scorecard/sc_seq_s<n>_v2`): no errors.

## Selection rows (mean over 5 seeds; rule from `scripts/compare_scorecards.py`)

| Row | Goal | sc_base | sc_seq | Verdict |
|---|---|---|---|---|
| G1 | Sex, AUC | 0.860 | 0.797 | tie: −0.064, CI excludes 0, but under 2 × paired sd |
| G2 | Daily rhythm, circular R² | 0.030 | 0.087 | **better** |
| G3 | Seasonal pattern, circular R² | 0.332 | 0.431 | **better** |
| G4 | Seasonal repeatability across years | 0.592 | 0.567 | tie |
| G6 | Behaviour from acceleration alone | 0.904 | 0.913 | tie: +0.010, under the 0.03 minimum effect |
| G7a | Odd bouts | 0.169 | 0.149 | tie: −0.021, under the minimum effect |
| G7c | Odd days, AUC | 0.632 | 0.674 | **better** |
| G8 | Movement strategy (suspended: below the nuisance floor 0.86) | 0.706 | 0.727 | tie |
| G9 | Identity, balanced accuracy (lower is better) | 0.310 | 0.582 | **worse** |
| G10 | Guard rails (speed, climb, VeDBA, fore-aft tilt, wingbeat) | – | −0.056 to −0.099 | **all five fail** (tolerance 0.02) |

Overall verdict of the rule: "tie", which the rule uses for a mixed result. sc_seq is better on three rows, worse on one, and fails every guard rail.

## Descriptive rows

| Row | sc_base | sc_seq |
|---|---|---|
| G1b sex, continuous ; scheduled | 0.85 ; 0.84 | 0.87 ; 0.71 (seed 0) |
| G7d odd order | 0.874 | 0.748 |
| G7b identity control (lower is better) | 0.661 | 0.664 |
| G8b unsupervised bird groups, min Jaccard | 0.65 (seed 0) | 0.94 (seed 0, k = 2, stable) |
| G8c largest stable behaviour K | 3 | 2 (seed 0) |
| G9b recording mode from acceleration | 0.947 | 1.000 (the level-2 model is given the recording format as an input) |

## Reading

- Context over about a day adds day-scale and season-scale information: G2, G3 and G7c improve.
- The same context makes individual birds about twice as recognisable and loses part of the per-window kinematics. This shows in the guard rails and in sex from scheduled bursts.
- The G2 gain has not been checked against a shuffled-order control. The model might count windows from recording landmarks rather than read behaviour.
- The two stable bird groups in G8b have not been compared with deployment timing or cohort.
