# Scorecard baseline (2026-11-27)

Spec: `docs/SCORECARD_spec.md` v2.1. Split: 62 training birds, 12 search birds, 20 locked birds (not scored). Frozen injection list sha256 777a9b2e….
Outputs on AI4WY: `results_2026-11/v4/scorecard/<run>/` and the shared references in `scorecard/scorecard_reference/`.

## Designs

| Design | What it is | Seeds | Jobs |
|---|---|---|---|
| sc_base | r6 recipe (±60° yaw augmentation, 4 birds per batch), trained on the 62 training birds | 5 | train 62598–62606 (even), score 62618–62626 |
| sc_noneigh | per-bird yaw correction + 180° flip + masked reconstruction, no neighbour positives | 5 | train 62599–62607 (odd), score 62619–62627 |

## Scores (mean ± sd over 5 seeds; reference = handcrafted features, sensor frame; floor = recording format and deployment variables)

| Row | Goal | sc_base | sc_noneigh | Reference | Floor |
|---|---|---|---|---|---|
| G1 | Sex, AUC | **0.860 ± 0.016** | 0.809 ± 0.006 | 0.737 | 0.382 |
| G2 | Daily rhythm, circular R² | 0.030 ± 0.001 | 0.027 ± 0.001 | 0.006 | 0.003 |
| G3 | Seasonal pattern, circular R² | 0.332 ± 0.007 | 0.320 ± 0.010 | 0.099 | 0.249 |
| G4 | Seasonal repeatability across years, AUC | 0.592 ± 0.005 | 0.585 ± 0.008 | 0.640 | – |
| G5 | Lifetime, young birds (5 birds; descriptive) | 0.989 | 0.936 | 0.933 | – |
| G6 | Behaviour from acceleration alone, mean AUC | 0.904 ± 0.002 | 0.902 ± 0.001 | 0.759 | – |
| G7a | Odd bouts, AUC(novel) − AUC(same mix) | 0.169 ± 0.004 | 0.179 ± 0.001 | 0.075 | – |
| G7c | Odd days, AUC | 0.632 ± 0.004 | 0.632 ± 0.003 | 0.637 | – |
| G7d | Odd order, AUC (descriptive) | 0.874 | 0.868 | 0.839 | – |
| G7b | Identity control, AUC (lower is better) | 0.661 | 0.639 | 0.858 | – |
| G8 | Movement strategy, AUC | 0.706 ± 0.009 | 0.729 ± 0.011 | 0.660 | 0.859 |
| G8c | Largest stable behaviour-type K | 3.0 ± 0.7 | 3.2 ± 1.1 | – | – |
| G9 | Individual identity, balanced accuracy (lower is better) | 0.310 ± 0.005 | 0.304 ± 0.005 | 0.276 | – |
| G9b | Recording mode from acceleration (lower is better) | 0.947 | 0.946 | 0.851 | – |

The selection rule (`scripts/compare_scorecards.py`) gives **sc_noneigh worse than sc_base**, for two reasons:
- sex is lower by 0.051, past the paired seed sd, the 0.03 minimum effect and the bird bootstrap;
- the fore-aft tilt guard rail drops by 0.034, past its 0.02 tolerance.

Every other selection row is a tie. The yaw-correction recipe therefore stops here.

## What the baseline says about each goal

**Strong:**
- **Sex:** 0.86, against 0.74 for the handcrafted measures and 0.38 for nuisance alone.
- **Behaviour identification from acceleration alone:** 0.90 against 0.76.
- **Odd-bout content:** about twice the handcrafted score.

**Weak:**
- **Daily rhythm** (G2, R² 0.03). Short scheduled bursts carry very little about the time of day within daylight.
- **Odd days** (G7c, no better than handcrafted).
- **Seasonal repeatability across years** (G4, below handcrafted).
- **Movement-strategy groups** (G8). They are below the nuisance floor (0.86), so they are explained by deployment timing.
- **Seasonal pattern** (G3) is above its floor by only 0.08.
- **Behaviour types:** about 3 stable types at the 10-s scale.

These weak rows are day-scale and season-scale goals, which the level-2 sequence model is meant to address. Five seeds of it (`sc_seq_s0..s4`, parents `sc_base_s<n>`, no time encoding) are running.

## Environment covariates (env features_v1)

- Wind at 10 and 100 m, boundary-layer height, sensible heat flux, convective velocity w*, terrain elevation, slope, aspect, relief, height above ground and orographic uplift, for 87% of the 5.5 M windows.
- 12% of windows have no GPS fix within 30 min. Windows after 2026-11-17 have no ERA5, because the ARCO store lacks those hours.
- Height above ground is not geoid-corrected.
- They are not used by any design yet.
