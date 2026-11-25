# Stage 3 results (AI4WY job 61944, 2026-11-24)

> **Superseded (2026-11-24):** built on profile_v2, which had the wrong sensitivity setting for 73% of
> accelerometer bursts (DESIGN.md §6). Interpretations that depend on VeDBA levels (e.g. "slow daytime windows
> at 0.19 g are activity on the ground") do not hold.

## Windows
3,099,831 ten-second windows from 62 birds (6,530 dropped with < 1.2 s of acceleration); 1,477,142 contain GPS.
Night windows are clean: median VeDBA 0.005 g, static vector 0.995 g (one bird checked). Slow daytime windows
(GPS speed < 1 m/s) have median VeDBA ≈ 0.19 g: activity on the ground or perch, not quiet perching.

## Handcrafted baseline (GMM + HMM smoothing, 1,413,975 windows with GPS and ≥ 2 s acceleration)
- BIC kept decreasing up to the largest k tried (16): no natural number of states at this sample size.
- Bird-bootstrap stability is poor: adjusted Rand index 0.35–0.44 between refits.
- Interpretable groups nonetheless: flapping (dominant frequency 2.8–2.9 Hz, the golden eagle wingbeat), circling
  soaring (turn 17–25 °/s, climb +1–2 m/s), straight gliding (16–19 m/s, sinking 1.4–1.8 m/s), slow movement.
- Several pairs of states differ mainly in pitch (e.g. −8.6° vs +10.9° with otherwise equal kinematics) and draw on
  different subsets of birds: the tag's mounting angle on each bird leaks into the posture features. Each bird's
  frame must be rotated so its roosting gravity vector points up before features or the encoder see it.

## Probes on the first-draft step-30k embeddings (hour level, 144,203 hours, held-out birds)

| Quantity | z (projection) | h (encoder) | Wanted |
|---|---|---|---|
| Magnetometer fitted, within 2024 birds (BA, chance 0.5) | 0.736 | 0.742 | ≈ 0.5 |
| Any GPS in hour (BA, chance 0.5) | 0.907 | 0.960 | ≈ 0.5 |
| Decoding version (BA, chance 0.2) | 0.355 | 0.425 | ≈ 0.2 |
| Mean VeDBA (R²) | 0.338 | 0.537 | high |
| Mean speed (R²) | 0.496 | 0.774 | high |
| Fraction flapping (R²) | 0.170 | 0.297 | high |
| Fraction circling (R²) | 0.350 | 0.609 | high |
| Fraction gliding (R²) | 0.290 | 0.530 | high |
| Fraction night (R²) | 0.075 | 0.125 | high |
| Cohort (descriptive, BA) | 0.776 | 0.855 | – |

These are the numbers the new representation is compared against (aggregated to the same hours).
