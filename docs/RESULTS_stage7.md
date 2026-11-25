# Stage 7: time budgets, flight structure, movement strategy, sensor QC and anomalies (2026-11-25)

Data: windows_v4 (92 birds with windows). Models: r6_yaw60_bpb4 (current model) and r4_v4.

## Code and runs

**Final outputs** (under `results_2026-11/v4/` on AI4WY):

| Analysis | Module | Code version | Output |
|---|---|---|---|
| Time budgets | `budgets.py` | 99d3e8c | `budgets_r6_v3` |
| Flight structure and bouts | `flight.py` | 7fb3bc5 | `flight_r6_r4_v3` |
| Movement strategy | `residency.py` | a18b894 | `residency_r6_v3b` |
| Sensor QC and anomalies | `anomaly.py` | 7fb3bc5 | `anomaly_v3` |

**Superseded** (do not use):
- `budgets_r6`, `budgets_r6_v2`
- `flight_r6_r4`
- `residency_r6`, `residency_r6_v2`
- `residency_r6_v3`: partial; the job crashed on an empty grouping and was rerun as `v3b`.
- `anomaly_v1`, `anomaly_v2`

**Review:** `docs/review_fable_2026-11-25.md` (supervisor review). Every critical item in it has been addressed in the final runs.

**Design facts found along the way** (verified on the data):
- The tags record almost nothing from about 20:00 to 05:00 local time. The window is fixed to the clock, not to the sun.
- Continuous mode is speed-triggered, and the scheduled burst clock is suppressed during continuous bouts: only 2.5% of the expected clock slots inside bouts hold a burst.
- Birds by deployment year: 2024 = 38, 2025 = 22–24, 2026 = 32. Earlier "cohort" labels had merged 2026 into 2025.
- Deployment year 2024 holds 20 migrant-flagged birds, 17 young-tagged birds and 1 other bird.
- 26 of the 92 birds never record in continuous mode.
- Counts across all k=1 windows: 3,007,083 k=1 windows, of which 2,041,063 are full (200 samples).
  - "Still" (label perched or VeDBA < 0.02): 1,550 of the full k=1 windows, 66,136 of all k=1 windows.
  - The old "25 of 91,913" came from a subsample.

## 1. Time budgets (descriptive; budgets_r6_v3)

**Method.**
- Each scheduled burst stands for the time to the next burst, capped at 1.5 × the interval.
- Bout time is classified by the K=2 and K=6 states of the bout's own windows.
- Days are built from 24 local-solar-hour bins. Unrecorded bins (mostly night) are filled from the bird's own month or all-month hour mean, otherwise from the population month × hour mean.
- Filling changes flight by only about 0.03 h per day. Night ground hours, however, come mostly from the fill.

**Daily flight**, month-balanced, with 95% CI over birds:

| Bird set | Birds | Flight h/day |
|---|---|---|
| All birds | 88 | 2.12 [2.00, 2.24] |
| Age not stated | 72 | 2.15 |
| Migrant-flagged | 20 | 2.39 |
| Age not stated, not flagged | 52 | 2.08 |

- Time inside continuous bouts: 0.59 h/day.
- The diel profile sums to 1.85 h.
- The old window-count estimate (43% of windows are flight) overstated flight by about 3.4×. The 24-h scaled estimate (3.1 h/day) was inflated by daytime-only coverage.

**Diel pattern.** Flight is about 0 at night, rises from 08:00 solar, peaks at 11:00–13:00 (27–28% of the hour) and falls to about 0 after 19:00.

**Season.**
- All birds, flight h/day: Jan 1.40, Feb 1.72, Mar 2.68, Apr 3.53, May 3.75, Jun 2.20, Jul 1.16, Aug 1.60, Sep 2.44, Oct 2.55, Nov 1.83, Dec 1.56.
- The July minimum persists in age-not-stated birds: 1.00 h (0.68 h in not-flagged birds). Young birds entering the data in summer do not explain it.

**Flight composition.** K=6 flight states are reported only as a descriptive acceleration-state mix, not as behaviours. The state stage 6 called "flapping" is 18–28% of flight, against about 2% flapping in published golden eagle data.

**Tests** (bird as the unit):

| Comparison | Effect | Result |
|---|---|---|
| Female − male | −0.20 h (g −0.31) | p 0.066 within deployment year; Holm 0.13; not significant |
| Migrant-flagged (deployed Sep/Oct) − young-tagged (deployed Jun/Jul), deployment year 2024 | +0.39 h (+0.30 h outside migration months) | One confounded contrast: deploy day separates the groups perfectly (AUC 1.0). Not testable. |

## 2. Flight structure (flight_r6_r4_v3)

**Axes (descriptive).** PCA of h over flight windows. In continuous mode:

| Axis | Main correlates |
|---|---|
| PC1 | circling and climbing vs fast and straight: turn +0.78, climb +0.71, speed −0.55 |
| PC2 | effort: VeDBA +0.68 |
| PC3 | wingbeat: flap band +0.33, dominant frequency +0.31 |
| PC4 | speed: +0.48 |

- These axes agree between r4 and r6 (Spearman 0.99, 0.99, 0.95 and 0.88 for PC1–4).
- In scheduled mode (no GPS), PC1 is an acceleration-intensity axis (VeDBA −0.49).

**Sex along the axes** (tested; sex permuted within deployment year × age group; Holm over 30 tests):
- Only one test survives Holm: scheduled-mode PC1, the acceleration-intensity axis (g 0.75, p 0.001, Holm 0.03). It does not hold with full windows only.
- The continuous-mode axes do not survive Holm (PC1 p 0.075, PC2 p 0.047).

The stage-6 kinematic sex differences (females faster, wider turns; `RESULTS_stage6_demography.md` §4, §7) were tested on handcrafted per-state medians. They stand separately and are not replaced by this axis test.

**Bouts** (continuous mode, canonical `flight.bout_ids`):
- 61,375 bouts, 34,784 of at least 60 s, in 65 birds.
- No K ≥ 2 meets the stability rule (K=2 minimum Jaccard 0.62 in r6), so bouts do not form stable types.
- Net circling is low: a median 0.12 circles/min, and 4% of bouts reach at least 1 circle/min. The old gross-turn measure (1.08) was inflated by heading noise.

## 3. Movement strategy from tracks (residency_r6_v3b; descriptive)

**Rule classes** (94 birds):

| Class | Birds |
|---|---|
| Migrant (≥ 1000 km north and back) | 21 |
| Migrant, departed (far north at track end) | 7 |
| Resident, full year (never beyond 200 km over ≥ 330 days) | 2 (both young-tagged) |
| Present, partial year | 10 |
| Juvenile dispersal | 11 |
| Nomad | 5 |
| Away at end | 5 |
| Insufficient track | 33 |

- 15 of the 20 'migrant'-flagged birds are migrants. 13 of 55 unflagged adults are migrants too, so the flag is incomplete.
- 11 of the 24 birds deployed in 2025 are migrants.
- Rule vs NSD models agree on 61% of 36 birds (kappa 0.22). Class counts change little between the 750, 1000 and 1250 km thresholds.

**Timing:**
- Departure, first day beyond 500 km north: median day 119 (29 April), range 97–132.
- Return: median day 288 (15 October). The range is 18–349, unexplained, and not caused by GPS gaps.

**Winter location.** Migrants winter a median of about 400 km from the study-area centre and about 150 km from their own deployment site. Only about 1% of their November–March days fall within 200 km of the centre, so they do not winter in the basin itself.

**Migrants vs residents.** Not testable. No grouping has at least 4 adult full-year residents. Where partial-year birds stand in, the nuisance floor (track length, deploy month, cohort) separates the groups perfectly (AUC 0.92–1.0).

## 4. Sensor and recording QC (anomaly_v3 part 1; descriptive)

These are deterministic checks on the raw data, with no model.

**Validation with injected faults** (recall among windows each check applies to):

| Fault | Recall | False-positive rate |
|---|---|---|
| Stuck axis | 1.0 | 0 |
| 90° tilt, still night windows | 0.98 | 0.019 |
| 90° yaw, flapping windows | 0.997 | 0.006 |
| Amplitude halved | 1.0 | 0 |
| GPS speed zeroed | 1.0 | 0.011–0.029 |
| GPS spike | 1.0 | 0 |
| Saturation | 0.22 | — |

Several of these injections set exactly the quantity the check thresholds, so they show that the code works, not real-world sensitivity.

**Persistent slips** (detected within ±4 nights; clean rate 1.3–1.5%):

| Slip | 20° | 25° | 30° | 35° | 40° | 45° | 90° |
|---|---|---|---|---|---|---|---|
| Tilt | 0.07 | 0.20 | 0.65 | 0.95 | 0.99 | 1.0 | 1.0 |
| Yaw | 0.97 | 0.99 | 0.99 | 0.99 | 1.0 | 1.0 | 0.99 |

- The 30° roost threshold was chosen after seeing the clean rate, so these recalls are not out-of-sample.
- A 180° yaw cannot be detected by any check.

**Real data:**
- Flag rates: tilt_night 1.5%, gps_flap_zero 0.95%, yaw_flap 0.55%, everything else < 0.02%.
- 15 roost-direction change events and 28 flapping-axis change events across all birds.
- No widespread sensor faults were found.

## 5. Behavioural anomalies (anomaly_v3 part 2; ranked for inspection, not detections)

**What gets scored:**
- Full windows only, and like with like: same bird, recording mode, K=2 state and ±45 days.
- 40% of windows (the short ones) are excluded by design. 52.5% of bird-days have no score at any level.

**Validation** (r6; injected behaviour):

| Injection | Recall | AUC |
|---|---|---|
| Bout from another bird with a mix this bird never shows | 0.38 | 0.92 |
| Bout from another bird with the same mix (identity control) | 0.04 | 0.70 |
| Shuffled bout order, bouts ≥ 85 windows | 0.88 | 0.99 |
| Reversed bout order | ≈ 0.02 (not detectable) | 0.56 |
| Day replaced by the same bird's day from another season | 0.21 | 0.77 |
| Day replaced by another bird's day | 0.04 | 0.63 |

False-positive rate about 1%.

**Format neutrality:**
- Bouts: met.
- Windows: not met. Continuous-mode segment edges, which are take-off and landing windows, rank 3.8× more often. This is likely real behaviour and is reported separately.

No individual flagged event has been read yet; the blind plot pages exist for that.

## What this means for the goals

| Goal | Status |
|---|---|
| Common behaviours | Ground vs flight is the stable division. Ground has 4 stable sub-states. Flight is a continuum along circling-climb / effort / wingbeat / speed axes, with no stable types at window or bout scale. |
| Time budgets, daily and seasonal | About 2.1 h/day of flight, peaking at midday and in April–May, lowest in July. |
| Sex | Females fly faster with wider turns (stage 6, robust). Daily flight time does not differ significantly. |
| Young vs old | Not separable from deployment timing at the bird level. Within-bird year-1 vs year-2 data remain too few (6 birds). |
| Movement strategy | Migrants are clearly identified from tracks, with timing and winter ranges. Migrants vs residents is untestable for lack of full-year adult residents. |
| Unusual birds and events | Sensor QC works. Behavioural anomaly ranking works for content and order changes at bout level, weakly at day level. Nothing has been inspected yet. |
